"""
StateVerifyOutput — ASR-roundtrip QC per synthesized clip (README.md §8).

Downloads each output WAV, transcribes with faster-whisper base
(multilingual), fuzzy-matches against the requested text. QC failure is
recorded per clip, never raised — the retry-with-new-seed policy is an
explicit open decision (§10) awaiting real fail-rate data; until then a
failed clip is flagged in the result, not retried. That "never raised"
covers BOTH a content mismatch (qc.verify_content's own contract) and a
QC pass that could not run at all (missing/corrupt WAV, whisper crash):
the latter records {"passed": None, "similarity": -1.0, "error": ...}
on that candidate rather than escaping the handler, because the ASL
"Verify output" state has no States.TaskFailed retry and one bad clip
would otherwise discard the whole batch's already-synthesized audio.

Checkpoint/resume (README.md §10, same pattern as
StateSynthesizeLLM/StateSynthesizeFlowHift): checked before each clip's
QC pass; on near-timeout, `item["outputs"]` (already mutated in place
with "qc" on completed entries) is persisted and
LambdaResourceTimeoutException raised for the ASL Retry rule. The
timeout check is per-transcript, not per-candidate — a best-of-N
transcript's whole candidate QC loop below runs as one uninterruptible
unit once started (same "per-transcript, not mid-generation" residual
risk already accepted for the other two states, now covering up to N
QC passes instead of one).

Expected event: StateSynthesizeFlowHift output. Adds "qc" to each
outputs[] entry: {"passed", "similarity", "threshold",
"detected_language"}, plus "transcribed" on failed/errored clips only,
and "error" when the QC pass itself could not run. The full
qc.verify_content result is trimmed on the way out and
`item["transcripts"]` is dropped — see _prune_qc_evidence for why the
Map's aggregate payload makes that a correctness matter, not tidiness.

Best-of-N selection (README.md §7a Tier 2): every outputs[] entry
carries a "candidates" list (StateSynthesizeFlowHift always produces
one, length 1 outside Tier 2 styles — see that module). Every
candidate is QC'd, the highest-similarity one is kept, and the entry
COLLAPSES to a flat {"wav_key", "duration_s", "stretched", "qc"} shape
— so get_job_status and every other downstream consumer needs zero
changes regardless of tier, and length-1 "best-of-1" costs nothing
extra (one QC pass, same as before this feature existed). Losing
candidates' WAVs are deleted from S3 (the other N-1 renders are not a
real output; leaving them in S3 forever is unbounded storage on every
styled request for no purpose).
"""

from __future__ import annotations

import logging
import os

import boto3
from botocore.exceptions import ClientError

from lib import checkpoint, qc
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.model_store import ensure_in_tmp

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
WHISPER_MODEL_S3_KEY = "_model_artefacts/faster-whisper-base/model.tar.gz"
WHISPER_TMP_DIR = "/tmp/models/faster_whisper_base"
STATE_NAME = "verify_output"

_s3 = boto3.client("s3")


def _prune_qc_evidence(entry: dict) -> None:
    """Strip the text this state does not need to hand forward.

    The Map's aggregate ResultPath carries EVERY iterator's terminal output in
    ONE 256 KiB payload, and it is the one payload bound that splitting a batch
    into more iterators cannot relieve — it is the sum over every transcript in
    the request, however they are grouped. Measured, each transcript's text used
    to appear THREE times in it: `transcripts[].text`, `qc.expected` and
    `qc.transcribed`. At MAX_TEXT_CHARS that made the aggregate 15.9x larger
    than the information in it requires, capping a request at 63 transcripts.

    What goes and why:
      - `qc.expected` is a byte-for-byte copy of the caller's own input, which
        the caller joins back by `transcript_index`. It is the clearest
        duplication of the three and never reaches the gateway.
      - `qc.transcribed` is the ONLY field here that is not derivable — it is
        what whisper actually heard, i.e. the evidence for a QC failure, which
        README.md calls "a review pointer, not an error". So it is kept **on
        failed and errored clips and dropped on passing ones**, where by
        definition of the pass threshold it is a near-copy of the expected text
        and carries nothing. Failures are the bounded case (3 of 73 on the
        reference run); a run where every clip fails degrades to one copy of
        the text, which is what `estimate_map_aggregate_bytes` now budgets for.

    `threshold` and `detected_language` stay — both are short and fixed-size.
    """
    q = entry.get("qc")
    if not q:
        return
    q.pop("expected", None)
    if q.get("passed") is True:
        q.pop("transcribed", None)


def lambda_handler(event: dict, context) -> dict:
    item = event["item_data"]
    transcripts = item["transcripts"]
    outputs = item["outputs"]
    out_bucket = item["out_bucket"]
    language = (item.get("voice") or {}).get("language")
    ckpt_key = checkpoint.checkpoint_key(
        item["out_prefix"], STATE_NAME, item["job_id"], item["batch_index"])

    ensure_in_tmp(MODEL_BUCKET, WHISPER_MODEL_S3_KEY, WHISPER_TMP_DIR,
                  "faster-whisper-base")

    ckpt = checkpoint.load(out_bucket, ckpt_key)
    if ckpt is not None:
        # outputs[] entries are keyed by position, not appended — the
        # checkpoint is the same list with "qc" already filled in on a
        # prefix of entries; restore it wholesale rather than merge.
        outputs = ckpt["outputs"]
        item["outputs"] = outputs
        log.info("resuming QC (checkpoint restored)")

    for i, (t, out) in enumerate(zip(transcripts, outputs)):
        if "error" in out or "qc" in out:
            continue  # already errored upstream, or QC already ran (resume)

        if context.get_remaining_time_in_millis() < checkpoint.REMAINING_TIME_BUFFER_MS:
            checkpoint.save(out_bucket, ckpt_key, {"outputs": outputs})
            log.info("checkpointed QC progress — requesting retry")
            raise LambdaResourceTimeoutException(
                "verify_output checkpointed mid-batch")

        # best-of-N (length 1 outside Tier 2 styles): QC every candidate,
        # keep the highest-similarity one, delete the rest.
        #
        # Per-candidate isolation, same reasoning as the LLM and Flow/HiFT
        # states: a corrupt WAV, a missing S3 key, or a whisper failure must
        # not take down the whole Map iteration. The "Verify output" ASL state
        # has no States.TaskFailed retry, so an escaping exception ends the
        # workflow — and the audio it was QC'ing is already durably in S3.
        # A candidate whose QC could not run keeps its entry with a
        # qc.error marker and a similarity of -1.0 so it always loses the
        # max() below to any candidate that did produce a score.
        candidates = out.pop("candidates")
        for ci, cand in enumerate(candidates):
            local = f"/tmp/qc_{i}_c{ci}.wav"
            try:
                _s3.download_file(out_bucket, cand["wav_key"], local)
                cand["qc"] = qc.verify_content(local, t["text"], WHISPER_TMP_DIR,
                                               language=language)
            except Exception as exc:
                log.exception("QC failed for transcript %d/%d candidate %d (%s)",
                              i + 1, len(transcripts), ci, cand["wav_key"])
                cand["qc"] = {"passed": None, "similarity": -1.0,
                              "error": f"{type(exc).__name__}: {exc}"}
            finally:
                # download_file may have failed before creating the file
                if os.path.exists(local):
                    os.remove(local)

        best = max(candidates, key=lambda c: c["qc"]["similarity"])
        for cand in candidates:
            if cand is not best:
                try:
                    _s3.delete_object(Bucket=out_bucket, Key=cand["wav_key"])
                except ClientError:
                    log.exception("failed to delete losing candidate %s",
                                 cand["wav_key"])
        out.update(best)
        log.info("transcript %d/%d: best-of-%d picked seed=%s (similarity=%.3f)",
                 i + 1, len(transcripts), len(candidates), best.get("seed"),
                 best["qc"]["similarity"])
        # After the log line, so the diagnostic above still sees the full entry,
        # and before any checkpoint save so a resumed run persists the pruned
        # shape too (the checkpoint is written to the same 4 KiB-per-entry
        # budget the aggregate is).
        _prune_qc_evidence(out)

    # The second of the three text copies. Nothing downstream of this state
    # reads item["transcripts"] — the LLM and Flow/HiFT states consume it before
    # here, the gateway joins by transcript_index, and usage_tracking never
    # touches item_data. Dropped last so the read and the removal stay visibly
    # ordered — `transcripts` above is a local binding to the same list, so the
    # QC loop would survive an earlier pop today, but only by accident of that
    # binding, and a mid-batch resume re-enters the loop from the top.
    item.pop("transcripts", None)

    checkpoint.clear(out_bucket, ckpt_key)
    n_pass = sum(1 for o in outputs if o.get("qc", {}).get("passed"))
    n_err = sum(1 for o in outputs if "error" in o)
    log.info("QC: %d passed, %d failed, %d errored (of %d)",
             n_pass, len(outputs) - n_pass - n_err, n_err, len(outputs))
    return event
