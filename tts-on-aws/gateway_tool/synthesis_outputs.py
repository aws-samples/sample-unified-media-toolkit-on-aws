"""
The SUCCEEDED-output flattener and the outputs.json persistence — shared
by the gateway dispatcher (get_job_status, the absorbed-terminal submit
path) and callback_waker, so the file a woken consumer reads and a live
poll can never disagree.

A separate module for a deliberate reason (review 2026-08-31): the waker
Lambda runs with a minimal environment (CALLBACK_TABLE, PROXY_BUCKET),
while lambda_function.py reads SYNTHESIS_STATE_MACHINE_ARN /
REGISTER_VOICE_STATE_MACHINE_ARN / VOICE_TABLE_NAME unconditionally at
import — so the waker importing the gateway to reach its parser raised
KeyError on every invocation and outputs.json was never written. This
module therefore reads NO environment and constructs NO clients
(callback_registry's rule): everything arrives as a parameter.
"""
from __future__ import annotations

import json
import re

S3URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def parse_succeeded_output(resp: dict, proxy_bucket: str) -> dict:
    """Parse a SUCCEEDED execution into the caller-facing result.

    `resp` needs only `output` and `executionArn` — satisfied by a
    DescribeExecution response AND by the EventBridge terminal event's
    `detail` (which carries both), so the waker persists without a
    redundant Describe call unless the event truncated `output`.

    Two shapes, chosen by which state machine ran: a `voice` record
    (register_voice) or a per-transcript `outputs[]` + `batch_qc_summary`
    (synthesis).

    `outputs[]` here is deliberately NOT the siblings' `{uri, mime, label}`
    triple, and this is a shape difference rather than a gap. Their `label` is a
    ROLE vocabulary — one entry per distinct KIND of file (transcript, caption,
    manifest, usage) — which is what makes "select by label, never parse
    filenames" meaningful. This target's outputs are N instances of ONE role
    (clip 0..N-1) carrying per-item metadata (`duration_s`, `stretched`,
    `qc_passed`, `qc_similarity`, per-entry `error`) that a triple has nowhere
    to put, plus `transcript_index` — the join key back to the caller's input
    array, which no sibling has because no sibling takes an input array. A
    conforming shape would either repeat one non-discriminating label N times or
    drop that metadata. speech-diarization reached the same conclusion for its
    per-speaker WAVs (non-contiguous speaker ids): it emits a `speakers/` prefix
    entry plus a manifest instead of N labeled entries. This outputs[] IS that
    manifest, inlined.

    What the siblings had and this did not — added 2026-08-07, all additive —
    was `workflow_id` on the poll, `output_s3_prefix`, a reachable
    `usage_tracking.json` (as `usage_uri`, since there is no label to hang it
    on), and a `preview_uri` on the register_voice record instead of the raw
    `s3_preview_key` that `get_voice` already resolved.
    """
    out = json.loads(resp.get("output") or "{}")
    # No `uuid` here. The caller's token is never echoed, and the event `uuid`
    # in the output is just the execution name — which is `workflow_id`, the
    # key the caller already polled with.
    # No `execution_arn` either.  It echoed resp["executionArn"] — the very ARN
    # the caller passed in as `job_id` — so it gave one value two names.
    # `job_id` is attached by the caller of this function, on every branch.
    result: dict = {"status": "SUCCEEDED"}

    # The execution NAME is the workflow_id (see _start_execution), so a poll
    # can recover it without the caller having kept the submit's response —
    # recovered the same way the sibling gateways do it. Every sibling poll
    # returns workflow_id and this one did not, which meant an agent that
    # resumed a conversation holding only a job_id had no way to name the run.
    _wid = (resp.get("executionArn") or "").rsplit(":", 1)[-1]
    if _wid:
        result["workflow_id"] = _wid

    data = out.get("data") or {}

    # register-voice output
    if "voice" in data:
        voice = dict(data["voice"])
        # Resolve the preview key to a presignable s3:// URI, exactly as
        # _get_voice does. Without this, register_voice and get_voice returned
        # the SAME record in two different shapes (raw s3_preview_key vs
        # preview_uri) from one target, so an agent that had just registered a
        # voice could not presign its preview without hand-building the URI.
        if voice.get("s3_preview_key"):
            voice["preview_uri"] = (
                f"s3://{proxy_bucket}/{voice.pop('s3_preview_key')}")
        result["voice"] = voice
        return result

    # synthesis output: flatten per-batch outputs into outputs[]
    #
    # transcript_index is a GLOBAL index across the flattened list, not
    # per-batch: with more than one batch a per-batch index would repeat
    # (two entries both claiming index 0) and batch_qc_summary's
    # failed_indices below would be ambiguous. The gateway's own
    # synthesize_speech only ever submits one batch, but the state machine
    # accepts many and this parser is the only consumer of that shape.
    outputs = []
    next_index = 0
    out_location = None
    for it in data.get("iterators", []):
        item = it.get("item_data") or it   # Map may or may not unwrap
        out_bucket = item.get("out_bucket")
        # Every batch of one execution shares one out_bucket/out_prefix
        # (StateCheckAndPreprocess derives both once), so first-non-empty wins
        # rather than last — a later batch that failed before recording them
        # must not blank out a location already established.
        if out_location is None and out_bucket and item.get("out_prefix"):
            out_location = f"s3://{out_bucket}/{item['out_prefix'].strip('/')}"
        for o in item.get("outputs", []):
            entry = {"transcript_index": next_index}
            next_index += 1
            if "error" in o or not out_bucket or not o.get("wav_key"):
                entry["error"] = o.get("error") or "no output produced"
            else:
                entry["uri"] = f"s3://{out_bucket}/{o['wav_key']}"
                entry["duration_s"] = o.get("duration_s")
                entry["stretched"] = o.get("stretched")
                qc = o.get("qc") or {}
                entry["qc_passed"] = qc.get("passed")
                entry["qc_similarity"] = qc.get("similarity")
                # QC that could not run at all (missing/corrupt WAV, whisper
                # crash) reports passed=None + an error string — surface it
                # rather than letting it read as "no QC result".
                if qc.get("error"):
                    entry["qc_error"] = qc["error"]
            outputs.append(entry)
    result["outputs"] = outputs

    # Top-level QC rollup — SUCCEEDED execution status says nothing about
    # content quality (StateVerifyOutput never fails the workflow; see
    # README.md §8/§10). Without this, a caller that only checks
    # `status` can silently ship a clip whose ASR roundtrip failed. Not a
    # hard gate (whisper-base's own transcription is measurably
    # non-deterministic across invocations on identical audio — a fixed
    # threshold WILL flip on borderline clips run to run), so this stays
    # informational: surfaced prominently, not enforced.
    qc_entries = [o for o in outputs if "qc_passed" in o]
    if qc_entries:
        n_failed = sum(1 for o in qc_entries if o["qc_passed"] is False)
        result["batch_qc_summary"] = {
            "all_passed": n_failed == 0,
            "passed": len(qc_entries) - n_failed,
            "failed": n_failed,
            "total": len(qc_entries),
            "failed_indices": [o["transcript_index"] for o in qc_entries
                              if o["qc_passed"] is False],
        }

    if out_location:
        result["output_s3_prefix"] = out_location

    usage = data.get("usage_tracking") or {}
    if usage:
        result["pipeline_wall_sec"] = usage.get("pipeline_wall_sec")
        result["total_estimated_cost"] = usage.get("total_estimated_cost")
        # usage_tracking.json holds the PER-STATE cost breakdown, and until
        # 2026-08-07 nothing here surfaced its location: the file was written,
        # the whole usage_doc was in hand (the two fields above are read out of
        # it), and no agent could find it. Every sibling advertises it as
        # outputs[label=usage]; this target has no label vocabulary (its
        # outputs[] is one entry per transcript, not one per file kind), so it
        # is exposed as a URI field instead. Same fact, shape that fits here.
        usage_bucket = usage.get("output_bucket")
        usage_key = usage.get("usage_key")
        if usage_bucket and usage_key:
            result["usage_uri"] = f"s3://{usage_bucket}/{usage_key}"
    return result


def _outputs_key(output_s3_prefix: str) -> tuple:
    m = S3URI_RE.match(output_s3_prefix or "")
    if not m:
        raise ValueError(
            f"not an s3:// prefix: {output_s3_prefix!r}")
    bucket, key = m.group(1), m.group(2)
    return bucket, f"{key.rstrip('/')}/outputs.json"


def persist_outputs(s3, resp: dict, proxy_bucket: str) -> bool:
    """Write the poll's SUCCEEDED result (outputs[], batch_qc_summary,
    output_s3_prefix, workflow_id) to {output_s3_prefix}/outputs.json.
    False when the execution has no synthesis outputs (a voice
    registration)."""
    parsed = parse_succeeded_output(resp, proxy_bucket)
    prefix = parsed.get("output_s3_prefix")
    if not prefix or "outputs" not in parsed:
        return False
    bucket, key = _outputs_key(prefix)
    s3.put_object(Bucket=bucket, Key=key,
                  Body=json.dumps(parsed, ensure_ascii=False,
                                  indent=1).encode("utf-8"),
                  ContentType="application/json")
    return True


def persist_failure(s3, output_s3_prefix: str, status: str,
                    error: str | None, execution_arn: str) -> None:
    """Write a FAILED/TIMED_OUT/ABORTED verdict to the SAME outputs.json
    pointer. Without this the failure existed only in the callback
    payload — which a waitForTaskToken consumer discards by design (the
    payload is a signal, never a data channel) — so the woken consumer
    died on a raw NoSuchKey that named neither tts nor the cause, while
    the real failure sat in this machine's execution history (review
    2026-08-31). One pointer, one file: the consumer's reader tells the
    two shapes apart by `status`."""
    bucket, key = _outputs_key(output_s3_prefix)
    doc = {"status": status,
           "error": error or f"synthesis execution reached {status}",
           "execution_arn": execution_arn}
    s3.put_object(Bucket=bucket, Key=key,
                  Body=json.dumps(doc, ensure_ascii=False,
                                  indent=1).encode("utf-8"),
                  ContentType="application/json")
