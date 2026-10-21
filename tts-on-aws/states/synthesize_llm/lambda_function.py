"""
StateSynthesizeLLM — text → speech tokens, one batch per invocation.

Python loop over the batch's transcripts INSIDE one invocation
(README.md §6): the LLM (~1.4 GB FP16) loads once per container and
is reused across every transcript — guaranteed, unlike Map-level
warm-container reuse which degrades as MaxConcurrency rises.

Per-transcript try/except: a failure on transcript 5 of 10 records a
per-item error and continues — Step Functions Retry would otherwise
re-run the whole batch including already-succeeded items.

Checkpoint/resume (README.md §10, same pattern as
face-recognition-on-aws's StateClusterIdentities): a batch can hold up
to 100 transcripts; at ~20s/transcript that is up to ~2000s of work
inside a 900s Lambda. Before each transcript, check remaining Lambda
time; if under REMAINING_TIME_BUFFER_MS, checkpoint results-so-far to
S3 and raise LambdaResourceTimeoutException. The ASL Retry rule on that
error re-invokes this SAME state; on re-entry the checkpoint is loaded
and already-done transcripts (success OR per-item error — neither is
retried today) are skipped. Residual risk, not solved here: a single
pathological transcript near MAX_TEXT_CHARS could itself exceed the
per-invocation budget (LLM's max_token_text_ratio=20x can decode up to
~5000 tokens for one transcript) — checkpointing is per-transcript, not
mid-generation.

Payload budget: this state's output carries every transcript's token
array for EVERY best-of-N candidate, and that is the single largest
payload the workflow moves. README.md §10 derived ~49,000 tokens of
256-KiB headroom; measured against the real JSON envelope, that budget
is reached at ~100 transcripts only for ~230-char texts at best-of-1,
and falls to ~24 transcripts at MAX_TEXT_CHARS=1000, or ~36 (230-char)
/ ~8 (1000-char) once emotion-tier best-of-3 triples the token arrays.
A fixed transcripts_per_batch count therefore cannot express the bound.
StateCheckAndPreprocess estimates the bytes up front and rejects an
oversize batch there (see SFN_PAYLOAD_BUDGET_BYTES /
estimate_batch_payload_bytes), so this state does not have to defend
against it — but any change to what this output carries has to be
re-checked against that estimator, not against the transcript count.

Expected event:
{
  "operation": "StateSynthesizeLLM",
  "item_data": {
    "voice_id": "...",
    "transcripts": [{"text": "...", "target_duration_s": null}, ...],
    "config": {"sampling_k": 25, "seed": null, "style": null},
    "job_id": "...", "batch_index": 0, "out_prefix": "...",
    ...  (passed through untouched)
  }
}
Adds to item_data: "llm_results": [{"candidates": [{"speech_tokens": [...],
                                   "seed": N}, ...]} | {"error": "..."}],
                                   "voice": {...}

Best-of-N candidates (README.md §7a Tier 2): "candidates" is always
a list, length 1 for no-style/pace requests, length
len(style_hints.CANDIDATE_SEEDS) for emotion-tier styles — one LLM
sampling attempt per seed, each producing its own speech-token array.
Kept as a uniform list shape (not a special-cased single value) so
StateSynthesizeFlowHift and StateVerifyOutput have exactly one code
path regardless of tier; StateVerifyOutput is where the best candidate
is actually chosen and the result collapses back to a single wav per
transcript. A per-candidate failure is skipped, not fatal — only a
transcript with zero surviving candidates is a transcript-level error.
"""

from __future__ import annotations

import logging
import os
import time

from lib import checkpoint, llm, style_hints, voice_store
from lib.conditioning_cache import get_conditioning
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.frontend import get_tokenizer
from lib.model_store import ensure_in_tmp

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
LLM_MODEL_S3_KEY = "_model_artefacts/cosyvoice3/llm/model.tar.gz"
LLM_TMP_DIR = "/tmp/models/cosyvoice3_llm"
STATE_NAME = "synthesize_llm"


def lambda_handler(event: dict, context) -> dict:
    item = event["item_data"]
    voice_id = item["voice_id"]
    transcripts = item["transcripts"]
    config = item.get("config", {})
    ckpt_bucket = item["out_bucket"]
    ckpt_key = checkpoint.checkpoint_key(
        item["out_prefix"], STATE_NAME, item["job_id"], item["batch_index"])

    voice = voice_store.get_voice(voice_id)
    if voice is None:
        raise ValueError(f"unknown voice_id: {voice_id}")

    ensure_in_tmp(MODEL_BUCKET, LLM_MODEL_S3_KEY, LLM_TMP_DIR, "cosyvoice3-llm")
    tokenizer = get_tokenizer(LLM_TMP_DIR)

    # registry keys are relative to the proxy bucket (§5)
    cond = get_conditioning(os.environ["PROXY_BUCKET"], voice["s3_conditioning_key"])
    ref_transcript = voice.get("ref_transcript", "")

    # style hint (README.md §7a): instruct2 mode replaces the LLM
    # stage's reference-audio conditioning with an instruction template —
    # ref_transcript="" and prompt_speech_tokens=[] are load-bearing, not
    # incidental (verified against upstream frontend_instruct2/
    # frontend_zero_shot: llm_prompt_speech_token is explicitly deleted in
    # that mode). Voice identity still survives via the Flow stage's own
    # conditioning (untouched, see synthesize_flow_hift), which is why
    # this is safe to do at the LLM stage alone.
    style = config.get("style")
    instruct_prefix = style_hints.resolve_instruct_prefix(style)
    explicit_seed = config.get("seed")
    if style is not None:
        llm_ref_transcript = ""
        style_prompt_tokens = []
        # An explicit caller seed always means single-shot (§7a: best-of-N
        # exists to pick among *unspecified* takes; a caller who asked for
        # a specific seed asked for a specific take, not a vote). Otherwise
        # Tier 1 (pace, candidates_for==1) uses the single verified-clean
        # seed=42; Tier 2 (emotion, candidates_for==3) uses the 3 seeds
        # this session verified 36/36 clean, in that fixed order.
        n = style_hints.candidates_for(style)
        if explicit_seed is not None:
            seeds = [explicit_seed]
        elif n == 1:
            seeds = [style_hints.DEFAULT_SEED]
        else:
            seeds = list(style_hints.CANDIDATE_SEEDS[:n])
    else:
        llm_ref_transcript = ref_transcript
        style_prompt_tokens = cond["speech_tokens"]
        seeds = [explicit_seed]

    ckpt = checkpoint.load(ckpt_bucket, ckpt_key)
    if ckpt is not None:
        results = ckpt["results"]
        log.info("resuming: %d/%d transcripts already done",
                 len(results), len(transcripts))
    else:
        results = []

    for i in range(len(results), len(transcripts)):
        if context.get_remaining_time_in_millis() < checkpoint.REMAINING_TIME_BUFFER_MS:
            checkpoint.save(ckpt_bucket, ckpt_key, {"results": results})
            log.info("checkpointed %d/%d transcripts — requesting retry",
                     len(results), len(transcripts))
            raise LambdaResourceTimeoutException(
                f"synthesize_llm checkpointed at {len(results)}/{len(transcripts)}")

        t = transcripts[i]
        t0 = time.time()
        candidates = []
        last_error = None
        for seed in seeds:
            try:
                tokens = llm.generate_speech_tokens(
                    tokenizer, LLM_TMP_DIR,
                    ref_transcript=llm_ref_transcript,
                    tts_text=t["text"],
                    prompt_speech_tokens=style_prompt_tokens,
                    instruct_prefix=instruct_prefix,
                    sampling_k=config.get("sampling_k", 25),
                    seed=seed,
                )
                candidates.append({"speech_tokens": tokens, "seed": seed})
            except Exception as exc:  # per-candidate isolation, same
                                      # reasoning as per-transcript below —
                                      # one bad seed shouldn't sink a
                                      # transcript that has other seeds left
                log.exception("transcript %d/%d seed=%s failed",
                              i + 1, len(transcripts), seed)
                last_error = str(exc)

        if candidates:
            results.append({"candidates": candidates,
                            "llm_seconds": round(time.time() - t0, 2)})
            log.info("transcript %d/%d: %d/%d candidates, %d tokens (first) in %.1fs",
                     i + 1, len(transcripts), len(candidates), len(seeds),
                     len(candidates[0]["speech_tokens"]), time.time() - t0)
        else:
            log.error("transcript %d/%d failed (all %d seeds): %s",
                     i + 1, len(transcripts), len(seeds), last_error)
            results.append({"error": last_error})

    # `results` is non-empty (StateCheckAndPreprocess rejects an empty
    # transcripts list), but all() is vacuously True on an empty list and the
    # results[0] below would then IndexError over the real failure.
    if results and all("error" in r for r in results):
        # nothing succeeded across the WHOLE batch (checkpoint-merged, not
        # just this invocation's slice) — this IS a state-level failure
        checkpoint.clear(ckpt_bucket, ckpt_key)
        raise RuntimeError(
            f"all {len(results)} transcripts failed in StateSynthesizeLLM; "
            f"first error: {results[0]['error']}")

    checkpoint.clear(ckpt_bucket, ckpt_key)
    item["llm_results"] = results
    item["voice"] = {"voice_id": voice_id,
                     "language": voice.get("language"),
                     "ref_transcript": ref_transcript,
                     "s3_conditioning_key": voice["s3_conditioning_key"]}
    return event
