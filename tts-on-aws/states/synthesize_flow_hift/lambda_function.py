"""
StateSynthesizeFlowHift — speech tokens → waveform, one batch per
invocation. Same in-Lambda batch loop and per-transcript isolation as
StateSynthesizeLLM (README.md §6).

Per transcript: Flow (mel) → HiFT (waveform) → optional bounded
time-stretch (§7 Layer 2, ±10% cap) → WAV to S3 under out_prefix.

Small verified-parallelizable pair inside Flow (speaker projection vs
token embedding, §10) is NOT threaded here — measured milliseconds
against a multi-second Euler loop; complexity not paid for.

Checkpoint/resume (README.md §10, same pattern and same residual
long-transcript caveat as StateSynthesizeLLM): checked before each
transcript; on near-timeout, `outputs` collected so far is persisted to
S3 and LambdaResourceTimeoutException raised for the ASL Retry rule to
re-invoke this state. Already-written WAVs are NOT re-synthesized on
resume — only their `outputs` entry needs to survive, since the actual
audio is already durably in S3 under out_prefix (not the ephemeral
checkpoint prefix).

Expected event: the StateSynthesizeLLM output (item_data now has
llm_results + voice). Adds "outputs": one entry per transcript —
{"candidates": [{"wav_key", "duration_s", "stretched": bool, "seed"},
...]} or {"error"}.

Best-of-N candidates (README.md §7a Tier 2): llm_results[i] always
carries a "candidates" list of speech-token arrays (length 1 outside
Tier 2 styles, see StateSynthesizeLLM) — this state synthesizes EVERY
candidate to its own WAV under out_prefix and passes them all through
unranked; StateVerifyOutput is where one candidate is actually chosen
(by QC similarity) and the losers' WAVs are deleted. Keeping this state
candidate-agnostic (always a list, always synthesize all of them) means
Flow/HiFT doesn't need to know which tier produced its input.
"""

from __future__ import annotations

import io
import logging
import os
import time

import boto3
import numpy as np
import soundfile as sf

from lib import checkpoint, flow, hift
from lib.conditioning_cache import get_conditioning
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.duration import DurationOutOfRangeError, fit_to_duration
from lib.model_store import ensure_in_tmp

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
FLOWHIFT_MODEL_S3_KEY = "_model_artefacts/cosyvoice3/flow_hift/model.tar.gz"
FLOWHIFT_TMP_DIR = "/tmp/models/cosyvoice3_flow_hift"
STATE_NAME = "synthesize_flow_hift"

_s3 = boto3.client("s3")


def lambda_handler(event: dict, context) -> dict:
    item = event["item_data"]
    voice = item["voice"]
    transcripts = item["transcripts"]
    llm_results = item["llm_results"]
    config = item.get("config", {})
    out_bucket = item["out_bucket"]
    out_prefix = item["out_prefix"].rstrip("/")
    ckpt_key = checkpoint.checkpoint_key(
        item["out_prefix"], STATE_NAME, item["job_id"], item["batch_index"])

    ensure_in_tmp(MODEL_BUCKET, FLOWHIFT_MODEL_S3_KEY, FLOWHIFT_TMP_DIR,
                  "cosyvoice3-flow-hift")
    cond = get_conditioning(os.environ["PROXY_BUCKET"],
                            voice["s3_conditioning_key"])

    n_timesteps = int(config.get("flow_steps", flow.DEFAULT_N_TIMESTEPS))

    ckpt = checkpoint.load(out_bucket, ckpt_key)
    if ckpt is not None:
        outputs = ckpt["outputs"]
        log.info("resuming: %d/%d transcripts already done",
                 len(outputs), len(transcripts))
    else:
        outputs = []

    for i in range(len(outputs), len(transcripts)):
        if context.get_remaining_time_in_millis() < checkpoint.REMAINING_TIME_BUFFER_MS:
            checkpoint.save(out_bucket, ckpt_key, {"outputs": outputs})
            log.info("checkpointed %d/%d transcripts — requesting retry",
                     len(outputs), len(transcripts))
            raise LambdaResourceTimeoutException(
                f"synthesize_flow_hift checkpointed at {len(outputs)}/{len(transcripts)}")

        t, llm_r = transcripts[i], llm_results[i]
        if "error" in llm_r:
            outputs.append({"error": f"skipped: LLM stage failed ({llm_r['error']})"})
            continue

        t0 = time.time()
        candidate_outputs = []
        last_error = None
        for ci, cand in enumerate(llm_r["candidates"]):
            try:
                mel = flow.synthesize_mel(
                    FLOWHIFT_TMP_DIR,
                    speech_tokens=cand["speech_tokens"],
                    prompt_speech_tokens=cond["speech_tokens"],
                    spk_embedding=cond["spk_embedding"],
                    prompt_mel=cond["prompt_mel"],
                    n_timesteps=n_timesteps,
                    # per-candidate seed, NOT config["seed"]: best-of-N only
                    # means anything if the candidates actually differ, and
                    # the Flow noise draw is half of what varies a take. A
                    # shared config seed would give every candidate the same
                    # noise and narrow the spread QC then picks from.
                    seed=cand["seed"],
                )
                audio = hift.synthesize_waveform(FLOWHIFT_TMP_DIR, mel)

                target = t.get("target_duration_s")
                stretched = False
                if target:
                    n_before = len(audio)
                    audio = fit_to_duration(audio, hift.SAMPLE_RATE, float(target))
                    # fit_to_duration returns the audio untouched when it is
                    # already within half a percent of the target, so the
                    # agent-facing `stretched` flag has to reflect what
                    # actually happened, not merely that a target was asked for.
                    stretched = len(audio) != n_before

                wav_key = (f"{out_prefix}/{item['job_id']}_{item['batch_index']}"
                          f"_{i}_c{ci}.wav")
                buf = io.BytesIO()
                sf.write(buf, audio, hift.SAMPLE_RATE, format="WAV", subtype="PCM_16")
                _s3.put_object(Bucket=out_bucket, Key=wav_key,
                               Body=buf.getvalue(), ContentType="audio/wav")

                candidate_outputs.append({
                    "wav_key": wav_key,
                    "duration_s": round(len(audio) / hift.SAMPLE_RATE, 2),
                    "stretched": stretched,
                    "seed": cand["seed"],
                })
            except DurationOutOfRangeError as exc:
                # explicit contract failure, not an infra error — record
                # and move on to the next candidate
                last_error = str(exc)
            except Exception as exc:
                log.exception("transcript %d/%d candidate %d failed",
                              i + 1, len(transcripts), ci)
                last_error = str(exc)

        if candidate_outputs:
            outputs.append({"candidates": candidate_outputs,
                            "flow_hift_seconds": round(time.time() - t0, 2)})
            log.info("transcript %d/%d: %d/%d candidates synthesized in %.1fs",
                     i + 1, len(transcripts), len(candidate_outputs),
                     len(llm_r["candidates"]), time.time() - t0)
        else:
            outputs.append({"error": last_error})

    # non-empty in practice; all() is vacuously True on [] and outputs[0]
    # below would IndexError over the real failure (see StateSynthesizeLLM)
    if outputs and all("error" in o for o in outputs):
        checkpoint.clear(out_bucket, ckpt_key)
        raise RuntimeError(
            f"all {len(outputs)} transcripts failed in StateSynthesizeFlowHift; "
            f"first error: {outputs[0]['error']}")

    checkpoint.clear(out_bucket, ckpt_key)
    # llm token arrays are no longer needed downstream — drop them so the
    # outgoing payload stays small (QC only needs wav_key + text)
    item.pop("llm_results", None)
    item["outputs"] = outputs
    return event
