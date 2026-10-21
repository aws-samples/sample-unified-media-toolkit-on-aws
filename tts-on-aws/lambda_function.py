"""
Lambda entry point — routes invocations to the correct state handler
based on the "operation" field in the input event. Same dispatch pattern
as speech-diarization-on-aws.

Two state machines share this one container image:

Synthesis pipeline (tts-synthesis)
──────────────────────────────────
  1. StateCheckAndPreprocess     validate batch request, expand iterators
  2. Map (one iteration per batch = one voice + N transcripts):
       a. StateSynthesizeLLM       text → speech tokens (in-Lambda batch
                                    loop; LLM loaded once per container)
       b. StateSynthesizeFlowHift  tokens → mel → waveform → S3, with
                                    optional bounded time-stretch
       c. StateVerifyOutput        ASR-roundtrip QC per clip
  3. StateUsageTracking          per-state cost/duration report

Register-voice pipeline (tts-register-voice)
────────────────────────────────────────────
  1. StateValidateReference      WAV contract + ASR transcript-match check
  2. Parallel:
       StateExtractSpeechToken       speech_tokenizer_v3 (~924 MB)
       StateExtractSpeakerEmbedding  CAMPPlus (~27 MB)
       StateExtractSpeechFeat        matcha mel (pure DSP, no model)
  3. StateWriteVoiceRecord       assemble conditioning blob → S3 + DDB

Handlers are imported lazily so a cold start for a light state never
pays a heavy state's import/session cost.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_h)


_last_operation: str | None = None


def lambda_handler(event, context):
    global _last_operation
    operation = event.get("operation")
    if not operation:
        raise ValueError("Missing 'operation'.")

    log.info("Routing: '%s'", operation)

    # Cross-state session eviction. All states share this image, so a warm
    # container that last served a different state still holds that state's
    # ORT sessions (~GBs). Found live: a register-voice speech-token
    # extraction (3.1 GB session) warmed the 2-cores container, the next
    # LLM invocation landed on it, and the LLM ran inside the leftover
    # memory — 900 s timeout thrashing at the memory ceiling instead of a
    # 95 s run. Same-state reuse (the batch-loop win) is preserved.
    if _last_operation is not None and _last_operation != operation:
        from lib.sessions import drop_sessions
        log.info("Operation changed (%s → %s) — dropping cached ORT sessions",
                 _last_operation, operation)
        drop_sessions()
    _last_operation = operation

    if operation == "StateCheckAndPreprocess":
        from states.check_and_preprocess import lambda_function as handler
    elif operation == "StateSynthesizeLLM":
        from states.synthesize_llm import lambda_function as handler
    elif operation == "StateSynthesizeFlowHift":
        from states.synthesize_flow_hift import lambda_function as handler
    elif operation == "StateVerifyOutput":
        from states.verify_output import lambda_function as handler
    elif operation == "StateValidateReference":
        from states.validate_reference import lambda_function as handler
    elif operation == "StateExtractSpeechToken":
        from states.extract_speech_token import lambda_function as handler
    elif operation == "StateExtractSpeakerEmbedding":
        from states.extract_spk_embedding import lambda_function as handler
    elif operation == "StateExtractSpeechFeat":
        from states.extract_speech_feat import lambda_function as handler
    elif operation == "StateWriteVoiceRecord":
        from states.write_voice_record import lambda_function as handler
    elif operation == "StateUsageTracking":
        from states.usage_tracking import lambda_function as handler
    else:
        raise ValueError(f"Unknown operation: '{operation}'.")

    return handler.lambda_handler(event, context)
