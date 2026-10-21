# lambda_function.py
"""
Lambda entry point — routes invocations to the correct state handler based
on the "operation" field in the input event.

Routing table
─────────────
  operation                                  → handler module
  ──────────────────────────────────────────────────────────────────────────
  StateCheckAndPreprocess                    → states/check_and_preprocess/lambda_function.py
  StateExtractProcessedAudio                 → states/extract_processed_audio/lambda_function.py
  StateFixedChunkingForAudioStemSeparation   → states/fixed_chunking_for_audio_stem_separation/lambda_function.py
  StateRunBanditV2Model                      → states/run_bandit_v2_model/lambda_function.py
  StateConcatenateAudioChunksByStems         → states/concatenate_audio_chunks_by_stems/lambda_function.py
  StateUsageTracking                         → states/usage_tracking/lambda_function.py

Design notes
────────────
  • Handlers are imported lazily (inside the dispatch branch) so that a cold
    start for a lightweight state does not pay the ORT session-load cost
    incurred by StateRunBanditV2Model.
  • Each handler module is responsible for its own logging setup and any
    module-level initialisation (ORT cpuinfo fix, session cache, etc.).
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


def lambda_handler(event, context):
    operation = event.get("operation")
    if not operation:
        raise ValueError(f"Missing 'operation'.")

    log.info(f"Routing: '{operation}'")

    if operation == "StateCheckAndPreprocess":
        from states.check_and_preprocess import lambda_function as handler
    elif operation == "StateSubmitMediaProcessing":
        from states.submit_media_processing import lambda_function as handler
    elif operation == "StateExtractProcessedAudio":
        from states.extract_processed_audio import lambda_function as handler
    elif operation == "StateFixedChunkingForAudioStemSeparation":
        from states.fixed_chunking_for_audio_stem_separation import lambda_function as handler
    elif operation == "StateRunBanditV2Model":
        from states.run_bandit_v2_model import lambda_function as handler
    elif operation == "StateConcatenateAudioChunksByStems":
        from states.concatenate_audio_chunks_by_stems import lambda_function as handler
    elif operation == "StateUsageTracking":
        from states.usage_tracking import lambda_function as handler
    else:
        raise ValueError(f"Unknown operation: '{operation}'.")

    return handler.lambda_handler(event, context)
