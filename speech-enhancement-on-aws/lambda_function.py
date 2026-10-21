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
  StateFixedChunkingForSpeechEnhancement   → states/fixed_chunking_for_speech_enhancement/lambda_function.py
  StateRunSpeechEnhancementModel                      → states/run_speech_enhancement_model/lambda_function.py
  StateConcatenateEnhancedAudio         → states/concatenate_enhanced_audio/lambda_function.py
  StateUsageTracking                         → states/usage_tracking/lambda_function.py

Design notes
────────────
  • Handlers are imported lazily (inside the dispatch branch) so that a cold
    start for a lightweight state does not pay the ORT session-load cost
    incurred by StateRunSpeechEnhancementModel.
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
    elif operation == "StateFixedChunkingForSpeechEnhancement":
        from states.fixed_chunking_for_speech_enhancement import lambda_function as handler
    elif operation == "StateRunSpeechEnhancementModel":
        from states.run_speech_enhancement_model import lambda_function as handler
    elif operation == "StateConcatenateEnhancedAudio":
        from states.concatenate_enhanced_audio import lambda_function as handler
    elif operation == "StateUsageTracking":
        from states.usage_tracking import lambda_function as handler
    else:
        raise ValueError(f"Unknown operation: '{operation}'.")

    return handler.lambda_handler(event, context)
