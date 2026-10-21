# lambda_function.py
"""
Lambda entry point — routes invocations to the correct state handler based
on the "operation" field in the input event.

Routing table
─────────────
  operation                                  → handler module
  ──────────────────────────────────────────────────────────────────────────
  StateProbeMediaFile       → states/probe_media_file/lambda_function.py
  StateChunkTranscode       → states/chunk_transcode/lambda_function.py
  StateMergeChunks          → states/merge_chunks/lambda_function.py
  StateUsageTracking        → states/usage_tracking/lambda_function.py

Design notes
────────────
Media Processing workflow can handle:
  • Proxy generation - Options to configure input cropping, audio channel selection, bitrate, gopsize, resolution
  • Audio extraction - to WAV and other formats. Options to configure sampling frequency, bit depth, mono/stereo output. Options to select input audio channel mapping.
  • Frame extraction - Options to configure frame capture rate, image resolution, input cropping.

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

    if operation == "StateProbeMediaFile":
        from states.probe_media_file import lambda_function as handler
    elif operation == "StateChunkTranscode":
        from states.chunk_transcode import lambda_function as handler
    elif operation == "StateMergeChunks":
        from states.merge_chunks import lambda_function as handler
    elif operation == "StateUsageTracking":
        from states.usage_tracking import lambda_function as handler
    else:
        raise ValueError(f"Unknown operation: '{operation}'.")

    return handler.lambda_handler(event, context)
