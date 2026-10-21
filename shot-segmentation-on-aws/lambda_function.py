# lambda_function.py
"""
Lambda entry point — routes invocations to the correct state handler
based on the "operation" field in the input event.

Routing table
─────────────
  operation                             → handler module
  ─────────────────────────────────────────────────────────────────────────
  StateProbeVideoForChunkTranscode      → states.probe_video_for_chunk_transcode
  StateRunFFmpegTranscode               → states.run_ffmpeg_transcode
  StateDynamicChunkingForFrameEmbedding → states.dynamic_chunking_for_frame_embedding
  StateExtractFrameFeatures             → states.run_dino_v2_model
  StateMergeAndReconcileShots           → states.merge_reconcile_shots
  StateStoreDebugShotImages             → states.debug_shot_scene_images
  StateUsageTracking                    → states.usage_tracking

Design notes
────────────
  Handlers are imported lazily so a cold start for a lightweight state
  (e.g. StateUsageTracking) does not pay the ORT session-load cost of
  StateExtractFrameFeatures.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S"))
    log.addHandler(_h)


def lambda_handler(event, context):
    operation = event.get("operation")
    if not operation:
        raise ValueError("Missing 'operation'.")

    log.info("Routing: '%s'", operation)

    if operation == "StateProbeVideoForChunkTranscode":
        from states.probe_video_for_chunk_transcode import lambda_function as handler
        event["data"]["is_large_lambda"] = False
    elif operation == "StateProbeVideoForChunkTranscodeLargeFile":
        from states.probe_video_for_chunk_transcode import lambda_function as handler
        event["data"]["is_large_lambda"] = True
    elif operation == "StateRunFFmpegTranscode":
        from states.run_ffmpeg_transcode import lambda_function as handler
    elif operation == "StateDynamicChunkingForFrameEmbedding":
        from states.dynamic_chunking_for_frame_embedding import lambda_function as handler
    elif operation == "StateExtractFrameFeatures":
        from states.run_dino_v2_model import lambda_function as handler
    elif operation == "StateMergeAndReconcileShots":
        from states.merge_reconcile_shots import lambda_function as handler
    elif operation == "StateStoreDebugShotImages":
        from states.debug_shot_scene_images import lambda_function as handler
    elif operation == "StateUsageTracking":
        from states.usage_tracking import lambda_function as handler
    else:
        raise ValueError(f"Unknown operation: '{operation}'.")

    return handler.lambda_handler(event, context)
