# lambda_function.py
"""
Lambda entry point — routes invocations to the correct state handler based
on the "operation" field in the input event.

Pipeline overview (README.md; definitions.json)
─────────────────────────────────────────────
  Face recognition on video via Amazon Rekognition image APIs. Frames are
  extracted at 1 fps (configurable) by a media-processing child execution;
  a DynamoDB counting semaphore bounds concurrent Rekognition usage
  account-wide (acquired AFTER frame extraction, released after the last
  Rekognition state).

  1. StatePrepareMediaProcessing   Build the media-processing child input
                                   (jpeg frame output group, fit-inside
                                   1920×1080, never upscale).
  2. (child SM via .sync:2)        Frame extraction → frame_manifest.json.
  3. StatePrepareProbeIterators    Frame triage (size heuristic + luma stats)
                                   + 3×3 patch dHashes + probe grid packing.
  4. StateAcquireSemaphore         waitForTaskToken slot admission.
  5. StateProbeDetectFaces  (Map)  Pass 1: DetectFaces on every Nth frame,
                                   packed into grid images (S3Object ref).
  6. StatePlanDenseDetection       Skip/dense gap classification (patch-noise
                                   ticker discount; endpoint drift check).
  7. StateDetectFaces       (Map)  Pass 2: dense grids over face regions with
                                   near-duplicate dedup.
  8. StateBuildFaceTracks          Temporal tracking (gap-dilated IoU, gender,
                                   hash-break veto) + box interpolation +
                                   centroid selection.
  9. StateRecognizeCelebrities(Map) Headshot grids → RecognizeCelebrities.
 10. StateClusterIdentities        Sandbox collection IndexFaces/SearchFaces →
                                   transitive identity clusters + similarity
                                   matrix capture (for MDS viz).
 11. StateSearchIndexFaces         Persistent-collection SearchFacesByImage +
                                   policy-gated auto-index + DDB registration.
 12. StateReleaseSemaphore         Slot release (holder delete only; the DDB
                                   stream decrements the counter).
 13. StateFinalizeOutputs          facemetadata.json / facegroup.json /
                                   face_clusters.json (MDS).
 14. StateStoreDebugImages         Per-person montages + indexed-faces grid.
 15. StateUsageTracking            Per-state cost + child MP cost + API tallies.

  Out-of-band handlers (not in the state machine):
    StateHandleSemaphoreRelease    DDB stream: exactly-once counter decrement
                                   + FIFO pending drain.
    StateHandleExecutionTerminal   EventBridge (ALL terminal states): holder
                                   delete + sandbox collection delete.

Recognition policy (GDPR use cases — README.md §2b)
─────────────────────────────────────────────────
  celebrity_recognition=false     no RecognizeCelebrities calls
  collection_id=""                no persistent-collection ops (search+index)
  max_faces_to_index=0            auto-index off (persistent search still on)
  index_named_faces_only=true     auto-index only high-confidence-named faces
  The sandbox collection is transient (deleted at terminal state) and runs in
  ALL use cases — clustering quality is identical across policy variants.

Self-checkpoint re-enter
────────────────────────
  Long-running states checkpoint partial results to S3, monitor remaining
  Lambda time, and raise LambdaResourceTimeoutException (lib/custom_exceptions.py)
  near the deadline; the ASL retries the state and completed work is skipped
  on re-entry.
"""

from __future__ import annotations

import json
import logging

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("face-recognition")
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    operation = event.get("operation") or ""
    logger.info("operation=%s", operation)
    logger.info("event event keys = %s", sorted(event.keys()))

    # Lazy imports: each state loads only its own dependencies.
    if operation == "StatePrepareMediaProcessing":
        from states.prepare_media_processing import lambda_function as handler
    elif operation == "StateSubmitMediaProcessing":
        from states.submit_media_processing import lambda_function as handler
    elif operation == "StateExtractProcessedFrames":
        from states.extract_processed_frames import lambda_function as handler
    elif operation == "StatePrepareProbeIterators":
        from states.prepare_probe_iterators import lambda_function as handler
    elif operation == "StateAcquireSemaphore":
        from states.acquire_semaphore import lambda_function as handler
    elif operation == "StateProbeDetectFaces":
        from states.probe_detect_faces import lambda_function as handler
    elif operation == "StatePlanDenseDetection":
        from states.plan_dense_detection import lambda_function as handler
    elif operation == "StateDetectFaces":
        from states.detect_faces import lambda_function as handler
    elif operation == "StateBuildFaceTracks":
        from states.build_face_tracks import lambda_function as handler
    elif operation == "StateRecognizeCelebrities":
        from states.recognize_celebrities import lambda_function as handler
    elif operation == "StateClusterIdentities":
        from states.cluster_identities import lambda_function as handler
    elif operation == "StateSearchIndexFaces":
        from states.search_index_faces import lambda_function as handler
    elif operation == "StateReleaseSemaphore":
        from states.release_semaphore import lambda_function as handler
    elif operation == "StateFinalizeOutputs":
        from states.finalize_outputs import lambda_function as handler
    elif operation == "StateStoreDebugImages":
        from states.store_debug_images import lambda_function as handler
    elif operation == "StateUsageTracking":
        from states.usage_tracking import lambda_function as handler
    elif operation == "StateValidateIndexBatch":
        from states.validate_index_batch import lambda_function as handler
    elif operation == "StateBatchIndexWorker":
        from states.batch_index_worker import lambda_function as handler
    elif operation == "StateBatchIndexFinalize":
        from states.batch_index_finalize import lambda_function as handler
    elif operation == "StateBatchDeleteFaces":
        from states.batch_delete_faces import lambda_function as handler
    elif _is_stream_event(event):
        from states.handle_semaphore_release import lambda_function as handler
    elif _is_eventbridge_terminal(event):
        from states.handle_execution_terminal import lambda_function as handler
    else:
        raise ValueError(f"unknown operation: {operation!r}")

    return handler.lambda_handler(event, context)


def _is_stream_event(event: dict) -> bool:
    records = event.get("Records") or []
    return bool(records and records[0].get("eventSource") == "aws:dynamodb")


def _is_eventbridge_terminal(event: dict) -> bool:
    return (event.get("source") == "aws.states"
            and "detail" in event)
