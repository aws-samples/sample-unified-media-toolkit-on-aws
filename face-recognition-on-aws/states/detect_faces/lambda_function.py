"""
StateDetectFaces — pass-2 dense-detection Map worker.

Identical mechanics to the probe worker (grid compositing → one DetectFaces
per grid → cell mapping → checkpoint/re-enter); only the iterator contents
differ (dense grids planned by StatePlanDenseDetection). Delegates.
"""

from __future__ import annotations

from states.probe_detect_faces.lambda_function import _run_detect


def lambda_handler(event: dict, context) -> dict:
    return _run_detect(event, context)
