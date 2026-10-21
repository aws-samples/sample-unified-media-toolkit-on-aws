"""
StatePrepareMediaProcessing — validate the event and build the
media-processing GATEWAY request (README.md §4; the family sibling
pattern since 2026-09-18 — this stack no longer invokes MP's state
machine directly, and the minted `facerec-{hex}` child uuid is retired:
MP names its own run from the request hash + our idempotency token).

MP extracts jpeg frames at frame_interval_sec, fit-inside 1920×1080
(MP's scale_dimension: never upscales, portrait handled via axis swap)
under ITS OWN service-derived run prefix; StateExtractProcessedFrames
derives the manifest/probe keys from MP's output contract.
Frame filenames encode the SOURCE frame number: frame_{n:07d}.jpg.
"""

from __future__ import annotations

import logging

from lib.input_schema import (
    FRAME_TARGET_HEIGHT,
    FRAME_TARGET_WIDTH,
    validate_and_normalize,
)

logger = logging.getLogger(__name__)


def lambda_handler(event: dict, context) -> dict:
    event = validate_and_normalize(event)
    cfg = event["input"]["input_config"]

    request: dict = {
        "asset_s3uri": event["input"]["s3uri"],
        "width": FRAME_TARGET_WIDTH,
        "height": FRAME_TARGET_HEIGHT,
        "output_groups": [
            {
                "group_name": "face_frames",
                "format": "jpeg",
                "frame_interval_sec": cfg["frame_interval_sec"],
            },
        ],
    }
    if cfg.get("use_partial") or (cfg.get("mark_in_out") or [0, 0])[1] > 0:
        mio = cfg["mark_in_out"]
        if mio[1] > mio[0] >= 0:
            request["use_partial"] = True
            request["mark_in_out"] = mio

    event["data"]["media_processing_request"] = request
    logger.info("media_processing_request built for %s", event["uuid"])
    return event
