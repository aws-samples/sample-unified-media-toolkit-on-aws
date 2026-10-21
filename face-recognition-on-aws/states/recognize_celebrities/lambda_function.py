"""
StateRecognizeCelebrities — Map worker (README.md §4).

Per batch (≤64 track centroids): crop centroid headshots (120×160, 30 %
margin), tile into one grid, ONE RecognizeCelebrities call (S3Object ref),
map CelebrityFaces/UnrecognizedFaces back to tiles by bbox centre. Grid JPEG
+ raw response stored under grids/ for audit. Checkpoint/re-enter per batch.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from lib import semaphore
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.grid_image import (compose_tile_grid, crop_headshot,
                            dominant_per_tile)
from lib.input_schema import (
    MIN_SUBJECT_TILE_SHARE,
    CHECKPOINT_EVERY_N_GRIDS,
    HEADSHOT_CROP_MARGIN,
    HEADSHOT_DIMENSION,
    REMAINING_TIME_BUFFER_MS,
)
from lib.rekognition_utils import CallCounter, recognize_celebrities
from lib.s3_utils import get_bytes, get_json, parse_s3uri, put_bytes, put_json

logger = logging.getLogger(__name__)

_TILE_COLS = 8


def lambda_handler(event: dict, context) -> dict:
    item = event["item_data"]
    ctx = item["context"]
    out_bucket = ctx["out_bucket"]
    execution_arn = event.get("execution_arn") or ""
    if execution_arn:
        semaphore.refresh_holder(execution_arn)

    iterator = get_json(out_bucket, item["key"])
    results: dict = iterator.get("results") or {}
    counter = CallCounter()
    counter.add(iterator.get("usage") or {})

    all_tracks = {tr["track_id"]: tr
                  for tr in get_json(out_bucket, ctx["tracks_key"])["tracks"]}
    grids_prefix = f"{parse_s3uri(ctx['output_s3uri'])[1].rstrip('/')}/grids"
    tile_w, tile_h = HEADSHOT_DIMENSION
    since_checkpoint = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        for bi, batch in enumerate(iterator["batches"]):
            if str(bi) in results:
                continue
            if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
                iterator["results"] = results
                iterator["usage"] = counter.as_dict()
                put_json(out_bucket, item["key"], iterator)
                raise LambdaResourceTimeoutException(
                    f"recognize checkpointed at batch {bi}")

            def _headshot(track_id: int) -> bytes:
                tr = all_tracks[track_id]
                face = tr["faces"][tr["centroid_idx"]]
                frame_jpeg = get_bytes(
                    ctx["frames_bucket"],
                    f"{ctx['frames_prefix']}/{face['frame_key']}")
                return crop_headshot(frame_jpeg, face["box"],
                                     HEADSHOT_CROP_MARGIN, tile_w, tile_h)

            tiles = list(pool.map(_headshot, batch))
            grid_jpeg, coords = compose_tile_grid(tiles, tile_w, tile_h,
                                                  _TILE_COLS)
            grid_key = (f"{grids_prefix}/celeb_{item['item_id']:03d}"
                        f"_{bi:04d}.jpg")
            put_bytes(out_bucket, grid_key, grid_jpeg, "image/jpeg")

            resp = recognize_celebrities(out_bucket, grid_key, counter)
            put_json(out_bucket, grid_key + ".json", resp)

            batch_result: dict = {}
            # ONE tag per tile, from the tile's DOMINANT face across BOTH
            # response lists: a background celebrity (a face on a studio
            # monitor behind the subject) must neither name the tile nor
            # outrank a larger unrecognized subject.
            candidates = [(celeb.get("Face") or {}, {
                "name": celeb.get("Name"),
                "confidence": float(celeb.get("MatchConfidence") or 0),
                # the recognizer's own identity key for this
                # celebrity — carried to the product so consumers
                # can join two clusters the recognizer asserts are
                # ONE person (the collection face_id analogue;
                # cast-resolution coalesces on it, 2026-09-07)
                "celebrity_id": celeb.get("Id"),
            }) for celeb in resp["CelebrityFaces"]]
            candidates += [(unrec, {
                "name": None,
                "confidence": float(unrec.get("Confidence") or 0),
            }) for unrec in resp["UnrecognizedFaces"]]
            for idx, tag in dominant_per_tile(
                    candidates, coords,
                    min_share=MIN_SUBJECT_TILE_SHARE).items():
                if idx < len(batch):
                    batch_result[str(batch[idx])] = tag
            results[str(bi)] = batch_result
            since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY_N_GRIDS:
                # Periodic checkpoint: without it, a NON-timeout crash
                # (throttle escaping the retry layers, Lambda.Unknown) loses
                # every completed batch since the last write — the Map retry
                # re-enters with results={} and RE-BILLS every
                # RecognizeCelebrities call made this invocation.
                iterator["results"] = results
                iterator["usage"] = counter.as_dict()
                put_json(out_bucket, item["key"], iterator)
                since_checkpoint = 0

    iterator["results"] = results
    iterator["usage"] = counter.as_dict()
    put_json(out_bucket, item["key"], iterator)
    return {"item_id": item["item_id"], "key": item["key"],
            "usage": counter.as_dict()}

