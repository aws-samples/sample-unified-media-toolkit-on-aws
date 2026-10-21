"""
StateProbeDetectFaces / StateDetectFaces — Map workers (README.md §3, §4).

Both passes share this implementation (the dispatcher routes both operations
here via detect_faces/lambda_function.py delegating to _run_detect):

Per grid: download the cell frames (threaded), composite with separators
(pyvips), upload to grids/, ONE DetectFaces call by S3Object reference, map
faces back to cells (bbox-centre; bleed artifacts rejected), recalibrate to
frame-relative boxes.

Grid-split safeguard: >= GRID_SPLIT_FACE_COUNT faces returned → split the
grid in half and re-detect both halves (recursive, min GRID_SPLIT_MIN_FRAMES
frames) — crowd scenes the old pipeline silently truncated.

Checkpoint/re-enter — O(n) side objects, not O(n²) full rewrites: each
grid's result (faces + its own usage delta) is written ONCE to
{iterator_key}.grid_{gi}.json immediately after its DetectFaces call
completes. Re-entry lists that side prefix to skip completed grids. When
all grids are done, the results are consolidated into the iterator JSON
(single write) and the side objects are batch-deleted. The old scheme
rewrote the ENTIRE accumulated results dict every 5 grids — total
checkpoint bytes grew quadratically with grids per iterator (~10-25 MB of
redundant PUT traffic per iterator on a 2 h asset), paid on every clean
run for a rare-crash benefit.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from lib import semaphore
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.face_quality import to_face_data
from lib.grid_image import compose_grid, map_face_to_cell
from lib.input_schema import (
    GRID_SPLIT_FACE_COUNT,
    GRID_SPLIT_MIN_FRAMES,
    REMAINING_TIME_BUFFER_MS,
)
from lib.rekognition_utils import CallCounter, detect_faces
from lib.s3_utils import (
    delete_keys,
    get_bytes,
    get_json,
    list_keys,
    parse_s3uri,
    put_bytes,
    put_json,
)

logger = logging.getLogger(__name__)


def lambda_handler(event: dict, context) -> dict:
    return _run_detect(event, context)


def _grid_result_key(iterator_key: str, gi: int) -> str:
    return f"{iterator_key}.grid_{gi:04d}.json"


def _run_detect(event: dict, context) -> dict:
    item = event["item_data"]
    # item = iterator entry {item_id, key, context} stamped by the packing
    # state; execution_arn arrives via the Map ItemSelector ($$.Execution.Id).
    ctx = item.get("context") or {}
    out_bucket = ctx["out_bucket"]
    frames_bucket = ctx["frames_bucket"]
    frames_prefix = ctx["frames_prefix"]
    execution_arn = event.get("execution_arn") or ""

    if execution_arn:
        semaphore.refresh_holder(execution_arn)

    iterator = get_json(out_bucket, item["key"])
    layout = iterator["layout"]
    # results may already be consolidated in the iterator (a retry AFTER the
    # final write); side objects cover the partial-progress window.
    results: dict = iterator.get("results") or {}
    counter = CallCounter()
    counter.add(iterator.get("usage") or {})

    side_keys = list_keys(out_bucket, item["key"] + ".grid_")
    for key in sorted(side_keys):
        side = get_json(out_bucket, key)
        gi_str = str(int(key.rsplit("grid_", 1)[1].split(".")[0]))
        if gi_str not in results:
            results[gi_str] = side["faces"]
            counter.add(side.get("usage") or {})

    grids_prefix = f"{parse_s3uri(ctx['output_s3uri'])[1].rstrip('/')}/grids"

    pending = [(gi, grid) for gi, grid in enumerate(iterator["grids"])
               if str(gi) not in results]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for gi, grid in pending:
            if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
                # Side objects already persist every completed grid — nothing
                # to flush; just signal the re-enter.
                raise LambdaResourceTimeoutException(
                    f"detect checkpointed at grid {gi}")

            frames = grid["frames"]
            jpegs = list(pool.map(
                lambda f: get_bytes(frames_bucket,
                                    f"{frames_prefix}/{f['key']}"),
                frames))
            grid_key = f"{grids_prefix}/{item['item_id']:03d}_{gi:04d}.jpg"
            grid_counter = CallCounter()
            faces = _detect_grid(
                jpegs, frames, layout, out_bucket, grid_key, grid_counter)
            # O(n) checkpoint: this grid's result + usage delta, written once.
            put_json(out_bucket, _grid_result_key(item["key"], gi),
                     {"faces": faces, "usage": grid_counter.as_dict()})
            results[str(gi)] = faces
            counter.add(grid_counter.as_dict())

    # Consolidate: single full write, then drop the side objects.
    iterator["results"] = results
    iterator["usage"] = counter.as_dict()
    put_json(out_bucket, item["key"], iterator)
    all_side = list_keys(out_bucket, item["key"] + ".grid_")
    if all_side:
        delete_keys(out_bucket, sorted(all_side))
    return {"item_id": item["item_id"], "key": item["key"],
            "usage": counter.as_dict()}


def _detect_grid(jpegs: list[bytes], frames: list[dict], layout: dict,
                 out_bucket: str, grid_key: str,
                 counter: CallCounter) -> list[dict]:
    """Detect one grid; recursive split on face-count overflow."""
    cols, rows = layout["cols"], layout["rows"]
    cell_w, cell_h = layout["cell_w"], layout["cell_h"]

    grid_jpeg = compose_grid(jpegs, cols, rows, cell_w, cell_h)
    put_bytes(out_bucket, grid_key, grid_jpeg, "image/jpeg")
    details = detect_faces(out_bucket, grid_key, counter)

    if len(details) >= GRID_SPLIT_FACE_COUNT and len(jpegs) > GRID_SPLIT_MIN_FRAMES:
        logger.info("grid split: %d faces in %s", len(details), grid_key)
        half = len(jpegs) // 2
        left = _detect_grid(jpegs[:half], frames[:half], layout,
                            out_bucket, grid_key + ".a.jpg", counter)
        right = _detect_grid(jpegs[half:], frames[half:], layout,
                             out_bucket, grid_key + ".b.jpg", counter)
        return left + right

    faces = []
    for detail in details:
        mapped = map_face_to_cell(detail["BoundingBox"], len(jpegs),
                                  cols, rows, cell_w, cell_h)
        if mapped is None:
            continue  # bleed artifact or empty cell
        idx, box = mapped
        frame = frames[idx]
        faces.append({
            "frame_num": frame["frame_num"],
            "timestamp": frame["timestamp"],
            "frame_key": frame["key"],
            "box": box,
            **to_face_data(detail),
        })
    return faces
