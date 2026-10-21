"""
StatePrepareProbeIterators — frame triage + patch hashing + probe grid
packing (README.md §3, §4). S3/CPU only — runs BEFORE the semaphore.

Reads frame_manifest.json ({total, prefix, keys, dimension?} — dimension is
OPTIONAL; falls back to decoding one frame). Every frame is decoded (threaded
S3 GET + pyvips shrink-on-load decode) and classified from actual pixel
content — no zero-decode byte-size pre-filter (removed: JPEG size tracks
scene complexity, not face presence; see lib/frame_triage.py docstring):
  luma stats (black / mono / extreme blur) → drop
  keep: 3×3 patch + global dHash recorded in frame_index

Probe frames (every probe_stride-th kept frame) are packed into grid
iterators for the probe Map. Checkpoint/re-enter: partial frame_index is
persisted and LambdaResourceTimeoutException raised near the deadline.

Also validates the optional shot-segmentation supplement (README.md §21)
here — this state has the exact probe fps + full-source duration for the
same-asset cross-checks and runs BEFORE the semaphore, so a wrong shots
file fails the run before any Rekognition spend.

Writes:
  ephemeral/frame_index.json      per-frame {key, frame_num, timestamp,
                                  size, dropped?, reason?, hashes?}
  ephemeral/shot_boundaries.json  validated shot-start frames (shots mode)
  ephemeral/probe_{i}.json        probe grid iterators
  data.iterators, data.frame_meta; clears media_processing_input
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from lib import frame_hash, frame_triage
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.grid_image import solve_layout
from lib.input_schema import (
    FRAMES_PER_GRID,
    REMAINING_TIME_BUFFER_MS,
    TRIAGE_POOL_THREADS,
    get_input_config,
)
from lib.s3_utils import (
    get_bytes,
    get_json,
    get_json_if_exists,
    parse_s3uri,
    put_json,
)

logger = logging.getLogger(__name__)

# Grids per probe iterator: bounded so one iterator ≈ minutes of API work.
_GRIDS_PER_ITERATOR = 40
_MAX_ITERATORS = 10


def lambda_handler(event: dict, context) -> dict:
    cfg = get_input_config(event)
    mp_out = event["data"]["media_processing_output"]
    frames_bucket = mp_out["frames_bucket"]
    manifest = get_json(frames_bucket, mp_out["manifest_key"])

    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")
    index_key = f"{out_prefix}/ephemeral/frame_index.json"

    fps = _resolve_fps(frames_bucket, mp_out.get("probe_key"))
    frames_prefix = manifest["prefix"].rstrip("/")
    keys = manifest["keys"]

    # ── Optional shot-segmentation supplement (README.md §21) ──────────────────
    # Validate BEFORE triage (fail fast on a wrong-asset file) and persist
    # the boundary list; downstream states consume the validated list only.
    shot_boundaries_key = None
    shots_uri = cfg.get("shot_segmentation_s3uri") or ""
    if shots_uri:
        from lib import shots as shots_lib
        duration_sec = mp_out.get("duration_sec")
        if not duration_sec or duration_sec <= 0:
            raise ValueError(
                "shot_segmentation_s3uri supplied but media_processing_output"
                " has no positive duration_sec — cannot cross-check the "
                "shots file against this asset")
        s_bucket, s_key = parse_s3uri(shots_uri)
        starts = shots_lib.load_and_validate(
            s_bucket, s_key, fps, float(duration_sec))
        shot_boundaries_key = f"{out_prefix}/ephemeral/shot_boundaries.json"
        put_json(out_bucket, shot_boundaries_key,
                 {"source": shots_uri, "starts": starts})
        logger.info("shot supplement validated: %d boundaries from %s",
                    len(starts), shots_uri)

    # Resume from checkpoint if a prior invocation timed out. Only a missing
    # key (no checkpoint yet) is treated as "start fresh" — any other S3
    # failure (throttling, access denied, corrupted JSON) propagates instead
    # of silently discarding a real checkpoint and re-triaging already-billed
    # work.
    existing = get_json_if_exists(out_bucket, index_key)
    if existing is not None:
        frame_index = existing
        done = {f["key"] for f in frame_index["frames"]}
        logger.info("resuming triage: %d/%d frames done", len(done), len(keys))
    else:
        frame_index = {"frames": []}
        done = set()

    dimension = manifest.get("dimension") or [0, 0]
    todo = [k for k in keys if k not in done]

    def _process(key: str) -> dict:
        full_key = f"{frames_prefix}/{key}"
        frame_num = int(key.split("_")[1].split(".")[0])
        entry: dict = {
            "key": key,
            "frame_num": frame_num,
            "timestamp": round(float(frame_num / fps), 4),
        }
        # Every frame is decoded and classified from actual pixel content —
        # no zero-decode byte-size pre-filter (removed: JPEG size tracks
        # scene complexity, not face presence; see frame_triage.py docstring
        # for the real frame that heuristic silently discarded).
        jpeg = get_bytes(frames_bucket, full_key)
        entry["size"] = len(jpeg)
        luma = frame_triage.decode_luma(jpeg)
        reason = frame_triage.triage_luma(luma)
        if reason:
            entry.update(dropped=True, reason=reason)
            return entry
        entry["hashes"] = frame_hash.frame_hashes(luma)
        return entry

    deadline_hit = False
    with ThreadPoolExecutor(max_workers=TRIAGE_POOL_THREADS) as pool:
        batch = 200
        for i in range(0, len(todo), batch):
            if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
                deadline_hit = True
                break
            chunk = todo[i:i + batch]
            frame_index["frames"].extend(pool.map(_process, chunk))

    # Fill dimension from a real frame when the manifest omitted it.
    if not (dimension[0] and dimension[1]):
        dimension = _probe_dimension(frames_bucket, frames_prefix,
                                     frame_index["frames"])
        frame_index["dimension"] = dimension

    put_json(out_bucket, index_key, frame_index)
    if deadline_hit:
        raise LambdaResourceTimeoutException(
            f"triage checkpointed at {len(frame_index['frames'])}/{len(keys)}")

    frame_index["frames"].sort(key=lambda f: f["frame_num"])
    kept = [f for f in frame_index["frames"] if not f.get("dropped")]
    dropped = len(frame_index["frames"]) - len(kept)
    logger.info("triage: %d kept, %d dropped", len(kept), dropped)

    # ── Probe selection + grid packing ────────────────────────────────────────
    stride = cfg["probe_stride"]
    probes = kept if stride == 0 else kept[::stride]

    cell_w, cell_h = dimension
    cols, rows = solve_layout(cell_w, cell_h)
    per_grid = min(cols * rows, FRAMES_PER_GRID)

    grids = [probes[i:i + per_grid] for i in range(0, len(probes), per_grid)]
    per_iter = max(_GRIDS_PER_ITERATOR,
                   math.ceil(len(grids) / _MAX_ITERATORS))
    # Shared worker context, stamped into every iterator entry (Map items
    # carry only item_data — workers never see the outer event).
    worker_ctx = {
        "out_bucket": out_bucket,
        "output_s3uri": event["output"]["s3uri"],
        "frames_bucket": frames_bucket,
        "frames_prefix": frames_prefix,
    }
    iterators = []
    for item_id, gi in enumerate(range(0, len(grids), per_iter)):
        iter_key = f"{out_prefix}/ephemeral/probe_{item_id}.json"
        put_json(out_bucket, iter_key, {
            "grids": [
                {"frames": [{"key": f["key"], "frame_num": f["frame_num"],
                             "timestamp": f["timestamp"]}
                            for f in grid]}
                for grid in grids[gi:gi + per_iter]
            ],
            "layout": {"cols": cols, "rows": rows,
                       "cell_w": cell_w, "cell_h": cell_h},
            "results": {},
            "usage": {},
        })
        iterators.append({"item_id": item_id, "key": iter_key,
                          "context": worker_ctx})

    event["data"]["iterators"] = iterators
    event["data"]["frame_meta"] = {
        # Exact fraction as a string (convention: shot-segmentation-on-aws
        # stores "frame_rate" the same way) — "fps" is a float convenience
        # DERIVED from it for logging/display only. Any future arithmetic
        # consumer must parse "frame_rate" via Fraction(), never trust "fps".
        "frame_rate": f"{fps.numerator}/{fps.denominator}",
        "fps": float(fps),
        "wxh": dimension,
        "frames_prefix": frames_prefix,
        "total_frames": len(keys),
        "kept_frames": len(kept),
        "triaged_out": dropped,
        "frame_index_key": index_key,
        "shot_boundaries_key": shot_boundaries_key,
        "grid_layout": {"cols": cols, "rows": rows},
    }
    event["data"]["media_processing_input"] = None
    return event


def _resolve_fps(bucket: str, probe_key: str | None) -> Fraction:
    """
    Exact fps from the media-processing probe.json, r_frame_rate as an exact
    fraction (e.g. "24000/1001"). NO fallback, NO default — every timestamp
    downstream is frame_num / fps, so a wrong fps silently corrupts every
    appearance span in facemetadata.json with no error anywhere. A wrong
    number is worse than a crash: fail loudly instead of guessing.

    probe.json schema (written by media-processing StateProbeMediaFile):
      { ..., "probe_info": { "streams": [ {"codec_type": "video",
                                            "r_frame_rate": "24000/1001", ...} ] } }
    Note "probe_info" — NOT top-level "streams".
    """
    if not probe_key:
        raise ValueError(
            "Cannot determine frame rate: media_processing_output has no "
            "probe_key — media-processing did not report a probe.json "
            "location. Refusing to guess a frame rate.")

    probe = get_json(bucket, probe_key)
    streams = (probe.get("probe_info") or {}).get("streams") or []
    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        rate = stream.get("r_frame_rate") or stream.get("avg_frame_rate") or ""
        if "/" not in rate:
            continue
        num_str, _, den_str = rate.partition("/")
        try:
            num, den = int(num_str), int(den_str)
        except ValueError:
            continue
        if den == 0 or num <= 0:
            continue
        return Fraction(num, den)

    raise ValueError(
        f"Cannot determine frame rate from probe.json "
        f"(s3://{bucket}/{probe_key}): no video stream with a valid "
        f"r_frame_rate/avg_frame_rate found in probe_info.streams "
        f"(streams={streams!r}). Refusing to guess a frame rate.")


def _probe_dimension(bucket: str, prefix: str, frames: list[dict]) -> list[int]:
    for f in frames:
        if not f.get("dropped"):
            import pyvips
            jpeg = get_bytes(bucket, f"{prefix}/{f['key']}")
            img = pyvips.Image.jpegload_buffer(jpeg)
            return [img.width, img.height]
    return [0, 0]
