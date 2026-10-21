# debug_shot_scene_images_handler.py
"""
Step Function State: Store Debug Shot Images
---------------------------------------------
For every shot, composes a grid image from the archived model-input
frames (native-AR JPEGs at proxy resolution, repackaged per shot by
the merge state), downscaling each into a native-AR cell whose longest
side is 112 px (e.g. 512×288 frames → 112×63 cells), and uploads to S3.

This state does NOT run ffmpeg.  Grid images show the exact pixels
DINOv2 processed — no seeking discrepancy is possible.

S3 output
─────────
  {prefix}/shot_images/
      shot_000000.jpg
      shot_000001.jpg
      ...
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tarfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
if not log.handlers:
    log.addHandler(_handler)

from lib.s3_utils import (
    get_json,
    get_binary,
    upload_file,
)

_CELL_SIZE = 112   # longest side of a grid cell; the short side follows the frame AR (e.g. 512×288 → 112×63)
_GRID_COLS = 12
_GRID_LINE_WIDTH = 2
_JPEG_QUALITY = 75
_SYNC_POP_BORDER_COLOR = (255, 220, 0)
_SYNC_POP_BORDER_WIDTH = 4


# =============================================================================
# Grid composition
# =============================================================================

def _compose_grid(
    frame_paths:     list[str],
    special_indices: set[int] | None = None,
) -> Image.Image:
    """
    Compose a grid of native-AR cells.

    The cell size is derived from the first readable frame's aspect
    ratio (all frames in a shot come from the same proxy, so they share
    it): the longest side is scaled to _CELL_SIZE and the other side
    follows the AR — e.g. 512×288 frames → 112×63 cells. No per-cell
    padding; unreadable frames render as a solid dark-red cell.
    """
    special_indices = special_indices or set()
    n_cells = len(frame_paths)
    n_cols = min(_GRID_COLS, n_cells)
    n_rows = math.ceil(n_cells / n_cols)

    # ── Derive cell dims from the first readable frame ───────────
    cell_w = cell_h = _CELL_SIZE
    for path in frame_paths:
        try:
            with Image.open(path) as probe:
                w, h = probe.size
            scale = _CELL_SIZE / max(w, h)
            cell_w = max(1, round(w * scale))
            cell_h = max(1, round(h * scale))
            break
        except Exception:
            continue

    grid_w = n_cols * cell_w
    grid_h = n_rows * cell_h
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))

    for idx, path in enumerate(frame_paths):
        try:
            cell = Image.open(path).convert("RGB")
            if cell.size != (cell_w, cell_h):
                cell = cell.resize((cell_w, cell_h), Image.BILINEAR)
        except Exception:
            cell = Image.new(
                "RGB", (cell_w, cell_h), color=(80, 0, 0))

        if idx in special_indices:
            draw = ImageDraw.Draw(cell)
            bw = _SYNC_POP_BORDER_WIDTH
            draw.rectangle(
                [(0, 0), (cell_w - 1, cell_h - 1)],
                outline=_SYNC_POP_BORDER_COLOR,
                width=bw,
            )

        row = idx // n_cols
        col = idx % n_cols
        grid_img.paste(cell, (col * cell_w, row * cell_h))

    # ── Draw white grid lines on top of the composite ────────────
    draw = ImageDraw.Draw(grid_img)
    lw = _GRID_LINE_WIDTH

    # Vertical lines (between columns)
    for c in range(1, n_cols):
        x = c * cell_w
        draw.line([(x, 0), (x, grid_h - 1)], fill=(255, 255, 255), width=lw)

    # Horizontal lines (between rows)
    for r in range(1, n_rows):
        y = r * cell_h
        draw.line([(0, y), (grid_w - 1, y)], fill=(255, 255, 255), width=lw)

    return grid_img


# =============================================================================
# Per-shot worker
# =============================================================================

def _process_shot_from_frames(
    shot:       dict,
    tmp_dir:    str,
    out_bucket: str,
    out_key:    str,
) -> dict:
    """
    Download per-shot frames tar.gz (native-AR selected frames written by
    StateMergeAndReconcileShots), downscale each frame to _CELL_SIZE for
    the debug grid, compose and upload.

    The shot archive contains only the selected representative frames —
    no need to filter by frame range as the old window-based approach did.
    """
    shot_id       = shot["shot_id"]
    frames_key    = shot.get("frames_key")
    sync_pop_info = shot.get("sync_pop")
    flash_info    = shot.get("flash_frame")
    t0 = time.monotonic()

    try:
        if not frames_key:
            return {
                "shot_id":     shot_id,
                "status":      "skipped_no_frames_key",
                "elapsed_sec": round(time.monotonic() - t0, 2),
            }

        # Download and extract per-shot tar.gz
        raw = get_binary(out_bucket, frames_key)

        shot_tmp = os.path.join(tmp_dir, f"shot_{shot_id:06d}")
        os.makedirs(shot_tmp, exist_ok=True)

        frame_paths: list[str] = []
        with tarfile.open(fileobj=__import__("io").BytesIO(raw),
                          mode="r:gz") as tar:
            for member in sorted(tar.getmembers(),
                                 key=lambda m: m.name):
                name = os.path.basename(member.name)
                if not name.startswith("frame_"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                dst = os.path.join(shot_tmp, name)
                with open(dst, "wb") as out:
                    out.write(f.read())
                frame_paths.append(dst)

        if not frame_paths:
            return {
                "shot_id":     shot_id,
                "status":      "skipped_empty_archive",
                "elapsed_sec": round(time.monotonic() - t0, 2),
            }

        # Determine highlighted indices (sync-pop / flash-frame anomaly)
        special_indices: set[int] = set()
        for info in (sync_pop_info, flash_info):
            if not info:
                continue
            sp_frame = info.get("frame")
            if sp_frame is None:
                continue
            sp_name = f"frame_{sp_frame:010d}.jpg"
            for idx, p in enumerate(frame_paths):
                if os.path.basename(p) == sp_name:
                    special_indices.add(idx)
                    break

        grid_img  = _compose_grid(frame_paths, special_indices)
        grid_path = os.path.join(shot_tmp, "grid.jpg")
        grid_img.save(grid_path, "JPEG", quality=_JPEG_QUALITY)
        upload_file(out_bucket, out_key, grid_path, {"ContentType": "image/jpeg"})

        elapsed = round(time.monotonic() - t0, 2)
        return {
            "shot_id":      shot_id,
            "status":       "ok",
            "s3_key":       out_key,
            "n_frames":     len(frame_paths),
            "has_sync_pop": sync_pop_info is not None,
            "has_flash_frame": flash_info is not None,
            "elapsed_sec":  elapsed,
        }

    except Exception as exc:
        return {
            "shot_id":     shot_id,
            "status":      f"error: {type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.monotonic() - t0, 2),
        }
    finally:
        shot_tmp = os.path.join(tmp_dir, f"shot_{shot_id:06d}")
        if os.path.isdir(shot_tmp):
            shutil.rmtree(shot_tmp, ignore_errors=True)


# =============================================================================
# Lambda handler
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    """
    StateStoreDebugShotImages — simplified after merge state changes.

    Reads per-shot frames tar.gz archives (written by StateMergeAndReconcileShots)
    which contain only the selected representative frames at proxy native
    resolution (native AR, e.g. 512x288 for 16:9).
    Downscales each frame to _CELL_SIZE (112px) for the debug grid image.

    No longer iterates window archives — each shot is self-contained.
    """
    log.info("StateStoreDebugShotImages event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()
    inv_uuid = str(uuid.uuid4())[:8]
    tmp_base = f"/tmp/debug_{inv_uuid}"

    try:
        data         = event.get("data", {})
        proxy_bucket = data["proxy_bucket"]
        proxy_prefix = data["proxy_prefix"]
        shots_key    = data["shots_key"]

        log.info("Loading shots: s3://%s/%s", proxy_bucket, shots_key)
        shots_doc = get_json(proxy_bucket, shots_key)
        shots     = shots_doc["shots"]
        n_shots   = len(shots)
        log.info("Loaded %d shots", n_shots)

        shots_with_frames = [s for s in shots if s.get("frames_key")]
        shots_without     = n_shots - len(shots_with_frames)
        if shots_without:
            log.warning("%d shot(s) missing frames_key", shots_without)
        if not shots_with_frames:
            raise ValueError(
                "No shots have frames_key — cannot generate debug images. "
                "Ensure StateMergeAndReconcileShots ran successfully."
            )

        os.makedirs(tmp_base, exist_ok=True)
        tmp_shots_dir = os.path.join(tmp_base, "shots")
        os.makedirs(tmp_shots_dir, exist_ok=True)

        tasks = [
            (
                shot,
                tmp_shots_dir,
                proxy_bucket,
                f"{proxy_prefix}/shot_images/shot_{shot['shot_id']:06d}.jpg",
            )
            for shot in shots
        ]

        n_workers = min(6, len(tasks))
        log.info("Processing %d shots with %d workers", len(tasks), n_workers)
        t_proc = time.monotonic()

        results: list[dict] = []
        errors   = 0
        skipped  = 0
        syncpops = 0
        flashes = 0

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_shot_from_frames, *task):
                    task[0]["shot_id"]
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result["status"] == "ok":
                    if result.get("has_sync_pop"):
                        syncpops += 1
                    if result.get("has_flash_frame"):
                        flashes += 1
                elif result["status"].startswith("skipped"):
                    skipped += 1
                else:
                    errors += 1
                    log.warning(
                        "Shot %d failed: %s",
                        result["shot_id"], result["status"],
                    )

        proc_ms = round((time.monotonic() - t_proc) * 1000, 1)
        elapsed = round(time.monotonic() - t0, 2)
        ok = len(results) - errors - skipped

        log.info(
            "Debug images complete: %d ok  %d sync-pop  %d flash-frame  "
            "%d skipped  %d errors  proc=%.0fms  elapsed=%.1fs",
            ok, syncpops, flashes, skipped, errors, proc_ms, elapsed,
        )

        return {
            **event,
            "data": {
                **data,
                "debug_images_prefix": f"{proxy_prefix}/shot_images/",
                "debug_n_shots":       n_shots,
                "debug_n_ok":          ok,
                "debug_n_sync_pop":    syncpops,
                "debug_n_flash_frame": flashes,
                "debug_n_skipped":     skipped,
                "debug_n_errors":      errors,
                "debug_elapsed_sec":   elapsed,
            },
        }

    except Exception as exc:
        log.error(
            "[ERR] StateStoreDebugShotImages — %s: %s",
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        log.info("Cleaned up %s", tmp_base)
