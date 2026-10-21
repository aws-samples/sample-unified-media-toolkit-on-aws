# dynamic_chunking_handler.py
"""
Step Function State: Dynamic chunking for frame embedding
----------------------------------------------------------
Two responsibilities:

  1. RECONCILE transcode iterator outputs
  2. CHUNK into 6-minute embedding windows with 30s overlap

Windows carry frames_to_shots_cfg so that each Map Lambda invocation
has everything it needs for inference + shot detection.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from lib.input_schema import (
    DEFAULT_OVERLAP_SECS,
    DEFAULT_WINDOW_SECS,
)
from lib.misc_utils import (
    round_half_up,
    fps_from_str,
)

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


# =============================================================================
# Reconciliation  (unchanged)
# =============================================================================

def _reconcile_segments(segments, fps):
    completed = [s for s in segments if s.get("status") == "COMPLETED"]
    empty = [s for s in segments if s.get("status") == "EMPTY"]
    other = [s for s in segments
             if s.get("status") not in ("COMPLETED", "EMPTY")]
    if other:
        log.warning("Unexpected status: %s",
                    [(s.get("segment_id"), s.get("status")) for s in other])
    log.info("Segments: total=%d  completed=%d  empty=%d",
             len(segments), len(completed), len(empty))
    if not completed:
        raise ValueError("No COMPLETED segments")

    completed.sort(key=lambda s: s["segment_id"])

    gaps = []
    for i in range(len(completed) - 1):
        cur, nxt = completed[i], completed[i + 1]
        cur_end = cur.get("source_end_frame")
        nxt_start = nxt.get("source_start_frame")
        if cur_end is None or nxt_start is None:
            continue
        delta = nxt_start - (cur_end + 1)
        if delta > 0:
            gaps.append({
                "between_segments": [cur["segment_id"], nxt["segment_id"]],
                "gap_frames": delta,
                "gap_sec": round(delta / fps, 6) if fps > 0 else None,
            })
        elif delta < 0:
            log.info("Frame overlap seg %d→%d: %d frames",
                     cur["segment_id"], nxt["segment_id"], abs(delta))

    last = completed[-1]
    first = completed[0]
    actual_end = last.get("source_end_frame", 0)
    actual_start = first.get("source_start_frame", 0)
    actual_count = actual_end - actual_start + 1
    actual_dur = round(actual_count / fps, 6) if fps > 0 else 0.0

    return completed, {
        "completed_segment_count": len(completed),
        "empty_segment_count":     len(empty),
        "actual_frame_count":      actual_count,
        "actual_duration_sec":     actual_dur,
        "actual_start_frame":      actual_start,
        "actual_end_frame":        actual_end,
        "contiguity_gaps":         gaps,
    }


# =============================================================================
# Window builder
# =============================================================================

def _build_windows(segment, window_secs, overlap_secs,
                   window_id_offset, cfg, dest_prefix):
    segment_id = segment["segment_id"]
    segment_name = segment["segment_name"]
    proxy_bucket = segment["proxy_bucket"]
    proxy_key = segment["proxy_key"]
    pts_key = segment.get("pts_key")
    frame_rate = segment["frame_rate"]
    fps = float(segment.get("frame_rate_fps", 0.0))
    time_base = segment.get("time_base", "")
    is_interlaced = bool(segment.get("is_interlaced", False))
    sar_str = segment.get("sample_aspect_ratio", "1:1")
    tc_first = segment.get("timecode_first_frame")
    drop_frame = bool(segment.get("drop_frame", False))
    seg_src_start = int(segment.get("source_start_frame", 0))
    seg_src_end = int(segment.get("source_end_frame", 0))

    if not fps:
        fps = fps_from_str(frame_rate)

    proxy_duration = float(
        segment.get("proxy_duration_sec")
        or segment.get("duration") or 0.0)
    if proxy_duration <= 0:
        return []

    windows = []
    local_id = 0
    pos = 0.0

    while pos < proxy_duration:
        content_end = min(pos + window_secs, proxy_duration)
        end_overlap = min(content_end + overlap_secs, proxy_duration)

        win_name = f"seg{segment_id:04d}_win{local_id:04d}"
        win_id = window_id_offset + local_id
        win_src_start = seg_src_start + round_half_up(pos * fps)
        win_src_end = min(
            seg_src_start + round_half_up(end_overlap * fps) - 1,
            seg_src_end)

        windows.append({
            "window_id":            win_id,
            "window_name":          win_name,
            "segment_id":           segment_id,
            "segment_name":         segment_name,
            "proxy_bucket":         proxy_bucket,
            "proxy_key":            proxy_key,
            "pts_key":              pts_key,
            "window_start":         round(pos, 6),
            "window_end":           round(end_overlap, 6),
            "content_end":          round(content_end, 6),
            "window_duration":      round(end_overlap - pos, 6),
            "overlap_secs":         round(end_overlap - content_end, 6),
            "source_start_frame":   win_src_start,
            "source_end_frame":     win_src_end,
            "frame_rate":           frame_rate,
            "frame_rate_fps":       fps,
            "time_base":            time_base,
            "is_interlaced":        is_interlaced,
            "sample_aspect_ratio":  sar_str,
            "timecode_first_frame": tc_first,
            "drop_frame":           drop_frame,
            "frames_to_shots_cfg":  cfg,
            "dest_prefix":          dest_prefix,
        })
        pos += window_secs
        local_id += 1

    return windows


# =============================================================================
# Lambda handler
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateDynamicChunkingForFrameEmbedding event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        data = event.get("data", {})
        segments = data.get("iterators", [])
        video = data.get("video", {})

        if not segments:
            raise ValueError("data.iterators is empty")

        fps_str = (video.get("frame_rate")
                   or (segments[0].get("frame_rate") if segments else None)
                   or "0/0")
        fps = fps_from_str(fps_str)
        if fps <= 0:
            raise ValueError(f"Cannot determine fps: {fps_str!r}")

        cfg = data.get("frames_to_shots_cfg", {})

        log.info("fps=%.4f  segments=%d  window=%ds  overlap=%.1fs",
                 fps, len(segments), DEFAULT_WINDOW_SECS, DEFAULT_OVERLAP_SECS)

        completed, reconcile = _reconcile_segments(segments, fps)

        dest_prefix = data["proxy_prefix"]
        all_windows = []
        offset = 0
        for seg in completed:
            wins = _build_windows(
                seg, DEFAULT_WINDOW_SECS, DEFAULT_OVERLAP_SECS,
                offset, cfg, dest_prefix)
            all_windows.extend(wins)
            offset += len(wins)

        updated_video = {
            **video,
            "actual_frame_count":       reconcile["actual_frame_count"],
            "actual_duration_sec":      reconcile["actual_duration_sec"],
            "actual_start_frame":       reconcile["actual_start_frame"],
            "actual_end_frame":         reconcile["actual_end_frame"],
            "completed_segment_count":  reconcile["completed_segment_count"],
            "empty_segment_count":      reconcile["empty_segment_count"],
            "contiguity_gaps":          reconcile["contiguity_gaps"],
        }

        elapsed = round(time.monotonic() - t0, 2)
        log.info("Done: windows=%d  frames=%d  dur=%.3fs  elapsed=%.2fs",
                 len(all_windows), reconcile["actual_frame_count"],
                 reconcile["actual_duration_sec"], elapsed)

        return {
            **event,
            "data": {**data, "video": updated_video,
                     "iterators": all_windows},
        }

    except Exception as exc:
        log.error("[ERR] %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise
