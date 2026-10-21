from __future__ import annotations

import json
import logging
import math
from fractions import Fraction

from lib.proc_utils import run
from lib.s3_utils import presign

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

FFPROBE_TIMEOUT = 840  # 14 min — large MXF may seek to footer index

MXF_FORMAT_NAMES = ("mxf", "mxf_d10", "mxf_opatom")


def is_mxf_format(format_name: str) -> bool:
    return any(m in (format_name or "").lower() for m in MXF_FORMAT_NAMES)


def run_ffprobe(bucket: str, key: str) -> dict:
    """
    ffprobe -show_format -show_streams -show_chapters -show_programs via presigned
    URL. Header-only, no decode.

    chapters and programs are measured free — they add no bytes to the read (both are
    bounded by probesize, same as format/streams) and ~52 bytes of JSON. Frame- and
    packet-level modes are deliberately absent: every one of them reads the entire
    file, including -skip_frame nokey.
    """
    url = presign(bucket, key)
    out = run(
        # -v error, not quiet: on failure the stderr text is the whole diagnosis
        # (S3 InvalidObjectState, 403, moov atom not found). quiet reduced it to
        # a bare "Command exited 1" with an empty body.
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", "-show_chapters", "-show_programs", url],
        timeout=FFPROBE_TIMEOUT,
    )
    return json.loads(out)


def scan_is_all_intra(bucket: str, key: str) -> bool:
    """
    Decode first 20s of video frames to determine if every frame is a keyframe.
    Used when codec heuristic returns None (ambiguous codec e.g. mpeg2video).
    Returns True on empty result or error — conservative fallback.
    """
    url = presign(bucket, key)
    try:
        out = run(
            ["ffprobe", "-v", "error",
             "-read_intervals", "%+20",
             "-select_streams", "v:0",
             "-show_frames",
             "-show_entries", "frame=key_frame",
             "-of", "json",
             url],
            timeout=FFPROBE_TIMEOUT,
        )
    except RuntimeError as exc:
        log.warning("scan_is_all_intra failed: %s — assuming all-intra", exc)
        return True

    frames = json.loads(out).get("frames", [])
    if not frames:
        log.warning("scan_is_all_intra: no frames in first 20s — assuming all-intra")
        return True

    result = all(f.get("key_frame", 0) == 1 for f in frames)
    log.info("scan_is_all_intra: %d frames sampled, all_intra=%s", len(frames), result)
    return result


def _safe_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def find_prior_keyframe(
    bucket: str,
    key: str,
    nominal_sec: float,
    window_sec: float = 20.0,
) -> dict:
    """
    Find the last keyframe at or before nominal_sec.
    Scans [max(0, nominal_sec - window_sec), nominal_sec] first.
    If no keyframe found, expands by 10 s and retries once.
    Raises RuntimeError if still nothing found.
    Returns the raw ffprobe frame dict — the caller uses
    best_effort_timestamp_time for both the -ss position and the frame number
    (via video_command.sec_to_frame). The per-frame `duration` is deliberately
    NOT requested: ffmpeg 6.1's MOV demuxer reports it as 1 for any media
    timescale below 1000, so anything derived from it is wrong on those files.

    best_effort_timestamp_time is ABSOLUTE, while nominal_sec is zero-based, so
    the `<= nominal_sec` filter assumes a zero-origin source. A start_time that
    is an INTEGER number of frame periods is harmless — the caller feeds the
    same value to both -ss and sec_to_frame, -ss is itself relative to
    start_time, and the errors cancel exactly (measured on a start_time=0.04
    file at 25 fps = 1 frame; every non-zero offset in the corpus is
    frame-aligned like this). A fractional-frame offset would NOT cancel
    exactly — round() vs ffmpeg's seek-discard cut can differ by one frame — so
    the pending start_time fix must subtract the stream start_time, never rely
    on cancellation. An offset larger than the search window makes the filter reject every
    keyframe, and this raises RuntimeError — loud, not a wrong frame. Live
    example: an MPEG-TS with start_time=49070 s in the ingest corpus.
    Caller must not invoke this for the first chunk (nominal_sec == 0); raises ValueError if so.
    """
    if nominal_sec == 0:
        raise ValueError("find_prior_keyframe must not be called for the first chunk (nominal_sec=0)")

    url = presign(bucket, key)

    # Scan from (nominal_sec - window_sec) for (window_sec + 5)s total,
    # covering 5s past nominal_sec so keyframes right at the boundary are included.
    # Filter to keyframes at or before nominal_sec after scanning.
    def _scan(start: float, duration: float) -> dict | None:
        interval = f"{start:.6f}%+{duration:.6f}"
        try:
            out = run(
                ["ffprobe", "-v", "error",
                 "-read_intervals", interval,
                 "-select_streams", "v:0",
                 "-show_frames",
                 "-show_entries",
                 "frame=key_frame,best_effort_timestamp_time,pict_type",
                 "-of", "json",
                 url],
                timeout=FFPROBE_TIMEOUT,
            )
        except RuntimeError as exc:
            log.warning("find_prior_keyframe scan failed: %s", exc)
            return None
        frames = json.loads(out).get("frames", [])
        keyframes = [
            f for f in frames
            if f.get("key_frame") == 1
            and _safe_float(f.get("best_effort_timestamp_time"), math.inf) <= nominal_sec
        ]
        return keyframes[-1] if keyframes else None

    start = max(0.0, nominal_sec - window_sec)
    result = _scan(start, window_sec + 5.0)
    if result:
        return result

    log.warning(
        "find_prior_keyframe: no keyframe in %.1fs window before %.3fs — expanding by 10s",
        window_sec, nominal_sec,
    )
    expanded_start = max(0.0, nominal_sec - window_sec - 10.0)
    result = _scan(expanded_start, window_sec + 15.0)
    if result:
        return result

    raise RuntimeError(
        f"find_prior_keyframe: no keyframe found within {window_sec + 10:.0f}s "
        f"before {nominal_sec:.3f}s"
    )


def fps_from_str(s: str) -> float:
    """Parse '25/1', '29.97', '30000/1001' → float. Returns 0.0 on failure."""
    if not s or s in ("0/0", "0:0"):
        return 0.0
    try:
        return float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0


def resolve_duration(ref: dict, fmt: dict) -> tuple[float, str]:
    """
    Resolve duration from ffprobe output.
      1. ffprobe duration_ts × time_base
      2. ffprobe fmt.duration (last resort)
    Raises ValueError if no source yields a positive duration.
    """
    ts = ref.get("duration_ts")
    tb = ref.get("time_base")
    if ts and ts not in ("N/A", "0") and tb:
        try:
            dur = float(Fraction(str(ts)) * Fraction(tb))
            if dur > 0:
                log.info("Duration from ffprobe duration_ts: %s × %s = %.3fs", ts, tb, dur)
                return round(dur, 6), "ffprobe_ts"
        except (ValueError, ZeroDivisionError):
            pass

    fd = fmt.get("duration")
    if fd and fd != "N/A":
        try:
            dur = float(fd)
            if dur > 0:
                log.warning("Duration from ffprobe fmt.duration=%.3fs — last resort", dur)
                return round(dur, 6), "ffprobe_fmt"
        except ValueError:
            pass

    raise ValueError("Could not determine duration from any available source")