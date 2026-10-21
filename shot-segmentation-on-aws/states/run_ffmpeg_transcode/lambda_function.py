# transcode_handler.py
"""
Step Function State: Run FFmpeg transcode  (Map state — one Lambda per segment)
--------------------------------------------------------------------------------
Transcodes one fixed-size time window from the source video to an
all-intra OpenH264 proxy MP4 fitting within 512×512 at the source aspect
ratio (no letterbox/pillarbox padding — e.g. 16:9 → 512×288), captures
exact source frame numbers via showinfo, and writes a frame index (pts)
JSON to S3.

Transcode path
-------------------
Fast seek snaps to a keyframe that may be before start_time. -copyts
preserves source PTS into the filter graph so showinfo reports real
source timestamps. A tee filter_complex splits the decoded stream into:
- showinfo branch  → nullsink (no encode, PTS captured to FFREPORT)
- encode branch    → setpts=PTS-STARTPTS resets PTS to 0 for MP4 muxer,
                        then all encode filters → output

Single decode pass. No MPEG-TS intermediate file needed.

ffmpeg -copyts -ss {start} -t {duration} -i source \
    -filter_complex \
    "[0:v]split=2[si][enc]; \
        [si]showinfo,nullsink; \
        [enc]setpts=PTS-STARTPTS,yadif,...,setsar=1/1[out]" \
    -map "[out]" \
    -c:v libopenh264 -g {gop} -an -movflags +faststart -v error \
    proxy.mp4

FFREPORT log
------------
FFREPORT=file={path}:level=32 captures showinfo output independently of
-v error on the console. Level 32 = AV_LOG_INFO.

showinfo config line (one per run, before first frame):
  [Parsed_showinfo_0 @ 0x...] config in time_base: 1/25, frame_rate: 25/1

showinfo frame line:
  [Parsed_showinfo_0 @ 0x...] n:   0 pts:    250 pts_time:10.000000 ...

Frame number computation (exact integer arithmetic, no float drift):
  frame_number = pts × Fraction(time_base) × Fraction(frame_rate)

  All-intra:  pts starts at 0, frame_number = 0, 1, 2 ...
              absolute = source_start_frame + frame_number
  IPB:        pts is source PTS (e.g. 250 for 10s@25fps)
              frame_number = absolute source frame directly

Empty segment detection
-----------------------
If ffmpeg produces 0 bytes of output (seek beyond actual EOF — possible when
reported duration > actual duration e.g. Sony eVTR MXF), the handler returns
status="EMPTY". This is not an error — the reconcile state filters EMPTY
segments out.

PTS JSON output
---------------
Always written for both paths. DINOv2 handler uses this for all frame
number and timecode computation.

{
    "segment_id":          0,
    "segment_name":        "segment_0000",
    "start_time":          0.0,
    "end_time":            1230.0,
    "time_base":           "1/25",        <- from showinfo config line
    "frame_rate":          "25/1",        <- from showinfo config line
    "source_start_frame":  0,             <- absolute frame number of first frame
    "source_end_frame":    30749,         <- absolute frame number of last frame
    "frame_count":         30750,
    "source_frame_numbers": [0, 1, 2, ...]  <- absolute, parallel to proxy frames
}

Expected input  (from Map ItemSelector)
----------------------------------------
{
    "operation": "StateRunFFmpegTranscode",
    "item_id":    0,
    "item_data": {
        "segment_id":          0,
        "segment_name":        "segment_0000",
        "source_bucket":       "ingest-bucket",
        "source_key":          "path/to/source.mxf",
        "proxy_bucket":        "proxy-bucket",
        "proxy_key":           "prefix/proxy/segment_0000.mp4",
        "start_time":          0.0,
        "content_end_time":    1200.0,
        "end_time":            1230.0,
        "duration":            1230.0,
        "overlap_secs":        30.0,
        "source_start_frame":  0,
        "frame_rate":          "25/1",
        "frame_rate_fps":      25.0,
        "time_base":           "1/25",
        "is_interlaced":       false,
        "sample_aspect_ratio": "152:135",
        "timecode_first_frame": "09:59:43:14",
        "drop_frame":          false,
        "input_cropping": {    <- optional
            "crop_x": 0,
            "crop_y": 32
        }
    }
}

Output
------
COMPLETED:
{
    ...all item_data fields...,
    "status":               "COMPLETED",
    "proxy_frame_count":    30750,
    "proxy_duration_sec":   1230.0,
    "proxy_fps":            25.0,
    "proxy_size_mb":        42.3,
    "source_start_frame":   0,
    "source_end_frame":     30749,
    "source_frame_count":   30750,
    "time_base":            "1/25",
    "pts_key":              "prefix/proxy/segment_0000_pts.json",
    "metrics_key":          "prefix/proxy/segment_0000.json",
    "timing_ms":            {...},
    "elapsed_sec":          46.25
}

EMPTY:
{
    ...all item_data fields...,
    "status":      "EMPTY",
    "elapsed_sec": 1.2
}
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from fractions import Fraction
from pathlib import Path
from lib.s3_utils import (
  presign,
  put_json,
  upload_file,
)
from lib.misc_utils import (
    round_half_up,
    fps_from_str,
    fmt_time,
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

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_W = 512 # proxy bounding box; output keeps source AR within this (e.g. 16:9 → 512×288)
TARGET_H = TARGET_W
PRESIGN_EXPIRY = 21600  # 6 hours — covers long transcode jobs
FFREPORT_LEVEL = 32     # AV_LOG_INFO — captures showinfo output

# showinfo config line:
# [Parsed_showinfo_0 @ 0x...] config in time_base: 1/25, frame_rate: 25/1
_SHOWINFO_CONFIG_RE = re.compile(
    r"config in time_base:\s*(\S+),\s*frame_rate:\s*(\S+)"
)

# showinfo frame line:
# [Parsed_showinfo_0 @ 0x...] n:   0 pts:    250 pts_time:10.000000 ...
_SHOWINFO_FRAME_RE = re.compile(
    r"Parsed_showinfo.*\bn:\s*\d+\s+pts:\s*(\d+)\s+pts_time:(\S+)"
)


# =============================================================================
# Filter chain builders
# =============================================================================

def _encode_filters(
    is_interlaced:  bool,
    sar_str:        str,
    input_cropping: dict | None,
) -> list[str]:
    """
    Return the ordered list of encode filters (no showinfo).
    Used by both all-intra and IPB paths.

    Order:
      1. yadif=mode=0      — deinterlace (if interlaced)
      2. scale=iw*sar:ih   — SAR unsqueeze (if non-square pixels)
      3. crop              — VBI / border crop (if configured)
      4. scale             — fit within TARGET_W × TARGET_H at native
                             aspect ratio (no pad — the proxy keeps the
                             source AR so image-property computation
                             downstream sees no letterbox bars;
                             force_divisible_by=2 keeps H.264 legal)
      5. setsar=1/1        — enforce square pixels on output
    """
    filters = []

    if is_interlaced:
        filters.append("yadif=mode=0")

    sar_is_square = sar_str in ("1:1", "0:1", "", "1/1")
    if not sar_is_square:
        filters.append("scale=iw*sar:ih:flags=lanczos")

    if input_cropping:
        cx = int(input_cropping.get("crop_x", 0))
        cy = int(input_cropping.get("crop_y", 0))
        if cx > 0 or cy > 0:
            filters.append(f"crop=iw-{cx*2}:ih-{cy}:{cx}:{cy}")

    filters.append(
        f"scale={TARGET_W}:{TARGET_H}"
        f":force_original_aspect_ratio=decrease"
        f":force_divisible_by=2"
        f":flags=lanczos"
    )
    filters.append("setsar=1/1")

    return filters


def _build_filter_complex(
    is_interlaced:  bool,
    sar_str:        str,
    input_cropping: dict | None,
) -> str:
    """
    Tee filter_complex for IPB path with -copyts.

    Splits decoded stream into two branches:
      [si]  → showinfo,nullsink     — captures real source PTS, no encode
      [enc] → setpts=PTS-STARTPTS,  — resets PTS to 0 for MP4 muxer,
               encode filters,       then all encode filters
               → [out]

    showinfo on [si] sees source PTS preserved by -copyts.
    setpts=PTS-STARTPTS on [enc] makes -t and the MP4 muxer work correctly.
    """
    encode = _encode_filters(is_interlaced, sar_str, input_cropping)
    encode_chain = ",".join(["setpts=PTS-STARTPTS"] + encode)

    fc = (
        f"[0:v]split=2[si][enc];"
        f"[si]showinfo,nullsink;"
        f"[enc]{encode_chain}[out]"
    )
    log.info("filter_complex (IPB): %s", fc)
    return fc


# =============================================================================
# FFREPORT log parser
# =============================================================================

def _parse_showinfo_log(
    log_path:           str,
    frame_rate_hint:    str,
    time_base_hint:     str,
) -> tuple[str, str, list[int], list[float]]:
    """
    Parse FFREPORT log to extract time_base, frame_rate, absolute source
    frame numbers, and pts_time floats from showinfo output.

    Single computation path for all content types:
      frame_number = pts × Fraction(time_base) × Fraction(frame_rate)

    -copyts preserves source PTS regardless of codec type, so this
    integer arithmetic always yields the absolute source frame number
    directly. source_start_frame is no longer used for computation —
    it is retained in the signature for pts_doc metadata only.

    Returns (time_base, frame_rate, source_frame_numbers, pts_times).
    Falls back to hint values if config line not found.
    """
    time_base = time_base_hint
    frame_rate = frame_rate_hint

    if not os.path.exists(log_path):
        log.warning("FFREPORT log not found: %s", log_path)
        return time_base, frame_rate, [], []

    pts_list:  list[int] = []
    pts_times: list[float] = []

    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if "Parsed_showinfo" not in line:
                    continue

                m_cfg = _SHOWINFO_CONFIG_RE.search(line)
                if m_cfg:
                    time_base = m_cfg.group(1)
                    frame_rate = m_cfg.group(2)
                    log.info(
                        "showinfo config: time_base=%s  frame_rate=%s",
                        time_base, frame_rate,
                    )
                    continue

                m_frm = _SHOWINFO_FRAME_RE.search(line)
                if m_frm:
                    try:
                        pts_list.append(int(m_frm.group(1)))
                        pts_times.append(float(m_frm.group(2)))
                    except ValueError:
                        continue

    except OSError as exc:
        log.warning("Failed to read FFREPORT log: %s", exc)
        return time_base, frame_rate, [], []

    if not pts_list:
        log.warning("showinfo: no frame lines found in FFREPORT log")
        return time_base, frame_rate, [], []

    # ── Compute absolute source frame numbers ──────────────────────────────
    # -copyts preserves source PTS for all content types.
    # frame_number = pts × Fraction(time_base) × Fraction(frame_rate)
    # is always the absolute source frame number directly.
    try:
        tb = Fraction(time_base)
        fps = Fraction(frame_rate)
        ticks_per_frame = tb * fps

        # Round to NEAREST, never truncate (README.md §9.8). Containers store
        # integer PTS on their own timebase, so when ticks_per_frame is
        # fractional (90000 ticks @ 24000/1001 fps = 3753.75) every 4th
        # frame's stored PTS sits just BELOW the exact frame boundary
        # (frame 3 = 11261.25 ticks, stored 11261 -> fn = 2.99993...).
        # int() truncation knocked those a full frame down — duplicating one
        # number and skipping the next. Measured on a real 720p 23.976 mp4:
        # 3597 duplicates + 3596 holes across 17238 frames (~21%), and any
        # cut landing on such a position was INVISIBLE to boundary detection
        # because no gap==1 sample pair can exist there.
        source_frame_numbers: list[int] = []
        non_integer = 0
        for pts in pts_list:
            fn = Fraction(pts) * ticks_per_frame
            if fn.denominator != 1:
                non_integer += 1
            source_frame_numbers.append(int(fn + Fraction(1, 2)))
        if non_integer:
            log.info(
                "showinfo: %d/%d pts values are not exact frame multiples "
                "(fractional ticks_per_frame=%s) — rounded to nearest",
                non_integer, len(pts_list), ticks_per_frame,
            )

        # Contiguity audit: CFR input must reconstruct to strictly
        # +1-contiguous numbers. Duplicates/holes here mean a mapping defect
        # (the §9.8 failure class) or genuinely variable frame timing —
        # either way boundary detection is blind at those positions, which
        # must be visible in the logs, not silent.
        dups = sum(1 for i in range(1, len(source_frame_numbers))
                   if source_frame_numbers[i] == source_frame_numbers[i - 1])
        holes = sum(1 for i in range(1, len(source_frame_numbers))
                    if source_frame_numbers[i] - source_frame_numbers[i - 1] > 1)
        if dups or holes:
            log.warning(
                "showinfo: source_frame_numbers NOT contiguous — "
                "%d duplicates, %d holes across %d frames; cuts at those "
                "positions are invisible to boundary detection (README.md §9.8)",
                dups, holes, len(source_frame_numbers),
            )

    except (ValueError, ZeroDivisionError) as exc:
        log.warning("Frame number computation failed: %s", exc)
        return time_base, frame_rate, [], []

    log.info(
        "showinfo: %d frames  time_base=%s  frame_rate=%s  "
        "first=%d (pts_time=%.6f)  last=%d (pts_time=%.6f)",
        len(source_frame_numbers),
        time_base, frame_rate,
        source_frame_numbers[0],  pts_times[0],
        source_frame_numbers[-1], pts_times[-1],
    )
    return time_base, frame_rate, source_frame_numbers, pts_times

# =============================================================================
# FFmpeg transcode
# =============================================================================


def _run_ffmpeg(
    source_url:     str,
    output_path:    str,
    ffreport_path:  str,
    start_time:     float,
    duration:       float,
    frame_rate:     str,
    is_interlaced:  bool,
    sar_str:        str,
    input_cropping: dict | None,
) -> tuple[float, bool]:
    """
    Transcode one segment. Returns (elapsed_sec, is_empty).

    Single transcode path for all content types:
      -copyts preserves source PTS through the filter graph.
      Tee filter_complex:
        [si] showinfo,nullsink  — captures real source PTS per frame
        [enc] setpts=PTS-STARTPTS,{encode_filters}[out]
                                — resets PTS to 0 for MP4 muxer

    This works identically for all-intra and IPB content:
      - showinfo always reports actual source PTS (preserved by -copyts)
      - frame_number = pts × Fraction(time_base) × Fraction(frame_rate)
        is always the absolute source frame number regardless of codec
      - setpts=PTS-STARTPTS ensures proxy MP4 always starts at PTS=0

    is_empty=True if ffmpeg produced 0 bytes (seek beyond actual EOF).
    Raises RuntimeError on genuine ffmpeg failure.
    """
    gop_size = max(1, round_half_up(fps_from_str(frame_rate)))
    fc = _build_filter_complex(is_interlaced, sar_str, input_cropping)

    cmd = ["ffmpeg", "-y", "-copyts"]
    if start_time > 0.0:
        cmd += ["-ss", fmt_time(start_time)]
    cmd += [
        "-t",  str(round(duration, 6)),
        "-i",  source_url,
        "-filter_complex", fc,
        "-map", "[out]",
        "-c:v", "libopenh264",
        "-b:v", "1000000",
        "-g",   str(gop_size),
        "-an",
        "-movflags", "+faststart",
        "-v", "error",
        output_path,
    ]

    env = os.environ.copy()
    env["FFREPORT"] = f"file={ffreport_path}:level={FFREPORT_LEVEL}"

    log.info(
        "ffmpeg: start=%.3fs  duration=%.3fs  "
        "interlaced=%s  sar=%s  gop=%d",
        start_time, duration, is_interlaced, sar_str, gop_size,
    )
    log.debug("ffmpeg cmd: %s", " ".join(cmd))

    t0 = time.monotonic()
    result = subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        cmd,
        capture_output=True,
        text=True,
        timeout=840,
        env=env,
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        output_exists = (
            os.path.exists(output_path)
            and os.path.getsize(output_path) > 0
        )
        if output_exists:
            log.warning(
                "ffmpeg rc=%d but output exists (%.1f KB) — "
                "treating as partial",
                result.returncode,
                os.path.getsize(output_path) / 1024,
            )
        else:
            stderr_tail = "\n".join(result.stderr.splitlines()[-30:])
            raise RuntimeError(
                f"ffmpeg failed (rc={result.returncode}) "
                f"start={start_time:.3f}s\n{stderr_tail}"
            )

    log.info("ffmpeg complete: %.1fs", elapsed)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        log.info(
            "ffmpeg produced no output at start=%.3fs — "
            "seek likely beyond actual EOF",
            start_time,
        )
        return elapsed, True

    return elapsed, False


# =============================================================================
# Proxy verification
# =============================================================================

def _verify_proxy(bucket: str, key: str) -> tuple[int, float, float, int, int]:
    """
    Lightweight ffprobe on the uploaded proxy — confirms readability
    and returns (frame_count, duration_sec, fps, width, height).
    No expected frame count check (we no longer have one from probe).
    """
    url = presign(bucket, key)
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_streams",
        "-show_entries", "stream=nb_frames,duration,r_frame_rate,avg_frame_rate,width,height",
        "-print_format", "json",
        url,
    ]
    log.info("ffprobe verify: s3://%s/%s", bucket, key)
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe verify failed (rc={result.returncode}): {result.stderr[:300]}"
        )

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe verify: no video stream in proxy")

    v = streams[0]

    dur_str = v.get("duration")
    actual_duration = float(dur_str) if dur_str and dur_str != "N/A" else 0.0

    fps_str = v.get("r_frame_rate") or v.get("avg_frame_rate") or "0/0"
    actual_fps = fps_from_str(fps_str)

    nb_str = v.get("nb_frames")
    if nb_str and nb_str != "N/A":
        actual_frames = int(nb_str)
    else:
        actual_frames = (
            round_half_up(actual_duration * actual_fps)
            if actual_fps > 0 else 0
        )

    proxy_w = int(v.get("width", 0) or 0)
    proxy_h = int(v.get("height", 0) or 0)

    log.info(
        "Proxy verify: frames=%d  duration=%.3fs  fps=%.3f  %dx%d  elapsed=%.2fs",
        actual_frames, actual_duration, actual_fps, proxy_w, proxy_h, elapsed,
    )
    return actual_frames, actual_duration, actual_fps, proxy_w, proxy_h


# =============================================================================
# Lambda handler
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateRunFFmpegTranscode event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    tmp_dir = Path(tempfile.gettempdir()) / f"transcode_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log.info("Temp dir: %s", tmp_dir)

    try:
        # ── Unpack ────────────────────────────────────────────────────────────
        item = event.get("item_data", event)

        required = {
            "segment_id", "segment_name",
            "source_bucket", "source_key",
            "proxy_bucket", "proxy_key",
            "start_time", "end_time", "duration",
            "frame_rate", "time_base",
            "source_start_frame",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"Missing item_data fields: {missing}")

        segment_id = int(item["segment_id"])
        segment_name = item["segment_name"]
        source_bucket = item["source_bucket"]
        source_key = item["source_key"].strip("/")
        proxy_bucket = item["proxy_bucket"]
        proxy_key = item["proxy_key"].strip("/")
        start_time = float(item["start_time"])
        end_time = float(item["end_time"])
        duration = float(item["duration"])
        frame_rate = item["frame_rate"]
        time_base = item["time_base"]
        source_start_frame = int(item["source_start_frame"])
        is_interlaced = bool(item.get("is_interlaced", False))
        input_cropping = item.get("input_cropping") or None
        sar_str = item.get("sample_aspect_ratio", "1:1") or "1:1"

        fps = fps_from_str(frame_rate)
        if fps <= 0:
            raise ValueError(f"Invalid frame_rate: {frame_rate!r}")

        proxy_key_base = proxy_key.rsplit(".", 1)[0]
        metrics_key = f"{proxy_key_base}.json"
        pts_key = f"{proxy_key_base}_pts.json"

        log.info(
            "Segment %d (%s): s3://%s/%s  [%.3fs → %.3fs]  duration=%.3fs  "
            "fps=%.3f  time_base=%s  interlaced=%s  sar=%s  "
            "source_start_frame=%d",
            segment_id, segment_name,
            source_bucket, source_key,
            start_time, end_time, duration,
            fps, time_base, is_interlaced, sar_str,
            source_start_frame,
        )

        source_url = presign(source_bucket, source_key)
        local_output = str(tmp_dir / f"{segment_name}.mp4")
        ffreport_path = str(tmp_dir / f"{segment_name}_ffreport.log")

        # ── FFmpeg ────────────────────────────────────────────────────────────
        ffmpeg_elapsed, is_empty = _run_ffmpeg(
            source_url=source_url,
            output_path=local_output,
            ffreport_path=ffreport_path,
            start_time=start_time,
            duration=duration,
            frame_rate=frame_rate,
            is_interlaced=is_interlaced,
            sar_str=sar_str,
            input_cropping=input_cropping,
        )
        ffmpeg_ms = round(ffmpeg_elapsed * 1_000, 1)

        # ── Empty segment ─────────────────────────────────────────────────────
        if is_empty:
            elapsed_sec = round(time.monotonic() - t0, 2)
            log.info(
                "EMPTY segment=%d (%s)  start=%.3fs — beyond actual EOF",
                segment_id, segment_name, start_time,
            )
            return {
                **item,
                "status":      "EMPTY",
                "elapsed_sec": elapsed_sec,
            }

        # ── Parse showinfo log ────────────────────────────────────────────────
        actual_time_base, actual_frame_rate, source_frame_numbers, pts_times = (
            _parse_showinfo_log(
                log_path=ffreport_path,
                frame_rate_hint=frame_rate,
                time_base_hint=time_base,
            )
        )

        if not source_frame_numbers:
            log.warning(
                "showinfo produced no frame numbers for segment %d — "
                "pts_key will be empty",
                segment_id,
            )

        frame_count = len(source_frame_numbers)
        src_start_frame = source_frame_numbers[0] if source_frame_numbers else source_start_frame
        src_end_frame = source_frame_numbers[-1] if source_frame_numbers else source_start_frame

        log.info(
            "Source frames: count=%d  first=%d  last=%d  "
            "time_base=%s  frame_rate=%s",
            frame_count, src_start_frame, src_end_frame,
            actual_time_base, actual_frame_rate,
        )

        # ── Upload proxy ──────────────────────────────────────────────────────
        proxy_size_mb = os.path.getsize(local_output) / (1024 ** 2)
        log.info("Proxy: %.1f MB", proxy_size_mb)

        t_upload = time.monotonic()
        upload_file(proxy_bucket, proxy_key, local_output)
        upload_ms = round((time.monotonic() - t_upload) * 1_000, 1)

        # ── Upload PTS JSON ───────────────────────────────────────────────────
        pts_doc = {
            "segment_id":           segment_id,
            "segment_name":         segment_name,
            "start_time":           start_time,
            "end_time":             end_time,
            "time_base":            actual_time_base,
            "frame_rate":           actual_frame_rate,
            "source_start_frame":   src_start_frame,
            "source_end_frame":     src_end_frame,
            "frame_count":          frame_count,
            # [int] exact, for computation
            "source_frame_numbers": source_frame_numbers,
            # [float] debug only
            "pts_times":            pts_times,
        }
        put_json(proxy_bucket, pts_key, pts_doc)
        log.info("PTS JSON written: s3://%s/%s", proxy_bucket, pts_key)

        # ── Verify proxy ──────────────────────────────────────────────────────
        t_verify = time.monotonic()
        actual_frames, actual_duration, actual_fps, proxy_w, proxy_h = \
            _verify_proxy(proxy_bucket, proxy_key)
        verify_ms = round((time.monotonic() - t_verify) * 1_000, 1)

        # Sanity cross-check — log warning only, never fail
        if source_frame_numbers and abs(frame_count - actual_frames) > 2:
            log.warning(
                "Frame count mismatch: showinfo=%d  proxy=%d — "
                "possible filter artefact (e.g. yadif field output)",
                frame_count, actual_frames,
            )

        # ── Timing / metrics ──────────────────────────────────────────────────
        elapsed_sec = round(time.monotonic() - t0, 2)
        elapsed_ms = round(elapsed_sec * 1_000, 1)
        timing_ms = {
            "ffmpeg_transcode": ffmpeg_ms,
            "s3_upload":        upload_ms,
            "ffprobe_verify":   verify_ms,
            "total":            elapsed_ms,
        }

        metrics_doc = {
            "segment_id":         segment_id,
            "segment_name":       segment_name,
            "source_bucket":      source_bucket,
            "source_key":         source_key,
            "proxy_bucket":       proxy_bucket,
            "proxy_key":          proxy_key,
            "start_time":         start_time,
            "end_time":           end_time,
            "duration":           duration,
            "frame_rate":         actual_frame_rate,
            "time_base":          actual_time_base,
            "is_interlaced":      is_interlaced,
            "sar":                sar_str,
            "input_cropping":     input_cropping,
            "proxy_size_mb":      round(proxy_size_mb, 2),
            "proxy_frame_count":  actual_frames,
            "proxy_duration_sec": round(actual_duration, 3),
            "proxy_fps":          actual_fps,
            "proxy_width":        proxy_w,
            "proxy_height":       proxy_h,
            "source_start_frame": src_start_frame,
            "source_end_frame":   src_end_frame,
            "source_frame_count": frame_count,
            "pts_key":            pts_key,
            "timing_ms":          timing_ms,
            "status":             "COMPLETED",
            "elapsed_sec":        elapsed_sec,
        }
        put_json(proxy_bucket, metrics_key, metrics_doc)
        log.info("Metrics written: s3://%s/%s", proxy_bucket, metrics_key)

        log.info(
            "DONE segment=%d (%s)  source_frames=%d  proxy_frames=%d  "
            "duration=%.1fs  size=%.1fMB  "
            "ffmpeg=%.1fs  upload=%.1fs  verify=%.1fs  total=%.1fs",
            segment_id, segment_name,
            frame_count, actual_frames,
            actual_duration, proxy_size_mb,
            ffmpeg_elapsed,
            upload_ms / 1_000,
            verify_ms / 1_000,
            elapsed_sec,
        )

        return {
            **item,
            "status":             "COMPLETED",
            "proxy_frame_count":  actual_frames,
            "proxy_duration_sec": round(actual_duration, 3),
            "proxy_fps":          actual_fps,
            "proxy_width":        proxy_w,
            "proxy_height":       proxy_h,
            "proxy_size_mb":      round(proxy_size_mb, 2),
            "source_start_frame": src_start_frame,
            "source_end_frame":   src_end_frame,
            "source_frame_count": frame_count,
            "time_base":          actual_time_base,
            "frame_rate":         actual_frame_rate,
            "pts_key":            pts_key,
            "metrics_key":        metrics_key,
            "timing_ms":          timing_ms,
            "elapsed_sec":        elapsed_sec,
        }

    except Exception as exc:
        log.error(
            "[ERR] StateRunFFmpegTranscode segment=%s — %s: %s",
            event.get("item_data", {}).get("segment_name", "?"),
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            log.info("Temp dir cleaned up: %s", tmp_dir)
