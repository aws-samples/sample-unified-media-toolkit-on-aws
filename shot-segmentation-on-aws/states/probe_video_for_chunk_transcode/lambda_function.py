# probe_video_handler.py
"""
Step Function State: Probe video for chunk transcode
-----------------------------------------------------
Fast header-only probe. No keyframe scan. No frame count.
Fail-fast if the file cannot be processed with frame-accurate chunk transcode.

Design principles
-----------------
  1. ffprobe -show_streams -show_format  — fast for all formats, header only
  2. mxf2raw --info --info-format xml    — MXF only, writes mxf_info.json to S3
  3. Test seek capability via ffmpeg -copyts -ss 10 -t 1 -vf showinfo
  4. Determine is_all_intra
  5. Route based on (can_seek, is_all_intra) — see routing table below
  6. Build fixed-size 20-minute iterators (no keyframe snapping)

Seek + is_all_intra routing
----------------------------
  can_seek=True,  is_all_intra=True  → all-intra path
  can_seek=True,  is_all_intra=False → IPB path (-copyts + showinfo PTS)
  can_seek=True,  is_all_intra=None  → IPB path (warn; safe for any codec)
  can_seek=False, is_all_intra=True  → all-intra path
                                        (seek snaps to nearest I-frame;
                                        every frame IS an I-frame → correct)
  can_seek=False, is_all_intra=False → FAIL: long-GOP without seek =
                                        wrong frames guaranteed
  can_seek=False, is_all_intra=None  → FAIL: cannot guarantee accuracy

Fail-fast conditions
--------------------
  - No video stream found
  - fps missing or invalid
  - codec_name not identifiable
  - can_seek=False AND is_all_intra is not True
  - duration missing or <= 0 after all fallbacks

Non-MXF long-GOP (H.264/HEVC in MP4, MOV, etc.) that can seek
  NOT a fail condition. IPB path:
    -copyts + showinfo captures actual source PTS per frame
    frame_number = pts × Fraction(time_base) × Fraction(frame_rate)
  Exact regardless of keyframe seek alignment.

Seek test
---------
  ffmpeg -copyts -ss 10.0 -t 1 -i {url} -vf showinfo -f null /dev/null
  Mimics the IPB transcode path exactly.
  Detects "could not seek to position" in stderr → can_seek=False.
  Captures pts_time of first decoded frame for diagnostic logging.

is_all_intra detection
-----------------------
  Confirmed True:
    - Codec always all-intra: prores, dnxhd, dnxhr, mjpeg, dvvideo
    - D10/IMX codec tag: mx5p, mx4p, mx3p, mx5n, mx4n, mx3n, mx5f, mx4f, mx3f
    - mxf2raw essence_type contains "d10", "imx", or "intra"
  Confirmed False:
    - Codec always long-GOP: h264, hevc, vp9, av1, mpeg4, etc.
    - mxf2raw essence_type contains "long_gop"
  Undetermined:
    - mpeg2video without D10 codec tag and without mxf2raw confirmation
    - Any unrecognised codec
    → if can_seek=True: proceed as IPB (warn); if can_seek=False: fail

Duration priority (MXF)
------------------------
  1. mxf2raw frame_count / fps
  2. ffprobe duration_ts × time_base
  3. ffprobe fmt.duration

Duration priority (non-MXF)
----------------------------
  1. ffprobe duration_ts × time_base
  2. ffprobe fmt.duration

Expected input
--------------
{
    "operation": "StateProbeVideoForChunkTranscode",
    "uuid":   "[UUID]",
    "input":  {
        "s3uri":  "s3://ingest-bucket/path/to/source.mxf",
        "input_config": {
            "segment_duration_secs": 1200,
            "coarse_fps":            4,
            "cosdist_threshold":     0.10,
            "shot_boundary_threshold": 0.20,
            "black_pixel_threshold": 0.04,
            "black_coverage_pct":    99.5,
            "black_entropy_max":     0.20,
            "max_recurrence_threshold": 0.07,
            "min_recurrence_count":  3
        }
    },
    "output": {
        "s3uri": "s3://proxy-bucket/output/prefix"
    },
    "data": {}
}

Output
------
{
    ...original payload...,
    "data": {
        ...original data...,
        "proxy_bucket": "bucket",
        "proxy_prefix": "path/to",
        "probe_s3_uri": "s3://proxy-bucket/prefix/probe.json",
        "probe_s3_key": "prefix/probe.json",
        "mxf_info_key": "prefix/mxf_info.json",
        "video": {
            "duration_sec",
            "duration_source",
            "frame_rate",
            "frame_rate_fps",
            "time_base",
            "is_all_intra",
            "can_seek",             <- bool
            "seek_pts_time",        <- float | null (first pts_time at 10s seek)
            "codec_name",
            ...
        },
        "streams":   [...],
        "iterators": [...]
    }
}

Iterator descriptor
-------------------
{
    "segment_id":           0,
    "segment_name":         "segment_0000",
    "source_bucket":        "...",
    "source_key":           "...",
    "proxy_bucket":         "...",
    "proxy_key":            ".../proxy/segment_0000.mp4",
    "start_time":           0.0,
    "end_time":             1230.0,
    "content_end_time":     1200.0,
    "duration":             1230.0,
    "overlap_secs":         30.0,
    "source_start_frame":   0,
    "frame_rate":           "25/1",
    "time_base":            "1/25",
    "frame_rate_fps":       25.0,
    "is_interlaced":        false,
    "is_all_intra":         true,
    "can_seek":             true,
    "sample_aspect_ratio":  "1:1",
    "timecode_first_frame": "09:59:43:14",
    "drop_frame":           false,
    "input_cropping":       {...} | null
}
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from fractions import Fraction

from lib.exceptions import LargeMediaFileException
from lib.misc_utils import (
    round_half_up,
    fps_from_str,
)
from lib.input_schema import (
    DEFAULT_SEGMENT_DURATION_SECS,
    validate_and_normalize,
    parse_s3uri,
)
from lib.s3_utils import (
    head_size,
)
from lib.mxf_utils import (
    MXF_FORMAT_NAMES,
    D10_CODEC_TAGS,
    ALWAYS_INTRA_CODECS,
    ALWAYS_LONGOP_CODECS,
    run_mxf2raw,
    parse_mxf2raw_info,
    write_mxf_info_to_s3,
    is_mxf as _is_mxf,
)
from lib.ffmpeg_utils import (
    run_ffprobe_streams,
    extract_timecode,
    test_seek_capability,
    write_probe_to_s3,
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
# Raise exception if file is too large to process on 1 core lambda
# =============================================================================
def _raise_on_large_file(bucket: str, key: str) -> None:
    file_size = head_size(bucket, key)
    if file_size >= 10737418240:
        raise LargeMediaFileException(
            f"Large filesize = {file_size}. Defer to 3 cores lambda to process.")


# =============================================================================
# Duration resolution
# =============================================================================


def _resolve_duration(
    v:        dict,
    fmt:      dict,
    mxf_info: dict | None,
    is_mxf:   bool,
    fps:      float,
) -> tuple[float, str]:
    if is_mxf and mxf_info is not None:
        bmx_frames, bmx_fps, _, _ = parse_mxf2raw_info(mxf_info)
        if bmx_frames > 0:
            effective_fps = fps if fps > 0 else bmx_fps
            if effective_fps > 0:
                dur = round(bmx_frames / effective_fps, 6)
                log.info(
                    "Duration from mxf2raw: %d frames / %.6f fps = %.3fs",
                    bmx_frames, effective_fps, dur,
                )
                return dur, "mxf2raw"
            log.warning(
                "mxf2raw frame_count=%d but fps unavailable — cannot use",
                bmx_frames,
            )

    duration_ts = v.get("duration_ts")
    time_base = v.get("time_base")
    if duration_ts is not None and duration_ts not in (0, "N/A", "0") and time_base:
        try:
            dur = float(Fraction(str(duration_ts)) * Fraction(time_base))
            if dur > 0:
                log.info(
                    "Duration from ffprobe duration_ts: %s × %s = %.3fs",
                    duration_ts, time_base, dur,
                )
                if is_mxf:
                    log.warning(
                        "MXF duration from ffprobe duration_ts — "
                        "may be bitrate-estimated for non-compliant files"
                    )
                return round(dur, 6), "ffprobe_ts"
        except (ValueError, ZeroDivisionError):
            pass

    fmt_dur = fmt.get("duration")
    if fmt_dur and fmt_dur != "N/A":
        try:
            dur = float(fmt_dur)
            if dur > 0:
                log.warning(
                    "Duration from ffprobe fmt.duration=%.3fs — "
                    "last resort, may be inaccurate",
                    dur,
                )
                return round(dur, 6), "ffprobe_fmt"
        except ValueError:
            pass

    raise ValueError(
        "Could not determine video duration from any available source"
    )


# =============================================================================
# is_all_intra detection
# =============================================================================

def _determine_is_all_intra(
    v:        dict,
    fmt:      dict,
    mxf_info: dict | None,
) -> bool | None:
    """
    Determine whether the input video is all-intra (I-frame only).

    Returns:
      True  — confirmed all-intra
      False — confirmed long-GOP
      None  — cannot determine

    Routing based on this result is handled in lambda_handler using the
    (can_seek, is_all_intra) table. is_all_intra alone does not fail fast.
    """
    codec = (v.get("codec_name") or "").lower()
    codec_tag = (v.get("codec_tag_string") or "").lower()
    format_name = (fmt.get("format_name") or "").lower()
    is_mxf = any(m in format_name for m in MXF_FORMAT_NAMES)

    if codec in ALWAYS_LONGOP_CODECS:
        log.info("is_all_intra=False: codec %s is always long-GOP", codec)
        return False

    if codec in ALWAYS_INTRA_CODECS:
        log.info("is_all_intra=True: codec %s is always all-intra", codec)
        return True

    if codec_tag in D10_CODEC_TAGS:
        log.info("is_all_intra=True: D10/IMX codec tag %s", codec_tag)
        return True

    if is_mxf and mxf_info is not None:
        _, _, _, essence_type = parse_mxf2raw_info(mxf_info)
        if essence_type:
            et = essence_type.lower()
            if "long_gop" in et:
                log.info(
                    "is_all_intra=False: mxf2raw essence_type=%s", essence_type)
                return False
            if any(x in et for x in ("d10", "imx", "intra")):
                log.info(
                    "is_all_intra=True: mxf2raw essence_type=%s", essence_type)
                return True
            log.warning(
                "is_all_intra undetermined: unrecognised essence_type=%s",
                essence_type,
            )
            return None

    if codec == "mpeg2video":
        log.warning(
            "is_all_intra undetermined: mpeg2video without D10 codec tag "
            "and without mxf2raw essence_type"
        )
        return None

    log.warning("is_all_intra undetermined: unrecognised codec %s", codec)
    return None


# =============================================================================
# Segment iterator builder
# =============================================================================

def _build_segments(
    duration_sec:   float,
    segment_secs:   int,
    overlap_secs:   float,
    source_bucket:  str,
    source_key:     str,
    proxy_bucket:   str,
    proxy_prefix:   str,
    video_meta:     dict,
    input_cropping: dict | None,
) -> list[dict]:
    """
    Build fixed-size segment iterators. No keyframe snapping.

    source_start_frame = round(start_time × fps) — exact for all-intra.
    For IPB content the transcode handler overwrites this from actual PTS.

    can_seek is propagated to each iterator so the transcode handler can
    log it for diagnostics.
    """
    frame_rate = video_meta["frame_rate"]
    fps = video_meta["frame_rate_fps"]
    time_base = video_meta["time_base"]
    is_interlaced = video_meta["is_interlaced"]
    is_all_intra = video_meta["is_all_intra"]
    can_seek = video_meta["can_seek"]
    sar_str = video_meta["sample_aspect_ratio"]
    tc_first = video_meta["timecode_first_frame"]
    drop_frame = video_meta["drop_frame"]

    segments = []
    i = 0
    t = 0.0

    while t < duration_sec:
        content_end = min(t + segment_secs, duration_sec)
        end_time = (
            content_end + overlap_secs
            if content_end < duration_sec
            else content_end
        )
        segment_name = f"segment_{i:04d}"
        proxy_key = f"{proxy_prefix}/proxy/{segment_name}.mp4"

        source_start_frame = round_half_up(t * fps)

        seg = {
            "segment_id":           i,
            "segment_name":         segment_name,
            "source_bucket":         source_bucket,
            "source_key":            source_key,
            "proxy_bucket":         proxy_bucket,
            "proxy_key":            proxy_key,
            "start_time":           round(t, 6),
            "content_end_time":     round(content_end, 6),
            "end_time":             round(end_time, 6),
            "duration":             round(end_time - t, 6),
            "overlap_secs":         round(end_time - content_end, 6),
            "source_start_frame":   source_start_frame,
            "frame_rate":           frame_rate,
            "frame_rate_fps":       fps,
            "time_base":            time_base,
            "is_interlaced":        is_interlaced,
            "is_all_intra":         is_all_intra,
            "can_seek":             can_seek,
            "sample_aspect_ratio":  sar_str,
            "timecode_first_frame": tc_first,
            "drop_frame":           drop_frame,
        }
        if input_cropping:
            seg["input_cropping"] = input_cropping

        segments.append(seg)
        log.info(
            "[%s] %.3fs → %.3fs  (content_end=%.3fs  overlap=%.1fs  "
            "duration=%.1fs  start_frame=%d)",
            segment_name, t, end_time, content_end,
            end_time - content_end,
            end_time - t,
            source_start_frame,
        )
        t += segment_secs
        i += 1

    log.info(
        "Built %d segments  segment_secs=%d  overlap=%.1fs",
        len(segments), segment_secs, overlap_secs,
    )
    return segments


# =============================================================================
# Lambda handler
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateProbeVideoForChunkTranscode event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Normalize input schema ─────────────────────────────────────────────
        normalized_event = validate_and_normalize(event)

        # ── Unpack ────────────────────────────────────────────────────────────
        uuid = normalized_event["uuid"]
        input = normalized_event["input"]
        output = normalized_event["output"]
        data = normalized_event.get("data", {})

        source_bucket,  source_key = parse_s3uri(input["s3uri"])
        proxy_bucket, proxy_prefix = parse_s3uri(output["s3uri"])
        proxy_prefix = proxy_prefix.rstrip("/")

        # ── Check file size to see if we should force a larger lambda ───────────
        is_large_lambda = data.get("is_large_lambda", False)
        if not is_large_lambda:
            _raise_on_large_file(source_bucket, source_key)

        cfg = data["frames_to_shots_cfg"]
        input_cropping = None   # input_cropping removed; not in new input schema

        # For large file, reduce the segment lenght per iterator by half
        segment_secs = int(max(60, cfg["segment_duration_secs"]))
        if is_large_lambda:
            segment_secs = min(segment_secs, int(DEFAULT_SEGMENT_DURATION_SECS / 2))

        overlap_secs = float(cfg["overlap_secs"])

        log.info(
            "Source: s3://%s/%s  segment=%ds  overlap=%.1fs",
            source_bucket, source_key, segment_secs, overlap_secs,
        )

        # ── ffprobe ───────────────────────────────────────────────────────────
        probe_raw = run_ffprobe_streams(source_bucket, source_key)
        streams = probe_raw.get("streams", [])
        fmt = probe_raw.get("format", {})
        format_name = fmt.get("format_name", "")
        is_mxf = _is_mxf(format_name)

        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            raise ValueError("No video stream found in source file")
        v = video_streams[0]

        # ── mxf2raw (MXF only) ────────────────────────────────────────────────
        mxf_info = None
        mxf_info_key = None
        if is_mxf:
            mxf_info = run_mxf2raw(source_bucket, source_key)
            if mxf_info is not None:
                try:
                    mxf_info_key = write_mxf_info_to_s3(
                        proxy_bucket, proxy_prefix, mxf_info
                    )
                except Exception as exc:
                    log.warning("Failed to write mxf_info.json: %s", exc)

        # ── fps — required ────────────────────────────────────────────────────
        fps_str_raw = v.get("r_frame_rate") or v.get("avg_frame_rate") or "0/0"
        fps = fps_from_str(fps_str_raw)

        if fps <= 0 and mxf_info is not None:
            _, bmx_fps, _, _ = parse_mxf2raw_info(mxf_info)
            if bmx_fps > 0:
                log.warning(
                    "ffprobe fps unavailable — using mxf2raw edit_rate=%.4f",
                    bmx_fps,
                )
                fps = bmx_fps
                fps_str_raw = str(Fraction(bmx_fps).limit_denominator(1001))

        if fps <= 0:
            raise ValueError(
                f"Cannot determine frame rate: "
                f"r_frame_rate={v.get('r_frame_rate')}  "
                f"avg_frame_rate={v.get('avg_frame_rate')}"
            )

        # ── codec_name — required ─────────────────────────────────────────────
        codec_name = v.get("codec_name", "").lower()
        if not codec_name:
            raise ValueError("Cannot determine codec_name from ffprobe output")

        # ── is_all_intra ──────────────────────────────────────────────────────
        is_all_intra = _determine_is_all_intra(v, fmt, mxf_info)

        # ── Seek capability test ──────────────────────────────────────────────
        # Mimics the IPB transcode path: -copyts -ss 10 -t 1 -vf showinfo
        # This is the definitive test — if it works, the transcode will work.
        can_seek, seek_pts_time = test_seek_capability(
            source_bucket, source_key)

        # ── Routing: (can_seek, is_all_intra) ─────────────────────────────────
        #
        #   can_seek=True,  any is_all_intra   → proceed
        #     All-intra path OR IPB path both work when seek is available.
        #     If is_all_intra=None, treat as IPB (safe for any codec).
        #
        #   can_seek=False, is_all_intra=True  → proceed (all-intra path)
        #     Every frame is an I-frame. ffmpeg linear scan from start is
        #     slow but produces correct output. Acceptable for all-intra.
        #
        #   can_seek=False, is_all_intra=False → FAIL
        #     Long-GOP without seek: ffmpeg must read from start but can only
        #     decode correctly from a keyframe. The first decoded frame at the
        #     segment boundary is a keyframe that may be seconds before the
        #     requested position — wrong frames guaranteed.
        #
        #   can_seek=False, is_all_intra=None  → FAIL
        #     Cannot confirm safety without seek.

        if not can_seek:
            if is_all_intra is not True:
                raise ValueError(
                    f"File cannot seek and codec {codec_name!r} is "
                    f"{'long-GOP' if is_all_intra is False else 'undetermined'} "
                    f"— cannot guarantee frame accuracy without seek. "
                    f"Use AWS Elemental MediaConvert for this file."
                )
            # can_seek=False, is_all_intra=True — safe to proceed
            log.info(
                "Seek unavailable but is_all_intra=True — "
                "all-intra path is safe (every frame is an I-frame; "
                "ffmpeg will read linearly from start for each segment)"
            )
        else:
            if is_all_intra is None:
                log.warning(
                    "is_all_intra undetermined for codec=%s — "
                    "file can seek, using IPB path as safe default",
                    codec_name,
                )
                is_all_intra = False   # IPB path handles both all-intra and long-GOP
            log.info(
                "File can seek  is_all_intra=%s  codec=%s  container=%s "
                "— proceeding",
                is_all_intra, codec_name, format_name,
            )

        # ── duration — required ───────────────────────────────────────────────
        duration_sec, duration_source = _resolve_duration(
            v, fmt, mxf_info, is_mxf, fps,
        )
        log.info(
            "Duration: %.3fs  source=%s  fps=%.6f",
            duration_sec, duration_source, fps,
        )

        # ── time_base ─────────────────────────────────────────────────────────
        time_base = v.get("time_base") or f"1/{int(round(fps))}"
        log.info("time_base=%s  frame_rate=%s", time_base, fps_str_raw)

        # ── geometry ──────────────────────────────────────────────────────────
        coded_width = int(v.get("coded_width",  v.get("width",  0)))
        coded_height = int(v.get("coded_height", v.get("height", 0)))
        sar_str = v.get("sample_aspect_ratio",  "1:1") or "1:1"
        dar_str = v.get("display_aspect_ratio", "") or ""

        if sar_str not in ("1:1", "0:1", ""):
            try:
                sar = Fraction(sar_str.replace(":", "/"))
                display_width = round_half_up(coded_width * float(sar))
                display_height = coded_height
            except (ValueError, ZeroDivisionError):
                display_width, display_height = coded_width, coded_height
        else:
            display_width, display_height = coded_width, coded_height

        # ── interlace ─────────────────────────────────────────────────────────
        field_order = v.get("field_order", "progressive") or "progressive"
        is_interlaced = field_order not in ("progressive", "unknown", "")

        # ── timecode ──────────────────────────────────────────────────────────
        tc, drop_frame, tc_source = extract_timecode(fmt, streams, v)

        # ── Assemble video_meta ───────────────────────────────────────────────
        video_meta = {
            "duration_sec":         duration_sec,
            "duration_source":      duration_source,
            "frame_rate":           fps_str_raw,
            "frame_rate_fps":       round(fps, 6),
            "time_base":            time_base,
            "is_all_intra":         is_all_intra,
            "can_seek":             can_seek,
            "seek_pts_time":        seek_pts_time,
            "codec_name":           v.get("codec_name",        ""),
            "codec_long_name":      v.get("codec_long_name",   ""),
            "profile":              v.get("profile",           ""),
            "codec_tag_string":     v.get("codec_tag_string",  ""),
            "field_order":          field_order,
            "is_interlaced":        is_interlaced,
            "coded_width":          coded_width,
            "coded_height":         coded_height,
            "display_width":        display_width,
            "display_height":       display_height,
            "sample_aspect_ratio":  sar_str,
            "display_aspect_ratio": dar_str,
            "pix_fmt":              v.get("pix_fmt",           ""),
            "color_space":          v.get("color_space",       ""),
            "color_range":          v.get("color_range",       ""),
            "color_primaries":      v.get("color_primaries",   ""),
            "color_transfer":       v.get("color_transfer",    ""),
            "timecode_first_frame": tc,
            "timecode_source":      tc_source,
            "drop_frame":           drop_frame,
        }

        # ── Build segments ────────────────────────────────────────────────────
        segments = _build_segments(
            duration_sec=duration_sec,
            segment_secs=segment_secs,
            overlap_secs=overlap_secs,
            source_bucket=source_bucket,
            source_key=source_key,
            proxy_bucket=proxy_bucket,
            proxy_prefix=proxy_prefix,
            video_meta=video_meta,
            input_cropping=input_cropping,
        )

        # ── Write probe JSON ──────────────────────────────────────────────────
        probe_data = {
            "uuid":          uuid,
            "source_bucket":  source_bucket,
            "source_key":     source_key,
            "proxy_bucket": proxy_bucket,
            "proxy_prefix": proxy_prefix,
            "video":         video_meta,
            "streams":       streams,
            "format": {
                "format_name":      fmt.get("format_name",      ""),
                "format_long_name": fmt.get("format_long_name", ""),
                "nb_streams":       fmt.get("nb_streams",       0),
                "size":             fmt.get("size"),
                "bit_rate":         fmt.get("bit_rate"),
                "tags":             fmt.get("tags", {}),
            },
            "segments":       segments,
            "input_cropping": input_cropping,
            "mxf_info_key":   mxf_info_key,
        }
        probe_key = write_probe_to_s3(proxy_bucket, proxy_prefix, probe_data)

        # ── Summary ───────────────────────────────────────────────────────────
        elapsed = round(time.monotonic() - t0, 2)
        log.info(
            "Probe complete: duration=%.1fs (%s)  fps=%.4f  time_base=%s  "
            "codec=%s  is_all_intra=%s  can_seek=%s  interlaced=%s  "
            "segments=%d  drop_frame=%s  timecode=%s (%s)  elapsed=%.2fs",
            duration_sec, duration_source, fps, time_base,
            video_meta["codec_name"], is_all_intra, can_seek, is_interlaced,
            len(segments), drop_frame, tc, tc_source, elapsed,
        )

        # ── Return enriched payload ───────────────────────────────────────────
        return_data = {
            **data,
            "source_bucket": source_bucket,
            "source_key":    source_key,
            "proxy_bucket":  proxy_bucket,
            "proxy_prefix":  proxy_prefix,
            "probe_s3_key":  probe_key,
            "video":         video_meta,
            "streams":       streams,
            "iterators":     segments,
        }
        if mxf_info_key:
            return_data["mxf_info_key"] = mxf_info_key

        return {
            **event,
            "data":     return_data,
        }

    except Exception as exc:
        log.error(
            "[ERR] StateProbeVideoForChunkTranscode — %s: %s",
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise
