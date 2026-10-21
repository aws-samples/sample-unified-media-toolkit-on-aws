"""
Video is_all_intra detection, bitrate computation, and scale helpers.
Seek command building removed — ChunkTranscode now owns seek strategy.
"""
from __future__ import annotations

import logging
import math
from fractions import Fraction

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ── Codec sets ─────────────────────────────────────────────────────────────────
_ALWAYS_INTRA = frozenset((
    "prores", "dnxhd", "dnxhr", "mjpeg", "dvvideo",
    "ffv1", "huffyuv", "v210", "r10k", "cineform",
))
_ALWAYS_LONGOP = frozenset((
    "h264", "hevc", "h265", "vp9", "av1", "mpeg4",
    "vp8", "theora", "wmv1", "wmv2", "wmv3",
))
_D10_TAGS = frozenset((
    "mx5p", "mx4p", "mx3p", "mx5n", "mx4n", "mx3n", "mx5f", "mx4f", "mx3f",
))

# ── Kush Gauge constants ───────────────────────────────────────────────────────
_KUSH_CONSTANT     = 0.07
_GOP_BASELINE      = 2.0
_GOP_OVERHEAD_RATE = 0.12
_CBR_IP_COMP       = 1.15
_DEFAULT_MOTION    = 1.5
_DEFAULT_GOP_SECS  = 2.0

# ── Segment constants ──────────────────────────────────────────────────────────
VIDEO_SEGMENT_SECS = 600   # 10 min
VIDEO_MIN_SECS     = 60    # merge final chunk if shorter


# ── is_all_intra ───────────────────────────────────────────────────────────────

def determine_is_all_intra(v: dict) -> bool | None:
    """
    Returns True (all-intra), False (long-GOP), or None (undetermined).
    When None is returned, caller should use scan_is_all_intra() to confirm
    empirically via frame scan.
    """
    codec     = (v.get("codec_name") or "").lower()
    codec_tag = (v.get("codec_tag_string") or "").lower()

    if codec in _ALWAYS_LONGOP:
        log.info("is_all_intra=False: codec %s is always long-GOP", codec)
        return False
    if codec in _ALWAYS_INTRA:
        log.info("is_all_intra=True: codec %s is always all-intra", codec)
        return True
    if codec_tag in _D10_TAGS:
        log.info("is_all_intra=True: D10/IMX codec tag %s", codec_tag)
        return True
    if codec == "mpeg2video":
        log.warning("is_all_intra undetermined: mpeg2video without D10 tag — frame scan needed")
        return None
    log.warning("is_all_intra undetermined: unrecognised codec %s — frame scan needed", codec)
    return None


# ── Bitrate (Kush Gauge) ───────────────────────────────────────────────────────

def compute_bitrate(
    w: int, h: int, fps: Fraction,
    gop_secs: float = _DEFAULT_GOP_SECS,
    motion: float   = _DEFAULT_MOTION,
) -> int:
    """
    Kush Gauge with GOP overhead and CBR/IP-only compensation.
    Result rounded to nearest 10,000 bps. Mirrors videosegment.js _computeBitrate.
    """
    gop_mult = 1 + _GOP_OVERHEAD_RATE * max(0.0, (_GOP_BASELINE / gop_secs) - 1)
    bps = round(w * h * fps * motion * _KUSH_CONSTANT * gop_mult * _CBR_IP_COMP)
    return round(bps / 10_000) * 10_000


# ── Scale dimension ────────────────────────────────────────────────────────────

def scale_dimension(dsp_w: int, dsp_h: int, target_w: int, target_h: int,
                    allow_upscale: bool = False) -> tuple[int, int]:
    """
    Fit display dimension inside target resolution preserving AR.
    Output dimensions are even-pixel aligned.
    If both target_w and target_h are 0, returns source dimensions (even-aligned).
    If only one is 0, the other controls the scaling axis.
    A portrait source SWAPS the target dims (a portrait proxy stays
    portrait) — use fit_dimensions below for a fixed output canvas.
    """
    if target_w <= 0 and target_h <= 0:
        return (dsp_w >> 1) << 1, (dsp_h >> 1) << 1

    tw = target_w if target_w > 0 else dsp_w
    th = target_h if target_h > 0 else dsp_h

    if dsp_w == dsp_h:
        m = min(tw, th); tw = th = m
    elif dsp_h > dsp_w:
        tw, th = th, tw

    factor = min(tw / dsp_w, th / dsp_h)
    if not allow_upscale:
        factor = min(1.0, factor)
    ow = (math.floor(dsp_w * factor + 0.5) >> 1) << 1
    oh = (math.floor(dsp_h * factor + 0.5) >> 1) << 1
    return ow, oh


def fit_dimensions(src_w: int, src_h: int,
                   canvas_w: int, canvas_h: int) -> tuple[int, int]:
    """
    AR-preserving fit inside a FIXED canvas (quick-edit mezzanine):
    always upscales to fill, never swaps orientation — a portrait source
    pillarboxes into a landscape canvas. Even-aligned, clamped so
    half-pixel rounding never exceeds the canvas.
    """
    factor = min(canvas_w / src_w, canvas_h / src_h)
    ow = (math.floor(src_w * factor + 0.5) >> 1) << 1
    oh = (math.floor(src_h * factor + 0.5) >> 1) << 1
    return min(ow, canvas_w), min(oh, canvas_h)


# ── Frame indexing ─────────────────────────────────────────────────────────────

# One snap quantum for every seconds→count conversion in this stack (frames here,
# samples in probe_media_file and chunk_transcode). 1 µs: fine enough that a
# 6-dp ffprobe timestamp is represented exactly and float64 noise still snaps
# away, coarse enough to never move a value across a frame boundary. The old
# fps.denominator*1001 bound was ~0.5 ms at integer fps — coarse enough to snap
# a mark point 0.3 ms past a boundary back onto it, clipping the frame the
# floor/ceil contract promises to keep.
SNAP_DENOMINATOR = 1_000_000


def sec_to_frame(sec: float, fps: Fraction) -> Fraction:
    """
    Exact frame position of a wall-clock second at fps, as a Fraction.

    Callers pick the rounding, because the right one depends on what sec is:
      - an arbitrary in/out point → floor the start, ceil the end, so the
        requested range is never clipped (build_segment_boundaries)
      - a timestamp that is already a frame boundary, e.g. a keyframe's
        best_effort_timestamp_time → round, since the only error present is
        ffprobe's 6-decimal printing (chunk_transcode preroll)

    The only correct way to turn a timestamp into a frame number in this
    stack — never divide a packet pts by its reported `duration`, which is
    not guaranteed to be ticks-per-frame (ffmpeg 6.1's MOV demuxer reports
    duration=1 for any media timescale below 1000).
    """
    return Fraction(sec).limit_denominator(SNAP_DENOMINATOR) * fps


# ── Segment boundaries ─────────────────────────────────────────────────────────

def build_segment_boundaries(
    start_sec: float,
    end_sec: float,
    fps: Fraction,
) -> list[tuple[int, int]]:
    """
    Returns list of (nominal_start_frame, nominal_end_frame) pairs.
    Uses Fraction arithmetic — no floating point accumulation.
    Final segment merged into previous if shorter than VIDEO_MIN_SECS.
    """
    seg_f    = int(Fraction(VIDEO_SEGMENT_SECS) * fps)
    min_f    = int(Fraction(VIDEO_MIN_SECS)     * fps)
    start_f  = math.floor(sec_to_frame(start_sec, fps))
    end_f    = math.ceil( sec_to_frame(end_sec,   fps))
    total_f  = end_f - start_f

    boundaries: list[tuple[int, int]] = []
    f = 0
    while f < total_f:
        seg = min(seg_f, total_f - f)
        boundaries.append((start_f + f, start_f + f + seg))
        f += seg

    if len(boundaries) > 1 and (boundaries[-1][1] - boundaries[-1][0]) < min_f:
        last = boundaries.pop()
        prev_start, _ = boundaries[-1]
        boundaries[-1] = (prev_start, last[1])

    return boundaries
