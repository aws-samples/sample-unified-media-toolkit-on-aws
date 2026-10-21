"""
Quick-edit ffmpeg filter builders — plain strings, one function per
concern, no filter-graph abstraction. The policies are the pinned
normalization matrix in README.md §14.2: fit + letterbox (never
stretch), SAR 1:1, fps dup/drop (no motion interpolation), SDR BT.709
yuv420p output, hable tonemap for HDR, yadif for interlaced, 48 kHz
stereo audio, EBU R128 two-pass loudnorm to −23 LUFS.

Pure string builders — no subprocess, no boto3. Rotation metadata needs
no filter here: ffmpeg auto-rotates on decode, so callers pass
POST-ROTATION display dimensions.
"""
from __future__ import annotations

from .video_command import fit_dimensions  # noqa: F401  (dimension math lives there)

MICRO_FADE_SEC = 0.02          # non-optional boundary click guard
_HDR_TRANSFERS = frozenset(("smpte2084", "arib-std-b67"))


def build_audio_conform(loudnorm: str | None = None,
                        gain: str | None = None) -> str:
    """The house audio conform, mirroring probe_helper.build_afilters:
    aresample 48 kHz with async=1 (timestamp-driven alignment), level
    passes, then apad — SILENCE-PADS so a sample-based atrim end is
    always reachable (a source whose audio runs short near the out
    point must not shorten the chunk) — then s16 stereo. Callers append
    the sample-based atrim + asetpts."""
    parts = ["aresample=48000:async=1"]
    if loudnorm:
        parts.append(loudnorm)
    if gain:
        parts.append(gain)
    parts += ["apad", "aformat=sample_fmts=s16:channel_layouts=stereo"]
    return ",".join(parts)

LOUDNORM_TARGET = "I=-23:TP=-1.5:LRA=11"


def is_hdr(color_transfer: str | None) -> bool:
    """The REFUSAL predicate: v1 refuses HDR sources by name (the
    LGPL build has no zscale, so the tonemap chain cannot run; see
    README.md §14.1). Probe calls this."""
    return (color_transfer or "").lower() in _HDR_TRANSFERS


def crop_pixels(crop: dict, src_w: int, src_h: int) -> tuple[int, int, int, int]:
    """Normalized 0–1 crop ratios → even-aligned source pixels (w,h,x,y),
    clamped so rounding never leaves the frame."""
    w = (round(crop["width"] * src_w) >> 1) << 1
    h = (round(crop["height"] * src_h) >> 1) << 1
    w, h = max(w, 2), max(h, 2)
    # offsets even-aligned too: odd x/y on 4:2:0 is legal but shifts
    # chroma half a sample — 1 px of placement buys a clean plane
    x = (min(round(crop["x"] * src_w), src_w - w) >> 1) << 1
    y = (min(round(crop["y"] * src_h), src_h - h) >> 1) << 1
    return w, h, max(x, 0), max(y, 0)


def build_normalize_vfilters(src_w: int, src_h: int, field_order: str | None,
                             canvas_w: int, canvas_h: int,
                             target_fps: str, crop: dict | None = None) -> list[str]:
    """
    Source → common mezzanine: deinterlace, [crop], fit, letterbox/
    pillarbox, SAR 1:1, constant target rate, yuv420p. HDR never
    reaches here — probe refuses it via is_hdr(). crop is the
    per-segment NORMALIZED reframe window (social-clip reframing).
    """
    filters: list[str] = []
    if field_order not in (None, "", "progressive", "unknown"):
        filters.append("yadif=deint=interlaced")   # the proxy chain's spelling
    if crop is not None:
        cw, ch, cx, cy = crop_pixels(crop, src_w, src_h)
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
        src_w, src_h = cw, ch
    ow, oh = fit_dimensions(src_w, src_h, canvas_w, canvas_h)
    filters.append(f"scale={ow}:{oh}")
    if (ow, oh) != (canvas_w, canvas_h):
        filters.append(f"pad={canvas_w}:{canvas_h}:"
                       f"{(canvas_w - ow) // 2}:{(canvas_h - oh) // 2}")
    filters.append("setsar=1")
    filters.append(f"fps={target_fps}")
    filters.append("format=yuv420p")
    return filters


def black_source_args(canvas_w: int, canvas_h: int, target_fps: str,
                      n_frames: int) -> list[str]:
    """-f lavfi input args generating the "black" segment's picture."""
    return ["-f", "lavfi", "-i",
            f"color=c=black:s={canvas_w}x{canvas_h}:r={target_fps}",
            "-frames:v", str(n_frames)]


def build_fade_vfilters(fade_in_frames: int, fade_out_frames: int,
                        total_frames: int) -> list[str]:
    """Within-segment fade envelopes, frame-exact (the `fade` transition)."""
    filters: list[str] = []
    if fade_in_frames > 0:
        filters.append(f"fade=t=in:s=0:n={fade_in_frames}")
    if fade_out_frames > 0:
        filters.append(f"fade=t=out:s={total_frames - fade_out_frames}:"
                       f"n={fade_out_frames}")
    return filters


def build_fade_afilters(fade_in_sec: float, fade_out_sec: float,
                        duration_sec: float) -> list[str]:
    """
    Audio envelope. The 20 ms micro-fade is the FLOOR on both edges —
    non-optional even on cuts (hard cuts at arbitrary sample positions
    click).
    """
    f_in = max(fade_in_sec, MICRO_FADE_SEC)
    f_out = max(fade_out_sec, MICRO_FADE_SEC)
    return [f"afade=t=in:st=0:d={f_in:g}",
            f"afade=t=out:st={duration_sec - f_out:g}:d={f_out:g}"]


def build_junction_filters(d_frames: int, fps_float: float) -> tuple[str, str]:
    """
    The crossfade junction: both inputs are exactly D long, blended over
    the full D (offset 0), so the output is D long. Returns (video
    filter, audio filter) for the two-input graph.
    """
    d_sec = d_frames / fps_float
    return (f"xfade=transition=fade:duration={d_sec:g}:offset=0",
            f"acrossfade=d={d_sec:g}")


def gain_filter(gain_db: float) -> str | None:
    """Static element gain (mix overlays/beds). None at 0 — no-op filter."""
    return None if gain_db == 0 else f"volume={gain_db:g}dB"


def loudnorm_measure_filter() -> str:
    """Pass 1: measure only. Parse the json block from stderr."""
    return f"loudnorm={LOUDNORM_TARGET}:print_format=json"


LOUDNESS_TARGET_I = -23.0      # EBU R128
LOUDNESS_GAIN_CLAMP_DB = 30.0  # a bigger correction than this is a
                               # measurement artifact, not a master


def loudness_gain_filter(measured: dict) -> str:
    """
    Pass 2: a PLAIN constant gain (volume=) computed from the pass-1
    measurement — deliberately NOT loudnorm. loudnorm's linear mode
    silently FALLS BACK to dynamic processing when the gain would
    exceed its TP target, and dynamic mode outputs 192 kHz — which
    quartered every sample-based atrim downstream (measured live
    2026-10-10: "Expect 7200 ms. Received 1800 ms"). volume= is pure
    gain at the input rate with no fallback semantics; TP safety is
    the chain-final alimiter.
    """
    gain = LOUDNESS_TARGET_I - float(measured["input_i"])
    gain = max(-LOUDNESS_GAIN_CLAMP_DB, min(LOUDNESS_GAIN_CLAMP_DB, gain))
    return f"volume={gain:.2f}dB"
