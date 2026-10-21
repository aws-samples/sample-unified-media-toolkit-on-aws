"""
Probe-time computation helpers for StateProbeMediaFile.

Separated from lambda_function.py to keep the handler readable.
None of these functions are called by ChunkTranscode — they produce
the pre-computed values stamped onto each iterator.
"""
from __future__ import annotations

import logging
import math
from fractions import Fraction

from lib.video_command import compute_bitrate, scale_dimension

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def resolve_video_group_settings(
    g: dict,
    vstream: dict,
    fps: Fraction,
    crop_x: int = 0,
    crop_y: int = 0,
    target_width: int = 0,
    target_height: int = 0,
) -> dict:
    """
    Resolve effective dimensions, bitrate, gopsize, and filter parameters.

    crop_x / crop_y are display-pixel overscan values from input_config (shared
    across video and frame groups). target_width / target_height are the recommended
    output dimensions, also from input_config.

    Crop is adjusted to preserve source DAR. scale_dimension receives post-crop
    display dimensions so output AR is correct.
    """
    gop_secs = g["gop_seconds"]
    motion   = g["motion_factor"]
    codec    = g["codec"]

    cw      = int(vstream.get("coded_width")  or vstream.get("width")  or 0)
    ch      = int(vstream.get("coded_height") or vstream.get("height") or 0)
    if cw <= 0 or ch <= 0:
        raise ValueError(
            f"Video stream has invalid dimensions {cw}x{ch} — "
            "file may be corrupt or stream metadata incomplete")
    sar_str = vstream.get("sample_aspect_ratio") or "1:1"
    try:
        sar = Fraction(sar_str.replace(":", "/"))
        if sar <= 0:
            raise ValueError(f"SAR must be positive, got {sar_str!r}")
    except (ValueError, ZeroDivisionError):
        log.warning("Unparseable SAR %r — assuming 1:1", sar_str)
        sar = Fraction(1)

    # Display dimensions before crop — used to compute DAR for AR preservation
    dsp_w_full = math.floor(cw * float(sar) + 0.5) if sar != Fraction(1) else cw
    dsp_h_full = ch
    dar = Fraction(dsp_w_full, dsp_h_full) if dsp_h_full > 0 else Fraction(1)

    if crop_x < 0 or crop_y < 0:
        raise ValueError(
            f"crop_x and crop_y must be >= 0, got crop_x={crop_x} crop_y={crop_y}")

    # Clamp crop to leave at least 2 pixels in each dimension (even-alignment headroom)
    max_crop_w = max(0, dsp_w_full - 2)
    max_crop_h = max(0, dsp_h_full - 2)
    if crop_x > max_crop_w or crop_y > max_crop_h:
        raise ValueError(
            f"crop_x={crop_x} crop_y={crop_y} exceeds source display dimensions "
            f"{dsp_w_full}x{dsp_h_full}")

    # Adjust display-space crop to preserve DAR (overscan removal must keep AR)
    crop_x_display = crop_x
    crop_y_display = crop_y
    if crop_x_display > 0 or crop_y_display > 0:
        crop_x2 = math.floor(crop_y_display * float(dar) + 0.5)
        if crop_x2 > crop_x_display:
            crop_x_display = crop_x2
        else:
            crop_y_display = math.floor(crop_x_display / float(dar) + 0.5)

    # Re-validate after AR adjustment — expansion can push adjusted value past source bounds
    if crop_x_display > max_crop_w or crop_y_display > max_crop_h:
        raise ValueError(
            f"Crop after AR adjustment (crop_x={crop_x_display}, crop_y={crop_y_display}) "
            f"exceeds source display dimensions {dsp_w_full}x{dsp_h_full}. "
            f"Reduce input crop_x={crop_x} or crop_y={crop_y}.")

    # Align to multiples of 4 so the crop offset (ccw//2, cch//2) is always even.
    # Required for YUV 4:2:0 chroma alignment — an odd offset shifts chroma by 1px.
    if sar != Fraction(1):
        coded_crop_w = (math.floor(crop_x_display * sar.denominator / sar.numerator + 0.5) >> 2) << 2
    else:
        coded_crop_w = (crop_x_display >> 2) << 2
    coded_crop_h = (crop_y_display >> 2) << 2

    # Post-crop display dimensions — what scale_dimension and bitrate must use
    post_cw = max(0, cw - coded_crop_w)
    post_ch = max(0, ch - coded_crop_h)
    dsp_w   = math.floor(post_cw * float(sar) + 0.5) if sar != Fraction(1) else post_cw
    dsp_h   = post_ch

    ow, oh  = scale_dimension(dsp_w, dsp_h, target_width, target_height)
    if ow <= 0 or oh <= 0:
        raise ValueError(
            f"scale_dimension returned invalid output size {ow}x{oh} "
            f"(post-crop display={dsp_w}x{dsp_h}, target={target_width}x{target_height})")
    gopsize = max(1, math.ceil(fps * gop_secs))
    bitrate = compute_bitrate(ow, oh, fps, gop_secs, motion)

    return {
        "codec":        codec,
        "width":        ow,
        "height":       oh,
        "bitrate":      bitrate,
        "gopsize":      gopsize,
        "coded_crop_w": coded_crop_w,
        "coded_crop_h": coded_crop_h,
        "sar":          str(sar),
        "src_width":    dsp_w,
        "src_height":   dsp_h,
    }


def build_vfilters(
    settings: dict,
    interlaced: bool,
    frame_interval: int | None,
    has_ts_output: bool = True,
    video_mapid: str = "0:v:0",
) -> list[str]:
    """
    Build vfilters array for filter_complex.
    Filter order: trim → setpts → yadif → crop → PAR normalise → scale → split

    has_ts_output: False when there is no mp4 output group (frame-only). In that
    case the split node is omitted and scale feeds [vjpeg] directly, avoiding a
    dangling [vts] label that ffmpeg would reject.
    """
    ccw      = settings["coded_crop_w"]
    cch      = settings["coded_crop_h"]
    ow       = settings["width"]
    oh       = settings["height"]
    has_jpeg = frame_interval is not None

    if has_jpeg and has_ts_output:
        # scale → [vscaled], then split into [vts] and [vjpeg]
        out_label = "[vscaled]"
    elif has_jpeg:
        # frame-only: scale feeds [vjpeg] directly — no [vts] produced
        out_label = "[vjpeg]"
    else:
        out_label = "[vts]"

    f = f"[{video_mapid}]trim=start_frame={{start_frame}}:end_frame={{end_frame}},setpts=PTS-STARTPTS"
    if interlaced:
        f += ",yadif=deint=interlaced"
    if ccw > 0 or cch > 0:
        f += f",crop=iw-{ccw}:ih-{cch}:{ccw // 2}:{cch // 2}"
    f += f",scale={ow}:{oh},setsar=1/1{out_label}"

    filters = [f]

    if has_jpeg and has_ts_output:
        filters.append("[vscaled]split=2[vts][vjpeg]")

    if has_jpeg:
        filters.append(f"[vjpeg]select='not(mod(n\\,{frame_interval}))'[jout]")

    return filters


def build_afilters(audio_streams: list[dict]) -> list[str]:
    """
    Build afilters array for filter_complex.
    Selects path based on number of streams and channel count.
    All paths converge at [_ax] → atrim → [aout].
    """
    n = len(audio_streams)

    if n == 0:
        return []

    if n >= 2:
        filters = [
            f"[{s['mapid']}]aresample=48000:async=1,apad,aformat=sample_fmts=s16:channel_layouts=mono[_a{i}]"
            for i, s in enumerate(audio_streams)
        ]
        inputs = "".join(f"[_a{i}]" for i in range(n))
        filters.append(f"{inputs}amerge=inputs={n}[_ax]")
    else:
        channels = int(audio_streams[0].get("channels") or 1)
        if channels > 2:
            log.warning("Audio stream has %d channels — downmixing to stereo", channels)
        layout = "mono" if channels == 1 else "stereo"
        filters = [
            f"[{audio_streams[0]['mapid']}]aresample=48000:async=1,apad,aformat=sample_fmts=s16:channel_layouts={layout}[_ax]"
        ]

    filters.append(
        f"[_ax]atrim=start_sample={{start_sample}}:end_sample={{end_sample}},asetpts=PTS-STARTPTS[aout]"
    )
    return filters


# ── metadata output group ──────────────────────────────────────────────────────

# Per-stream keys surfaced only at detail="full". Both are already present in the
# -show_streams result, so "full" is a filtering decision — it costs no second
# ffprobe invocation.
_FULL_ONLY_STREAM_KEYS = ("disposition", "tags")


def _strip_stream_keys(streams: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in s.items() if k not in _FULL_ONLY_STREAM_KEYS}
        for s in streams or []
    ]


def _filter_probe_info(probe: dict, detail: str) -> dict:
    """
    Return a copy of the ffprobe result with detail-gated keys applied.

    -show_programs nests a full copy of each program's streams under
    programs[].streams[], so the same keys must be stripped there too — filtering
    only the top-level list leaks disposition/tags at detail="default" on every
    MPEG-TS source (the only container that reports programs).

    Gated keys are per-*stream* only. Container-level format.tags, chapters[].tags
    (which is where a chapter's title lives) and programs[].tags (service_name) are
    the substance of those sections, not stream detail, and are always present.
    """
    if detail == "full":
        return probe

    filtered = dict(probe)
    filtered["streams"] = _strip_stream_keys(probe.get("streams", []))

    if probe.get("programs"):
        filtered["programs"] = [
            {**p, "streams": _strip_stream_keys(p.get("streams", []))}
            for p in probe["programs"]
        ]
    return filtered


def build_metadata_payload(
    detail: str,
    fmt_name: str,
    duration_sec: float,
    duration_source: str,
    file_size: int,
    has_video: bool,
    has_audio: bool,
    is_all_intra: bool,
    vstream: dict | None,
    audio_streams: list[dict],
    interlaced: bool,
    probe: dict,
    mxf_block: dict | None = None,
) -> dict:
    """
    Build the resolved-facts payload written to {prefix}/{group_name}/metadata.json.

    The value over raw ffprobe JSON is the resolved fields: duration_sec with its
    duration_source rung, and is_all_intra, which is not an ffprobe field at all.
    The `mxf` key is present only for MXF sources — its absence means "not MXF".
    """
    payload: dict = {
        "format_name":     fmt_name,
        "duration_sec":    duration_sec,
        "duration_source": duration_source,
        "file_size":       file_size,
        "has_video":       has_video,
        "has_audio":       has_audio,
        "is_all_intra":    is_all_intra,
    }

    if vstream:
        payload["video"] = {
            "codec":               vstream.get("codec_name"),
            "width":               vstream.get("width"),
            "height":              vstream.get("height"),
            "fps":                 vstream.get("r_frame_rate") or vstream.get("avg_frame_rate"),
            "interlaced":          interlaced,
            "sample_aspect_ratio": vstream.get("sample_aspect_ratio"),
            "field_order":         vstream.get("field_order") or "progressive",
        }

    payload["audio_streams"] = [
        {
            "mapid":       a.get("mapid"),
            "codec":       a.get("codec_name"),
            "channels":    a.get("channels"),
            "sample_rate": int(a["sample_rate"]) if str(a.get("sample_rate") or "").isdigit() else None,
        }
        for a in audio_streams
    ]

    if mxf_block is not None:
        payload["mxf"] = mxf_block

    payload["probe_info"] = _filter_probe_info(probe, detail)
    return payload


def build_chunk_outputs(
    item_id: int,
    settings: dict | None,
    has_audio: bool,
    frame_interval: int | None,
    frame_format: str = "jpeg",
    frame_group_name: str = "frames",
) -> list[dict]:
    """Build chunk_outputs list for the iterator."""
    outputs = []

    if settings:
        outputs.append({
            "s3_prefix": "segment_video",
            "s3_name":   f"video_{item_id}.ts",
            "map":       f"[vts] -c:v {settings['codec']} -b:v {settings['bitrate']} -g {settings['gopsize']} -f mpegts",
        })

    if has_audio:
        outputs.append({
            "s3_prefix": "segment_audio",
            "s3_name":   f"audio_{item_id}.flac",
            "map":       "[aout] -c:a flac -ar 48000 -ac 2",
        })

    if frame_interval is not None:
        ext    = "png" if frame_format == "png" else "jpg"
        q_flag = "-compression_level 6" if frame_format == "png" else "-q:v 2"
        outputs.append({
            "s3_prefix":     frame_group_name,
            "frame_format":  frame_format,
            "local_pattern": f"frame_%07d.{ext}",
            "map":           f"[jout] -fps_mode passthrough -start_number 0 {q_flag} -f image2",
        })

    if not outputs:
        raise ValueError(
            "build_chunk_outputs produced no outputs — "
            "video_group_settings, has_audio, and frame_interval are all absent/False")

    return outputs