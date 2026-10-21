"""
Input schema parsing and validation for the redesigned media-processing workflow.
"""
from __future__ import annotations

import logging
import math
import os
from fractions import Fraction

log = logging.getLogger(__name__)

VALID_FORMATS    = frozenset(("mp4", "wav", "m4a", "jpeg", "png",
                              "metadata", "ad_mix"))
VIDEO_FORMATS    = frozenset(("mp4",))
AUDIO_FORMATS    = frozenset(("wav", "m4a"))
FRAME_FORMATS    = frozenset(("jpeg", "png"))
METADATA_FORMATS = frozenset(("metadata",))
# ad_mix (2026-08-31, for audio-description-on-aws): the programme audio
# ducked under a bus of pre-placed narration takes. Rides the SAME .flac
# chunk machinery as wav/m4a; the render happens at merge over the
# concatenated bed, driven by a render manifest the AD workflow wrote.
AD_MIX_FORMATS   = frozenset(("ad_mix",))
# ONE name for "needs the .flac audio chunks" — probe's has_audio check
# and group partition both read this, so the next flac-backed format is
# a one-set change, not a convention to remember
FLAC_BACKED_FORMATS = AUDIO_FORMATS | AD_MIX_FORMATS

# metadata group `detail` levels — "default" is format/streams/chapters/programs,
# "full" additionally surfaces per-stream disposition and tags. Both are served by
# the single run_ffprobe call; frame/packet level is deliberately not offered.
METADATA_DETAIL_LEVELS = frozenset(("default", "full"))

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540

DEFAULT_VIDEO_GROUP = {
    "group_name":    "proxy",
    "format":        "mp4",
    "name":          "transcoded",
    "codec":         "libopenh264",
    "gop_seconds":   2.0,
    "motion_factor": 1.6,
}


class NoCompatibleOutputGroups(ValueError):
    """Every requested output_group needs a stream the source lacks (a wav
    group against a silent video is the canonical case). A NAMED class
    because the class name IS the execution's error field — the stable
    code consumers branch on (speech-diarization raises its own named
    NoAudioStream on it instead of an opaque failure).
    Subclasses ValueError so existing handling is unchanged."""


def parse_s3uri(uri: str) -> tuple[str, str]:
    """
    Parse 's3://bucket/key/path' → ('bucket', 'key/path').
    Raises ValueError on malformed input.
    """
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Invalid s3uri — must start with 's3://': {uri!r}")
    rest = uri[5:]
    if "/" not in rest:
        raise ValueError(f"Invalid s3uri — no key path after bucket: {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket:
        raise ValueError(f"Invalid s3uri — empty bucket: {uri!r}")
    key = key.strip("/")
    return bucket, key


def _sanitize_name(name: str, fallback: str) -> str:
    """Strip path separators and '..' components. Returns fallback if result is empty."""
    cleaned = os.path.basename(name.replace("..", "").strip("/\\"))
    return cleaned if cleaned else fallback


def _apply_video_defaults(g: dict) -> dict:
    out = dict(DEFAULT_VIDEO_GROUP)
    out.update({k: v for k, v in g.items() if v is not None})
    out["name"]          = _sanitize_name(str(out.get("name") or out["group_name"]), out["group_name"])
    out["codec"]         = out.get("codec") or "libopenh264"
    if out["codec"] == "openh264":
        out["codec"] = "libopenh264"
    out["gop_seconds"]   = float(out.get("gop_seconds")  or 2.0)
    out["motion_factor"] = float(out.get("motion_factor") or 1.6)
    return out


def _apply_audio_defaults(g: dict) -> dict:
    out = dict(g)
    out["name"]          = _sanitize_name(str(out.get("name") or out["group_name"]), out["group_name"])
    out["sampling_rate"] = int(out.get("sampling_rate") or 48000)
    out["audio_channels"]= int(out.get("audio_channels") or (2 if out["format"] == "m4a" else 1))
    if out["format"] == "wav":
        out["codec"] = out.get("codec") or "pcm_s16le"
    else:
        # m4a — codec field is accepted but always encodes AAC
        out["codec"]         = "aac"
        out["audio_bitrate"] = int(out.get("audio_bitrate") or 96)
    return out


def _apply_frame_defaults(g: dict) -> dict:
    out = dict(g)
    out["name"]               = _sanitize_name(str(out.get("name") or out["group_name"]), out["group_name"])
    out["frame_format"]       = out["format"]   # "jpeg" or "png" — preserved for downstream use
    out["frame_interval_sec"] = float(out.get("frame_interval_sec") or 1.0)
    if out["frame_interval_sec"] <= 0:
        raise ValueError(
            f"frame_interval_sec must be > 0, got {out['frame_interval_sec']!r} "
            f"in group {out['group_name']!r}")
    return out


def _apply_ad_mix_defaults(g: dict) -> dict:
    out = dict(g)
    out["name"]          = _sanitize_name(str(out.get("name") or out["group_name"]), out["group_name"])
    # sampling_rate deliberately NOT defaulted: the render manifest's
    # sample_rate is the fallback at merge, and stamping 48000 here made
    # that fallback dead code (review 2026-08-31) — only an explicit
    # caller value should outrank the manifest
    if out.get("sampling_rate"):
        out["sampling_rate"] = int(out["sampling_rate"])
    out["audio_channels"]= int(out.get("audio_channels") or 2)
    out["codec"]         = out.get("codec") or "pcm_s16le"
    manifest = (out.get("render_manifest_s3uri") or "").strip()
    if not manifest:
        raise ValueError(
            f"output_group {out['group_name']!r} (format=ad_mix) requires "
            "render_manifest_s3uri — the AD workflow's render_manifest.json "
            "(takes with wav/at/tempo + af_chain)")
    parse_s3uri(manifest)      # malformed uris refuse here
    out["render_manifest_s3uri"] = manifest
    return out


def _apply_metadata_defaults(g: dict) -> dict:
    out = dict(g)
    out["name"]   = _sanitize_name(str(out.get("name") or "metadata"), "metadata")
    out["detail"] = (str(out.get("detail") or "default")).lower().strip()
    if out["detail"] not in METADATA_DETAIL_LEVELS:
        raise ValueError(
            f"output_group {out['group_name']!r} has unsupported detail "
            f"{out['detail']!r}. Must be one of: {sorted(METADATA_DETAIL_LEVELS)}")
    return out


def parse_output_groups(output_groups: list | None, has_video: bool, has_audio: bool) -> list[dict]:
    """
    Validate, deduplicate, and apply defaults to output_groups.

    - group_name must be unique; first duplicate wins (logged as warning).
    - format must be one of: mp4, wav, m4a, jpeg, png, metadata.
    - Groups incompatible with the source (video group on audio-only, etc.) are
      silently dropped with a warning rather than raising.
    - metadata groups are never dropped — they need neither video nor audio, so
      their presence alone satisfies the "no compatible groups" check below.
    - If output_groups is None or empty, returns the default mp4 proxy group
      (raises ValueError if source has no video).
    - Raises ValueError if no groups remain after filtering.
    """
    if output_groups is not None and not isinstance(output_groups, list):
        raise ValueError(
            f"output_groups must be a list, got {type(output_groups).__name__!r}")

    if not output_groups:
        if not has_video:
            raise ValueError("No output_groups specified and source has no video stream")
        log.info("output_groups absent — using default mp4 proxy group")
        return [_apply_video_defaults(dict(DEFAULT_VIDEO_GROUP))]

    seen: dict[str, bool] = {}
    result: list[dict] = []

    for g in output_groups:
        if not isinstance(g, dict):
            raise ValueError(
                f"Each output_group must be a dict, got {type(g).__name__!r}: {g!r}")

        group_name = (g.get("group_name") or "").strip()
        fmt        = (g.get("format")     or "").lower().strip()

        if not group_name:
            raise ValueError(f"output_group missing group_name: {g!r}")
        if not fmt:
            raise ValueError(f"output_group {group_name!r} missing format")
        if fmt not in VALID_FORMATS:
            raise ValueError(
                f"output_group {group_name!r} has unsupported format {fmt!r}. "
                f"Must be one of: {sorted(VALID_FORMATS)}")

        if group_name in seen:
            log.warning("Duplicate group_name %r — keeping first, dropping subsequent", group_name)
            continue
        seen[group_name] = True

        g = dict(g)
        g["format"] = fmt
        g["group_name"] = group_name

        if fmt in VIDEO_FORMATS:
            if has_video:
                result.append(_apply_video_defaults(g))
            else:
                log.warning("Dropping output_group %r (format=%r): source has no video stream", group_name, fmt)
        elif fmt in FRAME_FORMATS:
            if has_video:
                result.append(_apply_frame_defaults(g))
            else:
                log.warning("Dropping output_group %r (format=%r): source has no video stream", group_name, fmt)
        elif fmt in AUDIO_FORMATS:
            if has_audio:
                result.append(_apply_audio_defaults(g))
            else:
                log.warning("Dropping output_group %r (format=%r): source has no audio stream", group_name, fmt)
        elif fmt in AD_MIX_FORMATS:
            if has_audio:
                result.append(_apply_ad_mix_defaults(g))
            else:
                log.warning("Dropping output_group %r (format=%r): source has no audio stream", group_name, fmt)
        elif fmt in METADATA_FORMATS:
            # Requires neither video nor audio — a probe result exists for any
            # decodable container, so there is no compatibility drop to apply.
            result.append(_apply_metadata_defaults(g))

    if not result:
        src_desc = []
        if not has_video:
            src_desc.append("no video")
        if not has_audio:
            src_desc.append("no audio")
        raise NoCompatibleOutputGroups(
            f"No output_groups are compatible with source ({', '.join(src_desc) or 'unknown streams'})"
        )

    return result


def compute_frame_interval(fps: Fraction, frame_interval_sec: float) -> int:
    """
    Compute integer frame_interval for ffmpeg select filter.
    frame_interval = ceil(fps × frame_interval_sec)
    Uses Fraction arithmetic — no floating point rounding.
    Returns minimum 1.
    """
    interval_f = Fraction(frame_interval_sec).limit_denominator(10000)
    result = math.ceil(fps * interval_f)
    return max(1, result)
