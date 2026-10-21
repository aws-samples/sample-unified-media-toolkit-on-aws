"""
Audio stream selection for the media-processing workflow.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ── Stream selection ───────────────────────────────────────────────────────────

# ffmpeg cannot decode these broadcast codecs
_UNSUPPORTED_CODECS = frozenset(("dolby_e", "dolby_ed2"))


def select_audio_streams(
    streams: list[dict],
    use_mapping: bool,
    channel_mapping: list[int],
) -> list[dict]:
    """
    If use_mapping=True, select streams by 1-based index from channel_mapping.
    Otherwise automatic: prefer multichannel → fall back to up to two mono tracks.
    Dolby E and other broadcast codecs ffmpeg cannot decode are always excluded.
    """
    if not streams:
        return []

    def _is_decodable(s: dict) -> bool:
        codec = (s.get("codec_name") or "").lower()
        if codec in _UNSUPPORTED_CODECS:
            log.warning("Skipping audio stream %s: codec %r is not decodable by ffmpeg",
                        s.get("mapid", s.get("index")), codec)
            return False
        return True

    if use_mapping and channel_mapping:
        # Channel mapping uses 1-based indices into the original stream order.
        # Filter unsupported codecs after selection so indices stay stable.
        # Cap at 2 streams — amerge beyond 2 produces unexpected downmix matrices.
        capped = channel_mapping[:2]
        invalid = [i for i in capped if not (0 < i <= len(streams))]
        if invalid:
            log.warning(
                "audio_channel_mapping indices %s out of range (file has %d audio stream(s)) "
                "— falling back to auto-select", invalid, len(streams))
        else:
            selected = [streams[i - 1] for i in capped]
            result   = [s for s in selected if _is_decodable(s)]
            if result:
                return result
            log.warning("All mapped audio streams are unsupported codecs — falling back to auto-select")

    decodable = [s for s in streams if _is_decodable(s)]
    if not decodable:
        return []
    multi = sorted(
        [s for s in decodable if s.get("channels", 1) > 1],
        key=lambda s: s.get("index", 0))
    if multi:
        return [multi[0]]
    mono = sorted(
        [s for s in decodable if s.get("channels", 1) == 1],
        key=lambda s: s.get("index", 0))
    return mono[:2] if mono else [decodable[0]]
