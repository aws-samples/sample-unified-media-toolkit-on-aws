"""
Helpers for StateCheckAndPreprocess.

check_source(bucket, key)
    Probe the S3 object to determine whether it is a compatible WAV.
    Non-WAV extension → returns incompatible without fetching.
    WAV extension → fetches 64 KB header and validates: exactly 48 kHz, mono, PCM/float.
    Returns {"is_compatible_wav": bool, "info": WavInfo | None, "reason": str}.
    info is set (non-None) when the file is a parseable WAV regardless of compatibility,
    so the caller can pass it to wav_to_wav.convert() without a second S3 fetch.

build_media_processing_request(src_bucket, src_key)
    Build the input JSON for the media-processing state machine.
    Requests one WAV output group named "audiostem_sep":
      format=wav, name=audio_48k_mono, sampling_rate=48000, channels=1, codec=pcm_s16le
    The child SFN writes to {out_prefix}/media_processing/audiostem_sep/audio_48k_mono.wav.
    StateExtractProcessedAudio reads that key from the child execution output.
"""
from __future__ import annotations

import logging
import os

from lib.input_schema import TARGET_SR as _REQUIRED_SAMPLE_RATE
from lib.s3_utils import s3 as _s3
from lib.wav_utils import parse_wav_header

log = logging.getLogger(__name__)

_HEADER_READ_SIZE      = 65_536   # 64 KB — same as wav_utils._HEADER_READ_SIZES[-1]


def check_source(bucket: str, key: str) -> dict:
    """
    Determine whether the S3 object is a compatible WAV file.

    Non-WAV extension → skip the S3 fetch entirely.
    Otherwise fetch up to 64 KB and attempt WAV header parse.

    Returns dict:
      is_compatible_wav: bool
      info:   WavInfo | None
      reason: str
    """
    if not key.lower().endswith(".wav"):
        return {
            "is_compatible_wav": False,
            "info":   None,
            "reason": f"not a WAV file (extension: {os.path.splitext(key)[1] or 'none'})",
        }

    try:
        resp = _s3.get_object(
            Bucket=bucket, Key=key,
            Range=f"bytes=0-{_HEADER_READ_SIZE - 1}",
        )
        raw = resp["Body"].read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read source from s3://{bucket}/{key}: {exc}"
        ) from exc

    try:
        info = parse_wav_header(raw)
    except ValueError as exc:
        return {"is_compatible_wav": False, "info": None, "reason": str(exc)}

    # Must be mono channel
    if info.num_channels != 1:
        return {
            "is_compatible_wav": False,
            "info":   info,
            "reason": f"WAV has {info.num_channels} channels (need mono)",
        }

    # Must be exactly 48 kHz — Bandit v2 requires 48 kHz input
    if info.sample_rate != _REQUIRED_SAMPLE_RATE:
        return {
            "is_compatible_wav": False,
            "info":   info,
            "reason": f"WAV sample_rate={info.sample_rate} Hz != {_REQUIRED_SAMPLE_RATE} Hz",
        }

    return {"is_compatible_wav": True, "info": info, "reason": "compatible WAV"}


def build_media_processing_request(
    src_bucket:   str,
    src_key:      str,
) -> dict:
    """
    Build the GATEWAY request for media-processing___process_media (the
    family sibling pattern, 2026-09-18 — this stack no longer invokes
    MP's state machine directly). One WAV output group: 48 kHz mono.

    MP writes to ITS OWN service-derived run prefix; per its output
    contract the wav lands at
    {output_s3_prefix}/audiostem_sep/audio_48k_mono.wav, which
    StateExtractProcessedAudio derives deterministically.
    """
    log.info("media-processing request: s3://%s/%s -> wav 48k mono "
             "(gateway-submitted)", src_bucket, src_key)
    return {
        "asset_s3uri": f"s3://{src_bucket}/{src_key}",
        "output_groups": [
            {
                "group_name":     "audiostem_sep",
                "format":         "wav",
                "name":           "audio_48k_mono",
                "sampling_rate":  48000,
                "audio_channels": 1,
                "codec":          "pcm_s16le",
            },
        ],
    }