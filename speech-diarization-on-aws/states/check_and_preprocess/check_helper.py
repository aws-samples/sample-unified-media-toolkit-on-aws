"""
Helpers for StateCheckAndPreprocess.

check_source(bucket, key)
    Probe the S3 object to determine whether it is a compatible WAV.
    Non-WAV extension → returns incompatible without fetching.
    WAV extension → fetches 64 KB header and validates: mono, ≥ 16 kHz, PCM/float.
    Returns {"is_compatible_wav": bool, "info": WavInfo | None, "reason": str}.

build_media_processing_request(src_bucket, src_key, use_partial, mark_in_out)
    Build the GATEWAY request for media-processing___process_media.
    Requests one WAV output group named "diarization_audio":
      format=wav, name=audio_16k_mono, sampling_rate=16000, channels=1, codec=pcm_s16le
    MP writes to ITS OWN run prefix; StateExtractProcessedAudio derives
    {output_s3_prefix}/diarization_audio/audio_16k_mono.wav from the handle.
"""
from __future__ import annotations

import logging
import os
import math

from lib.s3_utils import s3 as _s3
from lib.wav_utils import parse_wav_header

log = logging.getLogger(__name__)

_MIN_SAMPLE_RATE  = 16_000
_HEADER_READ_SIZE = 65_536   # 64 KB — same as wav_utils._HEADER_READ_SIZES[-1]


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

    if info.num_channels != 1:
        return {
            "is_compatible_wav": False,
            "info":   info,
            "reason": f"WAV has {info.num_channels} channels (need mono)",
        }
    if info.sample_rate != _MIN_SAMPLE_RATE:
        # EXACT rate required (was >=, 2026-09-29): a 48 kHz mono WAV
        # used to fast-path "as-is", so every downstream state decoded
        # 720 s windows at 48 kHz and resampled per read — 3x the
        # working set, OOM'd the VAD Lambdas on 2 h speech-dense
        # masters — and the wav-normalise converter (the one place
        # with capped-memory, anti-aliased conversion) never ran.
        # Routing non-16 kHz WAVs to conversion matches audiostem's
        # sibling predicate (exactly 48 kHz there).
        return {
            "is_compatible_wav": False,
            "info":   info,
            "reason": f"WAV sample_rate={info.sample_rate} Hz != {_MIN_SAMPLE_RATE} Hz",
        }

    return {"is_compatible_wav": True, "info": info, "reason": "compatible WAV"}


def build_media_processing_request(
    src_bucket:   str,
    src_key:      str,
    use_partial:  bool,
    mark_in_out:  list,
) -> dict:
    """
    Build the GATEWAY request for media-processing___process_media (the
    family sibling pattern, 2026-09-18 — this stack no longer invokes
    MP's state machine directly). One WAV output group: 16 kHz mono.

    MP writes to ITS OWN service-derived run prefix; per its output
    contract the wav lands at
    {output_s3_prefix}/diarization_audio/audio_16k_mono.wav, which
    StateExtractProcessedAudio derives deterministically.
    """
    request: dict = {
        "asset_s3uri": f"s3://{src_bucket}/{src_key}",
        "output_groups": [
            {
                "group_name":     "diarization_audio",
                "format":         "wav",
                "name":           "audio_16k_mono",
                "sampling_rate":  16000,
                "audio_channels": 1,
                "codec":          "pcm_s16le",
            },
        ],
    }
    if use_partial:
        request["use_partial"] = True
        request["mark_in_out"] = mark_in_out
    log.info("media-processing request: s3://%s/%s -> wav 16k mono "
             "(gateway-submitted, partial=%s)", src_bucket, src_key,
             use_partial)
    return request

def resolve_mark_in_out(use_partial, mark_in_out, duration_sec):
    if not use_partial:
        return use_partial, [0, 0] # Ignore mark_in_out settings

    mark_in, mark_out = mark_in_out
    content_duration = math.floor(duration_sec + 0.5)

    # duration <10s, no need to clip partially
    if content_duration <= 10:
        return False, [0, 0]

    # mark_in must be <duration and has >10s before the duration end
    if (mark_in - content_duration) >= 0:
        raise ValueError(f"Invalid mark_in_out: {mark_in} >= {duration_sec}")

    if (content_duration - mark_in) < 10:
        raise ValueError(f"Duration <10s: content_duration - mark_in = {content_duration - mark_in}")

    # adjust mark_out accordingly
    mark_out = min(mark_out, content_duration)
    if (mark_out - mark_in) < 10:
        raise ValueError(f"Duration <10s: mark_out - mark_in = {mark_out - mark_in}")

    log.info(
        "Use Partial Clip: %ds - %ds [%ds] (Content duration: %d)",
        mark_in, mark_out, mark_out - mark_in, content_duration
    )

    return use_partial, [mark_in, mark_out]
