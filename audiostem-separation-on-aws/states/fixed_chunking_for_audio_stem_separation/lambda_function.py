# fixed_chunking_handler.py
"""
Step Function State: Fix chunking for audio stem separation
────────────────────────────────────────────────────────────
Reads the WAV header from S3 (byte-range GET, no audio data transferred),
computes iterator boundaries aligned to Bandit v2's 8-second internal chunk
size, and writes data.iterators into the state machine payload.

Expected input
──────────────
{
    "operation": "StateFixedChunkingForAudioStemSeparation",
    "uuid":      "...",
    "input":     { ... },
    "output":    { ... },
    "data": {
        "audio_bucket":          "...",
        "audio_key":             "path/to/audio.wav",
        "output_bucket":         "...",
        "output_prefix":         "path/to/output/prefix",
        "fixed_chunk_size_secs": 180,
        "speech_stem":           true,   "speech_resampl_freq": 16000,
        "music_stem":            true,   "music_resampl_freq":  48000,
        "sfx_stem":              true,   "sfx_resampl_freq":    32000,
        "audio_cfg": {
            "duration_sec":    null,   ← null → determined from WAV header
            "sample_rate":     48000,
            "num_channels":    1,
            "bits_per_sample": 16
        }
    }
}

Output
──────
{
    ...original payload...,
    "data": {
        ...original data...,
        "audio_cfg":          { "duration_sec": 3600.0, ... },  ← filled if was null
        "aligned_chunk_secs": 176,
        "iterators": [
            {
                "chunk_idx":          0,
                "chunk_name":         "chunk_000",
                "fetch_start_sec":    0.0,
                "fetch_end_sec":      176.0,
                "audio_bucket":       "...",
                "audio_key":          "path/to/audio.wav",
                "output_bucket":      "...",
                "output_prefix":      "path/to/output/prefix",
                "speech_stem":        true,   "speech_resampl_freq": 16000,
                "music_stem":         true,   "music_resampl_freq":  48000,
                "sfx_stem":           true,   "sfx_resampl_freq":    32000,
                "speech_safe":        false,  "speech_safe_resampl_freq": 48000
            },
            ...
        ]
    }
}

Chunking rules
──────────────
• fixed_chunk_size_secs is rounded to the nearest multiple of 8 (Bandit's
  internal chunk size). Ties break downward:
      aligned = ceil((secs - 4) / 8) * 8
• Minimum effective chunk size is 8 seconds.
• The last iterator is clamped to the actual file duration — not padded.
• audio_cfg is not spread into iterators — duration_sec is passed to
  _build_iterators directly to keep the per-iterator payload minimal.
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback

from lib.input_schema import CHUNK_ALIGMENT_SECS, DEFAULT_SPEECH_SAFE_FREQ
from lib.wav_utils import get_wav_info

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
# HELPERS
# =============================================================================

def _align_chunk_size(secs: float, chunk_alignment_secs: int = CHUNK_ALIGMENT_SECS) -> int:
    """
    Round secs to the nearest multiple of chunk_alignment_secs.
    Tie-breaks downward (e.g. 244 → 240, not 248).
    Minimum return value is chunk_alignment_secs.

    Formula: ceil((secs - chunk_alignment_secs/2) / chunk_alignment_secs) * chunk_alignment_secs
    Examples (chunk_alignment_secs=8):
        180 → 176?  no, 180/8=22.5 → ceil((180-4)/8)*8 = ceil(22.0)*8 = 176? wrong
        Let's verify:
            180 → ceil((180-4)/8)*8 = ceil(176/8)*8 = ceil(22.0)*8 = 22*8 = 176
        Wait — 180 is itself a multiple of... no. 180/8 = 22.5.
        Nearest multiples: 176 (22*8) and 184 (23*8).
        Distance to 176 = 4, distance to 184 = 4 → tie → round down → 176.

        But the default is 180 which should survive as-is (it's not a multiple
        of 8). We align it → 176.

        238 → ceil((238-4)/8)*8 = ceil(234/8)*8 = ceil(29.25)*8 = 30*8 = 240 ✓
        242 → ceil((242-4)/8)*8 = ceil(238/8)*8 = ceil(29.75)*8 = 30*8 = 240 ✓
        244 → ceil((244-4)/8)*8 = ceil(240/8)*8 = ceil(30.0)*8  = 30*8 = 240 ✓ (tie→down)
        248 → ceil((248-4)/8)*8 = ceil(244/8)*8 = ceil(30.5)*8  = 31*8 = 248 ✓ (exact)
        180 → ceil((180-4)/8)*8 = ceil(176/8)*8 = ceil(22.0)*8  = 22*8 = 176
    """
    aligned = math.ceil((secs - chunk_alignment_secs / 2) /
                        chunk_alignment_secs) * chunk_alignment_secs
    return max(aligned, chunk_alignment_secs)


def _build_iterators(
    item_cfg: dict,
    aligned_chunk_secs: int,
    duration_sec: float,
) -> list[dict]:
    """
    Build the list of iterator dicts for the Map state.

    Each iterator covers [fetch_start_sec, fetch_end_sec) where the last
    iterator is clamped to duration_sec.
    """
    iterators = []
    start = 0.0
    idx = 0

    while start < duration_sec:
        end = min(start + aligned_chunk_secs, duration_sec)
        chunk_name = f"chunk_{idx:03d}"

        iterators.append({
            "chunk_idx":       idx,
            "chunk_name":      chunk_name,
            "fetch_start_sec": round(start, 3),
            "fetch_end_sec":   round(end,   3),
            **item_cfg,
        })

        start += aligned_chunk_secs
        idx += 1

    return iterators


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateFixedChunkingForAudioStemSeparation event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Unpack ────────────────────────────────────────────────────────────
        data = event.get("data",  {})

        audio_bucket = data["audio_bucket"]
        audio_key = data["audio_key"]
        output_bucket = data["output_bucket"]
        output_prefix = data["output_prefix"]
        model_id = data["model_id"]

        # ── Stem flags ────────────────────────────────────────────────────────
        speech_stem = bool(data["speech_stem"])
        speech_resampl_freq = int(data["speech_resampl_freq"])

        music_stem = bool(data["music_stem"])
        music_resampl_freq = int(data["music_resampl_freq"])

        sfx_stem = bool(data["sfx_stem"])
        sfx_resampl_freq = int(data["sfx_resampl_freq"])

        speech_safe = bool(data.get("speech_safe", False))
        speech_safe_resampl_freq = int(
            data.get("speech_safe_resampl_freq", DEFAULT_SPEECH_SAFE_FREQ))

        if not any([speech_stem, music_stem, sfx_stem, speech_safe]):
            raise ValueError(
                "audiostem_sep: at least one output must be enabled "
                "(speech_stem, music_stem, sfx_stem, speech_safe)"
            )

        # ── Audio duration ────────────────────────────────────────────────────
        # Use pre-supplied duration if available; otherwise read WAV header.
        audio_cfg = data["audio_cfg"]

        duration_sec = audio_cfg.get("duration_sec", None)
        if duration_sec is None:
            log.info(f"duration_sec not supplied — reading WAV header from S3")
            info = get_wav_info(audio_bucket, audio_key)
            audio_cfg = {
                **audio_cfg,
                "sample_rate":     info.sample_rate,
                "num_channels":    info.num_channels,
                "bits_per_sample": info.bits_per_sample,
                "duration_sec":    round(info.duration_sec, 3),
            }
            duration_sec = info.duration_sec
            log.info(f"WAV duration: {duration_sec:.2f}s")
        else:
            duration_sec = float(duration_sec)
            audio_cfg["duration_sec"] = duration_sec
            log.info(f"duration_sec supplied: {duration_sec:.2f}s")

        if duration_sec <= 0:
            raise ValueError(f"Invalid audio duration: {duration_sec}s")

        # ── Chunk size alignment ───────────────────────────────────────────────
        fixed_chunk_size_secs = int(data["fixed_chunk_size_secs"])
        aligned_chunk_secs = _align_chunk_size(fixed_chunk_size_secs)
        if fixed_chunk_size_secs != aligned_chunk_secs:
            log.info(
                f"fixed_chunk_size_secs: {fixed_chunk_size_secs}s → aligned to "
                f"{aligned_chunk_secs}s (nearest multiple of {CHUNK_ALIGMENT_SECS}s)"
            )
        else:
            log.info(
                f"fixed_chunk_size_secs: {aligned_chunk_secs}s (already aligned)")

        # ── Build iterators ───────────────────────────────────────────────────
        data["audio_cfg"] = audio_cfg
        item_cfg = {
            "audio_bucket":          audio_bucket,
            "audio_key":             audio_key,
            "output_bucket":         output_bucket,
            "output_prefix":         output_prefix,
            "speech_stem":           speech_stem,
            "speech_resampl_freq":   speech_resampl_freq,
            "music_stem":            music_stem,
            "music_resampl_freq":    music_resampl_freq,
            "sfx_stem":              sfx_stem,
            "sfx_resampl_freq":      sfx_resampl_freq,
            "speech_safe":           speech_safe,
            "speech_safe_resampl_freq": speech_safe_resampl_freq,
        }
        iterators = _build_iterators(item_cfg, aligned_chunk_secs, duration_sec)

        n = len(iterators)
        elapsed = round(time.monotonic() - t0, 2)
        log.info(
            f"Built {n} iterator(s)  "
            f"aligned_chunk_secs={aligned_chunk_secs}s  "
            f"duration={duration_sec:.1f}s  "
            f"stems=[speech={speech_stem}, music={music_stem}, sfx={sfx_stem}]  "
            f"elapsed={elapsed}s"
        )
        for it in iterators:
            log.debug(
                f"  [{it['chunk_name']}] "
                f"{it['fetch_start_sec']:.1f}s → {it['fetch_end_sec']:.1f}s  "
                f"({it['fetch_end_sec'] - it['fetch_start_sec']:.1f}s)"
            )

        # ── Return enriched payload ───────────────────────────────────────────
        return {
            **event,
            "data": {
                **data,
                "model_id": model_id,
                "aligned_chunk_secs": aligned_chunk_secs,
                "iterators": iterators,
            },
        }

    except Exception as exc:
        log.error(
            "[ERR] StateFixedChunkingForAudioStemSeparation — %s: %s",
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise
