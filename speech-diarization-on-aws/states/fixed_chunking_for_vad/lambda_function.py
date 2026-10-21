# fix_chunking_for_vad_handler.py
"""
StateFixedChunkingForVAD — divides audio into fixed chunks for the VAD Map state.

Runs after StateCheckAndPreprocess (fast-path) or StateExtractProcessedAudio
(conversion path).  Requires data.audio_bucket and data.audio_key to be set
by the preceding state.

Reads the WAV header from S3 (single 4 KB or 64 KB byte-range GET — no audio
transferred).  If audio_cfg is already fully populated, skips the S3 read.
Divides [0, duration_sec] into ceil(duration / fixed_chunk_size_secs) chunks
and injects data.iterators into the event for the downstream VAD Map state.

No ML models are loaded. Completes in < 2 s for any file.

Output iterator shape (each item is self-contained for the VAD Lambda):
  {
    "chunk_idx":       int,
    "fetch_start_sec": float,    — no overlap; exact boundaries
    "fetch_end_sec":   float,
    "segments_key":    null,     — filled in by StateRunVADModel
    "source_chunks":   [int],
    "audio_bucket":    str,
    "audio_key":       str,
    "output_bucket":   str,
    "output_prefix":   str,
    "audio_cfg":       dict,
    "model_cfg":       dict,
    "vad_cfg":         dict,
  }

Lambda configuration
─────────────────────
  Memory  : 896 MB (HalfCore)
  Timeout : 30 s
  /tmp    : not needed
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback
from lib.wav_utils import get_wav_info

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateFixedChunkingForVAD event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Unpack event ──────────────────────────────────────────────────────
        data = event["data"]
        audio_bucket = data["audio_bucket"]
        audio_key = data["audio_key"]
        output_bucket = data["output_bucket"]
        output_prefix = data["output_prefix"]

        if not audio_bucket or not audio_key:
            raise ValueError(
                f"data.audio_bucket and data.audio_key are required "
                f"(got audio_bucket={audio_bucket!r}, audio_key={audio_key!r})"
            )

        # Get audio config
        audio_cfg = data["audio_cfg"]
        duration_sec = audio_cfg.get("duration_sec")
        sample_rate = audio_cfg.get("sample_rate")
        num_channels = audio_cfg.get("num_channels")

        # ── Resolve audio duration ────────────────────────────────────────────
        if not (duration_sec and sample_rate and num_channels):
            wav_info = get_wav_info(audio_bucket, audio_key)
            duration_sec = wav_info.duration_sec
            sample_rate = wav_info.sample_rate
            num_channels = wav_info.num_channels

        # Update data.audio_cfg
        audio_cfg = {
            **audio_cfg,
            "duration_sec": round(duration_sec, 3),
            "sample_rate": int(sample_rate),
            "num_channels": int(num_channels),
        }

        vad_cfg = data["vad_cfg"]
        model_cfg = data["model_cfg"]

        # ── Build iterators ───────────────────────────────────────────────────
        iterators = _build_iterators(
            audio_bucket=audio_bucket,
            audio_key=audio_key,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            audio_cfg=audio_cfg,
            vad_cfg=vad_cfg,
            model_cfg=model_cfg,
        )

        fixed_chunk_size_secs: float = float(vad_cfg["fixed_chunk_size_secs"])

        log.info(
            f"Audio: {duration_sec:.1f}s  "
            f"sr={sample_rate}  ch={num_channels}  "
            f"chunk_size={fixed_chunk_size_secs}s"
            f"Iterators: {len(iterators)} chunks"
        )

        # ── Build output event ────────────────────────────────────────────────
        output = {
            **event,
            "data": {
                **data,
                "audio_cfg": audio_cfg,
                "iterators": iterators,
            },
        }
        log.info("StateFixedChunkingForVAD output keys = %s",
                 sorted(output.keys()))
        return output

    except Exception as exc:
        log.error(
            "StateFixedChunkingForVAD failed — %s: %s",
            type(exc).__name__, exc
        )
        traceback.print_exc()
        raise


# =============================================================================
# ITERATOR BUILDER
# =============================================================================

def _build_iterators(
    audio_bucket:    str,
    audio_key:       str,
    output_bucket:   str,
    output_prefix:   str,
    audio_cfg:       dict,
    vad_cfg:         dict,
    model_cfg:       dict,
) -> list[dict]:
    """
    Divide [0, duration_sec] into fixed chunks of chunk_size_secs.

    Each iterator item is self-contained — it carries all the context the
    VAD Lambda needs to run independently without access to the parent event.

    The last chunk may be shorter than chunk_size_secs.
    fetch_start_sec and fetch_end_sec are exact — no overlap is added here.
    Overlap is added later by StateDynamicChunkingBySpeechTurns based on
    actual speech turn boundaries.
    """

    duration_sec = audio_cfg["duration_sec"]
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0 (got {duration_sec})")

    fixed_chunk_size_secs = float(vad_cfg["fixed_chunk_size_secs"])
    if fixed_chunk_size_secs <= 0:
        raise ValueError(
            f"chunk_size_secs must be > 0 (got {fixed_chunk_size_secs})"
        )

    n_chunks = math.ceil(duration_sec / fixed_chunk_size_secs)
    iterators = []

    for i in range(n_chunks):
        fetch_start_sec = round(i * fixed_chunk_size_secs, 3)
        fetch_end_sec = round(
            min((i + 1) * fixed_chunk_size_secs, duration_sec), 3)
        iterators.append({
            # ── Chunk boundaries ──────────────────────────────────────────
            "chunk_idx":       i,
            "fetch_start_sec": fetch_start_sec,
            "fetch_end_sec":   fetch_end_sec,
            "source_chunks":   [i],

            # ── Filled in by VAD Lambda ───────────────────────────────────
            "segments_key":    None,

            # ── Audio source ──────────────────────────────────────────────
            "audio_bucket":    audio_bucket,
            "audio_key":       audio_key,

            # ── Output locations ──────────────────────────────────────────
            "output_bucket":   output_bucket,
            "output_prefix":   output_prefix,

            # ── Configurations ────────────────────────────────────────────
            "audio_cfg":       audio_cfg,
            "model_cfg":       model_cfg,
            "vad_cfg":         vad_cfg,
        })

    return iterators
