# states/fixed_chunking_for_speech_enhancement/lambda_function.py
"""
Step Function State: Fix chunking for speech enhancement
────────────────────────────────────────────────────────
Reads the WAV header from S3 (byte-range GET, no audio data
transferred), computes iterator boundaries aligned to the enhancement
chunk alignment (10 s — a whole multiple of DFN3's STFT hop), and
writes data.iterators for the Map state.

Chunking rules (ported from the original workflow)
──────────────────────────────────────────────────
• fixed_chunk_size_secs is rounded to the NEAREST multiple of 10 s,
  ties breaking downward; minimum effective chunk size is 10 s.
• The last iterator is clamped to the file duration — never padded.
• Seam handling is NOT here: the model state fetches ±1 s context
  around each window and trims it after inference, so these boundaries
  only decide parallelism, not audio quality.
• audio_cfg is not spread into iterators — duration_sec feeds
  _build_iterators directly to keep the per-iterator payload minimal.

Expected data fields (written by StateCheckAndPreprocess):
  audio_bucket, audio_key, output_bucket, output_prefix,
  fixed_chunk_size_secs, output_resampl_freq, atten_lim_db, audio_cfg

Output: data gains aligned_chunk_secs + iterators[], each iterator:
  { chunk_idx, chunk_name, fetch_start_sec, fetch_end_sec,
    audio_bucket, audio_key, output_bucket, output_prefix,
    enhancement: {output_resampl_freq, atten_lim_db} }
"""
from __future__ import annotations

import logging
import math
import time

from lib.input_schema import CHUNK_ALIGNMENT_SECS
from lib.wav_utils import get_wav_info

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)


def _align_chunk_size(secs: float,
                      alignment: int = CHUNK_ALIGNMENT_SECS) -> int:
    """Nearest multiple of `alignment`, ties downward, floor one unit."""
    aligned = math.ceil((secs - alignment / 2) / alignment) * alignment
    return max(aligned, alignment)


def _build_iterators(item_cfg: dict, aligned_chunk_secs: int,
                     duration_sec: float) -> list[dict]:
    iterators = []
    start, idx = 0.0, 0
    while start < duration_sec:
        end = min(start + aligned_chunk_secs, duration_sec)
        iterators.append({
            "chunk_idx": idx,
            "chunk_name": f"chunk_{idx:03d}",
            "fetch_start_sec": round(start, 3),
            "fetch_end_sec": round(end, 3),
            **item_cfg,
        })
        start += aligned_chunk_secs
        idx += 1
    return iterators


def lambda_handler(event: dict, context) -> dict:
    log.info("StateFixedChunkingForSpeechEnhancement event keys = %s",
             sorted(event.keys()))
    t0 = time.monotonic()

    data = event.get("data", {})
    audio_bucket = data["audio_bucket"]
    audio_key = data["audio_key"]

    # ── Audio duration (header read only when preprocess couldn't say) ──────
    audio_cfg = data["audio_cfg"] or {}
    duration_sec = audio_cfg.get("duration_sec")
    if duration_sec is None:
        info = get_wav_info(audio_bucket, audio_key)
        audio_cfg = {
            **audio_cfg,
            "sample_rate": info.sample_rate,
            "num_channels": info.num_channels,
            "bits_per_sample": info.bits_per_sample,
            "duration_sec": round(info.duration_sec, 3),
        }
        duration_sec = info.duration_sec
    else:
        duration_sec = float(duration_sec)
        audio_cfg["duration_sec"] = duration_sec
    if duration_sec <= 0:
        raise ValueError(f"Invalid audio duration: {duration_sec}s")

    # ── Chunk alignment ──────────────────────────────────────────────────────
    fixed_chunk_size_secs = int(data["fixed_chunk_size_secs"])
    aligned_chunk_secs = _align_chunk_size(fixed_chunk_size_secs)
    if fixed_chunk_size_secs != aligned_chunk_secs:
        log.info("fixed_chunk_size_secs: %ds → aligned to %ds "
                 "(nearest multiple of %ds)", fixed_chunk_size_secs,
                 aligned_chunk_secs, CHUNK_ALIGNMENT_SECS)

    item_cfg = {
        "audio_bucket": audio_bucket,
        "audio_key": audio_key,
        "output_bucket": data["output_bucket"],
        # chunk WAVs + per-chunk metrics are internal intermediates — keep
        # them out of the run root (the family segment_chunks/ pattern)
        "output_prefix": f"{data['output_prefix'].strip('/')}/enhanced_chunks",
        "enhancement": {
            "output_resampl_freq": int(data["output_resampl_freq"]),
            "atten_lim_db": data.get("atten_lim_db"),
        },
    }
    iterators = _build_iterators(item_cfg, aligned_chunk_secs, duration_sec)

    log.info("Built %d iterator(s)  aligned_chunk_secs=%ds  "
             "duration=%.1fs  elapsed=%.2fs", len(iterators),
             aligned_chunk_secs, duration_sec, time.monotonic() - t0)

    return {**event, "data": {**data, "audio_cfg": audio_cfg,
                              "aligned_chunk_secs": aligned_chunk_secs,
                              "iterators": iterators}}
