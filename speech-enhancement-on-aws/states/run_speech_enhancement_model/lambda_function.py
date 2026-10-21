# states/run_speech_enhancement_model/lambda_function.py
"""
Step Function State: Run speech enhancement model
─────────────────────────────────────────────────
Map state — one Lambda invocation per chunk. Fetches the audio window
from S3 (with context padding), runs DeepFilterNet3 enhancement,
resamples to the target output rate, and uploads the enhanced chunk
WAV + per-chunk metrics to S3.

Model loading
─────────────
  Cold container : loads the ONNX graphs from /opt/models/ (baked into
                   the image at docker build, sha256-pinned — no S3
                   download, no CDN; the v5 model_store pattern is
                   deliberately retired).
  Warm container : module-level singleton reuse.

Seam handling lives HERE, not at concat: each chunk fetches
±CONTEXT_*_SEC beyond its window, the model processes the padded audio,
and only [fetch_start_sec, fetch_end_sec] is uploaded — so adjacent
chunks concatenate without boundary artifacts beyond the model's own
2-frame lookahead. S3 byte-range reads beyond EOF return available
bytes silently, so the first/last chunks need no special case.

Expected input  (from Map state ItemSelector)
─────────────────────────────────────────────
{
    "operation": "StateRunSpeechEnhancementModel",
    "item_data": {
        "chunk_idx":       0,
        "chunk_name":      "chunk_000",
        "fetch_start_sec": 0.0,
        "fetch_end_sec":   360.0,
        "audio_bucket":    "...",
        "audio_key":       "path/to/audio.wav",
        "output_bucket":   "...",
        "enhancement": {
            "output_resampl_freq": 16000,
            "atten_lim_db":        null
        },
        "output_prefix":   "path/to/enhanced_chunks"
    }
}

S3 output
──────────
  {output_prefix}/chunk_000.wav    enhanced audio at output_resampl_freq
                                   (trimmed to the exact fetch window)
  {output_prefix}/chunk_000.json   per-chunk metrics

Returns the enriched iterator item (ResultPath replaces the item):
  { ...item_data..., "enhanced": {s3_key, resampl_freq, rms_db_in,
    rms_db_out, duration_sec}, "timing_ms": {...}, "metrics_key": ...,
    "status": "COMPLETED", "elapsed_sec": ... }
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import boto3
import numpy as np

# ── ORT cpuinfo fix — must precede onnxruntime import ────────────────────────
from lib.ort_init import ensure_cpuinfo_files, get_num_cpus
ensure_cpuinfo_files()
# ─────────────────────────────────────────────────────────────────────────────

from lib.model_store import find_file                               # noqa: E402
from lib.wav_utils import (                                         # noqa: E402
    encode_wav_mono, get_wav_info, read_wav_window, resample, rms_db,
)
from states.run_speech_enhancement_model.model_inference import (   # noqa: E402
    DFN3Enhancer,
)

_s3 = boto3.client("s3")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)

# ── DFN3 constants ────────────────────────────────────────────────────────────
_DFN_SR = 48_000
_DEFAULT_OUT_SR = 16_000
_MODEL_DIR = "/opt/models"

# Context padding to avoid boundary artifacts at chunk edges (processed
# by the model, trimmed from the upload — see the module docstring).
_CONTEXT_LEFT_SEC = 1.0
_CONTEXT_RIGHT_SEC = 1.0

# Module-level singleton — the model is immutable in the image, so the
# cache never invalidates (unlike the retired ETag-checked S3 store).
_cached_enhancer: Optional[DFN3Enhancer] = None


def _get_enhancer(memory_mb: Optional[int]) -> DFN3Enhancer:
    global _cached_enhancer
    if _cached_enhancer is not None:
        return _cached_enhancer
    enc_path = find_file(_MODEL_DIR, "enc.onnx", required=True)
    model_dir = os.path.dirname(enc_path)
    num_cpus = get_num_cpus(memory_mb)
    log.info("Loading DFN3 from %s  intra_op_num_threads=%d",
             model_dir, num_cpus)
    _cached_enhancer = DFN3Enhancer(model_dir, num_threads=num_cpus)
    return _cached_enhancer


def lambda_handler(event: dict, context) -> dict:
    # keys only — the values carry customer-named S3 keys
    log.info("StateRunSpeechEnhancementModel event keys = %s",
             sorted(event.keys()))
    t0 = time.monotonic()

    item = event.get("item_data", event)
    required = {"chunk_idx", "chunk_name", "fetch_start_sec",
                "fetch_end_sec", "audio_bucket", "audio_key",
                "enhancement", "output_bucket", "output_prefix"}
    missing = required - set(item)
    if missing:
        raise ValueError(f"Missing item_data fields: {sorted(missing)}")

    chunk_idx = int(item["chunk_idx"])
    chunk_name = item["chunk_name"]
    fetch_start_sec = float(item["fetch_start_sec"])
    fetch_end_sec = float(item["fetch_end_sec"])
    fetch_duration = fetch_end_sec - fetch_start_sec
    audio_bucket = item["audio_bucket"]
    audio_key = item["audio_key"]
    output_bucket = item["output_bucket"]
    output_prefix = item["output_prefix"].strip("/")
    cfg = item["enhancement"]

    output_sr = int(cfg.get("output_resampl_freq", _DEFAULT_OUT_SR))
    atten_lim = cfg.get("atten_lim_db")
    if atten_lim is not None:
        atten_lim = float(atten_lim)

    log.info("Chunk %d (%s): [%.1fs → %.1fs] (%.1fs)  output_sr=%d  "
             "atten_lim=%s", chunk_idx, chunk_name, fetch_start_sec,
             fetch_end_sec, fetch_duration, output_sr, atten_lim)

    # ── Model (baked, cached) ─────────────────────────────────────────────────
    t_model = time.monotonic()
    enhancer = _get_enhancer(getattr(context, "memory_limit_in_mb", None))
    model_load_ms = round((time.monotonic() - t_model) * 1_000, 1)

    # ── WAV header + padded window read (conformed to the model rate) ────────
    t_hdr = time.monotonic()
    info = get_wav_info(audio_bucket, audio_key)
    hdr_ms = round((time.monotonic() - t_hdr) * 1_000, 1)

    padded_start = max(0.0, fetch_start_sec - _CONTEXT_LEFT_SEC)
    padded_end = fetch_end_sec + _CONTEXT_RIGHT_SEC

    t_read = time.monotonic()
    audio_padded = read_wav_window(
        audio_bucket, audio_key, info,
        start_sec=padded_start,
        duration_sec=padded_end - padded_start,
        target_sr=_DFN_SR,
    )
    s3_read_ms = round((time.monotonic() - t_read) * 1_000, 1)

    n_padded = len(audio_padded)
    if n_padded == 0:
        raise RuntimeError(
            f"Empty audio for chunk {chunk_idx} "
            f"[{fetch_start_sec:.1f}s → {fetch_end_sec:.1f}s]")

    # RMS measured on the exact requested window, not the padding
    trim_start = int((fetch_start_sec - padded_start) * _DFN_SR)
    trim_end = min(trim_start + int(fetch_duration * _DFN_SR), n_padded)
    rms_in = rms_db(audio_padded[trim_start:trim_end])

    log.info("Read: hdr %.0fms + window %.0fms  padded=%.2fs  "
             "rms_in=%.1f dBFS", hdr_ms, s3_read_ms,
             n_padded / _DFN_SR, rms_in)

    # ── Enhance (on padded audio), trim, resample ─────────────────────────────
    t_dfn = time.monotonic()
    enhanced_padded = enhancer.enhance(audio_padded, atten_lim_db=atten_lim)
    dfn_ms = round((time.monotonic() - t_dfn) * 1_000, 1)
    rtf = (time.monotonic() - t_dfn) / (n_padded / _DFN_SR)

    enhanced_48k = enhanced_padded[trim_start:trim_end]
    rms_out = rms_db(enhanced_48k)
    log.info("DFN3: %.0fms  RTF=%.2fx  rms %.1f → %.1f dBFS",
             dfn_ms, rtf, rms_in, rms_out)

    # Anti-aliased polyphase resample — the family resampler. The
    # legacy linear-interp resampler is the aliasing bug class fixed
    # repo-wide 2026-09-28 and is deliberately NOT ported.
    t_resamp = time.monotonic()
    enhanced = resample(np.asarray(enhanced_48k, dtype=np.float32),
                        _DFN_SR, output_sr)
    resamp_ms = round((time.monotonic() - t_resamp) * 1_000, 1)

    # ── Upload chunk + metrics ────────────────────────────────────────────────
    t_write = time.monotonic()
    wav_bytes = encode_wav_mono(enhanced, output_sr)
    s3_key = f"{output_prefix}/{chunk_name}.wav"
    _s3.put_object(Bucket=output_bucket, Key=s3_key, Body=wav_bytes,
                   ContentType="audio/wav")
    write_ms = round((time.monotonic() - t_write) * 1_000, 1)
    dur_out = round(len(enhanced) / output_sr, 3)

    elapsed_ms = round((time.monotonic() - t0) * 1_000, 1)
    timing_ms = {"model_load": model_load_ms, "s3_read": s3_read_ms,
                 "dfn3": dfn_ms, "resample": resamp_ms,
                 "s3_write": write_ms, "total": elapsed_ms}
    enhanced_doc = {"s3_key": s3_key, "resampl_freq": output_sr,
                    "rms_db_in": rms_in, "rms_db_out": rms_out,
                    # exact PCM length — concat builds the RIFF header from
                    # this, never from the 3-decimal-rounded duration_sec
                    "n_samples": int(len(enhanced)),
                    "duration_sec": dur_out}
    metrics_doc = {
        "chunk_idx": chunk_idx, "chunk_name": chunk_name,
        "fetch_start_sec": fetch_start_sec, "fetch_end_sec": fetch_end_sec,
        "duration_sec": round(fetch_duration, 3),
        "enhanced": enhanced_doc, "timing_ms": timing_ms,
        "rtf": round(rtf, 3), "status": "COMPLETED", "error": None,
    }
    metrics_key = f"{output_prefix}/{chunk_name}.json"
    _s3.put_object(Bucket=output_bucket, Key=metrics_key,
                   Body=json.dumps(metrics_doc, indent=2).encode(),
                   ContentType="application/json")

    elapsed_sec = round(time.monotonic() - t0, 2)
    log.info("DONE chunk=%d  elapsed=%.2fs  rtf=%.2fx",
             chunk_idx, elapsed_sec, rtf)
    return {**item, "enhanced": enhanced_doc, "timing_ms": timing_ms,
            "metrics_key": metrics_key, "status": "COMPLETED",
            "elapsed_sec": elapsed_sec}
