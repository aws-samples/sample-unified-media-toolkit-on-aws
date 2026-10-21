# run_bandit_v2_handler.py
"""
Step Function State: Run audio stem separation model  (Map state — one Lambda per iterator)
────────────────────────────────────────────────────────────────────────────────────────────
Each invocation receives one iterator's parameters, fetches that audio slice
from S3 via byte-range GET, runs Bandit v2 ONNX (encoder + requested decoder
heads) over the slice in 8-second internal chunks with 1-second hop (OLA),
resamples each stem, uploads stem WAV files and a metrics JSON to S3.

Expected input event  (from Map ItemSelector)
──────────────────────────────────────────────
{
    "operation": "StateRunBanditV2Model",
    "item_id":    0,
    "item_data": {
        "chunk_idx":       0,
        "chunk_name":      "chunk_000",
        "fetch_start_sec": 0.0,
        "fetch_end_sec":   180.0,
        "audio_bucket":    "...",
        "audio_key":       "path/to/audio_en.wav",
        "output_bucket":   "..."
        "output_prefix":   "path/to/output/prefix"
        "speech_stem":           true,
        "music_stem":            true,
        "sfx_stem":              true,
        "speech_resampl_freq":   16000,
        "sfx_resampl_freq":      32000,
        "music_resampl_freq":    48000
    }
}

Output  (returned to Step Functions; replaces iterator item via ResultPath)
───────────────────────────────────────────────────────────────────────────
{
    ...all item_data fields...,
    "stems": {
        "speech": { "s3_key": "...", "resampl_freq": 16000,
                    "rms_db": -28.4, "duration_sec": 180.0 },
        "music":  { ... },
        "sfx":    { ... }
    },
    "timing_ms": {
        "s3_read": 320, "encoder": 11200,
        "speech_decoder": 540, "music_decoder": 510, "sfx_decoder": 490,
        "resample_write": 85, "total": 13145
    },
    "metrics_key": "path/to/audio/audiostem/chunk_000.json",
    "status":      "COMPLETED",
    "elapsed_sec": 13.1
}

Model warm-cache behaviour
──────────────────────────
  Cold container : loads model from /opt/models/ (baked into image)
  Warm container : _sessions cache hit → < 1 ms

S3 output layout
────────────────
  {output_prefix}/segment_chunks/speech/chunk_000.wav
  {output_prefix}/segment_chunks/music/chunk_000.wav
  {output_prefix}/segment_chunks/sfx/chunk_000.wav
  {output_prefix}/segment_chunks/speech_safe/chunk_000.wav  ← when speech_safe=true
  {output_prefix}/segment_chunks/chunk_000.json        ← metrics

Internal Bandit v2 OLA parameters
───────────────────────────────────
  sample_rate  = 48000 Hz
  chunk_size   = 384000 samples (8 s)
  hop_size     = chunk_size (non-overlapping — Bandit handles internal OLA)
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback

import numpy as np

# ── ORT cpuinfo fix — must precede onnxruntime import ────────────────────────
from lib.ort_init import ensure_cpuinfo_files, get_num_cpus
ensure_cpuinfo_files()
import onnxruntime as ort
# ─────────────────────────────────────────────────────────────────────────────

from lib.model_store import find_file
from lib.input_schema import DEFAULT_SPEECH_SAFE_FREQ
from lib.input_schema import TARGET_SR as _BANDIT_SR
from lib.s3_utils import s3 as _s3, put_json
from lib.speech_safe import build_speech_safe
from lib.speech_safe import params_doc as speech_safe_params
from lib.wav_utils import get_wav_info, read_wav_window, encode_wav_mono, rms_db, resample

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

# ── Bandit v2 constants ───────────────────────────────────────────────────────
_BANDIT_CHUNK = 384_000   # 8 s @ 48 kHz
_ALL_STEMS    = ("speech", "music", "sfx")

# ── Model location (baked into image at build time) ───────────────────────────
_MODEL_DIR = "/opt/models"

# ── Module-level session cache (survives warm container reuse) ────────────────
# Keys: frozenset of active stem names → dict of stem → InferenceSession
# e.g. frozenset({"speech","sfx"}) → {"encoder": sess, "speech": sess, "sfx": sess}
_sessions: dict[frozenset, dict[str, ort.InferenceSession]] = {}


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def _get_sessions(
    stems:     tuple[str, ...],
    memory_mb: int | None,
) -> dict[str, ort.InferenceSession]:
    """Load (or reuse) ONNX sessions for encoder + requested decoder heads."""
    global _sessions
    key = frozenset(stems)

    if key in _sessions:
        log.info(f"[cache ✅] sessions reused for stems={sorted(stems)}")
        return _sessions[key]

    num_cpus = get_num_cpus(memory_mb)
    log.info(f"ORT intra_op_num_threads = {num_cpus}  (memory_mb={memory_mb})")

    def _load(path: str, label: str) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_cpus
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t = time.monotonic()
        sess = ort.InferenceSession(
            path, opts, providers=["CPUExecutionProvider"])
        log.info(f"  loaded {label} from {path}: {time.monotonic() - t:.2f}s")
        return sess

    loaded: dict[str, ort.InferenceSession] = {}

    # Always load encoder
    enc_path = (
        find_file(_MODEL_DIR, "encoder.int8.onnx", required=False) or
        find_file(_MODEL_DIR, "encoder.onnx")
    )
    loaded["encoder"] = _load(enc_path, "encoder")

    for stem in stems:
        dec_path = (
            find_file(_MODEL_DIR, f"{stem}_decoder.int8.onnx", required=False) or
            find_file(_MODEL_DIR, f"{stem}_decoder.onnx")
        )
        loaded[stem] = _load(dec_path, f"{stem}_decoder")

    _sessions[key] = loaded
    return loaded


# =============================================================================
# AUDIO HELPERS
# =============================================================================

# =============================================================================
# BANDIT v2 OLA INFERENCE
# =============================================================================

def _run_bandit(
    audio:    np.ndarray,
    sessions: dict[str, ort.InferenceSession],
    stems:    tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Run Bandit v2 OLA inference over a variable-length 48 kHz mono float32
    array. Returns per-stem float32 arrays (same length as input) and per-
    stage timing in milliseconds.

    Inference parameters:
        chunk_size = 384000 samples (8 s) — non-overlapping

    Audio is split into non-overlapping 8s blocks. Bandit's ONNX model handles
    its own internal OLA (1s hop). Last chunk is zero-padded and trimmed.
    """
    n_samples = len(audio)
    chunk_size = _BANDIT_CHUNK   # 384000 = 8s @ 48kHz
    timing: dict[str, float] = {s: 0.0 for s in stems}
    timing["encoder"] = 0.0

    # Output buffers — collect non-overlapping chunk outputs then concatenate
    stem_bufs: dict[str, list[np.ndarray]] = {s: [] for s in stems}

    # Split into non-overlapping 8s chunks.
    # Bandit's ONNX model already performs its own internal OLA with a 1s hop
    # over each 8s input — we must NOT add another OLA layer on top.
    # The last chunk is zero-padded to chunk_size if shorter.
    n_chunks = max(1, math.ceil(n_samples / chunk_size))

    enc_sess = sessions["encoder"]

    for ci in range(n_chunks):
        start   = ci * chunk_size
        end     = min(start + chunk_size, n_samples)
        segment = audio[start:end]
        t_chunk = time.monotonic()

        # Zero-pad last chunk if shorter than chunk_size
        if len(segment) < chunk_size:
            segment = np.concatenate(
                [segment, np.zeros(chunk_size - len(segment), dtype=np.float32)])

        # Input shape: [1, 1, T]
        mixture = segment.reshape(1, 1, -1).astype(np.float32)

        # ── Encoder ───────────────────────────────────────────────────────────
        t_enc = time.monotonic()
        q, x = enc_sess.run(["q", "x"], {"mixture": mixture})
        enc_ms = (time.monotonic() - t_enc) * 1_000
        timing["encoder"] += enc_ms

        # ── Decoder heads ─────────────────────────────────────────────────────
        dec_ms: dict[str, float] = {}
        actual_len = end - start   # samples of real audio in this chunk
        for stem in stems:
            t_dec = time.monotonic()
            out = sessions[stem].run(
                [stem], {"q": q, "x": x})[0]      # [1, 1, chunk_size]
            dec_ms[stem] = (time.monotonic() - t_dec) * 1_000
            timing[stem] += dec_ms[stem]

            # Trim padding from last chunk before appending
            stem_bufs[stem].append(out[0, 0, :actual_len].astype(np.float32))

        chunk_ms  = (time.monotonic() - t_chunk) * 1_000
        dec_parts = "  ".join(f"{s}={dec_ms[s]:.0f}" for s in stems)
        log.info(
            f"  chunk [{ci+1:3d}/{n_chunks}] "
            f"t={start//_BANDIT_SR:.1f}s-{end//_BANDIT_SR:.1f}s  "
            f"enc={enc_ms:.0f}ms  {dec_parts}  "
            f"total={chunk_ms:.0f}ms"
        )

    # ── Concatenate chunks ────────────────────────────────────────────────────
    results: dict[str, np.ndarray] = {}
    for stem in stems:
        buf = np.concatenate(stem_bufs[stem], axis=0)
        results[stem] = np.clip(buf, -1.0, 1.0).astype(np.float32)

    # Round timings
    timing = {k: round(v, 1) for k, v in timing.items()}

    dec_summary = "  ".join(
        f"{s}={timing[s]:.0f}ms" for s in stems)
    log.info(
        f"  inference done: {n_chunks} chunks  "
        f"enc_total={timing['encoder']:.0f}ms  {dec_summary}"
    )

    return results, timing


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateRunAudioStemSeparationModel event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Unpack item_data ───────────────────────────────────────────────────
        item = event["item_data"]

        required = {
            "chunk_idx", "chunk_name", "fetch_start_sec", "fetch_end_sec",
            "audio_bucket", "audio_key", "output_bucket", "output_prefix",
            "speech_stem", "music_stem", "sfx_stem",
            "speech_resampl_freq", "music_resampl_freq", "sfx_resampl_freq",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"Missing item_data fields: {missing}")

        chunk_idx       = int(item["chunk_idx"])
        chunk_name      = item["chunk_name"]
        fetch_start_sec = float(item["fetch_start_sec"])
        fetch_end_sec   = float(item["fetch_end_sec"])
        fetch_duration  = fetch_end_sec - fetch_start_sec
        audio_bucket    = item["audio_bucket"]
        audio_key       = item["audio_key"]
        output_bucket   = item["output_bucket"]
        output_prefix   = item["output_prefix"].strip("/")

        # ── Stem selection ────────────────────────────────────────────────────
        requested_stems = tuple(
            s for s in _ALL_STEMS if item.get(f"{s}_stem", True)
        )
        # speech_safe (gated mix substitution — README.md Design record §5)
        # needs the speech decoder's output even when the raw stem WAV
        # itself was not requested; the raw stem is then not uploaded.
        speech_safe = bool(item.get("speech_safe", False))
        decoder_stems = requested_stems
        if speech_safe and "speech" not in decoder_stems:
            decoder_stems = decoder_stems + ("speech",)
        if not requested_stems and not speech_safe:
            raise ValueError("No stems enabled in config")

        # ── Resampling frequencies ────────────────────────────────────────────
        resampl_freq: dict[str, int] = {
            "speech": item["speech_resampl_freq"],
            "sfx":    item["sfx_resampl_freq"],
            "music":  item["music_resampl_freq"],
            "speech_safe": item.get("speech_safe_resampl_freq",
                                    DEFAULT_SPEECH_SAFE_FREQ),
        }

        log.info(
            f"Chunk {chunk_idx} ({chunk_name}): "
            f"[{fetch_start_sec:.1f}s → {fetch_end_sec:.1f}s]  "
            f"({fetch_duration:.1f}s)  stems={list(requested_stems)}  "
            f"speech_safe={speech_safe}"
        )

        # ── Load / reuse ONNX sessions ────────────────────────────────────────
        memory_mb = getattr(context, "memory_limit_in_mb", None)
        t_model = time.monotonic()
        sessions = _get_sessions(
            stems     = decoder_stems,
            memory_mb = memory_mb,
        )
        model_load_ms = round((time.monotonic() - t_model) * 1_000, 1)
        log.info(f"Sessions ready: {model_load_ms:.0f} ms")

        # ── Read WAV header ───────────────────────────────────────────────────
        t_hdr = time.monotonic()
        info = get_wav_info(audio_bucket, audio_key)
        hdr_ms = round((time.monotonic() - t_hdr) * 1_000, 1)
        log.info(
            f"WAV header: {hdr_ms:.0f} ms  "
            f"duration={info.duration_sec:.1f}s  "
            f"sr={info.sample_rate}  ch={info.num_channels}"
        )

        # ── Fetch audio window from S3 ────────────────────────────────────────
        t_read = time.monotonic()
        audio = read_wav_window(
            audio_bucket, audio_key, info,
            start_sec    = fetch_start_sec,
            duration_sec = fetch_duration,
            target_sr    = _BANDIT_SR,
        )
        s3_read_ms = round((time.monotonic() - t_read) * 1_000, 1)

        n_samples = len(audio)
        log.info(
            f"S3 read: {s3_read_ms:.0f} ms  "
            f"{n_samples:,} samples @ {_BANDIT_SR} Hz  "
            f"({n_samples / _BANDIT_SR:.2f}s)  "
            f"rms={rms_db(audio):.1f} dBFS"
        )

        if n_samples == 0:
            raise RuntimeError(
                f"Empty audio for chunk {chunk_idx} "
                f"[{fetch_start_sec:.1f}s → {fetch_end_sec:.1f}s]"
            )

        # ── Run Bandit v2 OLA ─────────────────────────────────────────────────
        t_bandit = time.monotonic()
        log.info(
            f"Running Bandit v2: {n_samples:,} samples  "
            f"chunk={_BANDIT_CHUNK} (non-overlapping)  "
            f"stems={list(decoder_stems)}"
        )
        stem_audio, bandit_timing = _run_bandit(audio, sessions, decoder_stems)
        bandit_total_ms = round((time.monotonic() - t_bandit) * 1_000, 1)
        log.info(f"Bandit OLA total: {bandit_total_ms:.0f} ms")
        for stage, ms in bandit_timing.items():
            log.info(f"  {stage:<20s}: {ms:.0f} ms")

        # ── speech_safe: gated mix substitution (Design record §5) ────────────
        # Built HERE because the mix slice and the raw stem coexist
        # sample-aligned pre-resample — zero extra inference or IO.
        patch_spans_ms: list = []
        patched_sec = 0.0
        if speech_safe:
            safe, patch_spans_ms = build_speech_safe(
                audio, stem_audio["speech"], _BANDIT_SR)
            stem_audio["speech_safe"] = safe
            if "speech" not in requested_stems:
                del stem_audio["speech"]
            patched_sec = round(
                sum(e - s for s, e in patch_spans_ms) / 1000.0, 3)
            log.info(
                f"  speech_safe: {len(patch_spans_ms)} patch span(s)  "
                f"patched={patched_sec:.1f}s of {fetch_duration:.1f}s"
            )

        # ── Resample + upload stems ───────────────────────────────────────────
        t_write = time.monotonic()
        stem_results: dict[str, dict] = {}

        for stem, raw in stem_audio.items():
            tgt_sr    = resampl_freq[stem]
            out_audio = resample(raw, _BANDIT_SR, tgt_sr)
            wav_bytes = encode_wav_mono(out_audio, tgt_sr)
            s3_key    = f"{output_prefix}/segment_chunks/{stem}/{chunk_name}.wav"

            _s3.put_object(
                Bucket      = output_bucket,
                Key         = s3_key,
                Body        = wav_bytes,
                ContentType = "audio/wav",
            )
            rms = rms_db(out_audio)
            dur = round(len(out_audio) / tgt_sr, 3)
            log.info(
                f"  uploaded {stem}: s3://{output_bucket}/{s3_key}  "
                f"sr={tgt_sr}  dur={dur:.2f}s  rms={rms:.1f} dBFS  "
                f"{len(wav_bytes):,} bytes"
            )
            stem_results[stem] = {
                "s3_key":       s3_key,
                "resampl_freq": tgt_sr,
                "rms_db":       rms,
                "duration_sec": dur,
            }

        write_ms = round((time.monotonic() - t_write) * 1_000, 1)

        # ── Build timing summary ──────────────────────────────────────────────
        elapsed_ms = round((time.monotonic() - t0) * 1_000, 1)
        timing_ms = {
            "model_load":      model_load_ms,
            "s3_read":         s3_read_ms,
            "encoder":         bandit_timing.get("encoder", 0.0),
            **{f"{s}_decoder": bandit_timing.get(s, 0.0) for s in decoder_stems},
            "resample_write":  write_ms,
            "total":           elapsed_ms,
        }

        # ── Metrics JSON ──────────────────────────────────────────────────────
        metrics_doc = {
            "chunk_idx":       chunk_idx,
            "chunk_name":      chunk_name,
            "fetch_start_sec": fetch_start_sec,
            "fetch_end_sec":   fetch_end_sec,
            "duration_sec":    round(fetch_duration, 3),
            "stems":           stem_results,
            "timing_ms":       timing_ms,
            "status":          "COMPLETED",
            "error":           None,
        }
        if speech_safe:
            # Chunk-relative patch record + the knobs that shaped it (a
            # params block that omits a knob that shaped the output lies).
            metrics_doc["speech_safe"] = {
                "patched_spans_ms": patch_spans_ms,
                "patched_sec": patched_sec,
                "params": speech_safe_params(),
            }
        metrics_key = f"{output_prefix}/segment_chunks/{chunk_name}.json"
        put_json(output_bucket, metrics_key, metrics_doc)
        log.info(f"  uploaded metrics: s3://{output_bucket}/{metrics_key}")

        # ── Return enriched iterator item ─────────────────────────────────────
        elapsed_sec = round((time.monotonic() - t0), 2)
        output = {
            **item,
            "stems":       stem_results,
            "timing_ms":   timing_ms,
            "metrics_key": metrics_key,
            "status":      "COMPLETED",
            "elapsed_sec": elapsed_sec,
        }
        log.info(
            f"DONE chunk={chunk_idx}  elapsed={elapsed_sec}s  "
            f"stems={list(stem_results)}"
        )
        return output

    except Exception as exc:
        log.error(
            "[ERR] chunk %s — %s: %s",
            event.get("item_data", {}).get("chunk_idx", "?"),
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise
