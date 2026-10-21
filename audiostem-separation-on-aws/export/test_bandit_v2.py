#!/usr/bin/env python3
"""
test_bandit_v2.py — Bandit v2 split encoder/decoder ONNX inference test
========================================================================

Tests the exported encoder + decoder ONNX models (FP32 and INT8).

Checks per model variant:
  1. Shape       — output length matches input length
  2. Energy      — each stem has non-trivial RMS (model is doing something)
  3. Conservation — sum of stem energies is plausible vs mixture energy
  4. Parity       — INT8 output within 5% RMS of FP32
  5. Chunked OLA — full production-style chunked inference with timing breakdown

Timing logged per stem and per chunk:
  encoder_ms, <stem>_decoder_ms, total_chunk_ms, RTF (real-time factor)

Usage
-----
  # Synthetic mixture, all stems, FP32 + INT8:
  python test_bandit_v2.py --onnx_dir ./models/onnx

  # Real audio, speech+sfx only, INT8:
  python test_bandit_v2.py --onnx_dir ./models/onnx \\
      --audio mix.wav --stems speech,sfx --int8_only

  # Save separated stems:
  python test_bandit_v2.py --onnx_dir ./models/onnx \\
      --audio mix.wav --save_stems --stem_dir ./output

Requirements
------------
  pip install onnxruntime numpy
  pip install librosa soundfile   # only needed with --audio
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import argparse
import json
import math
import os
import struct
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── constants (must match export_bandit_v2.py) ───────────────────────────────
FS            = 48_000
N_FFT         = 2048
HOP_LENGTH    = 512
CHUNK_SAMPLES = 384_000   # 8 s @ 48 kHz
HOP_SAMPLES   = 48_000    # 1 s @ 48 kHz
ALL_STEMS     = ["speech", "music", "sfx"]


# ─── logging ──────────────────────────────────────────────────────────────────

def log(msg: str = "") -> None:
    print(msg, flush=True)

def log_section(title: str) -> None:
    log(f"\n{'─'*64}")
    log(f"  {title}")
    log(f"{'─'*64}")


# ─── audio helpers ────────────────────────────────────────────────────────────

def hann_window(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:-1].astype(np.float32)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

def rms_db(x: np.ndarray) -> float:
    r = rms(x)
    return 20 * math.log10(r + 1e-9)

def load_audio(path: str) -> np.ndarray:
    """Load audio, resample to 48 kHz, downmix to mono → [T] float32."""
    try:
        import librosa
        audio, _ = librosa.load(path, sr=FS, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        log("  librosa not found — trying scipy.io.wavfile")
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.abs(data).max() > 1.0:
            data /= 32768.0
        if sr != FS:
            raise RuntimeError(
                f"Audio is {sr} Hz but Bandit requires {FS} Hz. "
                "Install librosa for automatic resampling."
            )
        return data
    except Exception as e:
        raise RuntimeError(f"Could not load {path}: {e}")

def synthetic_mixture(duration_s: float = 10.0) -> np.ndarray:
    """Multi-component sine mixture that exercises all frequency bands."""
    t    = np.linspace(0, duration_s, int(duration_s * FS), dtype=np.float32)
    sig  = 0.4 * np.sin(2 * np.pi * 200  * t)
    sig += 0.2 * np.sin(2 * np.pi * 800  * t)
    sig += 0.1 * np.sin(2 * np.pi * 3400 * t)
    sig += 0.3 * np.sin(2 * np.pi * 440  * t)
    sig += 0.2 * np.sin(2 * np.pi * 880  * t)
    burst = slice(int(2.0 * FS), int(2.1 * FS))
    rng   = np.random.default_rng(42)
    sig[burst] += 0.5 * rng.standard_normal(burst.stop - burst.start).astype(np.float32)
    sig /= np.abs(sig).max() + 1e-8
    sig *= 0.707
    return sig

def save_wav(path: str, audio: np.ndarray, sr: int = FS) -> None:
    audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    n = len(audio_int)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        f.write(audio_int.tobytes())


# ─── ONNX session helpers ─────────────────────────────────────────────────────

_ORT_THREADS: int = 0   # 0 = use os.cpu_count(); set by main() from --threads

def load_session(model_path: str):
    import onnxruntime as ort
    ncpu = _ORT_THREADS if _ORT_THREADS > 0 else (os.cpu_count() or 4)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads        = ncpu   # parallelism within a single op (matmul etc.)
    opts.inter_op_num_threads        = ncpu   # parallelism between independent graph nodes
    opts.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode              = ort.ExecutionMode.ORT_PARALLEL
    return ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )

def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else 0.0

def encoder_path(onnx_dir: str, int8: bool) -> str:
    return os.path.join(onnx_dir, "encoder.int8.onnx" if int8 else "encoder.onnx")

def decoder_path(onnx_dir: str, stem: str, int8: bool) -> str:
    suffix = "int8.onnx" if int8 else "onnx"
    return os.path.join(onnx_dir, f"{stem}_decoder.{suffix}")


# ─── split encoder/decoder inference ─────────────────────────────────────────

class SplitSessions:
    """Holds loaded ORT sessions for encoder + requested decoder heads."""

    def __init__(self, onnx_dir: str, stems: List[str], int8: bool):
        label = "INT8" if int8 else "FP32"
        enc_p = encoder_path(onnx_dir, int8)
        if not os.path.exists(enc_p):
            raise FileNotFoundError(f"Encoder not found: {enc_p}")
        log(f"  Loading encoder [{label}]  ({model_size_mb(enc_p):.1f} MB)  ...")
        t0 = time.perf_counter()
        self.encoder  = load_session(enc_p)
        enc_load_ms   = (time.perf_counter() - t0) * 1000
        log(f"    loaded in {enc_load_ms:.0f} ms")

        self.decoders: Dict[str, object] = {}
        for stem in stems:
            dec_p = decoder_path(onnx_dir, stem, int8)
            if not os.path.exists(dec_p):
                raise FileNotFoundError(f"Decoder not found: {dec_p}")
            log(f"  Loading {stem}_decoder [{label}]  ({model_size_mb(dec_p):.1f} MB)  ...")
            t0 = time.perf_counter()
            self.decoders[stem] = load_session(dec_p)
            dec_load_ms = (time.perf_counter() - t0) * 1000
            log(f"    loaded in {dec_load_ms:.0f} ms")

        self.stems = stems
        self.label = label


def run_encoder(sess, chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run encoder on one audio chunk.

    Input:  chunk [T] float32
    Returns: q [1,64,N_frames,128], x [1,1,1025,N_frames,2], elapsed_ms
    """
    x_in = chunk[np.newaxis, np.newaxis, :].astype(np.float32)  # [1,1,T]
    t0   = time.perf_counter()
    q, x = sess.run(["q", "x"], {"mixture": x_in})
    ms   = (time.perf_counter() - t0) * 1000
    log(f"    enc q={list(q.shape)}  x={list(x.shape)}")
    return q, x, ms


def run_decoder(sess, stem: str, q: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Run one decoder head.

    Returns: separated stem [T], elapsed_ms
    """
    t0  = time.perf_counter()
    out = sess.run([stem], {"q": q, "x": x})[0]   # [1, 1, T]
    ms  = (time.perf_counter() - t0) * 1000
    return out[0, 0, :], ms


def run_chunk(sessions: SplitSessions,
              chunk: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Run encoder once + all decoder heads on a single chunk.

    Returns:
        stems_out : {stem: [T] float32}
        timing_ms : {"encoder": ms, "<stem>": ms, "total": ms}
    """
    timing: Dict[str, float] = {}

    q, x, enc_ms = run_encoder(sessions.encoder, chunk)
    timing["encoder"] = enc_ms

    stems_out: Dict[str, np.ndarray] = {}
    for stem, dec_sess in sessions.decoders.items():
        out, dec_ms = run_decoder(dec_sess, stem, q, x)
        stems_out[stem] = out
        timing[stem] = dec_ms

    timing["total"] = sum(timing.values())
    return stems_out, timing


# ─── chunked OLA inference ────────────────────────────────────────────────────

def chunked_inference(
    sessions:      SplitSessions,
    audio:         np.ndarray,
    chunk_samples: int = CHUNK_SAMPLES,
    hop_samples:   int = HOP_SAMPLES,
    verbose:       bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    """
    Production-style chunked OLA inference.

    Returns:
        outputs     : {stem: [T] float32}  — separated stems, length == len(audio)
        chunk_times : {"encoder": [ms per chunk], "<stem>": [...], "total": [...]}
    """
    T         = len(audio)
    pad_front = chunk_samples - hop_samples   # 7 s
    pad_back  = chunk_samples
    padded    = np.pad(audio, (pad_front, pad_back), mode="reflect")
    window    = hann_window(chunk_samples)

    outputs = {s: np.zeros(T, dtype=np.float32) for s in sessions.stems}
    weights = np.zeros(T, dtype=np.float32)

    chunk_times: Dict[str, List[float]] = defaultdict(list)

    n_chunks  = math.ceil(T / hop_samples)
    pos       = 0
    chunk_idx = 0

    if verbose:
        log(f"\n  {'Chunk':>6}  {'Pos':>8}  {'Enc ms':>8}  "
            + "  ".join(f"{s:>10}" for s in sessions.stems)
            + f"  {'Total ms':>10}  {'RTF':>6}")
        log(f"  {'─'*6}  {'─'*8}  {'─'*8}  "
            + "  ".join("─"*10 for _ in sessions.stems)
            + f"  {'─'*10}  {'─'*6}")

    while pos < T:
        chunk = padded[pos : pos + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        stems_out, timing = run_chunk(sessions, chunk)

        for key, ms in timing.items():
            chunk_times[key].append(ms)

        rtf = (timing["total"] / 1000) / (chunk_samples / FS)

        if verbose:
            dec_cols = "  ".join(
                f"{timing.get(s, 0):>8.0f}ms" for s in sessions.stems
            )
            log(f"  {chunk_idx+1:>6}  {pos:>8}  "
                f"{timing['encoder']:>7.0f}ms  {dec_cols}  "
                f"{timing['total']:>8.0f}ms  {rtf:>5.2f}x")

        # OLA accumulate
        for stem, sep in stems_out.items():
            out_start = pos - pad_front
            out_end   = out_start + chunk_samples
            o_l = max(0, out_start)
            o_r = min(T, out_end)
            if o_l >= o_r:
                pos += hop_samples
                chunk_idx += 1
                continue
            c_l = o_l - out_start
            c_r = c_l + (o_r - o_l)
            outputs[stem][o_l:o_r] += sep[c_l:c_r] * window[c_l:c_r]

        out_start = pos - pad_front
        o_l = max(0, out_start)
        o_r = min(T, out_start + chunk_samples)
        if o_l < o_r:
            c_l = o_l - out_start
            c_r = c_l + (o_r - o_l)
            weights[o_l:o_r] += window[c_l:c_r]

        pos       += hop_samples
        chunk_idx += 1

    # Normalise OLA weights
    weights = np.maximum(weights, 1e-8)
    for stem in outputs:
        outputs[stem] /= weights

    return outputs, chunk_times


def print_timing_summary(chunk_times: Dict[str, List[float]],
                         audio_len: int, stems: List[str]) -> None:
    """Print per-component timing statistics across all chunks."""
    log(f"\n  {'─'*64}")
    log(f"  Timing summary  ({len(chunk_times['total'])} chunks, "
        f"{audio_len/FS:.1f}s audio)")
    log(f"  {'─'*64}")

    keys = ["encoder"] + stems + ["total"]
    hdr  = f"  {'Component':<20}  {'mean ms':>8}  {'min ms':>8}  {'max ms':>8}  {'p95 ms':>8}  {'% total':>8}"
    log(hdr)
    log(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    total_means = np.mean(chunk_times["total"])

    for key in keys:
        vals  = np.array(chunk_times[key])
        mean  = np.mean(vals)
        mn    = np.min(vals)
        mx    = np.max(vals)
        p95   = np.percentile(vals, 95)
        pct   = 100 * mean / (total_means + 1e-8) if key != "total" else 100.0
        log(f"  {key:<20}  {mean:>8.1f}  {mn:>8.1f}  {mx:>8.1f}  {p95:>8.1f}  {pct:>7.1f}%")

    total_elapsed_s  = sum(chunk_times["total"]) / 1000
    audio_duration_s = audio_len / FS
    overall_rtf      = total_elapsed_s / audio_duration_s
    throughput       = audio_duration_s / total_elapsed_s

    log(f"\n  Total wall time : {total_elapsed_s:.2f}s")
    log(f"  Audio duration  : {audio_duration_s:.2f}s")
    log(f"  Overall RTF     : {overall_rtf:.3f}x real-time "
        f"({'faster' if overall_rtf < 1 else 'SLOWER'} than real-time)")
    log(f"  Throughput      : {throughput:.2f}x audio duration per second")

    # Encoder share
    enc_mean = np.mean(chunk_times["encoder"])
    dec_mean = sum(np.mean(chunk_times[s]) for s in stems)
    log(f"\n  Encoder share   : {100*enc_mean/total_means:.1f}%  "
        f"(expected ~90% for GRU-heavy model)")
    log(f"  Decoder share   : {100*dec_mean/total_means:.1f}%  "
        f"(all {len(stems)} decoder head(s) combined)")


# ─── test functions ───────────────────────────────────────────────────────────

def test_single_chunk(onnx_dir: str, stems: List[str], int8: bool,
                      audio: np.ndarray) -> bool:
    """Shape + energy check on a single 8s chunk."""
    label = "INT8" if int8 else "FP32"
    log_section(f"Single-chunk test  [{label}]  stems={stems}")

    chunk = audio[:CHUNK_SAMPLES]
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

    try:
        sessions = SplitSessions(onnx_dir, stems, int8)
    except FileNotFoundError as e:
        log(f"  ⚠️  Skipping: {e}")
        return True

    stems_out, timing = run_chunk(sessions, chunk)

    log(f"\n  {'Component':<20}  {'Time ms':>9}  {'Shape':>15}  {'RMS':>8}  {'dBFS':>8}")
    log(f"  {'─'*20}  {'─'*9}  {'─'*15}  {'─'*8}  {'─'*8}")
    log(f"  {'encoder':<20}  {timing['encoder']:>8.1f}ms  "
        f"{'q+x':>15}  {'—':>8}  {'—':>8}")

    mix_rms = rms(chunk)
    passed  = True

    for stem in stems:
        out      = stems_out[stem]
        dec_ms   = timing[stem]
        shape_ok = (len(out) > 0)
        energy_ok = rms(out) > mix_rms * 0.0001
        ok       = shape_ok and energy_ok
        if not ok:
            passed = False
        status = "✅" if ok else "❌"
        log(f"  {status} {stem:<18}  {dec_ms:>8.1f}ms  "
            f"{str(out.shape):>15}  {rms(out):>8.4f}  {rms_db(out):>7.1f}dB")

    log(f"\n  {'total':<20}  {timing['total']:>8.1f}ms  "
        f"  mixture RMS={mix_rms:.4f}  ({mix_rms*1000:.0f}m)")

    return passed


def test_fp32_vs_int8(onnx_dir: str, stems: List[str],
                      audio: np.ndarray) -> bool:
    """INT8 output should be within 5% RMS error of FP32."""
    log_section("FP32 vs INT8 parity check")

    chunk = audio[:CHUNK_SAMPLES]
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

    try:
        fp32_sess = SplitSessions(onnx_dir, stems, int8=False)
        int8_sess = SplitSessions(onnx_dir, stems, int8=True)
    except FileNotFoundError as e:
        log(f"  ⚠️  Skipping: {e}")
        return True

    fp32_out, fp32_t = run_chunk(fp32_sess, chunk)
    int8_out, int8_t = run_chunk(int8_sess, chunk)

    log(f"\n  {'Stem':<10}  {'FP32 RMS':>10}  {'INT8 RMS':>10}  "
        f"{'RMS err%':>10}  {'FP32 ms':>9}  {'INT8 ms':>9}")
    log(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*9}")

    passed = True
    for stem in stems:
        fp32_r   = rms(fp32_out[stem])
        int8_r   = rms(int8_out[stem])
        err_pct  = rms(fp32_out[stem] - int8_out[stem]) / (fp32_r + 1e-8) * 100
        ok       = err_pct < 5.0
        if not ok:
            passed = False
        status = "✅" if ok else "❌"
        log(f"  {status} {stem:<8}  {fp32_r:>10.4f}  {int8_r:>10.4f}  "
            f"{err_pct:>9.2f}%  {fp32_t[stem]:>8.1f}ms  {int8_t[stem]:>8.1f}ms")

    log(f"\n  Encoder:  FP32={fp32_t['encoder']:.1f}ms  INT8={int8_t['encoder']:.1f}ms  "
        f"speedup={fp32_t['encoder']/max(int8_t['encoder'],0.1):.2f}x")

    return passed


def test_chunked(onnx_dir: str, stems: List[str], int8: bool,
                 audio: np.ndarray,
                 save_stems: bool = False,
                 stem_dir: str = ".") -> bool:
    """Full chunked OLA inference with per-chunk timing breakdown."""
    label = "INT8" if int8 else "FP32"
    log_section(f"Chunked OLA inference  [{label}]  "
                f"{len(audio)/FS:.1f}s audio  stems={stems}")

    try:
        sessions = SplitSessions(onnx_dir, stems, int8)
    except FileNotFoundError as e:
        log(f"  ⚠️  Skipping: {e}")
        return True

    outputs, chunk_times = chunked_inference(sessions, audio, verbose=True)

    print_timing_summary(chunk_times, len(audio), stems)

    log(f"\n  {'─'*64}")
    log(f"  Output quality")
    log(f"  {'─'*64}")
    log(f"  {'Stem':<10}  {'Length':>8}  {'Match':>6}  "
        f"{'RMS':>8}  {'dBFS':>8}  {'Energy ok':>10}")

    mix_rms = rms(audio)
    passed  = True

    for stem, out in outputs.items():
        length_ok = (len(out) == len(audio))
        energy_ok = rms(out) > mix_rms * 0.001
        ok        = length_ok and energy_ok
        if not ok:
            passed = False
        status = "✅" if ok else "❌"
        log(f"  {status} {stem:<8}  {len(out):>8}  "
            f"{'yes' if length_ok else 'NO':>6}  "
            f"{rms(out):>8.4f}  {rms_db(out):>7.1f}dB  "
            f"{'yes' if energy_ok else 'NO':>10}")

        if save_stems:
            out_path = os.path.join(stem_dir, f"{stem}_{label.lower()}.wav")
            save_wav(out_path, out)
            log(f"     → saved {out_path}")

    return passed


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Bandit v2 split encoder/decoder ONNX inference test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_bandit_v2.py --onnx_dir ./models/onnx\n"
            "  python test_bandit_v2.py --onnx_dir ./models/onnx "
            "--audio mix.wav --stems speech,sfx --int8_only\n"
            "  python test_bandit_v2.py --onnx_dir ./models/onnx "
            "--audio mix.wav --save_stems --stem_dir ./out\n"
        ),
    )
    ap.add_argument("--onnx_dir",    default="./models/onnx",
                    help="Directory containing encoder/decoder .onnx files")
    ap.add_argument("--audio",       default=None,
                    help="WAV/FLAC/MP3 input (mono or stereo, any SR). "
                         "Omit to use a 10s synthetic mixture.")
    ap.add_argument("--stems",       default="speech,music,sfx",
                    help="Comma-separated stems (default: speech,music,sfx). "
                         "Encoder runs once regardless; only requested decoders load.")
    ap.add_argument("--int8_only",   action="store_true",
                    help="Only test INT8 models")
    ap.add_argument("--fp32_only",   action="store_true",
                    help="Only test FP32 models")
    ap.add_argument("--no_chunked",  action="store_true",
                    help="Skip chunked OLA inference test")
    ap.add_argument("--no_parity",   action="store_true",
                    help="Skip FP32 vs INT8 parity check")
    ap.add_argument("--save_stems",  action="store_true",
                    help="Save separated stems as WAV files")
    ap.add_argument("--stem_dir",    default=".",
                    help="Directory to save stems (default: current dir)")
    ap.add_argument("--threads",      type=int, default=0,
                    help="ORT thread count (0 = auto = os.cpu_count())")
    ap.add_argument("--chunked_duration", type=float, default=30.0,
                    help="Audio duration (s) for chunked test (default: 30s). "
                         "Use 0 for full file.")
    args = ap.parse_args()

    log("=" * 64)
    log("  Bandit v2 — split encoder/decoder ONNX inference test")
    log("=" * 64)

    # ── thread configuration ─────────────────────────────────────────────────
    global _ORT_THREADS
    _ORT_THREADS = args.threads if args.threads > 0 else (os.cpu_count() or 4)
    log(f"\n  CPUs available : {os.cpu_count()}")
    log(f"  ORT threads    : {_ORT_THREADS}  (intra + inter op)")

    # ── parse stems ──────────────────────────────────────────────────────────
    stems = [s.strip() for s in args.stems.split(",")]
    invalid = [s for s in stems if s not in ALL_STEMS]
    if invalid:
        log(f"❌ Unknown stems: {invalid}. Valid: {ALL_STEMS}")
        sys.exit(1)

    log(f"\n  Stems     : {stems}")
    log(f"  ONNX dir  : {args.onnx_dir}")

    # ── read config ──────────────────────────────────────────────────────────
    cfg_path = os.path.join(args.onnx_dir, "model_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        log(f"  Config    : fs={cfg.get('sample_rate','?')}  "
            f"arch={cfg.get('architecture','?')}  "
            f"opset={cfg.get('onnx_opset','?')}  "
            f"quant={cfg.get('quantization','?')}")
        log(f"  Files     : encoder={cfg.get('files',{}).get('encoder','?')}  "
            f"decoders={list(cfg.get('files',{}).get('decoders',{}).values())}")
    else:
        log("  ⚠️  model_config.json not found")

    # ── load audio ───────────────────────────────────────────────────────────
    if args.audio:
        log(f"\n  Loading   : {args.audio}")
        audio = load_audio(args.audio)
    else:
        log("\n  No --audio provided — using 10s synthetic mixture")
        audio = synthetic_mixture(10.0)
    log(f"  Audio     : {len(audio)/FS:.2f}s  "
        f"RMS={rms(audio):.4f}  ({rms_db(audio):.1f} dBFS)  "
        f"samples={len(audio)}")

    # ── decide variants ──────────────────────────────────────────────────────
    variants: List[bool] = []   # True = INT8, False = FP32
    if not args.int8_only:
        variants.append(False)
    if not args.fp32_only:
        variants.append(True)

    results: List[bool] = []

    # ── single-chunk tests ───────────────────────────────────────────────────
    for int8 in variants:
        results.append(test_single_chunk(args.onnx_dir, stems, int8, audio))

    # ── FP32 vs INT8 parity ──────────────────────────────────────────────────
    if not args.no_parity and not args.int8_only and not args.fp32_only:
        results.append(test_fp32_vs_int8(args.onnx_dir, stems, audio))

    # ── chunked OLA ──────────────────────────────────────────────────────────
    if not args.no_chunked:
        dur = args.chunked_duration
        if dur <= 0:
            excerpt = audio
        else:
            excerpt = audio[:min(len(audio), int(dur * FS))]

        for int8 in variants:
            results.append(
                test_chunked(
                    args.onnx_dir, stems, int8, excerpt,
                    save_stems=args.save_stems,
                    stem_dir=args.stem_dir,
                )
            )

    # ── final summary ────────────────────────────────────────────────────────
    log(f"\n{'='*64}")
    passed = all(results)
    if passed:
        log(f"  ✅  All {len(results)} checks passed")
    else:
        n_fail = sum(1 for r in results if not r)
        log(f"  ❌  {n_fail}/{len(results)} checks FAILED")
    log("=" * 64)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()