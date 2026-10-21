"""
f0.py — per-slice fundamental-frequency statistics. Pure DSP (batched
autocorrelation over 4 kHz-decimated frames), numpy + scipy only — no
model, no new dependency.

Purpose (user, 2026-09-23): a voice-pitch statistic per speaker cluster,
consumed downstream as a face↔voice COMPATIBILITY signal (cast matching
against face gender estimates). Evidence, never a verdict: this module
emits Hz numbers; male/female thresholds — with their abstention band —
belong to the consumer, because the distributions overlap and children
sit off both charts.

Deliberately OUTSIDE every clustering path: nothing here feeds
embeddings, distances, or K-selection. It is a new output column.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

# F0 search band: covers adult speech with margin. 4 kHz working rate
# is 10x the band ceiling — decimation cost is what makes the batched
# autocorrelation cheap enough to run on EVERY sentence.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
WORK_SR = 4000
FRAME_SEC = 0.040
HOP_SEC = 0.020
# a frame is VOICED when its normalized autocorrelation peak in the F0
# lag band clears this — the classic periodicity test, not a tuned scalar
VOICED_AC_PEAK = 0.45
# a sentence's median stabilizes long before this; caps worst-case cost
MAX_SLICE_SEC = 6.0


def slice_f0(audio: np.ndarray, sr: int):
    """-> (f0_median_hz | None, voiced_ratio) for one mono float slice.
    None = no reliably voiced frames (whisper, noise, music bed)."""
    if audio is None or sr <= 0:
        return None, 0.0
    x = np.asarray(audio, dtype=np.float32)
    if len(x) < int(FRAME_SEC * sr):
        return None, 0.0
    if len(x) > int(MAX_SLICE_SEC * sr):
        x = x[: int(MAX_SLICE_SEC * sr)]
    if sr != WORK_SR:
        # polyphase decimation (anti-aliased); sr is int, WORK_SR divides
        # common rates (16k/44.1k/48k all reduce fine via up/down)
        x = resample_poly(x, WORK_SR, sr).astype(np.float32)
        sr = WORK_SR
    frame = int(FRAME_SEC * sr)
    hop = int(HOP_SEC * sr)
    lo = max(2, int(sr / F0_MAX_HZ))
    hi = min(frame - 2, int(sr / F0_MIN_HZ))
    if hi <= lo or len(x) < frame:
        return None, 0.0

    frs = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    if not len(frs):
        return None, 0.0
    frs = frs - frs.mean(axis=1, keepdims=True)
    # energy gate: skip frames far below the slice's own level
    rms = np.sqrt((frs * frs).mean(axis=1) + 1e-12)
    keep = rms >= 0.5 * np.sqrt((x * x).mean() + 1e-12)
    n_frames = len(frs)
    frs = frs[keep]
    if not len(frs):
        return None, 0.0

    # batched autocorrelation via FFT (Wiener–Khinchin)
    nfft = 1 << int(np.ceil(np.log2(2 * frame)))
    spec = np.fft.rfft(frs, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :frame]
    zero = ac[:, :1].copy()
    zero[zero <= 0] = 1.0
    ac = ac / zero
    band = ac[:, lo:hi]
    kbest = band.argmax(axis=1) + lo
    peaks = band.max(axis=1)
    voiced = peaks >= VOICED_AC_PEAK
    if not voiced.any():
        return None, 0.0
    f0s = sr / kbest[voiced]
    return float(np.median(f0s)), float(voiced.sum() / n_frames)


def aggregate_f0(sentence_stats: list) -> dict | None:
    """Per-cluster rollup over [(f0_hz | None, voiced_ratio), ...] ->
    {f0_median_hz, f0_iqr_hz, f0_voiced_ratio, f0_n_sentences} or None
    when nothing voiced. Median-of-sentence-medians: robust to the odd
    music-bled or whispered sentence."""
    voiced = [f for f, _ in sentence_stats if f is not None]
    if not voiced:
        return None
    arr = np.asarray(voiced, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    ratios = [r for _, r in sentence_stats]
    return {"f0_median_hz": round(float(np.median(arr)), 1),
            "f0_iqr_hz": round(float(q3 - q1), 1),
            "f0_voiced_ratio": round(float(np.mean(ratios)), 3),
            "f0_n_sentences": len(voiced)}
