"""
Torch-free DSP feature extraction for CosyVoice3 — NumPy/SciPy only.

Three extractors, each the exact frontend transform one ONNX model expects:

  whisper_log_mel(audio16k)  → [128, frames]   speech_tokenizer_v3 input
  kaldi_fbank(audio16k)      → [frames, 80]    CAMPPlus input
  matcha_mel(audio24k)       → [80, frames]    Flow prompt-mel conditioning

All three were verified numerically against the original torch/torchaudio
implementations on real reference audio during local ONNX verification
(onnx_verify/test_dsp.py, 9/9 comparisons): whisper mel matches to ~1e-5,
kaldi fbank and matcha mel to ~1e-3 (residual divergence confined to
near-silence bins at the log floor — float32 summation-order rounding,
not an algorithmic difference).

Parameter provenance (verified, not assumed):
  whisper_log_mel — openai-whisper audio.py::log_mel_spectrogram
  kaldi_fbank     — torchaudio.compliance.kaldi.fbank with the exact args
                    upstream frontend.py uses (num_mel_bins=80, dither=0,
                    sample_frequency=16000, all else at kaldi defaults)
  matcha_mel      — matcha.utils.audio.mel_spectrogram with the params
                    from the official cosyvoice3.yaml (n_fft=1920,
                    hop=480, win=1920, 80 mels, fmin=0, fmax=None)
"""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import get_window

# Slaney-style mel filterbanks, precomputed offline with
# librosa.filters.mel (see assets/README) and shipped as .npy — keeps
# librosa (and its numba/llvmlite chain) out of the runtime image.
# Regenerate with:
#   librosa_mel_fn(sr=16000, n_fft=400, n_mels=128)          → whisper
#   librosa_mel_fn(sr=24000, n_fft=1920, n_mels=80, fmax=None) → matcha
_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "assets")
_filterbanks: dict[str, np.ndarray] = {}


def _mel_filterbank(name: str) -> np.ndarray:
    if name not in _filterbanks:
        _filterbanks[name] = np.load(os.path.join(_ASSETS, f"{name}.npy"))
    return _filterbanks[name]


def _stft_frames(x: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """Framewise rfft: [n_fft//2+1, n_frames] complex."""
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:n_frames]
    return np.fft.rfft(frames * window, axis=1).T


def whisper_log_mel(audio: np.ndarray) -> np.ndarray:
    """audio: float32 [T] at 16 kHz → [128, frames] float32.

    Whisper-style: hann window, center=True (reflect pad), |stft|^2 with
    the final frame dropped, slaney mel filterbank, log10, max-8 floor,
    (x+4)/4 scaling.
    """
    n_fft, hop, sr = 400, 160, 16000
    window = get_window("hann", n_fft, fftbins=True).astype(np.float64)
    x = np.pad(audio.astype(np.float64), (n_fft // 2, n_fft // 2), mode="reflect")

    stft = _stft_frames(x, n_fft, hop, window)
    magnitudes = np.abs(stft[:, :-1]) ** 2  # whisper drops the last frame

    filters = _mel_filterbank("mel_filters_whisper_16k_128")
    mel_spec = filters @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def _kaldi_mel_banks(num_bins: int, window_length_padded: int, sample_freq: float,
                     low_freq: float = 20.0, high_freq: float = 0.0) -> np.ndarray:
    """Port of torchaudio.compliance.kaldi.get_mel_banks (vtln disabled).
    Returns [num_bins, num_fft_bins]; num_fft_bins = padded//2 — kaldi
    drops the nyquist bin."""
    num_fft_bins = window_length_padded // 2
    if high_freq <= 0.0:
        high_freq += 0.5 * sample_freq
    fft_bin_width = sample_freq / window_length_padded

    def mel(freq):
        return 1127.0 * np.log(1.0 + freq / 700.0)

    mel_low, mel_high = mel(low_freq), mel(high_freq)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)

    bins = np.arange(num_bins)[:, None]
    left = mel_low + bins * mel_delta
    center = mel_low + (bins + 1.0) * mel_delta
    right = mel_low + (bins + 2.0) * mel_delta

    mel_freqs = mel(fft_bin_width * np.arange(num_fft_bins)[None, :])
    up = (mel_freqs - left) / (center - left)
    down = (right - mel_freqs) / (right - center)
    return np.maximum(0.0, np.minimum(up, down))


def kaldi_fbank(audio: np.ndarray, num_mel_bins: int = 80) -> np.ndarray:
    """audio: float32 [T] in [-1, 1] at 16 kHz → [frames, num_mel_bins] float32.

    Port of torchaudio.compliance.kaldi.fbank(dither=0) at kaldi defaults:
    25 ms/10 ms framing (snip_edges), remove_dc_offset, 0.97 preemphasis,
    povey window (hann^0.85), pad-to-512 power spectrum without the nyquist
    bin, mel filterbank, log floored at float32 eps.

    NOTE the input is used as-is — torchaudio does NOT rescale to int16
    range here (verified empirically; an assumed x32768 scale produces a
    constant 2·log(32768) ≈ 20.8 offset in every bin).

    NOTE the caller applies mean-normalization (feat -= feat.mean(axis=0))
    before CAMPPlus, matching upstream frontend.py — not done here.
    """
    sr = 16000
    win_len, win_shift = 400, 160  # 25 ms / 10 ms
    padded = 512                   # round-to-pow2 default

    x = audio.astype(np.float64)
    n_frames = 1 + (len(x) - win_len) // win_shift
    frames = np.lib.stride_tricks.sliding_window_view(x, win_len)[::win_shift][:n_frames].copy()

    frames -= frames.mean(axis=1, keepdims=True)          # remove_dc_offset

    first = frames[:, :1].copy()                          # preemphasis 0.97
    frames[:, 1:] -= 0.97 * frames[:, :-1]
    frames[:, :1] -= 0.97 * first

    n = np.arange(win_len)
    povey = (0.5 - 0.5 * np.cos(2 * np.pi * n / (win_len - 1))) ** 0.85
    frames *= povey

    spec = np.fft.rfft(frames, n=padded)
    power = (spec.real ** 2 + spec.imag ** 2)[:, :padded // 2]

    banks = _kaldi_mel_banks(num_mel_bins, padded, sr)
    mel_energies = power @ banks.T
    # torchaudio floors at FLOAT32 eps (kaldi.py::_get_epsilon), not float64
    mel_energies = np.log(np.clip(mel_energies, np.finfo(np.float32).eps, None))
    return mel_energies.astype(np.float32)


def matcha_mel(audio: np.ndarray) -> np.ndarray:
    """audio: float32 [T] at 24 kHz → [80, frames] float32.

    Port of matcha.utils.audio.mel_spectrogram with cosyvoice3.yaml params:
    reflect-pad (n_fft-hop)/2 each side, hann window, center=False stft,
    sqrt(power + 1e-9), slaney mel filterbank, log(clamp 1e-5).

    float32 throughout, matching the torch reference — float64 here
    produces ~5e-4 extra divergence in the log domain.
    """
    n_fft, hop, win, sr = 1920, 480, 1920, 24000
    window = get_window("hann", win, fftbins=True).astype(np.float32)
    pad = (n_fft - hop) // 2
    x = np.pad(audio.astype(np.float32), (pad, pad), mode="reflect")

    stft = _stft_frames(x, n_fft, hop, window)
    spec = np.sqrt(stft.real.astype(np.float32) ** 2
                   + stft.imag.astype(np.float32) ** 2 + np.float32(1e-9))

    filters = _mel_filterbank("mel_filters_matcha_24k_80")
    mel = filters @ spec
    return np.log(np.clip(mel, 1e-5, None)).astype(np.float32)
