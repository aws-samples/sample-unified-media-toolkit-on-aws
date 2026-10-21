"""
CosyVoice3 HiFT vocoder stage — mel spectrogram → 24 kHz waveform.

Ports the verified onnx_verify/test_hift.py harness. Structure follows
cosyvoice/hifigan/generator.py::CausalHiFTGenerator's non-streaming
(finalize=True) path:

  f0_predictor  → per-frame F0
  source_gen    → NSF harmonic source signal (480x upsample = 120 mel
                  upsample x 4 iSTFT hop)
  numpy STFT    → source spectrum [18, T] (9 real + 9 imag, n_fft=16 hop=4)
  decoder       → magnitude/phase (exp/sin activations are INSIDE the
                  exported graph — verified empirically; do not re-apply)
  numpy iSTFT   → waveform, clipped to ±0.99 (upstream audio_limit)
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.signal import get_window

from lib.sessions import get_session

log = logging.getLogger(__name__)

N_FFT = 16
HOP_LEN = 4
SAMPLE_RATE = 24000
AUDIO_LIMIT = 0.99


def _stft(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """torch.stft(n_fft=16, hop=4, hann, center=True) equivalent."""
    window = get_window("hann", N_FFT, fftbins=True).astype(np.float32)
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP_LEN
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP_LEN][:n_frames]
    spec = np.fft.rfft(frames * window, axis=1).T
    return spec.real.astype(np.float32), spec.imag.astype(np.float32)


def _istft(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """torch.istft equivalent: overlap-add with window-sum normalization,
    center-padding removed. magnitude clipped at 1e2 (upstream _istft)."""
    window = get_window("hann", N_FFT, fftbins=True).astype(np.float32)
    magnitude = np.clip(magnitude, a_min=None, a_max=1e2)
    real = magnitude * np.cos(phase)
    imag = magnitude * np.sin(phase)

    n_frames = real.shape[1]
    out_len = N_FFT + (n_frames - 1) * HOP_LEN
    audio = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(real[:, i] + 1j * imag[:, i], n=N_FFT).astype(np.float32)
        start = i * HOP_LEN
        audio[start:start + N_FFT] += frame * window
        win_sum[start:start + N_FFT] += window ** 2
    audio /= np.maximum(win_sum, 1e-8)

    pad = N_FFT // 2
    return audio[pad:-pad]


def synthesize_waveform(model_dir: str, mel: np.ndarray) -> np.ndarray:
    """mel: [1, 80, T] → float32 waveform [T*480] at 24 kHz."""
    f0_predictor = get_session(f"{model_dir}/hift_f0_predictor_fp32.onnx")
    source_gen = get_session(f"{model_dir}/hift_source_generator_fp32.onnx")
    decoder = get_session(f"{model_dir}/hift_decoder_fp32.onnx")

    t0 = time.time()
    f0 = f0_predictor.run(None, {"mel": mel.astype(np.float32)})[0]      # [1, T]
    source = source_gen.run(None, {"f0": f0[:, np.newaxis, :].astype(np.float32)})[0]

    s_real, s_imag = _stft(source.squeeze())
    source_stft = np.concatenate([s_real, s_imag], axis=0)[np.newaxis, :, :]

    magnitude, phase = decoder.run(None, {
        "mel": mel.astype(np.float32),
        "source_stft": source_stft.astype(np.float32),
    })

    audio = _istft(magnitude.squeeze(0), phase.squeeze(0))
    audio = np.clip(audio, -AUDIO_LIMIT, AUDIO_LIMIT)
    log.info("HiFT: %d mel frames → %.2fs audio in %.2fs",
             mel.shape[2], len(audio) / SAMPLE_RATE, time.time() - t0)
    return audio
