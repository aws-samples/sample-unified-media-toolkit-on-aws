# states/run_deepfilternet_model/model_inference.py
"""
DeepFilterNet3 inference — pure numpy + ORT.

All signal processing is a direct numpy port of libDF Rust code
(lib.rs, transforms.rs). No Rust, no torch, no maturin required.

Public API
──────────
  DFN3Enhancer(model_dir, num_threads=None)
    .enhance(audio, atten_lim_db=None) → np.ndarray float32

Config (matched to custom DeepFilterNet3 checkpoint)
──────────────────────────────────────────────────────
  sr=48000, fft_size=960, hop_size=480   (resampling is the caller's
  job via lib.wav_utils.resample — the legacy linear-interp helper was
  dead code here and is the aliasing bug class fixed repo-wide 2026-09-28)
  nb_erb=32, nb_df=96, df_order=5, df_lookahead=2, norm_tau=1,
  conv_lookahead=2 (features shifted ahead of the encoder — upstream
  DfNet.pad_feat; the exported graphs are causal, so this is the
  caller's job)

Upstream fidelity (2026-10-10 audit)
────────────────────────────────────
  Four behaviors restored to match rikorose/DeepFilterNet's own
  enhance() batch path (the source of the exported graphs):
    1. STFT/ISTFT algorithmic delay (fft−hop = 480 samples) is
       compensated — output is sample-aligned with the input.
    2. Features are shifted by conv_lookahead frames before the
       encoder (the context the weights were trained with).
    3. The deep filter runs on the RAW spectrum; only bins ≥ nb_df
       come from the ERB-masked spectrum (DfNet.forward order).
    4. atten_lim_db post-mixes lim·noisy + (1−lim)·enhanced AFTER
       both stages (full-band guarantee; <0.01 dB = bypass,
       ≥100 dB = unlimited) instead of flooring the ERB mask.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_SR = 48_000
_FFT_SIZE = 960
_HOP_SIZE = 480
_NB_ERB = 32
_NB_DF = 96
_DF_ORDER = 5
_DF_LOOKAHEAD = 2
_CONV_LOOKAHEAD = 2
_NORM_TAU = 1
_N_FREQ = _FFT_SIZE // 2 + 1    # 481

# ERB band widths — exact values from libDF df_state.erb_widths()
_ERB_WIDTHS = np.array([
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    5,  5,  7,  7,  8, 10, 12, 13, 15, 18, 20,
    24, 28, 31, 37, 42, 50, 56, 67,
], dtype=np.int32)

assert _ERB_WIDTHS.sum() == _N_FREQ
assert len(_ERB_WIDTHS) == _NB_ERB

# Vorbis window — sin(pi/2 * sin^2(pi*(i+0.5) / window_size_h))
_WINDOW_SIZE_H = _FFT_SIZE // 2
_WINDOW = np.array([
    math.sin(0.5 * math.pi *
             math.sin(0.5 * math.pi * (i + 0.5) / _WINDOW_SIZE_H) ** 2)
    for i in range(_FFT_SIZE)
], dtype=np.float32)

# Analysis normalisation: wnorm = 1 / (fft_size^2 / (2*hop_size))
_WNORM = float(1.0 / (_FFT_SIZE ** 2 / (2.0 * _HOP_SIZE)))

# Norm initial states — from lib.rs UNIT_NORM_INIT / MEAN_NORM_INIT constants
_UNIT_NORM_INIT = np.linspace(0.001, 0.0001, _NB_DF,  dtype=np.float32)
_ERB_NORM_INIT = np.linspace(-60.0, -90.0,  _NB_ERB, dtype=np.float32)

# norm_alpha = exp(-hop/sr / tau), ROUNDED like upstream get_norm_alpha
# (shortest decimal < 1.0) so feature EMAs match upstream exactly: 0.990
def _calc_norm_alpha(sr: int, hop: int, tau: float) -> float:
    a_exact = math.exp(-(hop / sr) / tau)
    precision = 3
    a = 1.0
    while a >= 1.0:
        a = round(a_exact, precision)
        precision += 1
    return a


_NORM_ALPHA = _calc_norm_alpha(_SR, _HOP_SIZE, _NORM_TAU)


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def _build_bin_to_band(widths: np.ndarray) -> np.ndarray:
    """[n_freq] lookup: each FFT bin → its ERB band index."""
    n_freq = int(widths.sum())
    b2b = np.empty(n_freq, dtype=np.int32)
    pos = 0
    for band_idx, w in enumerate(widths):
        b2b[pos: pos + w] = band_idx
        pos += w
    return b2b


def _stft_full(audio: np.ndarray) -> np.ndarray:
    """
    Full-file STFT using stateful frame_analysis from lib.rs.
    Vorbis window applied to [mem | new_frame], scaled by wnorm.
    Returns complex64 [n_frames, n_freq].
    """
    hop = _HOP_SIZE
    overlap = _FFT_SIZE - hop
    wf = _WINDOW[:overlap]
    ws = _WINDOW[overlap:]
    n = len(audio)
    n_frames = max(1, math.ceil(n / hop))
    mem = np.zeros(overlap, dtype=np.float32)
    frames = []

    for i in range(n_frames):
        chunk = audio[i * hop: i * hop + hop]
        if len(chunk) < hop:
            chunk = np.pad(chunk, (0, hop - len(chunk)))
        buf = np.empty(_FFT_SIZE, dtype=np.float32)
        buf[:overlap] = mem * wf
        buf[overlap:] = chunk * ws
        mem[:overlap - hop] = mem[hop:]
        mem[overlap - hop:] = chunk
        spec = np.fft.rfft(buf.astype(np.float64)).astype(np.complex64)
        spec *= _WNORM
        frames.append(spec)

    return np.stack(frames, axis=0)   # [T, F]


def _istft_full(spec: np.ndarray, original_length: int) -> np.ndarray:
    """
    Full-file iSTFT using stateful frame_synthesis from lib.rs.
    np.fft.irfft normalises by 1/N; Rust's realfft does NOT — multiply by N.
    """
    T = spec.shape[0]
    hop = _HOP_SIZE
    overlap = _FFT_SIZE - hop
    mem = np.zeros(overlap, dtype=np.float32)
    out = np.empty(T * hop, dtype=np.float32)

    for i in range(T):
        x = np.fft.irfft(spec[i], n=_FFT_SIZE).astype(np.float32) * _FFT_SIZE
        x *= _WINDOW
        out[i * hop: (i + 1) * hop] = x[:hop] + mem[:hop]
        split = overlap - hop
        new_mem = np.empty(overlap, dtype=np.float32)
        new_mem[:split] = mem[hop: hop + split]
        new_mem[:split] += x[hop: hop + split]
        new_mem[split:] = x[hop + split:]
        mem = new_mem

    return out[:original_length]


def _feat_erb(
    spec:       np.ndarray,   # complex [T, F]
    widths:     np.ndarray,   # int [nb_erb]
    norm_state: np.ndarray,   # float32 [nb_erb], modified in place
    alpha:      float,
) -> np.ndarray:
    """
    ERB features — port of DFState::feat_erb() in lib.rs:
      1. mean |X|^2 per band
      2. log10 * 10  (dB)
      3. EMA subtract, divide by 40
    Returns float32 [T, nb_erb].
    """
    T = spec.shape[0]
    nb_erb = len(widths)
    erb = np.zeros((T, nb_erb), dtype=np.float32)
    pos = 0
    for b, w in enumerate(widths):
        s = spec[:, pos: pos + w]
        erb[:, b] = (s.real ** 2 + s.imag ** 2).mean(axis=1)
        pos += w
    erb = np.log10(erb + 1e-10).astype(np.float32) * 10.0
    for t in range(T):
        norm_state[:] = erb[t] * (1.0 - alpha) + norm_state * alpha
        erb[t] = (erb[t] - norm_state) / 40.0
    return erb


def _feat_spec(
    spec:       np.ndarray,   # complex [T, F]
    unit_state: np.ndarray,   # float32 [nb_df], modified in place
    alpha:      float,
) -> np.ndarray:
    """
    Complex spectrogram unit norm — port of band_unit_norm() in lib.rs.
    Returns float32 [1, 2, T, nb_df].
    """
    T = spec.shape[0]
    s = spec[:, :_NB_DF].copy().astype(np.complex64)
    for t in range(T):
        mag = np.abs(s[t])
        unit_state[:] = mag * (1.0 - alpha) + unit_state * alpha
        s[t] /= np.sqrt(unit_state).astype(np.float32)
    ri = np.stack([s.real, s.imag], axis=0)    # [2, T, nb_df]
    return ri[np.newaxis].astype(np.float32)   # [1, 2, T, nb_df]


def _apply_erb_mask(
    spec:        np.ndarray,   # complex [T, F]
    mask:        np.ndarray,   # float32 [T, nb_erb]
    bin_to_band: np.ndarray,   # int [F]
) -> np.ndarray:
    """
    Apply ERB gain mask — port of apply_interp_band_gain() in lib.rs.
    Piecewise constant: each band gain broadcast to all bins in that band.
    (atten_lim_db is NOT a mask floor — it post-mixes the noisy spectrum
    after both stages, see _mix_atten_lim.)
    """
    gain_full = mask[:, bin_to_band]   # [T, F]
    return spec * gain_full


def _shift_features(feat: np.ndarray, n: int, time_axis: int) -> np.ndarray:
    """
    conv_lookahead shift — port of DfNet.pad_feat ConstantPad(−n, +n):
    drop the first n frames, append n zero frames, so the encoder sees
    features n frames ahead of the spectrum it will be applied to.
    """
    if n <= 0:
        return feat
    feat = np.roll(feat, -n, axis=time_axis)
    idx = [slice(None)] * feat.ndim
    idx[time_axis] = slice(-n, None)
    feat[tuple(idx)] = 0.0
    return feat


def _mix_atten_lim(
    spec_e:       np.ndarray,         # complex [T, F] enhanced
    spec_raw:     np.ndarray,         # complex [T, F] noisy
    atten_lim_db: Optional[float],
) -> np.ndarray:
    """
    Attenuation limit — port of upstream enhance():
      lim = 10^(−|db|/20);  out = lim·noisy + (1−lim)·enhanced
    applied AFTER both stages, so the cap bounds the ERB mask and the
    deep filter together, full band. ≥100 dB = unlimited (upstream
    treats it as None); <0.01 dB = full bypass (upstream warns and
    returns the noisy signal).
    """
    if atten_lim_db is None:
        return spec_e
    lim_db = abs(float(atten_lim_db))
    if lim_db >= 100.0:
        return spec_e
    if lim_db < 0.01:
        return spec_raw.copy()
    lim = float(10.0 ** (-lim_db / 20.0))
    return spec_raw * lim + spec_e * (1.0 - lim)


def _apply_df(
    spec:  np.ndarray,           # complex [T, F]
    coefs: np.ndarray,           # float32 [T, nb_df, df_order, 2]
    alpha: Optional[np.ndarray],  # float32 [T, 1] or None
) -> np.ndarray:
    """
    Deep filter — port of DfOp.forward_real_loop() in modules.py.
    pad_before = df_order - lookahead - 1
    pad_after  = lookahead
    """
    T, F = spec.shape
    pad_before = _DF_ORDER - _DF_LOOKAHEAD - 1
    pad_after = _DF_LOOKAHEAD

    pad_spec = np.concatenate([
        np.zeros((pad_before, F), dtype=np.complex64),
        spec,
        np.zeros((pad_after,  F), dtype=np.complex64),
    ], axis=0)
    df_spec = pad_spec[:, :_NB_DF]

    c_re = coefs[:, :, :, 0]   # [T, nb_df, df_order]
    c_im = coefs[:, :, :, 1]

    spec_f_r = np.zeros((T, _NB_DF), dtype=np.float32)
    spec_f_i = np.zeros((T, _NB_DF), dtype=np.float32)

    for o in range(_DF_ORDER):
        sl = df_spec[o: o + T]
        s_re = sl.real
        s_im = sl.imag
        spec_f_r += s_re * c_re[:, :, o] - s_im * c_im[:, :, o]
        spec_f_i += s_im * c_re[:, :, o] + s_re * c_im[:, :, o]

    spec_f = (spec_f_r + 1j * spec_f_i).astype(np.complex64)
    spec_out = spec.copy()
    if alpha is not None:
        spec_out[:, :_NB_DF] = (
            spec_f * alpha + spec[:, :_NB_DF] * (1.0 - alpha)
        )
    else:
        spec_out[:, :_NB_DF] = spec_f
    return spec_out


# =============================================================================
# ENHANCER
# =============================================================================

class DFN3Enhancer:
    """
    DeepFilterNet3 enhancer — pure numpy + ORT, no Rust/torch required.

    Parameters
    ──────────
    model_dir   : directory containing enc.onnx, erb_dec.onnx, df_dec.onnx
    num_threads : ORT intra_op threads (None = auto)
    """

    def __init__(self, model_dir: str, num_threads: Optional[int] = None):
        self._bin_to_band = _build_bin_to_band(_ERB_WIDTHS)

        opts = ort.SessionOptions()
        if num_threads is not None:
            opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def _load(name: str) -> ort.InferenceSession:
            path = os.path.join(model_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Not found: {path}")
            s = ort.InferenceSession(path, opts,
                                     providers=["CPUExecutionProvider"])
            log.info(f"  loaded {name}")
            return s

        self._enc = _load("enc.onnx")
        self._erb_dec = _load("erb_dec.onnx")
        self._df_dec = _load("df_dec.onnx")
        log.info("DFN3Enhancer ready")

    def enhance(
        self,
        audio:        np.ndarray,
        atten_lim_db: Optional[float] = None,
    ) -> np.ndarray:
        """
        Enhance a 48 kHz mono float32 waveform.
        Returns enhanced float32 at 48 kHz (same length as input).

        atten_lim_db : positive dB attenuation limit.
            None = unlimited | 6-9 = studio | 12 = broadcast | 18-20 = field
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio, got shape {audio.shape}")
        if len(audio) == 0:
            return audio.copy()

        original_length = len(audio)
        # STFT/ISTFT algorithmic delay — upstream enhance() pads the input
        # by fft_size and slices [delay : delay+orig] so the output is
        # sample-aligned with the input.
        delay = _FFT_SIZE - _HOP_SIZE
        audio = np.pad(audio, (0, _FFT_SIZE))

        # 1. STFT
        spec = _stft_full(audio)    # [T, F] complex64
        T = spec.shape[0]

        # 2. Features — fresh state per call, then the conv_lookahead
        #    shift (the exported graphs are causal; the 2-frame feature
        #    advance the model was trained with is applied HERE)
        erb_state = _ERB_NORM_INIT.copy()
        unit_state = _UNIT_NORM_INIT.copy()

        feat_erb = _feat_erb(spec, _ERB_WIDTHS, erb_state,  _NORM_ALPHA)
        feat_spec = _feat_spec(spec, unit_state,  _NORM_ALPHA)
        feat_erb_in = feat_erb[np.newaxis, np.newaxis]   # [1,1,T,32]

        feat_erb_in = _shift_features(feat_erb_in, _CONV_LOOKAHEAD,
                                      time_axis=2)
        feat_spec = _shift_features(feat_spec, _CONV_LOOKAHEAD,
                                    time_axis=2)

        # 3. Encoder
        e0, e1, e2, e3, emb, c0, lsnr = self._enc.run(
            ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
            {"feat_erb": feat_erb_in, "feat_spec": feat_spec},
        )

        # 4. ERB decoder → gain mask [1,1,T,32]
        (m,) = self._erb_dec.run(
            ["m"],
            {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0},
        )
        mask = m[0, 0]   # [T, 32]

        # 5. DF decoder → coefs [1,T,96,10]  (no alpha in this model)
        df_dec_outputs = self._df_dec.run(None, {"emb": emb, "c0": c0})
        coefs_raw = df_dec_outputs[0]
        coefs = coefs_raw[0].reshape(T, _NB_DF, _DF_ORDER, 2)   # [T,96,5,2]
        # alpha=None → full DF replacement (no blending with original)
        alpha = None

        # 6/7. DfNet.forward order: ERB mask on the raw spec; the deep
        #      filter ALSO on the raw spec; low bins (< nb_df) from the
        #      DF result, everything above from the masked spec.
        spec_m = _apply_erb_mask(spec, mask, self._bin_to_band)
        spec_out = _apply_df(spec, coefs, alpha)
        spec_out[:, _NB_DF:] = spec_m[:, _NB_DF:]

        # 7b. Attenuation limit — full-band post-mix over both stages
        spec_out = _mix_atten_lim(spec_out, spec, atten_lim_db)

        # 8. iSTFT, delay-compensated
        out = _istft_full(spec_out, len(audio))
        return out[delay: delay + original_length]
