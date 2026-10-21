# lib/embedding.py
"""
Speaker embedding extraction using WeSpeakerResNet34 ONNX.

Called by: states/run_speech_embedding_model/lambda_function.py

Model contract (pyannote community-1 tarball, embedding.int8.onnx or embedding.onnx)
─────────────────────────────────────────────────────────────────────────────────────
  input  : "fbank"     float32[1, T, 80]
  outputs: varies by export — probe_embedding_model() identifies the real 2-D output
           by shape; "loss" (scalar training dummy) is filtered out automatically.

Feature extraction — pure numpy, Kaldi/WeSpeaker compatible
────────────────────────────────────────────────────────────
  25 ms window  (400 samples @ 16 kHz)
  10 ms hop     (160 samples)
  512-point FFT → 257 bins
  80 mel filters, f_min=20 Hz, f_max=7600 Hz
  Pre-emphasis 0.97
  Per-utterance CMVN (subtract time-axis mean)

Public API
──────────
  probe_embedding_model(session)
      → (input_name, output_name, embedding_dim)
      Identifies the 2-D embedding output from ONNX metadata; resolves via dummy
      inference if the shape is fully dynamic.

  compute_log_mel_fbank(wav, sr)
      → float32[T, 80]

  extract_embedding(wav, session, input_name, output_name, ...)
      → float32[D] | None
      Returns None if the segment is shorter than min_segment_sec or inference fails.

  extract_embedding_for_turn(wav, turn_start_abs, turn_end_abs, fetch_start_sec, ...)
      → float32[D] | None
      Slices wav to the turn window (all args in absolute file seconds) and delegates
      to extract_embedding.
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import onnxruntime as ort
from lib.input_schema import DEFAULT_MIN_EMBEDDING_DURATION_SEC

log = logging.getLogger(__name__)

# ── Fbank constants — must match WeSpeaker training defaults ──────────────────
_N_FFT = 512
_HOP = 160      # 10 ms @ 16 kHz
_WIN = 400      # 25 ms @ 16 kHz
_N_MELS = 80
_F_MIN = 20.0
_F_MAX = 7600.0
_PREEMPH = 0.97

_FB_CACHE: Optional[np.ndarray] = None   # built once per container


# =============================================================================
# MODEL PROBE
# =============================================================================

def probe_embedding_model(
    session: ort.InferenceSession,
) -> tuple[str, int]:
    """
    Return (input_name, embedding_dim) from the ONNX session metadata.
    Falls back to a dummy inference for fully dynamic output shapes.
    """
    input_name = session.get_inputs()[0].name

    # Find the real embedding output — NOT the scalar "embedding" output
    # which is a training loss dummy.  The real output has shape [batch, D].
    # We identify it as the first 2-D output with dim > 1 in both axes.
    outputs = session.get_outputs()
    emb_output_name = None
    embedding_dim = None

    for o in outputs:
        shape = o.shape
        if (len(shape) == 2
                and (shape[1] == 256
                     or (isinstance(shape[1], int) and shape[1] > 1))):
            emb_output_name = o.name
            embedding_dim = shape[1] if isinstance(shape[1], int) else None
            break

    if emb_output_name is None:
        raise RuntimeError(
            f"probe_embedding_model: no 2-D output found among "
            f"{[(o.name, o.shape) for o in outputs]}"
        )

    # If dim is dynamic (non-int shape element), resolve via dummy inference
    if embedding_dim is None:
        log.info(
            f"Dynamic embedding dim in output '{emb_output_name}' "
            f"— probing via dummy inference"
        )
        dummy = np.zeros((1, 298, _N_MELS), dtype=np.float32)
        out = session.run([emb_output_name], {input_name: dummy})
        arr = np.array(out[0])
        embedding_dim = int(arr.shape[1] if arr.ndim == 2 else arr.shape[0])

    log.info(
        f"Embedding model: input='{input_name}'  "
        f"output='{emb_output_name}'  dim={embedding_dim}"
    )
    return input_name, emb_output_name, embedding_dim


# =============================================================================
# LOG-MEL FILTERBANK
# =============================================================================

def _build_mel_fb() -> np.ndarray:
    """Build [n_mels, n_fft//2+1] triangular mel filterbank."""
    n_freq = _N_FFT // 2 + 1

    def hz2mel(h: float) -> float:
        return 2595.0 * np.log10(1.0 + h / 700.0)

    def mel2hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz2mel(_F_MIN), hz2mel(_F_MAX), _N_MELS + 2)
    bin_pts = np.floor(
        (_N_FFT + 1) * np.array([mel2hz(m) for m in mel_pts]) / 16_000
    ).astype(int)

    fb = np.zeros((_N_MELS, n_freq), dtype=np.float32)
    for m in range(1, _N_MELS + 1):
        fl, fc, fr = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(fl, fc):
            if fc > fl:
                fb[m - 1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr > fc:
                fb[m - 1, k] = (fr - k) / (fr - fc)
    return fb


def compute_log_mel_fbank(wav: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """
    Compute log-Mel filterbank features from a mono float32 waveform.
    Returns float32[T, 80].
    """
    global _FB_CACHE
    if _FB_CACHE is None:
        _FB_CACHE = _build_mel_fb()

    # Pre-emphasis
    emph = np.concatenate([[wav[0]], wav[1:] - _PREEMPH * wav[:-1]])

    # Framing with Hann window
    window = np.hanning(_WIN).astype(np.float32)
    n_frames = max(1, 1 + (len(emph) - _WIN) // _HOP)
    frames = np.lib.stride_tricks.as_strided(
        emph,
        shape=(n_frames, _WIN),
        strides=(emph.strides[0] * _HOP, emph.strides[0]),
    ).copy() * window

    # Power spectrum → mel filterbank → log
    power = np.abs(np.fft.rfft(frames, n=_N_FFT).astype(np.complex64)) ** 2
    mel = power @ _FB_CACHE.T
    logm = np.log(np.maximum(mel, 1e-10))

    # Per-utterance CMVN
    logm -= logm.mean(axis=0, keepdims=True)
    return logm.astype(np.float32)


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

def extract_embedding(
    wav:             np.ndarray,
    session:         ort.InferenceSession,
    input_name:      str,
    output_name:     str,
    sr:              int = 16_000,
    min_segment_sec: float = DEFAULT_MIN_EMBEDDING_DURATION_SEC,
) -> Optional[np.ndarray]:
    """
    Extract a single L2-normalised speaker embedding from a waveform segment.

    Returns float32[D], or None if the segment is too short or inference fails.
    min_segment_sec overrides the module default — pass from event config so
    the threshold is not hardcoded across all content types.
    """
    if len(wav) < int(min_segment_sec * sr):
        log.debug(
            f"Segment {len(wav)/sr:.2f}s < {min_segment_sec}s min — skipped"
        )
        return None

    try:
        fbank = compute_log_mel_fbank(wav, sr)              # [T, 80]
        inp = fbank[np.newaxis]                           # [1, T, 80]
        out = session.run([output_name], {input_name: inp})[0]  # [1, D] or [D]
        emb = (out[0] if out.ndim == 2 else out).astype(np.float32)  # [D]

        # Re-normalise defensively (model already does this)
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb

    except Exception as exc:
        log.warning(f"extract_embedding failed: {exc}")
        return None


def extract_embedding_for_turn(
    wav:             np.ndarray,
    turn_start_abs:  float,
    turn_end_abs:    float,
    fetch_start_sec: float,
    session:         ort.InferenceSession,
    input_name:      str,
    output_name:     str,
    sr:              int = 16_000,
    min_segment_sec: float = DEFAULT_MIN_EMBEDDING_DURATION_SEC,
) -> Optional[np.ndarray]:
    """
    Slice `wav` to the turn's time window and extract an embedding.

    All time arguments are in absolute file seconds.
    `fetch_start_sec` is the absolute time of wav[0].
    `min_segment_sec` is forwarded to extract_embedding — pass from event
    config so the threshold is consistent with the handler's vad_cfg.
    """
    s = max(0,        int((turn_start_abs - fetch_start_sec) * sr))
    e = min(len(wav), int((turn_end_abs - fetch_start_sec) * sr))
    if e <= s:
        return None
    return extract_embedding(
        wav[s:e], session, input_name, output_name, sr,
        min_segment_sec=min_segment_sec,
    )
