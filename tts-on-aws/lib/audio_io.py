"""
Torch-free audio I/O: WAV load with mono downmix + polyphase resample.

Replaces upstream's load_wav (torchaudio.load + torchaudio.transforms.Resample)
with soundfile + scipy.signal.resample_poly. Verified end-to-end during local
ONNX verification: resampler choice diverges from torchaudio only in mel bins
above ~6.5 kHz on 16 kHz-source material (anti-aliasing rolloff territory,
no real signal content), and the fully-scipy path passes the ASR-roundtrip
gate with transcripts identical to the torch path.

Input contract (same as Phase 1's handler): WAV/PCM. Any sample rate
8000-48000 Hz is accepted and resampled to the model's required rate.
No ffmpeg, no format conversion — callers must send WAV.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load a WAV file, downmix to mono, resample to target_sr.
    Returns float32 [T] in [-1, 1]."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def validate_ref_wav(path: str, max_duration_s: float = 30.0) -> dict:
    """Validate a reference WAV against the input contract. Returns file
    info on success, raises ValueError with a caller-actionable message
    on violation. The 30 s cap comes from upstream frontend.py's own
    assert in _extract_speech_token."""
    try:
        info = sf.info(path)
    except Exception as exc:
        raise ValueError(f"reference audio is not a readable audio file: {exc}") from exc
    if info.format != "WAV":
        raise ValueError(f"reference audio must be WAV, got {info.format}")
    if not 8000 <= info.samplerate <= 48000:
        raise ValueError(f"reference sample rate {info.samplerate} outside 8000-48000 Hz")
    if info.duration > max_duration_s:
        raise ValueError(f"reference audio is {info.duration:.1f}s; max {max_duration_s:.0f}s")
    if info.duration < 1.0:
        raise ValueError(f"reference audio is {info.duration:.2f}s; too short to clone from")
    return {"samplerate": info.samplerate, "duration_s": round(info.duration, 3),
            "channels": info.channels}
