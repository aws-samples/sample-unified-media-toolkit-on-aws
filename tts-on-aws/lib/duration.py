"""
Bounded duration control (README.md §7 Layer 2).

Pitch-preserving phase-vocoder time-stretch, pure NumPy/SciPy — applied
only when the synthesized duration misses a caller-requested target by
≤ MAX_STRETCH_RATIO (±10%). Beyond that: hard error, never a silent
larger stretch (both parties agreed audible degradation past ±10% makes
"success" worse than a clear failure).

Layer 1 (instruct-conditioned duration hint) is NOT implemented here —
it's a prompt-construction concern in the synthesis state, and remains
provisional (README.md §7).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

MAX_STRETCH_RATIO = 0.10  # ±10%, provisional per README.md §7


class DurationOutOfRangeError(ValueError):
    """Target duration misses actual by more than MAX_STRETCH_RATIO."""


def fit_to_duration(audio: np.ndarray, sr: int, target_s: float) -> np.ndarray:
    """Stretch/squeeze audio to target_s if within ±MAX_STRETCH_RATIO,
    else raise DurationOutOfRangeError with actionable numbers."""
    actual_s = len(audio) / sr
    ratio = target_s / actual_s
    if abs(ratio - 1.0) < 0.005:
        return audio  # already within half a percent — don't touch it
    if abs(ratio - 1.0) > MAX_STRETCH_RATIO:
        raise DurationOutOfRangeError(
            f"synthesized {actual_s:.2f}s vs target {target_s:.2f}s — "
            f"ratio {ratio:.3f} outside ±{MAX_STRETCH_RATIO:.0%}; refusing to "
            "stretch (quality degrades audibly past this bound)")
    stretched = _phase_vocoder_stretch(audio, rate=1.0 / ratio)
    log.info("time-stretch: %.2fs → %.2fs (ratio %.3f)",
             actual_s, len(stretched) / sr, ratio)
    return stretched


def _phase_vocoder_stretch(audio: np.ndarray, rate: float,
                           n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Classic phase-vocoder: STFT → time-axis resample with phase
    accumulation → iSTFT. rate > 1 shortens, rate < 1 lengthens."""
    from scipy.signal import stft, istft

    _, _, spec = stft(audio, nperseg=n_fft, noverlap=n_fft - hop, padded=True)
    n_bins, n_frames = spec.shape

    time_steps = np.arange(0, n_frames - 1, rate)
    phase_advance = np.linspace(0, np.pi * hop, n_bins)

    stretched = np.zeros((n_bins, len(time_steps)), dtype=np.complex64)
    phase_acc = np.angle(spec[:, 0])

    for i, step in enumerate(time_steps):
        left = int(np.floor(step))
        frac = step - left
        mag = (1 - frac) * np.abs(spec[:, left]) + frac * np.abs(spec[:, left + 1])
        stretched[:, i] = mag * np.exp(1j * phase_acc)

        dphase = np.angle(spec[:, left + 1]) - np.angle(spec[:, left]) - phase_advance
        dphase -= 2 * np.pi * np.round(dphase / (2 * np.pi))
        phase_acc += phase_advance + dphase

    _, out = istft(stretched, nperseg=n_fft, noverlap=n_fft - hop)
    return out.astype(np.float32)
