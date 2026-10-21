"""
Frame triage — decode-and-inspect (README.md §3).

A byte-size heuristic (drop frames whose JPEG compresses below a threshold,
scaled by pixel count) was tried and REMOVED: JPEG size tracks scene
complexity (texture, grain, motion blur), not "has a face". On a real test
asset it silently discarded a frame with a clear, in-focus face (22.7 KB,
under the scaled threshold) while correctly-black frames from the same dark
scene were as low as 5.6 KB — no reliable separating threshold exists
between those two cases from byte count alone. Every frame is now decoded
and classified from actual pixel content (tier below) — no zero-decode
shortcut.

Classification: pyvips thumbnail with shrink-on-load (JPEG DCT fast path,
releases the GIL so ThreadPoolExecutor scales across cores). Luma
statistics: black (mean/p99), mono-solid (stddev), extreme blur (Laplacian
variance).

The downsampled luma is reused for frame_hash.frame_hashes() — one decode
serves triage AND the skip predicate.

pyvips is an unconditional, pinned dependency (requirements.txt) — same as
lib/grid_image.py, which hard-imports it with no fallback. There is no
Pillow fallback here either: if pyvips fails to import, grid_image.py (used
by every state after triage) would already crash the container at the same
cold start, so a decode-only fallback in this module alone bought nothing
except a false sense of resilience. Removed.
"""

from __future__ import annotations

import logging

import numpy as np
import pyvips

from lib.input_schema import (
    TRIAGE_BLACK_LUMA_MEAN,
    TRIAGE_BLACK_LUMA_P99,
    TRIAGE_BLUR_LAPLACIAN_VAR,
    TRIAGE_MONO_STDDEV,
)

logger = logging.getLogger(__name__)

# Decode target: stats + dHash need ~1/8 scale; 240 px wide is ample.
_THUMB_WIDTH = 240


def decode_luma(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG to a downsampled 2-D uint8 luma array."""
    img = pyvips.Image.thumbnail_buffer(
        jpeg_bytes, _THUMB_WIDTH, size=pyvips.enums.Size.DOWN)
    if img.bands > 1:
        img = img.colourspace("b-w")
    return np.ndarray(
        buffer=img.write_to_memory(),
        dtype=np.uint8,
        shape=(img.height, img.width),
    ).copy()


def triage_luma(luma: np.ndarray) -> str | None:
    """
    Tier 1 classification. Returns a drop reason or None (keep).
    Reasons: "black", "mono", "blur".
    """
    f = luma.astype(np.float32)
    mean = float(f.mean())
    p99 = float(np.percentile(f, 99))
    if mean < TRIAGE_BLACK_LUMA_MEAN and p99 < TRIAGE_BLACK_LUMA_P99:
        return "black"
    if float(f.std()) < TRIAGE_MONO_STDDEV:
        return "mono"
    # Laplacian variance (4-neighbour) — extreme blur only
    lap = (-4 * f[1:-1, 1:-1] + f[:-2, 1:-1] + f[2:, 1:-1]
           + f[1:-1, :-2] + f[1:-1, 2:])
    if float(lap.var()) < TRIAGE_BLUR_LAPLACIAN_VAR:
        return "blur"
    return None
