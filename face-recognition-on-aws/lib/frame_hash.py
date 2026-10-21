"""
Perceptual frame hashing for the probe-then-fill skip predicate.

Per frame: 3×3 patch dHashes + 1 global dHash (README.md §3, user design).
Each dHash: resize the (luma) region to 9×8, horizontal gradient sign →
64-bit int. Computed on the triage-pass downsampled luma — microseconds.

Patch noise classification (ticker discount): patches whose hash changes in
> PATCH_NOISE_CHANGE_FREQ of adjacent frame pairs across the whole asset are
chronically noisy (lower-third tickers, scoreboards, clocks) and are excluded
from the stability predicate — capped at PATCH_NOISE_MAX_DISCOUNT of 9.
"""

from __future__ import annotations

import numpy as np

from lib.input_schema import (
    GLOBAL_HASH_SANITY_FACTOR,
    PATCH_GRID,
    PATCH_NOISE_CHANGE_FREQ,
    PATCH_NOISE_MAX_DISCOUNT,
)

_HASH_W, _HASH_H = 9, 8  # 8×8 gradient bits


def _dhash(luma: np.ndarray) -> int:
    """64-bit difference hash of a 2-D luma array (any size)."""
    h, w = luma.shape
    # Nearest-neighbour resample to 9×8 via index maps (cheap, deterministic)
    ys = (np.arange(_HASH_H) * h // _HASH_H).clip(0, h - 1)
    xs = (np.arange(_HASH_W) * w // _HASH_W).clip(0, w - 1)
    small = luma[np.ix_(ys, xs)].astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def frame_hashes(luma: np.ndarray) -> dict:
    """
    Returns {"global": int, "patches": [int ×9]} for one frame's luma array.
    Patches are row-major 3×3 regions.
    """
    h, w = luma.shape
    patches = []
    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            y0, y1 = r * h // PATCH_GRID, (r + 1) * h // PATCH_GRID
            x0, x1 = c * w // PATCH_GRID, (c + 1) * w // PATCH_GRID
            patches.append(_dhash(luma[y0:y1, x0:x1]))
    return {"global": _dhash(luma), "patches": patches}


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def classify_noisy_patches(hash_list: list[dict], threshold: int) -> list[int]:
    """
    Per-asset noise classification. hash_list is time-ordered frame hashes.
    Returns indices (0-8) of chronically-noisy patches, at most
    PATCH_NOISE_MAX_DISCOUNT. If more patches qualify, the content is
    genuinely dynamic — return [] (no discount).
    """
    n_pairs = len(hash_list) - 1
    if n_pairs < 10:
        return []
    n_patches = PATCH_GRID * PATCH_GRID
    change_counts = [0] * n_patches
    for i in range(n_pairs):
        pa = hash_list[i]["patches"]
        pb = hash_list[i + 1]["patches"]
        for p in range(n_patches):
            if hamming(pa[p], pb[p]) > threshold:
                change_counts[p] += 1
    noisy = [p for p in range(n_patches)
             if change_counts[p] / n_pairs > PATCH_NOISE_CHANGE_FREQ]
    if len(noisy) > PATCH_NOISE_MAX_DISCOUNT:
        return []  # too much churn everywhere — genuinely dynamic content
    return noisy


def stable_distance(a: dict, b: dict, noisy: list[int]) -> int:
    """Max hamming distance across STABLE patches only."""
    noisy_set = set(noisy)
    dists = [hamming(pa, pb)
             for p, (pa, pb) in enumerate(zip(a["patches"], b["patches"]))
             if p not in noisy_set]
    return max(dists) if dists else 0


def gap_is_skippable(
    gap_hashes: list[dict],
    endpoint_a: dict,
    endpoint_b: dict,
    noisy: list[int],
    threshold: int,
) -> bool:
    """
    Skip predicate conditions 2-4 (condition 1 — both endpoint probes no-face —
    is checked by the caller). README.md §3:
      2. max adjacent-pair stable-patch distance inside the gap < threshold
      3. endpoint-to-endpoint stable-patch distance < threshold (slow drift)
      4. endpoint global distance < 2× threshold (backstop when patches discounted)
    """
    seq = [endpoint_a, *gap_hashes, endpoint_b]
    for i in range(len(seq) - 1):
        if stable_distance(seq[i], seq[i + 1], noisy) >= threshold:
            return False
    if stable_distance(endpoint_a, endpoint_b, noisy) >= threshold:
        return False
    if hamming(endpoint_a["global"], endpoint_b["global"]) >= (
            threshold * GLOBAL_HASH_SANITY_FACTOR):
        return False
    return True
