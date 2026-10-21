# lib/assign.py
"""Locality-constrained speaker assignment with a rejection option.

One source of truth for every post-clustering mop-up (LOW sentences,
outlier turns): candidates are restricted to speakers with a clustered
HIGH anchor near the item in time, and the best candidate must beat the
runner-up by a margin — otherwise the item is UNKNOWN rather than
fattening the nearest big centroid.

Why (measured — README.md Design record P5): on a
labeled short-slice test set, GLOBAL nearest-centroid assignment scored
at or below chance (one asset: 13% against a 50% chance floor — short
embeddings are phonetics-dominated and map systematically wrong), while
restricting candidates to speakers anchored within +/-30 s scored
7x chance. The margin adds a modest honesty gate on top.

Called by: lib.cluster_pipeline._reassign_outliers,
           states/global_speakers_clustering (LOW assignment)
"""
from __future__ import annotations

import numpy as np


def assign_speaker_local(
    emb:           np.ndarray,
    t:             float,
    anchor_times:  np.ndarray,
    anchor_names:  list[str],
    centroids:     dict[str, np.ndarray],
    window_sec:    float,
    margin:        float,
) -> str | None:
    """
    Assign one L2-normalised embedding to a speaker, or reject.

    Parameters
    ──────────
    emb           [D] L2-normalised embedding of the item.
    t             item time (seconds) used for the locality window.
    anchor_times  [M] start times of clustered HIGH anchors.
    anchor_names  [M] speaker name per anchor (same order).
    centroids     speaker name → L2-normalised centroid.
    window_sec    locality window; candidates are speakers with an
                  anchor within ±window_sec of t. Falls back to ALL
                  speakers when the window is empty (the margin still
                  gates the answer).
    margin        required cosine-similarity lead over the runner-up.
                  A single candidate passes trivially.

    Returns the speaker name, or None to reject (caller maps to UNKNOWN).
    """
    if not centroids:
        return None

    near = np.abs(anchor_times - t) <= window_sec
    cand = {anchor_names[i] for i in np.where(near)[0]}
    names = [n for n in centroids if n in cand] or list(centroids)

    sims = sorted(((float(emb @ centroids[n]), n) for n in names),
                  reverse=True)
    if len(sims) == 1 or sims[0][0] - sims[1][0] >= margin:
        return sims[0][1]
    return None
