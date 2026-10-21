"""
Silhouette scoring, cluster confidence, and speaker-model computation.

Called by:
  lib.cluster_sweep             — _weighted_silhouette during K enumeration
  states/global_speakers_clustering/lambda_function.py
                                — _compute_cluster_confidence and _compute_speaker_models
                                  after clustering completes

Functions
─────────
  _weighted_silhouette          duration-weighted mean silhouette using a precomputed
                                distance matrix; vectorised O(N²·K), no per-sample loop
  _compute_cluster_confidence   per-cluster mean intra-cluster cosine similarity
  _compute_speaker_models       two-pass robust centroid per cluster (draft μ₀, then
                                trim periphery embeddings beyond centroid_trim_percentile)
"""
from __future__ import annotations

import logging

import numpy as np
from lib.input_schema import (
    DEFAULT_CENTROID_TRIM_PERCENTILE,
    DEFAULT_MIN_TURNS_FOR_TRIM,
)

log = logging.getLogger(__name__)


def _weighted_silhouette(
    square_dist: np.ndarray,
    labels:      np.ndarray,
    durations:   np.ndarray,
) -> float:
    """
    Duration-weighted mean silhouette score using a precomputed distance matrix.

    Vectorised O(N²·K) — no per-sample Python loop.
    Returns -1.0 when fewer than 2 clusters are present.
    """
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    N = len(labels)

    if K < 2:
        return -1.0

    # Degenerate: every turn is its own cluster.
    # Silhouette = 1.0 in this case (mathematically valid, semantically useless).
    # Also prevents the RuntimeWarning from np.where evaluating 0/0.
    if K >= N:
        return -1.0

    lbl_map = {old: new for new, old in enumerate(unique_labels)}
    mapped = np.array([lbl_map[l] for l in labels], dtype=np.int32)

    cluster_counts = np.bincount(mapped, minlength=K).astype(np.float64)

    one_hot = np.zeros((N, K), dtype=np.float64)
    one_hot[np.arange(N), mapped] = 1.0
    cluster_dist_sum = square_dist @ one_hot            # [N, K]

    own_sum = cluster_dist_sum[np.arange(N), mapped]
    own_count = cluster_counts[mapped]
    a = np.where(own_count > 1, own_sum / (own_count - 1), 0.0)

    mean_to_cluster = cluster_dist_sum / np.maximum(cluster_counts, 1.0)
    mean_to_cluster[np.arange(N), mapped] = np.inf
    b = mean_to_cluster.min(axis=1)

    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / denom, 0.0)
    return float(np.sum(s * durations) / max(durations.sum(), 1e-10))


def _compute_cluster_confidence(
    normed: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Per-cluster mean intra-cluster cosine similarity.

    Returns {"SPEAKER_XX": float} where 1.0 is perfectly cohesive.
    Clusters below 0.70 are acoustically diffuse.
    Singletons are assigned 1.0 by convention.
    """
    result: dict[str, float] = {}
    for k in np.unique(labels):
        mask = labels == k
        X_k = normed[mask]
        n = len(X_k)
        if n < 2:
            result[f"SPEAKER_{k:02d}"] = 1.0
            continue
        sim_mat = X_k @ X_k.T
        upper = sim_mat[np.triu_indices(n, k=1)]
        result[f"SPEAKER_{k:02d}"] = round(float(upper.mean()), 4)
    return result


def _compute_speaker_models(
    emb_matrix:         np.ndarray,
    labels:             np.ndarray,
    unique_turns:       list[dict],
    centroid_trim_percentile: float = DEFAULT_CENTROID_TRIM_PERCENTILE,
    min_turns_for_trim: int = DEFAULT_MIN_TURNS_FOR_TRIM,
) -> dict[str, dict]:
    """
    Two-pass robust centroid per speaker cluster.

    Pass 1  μ₀ = L2-norm( mean(X_k) )
            d_i = 1 − x_i · μ₀   cosine distance to draft centroid

    Pass 2  inliers = { x_i : d_i ≤ percentile(d, centroid_trim_percentile) }
            μ* = L2-norm( mean(inliers) )

    Trimming is skipped when M < min_turns_for_trim.
    """
    models: dict[str, dict] = {}

    for k in sorted(np.unique(labels).tolist()):
        mask = labels == k
        X_k = emb_matrix[mask].astype(np.float64)
        turns_k = [t for t, m in zip(unique_turns, mask) if m]
        M = len(X_k)

        total_speech = sum(float(t["end"]) - float(t["start"])
                           for t in turns_k)

        # Pass 1 — draft centroid
        raw_mu0 = X_k.mean(axis=0)
        norm0 = np.linalg.norm(raw_mu0)
        mu0 = (raw_mu0 / norm0) if norm0 > 1e-10 else X_k[0].copy()

        if M < min_turns_for_trim:
            centroid = mu0
            n_inliers = M
            n_outliers = 0
        else:
            # Pass 2 — trim periphery and recompute
            cos_dist = 1.0 - (X_k @ mu0)
            dist_cut = np.percentile(cos_dist, centroid_trim_percentile)
            inlier_mask = cos_dist <= dist_cut
            if not inlier_mask.any():
                inlier_mask[np.argmin(cos_dist)] = True

            X_in = X_k[inlier_mask]
            n_inliers = int(inlier_mask.sum())
            n_outliers = M - n_inliers

            raw_mu1 = X_in.mean(axis=0)
            norm1 = np.linalg.norm(raw_mu1)
            centroid = (raw_mu1 / norm1) if norm1 > 1e-10 else mu0

            if n_outliers:
                log.debug(
                    "  SPEAKER_%02d: %d turns  trimmed %d (%.0f%%)  "
                    "dist_cut=%.4f",
                    k, M, n_outliers, 100 * n_outliers / M, dist_cut,
                )

        models[f"SPEAKER_{k:02d}"] = {
            "embedding":        centroid.astype(np.float32).tolist(),
            "num_turns":        M,
            "num_inliers":      n_inliers,
            "num_outliers":     n_outliers,
            "total_speech_sec": round(total_speech, 3),
        }

    return models
