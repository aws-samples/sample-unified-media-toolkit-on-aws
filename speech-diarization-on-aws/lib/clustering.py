# lib/clustering.py
"""
Speaker clustering backends — AHC and VBx.

Called by: lib.cluster_pipeline (lazy imports of _merge_small_clusters, _reindex)
           lib.cluster_sweep    (lazy imports of _merge_small_clusters, _reindex)

Note: VBx is present but disabled in the pipeline. input_schema.py constrains
_VALID_ALGORITHMS to {"ahc"} only. cluster_vbx() is correct and can be re-enabled
by adding "vbx" to that set; it is not called today.

Public functions
────────────────
  cluster(embeddings, method, **kwargs)   dispatch to AHC or VBx
  cluster_ahc(embeddings, **kwargs)       scipy linkage/fcluster on cosine distance
  cluster_vbx(embeddings, **kwargs)       GMM Variational Bayes EM (AHC-initialised)

Private helpers (imported by cluster_pipeline and cluster_sweep)
────────────────────────────────────────────────────────────────
  _merge_small_clusters(labels, normed, min_size)
      Reassign embeddings from clusters smaller than min_size to the nearest
      large cluster by cosine similarity to centroid.

  _reindex(labels)
      Remap labels to contiguous 0-based integers after merging.

  _apply_lda(embeddings, lda_dim, init_threshold)
      Optional LDA projection using AHC-proxy class labels. Only called by
      cluster_vbx when lda_dim > 0.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def cluster(
    embeddings: np.ndarray,
    method:     str = "vbx",
    **kwargs,
) -> np.ndarray:
    """
    Cluster [N, D] embeddings → int32[N] labels.

    Parameters
    ──────────
    method : "ahc" or "vbx"  (case-insensitive)
    kwargs : forwarded to cluster_ahc() or cluster_vbx() respectively.
             Unknown keys are silently ignored so the caller can pass
             a merged config dict without filtering.

    AHC kwargs
    ──────────
      threshold        float  default 0.350   cosine-distance cut
      method           str    default "average"
      min_cluster_size int    default 15

    VBx kwargs
    ──────────
      fa               float  default 0.3     acoustic weight
      fb               float  default 17.0    Dirichlet prior
      lda_dim          int    default 0       LDA projection (0 = skip)
      max_iter         int    default 40
      init_threshold   float  default 0.350   AHC threshold for VBx init
      min_cluster_size int    default 15
    """
    m = method.lower().strip()
    if m == "ahc":
        ahc_keys = {"threshold", "method", "min_cluster_size"}
        return cluster_ahc(embeddings, **{k: v for k, v in kwargs.items()
                                          if k in ahc_keys})
    if m == "vbx":
        vbx_keys = {"fa", "fb", "lda_dim", "max_iter",
                    "init_threshold", "min_cluster_size"}
        return cluster_vbx(embeddings, **{k: v for k, v in kwargs.items()
                                          if k in vbx_keys})
    raise ValueError(
        f"Unknown clustering method '{method}'. Choose 'ahc' or 'vbx'."
    )


# =============================================================================
# AHC
# =============================================================================

def cluster_ahc(
    embeddings:       np.ndarray,
    threshold:        float = 0.350,        # updated default (see below)
    method:           str = "average",      # ← fixed
    min_cluster_size: int = 15,
) -> np.ndarray:
    """
    Agglomerative Hierarchical Clustering on L2-normalised cosine distance.

    Strengths
    ─────────
    • O(N² log N), deterministic, no hyperparameter iteration.
    • Matches pyannote 3.1 default for short/2-3-speaker recordings.

    Weaknesses
    ──────────
    • Single global threshold — mis-tuning causes systematic over/under-cluster.
    • Hard assignments on borderline embeddings.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    N = len(embeddings)
    if N == 0:
        return np.array([], dtype=np.int32)
    if N == 1:
        return np.array([0], dtype=np.int32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-10)
    normed = (embeddings / norms).astype(np.float64)

    cos_dist = np.clip(1.0 - normed @ normed.T, 0.0, 2.0)
    np.fill_diagonal(cos_dist, 0.0)
    condensed = cos_dist[np.triu_indices(N, k=1)]

    Z = linkage(condensed, method=method)
    labels = fcluster(
        Z, t=threshold, criterion="distance").astype(np.int32) - 1

    labels = _merge_small_clusters(labels, normed, min_cluster_size)
    return _reindex(labels)


# =============================================================================
# VBx
# =============================================================================

def cluster_vbx(
    embeddings:       np.ndarray,
    fa:               float = 0.3,
    fb:               float = 17.0,
    lda_dim:          int = 0,
    max_iter:         int = 40,
    init_threshold:   float = 0.350,
    min_cluster_size: int = 15,
    epsilon:          float = 1e-6,
) -> np.ndarray:
    """
    GMM-VBx: AHC initialisation + Variational Bayes EM.

    Mathematical summary
    ────────────────────
    Models x-vectors with a spherical-covariance GMM:
      p(x | k) ≈ N(x ; μ_k , (D/fa)·I)

    VB-EM:
      E-step: q(z_n=k) ∝ exp( fb·log(N_k) + fa·(x_n·μ_k) − ½·fa·‖μ_k‖² )
      M-step: μ_k = Σ_n q_nk·x_n / N_k

    Speakers with N_k < 1/K are pruned each iteration.

    Strengths over AHC
    ──────────────────
    • Soft assignments → more stable on borderline embeddings.
    • Automatic speaker pruning — no single threshold controls count.
    • fb acts as a smooth global regulariser.
    • Better on 4+ speakers and long recordings.

    Trade-offs
    ──────────
    • 2–5× slower than AHC for large N (still sub-second on Lambda).
    • fa/fb require tuning; defaults suit WeSpeaker 256-d embeddings.
    """
    N, D = embeddings.shape
    if N == 0:
        return np.array([], dtype=np.int32)
    if N == 1:
        return np.array([0], dtype=np.int32)

    # ── Optional LDA projection ───────────────────────────────────────────────
    if lda_dim > 0 and lda_dim < D:
        embeddings = _apply_lda(embeddings, lda_dim, init_threshold)
        D = embeddings.shape[1]

    # L2 normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-10)
    X = (embeddings / norms).astype(np.float64)   # [N, D]

    # ── AHC initialisation (deliberately under-clusters) ─────────────────────
    init_labels = cluster_ahc(
        embeddings, threshold=init_threshold,
        method="average", min_cluster_size=1,    # ← fixed
    )
    K = int(init_labels.max()) + 1
    log.debug(f"VBx init: K={K} clusters from AHC threshold={init_threshold}")

    # Hard-init responsibilities
    q = np.zeros((N, K), dtype=np.float64)
    for n, k in enumerate(init_labels):
        q[n, k] = 1.0

    lam = fa / D   # acoustic precision (spherical covariance approximation)

    # ── VB-EM ─────────────────────────────────────────────────────────────────
    for iteration in range(max_iter):
        q_old = q.copy()

        # M-step
        Nk = q.sum(0)                                               # [K]
        mu = (q.T @ X) / np.maximum(Nk[:, None], epsilon)          # [K, D]

        # E-step
        log_prior = fb * np.log(np.maximum(Nk, epsilon))            # [K]
        log_like = lam * (X @ mu.T) - 0.5 * lam * (mu ** 2).sum(1)  # [N,K]
        log_q = log_prior[None, :] + log_like

        # Numerically stable softmax
        log_q -= log_q.max(axis=1, keepdims=True)
        q = np.exp(log_q)
        q /= q.sum(axis=1, keepdims=True)

        # Prune speakers with negligible responsibility
        active = Nk > (1.0 / K)
        if active.sum() < K:
            K = int(active.sum())
            q = q[:, active]
            q /= q.sum(axis=1, keepdims=True)
            log.debug(f"VBx iter {iteration}: pruned to K={K}")
            if K == 1:
                break

        # Convergence check
        if K == q_old.shape[1] and np.abs(q - q_old).max() < 1e-4:
            log.debug(f"VBx converged at iteration {iteration}")
            break

    labels = q.argmax(axis=1).astype(np.int32)
    norms_64 = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-10)
    normed_64 = X / norms_64
    labels = _merge_small_clusters(labels, normed_64, min_cluster_size)
    return _reindex(labels)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _apply_lda(
    embeddings:     np.ndarray,
    lda_dim:        int,
    init_threshold: float,
) -> np.ndarray:
    """Project embeddings via LDA using AHC labels as proxy classes."""
    try:
        from scipy.linalg import eigh
    except ImportError:
        log.warning("scipy not available — skipping LDA projection")
        return embeddings

    init_lbls = cluster_ahc(embeddings, threshold=init_threshold,
                            min_cluster_size=1)
    D = embeddings.shape[1]
    X64 = embeddings.astype(np.float64)
    grand_mean = X64.mean(0)
    S_b = np.zeros((D, D), dtype=np.float64)
    S_w = np.zeros((D, D), dtype=np.float64)

    for k in np.unique(init_lbls):
        Xk = X64[init_lbls == k]
        mk = Xk.mean(0)
        nk = len(Xk)
        diff = (mk - grand_mean)[:, None]
        S_b += nk * (diff @ diff.T)
        S_w += (Xk - mk).T @ (Xk - mk)

    S_w += np.eye(D) * 1e-6
    try:
        _, vecs = eigh(S_b, S_w, subset_by_index=[D - lda_dim, D - 1])
        W = vecs[:, ::-1][:, :lda_dim]
        projected = (embeddings.astype(np.float64) @ W).astype(np.float32)
        log.info(f"LDA: {D}→{lda_dim} dims")
        return projected
    except Exception as exc:
        log.warning(f"LDA failed ({exc}) — using original embeddings")
        return embeddings


def _merge_small_clusters(
    labels:           np.ndarray,
    normed_embeddings: np.ndarray,
    min_cluster_size: int,
    cannot_links:     list | None = None,
) -> np.ndarray:
    """
    Reassign embeddings from clusters smaller than min_cluster_size
    to their nearest large cluster (by cosine similarity to centroid).

    cannot_links (P3): (i, j) index pairs that may never share a
    cluster. A small-cluster member is never reassigned into a cluster
    containing one of its cannot-partners — the next-best allowed
    cluster wins, or the member keeps its (small) cluster: violating a
    proven two-voices fact to satisfy a size heuristic would rebuild
    the exact merge the constraint forbids.
    """
    if min_cluster_size <= 1:
        return labels

    unique, counts = np.unique(labels, return_counts=True)
    small_lbls = unique[counts < min_cluster_size]
    large_lbls = unique[counts >= min_cluster_size]

    if len(small_lbls) == 0 or len(large_lbls) == 0:
        return labels

    partners: dict[int, set[int]] = {}
    for a, b in cannot_links or []:
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)

    centroids = {
        lbl: normed_embeddings[labels == lbl].mean(0)
        for lbl in large_lbls
    }
    labels = labels.copy()
    for i in np.where(np.isin(labels, small_lbls))[0]:
        forbidden = {int(labels[p]) for p in partners.get(int(i), ())}
        best_lbl, best_sim = -1, -np.inf
        for lbl, ctr in centroids.items():
            if int(lbl) in forbidden:
                continue
            norm = np.linalg.norm(ctr)
            sim = float(normed_embeddings[i] @ (ctr / max(norm, 1e-10)))
            if sim > best_sim:
                best_sim, best_lbl = sim, lbl
        if best_lbl != -1:
            labels[i] = best_lbl

    return labels


def _reindex(labels: np.ndarray) -> np.ndarray:
    """Remap labels to contiguous 0-indexed integers."""
    mapping = {old: new for new, old in enumerate(np.unique(labels))}
    return np.array([mapping[l] for l in labels], dtype=np.int32)
