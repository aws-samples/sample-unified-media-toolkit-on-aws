"""
Full AHC clustering pipeline for StateGlobalSpeakersClustering.

Called by: states/global_speakers_clustering/lambda_function.py
Imports:   lib.cluster_sweep (_k_sweep and its tuning constants)
           lib.clustering    (_merge_small_clusters, _reindex) — lazy import inside functions

Functions
─────────
  _build_distance_matrix      L2-normalise embeddings → cosine distance (condensed + square + normed)
  _apply_snorm                symmetric score normalisation on a square distance matrix
  _remove_outliers            flag turns that are acoustically far from the pack; guarded to
                              never remove more than 3% or leave fewer than 4 turns
  _detect_single_speaker      two-criterion check: final dendrogram height + mean pairwise distance
  _reassign_outliers          locality-constrained re-assignment of removed turns
                              (lib/assign.py); no confident local match → label -1
                              (UNKNOWN downstream, never nearest-centroid fattening)
  _select_optimal_clustering  orchestrates all steps; returns (labels, turns, method_str, diagnostics)

Pipeline in _select_optimal_clustering
───────────────────────────────────────
  1. Build full cosine distance matrix on all HIGH embeddings
  2. Remove outliers (conservative: ≤ 3% of turns)
  3. Build AHC dendrogram on clean subset
  4. Single-speaker detection (fast exit)
  5a. Hard K override if cfg["num_speakers"] is set
  5b. K enumeration via _k_sweep (dual elbow + combined scoring)
  6. Reassign outlier turns to nearest centroid
"""
from __future__ import annotations

import logging

import numpy as np

from lib.input_schema import (
    DEFAULT_ASSIGN_LOCAL_WINDOW_SEC,
    DEFAULT_ASSIGN_MARGIN,
    DEFAULT_OUTLIER_GLOBAL_PERCENTILE,
    DEFAULT_MAX_OUTLIER_FRACTION,
    DEFAULT_SINGLE_SPEAKER_MERGE_THRESHOLD,
    DEFAULT_SINGLE_SPEAKER_MEAN_DIST,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MAX_K,
    DEFAULT_MIN_TURNS_PER_SPEAKER,
    DEFAULT_SIL_WEIGHT,
    DEFAULT_CH_WEIGHT,
    DEFAULT_K_PENALTY,
    DEFAULT_UNRELIABLE_SIL,
)
from lib.cluster_sweep import _k_sweep

log = logging.getLogger(__name__)


def _apply_snorm(square_dist: np.ndarray) -> np.ndarray:
    """
    Apply symmetric score normalisation to a cosine distance matrix.
    Converts distances to similarities first, normalises, converts back.
    """
    sim = 1.0 - square_dist                          # [N, N]

    N = len(sim)
    np.fill_diagonal(sim, np.nan)
    mean = np.nanmean(sim, axis=1)                   # [N]
    std = np.nanstd(sim,  axis=1).clip(1e-6)        # [N]
    np.fill_diagonal(sim, 1.0)

    norm_sim = 0.5 * (
        (sim - mean[:, None]) / std[:, None] +
        (sim - mean[None, :]) / std[None, :]
    )

    norm_dist = np.clip(1.0 - norm_sim, 0.0, 2.0)
    np.fill_diagonal(norm_dist, 0.0)
    return norm_dist


def _build_distance_matrix(
    emb_matrix:  np.ndarray,
    apply_snorm: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    L2-normalise embeddings and compute the cosine distance matrix.

    Returns
    ───────
    condensed  Upper-triangle vector [N*(N-1)/2] for scipy linkage.
    square     Full [N, N] symmetric matrix; reused for silhouette scoring.
    normed     L2-normalised float64 [N, D] embeddings.
    """
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True).clip(1e-10)
    normed = (emb_matrix / norms).astype(np.float64)
    square = np.clip(1.0 - normed @ normed.T, 0.0, 2.0)

    if apply_snorm and len(normed) >= 50:
        square = _apply_snorm(square)

    np.fill_diagonal(square, 0.0)
    condensed = square[np.triu_indices(len(normed), k=1)]
    return condensed, square, normed


def _cannot_link_violations(cannot_links, unique_turns,
                            final_turns, final_labels):
    """FINAL-position violation count: proven-distinct pairs sharing a
    cluster after EVERY stage (cut, mop-ups, outlier reassignment).
    Keyed by (start, end) because reassignment re-sorts turns; a pair
    with a rejected (-1) or unmapped member does not count — UNKNOWN
    is abstention, not co-clustering. Report-only: the 1e9 linkage
    distance makes constrained merges happen LAST, not never (a low
    user K cuts below the constraints, and a reassigned outlier can
    rejoin its cannot-partner's cluster)."""
    if not cannot_links:
        return 0

    def _key(t):
        return (round(float(t["start"]), 3), round(float(t["end"]), 3))

    label_of = {_key(t): int(lab)
                for t, lab in zip(final_turns, final_labels)}

    def _label(i):
        if i >= len(unique_turns):
            return -1
        return label_of.get(_key(unique_turns[i]), -1)

    n = 0
    for a, b in cannot_links:
        la, lb = _label(a), _label(b)
        if la >= 0 and la == lb:
            n += 1
    return n


def _remove_outliers(
    square_dist:  np.ndarray,
    unique_turns: list[dict],
    percentile:   float = DEFAULT_OUTLIER_GLOBAL_PERCENTILE,
    max_fraction: float = DEFAULT_MAX_OUTLIER_FRACTION,
) -> tuple[np.ndarray, int]:
    """
    Flag turns whose mean pairwise cosine distance exceeds `percentile`
    of the overall distribution.

    Guards
    ──────
    • Never removes more than floor(N * max_fraction) turns.
    • Always keeps at least 4 turns.
    • Removed turns are tracked for nearest-centroid re-assignment.

    Returns (keep_mask [N bool], n_removed int).
    """
    N = len(square_dist)
    if N < 4:
        return np.ones(N, dtype=bool), 0

    mean_dist = (square_dist.sum(1) - np.diag(square_dist)) / (N - 1)
    cutoff = np.percentile(mean_dist, percentile)
    outlier_mask = mean_dist > cutoff

    max_remove = max(0, int(N * max_fraction))
    if int(outlier_mask.sum()) > max_remove:
        worst = np.argsort(mean_dist)[::-1][:max_remove]
        outlier_mask = np.zeros(N, dtype=bool)
        outlier_mask[worst] = True

    n_remove = int(outlier_mask.sum())
    if (N - n_remove) < 4:
        allow = max(0, N - 4)
        worst = np.argsort(mean_dist)[::-1][:allow]
        outlier_mask = np.zeros(N, dtype=bool)
        outlier_mask[worst] = True
        n_remove = int(outlier_mask.sum())

    log.debug(
        "  _remove_outliers: N=%d  cutoff=%.4f  final_removed=%d",
        N, cutoff, n_remove,
    )
    return ~outlier_mask, n_remove


def _detect_single_speaker(
    heights:             np.ndarray,
    square_dist:         np.ndarray,
    merge_threshold:     float = DEFAULT_SINGLE_SPEAKER_MERGE_THRESHOLD,
    mean_dist_threshold: float = DEFAULT_SINGLE_SPEAKER_MEAN_DIST,
) -> bool:
    """
    Return True when the recording is likely a single speaker.

    Both criteria must hold:
      1. Z[-1, 2] < merge_threshold  — all turns merge at low cosine distance
      2. Mean pairwise cosine distance < mean_dist_threshold
    """
    N = len(square_dist)
    final_height = float(heights[-1])
    mean_dist = (float(square_dist[np.triu_indices(N, k=1)].mean())
                 if N > 1 else 0.0)

    log.info(
        "  Single-speaker check: "
        "Z_final=%.4f (thr=%.2f)  mean_dist=%.4f (thr=%.2f)",
        final_height, merge_threshold, mean_dist, mean_dist_threshold,
    )
    return final_height < merge_threshold and mean_dist < mean_dist_threshold


def _reassign_outliers(
    outlier_matrix:    np.ndarray,
    outlier_turns:     list[dict],
    speaker_centroids: dict[str, np.ndarray],
    clean_labels:      np.ndarray,
    clean_turns:       list[dict],
    window_sec:        float = DEFAULT_ASSIGN_LOCAL_WINDOW_SEC,
    margin:            float = DEFAULT_ASSIGN_MARGIN,
) -> tuple[np.ndarray, list[dict]]:
    """
    Assign each outlier turn to a speaker via locality-constrained
    matching (lib/assign.py): candidates are speakers with a clean turn
    near the outlier in time, and the winner must beat the runner-up by
    `margin` — otherwise the turn is REJECTED (label -1, rendered
    UNKNOWN downstream) instead of fattening the nearest big centroid.

    Returns (labels, turns) sorted by start time; rejected turns carry
    label -1.
    """
    from lib.assign import assign_speaker_local

    anchor_times = np.array(
        [float(t["start"]) for t in clean_turns], dtype=np.float64)
    anchor_names = [f"SPEAKER_{int(l):02d}" for l in clean_labels]

    norms_out = np.linalg.norm(
        outlier_matrix, axis=1, keepdims=True).clip(1e-10)
    normed_out = (outlier_matrix / norms_out).astype(np.float64)

    out_labels = np.empty(len(outlier_turns), dtype=np.int32)
    for i, turn in enumerate(outlier_turns):
        spk = assign_speaker_local(
            normed_out[i], float(turn["start"]),
            anchor_times, anchor_names, speaker_centroids,
            window_sec, margin)
        out_labels[i] = int(spk.split("_")[1]) if spk else -1

    all_turns = list(clean_turns) + list(outlier_turns)
    all_labels = np.concatenate([clean_labels, out_labels])

    order = np.argsort([float(t["start"]) for t in all_turns])
    all_turns = [all_turns[i] for i in order]
    all_labels = all_labels[order]

    return all_labels, all_turns


# Cannot-link distance injected into the LINKAGE matrix only (scoring
# keeps the honest unconstrained matrix). Must dominate average-linkage
# dilution: a single constrained pair between clusters of sizes n and m
# contributes LARGE/(n*m) to their merge distance, so LARGE must stay
# >> max cosine distance (2.0) even at n*m ~ 10^6.
CANNOT_LINK_DISTANCE = 1.0e9


def _select_optimal_clustering(
    emb_matrix:       np.ndarray,
    unique_turns:     list[dict],
    cfg:              dict,
    total_speech_sec: float,
    cannot_links:     list | None = None,
) -> tuple[np.ndarray, list[dict], str, dict]:
    """
    Full clustering pipeline.

    Pipeline steps
    ──────────────
    1.  Build full cosine distance matrix
    2.  Conservative outlier removal (tracked for re-assignment)
    3.  Build distance matrix + AHC dendrogram on clean subset
    4.  Single-speaker detection
    5a. Hard k override (if cfg["num_speakers"] is set)
    5b. K enumeration with combined silhouette + Calinski-Harabasz scoring
    6.  Reassign outlier turns to nearest speaker centroid

    Returns (labels, final_turns, method_used, diagnostics).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from lib.clustering import _merge_small_clusters, _reindex

    N = len(emb_matrix)

    num_speakers = cfg.get("num_speakers", None)
    min_cs = int(
        cfg.get("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE))
    outlier_pct = float(
        cfg.get("outlier_global_percentile", DEFAULT_OUTLIER_GLOBAL_PERCENTILE))
    single_spk_thr = float(
        cfg.get("single_speaker_threshold", DEFAULT_SINGLE_SPEAKER_MERGE_THRESHOLD))
    apply_snorm = bool(
        cfg.get("apply_snorm", False))

    # K enumeration config
    max_k = int(
        cfg.get("max_k", DEFAULT_MAX_K))
    min_turns_per_spk = int(
        cfg.get("min_turns_per_speaker", DEFAULT_MIN_TURNS_PER_SPEAKER))
    sil_weight = float(
        cfg.get("sil_weight", DEFAULT_SIL_WEIGHT))
    ch_weight = float(
        cfg.get("ch_weight", DEFAULT_CH_WEIGHT))
    k_penalty = float(
        cfg.get("k_penalty", DEFAULT_K_PENALTY))
    unreliable_sil = float(
        cfg.get("unreliable_silhouette", DEFAULT_UNRELIABLE_SIL))

    diag: dict = {}

    # ── Step 1: Full distance matrix ──────────────────────────────────────────
    _, square_full, _ = _build_distance_matrix(emb_matrix, apply_snorm=False)
    log.info("  Distance matrix: %d×%d", N, N)

    # ── Step 2: Outlier removal ───────────────────────────────────────────────
    keep_mask, n_removed = _remove_outliers(
        square_full, unique_turns, outlier_pct, DEFAULT_MAX_OUTLIER_FRACTION,
    )
    clean_turns = [unique_turns[i] for i in np.where(keep_mask)[0]]
    outlier_turns = [unique_turns[i] for i in np.where(~keep_mask)[0]]
    clean_matrix = emb_matrix[keep_mask]
    outlier_matrix = emb_matrix[~keep_mask]

    log.info(
        "  Outlier removal: %d removed  %d clean  (percentile=%.0f)",
        n_removed, len(clean_turns), outlier_pct,
    )
    diag["n_outliers_removed"] = n_removed
    diag["n_clean_turns"] = len(clean_turns)

    if len(clean_turns) < 2:
        log.warning(
            "  Too few clean turns (%d) after outlier removal — "
            "assigning all to SPEAKER_00", len(clean_turns),
        )
        # cross-talk evidence still proves >=2 voices — report honestly
        # even though too little clean data remains to act on it
        diag.update({
            "single_speaker_detected": not cannot_links,
            "k_selected":              1,
            "sweep_log":               [],
            "selection_reason":        "insufficient_data",
            "cannot_link_violations":  _cannot_link_violations(
                cannot_links, unique_turns,
                unique_turns, np.zeros(N, dtype=np.int32)),
        })
        return (np.zeros(N, dtype=np.int32), list(unique_turns),
                "single(insufficient_data)", diag)

    # ── Step 2b: remap cannot-links onto the clean subset ────────────────────
    # (P3) pairs are indices into unique_turns; outlier removal reindexes.
    # Pairs touching a removed outlier are dropped — the outlier is
    # assigned post-hoc and cannot participate in tree constraints.
    keep_idx = np.where(keep_mask)[0]
    old_to_new = {int(o): n for n, o in enumerate(keep_idx)}
    clean_links = [(old_to_new[a], old_to_new[b])
                   for a, b in (cannot_links or [])
                   if a in old_to_new and b in old_to_new]
    diag["n_cannot_links"] = len(clean_links)
    if cannot_links:
        log.info("  cannot-links: %d supplied, %d survive outlier removal",
                 len(cannot_links), len(clean_links))

    # ── Step 3: Dendrogram on clean subset ────────────────────────────────────
    # Constraints shape the LINKAGE TREE ONLY; sq_clean stays honest for
    # silhouette/CH scoring and centroids (1e9 entries would poison them)
    cond_clean, sq_clean, nrm_clean = _build_distance_matrix(
        clean_matrix, apply_snorm=apply_snorm,
    )
    if clean_links:
        sq_linkage = sq_clean.copy()
        for a, b in clean_links:
            sq_linkage[a, b] = sq_linkage[b, a] = CANNOT_LINK_DISTANCE
        cond_clean = sq_linkage[np.triu_indices(len(clean_matrix), k=1)]
    Z = linkage(cond_clean, method="average")
    heights = Z[:, 2]

    def _ht(idx: int) -> str:
        return (f"{heights[idx]:.4f}"
                if 0 < abs(idx) <= len(heights) else "n/a")

    log.info(
        "  Dendrogram (clean N=%d): Z_final=%.4f  "
        "natural: 2-spk∈(%s,%s)  3-spk∈(%s,%s)",
        len(clean_turns), heights[-1],
        _ht(-2), _ht(-1), _ht(-3), _ht(-2),
    )
    diag["dendrogram_final_height"] = round(float(heights[-1]), 4)

    # ── Step 4: Single-speaker detection ─────────────────────────────────────
    # cross-talk evidence PROVES >=2 voices — keyed on the SUPPLIED
    # links, not the post-outlier remap: evidence stands even when
    # every constrained pair fell to outlier removal. (A constrained
    # tree's final height is CANNOT_LINK_DISTANCE, which would also
    # trip the height check.)
    is_single = (False if cannot_links else
                 _detect_single_speaker(heights, sq_clean, single_spk_thr))
    diag["single_speaker_detected"] = is_single

    if is_single:
        log.info("  Single speaker detected — skipping K enumeration")
        clean_labels = np.zeros(len(clean_turns), dtype=np.int32)
        method_used = "single_speaker(auto)"
        diag.update({
            "k_selected":       1,
            "sweep_log":        [],
            "selection_reason": "single_speaker_auto",
        })

    elif num_speakers is not None:
        # ── Step 5a: Hard k override ──────────────────────────────────────────
        k = max(1, int(num_speakers))
        log.info("  num_speakers override: k=%d", k)
        raw = fcluster(Z, t=k, criterion="maxclust").astype(np.int32) - 1
        clean_labels = _reindex(_merge_small_clusters(
            raw, nrm_clean, min_cs, cannot_links=clean_links))
        method_used = f"ahc(maxclust={k},user_override)"
        diag.update({
            "k_selected":       k,
            "sweep_log":        [],
            "selection_reason": "user_override",
        })

    else:
        # ── Step 5b: K enumeration ────────────────────────────────────────────
        clean_labels, best_k, selection_reason, sweep_log = _k_sweep(
            Z=Z,
            square_dist=sq_clean,
            normed=nrm_clean,
            clean_turns=clean_turns,
            min_cs=min_cs,
            max_k=max_k,
            min_turns_per_speaker=min_turns_per_spk,
            sil_weight=sil_weight,
            ch_weight=ch_weight,
            k_penalty=k_penalty,
            unreliable_silhouette=unreliable_sil,
            elbow_plateau_pct=float(cfg.get("elbow_plateau_pct", 0.03)),
            k_selection_method=str(cfg.get("k_selection_method", "elbow")),
            cannot_links=clean_links,
        )
        method_used = f"ahc(k_enum  k={best_k})"
        diag.update({
            "k_selected":       best_k,
            "selection_reason": selection_reason,
            "sweep_log":        sweep_log,
        })

    # ── Step 6: Reassign outliers ─────────────────────────────────────────────
    if n_removed > 0:
        n_clean_spk = int(clean_labels.max()) + 1 if len(clean_labels) else 1
        clean_centroids: dict[str, np.ndarray] = {}
        for k in range(n_clean_spk):
            mask_k = clean_labels == k
            X_k = nrm_clean[mask_k]
            if len(X_k) == 0:
                continue
            mu = X_k.mean(axis=0)
            nm = np.linalg.norm(mu)
            clean_centroids[f"SPEAKER_{k:02d}"] = (
                mu / nm if nm > 1e-10 else X_k[0]
            )
        final_labels, final_turns = _reassign_outliers(
            outlier_matrix, outlier_turns,
            clean_centroids, clean_labels, clean_turns,
            window_sec=float(cfg.get("assign_local_window_sec",
                                     DEFAULT_ASSIGN_LOCAL_WINDOW_SEC)),
            margin=float(cfg.get("assign_margin", DEFAULT_ASSIGN_MARGIN)),
        )
        n_rejected = int((final_labels < 0).sum())
        diag["n_outliers_rejected"] = n_rejected
        log.info(
            "  Outlier reassignment: %d turns → local match, %d rejected",
            len(outlier_turns), n_rejected,
        )
    else:
        final_labels = clean_labels
        final_turns = clean_turns

    # ── Step 7: Cannot-link violation check (report-only, FINAL position:
    # after the cut, the mop-ups AND outlier reassignment — a reassigned
    # outlier rejoining its cannot-partner's cluster counts too) ─────────
    if cannot_links:
        n_viol = _cannot_link_violations(cannot_links, unique_turns,
                                         final_turns, final_labels)
        diag["cannot_link_violations"] = n_viol
        if n_viol:
            log.warning("  %d/%d cannot-links VIOLATED in the final "
                        "assignment", n_viol, len(cannot_links))

    return final_labels, final_turns, method_used, diag
