# lib/cluster_sweep.py
"""
K-enumeration sweep with dual scoring for StateGlobalSpeakersClustering.

Called by: lib.cluster_pipeline._select_optimal_clustering (via _k_sweep)

Two scoring methods run in parallel on every candidate K:

  Combined
  ────────
  combined = sil_weight * norm(sil) + ch_weight * norm(CH) - k_penalty * (k / k_max)
  Tends to favour lower K when one dominant speaker has disproportionately many
  turns, because Calinski-Harabász rewards compact, well-separated clusters.

  Elbow
  ─────
  Finds the lowest K whose silhouette score is within elbow_plateau_pct (default 3%)
  of the global silhouette maximum.  "Enter the plateau early" — structurally, the
  last meaningful cluster split is usually where the silhouette curve first flattens.

Selection
─────────
  Both methods produce a K.  If they agree, that K is used unconditionally.
  If they disagree, k_selection_method picks the winner:
    "elbow"    (default) — take the elbow result
    "combined"           — take the combined result
  Both K values and the disagreement flag are preserved in the sweep_log for
  offline analysis.

Guardrails (validated by replaying 10 production sweeps
offline, 3 moved / 7 unchanged — see README.md Design record P1)
────────────────────────────────────────────────────────────────────────
  • Weak or flat silhouette curve carries no elbow signal: the plateau
    rule then collapses to the lowest k (one measured run selected k=2
    and put 92% of the episode's speech in one cluster).  When the best
    silhouette is below unreliable_silhouette, or the curve's range is
    below _FLAT_SIL_RANGE, a disagreement resolves to the LARGER of the
    two candidate K's regardless of k_selection_method — in the
    no-signal regime the downstream cost is asymmetric (measured):
    cast-resolution can weld over-split clusters back together, but an
    over-merged cluster can only have its names stripped.
  • Still climbing at the ceiling: when the last candidate's combined
    score is within _CLIMBING_FRACTION of the maximum AND the ceiling was
    imposed by max_k (not by N // min_turns_per_speaker), the ceiling is
    raised once to _CEILING_RAISE_FACTOR * max_k and the sweep extends.
    If the curve is still climbing after that, the selection_reason
    carries "still_climbing_at_ceiling" so the run is auditable.

Exported constants (imported by cluster_pipeline and global_speakers_clustering):
  _MAX_K, _MIN_TURNS_PER_SPEAKER, _SIL_WEIGHT, _CH_WEIGHT, _K_PENALTY,
  _UNRELIABLE_SIL, _ELBOW_PLATEAU_PCT, _K_SELECTION_METHOD
"""
from __future__ import annotations

import logging

import numpy as np

from lib.input_schema import (
    DEFAULT_MAX_K,
    DEFAULT_MIN_TURNS_PER_SPEAKER,
    DEFAULT_SIL_WEIGHT,
    DEFAULT_CH_WEIGHT,
    DEFAULT_K_PENALTY,
    DEFAULT_UNRELIABLE_SIL,
    DEFAULT_ELBOW_PLATEAU_PCT,
    DEFAULT_K_SELECTION_METHOD,
)
from lib.cluster_score import _weighted_silhouette

log = logging.getLogger(__name__)

# K-selection guardrails. The thresholds are the same tests
# test/diagnose_workflow.py already applies when diagnosing a finished
# run; here they gate the selection itself (see module docstring).
_FLAT_SIL_RANGE = 0.05      # silhouette range below this = no elbow signal
_CLIMBING_FRACTION = 0.98   # last combined within 98% of max = climbing
_CEILING_RAISE_FACTOR = 2   # raise a max_k-imposed ceiling once, then flag


def _calinski_harabasz(
    square_dist: np.ndarray,
    labels:      np.ndarray,
    normed:      np.ndarray,
) -> float:
    """
    Calinski-Harabasz index (Variance Ratio Criterion).
    Higher is better. Returns 0.0 for degenerate cases.
    """
    N = len(labels)
    K = int(labels.max()) + 1

    if K < 2 or K >= N:
        return 0.0

    grand_mean = normed.mean(axis=0)
    B = 0.0
    W = 0.0

    for k in range(K):
        mask = labels == k
        X_k = normed[mask]
        n_k = len(X_k)
        if n_k == 0:
            continue
        mu_k = X_k.mean(axis=0)
        diff = mu_k - grand_mean
        B += n_k * float(diff @ diff)
        W += float(np.sum((X_k - mu_k) ** 2))

    if W < 1e-12 or (N - K) == 0:
        return 0.0

    return (B / (K - 1)) / (W / (N - K))


def _k_sweep(
    Z:                     np.ndarray,
    square_dist:           np.ndarray,
    normed:                np.ndarray,
    clean_turns:           list[dict],
    min_cs:                int,
    max_k:                 int = DEFAULT_MAX_K,
    min_turns_per_speaker: int = DEFAULT_MIN_TURNS_PER_SPEAKER,
    sil_weight:            float = DEFAULT_SIL_WEIGHT,
    ch_weight:             float = DEFAULT_CH_WEIGHT,
    k_penalty:             float = DEFAULT_K_PENALTY,
    unreliable_silhouette: float = DEFAULT_UNRELIABLE_SIL,
    elbow_plateau_pct:     float = DEFAULT_ELBOW_PLATEAU_PCT,
    k_selection_method:    str = DEFAULT_K_SELECTION_METHOD,
    cannot_links:          list | None = None,
) -> tuple[np.ndarray, int, str, list[dict]]:
    """
    Enumerate K from 2 to k_max, score each assignment with both combined
    and elbow methods, return the best.

    Returns (labels, k_selected, selection_reason, sweep_log)
    """
    from scipy.cluster.hierarchy import fcluster
    from lib.clustering import _merge_small_clusters, _reindex

    N = len(clean_turns)

    durations = np.array(
        [float(t.get("duration", float(t["end"]) - float(t["start"])))
         for t in clean_turns],
        dtype=np.float64,
    )

    # ── K range ───────────────────────────────────────────────────────────────
    min_tps = max(2, min_turns_per_speaker)
    # hard_cap: more clusters than this cannot honour min_turns_per_speaker;
    # a max_k-imposed ceiling can be raised (guardrail), hard_cap never
    hard_cap = max(2, N // min_tps)
    k_hi = min(max_k, hard_cap)
    log.info(
        "  [k_sweep] N=%d  k_range=[2..%d]  method=%s  "
        "weights=(sil=%.1f, ch=%.1f)  penalty=%.3f  plateau=%.1f%%",
        N, k_hi, k_selection_method,
        sil_weight, ch_weight, k_penalty, elbow_plateau_pct * 100,
    )

    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-10:
            return np.ones_like(arr)
        return (arr - lo) / (hi - lo)

    candidates: list[dict] = []
    valid: list[dict] = []
    combined = np.array([], dtype=np.float64)
    sil_vals = np.array([], dtype=np.float64)
    ceiling_raised = False
    k_lo = 2

    while True:
        for k in range(k_lo, k_hi + 1):
            raw = fcluster(
                Z, t=k, criterion="maxclust").astype(np.int32) - 1
            merged = _reindex(_merge_small_clusters(
                raw, normed, min_cs, cannot_links=cannot_links))
            k_act = int(merged.max()) + 1 if len(merged) else 0

            if k_act < 2:
                sil = -1.0
                ch = 0.0
            else:
                sil = round(_weighted_silhouette(
                    square_dist, merged, durations), 4)
                ch = round(_calinski_harabasz(
                    square_dist, merged, normed), 4)

            candidates.append({
                "k":        k,
                "k_act":    k_act,
                "sil":      sil,
                "ch":       ch,
                "combined": None,
                "labels":   merged.copy(),
            })

        # candidates stay k-ordered, so valid[-1]/combined[-1] is the
        # largest explored k — what the climbing test needs
        valid = [c for c in candidates
                 if c["sil"] > -1.0 and c["k_act"] >= 2]

        # ── violation veto (2026-09-29, ⚠unreliable regime only) ──────
        # cannot-links are machine-held PROOF of distinctness; in a
        # no-signal curve regime both selection methods can agree on a
        # K whose cut merges proven-distinct pairs (a proven-3 asset
        # shipped K=2 with 8 violations, report-only). Selection must
        # not contradict proof: when the curve is unreliable/flat,
        # candidates that violate links are disqualified; if every
        # candidate violates, prefer the fewest violations. Healthy
        # curves are untouched.
        veto_applied = False
        if cannot_links and valid:
            _sils = [c["sil"] for c in valid]
            _unreliable = (max(_sils) < unreliable_silhouette
                           or (max(_sils) - min(_sils)) < _FLAT_SIL_RANGE)
            if _unreliable:
                for c in valid:
                    lb = c["labels"]
                    c["violations"] = int(sum(
                        1 for a, b in cannot_links
                        if a < len(lb) and b < len(lb)
                        and lb[a] == lb[b] and lb[a] >= 0))
                clean = [c for c in valid if c["violations"] == 0]
                if clean and len(clean) < len(valid):
                    valid = clean
                    veto_applied = True
                elif not clean:
                    vmin = min(c["violations"] for c in valid)
                    kept = [c for c in valid if c["violations"] == vmin]
                    if len(kept) < len(valid):
                        valid = kept
                        veto_applied = True
                if veto_applied:
                    log.info("  [k_sweep] violation veto: %d candidates "
                             "remain (unreliable curve, cannot-link "
                             "consistency enforced)", len(valid))
        if not valid:
            break

        # Combined scoring — min-max over the FULL candidate set, so an
        # extension recomputes every score, not just the new tail
        sil_vals = np.array([c["sil"] for c in valid], dtype=np.float64)
        ch_vals = np.array([c["ch"] for c in valid], dtype=np.float64)
        k_vals = np.array([c["k"] for c in valid], dtype=np.float64)
        penalty = k_penalty * (k_vals / k_hi)
        combined = (sil_weight * _minmax(sil_vals)
                    + ch_weight * _minmax(ch_vals) - penalty)
        for i, c in enumerate(valid):
            c["combined"] = round(float(combined[i]), 6)

        # Guardrail: still climbing at a max_k-imposed ceiling → raise once
        if (not ceiling_raised
                and k_hi < hard_cap
                and float(combined[-1])
                >= _CLIMBING_FRACTION * float(combined.max())):
            k_lo = k_hi + 1
            k_hi = min(_CEILING_RAISE_FACTOR * max_k, hard_cap)
            ceiling_raised = True
            log.info(
                "  [k_sweep] still climbing at ceiling — extending to "
                "k_range=[%d..%d]", k_lo, k_hi,
            )
            continue
        break

    climbing_at_final = bool(len(combined)) and (
        float(combined[-1]) >= _CLIMBING_FRACTION * float(combined.max()))

    if not valid:
        log.warning("  [k_sweep] no valid candidates — falling back to K=2")
        fallback = candidates[0] if candidates else {
            "k": 2, "k_act": 2, "sil": -1.0, "ch": 0.0, "combined": -1.0,
            "labels": fcluster(
                Z, t=2, criterion="maxclust").astype(np.int32) - 1,
        }
        sweep_log = [
            {"k": c["k"], "k_act": c["k_act"], "sil": c["sil"],
             "ch": c["ch"], "combined": None}
            for c in candidates
        ]
        return (fallback["labels"], fallback["k"],
                "fallback_no_valid_candidates", sweep_log)

    # ── Method 1: Combined scoring (computed in the sweep loop above) ───────
    best_combined_idx = int(np.argmax(combined))
    best_combined = valid[best_combined_idx]
    k_combined = best_combined["k_act"]

    log.info(
        "  [k_sweep] COMBINED: k=%d  k_act=%d  sil=%.4f  ch=%.2f  "
        "combined=%.4f",
        best_combined["k"], k_combined,
        best_combined["sil"], best_combined["ch"],
        best_combined["combined"],
    )

    # ── Method 2: Elbow detection (silhouette plateau) ────────────────────────
    # Deduplicate by k_act — keep the entry with highest silhouette for each
    seen_k_act: dict[int, dict] = {}
    for c in valid:
        ka = c["k_act"]
        if ka not in seen_k_act or c["sil"] > seen_k_act[ka]["sil"]:
            seen_k_act[ka] = c

    deduped = sorted(seen_k_act.values(), key=lambda c: c["k_act"])
    deduped_sils = [c["sil"] for c in deduped]
    sil_max = max(deduped_sils)
    plateau_threshold = sil_max * (1.0 - elbow_plateau_pct)

    # Elbow = lowest k_act in the plateau
    plateau_candidates = [
        c for c in deduped if c["sil"] >= plateau_threshold
    ]

    if plateau_candidates:
        elbow_candidate = min(plateau_candidates, key=lambda c: c["k_act"])
        k_elbow = elbow_candidate["k_act"]
    else:
        # No plateau found — fall back to argmax
        elbow_candidate = max(deduped, key=lambda c: c["sil"])
        k_elbow = elbow_candidate["k_act"]

    log.info(
        "  [k_sweep] ELBOW: k_act=%d  sil=%.4f  "
        "(plateau_threshold=%.4f  plateau_size=%d  sil_max=%.4f)",
        k_elbow, elbow_candidate["sil"],
        plateau_threshold, len(plateau_candidates), sil_max,
    )

    # Log deduped silhouette curve for diagnostics
    for c in deduped:
        marker = ""
        if c["k_act"] == k_elbow:
            marker = " ← ELBOW"
        if c["k_act"] == k_combined:
            marker += " ← COMBINED"
        in_plateau = "✓" if c["sil"] >= plateau_threshold else " "
        log.info(
            "  [k_sweep]   k_act=%2d  sil=%.4f  %s%s",
            c["k_act"], c["sil"], in_plateau, marker,
        )

    # ── Selection ─────────────────────────────────────────────────────────────
    # Guardrail: a weak or flat silhouette curve has no elbow signal —
    # the plateau rule then collapses to the lowest k. Resolve for
    # combined instead of trusting k_selection_method.
    sil_max_all = float(sil_vals.max())
    sil_range_all = float(sil_vals.max() - sil_vals.min())
    weak_curve = sil_max_all < unreliable_silhouette
    flat_curve = sil_range_all < _FLAT_SIL_RANGE

    if k_combined == k_elbow:
        selected_k = k_combined
        # Find the labels for this k_act
        selected_c = seen_k_act[selected_k]
        reason = (
            f"combined+elbow agree  k={selected_k}  "
            f"sil={selected_c['sil']:.4f}  ch={selected_c['ch']:.2f}"
        )
        log.info("  [k_sweep] AGREE: k_act=%d", selected_k)
    else:
        if weak_curve or flat_curve:
            # No-signal regime: neither selector is trustworthy, so use
            # the downstream cost asymmetry instead (measured on the
            # 2026-09-15 KPI run): cast-resolution welds over-SPLIT
            # clusters back together, but an over-MERGED cluster can
            # only have its names stripped — one merged run lost 2/3 of
            # its named speech while an over-split run improved. Prefer
            # the LARGER of the two disagreeing K's.
            selected_k = max(k_elbow, k_combined)
            selected_c = (best_combined if selected_k == k_combined
                          else seen_k_act[selected_k])
            why = (f"sil_max={sil_max_all:.3f}<{unreliable_silhouette:.2f}"
                   if weak_curve else
                   f"sil_range={sil_range_all:.3f}<{_FLAT_SIL_RANGE:.2f}")
            reason = (
                f"max-k selected (guardrail {why}, prefer-split)  "
                f"k_combined={k_combined}  k_elbow={k_elbow}"
            )
        elif k_selection_method == "elbow":
            selected_k = k_elbow
            selected_c = seen_k_act[selected_k]
            reason = (
                f"elbow selected (disagreement)  k_elbow={k_elbow}  "
                f"k_combined={k_combined}  sil={selected_c['sil']:.4f}"
            )
        else:
            selected_k = k_combined
            selected_c = best_combined
            reason = (
                f"combined selected (disagreement)  k_combined={k_combined}  "
                f"k_elbow={k_elbow}  "
                f"sil={selected_c['sil']:.4f}  ch={selected_c['ch']:.2f}"
            )
        log.warning(
            "  [k_sweep] DISAGREE: k_combined=%d  k_elbow=%d  "
            "→ using %s (method=%s)",
            k_combined, k_elbow, selected_k,
            "guardrail" if (weak_curve or flat_curve)
            else k_selection_method,
        )

    # Unreliable warning
    if selected_c["sil"] < unreliable_silhouette:
        reason += "  ⚠unreliable"
        log.warning(
            "  [k_sweep] best silhouette=%.4f < %.2f — flagged unreliable",
            selected_c["sil"], unreliable_silhouette,
        )

    # Ceiling audit trail
    if ceiling_raised:
        reason += f"  ceiling_raised→{k_hi}"
    if climbing_at_final:
        reason += "  ⚠still_climbing_at_ceiling"
        log.warning(
            "  [k_sweep] combined score still climbing at k=%d — "
            "K may be an under-count", k_hi,
        )

    # ── Sweep log ─────────────────────────────────────────────────────────────
    valid_by_k = {c["k"]: c for c in valid}
    sweep_log = [
        {
            "k":        c["k"],
            "k_act":    c["k_act"],
            "sil":      c["sil"],
            "ch":       c["ch"],
            "combined": (valid_by_k[c["k"]]["combined"]
                         if c["k"] in valid_by_k else None),
        }
        for c in candidates
    ]

    if veto_applied:
        reason += "  +violation_veto"
    return selected_c["labels"], selected_k, reason, sweep_log
