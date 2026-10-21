#!/usr/bin/env python3
"""Unit tests for the K-selection guardrails in lib/cluster_sweep.py
(README.md Design record P1). Self-contained — synthetic
embeddings, no AWS, no models.

    python3 test/test_k_sweep_guardrails.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy.cluster.hierarchy import linkage          # noqa: E402
from lib.cluster_pipeline import _build_distance_matrix  # noqa: E402
from lib.cluster_sweep import _k_sweep               # noqa: E402


def make_embeddings(n_clusters, per_cluster, noise, dim=16, seed=7):
    """Gaussian blobs on well-separated axes; noise controls overlap."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * 4.0
    rows = [centers[k] + rng.normal(size=dim) * noise
            for k in range(n_clusters) for _ in range(per_cluster)]
    emb = np.array(rows, dtype=np.float32)
    turns = [{"start": float(i), "end": float(i) + 1.0}
             for i in range(len(emb))]
    return emb, turns


def sweep(emb, turns, **kw):
    cond, square, normed = _build_distance_matrix(emb)
    Z = linkage(cond, method="average")
    return _k_sweep(Z=Z, square_dist=square, normed=normed,
                    clean_turns=turns, min_cs=2, **kw)


def test_clean_data_untouched():
    """Well-separated clusters: correct K, no guardrail, no ceiling."""
    emb, turns = make_embeddings(3, 15, noise=0.15)
    labels, k, reason, _ = sweep(emb, turns)
    assert k == 3, f"expected K=3, got {k} ({reason})"
    assert "guardrail" not in reason, reason
    assert "ceiling_raised" not in reason, reason
    print(f"  ok clean-data: K={k}")


def test_weak_curve_resolves_to_combined():
    """A weak curve (best silhouette below the unreliable floor) with
    method disagreement must resolve via the guardrail to the LARGER
    candidate K, never fall through to k_selection_method="elbow".
    Collapse shape: the elbow plateau rule chose k=2 on a real episode
    (92% of speech in one cluster). Prefer-split shape: choosing the
    LOWER candidate on another real episode merged distinct people and
    cost two-thirds of its named speech — over-split is recoverable
    downstream, over-merge is not.

    Construction: one dominant long-duration cluster + 5 overlapping
    small ones (seed pinned to a deterministic disagreement); the
    unreliable floor is forced high so the guardrail path is taken."""
    r = np.random.default_rng(3)
    dim = 16
    centers = r.normal(size=(6, dim)) * 2.0
    rows, turns = [], []
    for c in range(6):
        n = 40 if c == 0 else 8
        for _ in range(n):
            rows.append(centers[c] + r.normal(size=dim) * 1.8)
            turns.append({"start": float(len(rows)),
                          "end": float(len(rows))
                          + (8.0 if c == 0 else 1.0)})
    emb = np.array(rows, dtype=np.float32)
    labels, k, reason, sweep_log = sweep(
        emb, turns, unreliable_silhouette=0.99)  # force weak-curve path
    assert "guardrail" in reason, f"guardrail did not engage: {reason}"
    best_combined = max(
        (e for e in sweep_log if e["combined"] is not None),
        key=lambda e: e["combined"])
    assert k >= best_combined["k_act"], (k, reason)
    assert "prefer-split" in reason, reason
    print(f"  ok weak-curve: K={k}  ({reason.split('  ')[0]})")


def test_ceiling_raise_finds_true_k():
    """12 true clusters but max_k=6: the sweep must detect climbing,
    raise the ceiling once, and reach the true K."""
    emb, turns = make_embeddings(12, 8, noise=0.15, seed=3)
    labels, k, reason, sweep_log = sweep(emb, turns, max_k=6)
    assert "ceiling_raised" in reason, reason
    assert k == 12, f"expected K=12 after raise, got {k} ({reason})"
    assert max(e["k"] for e in sweep_log) > 6, "sweep did not extend"
    print(f"  ok ceiling-raise: K={k}")


def test_hard_cap_never_raised():
    """Ceiling imposed by N//min_turns_per_speaker (not max_k) must
    not be raised — there is nothing legal beyond it."""
    emb, turns = make_embeddings(4, 3, noise=0.15, seed=5)  # N=12, cap=4
    labels, k, reason, sweep_log = sweep(
        emb, turns, min_turns_per_speaker=3, max_k=50)
    assert "ceiling_raised" not in reason, reason
    assert max(e["k"] for e in sweep_log) <= 4, "exceeded hard cap"
    print(f"  ok hard-cap: K={k}")


if __name__ == "__main__":
    test_clean_data_untouched()
    test_weak_curve_resolves_to_combined()
    test_ceiling_raise_finds_true_k()
    test_hard_cap_never_raised()
    print("all guardrail tests passed")
