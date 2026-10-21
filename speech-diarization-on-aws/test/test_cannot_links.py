#!/usr/bin/env python3
"""Unit tests for P3 cannot-link constraints: evidence->pair mapping
(lib/constraints.py) and enforcement through the clustering pipeline
(lib/cluster_pipeline.py + _merge_small_clusters guard). Synthetic
data, no models, no AWS.

    python3 test/test_cannot_links.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.constraints import (evidence_to_cannot_links,   # noqa: E402
                             resolve_evidence)
from lib.cluster_pipeline import _select_optimal_clustering  # noqa: E402
from lib.clustering import _merge_small_clusters         # noqa: E402


def sents(spans):
    return [{"start": a, "end": b} for a, b in spans]


def test_mapping_happy_path_and_dedupe():
    ss = sents([(0, 2), (3, 5), (6, 8)])
    ev = [{"a": [0.2, 1.8], "b": [3.2, 4.8], "overlap": [2.0, 2.5]},
          {"a": [3.1, 4.6], "b": [0.1, 1.6], "overlap": [2.1, 2.6]}]
    assert evidence_to_cannot_links(resolve_evidence(ev, ss)) == [(0, 1)]
    print("  ok mapping + symmetric dedupe")


def test_mapping_coverage_gate_and_same_sentence():
    ss = sents([(0, 10)])
    # span half outside the only sentence -> unmapped; both spans in one
    # sentence -> contaminated, dropped
    ev = [{"a": [8.0, 14.0], "b": [1.0, 2.0]},
          {"a": [1.0, 2.0], "b": [4.0, 5.0]}]
    assert evidence_to_cannot_links(resolve_evidence(ev, ss)) == []
    print("  ok coverage gate + same-sentence drop")


def test_constraint_separates_identical_embeddings():
    """The core P3 property: two groups with IDENTICAL embeddings —
    acoustically inseparable by construction — are forced into
    different clusters by a single cannot-link, while the control run
    without constraints merges them."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=16)
    other = rng.normal(size=16)
    emb, turns = [], []
    for i in range(12):                    # one acoustic voice, 12 turns
        emb.append(base + rng.normal(size=16) * 0.01)
        turns.append({"start": float(i), "end": i + 0.9})
    for i in range(12):                    # a clearly different voice
        emb.append(other + rng.normal(size=16) * 0.01)
        turns.append({"start": 100.0 + i, "end": 100.9 + i})
    X = np.array(emb, dtype=np.float32)
    cfg = {"min_cluster_size": 2, "min_turns_per_speaker": 3}

    labels0, _, _, diag0 = _select_optimal_clustering(
        X, turns, cfg, 24.0)
    assert labels0[0] == labels0[6], "control: same voice split?"

    # cross-talk proves turns 0-5 vs 6-11 are two different people
    links = [(0, 6), (1, 7), (2, 8)]
    labels1, _, _, diag1 = _select_optimal_clustering(
        X, turns, cfg, 24.0, cannot_links=links)
    for a, b in links:
        assert labels1[a] != labels1[b], (a, b, labels1[:12])
    assert diag1["n_cannot_links"] == 3
    print("  ok identical embeddings forced apart; control merges")


def test_constraints_disable_single_speaker_fastpath():
    """All-identical embeddings normally trip single-speaker detection;
    a cannot-link proves two voices and must suppress it."""
    rng = np.random.default_rng(9)
    base = rng.normal(size=16)
    X = np.array([base + rng.normal(size=16) * 0.005 for _ in range(10)],
                 dtype=np.float32)
    turns = [{"start": float(i), "end": i + 0.9} for i in range(10)]
    cfg = {"min_cluster_size": 2, "min_turns_per_speaker": 3}
    _, _, _, d0 = _select_optimal_clustering(X, turns, cfg, 10.0)
    assert d0["single_speaker_detected"] is True
    labels, _, _, d1 = _select_optimal_clustering(
        X, turns, cfg, 10.0, cannot_links=[(0, 5)])
    assert d1["single_speaker_detected"] is False
    assert labels[0] != labels[5]
    print("  ok constraints suppress single-speaker fast-path")


def test_fastpath_suppressed_even_when_links_fall_to_outliers():
    """Cross-talk evidence proves >=2 voices even when every linked
    turn is removed as an outlier: suppression keys on the SUPPLIED
    links, never the post-outlier remap (review 2026-09-17)."""
    rng = np.random.default_rng(11)
    base = rng.normal(size=16)
    X = [base + rng.normal(size=16) * 0.005 for _ in range(34)]
    X.append(base + 50.0)                     # far outlier, index 34
    X = np.array(X, dtype=np.float32)
    turns = [{"start": float(i), "end": i + 0.9} for i in range(35)]
    cfg = {"min_cluster_size": 2, "min_turns_per_speaker": 3}
    labels, _, _, d = _select_optimal_clustering(
        X, turns, cfg, 35.0, cannot_links=[(0, 34)])
    assert d["n_outliers_removed"] >= 1, d
    assert d["single_speaker_detected"] is False
    print("  ok fast-path suppressed when links fall to outliers")


def test_violations_are_counted_when_user_k_cuts_below_links():
    """A num_speakers=1 override co-clusters a proven-distinct pair —
    the invariant is merge-LAST, not merge-never; the cut must report
    the violation instead of silently claiming enforcement."""
    rng = np.random.default_rng(12)
    base = rng.normal(size=16)
    X = np.array([base + rng.normal(size=16) * 0.005 for _ in range(10)],
                 dtype=np.float32)
    turns = [{"start": float(i), "end": i + 0.9} for i in range(10)]
    cfg = {"min_cluster_size": 2, "min_turns_per_speaker": 3,
           "num_speakers": 1}
    labels, _, _, d = _select_optimal_clustering(
        X, turns, cfg, 10.0, cannot_links=[(0, 5)])
    assert labels[0] == labels[5]             # the violation happened
    assert d["cannot_link_violations"] == 1   # ...and was reported
    print("  ok violations counted under a low user K")


def test_final_violation_check_sees_reassignment_and_rejection():
    """The violation count is FINAL-position: a reassigned outlier
    rejoining its cannot-partner's cluster counts; a rejected (-1)
    member does not (UNKNOWN is abstention). Order-independent via
    (start, end) keys because reassignment re-sorts turns."""
    from lib.cluster_pipeline import _cannot_link_violations
    uniq = [{"start": float(i), "end": i + 0.9} for i in range(4)]
    resorted = [uniq[2], uniq[0], uniq[3], uniq[1]]
    # pair (0, 2) share cluster 0 -> violated; pair (1, 3): 3 rejected
    labels = [0, 0, -1, 1]        # labels for resorted order
    assert _cannot_link_violations(
        [(0, 2), (1, 3)], uniq, resorted, labels) == 1
    # out-of-range index never counts
    assert _cannot_link_violations([(0, 9)], uniq, resorted, labels) == 0
    print("  ok final violation check: rejoin counts, rejection/oob do not")


def test_insufficient_data_exit_stays_honest_with_links():
    """The <2-clean-turns early exit must not claim single-speaker
    when supplied cannot-links prove >=2 voices."""
    X = np.array([[1.0] * 16], dtype=np.float32)
    turns = [{"start": 0.0, "end": 0.9}]
    labels, _, method, d = _select_optimal_clustering(
        X, turns, {"min_cluster_size": 2}, 1.0, cannot_links=[(0, 5)])
    assert method == "single(insufficient_data)"
    assert d["single_speaker_detected"] is False
    assert d["cannot_link_violations"] == 0
    print("  ok insufficient-data exit honest under links")


def test_small_cluster_merge_respects_links():
    """A small cluster's member may not be folded into a cluster
    containing its cannot-partner."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=8)
    X = np.array([a + rng.normal(size=8) * 0.01 for _ in range(6)],
                 dtype=np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = np.array([0, 0, 0, 0, 0, 1], dtype=np.int32)  # 5 is small
    merged = _merge_small_clusters(labels, X, min_cluster_size=2,
                                   cannot_links=[(5, 0)])
    assert merged[5] != merged[0], merged   # would otherwise fold into 0
    merged2 = _merge_small_clusters(labels, X, min_cluster_size=2)
    assert merged2[5] == merged2[0]         # control folds
    print("  ok small-cluster merge guard")


def test_overlap_segments_output():
    """Identity-resolved two-speaker segments: emitted only when both
    sides resolved to real distinct clusters; same-pair adjacency
    merges; UNKNOWN participants are skipped."""
    from lib.constraints import overlap_segments
    records = [
        {"i": 0, "j": 1, "overlap": [10.0, 11.0]},
        {"i": 0, "j": 1, "overlap": [11.05, 11.5]},   # adjacent, same pair
        {"i": 0, "j": 2, "overlap": [20.0, 20.6]},    # partner UNKNOWN
        {"i": 1, "j": 3, "overlap": [30.0, 30.4]},    # same cluster both
    ]
    labels = ["SPEAKER_00", "SPEAKER_01", "UNKNOWN", "SPEAKER_01"]
    segs = overlap_segments(records, labels)
    assert segs == [{"start": 10.0, "end": 11.5,
                     "speakers": ["SPEAKER_00", "SPEAKER_01"]}], segs
    print("  ok overlap_segments: resolve, merge, skip UNKNOWN/same")


if __name__ == "__main__":
    test_mapping_happy_path_and_dedupe()
    test_mapping_coverage_gate_and_same_sentence()
    test_constraint_separates_identical_embeddings()
    test_constraints_disable_single_speaker_fastpath()
    test_fastpath_suppressed_even_when_links_fall_to_outliers()
    test_violations_are_counted_when_user_k_cuts_below_links()
    test_final_violation_check_sees_reassignment_and_rejection()
    test_insufficient_data_exit_stays_honest_with_links()
    test_small_cluster_merge_respects_links()
    test_overlap_segments_output()
    print("all cannot-link tests passed")
