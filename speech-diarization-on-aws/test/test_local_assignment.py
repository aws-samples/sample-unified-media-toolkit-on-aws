#!/usr/bin/env python3
"""Unit tests for locality-constrained assignment with rejection
(lib/assign.py + the outlier reject path in lib/cluster_pipeline.py).
Self-contained — synthetic vectors, no AWS, no models.

    python3 test/test_local_assignment.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.assign import assign_speaker_local          # noqa: E402
from lib.cluster_pipeline import _reassign_outliers  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


A = unit([1, 0, 0, 0])
B = unit([0, 1, 0, 0])
C = unit([0, 0, 1, 0])
CENTROIDS = {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": C}
ANCHOR_T = np.array([10.0, 12.0, 500.0])          # A, B near t=11; C far
ANCHOR_N = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


def test_local_candidates_win():
    """An embedding closest to a FAR speaker must still be assigned
    among the LOCAL candidates — locality overrides raw similarity."""
    emb = unit([0.4, 0.1, 1.0, 0])   # globally closest to C (far away)
    spk = assign_speaker_local(emb, 11.0, ANCHOR_T, ANCHOR_N,
                               CENTROIDS, window_sec=30.0, margin=0.05)
    assert spk == "SPEAKER_00", spk  # best among the local {A, B}
    print("  ok locality overrides global similarity")


def test_margin_rejects_ties():
    """Equidistant between the two local candidates → UNKNOWN."""
    emb = unit([1, 1, 0, 0])
    spk = assign_speaker_local(emb, 11.0, ANCHOR_T, ANCHOR_N,
                               CENTROIDS, window_sec=30.0, margin=0.05)
    assert spk is None, spk
    print("  ok margin rejects ties")


def test_single_candidate_passes():
    """One local candidate passes trivially (the margin has no
    runner-up to test against)."""
    emb = unit([1, 1, 1, 1])         # ambiguous everywhere
    spk = assign_speaker_local(emb, 500.0, ANCHOR_T, ANCHOR_N,
                               CENTROIDS, window_sec=30.0, margin=0.05)
    assert spk == "SPEAKER_02", spk  # only C is anchored near t=500
    print("  ok single local candidate passes")


def test_empty_window_falls_back_to_global():
    """No anchors near t → all speakers are candidates; the margin
    still gates the answer."""
    clear = unit([1, 0.1, 0, 0])
    spk = assign_speaker_local(clear, 9999.0, ANCHOR_T, ANCHOR_N,
                               CENTROIDS, window_sec=30.0, margin=0.05)
    assert spk == "SPEAKER_00", spk
    tie = unit([1, 1, 0, 0])
    spk = assign_speaker_local(tie, 9999.0, ANCHOR_T, ANCHOR_N,
                               CENTROIDS, window_sec=30.0, margin=0.05)
    assert spk is None, spk
    print("  ok empty window: global fallback, margin still gates")


def test_outlier_reject_label():
    """_reassign_outliers marks unconfident outliers -1, keeps the
    confident ones, and preserves time order."""
    clean_turns = [{"start": 10.0, "end": 11.0},
                   {"start": 12.0, "end": 13.0}]
    clean_labels = np.array([0, 1], dtype=np.int32)
    outliers = [{"start": 11.5, "end": 11.9},    # tie between A and B
                {"start": 10.5, "end": 10.9}]    # clearly A
    out_m = np.stack([unit([1, 1, 0, 0]), unit([1, 0.05, 0, 0])])
    labels, turns = _reassign_outliers(
        out_m, outliers, {"SPEAKER_00": A, "SPEAKER_01": B},
        clean_labels, clean_turns, window_sec=30.0, margin=0.05)
    by_start = {t["start"]: int(l) for t, l in zip(turns, labels)}
    assert by_start[11.5] == -1, by_start      # tie → rejected
    assert by_start[10.5] == 0, by_start       # confident → SPEAKER_00
    assert [t["start"] for t in turns] == sorted(t["start"] for t in turns)
    print("  ok outlier reject path")


if __name__ == "__main__":
    test_local_candidates_win()
    test_margin_rejects_ties()
    test_single_candidate_passes()
    test_empty_window_falls_back_to_global()
    test_outlier_reject_label()
    print("all local-assignment tests passed")
