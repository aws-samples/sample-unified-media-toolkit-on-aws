#!/usr/bin/env python3
"""Unit tests for extract_overlap_evidence (P3 cannot-link input):
channel-resolved cross-talk from one powerset window. Synthetic
probability matrices, no model.

    python3 test/test_overlap_evidence.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.segmentation import extract_overlap_evidence  # noqa: E402

FR = 10.0        # frames per second, for round numbers


def window(rows):
    """rows = list of (n_frames, active_class); builds [F,7] probs."""
    frames = []
    for n, cls in rows:
        m = np.full((n, 7), 0.01, dtype=np.float32)
        m[:, cls] = 0.9
        frames.append(m)
    return np.concatenate(frames)


def extract(probs, thr=0.35, solo=0.4, ov=0.3):
    return extract_overlap_evidence(probs, 100.0, FR, thr, solo, ov)


def test_solo_overlap_solo_yields_record():
    """spk0 solo, spk0+1 overlap, spk1 solo -> one cannot-link record
    with the right spans and absolute times."""
    probs = window([(10, 1), (5, 4), (10, 2)])   # 1s, 0.5s, 1s
    ev = extract(probs)
    assert len(ev) == 1, ev
    r = ev[0]
    assert r["a"] == [100.0, 101.0], r
    assert r["overlap"] == [101.0, 101.5], r
    assert r["b"] == [101.5, 102.5], r
    print("  ok solo/overlap/solo yields one record")


def test_missing_solo_yields_nothing():
    """Overlap with only ONE participant's solo span: no constraint —
    we cannot say which sentence the second voice owns."""
    probs = window([(10, 1), (5, 4), (10, 0)])
    assert extract(probs) == []
    print("  ok missing second solo -> no record")


def test_short_overlap_filtered():
    probs = window([(10, 1), (2, 4), (10, 2)])   # 0.2s overlap < 0.3s
    assert extract(probs) == []
    print("  ok sub-minimum overlap filtered")


def test_pair_mapping_class6():
    """Class 6 = spk1+spk2 -> solos are classes 2 and 3."""
    probs = window([(8, 2), (4, 6), (8, 3)])
    ev = extract(probs)
    assert len(ev) == 1, ev
    assert ev[0]["a"] == [100.0, 100.8]
    assert ev[0]["b"] == [101.2, 102.0]
    print("  ok class-6 pair maps to spk1/spk2 solos")


def test_silence_only():
    assert extract(window([(30, 0)])) == []
    print("  ok silence yields nothing")


def test_wrong_class_count_yields_no_evidence():
    """An export with a non-community-1 class layout must degrade to
    no evidence, never IndexError (this call sits outside the VAD
    state's per-window try)."""
    for n_classes in (5, 8, 11):
        probs = np.zeros((40, n_classes), dtype=np.float32)
        probs[:, 1] = 1.0
        assert extract_overlap_evidence(
            probs, 0.0, FR, 0.5, 0.4, 0.3) == [], n_classes
    print("  ok wrong class count degrades to no evidence")


if __name__ == "__main__":
    test_solo_overlap_solo_yields_record()
    test_missing_solo_yields_nothing()
    test_short_overlap_filtered()
    test_pair_mapping_class6()
    test_silence_only()
    test_wrong_class_count_yields_no_evidence()
    print("all overlap-evidence tests passed")

