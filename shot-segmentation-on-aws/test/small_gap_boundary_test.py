#!/usr/bin/env python3
"""
small_gap_boundary_test.py — 9.8 fix 1: super-threshold spikes over SMALL
non-unit frame gaps are cuts with bounded localisation error, not noise.

Run
---
  python -m unittest test.small_gap_boundary_test -v     # from repo root
  python test/small_gap_boundary_test.py

Why this exists
---------------
`_find_shot_boundaries` used to require `frame_index[i+1] - frame_index[i]
== 1` before it would even look at a distance, so a 0.44 spike over a gap of
2 was silently discarded whatever its magnitude. That rule is how Meridian
shot 16 fused three real camera shots (both cuts landed on PTS-mapping
blind spots — a duplicate frame number and a hole) and face-recognition
renamed a 207-second person downstream. The mapping defect is fixed at the
root (9.8 fix 2), but the fresh 2026-08-21 corpus still carried two live
gap-2 cuts on film_b (d=0.22 and d=0.78 — the latter an unambiguous
hard cut), so the discard rule remained a real, if rare, silent failure.

The fix accepts a boundary over gaps 0..SMALL_GAP_BOUNDARY_MAX when the
distance exceeds the threshold, placed at the right-hand sample. Three
properties are pinned:

  1. SMALL gaps confirm; the boundary lands at the right-hand row.
     gap 0 (a duplicate frame number carrying two DIFFERENT images —
     distance proves it) is the same signal in disguise and also confirms.
  2. DELIBERATE sampling gaps stay excluded: the coarse stride is
     round(fps/coarse_fps) >= 6, and a super-threshold distance across a
     coarse gap must NOT fire — that would change detection sensitivity on
     every well-formed asset, not just broken-mapping ones.
  3. The gap==1 path is untouched: sub-threshold small-gap pairs stay
     silent, and an ordinary gap-1 cut still detects exactly as before.

No AWS, no fixtures — frame indices and embeddings are synthesised.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.input_schema import SMALL_GAP_BOUNDARY_MAX          # noqa: E402
from lib.shot_pipeline import _find_shot_boundaries, build_shots  # noqa: E402

DIM = 384
TH = 0.20


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _series(frame_index, cut_at_row):
    """Embeddings: identical vector up to (exclusive) cut_at_row, an
    unrelated vector from there on — distance ~1.0 across the seam, ~0
    everywhere else."""
    a, b = _emb(1), _emb(2)
    return np.stack([a if r < cut_at_row else b
                     for r in range(len(frame_index))])


class TestSmallGapConfirms(unittest.TestCase):

    def test_gap2_super_threshold_is_a_boundary_at_right_hand_row(self):
        # frames ...102,103 | 105,106... — frame 104 is a mapping hole and
        # the cut sits inside the gap
        fi = [100, 101, 102, 103, 105, 106, 107]
        b = _find_shot_boundaries(fi, _series(fi, cut_at_row=4), TH)
        self.assertEqual([4], [row for row, _ in b],
                         "the new shot starts at the right-hand sample")

    def test_gap0_duplicate_pair_is_a_boundary(self):
        # two DIFFERENT images share frame number 103 (PTS duplicate); the
        # distance between them proves they differ, so it is a cut
        fi = [100, 101, 102, 103, 103, 104, 105]
        b = _find_shot_boundaries(fi, _series(fi, cut_at_row=4), TH)
        self.assertEqual([4], [row for row, _ in b])

    def test_gap_at_the_cap_confirms(self):
        fi = [100, 101, 102, 102 + SMALL_GAP_BOUNDARY_MAX, 106, 107]
        b = _find_shot_boundaries(fi, _series(fi, cut_at_row=3), TH)
        self.assertIn(3, [row for row, _ in b])


class TestDeliberateGapsStayExcluded(unittest.TestCase):

    def test_coarse_stride_gap_does_not_fire(self):
        # gap 6 = the smallest deliberate coarse-sampling stride; a cut
        # falling between coarse samples is the zoom-in densifier's job,
        # and firing here would change sensitivity on well-formed input
        fi = [100, 106, 112, 118, 124, 130]
        b = _find_shot_boundaries(fi, _series(fi, cut_at_row=3), TH)
        self.assertEqual([], b)

    def test_gap_just_over_the_cap_does_not_fire(self):
        fi = [100, 101, 102, 103 + SMALL_GAP_BOUNDARY_MAX, 108, 109]
        b = _find_shot_boundaries(fi, _series(fi, cut_at_row=3), TH)
        self.assertEqual([], b)


class TestShotContiguityAcrossSmallGapBoundaries(unittest.TestCase):
    """The regression the first live film_b run with fix 1 found: a
    boundary across a gap-2 pair split the shots correctly, but the unsampled
    hole frame belonged to NO shot, and the merge state's continuity
    validator (correctly) failed the whole asset with 'gap=2'. The closing
    shot must absorb the hole: end_frame = next.start_frame - 1."""

    def _build(self, fi, cut_at_row):
        emb = _series(fi, cut_at_row)
        n = len(fi)
        return build_shots(
            embeddings=emb, frame_index=fi,
            frame_type=["normal"] * n,
            source_time=[f / 24.0 for f in fi],
            shot_boundary_threshold=TH,
            frame_entropy=[5.0] * n)

    def test_hole_frame_is_absorbed_into_the_left_shot(self):
        fi = [100, 101, 102, 103, 105, 106, 107]      # 104 is a mapping hole
        shots = self._build(fi, cut_at_row=4)
        self.assertEqual(2, len(shots))
        self.assertEqual(104, shots[0]["end_frame"],
                         "the unsampled hole belongs to the LEFT shot")
        self.assertEqual(105, shots[1]["start_frame"])
        for a, b in zip(shots, shots[1:]):
            self.assertEqual(1, b["start_frame"] - a["end_frame"],
                             "shots must tile the frame space contiguously")

    def test_gap1_boundaries_are_byte_identical_to_before(self):
        fi = [100, 101, 102, 103, 104, 105]
        shots = self._build(fi, cut_at_row=3)
        self.assertEqual(2, len(shots))
        self.assertEqual(102, shots[0]["end_frame"])   # == frame_index[row]-1
        self.assertEqual(103, shots[1]["start_frame"])


class TestExistingBehaviourUntouched(unittest.TestCase):

    def test_gap2_sub_threshold_stays_silent(self):
        fi = [100, 101, 102, 104, 105, 106]
        emb = np.stack([_emb(1)] * len(fi))     # no cut anywhere
        self.assertEqual([], _find_shot_boundaries(fi, emb, TH))

    def test_ordinary_gap1_cut_still_detects(self):
        fi = [100, 101, 102, 103, 104, 105]
        b = _find_shot_boundaries(fi, _series(fi, cut_at_row=3), TH)
        self.assertEqual([3], [row for row, _ in b])


if __name__ == "__main__":
    unittest.main(verbosity=2)
