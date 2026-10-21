#!/usr/bin/env python3
"""
shot_dispersion_test.py — intra-shot dispersion: what averaging throws away.

Run
---
  python -m unittest test.shot_dispersion_test -v     # from repo root
  python test/shot_dispersion_test.py

Why this exists
---------------
A shot's `centroid_embedding` is a mean, so every property derived from it
describes the shot's AVERAGE appearance and nothing describes how uniform the
shot is. `test/harness/features.py` measured the consequence: of twenty
properties scored for their ability to tell genuine structural furniture from a
recurring set, the two best (`coverage` 0.953 and `ent_range` 0.948) are both
about variation, and `ent_range` — the only intra-shot spread we happened to
already publish — got there as a by-product of a black-detection floor. The
embedding-space spread was being computed and discarded.

`_compute_shot_dispersion` keeps it. These tests pin the three properties that
make the four fields worth publishing rather than one:

  1. The FIELDS ARE NOT REDUNDANT. Mean frame-to-centroid distance is exactly
     `1 - frame_coherence` — an algebraic identity, not a coincidence — so
     publishing both would publish one number twice. That identity is asserted
     here, because it is the reason the mean is absent from the output and a
     future contributor "adding the obvious missing field" needs to see it.
  2. COHERENCE AND MOTION ARE INDEPENDENT. A shot alternating between two looks
     scores high adjacent motion with the SAME coherence as a shot that drifts
     monotonically between them; only the pair distinguishes them. If one field
     could stand for the other, the cheap one should have been the only one.
  3. UNDEFINED IS NOT ZERO. A single-frame shot has no adjacent pair, so its
     motion is null, not 0.0. Zero reads as "measured, and perfectly still" —
     which is the mistake `entropy_mean = 0.0` already makes for a shot with no
     frames, and one worth not repeating on a new field.

No AWS calls, no S3, no fixtures — shot dicts and embeddings are synthesised.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from states.merge_reconcile_shots.lambda_function import (         # noqa: E402
    _compute_shot_dispersion,
    _recompute_shot_centroid,
    _shot_frame_slice,
)

DIM = 384
SENTINEL = np.full(DIM, 1.0 / math.sqrt(DIM), dtype=np.float32)


def _emb(seed, dim=DIM):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _mix(a, b, w):
    """Unit vector w of the way from a to b."""
    v = (1.0 - w) * a + w * b
    return (v / np.linalg.norm(v)).astype(np.float32)


def _run(frames, start=0, dim=DIM):
    """Dispersion fields for one shot whose frames are `frames` (a list of
    vectors, or an empty list)."""
    shot = {"shot_id": 0, "start_frame": start,
            "end_frame": start + max(len(frames) - 1, 0)}
    X = (np.stack(frames).astype(np.float32) if frames
         else np.zeros((0, dim), dtype=np.float32))
    fnums = np.arange(start, start + len(frames), dtype=np.int64)
    _compute_shot_dispersion([shot], fnums, X)
    return shot


class TestFieldsAreNotRedundant(unittest.TestCase):

    def test_mean_distance_to_centroid_is_one_minus_coherence(self):
        # The identity that keeps `frame_dispersion_mean` OUT of the output.
        for seeds in ([1, 2, 3], [4, 4, 4, 5], list(range(10, 22))):
            frames = [_emb(s) for s in seeds]
            shot = _run(frames)
            X = np.stack(frames)
            normed = X / np.linalg.norm(X, axis=1, keepdims=True)
            raw = normed.mean(axis=0)
            centroid = raw / np.linalg.norm(raw)
            mean_dist = float((1.0 - normed @ centroid).mean())
            self.assertAlmostEqual(mean_dist,
                                   1.0 - shot["frame_coherence"],
                                   places=4, msg=f"seeds={seeds}")

    def test_coherence_matches_the_value_the_centroid_pass_discards(self):
        # _recompute_shot_centroid computes ||mean of unit rows|| as its
        # normalisation divisor and drops it. Same number, so the two passes
        # must not be allowed to drift into two definitions.
        frames = [_emb(7), _mix(_emb(7), _emb(8), 0.3), _emb(8)]
        shot = _run(frames)
        X = np.stack(frames)
        normed = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.assertAlmostEqual(float(np.linalg.norm(normed.mean(axis=0))),
                               shot["frame_coherence"], places=4)

    def test_dispersion_max_is_the_tail_the_mean_hides(self):
        # nine identical frames plus one outlier: coherence stays high (the
        # mean barely moves) while dispersion_max reports the outlier
        base = _emb(11)
        frames = [base] * 9 + [_emb(12)]
        shot = _run(frames)
        self.assertGreater(shot["frame_coherence"], 0.9)
        self.assertGreater(shot["frame_dispersion_max"], 0.5)


class TestCoherenceAndMotionAreIndependent(unittest.TestCase):

    def test_coherence_is_order_invariant_and_motion_is_not(self):
        # The same eight frames in two orders: a monotone drift A -> B, and a
        # PERMUTATION of it that ping-pongs between the ends. Coherence is a
        # property of the set (it is a mean) so it is identical; adjacent
        # motion is a property of the sequence, so it is not. Neither field can
        # stand in for the other.
        a, b = _emb(21), _emb(22)
        steps = [_mix(a, b, i / 7.0) for i in range(8)]
        drift = steps
        flash = [steps[i] for i in (0, 7, 1, 6, 2, 5, 3, 4)]

        d, f = _run(drift), _run(flash)
        self.assertEqual(d["frame_coherence"], f["frame_coherence"])
        self.assertEqual(d["frame_dispersion_max"], f["frame_dispersion_max"])
        self.assertGreater(f["frame_motion_mean"],
                           5 * d["frame_motion_mean"])

    def test_static_shot_is_coherent_and_still(self):
        shot = _run([_emb(31)] * 12)
        self.assertAlmostEqual(shot["frame_coherence"], 1.0, places=3)
        self.assertAlmostEqual(shot["frame_motion_mean"], 0.0, places=4)
        self.assertAlmostEqual(shot["frame_dispersion_max"], 0.0, places=4)

    def test_black_shot_frames_are_all_sentinel_so_perfectly_coherent(self):
        # black frames carry the identical sentinel embedding; the fields must
        # report that honestly rather than specially
        shot = _run([SENTINEL.copy() for _ in range(5)])
        self.assertAlmostEqual(shot["frame_coherence"], 1.0, places=3)
        self.assertAlmostEqual(shot["frame_motion_max"], 0.0, places=4)

    def test_motion_max_flags_an_untaken_boundary(self):
        # a hard change halfway through one shot: the mean is diluted by the
        # still frames either side, the max is not
        frames = [_emb(41)] * 6 + [_emb(42)] * 6
        shot = _run(frames)
        self.assertLess(shot["frame_motion_mean"], 0.2)
        self.assertGreater(shot["frame_motion_max"], 0.8)


class TestUndefinedIsNotZero(unittest.TestCase):

    def test_single_frame_shot_reports_null_motion(self):
        shot = _run([_emb(51)])
        self.assertIsNone(shot["frame_motion_mean"])
        self.assertIsNone(shot["frame_motion_max"])
        # ...but coherence and dispersion ARE defined for one frame
        self.assertAlmostEqual(shot["frame_coherence"], 1.0, places=3)
        self.assertAlmostEqual(shot["frame_dispersion_max"], 0.0, places=4)

    def test_shot_with_no_embedded_frames_reports_null_throughout(self):
        shot = _run([])
        for k in ("frame_coherence", "frame_dispersion_max",
                  "frame_motion_mean", "frame_motion_max"):
            self.assertIsNone(shot[k], k)

    def test_every_shot_gets_the_fields(self):
        # the pass runs over the final list, so no shot may be left without
        # them — the fields are absent-or-null, never missing
        frames = [_emb(61), _emb(62), _emb(61), _emb(63)]
        X = np.stack(frames)
        fnums = np.array([0, 1, 2, 3], dtype=np.int64)
        shots = [{"shot_id": 0, "start_frame": 0, "end_frame": 1},
                 {"shot_id": 1, "start_frame": 2, "end_frame": 3},
                 {"shot_id": 2, "start_frame": 90, "end_frame": 99}]  # no rows
        _compute_shot_dispersion(shots, fnums, X)
        for s in shots:
            for k in ("frame_coherence", "frame_dispersion_max",
                      "frame_motion_mean", "frame_motion_max"):
                self.assertIn(k, s, f"shot {s['shot_id']} missing {k}")


class TestSharedFrameSlice(unittest.TestCase):
    """`_shot_frame_slice` is now the only definition of "which rows are this
    shot's"; the centroid, entropy and dispersion passes all consult it."""

    def test_slice_is_inclusive_of_end_frame(self):
        fnums = np.array([0, 4, 8, 12, 16], dtype=np.int64)
        self.assertEqual((0, 3),
                         _shot_frame_slice({"start_frame": 0,
                                            "end_frame": 8}, fnums))

    def test_slice_excludes_frames_outside_the_shot(self):
        fnums = np.array([0, 4, 8, 12, 16], dtype=np.int64)
        self.assertEqual((2, 4),
                         _shot_frame_slice({"start_frame": 5,
                                            "end_frame": 13}, fnums))

    def test_centroid_and_dispersion_address_the_same_rows(self):
        frames = [_emb(71), _emb(72), _emb(73), _emb(74)]
        X = np.stack(frames)
        fnums = np.array([10, 20, 30, 40], dtype=np.int64)
        shot = {"shot_id": 0, "start_frame": 15, "end_frame": 35}
        _recompute_shot_centroid(shot, X, fnums)
        _compute_shot_dispersion([shot], fnums, X)
        # frame_count comes from the centroid pass, coherence from the
        # dispersion pass — they must have seen the same two rows
        self.assertEqual(2, shot["frame_count"])
        expected = np.stack([frames[1], frames[2]]).mean(axis=0)
        self.assertAlmostEqual(float(np.linalg.norm(expected)),
                               shot["frame_coherence"], places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
