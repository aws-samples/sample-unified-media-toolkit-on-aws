#!/usr/bin/env python3
"""
continuity_sync_pop_test.py — the validator must never fail a run for declining
to do something the merger is designed to refuse.

Run
---
  python -m unittest test.continuity_sync_pop_test -v     # from repo root
  python test/continuity_sync_pop_test.py

Why this exists
---------------
Check 2 of `_validate_continuity` declares a single-frame shot sitting between
two near-identical neighbours a violation, on the premise that
`_merge_noise_shots` should have healed it. That premise is only true where the
merge would actually run, and the merge declines in several cases on purpose.

Every time the two lists drifted apart, a correct shot list failed validation
and took the whole asset with it:

  1st  a single frame BOUNDED by black. Black carries the sentinel centroid, so
       the neighbour distance is identically 0 and always trips the threshold,
       while every merge pass refuses to merge across black. Any source using
       black separators between segments failed in proportion to how many
       flashes landed inside those separators.
  2nd  a single frame that IS black — a one-frame black flash inside a
       continuous scene. `_merge_noise_shots` excludes a black shot from
       candidacy outright, because merging it would yield a "content" shot
       spanning black frames. Unhealable by design, and it failed a real
       19-minute asset outright ("Shot 1917: unresolved sync-pop"). It only
       surfaced once black_entropy_max was raised for a redigitised-tape
       source, which is what let a one-frame tape black flash be typed black at
       all — so the parameter did not cause the bug, it revealed it.

The fix is structural: `_noise_merge_blocked` is the single predicate, and
check 2 consults it rather than restating a subset of it. These tests pin that
property directly — each case asserts BOTH that the merge leaves the shot alone
AND that validation accepts the result. A future contributor who adds a decline
condition to the merger without telling the validator fails here rather than in
production.

No AWS calls, no S3, no fixtures — shot dicts and embeddings are synthesised.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.input_schema import SYNC_POP_THRESHOLD                   # noqa: E402
from states.merge_reconcile_shots.lambda_function import (         # noqa: E402
    _merge_noise_shots,
    _noise_merge_blocked,
    _validate_continuity,
)

DIM = 384
SENTINEL = np.full(DIM, 1.0 / math.sqrt(DIM), dtype=np.float32)
FPS = 25.0
FRAME = 1.0 / FPS


def _emb(seed, dim=DIM):
    """A deterministic unit vector. Distinct seeds give near-orthogonal vectors."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _shot(shot_id, start, end, centroid, shot_type="content", **extra):
    s = {
        "shot_id": shot_id,
        "shot_type": shot_type,
        "start_frame": start,
        "end_frame": end,
        "start_time": round(start * FRAME, 4),
        "end_time": round(end * FRAME, 4),
        "frame_count": end - start + 1,
        "start_timecode": "00:00:00:00",
        "end_timecode": "00:00:00:00",
        "centroid_embedding": np.asarray(centroid, np.float32).tolist(),
    }
    s.update(extra)
    return s


def _chain(*specs):
    """Build a gap=1, sequentially-numbered shot list from (n_frames, centroid, type) specs."""
    shots, f = [], 0
    for i, (n, centroid, stype) in enumerate(specs):
        shots.append(_shot(i, f, f + n - 1, centroid, stype))
        f += n
    return shots


def _frame_index(shots):
    return list(range(shots[0]["start_frame"], shots[-1]["end_frame"] + 1))


def _validate(shots, cfg=None):
    """Run the validator; return the ValueError message, or None when it passes."""
    try:
        _validate_continuity(shots, _frame_index(shots), FPS, cfg or {})
        return None
    except ValueError as e:
        return str(e)


def _embeddings_for(shots):
    """Frame-level embeddings + frame numbers spanning the whole chain.

    Each frame takes its own shot's centroid, so a recomputed centroid after a
    merge is a genuine mean of the merged range rather than a re-used constant.
    """
    frames, rows = [], []
    for s in shots:
        for f in range(s["start_frame"], s["end_frame"] + 1):
            frames.append(f)
            rows.append(s["centroid_embedding"])
    return np.asarray(rows, np.float32), np.asarray(frames, np.int64)


class TestUnhealableSingleFramesAreNotViolations(unittest.TestCase):
    """Each case: the merge declines, and validation must accept that."""

    def _assert_declined_and_valid(self, shots, expected_reason):
        pop = shots[1]
        self.assertEqual(pop["start_frame"], pop["end_frame"],
                         "test fixture must make shot 1 a single frame")

        reason = _noise_merge_blocked(shots[0], pop, shots[2], {})
        self.assertEqual(reason, expected_reason)

        emb, fn = _embeddings_for(shots)
        merged = _merge_noise_shots([dict(s) for s in shots], emb, fn, {})
        self.assertEqual(len(merged), len(shots),
                         "the merge must leave an unhealable shot alone")

        self.assertIsNone(_validate(shots),
                          "validation must not demand a heal the merge refuses")

    def test_single_frame_is_itself_black(self):
        """The case that failed a real asset: a one-frame black flash mid-scene."""
        scene = _emb(1)
        self._assert_declined_and_valid(
            _chain((100, scene, "content"), (1, SENTINEL, "black"),
                   (100, scene, "content")),
            "shot is black")

    def test_single_frame_is_bounded_by_black(self):
        """The first case found: a flash inside a black separator."""
        self._assert_declined_and_valid(
            _chain((50, SENTINEL, "black"), (1, _emb(2), "content"),
                   (50, SENTINEL, "black")),
            "neighbour is black")

    def test_single_frame_follows_black(self):
        """One black neighbour is enough — the sentinel makes the distance 0."""
        self._assert_declined_and_valid(
            _chain((50, SENTINEL, "black"), (1, _emb(3), "content"),
                   (50, _emb(4), "content")),
            "neighbour is black")

    def test_single_frame_is_a_resolved_dissolve(self):
        """`transition_to_next` marks a dissolve already accounted for."""
        scene = _emb(5)
        shots = _chain((60, scene, "content"), (1, _emb(6), "content"),
                       (60, scene, "content"))
        shots[1]["transition_to_next"] = {"kind": "dissolve"}
        self._assert_declined_and_valid(shots, "resolved dissolve")

    def test_merge_threshold_tighter_than_sync_pop_threshold(self):
        """A caller may set noise_distance_threshold below SYNC_POP_THRESHOLD.

        The merge then legitimately declines on distance, so the check must not
        fire. Nothing in the shipped defaults reaches this case (0.18 > 0.10),
        which is exactly why it would have gone unnoticed.
        """
        a, b = _emb(7), _emb(8)
        near = a + 0.2 * b                        # d ~= eps^2/2, so this lands near 0.02
        near /= np.linalg.norm(near)
        shots = _chain((40, a, "content"), (1, _emb(9), "content"),
                       (40, near, "content"))
        cfg = {"noise_distance_threshold": 0.005}

        d = 1.0 - float(np.asarray(shots[0]["centroid_embedding"], np.float32)
                        @ np.asarray(shots[2]["centroid_embedding"], np.float32))
        self.assertLess(d, SYNC_POP_THRESHOLD,
                        "fixture must trip check 2's similarity test")
        self.assertGreater(d, cfg["noise_distance_threshold"],
                           "fixture must fall outside the merge threshold")

        self.assertEqual(
            _noise_merge_blocked(shots[0], shots[1], shots[2], cfg, dist=d),
            "neighbours not similar enough for the merge threshold")
        self.assertIsNone(_validate(shots, cfg))


class TestRealViolationsStillFail(unittest.TestCase):
    """The guard must not become a blanket amnesty for check 2."""

    def test_healable_single_frame_between_similar_content_is_a_violation(self):
        """The case check 2 exists for: the merge WOULD have healed this."""
        scene = _emb(11)
        shots = _chain((100, scene, "content"), (1, _emb(12), "content"),
                       (100, scene, "content"))

        self.assertIsNone(_noise_merge_blocked(shots[0], shots[1], shots[2], {}),
                          "nothing should block this merge")

        emb, fn = _embeddings_for(shots)
        merged = _merge_noise_shots([dict(s) for s in shots], emb, fn, {})
        self.assertEqual(len(merged), 1, "the merge must heal it")

        msg = _validate(shots)
        self.assertIsNotNone(msg, "an unhealed healable shot IS a violation")
        self.assertIn("unresolved sync-pop", msg)
        self.assertIn("Shot 1", msg)

    def test_single_frame_between_dissimilar_content_is_not_a_violation(self):
        """A real one-frame cut between two different scenes. Never was a violation."""
        shots = _chain((100, _emb(13), "content"), (1, _emb(14), "content"),
                       (100, _emb(15), "content"))
        self.assertIsNone(_validate(shots))

    def test_other_checks_are_unaffected(self):
        """The cfg parameter added for check 2 must not disturb checks 1-6."""
        shots = _chain((10, _emb(16), "content"), (10, _emb(17), "content"))
        shots[1]["start_frame"] += 3                      # break the gap=1 invariant
        msg = _validate(shots)
        self.assertIsNotNone(msg)
        self.assertIn("[5]", msg)


class TestMergerAndValidatorCannotDrift(unittest.TestCase):
    """The property that makes the fix structural rather than another patch."""

    def test_every_decline_reason_is_accepted_by_the_validator(self):
        """Enumerate the merger's declines; each must silence check 2.

        Rather than trusting the reason strings, this drives the real merger and
        asserts the implication in both directions: declined => validation
        passes, healed => the merge actually collapsed the chain.
        """
        scene = _emb(21)
        black = ("black", SENTINEL)
        cases = [
            ("self black",  _chain((80, scene, "content"), (1, SENTINEL, black[0]),
                                   (80, scene, "content")), {}),
            ("prev black",  _chain((80, SENTINEL, "black"), (1, _emb(22), "content"),
                                   (80, _emb(23), "content")), {}),
            ("next black",  _chain((80, _emb(24), "content"), (1, _emb(25), "content"),
                                   (80, SENTINEL, "black")), {}),
            ("healable",    _chain((80, scene, "content"), (1, _emb(26), "content"),
                                   (80, scene, "content")), {}),
        ]
        for name, shots, cfg in cases:
            with self.subTest(case=name):
                emb, fn = _embeddings_for(shots)
                merged = _merge_noise_shots([dict(s) for s in shots], emb, fn, cfg)
                healed = len(merged) < len(shots)
                blocked = _noise_merge_blocked(shots[0], shots[1], shots[2], cfg)

                self.assertEqual(healed, blocked is None,
                                 f"{name}: _noise_merge_blocked disagrees with the merger")
                if blocked:
                    self.assertIsNone(
                        _validate(shots, cfg),
                        f"{name}: merge declined, so check 2 must not fire")
                else:
                    self.assertIsNotNone(
                        _validate(shots, cfg),
                        f"{name}: merge would heal it, so an unhealed shot is a violation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
