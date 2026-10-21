#!/usr/bin/env python3
"""
recurrence_gates_test.py — Phase 9 recurrence gates (README.md 9.1-9.4, 9.6).

Run
---
  python test/recurrence_gates_test.py                        # from repo root
  python -m unittest discover -s test -p 'recurrence*' -v     # from repo root
(test/ is not a package — `-m unittest test.recurrence_gates_test` will not
resolve.)

Each gate kills a measured false-group shape from the main pipeline's
nine-asset live trial:
  9.1 dialogue coverage (A/B/A/B inside one scene: 32 of 33 groups on
      drama_e) — members without min_occurrence_gap_sec separation are ONE
      occurrence, and min_recurrence_count applies to occurrences;
  9.2 black shots share the identical sentinel centroid (distance 0.0 ->
      one giant fake group) and static bars/slates recur trivially;
  9.3 recurring SETS (anchor cameras n=49-69, confessional chairs n=11-25)
      pass every similarity gate because they ARE near-identical — the
      member-count ceiling removed ~90% of false groups in the trial;
  9.4 timecode-burn masters: ~99% black + burned glyphs reads HIGH entropy
      and hashes identically, so black DOMINANCE (not entropy) is the test;
  9.6 embeddings confirm same-SET, a hash family confirms same-RENDERED-
      IMAGE — the confirmation statistic is global-OR-median, never
      max-of-patches (measured on the real drama_e card, see input_schema).
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib.recurrence import (complete_linkage,  # noqa: E402
                            count_occurrences, detect_recurrence,
                            occurrence_labels)

_DIM = 384


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


_SENTINEL = (np.full(_DIM, 1.0 / np.sqrt(_DIM), dtype=np.float32)).tolist()


def _shot(sid, start, dur=2.0, emb=None, kind="content"):
    return {
        "shot_id": sid, "shot_type": kind,
        "start_frame": int(start * 24), "end_frame": int((start + dur) * 24),
        "start_time": float(start), "end_time": float(start + dur),
        "centroid_embedding": emb if emb is not None else _emb(sid),
    }


def _run(shots, duration=3600.0, **cfg):
    return detect_recurrence(shots=shots, cfg=cfg,
                             actual_duration_sec=duration)


class TestTemporalGate(unittest.TestCase):  # 9.1

    def test_gap_is_end_to_start_not_start_to_start(self):
        # a LONG shot must not promote its own coverage into a second
        # occurrence just by being long: shot ends at 650, next starts at
        # 660 — 10s end-to-start gap = ONE occurrence, even though the
        # starts are 60s apart (matches shotRecurrence.js semantics)
        card = _emb(1000)
        shots = ([_shot(0, 600, dur=50.0, emb=card),
                  _shot(1, 660, emb=card),
                  _shot(2, 665, emb=card)]
                 + [_shot(10 + i, 100 + i * 40) for i in range(4)])
        self.assertEqual([], _run(shots),
                         "3 members, 1 occurrence — must be dropped")

    def test_dialogue_coverage_is_one_occurrence_and_dropped(self):
        card = _emb(1000)
        shots = ([_shot(i, 600 + i * 6, emb=card) for i in range(6)]  # A/B/A/B
                 + [_shot(10 + i, 1000 + i * 300) for i in range(4)])  # filler
        groups = _run(shots)
        self.assertEqual([], groups,
                         "6 members 6s apart are ONE occurrence, not 6")

    def test_true_structural_card_survives_with_occurrence_count(self):
        card = _emb(1000)
        shots = ([_shot(i, 600 * (i + 1), emb=card) for i in range(4)]
                 + [_shot(10 + i, 100 + i * 40) for i in range(4)])
        groups = _run(shots)
        self.assertEqual(1, len(groups))
        self.assertEqual(4, groups[0]["recurrence_count"])
        self.assertEqual(4, groups[0]["occurrence_count"])

    def test_mixed_coverage_counts_occurrences_not_members(self):
        # 2 members at minute 10 + 2 members at minute 20: 4 raw members but
        # only 2 occurrences -> below min_count 3 -> dropped
        card = _emb(1000)
        shots = ([_shot(0, 600, emb=card), _shot(1, 610, emb=card),
                  _shot(2, 1200, emb=card), _shot(3, 1210, emb=card)]
                 + [_shot(10 + i, 100 + i * 40) for i in range(4)])
        self.assertEqual([], _run(shots))


class TestOccurrenceLabels(unittest.TestCase):  # 9.1, at the function level
    """`occurrence_labels` is public because the offline replays need the
    MEMBERS of each occurrence (to measure the gaps between them) while the
    pipeline needs only the count. Two views of one rule, so the two must be
    pinned to agree — three separate copies of this rule previously lived in
    the harness and two of them had drifted."""

    def test_count_agrees_with_labels(self):
        for spans in ([],
                      [(0.0, 1.0)],
                      [(0.0, 1.0), (100.0, 101.0)],
                      [(0.0, 1.0), (5.0, 6.0), (500.0, 501.0)]):
            self.assertEqual(
                count_occurrences(spans, 60.0),
                (occurrence_labels(spans, 60.0)[-1] + 1) if spans else 0,
                spans)

    def test_labels_are_contiguous_and_non_decreasing(self):
        spans = [(0.0, 1.0), (5.0, 6.0), (500.0, 501.0), (505.0, 506.0),
                 (5000.0, 5001.0)]
        labels = occurrence_labels(spans, 60.0)
        self.assertEqual([0, 0, 1, 1, 2], labels)

    def test_running_end_is_the_maximum_not_the_previous_span(self):
        # a long span followed by a short one nested inside it: the third span
        # is 20s after the SHORT span's end but only 5s after the LONG one's,
        # so the max-end rule keeps all three in one occurrence
        spans = [(0.0, 100.0), (10.0, 20.0), (105.0, 106.0)]
        self.assertEqual([0, 0, 0], occurrence_labels(spans, 60.0))
        self.assertEqual(1, count_occurrences(spans, 60.0))

    def test_gap_boundary_is_inclusive(self):
        # >= gap_sec starts a new occurrence, exactly at the threshold
        self.assertEqual(2, count_occurrences([(0.0, 1.0), (61.0, 62.0)], 60.0))
        self.assertEqual(1, count_occurrences([(0.0, 1.0), (60.9, 62.0)], 60.0))


class TestDiameterConstraint(unittest.TestCase):
    """Grouping is complete linkage: max_recurrence_threshold bounds group
    DIAMETER, so every published max_distance <= the threshold. Single linkage
    (shipped until 2026-08-21) bounded only the nearest-neighbour distance and
    chained near-neighbours transitively into mixed bags — measured diameters
    reached 2.6x the threshold on published output (README.md 9.11)."""

    @staticmethod
    def _vec(theta):
        """Unit vector in a 2-plane: cosine distance to _vec(0) is 1-cos(theta),
        so pairwise distances are exact by construction."""
        v = np.zeros(_DIM, dtype=np.float32)
        v[0], v[1] = np.cos(theta), np.sin(theta)
        return v.tolist()

    def test_chain_does_not_bridge(self):
        # Three recurring elements on a similarity chain: d(A,B)=d(B,C)=0.05,
        # d(A,C)=0.195. Single linkage returns ONE group of all nine shots with
        # max_distance 0.195 — 2.8x the threshold. Complete linkage may group
        # A with B (every pair is 0.05 <= 0.07, a legitimate diameter) but must
        # never put A and C in the same group.
        th = math.acos(0.95)   # 1 - cos(th) = 0.05
        a, b, c = self._vec(0.0), self._vec(th), self._vec(2 * th)
        shots = ([_shot(i, 600 * (i + 1), emb=a) for i in range(3)]
                 + [_shot(10 + i, 2400 + 600 * i, emb=b) for i in range(3)]
                 + [_shot(20 + i, 4200 + 600 * i, emb=c) for i in range(3)]
                 + [_shot(90 + i, 100 + i * 40) for i in range(4)])
        groups = _run(shots, duration=7200.0)
        a_ids, c_ids = {0, 1, 2}, {20, 21, 22}
        for g in groups:
            ids = {s["shot_id"] for s in g["shots"]}
            self.assertFalse(ids & a_ids and ids & c_ids,
                             f"chain bridged: {sorted(ids)}")
        self.assertTrue(any({s["shot_id"] for s in g["shots"]} >= c_ids
                            for g in groups),
                        "C's own clique must still publish as a group")

    def test_published_max_distance_never_exceeds_threshold(self):
        # same chain — the contract-level statement of the invariant, which the
        # shipped output used to violate (threshold 0.05 and max_distance
        # 0.1315 in one document)
        th = math.acos(0.95)
        shots = [_shot(k * 10 + i, 600 * (3 * i + k + 1),
                       emb=self._vec(k * th))
                 for k in range(3) for i in range(3)]
        shots += [_shot(90 + i, 100 + i * 40) for i in range(4)]
        for g in _run(shots, duration=7200.0):
            self.assertLessEqual(
                g["max_distance"],
                0.07,
                f"group {g['group_id']} diameter exceeds the threshold")

    def test_linkage_is_canonical_not_greedy(self):
        # Greedy first-admissible-pair merges (0,1) at 0.06 and strands 2;
        # canonical merges (1,2) at 0.02 first, after which {1,2}+{0} has
        # complete distance 0.2 and is inadmissible. The two disagree, and
        # greedy is order-dependent — on the corpus it reported no viable
        # threshold where a clean window exists (test/harness/linkage.py).
        D = np.array([[0.00, 0.06, 0.20],
                      [0.06, 0.00, 0.02],
                      [0.20, 0.02, 0.00]])
        clusters = {tuple(sorted(c)) for c in complete_linkage(D, 0.07)}
        self.assertEqual({(0,), (1, 2)}, clusters)

    def test_every_cluster_diameter_within_threshold(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(40, 8)).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        D = 1.0 - X @ X.T
        np.fill_diagonal(D, 0.0)
        for th in (0.05, 0.2, 0.5):
            for c in complete_linkage(D, th):
                if len(c) > 1:
                    self.assertLessEqual(
                        float(D[np.ix_(c, c)].max()), th + 1e-12)


class TestContentOnly(unittest.TestCase):  # 9.2

    def test_black_sentinels_form_no_group(self):
        shots = ([_shot(i, 600 * (i + 1), emb=list(_SENTINEL), kind="black")
                  for i in range(5)]
                 + [_shot(10 + i, 100 + i * 40) for i in range(4)])
        self.assertEqual([], _run(shots))

    def test_static_shots_form_no_group(self):
        bars = _emb(2000)
        shots = ([_shot(i, 600 * (i + 1), emb=bars, kind="static")
                  for i in range(4)]
                 + [_shot(10 + i, 100 + i * 40) for i in range(4)])
        self.assertEqual([], _run(shots))

    def test_black_neighbours_still_classify_act_boundary(self):
        # the card is flanked by black shots — 9.2 must not blind the
        # position classifier by removing blacks from the sorted list
        card = _emb(1000)
        shots = []
        for k, start in enumerate((600, 1200, 1800)):
            shots.append(_shot(100 + k, start - 3, emb=list(_SENTINEL),
                               kind="black"))
            shots.append(_shot(k, start, emb=card))
        shots += [_shot(10 + i, 100 + i * 40) for i in range(4)]
        groups = _run(shots)
        self.assertEqual(1, len(groups))
        self.assertEqual({"act_boundary"},
                         {s["position"] for s in groups[0]["shots"]})


class TestSetRecurrenceCeiling(unittest.TestCase):  # 9.3, annotate-not-drop
    """The ceiling no longer deletes anything (2026-08-21): member count is
    the weakest of the measured furniture/set discriminators (AUC 0.799, 9th
    of 20) and a ceiling of 8 destroyed 6 of the corpus's 12 verified cards.
    Oversized groups are published; the consumer derives "exceeds the set
    ceiling" from recurrence_count > the envelope's max_recurrence_count —
    deliberately no flag, because both numbers are already in the output."""

    def test_anchor_camera_shape_is_published_not_dropped(self):
        anchor = _emb(3000)
        shots = ([_shot(i, 120 * (i + 1), emb=anchor) for i in range(9)]
                 + [_shot(20 + i, 50 + i * 15) for i in range(4)])
        groups = _run(shots)
        self.assertEqual(1, len(groups),
                         "9 members exceed the reference ceiling of 8 and "
                         "must still be PUBLISHED")
        self.assertEqual(9, groups[0]["recurrence_count"])

    def test_eight_members_survive(self):
        card = _emb(3000)
        shots = ([_shot(i, 300 * (i + 1), emb=card) for i in range(8)]
                 + [_shot(20 + i, 50 + i * 15) for i in range(4)])
        groups = _run(shots)
        self.assertEqual(1, len(groups))
        self.assertEqual(8, groups[0]["recurrence_count"])

    def test_max_recurrence_count_no_longer_filters(self):
        # The knob survives only as the published reference value in the
        # recurrence.json envelope; passing it must not change the output.
        card = _emb(3000)
        shots = ([_shot(i, 300 * (i + 1), emb=card) for i in range(6)]
                 + [_shot(20 + i, 50 + i * 15) for i in range(4)])
        self.assertEqual(_run(shots),
                         _run(shots, max_recurrence_count=5))


class TestGateAttribution(unittest.TestCase):  # 9.10 observability
    """The per-gate counters are the only way a consumer learns WHICH gate
    dropped a candidate — lib log.info does not escape the Lambda, so before
    these existed the answer needed an offline replay against the run's own
    centroids. Only min-members and 9.1 drop any more (9.3 annotates, 9.6
    flags), so the counters cover exactly the two deleting gates and
    gates.groups == total published groups."""

    def test_each_gate_attributes_its_own_drop(self):
        card = _emb(1000)          # 4 separated occurrences -> survives
        pair = _emb(1100)          # 2 members -> below the member floor
        talk = _emb(1200)          # 5 members, all within one minute -> 9.1
        desk = _emb(1300)          # 9 separated members -> published (no 9.3)
        shots = (
            [_shot(i, 600 * (i + 1), emb=card) for i in range(4)]
            + [_shot(20 + i, 200 + i * 400, emb=pair) for i in range(2)]
            + [_shot(30 + i, 3000 + i * 6, emb=talk) for i in range(5)]
            + [_shot(40 + i, 120 * (i + 1) + 0.5, emb=desk) for i in range(9)]
            + [_shot(60 + i, 90 + i * 37, emb=list(_SENTINEL), kind="black")
               for i in range(3)]
        )
        stats = {}
        groups = detect_recurrence(shots=shots, cfg={},
                                   actual_duration_sec=7200.0, stats=stats)
        self.assertEqual(2, len(groups),
                         "the card AND the oversized desk group publish")
        self.assertEqual({4, 9},
                         {g["recurrence_count"] for g in groups})
        self.assertEqual(len(shots) - 3, stats["content_shots"])
        self.assertEqual(3, stats["excluded_non_content"])
        self.assertEqual(4, stats["components"])
        self.assertEqual(1, stats["min_members_dropped"])
        self.assertEqual(1, stats["occurrence_gate_dropped"])
        self.assertEqual(2, stats["groups"])
        self.assertNotIn("set_ceiling_dropped", stats,
                         "no ceiling counters: 9.3 never drops, and the "
                         "exceed count is derivable from the output")

    def test_stats_is_optional_and_default_callers_are_unaffected(self):
        card = _emb(1800)
        shots = ([_shot(i, 600 * (i + 1), emb=card) for i in range(4)]
                 + [_shot(20 + i, 100 + i * 40) for i in range(4)])
        stats = {}
        self.assertEqual(
            detect_recurrence(shots=shots, cfg={}, actual_duration_sec=3600.0),
            detect_recurrence(shots=shots, cfg={}, actual_duration_sec=3600.0,
                              stats=stats))
        self.assertEqual(1, stats["groups"])


class TestHashConfirmation(unittest.TestCase):  # 9.6 + 9.4
    """Embeddings confirm same-SET; the hash gate confirms same-RENDERED-
    IMAGE. Report-not-drop (README.md 9.9 option 3): the gate decides but never
    deletes — hash_confirmed is true (same render recurs), false (checked,
    same-set-not-same-render), or null (9.4 burn-in/glitch: nothing hashable,
    so UNVERIFIABLE — which must not read as verified-and-failed). Group ids
    are stable through the gate."""

    def _card(self, seed, ticker_val=None):
        rng = np.random.default_rng(seed)
        luma = (rng.integers(0, 255, (90, 90))).astype(np.uint8)
        if ticker_val is not None:  # churning bottom-right patch (index 8)
            luma[60:, 60:] = ticker_val
        return luma

    def _group(self, shot_ids):
        return {"group_id": 0, "recurrence_count": len(shot_ids),
                "occurrence_count": len(shot_ids),
                "shots": [{"shot_id": s} for s in shot_ids]}

    def _confirm(self, groups, lumas_by_shot, entropy=None):
        from lib.recurrence import confirm_groups_by_hash
        return confirm_groups_by_hash(
            groups, lambda sid: lumas_by_shot.get(sid, []), entropy)

    def test_same_render_confirms(self):
        card = self._card(1)
        lumas = {1: [(10, card)], 2: [(20, card.copy())]}
        (g,) = self._confirm([self._group([1, 2])], lumas)
        self.assertIs(True, g["hash_confirmed"])

    def test_same_set_different_render_flags_false_but_publishes(self):
        lumas = {1: [(10, self._card(1))], 2: [(20, self._card(2))]}
        (g,) = self._confirm([self._group([1, 2])], lumas)
        self.assertIs(False, g["hash_confirmed"])

    def test_ticker_discount_confirms_burned_card(self):
        # identical card, but a timecode window churns WITHIN each member —
        # classified noisy from the churn series and discounted, so the
        # cross-member comparison sees only the stable card art
        lumas = {
            1: [(i, self._card(1, ticker_val=(i * 37) % 251))
                for i in range(8)],
            2: [(100 + i, self._card(1, ticker_val=(i * 53) % 251))
                for i in range(8)],
        }
        (g,) = self._confirm([self._group([1, 2])], lumas)
        self.assertIs(True, g["hash_confirmed"],
                      "churning ticker must be discounted, not fatal")

    def test_untextured_group_is_unverifiable_not_false(self):  # 9.4
        black = np.zeros((90, 90), dtype=np.uint8)
        lumas = {1: [(10, black)], 2: [(20, black.copy())]}
        entropy = {10: 0.1, 20: 0.1}   # below RECURRENCE_HASH_MIN_ENTROPY
        (g,) = self._confirm([self._group([1, 2])], lumas, entropy)
        self.assertIsNone(g["hash_confirmed"],
                          "nothing hashable = could not check, which is "
                          "null, never false")

    def test_timecode_burn_black_is_unverifiable_despite_high_entropy(self):  # 9.4(a)
        # The measured defect: ~99% black + burned-in timecode glyphs. The
        # glyphs ARE texture, so entropy reads well ABOVE the floor and the
        # entropy test alone cannot see this frame — yet every frame in the
        # burn window is byte-identical, so the hash would confirm the group.
        # Only the black-DOMINANCE test drops it.
        burn = np.zeros((90, 90), dtype=np.uint8)
        burn[84:88, 2:26] = 220                    # ~1.2% of the frame
        lumas = {1: [(10, burn)], 2: [(20, burn.copy())]}
        entropy = {10: 3.4, 20: 3.4}               # above the floor
        (g,) = self._confirm([self._group([1, 2])], lumas, entropy)
        self.assertIsNone(g["hash_confirmed"],
                          "a black-dominated frame must be unhashable even "
                          "when its burned-in text lifts entropy")

    def test_dark_but_not_black_dominated_card_still_confirms(self):  # 9.4(a)
        # Guards the other direction: the floor must not eat genuinely dark
        # card art. 80% black is under RECURRENCE_MAX_BLACK_FRACTION, so the
        # frame stays hashable and the real re-render confirms.
        rng = np.random.default_rng(7)
        card = np.zeros((90, 90), dtype=np.uint8)
        card[:18, :] = rng.integers(40, 255, (18, 90)).astype(np.uint8)
        self.assertLess(float((card <= 10).mean()), 0.9)
        lumas = {1: [(10, card)], 2: [(20, card.copy())]}
        (g,) = self._confirm([self._group([1, 2])], lumas)
        self.assertIs(True, g["hash_confirmed"])

    def test_animated_card_confirms_across_phase_windows(self):
        # README.md 9.9 option 2. An ANIMATED card whose members are sampled at
        # different phases: member 1 archives the full 16-phase cycle, member
        # 2 only the tail half (phases 8-15). First-N sampling anchored
        # member 1's hashes to phases 0-7 — zero overlap with member 2, so
        # the one matching pair the gate needs could never be found and the
        # news_b "Tuesday" pan was rejected exactly this way. Even sampling
        # across the span reaches the shared phases at the same frame budget.
        phase = {p: self._card(9000 + p) for p in range(16)}
        lumas = {1: [(i, phase[i]) for i in range(16)],
                 2: [(100 + i, phase[8 + i]) for i in range(8)]}
        (g,) = self._confirm([self._group([1, 2])], lumas)
        self.assertIs(True, g["hash_confirmed"],
                      "members sharing only late animation phases must "
                      "still confirm")

    def test_group_ids_are_stable_through_the_gate(self):
        # Under drop semantics an unconfirmed group forced a reindex and a
        # card's id changed between detection and publication. Nothing is
        # removed any more, so ids must survive untouched.
        card = self._card(1)
        lumas = {1: [(10, card)], 2: [(20, card.copy())],
                 3: [(30, self._card(3))], 4: [(40, self._card(4))]}
        g0 = self._group([3, 4]); g0["group_id"] = 0   # flags false
        g1 = self._group([1, 2]); g1["group_id"] = 1   # flags true
        out = self._confirm([g0, g1], lumas)
        self.assertEqual([0, 1], [g["group_id"] for g in out])
        self.assertEqual([False, True],
                         [g["hash_confirmed"] for g in out])


if __name__ == "__main__":
    unittest.main()
