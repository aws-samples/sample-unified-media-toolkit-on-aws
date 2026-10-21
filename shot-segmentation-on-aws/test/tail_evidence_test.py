#!/usr/bin/env python3
"""
tail_evidence_test.py — programme-tail evidence: what sits after the last
long black, and does it re-render the programme?

Run
---
  python -m unittest test.tail_evidence_test -v     # from repo root
  python test/tail_evidence_test.py

Why this exists
---------------
Deliveries carry material after the programme proper, and two different kinds
demand opposite handling: a TEXTLESS block (clean copies of body shots,
appended for localisation — the programme effectively ends at the separator)
and a FURTHER PART (the programme continues after an inter-part black).
`analyse_programme_tail` publishes the separator plus tail-to-body match
stats; the corpus measurement behind the thresholds is recorded in
lib/tail_evidence.py (four firing assets: two textless, visually verified,
matched 0.88-0.96 and monotone 0.98; two part boundaries, matched 0.00-0.01).

Pinned properties:
  1. the separator must be LONG, LATE and FOLLOWED BY CONTENT — a break
     black in the first half, a short black, or a distributor end board
     (measured 12-21 s, an order of magnitude under the 60 s floor) must
     not produce evidence;
  2. a textless-shaped tail reads matched AND ordered; a new-content tail
     reads unmatched — and matching is by centroid distance, so the stats
     are in the same space as recurrence;
  3. multi-part masters have SEVERAL qualifying blacks: the separator is
     the one with the most content after it (the outermost), not the first;
  4. no verdict field exists — the numbers are the contract.

No AWS, no fixtures — shots and centroids are synthesised.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.tail_evidence import analyse_programme_tail          # noqa: E402

DIM = 384


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


_SENTINEL = np.full(DIM, 1.0 / np.sqrt(DIM), dtype=np.float32).tolist()


def _shot(sid, start, end, kind="content", emb=None):
    return {"shot_id": sid, "shot_type": kind,
            "start_time": float(start), "end_time": float(end),
            "centroid_embedding": (list(_SENTINEL) if kind == "black"
                                   else (emb or _emb(sid)))}


def _programme(n=20, shot_sec=30.0):
    """n content shots of shot_sec each, distinct embeddings."""
    return [_shot(i, i * shot_sec, (i + 1) * shot_sec) for i in range(n)]


class TestSeparatorQualification(unittest.TestCase):

    def test_no_long_black_means_no_evidence(self):
        self.assertIsNone(analyse_programme_tail(_programme()))

    def test_black_in_first_half_does_not_qualify(self):
        shots = _programme(20)
        shots.insert(5, _shot(100, 150.0, 190.0, kind="black"))
        self.assertIsNone(analyse_programme_tail(shots))

    def test_short_black_does_not_qualify(self):
        shots = _programme(20)                       # ends at 600
        shots.append(_shot(100, 600.0, 620.0, kind="black"))   # 20s < 30s
        shots.append(_shot(101, 620.0, 700.0))
        self.assertIsNone(analyse_programme_tail(shots))

    def test_end_board_tail_emits_honest_low_signal_evidence(self):
        # distributor end boards measured 12-21 s after the runout black.
        # The 60 s content floor that used to suppress these is REMOVED
        # (2026-08-23: it also suppressed a real 55.6 s textless block) —
        # an end board now reads honestly in the numbers instead: a
        # seconds-long tail with zero body match.
        shots = _programme(20)
        shots.append(_shot(100, 600.0, 650.0, kind="black"))
        shots.append(_shot(101, 650.0, 668.0))       # 18 s end board
        ev = analyse_programme_tail(shots)
        self.assertIsNotNone(ev)
        self.assertEqual(18.0, ev["tail_content_sec"])
        self.assertEqual(0.0, ev["match_fraction"])
        self.assertEqual(0.0, ev["match_fraction_duration_corroborated"])

    def test_fragmented_separator_coalesces_into_one_run(self):
        # The drama_c shape (ground-truthed 2026-08-23): one physical
        # 33 s black split by segmentation into FOUR consecutive black
        # shots, none reaching 30 s alone. The RUN is the separator, its
        # extent is published, and the tail begins at the run's end.
        body = _programme(20)                        # ends at 600
        shots = body + [
            _shot(100, 600.00, 600.92, kind="black"),
            _shot(101, 600.92, 610.84, kind="black"),
            _shot(102, 610.84, 613.12, kind="black"),
            _shot(103, 613.12, 633.08, kind="black"),   # longest piece: 20 s
        ]
        t = 633.08
        for k, src in enumerate(range(5, 15)):       # textless block
            shots.append(_shot(200 + k, t, t + 30.0,
                               emb=list(body[src]["centroid_embedding"])))
            t += 30.0
        ev = analyse_programme_tail(shots)
        self.assertIsNotNone(ev, "a fragmented >=30s black run must qualify")
        self.assertEqual(100, ev["separator_shot_id"])
        self.assertEqual(600.0, ev["separator_start_time"])
        self.assertEqual(633.08, ev["separator_end_time"])
        self.assertAlmostEqual(33.08, ev["separator_black_sec"], places=3)
        self.assertEqual(1.0, ev["match_fraction"])
        self.assertEqual(1.0, ev["match_duration_fraction"])

    def test_short_textless_tail_qualifies(self):
        # The factual_d shape: a clean 53.8 s single black followed by
        # only 55.6 s of textless content — under the old 60 s floor, so
        # the real block produced NO evidence.
        body = _programme(20)
        shots = body + [_shot(100, 600.0, 653.8, kind="black")]
        shots.append(_shot(200, 653.8, 667.4,
                           emb=list(body[2]["centroid_embedding"])))
        shots.append(_shot(201, 667.4, 683.4,
                           emb=list(body[7]["centroid_embedding"])))
        shots.append(_shot(202, 683.4, 709.4,
                           emb=list(body[19]["centroid_embedding"])))
        ev = analyse_programme_tail(shots)
        self.assertIsNotNone(ev)
        self.assertAlmostEqual(55.6, ev["tail_content_sec"], places=3)
        self.assertEqual(1.0, ev["match_fraction"])


class TestTailKinds(unittest.TestCase):

    def test_textless_tail_reads_matched_and_ordered(self):
        body = _programme(20)                        # ends at 600
        shots = body + [_shot(100, 600.0, 660.0, kind="black")]
        # clean copies of body shots 5..14, IN ORDER (textless block)
        t = 660.0
        for k, src in enumerate(range(5, 15)):
            shots.append(_shot(200 + k, t, t + 30.0,
                               emb=list(body[src]["centroid_embedding"])))
            t += 30.0
        ev = analyse_programme_tail(shots)
        self.assertIsNotNone(ev)
        self.assertEqual(100, ev["separator_shot_id"])
        self.assertEqual(600.0, ev["separator_start_time"])
        self.assertEqual(300.0, ev["tail_content_sec"])
        self.assertEqual(1.0, ev["match_fraction"])
        self.assertEqual(1.0, ev["match_duration_fraction"])
        self.assertEqual(1.0, ev["match_monotone"])

    def test_same_look_different_cut_fails_the_duration_test(self):
        # A tail shot that LOOKS like a body shot (same centroid — a
        # same-setup coincidence, e.g. a recurring interview framing) but is
        # a DIFFERENT CUT: a textless element is the same cut as its source,
        # so duration must agree, and this pair misses by far more than the
        # tolerance. The corpus case: title_card_a's one accidental embedding
        # match missed duration by 2.68 s.
        body = _programme(20)
        shots = body + [_shot(100, 600.0, 660.0, kind="black")]
        t = 660.0
        for k, src in enumerate(range(5, 9)):        # 30 s sources...
            shots.append(_shot(200 + k, t, t + 8.0,  # ...8 s tail cuts
                               emb=list(body[src]["centroid_embedding"])))
            t += 8.0
        shots.append(_shot(300, t, t + 40.0, emb=_emb(9500)))
        ev = analyse_programme_tail(shots)
        self.assertEqual(0.8, ev["match_fraction"],
                         "the look-alikes still MATCH by distance")
        self.assertEqual(0.0, ev["match_duration_fraction"],
                         "but none of them is the same cut")

    def test_further_part_reads_unmatched(self):
        shots = _programme(20) + [_shot(100, 600.0, 660.0, kind="black")]
        t = 660.0
        for k in range(10):                          # NEW content, part 2
            shots.append(_shot(300 + k, t, t + 30.0, emb=_emb(9000 + k)))
            t += 30.0
        ev = analyse_programme_tail(shots)
        self.assertIsNotNone(ev)
        self.assertEqual(0.0, ev["match_fraction"])
        self.assertIsNone(ev["match_duration_fraction"],
                          "duration agreement over ZERO matches is "
                          "undefined — null, never a number")
        self.assertIsNone(ev["match_monotone"],
                          "monotone is undefined with <2 matches — null, "
                          "never a number that reads as measured")

    def test_no_verdict_field(self):
        shots = _programme(20) + [_shot(100, 600.0, 660.0, kind="black")]
        t = 660.0
        for k in range(10):
            shots.append(_shot(300 + k, t, t + 30.0))
            t += 30.0
        ev = analyse_programme_tail(shots)
        self.assertEqual(
            {"separator_shot_id", "separator_start_time",
             "separator_end_time", "separator_black_sec",
             "tail_content_sec", "match_fraction",
             "match_duration_fraction", "match_monotone",
             "match_fraction_duration_corroborated"},
            set(ev))


class TestCorroboratedBand(unittest.TestCase):
    """Heavy burned graphics (a credit roll, a full-frame title) push a clean
    copy's distance past the strict 0.05 where subtitles do not — measured on
    the factual_d tail: the credit-roll background matched its clean copy
    at 0.058 with duration agreeing to 0.12 s. The wider band admits a match
    ONLY with the duration agreement; distance alone never widens it."""

    @staticmethod
    def _emb_at_distance(source, distance, seed=7):
        """A unit vector at the given cosine DISTANCE from `source`."""
        u = np.array(source, dtype=np.float64)
        u /= np.linalg.norm(u)
        rng = np.random.default_rng(seed)
        w = rng.normal(size=u.shape)
        w -= w.dot(u) * u
        w /= np.linalg.norm(w)
        cos = 1.0 - distance
        v = cos * u + np.sqrt(max(0.0, 1.0 - cos * cos)) * w
        return (v / np.linalg.norm(v)).tolist()

    def _tail_ev(self, tail_specs):
        """body + 60s black + tail shots built from (distance, duration)."""
        body = _programme(20)                        # 30 s shots, end 600
        shots = body + [_shot(100, 600.0, 660.0, kind="black")]
        t = 660.0
        for k, (dist, dur) in enumerate(tail_specs):
            emb = self._emb_at_distance(
                body[5 + k]["centroid_embedding"], dist, seed=40 + k)
            shots.append(_shot(200 + k, t, t + dur, emb=emb))
            t += dur
        return analyse_programme_tail(shots)

    def test_near_miss_with_frame_exact_duration_is_corroborated(self):
        # 0.058 from its source, duration agreeing (30 s source, 30.1 s
        # copy) — outside the strict threshold, inside the corroborated band.
        ev = self._tail_ev([(0.058, 30.1)])
        self.assertEqual(0.0, ev["match_fraction"],
                         "the strict threshold must NOT widen")
        self.assertEqual(1.0, ev["match_fraction_duration_corroborated"])

    def test_near_miss_with_wrong_duration_is_not_corroborated(self):
        # Same distance, but a 12 s cut of a 30 s source — a same-setup
        # coincidence, not the same cut. Distance alone never admits it.
        ev = self._tail_ev([(0.058, 12.0)])
        self.assertEqual(0.0, ev["match_fraction"])
        self.assertEqual(0.0, ev["match_fraction_duration_corroborated"])

    def test_beyond_the_band_never_corroborates(self):
        # Frame-exact duration but genuinely different content (0.3 away —
        # the measured range for new tail content is 0.2-0.6).
        ev = self._tail_ev([(0.30, 30.0)])
        self.assertEqual(0.0, ev["match_fraction"])
        self.assertEqual(0.0, ev["match_fraction_duration_corroborated"])

    def test_strict_matches_with_agreeing_duration_also_corroborate(self):
        # A subtitle-grade copy (distance ~0) counts in BOTH fractions; the
        # corroborated stat is a superset reading, not a disjoint band.
        ev = self._tail_ev([(0.0, 30.0), (0.058, 30.1)])
        self.assertEqual(0.5, ev["match_fraction"])
        self.assertEqual(1.0, ev["match_fraction_duration_corroborated"])


class TestMultiPartMasters(unittest.TestCase):

    def test_separator_is_the_black_with_most_content_after(self):
        # part1 [black A] part2 [black B] textless(60s): both blacks qualify;
        # black A has MORE content after it (part2 + textless), so A is the
        # separator — the outermost structure, matching moat.py's choice
        shots = _programme(20)                                    # 0..600
        shots.append(_shot(100, 600.0, 660.0, kind="black"))      # A
        t = 660.0
        for k in range(10):                                       # part 2
            shots.append(_shot(300 + k, t, t + 30.0, emb=_emb(9000 + k)))
            t += 30.0                                             # ..960
        shots.append(_shot(101, t, t + 45.0, kind="black"))       # B
        t += 45.0
        for k in range(3):
            shots.append(_shot(400 + k, t, t + 30.0))
            t += 30.0
        ev = analyse_programme_tail(shots)
        self.assertEqual(100, ev["separator_shot_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
