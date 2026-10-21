"""Seam-twin dedup (lib/transcribe._dedup_seam_twins) — pinned to the
measured incident shapes (2026-10-19 insertion-anatomy probe):

- A VAD group boundary that lands mid-word makes BOTH adjacent group
  windows decode the word; the aligner anchors both copies to the same
  audio event. The pad ladder cannot see them (they sit in owned audio,
  not the synthetic pad). Measured ~0.3% of words on the English caption
  corpus and ~6% on a Korean asset before the fix.
- Genuine repetitions carry disjoint advancing spans and must NEVER be
  merged (the shouted-repeat case is pinned below).
- The trigger is language-agnostic by construction: casefold equality
  (no linguistic normalization) + span IoU, nothing else.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.transcribe import _dedup_seam_twins  # noqa: E402


def w(word, start, end, prob=0.9):
    return {"word": word, "start": start, "end": end, "probability": prob}


class TestSeamTwinDedup(unittest.TestCase):

    def test_identical_time_twin_is_dropped_keeping_higher_probability(self):
        # the measured English shape: same text, identical aligned span,
        # one copy per adjacent group window
        words = [w("Ms.", 604.02, 604.34, prob=0.71),
                 w("Ms.", 604.02, 604.34, prob=0.93),
                 w("Chandler,", 604.40, 604.90)]
        out, dropped = _dedup_seam_twins(words)
        self.assertEqual(dropped, 1)
        self.assertEqual([x["word"] for x in out], ["Ms.", "Chandler,"])
        self.assertEqual(out[0]["probability"], 0.93)

    def test_genuine_repetition_with_advancing_spans_is_kept(self):
        # the measured must-not-merge case: a real shouted repeat has
        # disjoint, advancing spans — IoU is 0
        words = [w("Despacio!", 273.03, 273.43),
                 w("Despacio!", 273.43, 273.83)]
        out, dropped = _dedup_seam_twins(words)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(out), 2)

    def test_partial_overlap_below_half_iou_is_kept(self):
        # fast stutter with mostly-disjoint spans stays untouched
        words = [w("no", 10.00, 10.40), w("no", 10.30, 10.80)]
        out, dropped = _dedup_seam_twins(words)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(out), 2)

    def test_heavy_overlap_above_half_iou_is_judged(self):
        words = [w("no", 10.00, 10.40), w("no", 10.05, 10.42)]
        out, dropped = _dedup_seam_twins(words)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(out), 1)

    def test_casefold_equality_only_never_linguistic_normalization(self):
        # "No," vs "no," differ only by case -> twin; but "cant" vs
        # "can't" differ textually and MUST both survive (no contraction
        # or punctuation folding — that is scorer logic, not pipeline)
        twins = [w("No,", 5.0, 5.3), w("no,", 5.0, 5.3)]
        out, dropped = _dedup_seam_twins(twins)
        self.assertEqual(dropped, 1)
        distinct = [w("cant", 7.0, 7.3), w("can't", 7.0, 7.3)]
        out, dropped = _dedup_seam_twins(distinct)
        self.assertEqual(dropped, 0)

    def test_cjk_twin_handled_identically(self):
        # the Korean measured shape (character-based segmentation makes
        # phrase-chunk words straddle seams ~20x more often than English)
        word = "아이씨!"
        words = [w(word, 48.69, 49.10, prob=0.8),
                 w(word, 48.69, 49.10, prob=0.85)]
        out, dropped = _dedup_seam_twins(words)
        self.assertEqual(dropped, 1)
        self.assertEqual(out[0]["probability"], 0.85)

    def test_missing_timestamps_abstain(self):
        # abstention over deletion: unjudgeable words are never dropped
        words = [w("word", 1.0, 1.3),
                 {"word": "word", "start": None, "end": None,
                  "probability": 0.9}]
        out, dropped = _dedup_seam_twins(words)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(out), 2)

    def test_triple_twin_chain_collapses_to_one(self):
        words = [w("yes", 2.0, 2.3, prob=0.5),
                 w("yes", 2.0, 2.3, prob=0.9),
                 w("yes", 2.01, 2.31, prob=0.7)]
        out, dropped = _dedup_seam_twins(words)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["probability"], 0.9)

    def test_tie_keeps_the_earlier_copy(self):
        words = [w("hey", 3.0, 3.2, prob=0.8),
                 w("hey", 3.0, 3.2, prob=0.8)]
        out, dropped = _dedup_seam_twins(words)
        self.assertEqual(dropped, 1)
        self.assertIs(out[0] is words[0], True)


if __name__ == "__main__":
    unittest.main()
