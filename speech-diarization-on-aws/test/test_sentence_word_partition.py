"""Word-to-sentence assignment is a PARTITION (lib/sentence_split mappers
+ lib/transcribe step 5) — pinned to the measured defect (2026-10-19):

The original pipeline re-derived sentence membership from TIME spans
(`sent.start <= w.start <= sent.end`), so a boundary word whose span fell
inside two overlapping sentence spans joined BOTH word lists and was
emitted twice into the final transcript (~0.64% of words on the English
caption corpus, every build since March; worse for CJK). The assembler's
char-level ownership was always computed and then thrown away — these
tests pin that it is now returned (word_ids) and that it partitions.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.sentence_split import SentenceAssembler  # noqa: E402


def words_from(pairs):
    """pairs: list of (raw_token, start, end)."""
    return [{"word": r.strip(), "_raw": r, "start": s, "end": e,
             "probability": 0.9}
            for r, s, e in pairs]


class TestWordIndexPartition(unittest.TestCase):

    def setUp(self):
        self.asm = SentenceAssembler(language="en")

    def test_boundary_word_owned_by_exactly_one_sentence(self):
        # the measured shape: the second sentence's first word overlaps
        # the first sentence's last word in TIME; the text partitions them
        ww = words_from([
            ("Your", 602.12, 602.40), (" Honor,", 602.45, 602.90),
            (" objection.", 603.00, 604.34),
            (" Ms.", 604.02, 604.34),          # time-overlaps previous word
            (" Chandler,", 604.40, 605.00), (" please.", 605.10, 605.60),
        ])
        sents = ["Your Honor, objection.", "Ms. Chandler, please."]
        res = self.asm._map_word_index(sents, ww)
        self.assertEqual(len(res), 2)
        all_ids = [i for r in res for i in r["word_ids"]]
        self.assertEqual(len(all_ids), len(set(all_ids)),
                         "a word id appears in two sentences")
        self.assertEqual(res[0]["word_ids"], [0, 1, 2])
        self.assertEqual(res[1]["word_ids"], [3, 4, 5])
        # the boundary word belongs to the sentence whose TEXT carries it
        self.assertNotIn(3, res[0]["word_ids"])

    def test_every_word_owned_despite_time_overlaps(self):
        ww = words_from([(" a", 0.0, 0.5), (" b", 0.4, 0.9),
                         (" c", 0.8, 1.3)])
        res = self.asm._map_word_index(["a b", "c"], ww)
        owned = sorted(i for r in res for i in r["word_ids"])
        self.assertEqual(owned, [0, 1, 2])


class TestCharRatioPartition(unittest.TestCase):

    def test_consecutive_ranges_never_share_a_boundary_index(self):
        asm = SentenceAssembler(language="ja")
        ww = words_from([(c, i * 0.5, i * 0.5 + 0.4)
                         for i, c in enumerate("abcdefghij")])
        res = asm._map_char_ratio(["abcde", "fghij"], ww)
        all_ids = [i for r in res for i in r["word_ids"]]
        self.assertEqual(len(all_ids), len(set(all_ids)),
                         "char-ratio ranges overlap on a boundary index")


if __name__ == "__main__":
    unittest.main()
