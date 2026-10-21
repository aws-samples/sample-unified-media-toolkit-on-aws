#!/usr/bin/env python3
"""
burst_evidence_test.py — 9.15: burst runs are contiguous intervals of
sub-second shots, published so consumers can fold each run into one
inference instead of paying per member.

Run
---
  python -m unittest test.burst_evidence_test -v     # from repo root
  python test/burst_evidence_test.py

Pinned properties:
  1. a run is an INTERVAL of shot ids — {start_shot_id, end_shot_id} and
     nothing else, on the trim discipline (everything else joins);
  2. shot TYPE is ignored: black micro-shots RIDE ALONG, because jingles
     separate their cards with them and a type filter breaks every run at
     each separator (measured: jingle's longest run reads 8 content-only,
     35 type-blind);
  3. the qualification edges: exactly BURST_MIN_RUN qualifies, one fewer
     does not; a shot at exactly BURST_MAX_SHOT_SEC breaks the run (strict
     <, matching the corpus measurement in README.md 9.15);
  4. ordinary content produces an empty list, and a trailing run at end of
     file is not lost.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.burst_evidence import detect_burst_runs               # noqa: E402
from lib.input_schema import BURST_MAX_SHOT_SEC, BURST_MIN_RUN  # noqa: E402


def _shots(durations, kinds=None, start_id=0):
    t, out = 0.0, []
    for i, d in enumerate(durations):
        out.append({"shot_id": start_id + i,
                    "shot_type": (kinds[i] if kinds else "content"),
                    "start_time": t, "end_time": t + d})
        t += d
    return out


class TestBurstRuns(unittest.TestCase):

    def test_interval_only_shape(self):
        runs = detect_burst_runs(_shots([0.4] * BURST_MIN_RUN + [5.0]))
        self.assertEqual([{"start_shot_id": 0,
                           "end_shot_id": BURST_MIN_RUN - 1}], runs)

    def test_black_separators_ride_along(self):
        # card, black, card, black... — the jingle shape; a content-only
        # rule would break the run at every black
        n = BURST_MIN_RUN + 4
        kinds = ["content" if i % 2 == 0 else "black" for i in range(n)]
        runs = detect_burst_runs(_shots([0.4] * n + [5.0],
                                        kinds=kinds + ["content"]))
        self.assertEqual([{"start_shot_id": 0, "end_shot_id": n - 1}], runs)

    def test_one_short_of_min_run_is_nothing(self):
        self.assertEqual([], detect_burst_runs(
            _shots([0.4] * (BURST_MIN_RUN - 1) + [5.0])))

    def test_exactly_max_duration_breaks_the_run(self):
        # strict <: a 1.0 s shot is NOT a burst member, and it splits two
        # sub-minimum halves into nothing
        half = BURST_MIN_RUN // 2
        durs = [0.4] * half + [BURST_MAX_SHOT_SEC] + [0.4] * half + [5.0]
        self.assertEqual([], detect_burst_runs(_shots(durs)))

    def test_ordinary_content_is_empty(self):
        self.assertEqual([], detect_burst_runs(_shots([3.0, 8.0, 12.0, 2.5])))

    def test_trailing_run_at_end_of_file_is_kept(self):
        durs = [5.0, 6.0] + [0.4] * BURST_MIN_RUN
        runs = detect_burst_runs(_shots(durs))
        self.assertEqual([{"start_shot_id": 2,
                           "end_shot_id": 1 + BURST_MIN_RUN}], runs)

    def test_multiple_runs_stay_separate(self):
        durs = ([0.4] * BURST_MIN_RUN + [5.0]
                + [0.3] * (BURST_MIN_RUN + 2) + [4.0])
        runs = detect_burst_runs(_shots(durs))
        self.assertEqual(2, len(runs))
        self.assertEqual(BURST_MIN_RUN + 1, runs[1]["start_shot_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
