#!/usr/bin/env python3
"""Hermetic unittest for _collect_overlap_spans (merge state, 2026-08-26):
the cross-talk spans surfaced into final_transcript.json. No S3 — the
loader is injected.

    python3 -m unittest discover -s test -p 'test_overlap_spans.py'
"""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from states.merge_speakers_transcripts.lambda_function import (  # noqa: E402
    _collect_overlap_spans,
)


def _turn(start, end, overlap):
    return {"start": start, "end": end, "duration": round(end - start, 3),
            "has_overlap": overlap, "turn_idx": 0}



# Silence console log HANDLERS for the whole suite: the failure-path
# tests exercise refusal/retry/fallback branches whose designed
# behavior includes ERROR/WARNING logs, and those lines made build
# logs read as if the build were failing. Handlers only — record flow
# is untouched, so assertLogs() still sees everything.
import logging as _logging


def _silence_console_logging():
    _logging.getLogger().handlers = [_logging.NullHandler()]
    for _name in list(_logging.root.manager.loggerDict):
        _lg = _logging.getLogger(_name)
        if _lg.handlers:
            _lg.handlers = [_logging.NullHandler()]


_silence_console_logging()


class TestCollectOverlapSpans(unittest.TestCase):

    def _load_from(self, docs):
        def load(bucket, key):
            if key not in docs:
                raise RuntimeError(f"missing {key}")
            return docs[key]
        return load

    def test_flagged_turns_become_sorted_merged_spans(self):
        docs = {
            "dyn/0.json": {"turns": [_turn(0.0, 2.0, False),
                                     _turn(5.0, 7.5, True),
                                     _turn(7.5, 9.0, True)]},   # touching → merge
            "dyn/1.json": {"turns": [_turn(20.0, 22.25, True),
                                     _turn(30.0, 31.0, False)]},
        }
        its = [{"dyn_chunk_idx": 1, "dyn_segment_key": "dyn/1.json"},
               {"dyn_chunk_idx": 0, "dyn_segment_key": "dyn/0.json"}]
        spans = _collect_overlap_spans("b", its, load=self._load_from(docs))
        self.assertEqual(spans, [[5.0, 9.0], [20.0, 22.25]])

    def test_no_flagged_turns_is_an_empty_list_not_an_error(self):
        docs = {"dyn/0.json": {"turns": [_turn(0.0, 2.0, False)]}}
        its = [{"dyn_chunk_idx": 0, "dyn_segment_key": "dyn/0.json"}]
        self.assertEqual(
            _collect_overlap_spans("b", its, load=self._load_from(docs)), [])

    def test_a_missing_doc_degrades_never_raises(self):
        # AUXILIARY by design: cross-talk spans must not fail a merge whose
        # transcript is otherwise complete.
        docs = {"dyn/1.json": {"turns": [_turn(3.0, 4.0, True)]}}
        its = [{"dyn_chunk_idx": 0, "dyn_segment_key": "dyn/0.json"},   # 404
               {"dyn_chunk_idx": 1, "dyn_segment_key": "dyn/1.json"},
               {"dyn_chunk_idx": 2}]                    # old payload, no key
        spans = _collect_overlap_spans("b", its, load=self._load_from(docs))
        self.assertEqual(spans, [[3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
