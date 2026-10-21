#!/usr/bin/env python3
"""Unit tests for word-timing provenance + word-tight speech spans
(multilingual requirement, 2026-09-16). Self-contained.

    python3 test/test_word_timing.py
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.model_registry import word_timing_source  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "merge_mod", os.path.join(os.path.dirname(__file__), "..",
                              "states", "merge_speakers_transcripts",
                              "lambda_function.py"))
merge_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_mod)
_speech_spans = merge_mod._speech_spans


def test_timing_source():
    assert word_timing_source({"s3_key": "x"}, "FR") == "aligned:fr"
    assert word_timing_source({"s3_key": "x", "fallback": True},
                              "pl") == "aligned:en_fallback"
    assert word_timing_source({"s3_key": "x", "_fallback": True},
                              "pl") == "aligned:en_fallback"
    assert word_timing_source({}, "ja") == "whisper_native"
    assert word_timing_source(None, "ko") == "whisper_native"
    print("  ok timing-source mapping")


def test_spans_split_at_gaps():
    words = [{"start": 1.0, "end": 1.4}, {"start": 1.5, "end": 2.0},
             {"start": 3.2, "end": 3.6}]          # 1.2 s internal pause
    spans = _speech_spans(words, 1.0, 3.6, gap=0.5)
    assert spans == [[1.0, 2.0], [3.2, 3.6]], spans
    print("  ok spans split at gaps > 0.5s")


def test_spans_unsplit_without_alignment():
    """whisper_native (gap=None) must NOT trim — the multilingual
    guard: coarse word boundaries would cut real speech."""
    words = [{"start": 1.0, "end": 1.4}, {"start": 3.2, "end": 3.6}]
    assert _speech_spans(words, 0.8, 3.9, gap=None) == [[0.8, 3.9]]
    assert _speech_spans([], 0.8, 3.9, gap=0.5) == [[0.8, 3.9]]
    print("  ok whisper_native / empty words stay unsplit")


if __name__ == "__main__":
    test_timing_source()
    test_spans_split_at_gaps()
    test_spans_unsplit_without_alignment()
    print("all word-timing tests passed")
