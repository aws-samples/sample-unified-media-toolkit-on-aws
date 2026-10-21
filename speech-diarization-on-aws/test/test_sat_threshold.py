#!/usr/bin/env python3
"""Regression test for the SaT threshold-semantics bug (2026-09-16):
the boundary threshold is a PROBABILITY (upstream wtpsplit: 0.025 for
sat models), but _logits_to_sentences compared RAW LOGITS against it —
equivalent to demanding P>50.6%. Invisible on punctuated English;
measured on AISHELL-4 Mandarin as 16 "sentences" in a 36-minute
5-person meeting (457 s monsters) and 42% DER confusion.

Model-free: drives _logits_to_sentences with synthetic logits.

    python3 test/test_sat_threshold.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.wtpsplit_sat import SaT  # noqa: E402


def convert(text, logits, offsets, threshold):
    return SaT._logits_to_sentences(
        None, text, np.array(logits, dtype=np.float32), offsets,
        threshold)


def test_moderate_confidence_boundary_splits():
    """A boundary the model scores at P~12% (logit -2) must split at
    threshold 0.025 — this is the comma-only-Mandarin regime that the
    raw-logit comparison silently dropped."""
    text = "abcdef"
    offsets = [(0, 2), (2, 4), (4, 6)]
    sents = convert(text, [-8.0, -2.0, -8.0], offsets, 0.025)
    assert sents == ["abcd", "ef"], sents
    print("  ok P~12% boundary splits at threshold 0.025")


def test_low_scores_do_not_split():
    """Non-boundary tokens (P~e-4, logit -9) must not split."""
    text = "abcdef"
    offsets = [(0, 2), (2, 4), (4, 6)]
    sents = convert(text, [-9.0, -9.0, -9.0], offsets, 0.025)
    assert sents == ["abcdef"], sents
    print("  ok non-boundaries stay unsplit")


def test_confident_boundary_still_splits():
    """Punctuated-English regime (logit +5, P~99%) unchanged."""
    text = "abcdef"
    offsets = [(0, 2), (2, 4), (4, 6)]
    sents = convert(text, [-9.0, 5.0, -9.0], offsets, 0.025)
    assert sents == ["abcd", "ef"], sents
    print("  ok confident boundaries unchanged")


if __name__ == "__main__":
    test_moderate_confidence_boundary_splits()
    test_low_scores_do_not_split()
    test_confident_boundary_still_splits()
    print("all SaT threshold tests passed")
