#!/usr/bin/env python3
"""
black_detection_test.py — pin the frame-level black classifier, and pin the
reason redigitised-tape sources must override black_entropy_max at request
time.

Run
---
  python -m unittest test.black_detection_test -v     # from repo root
  python test/black_detection_test.py

Why this exists
---------------
The shipped default black_entropy_max = 0.20 bits is correct for clean digital
sources and is deliberately left alone — this service is general purpose. But
no frame off a real tape can reach it. Black recorded to tape and redigitised
is dithered across several low luma codes by transport and encoder noise, so it
measures 0.4-2.8 bits of luma-histogram entropy while every pixel still sits
below luma 17. Measured over 9,514 shots / 6 Betacam SX news masters, 14 shots
were flat black to the eye yet typed "content"; 0.20 vetoed 14 of 14. Two of
them opened a story segment, so the shot list reported a segment start 8.3 s
and 1.7 s early — 4 boundary errors on an asset otherwise scoring 0.947.

Such callers pass black_entropy_max = TAPE_BLACK_ENTROPY_MAX below. The tests
encode the three claims that remedy rests on:

  1. The shipped default rejects dithered black, and the override accepts it.
     This is the defect, kept legible so nobody "fixes" it by tightening the
     override instead of loosening it.
  2. Coverage + max-bright-pixel, not entropy, are what separate black from
     dark content. A frame carrying a faint picture, and a title card on
     black, must both be rejected on those two criteria alone, i.e. with the
     entropy criterion switched off entirely.
  3. The override stays a bounded backstop rather than a silent no-op.

Test 2 is the one that justifies overriding entropy at all. If it ever fails,
entropy was load-bearing after all and the override is unsafe.

No AWS calls, no model, no fixtures — frames are synthesised in-process.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.input_schema import (                                  # noqa: E402
    DEFAULT_BLACK_COVERAGE_PCT,
    DEFAULT_BLACK_ENTROPY_MAX,
    DEFAULT_BLACK_MAX_BRIGHT_PIXEL,
    DEFAULT_BLACK_PIXEL_THRESHOLD,
)
from states.run_dino_v2_model.lambda_function import (           # noqa: E402
    _analyze_frame,
)

H, W = 288, 512
ENTROPY_OFF = 9.0     # above log2(256); disables criterion 3 outright

# What a redigitised-tape workflow must pass as black_entropy_max. NOT the
# shipped default, which stays 0.20 for clean digital sources.
TAPE_BLACK_ENTROPY_MAX = 3.5


def _rgb(luma_u8: np.ndarray) -> np.ndarray:
    """Grey RGB frame whose luma equals luma_u8 (R=G=B => luma == channel)."""
    return np.repeat(luma_u8[:, :, None], 3, axis=2).astype(np.uint8)


def _analyze(frame, ent_max: float = TAPE_BLACK_ENTROPY_MAX):
    return _analyze_frame(
        frame,
        DEFAULT_BLACK_PIXEL_THRESHOLD,
        DEFAULT_BLACK_COVERAGE_PCT,
        ent_max,
        DEFAULT_BLACK_MAX_BRIGHT_PIXEL,
    )


def _dithered_black(rng: np.random.Generator, hi: int = 8) -> np.ndarray:
    """Black as it comes off tape: every pixel in [0, hi], spread over codes."""
    return _rgb(rng.integers(0, hi + 1, size=(H, W), dtype=np.uint8))


class TestDitheredBlackIsDetected(unittest.TestCase):
    """Claim 1 — the override detects tape black; the default does not."""

    def test_dithered_black_classifies_black_under_the_tape_override(self):
        rng = np.random.default_rng(0)
        for hi in (4, 8, 10):
            with self.subTest(max_luma=hi):
                is_black, entropy = _analyze(_dithered_black(rng, hi))
                self.assertTrue(
                    is_black,
                    f"dithered black (luma 0-{hi}, entropy {entropy:.3f}) "
                    f"must be black at black_entropy_max="
                    f"{TAPE_BLACK_ENTROPY_MAX}")

    def test_dithered_black_entropy_sits_between_default_and_override(self):
        """The measurement it all rests on: real black is ~1-3 bits, not ~0."""
        rng = np.random.default_rng(1)
        _, entropy = _analyze(_dithered_black(rng, 8))
        self.assertGreater(entropy, DEFAULT_BLACK_ENTROPY_MAX)
        self.assertLess(entropy, TAPE_BLACK_ENTROPY_MAX)

    def test_shipped_default_rejects_tape_black(self):
        """
        Documents the exact failure, so the cause stays legible. The remedy is
        to pass a looser black_entropy_max for tape sources — NOT to tighten
        this expectation or to change the shipped default, which is right for
        the clean digital sources this service also serves.
        """
        rng = np.random.default_rng(2)
        frame = _dithered_black(rng, 8)
        self.assertFalse(
            _analyze(frame, ent_max=DEFAULT_BLACK_ENTROPY_MAX)[0],
            "if the default ever accepts dithered black, the tape override is "
            "no longer needed — revisit the request-time guidance")
        self.assertTrue(_analyze(frame)[0])

    def test_uniform_black_detected_at_the_shipped_default(self):
        """The default is not broken — synthetic black satisfies it."""
        self.assertTrue(
            _analyze(_rgb(np.zeros((H, W), np.uint8)),
                     ent_max=DEFAULT_BLACK_ENTROPY_MAX)[0])

    def test_uniform_black_detected_under_the_tape_override(self):
        """Loosening entropy must not regress synthetic black."""
        self.assertTrue(_analyze(_rgb(np.zeros((H, W), np.uint8)))[0])


class TestCoverageAndBrightPixelDoTheSeparating(unittest.TestCase):
    """
    Claim 2 — with entropy disabled, coverage + max-bright-pixel alone must
    still reject everything that is not black.
    """

    def test_faint_picture_rejected_without_entropy(self):
        """A dark frame carrying real structure fails on coverage."""
        rng = np.random.default_rng(3)
        luma = rng.integers(0, 9, size=(H, W)).astype(np.uint8)
        luma[H // 3:2 * H // 3, W // 3:2 * W // 3] = 40      # visible subject
        is_black, _ = _analyze(_rgb(luma), ent_max=ENTROPY_OFF)
        self.assertFalse(is_black, "faint picture must not be black")

    def test_title_card_on_black_rejected_without_entropy(self):
        """Sparse bright text passes coverage, must fail max-bright-pixel."""
        rng = np.random.default_rng(4)
        luma = rng.integers(0, 9, size=(H, W)).astype(np.uint8)
        luma[10:14, 10:30] = 235                              # ~0.05% of pixels
        self.assertGreaterEqual(
            100.0 * np.mean(luma < DEFAULT_BLACK_PIXEL_THRESHOLD * 255),
            DEFAULT_BLACK_COVERAGE_PCT,
            "fixture invalid: this must pass coverage so the bright-pixel "
            "criterion is the thing under test")
        is_black, _ = _analyze(_rgb(luma), ent_max=ENTROPY_OFF)
        self.assertFalse(is_black, "title card on black must not be black")

    def test_ordinary_content_rejected_without_entropy(self):
        rng = np.random.default_rng(5)
        luma = rng.integers(0, 256, size=(H, W)).astype(np.uint8)
        self.assertFalse(_analyze(_rgb(luma), ent_max=ENTROPY_OFF)[0])

    def test_dithered_black_survives_entropy_being_disabled(self):
        """The pair is sufficient: black is still black with entropy off."""
        rng = np.random.default_rng(6)
        self.assertTrue(
            _analyze(_dithered_black(rng, 8), ent_max=ENTROPY_OFF)[0])


class TestEntropyRemainsABoundedBackstop(unittest.TestCase):
    """
    Claim 3 — criteria 1+2 confine luma to codes 0-30, so entropy cannot
    exceed log2(31) = 4.95 for any frame that reaches criterion 3. A value at
    or above that ceiling makes the parameter a silent no-op; staying below it
    keeps the override a backstop rather than a bypass.
    """

    def test_tape_override_is_below_the_reachable_ceiling(self):
        ceiling = float(np.log2(DEFAULT_BLACK_MAX_BRIGHT_PIXEL + 1.0))
        self.assertLess(TAPE_BLACK_ENTROPY_MAX, ceiling)

    def test_tape_override_clears_measured_tape_black(self):
        """0.389-2.777 bits measured over 6 assets; leave headroom above it."""
        self.assertGreater(TAPE_BLACK_ENTROPY_MAX, 2.777)

    def test_shipped_default_is_below_the_reachable_ceiling(self):
        ceiling = float(np.log2(DEFAULT_BLACK_MAX_BRIGHT_PIXEL + 1.0))
        self.assertLess(DEFAULT_BLACK_ENTROPY_MAX, ceiling)


if __name__ == "__main__":
    unittest.main(verbosity=2)
