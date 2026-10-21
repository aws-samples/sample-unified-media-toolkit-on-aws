"""lib/f0 — per-sentence F0 statistics (2026-09-23).

Synthetic ground truth only: pure tones and noise are the one case
where F0 has an exact answer, so the tests pin accuracy without any
audio fixture. The module is evidence-not-verdict — it emits Hz, never
a gender — and additive-only (nothing here touches clustering)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.f0 import aggregate_f0, slice_f0  # noqa: E402

SR = 16000


def _tone(hz: float, sec: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    # a slightly harmonic-rich tone (speech is never a pure sine)
    return (0.8 * np.sin(2 * np.pi * hz * t)
            + 0.2 * np.sin(2 * np.pi * 2 * hz * t)).astype(np.float32)



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


class TestSliceF0(unittest.TestCase):
    def test_low_and_high_tones_recovered(self):
        for hz in (120.0, 220.0):
            f0, voiced = slice_f0(_tone(hz), SR)
            self.assertIsNotNone(f0, hz)
            self.assertLess(abs(f0 - hz) / hz, 0.06,
                            f"{hz} Hz estimated as {f0}")
            self.assertGreater(voiced, 0.8)

    def test_noise_is_unvoiced(self):
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(SR * 2).astype(np.float32)
        f0, voiced = slice_f0(noise, SR)
        self.assertIsNone(f0)
        self.assertEqual(voiced, 0.0)

    def test_too_short_and_empty_abstain(self):
        self.assertEqual(slice_f0(np.zeros(10, dtype=np.float32), SR),
                         (None, 0.0))
        self.assertEqual(slice_f0(None, SR), (None, 0.0))

    def test_long_slice_is_capped_not_rejected(self):
        f0, _ = slice_f0(_tone(180.0, sec=20.0), SR)
        self.assertIsNotNone(f0)
        self.assertLess(abs(f0 - 180.0) / 180.0, 0.06)


class TestAggregateF0(unittest.TestCase):
    def test_rollup_is_median_of_sentence_medians(self):
        stats = aggregate_f0([(118.0, 0.9), (122.0, 0.8), (None, 0.0),
                              (126.0, 0.7)])
        self.assertEqual(stats["f0_n_sentences"], 3)
        self.assertAlmostEqual(stats["f0_median_hz"], 122.0, places=1)
        self.assertGreater(stats["f0_iqr_hz"], 0.0)

    def test_nothing_voiced_abstains(self):
        self.assertIsNone(aggregate_f0([(None, 0.0), (None, 0.0)]))
        self.assertIsNone(aggregate_f0([]))


if __name__ == "__main__":
    unittest.main()
