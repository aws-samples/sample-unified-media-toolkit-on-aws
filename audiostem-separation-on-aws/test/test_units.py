# test/test_units.py
"""Hermetic tests for lib/audio_analysis.py — the DSP behind
stem_activity.json and stem_waveforms.json (README.md Design record §4).

numpy only; no S3, no boto3, no model. The stress harness
(test/stress_test.py) deliberately does not match this discover pattern.

    python3 -m unittest discover -s test -p 'test_*.py'
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib import audio_analysis as aa                      # noqa: E402

SR = 48_000


def sine(freq_hz: float, dur_sec: float, level: float = 0.5,
         sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur_sec * sr), dtype=np.float64) / sr
    return (level * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def silence(dur_sec: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(dur_sec * sr), dtype=np.float32)



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


class TestEnvelopeAccumulator(unittest.TestCase):

    def test_streaming_equals_whole_array_regardless_of_chunking(self):
        """Chunk boundaries never align to bins — the carry buffer is the
        whole point. Any chunking must give the same envelope."""
        rng = np.random.default_rng(7)
        x = (rng.standard_normal(SR * 3) * 0.1).astype(np.float32)
        whole = aa.EnvelopeAccumulator(SR)
        whole.feed(x)
        ref = whole.finish()
        for sizes in ([SR * 3], [1000] * ((SR * 3) // 1000) + [0],
                      [7, SR, 13, SR * 2 - 20]):
            acc = aa.EnvelopeAccumulator(SR)
            pos = 0
            for n in sizes:
                acc.feed(x[pos: pos + n])
                pos += n
            acc.feed(x[pos:])
            np.testing.assert_allclose(acc.finish(), ref, rtol=1e-6)

    def test_bin_count_and_level(self):
        acc = aa.EnvelopeAccumulator(SR)
        acc.feed(sine(440, 1.0, level=0.5))
        rms = acc.finish()
        self.assertEqual(len(rms), 1000 // aa.BIN_MS)
        # RMS of a 0.5-amplitude sine is 0.5/sqrt(2)
        np.testing.assert_allclose(rms, 0.5 / np.sqrt(2), rtol=1e-2)

    def test_waveform_values_are_peak_normalized(self):
        vals = aa.waveform_values(np.array([0.0, 0.2, 0.4]))
        self.assertEqual(vals, [0.0, 0.5, 1.0])
        self.assertEqual(aa.waveform_values(np.zeros(3)), [0.0, 0.0, 0.0])


class TestActivitySpans(unittest.TestCase):

    def _rms(self, *segments):
        """Build an RMS bin array from (level, dur_sec) pairs."""
        acc = aa.EnvelopeAccumulator(SR)
        for level, dur in segments:
            acc.feed(sine(300, dur, level=level) if level else silence(dur))
        return acc.finish()

    def test_spans_threshold_merge_and_minimum(self):
        # loud 2 s | silent 0.4 s (bridged: < merge gap) | loud 2 s |
        # silent 3 s | loud 0.2 s (dropped: < min span)
        rms = self._rms((0.3, 2.0), (0.0, 0.4), (0.3, 2.0),
                        (0.0, 3.0), (0.3, 0.2))
        spans = aa.activity_spans(rms)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertEqual(start, 0)
        self.assertAlmostEqual(end, 4400, delta=2 * aa.BIN_MS)

    def test_silence_is_no_spans_and_quiet_is_below_threshold(self):
        self.assertEqual(aa.activity_spans(self._rms((0.0, 3.0))), [])
        # -60 dBFS sine sits below the -45 dBFS default threshold
        self.assertEqual(aa.activity_spans(self._rms((0.001, 3.0))), [])


class TestOutputConfigRefusesBeforeSpend(unittest.TestCase):
    """An all-disabled output_config must fail in state-1 validation —
    BEFORE the media-processing conversion is submitted and billed."""

    def _event(self, oc):
        return {"uuid": "u", "input": {"s3uri": "s3://b/k.mp4"},
                "output": {"s3uri": "s3://b/out", "output_config": oc},
                "data": {}}

    def test_all_outputs_disabled_is_refused_at_validation(self):
        from lib.input_schema import validate_and_normalize
        oc = {"speech_stem": False, "music_stem": False,
              "sfx_stem": False, "speech_safe": False}
        with self.assertRaises(ValueError):
            validate_and_normalize(self._event(oc))

    def test_speech_safe_alone_is_a_valid_output(self):
        from lib.input_schema import validate_and_normalize
        oc = {"speech_stem": False, "music_stem": False,
              "sfx_stem": False, "speech_safe": True}
        ev = validate_and_normalize(self._event(oc))
        self.assertTrue(ev["output"]["output_config"]["speech_safe"])


class TestStemGapSpans(unittest.TestCase):
    """speech_stem_gaps — stem-coverage gaps vs the mix (README.md Design
    record §5): mix active while the speech stem sits below the gap
    threshold. An upper bound on deleted speech by design."""

    def _rms(self, *segments):
        acc = aa.EnvelopeAccumulator(SR)
        for level, dur in segments:
            acc.feed(sine(300, dur, level=level) if level else silence(dur))
        return acc.finish()

    def test_gap_region_is_flagged(self):
        # mix loud throughout; stem deletes the middle 2 s to digital zero
        original = self._rms((0.3, 6.0))
        stem = self._rms((0.3, 2.0), (0.0, 2.0), (0.3, 2.0))
        spans = aa.stem_gap_spans(original, stem)
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0][0], 2000, delta=2 * aa.BIN_MS)
        self.assertAlmostEqual(spans[0][1], 4000, delta=2 * aa.BIN_MS)

    def test_both_active_and_both_silent_are_not_gaps(self):
        original = self._rms((0.3, 2.0), (0.0, 2.0))
        stem = self._rms((0.3, 2.0), (0.0, 2.0))
        self.assertEqual(aa.stem_gap_spans(original, stem), [])

    def test_quiet_stem_above_threshold_is_not_a_gap(self):
        # -40 dBFS surviving stem (0.014 amplitude) is quiet but PRESENT —
        # the -55 threshold separates model deletion from quiet speech
        original = self._rms((0.3, 3.0))
        stem = self._rms((0.014, 3.0))
        self.assertEqual(aa.stem_gap_spans(original, stem), [])

    def test_length_mismatch_from_stream_remainders_is_truncated(self):
        original = self._rms((0.3, 3.0))
        stem = self._rms((0.0, 3.0))
        self.assertEqual(len(aa.stem_gap_spans(original[:-1], stem)),
                         len(aa.stem_gap_spans(original, stem)))

    def test_short_blip_is_dropped_and_word_gap_is_bridged(self):
        # 0.1 s deletion (< 250 ms min span) is envelope flutter, not a
        # deletion; two deletions 0.4 s apart (< 500 ms gap) merge
        original = self._rms((0.3, 6.0))
        blip = self._rms((0.3, 2.0), (0.0, 0.1), (0.3, 3.9))
        self.assertEqual(aa.stem_gap_spans(original, blip), [])
        pair = self._rms((0.3, 1.0), (0.0, 1.0), (0.3, 0.4),
                         (0.0, 1.0), (0.3, 2.6))
        self.assertEqual(len(aa.stem_gap_spans(original, pair)), 1)


class TestToneDetector(unittest.TestCase):

    def _detect(self, x, sr=SR, chunks=None):
        det = aa.ToneDetector(sr)
        if chunks:
            pos = 0
            for n in chunks:
                det.feed(x[pos:pos + n])
                pos += n
            det.feed(x[pos:])
        else:
            det.feed(x)
        return det.finish()

    def test_a_drifted_tape_tone_reads_measured_and_matches_nominal(self):
        """The tolerance decision (user, 2026-08-26): 998.6 Hz is reported
        as ~998.6 (parabolic interpolation, not the nearest FFT bin) and
        still joins as nominal 1000."""
        tones = self._detect(sine(998.6, 5.0), chunks=[SR, SR // 3, SR * 2])
        self.assertEqual(len(tones), 1)
        t = tones[0]
        self.assertEqual(t["nominal_hz"], 1000)
        self.assertEqual(t["kind"], "steady")
        self.assertLess(abs(t["freq_hz"] - 998.6), 2.0)
        self.assertLessEqual(abs(t["start_ms"] - 0), aa.TONE_FRAME_MS)
        self.assertLessEqual(abs(t["end_ms"] - 5000), aa.TONE_FRAME_MS)

    def test_each_nominal_frequency_is_matched(self):
        for nom in aa.TONE_NOMINALS_HZ:
            with self.subTest(nominal=nom):
                tones = self._detect(sine(nom, 3.0))
                self.assertEqual([t["nominal_hz"] for t in tones], [nom])

    def test_out_of_tolerance_tone_reports_null_nominal(self):
        # A sustained 880 Hz musical note: pure and steady, but matches no
        # line-up frequency — reported as evidence with nominal null.
        tones = self._detect(sine(880, 3.0))
        self.assertEqual(len(tones), 1)
        self.assertIsNone(tones[0]["nominal_hz"])

    def test_the_two_pop_is_a_pop_not_a_steady(self):
        x = np.concatenate([silence(2.0), sine(1000, 0.25), silence(2.0)])
        tones = self._detect(x)
        self.assertEqual(len(tones), 1)
        self.assertEqual(tones[0]["kind"], "pop")
        self.assertEqual(tones[0]["nominal_hz"], 1000)

    def test_noise_and_tremolo_are_not_tones(self):
        rng = np.random.default_rng(3)
        noise = (rng.standard_normal(SR * 3) * 0.2).astype(np.float32)
        self.assertEqual(self._detect(noise), [])
        # 1 kHz with a deep 2 Hz tremolo: pure per frame, but the level
        # will not hold — TONE_MAX_LEVEL_STD_DB rejects the span.
        t = np.arange(SR * 3, dtype=np.float64) / SR
        trem = (np.sin(2 * np.pi * 1000 * t)
                * (0.5 + 0.45 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)
        self.assertEqual([x for x in self._detect(trem)
                          if x["kind"] == "steady"], [])

    def test_silence_yields_nothing(self):
        self.assertEqual(self._detect(silence(3.0)), [])

    def test_a_null_nominal_pop_is_not_reported(self):
        # Sweep-B calibration (n=6): short pure musical notes pass the
        # purity gate by the hundreds; a pop without a line-up frequency
        # is program music, not a sync mark. Steady null-nominal spans
        # (the 880 Hz test above) stay reportable.
        x = np.concatenate([silence(1.0), sine(880, 0.25), silence(1.0)])
        self.assertEqual(self._detect(x), [])

    def test_two_distinct_tones_are_two_spans(self):
        x = np.concatenate([sine(1000, 2.0), sine(2000, 2.0)])
        tones = self._detect(x)
        self.assertEqual([t["nominal_hz"] for t in tones], [1000, 2000])


class TestParamsDoc(unittest.TestCase):

    def test_the_effective_knobs_are_recorded(self):
        p = aa.params_doc()
        for key in ("bin_ms", "activity_threshold_dbfs", "tone_freqs_hz",
                    "tone_tolerance", "tone_min_purity"):
            self.assertIn(key, p)
        self.assertEqual(p["tone_freqs_hz"], [400, 1000, 2000])


if __name__ == "__main__":
    unittest.main()
