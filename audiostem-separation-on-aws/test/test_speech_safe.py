# test/test_speech_safe.py
"""Hermetic tests for lib/speech_safe.py — gated mix substitution
(README.md Design record §5), pinned to the measured deletion shape:
the separator writes DIGITAL ZERO over low-prominence speech while the
mix still carries it, so the gate must patch exactly there and leave
every surviving region bit-identical.

numpy only; no S3, no boto3, no model.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib import speech_safe as ss                          # noqa: E402
from test_units import SR, silence, sine                   # noqa: E402


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt((x.astype(np.float64) ** 2).mean())) if x.size else 0.0
    return 20.0 * np.log10(max(r, 1e-10))


class TestBuildSpeechSafe(unittest.TestCase):

    def test_deleted_span_is_patched_at_the_configured_gain(self):
        # the measured shape: mix has speech throughout; the stem carries
        # the first 2 s and deletes the middle 2 s to digital zero
        mix = sine(300, 6.0, level=0.3)
        stem = np.concatenate([mix[: 2 * SR], silence(2.0), mix[4 * SR:]])
        safe, spans = ss.build_speech_safe(mix, stem, SR)

        self.assertEqual(len(spans), 1)
        start_ms, end_ms = spans[0]
        self.assertAlmostEqual(start_ms, 2000, delta=2 * ss.SPEECH_SAFE_BIN_MS)
        self.assertAlmostEqual(end_ms, 4000, delta=2 * ss.SPEECH_SAFE_BIN_MS)

        # patch interior (clear of the crossfade) carries mix at the gain
        core = safe[int(2.5 * SR): int(3.5 * SR)]
        expect_db = rms_db(mix[int(2.5 * SR): int(3.5 * SR)]) \
            + ss.SPEECH_SAFE_PATCH_GAIN_DB
        self.assertAlmostEqual(rms_db(core), expect_db, delta=0.5)

    def test_surviving_regions_are_bit_identical_beyond_a_fade(self):
        mix = sine(300, 6.0, level=0.3)
        stem = np.concatenate([mix[: 2 * SR], silence(2.0), mix[4 * SR:]])
        safe, _ = ss.build_speech_safe(mix, stem, SR)
        fade = int(SR * ss.SPEECH_SAFE_FADE_MS / 1000.0)
        np.testing.assert_array_equal(safe[: 2 * SR - fade],
                                      stem[: 2 * SR - fade])
        np.testing.assert_array_equal(safe[4 * SR + fade:],
                                      stem[4 * SR + fade:])

    def test_no_patch_when_the_stem_survived(self):
        # foreground control: stem ≈ mix everywhere — nothing to patch
        mix = sine(300, 4.0, level=0.3)
        stem = (mix * 0.9).astype(np.float32)
        safe, spans = ss.build_speech_safe(mix, stem, SR)
        self.assertEqual(spans, [])
        np.testing.assert_array_equal(safe, stem)

    def test_no_patch_where_the_mix_is_silent_too(self):
        # true silence in the SOURCE (scene gap) must stay silent — the
        # mix floor is what separates "deleted" from "nothing was there"
        mix = np.concatenate([sine(300, 2.0, level=0.3), silence(2.0)])
        stem = np.concatenate([mix[: 2 * SR], silence(2.0)])
        safe, spans = ss.build_speech_safe(mix, stem, SR)
        self.assertEqual(spans, [])
        np.testing.assert_array_equal(safe, stem)

    def test_quiet_speech_below_activity_but_above_floor_is_patched(self):
        # a whisper: -50 dBFS in the mix (0.0045 amplitude) — above the
        # -60 mix floor, and the stem deleted it
        whisper = sine(300, 2.0, level=0.0045)
        mix = np.concatenate([sine(300, 2.0, level=0.3), whisper])
        stem = np.concatenate([mix[: 2 * SR], silence(2.0)])
        safe, spans = ss.build_speech_safe(mix, stem, SR)
        self.assertEqual(len(spans), 1)
        self.assertGreater(rms_db(safe[int(2.5 * SR): int(3.5 * SR)]), -70.0)

    def test_short_gap_between_patches_is_bridged_against_pumping(self):
        # two deleted spans 150 ms apart (< merge gap) — one patch, no
        # per-word gate chatter
        mix = sine(300, 5.0, level=0.3)
        stem = mix.copy()
        stem[int(1.0 * SR): int(2.0 * SR)] = 0.0
        stem[int(2.15 * SR): int(3.0 * SR)] = 0.0
        _, spans = ss.build_speech_safe(mix, stem, SR)
        self.assertEqual(len(spans), 1)

    def test_crossfade_has_no_step_discontinuity(self):
        mix = sine(300, 6.0, level=0.3)
        stem = np.concatenate([mix[: 2 * SR], silence(2.0), mix[4 * SR:]])
        safe, _ = ss.build_speech_safe(mix, stem, SR)
        # max sample-to-sample jump stays within what the carrier sine
        # itself produces (a hard gate splice would far exceed it)
        max_step = float(np.abs(np.diff(safe.astype(np.float64))).max())
        carrier_step = float(np.abs(np.diff(mix.astype(np.float64))).max())
        self.assertLess(max_step, carrier_step * 1.5)

    def test_patch_running_into_the_chunk_edge_holds_full_level(self):
        # a deletion that continues across the chunk boundary: the gate
        # mask is edge-extended before smoothing, so the LAST sample of
        # the chunk is patched at full envelope — the neighbor chunk sees
        # the same audio and gates the same way, so the seam carries no dip
        mix = sine(300, 4.0, level=0.3)
        stem = np.concatenate([mix[: 2 * SR], silence(2.0)])
        safe, spans = ss.build_speech_safe(mix, stem, SR)
        self.assertEqual(len(spans), 1)
        tail = safe[-int(0.02 * SR):]
        expect_db = rms_db(mix[-int(0.02 * SR):]) + ss.SPEECH_SAFE_PATCH_GAIN_DB
        self.assertAlmostEqual(rms_db(tail), expect_db, delta=1.0)

    def test_output_is_clipped_and_float32(self):
        mix = (np.ones(SR, dtype=np.float32) * 0.99)
        stem = np.zeros(SR, dtype=np.float32)
        stem[: SR // 2] = 0.99          # loud stem half, deleted half
        safe, _ = ss.build_speech_safe(mix, stem, SR)
        self.assertEqual(safe.dtype, np.float32)
        self.assertLessEqual(float(np.abs(safe).max()), 1.0)

    def test_span_end_is_clamped_to_the_chunk_duration(self):
        # a partial tail bin must not report a patch past the audio —
        # patched_sec would exceed the chunk's duration_sec in metrics
        dur = 1.975                       # 39.5 bins at 50 ms
        mix = sine(300, dur, level=0.3)
        stem = silence(dur)
        _, spans = ss.build_speech_safe(mix, stem, SR)
        self.assertEqual(len(spans), 1)
        self.assertLessEqual(spans[0][1], int(round(dur * 1000)))

    def test_empty_input_degrades_to_empty(self):
        safe, spans = ss.build_speech_safe(
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), SR)
        self.assertEqual(safe.size, 0)
        self.assertEqual(spans, [])

    def test_the_effective_knobs_are_recorded(self):
        # a params block that omits a knob that shaped the output lies
        doc = ss.params_doc()
        self.assertEqual(set(doc), {
            "bin_ms", "stem_gate_dbfs", "mix_floor_dbfs",
            "patch_gain_db", "merge_gap_ms", "fade_ms"})


if __name__ == "__main__":
    unittest.main()
