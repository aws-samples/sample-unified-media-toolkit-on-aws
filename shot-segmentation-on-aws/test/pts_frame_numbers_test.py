#!/usr/bin/env python3
"""
pts_frame_numbers_test.py — PTS -> source frame number reconstruction must
round to nearest, never truncate (README.md §9.8).

Run
---
  python -m unittest test.pts_frame_numbers_test -v     # from repo root
  python test/pts_frame_numbers_test.py

Why this exists
---------------
Containers store integer PTS on their own timebase. At time_base 1/90000 and
frame_rate 24000/1001 a frame is 3753.75 ticks, so every 4th frame's stored
PTS sits just below the exact frame boundary (frame 3 = 11261.25 -> stored
11261 -> reconstructed 2.99993...). `int()` truncation knocked those a full
frame down, duplicating one frame number and skipping the next — measured on
a real 720p 23.976 asset (the Meridian mislabel incident): 3597 duplicates +
3596 holes across 17238 frames, ~21% of all positions. Boundary detection
requires a gap==1 sample pair across a cut, so any cut landing on a
duplicated/holed position was undetectable — shot 16 fused >= 3 real shots,
and face-recognition downstream renamed an entire person.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "rft", os.path.join(ROOT, "states", "run_ffmpeg_transcode",
                            "lambda_function.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_ffreport(path, time_base, frame_rate, n_frames, muxer_rounding):
    """Synthesize showinfo lines exactly as a muxer would emit them: integer
    PTS on the container timebase. Real muxers differ in HOW they integerize
    a fractional tick position, so both directions are exercised:
    half-up (round to nearest) and floor (truncate toward zero)."""
    tpf = Fraction(1, 1) / (Fraction(time_base) * Fraction(frame_rate))
    with open(path, "w") as f:
        f.write(f"[Parsed_showinfo_0 @ 0x1] config in time_base: "
                f"{time_base}, frame_rate: {frame_rate}\n")
        for n in range(n_frames):
            exact = Fraction(n) * tpf
            if muxer_rounding == "half_up":
                pts = int(exact + Fraction(1, 2))
            else:  # floor
                pts = int(exact)
            pts_time = float(pts * Fraction(time_base))
            f.write(f"[Parsed_showinfo_0 @ 0x1] n: {n} pts: {pts} "
                    f"pts_time:{pts_time:.6f} duration: 0\n")


# The broadcast framerate grid (user requirement 2026-08-20):
# 23.976, 24, 25, 29.97, 30, 59.94, 60 — each against the mp4-standard
# 90 kHz timebase AND its native timebase, under both muxer rounding
# directions. Ticks-per-frame integrality at 90 kHz:
#   23.976 -> 3753.75 FRACTIONAL (the incident: every 4th frame corrupted)
#   24     -> 3750    integer
#   25     -> 3600    integer
#   29.97  -> 3003    integer
#   30     -> 3000    integer
#   59.94  -> 1501.5  FRACTIONAL (every other frame corrupted under a
#                     floor-rounding muxer)
#   60     -> 1500    integer
_RATES = ["24000/1001", "24/1", "25/1", "30000/1001", "30/1",
          "60000/1001", "60/1"]
_NATIVE_TB = {"24000/1001": "1/24000", "24/1": "1/24", "25/1": "1/25",
              "30000/1001": "1/30000", "30/1": "1/30",
              "60000/1001": "1/60000", "60/1": "1/60"}


class TestPtsFrameNumberReconstruction(unittest.TestCase):

    N = 500  # covers many corruption cycles (period 4 at 23.976, 2 at 59.94)

    def _reconstruct(self, time_base, frame_rate, n_frames, muxer_rounding):
        mod = _load_module()
        with tempfile.NamedTemporaryFile("w", suffix=".log",
                                         delete=False) as tf:
            path = tf.name
        try:
            _write_ffreport(path, time_base, frame_rate, n_frames,
                            muxer_rounding)
            _, _, sfn, _ = mod._parse_showinfo_log(
                log_path=path, frame_rate_hint=frame_rate,
                time_base_hint=time_base)
        finally:
            os.unlink(path)
        return sfn

    def _assert_grid(self, time_base_of):
        for rate in _RATES:
            tb = time_base_of(rate)
            for rounding in ("half_up", "floor"):
                with self.subTest(frame_rate=rate, time_base=tb,
                                  muxer_rounding=rounding):
                    sfn = self._reconstruct(tb, rate, self.N, rounding)
                    self.assertEqual(
                        list(range(self.N)), sfn,
                        f"non-contiguous reconstruction at {rate} on {tb} "
                        f"({rounding})")

    def test_90khz_timebase_all_rates_both_roundings(self):
        """The mp4-standard 90 kHz timebase across the full broadcast grid.
        Under int() truncation this fails at 23.976 (both muxer roundings)
        and 59.94 (floor-rounding muxer) — the exact incident class."""
        self._assert_grid(lambda rate: "1/90000")

    def test_native_timebase_all_rates_both_roundings(self):
        """Native timebases (1001 or 1 tick per frame — always integral):
        these were never at risk and must stay bit-identical."""
        self._assert_grid(lambda rate: _NATIVE_TB[rate])

    def test_the_incident_pattern_is_what_truncation_produces(self):
        """Documents the exact failure shape so a future reader can compare
        against a corrupt pts file: truncating the 90 kHz 23.976 sequence
        duplicates every 4th number and skips its successor."""
        from fractions import Fraction as F
        tpf = F(90000) * F(1001, 24000)
        truncated = [int(F(int(F(n) * tpf + F(1, 2))) / tpf)
                     for n in range(12)]
        self.assertEqual([0, 1, 2, 2, 4, 5, 6, 6, 8, 9, 10, 10], truncated)


if __name__ == "__main__":
    unittest.main()
