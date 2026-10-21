"""
OFFLINE unit tests for the frame-indexing helpers in lib/video_command.py.
Like test_ad_mix_units.py this needs no S3 and no ffmpeg binary.

Run: python3 test/test_video_command_units.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.video_command import build_segment_boundaries, sec_to_frame  # noqa: E402



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


class TestSecToFrame(unittest.TestCase):
    def test_integer_fps(self):
        self.assertEqual(math.floor(sec_to_frame(8400.0, Fraction(25))), 210000)
        self.assertEqual(math.floor(sec_to_frame(0.0,    Fraction(25))), 0)
        self.assertEqual(math.floor(sec_to_frame(600.0,  Fraction(25))), 15000)

    def test_ntsc_keyframe_timestamp_rounds_to_its_own_frame(self):
        # ffprobe prints best_effort_timestamp_time to 6 dp, so frame 28753 at
        # 24000/1001 (pts 28781753) arrives as 1199.239708 — just short of exact.
        # This is why the preroll path rounds: floor would say 28752.
        fps = Fraction(24000, 1001)
        self.assertEqual(round(sec_to_frame(1199.239708, fps)), 28753)
        self.assertEqual(math.floor(sec_to_frame(1199.239708, fps)), 28752)
        self.assertEqual(round(sec_to_frame(3603.6, fps)), 86400)

    def test_matches_upstream_iterator_frame(self):
        # Regression pin for the chunk_transcode preroll bug: a QuickTime file
        # with media timescale 1/600 at 25 fps. ffmpeg 6.1's MOV demuxer reports
        # per-frame duration=1 instead of 24 ticks, so the old
        # int(pts / duration) gave 5040000 for this keyframe — 24x too high —
        # and preroll_frames went negative. Deriving the frame from the
        # timestamp is independent of the reported duration.
        fps                 = Fraction(25)
        nominal_start_frame = 210000                  # chunk 14, 8400 s in
        self.assertEqual(round(sec_to_frame(8400.0, fps)), nominal_start_frame)
        # A keyframe genuinely before the boundary yields a positive preroll —
        # 8399.2 s is 20 frames earlier. (The failing asset's kf sat exactly on
        # the boundary, so preroll there is 0.)
        early_kf = round(sec_to_frame(8399.2, fps))
        self.assertEqual(nominal_start_frame - early_kf, 20)

    def test_snap_absorbs_float_noise_but_not_real_offsets(self):
        # The snap exists to clean float64/decimal-print noise, NOT to swallow
        # genuine sub-frame offsets. A mark point 0.3 ms past a boundary at
        # 25 fps must ceil to the NEXT frame — the old fps.denominator*1001
        # bound (~0.5 ms at integer fps) snapped it back and clipped the frame
        # the floor/ceil contract promises to keep.
        fps = Fraction(25)
        self.assertEqual(math.ceil(sec_to_frame(600.0003, fps)), 15001)
        self.assertEqual(math.floor(sec_to_frame(599.9997, fps)), 14999)
        # float noise on an exact boundary still snaps away
        self.assertEqual(math.floor(sec_to_frame(600.0000000001, fps)), 15000)
        self.assertEqual(math.ceil(sec_to_frame(599.9999999999, fps)), 15000)


class TestSegmentBoundaries(unittest.TestCase):
    def test_start_frame_agrees_with_sec_to_frame(self):
        # The iterators' nominal_start_frame comes from here; chunk_transcode
        # recomputes keyframe positions with sec_to_frame. One scale, or the
        # difference between them is meaningless.
        fps = Fraction(25)
        first_start = build_segment_boundaries(0.0, 9000.0, fps)[0][0]
        self.assertEqual(first_start, math.floor(sec_to_frame(0.0, fps)))
        mark_in = build_segment_boundaries(8400.0, 9000.0, fps)[0][0]
        self.assertEqual(mark_in, math.floor(sec_to_frame(8400.0, fps)))

    def test_short_tail_merged(self):
        fps = Fraction(25)
        bounds = build_segment_boundaries(0.0, 610.0, fps)
        self.assertEqual(len(bounds), 1)
        self.assertEqual(bounds[0], (0, 15250))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestScaleDimensionUpscale(unittest.TestCase):
    """allow_upscale (2026-10-09): proxies keep the no-upscale default;
    the quick-edit mezzanine opts in. fit_dimensions is the fixed-canvas
    variant — no orientation swap."""

    def test_default_never_upscales(self):
        from lib.video_command import scale_dimension
        self.assertEqual(scale_dimension(1280, 720, 1920, 1080), (1280, 720))

    def test_allow_upscale_fills_the_target(self):
        from lib.video_command import scale_dimension
        self.assertEqual(scale_dimension(1280, 720, 1920, 1080,
                                         allow_upscale=True), (1920, 1080))

    def test_portrait_still_swaps_target_dims(self):
        from lib.video_command import scale_dimension
        self.assertEqual(scale_dimension(1080, 1920, 1920, 1080,
                                         allow_upscale=True), (1080, 1920))

    def test_fit_dimensions_never_swaps(self):
        from lib.video_command import fit_dimensions
        self.assertEqual(fit_dimensions(1080, 1920, 1920, 1080), (608, 1080))
