"""Golden-string tests for lib/edit_filters.py — the quick-edit builders."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.edit_filters import (  # noqa: E402
    black_source_args, build_audio_conform, build_fade_afilters,
    build_fade_vfilters, build_junction_filters, build_normalize_vfilters,
    crop_pixels, fit_dimensions, gain_filter, is_hdr,
    loudness_gain_filter, loudnorm_measure_filter,
)


class TestFit(unittest.TestCase):

    def test_downscale_uhd_to_1080p(self):
        self.assertEqual(fit_dimensions(3840, 2160, 1920, 1080), (1920, 1080))

    def test_upscale_720p_to_1080p(self):
        # unlike scale_dimension, the mezzanine fit DOES upscale
        self.assertEqual(fit_dimensions(1280, 720, 1920, 1080), (1920, 1080))

    def test_portrait_pillarboxes(self):
        ow, oh = fit_dimensions(1080, 1920, 1920, 1080)
        self.assertEqual((ow, oh), (608, 1080))  # 607.5 → even-aligned 608

    def test_never_exceeds_canvas_after_rounding(self):
        ow, oh = fit_dimensions(853, 480, 1920, 1080)
        self.assertLessEqual(ow, 1920)
        self.assertLessEqual(oh, 1080)


class TestNormalizeChain(unittest.TestCase):

    def test_plain_sdr_progressive_match(self):
        self.assertEqual(
            build_normalize_vfilters(1920, 1080, "progressive",
                                     1920, 1080, "25"),
            ["scale=1920:1080", "setsar=1", "fps=25", "format=yuv420p"])

    def test_interlaced_portrait_full_chain(self):
        chain = build_normalize_vfilters(1080, 1920, "tt",
                                         1920, 1080, "30000/1001")
        self.assertEqual(chain, [
            "yadif=deint=interlaced",
            "scale=608:1080",
            "pad=1920:1080:656:0",
            "setsar=1",
            "fps=30000/1001",
            "format=yuv420p"])

    def test_is_hdr_is_the_refusal_predicate(self):
        # v1 refuses HDR by name (no zscale in the LGPL build) — probe
        # calls this; the normalize chain never sees an HDR source
        self.assertTrue(is_hdr("smpte2084"))
        self.assertTrue(is_hdr("arib-std-b67"))
        self.assertFalse(is_hdr("bt709"))
        self.assertFalse(is_hdr(None))

    def test_normalized_crop_prepends_and_refits(self):
        # 9:16 reframe out of a 1080p source into a vertical canvas
        crop = {"x": 0.34, "y": 0.0, "width": 0.3125, "height": 1.0}
        chain = build_normalize_vfilters(1920, 1080, "progressive",
                                         1080, 1920, "25", crop=crop)
        self.assertEqual(chain[0], "crop=600:1080:652:0")  # x even-aligned
        # 600:1080 is 5:9, not quite 9:16 — never-stretch pillarboxes
        # the 7 px difference rather than distorting
        self.assertIn("scale=1066:1920", chain)
        self.assertIn("pad=1080:1920:7:0", chain)

    def test_crop_pixels_even_aligned_and_clamped(self):
        # ratios that round to odd pixels get even-aligned, never off-frame
        w, h, x, y = crop_pixels({"x": 0.999, "y": 0.0,
                                  "width": 0.001, "height": 1.0}, 1920, 1080)
        self.assertEqual((w % 2, h % 2, x % 2, y % 2), (0, 0, 0, 0))
        self.assertLessEqual(x + w, 1920)
        self.assertLessEqual(y + h, 1080)

    def test_black_source_args(self):
        self.assertEqual(
            black_source_args(1920, 1080, "25", 40),
            ["-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=25",
             "-frames:v", "40"])


class TestAudioConform(unittest.TestCase):
    """The house chain from probe_helper.build_afilters: async
    resample, level passes, apad (silence-pad so sample trims always
    reach their end), s16 stereo."""

    def test_plain_conform(self):
        self.assertEqual(
            build_audio_conform(),
            "aresample=48000:async=1,apad,"
            "aformat=sample_fmts=s16:channel_layouts=stereo")

    def test_levels_sit_between_resample_and_pad(self):
        s = build_audio_conform(loudnorm="loudnorm=X", gain="volume=-15dB")
        self.assertEqual(
            s, "aresample=48000:async=1,loudnorm=X,volume=-15dB,apad,"
               "aformat=sample_fmts=s16:channel_layouts=stereo")


class TestFadesAndJunctions(unittest.TestCase):

    def test_frame_exact_video_fades(self):
        self.assertEqual(build_fade_vfilters(13, 20, 200),
                         ["fade=t=in:s=0:n=13", "fade=t=out:s=180:n=20"])
        self.assertEqual(build_fade_vfilters(0, 0, 200), [])

    def test_micro_fade_is_the_floor_even_on_cuts(self):
        self.assertEqual(build_fade_afilters(0, 0, 8.0),
                         ["afade=t=in:st=0:d=0.02",
                          "afade=t=out:st=7.98:d=0.02"])

    def test_fade_envelope_above_the_floor(self):
        self.assertEqual(build_fade_afilters(0.5, 0.5, 8.0),
                         ["afade=t=in:st=0:d=0.5",
                          "afade=t=out:st=7.5:d=0.5"])

    def test_junction_blends_over_the_full_d_at_offset_zero(self):
        vf, af = build_junction_filters(25, 25.0)
        self.assertEqual(vf, "xfade=transition=fade:duration=1:offset=0")
        self.assertEqual(af, "acrossfade=d=1")


class TestLoudnessAndGain(unittest.TestCase):

    def test_gain_zero_is_noop(self):
        self.assertIsNone(gain_filter(0))
        self.assertEqual(gain_filter(-15), "volume=-15dB")

    def test_measure_pass(self):
        self.assertEqual(loudnorm_measure_filter(),
                         "loudnorm=I=-23:TP=-1.5:LRA=11:print_format=json")

    def test_apply_pass_is_a_plain_volume_gain(self):
        # NEVER loudnorm for application: its linear mode silently
        # falls back to dynamic (192 kHz out) when the gain exceeds TP,
        # quartering sample-based atrims (measured live 2026-10-10)
        measured = {"input_i": "-19.5"}
        self.assertEqual(loudness_gain_filter(measured),
                         "volume=-3.50dB")

    def test_gain_is_clamped(self):
        self.assertEqual(loudness_gain_filter({"input_i": "-80.0"}),
                         "volume=30.00dB")


if __name__ == "__main__":
    unittest.main()
