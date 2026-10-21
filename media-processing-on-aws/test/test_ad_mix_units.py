"""
OFFLINE unit tests for the ad_mix output group — unlike this stack's
other test_*.py (live CLI scripts needing real S3), this one runs
anywhere: it covers only the pure pieces (schema defaults/validation and
the ffmpeg command builders).

Run: python3 test/test_ad_mix_units.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import ad_mix                          # noqa: E402
from lib.input_schema import parse_output_groups  # noqa: E402



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


class TestSchema(unittest.TestCase):
    def test_ad_mix_requires_manifest_uri(self):
        with self.assertRaises(ValueError):
            parse_output_groups(
                [{"group_name": "admix", "format": "ad_mix"}],
                has_video=False, has_audio=True)

    def test_ad_mix_defaults(self):
        groups = parse_output_groups(
            [{"group_name": "admix", "format": "ad_mix",
              "render_manifest_s3uri": "s3://b/render_manifest.json"}],
            has_video=False, has_audio=True)
        g = groups[0]
        self.assertEqual((g["audio_channels"], g["codec"]),
                         (2, "pcm_s16le"))
        # sampling_rate deliberately NOT stamped: the render manifest's
        # sample_rate is the fallback at merge, and a stamped default
        # made it dead code — only an explicit caller value survives
        self.assertNotIn("sampling_rate", g)
        g2 = parse_output_groups(
            [{"group_name": "admix", "format": "ad_mix",
              "sampling_rate": 44100,
              "render_manifest_s3uri": "s3://b/m.json"}],
            has_video=False, has_audio=True)[0]
        self.assertEqual(g2["sampling_rate"], 44100)

    def test_ad_mix_dropped_without_audio_stream(self):
        groups = parse_output_groups(
            [{"group_name": "admix", "format": "ad_mix",
              "render_manifest_s3uri": "s3://b/m.json"},
             {"group_name": "meta", "format": "metadata"}],
            has_video=True, has_audio=False)
        self.assertEqual([g["format"] for g in groups], ["metadata"])


class TestExistingFormatsUnchanged(unittest.TestCase):
    """REGRESSION PINS for the formats every other workflow depends on
    (face-recognition invokes this stack; agents proxy/extract through
    it). Added 2026-08-31 with ad_mix: these lock the reconciled group
    shapes that existed BEFORE the feature, so any future change that
    moves them fails here first. ad_mix's design guarantee is that a
    request containing no ad_mix group reconciles byte-identically to
    the pre-feature code (FLAC_BACKED_FORMATS membership is equivalent
    to AUDIO_FORMATS for every non-ad_mix format)."""

    def test_default_mp4_proxy_group_when_groups_omitted(self):
        groups = parse_output_groups(None, has_video=True, has_audio=True)
        g = groups[0]
        self.assertEqual((g["group_name"], g["format"], g["name"],
                          g["codec"], g["gop_seconds"],
                          g["motion_factor"]),
                         ("proxy", "mp4", "transcoded", "libopenh264",
                          2.0, 1.6))

    def test_wav_and_m4a_defaults_are_untouched(self):
        groups = parse_output_groups(
            [{"group_name": "audio_only", "format": "wav"},
             {"group_name": "audio_m4a", "format": "m4a"}],
            has_video=False, has_audio=True)
        wav, m4a = groups
        self.assertEqual((wav["sampling_rate"], wav["audio_channels"],
                          wav["codec"]), (48000, 1, "pcm_s16le"))
        self.assertEqual((m4a["sampling_rate"], m4a["audio_channels"],
                          m4a["codec"], m4a["audio_bitrate"]),
                         (48000, 2, "aac", 96))

    def test_frame_and_metadata_defaults_are_untouched(self):
        groups = parse_output_groups(
            [{"group_name": "fc", "format": "jpeg"},
             {"group_name": "meta", "format": "metadata"}],
            has_video=True, has_audio=False)
        jpeg, meta = groups
        self.assertEqual((jpeg["frame_format"],
                          jpeg["frame_interval_sec"]), ("jpeg", 1.0))
        self.assertEqual(meta["detail"], "default")

    def test_incompatible_groups_still_drop_with_warning_not_error(self):
        groups = parse_output_groups(
            [{"group_name": "proxy", "format": "mp4"},
             {"group_name": "audio", "format": "wav"}],
            has_video=True, has_audio=False)
        self.assertEqual([g["format"] for g in groups], ["mp4"])

    def test_unknown_format_still_raises(self):
        with self.assertRaises(ValueError):
            parse_output_groups(
                [{"group_name": "x", "format": "flac"}],
                has_video=True, has_audio=True)

    def test_flac_backed_is_equivalent_to_audio_for_existing_formats(self):
        """The ONLY shared-path change probe saw: AUDIO_FORMATS ->
        FLAC_BACKED_FORMATS at two membership tests. For every format
        that is not ad_mix, the two sets answer identically."""
        from lib.input_schema import (AUDIO_FORMATS, FLAC_BACKED_FORMATS,
                                      VALID_FORMATS)
        for fmt in VALID_FORMATS - {"ad_mix"}:
            self.assertEqual(fmt in AUDIO_FORMATS,
                             fmt in FLAC_BACKED_FORMATS, fmt)


class TestCommands(unittest.TestCase):
    def test_take_render_appends_atempo_only_when_fitted(self):
        c1 = ad_mix.take_render_cmd("in.wav", "out.wav", "loudnorm",
                                    1.0, 48000)
        self.assertIn("loudnorm", c1[c1.index("-af") + 1])
        self.assertNotIn("atempo", c1[c1.index("-af") + 1])
        c2 = ad_mix.take_render_cmd("in.wav", "out.wav", "loudnorm",
                                    1.13, 48000)
        self.assertIn("atempo=1.1300", c2[c2.index("-af") + 1])

    def test_bus_mix_sums_without_normalizing(self):
        cmd = ad_mix.bus_mix_cmd(["a.wav", "b.wav"], [1.5, 10.0],
                                 "bus.wav", 48000)
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("adelay=1500:all=1", fc)
        self.assertIn("adelay=10000:all=1", fc)
        # normalize=0 is load-bearing: amix's default AVERAGES, ducking
        # every non-overlapping take by 1/N
        self.assertIn("amix=inputs=2:normalize=0", fc)

    def test_duck_mix_pins_duration_to_the_bed(self):
        cmd = ad_mix.duck_mix_cmd("bed.txt", "bus.wav", "out.wav",
                                  48000, 2)
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("sidechaincompress", fc)
        self.assertIn("duration=first", fc)
        self.assertIn("loudnorm=I=-24", fc)
        # apad on the bus is load-bearing: sidechaincompress and amix
        # end at their SHORTEST input, and the bus ends at the LAST
        # take — unpadded, everything after it was silently lost
        self.assertIn("[1:a]apad", fc)

    def test_concat_inputs_whitelist_presigned_https(self):
        """The bed list holds presigned https URLs; the concat demuxer
        refuses nested https opens unless whitelisted (merge_helper
        passes the same flag for the same pattern)."""
        cmd = ad_mix.duck_mix_cmd("bed.txt", "bus.wav", "out.wav",
                                  48000, 2)
        i = cmd.index("-protocol_whitelist")
        self.assertIn("https", cmd[i + 1])
        self.assertLess(i, cmd.index("-i"))

    def test_batches_split_only_past_the_cap(self):
        self.assertEqual(ad_mix.plan_bus_batches(3, cap=150), [(0, 3)])
        self.assertEqual(ad_mix.plan_bus_batches(310, cap=150),
                         [(0, 150), (150, 300), (300, 310)])


if __name__ == "__main__":
    unittest.main()
