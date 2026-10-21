"""_derive_timecodes: every shot timecode is derived at merge time.

Timecodes in this pipeline are pure arithmetic (frame_to_smpte over the
container's first-frame origin), so since 2026-08-25 the windows ship no
per-frame strings and the merge state derives start/end timecodes — and the
sync_pop/flash_frame anomaly record's timecode — from frame numbers in one
pass just before shots.json is serialized.

Pins:
  1. derivation uses the same drop-frame arithmetic as the sampled frames,
     for every shot including post-hoc split boundaries;
  2. the anomaly records get their timecode from their own frame;
  3. a source with no timecode track derives from 00:00:00:00 NDF via the
     None origin; a missing fps leaves empty strings, never crashes.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "states", "merge_reconcile_shots"))

from lib.misc_utils import frame_to_smpte                    # noqa: E402
from lambda_function import _derive_timecodes                # noqa: E402

FPS = "30000/1001"
ORIGIN = "00:00:00;02"   # drop-frame origin, as on real 29.97 broadcast masters


class TestDeriveTimecodes(unittest.TestCase):

    def test_every_shot_derived_from_frames(self):
        shots = [
            {"shot_id": 0, "start_frame": 0, "end_frame": 359},
            {"shot_id": 1, "start_frame": 360, "end_frame": 424},   # split boundary
        ]
        _derive_timecodes(shots, FPS, True, ORIGIN)
        for sh in shots:
            self.assertEqual(frame_to_smpte(sh["start_frame"], FPS, True, ORIGIN),
                             sh["start_timecode"])
            self.assertEqual(frame_to_smpte(sh["end_frame"], FPS, True, ORIGIN),
                             sh["end_timecode"])
        self.assertIn(";", shots[1]["start_timecode"])   # drop-frame notation

    def test_anomaly_records_derive_from_their_own_frame(self):
        shots = [{"shot_id": 0, "start_frame": 100, "end_frame": 300,
                  "sync_pop": {"frame": 200, "time": 6.673}}]
        _derive_timecodes(shots, FPS, True, ORIGIN)
        self.assertEqual(frame_to_smpte(200, FPS, True, ORIGIN),
                         shots[0]["sync_pop"]["timecode"])

    def test_no_timecode_track_counts_from_zero_ndf(self):
        shots = [{"shot_id": 0, "start_frame": 0, "end_frame": 1500}]
        _derive_timecodes(shots, "25/1", False, None)
        self.assertEqual("00:00:00:00", shots[0]["start_timecode"])
        self.assertEqual("00:01:00:00", shots[0]["end_timecode"])

    def test_origin_is_preserved_frame_accurately(self):
        # origin 09:00:00;01 → frame 1 must read 09:00:00;02
        shots = [{"shot_id": 0, "start_frame": 1, "end_frame": 2}]
        _derive_timecodes(shots, FPS, True, "09:00:00;01")
        self.assertEqual("09:00:00;02", shots[0]["start_timecode"])
        self.assertEqual("09:00:00;03", shots[0]["end_timecode"])

    def test_no_fps_leaves_empty_strings_not_crash(self):
        shots = [{"shot_id": 0, "start_frame": 0, "end_frame": 10}]
        _derive_timecodes(shots, None, False, None)
        self.assertEqual("", shots[0]["start_timecode"])
        self.assertEqual("", shots[0]["end_timecode"])


class TestWindowPayloadShipsNoTimecodes(unittest.TestCase):
    """The extract state's _write_outputs must run with the post-refactor
    call chain and emit a window payload WITHOUT a per-frame timecode list.
    This pins the exact gap that shipped a NameError to the deployed stack
    on 2026-08-25: the timecodes variable was removed but one positional
    call-site survived, and no test exercised the persist path."""

    def test_write_outputs_runs_and_omits_timecode_key(self):
        import numpy as np
        from states.run_dino_v2_model import lambda_function as extract

        captured = {}
        orig_bin, orig_json = extract.put_binary, extract.put_json
        extract.put_binary = lambda bucket, key, data, mime: None
        extract.put_json = lambda bucket, key, payload: captured.update(payload)
        try:
            item = {"window_name": "seg0000_win0000", "window_id": 0,
                    "segment_id": 0, "segment_name": "segment_0000",
                    "window_start": 0.0, "window_end": 360.0,
                    "content_end": 360.0, "source_start_frame": 0,
                    "source_end_frame": 100, "frame_rate": "30000/1001",
                    "frames_to_shots_cfg": {}}
            extract._write_outputs(
                np.zeros((3, 384), dtype=np.float32),      # embeddings
                [0, 8, 15],                                # frame_index
                ["content"] * 3,                           # frame_types
                [0.0, 0.267, 0.5],                         # source_times
                [],                                        # black_frames
                [True] * 3,                                # is_coarse
                [4.0] * 3,                                 # entropy
                [{"shot_id": 0, "start_frame": 0, "end_frame": 15}],
                item, "test-bucket", "test/prefix")
        finally:
            extract.put_binary, extract.put_json = orig_bin, orig_json

        self.assertIn("frame_index", captured)
        self.assertNotIn("timecode", captured)


class TestDropFrameArithmetic(unittest.TestCase):
    """frames_to_df_smpte drops labels ;00..;01 at the start of every minute
    EXCEPT minutes divisible by 10 (SMPTE 12M, 29.97). The pre-2026-08-25
    implementation omitted the +d label offset inside dropping minutes, so
    every timecode in 54 of every 60 minutes was two frames early and minute
    starts carried invalid labels (00:01:00;00). Vectors below include the
    V5-verified case: V5's independent implementation labels source frame
    1800 under a 00:00:00;02 origin as 00:01:00;04."""

    VECTORS = [
        (0,      "00:00:00;00"),
        (1799,   "00:00:59;29"),
        (1800,   "00:01:00;02"),   # first frame of a dropping minute
        (3597,   "00:01:59;29"),   # last frame of a dropping minute
        (3598,   "00:02:00;02"),
        (17982,  "00:10:00;00"),   # minute 10: no drop
        (107892, "01:00:00;00"),   # exactly one hour
    ]

    def test_known_drop_frame_vectors(self):
        from lib.misc_utils import frames_to_df_smpte
        for frame, expect in self.VECTORS:
            self.assertEqual(expect, frames_to_df_smpte(frame, 30),
                             f"frame {frame}")

    def test_roundtrip_identity_over_two_hours(self):
        from lib.misc_utils import frames_to_df_smpte
        from lib.misc_utils import smpte_to_frames
        for frame in range(0, 216000, 7):   # every 7th frame across 2 h
            tc = frames_to_df_smpte(frame, 30)
            self.assertEqual(frame, smpte_to_frames(tc, "30000/1001", True), tc)

    def test_v5_cross_verified_origin_case(self):
        from lib.misc_utils import frame_to_smpte
        self.assertEqual("00:01:00;04",
                         frame_to_smpte(1800, "30000/1001", True, "00:00:00;02"))


class TestAllFrameRates(unittest.TestCase):
    """SMPTE arithmetic across every rate the pipeline meets. Drop-frame
    exists ONLY at 30/60-rounding (29.97, 59.94); 23.976 is NDF-only by the
    spec, and integer rates are NDF. A container that (invalidly) claims DF
    at any other rounding falls through to NDF counting."""

    def test_5994_drop_frame_vectors(self):
        from lib.misc_utils import frame_to_smpte
        # d=4: dropping minutes hold 3596 frames; minute 1 = frames 3600..7195
        for frame, expect in [
            (0,      "00:00:00;00"),
            (3599,   "00:00:59;59"),
            (3600,   "00:01:00;04"),   # labels ;00-;03 skipped
            (7195,   "00:01:59;59"),   # last frame of the dropping minute
            (7196,   "00:02:00;04"),
            (35964,  "00:10:00;00"),   # minute 10: no drop
            (215784, "01:00:00;00"),   # exactly one wall-clock hour
        ]:
            self.assertEqual(expect,
                             frame_to_smpte(frame, "60000/1001", True, None),
                             f"frame {frame}")

    def test_ndf_integer_rates(self):
        from lib.misc_utils import frame_to_smpte
        for fps_str, fps in [("24/1", 24), ("25/1", 25), ("30/1", 30),
                             ("50/1", 50), ("60/1", 60), ("24000/1001", 24)]:
            self.assertEqual("00:01:00:00",
                             frame_to_smpte(fps * 60, fps_str, False, None),
                             fps_str)
            self.assertEqual("01:00:00:00",
                             frame_to_smpte(fps * 3600, fps_str, False, None),
                             fps_str)

    def test_bogus_df_flag_outside_30_60_falls_to_ndf(self):
        from lib.misc_utils import frame_to_smpte
        self.assertEqual("00:01:00:00",
                         frame_to_smpte(1440, "24000/1001", True, None))
        self.assertEqual("00:01:00:00",
                         frame_to_smpte(1500, "25/1", True, None))

    def test_roundtrip_identity_every_rate(self):
        from lib.misc_utils import frame_to_smpte, smpte_to_frames
        for fps_str, df, n in [("30000/1001", True, 216000),
                               ("60000/1001", True, 432000),
                               ("24000/1001", False, 173000),
                               ("25/1", False, 180000),
                               ("50/1", False, 360000),
                               ("60/1", False, 432000)]:
            for frame in range(0, n, 101):
                tc = frame_to_smpte(frame, fps_str, df, None)
                self.assertEqual(frame, smpte_to_frames(tc, fps_str, df),
                                 f"{fps_str} {tc}")

    def test_origin_identity_across_rates(self):
        from lib.misc_utils import frame_to_smpte
        for fps_str, df, origin in [("30000/1001", True, "14:00:13;10"),
                                    ("60000/1001", True, "09:59:30;42"),
                                    ("25/1", False, "10:00:00:00")]:
            self.assertEqual(origin, frame_to_smpte(0, fps_str, df, origin))
if __name__ == "__main__":
    unittest.main(verbosity=2)
