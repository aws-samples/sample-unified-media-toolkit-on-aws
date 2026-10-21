"""
Offline tests for the frame-capture companion: lib/frame_capture.py
(pure), the probe branch (faked AWS), the capture renderer (command
text), and merge's capture_manifest branch (faked helpers).
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.cut_list import CutListError  # noqa: E402
from lib.frame_capture import batch_markers, validate_frame_capture  # noqa: E402
from states.chunk_transcode.capture_render import build_capture_vfilters  # noqa: E402
from states.probe_media_file import frame_capture_probe as fcp  # noqa: E402


def marker(src="s3://bkt/a.mp4", at_ms=1000, **kw):
    return {"source_s3uri": src, "at_ms": at_ms, **kw}


class TestValidation(unittest.TestCase):

    def refuse(self, fc, needle):
        with self.assertRaises(CutListError) as cm:
            validate_frame_capture(fc)
        self.assertIn(needle, str(cm.exception))

    def test_defaults_and_indexing(self):
        norm = validate_frame_capture({"markers": [marker(), marker(at_ms=0)]})
        self.assertEqual(norm["format"], "jpeg")
        self.assertIsNone(norm["resize"])
        self.assertEqual([m["index"] for m in norm["markers"]], [0, 1])

    def test_marker_cap(self):
        self.refuse({"markers": [marker(at_ms=i) for i in range(501)]},
                    "500-marker cap")

    def test_source_cap(self):
        self.refuse({"markers": [marker(src=f"s3://bkt/{i}.mp4")
                                 for i in range(26)]}, "25-source cap")

    def test_bad_format_and_resize(self):
        self.refuse({"markers": [marker()], "format": "gif"}, "format")
        self.refuse({"markers": [marker()], "resize": "720p"},
                    "WIDTHxHEIGHT")
        self.refuse({"markers": [marker()], "resize": "1079x1920"},
                    "must be even")

    def test_crop_shares_the_quick_edit_validator(self):
        self.refuse({"markers": [marker(crop={"x": 0.5, "y": 0,
                                              "width": 0.6, "height": 1})]},
                    "beyond the frame")

    def test_batching_groups_by_source_then_splits(self):
        markers = [dict(marker(src="s3://bkt/a.mp4", at_ms=i), index=i)
                   for i in range(150)]
        markers += [dict(marker(src="s3://bkt/b.mp4", at_ms=i), index=200 + i)
                    for i in range(10)]
        batches = batch_markers(markers)
        self.assertEqual([len(b) for b in batches], [100, 50, 10])
        self.assertTrue(all(len({m["source_s3uri"] for m in b}) == 1
                            for b in batches))

    def test_batching_splits_on_span_and_sorts(self):
        # markers 600 s apart in one source land in SEPARATE
        # iterations: one ffmpeg pass per batch reads the whole span
        markers = [dict(marker(at_ms=1201000), index=0),
                   dict(marker(at_ms=600500), index=1),
                   dict(marker(at_ms=600900), index=2)]
        batches = batch_markers(markers)
        self.assertEqual([[m["index"] for m in b] for b in batches],
                         [[1, 2], [0]])   # sorted, span-split


class TestCaptureFilters(unittest.TestCase):

    SRC = {"width": 1920, "height": 1080, "field_order": "progressive"}

    def test_native_no_filters(self):
        self.assertEqual(build_capture_vfilters(self.SRC, None, None), [])

    def test_crop_then_resize_fit(self):
        vf = build_capture_vfilters(
            self.SRC, {"x": 0.34, "y": 0.0, "width": 0.3125, "height": 1.0},
            (1080, 1920))
        self.assertEqual(vf[0], "crop=600:1080:652:0")
        self.assertEqual(vf[1], "scale=1066:1920")
        self.assertEqual(vf[2], "pad=1080:1920:7:0")

    def test_yadif_is_not_in_the_branch_chain(self):
        # deint lives pre-split in build_batch_command, never after
        # the per-branch trim
        src = dict(self.SRC, field_order="tt")
        self.assertEqual(build_capture_vfilters(src, None, None), [])


class TestBatchCommand(unittest.TestCase):
    # One ffmpeg per batch: one seek, split, per-branch select by COUNT
    # (the amortization that fixed the 900 s Lambda overrun)

    IT = {"source": {"bucket": "bkt", "key": "a.mxf", "fps": "25/1",
                     "field_order": "tt", "width": 1920, "height": 1080,
                     "is_all_intra": False},
          "format": "jpeg", "resize": None,
          "markers": [
              {"index": 3, "source_s3uri": "s3://bkt/a.mxf",
               "at_ms": 600500, "frame": 15013, "actual_ms": 600520.0},
              {"index": 5, "source_s3uri": "s3://bkt/a.mxf",
               "at_ms": 600900, "frame": 15023, "actual_ms": 600920.0},
          ]}

    def test_one_invocation_counts_from_the_keyframe(self):
        from states.chunk_transcode.capture_render import (
            batch_seek_plan, build_batch_command)
        with mock.patch("states.chunk_transcode.capture_render."
                        "find_prior_keyframe",
                        return_value={"best_effort_timestamp_time":
                                      "600.200000"}) as kf:
            seek, base = batch_seek_plan(self.IT["source"], 15013)
        kf.assert_called_once()          # ONE scan for the whole batch
        self.assertEqual((seek, base), (600.2, 15005))
        cmd, outs = build_batch_command(self.IT, "https://u", seek, base,
                                        "/tmp/x")
        fc = cmd[cmd.index("-filter_complex") + 1]
        # deint ONCE, pre-split; frames picked by TRIM COUNT (the house
        # frame-exact mechanism) — select is BANNED here (not
        # frame-exact, never EOFs its branch)
        self.assertIn("[0:v]yadif=deint=interlaced,split=2[c0][c1]", fc)
        self.assertIn("trim=start_frame=8:end_frame=9,setpts=PTS-STARTPTS",
                      fc)                          # 15013 - 15005
        self.assertIn("trim=start_frame=18:end_frame=19", fc)
        self.assertNotIn("select=", fc)
        self.assertEqual(cmd.count("-frames:v"), 2)
        self.assertEqual([o["s3_name"] for o in outs],
                         ["frame_0003.jpg", "frame_0005.jpg"])

    def test_all_intra_seeks_the_first_frame_directly(self):
        from states.chunk_transcode.capture_render import batch_seek_plan
        src = dict(self.IT["source"], is_all_intra=True)
        seek, base = batch_seek_plan(src, 15013)
        self.assertEqual(base, 15013)
        self.assertAlmostEqual(seek, 600.52)


class TestProbeBranch(unittest.TestCase):

    def facts(self, **over):
        base = {"bucket": "bkt", "key": "a.mp4", "etag": "x",
                "duration_sec": 600.0, "has_video": True, "has_audio": True,
                "width": 1920, "height": 1080, "fps": "25/1",
                "field_order": "progressive", "color_transfer": "bt709",
                "is_all_intra": True}
        base.update(over)
        return base

    def run_probe(self, fc, **facts_over):
        with mock.patch.object(fcp, "_probe_source",
                               return_value=self.facts(**facts_over)), \
             mock.patch.object(fcp, "put_json") as pj:
            result = fcp.handle_frame_capture(
                {"input": {"frame_capture": fc}, "data": {}}, fc,
                uuid="u1", out_bucket="proxy", out_prefix="p",
                output_groups_req=None, fn_arn="arn:fn")
        return result, pj

    def test_snap_to_source_grid(self):
        fc = {"markers": [marker(at_ms=1030)]}   # 25.75 fr @25 → frame 26
        result, _ = self.run_probe(fc)
        m = result["data"]["iterators"][0]["markers"][0]
        self.assertEqual(m["frame"], 26)
        self.assertEqual(m["actual_ms"], 1040.0)

    def test_hdr_refused(self):
        with self.assertRaises(ValueError) as cm:
            self.run_probe({"markers": [marker()]},
                           color_transfer="smpte2084")
        self.assertIn("HDR requires tonemapping", str(cm.exception))

    def test_marker_at_exact_end_refused_at_probe(self):
        # at_ms == duration passes the ms check but ROUNDS to a frame
        # index past the last frame — refused here by name, not after a
        # full sequential render read discovers an empty trim
        with self.assertRaises(ValueError) as cm:
            self.run_probe({"markers": [marker(at_ms=600000)]},
                           duration_sec=600.0)
        self.assertIn("past the source's last frame", str(cm.exception))

    def test_marker_beyond_duration(self):
        with self.assertRaises(ValueError) as cm:
            self.run_probe({"markers": [marker(at_ms=700000)]},
                           duration_sec=600.0)
        self.assertIn("beyond the source's", str(cm.exception))

    def test_iterator_shape(self):
        fc = {"markers": [marker(), marker(at_ms=2000)], "format": "png"}
        result, _ = self.run_probe(fc)
        it = result["data"]["iterators"][0]
        self.assertEqual(it["type"], "capture")
        self.assertEqual(it["format"], "png")
        self.assertEqual(it["source"]["bucket"], "bkt")
        self.assertEqual(len(it["markers"]), 2)
        self.assertEqual(result["data"]["output_groups"][0]["format"], "png")


class TestMergeCaptureBranch(unittest.TestCase):

    def test_capture_manifest_written_sorted(self):
        from states.merge_chunks import lambda_function as merge
        event = {
            "uuid": "u1",
            "data": {
                "destination": {"bucket": "proxy", "prefix": "p"},
                "output_groups": [{"group_name": "frames", "format": "jpeg",
                                   "name": "jpeg"}],
                "output_keys": {},
                "iterators": [
                    {"item_id": 1, "outputs": [
                        {"s3_prefix": "frames", "s3_key": "p/frames/frame_0001.jpg",
                         "capture": {"index": 1, "requested_ms": 2000,
                                     "actual_ms": 2000.0, "frame": 50,
                                     "source_s3uri": "s3://bkt/a.mp4"}}]},
                    {"item_id": 0, "outputs": [
                        {"s3_prefix": "frames", "s3_key": "p/frames/frame_0000.jpg",
                         "capture": {"index": 0, "requested_ms": 1000,
                                     "actual_ms": 1000.0, "frame": 25,
                                     "source_s3uri": "s3://bkt/a.mp4"}}]},
                ],
            },
        }
        with mock.patch.object(merge, "put_json") as pj:
            result = merge.lambda_handler(event, None)
        manifest = pj.call_args.args[2]
        self.assertEqual(manifest["total"], 2)
        self.assertEqual([c["index"] for c in manifest["captures"]], [0, 1])
        self.assertEqual(manifest["captures"][0]["s3_key"],
                         "p/frames/frame_0000.jpg")
        self.assertEqual(result["data"]["output_keys"]["frames"],
                         "p/frames/capture_manifest.json")


class TestFinalMp4FastStart(unittest.TestCase):
    """The deliverable MP4 must allow fast-start (moov up front) —
    user requirement 2026-10-09; quick-edit output rides merge_mp4, so
    this pin covers proxy and edit alike."""

    def test_merge_mp4_passes_faststart(self):
        from states.merge_chunks import merge_helper
        captured = {}
        with mock.patch.object(merge_helper, "_run",
                               side_effect=lambda cmd, **kw:
                               captured.setdefault("cmd", cmd)), \
             mock.patch.object(merge_helper, "upload_file"), \
             mock.patch.object(merge_helper.os, "unlink"), \
             mock.patch.object(merge_helper, "_presign_keys",
                               side_effect=lambda b, k: ["u"] * len(k)):
            merge_helper.merge_mp4("bkt", ["v0.ts"], ["a0.flac"],
                                   "/tmp", "p/edit/out.mp4")
        cmd = captured["cmd"]
        i = cmd.index("-movflags")
        self.assertEqual(cmd[i + 1], "+faststart")


if __name__ == "__main__":
    unittest.main()
