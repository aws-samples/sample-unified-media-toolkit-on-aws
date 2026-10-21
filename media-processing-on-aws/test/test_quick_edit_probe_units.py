"""
Offline tests for states/probe_media_file/quick_edit_probe.py — fakes
for S3/ffprobe/ffmpeg; the real cut_list/edit_filters logic runs.
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from states.probe_media_file import quick_edit_probe as qep  # noqa: E402

LOUDNORM_JSON = json.dumps({
    "input_i": "-19.5", "input_tp": "-0.3", "input_lra": "6.2",
    "input_thresh": "-30.1", "target_offset": "0.4"})


def fake_probe_for(sources):
    """sources: uri → dict(duration, video?, audio?, w, h, fps, transfer)."""
    def _run_ffprobe(bucket, key):
        uri = f"s3://{bucket}/{key}"
        f = sources[uri]
        streams = []
        if f.get("video", True):
            streams.append({"codec_type": "video",
                            "width": f.get("w", 1920),
                            "height": f.get("h", 1080),
                            "r_frame_rate": f.get("fps", "25/1"),
                            "field_order": "progressive",
                            "color_transfer": f.get("transfer", "bt709")})
        if f.get("audio", True):
            streams.append({"codec_type": "audio"})
        return {"streams": streams,
                "format": {"duration": str(f.get("duration", 600.0))}}
    return _run_ffprobe


def patched(sources, loudnorm_text=None):
    return mock.patch.multiple(
        qep,
        head_object=mock.Mock(return_value={"ETag": '"abc"'}),
        archive_tier=mock.Mock(return_value=None),
        run_ffprobe=fake_probe_for(sources),
        presign=mock.Mock(return_value="https://signed"),
        run=mock.Mock(return_value=loudnorm_text or
                      f"...measuring...\n{LOUDNORM_JSON}\n"),
        put_json=mock.DEFAULT,
    )


BASE_QE = {
    "segments": [
        {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 0, "out_ms": 8000},
        {"source_s3uri": "s3://bkt/b.mp4", "in_ms": 1000, "out_ms": 7000,
         "transition_in": {"type": "crossfade", "duration_ms": 1000}},
    ],
    "target": {"resolution": "1920x1080", "fps": "25"},
}
TWO_SDR = {"s3://bkt/a.mp4": {}, "s3://bkt/b.mp4": {}}


def call(qe, sources, **kw):
    with patched(sources, **kw) as mocks:
        result = qep.handle_quick_edit(
            {"input": {"quick_edit": qe}, "data": {}}, qe, uuid="u1",
            out_bucket="proxy", out_prefix="micro-workflow/mp/u1",
            output_groups_req=None, fn_arn="arn:fn")
    return result, mocks


class TestHappyPath(unittest.TestCase):

    def test_two_source_crossfade_edit(self):
        result, mocks = call(BASE_QE, TWO_SDR)
        data = result["data"]
        self.assertEqual(result["status"], "COMPLETED")
        # 8 s + 6 s − 1 s crossfade = 13 s
        self.assertEqual(data["duration_sec"], 13.0)
        # 2 bodies + 1 junction, slim iterators pointing at the plan
        self.assertEqual(len(data["iterators"]), 3)
        for i, it in enumerate(data["iterators"]):
            self.assertEqual(it["type"], "edit")
            self.assertEqual(it["entry_index"], i)
            self.assertEqual(it["plan_key"],
                             "micro-workflow/mp/u1/_edit/edit_plan.json")
        # default output group is the edit mp4; merge runs unchanged
        self.assertEqual(data["output_groups"][0]["format"], "mp4")
        self.assertEqual(data["output_groups"][0]["group_name"], "edit")
        # plan side object + probe.json both written
        keys = [c.args[1] for c in mocks["put_json"].call_args_list]
        self.assertIn("micro-workflow/mp/u1/_edit/edit_plan.json", keys)
        self.assertIn("micro-workflow/mp/u1/probe.json", keys)
        self.assertIn("micro-workflow/mp/u1/edit/edit_manifest.json", keys)
        self.assertEqual(data["output_keys"]["edit_manifest"],
                         "micro-workflow/mp/u1/edit/edit_manifest.json")
        # the manifest is the evidence output: record spans + requested
        # vs actual + overlap-trim mode stated
        man = self.doc_for(mocks, "edit_manifest.json")
        self.assertEqual(man["crossfade_mode"], "overlap-trim")
        self.assertEqual(man["total_ms"], 13000.0)
        self.assertEqual(len(man["segments"]), 2)
        self.assertEqual(man["segments"][1]["record_in_ms"], 7000.0)

    @staticmethod
    def doc_for(mocks, suffix):
        for c in mocks["put_json"].call_args_list:
            if c.args[1].endswith(suffix):
                return c.args[2]
        raise AssertionError(f"no put_json for *{suffix}")

    def test_loudnorm_default_on_for_multi_source(self):
        _, mocks = call(BASE_QE, TWO_SDR)
        plan_doc = self.doc_for(mocks, "edit_plan.json")
        self.assertTrue(plan_doc["loudnorm_on"])
        # per-SOURCE windowed measurement — ONE entry per unique source
        self.assertEqual(set(plan_doc["loudness"]),
                         {"s3://bkt/a.mp4", "s3://bkt/b.mp4"})
        self.assertEqual(plan_doc["loudness"]["s3://bkt/a.mp4"]["input_i"],
                         "-19.5")

    def test_loudness_measured_once_per_unique_source(self):
        # two segments from ONE source → one measurement (the dedupe),
        # over the fixed shallow window, not the segments' own ranges
        qe = {"segments": [
            {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 0, "out_ms": 5000},
            {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 400000,
             "out_ms": 405000}],
            "loudnorm": True,
            "target": {"resolution": "1920x1080", "fps": "25"}}
        run_mock = mock.Mock(return_value=f"x\n{LOUDNORM_JSON}\n")
        with patched(TWO_SDR):
            with mock.patch.object(qep, "run", run_mock):
                qep.handle_quick_edit(
                    {"input": {"quick_edit": qe}, "data": {}}, qe,
                    uuid="u1", out_bucket="proxy", out_prefix="p",
                    output_groups_req=None, fn_arn="arn:fn")
        ff_calls = [c for c in run_mock.call_args_list
                    if "loudnorm" in str(c)]
        self.assertEqual(len(ff_calls), 1)
        cmd = ff_calls[0].args[0]
        # window: start 180 s (skip heads), 300 s long — never the
        # segments' deep offsets
        self.assertEqual(cmd[cmd.index("-ss") + 1], "180.000")
        self.assertEqual(cmd[cmd.index("-to") + 1], "480.000")

    def test_loudnorm_default_off_single_source_flag_overrides(self):
        qe = {"segments": [
            {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 0, "out_ms": 5000},
            {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 9000,
             "out_ms": 14000}],
            "target": {"resolution": "1920x1080", "fps": "25"}}
        _, mocks = call(qe, TWO_SDR)
        self.assertFalse(self.doc_for(mocks, "edit_plan.json")["loudnorm_on"])
        qe2 = dict(qe, loudnorm=True)
        _, mocks = call(qe2, TWO_SDR)
        self.assertTrue(self.doc_for(mocks, "edit_plan.json")["loudnorm_on"])

    def test_target_derived_when_sources_agree(self):
        qe = {"segments": BASE_QE["segments"]}   # no explicit target
        result, _ = call(qe, TWO_SDR)
        self.assertEqual(result["data"]["duration_sec"], 13.0)


class TestProbePool(unittest.TestCase):
    """probe_sources: bounded-pool probing with sequential-loop semantics."""

    def test_results_map_uris_in_order(self):
        out = qep.probe_sources(["a", "b", "c"], lambda u: {"src": u})
        self.assertEqual(list(out), ["a", "b", "c"])
        self.assertEqual(out["b"], {"src": "b"})

    def test_refusal_surfaces_for_the_first_bad_source_by_input_order(self):
        # c fails INSTANTLY, b fails after a delay — the sequential loop
        # would have named b, and the pool must too (map order, not
        # completion order)
        import time as _time

        def probe(u):
            if u == "b":
                _time.sleep(0.05)
                raise ValueError("bad b")
            if u == "c":
                raise ValueError("bad c")
            return {"src": u}

        with self.assertRaises(ValueError) as cm:
            qep.probe_sources(["a", "b", "c"], probe)
        self.assertEqual(str(cm.exception), "bad b")


class TestRefusals(unittest.TestCase):

    def refuse(self, qe, sources, needle):
        with patched(sources):
            with self.assertRaises(ValueError) as cm:
                qep.handle_quick_edit(
                    {"input": {"quick_edit": qe}, "data": {}}, qe,
                    uuid="u1", out_bucket="proxy", out_prefix="p",
                    output_groups_req=None, fn_arn="arn:fn")
        self.assertIn(needle, str(cm.exception))

    def test_hdr_source_refused_by_name(self):
        self.refuse(BASE_QE,
                    {"s3://bkt/a.mp4": {"transfer": "smpte2084"},
                     "s3://bkt/b.mp4": {}},
                    "HDR requires tonemapping")

    def test_out_beyond_source_duration(self):
        self.refuse(BASE_QE,
                    {"s3://bkt/a.mp4": {"duration": 5.0},
                     "s3://bkt/b.mp4": {}},
                    "beyond the source's 5.000s duration")

    def test_channel_v_needs_video(self):
        qe = {"segments": [
            {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 0, "out_ms": 5000,
             "channel": "v"}],
            "target": {"resolution": "1920x1080", "fps": "25"}}
        self.refuse(qe, {"s3://bkt/a.mp4": {"video": False}},
                    "needs a video stream")

    def test_channel_a_needs_audio(self):
        qe = {"segments": [
            {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 0, "out_ms": 5000,
             "channel": "a"}],
            "target": {"resolution": "1920x1080", "fps": "25"}}
        self.refuse(qe, {"s3://bkt/a.mp4": {"audio": False}},
                    "needs an audio stream")

    def test_sources_disagree_without_target(self):
        qe = {"segments": BASE_QE["segments"]}
        self.refuse(qe, {"s3://bkt/a.mp4": {},
                         "s3://bkt/b.mp4": {"w": 1280, "h": 720}},
                    "sources disagree on resolution")

    def test_overlay_source_needs_audio(self):
        qe = dict(BASE_QE, audio_overlays=[
            {"source_s3uri": "s3://bkt/v.mp4", "in_ms": 0, "out_ms": 1000,
             "at_record_ms": 0, "mode": "mix"}])
        self.refuse(qe, dict(TWO_SDR, **{
            "s3://bkt/v.mp4": {"audio": False}}), "no audio stream")

    def test_bad_resolution_string(self):
        qe = dict(BASE_QE, target={"resolution": "1080p", "fps": "25"})
        self.refuse(qe, TWO_SDR, "WIDTHxHEIGHT")

    def test_odd_target_resolution_refused(self):
        qe = dict(BASE_QE, target={"resolution": "1079x1919", "fps": "25"})
        self.refuse(qe, TWO_SDR, "must be even")


if __name__ == "__main__":
    unittest.main()
