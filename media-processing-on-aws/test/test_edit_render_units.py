"""
Offline tests for states/chunk_transcode/edit_render.py: build the plan
with the real cut_list, then assert the ffmpeg argv/filter_complex each
entry kind produces. No ffmpeg runs.
"""
import os
import sys
import unittest
from fractions import Fraction
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.cut_list import build_plan, validate_quick_edit  # noqa: E402
from states.chunk_transcode.edit_render import build_entry_command  # noqa: E402

FPS = Fraction(25)

QE = {
    "segments": [
        {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 10000, "out_ms": 18000,
         "transition_in": {"type": "fade", "duration_ms": 500}},
        {"source_s3uri": "s3://bkt/b.mp4", "in_ms": 65000, "out_ms": 71000,
         "transition_in": {"type": "crossfade", "duration_ms": 1000}},
        {"source_s3uri": "black", "in_ms": 0, "out_ms": 2000},
        {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 30000, "out_ms": 36000,
         "channel": "v",
         "crop": {"x": 0.34, "y": 0.0, "width": 0.3125, "height": 1.0}},
    ],
    "audio_overlays": [
        {"source_s3uri": "s3://bkt/vo.wav", "in_ms": 0, "out_ms": 4000,
         "at_record_ms": 15000, "mode": "replace", "gain_db": 0,
         "fade_in_ms": 100, "fade_out_ms": 100},
    ],
}

# is_all_intra=True keeps the offline tests off the keyframe-scan
# network path (direct seek); the scan path has its own mocked test
SOURCES = {
    "s3://bkt/a.mp4": {"bucket": "bkt", "key": "a.mp4", "width": 1920,
                       "height": 1080, "field_order": "progressive",
                       "fps": "25/1", "is_all_intra": True,
                       "has_audio": True},
    "s3://bkt/b.mp4": {"bucket": "bkt", "key": "b.mp4", "width": 1280,
                       "height": 720, "field_order": "tt",
                       "fps": "25/1", "is_all_intra": True,
                       "has_audio": True},
    "s3://bkt/vo.wav": {"bucket": "bkt", "key": "vo.wav",
                        "has_audio": True},
}
URLS = {u: f"https://signed/{f['key']}" for u, f in SOURCES.items()}

MEASURED = {"input_i": "-19.5", "input_tp": "-0.3", "input_lra": "6.2",
            "input_thresh": "-30.1", "target_offset": "0.4"}


def plan_doc(loudnorm_on=True):
    norm = validate_quick_edit(QE)
    plan = build_plan(norm, FPS)
    # per-SOURCE windowed measurement (user decision 2026-10-10)
    loud = {u: MEASURED for u in SOURCES}
    return {
        "quick_edit": norm, "plan": plan,
        "target": {"width": 1920, "height": 1080, "fps": "25"},
        "sources": SOURCES,
        "loudness": loud if loudnorm_on else {},
        "loudnorm_on": loudnorm_on,
    }


def entry_of(doc, kind, segment_index=None):
    for e in doc["plan"]["entries"]:
        if e["kind"] == kind and (segment_index is None or
                                  e.get("segment_index") == segment_index):
            return e
    raise AssertionError(f"no {kind} entry for segment {segment_index}")


def fc_of(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


class TestBodyEntry(unittest.TestCase):

    def setUp(self):
        self.doc = plan_doc()

    def test_body_command_shape(self):
        e = entry_of(self.doc, "body", 0)
        cmd, outputs = build_entry_command(e, self.doc, 0, "/tmp/x", URLS)
        fc = fc_of(cmd)
        # SOURCE-grid cut first (proven order), window = n + 2 margin
        self.assertIn("trim=start_frame=0:end_frame=177,setpts=PTS-STARTPTS",
                      fc)
        # normalized to the mezzanine, then TARGET-grid pin + envelope
        self.assertIn("scale=1920:1080,setsar=1,fps=25,format=yuv420p", fc)
        self.assertIn("trim=start_frame=0:end_frame=175", fc)
        self.assertIn("fade=t=in:s=0:n=13", fc)        # 500 ms fade-in
        # all-intra source: direct frame seek (10.0 s = frame 250 @25)
        self.assertIn("10.000000", cmd[cmd.index("-ss") + 0 + 1])
        # audio: house conform + SAMPLE-exact trim against the 3 s
        # pre-roll (seek 7 s → start 144000; 175 fr → 336000 samples)
        self.assertIn("aresample=48000:async=1", fc)
        self.assertIn("apad", fc)
        self.assertIn("aformat=sample_fmts=s16:channel_layouts=stereo", fc)
        self.assertIn("atrim=start_sample=144000:end_sample=480000", fc)
        # loudness pass 2: a plain per-source volume gain (never
        # loudnorm — its dynamic fallback outputs 192 kHz)
        self.assertIn("volume=-3.50dB", fc)
        self.assertNotIn("linear=true", fc)
        # identical mezzanine encode params for every entry
        self.assertIn("libopenh264", " ".join(cmd))
        self.assertEqual(outputs[0]["s3_name"], "video_0.ts")
        self.assertEqual(outputs[1]["s3_name"], "audio_0.flac")

    def test_interlaced_source_gets_yadif(self):
        e = entry_of(self.doc, "body", 1)
        cmd, _ = build_entry_command(e, self.doc, 1, "/tmp/x", URLS)
        self.assertIn("yadif=deint=interlaced", fc_of(cmd))  # proxy spelling
        self.assertIn("scale=1920:1080", fc_of(cmd))   # 720p upscaled

    def test_inter_frame_source_counts_from_the_keyframe(self):
        # the proven machinery: seek the PRIOR KEYFRAME, pick the frame
        # by COUNTING (trim start_frame=preroll), never trust the seek
        import states.chunk_transcode.edit_render as er
        doc = plan_doc()
        doc["sources"] = {u: dict(f, is_all_intra=False)
                          for u, f in doc["sources"].items()}
        e = entry_of(doc, "body", 0)   # src frame 250 = 10.0 s
        with mock.patch.object(
                er, "find_prior_keyframe",
                return_value={"best_effort_timestamp_time": "9.200000"}):
            cmd, _ = build_entry_command(e, doc, 0, "/tmp/x", URLS)
        self.assertIn("9.200000", cmd[cmd.index("-ss") + 1])
        # keyframe = frame 230 @25 → preroll 20
        self.assertIn("trim=start_frame=20:end_frame=197", fc_of(cmd))

    def test_black_body_is_a_bounded_lavfi_source(self):
        e = entry_of(self.doc, "body", 2)
        cmd, _ = build_entry_command(e, self.doc, 2, "/tmp/x", URLS)
        joined = " ".join(cmd)
        self.assertIn("color=c=black:s=1920x1080:r=25:d=2.000000", joined)
        self.assertIn("anullsrc=r=48000:cl=stereo", joined)

    def test_channel_v_gets_silence_base(self):
        e = entry_of(self.doc, "body", 3)
        cmd, _ = build_entry_command(e, self.doc, 3, "/tmp/x", URLS)
        self.assertIn("anullsrc=r=48000:cl=stereo", " ".join(cmd))
        # the source IS seeked for video
        self.assertIn("https://signed/a.mp4", cmd)

    def test_segment_crop_lands_in_the_chain(self):
        e = entry_of(self.doc, "body", 3)
        cmd, _ = build_entry_command(e, self.doc, 3, "/tmp/x", URLS)
        self.assertIn("crop=600:1080:652:0", fc_of(cmd))

    def test_micro_fades_on_cut_edges_only(self):
        # seg2 (black) arrives on a cut; its body gets the 20 ms micros
        e = entry_of(self.doc, "body", 2)
        cmd, _ = build_entry_command(e, self.doc, 2, "/tmp/x", URLS)
        self.assertIn("afade=t=in:st=0:d=0.02", fc_of(cmd))
        # seg1's head is a junction splice — no head micro
        e1 = entry_of(self.doc, "body", 1)
        fc1 = fc_of(build_entry_command(e1, self.doc, 1, "/tmp/x", URLS)[0])
        self.assertNotIn("afade=t=in:st=0:d=0.02", fc1)


class TestJunctionEntry(unittest.TestCase):

    def test_a_channel_side_junctions_as_black_picture(self):
        # a crossfade adjacent to an "a"-only segment must blend
        # against black, not that segment's picture (review catch)
        qe = {"segments": [
            {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 0, "out_ms": 8000},
            {"source_s3uri": "s3://bkt/b.mp4", "in_ms": 0, "out_ms": 8000,
             "channel": "a",
             "transition_in": {"type": "crossfade", "duration_ms": 1000}}]}
        norm = validate_quick_edit(qe)
        doc = dict(plan_doc(), quick_edit=norm,
                   plan=build_plan(norm, FPS))
        e = entry_of(doc, "junction")
        cmd, _ = build_entry_command(e, doc, 1, "/tmp/x", URLS)
        joined = " ".join(cmd)
        # right side (the "a" segment) is a generated black input
        self.assertIn("color=c=black", joined)

    def test_junction_two_inputs_xfade_acrossfade(self):
        doc = plan_doc()
        e = entry_of(doc, "junction")
        cmd, _ = build_entry_command(e, doc, 1, "/tmp/x", URLS)
        fc = fc_of(cmd)
        self.assertIn("xfade=transition=fade:duration=1:offset=0", fc)
        self.assertIn("acrossfade=d=1", fc)
        # both sources present as seeked inputs
        joined = " ".join(cmd)
        self.assertIn("https://signed/a.mp4", joined)
        self.assertIn("https://signed/b.mp4", joined)
        # each side carries its own SOURCE's gain
        self.assertEqual(fc.count("volume=-3.50dB"), 2)


class TestOverlaySlices(unittest.TestCase):

    def test_replace_overlay_mutes_base_and_mixes(self):
        doc = plan_doc()
        # overlay 15000–19000 ms sits inside seg1's body (8000–19200 rec?)
        target = next(e for e in doc["plan"]["entries"]
                      if e["audio_slices"])
        cmd, _ = build_entry_command(target, doc, 9, "/tmp/x", URLS)
        fc = fc_of(cmd)
        self.assertIn("volume=enable='between(t,", fc)
        self.assertIn("amix=inputs=2:duration=first:normalize=0", fc)
        # delays are SAMPLE-exact (adelay=NS), never rounded ms
        self.assertRegex(fc, r"adelay=\d+S:all=1")
        self.assertIn("https://signed/vo.wav", " ".join(cmd))

    def test_no_overlays_passes_base_through(self):
        doc = plan_doc()
        e = entry_of(doc, "body", 0)
        cmd, _ = build_entry_command(e, doc, 0, "/tmp/x", URLS)
        # loudnorm on → the TP-safety limiter closes the chain
        self.assertIn("[abase]alimiter=limit=0.98[aout]", fc_of(cmd))

    def test_no_overlays_no_loudnorm_is_a_plain_passthrough(self):
        doc = plan_doc(loudnorm_on=False)
        e = entry_of(doc, "body", 0)
        cmd, _ = build_entry_command(e, doc, 0, "/tmp/x", URLS)
        self.assertIn("[abase]anull[aout]", fc_of(cmd))

    def test_loudnorm_off_omits_gain_passes(self):
        doc = plan_doc(loudnorm_on=False)
        e = entry_of(doc, "body", 0)
        cmd, _ = build_entry_command(e, doc, 0, "/tmp/x", URLS)
        self.assertNotIn("volume=", fc_of(cmd))
        # and no loudnorm anywhere in a render graph, ever
        self.assertNotIn("loudnorm", fc_of(cmd))


if __name__ == "__main__":
    unittest.main()
