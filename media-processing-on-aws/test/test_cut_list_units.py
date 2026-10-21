"""
Offline unit tests for lib/cut_list.py — the quick-edit cut list.

Every settled rule in README.md §14 lands here as an assertion:
the frame-mapping rule, overlap-trim record derivation, the junction
plan, the black generator, channel selectors, overlays/beds, and every
named refusal. The "worked example" tests pin a multi-source example
end to end (600 frames / 7 Map entries).
"""
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.cut_list import (  # noqa: E402
    BLACK_SOURCE, CutListError, MAX_OUTPUT_MS, build_plan, frame_to_ms,
    ms_to_frame, parse_fps, validate_quick_edit,
)

FPS25 = Fraction(25)
NTSC = Fraction(30000, 1001)


def seg(src="s3://bkt/a.mp4", in_ms=0, out_ms=10000, **kw):
    d = {"source_s3uri": src, "in_ms": in_ms, "out_ms": out_ms}
    d.update(kw)
    return d


def qe(segments, **kw):
    d = {"segments": segments}
    d.update(kw)
    return d


class TestFrameMath(unittest.TestCase):

    def test_mapping_rule_rounds_half_up(self):
        # 5500 ms at 25 fps = 137.5 frames — half-up ⇒ 138, deterministically
        self.assertEqual(ms_to_frame(5500, FPS25), 138)
        self.assertEqual(ms_to_frame(5499, FPS25), 137)

    def test_exact_at_ntsc_rates(self):
        # 1001 ms at 30000/1001 fps is exactly 30 frames — no float slop
        self.assertEqual(ms_to_frame(1001, NTSC), 30)
        self.assertEqual(ms_to_frame(100100, NTSC), 3000)

    def test_in_inclusive_out_exclusive_adjacency(self):
        # two segments cut at the same ms map to the same frame:
        # no duplicated, no dropped frame at the butt joint
        norm = validate_quick_edit(qe([
            seg(in_ms=0, out_ms=5000), seg(in_ms=5000, out_ms=10000)]))
        plan = build_plan(norm, FPS25)
        a, b = plan["entries"]
        self.assertEqual(a["src_out_frame"], b["src_in_frame"])
        self.assertEqual(a["record_out_frame"], b["record_in_frame"])
        self.assertEqual(plan["total_frames"], 250)

    def test_parse_fps_forms(self):
        self.assertEqual(parse_fps("25"), Fraction(25))
        self.assertEqual(parse_fps("30000/1001"), NTSC)
        self.assertEqual(parse_fps(25), Fraction(25))
        for bad in ("0", "abc", "25/0", -1):
            with self.assertRaises(CutListError):
                parse_fps(bad)

    def test_frame_to_ms_round_trips_the_grid(self):
        self.assertEqual(frame_to_ms(600, FPS25), 24000.0)


class TestValidationRefusals(unittest.TestCase):
    """Every refusal names its condition — asserted via message content."""

    def refuse(self, payload, needle, fps=FPS25):
        with self.assertRaises(CutListError) as cm:
            build_plan(validate_quick_edit(payload), fps)
        self.assertIn(needle, str(cm.exception))

    def test_first_segment_crossfade(self):
        self.refuse(qe([
            seg(transition_in={"type": "crossfade", "duration_ms": 500}),
            seg()]), "first segment")

    def test_channel_on_black(self):
        self.refuse(qe([seg(), seg(src=BLACK_SOURCE, in_ms=0, out_ms=1000,
                                   channel="v")]), "black")

    def test_unknown_channel(self):
        self.refuse(qe([seg(channel="audio")]), "channel")

    def test_cut_with_duration(self):
        self.refuse(qe([seg(), seg(transition_in={
            "type": "cut", "duration_ms": 500})]), "cut has no duration")

    def test_transition_over_cap(self):
        self.refuse(qe([seg(), seg(transition_in={
            "type": "fade", "duration_ms": 5001})]), "5000 ms cap")

    def test_boundary_budget_exceeded(self):
        # 1 s segment with 600 ms crossfades on both sides
        self.refuse(qe([
            seg(),
            seg(in_ms=0, out_ms=1000,
                transition_in={"type": "crossfade", "duration_ms": 600}),
            seg(transition_in={"type": "crossfade", "duration_ms": 600})]),
            "boundary demands")

    def test_crossfade_below_one_frame(self):
        self.refuse(qe([seg(), seg(transition_in={
            "type": "crossfade", "duration_ms": 10})]), "below one frame")

    def test_segment_below_one_frame(self):
        self.refuse(qe([seg(in_ms=0, out_ms=10)]), "shorter than one frame")

    def test_segment_cap(self):
        self.refuse(qe([seg(in_ms=i * 100, out_ms=i * 100 + 100)
                        for i in range(101)]), "100-segment cap")

    def test_source_cap_counts_overlays(self):
        payload = qe([seg(src=f"s3://bkt/{i}.mp4") for i in range(25)],
                     audio_overlays=[{"source_s3uri": "s3://bkt/extra.wav",
                                      "in_ms": 0, "out_ms": 1000,
                                      "at_record_ms": 0, "mode": "mix"}])
        self.refuse(payload, "source cap")

    def test_duration_cap(self):
        self.refuse(qe([seg(in_ms=0, out_ms=MAX_OUTPUT_MS + 1)]), "1 hour")

    def test_overlay_beyond_output(self):
        self.refuse(qe([seg()], audio_overlays=[
            {"source_s3uri": "s3://bkt/vo.wav", "in_ms": 0, "out_ms": 2000,
             "at_record_ms": 9000, "mode": "replace"}]), "beyond")

    def test_replace_overlap(self):
        self.refuse(qe([seg()], audio_overlays=[
            {"source_s3uri": "s3://bkt/x.wav", "in_ms": 0, "out_ms": 3000,
             "at_record_ms": 0, "mode": "replace"},
            {"source_s3uri": "s3://bkt/y.wav", "in_ms": 0, "out_ms": 3000,
             "at_record_ms": 2000, "mode": "replace"}]), "overlap")

    def test_gain_range(self):
        self.refuse(qe([seg()], audio_overlays=[
            {"source_s3uri": "s3://bkt/x.wav", "in_ms": 0, "out_ms": 1000,
             "at_record_ms": 0, "mode": "mix", "gain_db": 13}]), "gain_db")

    def test_black_as_overlay_source(self):
        self.refuse(qe([seg()], audio_overlays=[
            {"source_s3uri": BLACK_SOURCE, "in_ms": 0, "out_ms": 1000,
             "at_record_ms": 0, "mode": "mix"}]), "s3://")

    def test_bed_on_first_segment(self):
        self.refuse(qe([seg(transition_in={
            "type": "cut",
            "audio_bed": {"source_s3uri": "s3://bkt/x.wav", "in_ms": 0,
                          "out_ms": 1000, "lead_ms": 0}}), seg()]),
            "carries no")

    def test_bed_on_global_default(self):
        self.refuse(qe([seg(), seg()], transition={
            "type": "cut",
            "audio_bed": {"source_s3uri": "s3://bkt/x.wav", "in_ms": 0,
                          "out_ms": 1000, "lead_ms": 0}}),
            "global default")

    def test_bed_lead_before_output_start(self):
        self.refuse(qe([seg(), seg(transition_in={
            "type": "crossfade", "duration_ms": 1000,
            "audio_bed": {"source_s3uri": "s3://bkt/x.wav", "in_ms": 0,
                          "out_ms": 1000, "lead_ms": 60000}})]),
            "before the output begins")

    def test_overlay_fades_exceed_element(self):
        self.refuse(qe([seg()], audio_overlays=[
            {"source_s3uri": "s3://bkt/x.wav", "in_ms": 0, "out_ms": 1000,
             "at_record_ms": 0, "mode": "mix",
             "fade_in_ms": 600, "fade_out_ms": 600}]), "exceed")

    def test_crop_out_of_range(self):
        self.refuse(qe([seg(crop={"x": 0.5, "y": 0.0,
                                  "width": 0.6, "height": 1.0})]),
                    "beyond the frame")
        self.refuse(qe([seg(crop={"x": 0, "y": 0,
                                  "width": 1.2, "height": 1})]),
                    "ratio in [0, 1]")

    def test_crop_on_channel_a_refused(self):
        # an audio-only segment carries no picture, and the probe facts
        # for an audio source have no width/height to crop against
        self.refuse(qe([seg(channel="a",
                            crop={"x": 0, "y": 0, "width": 1,
                                  "height": 1})]),
                    "crop does not apply to channel 'a'")

    def test_crop_on_black(self):
        self.refuse(qe([seg(), seg(src=BLACK_SOURCE, in_ms=0, out_ms=1000,
                                   crop={"x": 0, "y": 0, "width": 1,
                                         "height": 1})]),
                    "crop does not apply")

    def test_crop_rides_into_the_entry(self):
        crop = {"x": 0.34, "y": 0.0, "width": 0.3125, "height": 1.0}
        norm = validate_quick_edit(qe([seg(crop=crop)]))
        plan = build_plan(norm, FPS25)
        self.assertEqual(plan["entries"][0]["crop"], crop)

    def test_non_s3_source(self):
        self.refuse(qe([seg(src="http://x/y.mp4")]), "s3://")

    def test_empty_range(self):
        self.refuse(qe([seg(in_ms=5000, out_ms=5000)]), "empty range")


class TestDefaults(unittest.TestCase):

    def test_defaults_applied(self):
        norm = validate_quick_edit(qe([seg(), seg()]))
        self.assertEqual(norm["segments"][0]["channel"], "av")
        self.assertEqual(norm["segments"][1]["transition_in"]["type"], "cut")

    def test_global_default_propagates(self):
        norm = validate_quick_edit(qe(
            [seg(), seg()],
            transition={"type": "fade", "duration_ms": 400}))
        self.assertEqual(norm["segments"][1]["transition_in"],
                         {"type": "fade", "duration_ms": 400})
        # the first segment never inherits a boundary default
        self.assertEqual(norm["segments"][0]["transition_in"]["type"], "cut")

    def test_bed_defaults(self):
        norm = validate_quick_edit(qe([seg(), seg(transition_in={
            "type": "cut",
            "audio_bed": {"source_s3uri": "s3://bkt/x.wav", "in_ms": 0,
                          "out_ms": 4000, "lead_ms": 0}})]))
        bed = norm["segments"][1]["transition_in"]["audio_bed"]
        self.assertEqual(bed["mode"], "mix")
        self.assertEqual(bed["fade_in_ms"], 500)
        self.assertEqual(bed["fade_out_ms"], 500)


class TestWorkedExample(unittest.TestCase):
    """The plan's own multi-source example, pinned end to end."""

    PAYLOAD = {
        "segments": [
            {"source_s3uri": "s3://bkt/intvw.mxf", "in_ms": 10000,
             "out_ms": 18000,
             "transition_in": {"type": "fade", "duration_ms": 500}},
            {"source_s3uri": "s3://bkt/drone.mp4", "in_ms": 65000,
             "out_ms": 71000,
             "transition_in": {"type": "crossfade", "duration_ms": 1000,
                               "audio_bed": {
                                   "source_s3uri": "s3://bkt/sting.wav",
                                   "in_ms": 0, "out_ms": 4000,
                                   "lead_ms": 1500, "mode": "mix",
                                   "gain_db": -15, "fade_in_ms": 300,
                                   "fade_out_ms": 700}}},
            {"source_s3uri": "s3://bkt/intvw.mxf", "in_ms": 120000,
             "out_ms": 126000},
            {"source_s3uri": BLACK_SOURCE, "in_ms": 0, "out_ms": 1600,
             "transition_in": {"type": "crossfade", "duration_ms": 800}},
            {"source_s3uri": "s3://bkt/phone.mov", "in_ms": 3000,
             "out_ms": 8000,
             "transition_in": {"type": "crossfade", "duration_ms": 800}},
        ],
        "audio_overlays": [
            {"source_s3uri": "s3://bkt/vo.wav", "in_ms": 0, "out_ms": 6000,
             "at_record_ms": 13000, "mode": "replace"},
        ],
        "transition": {"type": "cut"},
        "target": {"resolution": "1920x1080", "fps": "25"},
    }

    def setUp(self):
        self.norm = validate_quick_edit(self.PAYLOAD)
        self.plan = build_plan(self.norm, FPS25)

    def test_duration_budget(self):
        # Σ(out−in) = 26.6 s, crossfades 1.0+0.8+0.8 = 2.6 s ⇒ 24 s / 600 fr
        self.assertEqual(self.norm["total_authored_ms"], 26600)
        self.assertEqual(self.plan["total_frames"], 600)
        self.assertEqual(self.plan["total_ms"], 24000.0)

    def test_seven_map_entries_black_body_skipped(self):
        kinds = [e["kind"] for e in self.plan["entries"]]
        self.assertEqual(kinds.count("body"), 4)      # 5 bodies, 1 zero-length
        self.assertEqual(kinds.count("junction"), 3)
        body_segs = [e["segment_index"] for e in self.plan["entries"]
                     if e["kind"] == "body"]
        self.assertNotIn(3, body_segs)                # the black segment

    def test_record_timeline_matches_the_plan_table(self):
        spans = {(e.get("segment_index"), e["kind"]):
                 (e["record_in_frame"], e["record_out_frame"])
                 for e in self.plan["entries"]}
        self.assertEqual(spans[(0, "body")], (0, 175))     # 0–7000 ms
        self.assertEqual(spans[(1, "body")], (200, 325))   # 8000–13000 ms
        self.assertEqual(spans[(2, "body")], (325, 455))   # 13000–18200 ms
        self.assertEqual(spans[(4, "body")], (495, 600))   # 19800–24000 ms
        junctions = [e for e in self.plan["entries"] if e["kind"] == "junction"]
        self.assertEqual([(j["record_in_frame"], j["record_out_frame"])
                          for j in junctions],
                         [(175, 200), (455, 475), (475, 495)])

    def test_entries_tile_the_output_exactly(self):
        covered = sorted((e["record_in_frame"], e["record_out_frame"])
                         for e in self.plan["entries"])
        pos = 0
        for lo, hi in covered:
            self.assertEqual(lo, pos)
            pos = hi
        self.assertEqual(pos, self.plan["total_frames"])

    def test_micro_fade_edges_mark_cuts_only(self):
        micro = {e["segment_index"]: (e["micro_in"], e["micro_out"])
                 for e in self.plan["entries"] if e["kind"] == "body"}
        # seg0: fade-in envelope head, crossfade tail → no micros
        self.assertEqual(micro[0], (False, False))
        # seg1: junction head, CUT tail (seg2 arrives on a cut)
        self.assertEqual(micro[1], (False, True))
        # seg2: cut head, crossfade tail
        self.assertEqual(micro[2], (True, False))
        # seg4: junction head, end of output
        self.assertEqual(micro[4], (False, True))

    def test_junction_sources_and_src_positions(self):
        j1 = [e for e in self.plan["entries"] if e["kind"] == "junction"][0]
        self.assertEqual(j1["left_source_s3uri"], "s3://bkt/intvw.mxf")
        self.assertEqual(j1["left_src_in_frame"], 425)   # 17.0 s at 25 fps
        self.assertEqual(j1["right_source_s3uri"], "s3://bkt/drone.mp4")
        self.assertEqual(j1["right_src_in_frame"], 1625)  # 65.0 s

    def test_bed_resolved_against_the_boundary(self):
        bed = [o for o in self.plan["overlays"] if o["kind"] == "bed"][0]
        # junction 1 starts at frame 175 (7000 ms); lead 1500 ms = 38 fr
        self.assertEqual(bed["record_in_frame"], 137)
        self.assertEqual(bed["record_out_frame"], 237)    # +4000 ms = 100 fr
        # sliced across seg-1 tail, the junction, and seg-2 head
        slices = [(e["kind"], e.get("segment_index"))
                  for e in self.plan["entries"] if e["audio_slices"]
                  if any(s["kind"] == "bed" for s in e["audio_slices"])]
        self.assertEqual(slices, [("body", 0), ("junction", None),
                                  ("body", 1)])

    def test_replace_overlay_covers_segment_3_exactly(self):
        ov = [o for o in self.plan["overlays"] if o["kind"] == "overlay"][0]
        self.assertEqual((ov["record_in_frame"], ov["record_out_frame"]),
                         (325, 475))
        rec = self.plan["segment_records"][2]
        self.assertEqual((rec["record_in_frame"], rec["record_out_frame"]),
                         (325, 475))  # body + its fade-out junction

    def test_slice_src_offsets_advance_with_the_cut(self):
        slices = [s for e in self.plan["entries"]
                  for s in e["audio_slices"] if s["kind"] == "bed"]
        self.assertEqual(slices[0]["src_offset_frames"], 0)
        for prev, cur in zip(slices, slices[1:]):
            advance = cur["record_in_frame"] - prev["record_in_frame"]
            self.assertEqual(cur["src_offset_frames"] -
                             prev["src_offset_frames"], advance)


class TestCapPin(unittest.TestCase):

    def test_100_segments_all_crossfaded_stay_well_formed(self):
        # the plan's cap-insurance test: ~199 pieces, consistent timeline
        segments = [seg(in_ms=0, out_ms=2000)]
        segments += [seg(in_ms=0, out_ms=2000,
                         transition_in={"type": "crossfade",
                                        "duration_ms": 500})
                     for _ in range(99)]
        plan = build_plan(validate_quick_edit(qe(segments)), FPS25)
        kinds = [e["kind"] for e in plan["entries"]]
        self.assertEqual(kinds.count("body"), 100)
        self.assertEqual(kinds.count("junction"), 99)
        # Σ 200 s − 99 × 0.5 s = 150.5 s ⇒ 3762.5 ⇒ half-up mapping per cut
        self.assertEqual(plan["total_frames"],
                         100 * 50 - 99 * ms_to_frame(500, FPS25))
        covered = sorted((e["record_in_frame"], e["record_out_frame"])
                         for e in plan["entries"])
        self.assertEqual(covered[0][0], 0)
        self.assertEqual(covered[-1][1], plan["total_frames"])


if __name__ == "__main__":
    unittest.main()
