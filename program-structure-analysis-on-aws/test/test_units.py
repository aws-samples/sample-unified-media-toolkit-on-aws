"""
program-structure-analysis unit tests. Fixtures are SYNTHETIC (the
generalize-never-content-specific rule); each measured incident a rule
came from is named where the fixture encodes it.

Run: python3 -m unittest discover -s test -p 'test_*.py'
"""
from __future__ import annotations

import json
import os
import sys
import unittest

def _read(path, **kw):
    """_read() without the ResourceWarning — tests leave no
    unclosed handles (public build logs read clean)."""
    with open(path, **kw) as f:
        return f.read()


def _load_json(path):
    with open(path) as f:
        return json.load(f)



_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# the gateway dispatcher ships in its own flat zip; its modules import
# as top-level names, so its directory joins the path (AFTER the root:
# `import lambda_function` must resolve to the gateway, so it goes
# first in the search order)
sys.path.insert(0, os.path.join(_ROOT, "gateway_tool"))

os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("STATE_MACHINE_ARN",
                      "arn:aws:states:eu-west-1:000000000000:stateMachine:t")

from lib import acts        # noqa: E402
from lib import gaps        # noqa: E402
from lib import markers     # noqa: E402
from lib import roles       # noqa: E402
from lib import sequences   # noqa: E402
from lib import stories     # noqa: E402


def _times(durations, t0=0.0):
    out, t = {}, t0
    for i, d in enumerate(durations):
        out[i] = (t, t + d)
        t += d
    return out



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


class TestMarkers(unittest.TestCase):
    def test_runs_bridge_black_inside_but_split_on_long_black(self):
        docs = [{"shot_id": i, "shot_kind": k} for i, k in enumerate(
            ["normal", "credit_roll", "black", "credit_roll",
             "black", "credit_roll"])]
        tm = _times([10, 5, 2, 5, markers.BLACK_BRIDGE_MAX + 1, 5])
        m = markers.markers(docs, tm, None, 40)
        self.assertEqual([b["shot_ids"] for b in m["end_credits"]],
                         [[1, 2, 3], [5]])

    def test_opening_title_is_positional(self):
        docs = [{"shot_id": i, "shot_kind": k} for i, k in enumerate(
            ["normal", "title_card", "normal", "title_card", "normal"])]
        tm = _times([10, 5, 300, 5, 300])            # duration 620
        m = markers.markers(docs, tm, None, 620)
        self.assertEqual(len(m["opening_title"]), 1)      # the head one
        self.assertEqual(len(m["interior_title_cards"]), 1)

    def test_logo_only_blocks_are_positional(self):
        """A head studio logo must NOT become end_credits[0] (everything
        anchored to credits — content_end, textless tail — follows it);
        the same logo-only block in the tail IS end credits material."""
        docs = [{"shot_id": 0, "shot_kind": "logo_bumper"},
                {"shot_id": 1, "shot_kind": "normal"},
                {"shot_id": 2, "shot_kind": "logo_bumper"}]
        tm = {0: (0.0, 5.0), 1: (5.0, 2990.0), 2: (2990.0, 3000.0)}
        m = markers.markers(docs, tm, None, 3000)
        self.assertEqual([b["shot_ids"] for b in m["end_credits"]], [[2]])

    def test_silence_gaps_from_original_stem(self):
        stem = {"activity": {"original": [[0, 5000], [12000, 20000]]}}
        self.assertEqual(markers.silence_gaps(stem), [(5.0, 12.0)])

    def test_bars_tone_corroboration_needs_nominal_hz(self):
        docs = [{"shot_id": 0, "shot_kind": "bars"}]
        tm = _times([10])
        stem = {"tones": [{"start_ms": 0, "end_ms": 8000, "nominal_hz": 1000}]}
        m = markers.markers(docs, tm, stem, 10)
        self.assertTrue(m["bars_and_tone"][0]["tone_confirmed"])
        stem2 = {"tones": [{"start_ms": 0, "end_ms": 8000, "nominal_hz": None}]}
        m2 = markers.markers(docs, tm, stem2, 10)
        self.assertFalse(m2["bars_and_tone"][0]["tone_confirmed"])


class TestBreakCandidates(unittest.TestCase):
    def test_candidates_score_by_convergence_and_edges_are_excluded(self):
        docs = [{"shot_id": i, "shot_kind": k} for i, k in enumerate(
            ["black", "normal", "black", "normal"])]
        tm = {0: (10, 12), 1: (12, 500), 2: (500, 503), 3: (503, 1000)}
        stem = {"activity": {"original": [[12000, 499000], [504000, 990000]]}}
        m = markers.markers(docs, tm, stem, 1000)
        seams = acts.candidate_seams(m, tm, stem, None, 1000)
        # the head black (t=11) is inside EDGE_ZONE; only 501.5 remains
        self.assertEqual(len(seams), 1)
        self.assertAlmostEqual(seams[0]["at"], 501.5, places=1)
        self.assertTrue(seams[0]["evidence"]["silent"])

    def test_acts_abstention_is_structural(self):
        """The §8 negative result: nothing in this module emits acts[]
        without seams — masters get candidates, never act claims."""
        self.assertTrue(hasattr(acts, "candidate_seams"))
        # acts() exists for the as-aired/delivery-metadata path only; its
        # output is derived from ACCEPTED seams and never invented.
        m = {"blackframes": [], "opening_title": [], "end_credits": [],
             "interior_title_cards": []}
        out = acts.acts(m, {}, {"activity": {"original": []}}, None, 1000)
        self.assertEqual(len(out["acts"]), 1)     # no seams -> one span


class TestRoles(unittest.TestCase):
    def test_news_setting_is_primary_and_big_groups_mark_anchor(self):
        docs = [{"shot_id": 0, "news_setting": "studio_anchor_desk"},
                {"shot_id": 1, "news_setting": "field"},
                {"shot_id": 2, "shot_kind": "title_card"}]
        rec = {"recurrence_groups": [{"group_id": 7, "shots": [
            {"shot_id": 0, "position": "interior"}] * roles.MIN_ANCHOR_GROUP}]}
        r = roles.shot_roles(docs, rec)
        self.assertTrue(r[0]["anchor_set"])
        self.assertTrue(roles.is_story_role(r[1]))
        self.assertEqual(r[2]["role"], "graphic_full_screen")


class TestStories(unittest.TestCase):
    def _ca(self):
        return {"audio_segments": [
            {"start_time": "00:00:02.000", "end_time": "00:00:20.000",
             "segment_topic": "open", "news_form": "headlines_block",
             "slug": "OPEN"},
            {"start_time": "00:00:31.000", "end_time": "00:01:00.000",
             "segment_topic": "the story", "news_form": "package",
             "slug": "STORY-A"},
        ]}

    def _shots(self):
        # 0: graphics (0-6), 1: graphics promo tail cueless (20-28),
        # 2: anchor desk (28-40), 3: field (40-70)
        docs = [{"shot_id": 0, "news_setting": "graphic_full_screen"},
                {"shot_id": 1, "news_setting": "graphic_full_screen"},
                {"shot_id": 2, "news_setting": "studio_anchor_desk"},
                {"shot_id": 3, "news_setting": "field"}]
        tm = {0: (0.0, 6.0), 1: (20.0, 28.0), 2: (28.0, 40.0),
              3: (40.0, 70.0)}
        return docs, tm

    def test_three_timestamps_and_role_resolved_gap(self):
        """The promo-absorption defect class: the cueless graphics shot
        (20-28) between the headlines block and the story must NOT open
        the story; the anchor-desk shot at 28 is content_start."""
        docs, tm = self._shots()
        rows = stories.stories(self._ca(), docs, tm, None)
        story = rows[1]
        self.assertFalse(rows[0]["interstitial"] is False)
        self.assertTrue(rows[0]["interstitial"])
        self.assertEqual(story["start_sec"], 28.0)      # snapped to the cut
        self.assertEqual(story["content_start_sec"], 28.0)
        self.assertEqual(story["speech_start_sec"], 31.0)
        # the graphics tail joined the INTERSTITIAL side
        self.assertGreaterEqual(rows[0]["end_sec"], 28.0)

    def test_snap_never_crosses_previous_item(self):
        ca = {"audio_segments": [
            {"start_time": "00:00:00.000", "end_time": "00:00:29.500",
             "news_form": "package", "slug": "A", "segment_topic": "a"},
            {"start_time": "00:00:30.000", "end_time": "00:01:00.000",
             "news_form": "package", "slug": "B", "segment_topic": "b"},
        ]}
        docs = [{"shot_id": 0, "news_setting": "field"}]
        tm = {0: (25.0, 60.0)}       # story B's first cue sits mid-shot
        rows = stories.stories(ca, docs, tm, None)
        self.assertEqual(rows[1]["start_sec"], 30.0)    # snap refused

    def test_first_story_may_snap_to_the_programme_head(self):
        """No previous item exists for the first story — a shot starting
        at 0.0 is a legal snap target (the 0.0-initialized guard used to
        refuse it)."""
        ca = {"audio_segments": [
            {"start_time": "00:00:03.000", "end_time": "00:00:20.000",
             "news_form": "package", "slug": "A", "segment_topic": "a"}]}
        docs = [{"shot_id": 0, "news_setting": "field"}]
        tm = {0: (0.0, 25.0)}
        rows = stories.stories(ca, docs, tm, None)
        self.assertEqual(rows[0]["start_sec"], 0.0)

    def test_break_fence_keeps_the_pod_out_of_both_stories(self):
        """The measured tdy defect: a commercial pod between two stories
        (black+silent break candidates on BOTH edges) was absorbed into
        the next story's span. With fences, the pre-pod tail trails the
        previous story, the post-pod lead-in opens the next, and the pod
        shots join NEITHER."""
        ca = {"audio_segments": [
            {"start_time": "00:00:00.000", "end_time": "00:00:30.000",
             "news_form": "package", "slug": "A", "segment_topic": "a"},
            {"start_time": "00:02:20.000", "end_time": "00:03:00.000",
             "news_form": "package", "slug": "B", "segment_topic": "b"},
        ]}
        # gap 30-140: tail (30-34), black (34-36), POD (36-128),
        # black (128-131), lead-in (131-140)
        docs = [{"shot_id": i, "news_setting": "field"} for i in range(6)]
        tm = {0: (0.0, 30.0), 1: (30.0, 34.0), 2: (34.0, 36.0),
              3: (36.0, 128.0), 4: (128.0, 131.0), 5: (131.0, 140.0)}
        rows = stories.stories(ca, docs, tm, None, fences=[35.0, 129.5])
        self.assertEqual(rows[0]["end_sec"], 34.0)     # tail, not the pod
        self.assertEqual(rows[1]["start_sec"], 131.0)  # lead-in, not 30.0
        # without fences the same gap is absorbed (regression contrast)
        rows2 = stories.stories(ca, docs, tm, None)
        self.assertLess(rows2[1]["start_sec"], 131.0)

    def test_single_fence_splits_the_gap_at_the_seam(self):
        ca = {"audio_segments": [
            {"start_time": "00:00:00.000", "end_time": "00:00:30.000",
             "news_form": "package", "slug": "A", "segment_topic": "a"},
            {"start_time": "00:01:00.000", "end_time": "00:01:30.000",
             "news_form": "package", "slug": "B", "segment_topic": "b"},
        ]}
        docs = [{"shot_id": i, "news_setting": "field"} for i in range(3)]
        tm = {0: (0.0, 30.0), 1: (30.0, 44.0), 2: (44.0, 60.0)}
        rows = stories.stories(ca, docs, tm, None, fences=[44.0])
        self.assertEqual(rows[0]["end_sec"], 44.0)
        self.assertEqual(rows[1]["start_sec"], 44.0)

    def test_end_sec_treats_zero_as_a_real_timestamp(self):
        """Truthiness on timestamps is the defect class: a 0.0 end must
        survive, never crash max() on an empty filter."""
        ca = {"audio_segments": [
            {"start_time": "00:00:00.000", "end_time": "00:00:00.000",
             "news_form": "package", "slug": "A", "segment_topic": "a"}]}
        rows = stories.stories(ca, [], {}, None)
        self.assertEqual(rows[0]["end_sec"], 0.0)


class TestFetchPointers(unittest.TestCase):
    def test_artifact_key_tolerates_all_three_pointer_forms(self):
        """Run root, parent folder, full artifact uri — the ad-hoc
        per-branch checks double-appended on folder pointers and silently
        LOST the stem on a full-file pointer."""
        from states.assemble import lambda_function as assemble
        want = "run/transcripts/final_transcript.json"
        for key in ("run", "run/transcripts", want):
            self.assertEqual(
                assemble._artifact_key(key, "transcripts",
                                       "final_transcript.json"),
                want, key)
        self.assertEqual(
            assemble._artifact_key("run/stem_activity.json",
                                   "stem_activity.json"),
            "run/stem_activity.json")


class TestSequences(unittest.TestCase):
    def _scenes(self, specs):
        out = []
        for i, (start, end, setting, people) in enumerate(specs):
            out.append({"scene_id": i, "start_time": start, "end_time": end,
                        "setting": setting, "people": people,
                        "start_shot_id": i * 10, "end_shot_id": i * 10 + 9})
        return out

    def test_anchored_music_bridge(self):
        """The drama_b mega-sequence incident: an INHERITED bed must not
        bridge; a cue that STARTED in the earlier scene must."""
        scenes = self._scenes([(0, 30, "bar_restaurant", []),
                               (30, 60, "office", [])])
        inherited = {"activity": {"music": [[-100000, 65000]]}}
        anchored = {"activity": {"music": [[5000, 45000]]}}
        s1 = sequences.sequences(scenes, [], {}, None, inherited)
        s2 = sequences.sequences(scenes, [], {}, None, anchored)
        self.assertEqual(len(s1), 2)      # bed: no join
        self.assertEqual(len(s2), 1)      # cue: joins
        self.assertEqual(s2[0]["evidence"][0]["via"], "music_bridge")

    def test_micro_scene_rides_as_connective_tissue(self):
        scenes = self._scenes([(0, 30, "office", ["A"]),
                               (30, 32, None, []),          # a 2 s insert
                               (32, 60, "office", ["A"])])
        seqs = sequences.sequences(scenes, [], {}, None, None)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(seqs[0]["evidence"][0]["via"], "micro_scene")

    def test_cross_cut_threads(self):
        scenes = self._scenes([(0, 20, "office", ["A"]),
                               (20, 40, "street_sidewalk", ["B"]),
                               (40, 60, "office", ["A"]),
                               (60, 80, "street_sidewalk", ["B"])])
        seqs = sequences.sequences(scenes, [], {}, None, None)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(sorted(map(len, seqs[0]["threads"])), [2, 2])


class TestGaps(unittest.TestCase):
    def _transcript(self, word_times, duration=100.0):
        return {"audio_duration_sec": duration,
                "words": [{"word": f"w{i}", "start": a, "end": b}
                          for i, (a, b) in enumerate(word_times)],
                "sentences": []}

    def test_hard_windows_between_speech_with_pad(self):
        t = self._transcript([(0.0, 5.0), (20.0, 25.0)])
        gm = gaps.gap_map(t, [], [], None)
        hard = [w for w in gm["windows"] if w["kind"] == "hard"]
        self.assertEqual(hard[0]["start"], 5.3)      # PAD applied
        self.assertEqual(hard[0]["end"], 19.7)

    def test_soft_window_requires_wordless_stem_speech(self):
        """The graded-clearance predicate: stem speech with NO transcript
        words = walla. A span carrying even one word is NOT soft — the
        predicate is 'speech the transcript cannot hear', never 'speech we
        judged unimportant' (that judgment belongs to AD, and the mitj405
        measurement showed it is the DOMINANT professional mechanism —
        recorded, out of scope here)."""
        t = self._transcript([(0.0, 5.0), (40.0, 45.0)])
        stem = {"activity": {"speech": [[10000, 20000], [42000, 44000]]}}
        gm = gaps.gap_map(t, [], [], stem)
        soft = [w for w in gm["windows"] if w["kind"] == "soft"]
        self.assertEqual(len(soft), 1)
        self.assertAlmostEqual(soft[0]["start"], 10.1, places=1)
        self.assertEqual(soft[0]["evidence"], "untranscribed_speech")

    def test_orientation_dialogue_bridge_zeroes_sufficiency(self):
        scenes = [{"scene_id": 0, "start_time": 0.0, "end_time": 30.0,
                   "setting": "office", "people": ["A"]},
                  {"scene_id": 1, "start_time": 30.0, "end_time": 60.0,
                   "setting": "street_sidewalk", "people": ["B"]}]
        t = {"audio_duration_sec": 60.0,
             "words": [{"word": "x", "start": 29.0, "end": 31.0}],
             "sentences": [{"speaker_id": "S0", "start": 29.0}]}
        o = gaps.orientation(scenes, t, None)
        self.assertTrue(o[1]["dialogue_bridged"])
        self.assertEqual(o[1]["audio_sufficiency"], 0.0)


class TestGateway(unittest.TestCase):
    class _Ctx:
        def __init__(self, tool):
            custom = {"bedrockAgentCoreToolName":
                      f"program-structure-analysis___{tool}"}
            self.client_context = type("CC", (), {"custom": custom})()

    def test_refusals_and_contract(self):
        from unittest import mock
        import lambda_function as gw
        with mock.patch.object(gw, "PROXY_BUCKET", "b"), \
             mock.patch.object(gw, "STATE_MACHINE_ARN", "arn:x"):
            r = gw.lambda_handler({}, self._Ctx("analyze_program_structure"))
            self.assertEqual(r["errorCode"], "MISSING_PARAMETER")
            r = gw.lambda_handler(
                {"shot_segmentation_s3uri": "s3://b/x",
                 "shot_metadata_s3uri": "s3://b/y", "mode": "banana"},
                self._Ctx("analyze_program_structure"))
            self.assertEqual(r["errorCode"], "INVALID_PARAMETER")
        r = gw.lambda_handler({}, self._Ctx("describe_tool"))
        self.assertEqual(sorted(r["output_contract"]["outputs"]),
                         ["manifest", "rundown", "structure"])

    def test_target_and_tool_names_fit_the_64_char_rule(self):
        for tool in ("analyze_program_structure", "get_job_status",
                     "describe_tool"):
            self.assertLessEqual(
                len(f"program-structure-analysis___{tool}"), 64)

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        path = os.path.join(_ROOT, "gateway_tool", "input_examples.json")
        examples = _load_json(path)["tools"]
        self.assertEqual(sorted(examples),
                         ["analyze_program_structure", "describe_tool",
                          "get_job_status"])
        for tool, entry in examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        import lambda_function as gw
        r = gw.lambda_handler({}, self._Ctx("describe_tool"))
        self.assertIn("analyze_program_structure",
                      r["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        from unittest import mock
        import lambda_function as gw
        path = os.path.join(_ROOT, "gateway_tool", "input_examples.json")
        example = _load_json(path)["tools"][
            "analyze_program_structure"]["example"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gw, "_sfn", fake):
            r = gw.lambda_handler(dict(example),
                                  self._Ctx("analyze_program_structure"))
        self.assertNotIn("error", r, f"example refused: {r}")
        self.assertEqual(r["status"], "RUNNING")
        fake.start_execution.assert_called_once()


class TestValidatePlanShapes(unittest.TestCase):
    def test_news_without_ca_refuses(self):
        from states.validate_plan import lambda_function as validate_plan
        with self.assertRaises(validate_plan.ProgramStructureValidationError):
            validate_plan.lambda_handler(
                {"uuid": "u", "input": {
                    "shot_segmentation_s3uri": "s3://b/x",
                    "shot_metadata_s3uri": "s3://b/y",
                    "mode": "news"}}, None)


class TestCopiedModulesAreByteIdentical(unittest.TestCase):
    MASTERS = {"gateway_tool/workflow_identity.py":
               "../conversation-analysis-on-aws/gateway_tool/"
               "workflow_identity.py"}

    def test_pinned(self):
        for local, master in self.MASTERS.items():
            path = os.path.join(_ROOT, master)
            if not os.path.exists(path):
                continue
            with open(os.path.join(_ROOT, local), "rb") as a, \
                 open(path, "rb") as b:
                self.assertEqual(a.read(), b.read(),
                                 f"{local} drifted from {master}")


class TestAslDrift(unittest.TestCase):
    def test_cfnstack_inline_matches_definitions_json(self):
        import re
        text = _read(os.path.join(_ROOT, "cfnstack.yaml"))
        m = re.search(r"DefinitionString: \|-\n((?:                .*\n|\n)+)",
                      text)
        inline = json.loads("\n".join(l[16:] for l in m.group(1).splitlines()))
        ref = _load_json(os.path.join(_ROOT, "definitions.json"))
        self.assertEqual(inline, ref)


if __name__ == "__main__":
    unittest.main()
