"""Workflow-layer tests: gateway steering validation, router routing,
brief-amendment firewall, vision-verdict application, ASL parity.
Invented fixtures only. Run: python3 -m unittest discover -s test"""

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


from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# gateway modules import as top-level names (their zip is flat), and
# the tests import the dispatcher by its module name — gateway_tool
# goes FIRST so `import lambda_function` resolves there
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))
# the gateway's flat zip ships lib/profiles.py beside it (archetype
# validation at the door) — its runtime resolves `profiles` top-level
sys.path.append(os.path.join(ROOT, "lib"))

# env the workflow/gateway modules read at import time (region: boto3
# clients are created at module import)
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("STATE_MACHINE_ARN",
                      "arn:aws:states:eu-west-1:000000000000:"
                      "stateMachine:test-artwork-selection")
os.environ.setdefault("TOOL_VERSION", "test")

import lambda_function as gw                                # noqa: E402
from states.assemble_artwork_candidates import lambda_function as asm  # noqa: E402
from states.filter_rank_candidate_shots import lambda_function as frs  # noqa: E402
import importlib.util as _ilu                               # noqa: E402
_spec = _ilu.spec_from_file_location(
    "artsel_router", os.path.join(ROOT, "lambda_function.py"))
router = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(router)



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


class TestGatewaySteering(unittest.TestCase):
    def norm(self, **event):
        request = {}
        err = gw._normalize_steering(event, request)
        return err, request

    def test_defaults(self):
        err, req = self.norm()
        self.assertIsNone(err)
        self.assertEqual(req["candidates_per_archetype"], 4)
        self.assertEqual(req["spoiler_cutoff"], 0.75)
        self.assertIsNone(req["archetypes"])
        self.assertEqual(req["selection_brief"], "")

    def test_unknown_archetype_refused(self):
        err, _ = self.norm(archetypes=["hero_closeup", "nope"])
        self.assertTrue(err and err["error"])
        self.assertIn("nope", err["message"])

    def test_archetype_list_shape(self):
        for bad in ([], "hero_closeup", ["hero_closeup", "hero_closeup"],
                    [1]):
            err, _ = self.norm(archetypes=bad)
            self.assertTrue(err and err["error"], bad)
        err, req = self.norm(archetypes=["two_shot", "hero_closeup"])
        self.assertIsNone(err)
        # caller order preserved: order is slate priority (dedup)
        self.assertEqual(req["archetypes"], ["two_shot", "hero_closeup"])

    def test_count_and_cutoff_bounds(self):
        for bad in (0, 11, 2.5, "four", None):
            err, _ = self.norm(candidates_per_archetype=bad)
            self.assertTrue(err and err["error"], bad)
        for bad in (0, -0.1, 1.5, "most"):
            err, _ = self.norm(spoiler_cutoff=bad)
            self.assertTrue(err and err["error"], bad)
        err, req = self.norm(spoiler_cutoff=1)
        self.assertIsNone(err)
        self.assertEqual(req["spoiler_cutoff"], 1.0)

    def test_brief_ceiling(self):
        err, _ = self.norm(selection_brief="x" * 2001)
        self.assertTrue(err and err["error"])

    def test_model_chain_default_and_validation(self):
        err, req = self.norm()
        self.assertIsNone(err)
        self.assertEqual(req["model_chain"],
                         ["anthropic.claude-sonnet-4-6",
                          "amazon.nova-2-lite-v1:0"])
        for bad in ([], "sonnet", [""], ["a"] * 5, [1], None):
            err, _ = self.norm(model_chain=bad)
            self.assertTrue(err and err["error"], bad)
        err, req = self.norm(model_chain=[" anthropic.claude-opus-5 "])
        self.assertIsNone(err)
        self.assertEqual(req["model_chain"],
                         ["anthropic.claude-opus-5"])

    def test_use_batch_default_and_validation(self):
        err, req = self.norm()
        self.assertIsNone(err)
        self.assertIs(req["use_batch"], False)
        for bad in ("true", 1, None):
            err, _ = self.norm(use_batch=bad)
            self.assertTrue(err and err["error"], bad)
        err, req = self.norm(use_batch=True)
        self.assertIsNone(err)
        self.assertIs(req["use_batch"], True)

    def test_full_res_refused(self):
        resp = gw._select_artwork({"full_res": True})
        self.assertTrue(resp["error"])
        self.assertIn("full_res", resp["message"])
        resp = gw._select_artwork({"asset_s3_uri": "s3://b/k"})
        self.assertTrue(resp["error"])

    def test_required_sources(self):
        resp = gw._select_artwork(
            {"shot_segmentation_s3uri": "s3://b/micro-workflow/ss/1"})
        self.assertTrue(resp["error"])
        self.assertIn("shot_metadata_s3uri", resp["message"])

    class _Ctx:
        def __init__(self, tool):
            custom = {"bedrockAgentCoreToolName":
                      f"artwork-selection___{tool}"}
            self.client_context = type("CC", (), {"custom": custom})()

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        path = os.path.join(ROOT, "gateway_tool", "input_examples.json")
        examples = _load_json(path)["tools"]
        self.assertEqual(sorted(examples),
                         ["describe_tool", "get_job_status",
                          "select_artwork"])
        for tool, entry in examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        out = gw.lambda_handler({}, self._Ctx("describe_tool"))
        self.assertIn("select_artwork", out["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        path = os.path.join(ROOT, "gateway_tool", "input_examples.json")
        example = _load_json(path)["tools"]["select_artwork"]["example"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gw, "_sfn", fake):
            out = gw.lambda_handler(dict(example),
                                    self._Ctx("select_artwork"))
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()


class TestRouter(unittest.TestCase):
    def test_unknown_operation_raises(self):
        with self.assertRaises(ValueError):
            router.lambda_handler({"operation": "Approve"}, None)

    def test_all_asl_operations_routed(self):
        asl = _load_json(os.path.join(ROOT, "definitions.json"))
        ops = {s["Parameters"]["Payload"]["operation"]
               for s in asl["States"].values()}
        self.assertEqual(ops, set(router._OPERATIONS))


class TestBriefFirewall(unittest.TestCase):
    def amendments(self, payload):
        result = {"status": "SUCCEEDED", "output_s3_prefix": "s3://b/p"}
        with mock.patch.object(frs.s3io, "get_response_texts",
                               return_value=[json.dumps(payload)]):
            return frs._brief_amendments(result)

    def test_skipped_result_is_empty(self):
        self.assertEqual(frs._brief_amendments({"skipped": "no_brief"}),
                         {})
        self.assertEqual(frs._brief_amendments(None), {})

    def test_unknown_archetypes_dropped(self):
        out = self.amendments({"archetype_counts":
                               {"hero_closeup": 2, "invented": 9}})
        self.assertEqual(out["archetype_counts"], {"hero_closeup": 2})

    def test_counts_bounded(self):
        out = self.amendments({"archetype_counts":
                               {"hero_closeup": 99, "two_shot": 0}})
        self.assertNotIn("archetype_counts", out)

    def test_people_and_emphasis_pass(self):
        out = self.amendments({"must_include_people": ["A", "", "B"],
                               "emphasis": "warm tones",
                               "spoiler_cutoff": 0.2})
        self.assertEqual(out["must_include_people"], ["A", "B"])
        self.assertEqual(out["emphasis"], "warm tones")
        # the brief can NEVER move the spoiler cut
        self.assertNotIn("spoiler_cutoff", out)

    def test_garbage_degrades_to_empty(self):
        result = {"status": "SUCCEEDED", "output_s3_prefix": "s3://b/p"}
        with mock.patch.object(frs.s3io, "get_response_texts",
                               return_value=["not json at all"]):
            self.assertEqual(frs._brief_amendments(result), {})


class TestIdentityCascade(unittest.TestCase):
    # the 2026-10-05 gap: cast present but all persons unnamed must
    # CASCADE to the face-recognition fallback, not stop at the elif
    SHOTS = {1: {"shot_id": 1, "start_time": 0.0, "end_time": 10.0}}

    @staticmethod
    def _get_json(payloads):
        return lambda uri: next(v for k, v in payloads.items()
                                if k in uri)

    def test_unnamed_cast_falls_through_to_faces(self):
        inputs = {"cast_resolution_s3uri": "s3://x/cast",
                  "face_recognition_s3uri": "s3://x/fr"}
        payloads = {
            "/cast": {"persons": [{"person_id": 1, "name": None,
                                   "screen_sec": 50.0}]},
            "/fr": {"people": [{
                "person_id": 7, "named": False, "name": "Person_7",
                "total_screen_time_sec": 9.0,
                "appearances": [{"begin_ms": 0, "end_ms": 9000}]}]},
        }
        names, presence, source = frs.resolve_identity(
            inputs, self.SHOTS, self._get_json(payloads))
        self.assertEqual(names, ["person_7"])
        self.assertEqual(presence, {1: {"person_7"}})
        self.assertEqual(source, "face_recognition")

    def test_named_cast_wins_without_fr_read(self):
        inputs = {"cast_resolution_s3uri": "s3://x/cast",
                  "face_recognition_s3uri": "s3://x/fr"}
        payloads = {"/cast": {"persons": [
            {"person_id": 1, "name": "A", "screen_sec": 50.0}]}}
        names, presence, source = frs.resolve_identity(
            inputs, self.SHOTS, self._get_json(payloads))
        self.assertEqual((names, presence, source),
                         (["A"], None, "cast_resolution"))

    def test_no_identity_anywhere(self):
        names, presence, source = frs.resolve_identity(
            {}, self.SHOTS, self._get_json({}))
        self.assertEqual((names, presence, source), ([], None, None))


class TestEmptyPersonSlateFallback(unittest.TestCase):
    # user direction 2026-10-05: hero/two_shot never ship empty while
    # face identity exists — priority casts, THEN top faces
    SHOTS = {i: {"shot_id": i, "start_time": i * 100.0,
                 "end_time": i * 100.0 + 10.0,
                 "frame_motion_mean": 0.1, "entropy_mean": 5.0}
             for i in range(1, 4)}
    DOCS = {1: {"shot_id": 1, "shot_kind": "normal",
                "camera_shot_type": "close_up", "people_count": "1",
                "people": ["Nobody Matching"], "texts_on_screen": [],
                "unreadable_text_present": False, "brands": [],
                "depicted": [], "description": "s1"},
            2: {"shot_id": 2, "shot_kind": "normal",
                "camera_shot_type": "medium", "people_count": "2",
                "people": [], "texts_on_screen": [],
                "unreadable_text_present": False, "brands": [],
                "depicted": [], "description": "s2"},
            3: {"shot_id": 3, "shot_kind": "normal",
                "camera_shot_type": "aerial", "people_count": "0",
                "people": [], "texts_on_screen": [],
                "unreadable_text_present": False, "brands": [],
                "depicted": [], "description": "s3"}}
    FR = {"people": [{
        "person_id": 7, "named": False, "name": "Person_7",
        "total_screen_time_sec": 20.0,
        "appearances": [{"begin_ms": 100000, "end_ms": 110000},
                        {"begin_ms": 200000, "end_ms": 210000}]}]}
    INPUTS = {"face_recognition_s3uri": "s3://x/fr"}

    def _get_json(self, uri):
        assert "/fr" in uri
        return self.FR

    def test_empty_person_slates_refill_from_faces(self):
        # cast named someone SME never spells the same way: the name
        # join misses, hero/two_shot come back [] — the fallback
        # refills BOTH from the face-track join; vista is untouched
        counts = {"hero_closeup": 4, "two_shot": 4,
                  "establishing_vista": 4}
        slates = frs.select_candidates(
            self.SHOTS, self.DOCS, ["Dr Lena Voss"], counts, 900.0)
        self.assertEqual(slates["hero_closeup"], [])
        used = frs.fill_empty_person_slates(
            slates, counts, self.SHOTS, self.DOCS, 900.0, (),
            self.INPUTS, self._get_json)
        self.assertEqual(used, {"hero_closeup": "face_recognition",
                                "two_shot": "face_recognition"})
        self.assertEqual([c["shot_id"] for c in slates["hero_closeup"]],
                         [1])
        self.assertEqual([c["shot_id"] for c in slates["two_shot"]],
                         [2])

    def test_partial_slates_are_left_alone(self):
        counts = {"hero_closeup": 4}
        slates = {"hero_closeup": [{"shot_id": 1}]}
        used = frs.fill_empty_person_slates(
            slates, counts, self.SHOTS, self.DOCS, 900.0, (),
            self.INPUTS, self._get_json)
        self.assertEqual(used, {})
        self.assertEqual(slates["hero_closeup"], [{"shot_id": 1}])

    def test_fallback_never_reelects_taken_shots(self):
        # shot 1 already placed by the primary pass under another
        # archetype: the refill must skip it
        counts = {"hero_closeup": 4, "other": 4}
        frs.ARCHETYPES["other"] = {"camera": ("close_up",),
                                   "people_count": ("1",),
                                   "people": None}
        try:
            slates = {"other": [{"shot_id": 1}], "hero_closeup": []}
            used = frs.fill_empty_person_slates(
                slates, counts, self.SHOTS, self.DOCS, 900.0, (),
                self.INPUTS, self._get_json)
        finally:
            del frs.ARCHETYPES["other"]
        self.assertEqual(used, {})
        self.assertEqual(slates["hero_closeup"], [])


class TestExcludedSpans(unittest.TestCase):
    def test_all_marker_categories_union(self):
        structure = {"markers": {
            "end_credits": [{"start": 100.0, "end": 160.0}],
            "bars_and_tone": [{"start": 0.0, "end": 30.0}],
            "opening_title": [],
            "oddball": [{"no_start": True}],
        }}
        spans = frs._excluded_spans(structure)
        self.assertEqual(sorted(spans), [(0.0, 30.0), (100.0, 160.0)])


class TestFrameChoice(unittest.TestCase):
    # the FR-informed grab (2026-10-06): face attributes pick the
    # members; no usable evidence degrades to a spread, never a
    # single blind midpoint
    from states.grab_candidate_frames import lambda_function as gcf
    NAMES = [f"frame_{n:010d}.jpg" for n in range(3000, 3300, 30)]
    # 10 members, 30 fps -> timestamps 100.0 .. 109.0

    def test_good_moments_pick_nearest_members(self):
        got = self.gcf.choose_members(
            self.NAMES, 30.0,
            [(104.05, 90.0), (101.0, 50.0), (104.1, 80.0)], 3)
        # members scored by distance to their nearest good moment
        # (member@101 exact hit, member@104 0.05s off); members more
        # than one sampling step from every moment never qualify
        self.assertEqual(got, ["frame_0000003030.jpg",
                               "frame_0000003120.jpg",
                               "frame_0000003150.jpg"])

    def test_no_evidence_degrades_to_spread(self):
        got = self.gcf.choose_members(self.NAMES, 30.0, [], 3)
        self.assertEqual(got, ["frame_0000003150.jpg",
                               "frame_0000003060.jpg",
                               "frame_0000003210.jpg"])
        # out-of-window moments are no evidence either
        got = self.gcf.choose_members(self.NAMES, 30.0,
                                      [(500.0, 99.0)], 3)
        self.assertEqual(got[0], "frame_0000003150.jpg")

    def test_derive_fps_and_good_times(self):
        fg = {"people": [{"tracks": [{"faces": [
            {"timestamp": 10.0, "frame_num": 300, "eyes_open": True,
             "sharpness": 5.0},
            {"timestamp": 20.0, "frame_num": 600, "eyes_open": True,
             "sunglasses": True, "sharpness": 9.0},
            {"timestamp": 30.0, "frame_num": 900, "eyes_open": None,
             "sharpness": 9.0},
            {"timestamp": 40.0, "frame_num": 1200, "eyes_open": True,
             "face_occluded": True},
        ]}]}]}
        self.assertEqual(self.gcf.derive_fps(fg), 30.0)
        self.assertEqual(self.gcf.good_face_times(fg), [(10.0, 5.0)])


class TestAssembleApply(unittest.TestCase):
    def cand(self, sid):
        return {"shot_id": sid, "frame": f"s3://b/p/frames/a_{sid}.jpg"}

    def test_multiframe_candidate_best_frame_wins(self):
        from states.assemble_artwork_candidates import lambda_function as asm
        cands = [
            {"shot_id": 1, "frame": "s3://x/a_shot0001_f0.jpg",
             "frames": ["s3://x/a_shot0001_f0.jpg",
                        "s3://x/a_shot0001_f1.jpg"]},
            {"shot_id": 2, "frame": "s3://x/a_shot0002.jpg"},
        ]
        group = {"ranking": ["a_shot0002.jpg", "a_shot0001_f1.jpg",
                             "a_shot0001_f0.jpg"],
                 "top_reason": "r",
                 "unusable": [{"file": "a_shot0001_f0.jpg",
                               "why": "blink"}]}
        asm._apply(cands, group)
        # candidate 2's single frame outranks candidate 1's best
        self.assertEqual([c["shot_id"] for c in cands], [2, 1])
        # candidate 1's THE frame is its best-ranked sibling, and one
        # flagged sibling does not make the candidate unusable
        self.assertEqual(cands[1]["frame"], "s3://x/a_shot0001_f1.jpg")
        self.assertNotIn("unusable", cands[1])
        # all siblings flagged -> unusable
        group["unusable"].append({"file": "a_shot0001_f1.jpg",
                                  "why": "blur"})
        asm._apply(cands, group)
        self.assertEqual(cands[1]["unusable"], "blur")

    def test_vision_reorders_and_flags(self):
        cands = [self.cand(1), self.cand(2), self.cand(3)]
        block = asm._apply(cands, {
            "ranking": ["a_3.jpg", "a_1.jpg"],
            "top_reason": "sharpest",
            "unusable": [{"file": "a_2.jpg", "why": "eyes shut"}]})
        self.assertEqual([c["shot_id"] for c in cands], [3, 1, 2])
        self.assertEqual(cands[2]["unusable"], "eyes shut")
        self.assertNotIn("unusable", cands[0])
        self.assertEqual(block["n_unusable"], 1)

    def test_skipped_rerank_reported(self):
        verdict, rerank = asm._verdict({"skipped": "no_frames"})
        self.assertIsNone(verdict)
        self.assertEqual(rerank, {"status": "skipped",
                                  "reason": "no_frames"})

    def test_failed_rerank_reported(self):
        verdict, rerank = asm._verdict(
            {"status": "FAILED", "output_s3_prefix": "s3://b/p"})
        self.assertIsNone(verdict)
        self.assertEqual(rerank["status"], "failed")


class TestAslParity(unittest.TestCase):
    def test_inline_matches_copy(self):
        import re
        y = _read(os.path.join(ROOT, "cfnstack.yaml"))
        m = re.search(r"DefinitionString: \|-\n(.*?)\nOutputs:",
                      y, re.DOTALL)
        inline = json.loads("\n".join(ln[16:]
                                      for ln in m.group(1).splitlines()))
        copy = _load_json(os.path.join(ROOT, "definitions.json"))
        self.assertEqual(inline, copy,
                         "inline ASL drifted from definitions.json — "
                         "edit BOTH")



class TestGatewayZipColdImport(unittest.TestCase):
    # the deployed gateway sees ONLY its zip members (caught live
    # 2026-10-06: a trimmed gateway zip ImportError'd on wf_common while
    # the suite passed, because tests put lib/ on the path) — cold-
    # import the dispatcher from a dir holding exactly what ships.
    # MEMBERS mirrors build-deploy-s3-dist.sh's gateway member list.
    MEMBERS = ("gateway_tool/lambda_function.py", "gateway_tool/workflow_identity.py",
               "gateway_tool/output_contract.json", "gateway_tool/input_examples.json",
               "lib/profiles.py", "lib/wf_common.py")

    def test_dispatcher_imports_from_zip_members_alone(self):
        import shutil
        import subprocess
        import tempfile
        d = tempfile.mkdtemp()
        root = ROOT
        for m in self.MEMBERS:
            shutil.copy(os.path.join(root, m), d)
        env = dict(os.environ)
        env.setdefault("STATE_MACHINE_ARN", "arn:aws:states:eu-west-1:0:stateMachine:t")
        env.setdefault("PROXY_BUCKET", "b")
        env.setdefault("TOOL_VERSION", "t")
        r = subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "import lambda_function",
             d],
            capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])

if __name__ == "__main__":
    unittest.main()
