"""
Unit tests — shot-metadata-enrichment-on-aws (README.md §3-§5, §10). No AWS.

The windowing tests run against the REAL Meridian shots.json (97 shots,
the asset every prompt validation ran on): at the designed 1-3 frame tier
a 10-shot window can exceed 20 images (the 2026-08-18 latent-defect
finding), so the planner's budget behaviour is pinned against real
durations, not synthetic ones.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tarfile
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
# lib/ ships flat at the zip root, so its modules are TOP-LEVEL at runtime.
sys.path.insert(0, ROOT)

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")   # hermetic clients
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("BEDROCK_GATEWAY_ARN",
                      "arn:aws:lambda:eu-west-1:1:function:bedrock-gw")
os.environ.setdefault("PROMPT_REGISTRY_ARN",
                      "arn:aws:lambda:eu-west-1:1:function:registry-gw")
os.environ.setdefault("STATE_MACHINE_ARN",
                      "arn:aws:states:eu-west-1:1:stateMachine:"
                      "test-shot-metadata-enrichment")
os.environ.setdefault("TOOL_VERSION", "test.0")

from lib import faces as faces_mod                     # noqa: E402
from lib import output_parse                           # noqa: E402
from lib import scene_assembly                         # noqa: E402
from lib import sme_common                             # noqa: E402
from lib import windowing                              # noqa: E402

FIXTURES = os.path.join(ROOT, "test", "fixtures")
TIER = sme_common.FRAME_TIER


def meridian_shots() -> list:
    with open(os.path.join(FIXTURES, "meridian_shots.json"),
              encoding="utf-8") as f:
        return sorted(json.load(f)["shots"], key=lambda s: s["shot_id"])


def make_shot(sid, start, end, n_frames=8):
    return {"shot_id": sid, "start_time": start, "end_time": end,
            "start_timecode": f"00:00:{sid:02d}:00",
            "end_timecode": f"00:00:{sid:02d}:10",
            "selected_frames": list(range(n_frames)),
            "selected_frame_count": n_frames,
            "frames_key": f"runs/shot_{sid:04d}_frames.tar.gz"}


# ─────────────────────────────────────────────────────────────────────────────
# Windowing (README §3 — image-bounded, 10-shot target, settled 2026-08-19)
# ─────────────────────────────────────────────────────────────────────────────


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


class TestFrameTier(unittest.TestCase):
    def test_tier_boundaries(self):
        self.assertEqual(windowing.frames_for_shot(
            make_shot(0, 0.0, 3.9), TIER), 1)
        self.assertEqual(windowing.frames_for_shot(
            make_shot(0, 0.0, 4.0), TIER), 2)
        self.assertEqual(windowing.frames_for_shot(
            make_shot(0, 0.0, 11.9), TIER), 2)
        self.assertEqual(windowing.frames_for_shot(
            make_shot(0, 0.0, 12.0), TIER), 3)

    def test_capped_by_archive_contents(self):
        # The plan can never promise a frame the tarball does not hold.
        self.assertEqual(windowing.frames_for_shot(
            make_shot(0, 0.0, 30.0, n_frames=2), TIER), 2)
        self.assertEqual(windowing.frames_for_shot(
            make_shot(0, 0.0, 30.0, n_frames=1), TIER), 1)


class TestPlanWindowsMeridian(unittest.TestCase):
    """The real asset: 97 shots, 13 of them in the 3-frame tier."""

    def setUp(self):
        self.shots = meridian_shots()

    def test_every_window_fits_the_budget(self):
        for gallery in (set(), {10}):
            windows = windowing.plan_windows(
                self.shots, shot_target=10, image_budget=20, tier=TIER,
                gallery_shots=gallery)
            for w in windows:
                frames = sum(w["frames"].values())
                total = frames + (1 if gallery else 0)
                self.assertLessEqual(
                    total, 20,
                    f"window {w['index']} would render {total} blocks")

    def test_coverage_is_total_and_ordered(self):
        windows = windowing.plan_windows(
            self.shots, shot_target=10, image_budget=20, tier=TIER)
        covered = [sid for w in windows for sid in w["shot_ids"]]
        self.assertEqual(covered, [s["shot_id"] for s in self.shots])

    def test_shot_count_floats_downward_never_above_target(self):
        windows = windowing.plan_windows(
            self.shots, shot_target=10, image_budget=20, tier=TIER)
        sizes = [len(w["shot_ids"]) for w in windows]
        self.assertTrue(all(n <= 10 for n in sizes))
        # The 3-frame tier binds somewhere on this asset (README §3: at the
        # designed tier, 10-shot windows exceed 20 images on real content).
        self.assertTrue(any(n < 10 for n in sizes),
                        "expected the image budget to bind at least once")

    def test_context_shot_is_previous_and_first_has_none(self):
        windows = windowing.plan_windows(
            self.shots, shot_target=10, image_budget=20, tier=TIER)
        self.assertIsNone(windows[0]["context_shot_id"])
        for prev, w in zip(windows, windows[1:]):
            self.assertEqual(w["context_shot_id"], prev["shot_ids"][-1])
            self.assertIn(w["context_shot_id"], w["frames"])

    def test_gallery_reservation_shrinks_the_frame_room(self):
        plain = windowing.plan_windows(
            self.shots, shot_target=10, image_budget=20, tier=TIER)
        with_g = windowing.plan_windows(
            self.shots, shot_target=10, image_budget=20, tier=TIER,
            gallery_shots={0})
        self.assertGreaterEqual(len(with_g), len(plain))


class TestPlanWindowsMechanics(unittest.TestCase):
    def test_impossible_budget_refuses_loudly(self):
        shots = [make_shot(0, 0.0, 30.0), make_shot(1, 30.0, 60.0)]
        with self.assertRaises(ValueError):
            windowing.plan_windows(shots, shot_target=10, image_budget=4,
                                   tier=TIER, gallery_shots={0})


class TestBurstFold(unittest.TestCase):
    """Burst-run folding (README §11): every member keeps its REQUIRED entry
    and its output document; only first/middle/last carry frame images —
    the members of a run are visually near-interchangeable (a jingle's
    cards measure frame_coherence ~0.99), and each frame is the dominant
    token cost. Measured on jingle.mp4: 55 member frames fold to 9."""

    RUNS = [{"start_shot_id": 3, "end_shot_id": 12},     # 10 members
            {"start_shot_id": 20, "end_shot_id": 26}]    # 7 members

    def test_representatives_are_first_middle_last(self):
        run_of, frameless = windowing.burst_fold(self.RUNS, 3)
        self.assertEqual({3: 0, 12: 0}, {k: run_of[k] for k in (3, 12)})
        self.assertEqual(0, run_of[8])
        self.assertEqual(1, run_of[23])
        # run 0: keep {3, 8, 12}; run 1: keep {20, 23, 26}
        self.assertNotIn(3, frameless)
        self.assertNotIn(8, frameless)
        self.assertNotIn(12, frameless)
        self.assertEqual({4, 5, 6, 7, 9, 10, 11},
                         frameless & set(range(3, 13)))
        self.assertEqual({21, 22, 24, 25},
                         frameless & set(range(20, 27)))

    def test_zero_representatives_disables_folding_but_keeps_membership(self):
        run_of, frameless = windowing.burst_fold(self.RUNS, 0)
        self.assertEqual(set(), frameless,
                         "the knob disables folding, never the annotation")
        self.assertEqual(17, len(run_of))

    def test_frameless_members_cost_no_images_but_hold_required_slots(self):
        shots = [make_shot(i, i * 0.5, (i + 1) * 0.5) for i in range(14)]
        _, frameless = windowing.burst_fold(
            [{"start_shot_id": 0, "end_shot_id": 13}], 3)
        windows = windowing.plan_windows(
            shots, shot_target=10, image_budget=20, tier=TIER,
            frameless=frameless)
        w0 = windows[0]
        self.assertEqual(10, len(w0["shot_ids"]),
                         "frameless members still count against shot_target")
        self.assertEqual(0, w0["frames"][4])
        self.assertEqual(1, w0["frames"][0])   # representative
        self.assertEqual(sum(w0["frames"].values()), w0["image_blocks"])
        # Across all windows: the three representatives carry frames, plus
        # the second window's CONTEXT shot — a folded member, but context is
        # EXEMPT from folding (it exists solely to give the first required
        # shot's boundary judgment visual evidence).
        w1 = windows[1]
        self.assertEqual(1, w1["frames"][w1["context_shot_id"]],
                         "context is exempt from burst folding")
        every = [f for w in windows for f in w["frames"].values()]
        self.assertEqual(4, sum(1 for f in every if f > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Faces (README §5 — the join, the roster, the sentinels)
# ─────────────────────────────────────────────────────────────────────────────

FACEMETA = {
    "people": [
        {"person_id": 0, "name": "Ray Halloran", "named": True,
         "rep_face_key": "fr/people/person_0000_face.jpg",
         "total_screen_time_sec": 120.0,
         "appearances": [{"begin_ms": 0, "end_ms": 14_000}]},
        {"person_id": 1, "name": "Danny Kovac", "named": True,
         "rep_face_key": "fr/people/person_0001_face.jpg",
         "total_screen_time_sec": 80.0,
         "appearances": [{"begin_ms": 10_000, "end_ms": 30_000}]},
        {"person_id": 2, "name": "Person_2", "named": False,
         "rep_face_key": "fr/people/person_0002_face.jpg",
         "total_screen_time_sec": 500.0,
         "appearances": [{"begin_ms": 0, "end_ms": 90_000}]},
    ],
}


class TestFacesJoin(unittest.TestCase):
    def setUp(self):
        self.people = faces_mod.named_people(FACEMETA)
        self.shots = [make_shot(0, 0.0, 9.0), make_shot(1, 9.0, 15.0),
                      make_shot(2, 15.0, 25.0), make_shot(3, 40.0, 50.0)]

    def test_unnamed_people_never_participate(self):
        self.assertEqual([p["name"] for p in self.people],
                         ["Ray Halloran", "Danny Kovac"])

    def test_join_is_span_overlap(self):
        names = faces_mod.shots_with_faces(self.shots, self.people)
        self.assertEqual(names[0], ["Ray Halloran"])
        self.assertEqual(sorted(names[1]),
                         ["Danny Kovac", "Ray Halloran"])
        self.assertEqual(names[2], ["Danny Kovac"])
        self.assertNotIn(3, names)      # no overlap → no entry, no []

    def test_roster_ranks_by_window_appearances_and_caps(self):
        names = faces_mod.shots_with_faces(self.shots, self.people)
        roster = faces_mod.window_gallery([0, 1], names, self.people, 5)
        self.assertEqual([p["name"] for p in roster],
                         ["Ray Halloran", "Danny Kovac"])
        capped = faces_mod.window_gallery([0, 1], names, self.people, 1)
        self.assertEqual([p["name"] for p in capped], ["Ray Halloran"])

    def test_face_names_text_resolved_per_shot(self):
        names = faces_mod.shots_with_faces(self.shots, self.people)
        text = faces_mod.face_names_text([1, 2], 0, names)
        self.assertIn("shot 0: Ray Halloran", text)
        self.assertIn("shot 1: Ray Halloran, Danny Kovac", text)
        self.assertIn("shot 2: Danny Kovac", text)
        self.assertIn("not listed", text)

    def test_sentinel_never_an_empty_string(self):
        text = faces_mod.face_names_text([3], None, {})
        self.assertEqual(
            text, "(no face recognition supplied for this window)")


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing + the §4 pairing repair (false wins, BOTH directions)
# ─────────────────────────────────────────────────────────────────────────────

def doc_line(sid, flag=False, reason="continues_action", **kw):
    d = {"shot_id": sid, "description": f"shot {sid}",
         "starts_new_scene": flag, "boundary_reason": reason,
         "shot_kind": "normal", "camera_shot_type": "medium",
         "setting": "office", "time_of_day": "day", "lighting": [],
         "notable": [], "people_count": "1", "people": [],
         "texts_on_screen": [], "unreadable_text_present": False,
         "named_entities": [], "brands": [], "objects": [],
         "sentiment": "neutral"}
    d.update(kw)
    return json.dumps(d)


class TestOutputParse(unittest.TestCase):
    def test_fences_tolerated_and_a_truncated_tail_is_a_partial_not_a_gap(self):
        text = "```jsonl\n" + doc_line(1) + "\n" + doc_line(2) + "\n" \
            + doc_line(3)[:40] + "\n```"
        parsed = output_parse.parse_response(text)
        # The cut line kept its join key, so the prefix is KEPT (user
        # decision 2026-08-26: minimal gate) — and counted, never silent.
        self.assertEqual(sorted(parsed["docs"]), [1, 2, 3])
        self.assertEqual(parsed["line_repairs"], {"prefix": 1})
        self.assertEqual(parsed["docs"][3]["shot_id"], 3)
        self.assertEqual(parsed["bad_lines"], [])

    def test_duplicate_keeps_first(self):
        text = doc_line(5, setting="office") + "\n" \
            + doc_line(5, setting="kitchen")
        parsed = output_parse.parse_response(text)
        self.assertEqual(parsed["docs"][5]["setting"], "office")
        self.assertEqual(parsed["duplicates"], [5])

    def test_repair_true_flag_false_reason_forces_false(self):
        d = json.loads(doc_line(1, flag=True, reason="intercut_return"))
        r = output_parse.repair_pairing(d)
        self.assertTrue(r["repaired"])
        self.assertIs(d["starts_new_scene"], False)
        self.assertEqual(d["boundary_reason"], "intercut_return")  # verbatim

    def test_repair_false_flag_true_reason_keeps_false(self):
        # The Meridian shot-75 shape (v7 re-validation): the reason is a
        # true observation (after_black) that is not a boundary.
        d = json.loads(doc_line(75, flag=False, reason="after_black"))
        r = output_parse.repair_pairing(d)
        self.assertTrue(r["repaired"])
        self.assertEqual(r["kind"], "false_flag_true_reason")
        self.assertIs(d["starts_new_scene"], False)
        self.assertEqual(d["boundary_reason"], "after_black")      # verbatim

    def test_legal_pairings_untouched(self):
        for flag, reason in ((True, "location_change"),
                             (False, "reverse_angle")):
            d = json.loads(doc_line(1, flag=flag, reason=reason))
            r = output_parse.repair_pairing(d)
            self.assertFalse(r["repaired"])
            self.assertIs(d["starts_new_scene"], flag)

    # ── the salvage ladder (2026-08-25) ──────────────────────────────────
    # The real failure: sonnet transcribed a news chyron that literally
    # contains quotation marks and did not escape them. 10 shots of window
    # 89 were lost to it on a run whose bedrock record was SUCCESS,
    # end_turn, with a document for all 16 shots.
    REAL_CHYRON = ('{"shot_id":1428,"description":"A performer tilts their '
                   'head back.","starts_new_scene":true,"boundary_reason":'
                   '"new_line_of_action","shot_kind":"graphics_overlay",'
                   '"camera_shot_type":"close_up","time_of_day":"night",'
                   '"lighting":["low_light"],"notable":[],"people_count":'
                   '"1","people":[],"texts_on_screen":["POPBEAT","MIRAVOX '
                   'RETURNS WITH "STILL GOLD""],"unreadable_text_present":'
                   'false,"named_entities":[],"brands":[],"objects":[],'
                   '"sentiment":"neutral"}')

    def test_unescaped_inner_quotes_recovered_and_counted(self):
        parsed = output_parse.parse_response(self.REAL_CHYRON)
        self.assertEqual(sorted(parsed["docs"]), [1428])
        self.assertEqual(parsed["bad_lines"], [])
        # The rung is NAMED in the count — a recovery is never silent.
        self.assertEqual(parsed["line_repairs"], {"escaped": 1})
        self.assertEqual(parsed["docs"][1428]["texts_on_screen"],
                         ["POPBEAT", 'MIRAVOX RETURNS WITH "STILL GOLD"'])

    def test_the_escape_pass_only_ever_adds_a_backslash(self):
        # Neutrality is what makes the rung safe to run on every failure:
        # measured byte-identical on all 3452 real published documents,
        # 200 of which already carry escaped quotes.
        from lib.parse_partial_output import escape_inner_quotes
        for line in (doc_line(1),
                     doc_line(2, texts_on_screen=['He said "hi"']),
                     doc_line(3, description='a \\"quoted\\" word'),
                     doc_line(4, description="colons: and, commas")):
            self.assertEqual(escape_inner_quotes(line), line)

    def test_a_truncated_value_is_dropped_whole_never_trimmed(self):
        # The property that makes the minimal gate safe, and the measured
        # reason json-repair was removed (2026-08-26): both npm jsonrepair
        # and npm partial-json return the fabricated "indetermina" here.
        # Our salvage only admits values that parsed strictly from the
        # model's own bytes.
        line = doc_line(7, time_of_day="indeterminate")
        cut = line[:line.index("indeterminate") + 8]
        parsed = output_parse.parse_response(cut)
        self.assertEqual(parsed["line_repairs"], {"prefix": 1})
        doc = parsed["docs"][7]
        self.assertNotIn("indetermina", json.dumps(doc))
        for value in doc.values():
            self.assertIn(json.dumps(value, ensure_ascii=False), line)

    def test_a_prefix_that_lost_the_join_key_stays_an_honest_gap(self):
        # shot_id is the ONE required key: without it the document cannot
        # join shots.json. Cut before it, the line is a bad line and the
        # shot a gap — recovery never invents an identity.
        line = ('{"description": "the id comes last here", '
                '"people": ["someone"], "shot_id": 21}')
        parsed = output_parse.parse_response(line[:55])
        self.assertEqual(parsed["docs"], {})
        self.assertEqual(parsed["line_repairs"], {})
        self.assertEqual(len(parsed["bad_lines"]), 1)
        self.assertIn("shot_id", parsed["bad_lines"][0]["error"])

    def test_the_gate_is_the_minimum_the_code_cannot_join_without(self):
        # (user decision 2026-08-26: "required_keys must stay at minimum"
        # — wider throws away a recovered partial over OPTIONAL metadata.)
        # shot_id is required because normalize_doc hard-crashes without
        # it; everything else has a designed, counted degradation.
        self.assertEqual(output_parse.REQUIRED_KEYS, ("shot_id",))
        with self.assertRaises(KeyError):
            output_parse.normalize_doc({"description": "no id"})
        degraded, _pruned = output_parse.normalize_doc({"shot_id": 5})
        self.assertEqual(degraded["starts_new_scene"], False)
        self.assertEqual(degraded["people"], [])
        self.assertNotIn("depicted", degraded)
        repair = output_parse.repair_pairing({"shot_id": 5})
        self.assertEqual(repair["kind"], "missing_flag")

    def test_coverage_names_missing_and_extra(self):
        parsed = output_parse.parse_response(
            doc_line(1) + "\n" + doc_line(3) + "\n" + doc_line(99))
        v = output_parse.validate_coverage(parsed["docs"], [1, 2, 3])
        self.assertEqual(v["missing"], [2])
        self.assertEqual(v["extra"], [99])
        self.assertFalse(v["ok"])


class TestSalvageModule(unittest.TestCase):
    """The ladder lives in lib/parse_partial_output.py — a VERBATIM copy of
    bedrock-inference's stdlib-only master (micro-workflow-bundle/README.md §PI6.2;
    this is the module's first copy). No external salvage package, by
    decision (2026-08-26): json-repair was measured returning a silently
    trimmed value on a mid-word cut, and a fabricated value under the
    minimal gate would land in shot_metadata.jsonl unflagged."""

    MASTER = os.path.join(os.path.dirname(ROOT), "bedrock-inference-on-aws",
                          "lib", "parse_partial_output.py")

    def test_the_copy_is_byte_identical_to_the_master(self):
        if not os.path.isfile(self.MASTER):
            self.skipTest(f"master repo not present: {self.MASTER}")
        with open(self.MASTER, "rb") as f_m, \
                open(os.path.join(ROOT, "lib",
                                  "parse_partial_output.py"), "rb") as f_c:
            self.assertEqual(f_m.read(), f_c.read(),
                             "parse_partial_output.py drifted from its "
                             "master — copy it, never edit it")

    def test_no_external_salvage_package_is_installed_or_imported(self):
        # Asserted against the parsed TREE, not the text: docstrings
        # discuss the rejected package at length, so a substring search
        # would fail while the documentation is accurate.
        import ast
        reqs = _read(os.path.join(ROOT, "requirements.txt"))
        self.assertNotIn("json-repair==", reqs)
        for name in ("output_parse.py", "parse_partial_output.py"):
            with open(os.path.join(ROOT, "lib", name)) as fh:
                tree = ast.parse(fh.read())
            imported = {alias.name
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.Import, ast.ImportFrom))
                        for alias in node.names}
            self.assertNotIn("json_repair", imported, name)


# ─────────────────────────────────────────────────────────────────────────────
# Scene assembly (README §4 — flags → scenes, code-side aggregation)
# ─────────────────────────────────────────────────────────────────────────────

class TestSceneAssembly(unittest.TestCase):
    def setUp(self):
        self.shots = {i: make_shot(i, float(i * 10), float(i * 10 + 10))
                      for i in range(6)}

    def _docs(self, spec):
        """spec: {sid: (flag, extra_fields)}"""
        docs = {}
        for sid, (flag, extra) in spec.items():
            docs[sid], _ = output_parse.normalize_doc(json.loads(
                doc_line(sid, flag=flag,
                         reason="location_change" if flag
                         else "continues_action", **extra)))
        return docs

    def test_flags_group_and_gaps_do_not_break(self):
        docs = self._docs({0: (True, {}), 1: (False, {}),
                           3: (False, {}), 4: (True, {}), 5: (False, {})})
        scenes = scene_assembly.assemble_scenes(
            docs, self.shots, [0, 1, 2, 3, 4, 5])
        self.assertEqual([(s["start_shot_id"], s["end_shot_id"])
                          for s in scenes], [(0, 3), (4, 5)])
        self.assertEqual(scenes[0]["shots_gap"], [2])     # honest member gap
        # Times come from shots.json, never the model.
        self.assertEqual(scenes[0]["start_time"], 0.0)
        self.assertEqual(scenes[0]["end_time"], 40.0)

    def test_attributes_are_majority_never_first_shot(self):
        docs = self._docs({
            0: (True, {"time_of_day": "indeterminate"}),
            1: (False, {"time_of_day": "day"}),
            2: (False, {"time_of_day": "day"}),
        })
        scenes = scene_assembly.assemble_scenes(docs, self.shots, [0, 1, 2])
        self.assertEqual(scenes[0]["time_of_day"], "day")

    def test_named_unions_dedupe_on_name(self):
        docs = self._docs({
            0: (True, {"brands": [{"name": "Toyota", "evidence": "text"}]}),
            1: (False, {"brands": [{"name": "Toyota",
                                    "evidence": "logotype"}],
                        "people": ["Ray Halloran"]}),
        })
        scenes = scene_assembly.assemble_scenes(docs, self.shots, [0, 1])
        self.assertEqual(len(scenes[0]["brands"]), 1)
        self.assertEqual(scenes[0]["brands"][0]["evidence"], "text")
        self.assertEqual(scenes[0]["people"], ["Ray Halloran"])


# ─────────────────────────────────────────────────────────────────────────────
class TestDepicted(unittest.TestCase):
    """The `depicted` observables field (rolled into default.md
    2026-09-03 after the trailer/drama_b 202 trial): the vocabulary is
    the BOUND PROMPT'S OWN declaration — code has no list of its own, so
    a custom prompt_version brings its own vocabulary (user decision
    2026-09-03: conformance must not cost prompt customization)."""

    OUR_ACTS = frozenset((
        "firearm", "blade", "other_weapon", "weapon_threat",
        "physical_violence", "blood", "injury", "corpse", "smoking",
        "alcohol", "drug_material", "drug_use", "drug_paraphernalia",
        "nudity_partial", "nudity_full", "intimate_contact", "peril",
        "restraint", "self_harm", "dangerous_act", "hate_symbol"))

    def test_seed_prompt_declares_exactly_the_validated_vocabulary(self):
        # The runtime parser applied to OUR registry seed must yield the
        # 21 acts the trailer/drama_b trial validated — this is the
        # lockstep guard, now testing the real extraction path instead
        # of a code-side copy.
        seed = os.path.join(os.path.dirname(ROOT),
                            "prompt-registry-on-aws", "prompts",
                            "shot-metadata-enrichment", "default.md")
        if not os.path.isfile(seed):
            self.skipTest(f"registry repo not present: {seed}")
        vocab = output_parse.depicted_vocabulary(_read(seed))
        self.assertEqual(vocab, self.OUR_ACTS,
                         "the seed's depicted section drifted from the "
                         "validated vocabulary")
        # and default_news deliberately has no section at all
        news = seed.replace("default.md", "default_news.md")
        self.assertIsNone(
            output_parse.depicted_vocabulary(_read(news)))

    def test_out_of_vocab_acts_are_pruned_and_reported(self):
        # Pins the trial incident: the model invented `fight_violence`
        # and `chase_pursuit` on the trailer run — an act outside the
        # bound prompt's own list must never reach consumers who key on
        # that list, and the prune must be REPORTED, never silent.
        doc, pruned = output_parse.normalize_doc({
            "shot_id": 9,
            "depicted": [
                {"act": "drug_material", "detail": "wrapped bricks"},
                {"act": "fight_violence", "detail": "gunfire exchange"},
                {"act": "chase_pursuit", "detail": "skiff pursued"},
                "not-even-a-dict",
            ]}, depicted_vocab=self.OUR_ACTS)
        self.assertEqual([e["act"] for e in doc["depicted"]],
                         ["drug_material"])
        self.assertEqual(pruned,
                         ["fight_violence", "chase_pursuit",
                          "not-even-a-dict"])

    def test_custom_prompt_brings_its_own_vocabulary(self):
        # A user-designed version declaring different acts is enforced
        # against ITS list — ours must not fail them.
        custom = ("### depicted (required)\n"
                  "Each entry `{\"act\":\"\",\"detail\":\"\"}` with act "
                  "from: `gambling` · `weapon_threat`\n\n"
                  "## Output")
        vocab = output_parse.depicted_vocabulary(custom)
        self.assertEqual(vocab, frozenset({"gambling", "weapon_threat"}))
        doc, pruned = output_parse.normalize_doc(
            {"shot_id": 1, "depicted": [
                {"act": "gambling", "detail": "poker table"},
                {"act": "smoking", "detail": "cigarette"}]},
            depicted_vocab=vocab)
        self.assertEqual([e["act"] for e in doc["depicted"]],
                         ["gambling"])
        self.assertEqual(pruned, ["smoking"])

    def test_open_vocabulary_and_no_section_do_not_prune(self):
        # A section that declares no backticked tokens = the author
        # chose an OPEN vocabulary: shape is still coerced, no act is
        # judged. No section at all = None (never asked).
        open_section = "### depicted\nRecord anything policy-relevant.\n"
        self.assertIsNone(output_parse.depicted_vocabulary(open_section))
        self.assertIsNone(output_parse.depicted_vocabulary("no section"))
        doc, pruned = output_parse.normalize_doc(
            {"shot_id": 2, "depicted": [
                {"act": "anything_at_all", "detail": "kept"},
                "junk-still-dropped"]}, depicted_vocab=None)
        self.assertEqual([e["act"] for e in doc["depicted"]],
                         ["anything_at_all"])
        self.assertEqual(pruned, ["junk-still-dropped"])

    def test_media_depictions_mirrors_the_depicted_discipline(self):
        # the depicted-face guard channel (rolled in 2026-09-06): same
        # closed-vocabulary prune, same []-vs-absent attestation, SAME
        # declaration grammar as depicted — individually backticked
        # tokens (the pipe-form-in-snippet variant was conformed away,
        # review 2026-09-07: one grammar, one parser branch)
        section = ("### media_depictions (required)\n"
                   'record one entry per medium, `{"medium": "", '
                   '"detail": ""}`, with `medium` from this CLOSED '
                   "list:\n\n"
                   "`photo` · `screen` · `poster` · `mirror` · "
                   "`projection`\n\n## Output")
        vocab = output_parse.media_vocabulary(section)
        self.assertEqual(vocab, frozenset(
            {"photo", "screen", "poster", "mirror", "projection"}))
        doc, pruned = output_parse.normalize_doc(
            {"shot_id": 3, "media_depictions": [
                {"medium": "screen", "detail": "anchor on monitor"},
                {"medium": "hologram", "detail": "invented"},
                "junk"]},
            media_vocab=vocab)
        self.assertEqual([e["medium"] for e in doc["media_depictions"]],
                         ["screen"])
        self.assertEqual(pruned, ["hologram", "junk"])
        # absent stays absent (not assessed)
        doc, _ = output_parse.normalize_doc({"shot_id": 4})
        self.assertNotIn("media_depictions", doc)
        # no section -> None (never asked)
        self.assertIsNone(output_parse.media_vocabulary("no section"))

    def test_absent_key_is_not_assessed_never_attested_none(self):
        # [] asserts "none of these observables is visible" — an
        # assertion only a prompt that ASKED may make. normalize_doc
        # must leave an absent key absent, so a default_news doc (whose
        # prompt has no depicted section) never carries a false
        # attested-none; collect_finalize injects [] for omissions only
        # when the bound prompt text contains the section.
        doc, pruned = output_parse.normalize_doc(
            {"shot_id": 7, "description": "news anchor at desk"})
        self.assertNotIn("depicted", doc)
        self.assertEqual(pruned, [])
        # and an emitted-but-malformed value still normalizes in place
        doc, _ = output_parse.normalize_doc(
            {"shot_id": 8, "depicted": "not-a-list"})
        self.assertEqual(doc["depicted"], [])


# Copied modules (CONTRACT §5: byte-identical to the masters)
# ─────────────────────────────────────────────────────────────────────────────

class TestCopiedModules(unittest.TestCase):
    REGISTRY = os.path.join(os.path.dirname(ROOT), "prompt-registry-on-aws",
                            "gateway_tool")
    SIBLING = os.path.join(os.path.dirname(ROOT),
                           "conversation-analysis-on-aws", "gateway_tool")

    def _compare(self, master_dir, name, local):
        master = os.path.join(master_dir, name)
        if not os.path.isfile(master):
            self.skipTest(f"master repo not present: {master}")
        with open(master, "rb") as f_m, open(local, "rb") as f_c:
            self.assertEqual(f_m.read(), f_c.read(),
                             f"{name} drifted from its master — copy it, "
                             "never edit it")

    def test_prompt_render_is_byte_identical(self):
        self._compare(self.REGISTRY, "prompt_render.py",
                      os.path.join(ROOT, "lib", "prompt_render.py"))

    def test_workflow_history_is_byte_identical(self):
        self._compare(self.REGISTRY, "workflow_history.py",
                      os.path.join(ROOT, "lib", "workflow_history.py"))

    def test_workflow_identity_is_byte_identical(self):
        self._compare(self.SIBLING, "workflow_identity.py",
                      os.path.join(ROOT, "gateway_tool",
                                   "workflow_identity.py"))


# ─────────────────────────────────────────────────────────────────────────────
# State handlers against fakes
# ─────────────────────────────────────────────────────────────────────────────

# the gateway's flat-zip sibling (workflow_identity) must be importable
# — APPENDED so `import lambda_function` resolves to the ROOT router
sys.path.append(os.path.join(ROOT, "gateway_tool"))

from states.collect_finalize import lambda_function as collect_finalize  # noqa: E402
from states.render_windows import lambda_function as render_windows  # noqa: E402
import lambda_function as router                       # noqa: E402
from states.stage_records import lambda_function as stage_records  # noqa: E402
from states.submit_history import lambda_function as submit_history  # noqa: E402
from states.validate_plan import lambda_function as validate_plan  # noqa: E402
from lib.custom_exceptions import (LambdaResourceTimeoutException,  # noqa: E402
                               ShotMetadataValidationError)


class _FakeS3:
    """Dict-backed: get/put/list + multipart + the 404 distinction."""

    def __init__(self):
        self.objects: dict = {}
        self._mp: dict = {}
        self.aborted = []

    def put_object(self, Bucket, Key, Body, **_kw):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) \
            else Body.encode("utf-8")
        return {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey",
                                         "Message": "missing"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix, **_kw):
        keys = sorted(k for (b, k) in self.objects
                      if b == Bucket and k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys]}

    def create_multipart_upload(self, Bucket, Key, **_kw):
        self._mp[(Bucket, Key)] = []
        return {"UploadId": "up-1"}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        self._mp[(Bucket, Key)].append((PartNumber, Body))
        return {"ETag": f"etag-{PartNumber}"}

    def complete_multipart_upload(self, Bucket, Key, UploadId,
                                  MultipartUpload):
        parts = sorted(self._mp.pop((Bucket, Key)))
        self.objects[(Bucket, Key)] = b"".join(p[1] for p in parts)
        return {}

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.aborted.append(Key)
        self._mp.pop((Bucket, Key), None)
        return {}

    def json(self, key, bucket="test-proxy"):
        return json.loads(self.objects[(bucket, key)])


class _FakeLambda:
    def __init__(self):
        self.routes = {}
        self.calls = []

    def invoke(self, FunctionName, Payload, ClientContext=None, **_kw):
        tool = json.loads(base64.b64decode(ClientContext))["custom"][
            "bedrockAgentCoreToolName"]
        event = json.loads(Payload)
        self.calls.append((FunctionName, tool, event))
        result = self.routes[(FunctionName, tool)](event)
        return {"Payload": io.BytesIO(json.dumps(result).encode("utf-8"))}


class _FakeContext:
    def __init__(self, remaining_ms=600_000):
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self):
        return self.remaining_ms


PROMPT_ARTIFACT = {
    "category": "shot-metadata-enrichment", "version": "default",
    "stage": "prod", "source_md5": "d" * 32,
    "system": "Catalogue shots. JSONL, one document per REQUIRED shot.",
    "user_template": ("Program: {{program_names}}\n"
                      "Type: {{overall_content_type}}\n"
                      "Shots: {{shot_list}}\n"
                      "Faces: {{face_names}}\n"
                      "Gallery: {{face_images}}\n"
                      "Transcript: {{transcript}}\n"
                      "Frames: {{frame_images}}"),
    "variables": [
        {"token": t, "root": t.strip("{}"), "leafs": []}
        for t in ("{{program_names}}", "{{overall_content_type}}",
                  "{{shot_list}}", "{{face_names}}", "{{face_images}}",
                  "{{transcript}}", "{{frame_images}}")],
}

BEDROCK_GW = os.environ["BEDROCK_GATEWAY_ARN"]
REGISTRY_GW = os.environ["PROMPT_REGISTRY_ARN"]
SHOTSEG_URI = ("s3://test-proxy/micro-workflow/shot-segmentation/"
               "seg-wf-1/embeddings/shots.json")
SHOTSEG_KEY = SHOTSEG_URI[len("s3://test-proxy/"):]


def tarball(n_frames) -> bytes:
    """A shot archive of n one-byte-different JPEG-ish members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for i in range(n_frames):
            data = b"\xff\xd8FAKEJPEG" + bytes([i])
            info = tarfile.TarInfo(name=f"frames/{i:04d}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _WorkflowCase(unittest.TestCase):
    """Patched fakes + the real Meridian shots.json staged."""

    def setUp(self):
        self.s3 = _FakeS3()
        self.lam = _FakeLambda()
        self.lam.routes[(REGISTRY_GW, "prompt-registry___get_prompt")] = \
            lambda ev: dict(PROMPT_ARTIFACT)
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___get_model")] = \
            lambda ev: {"model": {"model_id": ev["model_id"],
                                  "model_type": ["vision", "text"]}}
        with open(os.path.join(FIXTURES, "meridian_shots.json"), "rb") as f:
            self.s3.objects[("test-proxy", SHOTSEG_KEY)] = f.read()
        patches = [mock.patch.object(sme_common, "_s3", self.s3),
                   mock.patch.object(sme_common, "_lambda", self.lam)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def plan(self, **overrides):
        event = {"uuid": "wf-test-1",
                 "input": {"shot_segmentation_s3uri": SHOTSEG_URI,
                           **overrides}}
        return validate_plan.lambda_handler(event, None)

    def stage_tarballs(self):
        shots = meridian_shots()
        for s in shots:
            self.s3.objects[("test-proxy", s["frames_key"])] = tarball(
                s["selected_frame_count"])


class TestValidatePlan(_WorkflowCase):
    def test_prompt_version_is_forwarded_to_the_registry(self):
        """A caller's prompt_version reaches get_prompt as `version`; absent
        means the category default (no version key at all — the registry's
        named-version-vs-default asymmetry stays the registry's)."""
        seen = {}
        def _get(ev):
            seen.clear(); seen.update(ev)
            return dict(PROMPT_ARTIFACT)
        self.lam.routes[(REGISTRY_GW, "prompt-registry___get_prompt")] = _get
        self.plan(prompt_version="default_news")
        self.assertEqual(seen.get("version"), "default_news")
        self.plan()
        self.assertNotIn("version", seen)

    def test_happy_path_defaults_land_concrete(self):
        out = self.plan()
        plan = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                            "wf-test-1/_plan/plan.json")
        self.assertEqual(plan["scope"], "shots.seg-wf-1")
        # User decisions 2026-08-19, as CONCRETE values:
        self.assertEqual(plan["model_chain"],
                         ["anthropic.claude-sonnet-4-6"])
        self.assertIs(plan["use_batch"], True)
        self.assertEqual(out["window_count"], len(plan["windows"]))
        self.assertEqual(
            sum(len(b["window_indices"]) for b in out["batches"]),
            len(plan["windows"]))

    def test_use_batch_false_is_honoured(self):
        self.plan(use_batch=False)
        plan = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                            "wf-test-1/_plan/plan.json")
        self.assertIs(plan["use_batch"], False)

    def test_run_prefix_uri_is_resolved_to_shots_json(self):
        event = {"uuid": "wf-test-1", "input": {
            "shot_segmentation_s3uri":
                SHOTSEG_URI.rsplit("/embeddings", 1)[0]}}
        validate_plan.lambda_handler(event, None)
        plan = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                            "wf-test-1/_plan/plan.json")
        self.assertTrue(plan["shots"]["s3uri"].endswith(
            "embeddings/shots.json"))

    def test_text_only_model_refused(self):
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___get_model")] = \
            lambda ev: {"model": {"model_id": ev["model_id"],
                                  "model_type": ["text"]}}
        with self.assertRaises(ShotMetadataValidationError):
            self.plan()

    def test_prompt_token_outside_catalog_refused(self):
        bad = dict(PROMPT_ARTIFACT)
        bad["variables"] = PROMPT_ARTIFACT["variables"] + [
            {"token": "{{sound_events}}", "root": "sound_events",  # nosec B105
             "leafs": []}]
        self.lam.routes[(REGISTRY_GW, "prompt-registry___get_prompt")] = \
            lambda ev: bad
        with self.assertRaises(ShotMetadataValidationError) as ctx:
            self.plan()
        self.assertIn("sound_events", str(ctx.exception))

    def test_facemetadata_without_rep_face_key_refused_by_name(self):
        meta = {"people": [{"person_id": 0, "name": "Jane Doe",
                            "named": True, "total_screen_time_sec": 5.0,
                            "appearances": [{"begin_ms": 0,
                                             "end_ms": 5000}]}]}
        self.s3.put_object(Bucket="test-proxy",
                           Key="micro-workflow/face-recognition/fr-1/"
                               "facemetadata.json",
                           Body=json.dumps(meta))
        with self.assertRaises(ShotMetadataValidationError) as ctx:
            self.plan(face_recognition_s3uri="s3://test-proxy/"
                      "micro-workflow/face-recognition/fr-1")
        self.assertIn("rep_face_key", str(ctx.exception))


class TestRenderWindows(_WorkflowCase):
    def _plan_and_batch(self, **overrides):
        out = self.plan(**overrides)
        self.stage_tarballs()
        return out

    def test_renders_records_with_labeled_frames_and_sentinels(self):
        out = self._plan_and_batch()
        first = out["batches"][0]
        result = render_windows.lambda_handler(
            {"plan_s3uri": out["plan_s3uri"], **first}, _FakeContext())
        self.assertEqual(result["rendered"], len(first["window_indices"]))
        rec = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                           "wf-test-1/_records/window_000.json")
        self.assertEqual(rec["record_id"], "window-000")
        text = json.dumps(rec["messages"])
        self.assertIn("no face recognition supplied", text)
        self.assertIn("no reference faces supplied", text)
        self.assertIn("no transcript supplied", text)
        self.assertIn("Image 1 = shot_id 0 (1 of 1) [REQUIRED]:", text)
        # Slim shot_list: no timecodes anywhere (README §5).
        self.assertNotIn("start_timecode", text)
        images = sum(1 for m in rec["messages"]
                     for b in m["content"] if "image" in b)
        plan = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                            "wf-test-1/_plan/plan.json")
        self.assertEqual(images,
                         sum(plan["windows"][0]["frames"].values()))

    def test_reenter_skips_existing_side_objects(self):
        out = self._plan_and_batch()
        first = out["batches"][0]
        render_windows.lambda_handler(
            {"plan_s3uri": out["plan_s3uri"], **first}, _FakeContext())
        again = render_windows.lambda_handler(
            {"plan_s3uri": out["plan_s3uri"], **first}, _FakeContext())
        self.assertEqual(again["rendered"], 0)
        self.assertEqual(again["skipped"], len(first["window_indices"]))

    def test_deadline_guard_checkpoints_and_raises(self):
        out = self._plan_and_batch()
        first = out["batches"][0]
        with self.assertRaises(LambdaResourceTimeoutException):
            render_windows.lambda_handler(
                {"plan_s3uri": out["plan_s3uri"], **first},
                _FakeContext(remaining_ms=1_000))

    def test_gallery_rides_when_faces_join(self):
        meta = dict(FACEMETA)
        self.s3.put_object(Bucket="test-proxy",
                           Key="micro-workflow/face-recognition/fr-1/"
                               "facemetadata.json",
                           Body=json.dumps(meta))
        self.s3.put_object(Bucket="test-proxy",
                           Key="fr/people/person_0000_face.jpg",
                           Body=b"RAYJPEG")
        self.s3.put_object(Bucket="test-proxy",
                           Key="fr/people/person_0001_face.jpg",
                           Body=b"DANJPEG")
        out = self.plan(face_recognition_s3uri="s3://test-proxy/"
                        "micro-workflow/face-recognition/fr-1")
        self.stage_tarballs()
        with mock.patch("lib.gallery.compose_gallery",
                        return_value=b"GALLERYJPEG") as composed:
            render_windows.lambda_handler(
                {"plan_s3uri": out["plan_s3uri"], **out["batches"][0]},
                _FakeContext())
        self.assertTrue(composed.called)
        rec = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                           "wf-test-1/_records/window_000.json")
        text = json.dumps(rec["messages"])
        self.assertIn("Reference face gallery", text)
        self.assertIn("Ray Halloran", text)
        self.assertIn(base64.b64encode(b"GALLERYJPEG").decode(), text)


class TestStageRecords(_WorkflowCase):
    def _rendered(self):
        out = self.plan()
        self.stage_tarballs()
        for batch in out["batches"]:
            render_windows.lambda_handler(
                {"plan_s3uri": out["plan_s3uri"], **batch}, _FakeContext())
        return out

    def test_concatenates_in_window_order(self):
        out = self._rendered()
        staged = stage_records.lambda_handler(
            {"uuid": "wf-test-1", "plan_s3uri": out["plan_s3uri"]}, None)
        body = self.s3.objects[("test-proxy",
                                "micro-workflow/shot-metadata-enrichment/"
                                "wf-test-1/records.jsonl")]
        ids = [json.loads(l)["record_id"]
               for l in body.decode().splitlines()]
        self.assertEqual(ids, [f"window-{i:03d}" for i in range(len(ids))])
        self.assertEqual(staged["record_count"], len(ids))

    def test_missing_checkpoint_refuses_and_aborts_multipart(self):
        out = self._rendered()
        del self.s3.objects[("test-proxy",
                             "micro-workflow/shot-metadata-enrichment/"
                             "wf-test-1/_records/window_001.json")]
        with self.assertRaises(RuntimeError):
            stage_records.lambda_handler(
                {"uuid": "wf-test-1", "plan_s3uri": out["plan_s3uri"]},
                None)
        self.assertTrue(self.s3.aborted)


class _SubmitCase(_WorkflowCase):
    def submit(self, response=None):
        out = self.plan()
        self.stage_tarballs()
        for batch in out["batches"]:
            render_windows.lambda_handler(
                {"plan_s3uri": out["plan_s3uri"], **batch}, _FakeContext())
        staged = stage_records.lambda_handler(
            {"uuid": "wf-test-1", "plan_s3uri": out["plan_s3uri"]}, None)
        self.lam.routes[(BEDROCK_GW,
                         "bedrock-inference___run_inference")] = \
            lambda ev: response or {
                "job_id": "arn:aws:states:eu-west-1:1:execution:b:x",
                "workflow_id": "bedrock-wf-9",
                "output_s3_prefix": "s3://test-proxy/micro-workflow/"
                                    "bedrock-inference/bedrock-wf-9",
                "status": "RUNNING"}
        submit_history.lambda_handler(
            {"uuid": "wf-test-1", "plan_s3uri": out["plan_s3uri"],
             "records_s3uri": staged["records_s3uri"],
             "record_count": staged["record_count"],
             "task_token": "tok-1"}, None)  # nosec B105
        return out


class TestSubmitHistory(_SubmitCase):
    def test_submit_carries_token_batch_and_writes_history(self):
        self.submit()
        call = [c for c in self.lam.calls
                if c[1] == "bedrock-inference___run_inference"][0][2]
        self.assertEqual(call["idempotency_token"], "wf-test-1")
        self.assertIs(call["use_batch"], True)
        self.assertEqual(call["callback_task_token"], "tok-1")
        self.assertNotIn("max_tokens", call)
        runs = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                            "wf-test-1/_plan/runs.json")
        self.assertEqual(runs[0]["workflow_id"], "bedrock-wf-9")
        self.assertFalse(runs[0]["absorbed"])
        self.assertTrue(any(k.startswith("micro-workflow/histories/"
                                         "shots.seg-wf-1/")
                            for (_, k) in self.s3.objects))


class TestCollectFinalize(_SubmitCase):
    def _respond(self, out, lines):
        self.s3.put_object(
            Bucket="test-proxy",
            Key="micro-workflow/bedrock-inference/bedrock-wf-9/"
                "response.jsonl",
            Body="\n".join(json.dumps(l) for l in lines))
        return collect_finalize.lambda_handler(
            {"uuid": "wf-test-1", "plan_s3uri": out["plan_s3uri"],
             "wait_result": {"status": "SUCCEEDED"}}, None)

    def _full_response(self, plan, mutate=None):
        lines = []
        for w in plan["windows"]:
            docs = [json.loads(doc_line(
                sid, flag=(sid == w["shot_ids"][0]),
                reason="location_change" if sid == w["shot_ids"][0]
                else "continues_action")) for sid in w["shot_ids"]]
            if mutate:
                docs = mutate(w, docs)
            lines.append({"record_id": f"window-{w['index']:03d}",
                          "status": "SUCCESS",
                          "output_text": "\n".join(
                              json.dumps(d) for d in docs)})
        return lines

    def test_happy_path_products_and_manifest(self):
        out = self.submit()
        plan = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                            "wf-test-1/_plan/plan.json")
        result = self._respond(out, self._full_response(plan))
        self.assertEqual(result["shots_gap"], 0)
        self.assertEqual(result["shots_enriched"], 97)
        self.assertEqual(result["scenes"], len(plan["windows"]))
        jsonl = self.s3.objects[("test-proxy",
                                 "micro-workflow/shot-metadata-enrichment/"
                                 "wf-test-1/shot_metadata.jsonl")]
        self.assertEqual(len(jsonl.decode().splitlines()), 97)
        scenes = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                              "wf-test-1/scenes.json")
        self.assertEqual(scenes["pairing_repairs"], [])
        manifest = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                                "wf-test-1/window_manifest.json")
        self.assertTrue(all(w["resolved"] for w in manifest["windows"]))

    def test_cross_pairing_repaired_counted_and_reason_verbatim(self):
        out = self.submit()
        plan = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                            "wf-test-1/_plan/plan.json")

        def mutate(w, docs):
            if w["index"] == 0:
                docs[1]["starts_new_scene"] = True
                docs[1]["boundary_reason"] = "intercut_return"
            return docs
        result = self._respond(out, self._full_response(plan, mutate))
        self.assertEqual(result["pairing_repairs"], 1)
        scenes = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                              "wf-test-1/scenes.json")
        rep = scenes["pairing_repairs"][0]
        self.assertEqual(rep["reason"], "intercut_return")
        # The repaired flag must NOT have opened a scene at that shot.
        self.assertEqual(result["scenes"], len(plan["windows"]))

    def test_failed_window_becomes_honest_gaps(self):
        out = self.submit()
        plan = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                            "wf-test-1/_plan/plan.json")
        lines = self._full_response(plan)
        lines[1] = {"record_id": lines[1]["record_id"], "status": "FAILED",
                    "error": "throttled"}
        result = self._respond(out, lines)
        expected_gap = len(plan["windows"][1]["shot_ids"])
        self.assertEqual(result["shots_gap"], expected_gap)
        scenes = self.s3.json("micro-workflow/shot-metadata-enrichment/"
                              "wf-test-1/scenes.json")
        self.assertEqual(len(scenes["shots_gap"]), expected_gap)

    def test_terminal_failure_raises(self):
        out = self.submit()
        from lib.custom_exceptions import ShotMetadataInferenceFailed
        with self.assertRaises(ShotMetadataInferenceFailed):
            collect_finalize.lambda_handler(
                {"uuid": "wf-test-1", "plan_s3uri": out["plan_s3uri"],
                 "wait_result": {"status": "FAILED"}}, None)


class TestRouter(unittest.TestCase):
    def test_unknown_operation_raises(self):
        with self.assertRaises(ValueError):
            router.lambda_handler({"operation": "Nope"}, None)

    def test_routes_are_the_asl_operations(self):
        defn = _load_json(os.path.join(ROOT, "definitions.json"))
        text = json.dumps(defn)
        for op in router._OPERATIONS:
            self.assertIn(f'"{op}"', text)


# ─────────────────────────────────────────────────────────────────────────────
# Gateway
# ─────────────────────────────────────────────────────────────────────────────

import importlib.util as _ilu                          # noqa: E402
_spec = _ilu.spec_from_file_location(
    "sme_gateway", os.path.join(ROOT, "gateway_tool", "lambda_function.py"))
gateway = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(gateway)


class _Ctx:
    class client_context:
        custom = {}

    def __init__(self, tool):
        self.client_context = type(
            "CC", (), {"custom": {"bedrockAgentCoreToolName": tool}})()


class TestGateway(unittest.TestCase):
    def setUp(self):
        self.sfn = mock.MagicMock()
        p = mock.patch.object(gateway, "_sfn", self.sfn)
        p.start()
        self.addCleanup(p.stop)

    def _submit(self, **kw):
        self.sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:eu-west-1:1:execution:sm:x"}
        event = {"shot_segmentation_s3uri": "s3://test-proxy/"
                 "micro-workflow/shot-segmentation/seg-1/", **kw}
        return gateway.lambda_handler(event, _Ctx(
            "shot-metadata-enrichment___enrich_shot_metadata"))

    def test_submit_returns_a_handle_never_results(self):
        out = self._submit()
        self.assertEqual(sorted(out), ["job_id", "output_s3_prefix",
                                       "status", "workflow_id"])

    def test_token_narrows_identity_and_is_never_echoed(self):
        a = self._submit()
        b = self._submit(idempotency_token="t-1")  # nosec B106
        self.assertNotEqual(a["workflow_id"], b["workflow_id"])
        for out in (a, b):
            self.assertNotIn("idempotency_token", out)
            self.assertNotIn("uuid", out)

    def test_prompt_version_enters_the_hash_and_absent_means_default(self):
        """prompt_version selects a NAMED registry version — different
        version, different work, so it must move the workflow_id; absent
        must hash exactly as before the knob existed (no re-identification
        of existing callers)."""
        base = self._submit()
        news = self._submit(prompt_version="default_news")
        self.assertNotEqual(base["workflow_id"], news["workflow_id"])
        sfn_input = json.loads(
            self.sfn.start_execution.call_args.kwargs["input"])
        self.assertEqual(sfn_input["input"]["prompt_version"],
                         "default_news")
        blank = self._submit(prompt_version="")
        self.assertEqual(base["workflow_id"], blank["workflow_id"])

    def test_retired_uuid_parameter_rejected(self):
        out = gateway.lambda_handler(
            {"uuid": "x", "shot_segmentation_s3uri": "s3://b/k"},
            _Ctx("enrich_shot_metadata"))
        self.assertTrue(out["error"])

    def test_missing_uri_refused(self):
        out = gateway.lambda_handler(
            {}, _Ctx("enrich_shot_metadata"))
        self.assertEqual(out["errorCode"], "MISSING_PARAMETER")

    def test_describe_tool_serves_the_contract(self):
        out = gateway.lambda_handler({}, _Ctx("describe_tool"))
        self.assertIn("output_contract", out)
        self.assertEqual(
            sorted(out["output_contract"]["outputs"]),
            ["scenes", "shots", "usage", "window-manifest"])

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        path = os.path.join(ROOT, "gateway_tool", "input_examples.json")
        examples = _load_json(path)["tools"]
        self.assertEqual(sorted(examples),
                         ["describe_tool", "enrich_shot_metadata",
                          "get_job_status"])
        for tool, entry in examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        out = gateway.lambda_handler({}, _Ctx("describe_tool"))
        self.assertIn("enrich_shot_metadata", out["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        path = os.path.join(ROOT, "gateway_tool", "input_examples.json")
        example = _load_json(path)["tools"]["enrich_shot_metadata"]["example"]
        self.sfn.start_execution.return_value = {"executionArn": "arn:fake"}
        out = gateway.lambda_handler(dict(example), _Ctx(
            "shot-metadata-enrichment___enrich_shot_metadata"))
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        self.sfn.start_execution.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Template parity
# ─────────────────────────────────────────────────────────────────────────────

class TestTemplateParity(unittest.TestCase):
    def test_inline_asl_matches_definitions_json(self):
        import re
        src = _read(os.path.join(ROOT, "cfnstack.yaml"))
        m = re.search(r" {12}DefinitionString: \|-\n((?: {16}.*\n)+)", src)
        self.assertIsNotNone(m)
        body = "\n".join(l[16:] for l in m.group(1).rstrip().splitlines())
        self.assertEqual(json.loads(body),
                         _load_json(os.path.join(ROOT,
                                                     "definitions.json")))

    def test_map_retries_on_the_timeout_exception(self):
        defn = _load_json(os.path.join(ROOT, "definitions.json"))
        task = defn["States"]["Render windows"]["ItemProcessor"][
            "States"]["Render window batch"]
        names = [e for r in task["Retry"] for e in r["ErrorEquals"]]
        self.assertIn("LambdaResourceTimeoutException", names)


class TestDeriveTextViews(unittest.TestCase):
    """Single-source text views (user design, 2026-09-09): the model
    samples each string once; named_entities and texts_on_screen are
    derived. Data-driven on doc shape — no genre branch."""

    def _norm(self, doc):
        from lib import output_parse
        out, _ = output_parse.normalize_doc(doc)
        return out

    def test_news_shape_flattens_names_and_derives_remainder(self):
        out = self._norm({
            "shot_id": 1, "supers": [
                {"text": "JORDAN FELLOWS | EXAMPLE NEWS",
                 "super_type": "name_super",
                 "names": [{"name": "Jordan Fellows", "type": "person"},
                           {"name": "Example News",
                            "type": "organization"}]},
                {"text": "MARKETS RALLY INTO THE CLOSE",
                 "super_type": "headline_banner", "names": []},
            ]})
        ents = {(e["name"], e["type"], e["evidence"])
                for e in out["named_entities"]}
        self.assertIn(("Jordan Fellows", "person", "overlay_text"), ents)
        self.assertIn(("Example News", "organization", "overlay_text"),
                      ents)
        # the flat view is the UNCLAIMED remainder: named supers excluded
        self.assertEqual(out["texts_on_screen"],
                         ["MARKETS RALLY INTO THE CLOSE"])

    def test_flatten_dedups_against_overlay_copies_only(self):
        # an existing OVERLAY copy suppresses the flatten (no duplicate)
        out = self._norm({
            "shot_id": 1,
            "named_entities": [{"name": "Sam Reyes", "type": "person",
                                "evidence": "overlay_text"}],
            "supers": [{"text": "SAM REYES", "super_type": "name_super",
                        "names": [{"name": "Sam Reyes",
                                   "type": "person"}]}]})
        self.assertEqual(
            len([e for e in out["named_entities"]
                 if e["name"] == "Sam Reyes"]), 1)

    def test_episodic_claimed_line_removed_partial_kept(self):
        out = self._norm({
            "shot_id": 2,
            "texts_on_screen": ["JORDAN FELLOWS", "PARIS, 1943",
                                "meet me at the docks"],
            "named_entities": [
                {"name": "Jordan Fellows", "type": "person",
                 "evidence": "overlay_text"},
                {"name": "Paris", "type": "place",
                 "evidence": "overlay_text"}]})
        # full coverage claims the name line; "PARIS, 1943" carries more
        # than its entity and survives; unrelated text untouched
        self.assertEqual(out["texts_on_screen"],
                         ["PARIS, 1943", "meet me at the docks"])

    def test_stray_misevidenced_copy_never_suppresses_the_overlay(self):
        # measured live 2026-09-09: the model double-emitted a chyron
        # person as environment_text; the derived overlay copy MUST
        # still land or cast-resolution's channel never sees them
        out = self._norm({
            "shot_id": 18,
            "named_entities": [{"name": "Terry Vale", "type": "person",
                                "evidence": "environment_text"}],
            "supers": [{"text": "TERRY VALE | ACME FOUNDER",
                        "super_type": "name_super",
                        "names": [{"name": "Terry Vale",
                                   "type": "person"}]}]})
        evs = {e["evidence"] for e in out["named_entities"]
               if e["name"] == "Terry Vale"}
        self.assertIn("overlay_text", evs)

    def test_environment_entities_never_claim_lines(self):
        out = self._norm({
            "shot_id": 3,
            "texts_on_screen": ["CHEZ JAY"],
            "named_entities": [{"name": "Chez Jay", "type": "place",
                                "evidence": "environment_text"}]})
        self.assertEqual(out["texts_on_screen"], ["CHEZ JAY"])


class TestNameSuperEntityGap(unittest.TestCase):
    """The gauge for the prompt's name_super->entity pairing rule
    (2026-09-09). Language-agnostic containment between the model's own
    outputs; the affiliation-only partial is a DOCUMENTED under-count."""

    def _gap(self, docs):
        from lib import output_parse
        return output_parse.name_super_entity_gap(docs)

    def test_uncovered_super_counts(self):
        docs = {3: {"supers": [{"super_type": "name_super",
                                "text": "JORDAN FELLOWS"}],
                    "named_entities": []}}
        g = self._gap(docs)
        self.assertEqual((g["count"], g["supers_total"], g["shot_ids"]),
                         (1, 1, [3]))

    def test_paired_entity_covers_case_insensitively(self):
        docs = {3: {"supers": [{"super_type": "name_super",
                                "text": "JORDAN FELLOWS | EXAMPLE NEWS"}],
                    "named_entities": [{"name": "Jordan Fellows",
                                        "type": "person",
                                        "evidence": "overlay_text"}]}}
        self.assertEqual(self._gap(docs)["count"], 0)

    def test_org_only_extraction_is_a_person_miss(self):
        # the legacy shape's worst case ("J. SMITH | ACME" -> just the
        # org): shot-level coverage requires a PERSON, so this now
        # counts — it is precisely the miss the gauge exists for
        docs = {7: {"supers": [{"super_type": "name_super",
                                "text": "JORDAN FELLOWS | EXAMPLE NEWS"}],
                    "named_entities": [{"name": "Example News",
                                        "type": "organization",
                                        "evidence": "overlay_text"}]}}
        self.assertEqual(self._gap(docs)["count"], 1)

    def test_two_line_graphic_office_super_is_covered_by_its_shot(self):
        # RNC eyeball 2026-09-09: name and office on separate
        # name_super lines; the office line correctly extracts nobody
        # and must not count while the shot's person was captured
        docs = {8: {"supers": [
                    {"super_type": "name_super",
                     "text": "Rep. Sam Reyes",
                     "names": [{"name": "Sam Reyes", "type": "person"}]},
                    {"super_type": "name_super", "text": "(R) STATE",
                     "names": []}],
                    "named_entities": [{"name": "Sam Reyes",
                                        "type": "person",
                                        "evidence": "overlay_text"}]}}
        g = self._gap(docs)
        self.assertEqual((g["count"], g["supers_total"]), (0, 2))

    def test_scene_person_never_covers_an_unextracted_chyron(self):
        # a placard person in the filmed scene is not the chyron's
        # extraction — the shot stays a gap
        docs = {9: {"supers": [{"super_type": "name_super",
                                "text": "JORDAN FELLOWS"}],
                    "named_entities": [{"name": "Sam Reyes",
                                        "type": "person",
                                        "evidence": "environment_text"}]}}
        self.assertEqual(self._gap(docs)["count"], 1)

    def test_other_super_types_ignored(self):
        docs = {1: {"supers": [{"super_type": "headline_banner",
                                "text": "MARKETS RALLY INTO THE CLOSE"}],
                    "named_entities": []}}
        g = self._gap(docs)
        self.assertEqual((g["count"], g["supers_total"]), (0, 0))

    def test_entity_from_elsewhere_does_not_cover(self):
        # the entity must be drawn FROM the super's text, not merely exist
        docs = {5: {"supers": [{"super_type": "name_super",
                                "text": "SAM REYES"}],
                    "named_entities": [{"name": "Capitol Building",
                                        "type": "place",
                                        "evidence": "landmark"}]}}
        self.assertEqual(self._gap(docs)["count"], 1)


if __name__ == "__main__":
    unittest.main()
