"""
Unit tests — cast-resolution P0. No AWS calls (sources.py is exercised only
through its pure normalization; S3 readers are covered by the P0 harness
against live runs). Run: python3 test/test_units.py
"""
from __future__ import annotations

import os
import re
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



ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# gateway modules import as top-level names (their zip is flat) —
# gateway_tool goes FIRST so `import lambda_function` resolves there
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))
# the gateway's flat zip ships lib/pb_common.py beside it (DEFAULT_FLAGS
# normalize into the identity hash at the door) — its runtime resolves
# `pb_common` top-level
sys.path.append(os.path.join(ROOT, "lib"))

# env the workflow/gateway modules read at import time (region: boto3
# clients are created at import; no call is ever made in tests)
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault(
    "STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:000000000000:stateMachine:test-cast-resolution")
os.environ.setdefault("TOOL_VERSION", "test")

from lib import class1  # noqa: E402
from lib import cooccur, persons, transcript_cues, vocatives  # noqa: E402,E401
from lib import critic  # noqa: E402

try:
    import yaml
except ImportError:                                  # pragma: no cover
    yaml = None


def _turn(sid, t0, t1, text="...", i=None):
    return {"speaker_id": sid, "start": float(t0), "end": float(t1),
            "text": text, "sentence_id": i if i is not None else int(t0)}


def _face(pid, spans, name=None, source=None, recognizer_ids=None):
    return {"face_person_id": pid, "name": name, "name_source": source,
            "screen_sec": sum(b - a for a, b in spans),
            "spans": [(float(a), float(b)) for a, b in spans],
            "recognizer_ids": recognizer_ids or {}}


# ─────────────────────────────────────────────────────────────────────────────
# transcript cues
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


class TestTranscriptCues(unittest.TestCase):
    def _one(self, text, kind):
        cues = transcript_cues.extract_cues([_turn("SPEAKER_00", 0, 5, text)])
        return [c for c in cues if c["kind"] == kind]

    def test_self_id_full_name_is_high_confidence(self):
        c = self._one("Good evening, I'm Miriam Okafor.", "self_id")
        self.assertEqual((c[0]["name"], c[0]["confidence"]),
                         ("Miriam Okafor", "high"))

    def test_self_id_given_name_only_is_low_confidence(self):
        c = self._one("Hi, I'm Kate.", "self_id")
        self.assertEqual((c[0]["name"], c[0]["confidence"]), ("Kate", "low"))

    def test_im_sorry_is_not_a_name(self):
        for phrase in ("I'm Sorry.", "I am Sure about it.", "I'm Back now."):
            self.assertEqual(self._one(phrase, "self_id"), [], phrase)

    def test_sign_off(self):
        c = self._one("Miriam Okafor, Metro News, Dallas.", "sign_off")
        self.assertEqual(c[0]["name"], "Miriam Okafor")

    def test_introduction_names_the_guest(self):
        c = self._one("Joining us now is Matt Barrow from the border.",
                      "introduction")
        self.assertEqual(c[0]["name"], "Matt Barrow")

    def test_vocative_extracted_but_low_confidence(self):
        c = self._one("Tom, back to you.", "vocative")
        self.assertEqual((c[0]["name"], c[0]["confidence"]), ("Tom", "low"))

    def test_interjections_are_not_vocatives(self):
        for phrase in ("Well, that's odd.", "Yeah, I know.", "Oh, no.",
                       "Look, over there."):
            self.assertEqual(self._one(phrase, "vocative"), [], phrase)
            self.assertEqual(self._one(phrase, "address_term"), [], phrase)

    def test_honorifics_are_address_terms_not_names(self):
        c = self._one("Detective, a word please.", "address_term")
        self.assertEqual(c[0]["name"], "Detective")
        self.assertEqual(self._one("Objection, Your Honor.", "vocative"), [])
        c = self._one("Objection, Your Honor.", "address_term")
        self.assertEqual(c[0]["name"], "Your Honor")

    def test_official_titles_are_address_terms_not_names(self):
        # measured 2026-09-24: "Ambassador" (7 votes) was vocative-named
        # onto a news guest and became his character_name; "Chairman"
        # was minted as a person on an earlier run
        for phrase in ("Ambassador, thank you for joining us.",
                       "Thank you, Mr. Chairman.",
                       "General, what's your read?"):
            self.assertEqual(self._one(phrase, "vocative"), [], phrase)


# ─────────────────────────────────────────────────────────────────────────────
# co-occurrence
# ─────────────────────────────────────────────────────────────────────────────

class TestCooccur(unittest.TestCase):
    def test_clean_interview_proposes_and_the_matcher_confirms(self):
        # one face exclusively on screen across many separate turns:
        # cooccur emits the PROPOSAL (its confirm path retired
        # 2026-09-24 — one binding engine); class1 makes the match
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 5, i=i)
                 for i in range(6)]
        faces = [_face(0, [(i * 10, i * 10 + 5) for i in range(6)])]
        out = cooccur.bind_speakers_to_faces(turns, faces)
        self.assertEqual(out["bindings"][0]["status"], "proposed")
        self.assertGreaterEqual(out["bindings"][0]["share"], 0.5)
        c1 = class1.match({"turns": turns,
                           "speakers": {"SPEAKER_00": 30.0},
                           "duration_sec": 60.0}, faces)
        self.assertEqual(c1["matches"][0]["face_person_id"], 0)

    def test_one_lucky_scene_only_proposes(self):
        # plenty of speech time but too few turns together
        turns = [_turn("SPEAKER_00", 0, 30, i=0),
                 _turn("SPEAKER_00", 40, 41, i=1)]
        faces = [_face(0, [(0, 30)])]
        out = cooccur.bind_speakers_to_faces(turns, faces)
        self.assertEqual(out["bindings"][0]["status"], "proposed")

    def test_narrator_over_broll_is_voice_only(self):
        # voice runs over many DIFFERENT faces, none stable (each face
        # owns ~1/6 of speech time, clearly below the no-partner line)
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 8, i=i)
                 for i in range(6)]
        faces = [_face(k, [(k * 10, k * 10 + 8)]) for k in range(6)]
        out = cooccur.bind_speakers_to_faces(turns, faces)
        self.assertEqual(out["speaker_class"]["SPEAKER_00"], "voice_only")
        self.assertEqual(out["bindings"], [])

    def test_no_faces_at_all_is_voice_only(self):
        turns = [_turn("SPEAKER_00", 0, 5, i=0), _turn("SPEAKER_00", 9, 14, i=1)]
        out = cooccur.bind_speakers_to_faces(turns, [])
        self.assertEqual(out["speaker_class"]["SPEAKER_00"], "voice_only")

    def test_crowd_dilutes_exclusivity(self):
        # two faces always together: neither reaches a 2x lead -> proposed
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 5, i=i)
                 for i in range(6)]
        spans = [(i * 10, i * 10 + 5) for i in range(6)]
        faces = [_face(0, spans), _face(1, spans)]
        out = cooccur.bind_speakers_to_faces(turns, faces)
        self.assertTrue(all(b["status"] == "proposed"
                            for b in out["bindings"]))

    def test_single_turn_speaker_is_insufficient(self):
        out = cooccur.bind_speakers_to_faces(
            [_turn("SPEAKER_00", 0, 3, i=0)], [_face(0, [(0, 3)])])
        self.assertEqual(out["speaker_class"]["SPEAKER_00"], "insufficient")


# ─────────────────────────────────────────────────────────────────────────────
# class-1 hierarchical matching
# ─────────────────────────────────────────────────────────────────────────────

def _desk_fixture():
    """The measured co-anchor desk: two voices alternating, both faces
    in a two-shot most of the time, each face solo-cutaway during two
    of its own voice's turns. Raw exclusivity share divides on every
    two-shot (the classifier ate both anchors); lift + solo does not."""
    turns = ([_turn("SPEAKER_00", i, i + 5, i=i) for i in (0, 20, 40, 60, 80)]
             + [_turn("SPEAKER_01", i, i + 5, i=i) for i in (10, 30, 50, 70, 90)])
    faces = [_face(0, [(0, 40), (45, 50), (55, 90), (95, 100)]),
             _face(1, [(0, 40), (45, 80), (85, 90), (95, 100)])]
    tx = {"turns": sorted(turns, key=lambda t: t["start"]),
          "speakers": {"SPEAKER_00": 25.0, "SPEAKER_01": 25.0},
          "duration_sec": 100.0, "language": "en"}
    return tx, faces


class TestClass1Matcher(unittest.TestCase):
    def test_selection_is_share_floor_not_coverage(self):
        # a 60% lead must NOT cover the line alone and exclude the 35%
        # co-lead; the 2% voice stays out (floor)
        kept = class1._top_by_share(
            [("A", 60.0), ("B", 35.0), ("C", 2.0)], 100.0)
        self.assertEqual(kept, ["A", "B"])

    def test_desk_two_shot_matches_both_anchors(self):
        tx, faces = _desk_fixture()
        out = class1.match(tx, faces)
        got = {(m["speaker_id"], m["face_person_id"])
               for m in out["matches"]}
        self.assertEqual(got, {("SPEAKER_00", 0), ("SPEAKER_01", 1)})
        self.assertEqual(out["abstentions"], [])
        self.assertEqual(out["unmatched_faces"], [])

    def test_f0_gender_veto_blocks_the_misbinding(self):
        # the measured case: a male-registry face bound to a 226 Hz
        # voice (a female guest's cluster) — the pair must be vetoed
        # and the voice must exit with a RECORDED abstention
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 5, i=i)
                 for i in range(6)]
        faces = [{**_face(0, [(i * 10, i * 10 + 5) for i in range(6)]),
                  "gender": "Male"}]
        tx = {"turns": turns, "speakers": {"SPEAKER_00": 30.0},
              "duration_sec": 60.0}
        f0 = {"SPEAKER_00": {"f0_median_hz": 226.0, "f0_n_sentences": 40}}
        out = class1.match(tx, faces, speaker_f0=f0)
        self.assertEqual(out["matches"], [])
        self.assertEqual(out["abstentions"][0]["reason"], "no_candidate")
        struck = out["abstentions"][0]["eliminated"][0]
        self.assertEqual(struck["eliminated"], "f0_gender")
        self.assertEqual(struck["f0"]["verdict"], "veto")

    def test_f0_gates_confidence_and_band(self):
        male = {"gender": "Male"}
        self.assertEqual(class1._f0_check(
            {"f0_median_hz": 226.0, "f0_n_sentences": 2}, "Male"),
            {"verdict": "unassessed"})   # too few voiced sentences
        self.assertEqual(class1._f0_check(
            {"f0_median_hz": 160.0, "f0_n_sentences": 10},
            male["gender"])["verdict"], "abstain")   # overlap band
        self.assertEqual(class1._f0_check(
            {"f0_median_hz": 120.0, "f0_n_sentences": 10},
            "Male")["verdict"], "compatible")
        self.assertEqual(class1._f0_check(
            {"f0_median_hz": 120.0, "f0_n_sentences": 10},
            "Female")["verdict"], "veto")
        self.assertEqual(class1._f0_check(None, "Male"),
                         {"verdict": "unassessed"})
        # AGE gates the strike (the measured 80-year-old actor whose
        # real 200 Hz voice was struck): elderly cross-register reading
        # abstains; a child's face is unassessable regardless of sex
        eld = class1._f0_check({"f0_median_hz": 200.0, "f0_n_sentences": 31},
                               "Male", age_range=[64, 70])
        self.assertEqual((eld["verdict"], eld["age_gate"]),
                         ("abstain", "elderly"))
        eldf = class1._f0_check({"f0_median_hz": 110.0, "f0_n_sentences": 20},
                                "Female", age_range=[62, 74])
        self.assertEqual(eldf["verdict"], "abstain")
        kid = class1._f0_check({"f0_median_hz": 250.0, "f0_n_sentences": 12},
                               "Male", age_range=[8, 14])
        self.assertEqual(kid["verdict"], "unassessed")
        # a mid-life face still strikes — the gate is the exception
        self.assertEqual(class1._f0_check(
            {"f0_median_hz": 226.0, "f0_n_sentences": 40},
            "Male", age_range=[46, 54])["verdict"], "veto")

    def test_name_join_settles_a_signal_less_two_shot(self):
        # both faces on screen for the whole programme, no cutaways:
        # lift and solo say nothing — the independently cued name does
        turns = ([_turn("SPEAKER_00", i, i + 5, i=i) for i in (0, 20, 40)]
                 + [_turn("SPEAKER_01", i, i + 5, i=i) for i in (10, 30, 50)])
        faces = [_face(0, [(0, 100)], name="Craig Merton",
                       source="collection"),
                 _face(1, [(0, 100)], name="Dana Okafor",
                       source="collection")]
        tx = {"turns": sorted(turns, key=lambda t: t["start"]),
              "speakers": {"SPEAKER_00": 15.0, "SPEAKER_01": 15.0},
              "duration_sec": 100.0}
        cues = [{"kind": "vocative_resolved", "name": "Craig",
                 "speaker_id": "SPEAKER_00", "confidence": "high"}]
        out = class1.match(tx, faces, cues=cues)
        self.assertIn(("SPEAKER_00", 0),
                      {(m["speaker_id"], m["face_person_id"])
                       for m in out["matches"]})
        m = next(x for x in out["matches"] if x["speaker_id"] == "SPEAKER_00")
        self.assertTrue(m["name_join"])
        # the other voice has no signal at all — recorded abstention
        self.assertEqual([a["speaker_id"] for a in out["abstentions"]],
                         ["SPEAKER_01"])

    def test_onset_following_settles_a_two_shot_without_solos(self):
        # shot/reverse-shot grammar: the edit cuts TO the speaker as
        # they start talking. No solos, no name cues — onset alone
        # (measured: 68-95% follow rate on scripted content)
        turns = ([_turn("SPEAKER_00", i, i + 4, i=i) for i in (10, 30, 50, 70)]
                 + [_turn("SPEAKER_01", i, i + 4, i=i) for i in (20, 40, 60, 80)])
        # face k's appearances BEGIN just after voice k's turn starts,
        # and both faces share every frame (crowd division everywhere)
        faces = [_face(0, [(i + 1, i + 9) for i in (10, 30, 50, 70)]),
                 _face(1, [(i + 1, i + 9) for i in (20, 40, 60, 80)])]
        tx = {"turns": sorted(turns, key=lambda t: t["start"]),
              "speakers": {"SPEAKER_00": 16.0, "SPEAKER_01": 16.0},
              "duration_sec": 100.0}
        out = class1.match(tx, faces)
        got = {(m["speaker_id"], m["face_person_id"]) for m in out["matches"]}
        self.assertEqual(got, {("SPEAKER_00", 0), ("SPEAKER_01", 1)})
        self.assertTrue(all(m["onset_share"] >= 0.9 for m in out["matches"]))

    def test_sole_survivor_below_bar_abstains(self):
        # the forced-by-elimination rung was DELETED (2026-09-24, zero
        # corpus firings): a sole survivor whose own evidence misses
        # the confident bar is a recorded abstention, never a match
        turns = [_turn("SPEAKER_00", 10 + i * 10, 14 + i * 10, i=i)
                 for i in range(4)]
        faces = [{**_face(0, [(5.0, 65.0)]), "gender": "Female"},
                 _face(1, [(5.0, 45.0)])]
        tx = {"turns": turns, "speakers": {"SPEAKER_00": 16.0},
              "duration_sec": 100.0}
        f0 = {"SPEAKER_00": {"f0_median_hz": 106.0, "f0_n_sentences": 18}}
        out = class1.match(tx, faces, speaker_f0=f0)
        self.assertEqual(out["matches"], [])
        a = out["abstentions"][0]
        self.assertEqual(a["reason"], "below_score")
        self.assertEqual(a["eliminated"][0]["eliminated"], "f0_gender")

    def test_never_present_is_eliminated_not_scored(self):
        # a face with zero co-presence and zero onset cuts cannot be
        # the voice's on-camera binding, however big its screen share
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 5, i=i)
                 for i in range(4)]
        faces = [_face(0, [(i * 10, i * 10 + 5) for i in range(4)]),
                 _face(1, [(60.0, 100.0)])]      # elsewhere entirely
        tx = {"turns": turns, "speakers": {"SPEAKER_00": 20.0},
              "duration_sec": 100.0}
        out = class1.match(tx, faces)
        self.assertEqual(out["matches"][0]["face_person_id"], 0)
        # the struck pair is recorded on the pair matrix, reason named
        self.assertEqual(len(out["matches"]), 1)

    def test_floor_holding_share_rides_the_monologue(self):
        # freeze one speaker's segments (user direction 2026-09-24):
        # during an uninterrupted monologue the camera favors the
        # speaker — the face present through the block earns the floor
        # share, the face that leaves earns none
        turns = [_turn("SPEAKER_00", t, t + 4, i=int(t))
                 for t in (10, 15, 20)]          # one 14s block
        faces = [_face(0, [(0.0, 30.0)]),        # present through it
                 _face(1, [(0.0, 10.0), (24.0, 100.0)])]   # absent
        tx = {"turns": turns, "speakers": {"SPEAKER_00": 12.0},
              "duration_sec": 100.0}
        out = class1.match(tx, faces)
        m = out["matches"][0]
        self.assertEqual(m["face_person_id"], 0)
        self.assertEqual(m["floor_share"], 1.0)

    def test_narrator_over_broll_abstains_with_dossier(self):
        # symmetric solo cutaways over diverse faces: every candidate
        # scores alike, margins never clear — the no-silent-exit
        # guarantee is the RECORDED abstention, not a guess
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 10, i=i)
                 for i in range(6)]
        faces = [_face(k, [(k * 10, k * 10 + 10)]) for k in range(6)]
        tx = {"turns": turns, "speakers": {"SPEAKER_00": 60.0},
              "duration_sec": 100.0}
        out = class1.match(tx, faces)
        self.assertEqual(out["matches"], [])
        a = out["abstentions"][0]
        self.assertEqual(a["reason"], "thin_margin")
        self.assertEqual(len(a["candidates"]), 2)

    def test_absent_modality_is_not_assessed(self):
        tx, faces = _desk_fixture()
        self.assertIsNone(class1.match(None, faces))
        self.assertIsNone(class1.match(tx, None))

    def test_cooccur_matched_voice_is_settled(self):
        # narrator-shaped stats would say voice_only — but class1
        # already matched this voice; cooccur must not re-judge it
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 8, i=i)
                 for i in range(6)]
        faces = [_face(k, [(k * 10, k * 10 + 8)]) for k in range(6)]
        c1 = {"speakers": ["SPEAKER_00"], "faces": [0],
              "matches": [{"speaker_id": "SPEAKER_00",
                           "face_person_id": 0}]}
        out = cooccur.bind_speakers_to_faces(turns, faces, class1=c1)
        self.assertEqual(out["speaker_class"]["SPEAKER_00"], "paired")
        self.assertEqual(out["bindings"], [])

    def test_cooccur_never_eats_an_unmatched_class1_voice(self):
        # same stats, class-1 but UNMATCHED: voice_only is barred (a
        # lead's exit is class1's abstention) and the best candidate is
        # still emitted as a PROPOSED binding for the dossier
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 8, i=i)
                 for i in range(6)]
        faces = [_face(k, [(k * 10, k * 10 + 8)]) for k in range(6)]
        c1 = {"speakers": ["SPEAKER_00"], "faces": [0], "matches": []}
        out = cooccur.bind_speakers_to_faces(turns, faces, class1=c1)
        self.assertEqual(out["speaker_class"]["SPEAKER_00"], "insufficient")
        self.assertEqual(out["bindings"][0]["status"], "proposed")

    def test_cooccur_discounts_a_matched_face_for_class2_voices(self):
        # a tail voice with the matched anchor face AND its own face in
        # every shot: undiscounted they split credit (lead 1, proposed);
        # with the anchor's presence explained by its own voice, the
        # real partner confirms
        turns = [_turn("SPEAKER_09", i * 10, i * 10 + 5, i=i)
                 for i in range(6)]
        spans = [(i * 10, i * 10 + 5) for i in range(6)]
        faces = [_face(0, spans), _face(1, spans)]
        c1 = {"speakers": ["SPEAKER_00"], "faces": [0],
              "matches": [{"speaker_id": "SPEAKER_00",
                           "face_person_id": 0}]}
        out = cooccur.bind_speakers_to_faces(turns, faces, class1=c1)
        # undiscounted they split credit and the anchor's face could
        # top the proposal; discounted, the real partner leads it
        self.assertEqual(out["bindings"][0]["face_person_id"], 1)

    def test_bind_carries_class1_and_merges_the_matches(self):
        from lib import bind_pipeline
        tx, faces = _desk_fixture()
        faces[0]["name"], faces[0]["name_source"] = ("Craig Merton",
                                                     "collection")
        doc = bind_pipeline.bind(tx, faces, None, {})
        self.assertEqual(len(doc["class1"]["matches"]), 2)
        craig = next(p for p in doc["persons"]
                     if p["name"] == "Craig Merton")
        self.assertEqual(craig["type"], "on_camera_speaker")
        self.assertIn("SPEAKER_00", craig["speaker_ids"])
        self.assertIn({"class1_assignment"},
                      [{e["mechanism"]} for e in craig["evidence"]
                       if e.get("mechanism") == "class1_assignment"])

    def test_subset_vocative_is_agreement_not_character(self):
        # class1 puts voice and face on one person at bind, so the
        # vocative "Craig" now meets registry "Craig Merton" INSIDE one
        # person: token-subset = agreement (the subset-merge rule), not
        # an actor/character split — no news anchor plays himself
        from lib import bind_pipeline
        tx, faces = _desk_fixture()
        faces[0]["name"], faces[0]["name_source"] = ("Craig Merton",
                                                     "collection")
        cues = [{"kind": "vocative_resolved", "name": "Craig",
                 "speaker_id": "SPEAKER_00", "confidence": "high"}]
        doc = bind_pipeline.bind(tx, faces, None, {}, extra_cues=cues)
        p = next(x for x in doc["persons"] if x["name"] == "Craig Merton")
        self.assertIsNone(p.get("character_name"))
        # a NON-subset spoken address still becomes the character
        cues = [{"kind": "vocative_resolved", "name": "Rodrigo",
                 "speaker_id": "SPEAKER_00", "confidence": "high"}]
        doc = bind_pipeline.bind(tx, faces, None, {}, extra_cues=cues)
        p = next(x for x in doc["persons"] if x["name"] == "Craig Merton")
        self.assertEqual(p.get("character_name"), "Rodrigo")

    def test_class1_registry_precedence_beats_cluster_noise(self):
        # the measured lead-loss class: the matched voice cluster
        # carries CONFLICTING spoken cues (absorbed content) — the sole
        # registry name must still win, noise demoted to aliases
        from lib import bind_pipeline
        tx, faces = _desk_fixture()
        faces[0]["name"], faces[0]["name_source"] = ("Victor Hale",
                                                     "collection")
        cues = [{"kind": "self_id", "name": "Detective Croft",
                 "speaker_id": "SPEAKER_00", "confidence": "high"},
                {"kind": "vocative_resolved", "name": "Henry",
                 "speaker_id": "SPEAKER_00", "confidence": "high"}]
        doc = bind_pipeline.bind(tx, faces, None, {}, extra_cues=cues)
        p = next(x for x in doc["persons"]
                 if "SPEAKER_00" in x["speaker_ids"])
        self.assertEqual(p["name"], "Victor Hale")
        self.assertEqual(p["aliases"], ["Detective Croft", "Henry"])
        self.assertIn("class1_name_precedence",
                      [e.get("mechanism") for e in p["evidence"]
                       if isinstance(e, dict)])

    def test_threading_requires_a_class1_voice(self):
        # an over-merged tail cluster scatters a guest's speech across
        # every eighth without carrying real share: with a class-1
        # assessment present, the anchor rung demands a class-1 voice
        from lib import reads
        person = {"person_id": 0, "type": "on_camera_speaker",
                  "name": "Guest", "speaker_ids": ["SPEAKER_20"],
                  "face_ids": [5], "role": None,
                  "name_mechanisms": ["collection"],
                  "evidence": [{"kind": "overlay_name", "name": "x"}]}
        spans = {"SPEAKER_20": [(i * 100.0, i * 100.0 + 5) for i in range(8)]}
        doc = {"persons": [dict(person)],
               "class1": {"speakers": ["SPEAKER_00"]}}
        reads.assign_roles(doc, duration_sec=800.0, speech_spans=spans)
        self.assertNotEqual(doc["persons"][0].get("role"), "anchor")
        # without the assessment (no class1 in doc) the spread rule
        # stands alone, as before
        doc = {"persons": [dict(person)]}
        reads.assign_roles(doc, duration_sec=800.0, speech_spans=spans)
        self.assertEqual(doc["persons"][0]["role"], "anchor")

    def test_dominant_repeated_name_names_the_unregistered_host(self):
        # the measured case: a host absent from the collection whose
        # own chyron repeats while every absorbed name shows once —
        # repetition is the corroboration (votes+lead, the vocative
        # constants); a genuine two-way repeat (host 4, absorbed
        # press-clip speaker 3) must still abstain
        from lib import bind_pipeline
        tx, faces = _desk_fixture()
        overlay = ([{"kind": "overlay_name", "name": "Jon Ferrell",
                     "t0": t, "t1": t + 2.0, "shot_id": i}
                    for i, t in enumerate((1.0, 21.0, 41.0, 61.0))]
                   + [{"kind": "overlay_name", "name": "Guest One",
                       "t0": 81.0, "t1": 83.0, "shot_id": 9}])
        doc = bind_pipeline.bind(tx, faces, overlay, {})
        p = next(x for x in doc["persons"]
                 if "SPEAKER_00" in x["speaker_ids"])
        self.assertEqual(p["name"], "Jon Ferrell")
        self.assertIn("dominant_name",
                      [e.get("mechanism") for e in p["evidence"]
                       if isinstance(e, dict)])
        # 4-vs-3 fails the lead: abstain, both stay aliases
        overlay += [{"kind": "overlay_name", "name": "Guest One",
                     "t0": t, "t1": t + 2.0, "shot_id": 10 + i}
                    for i, t in enumerate((22.0, 42.0))]
        doc = bind_pipeline.bind(tx, faces, overlay, {})
        p = next(x for x in doc["persons"]
                 if "SPEAKER_00" in x["speaker_ids"])
        self.assertIsNone(p["name"])
        self.assertIn("Jon Ferrell", p["aliases"])

    def test_case_variant_spellings_are_one_name(self):
        # the measured quiz-asset abstention (2026-10-05): the witness
        # read the same chyron as 'JOANNE' on some frames and 'Joanne'
        # on others, and the case-sensitive vote treated them as RIVAL
        # names (2-vs-2, lead fails) — a fully-evidenced contestant
        # shipped nameless. Spellings that casefold-join are ONE name.
        from lib import bind_pipeline
        tx, faces = _desk_fixture()
        overlay = [{"kind": "overlay_name", "name": nm,
                    "t0": t, "t1": t + 2.0, "shot_id": i}
                   for i, (nm, t) in enumerate(
                       (("JON FERRELL", 1.0), ("Jon Ferrell", 21.0),
                        ("JON FERRELL", 41.0), ("Jon Ferrell", 61.0)))]
        doc = bind_pipeline.bind(tx, faces, overlay, {})
        p = next(x for x in doc["persons"]
                 if "SPEAKER_00" in x["speaker_ids"])
        self.assertEqual(p["name"], "Jon Ferrell")

    def test_solo_chyron_votes_join_across_case(self):
        # one chyron read twice with different casing is TWO votes for
        # one name, and the display form is the natural casing
        faces = [{"face_person_id": 0, "name": None, "name_source": None,
                  "screen_sec": 20.0, "spans": [(10.0, 30.0)],
                  "recognizer_ids": {}}]
        overlay = [{"kind": "overlay_name", "name": "JOANNE",
                    "t0": 11.0, "t1": 13.0},
                   {"kind": "overlay_name", "name": "Joanne",
                    "t0": 21.0, "t1": 23.0}]
        self.assertEqual(persons.name_solo_chyron_faces(faces, overlay), 1)
        self.assertEqual(faces[0]["name"], "Joanne")

    def test_self_conflicting_cluster_has_no_name_join(self):
        # a cluster claiming to be TWO people (the measured cross-
        # merged co-anchor desk: the absorbed anchor's self-ID tipped
        # four tied pairs the wrong way) joins with nobody; its pairs
        # carry the mixed_voice flag for the dossier
        tx, faces = _desk_fixture()
        faces[0]["name"], faces[0]["name_source"] = ("Nora Vane",
                                                     "collection")
        cues = [{"kind": "self_id", "name": "Nora Vane",
                 "speaker_id": "SPEAKER_00", "confidence": "high"},
                {"kind": "self_id", "name": "Zola Erim",
                 "speaker_id": "SPEAKER_00", "confidence": "high"}]
        out = class1.match(tx, faces, cues=cues)
        m = next(x for x in out["matches"]
                 if x["speaker_id"] == "SPEAKER_00")
        self.assertFalse(m["name_join"])       # join voided
        self.assertTrue(m["mixed_voice"])
        # a clean single self-claim still joins
        out = class1.match(tx, faces, cues=cues[:1])
        m = next(x for x in out["matches"]
                 if x["speaker_id"] == "SPEAKER_00")
        self.assertTrue(m["name_join"])

    def test_rejected_names_never_reach_llm_windows(self):
        # the screens' verdict is a settled NON-answer: a rejected
        # word/interjection must not buy an LLM window (measured: 48%
        # of round-1 windows were junk sites before this gate)
        from lib import vocatives, vocatives_llm, transcript_cues
        turns = ([_turn("SPEAKER_00", 0, 4, "Mierda, that hurt.", 0),
                  _turn("SPEAKER_01", 5, 9, "He said mierda twice.", 1),
                  _turn("SPEAKER_00", 10, 14, "Dana, ask Dana about it.", 2),
                  _turn("SPEAKER_01", 15, 19, "Fine.", 3)])
        cues = transcript_cues.extract_cues(turns)
        screen = vocatives.name_screen(turns)
        kept = [c for c in cues
                if c["kind"] != "vocative" or screen(c["name"])]
        names = {w["name"] for w in vocatives_llm.build_windows(
            turns, kept, resolved=[])}
        self.assertIn("Dana", names)
        self.assertNotIn("Mierda", names)

    def test_parse_faces_carries_registry_gender_over_visual(self):
        from lib import sources
        doc = {"people": [{"person_id": 0, "named": False,
                           "known_gender": "Female", "gender": "Male",
                           "total_screen_time_sec": 1.0,
                           "appearances": []}]}
        self.assertEqual(sources.parse_faces(doc)[0]["gender"], "Female")

    def test_parse_speaker_f0_keeps_only_clusters_with_stats(self):
        from lib import sources
        gs = {"speaker_models": {
            "SPEAKER_00": {"f0_median_hz": 210.5, "f0_n_sentences": 12},
            "SPEAKER_01": {"confidence": 0.4}}}
        f0 = sources.parse_speaker_f0(gs)
        self.assertEqual(list(f0), ["SPEAKER_00"])
        self.assertEqual(f0["SPEAKER_00"]["f0_median_hz"], 210.5)


# ─────────────────────────────────────────────────────────────────────────────
# vocative resolution
# ─────────────────────────────────────────────────────────────────────────────

def _dialog(*lines, gap=1.0, dur=3.0):
    """lines: (speaker, text) -> alternating turns."""
    turns, t = [], 0.0
    for i, (sid, text) in enumerate(lines):
        turns.append(_turn(sid, t, t + dur, text, i))
        t += dur + gap
    return turns


class TestVocatives(unittest.TestCase):
    def _resolve(self, turns):
        cues = transcript_cues.extract_cues(turns)
        return vocatives.resolve_vocatives(turns, cues)

    def test_repeated_address_names_the_partner(self):
        turns = _dialog(
            ("SPEAKER_00", "Tom, what do you make of this?"),
            ("SPEAKER_01", "It is hard to say."),
            ("SPEAKER_00", "Tom, and the numbers?"),
            ("SPEAKER_01", "They are up."))
        out = self._resolve(turns)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["speaker_id"], out[0]["name"],
                          out[0]["votes"]), ("SPEAKER_01", "Tom", 2))

    def test_ordinary_words_never_resolve_however_many_votes(self):
        """The Mierda/Ah class (measured 2026-09-23): interjections and
        expletives crossed the vote floor and minted persons at
        confidence:high. The screen is structural, not a stoplist: a
        token the transcript ALSO uses lowercase is a word — in any
        Latin-script language — and votes cannot launder it."""
        turns = _dialog(
            ("SPEAKER_00", "Mierda, that was close."),
            ("SPEAKER_01", "It really was."),
            ("SPEAKER_00", "Mierda, again!"),
            ("SPEAKER_01", "Calm down."),
            # the same token, lowercase, in ordinary use
            ("SPEAKER_02", "He says mierda constantly."))
        self.assertEqual(self._resolve(turns), [])

    def test_a_name_word_survives_when_capitalized_use_dominates(self):
        """The cross-content guard on the guard: names that are also
        common words (Grace, Rose, Mark) exist in every drama, and ONE
        ASR-lowercased occurrence must not erase them. Majority rule:
        mid-sentence capitalized uses outvote lowercase ones."""
        turns = _dialog(
            ("SPEAKER_00", "Grace, what do you make of this?"),
            ("SPEAKER_01", "It is hard to say."),
            ("SPEAKER_00", "Grace, and the numbers?"),
            ("SPEAKER_01", "They are up."),
            ("SPEAKER_02", "I told Grace about it and asked Grace twice."),
            ("SPEAKER_02", "There but for the grace of luck."))
        out = self._resolve(turns)
        self.assertEqual([(o["speaker_id"], o["name"]) for o in out],
                         [("SPEAKER_01", "Grace")])

    def test_boundary_only_interjection_abstains(self):
        """The Yowie/Ah class survived the majority rule live
        (2026-09-23): an exclamation is ALWAYS sentence-initial, so it
        leaves no lowercase evidence. When the transcript references
        other tokens mid-sentence, a candidate never referenced
        mid-sentence is an interjection, not a name."""
        turns = _dialog(
            ("SPEAKER_00", "Yowie, that was something!"),
            ("SPEAKER_01", "It sure was."),
            ("SPEAKER_00", "Yowie, look at that!"),
            ("SPEAKER_01", "Amazing."),
            # a real mid-sentence reference exists for ANOTHER token, so
            # the referenced set is populated (a real programme's always is)
            ("SPEAKER_02", "We asked Dana about it earlier."))
        self.assertEqual(self._resolve(turns), [])

    def test_a_mostly_lowercase_word_still_abstains(self):
        turns = _dialog(
            ("SPEAKER_00", "Tom, what do you make of this?"),
            ("SPEAKER_01", "It is hard to say."),
            ("SPEAKER_00", "Tom, and the numbers?"),
            ("SPEAKER_01", "They are up."),
            ("SPEAKER_02", "A tom cat walked by."))
        # only sentence-initial capitals (they prove nothing) vs one
        # lowercase use -> the token reads as a word: abstain
        self.assertEqual(self._resolve(turns), [])

    def test_single_vote_binds_nothing(self):
        turns = _dialog(
            ("SPEAKER_00", "Tom, what do you make of this?"),
            ("SPEAKER_01", "It is hard to say."))
        self.assertEqual(self._resolve(turns), [])

    def test_trailing_vocative_targets_next_speaker(self):
        turns = _dialog(
            ("SPEAKER_00", "Over to you, Tom."),
            ("SPEAKER_01", "Thanks."),
            ("SPEAKER_02", "Back to you, Tom."),
            ("SPEAKER_01", "Thank you both."))
        out = self._resolve(turns)
        self.assertEqual([(o["speaker_id"], o["name"]) for o in out],
                         [("SPEAKER_01", "Tom")])
        self.assertEqual(sorted(out[0]["addressers"]),
                         ["SPEAKER_00", "SPEAKER_02"])

    def test_two_names_on_one_speaker_is_noise(self):
        turns = _dialog(
            ("SPEAKER_00", "Tom, your thoughts?"),
            ("SPEAKER_01", "Well."),
            ("SPEAKER_00", "Mike, your thoughts?"),
            ("SPEAKER_01", "Hmm."),
            ("SPEAKER_00", "Tom, again?"),
            ("SPEAKER_01", "Yes."),
            ("SPEAKER_00", "Mike, again?"),
            ("SPEAKER_01", "No."))
        self.assertEqual(self._resolve(turns), [])

    def test_distant_turns_do_not_vote(self):
        turns = _dialog(
            ("SPEAKER_00", "Tom, are you there?"),
            ("SPEAKER_01", "Yes."),
            ("SPEAKER_00", "Tom, still there?"),
            ("SPEAKER_01", "Yes."), gap=60.0)
        self.assertEqual(self._resolve(turns), [])

    def test_resolved_vocative_names_the_person(self):
        turns = _dialog(
            ("SPEAKER_00", "Tom, what do you make of this?"),
            ("SPEAKER_01", "It is hard to say."),
            ("SPEAKER_00", "Tom, and the numbers?"),
            ("SPEAKER_01", "They are up."))
        cues = transcript_cues.extract_cues(turns)
        cues += vocatives.resolve_vocatives(turns, cues)
        doc = persons.build_persons(_transcript(turns), None, None, cues,
                                    None, {})
        named = {p["speaker_ids"][0]: p["name"] for p in doc["persons"]}
        self.assertEqual(named["SPEAKER_01"], "Tom")
        self.assertIsNone(named["SPEAKER_00"])


# ─────────────────────────────────────────────────────────────────────────────
# person assembly
# ─────────────────────────────────────────────────────────────────────────────

def _transcript(turns):
    speakers = {}
    for t in turns:
        speakers[t["speaker_id"]] = (speakers.get(t["speaker_id"], 0.0)
                                     + t["end"] - t["start"])
    return {"turns": turns, "speakers": speakers, "duration_sec": 100}


class TestPersons(unittest.TestCase):
    def test_confirmed_binding_merges_and_names_from_face_source(self):
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 5, i=i)
                 for i in range(6)]
        faces = [_face(0, [(i * 10, i * 10 + 5) for i in range(6)],
                       name="Victor Hale", source="celebrity")]
        tx = _transcript(turns)
        c1 = class1.match(tx, faces)
        co = cooccur.bind_speakers_to_faces(turns, faces, class1=c1)
        co["bindings"] = list(c1["matches"]) + co["bindings"]
        doc = persons.build_persons(tx, faces, co, [], [],
                                    {"speech_diarization": "s3://a",
                                     "face_recognition": "s3://b"})
        named = [p for p in doc["persons"] if p["name"]]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0]["type"], "on_camera_speaker")
        self.assertEqual(named[0]["name"], "Victor Hale")
        self.assertEqual(named[0]["speaker_ids"], ["SPEAKER_00"])
        self.assertEqual(named[0]["face_ids"], [0])

    def test_proposed_binding_never_merges(self):
        turns = [_turn("SPEAKER_00", 0, 30, i=0), _turn("SPEAKER_00", 40, 41, i=1)]
        faces = [_face(0, [(0, 30)])]
        co = cooccur.bind_speakers_to_faces(turns, faces)
        doc = persons.build_persons(_transcript(turns), faces, co, [], [], {})
        self.assertEqual(len(doc["persons"]), 2)   # separate, not merged

    def test_person_appearances_merge_member_face_spans(self):
        # the consolidated presence timeline (2026-09-08): member
        # faces' spans union into begin_ms/end_ms appearances —
        # facemetadata's shape, one person-level answer to "when on
        # screen" so consumers stop re-joining raw clusters
        faces = [_face(0, [(0, 10), (20, 30)]),
                 _face(1, [(8, 12)])]
        doc = persons.build_persons(None, faces, None, None, None, {})
        by_fid = {tuple(p["face_ids"]): p for p in doc["persons"]}
        solo = by_fid[(1,)]
        self.assertEqual(solo["appearances"],
                         [{"begin_ms": 8000, "end_ms": 12000}])
        first = by_fid[(0,)]
        self.assertEqual(first["appearances"],
                         [{"begin_ms": 0, "end_ms": 10000},
                          {"begin_ms": 20000, "end_ms": 30000}])

    def test_faceless_run_exports_speech_basis_appearances(self):
        # GDPR posture: no face source at all — the PRODUCER degrades
        # presence to merged speaker turn spans and says so
        turns = [_turn("S0", 1, 4, "One.", 0),
                 _turn("S0", 3.5, 6, "Two.", 1)]
        doc = persons.build_persons(_transcript(turns), None, None,
                                    None, None, {})
        p0 = doc["persons"][0]
        self.assertEqual(p0["appearances_basis"], "speech")
        self.assertEqual(p0["appearances"],
                         [{"begin_ms": 1000, "end_ms": 6000}])

    def test_name_conflict_abstains(self):
        turns = [_turn("SPEAKER_00", 0, 5, "I'm Miriam Okafor.", 0),
                 _turn("SPEAKER_00", 10, 15, "Jane Doe, Metro News, Dallas.", 1)]
        cues = transcript_cues.extract_cues(turns)
        doc = persons.build_persons(_transcript(turns), None, None, cues,
                                    None, {})
        p = doc["persons"][0]
        self.assertIsNone(p["name"])
        self.assertEqual(p["aliases"], ["Jane Doe", "Miriam Okafor"])

    def test_self_id_names_speaker_without_any_faces(self):
        turns = [_turn("SPEAKER_00", 0, 5, "Good evening, I'm Miriam Okafor.", 0),
                 _turn("SPEAKER_00", 10, 15, "More news after the break.", 1)]
        cues = transcript_cues.extract_cues(turns)
        doc = persons.build_persons(_transcript(turns), None, None, cues,
                                    None, {"speech_diarization": "s3://a"})
        self.assertEqual(doc["persons"][0]["name"], "Miriam Okafor")

    def test_overlay_attaches_to_the_face_on_screen_not_the_speaker(self):
        # the 2026-09-30 audit class: the anchor narrates ("joining us
        # now...") over the GUEST's lower third — the chyron names its
        # SUBJECT on screen, never the narrating voice
        turns = [_turn("SPEAKER_00", 0, 30, i=0)]      # one active voice
        guest = _face(7, [(10, 20)])
        overlay = [{"kind": "overlay_name", "name": "Danny Kovac",
                    "t0": 12.0, "t1": 14.0, "shot_id": 7}]
        doc = persons.build_persons(_transcript(turns), [guest], None,
                                    [], overlay, {})
        by_face = {tuple(p["face_ids"]): p for p in doc["persons"]
                   if p["face_ids"]}
        self.assertEqual(by_face[(7,)]["name"], "Danny Kovac")
        spk = next(p for p in doc["persons"]
                   if p["speaker_ids"] == ["SPEAKER_00"])
        self.assertIsNone(spk["name"])
        self.assertEqual(by_face[(7,)]["evidence"][-1]["attach_basis"],
                         "face_on_screen")

    def test_overlay_two_shot_resolves_by_face_name_match(self):
        turns = [_turn("SPEAKER_00", 0, 30, i=0)]
        anchor = _face(1, [(0, 30)], name="Miriam Okafor",
                       source="celebrity")
        guest = _face(2, [(5, 25)], name="Danny Kovac", source="celebrity")
        overlay = [{"kind": "overlay_name", "name": "Danny Kovac",
                    "t0": 12.0, "t1": 14.0, "shot_id": 7},
                   {"kind": "overlay_name", "name": "Someone Else",
                    "t0": 16.0, "t1": 18.0, "shot_id": 8}]
        doc = persons.build_persons(_transcript(turns),
                                    [anchor, guest], None, [],
                                    overlay, {})
        g = next(p for p in doc["persons"] if p["face_ids"] == [2])
        self.assertEqual(g["name"], "Danny Kovac")
        self.assertEqual(
            g["evidence"][-1]["attach_basis"], "face_name_match")
        # the unmatched two-shot chyron attached to no one
        blob = [p["name"] for p in doc["persons"]] + \
            [a for p in doc["persons"] for a in p["aliases"]]
        self.assertNotIn("Someone Else", blob)

    def test_overlay_depicted_face_falls_back_to_speaker(self):
        # a chyron over a PHOTO of someone (majority-depicted face)
        # must not capture the name; the single live voice still can
        turns = [_turn("SPEAKER_00", 0, 30, i=0)]
        photo = dict(_face(9, [(10, 20)]), depicted_share=0.9)
        overlay = [{"kind": "overlay_name", "name": "Matt Barrow",
                    "t0": 12.0, "t1": 14.0, "shot_id": 7}]
        doc = persons.build_persons(_transcript(turns), [photo], None,
                                    [], overlay, {})
        spk = next(p for p in doc["persons"]
                   if p["speaker_ids"] == ["SPEAKER_00"])
        self.assertEqual(spk["name"], "Matt Barrow")

    def test_overlay_name_binds_only_when_one_speaker_active(self):
        turns = [_turn("SPEAKER_00", 0, 10, i=0), _turn("SPEAKER_01", 10, 20, i=1),
                 _turn("SPEAKER_00", 20, 30, i=2)]
        overlay = [{"kind": "overlay_name", "name": "Matt Barrow",
                    "t0": 12.0, "t1": 14.0, "shot_id": 7}]
        doc = persons.build_persons(_transcript(turns), None, None, [],
                                    overlay, {})
        by_speaker = {p["speaker_ids"][0]: p for p in doc["persons"]}
        self.assertEqual(by_speaker["SPEAKER_01"]["name"], "Matt Barrow")
        self.assertIsNone(by_speaker["SPEAKER_00"]["name"])

    def test_overlay_ambiguous_across_speakers_abstains(self):
        turns = [_turn("SPEAKER_00", 0, 10, i=0), _turn("SPEAKER_01", 8, 20, i=1)]
        overlay = [{"kind": "overlay_name", "name": "Matt Barrow",
                    "t0": 7.0, "t1": 9.0, "shot_id": 7}]
        doc = persons.build_persons(_transcript(turns), None, None, [],
                                    overlay, {})
        self.assertTrue(all(p["name"] is None for p in doc["persons"]))

    def test_overlay_during_crosstalk_abstains(self):
        # the transcript shows ONE active voice, but overlap_segments
        # prove a second was live under the chyron — SUGGESTIVE
        # evidence may only widen ambiguity (abstain), never attach
        turns = [_turn("SPEAKER_00", 0, 10, i=0),
                 _turn("SPEAKER_01", 10, 20, i=1)]
        t = _transcript(turns)
        t["overlap_segments"] = [{"start": 12.0, "end": 14.5,
                                  "speakers": ["SPEAKER_00",
                                               "SPEAKER_01"]}]
        overlay = [{"kind": "overlay_name", "name": "Matt Barrow",
                    "t0": 12.0, "t1": 14.0, "shot_id": 7}]
        doc = persons.build_persons(t, None, None, [], overlay, {})
        self.assertTrue(all(p["name"] is None for p in doc["persons"]))

    def test_overlay_outside_crosstalk_still_binds(self):
        # a span elsewhere in the asset must not poison a clean chyron
        turns = [_turn("SPEAKER_00", 0, 10, i=0),
                 _turn("SPEAKER_01", 10, 20, i=1)]
        t = _transcript(turns)
        t["overlap_segments"] = [{"start": 2.0, "end": 3.0,
                                  "speakers": ["SPEAKER_00",
                                               "SPEAKER_01"]}]
        overlay = [{"kind": "overlay_name", "name": "Matt Barrow",
                    "t0": 12.0, "t1": 14.0, "shot_id": 7}]
        doc = persons.build_persons(t, None, None, [], overlay, {})
        by_speaker = {p["speaker_ids"][0]: p for p in doc["persons"]}
        self.assertEqual(by_speaker["SPEAKER_01"]["name"], "Matt Barrow")

    def test_silent_face_type_and_unresolved(self):
        faces = [_face(3, [(0, 12)])]
        doc = persons.build_persons(None, faces, None, None, None,
                                    {"face_recognition": "s3://b"})
        self.assertEqual(doc["persons"][0]["type"], "silent_face")
        self.assertEqual(doc["unresolved"]["faces"], [3])

    def test_same_name_cross_mechanism_merge(self):
        # the Mia Chen case: overlay-named voice + celebrity-named
        # face, co-occurrence never confirmed -> ONE person by name
        turns = [_turn("SPEAKER_00", 0, 10, i=0), _turn("SPEAKER_00", 20, 30, i=1)]
        faces = [_face(5, [(50, 60)], name="Mia Chen",
                       source="collection")]
        overlay = [{"kind": "overlay_name", "name": "Mia Chen",
                    "t0": 2.0, "t1": 4.0, "shot_id": 1}]
        doc = persons.build_persons(_transcript(turns), faces, None, [],
                                    overlay, {})
        self.assertEqual(len(doc["persons"]), 1)
        p = doc["persons"][0]
        self.assertEqual(p["name"], "Mia Chen")
        self.assertEqual((p["speaker_ids"], p["face_ids"]),
                         (["SPEAKER_00"], [5]))
        self.assertIn("name_agreement",
                      [e.get("mechanism") for e in p["evidence"]])

    def test_different_names_never_merge(self):
        faces = [_face(0, [(0, 5)], name="Miriam Okafor", source="celebrity"),
                 _face(1, [(6, 9)], name="Jane Doe", source="celebrity")]
        doc = persons.build_persons(None, faces, None, None, None, {})
        self.assertEqual(len(doc["persons"]), 2)

    def test_speech_coherence_repairs_and_records(self):
        """The 2026-09-22 scramble class: a person carrying ANOTHER
        cluster's seconds. The invariant recomputes from the speaker
        map at every write point — and records the repair, because
        silent self-healing would hide exactly the recurrence this
        exists to catch."""
        speakers = {"SPEAKER_00": 550.6, "SPEAKER_01": 16.5}
        doc = {"persons": [
            {"person_id": 0, "speaker_ids": ["SPEAKER_01"],
             "speech_sec": 550.6, "evidence": []},        # scrambled
            {"person_id": 1, "speaker_ids": ["SPEAKER_00"],
             "speech_sec": 550.6, "evidence": []},        # coherent
            {"person_id": 2, "speaker_ids": [],
             "speech_sec": 0.0, "evidence": []},          # faces-only
        ]}
        out = persons.enforce_speech_coherence(doc, speakers)
        self.assertEqual(out["persons"][0]["speech_sec"], 16.5)
        self.assertEqual(
            out["vetoes"],
            [{"check": "speech_seconds_incoherent", "person_ids": [0],
              "was": 550.6, "recomputed": 16.5}])
        self.assertIn("coherence_repair",
                      [e.get("mechanism")
                       for e in out["persons"][0]["evidence"]])
        # the coherent person and the speaker-less person are untouched
        self.assertEqual(out["persons"][1]["speech_sec"], 550.6)
        self.assertEqual(out["persons"][1]["evidence"], [])
        self.assertEqual(out["persons"][2]["speech_sec"], 0.0)

    def test_subset_name_merges_the_first_name_voice_onto_the_full_name(self):
        # the co-anchor split (2026-09-23): "Miriam" (vocative voice,
        # split-box defeats co-occurrence entirely) + "Miriam Okafor"
        # (collection face) -> one person under the FULL name
        turns = [_turn("SPEAKER_00", 0, 10, i=0),
                 _turn("SPEAKER_00", 20, 30, i=1)]
        cues = [{"kind": "vocative_resolved", "name": "Miriam",
                 "speaker_id": "SPEAKER_00", "votes": 4,
                 "addressers": ["SPEAKER_01"], "confidence": "high"}]
        faces = [_face(5, [(50, 60)], name="Miriam Okafor",
                       source="collection")]
        doc = persons.build_persons(_transcript(turns), faces, None, cues,
                                    None, {})
        self.assertEqual(len(doc["persons"]), 1)
        p = doc["persons"][0]
        self.assertEqual(p["name"], "Miriam Okafor")
        self.assertEqual((p["speaker_ids"], p["face_ids"]),
                         (["SPEAKER_00"], [5]))
        self.assertIn("subset_name",
                      [e.get("mechanism") for e in p["evidence"]])

    def test_roles_anchor_needs_a_face_and_guests_are_the_registry_rest(self):
        """The migrated ladder (2026-09-23): speech threading makes an
        anchor ONLY with a bound on-camera face — an over-merged
        voice-only cluster threading the programme must not (the
        measured 487 s faceless 'anchor'); registry-named on-camera
        speakers otherwise read guest_or_subject; silent faces carry
        no role at all."""
        from lib import reads
        dur = 3600.0
        threaded = [(t, t + 30.0) for t in (10, 500, 1200, 1900, 2600, 3300)]
        speech = {"SPEAKER_00": threaded,      # threads the programme
                  "SPEAKER_01": threaded,      # threads too, but voice-only
                  "SPEAKER_02": [(1500.0, 1800.0)]}   # one guest block
        def mkdoc():
            return {"persons": [
                {"person_id": 0, "type": "on_camera_speaker",
                 "name": "Miriam Okafor", "speaker_ids": ["SPEAKER_00"],
                 "appearances": [],
                 "name_mechanisms": ["collection"], "evidence": []},
                {"person_id": 1, "type": "voice_only",
                 "name": "Dana West", "speaker_ids": ["SPEAKER_01"],
                 "appearances": [],
                 "name_mechanisms": ["transcript_cue"], "evidence": []},
                {"person_id": 2, "type": "on_camera_speaker",
                 "name": "Victor Hale", "speaker_ids": ["SPEAKER_02"],
                 # face APPEARANCES scattered across the programme by
                 # teases — the measured trap that crowned three guests
                 "appearances": [{"begin_ms": int(t * 1000),
                                  "end_ms": int((t + 10) * 1000)}
                                 for t in (60, 900, 1600, 2700, 3400)],
                 "name_mechanisms": ["celebrity"],
                 "evidence": [{"kind": "overlay_name",
                               "name": "Victor Hale", "t0": 1.0,
                               "t1": 2.0, "shot_id": 3}]},
                {"person_id": 3, "type": "silent_face",
                 "name": "Tomas Vega", "speaker_ids": [],
                 "appearances": [{"begin_ms": 0, "end_ms": 9000}],
                 "name_mechanisms": ["collection"], "evidence": []},
            ]}
        out = reads.assign_roles(mkdoc(), duration_sec=dur,
                                 speech_spans=speech)["persons"]
        self.assertEqual(out[0]["role"], "anchor")
        self.assertIn("speech_threading",
                      [e.get("mechanism") for e in out[0]["evidence"]])
        self.assertIsNone(out[1].get("role"))    # faceless: never anchor
        # tease-scattered FACE + one-block VOICE = guest, never anchor
        self.assertEqual(out[2]["role"], "guest_or_subject")
        self.assertIsNone(out[3].get("role"))    # silent face: abstain
        # NO broadcast chrome (a film: zero chyrons) -> behavioral rungs
        # never run, however programme-spanning a lead's speech is
        movie = mkdoc()
        movie["persons"][2]["evidence"] = []
        out2 = reads.assign_roles(movie, duration_sec=dur,
                                  speech_spans=speech)["persons"]
        self.assertTrue(all(p.get("role") is None for p in out2))
        # witness TITLE reads are not chrome (measured live 2026-09-24:
        # a drama's opening "PRODUCER Jane Doe" card opened the gate
        # and every lead threaded to anchor) — same evidence shape,
        # source title-read, and the gate stays shut
        credits_movie = mkdoc()
        credits_movie["persons"][2]["evidence"] = [
            {"kind": "overlay_name", "name": "Jane Doe",
             "role_text": "PRODUCER", "source": "title-read"}]
        out3 = reads.assign_roles(credits_movie, duration_sec=dur,
                                  speech_spans=speech)["persons"]
        self.assertTrue(all(p.get("role") is None for p in out3))

    def test_registry_plus_vocative_is_the_character_not_a_conflict(self):
        # the mitj lead-actor case (2026-09-23): the collection names the
        # ACTOR, dialogue addresses the CHARACTER — one person, actor as
        # name, spoken address as character_name (a later credits read
        # overwrites it: deterministic beats inferred)
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 5, i=i)
                 for i in range(6)]
        cues = [{"kind": "vocative_resolved", "name": "Dana",
                 "speaker_id": "SPEAKER_00", "votes": 3,
                 "addressers": ["SPEAKER_01"], "confidence": "high"}]
        faces = [_face(5, [(i * 10, i * 10 + 5) for i in range(6)],
                       name="Miriam Okafor", source="collection")]
        tx = _transcript(turns)
        c1 = class1.match(tx, faces, cues=cues)
        co = cooccur.bind_speakers_to_faces(turns, faces, class1=c1)
        co["bindings"] = list(c1["matches"]) + co["bindings"]
        doc = persons.build_persons(tx, faces, co, cues, None, {})
        p = next(x for x in doc["persons"] if x.get("name"))
        self.assertEqual(p["name"], "Miriam Okafor")
        self.assertEqual(p["character_name"], "Dana")
        self.assertIn("spoken_address",
                      [e.get("mechanism") for e in p["evidence"]])

    def test_subset_merge_requires_complementary_halves(self):
        """The scripted-content hazard: a CHARACTER sharing an actor's
        first name must not weld onto that actor. The actor already
        speaks (voice bound), so the merge is not completing a missing
        half — it would add a second voice to a speaking person."""
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 5, i=i)
                 for i in range(6)] + \
                [_turn("SPEAKER_07", 100 + i * 10, 100 + i * 10 + 4, i=6 + i)
                 for i in range(4)]
        cues = [{"kind": "vocative_resolved", "name": "Miriam",
                 "speaker_id": "SPEAKER_07", "votes": 4,
                 "addressers": ["SPEAKER_00"], "confidence": "high"}]
        faces = [_face(5, [(i * 10, i * 10 + 5) for i in range(6)],
                       name="Miriam Okafor", source="collection")]
        tx = _transcript(turns)
        c1 = class1.match(tx, faces, cues=cues)
        co = cooccur.bind_speakers_to_faces(turns, faces, class1=c1)
        co["bindings"] = list(c1["matches"]) + co["bindings"]
        doc = persons.build_persons(tx, faces, co, cues, None, {})
        # the actor (voice+face bound) and the "Miriam" voice stay apart
        names = sorted(p["name"] for p in doc["persons"] if p.get("name"))
        self.assertEqual(names, ["Miriam", "Miriam Okafor"])

    def test_ambiguous_first_name_abstains_from_subset_merge(self):
        # two supersets = an ambiguous first name: nobody merges
        turns = [_turn("SPEAKER_00", 0, 10, i=0)]
        cues = [{"kind": "vocative_resolved", "name": "Miriam",
                 "speaker_id": "SPEAKER_00", "votes": 4,
                 "addressers": ["SPEAKER_01"], "confidence": "high"}]
        faces = [_face(5, [(50, 60)], name="Miriam Okafor",
                       source="collection"),
                 _face(6, [(70, 80)], name="Miriam Vega",
                       source="collection")]
        doc = persons.build_persons(_transcript(turns), faces, None, cues,
                                    None, {})
        self.assertEqual(len(doc["persons"]), 3)

    def test_conflicted_person_is_excluded_from_name_merge(self):
        turns = [_turn("SPEAKER_00", 0, 5, "I'm Miriam Okafor.", 0),
                 _turn("SPEAKER_00", 6, 9, "Jane Doe, Metro News, Dallas.", 1)]
        cues = transcript_cues.extract_cues(turns)
        faces = [_face(0, [(20, 25)], name="Miriam Okafor", source="celebrity")]
        doc = persons.build_persons(_transcript(turns), faces, None, cues,
                                    None, {})
        # the conflicted speaker must NOT absorb the cleanly named face
        self.assertEqual(len(doc["persons"]), 2)
        self.assertEqual(sorted(p["name"] or "" for p in doc["persons"]),
                         ["", "Miriam Okafor"])

    def test_overlapping_speakers_never_merge_despite_same_name(self):
        # two voices claiming the same name while TALKING OVER EACH OTHER
        # are two people (or a bad resolution) — merge vetoed, and the
        # post-hoc collision sweep demotes both names
        turns = [_turn("SPEAKER_00", 0, 10, "I'm Miriam Okafor.", 0),
                 _turn("SPEAKER_01", 5, 15, "I'm Miriam Okafor.", 1),
                 _turn("SPEAKER_00", 20, 25, "More at eleven.", 2),
                 _turn("SPEAKER_01", 21, 26, "Indeed.", 3)]
        cues = transcript_cues.extract_cues(turns)
        doc = persons.build_persons(_transcript(turns), None, None, cues,
                                    None, {})
        self.assertEqual(len(doc["persons"]), 2)          # merge blocked
        self.assertTrue(all(p["name"] is None for p in doc["persons"]))
        self.assertTrue(all("Miriam Okafor" in p["aliases"]
                            for p in doc["persons"]))
        checks = [v["check"] for v in doc["vetoes"]]
        self.assertIn("speech_self_overlap", checks)
        self.assertIn("name_collision", checks)

    def test_nonoverlapping_same_name_speakers_still_merge(self):
        # split cluster (e.g. phone-filtered voice): same self-ID, no
        # simultaneous speech -> ONE person, no vetoes
        turns = [_turn("SPEAKER_00", 0, 5, "I'm Miriam Okafor.", 0),
                 _turn("SPEAKER_01", 10, 15, "I'm Miriam Okafor.", 1)]
        cues = transcript_cues.extract_cues(turns)
        doc = persons.build_persons(_transcript(turns), None, None, cues,
                                    None, {})
        self.assertEqual(len(doc["persons"]), 1)
        self.assertEqual(doc["persons"][0]["name"], "Miriam Okafor")
        self.assertEqual(doc["vetoes"], [])

    def test_heavily_overlapping_same_name_faces_do_not_merge(self):
        faces = [_face(0, [(0, 60)], name="Miriam Okafor", source="celebrity"),
                 _face(1, [(0, 60)], name="Miriam Okafor", source="celebrity")]
        doc = persons.build_persons(None, faces, None, None, None, {})
        self.assertEqual(len(doc["persons"]), 2)
        self.assertIn("face_self_overlap",
                      [v["check"] for v in doc["vetoes"]])
        self.assertTrue(all(p["name"] is None for p in doc["persons"]))

    def test_depicted_face_never_binds_to_speech_and_types_depicted_only(self):
        # the depicted-face guard (media_depictions join, 2026-09-06):
        # a face MAJORITY inside displayed-media shots is excluded from
        # co-occurrence (the photo/monitor trap) and typed depicted_only
        from lib import sources as src_mod
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 8, f"line {i}", i)
                 for i in range(6)]
        faces = [_face(0, [(i * 10, i * 10 + 8) for i in range(6)],
                       name="Miriam Okafor", source="celebrity")]
        docs = [{"shot_id": i,
                 "media_depictions": [{"medium": "screen", "detail": "x"}]}
                for i in range(6)]
        shots = {i: (i * 10.0, i * 10.0 + 8.0) for i in range(6)}
        spans = src_mod.parse_media_depiction_spans(docs, shots)
        src_mod.mark_depicted_faces(faces, spans)
        self.assertGreaterEqual(faces[0]["depicted_share"], 0.99)
        co = cooccur.bind_speakers_to_faces(turns, faces)
        self.assertEqual(co["bindings"], [])          # never proposed
        doc = persons.build_persons(_transcript(turns), faces, co,
                                    None, None, {})
        ptypes = {p["type"] for p in doc["persons"] if p["face_ids"]}
        self.assertEqual(ptypes, {"depicted_only"})

    def test_depicted_shot_credits_no_face_not_even_the_live_one(self):
        # the monitor-guest trap (review, 2026-09-06, CONFIRMED): the
        # depicted signal is SHOT-level, so during a depicted shot NO
        # face earns co-occurrence credit — excluding only the depicted
        # face whole-video handed the co-present live anchor undivided
        # credit and a false CONFIRMED binding for the guest's speech
        from lib import sources as src_mod
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 8, f"line {i}", i)
                 for i in range(6)]
        # the guest exists only on the studio wall; the anchor is live
        # in the same depicted shots
        guest = _face(0, [(i * 10, i * 10 + 8) for i in range(6)],
                      name="Miriam Okafor", source="celebrity")
        anchor = _face(1, [(i * 10, i * 10 + 8) for i in range(6)],
                       name="Beth Cole", source="celebrity")
        depicted = [(i * 10.0, i * 10.0 + 8.0) for i in range(6)]
        src_mod.mark_depicted_faces([guest, anchor], depicted)
        co = cooccur.bind_speakers_to_faces(turns, [guest, anchor])
        self.assertEqual(co["bindings"], [])   # nobody inherits credit

    def test_live_spans_still_earn_credit_outside_depicted_shots(self):
        from lib import sources as src_mod
        turns = [_turn("SPEAKER_00", i * 10, i * 10 + 8, f"line {i}", i)
                 for i in range(6)]
        face = _face(0, [(i * 10, i * 10 + 8) for i in range(6)],
                     name="Miriam Okafor", source="celebrity")
        # only the first shot is depicted; five live turns remain
        src_mod.mark_depicted_faces([face], [(0.0, 8.0)])
        co = cooccur.bind_speakers_to_faces(turns, [face])
        self.assertEqual(len(co["bindings"]), 1)
        self.assertEqual(co["bindings"][0]["status"], "proposed")
        # five of six turns still earn credit through the live spans
        self.assertGreaterEqual(co["bindings"][0]["share"], 0.5)

    def test_speaker_merged_with_fully_depicted_face_is_voice_only(self):
        # the phone-interview-over-photo case (review, 2026-09-06,
        # CONFIRMED): the same-name merge joins a self-ID'd speaker with
        # their fully-depicted face — the VOICE is real, the on-camera
        # presence is not, so the type is voice_only, never
        # on_camera_speaker
        from lib import sources as src_mod
        turns = [_turn("SPEAKER_00", 0, 5, "I'm Miriam Okafor.", 0)]
        cues = transcript_cues.extract_cues(turns)
        face = _face(0, [(10, 40)], name="Miriam Okafor", source="collection")
        src_mod.mark_depicted_faces([face], [(10.0, 40.0)])
        doc = persons.build_persons(_transcript(turns), [face], None,
                                    cues, None, {})
        self.assertEqual(len(doc["persons"]), 1)
        p = doc["persons"][0]
        self.assertEqual(p["name"], "Miriam Okafor")
        self.assertEqual(p["type"], "voice_only")

    def test_attested_none_and_absent_media_depictions_mark_nothing(self):
        from lib import sources as src_mod
        docs = [{"shot_id": 0, "media_depictions": []},   # attested none
                {"shot_id": 1}]                            # not assessed
        shots = {0: (0.0, 10.0), 1: (10.0, 20.0)}
        self.assertEqual(
            src_mod.parse_media_depiction_spans(docs, shots), [])
        faces = [_face(0, [(0, 20)], name="Miriam Okafor", source="celebrity")]
        src_mod.mark_depicted_faces(faces, [])
        self.assertNotIn("depicted_share", faces[0])
        doc = persons.build_persons(None, faces, None, None, None, {})
        self.assertEqual(doc["persons"][0]["type"], "silent_face")

    def test_minority_depicted_face_still_binds(self):
        # below the majority threshold the face is live more than
        # depicted — binding on the live majority is the design
        from lib import sources as src_mod
        faces = [_face(0, [(0, 80), (80, 100)],
                       name="Miriam Okafor", source="celebrity")]
        src_mod.mark_depicted_faces(faces, [(80.0, 100.0)])
        self.assertLess(faces[0]["depicted_share"], 0.5)
        doc = persons.build_persons(None, faces, None, None, None, {})
        self.assertEqual(doc["persons"][0]["type"], "silent_face")

    def test_tracker_duplicates_coalesce_at_parse_with_audit(self):
        # the tracker-duplicate class (2026-09-06: two parallel person
        # entries, 99% identical spans, SAME collection face UUID — the
        # gate's one false veto): coalesced at the PRODUCER boundary so
        # no consumer ever sees the duplicate, with an audit trail
        from lib import sources as src_mod
        doc = {"people": [
            {"person_id": 3, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-1",
             "total_screen_time_sec": 60.0,
             "appearances": [{"begin_ms": 0, "end_ms": 60000}]},
            {"person_id": 4, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-1",
             "total_screen_time_sec": 58.0,
             "appearances": [{"begin_ms": 0, "end_ms": 58000}]},
        ]}
        faces = src_mod.parse_faces(doc)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["coalesced_ids"], [4])
        # screen_sec is single-coverage of the UNION, never a sum
        self.assertAlmostEqual(faces[0]["screen_sec"], 60.0)
        built = persons.build_persons(None, faces, None, None, None, {})
        self.assertEqual(len(built["persons"]), 1)
        self.assertEqual(built["persons"][0]["name"], "Miriam Okafor")
        self.assertEqual(built["persons"][0]["face_ids"], [3, 4])
        self.assertEqual(built["vetoes"], [])
        self.assertIn("recognizer_coalesce",
                      [e.get("mechanism")
                       for e in built["persons"][0]["evidence"]])

    def test_celebrity_id_pairs_coalesce_like_face_uuids(self):
        # the celebrity analogue (2026-09-07): face-recognition now
        # emits the celebrity id on celebrity-named entries — the
        # residual same-confidence pair the UUID class could not see
        from lib import sources as src_mod
        doc = {"people": [
            {"person_id": 27, "named": True, "name": "Miriam Okafor",
             "name_source": "celebrity", "face_id": None,
             "celebrity_id": "1AbC2dE", "total_screen_time_sec": 15.2,
             "appearances": [{"begin_ms": 0, "end_ms": 15200}]},
            {"person_id": 35, "named": True, "name": "Miriam Okafor",
             "name_source": "celebrity", "face_id": None,
             "celebrity_id": "1AbC2dE", "total_screen_time_sec": 16.0,
             "appearances": [{"begin_ms": 20000, "end_ms": 36000}]},
        ]}
        faces = src_mod.parse_faces(doc)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["coalesced_ids"], [35])
        self.assertEqual(faces[0]["recognizer_ids"],
                         {"celebrity_id": "1AbC2dE"})
        # disjoint spans: single-coverage union, not a sum quirk
        self.assertAlmostEqual(faces[0]["screen_sec"], 31.2)

    def test_both_ids_entries_coalesce_on_the_shared_celebrity_id(self):
        # THE default-policy shape (review, 2026-09-07 CONFIRMED miss):
        # auto-indexing gives each celebrity-named cluster a DISTINCT
        # fresh face UUID while both carry the SAME celebrity id — the
        # or-precedence single key hid the shared assertion behind the
        # UUIDs and the flagship join never fired
        from lib import sources as src_mod
        doc = {"people": [
            {"person_id": 3, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-A",
             "celebrity_id": "1AbC2dE", "total_screen_time_sec": 20.0,
             "appearances": [{"begin_ms": 0, "end_ms": 20000}]},
            {"person_id": 4, "named": True, "name": "Miriam Okafor",
             "name_source": "celebrity", "face_id": "uuid-B",
             "celebrity_id": "1AbC2dE", "total_screen_time_sec": 15.0,
             "appearances": [{"begin_ms": 30000, "end_ms": 45000}]},
        ]}
        faces = src_mod.parse_faces(doc)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["coalesced_ids"], [4])
        self.assertEqual(faces[0]["recognizer_ids"],
                         {"face_id": "uuid-A", "celebrity_id": "1AbC2dE"})

    def test_collection_contradiction_blocks_the_celebrity_weld(self):
        # relative-FP rule (user, 2026-09-07): celebrity detection has a
        # higher FP rate than SearchFaces against the curated
        # collection, so when the collection asserts TWO people
        # (distinct UUIDs under distinct collection names) a shared
        # celebrity id must not weld them
        from lib import sources as src_mod
        doc = {"people": [
            {"person_id": 1, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-A",
             "celebrity_id": "1AbC2dE", "total_screen_time_sec": 20.0,
             "appearances": [{"begin_ms": 0, "end_ms": 20000}]},
            {"person_id": 2, "named": True, "name": "Jane Smith",
             "name_source": "collection", "face_id": "uuid-B",
             "celebrity_id": "1AbC2dE", "total_screen_time_sec": 15.0,
             "appearances": [{"begin_ms": 30000, "end_ms": 45000}]},
        ]}
        faces = src_mod.parse_faces(doc)
        self.assertEqual(len(faces), 2)
        self.assertEqual([f["name"] for f in faces],
                         ["Miriam Okafor", "Jane Smith"])

    def test_merge_adopts_the_collection_name_over_the_celebrity_name(self):
        # same shared-UUID weld, keeper named by celebrity detection,
        # merged entry named by the collection: the lower-FP source's
        # name wins regardless of arrival order
        from lib import sources as src_mod
        doc = {"people": [
            {"person_id": 1, "named": True, "name": "Anne Reed",
             "name_source": "celebrity", "face_id": "uuid-A",
             "total_screen_time_sec": 20.0,
             "appearances": [{"begin_ms": 0, "end_ms": 20000}]},
            {"person_id": 2, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-A",
             "total_screen_time_sec": 15.0,
             "appearances": [{"begin_ms": 30000, "end_ms": 45000}]},
        ]}
        faces = src_mod.parse_faces(doc)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["name"], "Miriam Okafor")
        self.assertEqual(faces[0]["name_source"], "collection")

    def test_bridging_entry_welds_two_prior_faces(self):
        # an entry carrying keys of TWO previously-separate faces is the
        # recognizer asserting all three are one person
        from lib import sources as src_mod
        doc = {"people": [
            {"person_id": 1, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-A",
             "total_screen_time_sec": 10.0,
             "appearances": [{"begin_ms": 0, "end_ms": 10000}]},
            {"person_id": 2, "named": True, "name": "Miriam Okafor",
             "name_source": "celebrity", "celebrity_id": "1AbC2dE",
             "total_screen_time_sec": 10.0,
             "appearances": [{"begin_ms": 20000, "end_ms": 30000}]},
            {"person_id": 3, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-A",
             "celebrity_id": "1AbC2dE", "total_screen_time_sec": 10.0,
             "appearances": [{"begin_ms": 40000, "end_ms": 50000}]},
        ]}
        faces = src_mod.parse_faces(doc)
        self.assertEqual(len(faces), 1)
        self.assertEqual(sorted(faces[0]["coalesced_ids"]), [2, 3])
        self.assertAlmostEqual(faces[0]["screen_sec"], 30.0)

    def test_absent_assertions_never_coalesce(self):
        # two unnamed people with face_id None and no celebrity_id must
        # stay two people — None is not an identity claim
        from lib import sources as src_mod
        doc = {"people": [
            {"person_id": 1, "named": False, "name": "Person_1",
             "face_id": None, "total_screen_time_sec": 5.0,
             "appearances": [{"begin_ms": 0, "end_ms": 5000}]},
            {"person_id": 2, "named": False, "name": "Person_2",
             "face_id": None, "total_screen_time_sec": 5.0,
             "appearances": [{"begin_ms": 0, "end_ms": 5000}]},
        ]}
        self.assertEqual(len(src_mod.parse_faces(doc)), 2)

    def test_distinct_recognizer_face_ids_still_veto(self):
        # two DIFFERENT collection identities sharing a name and heavy
        # co-presence remain two people — coalescing keys on the shared
        # assertion, never the mechanism
        from lib import sources as src_mod
        doc = {"people": [
            {"person_id": 0, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-1",
             "total_screen_time_sec": 60.0,
             "appearances": [{"begin_ms": 0, "end_ms": 60000}]},
            {"person_id": 1, "named": True, "name": "Miriam Okafor",
             "name_source": "collection", "face_id": "uuid-2",
             "total_screen_time_sec": 60.0,
             "appearances": [{"begin_ms": 0, "end_ms": 60000}]},
        ]}
        faces = src_mod.parse_faces(doc)
        self.assertEqual(len(faces), 2)
        built = persons.build_persons(None, faces, None, None, None, {})
        self.assertEqual(len(built["persons"]), 2)
        self.assertIn("face_self_overlap",
                      [v["check"] for v in built["vetoes"]])

    def test_briefly_overlapping_same_name_faces_merge(self):
        # live person + a 3s photo of them: under the face tolerance
        faces = [_face(0, [(0, 30)], name="Miriam Okafor", source="celebrity"),
                 _face(1, [(28, 31), (40, 50)], name="Miriam Okafor",
                       source="celebrity")]
        doc = persons.build_persons(None, faces, None, None, None, {})
        self.assertEqual(len(doc["persons"]), 1)
        self.assertEqual(doc["persons"][0]["name"], "Miriam Okafor")

    def test_remap_tables(self):
        turns = [_turn("SPEAKER_00", 0, 5, "I'm Miriam Okafor.", 0),
                 _turn("SPEAKER_00", 6, 9, "Thanks for watching.", 1)]
        cues = transcript_cues.extract_cues(turns)
        doc = persons.build_persons(_transcript(turns), None, None, cues,
                                    None, {})
        smap, fmap = persons.remap_tables(doc)
        self.assertEqual(smap["SPEAKER_00"]["name"], "Miriam Okafor")
        self.assertEqual(fmap, {})




# ─────────────────────────────────────────────────────────────────────────────
# workflow layer — router, validation, gateway, ASL parity
# ─────────────────────────────────────────────────────────────────────────────

import json  # noqa: E402
from unittest import mock  # noqa: E402


class _Ctx:
    class client_context:                            # noqa: N801 — stub shape
        custom = {}

    def __init__(self, tool):
        self.client_context = type(
            "CC", (), {"custom": {"bedrockAgentCoreToolName": tool}})()


class TestRouter(unittest.TestCase):
    def test_unknown_operation_raises(self):
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "cast_router", os.path.join(ROOT, "lambda_function.py"))
        router = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(router)
        with self.assertRaises(ValueError):
            router.lambda_handler({"operation": "MakeCoffee"}, None)

    def test_all_operations_are_routed(self):
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "cast_router", os.path.join(ROOT, "lambda_function.py"))
        router = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(router)
        current = {"AdjudicateCast", "ApplyEnrichment", "BindPersons",
                   "ReadOnscreenText", "ResolveAddressees",
                   "ValidateSources"}
        legacy = {"ValidateAndPlan", "ScheduleReadsAndWait",
                  "BindAndFinalize"}   # back-compat aliases, remove at GA
        self.assertEqual(set(router._OPERATIONS), current | legacy)


class TestValidateSources(unittest.TestCase):
    def _run(self, inp, exists=True):
        from states.validate_sources import lambda_function as vp
        with mock.patch.object(vp.s3io, "exists", return_value=exists), \
             mock.patch.object(vp, "PROMPT_REGISTRY_ARN", "arn:pr"), \
             mock.patch.object(vp.pb_common, "invoke_gateway_tool",
                               return_value={"system": "SYS"}), \
             mock.patch.object(vp.boto3, "client",
                               return_value=object()), \
             mock.patch.object(vp.s3io, "put_json") as put:
            out = vp.lambda_handler({"uuid": "wf-1", "input": inp}, None)
        return out, put

    def test_prompts_bind_when_enrichment_can_run(self):
        from states.validate_sources import lambda_function as vp
        from unittest import mock as m
        with m.patch.object(vp.s3io, "exists", return_value=True), \
             m.patch.object(vp, "PROMPT_REGISTRY_ARN", "arn:pr"), \
             m.patch.dict(vp.os.environ,
                          {"BEDROCK_GATEWAY_ARN": "arn:b"}), \
             m.patch.object(vp.pb_common, "invoke_gateway_tool",
                            return_value={"system": "SYS"}), \
             m.patch.object(vp.boto3, "client", return_value=object()), \
             m.patch.object(vp.s3io, "put_json") as put:
            vp.lambda_handler({"uuid": "wf-1", "input": {
                "speech_diarization_s3uri":
                    "s3://test-proxy/micro-workflow/speech-diarization/x"
            }}, None)
        plan = put.call_args[0][1]
        self.assertEqual(plan["prompts"]["cast-adjudicate"], "SYS")

    def test_bind_failure_degrades_to_no_prompts(self):
        # enhancement, never a dependency: an unseeded/rolled-back
        # registry must not fail the run — downstream states skip
        from states.validate_sources import lambda_function as vp
        from unittest import mock as m
        with m.patch.object(vp.s3io, "exists", return_value=True), \
             m.patch.object(vp, "PROMPT_REGISTRY_ARN", "arn:pr"), \
             m.patch.dict(vp.os.environ,
                          {"BEDROCK_GATEWAY_ARN": "arn:b"}), \
             m.patch.object(vp.pb_common, "invoke_gateway_tool",
                            return_value={"error": True,
                                          "errorCode": "PROMPT_NOT_FOUND",
                                          "message": "x"}), \
             m.patch.object(vp.boto3, "client", return_value=object()), \
             m.patch.object(vp.s3io, "put_json") as put:
            out = vp.lambda_handler({"uuid": "wf-1", "input": {
                "speech_diarization_s3uri":
                    "s3://test-proxy/micro-workflow/speech-diarization/x"
            }}, None)
        self.assertIn("plan_s3uri", out)
        self.assertNotIn("prompts", put.call_args[0][1])

    def test_no_identity_source_refuses(self):
        with self.assertRaises(ValueError) as cm:
            self._run({"shot_segmentation_s3uri":
                       "s3://test-proxy/micro-workflow/shot-segmentation/x"})
        self.assertIn("identity source", str(cm.exception))

    def test_foreign_bucket_refuses(self):
        with self.assertRaises(ValueError) as cm:
            self._run({"speech_diarization_s3uri":
                       "s3://other-bucket/micro-workflow/speech-diarization/x"})
        self.assertIn("proxy bucket", str(cm.exception))

    def test_missing_artifact_names_the_uri(self):
        with self.assertRaises(ValueError) as cm:
            self._run({"speech_diarization_s3uri":
                       "s3://test-proxy/micro-workflow/speech-diarization/x"},
                      exists=False)
        self.assertIn("final_transcript.json", str(cm.exception))

    def test_plan_pins_sibling_workflow_ids(self):
        out, put = self._run(
            {"speech_diarization_s3uri":
             "s3://test-proxy/micro-workflow/speech-diarization/dia-123",
             "face_recognition_s3uri":
             "s3://test-proxy/micro-workflow/face-recognition/face-456"})
        self.assertEqual(out["uuid"], "wf-1")
        plan = put.call_args[0][1]
        self.assertEqual(plan["sibling_workflow_ids"],
                         {"speech_diarization": "dia-123",
                          "face_recognition": "face-456"})
        self.assertEqual(plan["sources_present"],
                         ["face_recognition", "speech_diarization"])


class TestGateway(unittest.TestCase):
    def _call(self, tool, event):
        import lambda_function as gw
        return gw.lambda_handler(event, _Ctx(f"cast-resolution___{tool}"))

    def test_no_identity_source_is_missing_parameter(self):
        out = self._call("resolve_cast", {
            "shot_segmentation_s3uri": "s3://test-proxy/micro-workflow/x/y"})
        self.assertEqual(out["errorCode"], "MISSING_PARAMETER")

    def test_uuid_parameter_is_rejected(self):
        out = self._call("resolve_cast", {
            "uuid": "x",
            "speech_diarization_s3uri": "s3://test-proxy/micro-workflow/d/1"})
        self.assertEqual(out["errorCode"], "INVALID_PARAMETER")

    def test_non_s3_uri_is_invalid(self):
        out = self._call("resolve_cast", {
            "speech_diarization_s3uri": "/tmp/not-s3"})
        self.assertEqual(out["errorCode"], "INVALID_PARAMETER")

    def test_contract_example_pins_the_consumer_fields(self):
        # audio-description consumes these fields (its lock-step test
        # skips on single-repo checkouts, so the ALWAYS-RUN pin lives
        # here, on the producer): a rename must fail THIS build
        import json, os
        path = os.path.join(os.path.dirname(__file__), "..",
                            "gateway_tool", "output_contract.json")
        contract = _load_json(path)
        example = contract["outputs"]["persons"]["example"]
        wids = example["provenance"]["sibling_workflow_ids"]
        for key in ("face_recognition", "speech_diarization"):
            self.assertIn(key, wids)
        persons = example["persons"]
        for field in ("person_id", "name", "type", "role", "aliases",
                      "speaker_ids", "face_ids", "screen_sec",
                      "speech_sec", "appearances", "appearances_basis",
                      "name_mechanisms"):
            self.assertTrue(any(field in p for p in persons),
                            f"persons example lost {field!r}")
        self.assertIn("blocks", contract["outputs"]["credits"]["example"])

    def test_describe_tool_carries_contract(self):
        out = self._call("describe_tool", {})
        self.assertIn("output_contract", out)
        self.assertEqual(
            sorted(out["output_contract"]["outputs"]),
            ["credits", "face-map", "persons", "speaker-map", "usage"])

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        path = os.path.join(ROOT, "gateway_tool", "input_examples.json")
        examples = _load_json(path)["tools"]
        self.assertEqual(sorted(examples),
                         ["describe_tool", "get_job_status", "resolve_cast"])
        for tool, entry in examples.items():
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        out = self._call("describe_tool", {})
        self.assertIn("resolve_cast", out["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        import lambda_function as gw
        path = os.path.join(ROOT, "gateway_tool", "input_examples.json")
        example = _load_json(path)["tools"]["resolve_cast"]["example"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gw, "_sfn", fake):
            out = self._call("resolve_cast", dict(example))
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()

    def test_identity_is_stable_and_token_sensitive(self):
        import lambda_function as gw
        req = {"speech_diarization_s3uri":
               "s3://test-proxy/micro-workflow/speech-diarization/dia-123"}
        a = gw.compute_workflow_id(req, tool_version="test")
        b = gw.compute_workflow_id(dict(req), tool_version="test")
        c = gw.compute_workflow_id(req, tool_version="test",
                                   extra_material=("tok",))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


@unittest.skipIf(yaml is None,
                 "PyYAML not installed — pip install pyyaml for the ASL gate")
class TestAslParity(unittest.TestCase):
    def test_inline_definition_matches_definitions_json(self):
        class L(yaml.SafeLoader):
            pass

        def unknown(loader, suffix, node):
            if isinstance(node, yaml.ScalarNode):
                return loader.construct_scalar(node)
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            return loader.construct_mapping(node)
        L.add_multi_constructor("!", unknown)
        tpl = yaml.load(_read(os.path.join(ROOT, "cfnstack.yaml")), Loader=L)  # nosec B506
        inline = json.loads(
            tpl["Resources"]["CastResolutionStateMachine"]
            ["Properties"]["DefinitionString"])
        copy = _load_json(os.path.join(ROOT, "definitions.json"))
        self.assertEqual(inline, copy)

    def test_asl_operations_exist_in_router(self):
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "cast_router", os.path.join(ROOT, "lambda_function.py"))
        router = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(router)
        copy = _load_json(os.path.join(ROOT, "definitions.json"))
        ops = {s["Parameters"]["Payload"]["operation"]
               for s in copy["States"].values() if s["Type"] == "Task"}
        # router may carry back-compat aliases beyond the live ASL
        self.assertTrue(ops <= set(router._OPERATIONS),
                        ops - set(router._OPERATIONS))


# ─────────────────────────────────────────────────────────────────────────────
# P1 focus reads — scheduler and product consumption
# ─────────────────────────────────────────────────────────────────────────────

from lib import reads  # noqa: E402
from lib import sources  # noqa: E402


class TestReadScheduler(unittest.TestCase):
    def test_intro_cue_schedules_a_chyron_span(self):
        out = reads.plan_reads(
            cues=[{"kind": "introduction", "name": "Matt Barrow",
                   "speaker_id": "SPEAKER_00", "t": 120.0,
                   "sentence_id": 5, "confidence": "high"}],
            faces=[], text_shots=[], duration_sec=3600)
        self.assertEqual(len(out["spans"]), 1)
        self.assertEqual(out["spans"][0]["reason"], "introduction")
        self.assertEqual(out["requests"][0]["category"], "title-read")

    def test_unnamed_face_debut_schedules_named_face_does_not(self):
        faces = [_face(0, [(50, 70), (200, 230)]),
                 _face(1, [(90, 130)], name="Miriam Okafor", source="celebrity")]
        out = reads.plan_reads([], faces, [], 3600)
        self.assertEqual([s["reason"] for s in out["spans"]],
                         ["face_debut"])
        self.assertEqual(out["spans"][0]["start"], 50)

    def test_oneoff_broll_face_never_triggers(self):
        out = reads.plan_reads([], [_face(0, [(50, 52)])], [], 3600)
        self.assertEqual(out["spans"], [])

    def test_tail_text_alone_never_buys_credits(self):
        # spend audit 2026-09-30: text-in-the-tail is what bought
        # 24-30 min of channel logos on every news asset — credits now
        # need SME-typed credit_roll evidence or a caller hint
        out = reads.plan_reads([], [], [(87, 3500, 3520)], 3600)
        self.assertFalse(out["credits"])
        self.assertEqual(out["spans"], [])
        self.assertIsNone(out["requests"])

    def test_credit_roll_shots_are_the_read_spans(self):
        # the roll is read exactly (padded, merged); interleaved
        # textless black between roll shots is never paid for
        out = reads.plan_reads([], [], [], 3600,
                               credit_roll_spans=[(3400, 3420),
                                                  (3424, 3440),
                                                  (3520, 3560)])
        self.assertTrue(out["credits"])
        (req,) = out["requests"]
        self.assertEqual(req["category"], "credits-read")
        bounds = [(s["start"], s["end"]) for s in req["spans"]]
        self.assertEqual(bounds, [(3395.0, 3445.0), (3515.0, 3565.0)])
        self.assertEqual(
            [(c["t0"], c["t1"]) for c in out["ledger_candidates"]],
            bounds)

    def test_credits_hint_anchors_at_last_speech(self):
        # hinted with no typed roll: CREDITS_PROBE_SEC anchored at the
        # last dialogue's end (rolls start at/after the last line;
        # appended textless tails carry no dialogue)
        out = reads.plan_reads([], [], [], 7200, credits_hint=True,
                               last_speech_end=5400.0)
        (req,) = out["requests"]
        (span,) = req["spans"]
        self.assertEqual((span["start"], span["end"]), (5370.0, 5610.0))
        # no transcript: the final window of the tail
        out = reads.plan_reads([], [], [], 3600, credits_hint=True)
        (span,) = out["requests"][0]["spans"]
        self.assertEqual((span["start"], span["end"]), (3360.0, 3600.0))

    def test_need_windows_gate_debuts_and_text_shots(self):
        faces = [_face(0, [(50, 70), (200, 230)])]
        # need assessed, and it does NOT cover the debut at 50s
        out = reads.plan_reads(
            [], faces, [(3, 500, 505)], 3600,
            need_windows=[{"head": "voice:S9",
                           "spans": [(1000.0, 1100.0)]}])
        self.assertEqual(out["spans"], [])
        self.assertEqual(out["skipped"]["no_need"], 2)
        # need covers the debut: the read is back
        out = reads.plan_reads(
            [], faces, [], 3600,
            need_windows=[{"head": "voice:S9",
                           "spans": [(45.0, 60.0)]}])
        self.assertEqual([s["reason"] for s in out["spans"]],
                         ["face_debut"])

    def test_reads_per_head_budget(self):
        # one ever-present unnamed head must not legitimize every span
        # (2026-10-05: news title spend was cap-bound, not need-bound):
        # READS_PER_HEAD moments, then head_quota skips
        shots = [(i, 100.0 * i, 100.0 * i + 5) for i in range(1, 11)]
        out = reads.plan_reads(
            [], [], shots, 3600,
            need_windows=[{"head": "voice:S0",
                           "spans": [(0.0, 3600.0)]}])
        self.assertEqual(len(out["spans"]), pb_common.READS_PER_HEAD)
        self.assertEqual(out["skipped"]["head_quota"],
                         10 - pb_common.READS_PER_HEAD)
        # two heads with disjoint presence each carry their own budget
        out = reads.plan_reads(
            [], [], shots, 3600,
            need_windows=[{"head": "voice:S0",
                           "spans": [(0.0, 450.0)]},
                          {"head": "voice:S1",
                           "spans": [(450.0, 3600.0)]}])
        self.assertEqual(len(out["spans"]),
                         2 * pb_common.READS_PER_HEAD)
        # merged candidates widen an existing moment, charging nothing:
        # ten overlapping shots collapse into one span, quota intact
        shots = [(i, 100.0 + i, 108.0 + i) for i in range(10)]
        out = reads.plan_reads(
            [], [], shots, 3600,
            need_windows=[{"head": "voice:S0",
                           "spans": [(0.0, 3600.0)]}])
        self.assertEqual(len(out["spans"]), 1)
        self.assertEqual(out["skipped"]["head_quota"], 0)

    def test_introduction_cue_bypasses_the_need_gate(self):
        out = reads.plan_reads(
            [{"kind": "introduction", "name": "X", "speaker_id": "S",
              "t": 100.0, "sentence_id": 1, "confidence": "high"}],
            [], [], 3600, need_windows=[])
        self.assertEqual([s["reason"] for s in out["spans"]],
                         ["introduction"])

    def test_need_none_keeps_legacy_triggers(self):
        out = reads.plan_reads([], [_face(0, [(50, 70), (200, 230)])],
                               [], 3600, need_windows=None)
        self.assertEqual([s["reason"] for s in out["spans"]],
                         ["face_debut"])

    def test_bound_speakers_subtract_from_need(self):
        # the conditional witness (2026-09-30): a major voice the
        # class-1 forecast matched to a NAMED face creates no need —
        # an asset whose majors all resolve for free plans nothing
        tr = {"speakers": {"S1": 900.0, "S2": 800.0},
              "turns": [{"speaker_id": "S1", "start": 5, "end": 8},
                        {"speaker_id": "S2", "start": 50, "end": 58}]}
        wins = reads.need_windows_from_sources(None, tr,
                                               bound_speakers={"S1"})
        self.assertEqual([w["head"] for w in wins], ["voice:S2"])
        a, b = wins[0]["spans"][0]
        self.assertTrue(a < 50 < b)          # only S2's turns remain
        self.assertEqual(
            reads.need_windows_from_sources(None, tr,
                                            bound_speakers={"S1", "S2"}),
            [])

    def test_need_windows_from_sources(self):
        # unnamed major face creates need; named major face does not;
        # minor face does not
        faces = [_face(0, [(10, 400)]),
                 _face(1, [(500, 900)], name="N", source="celebrity"),
                 _face(2, [(950, 960)])]
        wins = reads.need_windows_from_sources(faces, None)
        self.assertEqual([w["head"] for w in wins], ["face:0"])
        a, b = wins[0]["spans"][0]
        self.assertLess(a, 10)
        self.assertGreater(b, 400)
        # major voice creates need; minor voice does not
        tr = {"speakers": {"S1": 900.0, "S2": 10.0},
              "turns": [{"speaker_id": "S1", "start": 5, "end": 8},
                        {"speaker_id": "S2", "start": 50, "end": 51}]}
        wins = reads.need_windows_from_sources(None, tr)
        self.assertEqual([w["head"] for w in wins], ["voice:S1"])
        # neither source -> not assessable
        self.assertIsNone(reads.need_windows_from_sources(None, None))

    def test_name_match_token_rules(self):
        # whole tokens, contiguous runs, symmetric containment —
        # never characters inside a token, never scattered tokens
        for a, b, want in (("Joanne", "JOANNE", True),
                           ("Joanne", "JOANNE SMITH", True),
                           ("Rowan Kemble", "ROWAN", True),
                           ("Rowan Kemble", "ROWAN KEMBLE \u2022 HOST", True),
                           ("Anne Marie", "ANNE-MARIE", True),
                           ("Anne", "JOANNE", False),
                           ("Al", "TOTAL", False),
                           ("Jane Doe", "JANE AND JOHN DOE", False),
                           ("Rowan Kemble", "R. KEMBLE", False),
                           ("Joanne", "\u00a32,500", False)):
            self.assertEqual(pb_common.name_match(a, b), want,
                             (a, b, want))

    def test_screen_text_cues_corroborate_only(self):
        shots = {1: (10.0, 15.0), 2: (20.0, 25.0), 3: (30.0, 35.0)}
        docs = [{"shot_id": 1, "texts_on_screen": ["JOANNE", "\u00a3250"]},
                {"shot_id": 2, "texts_on_screen": ["BANK", "2:30"]},
                {"shot_id": 3, "texts_on_screen": ["JOANNE SMITH"]}]
        st = sources.parse_screen_texts(docs, shots)
        self.assertEqual([s[0] for s in st], [1, 2, 3])
        # no proposal, no cue — junk and even real names stay inert
        self.assertEqual(sources.screen_text_cues(st, set()), [])
        cues = sources.screen_text_cues(st, {"Joanne"})
        self.assertEqual([(c["shot_id"], c["name"]) for c in cues],
                         [(1, "Joanne"), (3, "Joanne")])
        self.assertTrue(all(c["kind"] == "overlay_name" for c in cues))
        # a shot already covered by an overlay cue of the same name
        # (SME typed it / witness read it) never votes twice
        cues = sources.screen_text_cues(
            st, {"Joanne"},
            existing_overlay=[{"kind": "overlay_name", "name": "JOANNE",
                               "t0": 10.0, "t1": 15.0}])
        self.assertEqual([c["shot_id"] for c in cues], [3])

    def test_screen_text_confirmed_speakers(self):
        tx = {"turns": [{"speaker_id": "S1", "start": 11, "end": 14},
                        {"speaker_id": "S1", "start": 31, "end": 34},
                        {"speaker_id": "S2", "start": 21, "end": 24}],
              "speakers": {"S1": 6.0, "S2": 3.0}}
        cues = [{"kind": "vocative_resolved", "name": "Joanne",
                 "speaker_id": "S1", "confidence": "high"},
                {"kind": "vocative_resolved", "name": "Pat",
                 "speaker_id": "S2", "confidence": "high"}]
        st = [(1, 10.0, 15.0, ["JOANNE"]), (2, 20.0, 25.0, ["PAT"]),
              (3, 30.0, 35.0, ["JOANNE SMITH"])]
        got = reads.screen_text_confirmed_speakers(tx, cues, st)
        # S1: two distinct corroborating shots; S2: only one -> stays
        self.assertEqual(got, {"S1"})
        # conflicting proposals are never subtracted
        cues.append({"kind": "self_id", "name": "Josephine",
                     "speaker_id": "S1", "confidence": "high"})
        self.assertEqual(
            reads.screen_text_confirmed_speakers(tx, cues, st), set())

    def test_head_text_shot_is_a_read_span_not_credits(self):
        out = reads.plan_reads([], [], [(3, 40, 45)], 3600)
        self.assertFalse(out["credits"])
        self.assertEqual(out["spans"][0]["reason"], "text_shot:3")

    def test_satisfied_spans_are_skipped(self):
        out = reads.plan_reads(
            [{"kind": "introduction", "name": "X", "speaker_id": "S",
              "t": 100.0, "sentence_id": 1, "confidence": "high"}],
            [], [], 3600, satisfied_spans=[(95, 115)])
        self.assertEqual(out["spans"], [])
        self.assertEqual(out["skipped"]["satisfied"], 1)
        self.assertIsNone(out["requests"])

    def test_ledger_blocks_repayment(self):
        out = reads.plan_reads(
            [], [], [(3, 40, 45)], 3600,
            ledger=[(35, 50, "title-read", None, reads.PRODUCT_V)])
        self.assertEqual(out["spans"], [])
        self.assertEqual(out["skipped"]["ledgered"], 1)

    def test_pre_v2_ledger_entry_is_reread_once(self):
        # entries recorded before the typed-extraction products lack
        # product_v: their evidence was never obtained in the current
        # shape, so they must NOT satisfy — one re-read converges
        out = reads.plan_reads(
            [], [], [(3, 40, 45)], 3600,
            ledger=[(35, 50, "title-read")])
        self.assertEqual(out["skipped"]["ledgered"], 0)
        self.assertEqual(len(out["spans"]), 1)

    def test_budget_caps_and_counts(self):
        shots = [(i, i * 100.0, i * 100.0 + 5) for i in range(1, 60)]
        out = reads.plan_reads([], [], shots, 20000)
        from lib.pb_common import MAX_READ_SPANS
        self.assertEqual(len(out["spans"]), MAX_READ_SPANS)
        self.assertGreater(out["skipped"]["budget"], 0)


class TestReadConsumption(unittest.TestCase):
    """cues_from_titles over the witness's typed `names` (product v2,
    2026-09-09): the read model types identities in any language; this
    side keeps only the cross-frame statistics."""

    @staticmethod
    def _card(t, names, lines=("x",)):
        return {"t": t, "lines": list(lines), "names": names}

    def test_typed_person_on_two_frames_becomes_a_cue(self):
        n = {"name": "Matt Barrow", "type": "person",
             "descriptor": "Metro News Correspondent"}
        doc = {"blocks": [{"start": 100.0, "cards": [
            self._card(100.0, [n]), self._card(100.5, [dict(n)]),
            self._card(200.0, [{"name": "Metro News", "type": "program",
                                "descriptor": None}]),
            self._card(200.5, [{"name": "Metro News", "type": "program",
                                "descriptor": None}])]}]}
        cues = reads.cues_from_titles(doc)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["name"], "Matt Barrow")
        self.assertEqual(cues[0]["role_text"], "Metro News Correspondent")
        self.assertEqual(cues[0]["kind"], "overlay_name")

    def test_single_frame_name_is_ocr_noise(self):
        doc = {"blocks": [{"start": 10.0, "cards": [
            self._card(10.0, [{"name": "Mar Ina", "type": "person",
                               "descriptor": None}]),
            self._card(10.5, [{"name": "Mari Na", "type": "person",
                               "descriptor": None}])]}]}
        self.assertEqual(reads.cues_from_titles(doc), [])

    def test_org_and_program_names_never_cue(self):
        n = {"name": "Harborlight Films", "type": "organization",
             "descriptor": None}
        doc = {"blocks": [{"start": 1.0, "cards": [
            self._card(1.0, [n]), self._card(1.5, [dict(n)])]}]}
        self.assertEqual(reads.cues_from_titles(doc), [])

    def test_untyped_cards_contribute_nothing(self):
        # pre-v2 product: lines only — no consumer-side extraction; the
        # ledger versioning re-reads the span instead
        doc = {"blocks": [{"start": 100.0, "cards": [
            {"t": 100.0, "lines": ["Matt Barrow", "Correspondent"]},
            {"t": 100.5, "lines": ["Matt Barrow", "Correspondent"]}]}]}
        self.assertEqual(reads.cues_from_titles(doc), [])

    def test_apply_cast_adds_character_name_never_renames(self):
        doc = {"persons": [
            {"person_id": 0, "name": "Victor Hale", "evidence": []},
            {"person_id": 1, "name": "Someone Else", "evidence": []},
            {"person_id": 2, "name": None, "evidence": []}]}
        cast = {"pairs": [{"actor": "victor hale",
                           "character": "Dan Reyes"}]}
        out = reads.apply_cast(doc, cast)
        p = out["persons"][0]
        self.assertEqual(p["name"], "Victor Hale")   # never renamed
        self.assertEqual(p["character_name"], "Dan Reyes")
        self.assertIn("credits_cast",
                      [e["mechanism"] for e in p["evidence"]])
        self.assertNotIn("character_name", out["persons"][1])


class TestCreditColumns(unittest.TestCase):
    """Column layouts arrive AS typed pairings since product v2 — the
    witness sees the columns and states the kind; no typography
    heuristics on this side."""

    @staticmethod
    def _doc(pairings):
        return {"blocks": [{"cards": [
            {"t": 2982.0, "lines": ["x"], "pairings": pairings}]}]}

    def test_two_column_cast_form(self):
        cast = reads.cast_from_credits(self._doc([
            {"name": "LENA VOSS", "paired_with": "Mira Santos",
             "header": "Co-Starring", "kind": "character"},
            {"name": "LINK RUIZ", "paired_with": "C.O. Luis Ochoa",
             "header": "Co-Starring", "kind": "character"}]))
        self.assertIn({"actor": "LENA VOSS",
                       "character": "Mira Santos"}, cast["pairs"])
        self.assertIn({"actor": "LINK RUIZ",
                       "character": "C.O. Luis Ochoa"}, cast["pairs"])

    def test_reversed_crew_columns_are_not_cast(self):
        cast = reads.cast_from_credits(self._doc([
            {"name": "JEAN HOLT", "paired_with": "Key Hair",
             "header": None, "kind": "job"},
            {"name": "RITA POLLARD",
             "paired_with": "Hair Department Head",
             "header": None, "kind": "job"}]))
        self.assertEqual(cast["pairs"], [])
        self.assertEqual(sorted(cast["names"]),
                         ["JEAN HOLT", "RITA POLLARD"])

    def test_solo_principal_listing_stays_a_bare_name(self):
        cast = reads.cast_from_credits(self._doc([
            {"name": "VERA CALDWELL", "paired_with": None,
             "header": None, "kind": "listing"}]))
        self.assertEqual(cast["pairs"], [])
        self.assertEqual(cast["names"], ["VERA CALDWELL"])


class TestCharacterNameRule(unittest.TestCase):
    def test_credits_backed_character_name_survives_critique(self):
        from lib import critic
        doc = {"persons": [{"person_id": 0, "name": "Toni Torres",
                            "character_name": "Mira Santos",
                            "evidence": [{"mechanism": "credits_cast",
                                          "actor": "LENA VOSS",
                                          "character": "Mira Santos"}]}],
               "vetoes": []}
        out = critic.critique(doc)
        self.assertEqual(out["persons"][0]["character_name"],
                         "Mira Santos")
        self.assertEqual(out["vetoes"], [])

    def test_unsupported_character_name_is_removed(self):
        # whatever produced it — a similarity join, a bug — no stated
        # mapping means no character_name
        from lib import critic
        doc = {"persons": [{"person_id": 0, "name": "Billy Ray Marsh",
                            "character_name": "Billy",
                            "evidence": []}], "vetoes": []}
        out = critic.critique(doc)
        self.assertNotIn("character_name", out["persons"][0])
        self.assertEqual(out["vetoes"][0]["check"],
                         "unsupported_character_name")


class TestLanguageGate(unittest.TestCase):
    def test_non_english_transcript_yields_no_cues(self):
        # German capitalizes every noun: without the gate the vocative
        # shape would extract junk the English stoplist cannot filter
        turns = [_turn("SPEAKER_00", 0, 4, "Haus, das ist gut.", 0),
                 _turn("SPEAKER_01", 5, 9, "Haus, wirklich schön.", 1)]
        self.assertEqual(
            transcript_cues.extract_cues(turns, language="de"), [])
        self.assertEqual(
            transcript_cues.extract_cues(turns, language="fr-FR"), [])

    def test_english_variants_pass_the_gate(self):
        turns = [_turn("SPEAKER_00", 0, 5, "I'm Miriam Okafor.", 0)]
        for lang in ("en", "EN", "en-US"):
            cues = transcript_cues.extract_cues(turns, language=lang)
            self.assertEqual(cues[0]["name"], "Miriam Okafor", lang)

    def test_parse_transcript_carries_language(self):
        from lib import sources
        out = sources.parse_transcript(
            {"language": "FR", "language_confidence": 0.99,
             "sentences": [{"id": 0, "text": "Bonjour.", "start": 0,
                            "end": 1, "speaker_id": "SPEAKER_00"}]})
        self.assertEqual(out["language"], "fr")

    def test_parse_transcript_unknown_is_not_a_speaker(self):
        # "UNKNOWN" is diarization's rejection label (2026-09-15): its
        # aggregate is a union of DIFFERENT voices, so it must never
        # become a person — but the turns are real speech and must
        # survive for overlay-ambiguity abstention and share
        # denominators.
        from lib import sources
        out = sources.parse_transcript(
            {"sentences": [
                {"id": 0, "text": "Hello.", "start": 0, "end": 2,
                 "speaker_id": "SPEAKER_00"},
                {"id": 1, "text": "Hm.", "start": 2, "end": 3,
                 "speaker_id": "UNKNOWN"},
                {"id": 2, "text": "Bye.", "start": 3, "end": 4,
                 "speaker_id": "SPEAKER_00"}]})
        self.assertNotIn("UNKNOWN", out["speakers"])
        self.assertEqual(len(out["turns"]), 3)
        self.assertEqual(out["speakers"], {"SPEAKER_00": 3.0})

    def test_parse_transcript_overlap_segments_credit_both_voices(self):
        # overlap_segments (2026-09-18): cross-talk seconds accrue to
        # BOTH listed speakers — SUGGESTIVE evidence used only in the
        # aggregate denominators; a speaker sentences[] never
        # established does NOT get invented from overlap evidence.
        from lib import sources
        out = sources.parse_transcript(
            {"sentences": [
                {"id": 0, "text": "Hello.", "start": 0, "end": 2,
                 "speaker_id": "SPEAKER_00"},
                {"id": 1, "text": "No, you.", "start": 2, "end": 4,
                 "speaker_id": "SPEAKER_01"}],
             "overlap_segments": [
                {"start": 1.5, "end": 2.5,
                 "speakers": ["SPEAKER_00", "SPEAKER_01"]},
                {"start": 3.0, "end": 3.5,
                 "speakers": ["SPEAKER_01", "SPEAKER_07"]}]})
        # 00: 2.0 sentence + 1.0 overlap; 01: 2.0 + 1.0 + 0.5
        self.assertAlmostEqual(out["speakers"]["SPEAKER_00"], 3.0)
        self.assertAlmostEqual(out["speakers"]["SPEAKER_01"], 3.5)
        self.assertNotIn("SPEAKER_07", out["speakers"])

    def test_parse_transcript_without_overlap_segments_is_unchanged(self):
        # pre-2026-09-18 transcripts have no field: byte-identical math
        from lib import sources
        out = sources.parse_transcript(
            {"sentences": [
                {"id": 0, "text": "Hi.", "start": 0, "end": 2,
                 "speaker_id": "SPEAKER_00"}]})
        self.assertEqual(out["speakers"], {"SPEAKER_00": 2.0})


class TestCompanyCards(unittest.TestCase):
    def test_production_logo_yields_no_person(self):
        # the witness types a studio card organization (or names []);
        # nothing on this side turns company text into a person
        n = {"name": "Harborlight Films", "type": "organization",
             "descriptor": None}
        doc = {"blocks": [{"start": 1.0, "cards": [
            {"t": 1.0, "lines": ["HARBORLIGHT", "FILMS"], "names": [n]},
            {"t": 1.5, "lines": ["HARBORLIGHT", "FILMS"],
             "names": [dict(n)]}]}]}
        self.assertEqual(reads.cues_from_titles(doc), [])

# ─────────────────────────────────────────────────────────────────────────────
# review fixes, 2026-09-04 — role word-boundary, per-block chrome,
# zero-duration guard, ledger, the held-state submit contract
# ─────────────────────────────────────────────────────────────────────────────

from unittest import mock  # noqa: E402

from lib import pb_common  # noqa: E402


class TestRoleWordBoundary(unittest.TestCase):
    def test_substring_never_assigns_a_role(self):
        # 'anchor' in 'Anchorage', 'host' in 'Ghostwriter' (review F6)
        self.assertIsNone(reads._role_from("Anchorage Resident"))
        self.assertIsNone(reads._role_from("Ghostwriter"))

    def test_whole_word_assigns(self):
        self.assertEqual(reads._role_from("Metro News Anchor"), "anchor")
        self.assertEqual(reads._role_from("weather"), "weather")

    def test_assign_roles_appends_one_evidence_entry(self):
        doc = {"persons": [{"name": "A", "evidence": [
            {"mechanism": "overlay_text", "role_text": "News Anchor"},
            {"mechanism": "overlay_text", "role_text": "Anchor"}]}]}
        out = reads.assign_roles(doc)
        p = out["persons"][0]
        self.assertEqual(p["role"], "anchor")
        self.assertEqual(
            sum(1 for e in p["evidence"]
                if e.get("mechanism") == "role_text"), 1)


class TestChromePerBlock(unittest.TestCase):
    """Chrome (show brands, bugs) is handled by TYPING at the read since
    product v2 — a brand is program/organization, never person — so a
    real person may recur without a repetition veto eating them."""

    def test_recurring_person_across_blocks_still_cues(self):
        # an anchor re-chyroned in many blocks is a person every time —
        # the old cross-block chrome veto would have dropped them
        n = {"name": "Dana Whitfield", "type": "person",
             "descriptor": "Anchor"}
        blocks = [{"start": float(i * 100), "cards": [
            {"t": i * 100.0, "lines": ["x"], "names": [dict(n)]},
            {"t": i * 100.0 + 0.4, "lines": ["x"], "names": [dict(n)]}]}
            for i in range(4)]
        cues = reads.cues_from_titles({"blocks": blocks})
        self.assertEqual(len(cues), 4)      # one per block
        self.assertTrue(all(c["name"] == "Dana Whitfield" for c in cues))

    def test_typed_brand_never_cues_however_often_it_repeats(self):
        n = {"name": "The Real Harbor Lives", "type": "program",
             "descriptor": None}
        blocks = [{"start": float(i * 100), "cards": [
            {"t": i * 100.0, "lines": ["x"], "names": [dict(n)]},
            {"t": i * 100.0 + 0.4, "lines": ["x"], "names": [dict(n)]}]}
            for i in range(4)]
        self.assertEqual(reads.cues_from_titles({"blocks": blocks}), [])


class TestPlanReadsGuards(unittest.TestCase):
    def test_zero_duration_never_schedules_credits(self):
        plan = reads.plan_reads([], None, [], 0.0, credits_hint=True,
                                has_shot_metadata=False)
        self.assertIsNone(plan["requests"])

    def test_chyron_spans_request_the_lower_third_crop(self):
        cues = [{"kind": "introduction", "t": 100.0}]
        shots = [("s3", 40.0, 46.0)]     # head text shot: full frame
        plan = reads.plan_reads(cues, None, shots, 1000.0)
        req = next(r for r in plan["requests"]
                   if r["category"] == "title-read")
        by_reason = {s["reason"]: s for s in plan["spans"]}
        self.assertEqual(by_reason["introduction"].get("crop"),
                         "lower_third")
        self.assertNotIn("crop", by_reason["text_shot:s3"])
        # and the emitted request spans carry it through
        crops = {s["id"]: s.get("crop") for s in req["spans"]}
        self.assertIn("lower_third", crops.values())

    def test_fullframe_overlap_uncrops_and_widens_the_merge(self):
        # review 2026-09-05: a full-frame text_shot swallowed by a
        # cropped chyron span must UN-CROP and WIDEN it, never vanish
        cues = [{"kind": "introduction", "t": 100.0}]
        shots = [("s7", 104.0, 112.0)]
        plan = reads.plan_reads(cues, None, shots, 600.0)
        self.assertEqual(len(plan["spans"]), 1)
        span = plan["spans"][0]
        self.assertEqual((span["start"], span["end"]), (100.0, 112.0))
        self.assertNotIn("crop", span)     # widened AND un-cropped

    def test_cropped_ledger_entry_never_satisfies_fullframe_need(self):
        led = [(100.0, 108.0, "title-read", "lower_third",
                reads.PRODUCT_V)]
        shots = [("s7", 104.0, 107.0)]     # full-frame need, same window
        plan = reads.plan_reads([], None, shots, 600.0, ledger=led)
        self.assertEqual(plan["skipped"]["ledgered"], 0)
        self.assertEqual(len(plan["spans"]), 1)
        # but a cropped need IS satisfied by the cropped current entry
        cues = [{"kind": "introduction", "t": 101.0}]
        plan2 = reads.plan_reads(cues, None, [], 600.0, ledger=led)
        self.assertEqual(plan2["skipped"]["ledgered"], 1)
        # a legacy 3-tuple entry (pre-crop, pre-v2) satisfies NOTHING
        # now — its product lacks the typed extraction (re-read once)
        led3 = [(100.0, 108.0, "title-read")]
        plan3 = reads.plan_reads([], None, shots, 600.0, ledger=led3)
        self.assertEqual(plan3["skipped"]["ledgered"], 0)

    def test_ledger_candidates_carry_the_crop(self):
        cues = [{"kind": "introduction", "t": 100.0}]
        plan = reads.plan_reads(cues, None, [], 600.0)
        entry = next(c for c in plan["ledger_candidates"]
                     if c["category"] == "title-read")
        self.assertEqual(entry.get("crop"), "lower_third")

    def test_ledger_candidates_cover_every_scheduled_read(self):
        cues = [{"kind": "introduction", "t": 100.0}]
        # credits now require typed credit_roll evidence (2026-09-30)
        plan = reads.plan_reads(cues, None, [], 1000.0,
                                credit_roll_spans=[(940.0, 980.0)])
        cats = sorted(c["category"] for c in plan["ledger_candidates"])
        self.assertEqual(cats, ["credits-read", "title-read"])
        for c in plan["ledger_candidates"]:
            self.assertLess(c["t0"], c["t1"])


class TestLedgerModule(unittest.TestCase):
    def setUp(self):
        from lib import ledger
        self.ledger = ledger

    def test_uri_is_derived_beside_the_runs(self):
        uri = self.ledger.uri_for(
            "s3://b/micro-workflow/cast-resolution/u1/_plan/plan.json",
            "seg-9")
        self.assertEqual(
            uri, "s3://b/micro-workflow/cast-resolution/_ledger/seg-9.json")

    def test_append_dedups_exact_entries(self):
        stored = {}
        e = {"t0": 1.0, "t1": 2.0, "category": "title-read",
             "output_s3_prefix": "s3://b/p"}
        with mock.patch.object(self.ledger.s3io, "get_json",
                               side_effect=lambda u: stored.get(u)), \
             mock.patch.object(self.ledger.s3io, "put_json",
                               side_effect=stored.__setitem__):
            self.ledger.append("s3://b/l.json", [e])
            self.ledger.append("s3://b/l.json", [e, dict(e, t0=5.0)])
        self.assertEqual(len(stored["s3://b/l.json"]["entries"]), 2)


class TestScheduleReadsContract(unittest.TestCase):
    """The held-state submit rules: a response with a job handle means
    fi OWNS the token (park; review F3); a refusal or crash degrades to
    a self-completed skip; an already-gone token is a benign race."""

    def setUp(self):
        os.environ["FI_GATEWAY_ARN"] = "arn:fake"
        import importlib
        from states.read_onscreen_text import lambda_function as read_onscreen_text
        importlib.reload(read_onscreen_text)
        self.sr = read_onscreen_text
        self.plan = {
            "uuid": "u1", "use_batch": False,
            "read_onscreen_text": True, "llm_enrichment": True,
            "sibling_workflow_ids": {"shot_segmentation": "seg-1"},
            "inputs": {"shot_segmentation_s3uri":
                       "s3://test-proxy/micro-workflow/shot-segmentation"
                       "/seg-1"}}

    def tearDown(self):
        os.environ.pop("FI_GATEWAY_ARN", None)
        import importlib
        importlib.reload(self.sr)

    def _run(self, submit_response, plan_reads_result=None):
        sr = self.sr
        calls = {"sent": [], "put": []}

        def invoke(lam, arn, target, tool, payload):
            if tool == "describe_tool":
                return {"features": ["callback_task_token"]}
            if isinstance(submit_response, Exception):
                raise submit_response
            return submit_response

        with mock.patch.object(sr.pb_common, "invoke_gateway_tool",
                               side_effect=invoke), \
             mock.patch.object(sr.s3io, "get_json",
                               return_value=self.plan), \
             mock.patch.object(sr.s3io, "put_json",
                               side_effect=lambda u, d:
                               calls["put"].append((u, d))), \
             mock.patch.object(sr.sources, "load_transcript",
                               return_value=None), \
             mock.patch.object(sr.sources, "load_faces",
                               return_value=None), \
             mock.patch.object(sr.sources, "load_shots",
                               return_value={"s0": (0.0, 100.0)}), \
             mock.patch.object(sr.ledger, "load", return_value=[]), \
             mock.patch.object(sr.reads, "plan_reads",
                               return_value=plan_reads_result or {
                                   "requests": [{"category": "title-read",
                                                 "spans": []}],
                                   "spans": [], "credits": False,
                                   "skipped": {"ledgered": 0},
                                   "ledger_candidates": []}), \
             mock.patch.object(sr._sfn, "send_task_success",
                               side_effect=lambda **k:
                               calls["sent"].append(k)):
            out = sr.lambda_handler(
                {"uuid": "u1", "task_token": "tok",  # nosec B105
                 "plan_s3uri": "s3://test-proxy/micro-workflow/"
                               "cast-resolution/u1/_plan/plan.json"}, None)
        return out, calls

    def test_crop_is_stripped_when_the_witness_lacks_the_capability(self):
        sr = self.sr
        captured = {}

        def invoke(lam, arn, target, tool, payload):
            if tool == "describe_tool":
                return {"features": ["callback_task_token"]}  # no span_crop
            captured.update(payload)
            return {"job_id": "arn:j", "status": "RUNNING",
                    "output_s3_prefix": "s3://x/y"}
        plan_result = {
            "requests": [{"category": "title-read", "spans": [
                {"start": 1.0, "end": 2.0, "id": "pb-000",
                 "crop": "lower_third"}]}],
            "spans": [], "credits": False,
            "skipped": {"ledgered": 0},
            "ledger_candidates": [{"t0": 1.0, "t1": 2.0,
                                   "category": "title-read",
                                   "crop": "lower_third"}]}
        with mock.patch.object(sr.pb_common, "invoke_gateway_tool",
                               side_effect=invoke), \
             mock.patch.object(sr.s3io, "get_json",
                               return_value=self.plan), \
             mock.patch.object(sr.s3io, "put_json"), \
             mock.patch.object(sr.sources, "load_transcript",
                               return_value=None), \
             mock.patch.object(sr.sources, "load_faces",
                               return_value=None), \
             mock.patch.object(sr.sources, "load_shots",
                               return_value={"s0": (0.0, 100.0)}), \
             mock.patch.object(sr.ledger, "load", return_value=[]), \
             mock.patch.object(sr.reads, "plan_reads",
                               return_value=plan_result), \
             mock.patch.object(sr._sfn, "send_task_success"):
            out = sr.lambda_handler(
                {"uuid": "u1", "task_token": "tok",  # nosec B105
                 "plan_s3uri": "s3://test-proxy/micro-workflow/"
                               "cast-resolution/u1/_plan/plan.json"}, None)
        self.assertTrue(out.get("parked"))
        span = captured["assets"][0]["requests"][0]["spans"][0]
        self.assertNotIn("crop", span)
        self.assertNotIn("crop", plan_result["ledger_candidates"][0])

    def test_absorbed_failed_run_parks_and_never_touches_the_token(self):
        out, calls = self._run({
            "job_id": "arn:aws:states:r:1:execution:m:x",
            "status": "FAILED", "error": "States.Timeout",
            "output_s3_prefix": "s3://test-proxy/mw/fi/x",
            "message": "identical request already submitted"})
        self.assertTrue(out.get("parked"))
        self.assertEqual(calls["sent"], [])       # fi owns the token

    def test_gateway_refusal_self_completes_a_skip(self):
        out, calls = self._run({"error": True, "errorCode": "X",
                                "message": "assets must be a list"})
        self.assertEqual(out["reads"], "skipped")
        self.assertEqual(len(calls["sent"]), 1)

    def test_invocation_crash_degrades_to_skip(self):
        out, _ = self._run(RuntimeError("frame-inspection crashed"))
        self.assertEqual(out["reads"], "skipped")
        self.assertIn("submit_failed", out["reads_reason"])

    def test_fully_ledgered_run_hands_binding_the_prior_products(self):
        sr = self.sr
        entries = [{"t0": 0.0, "t1": 9.0, "category": "title-read",
                    "output_s3_prefix": "s3://test-proxy/mw/fi/old"}]
        out, calls = None, None
        with mock.patch.object(sr.ledger, "load", return_value=entries):
            out, calls = self._run_with_ledger(entries)
        self.assertEqual(out["reads"], "ledgered")
        self.assertEqual(out["read_prefixes"],
                         ["s3://test-proxy/mw/fi/old"])

    def _run_with_ledger(self, entries):
        sr = self.sr
        calls = {"sent": []}
        with mock.patch.object(sr.pb_common, "invoke_gateway_tool",
                               return_value={"features":
                                             ["callback_task_token"]}), \
             mock.patch.object(sr.s3io, "get_json",
                               return_value=self.plan), \
             mock.patch.object(sr.sources, "load_transcript",
                               return_value=None), \
             mock.patch.object(sr.sources, "load_faces",
                               return_value=None), \
             mock.patch.object(sr.sources, "load_shots",
                               return_value={"s0": (0.0, 100.0)}), \
             mock.patch.object(sr.ledger, "load", return_value=entries), \
             mock.patch.object(sr.reads, "plan_reads",
                               return_value={
                                   "requests": None, "spans": [],
                                   "credits": False,
                                   "skipped": {"ledgered": 2},
                                   "ledger_candidates": []}), \
             mock.patch.object(sr._sfn, "send_task_success",
                               side_effect=lambda **k:
                               calls["sent"].append(k)):
            out = sr.lambda_handler(
                {"uuid": "u1", "task_token": "tok",  # nosec B105
                 "plan_s3uri": "s3://test-proxy/micro-workflow/"
                               "cast-resolution/u1/_plan/plan.json"}, None)
        return out, calls

    def test_witness_without_callback_support_skips_before_planning(self):
        sr = self.sr
        sent = []
        with mock.patch.object(sr.pb_common, "invoke_gateway_tool",
                               return_value={"features": []}), \
             mock.patch.object(sr.s3io, "get_json",
                               return_value=self.plan), \
             mock.patch.object(sr._sfn, "send_task_success",
                               side_effect=lambda **k: sent.append(k)):
            out = sr.lambda_handler(
                {"uuid": "u1", "task_token": "tok",  # nosec B105
                 "plan_s3uri": "s3://x/_plan/plan.json"}, None)
        self.assertEqual(out["reads_reason"], "fi_no_callback")

    def test_token_gone_race_is_benign(self):
        from botocore.exceptions import ClientError
        sr = self.sr
        gone = ClientError({"Error": {"Code": "TaskDoesNotExist",
                                      "Message": "x"}}, "SendTaskSuccess")
        with mock.patch.object(sr._sfn, "send_task_success",
                               side_effect=gone):
            out = sr._skip("tok", "u1", "s3://x/plan.json", "r")
        self.assertEqual(out["reads"], "skipped")   # no raise


class TestInvokeGatewayTool(unittest.TestCase):
    def test_function_error_raises_never_parses(self):
        # an unhandled gateway exception returns 200 + FunctionError; the
        # old copies parsed the error body as a tool response (review F1)
        class _Payload:
            def read(self):
                return (b'{"errorMessage": "boom", '
                        b'"errorType": "KeyError"}')

        class _Lam:
            def invoke(self, **kwargs):
                return {"FunctionError": "Unhandled", "Payload": _Payload()}
        with self.assertRaises(RuntimeError) as cm:
            pb_common.invoke_gateway_tool(_Lam(), "arn:x", "t", "op", {})
        self.assertIn("boom", str(cm.exception))


class TestBindPersonsConsumption(unittest.TestCase):
    """Witness products come from fi's OWN manifest at the wake
    payload's prefix (review F4), and usage reports the witness run's
    recorded spend (review F8)."""

    def _run(self, wait_result, s3_docs):
        from states.bind_persons import lambda_function as bf
        puts = {}
        plan_uri = ("s3://test-proxy/micro-workflow/cast-resolution/u1/"
                    "_plan/plan.json")
        docs = {plan_uri: {
            "uuid": "u1", "tool_version": "test",
            "sibling_workflow_ids": {"shot_segmentation": "seg-1"},
            "inputs": {"shot_segmentation_s3uri":
                       "s3://test-proxy/micro-workflow/shot-segmentation"
                       "/seg-1"}}, **s3_docs}

        def get_json(uri):
            if uri not in docs:
                raise KeyError(uri)
            return docs[uri]

        with mock.patch.object(bf.s3io, "get_json",
                               side_effect=get_json), \
             mock.patch.object(bf.s3io, "put_json",
                               side_effect=puts.__setitem__), \
             mock.patch.object(bf.sources, "load_transcript",
                               return_value=None), \
             mock.patch.object(bf.sources, "load_faces",
                               return_value=None), \
             mock.patch.object(bf.sources, "load_shots",
                               return_value={}), \
             mock.patch.object(bf.sources, "load_overlay_names",
                               return_value=None), \
             mock.patch.object(bf.bind_pipeline, "bind",
                               return_value={"persons": [],
                                             "vetoes": []}), \
             mock.patch.object(bf.persons, "remap_tables",
                               return_value=({}, {})), \
             mock.patch.object(bf.ledger, "append") as led:
            out = bf.lambda_handler({"plan_s3uri": plan_uri,
                                     "reads_result": wait_result}, None)
        return out, puts, led

    def test_ready_wake_also_consumes_ledgered_products(self):
        # never-re-pay must not become never-RE-USE (2026-09-09): a
        # fresh read tranche on the ready path must still consume the
        # LEDGERED runs' products — here the credits live only in the
        # earlier ledgered run, and the fresh run has titles only
        fi_new = "s3://test-proxy/mw/fi/new"
        fi_old = "s3://test-proxy/mw/fi/old"
        plan_prefix = "s3://test-proxy/micro-workflow/cast-resolution/u1"
        reads_uri = f"{plan_prefix}/_plan/reads.json"
        ledger_uri = f"{plan_prefix}/_ledger/seg-1.json"
        out, puts, led = self._run(
            {"status": "SUCCEEDED", "output_s3_prefix": fi_new,
             "job_id": "arn:j"},
            {reads_uri: {"ledger_uri": ledger_uri,
                         "ledger_candidates": []},
             ledger_uri: {"entries": [
                 {"t0": 0, "t1": 9, "category": "credits-read",
                  "product_v": 2, "output_s3_prefix": fi_old}]},
             f"{fi_old}/inspection_manifest.json": {"products": [
                 {"category": "credits-read",
                  "s3uri": f"{fi_old}/0/credits.json"}]},
             f"{fi_old}/0/credits.json": {"blocks": [{"start": 1.0,
                                                      "end": 2.0,
                                                      "cards": []}]},
             f"{fi_new}/inspection_manifest.json": {"products": [
                 {"category": "title-read",
                  "s3uri": f"{fi_new}/0/titles.json"}]},
             f"{fi_new}/0/titles.json": {"blocks": []},
             f"{fi_new}/usage_tracking.json": {"total_estimated_cost": 1.0,
                                               "cost_is_complete": True}})
        credits = puts.get(f"{plan_prefix}/credits.json")
        self.assertIsNotNone(credits, "ledgered credits product lost")
        self.assertEqual(len(credits["blocks"]), 1)

    def test_ready_wake_fetches_via_manifest_and_reports_cost(self):
        fi = "s3://test-proxy/mw/fi/x"
        out, puts, led = self._run(
            {"status": "SUCCEEDED", "output_s3_prefix": fi,
             "job_id": "arn:j"},
            {f"{fi}/inspection_manifest.json": {"products": [
                {"category": "credits-read", "s3uri": f"{fi}/0/credits.json"},
                {"category": "title-read", "s3uri": f"{fi}/0/titles.json"}]},
             f"{fi}/0/credits.json": {"blocks": []},
             f"{fi}/0/titles.json": {"blocks": []},
             f"{fi}/usage_tracking.json": {"total_estimated_cost": 1.23,
                                           "cost_is_complete": True},
             "s3://test-proxy/micro-workflow/cast-resolution/u1/_plan/"
             "reads.json": {
                 "ledger_uri": "s3://test-proxy/mw/pb/_ledger/seg-1.json",
                 "ledger_candidates": [{"t0": 0.0, "t1": 9.0,
                                        "category": "title-read"}]}})
        self.assertEqual(out["reads"], "ready")
        usage = puts["s3://test-proxy/micro-workflow/cast-resolution/u1/"
                     "usage_tracking.json"]
        self.assertEqual(usage["total_estimated_cost"], 1.23)
        self.assertTrue(usage["cost_is_complete"])
        self.assertTrue(led.called)
        entry = led.call_args[0][1][0]
        self.assertEqual(entry["output_s3_prefix"], fi)

    def test_failed_wake_degrades_and_flags_incomplete_cost(self):
        out, puts, led = self._run(
            {"status": "FAILED", "job_id": "arn:j",
             "output_s3_prefix": "s3://test-proxy/mw/fi/x"}, {})
        self.assertEqual(out["reads"], "failed")
        usage = puts["s3://test-proxy/micro-workflow/cast-resolution/u1/"
                     "usage_tracking.json"]
        self.assertFalse(usage["cost_is_complete"])
        self.assertFalse(led.called)

    def test_skipped_wake_is_honest_free_evidence(self):
        out, puts, _ = self._run(
            {"reads": "skipped", "reads_reason": "no_fi_gateway"}, {})
        self.assertEqual(out["reads"], "skipped")
        usage = puts["s3://test-proxy/micro-workflow/cast-resolution/u1/"
                     "usage_tracking.json"]
        self.assertEqual(usage["total_estimated_cost"], 0.0)
        self.assertTrue(usage["cost_is_complete"])


class TestFlagIdentity(unittest.TestCase):
    """Flags are normalized into the hash: omitted == explicit default
    (same work, absorbs), and each flip is DIFFERENT work."""

    def _wid(self, extra):
        import lambda_function as gw
        with mock.patch.object(gw._sfn, "start_execution",
                               return_value={"executionArn":
                                             "arn:aws:states:r:1:"
                                             "execution:m:x"}):
            base = {"speech_diarization_s3uri":
                    "s3://test-proxy/micro-workflow/speech-diarization/d1"}
            return gw.lambda_handler(
                {**base, **extra},
                _Ctx("cast-resolution___resolve_cast"))["workflow_id"]

    def test_omitted_equals_explicit_defaults(self):
        self.assertEqual(self._wid({}),
                         self._wid({"use_batch": True,
                                    "read_onscreen_text": True,
                                    "llm_enrichment": True}))

    def test_each_flag_flip_is_different_work(self):
        base = self._wid({})
        for flag in ("use_batch", "read_onscreen_text",
                     "llm_enrichment"):
            self.assertNotEqual(base, self._wid({flag: False}), flag)


class TestWitnessPairings(unittest.TestCase):
    """cast_from_credits over the witness's typed pairings (product v2,
    2026-09-09): `kind: character` states the actor->part mapping — the
    card's own claim in any language; every other kind is presence
    evidence. The crew/connector lexicons and case-direction typography
    rules are gone, and the verbatim lines are never a name source."""

    def _doc(self, pairings, lines=("x",)):
        return {"blocks": [{"cards": [
            {"t": 1.0, "lines": list(lines), "pairings": pairings}]}]}

    def test_character_kind_states_the_pair(self):
        cast = reads.cast_from_credits(self._doc(
            [{"name": "RAY COLTER", "paired_with": "Dario",
              "header": None, "kind": "character"}]))
        self.assertEqual(cast["pairs"],
                         [{"actor": "RAY COLTER",
                           "character": "Dario"}])

    def test_character_kind_needs_no_case_signal(self):
        # the old typography rule abstained here; the witness's kind
        # settles it — and works on scripts with no case at all
        cast = reads.cast_from_credits(self._doc(
            [{"name": "Marie Duval", "paired_with": "Camille Noel",
              "header": None, "kind": "character"}]))
        self.assertEqual(cast["pairs"],
                         [{"actor": "Marie Duval",
                           "character": "Camille Noel"}])

    def test_job_and_billing_and_listing_are_presence_only(self):
        cast = reads.cast_from_credits(self._doc(
            [{"name": "MARK ELLISON", "paired_with": "Directed by",
              "header": None, "kind": "job"},
             {"name": "JOHN MARR", "paired_with": "with",
              "header": None, "kind": "billing"},
             {"name": "TOM BAYLES", "paired_with": None,
              "header": "Key Asst. Location Managers",
              "kind": "listing"}]))
        self.assertEqual(cast["pairs"], [])
        self.assertEqual(sorted(cast["names"]),
                         ["JOHN MARR", "MARK ELLISON", "TOM BAYLES"])

    def test_kindless_legacy_entry_degrades_to_presence(self):
        # pre-v2 pairing: no kind — never guess the mapping; the ledger
        # versioning re-reads the span for the typed product
        cast = reads.cast_from_credits(self._doc(
            [{"name": "RAY COLTER", "paired_with": "Dario",
              "header": None}]))
        self.assertEqual(cast["pairs"], [])
        self.assertEqual(cast["names"], ["RAY COLTER"])

    def test_lines_are_never_a_name_source(self):
        # the v2 prompt emits one entry per person name on the card, so
        # an as-line in the verbatim lines carries no extra information
        doc = {"blocks": [{"cards": [
            {"t": 1.0, "lines": ["Victor Hale as Dan Reyes"],
             "pairings": []},
            {"t": 2.0, "lines": ["Rosa Quint as Ada Lorne"]}]}]}
        cast = reads.cast_from_credits(doc)
        self.assertEqual(cast["pairs"], [])
        self.assertEqual(cast["names"], [])

    def test_duplicate_pairings_across_scroll_frames_dedup(self):
        p = {"name": "RAY COLTER", "paired_with": "Dario",
             "header": None, "kind": "character"}
        doc = {"blocks": [{"cards": [
            {"t": 1.0, "lines": ["x"], "pairings": [p]},
            {"t": 1.5, "lines": ["x"], "pairings": [dict(p)]}]}]}
        self.assertEqual(len(reads.cast_from_credits(doc)["pairs"]), 1)


class TestOverclusteringAccommodation(unittest.TestCase):
    """Diarization over-merge is an INPUT CONDITION (user direction,
    2026-09-09): the accommodation welds round-2-named voices into
    their recognizer-named faces (trust order collection > celebrity),
    face-self-overlap vetoed. Top-talker recovery was DELETED
    2026-09-24 (subsumed by the class-1/2 matcher: 34/35 welds, and it
    caused the one proven misbinding)."""

    @staticmethod
    def _face_person(pid, name, spans_ms):
        return {"person_id": pid, "type": "silent_face", "name": name,
                "aliases": [], "speaker_ids": [], "face_ids": [pid],
                "speech_sec": 0.0, "screen_sec": 900.0,
                "appearances": [{"begin_ms": a, "end_ms": b}
                                for a, b in spans_ms],
                "name_mechanisms": ["collection"], "evidence": []}

    @staticmethod
    def _voice_person(pid, sid, name=None, speech_sec=300.0,
                      mechanisms=("cast_adjudication",)):
        return {"person_id": pid, "type": "uncertain", "name": name,
                "aliases": [], "speaker_ids": [sid], "face_ids": [],
                "speech_sec": speech_sec, "screen_sec": 0.0,
                "appearances": [],
                "name_mechanisms": list(mechanisms) if name else [],
                "evidence": []}

    def test_same_name_voice_absorbed_into_collection_face(self):
        doc = {"persons": [
            self._face_person(0, "Nora Vane", [(0, 600_000)]),
            self._voice_person(141, "SPEAKER_20", name="Nora Vane")],
            "vetoes": []}
        out = persons.accommodate_overclustering(doc)
        self.assertEqual(len(out["persons"]), 1)
        p = out["persons"][0]
        self.assertEqual(p["name"], "Nora Vane")
        self.assertEqual(p["speaker_ids"], ["SPEAKER_20"])
        self.assertEqual(p["type"], "on_camera_speaker")
        self.assertIn("name_agreement", p["name_mechanisms"])

    def test_face_overlap_contradiction_blocks_the_absorb(self):
        # the voice person carries its OWN face on screen at the same
        # time as the keeper's — two people whatever the names say
        voice = self._voice_person(9, "SPEAKER_02", name="Nora Vane")
        voice["face_ids"] = [7]
        voice["appearances"] = [{"begin_ms": 0, "end_ms": 60_000}]
        doc = {"persons": [
            self._face_person(0, "Nora Vane", [(0, 60_000)]), voice],
            "vetoes": []}
        out = persons.accommodate_overclustering(doc)
        self.assertEqual(len(out["persons"]), 2)
        self.assertTrue(any(v.get("action") == "absorb_blocked"
                            for v in out["vetoes"]))

    def test_celebrity_keeper_welds_same_name_voice(self):
        # GDPR posture: no collection (collection_id "") — celebrity is
        # the only face-naming tier, and the host weld must still work
        face = self._face_person(0, "Nora Vane", [(0, 600_000)])
        face["name_mechanisms"] = ["celebrity"]
        doc = {"persons": [
            face, self._voice_person(141, "SPEAKER_20", name="Nora Vane")],
            "vetoes": []}
        out = persons.accommodate_overclustering(doc)
        self.assertEqual(len(out["persons"]), 1)
        self.assertEqual(out["persons"][0]["name"], "Nora Vane")


class TestCollisionCollectionPreference(unittest.TestCase):
    def _doc(self, mechanisms_a, mechanisms_b):
        return {"persons": [
            {"person_id": 0, "name": "Nora Vane", "aliases": [],
             "type": "silent_face", "name_mechanisms": mechanisms_a,
             "evidence": []},
            {"person_id": 141, "name": "Nora Vane", "aliases": [],
             "type": "uncertain", "name_mechanisms": mechanisms_b,
             "evidence": []}], "vetoes": []}

    def test_collection_side_keeps_the_name(self):
        out = critic.critique(self._doc(["collection"],
                                        ["overlay_name",
                                         "cast_adjudication"]))
        a, b = out["persons"]
        self.assertEqual(a["name"], "Nora Vane")
        self.assertIsNone(b["name"])
        self.assertIn("Nora Vane", b["aliases"])
        veto = next(v for v in out["vetoes"]
                    if v["check"] == "name_collision")
        self.assertEqual(veto["resolution"], "collection_preferred")
        self.assertEqual(veto["kept_person_id"], 0)

    def test_two_collection_sides_still_strip_both(self):
        out = critic.critique(self._doc(["collection"], ["collection"]))
        self.assertTrue(all(p["name"] is None for p in out["persons"]))

    def test_no_collection_side_strips_both_as_before(self):
        out = critic.critique(self._doc(["celebrity"], ["overlay_name"]))
        self.assertTrue(all(p["name"] is None for p in out["persons"]))


class TestCritic2Contract(unittest.TestCase):
    """Tooth-2 v2 (post first-sweep): contradiction-only scope. The
    first sweep flagged 94% of named persons on absence-based doubt —
    these tests pin the mechanisms that prevent that class."""

    def setUp(self):
        from lib import critic2
        self.c2 = critic2

    def _person(self, **kw):
        return {"person_id": 1, "type": "silent_face", "name": "A B",
                "aliases": [], "speaker_ids": [], "face_ids": ["f1"],
                "name_mechanisms": ["celebrity"], "evidence": [
                    {"mechanism": "celebrity", "name": "A B"}], **kw}

    def test_single_oracle_person_gets_no_dossier(self):
        # one observed name, nothing to contradict -> rides free
        doc = {"persons": [self._person()]}
        self.assertEqual(self.c2.build_dossiers(doc), [])

    def test_only_suspicion_is_reviewable(self):
        # tokens chase what is NOT confirmed (user principle,
        # 2026-09-24): abstained names, mixed-voice clusters,
        # coherence repairs, tooth-1 vetoes, speech-inferred
        # characters — never corroboration itself
        for extra in ({"name": None, "aliases": ["C", "D"]},
                      {"character_name": "X"},   # no credits behind it
                      {"evidence": [{"mechanism": "class1_assignment",
                                     "mixed_voice": True}]},
                      {"evidence": [{"mechanism": "coherence_repair",
                                     "was": 10.0, "recomputed": 90.0}]},
                      {"evidence": [{"mechanism": "critic_veto",
                                     "kind": "name_collision"}]}):
            doc = {"persons": [self._person(**extra)]}
            self.assertEqual(len(self.c2.build_dossiers(doc)), 1, extra)

    def test_confirmed_corroboration_rides_free(self):
        # two agreeing mechanisms, precedence-minted aliases on a
        # NAMED person, a credits-backed character: all CONFIRMED —
        # reviewing them re-spends tokens on settled facts
        for extra in ({"name_mechanisms": ["celebrity", "self_id"]},
                      {"aliases": ["Old Chyron Name"]},
                      {"character_name": "X",
                       "evidence": [{"mechanism": "credits_cast",
                                     "actor": "B", "character": "X"}]}):
            doc = {"persons": [self._person(**extra)]}
            self.assertEqual(self.c2.build_dossiers(doc), [], extra)

    def test_dossier_evidence_is_idd_and_classed(self):
        doc = {"persons": [self._person(
            name=None, aliases=["A B", "C D"],
            name_mechanisms=["celebrity", "vocative_resolved"],
            evidence=[{"mechanism": "celebrity", "name": "A B"},
                      {"kind": "vocative_resolved", "name": "A B"}])]}
        d = self.c2.build_dossiers(doc)[0]
        self.assertEqual(
            [(e["id"], e["class"]) for e in d["evidence"]],
            [("e0", "observed"), ("e1", "inferred")])

    def test_uncited_suspect_is_dropped(self):
        texts = ['{"person_id": 1, "verdict": "suspect", '
                 '"suspect_field": "name", "reason": "no conflicts key"}',
                 '{"person_id": 2, "verdict": "suspect", '
                 '"conflicts": ["e0"], "reason": "one id only"}',
                 '{"person_id": 3, "verdict": "suspect", '
                 '"conflicts": ["q0", "q1"], "reason": "quotes only"}']
        self.assertEqual(self.c2.parse_reviews(texts), [])

    def test_cited_suspect_survives(self):
        texts = ['{"person_id": 4, "verdict": "suspect", '
                 '"suspect_field": "name", "conflicts": ["e0", "q1"], '
                 '"reason": "role text conflicts with quotes"}']
        flags = self.c2.parse_reviews(texts)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["conflicts"], ["e0", "q1"])



class TestRegistryPromptPairs(unittest.TestCase):
    """The matched prompt+parser pairs, MIGRATED to registry INTERNAL
    categories (2026-09-08, the admin-surface design). The parsers DROP
    unrecognized rows by design, so a schema drift between prompt and
    parser is silent zero-output — these pins read the SIBLING repo's
    prompt files (skip when it is not checked out; prompt-registry's
    own suite parses the same files unconditionally)."""

    _PAIRS = (
        ("cast-adjudicate",
         ("person_id", "verdict", "name", "character_name", "evidence")),
        ("cast-critic2",
         ("person_id", "verdict", "suspect_field", "conflicts")),
        ("cast-vocatives", ("window_id", "addressee")),
    )

    @staticmethod
    def _prompt(category):
        path = os.path.join(ROOT, "..", "prompt-registry-on-aws",
                            "prompts", category, "default.md")
        if not os.path.isfile(path):
            return None
        return _read(path, encoding="utf-8")

    def test_each_prompt_states_the_fields_its_parser_reads(self):
        for category, tokens in self._PAIRS:
            text = self._prompt(category)
            if text is None:
                self.skipTest("prompt-registry repo not checked out")
            for token in tokens:
                self.assertIn(f'"{token}"', text,
                              f"{category} no longer names {token!r} — "
                              "matched pair broken")

    def test_prompts_are_marked_internal(self):
        for category, _ in self._PAIRS:
            text = self._prompt(category)
            if text is None:
                self.skipTest("prompt-registry repo not checked out")
            self.assertIn("Audience: workflow-internal", text)

    def test_critic2_prompt_states_the_two_rules(self):
        text = self._prompt("cast-critic2")
        if text is None:
            self.skipTest("prompt-registry repo not checked out")
        self.assertIn("ABSENCE of additional evidence is NEVER suspect",
                      text)
        self.assertIn("NEVER question an observed item's reliability",
                      text)

    def test_build_records_carry_the_bound_system(self):
        from lib import adjudicate
        from lib import critic2
        from lib import vocatives_llm
        recs = critic2.build_records(
            [{"person_id": 1, "items": []}], "SYS-C")
        self.assertEqual(recs[0]["system"], "SYS-C")
        recs = adjudicate.build_records(
            [{"person_id": 1}], {"pairs": [], "names": []}, "SYS-A")
        self.assertEqual(recs[0]["system"], "SYS-A")
        recs = vocatives_llm.build_records(
            [{"id": 0, "name": "X", "lines": []}], "SYS-V")
        self.assertEqual(recs[0]["system"], "SYS-V")


class TestCastAdjudication(unittest.TestCase):
    """P2 cast_adjudication (2026-09-05): dossier gating and the
    fabrication firewall — names verbatim from the universe, characters
    verbatim from the pairs, citations required."""

    def setUp(self):
        from lib import adjudicate
        self.adj = adjudicate
        self.cast = {"pairs": [{"actor": "LENA VOSS",
                                "character": "Mira Santos"}],
                     "names": ["RAY COLTER"]}

    def test_no_universe_no_dossiers(self):
        doc = {"persons": [{"person_id": 1, "type": "voice_only",
                            "name": None, "speaker_ids": ["S1"],
                            "evidence": []}]}
        self.assertEqual(self.adj.build_dossiers(doc, {}), [])

    def test_complete_cast_identity_is_not_reviewed(self):
        doc = {"persons": [{"person_id": 1, "type": "on_camera_speaker",
                            "name": "A", "character_name": "B",
                            "speaker_ids": [], "evidence": []}]}
        self.assertEqual(self.adj.build_dossiers(doc, self.cast), [])

    def test_addressed_as_evidence_is_idd(self):
        doc = {"persons": [{"person_id": 2, "type": "on_camera_speaker",
                            "name": None, "speaker_ids": ["S1"],
                            "evidence": [{"kind": "vocative_resolved",
                                          "name": "Mira"}]}]}
        d = self.adj.build_dossiers(doc, self.cast)[0]
        self.assertEqual(d["evidence"][0]["kind"], "addressed_as")
        self.assertEqual(d["evidence"][0]["name"], "Mira")

    def test_invented_actor_or_character_is_dropped(self):
        texts = ['{"person_id": 2, "verdict": "assign", '
                 '"name": "MADE UP", "character_name": "Mira Santos", '
                 '"evidence": ["v0"], "reason": "x"}',
                 '{"person_id": 2, "verdict": "assign", '
                 '"name": "LENA VOSS", "character_name": "Mira", '
                 '"evidence": ["v0"], "reason": "not verbatim"}']
        self.assertEqual(self.adj.parse_verdicts(texts, self.cast), [])

    def test_uncited_assignment_is_dropped(self):
        texts = ['{"person_id": 2, "verdict": "assign", '
                 '"name": "LENA VOSS", '
                 '"character_name": "Mira Santos", '
                 '"evidence": [], "reason": "vibes"}']
        self.assertEqual(self.adj.parse_verdicts(texts, self.cast), [])

    def test_cited_verbatim_assignment_survives_and_applies(self):
        texts = ['{"person_id": 2, "verdict": "assign", '
                 '"name": "LENA VOSS", '
                 '"character_name": "Mira Santos", '
                 '"evidence": ["v0", "q1"], "reason": "addressed Mira"}']
        vs = self.adj.parse_verdicts(texts, self.cast)
        self.assertEqual(len(vs), 1)
        doc = {"persons": [{"person_id": 2, "type": "on_camera_speaker",
                            "name": None, "speaker_ids": ["S1"],
                            "evidence": []}]}
        out = self.adj.apply_verdicts(doc, vs)
        p = out["persons"][0]
        self.assertEqual(p["name"], "LENA VOSS")
        self.assertEqual(p["character_name"], "Mira Santos")
        self.assertIn("cast_adjudication", p["name_mechanisms"])

    def test_adjudicated_character_survives_the_critic(self):
        from lib import critic
        doc = {"persons": [{"person_id": 2, "type": "on_camera_speaker",
                            "name": "LENA VOSS",
                            "character_name": "Mira Santos",
                            "speaker_ids": ["S1"], "face_ids": [],
                            "aliases": [], "name_mechanisms": ["celebrity"],
                            "evidence": [{"mechanism": "cast_adjudication",
                                          "character": "Mira Santos"}]}],
               "unresolved": {"speakers": [], "faces": []}}
        out = critic.critique(doc)
        self.assertEqual(out["persons"][0].get("character_name"),
                         "Mira Santos")

    def test_wrong_join_is_dropped_by_the_agreement_check(self):
        # measured 2026-09-05: the model assigned 'MAY LIN' to a person
        # whose own evidence says 'Miriam Okafor' — verbatim but WRONG
        cast = {"pairs": [], "names": ["MAY LIN"]}
        dossiers = [{"person_id": 7, "name": None, "aliases": [],
                     "evidence": [{"id": "v0", "kind": "addressed_as",
                                   "name": "Miriam Okafor"}]}]
        texts = ['{"person_id": 7, "verdict": "assign", '
                 '"name": "MAY LIN", "character_name": null, '
                 '"evidence": ["v0"], "reason": "forced pick"}']
        self.assertEqual(
            self.adj.parse_verdicts(texts, cast, dossiers), [])

    def test_character_route_satisfies_agreement(self):
        # addressed as a single-word character -> its credited pair
        cast = {"pairs": [{"actor": "NEIL HARFORD",
                           "character": "Porter"}], "names": []}
        dossiers = [{"person_id": 8, "name": None, "aliases": [],
                     "evidence": [{"id": "v0", "kind": "addressed_as",
                                   "name": "Porter"}]}]
        texts = ['{"person_id": 8, "verdict": "assign", '
                 '"name": "NEIL HARFORD", "character_name": "Porter", '
                 '"evidence": ["v0"], "reason": "addressed as Porter"}']
        self.assertEqual(
            len(self.adj.parse_verdicts(texts, cast, dossiers)), 1)

    def test_never_overwrites_an_existing_name(self):
        vs = [{"person_id": 2, "name": "LENA VOSS",
               "character_name": None, "evidence_ids": ["v0"],
               "reason": "x"}]
        doc = {"persons": [{"person_id": 2, "name": "Someone Else",
                            "evidence": []}]}
        out = self.adj.apply_verdicts(doc, vs)
        self.assertEqual(out["persons"][0]["name"], "Someone Else")


class TestVocativesLLM(unittest.TestCase):
    """P2-b: the LLM replaces only the neighbor heuristic — answers are
    votes under the deterministic thresholds, and the firewall drops
    self-address and out-of-window speakers."""

    def setUp(self):
        from lib import vocatives_llm
        self.vl = vocatives_llm
        self.turns = [
            _turn("S0", 0, 4, "Porter, take the back door.", 0),
            _turn("S1", 5, 8, "On it.", 1),
            _turn("S0", 9, 12, "And Porter — quietly.", 2)]
        self.cues = [
            {"kind": "vocative", "name": "Porter", "sentence_id": 0},
            {"kind": "vocative", "name": "Porter", "sentence_id": 2}]

    def test_settled_names_are_never_reasked(self):
        resolved = [{"kind": "vocative_resolved", "name": "Porter",
                     "speaker_id": "S1"}]
        self.assertEqual(
            self.vl.build_windows(self.turns, self.cues, resolved), [])

    def test_windows_mark_the_site_and_list_speakers(self):
        ws = self.vl.build_windows(self.turns, self.cues, [])
        self.assertEqual(len(ws), 2)
        self.assertEqual(ws[0]["cue_speaker"], "S0")
        self.assertIn("S1", ws[0]["speakers"])
        self.assertTrue(any(ln["site"] for ln in ws[0]["lines"]))

    def _answers(self, ws, addressee):
        return ['\n'.join(json.dumps(
            {"window_id": w["id"], "addressee": addressee,
             "reason": "r"}) for w in ws)]

    def test_firewall_drops_self_address_and_foreign_speakers(self):
        ws = self.vl.build_windows(self.turns, self.cues, [])
        self.assertEqual(
            self.vl.parse_answers(self._answers(ws, "S0"), ws), [])
        self.assertEqual(
            self.vl.parse_answers(self._answers(ws, "S9"), ws), [])

    def test_votes_aggregate_under_the_deterministic_thresholds(self):
        ws = self.vl.build_windows(self.turns, self.cues, [])
        votes = self.vl.parse_answers(self._answers(ws, "S1"), ws)
        self.assertEqual(len(votes), 2)
        out = self.vl.aggregate(votes)
        self.assertEqual(len(out), 1)
        c = out[0]
        self.assertEqual((c["kind"], c["name"], c["speaker_id"],
                          c["resolver"]),
                         ("vocative_resolved", "Porter", "S1", "llm"))

    def test_single_vote_resolves_nothing(self):
        ws = self.vl.build_windows(self.turns, self.cues[:1], [])
        votes = self.vl.parse_answers(self._answers(ws, "S1"), ws)
        self.assertEqual(self.vl.aggregate(votes), [])

    def test_split_votes_without_lead_resolve_nothing(self):
        votes = [{"window_id": "w000", "name": "Porter",
                  "addressee": "S1", "addresser": "S0", "reason": ""},
                 {"window_id": "w001", "name": "Porter",
                  "addressee": "S1", "addresser": "S0", "reason": ""},
                 {"window_id": "w002", "name": "Porter",
                  "addressee": "S2", "addresser": "S0", "reason": ""},
                 {"window_id": "w003", "name": "Porter",
                  "addressee": "S2", "addresser": "S0", "reason": ""}]
        self.assertEqual(self.vl.aggregate(votes), [])

    def test_spelling_variants_cluster_into_one_vote_pool(self):
        # ASR wobble: one addressee, four 1-vote spellings — clustering
        # is what lets the threshold see them as one name
        votes = [{"window_id": f"w{i:03d}", "name": n,
                  "addressee": "S9", "addresser": "S0", "reason": ""}
                 for i, n in enumerate(
                     ["Porter", "Porder", "Detective Porter",
                      "Detective Porder"])]
        out = self.vl.aggregate(votes)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speaker_id"], "S9")
        self.assertEqual(out[0]["votes"], 4)
        self.assertIn("Porder", out[0]["variants"])

    def test_shared_honorific_never_fuses_two_names(self):
        # 'Detective Reyes' x1 + 'Detective Vega' x1 must NOT pool past
        # the 2-vote floor on the shared honorific (review, 2026-09-05)
        votes = [{"window_id": "w000", "name": "Detective Reyes",
                  "addressee": "S9", "addresser": "S0", "reason": ""},
                 {"window_id": "w001", "name": "Detective Vega",
                  "addressee": "S9", "addresser": "S0", "reason": ""}]
        self.assertEqual(self.vl.aggregate(votes), [])

    def test_duplicate_answer_lines_count_once_per_window(self):
        ws = self.vl.build_windows(self.turns, self.cues[:1], [])
        line = json.dumps({"window_id": ws[0]["id"], "addressee": "S1",
                           "reason": "r"})
        votes = self.vl.parse_answers([line + "\n" + line], ws)
        self.assertEqual(len(votes), 1)   # one window = one vote

    def test_extra_cues_ride_the_pipeline(self):
        from lib import bind_pipeline
        transcript = {**_transcript([
            _turn("S0", 0, 4, "Take the back door.", 0),
            _turn("S1", 5, 8, "On it.", 1)]), "language": "en"}
        extra = [{"kind": "vocative_resolved", "name": "Porter",
                  "speaker_id": "S1", "votes": 2,
                  "confidence": "high", "resolver": "llm"}]
        doc = bind_pipeline.bind(transcript, None, None, {},
                                 extra_cues=extra)
        names = {p.get("name") for p in doc["persons"]}
        self.assertIn("Porter", names)


class TestEnrichmentStates(unittest.TestCase):
    """The two held enrichment states + their consumer: every skip is a
    self-completed token with the reason as data, and a failed round 1
    degrades instead of blocking."""

    _PLAN = {"uuid": "u1", "llm_enrichment": True,
             "sibling_workflow_ids": {},
             "prompts": {"cast-adjudicate": "SYS-A",
                         "cast-critic2": "SYS-C",
                         "cast-vocatives": "SYS-V"},
             "inputs": {"speech_diarization_s3uri":
                        "s3://test-proxy/micro-workflow/"
                        "speech-diarization/d1"}}

    def _resolve(self, plan, transcript, submit=None):
        os.environ["BEDROCK_GATEWAY_ARN"] = "arn:fake"
        import importlib
        from states.resolve_addressees import lambda_function as ra
        importlib.reload(ra)
        sent, puts = [], {}
        with mock.patch.object(ra.s3io, "get_json", return_value=plan), \
             mock.patch.object(ra.s3io, "put_json",
                               side_effect=puts.__setitem__), \
             mock.patch.object(ra.sources, "load_transcript",
                               return_value=transcript), \
             mock.patch.object(ra.pb_common, "invoke_gateway_tool",
                               return_value=submit or {}) as inv, \
             mock.patch.object(ra._sfn, "send_task_success",
                               side_effect=lambda **k: sent.append(k)):
            out = ra.lambda_handler(
                {"plan_s3uri": "s3://test-proxy/micro-workflow/"
                               "cast-resolution/u1/_plan/plan.json",
                 "task_token": "tok"}, None)  # nosec B105
        os.environ.pop("BEDROCK_GATEWAY_ARN", None)
        return out, sent, puts, inv

    def test_flag_off_self_completes(self):
        out, sent, _, inv = self._resolve(
            {**self._PLAN, "llm_enrichment": False}, None)
        self.assertEqual(out["reason"], "llm_enrichment=false")
        self.assertEqual(len(sent), 1)
        self.assertFalse(inv.called)

    def test_no_transcript_self_completes(self):
        out, sent, _, _ = self._resolve(self._PLAN, None)
        self.assertEqual(out["reason"], "no_transcript")

    def test_unresolved_sites_submit_with_callback_and_park(self):
        turns = [_turn("S0", 0, 4, "Porter, take the back door.", 0),
                 _turn("S1", 5, 8, "On it.", 1)]
        out, sent, puts, inv = self._resolve(
            self._PLAN, {"language": "en", "turns": turns},
            submit={"job_id": "arn:j", "status": "RUNNING",
                    "output_s3_prefix": "s3://x/y"})
        self.assertTrue(out.get("parked"))
        self.assertEqual(sent, [])            # bedrock owns the token
        payload = inv.call_args[0][4]
        self.assertEqual(payload["callback_task_token"], "tok")
        self.assertFalse(payload["use_batch"])  # enrichment is sync
        self.assertTrue(any(k.endswith("_enrich/windows.json")
                            for k in puts))

    def test_adjudicate_skips_before_writing_context(self):
        # nothing to ask -> skip token, and NO misleading _enrich/
        # artifacts left behind (review finding)
        os.environ["BEDROCK_GATEWAY_ARN"] = "arn:fake"
        import importlib
        from states.adjudicate_cast import lambda_function as ac
        importlib.reload(ac)
        prefix = "s3://test-proxy/micro-workflow/cast-resolution/u1"
        docs = {f"{prefix}/_plan/plan.json": self._PLAN,
                f"{prefix}/persons.json": {"persons": []}}
        sent, puts = [], {}

        def get_json(u):
            if u not in docs:
                raise KeyError(u)
            return docs[u]
        with mock.patch.object(ac.s3io, "get_json",
                               side_effect=get_json), \
             mock.patch.object(ac.s3io, "put_json",
                               side_effect=puts.__setitem__), \
             mock.patch.object(ac.sources, "load_transcript",
                               return_value=None), \
             mock.patch.object(ac._sfn, "send_task_success",
                               side_effect=lambda **k: sent.append(k)):
            out = ac.lambda_handler(
                {"plan_s3uri": f"{prefix}/_plan/plan.json",
                 "task_token": "tok"}, None)  # nosec B105
        os.environ.pop("BEDROCK_GATEWAY_ARN", None)
        self.assertEqual(out["reason"], "nothing_to_adjudicate")
        self.assertEqual(len(sent), 1)        # skip token, not a crash
        self.assertEqual(puts, {})            # no orphaned context

    def test_bind_consumes_round1_addressees_as_cues(self):
        from states.bind_persons import lambda_function as bp
        windows = [{"id": "w000", "name": "Porter", "cue_speaker": "S0",
                    "speakers": ["S0", "S1"], "lines": []},
                   {"id": "w001", "name": "Porter", "cue_speaker": "S0",
                    "speakers": ["S0", "S1"], "lines": []}]
        texts = [json.dumps({"window_id": w, "addressee": "S1",
                             "reason": "r"}) for w in ("w000", "w001")]
        src = {}
        with mock.patch.object(bp.s3io, "get_json",
                               return_value=windows), \
             mock.patch.object(bp.s3io, "get_response_texts",
                               return_value=texts):
            cues = bp._addressee_cues(
                "s3://p/x", {"status": "SUCCEEDED",
                             "output_s3_prefix": "s3://x/r1"}, src)
        self.assertEqual([(c["name"], c["speaker_id"]) for c in cues],
                         [("Porter", "S1")])
        self.assertEqual(src["addressees"], "resolved:1")

    def test_bind_degrades_when_round1_failed(self):
        from states.bind_persons import lambda_function as bp
        src = {}
        cues = bp._addressee_cues("s3://p/x", {"status": "FAILED"}, src)
        self.assertEqual(cues, [])
        self.assertEqual(src["addressees"], "FAILED")

    def test_apply_is_a_noop_when_round2_skipped(self):
        from states.apply_enrichment import lambda_function as ae
        puts = {}
        with mock.patch.object(ae.s3io, "get_json",
                               return_value=self._PLAN), \
             mock.patch.object(ae.s3io, "put_json",
                               side_effect=puts.__setitem__):
            out = ae.lambda_handler(
                {"plan_s3uri": "s3://test-proxy/micro-workflow/"
                               "cast-resolution/u1/_plan/plan.json",
                 "addressees_result": {"addressees": "skipped"},
                 "adjudication_result": {"adjudication": "skipped",
                                         "reason": "nothing"}}, None)
        self.assertEqual(out["enrichment"], "skipped")
        self.assertEqual(puts, {})            # Bind's outputs untouched

    def test_apply_amends_bind_doc_and_recomputes_usage(self):
        # AMEND-only architecture: the doc comes from persons.json (the
        # provenance pin carries through) and usage recomputes from the
        # recorded pre_enrichment_total — idempotent by construction
        from states.apply_enrichment import lambda_function as ae
        prefix = "s3://test-proxy/micro-workflow/cast-resolution/u1"
        provenance = {"tool_version": "test",
                      "sibling_workflow_ids": {"speech_diarization": "d1"},
                      "inputs": {}, "generated_at": "t"}
        docs = {
            f"{prefix}/_plan/plan.json": self._PLAN,
            f"{prefix}/persons.json": {
                "persons": [{"person_id": 1, "name": None,
                             "aliases": [], "type": "voice_only",
                             "speaker_ids": ["S1"], "face_ids": [],
                             "name_mechanisms": [], "evidence": []}],
                "unresolved": {"speakers": [], "faces": []},
                "vetoes": [], "provenance": dict(provenance)},
            f"{prefix}/_enrich/context.json": {
                "dossiers": [{"person_id": 1, "name": None,
                              "aliases": [], "evidence": [
                                  {"id": "v0", "kind": "addressed_as",
                                   "name": "Mira"}]}],
                "cast": {"pairs": [{"actor": "LENA VOSS",
                                    "character": "Mira Santos"}],
                         "names": []}},
            f"{prefix}/usage_tracking.json": {
                "total_estimated_cost": 0.37, "cost_is_complete": True,
                "pre_enrichment_total": 0.37},
            "s3://x/r2/usage_tracking.json": {
                "total_estimated_cost": 0.03, "cost_is_complete": True},
        }
        puts = {}
        texts = ['{"person_id": 1, "verdict": "assign", '
                 '"name": "LENA VOSS", '
                 '"character_name": "Mira Santos", '
                 '"evidence": ["v0"], "reason": "addressed as Mira"}']
        event = {"plan_s3uri": f"{prefix}/_plan/plan.json",
                 "adjudication_result": {
                     "status": "SUCCEEDED",
                     "output_s3_prefix": "s3://x/r2"}}
        with mock.patch.object(ae.s3io, "get_json",
                               side_effect=lambda u: docs[u]), \
             mock.patch.object(ae.s3io, "put_json",
                               side_effect=puts.__setitem__), \
             mock.patch.object(ae.s3io, "get_response_texts",
                               return_value=texts):
            out = ae.lambda_handler(event, None)
            # idempotency: a retried apply re-reads its own outputs
            docs[f"{prefix}/persons.json"] = \
                puts[f"{prefix}/persons.json"]
            docs[f"{prefix}/usage_tracking.json"] = \
                puts[f"{prefix}/usage_tracking.json"]
            ae.lambda_handler(event, None)
        self.assertEqual(out["enrichment"], "applied")
        self.assertEqual(out["adjudicated"], 1)
        doc = puts[f"{prefix}/persons.json"]
        self.assertEqual(doc["persons"][0]["name"], "LENA VOSS")
        self.assertEqual(doc["persons"][0]["character_name"],
                         "Mira Santos")
        # provenance pin survived the amend (review finding)
        self.assertEqual(doc["provenance"]["sibling_workflow_ids"],
                         {"speech_diarization": "d1"})
        self.assertIn("enriched_at", doc["provenance"])
        usage = puts[f"{prefix}/usage_tracking.json"]
        self.assertAlmostEqual(usage["total_estimated_cost"], 0.40)
        self.assertTrue(usage["cost_is_complete"])   # 2nd run: same



class TestGatewayZipColdImport(unittest.TestCase):
    # the deployed gateway sees ONLY its zip members (caught live
    # 2026-10-06: a trimmed gateway zip ImportError'd on pb_common while
    # the suite passed, because tests put lib/ on the path) — cold-
    # import the dispatcher from a dir holding exactly what ships.
    # MEMBERS mirrors build-deploy-s3-dist.sh's gateway member list.
    MEMBERS = ("gateway_tool/lambda_function.py", "gateway_tool/workflow_identity.py",
               "gateway_tool/output_contract.json", "gateway_tool/input_examples.json",
               "lib/pb_common.py")

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
    unittest.main(verbosity=1)
