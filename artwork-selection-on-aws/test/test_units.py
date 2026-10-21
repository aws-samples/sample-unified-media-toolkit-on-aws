"""Unit tests for lib/ — invented fixtures only (never real-content
metadata). Run: python3 -m unittest discover -s test -p 'test_*.py'"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from lib import filter_rank                                 # noqa: E402
from lib.filter_rank import (eligibility_audit, eligible,    # noqa: E402
                             face_presence, rank_key, reject_reason,
                             select_candidates, tier, top_cast,
                             top_faces)
from lib.profiles import PROFILES, profile_for              # noqa: E402


def shot(sid, start, end, motion=0.1, entropy=5.0):
    return {"shot_id": sid, "start_time": start, "end_time": end,
            "frame_motion_mean": motion, "entropy_mean": entropy}


def doc(sid, camera="close_up", count="1", people=(), **over):
    d = {"shot_id": sid, "shot_kind": "normal", "camera_shot_type":
         camera, "people_count": count, "people": list(people),
         "texts_on_screen": [], "unreadable_text_present": False,
         "brands": [], "depicted": [], "description": f"shot {sid}"}
    d.update(over)
    return d


def lattice():
    shots = {i: shot(i, i * 100, i * 100 + 10) for i in range(1, 6)}
    docs = {
        1: doc(1, "close_up", "1", ["A"]),            # hero (lead)
        2: doc(2, "close_up", "1", ["B"]),            # not lead
        3: doc(3, "medium", "2", ["B", "C"]),         # two_shot
        4: doc(4, "aerial", "0"),                     # vista
        5: doc(5, "extreme_wide", None),              # vista (absent)
    }
    return shots, docs


CUT = 900.0  # spoiler cut used throughout; content ends later



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


class TestEligibility(unittest.TestCase):
    def test_disqualifiers(self):
        s = shot(1, 10, 20)
        self.assertTrue(eligible(s, doc(1), CUT))
        self.assertEqual(reject_reason(None, doc(1), CUT), "missing")
        self.assertEqual(reject_reason(s, None, CUT), "missing")
        self.assertEqual(
            reject_reason(s, doc(1, shot_kind="slate"), CUT),
            "shot_kind")
        self.assertEqual(
            reject_reason(s, doc(1, unreadable_text_present=True), CUT),
            "unreadable_text")
        self.assertEqual(
            reject_reason(s, doc(1, brands=["a brand"]), CUT), "brands")
        self.assertEqual(
            reject_reason(s, doc(1, depicted=["a poster"]), CUT),
            "depicted")
        self.assertEqual(reject_reason(shot(1, 10, 11.0), doc(1), CUT),
                         "short")
        self.assertEqual(reject_reason(shot(1, 901, 910), doc(1), CUT),
                         "spoiler")

    def test_graphics_overlay_is_footage_with_a_tier(self):
        # news support (2026-10-05): graphics_overlay is real footage
        # wearing channel furniture — eligible, but tier 1; on-screen
        # text no longer disqualifies, it tiers
        s = shot(1, 10, 20)
        self.assertTrue(eligible(
            s, doc(1, shot_kind="graphics_overlay"), CUT))
        self.assertTrue(eligible(s, doc(1, texts_on_screen=["LIVE"]),
                                 CUT))
        self.assertEqual(tier(doc(1)), 0)
        self.assertEqual(tier(doc(1, texts_on_screen=["LIVE"])), 1)
        self.assertEqual(tier(doc(1, shot_kind="graphics_overlay")), 1)
        # the other disqualifiers hold for BOTH tiers
        self.assertEqual(
            reject_reason(s, doc(1, shot_kind="graphics_overlay",
                                 brands=["bug"]), CUT), "brands")

    def test_eligibility_audit(self):
        shots = {1: shot(1, 10, 20), 2: shot(2, 30, 40),
                 3: shot(3, 50, 60), 4: shot(4, 70, 71)}
        docs = {1: doc(1),
                2: doc(2, shot_kind="graphics_overlay"),
                3: doc(3, brands=["bug"]),
                4: doc(4)}
        self.assertEqual(
            eligibility_audit(shots, docs, CUT),
            {"eligible_clean": 1, "eligible_overlaid": 1,
             "brands": 1, "short": 1})

    def test_excluded_spans(self):
        s = shot(1, 100, 110)
        self.assertTrue(eligible(s, doc(1), CUT, [(0, 60)]))
        self.assertFalse(eligible(s, doc(1), CUT, [(105, 200)]))
        self.assertFalse(eligible(s, doc(1), CUT, [(0, 101)]))


class TestRanking(unittest.TestCase):
    def test_clean_tier_dominates(self):
        # a clean plate beats ANY furniture frame, however long and
        # still; within the furniture tier fewer strings win
        clean = rank_key(shot(1, 0, 2), doc(1))
        furn = rank_key(shot(2, 0, 60, motion=0.0),
                        doc(2, shot_kind="graphics_overlay"))
        self.assertLess(clean, furn)
        few = rank_key(shot(3, 0, 5),
                       doc(3, texts_on_screen=["LIVE"]))
        many = rank_key(shot(4, 0, 5),
                        doc(4, texts_on_screen=["LIVE", "BANK", "9:12"]))
        self.assertLess(few, many)

    def test_ordering(self):
        self.assertLess(rank_key(shot(1, 0, 6)),
                        rank_key(shot(2, 0, 3)))         # longer first
        self.assertEqual(rank_key(shot(1, 0, 9))[0],
                         rank_key(shot(2, 0, 30))[0])    # cap saturates
        self.assertLess(rank_key(shot(1, 0, 5, motion=0.01)),
                        rank_key(shot(2, 0, 5, motion=0.9)))
        self.assertLess(rank_key(shot(1, 0, 5, entropy=7.0)),
                        rank_key(shot(2, 0, 5, entropy=2.0)))


class TestCast(unittest.TestCase):
    def test_top_cast(self):
        persons = [{"name": "B", "screen_sec": 50},
                   {"name": "A", "screen_sec": 900},
                   {"screen_sec": 999},          # unnamed: skipped
                   {"name": "C"}]                # no screen time: last
        self.assertEqual(top_cast(persons), ["A", "B", "C"])
        self.assertEqual(top_cast([]), [])


class TestSelection(unittest.TestCase):
    def test_archetype_selection(self):
        shots, docs = lattice()
        slates = select_candidates(shots, docs, ["A", "B"],
                                   PROFILES["default"], CUT)
        # hero spans the top casts, lead first (user 2026-10-06)
        self.assertEqual([c["shot_id"] for c in slates["hero_closeup"]],
                         [1, 2])
        self.assertEqual([c["shot_id"] for c in slates["two_shot"]], [3])
        self.assertEqual(
            {c["shot_id"] for c in slates["establishing_vista"]}, {4, 5})
        c = slates["hero_closeup"][0]
        self.assertEqual(c["t"], 105.0)
        self.assertEqual(c["people"], ["A"])

    def test_hero_round_robin_across_top_casts(self):
        # 3 close-ups of lead A, 1 of co-lead B: the slate interleaves
        # (A, B, A, A) — the co-lead is never starved by the lead's
        # abundance; a 4th cast name beyond HERO_CAST_N never matches
        shots = {i: shot(i, i * 100, i * 100 + 10) for i in range(1, 6)}
        docs = {1: doc(1, "close_up", "1", ["A"]),
                2: doc(2, "close_up", "1", ["A"]),
                3: doc(3, "close_up", "1", ["A"]),
                4: doc(4, "close_up", "1", ["B"]),
                5: doc(5, "close_up", "1", ["D"])}
        slates = select_candidates(shots, docs, ["A", "B", "C", "D"],
                                   {"hero_closeup": 4}, CUT)
        got = [c["people"][0] for c in slates["hero_closeup"]]
        self.assertEqual(got, ["A", "B", "A", "A"])

    def test_no_cast_means_no_person_archetypes(self):
        shots, docs = lattice()
        slates = select_candidates(shots, docs, [],
                                   PROFILES["default"], CUT)
        self.assertEqual(slates["hero_closeup"], [])
        self.assertEqual(slates["two_shot"], [])
        self.assertTrue(slates["establishing_vista"])

    def test_cross_archetype_dedup_and_counts(self):
        shots = {i: shot(i, i * 100, i * 100 + 10) for i in range(1, 8)}
        docs = {i: doc(i, "aerial", "0") for i in range(1, 8)}
        filter_rank.ARCHETYPES["vista_b"] = dict(
            filter_rank.ARCHETYPES["establishing_vista"])
        try:
            slates = select_candidates(
                shots, docs, [],
                {"establishing_vista": 4, "vista_b": 4}, CUT)
        finally:
            del filter_rank.ARCHETYPES["vista_b"]
        a = {c["shot_id"] for c in slates["establishing_vista"]}
        b = {c["shot_id"] for c in slates["vista_b"]}
        self.assertEqual(len(a), 4)
        self.assertFalse(a & b)     # elected once, never offered again
        self.assertEqual(len(b), 3)  # only 7 eligible shots total


def fr_person(pid, spans_sec, name=None):
    return {"person_id": pid,
            "named": bool(name), "name": name or f"Person_{pid}",
            "total_screen_time_sec": sum(b - a for a, b in spans_sec),
            "appearances": [{"begin_ms": int(a * 1000),
                             "end_ms": int(b * 1000)}
                            for a, b in spans_sec]}


class TestFaceFallback(unittest.TestCase):
    def test_top_faces_ranking_and_labels(self):
        fr = {"people": [
            fr_person(1, [(0, 50)], name="A"),
            fr_person(2, [(0, 900)]),               # unnamed, dominant
            fr_person(3, []),                       # no spans: dropped
        ]}
        labels, spans = top_faces(fr)
        # placeholder Person_* names are NOT names -> anonymous label
        self.assertEqual(labels, ["person_2", "A"])
        self.assertEqual(spans["A"], [(0.0, 50.0)])

    def test_face_presence_temporal_join(self):
        shots = {1: shot(1, 0, 10), 2: shot(2, 100, 110)}
        presence = face_presence(shots, {"A": [(5, 8)],
                                         "B": [(9.5, 105)],
                                         "C": [(200, 300)]})
        self.assertEqual(presence, {1: {"A", "B"}, 2: {"B"}})

    def test_selection_via_presence_ignores_metadata_people(self):
        shots, docs = lattice()
        # metadata people say A is in shot 1 — but presence says the
        # face-track lead is only in shot 2
        presence = {2: {"person_9"}}
        slates = select_candidates(shots, docs, ["person_9"],
                                   PROFILES["default"], CUT,
                                   presence=presence)
        self.assertEqual([c["shot_id"] for c in slates["hero_closeup"]],
                         [2])
        self.assertEqual(slates["two_shot"], [])


class TestProfiles(unittest.TestCase):
    def test_profile_for(self):
        self.assertIs(profile_for(None), PROFILES["default"])
        self.assertIs(profile_for("no-such-genre"), PROFILES["default"])
        self.assertIs(profile_for(" DEFAULT "), PROFILES["default"])


if __name__ == "__main__":
    unittest.main()
