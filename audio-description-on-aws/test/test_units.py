"""
audio-description unit tests over the PURE lib layer. Fixtures are
SYNTHETIC (generalize-never-content-specific); each measured incident a
rule came from is named where a fixture encodes it.

Run: python3 -m unittest discover -s test -p 'test_*.py'
"""
from __future__ import annotations

import os
import sys
import json
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
# gateway modules import as top-level names (their zip is flat) —
# gateway_tool goes FIRST so `import lambda_function` resolves there
sys.path.insert(0, os.path.join(_ROOT, "gateway_tool"))
# the gateway's flat zip ships lib/ad_common.py beside it (input-uri
# contract at the door) — its runtime resolves `ad_common` top-level
sys.path.append(os.path.join(_ROOT, "lib"))

os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("STATE_MACHINE_ARN",
                      "arn:aws:states:eu-west-1:000000000000:stateMachine:t")

from lib import ad_common as C      # noqa: E402
from lib import coverage            # noqa: E402
from lib import fitting             # noqa: E402
from lib import ledger              # noqa: E402
from lib import oracles             # noqa: E402
from lib import windows             # noqa: E402


def _transcript(words, duration=100.0):
    return {"words": [{"word": w, "start": a, "end": b}
                      for w, a, b in words],
            "sentences": [], "audio_duration_sec": duration}



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


class TestWordSpans(unittest.TestCase):
    def test_words_merge_within_gap(self):
        t = _transcript([("a", 1.0, 1.5), ("b", 1.7, 2.0),
                         ("c", 5.0, 5.5)])
        self.assertEqual(windows.word_spans(t), [[1.0, 2.0], [5.0, 5.5]])


class TestSungVocalPolicy(unittest.TestCase):
    def test_untranscribed_speech_stem_is_sung_vocals(self):
        """The mechanical predicate: speech-stem activity with NO
        transcript words = untranscribed singing (or walla)."""
        t = _transcript([("word", 10.0, 11.0)])
        stem = {"activity": {"speech": [[10_000, 11_000],
                                        [40_000, 50_000]]}}
        self.assertEqual(windows.sung_vocal_spans(stem, t),
                         [[40.0, 50.0]])

    def test_avoid_policy_closes_windows_over_singing(self):
        """Default narrate keeps the gap open; avoid treats the sung
        span as speech (user decision 2026-08-31: configurable,
        default narrate)."""
        t = _transcript([("a", 1.0, 2.0), ("b", 58.0, 59.0)],
                        duration=60.0)
        stem = {"activity": {"speech": [[20_000, 40_000]],
                             "music": [[20_000, 40_000]]}}
        ev = {"transcript": t, "scenes": [
                  {"scene_id": 0, "start_time": 0.0, "end_time": 60.0,
                   "setting": "bar", "people": []}],
              "shots": [{"shot_id": 0, "start_time": 0.0,
                         "end_time": 60.0}],
              "people": [], "shot_docs": {}, "stem_activity": stem}
        narrate = windows.build_windows(ev, {}, "narrate")["windows"]
        avoid = windows.build_windows(ev, {}, "avoid")["windows"]

        def covers(wins, t0, t1):
            return any(w["start"] < t1 and w["end"] > t0 for w in wins)

        self.assertTrue(covers(narrate, 25.0, 35.0))
        self.assertFalse(covers(avoid, 25.0, 35.0))


class TestEntryLadder(unittest.TestCase):
    def _ev(self):
        # the Meridian office->car incident, synthesized: a 0.8s window
        # at the cut and a 4s window 6s later — the duty must go to the
        # funded window, the stub becomes its PRELUDE
        t = _transcript([("a", 0.0, 30.0), ("b", 32.2, 32.5),
                         ("d", 41.0, 60.0)], duration=80.0)
        scenes = [{"scene_id": 0, "start_time": 0.0, "end_time": 30.2,
                   "setting": "office", "people": ["A"]},
                  {"scene_id": 1, "start_time": 30.2, "end_time": 80.0,
                   "setting": "car", "people": ["B"],
                   "boundary_reason": "location_change"}]
        shots = [{"shot_id": 0, "start_time": 0.0, "end_time": 30.2},
                 {"shot_id": 1, "start_time": 30.2, "end_time": 80.0}]
        return {"transcript": t, "scenes": scenes, "shots": shots,
                "people": [], "shot_docs": {}, "stem_activity": None}

    def test_duty_goes_to_the_cheapest_sufficient_window(self):
        wins = windows.build_windows(self._ev(), {})["windows"]
        entry = [w for w in wins if w["scene_entry"]
                 and w["entry_for_scene"] == 1]
        self.assertEqual(len(entry), 1)
        self.assertGreaterEqual(entry[0]["honest_words"],
                                C.MIN_GROUND_WORDS)
        preludes = [w for w in wins
                    if w.get("entry_prelude_for_scene") == 1]
        self.assertEqual(len(preludes), 1)
        self.assertLess(preludes[0]["start"], entry[0]["start"])


class TestOrientation(unittest.TestCase):
    def test_dialogue_bridging_zeroes_sufficiency_and_raises_break(self):
        t = {"words": [{"word": "x", "start": 29.0, "end": 31.0}],
             "sentences": [], "audio_duration_sec": 60.0}
        scenes = [{"scene_id": 0, "start_time": 0.0, "end_time": 30.0,
                   "setting": "office", "people": ["A"]},
                  {"scene_id": 1, "start_time": 30.0, "end_time": 60.0,
                   "setting": "car", "people": ["B"]}]
        spans = windows.word_spans(t)
        o = windows.orientation(scenes, t, spans, [], [])
        self.assertTrue(o[1]["dialogue_bridged"])
        self.assertEqual(o[1]["audio_sufficiency"], 0.0)

    def test_montage_run_folds_into_its_head(self):
        t = _transcript([], duration=100.0)
        scenes = [{"scene_id": i, "start_time": i * 3.0,
                   "end_time": (i + 1) * 3.0, "setting": f"s{i}",
                   "people": []} for i in range(4)]
        scenes.append({"scene_id": 4, "start_time": 12.0,
                       "end_time": 100.0, "setting": "long",
                       "people": []})
        o = windows.orientation(scenes, t, [], [], [])
        self.assertEqual(o[0].get("montage_scenes"), 4)
        for sid in (1, 2, 3):
            self.assertEqual(o[sid].get("montage_member_of"), 0)


class TestQuotedToks(unittest.TestCase):
    def test_both_directions(self):
        """The measured pair: an apostrophe INSIDE a word opens nothing,
        while the colon form ('A title reads: Meridian') must parse."""
        self.assertIn("meridian",
                      oracles.quoted_toks("A title reads: Meridian"))
        self.assertFalse(oracles.quoted_toks(
            "Warner's and Broadway Hollywood signs pass in traffic"))

    def test_shot_texts_unions_environment_text(self):
        """drama_b w33: diegetic signage lives in named_entities with
        evidence environment_text, not texts_on_screen."""
        d = {"texts_on_screen": ["THE END"],
             "named_entities": [{"name": "CHEZ JAY",
                                 "evidence": "environment_text"},
                                {"name": "LA", "evidence": "dialogue"}]}
        self.assertEqual(oracles.shot_texts(d), ["THE END", "CHEZ JAY"])


class TestFalseTextClaims(unittest.TestCase):
    def test_unsupported_quote_is_flagged_and_supported_is_not(self):
        ws = {0: {"window_id": 0, "shot_ids": [0]},
              1: {"window_id": 1, "shot_ids": [1]}}
        docs = {0: {"texts_on_screen": []},
                1: {"texts_on_screen": ["MERIDIAN"]}}
        tm = {0: (0.0, 10.0), 1: (20.0, 30.0)}
        cues = {"0": 'A title reads: "Meridian".',
                "1": 'A title reads: "Meridian".'}
        placed = [{"window_id": 0, "at": 2.0, "dur": 2.0},
                  {"window_id": 1, "at": 21.0, "dur": 2.0}]
        bad = oracles.false_text_claims(ws, cues, placed, docs, tm)
        self.assertIn(0, bad)
        self.assertNotIn(1, bad)
        # the true location travels with the finding
        self.assertEqual(bad[0]["text_lives_at"][0][0], 20.0)


class TestDescriptorHits(unittest.TestCase):
    def test_generic_person_noun_identifies_nobody(self):
        """drama_b: 'A man...' must never rewrite to a name on the
        strength of the bare noun."""
        cmap = [{"face_key": "Some Actor", "character_name": "Billy",
                 "descriptor": "a middle-aged man",
                 "speakable_from_sec": 0.0}]
        ws = {0: {"window_id": 0, "shot_ids": [0],
                  "people_on_screen": ["Some Actor"]}}
        cues = {"0": "A man raises his arms."}
        placed = [{"window_id": 0, "at": 5.0, "dur": 2.0}]
        hits = oracles.descriptor_hits(cmap, ws, cues, placed,
                                       {0: {"description": ""}})
        self.assertEqual(hits, {})

    def test_name_already_in_text_means_somebody_else(self):
        """'Billy meeting Billy': any token of the name already in the
        cue vetoes the join."""
        cmap = [{"face_key": "Some Actor", "character_name": "Billy",
                 "descriptor": "a weathered lawyer",
                 "speakable_from_sec": 0.0}]
        ws = {0: {"window_id": 0, "shot_ids": [0],
                  "people_on_screen": ["Some Actor"]}}
        cues = {"0": "Billy shakes hands with the weathered lawyer."}
        placed = [{"window_id": 0, "at": 5.0, "dur": 2.0}]
        hits = oracles.descriptor_hits(cmap, ws, cues, placed,
                                       {0: {"description": ""}})
        self.assertEqual(hits, {})


class TestRedundancy(unittest.TestCase):
    def test_bare_place_echo_is_detected(self):
        cues = {"0": "The dome.",
                "1": "The dome rehearsal space. Thomas addresses the "
                     "orchestra."}
        placed = [{"window_id": 0, "at": 10.0, "dur": 1.0},
                  {"window_id": 1, "at": 13.0, "dur": 3.0}]
        ws = {0: {"scene_id": 1}, 1: {"scene_id": 1}}
        pairs = oracles.redundant_pairs(cues, placed, ws, [])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["echo"], 0)


class TestSafeDeletions(unittest.TestCase):
    def test_deletion_that_loses_a_beat_is_refused(self):
        """The rearview incident: a duty-flag guard passed the deletion;
        the coverage oracle is the guard that caught it."""
        placed = [{"window_id": 0, "at": 100.0, "dur": 2.0},
                  {"window_id": 1, "at": 130.0, "dur": 2.0}]
        cues = {"0": "A woman in the mirror.", "1": "Rain falls."}
        notes = {"beats": [{"sec": 101.0, "what": "the reveal"}],
                 "reveals": []}
        ok, refused = coverage.safe_deletions(placed, cues, [0], notes,
                                              [])
        self.assertEqual(ok, [])
        self.assertEqual(refused[0][0], 0)

    def test_neutral_deletion_passes(self):
        placed = [{"window_id": 0, "at": 100.0, "dur": 2.0},
                  {"window_id": 1, "at": 101.0, "dur": 2.0}]
        cues = {"0": "A road.", "1": "A rain-streaked road at night."}
        notes = {"beats": [{"sec": 101.0, "what": "arrival"}],
                 "reveals": []}
        ok, refused = coverage.safe_deletions(placed, cues, [0], notes,
                                              [])
        self.assertEqual(ok, [0])


class TestFitting(unittest.TestCase):
    def test_tts_normalize_years_and_caps(self):
        self.assertEqual(fitting.tts_normalize("in 1947 FBI agents"),
                         "in nineteen forty-seven Fbi agents")

    def test_fit_is_arithmetic_over_tts_duration(self):
        """No audio is opened: dur = raw_dur / tempo, tempo capped."""
        w = {"window_id": 0, "start": 10.0, "end": 14.0, "anchor": 10.0,
             "seconds": 4.0}
        spans = [[15.0, 20.0]]     # true onset at 15.0 -> avail 4.92
        p, r = fitting.fit_one(0, 5.0, w, spans)      # needs x1.02
        self.assertIsNone(r)
        self.assertAlmostEqual(p["dur"], 5.0 / p["tempo"], places=2)
        self.assertLessEqual(p["tempo"], fitting.C.TEMPO_CAP)
        p2, r2 = fitting.fit_one(0, 8.0, w, spans)    # needs x1.63
        self.assertIsNone(p2)
        self.assertGreater(r2["needed_tempo"], fitting.C.TEMPO_CAP)

    def test_refit_respects_the_true_onset(self):
        fix = fitting.refit(0, 3.0, 10.0, [[12.5, 20.0]])
        self.assertIsNotNone(fix)
        self.assertLessEqual(10.0 + fix["dur"], 12.5 - 0.05)
        self.assertIsNone(fitting.refit(0, 6.0, 10.0, [[12.5, 20.0]]))

    def test_collide_drops_a_cue_pushed_into_speech(self):
        spans = [[10.0, 20.0]]
        placed = [{"window_id": 0, "at": 5.0, "dur": 4.5},
                  {"window_id": 1, "at": 9.7, "dur": 3.0}]
        out, drops = fitting.collide(placed, spans)
        self.assertEqual([p["window_id"] for p in out], [0])
        self.assertEqual(drops[0]["window_id"], 1)
        self.assertEqual(drops[0]["kind"], "speech_overlap")

    def test_collateral_names_the_harmed_neighbour(self):
        spans = [[30.0, 40.0]]
        placed = [{"window_id": 0, "at": 5.0, "dur": 2.0},
                  {"window_id": 1, "at": 7.5, "dur": 2.0}]
        harm = fitting.collateral(0, {"at": 5.0, "dur": 4.0}, placed,
                                  spans)
        self.assertIn("w1", harm or "")
        self.assertIsNone(fitting.collateral(
            0, {"at": 5.0, "dur": 2.1}, placed, spans))


class TestLedger(unittest.TestCase):
    def test_not_in_mix_invariant(self):
        """A row that is not in the mix ALWAYS carries a violation
        (measured: a clean 'ship' verdict on a cue nobody will hear)."""
        ws = {7: {"window_id": 7, "start": 10.0, "end": 14.0,
                  "anchor": 10.0, "scene_entry": True,
                  "entry_for_scene": 2, "word_budget": 8,
                  "entry_necessity": 0.7}}
        rows = ledger.build_ledger(
            ws, {"7": "Coastal highway."}, [],
            [{"scene_id": 2, "at": 10.0, "necessity": 0.7,
              "factors": {}}],
            {"beats": []}, [], [])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["violations"])
        self.assertEqual(rows[0]["violations"][0]["kind"], "unplaced")

    def test_severity_is_computed_from_the_measurement(self):
        self.assertEqual(ledger.severity("over_budget",
                                         {"needed_tempo": 1.3}), "minor")
        self.assertEqual(ledger.severity("over_budget",
                                         {"needed_tempo": 1.7}), "major")
        self.assertEqual(ledger.severity(
            "speech_overlap", {"starts_inside_speech": True}), "major")


class TestReviewRegressions(unittest.TestCase):
    """Each of these encodes a CONFIRMED 2026-08-31 review finding."""

    def test_artifact_key_is_segment_based_not_suffix_based(self):
        """A run root ending in '0' must still gain the '/0' folder
        (~1/16 of hex ids ended in '0' and silently lost credits)."""
        from lib import evidence
        self.assertEqual(
            evidence.artifact_key("fi/a1b2c0", "0", "credits.json"),
            "fi/a1b2c0/0/credits.json")
        self.assertEqual(
            evidence.artifact_key("fi/run/0/credits.json",
                                  "0", "credits.json"),
            "fi/run/0/credits.json")
        self.assertEqual(
            evidence.artifact_key("run/transcripts", "transcripts",
                                  "final_transcript.json"),
            "run/transcripts/final_transcript.json")

    def test_coverage_guard_survives_int_keyed_cues(self):
        """The dead-armed guard: int-keyed cues made every lookup miss
        and every deletion 'proved' neutral."""
        placed = [{"window_id": 0, "at": 100.0, "dur": 2.0}]
        cues_int = {0: "A woman in the mirror."}
        notes = {"beats": [{"sec": 101.0, "what": "the reveal"}],
                 "reveals": []}
        ok, refused = coverage.safe_deletions(placed, cues_int, [0],
                                              notes, [])
        self.assertEqual(ok, [])
        self.assertEqual(refused[0][0], 0)

    def test_parse_jsonl_cues_survives_model_deviations(self):
        from lib import submit
        out = "\n".join([
            '{"window_id": null, "text": "x"}',
            '[{"window_id": 1, "text": "arr"}]',
            '{"window_id": "w3", "text": "x"}',
            '{"window_id": 4, "text": null}',
            '{"window_id": 5, "text": "good"}'])
        cues, bad = submit.parse_jsonl_cues(out)
        self.assertEqual(cues, {4: "", 5: "good"})
        self.assertEqual(len(bad), 3)

    def test_prune_and_shards(self):
        from lib import submit
        kept, extras = submit.prune_cues({1: "a", 999: "x"}, {1, 2})
        self.assertEqual((kept, extras), ({1: "a"}, [999]))
        shards = submit.tts_shards(list(range(200)))
        self.assertEqual([len(s) for s in shards],
                         [C.TTS_SHARD_SIZE, C.TTS_SHARD_SIZE,
                          200 - 2 * C.TTS_SHARD_SIZE])

    def test_year_2000_is_two_thousand(self):
        self.assertEqual(fitting.tts_normalize("in 2000 and in 1900"),
                         "in two thousand and in nineteen hundred")

    def test_tts_submit_refuses_a_precallback_deployment(self):
        """A pre-alpha.20260831a tts silently IGNORES callback_task_token —
        the wait state would sleep to its 26h timeout. output_s3_prefix
        shipped with callbacks, so its absence is the mechanical probe."""
        from lib import submit
        old = C.invoke_gateway_tool
        C.invoke_gateway_tool = lambda *a, **k: {
            "status": "RUNNING", "job_id": "arn:x", "workflow_id": "w"}
        try:
            with self.assertRaises(RuntimeError) as ctx:
                submit.tts_submit(["hi"], "token", {
                    "workflow_id": "w", "voice_id": "v",
                    "output_prefix": "p"}, "t")
            self.assertIn("predates callback support", str(ctx.exception))
        finally:
            C.invoke_gateway_tool = old

    def test_fit_never_sprawls_past_a_cut_bounded_window(self):
        """A pre-cut window with the next speech 25s away: the take must
        cap at the window end (+PAD), never fit at tempo 1.0 across the
        cut."""
        w = {"window_id": 0, "start": 10.0, "end": 13.0, "anchor": 10.0,
             "seconds": 3.0}
        spans = [[38.0, 40.0]]
        p, r = fitting.fit_one(0, 9.0, w, spans)   # 9s take, 3.3s legal
        self.assertIsNone(p)
        self.assertGreater(r["needed_tempo"], C.TEMPO_CAP)


class TestTtsContract(unittest.TestCase):
    """The AD side of the tts push-completion contract (review
    2026-08-31 fixes)."""

    def test_push_completion_probe_refuses_a_precallback_deployment(self):
        """ValidateAndPlan probes the CAPABILITY before any bedrock spend;
        a proxy-field check two states later was the only guard before."""
        from lib import submit
        old = C.invoke_gateway_tool
        try:
            C.invoke_gateway_tool = lambda *a, **k: {"version": "old"}
            with self.assertRaises(RuntimeError) as ctx:
                submit.assert_tts_push_completion()
            self.assertIn("predates callback support", str(ctx.exception))
            C.invoke_gateway_tool = lambda *a, **k: {
                "version": "alpha.20260831a", "push_completion": True}
            submit.assert_tts_push_completion()   # must not raise
        finally:
            C.invoke_gateway_tool = old

    def test_a_failed_tts_run_raises_by_name_not_nosuchkey(self):
        """The waker persists a failure doc to the SAME outputs.json
        pointer; the reader must surface tts's own verdict instead of
        crashing on a missing 'outputs' key."""
        from lib import submit
        old = C.read_json
        try:
            C.read_json = lambda b, k: {
                "status": "FAILED", "error": "GPU fell over",
                "execution_arn": "arn:e"}
            with self.assertRaises(RuntimeError) as ctx:
                submit.tts_outputs({"output_s3_prefix": "s3://b/p",
                                    "count": 3})
            self.assertIn("did not succeed", str(ctx.exception))
            self.assertIn("GPU fell over", str(ctx.exception))
        finally:
            C.read_json = old

    def test_submit_register_is_row_per_file_and_replay_safe(self):
        """The fan-out Map registers shards CONCURRENTLY, so rows never
        share a document (a single-list read-modify-write loses whichever
        row read before another's write) — and _register must never READ
        at all, which is also what makes a replay free (the old append
        double-counted usage, review 2026-08-31)."""
        from lib import submit
        store, reads = {}, []
        old_read, old_write, old_list = (C.read_json_if_exists,
                                         C.write_json, C.list_keys)
        old_read_json = C.read_json
        try:
            C.read_json_if_exists = \
                lambda b, k: reads.append(k) or store.get(k)
            C.read_json = lambda b, k: store[k]
            C.write_json = lambda b, k, v: store.__setitem__(k, v)
            C.list_keys = lambda b, p: sorted(
                k for k in store if k.startswith(p))
            plan = {"output_prefix": "pfx"}
            rec = {"workflow_id": "w", "job_id": "j",
                   "output_s3_prefix": "s3://b/p", "count": 2}
            submit._register(plan, "tts", "main-s00", rec)
            submit._register(plan, "tts", "main-s00", rec)   # the replay
            submit._register(plan, "tts", "main-s01", rec)
            self.assertEqual(sorted(store),
                             ["pfx/_plan/submits/tts-main-s00.json",
                              "pfx/_plan/submits/tts-main-s01.json"])
            self.assertEqual(reads, [],
                             "_register read S3 — a concurrent-writer "
                             "register may never read-modify-write")
            rows = submit.read_submits("pfx")
            self.assertEqual([r["tag"] for r in rows],
                             ["main-s00", "main-s01"])
            self.assertEqual(
                submit.read_submit("pfx", "tts", "main-s01")["kind"],
                "tts")
        finally:
            C.read_json_if_exists, C.write_json, C.list_keys = \
                old_read, old_write, old_list
            C.read_json = old_read_json


class TestTtsFanout(unittest.TestCase):
    """The 2026-09-02 parallel-shard contract: a loop resolves its own
    token with {tts_shard_tags}, the Map's iterations submit one shard
    each against their own token, the re-entry fits from the register."""

    def _synth_env(self, store):
        from states.synth_fit import lambda_function as synth_fit
        olds = (C.read_json, C.read_json_if_exists, C.write_json)
        C.read_json = lambda b, k: store[k]
        C.read_json_if_exists = lambda b, k: store.get(k)
        C.write_json = lambda b, k, v: store.__setitem__(k, v)
        return synth_fit, olds

    def _restore(self, olds):
        C.read_json, C.read_json_if_exists, C.write_json = olds

    def test_synth_start_checkpoints_specs_then_fans_out(self):
        """The payload carries only TAGS; every shard's order+seed lives
        in the checkpoint the Map iterations re-read (signals, never a
        data channel)."""
        from lib import loop_state
        store = {"s3plan": {"output_prefix": "pfx"},
                 "pfx/cues.json": {str(k): f"cue {k}" for k in range(100)}}
        synth_fit, olds = self._synth_env(store)
        sent = {}
        old_fan = loop_state.request_fanout
        old_parse = C.parse_s3_uri
        try:
            loop_state.request_fanout = lambda tok, tags: (
                sent.__setitem__("tags", sorted(tags))
                or {"waiting": f"fanout:{len(tags)}"})
            C.parse_s3_uri = lambda u: ("b", "s3plan")
            synth_fit.lambda_handler(
                {"plan_s3uri": "s3://b/s3plan", "task_token": "T"}, None)  # nosec B105
            self.assertEqual(sent["tags"], ["main-s00", "main-s01"])
            ck = store["pfx/_plan/synth_state.json"]
            self.assertEqual(ck["phase"], "tts")
            self.assertEqual(sorted(ck["shard_specs"]),
                             ["main-s00", "main-s01"])
            self.assertEqual(len(ck["shard_specs"]["main-s00"]["order"]),
                             C.TTS_SHARD_SIZE)
            self.assertEqual(len(ck["shard_specs"]["main-s01"]["order"]),
                             100 - C.TTS_SHARD_SIZE)
            self.assertEqual(ck["shard_specs"]["main-s00"]["seed"], 1234)
        finally:
            loop_state.request_fanout = old_fan
            C.parse_s3_uri = old_parse
            self._restore(olds)

    def test_shard_iteration_reads_spec_and_submits_its_own_token(self):
        """One Map iteration = one shard against the ITERATION's token —
        a single token cannot wait for N callbacks (the first consumes
        it; the rest are TOKEN_GONE and the state sleeps to its
        deadman). Repair shards read texts from the checkpoint, synth
        shards from cues.json; seed travels in the spec."""
        from lib import submit as submit_mod
        from states.submit_tts_shard import lambda_function as submit_tts_shard
        store = {
            "s3plan": {"output_prefix": "pfx"},
            "pfx/cues.json": {"7": " seven ", "9": "nine"},
            "pfx/_plan/synth_state.json": {
                "phase": "tts",
                "shard_specs": {"reask1-s00": {"order": [7, 9],
                                               "seed": 4321}}},
            "pfx/_plan/repair_state.json": {
                "phase": "tts",
                "texts": {"7": "fixed seven"},
                "shard_specs": {"repair08-x-s00": {"order": [7],
                                                   "seed": 1234}}},
        }
        olds = (C.read_json, C.parse_s3_uri)
        calls = []
        old_tts = submit_mod.tts_submit
        try:
            C.read_json = lambda b, k: store[k]
            C.parse_s3_uri = lambda u: ("b", "s3plan")
            submit_mod.tts_submit = \
                lambda texts, token, plan, tag, seed=1234, attempt=0: \
                calls.append({"texts": texts, "token": token, "tag": tag,
                              "seed": seed}) or {}
            submit_tts_shard.lambda_handler(
                {"plan_s3uri": "s3://b/s3plan", "loop": "synth",
                 "tag": "reask1-s00", "task_token": "TOK-A"}, None)  # nosec B105
            submit_tts_shard.lambda_handler(
                {"plan_s3uri": "s3://b/s3plan", "loop": "repair",
                 "tag": "repair08-x-s00", "task_token": "TOK-B"}, None)  # nosec B105
            self.assertEqual(calls[0], {"texts": ["seven", "nine"],
                                        "token": "TOK-A",  # nosec B105
                                        "tag": "reask1-s00",
                                        "seed": 4321})
            self.assertEqual(calls[1], {"texts": ["fixed seven"],
                                        "token": "TOK-B",  # nosec B105
                                        "tag": "repair08-x-s00",
                                        "seed": 1234})
            # a tag the checkpoint never planned must RAISE, not submit
            with self.assertRaises(KeyError):
                submit_tts_shard.lambda_handler(
                    {"plan_s3uri": "s3://b/s3plan", "loop": "synth",
                     "tag": "main-s99", "task_token": "TOK-C"}, None)  # nosec B105
        finally:
            C.read_json, C.parse_s3_uri = olds
            submit_mod.tts_submit = old_tts

    def test_collect_fanout_is_tolerant_not_strict(self):
        """Review 2026-09-02: the strict register read wedged every
        redrive when the fan-out itself was lost. Missing rows/docs and
        never-landed retries re-fan; FAILED shards retry under a bumped
        attempt until TTS_SHARD_ATTEMPTS, then raise BY NAME."""
        from lib import submit
        store = {}
        olds = (C.read_json_if_exists, C.read_json)
        row = lambda tag, att=0: {  # noqa: E731
            "kind": "tts", "tag": tag, "workflow_id": "w",
            "output_s3_prefix": f"s3://b/runs/{tag}", "count": 1,
            "attempt": att}
        good_doc = {"status": "SUCCEEDED", "outputs": [
            {"transcript_index": 0, "uri": "s3://b/t.wav",
             "duration_s": 1.0, "qc_passed": True}]}
        try:
            C.read_json_if_exists = lambda b, k=None: (
                store.get(k) if k is not None else store.get(b))
            C.read_json = C.read_json_if_exists
            specs = {"main-s00": {"order": [1], "seed": 1234,
                                  "attempt": 0}}
            # no row at all -> refan
            self.assertEqual(submit.collect_fanout("pfx", specs),
                             {"refan": ["main-s00"]})
            # row landed but the outputs doc has not -> refan
            store["pfx/_plan/submits/tts-main-s00.json"] = row("main-s00")
            self.assertEqual(submit.collect_fanout("pfx", specs),
                             {"refan": ["main-s00"]})
            # FAILED doc, attempts left -> retry (caller bumps)
            store["runs/main-s00/outputs.json"] = {
                "status": "FAILED", "error": "GPU fell over",
                "execution_arn": "arn:e"}
            self.assertEqual(submit.collect_fanout("pfx", specs),
                             {"retry": ["main-s00"]})
            # bumped attempt whose retry never landed -> refan, NOT a
            # second bump (row still says attempt 0)
            specs["main-s00"]["attempt"] = 1
            self.assertEqual(submit.collect_fanout("pfx", specs),
                             {"refan": ["main-s00"]})
            # retry landed and FAILED again, attempts exhausted -> named
            store["pfx/_plan/submits/tts-main-s00.json"] = \
                row("main-s00", att=1)
            with self.assertRaises(RuntimeError) as ctx:
                submit.collect_fanout("pfx", specs)
            self.assertIn("failed on all 2 attempt(s)", str(ctx.exception))
            self.assertIn("bump .version", str(ctx.exception))
            # retry landed and SUCCEEDED -> outs
            store["runs/main-s00/outputs.json"] = good_doc
            got = submit.collect_fanout("pfx", specs)
            self.assertEqual(list(got["outs"]), ["main-s00"])
            self.assertEqual(got["outs"]["main-s00"][0]["duration_s"], 1.0)
        finally:
            C.read_json_if_exists, C.read_json = olds

    def test_reconcile_fanout_bumps_and_checkpoints_before_refan(self):
        """The retry's -r{n} idempotency token must survive a lost send:
        the attempt bump is CHECKPOINTED before the fan-out is re-issued
        (the fit leg then sees row.attempt < spec.attempt and re-fans
        without a second bump)."""
        from lib import loop_state
        store = {}
        old = C.write_json
        try:
            C.write_json = lambda b, k, v: store.__setitem__(k, v)
            state = {"phase": "tts",
                     "shard_specs": {"main-s00": {"order": [1],
                                                  "attempt": 0},
                                     "main-s01": {"order": [2],
                                                  "attempt": 0}}}
            self.assertIsNone(loop_state.reconcile_fanout(
                {"outs": {}}, state, "ck"))
            self.assertEqual(store, {})
            # refan re-fans the FULL tag list without touching attempts
            tags = loop_state.reconcile_fanout(
                {"refan": ["main-s01"]}, state, "ck")
            self.assertEqual(sorted(tags), ["main-s00", "main-s01"])
            self.assertEqual(store, {})
            # retry bumps only the failed tag and checkpoints
            tags = loop_state.reconcile_fanout(
                {"retry": ["main-s01"]}, state, "ck")
            self.assertEqual(sorted(tags), ["main-s00", "main-s01"])
            self.assertEqual(
                store["ck"]["shard_specs"]["main-s01"]["attempt"], 1)
            self.assertEqual(
                store["ck"]["shard_specs"]["main-s00"]["attempt"], 0)
        finally:
            C.write_json = old

    def test_absorbed_failure_submit_registers_instead_of_raising(self):
        """An absorbed resubmit of a FAILED run answers with the handle
        PLUS its error — and the gateway has already completed the
        iteration's token, so raising would fail a task that already
        resolved. A true refusal (no handle) still raises."""
        from lib import submit
        old_gw, old_write = C.invoke_gateway_tool, C.write_json
        store = {}
        try:
            C.write_json = lambda b, k, v: store.__setitem__(k, v)
            C.invoke_gateway_tool = lambda *a, **k: {
                "error": True, "errorCode": "EXECUTION_FAILED",
                "message": "run failed", "workflow_id": "w",
                "job_id": "j", "output_s3_prefix": "s3://b/p"}
            rec = submit.tts_submit(
                ["hi"], "TOK", {"workflow_id": "w", "voice_id": "v",
                                "output_prefix": "pfx"},
                "main-s00", attempt=1)
            self.assertEqual(rec["attempt"], 1)
            self.assertIn("pfx/_plan/submits/tts-main-s00.json", store)
            C.invoke_gateway_tool = lambda *a, **k: {
                "error": True, "errorCode": "INVALID_PARAMETER",
                "message": "no such voice"}
            with self.assertRaises(RuntimeError) as ctx:
                submit.tts_submit(
                    ["hi"], "TOK", {"workflow_id": "w", "voice_id": "v",
                                    "output_prefix": "pfx"}, "main-s00")
            self.assertIn("refused", str(ctx.exception))
        finally:
            C.invoke_gateway_tool, C.write_json = old_gw, old_write

    def test_retry_idempotency_token_is_attempt_scoped(self):
        """Attempt 0 keeps the plain {workflow_id}-{tag} token (replays
        must absorb); a retry submits under -r{n} so it starts a FRESH
        run instead of absorbing into the sticky failure."""
        from lib import submit
        sent = []
        old_gw, old_write = C.invoke_gateway_tool, C.write_json
        try:
            C.write_json = lambda b, k, v: None
            C.invoke_gateway_tool = lambda arn, tool, ev: sent.append(
                ev) or {"workflow_id": "w", "job_id": "j",
                        "output_s3_prefix": "s3://b/p"}
            plan = {"workflow_id": "W", "voice_id": "v",
                    "output_prefix": "pfx"}
            submit.tts_submit(["a"], "T", plan, "main-s00")
            submit.tts_submit(["a"], "T", plan, "main-s00", attempt=1)
            self.assertEqual(sent[0]["idempotency_token"], "W-main-s00")
            self.assertEqual(sent[1]["idempotency_token"],
                             "W-main-s00-r1")
        finally:
            C.invoke_gateway_tool, C.write_json = old_gw, old_write

    def test_pre_fanout_checkpoint_is_refused_by_name(self):
        """An execution that crossed the 2026-09-02 deploy mid-loop
        carries the serial keys and is pinned to an ASL without the
        Maps — a named refusal beats a bare KeyError."""
        from lib import loop_state
        with self.assertRaises(RuntimeError) as ctx:
            loop_state.assert_fanout_checkpoint(
                {"phase": "tts", "shards": [[1, 2]], "shard": 0,
                 "order": [1, 2], "submit": {}}, "synth")
        msg = str(ctx.exception)
        self.assertIn("predates the 2026-09-02 shard fan-out", msg)
        self.assertIn("re-run", msg)
        loop_state.assert_fanout_checkpoint(
            {"phase": "tts", "shard_specs": {}}, "synth")  # must not raise

    def test_usage_sweep_folds_in_the_legacy_register(self):
        """A run that registered under the pre-fan-out single-file
        register and finishes under this code must not report $0 spend
        as complete."""
        from lib import submit
        from states.usage_tracking import lambda_function as usage_tracking
        store = {
            "plankey": {"output_prefix": "pfx", "workflow_id": "W",
                        "tool_version": "t"},
            "pfx/_plan/submits.json": [
                {"kind": "bedrock", "tag": "writer",
                 "output_s3_prefix": "s3://b/runs/writer"},
                {"kind": "tts", "tag": "main-s00"}],
            "runs/writer/summary.json": {"bedrock_usage": {
                "estimated_cost": 0.5, "cost_is_complete": True,
                "input_tokens": 10, "output_tokens": 2}},
        }
        olds = (C.read_json, C.read_json_if_exists, C.write_json,
                C.list_keys, C.parse_s3_uri)
        try:
            C.read_json = lambda b, k=None: store[k if k is not None
                                                  else b]
            C.read_json_if_exists = lambda b, k=None: store.get(
                k if k is not None else b)
            C.write_json = lambda b, k, v: store.__setitem__(k, v)
            C.list_keys = lambda b, p: sorted(
                k for k in store if k.startswith(p))
            C.parse_s3_uri = lambda u: ("b", u[5:].partition("/")[2])
            usage_tracking.lambda_handler(
                {"uuid": "W", "plan_s3uri": "s3://b/plankey"}, None)
            usage = store["pfx/usage_tracking.json"]
            self.assertEqual(usage["total_estimated_cost"], 0.5)
            self.assertTrue(usage["cost_is_complete"])
            self.assertEqual(usage["tts_submissions"], 1)
        finally:
            (C.read_json, C.read_json_if_exists, C.write_json,
             C.list_keys, C.parse_s3_uri) = olds

    def test_fanout_choices_route_our_own_payloads_only(self):
        """ASL structure: both loop Choices check ad_loop_done FIRST,
        then tts_shard_tags -> the loop's own Map; the Map's iteration
        Task is waitForTaskToken and its Next re-enters the loop state.
        Sibling payloads carry neither key, so they still re-enter."""
        import json as _json
        with open(os.path.join(_ROOT, "definitions.json")) as f:
            asl = _json.load(f)["States"]
        for choice, loop_state_name, map_name in (
                ("All takes fitted?", "Synthesize and fit",
                 "Synthesize shards in parallel"),
                ("Repair chain complete?", "Repair defects one at a time",
                 "Re-synthesize repaired shards")):
            nexts = [c["Next"] for c in asl[choice]["Choices"]]
            variables = [c["Variable"] for c in asl[choice]["Choices"]]
            self.assertEqual(variables,
                             ["$.callback.ad_loop_done",
                              "$.callback.tts_shard_tags"])
            self.assertEqual(nexts[1], map_name)
            self.assertEqual(asl[choice]["Default"], loop_state_name)
            m = asl[map_name]
            self.assertEqual(m["Type"], "Map")
            self.assertEqual(m["ItemsPath"], "$.callback.tts_shard_tags")
            self.assertEqual(m["MaxConcurrency"], 0)
            self.assertIsNone(m["ResultPath"])
            self.assertEqual(m["Next"], loop_state_name)
            (task,) = m["ItemProcessor"]["States"].values()
            self.assertTrue(task["Resource"].endswith("waitForTaskToken"))
            payload = task["Parameters"]["Payload"]
            self.assertEqual(payload["operation"], "SubmitTtsShard")
            self.assertEqual(payload["task_token.$"], "$$.Task.Token")
            self.assertEqual(payload["tag.$"], "$.tag")


class TestCastResolutionJoin(unittest.TestCase):
    """The optional sixth sibling (cast-resolution, 2026-09-07):
    validate_plan reads persons.json ONCE — provenance pinned on the
    producer's documented sibling_workflow_ids — and forwards the
    filtered bindings via plan.json; understand renders them above the
    raw hints; parse prunes face_key-less characters."""

    PLAN = {"prompts": {"ad-name-map": {
        "system": "sys", "user_template": "{{body}}",
        "variables": [{"token": "{{body}}"}]}}}  # nosec B105

    PEOPLE = [{"person_id": 3, "name": "Miriam Okafor",
               "named": True, "total_screen_time_sec": 100.0,
               "appearances": [{"begin_ms": 0, "end_ms": 5000}]}]

    @staticmethod
    def _ev():
        return {
            "transcript": {"sentences": [
                {"text": "Good evening from the studio.",
                 "start": 1.0, "speaker_id": "SPEAKER_00"}]},
            "ca": {"overall_content_type": "News",
                   "top_3_genres": ["news"], "overall_summary": "s"},
            "shots": [{"shot_id": 0, "start_time": 0.0,
                       "end_time": 10.0}],
            "shot_docs": [{"shot_id": 0, "description": "studio desk"}],
            "people": TestCastResolutionJoin.PEOPLE,
        }

    def test_cast_block_enters_the_name_map_record(self):
        from states.understand import lambda_function as understand
        plan = dict(self.PLAN, cast_bindings=[{
            "name": "Miriam Okafor", "type": "on_camera_speaker",
            "role": "anchor", "character_name": "Det. Rios",
            "aliases": [],
            "speaker_ids": ["SPEAKER_00"], "face_ids": [3],
            "name_mechanisms": ["self_id", "chyron"]}])
        rec = understand._name_map_record(plan, self._ev())
        body = rec["messages"][0]["content"][0]["text"]
        self.assertIn("ADJUDICATED PERSON BINDINGS", body)
        self.assertIn("Miriam Okafor (on_camera_speaker, role=anchor, "
                      "plays Det. Rios)", body)
        self.assertIn("speaks as SPEAKER_00", body)
        self.assertIn("face keys: Miriam Okafor", body)
        self.assertIn("via self_id,chyron", body)
        self.assertLess(body.index("ADJUDICATED"),
                        body.index("FACE KEYS"))

    def test_absent_bindings_change_nothing(self):
        from states.understand import lambda_function as understand
        rec = understand._name_map_record(self.PLAN, self._ev())
        self.assertNotIn("ADJUDICATED",
                         rec["messages"][0]["content"][0]["text"])

    def test_face_keys_are_deduped_and_capped(self):
        # coalesced duplicates carry every original facemetadata id on
        # purpose; the rendered line must not repeat one name per id
        from states.understand import lambda_function as understand
        people = [{"person_id": i, "name": "Miriam Okafor"}
                  for i in range(3)] + \
                 [{"person_id": i, "name": f"Person_{i}"}
                  for i in range(3, 10)]
        block = understand._cast_block(
            [{"name": "Miriam Okafor", "type": "on_camera_speaker",
              "face_ids": list(range(10)), "name_mechanisms": []}],
            people)
        self.assertEqual(block.count("Miriam Okafor"), 2)  # name + 1 key
        self.assertIn("face keys: Miriam Okafor, Person_3, Person_4, "
                      "Person_5; via", block)

    def test_cast_block_survives_absent_people(self):
        from states.understand import lambda_function as understand
        block = understand._cast_block(
            [{"name": "Miriam Okafor", "type": "voice_only",
              "speaker_ids": ["SPEAKER_07"], "face_ids": [],
              "name_mechanisms": ["self_id"]}], None)
        self.assertIn("Miriam Okafor (voice_only)", block)
        self.assertNotIn("; face keys:", block)

    def test_filter_keeps_named_top_cap_and_strips_evidence(self):
        from lib import evidence
        from lib import ad_common as CC
        doc = {"persons": [
            {"person_id": i, "name": f"Person Name {i}",
             "type": "silent_face", "screen_sec": float(i),
             "speech_sec": 0.0, "character_name": f"Character {i}",
             "evidence": [{"mechanism": "x"}]}
            for i in range(50)] + [
            {"person_id": 99, "name": None, "screen_sec": 999.0}]}
        out = evidence.filter_cast_bindings(doc, CC.CAST_BINDINGS_CAP)
        self.assertEqual(len(out), CC.CAST_BINDINGS_CAP)
        self.assertEqual(out[0]["name"], "Person Name 49")
        self.assertEqual(out[0]["character_name"], "Character 49")
        self.assertNotIn("evidence", out[0])
        self.assertNotIn("screen_sec", out[0])

    def test_provenance_pins_on_sibling_workflow_ids(self):
        from lib import evidence
        inputs = {"face_recognition_s3uri":
                  "s3://p/micro-workflow/face-recognition/RUN-A",
                  "speech_diarization_s3uri":
                  "s3://p/micro-workflow/speech-diarization/RUN-B/"}
        ok = {"provenance": {"sibling_workflow_ids": {
            "face_recognition": "RUN-A",
            "speech_diarization": "RUN-B"}}}
        self.assertIsNone(evidence.cast_provenance_error(inputs, ok))
        bad = {"provenance": {"sibling_workflow_ids": {
            "face_recognition": "RUN-OTHER"}}}
        msg = evidence.cast_provenance_error(inputs, bad)
        self.assertIn("RUN-OTHER", msg)
        self.assertIn("per-run labels", msg)
        # a full-artifact spelling still names the run id as a segment
        deep = dict(inputs, face_recognition_s3uri=
                    "s3://p/micro-workflow/face-recognition/RUN-A/"
                    "facemetadata.json")
        self.assertIsNone(evidence.cast_provenance_error(deep, ok))
        # no pin recorded -> no claim -> pass (lock-step test below
        # keeps the producer honest about the field name)
        self.assertIsNone(evidence.cast_provenance_error(inputs, {}))

    def test_parse_prunes_face_keyless_characters(self):
        from states.understand import lambda_function as understand
        out = ('{"characters": ['
               '{"character_name": "A", "face_key": "Miriam Okafor",'
               ' "first_established_sec": 5},'
               '{"character_name": "B", "first_established_sec": 9}]}')
        cmap = understand._parse_name_map(out)
        self.assertEqual(len(cmap["characters"]), 1)
        self.assertEqual(cmap["characters"][0]["face_key"],
                         "Miriam Okafor")
        with self.assertRaises(RuntimeError):
            understand._parse_name_map(
                '{"characters": [{"character_name": "B"}]}')

    def test_pointer_tuples_have_one_home_and_probes_cover_them(self):
        # the tuples live ONCE in ad_common; the gateway's flat zip
        # carries its own COPY of the module (separate Lambda, separate
        # zip since the 2026-10-06 tree restructure), so the invariant
        # is value equality — the workflow side still shares the one
        # module object. The probe table covers every pointer except
        # the documented exception — decisions.json absent is the
        # designed "nothing accepted yet" state of a real review, not
        # a caller error
        from lib import ad_common as CC
        from states.validate_plan import lambda_function as validate_plan
        import lambda_function as gw
        self.assertEqual(gw._REQUIRED, CC.REQUIRED_INPUT_URIS)
        self.assertEqual(gw._OPTIONAL, CC.OPTIONAL_INPUT_URIS)
        self.assertIs(validate_plan.REQUIRED, CC.REQUIRED_INPUT_URIS)
        self.assertIs(validate_plan.OPTIONAL, CC.OPTIONAL_INPUT_URIS)
        pointers = set(CC.REQUIRED_INPUT_URIS) | set(CC.OPTIONAL_INPUT_URIS)
        self.assertEqual(pointers - set(validate_plan.PROBES),
                         {"decisions_s3uri"},
                         "probe table drifted from the pointer tuples")

    def test_people_source_rule_one_home(self):
        # fr OR cast (2026-09-08): the rule lives once in ad_common;
        # both enforcement points call it
        from lib import ad_common as CC
        self.assertIsNone(CC.people_source_error(
            {"face_recognition_s3uri"}))
        self.assertIsNone(CC.people_source_error(
            {"cast_resolution_s3uri"}))
        msg = CC.people_source_error({"audiostem_s3uri"})
        self.assertIn("people source", msg)
        self.assertNotIn("face_recognition_s3uri",
                         CC.REQUIRED_INPUT_URIS)
        self.assertIn("face_recognition_s3uri", CC.OPTIONAL_INPUT_URIS)

    def test_cast_only_people_synthesis_shape(self):
        # cast as the sole people source: fetch synthesizes the
        # facemetadata-people shape from persons.json; the PRODUCER
        # declares the presence basis per person, and appearance-less
        # persons stay out of the people list
        from lib import evidence
        from unittest import mock
        doc = {"persons": [
            {"person_id": 0, "name": "Miriam Okafor",
             "screen_sec": 100.0, "appearances_basis": "face",
             "appearances": [{"begin_ms": 0, "end_ms": 5000}]},
            {"person_id": 1, "name": None, "screen_sec": 20.0,
             "appearances_basis": "face",
             "appearances": [{"begin_ms": 9000, "end_ms": 12000}]},
            {"person_id": 2, "name": "Radio Voice", "screen_sec": 0.0,
             "appearances_basis": "face", "appearances": []},
        ]}
        with mock.patch.object(evidence.C, "read_json",
                               return_value=doc):
            ev = evidence.fetch(
                {"cast_resolution_s3uri": "s3://p/mw/cr/RUN"},
                only={"people"})
        self.assertEqual(ev["people_basis"], "cast")
        self.assertEqual([p["name"] for p in ev["people"]],
                         ["Miriam Okafor", "Person_1"])
        self.assertTrue(ev["people"][0]["named"])
        self.assertFalse(ev["people"][1]["named"])

    def test_faceless_cast_reads_the_producers_speech_basis(self):
        # GDPR posture: the producer exported speech-basis timelines
        # and SAID so — the consumer never re-derives (review,
        # 2026-09-08: the emptiness heuristic falsely refused faced
        # runs that legitimately detected nothing)
        from lib import evidence
        from unittest import mock
        doc = {"persons": [
            {"person_id": 0, "name": "Miriam Okafor",
             "appearances_basis": "speech",
             "appearances": [{"begin_ms": 1000, "end_ms": 6000}]},
        ]}
        with mock.patch.object(evidence.C, "read_json",
                               return_value=doc):
            ev = evidence.fetch(
                {"cast_resolution_s3uri": "s3://p/mw/cr/RUN"},
                only={"people"})
        self.assertEqual(ev["people_basis"], "cast_speech")
        self.assertEqual(ev["people"][0]["appearances"],
                         [{"begin_ms": 1000, "end_ms": 6000}])

    def test_cast_only_record_keys_faces_by_own_name(self):
        # cast modes: a binding's own name is its identity key exactly
        # when that person made it into FACE KEYS (name membership) —
        # one rule for face- and speech-basis alike, so the two record
        # sections can never contradict
        from states.understand import lambda_function as understand
        block = understand._cast_block(
            [{"name": "Miriam Okafor", "type": "on_camera_speaker",
              "face_ids": [7], "name_mechanisms": ["chyron"]},
             {"name": "Radio Voice", "type": "voice_only",
              "speaker_ids": ["SPEAKER_09"], "face_ids": [],
              "name_mechanisms": ["self_id"]}],
            [{"person_id": 0, "name": "Miriam Okafor"}],
            basis="cast")
        self.assertIn("face keys: Miriam Okafor", block)
        self.assertNotIn("face keys: Radio Voice", block)

    def test_faceless_block_keys_by_voice(self):
        from states.understand import lambda_function as understand
        block = understand._cast_block(
            [{"name": "Miriam Okafor", "type": "uncertain",
              "speaker_ids": ["S0"], "face_ids": [],
              "name_mechanisms": ["self_id"]}],
            [{"person_id": 0, "name": "Miriam Okafor"}],
            basis="cast_speech")
        self.assertIn("face keys: Miriam Okafor", block)
        # a binding whose person never earned a timeline gets no key
        block2 = understand._cast_block(
            [{"name": "Ghost Voice", "type": "uncertain",
              "speaker_ids": ["S9"], "face_ids": [],
              "name_mechanisms": ["self_id"]}],
            [], basis="cast_speech")
        self.assertNotIn("face keys: Ghost Voice", block2)

    def test_faceless_sentinel_paired_with_the_registry_prompt(self):
        # the faceless behavior is a matched pair split across repos:
        # understand.py's FACE KEYS header and ad-name-map's faceless
        # paragraph — pin the shared sentinel so a reword cannot strand
        # one half (silent empty character maps)
        import inspect
        from states.understand import lambda_function as understand
        self.assertIn("FACELESS", inspect.getsource(understand))
        sibling = os.path.join(os.path.dirname(_ROOT),
                               "prompt-registry-on-aws", "prompts",
                               "ad-name-map", "default.md")
        if not os.path.isfile(sibling):
            self.skipTest("prompt-registry repo not checked out")
        self.assertIn("FACELESS", _read(sibling, encoding="utf-8"))

    def test_lock_step_with_cast_resolution_contract(self):
        # the provenance pin consumes the producer's documented field;
        # a producer-side rename must fail a build here, not fail open
        sibling = os.path.join(os.path.dirname(_ROOT),
                               "cast-resolution-on-aws", "gateway_tool",
                               "output_contract.json")
        if not os.path.isfile(sibling):
            self.skipTest("cast-resolution repo not checked out")
        import json as _json
        contract = _load_json(sibling)
        example = ((contract.get("outputs") or {}).get("persons")
                   or {}).get("example")
        self.assertIsNotNone(example, "no persons example in contract")
        prov = example.get("provenance") or {}
        wids = prov.get("sibling_workflow_ids") or {}
        for key in ("face_recognition", "speech_diarization"):
            self.assertIn(key, wids,
                          "provenance.sibling_workflow_ids lost the "
                          f"{key} pin — cast_provenance_error would "
                          "fail open")
        # EVERY field filter_cast_bindings forwards or _cast_block
        # renders, plus the two sort keys — a producer rename of any of
        # them must fail a build, not fail open into blank line parts
        persons = example.get("persons") or [{}]
        pinned = ("person_id", "name", "type", "speaker_ids",
                  "face_ids", "screen_sec", "speech_sec", "role",
                  "character_name", "aliases", "name_mechanisms")
        for field in pinned:
            self.assertTrue(any(field in p for p in persons),
                            f"no persons example carries {field!r}")


class TestTailBlockCreditsShape(unittest.TestCase):
    def test_tail_check_consumes_the_witness_doc_shape(self):
        # credits.json is {"blocks": [...], "gaps": [...]} — iterating
        # the doc itself yielded its string KEYS and TypeError'd after
        # full spend (latent through the fi-pointer era, review
        # 2026-09-08 CONFIRMED; every cast-anchored run reaches it now)
        import types
        from unittest import mock
        from states.repair_chain import lambda_function as repair_chain
        run = types.SimpleNamespace(
            ev={"shots": [{"end_time": 100.0}],
                "credits": {"blocks": [
                    {"start": 90.0, "end": 99.0,
                     "cards": [{"t": 91.0, "lines": ["Directed by",
                                                     "Mira Santos"]}]}],
                    "gaps": []}},
            docs=[], tm=[], cues={}, placed=[])
        m = {"start": 88.0, "end": 100.0, "shot_ids": [9],
             "kinds": ["credit_roll"], "cues": []}
        with mock.patch.object(repair_chain.oracles,
                               "tail_overdescribed", return_value=m):
            rec = repair_chain._TailBlockCheck().detect(run, {})
        body = "\n".join(rec["lines"])
        self.assertIn("Mira Santos", body)
        self.assertIn("EVERY CARD READ", body)


class TestSnapTextBeats(unittest.TestCase):
    def test_beat_re_timed_to_the_shot_carrying_the_text(self):
        """The Meridian root cause: the watch-through said 12.0s, the
        card lives at 23.1-29.7s."""
        notes = {"beats": [{"sec": 12.0,
                            "what": "Title card 'Meridian' appears"}],
                 "reveals": []}
        docs = {5: {"texts_on_screen": ["MERIDIAN"]}}
        tm = {5: (23.1, 29.7)}
        fixed = oracles.snap_text_beats(notes, docs, tm)
        self.assertEqual(len(fixed), 1)
        self.assertEqual(notes["beats"][0]["sec"], 23.1)


class TestTailBlocks(unittest.TestCase):
    def test_short_black_bridges_and_long_black_splits(self):
        docs = {0: {"shot_kind": "credit_roll"},
                1: {"shot_kind": "black"},
                2: {"shot_kind": "logo_bumper"},
                3: {"shot_kind": "black"},
                4: {"shot_kind": "credit_roll"}}
        tm = {0: (0, 10), 1: (10, 12), 2: (12, 20),
              3: (20, 20 + C.TAIL_BRIDGE_MAX + 1), 4: (27, 30)}
        blocks = oracles.nondiegetic_blocks(docs, tm)
        self.assertEqual([b["shot_ids"] for b in blocks],
                         [[0, 1, 2], [4]])


class TestOperationsResolveInRouter(unittest.TestCase):
    def test_every_asl_operation_is_a_router_key(self):
        # the state NAME is display; the Payload `operation` is the
        # ROUTER KEY. The 2026-09-08 rename corrupted the one state
        # whose name equaled its operation string ("Understand") and
        # nothing failed until runtime — this pins the join
        import json
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "ad_router", os.path.join(_ROOT, "lambda_function.py"))
        router = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(router)
        d = _load_json(os.path.join(_ROOT, "definitions.json"))
        ops = set()
        def collect(states):
            for s in states.values():
                payload = (s.get("Parameters") or {}).get("Payload") or {}
                if isinstance(payload, dict) and "operation" in payload:
                    ops.add(payload["operation"])
                for k in ("Iterator", "ItemProcessor"):
                    if k in s:
                        collect(s[k]["States"])
        collect(d["States"])
        self.assertTrue(ops)
        unknown = ops - set(router._OPERATIONS)
        self.assertFalse(unknown, f"operations with no handler: {unknown}")


class TestStateMachineParity(unittest.TestCase):
    """cfnstack.yaml inlines the state machine as DefinitionString;
    definitions.json is the readability copy. Nothing in the build
    enforces agreement — this test does (the tts/PSA convention; it
    replaced the build script's inline-python heredoc 2026-08-31)."""

    def test_inline_asl_equals_definitions_json(self):
        import json as _json

        import yaml

        class Loader(yaml.SafeLoader):
            pass

        def _passthrough(loader, tag_suffix, node):
            if isinstance(node, yaml.ScalarNode):
                return {tag_suffix: loader.construct_scalar(node)}
            if isinstance(node, yaml.SequenceNode):
                return {tag_suffix: loader.construct_sequence(node,
                                                              deep=True)}
            return {tag_suffix: loader.construct_mapping(node, deep=True)}

        Loader.add_multi_constructor("!", _passthrough)
        with open(os.path.join(_ROOT, "cfnstack.yaml")) as f:
            tpl = yaml.load(f, Loader=Loader)  # nosec B506
        inline = _json.loads(
            tpl["Resources"]["AudioDescriptionStateMachine"]
               ["Properties"]["DefinitionString"])
        with open(os.path.join(_ROOT, "definitions.json")) as f:
            copy = _json.load(f)
        self.assertEqual(
            inline, copy,
            "the template's inline ASL drifted from definitions.json — "
            "edit BOTH")


class TestBedrockSubmitBatchFlag(unittest.TestCase):
    """bedrock_submit forwards the call site's latency posture: True on
    the big one-shot legs, ABSENT (guard decides, §5.6) when the caller
    passes None — a small serial leg must never be forced into a padded
    100-record batch job."""

    def _capture(self, **kwargs):
        from lib import submit
        sent = {}

        def fake_invoke(arn, tool, payload):
            sent.update(payload)
            return {"workflow_id": "w", "output_s3_prefix": "s3://b/p"}

        orig = (C.invoke_gateway_tool, C.write_bytes,
                C.write_json, C.read_json_if_exists)
        C.invoke_gateway_tool = fake_invoke
        C.write_bytes = lambda *a, **k: None
        C.write_json = lambda *a, **k: None
        C.read_json_if_exists = lambda *a, **k: None
        try:
            submit.bedrock_submit(
                [{"record_id": "r"}], "tok",
                {"output_prefix": "p", "workflow_id": "w",
                 "model_chain": ["m"]},
                "tag", **kwargs)
        finally:
            (C.invoke_gateway_tool, C.write_bytes,
             C.write_json, C.read_json_if_exists) = orig
        return sent

    def test_true_is_forwarded(self):
        self.assertIs(self._capture(use_batch=True)["use_batch"], True)

    def test_none_is_omitted_not_false(self):
        self.assertNotIn("use_batch", self._capture())



class TestGateway(unittest.TestCase):
    class _Ctx:
        def __init__(self, tool):
            custom = {"bedrockAgentCoreToolName":
                      f"audio-description___{tool}"}
            self.client_context = type("CC", (), {"custom": custom})()

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        path = os.path.join(_ROOT, "gateway_tool", "input_examples.json")
        examples = _load_json(path)["tools"]
        self.assertEqual(sorted(examples),
                         ["describe_tool", "generate_ad",
                          "get_job_status"])
        for tool, entry in examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        import lambda_function as gw
        out = gw.lambda_handler({}, self._Ctx("describe_tool"))
        self.assertIn("generate_ad", out["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        from unittest import mock
        import lambda_function as gw
        path = os.path.join(_ROOT, "gateway_tool", "input_examples.json")
        example = _load_json(path)["tools"]["generate_ad"]["example"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gw, "_sfn", fake):
            out = gw.lambda_handler(dict(example),
                                    self._Ctx("generate_ad"))
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()

    def test_poll_aliases_pre_rename_manifest_labels(self):
        # Labels went kebab-case 2026-10-13 (cue_ledger → cue-ledger,
        # render_manifest → render-manifest), but a run finalized BEFORE
        # that deploy — or an idempotent resubmit returning one — recorded
        # the old spellings in ad_manifest.json. The poll must alias them
        # so every response speaks the current contract (otherwise
        # _contract_desc finds no entry and prompt-directed selection by
        # the kebab label finds nothing).
        from unittest import mock
        import lambda_function as gw
        arn = ("arn:aws:states:eu-west-1:000000000000:execution:"
               "x-audio-description:abc123")
        fake = mock.Mock()
        fake.describe_execution.return_value = {"status": "SUCCEEDED"}
        old_manifest = {"shipped_cues": 5, "gaps": 0, "outputs": {
            "render_manifest": "s3://b/p/render_manifest.json",
            "cue_ledger": "s3://b/p/cue_ledger.json",
            "srt": "s3://b/p/delivery/t.srt"}}
        with mock.patch.object(gw, "_sfn", fake), \
                mock.patch.object(gw, "_read_json_if_exists",
                                  return_value=old_manifest):
            out = gw.lambda_handler({"job_id": arn},
                                    self._Ctx("get_job_status"))
        labels = [o["label"] for o in out["outputs"]]
        self.assertIn("render-manifest", labels)
        self.assertIn("cue-ledger", labels)
        self.assertNotIn("render_manifest", labels)
        self.assertNotIn("cue_ledger", labels)
        self.assertIn("srt", labels)        # untouched labels pass through
        # the aliased entries bind to the contract (desc present)
        by_label = {o["label"]: o for o in out["outputs"]}
        self.assertTrue(by_label["render-manifest"].get("desc"))

    def test_staged_alternative_passes_the_gateway_validation(self):
        # the face_recognition_s3uri people-source form — the other arm
        # of the one-of rule — must round-trip the same validation
        from unittest import mock
        import lambda_function as gw
        path = os.path.join(_ROOT, "gateway_tool", "input_examples.json")
        alt = (_load_json(path)["tools"]["generate_ad"]
               ["alternatives"]["face_recognition_s3uri"])
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gw, "_sfn", fake):
            out = gw.lambda_handler(dict(alt), self._Ctx("generate_ad"))
        self.assertNotIn("error", out, f"alternative refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()
        sent = json.loads(fake.start_execution.call_args.kwargs["input"])
        self.assertEqual(sent["input"]["face_recognition_s3uri"],
                         alt["face_recognition_s3uri"])
        self.assertNotIn("cast_resolution_s3uri", sent["input"])


class TestGatewayZipColdImport(unittest.TestCase):
    # the deployed gateway sees ONLY its zip members (caught live
    # 2026-10-06: a trimmed gateway zip ImportError'd on ad_common while
    # the suite passed, because tests put lib/ on the path) — cold-
    # import the dispatcher from a dir holding exactly what ships.
    # MEMBERS mirrors build-deploy-s3-dist.sh's gateway member list.
    MEMBERS = ("gateway_tool/lambda_function.py", "gateway_tool/workflow_identity.py",
               "gateway_tool/output_contract.json", "gateway_tool/input_examples.json",
               "lib/ad_common.py")

    def test_dispatcher_imports_from_zip_members_alone(self):
        import shutil
        import subprocess
        import tempfile
        d = tempfile.mkdtemp()
        root = _ROOT
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
