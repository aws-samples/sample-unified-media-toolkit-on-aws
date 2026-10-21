"""
Unit tests — conversation-analysis-on-aws core (README.md §2-§7). No AWS.

The chunker tests run against THREE REAL assets (fixtures/): Meridian's
full transcript (Netflix open test content) and shape skeletons of two
recorded news broadcasts (real timings/speakers, text replaced by
same-length placeholder — chunking depends only on gaps, speakers and text
length). The pinned chunk plans are the 2026-08-17 validation runs from
README.md §5: if a chunker change moves these cuts, that is a behavior change
to justify against the measurements, not a pin to update casually.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# lib/ ships flat at the zip root, so its modules are TOP-LEVEL at runtime.
sys.path.insert(0, ROOT)

from lib import chunker                               # noqa: E402
from lib import envelope                              # noqa: E402
from lib import merge                                 # noqa: E402
from lib import output_parse                          # noqa: E402
from lib import transcript as transcript_mod          # noqa: E402
from lib.transcript import (TranscriptError, load_transcript,  # noqa: E402
                        seconds_to_ts)

FIXTURES = os.path.join(ROOT, "test", "fixtures")


def load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return load_transcript(f.read())


def make_sentences(spec: list[tuple]) -> list[dict]:
    """(start, end, speaker, text) tuples -> sentence dicts, ids 0..n-1."""
    return [{"id": i, "start": a, "end": b, "speaker_id": sp, "text": t}
            for i, (a, b, sp, t) in enumerate(spec)]



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


class TestTranscriptJson(unittest.TestCase):
    def test_meridian_loads_losslessly(self):
        t = load_fixture("meridian_transcript.json")
        self.assertEqual(t["source_format"], "diarization-json")
        self.assertEqual(len(t["sentences"]), 88)
        self.assertEqual(t["sentences"][0]["id"], 0)
        self.assertEqual(t["sentences"][0]["speaker_id"], "SPEAKER_00")
        self.assertAlmostEqual(t["sentences"][0]["start"], 26.944)

    def test_missing_sentences_key_names_the_expected_shape(self):
        with self.assertRaisesRegex(TranscriptError, "final_transcript"):
            load_transcript('{"words": []}')

    def test_empty_and_unrecognized_inputs(self):
        with self.assertRaises(TranscriptError):
            load_transcript("")
        with self.assertRaisesRegex(TranscriptError, "unrecognized"):
            load_transcript("1\n00:00:01.000 --> 00:00:02.000\nhi\n")


class TestTranscriptWebVtt(unittest.TestCase):
    VTT_WITH_IDS = (
        "WEBVTT\n\nNOTE\nsome note\n\n"
        "10\n00:00:01.000 --> 00:00:03.500\n<v ALICE>Hello there.\n\n"
        "11\n00:00:04.000 --> 00:01:06.250\n<v BOB>Hi. How are you?\n")
    VTT_NO_IDS = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.500\nHello there,\nsplit line.\n\n"
        "00:00:04.000 --> 00:00:06.250\nSecond cue.\n")

    def test_integer_cue_identifiers_become_ids(self):
        t = load_transcript(self.VTT_WITH_IDS)
        self.assertEqual(t["source_format"], "webvtt")
        self.assertEqual([s["id"] for s in t["sentences"]], [10, 11])
        self.assertEqual(t["sentences"][0]["speaker_id"], "ALICE")
        self.assertEqual(t["sentences"][0]["text"], "Hello there.")
        self.assertAlmostEqual(t["sentences"][1]["end"], 66.25)

    def test_no_identifiers_fall_back_to_positions_and_unknown_speaker(self):
        t = load_transcript(self.VTT_NO_IDS)
        self.assertEqual([s["id"] for s in t["sentences"]], [0, 1])
        self.assertEqual(t["sentences"][0]["speaker_id"],
                         transcript_mod.UNKNOWN_SPEAKER)
        # multi-line cue text joins with a space
        self.assertEqual(t["sentences"][0]["text"],
                         "Hello there, split line.")

    def test_format_override_beats_the_sniff(self):
        with self.assertRaises(TranscriptError):
            load_transcript(self.VTT_WITH_IDS,
                            transcript_format="diarization-json")

    def test_seconds_to_ts_round_trip_shape(self):
        self.assertEqual(seconds_to_ts(0), "00:00:00.000")
        self.assertEqual(seconds_to_ts(3661.5), "01:01:01.500")


class TestChunkerRealAssets(unittest.TestCase):
    """The §5 validation runs, pinned (2026-08-17)."""

    def test_meridian_is_a_single_chunk_even_at_8k(self):
        t = load_fixture("meridian_transcript.json")
        plan = chunker.plan_chunks(t["sentences"], 8000)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["cut_tier"], chunker.TIER_END)
        self.assertIsNone(plan[0]["context_start_pos"])
        self.assertEqual((plan[0]["start_id"], plan[0]["end_id"]), (0, 87))

    def test_news_a_at_15k_cuts_weak_strong_strong(self):
        t = load_fixture("news_a_skeleton.json")
        plan = chunker.plan_chunks(t["sentences"], 15000)
        self.assertEqual([c["cut_tier"] for c in plan],
                         [chunker.TIER_WEAK_PAUSE, chunker.TIER_STRONG_PAUSE,
                          chunker.TIER_STRONG_PAUSE, chunker.TIER_END])
        # the two strong cuts land on verified real breaks (19.9s, 126.9s)
        self.assertGreater(plan[1]["cut_gap_s"], 15)
        self.assertGreater(plan[2]["cut_gap_s"], 100)
        # weak-pause cut carries context; strong cuts do not
        self.assertIsNotNone(plan[1]["context_start_pos"])
        self.assertIsNone(plan[2]["context_start_pos"])
        self.assertIsNone(plan[3]["context_start_pos"])

    def test_markets_a_at_15k_cuts_speaker_strong_speaker_speaker(self):
        t = load_fixture("markets_a_skeleton.json")
        plan = chunker.plan_chunks(t["sentences"], 15000)
        self.assertEqual([c["cut_tier"] for c in plan],
                         [chunker.TIER_SPEAKER_CHANGE,
                          chunker.TIER_STRONG_PAUSE,
                          chunker.TIER_SPEAKER_CHANGE,
                          chunker.TIER_SPEAKER_CHANGE, chunker.TIER_END])
        for c in plan[:-1]:
            self.assertLessEqual(c["est_tokens"], 15000 * 1.05)

    def test_chunks_tile_the_transcript_exactly(self):
        for name in ("news_a_skeleton.json", "markets_a_skeleton.json"):
            t = load_fixture(name)
            plan = chunker.plan_chunks(t["sentences"], 8000)
            covered = []
            for c in plan:
                covered.extend(range(c["start_pos"], c["end_pos"] + 1))
            self.assertEqual(covered, list(range(len(t["sentences"]))), name)


class TestChunkerMechanics(unittest.TestCase):
    def test_strong_pause_beats_weak_and_speaker_change(self):
        spec = [(i * 10.0, i * 10.0 + 8.0, "A" if i % 2 else "B", "x" * 300)
                for i in range(20)]
        spec[14] = (140.0, 148.0, "A", "x" * 300)
        # gap after cue 14 becomes 12s (strong): next starts at 160
        spec[15] = (160.0, 168.0, "B", "x" * 300)
        for i in range(16, 20):
            spec[i] = (i * 10.0 + 20, i * 10.0 + 28, "A" if i % 2 else "B",
                       "x" * 300)
        sents = make_sentences(spec)
        plan = chunker.plan_chunks(sents, 1500)   # cut lands near cue 15
        self.assertEqual(plan[0]["cut_tier"], chunker.TIER_STRONG_PAUSE)
        self.assertIsNone(plan[1]["context_start_pos"])

    def test_hard_cut_on_single_speaker_monologue(self):
        sents = make_sentences([(i * 5.0, i * 5.0 + 4.9, "SOLO", "y" * 300)
                                for i in range(30)])
        plan = chunker.plan_chunks(sents, 1000)
        self.assertGreater(len(plan), 1)
        self.assertEqual(plan[0]["cut_tier"], chunker.TIER_HARD_CUT)
        self.assertIsNotNone(plan[1]["context_start_pos"])

    def test_negative_gaps_do_not_break_the_math(self):
        """Crosstalk: a sentence starting before the previous one ends
        (measured -0.38s on markets_a) must plan cleanly."""
        spec = [(0.0, 5.0, "A", "z" * 300), (4.6, 9.0, "B", "z" * 300),
                (9.2, 14.0, "A", "z" * 300), (13.8, 18.0, "B", "z" * 300)]
        plan = chunker.plan_chunks(make_sentences(spec), 200)
        self.assertGreaterEqual(len(plan), 2)

    def test_context_is_capped_by_overlap_max(self):
        # One speaker turn far longer than the cap: context must not walk
        # past overlap_max_s of timeline.
        spec = [(float(i), i + 0.9, "A", "w" * 300) for i in range(40)]
        sents = make_sentences(spec)
        start = chunker._context_start(sents, 39, 10.0)
        self.assertGreaterEqual(sents[39]["end"] - sents[start]["start"],
                                0.0)
        self.assertLessEqual(sents[39]["end"] - sents[start]["start"],
                             11.0)


class TestEnvelope(unittest.TestCase):
    def setUp(self):
        self.t = load_fixture("meridian_transcript.json")
        self.plan = chunker.plan_chunks(self.t["sentences"], 8000)

    def test_single_chunk_note_is_one_line(self):
        vtt = envelope.build_envelope(
            self.t["sentences"], self.plan[0],
            duration_sec=self.t["duration_sec"], total_chunks=1)
        self.assertTrue(vtt.startswith("WEBVTT\n\nNOTE\nprogram duration "))
        self.assertNotIn("chunk", vtt.split("\n\n")[1])
        self.assertNotIn("CONTEXT", vtt)

    def test_cue_shape_id_timestamps_voice(self):
        vtt = envelope.build_envelope(
            self.t["sentences"], self.plan[0],
            duration_sec=self.t["duration_sec"], total_chunks=1)
        lines = vtt.split("\n")
        i = lines.index("0")
        self.assertEqual(lines[i + 1], "00:00:26.944 --> 00:00:29.244")
        self.assertEqual(lines[i + 2],
                         "<v SPEAKER_00>So these guys just disappeared?")

    def test_multi_chunk_note_marks_coverage_and_context(self):
        chunk = {"index": 1, "start_pos": 40, "end_pos": 79,
                 "context_start_pos": 36, "cut_tier": "weak-pause",
                 "cut_gap_s": 3.5, "est_tokens": 0}
        vtt = envelope.build_envelope(
            self.t["sentences"], chunk,
            duration_sec=self.t["duration_sec"], total_chunks=3)
        note = vtt.split("\n\n")[1]
        self.assertIn("chunk 2 of 3", note)
        self.assertIn("coverage cues 40-79", note)
        self.assertIn("cues 36-39 are CONTEXT", note)
        self.assertIn("your coverage begins at cue 40", note)
        # context cues are rendered, before the coverage cues
        self.assertLess(vtt.index("\n36\n"), vtt.index("\n40\n"))

    def test_token_estimate_tracks_the_rendered_bytes(self):
        """The chunker prices what the envelope renders: the estimate must
        stay within 15% of the actual rendered size or budgets lie."""
        vtt = envelope.build_envelope(
            self.t["sentences"], self.plan[0],
            duration_sec=self.t["duration_sec"], total_chunks=1)
        actual = len(vtt) / 3.5
        est = self.plan[0]["est_tokens"]
        self.assertLess(abs(actual - est) / actual, 0.15)


class TestOutputParse(unittest.TestCase):
    GOOD = (
        '{"overall_content_type": "News Program", '
        '"top_3_genres": ["News", "Politics", "Finance"]}\n'
        '{"start_id": 0, "end_id": 4, "segment_topic": "open", '
        '"reasoning": "r", "named_entities": [], "key_events": [], '
        '"sentiment": "neutral", "tone": "calm", '
        '"conversation_type": "news segment reporting", '
        '"continues_previous": false}\n'
        '{"start_id": 5, "end_id": 9, "segment_topic": "markets", '
        '"reasoning": "r", "named_entities": ["NYSE"], "key_events": [], '
        '"sentiment": "neutral", "tone": "calm", '
        '"conversation_type": "market/stock/election analysis", '
        '"continues_previous": false}\n')

    def test_happy_path(self):
        r = output_parse.parse_response(self.GOOD)
        self.assertEqual(r["header"]["overall_content_type"], "News Program")
        self.assertEqual(len(r["segments"]), 2)
        self.assertEqual(r["bad_lines"], [])
        v = output_parse.validate_coverage(r["segments"], list(range(10)))
        self.assertTrue(v["ok"])

    def test_code_fences_are_tolerated(self):
        r = output_parse.parse_response("```jsonl\n" + self.GOOD + "```\n")
        self.assertEqual(len(r["segments"]), 2)

    def test_a_truncated_last_line_is_a_recovered_partial_not_a_gap(self):
        # The cut segment kept its cue ids, so the honest prefix is KEPT
        # (user decision 2026-08-26: minimal gate) — counted, never silent
        # — and coverage survives. normalize defaults what the cut lost.
        truncated = self.GOOD[:self.GOOD.rindex('"conversation_type"')]
        r = output_parse.parse_response(truncated)
        self.assertEqual(len(r["segments"]), 2)
        self.assertEqual(r["bad_lines"], [])
        self.assertEqual(r["line_repairs"], {"prefix": 1})
        v = output_parse.validate_coverage(r["segments"], list(range(10)))
        self.assertTrue(v["ok"])
        recovered = output_parse.normalize_segments(r["segments"])[1]
        self.assertEqual(recovered["conversation_type"], "")

    def test_a_cut_that_destroys_the_cue_ids_stays_an_honest_gap(self):
        # Cut inside end_id's value: the prefix cannot be placed, so the
        # line is refused and the range is the retryable tail.
        second = self.GOOD.splitlines()[2]
        truncated = ("\n".join(self.GOOD.splitlines()[:2]) + "\n"
                     + second[:second.index('"end_id"') + 10])
        r = output_parse.parse_response(truncated)
        self.assertEqual(len(r["segments"]), 1)
        self.assertEqual(len(r["bad_lines"]), 1)
        self.assertEqual(r["line_repairs"], {})
        self.assertIn("prefix lost required", r["bad_lines"][0]["error"])
        v = output_parse.validate_coverage(r["segments"], list(range(10)))
        self.assertEqual(v["gaps"], [[5, 9]])        # the retryable tail

    def test_an_unescaped_quote_in_a_value_is_recovered_and_counted(self):
        # The malformation that cost shot-metadata-enrichment 10 shots
        # (2026-08-25): default_news asks for sig_out VERBATIM, so it is
        # this workflow's exposure too.
        line = ('{"start_id": 0, "end_id": 4, "segment_topic": "promo", '
                '"reasoning": "the anchor teases "SQUAWK ON THE STREET" '
                'next", "continues_previous": false}')
        r = output_parse.parse_response(line)
        self.assertEqual(r["bad_lines"], [])
        self.assertEqual(r["line_repairs"], {"escaped": 1})
        self.assertEqual(r["segments"][0]["reasoning"],
                         'the anchor teases "SQUAWK ON THE STREET" next')

    def test_validation_names_overlap_unknown_and_inversion(self):
        segs = [{"start_id": 0, "end_id": 5}, {"start_id": 4, "end_id": 12},
                {"start_id": 9, "end_id": 7}]
        v = output_parse.validate_coverage(segs, list(range(10)))
        self.assertFalse(v["ok"])
        text = " ".join(v["problems"])
        self.assertIn("overlaps", text)
        self.assertIn("outside coverage", text)
        self.assertIn("inverted", text)

    def test_normalize_sorts_and_defaults(self):
        segs = output_parse.normalize_segments([
            {"start_id": 5, "end_id": 9, "named_entities": "notalist"},
            {"start_id": "0", "end_id": 4, "continues_previous": True}])
        self.assertEqual([s["start_id"] for s in segs], [0, 5])
        self.assertTrue(segs[0]["continues_previous"])
        self.assertEqual(segs[1]["named_entities"], [])
        self.assertEqual(segs[0]["segment_topic"], "")

    def test_normalize_passes_prompt_extension_fields_through(self):
        """A NAMED prompt version (prompt_version, e.g. default_news) may
        extend the segment schema — news_form, slug, sig_out, ... — and
        extending a prompt must never require a workflow code change.
        Dropping declared prompt output is data loss, not tolerance: the
        2026-08-23 default_news validation run emitted these fields and the
        old whitelist silently discarded every one."""
        segs = output_parse.normalize_segments([
            {"start_id": 0, "end_id": 4, "segment_topic": "t",
             "news_form": "package", "interaction": "none",
             "sig_out": "Jane Smith, GNN News, London",
             "slug": "HELENE-PREPS", "correspondent": "Jane Smith"}])
        s = segs[0]
        self.assertEqual(s["news_form"], "package")
        self.assertEqual(s["slug"], "HELENE-PREPS")
        self.assertEqual(s["sig_out"], "Jane Smith, GNN News, London")
        # ...while the core coercions still apply on the same segment.
        self.assertEqual(s["named_entities"], [])
        self.assertEqual(s["segment_topic"], "t")


class TestMerge(unittest.TestCase):
    def test_header_extension_fields_pass_through_by_plurality(self):
        """A NAMED prompt version may extend the HEADER (default_news adds
        program_name — measured dropped by the old reconciler while every
        chunk emitted it). Scalars pass through by weighted plurality; the
        core fields keep their dedicated reconciliation."""
        sentences = make_sentences(
            [(i * 10.0, i * 10.0 + 9.0, "A", "t") for i in range(20)])
        seg = {"start_id": 0, "end_id": 19, "segment_topic": "t",
               "reasoning": "", "named_entities": [], "key_events": [],
               "sentiment": "n", "tone": "c",
               "conversation_type": "narration", "continues_previous": False}
        chunks = [
            {"header": {"overall_content_type": "News Program",
                        "top_3_genres": ["News"],
                        "program_name": "GNN Newshour",
                        "content_type_candidates": "clobber-attempt"},
             "segments": [dict(seg)], "gaps": []},
            {"header": {"overall_content_type": "News Program",
                        "top_3_genres": ["News"],
                        "program_name": "GNN Newshour"},
             "segments": [], "gaps": []},
        ]
        out = merge.merge_chunks(chunks, sentences,
                                 sentences[-1]["end"])
        self.assertEqual(out["program_name"], "GNN Newshour")
        self.assertEqual(out["overall_content_type"], "News Program")
        # COMPUTED fields never yield to a same-named prompt emission: the
        # unanimous vote above means no candidates report exists, and the
        # prompt's "clobber-attempt" must not become one.
        self.assertNotIn("content_type_candidates", out)

    def setUp(self):
        self.sentences = make_sentences(
            [(i * 10.0, i * 10.0 + 9.0, "A", "t") for i in range(20)])

    def _seg(self, a, b, topic, cont=False, ents=()):
        return {"start_id": a, "end_id": b, "segment_topic": topic,
                "reasoning": f"why {topic}", "named_entities": list(ents),
                "key_events": [], "sentiment": "neutral", "tone": "calm",
                "conversation_type": "narration",
                "continues_previous": cont}

    def _chunk(self, ct, genres, segs):
        return {"header": {"overall_content_type": ct,
                           "top_3_genres": genres},
                "segments": segs, "gaps": []}

    def test_continues_previous_stitches_across_the_cut(self):
        out = merge.merge_chunks([
            self._chunk("Episodic", ["Drama"], [
                self._seg(0, 4, "setup"),
                self._seg(5, 9, "the heist", ents=("Bank",))]),
            self._chunk("Episodic", ["Drama"], [
                self._seg(10, 12, "the heist continues", cont=True,
                          ents=("Bank", "Vault")),
                self._seg(13, 19, "aftermath")]),
        ], self.sentences, 200.0)
        segs = out["audio_segments"]
        self.assertEqual(len(segs), 3)
        stitched = segs[1]
        self.assertEqual((stitched["start_id"], stitched["end_id"]), (5, 12))
        self.assertEqual(stitched["segment_topic"], "the heist")   # earlier wins
        self.assertEqual(stitched["named_entities"], ["Bank", "Vault"])
        self.assertEqual(stitched["start_time"], "00:00:50.000")
        self.assertEqual(stitched["end_time"], "00:02:09.000")

    def test_false_flag_keeps_the_boundary(self):
        out = merge.merge_chunks([
            self._chunk("Episodic", ["Drama"], [self._seg(0, 9, "one")]),
            self._chunk("Episodic", ["Drama"], [self._seg(10, 19, "two")]),
        ], self.sentences, 200.0)
        self.assertEqual(len(out["audio_segments"]), 2)

    def test_first_chunk_ignores_a_spurious_continues_flag(self):
        """No previous segment exists — a hallucinated true on chunk 1 must
        not crash or drop the segment."""
        out = merge.merge_chunks([
            self._chunk("Episodic", ["Drama"],
                        [self._seg(0, 9, "only", cont=True)]),
        ], self.sentences, 200.0)
        self.assertEqual(len(out["audio_segments"]), 1)

    def test_majority_vote_yields_a_single_content_type(self):
        out = merge.merge_chunks([
            self._chunk("News Program", ["News"], [self._seg(0, 9, "a")]),
            self._chunk("News Program", ["News"], [self._seg(10, 14, "b")]),
            self._chunk("Documentary", ["News"], [self._seg(15, 19, "c")]),
        ], self.sentences, 200.0)
        self.assertEqual(out["overall_content_type"], "News Program")
        self.assertNotIn("content_type_candidates", out)

    def test_no_majority_reports_candidates_weight_ordered(self):
        """50/50 split by coverage duration -> no strict majority: the
        top-weighted value leads and the disagreement is REPORTED, never
        adjudicated by another inference (user decision)."""
        out = merge.merge_chunks([
            self._chunk("News Program", ["News"], [self._seg(0, 9, "a")]),
            self._chunk("Documentary", ["History"], [self._seg(10, 19, "b")]),
        ], self.sentences, 200.0)
        self.assertIn("content_type_candidates", out)
        self.assertEqual(set(out["content_type_candidates"]),
                         {"News Program", "Documentary"})
        self.assertEqual(out["overall_content_type"],
                         out["content_type_candidates"][0])

    def test_genres_dedup_and_weight_across_chunks(self):
        out = merge.merge_chunks([
            self._chunk("News Program", ["News", "Politics", "Weather"],
                        [self._seg(0, 14, "a")]),                 # 149s
            self._chunk("News Program", ["News", "Finance", "Politics"],
                        [self._seg(15, 19, "b")]),                # 49s
        ], self.sentences, 200.0)
        self.assertEqual(out["top_3_genres"][:2], ["News", "Politics"])
        self.assertEqual(len(out["top_3_genres"]), 3)

    def test_gaps_propagate_to_segments_gap(self):
        chunk = self._chunk("News Program", ["News"], [self._seg(0, 7, "a")])
        chunk["gaps"] = [[8, 9]]
        out = merge.merge_chunks([chunk], self.sentences, 200.0)
        self.assertEqual(out["segments_gap"], [[8, 9]])


class TestCopiedModules(unittest.TestCase):
    """micro-workflow-bundle/README.md §C5: the copies must stay byte-identical to the masters in
    prompt-registry-on-aws. This is the in-repo half; the bundle's parity
    classes are the cross-repo gate at registration time."""

    MASTER = os.path.join(os.path.dirname(ROOT), "prompt-registry-on-aws",
                          "gateway_tool")

    def _compare(self, name):
        master = os.path.join(self.MASTER, name)
        if not os.path.isfile(master):
            self.skipTest(f"master repo not present: {master}")
        with open(master, "rb") as f_m, \
                open(os.path.join(ROOT, "lib", name), "rb") as f_c:
            self.assertEqual(f_m.read(), f_c.read(),
                             f"{name} drifted from its master — copy it, "
                             "never edit it")

    def test_prompt_render_is_byte_identical(self):
        self._compare("prompt_render.py")

    def test_workflow_history_is_byte_identical(self):
        self._compare("workflow_history.py")

    def test_parse_partial_output_is_byte_identical(self):
        # A different master: bedrock-inference owns the salvage module
        # (micro-workflow-bundle/README.md §PI6.2; this repo is its second copy,
        # 2026-08-26 — the copy that activates the bundle parity class).
        master = os.path.join(os.path.dirname(ROOT),
                              "bedrock-inference-on-aws", "lib",
                              "parse_partial_output.py")
        if not os.path.isfile(master):
            self.skipTest(f"master repo not present: {master}")
        with open(master, "rb") as f_m, \
                open(os.path.join(ROOT, "lib",
                                  "parse_partial_output.py"), "rb") as f_c:
            self.assertEqual(f_m.read(), f_c.read(),
                             "parse_partial_output.py drifted from its "
                             "master — copy it, never edit it")

    def test_render_smoke_over_the_real_default_prompt_shape(self):
        sys.path.insert(0, os.path.join(ROOT, "lib"))
        import prompt_render
        artifact = {
            "system": "Analyze.",
            "user_template": "Go:\n{{transcript}}",
            "variables": [{"token": "{{transcript}}",  # nosec B105
                           "root": "transcript", "leafs": []}],
        }
        r = prompt_render.render(artifact, {"{{transcript}}": "WEBVTT\n..."})
        self.assertEqual(r["system"], "Analyze.")
        self.assertEqual(r["messages"],
                         [{"role": "user",
                           "content": [{"text": "Go:\nWEBVTT\n..."}]}])


if __name__ == "__main__":
    unittest.main()


# ─────────────────────────────────────────────────────────────────────────────
# The workflow: state handlers, gateway, template (README.md §7-§8)
# ─────────────────────────────────────────────────────────────────────────────

import io                                              # noqa: E402
from unittest import mock                              # noqa: E402

# the gateway dispatcher is importlib-loaded by name below; its own
# flat-zip sibling (workflow_identity) must be importable — APPENDED
# so `import lambda_function` keeps resolving to the ROOT router
sys.path.append(os.path.join(ROOT, "gateway_tool"))

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")   # hermetic clients
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("BEDROCK_GATEWAY_ARN", "arn:aws:lambda:eu-west-1:1:function:bedrock-gw")
os.environ.setdefault("PROMPT_REGISTRY_ARN", "arn:aws:lambda:eu-west-1:1:function:registry-gw")
os.environ.setdefault("STATE_MACHINE_ARN",
                      "arn:aws:states:eu-west-1:1:stateMachine:test-conversation-analysis")
os.environ.setdefault("TOOL_VERSION", "test.0")


from lib import ca_common                              # noqa: E402
from states.collect_finalize import lambda_function as collect_finalize  # noqa: E402
from states.render_stage import lambda_function as render_stage  # noqa: E402
import lambda_function as router                       # noqa: E402
from states.submit_history import lambda_function as submit_history  # noqa: E402
from states.usage_tracking import lambda_function as usage_mod  # noqa: E402
from states.validate_plan import lambda_function as validate_plan  # noqa: E402


class _FakeS3:
    """Dict-backed: get/put + the 404-vs-403 distinction ca_common relies on."""

    def __init__(self):
        self.objects: dict[tuple, bytes] = {}

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

    def json(self, key, bucket="test-proxy"):
        return json.loads(self.objects[(bucket, key)])


class _FakeLambda:
    """Routes (function_arn, tool_name) -> callable(event). Records calls."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def invoke(self, FunctionName, Payload, ClientContext=None, **_kw):
        import base64
        tool = json.loads(base64.b64decode(ClientContext))["custom"][
            "bedrockAgentCoreToolName"]
        event = json.loads(Payload)
        self.calls.append((FunctionName, tool, event))
        handler = self.routes[(FunctionName, tool)]
        result = handler(event)
        return {"Payload": io.BytesIO(json.dumps(result).encode("utf-8"))}


PROMPT_ARTIFACT = {
    "category": "conversation-analysis", "version": "default",
    "stage": "prod", "source_md5": "d" * 32,
    "system": "Analyze. Output JSONL with start_id/end_id, one JSON object "
              "per line.",
    "user_template": "Go:\n{{transcript}}",
    "variables": [{"token": "{{transcript}}", "root": "transcript",  # nosec B105
                   "leafs": []}],
}

BEDROCK_GW = os.environ["BEDROCK_GATEWAY_ARN"]
REGISTRY_GW = os.environ["PROMPT_REGISTRY_ARN"]
TRANSCRIPT_URI = ("s3://test-proxy/micro-workflow/speech-diarization/"
                  "diar-wf-1/transcripts/final_transcript.json")
TRANSCRIPT_KEY = TRANSCRIPT_URI[len("s3://test-proxy/"):]


class _WorkflowCase(unittest.TestCase):
    """Patched fakes + the Meridian fixture staged as the transcript."""

    def setUp(self):
        self.s3 = _FakeS3()
        self.lam = _FakeLambda()
        self.lam.routes[(REGISTRY_GW, "prompt-registry___get_prompt")] = \
            lambda ev: dict(PROMPT_ARTIFACT)
        # The default chain (sonnet-4-6) is validated on EVERY plan.
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___get_model")] = \
            lambda ev: {"model": {"model_id": ev["model_id"],
                                  "model_type": ["vision", "text"]}}
        with open(os.path.join(FIXTURES, "meridian_transcript.json"),
                  "rb") as f:
            self.s3.objects[("test-proxy", TRANSCRIPT_KEY)] = f.read()
        patches = [mock.patch.object(ca_common, "_s3", self.s3),
                   mock.patch.object(ca_common, "_lambda", self.lam)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def plan(self, **overrides):
        event = {"uuid": "wf-test-1",
                 "input": {"transcript_s3uri": TRANSCRIPT_URI,
                           **overrides}}
        return validate_plan.lambda_handler(event, None)


class TestValidatePlan(_WorkflowCase):
    def test_prompt_version_is_forwarded_to_the_registry(self):
        """A caller's prompt_version reaches get_prompt as `version`; absent
        means the category default (no version key at all)."""
        seen = {}
        def _get(ev):
            seen.clear(); seen.update(ev)
            return dict(PROMPT_ARTIFACT)
        self.lam.routes[(REGISTRY_GW, "prompt-registry___get_prompt")] = _get
        self.plan(prompt_version="default_news")
        self.assertEqual(seen.get("version"), "default_news")
        self.plan()
        self.assertNotIn("version", seen)

    def test_happy_path_plans_and_persists(self):
        out = self.plan()
        self.assertEqual(out["chunk_count"], 1)          # Meridian @15k
        plan = self.s3.json("micro-workflow/conversation-analysis/wf-test-1/"
                            "_plan/plan.json")
        self.assertEqual(plan["scope"], "transcript.diar-wf-1")
        self.assertEqual(plan["prompt"]["version"], "default")
        # Caller-absent defaults land CONCRETE in the plan (2026-08-18):
        # provenance names ids, never "default"; use_batch is an explicit
        # bool, never null.
        self.assertEqual(plan["model_chain"],
                 ["anthropic.claude-sonnet-4-6"])
        self.assertIs(plan["use_batch"], False)
        self.assertNotIn("max_tokens", plan)    # no knob anywhere (2026-08-18)
        self.assertEqual(plan["chunks"][0]["cut_tier"], "end")

    def test_scope_precedence_explicit_then_diarization_then_hash(self):
        self.assertEqual(self.plan(history_scope="asset.my-show")
                         and self.s3.json(
                             "micro-workflow/conversation-analysis/"
                             "wf-test-1/_plan/plan.json")["scope"],
                         "asset.my-show")
        other = "s3://test-proxy/micro-workflow/uploads/x.json"
        self.s3.objects[("test-proxy", other[len("s3://test-proxy/"):])] = \
            self.s3.objects[("test-proxy", TRANSCRIPT_KEY)]
        event = {"uuid": "wf-test-2",
                 "input": {"transcript_s3uri": other}}
        validate_plan.lambda_handler(event, None)
        scope = self.s3.json("micro-workflow/conversation-analysis/"
                             "wf-test-2/_plan/plan.json")["scope"]
        self.assertTrue(scope.startswith("transcript.")
                        and len(scope) == len("transcript.") + 12)

    def test_bad_explicit_scope_refuses(self):
        with self.assertRaisesRegex(
                validate_plan.ConversationAnalysisValidationError,
                "history_scope"):
            self.plan(history_scope="bad scope with spaces")

    def test_foreign_bucket_refuses(self):
        with self.assertRaisesRegex(
                validate_plan.ConversationAnalysisValidationError,
                "proxy bucket"):
            validate_plan.lambda_handler(
                {"uuid": "x", "input": {
                    "transcript_s3uri": "s3://elsewhere/t.json"}}, None)

    def test_prompt_with_uncatalogued_variable_refuses_before_spend(self):
        artifact = dict(PROMPT_ARTIFACT)
        artifact["variables"] = PROMPT_ARTIFACT["variables"] + [
            {"token": "{{frame_images}}", "root": "frame_images",  # nosec B105
             "leafs": []}]
        self.lam.routes[(REGISTRY_GW, "prompt-registry___get_prompt")] = \
            lambda ev: artifact
        with self.assertRaisesRegex(
                validate_plan.ConversationAnalysisValidationError,
                "frame_images"):
            self.plan()

    def test_model_chain_validated_against_the_registry(self):
        seen = []
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___get_model")] = \
            lambda ev: (seen.append(ev["model_id"]) or
                        {"model": {"model_id": ev["model_id"],
                                   "model_type": ["text"]}})
        self.plan(model_chain="model-a, model-b")
        self.assertEqual(seen, ["model-a", "model-b"])

    def test_the_default_chain_is_validated_too(self):
        """A deployment whose registry lacks the default fails at PLAN time,
        loudly — the default goes through the same validation door."""
        seen = []
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___get_model")] = \
            lambda ev: (seen.append(ev["model_id"]) or
                        {"model": {"model_type": ["text"]}})
        self.plan()
        self.assertEqual(seen, ["anthropic.claude-sonnet-4-6"])

        self.lam.routes[(BEDROCK_GW, "bedrock-inference___get_model")] = \
            lambda ev: {"error": True, "errorCode": "MODEL_NOT_REGISTERED",
                        "message": "no"}
        with self.assertRaisesRegex(
                validate_plan.ConversationAnalysisValidationError,
                "list_models"):
            self.plan(model_chain=["ghost-model"])

    def test_non_text_model_refuses(self):
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___get_model")] = \
            lambda ev: {"model": {"model_type": ["embedding"]}}
        with self.assertRaisesRegex(
                validate_plan.ConversationAnalysisValidationError,
                "text model"):
            self.plan(model_chain=["embed-1"])


class TestRenderStage(_WorkflowCase):
    def _plan_first(self):
        out = self.plan()
        return {"uuid": out["uuid"], "plan_s3uri": out["plan_s3uri"]}

    def test_renders_and_stages_every_chunk(self):
        out = render_stage.lambda_handler(self._plan_first(), None)
        self.assertEqual(out["record_count"], 1)
        lines = self.s3.objects[
            ("test-proxy", "micro-workflow/conversation-analysis/wf-test-1/"
             "records.jsonl")].decode().splitlines()
        record = json.loads(lines[0])
        self.assertEqual(record["record_id"], "chunk-000")
        self.assertEqual(record["system"], PROMPT_ARTIFACT["system"])
        text = record["messages"][0]["content"][0]["text"]
        self.assertIn("WEBVTT", text)
        self.assertIn("<v SPEAKER_00>So these guys just disappeared?", text)


class _SubmitCase(_WorkflowCase):
    HANDLE = {"job_id": "arn:aws:states:eu-west-1:1:execution:bi:run-1",
              "status": "RUNNING", "workflow_id": "run-1",
              "records_total": 1, "output_s3_prefix":
                  "s3://test-proxy/micro-workflow/bedrock-inference/run-1",
              "request_s3uri": "s3://test-proxy/micro-workflow/"
                               "bedrock-inference/run-1/input_params.json"}

    def submit_event(self):
        out = self.plan()
        staged = render_stage.lambda_handler(
            {"uuid": out["uuid"], "plan_s3uri": out["plan_s3uri"]}, None)
        staged["task_token"] = "tok-abc"  # nosec B105
        return staged


class TestSubmitHistory(_SubmitCase):
    def test_fresh_submit_writes_history_and_runs(self):
        submits = []
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___run_inference")] = \
            lambda ev: submits.append(ev) or dict(self.HANDLE)
        submit_history.lambda_handler(self.submit_event(), None)
        sent = submits[0]
        self.assertEqual(sent["callback_task_token"], "tok-abc")
        # ONE token = this run's uuid (2026-08-18): a redriven submit
        # absorbs into its own run; a caller rerun (fresh caller token ->
        # new uuid) gets a genuinely fresh inference instead of absorbing
        # into the run it is trying to escape.
        self.assertEqual(sent["idempotency_token"], "wf-test-1")
        self.assertEqual(sent["model_chain"],
                 ["anthropic.claude-sonnet-4-6"])
        self.assertIs(sent["use_batch"], False)
        self.assertNotIn("max_tokens", sent)   # no knob anywhere (2026-08-18)
        runs = self.s3.json("micro-workflow/conversation-analysis/wf-test-1/"
                            "_plan/runs.json")
        self.assertEqual(runs[0]["workflow_id"], "run-1")
        self.assertFalse(runs[0]["absorbed"])
        history_keys = [k for _, k in self.s3.objects
                        if k.startswith("micro-workflow/histories/"
                                        "transcript.diar-wf-1/"
                                        "conversation-analysis/default/")]
        self.assertEqual(len(history_keys), 1)

    def test_absorbed_submit_skips_history_but_records_the_run(self):
        absorbed = {**self.HANDLE,
                    "message": "identical request already submitted"}
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___run_inference")] = \
            lambda ev: absorbed
        submit_history.lambda_handler(self.submit_event(), None)
        runs = self.s3.json("micro-workflow/conversation-analysis/wf-test-1/"
                            "_plan/runs.json")
        self.assertTrue(runs[0]["absorbed"])
        self.assertIsNone(runs[0]["history_key"])

    def test_refusal_raises_because_the_return_is_discarded(self):
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___run_inference")] = \
            lambda ev: {"error": True, "errorCode": "INVALID_PARAMETER",
                        "message": "bad records"}
        with self.assertRaisesRegex(RuntimeError, "bad records"):
            submit_history.lambda_handler(self.submit_event(), None)




class TestCollectFinalize(_SubmitCase):
    RESPONSE_TEXT = (
        '{"overall_content_type": "Episodic", "top_3_genres": ["Drama", '
        '"Mystery", "Supernatural"]}\n'
        '{"start_id": 0, "end_id": 87, "segment_topic": "all", '
        '"reasoning": "r", "named_entities": [], "key_events": [], '
        '"sentiment": "neutral", "tone": "calm", '
        '"conversation_type": "narrative dialogue", '
        '"continues_previous": false}\n')

    def _submitted(self, output_text=None, record_status="SUCCESS"):
        self.lam.routes[(BEDROCK_GW, "bedrock-inference___run_inference")] = \
            lambda ev: dict(self.HANDLE)
        event = self.submit_event()
        submit_history.lambda_handler(event, None)
        line = {"record_id": "chunk-000", "status": record_status,
                "output_text": output_text or self.RESPONSE_TEXT}
        self.s3.put_object("test-proxy",
                           "micro-workflow/bedrock-inference/run-1/"
                           "response.jsonl", json.dumps(line) + "\n")
        event.pop("task_token")
        return {"operation": "CollectAndFinalize", **event,
                "wait_result": {"status": "SUCCEEDED"}}

    def test_complete_coverage_writes_both_products(self):
        out = router.lambda_handler(self._submitted(), None)
        self.assertEqual(out["segments"], 1)
        self.assertEqual(out["segments_gap"], 0)
        analysis = self.s3.json("micro-workflow/conversation-analysis/"
                                "wf-test-1/conversation_analysis.json")
        seg = analysis["audio_segments"][0]
        self.assertEqual(seg["start_time"], "00:00:26.944")   # authoritative
        self.assertEqual(analysis["provenance"]["prompt"]["version"],
                         "default")
        manifest = self.s3.json("micro-workflow/conversation-analysis/"
                                "wf-test-1/chunk_manifest.json")
        self.assertTrue(manifest["chunks"][0]["resolved"])
        self.assertEqual(manifest["inference_runs"][0]["workflow_id"],
                         "run-1")

    def test_coverage_gap_is_reported_honestly_and_still_succeeds(self):
        """No in-machine retry (2026-08-18): the uncovered tail lands in
        segments_gap and the run SUCCEEDS — the rerun is the caller's
        fresh idempotency_token."""
        partial = self.RESPONSE_TEXT.replace('"end_id": 87', '"end_id": 50')
        out = router.lambda_handler(self._submitted(partial), None)
        self.assertEqual(out["segments"], 1)
        self.assertEqual(out["segments_gap"], 1)
        analysis = self.s3.json("micro-workflow/conversation-analysis/"
                                "wf-test-1/conversation_analysis.json")
        self.assertEqual(analysis["segments_gap"], [[51, 87]])
        manifest = self.s3.json("micro-workflow/conversation-analysis/"
                                "wf-test-1/chunk_manifest.json")
        self.assertFalse(manifest["chunks"][0]["resolved"])

    def test_failed_record_becomes_a_whole_chunk_gap(self):
        out = router.lambda_handler(
            self._submitted(record_status="FAILED"), None)
        self.assertEqual(out["segments"], 0)
        analysis = self.s3.json("micro-workflow/conversation-analysis/"
                                "wf-test-1/conversation_analysis.json")
        self.assertEqual(analysis["segments_gap"], [[0, 87]])

    def test_terminal_failure_raises_as_itself(self):
        event = self._submitted()
        event["wait_result"] = {"status": "FAILED"}
        with self.assertRaises(
                collect_finalize.ConversationAnalysisInferenceFailed):
            router.lambda_handler(event, None)

    def test_usage_aggregates_and_never_raises(self):
        event = self._submitted()
        router.lambda_handler(event, None)
        self.s3.put_object("test-proxy",
                           "micro-workflow/bedrock-inference/run-1/"
                           "summary.json", json.dumps({
                               "inference_mode": "sync",
                               "records_succeeded": 1, "records_failed": 0,
                               "bedrock_usage": {"input_tokens": 100,
                                                 "output_tokens": 50,
                                                 "estimated_cost": 0.01,
                                                 "cost_is_complete": True}}))
        out = usage_mod.lambda_handler(event, None)
        self.assertEqual(out["total_estimated_cost"], 0.01)
        doc = self.s3.json("micro-workflow/conversation-analysis/wf-test-1/"
                           "usage_tracking.json")
        self.assertTrue(doc["cost_is_complete"])
        with mock.patch.object(ca_common, "read_json",
                               side_effect=RuntimeError("boom")):
            out = usage_mod.lambda_handler(event, None)
        self.assertEqual(out.get("usage"), "unavailable")


class TestRouter(unittest.TestCase):
    def test_operations_cover_the_asl_exactly(self):
        with open(os.path.join(ROOT, "definitions.json")) as f:
            asl = f.read()
        import re as _re
        named = set(_re.findall(r'"operation": "(\w+)"', asl))
        self.assertEqual(named, set(router._OPERATIONS))

    def test_unknown_operation_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown operation"):
            router.lambda_handler({"operation": "Nope"}, None)


class _GatewayCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ca_gateway", os.path.join(ROOT, "gateway_tool",
                                       "lambda_function.py"))
        cls.gw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gw)

    def setUp(self):
        self.sfn = mock.MagicMock()
        self.sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:eu-west-1:1:execution:sm:x"}
        self.s3 = mock.MagicMock()
        for p in (mock.patch.object(self.gw, "_sfn", self.sfn),
                  mock.patch.object(self.gw, "_s3", self.s3)):
            p.start()
            self.addCleanup(p.stop)

    def submit(self, **overrides):
        event = {"transcript_s3uri": TRANSCRIPT_URI}
        event.update(overrides)
        return self.gw._analyze_conversation(event)


class TestGatewayIdentity(_GatewayCase):
    def test_stable_across_key_order_and_sensitive_to_parameters(self):
        a = self.submit()
        b = self.submit()
        self.assertEqual(a["workflow_id"], b["workflow_id"])
        c = self.submit(model_chain=["some-model"])
        self.assertNotEqual(a["workflow_id"], c["workflow_id"])
        d = self.submit(idempotency_token="rerun-1")  # nosec B106
        self.assertNotEqual(a["workflow_id"], d["workflow_id"])

    def test_chain_string_and_list_are_one_identity(self):
        a = self.submit(model_chain="m-1, m-2")
        b = self.submit(model_chain=["m-1", "m-2"])
        self.assertEqual(a["workflow_id"], b["workflow_id"])

    def test_prompt_version_enters_the_hash_and_absent_means_default(self):
        """prompt_version selects a NAMED registry version — different
        version, different work, so it must move the workflow_id; absent
        must hash exactly as before the knob existed."""
        base = self.submit()
        news = self.submit(prompt_version="default_news")
        self.assertNotEqual(base["workflow_id"], news["workflow_id"])
        sfn_input = json.loads(
            self.sfn.start_execution.call_args.kwargs["input"])
        self.assertEqual(sfn_input["input"]["prompt_version"],
                         "default_news")
        blank = self.submit(prompt_version="")
        self.assertEqual(base["workflow_id"], blank["workflow_id"])

    def test_retired_uuid_is_rejected_on_submit(self):
        res = self.submit(uuid="x")
        self.assertEqual(res["errorCode"], "INVALID_PARAMETER")
        self.sfn.start_execution.assert_not_called()

    def test_missing_uri_names_what_to_supply(self):
        res = self.gw._analyze_conversation({})
        self.assertEqual(res["errorCode"], "MISSING_PARAMETER")
        self.assertIn("final_transcript.json", res["message"])

    def test_submit_returns_a_handle_never_results(self):
        res = self.submit()
        self.assertEqual(set(res), {"job_id", "status", "workflow_id",
                                    "output_s3_prefix"})

    def test_absorbed_resubmit_carries_the_pinned_message(self):
        from botocore.exceptions import ClientError
        self.sfn.start_execution.side_effect = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "t"}},
            "StartExecution")
        self.sfn.describe_execution.return_value = {"status": "SUCCEEDED"}
        res = self.submit()
        self.assertIn("identical request already submitted", res["message"])
        self.assertEqual(res["status"], "SUCCEEDED")


class TestGatewayStatus(_GatewayCase):
    ARN = ("arn:aws:states:eu-west-1:123456789012:execution:"
           "pfx-conversation-analysis:wf-1")

    def test_succeeded_lifts_header_and_lists_the_three_labels(self):
        self.sfn.describe_execution.return_value = {"status": "SUCCEEDED"}
        analysis = {"overall_content_type": "Episodic",
                    "top_3_genres": ["Drama"], "audio_segments": [{}, {}],
                    "segments_gap": []}
        self.s3.get_object.return_value = {
            "Body": io.BytesIO(json.dumps(analysis).encode())}
        res = self.gw._get_job_status({"job_id": self.ARN})
        self.assertEqual(res["segments"], 2)
        self.assertEqual([o["label"] for o in res["outputs"]],
                         ["analysis", "chunk-manifest", "usage"])
        for entry in res["outputs"]:
            self.assertIn("desc", entry)     # joined from the contract
            self.assertIn("wf-1", entry["uri"])

    def test_running_reports_current_state(self):
        self.sfn.describe_execution.return_value = {"status": "RUNNING"}
        self.sfn.get_execution_history.return_value = {"events": [
            {"stateEnteredEventDetails": {"name": "Submit and wait"}}]}
        res = self.gw._get_job_status({"job_id": self.ARN})
        self.assertEqual(res["current_state"], "Submit and wait")
        self.assertNotIn("outputs", res)

    def test_failed_carries_the_cause(self):
        self.sfn.describe_execution.return_value = {"status": "FAILED"}
        self.sfn.get_execution_history.return_value = {"events": [
            {"executionFailedEventDetails": {"cause": "boom"}}]}
        res = self.gw._get_job_status({"job_id": self.ARN})
        self.assertEqual(res["error"], "boom")

    def test_describe_tool_serves_the_contract_and_no_lies(self):
        res = self.gw.lambda_handler(
            {}, mock.MagicMock(client_context=mock.MagicMock(custom={
                "bedrockAgentCoreToolName":
                    "conversation-analysis___describe_tool"})))
        self.assertIn("output_contract", res)
        self.assertEqual(res["output_contract"]["target"],
                         "conversation-analysis")
        self.assertIn("state_machine_arn", res)

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            examples = json.load(f)["tools"]
        self.assertEqual(sorted(examples),
                         ["analyze_conversation", "describe_tool",
                          "get_job_status"])
        for tool, entry in examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        res = self.gw.lambda_handler(
            {}, mock.MagicMock(client_context=mock.MagicMock(custom={
                "bedrockAgentCoreToolName":
                    "conversation-analysis___describe_tool"})))
        self.assertIn("analyze_conversation", res["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            example = json.load(f)["tools"]["analyze_conversation"]["example"]
        self.sfn.start_execution.return_value = {"executionArn": "arn:fake"}
        res = self.gw.lambda_handler(
            dict(example),
            mock.MagicMock(client_context=mock.MagicMock(custom={
                "bedrockAgentCoreToolName":
                    "conversation-analysis___analyze_conversation"})))
        self.assertNotIn("error", res, f"example refused: {res}")
        self.assertEqual(res["status"], "RUNNING")
        self.sfn.start_execution.assert_called_once()


class TestTemplateAndBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml

        class L(yaml.SafeLoader):
            pass

        def unknown(loader, suffix, node):
            if isinstance(node, yaml.ScalarNode):
                return loader.construct_scalar(node)
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            return loader.construct_mapping(node)
        L.add_multi_constructor("!", unknown)
        with open(os.path.join(ROOT, "cfnstack.yaml")) as f:
            cls.tpl = yaml.load(f, Loader=L)  # nosec B506
        with open(os.path.join(ROOT, "build-deploy-s3-dist.sh")) as f:
            cls.build = f.read()

    def test_inline_asl_equals_the_readability_copy(self):
        inline = json.loads(
            self.tpl["Resources"]["ConversationAnalysisStateMachine"]
            ["Properties"]["DefinitionString"])
        with open(os.path.join(ROOT, "definitions.json")) as f:
            self.assertEqual(inline, json.load(f))

    def test_the_machine_is_five_linear_states(self):
        """The 2026-08-18 simplification, pinned: waitForTaskToken with a
        26 h timeout and NO Catch (a timed-out wait FAILS; a redrive of the
        submit state absorbs into the finished run and is woken
        synchronously — redrive IS the reconcile), no retry loop (the
        caller's fresh token is the semantic retry), no Wait, no Choice, no
        Fail states anywhere."""
        asl = json.loads(
            self.tpl["Resources"]["ConversationAnalysisStateMachine"]
            ["Properties"]["DefinitionString"])
        self.assertEqual(list(asl["States"]), [
            "Validate and plan", "Render and stage", "Submit and wait",
            "Collect and finalize", "Usage tracking"])
        wait = asl["States"]["Submit and wait"]
        self.assertTrue(wait["Resource"].endswith(
            "lambda:invoke.waitForTaskToken"))
        self.assertEqual(wait["TimeoutSeconds"], 93600)
        self.assertNotIn("Catch", wait)
        self.assertEqual({s["Type"] for s in asl["States"].values()},
                         {"Task"})

    def test_no_state_machine_starts_and_no_ddb(self):
        """The consumer never touches the siblings' state machines (§8) and
        owns no table — invoke + S3 is the whole surface."""
        text = json.dumps(self.tpl)
        self.assertNotIn("states:StartExecution\", \"Resource\": \"arn",
                         text)
        workflow_policy = json.dumps(
            self.tpl["Resources"]["WorkflowRole"]["Properties"])
        self.assertNotIn("states:", workflow_policy)
        self.assertNotIn("AWS::DynamoDB::Table",
                         {r["Type"] for r in self.tpl["Resources"].values()})

    def test_log_groups_use_the_log_suffix_and_logging_config(self):
        for name, res in self.tpl["Resources"].items():
            if res["Type"] == "AWS::Lambda::Function":
                self.assertNotIn("FunctionName", res["Properties"], name)
                self.assertIn("LoggingConfig", res["Properties"], name)
        for role in ("GatewayToolRole", "WorkflowRole"):
            policy = json.dumps(self.tpl["Resources"][role]["Properties"])
            self.assertNotIn("AWSLambdaBasicExecutionRole", policy, role)
            self.assertNotIn("logs:CreateLogGroup", policy, role)

    def test_conditioned_listbucket_on_both_roles(self):
        """The S3 403-vs-404 lesson, pinned here the way prompt-registry
        pinned it after the live failure."""
        for role in ("GatewayToolRole", "WorkflowRole"):
            policy = json.dumps(self.tpl["Resources"][role]["Properties"])
            self.assertIn("s3:ListBucket", policy, role)

    def test_workflow_role_invokes_exactly_the_two_siblings(self):
        policy = self.tpl["Resources"]["WorkflowRole"]["Properties"][
            "Policies"][0]["PolicyDocument"]["Statement"]
        invoke = [s for s in policy
                  if "lambda:InvokeFunction" in json.dumps(s)]
        self.assertEqual(len(invoke), 1)
        self.assertIn("BedrockInferenceGatewayLambdaArn",
                      json.dumps(invoke[0]))
        self.assertIn("PromptRegistryGatewayLambdaArn",
                      json.dumps(invoke[0]))

    def test_build_guards_every_member(self):
        for member in ("lambda_function.py", "workflow_identity.py",
                       "output_contract.json", "lib/ca_common.py",
                       "lib/chunker.py",
                       "states/submit_history/lambda_function.py",
                       "states/collect_finalize/lambda_function.py",
                       "lib/prompt_render.py", "lib/workflow_history.py"):
            self.assertIn(member, self.build)
        self.assertIn('[ ! -f "gateway_tool/$f" ]', self.build)
        self.assertIn("verify_sfn_definition", self.build)

    def test_token_guard_is_digit_tolerant(self):
        self.assertIn("%%[A-Z0-9_]*%%", self.build)


class TestOutputContractFile(unittest.TestCase):
    def test_labels_match_the_gateway_and_the_states(self):
        with open(os.path.join(ROOT, "gateway_tool",
                               "output_contract.json")) as f:
            contract = json.load(f)
        self.assertEqual(set(contract["outputs"]),
                         {"analysis", "chunk-manifest", "usage"})
        for label, entry in contract["outputs"].items():
            self.assertEqual(entry["kind"], "file", label)
            self.assertIn("basename", entry, label)
            self.assertIn("example", entry, label)


class TestOverlappedAttribution(unittest.TestCase):
    """discourse_facts inside cross-talk spans gain a recorded
    ambiguity (attribution_overlapped_with) — never a re-attribution,
    and only when the span involves the attributed speaker."""

    def test_annotation(self):
        import merge
        sentences = [
            {"id": 0, "start": 0.0, "end": 2.0,
             "speaker_id": "SPEAKER_00", "text": "a"},
            {"id": 1, "start": 2.0, "end": 4.0,
             "speaker_id": "SPEAKER_01", "text": "b"},
        ]
        analysis = {"audio_segments": [{"discourse_facts": [
            {"kind": "x", "cue_id": 0, "quote": "a"},
            {"kind": "y", "cue_id": 1, "quote": "b"},
        ]}]}
        n = merge.annotate_overlapped_attribution(
            analysis, sentences,
            [{"start": 1.5, "end": 2.5,
              "speakers": ["SPEAKER_00", "SPEAKER_02"]}])
        facts = analysis["audio_segments"][0]["discourse_facts"]
        self.assertEqual(n, 1)
        self.assertEqual(facts[0]["attribution_overlapped_with"],
                         ["SPEAKER_02"])
        # cue 1's span intersects but its SPEAKER is not in the pair
        self.assertNotIn("attribution_overlapped_with", facts[1])

    def test_no_segments_is_a_noop(self):
        import merge
        analysis = {"audio_segments": [{"discourse_facts": [
            {"kind": "x", "cue_id": 0, "quote": "a"}]}]}
        self.assertEqual(merge.annotate_overlapped_attribution(
            analysis, [], []), 0)

    def test_loader_carries_overlap_segments(self):
        import json as _json
        from transcript import load_transcript
        doc = {"sentences": [{"id": 0, "start": 0, "end": 1,
                              "speaker_id": "SPEAKER_00", "text": "hi"}],
               "overlap_segments": [{"start": 0.2, "end": 0.6,
                                     "speakers": ["SPEAKER_00",
                                                  "SPEAKER_01"]}]}
        out = load_transcript(_json.dumps(doc))
        self.assertEqual(len(out["overlap_segments"]), 1)
        out2 = load_transcript(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v A>hi\n")
        self.assertEqual(out2["overlap_segments"], [])
