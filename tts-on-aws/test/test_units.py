"""
Unit tests — tts-on-aws invariants that a code review had to re-derive by
hand. Every test here corresponds to a real finding: the bug it would have
caught is named in the test's docstring, so the next review can trust the
invariant instead of re-measuring it.

Mocked boto3 only; no AWS calls, no model loads. The heavy ML libraries
(onnxruntime, faster_whisper/ctranslate2) are NOT imported — modules that
need them are loaded with importlib from their file path so they get their
module-level constants without pulling the container's dependency tree.

Run: cd tts-on-aws && python3 -m unittest discover -s test
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# gateway_tool/ goes on the path too because the gateway zip is FLAT (`zip -j`):
# at runtime `workflow_identity` is a top-level module, not `gateway_tool.
# workflow_identity`. Loading lambda_function.py by file path without this makes
# its own import fail here while working perfectly in Lambda.
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))

os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("MODEL_BUCKET", "test-proxy")
os.environ.setdefault("VOICE_TABLE_NAME", "test-voices")
os.environ.setdefault("SYNTHESIS_STATE_MACHINE_ARN",
                      "arn:aws:states:us-east-1:123456789012:stateMachine:m2c-tts")
os.environ.setdefault(
    "REGISTER_VOICE_STATE_MACHINE_ARN",
    "arn:aws:states:us-east-1:123456789012:stateMachine:m2c-tts-register")


def _client_error(code: str, op: str = "Op"):
    from botocore.exceptions import ClientError
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


def _load(relpath: str, name: str):
    """Import a module from its path, bypassing package __init__ chains."""
    spec = importlib.util.spec_from_file_location(name,
                                                  os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_NO_VOICE = object()


def _load_check_and_preprocess(candidates_for=None, voice=_NO_VOICE):
    """StateCheckAndPreprocess with lib.style_hints/lib.voice_store stubbed.
    voice=None models an unknown voice_id (get_voice returns None)."""
    lib = types.ModuleType("lib")
    lib.__path__ = []
    sh = types.ModuleType("lib.style_hints")
    sh.ALL_STYLES = {"fast": "t", "happy": "t"}
    sh.candidates_for = candidates_for or (lambda s: 1 if s is None else 3)
    vs = types.ModuleType("lib.voice_store")
    default = {"language": "en"}
    vs.get_voice = lambda v: (default if voice is _NO_VOICE else voice)
    lib.style_hints, lib.voice_store = sh, vs
    with mock.patch.dict(sys.modules, {"lib": lib, "lib.style_hints": sh,
                                       "lib.voice_store": vs}):
        return _load("states/check_and_preprocess/lambda_function.py", "cp")


# ─────────────────────────────────────────────────────────────────────────────
# Step Functions 256 KiB payload ceiling
#
# README.md §10 required that "whatever default [transcripts_per_batch] gets
# must be checked against this ceiling, not assumed safe by extrapolation".
# MAX_TRANSCRIPTS_PER_BATCH=100 alone does NOT satisfy that: measured with
# real json.dumps envelopes, 100 transcripts blows 256 KiB at ~234 chars
# best-of-1, and the true limits are ~24 (1000 chars, best-of-1) and ~8
# (1000 chars, best-of-3). Without the estimator the workflow paid for a
# full LLM decode and then died on States.DataLimitExceeded.
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


class TestPayloadCeiling(unittest.TestCase):
    def setUp(self):
        self.cp = _load_check_and_preprocess()

    def test_budget_is_under_the_hard_limit(self):
        self.assertEqual(self.cp.SFN_PAYLOAD_LIMIT_BYTES, 256 * 1024)
        self.assertLess(self.cp.SFN_PAYLOAD_BUDGET_BYTES,
                        self.cp.SFN_PAYLOAD_LIMIT_BYTES)

    def test_estimator_rejects_every_measured_over_limit_shape(self):
        """(text_chars, n_candidates, count) combinations measured to exceed
        256 KiB as real JSON must all estimate over budget."""
        over = [(234, 1, 99), (234, 3, 36), (500, 1, 47),
                (1000, 1, 24), (1000, 3, 8), (100, 3, 82)]
        for chars, n, count in over:
            est = self.cp.estimate_batch_payload_bytes(["x" * chars] * count, n)
            self.assertGreater(est, self.cp.SFN_PAYLOAD_BUDGET_BYTES,
                               f"{count}x{chars}ch n={n} should be rejected")

    def test_estimator_accepts_a_realistic_safe_batch(self):
        est = self.cp.estimate_batch_payload_bytes(["x" * 100] * 50, 1)
        self.assertLess(est, self.cp.SFN_PAYLOAD_BUDGET_BYTES)

    def test_estimate_grows_with_candidate_count(self):
        """Best-of-3 must cost visibly more than best-of-1 — an estimator
        that ignored n_candidates would pass the batch gate and then blow
        the real limit on emotion-tier requests."""
        one = self.cp.estimate_batch_payload_bytes(["x" * 200] * 30, 1)
        three = self.cp.estimate_batch_payload_bytes(["x" * 200] * 30, 3)
        self.assertGreater(three, one * 2)

    def test_map_aggregate_is_bounded_across_batches(self):
        """Each batch can be individually legal while the Map's aggregate
        ResultPath still exceeds 256 KiB — the per-batch gate alone is not
        sufficient. It is also the ONE bound splitting a batch into more
        iterators cannot relieve: the total is the sum over every transcript in
        the request, however they are grouped."""
        one = ["x" * 200] * 100
        self.assertLess(self.cp.estimate_map_aggregate_bytes(one),
                        self.cp.SFN_PAYLOAD_BUDGET_BYTES)
        self.assertGreater(self.cp.estimate_map_aggregate_bytes(one * 6),
                           self.cp.SFN_PAYLOAD_BUDGET_BYTES)

    def test_aggregate_estimate_never_undercounts_a_real_envelope(self):
        """The gate must be pessimistic: over-estimating rejects a request the
        caller can resize, while under-estimating pays for a full synthesis and
        then dies on States.DataLimitExceeded with the audio already in S3.

        This builds the actual post-QC ResultPath json.dumps sees and compares.
        The expensive entry is a clip whose QC could not RUN — it keeps
        `qc.transcribed` AND carries a `qc.error` string — which is what
        BYTES_PER_OUTPUT_OVERHEAD is calibrated against, not the typical
        passing entry. 100 % failure is a real workload, not a pathological
        one: a wrong `language` fails every clip in the request.

        **The `voice` record in the envelope is load-bearing and was missing
        from this fixture until 2026-08-10.** That omission is exactly why
        BYTES_PER_ITERATOR_ENVELOPE shipped at 380 when the real figure measured
        751 against live execution 5512a1af: `voice` is the largest member at
        334 bytes because it carries `ref_transcript` verbatim, and nothing
        bounds that string (363 chars is the longest in the live catalog). A
        fixture that models the payload more cheaply than production is worse
        than no fixture — it certifies the wrong number."""
        cp = self.cp
        for chars in (5, 42, 230, 1000):
            for n in (1, 40, 100, 300):
                for fail_every, err in ((0, False), (20, False), (1, False),
                                        (1, True)):
                    iters, i = [], 0
                    n_per = max(1, -(-n // cp.MAP_MAX_CONCURRENCY))
                    while i < n:
                        outs = []
                        for j in range(min(n_per, n - i)):
                            failed = fail_every and (i + j) % fail_every == 0
                            q = {"passed": not failed, "similarity": 0.412,
                                 "threshold": 0.85, "detected_language": "en"}
                            if failed:
                                q["transcribed"] = "y" * chars
                                if err:
                                    q["error"] = ("RuntimeError: whisper "
                                                  "crashed on a corrupt frame")
                            outs.append({
                                "wav_key": f"micro-workflow/tts-on-aws/output/"
                                           f"630f729f-59b0-57f2-8bcb-"
                                           f"6a42ecedc987/0123456789ab_{i}_"
                                           f"{j}_c0.wav",
                                "duration_s": 4.234, "stretched": False,
                                "seed": 1234, "qc": q})
                        iters.append({"item_id": len(iters), "item_data": {
                            "job_id": "0123456789ab-aaaa",
                            "batch_index": len(iters), "source_batch_index": 0,
                            "voice_id": "librispeech-1234",
                            "out_bucket": "umt-proxy-eu-west-1-111122223333",
                            "out_prefix": "micro-workflow/tts-on-aws/output/"
                                          "630f729f-59b0-57f2-8bcb-6a42ecedc987",
                            "config": {"flow_steps": 6, "sampling_k": 25,
                                       "seed": None, "style": None},
                            # The member the old fixture omitted. ref_transcript
                            # at the live catalog's longest (363 chars).
                            "voice": {
                                "voice_id": "librispeech-1234",
                                "language": "en",
                                "ref_transcript": "w" * 363,
                                "s3_conditioning_key":
                                    "_model_artefacts/voice_catalog/default/"
                                    "a616655a-50b3-462a-b7ad-fa1a90ac4baa/"
                                    "conditioning.npz"},
                            "outputs": outs}})
                        i += min(n_per, n - i)
                    real = len(json.dumps(iters))
                    est = cp.estimate_map_aggregate_bytes(["x" * chars] * n,
                                                          len(iters))
                    self.assertGreaterEqual(
                        est, real,
                        f"{n} x {chars}ch, fail_every={fail_every} err={err}: "
                        f"estimate {est} UNDER the real {real} bytes")

    def test_dropping_the_redundant_text_copies_raised_the_ceiling(self):
        """The aggregate used to carry each text THREE times — transcripts[],
        qc.expected, qc.transcribed — making it 15.9x larger than the
        information in it required and capping a request at 63 transcripts at
        MAX_TEXT_CHARS. verify_output now drops the first two outright and keeps
        transcribed only on failures, so the multiplier is 1. Pinning the
        ceiling because the regression is silent: re-adding a text copy makes
        long-text requests start failing a gate that used to pass them.

        The floors sit below the measured ceilings (374 / 263 / 118 at 42 / 230
        / 1000 chars) rather than equal to them — this test guards the 3x
        regression, and pinning exact values would make it fail every time the
        envelope terms are honestly re-measured, which is what happened twice on
        2026-08-10: BYTES_PER_ITERATOR_ENVELOPE went 380 -> 1100 once the live
        envelope was measured and the unbounded ref_transcript accounted for.
        The 3x shape capped 1000-char requests at 63, so the floor at that
        length is what actually matters and it is still ~1.9x better."""
        cp = self.cp
        self.assertEqual(cp.TEXT_COPIES_IN_AGGREGATE, 1)

        def ceiling(chars):
            n = 1
            while cp.estimate_map_aggregate_bytes(
                    ["x" * chars] * n, min(cp.MAP_MAX_CONCURRENCY, n)) <= \
                    cp.SFN_PAYLOAD_BUDGET_BYTES:
                n += 1
            return n - 1

        # Was 515 / 215 / 63 under the 3x shape. The long-text case is the one
        # that was actually binding and it is still ~1.9x better.
        self.assertGreater(ceiling(1000), 110)
        self.assertGreater(ceiling(230), 250)
        self.assertGreater(ceiling(42), 360)

    def test_iterator_envelope_covers_the_measured_live_envelope(self):
        """Pins the figure measured on live execution 5512a1af (eu-west-1,
        2026-08-10): 751 bytes per iterator, of which `voice` is 334 because it
        carries `ref_transcript` verbatim and nothing bounds that string.

        This exists because the constant shipped at 380 — derived from a
        hand-built envelope with `voice` left out — and the error was INVISIBLE
        in aggregate: BYTES_PER_OUTPUT_OVERHEAD ran ~96 bytes high per entry, so
        on the reference run's 73-clip shape the two errors nearly cancelled and
        the total landed 1.1 % low. Low is the direction that pays for a full
        synthesis and then dies on States.DataLimitExceeded. Two independently
        wrong terms that happen to sum right are not calibrated, so each is
        pinned separately here."""
        cp = self.cp
        MEASURED_LIVE_ENVELOPE = 751
        self.assertGreaterEqual(
            cp.BYTES_PER_ITERATOR_ENVELOPE, MEASURED_LIVE_ENVELOPE,
            "the per-iterator envelope must cover what production actually "
            "emits; 5512a1af measured 751 bytes")
        # A longer ref_transcript than the live catalog's worst must still fit:
        # 363 chars is today's longest of 46 voices, and register_voice does not
        # bound it, so the headroom is the only thing standing between a chatty
        # reference clip and a failed job.
        self.assertGreaterEqual(
            cp.BYTES_PER_ITERATOR_ENVELOPE,
            MEASURED_LIVE_ENVELOPE + (363 - 112),
            "no headroom for a ref_transcript longer than the reference "
            "voice's 112 chars")
        # And the per-entry term stays above the worst real entry (324 bytes:
        # a failed clip keeping qc.transcribed) without absorbing the envelope.
        self.assertGreaterEqual(cp.BYTES_PER_OUTPUT_OVERHEAD, 324)
        self.assertLess(cp.BYTES_PER_OUTPUT_OVERHEAD, MEASURED_LIVE_ENVELOPE,
                        "per-entry overhead has grown large enough to be "
                        "hiding the per-iterator envelope again")

    def test_aggregate_estimate_scales_with_the_real_iterator_count(self):
        """The gate runs AFTER the split, so it passes the real iterator count
        rather than assuming MAP_MAX_CONCURRENCY. A 3-transcript request creates
        3 iterators, not 40, and charging it for 40 envelopes (~36 KiB, 17 % of
        budget) would reject small requests for payload they never produce.
        Omitting the argument must still fall back to the worst case."""
        cp = self.cp
        texts = ["x" * 42] * 3
        self.assertLess(cp.estimate_map_aggregate_bytes(texts, 3),
                        cp.estimate_map_aggregate_bytes(texts))
        self.assertEqual(cp.estimate_map_aggregate_bytes(texts),
                         cp.estimate_map_aggregate_bytes(
                             texts, cp.MAP_MAX_CONCURRENCY))

    def test_handler_rejects_oversize_batch(self):
        cp = self.cp
        event = {"uuid": "u1", "input": {"batches": [
            {"voice_id": "v1",
             "transcripts": [{"text": "x" * 900} for _ in range(40)]}]}}
        with self.assertRaises(ValueError) as ctx:
            cp.lambda_handler(event, None)
        self.assertIn("KiB", str(ctx.exception))

    def test_handler_rejects_oversize_aggregate_of_legal_batches(self):
        """6 x 80 transcripts @ 200 chars: each batch is well under the
        per-batch budget, the Map aggregate is not.

        This was 3 x 80 until the redundant text copies were removed from the
        aggregate — that shape is now genuinely legal and passing it is the
        point of the change, so the test needs a request that crosses the new,
        higher ceiling rather than the old one."""
        cp = self.cp
        batch = [{"text": "x" * 200} for _ in range(80)]
        self.assertLess(cp.estimate_batch_payload_bytes(
            [t["text"] for t in batch], 1), cp.SFN_PAYLOAD_BUDGET_BYTES)
        event = {"uuid": "u1", "input": {"batches": [
            {"voice_id": "v1", "transcripts": [dict(t) for t in batch]}
            for _ in range(6)]}}
        with self.assertRaises(ValueError) as ctx:
            cp.lambda_handler(event, None)
        self.assertIn("Map", str(ctx.exception))

    def test_handler_accepts_a_normal_request(self):
        cp = self.cp
        event = {"uuid": "u1", "input": {"batches": [
            {"voice_id": "v1", "transcripts": [{"text": "hello world"}]}]}}
        out = cp.lambda_handler(event, None)
        it = out["data"]["iterators"][0]
        self.assertEqual(it["voice_id"], "v1")
        self.assertEqual(it["batch_index"], 0)
        self.assertEqual(it["out_bucket"], "test-proxy")

    def test_explicit_seed_forces_single_shot_in_the_estimate(self):
        """The gate's n_candidates must mirror StateSynthesizeLLM's own seed
        logic: an explicit caller seed means one candidate REGARDLESS of
        style. If the gate assumed 3 it would reject batches the pipeline
        would actually have handled fine."""
        cp = self.cp
        transcripts = [{"text": "x" * 250} for _ in range(30)]
        styled = {"uuid": "u", "input": {
            "batches": [{"voice_id": "v", "transcripts": transcripts}],
            "config": {"style": "happy"}}}
        with self.assertRaises(ValueError):
            cp.lambda_handler(json.loads(json.dumps(styled)), None)
        seeded = json.loads(json.dumps(styled))
        seeded["input"]["config"]["seed"] = 42
        cp.lambda_handler(seeded, None)  # must NOT raise


# ─────────────────────────────────────────────────────────────────────────────
# verify_output's aggregate pruning
#
# The other half of the ceiling change above: the estimator only predicts what
# this state actually emits. If the two drift apart the gate is wrong in
# whichever direction the drift went, and the expensive direction
# (under-estimating) fails a job after every clip has been synthesized and paid
# for.
# ─────────────────────────────────────────────────────────────────────────────

def _load_verify_output():
    """StateVerifyOutput with lib.qc / lib.model_store stubbed — importing it
    for real pulls faster-whisper and ctranslate2."""
    lib = types.ModuleType("lib")
    lib.__path__ = []
    ck = _load("lib/checkpoint.py", "vo_ckpt")
    q = types.ModuleType("lib.qc")
    q.verify_content = lambda *a, **k: {}
    ms = types.ModuleType("lib.model_store")
    ms.ensure_in_tmp = lambda *a, **k: None
    ce = _load("lib/custom_exceptions.py", "vo_exc")
    lib.checkpoint, lib.qc, lib.model_store = ck, q, ms
    lib.custom_exceptions = ce
    mods = {"lib": lib, "lib.checkpoint": ck, "lib.qc": q,
            "lib.model_store": ms, "lib.custom_exceptions": ce}
    with mock.patch.dict(sys.modules, mods), \
            mock.patch("boto3.client", return_value=mock.MagicMock()):
        return _load("states/verify_output/lambda_function.py", "vo")


class TestVerifyOutputPrunesTheAggregate(unittest.TestCase):
    """The Map's aggregate ResultPath carries every iterator's terminal output
    in ONE 256 KiB payload, so it is the sum over the whole request and no
    amount of splitting relieves it. Each text used to appear three times in
    there; this state now emits one. estimate_map_aggregate_bytes gates on that
    assumption, so these tests hold the emitter to it."""

    def setUp(self):
        self.vo = _load_verify_output()

    def test_expected_is_always_dropped(self):
        """qc.expected is a byte-for-byte copy of the caller's own input, which
        the caller joins back by transcript_index. Nothing reads it — not the
        gateway, not usage_tracking — so it is pure duplication."""
        for passed in (True, False, None):
            e = {"qc": {"passed": passed, "expected": "hello there",
                        "transcribed": "hello there", "similarity": 0.9}}
            self.vo._prune_qc_evidence(e)
            self.assertNotIn("expected", e["qc"])

    def test_transcribed_survives_on_failures_and_goes_on_passes(self):
        """qc.transcribed is the only field here that is NOT derivable — it is
        what whisper actually heard, i.e. the evidence for a QC failure, which
        README.md calls "a review pointer, not an error". Dropping it on a
        FAILED clip would delete the reason the clip is flagged. On a passing
        clip it is, by definition of the pass threshold, a near-copy of the
        expected text and carries nothing."""
        failed = {"qc": {"passed": False, "similarity": 0.41,
                         "transcribed": "hello dere"}}
        self.vo._prune_qc_evidence(failed)
        self.assertEqual(failed["qc"]["transcribed"], "hello dere")

        errored = {"qc": {"passed": None, "similarity": -1.0,
                          "transcribed": "", "error": "RuntimeError: boom"}}
        self.vo._prune_qc_evidence(errored)
        self.assertIn("transcribed", errored["qc"])
        self.assertIn("error", errored["qc"])

        ok = {"qc": {"passed": True, "similarity": 0.98,
                     "transcribed": "hello there"}}
        self.vo._prune_qc_evidence(ok)
        self.assertNotIn("transcribed", ok["qc"])

    def test_fields_the_gateway_reads_are_never_pruned(self):
        """_parse_succeeded_output reads passed / similarity / error. Pruning
        any of those turns a QC result into "no QC result" at the API."""
        e = {"qc": {"passed": False, "similarity": 0.41, "threshold": 0.85,
                    "detected_language": "en", "expected": "a",
                    "transcribed": "b", "error": "x"}}
        self.vo._prune_qc_evidence(e)
        for k in ("passed", "similarity", "error", "threshold",
                  "detected_language"):
            self.assertIn(k, e["qc"], f"{k} must survive pruning")

    def test_prune_tolerates_entries_with_no_qc(self):
        """An entry that errored upstream in synthesize_flow_hift has no qc key
        at all — pruning must not raise on it, because an exception escaping
        this state discards audio already durably in S3 (the ASL has no
        States.TaskFailed retry on "Verify output")."""
        for e in ({}, {"error": "flow failed"}, {"qc": None}, {"qc": {}}):
            self.vo._prune_qc_evidence(e)  # must not raise

    def _run(self, texts, similarity=0.99, verify=None):
        """Drive the real handler over `texts`, stubbing S3 and whisper."""
        item = {
            "job_id": "j1", "batch_index": 0, "out_bucket": "b",
            "out_prefix": "p", "voice": {"language": "en"},
            "transcripts": [{"text": t} for t in texts],
            "outputs": [{"candidates": [
                {"wav_key": f"p/{i}.wav", "duration_s": 1.0}]}
                for i in range(len(texts))],
        }
        ctx = mock.Mock()
        ctx.get_remaining_time_in_millis.return_value = 900_000
        self.vo.qc.verify_content = verify or (lambda path, text, *a, **k: {
            "passed": similarity >= 0.85, "similarity": similarity,
            "threshold": 0.85, "detected_language": "en",
            "expected": text, "transcribed": text,
        })
        with mock.patch.object(self.vo.checkpoint, "load", return_value=None), \
                mock.patch.object(self.vo.checkpoint, "save"), \
                mock.patch.object(self.vo.checkpoint, "clear"), \
                mock.patch("os.path.exists", return_value=False):
            return self.vo.lambda_handler({"item_data": item}, ctx)

    def test_handler_output_carries_each_text_at_most_once(self):
        """The end-to-end property the estimator encodes
        (TEXT_COPIES_IN_AGGREGATE == 1). Counting occurrences of the text in the
        serialized state output is the only check that catches a THIRD place
        re-introducing a copy — a per-field assertion would not."""
        text = "the quick brown fox jumps over the lazy dog"
        for sim, label in ((0.99, "passing"), (0.41, "failing")):
            out = self._run([text] * 4, similarity=sim)
            blob = json.dumps(out)
            self.assertLessEqual(blob.count(text), 4,
                                 f"{label}: text duplicated in the aggregate")

    def test_handler_drops_the_input_transcripts(self):
        """Nothing downstream reads item["transcripts"] — the LLM and Flow/HiFT
        states consume it before here, the gateway joins by transcript_index,
        and usage_tracking never touches item_data. It is the second of the
        three copies."""
        out = self._run(["hello world", "goodbye world"])
        self.assertNotIn("transcripts", out["item_data"])

    def test_handler_still_qcs_against_the_expected_text(self):
        """Dropping transcripts must not cost the QC pass the text it matches
        against. This holds regardless of where the pop sits, because
        `transcripts` is bound to a local before it — the point is the failure
        mode: verify_content's exceptions are swallowed per-clip, so a lost
        expected text would come back as 73 qc.error entries and a SUCCEEDED
        execution, not as a workflow failure anyone would notice."""
        seen = []
        out = self._run(["alpha", "beta"], verify=lambda path, text, *a, **k: (
            seen.append(text) or {"passed": True, "similarity": 0.99,
                                  "expected": text, "transcribed": text}))
        self.assertEqual(seen, ["alpha", "beta"])
        for o in out["item_data"]["outputs"]:
            self.assertNotIn("error", o.get("qc", {}))

    def test_handler_output_keeps_what_the_gateway_needs(self):
        """Pruning must not cost the aggregate its actual purpose: the gateway
        builds each result entry from wav_key + qc.passed/similarity."""
        out = self._run(["hello world"])
        entry = out["item_data"]["outputs"][0]
        self.assertEqual(entry["wav_key"], "p/0.wav")
        self.assertIs(entry["qc"]["passed"], True)
        self.assertAlmostEqual(entry["qc"]["similarity"], 0.99)
        self.assertNotIn("candidates", entry)

    def test_a_fully_failed_run_stays_inside_the_estimated_budget(self):
        """The worst case the estimator budgets for: every clip fails, so every
        clip keeps its transcribed copy. If the real envelope exceeded the
        estimate here, a request that passed the gate would blow the 256 KiB
        limit at the Map's aggregate — after paying for all the synthesis."""
        cp = _load_check_and_preprocess()
        # 260 x 230 chars, just inside the measured 275-transcript ceiling.
        texts = ["x" * 230] * 260
        est = cp.estimate_map_aggregate_bytes(texts, cp.MAP_MAX_CONCURRENCY)
        self.assertLess(est, cp.SFN_PAYLOAD_BUDGET_BYTES)
        out = self._run(texts, similarity=0.41)
        real = len(json.dumps(out["item_data"]["outputs"]).encode())
        self.assertLess(real, est, "real aggregate exceeds the gate's estimate")


class TestCheckAndPreprocessValidation(unittest.TestCase):
    def test_style_rejected_for_unverified_language(self):
        """README.md §7a: style_hint measurably corrupts content on non-English
        voices (QC 0.95-1.0 → 0.44-0.67). The gate compares the voice's
        persisted `language` for EQUALITY against {"en"} — which is why the
        register path must normalize it (see TestLanguageNormalization)."""
        cp = _load_check_and_preprocess(voice={"language": "de"})
        event = {"uuid": "u", "input": {
            "batches": [{"voice_id": "v", "transcripts": [{"text": "hi"}]}],
            "config": {"style": "fast"}}}
        with self.assertRaises(ValueError) as ctx:
            cp.lambda_handler(event, None)
        self.assertIn("only verified", str(ctx.exception))

    def test_unknown_style_rejected(self):
        cp = _load_check_and_preprocess()
        event = {"uuid": "u", "input": {
            "batches": [{"voice_id": "v", "transcripts": [{"text": "hi"}]}],
            "config": {"style": "sarcastic"}}}
        with self.assertRaises(ValueError):
            cp.lambda_handler(event, None)

    def test_unknown_voice_rejected(self):
        cp = _load_check_and_preprocess(voice=None)
        # voice=None means get_voice returns None → unknown voice
        cp_mod = cp
        event = {"uuid": "u", "input": {
            "batches": [{"voice_id": "nope",
                         "transcripts": [{"text": "hi"}]}]}}
        with self.assertRaises(ValueError) as ctx:
            cp_mod.lambda_handler(event, None)
        self.assertIn("unknown", str(ctx.exception))

    def test_empty_and_oversize_text_rejected(self):
        cp = _load_check_and_preprocess()
        for transcripts in ([{"text": "   "}],
                            [{"text": "x" * (cp.MAX_TEXT_CHARS + 1)}]):
            with self.assertRaises(ValueError):
                cp.lambda_handler(
                    {"uuid": "u",
                     "input": {"batches": [{"voice_id": "v",
                                            "transcripts": transcripts}]}},
                    None)


# ─────────────────────────────────────────────────────────────────────────────
# Map iterator scheduling
#
# Execution 630f729f ran 73 transcripts in 52 min at Map concurrency 1: the
# gateway always submits exactly ONE caller batch, and this state emitted one
# iterator per batch, so 19 of the 20 available slots sat idle. The state now
# splits a batch into MAP_MAX_CONCURRENCY iterators balanced by ESTIMATED WORK
# rather than by transcript count, because a group of long sentences and a
# group of short ones are not the same job. Modelled on that run's fitted
# coefficients, count-splitting a bimodal workload takes 3.7x as long as
# work-splitting it.
#
# Three ways this silently regresses, one test each: the split stops matching
# the ASL MaxConcurrency (surplus iterators queue into a second serial wave
# that re-pays every cold model load); the tail strands a straggler that the
# whole Map then waits on; or a group is reordered, which corrupts the
# caller-facing transcript_index without raising anything.
# ─────────────────────────────────────────────────────────────────────────────

# (chars per transcript, label). The first is execution 630f729f's real input
# shape; the rest are the distributions that broke earlier versions.
_WORKLOADS = [
    ("real 73", [110, 168, 55, 213, 96, 141, 78, 187, 62, 154, 99, 132, 205,
                 71, 118, 163, 84, 149, 107, 176, 68, 128, 195, 92, 145, 113,
                 158, 74, 136, 201, 88, 122, 167, 103, 151, 79, 189, 116, 142,
                 65, 173, 97, 131, 184, 108, 156, 72, 139, 197, 85, 124, 161,
                 101, 147, 76, 181, 119, 134, 69, 170, 94, 129, 192, 106, 153,
                 81, 143, 115, 165, 90, 137, 199, 111]),
    ("uniform", [40] * 73),
    ("bimodal", [20] * 60 + [950] * 13),
    ("heavy tail", [30] * 40 + [300] * 20 + [900] * 10),
    ("few whales", [30] * 69 + [1000] * 4),
    ("all at MAX_TEXT_CHARS", [1000] * 4),
    # The worst legal batch: MAX_TRANSCRIPTS_PER_BATCH transcripts all at
    # MAX_TEXT_CHARS. 100/20 = 5 per group is ~1002 s of Flow alone, so this is
    # the only workload here that reaches pass 2's timeout cap.
    ("worst legal batch", [1000] * 100),
    ("single indivisible", [1000]),
    ("one tiny", [5]),
    ("fewer transcripts than slots", [100, 200, 300]),
]


class TestIteratorScheduling(unittest.TestCase):
    def setUp(self):
        self.cp = _load_check_and_preprocess()

    def _split(self, chars, k=None):
        ts = [{"text": "x" * c} for c in chars]
        return ts, self.cp.split_into_iterator_groups(
            ts, self.cp.MAP_MAX_CONCURRENCY if k is None else k)

    def test_map_max_concurrency_equals_the_asl_max_concurrency(self):
        """The split targets MAP_MAX_CONCURRENCY iterators; the ASL decides how
        many run at once. Set the ASL lower and the surplus iterators queue
        into a second serial wave that pays every cold model load again (48.8 s
        LLM + 35.3 s Flow per invocation) — modelled at 10.2 min against 7.0
        for a matched pair. Set it higher and slots idle. Both copies of the
        definition are checked because cfnstack.yaml is what actually deploys
        and definitions.json is what gets edited."""
        want = self.cp.MAP_MAX_CONCURRENCY
        for src in ("definitions.json", "cfnstack.yaml"):
            if src == "definitions.json":
                with open(os.path.join(ROOT, src)) as f:
                    asl = json.load(f)
            else:
                asl = TestStateMachineParity._inline_for_comment(
                    "CosyVoice3 TTS synthesis - batch text-to-speech with "
                    "voice cloning, per-clip ASR QC.")
            got = asl["States"]["Map batch synthesis"]["MaxConcurrency"]
            self.assertEqual(got, want,
                             f"{src}: Map MaxConcurrency is {got} but "
                             f"check_and_preprocess splits into {want} "
                             "iterators — they must be equal")

    def test_usable_invocation_sec_matches_the_checkpoint_buffer(self):
        """USABLE_INVOCATION_SEC is a literal here so the scheduling has no
        import-time dependency on lib/, which means nothing but this test stops
        it drifting from the buffer the states actually reserve."""
        import importlib
        ck = importlib.import_module("lib.checkpoint")
        self.assertEqual(self.cp.USABLE_INVOCATION_SEC,
                         900.0 - ck.REMAINING_TIME_BUFFER_MS / 1000.0)

    def test_no_transcript_is_lost_reordered_or_duplicated(self):
        """`outputs` is built by positional append in synthesize_flow_hift and
        the gateway derives the caller-facing transcript_index from a running
        counter over the iterators in order, so a reorder hands clip 40's audio
        back as clip 3 with nothing raising. This is why the splitter only
        chooses where to CUT — LPT bin-packing balances ~1-6 % better and is
        unusable for exactly this reason."""
        for label, chars in _WORKLOADS:
            ts, groups = self._split(chars)
            flat = [t for g in groups for t in g]
            self.assertEqual([id(t) for t in flat], [id(t) for t in ts],
                             f"{label}: transcripts lost, duplicated or "
                             "reordered by the split")

    def test_heaviest_group_is_near_the_theoretical_floor(self):
        """Every iterator runs concurrently, so the Map costs what its SLOWEST
        iterator costs — the heaviest group is the only number that matters.
        No contiguous split can beat max(heaviest transcript, total/slots), so
        that ratio is the real measure. The version that filled to a running
        average instead of bisecting scored 2.23x here: 19 groups at ~145 s and
        a 20th at 286 s, because the accumulated shortfall of 19 slightly-early
        cuts stranded 8 transcripts in the last group."""
        for label, chars in _WORKLOADS:
            ts, groups = self._split(chars)
            work = [self.cp.estimate_transcript_work_sec(t["text"]) for t in ts]
            per = [sum(self.cp.estimate_transcript_work_sec(t["text"])
                       for t in g) for g in groups]
            floor = max(max(work), sum(work) / self.cp.MAP_MAX_CONCURRENCY)
            self.assertLessEqual(max(per) / floor, 1.2,
                                 f"{label}: heaviest iterator {max(per):.0f}s "
                                 f"against a {floor:.0f}s floor")

    def test_split_never_exceeds_max_concurrency_unless_capped_for_timeout(self):
        """Overshooting the ASL limit is the regression this guards: the
        bisection's replay can lose a cut to float rounding and emit
        target_groups + 1. Pass 2 may legitimately exceed it — a group whose
        Flow+HiFT work alone overruns one invocation is better split than
        checkpointed, since resumes are serial where iterators are parallel —
        so the ceiling is relaxed only for a workload where no 20-way split
        could have fitted in the first place, and then only to the number of
        invocations that workload genuinely needs."""
        k = self.cp.MAP_MAX_CONCURRENCY
        for label, chars in _WORKLOADS:
            ts, groups = self._split(chars)
            flow = [self.cp.FLOW_FIXED_SEC
                    + self.cp.FLOW_PER_CHAR_SEC * len(t["text"]) for t in ts]
            # Groups the timeout cap cannot avoid, whatever the split. A
            # transcript is indivisible, so the bound is per-ITEM capacity, not
            # total/USABLE: 100 transcripts at 200.5 s of Flow each fit 4 to an
            # invocation, so 25 groups is optimal even though the total says 24.
            per_group = max(1, int(self.cp.USABLE_INVOCATION_SEC / max(flow)))
            forced = math.ceil(len(ts) / per_group)
            self.assertLessEqual(len(groups), max(k, forced),
                                 f"{label}: {len(groups)} iterators against a "
                                 f"MaxConcurrency of {k} (timeout cap forces "
                                 f"at least {forced})")

    def test_multi_transcript_groups_fit_one_invocation(self):
        """Pass 2's cap. A single transcript is never split — it is the
        indivisible unit, so one pathological near-MAX_TEXT_CHARS transcript
        still overruns on its own, which is the residual caveat
        lib/checkpoint.py documents."""
        for label, chars in _WORKLOADS:
            _, groups = self._split(chars)
            for g in groups:
                if len(g) == 1:
                    continue
                flow = sum(self.cp.FLOW_FIXED_SEC
                           + self.cp.FLOW_PER_CHAR_SEC * len(t["text"])
                           for t in g)
                self.assertLessEqual(flow, self.cp.USABLE_INVOCATION_SEC,
                                     f"{label}: group of {len(g)} needs "
                                     f"{flow:.0f}s of Flow in one invocation")

    def test_work_balancing_beats_count_balancing_on_a_bimodal_batch(self):
        """The question this whole change answers: does transcript COUNT
        suffice, or does length have to be scheduled? 60 short sentences and 13
        long ones split by count puts every long transcript in the last few
        groups. Chars are a proxy — speech tokens fit far better (R2 0.98 vs
        0.87) but do not exist until the LLM has decoded, so they cannot
        schedule anything here."""
        chars = [20] * 60 + [950] * 13
        ts, groups = self._split(chars)
        # Compare against a count split into the SAME number of groups, so the
        # only variable is the rule that chose the cuts. Deriving the chunk size
        # from MAP_MAX_CONCURRENCY instead makes this test track that constant
        # rather than the property it is testing.
        size = math.ceil(len(ts) / len(groups))
        by_count = [ts[j:j + size] for j in range(0, len(ts), size)]

        def heaviest(gs):
            return max(sum(self.cp.estimate_transcript_work_sec(t["text"])
                           for t in g) for g in gs)
        self.assertLess(heaviest(groups), heaviest(by_count) / 2.0,
                        "work-balanced split should more than halve the "
                        "slowest iterator against a count-based one")

    def test_degenerate_inputs(self):
        self.assertEqual(self.cp.split_into_iterator_groups([], 20), [])
        ts = [{"text": "hi"}]
        for k in (0, -5, 1, 20):
            self.assertEqual(self.cp.split_into_iterator_groups(ts, k), [[ts[0]]])

    def test_handler_emits_several_iterators_with_unique_batch_indexes(self):
        """batch_index is not decoration: wav_key is
        `{out_prefix}/{job_id}_{batch_index}_{i}_c{ci}.wav` and the checkpoint
        key is `out_prefix + state_name + job_id + batch_index`, so two
        iterators sharing one would overwrite each other's audio AND resume
        from each other's checkpoints. It counts ITERATORS, not caller batches,
        for that reason."""
        cp = _load_check_and_preprocess()
        event = {"uuid": "u", "input": {"batches": [
            {"voice_id": "v",
             "transcripts": [{"text": "sentence %d" % i} for i in range(40)]}]}}
        out = cp.lambda_handler(event, None)
        iters = out["data"]["iterators"]
        self.assertGreater(len(iters), 1,
                           "one caller batch must fan out across the Map")
        self.assertEqual(sorted(it["batch_index"] for it in iters),
                         list(range(len(iters))))
        self.assertEqual(
            [t["text"] for it in iters for t in it["transcripts"]],
            [t["text"] for t in event["input"]["batches"][0]["transcripts"]])

    def test_map_slots_are_shared_between_batches_by_work_not_evenly(self):
        """Slots are a shared resource across the caller's batches. Dividing
        them evenly (MAP_MAX_CONCURRENCY // len(batches)) hands a 60-transcript
        batch and a 1-transcript batch half the slots each — most idle on one
        side while the other packs many transcripts per iterator. Every batch
        keeps at least one slot, so many tiny batches can overshoot the total;
        the Map queues that surplus, which is why this is a share, not a cap."""
        cp = _load_check_and_preprocess()
        big = [{"text": "a longer sentence number %d" % i} for i in range(60)]
        event = {"uuid": "u", "input": {"batches": [
            {"voice_id": "v", "transcripts": big},
            {"voice_id": "v", "transcripts": [{"text": "tiny"}]}]}}
        iters = cp.lambda_handler(event, None)["data"]["iterators"]
        per_batch = {}
        for it in iters:
            per_batch.setdefault(it["source_batch_index"], []).append(it)
        self.assertEqual(len(per_batch[1]), 1,
                         "a one-transcript batch cannot use more than 1 slot")
        # The heavy batch holds ~98 % of the work, so it should get most of the
        # slots — but bounded by its 60 transcripts, and the splitter
        # legitimately returns FEWER groups when more would not lower the
        # heaviest one (30 pairs and 39 near-singletons both peak at ~70 s, and
        # 30 iterators pay 9 fewer cold model loads for the same makespan). So
        # the floor is "beats an even share", not "MAP_MAX_CONCURRENCY minus a
        # couple" — that would pin this test to the constant, not the property.
        even_share = cp.MAP_MAX_CONCURRENCY // 2
        self.assertGreater(
            len(per_batch[0]), even_share,
            f"the heavy batch got {len(per_batch[0])} slots, no better than "
            f"the {even_share} an even per-batch split would have given it")
        self.assertEqual(sorted(it["batch_index"] for it in iters),
                         list(range(len(iters))),
                         "batch_index must stay unique ACROSS caller batches")


# ─────────────────────────────────────────────────────────────────────────────
# lib/duration — the `stretched` flag's meaning
#
# fit_to_duration returns the audio UNTOUCHED when it is already within half
# a percent of the target. StateSynthesizeFlowHift originally set
# stretched=True whenever a target was merely requested, so the agent-facing
# flag reported a stretch that never happened.
# ─────────────────────────────────────────────────────────────────────────────

class TestDurationFit(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.duration = _load("lib/duration.py", "duration")
        self.sr = 24000
        self.audio = np.zeros(2 * self.sr, dtype=np.float32)  # exactly 2.000 s

    def test_near_exact_target_is_a_noop(self):
        out = self.duration.fit_to_duration(self.audio, self.sr, 2.002)
        self.assertIs(out, self.audio)
        self.assertEqual(len(out), len(self.audio))  # what `stretched` reads

    def test_in_range_target_changes_length(self):
        out = self.duration.fit_to_duration(self.audio, self.sr, 2.10)
        self.assertNotEqual(len(out), len(self.audio))

    def test_out_of_range_target_raises(self):
        with self.assertRaises(self.duration.DurationOutOfRangeError):
            self.duration.fit_to_duration(self.audio, self.sr, 2.5)

    def test_max_stretch_ratio_bound(self):
        self.assertEqual(self.duration.MAX_STRETCH_RATIO, 0.10)


# ─────────────────────────────────────────────────────────────────────────────
# lib/qc tokenization
#
# The CJK character-tokenization branch originally matched only CJK
# ideographs, so all-kana Japanese collapsed to ONE token: measured,
# similarity("ありがとうございます", "ありがとございます") == 0.0 — a single
# dropped character read as total content failure.
# ─────────────────────────────────────────────────────────────────────────────

class TestQcTokenize(unittest.TestCase):
    def setUp(self):
        self.qc = _load("lib/qc.py", "qcmod")

    def test_hiragana_tokenizes_per_character(self):
        self.assertEqual(len(self.qc._tokenize("こんにちはせかい")), 8)

    def test_katakana_and_halfwidth_tokenize_per_character(self):
        self.assertEqual(len(self.qc._tokenize("コンピュータ")), 6)
        self.assertEqual(len(self.qc._tokenize("ｱｲｳ")), 3)

    def test_kana_one_char_drift_is_a_high_score_not_zero(self):
        sim = self.qc.similarity("ありがとうございます", "ありがとございます")
        self.assertGreater(sim, 0.9)

    def test_chinese_still_tokenizes_per_character(self):
        self.assertEqual(len(self.qc._tokenize("你好世界")), 4)

    def test_latin_still_tokenizes_per_word(self):
        self.assertEqual(self.qc._tokenize("hello world there"),
                         ["hello", "world", "there"])
        self.assertEqual(self.qc.similarity("hello world", "hello world"), 1.0)

    def test_thresholds(self):
        self.assertEqual(self.qc.DEFAULT_PASS_THRESHOLD, 0.80)
        self.assertEqual(self.qc.DEFAULT_REGISTER_THRESHOLD, 0.70)

    def test_whisper_repo_is_multilingual(self):
        """base.en would silently break every non-English QC pass."""
        self.assertEqual(self.qc.WHISPER_REPO_ID, "Systran/faster-whisper-base")
        self.assertNotIn(".en", self.qc.WHISPER_REPO_ID)


# ─────────────────────────────────────────────────────────────────────────────
# language normalization
#
# The gateway's register_voice InputSchema declares `language` as free-form
# prose (CFN SchemaDefinition has no enum), and faster-whisper raises
# ValueError for anything outside its 100-code tuple — so "English" used to
# fail StateValidateReference with a raw internal traceback, and any raw
# string that DID survive would later fail the == "en" style gate.
# ─────────────────────────────────────────────────────────────────────────────

class TestLanguageNormalization(unittest.TestCase):
    def setUp(self):
        self.qc = _load("lib/qc.py", "qcmod")

    @unittest.skipUnless(
        importlib.util.find_spec("faster_whisper"),
        "faster_whisper not installed (container-only dependency)")
    def test_codes_and_locales_normalize(self):
        for raw, want in [("en", "en"), ("EN", "en"), ("en-US", "en"),
                          ("zh_CN", "zh"), ("de", "de"), ("yue", "yue")]:
            self.assertEqual(self.qc.normalize_language(raw), want, raw)

    @unittest.skipUnless(
        importlib.util.find_spec("faster_whisper"),
        "faster_whisper not installed (container-only dependency)")
    def test_unrecognized_degrades_to_autodetect(self):
        for raw in ("English", "eng", "klingon"):
            self.assertIsNone(self.qc.normalize_language(raw), raw)

    def test_empty_is_autodetect_without_importing_whisper(self):
        self.assertIsNone(self.qc.normalize_language(None))
        self.assertIsNone(self.qc.normalize_language(""))


# ─────────────────────────────────────────────────────────────────────────────
# gateway_tool contracts
#
# The error shape is shared with 8 sibling docker/*/gateway_tool dispatchers
# and is documented by the InputSchema/OutputSchema blocks in
# media-toolkit-gateway/cfnstack.yaml — `error` is a BOOLEAN there, not the
# message string. tts was the sole outlier.
# ─────────────────────────────────────────────────────────────────────────────

class TestGatewayTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boto = mock.patch("boto3.client", return_value=mock.MagicMock())
        cls.boto.start()
        cls.gw = _load("gateway_tool/lambda_function.py", "gw")

    @classmethod
    def tearDownClass(cls):
        cls.boto.stop()

    def test_error_shape_matches_the_gateway_schema(self):
        err = self.gw._error("boom", "MY_CODE")
        self.assertIs(err["error"], True)
        self.assertEqual(err["errorCode"], "MY_CODE")
        self.assertEqual(err["message"], "boom")

    def test_number_unmarshalling_returns_real_numbers(self):
        """duration_sec is Type: number in the OutputSchema — a quoted
        "1.86" reads to an agent as a string and breaks arithmetic."""
        out = self.gw._unmarshal({"duration_sec": {"N": "1.86"},
                                  "n": {"N": "7"},
                                  "name": {"S": "alice"},
                                  "flag": {"BOOL": True}})
        self.assertEqual(out["duration_sec"], 1.86)
        self.assertIsInstance(out["duration_sec"], float)
        self.assertEqual(out["n"], 7)
        self.assertIsInstance(out["n"], int)
        self.assertEqual(out["name"], "alice")
        self.assertIs(out["flag"], True)

    def test_created_at_stays_a_string(self):
        """created_at is Type: string in the OutputSchema even though it is
        stored as a DDB "N" — the response must match its own contract."""
        out = self.gw._unmarshal({"created_at": {"N": "1750000000000"}})
        self.assertEqual(out["created_at"], "1750000000000")
        self.assertIsInstance(out["created_at"], str)

    def test_foreign_ref_bucket_rejected_at_submit_time(self):
        """TtsLambdaRole grants s3:GetObject on ${ProxyBucket}/* only — a
        reference in another bucket must fail here, not as an opaque
        AccessDenied a minute into the workflow."""
        err = self.gw._register_voice(
            {"ref_s3uri": "s3://someone-elses-bucket/ref.wav",
             "ref_transcript": "hello"})
        self.assertIs(err["error"], True)
        self.assertEqual(err["errorCode"], "INVALID_PARAMETER")

    def test_transcript_index_is_global_across_batches(self):
        """batch_qc_summary.failed_indices would be ambiguous if two entries
        both claimed index 0."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {"iterators": [
                    {"out_bucket": "b", "outputs": [
                        {"wav_key": "k0", "duration_s": 1.0,
                         "qc": {"passed": True, "similarity": 1.0}},
                        {"wav_key": "k1", "duration_s": 1.0,
                         "qc": {"passed": False, "similarity": 0.4}}]},
                    {"out_bucket": "b", "outputs": [
                        {"wav_key": "k2", "duration_s": 1.0,
                         "qc": {"passed": True, "similarity": 0.99}}]}]}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertEqual([o["transcript_index"] for o in out["outputs"]],
                         [0, 1, 2])
        self.assertEqual(out["batch_qc_summary"]["failed_indices"], [1])
        self.assertEqual(out["batch_qc_summary"]["total"], 3)

    def test_missing_out_bucket_becomes_an_entry_error_not_a_crash(self):
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {"iterators": [
                    {"outputs": [{"wav_key": "k0"}]}]}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertIn("error", out["outputs"][0])

    def test_qc_error_is_surfaced(self):
        """A QC pass that could not run reports passed=None + an error
        string; without surfacing it the entry reads as "no QC result"."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {"iterators": [
                    {"out_bucket": "b", "outputs": [
                        {"wav_key": "k0", "qc": {"passed": None,
                                                 "similarity": -1.0,
                                                 "error": "ClientError: 404"}}]}
                ]}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertEqual(out["outputs"][0]["qc_error"], "ClientError: 404")

    # ── Sibling-parity fields (added 2026-08-07) ────────────────────────────
    # This target's outputs[] shape deliberately DIVERGES from the siblings'
    # {uri, mime, label} triple — their label is a role vocabulary (one entry
    # per KIND of file), while these are N instances of one role carrying
    # per-clip metadata a triple cannot hold, plus transcript_index as the join
    # key back to the caller's input array. See _parse_succeeded_output's
    # docstring. What follows are the four things that were NOT a shape
    # difference but a plain gap: every sibling had them and this did not.

    def test_poll_returns_workflow_id(self):
        """Every sibling poll returns workflow_id and this one did not, so an
        agent that resumed a conversation holding only a job_id had no way to
        name the run. The execution NAME is the workflow_id (see
        _start_execution), so it is recoverable without the submit's response."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:wid-42",
                "output": json.dumps({"uuid": "u", "data": {"iterators": [
                    {"out_bucket": "b", "out_prefix": "p/q",
                     "outputs": [{"wav_key": "k0"}]}]}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertEqual(out["workflow_id"], "wid-42")

    def test_poll_returns_the_output_prefix(self):
        """Without this an agent could only tell a user where the clips live by
        string-slicing an entry's uri."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {"iterators": [
                    {"out_bucket": "b",
                     "out_prefix": "micro-workflow/tts-on-aws/output/u/",
                     "outputs": [{"wav_key": "k0"}]}]}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertEqual(out["output_s3_prefix"],
                         "s3://b/micro-workflow/tts-on-aws/output/u")

    def test_output_prefix_is_first_non_empty_not_last(self):
        """Every batch of one execution shares one out_bucket/out_prefix
        (StateCheckAndPreprocess derives both once), so a later batch that
        failed before recording them must not blank out a location already
        established. Last-wins would return no prefix here."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {"iterators": [
                    {"out_bucket": "b", "out_prefix": "p",
                     "outputs": [{"wav_key": "k0"}]},
                    {"outputs": [{"wav_key": "k1"}]}]}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertEqual(out["output_s3_prefix"], "s3://b/p")

    def test_usage_tracking_json_is_reachable(self):
        """usage_tracking.json holds the PER-STATE cost breakdown. It was
        written, the whole usage_doc was in hand (pipeline_wall_sec and
        total_estimated_cost are read out of it), and NO agent could find it.
        Siblings advertise it as outputs[label=usage]; there is no label
        vocabulary here, so it is a URI field."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {
                    "iterators": [{"out_bucket": "b", "out_prefix": "p",
                                   "outputs": [{"wav_key": "k0"}]}],
                    "usage_tracking": {
                        "output_bucket": "b",
                        "usage_key": "p/usage_tracking.json",
                        "pipeline_wall_sec": 12.5,
                        "total_estimated_cost": 0.0231}}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertEqual(out["usage_uri"], "s3://b/p/usage_tracking.json")
        self.assertEqual(out["total_estimated_cost"], 0.0231)

    def test_usage_uri_absent_rather_than_malformed(self):
        """A half-populated usage doc must not yield "s3://None/None"."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {
                    "iterators": [{"out_bucket": "b", "out_prefix": "p",
                                   "outputs": [{"wav_key": "k0"}]}],
                    "usage_tracking": {"pipeline_wall_sec": 1.0}}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertNotIn("usage_uri", out)
        self.assertEqual(out["pipeline_wall_sec"], 1.0)

    def test_register_voice_poll_resolves_preview_uri(self):
        """register_voice and get_voice returned the SAME record in two shapes
        from one target — raw s3_preview_key here vs resolved preview_uri
        there — so an agent that had just registered a voice could not presign
        the preview without hand-building the URI."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {"voice": {
                    "voice_id": "v1", "name": "Ada",
                    "s3_preview_key": "micro-workflow/tts-on-aws/voices/v1/"
                                      "preview.wav"}}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertEqual(
            out["voice"]["preview_uri"],
            f"s3://{self.gw.PROXY_BUCKET}/micro-workflow/tts-on-aws/"
            "voices/v1/preview.wav")
        # The raw key is replaced, not carried alongside — two spellings of one
        # fact is what get_voice already avoids.
        self.assertNotIn("s3_preview_key", out["voice"])
        self.assertEqual(out["voice"]["voice_id"], "v1")

    def test_register_voice_poll_without_a_preview_key_is_unchanged(self):
        """A voice record predating the preview field must pass through."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {"voice": {
                    "voice_id": "v1", "name": "Ada"}}})}
        out = self.gw._parse_succeeded_output(resp)
        self.assertEqual(out["voice"], {"voice_id": "v1", "name": "Ada"})
        self.assertNotIn("preview_uri", out["voice"])

    def test_synthesis_fields_absent_from_a_register_voice_poll(self):
        """The two shapes must not bleed: register_voice returns early, so a
        voice record carries no outputs[]/output_s3_prefix/usage_uri."""
        resp = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "output": json.dumps({"uuid": "u", "data": {
                    "voice": {"voice_id": "v1"}}})}
        out = self.gw._parse_succeeded_output(resp)
        for absent in ("outputs", "output_s3_prefix", "usage_uri",
                       "batch_qc_summary"):
            self.assertNotIn(absent, out)
        # workflow_id is set BEFORE the shape split, so both shapes carry it.
        self.assertIn("workflow_id", out)

    def _dup_submit(self, described, sm=None):
        """Drive _start_execution into its ExecutionAlreadyExists branch with
        describe_execution returning `described`."""
        gw = self.gw
        exc = type("ExecutionAlreadyExists", (Exception,), {})
        gw._sfn.exceptions.ExecutionAlreadyExists = exc
        gw._sfn.start_execution.side_effect = exc("dup")
        gw._sfn.describe_execution.side_effect = None
        gw._sfn.describe_execution.return_value = described
        try:
            return gw._start_execution(
                sm or "arn:aws:states:us-east-1:123456789012:"
                      "stateMachine:m2c-tts-synthesis",
                {"input": {}, "data": {}},
                # the synthesis path (what these tests describe): its
                # handle carries output_s3_prefix since 2026-08-31
                with_output_prefix=True)
        finally:
            gw._sfn.start_execution.side_effect = None
            gw._sfn.describe_execution.return_value = None

    def test_already_exists_arn_is_partition_agnostic(self):
        """Reconstructed from the state machine ARN's own parts, so it holds
        in aws-cn / aws-us-gov (which _SFN_ARN_RE also accepts)."""
        out = self._dup_submit(
            {"executionArn": "arn:x", "status": "RUNNING"},
            sm="arn:aws-cn:states:cn-north-1:123456789012:"
               "stateMachine:m2c-tts")
        self.assertTrue(out["job_id"].startswith(
            "arn:aws-cn:states:cn-north-1:123456789012:execution:m2c-tts:"))
        self.assertTrue(self.gw._SFN_ARN_RE.match(out["job_id"]),
                        f"{out['job_id']} must satisfy the get_job_status "
                        "ARN check the same tool applies to its own output")

    def test_duplicate_submit_reports_the_existing_runs_status_not_results(self):
        """A submit returns a HANDLE, on every path. Two prior shapes were both
        wrong: {"status": "ALREADY_RUNNING"} said nothing about the run, and
        then inlining _parse_succeeded_output(existing) said too much — a submit
        could hand back a full per-transcript outputs[] the schema never
        declared and the caller never asked for. `status` tells the caller the
        run finished; get_job_status is the one place results come from."""
        wav = {"wav_key": "k0", "duration_s": 1.0,
               "qc": {"passed": True, "similarity": 0.9}}
        out = self._dup_submit({
            "executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
            "status": "SUCCEEDED",
            "output": json.dumps({"uuid": "u", "data": {"iterators": [
                {"out_bucket": "b", "outputs": [wav]}]}})})
        self.assertEqual(out["status"], "SUCCEEDED")
        self.assertIn("did not resubmit", out["message"])
        self.assertIn("workflow_id", out)
        self.assertIn("job_id", out)
        # None of the result payload leaks onto the submit response.
        # output_s3_prefix moved OUT of this list 2026-08-31: it is part
        # of the HANDLE now (every sibling's submit returns it, and a
        # callback consumer needs the outputs.json location before any
        # poll) — it is a pointer, not a result.
        for absent in ("outputs", "batch_qc_summary",
                       "voice", "usage_uri", "pipeline_wall_sec",
                       "total_estimated_cost", "error"):
            self.assertNotIn(absent, out)
        self.assertTrue(out["output_s3_prefix"].startswith("s3://"))

    def test_a_fresh_start_and_an_absorbed_resubmit_differ_only_in_status(self):
        """The invariant the handle-only contract exists to create: an agent
        needs ONE reader for a submit response, whichever path it took. `message`
        is the exception — it is the only absorption signal — and `error` appears
        only when the absorbed run failed terminally."""
        gw = self.gw
        gw._sfn.start_execution.side_effect = None
        gw._sfn.start_execution.return_value = {"executionArn": "arn:x"}
        fresh = gw._synthesize_speech({"voice_id": "v", "transcripts": ["hi"],
                                       "idempotency_token": "t"})  # nosec B105
        for existing_status in ("RUNNING", "SUCCEEDED"):
            absorbed = self._dup_submit({
                "executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "status": existing_status})
            self.assertEqual(set(absorbed) - {"message"}, set(fresh),
                             f"absorbed({existing_status}) field set must equal "
                             "a fresh start's, plus message")

    def test_duplicate_submit_reports_a_terminal_failure_as_such(self):
        """"ALREADY_RUNNING" was a claim about the run that nothing checked — a
        FAILED original answered it too. A terminal failure is also STICKY (the
        execution name stays consumed), so misreporting it as running sends the
        caller into a poll loop that can never resolve."""
        for status in ("FAILED", "TIMED_OUT", "ABORTED"):
            out = self._dup_submit({
                "executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                "status": status, "cause": "boom"})
            self.assertEqual(out["status"], status)
            self.assertEqual(out["error"], "boom")
            self.assertNotIn("outputs", out)

    def test_duplicate_submit_of_a_running_job_says_running(self):
        """The one case the old status word got right — and the siblings' word
        for it is RUNNING, not ALREADY_RUNNING."""
        out = self._dup_submit(
            {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
             "status": "RUNNING"})
        self.assertEqual(out["status"], "RUNNING")
        self.assertNotIn("outputs", out)

    def test_duplicate_submit_surfaces_a_describe_failure(self):
        """The new DescribeExecution call is a new failure mode: a throttle, or
        the narrow race where the name collides but describe 404s. It must be a
        clean error, not an exception escaping to the agent."""
        gw = self.gw
        exc = type("ExecutionAlreadyExists", (Exception,), {})
        gw._sfn.exceptions.ExecutionAlreadyExists = exc
        gw._sfn.start_execution.side_effect = exc("dup")
        gw._sfn.describe_execution.side_effect = _client_error("ThrottlingException")
        try:
            out = gw._start_execution(
                "arn:aws:states:us-east-1:123456789012:stateMachine:m2c-tts",
                {"input": {}, "data": {}})
        finally:
            gw._sfn.start_execution.side_effect = None
            gw._sfn.describe_execution.side_effect = None
        self.assertIs(out["error"], True)
        self.assertEqual(out["errorCode"], "SFN_DESCRIBE_FAILED")

    def test_already_running_is_gone_from_the_vocabulary(self):
        """The status word described the gateway's bookkeeping, not the job. Its
        one fact — "this call did not resubmit" — moved to `message`, so no
        caller has to branch on a status that means "the name is taken"."""
        described = {"executionArn": "arn:aws:states:us-east-1:1:execution:sm:e",
                     "status": "SUCCEEDED",
                     "output": json.dumps({"uuid": "u", "data": {}})}
        out = self._dup_submit(described)
        self.assertNotEqual(out["status"], "ALREADY_RUNNING")
        # No CODE path may still emit it. Comments naming the retired status are
        # fine (and wanted — they explain why it went), so strip comment bodies
        # before scanning rather than grepping the whole file.
        src = open(os.path.join(ROOT, "gateway_tool",
                                "lambda_function.py")).read()
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        self.assertNotIn("ALREADY_RUNNING", code)

    def test_job_id_must_be_an_execution_arn(self):
        err = self.gw._get_job_status({"job_id": "not-an-arn"})
        self.assertEqual(err["errorCode"], "INVALID_JOB_ID")

    def test_synthesize_speech_validates_hostile_input(self):
        gw = self.gw
        for event, code in [
            ({}, "MISSING_PARAMETER"),
            ({"voice_id": "v"}, "MISSING_PARAMETER"),
            ({"voice_id": "v", "transcripts": [""]}, "INVALID_PARAMETER"),
            ({"voice_id": "v", "transcripts": [1]}, "INVALID_PARAMETER"),
            ({"voice_id": "v", "transcripts": ["hi"], "flow_steps": 0},
             "INVALID_PARAMETER"),
            ({"voice_id": "v", "transcripts": ["hi"], "flow_steps": 99},
             "INVALID_PARAMETER"),
            ({"voice_id": "v", "transcripts": ["hi"], "flow_steps": "6"},
             "INVALID_PARAMETER"),
            ({"voice_id": "v", "transcripts": ["a", "b"],
              "target_durations_s": [1.0]}, "INVALID_PARAMETER"),
            ({"voice_id": "v", "transcripts": ["hi"], "style": "  "},
             "INVALID_PARAMETER"),
        ]:
            out = gw._synthesize_speech(event)
            self.assertEqual(out.get("errorCode"), code, f"event={event}")

    def test_single_transcript_string_is_accepted(self):
        gw = self.gw
        gw._sfn.start_execution.side_effect = None
        gw._sfn.start_execution.return_value = {"executionArn": "arn:x"}
        out = gw._synthesize_speech({"voice_id": "v", "transcripts": "hello"})
        self.assertEqual(out["status"], "RUNNING")
        sent = json.loads(gw._sfn.start_execution.call_args.kwargs["input"])
        self.assertEqual(
            sent["input"]["batches"][0]["transcripts"], [{"text": "hello"}])

    # ── idempotency_token: the caller's "same request as before" assertion ──
    #
    # The sibling workflows dedup on parameters alone, because their `uuid`
    # names the asset they process and the parameters fully determine the
    # output. tts has no single asset: it synthesizes against a MUTABLE voice
    # catalog and (absent an explicit seed) samples freshly, so identical
    # parameters do not imply an identical result. No token therefore means
    # "do not deduplicate me".
    #
    # The token is consumed into the workflow_id hash and DISCARDED — it never
    # reaches the state machine and is never echoed back. It used to be echoed
    # as `uuid`, and the read that fed the echo was the shape of the
    # face-recognition bug this design replaced.

    def _submit(self, **kwargs):
        gw = self.gw
        gw._sfn.start_execution.side_effect = None
        gw._sfn.start_execution.return_value = {"executionArn": "arn:x"}
        return gw._synthesize_speech(
            {"voice_id": "v", "transcripts": ["hi"], **kwargs})

    def test_no_token_means_a_fresh_run_every_time(self):
        """THE bug this guards: dedup-on-parameters silently returned an
        earlier run's audio to a caller who never asked to be deduplicated.
        A second identical call must be able to synthesize again."""
        a = self._submit()
        b = self._submit()
        self.assertNotEqual(a["workflow_id"], b["workflow_id"])

    def test_a_supplied_token_makes_the_submit_idempotent(self):
        """The point of the token: a caller that wants retry safety asks for
        it, and identical parameters + the same token collide on one name."""
        a = self._submit(idempotency_token="take-1")  # nosec B106
        b = self._submit(idempotency_token="take-1")  # nosec B106
        self.assertEqual(a["workflow_id"], b["workflow_id"])

    def test_the_same_token_with_different_parameters_still_re_runs(self):
        """The token narrows dedup, it does not replace the parameter hash —
        reusing a token for different text must not return the wrong audio."""
        a = self._submit(idempotency_token="take-1")  # nosec B106
        c = self._submit(idempotency_token="take-1", transcripts=["hi!"])  # nosec B106
        self.assertNotEqual(a["workflow_id"], c["workflow_id"])

    def test_different_tokens_with_identical_parameters_both_run(self):
        """Two callers asking for the same text are two jobs, not one."""
        a = self._submit(idempotency_token="take-1")  # nosec B106
        b = self._submit(idempotency_token="take-2")  # nosec B106
        self.assertNotEqual(a["workflow_id"], b["workflow_id"])

    def test_a_bad_token_is_rejected_at_submit(self):
        out = self._submit(idempotency_token="not a valid name!")  # nosec B106
        self.assertIs(out["error"], True)
        self.assertEqual(out["errorCode"], "INVALID_PARAMETER")

    def test_the_retired_uuid_parameter_is_rejected_not_ignored(self):
        """`uuid` meant BOTH "which asset" and "same request", so a caller still
        sending it cannot be assumed to mean the surviving half. Ignoring it
        would silently start a fresh run for a caller who believed they were
        protected from paying twice."""
        # The mock is shared across this class, so count the delta rather than
        # asserting never-called — a stale counter would make this pass for the
        # wrong reason.
        before = self.gw._sfn.start_execution.call_count
        out = self._submit(uuid="take-1")
        self.assertIs(out["error"], True)
        self.assertEqual(out["errorCode"], "INVALID_PARAMETER")
        self.assertIn("idempotency_token", out["message"])
        self.assertEqual(self.gw._sfn.start_execution.call_count, before,
                         "a rejected request still started an execution")

    def test_the_token_is_never_echoed_on_submit(self):
        out = self._submit(idempotency_token="take-1")  # nosec B106
        self.assertNotIn("uuid", out)
        self.assertNotIn("idempotency_token", out)
        self.assertTrue(out["workflow_id"])

    def test_the_token_never_reaches_the_state_machine(self):
        """It rides in the hash preimage as extra material, not in sfn_input, so
        "never forwarded, never stored" holds by construction. The event `uuid`
        that IS sent must stay the execution name — the states derive out_prefix
        from it, so two runs sharing one would overwrite each other's audio."""
        out = self._submit(idempotency_token="take-1")  # nosec B106
        sent = json.loads(self.gw._sfn.start_execution.call_args.kwargs["input"])
        self.assertEqual(sent["uuid"], out["workflow_id"])
        self.assertNotIn("idempotency_token", sent)
        self.assertNotIn("take-1", json.dumps(sent))

    def test_a_poll_returns_no_token_from_either_side(self):
        """Nothing forwards a token, so one found in an execution is stale data
        from an older run. Planting it in BOTH the input and the output guards
        against re-adding a reader of any kind — the predecessor helper read the
        input, and the version before that read the output."""
        out = self.gw._parse_succeeded_output({
            "executionArn": "arn:x",
            "input": json.dumps({"uuid": "derived-name",
                                 "idempotency_token": "stale-take-1"}),  # nosec B105
            "output": json.dumps({"uuid": "derived-name",
                                  "idempotency_token": "stale-take-1",  # nosec B105
                                  "data": {}})})
        self.assertNotIn("uuid", out)
        self.assertNotIn("idempotency_token", out)
        # `execution_arn` used to be echoed here. It is gone: it repeated the
        # ARN the caller passed in as `job_id`, which _get_job_status now
        # attaches on every branch.
        self.assertNotIn("execution_arn", out)

    def test_the_retired_reader_is_gone(self):
        """_idempotency_token() read the caller's token out of an execution. Its
        deletion is the fix; a re-added helper is the regression."""
        self.assertFalse(hasattr(self.gw, "_idempotency_token"))

    def test_register_voice_stays_deduped_on_parameters_alone(self):
        """Deliberately exempt, and it is the one case with a DATA effect
        rather than a cost effect: write_voice_record mints voice_id = uuid4()
        and writes it with an unconditional put_item, and no tool deletes a
        voice — so a duplicate run means a second permanent catalog entry
        nobody can remove. Absorbing that retry beats letting a caller force
        a duplicate registration."""
        gw = self.gw
        gw._sfn.start_execution.side_effect = None
        gw._sfn.start_execution.return_value = {"executionArn": "arn:x"}
        ev = {"ref_s3uri": f"s3://{gw.PROXY_BUCKET}/ref.wav",
              "ref_transcript": "hello there"}
        a = gw._register_voice(dict(ev))
        b = gw._register_voice(dict(ev))
        self.assertEqual(a["workflow_id"], b["workflow_id"])
        sent = json.loads(gw._sfn.start_execution.call_args.kwargs["input"])
        self.assertNotIn("idempotency_token", sent)

    def test_register_voice_accepts_a_token_and_says_the_right_thing(self):
        """`register_voice` is the counter-example to "ignores a token" meaning
        "starts nothing", and the warning must not claim otherwise.

        It ignores a token, STARTS an execution, and dedups that execution on
        the parameter hash — so an identical resubmit collapses to one run. The
        shared warning said "this tool starts no execution, so every call runs
        and reflects current state" until 2026-08-06: both clauses false here,
        and the second one argues against this exemption in the exact place a
        maintainer reads before removing it. The exemption is load-bearing —
        `write_voice_record` mints `voice_id = uuid4()` under an unconditional
        `put_item` and no tool deletes a voice.

        Routed through `lambda_handler` on purpose: the warn lives at the
        dispatcher, so calling `_register_voice` directly would assert nothing.
        """
        gw = self.gw
        gw._sfn.start_execution.side_effect = None
        gw._sfn.start_execution.return_value = {"executionArn": "arn:x"}
        ev = {"ref_s3uri": f"s3://{gw.PROXY_BUCKET}/ref.wav",
              "ref_transcript": "hello there",
              "idempotency_token": "tok-1"}  # nosec B105
        ctx = mock.MagicMock()
        ctx.client_context.custom = {
            "bedrockAgentCoreToolName": "text-to-speech___register_voice"}
        with mock.patch.object(gw.logger, "warning") as warn:
            out = gw.lambda_handler(dict(ev), ctx)

        self.assertNotIn("error", out,
                         "a token must not fail an otherwise-valid request")
        note = " ".join(str(c) for c in warn.call_args_list).lower()
        self.assertIn("idempotency_token", note,
                      "an ignored token must leave a trace")
        for phrase in ("starts no execution", "every call runs"):
            self.assertNotIn(
                phrase, note,
                f"the warning claims {phrase!r}, which is false for "
                "register_voice — it starts an execution AND dedups it")

        # The token is ignored, so it cannot have reached the hash: the same
        # parameters without it must produce the same run.
        bare = gw._register_voice(
            {"ref_s3uri": f"s3://{gw.PROXY_BUCKET}/ref.wav",
             "ref_transcript": "hello there"})
        self.assertEqual(out["workflow_id"], bare["workflow_id"],
                         "a token that is 'ignored' still changed the id")


class TestGatewayInputExamples(unittest.TestCase):
    """input_examples.json is the INPUT twin of output_contract.json: one
    minimal valid request per tool, served by describe_tool so an agent gets
    a known-good starting call instead of guessing shapes from prose. The
    round-trip tests drive EVERY example through the real lambda_handler
    with mocked boto3 clients, so an example that drifts from the real
    validation fails this build, not an agent's first call. Nothing here is
    validation-only — the catalog reads' DDB stubs are one line each, so the
    examples are driven to a successful response, not just past the
    parameter checks.
    """

    TOOLS = ("synthesize_speech", "register_voice", "get_job_status",
             "list_voice_collection", "get_voice", "describe_tool")

    @classmethod
    def setUpClass(cls):
        cls.boto = mock.patch("boto3.client", return_value=mock.MagicMock())
        cls.boto.start()
        cls.gw = _load("gateway_tool/lambda_function.py", "gw_examples")
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            cls.examples = json.load(f)["tools"]

    @classmethod
    def tearDownClass(cls):
        cls.boto.stop()

    def _call(self, tool, event):
        ctx = mock.MagicMock()
        ctx.client_context.custom = {
            "bedrockAgentCoreToolName": f"text-to-speech___{tool}"}
        return self.gw.lambda_handler(dict(event), ctx)

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        self.assertEqual(sorted(self.examples), sorted(self.TOOLS))
        for tool, entry in self.examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)
            plain = {n for n in entry["required"]
                     if " " not in n and "|" not in n}
            # Every PLAIN required name must appear in the tool's own
            # example — prose one-of entries (containing a space or "|")
            # state rules the example can only satisfy one arm of.
            for name in plain:
                self.assertIn(name, entry["example"],
                              f"{tool}: required field {name!r} missing "
                              f"from its example")

    def test_describe_tool_carries_input_examples(self):
        out = self._call("describe_tool", {})
        self.assertEqual(out["input_examples"]["tools"], self.examples)

    def test_submit_examples_round_trip_through_the_real_handler(self):
        """The examples ARE the contract: round-trip both submits through
        the real handler so drift fails this build, not an agent's first
        call. The traps the shapes must not re-grow: synthesize_speech is
        the flat voice_id + transcripts contract, never the internal
        `batches` shape; neither submit may carry the retired `uuid`
        (rejected); and register_voice carries no idempotency_token by
        design (parameter-hash dedup — a duplicate registration would be a
        second permanent catalog entry)."""
        gw = self.gw
        gw._sfn.start_execution.side_effect = None
        gw._sfn.start_execution.return_value = {"executionArn": "arn:fake"}

        synth = self.examples["synthesize_speech"]["example"]
        self.assertNotIn("uuid", synth)
        self.assertNotIn("batches", synth)
        self.assertIsInstance(synth["transcripts"], list)
        out = self._call("synthesize_speech", synth)
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        self.assertIn("job_id", out)

        reg = self.examples["register_voice"]["example"]
        self.assertNotIn("uuid", reg)
        self.assertNotIn("idempotency_token", reg)
        # the validator pins ref_s3uri to THIS deployment's proxy bucket, so
        # point the module at the example's placeholder bucket for the call
        with mock.patch.object(gw, "PROXY_BUCKET", "my-proxy-bucket"):
            out = self._call("register_voice", reg)
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        self.assertIn("job_id", out)

    def test_read_examples_round_trip_through_the_real_handler(self):
        gw = self.gw
        gw._ddb.query.side_effect = None
        gw._ddb.query.return_value = {"Items": []}
        out = self._call("list_voice_collection",
                         self.examples["list_voice_collection"]["example"])
        self.assertNotIn("error", out, f"example refused: {out}")

        gw._ddb.get_item.side_effect = None
        gw._ddb.get_item.return_value = {"Item": {
            "voice_id":
                {"S": self.examples["get_voice"]["example"]["voice_id"]},
            "name": {"S": "Lena Voss"}}}
        out = self._call("get_voice", self.examples["get_voice"]["example"])
        self.assertNotIn("error", out, f"example refused: {out}")

        job_id = self.examples["get_job_status"]["example"]["job_id"]
        self.assertTrue(gw._SFN_ARN_RE.match(job_id),
                        "the example job_id must satisfy the same ARN check "
                        "the tool applies to a real poll")
        gw._sfn.describe_execution.side_effect = None
        gw._sfn.describe_execution.return_value = {
            "status": "RUNNING", "executionArn": job_id}
        out = self._call("get_job_status",
                         self.examples["get_job_status"]["example"])
        self.assertNotIn("error", out, f"example refused: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CFN custom resource: every path MUST send a response
#
# This is not hypothetical. The ProxyBucketValidator resource read
# ResourceProperties["AccountId"] outside its try block, and CFN's Delete
# event does not carry it — so Delete raised KeyError, _send() never ran, and
# CloudFormation waited a full hour before giving up. It happened twice
# (2026-07-31 and 2026-08-02), each time orphaning the validator Lambda and
# its IAM role, whose surviving FunctionName then blocked the next
# create-stack with an EarlyValidation ResourceExistenceCheck failure.
# An unresponsive custom resource is an hour of downtime, so the contract is:
# no statement that can raise may sit outside the try.
# ─────────────────────────────────────────────────────────────────────────────

class TestBucketValidatorAlwaysResponds(unittest.TestCase):
    def setUp(self):
        with mock.patch("boto3.client"):
            self.bv = _load("gateway_tool/bucket_validator.py", "bv")
        self.sent = []
        self.bv._send = lambda ev, ctx, status, bucket, reason: self.sent.append(
            {"type": ev["RequestType"], "status": status,
             "bucket": bucket, "reason": reason})

    class _Ctx:
        log_stream_name = "log-stream"

    def _invoke(self, **overrides):
        event = {"RequestType": "Delete", "StackId": "s", "RequestId": "r",
                 "LogicalResourceId": "l", "PhysicalResourceId": "b",
                 "ResourceProperties": {"BucketName": "b"}}
        event.update(overrides)
        self.bv.handler(event, self._Ctx())
        return self.sent[-1]

    def test_delete_without_accountid_responds_success(self):
        """THE bug: CFN's Delete event carries no AccountId. Reading it
        outside the try raised KeyError, no response was sent, and the stack
        hung for an hour and orphaned resources."""
        r = self._invoke(RequestType="Delete",
                         ResourceProperties={"BucketName": "b"})
        self.assertEqual(r["status"], "SUCCESS")

    def test_delete_without_any_properties_responds_success(self):
        """A Delete for a resource whose create never completed can arrive
        with no ResourceProperties at all — still must not raise."""
        ev = {"RequestType": "Delete", "StackId": "s", "RequestId": "r",
              "LogicalResourceId": "l", "PhysicalResourceId": "b2"}
        self.bv.handler(ev, self._Ctx())
        self.assertEqual(self.sent[-1]["status"], "SUCCESS")
        self.assertEqual(self.sent[-1]["bucket"], "b2")

    def test_create_missing_accountid_fails_loudly_not_silently(self):
        """The fix must not paper over a genuinely broken Create: a missing
        AccountId there is a template bug and must FAIL with a named reason,
        not succeed and not hang."""
        r = self._invoke(RequestType="Create",
                         ResourceProperties={"BucketName": "b3"})
        self.assertEqual(r["status"], "FAILED")
        self.assertIn("AccountId", r["reason"])

    def test_no_property_reads_precede_the_try_block(self):
        """Structural guard: the handler's first statement that can raise must
        be inside the try. Catches a future edit that hoists a lookup back out
        (the exact regression), which no behavioural test would see if the new
        key happened to be present in the test event."""
        import inspect
        src = inspect.getsource(self.bv.handler)
        head = src.split("try:", 1)[0]
        self.assertNotIn('["', head,
                         "subscript before try: — an escaping KeyError here "
                         "means CFN never gets a response")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-file contract: the two Lambda-side copies of each state machine
#
# cfnstack.yaml defines both state machines INLINE (DefinitionString: !Sub),
# while definitions.json / definitions_register_voice.json are the editable
# source of truth shipped in source_code.zip. Nothing in the build enforces
# that the two agree — a fix applied to only one silently deploys the other.
# ─────────────────────────────────────────────────────────────────────────────

class TestStateMachineParity(unittest.TestCase):
    @staticmethod
    def _inline_definitions() -> list[dict]:
        """Every inline `DefinitionString: !Sub |- {...}` block in
        cfnstack.yaml, parsed. The !Sub placeholders (${TtsLambda1Core.Arn})
        are left as literal text — definitions.json carries the same ones, so
        the two parse to identical structures."""
        with open(os.path.join(ROOT, "cfnstack.yaml")) as f:
            lines = f.readlines()
        blocks = []
        for i, ln in enumerate(lines):
            if ln.strip() != "DefinitionString:":
                continue
            j = i + 1
            while lines[j].strip() in ("!Sub |-", "!Sub |"):
                j += 1
            indent = len(lines[j]) - len(lines[j].lstrip())
            body = []
            for ln2 in lines[j:]:
                if ln2.strip() and (len(ln2) - len(ln2.lstrip())) < indent:
                    break
                body.append(ln2[indent:] if ln2.strip() else "\n")
            blocks.append(json.loads("".join(body)))
        return blocks

    @classmethod
    def _inline_for_comment(cls, comment: str) -> dict:
        """Same pairing-by-Comment as _inline_for, reachable without a test
        instance so TestIteratorScheduling can read the deployed ASL too."""
        matches = [b for b in cls._inline_definitions()
                   if b.get("Comment") == comment]
        assert len(matches) == 1, (
            f"expected exactly one inline definition with Comment={comment!r},"
            f" found {len(matches)}")
        return matches[0]

    def _inline_for(self, comment: str) -> dict:
        blocks = self._inline_definitions()
        matches = [b for b in blocks if b.get("Comment") == comment]
        self.assertEqual(len(matches), 1,
                         f"expected exactly one inline definition with "
                         f"Comment={comment!r}, found {len(matches)} among "
                         f"{len(blocks)} blocks — the Comment is how the two "
                         "copies are paired up, so it must match verbatim")
        return matches[0]

    def _definitions(self, name: str) -> dict:
        with open(os.path.join(ROOT, name)) as f:
            return json.load(f)

    def test_two_inline_definitions_exist(self):
        self.assertEqual(len(self._inline_definitions()), 2)

    def test_synthesis_definition_matches_inline_copy(self):
        want = self._definitions("definitions.json")
        self.assertEqual(self._inline_for(want["Comment"]), want,
                         "definitions.json and cfnstack.yaml's inline "
                         "DefinitionString have diverged")

    def test_register_voice_definition_matches_inline_copy(self):
        want = self._definitions("definitions_register_voice.json")
        self.assertEqual(self._inline_for(want["Comment"]), want,
                         "definitions_register_voice.json and cfnstack.yaml's "
                         "inline DefinitionString have diverged")

    # The full house tier set. All seven are declared even though only four
    # are routed to, so a state can be re-pointed at another tier without a
    # template change. Declared-but-unrouted is the established norm, not a
    # concession: media-processing routes to 3 of its 7 tiers, audiostem 4 of
    # 7, speech-diarization 5 of 7. An unrouted function is never invoked, so
    # it costs nothing at runtime.
    # Pinned as an exact set in BOTH directions on purpose: a missing tier
    # silently removes the swap-during-testing capability, and an extra one is
    # a name no sibling shares.
    _TIERS = (
        "TtsLambdaHalfCore",
        "TtsLambda1Core",
        "TtsLambda2Cores",
        "TtsLambda3Cores",
        "TtsLambda4Cores",
        "TtsLambda5Cores",
        "TtsLambda6Cores",
    )

    def _declared_tiers(self, cfn: str) -> set:
        return {ln.strip().rstrip(":") for ln in cfn.splitlines()
                if ln.startswith("    TtsLambda")
                and ln.rstrip().endswith(":")
                and "LogGroup" not in ln and "Role" not in ln}

    def _referenced_tiers(self) -> set:
        referenced = set()
        for name in ("definitions.json", "definitions_register_voice.json"):
            with open(os.path.join(ROOT, name)) as f:
                for match in json.dumps(json.load(f)).split("${")[1:]:
                    referenced.add(match.split(".Arn}")[0])
        return referenced

    def test_all_seven_memory_tiers_are_declared_and_granted(self):
        """Every declared tier must also be granted lambda:InvokeFunction —
        a tier the state-machine role cannot invoke fails the execution at
        runtime, and that failure only appears once a state is re-pointed at
        it, i.e. exactly when someone is mid-experiment."""
        with open(os.path.join(ROOT, "cfnstack.yaml")) as f:
            cfn = f.read()
        self.assertEqual(self._declared_tiers(cfn), set(self._TIERS),
                         "the declared tier functions must be exactly the "
                         "seven the sibling workflows declare")
        for fn in self._TIERS:
            self.assertIn(f"!GetAtt {fn}.Arn", cfn,
                          f"{fn} is declared but not granted "
                          "lambda:InvokeFunction")

    def test_every_referenced_lambda_tier_exists(self):
        """A state machine referencing a tier this template does not declare
        fails at deploy time. The reverse is NOT an error: declared-but-
        unrouted tiers are the point (see _TIERS)."""
        with open(os.path.join(ROOT, "cfnstack.yaml")) as f:
            cfn = f.read()
        self.assertLessEqual(self._referenced_tiers(),
                             self._declared_tiers(cfn),
                             "a state machine references a Lambda this "
                             "template does not declare")

    def test_the_seven_memory_sizes_match_the_sibling_workflows(self):
        """The tier NAMES only buy a swap if the MEMORY behind them matches
        the siblings. Measured: no sibling declares a tier at a different
        memory or name from these, so the pairs are the house vocabulary —
        3-cores meaning 5376 MB everywhere else and something else here would
        make a re-point silently non-comparable."""
        import yaml

        class Loader(yaml.SafeLoader):
            pass

        def _passthrough(loader, tag_suffix, node):
            if isinstance(node, yaml.ScalarNode):
                return {tag_suffix: loader.construct_scalar(node)}
            if isinstance(node, yaml.SequenceNode):
                return {tag_suffix: loader.construct_sequence(node)}
            return {tag_suffix: loader.construct_mapping(node)}

        Loader.add_multi_constructor("!", _passthrough)
        with open(os.path.join(ROOT, "cfnstack.yaml")) as f:
            mappings = yaml.load(f, Loader=Loader)["Mappings"]["Lambda"]  # nosec B506
        want = {
            "HalfCore": (896, "half-core"),
            "OneCore": (1792, "1-core"),
            "TwoCores": (3584, "2-cores"),
            "ThreeCores": (5376, "3-cores"),
            "FourCores": (7168, "4-cores"),
            "FiveCores": (8960, "5-cores"),
            "SixCores": (10240, "6-cores"),
        }
        got = {k: (v["Memory"], v["Name"]) for k, v in mappings.items()
               if k not in ("Function", "GatewayTool")}
        self.assertEqual(got, want,
                         "tier memory/name pairs must match the sibling "
                         "micro-workflows exactly")


# ─────────────────────────────────────────────────────────────────────────────
# Every Lambda needs a DECLARED log group
#
# A function with no AWS::Logs::LogGroup still gets one: Lambda auto-creates
# /aws/lambda/<name> on first invocation, with never-expire retention and no
# stack ownership. Two consequences, both observed on this stack in eu-west-1 —
# logs accumulate forever, and the orphaned group survives stack deletion, so
# the next create-stack of a declared group at the same name fails with
# "already exists". proxy-bucket-validator and start-build both shipped that
# way; the tier functions and the gateway never did.
# ─────────────────────────────────────────────────────────────────────────────

class TestEveryLambdaHasADeclaredLogGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml

        class Loader(yaml.SafeLoader):
            pass

        def _passthrough(loader, tag_suffix, node):
            if isinstance(node, yaml.ScalarNode):
                return {tag_suffix: loader.construct_scalar(node)}
            if isinstance(node, yaml.SequenceNode):
                return {tag_suffix: loader.construct_sequence(node, deep=True)}
            return {tag_suffix: loader.construct_mapping(node, deep=True)}

        Loader.add_multi_constructor("!", _passthrough)
        with open(os.path.join(ROOT, "cfnstack.yaml")) as f:
            cls.tpl = yaml.load(f, Loader=Loader)  # nosec B506
        cls.res = cls.tpl["Resources"]

    @staticmethod
    def _resolve(sub, mappings):
        """Render a !Sub value to a plain string. Two forms appear here:
        a bare template, and [template, {var: !FindInMap [...]}] — the tier
        functions build their names out of Mappings lookups, so a name-shaped
        string comparison has to resolve those rather than skip them."""
        if isinstance(sub, str):
            return sub
        template, variables = sub[0], sub[1]
        for name, value in variables.items():
            if "FindInMap" in value:
                top, second, third = value["FindInMap"]
                value = mappings[top][second][third]
            template = template.replace("${" + name + "}", str(value))
        return template

    def _names(self, resource_type, key):
        """Resources of `resource_type` that DECLARE `key`, resolved to plain
        strings. StartBuildLambda is deliberately unnamed (matching the six
        siblings), so `key` is genuinely absent there — skipping it here is
        correct, and test_unnamed_functions_route_logs_via_logging_config
        covers that path instead."""
        mappings = self.tpl["Mappings"]
        return {logical: self._resolve(body["Properties"][key]["Sub"], mappings)
                for logical, body in self.res.items()
                if body["Type"] == resource_type
                and key in body["Properties"]}

    def test_every_function_has_a_log_group_at_its_exact_name(self):
        functions = self._names("AWS::Lambda::Function", "FunctionName")
        groups = set(self._names("AWS::Logs::LogGroup", "LogGroupName").values())
        missing = {logical: f"/aws/lambda/{name}"
                   for logical, name in functions.items()
                   if f"/aws/lambda/{name}" not in groups}
        self.assertEqual(missing, {},
                         "these functions would get an unmanaged, never-expire "
                         "log group auto-created by Lambda, which then survives "
                         "stack deletion and blocks the next create-stack")

    def test_unnamed_functions_route_logs_via_logging_config(self):
        """An unnamed function CANNOT be covered by the test above: CFN
        generates its physical name, so /aws/lambda/{FunctionName} is
        unknowable at author time. LoggingConfig.LogGroup is the only way it
        reaches a declared, stack-owned group — without one Lambda
        auto-creates a never-expire group named after the generated name,
        which is what left 16 orphaned *-StartBuildLambda-<random> groups in
        eu-west-1. This test is the reason dropping a FunctionName is safe."""
        unnamed = [logical for logical, body in self.res.items()
                   if body["Type"] == "AWS::Lambda::Function"
                   and "FunctionName" not in body["Properties"]]
        self.assertEqual(
            unnamed, ["StartBuildLambda"],
            "a function lost its FunctionName without gaining a "
            "LoggingConfig check here — every unnamed function needs one")
        for logical in unnamed:
            props = self.res[logical]["Properties"]
            self.assertIn("LoggingConfig", props, logical)
            self.assertEqual(
                props["LoggingConfig"]["LogGroup"],
                {"Ref": "StartBuildLambdaLogGroup"},
                f"{logical} must route to its declared group via "
                "LoggingConfig, or Lambda auto-creates an unmanaged one")

    def test_every_log_group_sets_retention(self):
        """A declared group with no RetentionInDays is never-expire too — the
        declaration alone does not fix the cost half of the problem."""
        for logical, body in self.res.items():
            if body["Type"] == "AWS::Logs::LogGroup":
                self.assertIn("RetentionInDays", body["Properties"], logical)

    def test_custom_resources_depend_on_their_function_log_group(self):
        """A log group declared but created AFTER the function's first
        invocation loses the race: Lambda has already made the unmanaged one.
        Custom resources invoke during stack create, so they are the only
        functions here that need the ordering stated explicitly."""
        for custom, group in (("ProxyBucketValidator",
                               "ProxyBucketValidatorLogGroup"),
                              ("BuildAndSeedModels", "StartBuildLambdaLogGroup")):
            depends = self.res[custom].get("DependsOn")
            depends = [depends] if isinstance(depends, str) else (depends or [])
            self.assertIn(group, depends,
                          f"{custom} must DependsOn {group} or Lambda wins the "
                          "race and creates an unmanaged group anyway")

    def test_custom_resource_types_match_the_sibling_workflows(self):
        """The Custom:: TYPE is part of a custom resource's identity, so these
        spellings are not cosmetic — changing one is a replacement that fires
        Delete on the old resource. They were tts-only spellings
        (Custom::ProxyBucketValidation, Custom::BuildAndSeedModels) because
        this stack was ported from a separate upstream repo, which forced the
        bundle's docs to tell readers to grep for both. Pinning them here is
        what keeps that special case from coming back."""
        self.assertEqual(self.res["ProxyBucketValidator"]["Type"],
                         "Custom::ProxyBucketValidator")
        self.assertEqual(self.res["BuildAndSeedModels"]["Type"],
                         "Custom::StartBuildDelayResponse")

    def test_start_build_role_cannot_create_its_own_log_group(self):
        """Same invariant as the validator role below, and it regressed here
        for longer: this role carried AWSLambdaBasicExecutionRole even after
        the group was declared, so the function had standing permission to
        create an unmanaged group instead of failing visibly."""
        props = self.res["StartBuildLambdaRole"]["Properties"]
        self.assertNotIn("ManagedPolicyArns", props,
                         "AWSLambdaBasicExecutionRole grants "
                         "logs:CreateLogGroup on every group")
        statements = props["Policies"][0]["PolicyDocument"]["Statement"]
        actions = [a for s in statements
                   for a in (s["Action"] if isinstance(s["Action"], list)
                             else [s["Action"]])]
        self.assertNotIn("logs:CreateLogGroup", actions)
        self.assertTrue(
            any("StartBuildLambdaLogGroup" in str(s.get("Resource"))
                for s in statements),
            "no statement scopes logs to the declared group's ARN")

    def test_function_names_fit_the_64_char_limit_at_the_prefix_ceiling(self):
        """FunctionName's hard limit is 64 (measured: 65 is rejected with an
        explicit length constraint; at exactly 64 the name validates). The
        sibling workflows' ResourcePrefix ceiling is 27, set by hybrid-search's
        S3 Vectors bucket name — so every name here must fit at prefix 27, not
        merely at the 20-char prefix that happens to be deployed today. The
        old -tts-proxy-bucket-validator name passed at 20 and its
        audiostem-separation equivalent sat at exactly 64, one character from
        failing the deploy."""
        CEILING = 27
        mappings = self.tpl["Mappings"]
        for logical, name in self._names("AWS::Lambda::Function",
                                         "FunctionName").items():
            resolved = name.replace("${ResourcePrefix}", "P" * CEILING)
            self.assertLessEqual(
                len(resolved), 64,
                f"{logical} renders to {len(resolved)} chars at "
                f"ResourcePrefix={CEILING} ({resolved!r}) — over Lambda's "
                "64-char FunctionName limit")

    def test_validator_role_cannot_create_its_own_log_group(self):
        """AWSLambdaBasicExecutionRole grants logs:CreateLogGroup on every
        group. Alongside a DECLARED group that is actively harmful: a function
        that cannot reach its declared group silently creates an unmanaged one
        instead of failing visibly. Logs must be scoped to the declared
        group's ARN and the managed policy must be absent."""
        props = self.res["ProxyBucketValidatorRole"]["Properties"]
        self.assertNotIn(
            "ManagedPolicyArns", props,
            "AWSLambdaBasicExecutionRole grants logs:CreateLogGroup — scope "
            "logs to ProxyBucketValidatorLogGroup's ARN instead")
        statements = props["Policies"][0]["PolicyDocument"]["Statement"]
        actions = [a for s in statements
                   for a in (s["Action"] if isinstance(s["Action"], list)
                             else [s["Action"]])]
        self.assertNotIn("logs:CreateLogGroup", actions)
        self.assertTrue(
            any("ProxyBucketValidatorLogGroup" in str(s.get("Resource"))
                for s in statements),
            "no statement scopes logs to the declared group's ARN")


class TestBuildScriptNamingConvention(unittest.TestCase):
    """The micro-workflow-bundle composes each child's template S3 key from
    {solution}/{version}/{WORKFLOW_NAME}/{TEMPLATE_NAME}. This stack shipped
    TEMPLATE_NAME hardcoded as "tts-on-aws-stack.template" through
    alpha.0803c while its seven siblings derived it, which made the bundle
    carry a special case. Drifting back would break the bundle's TemplateURL
    at deploy time — ~90 minutes into a CodeBuild-heavy nested-stack create.
    """

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "build-deploy-s3-dist.sh")) as f:
            cls.build = f.read()
        with open(os.path.join(ROOT, "create_update_stack.sh")) as f:
            cls.deploy = f.read()

    def test_workflow_name_is_the_directory_minus_the_on_aws_suffix(self):
        self.assertIn('WORKFLOW_NAME="tts"', self.build)

    def test_template_name_is_derived_not_hardcoded(self):
        self.assertIn('TEMPLATE_NAME="${WORKFLOW_NAME}-stack.template"',
                      self.build)
        self.assertNotIn("tts-on-aws-stack.template", self.build)

    def test_deploy_script_reads_the_same_template_basename(self):
        """create_update_stack.sh hardcodes the path to the built template. It
        is a separate file from the one that names it, so the two can disagree
        silently — and then the deploy fails on a missing file after the build
        reported success."""
        self.assertIn("dist/global-s3-assets/tts-stack.template", self.deploy)
        self.assertNotIn("tts-on-aws-stack.template", self.deploy)


def _read(relpath: str) -> str:
    with open(os.path.join(ROOT, relpath)) as f:
        return f.read()


class TestCallbackMechanism(unittest.TestCase):
    """The push-completion port (2026-08-31). Every test names the review
    finding it pins; the mechanism shipped with zero coverage and the
    review found four live defects in it."""

    @classmethod
    def setUpClass(cls):
        cls.boto = mock.patch("boto3.client", return_value=mock.MagicMock())
        cls.boto.start()
        os.environ.setdefault("CALLBACK_TABLE", "test-callbacks")
        cls.gw = _load("gateway_tool/lambda_function.py", "cb_gw")
        cls.waker = _load("gateway_tool/callback_waker.py", "cb_waker")

    @classmethod
    def tearDownClass(cls):
        cls.boto.stop()

    def test_the_shared_persist_module_reads_no_environment(self):
        """The 2026-08-31 stopper: the waker imported lambda_function to
        reach the flattener, and its hard os.environ reads raised KeyError
        on the waker's minimal env — outputs.json was NEVER written on the
        event path. The shared home must stay env-free, and the waker must
        never import the gateway module."""
        shared = _read("gateway_tool/synthesis_outputs.py")
        self.assertNotIn("os.environ", shared)
        self.assertNotIn("boto3", shared)
        self.assertNotIn("import lambda_function",
                         _read("gateway_tool/callback_waker.py"))

    def test_callback_registry_is_byte_identical_to_the_bedrock_master(self):
        """The row vocabulary is bedrock-inference's, copied — the pin is
        what makes a master-side fix that fails to propagate loud (same
        convention as the six bucket_validator copies)."""
        master = os.path.join(ROOT, "..", "bedrock-inference-on-aws",
                              "gateway_tool", "callback_registry.py")
        if not os.path.exists(master):
            self.skipTest("bedrock-inference master not checked out")
        with open(master) as f:
            self.assertEqual(_read("gateway_tool/callback_registry.py"),
                             f.read())

    def test_waker_gone_vocabulary_is_the_registrys(self):
        """One TOKEN_GONE tuple in the zip: a private waker copy drifts
        the moment a new benign race code lands in only one of them."""
        # the pin is structural: the waker holds no literal tuple of its
        # own, it imports the registry's
        waker_src = _read("gateway_tool/callback_waker.py")
        self.assertNotIn("_TOKEN_GONE", waker_src)
        self.assertIn("from callback_registry import TOKEN_GONE", waker_src)

    def test_fresh_submit_registers_the_token_before_start_execution(self):
        """bedrock-inference's settled ordering, inverted by the first cut
        of this port: a run whose terminal event beats the row into the
        table wakes nobody and the consumer sleeps to its 26h deadman."""
        gw = self.gw
        calls = []
        gw._ddb.put_item.side_effect = \
            lambda **kw: calls.append("register")
        gw._sfn.start_execution.side_effect = \
            lambda **kw: calls.append("start") or {"executionArn": "arn:x"}
        try:
            out = gw._start_execution(
                "arn:aws:states:us-east-1:123456789012:"
                "stateMachine:m2c-tts-synthesis",
                {"input": {}, "data": {}},
                callback_token="tok-1", with_output_prefix=True)  # nosec B106
        finally:
            gw._ddb.put_item.side_effect = None
            gw._sfn.start_execution.side_effect = None
        self.assertEqual(calls, ["register", "start"])
        self.assertEqual(out["status"], "RUNNING")
        # the pre-derived ARN equals what StartExecution would have named
        self.assertTrue(out["job_id"].startswith(
            "arn:aws:states:us-east-1:123456789012:execution:"
            "m2c-tts-synthesis:"))
        self.assertTrue(out["output_s3_prefix"].startswith("s3://"))

    def test_failed_start_deletes_the_preregistered_row(self):
        """Register-before-start's cleanup half: a row for a run that never
        started must not sit waiting for an event that cannot come."""
        from botocore.exceptions import ClientError
        gw = self.gw
        gw._ddb.delete_item.reset_mock()
        gw._sfn.exceptions.ExecutionAlreadyExists = type(
            "ExecutionAlreadyExists", (Exception,), {})
        gw._sfn.start_execution.side_effect = _client_error(
            "AccessDeniedException", "StartExecution")
        try:
            out = gw._start_execution(
                "arn:aws:states:us-east-1:123456789012:"
                "stateMachine:m2c-tts-synthesis",
                {"input": {}, "data": {}},
                callback_token="tok-2", with_output_prefix=True)  # nosec B106
        finally:
            gw._sfn.start_execution.side_effect = None
        self.assertIs(out.get("error"), True)
        gw._ddb.delete_item.assert_called_once()

    # ── waker ───────────────────────────────────────────────────────────

    def _row(self, prefix="s3://test-proxy/micro-workflow/tts-on-aws/output/w1"):
        handle = {"job_id": "arn:e", "workflow_id": "w1",
                  "status": "RUNNING", "output_s3_prefix": prefix}
        return {"execution_arn": {"S": "arn:e"},
                "token_hash": {"S": "h"},
                "token": {"S": "tok"},
                "handle": {"S": json.dumps(handle)}}

    def _drive(self, status, output=None, rows=None):
        w = self.waker
        w._ddb.query.side_effect = None
        w._ddb.query.return_value = {
            "Items": self._row() and (rows if rows is not None
                                      else [self._row()])}
        w._s3.put_object.reset_mock()
        w._sfn.describe_execution.reset_mock()
        detail = {"executionArn": "arn:e", "status": status}
        if output is not None:
            detail["output"] = output
        return w.lambda_handler({"detail": detail}, None)

    def test_succeeded_persists_from_the_event_detail_itself(self):
        """The detail carries `output` and `executionArn` — the only keys
        the flattener reads — so the happy path pays no DescribeExecution
        (it draws from the same throttle bucket every poll uses)."""
        w = self.waker
        w._sfn.send_task_success.side_effect = None
        out = json.dumps({"data": {"iterators": [
            {"out_bucket": "b", "out_prefix": "p", "outputs": [
                {"wav_key": "k0", "duration_s": 1.0,
                 "qc": {"passed": True, "similarity": 1.0}}]}]}})
        res = self._drive("SUCCEEDED", output=out)
        self.assertEqual(res["completed"], 1)
        w._sfn.describe_execution.assert_not_called()
        _, kwargs = w._s3.put_object.call_args
        self.assertTrue(kwargs["Key"].endswith("/outputs.json"))
        doc = json.loads(kwargs["Body"])
        self.assertEqual(doc["status"], "SUCCEEDED")
        self.assertEqual(len(doc["outputs"]), 1)

    def test_a_failed_run_persists_a_named_verdict_not_nothing(self):
        """Review 2026-08-31: SUCCEEDED-only persistence left a FAILED
        run's verdict solely in the callback payload — which the consumer
        discards by design — so the woken consumer died on a NoSuchKey
        naming neither tts nor the cause."""
        w = self.waker
        w._sfn.send_task_success.side_effect = None
        res = self._drive("FAILED")
        self.assertEqual(res["completed"], 1)
        _, kwargs = w._s3.put_object.call_args
        self.assertTrue(kwargs["Key"].endswith("/outputs.json"))
        doc = json.loads(kwargs["Body"])
        self.assertEqual(doc["status"], "FAILED")
        self.assertIn("error", doc)
        self.assertNotIn("outputs", doc)

    def test_send_failure_raises_so_the_async_retry_redelivers(self):
        """The old `continue` kept the row 'for a redelivery' that could
        never occur: a normal return marks the async invoke successful.
        Redelivery only happens if the invocation ERRORS."""
        w = self.waker
        w._ddb.delete_item.reset_mock()
        w._sfn.send_task_success.side_effect = _client_error(
            "ThrottlingException", "SendTaskSuccess")
        try:
            with self.assertRaises(RuntimeError):
                self._drive("SUCCEEDED", output=json.dumps({"data": {}}))
        finally:
            w._sfn.send_task_success.side_effect = None
        w._ddb.delete_item.assert_not_called()   # row kept for the retry

    def test_token_gone_is_benign_and_the_row_is_deleted(self):
        w = self.waker
        w._ddb.delete_item.reset_mock()
        w._sfn.send_task_success.side_effect = _client_error(
            "TaskTimedOut", "SendTaskSuccess")
        try:
            res = self._drive("SUCCEEDED", output=json.dumps({"data": {}}))
        finally:
            w._sfn.send_task_success.side_effect = None
        self.assertEqual(res["gone"], 1)
        w._ddb.delete_item.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
