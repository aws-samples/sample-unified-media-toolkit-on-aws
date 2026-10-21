"""
frame-inspection unit tests. Fixtures are SYNTHETIC by design (the
generalize-never-content-specific rule): card/scroll behavior is modeled
with constructed embeddings, not clips from any programme. The measured
incidents the fixtures encode are named where they apply.

Run: python3 -m unittest discover -s test -p 'test_*.py'
"""
from __future__ import annotations

import io
import json
import os
import struct
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# gateway modules import as top-level names (their zip is flat), and the
# tests import the DISPATCHER by its module name — gateway_tool goes
# FIRST so `import lambda_function` resolves there, and the root router
# is reachable as states.* / importlib when ever needed
sys.path.insert(0, os.path.join(_ROOT, "gateway_tool"))

os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("STATE_MACHINE_ARN",
                      "arn:aws:states:eu-west-1:000000000000:stateMachine:t")

from lib import fi_common                          # noqa: E402
from lib import selection                          # noqa: E402
from lib import spans                              # noqa: E402


def _npy(rows):
    """A minimal little-endian float32 2-D .npy, the shape shot-seg writes."""
    n, dim = len(rows), len(rows[0])
    header = ("{'descr': '<f4', 'fortran_order': False, "
              f"'shape': ({n}, {dim}), }}").encode("latin-1")
    pad = b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    body = b"".join(struct.pack(f"<{dim}f", *row) for row in rows)
    header = header + pad
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + body


def _manifest(frame_indices, rate="24/1"):
    return {"frame_index": list(frame_indices), "frame_rate": rate}



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


class TestSpans(unittest.TestCase):
    def _docs(self, kinds):
        return [{"shot_id": i, "shot_kind": k} for i, k in enumerate(kinds)]

    def _times(self, durations):
        out, t = {}, 0.0
        for i, d in enumerate(durations):
            out[i] = (t, t + d)
            t += d
        return out

    def test_black_bridges_inside_a_block_but_never_ends_one(self):
        kinds = ["normal", "credit_roll", "black", "credit_roll",
                 "logo_bumper", "normal"]
        times = self._times([10, 5, 2, 5, 3, 10])
        blocks = spans.nondiegetic_blocks(self._docs(kinds), times)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["shot_ids"], [1, 2, 3, 4])
        self.assertEqual(blocks[0]["kinds"], ["credit_roll", "logo_bumper"])

    def test_long_black_splits_blocks(self):
        kinds = ["credit_roll", "black", "credit_roll"]
        times = self._times([5, spans.BLACK_BRIDGE_MAX + 1, 5])
        blocks = spans.nondiegetic_blocks(self._docs(kinds), times)
        self.assertEqual([b["shot_ids"] for b in blocks], [[0], [2]])

    def test_black_alone_is_not_a_block(self):
        blocks = spans.nondiegetic_blocks(
            self._docs(["normal", "black", "normal"]), self._times([5, 2, 5]))
        self.assertEqual(blocks, [])

    def test_unreadable_text_fires_only_outside_blocks(self):
        docs = self._docs(["normal", "credit_roll", "normal"])
        docs[0]["unreadable_text_present"] = True    # a super over picture
        docs[1]["unreadable_text_present"] = True    # inside the block: no
        times = self._times([5, 5, 5])
        out = spans.unreadable_text_spans(docs, times)
        self.assertEqual([s["shot_ids"] for s in out], [[0]])


class TestSelection(unittest.TestCase):
    def test_npy_round_trip(self):
        rows = [[0.1, 0.2, 0.3], [1.0, -1.0, 0.5]]
        loaded, dim = selection.load_npy(_npy(rows))
        self.assertEqual(dim, 3)
        self.assertAlmostEqual(loaded[1][2], 0.5, places=6)

    def test_npy_refuses_non_f4(self):
        blob = bytearray(_npy([[1.0, 2.0]]))
        blob = blob.replace(b"<f4", b"<f8")
        with self.assertRaises(ValueError):
            selection.load_npy(bytes(blob))

    def test_static_cards_collapse_and_cuts_are_kept(self):
        # three cards, 4 frames each, at 1 fps: 12 frames -> 3 kept
        card_a, card_b, card_c = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        rows = [card_a] * 4 + [card_b] * 4 + [card_c] * 4
        manifest = _manifest(range(0, 12 * 24, 24))   # 24 fps, 1 frame/s
        kept = selection.select(manifest, _npy(rows),
                                {"start": 0, "end": 12},
                                change_distance=0.25, heartbeat_sec=1e9)
        self.assertEqual(kept, [0, 4, 8])

    def test_heartbeat_samples_through_a_slow_scroll(self):
        # the mitj404 cast-roll incident: a crawl whose consecutive frames
        # differ under the change gate must still be sampled every 6 s
        rows = [[1.0, 0.001 * i] for i in range(20)]    # tiny drift
        manifest = _manifest(range(0, 20 * 24, 24))
        kept = selection.select(manifest, _npy(rows),
                                {"start": 0, "end": 20},
                                change_distance=0.25, heartbeat_sec=6.0)
        self.assertEqual(kept, [0, 6, 12, 18])

    def test_span_bounds_are_respected(self):
        rows = [[1.0, 0.0]] * 10
        manifest = _manifest(range(0, 10 * 24, 24))
        kept = selection.select(manifest, _npy(rows),
                                {"start": 3.0, "end": 5.0},
                                change_distance=0.25, heartbeat_sec=1e9)
        self.assertEqual(selection.frame_names(manifest, kept)[0][0], 3.0)

    def test_frame_names_use_source_frame_numbers(self):
        manifest = _manifest([37403, 37409], rate="24000/1001")
        names = selection.frame_names(manifest, [0, 1])
        self.assertEqual(names[0][1], "frame_0000037403.jpg")
        self.assertAlmostEqual(names[0][0], 37403 * 1001 / 24000, places=2)

    def test_evenly_spaced_caps_with_both_endpoints(self):
        seq = list(range(1000))
        capped = selection.evenly_spaced(seq, 240)
        self.assertEqual(len(capped), 240)
        self.assertEqual((capped[0], capped[-1]), (0, 999))
        self.assertEqual(capped, sorted(capped))
        # under the cap: untouched
        self.assertEqual(selection.evenly_spaced([1, 2, 3], 240), [1, 2, 3])


class TestCategoryTable(unittest.TestCase):
    def test_every_category_names_its_contract(self):
        for name, conf in fi_common.CATEGORIES.items():
            self.assertIn("trigger", conf, name)
            self.assertIn("product", conf, name)
            self.assertIn("needs_question", conf, name)
            # request category = the bare task (API vocabulary);
            # registry category = the owner-prefixed taxonomy name
            self.assertEqual(conf["prompt_category"], f"fi-{name}", name)
            if conf["trigger"] is not None:
                self.assertTrue(hasattr(spans, conf["trigger"]),
                                f"{name}: trigger {conf['trigger']} is not a "
                                "spans predicate")

    def test_claim_verify_is_explicit_only_with_questions(self):
        conf = fi_common.CATEGORIES["claim-verify"]
        self.assertIsNone(conf["trigger"])
        self.assertTrue(conf["needs_question"])

    def test_selection_knobs_pin_the_measured_values(self):
        """change_distance is MEASURED (fi-dev live validation, mitj404):
        within-card <=0.026 vs real cuts 0.035-0.26. A revert to the
        first-guess 0.25 loses card cuts silently, and a deleted key is a
        runtime KeyError in SelectFrames — pin both the presence and the
        numbers."""
        for name, conf in fi_common.CATEGORIES.items():
            self.assertIn("change_distance", conf, name)
            self.assertIn("heartbeat_sec", conf, name)
        cd = {n: c["change_distance"]
              for n, c in fi_common.CATEGORIES.items()}
        self.assertEqual(cd, {"credits-read": 0.03, "title-read": 0.03,
                              "claim-verify": 0.15})
        self.assertGreater(fi_common.MAX_FRAMES_PER_SPAN, 0)


class TestValidatePlanShapes(unittest.TestCase):
    def test_single_asset_sugar_normalizes(self):
        from states.validate_plan import lambda_function as validate_plan
        assets = validate_plan._normalize_assets({
            "shot_segmentation_s3uri": "s3://b/x",
            "category": "credits-read",
        })
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]["requests"][0]["category"], "credits-read")

    def test_claim_verify_span_without_question_refuses(self):
        from states.validate_plan import lambda_function as validate_plan
        with self.assertRaises(validate_plan.FrameInspectionValidationError):
            validate_plan._resolve_spans(
                {}, {"category": "claim-verify",
                     "spans": [{"start": 1.0, "end": 2.0}]},
                fi_common.CATEGORIES["claim-verify"])

    def test_default_chain_is_mode_aware(self):
        # Sync reads ride the cheaper Sonnet-5-led chain; batch keeps the
        # batch-capable 4.6 (Sonnet 5 has no batch support). Mirrors
        # bedrock-inference's chain-less default (user, 2026-09-23).
        self.assertEqual(fi_common.default_model_chain(True),
                         ["anthropic.claude-sonnet-4-6"])
        self.assertEqual(fi_common.default_model_chain(False),
                         ["anthropic.claude-sonnet-5-5",
                          "anthropic.claude-sonnet-4-6"])
        # a COPY each call — mutation must not leak into the next plan
        chain = fi_common.default_model_chain(False)
        chain.append("mutated")
        self.assertNotIn("mutated", fi_common.default_model_chain(False))

    def test_bad_span_refuses(self):
        from states.validate_plan import lambda_function as validate_plan
        with self.assertRaises(validate_plan.FrameInspectionValidationError):
            validate_plan._resolve_spans(
                {}, {"category": "credits-read",
                     "spans": [{"start": 5.0, "end": 5.0}]},
                fi_common.CATEGORIES["credits-read"])


class TestCollectAssembly(unittest.TestCase):
    def test_verdict_enum_is_policed(self):
        from states.collect_finalize import lambda_function as collect_finalize
        good = ({"start": 1, "end": 2, "id": "a"}, 0,
                [{"verdict": "SUPPORTED", "evidence": "seen"}])
        bad = ({"start": 3, "end": 4, "id": "b"}, 1,
               [{"verdict": "probably", "evidence": "?"}])
        product = collect_finalize._assemble("claim-verify", [good, bad], [])
        self.assertEqual(product["verdicts"],
                         [{"id": "a", "verdict": "supported",
                           "evidence": "seen"}])
        self.assertEqual(len(product["gaps"]), 1)
        self.assertIn("outside", product["gaps"][0]["reason"])

    def test_cards_sort_and_reject_malformed(self):
        from states.collect_finalize import lambda_function as collect_finalize
        rows = [{"t": 9.0, "lines": ["B"]}, {"t": 3.0, "lines": ["A"]},
                {"t": "not-a-time", "lines": ["X"]}, {"lines": ["Y"]}]
        product = collect_finalize._assemble(
            "credits-read", [({"start": 0, "end": 10}, 0, rows)], [])
        cards = product["blocks"][0]["cards"]
        self.assertEqual([c["lines"][0] for c in cards], ["A", "B"])

    def test_cards_carry_validated_pairings(self):
        from states.collect_finalize import lambda_function as collect_finalize
        rows = [{"t": 1.0, "lines": ["LENA VOSS    Mira Santos"],
                 "pairings": [
                     {"name": "LENA VOSS",
                      "paired_with": "Mira Santos", "header": None},
                     {"name": "", "paired_with": "dropped"},   # no name
                     "not-a-dict",
                     {"name": "TOM BAYLES", "paired_with": None,
                      "header": "Key Asst. Harbor Managers"}]},
                {"t": 2.0, "lines": ["plain card"]}]
        product = collect_finalize._assemble(
            "credits-read", [({"start": 0, "end": 10}, 0, rows)], [])
        cards = product["blocks"][0]["cards"]
        self.assertEqual(len(cards[0]["pairings"]), 2)
        self.assertEqual(cards[0]["pairings"][1]["header"],
                         "Key Asst. Harbor Managers")
        self.assertNotIn("pairings", cards[1])   # absent, not empty

    def test_non_list_pairings_degrades_card_not_state(self):
        # a model answering `"pairings": 3` (or a dict) must cost the
        # card its pairings, never TypeError after the inference spend
        from states.collect_finalize import lambda_function as collect_finalize
        rows = [{"t": 1.0, "lines": ["A"], "pairings": 3},
                {"t": 2.0, "lines": ["B"], "pairings": {"name": "X"}}]
        product = collect_finalize._assemble(
            "credits-read", [({"start": 0, "end": 10}, 0, rows)], [])
        cards = product["blocks"][0]["cards"]
        self.assertEqual(len(cards), 2)
        for card in cards:
            self.assertNotIn("pairings", card)

    def test_attested_empty_pairings_survive_but_garbage_degrades(self):
        # []-vs-absent (review, 2026-09-06): "pairings": [] is an
        # ATTESTATION — the model looked and paired nothing — and the
        # consumer must not fall back to text heuristics on it. A
        # nonempty list whose entries ALL fail validation is a DEGRADED
        # answer and omits the key so the consumer's fallback applies.
        from states.collect_finalize import lambda_function as collect_finalize
        rows = [{"t": 1.0, "lines": ["names only"], "pairings": []},
                {"t": 2.0, "lines": ["garbled"],
                 "pairings": [{"name": ""}, "not-a-dict"]}]
        product = collect_finalize._assemble(
            "credits-read", [({"start": 0, "end": 10}, 0, rows)], [])
        cards = product["blocks"][0]["cards"]
        self.assertEqual(cards[0]["pairings"], [])   # attested none
        self.assertNotIn("pairings", cards[1])       # degraded

    def test_parse_span_reports_honest_gaps(self):
        from states.collect_finalize import lambda_function as collect_finalize
        parsed, why = collect_finalize._parse_span({}, 0, "credits-read", 0)
        self.assertIsNone(parsed)
        self.assertIn("no record", why)


class TestGateway(unittest.TestCase):
    class _Ctx:
        def __init__(self, tool):
            custom = {"bedrockAgentCoreToolName": f"frame-inspection___{tool}"}
            self.client_context = type("CC", (), {"custom": custom})()

    def _gw(self):
        import lambda_function
        return lambda_function

    def test_refusals(self):
        gw = self._gw()
        with mock.patch.object(gw, "PROXY_BUCKET", "b"), \
             mock.patch.object(gw, "STATE_MACHINE_ARN", "arn:x"):
            self.assertEqual(
                gw.lambda_handler({}, self._Ctx("inspect_frames"))["errorCode"],
                "MISSING_PARAMETER")
            self.assertEqual(
                gw.lambda_handler({"shot_segmentation_s3uri": "s3://b/k"},
                                  self._Ctx("inspect_frames"))["errorCode"],
                "MISSING_PARAMETER")
            self.assertEqual(
                gw.lambda_handler({"assets": "nope"},
                                  self._Ctx("inspect_frames"))["errorCode"],
                "INVALID_PARAMETER")

    def test_describe_tool_serves_the_contract(self):
        gw = self._gw()
        resp = gw.lambda_handler({}, self._Ctx("describe_tool"))
        self.assertIn("version", resp)
        self.assertEqual(
            sorted(resp["output_contract"]["outputs"]),
            ["claim-verify", "credits-read", "manifest", "title-read",
             "usage"])

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        path = os.path.join(_ROOT, "gateway_tool", "input_examples.json")
        examples = _load_json(path)["tools"]
        self.assertEqual(sorted(examples),
                         ["describe_tool", "get_job_status",
                          "inspect_frames"])
        for tool, entry in examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        gw = self._gw()
        resp = gw.lambda_handler({}, self._Ctx("describe_tool"))
        self.assertIn("inspect_frames", resp["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        gw = self._gw()
        path = os.path.join(_ROOT, "gateway_tool", "input_examples.json")
        example = _load_json(path)["tools"]["inspect_frames"]["example"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gw, "_sfn", fake):
            out = gw.lambda_handler(dict(example),
                                    self._Ctx("inspect_frames"))
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()

    def test_staged_alternative_passes_the_gateway_validation(self):
        # the pooled `assets` form — the other arm of the one-of rule —
        # must round-trip the same validation as the single-asset sugar
        gw = self._gw()
        path = os.path.join(_ROOT, "gateway_tool", "input_examples.json")
        alt = (_load_json(path)["tools"]["inspect_frames"]
               ["alternatives"]["assets"])
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gw, "_sfn", fake):
            out = gw.lambda_handler(dict(alt), self._Ctx("inspect_frames"))
        self.assertNotIn("error", out, f"alternative refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()
        sent = json.loads(fake.start_execution.call_args.kwargs["input"])
        self.assertEqual(sent["input"]["assets"], alt["assets"])

    def test_identity_semantics(self):
        from workflow_identity import compute_workflow_id
        req = {"assets": [{"shot_segmentation_s3uri": "s3://b/x",
                           "requests": [{"category": "credits-read"}]}]}
        same = json.loads(json.dumps(req))
        self.assertEqual(compute_workflow_id(req, tool_version="t1"),
                         compute_workflow_id(same, tool_version="t1"))
        self.assertNotEqual(compute_workflow_id(req, tool_version="t1"),
                            compute_workflow_id(req, tool_version="t2"))
        q1 = {"assets": [{"requests": [{"category": "claim-verify",
                                        "spans": [{"start": 1, "end": 2,
                                                   "question": "A?"}]}]}]}
        q2 = json.loads(json.dumps(q1))
        q2["assets"][0]["requests"][0]["spans"][0]["question"] = "B?"
        self.assertNotEqual(compute_workflow_id(q1, tool_version="t1"),
                            compute_workflow_id(q2, tool_version="t1"))


class TestCopiedModulesAreByteIdentical(unittest.TestCase):
    """The family rule: copied modules never drift from their masters."""
    MASTERS = {
        "lib/prompt_render.py":
            "../prompt-registry-on-aws/gateway_tool/prompt_render.py",
        "lib/workflow_history.py":
            "../prompt-registry-on-aws/gateway_tool/workflow_history.py",
        "gateway_tool/workflow_identity.py":
            "../conversation-analysis-on-aws/gateway_tool/workflow_identity.py",
        "lib/parse_partial_output.py":
            "../conversation-analysis-on-aws/lib/parse_partial_output.py",
    }

    def test_pinned(self):
        for local, master in self.MASTERS.items():
            master_path = os.path.join(_ROOT, master)
            if not os.path.exists(master_path):
                continue        # master repo absent in this checkout
            with self.subTest(local):
                with open(os.path.join(_ROOT, local), "rb") as a, \
                     open(master_path, "rb") as b:
                    self.assertEqual(a.read(), b.read(),
                                     f"{local} drifted from {master}")


class TestAslDrift(unittest.TestCase):
    def test_cfnstack_inline_matches_definitions_json(self):
        import re
        text = _read(os.path.join(_ROOT, "cfnstack.yaml"))
        m = re.search(r"DefinitionString: \|-\n((?:                .*\n|\n)+)",
                      text)
        self.assertIsNotNone(m)
        inline = json.loads("\n".join(l[16:] for l in m.group(1).splitlines()))
        ref = _load_json(os.path.join(_ROOT, "definitions.json"))
        self.assertEqual(inline, ref,
                         "the template's inline ASL drifted from "
                         "definitions.json — edit BOTH")




# ─────────────────────────────────────────────────────────────────────────────
# callback_task_token (copied shape: README.md §11)
# ─────────────────────────────────────────────────────────────────────────────

class TestCallbackToken(unittest.TestCase):
    def _gw(self):
        import lambda_function as gw
        return gw

    def _ctx(self):
        class _C:
            pass
        c = _C()
        c.client_context = type(
            "CC", (), {"custom": {"bedrockAgentCoreToolName":
                                  "frame-inspection___inspect_frames"}})()
        return c

    def _event(self, **extra):
        return {"shot_segmentation_s3uri": "s3://p/micro-workflow/ss/x",
                "category": "credits-read", **extra}

    def test_bad_token_raises_never_returns(self):
        # a waitForTaskToken caller cannot read a returned error dict
        gw = self._gw()
        with self.assertRaises(ValueError):
            gw.lambda_handler(self._event(callback_task_token="  "),  # nosec B106
                              self._ctx())

    def test_token_without_table_raises(self):
        gw = self._gw()
        old = gw.CALLBACK_TABLE
        gw.CALLBACK_TABLE = ""
        try:
            with self.assertRaises(RuntimeError):
                gw.lambda_handler(self._event(callback_task_token="tok"),  # nosec B106
                                  self._ctx())
        finally:
            gw.CALLBACK_TABLE = old

    def test_register_happens_before_start_execution(self):
        gw = self._gw()
        order = []
        with mock.patch.object(gw, "CALLBACK_TABLE", "test-callbacks"), \
             mock.patch.object(gw.callback_registry, "register",
                               side_effect=lambda *a, **k:
                               order.append("register")), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=lambda **k:
                               order.append("start") or
                               {"executionArn": "arn:aws:states:r:1:execution:m:x"}):
            out = gw.lambda_handler(
                self._event(callback_task_token="tok"), self._ctx())  # nosec B106
        self.assertEqual(order, ["register", "start"])
        self.assertEqual(out["status"], "RUNNING")

    def test_any_error_return_with_a_token_raises_at_the_choke_point(self):
        # CONFIG_ERROR runs before the token block; the lambda_handler
        # choke point must convert its return into a raise (review)
        gw = self._gw()
        with mock.patch.object(gw, "PROXY_BUCKET", ""):
            with self.assertRaises(RuntimeError) as cm:
                gw.lambda_handler(
                    {"shot_segmentation_s3uri": "s3://p/x",
                     "category": "credits-read",
                     "callback_task_token": "tok"}, self._ctx())  # nosec B105
        self.assertIn("CONFIG_ERROR", str(cm.exception))
        # and without a token the same misconfig still RETURNS
        with mock.patch.object(gw, "PROXY_BUCKET", ""):
            out = gw.lambda_handler(
                {"shot_segmentation_s3uri": "s3://p/x",
                 "category": "credits-read"}, self._ctx())
        self.assertEqual(out["errorCode"], "CONFIG_ERROR")

    def test_start_failure_with_token_deletes_row_and_raises(self):
        from botocore.exceptions import ClientError
        gw = self._gw()
        boom = ClientError({"Error": {"Code": "ThrottlingException",
                                      "Message": "x"}}, "StartExecution")
        with mock.patch.object(gw, "CALLBACK_TABLE", "test-callbacks"), \
             mock.patch.object(gw.callback_registry, "register"), \
             mock.patch.object(gw.callback_registry, "delete") as gone, \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=boom):
            with self.assertRaises(ClientError):
                gw.lambda_handler(self._event(callback_task_token="tok"),  # nosec B106
                                  self._ctx())
        self.assertTrue(gone.called)

    def test_request_refusal_with_token_raises_never_returns(self):
        # the choke point is the ONE raise-vs-return mechanism: the site
        # returns a plain _error dict and the wrapper converts it
        gw = self._gw()
        with mock.patch.object(gw, "CALLBACK_TABLE", "test-callbacks"):
            with self.assertRaises(RuntimeError):
                gw.lambda_handler(
                    {"shot_segmentation_s3uri": "s3://p/x",
                     "callback_task_token": "tok"},   # category missing  # nosec B105
                    self._ctx())

    def test_describe_tool_advertises_the_capability(self):
        gw = self._gw()
        ctx = self._ctx()
        ctx.client_context.custom["bedrockAgentCoreToolName"] = \
            "frame-inspection___describe_tool"
        with mock.patch.object(gw, "CALLBACK_TABLE", "test-callbacks"):
            out = gw.lambda_handler({}, ctx)
        self.assertIn("callback_task_token", out["features"])

    def test_waker_raises_on_stuck_token_so_eventbridge_retries(self):
        import callback_waker as w
        from botocore.exceptions import ClientError
        boom = ClientError({"Error": {"Code": "ThrottlingException",
                                      "Message": "x"}}, "SendTaskSuccess")
        with mock.patch.object(w, "CALLBACK_TABLE", "t"), \
             mock.patch.object(w._ddb, "query", return_value={"Items": [
                 {"execution_arn": {"S": "arn:x"},
                  "token_hash": {"S": "h"},
                  "token": {"S": "tok"},
                  "handle": {"S": "{}"}}]}), \
             mock.patch.object(w._sfn, "send_task_success",
                               side_effect=boom), \
             mock.patch.object(w._ddb, "delete_item") as del_:
            with self.assertRaises(RuntimeError):
                w.lambda_handler({"detail": {"executionArn": "arn:x",
                                             "status": "SUCCEEDED"}}, None)
        self.assertFalse(del_.called)    # row kept for the retry

    def test_absorbed_closed_run_completes_token_synchronously(self):
        from botocore.exceptions import ClientError
        gw = self._gw()
        patcher = mock.patch.object(gw, "CALLBACK_TABLE",
                                    "test-callbacks")
        patcher.start()
        self.addCleanup(patcher.stop)
        exists = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        with mock.patch.object(gw.callback_registry, "register"), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=exists), \
             mock.patch.object(gw._sfn, "describe_execution",
                               return_value={"status": "SUCCEEDED"}), \
             mock.patch.object(gw.callback_registry,
                               "complete_now") as done, \
             mock.patch.object(gw.callback_registry, "delete") as gone:
            out = gw.lambda_handler(
                self._event(callback_task_token="tok"), self._ctx())  # nosec B106
        self.assertTrue(done.called)
        self.assertTrue(gone.called)
        self.assertIn("message", out)

    def test_absorbed_failed_run_is_data_not_a_detonation(self):
        # the absorbed-into-already-FAILED reconcile: complete_now has
        # redeemed the token with FAILED-as-data — the choke point must
        # NOT raise on the error STRING riding beside the job handle
        # (review, 2026-09-06: only the _error refusal shape, error is
        # True, raises)
        from botocore.exceptions import ClientError
        gw = self._gw()
        exists = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        with mock.patch.object(gw, "CALLBACK_TABLE", "test-callbacks"), \
             mock.patch.object(gw.callback_registry, "register"), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=exists), \
             mock.patch.object(gw._sfn, "describe_execution",
                               return_value={"status": "FAILED",
                                             "cause": "boom"}), \
             mock.patch.object(gw.callback_registry,
                               "complete_now") as done, \
             mock.patch.object(gw.callback_registry, "delete") as gone:
            out = gw.lambda_handler(
                self._event(callback_task_token="tok"), self._ctx())  # nosec B106
        self.assertTrue(done.called)      # token redeemed, FAILED as data
        self.assertTrue(gone.called)
        self.assertEqual(out["error"], "boom")
        self.assertIn("job_id", out)

    def test_absorb_describe_failure_deletes_row_and_raises(self):
        # ExecutionAlreadyExists then describe fails: the raise tells the
        # holder the submit failed (it will consume its own token), so
        # the row THIS call registered must go — StartExecution-failure
        # symmetry (review, 2026-09-05)
        from botocore.exceptions import ClientError
        gw = self._gw()
        exists = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        boom = ClientError({"Error": {"Code": "ThrottlingException",
                                      "Message": "x"}}, "DescribeExecution")
        with mock.patch.object(gw, "CALLBACK_TABLE", "test-callbacks"), \
             mock.patch.object(gw.callback_registry, "register"), \
             mock.patch.object(gw.callback_registry, "delete") as gone, \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=exists), \
             mock.patch.object(gw._sfn, "describe_execution",
                               side_effect=boom):
            with self.assertRaises(ClientError):
                gw.lambda_handler(self._event(callback_task_token="tok"),  # nosec B106
                                  self._ctx())
        self.assertTrue(gone.called)

    def test_absorbed_running_run_leaves_token_to_the_waker(self):
        from botocore.exceptions import ClientError
        gw = self._gw()
        patcher = mock.patch.object(gw, "CALLBACK_TABLE",
                                    "test-callbacks")
        patcher.start()
        self.addCleanup(patcher.stop)
        exists = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        with mock.patch.object(gw.callback_registry, "register"), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=exists), \
             mock.patch.object(gw._sfn, "describe_execution",
                               return_value={"status": "RUNNING"}), \
             mock.patch.object(gw, "_current_state", return_value=None), \
             mock.patch.object(gw.callback_registry,
                               "complete_now") as done:
            gw.lambda_handler(self._event(callback_task_token="tok"),  # nosec B106
                              self._ctx())
        self.assertFalse(done.called)

    def test_waker_sends_success_with_status_as_data(self):
        import callback_waker as w
        with mock.patch.object(w, "CALLBACK_TABLE", "t"), \
             mock.patch.object(w._ddb, "query", return_value={"Items": [
                 {"execution_arn": {"S": "arn:x"},
                  "token_hash": {"S": "h"},
                  "token": {"S": "tok"},
                  "handle": {"S": "{\"workflow_id\": \"w\"}"}}]}), \
             mock.patch.object(w._sfn, "send_task_success") as sts, \
             mock.patch.object(w._ddb, "delete_item") as del_:
            out = w.lambda_handler(
                {"detail": {"executionArn": "arn:x", "status": "FAILED"}},
                None)
        self.assertEqual(out["completed"], 1)
        import json as _json
        payload = _json.loads(sts.call_args.kwargs["output"])
        self.assertEqual(payload["status"], "FAILED")   # data, not detonation
        self.assertTrue(del_.called)
class TestLowerThirdCrop(unittest.TestCase):
    """The witness's one image operation: caller-requested per-span
    lower-third cropping (pyvips; the container's reason to exist)."""

    def setUp(self):
        try:
            import pyvips  # noqa: F401
        except ImportError:                          # pragma: no cover
            self.skipTest("pyvips not installed locally")

    def _jpeg(self, w=512, h=288):
        import pyvips
        return (pyvips.Image.black(w, h) + 128).jpegsave_buffer(Q=80)

    def test_crop_keeps_the_bottom_band(self):
        from lib import imaging
        import pyvips
        out = imaging.crop(self._jpeg(), "lower_third")
        img = pyvips.Image.new_from_buffer(out, "")
        self.assertEqual(img.width, 512)
        self.assertEqual(img.height,
                         int(288 * imaging.LOWER_THIRD_FRACTION))

    def test_unknown_mode_raises(self):
        from lib import imaging
        with self.assertRaises(ValueError):
            imaging.crop(self._jpeg(), "upper_half")

    def test_plan_refuses_a_bad_crop_mode(self):
        from states.validate_plan import lambda_function as validate_plan
        with self.assertRaises(
                validate_plan.FrameInspectionValidationError):
            validate_plan._resolve_spans(
                {}, {"category": "title-read",
                     "spans": [{"start": 1.0, "end": 2.0,
                                "crop": "sideways"}]},
                fi_common.CATEGORIES["title-read"])

    def test_plan_carries_a_valid_crop(self):
        from states.validate_plan import lambda_function as validate_plan
        spans = validate_plan._resolve_spans(
            {}, {"category": "title-read",
                 "spans": [{"start": 1.0, "end": 2.0,
                            "crop": "lower_third"}]},
            fi_common.CATEGORIES["title-read"])
        self.assertEqual(spans[0]["crop"], "lower_third")

    def test_render_crops_span_frames(self):
        import base64
        from states.select_frames import lambda_function as select_frames
        import pyvips
        from unittest import mock
        plan = {"assets": [{"asset_idx": 0, "requests": [
                    {"category": "title-read",
                     "spans": [{"start": 0.0, "end": 2.0, "id": "s0",
                                "crop": "lower_third"}]}]}],
                "prompts": {"title-read": {
                    "system": "sys", "user_template": "{{frame_images}}",
                    "variables": [
                        {"token": "{{frame_images}}",  # nosec B105
                         "root": "frame_images", "leafs": []},
                        {"token": "{{span_context}}",  # nosec B105
                         "root": "span_context", "leafs": []}]},
                },
                "images_per_record": 18}
        # the template must also declare span_context for render()
        plan["prompts"]["title-read"]["user_template"] = \
            "{{span_context}}\n{{frame_images}}"
        frames = [(0.5, self._jpeg())]
        with mock.patch.object(select_frames, "_select_span_frames",
                               return_value=frames):
            records = select_frames._render_unit(
                plan, {"asset_idx": 0, "category": "title-read",
                       "span_idx": 0})
        img_b64 = next(
            b["image"]["source"]["bytes"]
            for b in records[0]["messages"][0]["content"]
            if "image" in b)
        img = pyvips.Image.new_from_buffer(
            base64.b64decode(img_b64), "")
        self.assertEqual(img.height,
                         int(288 * 0.4))   # cropped, not full-frame


if __name__ == "__main__":
    unittest.main()
