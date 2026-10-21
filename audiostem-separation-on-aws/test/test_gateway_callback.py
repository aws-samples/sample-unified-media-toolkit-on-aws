# test/test_gateway_callback.py
"""Hermetic tests for the gateway+callback conversion (2026-09-18):
the family sibling pattern replacing the direct startExecution.sync:2
of media-processing's state machine.

Covers lib/gateway.py, StateSubmitMediaProcessing (the held state's
handler), StateExtractProcessedAudio's two reader branches, and the
definitions.json / cfnstack.yaml inline-ASL parity gate — the inline
DefinitionString is what deploys, definitions.json is the shipped
reference copy, and nothing else keeps them in sync.

No AWS: boto3 is stubbed before the handler modules load (they build
clients at import time), and the WAV probe is monkey-patched out.

    python3 -m unittest discover -s test -p 'test_*.py'
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_module(rel_path: str, name: str):
    """Load a handler by file path with boto3/botocore stubbed for the
    duration of the exec — the modules build clients at import time.
    The previous sys.modules entries are restored in the finally so a
    leaked stub cannot disarm other tests in the run."""
    saved = {k: sys.modules.get(k)
             for k in ("boto3", "botocore", "botocore.config",
                       "boto3.s3", "boto3.s3.transfer")}
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *a, **k: object()
    boto3.s3 = types.ModuleType("boto3.s3")
    transfer = types.ModuleType("boto3.s3.transfer")
    transfer.TransferConfig = lambda *a, **k: object()
    botocore = types.ModuleType("botocore")
    bc_config = types.ModuleType("botocore.config")
    bc_config.Config = lambda *a, **k: object()
    sys.modules.update({"boto3": boto3, "botocore": botocore,
                        "botocore.config": bc_config,
                        "boto3.s3": boto3.s3,
                        "boto3.s3.transfer": transfer})
    try:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(ROOT, rel_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


from lib.gateway import invoke_gateway_tool, require_callback_feature  # noqa: E402

submit_mod = _load_module(
    "states/submit_media_processing/lambda_function.py", "submit_mp")
extract_mod = _load_module(
    "states/extract_processed_audio/lambda_function.py", "extract_mp")

# ── Gateway dispatcher (gateway_tool/lambda_function.py) ─────────────
# Loaded by file path — a bare `import lambda_function` would collide
# with the repo-root operation router. Real boto3 builds its clients at
# import (no call is ever made; the tests patch _sfn); env mirrors the
# deployed function's required configuration.
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault(
    "AUDIOSTEM_STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:111122223333:stateMachine:audiostem")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("TARGET_NAME", "audio-stem")
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))  # workflow_identity

_gw_spec = importlib.util.spec_from_file_location(
    "audiostem_gateway",
    os.path.join(ROOT, "gateway_tool", "lambda_function.py"))
gateway_mod = importlib.util.module_from_spec(_gw_spec)
_gw_spec.loader.exec_module(gateway_mod)



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


class FakeLambda:
    """Captures invoke() kwargs and replays canned responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        body = json.dumps(r.get("body", {})).encode()
        out = {"Payload": io.BytesIO(body)}
        if r.get("FunctionError"):
            out["FunctionError"] = r["FunctionError"]
        return out


class TestGatewayHelper(unittest.TestCase):

    def test_tool_routing_rides_client_context(self):
        lam = FakeLambda([{"body": {"ok": 1}}])
        out = invoke_gateway_tool(lam, "arn:gw", "media-processing",
                                  "process_media", {"a": 1})
        self.assertEqual(out, {"ok": 1})
        call = lam.calls[0]
        ctx = json.loads(base64.b64decode(call["ClientContext"]))
        self.assertEqual(
            ctx, {"custom": {"bedrockAgentCoreToolName":
                             "media-processing___process_media"}})
        self.assertEqual(json.loads(call["Payload"]), {"a": 1})

    def test_function_error_raises_not_parsed_as_refusal(self):
        lam = FakeLambda([{"body": {"errorMessage": "boom"},
                           "FunctionError": "Unhandled"}])
        with self.assertRaisesRegex(RuntimeError,
                                    "media-processing___process_media"):
            invoke_gateway_tool(lam, "arn:gw", "media-processing",
                                "process_media", {})

    def test_feature_probe_accepts_and_refuses(self):
        ok = FakeLambda([{"body": {"features": ["callback_task_token"]}}])
        require_callback_feature(ok, "arn:gw", "media-processing")

        stale = FakeLambda([{"body": {"version": "old"}}])
        with self.assertRaisesRegex(RuntimeError,
                                    "predates callback support"):
            require_callback_feature(stale, "arn:gw", "media-processing")


class TestSubmitMediaProcessing(unittest.TestCase):
    """The held state's handler: refuse-before-spend probe, then the
    submit with this run's uuid as the idempotency token. MP is a HARD
    dependency, so every failure path raises rather than self-skips."""

    EVENT = {
        "uuid": "abc-123",
        "task_token": "TOKEN",  # nosec B105
        "mp_gateway_arn": "arn:gw",
        "data": {"media_processing_request": {
            "asset_s3uri": "s3://in/ep1.mxf", "output_groups": []}},
    }

    def _patch(self, probe=None, submit=None):
        calls = []

        def fake_probe(lam, arn, target):
            calls.append(("probe", target))
            if probe:
                raise probe

        def fake_invoke(lam, arn, target, tool, payload):
            calls.append(("invoke", target, tool, payload))
            return submit

        self._orig = (submit_mod.require_callback_feature,
                      submit_mod.invoke_gateway_tool)
        submit_mod.require_callback_feature = fake_probe
        submit_mod.invoke_gateway_tool = fake_invoke
        self.addCleanup(self._restore)
        return calls

    def _restore(self):
        (submit_mod.require_callback_feature,
         submit_mod.invoke_gateway_tool) = self._orig

    def test_missing_gateway_arn_raises(self):
        event = {**self.EVENT, "mp_gateway_arn": ""}
        os.environ.pop("MEDIA_PROCESSING_GATEWAY_ARN", None)
        with self.assertRaisesRegex(RuntimeError, "gateway ARN"):
            submit_mod.lambda_handler(event, None)

    def test_missing_request_raises(self):
        event = {**self.EVENT, "data": {}}
        with self.assertRaisesRegex(RuntimeError,
                                    "media_processing_request"):
            submit_mod.lambda_handler(event, None)

    def test_submit_parks_with_uuid_derived_idempotency(self):
        calls = self._patch(submit={"job_id": "arn:exec", "status": "RUNNING",
                                    "workflow_id": "wf1"})
        out = submit_mod.lambda_handler(dict(self.EVENT), None)
        self.assertEqual(out, {"parked": True, "job_id": "arn:exec"})
        kind, target, tool, payload = calls[-1]
        self.assertEqual((kind, target, tool),
                         ("invoke", "media-processing", "process_media"))
        self.assertEqual(payload["idempotency_token"], "audiostem-abc-123")
        self.assertEqual(payload["callback_task_token"], "TOKEN")
        self.assertEqual(payload["asset_s3uri"], "s3://in/ep1.mxf")

    def test_handle_less_response_is_a_refusal_and_raises(self):
        self._patch(submit={"error": True, "message": "bad params"})
        with self.assertRaisesRegex(RuntimeError, "refused"):
            submit_mod.lambda_handler(dict(self.EVENT), None)

    def test_probe_refusal_stops_before_the_submit(self):
        calls = self._patch(probe=RuntimeError("predates callback support"),
                            submit={"job_id": "never"})
        with self.assertRaisesRegex(RuntimeError, "predates"):
            submit_mod.lambda_handler(dict(self.EVENT), None)
        self.assertEqual([c[0] for c in calls], ["probe"])


class TestExtractProcessedAudio(unittest.TestCase):
    """Reader compatibility: the gateway handle (status AS DATA — a
    FAILED verdict is enforced HERE) and the legacy nested-execution
    shape kept for runs in flight across the deploy boundary."""

    CFG = {"sample_rate": 48000, "num_channels": 1,
           "bits_per_sample": 16, "duration_sec": 12.0}

    def setUp(self):
        self._orig = extract_mod._probe_audio_cfg
        extract_mod._probe_audio_cfg = lambda b, k: dict(self.CFG)
        self.addCleanup(
            lambda: setattr(extract_mod, "_probe_audio_cfg", self._orig))

    def _event(self, mp_out):
        return {"uuid": "u", "data": {
            "media_processing_request": {"asset_s3uri": "s3://in/a.mxf"},
            "media_processing_output": mp_out}}

    def test_gateway_handle_succeeded_derives_the_contract_key(self):
        out = extract_mod.lambda_handler(self._event({
            "job_id": "arn:exec:mp",
            "workflow_id": "wf1",
            "status": "SUCCEEDED",
            "output_s3_prefix":
                "s3://proxy/micro-workflow/media-processing/wf1"}), None)
        data = out["data"]
        self.assertEqual(data["audio_bucket"], "proxy")
        self.assertEqual(
            data["audio_key"],
            "micro-workflow/media-processing/wf1/"
            "audiostem_sep/audio_48k_mono.wav")
        self.assertEqual(data["media_processing_arn"], "arn:exec:mp")
        # usage tracking reads MP's published bill from here (no
        # cross-machine IAM — the machine-ARN parameter is gone)
        self.assertEqual(
            data["media_processing_usage_key"],
            "micro-workflow/media-processing/wf1/usage_tracking.json")
        self.assertIsNone(data["media_processing_request"])
        self.assertIsNone(data["media_processing_output"])

    def test_gateway_handle_failure_status_raises(self):
        with self.assertRaisesRegex(RuntimeError, "FAILED"):
            extract_mod.lambda_handler(self._event({
                "job_id": "arn:exec:mp",
                "workflow_id": "wf1",
                "status": "FAILED",
                "error": "transcode blew up",
                "output_s3_prefix": "s3://proxy/run/wf1"}), None)

    def test_non_handle_shape_raises(self):
        # the legacy nested-execution shape was retired 2026-09-20;
        # anything that is not a gateway handle is a pipeline bug
        with self.assertRaisesRegex(ValueError, "not a gateway handle"):
            extract_mod.lambda_handler(self._event({
                "audio_bucket": "proxy",
                "audio_key": "pre/media_processing/audiostem_sep/a.wav",
                "media_processing_arn": "arn:exec:legacy"}), None)


class TestAslParity(unittest.TestCase):
    """The inline DefinitionString in cfnstack.yaml is what deploys;
    definitions.json is the shipped reference copy. Whole-document
    equality is the drift gate — both carry the ${...} Fn::Sub tokens
    and $$ context sigils literally, so they parse to identical JSON."""

    @staticmethod
    def _inline_asl() -> dict:
        lines = open(os.path.join(ROOT, "cfnstack.yaml")).read().split("\n")
        i = next(n for n, l in enumerate(lines)
                 if l.strip() == "DefinitionString:")
        assert lines[i + 1].strip() == "!Sub |-"
        sub_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
        body = []
        for line in lines[i + 2:]:
            if line.strip() and (len(line) - len(line.lstrip())) <= sub_indent:
                break
            body.append(line)
        indent = min(len(l) - len(l.lstrip()) for l in body if l.strip())
        return json.loads(
            "\n".join(l[indent:] if l.strip() else "" for l in body))

    def test_inline_asl_matches_definitions_json(self):
        defs = json.load(open(os.path.join(ROOT, "definitions.json")))
        self.assertEqual(self._inline_asl(), defs)

    def test_run_media_processing_is_a_held_gateway_submit(self):
        state = json.load(open(os.path.join(ROOT, "definitions.json")))[
            "States"]["Run media processing"]
        self.assertEqual(state["Resource"],
                         "arn:aws:states:::lambda:invoke.waitForTaskToken")
        payload = state["Parameters"]["Payload"]
        self.assertEqual(payload["operation"], "StateSubmitMediaProcessing")
        self.assertEqual(payload["task_token.$"], "$$.Task.Token")
        self.assertEqual(payload["mp_gateway_arn"],
                         "${MediaProcessingGatewayLambdaArn}")
        self.assertEqual(state["ResultPath"], "$.data.media_processing_output")


class TestGatewayInputExamples(unittest.TestCase):
    """input_examples.json — the input twin of the output contract,
    served by describe_tool (replicates the cast-resolution pilot)."""

    @staticmethod
    def _ctx(tool):
        return types.SimpleNamespace(client_context=types.SimpleNamespace(
            custom={"bedrockAgentCoreToolName": f"audio-stem___{tool}"}))

    @staticmethod
    def _examples():
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            return json.load(f)["tools"]

    def test_examples_cover_exactly_the_dispatched_tools(self):
        examples = self._examples()
        self.assertEqual(sorted(examples),
                         ["describe_tool", "get_job_status",
                          "separate_audio_stems"])
        for tool, entry in examples.items():
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        out = gateway_mod.lambda_handler({}, self._ctx("describe_tool"))
        self.assertIn("separate_audio_stems",
                      out["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        example = self._examples()["separate_audio_stems"]["example"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gateway_mod, "_sfn", fake):
            out = gateway_mod.lambda_handler(
                dict(example), self._ctx("separate_audio_stems"))
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()


if __name__ == "__main__":
    unittest.main()
