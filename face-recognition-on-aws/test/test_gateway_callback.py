# test/test_gateway_callback.py
"""Hermetic tests for the gateway+callback conversion (2026-09-18):
the family sibling pattern replacing the direct startExecution.sync:2
of media-processing's state machine.

Covers lib/gateway.py, StateSubmitMediaProcessing (the held state's
handler), StateExtractProcessedFrames (handle normalization to the
legacy field shape every downstream state reads), and the
definitions.json wiring (no inline-ASL parity gate here — this stack's
build script injects definitions.json via %%STATE_MACHINE_DEFINITION%%,
so there is no duplicate to drift).

No AWS: boto3 is stubbed while the handler modules load (they build
clients at import time), and S3 reads are monkey-patched out.

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_module(rel_path: str, name: str):
    """Load a handler by file path with boto3 stubbed for the duration
    of the exec — the modules build clients at import time. Previous
    sys.modules entries are restored in the finally so a leaked stub
    cannot disarm other tests in the run."""
    saved = {k: sys.modules.get(k) for k in ("boto3",)}
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *a, **k: object()
    sys.modules["boto3"] = boto3
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
    "states/submit_media_processing/lambda_function.py", "fr_submit_mp")
extract_mod = _load_module(
    "states/extract_processed_frames/lambda_function.py", "fr_extract_mp")



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
        ctx = json.loads(base64.b64decode(lam.calls[0]["ClientContext"]))
        self.assertEqual(
            ctx, {"custom": {"bedrockAgentCoreToolName":
                             "media-processing___process_media"}})

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
    submit with this run's uuid as the idempotency token (which retires
    the minted facerec-{hex} child uuid). MP is a HARD dependency, so
    every failure path raises rather than self-skips."""

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
        calls = self._patch(submit={"job_id": "arn:exec",
                                    "status": "RUNNING"})
        out = submit_mod.lambda_handler(dict(self.EVENT), None)
        self.assertEqual(out, {"parked": True, "job_id": "arn:exec"})
        kind, target, tool, payload = calls[-1]
        self.assertEqual((kind, target, tool),
                         ("invoke", "media-processing", "process_media"))
        self.assertEqual(payload["idempotency_token"], "facerec-abc-123")
        self.assertEqual(payload["callback_task_token"], "TOKEN")

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


class TestExtractProcessedFrames(unittest.TestCase):
    """Handle normalization: the gateway handle (status AS DATA — a
    FAILED verdict is enforced HERE) becomes the field shape every
    downstream state reads; non-handle shapes raise (legacy retired
    2026-09-20)."""

    def setUp(self):
        self._orig = extract_mod.get_json
        self._reads = []

        def fake_get_json(bucket, key):
            self._reads.append((bucket, key))
            return {"duration_sec": 719.0}

        extract_mod.get_json = fake_get_json
        self.addCleanup(
            lambda: setattr(extract_mod, "get_json", self._orig))

    def _event(self, mp_out):
        return {"uuid": "u", "data": {
            "media_processing_request": {"asset_s3uri": "s3://in/a.mxf"},
            "media_processing_output": mp_out}}

    def test_gateway_handle_normalizes_to_the_legacy_shape(self):
        out = extract_mod.lambda_handler(self._event({
            "job_id": "arn:exec:mp",
            "workflow_id": "wf1",
            "status": "SUCCEEDED",
            "output_s3_prefix":
                "s3://proxy/micro-workflow/media-processing/wf1"}), None)
        mp = out["data"]["media_processing_output"]
        self.assertEqual(mp, {
            "frames_bucket": "proxy",
            "manifest_key":
                "micro-workflow/media-processing/wf1/"
                "face_frames/frame_manifest.json",
            "probe_key": "micro-workflow/media-processing/wf1/probe.json",
            "duration_sec": 719.0,
            "media_processing_arn": "arn:exec:mp",
        })
        # duration came from MP's published probe.json
        self.assertEqual(self._reads, [
            ("proxy", "micro-workflow/media-processing/wf1/probe.json")])
        self.assertIsNone(out["data"]["media_processing_request"])

    def test_gateway_handle_failure_status_raises(self):
        with self.assertRaisesRegex(RuntimeError, "FAILED"):
            extract_mod.lambda_handler(self._event({
                "job_id": "arn:exec:mp",
                "workflow_id": "wf1",
                "status": "FAILED",
                "error": "frame extraction blew up",
                "output_s3_prefix": "s3://proxy/run/wf1"}), None)

    def test_non_handle_shape_raises(self):
        # the legacy nested-execution shape was retired 2026-09-20;
        # anything that is not a gateway handle is a pipeline bug
        with self.assertRaisesRegex(ValueError, "not a gateway handle"):
            extract_mod.lambda_handler(self._event({
                "frames_bucket": "proxy",
                "manifest_key": "pre/media_processing/face_frames/frame_manifest.json",
                "media_processing_arn": "arn:exec:legacy"}), None)
        self.assertEqual(self._reads, [])


class TestDefinitionsWiring(unittest.TestCase):
    """definitions.json is the single ASL source (the build script
    injects it into cfnstack.yaml via %%STATE_MACHINE_DEFINITION%%)."""

    @classmethod
    def setUpClass(cls):
        cls.defs = json.load(open(os.path.join(ROOT, "definitions.json")))

    def test_run_media_processing_is_a_held_gateway_submit(self):
        state = self.defs["States"]["Run media processing"]
        self.assertEqual(state["Resource"],
                         "arn:aws:states:::lambda:invoke.waitForTaskToken")
        payload = state["Parameters"]["Payload"]
        self.assertEqual(payload["operation"], "StateSubmitMediaProcessing")
        self.assertEqual(payload["task_token.$"], "$$.Task.Token")
        self.assertEqual(payload["mp_gateway_arn"],
                         "${MediaProcessingGatewayLambdaArn}")
        self.assertEqual(state["ResultPath"],
                         "$.data.media_processing_output")
        self.assertEqual(state["Next"], "Extract processed frames")

    def test_extract_state_normalizes_before_the_first_reader(self):
        state = self.defs["States"]["Extract processed frames"]
        self.assertEqual(
            state["Parameters"]["Payload"]["operation"],
            "StateExtractProcessedFrames")
        self.assertEqual(state["Next"], "Prepare probe iterators")

    def test_the_template_still_injects_definitions(self):
        text = open(os.path.join(ROOT, "cfnstack.yaml")).read()
        self.assertIn("%%STATE_MACHINE_DEFINITION%%", text)


if __name__ == "__main__":
    unittest.main()
