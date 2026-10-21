"""callback_task_token on the media-processing gateway (2026-09-18,
the family consumer pattern ported from bedrock-inference): validated
loudly, registered BEFORE StartExecution, completed synchronously on
absorbed-finished runs, and the dispatcher's raise-vs-return choke
point. Plus the waker's terminal-event completion."""

import json
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))

os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault(
    "MEDIA_PROCESSING_STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:111122223333:stateMachine:mp")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("TARGET_NAME", "media-processing")
os.environ.setdefault("CALLBACK_TABLE", "test-callbacks")

import callback_waker                                       # noqa: E402
import lambda_function as gw                                # noqa: E402

EVENT = {"asset_s3uri": "s3://in/a.mp4",
         "callback_task_token": "tok-123"}  # nosec B105


def _ctx(tool):
    return types.SimpleNamespace(client_context=types.SimpleNamespace(
        custom={"bedrockAgentCoreToolName": f"media-processing___{tool}"}))



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


class TestCallbackSubmit(unittest.TestCase):
    def test_bad_token_raises(self):
        for bad in ("", "   ", 7, "x" * 3000):
            with self.assertRaises(Exception):
                gw._process_media({"asset_s3uri": "s3://in/a.mp4",
                                   "callback_task_token": bad})

    def test_token_without_table_raises(self):
        with mock.patch.object(gw, "CALLBACK_TABLE", ""):
            with self.assertRaises(RuntimeError):
                gw._process_media(dict(EVENT))

    def test_register_before_start_and_handle(self):
        calls = []
        with mock.patch.object(gw._ddb, "put_item",
                               side_effect=lambda **kw: calls.append(("put", kw))), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=lambda **kw: calls.append(("start", kw))
                               or {"executionArn": "arn:x"}):
            r = gw._process_media(dict(EVENT))
        self.assertEqual([c[0] for c in calls], ["put", "start"],
                         "the row must exist before the run can finish")
        item = calls[0][1]["Item"]
        self.assertIn(":execution:", item["execution_arn"]["S"])
        handle = json.loads(item["handle"]["S"])
        self.assertEqual(handle["output_s3_prefix"],
                         r["output_s3_prefix"])
        self.assertEqual(r["status"], "RUNNING")

    def test_absorbed_finished_completes_now(self):
        from botocore.exceptions import ClientError
        sent, deleted = [], []
        already = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        with mock.patch.object(gw._ddb, "put_item"), \
             mock.patch.object(gw._ddb, "delete_item",
                               side_effect=lambda **kw: deleted.append(kw)), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=already), \
             mock.patch.object(gw._sfn, "describe_execution",
                               return_value={"status": "SUCCEEDED"}), \
             mock.patch.object(gw._sfn, "send_task_success",
                               side_effect=lambda **kw: sent.append(kw)), \
             mock.patch.object(gw, "_extract_asset_s3uri",
                               return_value=None), \
             mock.patch.object(gw, "_extract_output_s3_prefix",
                               return_value=None):
            r = gw._process_media(dict(EVENT))
        self.assertEqual(len(sent), 1, "finished absorb must wake NOW")
        self.assertEqual(json.loads(sent[0]["output"])["status"],
                         "SUCCEEDED")
        self.assertEqual(len(deleted), 1)
        self.assertIn("message", r)

    def test_absorbed_failed_token_payload_uses_the_waker_shape(self):
        """The synchronous completion must emit the ONE shape the waker
        emits — error = class name, cause = compact parse-safe JSON —
        so no consumer ever shape-sniffs two completion formats; the
        agent-facing result keeps its documented error-is-the-cause."""
        from botocore.exceptions import ClientError
        sent = []
        already = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        cause = json.dumps({
            "errorMessage": "No output_groups are compatible with source "
                            "(no audio)",
            "errorType": "NoCompatibleOutputGroups",
            "stackTrace": ["x" * 300] * 5})
        with mock.patch.object(gw._ddb, "put_item"), \
             mock.patch.object(gw._ddb, "delete_item"), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=already), \
             mock.patch.object(gw._sfn, "describe_execution",
                               return_value={"status": "FAILED",
                                             "error":
                                             "NoCompatibleOutputGroups",
                                             "cause": cause}), \
             mock.patch.object(gw._sfn, "send_task_success",
                               side_effect=lambda **kw: sent.append(kw)), \
             mock.patch.object(gw, "_extract_asset_s3uri",
                               return_value=None), \
             mock.patch.object(gw, "_extract_output_s3_prefix",
                               return_value=None):
            result = gw._process_media(dict(EVENT))
        payload = json.loads(sent[0]["output"])
        self.assertEqual(payload["error"], "NoCompatibleOutputGroups")
        inner = json.loads(payload["cause"])       # parse-safe, compact
        self.assertIn("no audio", inner["errorMessage"])
        self.assertLessEqual(len(payload["cause"]), 600)
        # the agent-facing response keeps the documented cause-as-error
        self.assertIn("no audio", result["error"])

    def test_pending_redrive_absorb_does_not_complete_the_token(self):
        """PENDING_REDRIVE is not terminal: completing now would hand the
        consumer the OLD failure while the redriven run may yet succeed —
        the redrive's own terminal event reaches the waker."""
        from botocore.exceptions import ClientError
        sent = []
        already = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        with mock.patch.object(gw._ddb, "put_item"), \
             mock.patch.object(gw._ddb, "delete_item"), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=already), \
             mock.patch.object(gw._sfn, "describe_execution",
                               return_value={"status": "PENDING_REDRIVE",
                                             "error": "ValueError",
                                             "cause": "old failure"}), \
             mock.patch.object(gw._sfn, "send_task_success",
                               side_effect=lambda **kw: sent.append(kw)), \
             mock.patch.object(gw, "_extract_asset_s3uri",
                               return_value=None), \
             mock.patch.object(gw, "_extract_output_s3_prefix",
                               return_value=None):
            gw._process_media(dict(EVENT))
        self.assertEqual(sent, [])     # the waker owns this completion

    def test_choke_point_raises_refusals_with_token(self):
        # a refusal (missing asset uri) RETURNS an error dict without a
        # token, and RAISES with one — a waiting caller can't read a return
        r = gw.lambda_handler({"output_groups": []}, _ctx("process_media"))
        self.assertTrue(r.get("error"))
        with self.assertRaises(RuntimeError):
            gw.lambda_handler({"output_groups": [],
                               "callback_task_token": "tok-1"},  # nosec B105
                              _ctx("process_media"))


class TestInputExamples(unittest.TestCase):
    """input_examples.json — the input twin of the output contract,
    served by describe_tool (replicates the cast-resolution pilot)."""

    @staticmethod
    def _examples():
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            return json.load(f)["tools"]

    def test_examples_cover_exactly_the_dispatched_tools(self):
        examples = self._examples()
        self.assertEqual(sorted(examples),
                         ["capture_frames", "describe_tool",
                          "get_job_status", "process_media",
                          "quick_edit"])
        for tool, entry in examples.items():
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        out = gw.lambda_handler({}, _ctx("describe_tool"))
        self.assertIn("process_media", out["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        example = self._examples()["process_media"]["example"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gw, "_sfn", fake):
            out = gw.lambda_handler(dict(example), _ctx("process_media"))
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()


class TestWaker(unittest.TestCase):
    def _event(self, status="SUCCEEDED"):
        return {"detail": {"executionArn": "arn:e:1", "status": status}}

    def test_completes_registered_tokens_with_status_as_data(self):
        rows = [{"execution_arn": {"S": "arn:e:1"},
                 "token_hash": {"S": "h1"}, "token": {"S": "tok-1"},
                 "handle": {"S": json.dumps({"job_id": "arn:e:1"})}}]
        sent, deleted = [], []
        with mock.patch.object(callback_waker, "CALLBACK_TABLE", "t"), \
             mock.patch.object(callback_waker._ddb, "query",
                               return_value={"Items": rows}), \
             mock.patch.object(callback_waker._ddb, "delete_item",
                               side_effect=lambda **kw: deleted.append(kw)), \
             mock.patch.object(callback_waker._sfn, "send_task_success",
                               side_effect=lambda **kw: sent.append(kw)):
            out = callback_waker.lambda_handler(self._event("FAILED"), None)
        self.assertEqual(out["completed"], 1)
        payload = json.loads(sent[0]["output"])
        self.assertEqual(payload["status"], "FAILED")   # data, not a raise
        self.assertEqual(len(deleted), 1)

    def test_no_rows_is_a_quiet_miss(self):
        with mock.patch.object(callback_waker, "CALLBACK_TABLE", "t"), \
             mock.patch.object(callback_waker._ddb, "query",
                               return_value={"Items": []}):
            out = callback_waker.lambda_handler(self._event(), None)
        self.assertEqual(out["completed"], 0)

    def test_stuck_completion_raises_for_redelivery(self):
        from botocore.exceptions import ClientError
        rows = [{"execution_arn": {"S": "arn:e:1"},
                 "token_hash": {"S": "h1"}, "token": {"S": "tok-1"},
                 "handle": {"S": "{}"}}]
        denied = ClientError({"Error": {"Code": "AccessDenied",
                                        "Message": "no"}},
                             "SendTaskSuccess")
        with mock.patch.object(callback_waker, "CALLBACK_TABLE", "t"), \
             mock.patch.object(callback_waker._ddb, "query",
                               return_value={"Items": rows}), \
             mock.patch.object(callback_waker._sfn, "send_task_success",
                               side_effect=denied):
            with self.assertRaises(RuntimeError):
                callback_waker.lambda_handler(self._event(), None)

class TestNoAudioStableCode(unittest.TestCase):
    """The no-audio refusal's class name IS the stable code consumers
    branch on (speech-diarization's empty-transcript path keys on the
    execution error field being 'NoCompatibleOutputGroups')."""

    def test_no_compatible_output_groups_is_a_named_valueerror(self):
        # import via explicit path — inserting ROOT and importing the
        # generic package name `lib` collides with sibling workflows'
        # `lib` in a monorepo-wide test session
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mp_input_schema", os.path.join(ROOT, "lib", "input_schema.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        NoCompatibleOutputGroups = mod.NoCompatibleOutputGroups
        parse_output_groups = mod.parse_output_groups
        with self.assertRaises(NoCompatibleOutputGroups) as raised:
            parse_output_groups([{"group_name": "a", "format": "wav"}],
                                has_video=True, has_audio=False)
        self.assertIsInstance(raised.exception, ValueError)
        self.assertEqual(type(raised.exception).__name__,
                         "NoCompatibleOutputGroups")
        self.assertIn("no audio", str(raised.exception))


class TestWakerForwardsTheCondition(unittest.TestCase):
    """A FAILED completion must carry the execution's error/cause — the
    consumer otherwise reports 'see the run prefix' over an empty prefix
    (the measured speech-diarization incident, 2026-10-15)."""

    def _run(self, status, error=None, cause=None):
        import importlib
        sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))
        import callback_waker
        importlib.reload(callback_waker)
        sent = []
        row = {"token": {"S": "tok-1"},
               "execution_arn": {"S": "arn:exec"},
               "token_hash": {"S": "th-1"},
               "handle": {"S": json.dumps({"job_id": "arn:x",
                                           "workflow_id": "wf"})}}
        detail = {"executionArn": "arn:exec", "status": status}
        if error:
            detail["error"] = error
        if cause:
            detail["cause"] = cause
        with mock.patch.object(callback_waker, "_query_all",
                               lambda arn: [row]), \
                mock.patch.object(callback_waker, "_ddb", mock.MagicMock()), \
                mock.patch.object(callback_waker, "_sfn", mock.MagicMock(
                    send_task_success=lambda taskToken, output: sent.append(
                        json.loads(output)))):
            callback_waker.lambda_handler({"detail": detail}, None)
        return sent[0]

    def test_failed_forwards_error_and_a_parse_safe_cause(self):
        huge = json.dumps({
            "errorMessage": "No output_groups are compatible with source "
                            "(no audio)",
            "errorType": "NoCompatibleOutputGroups",
            "stackTrace": ["x" * 400] * 10})       # real causes exceed 600
        payload = self._run("FAILED", error="NoCompatibleOutputGroups",
                            cause=huge)
        self.assertEqual(payload["status"], "FAILED")
        self.assertEqual(payload["error"], "NoCompatibleOutputGroups")
        # a blind [:600] made the forwarded cause INVALID JSON — it must
        # stay parseable, carrying the condition without the stack trace
        inner = json.loads(payload["cause"])
        self.assertIn("no audio", inner["errorMessage"])
        self.assertEqual(inner["errorType"], "NoCompatibleOutputGroups")
        self.assertLessEqual(len(payload["cause"]), 600)

    def test_escape_heavy_cause_stays_bounded_after_serialization(self):
        noisy = json.dumps({"errorMessage": ('line "with quotes"\n' * 60),
                            "errorType": "ValueError"})
        payload = self._run("FAILED", error="ValueError", cause=noisy)
        self.assertLessEqual(len(payload["cause"]), 600)
        inner = json.loads(payload["cause"])      # still parse-safe
        self.assertIn("with quotes", inner["errorMessage"])

    def test_pathological_error_type_cannot_breach_the_cause_bound(self):
        noisy = json.dumps({"errorMessage": "x",
                            "errorType": "\u00e9" * 120})
        payload = self._run("FAILED", error="ValueError", cause=noisy)
        self.assertLessEqual(len(payload["cause"]), 600)
        json.loads(payload["cause"])   # still parse-safe

    def test_error_type_only_cause_stays_parse_safe(self):
        noisy = json.dumps({"errorType": "SomeError",
                            "stackTrace": ["x" * 300] * 5})
        payload = self._run("FAILED", error="SomeError", cause=noisy)
        inner = json.loads(payload["cause"])       # parse-safe, not a
        self.assertEqual(inner["errorType"], "SomeError")  # cut fragment
        self.assertLessEqual(len(payload["cause"]), 600)

    def test_a_non_json_cause_is_truncated_raw(self):
        payload = self._run("FAILED", error="States.Timeout",
                            cause="plain text " * 100)
        self.assertFalse(payload["cause"].startswith("{"))
        self.assertLessEqual(len(payload["cause"]), 600)

    def test_succeeded_carries_no_error_fields(self):
        payload = self._run("SUCCEEDED")
        self.assertEqual(payload["status"], "SUCCEEDED")
        self.assertNotIn("error", payload)
        self.assertNotIn("cause", payload)

if __name__ == "__main__":
    unittest.main()
