"""
Offline tests for the quick_edit / capture_frames gateway tools: real
validators, mocked SFN. Plus the pins that keep the surface honest —
every input example passes the validation it documents, and the flat
gateway copies of cut_list/frame_capture stay byte-identical to lib/.
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway_tool"))

os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault("PROXY_BUCKET", "proxy")
os.environ.setdefault("TARGET_NAME", "media-processing")
os.environ.setdefault(
    "MEDIA_PROCESSING_STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:000000000000:stateMachine:mp")

import lambda_function as gw  # noqa: E402  (flat, like the zip)

ROOT = os.path.join(os.path.dirname(__file__), "..")

QE_EVENT = {
    "segments": [
        {"source_s3uri": "s3://bkt/a.mp4", "in_ms": 0, "out_ms": 8000},
        {"source_s3uri": "s3://bkt/b.mp4", "in_ms": 0, "out_ms": 6000,
         "transition_in": {"type": "crossfade", "duration_ms": 1000}},
    ],
    "target": {"resolution": "1920x1080", "fps": "25"},
}


class TestQuickEditTool(unittest.TestCase):

    def submit(self, event, start_side_effect=None):
        with mock.patch.object(gw, "_sfn") as sfn:
            if start_side_effect:
                sfn.start_execution.side_effect = start_side_effect
            else:
                sfn.start_execution.return_value = {
                    "executionArn": "arn:...:execution:mp:x"}
            result = gw._quick_edit(event)
        return result, sfn

    def test_happy_submit_returns_a_handle(self):
        result, sfn = self.submit(dict(QE_EVENT))
        self.assertEqual(result["status"], "RUNNING")
        self.assertIn("workflow_id", result)
        self.assertIn("output_s3_prefix", result)
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["source_count"], 2)
        sent = json.loads(sfn.start_execution.call_args.kwargs["input"])
        self.assertIn("quick_edit", sent["input"])
        self.assertEqual(sent["uuid"], result["workflow_id"])
        # the execution is NAMED by the workflow_id — idempotency
        self.assertEqual(sfn.start_execution.call_args.kwargs["name"],
                         result["workflow_id"])

    def test_validation_refuses_before_any_execution(self):
        bad = dict(QE_EVENT, segments=[
            {"source_s3uri": "ftp://x", "in_ms": 0, "out_ms": 1000}])
        result, sfn = self.submit(bad)
        self.assertTrue(result["error"])
        self.assertEqual(result["errorCode"], "INVALID_PARAMETER")
        sfn.start_execution.assert_not_called()

    def test_absorbed_handle_reads_back_the_existing_runs_prefix(self):
        # the handle locates the artifacts of the run that actually EXISTS:
        # output.s3uri is excluded from the work-hash, so a repointed proxy
        # bucket would otherwise hand back a prefix where nothing lives
        from botocore.exceptions import ClientError
        already = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        existing_input = json.dumps(
            {"output": {"s3uri": "s3://old-proxy/micro-workflow/mp/abc"}})
        with mock.patch.object(gw._sfn, "start_execution",
                               side_effect=already), \
             mock.patch.object(gw._sfn, "describe_execution",
                               return_value={"status": "SUCCEEDED",
                                             "input": existing_input}):
            r = gw._quick_edit(dict(QE_EVENT))
        self.assertEqual(r["output_s3_prefix"],
                         "s3://old-proxy/micro-workflow/mp/abc")
        self.assertIn("message", r)

    def test_callback_token_registers_before_start(self):
        # same contract as process_media: the row exists before the run
        # can possibly finish, and the handle carries the edit summary
        calls = []
        with mock.patch.object(gw, "CALLBACK_TABLE", "cb-table"), \
             mock.patch.object(gw._ddb, "put_item",
                               side_effect=lambda **kw: calls.append(("put", kw))), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=lambda **kw: calls.append(("start", kw))
                               or {"executionArn": "arn:x"}):
            r = gw._quick_edit(dict(QE_EVENT, callback_task_token="tok-qe"))
        self.assertEqual([c[0] for c in calls], ["put", "start"])
        handle = json.loads(calls[0][1]["Item"]["handle"]["S"])
        self.assertEqual(handle["output_s3_prefix"], r["output_s3_prefix"])
        self.assertEqual(handle["segment_count"], 2)
        self.assertEqual(r["status"], "RUNNING")

    def test_callback_token_absorbed_finished_completes_now(self):
        from botocore.exceptions import ClientError
        sent, deleted = [], []
        already = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        with mock.patch.object(gw, "CALLBACK_TABLE", "cb-table"), \
             mock.patch.object(gw._ddb, "put_item"), \
             mock.patch.object(gw._ddb, "delete_item",
                               side_effect=lambda **kw: deleted.append(kw)), \
             mock.patch.object(gw._sfn, "start_execution",
                               side_effect=already), \
             mock.patch.object(gw._sfn, "describe_execution",
                               return_value={"status": "SUCCEEDED"}), \
             mock.patch.object(gw._sfn, "send_task_success",
                               side_effect=lambda **kw: sent.append(kw)):
            r = gw._quick_edit(dict(QE_EVENT, callback_task_token="tok-qe"))
        self.assertEqual(len(sent), 1, "finished absorb must wake NOW")
        self.assertEqual(json.loads(sent[0]["output"])["status"], "SUCCEEDED")
        self.assertEqual(len(deleted), 1)
        self.assertIn("message", r)

    def test_choke_point_raises_validation_refusal_with_token(self):
        # a CutListError refusal RETURNS a named error dict for a polling
        # caller, and RAISES when a callback token is present — a waiting
        # caller cannot read a returned error
        class _Ctx:
            client_context = type("C", (), {"custom": {
                "bedrockAgentCoreToolName": "media-processing___quick_edit"}})()
        bad = {"segments": [], "callback_task_token": "tok-qe"}
        with mock.patch.object(gw, "CALLBACK_TABLE", "cb-table"):
            with self.assertRaises(RuntimeError):
                gw.lambda_handler(bad, _Ctx())
        del bad["callback_task_token"]
        r = gw.lambda_handler(bad, _Ctx())
        self.assertTrue(r.get("error"))

    def test_token_changes_workflow_id(self):
        r1, _ = self.submit(dict(QE_EVENT))
        r2, _ = self.submit(dict(QE_EVENT, idempotency_token="t1"))
        r3, _ = self.submit(dict(QE_EVENT, idempotency_token="t1"))
        self.assertNotEqual(r1["workflow_id"], r2["workflow_id"])
        self.assertEqual(r2["workflow_id"], r3["workflow_id"])


class TestCaptureFramesTool(unittest.TestCase):

    def test_happy_submit(self):
        event = {"markers": [
            {"source_s3uri": "s3://bkt/a.mp4", "at_ms": 1000}]}
        with mock.patch.object(gw, "_sfn") as sfn:
            sfn.start_execution.return_value = {
                "executionArn": "arn:...:execution:mp:y"}
            result = gw._capture_frames(event)
        self.assertEqual(result["marker_count"], 1)
        sent = json.loads(sfn.start_execution.call_args.kwargs["input"])
        self.assertIn("frame_capture", sent["input"])

    def test_refusal_is_named(self):
        result = gw._capture_frames({"markers": [], "format": "jpeg"})
        self.assertEqual(result["errorCode"], "INVALID_PARAMETER")
        self.assertIn("non-empty", result["message"])


class TestInputExamplesPinned(unittest.TestCase):
    """Every documented example passes the validation it documents —
    an example that drifts from the contract fails the build."""

    def setUp(self):
        with open(os.path.join(ROOT, "gateway_tool", "input_examples.json")) as f:
            self.tools = json.load(f)["tools"]

    def iter_examples(self, tool):
        entry = self.tools[tool]
        yield entry["example"]
        yield from (entry.get("alternatives") or {}).values()

    def test_quick_edit_examples_validate(self):
        from lib.cut_list import validate_quick_edit
        for ex in self.iter_examples("quick_edit"):
            validate_quick_edit(ex)   # must not raise

    def test_capture_frames_examples_validate(self):
        from lib.frame_capture import validate_frame_capture
        for ex in self.iter_examples("capture_frames"):
            validate_frame_capture(ex)   # must not raise


class TestFlatCopyParity(unittest.TestCase):
    """The gateway zip is flat; cut_list/frame_capture ship as copies.
    A fix in lib/ must be copied — this failing IS the reminder."""

    def test_copies_byte_identical(self):
        for name in ("cut_list.py", "frame_capture.py"):
            with self.subTest(name=name):
                with open(os.path.join(ROOT, "lib", name), "rb") as a, \
                     open(os.path.join(ROOT, "gateway_tool", name), "rb") as b:
                    self.assertEqual(a.read(), b.read(),
                                     f"{name}: lib/ and gateway_tool/ "
                                     "copies differ — copy the fix")


class TestContractCoversNewLabels(unittest.TestCase):

    def test_contract_entries_exist(self):
        with open(os.path.join(ROOT, "gateway_tool",
                               "output_contract.json")) as f:
            outputs = json.load(f)["outputs"]
        self.assertIn("edit_manifest", outputs)
        self.assertIn("capture_manifest", outputs)
        self.assertEqual(outputs["edit_manifest"]["basename"],
                         "edit_manifest.json")


if __name__ == "__main__":
    unittest.main()
