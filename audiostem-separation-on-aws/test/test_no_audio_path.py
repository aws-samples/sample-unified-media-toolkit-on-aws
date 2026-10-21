"""Named content-fact failure (2026-10-15, the speech-diarization
fail-by-name decision applied family-wide): a source MP refuses by name
(NoCompatibleOutputGroups) raises NoAudioStream here instead of an opaque
"see the run prefix" RuntimeError; every other MP failure passes the
inner condition through. Pins the verdict normalization across BOTH
completion shapes plus the legacy ValueError wording, and the gateway's
error_code surfacing. The MP-side halves are pinned in media-processing's
own suite."""
import json
import os
import sys
import unittest

import importlib.util

# lib.s3_utils builds a boto3 client at import, and the gateway module
# reads its state-machine ARN at import — a region-less/env-less machine
# must not fail the collection of a suite that never touches AWS
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault(
    "AUDIOSTEM_STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:000000000000:stateMachine:test-audio-stem")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("TARGET_NAME", "audio-stem")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT goes on the path because the state module imports `lib.*` itself,
# but the TEST never imports the generic `lib`/`states` names: a
# monorepo-wide session may already hold a sibling workflow's `lib` in
# sys.modules, so every module this file needs is loaded by file path.
sys.path.insert(0, ROOT)
# the gateway module imports its flat-zip siblings (workflow_identity),
# so its directory must be importable before any _load of it
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))


# the state module's own `from lib...` imports resolve through
# sys.modules — a sibling repo's identically-named `lib` cached by an
# earlier-collected suite would hijack them (ImportError on a class the
# foreign copy lacks, or silently binding the wrong repo's class), so
# evict any cached copy and let this ROOT, now first on sys.path, win
for _mod in [m for m in list(sys.modules)
             if m == "lib" or m.startswith("lib.")]:
    del sys.modules[_mod]


def _load(name, *parts):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, *parts))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


extract = _load("as_extract", "states", "extract_processed_audio",
                "lambda_function.py")
# the class the state actually RAISES — a second file-path load of
# lib/exceptions.py would be a different class object and assertRaises
# would miss it (the classic double-import identity trap)
NoAudioStream = extract.NoAudioStream

REFUSAL_CAUSE = json.dumps({
    "errorMessage": "No output_groups are compatible with source (no audio)",
    "errorType": "NoCompatibleOutputGroups"})


def _event(mp_out):
    return {"uuid": "u1", "input": {}, "output": {},
            "data": {"output_bucket": "b", "output_prefix": "p",
                     "media_processing_output": mp_out}}


class TestExtractRaisesTheNamedCondition(unittest.TestCase):
    def test_refusal_raises_the_named_exception(self):
        with self.assertRaises(NoAudioStream) as raised:
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-1",
                 "job_id": "arn:mp", "output_s3_prefix": "s3://b/x",
                 "error": "NoCompatibleOutputGroups",
                 "cause": REFUSAL_CAUSE}), None)
        msg = str(raised.exception)
        self.assertIn("no audio stream", msg)
        self.assertIn("no audio", msg)        # MP's message rides along
        # the class NAME is the stable code the execution error carries
        self.assertEqual(type(raised.exception).__name__, "NoAudioStream")

    def test_absorbed_completion_shape_raises_it_too(self):
        # MP's synchronous absorbed completion may carry the cause JSON in
        # `error` with no `cause` key
        with self.assertRaises(NoAudioStream):
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-2",
                 "output_s3_prefix": "s3://b/x",
                 "error": REFUSAL_CAUSE}), None)

    def test_bare_class_name_with_no_message_raises_it(self):
        # this consumer's MP request is single-purpose, so the class
        # alone is decisive even when no message survived
        with self.assertRaises(NoAudioStream):
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-3",
                 "output_s3_prefix": "s3://b/x",
                 "error": "NoCompatibleOutputGroups"}), None)

    def test_type_only_compact_cause_raises_it(self):
        # an exception raised with empty args arrives type-only in the
        # waker's compact cause and must still be decisive
        with self.assertRaises(NoAudioStream):
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-3b",
                 "output_s3_prefix": "s3://b/x",
                 "cause": json.dumps(
                     {"errorType": "NoCompatibleOutputGroups"})}), None)

    def test_legacy_valueerror_shape_is_recognized_by_wording(self):
        legacy = json.dumps({"errorMessage": "No output_groups are "
                             "compatible with source (no audio)",
                             "errorType": "ValueError"})
        with self.assertRaises(NoAudioStream):
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-4",
                 "output_s3_prefix": "s3://b/x",
                 "error": "ValueError", "cause": legacy}), None)

    def test_legacy_other_stream_wording_stays_a_plain_failure(self):
        with self.assertRaises(RuntimeError) as raised:
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-5",
                 "output_s3_prefix": "s3://b/x",
                 "error": "ValueError",
                 "cause": json.dumps({"errorMessage": "No output_groups are "
                                      "compatible with source (no video)"})}),
                None)
        self.assertNotIsInstance(raised.exception, NoAudioStream)

    def test_other_failures_pass_the_inner_condition_through(self):
        with self.assertRaises(RuntimeError) as raised:
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-6",
                 "output_s3_prefix": "s3://b/x",
                 "error": "States.Timeout",
                 "cause": json.dumps({"errorMessage": "probe timed out"})}),
                None)
        msg = str(raised.exception)
        self.assertNotIsInstance(raised.exception, NoAudioStream)
        self.assertIn("States.Timeout", msg)
        self.assertIn("probe timed out", msg)        # passthrough, not
        self.assertNotIn("see the run prefix", msg)  # the old opaque line

    def test_a_detail_less_failure_names_the_gap_honestly(self):
        with self.assertRaises(RuntimeError) as raised:
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-7",
                 "output_s3_prefix": "s3://b/x"}), None)
        self.assertIn("carried no error detail", str(raised.exception))


class TestGatewaySurfacesTheCode(unittest.TestCase):
    def test_failed_poll_carries_error_code_and_compact_error(self):
        """The schema tells agents to branch on error_code — the poll
        must surface errorType as error_code and a compact
        'Code: message' error, never a raw stack-trace blob."""
        gw = _load("as_gateway_sc", "gateway_tool", "lambda_function.py")
        cause = json.dumps({
            "errorMessage": "the source has no audio stream — nothing to separate (No output_groups are compatible with source (no audio))",
            "errorType": "NoAudioStream",
            "stackTrace": ["x" * 400] * 8})
        code, message = gw._split_cause(cause)
        self.assertEqual(code, "NoAudioStream")
        self.assertIn("no audio stream", message)
        self.assertLessEqual(len(message), 400)
        self.assertEqual(gw._split_cause("plain text"), ("", ""))
        self.assertEqual(gw._split_cause(None), ("", ""))

    def test_unparseable_cause_falls_back_to_sfn_error_field(self):
        """When the cause is not Lambda-error JSON, DescribeExecution's
        own error field is still a stable code — the poll must carry it
        as error_code rather than omitting the field."""
        gw = _load("as_gateway_fb", "gateway_tool", "lambda_function.py")
        arn = ("arn:aws:states:eu-west-1:000000000000:execution:"
               "test-audio-stem:wf-fb")

        class _StubSfn:
            class exceptions:
                class ExecutionDoesNotExist(Exception):
                    pass

            @staticmethod
            def describe_execution(executionArn):
                return {"executionArn": executionArn, "name": "wf-fb",
                        "status": "FAILED", "error": "Lambda.Unknown",
                        "cause": "The Lambda function returned an "
                                 "unexpected response",
                        "input": json.dumps({"input": {
                            "asset_s3uri": "s3://b/a.mp4",
                            "output_s3_prefix": "s3://b/p/"}})}

        gw._sfn = _StubSfn()
        result = gw._get_job_status({"job_id": arn})
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "Lambda.Unknown")
        self.assertIn("unexpected response", result["error"])


if __name__ == "__main__":
    unittest.main()
