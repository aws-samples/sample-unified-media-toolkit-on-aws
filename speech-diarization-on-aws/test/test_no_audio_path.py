"""Named content-fact failures (2026-10-15, design settled WITH the user:
raise the correct exception and let the execution fail — the agent probes
first and handles the named condition; an empty-transcript success was
built, challenged, and reverted).

Pins: extract raises NoAudioStream (stable class-name code) with
media-processing's passed-through message, across BOTH completion shapes
and the legacy ValueError shape; other failures pass the inner condition
through instead of 'see the run prefix'; clustering's empty-starved raise
is the named NoSpeechDetected carrying the evidence counts; the ASL keeps
its straight Extract -> Fix chunking edge (no Choice, no emit state). The
MP-side halves (NoCompatibleOutputGroups, waker error/cause forwarding,
unified absorbed shape) are pinned in media-processing's own suite."""
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
    "DIARIZATION_STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:000000000000:stateMachine:test-diarization")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("TARGET_NAME", "speech-diarization")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT goes on the path because the state module imports `lib.*` itself,
# but the TEST never imports the generic `lib`/`states` names: a
# monorepo-wide session may already hold a sibling workflow's `lib` in
# sys.modules, so every module this file needs is loaded by file path.
sys.path.insert(0, ROOT)
# the gateway module imports its flat-zip sibling (workflow_identity), so
# its directory must be importable BEFORE any _load of it — at module level,
# not inside one test method, or isolation runs of a later method break
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


extract = _load("sd_extract", "states", "extract_processed_audio",
                "lambda_function.py")
# the class the state actually RAISES — a second file-path load of
# lib/exceptions.py would be a different class object and assertRaises
# would miss it (the classic double-import identity trap)
NoAudioStream = extract.NoAudioStream
# only an issubclass assertion needs this one, so identity is moot
NoSpeechDetected = _load("sd_exceptions", "lib",
                         "exceptions.py").NoSpeechDetected

NO_AUDIO_CAUSE = json.dumps({
    "errorMessage": "No output_groups are compatible with source (no audio)",
    "errorType": "NoCompatibleOutputGroups"})


def _event(mp_out):
    return {"uuid": "u1", "input": {}, "output": {},
            "data": {"output_bucket": "b", "output_prefix": "p",
                     "media_processing_output": mp_out}}


class TestExtractRaisesTheNamedCondition(unittest.TestCase):
    def test_no_audio_failure_raises_no_audio_stream(self):
        with self.assertRaises(NoAudioStream) as raised:
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-1",
                 "job_id": "arn:mp", "output_s3_prefix": "s3://b/x",
                 "error": "NoCompatibleOutputGroups",
                 "cause": NO_AUDIO_CAUSE}), None)
        msg = str(raised.exception)
        self.assertIn("no audio stream", msg)
        self.assertIn("no audio", msg)              # MP's message rides along
        # the class NAME is the stable code the execution error carries
        self.assertEqual(type(raised.exception).__name__, "NoAudioStream")

    def test_absorbed_completion_shape_raises_it_too(self):
        # MP's synchronous absorbed completion may carry the cause JSON in
        # `error` with no `cause` key
        with self.assertRaises(NoAudioStream):
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-2",
                 "output_s3_prefix": "s3://b/x",
                 "error": NO_AUDIO_CAUSE}), None)

    def test_bare_class_name_with_no_message_raises_it(self):
        # this consumer's MP request is ALWAYS audio-only, so the class
        # alone is decisive even when no message survived
        with self.assertRaises(NoAudioStream):
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-3",
                 "output_s3_prefix": "s3://b/x",
                 "error": "NoCompatibleOutputGroups"}), None)

    def test_type_only_compact_cause_raises_it(self):
        # the waker's compact cause keeps whichever of errorType /
        # errorMessage the execution carried — an exception raised with
        # empty args arrives type-only and must still be decisive
        with self.assertRaises(NoAudioStream):
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-3b",
                 "output_s3_prefix": "s3://b/x",
                 "cause": json.dumps(
                     {"errorType": "NoCompatibleOutputGroups"})}), None)

    def test_type_only_cause_never_reembeds_the_raw_blob(self):
        # a parsed dict with no errorMessage already gave everything it
        # had in the code — the raw JSON must not ride along as "detail"
        with self.assertRaises(RuntimeError) as raised:
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-3c",
                 "output_s3_prefix": "s3://b/x",
                 "cause": json.dumps({"errorType": "OSError"})}), None)
        msg = str(raised.exception)
        self.assertIn("OSError", msg)
        self.assertNotIn("errorType", msg)       # no raw JSON in the message
        self.assertNotIn("{", msg)

    def test_legacy_valueerror_shape_is_recognized_by_wording(self):
        # a TRUE legacy cause carries errorType ValueError — using the
        # modern NO_AUDIO_CAUSE here would silently ride the class-name
        # branch and never exercise the wording guard (review catch)
        legacy = json.dumps({"errorMessage": "No output_groups are "
                             "compatible with source (no audio)",
                             "errorType": "ValueError"})
        with self.assertRaises(NoAudioStream):
            extract.lambda_handler(_event(
                {"status": "FAILED", "workflow_id": "wf-4",
                 "output_s3_prefix": "s3://b/x",
                 "error": "ValueError", "cause": legacy}), None)

    def test_legacy_no_video_wording_stays_a_plain_failure(self):
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


class TestClusteringNamesTheEmptyStarvation(unittest.TestCase):
    def test_the_raise_is_no_speech_detected_with_evidence(self):
        """Measured 2026-10-15: a 2 s music-only WAV died here as an opaque
        'No high-quality embeddings found'. Pin the named replacement at
        the source (running the state needs model artifacts)."""
        with open(os.path.join(ROOT, "states", "global_speakers_clustering",
                               "lambda_function.py")) as f:
            src = f.read()
        self.assertNotIn('RuntimeError("No high-quality embeddings', src)
        self.assertIn("raise NoSpeechDetected(", src)
        self.assertIn("skipped={len(all_skipped)}", src)   # evidence counts
        self.assertTrue(issubclass(NoSpeechDetected, RuntimeError))


class TestGatewaySurfacesTheCode(unittest.TestCase):
    def test_failed_poll_carries_error_code_and_compact_error(self):
        """The prompt tells agents to branch on the named codes — the poll
        must surface errorType as error_code and a compact
        'Code: message' error, never a raw stack-trace blob."""
        gw = _load("sd_gateway_na", "gateway_tool", "lambda_function.py")
        cause = json.dumps({
            "errorMessage": "the source has no audio stream — nothing to "
                            "diarize (No output_groups are compatible with "
                            "source (no audio))",
            "errorType": "NoAudioStream",
            "stackTrace": ["x" * 400] * 8})
        code, message = gw._split_cause(cause)
        self.assertEqual(code, "NoAudioStream")
        self.assertIn("no audio stream", message)
        self.assertLessEqual(len(message), 400)
        self.assertEqual(gw._split_cause("plain text"), ("", ""))
        self.assertEqual(gw._split_cause(None), ("", ""))

    def test_unparseable_cause_falls_back_to_sfn_error_field(self):
        """When the cause is not Lambda-error JSON (Lambda.Unknown's
        plain-text cause, States.Timeout with none), DescribeExecution's
        own error field is still a stable code — the poll must carry it
        as error_code rather than omitting the field."""
        gw = _load("sd_gateway_fb", "gateway_tool", "lambda_function.py")
        arn = ("arn:aws:states:eu-west-1:000000000000:execution:"
               "test-diarization:wf-fb")

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


class TestAslKeepsTheStraightEdge(unittest.TestCase):
    def test_no_choice_no_emit_state(self):
        doc = json.load(open(os.path.join(ROOT, "definitions.json")))
        states = doc["States"]
        self.assertEqual(states["Extract processed audio"]["Next"],
                         "Fix chunking for voice activity detection")
        self.assertNotIn("No audio stream?", states)
        self.assertNotIn("Emit empty transcript", states)
        # cfnstack parity is TestAslParity's job (whole-document equality)


if __name__ == "__main__":
    unittest.main()
