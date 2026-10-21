# test/test_gateway_examples.py
"""input_examples.json — the input twin of the output contract, served
by describe_tool (replicates the cast-resolution pilot).

Self-contained: this repo has no other gateway test harness, so the env
the dispatcher reads at import time is set here, and the dispatcher is
loaded by file path — a bare `import lambda_function` would collide
with the repo-root operation router. Real boto3 builds its clients at
import; no call is ever made (the round-trip test patches _sfn).

    python3 -m unittest discover -s test -p 'test_*.py'
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault(
    "STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:111122223333:stateMachine:shot-segmentation")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("TARGET_NAME", "shot-segmentation")

# the dispatcher's flat zip resolves `workflow_identity` top-level
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))

_gw_spec = importlib.util.spec_from_file_location(
    "shot_segmentation_gateway",
    os.path.join(ROOT, "gateway_tool", "lambda_function.py"))
gateway_mod = importlib.util.module_from_spec(_gw_spec)
_gw_spec.loader.exec_module(gateway_mod)


# Silence console log HANDLERS for the whole suite: the gateway logs
# every event at INFO and refusal paths at ERROR by design, and those
# lines made build logs read as if the build were failing. Handlers
# only — record flow is untouched, so assertLogs() still sees
# everything.
import logging as _logging


def _silence_console_logging():
    _logging.getLogger().handlers = [_logging.NullHandler()]
    for _name in list(_logging.root.manager.loggerDict):
        _lg = _logging.getLogger(_name)
        if _lg.handlers:
            _lg.handlers = [_logging.NullHandler()]


_silence_console_logging()


class TestGatewayInputExamples(unittest.TestCase):
    """input_examples.json — the input twin of the output contract,
    served by describe_tool (replicates the cast-resolution pilot)."""

    @staticmethod
    def _ctx(tool):
        return types.SimpleNamespace(client_context=types.SimpleNamespace(
            custom={"bedrockAgentCoreToolName":
                    f"shot-segmentation___{tool}"}))

    @staticmethod
    def _examples():
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            return json.load(f)["tools"]

    def test_examples_cover_exactly_the_dispatched_tools(self):
        examples = self._examples()
        self.assertEqual(sorted(examples),
                         ["describe_tool", "get_job_status",
                          "segment_shots"])
        for tool, entry in examples.items():
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)

    def test_describe_tool_carries_input_examples(self):
        out = gateway_mod.lambda_handler({}, self._ctx("describe_tool"))
        self.assertIn("segment_shots", out["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        # The example IS the contract: round-trip it through the real
        # handler so drift fails this build, not an agent's first call.
        example = self._examples()["segment_shots"]["example"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        with mock.patch.object(gateway_mod, "_sfn", fake):
            out = gateway_mod.lambda_handler(
                dict(example), self._ctx("segment_shots"))
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()


if __name__ == "__main__":
    unittest.main()
