"""
AgentCore Gateway tool dispatcher for the program-structure-analysis
micro-workflow (README.md design record).

Three tools:
  analyze_program_structure -> derive workflow_id, StartExecution
  get_job_status            -> DescribeExecution; outputs[] when complete
  describe_tool             -> version, state machine arn, output contract

THIN by design (identity only): validation that costs a read lives in
StateValidateAndPlan. PSA makes NO model calls — the identity caveats the
prompt-bound siblings carry do not apply here; same inputs really are the
same work, and re-running after a SIBLING re-run means new input uris,
which re-identify naturally.
"""
from __future__ import annotations

import json
import logging
import os
import re

import boto3
from botocore.exceptions import ClientError

from workflow_identity import (compute_workflow_id,
                               read_idempotency_token,
                               warn_idempotency_token)  # noqa: F401

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_s3 = boto3.client("s3")

STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "program-structure-analysis")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")

_TOOL_DELIMITER = "___"
_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$")

try:
    with open(os.path.join(os.path.dirname(__file__),
                           "output_contract.json")) as _f:
        _OUTPUT_CONTRACT = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("output_contract.json unavailable: %s", _exc)
    _OUTPUT_CONTRACT = None

# input_examples.json is the INPUT twin of the output contract: one minimal
# valid request per tool, served by describe_tool so an agent gets a
# known-good starting call instead of guessing shapes from prose. A unit
# test round-trips the submit example through this handler's own
# validation, so an example that drifts from the real contract fails the
# build. Same degradation contract as the output contract: never takes
# submit/status down with it.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "input_examples.json")) as _f:
        _INPUT_EXAMPLES = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("input_examples.json unavailable: %s", _exc)
    _INPUT_EXAMPLES = None


def lambda_handler(event: dict, context) -> dict:
    event = event or {}
    logger.info("Event keys: %s", sorted(event.keys()))
    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    if tool == "analyze_program_structure":
        return _analyze_program_structure(event)
    if tool == "get_job_status":
        note = warn_idempotency_token(event)
        if note:
            logger.warning("get_job_status: %s", note)
        return _get_job_status(event)
    if tool == "describe_tool":
        note = warn_idempotency_token(event)
        if note:
            logger.warning("describe_tool: %s", note)
        resp = {
            "version": TOOL_VERSION,
            "state_machine_arn": STATE_MACHINE_ARN,
            "description": (
                "Program-structure analysis (Layer-2): a code-only rollup "
                "over sibling runs — CPL-style markers (blackframes, slate, "
                "bars+tone, opening title, end credits, textless tail), "
                "graded break candidates, narrative sequences, and in news "
                "mode the industrialized rundown (shot roles + stories with "
                "editorial/content/speech start times + first-class "
                "interstitials). No model calls; every judgment is Layer-1 "
                "or conversation-analysis evidence, assembled and "
                "annotated."),
        }
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# Tool: analyze_program_structure
# ---------------------------------------------------------------------------

def _analyze_program_structure(event: dict) -> dict:
    """
    Required:
      shot_segmentation_s3uri   the shot-segmentation run root
      shot_metadata_s3uri       the shot-metadata-enrichment run root

    Optional:
      conversation_analysis_s3uri  unlocks the news rundown (stories)
      transcript_s3uri             unlocks gap_map + orientation
      audiostem_s3uri              tone/silence/music corroboration
      mode               auto | narrative | news. Absent → auto (decided
                         from the evidence, recorded in mode_evidence).
      idempotency_token  Same token + same parameters returns the EXISTING
                         run (the workflow_id is a hash of both).

    Code-only: no model calls — every judgment already happened in the
    sibling runs; this workflow assembles markers, break candidates,
    sequences and (news) the rundown from them.

    Returns a HANDLE — job_id, workflow_id, status, output_s3_prefix —
    never results; poll get_job_status.
    """
    if not PROXY_BUCKET or not STATE_MACHINE_ARN:
        return _error("PROXY_BUCKET and STATE_MACHINE_ARN env vars must be "
                      "set on this function", "CONFIG_ERROR")

    token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")

    request: dict = {}
    for name, required in (("shot_segmentation_s3uri", True),
                           ("shot_metadata_s3uri", True),
                           ("conversation_analysis_s3uri", False),
                           ("transcript_s3uri", False),
                           ("audiostem_s3uri", False)):
        uri = (event.get(name) or "").strip()
        if required and not uri.startswith("s3://"):
            return _error(f"{name} is required (an s3:// run prefix)",
                          "MISSING_PARAMETER")
        if uri:
            request[name] = uri
    mode = (event.get("mode") or "").strip()
    if mode:
        if mode not in ("auto", "narrative", "news"):
            return _error("mode must be auto | narrative | news",
                          "INVALID_PARAMETER")
        request["mode"] = mode

    workflow_id = compute_workflow_id(
        request, tool_version=TOOL_VERSION,
        extra_material=(token,) if token else ())
    output_s3_prefix = (f"s3://{PROXY_BUCKET}/micro-workflow/"
                        f"{TARGET_NAME}/{workflow_id}")
    logger.info("analyze_program_structure: workflow_id=%s", workflow_id)

    try:
        resp = _sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN, name=workflow_id,
            input=json.dumps({"uuid": workflow_id, "input": request}))
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ExecutionAlreadyExists":
            return _existing_execution(workflow_id, output_s3_prefix)
        message = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s", message)
        return _error(f"Failed to start the program-structure job: {message}",
                      "SFN_START_FAILED")

    return {"job_id": resp["executionArn"], "status": "RUNNING",
            "workflow_id": workflow_id,
            "output_s3_prefix": output_s3_prefix}


def _existing_execution(workflow_id: str, output_s3_prefix: str) -> dict:
    execution_arn = (f"{STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
                     f":{workflow_id}")
    try:
        existing = _sfn.describe_execution(executionArn=execution_arn)
    except ClientError as exc:
        return _error(f"Job already exists but could not retrieve status: "
                      f"{exc.response['Error']['Message']}",
                      "SFN_DESCRIBE_FAILED")
    status = existing.get("status", "RUNNING")
    result = {
        "job_id": execution_arn, "status": status,
        "workflow_id": workflow_id, "output_s3_prefix": output_s3_prefix,
        # Byte-pinned across every bundle gateway — do not reword locally.
        "message": ("identical request already submitted — this call did "
                    "not resubmit; call get_job_status with this job_id "
                    "for status and results"),
    }
    if status == "RUNNING":
        current = _current_state(execution_arn)
        if current:
            result["current_state"] = current
    elif status != "SUCCEEDED":
        result["error"] = existing.get("cause") or status
    return result


# ---------------------------------------------------------------------------
# Tool: get_job_status
# ---------------------------------------------------------------------------

def _get_job_status(event: dict) -> dict:
    job_id = (event.get("job_id") or "").strip()
    if not job_id:
        return _error("job_id is required — the execution ARN returned by "
                      "analyze_program_structure", "MISSING_PARAMETER")
    if not _SFN_ARN_RE.match(job_id):
        return _error(f"job_id does not look like a Step Functions "
                      f"execution ARN: {job_id}", "INVALID_PARAMETER")
    try:
        execution = _sfn.describe_execution(executionArn=job_id)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ExecutionDoesNotExist":
            return _error("no such execution — check the job_id (execution "
                          "records are kept ~90 days)", "JOB_NOT_FOUND")
        return _error(exc.response["Error"]["Message"], code)

    workflow_id = job_id.rsplit(":", 1)[-1]
    prefix = f"micro-workflow/{TARGET_NAME}/{workflow_id}"
    status = execution["status"]
    result = {"job_id": job_id, "status": status,
              "workflow_id": workflow_id,
              "output_s3_prefix": f"s3://{PROXY_BUCKET}/{prefix}"}

    if status == "RUNNING":
        current = _current_state(job_id)
        if current:
            result["current_state"] = current
        return result
    if status != "SUCCEEDED":
        result["error"] = _failure_cause(job_id) or status
        return result

    outputs = []
    try:
        manifest = _read_json_if_exists(f"{prefix}/psa_manifest.json")
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("manifest lift skipped: %s", exc)
        manifest = None
    if manifest:
        result["mode"] = manifest.get("mode")
        for label, uri in (manifest.get("outputs") or {}).items():
            outputs.append(_output_entry(uri, label))
    outputs.append(_output_entry(
        f"s3://{PROXY_BUCKET}/{prefix}/psa_manifest.json", "manifest"))
    result["outputs"] = outputs
    return result


# ---------------------------------------------------------------------------

def _contract_desc(label: str):
    if not _OUTPUT_CONTRACT:
        return None
    entry = (_OUTPUT_CONTRACT.get("outputs") or {}).get(label) or {}
    return entry.get("description")


def _output_entry(uri: str, label: str) -> dict:
    entry = {"uri": uri, "mime": "application/json", "label": label}
    desc = _contract_desc(label)
    if desc:
        entry["desc"] = desc
    return entry


def _current_state(execution_arn: str):
    """The most recently ENTERED state — the caller's actual question on a
    RUNNING poll. One page walked backwards; None when nothing useful."""
    try:
        events = _sfn.get_execution_history(
            executionArn=execution_arn, maxResults=40, reverseOrder=True
        ).get("events", [])
    except ClientError:
        return None
    for entry in events:
        detail = entry.get("stateEnteredEventDetails")
        if detail and detail.get("name"):
            return detail["name"]
    return None


def _failure_cause(execution_arn: str):
    try:
        events = _sfn.get_execution_history(
            executionArn=execution_arn, maxResults=10, reverseOrder=True
        ).get("events", [])
    except ClientError:
        return None
    for entry in events:
        for key in ("executionFailedEventDetails",
                    "executionAbortedEventDetails",
                    "executionTimedOutEventDetails"):
            detail = entry.get(key)
            if detail:
                return detail.get("cause") or detail.get("error")
    return None


def _read_json_if_exists(key: str):
    try:
        body = _s3.get_object(Bucket=PROXY_BUCKET, Key=key)["Body"].read()
        return json.loads(body.decode("utf-8"))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _resolve_tool(context) -> str:
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        if _TOOL_DELIMITER in raw:
            return raw.split(_TOOL_DELIMITER, 1)[1]
        return raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("bedrockAgentCoreToolName not found in the Lambda "
                       "context")
        return ""


def _error(message: str, code: str) -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
