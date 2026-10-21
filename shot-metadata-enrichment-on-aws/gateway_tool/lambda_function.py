"""
AgentCore Gateway tool dispatcher for the shot-metadata-enrichment
micro-workflow (README.md §10 — conversation-analysis's dispatcher, adapted).

Three tools:
  enrich_shot_metadata → derive workflow_id, StartExecution
  get_job_status       → DescribeExecution; outputs[] when complete
  describe_tool        → version, state machine arn, output contract

This dispatcher is deliberately THIN: every check that costs a network call
(shots.json exists and parses, face join, prompt fetch, model registry)
lives in StateValidateAndPlan, where a refusal is still before any
inference spend. What the gateway owns is IDENTITY: `workflow_id =
uuid5(request + tool_version [+ token])` via the copied `workflow_identity`
module, the execution named by it, and the family's absorbed-resubmit
contract.

The one identity caveat, inherited from the template: the workflow BINDS
the registry's `default` prompt, whose CONTENT can change pre-GA through
the seeder backdoor — prompt bytes are NOT in this hash. A rebuilt default
needs a fresh `idempotency_token` to re-enrich an already-processed run;
the actual prompt bytes are recorded in the scenes provenance (source_md5).
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
                               warn_idempotency_token)  # noqa: F401  (bundle
# contract: every gateway exposes the shared warn helper as an attribute)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_s3 = boto3.client("s3")

STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN", "")
# Must equal the GatewayTarget Name in media-toolkit-gateway — it is the {target}
# segment of the derived output prefix.
TARGET_NAME = os.environ.get("TARGET_NAME", "shot-metadata-enrichment")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")

_TOOL_DELIMITER = "___"
_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$")

# The labeled products — basenames are stable API.
_OUTPUTS = (("shots", "shot_metadata.jsonl", "application/jsonl"),
            ("scenes", "scenes.json", "application/json"),
            ("window-manifest", "window_manifest.json", "application/json"),
            ("usage", "usage_tracking.json", "application/json"))

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

    if tool == "enrich_shot_metadata":
        return _enrich_shot_metadata(event)
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
                "Shot-metadata-enrichment micro-workflow: takes a "
                "shot-segmentation run, renders windows of labeled frame "
                "images through the registry's shot-metadata-enrichment "
                "prompt (joining face-recognition names and a reference-"
                "face gallery, and the diarization transcript, when "
                "supplied), and emits one validated metadata document per "
                "shot plus deterministically assembled scenes. Uses "
                "prompt-registry and bedrock-inference internally; callers "
                "never render or submit prompts themselves. Batch inference "
                "by default."),
        }
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# Tool: enrich_shot_metadata
# ---------------------------------------------------------------------------

def _enrich_shot_metadata(event: dict) -> dict:
    """
    Required:
      shot_segmentation_s3uri  s3:// uri of a completed shot-segmentation
                               run on the proxy bucket — the run prefix or
                               its embeddings/shots.json directly.

    Optional:
      face_recognition_s3uri   A face-recognition run (prefix or its
                               facemetadata.json): named people join per
                               shot as text, and each window carries a
                               labeled reference-face gallery built from
                               the run's rep_face_key crops.
      transcript_s3uri        A speech-diarization final_transcript.json;
                               sliced per window as SPOKEN evidence.
      program_names            Program title(s) for the prompt's context.
      overall_content_type     e.g. "Movies/Feature Films",
                               "News/Current Affairs".
      model_chain              Bedrock model ids, primary first (list or
                               comma-separated), every one IMAGE-capable.
                               Absent → anthropic.claude-sonnet-4-6 (the
                               model the prompt was validated on; a
                               versioned constant). Validated against
                               bedrock-inference's registry before spend.
      use_batch                Absent → TRUE: an hour of content is
                               ~100+ image-heavy windows and nothing
                               downstream is interactive, so the 50% batch
                               discount (per the published Amazon Bedrock
                               batch-inference pricing) is the default.
                               Pass false for sync latency.
      history_scope            Your OWN asset identity for the inference
                               history prefix. Absent → derived from the
                               shot-segmentation run's workflow_id.
      prompt_version           A NAMED version of the registry's
                               shot-metadata-enrichment prompt (e.g.
                               "default_news" for news programming).
                               Absent → the category default. In the
                               hash: a different version is different
                               work.
      idempotency_token        Same token + same parameters returns the
                               EXISTING run. Also the RERUN mechanism:
                               prompt content is not in the hash.

    Returns a HANDLE — job_id, workflow_id, status, output_s3_prefix —
    never results; poll get_job_status.
    """
    if not PROXY_BUCKET or not STATE_MACHINE_ARN:
        return _error("PROXY_BUCKET and STATE_MACHINE_ARN env vars must be "
                      "set on this function", "CONFIG_ERROR")
    if "uuid" in event:
        return _error(
            "`uuid` is no longer a parameter of this tool. It has been "
            "replaced by `idempotency_token` (optional): same token + same "
            "parameters returns the existing run", "INVALID_PARAMETER")

    token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")

    uri = (event.get("shot_segmentation_s3uri") or "").strip()
    if not uri.startswith("s3://"):
        return _error(
            "shot_segmentation_s3uri is required: an s3:// uri to a "
            "completed shot-segmentation run (its prefix or its "
            "embeddings/shots.json) on the proxy bucket",
            "MISSING_PARAMETER")

    request: dict = {"shot_segmentation_s3uri": uri}
    for key in ("face_recognition_s3uri", "transcript_s3uri"):
        val = (event.get(key) or "").strip()
        if val:
            if not val.startswith("s3://"):
                return _error(f"{key} must be an s3:// uri",
                              "INVALID_PARAMETER")
            request[key] = val
    for key in ("program_names", "overall_content_type"):
        val = (event.get(key) or "").strip()
        if val:
            request[key] = val
    chain = event.get("model_chain")
    if isinstance(chain, str):
        chain = [part.strip() for part in chain.split(",") if part.strip()]
    if chain:
        request["model_chain"] = chain
    if event.get("use_batch") is not None:
        request["use_batch"] = bool(event["use_batch"])
    scope = (event.get("history_scope") or "").strip()
    if scope:
        request["history_scope"] = scope
    # prompt_version selects a NAMED registry version of the
    # shot-metadata-enrichment prompt (e.g. "default_news"); absent means
    # the category default. It rides the request, so it is IN the hash:
    # the same shots under a different prompt version are different work.
    version = (event.get("prompt_version") or "").strip()
    if version:
        request["prompt_version"] = version

    workflow_id = compute_workflow_id(
        request, tool_version=TOOL_VERSION,
        extra_material=(token,) if token else ())
    output_s3_prefix = (f"s3://{PROXY_BUCKET}/micro-workflow/"
                        f"{TARGET_NAME}/{workflow_id}")
    workflow_input = {"uuid": workflow_id, "input": request}
    logger.info("enrich_shot_metadata: workflow_id=%s uri=%s",
                workflow_id, uri)

    try:
        resp = _sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN, name=workflow_id,
            input=json.dumps(workflow_input))
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ExecutionAlreadyExists":
            return _existing_execution(workflow_id, output_s3_prefix)
        message = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s", message)
        return _error(f"Failed to start the enrichment job: {message}",
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
                      "enrich_shot_metadata", "MISSING_PARAMETER")
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

    # Best-effort header lift: the labeled outputs[] below are the
    # contract; an unreadable scenes file degrades the poll to
    # outputs-only, never fails it.
    try:
        scenes = _read_json_if_exists(f"{prefix}/scenes.json")
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("scenes header lift skipped: %s", exc)
        scenes = None
    if scenes:
        result["scenes"] = len(scenes.get("scenes") or [])
        result["shots_total"] = scenes.get("shots_total")
        result["shots_gap"] = scenes.get("shots_gap") or []
        result["pairing_repair_rate"] = scenes.get("pairing_repair_rate")
    result["outputs"] = [
        _output_entry(prefix, basename, label, mime)
        for label, basename, mime in _OUTPUTS]
    return result


def _contract_desc(label: str):
    if not _OUTPUT_CONTRACT:
        return None
    entry = (_OUTPUT_CONTRACT.get("outputs") or {}).get(label) or {}
    return entry.get("description")


def _output_entry(prefix: str, basename: str, label: str, mime: str) -> dict:
    entry = {"uri": f"s3://{PROXY_BUCKET}/{prefix}/{basename}",
             "mime": mime, "label": label}
    desc = _contract_desc(label)
    if desc:
        entry["desc"] = desc
    return entry


# ---------------------------------------------------------------------------

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
