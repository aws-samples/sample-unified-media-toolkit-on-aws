"""
AgentCore Gateway tool dispatcher for the cast-resolution micro-workflow.

Three tools:
  resolve_cast    → derive workflow_id, StartExecution
  get_job_status  → DescribeExecution; outputs[] when complete
  describe_tool   → version, state machine arn, output contract

This dispatcher is deliberately THIN: every check that costs a network
call (sibling artifacts exist and parse) lives in ValidateSources. What
the gateway owns is IDENTITY: `workflow_id = uuid5(request + tool_version
[+ token])` via the copied `workflow_identity` module, the execution
named by it, and the family's absorbed-resubmit contract.

Identity caveat, documented rather than engineered away: the request
names sibling RUN PREFIXES, so identical prefixes = identical work — a
re-run of a sibling changes its prefix and correctly re-binds. The remap
tables are only meaningful against the sibling workflow_ids pinned in
their own provenance.
"""
from __future__ import annotations

import json
import logging
import os
import re

import boto3
from botocore.exceptions import ClientError

# the zip is flat, so this is a top-level import at runtime; test loaders
# put lib/ on the path rather than this module carrying a fallback import
from pb_common import DEFAULT_FLAGS
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
TARGET_NAME = os.environ.get("TARGET_NAME", "cast-resolution")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")

_TOOL_DELIMITER = "___"
_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$")

# The four sibling-run inputs; at least one IDENTITY source required.
_SOURCE_KEYS = ("speech_diarization_s3uri", "face_recognition_s3uri",
                "shot_segmentation_s3uri", "shot_metadata_s3uri")
_IDENTITY_KEYS = ("speech_diarization_s3uri", "face_recognition_s3uri")

# Labeled products. speaker-map / face-map are CONDITIONAL on their source
# being present (three-state semantics: absent source -> absent output),
# and credits on a credits read having happened — get_job_status lists
# what actually exists rather than a fixed set.
_OUTPUTS = (("persons", "persons.json", "application/json"),
            ("speaker-map", "speaker_map.json", "application/json"),
            ("face-map", "face_map.json", "application/json"),
            ("credits", "credits.json", "application/json"),
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

    if tool == "resolve_cast":
        return _resolve_cast(event)
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
                "Person-binding micro-workflow: binds the anonymous "
                "identities sibling runs emit — diarization SPEAKER_IDs, "
                "face-recognition person ids, on-screen text names — into "
                "person entities with names, types, and evidence. Names "
                "come only from independent naming mechanisms "
                "(self-introductions, vocatives, chyrons, face sources); "
                "co-occurrence alone proposes, never names, and a critic "
                "demotes contradictions rather than guessing. Remap "
                "tables are pinned to the exact sibling workflow_ids they "
                "were computed from."),
        }
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# Tool: resolve_cast
# ---------------------------------------------------------------------------

def _resolve_cast(event: dict) -> dict:
    """
    Inputs (all sibling RUN PREFIXES on the proxy bucket; at least one of
    the two identity sources is required):
      speech_diarization_s3uri  speakers, quotes, transcript name cues
      face_recognition_s3uri    face tracks, celebrity/collection names
      shot_segmentation_s3uri   shot lattice (optional)
      shot_metadata_s3uri       enrichment overlay-text names (optional)
      read_onscreen_text        optional, default true: read credits/
                                chyrons/titles from frames (the on-screen
                                name channel, ~$0.35/asset batch,
                                measured on the development corpus).
                                false binds on free evidence only.
      llm_enrichment            optional, default true: the LLM
                                enrichment tier — dialogue-addressee
                                resolution, credits cast adjudication,
                                report-only contradiction flags
                                (~$0.05/asset, measured on the
                                development corpus). false skips both
                                enrichment rounds.
      use_batch                 optional, default true: batch-price the
                                witness reads (24h SLA ceiling). false =
                                sync, for latency-sensitive callers, ~2x
                                read cost per the published Amazon
                                Bedrock batch-inference pricing.
      idempotency_token         optional; same token + same parameters
                                returns the EXISTING run.

    Returns a HANDLE — job_id, workflow_id, status, output_s3_prefix —
    never results; poll get_job_status.
    """
    if not PROXY_BUCKET or not STATE_MACHINE_ARN:
        return _error("PROXY_BUCKET and STATE_MACHINE_ARN env vars must be "
                      "set on this function", "CONFIG_ERROR")
    if "uuid" in event:
        return _error(
            "`uuid` is not a parameter of this tool. Use "
            "`idempotency_token` (optional): same token + same parameters "
            "returns the existing run", "INVALID_PARAMETER")

    token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")

    request: dict = {}
    for key in _SOURCE_KEYS:
        uri = (event.get(key) or "").strip().rstrip("/")
        if not uri:
            continue
        if not uri.startswith("s3://"):
            return _error(f"{key} must be an s3:// run prefix",
                          "INVALID_PARAMETER")
        request[key] = uri
    if not any(k in request for k in _IDENTITY_KEYS):
        return _error(
            "at least one identity source is required: "
            "speech_diarization_s3uri or face_recognition_s3uri",
            "MISSING_PARAMETER")
    # Caller flags, all NORMALIZED into the hash (omitted == default ==
    # same work, absorbs into the same run). Semantics + costs: the
    # docstring table above; defaults: pb_common.DEFAULT_FLAGS (the one
    # home). Flags exist to REMOVE cost, not to unlock features.
    for _flag, _default in DEFAULT_FLAGS.items():
        request[_flag] = bool(event.get(_flag, _default))
    # a misspelled or renamed flag silently gets its DEFAULT — the caller
    # believes they turned something off and pays for it anyway. Warn on
    # any boolean parameter this tool does not know (bedrock's retired-
    # `mode` treatment): warned and ignored, never an error.
    _known = (set(_SOURCE_KEYS) | set(DEFAULT_FLAGS)
              | {"idempotency_token"})
    for _key in sorted(set(event) - _known):
        if isinstance(event.get(_key), bool):
            logger.warning(
                "unknown flag %r ignored — this tool's flags are %s; the "
                "run proceeds with defaults for anything not named",
                _key, sorted(DEFAULT_FLAGS))

    workflow_id = compute_workflow_id(
        request, tool_version=TOOL_VERSION,
        extra_material=(token,) if token else ())
    output_s3_prefix = (f"s3://{PROXY_BUCKET}/micro-workflow/"
                        f"{TARGET_NAME}/{workflow_id}")
    workflow_input = {"uuid": workflow_id, "input": request}
    logger.info("resolve_cast: workflow_id=%s sources=%s",
                workflow_id, sorted(request))

    try:
        resp = _sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN, name=workflow_id,
            input=json.dumps(workflow_input))
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ExecutionAlreadyExists":
            return _existing_execution(workflow_id, output_s3_prefix)
        message = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s", message)
        return _error(f"Failed to start the binding job: {message}",
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
                      "resolve_cast", "MISSING_PARAMETER")
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

    # Best-effort header lift; an unreadable persons.json degrades the poll
    # to outputs-only, never fails it.
    try:
        doc = _read_json_if_exists(f"{prefix}/persons.json")
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("persons header lift skipped: %s", exc)
        doc = None
    if doc:
        ps = doc.get("persons") or []
        result["persons"] = len(ps)
        result["named"] = sum(1 for p in ps if p.get("name"))
        result["cross_modal"] = sum(
            1 for p in ps if p.get("type") == "on_camera_speaker")
        result["vetoes"] = len(doc.get("vetoes") or [])
        result["sources"] = sorted((doc.get("sources") or {}).keys())

    # conditional outputs: list what EXISTS (absent source -> absent map)
    present = _existing_keys(prefix)
    result["outputs"] = [
        _output_entry(prefix, basename, label, mime)
        for label, basename, mime in _OUTPUTS
        if f"{prefix}/{basename}" in present]
    return result


# always written by a run that reached Bind; safe to advertise when the
# existence listing itself fails
_UNCONDITIONAL = ("persons.json", "usage_tracking.json")


def _existing_keys(prefix: str) -> set:
    try:
        # MaxKeys 200: underscore prefixes (_enrich/, _plan/) sort BEFORE
        # the products, so a small cap could return a page of internals
        # and silently omit real outputs (review, 2026-09-08)
        resp = _s3.list_objects_v2(Bucket=PROXY_BUCKET, Prefix=prefix + "/",
                                   MaxKeys=200)
        return {o["Key"] for o in resp.get("Contents", [])}
    except ClientError as exc:
        logger.warning("output listing failed: %s", exc)
        # degrade to the UNCONDITIONAL products only: advertising a
        # conditional one (credits, the maps) here would hand pollers a
        # presignable-looking uri that 404s on most runs
        return {f"{prefix}/{b}" for b in _UNCONDITIONAL}


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
