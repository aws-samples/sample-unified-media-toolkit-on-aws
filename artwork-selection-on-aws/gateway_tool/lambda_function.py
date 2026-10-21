"""
AgentCore Gateway tool dispatcher for the artwork-selection
micro-workflow.

Three tools:
  select_artwork  → derive workflow_id, StartExecution
  get_job_status  → DescribeExecution; outputs[] when complete
  describe_tool   → version, state machine arn, output contract

This dispatcher is deliberately THIN: every check that costs a network
call (sibling artifacts exist and parse) lives in
ValidateAndCompilePlan. What the gateway owns is IDENTITY —
`workflow_id = uuid5(request + tool_version [+ token])` — and the
CHEAP validation of the steering parameters against the closed
vocabulary, so a typo'd archetype refuses here, not three states in.
"""
from __future__ import annotations

import json
import logging
import os
import re

import boto3
from botocore.exceptions import ClientError

# the zip is flat, so these are top-level imports at runtime
from profiles import ARCHETYPES
from wf_common import DEFAULT_FLAGS
from workflow_identity import (compute_workflow_id,
                               read_idempotency_token,
                               warn_idempotency_token)  # noqa: F401  (bundle
# contract: every gateway exposes the shared warn helper as an attribute)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_s3 = boto3.client("s3")

STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "artwork-selection")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")

_TOOL_DELIMITER = "___"
_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$")

_SOURCE_KEYS = ("shot_segmentation_s3uri", "shot_metadata_s3uri",
                "cast_resolution_s3uri", "face_recognition_s3uri",
                "program_structure_s3uri")
_REQUIRED_KEYS = ("shot_segmentation_s3uri", "shot_metadata_s3uri")

_MAX_PER_ARCHETYPE = 10
_MAX_BRIEF_CHARS = 2000

_OUTPUTS = (("artwork", "artwork_candidates.json", "application/json"),
            ("candidates", "candidates.json", "application/json"))

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

    if tool == "select_artwork":
        return _select_artwork(event)
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
                "Artwork-selection micro-workflow: selects "
                "representative artwork frames for an asset. Code "
                "filters and ranks candidate shots per archetype "
                "(hero close-up, two-shot, establishing vista) from "
                "sibling run outputs; one vision pass reranks WITHIN "
                "each pre-qualified slate and flags unusable frames — "
                "the model never scans the program and cannot add a "
                "candidate. Output is a ranked diverse slate per "
                "archetype with the evidence that qualified each "
                "frame; the final pick is editorial."),
        }
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# Tool: select_artwork
# ---------------------------------------------------------------------------

def _select_artwork(event: dict) -> dict:
    """
    Inputs (sibling RUN PREFIXES on the proxy bucket):
      shot_segmentation_s3uri   REQUIRED: shot lattice + frame archives
      shot_metadata_s3uri       REQUIRED: the semantic layer the
                                archetype queries read
      cast_resolution_s3uri     recommended: top cast by screen time —
                                hero/two-shot targeting
      face_recognition_s3uri    optional: identity FALLBACK when cast
                                is absent — top faces by screen time,
                                temporal track join; unnamed clusters
                                ride under anonymous labels (the
                                brief's must-include only bites with
                                real names). Ignored for identity when
                                cast_resolution_s3uri is present.
      program_structure_s3uri   optional: credits/recap spans excluded
                                in addition to the spoiler cut
    Steering (all optional):
      selection_brief           free text, the editorial seat —
                                translated into the closed vocabulary;
                                re-weights among pre-qualified
                                candidates, never reaches eligibility
      genre                     selects the archetype profile; never
                                inferred; unknown/absent = neutral
      archetypes                explicit list from the closed
                                vocabulary; order = slate priority
      candidates_per_archetype  1-10, default 4
      spoiler_cutoff            0-1, default 0.75 (position cut)
      model_chain               1-4 Bedrock model ids, primary first,
                                for BOTH inference rounds (brief
                                translation + vision rerank). Default:
                                Sonnet 4.6 with Nova 2 Lite backup.
                                Ids must be registered with the
                                bedrock-inference deployment.
      use_batch                 default false (sync — the rounds are
                                single records). true batch-prices
                                them at the batch SLA.
      idempotency_token         same token + same parameters returns
                                the EXISTING run.

    Returns a HANDLE — job_id, workflow_id, status, output_s3_prefix —
    never results; poll get_job_status.
    """
    if not PROXY_BUCKET or not STATE_MACHINE_ARN:
        return _error("PROXY_BUCKET and STATE_MACHINE_ARN env vars must "
                      "be set on this function", "CONFIG_ERROR")
    if "uuid" in event:
        return _error(
            "`uuid` is not a parameter of this tool. Use "
            "`idempotency_token` (optional): same token + same "
            "parameters returns the existing run", "INVALID_PARAMETER")
    if (event.get("full_res") or event.get("asset_s3uri")
            or event.get("asset_s3_uri")):
        return _error(
            "full_res source extraction is not yet supported — the "
            "default path serves shot-segmentation's stored frame "
            "archives; omit full_res and asset_s3uri",
            "INVALID_PARAMETER")

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
    for key in _REQUIRED_KEYS:
        if key not in request:
            return _error(f"{key} is required — it defines the "
                          f"candidate universe", "MISSING_PARAMETER")

    err = _normalize_steering(event, request)
    if err:
        return err

    workflow_id = compute_workflow_id(
        request, tool_version=TOOL_VERSION,
        extra_material=(token,) if token else ())
    output_s3_prefix = (f"s3://{PROXY_BUCKET}/micro-workflow/"
                        f"{TARGET_NAME}/{workflow_id}")
    workflow_input = {"uuid": workflow_id, "input": request}
    logger.info("select_artwork: workflow_id=%s sources=%s",
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
        return _error(f"Failed to start the selection job: {message}",
                      "SFN_START_FAILED")

    return {"job_id": resp["executionArn"], "status": "RUNNING",
            "workflow_id": workflow_id,
            "output_s3_prefix": output_s3_prefix}


def _normalize_steering(event: dict, request: dict):
    """Validate + normalize the steering parameters INTO the request
    (they are all identity: different steering = different work).
    Returns an error dict, or None when the request is extended in
    place. Cheap checks only — the closed vocabulary lives in this
    zip, so a typo refuses here without a network call."""
    brief = str(event.get("selection_brief")
                or DEFAULT_FLAGS["selection_brief"]).strip()
    if len(brief) > _MAX_BRIEF_CHARS:
        return _error(f"selection_brief is {len(brief)} chars — the "
                      f"ceiling is {_MAX_BRIEF_CHARS}",
                      "INVALID_PARAMETER")
    request["selection_brief"] = brief

    request["genre"] = str(event.get("genre")
                           or DEFAULT_FLAGS["genre"]).strip().lower()

    archetypes = event.get("archetypes", DEFAULT_FLAGS["archetypes"])
    if archetypes is not None:
        if not isinstance(archetypes, list) or not archetypes \
                or not all(isinstance(a, str) for a in archetypes):
            return _error("archetypes must be a non-empty list of "
                          "archetype names", "INVALID_PARAMETER")
        unknown = [a for a in archetypes if a not in ARCHETYPES]
        if unknown:
            return _error(
                f"unknown archetype(s): {unknown} — the closed "
                f"vocabulary is {sorted(ARCHETYPES)}",
                "INVALID_PARAMETER")
        if len(set(archetypes)) != len(archetypes):
            return _error("archetypes contains duplicates",
                          "INVALID_PARAMETER")
    request["archetypes"] = archetypes

    n = event.get("candidates_per_archetype",
                  DEFAULT_FLAGS["candidates_per_archetype"])
    if not isinstance(n, (int, float)) or int(n) != n \
            or not 1 <= int(n) <= _MAX_PER_ARCHETYPE:
        return _error(f"candidates_per_archetype must be an integer "
                      f"1-{_MAX_PER_ARCHETYPE}", "INVALID_PARAMETER")
    request["candidates_per_archetype"] = int(n)

    cut = event.get("spoiler_cutoff", DEFAULT_FLAGS["spoiler_cutoff"])
    try:
        cut = float(cut)
    except (TypeError, ValueError):
        cut = -1.0
    if not 0.0 < cut <= 1.0:
        return _error("spoiler_cutoff must be a number in (0, 1]",
                      "INVALID_PARAMETER")
    request["spoiler_cutoff"] = cut

    chain = event.get("model_chain", DEFAULT_FLAGS["model_chain"])
    if not isinstance(chain, list) or not chain or len(chain) > 4 \
            or not all(isinstance(m, str) and m.strip() for m in chain):
        return _error("model_chain must be a list of 1-4 model ids, "
                      "primary first", "INVALID_PARAMETER")
    request["model_chain"] = [m.strip() for m in chain]

    use_batch = event.get("use_batch", DEFAULT_FLAGS["use_batch"])
    if not isinstance(use_batch, bool):
        return _error("use_batch must be a boolean",
                      "INVALID_PARAMETER")
    request["use_batch"] = use_batch

    # a misspelled parameter silently gets its DEFAULT — warn on any
    # key this tool does not know, never an error
    known = (set(_SOURCE_KEYS) | set(DEFAULT_FLAGS)
             | {"idempotency_token", "full_res", "asset_s3uri",
              "asset_s3_uri"})
    for key in sorted(set(event) - known):
        logger.warning(
            "unknown parameter %r ignored — this tool's parameters "
            "are %s; the run proceeds with defaults for anything "
            "not named", key, sorted(known))
    return None


def _existing_execution(workflow_id: str, output_s3_prefix: str) -> dict:
    execution_arn = (f"{STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
                     f":{workflow_id}")
    try:
        existing = _sfn.describe_execution(executionArn=execution_arn)
    except ClientError as exc:
        return _error(f"Job already exists but could not retrieve "
                      f"status: {exc.response['Error']['Message']}",
                      "SFN_DESCRIBE_FAILED")
    status = existing.get("status", "RUNNING")
    result = {
        "job_id": execution_arn, "status": status,
        "workflow_id": workflow_id,
        "output_s3_prefix": output_s3_prefix,
        # Byte-pinned across every bundle gateway — do not reword locally.
        "message": ("identical request already submitted — this call "
                    "did not resubmit; call get_job_status with this "
                    "job_id for status and results"),
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
        return _error("job_id is required — the execution ARN returned "
                      "by select_artwork", "MISSING_PARAMETER")
    if not _SFN_ARN_RE.match(job_id):
        return _error(f"job_id does not look like a Step Functions "
                      f"execution ARN: {job_id}", "INVALID_PARAMETER")
    try:
        execution = _sfn.describe_execution(executionArn=job_id)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ExecutionDoesNotExist":
            return _error("no such execution — check the job_id "
                          "(execution records are kept ~90 days)",
                          "JOB_NOT_FOUND")
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

    # Best-effort header lift; an unreadable product degrades the poll
    # to outputs-only, never fails it.
    try:
        doc = _read_json_if_exists(f"{prefix}/artwork_candidates.json")
    except Exception as exc:                               # noqa: BLE001
        logger.warning("artwork header lift skipped: %s", exc)
        doc = None
    if doc:
        slates = doc.get("slates") or {}
        result["candidates"] = {
            a: len(s.get("candidates") or []) for a, s in slates.items()}
        result["rerank"] = (doc.get("rerank") or {}).get("status")

    present = _existing_keys(prefix)
    result["outputs"] = [
        _output_entry(prefix, basename, label, mime)
        for label, basename, mime in _OUTPUTS
        if f"{prefix}/{basename}" in present]
    return result


def _existing_keys(prefix: str) -> set:
    try:
        # MaxKeys 200: underscore prefixes (_plan/, _rerank/) sort
        # BEFORE the products, so a small cap could return a page of
        # internals and silently omit real outputs
        resp = _s3.list_objects_v2(Bucket=PROXY_BUCKET,
                                   Prefix=prefix + "/", MaxKeys=200)
        return {o["Key"] for o in resp.get("Contents", [])}
    except ClientError as exc:
        logger.warning("output listing failed: %s", exc)
        return {f"{prefix}/artwork_candidates.json"}


def _contract_desc(label: str):
    if not _OUTPUT_CONTRACT:
        return None
    entry = (_OUTPUT_CONTRACT.get("outputs") or {}).get(label) or {}
    return entry.get("description")


def _output_entry(prefix: str, basename: str, label: str,
                  mime: str) -> dict:
    entry = {"uri": f"s3://{PROXY_BUCKET}/{prefix}/{basename}",
             "mime": mime, "label": label}
    desc = _contract_desc(label)
    if desc:
        entry["desc"] = desc
    return entry


# ---------------------------------------------------------------------------

def _current_state(execution_arn: str):
    """The most recently ENTERED state — the caller's actual question
    on a RUNNING poll. One page walked backwards; None when nothing
    useful."""
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
        logger.warning("bedrockAgentCoreToolName not found in the "
                       "Lambda context")
        return ""


def _error(message: str, code: str) -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
