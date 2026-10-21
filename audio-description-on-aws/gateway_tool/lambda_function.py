"""
AgentCore Gateway tool dispatcher for the audio-description
micro-workflow (README.md Design record #8).

Three tools:
  generate_ad     -> derive workflow_id, StartExecution
  get_job_status  -> DescribeExecution; outputs[] when complete
  describe_tool   -> version, state machine arn, output contract

THIN by design (identity only): validation that costs a read lives in
StateValidateAndPlan. AD is prompt-bound (five registry categories), so
a category prompt rebuild re-runs via a fresh idempotency_token — prompt
bytes are not in the hash.
"""
from __future__ import annotations

import json
import logging
import os
import re

import boto3
from botocore.exceptions import ClientError

from ad_common import (OPTIONAL_INPUT_URIS, REQUIRED_INPUT_URIS,
                       people_source_error)
from workflow_identity import (compute_workflow_id,
                               read_idempotency_token,
                               warn_idempotency_token)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_s3 = boto3.client("s3")

STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "audio-description")
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

# Labels went kebab-case 2026-10-13, but runs finalized BEFORE that deploy
# (and their idempotent resubmits, for the ~90-day execution retention)
# recorded the retired spellings in ad_manifest.json — the poll aliases
# them so every response speaks the current contract.
_LABEL_ALIASES = {"cue_ledger": "cue-ledger",
                  "render_manifest": "render-manifest"}

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

_REQUIRED = REQUIRED_INPUT_URIS
_OPTIONAL = OPTIONAL_INPUT_URIS


def lambda_handler(event: dict, context) -> dict:
    event = event or {}
    logger.info("Event keys: %s", sorted(event.keys()))
    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    if tool == "generate_ad":
        return _generate_ad(event)
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
                "Audio description: sibling runs in (shot-seg, enrichment, "
                "diarization, conversation-analysis, face-rec; optional "
                "audiostem, reviewer decisions, cast-resolution person "
                "bindings — credits ride the cast run), "
                "a narrated AD script out — windows with duties, name-map "
                "discipline, TTS-fitted takes, a mechanical repair chain, "
                "coverage oracles and a two-axis reviewer ledger. "
                "Delivers srt/vtt/DAPT/review.html plus a "
                "render-manifest for media-processing's ad_mix group "
                "(the mix is a chained process_media call — this "
                "workflow renders no audio)."),
        }
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# Tool: generate_ad
# ---------------------------------------------------------------------------

def _generate_ad(event: dict) -> dict:
    """
    Required:
      shot_segmentation_s3uri, shot_metadata_s3uri,
      speech_diarization_s3uri, conversation_analysis_s3uri
                                 sibling run roots
      face_recognition_s3uri     OR cast_resolution_s3uri — at least
                                 one people source (cast runs from
                                 2026-09-08 export person appearance
                                 timelines and can stand alone)
      voice                      a text-to-speech voice_id

    Optional:
      audiostem_s3uri            music/sfx gating evidence
      decisions_s3uri            reviewer-accepted absences
      cast_resolution_s3uri      adjudicated person bindings anchoring
                                 the name map. Must be a COMPLETED run
                                 built from THIS request's
                                 face_recognition and speech_diarization
                                 runs (ids are per-run labels) — the
                                 RUN refuses at validation otherwise;
                                 the submit still returns a handle, so
                                 poll get_job_status
      sung_vocal_policy          narrate (default) | avoid
      language                   narration language (default: the
                                 program's, from diarization)
      title                      delivery basename
      idempotency_token          same token + same parameters returns the
                                 EXISTING run; also the rerun mechanism
                                 after a prompt rebuild

    Returns a HANDLE — job_id, workflow_id, status, output_s3_prefix —
    never results; poll get_job_status.
    """
    if not PROXY_BUCKET or not STATE_MACHINE_ARN:
        return _error("PROXY_BUCKET and STATE_MACHINE_ARN env vars must "
                      "be set on this function", "CONFIG_ERROR")

    token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")

    request: dict = {}
    for name in _REQUIRED:
        uri = (event.get(name) or "").strip()
        if not uri.startswith("s3://"):
            return _error(f"{name} is required (an s3:// run prefix)",
                          "MISSING_PARAMETER")
        request[name] = uri
    for name in _OPTIONAL:
        uri = (event.get(name) or "").strip()
        if uri:
            request[name] = uri
    rule = people_source_error(request)
    if rule:
        return _error(rule, "MISSING_PARAMETER")
    voice = (event.get("voice") or "").strip()
    if not voice:
        return _error("voice is required (a text-to-speech voice_id)",
                      "MISSING_PARAMETER")
    request["voice"] = voice
    for name in ("sung_vocal_policy", "language", "title",
                 "history_scope"):
        val = (event.get(name) or "").strip()
        if val:
            request[name] = val
    if event.get("frame_inspection_s3uri"):
        # retired 2026-09-08 — refused by NAME like `mix`, never
        # silently dropped: a caller relying on it would otherwise get
        # a run whose credits evidence quietly vanished (and whose
        # workflow_id hashes identically to the credit-less request)
        return _error(
            "frame_inspection_s3uri is no longer a parameter: "
            "frame-inspection is internal, and its credits-read "
            "evidence rides the cast-resolution run — pass "
            "cast_resolution_s3uri instead", "INVALID_PARAMETER")
    if event.get("mix") or event.get("audio_proxy_s3uri"):
        # refused at SUBMIT, not one Lambda deep: this workflow renders
        # no audio, and starting an execution certain to fail would
        # also occupy its workflow_id (review 2026-08-31)
        return _error(
            "this workflow renders no audio — for the mix, chain "
            "media-processing___process_media with output_groups="
            "[{format: 'ad_mix', render_manifest_s3uri: <this run's "
            "render-manifest output>}] on the programme source",
            "INVALID_PARAMETER")

    workflow_id = compute_workflow_id(
        request, tool_version=TOOL_VERSION,
        extra_material=(token,) if token else ())
    output_s3_prefix = (f"s3://{PROXY_BUCKET}/micro-workflow/"
                        f"{TARGET_NAME}/{workflow_id}")
    logger.info("generate_ad: workflow_id=%s", workflow_id)

    try:
        resp = _sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN, name=workflow_id,
            input=json.dumps({"uuid": workflow_id, "input": request}))
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ExecutionAlreadyExists":
            return _existing_execution(workflow_id, output_s3_prefix)
        message = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s", message)
        return _error(f"Failed to start the audio-description job: "
                      f"{message}", "SFN_START_FAILED")

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
                      "generate_ad", "MISSING_PARAMETER")
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
        manifest = _read_json_if_exists(f"{prefix}/ad_manifest.json")
        if manifest:
            result["shipped_cues"] = manifest.get("shipped_cues")
            result["gaps"] = manifest.get("gaps")
            for label, uri in (manifest.get("outputs") or {}).items():
                outputs.append(_output_entry(uri,
                                             _LABEL_ALIASES.get(label, label)))
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("manifest lift skipped: %s", exc)
        result.pop("shipped_cues", None)
        outputs = []
    outputs.append(_output_entry(
        f"s3://{PROXY_BUCKET}/{prefix}/ad_manifest.json", "manifest"))
    outputs.append(_output_entry(
        f"s3://{PROXY_BUCKET}/{prefix}/usage_tracking.json", "usage"))
    result["outputs"] = outputs
    return result


# ---------------------------------------------------------------------------

def _contract_desc(label: str):
    if not _OUTPUT_CONTRACT:
        return None
    entry = (_OUTPUT_CONTRACT.get("outputs") or {}).get(label) or {}
    return entry.get("description")


def _output_entry(uri: str, label: str) -> dict:
    mime = ("application/json" if uri.endswith(".json") else
            "text/html" if uri.endswith(".html") else
            "application/ttml+xml" if uri.endswith(".xml") else
            "audio/wav" if uri.endswith(".wav") else
            "text/vtt" if uri.endswith(".vtt") else "text/plain")
    entry = {"uri": uri, "mime": mime, "label": label}
    desc = _contract_desc(label)
    if desc:
        entry["desc"] = desc
    return entry


def _current_state(execution_arn: str):
    try:
        events = _sfn.get_execution_history(
            executionArn=execution_arn, reverseOrder=True, maxResults=40)
        for ev in events.get("events", []):
            detail = ev.get("stateEnteredEventDetails")
            if detail:
                return detail.get("name")
    except ClientError:
        return None
    return None


def _failure_cause(execution_arn: str):
    try:
        events = _sfn.get_execution_history(
            executionArn=execution_arn, reverseOrder=True, maxResults=40)
        for ev in events.get("events", []):
            for key in ("executionFailedEventDetails",
                        "executionAbortedEventDetails",
                        "executionTimedOutEventDetails"):
                detail = ev.get(key)
                if detail:
                    return (detail.get("error", "") + ": "
                            + detail.get("cause", ""))[:400]
    except ClientError:
        return None
    return None


def _read_json_if_exists(key: str):
    try:
        body = _s3.get_object(Bucket=PROXY_BUCKET, Key=key)["Body"].read()
        return json.loads(body.decode("utf-8"))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _resolve_tool(context):
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        return (raw.split(_TOOL_DELIMITER, 1)[1]
                if _TOOL_DELIMITER in raw else raw)
    except (AttributeError, KeyError, TypeError):
        logger.warning("bedrockAgentCoreToolName not found in the Lambda "
                       "client context")
        return ""


def _error(message: str, code: str = "ERROR") -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
