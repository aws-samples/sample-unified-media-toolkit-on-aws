"""
AgentCore Gateway tool dispatcher for the frame-inspection micro-workflow
(README.md §5-§10).

Three tools:
  inspect_frames  → derive workflow_id, StartExecution
  get_job_status  → DescribeExecution; outputs[] from the manifest when done
  describe_tool   → version, state machine arn, output contract

This dispatcher is deliberately THIN: every check that costs a network
call (span auto-triggering over shot metadata, prompt fetch, window
indexing) lives in StateValidateAndPlan, where a refusal is still before
any inference spend. Even the category names are NOT validated here — the
category table lives in the workflow's fi_common (one source of truth); a
bad category costs one StartExecution and refuses at plan time. What the
gateway owns is IDENTITY: `workflow_id = uuid5(request + tool_version
[+ token])` via the copied `workflow_identity` module, the execution named
by it, and the family's absorbed-resubmit contract. `assets`/`spans`/
`question` are all in the hash: the same sweep absorbs, a new question is
new work.

The prompt-content caveat is conversation-analysis's, verbatim: category
prompts can be rebuilt pre-GA through the seeder backdoor and prompt bytes
are not in this hash — re-inspecting after a prompt rebuild needs a fresh
idempotency_token; provenance records what actually ran.
"""
from __future__ import annotations

import json
import logging
import os
import re

import boto3
from botocore.exceptions import ClientError

import callback_registry
from workflow_identity import (compute_workflow_id,
                               read_idempotency_token,
                               warn_idempotency_token)  # noqa: F401

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_ddb = boto3.client("dynamodb")
_s3 = boto3.client("s3")

STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "frame-inspection")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
# The callback-token table (README.md §11, copied shape):
# rows registered at submit, completed by callback_waker on the machine's
# terminal EventBridge event. Absent env = feature disabled loudly.
CALLBACK_TABLE = os.environ.get("CALLBACK_TABLE", "")

_TOOL_DELIMITER = "___"

_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$")


def _execution_arn(workflow_id: str) -> str:
    """The one place the execution ARN is derived from the machine ARN —
    the register-before-StartExecution invariant depends on the submit
    row and the absorbed-path completion computing the SAME key."""
    return (f"{STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
            f":{workflow_id}")

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
    # THE raise-vs-return choke point (review, 2026-09-05): a
    # waitForTaskToken caller never reads a returned error dict, so any
    # refusal that RETURNS while a callback token is present strands the
    # caller to its deadman. This wrapper is the ONE mechanism — every
    # present and FUTURE `return _error(...)` is converted here.
    #
    # The discriminator is `error is True` — the _error() REFUSAL shape
    # — never mere truthiness: an absorbed already-FAILED run carries
    # its cause as a STRING under the same key, as DATA beside a
    # job_id, AFTER complete_now already redeemed the caller's token
    # with that verdict. Raising on it would detonate the documented
    # "a FAILED verdict is DATA, never a detonation" path for a request
    # the gateway had already fulfilled (review, 2026-09-06).
    result = _dispatch(event or {}, context)
    token = (event or {}).get("callback_task_token")
    if token and isinstance(result, dict) and result.get("error") is True:
        raise RuntimeError(
            f"[{result.get('errorCode')}] {result.get('message')} — "
            "raised instead of returned because a waitForTaskToken "
            "caller never reads a return value")
    return result


def _dispatch(event: dict, context) -> dict:
    logger.info("Event keys: %s", sorted(event.keys()))
    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    if tool == "inspect_frames":
        return _inspect_frames(event)
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
            # capability probe for workflow consumers: a deployment that
            # predates callback_task_token would silently accept-and-
            # ignore the field, stranding the caller's token to its
            # deadman — consumers check this list before parking.
            # span_crop: per-span "crop" on title-read spans; consumers
            # strip the key when this is absent (an older build would
            # accept-and-ignore it and silently read full frames).
            "features": ((["callback_task_token"] if CALLBACK_TABLE
                          else []) + ["span_crop"]),
            "description": (
                "Frame-inspection micro-workflow ('the witness'): targeted, "
                "dense re-looks at archived shot-segmentation frames for "
                "spans of a programme — verbatim credits/title reading and "
                "visual claim verification. Requests pool across categories "
                "AND assets into one bedrock-inference batch job. Reads ONLY "
                "shot-segmentation's own artifacts (frame archives + "
                "embeddings); never the source video. Categories are "
                "prompt-registry prompts: credits-read, title-read (span "
                "auto-triggers over shot metadata) and claim-verify "
                "(explicit spans, each with a question)."),
        }
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# Tool: inspect_frames
# ---------------------------------------------------------------------------

def _inspect_frames(event: dict) -> dict:
    """
    Pooled form:
      assets             [{shot_segmentation_s3uri, shot_metadata_s3uri?,
                           requests: [{category, spans?}]}] — spans each
                          {start, end, id?, question?}; claim-verify spans
                          are explicit and each carries a question.
    Single-asset sugar:
      shot_segmentation_s3uri + category (+ shot_metadata_s3uri, spans)

    Optional:
      model_chain        Bedrock model ids, primary first (list or comma-
                         separated). Absent → the workflow's IMAGE-capable
                         default (a versioned constant).
      use_batch          Absent → TRUE: this workflow pools requests for
                         batch economics (catalog sweeps). Pass false for a
                         latency-sensitive claim-verify inside a waiting run.
      history_scope      Your OWN asset identity for the inference history
                         prefix. Absent → derived from the workflow_id.
      idempotency_token  Same token + same parameters returns the EXISTING
                         run. Also the RERUN mechanism after a category
                         prompt rebuild (prompt bytes are not in the hash).

    Returns a HANDLE — job_id, workflow_id, status, output_s3_prefix —
    never results; poll get_job_status.
    """
    if not PROXY_BUCKET or not STATE_MACHINE_ARN:
        return _error("PROXY_BUCKET and STATE_MACHINE_ARN env vars must be "
                      "set on this function", "CONFIG_ERROR")

    # ── callback_task_token: validated LOUDLY, before any submit ─────────
    # With a token present the caller is by definition a waitForTaskToken
    # workflow, and such a caller NEVER reads this function's return
    # value — so every failure on the callback path RAISES (failing the
    # caller's wait state fast) instead of returning an error dict nobody
    # will see. Deliberately NOT in the identity hash: two consumers (or
    # a redriven wait state) submitting identical work must absorb into
    # ONE run, each with its own registered wake.
    raw_token = event.get("callback_task_token")
    callback_token = raw_token.strip() if isinstance(raw_token, str) else None
    if raw_token is not None:
        if not callback_token or len(callback_token) > 2048:
            raise ValueError(
                "callback_task_token must be a non-empty Step Functions "
                "task token string (a waitForTaskToken caller cannot read "
                "a returned error, so this raises)")
        if not CALLBACK_TABLE:
            raise RuntimeError(
                "callback_task_token was supplied but CALLBACK_TABLE is "
                "not configured on this deployment")

    token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")

    request: dict = {}
    if event.get("assets"):
        if not isinstance(event["assets"], list):
            return _error("assets must be a list of "
                          "{shot_segmentation_s3uri, requests}",
                          "INVALID_PARAMETER")
        request["assets"] = event["assets"]
    elif (event.get("shot_segmentation_s3uri") or "").strip():
        request["shot_segmentation_s3uri"] = \
            event["shot_segmentation_s3uri"].strip()
        if not (event.get("category") or "").strip():
            return _error("category is required with the "
                          "single-asset form", "MISSING_PARAMETER")
        request["category"] = event["category"].strip()
        if event.get("shot_metadata_s3uri"):
            request["shot_metadata_s3uri"] = \
                event["shot_metadata_s3uri"].strip()
        if event.get("spans"):
            request["spans"] = event["spans"]
    else:
        return _error("either assets[] or shot_segmentation_s3uri + "
                      "category is required", "MISSING_PARAMETER")

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

    workflow_id = compute_workflow_id(
        request, tool_version=TOOL_VERSION,
        extra_material=(token,) if token else ())
    output_s3_prefix = (f"s3://{PROXY_BUCKET}/micro-workflow/"
                        f"{TARGET_NAME}/{workflow_id}")
    logger.info("inspect_frames: workflow_id=%s", workflow_id)

    # Register the callback BEFORE StartExecution (bedrock's rule): the
    # execution ARN is derivable from the name we are about to use, so
    # the row exists before the run can possibly finish — the waker can
    # never fire first. A stale row from a failed start is TTL-cleaned.
    execution_arn = _execution_arn(workflow_id)
    if callback_token:
        callback_registry.register(_ddb, CALLBACK_TABLE, execution_arn,
                                   callback_token, {
            "job_id": execution_arn,
            "workflow_id": workflow_id,
            "output_s3_prefix": output_s3_prefix,
        })

    try:
        resp = _sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN, name=workflow_id,
            input=json.dumps({"uuid": workflow_id, "input": request}))
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ExecutionAlreadyExists":
            return _existing_execution(workflow_id, output_s3_prefix,
                                       callback_token)
        message = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s", message)
        if callback_token:
            # the function's own contract: every failure on the callback
            # path RAISES — a token holder must fail fast and retry, never
            # sleep toward a deadman on a stranded token. Delete the row
            # registered above (no execution will ever emit its terminal
            # event); the TTL stays a backstop, not the mechanism.
            callback_registry.delete(_ddb, CALLBACK_TABLE, execution_arn,
                                     callback_token)
            raise
        return _error(f"Failed to start the inspection job: {message}",
                      "SFN_START_FAILED")

    return {"job_id": resp["executionArn"], "status": "RUNNING",
            "workflow_id": workflow_id,
            "output_s3_prefix": output_s3_prefix}


def _existing_execution(workflow_id: str, output_s3_prefix: str,
                        callback_token: str | None = None) -> dict:
    execution_arn = _execution_arn(workflow_id)
    try:
        existing = _sfn.describe_execution(executionArn=execution_arn)
    except ClientError as exc:
        if callback_token:
            # callback path RAISES, always — and the raise tells the
            # holder the submit failed, so it will consume its token
            # itself. Delete the row THIS call registered (StartExecution-
            # failure symmetry); leaving it points the waker at a token
            # that is already spent.
            callback_registry.delete(_ddb, CALLBACK_TABLE, execution_arn,
                                     callback_token)
            raise
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
    if callback_token and status != "RUNNING":
        # Absorbed into a run that already FINISHED: its terminal event
        # fired long ago, so no waker will ever see this token. Complete
        # it NOW — the paid-work-reuse path, and what makes a REDRIVE of
        # the consumer's wait state the reconcile. RUNNING absorbs need
        # nothing: the row was registered before StartExecution and the
        # terminal event is still coming.
        callback_registry.complete_now(_sfn, callback_token, result)
        callback_registry.delete(_ddb, CALLBACK_TABLE, execution_arn,
                                 callback_token)
    return result


# ---------------------------------------------------------------------------
# Tool: get_job_status
# ---------------------------------------------------------------------------

def _get_job_status(event: dict) -> dict:
    job_id = (event.get("job_id") or "").strip()
    if not job_id:
        return _error("job_id is required — the execution ARN returned by "
                      "inspect_frames", "MISSING_PARAMETER")
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

    # The product list is DYNAMIC (per asset × category), so outputs[] comes
    # from the manifest — best-effort: an unreadable manifest degrades the
    # poll to the two fixed entries, never fails it.
    outputs = []
    try:
        manifest = _read_json_if_exists(f"{prefix}/inspection_manifest.json")
        if manifest:
            # the SHAPE consumption stays inside the guard too: a
            # products[] entry missing a key (older-TOOL_VERSION manifest
            # at the same prefix) must degrade, not fail the poll
            result["products"] = [
                {"asset_idx": p["asset_idx"], "category": p["category"],
                 "spans": p["spans"], "gaps": p["gaps"]}
                for p in manifest.get("products", [])]
            outputs = [_output_entry(p["s3uri"], p["category"])
                       for p in manifest.get("products", [])]
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("manifest lift skipped: %s", exc)
        result.pop("products", None)
        outputs = []
    outputs.append(_output_entry(
        f"s3://{PROXY_BUCKET}/{prefix}/inspection_manifest.json", "manifest"))
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
