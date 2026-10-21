"""
AgentCore Gateway tool dispatcher for the face-recognition micro-workflow.

Three tools:
  recognize_faces  → StartExecution on the face-recognition state machine.
  get_job_status   → DescribeExecution; parses the workflow output when complete.
  describe_tool    → Returns the deployed tool version, state machine ARN,
                     a short backend description, and the output contract
                     (field-level example per outputs[] label, from
                     output_contract.json in this zip).

AgentCore passes the selected tool name in Lambda context:
  context.client_context.custom["bedrockAgentCoreToolName"]
  → "{target_name}___{tool_name}"   e.g. "face-recognition___recognize_faces"

The event object is a flat dict of that tool's inputSchema properties.

Recognition policy (GDPR use cases — see README.md §2b). Three orthogonal
toggles compose every use case:
  celebrity_recognition=false  no RecognizeCelebrities calls
  collection_id=""             no persistent-collection ops (search AND index)
  max_faces_to_index=0         auto-index off — THE DEFAULT since 2026-10-08
                               (persistent search still on; pass >0 to opt in)
  index_named_faces_only=true  auto-index only high-confidence-named faces
The transient per-execution sandbox collection is deleted at terminal state
and runs in ALL use cases.

workflow_id / idempotency strategy:
  workflow_id = uuid5(canonical JSON of the SFN input WITHOUT output.s3uri
  + "|" + TOOL_VERSION + "|" + the idempotency token).  ANY parameter, token,
  or tool-version change re-runs to a fresh prefix.  Outputs go to the
  service-derived prefix
  s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{workflow_id} — the
  caller-supplied output_s3_prefix is deprecated and ignored.

  `idempotency_token` (optional, 1-80 chars of A-Z a-z 0-9 _ . -) is the
  caller's assertion "this is the same request I sent before".  The sibling
  workflows dedup on parameters alone, because their `uuid` names the asset
  they process and the parameters fully determine the output.  This tool is
  not tied to one asset: it reads and writes a PERSISTENT, MUTABLE face
  collection, so identical parameters can legitimately need to run again
  ("recognize this video again now that the collection knows more faces").
  An ABSENT token therefore means "do not deduplicate me" — one is generated,
  and the call gets its own run.  A caller who wants retry safety supplies a
  stable token.

  The token rides in the hash preimage as extra material and is DISCARDED —
  never forwarded to the state machine, never stored, never echoed back.  A
  caller that supplied one already has it; the service answers with its own
  name for the run, `workflow_id`.  The retired `uuid` parameter is REJECTED
  here, not ignored: it used to mean both "which asset" and "same request", and
  a caller still sending it cannot be assumed to mean the surviving half.

  That rejection is scoped to `recognize_faces` — the only tool where either key
  changes what runs.  Every other tool on this target ACCEPTS both spellings and
  merely logs.  The caller is an LLM, so a hallucinated parameter name is a
  routine failure mode, and failing an otherwise-correct poll, read or sync
  mutation over a field that could not have altered the answer is the worse
  trade.  The index tools that DO start executions (`batch_index_faces`,
  `batch_delete_faces`, `delete_collection`) are in that ignoring set on
  purpose: they name their executions `uuid4()` because they are accumulating
  writes to shared state — two calls with different manifests must both run —
  and their dedup is content-addressed one layer down, on the source ETag, which
  already survives a re-run without the caller remembering a token.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid as _uuid_mod

from workflow_identity import (compute_workflow_id,
                               read_idempotency_token,
                               warn_idempotency_token)

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")

STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]
# Must equal the GatewayTarget Name in media-toolkit-gateway/cfnstack.yaml —
# it is the {target} segment of the derived output prefix.
TARGET_NAME = os.environ.get("TARGET_NAME", "")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
DEFAULT_COLLECTION_ID = os.environ.get("DEFAULT_COLLECTION_ID", "")

_TOOL_DELIMITER = "___"

_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$"
)


# The only signal that a submit was absorbed into an existing run rather than
# starting a new one.  Every field except `status` is identical to a fresh
# start's, so without this the agent cannot tell the two apart.  Wording is kept
# byte-identical across every micro-workflow gateway.
_ABSORBED_MESSAGE = (
    "identical request already submitted — this call did not resubmit; "
    "call get_job_status with this job_id for status and results"
)

# definitions.json StartAt — a fresh execution is always in this state, so the
# submit can report it without paying for a GetExecutionHistory call.
_FIRST_STATE = "Prepare media processing"

# outputs[] MIME vocabulary
_MIME_JSON = "application/json"

# ── Output contract ─────────────────────────────────────────────────────────
# output_contract.json ships in this zip (the build script pins the member) and
# is the single source for both surfaces that describe outputs: the per-label
# one-line `desc` joined into each outputs[] entry, and the full field-level
# contract (examples per label) returned by describe_tool.  Static per
# TOOL_VERSION, loaded once at cold start.  A missing or unparseable file must
# not take submit/status down with it, so it degrades to None with an ERROR
# log; the build script's member check is what makes that state unreachable in
# a real deployment.
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
# test round-trips the examples through this handler's own validation, so
# an example that drifts from the real contract fails the build. Same
# degradation contract as the output contract: never takes submit/status
# down with it.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "input_examples.json")) as _f:
        _INPUT_EXAMPLES = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("input_examples.json unavailable: %s", _exc)
    _INPUT_EXAMPLES = None


def _contract_desc(label: str) -> str | None:
    """The contract's one-line description for an outputs[] label."""
    if not _OUTPUT_CONTRACT:
        return None
    entry = (_OUTPUT_CONTRACT.get("outputs") or {}).get(label) or {}
    return entry.get("description")


def _output_entry(bucket, key, label, mime):
    """One outputs[] entry. Prefix entries (key ends with /) carry no mime.
    `desc` is joined from the output contract so the two surfaces cannot
    disagree; it is omitted, never fabricated, when the contract lacks the
    label."""
    entry = {"uri": f"s3://{bucket}/{key}", "label": label}
    if mime is not None:
        entry["mime"] = mime
    desc = _contract_desc(label)
    if desc:
        entry["desc"] = desc
    return entry


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    # keys only — the values can carry callback task tokens and
    # customer-named S3 uris, neither of which belongs in a log group
    logger.info("Event keys: %s", sorted(event.keys()))

    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    # `recognize_faces` is the only tool here whose execution a token names.
    # Everything else — polls, sync reads, sync mutations, and the three index
    # tools that DO start executions — accepts the field and IGNORES it.  The
    # index tools are in that set on purpose: they name executions `uuid4()`
    # because they are accumulating writes to shared state (two calls with
    # different manifests must both run) and dedup content-addressed on the
    # source ETag one layer down, which survives a re-run with no token at all.
    # Ignoring rather than rejecting is the deliberate trade: the caller is an
    # LLM, a hallucinated parameter name is a routine failure mode, and failing
    # an otherwise-correct request over a field that could not have changed the
    # answer is worse than a log line.  This runs before dispatch so a tool
    # added later is covered by default.
    if tool != "recognize_faces":
        _token_note = warn_idempotency_token(event)
        if _token_note:
            logger.warning("%s: %s", tool, _token_note)

    if tool == "recognize_faces":
        return _recognize_faces(event)

    if tool == "get_job_status":
        return _get_job_status(event)

    # Phase 2 — face index management (README.md §19)
    if tool in ("create_collection", "list_collections", "list_faces",
                "get_face", "batch_update_faces", "batch_index_faces",
                "batch_delete_faces", "delete_collection"):
        import index_tools
        return getattr(index_tools, tool)(event)

    if tool == "describe_tool":
        resp = {
            "version": TOOL_VERSION,
            "state_machine_arn": STATE_MACHINE_ARN,
            "description": ("Amazon Rekognition face recognition Step Functions "
                            "pipeline: probe-then-fill detection on frame grids, "
                            "temporal tracking, celebrity recognition, identity "
                            "clustering, persistent-collection search/index"),
        }
        # Field-level contract for every outputs[] label — what is INSIDE each
        # output file (compact example per label).  Static per TOOL_VERSION, so
        # a caller caches it for the session and CI can pin it without running
        # a job.  Deliberately here and not in get_job_status: examples are
        # per-version data and would otherwise be re-delivered with every job.
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _parse_int(event, key, min_val=None, max_val=None):
    raw = event.get(key)
    if raw is None:
        return None
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _error(f"{key} must be an integer", "INVALID_PARAMETER")
    if min_val is not None and val < min_val:
        return _error(f"{key} must be >= {min_val} (got {val})",
                      "INVALID_PARAMETER")
    if max_val is not None and val > max_val:
        return _error(f"{key} must be <= {max_val} (got {val})",
                      "INVALID_PARAMETER")
    return val


def _parse_float(event, key, min_val=None, max_val=None):
    raw = event.get(key)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _error(f"{key} must be a number", "INVALID_PARAMETER")
    if min_val is not None and val < min_val:
        return _error(f"{key} must be >= {min_val} (got {val})",
                      "INVALID_PARAMETER")
    if max_val is not None and val > max_val:
        return _error(f"{key} must be <= {max_val} (got {val})",
                      "INVALID_PARAMETER")
    return val


def _parse_bool(event, key):
    raw = event.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False
    return _error(f"{key} must be a boolean", "INVALID_PARAMETER")


# ---------------------------------------------------------------------------
# Tool: recognize_faces
# ---------------------------------------------------------------------------

def _recognize_faces(event: dict) -> dict:
    asset_s3uri = (event.get("asset_s3uri") or "").strip()
    if not asset_s3uri.startswith("s3://"):
        return _error("asset_s3uri is required and must be an s3:// uri",
                      "MISSING_PARAMETER")

    # Ignored, not rejected — same trade as the idempotency token at the
    # dispatcher.  It cannot be honoured: the prefix is derived from
    # workflow_id, which is a hash of this input, so a caller-chosen prefix
    # would have to be known before the id that produces it.  The response
    # returns the real prefix.
    if event.get("output_s3_prefix"):
        logger.warning("output_s3_prefix is deprecated and ignored — outputs "
                       "are written to the service-derived prefix, which this "
                       "response returns as output_s3_prefix")

    if not PROXY_BUCKET or not TARGET_NAME:
        return _error("service misconfigured: PROXY_BUCKET/TARGET_NAME unset",
                      "CONFIG_ERROR")

    # ── Optional parameters ───────────────────────────────────────────────────
    input_config: dict = {}

    fi = _parse_float(event, "frame_interval_sec", 0.2, 10.0)
    if isinstance(fi, dict):
        return fi
    if fi is not None:
        input_config["frame_interval_sec"] = fi

    mio = event.get("mark_in_out")
    if mio is not None:
        if (not isinstance(mio, (list, tuple)) or len(mio) != 2):
            return _error("mark_in_out must be [start_sec, end_sec]",
                          "INVALID_PARAMETER")
        try:
            start, end = float(mio[0]), float(mio[1])
        except (TypeError, ValueError):
            return _error("mark_in_out values must be numbers",
                          "INVALID_PARAMETER")
        if start < 0 or end <= start:
            return _error("mark_in_out requires 0 <= start < end",
                          "INVALID_PARAMETER")
        input_config["use_partial"] = True
        input_config["mark_in_out"] = [start, end]

    for key, parser, args in (
            ("probe_stride", _parse_int, (0, 30)),
            ("dedup_hamming_threshold", _parse_int, (0, 16)),
            ("max_faces_to_index", _parse_int, (0, None)),
            ("min_appearances", _parse_int, (1, None)),
            ("min_confidence", _parse_float, (0.0, 100.0)),
            ("max_concurrent_executions", _parse_int, (1, 50)),
            ("detect_map_concurrency", _parse_int, (1, 10)),
    ):
        val = parser(event, key, *args)
        if isinstance(val, dict):
            return val
        if val is not None:
            input_config[key] = val

    for key in ("celebrity_recognition", "index_named_faces_only"):
        val = _parse_bool(event, key)
        if isinstance(val, dict):
            return val
        if val is not None:
            input_config[key] = val

    collection_id = event.get("collection_id")
    if collection_id is not None:
        if not isinstance(collection_id, str):
            return _error("collection_id must be a string", "INVALID_PARAMETER")
        input_config["collection_id"] = collection_id.strip()
    elif DEFAULT_COLLECTION_ID:
        input_config["collection_id"] = DEFAULT_COLLECTION_ID

    # Optional shot-segmentation supplement (README.md §21): the shots.json
    # FILE uri a shot-segmentation job advertises as outputs[label=shot-list].
    # IAM only grants reads on the proxy bucket — reject foreign buckets
    # here with a usable message (same boundary as batch_index_faces).
    shots_uri = event.get("shot_segmentation_s3uri") or ""
    if not isinstance(shots_uri, str):
        return _error("shot_segmentation_s3uri must be a string",
                      "INVALID_PARAMETER")
    shots_uri = shots_uri.strip()
    if shots_uri:
        if not shots_uri.startswith(f"s3://{PROXY_BUCKET}/"):
            return _error(
                f"shot_segmentation_s3uri must be under s3://{PROXY_BUCKET}/ "
                f"— pass the shot-list output uri from a shot-segmentation "
                f"job (the stack only has read access to its proxy bucket)",
                "FOREIGN_BUCKET")
        input_config["shot_segmentation_s3uri"] = shots_uri

    if (input_config.get("index_named_faces_only")
            and input_config.get("max_faces_to_index") == 0):
        logger.warning("index_named_faces_only with max_faces_to_index=0 is "
                       "a no-op combination")

    # ── SFN input (without output.s3uri — the prefix derives from the
    #    workflow_id, so it cannot participate in the hash) ─────────────────────
    sfn_input: dict = {
        "input": {
            "s3uri": asset_s3uri,
            **({"input_config": input_config} if input_config else {}),
        },
        "output": {},
        "data": {},
    }

    # ── workflow_id ────────────────────────────────────────────────────────────
    # workflow_id is the WORK identity: uuid5 over the canonical SFN input
    # + tool version + the idempotency token.  It NAMES the execution, so
    # idempotency always reflects the actual parameters — the same token with
    # different parameters (or a new tool version) re-runs instead of returning
    # stale results.
    #
    # Unlike the sibling workflows, an ABSENT token here means "do not
    # deduplicate me" rather than "dedup on the parameters".  Those workflows
    # take a uuid naming the asset they process, so their parameters fully
    # determine their output.  This tool reads and writes a PERSISTENT, MUTABLE
    # face collection: "recognize this video again now that the collection knows
    # more faces" is a real request, and with parameter-only dedup it could not
    # be expressed without perturbing an unrelated parameter.  So a generated
    # token makes the hash unique and the run fresh.
    #
    # That is worth the cost it exposes: a fresh run burns a semaphore slot and
    # billed Rekognition calls.  A caller who cares supplies a token; a caller
    # who supplies none has asked for a new run.
    idempotency_token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")
    job_uuid = compute_workflow_id(
        sfn_input, tool_version=TOOL_VERSION, exclude=("output.s3uri",),
        extra_material=(idempotency_token or str(_uuid_mod.uuid4()),))

    output_s3_prefix = (
        f"s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{job_uuid}"
    )
    sfn_input["output"]["s3uri"] = output_s3_prefix

    # The event `uuid` is this execution's LOCAL id, and here it MUST equal the
    # execution name: in-workflow states reconstruct the execution ARN (semaphore
    # holder key) and the sandbox-{workflow_id} collection name from
    # STATE_MACHINE_ARN + event uuid.  The token rides in the hash preimage only,
    # so nothing carries it into the state machine.
    sfn_input["uuid"] = job_uuid

    logger.info("recognize_faces: uuid=%s asset=%s output=%s",
                job_uuid, asset_s3uri, output_s3_prefix)

    try:
        resp = _sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=job_uuid,
            input=json.dumps(sfn_input),
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ExecutionAlreadyExists":
            # Idempotent resubmit: the execution name is the full work-hash,
            # so a collision implies an identical request.
            execution_arn = (
                f"{STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
                f":{job_uuid}"
            )
            logger.info("Execution already exists: %s", execution_arn)
            try:
                existing = _sfn.describe_execution(executionArn=execution_arn)
            except ClientError as desc_exc:
                desc_msg = desc_exc.response["Error"]["Message"]
                return _error(
                    f"Job already exists but could not retrieve status: "
                    f"{desc_msg}", "SFN_DESCRIBE_FAILED")
            # An absorbed resubmit returns a HANDLE, exactly like a fresh
            # start — only `status` varies, plus `message` to say the call did
            # not resubmit.  It deliberately does NOT return the existing run's
            # results, even when that run already SUCCEEDED: results come from
            # get_job_status and only from there, so a submit's reachable field
            # set equals its declared field set by construction and the agent
            # has one rule to follow instead of two.
            actual_status = existing.get("status", "RUNNING")
            result = {
                "job_id": execution_arn,
                "status": actual_status,
                "workflow_id": job_uuid,
                "asset_s3uri": _extract_asset_s3uri(existing) or asset_s3uri,
                "output_s3_prefix": (_extract_output_s3_prefix(existing)
                                     or output_s3_prefix),
                "message": _ABSORBED_MESSAGE,
            }
            if actual_status == "RUNNING":
                # current_state costs a GetExecutionHistory paging pass, and is
                # meaningless once the execution has stopped.
                result["current_state"] = _get_current_state(execution_arn)
            elif actual_status != "SUCCEEDED":
                # The agent needs the reason to choose between "change a
                # parameter" and "resubmit with a different idempotency_token" —
                # a terminal failure is sticky for as long as Step Functions
                # reserves the execution name, so a plain retry cannot clear it.
                # Same compaction as get_job_status: the stable code and
                # the evidence as fields, never a raw stack-trace blob —
                # the absorbed resubmit is the one response an agent gets
                # when it DOES resubmit, so it must carry error_code too.
                cause = (existing.get("cause") or existing.get("error")
                         or actual_status)
                error_code, error_message = _split_cause(cause)
                detail = ": ".join(
                    x for x in (error_code, error_message) if x)
                result["error"] = detail or str(cause)[:600]
                _set(result, "error_code",
                     error_code or existing.get("error") or None)
            return result
        msg = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s — %s", code, msg)
        return _error(f"Failed to start face recognition job: {msg}",
                      "SFN_START_FAILED")

    return {
        "job_id": resp["executionArn"],
        "status": "RUNNING",
        "current_state": _FIRST_STATE,
        "workflow_id": job_uuid,
        "asset_s3uri": asset_s3uri,
        "output_s3_prefix": output_s3_prefix,
    }


# ---------------------------------------------------------------------------
# Tool: get_job_status
# ---------------------------------------------------------------------------

def _get_job_status(event: dict) -> dict:
    job_id = (event.get("job_id") or "").strip()
    if not job_id:
        return _error("job_id is required", "MISSING_PARAMETER")
    if not _SFN_ARN_RE.match(job_id):
        return _error("job_id must be a Step Functions execution ARN",
                      "INVALID_JOB_ID")

    try:
        resp = _sfn.describe_execution(executionArn=job_id)
    except _sfn.exceptions.ExecutionDoesNotExist:
        return _error(f"No execution found: {job_id}", "JOB_NOT_FOUND")
    except ClientError as exc:
        msg = exc.response["Error"]["Message"]
        return _error(f"Failed to describe job: {msg}", "SFN_DESCRIBE_FAILED")

    sfn_status = resp["status"]
    asset_s3uri = _extract_asset_s3uri(resp)

    if sfn_status == "RUNNING":
        result = {"job_id": job_id,
                  "status": "RUNNING",
                  "current_state": _get_current_state(job_id)}
        # Omitted rather than null for an index-management execution, which has
        # no source asset.
        _set(result, "asset_s3uri", asset_s3uri)
        _attach_workflow_id(result, resp)
        _set(result, "output_s3_prefix", _extract_output_s3_prefix(resp))
        return result

    if sfn_status in ("FAILED", "TIMED_OUT", "ABORTED"):
        cause = resp.get("cause") or resp.get("error") or sfn_status
        # the cause of a Lambda-raised failure is an error JSON whose
        # errorType IS the stable code (NoVideoStream, ...) and whose
        # errorMessage carries the evidence — surface them as fields so
        # an agent branches on error_code instead of substring-scanning
        # a stack-trace blob; the raw blob stays in `error` only when
        # the cause is not that shape
        error_code, error_message = _split_cause(cause)
        detail = ": ".join(x for x in (error_code, error_message) if x)
        result = {"job_id": job_id, "status": sfn_status,
                  "error": detail or str(cause)[:600]}
        # when the cause is not Lambda-error JSON (plain text, or absent),
        # DescribeExecution's own error field still carries a stable code
        # (Lambda.Unknown, States.Timeout, ...) — don't discard it
        _set(result, "error_code", error_code or resp.get("error") or None)
        _set(result, "asset_s3uri", asset_s3uri)
        _attach_workflow_id(result, resp)
        # Present on a failed run too: partial outputs are often written before
        # the failure, and an absorbed resubmit reports this prefix, so a poll
        # that omitted it would contradict the submit it followed.
        _set(result, "output_s3_prefix", _extract_output_s3_prefix(resp))
        return result

    if sfn_status == "SUCCEEDED":
        # Index-management executions (face-index SM) return a different
        # output shape than the recognition pipeline.
        if ":execution:" in job_id and _is_index_execution(job_id):
            result = _parse_index_output(resp)
        else:
            result = _parse_succeeded_output(resp)
        result["job_id"] = job_id
        return result

    # Any status Step Functions adds later (e.g. PENDING_REDRIVE) lands here.
    # Still name the run, so the agent can report and re-poll it.
    result = {"job_id": job_id, "status": sfn_status}
    _set(result, "asset_s3uri", asset_s3uri)
    _attach_workflow_id(result, resp)
    _set(result, "output_s3_prefix", _extract_output_s3_prefix(resp))
    return result


def _is_index_execution(job_id: str) -> bool:
    sm_name = job_id.split(":execution:", 1)[1].split(":", 1)[0]
    index_arn = os.environ.get("INDEX_STATE_MACHINE_ARN", "")
    return bool(index_arn) and sm_name == index_arn.rsplit(":", 1)[-1]


def _parse_index_output(resp: dict) -> dict:
    """SUCCEEDED output of the face-index SM (batch_index / batch_delete /
    delete_collection)."""
    result: dict = {"status": "SUCCEEDED"}
    try:
        payload = json.loads(resp.get("output") or "{}")
        data = payload.get("data") or {}
        _set(result, "action", payload.get("action"))
        _set(result, "workflow_id",
             (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)
        _set(result, "collection_id",
             (payload.get("input") or {}).get("collection_id"))
        out_s3uri = (payload.get("output") or {}).get("s3uri") or ""
        _set(result, "output_s3_prefix",
             out_s3uri or _extract_output_s3_prefix(resp))
        for k in ("num_indexed", "num_skipped", "num_failed", "num_deleted",
                  "num_manifest_entries", "usage"):
            _set(result, k, data.get(k))
        if data.get("report_key") and out_s3uri.startswith("s3://"):
            bucket = out_s3uri[5:].partition("/")[0]
            result["report_uri"] = f"s3://{bucket}/{data['report_key']}"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not parse index execution output: %s", exc)
    return result


def _split_cause(cause) -> tuple:
    """(errorType, errorMessage) from a Lambda error-JSON cause, or
    ("", "") when the cause is not that shape."""
    try:
        parsed = json.loads(cause)
    except (TypeError, ValueError):
        return "", ""
    if not isinstance(parsed, dict):
        return "", ""
    # a real errorType is a class name; the [:100] cap (matching MP's
    # waker _compact_cause) keeps a pathological cause out of error_code
    return (str(parsed.get("errorType") or "")[:100],
            str(parsed.get("errorMessage") or "")[:400])


def _parse_succeeded_output(resp: dict) -> dict:
    """
    The final state (Usage tracking) returns the full event; surface:
      identity, output location, people counts, cost/timing, outputs[].
    """
    result: dict = {"status": "SUCCEEDED"}
    try:
        payload = json.loads(resp.get("output") or "{}")
        data = payload.get("data") or {}

        # ── Identity ───────────────────────────────────────────────────────
        _set(result, "workflow_id",
             (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)
        _set(result, "asset_s3uri", _extract_asset_s3uri(resp))

        # ── Output location ───────────────────────────────────────────────────
        # Falls back to the execution input, which always carries the prefix —
        # the submit promised this field, so a poll must not drop it.
        out_s3uri = (payload.get("output") or {}).get("s3uri") or ""
        _set(result, "output_s3_prefix",
             out_s3uri or _extract_output_s3_prefix(resp))
        bucket = prefix = None
        if out_s3uri.startswith("s3://"):
            bucket, _, prefix = out_s3uri[5:].partition("/")
            prefix = prefix.rstrip("/")

        # ── Content facts ─────────────────────────────────────────────────────
        _set(result, "num_people_recognized",
             data.get("num_people_recognized"))
        _set(result, "num_people_unrecognized",
             data.get("num_people_unrecognized"))
        _set(result, "num_faces_indexed", data.get("num_faces_indexed"))
        cfg = (payload.get("input") or {}).get("input_config") or {}
        _set(result, "collection_id", cfg.get("collection_id"))
        mp = data.get("media_processing_output") or {}
        _set(result, "duration_sec", mp.get("duration_sec"))
        _set(result, "media_processing_arn", mp.get("media_processing_arn"))
        meta = data.get("frame_meta") or {}
        _set(result, "frames_analyzed", meta.get("kept_frames"))
        _set(result, "frames_skipped", meta.get("triaged_out"))

        # ── Cost / timing ─────────────────────────────────────────────────────
        usage = data.get("usage_tracking") or {}
        _set(result, "pipeline_wall_sec", usage.get("pipeline_wall_sec"))
        _set(result, "total_estimated_cost",
             usage.get("total_estimated_cost"))
        _set(result, "usage_key", usage.get("usage_key"))

        # ── Self-describing outputs[] ─────────────────────────────────────────
        # Uniform across all gateway tools: {uri, mime, label, desc}.  Entries
        # whose uri ends with "/" are prefixes (many objects) and carry no mime.
        # `desc` is the one-line role description joined from
        # output_contract.json; the field-level contract (what is inside each
        # file) lives in describe_tool's output_contract, not here.
        # Labels and basenames are API — renames are breaking changes.
        if bucket and prefix:
            outputs = []
            for key, label, mime in (
                    (data.get("facemetadata_key"), "face-metadata", _MIME_JSON),
                    (data.get("facegroup_key"), "face-groups", _MIME_JSON),
                    (data.get("face_clusters_key"), "face-clusters", _MIME_JSON),
                    (usage.get("usage_key"), "usage", _MIME_JSON),
            ):
                if key:
                    outputs.append(_output_entry(bucket, key, label, mime))
            if data.get("debug_people_prefix"):
                outputs.append(_output_entry(
                    bucket, data["debug_people_prefix"].rstrip("/") + "/",
                    "people-images", None))
            outputs.append(_output_entry(
                bucket, f"{prefix}/grids/", "debug-grids", None))
            if outputs:
                result["outputs"] = outputs

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not parse execution output: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach_workflow_id(result: dict, resp: dict) -> None:
    """Attach workflow_id — the execution name, i.e. the ARN's last segment.

    It no longer reads the execution input: the caller's token is not echoed, so
    there is nothing else in there to surface. Reading it from the OUTPUT is
    where the predecessor bug lived — no state carries the key, so a SUCCEEDED
    poll silently answered with the derived id while the submit answered with
    the caller's.
    """
    _set(result, "workflow_id",
         (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)


def _extract_asset_s3uri(resp: dict):
    try:
        sfn_input = json.loads(resp.get("input") or "{}")
        return sfn_input.get("input", {}).get("s3uri") or None
    except (json.JSONDecodeError, AttributeError):
        return None


def _extract_output_s3_prefix(resp: dict) -> str | None:
    """
    The service-derived output prefix, read back from the execution's own input.

    Read from the input rather than the output so it is available on every
    branch — RUNNING and failed executions have no output at all, and the prefix
    is where partial results already live.
    """
    try:
        sfn_input = json.loads(resp.get("input") or "{}")
        return sfn_input.get("output", {}).get("s3uri") or None
    except (json.JSONDecodeError, AttributeError):
        return None


def _get_current_state(execution_arn: str) -> str:
    """
    Active pipeline stage for a RUNNING execution: count entered/exited per
    state; active = entered > exited; lowest entry count wins so the outer
    stage beats Map iterator sub-states.
    """
    try:
        entered: dict[str, int] = {}
        exited: dict[str, int] = {}
        # Names/types only — skip per-event payloads (tens of MB on long runs).
        kwargs: dict = {"executionArn": execution_arn, "maxResults": 1000,
                        "includeExecutionData": False}
        while True:
            page = _sfn.get_execution_history(**kwargs)
            for e in page["events"]:
                if e["type"] in ("TaskStateEntered", "MapStateEntered"):
                    name = e["stateEnteredEventDetails"]["name"]
                    entered[name] = entered.get(name, 0) + 1
                elif e["type"] in ("TaskStateExited", "MapStateExited"):
                    name = e["stateExitedEventDetails"]["name"]
                    exited[name] = exited.get(name, 0) + 1
            token = page.get("nextToken")
            if not token:
                break
            kwargs["nextToken"] = token
        active = [n for n in entered if entered.get(n, 0) > exited.get(n, 0)]
        if not active:
            return "unknown"
        return min(active, key=lambda n: entered[n])
    except Exception as exc:
        logger.error("Could not determine current state: %s", exc)
        return "unknown"


def _resolve_tool(context) -> str:
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        if _TOOL_DELIMITER in raw:
            return raw.split(_TOOL_DELIMITER, 1)[1]
        return raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("No bedrockAgentCoreToolName in client context")
        return ""


def _set(d: dict, key: str, value) -> None:
    if value is not None:
        d[key] = value


def _error(message: str, code: str) -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
