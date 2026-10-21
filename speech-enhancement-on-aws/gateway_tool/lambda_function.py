"""
AgentCore Gateway tool dispatcher for the speech-enhancement micro-workflow.

Three tools:
  enhance_speech  → StartExecution on the speech-enhancement state machine.
  get_job_status        → DescribeExecution; parses the workflow output when complete.
  describe_tool         → Returns the deployed tool version, state machine ARN,
                          a short backend description, and the output contract
                          (field-level example per outputs[] label, from
                          output_contract.json in this zip).

AgentCore passes the selected tool name in Lambda context:
  context.client_context.custom["bedrockAgentCoreToolName"]
  → "{target_name}___{tool_name}"   e.g. "speech-enhancement___enhance_speech"

The event object is a flat dict of that tool's inputSchema properties.

SFN input contract (speech-enhancement state machine):
  {
    "uuid":   str,
    "input":  {
      "s3uri": str,
      "input_config": {
        "fixed_chunk_size_secs": int   (≥ 10; aligned to nearest 10 s multiple)
      }
    },
    "output": {
      "s3uri": str,
      "output_config": {
        "output_resampl_freq": int,
        "atten_lim_db":        float | null
      }
    },
    "data": {}
  }

  Note: model_id is always "deepfilternet3" — not user-configurable.

SFN final output (returned by StateUsageTracking, which is End=true):
  The full event dict is returned as-is.  Fields surfaced to the agent:
    Correlation
      event.uuid                                          ← job UUID
    Output S3 keys
      event.data.enhancement_output.enhanced_key   ← enhanced audio WAV
      event.data.enhancement_output.metrics_key    ← speech_enhancement.json
    Cost / timing
      event.data.usage_tracking.pipeline_wall_sec         ← float seconds
      event.data.usage_tracking.total_estimated_cost      ← float USD
      event.data.usage_tracking.usage_key                 ← S3 key of full cost breakdown
    Child execution (media conversion path, absent otherwise)
      event.data.media_processing_arn

workflow_id / idempotency strategy:
  workflow_id = uuid5(canonical JSON of the SFN input WITHOUT output.s3uri
  + "|" + TOOL_VERSION + "|" + idempotency_token when one is supplied).
  It NAMES the execution, so identical requests are idempotent (a resubmit
  returns the existing run's status and results via ExecutionAlreadyExists)
  while ANY parameter, token, or tool-version change re-runs to a fresh prefix.

  `idempotency_token` (optional, 1-80 chars of A-Z a-z 0-9 _ . -) is the
  caller's assertion "this is the same request I sent before".  It is consumed
  into the hash and DISCARDED — never forwarded to the state machine, never
  stored, never echoed back.  A caller that supplied one already has it; the
  service answers with its own name for the run, `workflow_id`.  The retired
  `uuid` parameter is REJECTED here, not ignored: it used to mean both "which
  asset" and "same request", and a caller still sending it cannot be assumed to
  mean the surviving half.

  That rejection is scoped to enhance_speech — the only tool where either
  key changes what runs.  get_job_status and describe_tool ACCEPT both spellings
  and merely log, because the caller is an LLM: a hallucinated parameter should
  not fail a poll that is otherwise correct, and neither key could have altered
  the answer anyway.

  Outputs are written to the service-derived prefix
  s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{workflow_id} —
  the caller-supplied output_s3_prefix is deprecated and ignored.
"""

from __future__ import annotations

import json
import logging
import os
import re

from workflow_identity import (compute_workflow_id,
                               read_idempotency_token,
                               warn_idempotency_token)

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")

SPEECH_ENHANCEMENT_STATE_MACHINE_ARN = os.environ["SPEECH_ENHANCEMENT_STATE_MACHINE_ARN"]
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
# Must equal the GatewayTarget Name in media-toolkit-gateway/cfnstack.yaml —
# it is the {target} segment of the derived output prefix.
TARGET_NAME  = os.environ.get("TARGET_NAME", "")

_TOOL_DELIMITER = "___"

# Execution ARN pattern — used to reject obviously invalid job_ids early.
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
_FIRST_STATE = "Check and preprocess"


# outputs[] MIME vocabulary — exact registered types only; must match the
# ContentType set at upload where one is set
_MIME_JSON = "application/json"
_MIME_WAV  = "audio/wav"

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

_VALID_RESAMPLE_FREQS = frozenset((8000, 16000, 32000, 44100, 48000, 96000))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    # keys only — the values can carry callback task tokens and
    # customer-named S3 uris, neither of which belongs in a log group
    logger.info("Event keys: %s", sorted(event.keys()))

    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    # `enhance_speech` is the only tool here whose execution a token names.
    # `get_job_status` and `describe_tool` accept the field and IGNORE it — a
    # poll reflects whatever the execution is doing now, and no token could have
    # changed that.  Ignoring rather than rejecting is the deliberate trade: the
    # caller is an LLM, a hallucinated parameter name is a routine failure mode,
    # and failing an otherwise-correct poll over a discarded field is worse than
    # a log line.  This runs before dispatch so a tool added later is covered by
    # default.
    if tool != "enhance_speech":
        _token_note = warn_idempotency_token(event)
        if _token_note:
            logger.warning("%s: %s", tool, _token_note)

    if tool == "enhance_speech":
        return _enhance_speech(event)

    if tool == "get_job_status":
        return _get_job_status(event)

    if tool == "describe_tool":
        resp = {
            "version": TOOL_VERSION,
            "state_machine_arn": SPEECH_ENHANCEMENT_STATE_MACHINE_ARN,
            "description": ("DeepFilterNet3 speech-enhancement Step Functions "
                            "pipeline: noise-suppresses the dialogue in any "
                            "media file's audio, in place (no stem "
                            "extraction). A listening deliverable: the "
                            "current model measured worse than the original "
                            "mix as a transcription input, so the mix is the "
                            "validated ASR/diarization default and the "
                            "enhanced track is not recommended there today"),
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
# Parameter parsing helper
# ---------------------------------------------------------------------------

def _parse_int(event: dict, key: str, min_val: int | None = None, max_val: int | None = None):
    """
    Return the int value for key, None if absent, or an error dict if invalid.
    Caller checks: if isinstance(result, dict): return result
    """
    raw = event.get(key)
    if raw is None:
        return None
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _error(f"{key} must be an integer", "INVALID_PARAMETER")
    if min_val is not None and val < min_val:
        return _error(f"{key} must be >= {min_val} (got {val})", "INVALID_PARAMETER")
    if max_val is not None and val > max_val:
        return _error(f"{key} must be <= {max_val} (got {val})", "INVALID_PARAMETER")
    return val


# ---------------------------------------------------------------------------
# Tool: enhance_speech
# ---------------------------------------------------------------------------

def _enhance_speech(event: dict) -> dict:
    """
    Validate inputs and start a speech-enhancement execution.

    Required:
      asset_s3uri      S3 URI of the source media file

    Deprecated:
      output_s3_prefix  Ignored — outputs are written to the service-derived
                        prefix (returned as output_s3_prefix in the response)

    Optional — input:
      idempotency_token      Optional idempotency token — assert that this request is a
                             resend of a previous one. Omit it and the parameters
                             alone determine the run. Consumed into workflow_id and
                             discarded; never echoed back.
      fixed_chunk_size_secs  Processing chunk size in seconds (minimum 10; default 360).
                             Aligned internally to the nearest 10 s multiple (DFN3's
                             internal window). Smaller chunks allow more parallelism but
                             add per-chunk overhead. Default 360 s is suitable for most
                             content.

    Optional — output:
      output_resampl_freq    Enhanced WAV sample rate in Hz (default 16000).
                             16000 is kept as the default because a quiet
                             request's hash omits the field — changing the
                             default would hand old-rate results to quiet
                             resubmits under the SAME workflow_id, unless a
                             TOOL_VERSION bump re-ran everything. Pass
                             48000 explicitly for the product's primary
                             uses — human listening or mixing. (As an
                             ASR/diarization input the enhanced track is
                             not recommended today — see the tool
                             description.)
      atten_lim_db           Cap on noise attenuation in dB, enforced as a
                             full-band mix of the original back into the
                             enhanced output (never attenuates more than
                             this many dB). ~0 = bypass, >=100 = unlimited;
                             omit for the model default (full suppression).
      Valid sample rates: 8000, 16000, 32000, 44100, 48000, 96000.

    Returns a HANDLE, never results:
      {"job_id", "workflow_id", "status", "current_state", "asset_s3uri",
       "output_s3_prefix"}, plus "message" when absorbed and "error" when the
       absorbed run failed terminally.

    An absorbed resubmit (same derived workflow_id as an existing run) returns
    this same shape with `status` set to the existing run's, plus `message`.  It
    never returns that run's results even when it already SUCCEEDED: results come
    from get_job_status and only from there.
    """
    # ── Required fields ────────────────────────────────────────────────────────
    asset_s3uri = (event.get("asset_s3uri") or "").strip()
    if not asset_s3uri:
        return _error("asset_s3uri is required", "MISSING_PARAMETER")
    if not asset_s3uri.startswith("s3://"):
        return _error("asset_s3uri must be an S3 URI (s3://bucket/key)", "INVALID_PARAMETER")

    # ── Deprecated output_s3_prefix: outputs are service-owned ─────────────────
    # Ignored, not rejected — same trade as the idempotency token at the
    # dispatcher.  It CANNOT be honoured: the prefix is derived from
    # workflow_id, and workflow_id is a hash of this input, so a
    # caller-chosen prefix would have to be known before the id that
    # produces it.  The response carries the real prefix, so a caller who
    # reads it is never left looking at an address that does not exist.
    if (event.get("output_s3_prefix") or "").strip():
        logger.warning(
            "output_s3_prefix is deprecated and ignored; outputs are written "
            "to the service-derived prefix returned as output_s3_prefix")

    if not PROXY_BUCKET or not TARGET_NAME:
        return _error(
            "PROXY_BUCKET and TARGET_NAME env vars must be set on this function",
            "CONFIG_ERROR",
        )

    # ── fixed_chunk_size_secs ──────────────────────────────────────────────────
    err = _parse_int(event, "fixed_chunk_size_secs", min_val=10)
    if isinstance(err, dict): return err
    fixed_chunk_size_secs = err

    # ── output sample rate ─────────────────────────────────────────────────────
    err = _parse_int(event, "output_resampl_freq")
    if isinstance(err, dict): return err
    output_resampl_freq = err
    if output_resampl_freq is not None and \
            output_resampl_freq not in _VALID_RESAMPLE_FREQS:
        return _error(
            f"output_resampl_freq must be one of: "
            f"{sorted(_VALID_RESAMPLE_FREQS)}",
            "INVALID_PARAMETER",
        )

    # ── attenuation limit (cap on noise reduction; null = full) ────────────────
    atten_lim_db = event.get("atten_lim_db")
    if atten_lim_db is not None:
        if not isinstance(atten_lim_db, (int, float)) \
                or isinstance(atten_lim_db, bool) \
                or not (0.0 <= float(atten_lim_db) <= 100.0):
            return _error(
                "atten_lim_db must be a number in [0, 100] or omitted",
                "INVALID_PARAMETER",
            )
        atten_lim_db = float(atten_lim_db)

    # ── Build nested SFN input (no uuid yet) ───────────────────────────────────
    input_config: dict = {}
    if fixed_chunk_size_secs is not None:
        input_config["fixed_chunk_size_secs"] = fixed_chunk_size_secs

    output_config: dict = {}
    if output_resampl_freq is not None:
        output_config["output_resampl_freq"] = output_resampl_freq
    if atten_lim_db is not None:
        output_config["atten_lim_db"] = atten_lim_db

    # ── SFN input (without output.s3uri — the prefix derives from the
    #    workflow_id, so it cannot participate in the hash) ─────────────────────
    sfn_input: dict = {
        "input":  {
            "s3uri": asset_s3uri,
            **({"input_config": input_config} if input_config else {}),
        },
        "output": {
            **({"output_config": output_config} if output_config else {}),
        },
        "data": {},
    }

    # ── workflow_id ────────────────────────────────────────────────────────────
    # workflow_id is the WORK identity: uuid5 over the canonical SFN input
    # (minus the derived output prefix) + tool version + the optional token.
    # It NAMES the execution, so idempotency always reflects the actual
    # parameters: the same token with different parameters re-runs rather than
    # returning another asset's results.
    #
    # The token rides in extra_material, never in sfn_input, so "consumed into
    # the hash and discarded" holds by construction — no state sees it, nothing
    # stores it, and no response echoes it back to a caller that already has it.
    idempotency_token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")
    job_uuid = compute_workflow_id(
        sfn_input, tool_version=TOOL_VERSION, exclude=("output.s3uri",),
        extra_material=(idempotency_token,) if idempotency_token else ())

    # The event `uuid` is this execution's LOCAL id, which every state-side
    # input_schema validator requires. It equals workflow_id here; it is not the
    # caller's token and is not renamed to workflow_id, because elsewhere in the
    # pipeline the same key legitimately carries a random child tag.
    sfn_input["uuid"] = job_uuid
    output_s3_prefix = (
        f"s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{job_uuid}"
    )
    sfn_input["output"]["s3uri"] = output_s3_prefix

    logger.info(
        "enhance_speech: workflow_id=%s asset=%s output=%s",
        job_uuid, asset_s3uri, output_s3_prefix,
    )

    try:
        resp = _sfn.start_execution(
            stateMachineArn=SPEECH_ENHANCEMENT_STATE_MACHINE_ARN,
            name=job_uuid,
            input=json.dumps(sfn_input),
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ExecutionAlreadyExists":
            # Idempotent resubmit: same asset+config+version → return the
            # existing run's status/results instead of failing.
            execution_arn = (
                f"{SPEECH_ENHANCEMENT_STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
                f":{job_uuid}"
            )
            logger.info("Execution already exists for workflow_id=%s: %s",
                        job_uuid, execution_arn)
            try:
                existing = _sfn.describe_execution(executionArn=execution_arn)
            except ClientError as desc_exc:
                desc_msg = desc_exc.response["Error"]["Message"]
                logger.exception("DescribeExecution failed for existing job: %s", desc_msg)
                return _error(
                    f"Job already exists but could not retrieve status: {desc_msg}",
                    "SFN_DESCRIBE_FAILED")
            # An absorbed resubmit returns a HANDLE, exactly like a fresh start —
            # only `status` varies, plus `message` to say the call did not
            # resubmit.  It deliberately does NOT return the existing run's
            # results, even when that run already SUCCEEDED: results come from
            # get_job_status and only from there, so a submit's reachable field
            # set equals its declared field set by construction and the agent has
            # one rule to follow instead of two.
            actual_status = existing.get("status", "RUNNING")
            # Echo the existing execution's asset/prefix.  The execution
            # name is the full work-hash, so a name collision implies an
            # identical request — these match the new request's values.
            result = {
                "job_id":           execution_arn,
                "status":           actual_status,
                "workflow_id":      job_uuid,
                "asset_s3uri":     _extract_asset_s3uri(existing) or asset_s3uri,
                "output_s3_prefix": (_extract_output_s3_prefix(existing)
                                     or output_s3_prefix),
                "message":          _ABSORBED_MESSAGE,
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
        return _error(f"Failed to start speech enhancement job: {msg}", "SFN_START_FAILED")

    job_id = resp["executionArn"]
    logger.info("Started execution: %s", job_id)
    return {
        "job_id":           job_id,
        "status":           "RUNNING",
        "current_state":    _FIRST_STATE,
        "workflow_id":      job_uuid,
        "asset_s3uri":     asset_s3uri,
        "output_s3_prefix": output_s3_prefix,
    }


# ---------------------------------------------------------------------------
# Tool: get_job_status
# ---------------------------------------------------------------------------

def _get_job_status(event: dict) -> dict:
    """
    Poll a speech-enhancement execution by its ARN and return structured output.

    Required:
      job_id  Step Functions execution ARN returned by enhance_speech

    Ignored if sent:
      idempotency_token  and the retired `uuid` — a poll starts no execution, so
                         neither can change what this returns.  Accepted and
                         logged rather than rejected, so a hallucinated
                         parameter does not fail an otherwise-correct poll.

    Every branch returns `job_id` and `workflow_id`, so one response is enough to
    name the run in a later call or a report.

    Returns on RUNNING:
      {"job_id": str, "workflow_id": str, "status": "RUNNING",
       "current_state": str, "asset_s3uri": str, "output_s3_prefix": str}

    Returns on SUCCEEDED:
      {
        "status":              "SUCCEEDED",
        "job_id":              str,
        "workflow_id":         str,
        "asset_s3uri":        str,
        "output_s3_prefix":    str,
        "enhanced_key":        str,   (the enhanced audio WAV)
        "metrics_key":         str,   (speech_enhancement.json — always present)
        "pipeline_wall_sec":   float,
        "total_estimated_cost":      float,
        "usage_key":           str,
        "media_processing_arn":str    (absent if no conversion was required)
      }
      All fields are omitted when absent in the workflow output.

    Returns on FAILED / TIMED_OUT / ABORTED:
      {"job_id": str, "workflow_id": str, "status": <status>,
       "error": <cause>, "asset_s3uri": str, "output_s3_prefix": str}
    """
    job_id = (event.get("job_id") or "").strip()
    if not job_id:
        return _error("job_id is required", "MISSING_PARAMETER")

    if not _SFN_ARN_RE.match(job_id):
        return _error(
            "job_id must be a Step Functions execution ARN",
            "INVALID_JOB_ID",
        )

    try:
        resp = _sfn.describe_execution(executionArn=job_id)
    except _sfn.exceptions.ExecutionDoesNotExist:
        return _error(f"No execution found: {job_id}", "JOB_NOT_FOUND")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        msg  = exc.response["Error"]["Message"]
        logger.exception("DescribeExecution failed: %s — %s", code, msg)
        return _error(f"Failed to describe job: {msg}", "SFN_DESCRIBE_FAILED")

    sfn_status = resp["status"]  # RUNNING | SUCCEEDED | FAILED | TIMED_OUT | ABORTED
    asset_s3uri = _extract_asset_s3uri(resp)

    if sfn_status == "RUNNING":
        result = {
            "job_id":        job_id,
            "status":        "RUNNING",
            "current_state": _get_current_state(job_id),
            "asset_s3uri":  asset_s3uri,
        }
        _attach_workflow_id(result, resp)
        _set(result, "output_s3_prefix", _extract_output_s3_prefix(resp))
        return result

    if sfn_status in ("FAILED", "TIMED_OUT", "ABORTED"):
        cause = resp.get("cause") or resp.get("error") or sfn_status
        # the cause of a Lambda-raised failure is an error JSON whose
        # errorType IS the stable code (NoAudioStream, ...) and whose
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


def _attach_workflow_id(result: dict, resp: dict) -> None:
    """Attach workflow_id — the execution name, i.e. the ARN's last segment.

    It no longer reads the execution input: the caller's token is not echoed, so
    there is nothing else in there to surface. Reading the input was where the
    predecessor bug lived (it looked in the OUTPUT for a key no state writes).
    """
    _set(result, "workflow_id",
         (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)


def _get_current_state(execution_arn: str) -> str:
    """
    Return the name of the currently active pipeline state for a RUNNING execution.

    Strategy: count TaskStateEntered and TaskStateExited events per state name.
    States where entered_count > exited_count are still active. When inside a Map,
    the outer state has entered_count==1 while each iterator sub-state has
    entered_count==N. Returning the state with the lowest entry count surfaces the
    outer pipeline stage, which is more useful to the agent.
    """
    try:
        events: list[dict] = []
        kwargs: dict = {"executionArn": execution_arn, "maxResults": 1000}
        while True:
            page = _sfn.get_execution_history(**kwargs)
            events.extend(page["events"])
            token = page.get("nextToken")
            if not token:
                break
            kwargs["nextToken"] = token

        entered:     dict[str, int] = {}
        exited:      dict[str, int] = {}
        entry_order: list[str]      = []

        for e in events:
            if e["type"] == "TaskStateEntered":
                name = e["stateEnteredEventDetails"]["name"]
                if name not in entered:
                    entry_order.append(name)
                entered[name] = entered.get(name, 0) + 1
            elif e["type"] == "TaskStateExited":
                name = e["stateExitedEventDetails"]["name"]
                exited[name] = exited.get(name, 0) + 1

        active = [n for n in entry_order if entered.get(n, 0) > exited.get(n, 0)]
        if not active:
            return "unknown"
        return min(active, key=lambda n: entered[n])

    except Exception as exc:
        logger.warning("Could not determine current state: %s", exc)
        return "unknown"


def _extract_asset_s3uri(resp: dict) -> str | None:
    """Extract the original asset_s3uri from the SFN execution input."""
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
    Extract structured results from a SUCCEEDED execution.

    StateUsageTracking is the final state and returns the full event unchanged,
    so resp["output"] is the complete event dict with all data.* fields populated.

    We surface only what an agent needs to act on or report:
      - Identifier (workflow_id)
      - Enhanced WAV key and metrics key (from data.enhancement_output)
      - Cost / timing (pipeline_wall_sec, total_estimated_cost, usage_key)
      - Child execution ARN when media conversion was performed

    Internal plumbing (audio_cfg, model_id, fixed_chunk_size_secs, iterators,
    audio_bucket, audio_key, usage_tracking.states, media_processing_input/output)
    is excluded.
    """
    result: dict = {"status": "SUCCEEDED"}
    try:
        payload = json.loads(resp.get("output") or "{}")
        data    = payload.get("data") or {}

        # ── Correlation ───────────────────────────────────────────────────────
        _set(result, "workflow_id",
             (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)
        _set(result, "asset_s3uri", _extract_asset_s3uri(resp))
        _set(result, "output_s3_prefix", _extract_output_s3_prefix(resp))

        # ── Enhancement output keys ──────────────────────────────────────────
        enh = data.get("enhancement_output") or {}
        _set(result, "enhanced_key", enh.get("enhanced_key"))
        _set(result, "metrics_key", enh.get("metrics_key"))
        _set(result, "resampl_freq", enh.get("resampl_freq"))
        _set(result, "rms_db_in", enh.get("rms_db_in"))
        _set(result, "rms_db_out", enh.get("rms_db_out"))
        _set(result, "duration_sec", enh.get("duration_sec"))

        # ── Cost / timing ─────────────────────────────────────────────────────
        usage = data.get("usage_tracking") or {}
        _set(result, "pipeline_wall_sec", usage.get("pipeline_wall_sec"))
        _set(result, "total_estimated_cost",    usage.get("total_estimated_cost"))
        _set(result, "usage_key",         usage.get("usage_key"))

        # ── Child execution (media conversion path) ───────────────────────────
        _set(result, "media_processing_arn", data.get("media_processing_arn"))

        # ── Self-describing outputs[] ─────────────────────────────────────────
        # Uniform across all gateway tools: {uri, mime, label, desc}.  `desc`
        # is the one-line role description joined from output_contract.json;
        # the field-level contract (what is inside each file) lives in
        # describe_tool's output_contract, not here.
        # Labels and basenames are API — renames are breaking changes.
        output_bucket = data.get("output_bucket")
        if output_bucket:
            outputs = []
            for key, label, mime in (
                (enh.get("enhanced_key"), "enhanced", _MIME_WAV),
                (enh.get("metrics_key"),  "metrics",  _MIME_JSON),
                (usage.get("usage_key"),  "usage",    _MIME_JSON),
            ):
                if key:
                    outputs.append(_output_entry(output_bucket, key, label, mime))
            if outputs:
                result["outputs"] = outputs

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not parse execution output: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_tool(context) -> str:
    """
    Extract the tool name from the AgentCore client context.
    The raw value is "{target_name}___{tool_name}"; strip the prefix.
    """
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        if _TOOL_DELIMITER in raw:
            return raw.split(_TOOL_DELIMITER, 1)[1]
        return raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("bedrockAgentCoreToolName not found in Lambda context")
        return ""


def _set(d: dict, key: str, value) -> None:
    """Set key in d only when value is not None."""
    if value is not None:
        d[key] = value


def _error(message: str, code: str = "ERROR") -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
