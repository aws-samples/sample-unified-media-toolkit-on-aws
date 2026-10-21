"""
AgentCore Gateway tool dispatcher for the speech-diarization micro-workflow.

Three tools:
  diarize_speech   → StartExecution on the speech-diarization state machine.
  get_job_status   → DescribeExecution; parses the workflow output when complete.
  describe_tool    → Returns the deployed tool version, state machine ARN,
                     a short backend description, and the output contract
                     (field-level example per outputs[] label, from
                     output_contract.json in this zip).

AgentCore passes the selected tool name in Lambda context:
  context.client_context.custom["bedrockAgentCoreToolName"]
  → "{target_name}___{tool_name}"   e.g. "speech-diarization___diarize_speech"

The event object is a flat dict of that tool's inputSchema properties.

SFN input contract (speech-diarization state machine):
  {
    "uuid":   str,
    "input":  {
      "s3uri": str,
      "input_config": {
        "language":    str|null,
        "use_partial": bool,
        "mark_in_out": [float, float],
        "vad":         {"min_speech_duration_sec": float, ...},
        "asr":         {"beam_size": int, "initial_prompt": str, ...},
        "embed":       {"min_embedding_duration_sec": float},
        "clustering":  {"num_speakers": int|null, "max_k": int, ...}
      }
    },
    "output": {
      "s3uri": str,
      "output_config": {"speaker_channels": bool, "channel_mode": str}
    },
    "data": {}
  }

SFN final output (returned by StateUsageTracking, which is End=true):
  The full event dict is returned as-is.  Fields surfaced to the agent:
    Correlation
      event.uuid                                        ← job UUID
      event.execution_arn                               ← SFN execution ARN
                                                          (written by the state,
                                                          not surfaced — `job_id`
                                                          already is this value)
    Output location
      event.output.s3uri                                ← prefix for all outputs
    Primary output S3 keys
      event.data.final_transcript_key                   ← transcript JSON
      event.data.final_transcript_vtt                   ← WebVTT
      event.data.speaker_manifest_key                   ← per-speaker turn list
      event.data.global_speakers.global_speakers_key    ← clustering detail
    Content facts
      event.data.language                               ← detected/forced BCP-47 code
      event.data.language_confidence                    ← float 0–1
      event.data.num_sentences                          ← int
      event.data.audio_cfg.duration_sec                 ← float seconds
    Speaker results
      event.data.num_global_speakers                    ← int
      event.data.global_speakers.clustering_method      ← e.g. "ahc(k_enum  k=28)"
      event.data.global_speakers.single_speaker_detected← bool
      event.data.speaker_channels_status                ← "COMPLETED" | "SKIPPED" | …
      event.data.speaker_channels_num_files             ← int
    Cost / timing
      event.data.usage_tracking.pipeline_wall_sec       ← float seconds
      event.data.usage_tracking.total_estimated_cost    ← float USD
      event.data.usage_tracking.usage_key               ← S3 key with full cost breakdown
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

  That rejection is scoped to diarize_speech — the only tool where either
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

DIARIZATION_STATE_MACHINE_ARN = os.environ["DIARIZATION_STATE_MACHINE_ARN"]
# Must equal the GatewayTarget Name in media-toolkit-gateway/cfnstack.yaml —
# it is the {target} segment of the derived output prefix.
TARGET_NAME = os.environ.get("TARGET_NAME", "")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")

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
_MIME_VTT  = "text/vtt"

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

_VALID_CHANNEL_MODES = frozenset(("aligned", "compact"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    # keys only — the values can carry callback task tokens and
    # customer-named S3 uris, neither of which belongs in a log group
    logger.info("Event keys: %s", sorted(event.keys()))

    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    # `diarize_speech` is the only tool here whose execution a token names.
    # `get_job_status` and `describe_tool` accept the field and IGNORE it — a
    # poll reflects whatever the execution is doing now, and no token could have
    # changed that.  Ignoring rather than rejecting is the deliberate trade: the
    # caller is an LLM, a hallucinated parameter name is a routine failure mode,
    # and failing an otherwise-correct poll over a discarded field is worse than
    # a log line.  This runs before dispatch so a tool added later is covered by
    # default.
    if tool != "diarize_speech":
        _token_note = warn_idempotency_token(event)
        if _token_note:
            logger.warning("%s: %s", tool, _token_note)

    if tool == "diarize_speech":
        return _diarize_speech(event)

    if tool == "get_job_status":
        return _get_job_status(event)

    if tool == "describe_tool":
        resp = {
            "version": TOOL_VERSION,
            "state_machine_arn": DIARIZATION_STATE_MACHINE_ARN,
            "description": ("Whisper ASR + speaker clustering diarization Step Functions pipeline: "
                            "word-level transcripts, WebVTT captions, per-speaker audio"),
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
# Parameter parsing helpers
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


def _parse_float(event: dict, key: str, min_val: float | None = None, max_val: float | None = None):
    """
    Return the float value for key, None if absent, or an error dict if invalid.
    Caller checks: if isinstance(result, dict): return result
    """
    raw = event.get(key)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _error(f"{key} must be a number", "INVALID_PARAMETER")
    if min_val is not None and val < min_val:
        return _error(f"{key} must be >= {min_val} (got {val})", "INVALID_PARAMETER")
    if max_val is not None and val > max_val:
        return _error(f"{key} must be <= {max_val} (got {val})", "INVALID_PARAMETER")
    return val


# ---------------------------------------------------------------------------
# Tool: diarize_speech
# ---------------------------------------------------------------------------

def _diarize_speech(event: dict) -> dict:
    """
    Validate inputs and start a diarization execution.

    Required:
      asset_s3uri      S3 URI of the source media file

    Deprecated:
      output_s3_prefix  Ignored — outputs are written to the service-derived
                        prefix (returned as output_s3_prefix in the response)

    Optional:
      idempotency_token             Optional idempotency token — assert that this request is a
                                    resend of a previous one. Omit it and the parameters
                                    alone determine the run. Consumed into workflow_id and
                                    discarded; never echoed back.
      language                      BCP-47 code e.g. "en", "ar", "cy"; null = auto-detect
      mark_in_out                   [start_sec, end_sec] — clip window; sets use_partial=True
                                    automatically; window must be > 10 s
      min_speech_duration_sec       VAD: minimum speech segment duration (default 0.0 — short runs are admitted and grouped; they never enter clustering)
      min_silence_duration_sec      VAD: minimum silence gap to split segments (default 0.3)
      speech_group_max_gap_sec      VAD: merge adjacent turns within this gap (default 1.0)
      beam_size                     ASR: Whisper beam size 1–5 (default 2; higher = slower + better)
      initial_prompt                ASR: context hint passed to Whisper (e.g. domain vocabulary)
      merge_gap_threshold_sec       ASR: merge short inter-sentence gaps below this (default 0.20)
      min_embedding_duration_sec    Embed: minimum sentence duration for a full embedding (default 0.75)
      num_speakers                  Clustering: known speaker count; null = auto-detect
      max_k                         Clustering: upper bound on speaker count search (default 50)
      min_cluster_size              Clustering: minimum embeddings per cluster, ≥ 2 (default 2)
      min_turns_per_speaker         Clustering: minimum turns to keep a speaker, ≥ 2 (default 3)
      outlier_global_percentile     Clustering: remove top (100-x)% outlier turns (default 97.0)
      apply_snorm                   Clustering: apply score normalisation (default false)
      speaker_channels              Output: write per-speaker WAV files (default true)
      channel_mode                  Output: "aligned" or "compact" (default "aligned")

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

    # ── mark_in_out → use_partial ──────────────────────────────────────────────
    mark_in_out = event.get("mark_in_out")
    use_partial  = False
    if mark_in_out is not None:
        if (not isinstance(mark_in_out, list) or len(mark_in_out) != 2):
            return _error("mark_in_out must be [start_sec, end_sec]", "INVALID_PARAMETER")
        try:
            mark_in_out = [float(mark_in_out[0]), float(mark_in_out[1])]
        except (TypeError, ValueError):
            return _error("mark_in_out values must be numbers", "INVALID_PARAMETER")
        if (mark_in_out[1] - mark_in_out[0]) <= 10:
            return _error("mark_in_out window must be greater than 10 seconds", "INVALID_PARAMETER")
        use_partial = True

    # ── int params ─────────────────────────────────────────────────────────────
    err = _parse_int(event, "num_speakers",          min_val=1)
    if isinstance(err, dict): return err
    num_speakers = err

    err = _parse_int(event, "max_k",                 min_val=1)
    if isinstance(err, dict): return err
    max_k = err

    err = _parse_int(event, "min_cluster_size",      min_val=2)
    if isinstance(err, dict): return err
    min_cluster_size = err

    err = _parse_int(event, "min_turns_per_speaker", min_val=2)
    if isinstance(err, dict): return err
    min_turns_per_speaker = err

    err = _parse_int(event, "beam_size",             min_val=1, max_val=5)
    if isinstance(err, dict): return err
    beam_size = err

    # ── float params ───────────────────────────────────────────────────────────
    err = _parse_float(event, "min_speech_duration_sec",   min_val=0.0)
    if isinstance(err, dict): return err
    min_speech_duration_sec = err

    err = _parse_float(event, "min_silence_duration_sec",  min_val=0.0)
    if isinstance(err, dict): return err
    min_silence_duration_sec = err

    err = _parse_float(event, "speech_group_max_gap_sec",  min_val=0.0)
    if isinstance(err, dict): return err
    speech_group_max_gap_sec = err

    err = _parse_float(event, "merge_gap_threshold_sec",   min_val=0.0)
    if isinstance(err, dict): return err
    merge_gap_threshold_sec = err

    err = _parse_float(event, "min_embedding_duration_sec", min_val=0.0)
    if isinstance(err, dict): return err
    min_embedding_duration_sec = err

    err = _parse_float(event, "outlier_global_percentile", min_val=0.0, max_val=100.0)
    if isinstance(err, dict): return err
    outlier_global_percentile = err

    # ── string / bool params ───────────────────────────────────────────────────
    language        = event.get("language") or None
    initial_prompt  = event.get("initial_prompt") or None
    apply_snorm     = event.get("apply_snorm")
    if apply_snorm is not None:
        apply_snorm = bool(apply_snorm)

    speaker_channels = event.get("speaker_channels")
    if speaker_channels is not None:
        speaker_channels = bool(speaker_channels)

    channel_mode = event.get("channel_mode") or None
    if channel_mode is not None and channel_mode not in _VALID_CHANNEL_MODES:
        return _error(
            f"channel_mode must be one of: {sorted(_VALID_CHANNEL_MODES)}",
            "INVALID_PARAMETER",
        )

    # ── Build nested SFN input (no uuid yet) ───────────────────────────────────
    vad: dict = {}
    if min_speech_duration_sec  is not None: vad["min_speech_duration_sec"]  = min_speech_duration_sec
    if min_silence_duration_sec is not None: vad["min_silence_duration_sec"] = min_silence_duration_sec
    if speech_group_max_gap_sec is not None: vad["speech_group_max_gap_sec"] = speech_group_max_gap_sec

    asr: dict = {}
    if beam_size              is not None: asr["beam_size"]               = beam_size
    if initial_prompt         is not None: asr["initial_prompt"]          = initial_prompt
    if merge_gap_threshold_sec is not None: asr["merge_gap_threshold_sec"] = merge_gap_threshold_sec

    embed: dict = {}
    if min_embedding_duration_sec is not None: embed["min_embedding_duration_sec"] = min_embedding_duration_sec

    clustering: dict = {}
    if num_speakers           is not None: clustering["num_speakers"]           = num_speakers
    if max_k                  is not None: clustering["max_k"]                  = max_k
    if min_cluster_size       is not None: clustering["min_cluster_size"]       = min_cluster_size
    if min_turns_per_speaker  is not None: clustering["min_turns_per_speaker"]  = min_turns_per_speaker
    if outlier_global_percentile is not None: clustering["outlier_global_percentile"] = outlier_global_percentile
    if apply_snorm            is not None: clustering["apply_snorm"]            = apply_snorm

    input_config: dict = {}
    if language    is not None: input_config["language"]    = language
    if use_partial:
        input_config["use_partial"]  = True
        input_config["mark_in_out"]  = mark_in_out
    if vad:         input_config["vad"]         = vad
    if asr:         input_config["asr"]         = asr
    if embed:       input_config["embed"]       = embed
    if clustering:  input_config["clustering"]  = clustering

    output_config: dict = {}
    if speaker_channels is not None: output_config["speaker_channels"] = speaker_channels
    if channel_mode     is not None: output_config["channel_mode"]     = channel_mode

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
        "diarize_speech: workflow_id=%s asset=%s output=%s",
        job_uuid, asset_s3uri, output_s3_prefix,
    )

    try:
        resp = _sfn.start_execution(
            stateMachineArn=DIARIZATION_STATE_MACHINE_ARN,
            name=job_uuid,
            input=json.dumps(sfn_input),
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ExecutionAlreadyExists":
            # Idempotent resubmit: same asset+config+version → return the
            # existing run's status/results instead of failing.
            execution_arn = (
                f"{DIARIZATION_STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
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
        return _error(f"Failed to start diarization job: {msg}", "SFN_START_FAILED")

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
    Poll a diarization execution by its ARN and return structured output.

    Required:
      job_id  Step Functions execution ARN returned by diarize_speech

    Ignored if sent:
      idempotency_token  and the retired `uuid` — a poll starts no execution, so
                         neither can change what this returns.  Accepted and
                         logged rather than rejected, so a hallucinated
                         parameter does not fail an otherwise-correct poll.

    Returns on RUNNING:
      {"job_id": str, "workflow_id": str, "status": "RUNNING",
       "current_state": str, "asset_s3uri": str, "output_s3_prefix": str}

    Returns on SUCCEEDED:
      {
        "status":                    "SUCCEEDED",
        "job_id":                    str,
        "workflow_id":               str,
        "asset_s3uri":              str,
        "output_s3_prefix":          str,
        "transcript_json_key":       str,
        "transcript_vtt_key":        str,
        "speaker_manifest_key":      str,
        "global_speakers_key":       str,
        "language":                  str,
        "language_confidence":       float,
        "num_sentences":             int,
        "duration_sec":              float,
        "num_speakers":              int,
        "clustering_method":         str,
        "single_speaker_detected":   bool,
        "speaker_channels_status":   str,
        "speaker_channels_num_files":int,
        "pipeline_wall_sec":         float,
        "total_estimated_cost":            float,
        "usage_key":                 str,
        "media_processing_arn":      str   (absent if no conversion)
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
        # errorType IS the stable code (NoAudioStream, NoSpeechDetected,
        # ...) and whose errorMessage carries the evidence — surface them
        # as fields so an agent branches on error_code instead of
        # substring-scanning a stack-trace blob; the raw blob stays in
        # `error` only when the cause is not that shape
        error_code, error_message = _split_cause(cause)
        # prefer the parsed pair in whatever combination survived: a
        # cause may carry a message without a type (SFN-generated) or a
        # type without a message (an exception raised with empty args)
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

    # Guard for any future SFN status values (e.g. PENDING_REDRIVE).
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
    States where entered_count > exited_count are still active.  When inside a
    Map, the outer state (e.g. "Map VAD processing") has entered_count==1 while
    each parallel iterator sub-state has entered_count==N.  Returning the state
    with the lowest entry count surfaces the outer pipeline stage, which is more
    useful to the agent than an internal iterator name.
    """
    try:
        events: list[dict] = []
        kwargs: dict = {"executionArn": execution_arn, "maxResults": 1000}
        while True:
            resp = _sfn.get_execution_history(**kwargs)
            events.extend(resp["events"])
            token = resp.get("nextToken")
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
      - Identifiers (workflow_id; `job_id` is attached by the caller)
      - Output location (output_s3_prefix)
      - Primary output S3 keys (transcript, vtt, speaker manifest, global speakers)
      - Content facts (language, confidence, duration, sentence count)
      - Speaker results (num_speakers, clustering_method, single_speaker_detected,
                         speaker_channels_status, speaker_channels_num_files)
      - Cost / timing (pipeline_wall_sec, total_estimated_cost, usage_key)
      - Child execution ARN when media conversion was performed

    Internal plumbing (*_cfg, model_manifest, iterators, intermediate S3 keys,
    language_probe_results, usage_tracking.states) is excluded — it lives on S3
    in usage_key and final_transcript_key for anything that needs it.
    """
    result: dict = {"status": "SUCCEEDED"}
    try:
        payload = json.loads(resp.get("output") or "{}")
        data    = payload.get("data") or {}
        out     = payload.get("output") or {}

        # ── Correlation ───────────────────────────────────────────────────────
        # `execution_arn` used to be echoed here from the usage-tracking state's
        # $.Execution.Id.  It was removed: it is byte-identical to the `job_id`
        # the caller passed in, and `job_id` is now on every branch, so the echo
        # only gave the same value a second name.  `media_processing_arn` below
        # stays — that one names a DIFFERENT (child) execution.
        _set(result, "workflow_id",
             (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)
        _set(result, "asset_s3uri",  _extract_asset_s3uri(resp))

        # ── Output location ───────────────────────────────────────────────────
        # Falls back to the execution input, which always carries the prefix —
        # the submit promised this field, so a poll must not drop it.
        _set(result, "output_s3_prefix",
             (out.get("s3uri") or "").rstrip("/") or _extract_output_s3_prefix(resp))

        # ── Primary output keys ───────────────────────────────────────────────
        _set(result, "transcript_json_key",   data.get("final_transcript_key"))
        _set(result, "transcript_vtt_key",    data.get("final_transcript_vtt"))
        _set(result, "speaker_manifest_key",  data.get("speaker_manifest_key"))
        _set(result, "global_speakers_key",
             (data.get("global_speakers") or {}).get("global_speakers_key"))

        # ── Content facts ─────────────────────────────────────────────────────
        _set(result, "language",             data.get("language"))
        _set(result, "language_confidence",  data.get("language_confidence"))
        _set(result, "num_sentences",        data.get("num_sentences"))

        duration = (data.get("audio_cfg") or {}).get("duration_sec")
        _set(result, "duration_sec", duration)

        # ── Speaker results ───────────────────────────────────────────────────
        _set(result, "num_speakers", data.get("num_global_speakers"))

        gs = data.get("global_speakers") or {}
        _set(result, "clustering_method",       gs.get("clustering_method"))
        _set(result, "single_speaker_detected", gs.get("single_speaker_detected"))

        _set(result, "speaker_channels_status",    data.get("speaker_channels_status"))
        _set(result, "speaker_channels_num_files", data.get("speaker_channels_num_files"))

        # ── Cost / timing ─────────────────────────────────────────────────────
        usage = data.get("usage_tracking") or {}
        _set(result, "pipeline_wall_sec", usage.get("pipeline_wall_sec"))
        _set(result, "total_estimated_cost",    usage.get("total_estimated_cost"))
        _set(result, "usage_key",         usage.get("usage_key"))

        # ── Child execution (media conversion path) ───────────────────────────
        _set(result, "media_processing_arn", data.get("media_processing_arn"))

        # ── Self-describing outputs[] ─────────────────────────────────────────
        # Uniform across all gateway tools: {uri, mime, label, desc}.  Entries
        # whose uri ends with "/" are prefixes (many objects) and carry no mime.
        # `desc` is the one-line role description joined from
        # output_contract.json; the field-level contract (what is inside each
        # file) lives in describe_tool's output_contract, not here.
        # Per-speaker WAVs are NOT enumerated (speaker IDs can be
        # non-contiguous) — one speakers/ prefix entry + the manifest lets the
        # agent resolve exact keys.  Labels and basenames are API — renames are
        # breaking changes.
        output_bucket = data.get("output_bucket")
        if output_bucket:
            outputs = []
            for key, label, mime in (
                (data.get("final_transcript_key"),  "transcript",       _MIME_JSON),
                (data.get("final_transcript_vtt"),  "caption",          _MIME_VTT),
                (data.get("speaker_manifest_key"),  "speaker-manifest", _MIME_JSON),
                ((data.get("global_speakers") or {}).get("global_speakers_key"),
                                                    "global-speakers",  _MIME_JSON),
                (usage.get("usage_key"),            "usage",            _MIME_JSON),
            ):
                if key:
                    outputs.append(_output_entry(output_bucket, key, label, mime))
            output_prefix = data.get("output_prefix")
            if (output_prefix
                    and data.get("speaker_channels_status") == "COMPLETED"
                    and (data.get("speaker_channels_num_files") or 0) > 0):
                outputs.append(_output_entry(
                    output_bucket,
                    f"{output_prefix.strip('/')}/speakers/",
                    "speaker-audio", None))
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
