"""
AgentCore Gateway tool dispatcher for the media-processing micro-workflow.

Three tools:
  process_media   → StartExecution on the media-processing state machine.
  get_job_status  → DescribeExecution; parses the workflow output when complete.
  describe_tool   → Returns the deployed tool version, state machine ARN,
                    a short backend description, and the output contract
                    (field-level example per output FORMAT KIND — this stack's
                    outputs[] labels are runtime group names — from
                    output_contract.json in this zip).

AgentCore passes the selected tool name in Lambda context:
  context.client_context.custom["bedrockAgentCoreToolName"]
  → "{target_name}___{tool_name}"   e.g. "media-processing___process_media"

The event object is a flat dict of that tool's inputSchema properties.

SFN input contract (media-processing state machine):
  {
    "uuid":   str,
    "input":  {
      "s3uri": str,
      "input_config": {
        "use_channel_mapping":  bool,
        "audio_channel_mapping": [int, int],
        "use_partial":           bool,
        "mark_in_out":           [float, float],
        "crop_x":  int,
        "crop_y":  int,
        "width":   int,
        "height":  int
      }
    },
    "output": {
      "s3uri": str,
      "output_groups": [
        {"group_name": str, "format": str, ...}
      ]
    },
    "data": {}
  }

SFN final output (returned by StateUsageTracking, which is End=true):
  The full event dict is returned as-is.  Fields surfaced to the agent:
    Correlation
      event.uuid                                      ← job UUID
    Content facts
      event.data.duration_sec                         ← source duration in seconds
      event.data.has_video                            ← bool
      event.data.has_audio                            ← bool
      event.data.is_all_intra                         ← bool; None for ambiguous codecs
    Output S3 keys (one per output group)
      event.data.output_keys                          ← {group_name: s3_key}
                                                         frame groups:    s3_key = frame_manifest.json
                                                         metadata groups: s3_key = metadata.json
    Probe
      event.data.probe_s3_key                         ← S3 key of probe.json
    Cost / timing
      event.data.usage_tracking.pipeline_wall_sec     ← float seconds
      event.data.usage_tracking.total_estimated_cost  ← float USD
      event.data.usage_tracking.usage_key             ← S3 key of full cost breakdown

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

  That rejection is scoped to process_media — the only tool where either
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

import callback_registry

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_ddb = boto3.client("dynamodb")

MEDIA_PROCESSING_STATE_MACHINE_ARN = os.environ["MEDIA_PROCESSING_STATE_MACHINE_ARN"]
# Must equal the GatewayTarget Name in media-toolkit-gateway/cfnstack.yaml —
# it is the {target} segment of the derived output prefix.
TARGET_NAME = os.environ.get("TARGET_NAME", "")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
# The callback-token table (family pattern, 2026-09-18): rows registered
# at submit (callback_registry — the row format's one home), completed by
# callback_waker at the execution's terminal event.
CALLBACK_TABLE = os.environ.get("CALLBACK_TABLE", "")

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
_FIRST_STATE = "Probe media file"


# outputs[] MIME vocabulary — exact registered types only; must match the
# ContentType set at upload where one is set
_MIME_JSON = "application/json"
_EXT_MIME = {
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _mime_for_key(key):
    dot = key.rfind(".")
    return _EXT_MIME.get(key[dot:].lower()) if dot >= 0 else None


# ── Output contract ─────────────────────────────────────────────────────────
# output_contract.json ships in this zip (the build script pins the member) and
# is the single source for both surfaces that describe outputs: the per-entry
# one-line `desc` joined into each outputs[] entry, and the full field-level
# contract (examples per format kind) returned by describe_tool.  Static per
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


def _contract_desc(kind):
    """The contract's one-line description for an output FORMAT KIND.

    This stack's outputs[] labels are the caller-chosen group names, so the
    contract cannot key on labels; its keys are the format kinds (`mp4`,
    `wav`, `m4a`, `jpeg`, `png`, `metadata`) plus the fixed `probe`/`usage`.
    """
    if not _OUTPUT_CONTRACT or not kind:
        return None
    entry = (_OUTPUT_CONTRACT.get("outputs") or {}).get(kind) or {}
    return entry.get("description")


def _output_entry(bucket, key, label, mime, kind=None):
    """One outputs[] entry. Prefix entries (key ends with /) carry no mime.
    `desc` is joined from the output contract by the entry's format KIND —
    the label is a runtime group name the contract cannot enumerate — so the
    two surfaces cannot disagree; it is omitted, never fabricated, when the
    contract lacks the kind."""
    entry = {"uri": f"s3://{bucket}/{key}", "label": label}
    if mime is not None:
        entry["mime"] = mime
    desc = _contract_desc(kind)
    if desc:
        entry["desc"] = desc
    return entry

_VALID_FORMATS     = frozenset(("mp4", "wav", "m4a", "jpeg", "png",
                                "metadata", "ad_mix"))
_VALID_MP4_CODECS  = frozenset(("libopenh264", "openh264"))
_VALID_WAV_CODECS  = frozenset(("pcm_s16le",))
_VALID_M4A_CODECS  = frozenset(("aac",))
_VALID_DETAIL      = frozenset(("default", "full"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    # keys only — the values can carry callback task tokens and
    # customer-named S3 uris, neither of which belongs in a log group
    logger.info("Event keys: %s", sorted(event.keys()))

    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    # `process_media` is the only tool here whose execution a token names.
    # `get_job_status` and `describe_tool` accept the field and IGNORE it — a
    # poll reflects whatever the execution is doing now, and no token could have
    # changed that.  Ignoring rather than rejecting is the deliberate trade: the
    # caller is an LLM, a hallucinated parameter name is a routine failure mode,
    # and failing an otherwise-correct poll over a discarded field is worse than
    # a log line.  This runs before dispatch so a tool added later is covered by
    # default.
    if tool not in ("process_media", "quick_edit", "capture_frames"):
        _token_note = warn_idempotency_token(event)
        if _token_note:
            logger.warning("%s: %s", tool, _token_note)

    _SUBMIT_TOOLS = {"process_media": _process_media,
                     "quick_edit": _quick_edit,
                     "capture_frames": _capture_frames}
    if tool in _SUBMIT_TOOLS:
        result = _SUBMIT_TOOLS[tool](event)
        # raise-vs-return choke point (fi/bedrock convention): a
        # waitForTaskToken caller never reads a returned error dict, so
        # any refusal that RETURNS while a callback token is present
        # would strand the consumer until its deadman. The discriminator
        # is `error is True` — the _error() refusal shape. Applies to all
        # three submit tools, which is what lets quick_edit/capture_frames
        # validation keep returning named refusals for the polling caller.
        if (isinstance(result, dict) and result.get("error") is True
                and (event or {}).get("callback_task_token")):
            raise RuntimeError(
                f"[{result.get('errorCode')}] {result.get('message')}")
        return result

    if tool == "get_job_status":
        return _get_job_status(event)

    if tool == "describe_tool":
        resp = {
            "version": TOOL_VERSION,
            "state_machine_arn": MEDIA_PROCESSING_STATE_MACHINE_ARN,
            "description": ("ffmpeg media processing Step Functions pipeline: "
                            "proxy transcode, audio extraction, frame captures, "
                            "technical metadata (ffprobe + mxf2raw on MXF)"),
            # consumers parking on callback_task_token must be able to
            # ask whether this deployment completes tokens
            "features": (["callback_task_token"] if CALLBACK_TABLE else []),
        }
        # Field-level contract for every output format kind — what is INSIDE
        # each output file (compact example per kind).  Static per
        # TOOL_VERSION, so a caller caches it for the session and CI can pin
        # it without running a job.  Deliberately here and not in
        # get_job_status: examples are per-version data and would otherwise be
        # re-delivered with every job.
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
# Tool: process_media
# ---------------------------------------------------------------------------

def _read_callback_token(event: dict):
    """callback_task_token, validated LOUDLY (bedrock/fi convention): a
    caller with a token is a waitForTaskToken workflow and never reads
    this function's return — failures on this path must RAISE. Shared by
    all three submit tools (process_media / quick_edit / capture_frames)
    so the contract cannot drift between them."""
    callback_token = event.get("callback_task_token")
    if callback_token is None:
        return None
    if (not isinstance(callback_token, str)
            or not callback_token.strip()
            or len(callback_token) > 2048):
        raise ValueError(
            "callback_task_token must be a non-empty Step Functions "
            "task token string (a waitForTaskToken caller cannot read "
            "a returned error, so this raises)")
    if not CALLBACK_TABLE:
        raise RuntimeError(
            "callback_task_token was supplied but CALLBACK_TABLE is "
            "not configured on this deployment")
    return callback_token.strip()


def _process_media(event: dict) -> dict:
    """
    Validate inputs and start a media-processing execution.

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
      callback_task_token    A Step Functions task token from a consumer
                             workflow's ONE held waitForTaskToken state; the
                             deployment's waker completes it with the handle
                             + terminal status AS DATA at execution end (the
                             family sibling pattern — never invoke this
                             stack's state machine directly). With a token
                             present every refusal RAISES (a waiting caller
                             cannot read a returned error). Not part of the
                             request identity.
      use_channel_mapping    Apply audio_channel_mapping (default false)
      audio_channel_mapping  1-based audio stream indices, up to 2 elements
      use_partial            Trim source to mark_in_out window (default false)
      mark_in_out            [start_sec, end_sec] clip window; requires use_partial=true
      crop_x                 Pixels to crop from the left (default 0; shared across all groups)
      crop_y                 Pixels to crop from the top (default 0; shared across all groups)
      width                  Output video width in pixels (default 960)
      height                 Output video height in pixels (default 540)

    Optional — output groups:
      output_groups  List of output group dicts. Each must have group_name and format.
                     Supported formats: mp4, wav, m4a, jpeg, png, metadata, ad_mix.
                     Defaults to a single mp4 proxy group if omitted.
                     Each group may also carry codec, gop_seconds, motion_factor,
                     sampling_rate, audio_channels, audio_bitrate, frame_interval_sec,
                     detail (metadata only: "default" | "full"),
                     render_manifest_s3uri (ad_mix only, REQUIRED there: the
                     audio-description workflow's render_manifest.json — the
                     programme audio is ducked under the narration takes it
                     places, then loudness-normalized to one wav).
                     A metadata-only request skips the transcode pipeline entirely —
                     two Lambda invocations, no proxy produced. width/height/crop_*
                     are ignored for it.

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

    # ── mark_in_out ────────────────────────────────────────────────────────────
    mark_in_out = event.get("mark_in_out")
    use_partial  = bool(event.get("use_partial", False))
    if mark_in_out is not None:
        if not isinstance(mark_in_out, list) or len(mark_in_out) != 2:
            return _error("mark_in_out must be [start_sec, end_sec]", "INVALID_PARAMETER")
        try:
            mark_in_out = [float(mark_in_out[0]), float(mark_in_out[1])]
        except (TypeError, ValueError):
            return _error("mark_in_out values must be numbers", "INVALID_PARAMETER")
        if mark_in_out[1] <= mark_in_out[0]:
            return _error("mark_in_out end_sec must be greater than start_sec", "INVALID_PARAMETER")
        use_partial = True

    # ── audio_channel_mapping ──────────────────────────────────────────────────
    audio_channel_mapping = event.get("audio_channel_mapping")
    if audio_channel_mapping is not None:
        if not isinstance(audio_channel_mapping, list):
            return _error("audio_channel_mapping must be a list of integers", "INVALID_PARAMETER")
        try:
            audio_channel_mapping = [int(x) for x in audio_channel_mapping[:2]]
        except (TypeError, ValueError):
            return _error("audio_channel_mapping elements must be integers", "INVALID_PARAMETER")
        if any(x < 1 for x in audio_channel_mapping):
            return _error("audio_channel_mapping indices are 1-based (minimum 1)", "INVALID_PARAMETER")

    use_channel_mapping = bool(event.get("use_channel_mapping", False))
    if audio_channel_mapping:
        use_channel_mapping = True

    # ── int / float video params ───────────────────────────────────────────────
    err = _parse_int(event, "crop_x", min_val=0)
    if isinstance(err, dict): return err
    crop_x = err

    err = _parse_int(event, "crop_y", min_val=0)
    if isinstance(err, dict): return err
    crop_y = err

    err = _parse_int(event, "width", min_val=1)
    if isinstance(err, dict): return err
    width = err

    err = _parse_int(event, "height", min_val=1)
    if isinstance(err, dict): return err
    height = err

    # ── output_groups ──────────────────────────────────────────────────────────
    raw_groups = event.get("output_groups")
    output_groups: list[dict] | None = None
    if raw_groups is not None:
        if not isinstance(raw_groups, list) or len(raw_groups) == 0:
            return _error("output_groups must be a non-empty list", "INVALID_PARAMETER")
        output_groups = []
        for i, g in enumerate(raw_groups):
            if not isinstance(g, dict):
                return _error(f"output_groups[{i}] must be an object", "INVALID_PARAMETER")
            group_name = (g.get("group_name") or "").strip()
            if not group_name:
                return _error(f"output_groups[{i}].group_name is required", "INVALID_PARAMETER")
            fmt = (g.get("format") or "").strip().lower()
            if fmt not in _VALID_FORMATS:
                return _error(
                    f"output_groups[{i}].format must be one of: {sorted(_VALID_FORMATS)}",
                    "INVALID_PARAMETER",
                )
            group: dict = {"group_name": group_name, "format": fmt}
            if g.get("detail") is not None:
                detail = str(g["detail"]).strip().lower()
                if detail not in _VALID_DETAIL:
                    return _error(
                        f"output_groups[{i}].detail must be one of: {sorted(_VALID_DETAIL)}",
                        "INVALID_PARAMETER",
                    )
                group["detail"] = detail
            for optional_str in ("name", "codec"):
                if g.get(optional_str):
                    group[optional_str] = g[optional_str]
            for optional_num in ("gop_seconds", "motion_factor", "frame_interval_sec"):
                if g.get(optional_num) is not None:
                    try:
                        group[optional_num] = float(g[optional_num])
                    except (TypeError, ValueError):
                        return _error(
                            f"output_groups[{i}].{optional_num} must be a number",
                            "INVALID_PARAMETER",
                        )
            for optional_int in ("sampling_rate", "audio_channels", "audio_bitrate"):
                if g.get(optional_int) is not None:
                    try:
                        group[optional_int] = int(g[optional_int])
                    except (TypeError, ValueError):
                        return _error(
                            f"output_groups[{i}].{optional_int} must be an integer",
                            "INVALID_PARAMETER",
                        )
            # ad_mix (audio-description render): the manifest pointer is
            # REQUIRED for the format and meaningless on any other
            if fmt == "ad_mix":
                raw_manifest = g.get("render_manifest_s3uri")
                manifest = (raw_manifest.strip()
                            if isinstance(raw_manifest, str) else "")
                # full bucket+key shape validated HERE, not just the
                # scheme — "s3://bucket" with no key used to start an
                # execution that died at probe (review 2026-08-31)
                bucket_key = manifest[5:].split("/", 1) \
                    if manifest.startswith("s3://") else []
                if len(bucket_key) != 2 or not bucket_key[0] \
                        or not bucket_key[1].strip("/"):
                    return _error(
                        f"output_groups[{i}]: format ad_mix requires "
                        "render_manifest_s3uri (a full s3://bucket/key "
                        "uri of the AD workflow's render_manifest.json)",
                        "INVALID_PARAMETER",
                    )
                group["render_manifest_s3uri"] = manifest
            elif g.get("render_manifest_s3uri"):
                return _error(
                    f"output_groups[{i}].render_manifest_s3uri only "
                    "applies to format ad_mix",
                    "INVALID_PARAMETER",
                )
            output_groups.append(group)

    # ── Build nested SFN input (no uuid yet) ───────────────────────────────────
    input_config: dict = {}
    if use_channel_mapping:
        input_config["use_channel_mapping"] = True
    if audio_channel_mapping:
        input_config["audio_channel_mapping"] = audio_channel_mapping
    if use_partial:
        input_config["use_partial"] = True
    if mark_in_out is not None:
        input_config["mark_in_out"] = mark_in_out
    if crop_x is not None:
        input_config["crop_x"] = crop_x
    if crop_y is not None:
        input_config["crop_y"] = crop_y
    if width is not None:
        input_config["width"] = width
    if height is not None:
        input_config["height"] = height

    # ── SFN input (without output.s3uri — the prefix derives from the
    #    workflow_id, so it cannot participate in the hash) ─────────────────────
    sfn_input: dict = {
        "input":  {
            "s3uri": asset_s3uri,
            **({"input_config": input_config} if input_config else {}),
        },
        "output": {
            **({"output_groups": output_groups} if output_groups is not None else {}),
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

    callback_token = _read_callback_token(event)

    # ── Register the callback BEFORE StartExecution ───────────────────────
    # The execution ARN is derivable from the name we are about to use, so
    # the row exists before the run can possibly finish — the waker can
    # never fire first. A stale row from a failed start is TTL-cleaned.
    execution_arn = (
        f"{MEDIA_PROCESSING_STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
        f":{job_uuid}")
    if callback_token:
        callback_registry.register(_ddb, CALLBACK_TABLE, execution_arn,
                                   callback_token, {
            "job_id": execution_arn,
            "workflow_id": job_uuid,
            "asset_s3uri": asset_s3uri,
            "output_s3_prefix": output_s3_prefix,
        })

    logger.info(
        "process_media: workflow_id=%s asset=%s output=%s",
        job_uuid, asset_s3uri, output_s3_prefix,
    )

    # Start-or-absorb is the ONE shared tail for all three submit tools;
    # the asset in handle_extra makes the absorb branch read the existing
    # run's asset_s3uri back alongside its output prefix.
    return _start_or_absorb(sfn_input, job_uuid,
                            {"asset_s3uri": asset_s3uri}, callback_token)


# ---------------------------------------------------------------------------
# Tools: quick_edit / capture_frames (2026-10-09)
# ---------------------------------------------------------------------------
# Separate TOOLS on this target, deliberately not process_media output
# groups: process_media's whole contract hangs off ONE asset_s3uri, and a
# cut list over N sources has no honest value for it (README.md §14.1,
# "Gateway surface"). Validation is the SAME module the probe state runs
# (cut_list/frame_capture, byte-identical flat copies), so a bad request
# refuses here, before StartExecution, with the same named condition it
# would get at probe. Both tools take callback_task_token like
# process_media (added 2026-10-10; v1 shipped without it).

try:
    from lib.cut_list import CutListError, validate_quick_edit
    from lib.frame_capture import validate_frame_capture
except ImportError:          # the flat gateway zip has no lib/ package
    from cut_list import CutListError, validate_quick_edit
    from frame_capture import validate_frame_capture


def _start_or_absorb(sfn_input: dict, job_uuid: str, handle_extra: dict,
                     callback_token=None) -> dict:
    """StartExecution-or-absorb — the ONE shared tail for all three submit
    tools (process_media / quick_edit / capture_frames; consolidated
    2026-10-12). Returns the standard HANDLE; an absorbed resubmit differs
    only in `status` + `message` (+ `error` on a terminal failure) — never
    results: those come from get_job_status alone, so a submit's reachable
    field set equals its declared field set by construction.

    On absorb, `output_s3_prefix` (and `asset_s3uri`, when the caller's
    handle carries one) is READ BACK from the existing execution's own
    input rather than re-derived: the handle must locate the artifacts of
    the run that actually exists. The two normally match — the execution
    name is the full work-hash — but output.s3uri is excluded from that
    hash, so a deployment whose proxy bucket was repointed between the
    original submit and this one would otherwise hand back a prefix where
    nothing lives (user ruling 2026-10-12; process_media had this read-back
    from the start, the edit tools gained it at the consolidation).

    With a callback_token (already registered by the caller): a failure to
    start deletes the row and RAISES (the holder cannot read a returned
    error), and an absorb into an already-finished run completes the token
    NOW (paid-work reuse) — a RUNNING absorb leaves the row for the waker.
    """
    output_s3_prefix = sfn_input["output"]["s3uri"]
    execution_arn = (
        f"{MEDIA_PROCESSING_STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
        f":{job_uuid}")
    try:
        resp = _sfn.start_execution(
            stateMachineArn=MEDIA_PROCESSING_STATE_MACHINE_ARN,
            name=job_uuid,
            input=json.dumps(sfn_input),
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ExecutionAlreadyExists":
            msg = exc.response["Error"]["Message"]
            logger.exception("StartExecution failed: %s", msg)
            if callback_token:
                # no execution will ever emit an event for this row, and
                # the waiting caller cannot read a returned error
                callback_registry.delete(_ddb, CALLBACK_TABLE,
                                         execution_arn, callback_token)
                raise RuntimeError(f"Failed to start job: {msg}") from exc
            return _error(f"Failed to start job: {msg}", "SFN_START_FAILED")
        logger.info("Execution already exists for workflow_id=%s: %s",
                    job_uuid, execution_arn)
        try:
            existing = _sfn.describe_execution(executionArn=execution_arn)
        except ClientError as desc_exc:
            if callback_token:
                # the raise tells the holder the submit failed, so it
                # consumes its token itself — delete the row THIS call
                # registered, or the waker later fires on a spent token
                callback_registry.delete(_ddb, CALLBACK_TABLE,
                                         execution_arn, callback_token)
                raise
            return _error(
                "Job already exists but could not retrieve status: "
                f"{desc_exc.response['Error']['Message']}",
                "SFN_DESCRIBE_FAILED")
        actual_status = existing.get("status", "RUNNING")
        result = {
            "job_id": execution_arn,
            "status": actual_status,
            "workflow_id": job_uuid,
            # the existing run's own location, not this call's derivation
            "output_s3_prefix": (_extract_output_s3_prefix(existing)
                                 or output_s3_prefix),
            "message": _ABSORBED_MESSAGE,
            **handle_extra,
        }
        if "asset_s3uri" in handle_extra:
            result["asset_s3uri"] = (_extract_asset_s3uri(existing)
                                     or handle_extra["asset_s3uri"])
        if actual_status == "RUNNING":
            # current_state costs a GetExecutionHistory paging pass, and is
            # meaningless once the execution has stopped.
            result["current_state"] = _get_current_state(execution_arn)
        elif actual_status != "SUCCEEDED":
            # The agent needs the reason to choose between "change a
            # parameter" and "resubmit with a different idempotency_token" —
            # a terminal failure is sticky for as long as Step Functions
            # reserves the execution name, so a plain retry cannot clear it.
            result["error"] = (existing.get("cause")
                               or existing.get("error") or actual_status)
        if (callback_token
                and actual_status not in ("RUNNING", "PENDING_REDRIVE")):
            # Absorbed into a run that already FINISHED: its terminal event
            # fired long ago, so no waker will ever see this token —
            # complete it NOW (paid-work reuse). RUNNING absorbs need
            # nothing: the row was registered before StartExecution and the
            # terminal event is still coming. PENDING_REDRIVE is the same
            # case, not a terminal: completing now would hand the consumer
            # the OLD failure while the redriven run may yet succeed — the
            # redrive's own terminal event reaches the waker.
            # the TOKEN payload uses the ONE waker shape (error = the
            # raising class name, cause = compact parse-safe JSON) so a
            # consumer never has to shape-sniff two completion formats;
            # the agent-facing `result` above keeps its documented
            # error-is-the-cause field untouched
            payload = dict(result)
            if actual_status == "SUCCEEDED":
                payload["message"] = (
                    "completed via callback: the media-processing run "
                    "reached SUCCEEDED; outputs are at output_s3_prefix "
                    "(see the output contract for per-group keys)")
            else:
                payload.pop("error", None)
                if existing.get("error"):
                    payload["error"] = existing["error"]
                if existing.get("cause"):
                    # function-local: callback_waker's module body builds
                    # its own DDB/SFN clients, and this 256 MB dispatcher
                    # must not pay for them at every cold start just to
                    # borrow a pure string helper (settled after trying
                    # the top-level import)
                    from callback_waker import _compact_cause
                    payload["cause"] = _compact_cause(existing["cause"])
                # mirror the waker's message contract too — a machine
                # state cannot "call get_job_status", and TIMED_OUT/
                # ABORTED details carry no error fields to point at.
                # MUST-MATCH: callback_waker.lambda_handler builds the
                # same payload shape — edit both or the two completion
                # paths drift (a shared builder would live in
                # callback_registry, but that is the byte-pinned family
                # copy; widening its API is a family-wide change)
                payload["message"] = (
                    f"completed via callback: the media-processing run "
                    f"reached {actual_status} — see error/cause for the "
                    f"condition"
                    if ("error" in payload or "cause" in payload) else
                    f"completed via callback: the media-processing run "
                    f"reached {actual_status}; the execution record "
                    f"carried no error detail")
            callback_registry.complete_now(_sfn, callback_token, payload)
            callback_registry.delete(_ddb, CALLBACK_TABLE,
                                     execution_arn, callback_token)
        return result
    return {
        "job_id": resp["executionArn"],
        "status": "RUNNING",
        "current_state": _FIRST_STATE,
        "workflow_id": job_uuid,
        "output_s3_prefix": output_s3_prefix,
        **handle_extra,
    }


def _submit_edit_mode(event: dict, input_key: str, payload: dict,
                      summary: dict) -> dict:
    """Shared tail: workflow_id, service-derived prefix, start/absorb."""
    if not PROXY_BUCKET or not TARGET_NAME:
        return _error(
            "PROXY_BUCKET and TARGET_NAME env vars must be set on this function",
            "CONFIG_ERROR")
    callback_token = _read_callback_token(event)
    sfn_input: dict = {"input": {input_key: payload}, "output": {}, "data": {}}
    idempotency_token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")
    job_uuid = compute_workflow_id(
        sfn_input, tool_version=TOOL_VERSION, exclude=("output.s3uri",),
        extra_material=(idempotency_token,) if idempotency_token else ())
    sfn_input["uuid"] = job_uuid
    output_s3_prefix = (
        f"s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{job_uuid}")
    sfn_input["output"]["s3uri"] = output_s3_prefix
    if callback_token:
        # register BEFORE StartExecution — the row exists before the run
        # can possibly finish, so the waker can never fire first
        execution_arn = (
            f"{MEDIA_PROCESSING_STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
            f":{job_uuid}")
        callback_registry.register(_ddb, CALLBACK_TABLE, execution_arn,
                                   callback_token, {
            "job_id": execution_arn,
            "workflow_id": job_uuid,
            "output_s3_prefix": output_s3_prefix,
            **summary,
        })
    logger.info("%s: workflow_id=%s %s", input_key, job_uuid, summary)
    return _start_or_absorb(sfn_input, job_uuid, summary, callback_token)


def _quick_edit(event: dict) -> dict:
    """Compose one output video from a cut list over one or more
    sources. The inline JSON cut list (segments / audio_overlays /
    transition / target / loudnorm) is the contract; validation refuses
    by named condition before any execution starts. Takes an optional
    callback_task_token (the family consumer pattern — completed by the
    waker with the handle + terminal status AS DATA)."""
    qe = {k: event[k] for k in
          ("segments", "audio_overlays", "transition", "target", "loudnorm")
          if event.get(k) is not None}
    try:
        norm = validate_quick_edit(qe)
    except CutListError as exc:
        return _error(str(exc), "INVALID_PARAMETER")
    return _submit_edit_mode(
        event, "quick_edit", qe,
        {"segment_count": len(norm["segments"]),
         "source_count": len(norm["distinct_sources"])})


def _capture_frames(event: dict) -> dict:
    """Capture one still per marker (ms, snapped to the source frame
    grid) across one or more sources. Takes an optional
    callback_task_token, same contract as quick_edit."""
    fc = {k: event[k] for k in ("markers", "format", "resize")
          if event.get(k) is not None}
    try:
        norm = validate_frame_capture(fc)
    except CutListError as exc:
        return _error(str(exc), "INVALID_PARAMETER")
    return _submit_edit_mode(
        event, "frame_capture", fc,
        {"marker_count": len(norm["markers"]),
         "source_count": len(norm["distinct_sources"])})


# ---------------------------------------------------------------------------
# Tool: get_job_status
# ---------------------------------------------------------------------------

def _get_job_status(event: dict) -> dict:
    """
    Poll a media-processing execution by its ARN and return structured output.

    Required:
      job_id  Step Functions execution ARN returned by process_media

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
        "duration_sec":        float,
        "has_video":           bool,
        "has_audio":           bool,
        "is_all_intra":        bool | None,
        "output_keys":         {group_name: s3_key},   frame groups → frame_manifest.json
        "probe_s3_key":        str,
        "pipeline_wall_sec":   float,
        "total_estimated_cost":      float,
        "usage_key":           str
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
        result = {"job_id": job_id, "status": sfn_status, "error": cause}
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

    The "Metadata only?" Choice state is invisible here by construction — it emits
    ChoiceStateEntered/Exited, not TaskStateEntered/Exited — so a metadata-only run
    reports the Task stages only, never a raw routing name.
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


def _parse_succeeded_output(resp: dict) -> dict:
    """
    Extract structured results from a SUCCEEDED execution.

    StateUsageTracking is the final state and returns the full event unchanged,
    so resp["output"] is the complete event dict with all data.* fields populated.

    We surface only what an agent needs to act on or report:
      - Identifier (workflow_id)
      - Content facts (duration_sec, has_video, has_audio, is_all_intra)
      - Output S3 keys per group (output_keys dict)
      - Probe key (probe_s3_key)
      - Cost / timing (pipeline_wall_sec, total_estimated_cost, usage_key)

    Internal plumbing (iterators, destination, output_groups reconciled list,
    function_arn, usage_tracking.states) is excluded.
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

        # ── Content facts ─────────────────────────────────────────────────────
        _set(result, "duration_sec",  data.get("duration_sec"))
        _set(result, "has_video",     data.get("has_video"))
        _set(result, "has_audio",     data.get("has_audio"))
        # is_all_intra may legitimately be None (ambiguous codec) — only omit if absent
        if "is_all_intra" in data:
            result["is_all_intra"] = data["is_all_intra"]

        # ── Output keys ───────────────────────────────────────────────────────
        _set(result, "output_keys",   data.get("output_keys"))
        _set(result, "probe_s3_key",  data.get("probe_s3_key"))

        # ── Cost / timing ─────────────────────────────────────────────────────
        usage = data.get("usage_tracking") or {}
        _set(result, "pipeline_wall_sec", usage.get("pipeline_wall_sec"))
        _set(result, "total_estimated_cost",    usage.get("total_estimated_cost"))
        _set(result, "usage_key",         usage.get("usage_key"))

        # ── Self-describing outputs[] ─────────────────────────────────────────
        # Uniform across all gateway tools: {uri, mime, label, desc}.  Group
        # outputs use the caller-chosen group_name as label (runtime-dynamic);
        # frame groups point at their frame_manifest.json.  `desc` is the
        # one-line role description joined from output_contract.json by the
        # group's FORMAT KIND; the field-level contract (what is inside each
        # file) lives in describe_tool's output_contract, not here.
        # Labels and basenames are API — renames are breaking changes.
        dest = data.get("destination") or {}
        output_bucket = dest.get("bucket")
        if output_bucket:
            outputs = []
            # Format kind per group, read from the reconciled output_groups
            # the probe state stamped — NOT from the key's file extension: a
            # jpeg group's output object is frame_manifest.json, so the
            # extension would join the wrong contract entry.
            group_kinds = {
                g.get("group_name"): g.get("format")
                for g in (data.get("output_groups") or [])
                if isinstance(g, dict)
            }
            output_keys = data.get("output_keys")
            if isinstance(output_keys, dict):
                for group_name in sorted(output_keys):
                    key = output_keys[group_name]
                    if key:
                        outputs.append(_output_entry(
                            output_bucket, key, group_name,
                            _mime_for_key(key),
                            # fixed-label keys (edit_manifest) are not
                            # groups — their contract entry IS the label
                            kind=group_kinds.get(group_name) or group_name))
            for key, label in (
                (data.get("probe_s3_key"),  "probe"),
                (usage.get("usage_key"),    "usage"),
            ):
                if key:
                    outputs.append(_output_entry(
                        output_bucket, key, label, _MIME_JSON, kind=label))
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
