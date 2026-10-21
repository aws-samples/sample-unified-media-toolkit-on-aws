"""
AgentCore Gateway tool dispatcher for the shot-segmentation micro-workflow.

Three tools:
  segment_shots     → StartExecution on the shot-segmentation state machine.
  get_job_status    → DescribeExecution; parses the workflow output when complete.
  describe_tool     → Returns the deployed tool version, state machine ARN,
                      a short backend description, and the output contract
                      (field-level example per outputs[] label, from
                      output_contract.json in this zip).

AgentCore passes the selected tool name in Lambda context:
  context.client_context.custom["bedrockAgentCoreToolName"]
  → "{target_name}___{tool_name}"   e.g. "shot-segmentation___segment_shots"

The event object is a flat dict of that tool's inputSchema properties.

SFN input contract (shot-segmentation state machine):
  {
    "uuid":   str,
    "input":  {
      "s3uri":  "s3://bucket/path/to/video.mxf",
      "input_config": {             ← optional; all fields have defaults
        "segment_duration_secs":       int,
        "coarse_fps":                  int,
        "cosdist_threshold":           float,
        "shot_boundary_threshold":     float,
        "black_pixel_threshold":       float,
        "black_coverage_pct":          float,
        "black_entropy_max":           float,
        "black_max_bright_pixel":      float,
        "noise_distance_threshold":    float,
        "min_shot_duration":           float,
        "adjacent_centroid_threshold": float,
        "max_adjacent_merge_duration": float,
        "near_black_entropy_max":      float,
        "fade_split_threshold":        float,
        "fade_split_min_duration_sec": float,
        "drift_split_threshold":       float,
        "drift_split_min_duration_sec": float,
        "embed_geometry":              str,
        "max_recurrence_threshold":    float,
        "min_recurrence_count":        int,
        "min_occurrence_gap_sec":      float,
        "max_recurrence_count":        int
      }
    },
    "output": {
      "s3uri": "s3://proxy-bucket/uuid"
    },
    "data": {}
  }

SFN final output (returned by StateUsageTracking, which is End=true):
  The full event dict is returned as-is.  Fields surfaced to the agent:
    Correlation
      event.uuid                                          ← job UUID
    Output location
      event.data.proxy_bucket + proxy_prefix              ← where all outputs are stored
    Primary output S3 keys
      event.data.shots_key                                ← shots.json
      event.data.embeddings_key                           ← embeddings.npz
      event.data.recurrence_key                           ← recurrence.json
    Content facts
      event.data.n_shots                                  ← int
      event.data.n_recurrence_groups                      ← int
      event.data.total_frames                             ← int
      event.data.video.actual_duration_sec                ← float
    Cost / timing
      event.data.usage_tracking.pipeline_wall_sec         ← float
      event.data.usage_tracking.total_estimated_cost      ← float USD
      event.data.usage_tracking.usage_key                 ← S3 key with full cost breakdown

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

  That rejection is scoped to segment_shots — the only tool where either
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
import math
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

STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]
PROXY_BUCKET      = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION      = os.environ.get("TOOL_VERSION", "unknown")
# Must equal the GatewayTarget Name in media-toolkit-gateway/cfnstack.yaml —
# it is the {target} segment of the derived output prefix.
TARGET_NAME       = os.environ.get("TARGET_NAME", "")

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

# outputs[] MIME vocabulary — exact registered types only; must match the
# ContentType set at upload where one is set
_MIME_JSON = "application/json"
_MIME_BIN  = "application/octet-stream"   # .npz / .npy — no registered type

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


def _output_entry(bucket: str, key: str, label: str, mime: str | None):
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

    # `segment_shots` is the only tool here whose execution a token names.
    # `get_job_status` and `describe_tool` accept the field and IGNORE it — a
    # poll reflects whatever the execution is doing now, and no token could have
    # changed that.  Ignoring rather than rejecting is the deliberate trade: the
    # caller is an LLM, a hallucinated parameter name is a routine failure mode,
    # and failing an otherwise-correct poll over a discarded field is worse than
    # a log line.  This runs before dispatch so a tool added later is covered by
    # default.
    if tool != "segment_shots":
        _token_note = warn_idempotency_token(event)
        if _token_note:
            logger.warning("%s: %s", tool, _token_note)

    if tool == "segment_shots":
        return _segment_shots(event)

    if tool == "get_job_status":
        return _get_job_status(event)

    if tool == "describe_tool":
        resp = {
            "version": TOOL_VERSION,
            "state_machine_arn": STATE_MACHINE_ARN,
            "description": ("DINOv2 shot segmentation Step Functions pipeline: "
                            "shot boundaries, frame/shot embeddings, recurring shot groups"),
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


def _parse_float(event: dict, key: str,
                 min_val: float | None = None, max_val: float | None = None,
                 min_exclusive: bool = False, max_exclusive: bool = False):
    raw = event.get(key)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _error(f"{key} must be a number", "INVALID_PARAMETER")
    if not math.isfinite(val):
        return _error(f"{key} must be a finite number (got {val})", "INVALID_PARAMETER")
    if min_val is not None:
        if min_exclusive and val <= min_val:
            return _error(f"{key} must be > {min_val} (got {val})", "INVALID_PARAMETER")
        elif not min_exclusive and val < min_val:
            return _error(f"{key} must be >= {min_val} (got {val})", "INVALID_PARAMETER")
    if max_val is not None:
        if max_exclusive and val >= max_val:
            return _error(f"{key} must be < {max_val} (got {val})", "INVALID_PARAMETER")
        elif not max_exclusive and val > max_val:
            return _error(f"{key} must be <= {max_val} (got {val})", "INVALID_PARAMETER")
    return val


def _parse_s3uri(uri: str) -> tuple[str, str] | None:
    """Return (bucket, key) or None if invalid."""
    if not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    if "/" not in rest:
        return None
    bucket, key = rest.split("/", 1)
    if not bucket:
        return None
    key = key.strip("/")
    if not key:
        return None
    return bucket, key


# ---------------------------------------------------------------------------
# Tool: segment_shots
# ---------------------------------------------------------------------------

def _segment_shots(event: dict) -> dict:
    """
    Validate inputs and start a shot segmentation execution.

    Required:
      asset_s3uri    S3 URI of the source video file

    Optional:
      output_s3_prefix            DEPRECATED and ignored — outputs are written
                                  to the service-derived prefix (returned as
                                  output_s3_prefix in the response)
      idempotency_token           Optional idempotency token — assert that this request is a
                                  resend of a previous one. Omit it and the parameters
                                  alone determine the run. Consumed into workflow_id and
                                  discarded; never echoed back.
      segment_duration_secs       Transcode segment length (>= 60, default 1200)
      coarse_fps                  DINOv2 frame sampling rate (>= 1, default 4)
      cosdist_threshold           Cosine-distance zoom-in trigger (0,1], default 0.10
      shot_boundary_threshold     Primary shot-cut sensitivity (0,1], default 0.20
      black_pixel_threshold       Black frame pixel threshold [0,1), default 0.04
      black_coverage_pct          Black frame coverage threshold (0,100], default 99.5
      black_entropy_max           Black frame entropy ceiling [0,10), default 0.20
      black_max_bright_pixel      Black frame bright-pixel ceiling in luma (0–255], default 30.0
                                  Any pixel brighter than this disqualifies the frame as black
                                  (catches title cards / logos on a black background)
      noise_distance_threshold    Max centroid distance between neighbours of a short noise shot
                                  to trigger a three-way merge (0,1), default 0.18
      min_shot_duration           Unconditional minimum shot duration in seconds (> 0, default 0.25).
                                  Shots shorter than this are absorbed into the more similar
                                  neighbouring shot — they cannot be real editorial
                                  shots. Must be < 0.5 (the fixed noise-shot duration ceiling).
      adjacent_centroid_threshold Max centroid distance between two adjacent content shots to
                                  merge them as a motion-blur false cut (0, shot_boundary_threshold],
                                  default 0.20
      fade_split_threshold        Min centroid-of-halves distance between the two halves of a long
                                  shot to split it as a fade transition (0,1], default 0.45.
                                  Higher = fewer splits. This default is provisional and
                                  content-type dependent — tune down for cross-fades and subtle
                                  dissolves, up for high-motion content (sport, news) where
                                  intra-shot centroid drift is naturally higher.
      fade_split_min_duration_sec Minimum shot duration in seconds before fade-split is attempted
                                  (> 0, default 8.0)
      max_adjacent_merge_duration Hard ceiling in seconds on the result of an adjacent-drift
                                  merge (> 0, default 30.0). Prevents runaway merge chains from
                                  producing multi-minute shots across commercial breaks.
      near_black_entropy_max      Shots with entropy_mean below this are reclassified as 'black'
                                  before and after the merge passes ([0,10), default 1.5).
                                  Catches near-black shots the per-frame black detector missed.
      drift_split_threshold       Minimum 1s-stride cosine distance peak to split a very long
                                  shot at a slow scene drift ((0,1], default 0.40). Gradual
                                  transitions never exceed shot_boundary_threshold frame-to-frame
                                  but accumulate over a second.
      drift_split_min_duration_sec Only shots longer than this are drift-split candidates
                                  (> fade_split_min_duration_sec, default 60.0). A >60s single
                                  shot is a-priori improbable in edited content.
      embed_geometry              Model-input geometry: 'letterbox' (default), 'squeeze', or
                                  'crop'. All distance thresholds are calibrated on 'letterbox';
                                  changing this shifts the embedding distribution and requires
                                  threshold recalibration.
      max_recurrence_threshold    Recurring-shot grouping DIAMETER bound [0,1) — every pair
                                  inside a group is within it (complete linkage), default 0.07
      min_recurrence_count        Minimum OCCURRENCES per recurrence group (>= 2, default 3)
      min_occurrence_gap_sec      9.1 temporal gate: members closer than this are ONE
                                  occurrence (>= 0, default 60.0)
      max_recurrence_count        9.3 set-recurrence ceiling: drop groups with more raw
                                  members than this (>= 2, default 8)

    Returns a HANDLE, never results:
      {
        "job_id":           str,   # execution ARN — pass to get_job_status
        "workflow_id":      str,   # deterministic id of the run (the execution name)
        "status":           str,   # RUNNING on a fresh start; the existing run's
                                   # status when absorbed (see below)
        "current_state":    str,   # RUNNING only
        "asset_s3uri":     str,
        "output_s3_prefix": str,
        "error":            str,   # terminal-failure absorb only
        "message":          str    # absorbed only — the sole absorption signal
      }

    An absorbed resubmit (same derived workflow_id as a run that already exists)
    returns the same shape as a fresh start with `status` set to the existing
    run's, plus `message`.  It never returns that run's results: every caller
    reads results from get_job_status, so there is exactly one result-shaped
    response in the protocol and a submit's reachable fields equal its declared
    fields by construction.
    """
    # ── Required ───────────────────────────────────────────────────────────────
    asset_s3uri = (event.get("asset_s3uri") or "").strip()
    if not asset_s3uri:
        return _error("asset_s3uri is required", "MISSING_PARAMETER")
    parsed = _parse_s3uri(asset_s3uri)
    if parsed is None:
        return _error("asset_s3uri must be a valid S3 URI (s3://bucket/key)", "INVALID_PARAMETER")

    # ── int params ─────────────────────────────────────────────────────────────
    err = _parse_int(event, "segment_duration_secs",  min_val=60)
    if isinstance(err, dict): return err
    segment_duration_secs = err

    err = _parse_int(event, "coarse_fps",             min_val=1)
    if isinstance(err, dict): return err
    coarse_fps = err

    err = _parse_int(event, "min_recurrence_count",   min_val=2)
    if isinstance(err, dict): return err
    min_recurrence_count = err

    err = _parse_int(event, "max_recurrence_count",   min_val=2)
    if isinstance(err, dict): return err
    max_recurrence_count = err

    # ── float params ───────────────────────────────────────────────────────────
    err = _parse_float(event, "cosdist_threshold",         min_val=0.0, max_val=1.0, min_exclusive=True)
    if isinstance(err, dict): return err
    cosdist_threshold = err

    err = _parse_float(event, "shot_boundary_threshold",   min_val=0.0, max_val=1.0, min_exclusive=True)
    if isinstance(err, dict): return err
    shot_boundary_threshold = err

    err = _parse_float(event, "black_pixel_threshold",     min_val=0.0, max_val=1.0, max_exclusive=True)
    if isinstance(err, dict): return err
    black_pixel_threshold = err

    err = _parse_float(event, "black_coverage_pct",        min_val=0.0, max_val=100.0, min_exclusive=True)
    if isinstance(err, dict): return err
    black_coverage_pct = err

    err = _parse_float(event, "black_entropy_max",         min_val=0.0, max_val=10.0, max_exclusive=True)
    if isinstance(err, dict): return err
    black_entropy_max = err

    err = _parse_float(event, "max_recurrence_threshold",    min_val=0.0, max_val=1.0, max_exclusive=True)
    if isinstance(err, dict): return err
    max_recurrence_threshold = err

    err = _parse_float(event, "min_occurrence_gap_sec",      min_val=0.0)
    if isinstance(err, dict): return err
    min_occurrence_gap_sec = err

    err = _parse_float(event, "noise_distance_threshold",    min_val=0.0, max_val=1.0, min_exclusive=True, max_exclusive=True)
    if isinstance(err, dict): return err
    noise_distance_threshold = err

    err = _parse_float(event, "min_shot_duration",           min_val=0.0, min_exclusive=True)
    if isinstance(err, dict): return err
    min_shot_duration = err

    err = _parse_float(event, "adjacent_centroid_threshold", min_val=0.0, max_val=1.0, min_exclusive=True)
    if isinstance(err, dict): return err
    adjacent_centroid_threshold = err

    err = _parse_float(event, "black_max_bright_pixel",      min_val=0.0, max_val=255.0, min_exclusive=True)
    if isinstance(err, dict): return err
    black_max_bright_pixel = err

    err = _parse_float(event, "fade_split_threshold",        min_val=0.0, max_val=1.0, min_exclusive=True)
    if isinstance(err, dict): return err
    fade_split_threshold = err

    err = _parse_float(event, "fade_split_min_duration_sec", min_val=0.0, min_exclusive=True)
    if isinstance(err, dict): return err
    fade_split_min_duration_sec = err

    err = _parse_float(event, "max_adjacent_merge_duration", min_val=0.0, min_exclusive=True)
    if isinstance(err, dict): return err
    max_adjacent_merge_duration = err

    err = _parse_float(event, "near_black_entropy_max",      min_val=0.0, max_val=10.0, max_exclusive=True)
    if isinstance(err, dict): return err
    near_black_entropy_max = err

    err = _parse_float(event, "drift_split_threshold",       min_val=0.0, max_val=1.0, min_exclusive=True)
    if isinstance(err, dict): return err
    drift_split_threshold = err

    err = _parse_float(event, "drift_split_min_duration_sec", min_val=0.0, min_exclusive=True)
    if isinstance(err, dict): return err
    drift_split_min_duration_sec = err

    # ── enum params ────────────────────────────────────────────────────────────
    embed_geometry = (event.get("embed_geometry") or "").strip() or None
    if embed_geometry is not None and embed_geometry not in ("letterbox", "squeeze", "crop"):
        return _error(
            f"embed_geometry must be one of letterbox, squeeze, crop "
            f"(got {embed_geometry!r})",
            "INVALID_PARAMETER",
        )

    # ── Cross-parameter validation ─────────────────────────────────────────────
    # Mirror the workflow's validate_and_normalize() constraints so agents get
    # an immediate tool error instead of an async execution failure.  Defaults
    # here must match lib/input_schema.py.
    _sbt = shot_boundary_threshold if shot_boundary_threshold is not None else 0.20
    _act = adjacent_centroid_threshold if adjacent_centroid_threshold is not None else 0.20
    if _act > _sbt:
        return _error(
            f"adjacent_centroid_threshold ({_act}"
            f"{'' if adjacent_centroid_threshold is not None else ', default'}) "
            f"must be <= shot_boundary_threshold ({_sbt}"
            f"{'' if shot_boundary_threshold is not None else ', default'}) — "
            f"when lowering shot_boundary_threshold below 0.20, also set "
            f"adjacent_centroid_threshold at or below it",
            "INVALID_PARAMETER",
        )

    # max_noise_duration is fixed at 0.5s (not exposed as a parameter)
    if min_shot_duration is not None and min_shot_duration >= 0.5:
        return _error(
            f"min_shot_duration ({min_shot_duration}) must be < 0.5 "
            f"(the fixed max_noise_duration)",
            "INVALID_PARAMETER",
        )

    _fsd = fade_split_min_duration_sec if fade_split_min_duration_sec is not None else 8.0
    _dsd = drift_split_min_duration_sec if drift_split_min_duration_sec is not None else 60.0
    if _dsd <= _fsd:
        return _error(
            f"drift_split_min_duration_sec ({_dsd}"
            f"{'' if drift_split_min_duration_sec is not None else ', default'}) "
            f"must be > fade_split_min_duration_sec ({_fsd}"
            f"{'' if fade_split_min_duration_sec is not None else ', default'})",
            "INVALID_PARAMETER",
        )

    # ── Build input_config (omit defaults — the workflow fills them) ───────────
    input_config: dict = {}
    if segment_duration_secs       is not None: input_config["segment_duration_secs"]       = segment_duration_secs
    if coarse_fps                  is not None: input_config["coarse_fps"]                  = coarse_fps
    if cosdist_threshold           is not None: input_config["cosdist_threshold"]           = cosdist_threshold
    if shot_boundary_threshold     is not None: input_config["shot_boundary_threshold"]     = shot_boundary_threshold
    if black_pixel_threshold       is not None: input_config["black_pixel_threshold"]       = black_pixel_threshold
    if black_coverage_pct          is not None: input_config["black_coverage_pct"]          = black_coverage_pct
    if black_entropy_max           is not None: input_config["black_entropy_max"]           = black_entropy_max
    if black_max_bright_pixel      is not None: input_config["black_max_bright_pixel"]      = black_max_bright_pixel
    if noise_distance_threshold    is not None: input_config["noise_distance_threshold"]    = noise_distance_threshold
    if min_shot_duration           is not None: input_config["min_shot_duration"]           = min_shot_duration
    if adjacent_centroid_threshold is not None: input_config["adjacent_centroid_threshold"] = adjacent_centroid_threshold
    if fade_split_threshold        is not None: input_config["fade_split_threshold"]        = fade_split_threshold
    if fade_split_min_duration_sec is not None: input_config["fade_split_min_duration_sec"] = fade_split_min_duration_sec
    if max_adjacent_merge_duration is not None: input_config["max_adjacent_merge_duration"] = max_adjacent_merge_duration
    if near_black_entropy_max      is not None: input_config["near_black_entropy_max"]      = near_black_entropy_max
    if drift_split_threshold       is not None: input_config["drift_split_threshold"]       = drift_split_threshold
    if drift_split_min_duration_sec is not None: input_config["drift_split_min_duration_sec"] = drift_split_min_duration_sec
    if embed_geometry              is not None: input_config["embed_geometry"]              = embed_geometry
    if max_recurrence_threshold    is not None: input_config["max_recurrence_threshold"]    = max_recurrence_threshold
    if min_recurrence_count        is not None: input_config["min_recurrence_count"]        = min_recurrence_count
    if min_occurrence_gap_sec      is not None: input_config["min_occurrence_gap_sec"]      = min_occurrence_gap_sec
    if max_recurrence_count        is not None: input_config["max_recurrence_count"]        = max_recurrence_count

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

    # ── SFN input (without output.s3uri — the prefix derives from the
    #    workflow_id, so it cannot participate in the hash) ─────────────────────
    sfn_input: dict = {
        "input": {
            "s3uri":  asset_s3uri,
            **({"input_config": input_config} if input_config else {}),
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
    sfn_input["output"] = {"s3uri": output_s3_prefix}

    logger.info(
        "segment_shots: workflow_id=%s asset=%s output=%s",
        job_uuid, asset_s3uri, output_s3_prefix,
    )

    try:
        resp = _sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=job_uuid,
            input=json.dumps(sfn_input),
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ExecutionAlreadyExists":
            execution_arn = (
                f"{STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}:{job_uuid}"
            )
            logger.info("Execution already exists for workflow_id=%s: %s", job_uuid, execution_arn)
            try:
                existing = _sfn.describe_execution(executionArn=execution_arn)
            except ClientError as desc_exc:
                desc_code = desc_exc.response["Error"]["Code"]
                desc_msg  = desc_exc.response["Error"]["Message"]
                logger.exception("DescribeExecution failed for existing job: %s — %s", desc_code, desc_msg)
                return _error(f"Job already exists but could not retrieve status: {desc_msg}", "SFN_DESCRIBE_FAILED")
            # An absorbed resubmit returns a HANDLE, exactly like a fresh start —
            # only `status` varies, plus `message` to say the call did not
            # resubmit.  It deliberately does NOT return the existing run's
            # results, even when that run already SUCCEEDED: results come from
            # get_job_status and only from there, so a submit's reachable field
            # set equals its declared field set by construction and the agent has
            # one rule to follow instead of two.
            actual_status = existing.get("status", "RUNNING")
            result = {
                "job_id":           execution_arn,
                "status":           actual_status,
                "workflow_id":      job_uuid,
                # Read back from the existing run rather than echoing this
                # call's values: identical parameters hash to the same
                # workflow_id, so these must agree, and reporting what the run
                # actually holds is the honest answer if they ever do not.
                "asset_s3uri":     _extract_asset_s3uri(existing) or asset_s3uri,
                "output_s3_prefix": (
                    _extract_output_s3_prefix(existing) or output_s3_prefix
                ),
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
                result["error"] = (
                    existing.get("cause") or existing.get("error") or actual_status
                )
            return result
        msg = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s — %s", code, msg)
        return _error(f"Failed to start shot segmentation job: {msg}", "SFN_START_FAILED")

    job_id = resp["executionArn"]
    logger.info("Started execution: %s", job_id)
    return {
        "job_id":           job_id,
        "status":           "RUNNING",
        "current_state":    "Probe video for chunk transcode",
        "workflow_id":      job_uuid,
        "asset_s3uri":     asset_s3uri,
        "output_s3_prefix": output_s3_prefix,
    }


# ---------------------------------------------------------------------------
# Tool: get_job_status
# ---------------------------------------------------------------------------

def _get_job_status(event: dict) -> dict:
    """
    Poll a shot segmentation execution by its ARN and return structured output.

    Required:
      job_id  Step Functions execution ARN returned by segment_shots

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
        "job_id":                 str,
        "workflow_id":            str,
        "status":                 "SUCCEEDED",
        "asset_s3uri":           str,
        "output_s3_prefix":       str,
        "shots_key":              str,
        "embeddings_key":         str,
        "recurrence_key":         str,
        "n_shots":                int,
        "n_recurrence_groups":    int,
        "total_frames":           int,
        "actual_duration_sec":    float,
        "pipeline_wall_sec":      float,
        "total_estimated_cost":   float,
        "usage_key":              str
      }
      All fields are omitted when absent in the workflow output.

    Returns on FAILED / TIMED_OUT / ABORTED:
      {"job_id": str, "workflow_id": str, "status": <status>,
       "error": <cause>, "asset_s3uri": str}
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

    sfn_status   = resp["status"]
    asset_s3uri = _extract_asset_s3uri(resp)

    if sfn_status == "RUNNING":
        result = {
            "job_id":        job_id,
            "status":        "RUNNING",
            "current_state": _get_current_state(job_id),
            "asset_s3uri":  asset_s3uri,
        }
        _set(result, "workflow_id", job_id.rsplit(":", 1)[-1] or None)
        _set(result, "output_s3_prefix", _extract_output_s3_prefix(resp))
        return result

    if sfn_status in ("FAILED", "TIMED_OUT", "ABORTED"):
        cause  = resp.get("cause") or resp.get("error") or sfn_status
        result = {"job_id": job_id, "status": sfn_status, "error": cause}
        _set(result, "asset_s3uri", asset_s3uri)
        _set(result, "workflow_id", job_id.rsplit(":", 1)[-1] or None)
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
    _set(result, "workflow_id", job_id.rsplit(":", 1)[-1] or None)
    _set(result, "output_s3_prefix", _extract_output_s3_prefix(resp))
    return result


def _get_current_state(execution_arn: str) -> str:
    """
    Return the name of the currently active pipeline state for a RUNNING execution.

    Strategy: count TaskStateEntered and TaskStateExited events per state name.
    States where entered_count > exited_count are still active.  For Map states,
    the outer state has entered_count==1 while each iterator sub-state has
    entered_count==N — returning the state with the lowest entry count surfaces
    the outer pipeline stage, which is most useful to the agent.
    """
    try:
        entered: dict[str, int] = {}
        exited:  dict[str, int] = {}

        kwargs: dict = {"executionArn": execution_arn, "maxResults": 1000}
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


def _extract_asset_s3uri(resp: dict) -> str | None:
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
    result: dict = {"status": "SUCCEEDED"}
    try:
        payload = json.loads(resp.get("output") or "{}")
        data    = payload.get("data") or {}

        # ── Correlation ───────────────────────────────────────────────────────
        _set(result, "workflow_id",
             (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)
        _set(result, "asset_s3uri",  _extract_asset_s3uri(resp))

        # ── Output location ───────────────────────────────────────────────────
        # Prefer the workflow's own report; fall back to the execution input,
        # which always carries it — the submit promised this field, so a poll
        # must not drop it just because a state stopped reporting proxy_bucket.
        proxy_bucket = data.get("proxy_bucket", None)
        proxy_prefix = data.get("proxy_prefix", None)
        if proxy_bucket and proxy_prefix:
            _set(result, "output_s3_prefix",
                 f"s3://{proxy_bucket}/{proxy_prefix.strip('/')}")
        else:
            _set(result, "output_s3_prefix", _extract_output_s3_prefix(resp))

        # ── Primary output keys ───────────────────────────────────────────────
        _set(result, "shots_key",       data.get("shots_key"))
        _set(result, "embeddings_key",  data.get("embeddings_key"))
        _set(result, "centroids_key",   data.get("centroids_key"))
        _set(result, "recurrence_key",  data.get("recurrence_key"))

        # ── Content facts ─────────────────────────────────────────────────────
        _set(result, "n_shots",             data.get("n_shots"))
        _set(result, "n_recurrence_groups", data.get("n_recurrence_groups"))
        _set(result, "total_frames",        data.get("total_frames"))

        video = data.get("video") or {}
        _set(result, "actual_duration_sec", video.get("actual_duration_sec"))

        # ── Cost / timing ─────────────────────────────────────────────────────
        usage = data.get("usage_tracking") or {}
        _set(result, "pipeline_wall_sec",     usage.get("pipeline_wall_sec"))
        _set(result, "total_estimated_cost",  usage.get("total_estimated_cost"))
        _set(result, "usage_key",             usage.get("usage_key"))

        # ── Self-describing outputs[] ─────────────────────────────────────────
        # Uniform across all gateway tools: {uri, mime, label, desc}.  Entries
        # whose uri ends with "/" are prefixes (many objects) and carry no mime.
        # `desc` is the one-line role description joined from
        # output_contract.json; the field-level contract (what is inside each
        # file) lives in describe_tool's output_contract, not here.
        # Labels and basenames are API — renames are breaking changes.
        if proxy_bucket:
            outputs = []
            for key, label, mime in (
                (data.get("shots_key"),      "shot-list",        _MIME_JSON),
                (data.get("embeddings_key"), "frame-embeddings", _MIME_BIN),
                (data.get("centroids_key"),  "shot-centroids",   _MIME_BIN),
                (data.get("recurrence_key"), "recurrence",       _MIME_JSON),
                (usage.get("usage_key"),     "usage",            _MIME_JSON),
            ):
                if key:
                    outputs.append(_output_entry(proxy_bucket, key, label, mime))
            debug_prefix = data.get("debug_images_prefix")
            if not debug_prefix and proxy_prefix:
                debug_prefix = f"{proxy_prefix.strip('/')}/shot_images/"
            if debug_prefix:
                outputs.append(_output_entry(
                    proxy_bucket, debug_prefix.rstrip("/") + "/",
                    "debug-grids", None))
            if outputs:
                result["outputs"] = outputs

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not parse execution output: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_tool(context) -> str:
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        if _TOOL_DELIMITER in raw:
            return raw.split(_TOOL_DELIMITER, 1)[1]
        return raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("bedrockAgentCoreToolName not found in Lambda context")
        return ""


def _set(d: dict, key: str, value) -> None:
    if value is not None:
        d[key] = value


def _error(message: str, code: str = "ERROR") -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
