# usage_tracking_handler.py
"""
Step Function State: Usage Tracking
─────────────────────────────────────────────────────────────────────────────
Last state in the video embedding pipeline. Enumerates the current
Step Functions execution history to compute per-state Lambda invocation
counts, durations, and estimated costs.

Architecture (arm64 vs x86_64) and memory are resolved via
GetFunctionConfiguration — cached in a local dict for the duration of the
invocation (never persisted to module level).

Lambda pricing (as of 2025)
────────────────────────────
  x86_64 : $0.0000166667 / GB-second  +  $0.0000002 / request
  arm64  : $0.0000133334 / GB-second  +  $0.0000002 / request

Cost per invocation
  = (memory_mb / 1024) × duration_sec × price_per_gb_sec
  + price_per_request

Billed duration
  = LambdaFunctionSucceeded.timestamp − LambdaFunctionScheduled.timestamp
  (slight overestimate — includes Step Functions scheduling overhead ~10-50 ms)

Map state
  Each iterator produces its own Scheduled/Started/Succeeded event chain.
  All iterator costs are summed; wall-clock duration = max iterator duration.

Usage tracking state itself
  Its own Succeeded event has not been emitted yet when this handler runs.
  Duration is estimated as time.time() − TaskScheduled.timestamp from the
  execution history. This is consistent with how all other states are
  measured and correctly includes Lambda cold start and SFN scheduling
  overhead. Falls back to internal elapsed time if the scheduled event
  is not found.

execution_arn
  Always passed explicitly from the state machine Parameters block:
    "execution_arn.$": "$$Execution.Id"
  No environment variable fallback — the explicit field is required.

Expected input event
─────────────────────
{
    "operation":     "StateUsageTracking",
    "uuid":          "...",
    "execution_arn": "arn:aws:states:eu-west-1:...",
    "input":         { ... },
    "data": {
        "proxy_bucket": "bucket",
        "proxy_prefix": "path/to",
        "video": {
            "frame_rate_fps":      25.0,
            "actual_frame_count":  91750,
            "actual_duration_sec": 3670.0
        },
        "total_frames":       6720,
        "total_black_frames": 48,
        "embed_dim":          384,
        ...
    }
}

S3 output
──────────
  {prefix}/usage_tracking.json

Output  (returned to Step Functions)
──────────────────────────────────────
{
    ...original payload...,
    "data": {
        ...original data...,
        "usage_tracking": { ...usage doc... }
    }
}

usage_tracking.json structure
──────────────────────────────
{
    "execution_arn":        "arn:aws:states:...",
    "uuid":                 "...",
    "proxy_bucket":         "proxy-bucket",
    "proxy_prefix":          "prefix",
    "pipeline_wall_sec":    142.3,
    "total_estimated_cost": 0.00412,
    "actual_frame_count":   91750,
    "actual_duration_sec":  3670.0,
    "cost_per_minute":      0.000067,
    "cost_per_frame":       0.000000045,
    "states": [
        {
            "state_name":       "Usage tracking",
            "operation":        "StateUsageTracking",
            "lambda_arn":       "arn:aws:lambda:...",
            "architecture":     "arm64",
            "memory_mb":        896,
            "invocations":      1,
            "wall_sec":         3.2,
            "total_billed_sec": 3.2,
            "estimated_cost":   0.000012
        },
        ...
    ]
}
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Optional

import boto3

_sfn = boto3.client("stepfunctions")
_lambda = boto3.client("lambda")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
_h = logging.StreamHandler()
_h.setLevel(logging.DEBUG)
_h.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
if not log.handlers:
    log.addHandler(_h)

from lib.s3_utils import put_json

# =============================================================================
# PRICING
# =============================================================================

_PRICING: dict[str, dict] = {
    "x86_64": {
        "per_gb_second": 0.0000166667,
        "per_request":   0.0000002,
    },
    "arm64": {
        "per_gb_second": 0.0000133334,
        "per_request":   0.0000002,
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def _get_lambda_config(arn: str, cache: dict[str, dict]) -> dict:
    """
    Return {memory_mb, architecture} for a Lambda ARN.
    Results are cached in the caller-supplied dict (local to the invocation —
    never persisted at module level across warm starts).
    """
    if arn not in cache:
        resp = _lambda.get_function_configuration(FunctionName=arn)
        cache[arn] = {
            "memory_mb":    resp["MemorySize"],
            "architecture": resp.get("Architectures", ["x86_64"])[0],
        }
        log.info(
            "Lambda config: %s  mem=%d MB  arch=%s",
            arn.split(":")[-1],
            cache[arn]["memory_mb"],
            cache[arn]["architecture"],
        )
    return cache[arn]


def _estimated_cost(
    memory_mb:     int,
    duration_sec:  float,
    architecture:  str,
    n_invocations: int = 1,
) -> float:
    """
    Estimated Lambda cost in USD.
    duration_sec is the total billed seconds across all n_invocations
    (i.e. already summed for Map state iterators).
    """
    p = _PRICING.get(architecture, _PRICING["x86_64"])
    compute = (memory_mb / 1024.0) * duration_sec * p["per_gb_second"]
    requests = n_invocations * p["per_request"]
    return round(compute + requests, 8)


def _ts_to_epoch(ts) -> float:
    """Convert boto3 datetime to UTC epoch float."""
    if hasattr(ts, "timestamp"):
        return ts.timestamp()
    return float(ts)


def _billed_sec(inv: dict) -> float:
    """
    Billed seconds for ONE invocation attempt — counting failures.

    AWS bills a Lambda that ran to its timeout and was retried; the most
    expensive invocation a workflow can have is precisely the one that burned
    its full budget and raised. Reading only `succeeded_ts` used to charge such
    an attempt 0 s, so a run under-reported its own cost by up to 7.4x
    (measured on tts: $0.031 reported against a $0.2318 bill) — and it
    under-reported *worst* exactly when the workflow behaved worst. Any state
    using checkpoint/resume makes a timeout a NORMAL path, so this is
    ordinary-run accounting, not a failure-only edge case.

    Falls back to 0.0 only when an attempt has NO terminal event at all: this
    state's own record while it is still running, or an attempt whose terminal
    event is past the end of the history window. Zero is genuinely right there
    and nowhere else.

    The failed_ts fallback and the per-attempt wall_sec summation are the
    two invariants to preserve; tts-on-aws holds the reference copy of this
    logic, and the bundle test suite pins it cross-stack (see the usage
    tracking gotcha in README.md).
    """
    # A waitForTaskToken state emits TaskSubmitted when its Lambda RETURNS;
    # TaskSucceeded only fires when the held token completes — potentially a
    # whole sibling run later. The Lambda's billed runtime ends at submit:
    # the wait is Step Functions holding a token, not compute, and the
    # sibling's own published bill already covers that time (it was double
    # counted here until 2026-10-12). Plain lambda:invoke emits no
    # TaskSubmitted, so this prefers it only where it exists.
    end = inv.get("submitted_ts")
    if end is None:
        end = inv["succeeded_ts"]
    if end is None:
        end = inv.get("failed_ts")
    return (end - inv["scheduled_ts"]) if end is not None else 0.0


def _attempt_span_sec(inv: dict) -> float:
    """Wall span of one attempt: scheduled → terminal event, INCLUDING any
    waitForTaskToken hold — the pipeline really did wait that long. Only
    _billed_sec (Lambda money) stops at TaskSubmitted."""
    end = inv["succeeded_ts"]
    if end is None:
        end = inv.get("failed_ts")
    return (end - inv["scheduled_ts"]) if end is not None else 0.0


def _get_execution_history(execution_arn: str) -> list[dict]:
    """Page through GetExecutionHistory and return all events."""
    events: list[dict] = []
    kwargs: dict = {
        "executionArn": execution_arn,
        "maxResults":   1000,
        "reverseOrder": False,
    }
    while True:
        resp = _sfn.get_execution_history(**kwargs)
        events.extend(resp["events"])
        token = resp.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token
    log.info("Execution history: %d events", len(events))
    return events


def _usage_output_key(proxy_prefix: str) -> str:
    return f"{proxy_prefix.strip('/')}/usage_tracking.json"


# =============================================================================
# EXECUTION HISTORY PARSER
# =============================================================================

def _get_operation_from_input(event: dict) -> Optional[str]:
    """
    Extract the 'operation' field from the Lambda input payload.
    Handles both SDK integration (taskScheduledEventDetails) and
    legacy integration (lambdaFunctionScheduledEventDetails).
    """
    # SDK integration (arn:aws:states:::lambda:invoke)
    details = event.get("taskScheduledEventDetails", {})
    raw = details.get("parameters")
    if raw:
        try:
            params = json.loads(raw)
            payload = params.get("Payload", {})
            if isinstance(payload, str):
                payload = json.loads(payload)
            return payload.get("operation")
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    # Legacy integration (LambdaFunctionScheduled)
    details = event.get("lambdaFunctionScheduledEventDetails", {})
    raw = details.get("input")
    if raw:
        try:
            payload = json.loads(raw)
            return payload.get("operation")
        except (json.JSONDecodeError, AttributeError):
            pass

    return None


def _get_arn_from_scheduled(event: dict) -> str:
    """
    Extract the Lambda function ARN from a TaskScheduled or
    LambdaFunctionScheduled event.
    """
    # SDK integration — resource is nested in parameters JSON
    details = event.get("taskScheduledEventDetails", {})
    raw = details.get("parameters")
    if raw:
        try:
            params = json.loads(raw)
            return params.get("FunctionName", "")
        except (json.JSONDecodeError, AttributeError):
            pass

    # Legacy integration
    details = event.get("lambdaFunctionScheduledEventDetails", {})
    return details.get("resource", "")


def _find_state_entry(event: dict, by_id: dict[int, dict]) -> Optional[dict]:
    """
    Walk up the previousEventId chain from a TaskScheduled event to the
    enclosing TaskStateEntered event, which carries the state name.
    Works for both regular states and Map iterator states.

    Every retry attempt of one state occurrence chains back to the SAME
    TaskStateEntered (verified against real execution history), so this event's
    id identifies the occurrence — which is what separates "3 sequential
    attempts of one iterator" from "3 concurrent iterators" when computing
    wall time.
    """
    current_id = event.get("previousEventId")
    seen = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if not parent:
            break
        if parent["type"] == "TaskStateEntered":
            return parent
        current_id = parent.get("previousEventId")
    return None


def _find_state_name(event: dict, by_id: dict[int, dict]) -> Optional[str]:
    """State name of the TaskStateEntered enclosing this scheduled event."""
    parent = _find_state_entry(event, by_id)
    if parent is None:
        return None
    return parent.get("stateEnteredEventDetails", {}).get("name")


def _find_state_entered_id(event: dict, by_id: dict[int, dict]):
    """Id of the enclosing TaskStateEntered — the state OCCURRENCE key."""
    parent = _find_state_entry(event, by_id)
    return None if parent is None else parent.get("id")


def _find_scheduled_id(
    event:           dict,
    by_id:           dict[int, dict],
    scheduled_types: set[str],
) -> Optional[int]:
    """
    Walk up the previousEventId chain from a Succeeded/Failed event
    to find the corresponding TaskScheduled (or LambdaFunctionScheduled)
    event id.
    """
    current_id = event.get("previousEventId")
    seen: set[int] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if not parent:
            break
        if parent["type"] in scheduled_types:
            return parent["id"]
        current_id = parent.get("previousEventId")
    return None


def _parse_execution(
    events:           list[dict],
    lambda_cfg_cache: dict[str, dict],
) -> list[dict]:
    """
    Parse Step Functions execution history events into per-state summaries.

    Returns a list of state dicts ordered by state entry time:
    [
        {
            "state_name":       str,
            "operation":        str,
            "lambda_arn":       str,
            "architecture":     str,
            "memory_mb":        int,
            "invocations":      int,
            "wall_sec":         float,   # max duration (parallel Map state)
            "total_billed_sec": float,   # sum duration (each iterator billed)
            "estimated_cost":   float,
        },
        ...
    ]
    """
    _SCHEDULED_TYPES = {"TaskScheduled",  "LambdaFunctionScheduled"}
    _SUCCEEDED_TYPES = {"TaskSucceeded",  "LambdaFunctionSucceeded"}
    _FAILED_TYPES = {"TaskFailed",     "LambdaFunctionFailed"}

    # Index all events by id for O(1) chain walks
    by_id: dict[int, dict] = {e["id"]: e for e in events}

    # First pass: collect all scheduled events
    scheduled: dict[int, dict] = {}
    for e in events:
        if e["type"] not in _SCHEDULED_TYPES:
            continue
        arn = _get_arn_from_scheduled(e)
        if not arn:
            # Non-Lambda task (e.g. AWS SDK S3, DynamoDB, etc.)
            # No FunctionName in parameters — skip, nothing to cost-track.
            log.debug(
                "Skipping non-Lambda scheduled event id=%d state=%s",
                e["id"],
                _find_state_name(e, by_id) or "unknown",
            )
            continue
        state_name = _find_state_name(e, by_id)
        scheduled[e["id"]] = {
            "arn":          arn,
            "scheduled_ts": _ts_to_epoch(e["timestamp"]),
            "succeeded_ts": None,
            "submitted_ts": None,
            "failed":       False,
            "failed_ts":    None,
            # Identifies the state OCCURRENCE, so
            # sequential retry attempts of one Map iterator
            # are not mistaken for concurrent iterators.
            "entered_id":   _find_state_entered_id(e, by_id),
            "state_name":   state_name,
            "operation":    _get_operation_from_input(e),
        }

    # Second pass: match Succeeded/Failed to their Scheduled event
    for e in events:
        if e["type"] in _SUCCEEDED_TYPES:
            sched_id = _find_scheduled_id(e, by_id, _SCHEDULED_TYPES)
            if sched_id in scheduled:
                scheduled[sched_id]["succeeded_ts"] = _ts_to_epoch(
                    e["timestamp"])
        elif e["type"] in _FAILED_TYPES:
            sched_id = _find_scheduled_id(e, by_id, _SCHEDULED_TYPES)
            if sched_id in scheduled:
                scheduled[sched_id]["failed"] = True
                # A failed attempt is billed by AWS, so it needs an end
                # timestamp like any other. `failed` alone was recorded here
                # and never read by anything — see _billed_sec.
                scheduled[sched_id]["failed_ts"] = _ts_to_epoch(e["timestamp"])
        elif e["type"] == "TaskSubmitted":
            # a waitForTaskToken Lambda has returned; the token is now held
            sched_id = _find_scheduled_id(e, by_id, _SCHEDULED_TYPES)
            if sched_id in scheduled:
                scheduled[sched_id]["submitted_ts"] = _ts_to_epoch(
                    e["timestamp"])

    # Group by state name
    from collections import defaultdict
    by_state: dict[str, list[dict]] = defaultdict(list)
    for inv in scheduled.values():
        if inv["state_name"]:
            by_state[inv["state_name"]].append(inv)

    # Preserve chronological state order (first scheduled timestamp)
    state_order = sorted(
        by_state.keys(),
        key=lambda s: min(inv["scheduled_ts"] for inv in by_state[s]),
    )

    results = []
    for state_name in state_order:
        invocations = by_state[state_name]
        n = len(invocations)
        arn = invocations[0]["arn"]
        cfg = _get_lambda_config(arn, lambda_cfg_cache)

        # Billed duration per invocation ATTEMPT, failures included.
        durations = [_billed_sec(inv) for inv in invocations]

        # Wall-clock = max (parallel Map iterations run concurrently).
        # Retry attempts are SEQUENTIAL, not concurrent, so they must be summed
        # before the max — otherwise a state retried 3x reports its longest
        # single attempt as its wall time. Grouping by the state-entry event id
        # does this exactly: every attempt of one Map iterator chains back to
        # the same TaskStateEntered, so each group is one iterator's full
        # sequence of attempts.
        _per_entry: dict = defaultdict(float)
        for _i, inv in enumerate(invocations):
            _per_entry[inv.get("entered_id") or ("__none__", _i)] += \
                _attempt_span_sec(inv)
        wall_sec = round(max(_per_entry.values(), default=0.0), 3)

        # Total billed = sum over every attempt (each is billed independently)
        total_sec = round(sum(durations), 3)

        # True when some attempt had no terminal event at all — normally just
        # this state's own in-flight record. NOT the retry case any more.
        incomplete = any(
            inv["succeeded_ts"] is None and inv.get("failed_ts") is None
            for inv in invocations
        )

        cost = _estimated_cost(
            memory_mb=cfg["memory_mb"],
            duration_sec=total_sec,
            architecture=cfg["architecture"],
            n_invocations=n,
        )

        results.append({
            "state_name":       state_name,
            "operation":        invocations[0].get("operation") or state_name,
            "lambda_arn":       arn,
            "architecture":     cfg["architecture"],
            "memory_mb":        cfg["memory_mb"],
            "invocations":      n,
            # Attempts that ended in TaskFailed. Non-zero is normal for a
            # checkpoint/resume state: it raises on purpose and the ASL
            # Retry re-invokes. Surfaced so a reader can see WHY a
            # state's billed time exceeds its wall time.
            "retried":          sum(1 for i in invocations if i["failed"]),
            "wall_sec":         wall_sec,
            "total_billed_sec": total_sec,
            "incomplete":       incomplete,
            "estimated_cost":   cost,
        })

    return results


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateUsageTracking event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Unpack ────────────────────────────────────────────────────────────
        data = event.get("data",  {})
        uuid_str = event.get("uuid",  "unknown")
        video = data.get("video", {})

        proxy_bucket = data["proxy_bucket"]
        proxy_prefix = data["proxy_prefix"]

        # actual_frame_count and actual_duration_sec are set by dynamic
        # chunking from pts ground truth and propagated through data.video
        # for the lifetime of the execution.
        actual_frame_count = video.get("actual_frame_count")
        actual_duration_sec = video.get("actual_duration_sec")

        # ── execution_arn — always explicit from state machine ─────────────────
        # State machine must pass: "execution_arn.$": "$$Execution.Id"
        execution_arn = event.get("execution_arn")
        if not execution_arn:
            raise ValueError(
                "execution_arn not found in event payload — "
                "ensure the state machine Parameters block includes: "
                "\"execution_arn.$\": \"$$Execution.Id\""
            )
        log.info("Execution ARN: %s", execution_arn)

        # ── Lambda config cache — local to this invocation ────────────────────
        lambda_cfg_cache: dict[str, dict] = {}

        # ── Fetch execution history ───────────────────────────────────────────
        t_hist = time.monotonic()
        history_events = _get_execution_history(execution_arn)
        hist_ms = round((time.monotonic() - t_hist) * 1_000, 1)
        log.info(
            "History fetched: %d events  %.0f ms",
            len(history_events), hist_ms,
        )

        # ── Parse into per-state summaries ────────────────────────────────────
        state_summaries = _parse_execution(history_events, lambda_cfg_cache)

        # ── Estimate usage tracking state itself ──────────────────────────────
        # This state's Succeeded event has not been emitted yet.
        # Find the TaskScheduled timestamp for this state from execution
        # history and subtract from time.time(). This is consistent with
        # how all other states are measured and correctly includes Lambda
        # cold start and SFN scheduling overhead.
        # Iterate in reverse to find the most recent matching event first —
        # handles retried executions where an older StateUsageTracking
        # scheduled event may also exist in history.
        self_scheduled_ts: float | None = None
        for e in reversed(history_events):
            if e["type"] not in ("TaskScheduled", "LambdaFunctionScheduled"):
                continue
            if not _get_arn_from_scheduled(e):
                continue
            if _get_operation_from_input(e) == "StateUsageTracking":
                self_scheduled_ts = _ts_to_epoch(e["timestamp"])
                break

        if self_scheduled_ts is not None:
            estimated_self_sec = round(time.time() - self_scheduled_ts, 3)
            log.info(
                "Usage tracking self-timing: "
                "scheduled_ts=%.3f  now=%.3f  estimated=%.3fs",
                self_scheduled_ts, time.time(), estimated_self_sec,
            )
        else:
            # Fallback — scheduled event not found (should not happen in
            # normal execution). Use internal elapsed time as best estimate.
            log.warning(
                "Usage tracking: TaskScheduled event not found in history — "
                "falling back to internal elapsed time"
            )
            estimated_self_sec = round(time.monotonic() - t0, 3)

        self_arn = (
            context.invoked_function_arn
            if context else
            next(
                (s["lambda_arn"] for s in state_summaries
                 if s["operation"] == "StateUsageTracking"),
                "",
            )
        )
        self_cfg = _get_lambda_config(self_arn, lambda_cfg_cache)
        self_cost = _estimated_cost(
            memory_mb=self_cfg["memory_mb"],
            duration_sec=estimated_self_sec,
            architecture=self_cfg["architecture"],
            n_invocations=1,
        )

        self_summary = {
            "state_name":       "StateUsageTracking",  # overwritten below if found
            "operation":        "StateUsageTracking",
            "lambda_arn":       self_arn,
            "architecture":     self_cfg["architecture"],
            "memory_mb":        self_cfg["memory_mb"],
            "invocations":      1,
            "wall_sec":         estimated_self_sec,
            "total_billed_sec": estimated_self_sec,
            "estimated_cost":   self_cost,
        }

        # Replace the parsed entry for this state if present.
        # Match on 'operation' — state_name is the human-readable state
        # machine key (e.g. "Usage tracking") and varies by deployment.
        # Preserve the parsed state_name so the output is consistent with
        # all other entries.
        existing = next(
            (i for i, s in enumerate(state_summaries)
             if s["operation"] == "StateUsageTracking"),
            None,
        )
        if existing is not None:
            # Carry forward the actual state machine name from the parsed
            # entry (e.g. "Usage tracking") — do not overwrite with the
            # operation name.
            self_summary["state_name"] = state_summaries[existing]["state_name"]
            state_summaries[existing] = self_summary
        else:
            state_summaries.append(self_summary)

        # ── Totals ────────────────────────────────────────────────────────────
        total_cost = round(
            sum(s["estimated_cost"] for s in state_summaries), 8)

        # Pipeline wall time from execution history timestamps
        exec_ts = [
            _ts_to_epoch(e["timestamp"])
            for e in history_events
            if e["type"] in ("ExecutionStarted", "ExecutionSucceeded")
        ]
        if len(exec_ts) >= 2:
            pipeline_wall_sec = round(exec_ts[-1] - exec_ts[0], 3)
        else:
            pipeline_wall_sec = round(
                sum(s["wall_sec"] for s in state_summaries), 3)

        # ── Cost-per-minute and cost-per-frame ────────────────────────────────
        cost_per_minute = None
        cost_per_frame = None
        if actual_duration_sec and actual_duration_sec > 0:
            cost_per_minute = round(
                total_cost / (actual_duration_sec / 60.0), 8)
        if actual_frame_count and actual_frame_count > 0:
            cost_per_frame = round(total_cost / actual_frame_count, 10)

        log.info(
            "Cost summary: total=$%.6f  per_min=$%.6f  per_frame=$%.8f",
            total_cost,
            cost_per_minute or 0.0,
            cost_per_frame or 0.0,
        )
        for s in state_summaries:
            log.info(
                "  %-40s  inv=%2d  wall=%6.1fs  billed=%7.1fs  $%.6f",
                s["state_name"], s["invocations"],
                s["wall_sec"], s["total_billed_sec"], s["estimated_cost"],
            )

        # ── Build and upload usage doc ─────────────────────────────────────────
        out_key = _usage_output_key(proxy_prefix)
        usage_doc = {
            "execution_arn":        execution_arn,
            "uuid":                 uuid_str,
            "proxy_bucket":         proxy_bucket,
            "proxy_prefix":          proxy_prefix,
            "pipeline_wall_sec":    pipeline_wall_sec,
            "total_estimated_cost": total_cost,
            "actual_frame_count":   actual_frame_count,
            "actual_duration_sec":  actual_duration_sec,
            "cost_per_minute":      cost_per_minute,
            "cost_per_frame":       cost_per_frame,
            "states":               state_summaries,
        }

        put_json(proxy_bucket, out_key, usage_doc)
        log.info("Uploaded: s3://%s/%s", proxy_bucket, out_key)

        # ── Return enriched payload ───────────────────────────────────────────
        elapsed_sec = round(time.monotonic() - t0, 2)
        log.info("DONE  elapsed=%.1fs", elapsed_sec)

        return {
            **event,
            "data": {
                **data,
                "usage_tracking": usage_doc,
            },
        }

    except Exception as exc:
        log.error(
            "[ERR] StateUsageTracking — %s: %s",
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise
