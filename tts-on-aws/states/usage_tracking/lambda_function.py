# state_usage_tracking.py
"""
Step Function State: Usage Tracking
─────────────────────────────────────────────────────────────────────────────
Last state in the tts-on-aws synthesis pipeline. Enumerates the current
Step Functions execution history to compute per-state Lambda invocation
counts, durations, and estimated costs.

Architecture (arm64 vs x86_64) and memory are resolved via
GetFunctionConfiguration — cached in a local dict for the duration of the
invocation (never persisted to module level).

Lambda pricing (us-east-1, as of 2025)
───────────────────────────────────────
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
  Cost is estimated from actual elapsed time measured inside the handler.

Expected input event
─────────────────────
{
    "operation": "StateUsageTracking",
    "uuid":      "...",
    "input":     { "bucket": "...", "audio_key": "path/to/audio_en.wav" },
    "data":      { ... }         ← full pipeline data from prior states
}

S3 output
──────────
  path/to/audio/usage_tracking.json

Output  (returned to Step Functions)
──────────────────────────────────────
{
    ...original payload...,
    "data": {
        ...original data...,
        "usage_tracking": { ...usage doc... }
    }
}
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections import defaultdict
from typing import Optional

import boto3

from lib.s3_utils import put_json

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

# First pass: collect all scheduled events (SDK or legacy integration)
_SCHEDULED_TYPES = {"TaskScheduled", "LambdaFunctionScheduled"}
_SUCCEEDED_TYPES = {"TaskSucceeded",  "LambdaFunctionSucceeded"}
_FAILED_TYPES    = {"TaskFailed",     "LambdaFunctionFailed"}

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
            f"  lambda config: {arn.split(':')[-1]}  "
            f"mem={cache[arn]['memory_mb']} MB  "
            f"arch={cache[arn]['architecture']}"
        )
    return cache[arn]


def _estimated_cost(memory_mb: int, duration_sec: float,
                    architecture: str, n_invocations: int = 1) -> float:
    """
    Estimated Lambda cost in USD.
    duration_sec is the total billed seconds across all n_invocations
    (i.e. already summed for Map state).
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
    the full 900 s budget and raised. Reading only `succeeded_ts` used to
    charge such an attempt 0 s, so a run under-reported its own cost by 7.4x
    (measured: $0.031 reported against a $0.2318 bill) — and it under-reported
    *worst* exactly when the workflow behaved worst. Checkpoint/resume makes a
    timeout the NORMAL path at real batch sizes, so this is ordinary-run
    accounting, not a failure-only edge case.

    Falls back to 0.0 only when an attempt has NO terminal event at all: this
    state's own record while it is still running, or an attempt whose terminal
    event is past the end of the history window. Zero is genuinely right there
    and nowhere else.

    See the usage-tracking notes in README.md.
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
    """
    Page through GetExecutionHistory and return all events.
    """
    events = []
    kwargs = {
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
    log.info(f"  execution history: {len(events)} events")
    return events


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
            # Parameters block: {"FunctionName": "...", "Payload": {...}}
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


def _parse_execution(
    events:            list[dict],
    lambda_cfg_cache:  dict[str, dict],
) -> list[dict]:
    """
    Parse Step Functions execution history events into per-state summaries.

    Returns a list of state dicts ordered by state entry time:
    [
        {
            "state_name":      str,
            "operation":       str,
            "lambda_arn":      str,
            "architecture":    str,
            "memory_mb":       int,
            "invocations":     int,
            "duration_sec":    float,   # wall time (max for Map state)
            "estimated_cost":  float,
        },
        ...
    ]
    """
    # ── Index events by id ────────────────────────────────────────────────────
    by_id: dict[int, dict] = {e["id"]: e for e in events}

    scheduled: dict[int, dict] = {}

    for e in events:
        if e["type"] not in _SCHEDULED_TYPES:
            continue
        arn = _get_arn_from_scheduled(e)
        if not arn:
            # Non-Lambda task (e.g. startExecution.sync:2) — skip
            continue
        entry = _find_state_entry(e, by_id)
        scheduled[e["id"]] = {
            "arn":          arn,
            "scheduled_ts": _ts_to_epoch(e["timestamp"]),
            "succeeded_ts": None,
            "submitted_ts": None,
            "failed":       False,
            "failed_ts":    None,
            "state_name":   (entry or {}).get(
                "stateEnteredEventDetails", {}).get("name"),
            # Identifies the state OCCURRENCE, so sequential retry attempts of
            # one Map iterator are not mistaken for concurrent iterators.
            "entered_id":   (entry or {}).get("id"),
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

    # ── Group by state name ───────────────────────────────────────────────────
    by_state: dict[str, list[dict]] = defaultdict(list)
    for inv in scheduled.values():
        if inv["state_name"]:
            by_state[inv["state_name"]].append(inv)

    # ── Build per-state summaries ─────────────────────────────────────────────
    # Preserve state order using first scheduled timestamp
    state_order = sorted(
        by_state.keys(),
        key=lambda s: min(
            inv["scheduled_ts"] for inv in by_state[s]
        )
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
        # before the max — otherwise a state retried 3x reports the longest
        # single attempt as its wall time. Grouping by the state-entry event
        # does this exactly: every attempt of one Map iterator chains back to
        # the same TaskStateEntered (verified against real history), so each
        # group below is one iterator's full sequence of attempts.
        per_entry: dict[object, float] = defaultdict(float)
        for inv in invocations:
            per_entry[inv.get("entered_id", id(inv))] += _attempt_span_sec(inv)
        wall_sec = round(max(per_entry.values(), default=0.0), 3)

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
            "state_name":     state_name,
            "operation":      invocations[0].get("operation") or state_name,
            "lambda_arn":     arn,
            "architecture":   cfg["architecture"],
            "memory_mb":      cfg["memory_mb"],
            "invocations":    n,
            # Attempts that ended in TaskFailed. Non-zero is normal here:
            # checkpoint/resume raises LambdaResourceTimeoutException on
            # purpose and the ASL Retry re-invokes. Surfaced so a reader can
            # see WHY a state's billed time exceeds its wall time.
            "retried":        sum(1 for inv in invocations if inv["failed"]),
            "wall_sec":       wall_sec,
            "total_billed_sec": total_sec,
            "incomplete":     incomplete,
            "estimated_cost": cost,
        })

    return results


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


def _find_scheduled_id(
    event:           dict,
    by_id:           dict[int, dict],
    scheduled_types: set[str],
) -> Optional[int]:
    """
    Walk up the previousEventId chain from a Succeeded/Failed event
    to find the corresponding TaskScheduled (or LambdaFunctionScheduled) event id.
    """
    current_id = event.get("previousEventId")
    seen = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if not parent:
            break
        if parent["type"] in scheduled_types:
            return parent["id"]
        current_id = parent.get("previousEventId")
    return None


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateUsageTracking event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        data = event.get("data", {})
        inp = event.get("input", {})
        output_bucket = inp.get("out_bucket") or os.environ["PROXY_BUCKET"]
        output_prefix = (inp.get("out_prefix")
                         or f"micro-workflow/tts-on-aws/output/{event['uuid']}").rstrip("/")

        # ── Execution ARN ─────────────────────────────────────────────────────
        # The Lambda context does NOT carry it — there is no execution ARN in
        # the Lambda context object for a Task-state invocation, and no
        # environment variable holds it either. The only source is the ASL,
        # which passes "execution_arn.$": "$$.Execution.Id" in this state's
        # Parameters (both definitions.json and the inline DefinitionString).
        execution_arn = event.get("execution_arn")
        if not execution_arn:
            raise ValueError(
                "Could not determine Step Functions execution ARN. The "
                "'Usage tracking' state must pass "
                "\"execution_arn.$\": \"$$.Execution.Id\" in its Parameters."
            )

        log.info(f"Execution ARN: {execution_arn}")

        # ── Lambda config cache — local to this invocation ────────────────────
        lambda_cfg_cache: dict[str, dict] = {}

        # ── Fetch execution history ───────────────────────────────────────────
        t_hist = time.monotonic()
        events = _get_execution_history(execution_arn)
        hist_ms = round((time.monotonic() - t_hist) * 1_000, 1)
        log.info(f"  history fetched: {hist_ms:.0f} ms")

        # ── Parse into per-state summaries ────────────────────────────────────
        state_summaries = _parse_execution(events, lambda_cfg_cache)

        # (No child-execution accounting — tts-on-aws invokes no child
        # state machines, unlike diarization's media-processing path.)
        child_usage: dict | None = None

        # ── Estimate usage tracking state itself ──────────────────────────────
        # This state's Succeeded event hasn't been emitted yet.
        # Estimate: time elapsed so far + a small buffer for S3 write.
        elapsed_so_far = time.monotonic() - t0
        # +1s buffer for S3 write
        estimated_self_sec = round(elapsed_so_far + 1.0, 3)

        self_cfg = _get_lambda_config(
            context.invoked_function_arn if context else
            next((s["lambda_arn"] for s in state_summaries
                  if s["operation"] == "StateUsageTracking"), ""),
            lambda_cfg_cache,
        )
        self_cost = _estimated_cost(
            memory_mb=self_cfg["memory_mb"],
            duration_sec=estimated_self_sec,
            architecture=self_cfg["architecture"],
            n_invocations=1,
        )

        # Update or append usage tracking state entry
        self_summary = {
            "state_name":       "Usage tracking",
            "operation":        "StateUsageTracking",
            "lambda_arn":       context.invoked_function_arn if context else "",
            "architecture":     self_cfg["architecture"],
            "memory_mb":        self_cfg["memory_mb"],
            "invocations":      1,
            "wall_sec":         estimated_self_sec,
            "total_billed_sec": estimated_self_sec,
            "estimated_cost":   self_cost,
        }

        # Replace if already parsed from history (unlikely), else append
        existing = next(
            (i for i, s in enumerate(state_summaries)
             if s["state_name"] == "Usage tracking"), None)
        if existing is not None:
            state_summaries[existing] = self_summary
        else:
            state_summaries.append(self_summary)

        # ── Totals ────────────────────────────────────────────────────────────
        total_cost = round(
            sum(s["estimated_cost"] for s in state_summaries)
            + (child_usage["estimated_cost"] if child_usage else 0.0),
            8,
        )

        # Wall time = ExecutionStarted → now. It cannot be measured
        # ExecutionStarted→ExecutionSucceeded: this state IS still running, so
        # ExecutionSucceeded is not in the history it just fetched and never
        # will be. "now" is the honest end point (it undercounts only this
        # state's own remaining S3 write, already covered by estimated_self_sec).
        #
        # Fall back to summing per-state wall time only if ExecutionStarted is
        # somehow absent — that sum is NOT the same quantity: Map iterators run
        # concurrently, so summing them overcounts, and it excludes SFN
        # scheduling gaps between states.
        started = next((_ts_to_epoch(e["timestamp"]) for e in events
                        if e["type"] == "ExecutionStarted"), None)
        if started is not None:
            pipeline_wall_sec = round(time.time() - started, 3)
        else:
            log.warning("ExecutionStarted not found in history — falling back "
                        "to the sum of per-state wall times (overcounts Map "
                        "concurrency, excludes inter-state gaps)")
            pipeline_wall_sec = round(
                sum(s["wall_sec"] for s in state_summaries), 3)

        # ── Upload usage_tracking.json ────────────────────────────────────────
        out_key = f"{output_prefix}/usage_tracking.json"

        usage_doc = {
            "execution_arn":        execution_arn,
            "output_bucket":        output_bucket,
            "usage_key":            out_key,
            "pipeline_wall_sec":    pipeline_wall_sec,
            "total_estimated_cost": total_cost,
            "states":               state_summaries,
        }

        put_json(output_bucket, out_key, usage_doc)
        log.info(f"  uploaded: s3://{output_bucket}/{out_key}")

        # ── Return enriched payload ───────────────────────────────────────────
        elapsed_sec = round(time.monotonic() - t0, 2)
        log.info(
            f"DONE  elapsed={elapsed_sec}s  "
            f"total_estimated_cost=${total_cost:.6f}"
        )

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
