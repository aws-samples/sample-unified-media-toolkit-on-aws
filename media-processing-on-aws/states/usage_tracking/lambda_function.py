"""
Step Function State: StateUsageTracking
─────────────────────────────────────────────────────────────────────────────
Last state in the media-processing pipeline. Enumerates the current Step
Functions execution history to compute per-state Lambda invocation counts,
durations, and estimated costs, then writes usage_tracking.json to S3.

Architecture (arm64 vs x86_64) and memory are resolved via
GetFunctionConfiguration, cached locally for the duration of this invocation.

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
  Cost is estimated from actual elapsed time measured inside the handler + 1s buffer.

Expected input event
─────────────────────
{
    "operation":     "StateUsageTracking",
    "uuid":          "...",
    "execution_arn": "arn:aws:states:...",
    "input":         {"bucket": "...", "key": "..."},
    "data":          { ... }
}

S3 output
──────────
  <destination.prefix>/usage_tracking.json

Output  (returned to Step Functions)
──────────────────────────────────────
{
    ...original payload...,
    "data": {
        ...original data...,
        "usage_tracking": { ...usage doc... }
    },
    "status": "COMPLETED"
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

from lib.s3_utils import s3 as _s3

_sfn    = boto3.client("stepfunctions")
_lambda = boto3.client("lambda")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)

# ── Pricing (us-east-1, 2025) ──────────────────────────────────────────────────

_PRICING: dict[str, dict] = {
    "x86_64": {"per_gb_second": 0.0000166667, "per_request": 0.0000002},
    "arm64":  {"per_gb_second": 0.0000133334, "per_request": 0.0000002},
}

_SCHEDULED_TYPES = {"TaskScheduled", "LambdaFunctionScheduled"}
_SUCCEEDED_TYPES = {"TaskSucceeded",  "LambdaFunctionSucceeded"}
_FAILED_TYPES    = {"TaskFailed",     "LambdaFunctionFailed"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts_to_epoch(ts) -> float:
    return ts.timestamp() if hasattr(ts, "timestamp") else float(ts)


def _billed_sec(inv: dict) -> float:
    """
    Billed seconds for ONE invocation attempt — counting failures.

    AWS bills a Lambda that ran to its timeout and was retried; the most
    expensive invocation a workflow can have is precisely the one that burned
    its full budget and raised. Reading only `succeeded_ts` used to charge such
    an attempt 0 s, so a run under-reported its own cost by up to 7.4x
    (measured on tts: $0.031 reported against a $0.2318 bill) — and it
    under-reported *worst* exactly when the workflow behaved worst. Chunk
    transcode retries are routine in this stack.

    Falls back to 0.0 only when an attempt has NO terminal event at all: this
    state's own record while it is still running, or an attempt whose terminal
    event is past the end of the history window. Zero is genuinely right there
    and nowhere else.

    See README.md §15.
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


def _get_lambda_config(arn: str, cache: dict) -> dict:
    if arn not in cache:
        resp = _lambda.get_function_configuration(FunctionName=arn)
        cache[arn] = {
            "memory_mb":    resp["MemorySize"],
            "architecture": resp.get("Architectures", ["x86_64"])[0],
        }
    return cache[arn]


def _estimated_cost(memory_mb: int, duration_sec: float,
                    architecture: str, n_invocations: int = 1) -> float:
    p = _PRICING.get(architecture, _PRICING["x86_64"])
    compute  = (memory_mb / 1024.0) * duration_sec * p["per_gb_second"]
    requests = n_invocations * p["per_request"]
    return round(compute + requests, 8)


def _get_execution_history(execution_arn: str) -> list[dict]:
    events, kwargs = [], {"executionArn": execution_arn, "maxResults": 1000}
    while True:
        resp = _sfn.get_execution_history(**kwargs)
        events.extend(resp["events"])
        token = resp.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token
    log.info("execution history: %d events", len(events))
    return events


def _get_operation_from_input(event: dict) -> Optional[str]:
    # SDK integration (arn:aws:states:::lambda:invoke)
    raw = event.get("taskScheduledEventDetails", {}).get("parameters")
    if raw:
        try:
            params  = json.loads(raw)
            payload = params.get("Payload", {})
            if isinstance(payload, str):
                payload = json.loads(payload)
            return payload.get("operation")
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    # Legacy integration
    raw = event.get("lambdaFunctionScheduledEventDetails", {}).get("input")
    if raw:
        try:
            return json.loads(raw).get("operation")
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def _get_arn_from_scheduled(event: dict) -> str:
    raw = event.get("taskScheduledEventDetails", {}).get("parameters")
    if raw:
        try:
            return json.loads(raw).get("FunctionName", "")
        except (json.JSONDecodeError, AttributeError):
            pass
    return event.get("lambdaFunctionScheduledEventDetails", {}).get("resource", "")


def _find_state_entry(event: dict, by_id: dict) -> Optional[dict]:
    """
    The TaskStateEntered enclosing this scheduled event.

    Every retry attempt of one state occurrence chains back to the SAME
    TaskStateEntered (verified against real execution history), so this event's
    id identifies the occurrence — which is what separates "3 sequential
    attempts of one iterator" from "3 concurrent iterators" in wall time.
    """
    current_id, seen = event.get("previousEventId"), set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if not parent:
            break
        if parent["type"] == "TaskStateEntered":
            return parent
        current_id = parent.get("previousEventId")
    return None


def _find_state_name(event: dict, by_id: dict) -> Optional[str]:
    parent = _find_state_entry(event, by_id)
    if parent is None:
        return None
    return parent.get("stateEnteredEventDetails", {}).get("name")


def _find_scheduled_id(event: dict, by_id: dict,
                       scheduled_types: set) -> Optional[int]:
    current_id, seen = event.get("previousEventId"), set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if not parent:
            break
        if parent["type"] in scheduled_types:
            return parent["id"]
        current_id = parent.get("previousEventId")
    return None


# ── Execution history parser ───────────────────────────────────────────────────

def _parse_execution(events: list[dict], lambda_cfg_cache: dict) -> list[dict]:
    by_id: dict[int, dict] = {e["id"]: e for e in events}
    scheduled: dict[int, dict] = {}

    for e in events:
        if e["type"] not in _SCHEDULED_TYPES:
            continue
        entry = _find_state_entry(e, by_id)
        scheduled[e["id"]] = {
            "arn":          _get_arn_from_scheduled(e),
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

    for e in events:
        if e["type"] in _SUCCEEDED_TYPES:
            sched_id = _find_scheduled_id(e, by_id, _SCHEDULED_TYPES)
            if sched_id in scheduled:
                scheduled[sched_id]["succeeded_ts"] = _ts_to_epoch(e["timestamp"])
        elif e["type"] in _FAILED_TYPES:
            # Previously not tracked at all here: a failed attempt is billed by
            # AWS, and chunk-transcode retries are routine in this stack.
            sched_id = _find_scheduled_id(e, by_id, _SCHEDULED_TYPES)
            if sched_id in scheduled:
                scheduled[sched_id]["failed"] = True
                scheduled[sched_id]["failed_ts"] = _ts_to_epoch(e["timestamp"])
        elif e["type"] == "TaskSubmitted":
            # a waitForTaskToken Lambda has returned; the token is now held
            sched_id = _find_scheduled_id(e, by_id, _SCHEDULED_TYPES)
            if sched_id in scheduled:
                scheduled[sched_id]["submitted_ts"] = _ts_to_epoch(
                    e["timestamp"])

    by_state: dict[str, list] = defaultdict(list)
    for inv in scheduled.values():
        if inv["state_name"]:
            by_state[inv["state_name"]].append(inv)

    state_order = sorted(
        by_state, key=lambda s: min(i["scheduled_ts"] for i in by_state[s])
    )

    results = []
    for state_name in state_order:
        invocations = by_state[state_name]
        cfg0 = _get_lambda_config(invocations[0]["arn"], lambda_cfg_cache) \
               if invocations[0]["arn"] else {"memory_mb": 1792, "architecture": "arm64"}
        # Billed duration per invocation ATTEMPT, failures included.
        durations = [_billed_sec(i) for i in invocations]

        # Retry attempts are SEQUENTIAL, so sum them per state occurrence
        # before taking the max across concurrent Map iterators.
        _per_entry: dict = defaultdict(float)
        for _n, i in enumerate(invocations):
            _per_entry[i.get("entered_id") or ("__none__", _n)] += _attempt_span_sec(i)
        wall_sec = round(max(_per_entry.values(), default=0.0), 3)

        total_cost = sum(
            _estimated_cost(
                (_get_lambda_config(i["arn"], lambda_cfg_cache) if i["arn"] else cfg0)["memory_mb"],
                _billed_sec(i),
                (_get_lambda_config(i["arn"], lambda_cfg_cache) if i["arn"] else cfg0)["architecture"],
                1,
            )
            for i in invocations
        )
        results.append({
            "state_name":       state_name,
            "operation":        invocations[0].get("operation") or state_name,
            "lambda_arn":       invocations[0]["arn"],
            "architecture":     cfg0["architecture"],
            "memory_mb":        cfg0["memory_mb"],
            "invocations":      len(invocations),
            # Attempts that ended in TaskFailed — routine here, since chunk
            # transcode retries are normal. Surfaced so a reader can see why a
            # state's billed time exceeds its wall time.
            "retried":          sum(1 for i in invocations if i["failed"]),
            "wall_sec":         wall_sec,
            "total_billed_sec": round(sum(durations), 3),
            # True only when an attempt had NO terminal event at all (this
            # state's own in-flight record), not for the retry case.
            "incomplete":       any(
                i["succeeded_ts"] is None and i.get("failed_ts") is None
                for i in invocations
            ),
            "estimated_cost":   round(total_cost, 8),
        })
    return results


# ── Lambda handler ─────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    log.info("StateUsageTracking event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        inp  = event.get("input", {})
        data = event.get("data",  {})
        dest = data.get("destination") or inp.get("destination") or {}

        bucket     = dest.get("bucket")
        if not bucket:
            raise ValueError("Cannot resolve output bucket — data.destination missing")
        out_prefix = dest.get("prefix") or event.get("uuid", "unknown")
        s3uri      = inp.get("s3uri", "")
        src_key    = s3uri.split("/", 3)[-1] if s3uri else \
                     inp.get("key") or inp.get("media_key") or ""
        execution_arn = event.get("execution_arn", "")

        if not execution_arn:
            raise ValueError(
                "execution_arn missing — pass it in the event payload")

        lambda_cfg_cache: dict = {}

        history = _get_execution_history(execution_arn)
        state_summaries = _parse_execution(history, lambda_cfg_cache)

        # Estimate this state's own cost — Succeeded event not yet emitted
        estimated_self_sec = round(time.monotonic() - t0 + 1.0, 3)
        self_arn = context.invoked_function_arn if context else ""
        self_cfg = _get_lambda_config(self_arn, lambda_cfg_cache) if self_arn else \
                   {"memory_mb": 1792, "architecture": "arm64"}
        self_summary = {
            "state_name":       "Usage tracking",
            "operation":        "StateUsageTracking",
            "lambda_arn":       self_arn,
            "architecture":     self_cfg["architecture"],
            "memory_mb":        self_cfg["memory_mb"],
            "invocations":      1,
            "wall_sec":         estimated_self_sec,
            "total_billed_sec": estimated_self_sec,
            "estimated_cost":   _estimated_cost(
                self_cfg["memory_mb"], estimated_self_sec,
                self_cfg["architecture"], 1),
        }
        existing = next(
            (i for i, s in enumerate(state_summaries)
             if s["state_name"] == "Usage tracking"), None)
        if existing is not None:
            state_summaries[existing] = self_summary
        else:
            state_summaries.append(self_summary)

        total_cost = round(sum(s["estimated_cost"] for s in state_summaries), 8)

        start_ts = next(
            (_ts_to_epoch(e["timestamp"]) for e in history
             if e["type"] == "ExecutionStarted"), None)
        pipeline_wall_sec = round(time.time() - start_ts, 3) if start_ts \
                            else round(sum(s["wall_sec"] for s in state_summaries), 3)

        usage_doc = {
            "execution_arn":        execution_arn,
            "source_key":           src_key,
            "pipeline_wall_sec":    pipeline_wall_sec,
            "total_estimated_cost": total_cost,
            "states":               state_summaries,
        }

        out_key = f"{out_prefix}/usage_tracking.json"
        _s3.put_object(
            Bucket=bucket, Key=out_key,
            Body=json.dumps(usage_doc, indent=2).encode(),
            ContentType="application/json",
        )
        log.info("uploaded s3://%s/%s  cost=$%.6f", bucket, out_key, total_cost)

        return {
            **event,
            "status": "COMPLETED",
            "data":   {**data, "usage_tracking": usage_doc},
        }

    except Exception as exc:
        log.error("StateUsageTracking — %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise
