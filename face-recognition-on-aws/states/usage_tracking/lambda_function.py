"""
StateUsageTracking — per-state Lambda cost + child media-processing cost +
Rekognition API cost (README.md §4).

Lambda cost: walk the execution history, price each Lambda invocation ATTEMPT
at its OWN function's memory (us-east-1 arm64 figures as of 2025 — same caveat
as the sibling repos; Step Functions scheduling latency of ~10-50 ms inflates
estimates slightly).

Only Lambda invocations are billed here. Every state still appears in the
report, tagged with `kind`, but a non-Lambda state carries
`estimated_cost: 0.0` — see `_KIND_*` below and README.md Gotchas
(usage-tracking cost model).
This replaced a state-span model that charged `TaskStateEntered`→`Exited`
wall time for *every* state at the largest memory seen anywhere in the run,
which over-reported this stack by 2.17x on a real execution.

Rekognition cost: API call tallies accumulated through the pipeline
(counter dicts in iterator/cluster/search JSONs) × $0.001 per image call.

Child MP cost: read from MP's own published usage_tracking.json under
its run prefix (derived from media_processing_output.manifest_key) —
no history walk, no cross-machine IAM (2026-09-18).
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

import boto3

from lib.input_schema import CHARGED_API_COUNTER_KEYS
from lib.s3_utils import get_json, parse_s3uri, put_json

logger = logging.getLogger(__name__)

_sfn = boto3.client("stepfunctions")
_lambda = boto3.client("lambda")

# us-east-1 arm64, 2025
_PRICE_PER_GB_SEC = 0.0000133334
_PRICE_PER_REQUEST = 0.0000002
_PRICE_PER_REKOGNITION_IMAGE = 0.001

# The COST report sums exactly the charged counters (user, 2026-09-07:
# uncharged GetCelebrityInfo is deliberately absent here; its count
# stays visible in facemetadata.json's usage block). ONE source of
# truth: CallCounter derives its fields from the same input_schema
# tuples, so a new counter cannot be written upstream yet silently
# dropped by this sum.
_REKOGNITION_COUNTER_KEYS = CHARGED_API_COUNTER_KEYS

_SCHEDULED_TYPES = {"TaskScheduled", "LambdaFunctionScheduled"}
_SUCCEEDED_TYPES = {"TaskSucceeded", "LambdaFunctionSucceeded"}
_FAILED_TYPES = {"TaskFailed", "LambdaFunctionFailed"}

_DEFAULT_MEMORY_MB = 1792

# What a state's span actually costs THIS stack.
#
# lambda          — a real invocation; billed.
# child_execution — `states:startExecution.sync:2`. LEGACY (until 2026-09-18
#                   the MP frame extraction ran this way; only in-flight runs
#                   across the deploy boundary still carry it). Another state
#                   machine's wall clock; its cost is reported separately as
#                   `media_processing_cost`, so pricing the span here
#                   double-counts a different workflow.
# task_token      — `invoke.waitForTaskToken` (Acquire semaphore, and since
#                   2026-09-18 the held Run media processing submit). The span
#                   is scheduled→SendTaskSuccess, i.e. QUEUE WAIT (semaphore)
#                   or the MP run's wall clock (billed separately as
#                   media_processing_cost). Execution history carries no event
#                   for the Lambda's return, so the compute is not
#                   recoverable — reporting the wait honestly and billing 0 is
#                   the accurate choice, and it understates by well under
#                   $0.0001.
# map             — a `Map` envelope. Not an invocation at all; its iterators
#                   are billed individually under their own state names.
_KIND_LAMBDA = "lambda"
_KIND_CHILD = "child_execution"
_KIND_TOKEN = "task_token"  # nosec B105
_KIND_MAP = "map"


def lambda_handler(event: dict, context) -> dict:
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")
    execution_arn = event.get("execution_arn") or ""

    states_cost, wall_sec = _execution_cost(execution_arn)
    mp_out = event["data"].get("media_processing_output") or {}
    mp_arn = mp_out.get("media_processing_arn")
    # MP publishes its own usage_tracking.json under its run prefix
    # (beside the face_frames group — derivable from manifest_key),
    # so we read its bill instead
    # of walking its execution history (2026-09-18 — this is what let
    # the cross-machine states IAM and the MediaProcessingStateMachineArn
    # parameter go). A missing file after a successful run is a
    # pipeline bug, so get_json is allowed to raise.
    child_cost = 0.0
    if mp_out.get("manifest_key"):
        mp_usage_key = (f"{mp_out['manifest_key'].rsplit('/', 2)[0]}"
                        f"/usage_tracking.json")
        child_cost = get_json(mp_out["frames_bucket"],
                              mp_usage_key)["total_estimated_cost"]

    api_counts = _sum_api_counts(event, out_bucket)
    rek_calls = sum(api_counts.get(k, 0) for k in _REKOGNITION_COUNTER_KEYS)
    rek_cost = rek_calls * _PRICE_PER_REKOGNITION_IMAGE

    lambda_cost = sum(s["estimated_cost"] for s in states_cost.values())
    total = round(lambda_cost + child_cost + rek_cost, 6)

    usage = {
        "pipeline_wall_sec": wall_sec,
        "total_estimated_cost": total,
        "lambda_cost": round(lambda_cost, 6),
        "media_processing_cost": round(child_cost, 6),
        "rekognition_cost": round(rek_cost, 6),
        "rekognition_api_calls": rek_calls,
        **api_counts,
        "states": states_cost,
        "media_processing_arn": mp_arn,
        "usage_key": f"{out_prefix}/usage_tracking.json",
    }
    put_json(out_bucket, f"{out_prefix}/usage_tracking.json", usage)

    event["data"]["usage_tracking"] = {
        "pipeline_wall_sec": wall_sec,
        "total_estimated_cost": total,
        "usage_key": usage["usage_key"],
        **api_counts,
    }
    event["data"]["iterators"] = None
    return event


def _sum_api_counts(event: dict, out_bucket: str) -> dict:
    """
    Sum Rekognition call counters from every state that recorded usage.
    Each key here is only added by the corresponding upstream state AFTER
    it successfully wrote that file (see the `if event["data"].get(...)`
    guards) — so by the time this runs, every key in the list is expected
    to exist. A read failure means real, already-billed API usage is about
    to go unreported in total_estimated_cost — that must raise, not be
    logged-and-ignored (the exact failure mode of the recognize_celebrities
    usage bug: a silently dropped count that nobody could see was wrong).
    """
    total = {k: 0 for k in _REKOGNITION_COUNTER_KEYS}
    keys = []
    if event["data"].get("tracks_key"):
        keys.append(event["data"]["tracks_key"])
    if event["data"].get("clusters_key"):
        keys.append(event["data"]["clusters_key"])
    if event["data"].get("search_index_key"):
        keys.append(event["data"]["search_index_key"])
    for key in keys:
        usage = get_json(out_bucket, key).get("usage") or {}
        for k in _REKOGNITION_COUNTER_KEYS:
            total[k] += int(usage.get(k) or 0)
    return total


def _ts(value) -> datetime:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _lambda_arn_of(event: dict) -> str:
    """
    The Lambda ARN a scheduled event names, or "" if it names no Lambda.

    Two integration styles. The SDK style (`arn:aws:states:::lambda:invoke`,
    which is what every state in definitions.json uses) puts the ARN in
    `parameters.FunctionName`; the legacy style puts it in `resource`.
    `parameters` survives `includeExecutionData=False` — verified against real
    history — so this works on the same cheap page fetch.

    Returns "" for `states:startExecution.sync:2`, which is how a child state
    machine is told apart from a Lambda. The sibling stacks tell them apart the
    same way.
    """
    details = event.get("taskScheduledEventDetails") or {}
    raw = details.get("parameters")
    if raw:
        try:
            return json.loads(raw).get("FunctionName", "") or ""
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    legacy = (event.get("lambdaFunctionScheduledEventDetails") or {}).get(
        "resource") or ""
    # The legacy field is a real function ARN; the SDK-style `resource` is the
    # integration name ("invoke"), never an ARN, so require the arn: prefix.
    return legacy if legacy.startswith("arn:") else ""


def _kind_of(event: dict) -> str:
    details = event.get("taskScheduledEventDetails") or {}
    resource = details.get("resource") or ""
    if details.get("resourceType") == "states" or "startExecution" in resource:
        return _KIND_CHILD
    if "waitForTaskToken" in resource:
        return _KIND_TOKEN
    return _KIND_LAMBDA


def _billed_sec(attempt: dict) -> float:
    """
    Billed seconds for ONE invocation attempt — counting failures.

    AWS bills a Lambda that ran to its timeout and was retried; the most
    expensive invocation a workflow can have is precisely the one that burned
    its full budget and raised. This stack's previous state-span model happened
    to cover retries by construction (an ASL `Retry` does not re-enter the
    state), and that property has to survive the move to per-attempt pricing —
    hence the `failed_ts` fallback rather than reading `succeeded_ts` alone.
    Every checkpoint/resume state here raises `LambdaResourceTimeoutException`
    on purpose, so this is the ordinary path, not a failure-only edge case.

    Falls back to 0.0 only when an attempt has NO terminal event at all: this
    state's own record while it is still running, or an attempt whose terminal
    event is past the end of the history window.

    See README.md Gotchas (usage-tracking cost model).
    """
    end = attempt["succeeded_ts"]
    if end is None:
        end = attempt["failed_ts"]
    if end is None:
        return 0.0
    return (end - attempt["scheduled_ts"]).total_seconds()


def _execution_cost(execution_arn: str) -> tuple[dict, float]:
    """
    Per-state cost from execution history, priced per Lambda invocation.

    Non-Lambda states are reported with their span but billed 0.0 — see the
    `_KIND_*` comments for why each one is not this stack's compute.
    """
    events: list[dict] = []
    # includeExecutionData=False strips per-event input/output payloads —
    # this walk reads only type/timestamp/state-name/resource, and a 2 h run
    # has ~7000 events whose full payloads total tens of MB per execution.
    kwargs: dict = {"executionArn": execution_arn, "maxResults": 1000,
                    "includeExecutionData": False}
    while True:
        page = _sfn.get_execution_history(**kwargs)
        events.extend(page["events"])
        token = page.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token

    by_id = {e["id"]: e for e in events}
    started = None
    # Span per state occurrence, keyed by the entry event's id. Kept because
    # `duration_sec` is a wall-clock the report has always carried, and because
    # it is the only measurement available for the states that are not billed.
    spans: dict[int, dict] = {}
    attempts: dict[int, dict] = {}
    memory_cache: dict[str, int] = {}

    for e in events:
        etype = e["type"]
        if etype == "ExecutionStarted":
            started = _ts(e["timestamp"])
        elif etype in ("TaskStateEntered", "MapStateEntered"):
            spans[e["id"]] = {
                "name": e["stateEnteredEventDetails"]["name"],
                "entered_ts": _ts(e["timestamp"]),
                "exited_ts": None,
                # A Map envelope is identified by its own event type; a Task's
                # kind comes from the integration on its TaskScheduled event.
                "kind": _KIND_MAP if etype == "MapStateEntered" else None,
            }
        elif etype in ("TaskStateExited", "MapStateExited"):
            entry = _walk_back(e, by_id, {"TaskStateEntered", "MapStateEntered"})
            if entry is not None and entry["id"] in spans:
                spans[entry["id"]]["exited_ts"] = _ts(e["timestamp"])
        elif etype in _SCHEDULED_TYPES:
            entry = _walk_back(e, by_id, {"TaskStateEntered"})
            entry_id = entry["id"] if entry is not None else None
            arn = _lambda_arn_of(e)
            kind = _kind_of(e)
            if entry_id in spans and spans[entry_id]["kind"] is None:
                spans[entry_id]["kind"] = kind
            attempts[e["id"]] = {
                "name": (entry or {}).get(
                    "stateEnteredEventDetails", {}).get("name"),
                # Identifies the state OCCURRENCE: every retry attempt chains
                # back to the SAME entry event, so this separates sequential
                # attempts of one Map iterator from concurrent iterators.
                "entered_id": entry_id,
                "arn": arn,
                "kind": kind,
                "scheduled_ts": _ts(e["timestamp"]),
                "succeeded_ts": None,
                "failed_ts": None,
            }
            if kind == _KIND_LAMBDA and arn and arn not in memory_cache:
                memory_cache[arn] = _memory_of(arn)
        elif etype in _SUCCEEDED_TYPES or etype in _FAILED_TYPES:
            # Each terminal event walks back to its OWN attempt's scheduled
            # event, which is what makes a retried state bill every attempt.
            sched = _walk_back(e, by_id, _SCHEDULED_TYPES)
            if sched is not None and sched["id"] in attempts:
                field = ("succeeded_ts" if etype in _SUCCEEDED_TYPES
                         else "failed_ts")
                attempts[sched["id"]][field] = _ts(e["timestamp"])

    states: dict[str, dict] = {}
    for span in spans.values():
        slot = states.setdefault(span["name"], {
            "kind": span["kind"] or _KIND_LAMBDA,
            "invocations": 0, "retried": 0, "duration_sec": 0.0,
            "billed_sec": 0.0, "memory_mb": None, "estimated_cost": 0.0,
            "incomplete": False,
        })
        if span["exited_ts"] is not None:
            slot["duration_sec"] = round(
                slot["duration_sec"]
                + (span["exited_ts"] - span["entered_ts"]).total_seconds(), 3)

    # Wall-clock per state: the longest single OCCURRENCE. Concurrent Map
    # iterators overlap so the state's wall time is the max, not the sum, and
    # the sum is what `duration_sec` already reports.
    longest: dict[str, float] = defaultdict(float)
    for span in spans.values():
        if span["exited_ts"] is not None:
            longest[span["name"]] = max(
                longest[span["name"]],
                (span["exited_ts"] - span["entered_ts"]).total_seconds())

    for attempt in attempts.values():
        name = attempt["name"]
        if not name:
            continue
        slot = states.setdefault(name, {
            "kind": attempt["kind"], "invocations": 0, "retried": 0,
            "duration_sec": 0.0, "billed_sec": 0.0, "memory_mb": None,
            "estimated_cost": 0.0, "incomplete": False,
        })
        slot["invocations"] += 1
        if attempt["failed_ts"] is not None:
            slot["retried"] += 1
        if attempt["succeeded_ts"] is None and attempt["failed_ts"] is None:
            # No terminal event at all — normally just this state's own
            # in-flight record. NOT the retry case.
            slot["incomplete"] = True
        if attempt["kind"] != _KIND_LAMBDA:
            continue
        memory_mb = memory_cache.get(attempt["arn"], _DEFAULT_MEMORY_MB)
        # Reported per state; a state maps 1:1 to a variant, so a differing
        # value across attempts would mean the ASL routes one state at two
        # sizes and the per-attempt cost below is what stays correct.
        slot["memory_mb"] = memory_mb
        billed = _billed_sec(attempt)
        slot["billed_sec"] = round(slot["billed_sec"] + billed, 3)
        slot["estimated_cost"] = round(
            slot["estimated_cost"]
            + billed * (memory_mb / 1024.0) * _PRICE_PER_GB_SEC
            + _PRICE_PER_REQUEST, 6)

    for name, slot in states.items():
        slot["wall_sec"] = round(longest.get(name, slot["duration_sec"]), 3)

    wall = 0.0
    if started is not None and events:
        wall = round((_ts(events[-1]["timestamp"]) - started).total_seconds(), 2)
    return states, wall


def _walk_back(event: dict, by_id: dict, match_types: set):
    """
    Follow `previousEventId` until an event of one of `match_types` is reached.

    `seen` guards against a cycle: a malformed history would otherwise spin
    until the Lambda timed out, turning a reporting problem into a 15-minute
    bill.
    """
    current_id, seen = event.get("previousEventId"), set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if parent is None:
            return None
        if parent["type"] in match_types:
            return parent
        current_id = parent.get("previousEventId")
    return None


def _memory_of(arn: str) -> int:
    try:
        cfg = _lambda.get_function_configuration(FunctionName=arn)
        return int(cfg.get("MemorySize") or 1792)
    except Exception:
        return 1792
