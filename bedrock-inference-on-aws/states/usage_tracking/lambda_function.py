"""
StateUsageTracking — Lambda cost across the orchestrator AND every descendant
(README.md §M4).

The orchestrator's last state. Walks the whole execution TREE — its own history, the
child it started, and anything that child started in turn — prices each Lambda
invocation ATTEMPT at its own function's memory (us-east-1 arm64 figures as of 2025;
Step Functions scheduling latency of ~10-50 ms inflates estimates slightly), and writes
`usage_tracking.json` beside the summary.

**The whole tree, because one history would report near-zero.** The orchestrator's own
history is two short Lambda calls and one `.sync:2` span. Every Converse retry, every
checkpoint re-entry, and the whole lease wait live in the child. A rollup that read only
`$$.Execution.Id` would attribute a two-hour run to about four seconds of compute.

**And it is a TREE, not a pair, since the batch repair tail (§8).** A repaired batch run
is orchestrator → batch child → sync child, and that grandchild is where the repair's
Converse calls and its lease wait are billed — the most expensive part of exactly the
run that went worst. So the descendants are DISCOVERED rather than passed in: each
`.sync:2` state's `TaskSubmitted` event carries the `ExecutionArn` it started, and the
walk recurses on it. A scalar `child_execution_arn` parameter used to be threaded from
the orchestrator's `ResultSelector`; it could name only one execution, and it named the
one edge the discovery already covers.

**Only Lambda invocations are billed.** Every state still appears in the report,
tagged with `kind`, but a non-Lambda state carries `estimated_cost: 0.0` — see the
`_KIND_*` comments below. This is face-recognition's model rather than the state-span
model the other seven stacks use, and the choice is forced by this stack's shape: a
state-span model charges `TaskStateEntered`→`Exited` wall time for every state, which
here means billing a full `.sync:2` child span as Lambda time on top of the child's
own itemised history, plus billing a 24-hour lease queue wait as compute. The same
model over-reported face-recognition by 2.17× on a real execution for the same reason.

Bedrock token cost is NOT recomputed here. `summary.bedrock_usage.estimated_cost` is
authoritative — it is derived per record from actual token counts, carries
`rounds_unpriced`/`cost_is_complete`, and is written by whichever child ran. This
state reports it alongside its own Lambda figure and sums the two; recomputing it
from the response lines would be a second implementation of the same arithmetic that
could silently disagree.

**Success path only.** There is no failure-path twin, and the state that used to be one
is gone. A failed run's Lambda time is worth well under a cent — every function here is
1024-2048 MB, so even the worst case (`Run inference` burning all ten 900 s timeout
retries) is about $0.12 — and it is an *estimate* from history timestamps, not billing
data. The money that matters is Bedrock tokens, and those already survive a failure
without this module: `run_inference` checkpoints a full priced response line per record
into `_progress.json` before it can fail. Buying a sub-cent estimate cost a `Catch` that
laundered every distinct child error into one static `Fail`, which is the trade that was
not worth making.

**This handler never raises** — for a different reason than it used to. It is now the
orchestrator's TERMINAL state, so an exception here would fail an execution whose
`response.jsonl` and `summary.json` are already written and whose records all succeeded,
turning a completed run into a FAILED one over a cost report. So: log loudly, return a
partial figure with `complete: False`, and let the execution succeed.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime

import boto3

from lib import s3_utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_SFN = None
_LAMBDA = None


def _sfn():
    """Lazy — a client built at import time would make an offline suite pass or fail
    according to the developer's shell."""
    global _SFN
    if _SFN is None:
        _SFN = boto3.client("stepfunctions")
    return _SFN


def _lambda():
    global _LAMBDA
    if _LAMBDA is None:
        _LAMBDA = boto3.client("lambda")
    return _LAMBDA


# us-east-1 arm64, 2025
_PRICE_PER_GB_SEC = 0.0000133334
_PRICE_PER_REQUEST = 0.0000002

_SCHEDULED_TYPES = {"TaskScheduled", "LambdaFunctionScheduled"}
_SUCCEEDED_TYPES = {"TaskSucceeded", "LambdaFunctionSucceeded"}
_FAILED_TYPES = {"TaskFailed", "LambdaFunctionFailed"}

# The tier ladder spreads the workflow's functions over 896–10240 MB, and 3584 (the
# 2-cores tier) is where all but one of the states live — so it is the likeliest
# fallback when a `GetFunctionConfiguration` fails. This number matters more since the
# collapse onto tiers than it did when everything was 1024: a wrong fallback used to be
# wrong by a rounding error and can now be wrong by 4x.
_DEFAULT_MEMORY_MB = 3584

# What a state's span actually costs this stack.
#
# lambda          — a real invocation; billed.
# child_execution — `states:startExecution.sync:2`. Another state machine's wall
#                   clock. Its Lambda cost is reported separately from its own
#                   history walk, so pricing the span here double-counts it. This
#                   is the orchestrator's two branch states and the batch child's
#                   repair tail. These are also the states that carry
#                   `child_execution_arns`, which is what the tree walk recurses on.
# task_token      — `invoke.waitForTaskToken`: `Acquire lease` in the sync child and
#                   `Submit batch` in the batch child. The span is
#                   scheduled→SendTaskSuccess, i.e. mostly QUEUE WAIT — a lease wait
#                   is unbounded under contention and a batch park is measured in
#                   hours, while the Lambda's own compute stays ~1 s. Execution
#                   history carries no event for the Lambda's return, so that compute
#                   is not recoverable; reporting the wait honestly and billing 0 is
#                   the accurate choice and understates by well under $0.0001.
# map             — a `Map` envelope. There is no Map in any of the three ASLs
#                   (consumer contract 6) and a test asserts there never is; the kind
#                   exists so that adding one shows up as an unbilled envelope with
#                   its iterators billed individually, rather than as a silent gap.
# Bounds on the child walk. Depth 2 is what the start graph actually needs
# (orchestrator → batch child → repair grandchild); 3 leaves one level of headroom so a
# new tier is under-reported rather than silently truncated. See `_descendants`.
_MAX_CHILD_DEPTH = 3
_MAX_CHILD_EXECUTIONS = 8

_KIND_LAMBDA = "lambda"
_KIND_CHILD = "child_execution"
_KIND_TOKEN = "task_token"  # nosec B105
_KIND_MAP = "map"


def lambda_handler(event: dict, context) -> dict:
    data = event.get("data") or {}
    execution_arn = event.get("execution_arn") or ""

    usage, detail = _usage(data, execution_arn)

    return {
        "operation": "StateUsageTracking",
        "status": "COMPLETED",
        "uuid": event.get("uuid") or "",
        "input": event.get("input") or {},
        "data": {
            **data,
            "usage_tracking": usage,
            "usage_tracking_detail": detail,
        },
    }


def _usage(data: dict, execution_arn: str) -> tuple[dict, str]:
    """
    The whole body is guarded: see the module docstring. A partial `usage_tracking`
    with `complete: False` is strictly better than an exception that hides why the
    run failed.
    """
    try:
        return _build_usage(data, execution_arn), "ok"
    except Exception as exc:                                       # noqa: BLE001
        logger.error(
            "USAGE TRACKING FAILED for %s: %s — the run's outcome is unaffected and "
            "summary.json is authoritative for Bedrock cost", execution_arn, exc)
        return ({"complete": False,
                 "reason": f"{type(exc).__name__}: {exc}"},
                f"{type(exc).__name__}: {exc}")


def _build_usage(data: dict, execution_arn: str) -> dict:
    orchestrator_states, wall_sec = _execution_cost(execution_arn)
    children = _descendants(orchestrator_states)

    orchestrator_cost = sum(s["estimated_cost"] for s in orchestrator_states.values())
    child_cost = sum(s["estimated_cost"]
                     for child in children.values()
                     for s in child["states"].values())
    # The LONGEST descendant, not the sum. A repaired batch run nests the repair
    # grandchild's span inside the batch child's, so summing double-counts it.
    child_wall = max((child["wall_sec"] for child in children.values()), default=0.0)
    lambda_cost = orchestrator_cost + child_cost

    # summary.bedrock_usage is authoritative and is NOT recomputed here. `{}` when the
    # summary cannot be read, in which case `complete` says so rather than a zero
    # passing for a priced figure.
    bedrock = _bedrock_usage(data)
    bedrock_cost = float(bedrock.get("estimated_cost") or 0.0)
    bedrock_complete = bool(bedrock) and bool(bedrock.get("cost_is_complete"))

    usage = {
        "complete": bedrock_complete,
        "pipeline_wall_sec": wall_sec,
        "child_wall_sec": child_wall,
        "total_estimated_cost": round(lambda_cost + bedrock_cost, 6),
        "lambda_cost": round(lambda_cost, 6),
        "orchestrator_lambda_cost": round(orchestrator_cost, 6),
        "child_lambda_cost": round(child_cost, 6),
        "bedrock_cost": round(bedrock_cost, 6),
        "bedrock_cost_is_complete": bedrock_complete,
        "inference_mode": data.get("mode") or "",
        "execution_arn": execution_arn,
        # A LIST, and keyed reporting below, because a repaired batch run has two
        # descendants. A scalar here could name only the first one.
        "child_execution_arns": [child["execution_arn"]
                                 for child in children.values()],
        "states": {"orchestrator": orchestrator_states, "children": children},
    }

    key = _usage_key(data)
    if key:
        bucket = ((data.get("output") or {}).get("bucket")) or ""
        s3_utils.put_json(bucket, key, usage)
        usage["usage_key"] = key
        logger.info("usage: $%.6f total (lambda $%.6f + bedrock $%.6f), %.1f s wall",
                    usage["total_estimated_cost"], usage["lambda_cost"],
                    usage["bedrock_cost"], wall_sec)
    return usage


def _usage_key(data: dict) -> str:
    """
    `{output.prefix}/usage_tracking.json`, or "" when there is no prefix to write to.

    `Validate & plan` always produces a prefix (`parse_request` rejects a request
    without one), so the empty case is a malformed `$.child.data` rather than a normal
    path. Returning "" rather than raising keeps a reporting problem from failing a
    finished run, and inventing a location would scatter orphaned reports across the
    bucket.
    """
    output = data.get("output") or {}
    prefix = (output.get("prefix") or "").rstrip("/")
    if not (output.get("bucket") and prefix):
        return ""
    return f"{prefix}/usage_tracking.json"


def _bedrock_usage(data: dict) -> dict:
    """
    Read `summary.bedrock_usage` from the summary the child already wrote.

    Normally present — this state runs after the child's `Summarize` wrote it. Returns
    `{}` rather than raising when it is not: this is a reporting field, and the caller
    learns the figure is partial from `complete: False`.
    """
    bucket = ((data.get("output") or {}).get("bucket")) or ""
    key = ((data.get("keys") or {}).get("summary")) or ""
    if not (bucket and key):
        return {}
    summary = s3_utils.get_json_if_exists(bucket, key)
    if not summary:
        return {}
    return summary.get("bedrock_usage") or {}


def _ts(value) -> datetime:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _lambda_arn_of(event: dict) -> str:
    """
    The Lambda ARN a scheduled event names, or "" if it names no Lambda.

    Two integration styles. The SDK style (`arn:aws:states:::lambda:invoke`, which is
    what every state in all three definitions uses) puts the ARN in
    `parameters.FunctionName`; the legacy style puts it in `resource`. Both fields are
    event *details* rather than payload, so this worked before the walk started asking
    for `includeExecutionData=True` and is unaffected by it.

    Returns "" for `states:startExecution.sync:2`, which is how a child state machine
    is told apart from a Lambda.
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


def _submitted_execution_arn(event: dict) -> str:
    """
    The child execution arn a `TaskSubmitted` event reports, or "" for anything else.

    `resourceType == "states"` is the discriminator rather than the state's name or its
    `Resource` string: `TaskSubmitted` is emitted by every `.sync` integration this
    stack could grow, and the field is set by Step Functions rather than by our ASL.

    The arn lives in the event's `output`, a JSON **string** — which is why the history
    walk asks for `includeExecutionData=True`. Returns "" rather than raising on a
    malformed payload: an unwalkable child costs an under-reported figure, and this
    module must never fail a finished run over a cost report.
    """
    details = event.get("taskSubmittedEventDetails") or {}
    if details.get("resourceType") != "states":
        return ""
    try:
        return json.loads(details.get("output") or "{}").get("ExecutionArn", "") or ""
    except (json.JSONDecodeError, AttributeError, TypeError):
        logger.warning("usage: unparseable TaskSubmitted output for event %s",
                       event.get("id"))
        return ""


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

    AWS bills a Lambda that ran to its timeout and was retried; the most expensive
    invocation a workflow can have is precisely the one that burned its full budget
    and raised. `Run inference` raises `LambdaResourceTimeoutException` on purpose up
    to ten times per run, so a model that read `succeeded_ts` alone would under-report
    the sync path by nearly its whole compute bill — hence the `failed_ts` fallback.

    Falls back to 0.0 only when an attempt has NO terminal event at all: this state's
    own record while it is still running, or an attempt whose terminal event is past
    the end of the history window.
    """
    end = attempt["succeeded_ts"]
    if end is None:
        end = attempt["failed_ts"]
    if end is None:
        return 0.0
    return (end - attempt["scheduled_ts"]).total_seconds()


def _descendants(root_states: dict) -> dict:
    """
    Every execution the tree contains below the root, keyed by execution NAME.

    Breadth-first from the `child_execution_arns` each `.sync:2` state's own walk
    reported, recursing so a repaired batch run yields both `{id}-batch` and
    `{id}-repair`. Three guards, and each has a live failure behind it:

    - **`seen`.** The start graph is acyclic by construction (orchestrator → either
      child, batch child → sync child, sync child → nothing), but this walk is the one
      place a future cycle would cost 15 minutes of Lambda and a truncated report
      instead of raising.
    - **`_MAX_CHILD_DEPTH`.** Bounds the walk even if a cycle evades `seen` by naming
      the same machine under two execution names.
    - **`_MAX_CHILD_EXECUTIONS`.** Bounds the number of `GetExecutionHistory` page
      walks, so a report can never cost more than the run it is reporting on.

    Keyed by name rather than by arn because the name is the readable half — `-batch`
    vs `-repair` says which machine ran, where two arns differ only in a uuid.
    """
    children: dict[str, dict] = {}
    seen = set()
    frontier = [(arn, 1) for arn in _child_arns_of(root_states)]
    while frontier:
        arn, depth = frontier.pop(0)
        if arn in seen or depth > _MAX_CHILD_DEPTH:
            continue
        seen.add(arn)
        if len(children) >= _MAX_CHILD_EXECUTIONS:
            logger.warning(
                "usage: stopping the child walk at %d executions; %s and any of its "
                "own children are NOT billed in this report",
                _MAX_CHILD_EXECUTIONS, arn)
            break
        states, wall = _execution_cost(arn)
        children[arn.rsplit(":", 1)[-1] or arn] = {
            "execution_arn": arn, "wall_sec": wall, "states": states}
        frontier.extend((child, depth + 1) for child in _child_arns_of(states))
    return children


def _child_arns_of(states: dict) -> list[str]:
    """Every execution arn the `.sync:2` states in one history reported starting."""
    return [arn for slot in states.values()
            for arn in (slot.get("child_execution_arns") or [])]


def _execution_cost(execution_arn: str) -> tuple[dict, float]:
    """
    Per-state cost from execution history, priced per Lambda invocation.

    Non-Lambda states are reported with their span but billed 0.0 — see the `_KIND_*`
    comments for why each one is not this stack's compute.
    """
    if not execution_arn:
        return {}, 0.0

    events: list[dict] = []
    # includeExecutionData=True, and it is NOT the extravagance it looks like in this
    # stack specifically. It is what puts `ExecutionArn` on a `.sync:2` TaskSubmitted
    # event, i.e. it is the whole discovery mechanism for the tree walk. The obvious
    # objection — that a history's full payloads run to tens of MB including every
    # prompt — was measured false here, because the records live in `request.jsonl` on
    # S3 and never in the state payload: two real executions grew 15292 → 49748 and
    # 11902 → 55596 bytes with data included. ~40 KB is not worth a second, cheaper
    # walk plus a scalar arn threaded through the ASL to cover one edge of two.
    kwargs: dict = {"executionArn": execution_arn, "maxResults": 1000,
                    "includeExecutionData": True}
    while True:
        page = _sfn().get_execution_history(**kwargs)
        events.extend(page["events"])
        token = page.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token

    by_id = {e["id"]: e for e in events}
    started = None
    # Span per state occurrence, keyed by the entry event's id. Kept because
    # `duration_sec` is a wall clock worth reporting, and because it is the only
    # measurement available for the states that are not billed.
    spans: dict[int, dict] = {}
    attempts: dict[int, dict] = {}
    memory_cache: dict[str, int] = {}
    # state name → the execution arns its `.sync:2` integration reported starting.
    started_children: dict[str, list[str]] = defaultdict(list)

    for e in events:
        etype = e["type"]
        if etype == "ExecutionStarted":
            started = _ts(e["timestamp"])
        elif etype in ("TaskStateEntered", "MapStateEntered"):
            spans[e["id"]] = {
                "name": e["stateEnteredEventDetails"]["name"],
                "entered_ts": _ts(e["timestamp"]),
                "exited_ts": None,
                # A Map envelope is identified by its own event type; a Task's kind
                # comes from the integration on its TaskScheduled event.
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
                # Identifies the state OCCURRENCE: every retry attempt chains back to
                # the SAME entry event, which is what makes ten `Run inference`
                # re-entries bill as ten attempts of one occurrence.
                "entered_id": entry_id,
                "arn": arn,
                "kind": kind,
                "scheduled_ts": _ts(e["timestamp"]),
                "succeeded_ts": None,
                "failed_ts": None,
            }
            if kind == _KIND_LAMBDA and arn and arn not in memory_cache:
                memory_cache[arn] = _memory_of(arn)
        elif etype == "TaskSubmitted":
            # The ONE event that names a started child execution. `.sync:2` emits it
            # once the child exists and long before it finishes, so this works on a
            # still-running tree as well as a finished one — and it is attributed to
            # the state that started it by walking back to that state's own
            # TaskScheduled, not by matching on the state name.
            sched = _walk_back(e, by_id, _SCHEDULED_TYPES)
            attempt = attempts.get((sched or {}).get("id"))
            arn = _submitted_execution_arn(e)
            if attempt and arn and attempt["name"]:
                started_children[attempt["name"]].append(arn)
        elif etype in _SUCCEEDED_TYPES or etype in _FAILED_TYPES:
            # Each terminal event walks back to its OWN attempt's scheduled event,
            # which is what makes a retried state bill every attempt.
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

    # Wall clock per state: the longest single OCCURRENCE. `duration_sec` already
    # reports the sum, and the two differ whenever a state is entered more than once.
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
        # **Attempts beyond the first, NOT attempts that failed.** Those were the same
        # number until the repair tail landed: with no `Catch` anywhere, a failed
        # attempt was always either retried or fatal. `Collect batch output` now fails
        # on purpose and is CAUGHT, so counting failures reported `retried: 1` on a
        # state that ran exactly once — a phantom flake on every repaired batch run,
        # in the one report a reader consults to find flaky states. Counting
        # re-entries instead is right on every path: a caught failure is 1 - 1 = 0, a
        # checkpoint yield resumed 4 times is 3, and a state that exhausted a
        # 3-attempt ladder is 2 rather than an inflated 3.
        slot["retried"] = max(0, slot["invocations"] - 1)
        if attempt["succeeded_ts"] is None and attempt["failed_ts"] is None:
            # No terminal event at all — normally just this state's own in-flight
            # record. NOT the retry case.
            slot["incomplete"] = True
        if attempt["kind"] != _KIND_LAMBDA:
            continue
        memory_mb = memory_cache.get(attempt["arn"], _DEFAULT_MEMORY_MB)
        # Reported per state; a state maps 1:1 to a function, so a differing value
        # across attempts would mean the ASL routes one state at two sizes, and the
        # per-attempt cost below is what stays correct.
        slot["memory_mb"] = memory_mb
        billed = _billed_sec(attempt)
        slot["billed_sec"] = round(slot["billed_sec"] + billed, 3)
        slot["estimated_cost"] = round(
            slot["estimated_cost"]
            + billed * (memory_mb / 1024.0) * _PRICE_PER_GB_SEC
            + _PRICE_PER_REQUEST, 6)

    for name, slot in states.items():
        slot["wall_sec"] = round(longest.get(name, slot["duration_sec"]), 3)
        if name in started_children:
            # Deduped: an ASL Retry on a `.sync:2` state re-submits the SAME execution
            # name, so two attempts can report one arn twice, and the tree walk would
            # then bill that child twice. `_descendants`' own `seen` set is the second
            # line of defence, not the first.
            slot["child_execution_arns"] = list(dict.fromkeys(started_children[name]))

    wall = 0.0
    if started is not None and events:
        wall = round((_ts(events[-1]["timestamp"]) - started).total_seconds(), 2)
    return states, wall


def _walk_back(event: dict, by_id: dict, match_types: set):
    """
    Follow `previousEventId` until an event of one of `match_types` is reached.

    `seen` guards against a cycle: a malformed history would otherwise spin until the
    Lambda timed out, turning a reporting problem into a 15-minute bill.
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
        cfg = _lambda().get_function_configuration(FunctionName=arn)
        return int(cfg.get("MemorySize") or _DEFAULT_MEMORY_MB)
    except Exception:                                              # noqa: BLE001
        return _DEFAULT_MEMORY_MB
