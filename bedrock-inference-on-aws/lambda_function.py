"""
bedrock-inference-on-aws — Lambda entry point.

The micro-workflow that makes Bedrock calls on a caller's behalf: N records in, N
responses out, sync or batch. **Self-contained** — it owns its own concurrency
semaphore (`sync-infer-quotas`) and its own batch queue (`batch-infer-tokens`), and
depends on no sibling stack. Three state machines: an orchestrator that plans and
routes, and two standalone children (README.md §M1).

Routing
───────
  event shape / operation                → handler
  ───────────────────────────────────────────────────────────────────────────────
  operation="StateValidateAndPlan"       → states/validate_plan
  operation="StateUsageTracking"         → states/usage_tracking
  operation="StateAcquireLease"          → states/acquire_lease    (waitForTaskToken)
  operation="StateRunInference"          → states/run_inference
  operation="StateReleaseLease"          → states/release_lease
  operation="StateSubmitBatch"           → states/submit_batch     (waitForTaskToken)
  operation="StateCollectBatch"          → states/collect_batch
  operation="StateSummarize"             → states/summarize
  operation="Drain"                      → states/drain            (scheduled sweep)
  source="aws.states"                    → states/handle_execution_terminal
  source="aws.bedrock"                   → states/batch_status | states/drain
  Records[].eventSource=aws:dynamodb     → states/notify | states/drain
                                             | states/handle_semaphore_release
                                           (by STREAM_HANDLER, below)

**ONE container image, SEVEN memory tiers, ONE role** (2026-08-12). Every function
in this workflow runs this file out of the same ECR image; they differ only in
`MemorySize` and in `STREAM_HANDLER`, and every one of them points at the same
`BedrockInferenceLambdaRole`. That is the house convention — the sibling workflows in
this repo all shape their Lambdas as a `HalfCore … SixCores` ladder over one image and
one role — and it replaced a six-function, six-role split.

**The split was doing real work and this file must be read knowing it is gone.** A
`states:SendTask*` grant cannot be resource-scoped: a task token is an opaque bearer
credential, so anything holding it can complete any execution whose token it obtains.
That grant used to live only on roles that could not call Bedrock, and `run_inference`
could not `CreateModelInvocationJob` while `submit_batch` could not `Converse`. One
role holds all of it now. The pairwise impossibilities are conventions of THIS FILE
and the modules it dispatches to, and nothing outside the code enforces them —
cfnstack.yaml's IAM header and README.md record the trade.

The one that matters most, because a violation is silent and unrecoverable: exactly
one code path decrements the semaphore counter (`states/handle_semaphore_release`, on
the quota table's stream, deduped per `eventID`). Every other holder-disappearance
path DELETES a holder row and lets the stream do it. A unit test pins the caller set.

Three stream consumers still cannot be told apart by event shape, so `STREAM_HANDLER`
names which one this invocation is, set per-TIER in the template:

    notify      half-core tier  → batch-infer-tokens  → states/notify
    drain       1-core tier     → batch-infer-tokens  → states/drain
    semaphore   2-cores tier    → sync-infer-quotas   → states/handle_semaphore_release

**The markers are what pin the tier packing**, and they are read ONLY for a DynamoDB
stream record or an `aws.bedrock` event. An `operation` dispatch never consults them,
which is why a state may share a tier with a stream consumer harmlessly — the 2-cores
tier is most of the states AND the semaphore stream AND the `aws.states` terminal
rule, separated by event shape alone. An unset or unrecognised marker raises rather
than defaulting to a branch, which is why the four unrouted tiers set none.

Imports are deferred into each branch: one image serving every state, and no reason
for a notifier to pay the cold start of a Bedrock client it will not use.
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bedrock-inference")
logger.setLevel(logging.INFO)

# `StateAcquireLease` is here now. Before the merge it deliberately was not: the
# `Acquire lease` state targeted the quota stack's own function, because a local
# acquire handler needs `states:SendTaskSuccess` to be a `waitForTaskToken` target and
# that grant cannot be resource-scoped. The semaphore is in-stack now, so the grant is
# on the one workflow role — which, since the tier collapse, is also the role that
# calls Bedrock. README.md §M2.4 records the first half of that cost; the docstring
# above records the second.
_STATES = {
    "StateValidateAndPlan": "states.validate_plan",
    "StateUsageTracking": "states.usage_tracking",
    "StateAcquireLease": "states.acquire_lease",
    "StateRunInference": "states.run_inference",
    "StateReleaseLease": "states.release_lease",
    "StateSubmitBatch": "states.submit_batch",
    "StateCollectBatch": "states.collect_batch",
    "StateSummarize": "states.summarize",
    "Drain": "states.drain",
}

STREAM_HANDLER_ENV = "STREAM_HANDLER"


def lambda_handler(event: dict, context):
    event = event or {}
    operation = event.get("operation") or ""

    if operation in _STATES:
        module = _STATES[operation]
    elif _is_stream_event(event):
        module = _stream_module()
    elif _is_sfn_terminal_event(event):
        module = "states.handle_execution_terminal"
    elif _is_bedrock_batch_event(event):
        # The token rule (prefix-scoped) and the drain rule (unscoped) deliver the
        # same event shape to two different functions — §6.5's asymmetry. Same
        # marker, same reason as the stream split.
        #
        # Tested for "drain" explicitly rather than "not notify": `WorkflowFunction`
        # carries STREAM_HANDLER="semaphore" and must land on batch_status here if it
        # ever receives an aws.bedrock event, never on a second capacity sweep.
        module = ("states.drain" if _stream_handler() == "drain"
                  else "states.batch_status")
    else:
        raise ValueError(
            f"unroutable event: operation={operation!r} — expected one of "
            f"{sorted(_STATES)}, a DynamoDB stream batch, an aws.states terminal "
            "event, or an aws.bedrock batch-job event")

    handler = _import(module)
    return handler.lambda_handler(event, context)


def _import(module: str):
    import importlib

    return importlib.import_module(module + ".lambda_function")


def _stream_handler() -> str:
    return (os.environ.get(STREAM_HANDLER_ENV) or "").strip().lower()


def _stream_module() -> str:
    marker = _stream_handler()
    if marker == "drain":
        return "states.drain"
    if marker == "notify":
        return "states.notify"
    if marker == "semaphore":
        return "states.handle_semaphore_release"
    raise ValueError(
        f"{STREAM_HANDLER_ENV} must be 'notify', 'drain' or 'semaphore' on a function "
        f"subscribed to a table stream, got {marker!r} — the three handlers share this "
        "zip and differ only in their IAM role and which table they are wired to")


def _is_stream_event(event: dict) -> bool:
    records = event.get("Records") or []
    return bool(records and records[0].get("eventSource") == "aws:dynamodb")


def _is_sfn_terminal_event(event: dict) -> bool:
    """
    The terminal-state rule fires on all three of this stack's state machines. Matched
    on `source` alone rather than on `detail-type`: EventBridge already filtered to
    "Step Functions Execution Status Change" in the rule, and the handler reads only
    `detail.executionArn`.
    """
    return event.get("source") == "aws.states" and "detail" in event


def _is_bedrock_batch_event(event: dict) -> bool:
    return event.get("source") == "aws.bedrock" and "detail" in event
