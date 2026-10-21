"""
DynamoDB counting semaphore — per-holder items + transactional counter.

Design (README.md §5). Items in the single stack table:

    pk="semaphore"  sk="counter"              holders, max_holders (informational)
    pk="semaphore"  sk="holder#{execution}"   ttl (refreshed on state entry), acquired_at
    pk="pending"    sk=sha256(task_token)     task_token, execution_arn, created_at, ttl
    pk="stream-dedup" sk={eventID}            ttl  (exactly-once decrement marker)

Invariants
──────────
1. SINGLE DECREMENT PATH: the counter is decremented ONLY by the stream
   handler processing a holder-item REMOVE. Normal release, EventBridge
   terminal cleanup, and DDB TTL expiry all just delete the holder item;
   the stream converges the three paths.
2. EXACTLY-ONCE DECREMENT: streams are at-least-once. Each decrement is a
   transaction with a per-eventID dedup marker — a retried record trips the
   marker's condition and skips.
3. TTL REFRESH: 3600 s bounds a single STATE, not the whole span. Every
   Rekognition-span state calls refresh_holder() on entry; a live job never
   expires, a crashed one reclaims <= 1 h after its last state entry.
4. max_holders is REQUEST-LEVEL (input_config.max_concurrent_executions), not
   a stack parameter. try_acquire ratchets the stored max_holders attribute
   UP-ONLY: a request below the current value is a no-op. There is no
   per-request lowering — a stack-wide cap can only ever grow, which is the
   intended behavior (one low-value caller must not throttle everyone else).
5. COUNTER TTL, anchored to holder liveness (mirrors core-lib's
   atomicLockTable, adapted for the per-holder-item design — see README.md
   §14): the counter item ALSO carries a ttl, refreshed by try_acquire,
   decrement_once, AND refresh_holder. This is the self-heal for a
   permanently-lost stream decrement (event-source-mapping outage, exhausted
   retries) — without it, holders could drift upward forever with no path
   back to zero. refresh_holder only refreshes the counter's ttl AFTER
   confirming ITS OWN holder item still exists — a zombie execution whose
   holder was already TTL-reclaimed must not resurrect the counter. Refreshing
   only on acquire/decrement (the core-lib pattern) would falsely expire the
   counter under a saturated, long-lived queue (5 slots held for over an hour
   with no new arrivals or completions) even though every holder is alive and
   refreshing; anchoring to refresh_holder's per-state-entry cadence during
   the Rekognition span avoids that. Side effect (accepted): if the table
   goes fully idle for over 3600 s, the counter (and any ratcheted
   max_holders) expires and starts over from a fresh acquire — reasonable,
   since "idle for an hour" means there are no concurrent callers left whose
   cap could be violated by a reset.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

from lib.input_schema import (
    PENDING_PK,
    SEMAPHORE_COUNTER_SK,
    SEMAPHORE_COUNTER_TTL_SEC,
    SEMAPHORE_HOLDER_SK_PREFIX,
    SEMAPHORE_HOLDER_TTL_SEC,
    SEMAPHORE_PENDING_TTL_SEC,
    SEMAPHORE_PK,
    STREAM_DEDUP_PK,
    STREAM_DEDUP_TTL_SEC,
)

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("FACE_TABLE_NAME", "")

_ddb = boto3.client("dynamodb")
_sfn = boto3.client("stepfunctions")


def _now() -> int:
    return int(time.time())


def _holder_sk(execution_arn: str) -> str:
    return f"{SEMAPHORE_HOLDER_SK_PREFIX}{execution_arn}"


# ── Acquire ────────────────────────────────────────────────────────────────────

def _raise_max_holders(requested_max: int) -> None:
    """
    Ratchet the stored max_holders UP-ONLY. A request at or below the
    current stored value is a no-op (ConditionalCheckFailedException,
    ignored) — the effective cap never shrinks once raised.
    """
    try:
        _ddb.update_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": SEMAPHORE_PK}, "sk": {"S": SEMAPHORE_COUNTER_SK}},
            UpdateExpression="SET max_holders = :requested",
            ConditionExpression=(
                "attribute_not_exists(max_holders) OR max_holders < :requested"),
            ExpressionAttributeValues={":requested": {"N": str(requested_max)}},
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise


def try_acquire(execution_arn: str, requested_max: int) -> str:
    """
    Atomically claim a slot: counter += 1 (guarded by < max_holders) AND put
    the holder item, in one transaction.

    Returns (all truthy values mean "held"):
      "acquired"  — fresh slot claimed (counter incremented)
      "reentrant" — OUR holder item already exists (retry after a successful
                    acquire); counter untouched. Callers that release a slot
                    on failure paths MUST NOT release a reentrant hold — the
                    holder belongs to an already-admitted, live execution.
      ""          — no free slot (falsy)
    """
    _raise_max_holders(requested_max)
    try:
        _ddb.transact_write_items(TransactItems=[
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"pk": {"S": SEMAPHORE_PK},
                            "sk": {"S": SEMAPHORE_COUNTER_SK}},
                    "UpdateExpression":
                        "SET holders = if_not_exists(holders, :zero) + :one, "
                        "#ttl = :ttl",
                    "ConditionExpression":
                        "attribute_not_exists(holders) OR holders < max_holders",
                    "ExpressionAttributeNames": {"#ttl": "ttl"},
                    "ExpressionAttributeValues": {
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                        ":ttl": {"N": str(_now() + SEMAPHORE_COUNTER_TTL_SEC)},
                    },
                },
            },
            {
                "Put": {
                    "TableName": TABLE_NAME,
                    "Item": {
                        "pk": {"S": SEMAPHORE_PK},
                        "sk": {"S": _holder_sk(execution_arn)},
                        "ttl": {"N": str(_now() + SEMAPHORE_HOLDER_TTL_SEC)},
                        "acquired_at": {"N": str(_now())},
                    },
                    "ConditionExpression": "attribute_not_exists(sk)",
                },
            },
        ])
        return "acquired"
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "TransactionCanceledException":
            raise
        reasons = exc.response.get("CancellationReasons") or []
        # Slot full → counter condition failed (reason[0]).
        # Re-entrant retry → holder Put condition failed (reason[1]) while
        # counter would have succeeded: our item already exists → treat as held.
        if (len(reasons) == 2
                and reasons[0].get("Code") == "None"
                and reasons[1].get("Code") == "ConditionalCheckFailed"):
            logger.info("re-entrant acquire for %s — already held", execution_arn)
            return "reentrant"
        return ""


def store_pending(execution_arn: str, task_token: str, requested_max: int) -> None:
    # Raise here too — a queued request's max_concurrent_executions should
    # still ratchet the cap even before it is drained (try_acquire raises
    # again per admitted holder, but that only happens once a slot frees up).
    _raise_max_holders(requested_max)
    sk = hashlib.sha256(task_token.encode()).hexdigest()
    _ddb.put_item(
        TableName=TABLE_NAME,
        Item={
            "pk": {"S": PENDING_PK},
            "sk": {"S": sk},
            "task_token": {"S": task_token},
            "execution_arn": {"S": execution_arn},
            "requested_max": {"N": str(requested_max)},
            "created_at": {"N": str(int(time.time() * 1000))},
            "ttl": {"N": str(_now() + SEMAPHORE_PENDING_TTL_SEC)},
        },
    )


# ── Hold maintenance ───────────────────────────────────────────────────────────

def refresh_holder(execution_arn: str) -> None:
    """
    Best-effort TTL refresh; called on entry of every Rekognition-span state.
    Also refreshes the counter's ttl (invariant 5) — but ONLY after confirming
    OUR holder item still exists, so a zombie execution whose holder was
    already TTL-reclaimed cannot resurrect the counter.
    """
    try:
        _ddb.update_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": SEMAPHORE_PK},
                 "sk": {"S": _holder_sk(execution_arn)}},
            UpdateExpression="SET #ttl = :ttl",
            ConditionExpression="attribute_exists(sk)",
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={
                ":ttl": {"N": str(_now() + SEMAPHORE_HOLDER_TTL_SEC)}},
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        logger.warning("refresh_holder: no holder item for %s (TTL-reclaimed?)",
                       execution_arn)
        return

    try:
        _ddb.update_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": SEMAPHORE_PK}, "sk": {"S": SEMAPHORE_COUNTER_SK}},
            UpdateExpression="SET #ttl = :ttl",
            ConditionExpression="attribute_exists(sk)",
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={
                ":ttl": {"N": str(_now() + SEMAPHORE_COUNTER_TTL_SEC)}},
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise


# ── Release ────────────────────────────────────────────────────────────────────

def release_holder(execution_arn: str) -> bool:
    """
    Delete the holder item ONLY — no counter decrement here (invariant 1).
    Idempotent: returns False when already released.
    """
    try:
        _ddb.delete_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": SEMAPHORE_PK},
                 "sk": {"S": _holder_sk(execution_arn)}},
            ConditionExpression="attribute_exists(sk)",
        )
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


# ── Stream handler support ─────────────────────────────────────────────────────

def decrement_once(event_id: str) -> bool:
    """
    Exactly-once counter decrement for one holder REMOVE stream record.
    Dedup marker + decrement in one transaction; a retried record returns False.
    """
    try:
        _ddb.transact_write_items(TransactItems=[
            {
                "Put": {
                    "TableName": TABLE_NAME,
                    "Item": {
                        "pk": {"S": STREAM_DEDUP_PK},
                        "sk": {"S": event_id},
                        "ttl": {"N": str(_now() + STREAM_DEDUP_TTL_SEC)},
                    },
                    "ConditionExpression": "attribute_not_exists(sk)",
                },
            },
            {
                "Update": {
                    "TableName": TABLE_NAME,
                    "Key": {"pk": {"S": SEMAPHORE_PK},
                            "sk": {"S": SEMAPHORE_COUNTER_SK}},
                    "UpdateExpression": "SET holders = holders - :one, #ttl = :ttl",
                    "ConditionExpression": "holders > :zero",
                    "ExpressionAttributeNames": {"#ttl": "ttl"},
                    "ExpressionAttributeValues": {
                        ":one": {"N": "1"},
                        ":zero": {"N": "0"},
                        ":ttl": {"N": str(_now() + SEMAPHORE_COUNTER_TTL_SEC)},
                    },
                },
            },
        ])
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "TransactionCanceledException":
            logger.info("decrement_once: skipped (dup record or zero counter) %s",
                        event_id)
            return False
        raise


_DEAD_TOKEN_CODES = ("TaskDoesNotExist", "TaskTimedOut")


def drain_pending() -> int:
    """
    Admit waiters FIFO while slots are free. For each: acquire with the
    waiter's execution_arn, SendTaskSuccess, delete the pending item.
    Returns the number of waiters admitted.

    Failure discipline (each case observed or derived from real semantics):
    - DEAD token (TaskDoesNotExist/TaskTimedOut): the waiter's execution is
      gone or timed out. Release the slot ONLY if this call freshly acquired
      it — a REENTRANT hold means a prior drain already admitted this waiter
      and crashed before deleting the pending item; its execution is LIVE and
      owns that holder, so releasing it would break the concurrency cap
      (holder REMOVE → decrement → over-admission). Either way the pending
      item is deleted (the token can never be redeemed again).
    - ANY OTHER error (SFN throttling, transient 5xx): the waiter is still
      real. Re-raise WITHOUT deleting the pending item — the stream retry
      re-runs the drain and the waiter is admitted then. Deleting it here
      would permanently strand a live execution on its task token (it would
      only die at the Acquire state's HeartbeatSeconds timeout).
    """
    admitted = 0
    while True:
        resp = _ddb.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": PENDING_PK}},
            Limit=20,
        )
        items = resp.get("Items") or []
        if not items:
            return admitted
        # FIFO: oldest created_at first
        items.sort(key=lambda i: int(i["created_at"]["N"]))
        progressed = False
        for item in items:
            execution_arn = item["execution_arn"]["S"]
            requested_max = int(item["requested_max"]["N"])
            held = try_acquire(execution_arn, requested_max)
            if not held:
                return admitted  # no free slot — stop draining
            token = item["task_token"]["S"]
            try:
                _sfn.send_task_success(taskToken=token, output="{}")
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code not in _DEAD_TOKEN_CODES:
                    # Transient (throttle/5xx): waiter is live — keep its
                    # pending item and slot; the stream retry finishes the job.
                    logger.warning("SendTaskSuccess transient failure for %s: "
                                   "%s — leaving pending for retry",
                                   execution_arn, code)
                    raise
                logger.warning("dead task token for %s: %s — cascading",  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
                               execution_arn, code)
                if held == "acquired":
                    # Freshly claimed for a dead waiter — give it back. The
                    # holder REMOVE re-triggers the stream to keep draining.
                    release_holder(execution_arn)
                # reentrant: the holder belongs to the live, already-admitted
                # execution from a prior drain — do NOT release it.
            _ddb.delete_item(
                TableName=TABLE_NAME,
                Key={"pk": {"S": PENDING_PK}, "sk": item["sk"]},
            )
            admitted += 1
            progressed = True
        if not progressed:
            return admitted
