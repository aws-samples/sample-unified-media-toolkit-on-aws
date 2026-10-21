"""
sync-infer-quotas — DynamoDB counting semaphore. Per-holder items + transactional
counter, scoped by an opaque `{service}#{resource}` string.

Guards the SYNC inference path only: one lease per execution, `resource` being the
primary model id, so each model is its own cap and a fallback chain does not
multiply it. The batch path takes no lease at all (the in-progress-job limit is
account-level and service-enforced).

ORIGIN: this file was the vendored client of `service-quotas-on-aws`, which
no longer exists — its semaphore was merged into this stack so bedrock-inference is
self-contained (README.md §M2). It remains deliberately self-contained —
constants, exceptions, and key construction all live here and it imports nothing
from this project — because that is what let it be vendored, and a future consumer
may vendor it again.

Two properties of the merged form differ from the standalone stack, both recorded
because they were load-bearing there:

1. **The cap is now per deployment prefix, not per region.** The standalone stack's
   singleton property was enforced by the table NAME (`${ResourcePrefix}-service-quotas`,
   one per region): a second stack failed at CREATE. Two bedrock-inference stacks
   with different prefixes now get two independent caps against one real Bedrock
   quota and can jointly exceed it. `summary.quota_table` reports the table name so
   the split is visible in the output rather than only in the throttling.
2. **`states:SendTask*` is no longer isolated to a Bedrock-free function, and since
   the tier collapse it is not isolated at all.** This module still has NO Step
   Functions dependency — `drain_pending` takes the sender as a parameter, and
   nothing here builds a `stepfunctions` client, which is what keeps it vendorable.
   But the handlers that call it (`states/acquire_lease`,
   `states/handle_semaphore_release`) run on the one workflow role, and so does
   everything that calls Bedrock. A task token is an opaque bearer credential and
   cannot be resource-scoped, which is why this mattered; the merge narrowed the
   isolation from one stack to one role (README.md §M2.4) and the tier collapse
   removed the last of it. Nothing outside the code now separates redeeming a token
   from calling a model.

Items in `{ResourcePrefix}-sync-infer-quotas`
────────────────────────────────────────────
    pk="semaphore#{svc}#{res}"  sk="counter"             holders, max_holders, ttl
    pk="semaphore#{svc}#{res}"  sk="holder#{execution}"  ttl, acquired_at
    pk="lease#{execution}"      sk="scope#{svc}#{res}"   service, resource, ttl
    pk="pending#{svc}#{res}"    sk=sha256(task_token)    task_token, requested_max, ttl
    pk="stream-dedup"           sk={eventID}             ttl

The `lease#` pointer is the one addition to the reference implementation
(`face-recognition-on-aws/lib/semaphore.py`), and it exists for one reason:
that reference has a single hardcoded `pk="semaphore"`, so its EventBridge
terminal-state handler can delete a holder knowing only the execution ARN. Here the
scope is part of the partition key, and an EventBridge terminal event carries only
`executionArn` — with no reverse pointer, the terminal trigger below could not find
the item to delete, and an aborted execution would hold its slot until TTL. A GSI on
the execution ARN would buy the same lookup; a second item in the same transaction
is cheaper and keeps the table GSI-free.

Invariants — each has a SILENT failure mode if violated
──────────────────────────────────────────────────────
1. SINGLE DECREMENT PATH. All THREE ways a holder disappears (normal release, the
   EventBridge terminal-state rule, DDB TTL expiry) do exactly one thing:
   DeleteItem on the holder. Only the stream handler decrements the counter, on the
   holder's REMOVE. Two code paths that each decrement is two chances to
   double-decrement.

   There used to be a fourth: an ASL `Catch` on every post-acquire sync state
   routing to a `Release lease on failure` state. It was removed with the merge
   (README.md §M1.1) to match the reference implementation, whose ASL has no
   Catch on its semaphore states either. What that cost is named there and not
   hidden: an acquire whose `SendTaskSuccess` fails after `try_acquire` succeeded
   leaves a live holder on an unredeemable token, and the slot is now reclaimed when
   the state times out (`TimeoutSeconds: 86400`) rather than in-band. The in-band
   release only ever shortened that window — the slot was unusable to the drain
   either way — so it is a latency regression on the rarest path, not a leak.
2. EXACTLY-ONCE DECREMENT. Streams are at-least-once. The decrement and a
   per-eventID dedup marker Put happen in ONE TransactWriteItems, the Put guarded
   by attribute_not_exists — a redelivered shard record trips the marker and skips.
   Without it the cap inflates permanently.
3. TTL BOUNDS A STATE, NOT A SPAN. `refresh_holder` is called ONCE on entry of
   every state that holds the lease — not per unit of work inside it. 3600 s
   against Lambda's 900 s hard ceiling means no single invocation can outlive its
   own refresh; a span built from many invocations (a checkpoint/re-enter loop) is
   many state entries, so the cadence is unchanged. If the TTL ever does expire
   mid-work, DDB reclaims the holder, the stream decrements, and a waiter is
   admitted WHILE THE FIRST CALLER IS STILL CALLING.
4. max_holders IS REQUEST-LEVEL, first-writer-wins, UP-ONLY, clamped at
   MAX_HOLDERS_CEILING. There is no per-request lowering: one low-value caller must
   not throttle everyone else. There is deliberately NO DEFAULT here — no value is
   right for both Bedrock (3) and Rekognition (5), so the default belongs to each
   consumer and a missing value is a ValidationException, never a silent 1.
5. COUNTER TTL, ANCHORED TO HOLDER LIVENESS. The counter also carries a ttl,
   refreshed by try_acquire, decrement_once, AND refresh_holder — the self-heal for
   a permanently-lost stream decrement (event-source-mapping outage, exhausted
   retries), which would otherwise leave `holders` too high forever with no path
   back to zero. `refresh_holder` refreshes the counter only AFTER confirming ITS
   OWN holder still exists, so a zombie execution whose holder was already
   TTL-reclaimed cannot resurrect the counter. Refreshing only on acquire/decrement
   (core-lib's atomicLockTable pattern) would falsely expire the counter under a
   saturated, long-lived queue — every holder alive and refreshing, no arrivals, no
   completions.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from typing import Callable

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# A holder's TTL bounds ONE state entry (invariant 3), not the whole guarded span.
HOLDER_TTL_SEC = 3600
COUNTER_TTL_SEC = 3600
# A waiter parked on a task token outlives any single state by design — the
# guarded work it is queued behind is measured in hours.
PENDING_TTL_SEC = 86400
STREAM_DEDUP_TTL_SEC = 3600

# The reference implementation's ratchet is unbounded. A caller typo of 1000 must
# not become deployment policy for every other consumer of the table, and the only
# way back down is the counter's TTL.
MAX_HOLDERS_CEILING = 10

SEMAPHORE_PK_PREFIX = "semaphore#"
LEASE_PK_PREFIX = "lease#"
PENDING_PK_PREFIX = "pending#"
STREAM_DEDUP_PK = "stream-dedup"
COUNTER_SK = "counter"
HOLDER_SK_PREFIX = "holder#"
SCOPE_SK_PREFIX = "scope#"

TABLE_NAME_ENV = "SYNC_INFER_QUOTAS_TABLE_NAME"

# SendTaskSuccess failure codes that mean the waiter is GONE, as opposed to a
# transient Step Functions failure. The distinction decides whether the pending
# item is deleted or kept for the next drain (§5.3) — collapsing the two either
# strands a live execution on its token or retries a dead one forever.
DEAD_TOKEN_CODES = ("TaskDoesNotExist", "TaskTimedOut")

# DynamoDB's optimistic-concurrency refusals on the counter item, and the reason a
# hand-rolled retry exists here at all: **botocore does not retry either of these.**
# Its dynamodb policy retries `TransactionInProgressException` and
# `ReplicatedWriteConflictException` and stops there — verified in
# `botocore/data/_retry.json`, where the string `TransactionConflict` does not appear.
#
# `TransactionConflictException` is raised on a plain `UpdateItem` whose item is
# concurrently inside someone else's transaction. Every writer in this module touches
# the SAME counter item, so under real contention that is the expected steady state,
# not an anomaly — a 24-execution burst against a cap of 3 failed 11 executions
# outright before this existed, split across both `_raise_max_holders` call sites.
#
# `TransactionCanceledException` is deliberately NOT here: it is the mechanism the
# cap is enforced with. Retrying a saturated `try_acquire` would spin instead of
# parking the waiter on its task token, which is the whole design.
_WRITE_CONFLICT_CODES = ("TransactionConflictException",
                         "TransactionInProgressException")

# Attempt budget for a conflicted counter write. Jittered exponential from 50 ms —
# the same base botocore uses for DynamoDB — for a worst case of about 1.6 s across
# six attempts. Sized against contention, not latency: with a cap of N, at most N
# writers can be inside the counter transaction at once, and each holds it for a
# single-digit-millisecond transaction.
_CONFLICT_MAX_ATTEMPTS = 6
_CONFLICT_BASE_SLEEP_SEC = 0.05


class ValidationException(Exception):
    """
    Malformed request — never retried.

    The class NAME is the contract: Step Functions matches `errorType`, which for
    a Python Lambda is the exception class name. `ErrorEquals: ["ValidationException"]`
    matches this; it could never match a botocore ClientError whatever its error
    code, which is why the retry ladders in source/misc/web-inference/definition.json
    work in Node and would silently never fire in Python.
    """


# ── Client ────────────────────────────────────────────────────────────────────

_DDB = None


def _ddb():
    """
    Lazy so that importing this module needs neither credentials nor a region.
    Tests assign the module global directly; a client built at import time would
    make an offline suite pass or fail according to the developer's shell.
    """
    global _DDB
    if _DDB is None:
        _DDB = boto3.client("dynamodb")
    return _DDB


def table_name() -> str:
    name = os.environ.get(TABLE_NAME_ENV) or ""
    if not name:
        raise ValidationException(f"{TABLE_NAME_ENV} is not set")
    return name


def _now() -> int:
    return int(time.time())


def _sleep(seconds: float) -> None:
    """Indirected so the offline suite can assert on the backoff without serving it."""
    time.sleep(seconds)


# ── Keys ──────────────────────────────────────────────────────────────────────

def _scope(service: str, resource: str) -> str:
    """
    Validate and join the scope. `#` is rejected in either half because the key is
    a two-level join: service="bedrock#us", resource="sonnet" and
    service="bedrock", resource="us#sonnet" would otherwise produce the SAME
    partition and silently share one cap.

    Neither half is interpreted here. What a unit of contention is — a model id, a
    Rekognition API name — is the caller's decision.
    """
    for label, value in (("service", service), ("resource", resource)):
        if not isinstance(value, str) or not value.strip():
            raise ValidationException(f"{label} is required and must be a non-empty string")
        if "#" in value:
            raise ValidationException(
                f"{label} must not contain '#' (got {value!r}) — it is a key separator")
    return f"{service}#{resource}"


def semaphore_pk(service: str, resource: str) -> str:
    return f"{SEMAPHORE_PK_PREFIX}{_scope(service, resource)}"


def pending_pk(service: str, resource: str) -> str:
    return f"{PENDING_PK_PREFIX}{_scope(service, resource)}"


def lease_pk(execution_arn: str) -> str:
    if not isinstance(execution_arn, str) or not execution_arn.strip():
        raise ValidationException("execution_arn is required")
    return f"{LEASE_PK_PREFIX}{execution_arn}"


def holder_sk(execution_arn: str) -> str:
    if not isinstance(execution_arn, str) or not execution_arn.strip():
        raise ValidationException("execution_arn is required")
    return f"{HOLDER_SK_PREFIX}{execution_arn}"


def scope_sk(service: str, resource: str) -> str:
    return f"{SCOPE_SK_PREFIX}{_scope(service, resource)}"


def parse_semaphore_pk(pk: str) -> tuple[str, str]:
    """
    `semaphore#{service}#{resource}` → (service, resource). Used by the stream and
    drain handlers, which learn the scope from the record's own keys rather than
    from an event field that could disagree with it.
    """
    if not pk.startswith(SEMAPHORE_PK_PREFIX):
        raise ValidationException(f"not a semaphore partition key: {pk!r}")
    rest = pk[len(SEMAPHORE_PK_PREFIX):]
    service, sep, resource = rest.partition("#")
    if not sep or not service or not resource:
        raise ValidationException(f"malformed semaphore partition key: {pk!r}")
    return service, resource


def execution_arn_from_holder_sk(sk: str) -> str:
    if not sk.startswith(HOLDER_SK_PREFIX):
        raise ValidationException(f"not a holder sort key: {sk!r}")
    return sk[len(HOLDER_SK_PREFIX):]


# ── max_holders ratchet ───────────────────────────────────────────────────────

def normalize_max_concurrent(max_concurrent) -> int:
    """
    Required, integral, >= 1, clamped to MAX_HOLDERS_CEILING (invariant 4).

    Clamped rather than rejected: a caller asking for more headroom than the
    deployment allows should still run, at the ceiling. Missing is a different
    thing entirely and raises — a stack-level default would be the wrong cap for
    every consumer but one, applied silently whenever the field was omitted.
    """
    if max_concurrent is None or isinstance(max_concurrent, bool):
        raise ValidationException("max_concurrent is required (no default — see invariant 4)")
    try:
        value = int(max_concurrent)
    except (TypeError, ValueError):
        raise ValidationException(f"max_concurrent must be an integer, got {max_concurrent!r}")
    if value < 1:
        raise ValidationException(f"max_concurrent must be >= 1, got {value}")
    if value > MAX_HOLDERS_CEILING:
        logger.warning("max_concurrent %d clamped to ceiling %d", value, MAX_HOLDERS_CEILING)
        return MAX_HOLDERS_CEILING
    return value


def is_transaction_conflict(exc) -> bool:
    """
    True when a `TransactionCanceledException` was cancelled by CONTENTION rather
    than by one of our own conditions.

    This distinction is load-bearing and easy to miss, because a conflicted
    transaction and a legitimately-refused one arrive as the *same exception class*.
    The difference is only in `CancellationReasons[i].Code`: `TransactionConflict`
    means "another writer held this item, nothing was evaluated", while
    `ConditionalCheckFailed` means "your condition was evaluated and it was false".

    Every caller in this module treats a cancellation as a meaningful answer —
    "saturated", "no holder", "already released", "duplicate stream record" — so
    reading a conflict as one of those is a wrong answer, not a retry. The worst is
    `decrement_once`, where it would drop a real decrement: the counter keeps a slot
    that no longer has a holder, and nothing frees it until the counter's own TTL an
    hour later.
    """
    response = getattr(exc, "response", None) or {}
    if (response.get("Error") or {}).get("Code") != "TransactionCanceledException":
        return False
    return any((reason or {}).get("Code") == "TransactionConflict"
               for reason in response.get("CancellationReasons") or [])


def _conflict_retry(operation: Callable[[], None], what: str) -> None:
    """
    Run a counter write, retrying DynamoDB's write-conflict refusals.

    Necessary because **botocore retries neither** `TransactionConflictException` nor
    a plain-UpdateItem conflict (`_WRITE_CONFLICT_CODES`), and every writer in this
    module contends on one item: the counter. A conflict is therefore the expected
    behaviour of a working semaphore under load, and letting it escape turns
    contention — the exact condition the semaphore exists to manage — into a failed
    caller execution.

    Deliberately narrow. Retried: the top-level codes in `_WRITE_CONFLICT_CODES`, and
    a `TransactionCanceledException` whose cancellation reasons say `TransactionConflict`
    (`is_transaction_conflict`). NOT retried: a cancellation caused by one of our own
    `ConditionExpression`s, which is the cap and the ratchet doing their jobs and must
    reach the caller unchanged.

    Exhausting the budget re-raises rather than returning quietly: at that point the
    write genuinely did not happen, and for `try_acquire` a swallowed failure would
    report a slot that was never claimed.
    """
    for attempt in range(_CONFLICT_MAX_ATTEMPTS):
        try:
            operation()
            return
        except ClientError as exc:
            if (exc.response["Error"]["Code"] not in _WRITE_CONFLICT_CODES
                    and not is_transaction_conflict(exc)):
                raise
            if attempt == _CONFLICT_MAX_ATTEMPTS - 1:
                logger.error(
                    "%s: still conflicted after %d attempts — giving up",
                    what, _CONFLICT_MAX_ATTEMPTS)
                raise
            # Jittered: a fixed backoff re-collides the same set of writers on every
            # attempt, which is the failure mode this is meant to break up.
            delay = _CONFLICT_BASE_SLEEP_SEC * (2 ** attempt) * (
                0.5 + random.random())
            logger.info("%s: write conflict, attempt %d/%d, sleeping %.3fs",
                        what, attempt + 1, _CONFLICT_MAX_ATTEMPTS, delay)
            _sleep(delay)


def _raise_max_holders(service: str, resource: str, requested_max: int) -> None:
    """
    Ratchet the stored max_holders UP-ONLY. A request at or below the current value
    is a no-op — the effective cap never shrinks once raised, and the only way back
    down is counter TTL expiry.

    **Read-before-write, and the read is the point.** The steady state is "the cap is
    already correct": every execution in a burst asks for the same `max_concurrent`,
    so the first one ratchets and the other N-1 have nothing to do. Writing
    unconditionally made all N contend on the counter — the single hottest item in the
    table, already carrying every acquire and release transaction — to change nothing.
    A `GetItem` on a fully-satisfied cap costs one eventually-consistent read and
    issues no write at all.

    Eventual consistency is safe HERE specifically because the ratchet is monotonic
    and the write is still guarded. A stale low read leads to a conditional write that
    either wins or fails `ConditionalCheckFailed`, which is already the ignored no-op
    case; it cannot lower the cap. That reasoning does not transfer to the counter's
    `holders`, which is why nothing else in this module reads before writing.
    """
    key = {"pk": {"S": semaphore_pk(service, resource)},
           "sk": {"S": COUNTER_SK}}
    try:
        current = (_ddb().get_item(TableName=table_name(), Key=key)
                   .get("Item") or {}).get("max_holders")
        if current and int(current["N"]) >= requested_max:
            return
    except (ClientError, KeyError, TypeError, ValueError):
        # A failed or unparseable read is not a reason to skip the ratchet — fall
        # through to the guarded write, which is correct on its own.
        logger.info("max_holders pre-read failed; falling through to the write")

    def write() -> None:
        try:
            _ddb().update_item(
                TableName=table_name(),
                Key=key,
                UpdateExpression="SET max_holders = :requested",
                ConditionExpression=(
                    "attribute_not_exists(max_holders) OR max_holders < :requested"),
                ExpressionAttributeValues={":requested": {"N": str(requested_max)}},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise

    try:
        _conflict_retry(write, f"max_holders ratchet on {semaphore_pk(service, resource)}")
    except ClientError as exc:
        if not is_transaction_conflict(exc) and exc.response["Error"]["Code"] \
                not in _WRITE_CONFLICT_CODES:
            raise
        # Exhaustion is swallowed, and this is the one place that is clearly right:
        # a lost ratchet is indistinguishable from the `ConditionalCheckFailed` this
        # already ignores, because contention here means OTHER callers are writing the
        # same item — and they all want the same cap. Raising instead would fail the
        # acquire before it was even attempted, which is the exact bug this whole
        # change exists to fix: 11 of 24 burst executions died at this line.
        logger.warning("max_holders ratchet on %s: conflict budget exhausted — the "
                       "cap may still be another caller's",
                       semaphore_pk(service, resource))


# ── Acquire ───────────────────────────────────────────────────────────────────

def try_acquire(service: str, resource: str, execution_arn: str,
                max_concurrent) -> str:
    """
    Atomically claim a slot: counter += 1 (guarded by < max_holders), holder Put
    (guarded by non-existence), and the lease pointer, in ONE transaction.

    Returns — both truthy values mean "held":
      "acquired"  fresh slot claimed, counter incremented
      "reentrant" OUR holder already exists (an ASL/Lambda retry after a
                  successful acquire); counter untouched
      ""          no free slot (falsy)

    The distinction is load-bearing on retry: a retried acquire must not
    double-count, and a REENTRANT hold MUST NOT be released by the caller that
    found it — the original acquirer is live and still owns it. Releasing it frees
    a slot that is still in use, and the cap is silently exceeded by one for the
    remainder of that execution's span.
    """
    requested_max = normalize_max_concurrent(max_concurrent)
    pk = semaphore_pk(service, resource)
    now = _now()

    _raise_max_holders(service, resource, requested_max)

    def claim() -> None:
        _ddb().transact_write_items(TransactItems=[
            {
                "Update": {
                    "TableName": table_name(),
                    "Key": {"pk": {"S": pk}, "sk": {"S": COUNTER_SK}},
                    "UpdateExpression": (
                        "SET holders = if_not_exists(holders, :zero) + :one, "
                        "#ttl = :cttl"),
                    "ConditionExpression": (
                        "attribute_not_exists(holders) OR holders < max_holders"),
                    "ExpressionAttributeNames": {"#ttl": "ttl"},
                    "ExpressionAttributeValues": {
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                        ":cttl": {"N": str(now + COUNTER_TTL_SEC)},
                    },
                },
            },
            {
                "Put": {
                    "TableName": table_name(),
                    "Item": {
                        "pk": {"S": pk},
                        "sk": {"S": holder_sk(execution_arn)},
                        "ttl": {"N": str(now + HOLDER_TTL_SEC)},
                        "acquired_at": {"N": str(now)},
                    },
                    "ConditionExpression": "attribute_not_exists(pk)",
                },
            },
            {
                # Reverse pointer for the EventBridge terminal handler. Written
                # UNCONDITIONALLY: a conditional Put here would add a second
                # failure reason to a re-entrant transaction for no benefit, and
                # an overwrite is exactly the right behaviour (it refreshes the
                # ttl alongside the holder's).
                "Put": {
                    "TableName": table_name(),
                    "Item": {
                        "pk": {"S": lease_pk(execution_arn)},
                        "sk": {"S": scope_sk(service, resource)},
                        "service": {"S": service},
                        "resource": {"S": resource},
                        "ttl": {"N": str(now + HOLDER_TTL_SEC)},
                    },
                },
            },
        ])

    try:
        # Retried on CONTENTION only. A conflict here is not "saturated": nothing was
        # evaluated, so returning "" would park a waiter that could have had a free
        # slot — and under a burst that is every waiter, which is exactly how a
        # 24-execution burst against a cap of 3 produced no pending rows at all.
        _conflict_retry(claim, f"acquire on {pk} for {execution_arn}")
        return "acquired"
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "TransactionCanceledException":
            # Covers the top-level `TransactionConflictException` too: an exhausted
            # budget is not an answer, and re-raising is what we want either way.
            raise
        if is_transaction_conflict(exc):
            # The retry budget is exhausted, not an answer. Nothing was evaluated, so
            # neither "" nor "reentrant" is true, and "" is the more dangerous guess:
            # it parks a waiter on a queue whose only drain trigger is some OTHER
            # holder's release. With slots free and nothing holding them, no release
            # is coming, so the waiter sits until its pending TTL. Fail the caller.
            raise
        reasons = exc.response.get("CancellationReasons") or []
        # reasons[1] is the holder Put, guarded on attribute_not_exists(pk) at a key
        # that is OURS by construction (pk = scope, sk = holder#{our execution_arn}).
        # Its failure can therefore mean exactly one thing: our own holder already
        # exists — a retry after a successful acquire.
        #
        # Read reason 1 ALONE, independently of reason 0. When we are the holder that
        # filled the last slot, the counter guard (holders < max_holders) fails too,
        # so BOTH reasons are ConditionalCheckFailed; also requiring reason 0 to be
        # "None" would report that as saturated and park a live holder on a pending
        # item — deadlocked behind its own slot until its Acquire state timed out.
        if len(reasons) >= 2 and reasons[1].get("Code") == "ConditionalCheckFailed":
            logger.info("re-entrant acquire on %s for %s", pk, execution_arn)
            return "reentrant"
        return ""


def store_pending(service: str, resource: str, execution_arn: str,
                  task_token: str, max_concurrent) -> None:
    """
    Park a waiter's task token. Keyed by sha256(token) so the item is
    self-deduplicating across acquire retries with the same token.
    """
    if not isinstance(task_token, str) or not task_token:
        raise ValidationException("task_token is required")
    requested_max = normalize_max_concurrent(max_concurrent)
    # Ratchet here too: a queued request's max_concurrent should raise the cap
    # before it is drained, not only once a slot frees up and try_acquire runs.
    _raise_max_holders(service, resource, requested_max)
    _ddb().put_item(
        TableName=table_name(),
        Item={
            "pk": {"S": pending_pk(service, resource)},
            "sk": {"S": hashlib.sha256(task_token.encode()).hexdigest()},
            "task_token": {"S": task_token},
            "execution_arn": {"S": execution_arn},
            "requested_max": {"N": str(requested_max)},
            # Milliseconds: FIFO ordering across a burst of waiters that arrive
            # within the same second is otherwise arbitrary.
            "created_at": {"N": str(int(time.time() * 1000))},
            "ttl": {"N": str(_now() + PENDING_TTL_SEC)},
        },
    )


# ── Hold maintenance ──────────────────────────────────────────────────────────

def refresh_holder(service: str, resource: str, execution_arn: str) -> bool:
    """
    Extend the holder's (and its pointer's) TTL. Call ONCE on entry of every state
    that holds the lease — invariant 3.

    Returns False ONLY on evidence the holder no longer exists (already
    TTL-reclaimed, or never acquired). Best-effort by design: the caller is mid-span
    and cannot usefully abort, and the counter's own TTL is the backstop.

    Refreshes the COUNTER's ttl only after the holder refresh succeeded
    (invariant 5) — a zombie execution must not resurrect the counter.
    """
    pk = semaphore_pk(service, resource)
    now = _now()

    def touch() -> None:
        _ddb().transact_write_items(TransactItems=[
            {
                "Update": {
                    "TableName": table_name(),
                    "Key": {"pk": {"S": pk}, "sk": {"S": holder_sk(execution_arn)}},
                    "UpdateExpression": "SET #ttl = :ttl",
                    "ConditionExpression": "attribute_exists(pk)",
                    "ExpressionAttributeNames": {"#ttl": "ttl"},
                    "ExpressionAttributeValues": {
                        ":ttl": {"N": str(now + HOLDER_TTL_SEC)}},
                },
            },
            {
                # The pointer must not outlive the holder, and must not die first
                # either: an expired pointer blinds the EventBridge terminal
                # handler, silently demoting trigger 3 to trigger 4 (TTL expiry,
                # up to an hour later).
                "Put": {
                    "TableName": table_name(),
                    "Item": {
                        "pk": {"S": lease_pk(execution_arn)},
                        "sk": {"S": scope_sk(service, resource)},
                        "service": {"S": service},
                        "resource": {"S": resource},
                        "ttl": {"N": str(now + HOLDER_TTL_SEC)},
                    },
                },
            },
        ])

    try:
        # Retried on contention: reading a conflict as "no holder" would log HOLDER
        # LOST and report the cap as exceeded while the holder is in fact perfectly
        # alive — a false alarm on the one signal that means the cap is being broken.
        _conflict_retry(touch, f"refresh_holder on {pk} for {execution_arn}")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        # Conflict FIRST, and both of its shapes. A `TransactionConflictException` is
        # not a `TransactionCanceledException`, so gating on the cancellation code
        # before classifying lets the top-level shape — the one the burst actually
        # produced — straight past the exhaustion branch below.
        if code in _WRITE_CONFLICT_CODES or is_transaction_conflict(exc):
            # Budget exhausted. Unlike everywhere else, this returns the TRUTHY
            # value: a conflict is no evidence the holder is gone, and both other
            # options are worse than a skipped bump. False triggers the caller's
            # HOLDER LOST error path — a false alarm on the one signal meaning the cap
            # is broken — and raising fails an execution that does legitimately hold
            # its slot. A missed bump costs TTL headroom, of which there is an hour,
            # and every concurrent acquire is refreshing the counter's ttl anyway.
            logger.warning(
                "refresh_holder on %s for %s: conflict budget exhausted — the TTL was "
                "NOT extended; treating the hold as live", pk, execution_arn)
            return True
        if code != "TransactionCanceledException":
            raise
        logger.warning("refresh_holder: no holder on %s for %s (TTL-reclaimed?)",
                       pk, execution_arn)
        return False

    def bump_counter_ttl() -> None:
        try:
            _ddb().update_item(
                TableName=table_name(),
                Key={"pk": {"S": pk}, "sk": {"S": COUNTER_SK}},
                UpdateExpression="SET #ttl = :ttl",
                ConditionExpression="attribute_exists(pk)",
                ExpressionAttributeNames={"#ttl": "ttl"},
                ExpressionAttributeValues={":ttl": {"N": str(now + COUNTER_TTL_SEC)}},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise

    try:
        _conflict_retry(bump_counter_ttl, f"counter ttl on {pk}")
    except ClientError as exc:
        if not is_transaction_conflict(exc) and exc.response["Error"]["Code"] \
                not in _WRITE_CONFLICT_CODES:
            raise
        # Swallowed for the same reason as above, and with less at stake: the holder
        # refresh already succeeded, so the hold is real. Every concurrent acquire
        # bumps this same ttl, so under the contention that caused this the counter is
        # being refreshed by someone else anyway.
        logger.warning("counter ttl on %s: conflict budget exhausted — not extended",
                       pk)
    return True


# ── Release ───────────────────────────────────────────────────────────────────

def release(service: str, resource: str, execution_arn: str) -> bool:
    """
    Delete the holder item (and its pointer). NO counter decrement here —
    invariant 1: the stream handler owns that, which is what makes all three
    disappearance paths converge on one decrement and one drain.

    Idempotent: returns False when already released.
    """
    def delete_holder() -> None:
        _ddb().transact_write_items(TransactItems=[
            {
                "Delete": {
                    "TableName": table_name(),
                    "Key": {"pk": {"S": semaphore_pk(service, resource)},
                            "sk": {"S": holder_sk(execution_arn)}},
                    "ConditionExpression": "attribute_exists(pk)",
                },
            },
            {
                "Delete": {
                    "TableName": table_name(),
                    "Key": {"pk": {"S": lease_pk(execution_arn)},
                            "sk": {"S": scope_sk(service, resource)}},
                },
            },
        ])

    try:
        # Retried on contention: a conflict read as "already released" returns False to
        # a caller that still holds the slot, and the delete that drives the stream
        # decrement never happens. The slot then survives until the holder TTL — up to
        # an hour of a cap short by one, with `release_lease` reporting success.
        _conflict_retry(delete_holder, f"release {execution_arn}")
        return True
    except ClientError as exc:
        if (exc.response["Error"]["Code"] == "TransactionCanceledException"
                and not is_transaction_conflict(exc)):
            return False
        raise


def get_holder(service: str, resource: str, execution_arn: str) -> dict | None:
    """
    The holder item, or None. Read path for a consumer that needs its own
    `acquired_at` — a checkpoint/re-enter loop measuring how long it has held the
    lease across invocations cannot keep that in the event, because the ASL retry
    re-invokes with the same input.
    """
    resp = _ddb().get_item(
        TableName=table_name(),
        Key={"pk": {"S": semaphore_pk(service, resource)},
             "sk": {"S": holder_sk(execution_arn)}},
        ConsistentRead=True,
    )
    return resp.get("Item")


def get_counter(service: str, resource: str) -> dict:
    """
    Observed `{holders, max_holders}` for one scope, `{}` when the counter is absent
    (never acquired, or TTL-reset). Reporting only — nothing decides admission from
    this, because a read-then-act would race the transaction that owns the cap.
    """
    resp = _ddb().get_item(
        TableName=table_name(),
        Key={"pk": {"S": semaphore_pk(service, resource)},
             "sk": {"S": COUNTER_SK}},
        ConsistentRead=True,
    )
    item = resp.get("Item") or {}
    out = {}
    for field in ("holders", "max_holders"):
        if field in item:
            out[field] = int(item[field]["N"])
    return out


def list_leases(execution_arn: str) -> list[tuple[str, str]]:
    """
    Every scope this execution holds, from the reverse pointer. One
    single-partition Query, no GSI, no scan — the EventBridge terminal handler
    (trigger 3) knows only the execution ARN.

    Returns [] for an execution that never acquired, which is the common case: the
    rule fires on every terminal state of every consumer state machine, including
    executions that failed before reaching their Acquire state.
    """
    out: list[tuple[str, str]] = []
    start_key = None
    while True:
        kwargs = {
            "TableName": table_name(),
            "KeyConditionExpression": "pk = :pk",
            "ExpressionAttributeValues": {":pk": {"S": lease_pk(execution_arn)}},
        }
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key
        resp = _ddb().query(**kwargs)
        for item in resp.get("Items") or []:
            service = (item.get("service") or {}).get("S")
            resource = (item.get("resource") or {}).get("S")
            if service and resource:
                out.append((service, resource))
        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            return out


# ── Stream handler support ────────────────────────────────────────────────────

def decrement_once(service: str, resource: str, event_id: str) -> bool:
    """
    Exactly-once counter decrement for ONE holder-REMOVE stream record
    (invariant 2). Dedup marker + decrement in one transaction; a redelivered
    record returns False.

    Also returns False when the counter is already at zero or absent — which
    happens legitimately after a counter TTL reset (invariant 5) whose holders had
    not yet been reclaimed. Guarding on `holders > 0` is what stops that from
    driving the counter negative and permanently inflating the cap.
    """
    if not event_id:
        raise ValidationException("event_id is required for exactly-once decrement")
    def decrement() -> None:
        now = _now()
        _ddb().transact_write_items(TransactItems=[
            {
                "Put": {
                    "TableName": table_name(),
                    "Item": {
                        "pk": {"S": STREAM_DEDUP_PK},
                        "sk": {"S": event_id},
                        "ttl": {"N": str(now + STREAM_DEDUP_TTL_SEC)},
                    },
                    "ConditionExpression": "attribute_not_exists(pk)",
                },
            },
            {
                "Update": {
                    "TableName": table_name(),
                    "Key": {"pk": {"S": semaphore_pk(service, resource)},
                            "sk": {"S": COUNTER_SK}},
                    "UpdateExpression": "SET holders = holders - :one, #ttl = :ttl",
                    "ConditionExpression": "holders > :zero",
                    "ExpressionAttributeNames": {"#ttl": "ttl"},
                    "ExpressionAttributeValues": {
                        ":one": {"N": "1"},
                        ":zero": {"N": "0"},
                        ":ttl": {"N": str(now + COUNTER_TTL_SEC)},
                    },
                },
            },
        ])

    try:
        # The worst of the conflict misreads, and the reason `_conflict_retry` exists:
        # this is the ONLY decrement a holder-REMOVE record ever gets. Read as
        # "duplicate record or zero counter", the stream record is acked and the
        # decrement is lost outright — the counter keeps a slot with no holder row
        # behind it until the counter's own TTL an hour later, so the effective cap is
        # permanently one lower with nothing in any log to say so.
        _conflict_retry(decrement, f"decrement for stream record {event_id}")
        return True
    except ClientError as exc:
        if (exc.response["Error"]["Code"] == "TransactionCanceledException"
                and not is_transaction_conflict(exc)):
            logger.info("decrement_once skipped (duplicate record or zero counter): %s",
                        event_id)
            return False
        # An exhausted conflict budget must reach the stream handler so the shard
        # RETRIES the record. Returning False acks a decrement that never happened.
        raise


def drain_pending(service: str, resource: str,
                  send_task_success: Callable[[str], None]) -> int:
    """
    Admit waiters FIFO while slots are free. For each: acquire with the WAITER's
    execution_arn, send its task result, delete the pending item. Returns the
    number admitted.

    `send_task_success(task_token)` is injected — this module never builds a
    stepfunctions client (README.md §5.1). It must raise botocore's ClientError so
    the failure discipline below can read the error code.

    Failure discipline, each case derived from real semantics:
    - DEAD token (TaskDoesNotExist/TaskTimedOut): the waiter is gone or timed out.
      Release the slot ONLY if THIS call freshly acquired it — a "reentrant" hold
      means a prior drain already admitted this waiter and died before deleting
      the pending item; that execution is LIVE and owns the holder, so releasing
      it would over-admit. Either way the pending item is deleted: the token can
      never be redeemed again.
    - ANY OTHER error (throttling, transient 5xx): the waiter is still real.
      Re-raise WITHOUT deleting — the stream retry re-runs the drain and admits it
      then. Deleting here would strand a live execution on its token until its
      Acquire state's own timeout, hours later.
    """
    admitted = 0
    while True:
        resp = _ddb().query(
            TableName=table_name(),
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": pending_pk(service, resource)}},
            Limit=20,
        )
        items = resp.get("Items") or []
        if not items:
            return admitted
        items.sort(key=lambda i: int(i["created_at"]["N"]))
        progressed = False
        for item in items:
            execution_arn = item["execution_arn"]["S"]
            requested_max = int(item["requested_max"]["N"])
            held = try_acquire(service, resource, execution_arn, requested_max)
            if not held:
                return admitted          # no free slot — stop draining
            try:
                send_task_success(item["task_token"]["S"])
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code not in DEAD_TOKEN_CODES:
                    logger.warning(
                        "send_task_success transient failure for %s: %s — leaving "
                        "the pending item for the next drain", execution_arn, code)
                    raise
                logger.warning("dead task token for %s: %s — cascading",  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
                               execution_arn, code)
                if held == "acquired":
                    # Freshly claimed for a waiter that is gone — give it back.
                    # The holder REMOVE re-triggers the stream, which keeps
                    # draining, so this does not need a recursive call here.
                    release(service, resource, execution_arn)
            _ddb().delete_item(
                TableName=table_name(),
                Key={"pk": {"S": pending_pk(service, resource)},
                     "sk": item["sk"]},
            )
            admitted += 1
            progressed = True
        if not progressed:
            return admitted
