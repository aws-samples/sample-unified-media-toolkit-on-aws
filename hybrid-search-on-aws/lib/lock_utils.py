"""
lock_utils.py — DynamoDB-based distributed lock, one per physical table.

Lock item:    pk="lock",    sk=index_key ("{collection_id}:{table_key}")
Pending item: pk="pending", sk="{index_key}#{sha256(task_token)}"

One lock per physical LanceDB table / S3 Vectors index (README.md Design record
§1.4): writers to different tables — across collections or across doc_types
within one collection — never contend. Only same-physical-table writers
serialize. The pending sk is prefixed with the lock's sk so the stream
handler can dequeue waiters for exactly the lock that was released
(begins_with query); FIFO within a lock via created_at (integer epoch ms).

acquire(lock_sk, holder, execution_arn) — conditional PutItem; stores
    execution_arn so release_if_owner() can verify ownership. Re-entrant
    for the same execution (Lambda retry after crash).
release_if_owner(lock_sk, execution_arn) — atomic conditional delete;
    safe under duplicate EventBridge delivery.
release(lock_sk) — unconditional delete; only when ownership is certain.
store_pending(lock_sk, ...) — queue a waiter for a specific lock.
pop_next_pending(lock_sk) — re-acquire the lock for the oldest waiter of
    that lock and dequeue it.
release_all_for_execution(execution_arn) — release every lock held by one
    execution (failure-handler path: an Index/Rebuild SFN may hold several
    table locks across Map iterations when it dies).
"""
from __future__ import annotations

import hashlib
import logging
import time

from boto3.dynamodb.conditions import Key

from lib.ddb_utils import (
    conditional_delete_item,
    conditional_put_item,
    delete_item,
    get_table,
    put_item,
    ttl_from_now,
)
from lib.input_schema import LOCK_TTL_SECS

log = logging.getLogger(__name__)

_LOCK_PK    = "lock"
_PENDING_PK = "pending"


def acquire(lock_sk: str, holder: str, execution_arn: str = "",
            reentrant: bool = True) -> bool:
    """
    Try to acquire one table's lock.
    holder: "index" or "rebuild" — stored for observability.
    reentrant=False forces strict attribute_not_exists — required on the
    DEQUEUE paths (pop_next_pending, lost-wakeup self-serve): two resumers
    acting for the same waiter would BOTH pass the re-entrant condition,
    both SendTaskSuccess, and the loser's dead-token cascade would release
    the lock the winner holds. Strict mode guarantees a single winner.
    Returns True if acquired, False if held by another execution.
    """
    if not lock_sk:
        raise ValueError("acquire: lock_sk is required")
    item = {
        "pk":            _LOCK_PK,
        "sk":            lock_sk,
        "holder":        holder,
        "execution_arn": execution_arn,
        "ttl":           ttl_from_now(LOCK_TTL_SECS),
    }
    # Re-entrant for the same execution (Lambda retry after crash). Guard on
    # non-empty ARN: "" would match any lock item that also stored "".
    if reentrant and execution_arn:
        condition   = "attribute_not_exists(pk) OR execution_arn = :arn"
        attr_values = {":arn": execution_arn}
    else:
        condition   = "attribute_not_exists(pk)"
        attr_values = None
    acquired = conditional_put_item(item, condition=condition, attr_values=attr_values)
    log.info("Lock %s %s by holder=%s execution_arn=%s",
             lock_sk, "acquired" if acquired else "HELD — not acquired",
             holder, execution_arn)
    return acquired


def release(lock_sk: str) -> None:
    """Unconditional release. Fires the DDB stream REMOVE event."""
    delete_item(_LOCK_PK, lock_sk)
    log.info("Lock %s released", lock_sk)


def release_if_owner(lock_sk: str, execution_arn: str) -> bool:
    """
    Atomically release one lock only if execution_arn matches the stored
    owner. Conditional DeleteItem — no TOCTOU race; safe under duplicate
    EventBridge delivery and concurrent re-acquisition by a waiter.
    """
    if not execution_arn:
        log.warning("release_if_owner: empty execution_arn — skipping")
        return False
    released = conditional_delete_item(
        _LOCK_PK, lock_sk,
        condition="execution_arn = :arn",
        attr_values={":arn": execution_arn},
    )
    log.info("release_if_owner %s: %s (execution_arn=%s)",
             lock_sk, "released" if released else "not owner / absent", execution_arn)
    return released


def release_all_for_execution(execution_arn: str) -> int:
    """
    Release every lock held by one execution. Used by the EventBridge
    failure handler: an Index/Rebuild SFN acquires one lock per Map
    iteration and may hold several when it dies mid-flight.
    """
    if not execution_arn:
        return 0
    table    = get_table()
    released = 0
    kwargs: dict = {"KeyConditionExpression": Key("pk").eq(_LOCK_PK)}
    while True:
        resp = table.query(**kwargs)
        for item in resp.get("Items", []):
            if item.get("execution_arn") == execution_arn:
                if release_if_owner(item["sk"], execution_arn):
                    released += 1
        last = resp.get("LastEvaluatedKey")
        if not last:
            break
        kwargs["ExclusiveStartKey"] = last
    log.info("release_all_for_execution %s: released %d lock(s)", execution_arn, released)
    return released


def store_pending(lock_sk: str, task_token: str, uuid: str,
                  execution_arn: str = "") -> None:
    """
    Queue a waiter for a specific lock. sk = "{lock_sk}#{sha256(token)}" —
    deterministic (Lambda retry on the same SFN state overwrites, never
    duplicates) and prefix-queryable per lock.
    """
    sk = f"{lock_sk}#{hashlib.sha256(task_token.encode()).hexdigest()}"
    put_item({
        "pk":            _PENDING_PK,
        "sk":            sk,  # lock derivable: sk.rsplit("#", 1)[0]
        "task_token":    task_token,
        "uuid":          uuid,
        "execution_arn": execution_arn,
        "created_at":    int(time.time() * 1000),
        "ttl":           ttl_from_now(86400),
    })
    log.info("Stored pending waiter for lock=%s uuid=%s", lock_sk, uuid)


def remove_pending(lock_sk: str, task_token: str) -> None:
    """Dequeue our own waiter entry (lost-wakeup self-service path)."""
    sk = f"{lock_sk}#{hashlib.sha256(task_token.encode()).hexdigest()}"
    delete_item(_PENDING_PK, sk)


def requeue_pending(item: dict) -> None:
    """
    Put a popped waiter back at its ORIGINAL queue position (created_at
    preserved). Used when SendTaskSuccess fails TRANSIENTLY after
    pop_next_pending already dequeued the item — without this, the waiter
    is orphaned: no pending record, no future REMOVE event targets it.
    """
    put_item(dict(item))
    log.info("Re-queued pending waiter sk=%s uuid=%s",
             item.get("sk", "?"), item.get("uuid"))


def pop_next_pending(lock_sk: str) -> dict | None:
    """
    Re-acquire ONE lock on behalf of its oldest waiter and dequeue it.
    Returns the pending item (with task_token), or None when no waiters
    exist for this lock or another handler won the race.

    Crash-window ordering: read waiters → acquire lock (conditional write)
    → delete pending item. A crash between acquire and delete self-heals
    via the lock TTL → stream REMOVE → re-entry (waiter still queued).
    """
    table = get_table()
    resp  = table.query(
        KeyConditionExpression=(
            Key("pk").eq(_PENDING_PK) & Key("sk").begins_with(f"{lock_sk}#")
        ),
        Limit=20,
    )
    items = resp.get("Items", [])
    if not items:
        return None

    oldest        = min(items, key=lambda x: int(x.get("created_at", 0)))
    execution_arn = oldest.get("execution_arn", "")

    # Strict (non-re-entrant) acquire: the waiter's own lost-wakeup retry
    # may race us with the SAME execution_arn — exactly one resumer may win.
    if not acquire(lock_sk, holder="index", execution_arn=execution_arn,
                   reentrant=False):
        log.info("Lock %s re-acquisition failed — another handler won", lock_sk)
        return None

    # Lock is ours — dequeue. On delete failure, release immediately so the
    # stream fires again and retry happens without waiting out the lock TTL.
    try:
        delete_item(_PENDING_PK, oldest["sk"])
    except Exception as exc:
        log.warning("pending delete failed for %s (%s) — releasing lock for fast retry",
                    oldest["sk"], exc)
        try:
            release(lock_sk)
        except Exception:
            pass
        return None

    log.info("Popped pending waiter for lock=%s uuid=%s", lock_sk, oldest.get("uuid"))
    return oldest
