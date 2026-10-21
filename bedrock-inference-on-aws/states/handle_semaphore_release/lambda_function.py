"""
StateHandleSemaphoreRelease — the quota table's stream handler (README.md §M2.2).

Triggered by REMOVE on holder items only (`SemaphoreStreamMapping`'s
`FilterCriteria`: `eventName=REMOVE`, pk prefix `semaphore#`, sk prefix `holder#`).
All THREE holder-deletion paths converge here: normal release, the EventBridge
terminal-state rule, and DDB TTL expiry.

Per record: exactly-once counter decrement, deduped on the stream record's own
`eventID` inside the decrementing transaction, because streams are at-least-once.
Then one drain per distinct scope.

**The drain is called directly, and that is the merge's one behavioural change
here.** The standalone stack invoked `AcquireFunction` asynchronously
(`InvocationType="Event"`) so that this handler could hold no `states:SendTask*`.
Both handlers now share one role, so the indirection protects nothing and
only adds a hop that can fail silently — `handle_execution_terminal`'s sibling
reasoning applies: the capability is on the role either way. What the async invoke
*did* buy is noted and given up deliberately: a drain failure can now fail the
stream record, which the stream will redeliver, and the redelivered decrement dedups
to False while the drain runs again. That is the intended shape — an idempotent
decrement in front of a retryable drain.

The scope comes from each record's own keys, never from an event field: the keys are
the only source that cannot disagree with the item that was actually removed. And
the drain is per scope, not global — this table is multi-scope (one cap per model),
so a single argless drain would only ever service one of them.
"""

from __future__ import annotations

import logging

from lib import semaphore_client as sem
from states.acquire_lease.lambda_function import send_task_success

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    decremented = 0
    scopes: set[tuple[str, str]] = set()

    for record in event.get("Records") or []:
        keys = ((record.get("dynamodb") or {}).get("Keys") or {})
        pk = (keys.get("pk") or {}).get("S") or ""
        sk = (keys.get("sk") or {}).get("S") or ""
        if not pk.startswith(sem.SEMAPHORE_PK_PREFIX) \
                or not sk.startswith(sem.HOLDER_SK_PREFIX):
            # The FilterCriteria should have excluded this. Log rather than raise: a
            # raise fails the whole batch and the stream redelivers it forever. The
            # record this guard really exists for is a `stream-dedup` marker's own
            # TTL expiry, which arrives as a REMOVE and must never read as a
            # released holder.
            logger.warning("unexpected stream record keys pk=%r sk=%r", pk, sk)
            continue

        service, resource = sem.parse_semaphore_pk(pk)
        scopes.add((service, resource))
        if sem.decrement_once(service, resource, record.get("eventID") or ""):
            decremented += 1

    # Drain UNCONDITIONALLY, once per scope — not only for scopes that decremented.
    # Streams redeliver records after a crash: the retry's decrements all dedup to
    # False, but the crashed attempt may have died mid-drain or before draining at
    # all. If the removed holder was the LAST one on that scope, no future REMOVE
    # will ever fire, and skipping the drain would strand every waiter until its
    # acquire task token times out 24 h later. A drain on an empty queue is one
    # cheap Query returning 0.
    admitted = 0
    for service, resource in sorted(scopes):
        admitted += sem.drain_pending(service, resource, send_task_success)

    logger.info("stream: %d decremented, %d scope(s) drained, %d admitted",
                decremented, len(scopes), admitted)
    return {"decremented": decremented, "scopes_drained": len(scopes),
            "admitted": admitted}
