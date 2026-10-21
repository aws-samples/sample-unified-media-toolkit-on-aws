"""
StateHandleExecutionTerminal — EventBridge handler on ALL terminal states of all
three state machines (README.md §M3.4). The last of the three
holder-disappearance triggers that is not a TTL.

SUCCEEDED | FAILED | TIMED_OUT | ABORTED  →  delete every holder this execution
still owns.

Including SUCCEEDED is not redundant with the sync machine's own `Release lease`
state: it covers the window where the release Lambda deleted the holder, crashed
before returning, and the ASL never advanced — and it makes cleanup idempotent,
which is what lets one handler serve all four statuses with no branching.

**Why all three machines are on the rule, when only the sync child ever holds a
lease.** An orchestrator killed before its child starts holds nothing, so this is a
no-op for it. But an orchestrator ABORTED mid-`.sync:2` does not necessarily
terminate its child, so the child's own terminal event is the one that reclaims —
and the batch machine is listed so that a future lease on that path cannot become a
silent leak. `{"handled": True, "released": 0}` on a lease-less execution is the
expected common case, not a warning sign.

THIS HANDLER DELETES, AND ONLY DELETES. It does not decrement (semaphore invariant
1) and it does not drain. The delete reaches the stream, which decrements and
drains — that is what makes an aborted execution wake a waiter exactly as a clean
release does.

An EventBridge terminal event carries only `executionArn`, and the scope is part of
the partition key here, so the holder is found through the `lease#{execution_arn}`
reverse pointer rather than a GSI (see `semaphore_client`'s module docstring).
"""

from __future__ import annotations

import logging

from lib import semaphore_client as sem

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    detail = event.get("detail") or {}
    execution_arn = detail.get("executionArn") or ""
    status = detail.get("status") or ""

    if not execution_arn:
        # Not raised: a malformed event would be retried by EventBridge to no
        # purpose, and there is nothing to clean up.
        logger.warning("terminal event without executionArn: %s", event)
        return {"handled": False}

    leases = sem.list_leases(execution_arn)
    if not leases:
        # The common case. The rule fires on every terminal state of all three
        # machines — the orchestrator and the batch child never hold a lease, and a
        # sync child whose Release already ran holds none either.
        logger.info("terminal %s for %s — no lease held", status, execution_arn)
        return {"handled": True, "status": status, "released": 0}

    released = 0
    for service, resource in leases:
        if sem.release(service, resource, execution_arn):
            released += 1
            logger.info("terminal %s: released %s#%s for %s",
                        status, service, resource, execution_arn)

    return {"handled": True, "status": status, "released": released,
            "leases": len(leases)}
