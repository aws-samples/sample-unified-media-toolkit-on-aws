"""
StateReleaseLease — give the slot back (README.md §5).

Appears **ONCE**, on the sync child's happy path. It used to appear twice: the second
occurrence was `Release lease on failure`, the target of a `Catch` on every
post-acquire state, and the merge deleted it (README.md §M3.2). The failure path is
now the terminal-state EventBridge rule, which covers strictly more: a `Catch` cannot
fire for an ABORTED execution or a Lambda killed between releasing and returning,
which are precisely the cases a cleanup path exists for. Two mechanisms for one job,
where one of them handles a subset of the other's cases, is a second thing to keep
correct for no coverage gained.

So three holder-disappearance triggers, not four: this state, the terminal rule, and
the holder's own TTL. That is why this state is allowed to be best-effort.

**Never raises.** The reason is no longer that a raise would clobber a `Catch`'s error
— there is no `Catch` — but that a release failure is not worth failing a completed
inference over. Every record is already answered and `response.jsonl` is already
written; raising here would turn a successful run into a failed one, and the slot comes
back from the terminal rule or the TTL either way. So: log loudly, return the release
status, and let the ASL continue to `Summarize`.
"""
from __future__ import annotations

import logging

from lib import semaphore_client

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    data = event.get("data") or {}
    quota = data.get("quota") or {}
    service = quota.get("service") or ""
    resource = quota.get("resource") or ""
    execution_arn = (event.get("execution_arn") or "").strip()

    released, detail = _release(service, resource, execution_arn)

    return {
        "operation": "StateReleaseLease",
        "status": "COMPLETED",
        "uuid": event.get("uuid") or "",
        "input": event.get("input") or {},
        "data": {
            **data,
            "lease": {"released": released, "detail": detail},
        },
    }


def _release(service: str, resource: str, execution_arn: str) -> tuple[bool, str]:
    if not (service and resource and execution_arn):
        # Reached when the Catch fires *before* Acquire — a ValidationException in
        # Validate & plan, say. There is nothing to release, and that is not an error.
        return False, "no lease to release (no quota scope on the event)"

    try:
        semaphore_client.release(service, resource, execution_arn)
    except Exception as exc:                                       # noqa: BLE001
        # Loud, because the slot is now held until the TTL or the terminal-state rule
        # reclaims it — correct, but with up to an hour of unnecessary queueing for
        # the next waiter.
        logger.error(
            "RELEASE FAILED for %s#%s (%s): %s — the slot stays held until the "
            "terminal-state rule or the holder TTL reclaims it",
            service, resource, execution_arn, exc)
        return False, f"{type(exc).__name__}: {exc}"

    logger.info("released %s#%s for %s", service, resource, execution_arn)
    return True, "released"
