"""
StateAcquireLease — slot admission for the sync path (README.md §M2.2).

  slot free → try_acquire succeeds → SendTaskSuccess immediately
  slot full → store_pending; the execution stays parked on its token until a
              drain admits it in FIFO order

Re-entrant: a Lambda retry after a successful acquire lands in `try_acquire`'s
own-holder branch and re-sends task success. Idempotent — a duplicate send on a
consumed token raises `TaskDoesNotExist`, swallowed here.

**This module used to live in another stack, and that was the point.** It was
`service-quotas-on-aws/states/acquire/`, on a function that held
`states:SendTask*` and could not call Bedrock, because a task token is an opaque
bearer credential that cannot be resource-scoped. The merge narrowed that isolation
from one stack to one role (README.md §M2.4); the tier collapse then put every
state on the same role, so the function that redeems the token is now also the one
that can call the model. Nothing outside this file's own discipline separates them.

The drain is NOT here. The standalone stack routed `{"operation": "Drain"}` to this
same function precisely so the decrement handler could hold no `SendTask*`; with
everything on one role that indirection buys nothing, so
`states/handle_semaphore_release` calls `drain_pending` directly. The async
self-invoke and its `ACQUIRE_FUNCTION_NAME` env var are gone.
"""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import ClientError

from lib import semaphore_client as sem

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_SFN = None


def _sfn():
    """
    Lazy: this module is imported by the router only on the acquire branch, but the
    suite imports it directly, and a client built at import time would make an
    offline test pass or fail according to the developer's shell.
    """
    global _SFN
    if _SFN is None:
        _SFN = boto3.client("stepfunctions")
    return _SFN


def send_task_success(task_token: str) -> None:
    """
    Raises botocore `ClientError` unchanged — `drain_pending` reads the error code to
    tell a dead token from a transient Step Functions failure, and swallowing it here
    would make a real outage look like a delivered completion.

    Public because `states/handle_semaphore_release` injects it into `drain_pending`:
    one sender, one place the token is redeemed.
    """
    _sfn().send_task_success(taskToken=task_token, output="{}")


def lambda_handler(event: dict, context) -> dict:
    service = event.get("service")
    resource = event.get("resource")
    execution_arn = event.get("execution_arn")
    task_token = event.get("task_token")
    # No default (semaphore invariant 4). A stack-level default here would be the
    # wrong cap applied silently whenever the field was omitted; the ASL passes
    # $.data.quota.max_concurrent, which validate_plan always sets.
    max_concurrent = event.get("max_concurrent")

    if not task_token:
        raise sem.ValidationException(
            "task_token is required — Acquire lease is a waitForTaskToken task")

    held = sem.try_acquire(service, resource, execution_arn, max_concurrent)
    if held:
        logger.info("%s slot on %s#%s for %s", held, service, resource, execution_arn)
        try:
            send_task_success(task_token)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in sem.DEAD_TOKEN_CODES:
                raise
            logger.warning("task token already consumed for %s", execution_arn)  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
        return {"acquired": True, "mode": held}

    logger.info("saturated on %s#%s — queueing %s", service, resource, execution_arn)
    sem.store_pending(service, resource, execution_arn, task_token, max_concurrent)
    return {"acquired": False, "queued": True}
