"""
StateAcquireSemaphore — waitForTaskToken slot admission (README.md §5).

Slot free  → try_acquire succeeds → SendTaskSuccess immediately.
Slot full  → store_pending; the execution waits on the token until the DDB
             stream handler drains it (FIFO) after a holder release.

Re-entrant: a Lambda retry after a successful acquire re-lands in
try_acquire's own-holder branch and re-sends task success (idempotent —
a duplicate SendTaskSuccess on a consumed token raises TaskDoesNotExist,
swallowed here).
"""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import ClientError

from lib import semaphore

logger = logging.getLogger(__name__)

_sfn = boto3.client("stepfunctions")


def lambda_handler(event: dict, context) -> dict:
    execution_arn = event["execution_arn"]
    task_token = event["task_token"]
    requested_max = event["max_concurrent_executions"]

    if semaphore.try_acquire(execution_arn, requested_max):
        logger.info("slot acquired for %s", execution_arn)
        try:
            _sfn.send_task_success(taskToken=task_token, output="{}")
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in (
                    "TaskDoesNotExist", "TaskTimedOut"):
                raise
            logger.warning("task token already consumed for %s", execution_arn)  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
        return {"acquired": True}

    logger.info("slots full — queueing %s", execution_arn)
    semaphore.store_pending(execution_arn, task_token, requested_max)
    return {"acquired": False, "queued": True}
