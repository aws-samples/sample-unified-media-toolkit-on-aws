"""
HandleExecutionFailure — EventBridge handler for SFN execution failures.

On FAILED / ABORTED / TIMED_OUT of an Index or Rebuild execution, releases
EVERY lock that execution still holds (an execution acquires one lock per
Map iteration and may hold several when it dies). Conditional deletes —
safe under duplicate EventBridge delivery.
"""
from __future__ import annotations

import json
import logging
import traceback

from lib.lock_utils import release_all_for_execution

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_TERMINAL_FAILURE_STATUSES = {"FAILED", "ABORTED", "TIMED_OUT"}


def lambda_handler(event: dict, context) -> dict:
    detail        = event.get("detail", {})
    status        = detail.get("status", "")
    execution_arn = detail.get("executionArn", "")

    if status not in _TERMINAL_FAILURE_STATUSES:
        return {"released": 0, "reason": "not_a_failure"}
    if not execution_arn:
        log.warning("No executionArn in event detail")
        return {"released": 0, "reason": "missing_execution_arn"}

    try:
        released = release_all_for_execution(execution_arn)
        return {"released": released, "execution_arn": execution_arn,
                "status": status}
    except Exception as exc:
        log.error("HandleExecutionFailure failed — %s: %s",
                  type(exc).__name__, exc)
        traceback.print_exc()
        raise
