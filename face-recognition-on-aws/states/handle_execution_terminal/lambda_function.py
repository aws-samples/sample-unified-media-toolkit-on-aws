"""
StateHandleExecutionTerminal — EventBridge handler on ALL terminal states
(SUCCEEDED, FAILED, TIMED_OUT, ABORTED) of the face-recognition state machine
(README.md §2; user decision).

One uniform cleanup path, both actions idempotent:
  1. Delete the semaphore holder item (no-op when the normal Release state
     already ran, or when the failure preceded acquire).
  2. Delete the sandbox collection "sandbox-{execution_name}" — fired
     directly, ResourceNotFound swallowed (user decision: no existence
     pre-check, saves an API call).

execution_name == workflow_id (the gateway names executions by workflow_id),
so the sandbox name is derivable from the EventBridge detail alone.
"""

from __future__ import annotations

import logging

from lib import semaphore
from lib.rekognition_utils import delete_collection_ignore_missing

logger = logging.getLogger(__name__)


def lambda_handler(event: dict, context) -> dict:
    detail = event.get("detail") or {}
    execution_arn = detail.get("executionArn") or ""
    status = detail.get("status") or ""
    if not execution_arn:
        logger.warning("terminal event without executionArn: %s", event)
        return {"handled": False}

    execution_name = execution_arn.rsplit(":", 1)[-1]
    logger.info("terminal %s for %s", status, execution_name)

    released = semaphore.release_holder(execution_arn)
    delete_collection_ignore_missing(f"sandbox-{execution_name}")

    return {"handled": True, "status": status, "released": released}
