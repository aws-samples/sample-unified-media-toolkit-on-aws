"""
StateAcquireLock — acquire ONE physical table's lock (sk = index_key).

Index path (waitForTaskToken, inside the Index SFN Map — one lock per
(collection, table) iteration):
  - Lock free → acquire + SendTaskSuccess immediately
  - Lock held → queue (task_token, index_key) in the per-lock pending
    queue → SFN waits until HandleLockRelease resumes it

Rebuild path (standard task): lock free → acquire; held → LockHeldError
(matched by the SFN Retry block).

Writers to DIFFERENT tables never contend (README.md Design record §1.4) — the
event's index_key selects the lock item.
"""
from __future__ import annotations

import json
import logging
import traceback

import boto3

from lib.input_schema import AWS_REGION
from lib.lock_utils import acquire, remove_pending, store_pending

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions", region_name=AWS_REGION)


class LockHeldError(Exception):
    pass


def lambda_handler(event: dict, context) -> dict:
    index_key     = event.get("index_key", "")
    task_token    = event.get("task_token")
    holder        = "index" if task_token else "rebuild"
    execution_arn = event.get("execution_arn", "")
    uuid          = event.get("uuid", "unknown")

    if not index_key:
        raise ValueError("StateAcquireLock: index_key is required")
    log.info("StateAcquireLock: index_key=%s holder=%s", index_key, holder)

    try:
        if acquire(index_key, holder=holder, execution_arn=execution_arn):
            if task_token:
                _sfn.send_task_success(
                    taskToken=task_token,
                    output=json.dumps({"lock_acquired": True}),
                )
                return {"lock_acquired": True}
            return {**event, "lock_acquired": True}

        if task_token:
            store_pending(index_key, task_token, uuid=uuid,
                          execution_arn=execution_arn)
            # Lost-wakeup guard: if the holder released BETWEEN our failed
            # acquire and the store_pending write, the stream handler saw an
            # empty queue and no further REMOVE event will come. Re-try once
            # now that our pending item is visible; on success, dequeue
            # ourselves and resume immediately. Strict (non-re-entrant): the
            # stream handler may resume us with our own execution_arn in the
            # same window — only one resumer may SendTaskSuccess.
            if acquire(index_key, holder=holder, execution_arn=execution_arn,
                       reentrant=False):
                remove_pending(index_key, task_token)
                _sfn.send_task_success(
                    taskToken=task_token,
                    output=json.dumps({"lock_acquired": True}),
                )
                return {"lock_acquired": True, "raced": True}
            return {"lock_acquired": False, "queued": True}
        raise LockHeldError(f"Lock {index_key} held — rebuild will retry")

    except LockHeldError:
        raise
    except Exception as exc:
        if task_token:
            # Fail the SFN task explicitly so it doesn't hang on heartbeat.
            # Cleanup (if the lock WAS acquired) falls to the EventBridge
            # failure handler's release_all_for_execution.
            try:
                _sfn.send_task_failure(taskToken=task_token,
                                       error=type(exc).__name__, cause=str(exc))
            except Exception:
                pass
        log.error("StateAcquireLock failed — %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise
