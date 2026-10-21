"""
HandleLockRelease — DynamoDB Stream handler for per-table lock releases.

Triggered on REMOVE events from LanceDBIndexState (CFN FilterCriteria
matches pk="lock" — ANY sk). Extracts the released lock's sk from the
stream record's Keys and resumes the oldest waiter queued for exactly
that lock (per-lock pending queue, README.md Design record §1.4). Releases of
different locks in one batch are handled independently.
"""
from __future__ import annotations

import json
import logging
import traceback

import boto3

from lib.input_schema import AWS_REGION
from lib.lock_utils import pop_next_pending, release, requeue_pending

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions", region_name=AWS_REGION)


def _released_lock_sk(record: dict) -> str | None:
    """Return the lock sk for a REMOVE-on-pk=lock record, else None."""
    if record.get("eventName") != "REMOVE":
        return None
    keys = record.get("dynamodb", {}).get("Keys", {})
    if keys.get("pk", {}).get("S", "") != "lock":
        return None
    return keys.get("sk", {}).get("S", "") or None


def lambda_handler(event: dict, context) -> dict:
    records  = event.get("Records", [])
    lock_sks = {sk for r in records if (sk := _released_lock_sk(r))}
    if not lock_sks:
        return {"processed": 0}

    log.info("HandleLockRelease: %d lock release(s): %s",
             len(lock_sks), sorted(lock_sks))
    processed = 0
    for lock_sk in sorted(lock_sks):
        try:
            pending = pop_next_pending(lock_sk)
            if not pending:
                continue
            task_token = pending["task_token"]
            try:
                _sfn.send_task_success(
                    taskToken=task_token,
                    output=json.dumps({"lock_acquired": True}),
                )
                processed += 1
                log.info("Resumed waiter for lock=%s uuid=%s",
                         lock_sk, pending.get("uuid"))
            except (_sfn.exceptions.TaskDoesNotExist,
                    _sfn.exceptions.TaskTimedOut,
                    _sfn.exceptions.InvalidToken) as exc:
                # DEAD token (execution aborted / heartbeat expired): release
                # the just-re-acquired lock so the NEXT waiter gets a chance
                # via the new REMOVE event. Cascades until the queue drains.
                log.warning("Dead token for lock=%s (%s) — releasing to "  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
                            "cascade", lock_sk, exc)
                try:
                    release(lock_sk)
                except Exception:
                    pass
            except Exception as exc:
                # TRANSIENT SFN failure (throttle/5xx): the token is alive but
                # the waiter is already dequeued — re-queue it at its original
                # position BEFORE releasing, or no future REMOVE event could
                # ever resume it (permanent orphan until heartbeat timeout).
                log.warning("Transient SendTaskSuccess failure for lock=%s "
                            "(%s) — re-queueing waiter and releasing for "
                            "retry", lock_sk, exc)
                try:
                    requeue_pending(pending)
                finally:
                    try:
                        release(lock_sk)
                    except Exception:
                        pass
        except Exception as exc:
            log.error("HandleLockRelease failed for %s — %s: %s",
                      lock_sk, type(exc).__name__, exc)
            traceback.print_exc()
    return {"processed": processed}
