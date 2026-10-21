"""
NotifyFunction — the **only** holder of `states:SendTask*` in this deployment (§6.3).

Triggered by the service-token table's stream on a transition **to** a terminal status.
Reads the token off the row and resumes the parked execution. It makes no Bedrock call
and never submits anything; `DrainFunction` gets `bedrock:*` and no Step Functions.
Neither can do the other's damage.

Why the grant cannot be narrowed, and therefore why this function exists at all: a task
token is an opaque bearer credential, so `states:SendTaskSuccess` **cannot be
resource-scoped** — the policy resource is effectively `*`, and anything holding it can
complete any execution whose token it obtains. Concentrating that in one function with
one trigger and ~60 lines of code is the whole mitigation (§1.2).

**Idempotency.** Streams are at-least-once, so this will occasionally send twice. The
second send raises `TaskDoesNotExist` or `TaskTimedOut`; those two are swallowed and
**everything else re-raises**. A broad `except` here is a listed silent-failure site
(§12.3): it turns a genuine Step Functions outage into a delivered completion, and the
caller waits out its full 24 h timeout with the answer already in S3.

Same dead-token distinction `semaphore_client.drain_pending` makes, and deliberately the
same two error codes.
"""
from __future__ import annotations

import logging

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from lib import token_table

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# The exact two codes that mean "this token can never be redeemed again". Any other
# failure means the token is still live and the delivery must be retried.
DEAD_TOKEN_CODES = ("TaskDoesNotExist", "TaskTimedOut")

_SFN = None


def _sfn():
    global _SFN
    if _SFN is None:
        _SFN = boto3.client(
            "stepfunctions",
            config=Config(retries={"max_attempts": 4, "mode": "standard"}))
    return _SFN


def lambda_handler(event: dict, context) -> dict:
    notified, skipped, dead = 0, 0, 0

    for record in event.get("Records") or []:
        outcome = _handle(record)
        if outcome == "notified":
            notified += 1
        elif outcome == "dead":
            dead += 1
        else:
            skipped += 1

    logger.info("notified=%d dead_token=%d skipped=%d", notified, dead, skipped)
    return {"notified": notified, "dead_token": dead, "skipped": skipped}


def _handle(record: dict) -> str:
    if record.get("eventName") not in ("INSERT", "MODIFY"):
        # A REMOVE is the drain's trigger, not the notifier's — the token was already
        # redeemed before the row was deleted.
        return "skipped"

    new = token_table.stream_row(record, "NewImage")
    old = token_table.stream_row(record, "OldImage")

    status = new.get("status") or ""
    if status not in token_table.TERMINAL_STATUSES:
        return "skipped"
    if (old.get("status") or "") in token_table.TERMINAL_STATUSES:
        # Terminal → terminal. Either a redelivery or a status_message rewrite; the
        # token was sent on the first transition. Checking the OLD image is what makes
        # this a *transition* trigger rather than a *state* trigger, and it is the
        # cheap half of the idempotency story (the SFN error is the other half).
        return "skipped"

    task_token = new.get("task_token") or ""
    workflow_id = new.get("sk") or ""
    model_id = new.get("model_id") or token_table.model_id_from_pk(new.get("pk", ""))
    if not task_token:
        logger.error("row %s is %s with no task_token — the execution cannot be "
                     "resumed and will wait out its own timeout", workflow_id, status)
        return "skipped"

    try:
        _send(task_token, status, new, workflow_id)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code not in DEAD_TOKEN_CODES:
            # Still live. Re-raise so DynamoDB retries the stream batch — a stream
            # handler failure is retried, which is precisely why this sits on the
            # stream rather than inline in the EventBridge handler (§6.4).
            logger.warning("SendTask* transient failure for %s: %s — the stream will "
                           "retry", workflow_id, code)
            raise
        logger.warning("dead task token for %s: %s — the caller is gone or timed out",  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
                       workflow_id, code)
        _cleanup(model_id, workflow_id)
        return "dead"

    _cleanup(model_id, workflow_id)
    return "notified"


def _send(task_token: str, status: str, row: dict, workflow_id: str) -> None:
    """
    Resume the parked execution.

    A `FAILED` job sends `SendTaskFailure`, not a success carrying a failure flag. The
    §12.3 error-as-success site: a caller handed `{"status": "FAILED"}` through the
    success channel has to remember to check it, and the one that forgets treats an
    error payload as a result. Step Functions has a failure channel; this uses it.
    """
    import json

    if status == token_table.STATUS_FAILED:
        _sfn().send_task_failure(
            taskToken=task_token,
            error="BedrockInferenceFailed",
            cause=(row.get("status_message")
                   or f"batch job for {workflow_id} did not complete")[:32768])
        logger.info("sent SendTaskFailure for %s", workflow_id)
        return

    _sfn().send_task_success(
        taskToken=task_token,
        output=json.dumps({
            "status": status,
            "workflow_id": workflow_id,
            "job_arn": row.get("job_arn") or "",
            "status_message": row.get("status_message") or "",
        }))
    logger.info("sent SendTaskSuccess for %s", workflow_id)


def _cleanup(model_id: str, workflow_id: str) -> None:
    """
    Delete the notified row. Two effects, both wanted: the token is gone so it cannot
    be sent twice, and the REMOVE stream record triggers the drain — capacity freed,
    so submit whatever is PENDING on this model (§6.4).

    Best-effort. A failed delete leaves a terminal row that expires on its TTL and
    whose stream re-delivery is caught by the terminal→terminal check above; raising
    here would instead re-run a delivery that already succeeded.
    """
    try:
        token_table.delete_row(model_id, workflow_id)
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("could not delete row %s/%s after notifying: %s — it will "
                       "expire on its TTL", model_id, workflow_id, exc)
