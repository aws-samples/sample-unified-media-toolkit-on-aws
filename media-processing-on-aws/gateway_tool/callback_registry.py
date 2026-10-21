"""
Callback-token row format (README.md §11) — the ONE place the
(execution_arn, token_hash) key scheme lives, shared by the gateway
dispatcher (register / synchronous-complete / delete) and read back by
callback_waker.

A separate module for a deliberate reason: gateway_tool/lambda_function.py
is pinned to contain NO hashing at all (`assertNotIn("hashlib", ...)`) so
the workflow-id hash can never be quietly reimplemented beside its one
true home in lib/identity. The token hash here is a ROW KEY, not an
identity: sha256(token)[:32] keys multiple waiters absorbed into one run,
nothing more.

Clients are passed in, never constructed here — the dispatcher's tests
patch its module-level clients, and a second set of clients in this module
would dodge those patches.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from botocore.exceptions import ClientError

logger = logging.getLogger()

TTL_SECONDS = 8 * 24 * 3600      # past any batch SLA + the consumer deadman

# Benign SendTaskSuccess races: the consumer's deadman fired first, or a
# duplicate delivery already completed the token.
TOKEN_GONE = ("TaskDoesNotExist", "TaskTimedOut", "InvalidToken")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]


def register(ddb, table: str, execution_arn: str, token: str,
             handle: dict) -> None:
    """The row callback_waker completes at the terminal event. Raises on
    failure — a waitForTaskToken caller can only see a raise."""
    ddb.put_item(
        TableName=table,
        Item={
            "execution_arn": {"S": execution_arn},
            "token_hash": {"S": token_hash(token)},
            "token": {"S": token},
            "handle": {"S": json.dumps(handle)},
            "ttl": {"N": str(int(time.time()) + TTL_SECONDS)},
        })


def delete(ddb, table: str, execution_arn: str, token: str) -> None:
    try:
        ddb.delete_item(
            TableName=table,
            Key={"execution_arn": {"S": execution_arn},
                 "token_hash": {"S": token_hash(token)}})
    except ClientError:
        logger.warning("callback row cleanup failed for %s — the TTL will "
                       "collect it", execution_arn)


def complete_now(sfn, token: str, payload: dict) -> None:
    """Synchronous completion for absorbed-finished runs. Token-gone races
    are benign; anything else raises — the waiting caller must fail fast,
    not sleep to its timeout."""
    try:
        sfn.send_task_success(taskToken=token, output=json.dumps(payload))
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in TOKEN_GONE:
            logger.warning("callback token already gone (%s)", code)  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
            return
        raise
