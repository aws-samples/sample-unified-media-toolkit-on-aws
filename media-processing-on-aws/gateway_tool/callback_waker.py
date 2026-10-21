"""
Callback waker — completes waiting consumers' task tokens when an
orchestrator execution reaches a terminal state (README.md §11).

Consumer workflows submit through the gateway with an optional
`callback_task_token` and sleep on ONE `waitForTaskToken` state. This
handler is the service-side other half: an EventBridge rule on THIS
deployment's orchestrator machine (native Step Functions status-change
events — the same mechanism `.sync` uses internally) invokes it with the
terminal event; it completes every token registered for that execution.

Notification is a SERVICE concern, owned here ONCE: the consumer-side
alternative (a table + waker + rule in every consumer stack) multiplies
by N workflows and fans every completion out to N rules for N-1 misses.

- ALWAYS SendTaskSuccess, carrying the terminal status: a FAILED verdict
  is DATA for the consumer's Choice state (partial results, honest
  reporting), never a mid-machine detonation.
- The payload is the stored submit handle + `status` (+ `message`) — with
  waitForTaskToken the submit Lambda's return value is discarded, so this
  payload IS the consumer's state output and must stand alone.
- A token already completed/timed out (deadman fired first, duplicate
  event delivery) is logged and dropped — the row is deleted either way.
- Foreign executions of the shared machine (agent-submitted runs, no
  registered waiter) are one Query miss, dropped silently.
"""
from __future__ import annotations

import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

from callback_registry import TOKEN_GONE as _TOKEN_GONE  # one home for
# the benign-race code list — the waker and the registry must agree on
# which SendTaskSuccess losses are benign (fi's convention, 2026-09-05)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_ddb = boto3.client("dynamodb")
_sfn = boto3.client("stepfunctions")

CALLBACK_TABLE = os.environ.get("CALLBACK_TABLE", "")


def _compact_cause(raw) -> str:
    """A Lambda error cause is JSON whose errorMessage/errorType are the
    condition — a stack trace routinely pushes it past any sane payload
    bound, and a blind truncation yields INVALID JSON the consumer can no
    longer parse cleanly. Re-serialize just the two fields that matter
    (parse-safe by construction); fall back to a bounded raw slice when
    the cause is not JSON."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # ANY JSON dict compacts parse-safely — errorType-only (an
            # exception raised with empty args) and even neither-key
            # shapes included: a blind truncation of valid JSON is the
            # exact defect this function exists to kill, so the raw
            # slice below is reserved for causes that are not JSON
            message = str(parsed.get("errorMessage") or "")[:400]
            etype = str(parsed.get("errorType") or "")[:100]
            # bound the SERIALIZED string, not the input: escaping can
            # double a newline/quote-heavy message past any input cap
            while True:
                out = json.dumps({"errorMessage": message,
                                  "errorType": etype})
                if len(out) <= 600:
                    return out
                if message:
                    message = message[: int(len(message) * 0.7)]
                else:
                    # a pathological errorType (non-ASCII escapes 6x) can
                    # alone breach the bound — shrink it last
                    etype = etype[: max(0, int(len(etype) * 0.7))]
                    if not etype:
                        return json.dumps({"errorMessage": "",
                                           "errorType": ""})
    except (TypeError, ValueError):
        pass
    return str(raw)[:600]


def lambda_handler(event: dict, context) -> dict:
    detail = (event or {}).get("detail") or {}
    execution_arn = detail.get("executionArn") or ""
    status = detail.get("status") or ""
    if not execution_arn or not status:
        logger.warning("not a Step Functions status-change event; ignoring")
        return {"completed": 0}

    rows = _query_all(execution_arn)
    if not rows:
        # An agent-submitted run with no waiter — the common case.
        logger.info("no registered callbacks for %s", execution_arn)
        return {"completed": 0}

    completed, gone, stuck = 0, 0, 0
    for row in rows:
        token = row["token"]["S"]
        payload = json.loads(row["handle"]["S"])
        # MUST-MATCH: the gateway dispatcher's absorbed-into-finished
        # branch builds this same payload shape — edit both together
        payload["status"] = status
        if status == "SUCCEEDED":
            payload["message"] = (
                "completed via callback: the media-processing run reached "
                "SUCCEEDED; outputs are at output_s3_prefix (see the output "
                "contract for per-group keys)")
        else:
            # the status-change detail carries the execution's error/cause
            # on FAILED (the tts waker has consumed them in production
            # since 2026-09) — forward them so the consumer can name the
            # condition instead of reporting "see the run prefix" over an
            # empty prefix. cause is a Lambda error JSON incl. stack trace;
            # truncate, the errorMessage lives in the head.
            if detail.get("error"):
                payload["error"] = detail["error"]
            if detail.get("cause"):
                payload["cause"] = _compact_cause(detail["cause"])
            if "error" in payload or "cause" in payload:
                payload["message"] = (
                    f"completed via callback: the media-processing run "
                    f"reached {status} — see error/cause for the condition")
            else:
                # TIMED_OUT/ABORTED status-change details carry neither
                payload["message"] = (
                    f"completed via callback: the media-processing run "
                    f"reached {status}; the status-change event carried no "
                    f"error detail")
        try:
            _sfn.send_task_success(taskToken=token,
                                   output=json.dumps(payload))
            completed += 1
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in _TOKEN_GONE:
                logger.warning("token for %s already gone (%s) — the "  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
                               "consumer's deadman or a duplicate event "
                               "beat us", execution_arn, code)
                gone += 1
            else:
                # AccessDenied lands here — loud, because the consumer is
                # now sleeping toward its deadman timeout.
                logger.error("SendTaskSuccess failed for %s: %s",
                             execution_arn, code)
                stuck += 1
                continue          # keep the row for the retried delivery
        _ddb.delete_item(
            TableName=CALLBACK_TABLE,
            Key={"execution_arn": row["execution_arn"],
                 "token_hash": row["token_hash"]})

    logger.info("%s %s: completed %d callback(s), %d already gone",
                execution_arn, status, completed, gone)
    if stuck:
        # EventBridge only redelivers on FUNCTION ERROR — a normal return
        # is recorded as success and the once-only terminal event never
        # comes again, stranding the kept rows until the consumers'
        # deadmen. Raising is what makes "keep the row" a recovery path
        # (fi's fix, backported 2026-09-05).
        raise RuntimeError(
            f"{stuck} callback completion(s) failed transiently for "
            f"{execution_arn}; raising so EventBridge redelivers")
    return {"completed": completed, "gone": gone}


def _query_all(execution_arn: str) -> list:
    rows, kwargs = [], {
        "TableName": CALLBACK_TABLE,
        "KeyConditionExpression": "execution_arn = :arn",
        "ExpressionAttributeValues": {":arn": {"S": execution_arn}},
    }
    while True:
        resp = _ddb.query(**kwargs)
        rows.extend(resp.get("Items") or [])
        if "LastEvaluatedKey" not in resp:
            return rows
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
