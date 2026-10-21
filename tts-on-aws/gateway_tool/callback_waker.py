"""
Callback waker for the SYNTHESIS machine — derived from
bedrock-inference's callback_waker (the notification mechanism's master;
its docstring carries the full design reasoning), with ONE tts-specific
addition: BEFORE completing any token it persists the run's verdict to
{output_s3_prefix}/outputs.json — the poll's outputs[] shape on
SUCCEEDED, a {status, error} failure doc on any other terminal status —
because a waitForTaskToken consumer reads results from S3, never from
the callback payload (the payload is a signal, not a data channel). The
persist uses synthesis_outputs, the gateway dispatcher's OWN flattener
(same zip, one parser), so the file and a live poll can never disagree.
synthesis_outputs reads no environment — importing lambda_function from
here raised KeyError on this function's minimal env and outputs.json was
never written (review 2026-08-31).

Consumers: audio-description's synth/repair loops (submit with
`callback_task_token`; read outputs.json by pointer). register_voice has
no callback path — nothing chains on a voice registration.
"""
from __future__ import annotations

import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

import synthesis_outputs
from callback_registry import TOKEN_GONE

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_ddb = boto3.client("dynamodb")
_sfn = boto3.client("stepfunctions")
_s3 = boto3.client("s3")

CALLBACK_TABLE = os.environ.get("CALLBACK_TABLE", "")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")


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

    # BEFORE any token completes: the woken consumer's next act is
    # reading this file. A persist failure is logged loudly and the
    # tokens still complete — the consumer then fails reading a
    # missing outputs.json, which is honest and diagnosable, unlike
    # sleeping to its deadman.
    try:
        _persist_verdict(detail, execution_arn, status, rows)
    except Exception as exc:                               # noqa: BLE001
        logger.error("outputs.json persist failed for %s: %s — "
                     "completing tokens anyway", execution_arn, exc)

    completed, gone, failed = 0, 0, 0
    for row in rows:
        token = row["token"]["S"]
        payload = json.loads(row["handle"]["S"])
        payload["status"] = status
        payload["message"] = (
            f"completed via callback: the synthesis run reached {status}; "
            "read outputs.json at output_s3_prefix for per-clip results")
        try:
            _sfn.send_task_success(taskToken=token,
                                   output=json.dumps(payload))
            completed += 1
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in TOKEN_GONE:
                logger.warning("token for %s already gone (%s) — the "  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
                               "consumer's deadman or a duplicate event "
                               "beat us", execution_arn, code)
                gone += 1
            else:
                logger.error("SendTaskSuccess failed for %s: %s — keeping "
                             "the row and failing this invocation so the "
                             "async retry redelivers", execution_arn, code)
                failed += 1
                continue          # keep the row for the retry
        _ddb.delete_item(
            TableName=CALLBACK_TABLE,
            Key={"execution_arn": row["execution_arn"],
                 "token_hash": row["token_hash"]})

    logger.info("%s %s: completed %d callback(s), %d already gone",
                execution_arn, status, completed, gone)
    if failed:
        # Redelivery only happens if this invocation ERRORS — a normal
        # return marks the async invoke successful and nothing ever
        # retries (review 2026-08-31: the row was kept for a redelivery
        # that could not occur, and the consumer slept to its deadman).
        # Completed rows are already deleted, and a retried completion
        # of a since-resolved token is a benign TOKEN_GONE, so raising
        # here is idempotent.
        raise RuntimeError(
            f"{failed} SendTaskSuccess call(s) failed for {execution_arn}"
            " — raising so the async retry redelivers the event")
    return {"completed": completed, "gone": gone}


def _persist_verdict(detail: dict, execution_arn: str, status: str,
                     rows: list) -> None:
    """SUCCEEDED: the poll shape, parsed from the event detail itself
    (it carries `output` and `executionArn` — the only keys the parser
    reads) with DescribeExecution as the fallback for a truncated
    `output`. Any other terminal status: the failure doc, at the
    output_s3_prefix the registered handle already carries."""
    if status == "SUCCEEDED":
        resp = detail if detail.get("output") \
            else _sfn.describe_execution(executionArn=execution_arn)
        synthesis_outputs.persist_outputs(_s3, resp, PROXY_BUCKET)
        return
    for row in rows:
        prefix = (json.loads(row["handle"]["S"])
                  .get("output_s3_prefix"))
        if prefix:      # every waiter of one run shares one prefix
            synthesis_outputs.persist_failure(
                _s3, prefix, status,
                detail.get("cause") or detail.get("error"),
                execution_arn)
            return


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
