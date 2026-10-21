"""
The service-token table: one row per Bedrock batch job (README.md §6.1).

`{ResourcePrefix}-service-token`, streams `NEW_AND_OLD_IMAGES`, TTL on `ttl`:

```
pk = "batch#{model_id}"
sk = "{workflow_id}"
     status        = "PENDING" | "SUBMITTING" | "SUBMITTED" | "COMPLETED" | "FAILED"
     task_token    = "..."
     job_arn       = "..."          # set on a successful submit
     request_s3uri = "..."
     model_params  = {...}          # what the drain needs to resubmit
     ttl           = ...
```

**Why this is not the quota table.** `states:SendTaskSuccess` cannot be
resource-scoped — a task token is an opaque bearer credential, so the policy
resource is effectively `*`, and anything holding the grant can complete any
execution whose token it obtains. That is not a capability to hand to the quota
table's stream handler, which fires on every holder removal in the deployment.
Tokens live where tokens are used (§1.2).

**Why one table and not v5's two.** V5 splits this across a backlog table
(`status=PENDING` on a status GSI) and a service-token table
(`{token, service, api, responseData}` keyed by `backlogId`), joined on `jobName`.
The split exists because `register()` runs in one Lambda in one stack while the
completion arrives at a different Lambda in a different stack. Here both are in
this stack, so the token and the queue entry are one item and the join disappears.

One GSI, one purpose: the EventBridge completion handler receives a `batchJobArn`
and nothing else.
"""
from __future__ import annotations

import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

from lib.custom_exceptions import ValidationException

logger = logging.getLogger(__name__)

TABLE_NAME_ENV = "SERVICE_TOKEN_TABLE_NAME"
JOB_ARN_INDEX = "job_arn-index"

BATCH_PK_PREFIX = "batch#"

STATUS_PENDING = "PENDING"
STATUS_SUBMITTING = "SUBMITTING"
STATUS_SUBMITTED = "SUBMITTED"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED)

# Long enough that the row outlives the caller's own `waitForTaskToken`
# (`TimeoutSeconds: 86400`) plus the batch job's 24 h ceiling. A row that expires
# while its execution is still parked strands the caller until its task timeout —
# so the TTL is a leak backstop, never the mechanism.
ROW_TTL_SEC = int(os.environ.get("SERVICE_TOKEN_TTL_SEC", str(3 * 86400)))

# Bedrock batch job states → our row status. `PartiallyCompleted` is COMPLETED on
# purpose: per-record failures are read out of the output manifest and become FAILED
# lines in `response.jsonl` (§5.2), which is partial success, not a failed job.
BEDROCK_STATUS_MAP = {
    "Completed": STATUS_COMPLETED,
    "PartiallyCompleted": STATUS_COMPLETED,
    "Failed": STATUS_FAILED,
    "Stopped": STATUS_FAILED,
    "Expired": STATUS_FAILED,
}

_DDB = None


def _ddb():
    """
    Lazy, for the same reason as `semaphore_client._ddb`: importing this module must
    need neither credentials nor a region, or an offline suite passes or fails
    according to the developer's shell.
    """
    global _DDB
    if _DDB is None:
        _DDB = boto3.client("dynamodb")
    return _DDB


def table_name() -> str:
    name = os.environ.get(TABLE_NAME_ENV) or ""
    if not name:
        raise ValidationException(f"{TABLE_NAME_ENV} is not set")
    return name


def _now() -> int:
    return int(time.time())


def batch_pk(model_id: str) -> str:
    if not model_id or not isinstance(model_id, str):
        raise ValidationException(f"model_id is required, got {model_id!r}")
    return f"{BATCH_PK_PREFIX}{model_id}"


def model_id_from_pk(pk: str) -> str:
    if not str(pk or "").startswith(BATCH_PK_PREFIX):
        raise ValidationException(f"not a batch partition key: {pk!r}")
    return pk[len(BATCH_PK_PREFIX):]


# ── Serialization ─────────────────────────────────────────────────────────────
# The row carries `model_params` — a nested object the drain needs to resubmit.
# Hand-rolled rather than pulled in through `boto3.dynamodb.types`: this module
# already speaks the low-level client's shape everywhere else, and mixing the two
# is how a nested Decimal ends up in a JSON dump.

def _to_attr(value):
    if isinstance(value, bool):
        return {"BOOL": value}
    if value is None:
        return {"NULL": True}
    if isinstance(value, (int, float)):
        return {"N": str(value)}
    if isinstance(value, str):
        return {"S": value}
    if isinstance(value, dict):
        return {"M": {key: _to_attr(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"L": [_to_attr(item) for item in value]}
    raise ValidationException(
        f"cannot store {type(value).__name__} in the service-token table")


def _from_attr(attr: dict):
    if "S" in attr:
        return attr["S"]
    if "N" in attr:
        text = attr["N"]
        return int(text) if "." not in text and "e" not in text.lower() else float(text)
    if "BOOL" in attr:
        return attr["BOOL"]
    if "NULL" in attr:
        return None
    if "M" in attr:
        return {key: _from_attr(item) for key, item in attr["M"].items()}
    if "L" in attr:
        return [_from_attr(item) for item in attr["L"]]
    return None


def to_item(row: dict) -> dict:
    return {key: _to_attr(value) for key, value in row.items() if value is not None}


def from_item(item: dict) -> dict:
    return {key: _from_attr(value) for key, value in (item or {}).items()}


# ── Writes ────────────────────────────────────────────────────────────────────

def put_submitting(*, model_id: str, workflow_id: str, task_token: str,
                   request_s3uri: str, model_params: dict) -> dict:
    """
    Write the row **before** calling Bedrock (§6.2 step 1).

    Order matters: a row written after a successful submit would be lost if the
    Lambda died in between, leaving a running batch job whose completion event finds
    no row and no token — the caller parks for its full 24 h with the work already
    paid for.
    """
    if not task_token or not isinstance(task_token, str):
        raise ValidationException("task_token is required")
    if not workflow_id:
        raise ValidationException("workflow_id is required")

    row = {
        "pk": batch_pk(model_id),
        "sk": workflow_id,
        "status": STATUS_SUBMITTING,
        "task_token": task_token,
        "request_s3uri": request_s3uri,
        "model_params": model_params or {},
        "model_id": model_id,
        "created_at": int(time.time() * 1000),
        "ttl": _now() + ROW_TTL_SEC,
    }
    _ddb().put_item(TableName=table_name(), Item=to_item(row))
    return row


def _update(model_id: str, workflow_id: str, sets: dict,
            condition: str | None = None) -> bool:
    names, values, expression = {}, {}, []
    for index, (key, value) in enumerate(sets.items()):
        names[f"#k{index}"] = key
        values[f":v{index}"] = _to_attr(value)
        expression.append(f"#k{index} = :v{index}")
    kwargs = {
        "TableName": table_name(),
        "Key": {"pk": {"S": batch_pk(model_id)}, "sk": {"S": workflow_id}},
        "UpdateExpression": "SET " + ", ".join(expression),
        "ExpressionAttributeNames": names,
        "ExpressionAttributeValues": values,
    }
    if condition:
        kwargs["ConditionExpression"] = condition
    try:
        _ddb().update_item(**kwargs)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def mark_submitted(model_id: str, workflow_id: str, job_arn: str) -> bool:
    return _update(model_id, workflow_id,
                   {"status": STATUS_SUBMITTED, "job_arn": job_arn,
                    "submitted_at": int(time.time() * 1000)},
                   condition="attribute_exists(pk)")


def mark_pending(model_id: str, workflow_id: str, reason: str) -> bool:
    """
    Queue the row (§6.2 step 4). **Not an error to the caller** — the execution stays
    parked on its task token, and a drain submits it when capacity frees.
    """
    return _update(model_id, workflow_id,
                   {"status": STATUS_PENDING, "queued_reason": reason,
                    "queued_at": int(time.time() * 1000)},
                   condition="attribute_exists(pk)")


def mark_terminal(model_id: str, workflow_id: str, status: str,
                  job_arn: str | None = None,
                  message: str | None = None) -> bool:
    """
    Record a terminal status. The stream then triggers `NotifyFunction`, which is the
    only holder of `states:SendTask*` in this deployment (§6.3).

    Guarded on `status <> :terminal` so a redelivered completion event does not
    rewrite an already-terminal row — which would emit a second stream record and a
    second `SendTaskSuccess`. The notifier tolerates that (it swallows
    `TaskDoesNotExist`), but not generating it is cheaper than absorbing it.
    """
    if status not in TERMINAL_STATUSES:
        raise ValidationException(
            f"status must be one of {TERMINAL_STATUSES}, got {status!r}")
    sets = {"status": status, "completed_at": int(time.time() * 1000)}
    if job_arn:
        sets["job_arn"] = job_arn
    if message:
        sets["status_message"] = message

    names = {"#s": "status"}
    values = {":terminal_a": {"S": STATUS_COMPLETED},
              ":terminal_b": {"S": STATUS_FAILED}}
    expression, index = [], 0
    for key, value in sets.items():
        names[f"#k{index}"] = key
        values[f":v{index}"] = _to_attr(value)
        expression.append(f"#k{index} = :v{index}")
        index += 1
    try:
        _ddb().update_item(
            TableName=table_name(),
            Key={"pk": {"S": batch_pk(model_id)}, "sk": {"S": workflow_id}},
            UpdateExpression="SET " + ", ".join(expression),
            ConditionExpression=(
                "attribute_exists(pk) AND #s <> :terminal_a AND #s <> :terminal_b"),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def delete_row(model_id: str, workflow_id: str) -> None:
    """
    Delete a notified row. The resulting REMOVE stream record is the drain's first
    trigger (§6.4) — capacity freed, so submit whatever is PENDING on this model.
    """
    _ddb().delete_item(
        TableName=table_name(),
        Key={"pk": {"S": batch_pk(model_id)}, "sk": {"S": workflow_id}})


# ── Reads ─────────────────────────────────────────────────────────────────────

def get_row(model_id: str, workflow_id: str) -> dict | None:
    response = _ddb().get_item(
        TableName=table_name(),
        Key={"pk": {"S": batch_pk(model_id)}, "sk": {"S": workflow_id}},
        ConsistentRead=True)
    item = response.get("Item")
    return from_item(item) if item else None


def find_by_job_arn(job_arn: str) -> dict | None:
    """
    The row for a Bedrock job arn, via the `job_arn-index` GSI.

    This is the GSI's whole reason for existing: the EventBridge completion event
    carries a `batchJobArn` and nothing else, and v5 answers the same question with a
    second table and a `jobName` join.
    """
    if not job_arn:
        raise ValidationException("job_arn is required")
    response = _ddb().query(
        TableName=table_name(),
        IndexName=JOB_ARN_INDEX,
        KeyConditionExpression="job_arn = :arn",
        ExpressionAttributeValues={":arn": {"S": job_arn}},
        Limit=1)
    items = response.get("Items") or []
    return from_item(items[0]) if items else None


def query_pending(model_id: str, limit: int = 100) -> list:
    """
    PENDING rows for one model, oldest first — the drain's work list.

    FIFO by `created_at`, **not** by sort key: the sort key is `workflow_id`, a uuid5,
    so key order is arbitrary. Draining in key order would admit callers out of turn,
    invisible in aggregate, and starve whichever workflow_id happens to hash low.
    """
    rows = []
    kwargs = {
        "TableName": table_name(),
        "KeyConditionExpression": "pk = :pk",
        "FilterExpression": "#s = :pending",
        "ExpressionAttributeNames": {"#s": "status"},
        "ExpressionAttributeValues": {":pk": {"S": batch_pk(model_id)},
                                      ":pending": {"S": STATUS_PENDING}},
    }
    while True:
        response = _ddb().query(**kwargs)
        rows.extend(from_item(item) for item in response.get("Items") or [])
        token = response.get("LastEvaluatedKey")
        if not token or len(rows) >= limit:
            break
        kwargs["ExclusiveStartKey"] = token
    rows.sort(key=lambda row: row.get("created_at") or 0)
    return rows[:limit]


def models_with_pending() -> list:
    """
    Every model id that has at least one PENDING row — the scheduled sweep's work
    list, since a sweep arrives with no stream image and therefore no model (§6.5).

    A **Scan**, deliberately, rather than a second GSI on `status`. The table holds one
    row per in-flight or queued batch job and rows are deleted on notification, so it
    is bounded by concurrent batch jobs — tens, not millions. A status GSI would be a
    second index to keep, and its hot `PENDING` partition is exactly the shape DynamoDB
    warns about.
    """
    models, kwargs = set(), {
        "TableName": table_name(),
        "FilterExpression": "#s = :pending",
        "ExpressionAttributeNames": {"#s": "status"},
        "ExpressionAttributeValues": {":pending": {"S": STATUS_PENDING}},
        "ProjectionExpression": "pk",
    }
    while True:
        response = _ddb().scan(**kwargs)
        for item in response.get("Items") or []:
            pk = (item.get("pk") or {}).get("S") or ""
            if pk.startswith(BATCH_PK_PREFIX):
                models.add(model_id_from_pk(pk))
        token = response.get("LastEvaluatedKey")
        if not token:
            break
        kwargs["ExclusiveStartKey"] = token
    return sorted(models)


# ── Stream helpers ────────────────────────────────────────────────────────────

def stream_row(record: dict, image: str = "NewImage") -> dict:
    """
    One stream record's image as a plain dict. `image="OldImage"` for a REMOVE.

    Streams carry the low-level attribute shape, so this is `from_item` with the
    lookup — kept here so no handler has to know that.
    """
    return from_item(((record.get("dynamodb") or {}).get(image)) or {})
