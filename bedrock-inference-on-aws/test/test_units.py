"""
Unit tests — bedrock-inference-on-aws. Mocked S3, DynamoDB, Bedrock and Step
Functions only; no AWS calls, no moto, no third-party test dependency.
Run: python3 -m unittest discover -s test -p 'test_*.py'

README.md §12.1 is the checklist. The through-line of every test here is that the
failures this workflow can produce are SILENT — they succeed, and the damage is on
the bill or in a result attributed to the wrong inputs:

- a `workflow_id` that is not stable re-pays for work already done, and the
  execution SUCCEEDS both times;
- a checkpoint read as absent re-inferences every completed record, and the more
  loaded S3 is the more it costs — a broad `except` around the progress GET is the
  one-line version of that bug;
- an `ErrorEquals` naming a botocore error code instead of an exception CLASS name
  is dead retry configuration that reads, in the console, as working back-pressure;
- a geo (`"us"`) reaching `boto3.client(region_name=...)` fails as a DNS error that
  reads as a Bedrock outage;
- `states:SendTask*` on a function that also calls Bedrock concentrates two
  capabilities that must not be co-resident, and nothing at runtime says so.

None of those raise anywhere. This file is where they are visible.

Why hand-written doubles rather than moto: the load-bearing behaviours are the
`job_arn-index` GSI lookup (the completion event carries an arn and nothing else),
`mark_terminal`'s compound ConditionExpression (the redelivery guard), the Scan
behind `models_with_pending()` (both drain paths), and — since the semaphore moved
in-stack (README.md §M5.1) — `TransactWriteItems`' all-or-nothing semantics and
the ORDER and CODES of its `CancellationReasons` array, which is the sole thing
telling "saturated" apart from "re-entrant" in `try_acquire`. A fake that models
exactly those makes them testable and the suite hermetic. Unmodelled expressions
raise rather than defaulting to True — a silently ignored guard would make a broken
cap or a broken idempotency check look tested.

A cap bug is SILENT in production by construction: the workflow succeeds, Bedrock
throttles, and nothing anywhere says "the semaphore admitted four holders into three
slots".
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import re
import sys
import unittest
from unittest import mock

# Set before the first import. `input_schema` binds PROXY_BUCKET / AWS_REGION /
# TOOL_VERSION at import time, both table-name helpers read the env on every call,
# and a client built from an unset region would make the suite pass or fail
# according to the developer's shell.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("SERVICE_TOKEN_TABLE_NAME", "test-batch-infer-tokens")
os.environ.setdefault("SYNC_INFER_QUOTAS_TABLE_NAME", "test-sync-infer-quotas")
os.environ.setdefault("BATCH_SERVICE_ROLE_ARN",
                      "arn:aws:iam::123456789012:role/test-batch")
os.environ.setdefault("TOOL_VERSION", "1.0.0-test")
os.environ.setdefault("TARGET_NAME", "bedrock-inference")
# gateway_tool does `os.environ["INFERENCE_STATE_MACHINE_ARN"]` at module scope —
# a hard KeyError at import. Set even though the tests below read its source
# rather than importing it, so adding an import later does not fail mysteriously.
os.environ.setdefault(
    "INFERENCE_STATE_MACHINE_ARN",
    "arn:aws:states:us-east-1:123456789012:stateMachine:test-inference")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from botocore.exceptions import ClientError  # noqa: E402

import lambda_function as router  # noqa: E402
from lib import batch_client, identity, response as response_lib  # noqa: E402
from lib import bedrock as bedrock_lib  # noqa: E402
from lib import input_schema, s3_utils, semaphore_client, token_table  # noqa: E402
from lib.custom_exceptions import (  # noqa: E402
    BatchRecordsIncompleteException,
    BedrockBatchQuotaExceeded,
    BedrockInferenceFailed,
    LambdaResourceTimeoutException,
    ValidationException,
)
from states.acquire_lease import lambda_function as acquire_lease  # noqa: E402
from states.batch_status import lambda_function as batch_status  # noqa: E402
from states.collect_batch import lambda_function as collect_batch  # noqa: E402
from states.drain import lambda_function as drain  # noqa: E402
from states.handle_execution_terminal import (  # noqa: E402
    lambda_function as handle_execution_terminal,
)
from states.handle_semaphore_release import (  # noqa: E402
    lambda_function as handle_semaphore_release,
)
from states.notify import lambda_function as notify  # noqa: E402
from states.release_lease import lambda_function as release_lease  # noqa: E402
from states.run_inference import lambda_function as run_inference  # noqa: E402
from states.submit_batch import lambda_function as submit_batch  # noqa: E402
from states.summarize import lambda_function as summarize  # noqa: E402
from states.usage_tracking import lambda_function as usage_tracking  # noqa: E402
from states.validate_plan import lambda_function as validate_plan  # noqa: E402

# Short alias for the semaphore tests ported from service-quotas-on-aws
# (README.md §M5.1), where every body reads `sem.`.
sem = semaphore_client

PROXY = os.environ["PROXY_BUCKET"]
PREFIX = "micro-workflow/bedrock-inference/test"
REQUEST_KEY = f"{PREFIX}/input_params.json"
RECORDS_KEY = f"{PREFIX}/request.jsonl"
REQUEST_S3URI = f"s3://{PROXY}/{REQUEST_KEY}"

# Mirrors deployment/media2cloud-core-stack.yaml's CreateModelList entries exactly —
# `modelPricing.*PerTokenCost`, `supportedRegions` as a comma-joined STRING, and
# `batchJob: {canSupport, region}` where region is a GEO. A fixture with a different
# shape would test a registry this workflow never reads.
REGISTRY = [
    {
        "modelName": "Claude Sonnet 5",
        "modelId": "us.anthropic.claude-sonnet-5-v1:0",
        "modelVersion": "",
        "modelType": ["vision", "text"],
        "modelPricing": {
            "inputPerTokenCost": 0.000003,
            "outputPerTokenCost": 0.000015,
            "cacheReadPerTokenCost": 0.0000003,
        },
        "useInferenceProfile": True,
        "supportedRegions": "us-west-2,us-east-1,us-east-2",
        # Deliberately NOT the deployment region and NOT the runtime `us.` prefix:
        # a model invoked through `us.` may only batch through `global.`.
        "batchJob": {"canSupport": True, "region": "global"},
        # The request contract lives ONLY here since the family tuples were
        # retired (2026-10-15) — a capability-less generative entry is refused
        # by the seeder, so the fixture carries what a real entry carries.
        "capabilities": {"temperature": False, "thinking": "required",
                         "thinking_default": "disabled", "effort": True,
                         "effort_default": "low"},
    },
    {
        "modelName": "Amazon Nova Pro",
        "modelId": "amazon.nova-pro-v1:0",
        "modelVersion": "",
        "modelType": ["vision", "text"],
        "modelPricing": {
            "inputPerTokenCost": 0.0000008,
            "outputPerTokenCost": 0.0000032,
        },
        "useInferenceProfile": False,
        "supportedRegions": "us-east-1,us-west-2",
        "batchJob": {"canSupport": True, "region": "us"},
        "capabilities": {"temperature": True, "thinking": "absent",
                         "effort": False},
    },
    {
        "modelName": "Unpriced Test Model",
        "modelId": "vendor.unpriced-v1:0",
        "modelPricing": {},
        "supportedRegions": "us-east-1",
        "batchJob": {"canSupport": False},
        "capabilities": {"temperature": True, "thinking": "absent",
                         "effort": False},
    },
]

OPUS5 = "us.anthropic.claude-opus-5-v1:0"
FABLE5 = "us.anthropic.claude-fable-5-v1:0"

# A separate small registry rather than adding these to REGISTRY: every OTHER test
# in this file resolves `[PRIMARY, FALLBACK]` by index, and inserting a fourth entry
# would silently change which model `chain[1]` means everywhere else.
# Both carry their REAL contracts (tuples retired 2026-10-15): without the
# capabilities block these specs would read thinking-absent/temperature-on —
# the opposite of production — and the prefill tests would shape requests
# these families never see.
_OPUS_FABLE_CAPS = {"temperature": False, "thinking": "required",
                    "thinking_default": "adaptive", "effort": True,
                    "effort_default": "low"}
PREFILL_REGISTRY = [
    {"modelName": "Claude Opus 5", "modelId": OPUS5,
     "modelPricing": {"inputPerTokenCost": 0.000005, "outputPerTokenCost": 0.000025},
     "supportedRegions": "us-east-1", "batchJob": {"canSupport": True, "region": "us"},
     "capabilities": dict(_OPUS_FABLE_CAPS)},
    {"modelName": "Claude Fable 5", "modelId": FABLE5,
     "modelPricing": {"inputPerTokenCost": 0.00001, "outputPerTokenCost": 0.00005},
     "supportedRegions": "us-east-1", "batchJob": {"canSupport": True, "region": "us"},
     "capabilities": dict(_OPUS_FABLE_CAPS)},
]

PRIMARY = "us.anthropic.claude-sonnet-5-v1:0"
FALLBACK = "amazon.nova-pro-v1:0"
BATCH_MODEL = "global.anthropic.claude-sonnet-5-v1:0"

JOB_ID = "abc123def456"
JOB_ARN = f"arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/{JOB_ID}"


def _read(*parts) -> str:
    with open(os.path.join(*parts)) as handle:
        return handle.read()


def _code(*parts) -> str:
    """
    Source with docstrings and comments removed, for the capability tests below.

    Necessary because these modules' docstrings DISCUSS the grants they must not hold
    — batch_status.py's says "**No `SendTask*` here.**" — so a substring test against
    the raw file asserts against the prose instead of the code: it fails when the
    comment is accurate and passes when the comment is deleted, which is backwards.
    Comments go too, for the same reason. Note that `ast.unparse` normalizes string
    quoting, so markers matched against this must be quote-agnostic.
    """
    import ast

    tree = ast.parse(_read(*parts))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _as_list(value):
    """
    An IAM `Action` or `Resource` is legally either a string or a list, and the
    template uses both forms. Normalising is what lets a grant assertion survive
    someone collapsing a one-element list into a bare string — a purely stylistic
    edit that would otherwise turn an `in` check into a substring match on the
    string's characters.
    """
    return value if isinstance(value, list) else [value]


def _client_error(code, op="Unknown", status=400, reasons=None):
    """
    `reasons` attaches a `CancellationReasons` array, which is what a real
    `TransactionCanceledException` carries and the only thing `try_acquire` can read
    to tell "saturated" from "re-entrant".
    """
    err = ClientError(
        {"Error": {"Code": code, "Message": f"{code} from the test double"},
         "ResponseMetadata": {"HTTPStatusCode": status}},
        op)
    if reasons is not None:
        err.response["CancellationReasons"] = reasons
    return err


# ─────────────────────────────────────────────────────────────────────────────
# S3 double
# ─────────────────────────────────────────────────────────────────────────────


# Silence console log HANDLERS for the whole suite: the failure-path
# tests exercise refusal/retry/fallback branches whose designed
# behavior includes ERROR/WARNING logs, and those lines made build
# logs read as if the build were failing. Handlers only — record flow
# is untouched, so assertLogs() still sees everything.
import logging as _logging


def _silence_console_logging():
    _logging.getLogger().handlers = [_logging.NullHandler()]
    for _name in list(_logging.root.manager.loggerDict):
        _lg = _logging.getLogger(_name)
        if _lg.handlers:
            _lg.handlers = [_logging.NullHandler()]


_silence_console_logging()


class _Body:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def iter_lines(self, chunk_size=1024):
        # the streaming read path; mirror botocore's StreamingBody
        # surface (accepts chunk_size; irrelevant to the fake) — which
        # YIELDS blank lines (iter_jsonl's file-line numbering depends on
        # seeing them) but not a trailing empty after a final newline
        lines = self._payload.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        yield from lines


class _FakePaginator:
    def __init__(self, objects, page_size):
        self._objects = objects
        self._page_size = page_size

    def paginate(self, Bucket, Prefix, **kw):
        keys = sorted(key for (bucket, key) in self._objects
                      if bucket == Bucket and key.startswith(Prefix))
        for start in range(0, max(len(keys), 1), self._page_size):
            page = keys[start:start + self._page_size]
            yield {"Contents": [{"Key": key} for key in page]}


class _FakeS3:
    """
    Enough S3 to run the workflow, plus per-key error injection.

    `errors` is the whole point of the double: `get_json_if_exists` must return None
    for a genuine 404 and RE-RAISE everything else, and the only way to see the
    difference is to inject a 500 on one specific key.
    """

    def __init__(self, page_size=1000):
        self.objects: dict[tuple, bytes] = {}
        self.errors: dict[tuple, Exception] = {}
        self.page_size = page_size
        self.calls: list[tuple] = []

    # -- test helpers -------------------------------------------------------
    def put(self, bucket, key, payload):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.objects[(bucket, key)] = payload

    def put_lines(self, bucket, key, rows):
        self.put(bucket, key,
                 "".join(json.dumps(row) + "\n" for row in rows))

    def json_at(self, bucket, key):
        return json.loads(self.objects[(bucket, key)])

    def lines_at(self, bucket, key):
        text = self.objects[(bucket, key)].decode("utf-8")
        return [json.loads(line) for line in text.split("\n") if line.strip()]

    # -- API ----------------------------------------------------------------
    def get_object(self, Bucket, Key, **kw):
        self.calls.append(("get_object", Bucket, Key))
        if (Bucket, Key) in self.errors:
            raise self.errors[(Bucket, Key)]
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", "GetObject", status=404)
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType=None, **kw):
        self.calls.append(("put_object", Bucket, Key))
        self.objects[(Bucket, Key)] = Body

    # -- multipart trio: the streamed batch-input writer's surface ---------
    def create_multipart_upload(self, Bucket, Key, ContentType=None, **kw):
        self.calls.append(("create_multipart_upload", Bucket, Key))
        self._mpu = {"bucket": Bucket, "key": Key, "parts": {}}
        return {"UploadId": "mpu-1"}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body, **kw):
        self.calls.append(("upload_part", Bucket, Key, PartNumber))
        self._mpu["parts"][PartNumber] = bytes(Body)
        return {"ETag": f"etag-{PartNumber}"}

    def complete_multipart_upload(self, Bucket, Key, UploadId,
                                  MultipartUpload, **kw):
        self.calls.append(("complete_multipart_upload", Bucket, Key))
        body = b"".join(self._mpu["parts"][p["PartNumber"]]
                        for p in MultipartUpload["Parts"])
        self.objects[(Bucket, Key)] = body
        return {}

    def abort_multipart_upload(self, Bucket, Key, UploadId, **kw):
        self.calls.append(("abort_multipart_upload", Bucket, Key))
        self._mpu = None
        return {}

    def head_object(self, Bucket, Key, **kw):
        self.calls.append(("head_object", Bucket, Key))
        if (Bucket, Key) in self.errors:
            raise self.errors[(Bucket, Key)]
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", "HeadObject", status=404)
        return {}

    def get_paginator(self, name):
        assert name == "list_objects_v2", name
        return _FakePaginator(self.objects, self.page_size)


# ─────────────────────────────────────────────────────────────────────────────
# DynamoDB double — ONE class, TWO tables (`batch-infer-tokens` and
# `sync-infer-quotas`), because one zip now talks to both
# ─────────────────────────────────────────────────────────────────────────────

class _FakeDdb:
    """
    Models the calls `lib/token_table.py` AND `lib/semaphore_client.py` make.

    The two doubles merged when the semaphore moved in-stack (README.md §M5.1).
    Each contributed something the other had no need for, and every part of both is
    load-bearing:

    From the semaphore client's side:

    - **`transact_write_items`.** Conditions are evaluated for EVERY op before any
      is applied (all-or-nothing), and a failure raises
      `TransactionCanceledException` carrying one `CancellationReasons` entry per
      op, **in op order**. `try_acquire` reads `reasons[1]` positionally, and that
      read is the only thing telling "saturated" from "re-entrant".
    - **The five counter/holder condition expressions**, including real DDB's rule
      that comparing against a MISSING attribute is FALSE rather than an error.
    - **Conditional `delete_item`**, which is how `release` reports "already
      released" rather than raising.

    From the token table's side:

    - **`scan`**. `models_with_pending()` is a Scan and BOTH drain paths call it;
      without one the sweep is untestable.
    - **`query` honouring `IndexName` and `FilterExpression`.** `find_by_job_arn`
      keys off the `job_arn-index` GSI, and a fake that keys off `:pk` regardless
      would return a row for a lookup the real table answers with nothing.
    - **`#k{i}` resolution through `ExpressionAttributeNames`**, because
      `token_table._update` builds a GENERIC `SET #k0 = :v0, ...` expression rather
      than the literal ones the semaphore client uses.

    `_apply_update` therefore dispatches: the semaphore's literal forms first (one
    of them, `if_not_exists(holders, :zero) + :one`, contains a comma and would be
    mangled by generic clause splitting), then the generic form.

    Unmodelled expressions RAISE. A silently ignored guard would make a broken cap
    or a broken idempotency check look tested.
    """

    def __init__(self, page_size=1000):
        self.items: dict[tuple, dict] = {}
        self.page_size = page_size          # force query pagination in a test
        self.calls: list[str] = []
        self.tables: set = set()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _key(av: dict) -> tuple:
        return (av["pk"]["S"], av["sk"]["S"])

    @staticmethod
    def _check(cond, item, values=None) -> bool:
        if not cond:
            return True
        values = values or {}
        exists = item is not None
        # -- token table ----------------------------------------------------
        if cond == "attribute_exists(pk)":
            return exists
        if cond == ("attribute_exists(pk) AND #s <> :terminal_a "
                    "AND #s <> :terminal_b"):
            # The redelivery guard. Modelled explicitly rather than approximated:
            # it is the only thing preventing a second stream record and therefore a
            # second SendTaskSuccess for one job.
            if not exists:
                return False
            status = (item.get("status") or {}).get("S")
            return status not in (values[":terminal_a"]["S"],
                                  values[":terminal_b"]["S"])
        # -- semaphore ------------------------------------------------------
        if cond == "attribute_not_exists(pk)":
            return not exists
        if cond == "attribute_not_exists(holders) OR holders < max_holders":
            if not exists or "holders" not in item:
                return True
            if "max_holders" not in item:
                # Real DDB: comparing against a missing attribute is FALSE, not an
                # error and not True.
                return False
            return int(item["holders"]["N"]) < int(item["max_holders"]["N"])
        if cond == "holders > :zero":
            return exists and int(item.get("holders", {"N": "0"})["N"]) > 0
        if cond == "attribute_not_exists(max_holders) OR max_holders < :requested":
            if not exists or "max_holders" not in item:
                return True
            return int(item["max_holders"]["N"]) < int(values[":requested"]["N"])
        raise AssertionError(f"_FakeDdb: unmodelled condition {cond!r}")

    def _apply_update(self, key, expression, names, values):
        item = self.items.setdefault(
            key, {"pk": {"S": key[0]}, "sk": {"S": key[1]}})
        body = expression.strip()
        assert body.upper().startswith("SET "), body
        values = values or {}

        # The semaphore client's LITERAL expressions, matched by substring.
        literal = False
        if "if_not_exists(holders, :zero) + :one" in body:
            item["holders"] = {
                "N": str(int(item.get("holders", {"N": "0"})["N"]) + 1)}
            literal = True
        elif "holders = holders - :one" in body:
            item["holders"] = {"N": str(int(item["holders"]["N"]) - 1)}
            literal = True
        if "max_holders = :requested" in body:
            item["max_holders"] = values[":requested"]
            literal = True
        for placeholder in (":ttl", ":cttl"):
            if f"#ttl = {placeholder}" in body:
                item["ttl"] = values[placeholder]
                literal = True
        if literal:
            return

        # token_table._update's GENERIC `SET #k0 = :v0, ...`.
        for clause in body[4:].split(","):
            lhs, rhs = (part.strip() for part in clause.split("="))
            item[(names or {}).get(lhs, lhs)] = values[rhs]

    @staticmethod
    def _matches_filter(item, expression, names, values) -> bool:
        if not expression:
            return True
        if expression == "#s = :pending":
            attr = (names or {}).get("#s", "#s")
            return (item.get(attr) or {}).get("S") == values[":pending"]["S"]
        raise AssertionError(f"_FakeDdb: unmodelled filter {expression!r}")

    # -- API ----------------------------------------------------------------
    def transact_write_items(self, TransactItems, **kw):
        """
        All-or-nothing, with one `CancellationReasons` entry per op IN OP ORDER.
        `try_acquire` reads `reasons[1]` — the holder Put — alone, so an
        order-insensitive or reason-less fake would make the re-entrant-at-cap bug
        untestable.
        """
        self.calls.append("transact_write_items")
        reasons, staged, failed = [], [], False
        for op in TransactItems:
            (kind, body), = op.items()
            self.tables.add(body["TableName"])
            key = self._key(body["Item"] if kind == "Put" else body["Key"])
            ok = self._check(body.get("ConditionExpression"),
                             self.items.get(key),
                             body.get("ExpressionAttributeValues"))
            reasons.append({"Code": "None" if ok else "ConditionalCheckFailed"})
            failed = failed or not ok
            staged.append((kind, key, body))
        if failed:
            raise _client_error("TransactionCanceledException",
                                "TransactWriteItems", reasons=reasons)
        for kind, key, body in staged:
            if kind == "Put":
                self.items[key] = dict(body["Item"])
            elif kind == "Update":
                self._apply_update(key, body["UpdateExpression"],
                                   body.get("ExpressionAttributeNames"),
                                   body.get("ExpressionAttributeValues"))
            else:
                self.items.pop(key, None)
        return {}

    def put_item(self, TableName, Item, **kw):
        self.calls.append("put_item")
        self.tables.add(TableName)
        self.items[self._key(Item)] = dict(Item)
        return {}

    def update_item(self, TableName, Key, UpdateExpression=None,
                    ConditionExpression=None, ExpressionAttributeNames=None,
                    ExpressionAttributeValues=None, **kw):
        self.calls.append("update_item")
        self.tables.add(TableName)
        key = self._key(Key)
        if not self._check(ConditionExpression, self.items.get(key),
                           ExpressionAttributeValues):
            raise _client_error("ConditionalCheckFailedException", "UpdateItem")
        self._apply_update(key, UpdateExpression, ExpressionAttributeNames,
                           ExpressionAttributeValues or {})
        return {}

    def delete_item(self, TableName, Key, ConditionExpression=None, **kw):
        self.calls.append("delete_item")
        self.tables.add(TableName)
        key = self._key(Key)
        if not self._check(ConditionExpression, self.items.get(key)):
            raise _client_error("ConditionalCheckFailedException", "DeleteItem")
        self.items.pop(key, None)
        return {}

    def get_item(self, TableName, Key, **kw):
        self.calls.append("get_item")
        self.tables.add(TableName)
        item = self.items.get(self._key(Key))
        return {"Item": dict(item)} if item else {}

    def query(self, TableName, KeyConditionExpression,
              ExpressionAttributeValues, IndexName=None, Limit=None,
              FilterExpression=None, ExpressionAttributeNames=None,
              ExclusiveStartKey=None, **kw):
        self.calls.append("query")
        self.tables.add(TableName)

        if IndexName == token_table.JOB_ARN_INDEX:
            assert KeyConditionExpression == "job_arn = :arn", KeyConditionExpression
            wanted = ExpressionAttributeValues[":arn"]["S"]
            rows = [dict(item) for item in self.items.values()
                    if (item.get("job_arn") or {}).get("S") == wanted]
        else:
            assert IndexName is None, IndexName
            assert KeyConditionExpression == "pk = :pk", KeyConditionExpression
            pk = ExpressionAttributeValues[":pk"]["S"]
            rows = sorted((dict(item) for (ipk, _), item in self.items.items()
                           if ipk == pk),
                          key=lambda item: item["sk"]["S"])

        rows = [row for row in rows
                if self._matches_filter(row, FilterExpression,
                                        ExpressionAttributeNames,
                                        ExpressionAttributeValues)]
        if ExclusiveStartKey:
            after = ExclusiveStartKey["sk"]["S"]
            rows = [row for row in rows if row["sk"]["S"] > after]

        limit = min(Limit or self.page_size, self.page_size)
        page, rest = rows[:limit], rows[limit:]
        out = {"Items": page}
        if rest and page:
            out["LastEvaluatedKey"] = {"pk": page[-1]["pk"], "sk": page[-1]["sk"]}
        return out

    def scan(self, TableName, FilterExpression=None,
             ExpressionAttributeNames=None, ExpressionAttributeValues=None,
             ProjectionExpression=None, ExclusiveStartKey=None, **kw):
        self.calls.append("scan")
        self.tables.add(TableName)
        rows = sorted((dict(item) for item in self.items.values()),
                      key=lambda item: (item["pk"]["S"], item["sk"]["S"]))
        rows = [row for row in rows
                if self._matches_filter(row, FilterExpression,
                                        ExpressionAttributeNames,
                                        ExpressionAttributeValues or {})]
        if ExclusiveStartKey:
            after = (ExclusiveStartKey["pk"]["S"], ExclusiveStartKey["sk"]["S"])
            rows = [row for row in rows
                    if (row["pk"]["S"], row["sk"]["S"]) > after]
        page, rest = rows[:self.page_size], rows[self.page_size:]
        if ProjectionExpression:
            wanted = [name.strip() for name in ProjectionExpression.split(",")]
            page = [{key: row[key] for key in wanted if key in row}
                    for row in page]
        out = {"Items": page}
        if rest and page:
            out["LastEvaluatedKey"] = {"pk": rows[len(page) - 1]["pk"],
                                       "sk": rows[len(page) - 1]["sk"]}
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Bedrock and Step Functions doubles
# ─────────────────────────────────────────────────────────────────────────────

def _converse_response(text="ok", stop_reason="end_turn", tokens=(10, 5)):
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": tokens[0], "outputTokens": tokens[1],
                  "totalTokens": tokens[0] + tokens[1]},
        "metrics": {"latencyMs": 42},
    }


class _FakeRuntime:
    """
    `bedrock-runtime`. `script` entries are consumed in order; each may be a dict
    (returned), an Exception (raised), or a callable taking the params.
    """

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    def converse(self, **params):
        self.calls.append(params)
        if not self.script:
            return _converse_response()
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(params)
        return item


class _FakeControl:
    """`bedrock` control plane — CreateModelInvocationJob only."""

    def __init__(self, script=None, job_arn=JOB_ARN):
        self.script = list(script or [])
        self.job_arn = job_arn
        self.calls: list[dict] = []

    def create_model_invocation_job(self, **params):
        self.calls.append(params)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
        return {"jobArn": self.job_arn}


class _FakeSfn:
    def __init__(self, error=None):
        self.error = error
        self.successes: list[dict] = []
        self.failures: list[dict] = []

    def send_task_success(self, **kw):
        if self.error:
            raise self.error
        self.successes.append(kw)
        return {}

    def send_task_failure(self, **kw):
        if self.error:
            raise self.error
        self.failures.append(kw)
        return {}


class _Context:
    """
    The only clock the sync loop reads. `remaining` is a list consumed one entry per
    call, then held at its last value — so a test forces a checkpoint at an exact
    record index rather than by wall-clock timing.
    """

    def __init__(self, remaining=None):
        self.remaining = list(remaining or [900_000])
        self.reads = 0

    def get_remaining_time_in_millis(self):
        self.reads += 1
        if len(self.remaining) > 1:
            return self.remaining.pop(0)
        return self.remaining[0]


def _records(count, prefix="chunk"):
    return [
        {"record_id": f"{prefix}_{index:03d}",
         "system": "You are a helpful assistant.",
         "messages": [{"role": "user",
                       "content": [{"text": f"summarize part {index}"}]}]}
        for index in range(count)
    ]


_UNSET = object()


def _request(model_chain=_UNSET, mode="sync", options=_UNSET,
             output_prefix=PREFIX):
    # `is _UNSET` rather than `or`: `model_chain=[]` and `options={}` are values a
    # test needs to pass THROUGH, and a truthiness default silently replaces them
    # with the valid default — so the reject-case tests would assert against a
    # well-formed request and fail for the wrong reason.
    #
    # `mode` here is TEST-FIXTURE shorthand translated to the surviving
    # `use_batch` boolean (§5.6 retired the request-level mode) — most tests
    # predate the retirement and say what they mean with the old vocabulary.
    request = {
        "model_chain": ([PRIMARY, FALLBACK] if model_chain is _UNSET
                        else model_chain),
        "options": {"max_tokens": 4096} if options is _UNSET else options,
        "output": {"bucket": PROXY, "prefix": output_prefix},
    }
    if mode == "sync":
        request["use_batch"] = False
    elif mode == "batch":
        request["use_batch"] = True
    return request


def _stream_record(event_name, new=None, old=None):
    record = {"eventName": event_name, "eventSource": "aws:dynamodb",
              "dynamodb": {}}
    if new is not None:
        record["dynamodb"]["NewImage"] = token_table.to_item(new)
    if old is not None:
        record["dynamodb"]["OldImage"] = token_table.to_item(old)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Base case
# ─────────────────────────────────────────────────────────────────────────────

class _SemaphoreCase(unittest.TestCase):
    """
    Fresh `sync-infer-quotas` table per test; the client's lazy DDB global is
    replaced directly.

    A SEPARATE instance from `_WorkflowCase`'s token-table double, because the two
    tables are separate resources now (README.md §M decision 2) and one shared
    instance would let a semaphore key collide with a batch row — the exact failure
    the per-table split exists to make impossible.
    """

    def setUp(self):
        self.quota = _FakeDdb()
        self._saved_quota = semaphore_client._DDB
        semaphore_client._DDB = self.quota

    def tearDown(self):
        semaphore_client._DDB = self._saved_quota

    # convenience
    def counter(self, service="bedrock", resource="sonnet"):
        return self.quota.items.get(
            (semaphore_client.semaphore_pk(service, resource), "counter"))

    def holder(self, execution_arn, service="bedrock", resource="sonnet"):
        return self.quota.items.get(
            (semaphore_client.semaphore_pk(service, resource),
             semaphore_client.holder_sk(execution_arn)))


class _WorkflowCase(_SemaphoreCase):
    """Fresh S3 and both tables per test; module globals are replaced directly."""

    def setUp(self):
        super().setUp()
        self.s3 = _FakeS3()
        self.ddb = _FakeDdb()
        self._saved_s3 = s3_utils.s3
        self._saved_ddb = token_table._DDB
        s3_utils.s3 = self.s3
        token_table._DDB = self.ddb

        self.s3.put(PROXY, input_schema.MODEL_REGISTRY_KEY, REGISTRY)

    def tearDown(self):
        s3_utils.s3 = self._saved_s3
        token_table._DDB = self._saved_ddb  # nosemgrep: generic.secrets.gitleaks.generic-api-key
        super().tearDown()

    # -- helpers ------------------------------------------------------------
    def write_request(self, request=None, records=None):
        self.s3.put(PROXY, REQUEST_KEY, request or _request())
        self.s3.put_lines(PROXY, RECORDS_KEY, records or _records(2))

    def plan(self, request=None, records=None, execution_arn=""):
        """
        The real `Validate & plan` output, not a hand-built dict. Every downstream
        test therefore consumes the contract the state actually emits — a plan key
        renamed in one place breaks here rather than in production.
        """
        self.write_request(request, records)
        event = {"operation": "StateValidateAndPlan", "uuid": "u1",
                 "input": {"request_s3uri": REQUEST_S3URI}, "data": {}}
        if execution_arn:
            event["execution_arn"] = execution_arn
        return validate_plan.lambda_handler(event, None)

    def run_sync(self, plan, runtime, context=None):
        with mock.patch.object(run_inference, "bedrock_runtime",
                              return_value=runtime), \
             mock.patch.object(run_inference.semaphore_client, "refresh_holder",
                               return_value=True):
            return run_inference.lambda_handler(plan, context or _Context())


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 item 1 — workflow_id stability
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowId(unittest.TestCase):
    """
    A `workflow_id` that moves when nothing semantic moved re-pays for work already
    done, and BOTH executions succeed. There is no failure to notice.
    """

    def _id(self, request, records, tool_version="1.0.0-test"):
        parsed = input_schema.parse_records(records)
        return identity.compute_workflow_id(
            request, records_digest=identity.digest_records(parsed),
            tool_version=tool_version)

    def test_stable_across_output_and_mode(self):
        records = _records(3)
        base = _request()
        moved = _request(output_prefix="somewhere/else")
        moved["mode"] = "batch"
        moved["use_batch"] = True
        moved["tool_version"] = "ignored"
        self.assertEqual(self._id(base, records), self._id(moved, records))
        self.assertEqual(set(identity.EXCLUDED_KEYS),
                         {"output", "mode", "use_batch", "tool_version"})

    def test_stable_across_key_insertion_order(self):
        records = _records(2)
        forward = {"model_chain": [PRIMARY], "mode": "sync",
                   "options": {"temperature": 0.0},
                   "output": {"bucket": PROXY, "prefix": PREFIX}}
        reversed_ = {"output": {"prefix": PREFIX, "bucket": PROXY},
                     "options": {"temperature": 0.0},
                     "mode": "sync", "model_chain": [PRIMARY]}
        self.assertEqual(self._id(forward, records), self._id(reversed_, records))

    def test_one_prompt_byte_changes_it(self):
        records = _records(3)
        edited = json.loads(json.dumps(records))
        edited[1]["messages"][0]["content"][0]["text"] += "."
        self.assertNotEqual(self._id(_request(), records),
                            self._id(_request(), edited))

    def test_record_order_changes_it(self):
        """
        `response.jsonl` is in REQUEST order (§4.2), so a reordered request is
        different work — sorting the digest would hand back a result whose lines
        pair with the wrong inputs.
        """
        records = _records(3)
        self.assertNotEqual(self._id(_request(), records),
                            self._id(_request(), list(reversed(records))))

    def test_model_chain_changes_it(self):
        records = _records(2)
        self.assertNotEqual(
            self._id(_request(model_chain=[PRIMARY]), records),
            self._id(_request(model_chain=[PRIMARY, FALLBACK]), records))

    def test_tool_version_changes_it(self):
        records = _records(2)
        self.assertNotEqual(
            self._id(_request(), records, tool_version="1.0.0-test"),
            self._id(_request(), records, tool_version="1.0.1-test"))

    def test_output_prefix_layout(self):
        self.assertEqual(identity.output_prefix("bedrock-inference", "abc"),
                         "micro-workflow/bedrock-inference/abc")


class TestValidatePlan(_WorkflowCase):
    def test_plan_recomputes_the_same_workflow_id(self):
        plan = self.plan()
        parsed = input_schema.parse_records(
            self.s3.lines_at(PROXY, RECORDS_KEY))
        expected = identity.compute_workflow_id(
            self.s3.json_at(PROXY, REQUEST_KEY),
            records_digest=identity.digest_records(parsed),
            tool_version=input_schema.TOOL_VERSION)
        self.assertEqual(plan["data"]["workflow_id"], expected)

    def test_streamed_plan_still_rejects_empty_records(self):
        # the guards moved from parse_records into the handler's one
        # streamed pass (2026-09-06: the state held 2651 MB for a 668 MB
        # request as the last materialiser) — pin them THROUGH the handler.
        # write_request treats [] as "use defaults", so seed the empty
        # object directly.
        self.s3.put(PROXY, REQUEST_KEY, _request())
        self.s3.put(PROXY, RECORDS_KEY, "")
        event = {"operation": "StateValidateAndPlan", "uuid": "u1",
                 "input": {"request_s3uri": REQUEST_S3URI}, "data": {}}
        with self.assertRaises(ValidationException) as ctx:
            validate_plan.lambda_handler(event, None)
        self.assertIn("empty", str(ctx.exception))

    def test_streamed_plan_still_rejects_duplicate_record_ids(self):
        rows = [{"record_id": "r0",
                 "messages": [{"role": "user", "content": [{"text": "a"}]}]},
                {"record_id": "r0",
                 "messages": [{"role": "user", "content": [{"text": "b"}]}]}]
        with self.assertRaises(ValidationException) as ctx:
            self.plan(records=rows)
        self.assertIn("duplicate", str(ctx.exception))

    def test_streamed_plan_still_enforces_the_record_ceiling(self):
        rows = [{"record_id": f"r{i}",
                 "messages": [{"role": "user", "content": [{"text": "x"}]}]}
                for i in range(input_schema.MAX_RECORDS_CEILING + 1)]
        with self.assertRaises(ValidationException) as ctx:
            self.plan(records=rows)
        self.assertIn("ceiling", str(ctx.exception))

    def test_execution_name_mismatch_warns_and_the_name_wins(self):
        """
        A mismatch warns but never raises — a human console start and a tokened
        gateway submit both legitimately mismatch, and refusing would make the
        workflow undebuggable by hand. The NAME becomes data.workflow_id, and that
        is load-bearing: it is the batch job name and the token-table sort key, so
        a tokened run keyed on the recomputed parameter hash instead would collide
        with the token-less run of the same parameters and overwrite its parked
        task token.
        """
        arn = "arn:aws:states:us-east-1:123456789012:execution:sm:hand-named"
        with self.assertLogs(validate_plan.logger, level="WARNING") as logs:
            plan = self.plan(execution_arn=arn)
        self.assertTrue(any("changed after" in line for line in logs.output))
        self.assertEqual(plan["data"]["workflow_id"], "hand-named")

    def test_without_an_execution_arn_the_parameter_hash_is_the_id(self):
        """A gateway-less start (no execution_arn on the event) falls back to the
        recomputed parameter hash — the value the execution name would have
        carried on a token-less gateway submit."""
        plan = self.plan()
        parsed = input_schema.parse_records(
            self.s3.lines_at(PROXY, RECORDS_KEY))
        self.assertEqual(
            plan["data"]["workflow_id"],
            identity.compute_workflow_id(
                self.s3.json_at(PROXY, REQUEST_KEY),
                records_digest=identity.digest_records(parsed),
                tool_version=input_schema.TOOL_VERSION))

    def test_batch_geo_is_reported_separately_from_the_aws_region(self):
        """
        The registry's `batchJob.region` is a GEO — "us", "eu", "apac", "global".
        Emitting it as `batch_region` sends it to `submit_batch`, which passes it to
        `boto3.client(region_name=...)`: a region that does not exist, failing as a
        DNS error that reads as a Bedrock outage. Worse, `submit_batch` PERSISTS it
        into the token row's `model_params["region"]` and every drain replays it, so
        the poison outlives the execution.
        """
        plan = self.plan()
        batch = plan["data"]["batch"]
        self.assertEqual(batch["batch_geo"], "global")
        self.assertEqual(batch["batch_region"], input_schema.AWS_REGION)
        self.assertRegex(batch["batch_region"], r"^[a-z]{2}(-gov)?-[a-z]+-\d$")
        self.assertNotIn(batch["batch_region"], bedrock_lib._BATCH_GEO_PREFIXES)
        # The geo's ONE legitimate use: it prefixes the batch model id.
        self.assertEqual(batch["batch_model_id"], BATCH_MODEL)

    def test_only_validate_plan_mentions_batch_geo(self):
        """
        Structural, because the tempting "fix" when a batch model id looks wrong is
        to feed `batch_geo` to a client. Nothing downstream may read it.
        """
        for name in sorted(os.listdir(os.path.join(ROOT, "states"))):
            path = os.path.join(ROOT, "states", name, "lambda_function.py")
            if not os.path.isfile(path) or name == "validate_plan":
                continue
            self.assertNotIn("batch_geo", _read(path), name)

    def test_quota_resource_is_the_primary_model_only(self):
        """A chain's fallbacks are per-record escalations inside ONE held slot, so a
        chain must not multiply the cap."""
        plan = self.plan()
        self.assertEqual(plan["data"]["quota"]["resource"], PRIMARY)
        self.assertEqual(plan["data"]["quota"]["service"],
                         input_schema.QUOTA_SERVICE)

    def test_unregistered_model_raises_before_a_token_is_spent(self):
        with self.assertRaises(bedrock_lib.ModelNotRegistered):
            self.plan(request=_request(model_chain=["vendor.not-deployed-v1:0"]))

    def test_use_batch_on_a_non_batch_model_downgrades_with_the_reason(self):
        """§5.6 retired the hard error: the caller is TOLD (mode_reason and the
        summary), not refused — the old rationale ("do not bill full price
        without telling") is preserved by the telling."""
        plan = self.plan(request=_request(model_chain=["vendor.unpriced-v1:0"],
                                          mode="batch"))
        self.assertEqual(plan["data"]["mode"], "sync")
        self.assertIn("downgraded", plan["data"]["mode_reason"])

    def test_auto_mode_is_the_dollar_guard_not_a_record_count(self):
        """3 long chunks deserve batch; 100 greetings deserve sync-in-seconds. A
        count rule gets both wrong (§5.6), so the guard is the estimated saving
        itself. PRIMARY prices at 3e-6/input token: the guard needs
        ~BATCH_MIN_SAVINGS/(3e-6*0.5) ≈ 167k input tokens, whatever the count."""
        plan = self.plan(request=_request(mode="auto"), records=_records(100))
        self.assertEqual(plan["data"]["mode"], "sync")
        self.assertIn("under the", plan["data"]["mode_reason"])

        big = [{"record_id": "big_0",
                "messages": [{"role": "user",
                              "content": [{"text": "x" * 700_000}]}]}]
        plan = self.plan(request=_request(mode="auto"), records=big)
        self.assertEqual(plan["data"]["mode"], "batch")
        self.assertIn("clears the", plan["data"]["mode_reason"])

    def test_an_unpriced_model_falls_back_to_token_volume(self):
        """A missing price must not read as "no saving" — the None-not-zero
        discipline, applied to the guard."""
        big = [{"record_id": "big_0",
                "messages": [{"role": "user", "content": [{"text": "x" * (
                    4 * input_schema.UNPRICED_BATCH_MIN_TOKENS)}]}]}]
        # vendor.unpriced has no pricing AND cannot batch -> sync regardless;
        # the fallback is observable through choose_mode directly instead.
        mode, reason = input_schema.choose_mode(
            None, True, estimated_saving=None,
            estimated_input_tokens=input_schema.UNPRICED_BATCH_MIN_TOKENS)
        self.assertEqual(mode, "batch")
        self.assertIn("fallback", reason)


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 items 2 and 3 — checkpoint / re-enter, and checkpoint read failure
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointAndReentry(_WorkflowCase):
    TOTAL = 40
    CUT = 17

    def test_checkpoints_and_re_enters_paying_for_each_record_once(self):
        """
        The re-entry loop's whole purpose. 40 serial Converse calls do not fit in one
        15-minute invocation, so the loop yields; if the checkpoint is not honoured
        on re-entry the workflow still SUCCEEDS and simply costs double.
        """
        plan = self.plan(records=_records(self.TOTAL))
        runtime = _FakeRuntime()
        # Large for the first CUT deadline checks, then below the buffer forever.
        context = _Context([900_000] * self.CUT
                           + [input_schema.REMAINING_TIME_BUFFER_MS - 1])

        with self.assertRaises(LambdaResourceTimeoutException) as raised:
            self.run_sync(plan, runtime, context)
        self.assertIn(f"record {self.CUT} of {self.TOTAL}", str(raised.exception))
        self.assertEqual(len(runtime.calls), self.CUT)

        progress = self.s3.json_at(PROXY, plan["data"]["keys"]["progress"])
        self.assertEqual(len(progress["done"]), self.CUT)
        self.assertEqual(progress["records_total"], self.TOTAL)

        second = _FakeRuntime()
        result = self.run_sync(plan, second, _Context([900_000]))
        self.assertEqual(len(second.calls), self.TOTAL - self.CUT)
        self.assertEqual(result["data"]["records_resumed"], self.CUT)
        self.assertEqual(result["data"]["records_total"], self.TOTAL)
        self.assertEqual(len(runtime.calls) + len(second.calls), self.TOTAL)

        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual([line["record_id"] for line in lines],
                         [f"chunk_{index:03d}" for index in range(self.TOTAL)])

    def test_response_lines_are_in_request_order_not_completion_order(self):
        """
        The checkpoint is a dict keyed by `record_id`; the order has to be re-imposed
        from the record list. A caller zipping its own inputs against the response
        positionally would otherwise silently mis-pair them after a re-entry.
        """
        records = _records(4)
        plan = self.plan(records=records)
        progress = {"done": {records[2]["record_id"]: response_lib.success_line(
            record_id=records[2]["record_id"], model_id=PRIMARY,
            output_text="early", stop_reason="end_turn", usage={},
            estimated_cost=0.0)}, "records_total": 4}
        self.s3.put(PROXY, plan["data"]["keys"]["progress"], progress)

        self.run_sync(plan, _FakeRuntime())
        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual([line["record_id"] for line in lines],
                         [record["record_id"] for record in records])

    def test_a_failed_progress_read_raises_rather_than_reading_as_absent(self):
        """
        §12.3's headline: a broad `except` around the progress GET reads a throttled
        or denied read as "no checkpoint yet" and re-inferences every completed
        record — so the failure gets MORE expensive under load, which is exactly when
        it happens. A 500 must raise; only a 404 may return None.
        """
        plan = self.plan(records=_records(4))
        key = plan["data"]["keys"]["progress"]
        self.s3.errors[(PROXY, key)] = _client_error(
            "InternalError", "GetObject", status=500)

        runtime = _FakeRuntime()
        with self.assertRaises(ClientError):
            self.run_sync(plan, runtime)
        self.assertEqual(runtime.calls, [],
                         "a failed checkpoint read must not start inferring")

    def test_an_absent_progress_object_is_not_an_error(self):
        plan = self.plan(records=_records(2))
        result = self.run_sync(plan, _FakeRuntime())
        self.assertEqual(result["data"]["records_resumed"], 0)

    def test_lost_holder_is_logged_at_error_and_does_not_abort(self):
        """
        Best-effort by design: the caller is mid-span and cannot usefully abort, and
        the counter's TTL is the backstop. Logged at ERROR because a lost holder
        means the cap is being exceeded RIGHT NOW.
        """
        plan = self.plan(records=_records(1))
        plan["execution_arn"] = "arn:aws:states:us-east-1:1:execution:sm:x"
        with mock.patch.object(run_inference, "bedrock_runtime",
                              return_value=_FakeRuntime()), \
             mock.patch.object(run_inference.semaphore_client, "refresh_holder",
                               return_value=False), \
             self.assertLogs(run_inference.logger, level="ERROR") as logs:
            result = run_inference.lambda_handler(plan, _Context())
        self.assertTrue(any("HOLDER LOST" in line for line in logs.output))
        self.assertEqual(result["data"]["records_succeeded"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 item 4 — the chain walk
# ─────────────────────────────────────────────────────────────────────────────

class TestChainWalk(_WorkflowCase):
    def _chain(self, ids=None):
        return bedrock_lib.build_chain(REGISTRY, ids or [PRIMARY, FALLBACK])

    def test_retryable_failure_exhausts_the_first_model_then_advances(self):
        chain = self._chain()
        script = [_client_error("ThrottlingException", "Converse")] * 4 + [
            lambda params: _converse_response(text="from the fallback")]
        runtime = _FakeRuntime(script)
        result = bedrock_lib.converse_with_chain(
            runtime, chain, "sys", [{"role": "user", "content": [{"text": "hi"}]}],
            max_retries=3, sleep=lambda _: None)

        self.assertEqual(result.model_id, FALLBACK)
        self.assertEqual(len(runtime.calls), 5)      # 1 + 3 retries, then fallback
        self.assertEqual([call["modelId"] for call in runtime.calls[:4]],
                         [PRIMARY] * 4)
        self.assertEqual(result.rounds, 5)
        self.assertTrue(any("fell back from" in note for note in result.notes))

    def test_quota_failure_escalates_immediately_without_retrying(self):
        """Backoff cannot fix an exhausted quota; escalating at once is both faster
        and correct."""
        chain = self._chain()
        runtime = _FakeRuntime([_client_error("ServiceQuotaExceededException",
                                              "Converse")])
        result = bedrock_lib.converse_with_chain(
            runtime, chain, None, [{"role": "user", "content": [{"text": "hi"}]}],
            sleep=lambda _: None)
        self.assertEqual(result.model_id, FALLBACK)
        self.assertEqual(len(runtime.calls), 2)

    def test_validation_exception_never_advances_the_chain(self):
        """
        A malformed request is malformed for every model. Walking the chain on it
        multiplies the error, the latency and the input-token bill.
        """
        chain = self._chain()
        runtime = _FakeRuntime([_client_error("ValidationException", "Converse")])
        with self.assertRaises(bedrock_lib.BedrockError) as raised:
            bedrock_lib.converse_with_chain(
                runtime, chain, None,
                [{"role": "user", "content": [{"text": "hi"}]}],
                sleep=lambda _: None)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual([call["modelId"] for call in runtime.calls], [PRIMARY])
        self.assertIn("not retried on any other model", str(raised.exception))
        self.assertEqual(raised.exception.codes, ["ValidationException"])

    def test_a_refusal_advances_but_the_last_refusal_is_returned(self):
        chain = self._chain()
        runtime = _FakeRuntime([
            _converse_response(stop_reason="content_filtered"),
            _converse_response(stop_reason="guardrail_intervened"),
        ])
        result = bedrock_lib.converse_with_chain(
            runtime, chain, None, [{"role": "user", "content": [{"text": "x"}]}],
            sleep=lambda _: None)
        self.assertTrue(result.refused)
        self.assertEqual(result.model_id, FALLBACK)
        self.assertEqual(len(runtime.calls), 2)

    def test_a_poisoned_record_does_not_downgrade_its_neighbours(self):
        """
        §5.3. The chain walk is PER RECORD; one bad record must not fail the other
        39, and must not push the good ones onto the fallback model either.
        """
        plan = self.plan(records=_records(3))
        runtime = _FakeRuntime([
            _converse_response(text="first"),
            _client_error("ValidationException", "Converse"),
            _converse_response(text="third"),
        ])
        result = self.run_sync(plan, runtime)

        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual([line["status"] for line in lines],
                         ["SUCCESS", "FAILED", "SUCCESS"])
        self.assertEqual(result["data"]["records_succeeded"], 2)
        self.assertEqual([call["modelId"] for call in runtime.calls],
                         [PRIMARY] * 3)
        self.assertIn("ValidationException", lines[1]["reason"])

    def test_a_failed_record_still_reports_the_rounds_it_burned(self):
        """
        A record that failed after burning three throttled attempts cost real money.
        Dropping its rounds under-reports the bill for exactly the runs that went
        worst.
        """
        exc = bedrock_lib.BedrockError("boom", codes=["ThrottlingException"],
                                       rounds=5)
        line = response_lib.failure_line(
            record_id="r", reason="x", model_id=PRIMARY,
            rounds=getattr(exc, "rounds", 0))
        self.assertEqual(line["rounds"], 5)

    def test_same_family_chain_warns_rather_than_refusing(self):
        with self.assertLogs(bedrock_lib.log, level="WARNING") as logs:
            chain = bedrock_lib.build_chain(
                REGISTRY, [PRIMARY, "us.anthropic.claude-sonnet-5-v1:0"])
        self.assertEqual(len(chain), 1)     # exact duplicate dropped
        self.assertTrue(any("dropping the duplicate" in line
                            for line in logs.output))


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 item 5 — partial success
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialSuccess(_WorkflowCase):
    def _summarize(self, lines, mode="sync"):
        plan = self.plan(request=_request(mode=mode), records=_records(1))
        self.s3.put_lines(PROXY, plan["data"]["keys"]["response"], lines)
        return plan, summarize.lambda_handler(plan, None)

    @staticmethod
    def _lines(succeeded, failed):
        lines = [response_lib.success_line(
            record_id=f"ok_{i:03d}", model_id=PRIMARY, output_text="text",
            stop_reason="end_turn", usage={"inputTokens": 10, "outputTokens": 5},
            estimated_cost=0.0001) for i in range(succeeded)]
        lines += [response_lib.failure_line(
            record_id=f"bad_{i:03d}", reason=f"reason {i}", model_id=PRIMARY,
            estimated_cost=0.0) for i in range(failed)]
        return lines

    def test_thirty_nine_of_forty_is_a_successful_execution(self):
        """Partial success IS success (§7). One FAILED line and its reason in
        summary.failures[]; the caller gets its 39 results."""
        plan, result = self._summarize(self._lines(39, 1))
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])
        self.assertEqual(result["data"]["records_succeeded"], 39)
        self.assertEqual(result["data"]["records_failed"], 1)
        self.assertEqual(len(summary["failures"]), 1)
        self.assertEqual(summary["failures"][0]["record_id"], "bad_000")

    def test_every_record_failing_raises_and_names_up_to_five_reasons(self):
        plan = self.plan(records=_records(1))
        self.s3.put_lines(PROXY, plan["data"]["keys"]["response"],
                          self._lines(0, 40))
        with self.assertRaises(BedrockInferenceFailed) as raised:
            summarize.lambda_handler(plan, None)

        message = str(raised.exception)
        self.assertIn("every record failed (40 of 40)", message)
        self.assertEqual(message.count("reason "), response_lib.MAX_REPORTED_FAILURES)
        self.assertIn("(+35 more)", message)

        # Written FIRST, raised second — the per-record reasons survive the failure.
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])
        self.assertEqual(summary["records_succeeded"], 0)
        self.assertEqual(len(summary["failures"]), 40)

    def test_an_empty_record_set_cannot_reach_summarize(self):
        """`parse_records` refuses an empty request.jsonl, which is what keeps
        "every record failed" distinguishable from "no records"."""
        with self.assertRaises(ValidationException):
            input_schema.parse_records([])

    def test_unpriced_rounds_are_reported_beside_the_cost(self):
        lines = self._lines(2, 0)
        lines.append(response_lib.success_line(
            record_id="unpriced", model_id="vendor.unpriced-v1:0",
            output_text="t", stop_reason="end_turn", usage={"inputTokens": 9},
            estimated_cost=None))
        plan, _ = self._summarize(lines)
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])
        self.assertEqual(summary["bedrock_usage"]["rounds_unpriced"], 1)
        self.assertFalse(summary["bedrock_usage"]["cost_is_complete"])

    def test_quota_table_is_empty_on_the_batch_path(self):
        """Batch takes no lease, so no table governed the run; naming one would
        imply a cap that was never consulted."""
        plan, _ = self._summarize(self._lines(1, 0), mode="batch")
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])
        self.assertEqual(summary["quota_table"], "")

        plan, _ = self._summarize(self._lines(1, 0), mode="sync")
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])
        self.assertEqual(summary["quota_table"], semaphore_client.table_name())


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 items 6 and 7 — batch padding, and batch failure classification
# ─────────────────────────────────────────────────────────────────────────────

def _padded(rows, spec=None):
    """Test stand-in for the deleted pad_records wrapper: rows + the pads
    iter_pads yields, plus the pad count. iter_pads refuses a None spec
    whenever pads are emitted, so the default is a minimal contract-
    carrying spec (nova-shaped: temperature on, no thinking, no effort)."""
    if spec is None:
        spec = bedrock_lib.ModelSpec(
            {"model_id": "pad.fixture-v1:0",
             "model_type": ["text"],
             "capabilities": {"temperature": True, "thinking": "absent",
                              "effort": False}})
    pad_list = list(batch_client.iter_pads(len(rows), spec))
    return list(rows) + pad_list, len(pad_list)


class TestBatchPadding(unittest.TestCase):
    def test_forty_records_become_one_padded_job(self):
        """
        Padding is applied ONCE, after all real records. Per-record padding would
        multiply the floor by the record count — 40 records → 4000 lines.
        """
        rows = [{"recordId": f"r{i}"} for i in range(40)]
        padded, pads = _padded(rows)
        self.assertEqual(len(padded), input_schema.MIN_BATCH_RECORDS)
        self.assertEqual(pads, input_schema.MIN_BATCH_RECORDS - 40)
        self.assertEqual([row["recordId"] for row in padded[:40]],
                         [row["recordId"] for row in rows])
        self.assertTrue(all(bedrock_lib.is_padding(row["recordId"])
                            for row in padded[40:]))

    def test_at_or_over_the_floor_nothing_is_padded(self):
        rows = [{"recordId": f"r{i}"} for i in range(150)]
        padded, pads = _padded(rows)
        self.assertEqual((len(padded), pads), (150, 0))

        rows = [{"recordId": f"r{i}"} for i in range(input_schema.MIN_BATCH_RECORDS)]
        self.assertEqual(_padded(rows)[1], 0)

    def test_a_real_record_may_not_use_the_reserved_pad_prefix(self):
        """`read_job_output` drops pad_* records, so a real record named pad_000
        would vanish from the response with no error anywhere."""
        with self.assertRaises(ValidationException):
            input_schema.parse_records(
                [{"record_id": "pad_000",
                  "messages": [{"role": "user", "content": [{"text": "x"}]}]}])

    def test_padding_records_are_minimal_and_converse_shaped(self):
        """One dialect for every record since §5.4 resolved (2026-08-11): the job
        declares `modelInvocationType: "Converse"`, so a pad is a one-token
        Converse call — there is no per-family pad shape left to get wrong."""
        pad = bedrock_lib.padding_record(3, None)
        self.assertEqual(pad["recordId"], "pad_0003")
        self.assertEqual(pad["modelInput"]["inferenceConfig"]["maxTokens"], 1)
        self.assertNotIn("anthropic_version", pad["modelInput"])
        self.assertNotIn("max_tokens", pad["modelInput"])

    def test_pads_and_real_records_share_one_shape(self):
        """A job validates ALL-OR-NOTHING, so pads and real records must be in the
        same dialect — one unscorable pad fails all 100 lines including the real
        ones (measured on the 2026-08-09 job: 0/100, 99 of them option-free pads)."""
        real = bedrock_lib.to_batch_record(
            "r1", {"modelId": "m",
                   "messages": [{"role": "user",
                                 "content": [{"text": "hi"}]}],
                   "inferenceConfig": {"maxTokens": 8, "temperature": 0}})
        padded, pads = _padded([real])
        self.assertEqual(pads, input_schema.MIN_BATCH_RECORDS - 1)
        self.assertEqual(
            {frozenset(row["modelInput"]) for row in padded},
            {frozenset(real["modelInput"])})

    def test_create_job_declares_the_converse_invocation_type(self):
        """THE §5.4 resolution, pinned: without this parameter the API defaults to
        InvokeModel and judges Converse-shaped records malformed — 0/100 with
        `messages.0.content.0.type: Field required` and the job reporting
        Completed (job 731t09iortbs). Declared in create_job so submit_batch and
        the drain's replay of persisted model_params both get it."""
        client = mock.MagicMock()
        client.create_model_invocation_job.return_value = {"jobArn": "arn:j"}
        with mock.patch.object(batch_client, "bedrock_control",
                               return_value=client):
            batch_client.create_job(
                job_name="j", model_id="m", input_s3uri="s3://b/in.jsonl",
                output_s3uri="s3://b/out", role_arn="arn:role")
        _, kwargs = client.create_model_invocation_job.call_args
        self.assertEqual(kwargs["modelInvocationType"], "Converse")


# The version that first carries `modelInvocationType` on
# CreateModelInvocationJobRequest. Re-derivable: bisect the published botocore
# wheels reading
# `botocore/data/bedrock/2023-04-20/service-2.json.gz` →
# shapes.CreateModelInvocationJobRequest.members.
BOTOCORE_FLOOR = (1, 42, 59)
# Earliest published public.ecr.aws/lambda/python:3.13.* tag shipping a botocore
# at or above that floor is dated 2026.08.07 (it ships 1.42.97).
BASE_IMAGE_FLOOR_DATE = (2026, 8, 7)


class TestBotocoreIsNewEnoughForTheParametersWeSend(unittest.TestCase):
    """
    The gap the test above cannot see, and the reason this class exists.

    `test_create_job_declares_the_converse_invocation_type` asserts against a
    `MagicMock`, which accepts ANY kwarg. Real botocore does not: it validates the
    input against the bundled service model and raises
    `ParamValidationError: Unknown parameter in input` before the request leaves
    the process. So a parameter the code needs and the installed botocore has
    never heard of is a 100%-reproducible runtime failure that the entire offline
    suite passes.

    That happened on 2026-08-12. Moving the tier functions from the managed
    python3.13 runtime (where AWS keeps botocore moving) to a pinned container
    image (where the FROM tag freezes it, at 1.40.4) broke every batch submit
    while sync, the semaphore, cfn-lint, cfn_nag and 364 unit tests stayed green.
    It surfaced only because the burst harness had just been fixed to actually
    exercise the batch arm.
    """

    def test_the_batch_submit_parameters_validate_against_the_real_service_model(self):
        """The direct catch: botocore's OWN validator, against botocore's own
        bedrock model, on the exact kwargs create_job builds. No mock in the
        assertion path, so an unknown-parameter regression fails here rather than
        four minutes into a live batch run."""
        import botocore
        from botocore.session import get_session
        from botocore.validate import ParamValidator

        installed = tuple(int(part) for part in botocore.__version__.split(".")[:3])
        if installed < BOTOCORE_FLOOR:
            self.skipTest(
                f"installed botocore {botocore.__version__} is below the "
                f"{'.'.join(map(str, BOTOCORE_FLOOR))} floor requirements.txt "
                "declares, so its bedrock model cannot validate what the image's "
                "botocore validates — the floor itself is checked by "
                "test_the_base_image_is_new_enough_for_every_batch_parameter")

        client = mock.MagicMock()
        client.create_model_invocation_job.return_value = {"jobArn": "arn:j"}
        with mock.patch.object(batch_client, "bedrock_control",
                               return_value=client):
            batch_client.create_job(
                job_name="j", model_id="m", input_s3uri="s3://b/in.jsonl",
                output_s3uri="s3://b/out", role_arn="arn:role")
        _, kwargs = client.create_model_invocation_job.call_args

        # get_service_model reads the packaged JSON off disk; no network, no creds.
        operation = get_session().get_service_model("bedrock").operation_model(
            "CreateModelInvocationJob")
        report = ParamValidator().validate(dict(kwargs), operation.input_shape)
        self.assertFalse(report.has_errors(), report.generate_report())

    def test_the_base_image_is_new_enough_for_every_batch_parameter(self):
        """
        The hermetic half: requirements.txt's floor and the Dockerfile's pinned
        `FROM` are two numbers that must agree, and nothing else in the build
        checks them. requirements.txt installs nothing here — it is the contract
        the base tag has to satisfy — so a base bump that lowers botocore is
        invisible unless it is compared to the floor.
        """
        requirements = _read(ROOT, "requirements.txt")
        floors = dict(re.findall(r"^(boto3|botocore)>=([\d.]+)", requirements,
                                 re.M))
        self.assertEqual(set(floors), {"boto3", "botocore"}, requirements)
        for package, floor in floors.items():
            parsed = tuple(int(part) for part in floor.split("."))
            self.assertGreaterEqual(
                parsed, BOTOCORE_FLOOR,
                f"{package}>={floor} is below the version that first carries "
                "modelInvocationType; create_job would fail client-side")

        dockerfile = _read(ROOT, "Dockerfile")
        tags = re.findall(r"^FROM\s+public\.ecr\.aws/lambda/python:(\S+)",
                          dockerfile, re.M)
        self.assertEqual(len(tags), 1, tags)
        # 3.13.YYYY.MM.DD.HH — dated, never floating on :3.13, so that an
        # unchanged .version cannot put a different digest behind the same ECR tag.
        match = re.fullmatch(r"3\.13\.(\d{4})\.(\d{2})\.(\d{2})\.\d{2}", tags[0])
        self.assertIsNotNone(match, f"{tags[0]} is not a dated 3.13 tag")
        self.assertGreaterEqual(
            tuple(int(part) for part in match.groups()), BASE_IMAGE_FLOOR_DATE,
            f"base tag {tags[0]} predates the first 3.13 image shipping "
            f"botocore {'.'.join(map(str, BOTOCORE_FLOOR))}")

    def test_the_base_image_still_carries_the_error_shape_the_semaphore_reads(self):
        """The other half of the pin, and the reason bumping it is not automatic:
        try_acquire tells "saturated" from "re-entrant" by reading
        `CancellationReasons` off TransactionCanceledException. That member is
        parsed from the service model, so a base image whose botocore dropped or
        renamed it would make the cap silently unenforceable — the workflow would
        succeed and admit four holders into three slots."""
        from botocore.session import get_session

        shape = get_session().get_service_model("dynamodb").shape_for(
            "TransactionCanceledException")
        self.assertIn("CancellationReasons", shape.members)


class TestBatchSubmitClassification(_WorkflowCase):
    def _submit(self, control, plan=None, records=None):
        plan = plan or self.plan(request=_request(mode="batch"),
                                 records=records or _records(4))
        plan["task_token"] = "token-abc"  # nosec B105
        with mock.patch.object(batch_client, "bedrock_control",
                               return_value=control):
            return plan, submit_batch.lambda_handler(plan, None)

    def test_adaptive_default_pads_thread_through_the_submit_path(self):
        """The disabled-shape pin above covers the retired Sonnet 5 fixture;
        every SHIPPED 5-family entry is adaptive-default — pin the
        adaptive pad shape (thinking adaptive + effort low, no temperature;
        live-probed ACCEPTED 2026-10-15) end-to-end through
        _write_batch_input, not just padding_record in isolation."""
        self.s3.put(PROXY, input_schema.MODEL_REGISTRY_KEY,
                    REGISTRY + PREFILL_REGISTRY)
        control = _FakeControl([{"jobArn": "arn:aws:bedrock:eu-west-1:1:"
                                 "model-invocation-job/adap"}])
        plan, result = self._submit(
            control,
            plan=self.plan(request=_request(mode="batch", options={},
                                            model_chain=[OPUS5]),
                           records=_records(4)))
        self.assertEqual(result["status"], token_table.STATUS_SUBMITTED)
        rows = self.s3.lines_at(PROXY, plan["data"]["keys"]["batch_input"])
        pads = [r for r in rows
                if bedrock_lib.is_padding(r.get("recordId", ""))]
        self.assertTrue(pads)
        pad = pads[0]["modelInput"]
        self.assertEqual(pad["additionalModelRequestFields"]["thinking"],
                         {"type": "adaptive"})
        self.assertEqual(pad["additionalModelRequestFields"]["output_config"],
                         {"effort": "low"})
        self.assertNotIn("temperature",
                         json.dumps(pad.get("inferenceConfig") or {}))

    def test_batch_submit_survives_records_without_max_tokens(self):
        """THE 2026-08-18 REGRESSION PIN. When absent max_tokens stopped being
        materialized into parsed records, submit_batch's bracket access broke
        EVERY batch submit — and this suite stayed green because _request()'s
        fixture default (`options: {"max_tokens": 4096}`) materialized the key
        into every batch test's records. The model_matrix harness caught it
        live within the hour. This test is the offline half: a real plan with
        EMPTY options (the default-relying caller) must submit a batch job."""
        control = _FakeControl([{"jobArn": "arn:aws:bedrock:eu-west-1:1:"
                                 "model-invocation-job/regr"}])
        plan, result = self._submit(
            control,
            plan=self.plan(request=_request(mode="batch", options={}),
                           records=_records(4)))
        self.assertEqual(result["status"], token_table.STATUS_SUBMITTED)

    def test_every_quota_shaped_refusal_queues_the_row_as_pending(self):
        """
        §6.2's load-bearing classification. Queueing a permanent error retries it
        forever, invisibly, until the caller's token times out; failing a quota
        refusal throws away a submit the drain would have completed.
        """
        for code in bedrock_lib.BATCH_QUOTA_EXCEPTIONS:
            with self.subTest(code=code):
                self.ddb.items.clear()
                control = _FakeControl(
                    [_client_error(code, "CreateModelInvocationJob")])
                plan, result = self._submit(control)

                self.assertEqual(result["status"], token_table.STATUS_PENDING)
                self.assertIn(code, result["reason"])
                row = token_table.get_row(BATCH_MODEL,
                                          plan["data"]["workflow_id"])
                self.assertEqual(row["status"], token_table.STATUS_PENDING)
                self.assertEqual(row["task_token"], "token-abc")
                self.assertTrue(row["queued_reason"])

    def test_a_non_quota_refusal_raises_and_is_never_queued(self):
        """
        The row is left SUBMITTING rather than PENDING or deleted: it is the only
        record that a submit was attempted, it expires on its own TTL, and a PENDING
        row here would be retried forever by every future drain.
        """
        control = _FakeControl(
            [_client_error("ValidationException", "CreateModelInvocationJob")])
        plan = self.plan(request=_request(mode="batch"), records=_records(4))
        plan["task_token"] = "token-abc"  # nosec B105
        with mock.patch.object(batch_client, "bedrock_control",
                               return_value=control), \
             self.assertRaises(ClientError):
            submit_batch.lambda_handler(plan, None)

        row = token_table.get_row(BATCH_MODEL, plan["data"]["workflow_id"])
        self.assertEqual(row["status"], token_table.STATUS_SUBMITTING)
        self.assertNotEqual(row["status"], token_table.STATUS_PENDING)

    def test_a_successful_submit_writes_the_row_before_calling_bedrock(self):
        control = _FakeControl()
        plan, result = self._submit(control)
        self.assertEqual(result["status"], token_table.STATUS_SUBMITTED)
        self.assertEqual(result["job_arn"], JOB_ARN)

        row = token_table.get_row(BATCH_MODEL, plan["data"]["workflow_id"])
        self.assertEqual(row["job_arn"], JOB_ARN)
        # put_item precedes create_model_invocation_job: a Lambda death in between
        # must still leave a record of the attempt.
        self.assertLess(self.ddb.calls.index("put_item"),
                        self.ddb.calls.index("update_item"))

        job = control.calls[0]
        self.assertEqual(job["jobName"], plan["data"]["workflow_id"])
        self.assertEqual(job["modelId"], BATCH_MODEL)
        self.assertEqual(job["timeoutDurationInHours"],
                         input_schema.BATCH_TIMEOUT_HOURS)

    def test_the_output_uri_ends_with_a_slash_and_the_input_uri_does_not(self):
        """
        Bedrock's `outputDataConfig.s3OutputDataConfig.s3Uri` pattern ends in a
        literal `/` and rejects a prefix without one. This was a real deploy-time
        failure and it is an expensive shape of failure: the refusal is a
        `ValidationException`, which §6.2 correctly classifies as permanent, so the
        row is left SUBMITTING, nothing retries, and the caller's token stays parked
        for the full 24 h with the reason only in the submit function's log.

        Asserted on the params the client actually received, not on the plan's keys —
        `batch_output_prefix` legitimately carries no trailing slash and the
        normalisation happens at the API boundary in `create_job`, so that it also
        covers the drain replaying a PERSISTED `output_s3uri` written before this fix.

        The input uri must NOT be normalised: it names a single JSONL object, and a
        trailing slash there would point at a key that does not exist.
        """
        control = _FakeControl()
        self._submit(control)
        job = control.calls[0]

        out_uri = job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        self.assertTrue(out_uri.endswith("/"), out_uri)
        self.assertFalse(out_uri.endswith("//"), out_uri)

        in_uri = job["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
        self.assertFalse(in_uri.endswith("/"), in_uri)

        # Both must satisfy Bedrock's own documented pattern, which is what the
        # service rejected on — a hand-rolled endswith check would pass on
        # `s3://BUCKET//`, and the regex is the thing being violated.
        pattern = re.compile(
            r"^s3://[a-z0-9][-.a-z0-9]{1,61}[a-z0-9]"
            r"(?:/[-!_*'().a-z0-9A-Z]+(?:/[-!_*'().a-z0-9A-Z]+)*)?/$")
        self.assertRegex(out_uri, pattern)

        # Already-slashed input is idempotent, which is what makes replaying a
        # persisted uri safe whether it was written before or after this fix.
        control2 = _FakeControl()
        with mock.patch.object(batch_client, "bedrock_control",
                               return_value=control2):
            batch_client.create_job(
                job_name="wf", model_id=BATCH_MODEL,
                input_s3uri="s3://bucket/in.jsonl",
                output_s3uri="s3://bucket/out/",
                role_arn="arn:aws:iam::123456789012:role/r")
        self.assertEqual(
            control2.calls[0]["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"],
            "s3://bucket/out/")

    def test_the_persisted_region_is_an_aws_region_never_a_geo(self):
        """The value `drain` replays on every retry. A geo here outlives the
        execution."""
        control = _FakeControl()
        plan, _ = self._submit(control)
        row = token_table.get_row(BATCH_MODEL, plan["data"]["workflow_id"])
        self.assertEqual(row["model_params"]["region"], input_schema.AWS_REGION)
        self.assertNotIn(row["model_params"]["region"],
                         bedrock_lib._BATCH_GEO_PREFIXES)

    def test_submit_without_a_task_token_raises(self):
        plan = self.plan(request=_request(mode="batch"), records=_records(4))
        with self.assertRaises(ValidationException):
            submit_batch.lambda_handler(plan, None)

    def test_batch_input_is_padded_and_holds_no_cache_point(self):
        control = _FakeControl()
        plan, _ = self._submit(control, records=_records(40))
        rows = self.s3.lines_at(PROXY, plan["data"]["keys"]["batch_input"])
        self.assertEqual(len(rows), input_schema.MIN_BATCH_RECORDS)
        self.assertNotIn("cachePoint", json.dumps(rows))
        self.assertNotIn("modelId", rows[0])       # hoisted out for batch
        self.assertEqual(len(control.calls), 1)
        # the pads must carry the BATCH MODEL's contract — this is the pin
        # on submit_batch threading its spec into iter_pads: reverting to
        # iter_pads(count, None) drops the thinking field below and would
        # fail an opus-5-class job ALL-OR-NOTHING (probed 2026-10-15)
        pads = [r for r in rows
                if bedrock_lib.is_padding(r.get("recordId", ""))]
        self.assertTrue(pads)
        pad = pads[0]["modelInput"]
        self.assertNotIn("temperature",
                         json.dumps(pad.get("inferenceConfig") or {}))
        self.assertEqual(
            pad["additionalModelRequestFields"]["thinking"],
            {"type": "disabled"})     # PRIMARY's thinking_default


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 item 8 — batch drain
# ─────────────────────────────────────────────────────────────────────────────

class TestDrain(_WorkflowCase):
    def _pending(self, model_id, workflow_id, created_at=0):
        token_table.put_submitting(
            model_id=model_id, workflow_id=workflow_id, task_token=f"t-{workflow_id}",
            request_s3uri=REQUEST_S3URI,
            model_params={"job_name": workflow_id, "model_id": model_id,
                          "input_s3uri": f"s3://{PROXY}/{PREFIX}/batch/input.jsonl",
                          "output_s3uri": f"s3://{PROXY}/{PREFIX}/batch/output",
                          "region": input_schema.AWS_REGION})
        token_table.mark_pending(model_id, workflow_id, "queued for the test")
        key = (token_table.batch_pk(model_id), workflow_id)
        self.ddb.items[key]["created_at"] = {"N": str(created_at)}

    def test_stops_on_the_first_quota_wall_leaving_the_remainder_pending(self):
        for index in range(4):
            self._pending(BATCH_MODEL, f"wf-{index}", created_at=index)
        control = _FakeControl([
            None, None,
            _client_error("ServiceQuotaExceededException",
                          "CreateModelInvocationJob"),
        ])
        with mock.patch.object(batch_client, "bedrock_control",
                               return_value=control):
            result = drain.lambda_handler({"operation": "Drain"}, None)

        self.assertEqual(result["submitted"], 2)
        self.assertTrue(result["detail"][BATCH_MODEL]["stopped_on_quota"])
        statuses = [token_table.get_row(BATCH_MODEL, f"wf-{i}")["status"]
                    for i in range(4)]
        self.assertEqual(statuses, [token_table.STATUS_SUBMITTED,
                                    token_table.STATUS_SUBMITTED,
                                    token_table.STATUS_PENDING,
                                    token_table.STATUS_PENDING])

    def test_drains_oldest_first_by_created_at_not_by_sort_key(self):
        """
        The sort key is `workflow_id`, a uuid5 — key order is arbitrary. Draining in
        key order admits callers out of turn, invisibly, and starves whichever
        workflow_id happens to hash low.
        """
        self._pending(BATCH_MODEL, "zzz-oldest", created_at=1)
        self._pending(BATCH_MODEL, "aaa-newest", created_at=2)
        rows = token_table.query_pending(BATCH_MODEL)
        self.assertEqual([row["sk"] for row in rows], ["zzz-oldest", "aaa-newest"])

    def test_a_stream_event_for_one_model_does_not_drain_another(self):
        """
        Capacity is accounted per model. Draining B on A's news submits B's queue
        against a limit nothing told us about.
        """
        self._pending(BATCH_MODEL, "wf-a")
        self._pending("us.amazon.nova-pro-v1:0", "wf-b")
        event = {"Records": [_stream_record(
            "REMOVE", old={"pk": token_table.batch_pk(BATCH_MODEL),
                           "sk": "wf-done", "model_id": BATCH_MODEL})]}
        control = _FakeControl()
        with mock.patch.object(batch_client, "bedrock_control",
                              return_value=control):
            result = drain.lambda_handler(event, None)

        self.assertEqual(result["models"], [BATCH_MODEL])
        self.assertEqual(result["submitted"], 1)
        self.assertEqual(
            token_table.get_row("us.amazon.nova-pro-v1:0", "wf-b")["status"],
            token_table.STATUS_PENDING)

    def test_a_scheduled_sweep_with_no_image_drains_every_queued_model(self):
        """§6.5 fix 2, and what makes the queue self-healing rather than
        event-dependent."""
        self._pending(BATCH_MODEL, "wf-a")
        self._pending("us.amazon.nova-pro-v1:0", "wf-b")
        control = _FakeControl()
        with mock.patch.object(batch_client, "bedrock_control",
                              return_value=control):
            result = drain.lambda_handler({"operation": "Drain"}, None)
        self.assertEqual(result["models"],
                         sorted([BATCH_MODEL, "us.amazon.nova-pro-v1:0"]))
        self.assertEqual(result["submitted"], 2)
        self.assertIn("scan", self.ddb.calls)

    def test_a_foreign_bedrock_completion_drains_everything(self):
        """
        §6.5 fix 1. The account-level job limit is shared and this stack is not a
        singleton, so a FOREIGN completion is the only signal that capacity freed.
        Its arn is not in our table, so all we learn is "there is capacity now".
        """
        self._pending(BATCH_MODEL, "wf-a")
        control = _FakeControl()
        event = {"source": "aws.bedrock",
                 "detail": {"batchJobArn": "arn:aws:bedrock:us-east-1:999:job/x",
                            "status": "Completed"}}
        with mock.patch.object(batch_client, "bedrock_control",
                              return_value=control):
            result = drain.lambda_handler(event, None)
        self.assertEqual(result["submitted"], 1)

    def test_nothing_queued_is_not_an_error(self):
        result = drain.lambda_handler({"operation": "Drain"}, None)
        self.assertEqual(result, {"models": [], "submitted": 0,
                                  "reason": "nothing to drain"})

    def test_a_permanent_submit_failure_marks_the_row_failed(self):
        """
        Marked FAILED so the stream tells the notifier, which fails the parked caller
        with the reason — rather than leaving it PENDING to be retried forever by
        every future drain.
        """
        self._pending(BATCH_MODEL, "wf-bad")
        control = _FakeControl(
            [_client_error("AccessDeniedException", "CreateModelInvocationJob")])
        with mock.patch.object(batch_client, "bedrock_control",
                              return_value=control):
            result = drain.lambda_handler({"operation": "Drain"}, None)
        self.assertEqual(result["submitted"], 0)
        row = token_table.get_row(BATCH_MODEL, "wf-bad")
        self.assertEqual(row["status"], token_table.STATUS_FAILED)
        self.assertIn("drain submit failed permanently", row["status_message"])

    def test_a_row_without_usable_params_is_left_pending_not_deleted(self):
        token_table.put_submitting(
            model_id=BATCH_MODEL, workflow_id="wf-partial", task_token="t",  # nosec B106
            request_s3uri=REQUEST_S3URI, model_params={})
        token_table.mark_pending(BATCH_MODEL, "wf-partial", "queued")
        with self.assertLogs(drain.logger, level="ERROR"):
            result = drain.lambda_handler({"operation": "Drain"}, None)
        self.assertEqual(result["submitted"], 0)
        self.assertEqual(token_table.get_row(BATCH_MODEL, "wf-partial")["status"],
                         token_table.STATUS_PENDING)

    def test_the_per_invocation_bound_is_honoured(self):
        for index in range(drain.MAX_SUBMITS_PER_MODEL + 5):
            self._pending(BATCH_MODEL, f"wf-{index:03d}", created_at=index)
        control = _FakeControl()
        with mock.patch.object(batch_client, "bedrock_control",
                              return_value=control):
            result = drain.lambda_handler({"operation": "Drain"}, None)
        self.assertEqual(result["submitted"], drain.MAX_SUBMITS_PER_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 items 9 and 10 — notifier idempotency, and the token lookup
# ─────────────────────────────────────────────────────────────────────────────

class TestNotifier(_WorkflowCase):
    def setUp(self):
        super().setUp()
        self.sfn = _FakeSfn()
        self._saved_sfn = notify._SFN
        notify._SFN = self.sfn

    def tearDown(self):
        notify._SFN = self._saved_sfn
        super().tearDown()

    @staticmethod
    def _row(status, **extra):
        row = {"pk": token_table.batch_pk(BATCH_MODEL), "sk": "wf-1",
               "model_id": BATCH_MODEL, "task_token": "tok-1", "status": status,  # nosec B105
               "job_arn": JOB_ARN}
        row.update(extra)
        return row

    def test_a_redelivered_terminal_transition_sends_once(self):
        """
        Checking the OLD image makes this a TRANSITION trigger rather than a STATE
        trigger — the cheap half of idempotency. The SFN error is the other half.
        """
        completed = self._row(token_table.STATUS_COMPLETED)
        first = notify.lambda_handler(
            {"Records": [_stream_record("MODIFY", new=completed,
                                        old=self._row("SUBMITTED"))]}, None)
        second = notify.lambda_handler(
            {"Records": [_stream_record("MODIFY", new=completed,
                                        old=completed)]}, None)
        self.assertEqual(first["notified"], 1)
        self.assertEqual(second, {"notified": 0, "dead_token": 0, "skipped": 1})  # nosec B105
        self.assertEqual(len(self.sfn.successes), 1)

    def test_a_failed_job_uses_the_failure_channel(self):
        """
        §12.3's error-as-success site. A caller handed {"status": "FAILED"} through
        the SUCCESS channel has to remember to check it, and the one that forgets
        treats an error payload as a result. Step Functions has a failure channel.
        """
        notify.lambda_handler({"Records": [_stream_record(
            "MODIFY", new=self._row(token_table.STATUS_FAILED,
                                    status_message="Expired: ran out of time"),
            old=self._row("SUBMITTED"))]}, None)
        self.assertEqual(self.sfn.successes, [])
        self.assertEqual(len(self.sfn.failures), 1)
        self.assertEqual(self.sfn.failures[0]["error"], "BedrockInferenceFailed")
        self.assertIn("Expired", self.sfn.failures[0]["cause"])

    def test_success_output_carries_the_job_arn(self):
        """The notifier DELETES the row after sending, so by the time
        `Collect batch output` runs the table lookup is usually a miss — which is
        why job_arn is in the task output at all."""
        notify.lambda_handler({"Records": [_stream_record(
            "MODIFY", new=self._row(token_table.STATUS_COMPLETED),
            old=self._row("SUBMITTED"))]}, None)
        payload = json.loads(self.sfn.successes[0]["output"])
        self.assertEqual(payload["job_arn"], JOB_ARN)
        self.assertEqual(payload["workflow_id"], "wf-1")

    def test_a_dead_token_is_swallowed_and_the_row_cleaned_up(self):
        for code in notify.DEAD_TOKEN_CODES:
            with self.subTest(code=code):
                self.ddb.items.clear()
                self.ddb.items[(token_table.batch_pk(BATCH_MODEL), "wf-1")] = \
                    token_table.to_item(self._row(token_table.STATUS_COMPLETED))
                notify._SFN = _FakeSfn(error=_client_error(code, "SendTaskSuccess"))
                result = notify.lambda_handler({"Records": [_stream_record(
                    "MODIFY", new=self._row(token_table.STATUS_COMPLETED),
                    old=self._row("SUBMITTED"))]}, None)
                self.assertEqual(result["dead_token"], 1)
                self.assertIsNone(token_table.get_row(BATCH_MODEL, "wf-1"))

    def test_every_other_send_task_error_re_raises_so_the_stream_retries(self):
        """A transient failure means the token is still LIVE. Swallowing it strands
        the caller for its full 24 h."""
        notify._SFN = _FakeSfn(
            error=_client_error("ThrottlingException", "SendTaskSuccess"))
        with self.assertRaises(ClientError):
            notify.lambda_handler({"Records": [_stream_record(
                "MODIFY", new=self._row(token_table.STATUS_COMPLETED),
                old=self._row("SUBMITTED"))]}, None)

    def test_a_remove_is_the_drains_trigger_not_the_notifiers(self):
        result = notify.lambda_handler({"Records": [_stream_record(
            "REMOVE", old=self._row(token_table.STATUS_COMPLETED))]}, None)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(self.sfn.successes, [])

    def test_a_non_terminal_status_is_skipped(self):
        result = notify.lambda_handler({"Records": [_stream_record(
            "MODIFY", new=self._row(token_table.STATUS_SUBMITTED),
            old=self._row(token_table.STATUS_SUBMITTING))]}, None)
        self.assertEqual(result["skipped"], 1)

    def test_a_terminal_row_with_no_task_token_is_logged_not_sent(self):
        row = self._row(token_table.STATUS_COMPLETED)
        row.pop("task_token")
        with self.assertLogs(notify.logger, level="ERROR"):
            result = notify.lambda_handler({"Records": [_stream_record(
                "MODIFY", new=row, old=self._row("SUBMITTED"))]}, None)
        self.assertEqual(result["skipped"], 1)


class TestTokenLookup(_WorkflowCase):
    def test_a_completion_finds_its_row_by_the_gsi_with_no_second_table(self):
        """
        The GSI's whole reason for existing: the EventBridge completion event carries
        a `batchJobArn` and nothing else. v5 answers the same question with a SECOND
        TABLE and a jobName join.
        """
        token_table.put_submitting(
            model_id=BATCH_MODEL, workflow_id="wf-1", task_token="tok",  # nosec B106
            request_s3uri=REQUEST_S3URI, model_params={"model_id": BATCH_MODEL})
        token_table.mark_submitted(BATCH_MODEL, "wf-1", JOB_ARN)

        row = token_table.find_by_job_arn(JOB_ARN)
        self.assertEqual(row["sk"], "wf-1")
        self.assertEqual(row["model_id"], BATCH_MODEL)
        self.assertEqual(self.ddb.tables, {os.environ["SERVICE_TOKEN_TABLE_NAME"]})

    def test_an_unknown_arn_finds_nothing(self):
        self.assertIsNone(token_table.find_by_job_arn("arn:aws:bedrock:::job/none"))

    def test_batch_status_marks_terminal_and_ignores_the_redelivery(self):
        token_table.put_submitting(
            model_id=BATCH_MODEL, workflow_id="wf-1", task_token="tok",  # nosec B106
            request_s3uri=REQUEST_S3URI, model_params={})
        token_table.mark_submitted(BATCH_MODEL, "wf-1", JOB_ARN)

        event = {"source": "aws.bedrock", "detail-type": "Batch Inference Job State Change",
                 "detail": {"batchJobArn": JOB_ARN, "status": "PartiallyCompleted",
                            "message": "3 records failed"}}
        first = batch_status.lambda_handler(event, None)
        self.assertTrue(first["handled"])
        self.assertTrue(first["wrote"])
        # PartiallyCompleted is COMPLETED on purpose: per-record failures become
        # FAILED lines in response.jsonl, which is partial success, not a failed job.
        self.assertEqual(first["status"], token_table.STATUS_COMPLETED)

        second = batch_status.lambda_handler(event, None)
        self.assertTrue(second["handled"])
        self.assertFalse(second["wrote"],
                         "the compound condition must refuse a terminal rewrite")

    def test_every_bedrock_terminal_status_maps(self):
        self.assertEqual(
            set(token_table.BEDROCK_STATUS_MAP),
            {"Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired"})
        for status in token_table.BEDROCK_STATUS_MAP.values():
            self.assertIn(status, token_table.TERMINAL_STATUSES)

    def test_a_non_terminal_event_is_ignored(self):
        result = batch_status.lambda_handler(
            {"source": "aws.bedrock", "detail": {"batchJobArn": JOB_ARN,
                                                 "status": "InProgress"}}, None)
        self.assertFalse(result["handled"])
        self.assertIn("non-terminal", result["reason"])

    def test_a_foreign_job_is_ignored_rather_than_retried(self):
        """Raising would make EventBridge retry a lookup that can never succeed."""
        result = batch_status.lambda_handler(
            {"source": "aws.bedrock",
             "detail": {"batchJobArn": "arn:aws:bedrock:us-east-1:999:job/foreign",
                        "status": "Completed"}}, None)
        self.assertFalse(result["handled"])
        self.assertEqual(result["reason"], "no row for this job arn")

    def test_mark_terminal_refuses_a_non_terminal_status(self):
        with self.assertRaises(ValidationException):
            token_table.mark_terminal(BATCH_MODEL, "wf-1",
                                      token_table.STATUS_PENDING)

    def test_pk_round_trips_and_rejects_a_foreign_partition(self):
        self.assertEqual(
            token_table.model_id_from_pk(token_table.batch_pk(BATCH_MODEL)),
            BATCH_MODEL)
        with self.assertRaises(ValidationException):
            token_table.model_id_from_pk("semaphore#bedrock#sonnet")

    def test_table_name_is_read_per_call_and_raises_when_unset(self):
        with mock.patch.dict(os.environ, {token_table.TABLE_NAME_ENV: ""}):
            with self.assertRaises(ValidationException):
                token_table.table_name()


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 item 11 — batch output normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectBatch(_WorkflowCase):
    def _write_output(self, plan, rows, shuffle=False):
        prefix = f"{plan['data']['keys']['batch_output_prefix']}/{JOB_ID}"
        self.s3.put(PROXY, f"{prefix}/manifest.json.out",
                    {"totalRecordCount": len(rows)})
        ordered = list(reversed(rows)) if shuffle else rows
        self.s3.put(PROXY, f"{prefix}/input.jsonl.out",
                    "".join(json.dumps(row) + "\n" for row in ordered))

    @staticmethod
    def _out(record_id, text="answer", stop_reason="end_turn"):
        return {"recordId": record_id,
                "modelOutput": {
                    "output": {"message": {"content": [{"text": text}]}},
                    "stopReason": stop_reason,
                    "usage": {"inputTokens": 20, "outputTokens": 8,
                              "totalTokens": 28}}}

    def _collect(self, plan):
        plan = dict(plan)
        plan["batch_result"] = {"status": "COMPLETED", "job_arn": JOB_ARN}
        return collect_batch.lambda_handler(plan, None)

    def _collect_incomplete(self, plan):
        """
        The repairable-gap path: this state RAISES (§8), and the ASL `Catch` on the
        exception name is what routes to the repair child. Nothing it would have
        returned survives, so every assertion below reads S3 — which is the point of
        write-then-raise.
        """
        with self.assertRaises(BatchRecordsIncompleteException) as caught:
            self._collect(plan)
        return caught.exception

    def test_one_per_record_error_yields_one_failed_line_and_a_success(self):
        """
        §5.2. A per-record failure is a FAILED LINE, not a failed job — the caller
        gets its other results and a summary that says so. The line is written before
        the repair signal is raised, so it survives the caught error.
        """
        records = _records(4)
        plan = self.plan(request=_request(mode="batch"), records=records)
        rows = [self._out(record["record_id"]) for record in records]
        rows[2] = {"recordId": records[2]["record_id"],
                   "error": {"errorCode": 400, "errorMessage": "content too long"}}
        self._write_output(plan, rows)

        self._collect_incomplete(plan)
        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual([line["status"] for line in lines],
                         ["SUCCESS", "SUCCESS", "FAILED", "SUCCESS"])
        self.assertEqual(
            sum(1 for line in lines if line["status"] == "SUCCESS"), 3)
        self.assertIn("content too long", lines[2]["reason"])
        # A rejected record produced no tokens, so it is priced at zero rather than
        # left unpriced — unpriced would make a COMPLETE cost figure look partial.
        self.assertEqual(lines[2]["estimated_cost"], 0.0)

    def test_records_are_joined_by_id_not_by_position(self):
        """Bedrock does not guarantee output order and interleaves padding
        records."""
        records = _records(4)
        plan = self.plan(request=_request(mode="batch"), records=records)
        self._write_output(plan, [self._out(record["record_id"], text=record["record_id"])
                                  for record in records], shuffle=True)
        self._collect(plan)
        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual([line["record_id"] for line in lines],
                         [record["record_id"] for record in records])
        for line in lines:
            self.assertEqual(line["output_text"], line["record_id"])

    def test_padding_records_are_dropped(self):
        records = _records(2)
        plan = self.plan(request=_request(mode="batch"), records=records)
        rows = [self._out(record["record_id"]) for record in records]
        rows.append(self._out("pad_0000", text="Hello"))
        self._write_output(plan, rows)
        self._collect(plan)
        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual(len(lines), 2)
        self.assertNotIn("pad_0000", json.dumps(lines))

    def test_a_missing_output_record_is_a_failed_line_with_zero_rounds(self):
        records = _records(3)
        plan = self.plan(request=_request(mode="batch"), records=records)
        self._write_output(plan, [self._out(records[0]["record_id"]),
                                  self._out(records[1]["record_id"])])
        self._collect_incomplete(plan)
        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual(lines[2]["status"], "FAILED")
        self.assertEqual(lines[2]["rounds"], 0)
        self.assertIn("no output record", lines[2]["reason"])

    def test_a_refusal_is_a_failed_line_priced_at_the_batch_rate(self):
        records = _records(1)
        plan = self.plan(request=_request(mode="batch"), records=records)
        self._write_output(plan, [self._out(records[0]["record_id"],
                                            stop_reason="content_filtered")])
        self._collect(plan)
        line = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])[0]
        self.assertEqual(line["status"], "FAILED")
        self.assertIn("content_filtered", line["reason"])

    def test_batch_lines_carry_exactly_one_round(self):
        """There is no retry ladder and no chain fallback inside a batch job — a
        distinction `summary.rounds` is meant to carry honestly."""
        records = _records(3)
        plan = self.plan(request=_request(mode="batch"), records=records)
        self._write_output(plan, [self._out(r["record_id"]) for r in records])
        self._collect(plan)
        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual([line["rounds"] for line in lines], [1, 1, 1])

    def test_batch_cost_is_half_the_on_demand_rate(self):
        spec = bedrock_lib.resolve_model(REGISTRY, PRIMARY)
        usage = {"inputTokens": 1000, "outputTokens": 100}
        self.assertAlmostEqual(spec.estimate_cost(usage, batch=True),
                               spec.estimate_cost(usage)
                               * bedrock_lib.BATCH_COST_MULTIPLIER)

    def test_null_usage_and_pricing_fields_price_as_zero(self):
        """
        Nova and Opus 5 batch output carries `"cacheReadInputTokens": null` —
        PRESENT, so `usage.get(key, 0)` returns the null and `None * 0.0` is the
        TypeError that failed four otherwise-complete batch executions on
        2026-08-13, AFTER Bedrock had run and billed the jobs. `.get()` defaults
        do not guard present-but-null keys; every token and pricing field
        defaults to 0 on null instead (settled 2026-08-14; narrowed 2026-10-16:
        the GPT-6 entries DO carry cache pricing now, and a null write field
        counts as absent and routes through reconstructed_cache_write — see
        TestPromptCacheAccounting).

        `normalize_usage` always had this guard — the trap was `_to_line`
        pricing the RAW usage before normalizing it, so the fix lives in
        `estimate_cost`, which every caller shares.
        """
        spec = bedrock_lib.resolve_model(REGISTRY, PRIMARY)
        nova_shaped = {"inputTokens": 1000, "outputTokens": 100,
                       "cacheReadInputTokens": None,
                       "cacheWriteInputTokens": None}
        self.assertAlmostEqual(
            spec.estimate_cost(nova_shaped, batch=True),
            spec.estimate_cost({"inputTokens": 1000, "outputTokens": 100},
                               batch=True))
        # Null PRICING fields too — same idiom, same trap, and "no price" must
        # keep meaning None (unpriced), never crash and never 0.0.
        broke = bedrock_lib.ModelSpec({
            "model_id": "m", "model_pricing": {
                "input_per_token_cost": None, "output_per_token_cost": None,  # nosec B105
                "cacheReadPerTokenCost": None}})
        self.assertIsNone(broke.estimate_cost({"inputTokens": 5}))
        half = bedrock_lib.ModelSpec({
            "model_id": "m", "model_pricing": {
                "input_per_token_cost": 1e-6, "output_per_token_cost": None,  # nosec B105
                "cacheReadPerTokenCost": None}})
        self.assertAlmostEqual(
            half.estimate_cost({"inputTokens": 10, "outputTokens": 10,
                                "cacheReadInputTokens": 7}), 1e-5)

    def test_the_output_listing_is_scoped_to_this_jobs_own_subfolder(self):
        """
        Bedrock writes to {s3Uri}/{jobId}/. Listing the shared parent picks up
        another job's partially-written output, and every execution of this workflow
        writes under the same micro-workflow/ root.
        """
        records = _records(1)
        plan = self.plan(request=_request(mode="batch"), records=records)
        self._write_output(plan, [self._out(records[0]["record_id"], text="mine")])
        other = f"{plan['data']['keys']['batch_output_prefix']}/otherjob"
        self.s3.put(PROXY, f"{other}/input.jsonl.out",
                    json.dumps(self._out(records[0]["record_id"],
                                         text="SOMEONE ELSE")) + "\n")
        self._collect(plan)
        line = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])[0]
        self.assertEqual(line["output_text"], "mine")

    def test_no_job_arn_anywhere_raises(self):
        plan = self.plan(request=_request(mode="batch"), records=_records(1))
        with self.assertRaises(ValidationException):
            collect_batch.lambda_handler(plan, None)


# ─────────────────────────────────────────────────────────────────────────────
# §8 — the batch repair tail, and the execution TREE it makes of the cost walk
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchRepairTail(_WorkflowCase):
    """
    §8 — the repair pass IS the sync child, started as a nested `.sync:2` execution
    from the batch child and routed to by a `Catch` on the exception `collect_batch`
    raises. Not a re-implementation of the sync spine inside the batch machine, which
    is what this replaced: the lease, the checkpoint and the release are the sync
    child's, unmodified, and the only thing the batch machine contributes is the
    seeded `_progress.json` that turns a fresh run into a gap-filling one.

    These tests model the ASL's data flow explicitly — `_catch_input` and
    `_run_repair_child` are what the two ASL states do — because the plumbing IS the
    design here: a caught error replaces a state's RESULT, not its INPUT, which is the
    single fact that makes `$.data` (and therefore the plan) still available to the
    repair.
    """

    REPAIR_ARN = ("arn:aws:states:us-east-1:123456789012:execution:"
                  "prefix-bedrock-inference-sync:wid-repair")

    def _write_output(self, plan, rows):
        prefix = f"{plan['data']['keys']['batch_output_prefix']}/{JOB_ID}"
        self.s3.put(PROXY, f"{prefix}/manifest.json.out",
                    {"totalRecordCount": len(rows)})
        self.s3.put(PROXY, f"{prefix}/input.jsonl.out",
                    "".join(json.dumps(row) + "\n" for row in rows))

    @staticmethod
    def _out(record_id, text="answer", stop_reason="end_turn"):
        return {"recordId": record_id,
                "modelOutput": {
                    "output": {"message": {"content": [{"text": text}]}},
                    "stopReason": stop_reason,
                    "usage": {"inputTokens": 20, "outputTokens": 8,
                              "totalTokens": 28}}}

    def _collect_input(self, plan):
        return {**plan,
                "batch_result": {"status": "COMPLETED", "job_arn": JOB_ARN}}

    def _collect(self, plan):
        return collect_batch.lambda_handler(self._collect_input(plan), None)

    def _catch_input(self, plan):
        """
        What Step Functions hands `Run repair inference`.

        `Collect batch output` raises, so its result is replaced by the error — but a
        caught error is combined with the state's **raw input**, and `Submit batch`
        merged its own result at `$.batch_result` rather than replacing the state input,
        so the whole plan is still on `$.data`. The Catch's explicit nested
        `ResultPath: $.data.batch_incomplete` is what keeps it that way; the default
        `$` would have replaced the input with the error object and left the repair
        with no plan to run.
        """
        with self.assertRaises(BatchRecordsIncompleteException) as caught:
            self._collect(plan)
        event = self._collect_input(plan)
        return {**event,
                "data": {**event["data"],
                         "batch_incomplete": {
                             "Error": "BatchRecordsIncompleteException",
                             "Cause": str(caught.exception)}}}

    def _run_repair_child(self, catch_input, runtime):
        """
        The nested `.sync:2` call: the child is handed `{uuid, input, data}` and the
        parent keeps two fields of its output through the `ResultSelector`.

        The child's own `Summarize` runs too, and writes `summary.json` — harmless, and
        deliberately exercised here: it is a pure function of `response.jsonl` plus the
        plan, so the batch child's Summarize overwrites it with the same numbers plus
        the repair's.
        """
        ran = self.run_sync({"uuid": catch_input.get("uuid") or "",
                             "input": catch_input.get("input") or {},
                             "data": catch_input["data"]}, runtime)
        summarize.lambda_handler(ran, None)
        return {"execution_arn": self.REPAIR_ARN,
                "records_resumed": ran["data"]["records_resumed"]}

    def _batch_with_failures(self, total=4, failed=(2,)):
        records = _records(total)
        plan = self.plan(request=_request(mode="batch"), records=records)
        rows = []
        for index, record in enumerate(records):
            if index in failed:
                rows.append({"recordId": record["record_id"],
                             "error": {"errorCode": 500,
                                       "errorMessage": "transient"}})
            else:
                rows.append(self._out(record["record_id"]))
        self._write_output(plan, rows)
        return records, plan

    def test_collect_seeds_the_checkpoint_and_raises_the_routing_signal(self):
        """
        Write-then-raise, in that order. Both S3 objects the repair depends on are
        durable BEFORE the exception, so the raise is a routing signal over committed
        state rather than a lost result — raising first would leave the repair with no
        checkpoint to resume from and the run with no `response.jsonl` at all.
        """
        records, plan = self._batch_with_failures()
        with self.assertRaises(BatchRecordsIncompleteException) as caught:
            self._collect(plan)

        state = self.s3.json_at(PROXY, plan["data"]["keys"]["progress"])
        self.assertEqual(set(state["done"]),
                         {r["record_id"] for i, r in enumerate(records)
                          if i != 2})
        self.assertEqual(state["records_total"], 4)
        self.assertEqual(len(self.s3.lines_at(
            PROXY, plan["data"]["keys"]["response"])), 4)
        # The message is the only thing the Catch carries forward, so it names the
        # counts and says both objects are already written.
        self.assertIn("1/4 record(s) incomplete", str(caught.exception))
        self.assertIn("checkpoint seeded with 3", str(caught.exception))

    def test_the_plan_survives_the_caught_error_so_the_repair_can_run(self):
        """
        The one fact the whole shape rests on: a caught error replaces the state's
        RESULT, not its INPUT. `Submit batch` merged at `$.batch_result`, so `$.data`
        is untouched and the repair child receives the same plan the sync path would
        have been started with.
        """
        _records_, plan = self._batch_with_failures()
        data = self._catch_input(plan)["data"]
        for key in ("workflow_id", "keys", "config", "request", "output", "quota"):
            self.assertIn(key, data, "the repair child cannot run without the plan")
        self.assertEqual(data["batch_incomplete"]["Error"],
                         "BatchRecordsIncompleteException")

    def test_the_repair_is_the_unmodified_sync_child_filling_only_the_gaps(self):
        """
        The child resumes from the seeded checkpoint: ONE Converse call for the one
        failed record, `response.jsonl` complete and in request order, and the batch
        child's Summarize reports what was RECOVERED — `succeeded - resumed`, derived
        from the child's own number because nothing `collect_batch` returned survived.
        """
        _records_, plan = self._batch_with_failures()
        caught = self._catch_input(plan)
        runtime = _FakeRuntime([_converse_response(text="repaired!")])
        repair = self._run_repair_child(caught, runtime)

        self.assertEqual(len(runtime.calls), 1,
                         "only the gap may be re-inferred — the successful "
                         "batch lines are already done")
        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual([line["status"] for line in lines], ["SUCCESS"] * 4)
        self.assertEqual(lines[2]["output_text"], "repaired!")
        self.assertEqual(repair["records_resumed"], 3)

        # ResultPath: $.data.repair — the batch child's own Summarize, running last.
        summary_out = summarize.lambda_handler(
            {**caught, "data": {**caught["data"], "repair": repair}}, None)
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])
        self.assertEqual(summary["records_repaired"], 1)
        self.assertEqual(summary["records_succeeded"], 4)
        self.assertEqual(summary["repair_execution_arn"], self.REPAIR_ARN)
        self.assertIn("+ sync repair", summary["mode_reason"])
        self.assertEqual(summary_out["data"]["records_failed"], 0)
        # A repaired batch run DID take a lease — the repair is the sync child, lease
        # and all — so reporting "" would hide it from the split-brain check.
        self.assertEqual(summary["quota_table"],
                         semaphore_client.table_name())

    def test_a_record_that_fails_the_repair_too_stays_failed_and_is_counted(self):
        _records_, plan = self._batch_with_failures()
        caught = self._catch_input(plan)
        malformed = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "malformed"}},
            "Converse")
        repair = self._run_repair_child(caught, _FakeRuntime([malformed]))

        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual(lines[2]["status"], "FAILED")
        result = summarize.lambda_handler(
            {**caught, "data": {**caught["data"], "repair": repair}}, None)
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])
        self.assertEqual(summary["records_repaired"], 0)
        self.assertEqual(result["data"]["records_failed"], 1)

    def test_a_failed_repair_degrades_to_a_normal_summary(self):
        """
        The second Catch, on `States.ALL`. Repair-off would have SUCCEEDED with these
        same FAILED lines, so repair-on must not do worse: the execution finishes with
        the response it already has and the reason in the summary.
        """
        _records_, plan = self._batch_with_failures()
        caught = self._catch_input(plan)
        degraded = {**caught,
                    "data": {**caught["data"],
                             "repair_error": {
                                 "Error": "States.ExecutionLimitExceeded",
                                 "Cause": "too many open executions"}}}
        result = summarize.lambda_handler(degraded, None)
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])

        self.assertIn("too many open executions", summary["repair_error"])
        self.assertIn("failed, degraded", summary["mode_reason"])
        self.assertNotIn("records_repaired", summary,
                         "nothing was recovered, so there is no figure to report")
        # 3 of 4 is partial success, which is success — the same outcome the run would
        # have had with the repair switched off.
        self.assertEqual(result["data"]["records_succeeded"], 3)
        self.assertEqual(result["data"]["records_failed"], 1)

    def test_the_guards(self):
        """
        Bounded and opt-out: a repairable gap RAISES (the Catch is the routing), and
        every refusal returns normally to `Summarize` instead. All-failed is systemic,
        a clean run needs none, over-half is refused, and options.batch_repair=false
        wins over all of it.
        """
        def refused(plan):
            skipped = self._collect(plan)["data"].get("repair_skipped") or {}
            return skipped.get("reason") or "nothing to repair"

        # all failed -> systemic, and Summarize raises BedrockInferenceFailed as ever
        _r, plan = self._batch_with_failures(total=4, failed=(0, 1, 2, 3))
        self.assertIn("systemic", refused(plan))
        # clean -> nothing to repair, and no repair_skipped key to explain
        _r, plan = self._batch_with_failures(total=4, failed=())
        self.assertNotIn("repair_skipped", self._collect(plan)["data"])
        # 3 of 4 -> over the 50% bound
        _r, plan = self._batch_with_failures(total=4, failed=(0, 1, 2))
        self.assertIn("systemic", refused(plan))
        # exactly half -> allowed, so it raises
        _r, plan = self._batch_with_failures(total=4, failed=(0, 1))
        with self.assertRaises(BatchRecordsIncompleteException):
            self._collect(plan)
        # caller opt-out beats everything
        request = _request(mode="batch")
        request["options"] = {**(request.get("options") or {}),
                              "batch_repair": False}
        records = _records(4)
        plan = self.plan(request=request, records=records)
        rows = [self._out(r["record_id"]) for r in records[:3]]
        rows.append({"recordId": records[3]["record_id"],
                     "error": {"errorCode": 500, "errorMessage": "x"}})
        self._write_output(plan, rows)
        self.assertIn("disabled", refused(plan))

    def test_a_refused_repair_says_why_in_the_summary(self):
        """
        A caller staring at 3 FAILED lines is owed the reason the tail did not run, and
        `summary.json` is the only object it reads. `collect_batch` returned normally
        here, so — uniquely among the repair outcomes — its own value is what reports it.
        """
        _r, plan = self._batch_with_failures(total=4, failed=(0, 1, 2))
        collected = self._collect(plan)
        # 3 failed, 1 succeeded: partial success IS success, so this returns.
        result = summarize.lambda_handler(collected, None)
        summary = self.s3.json_at(PROXY, plan["data"]["keys"]["summary"])
        self.assertIn("no sync repair", summary["mode_reason"])
        self.assertIn("systemic", summary["mode_reason"])
        self.assertEqual(result["data"]["records_failed"], 3)
        # No repair ran, so none of the repair fields appear at all.
        for absent in ("records_repaired", "repair_execution_arn", "repair_error"):
            self.assertNotIn(absent, summary)

    def test_truncated_lines_are_success_and_never_repaired(self):
        """stop_reason max_tokens is a SUCCESS with partial text — re-running it
        truncates again; salvage is the caller's parse_partial_output."""
        records = _records(3)
        plan = self.plan(request=_request(mode="batch"), records=records)
        rows = [self._out(records[0]["record_id"]),
                self._out(records[1]["record_id"], text="partia",
                          stop_reason="max_tokens"),
                self._out(records[2]["record_id"])]
        self._write_output(plan, rows)
        # Returns rather than raising: a truncated line is a SUCCESS, so there is no
        # gap and no repair to route to.
        result = self._collect(plan)
        self.assertNotIn("repair_skipped", result["data"])
        lines = self.s3.lines_at(PROXY, plan["data"]["keys"]["response"])
        self.assertEqual(lines[1]["status"], "SUCCESS")
        self.assertEqual(lines[1]["stop_reason"], "max_tokens")


_TREE_ROOT = ("arn:aws:states:us-east-1:123456789012:execution:"
              "pfx-bedrock-inference:wid")
_TREE_BATCH = ("arn:aws:states:us-east-1:123456789012:execution:"
               "pfx-bedrock-inference-batch:wid-batch")
_TREE_REPAIR = ("arn:aws:states:us-east-1:123456789012:execution:"
                "pfx-bedrock-inference-sync:wid-repair")


class _FakeSfnHistories:
    """`GetExecutionHistory` over a dict of arn → synthetic event list."""

    def __init__(self, histories):
        self.histories = histories
        self.calls = []

    def get_execution_history(self, **kwargs):
        self.calls.append(kwargs)
        return {"events": self.histories.get(kwargs["executionArn"], [])}


class _FakeLambdaConfig:
    def get_function_configuration(self, FunctionName):
        return {"MemorySize": 1024, "Architectures": ["arm64"]}


def _hist(eid, prev, etype, second, **details):
    return {"id": eid, "previousEventId": prev, "type": etype,
            "timestamp": f"2026-08-12T00:00:{second:02d}+00:00", **details}


def _child_task_history(state_name, child_arn, start=0, end=50):
    """One `.sync:2` state that starts `child_arn` and succeeds.

    `TaskSubmitted` carries the arn in its `output`, a JSON STRING, which is the whole
    reason the walk asks for `includeExecutionData=True`.
    """
    return [
        _hist(1, 0, "ExecutionStarted", start),
        _hist(2, 1, "TaskStateEntered", start,
              stateEnteredEventDetails={"name": state_name}),
        _hist(3, 2, "TaskScheduled", start,
              taskScheduledEventDetails={
                  "resourceType": "states",
                  "resource": "startExecution.sync:2"}),
        _hist(4, 3, "TaskSubmitted", start,
              taskSubmittedEventDetails={
                  "resourceType": "states",
                  "output": json.dumps({"ExecutionArn": child_arn})}),
        _hist(5, 4, "TaskSucceeded", end),
        _hist(6, 5, "TaskStateExited", end,
              stateExitedEventDetails={"name": state_name}),
    ]


def _lambda_task_history(state_name, start=0, end=10):
    return [
        _hist(1, 0, "ExecutionStarted", start),
        _hist(2, 1, "TaskStateEntered", start,
              stateEnteredEventDetails={"name": state_name}),
        _hist(3, 2, "TaskScheduled", start,
              taskScheduledEventDetails={
                  "resource": "invoke",
                  "parameters": json.dumps({"FunctionName": "arn:aws:lambda:x"})}),
        _hist(4, 3, "TaskSucceeded", end),
        _hist(5, 4, "TaskStateExited", end,
              stateExitedEventDetails={"name": state_name}),
    ]


class TestUsageTrackingWalksTheTree(unittest.TestCase):
    """
    §8's one cost to the cost report: a repaired batch run is orchestrator → batch
    child → sync GRANDCHILD, and the grandchild is where the repair's Converse calls
    and its lease wait are billed — the most expensive part of exactly the run that
    went worst.

    The descendants are DISCOVERED from history rather than passed in. A scalar
    `child_execution_arn` parameter used to be threaded from the orchestrator's
    `ResultSelector`; it could name only one execution, and it named the one edge the
    discovery already covers.
    """

    def setUp(self):
        self.saved = (usage_tracking._SFN, usage_tracking._LAMBDA)
        usage_tracking._LAMBDA = _FakeLambdaConfig()

    def tearDown(self):
        usage_tracking._SFN, usage_tracking._LAMBDA = self.saved

    def sfn(self, histories):
        fake = _FakeSfnHistories(histories)
        usage_tracking._SFN = fake
        return fake

    def test_the_walk_reaches_the_repair_grandchild(self):
        self.sfn({
            _TREE_ROOT: _child_task_history("Run batch inference", _TREE_BATCH,
                                            end=50),
            _TREE_BATCH: _child_task_history("Run repair inference", _TREE_REPAIR,
                                             start=1, end=40),
            _TREE_REPAIR: _lambda_task_history("Run inference", start=2, end=30),
        })
        states, wall = usage_tracking._execution_cost(_TREE_ROOT)
        children = usage_tracking._descendants(states)

        # Keyed by execution NAME: `-batch` vs `-repair` says which machine ran, where
        # two arns differ only in a uuid.
        self.assertEqual(set(children), {"wid-batch", "wid-repair"})
        self.assertEqual(children["wid-repair"]["execution_arn"], _TREE_REPAIR)
        # The grandchild's Lambda time is the figure the whole walk exists for.
        self.assertGreater(
            children["wid-repair"]["states"]["Run inference"]["estimated_cost"], 0)
        # The `.sync:2` spans themselves are reported and billed ZERO — pricing them
        # would double-count the child's own itemised history.
        self.assertEqual(states["Run batch inference"]["kind"], "child_execution")
        self.assertEqual(states["Run batch inference"]["estimated_cost"], 0.0)
        self.assertGreater(wall, 0)

    def test_the_child_wall_is_the_longest_span_not_the_sum(self):
        """The repair grandchild's span is CONTAINED in the batch child's, so summing
        would report more wall clock than the run actually took."""
        self.sfn({
            _TREE_ROOT: _child_task_history("Run batch inference", _TREE_BATCH,
                                            end=50),
            _TREE_BATCH: _child_task_history("Run repair inference", _TREE_REPAIR,
                                             start=1, end=40),
            _TREE_REPAIR: _lambda_task_history("Run inference", start=2, end=30),
        })
        usage = usage_tracking._build_usage({}, _TREE_ROOT)
        walls = [child["wall_sec"] for child in
                 usage["states"]["children"].values()]
        self.assertEqual(usage["child_wall_sec"], max(walls))
        self.assertLess(usage["child_wall_sec"], sum(walls))
        # A list, because a repaired run has two descendants.
        self.assertEqual(sorted(usage["child_execution_arns"]),
                         sorted([_TREE_BATCH, _TREE_REPAIR]))
        self.assertEqual(usage["child_lambda_cost"],
                         round(sum(s["estimated_cost"]
                                   for c in usage["states"]["children"].values()
                                   for s in c["states"].values()), 6))

    def test_a_caught_failure_is_not_reported_as_a_retry(self):
        """
        The exact shape the live run produced, and the reason `retried` counts
        re-entries rather than failures now.

        `Collect batch output` raises `BatchRecordsIncompleteException` on purpose and
        the `Catch` routes forward, so the state ran EXACTLY ONCE and was never
        retried. Counting failed attempts reported `retried: 1` — a phantom flake on
        every repaired batch run, in the one report a reader consults to find flaky
        states. Verified live 2026-08-12: the deployed alpha.20260812a reported
        `invocations: 1, retried: 1` here before this changed.
        """
        events = [
            _hist(1, 0, "ExecutionStarted", 0),
            _hist(2, 1, "TaskStateEntered", 0,
                  stateEnteredEventDetails={"name": "Collect batch output"}),
            _hist(3, 2, "TaskScheduled", 0,
                  taskScheduledEventDetails={
                      "resource": "invoke",
                      "parameters": json.dumps({"FunctionName": "arn:aws:lambda:x"})}),
            _hist(4, 3, "TaskFailed", 2,
                  taskFailedEventDetails={
                      "error": "BatchRecordsIncompleteException"}),
            _hist(5, 4, "TaskStateExited", 2,
                  stateExitedEventDetails={"name": "Collect batch output"}),
        ]
        self.sfn({_TREE_ROOT: events})
        states, _wall = usage_tracking._execution_cost(_TREE_ROOT)
        collect = states["Collect batch output"]
        self.assertEqual(collect["invocations"], 1)
        self.assertEqual(collect["retried"], 0)
        # A caught failure is a completed attempt, not a missing one — `incomplete`
        # marks a state with NO terminal event, which would drop it from the bill.
        self.assertFalse(collect["incomplete"])
        # It still BILLS: the invocation ran, read the whole batch output and wrote
        # response.jsonl before it raised. Charging zero for it would under-report
        # exactly the runs that did the most work.
        self.assertGreater(collect["estimated_cost"], 0)

    def test_a_genuine_retry_ladder_still_counts_its_re_entries(self):
        """The other half: `retried` must not become uniformly zero. Three attempts,
        two of them failed, is two re-entries — not three, which is what counting
        failures reported when a ladder was exhausted."""
        events = [_hist(1, 0, "ExecutionStarted", 0),
                  _hist(2, 1, "TaskStateEntered", 0,
                        stateEnteredEventDetails={"name": "Run inference"})]
        eid = 3
        for attempt, second in enumerate((0, 5, 10)):
            events.append(_hist(
                eid, 2, "TaskScheduled", second,
                taskScheduledEventDetails={
                    "resource": "invoke",
                    "parameters": json.dumps({"FunctionName": "arn:aws:lambda:x"})}))
            terminal = "TaskSucceeded" if attempt == 2 else "TaskFailed"
            events.append(_hist(eid + 1, eid, terminal, second + 4))
            eid += 2
        events.append(_hist(eid, eid - 1, "TaskStateExited", 14,
                            stateExitedEventDetails={"name": "Run inference"}))
        self.sfn({_TREE_ROOT: events})
        states, _wall = usage_tracking._execution_cost(_TREE_ROOT)
        self.assertEqual(states["Run inference"]["invocations"], 3)
        self.assertEqual(states["Run inference"]["retried"], 2)

    def test_execution_data_is_requested_because_it_IS_the_discovery(self):
        """
        `includeExecutionData=True` is not an extravagance here: `ExecutionArn` lives in
        the `TaskSubmitted` event's `output`, so without it the walk finds no children
        at all and silently reports the orchestrator's ~4 seconds as the whole run.
        """
        fake = self.sfn({_TREE_ROOT: _child_task_history(
            "Run batch inference", _TREE_BATCH)})
        usage_tracking._execution_cost(_TREE_ROOT)
        self.assertTrue(all(call["includeExecutionData"] for call in fake.calls))

    def test_a_resubmitted_child_is_billed_once(self):
        """
        An ASL `Retry` on a `.sync:2` state re-submits under the SAME execution name, so
        two attempts report one arn — and billing it twice would double the most
        expensive part of the report.
        """
        events = _child_task_history("Run batch inference", _TREE_BATCH, end=50)
        events += [
            _hist(7, 2, "TaskScheduled", 51,
                  taskScheduledEventDetails={
                      "resourceType": "states",
                      "resource": "startExecution.sync:2"}),
            _hist(8, 7, "TaskSubmitted", 51,
                  taskSubmittedEventDetails={
                      "resourceType": "states",
                      "output": json.dumps({"ExecutionArn": _TREE_BATCH})}),
            _hist(9, 8, "TaskSucceeded", 55),
        ]
        self.sfn({_TREE_ROOT: events,
                  _TREE_BATCH: _lambda_task_history("Collect batch output")})
        states, _wall = usage_tracking._execution_cost(_TREE_ROOT)
        self.assertEqual(states["Run batch inference"]["child_execution_arns"],
                         [_TREE_BATCH])
        self.assertEqual(
            list(usage_tracking._descendants(states)), ["wid-batch"])

    def test_a_cycle_cannot_spin_the_walk(self):
        """
        The start graph is acyclic by construction and a template test pins it, but this
        walk is the one place a future cycle would cost 15 minutes of Lambda and a
        truncated report instead of raising.
        """
        self.sfn({
            _TREE_ROOT: _child_task_history("Run batch inference", _TREE_BATCH),
            # The child names the ROOT back — impossible under the current IAM, which is
            # the point: the guard must not depend on IAM staying that way.
            _TREE_BATCH: _child_task_history("Run repair inference", _TREE_ROOT),
        })
        states, _wall = usage_tracking._execution_cost(_TREE_ROOT)
        children = usage_tracking._descendants(states)
        self.assertEqual(set(children), {"wid-batch", "wid"})
        self.assertLessEqual(len(children), usage_tracking._MAX_CHILD_EXECUTIONS)

    def test_a_malformed_submitted_payload_costs_a_child_not_the_run(self):
        """This module must never fail a finished run over a cost report."""
        events = _child_task_history("Run batch inference", _TREE_BATCH)
        events[3]["taskSubmittedEventDetails"]["output"] = "{not json"
        self.sfn({_TREE_ROOT: events})
        states, _wall = usage_tracking._execution_cost(_TREE_ROOT)
        self.assertNotIn("child_execution_arns", states["Run batch inference"])
        self.assertEqual(usage_tracking._descendants(states), {})

    def test_a_non_states_task_submitted_is_not_a_child_execution(self):
        """`resourceType` is the discriminator, not the state name or its Resource
        string — the field is set by Step Functions rather than by our ASL, and every
        `.sync` integration emits a TaskSubmitted."""
        self.assertEqual(usage_tracking._submitted_execution_arn({
            "taskSubmittedEventDetails": {
                "resourceType": "sqs",
                "output": json.dumps({"ExecutionArn": _TREE_BATCH})}}), "")

    def test_execution_cost_still_returns_exactly_two_values(self):
        """
        A CROSS-REPO contract, and the reason child arns ride inside the per-state slots
        rather than in a third return value: `micro-workflow-bundle`'s cost-parity suite
        imports this module and unpacks `states, _wall = mod._execution_cost(arn)`. A
        third element breaks a sibling repo's tests, not this one's.
        """
        self.sfn({_TREE_ROOT: _lambda_task_history("Validate & plan")})
        result = usage_tracking._execution_cost(_TREE_ROOT)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_the_handler_never_raises_and_takes_no_child_arn(self):
        """
        Terminal state: an exception here would fail an execution whose `response.jsonl`
        and `summary.json` are already written. A broken history therefore returns
        `complete: False` with the reason, and the run still succeeds.
        """
        class _Boom:
            def get_execution_history(self, **kwargs):
                raise RuntimeError("history unavailable")

        usage_tracking._SFN = _Boom()
        out = usage_tracking.lambda_handler(
            {"uuid": "u1", "data": {"mode": "batch"},
             "execution_arn": _TREE_ROOT}, None)
        self.assertEqual(out["status"], "COMPLETED")
        self.assertFalse(out["data"]["usage_tracking"]["complete"])
        self.assertIn("history unavailable",
                      out["data"]["usage_tracking"]["reason"])
        # The scalar parameter is gone from the handler as well as from the ASL: a
        # `child_execution_arn` on the event is ignored entirely, so a leftover payload
        # field cannot quietly become the only edge that gets walked. Asserted by
        # BEHAVIOUR rather than by grepping the source, which still contains the plural
        # `child_execution_arns` the report is keyed on.
        self.sfn({_TREE_ROOT: _lambda_task_history("Validate & plan")})
        out = usage_tracking.lambda_handler(
            {"uuid": "u1", "data": {}, "execution_arn": _TREE_ROOT,
             "child_execution_arn": _TREE_BATCH}, None)
        self.assertEqual(out["data"]["usage_tracking"]["child_execution_arns"], [])


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 items 12 and 13 — mode parity and param-shaping parity
# ─────────────────────────────────────────────────────────────────────────────

class TestModeParity(_WorkflowCase):
    """
    A caller must not have to know which branch ran. Mode is excluded from
    `workflow_id` (§3), so the summary is the ONLY place the path is reported — and
    that only helps if everything else is the same shape.
    """

    LINE_KEYS = {"record_id", "status", "model_id", "output_text", "stop_reason",
                 "usage", "estimated_cost", "rounds"}

    def _sync_summary(self, records):
        plan = self.plan(records=records)
        result = self.run_sync(plan, _FakeRuntime())
        summarize.lambda_handler(result, None)
        return (self.s3.lines_at(PROXY, plan["data"]["keys"]["response"]),
                self.s3.json_at(PROXY, plan["data"]["keys"]["summary"]))

    def _batch_summary(self, records):
        plan = self.plan(request=_request(mode="batch"), records=records)
        prefix = f"{plan['data']['keys']['batch_output_prefix']}/{JOB_ID}"
        self.s3.put(PROXY, f"{prefix}/manifest.json.out", {"ok": True})
        self.s3.put(PROXY, f"{prefix}/input.jsonl.out", "".join(
            json.dumps({"recordId": record["record_id"],
                        "modelOutput": {
                            "output": {"message": {"content": [{"text": "ok"}]}},
                            "stopReason": "end_turn",
                            "usage": {"inputTokens": 10, "outputTokens": 5,
                                      "totalTokens": 15}}}) + "\n"
            for record in records))
        plan["batch_result"] = {"status": "COMPLETED", "job_arn": JOB_ARN}
        result = collect_batch.lambda_handler(plan, None)
        summarize.lambda_handler(result, None)
        return (self.s3.lines_at(PROXY, plan["data"]["keys"]["response"]),
                self.s3.json_at(PROXY, plan["data"]["keys"]["summary"]))

    def test_response_lines_have_the_same_shape_on_both_paths(self):
        records = _records(3)
        sync_lines, sync_summary = self._sync_summary(records)
        self.ddb.items.clear()
        batch_lines, batch_summary = self._batch_summary(records)

        for line in sync_lines + batch_lines:
            # `notes` is the one optional key: only the sync path can record a
            # fallback or a stripped temperature, and an empty list would read as a
            # note that was made.
            self.assertEqual(set(line) - {"notes"}, self.LINE_KEYS)
        self.assertEqual([line["record_id"] for line in sync_lines],
                         [line["record_id"] for line in batch_lines])
        self.assertEqual(set(sync_summary["bedrock_usage"]),
                         set(batch_summary["bedrock_usage"]))
        self.assertEqual(set(sync_summary) - {"records_resumed"},
                         set(batch_summary) - {"records_resumed"})
        self.assertEqual(sync_summary["inference_mode"], "sync")
        self.assertEqual(batch_summary["inference_mode"], "batch")


class TestMaxTokensIsRetired(unittest.TestCase):
    """max_tokens is GONE as a CALLER knob (2026-08-18, user decision: "one
    more parameter; one more maintenance cost — keep it simple and stupid").
    The knob had already been through two designs in one day — a flat 8192
    default (capped the 128k families at ~6%), then per-model registry
    resolution with clamping — before the simpler truth landed: never emit
    maxTokens and Bedrock's own default (the model's maximum output)
    applies. Probed live the day of removal: every reachable model answered
    normally with inferenceConfig fully omitted.

    AMENDED 2026-10-16 — the deliberate re-addition, as ENTRY DATA: the
    "default = model max" premise was REFUTED for the thinking-required 5
    family, whose omitted-field Converse default is an undocumented flat
    4096 (every 16-shot SME window truncated at exactly 4096; an explicit
    8192 → end_turn at 5,154). An entry may declare
    `capabilities.emit_max_tokens` and build_converse_params emits it
    verbatim. The CALLER-side retirement is unchanged — properties 2-4
    below — and property 1 narrows to "never from a caller, only from the
    entry":
      1. maxTokens appears in built params IFF the entry declares
         emit_max_tokens — sync and batch share the one shaping funnel;
      2. inferenceConfig itself is omitted when it has nothing to say,
         and carries ONLY temperature when only temperature applies;
      3. a stray caller max_tokens — request-level or per-record — is
         warned and ignored, never rejected, and never reaches the records
         digest, so two requests differing only in the retired field are
         ONE run;
      4. the batch PADS keep their internal maxTokens:1 — machinery
         overriding any entry value: 60-99 pad records per padded job must
         stay nearly free.
    """

    def _spec(self, **cfg):
        return bedrock_lib.ModelSpec(
            {"model_id": "test.model-v1:0", "inference_region": "global",
             **cfg})

    def test_max_tokens_never_appears_without_an_entry_declaration(self):
        params = bedrock_lib.build_converse_params(
            self._spec(), "sys",
            [{"role": "user", "content": [{"text": "hi"}]}])
        self.assertNotIn("maxTokens",
                         params.get("inferenceConfig") or {})

    def test_a_declaring_entry_emits_its_value_verbatim(self):
        # the 5-family shape: temperature rejected, thinking required —
        # inferenceConfig exists SOLELY to carry the entry's maxTokens
        spec = self._spec(capabilities={
            "temperature": False, "thinking": "required",
            "thinking_default": "adaptive", "emit_max_tokens": 16384,
            "effort": True, "effort_default": "low"})
        params = bedrock_lib.build_converse_params(
            spec, "sys", [{"role": "user", "content": [{"text": "hi"}]}])
        self.assertEqual(params["inferenceConfig"], {"maxTokens": 16384})
        # ...and beside temperature when both apply (a non-thinking entry)
        both = self._spec(capabilities={"temperature": True,
                                        "thinking": "absent",
                                        "emit_max_tokens": 9000})
        params = bedrock_lib.build_converse_params(
            both, "sys", [{"role": "user", "content": [{"text": "hi"}]}],
            temperature=0.3)
        self.assertEqual(params["inferenceConfig"],
                         {"temperature": 0.3, "maxTokens": 9000})

    def test_a_malformed_emit_max_tokens_is_a_finding_and_emits_nothing(self):
        for bad in (True, "16384", -5, 0):
            entry = {"model_id": "v.cap", "model_type": ["text"],
                     "capabilities": {"temperature": False,
                                      "thinking": "absent", "effort": False,
                                      "emit_max_tokens": bad}}
            findings = bedrock_lib.contract_findings(entry)
            self.assertTrue(any("emit_max_tokens" in f for f in findings),
                            (bad, findings))
            self.assertIsNone(
                bedrock_lib.ModelSpec(entry).emit_max_tokens(), bad)

    def test_temperature_recovery_keeps_the_entrys_max_tokens(self):
        """The recovery used to drop the WHOLE inferenceConfig block
        ("temperature is all it can carry") — with emit_max_tokens that
        would silently restore the 4096 truncation the entry declared the
        lift against, on exactly the retry nobody watches."""
        spec = self._spec(capabilities={"temperature": True,
                                        "thinking": "absent",
                                        "emit_max_tokens": 16384})
        # a real rejection names the parameter in the MESSAGE (the recovery
        # greps str(exc)) — _client_error's fixed message can't carry it
        err = ClientError(
            {"Error": {"Code": "ValidationException",
                       "Message": "temperature is not supported for this "
                                  "model"},
             "ResponseMetadata": {"HTTPStatusCode": 400}}, "Converse")
        runtime = _FakeRuntime(script=[err])
        result = bedrock_lib.converse(
            runtime, spec, None,
            [{"role": "user", "content": [{"text": "hi"}]}],
            temperature=0.0)
        self.assertEqual(len(runtime.calls), 2)
        retry_config = runtime.calls[1].get("inferenceConfig")
        self.assertEqual(retry_config, {"maxTokens": 16384})
        self.assertTrue(any("rejected `temperature`" in n
                            for n in result.notes), result.notes)

    def test_pads_override_a_declaring_entrys_value(self):
        spec = self._spec(capabilities={
            "temperature": False, "thinking": "required",
            "thinking_default": "adaptive", "emit_max_tokens": 16384,
            "effort": True, "effort_default": "low"})
        pad = bedrock_lib.padding_record(0, spec)["modelInput"]
        self.assertEqual(pad["inferenceConfig"]["maxTokens"], 1)

    def test_the_four_truncating_5_family_entries_declare_the_lift(self):
        """Pinned against the MASTER: the four thinking-required entries
        measured truncating at Bedrock's 4096 default (fable-5-1, opus-5,
        sonnet-5-5, haiku-5-5 — all stopped max_tokens at exactly 4096 on a
        >4k ask, 2026-10-16) carry emit_max_tokens 16384; the measured-clean
        entries (sonnet-4-6, haiku-4-5, the GPT trio finished end_turn past
        4096) carry none."""
        import json as _json
        with open(os.path.join(ROOT, "models",
                               "availablemodels_master.json")) as f:
            master = {m["model_id"]: m for m in _json.load(f)["models"]}
        lifted = {"anthropic.claude-fable-5-1", "anthropic.claude-opus-5",
                  "anthropic.claude-sonnet-5-5",
                  "anthropic.claude-haiku-5-5"}
        for model_id, entry in master.items():
            caps = entry.get("capabilities") or {}
            if model_id in lifted:
                self.assertEqual(caps.get("emit_max_tokens"), 16384,
                                 model_id)
                self.assertEqual(bedrock_lib.contract_findings(entry), [],
                                 model_id)
            else:
                self.assertNotIn("emit_max_tokens", caps, model_id)

    def test_inference_config_is_omitted_when_it_has_nothing_to_say(self):
        spec = self._spec(capabilities={"temperature": False})
        params = bedrock_lib.build_converse_params(
            spec, "sys", [{"role": "user", "content": [{"text": "hi"}]}])
        self.assertNotIn("inferenceConfig", params)

    def test_inference_config_carries_only_temperature_when_present(self):
        params = bedrock_lib.build_converse_params(
            self._spec(), "sys",
            [{"role": "user", "content": [{"text": "hi"}]}],
            temperature=0.3)
        self.assertEqual(params.get("inferenceConfig"),
                         {"temperature": 0.3})

    def test_stray_per_record_max_tokens_is_ignored_and_undigested(self):
        base = [{"record_id": "a",
                 "messages": [{"role": "user", "content": [{"text": "x"}]}]}]
        with_stray = [dict(base[0], max_tokens=64)]
        parsed_a = input_schema.parse_records(base)
        parsed_b = input_schema.parse_records(with_stray)
        self.assertNotIn("max_tokens", parsed_b[0])
        self.assertEqual(identity.digest_records(parsed_a),
                         identity.digest_records(parsed_b))

    def test_request_level_max_tokens_is_gone_from_config(self):
        config = input_schema.parse_request({
            "model_chain": ["m"], "tool_version": "t",
            "options": {"max_tokens": 4096},
            "output": {"bucket": "b", "prefix": "p"}})
        self.assertNotIn("max_tokens", config)

    def test_the_registry_carries_no_output_ceilings(self):
        """The max_output_tokens column lived for one day (its probe data —
        128k for the Claude families vs the old flat 8192 — is preserved in
        README.md as the evidence that killed the flat default). A model
        entry regrowing the field means someone is rebuilding the resolved
        design — read the removal decision first."""
        import json as _json
        with open(os.path.join(ROOT, "models",
                               "availablemodels_master.json")) as f:
            models = _json.load(f)["models"]
        for entry in models:
            self.assertNotIn("max_output_tokens", entry, entry["model_id"])

    def test_batch_pads_keep_their_internal_cap(self):
        padded, pad_count = _padded([])
        self.assertEqual(pad_count, batch_client.MIN_BATCH_RECORDS)
        for pad in padded:
            self.assertEqual(
                pad["modelInput"]["inferenceConfig"]["maxTokens"], 1)


class TestParamShapingParity(unittest.TestCase):
    """
    ONE shaping function, so there is no second code path to drift. `web-inference`
    has two, and its batch path skips the temperature range check and the effort
    setting its sync path applies.
    """

    def _params(self, model_id, **kw):
        spec = bedrock_lib.resolve_model(REGISTRY, model_id)
        base = {"temperature": 0.5, "thinking": "disabled",
                "effort": None}
        base.update(kw)
        return spec, bedrock_lib.build_converse_params(
            spec, "system text", [{"role": "user", "content": [{"text": "hi"}]}],
            **base)

    def test_batch_record_keeps_the_sync_inference_config_and_extras(self):
        """
        Batch must not silently drop a budget or a reasoning setting the sync path
        applies — the same model and the same prompt would then behave differently
        by path. One dialect since §5.4 resolved: the Converse request passes
        through verbatim, `additionalModelRequestFields` envelope and all.
        """
        for model_id in (PRIMARY, FALLBACK):
            with self.subTest(model_id=model_id):
                spec, params = self._params(model_id, thinking="adaptive",
                                            effort="high")
                record = bedrock_lib.to_batch_record("r1", params)
                body = record["modelInput"]
                self.assertEqual(record["recordId"], "r1")
                self.assertNotIn("modelId", body)
                self.assertEqual(body.get("inferenceConfig"),
                                 params.get("inferenceConfig"))
                self.assertEqual(
                    body.get("additionalModelRequestFields") or {},
                    params.get("additionalModelRequestFields") or {})

    def test_the_batch_record_is_the_converse_request_verbatim(self):
        """
        §5.4 resolved: `modelInvocationType: "Converse"` is declared job-level, so
        `modelInput` IS the Converse request minus the job-level modelId — the
        SAME dialect for every family. The 2026-08-09 defect
        (`messages.0.content.0.type: Field required`, 0/100) was the missing job
        parameter, not the record shape; a Messages-API translation here would now
        be the bug it was thought to fix.
        """
        for model_id in (PRIMARY, FALLBACK):
            with self.subTest(model_id=model_id):
                _, params = self._params(model_id)
                body = bedrock_lib.to_batch_record("r1", params)["modelInput"]
                self.assertNotIn("modelId", body)
                self.assertEqual(body["messages"], params["messages"])
                self.assertEqual(body.get("inferenceConfig"),
                                 params.get("inferenceConfig"))
                self.assertNotIn("anthropic_version", body)
                self.assertNotIn("max_tokens", body)

    def test_the_batch_row_is_read_in_the_converse_dialect(self):
        """The response half of the same resolution: a Converse-typed job's
        modelOutput carries `stopReason`/`usage.inputTokens`, the exact keys the
        sync path reads — no translation layer left to go stale."""
        converse_row = {
            "output": {"message": {"content": [{"text": "Hi"}]}},
            "stopReason": "content_filtered",
            "usage": {"inputTokens": 7, "outputTokens": 1, "totalTokens": 8},
        }
        self.assertEqual(bedrock_lib.extract_text(converse_row), "Hi")
        self.assertIn(converse_row["stopReason"],
                      bedrock_lib.REFUSAL_STOP_REASONS)
        # The InvokeModel-dialect translators are GONE — reappearing means the
        # two-dialect era is creeping back in.
        self.assertFalse(hasattr(bedrock_lib, "normalize_batch_usage"))
        self.assertFalse(hasattr(bedrock_lib, "batch_stop_reason"))
        self.assertFalse(hasattr(bedrock_lib, "_to_anthropic_body"))

    def test_explicit_null_union_blocks_do_not_crash_or_leak_reasoning(self):
        """Opus 5 batch output writes each content block as an EXPLICIT-NULL
        UNION — every field present, unused ones null — so `"text" in block` is
        true on EVERY block, and the adaptive-thinking reasoning block carries
        `text: null`. That None reached `"".join` and failed the whole collect
        (live 2026-08-14, the matrix's only red, surfaced by the same change
        that put Opus 5 batch on adaptive thinking). Shape below is the live
        artifact's, abridged: the fix must skip null-text blocks WITHOUT
        skipping the real answer, and must still exclude reasoning content."""
        opus5_row = {"output": {"message": {"content": [
            {"image": None, "toolUse": None, "cachePoint": None, "text": None,
             "reasoningContent": {"reasoningText": {"text": "scratchpad"}}},
            {"image": None, "toolUse": None, "cachePoint": None,
             "reasoningContent": None, "text": "It's a flowchart."},
        ]}}}
        self.assertEqual(bedrock_lib.extract_text(opus5_row),
                         "It's a flowchart.")
        self.assertNotIn("scratchpad", bedrock_lib.extract_text(opus5_row))

    def test_every_model_rejects_an_assistant_prefill(self):
        """
        A REQUEST-SHAPE rule since 2026-08-11, not a capability: Claude rejects an
        assistant-role last message family-wide, Llama 4 does too, and Nova merely
        tolerates one — so no flag, no tuple, no per-model branch. Loud rather than
        v5's silent drop: a prefill-dependent prompt whose message vanishes fails
        downstream with no clue why.
        """
        for registry, model_id in ((PREFILL_REGISTRY, OPUS5),
                                   (PREFILL_REGISTRY, FABLE5),
                                   (REGISTRY, PRIMARY),
                                   (REGISTRY, FALLBACK)):
            with self.subTest(model_id=model_id):
                spec = bedrock_lib.resolve_model(registry, model_id)
                with self.assertRaises(bedrock_lib.BedrockError) as raised:
                    bedrock_lib.build_converse_params(
                        spec, "system text",
                        [{"role": "user", "content": [{"text": "hi"}]},
                         {"role": "assistant", "content": [{"text": "```json"}]}])
                self.assertIn("assistant-role", str(raised.exception))

    def test_the_prefill_rejection_applies_identically_to_batch(self):
        """
        Not a second rule: `to_batch_record` is reached through the exact same
        `build_converse_params` call the sync loop makes (§12.1's param-shaping
        parity), so there is no batch-specific code path that could forget the guard.
        """
        spec = bedrock_lib.resolve_model(PREFILL_REGISTRY, OPUS5)
        with self.assertRaises(bedrock_lib.BedrockError):
            bedrock_lib.build_converse_params(
                spec, None,
                [{"role": "user", "content": [{"text": "hi"}]},
                 {"role": "assistant", "content": [{"text": "```json"}]}],
                thinking="adaptive")

    def test_a_prefill_from_a_non_terminal_position_is_not_rejected(self):
        """Only the LAST message is a prefill; the same role earlier is ordinary
        conversation history and must not be flagged."""
        spec = bedrock_lib.resolve_model(PREFILL_REGISTRY, OPUS5)
        params = bedrock_lib.build_converse_params(
            spec, None,
            [{"role": "user", "content": [{"text": "hi"}]},
             {"role": "assistant", "content": [{"text": "earlier turn"}]},
             {"role": "user", "content": [{"text": "and then?"}]}])
        self.assertEqual(len(params["messages"]), 3)

    def test_a_prefill_spends_no_api_call_on_any_model_in_the_chain(self):
        """
        Under the request-shape rule a prefill is malformed for EVERY model, so
        the walk must burn ZERO API calls — the raise happens in
        build_converse_params, before the client is touched, on each chain
        member. The record fails with the assistant-role reason; the neighbours
        are untouched (per-record isolation as usual).
        """
        chain = bedrock_lib.build_chain(
            REGISTRY + PREFILL_REGISTRY, [OPUS5, FALLBACK])
        runtime = _FakeRuntime([_converse_response(text="never reached")])
        with self.assertRaises(bedrock_lib.BedrockError) as raised:
            bedrock_lib.converse_with_chain(
                runtime, chain, None,
                [{"role": "user", "content": [{"text": "hi"}]},
                 {"role": "assistant", "content": [{"text": "```json"}]}],
                sleep=lambda _: None)
        self.assertEqual(len(runtime.calls), 0)
        self.assertIn("assistant-role", str(raised.exception))
        self.assertEqual(raised.exception.rounds, 0,
                         "no call was made, so no round may be billed")

    def test_temperature_is_omitted_for_families_that_reject_it(self):
        """Since the max_tokens retirement, temperature is the ONLY thing
        inferenceConfig can carry — so withholding it withholds the block."""
        _, params = self._params(PRIMARY)
        self.assertNotIn("inferenceConfig", params)
        _, params = self._params(FALLBACK)
        self.assertEqual(params["inferenceConfig"], {"temperature": 0.5})

    def test_temperature_is_omitted_whenever_reasoning_is_on(self):
        """Bedrock rejects sampling params alongside reasoning, and shaping it in one
        place is what keeps the two paths identical."""
        _, params = self._params(PRIMARY, thinking="adaptive")
        self.assertNotIn("inferenceConfig", params)
        self.assertEqual(params["additionalModelRequestFields"]["thinking"],
                         {"type": "adaptive"})

    def test_budget_tokens_is_never_emitted_and_has_no_numeric_form(self):
        """Deprecated on 4.6 and 400s on 4.7+ (§9.1), so accepting a number would
        only let a caller build a request Bedrock rejects."""
        _, params = self._params(PRIMARY, thinking="adaptive")
        self.assertNotIn("budget_tokens", json.dumps(params))
        with self.assertRaises(ValidationException):
            input_schema.parse_request(
                _request(options={"thinking": 4096}))

    def test_effort_only_reaches_models_that_accept_it(self):
        _, params = self._params(PRIMARY, effort="low")
        self.assertEqual(params["additionalModelRequestFields"]["output_config"],
                         {"effort": "low"})
        _, params = self._params(FALLBACK, effort="low")
        self.assertNotIn("output_config",
                         params.get("additionalModelRequestFields") or {})

    def test_temperature_thinking_and_effort_are_best_effort_by_capability(self):
        """
        The 2026-08-14 review requirement, pinned in one place: the gateway does
        NOT cross-validate these three (deliberately — the workflow is the
        authority), so the shaper must apply each one exactly as far as the
        model's capability block allows and silently no further. A request
        naming all three must be shapeable for EVERY registry model — the same
        request falling back down a mixed chain must never become malformed at
        the boundary between a model that takes a knob and one that does not.
        """
        # FALLBACK (nova-pro): thinking absent, no effort → both dropped, and
        # the caller's temperature survives, because nothing excludes it there.
        _, params = self._params(FALLBACK, thinking="adaptive", effort="low",
                                 temperature=0.7)
        self.assertNotIn("additionalModelRequestFields", params)
        self.assertEqual(params["inferenceConfig"]["temperature"], 0.7)
        # PRIMARY (sonnet-4-6): all three accepted — but temperature and
        # reasoning are mutually exclusive AT THE MODEL, so reasoning wins and
        # temperature is withheld rather than sent to be rejected.
        _, params = self._params(PRIMARY, thinking="adaptive", effort="low",
                                 temperature=0.7)
        self.assertEqual(params["additionalModelRequestFields"]["thinking"],
                         {"type": "adaptive"})
        self.assertNotIn("inferenceConfig", params)

    def test_no_code_path_emits_a_cache_point_and_batch_strips_a_callers(self):
        """
        Prompt caching is GONE — `cache_system_prompt` was removed 2026-08-14 (the
        use cases never cache; Nova rejects an explicit system cachePoint; batch
        has no prompt cache at all). Two halves survive it: nothing this module
        shapes may carry a cachePoint, and `to_batch_record` still strips one a
        CALLER embedded in their own system list, because a rejected job is a
        24 h park before anyone notices.
        """
        spec, params = self._params(PRIMARY)
        self.assertNotIn("cachePoint", json.dumps(params))
        self.assertNotIn("cache_system_prompt", json.dumps(params))
        # Caller-embedded cachePoint blocks, arriving through the request bytes —
        # in system AND in message content, because Converse allows both and
        # batch rejects both:
        params["system"] = [{"text": "sys"}, {"cachePoint": {"type": "default"}}]
        params["messages"] = [
            {"role": "user", "content": [{"text": "hi"},
                                         {"cachePoint": {"type": "default"}}]}]
        record = bedrock_lib.to_batch_record("r1", params)
        self.assertNotIn("cachePoint", json.dumps(record))
        # Stripping the block must not take the rest with it: the cachePoint is
        # its own block, so dropping the whole list would silently unsystem-
        # prompt (or unprompt) every record.
        self.assertTrue(record["modelInput"]["system"])
        self.assertEqual(record["modelInput"]["messages"][0]["content"],
                         [{"text": "hi"}])

    def test_batch_model_id_is_not_the_runtime_inference_profile_id(self):
        spec = bedrock_lib.resolve_model(REGISTRY, PRIMARY)
        self.assertEqual(bedrock_lib.resolve_batch_model_id(spec), BATCH_MODEL)
        self.assertNotEqual(bedrock_lib.resolve_batch_model_id(spec), spec.model_id)
        unpriced = bedrock_lib.resolve_model(REGISTRY, "vendor.unpriced-v1:0")
        with self.assertRaises(ValidationException):
            bedrock_lib.resolve_batch_model_id(unpriced)

    def test_reasoning_content_is_not_part_of_the_answer(self):
        """A caller parsing XML or JSON out of `output_text` would otherwise find the
        model's scratchpad prepended to it."""
        response = {"output": {"message": {"content": [
            {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
            {"text": "the answer"}]}}}
        self.assertEqual(bedrock_lib.extract_text(response), "the answer")

    def test_a_rejected_temperature_is_recovered_once_and_recorded(self):
        """
        core-lib's base.js deletes temperature and topP before EVERY retry
        regardless of cause, so a run configured at 0.0 silently finishes at the
        model's default sampling and the log says only "RETRY AGAIN".
        """
        spec = bedrock_lib.resolve_model(REGISTRY, FALLBACK)
        error = ClientError(
            {"Error": {"Code": "ValidationException",
                       "Message": "temperature is not supported"}}, "Converse")
        runtime = _FakeRuntime([error, _converse_response()])
        result = bedrock_lib.converse(
            runtime, spec, "sys", [{"role": "user", "content": [{"text": "x"}]}],
            temperature=0.5, sleep=lambda _: None)
        self.assertEqual(len(runtime.calls), 2)
        # temperature is all inferenceConfig can carry now, so the recovery
        # drops the whole block — the probed-as-accepted omitted shape.
        self.assertNotIn("inferenceConfig", runtime.calls[1])
        self.assertTrue(any("rejected `temperature`" in note
                            for note in result.notes))

    def test_a_validation_error_without_a_config_reraises_cleanly(self):
        """The recovery guard must tolerate an ABSENT inferenceConfig (the
        normal shape since the retirement) — indexing it raised KeyError in
        place of the model's own error."""
        spec = bedrock_lib.resolve_model(REGISTRY, PRIMARY)
        error = ClientError(
            {"Error": {"Code": "ValidationException",
                       "Message": "temperature is not supported"}}, "Converse")
        runtime = _FakeRuntime([error])
        with self.assertRaises(ClientError):
            bedrock_lib.converse(
                runtime, spec, "sys",
                [{"role": "user", "content": [{"text": "x"}]}],
                sleep=lambda _: None)
        self.assertEqual(len(runtime.calls), 1)


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 item 14 — release on failure
# ─────────────────────────────────────────────────────────────────────────────

class TestReleaseLease(_WorkflowCase):
    def test_release_never_raises_even_when_the_semaphore_does(self):
        """
        A raise here would be caught by `Release lease on failure`'s own Catch and
        replace the original error. The slot is reclaimed by the TTL or the
        terminal-state rule either way, so reporting beats raising.
        """
        event = {"data": {"quota": {"service": "bedrock", "resource": PRIMARY}},
                 "execution_arn": "arn:aws:states:us-east-1:1:execution:sm:x"}
        with mock.patch.object(release_lease.semaphore_client, "release",
                               side_effect=RuntimeError("ddb down")), \
             self.assertLogs(release_lease.logger, level="ERROR") as logs:
            result = release_lease.lambda_handler(event, None)
        self.assertFalse(result["data"]["lease"]["released"])
        self.assertIn("RuntimeError", result["data"]["lease"]["detail"])
        self.assertTrue(any("RELEASE FAILED" in line for line in logs.output))

    def test_no_scope_is_not_an_error(self):
        """Reached when the Catch fires BEFORE Acquire — a ValidationException in
        Validate & plan, say. There is nothing to release."""
        result = release_lease.lambda_handler({"data": {}}, None)
        self.assertFalse(result["data"]["lease"]["released"])
        self.assertIn("no lease to release", result["data"]["lease"]["detail"])

    def test_a_successful_release_is_reported(self):
        event = {"data": {"quota": {"service": "bedrock", "resource": PRIMARY}},
                 "execution_arn": "arn:aws:states:us-east-1:1:execution:sm:x"}
        with mock.patch.object(release_lease.semaphore_client, "release",
                               return_value=True):
            result = release_lease.lambda_handler(event, None)
        self.assertTrue(result["data"]["lease"]["released"])

    def test_the_plan_survives_the_release_state(self):
        """`Release lease` is on the happy path, so dropping `data` here would strand
        Summarize with no keys to read."""
        plan = self.plan()
        event = dict(plan)
        event["execution_arn"] = "arn:aws:states:us-east-1:1:execution:sm:x"
        with mock.patch.object(release_lease.semaphore_client, "release",
                               return_value=True):
            result = release_lease.lambda_handler(event, None)
        self.assertEqual(result["data"]["keys"], plan["data"]["keys"])
        self.assertEqual(result["data"]["workflow_id"], plan["data"]["workflow_id"])


# ═════════════════════════════════════════════════════════════════════════════
# The semaphore — ported from service-quotas-on-aws (README.md §M5.1)
#
# These came across with the code when the concurrency cap moved in-stack. A cap
# bug is silent by construction: the workflow succeeds, Bedrock throttles, and
# nothing anywhere says "four holders were admitted into three slots".
#
# The ONE semantic change the merge makes is not testable here and is deliberately
# not asserted: the table name is now `${ResourcePrefix}-sync-infer-quotas`, so the
# cap is per prefix rather than a region-wide singleton (README.md §M2.3). The
# tests below are about the cap being correct, not about how many of them there are.
# ═════════════════════════════════════════════════════════════════════════════

class TestSemaphoreKeys(unittest.TestCase):
    def test_scope_shape(self):
        self.assertEqual(sem.semaphore_pk("bedrock", "sonnet"),
                         "semaphore#bedrock#sonnet")
        self.assertEqual(sem.pending_pk("bedrock", "sonnet"),
                         "pending#bedrock#sonnet")
        self.assertEqual(sem.lease_pk("arn:e1"), "lease#arn:e1")
        self.assertEqual(sem.holder_sk("arn:e1"), "holder#arn:e1")
        self.assertEqual(sem.scope_sk("bedrock", "sonnet"), "scope#bedrock#sonnet")

    def test_hash_in_either_half_is_rejected(self):
        """
        ("bedrock#us", "sonnet") and ("bedrock", "us#sonnet") would produce the SAME
        partition key and silently share one cap — the "two quota tables" failure
        shape, reproduced inside one table.
        """
        with self.assertRaises(sem.ValidationException):
            sem.semaphore_pk("bedrock#us", "sonnet")
        with self.assertRaises(sem.ValidationException):
            sem.semaphore_pk("bedrock", "us#sonnet")

    def test_empty_and_non_string_rejected(self):
        for bad in ("", "   ", None, 7):
            with self.assertRaises(sem.ValidationException):
                sem.semaphore_pk("bedrock", bad)
            with self.assertRaises(sem.ValidationException):
                sem.semaphore_pk(bad, "sonnet")

    def test_parse_semaphore_pk_round_trip(self):
        self.assertEqual(
            sem.parse_semaphore_pk(sem.semaphore_pk("bedrock", "sonnet")),
            ("bedrock", "sonnet"))

    def test_parse_semaphore_pk_rejects_foreign_and_malformed(self):
        for bad in ("pending#bedrock#sonnet", "semaphore#bedrock", "semaphore##x",
                    "semaphore#x#", "stream-dedup"):
            with self.assertRaises(sem.ValidationException):
                sem.parse_semaphore_pk(bad)

    def test_execution_arn_from_holder_sk(self):
        arn = "arn:aws:states:us-east-1:1:execution:sm:exec"
        self.assertEqual(sem.execution_arn_from_holder_sk(sem.holder_sk(arn)), arn)
        with self.assertRaises(sem.ValidationException):
            sem.execution_arn_from_holder_sk("counter")

    def test_lease_pk_requires_execution_arn(self):
        for bad in ("", None):
            with self.assertRaises(sem.ValidationException):
                sem.lease_pk(bad)
            with self.assertRaises(sem.ValidationException):
                sem.holder_sk(bad)

    def test_pending_lives_on_its_own_PARTITION_not_a_sort_key(self):
        """
        Pinned because an external observer got this wrong and it cost a fake finding:
        the burst harness probed the semaphore partition for `pending#`-prefixed SORT
        keys and reported "0 waiters queued" while 21 waiters were in fact parked. A
        probe that looks in the wrong partition returns an empty page, not an error.
        """
        self.assertTrue(sem.pending_pk("bedrock", "sonnet").startswith("pending#"))
        self.assertNotEqual(sem.pending_pk("bedrock", "sonnet"),
                            sem.semaphore_pk("bedrock", "sonnet"))
        for sk_builder in (sem.holder_sk("arn:e1"), sem.scope_sk("bedrock", "sonnet"),
                           sem.COUNTER_SK):
            self.assertFalse(sk_builder.startswith("pending#"))


class TestSemaphoreTableName(unittest.TestCase):
    def test_missing_env_raises(self):
        with mock.patch.dict(os.environ, {sem.TABLE_NAME_ENV: ""}):
            with self.assertRaises(sem.ValidationException):
                sem.table_name()

    def test_present(self):
        self.assertEqual(sem.table_name(), "test-sync-infer-quotas")

    def test_the_two_tables_are_named_by_two_different_env_vars(self):
        """
        Decision 2: two tables, `sync-infer-quotas` and `batch-infer-tokens`. One env
        var serving both would put a semaphore counter and a batch row in one
        keyspace, where `pending#...` and `batch#...` collide by prefix only.
        """
        self.assertNotEqual(sem.TABLE_NAME_ENV, token_table.TABLE_NAME_ENV)
        self.assertNotEqual(sem.table_name(), token_table.table_name())


class TestNormalizeMaxConcurrent(unittest.TestCase):
    def test_required_no_default(self):
        """
        A missing value must RAISE, never default. No single number is right for
        every model, so a stack-level default would be the wrong cap applied
        silently whenever the field was omitted.
        """
        with self.assertRaises(sem.ValidationException):
            sem.normalize_max_concurrent(None)

    def test_bool_rejected(self):
        # int(True) is 1: a caller passing `true` would silently get a cap of one.
        with self.assertRaises(sem.ValidationException):
            sem.normalize_max_concurrent(True)

    def test_below_one_rejected(self):
        for bad in (0, -1):
            with self.assertRaises(sem.ValidationException):
                sem.normalize_max_concurrent(bad)

    def test_non_numeric_rejected(self):
        with self.assertRaises(sem.ValidationException):
            sem.normalize_max_concurrent("three")

    def test_numeric_string_and_float_accepted(self):
        self.assertEqual(sem.normalize_max_concurrent("3"), 3)
        self.assertEqual(sem.normalize_max_concurrent(3.0), 3)

    def test_clamped_not_rejected(self):
        # A caller asking for more headroom than the deployment allows should still
        # run, at the ceiling — a different thing from omitting the field.
        self.assertEqual(sem.normalize_max_concurrent(10_000),
                         sem.MAX_HOLDERS_CEILING)


class TestAcquire(_SemaphoreCase):
    def test_fills_to_cap_then_saturates(self):
        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2),
                         "acquired")
        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e2", 2),
                         "acquired")
        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e3", 2), "")
        self.assertEqual(self.counter()["holders"]["N"], "2")

    def test_reentrant_does_not_double_count(self):
        """An ASL/Lambda retry after a successful acquire must not consume a second
        slot — and must be reported as held, not as saturated."""
        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2),
                         "acquired")
        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2),
                         "reentrant")
        self.assertEqual(self.counter()["holders"]["N"], "1")

    def test_reentrant_reported_even_when_saturated(self):
        """
        The single most valuable test in the ported set, and the regression test for
        `try_acquire` reading `CancellationReasons[1]` ALONE.

        The holder's own slot is what makes the counter full, so a re-entrant acquire
        at the cap fails BOTH conditions — the counter guard and the holder Put.
        Requiring reason 0 to be "None" as well would report that as saturated and
        park a live holder on a pending item, deadlocked behind its own slot until
        its Acquire state timed out 24 h later.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 1)
        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 1),
                         "reentrant")

    def test_scopes_are_independent(self):
        """Sonnet saturating does not imply Haiku is saturated — the cap is per
        model, which is why `quota.resource` is the PRIMARY model id."""
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 1)
        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e2", 1), "")
        self.assertEqual(sem.try_acquire("bedrock", "haiku", "arn:e2", 1),
                         "acquired")

    def test_writes_lease_reverse_pointer(self):
        """
        The EventBridge terminal handler receives only an executionArn and the scope
        is part of the partition key, so without this pointer that trigger silently
        degrades to TTL expiry, up to an hour later.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.assertEqual(sem.list_leases("arn:e1"), [("bedrock", "sonnet")])

    def test_lease_pointer_covers_every_held_scope(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.try_acquire("bedrock", "haiku", "arn:e1", 2)
        self.assertEqual(sorted(sem.list_leases("arn:e1")),
                         [("bedrock", "haiku"), ("bedrock", "sonnet")])

    def test_list_leases_empty_for_execution_that_never_acquired(self):
        """The common case: the terminal rule fires on all three machines, and only
        the sync child ever holds a lease."""
        self.assertEqual(sem.list_leases("arn:never"), [])

    def test_list_leases_pages(self):
        self.quota.page_size = 1
        for res in ("a", "b", "c"):
            sem.try_acquire("bedrock", res, "arn:e1", 5)
        self.assertEqual(sorted(r for _, r in sem.list_leases("arn:e1")),
                         ["a", "b", "c"])

    def test_client_error_that_is_not_a_cancellation_propagates(self):
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("ProvisionedThroughputExceededException",
                                      "TransactWriteItems"))
        with self.assertRaises(ClientError):
            sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)

    def test_the_counter_and_the_holder_row_move_in_ONE_transaction(self):
        """
        The invariant that makes "counter behind its holder rows" impossible by
        construction, and therefore the one an external observer can rely on: a holder
        row can never exist that the counter is not already counting. Split into two
        calls, a crash between them leaves an UNCOUNTED holder and the cap is exceeded
        by one for the rest of that TTL.

        Asserted on the transaction's own contents rather than on end state, because
        end state cannot tell one transaction from two sequential writes — which is
        exactly the refactor this is guarding against.
        """
        captured = []
        real = self.quota.transact_write_items
        self.quota.transact_write_items = lambda TransactItems, **kw: (
            captured.append(TransactItems) or real(TransactItems=TransactItems, **kw))

        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2), "acquired")
        self.assertEqual(len(captured), 1, "the acquire must be ONE transaction")
        ops = captured[0]
        counter_updates = [
            op["Update"] for op in ops
            if "Update" in op and op["Update"]["Key"]["sk"]["S"] == sem.COUNTER_SK]
        holder_puts = [
            op["Put"] for op in ops
            if "Put" in op
            and op["Put"]["Item"]["sk"]["S"] == sem.holder_sk("arn:e1")]
        self.assertEqual(len(counter_updates), 1)
        self.assertEqual(len(holder_puts), 1)
        # And the counter guard is what bounds the cap, so it travels with them.
        self.assertIn("max_holders", counter_updates[0]["ConditionExpression"])


class TestMaxHoldersRatchet(_SemaphoreCase):
    def test_up_only(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 5)
        self.assertEqual(self.counter()["max_holders"]["N"], "5")
        sem.try_acquire("bedrock", "sonnet", "arn:e2", 2)   # must not lower it
        self.assertEqual(self.counter()["max_holders"]["N"], "5")
        sem.try_acquire("bedrock", "sonnet", "arn:e3", 8)   # may raise it
        self.assertEqual(self.counter()["max_holders"]["N"], "8")

    def test_admission_bound_by_live_cap_not_the_requester(self):
        """A later caller's smaller request does not throttle everyone else: it is
        admitted up to the RAISED cap, not its own lower number."""
        self.assertTrue(sem.try_acquire("bedrock", "sonnet", "arn:e1", 3))
        self.assertTrue(sem.try_acquire("bedrock", "sonnet", "arn:e2", 1))
        self.assertTrue(sem.try_acquire("bedrock", "sonnet", "arn:e3", 1))
        self.assertFalse(sem.try_acquire("bedrock", "sonnet", "arn:e4", 1))

    def test_ceiling_applies_to_the_stored_cap(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 10_000)
        self.assertEqual(int(self.counter()["max_holders"]["N"]),
                         sem.MAX_HOLDERS_CEILING)

    def test_ratchet_ignores_only_conditional_failures(self):
        self.quota.update_item = mock.Mock(
            side_effect=_client_error("AccessDeniedException", "UpdateItem"))
        with self.assertRaises(ClientError):
            sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)

    def test_a_satisfied_cap_issues_no_write_at_all(self):
        """
        The read-before-write, and the reason it exists. Every execution in a burst
        asks for the SAME max_concurrent, so the first ratchets and the rest have
        nothing to do — but an unconditional write made all of them contend on the
        single hottest item in the table to change nothing. That contention is what
        killed 11 of 24 burst executions.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 3)
        before = self.quota.calls.count("update_item")
        sem.try_acquire("bedrock", "sonnet", "arn:e2", 3)      # same cap: no-op
        self.assertEqual(self.quota.calls.count("update_item"), before)
        self.assertEqual(self.counter()["max_holders"]["N"], "3")

    def test_a_failed_pre_read_still_ratchets(self):
        """The read is an optimisation; the guarded write is the correctness."""
        self.quota.get_item = mock.Mock(
            side_effect=_client_error("InternalServerError", "GetItem"))
        self.assertTrue(sem.try_acquire("bedrock", "sonnet", "arn:e1", 4))
        self.assertEqual(self.counter()["max_holders"]["N"], "4")

    def test_exhausted_ratchet_conflict_does_not_fail_the_acquire(self):
        """
        The regression this whole change exists for. A ratchet that cannot get its
        write in must not take the acquire down with it — contention here means other
        callers are writing the same item, and they all want the same cap.
        """
        self.quota.update_item = mock.Mock(
            side_effect=_client_error("TransactionConflictException", "UpdateItem"))
        with mock.patch.object(sem, "_sleep"):
            self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2),
                             "acquired")
        self.assertIsNotNone(self.holder("arn:e1"))


class TestTransactionConflict(_SemaphoreCase):
    """
    Contention on the counter, which under a burst is the STEADY STATE of a working
    semaphore rather than an anomaly: every writer in the client contends on one item,
    `semaphore#{service}#{resource} / counter`.

    Two things make this its own test class rather than a line in each writer's:

    - **botocore does not retry it.** Verified in `botocore/data/_retry.json`: the
      dynamodb policy retries `ReplicatedWriteConflictException` and
      `TransactionInProgressException` and the string `TransactionConflict` appears
      nowhere in botocore. The ASL `Acquire lease` Retry names only Lambda-level
      errors, so nothing else absorbs it either — it reached the caller and failed
      11 of 24 burst executions.
    - **A conflicted transaction and a legitimately-refused one are the SAME
      exception class**, differing only in `CancellationReasons[i].Code`. Every
      caller here reads a cancellation as a meaningful answer ("saturated", "no
      holder", "already released", "duplicate record"), so a misread conflict is a
      WRONG ANSWER, not a retry.
    """

    def _conflicts_then(self, method_name, count, reasons=None):
        """
        Raise a conflict `count` times, then delegate to the real double.

        Two shapes, because DynamoDB has two: a top-level
        `TransactionConflictException` (what the burst actually produced) and a
        `TransactionCanceledException` whose reasons say `TransactionConflict`, which
        is the one that impersonates a condition failure.
        """
        real = getattr(self.quota, method_name)
        state = {"left": count}

        def wrapper(*a, **kw):
            if state["left"] > 0:
                state["left"] -= 1
                if reasons is None:
                    raise _client_error("TransactionConflictException", method_name)
                raise _client_error("TransactionCanceledException", method_name,
                                    reasons=reasons)
            return real(*a, **kw)

        setattr(self.quota, method_name, wrapper)
        return state

    def test_classifier_separates_contention_from_our_own_conditions(self):
        conflict = _client_error(
            "TransactionCanceledException", "TransactWriteItems",
            reasons=[{"Code": "None"}, {"Code": "TransactionConflict"}])
        refusal = _client_error(
            "TransactionCanceledException", "TransactWriteItems",
            reasons=[{"Code": "ConditionalCheckFailed"}, {"Code": "None"}])
        self.assertTrue(sem.is_transaction_conflict(conflict))
        self.assertFalse(sem.is_transaction_conflict(refusal))
        # A different exception class entirely, and a reason-less cancellation.
        self.assertFalse(sem.is_transaction_conflict(
            _client_error("ProvisionedThroughputExceededException", "UpdateItem")))
        self.assertFalse(sem.is_transaction_conflict(
            _client_error("TransactionCanceledException", "TransactWriteItems")))

    def test_acquire_retries_through_a_conflict(self):
        state = self._conflicts_then("transact_write_items", 3)
        with mock.patch.object(sem, "_sleep") as sleep:
            self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2),
                             "acquired")
        self.assertEqual(state["left"], 0)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(self.counter()["holders"]["N"], "1")

    def test_acquire_retries_a_conflict_disguised_as_a_cancellation(self):
        """The dangerous shape: read as an answer, this returns "" — "saturated" —
        and parks a waiter on a queue whose only drain trigger is another holder's
        release. With slots free, no release is coming."""
        self._conflicts_then(
            "transact_write_items", 2,
            reasons=[{"Code": "TransactionConflict"}, {"Code": "None"},
                     {"Code": "None"}])
        with mock.patch.object(sem, "_sleep"):
            self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2),
                             "acquired")

    def test_acquire_still_reports_saturated_on_a_real_condition_failure(self):
        """The retry must be narrow: the cap doing its job is not contention."""
        self.assertTrue(sem.try_acquire("bedrock", "sonnet", "arn:e1", 1))
        with mock.patch.object(sem, "_sleep") as sleep:
            self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e2", 1), "")
        sleep.assert_not_called()

    def test_acquire_still_reports_reentrant_on_a_real_condition_failure(self):
        self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2),
                         "acquired")
        with mock.patch.object(sem, "_sleep") as sleep:
            self.assertEqual(sem.try_acquire("bedrock", "sonnet", "arn:e1", 2),
                             "reentrant")
        sleep.assert_not_called()

    def test_exhausted_acquire_conflict_raises_rather_than_guessing(self):
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("TransactionConflictException",
                                      "TransactWriteItems"))
        with mock.patch.object(sem, "_sleep") as sleep:
            with self.assertRaises(ClientError):
                sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        # One sleep per retry, none after the final attempt.
        self.assertEqual(sleep.call_count, sem._CONFLICT_MAX_ATTEMPTS - 1)

    def test_exhausted_DISGUISED_conflicts_are_not_read_as_answers(self):
        """
        The shape that needs its own test, because the top-level
        `TransactionConflictException` cannot reach the code paths this covers: it is
        re-raised by the `!= "TransactionCanceledException"` guard before any
        classification happens. Only the DISGUISED shape — a cancellation whose
        reasons say `TransactionConflict` — arrives at the branch that has to choose
        between "an answer" and "the budget ran out", and for all three callers here
        an answer would be a lie: "saturated" while slots are free, "already released"
        while we still hold one, "duplicate record" for the only decrement it gets.
        """
        cancelled = _client_error(
            "TransactionCanceledException", "TransactWriteItems",
            reasons=[{"Code": "TransactionConflict"}, {"Code": "None"},
                     {"Code": "None"}])
        with mock.patch.object(sem, "_sleep"):
            self.quota.transact_write_items = mock.Mock(side_effect=cancelled)
            with self.assertRaises(ClientError):
                sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
            with self.assertRaises(ClientError):
                sem.release("bedrock", "sonnet", "arn:e1")
            with self.assertRaises(ClientError):
                sem.decrement_once("bedrock", "sonnet", "ev-1")

    def test_decrement_retries_rather_than_dropping_the_only_decrement(self):
        """
        The worst misread in the module. A holder-REMOVE record gets exactly ONE
        decrement; read as "duplicate record or zero counter" it is acked and lost,
        and the counter keeps a slot with no holder behind it until its own TTL an
        hour later — an effective cap permanently one lower, with nothing in any log.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self._conflicts_then("transact_write_items", 2)
        with mock.patch.object(sem, "_sleep"):
            self.assertTrue(sem.decrement_once("bedrock", "sonnet", "ev-1"))
        self.assertEqual(self.counter()["holders"]["N"], "0")

    def test_exhausted_decrement_conflict_raises_so_the_shard_retries(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("TransactionConflictException",
                                      "TransactWriteItems"))
        with mock.patch.object(sem, "_sleep"):
            with self.assertRaises(ClientError):
                sem.decrement_once("bedrock", "sonnet", "ev-1")

    def test_decrement_still_dedups_on_a_real_condition_failure(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.assertTrue(sem.decrement_once("bedrock", "sonnet", "ev-1"))
        with mock.patch.object(sem, "_sleep") as sleep:
            self.assertFalse(sem.decrement_once("bedrock", "sonnet", "ev-1"))
        sleep.assert_not_called()

    def test_release_retries_rather_than_reporting_already_released(self):
        """A conflict read as False returns "already released" to a caller that still
        holds the slot, and the holder delete that drives the stream decrement never
        happens — the slot survives to its TTL with release reporting success."""
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self._conflicts_then("transact_write_items", 2)
        with mock.patch.object(sem, "_sleep"):
            self.assertTrue(sem.release("bedrock", "sonnet", "arn:e1"))
        self.assertIsNone(self.holder("arn:e1"))

    def test_release_still_idempotent_on_a_real_condition_failure(self):
        with mock.patch.object(sem, "_sleep") as sleep:
            self.assertFalse(sem.release("bedrock", "sonnet", "arn:never-held"))
        sleep.assert_not_called()

    def test_exhausted_release_conflict_raises(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("TransactionConflictException",
                                      "TransactWriteItems"))
        with mock.patch.object(sem, "_sleep"):
            with self.assertRaises(ClientError):
                sem.release("bedrock", "sonnet", "arn:e1")

    def test_refresh_retries_rather_than_crying_holder_lost(self):
        """
        Asserts the TTL actually MOVED, not just that the call returned truthy: the
        exhaustion branch below also returns True, so a return-value-only assertion
        passes identically whether the retry ran or every attempt was abandoned.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        before = int(self.holder("arn:e1")["ttl"]["N"])
        self._conflicts_then("transact_write_items", 2)
        with mock.patch.object(sem, "_sleep"), \
                mock.patch.object(sem, "_now", return_value=sem._now() + 300):
            self.assertTrue(sem.refresh_holder("bedrock", "sonnet", "arn:e1"))
        self.assertEqual(int(self.holder("arn:e1")["ttl"]["N"]), before + 300)

    def test_exhausted_refresh_conflict_reports_the_hold_as_LIVE(self):
        """
        The one place exhaustion returns the truthy value. A conflict is no evidence
        the holder is gone: False triggers the caller's HOLDER LOST error path — a
        false alarm on the one signal meaning the cap is broken — and raising fails an
        execution that does legitimately hold its slot. A missed bump costs TTL
        headroom, of which there is an hour.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("TransactionConflictException",
                                      "TransactWriteItems"))
        with mock.patch.object(sem, "_sleep"):
            self.assertTrue(sem.refresh_holder("bedrock", "sonnet", "arn:e1"))

    def test_refresh_still_reports_a_genuinely_missing_holder(self):
        with mock.patch.object(sem, "_sleep") as sleep:
            self.assertFalse(sem.refresh_holder("bedrock", "sonnet", "arn:never"))
        sleep.assert_not_called()

    def test_an_exhausted_counter_ttl_bump_does_not_fail_the_refresh(self):
        """Best-effort, and less at stake than the rest: the holder refresh already
        succeeded, and every concurrent acquire bumps this same ttl."""
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.quota.update_item = mock.Mock(
            side_effect=_client_error("TransactionConflictException", "UpdateItem"))
        with mock.patch.object(sem, "_sleep"):
            self.assertTrue(sem.refresh_holder("bedrock", "sonnet", "arn:e1"))

    def test_the_counter_ttl_bump_is_itself_retried(self):
        """
        Asserts the ttl MOVED. Checking only the return value cannot see this write at
        all: the exhaustion branch swallows the conflict, so removing the retry
        entirely leaves every return value identical and the ttl simply never extended.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        counter_key = (sem.semaphore_pk("bedrock", "sonnet"), "counter")
        before = int(self.quota.items[counter_key]["ttl"]["N"])
        self._conflicts_then("update_item", 2)
        with mock.patch.object(sem, "_sleep"), \
                mock.patch.object(sem, "_now", return_value=sem._now() + 300):
            self.assertTrue(sem.refresh_holder("bedrock", "sonnet", "arn:e1"))
        self.assertEqual(int(self.quota.items[counter_key]["ttl"]["N"]),
                         before + 300)

    def test_a_non_conflict_counter_ttl_error_still_propagates(self):
        """The swallow is scoped to contention. A permissions or throughput error on
        the counter is a real problem and must not be hidden behind a warning."""
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.quota.update_item = mock.Mock(
            side_effect=_client_error("AccessDeniedException", "UpdateItem"))
        with self.assertRaises(ClientError):
            sem.refresh_holder("bedrock", "sonnet", "arn:e1")

    def test_backoff_is_jittered_and_grows(self):
        """A fixed backoff re-collides the same set of writers on every attempt —
        the exact failure mode the retry is meant to break up."""
        delays = []
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("TransactionConflictException",
                                      "TransactWriteItems"))
        with mock.patch.object(sem, "_sleep", side_effect=delays.append):
            with self.assertRaises(ClientError):
                sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.assertEqual(len(delays), sem._CONFLICT_MAX_ATTEMPTS - 1)
        # Jitter is a 0.5–1.5 band on an exponential base, so consecutive delays can
        # overlap; the floors and ceilings cannot.
        for attempt, delay in enumerate(delays):
            base = sem._CONFLICT_BASE_SLEEP_SEC * (2 ** attempt)
            self.assertGreaterEqual(delay, base * 0.5)
            self.assertLessEqual(delay, base * 1.5)
        self.assertNotEqual(len(set(delays)), 1)

    def test_a_non_conflict_error_is_never_retried(self):
        """Retrying a throttle here would be harmless; retrying a validation or
        access error burns the budget on something that cannot succeed."""
        for code in ("AccessDeniedException", "ValidationException",
                     "ResourceNotFoundException"):
            with self.subTest(code=code):
                self.quota.transact_write_items = mock.Mock(
                    side_effect=_client_error(code, "TransactWriteItems"))
                with mock.patch.object(sem, "_sleep") as sleep:
                    with self.assertRaises(ClientError):
                        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
                sleep.assert_not_called()


class TestSemaphoreRelease(_SemaphoreCase):
    def test_deletes_holder_and_pointer_and_is_idempotent(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.assertTrue(sem.release("bedrock", "sonnet", "arn:e1"))
        self.assertIsNone(self.holder("arn:e1"))
        self.assertEqual(sem.list_leases("arn:e1"), [])
        self.assertFalse(sem.release("bedrock", "sonnet", "arn:e1"))

    def test_does_not_decrement_the_counter(self):
        """
        Semaphore invariant 1. The stream owns the decrement; that is what makes all
        THREE remaining disappearance paths converge on ONE decrement and ONE drain.
        A release that also decremented would double-count against the stream record
        it produces.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.release("bedrock", "sonnet", "arn:e1")
        self.assertEqual(self.counter()["holders"]["N"], "1")

    def test_unexpected_client_error_propagates(self):
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("InternalServerError", "TransactWriteItems"))
        with self.assertRaises(ClientError):
            sem.release("bedrock", "sonnet", "arn:e1")


class TestRefreshHolder(_SemaphoreCase):
    def test_refreshes_holder_pointer_and_counter(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        holder_before = self.holder("arn:e1")["ttl"]["N"]
        counter_key = (sem.semaphore_pk("bedrock", "sonnet"), "counter")
        self.quota.items[counter_key].pop("ttl")     # simulate a stale counter ttl
        with mock.patch.object(sem, "_now", return_value=sem._now() + 100):
            self.assertTrue(sem.refresh_holder("bedrock", "sonnet", "arn:e1"))
        self.assertGreater(int(self.holder("arn:e1")["ttl"]["N"]),
                           int(holder_before))
        self.assertIn("ttl", self.quota.items[counter_key])

    def test_pointer_ttl_tracks_the_holder_ttl(self):
        """An expired pointer with a live holder blinds the terminal handler; a
        pointer that outlives the holder makes cleanup chase a scope it no longer
        holds."""
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        with mock.patch.object(sem, "_now", return_value=sem._now() + 100):
            sem.refresh_holder("bedrock", "sonnet", "arn:e1")
        holder = self.holder("arn:e1")
        pointer = self.quota.items[(sem.lease_pk("arn:e1"),
                                    sem.scope_sk("bedrock", "sonnet"))]
        self.assertEqual(holder["ttl"]["N"], pointer["ttl"]["N"])

    def test_returns_false_when_holder_is_gone(self):
        self.assertFalse(sem.refresh_holder("bedrock", "sonnet", "arn:missing"))

    def test_zombie_cannot_resurrect_the_counter(self):
        """
        Semaphore invariant 5. An execution whose holder was already TTL-reclaimed
        must not extend the counter's life — the counter TTL is the self-heal for a
        permanently-lost stream decrement, and a zombie refreshing it removes the
        only path back to zero.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        counter_key = (sem.semaphore_pk("bedrock", "sonnet"), "counter")
        # TTL reclaim: DDB removes the holder, nothing else.
        del self.quota.items[(sem.semaphore_pk("bedrock", "sonnet"),
                              sem.holder_sk("arn:e1"))]
        self.quota.items[counter_key].pop("ttl")
        self.assertFalse(sem.refresh_holder("bedrock", "sonnet", "arn:e1"))
        self.assertNotIn("ttl", self.quota.items[counter_key])

    def test_ttl_bounds_a_state_not_a_span(self):
        """
        Semaphore invariant 3 is structural, not measured: HOLDER_TTL_SEC must exceed
        Lambda's 900 s hard ceiling by enough that no single invocation can outlive
        its own refresh, since `run_inference` refreshes ONCE on state entry and the
        checkpoint re-entry IS the state entry.
        """
        self.assertGreaterEqual(sem.HOLDER_TTL_SEC, 900 * 2)

    def test_unexpected_client_error_propagates(self):
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("InternalServerError", "TransactWriteItems"))
        with self.assertRaises(ClientError):
            sem.refresh_holder("bedrock", "sonnet", "arn:e1")


class TestSemaphoreObservation(_SemaphoreCase):
    def test_get_counter_and_get_holder(self):
        self.assertEqual(sem.get_counter("bedrock", "sonnet"), {})
        self.assertIsNone(sem.get_holder("bedrock", "sonnet", "arn:e1"))
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 3)
        self.assertEqual(sem.get_counter("bedrock", "sonnet"),
                         {"holders": 1, "max_holders": 3})
        self.assertIn("acquired_at", sem.get_holder("bedrock", "sonnet", "arn:e1"))


class TestDecrementOnce(_SemaphoreCase):
    def test_dedups_on_event_id(self):
        """Streams are at-least-once. A redelivered shard record that decremented
        again would inflate the effective cap permanently."""
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        self.assertTrue(sem.decrement_once("bedrock", "sonnet", "ev-1"))
        self.assertFalse(sem.decrement_once("bedrock", "sonnet", "ev-1"))
        self.assertEqual(self.counter()["holders"]["N"], "0")

    def test_distinct_event_ids_each_decrement(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.try_acquire("bedrock", "sonnet", "arn:e2", 2)
        self.assertTrue(sem.decrement_once("bedrock", "sonnet", "ev-1"))
        self.assertTrue(sem.decrement_once("bedrock", "sonnet", "ev-2"))
        self.assertEqual(self.counter()["holders"]["N"], "0")

    def test_never_goes_negative(self):
        """
        Legitimately reachable: a counter TTL reset (invariant 5) whose holders had
        not yet been reclaimed. Driving the counter negative would inflate the cap by
        the size of the deficit, with no path back.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.decrement_once("bedrock", "sonnet", "ev-1")
        self.assertFalse(sem.decrement_once("bedrock", "sonnet", "ev-2"))
        self.assertEqual(self.counter()["holders"]["N"], "0")

    def test_dedup_marker_is_scope_independent(self):
        """The marker is keyed on the stream's eventID, which is globally unique —
        one scope's redelivery must not mask another scope's real record."""
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.try_acquire("bedrock", "haiku", "arn:e1", 2)
        self.assertTrue(sem.decrement_once("bedrock", "sonnet", "ev-1"))
        self.assertTrue(sem.decrement_once("bedrock", "haiku", "ev-2"))
        self.assertEqual(self.counter("bedrock", "haiku")["holders"]["N"], "0")

    def test_event_id_required(self):
        with self.assertRaises(sem.ValidationException):
            sem.decrement_once("bedrock", "sonnet", "")

    def test_unexpected_client_error_propagates(self):
        self.quota.transact_write_items = mock.Mock(
            side_effect=_client_error("InternalServerError", "TransactWriteItems"))
        with self.assertRaises(ClientError):
            sem.decrement_once("bedrock", "sonnet", "ev-1")


class TestSemaphoreDrain(_SemaphoreCase):
    """
    The lease queue's drain. Distinct from `TestDrain` above, which is the BATCH
    queue's — two queues, two drains, and the only thing they share is the word.
    """

    def _pending_key(self, token, service="bedrock", resource="sonnet"):
        return (sem.pending_pk(service, resource),
                hashlib.sha256(token.encode()).hexdigest())

    def _order(self, token, created_at):
        self.quota.items[self._pending_key(token)]["created_at"] = {
            "N": str(created_at)}

    def test_fifo_and_stops_at_the_cap(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.try_acquire("bedrock", "sonnet", "arn:e2", 2)
        sem.store_pending("bedrock", "sonnet", "arn:e3", "token-3", 2)
        sem.store_pending("bedrock", "sonnet", "arn:e4", "token-4", 2)
        self._order("token-3", 1000)
        self._order("token-4", 2000)

        sem.release("bedrock", "sonnet", "arn:e1")
        sem.decrement_once("bedrock", "sonnet", "ev-1")

        sent = []
        self.assertEqual(sem.drain_pending("bedrock", "sonnet", sent.append), 1)
        self.assertEqual(sent, ["token-3"])            # oldest first
        self.assertIsNotNone(self.holder("arn:e3"))
        self.assertIsNone(self.holder("arn:e4"))

    def test_fifo_ordering_is_by_created_at_not_by_key(self):
        """
        The pending sort key is sha256(task_token) — arbitrary order. Draining in key
        order would admit waiters out of turn, which is invisible in aggregate and
        starves whichever token happens to hash high.
        """
        sem.store_pending("bedrock", "sonnet", "arn:old", "token-zzz", 1)
        sem.store_pending("bedrock", "sonnet", "arn:new", "token-aaa", 1)
        self._order("token-zzz", 1000)
        self._order("token-aaa", 5000)
        sent = []
        sem.drain_pending("bedrock", "sonnet", sent.append)
        self.assertEqual(sent, ["token-zzz"])

    def test_empty_queue_is_a_cheap_noop(self):
        self.assertEqual(sem.drain_pending("bedrock", "sonnet", mock.Mock()), 0)

    def test_admits_the_whole_queue_when_slots_allow(self):
        for index in range(3):
            sem.store_pending("bedrock", "sonnet", f"arn:e{index}",
                              f"token-{index}", 3)
            self._order(f"token-{index}", 1000 + index)
        sent = []
        self.assertEqual(sem.drain_pending("bedrock", "sonnet", sent.append), 3)
        self.assertEqual(sent, ["token-0", "token-1", "token-2"])
        self.assertEqual(self.counter()["holders"]["N"], "3")

    def test_dead_token_releases_the_slot_it_just_took(self):
        """A waiter that timed out or vanished must not keep the slot the drain
        claimed on its behalf — the pending item is gone, so nothing would ever
        release it."""
        sem.store_pending("bedrock", "sonnet", "arn:dead", "token-dead", 2)
        self._order("token-dead", 1000)

        def sender(_token):
            raise _client_error("TaskTimedOut", "SendTaskSuccess")

        self.assertEqual(sem.drain_pending("bedrock", "sonnet", sender), 1)
        self.assertIsNone(self.holder("arn:dead"))
        self.assertIsNone(self.quota.items.get(self._pending_key("token-dead")))

    def test_dead_token_on_a_REENTRANT_hold_does_not_release(self):
        """
        The critical case. A "reentrant" hold means a PRIOR drain already admitted
        this waiter and died before deleting the pending item — that execution is
        LIVE and owns the holder. Releasing it here frees a slot that is still in use
        and the cap is silently exceeded for the rest of that execution's span.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:live", 2)     # already admitted
        sem.store_pending("bedrock", "sonnet", "arn:live", "token-live", 2)
        self._order("token-live", 1000)

        def sender(_token):
            raise _client_error("TaskDoesNotExist", "SendTaskSuccess")

        sem.drain_pending("bedrock", "sonnet", sender)
        self.assertIsNotNone(self.holder("arn:live"))            # NOT released
        self.assertIsNone(self.quota.items.get(self._pending_key("token-live")))

    def test_transient_send_failure_keeps_the_pending_item_and_raises(self):
        """
        The waiter is still real. Deleting here would strand a live execution on its
        token until its Acquire state times out 24 h later; the stream retry re-runs
        the drain and admits it then.
        """
        sem.store_pending("bedrock", "sonnet", "arn:e1", "token-1", 2)
        self._order("token-1", 1000)

        def sender(_token):
            raise _client_error("ThrottlingException", "SendTaskSuccess")

        with self.assertRaises(ClientError):
            sem.drain_pending("bedrock", "sonnet", sender)
        self.assertIsNotNone(self.quota.items.get(self._pending_key("token-1")))

    def test_store_pending_requires_a_token(self):
        with self.assertRaises(sem.ValidationException):
            sem.store_pending("bedrock", "sonnet", "arn:e1", "", 2)

    def test_store_pending_ratchets_the_cap_before_the_drain_runs(self):
        """A queued request's max_concurrent must raise the cap while it waits, not
        only once a slot frees and try_acquire runs."""
        sem.store_pending("bedrock", "sonnet", "arn:e1", "token-1", 4)
        self.assertEqual(self.counter()["max_holders"]["N"], "4")

    def test_store_pending_is_token_idempotent(self):
        sem.store_pending("bedrock", "sonnet", "arn:e1", "token-1", 2)
        sem.store_pending("bedrock", "sonnet", "arn:e1", "token-1", 2)
        pending = [key for key in self.quota.items
                   if key[0] == sem.pending_pk("bedrock", "sonnet")]
        self.assertEqual(len(pending), 1)


class TestAcquireLeaseHandler(_SemaphoreCase):
    def setUp(self):
        super().setUp()
        self.sfn = mock.MagicMock()
        self._saved_sfn = acquire_lease._SFN
        acquire_lease._SFN = self.sfn

    def tearDown(self):
        acquire_lease._SFN = self._saved_sfn
        super().tearDown()

    @staticmethod
    def _event(**kw):
        base = {"operation": "StateAcquireLease", "service": "bedrock",
                "resource": "sonnet", "execution_arn": "arn:e1",
                "task_token": "token-1", "max_concurrent": 1}  # nosec B105
        base.update(kw)
        return base

    def test_free_slot_sends_task_success_immediately(self):
        out = acquire_lease.lambda_handler(self._event(), None)
        self.assertEqual(out, {"acquired": True, "mode": "acquired"})
        self.sfn.send_task_success.assert_called_once_with(
            taskToken="token-1", output="{}")

    def test_saturated_queues_instead_of_failing(self):
        acquire_lease.lambda_handler(self._event(), None)
        self.sfn.reset_mock()
        out = acquire_lease.lambda_handler(
            self._event(execution_arn="arn:e2", task_token="token-2"), None)  # nosec B106
        self.assertEqual(out, {"acquired": False, "queued": True})
        self.sfn.send_task_success.assert_not_called()

    def test_task_token_is_required(self):
        with self.assertRaises(sem.ValidationException):
            acquire_lease.lambda_handler(self._event(task_token=None), None)

    def test_max_concurrent_is_required(self):
        with self.assertRaises(sem.ValidationException):
            acquire_lease.lambda_handler(self._event(max_concurrent=None), None)

    def test_already_consumed_token_is_swallowed_on_a_retry(self):
        """A Lambda retry after a successful acquire lands in the re-entrant branch
        and re-sends. The duplicate send is not a failure — the execution already
        moved on."""
        acquire_lease.lambda_handler(self._event(), None)
        self.sfn.send_task_success.side_effect = _client_error(
            "TaskDoesNotExist", "SendTaskSuccess")
        out = acquire_lease.lambda_handler(self._event(), None)
        self.assertEqual(out, {"acquired": True, "mode": "reentrant"})

    def test_transient_send_failure_is_not_swallowed(self):
        self.sfn.send_task_success.side_effect = _client_error(
            "ThrottlingException", "SendTaskSuccess")
        with self.assertRaises(ClientError):
            acquire_lease.lambda_handler(self._event(), None)

    def test_the_handler_has_no_drain_operation(self):
        """
        The standalone stack routed `{"operation": "Drain"}` here so the decrement
        handler could hold no `SendTask*`. Both are on the shared image now, so
        `handle_semaphore_release` calls `drain_pending` directly and the async
        self-invoke plus its `ACQUIRE_FUNCTION_NAME` are gone — a hop that could fail
        silently, protecting nothing (README.md §M2.2).
        """
        source = _code(ROOT, "states", "acquire_lease", "lambda_function.py")
        self.assertNotIn("drain_pending", source)
        self.assertNotIn("ACQUIRE_FUNCTION_NAME", source)
        self.assertNotIn("InvocationType", source)


class TestSemaphoreStreamHandler(_SemaphoreCase):
    """`states/handle_semaphore_release` — the `sync-infer-quotas` stream consumer."""

    def setUp(self):
        super().setUp()
        self.sfn = mock.MagicMock()
        self._saved_sfn = acquire_lease._SFN
        acquire_lease._SFN = self.sfn

    def tearDown(self):
        acquire_lease._SFN = self._saved_sfn
        super().tearDown()

    @staticmethod
    def _record(pk, sk, event_id):
        return {"eventSource": "aws:dynamodb", "eventName": "REMOVE",
                "eventID": event_id,
                "dynamodb": {"Keys": {"pk": {"S": pk}, "sk": {"S": sk}}}}

    def _holder_record(self, execution_arn, event_id,
                       service="bedrock", resource="sonnet"):
        return self._record(sem.semaphore_pk(service, resource),
                            sem.holder_sk(execution_arn), event_id)

    def test_decrements_and_drains_once_per_scope(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 3)
        sem.try_acquire("bedrock", "sonnet", "arn:e2", 3)
        sem.try_acquire("bedrock", "haiku", "arn:e3", 3)
        with mock.patch.object(semaphore_client, "drain_pending",
                               return_value=0) as drained:
            out = handle_semaphore_release.lambda_handler({"Records": [
                self._holder_record("arn:e1", "ev-1"),
                self._holder_record("arn:e2", "ev-2"),
                self._holder_record("arn:e3", "ev-3", resource="haiku"),
            ]}, None)
        self.assertEqual(out, {"decremented": 3, "scopes_drained": 2,
                               "admitted": 0})
        self.assertEqual(sorted(call.args[:2] for call in drained.call_args_list),
                         [("bedrock", "haiku"), ("bedrock", "sonnet")])

    def test_the_drain_is_called_directly_with_the_one_sender(self):
        """
        The merge's one behavioural change here: no async self-invoke. The sender is
        `acquire_lease.send_task_success` — one place the token is redeemed, injected
        rather than rebuilt, which is what keeps `semaphore_client` free of a
        stepfunctions client.
        """
        sem.store_pending("bedrock", "sonnet", "arn:e9", "token-9", 2)
        out = handle_semaphore_release.lambda_handler(
            {"Records": [self._holder_record("arn:e9", "ev-1")]}, None)
        self.assertEqual(out["admitted"], 1)
        self.sfn.send_task_success.assert_called_once_with(
            taskToken="token-9", output="{}")

    def test_drains_even_when_every_decrement_dedups(self):
        """
        A redelivered batch: the decrements all dedup, but the crashed attempt may
        have died before draining at all. If the removed holder was the LAST one on
        that scope, no future REMOVE will ever fire, and skipping the drain would
        strand every waiter until its acquire task token times out 24 h later.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.decrement_once("bedrock", "sonnet", "ev-1")
        with mock.patch.object(semaphore_client, "drain_pending",
                               return_value=0) as drained:
            out = handle_semaphore_release.lambda_handler(
                {"Records": [self._holder_record("arn:e1", "ev-1")]}, None)
        self.assertEqual(out, {"decremented": 0, "scopes_drained": 1,
                               "admitted": 0})
        drained.assert_called_once()

    def test_unexpected_keys_are_logged_not_raised(self):
        """
        A raise fails the whole batch and the stream redelivers it forever. The
        record this guard really exists for is a `stream-dedup` marker's own TTL
        expiry, which arrives as a REMOVE and must never read as a released holder.
        """
        with mock.patch.object(semaphore_client, "drain_pending",
                               return_value=0) as drained:
            out = handle_semaphore_release.lambda_handler({"Records": [
                self._record(sem.STREAM_DEDUP_PK, "ev-old", "ev-9"),
                self._record(sem.semaphore_pk("bedrock", "sonnet"), "counter",
                             "ev-10"),
            ]}, None)
        self.assertEqual(out, {"decremented": 0, "scopes_drained": 0,
                               "admitted": 0})
        drained.assert_not_called()


class TestExecutionTerminalHandler(_SemaphoreCase):
    """`states/handle_execution_terminal` — trigger 3, the EventBridge rule on all
    three machines' terminal states."""

    @staticmethod
    def _event(execution_arn, status="ABORTED"):
        return {"source": "aws.states",
                "detail-type": "Step Functions Execution Status Change",
                "detail": {"executionArn": execution_arn, "status": status}}

    def test_releases_every_lease_the_execution_still_holds(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.try_acquire("bedrock", "haiku", "arn:e1", 2)
        out = handle_execution_terminal.lambda_handler(self._event("arn:e1"), None)
        self.assertEqual(out["released"], 2)
        self.assertIsNone(self.holder("arn:e1"))
        self.assertEqual(sem.list_leases("arn:e1"), [])

    def test_does_not_decrement_or_drain(self):
        """
        Invariant 1 from the other side: this handler deletes and ONLY deletes. The
        delete reaches the stream, which decrements and drains — that is what makes
        an aborted execution wake a waiter exactly as a clean release does, with no
        branching here.
        """
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        handle_execution_terminal.lambda_handler(self._event("arn:e1"), None)
        self.assertEqual(self.counter()["holders"]["N"], "1")

    def test_succeeded_is_handled_too(self):
        """Not redundant with the sync child's own `Release lease` state: it covers a
        Lambda that released and crashed before returning, and an ASL that never
        reached its release state."""
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        out = handle_execution_terminal.lambda_handler(
            self._event("arn:e1", "SUCCEEDED"), None)
        self.assertEqual(out["released"], 1)

    def test_execution_that_never_acquired_is_the_common_case(self):
        """The rule fires on all three machines; the orchestrator and the batch child
        never hold a lease."""
        out = handle_execution_terminal.lambda_handler(
            self._event("arn:never"), None)
        self.assertEqual(out, {"handled": True, "status": "ABORTED",
                               "released": 0})

    def test_already_released_is_not_an_error(self):
        sem.try_acquire("bedrock", "sonnet", "arn:e1", 2)
        sem.release("bedrock", "sonnet", "arn:e1")
        out = handle_execution_terminal.lambda_handler(self._event("arn:e1"), None)
        self.assertEqual(out["released"], 0)

    def test_malformed_event_is_not_retried(self):
        self.assertEqual(
            handle_execution_terminal.lambda_handler(
                {"source": "aws.states", "detail": {}}, None),
            {"handled": False})


class TestTheSemaphoreClientBuildsNoStepFunctionsClient(unittest.TestCase):
    """
    `states:SendTask*` cannot be resource-scoped — a task token is an opaque bearer
    credential, so the policy resource is effectively "*" and anything holding it can
    complete any execution whose token it obtains. `drain_pending` therefore takes
    the sender as a PARAMETER: the client module has no stepfunctions client and no
    import, which makes the boundary a property of the code rather than a convention
    someone can drift from.
    """

    def test_the_client_builds_no_step_functions_client(self):
        source = _read(ROOT, "lib", "semaphore_client.py")
        self.assertNotIn('boto3.client("stepfunctions")', source)
        # `send_task_success` appears only as drain_pending's PARAMETER — the
        # injection point. That it is a parameter and not a client is the whole
        # point, so assert on the signature rather than on the name's absence.
        self.assertIn("send_task_success: Callable[[str], None]", source)

    def test_exactly_three_modules_build_a_stepfunctions_client(self):
        """
        The standalone stack had ONE, and that was the isolation: one function, one
        role. Post-merge there are three legitimate builders, and each is here for a
        different reason — which is why the set is pinned rather than the count.
        """
        builders = []
        for dirpath, _dirs, files in os.walk(ROOT):
            if ("__pycache__" in dirpath or "/test" in dirpath
                    or "/dist" in dirpath):
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                if 'boto3.client("stepfunctions")' in _read(path):
                    builders.append(os.path.relpath(path, ROOT))
        self.assertEqual(sorted(builders), [
            "gateway_tool/callback_waker.py",        # consumer-token redeemer (§11)
            "gateway_tool/lambda_function.py",       # StartExecution; redeems via
                                                     # callback_registry on the
                                                     # absorbed-finished path only
            "states/acquire_lease/lambda_function.py",   # acquire-token redeemer
            "states/usage_tracking/lambda_function.py",  # GetExecutionHistory only
        ])

    def test_only_four_modules_redeem_a_task_token(self):
        """
        The invariant that survived the merge AND the tier collapse. IAM no longer
        separates anything — one role holds `states:SendTask*` and `bedrock:*`
        together (2026-08-12) — so this walk is now the ONLY thing keeping the two
        capabilities in different modules. An exact SET, not a count.

        `semaphore_client` appears because it CALLS its injected `send_task_success`
        parameter — the whole point of the injection, and the reason it holds no
        client of its own (asserted above). `handle_semaphore_release` does not
        appear: it passes the sender by name and never calls it.

        The two callback modules joined 2026-08-17 (§11): they redeem CONSUMER
        tokens — tokens this stack never issued, arriving via
        `callback_task_token` — never the semaphore's or the batch child's, and
        both run OUTSIDE the workflow role (gateway / waker roles, neither of
        which can call Bedrock — the compartment the tier collapse lost is
        partially back for exactly this pair).
        """
        callers = []
        for dirpath, _dirs, files in os.walk(ROOT):
            if ("__pycache__" in dirpath or "/test" in dirpath
                    or "/dist" in dirpath):
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                source = _code(path)
                if ".send_task_success(" in source or ".send_task_failure(" in source:
                    callers.append(os.path.relpath(path, ROOT))
        self.assertEqual(sorted(callers), [
            "gateway_tool/callback_registry.py",         # consumer tokens (sync path)
            "gateway_tool/callback_waker.py",            # consumer tokens (event path)
            "states/acquire_lease/lambda_function.py",   # acquire tokens
            "states/notify/lambda_function.py",          # batch tokens
        ])


# ─────────────────────────────────────────────────────────────────────────────
# §12.1 item 15 — capability separation, asserted against the CODE PATH
# ─────────────────────────────────────────────────────────────────────────────

class TestCapabilitySeparation(unittest.TestCase):
    """
    The six workflow functions share ONE zip and differ only in their IAM role and,
    for the two on the same stream, their STREAM_HANDLER marker. The IAM tests below
    cover the grants; these cover the code, because a grant absent from the policy
    but present in the code is a runtime AccessDenied on the unhappy path — the path
    nobody exercises before deploying.

    All of these read `_code()`, not `_read()`: these modules' docstrings explain the
    grants they must not hold, so matching the raw file would assert against the
    prose — passing the moment someone deletes the comment.
    """

    def test_the_drain_code_path_never_sends_a_task_result(self):
        source = _code(ROOT, "states", "drain", "lambda_function.py")
        self.assertNotIn("SendTask", source)
        self.assertNotIn("send_task", source)
        self.assertNotIn("stepfunctions", source)

    def test_the_notify_code_path_never_calls_bedrock(self):
        source = _code(ROOT, "states", "notify", "lambda_function.py")
        for marker in ("batch_client", "bedrock_runtime", "bedrock_control",
                       "boto3.client('bedrock", "converse"):
            self.assertNotIn(marker, source)

    def test_the_batch_status_code_path_calls_neither(self):
        source = _code(ROOT, "states", "batch_status", "lambda_function.py")
        for marker in ("SendTask", "send_task", "batch_client", "boto3"):
            self.assertNotIn(marker, source)

    def test_run_inference_never_touches_the_batch_control_plane(self):
        source = _code(ROOT, "states", "run_inference", "lambda_function.py")
        self.assertNotIn("create_job", source)
        self.assertNotIn("bedrock_control", source)
        self.assertNotIn("token_table", source)

    def test_the_router_owns_the_acquire_handler_now(self):
        """
        The inversion of the pre-merge test. `StateAcquireLease` used to be
        deliberately ABSENT — a local acquire handler needs the unscopeable
        `states:SendTaskSuccess` grant, and having it on a function that also called
        Bedrock was the concentration §1.2 prevented.

        It is present now, and after the 2026-08-12 tier collapse NOTHING in IAM
        keeps the two capabilities apart — the one workflow role holds `SendTask*`
        and `bedrock:InvokeModel` together, and every tier runs under it. The pinned
        key set is what is left of the guard: a ninth key added here without a
        matching decision means a state landed on the shared image without anyone
        deciding where it belongs.
        """
        self.assertEqual(set(router._STATES), {
            "StateValidateAndPlan", "StateUsageTracking", "StateAcquireLease",
            "StateRunInference", "StateReleaseLease", "StateSubmitBatch",
            "StateCollectBatch", "StateSummarize", "Drain"})

    def test_the_acquire_handler_cannot_call_bedrock(self):
        """
        The invariant that replaced "no local acquire handler". The token grant and
        the Bedrock grant are still never co-resident — asserted here on the code
        path, since a stack-level split no longer enforces it.
        """
        source = _code(ROOT, "states", "acquire_lease", "lambda_function.py")
        for marker in ("bedrock", "converse", "batch_client", "token_table"):
            self.assertNotIn(marker, source, marker)

    def test_validation_exception_is_re_exported_not_redefined(self):
        """
        Two classes with the same name in two modules satisfy `ErrorEquals`
        identically, so the duplication would never show up as a failure — it would
        show up as an `except ValidationException` that does not catch the one the
        semaphore raised.

        `QuotaSaturated` was the second name checked here and no longer exists in
        either module: a full cap queues the task token rather than raising
        (README.md §M2.1), so an exception name for it would be dead
        `ErrorEquals` configuration that reads as handled back-pressure.
        """
        from lib import custom_exceptions
        self.assertIs(custom_exceptions.ValidationException,
                      semaphore_client.ValidationException)
        self.assertFalse(hasattr(custom_exceptions, "QuotaSaturated"))
        self.assertFalse(hasattr(semaphore_client, "QuotaSaturated"))

    def test_the_gateway_tool_reuses_the_workflow_hash_rather_than_copying_it(self):
        """
        Two copies of the hash rule is a hash that silently disagrees, and the
        symptom is a resubmit that re-pays for work already done (§3).
        """
        source = _code(ROOT, "gateway_tool", "lambda_function.py")
        self.assertIn("from lib import identity", source)
        self.assertIn("identity.compute_workflow_id", source)
        self.assertNotIn("uuid5", source)
        self.assertNotIn("hashlib", source)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class TestRouter(unittest.TestCase):
    def test_routes_each_shape(self):
        stream = {"Records": [{"eventSource": "aws:dynamodb",
                               "eventName": "MODIFY", "dynamodb": {}}]}
        bedrock_event = {"source": "aws.bedrock",
                         "detail": {"batchJobArn": JOB_ARN, "status": "Completed"}}
        terminal_event = {"source": "aws.states",
                          "detail": {"executionArn": "arn:e1",
                                     "status": "ABORTED"}}
        cases = [
            ({"operation": "StateValidateAndPlan"},
             "states.validate_plan.lambda_function", None),
            ({"operation": "StateUsageTracking"},
             "states.usage_tracking.lambda_function", None),
            ({"operation": "StateAcquireLease"},
             "states.acquire_lease.lambda_function", None),
            ({"operation": "StateRunInference"},
             "states.run_inference.lambda_function", None),
            ({"operation": "StateReleaseLease"},
             "states.release_lease.lambda_function", None),
            ({"operation": "StateSubmitBatch"},
             "states.submit_batch.lambda_function", None),
            ({"operation": "StateCollectBatch"},
             "states.collect_batch.lambda_function", None),
            ({"operation": "StateSummarize"},
             "states.summarize.lambda_function", None),
            ({"operation": "Drain"}, "states.drain.lambda_function", None),
            # Same event, THREE functions across TWO tables, one marker apart —
            # §6.3, §6.5 and README.md §M2.4.
            (stream, "states.notify.lambda_function", "notify"),
            (stream, "states.drain.lambda_function", "drain"),
            (stream, "states.handle_semaphore_release.lambda_function",
             "semaphore"),
            (bedrock_event, "states.batch_status.lambda_function", "notify"),
            (bedrock_event, "states.drain.lambda_function", "drain"),
            # WorkflowFunction is the target of three trigger kinds; the terminal
            # rule fires on all three state machines and is matched on shape alone.
            (terminal_event, "states.handle_execution_terminal.lambda_function",
             "semaphore"),
        ]
        import importlib

        for event, module_name, marker in cases:
            with self.subTest(module=module_name, marker=marker):
                module = importlib.import_module(module_name)
                env = {router.STREAM_HANDLER_ENV: marker or ""}
                with mock.patch.dict(os.environ, env), \
                     mock.patch.object(module, "lambda_handler",
                                       return_value={"ok": module_name}) as handler:
                    self.assertEqual(router.lambda_handler(event, None),
                                     {"ok": module_name})
                handler.assert_called_once_with(event, None)

    def test_unroutable_event_raises(self):
        for event in ({}, None, {"operation": "Whatever"},
                      {"Records": [{"eventSource": "aws:sqs"}]},
                      {"source": "aws.states"},
                      {"source": "aws.bedrock"}):
            with self.subTest(event=event):
                with self.assertRaises(ValueError):
                    router.lambda_handler(event, None)

    def test_a_stream_subscription_without_a_marker_raises(self):
        """The three handlers share this zip; an unset marker would silently route
        stream records to whichever branch happened to be first — and the three are
        wired to two DIFFERENT tables, so a wrong branch reads keys that do not
        exist rather than failing loudly."""
        stream = {"Records": [{"eventSource": "aws:dynamodb", "dynamodb": {}}]}
        for marker in ("", "release", "Notify ", "quota", "sync-infer-quotas"):
            with self.subTest(marker=marker):
                with mock.patch.dict(os.environ,
                                     {router.STREAM_HANDLER_ENV: marker}):
                    if marker.strip().lower() in ("notify", "drain", "semaphore"):
                        continue
                    with self.assertRaises(ValueError):
                        router.lambda_handler(stream, None)


# ─────────────────────────────────────────────────────────────────────────────
# Input schema
# ─────────────────────────────────────────────────────────────────────────────

def _image_record(image: dict, record_id: str = "r1") -> list:
    return [{"record_id": record_id,
             "messages": [{"role": "user", "content": [
                 {"text": "describe"}, {"image": image}]}]}]


# Real file magic, base64'd with filler so the 24-char head is decodable.
_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24).decode()
_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 24).decode()
_WEBP_B64 = base64.b64encode(
    b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 24).decode()


class TestImageBlockValidation(unittest.TestCase):
    """
    Plan-time pre-flight on image blocks (added 2026-08-14, at the user's call:
    "if it is cheap to validate with the magic id, go for it" — it is: a 24-char
    head decode per image, no payload re-decode, no S3 GET).

    The marquee case is DOUBLE-ENCODED base64: on 2026-08-13 every vision model
    failed sync with `detected file MIME type text/plain` because the transport
    re-encoded the base64 string. That defect is fixed at the sync boundary
    (decode_blob_sources), and this validator is the plan-time tripwire that
    names the same class of corruption before a token is spent.
    """

    def _parse(self, image: dict):
        return input_schema.parse_records(_image_record(image))

    def test_a_wellformed_block_passes_per_format(self):
        for fmt, b64 in (("png", _PNG_B64), ("jpeg", _JPEG_B64),
                         ("webp", _WEBP_B64)):
            with self.subTest(fmt=fmt):
                self._parse({"format": fmt, "source": {"bytes": b64}})

    def test_nova_s3location_source_is_accepted_and_not_fetched(self):
        """The Nova family accepts `source.s3Location.uri`; only the scheme is
        checked — validation must not cost an S3 GET per record."""
        self._parse({"format": "png",
                     "source": {"s3Location": {"uri": "s3://b/k.png"}}})
        with self.assertRaises(ValidationException):
            self._parse({"format": "png",
                         "source": {"s3Location": {"uri": "https://b/k.png"}}})

    def test_double_encoded_base64_is_named_at_plan_time(self):
        """base64(base64(png)) decodes to ASCII (`iVBO...`) and no magic — the
        exact bytes every vision model rejected live on 2026-08-13."""
        double = base64.b64encode(_PNG_B64.encode()).decode()
        with self.assertRaisesRegex(ValidationException, "double-encoding"):
            self._parse({"format": "png", "source": {"bytes": double}})

    def test_a_format_that_contradicts_the_bytes_names_the_detected_one(self):
        with self.assertRaisesRegex(ValidationException, "look like 'png'"):
            self._parse({"format": "jpeg", "source": {"bytes": _PNG_B64}})

    def test_the_rest_of_the_reject_matrix(self):
        cases = {
            "unknown format": {"format": "bmp",
                               "source": {"bytes": _PNG_B64}},
            "no source": {"format": "png"},
            "both sources": {"format": "png",
                             "source": {"bytes": _PNG_B64,
                                        "s3Location": {"uri": "s3://b/k"}}},
            "raw-bytes-shaped value": {"format": "png",
                                       "source": {"bytes": 123}},
            "not base64": {"format": "png",
                           "source": {"bytes": "!!not-base64!!"}},
        }
        for name, image in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValidationException):
                    self._parse(image)


class TestBlobSourceDecodeAsymmetry(unittest.TestCase):
    """
    THE TWO TRANSPORTS WANT OPPOSITE THINGS — the 2026-08-13 finding, pinned.

    A record's image travels as a base64 STRING (records are JSON). Batch
    forwards the string verbatim (its input JSONL is JSON too; Bedrock decodes
    there); sync must hand boto3 RAW BYTES, because a str reaching a blob
    parameter is re-encoded as text — which is how every vision model failed
    sync (`detected file MIME type text/plain`) while the identical record
    PASSED via batch. The decode happens inside converse(), after the shared
    shaping, so sync and batch stay shaped identically.
    """
    IMG = {"format": "png", "source": {"bytes": _PNG_B64}}

    def _messages(self):
        return [{"role": "user", "content": [
            {"text": "describe"}, {"image": json.loads(json.dumps(self.IMG))}]}]

    def test_sync_hands_boto3_raw_bytes_and_batch_keeps_the_string(self):
        spec = bedrock_lib.resolve_model(REGISTRY, PRIMARY)
        messages = self._messages()
        runtime = _FakeRuntime()
        bedrock_lib.converse(runtime, spec, None, messages)
        sent = runtime.calls[0]["messages"][0]["content"][1]["image"]
        self.assertIsInstance(sent["source"]["bytes"], bytes)
        self.assertTrue(sent["source"]["bytes"].startswith(b"\x89PNG"))

        params = bedrock_lib.build_converse_params(
            spec, None, self._messages())
        record = bedrock_lib.to_batch_record("r1", params)
        kept = record["modelInput"]["messages"][0]["content"][1]["image"]
        self.assertEqual(kept["source"]["bytes"], _PNG_B64)
        json.dumps(record)   # the JSONL line must stay serializable

    def test_the_callers_record_is_never_mutated(self):
        """Records are re-read on checkpoint re-entry; an in-place decode would
        make the second read see bytes where JSON came from."""
        spec = bedrock_lib.resolve_model(REGISTRY, PRIMARY)
        messages = self._messages()
        bedrock_lib.converse(_FakeRuntime(), spec, None, messages)
        self.assertEqual(
            messages[0]["content"][1]["image"]["source"]["bytes"], _PNG_B64)

    def test_s3location_sources_pass_through_untouched_on_both_paths(self):
        """Nothing to decode: the Nova family's s3Location source is a pointer,
        and 'touch nothing you don't understand' is the rule for blocks."""
        spec = bedrock_lib.resolve_model(REGISTRY, PRIMARY)
        loc = {"format": "png",
               "source": {"s3Location": {"uri": "s3://b/k.png"}}}
        messages = [{"role": "user", "content": [{"image": loc}]}]
        runtime = _FakeRuntime()
        bedrock_lib.converse(runtime, spec, None, messages)
        self.assertEqual(
            runtime.calls[0]["messages"][0]["content"][0]["image"], loc)


class TestInputSchema(unittest.TestCase):
    def test_records_key_is_derived_not_accepted(self):
        """
        One prefix holds one request. Accepting a second uri adds the failure mode
        where request.json and request.jsonl point at different requests — and the
        hash would be computed over one and the work done on the other.
        """
        bucket, key, prefix = input_schema.request_pointer(
            {"input": {"request_s3uri": REQUEST_S3URI}})
        self.assertEqual((bucket, key, prefix), (PROXY, REQUEST_KEY, PREFIX))
        self.assertEqual(input_schema.records_key(prefix), RECORDS_KEY)

    def test_a_pointer_that_is_not_request_json_is_rejected(self):
        for uri in (f"s3://{PROXY}/{PREFIX}/payload.json", "", "not-an-s3-uri",
                    f"s3://{PROXY}"):
            with self.subTest(uri=uri):
                with self.assertRaises(ValidationException):
                    input_schema.request_pointer({"input": {"request_s3uri": uri}})

    def test_rejects_every_unrunnable_request(self):
        cases = {
            "no chain": {"output": {"bucket": PROXY, "prefix": PREFIX}},
            "empty chain": _request(model_chain=[]),
            "over the chain ceiling": _request(model_chain=[f"m{i}" for i in range(5)]),
            "non-string entry": _request(model_chain=[123]),
            "bad use_batch": {**_request(), "use_batch": "eventually"},
            "options not an object": {**_request(), "options": "fast"},
            "boolean temperature": _request(options={"temperature": True}),
            "temperature over range": _request(options={"temperature": 1.5}),
            "bad effort": _request(options={"effort": "maximum"}),
            "bad thinking": _request(options={"thinking": "enabled"}),
            "no output bucket": {**_request(), "output": {"prefix": PREFIX}},
            "no output prefix": {**_request(), "output": {"bucket": PROXY}},
        }
        for name, request in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValidationException):
                    input_schema.parse_request(request)

    def test_a_bare_string_chain_is_accepted_and_deduped_in_order(self):
        config = input_schema.parse_request(_request(model_chain=PRIMARY))
        self.assertEqual(config["model_chain"], [PRIMARY])
        config = input_schema.parse_request(
            _request(model_chain=[PRIMARY, FALLBACK, PRIMARY]))
        self.assertEqual(config["model_chain"], [PRIMARY, FALLBACK])

    def test_rejects_every_unrunnable_record_set(self):
        good = {"record_id": "r1",
                "messages": [{"role": "user", "content": [{"text": "x"}]}]}
        cases = {
            "empty": [],
            "not an object": [good, "nope"],
            "no record_id": [{"messages": good["messages"]}],
            "reserved prefix": [{**good, "record_id": "pad_1"}],
            "duplicate id": [good, dict(good)],
            "no messages": [{"record_id": "r1", "messages": []}],
            "message without a role": [
                {"record_id": "r1", "messages": [{"content": []}]}],
            "content not a list": [
                {"record_id": "r1",
                 "messages": [{"role": "user", "content": "text"}]}],
            "system of the wrong type": [{**good, "system": 42}],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValidationException):
                    input_schema.parse_records(rows)

    def test_over_the_record_ceiling_is_rejected(self):
        rows = [{"record_id": f"r{i}",
                 "messages": [{"role": "user", "content": [{"text": "x"}]}]}
                for i in range(input_schema.MAX_RECORDS_CEILING + 1)]
        with self.assertRaises(ValidationException):
            input_schema.parse_records(rows)

    def test_streamed_token_estimate_matches_the_list_form(self):
        """The streaming plan sums record_text_chars and converts ONCE —
        per-record integer division would drift from estimate_input_tokens'
        total on any corpus whose per-record chars aren't multiples of the
        chars/token ratio. Pin the two forms equal on such a corpus."""
        # three records of 3 chars each: per-record int(3/4)=0 sums to 0,
        # while the true total int(9/4)=2 — the drift case
        records = [
            {"record_id": "a", "system": "ab",
             "messages": [{"role": "user", "content": [{"text": "c"}]}]},
            {"record_id": "b",
             "system": [{"text": "d"}, "e"],
             "messages": [{"role": "user", "content": [{"text": "f"}]}]},
            {"record_id": "c",
             "messages": [{"role": "user", "content": [{"text": "ghi"}]}]},
        ]
        self.assertEqual(
            input_schema.estimate_input_tokens(records),
            input_schema.tokens_from_chars(
                sum(input_schema.record_text_chars(r) for r in records)))
        # and the drift the rule exists to prevent is real on this corpus
        self.assertNotEqual(
            input_schema.estimate_input_tokens(records),
            sum(input_schema.tokens_from_chars(
                input_schema.record_text_chars(r)) for r in records))

    def test_max_tokens_is_retired_from_parsed_records(self):
        """A stray per-record value is dropped, never carried — the knob is
        gone (see TestMaxTokensIsRetired for the full retirement contract)."""
        rows = [{"record_id": "r1", "max_tokens": 4096,
                 "messages": [{"role": "user", "content": [{"text": "x"}]}]},
                {"record_id": "r2", "messages": [{"role": "user", "content": [{"text": "x"}]}]}]
        parsed = input_schema.parse_records(rows)
        self.assertTrue(all("max_tokens" not in record for record in parsed))

    def test_choose_mode_is_the_resolution_table(self):
        """§5.6's table, row by row. `use_batch: false` ALWAYS means sync — the
        caller can opt out of batch but never force it into being — and true on
        an incapable model DOWNGRADES with the reason rather than raising."""
        cases = [
            ((False, True), dict(estimated_saving=9.0), "sync", "false"),
            ((True, True), dict(estimated_saving=0.0), "batch", "true"),
            ((True, False), dict(estimated_saving=9.0), "sync", "downgraded"),
            ((None, False), dict(estimated_saving=9.0), "sync", "cannot batch"),
            ((None, True), dict(estimated_saving=9.0), "batch", "clears"),
            ((None, True), dict(estimated_saving=0.001), "sync", "under"),
            ((None, True), dict(estimated_saving=None,
                                estimated_input_tokens=10 ** 9),
             "batch", "fallback"),
        ]
        for (use_batch, can), kwargs, want_mode, want_word in cases:
            kwargs.setdefault("estimated_input_tokens", 100)
            mode, reason = input_schema.choose_mode(use_batch, can, **kwargs)
            self.assertEqual(mode, want_mode, (use_batch, can, kwargs))
            self.assertIn(want_word, reason, (use_batch, can, kwargs))

    def test_jsonl_parse_names_the_offending_line(self):
        """Silently skipping a malformed line would under-report `records_total`, and
        the summary's arithmetic is how a caller detects partial success (§7)."""
        fake = _FakeS3()
        fake.put(PROXY, "a.jsonl", '{"a":1}\nnot json\n')
        saved, s3_utils.s3 = s3_utils.s3, fake
        try:
            with self.assertRaises(ValidationException) as raised:
                s3_utils.get_jsonl(PROXY, "a.jsonl")
            self.assertIn("line 2", str(raised.exception))
        finally:
            s3_utils.s3 = saved

    def test_access_denied_never_reads_as_absent(self):
        fake = _FakeS3()
        fake.errors[(PROXY, "x")] = _client_error("AccessDenied", "HeadObject", 403)
        saved, s3_utils.s3 = s3_utils.s3, fake
        try:
            with self.assertRaises(ClientError):
                s3_utils.object_exists(PROXY, "x")
        finally:
            s3_utils.s3 = saved


# ─────────────────────────────────────────────────────────────────────────────
# CloudFormation template
# ─────────────────────────────────────────────────────────────────────────────

class _CfnLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml

        class Loader(yaml.SafeLoader):
            pass

        def _passthrough(loader, tag_suffix, node):
            if isinstance(node, yaml.ScalarNode):
                return {tag_suffix: loader.construct_scalar(node)}
            if isinstance(node, yaml.SequenceNode):
                return {tag_suffix: loader.construct_sequence(node, deep=True)}
            return {tag_suffix: loader.construct_mapping(node, deep=True)}

        Loader.add_multi_constructor("!", _passthrough)
        cls.path = os.path.join(ROOT, "cfnstack.yaml")
        cls.raw = _read(cls.path)
        cls.tpl = yaml.load(cls.raw, Loader=Loader)  # nosec B506

    def policy(self, role):
        return json.dumps(self.tpl["Resources"][role]["Properties"]["Policies"])

    def env(self, function):
        return (self.tpl["Resources"][function]["Properties"]["Environment"]
                ["Variables"])


class TestCfnStack(_CfnLoader):
    def test_the_template_parses_at_all(self):
        """
        Trivial-looking and the reason this file exists at the top: a plain YAML
        scalar may not END with a colon, and `arn:aws:bedrock:${Region}:${Account}:`
        unquoted makes the WHOLE template unparseable — a failure that surfaces
        first as a CloudFormation error at deploy time, hours after the build.
        """
        self.assertIn("Resources", self.tpl)
        self.assertIn("Outputs", self.tpl)

    def test_there_are_exactly_three_tables_and_they_are_named_apart(self):
        """
        README.md §M decision 2. One table serving both would put a semaphore
        counter and a batch row in one keyspace, and the two clients marshal
        attributes differently — `pending#...` and `batch#...` would differ by prefix
        alone, with a Scan (the drain's) reading across both.

        This is also where the merge's ONE semantic change lives: the quota table is
        `${ResourcePrefix}-...` rather than a fixed region-wide name, so the cap is
        per prefix. Deliberate, recorded in §2.3, and the reason the table name is an
        Output.

        The THIRD table is the consumer callback-token store
        (README.md §11) — disjoint access pattern (gateway writes, waker
        completes) and disjoint roles from both others.
        """
        tables = {name for name, res in self.tpl["Resources"].items()
                  if res["Type"] == "AWS::DynamoDB::Table"}
        self.assertEqual(tables, {"SyncInferQuotasTable", "BatchInferTokensTable",
                                  "CallbackTokensTable"})
        for name in tables:
            table = self.tpl["Resources"][name]["Properties"]
            self.assertIn("ResourcePrefix", json.dumps(table["TableName"]), name)
            # Every row in both tables is transient and self-healing; a table
            # without TTL leaks a stranded holder or a parked batch row forever.
            self.assertTrue(table["TimeToLiveSpecification"]["Enabled"], name)
            self.assertEqual(table["TimeToLiveSpecification"]["AttributeName"],
                             "ttl", name)

    def test_the_quota_table_streams_and_has_no_index(self):
        """
        The stream is not an optimisation — it is the single decrement path all
        three remaining holder-disappearance triggers converge on (semaphore
        invariant 1). No GSI: the terminal handler finds a holder through the
        `lease#{execution_arn}` reverse pointer, which is a Query on the base table.
        """
        table = self.tpl["Resources"]["SyncInferQuotasTable"]["Properties"]
        self.assertEqual({attr["AttributeName"]
                          for attr in table["AttributeDefinitions"]}, {"pk", "sk"})
        self.assertNotIn("GlobalSecondaryIndexes", table)
        self.assertEqual(table["StreamSpecification"]["StreamViewType"],
                         "NEW_AND_OLD_IMAGES")

    def test_the_token_table_has_the_gsi_ttl_and_both_stream_images(self):
        table = self.tpl["Resources"]["BatchInferTokensTable"]["Properties"]
        self.assertEqual({attr["AttributeName"]
                          for attr in table["AttributeDefinitions"]},
                         {"pk", "sk", "job_arn"})
        self.assertTrue(table["TimeToLiveSpecification"]["Enabled"])
        self.assertEqual(table["TimeToLiveSpecification"]["AttributeName"], "ttl")
        # OLD as well as NEW: the notifier's terminal→terminal check reads the OLD
        # image, which is what makes it a transition trigger.
        self.assertEqual(table["StreamSpecification"]["StreamViewType"],
                         "NEW_AND_OLD_IMAGES")

        gsi, = table["GlobalSecondaryIndexes"]
        self.assertEqual(gsi["IndexName"], token_table.JOB_ARN_INDEX)
        self.assertEqual([key["AttributeName"] for key in gsi["KeySchema"]],
                         ["job_arn"])
        # KEYS_ONLY is not enough: BatchStatusFunction needs `model_id` to build the
        # pk for its UpdateItem, and a second GetItem would defeat the index.
        self.assertEqual(gsi["Projection"]["ProjectionType"], "INCLUDE")
        self.assertEqual(set(gsi["Projection"]["NonKeyAttributes"]),
                         {"status", "model_id"})

    def test_the_stream_filters_match_the_client_key_and_status_constants(self):
        """
        The FilterCriteria and the client's constants are one contract split across
        two files. Drift means either the handler never fires — the parked caller
        waits out its 24 h — or it fires on rows it must ignore.
        """
        notify_map = self.tpl["Resources"]["NotifyStreamMapping"]["Properties"]
        pattern, = [json.loads(f["Pattern"])
                    for f in notify_map["FilterCriteria"]["Filters"]]
        self.assertEqual(pattern["eventName"], ["INSERT", "MODIFY"])
        image = pattern["dynamodb"]["NewImage"]
        self.assertEqual(image["pk"]["S"][0]["prefix"],
                         token_table.BATCH_PK_PREFIX)
        self.assertEqual(set(image["status"]["S"]),
                         set(token_table.TERMINAL_STATUSES))

        drain_map = self.tpl["Resources"]["DrainStreamMapping"]["Properties"]
        patterns = [json.loads(f["Pattern"])
                    for f in drain_map["FilterCriteria"]["Filters"]]
        self.assertEqual(len(patterns), 2)
        events = {tuple(pattern["eventName"]) for pattern in patterns}
        self.assertEqual(events, {("REMOVE",), ("MODIFY",)})
        for pattern in patterns:
            keys = pattern["dynamodb"].get("Keys") or pattern["dynamodb"]["NewImage"]
            self.assertEqual(keys["pk"]["S"][0]["prefix"],
                             token_table.BATCH_PK_PREFIX)

    def test_the_semaphore_stream_filter_matches_the_client_key_prefixes(self):
        """
        A correctness requirement, not a cost saving. The one record that MUST NOT
        reach the decrement handler is a `stream-dedup` marker's own TTL expiry: it
        arrives as a REMOVE on the same table, and a handler that read it as a
        released holder would decrement a counter no holder ever incremented.
        """
        mapping = self.tpl["Resources"]["SemaphoreStreamMapping"]["Properties"]
        pattern, = [json.loads(f["Pattern"])
                    for f in mapping["FilterCriteria"]["Filters"]]
        self.assertEqual(pattern["eventName"], ["REMOVE"])
        keys = pattern["dynamodb"]["Keys"]
        self.assertEqual(keys["pk"]["S"][0]["prefix"],
                         semaphore_client.SEMAPHORE_PK_PREFIX)
        self.assertEqual(keys["sk"]["S"][0]["prefix"],
                         semaphore_client.HOLDER_SK_PREFIX)

    def test_all_three_stream_mappings_bound_and_bisect_their_retries(self):
        """
        Three consumers across TWO tables, and each must be bound to the right one: a
        mapping pointed at the other table's stream delivers records whose keys the
        handler simply does not recognise, which is a silent no-op rather than an
        error.
        """
        expected = {
            "NotifyStreamMapping": "BatchInferTokensTable.StreamArn",
            "DrainStreamMapping": "BatchInferTokensTable.StreamArn",
            "SemaphoreStreamMapping": "SyncInferQuotasTable.StreamArn",
        }
        mappings = {name for name, res in self.tpl["Resources"].items()
                    if res["Type"] == "AWS::Lambda::EventSourceMapping"}
        self.assertEqual(mappings, set(expected))
        for name, stream in expected.items():
            mapping = self.tpl["Resources"][name]["Properties"]
            self.assertTrue(mapping["BisectBatchOnFunctionError"], name)
            self.assertGreaterEqual(mapping["MaximumRetryAttempts"], 1, name)
            self.assertEqual(mapping["EventSourceArn"], {"GetAtt": stream}, name)

    def test_exactly_one_role_sends_task_results_and_it_can_also_call_bedrock(self):
        """
        **This test records a LOSS, and asserts it rather than quietly passing.**

        A task token is an opaque BEARER CREDENTIAL, not an ARN, so this grant cannot
        be resource-scoped — `states:SendTask*` on a role is the ability to complete
        any execution in the account. The standalone quota stack confined it to one
        function per stack, neither of which could call Bedrock; the merge widened
        that to two roles (README.md §M2.4) and kept the property that mattered:
        neither holder could reach a model.

        The tier collapse (2026-08-12) gave that up. There is ONE workflow role now
        and it holds both, so `assertIn("bedrock:")` is deliberate: the co-residency
        is asserted as the current state of affairs so that the day someone re-splits
        the role, this test fails and the reader is sent to the reasoning instead of
        finding a stale comment claiming an isolation that ended months earlier.

        An exact set, still — and it GREW on 2026-08-17 (§11): the gateway and
        waker roles now redeem CONSUMER callback tokens (`callback_task_token`),
        which is deliberate and bounded — SendTaskSuccess only (no
        SendTaskFailure: a verdict travels as DATA in a success payload), and
        NEITHER role can call Bedrock, so the co-residency this docstring
        records remains unique to the workflow role. The seeder joining this
        set would still be the much-worse change the original pin guarded.
        """
        holders = sorted(name for name, res in self.tpl["Resources"].items()
                         if res["Type"] == "AWS::IAM::Role"
                         and "SendTask" in json.dumps(
                             res["Properties"].get("Policies", [])))
        self.assertEqual(holders, ["BedrockInferenceLambdaRole",
                                   "CallbackWakerRole", "GatewayToolRole"])
        for role in ("GatewayToolRole", "CallbackWakerRole"):
            policy = json.dumps(
                self.tpl["Resources"][role]["Properties"]["Policies"])
            self.assertNotIn("bedrock:", policy, role)
            self.assertNotIn("SendTaskFailure", policy, role)
        self.assertIn("bedrock:InvokeModel", self.policy(holders[0]))
        # What remains of the old split (2026-08-17 revision): the
        # internet-facing dispatcher and the custom resource still reach no
        # Bedrock model, and the SEEDER still touches no task token at all.
        # The gateway DOES now redeem consumer callback tokens — that half of
        # the 2026-08-12 claim ended with §11, and the holders set above is
        # where the new boundary is asserted.
        for role in ("GatewayToolRole", "SeedModelRegistryRole"):
            self.assertNotIn("bedrock:", self.policy(role), role)
        self.assertNotIn("SendTask", self.policy("SeedModelRegistryRole"))

    def test_the_workflow_role_writes_the_counter_but_only_one_handler_does(self):
        """
        The narrowest form of the invariant that survives, and the ONLY one left.

        `WorkflowRole` used to be denied `dynamodb:UpdateItem` outright: the counter
        belonged to the quota stack's own stream handler, and the IAM boundary was
        what made all the holder-disappearance paths converge on one decrement. The
        merge put the stream handler on the workflow role, so the grant had to move
        there; the tier collapse then put every state on that one role. IAM enforces
        nothing about the convergence now, in either direction.

        What enforces it is the caller walk below — exactly one caller of
        `decrement_once` in the whole tree — and that walk is now the whole of the
        guarantee rather than a belt beside IAM's braces. A second caller is how a
        counter drifts below its holder count and silently inflates the cap with no
        path back, and nothing else in the stack would notice.
        """
        policy = self.policy("BedrockInferenceLambdaRole")
        self.assertIn("dynamodb:UpdateItem", policy)
        self.assertIn("dynamodb:DeleteItem", policy)
        # ListBucket so `object_exists()` can tell "absent" from "denied".
        self.assertIn("s3:ListBucket", policy)
        self.assertIn("index/job_arn-index", policy)
        # `models_with_pending()` is a Scan and both drain paths call it. It used to
        # be DrainRole's alone — one role means every state can read the whole token
        # table now, which is the same trade as the paragraph above.
        self.assertIn("dynamodb:Scan", policy)

        callers = []
        for dirpath, _dirs, files in os.walk(ROOT):
            if ("__pycache__" in dirpath or "/test" in dirpath
                    or "/dist" in dirpath):
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                if "decrement_once(" in _code(path):
                    callers.append(os.path.relpath(path, ROOT))
        self.assertEqual(sorted(callers), [
            "lib/semaphore_client.py",                             # the definition
            "states/handle_semaphore_release/lambda_function.py",   # the one caller
        ])

    def test_the_run_inference_code_cannot_decrement_either(self):
        """
        The IAM half of this test is gone with the role split: `run_inference` runs
        on the union role and holds `UpdateItem`, `Scan` and the stream grants it has
        no use for. What is left is the half that was always the real assertion —
        the code does not call any of the three, so the counter is not its business
        even though the permission to touch it is now sitting right there.

        Asserted against `_code()`, not the raw file, because the module's docstring
        discusses the counter it must not touch.
        """
        source = _code(ROOT, "states", "run_inference", "lambda_function.py")
        for marker in ("decrement_once", "drain_pending", "try_acquire"):
            self.assertNotIn(marker, source, marker)

    def test_the_batch_service_role_is_scoped_and_guards_the_confused_deputy(self):
        role = self.tpl["Resources"]["BedrockBatchServiceRole"]["Properties"]
        document = json.dumps(role["AssumeRolePolicyDocument"])
        self.assertIn("bedrock.amazonaws.com", document)
        self.assertIn("aws:SourceAccount", document)
        self.assertIn("aws:SourceArn", document)
        policy = json.dumps(role["Policies"])
        self.assertIn("micro-workflow/*", policy)
        self.assertNotIn('"s3:*"', policy)

    def test_the_batch_service_role_can_invoke_the_model_itself(self):
        """
        **The job invokes the model under THIS role, not under the workflow role.** S3
        i/o alone is not enough, and this was a real deploy-time failure whose cost is
        in how late it surfaces: the submit is accepted, the row goes SUBMITTED, the
        token parks, the job validates and schedules and runs, and only minutes later
        does it end `Failed` with "Customer doesn't have permissions to invokeModel".
        Nothing in this stack rejects anything — the drain sees a healthy row and the
        only signal is the job's own terminal `message`.

        Both resource forms are asserted because a cross-region inference profile
        resolves to a foundation-model arn carrying EITHER an empty region or a real
        one, and the profile arn itself is account-scoped. `get-inference-profile` on
        global.anthropic.claude-sonnet-4-6 returns exactly those two shapes in its
        `models` list. This is the same grant the workflow role holds, which is
        precisely why sync worked while batch failed — so the two are compared rather
        than each being spelled out twice.

        Compared by CONTAINMENT, not equality: the workflow role's single Bedrock
        statement carries `model-invocation-job/*` too, because the tier collapse
        merged the data plane and the control plane into one statement. The service
        role must NOT gain that third resource — it is the confused-deputy guard at
        the end of this test — so the shared requirement is the two model forms, and
        equality here would force the wrong one of the two roles to widen to keep the
        test green.
        """
        def invoke_resources(role_name):
            statements = [
                statement
                for policy in (self.tpl["Resources"][role_name]["Properties"]
                               ["Policies"])
                for statement in policy["PolicyDocument"]["Statement"]
                if "bedrock:InvokeModel" in _as_list(statement["Action"])]
            self.assertEqual(len(statements), 1, role_name)
            return _as_list(statements[0]["Resource"])

        batch = invoke_resources("BedrockBatchServiceRole")
        workflow = invoke_resources("BedrockInferenceLambdaRole")

        # A foundation-model arn with an EMPTY region and an account-scoped
        # inference-profile arn. Either one missing is a job that runs and then fails.
        for required in ("foundation-model/", "inference-profile/"):
            self.assertIn(required, json.dumps(batch), required)
            self.assertIn(required, json.dumps(workflow), required)
        self.assertTrue(set(map(json.dumps, batch))
                        <= set(map(json.dumps, workflow)))
        self.assertNotIn("model-invocation-job/", json.dumps(batch))

        # It reads and writes, but it must never be able to submit or cancel a job:
        # the control plane belongs to the Lambdas, and a service role that could
        # create jobs could create them outside this workflow's accounting.
        policy_text = self.policy("BedrockBatchServiceRole")
        self.assertNotIn("CreateModelInvocationJob", policy_text)
        self.assertNotIn("SendTask", policy_text)

    def test_the_capacity_rule_is_unscoped_and_the_token_rule_is_not(self):
        """
        §6.5's asymmetry, and the v5 defect it fixes. The batch job limit is per
        ACCOUNT while this stack is not a singleton, so a FOREIGN completion is the
        only signal that capacity freed — but a foreign completion is never news
        about one of OUR task tokens.
        """
        token = self.tpl["Resources"]["BatchTokenRule"]["Properties"]["EventPattern"]
        self.assertIn("batchJobArn", token["detail"])
        self.assertIn("region", token)

        capacity = (self.tpl["Resources"]["BatchCapacityRule"]["Properties"]
                    ["EventPattern"])
        self.assertNotIn("batchJobArn", capacity["detail"])
        self.assertNotIn("region", capacity)
        self.assertEqual(set(token["detail"]["status"]),
                         set(capacity["detail"]["status"]))
        self.assertEqual(set(token["detail"]["status"]),
                         set(token_table.BEDROCK_STATUS_MAP))

    def test_the_two_bedrock_rules_target_different_tiers(self):
        """
        They used to target two different FUNCTIONS (`BatchStatusLambda` and
        `DrainLambda`) and the split was what made the router's `aws.bedrock` branch
        resolve differently for each. It is two different TIERS now, and the marker
        still does the work: the token rule lands on the tier whose `STREAM_HANDLER`
        is `notify` (→ `states.batch_status`), the capacity rule on the tier marked
        `drain`. Point them at the same tier and one of the two behaviours vanishes
        with no error — a token that is never redeemed, or a queue that never moves.
        """
        def target(rule):
            targets = self.tpl["Resources"][rule]["Properties"]["Targets"]
            return json.dumps([entry["Arn"] for entry in targets])

        self.assertIn("BedrockInferenceLambdaHalfCore", target("BatchTokenRule"))
        self.assertIn("BedrockInferenceLambda1Core", target("BatchCapacityRule"))
        self.assertIn("BedrockInferenceLambda1Core", target("DrainScheduleRule"))
        self.assertNotEqual(target("BatchTokenRule"), target("BatchCapacityRule"))

    def test_the_scheduled_sweep_carries_no_image(self):
        """The payload has no model, so the drain falls back to
        `models_with_pending()` — the backstop for TTL expiry, manual deletion and
        any missed event."""
        targets = self.tpl["Resources"]["DrainScheduleRule"]["Properties"]["Targets"]
        payload = json.loads(targets[0]["Input"])
        self.assertEqual(payload, {"operation": "Drain"})
        self.assertIn(payload["operation"], router._STATES)

    def test_the_terminal_rule_covers_all_three_machines(self):
        """
        Trigger 3, and the reason the sync child's ASL needs no `Catch` at all
        (README.md §M3.2/§M3.4). All THREE machines are listed: an orchestrator
        ABORTED mid-`.sync:2` does not necessarily terminate its child, so the
        child's own terminal event is the one that reclaims — and a machine omitted
        here is a lease that lingers for the full holder TTL with nothing reporting
        it, which is a silent throughput loss rather than an error.

        An exact list, not a subset: a fourth machine added without a decision means
        an execution can hold a lease that this rule does not watch.
        """
        rule = self.tpl["Resources"]["QuotaTerminalRule"]["Properties"]
        pattern = rule["EventPattern"]
        self.assertEqual(pattern["source"], ["aws.states"])
        self.assertEqual(set(pattern["detail"]["status"]),
                         {"SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"})
        self.assertEqual(
            [entry["GetAtt"] for entry in pattern["detail"]["stateMachineArn"]],
            ["InferenceStateMachine.Arn", "SyncInferenceStateMachine.Arn",
             "BatchInferenceStateMachine.Arn"])
        machines = {name for name, res in self.tpl["Resources"].items()
                    if res["Type"] == "AWS::StepFunctions::StateMachine"}
        self.assertEqual(
            {entry["GetAtt"].split(".")[0]
             for entry in pattern["detail"]["stateMachineArn"]}, machines)
        # The target is in-stack now; it used to be the quota stack's cleanup
        # function ARN passed in as a parameter. It is the 2-cores tier because that
        # is where the `aws.states` branch's states live — and note the tier is also
        # the semaphore stream consumer, which is harmless: the router picks the
        # branch by event shape, and `STREAM_HANDLER` is read only for a DynamoDB
        # record.
        self.assertEqual(rule["Targets"][0]["Arn"],
                         {"GetAtt": "BedrockInferenceLambda2Cores.Arn"})

    def test_every_rule_target_has_an_invoke_permission(self):
        permissions = json.dumps([
            res["Properties"] for res in self.tpl["Resources"].values()
            if res["Type"] == "AWS::Lambda::Permission"])
        for rule in (name for name, res in self.tpl["Resources"].items()
                     if res["Type"] == "AWS::Events::Rule"):
            self.assertIn(rule, permissions, rule)

    def test_each_machine_may_invoke_only_the_tiers_its_own_asl_names(self):
        """
        Three machines, three roles, and each one's `lambda:InvokeFunction` grant is
        exactly the set of TIERS its own ASL targets — which is a weaker statement
        than it used to be, and worth being precise about rather than deleting.

        It used to read as capability: the orchestrator could not invoke
        `RunInferenceLambda`, the only holder of `bedrock:InvokeModel`, so a machine
        that takes no lease structurally could not make a model call. Every tier now
        runs on the union role, so reaching ANY tier is reaching Bedrock and that
        reading is gone.

        What is left is real but narrower: a machine can invoke only the tiers its
        own states are on, so the SYNC child cannot be made to run `Submit batch`'s
        state (2-cores is granted to it, but `SubmitBatchLambdaArn` is not in its
        substitutions) and — the one that still bites — a state moved to a new tier
        without granting it fails with `AccessDenied` at deploy-test time rather than
        silently working, which is what keeps the grant honest as states move.
        """
        every = {"HalfCore", "1Core", "2Cores", "3Cores", "4Cores", "5Cores",
                 "6Cores"}
        expected = {
            # The orchestrator's four states are all on the 2-cores tier.
            "StateMachineRole": {"2Cores"},
            # `Acquire lease`, `Release lease`, `Summarize` on 2-cores; the sync
            # loop's `Run inference` on 1-core.
            "SyncStateMachineRole": {"1Core", "2Cores"},
            # `Submit batch`, `Collect batch output` and `Summarize` are all on
            # 2-cores, so the batch machine needs exactly one tier. It notably does
            # NOT get 1-core, which is where `Run inference` lives: the repair tail
            # STARTS the sync child rather than invoking its functions (2026-08-12),
            # so the batch machine still cannot invoke the sync loop directly.
            "BatchStateMachineRole": {"2Cores"},
        }
        for role, allowed in expected.items():
            statements = [
                stmt for policy in
                self.tpl["Resources"][role]["Properties"]["Policies"]
                for stmt in policy["PolicyDocument"]["Statement"]
                if "lambda:InvokeFunction" in json.dumps(stmt["Action"])]
            invokable = set(re.findall(r"BedrockInferenceLambda(\w+)\.Arn",
                                       json.dumps(statements)))
            self.assertEqual(invokable, allowed, role)
            # The inverse, spelled out: no other tier is reachable from this role at
            # all, not merely absent from the invoke statement.
            policy = self.policy(role)
            for denied in every - allowed:
                self.assertNotIn(f"BedrockInferenceLambda{denied}.Arn", policy,
                                 f"{role}: {denied}")
            # And the gateway's own function is reachable from none of them.
            self.assertNotIn("GatewayToolLambda", policy, role)

    def test_the_start_graph_is_exactly_this_and_is_acyclic(self):
        """
        **The invariant is acyclicity, not "only the orchestrator may start."** That
        older, stricter rule was what the nested-child repair had to give up: the batch
        machine starts the sync child as its repair pass (§8), which is precisely what
        lets the repair BE the sync spine instead of a second copy of it.

        What must survive is the reason the old rule existed — a machine that can reach
        itself, directly or through a sibling, is unbounded recursion that bills a real
        Bedrock job per level. A DAG with no self-edge gives that, and it is checkable
        rather than assumed. The depth also matters: the deepest path is
        orchestrator → batch → sync, three levels, which is what `Usage tracking`'s
        `_MAX_CHILD_DEPTH` is sized for.

        An exact set per role, because "InferenceStateMachine" is a SUFFIX of both
        child names — a substring assertion would pass on the recursion it forbids.
        """
        machines = {"StateMachineRole": "InferenceStateMachine",
                    "SyncStateMachineRole": "SyncInferenceStateMachine",
                    "BatchStateMachineRole": "BatchInferenceStateMachine"}

        def startable(role):
            starts = [stmt for p in
                      self.tpl["Resources"][role]["Properties"]["Policies"]
                      for stmt in p["PolicyDocument"]["Statement"]
                      if "states:StartExecution" in json.dumps(stmt["Action"])]
            return set(re.findall(r"(\w+StateMachine)\.Arn", json.dumps(starts)))

        graph = {machines[role]: startable(role) for role in machines}
        self.assertEqual(graph, {
            "InferenceStateMachine": {"SyncInferenceStateMachine",
                                      "BatchInferenceStateMachine"},
            "BatchInferenceStateMachine": {"SyncInferenceStateMachine"},
            "SyncInferenceStateMachine": set(),
        })

        # Acyclic AND no self-edge, derived from the graph rather than eyeballed, so a
        # future grant is checked by the same rule instead of by whoever reviews it.
        def depth(node, seen=()):
            self.assertNotIn(node, seen, f"start cycle: {' → '.join(seen)} → {node}")
            return 1 + max((depth(child, seen + (node,))
                            for child in graph[node]), default=0)

        self.assertEqual(depth("InferenceStateMachine"), 3)

        # `.sync:2` needs FIVE grants, not one, and every role that starts a child
        # needs all five: DescribeExecution + StopExecution on the execution arn, plus
        # the managed rule EventBridge creates. Omitting the events: three is the worst
        # of the failures — the child SUCCEEDS and the parent hangs to its own timeout.
        for role, machine in machines.items():
            if not graph[machine]:
                self.assertNotIn("states:StartExecution", self.policy(role), role)
                continue
            policy = self.policy(role)
            for action in ("states:DescribeExecution", "states:StopExecution",
                           "events:PutRule", "events:PutTargets",
                           "events:DescribeRule"):
                self.assertIn(action, policy, f"{role}: {action}")

    def test_each_machines_definition_substitutions_balance_its_own_asl_tokens(self):
        """
        Three ASLs, three substitution maps. A token used but not declared fails at
        deploy; a token declared but unused is the interesting half — it means a
        state was moved to another machine and its ARN grant was left behind.
        """
        for machine, resource in (("orchestrator", "InferenceStateMachine"),
                                  ("sync", "SyncInferenceStateMachine"),
                                  ("batch", "BatchInferenceStateMachine")):
            declared = set(self.tpl["Resources"][resource]["Properties"]
                           ["DefinitionSubstitutions"])
            used = set(re.findall(r"\$\{(\w+)\}",
                                  _read(ROOT, _ASL_FILES[machine])))
            self.assertEqual(declared, used, machine)

    def test_each_machine_gets_its_own_role_and_log_group(self):
        """A shared role would give the orchestrator the union of all three grants —
        including the one function that can call Bedrock."""
        machines = {
            "InferenceStateMachine": ("StateMachineRole", "StateMachineLogGroup"),
            "SyncInferenceStateMachine": ("SyncStateMachineRole",
                                          "SyncStateMachineLogGroup"),
            "BatchInferenceStateMachine": ("BatchStateMachineRole",
                                           "BatchStateMachineLogGroup"),
        }
        found = {name for name, res in self.tpl["Resources"].items()
                 if res["Type"] == "AWS::StepFunctions::StateMachine"}
        self.assertEqual(found, set(machines))
        for name, (role, log_group) in machines.items():
            props = self.tpl["Resources"][name]["Properties"]
            self.assertEqual(props["RoleArn"], {"GetAtt": f"{role}.Arn"}, name)
            self.assertIn(log_group, json.dumps(props["LoggingConfiguration"]), name)

    def test_the_stream_marker_distinguishes_the_shared_image_tiers(self):
        """
        THREE stream consumers now, across two tables, and the marker is what tells
        them apart — a deploy-time misconfiguration in one line of YAML rather than a
        runtime privilege question.

        **The marker is what PINS the tier packing** (2026-08-12). One marker per
        tier, so the two old `notify`-marked functions had to land on the same tier
        and the other four old functions had nowhere else to go: given three markers
        and six functions, the assignment below is the only legal one. That is why
        this test is the one to read before moving a state between tiers.

        The half-core tier's marker is 'notify', NOT 'drain', and that is exactly
        what makes the router's `aws.bedrock` branch resolve to `states.batch_status`
        for the token rule. Setting it to 'drain' would silently turn that rule into a
        second capacity sweep and no task token would ever be redeemed.

        The 2-cores tier's is 'semaphore', which the router also relies on in the
        NEGATIVE: its `aws.bedrock` branch tests for 'drain' explicitly rather than
        "not notify", so a stray Bedrock event reaching that tier lands on
        batch_status rather than on a second sweep.

        The four unrouted tiers carry NO marker at all, deliberately: an unset marker
        raises, and a tier nothing is subscribed to should fail loudly if a stream is
        ever pointed at it without a decision about which handler it runs.
        """
        def _vars(name):
            return ((self.tpl["Resources"][name]["Properties"]
                     .get("Environment") or {}).get("Variables") or {})

        markers = {name: _vars(name).get("STREAM_HANDLER")
                   for name, res in self.tpl["Resources"].items()
                   if res["Type"] == "AWS::Lambda::Function"
                   and "STREAM_HANDLER" in _vars(name)}
        self.assertEqual(markers, {
            "BedrockInferenceLambdaHalfCore": "notify",
            "BedrockInferenceLambda1Core": "drain",
            "BedrockInferenceLambda2Cores": "semaphore"})
        # And every value the template sets is one the router accepts — the other
        # half of the same contract, which is what makes a typo a test failure
        # instead of an AccessDenied on the first real invocation.
        for name, marker in markers.items():
            with mock.patch.dict(os.environ,
                                 {router.STREAM_HANDLER_ENV: marker}):
                self.assertIn(router._stream_module(),
                              ("states.notify", "states.drain",
                               "states.handle_semaphore_release"), name)

    def test_every_tier_carries_the_full_environment(self):
        """
        **Every tier gets the WHOLE environment, not the subset its own states read**,
        and that is the point of the assertion being over all seven rather than a
        per-tier expectation.

        A per-tier subset would make moving a state between tiers a two-line change —
        the ASL substitution and the env — with `KeyError` at runtime as the failure
        mode for forgetting the second. The tiers are interchangeable by construction
        instead: one image, one role, one env, and the only per-tier differences are
        `MemorySize` and `STREAM_HANDLER`.
        """
        tiers = [name for name in self.tpl["Resources"]
                 if name.startswith("BedrockInferenceLambda")
                 and self.tpl["Resources"][name]["Type"] == "AWS::Lambda::Function"]
        self.assertEqual(len(tiers), 7, tiers)
        for name in tiers:
            env = self.env(name)
            self.assertIn(token_table.TABLE_NAME_ENV, env, name)
            self.assertIn(semaphore_client.TABLE_NAME_ENV, env, name)
            self.assertIn("BATCH_SERVICE_ROLE_ARN", env, name)
            self.assertIn("PROXY_BUCKET", env, name)
            self.assertIn("MODEL_REGISTRY_KEY", env, name)
        # The gateway is NOT a tier and must not be handed either table: it is the
        # untrusted edge and its role cannot reach them either.
        gateway = self.env("GatewayToolLambda")
        self.assertNotIn(token_table.TABLE_NAME_ENV, gateway)
        self.assertNotIn(semaphore_client.TABLE_NAME_ENV, gateway)

    def test_the_seven_tiers_differ_only_in_memory_and_marker(self):
        """
        The ladder's whole value is that a state can move between memory sizes with a
        one-line ASL substitution change, and that only holds while the tiers are
        otherwise identical. Asserted by DIFFING the property dicts rather than by
        listing what each must contain, so a property added to one tier and forgotten
        on the other six fails here instead of at the first invocation on that tier.

        `MemorySize` and `STREAM_HANDLER` are the two allowed differences;
        `FunctionName` differs by construction (it is derived from the tier name) and
        is normalised out.

        Note the two spellings, and that the pairing is asserted rather than assumed:
        the LOGICAL id suffix is `1Core`/`2Cores` (a CloudFormation logical id may not
        start with a digit, so `HalfCore` keeps its word form) while the MAPPING key is
        `OneCore`/`TwoCores`. A function reaching the wrong tier's mapping entry is a
        silently mis-sized function, since either key resolves.
        """
        ladder = {"HalfCore": ("HalfCore", 896), "1Core": ("OneCore", 1792),
                  "2Cores": ("TwoCores", 3584), "3Cores": ("ThreeCores", 5376),
                  "4Cores": ("FourCores", 7168), "5Cores": ("FiveCores", 8960),
                  "6Cores": ("SixCores", 10240)}
        shapes = {}
        for suffix, (key, _memory) in ladder.items():
            name = f"BedrockInferenceLambda{suffix}"
            props = json.loads(json.dumps(
                self.tpl["Resources"][name]["Properties"]))
            self.assertEqual(props["MemorySize"],
                             {"FindInMap": ["Lambda", key, "Memory"]}, name)
            del props["MemorySize"], props["FunctionName"]
            props["Environment"]["Variables"].pop("STREAM_HANDLER", None)
            shapes[suffix] = json.dumps(props, sort_keys=True)
        self.assertEqual(len(set(shapes.values())), 1, shapes)
        # And the ladder itself is the house's, not a set of numbers picked here: the
        # sibling workflows all use these seven values under these seven
        # names, which is what makes a tier name mean the same thing across stacks.
        self.assertEqual(
            {key: entry["Memory"]
             for key, entry in self.tpl["Mappings"]["Lambda"].items()
             if entry.get("Memory") and "Timeout" not in entry},
            {key: memory for key, memory in ladder.values()})

    def test_the_tiers_are_images_and_the_zip_functions_are_python313(self):
        """
        Two packaging kinds in one stack now (2026-08-12), and the split is the one
        every sibling makes: the seven tiers run the CodeBuild-built image, while the
        three functions that are not workflow states stay flat zips — the gateway
        dispatcher (a different artifact with a different lifecycle), the
        build-starter (it builds the image and so cannot be in it), and the
        model-registry seeder (a CloudFormation custom resource; no sibling puts one
        in the image, and on the image it could not seed until the build finished).

        **An image function has NO `Handler`**, and that is the assertion that matters:
        the entry point comes from the image `CMD`. There is no `ImageConfig` anywhere
        in this template, so all seven tiers share ONE entry point — an
        `ImageConfig.Command` on a tier would silently override `CMD` with something
        the router never sees.

        The seeder was the only image function with a custom entry point, and moving
        it off is also what unblocked the zip→image migration: `PackageType` requires
        replacement, and CloudFormation refuses to replace a CUSTOM-NAMED resource, so
        a function keeping its name across that change rolls the whole stack back.
        """
        functions = {name: res["Properties"]
                     for name, res in self.tpl["Resources"].items()
                     if res["Type"] == "AWS::Lambda::Function"}
        images = {name for name, props in functions.items()
                  if props.get("PackageType") == "Image"}
        zips = set(functions) - images
        self.assertEqual(len(images), 7, images)   # the seven tiers, nothing else
        self.assertEqual(zips, {"GatewayToolLambda", "StartBuildLambda",
                                "SeedModelRegistryLambda",
                                "CallbackWakerLambda"})

        # arm64 everywhere the workflow's own code runs. `StartBuildLambda` is the
        # exception and is left at the default deliberately: it is copied verbatim
        # from the siblings, it runs three boto3 calls from an inline `ZipFile`, and
        # the architecture it runs on has nothing to do with the image it builds —
        # that is `ARM_CONTAINER` on the CodeBuild project.
        for name in set(functions) - {"StartBuildLambda"}:
            self.assertEqual(functions[name]["Architectures"], ["arm64"], name)
        self.assertNotIn("Architectures", functions["StartBuildLambda"])
        for name in images:
            props = functions[name]
            self.assertNotIn("Handler", props, name)
            self.assertNotIn("Runtime", props, name)
            self.assertIn("ImageUri", props["Code"], name)
        # ONE image, ONE entry point. Nothing overrides `CMD`.
        self.assertEqual({name for name in images
                          if "ImageConfig" in functions[name]}, set())
        self.assertNotIn("ImageConfig", json.dumps(self.tpl))

        for name in zips:
            props = functions[name]
            self.assertEqual(props["Runtime"], "python3.13", name)
            self.assertIn("Handler", props, name)

        # The seeder ships in the GATEWAY TOOL package, not in a third zip and not
        # in the image — media-processing's pattern for its own custom resource. The
        # dotted handler is what forces the build script to keep `states/` and
        # `models/` unflattened; a `-j` there would package a seeder that cannot
        # import and cannot find its master list.
        seeder = functions["SeedModelRegistryLambda"]
        self.assertEqual(seeder["Handler"],
                         "states.seed_model_registry.lambda_function.lambda_handler")
        self.assertEqual(
            seeder["Code"]["S3Key"]["Sub"][1]["pkg"],
            {"FindInMap": ["CodeBuild", "Source", "GatewayToolPackage"]})
        self.assertEqual(seeder["Code"]["S3Key"]["Sub"][1]["pkg"],
                         functions["GatewayToolLambda"]["Code"]["S3Key"]["Sub"][1]["pkg"])
        # And it must NOT wait on the build any more — that dependency was the
        # reason it was on the image, and it is the thing being given up.
        self.assertNotIn(
            "StartBuild",
            json.dumps(self.tpl["Resources"]["SeedModelRegistryLambda"]
                       .get("DependsOn", [])))

    def test_every_function_has_its_own_declared_log_group(self):
        """
        Including the four tiers nothing routes to. A declared group with no traffic
        costs nothing; a function whose group is not declared creates an unmanaged one
        with never-expire retention on its first invocation — which then orphans on
        stack delete and makes the NEXT create fail with "already exists". That is
        exactly what would happen the first time a state is moved onto an unused tier.

        Matched by NAME rather than by logical-id convention, because the convention
        is what the pairing actually rests on: a named function logs to
        `/aws/lambda/{FunctionName}` whether or not a group is declared, so a group
        declared under the right logical id with the wrong `LogGroupName` — a tier's
        group pointing at another tier's name is one `!FindInMap` typo away — reads as
        present and is silently unused. Comparing the `!Sub` structures catches it.

        `StartBuildLambda` is the one exception and it inverts the rule: it is
        deliberately UNNAMED (matching the siblings), so `/aws/lambda/{FunctionName}`
        is unknowable at author time and `LoggingConfig` is the only thing that can
        reach a declared group at all.
        """
        groups = {json.dumps(res["Properties"]["LogGroupName"], sort_keys=True)
                  for res in self.tpl["Resources"].values()
                  if res["Type"] == "AWS::Logs::LogGroup"}

        def prefixed(function_name):
            """`/aws/lambda/` + the function's own name, in !Sub structure."""
            body, variables = function_name["Sub"]
            return json.dumps({"Sub": [f"/aws/lambda/{body}", variables]},
                              sort_keys=True)

        for name, res in self.tpl["Resources"].items():
            if res["Type"] != "AWS::Lambda::Function":
                continue
            props = res["Properties"]
            if "FunctionName" not in props:
                self.assertIn("LoggingConfig", props, name)
                continue
            self.assertIn(prefixed(props["FunctionName"]), groups, name)
        # Every group is retention-bounded; an unbounded one bills forever.
        for name, res in self.tpl["Resources"].items():
            if res["Type"] == "AWS::Logs::LogGroup":
                self.assertIn("RetentionInDays", res["Properties"], name)

    def test_required_outputs_present_and_none_exported(self):
        """
        `InferenceStateMachineArn` is still the ORCHESTRATOR's arn — that is what
        makes the split invisible to a caller who was already starting this stack.
        The two children are deliberately NOT output: starting one directly requires
        a plan, and a consumer holding a child arn would bypass `Validate & plan`.
        """
        for key in ("InferenceStateMachineArn", "InferenceStateMachineName",
                    "GatewayToolLambdaArn", "SyncInferQuotasTableName",
                    "BatchInferTokensTableName", "BedrockBatchServiceRoleArn"):
            self.assertIn(key, self.tpl["Outputs"])
        outputs = json.dumps(self.tpl["Outputs"])
        self.assertIn("InferenceStateMachine.Arn", outputs)
        for child in ("SyncInferenceStateMachine", "BatchInferenceStateMachine"):
            self.assertNotIn(child, outputs, child)
        # An Export cannot be changed or deleted while any importer exists, which
        # would weld every consumer stack's lifecycle to this one.
        for name, output in self.tpl["Outputs"].items():
            self.assertNotIn("Export", output, name)

    def test_the_stack_names_no_sibling_stack_at_all(self):
        """
        The merge's headline: self-contained. Three parameters used to name
        `service-quotas-on-aws` (`ServiceQuotasTableName`, `AcquireFunctionArn`,
        `ServiceQuotasCleanupFunctionArn`) and are gone — the parameter list is
        pinned exactly, because a leftover one would deploy fine and be silently
        ignored while the operator believed the cap was still shared.

        Asserted against the PARSED template, not the raw text: the header comment
        records that the split with `service-quotas-on-aws` is history, and a
        substring test would fail on that accurate note and pass once it was deleted.
        """
        self.assertEqual(set(self.tpl["Parameters"]),
                         {"ResourcePrefix", "ProxyBucket"})
        self.assertNotIn("Fn::ImportValue", self.raw)
        body = json.dumps(self.tpl)
        for gone in ("ServiceQuotas", "AcquireFunctionArn", "ServiceTokenTable"):
            self.assertNotIn(gone, body, gone)

    def test_no_request_default_is_a_stack_parameter(self):
        """
        Only two parameters survive, and the test is WHY: a parameter names
        something outside this stack that the deployer alone knows. A per-request
        default is not that — the code has to carry one anyway for the direct-SFN
        caller, so a parameter makes two spellings of one number where only the
        parameter can drift per-stack with nothing in the tree to show it.

        DefaultModelChain is the one that proves the cost rather than merely
        risking it: the gateway writes `model_chain` into the request, every key
        in that request is inside `workflow_id`, and the template shipped
        haiku-4-5 while eu-west-1 ran sonnet-4-6. Every chain-less request
        submitted against that stack therefore hashed differently from the same
        request against the tree — silent re-identification of finished work.

        Asserted on the ENV WIRING, not just the parameter list: re-adding a
        parameter and plumbing it to these variables is the regression, and
        deleting the parameter while leaving `Environment` intact would not even
        deploy. Pinned per-variable so the failure names the one that came back.

        Read out of the PARSED template rather than matched against the raw text,
        for the reason the parameter-list test above records: the Parameters block
        now carries a note naming each departed parameter and where it went, and a
        substring test would fail on that accurate note and pass once someone
        deleted it.
        """
        wired = set()
        for resource in self.tpl["Resources"].values():
            if resource.get("Type") != "AWS::Lambda::Function":
                continue
            env = (resource["Properties"].get("Environment") or {})
            wired |= set(env.get("Variables") or {})
        for var in ("DEFAULT_MODEL_CHAIN", "DEFAULT_SYNC_MODEL_CHAIN",
                    "MAX_CONCURRENT_DEFAULT", "BATCH_MIN_SAVINGS"):
            self.assertNotIn(var, wired, var)

        # ...and the code does carry each one, which is what makes removing the
        # env wiring a no-op rather than a behaviour change.
        self.assertEqual(input_schema.MAX_CONCURRENT_DEFAULT, 3)
        self.assertEqual(input_schema.BATCH_MIN_SAVINGS, 0.25)
        # haiku-4-5 -> sonnet-4-6 (user, 2026-09-17). A default-chain
        # change re-identifies every chain-less request — expected and
        # reviewable exactly because it arrives as a code change + a
        # .version bump, which is this commit.
        self.assertEqual(input_schema.DEFAULT_MODEL_CHAIN,
                         ["anthropic.claude-sonnet-4-6"])
        # Explicit-sync requests default to the cheaper Sonnet-led chain
        # (user, 2026-09-23: Sonnet 5 on-demand undercuts 4.6, but cannot
        # batch — so ONLY use_batch: false gets it; auto/true keep the
        # batch-capable default above). sonnet-5 -> sonnet-5-5 (user,
        # 2026-10-13, registry refresh): same price ($2/$10 per MTok,
        # verified via test/bedrock_pricing.py), same no-batch.
        self.assertEqual(input_schema.DEFAULT_SYNC_MODEL_CHAIN,
                         ["anthropic.claude-sonnet-5-5",
                          "anthropic.claude-sonnet-4-6"])
        # Bare id: the geo prefix names routing, not identity — resolve_model
        # strips it and the registry entry's inference_region re-derives it, so a
        # prefix here is decoration that reads like residency control.
        for model_id in (input_schema.DEFAULT_MODEL_CHAIN
                         + input_schema.DEFAULT_SYNC_MODEL_CHAIN):
            self.assertFalse(
                re.match(r"^(global|us|eu|apac)\.", model_id), model_id)

    def test_the_two_deploy_time_constants_are_read_where_a_json_cannot_reach(self):
        """
        ModelRegistryKey and DrainScheduleExpression left `Parameters` for
        `Mappings` rather than for input_schema.py, and this pins the reason so
        nobody "finishes the job" by moving them into a runtime config: both are
        read by CloudFormation itself while building a resource.

        The registry key is interpolated into an IAM policy resource ARN — an S3
        grant must name its object at deploy time, and a key read from a runtime
        JSON could not be scoped at all, which inverts the point of the grant.
        The schedule is an AWS::Events::Rule property, evaluated before any code
        runs.
        """
        body = json.dumps(self.tpl)
        self.assertIn("ModelRegistry", json.dumps(self.tpl["Mappings"]))
        # _model_configs/, not _settings/ — the proxy bucket may be shared with
        # an M2C v5 deployment whose core stack owns _settings/.
        self.assertEqual(
            self.tpl["Mappings"]["ModelRegistry"]["Object"]["Key"],
            "_model_configs/availablemodels.json")
        # Every 6 hours: the sweep is the missed-event backstop behind two
        # event triggers, so its rate is worst-case latency on a path that
        # almost never runs — and 6 h still gives a stranded row three more
        # sweeps inside the 24 h token budget.
        self.assertEqual(
            self.tpl["Mappings"]["Drain"]["Schedule"]["Expression"],
            "rate(6 hours)")
        # The grant and the reader come from the ONE mapping entry, so they
        # cannot disagree — the failure mode being excluded is a function reading
        # key A while its role is scoped to key B.
        self.assertGreaterEqual(body.count('"ModelRegistry"'), 2)
        # The three offline FALLBACK copies of the key, though, are plain strings
        # the mapping cannot reach — pin them to it, or a key move (like
        # _settings/ → _model_configs/, 2026-08-13) leaves a stale default that
        # only bites where the env var is absent, i.e. only offline.
        key = self.tpl["Mappings"]["ModelRegistry"]["Object"]["Key"]
        self.assertEqual(bedrock_lib.DEFAULT_REGISTRY_KEY, key)
        self.assertEqual(input_schema.MODEL_REGISTRY_KEY, key)
        gateway = _read(ROOT, "gateway_tool", "lambda_function.py")
        self.assertIn(f'"{key}"', gateway)

    def test_every_findinmap_names_three_levels_that_exist(self):
        """
        `Fn::FindInMap` takes EXACTLY three arguments — map, top-level key,
        second-level key — and a two-argument form is a **deploy-time** failure that
        every offline check here used to pass. The YAML parses, the drift check
        passes, the build packages, and CREATE/UPDATE then fails on a template that
        looked green.

        That is not hypothetical: `ModelRegistry` and `Drain` were added flat on
        2026-08-12 (`!FindInMap [ModelRegistry, Key]`) when the last two stack
        Parameters moved into Mappings, and only cfn-lint's E1011 caught it — which
        the build does not run. Hence the filler middle keys `Object` and `Schedule`,
        and hence this test.

        It also resolves each lookup, so a renamed mapping entry fails here rather
        than at deploy: a `!FindInMap [Lambda, OneCore, Memory]` against a mapping
        key spelled `1Core` is the same class of green-then-broken.
        """
        mappings = self.tpl["Mappings"]
        found = []

        def walk(node):
            if isinstance(node, dict):
                args = node.get("FindInMap")
                if isinstance(args, list):
                    found.append(args)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(self.tpl)
        # If this ever reaches 0 the test has gone vacuous — the template is built
        # on FindInMap for every name, memory and package.
        self.assertGreater(len(found), 20)
        for args in found:
            self.assertEqual(len(args), 3, args)
            map_name, top, second = args
            # Only literal keys can be checked. Nothing here uses a !Ref key, and
            # if something starts to, this skip is the place to look.
            if not all(isinstance(part, str) for part in args):
                continue
            self.assertIn(map_name, mappings, args)
            self.assertIn(top, mappings[map_name], args)
            self.assertIn(second, mappings[map_name][top], args)

    def test_each_inline_definition_still_matches_its_readability_copy(self):
        """
        The three ASLs live INLINE in the template — the template is what deploys —
        and `definitions*.json` are readability copies: lintable, diffable, and
        openable in a state machine editor on their own. Nothing but this test and
        the build's `verify_sfn_definition` keeps the copy honest.

        It has to be kept honest. A reader who edits `definitions_sync.json`,
        reviews that diff, and deploys would ship the template's OLDER body while
        believing the reviewed one went out — the exact failure a readability copy
        exists to prevent, not to introduce.

        An EXACT string compare, not a JSON compare: the two are maintained as
        literal copies, so identical-modulo-formatting is not the property claimed,
        and a whitespace-only divergence is the first sign the two are being edited
        independently. (The build's check is the looser JSON compare, so a
        re-indent breaks the test here and never a deploy.)
        """
        for resource, filename in (("InferenceStateMachine", "definitions.json"),
                                   ("SyncInferenceStateMachine",
                                    "definitions_sync.json"),
                                   ("BatchInferenceStateMachine",
                                    "definitions_batch.json")):
            inline = (self.tpl["Resources"][resource]["Properties"]
                      ["DefinitionString"])
            self.assertEqual(inline.rstrip("\n"),
                             _read(ROOT, filename).rstrip("\n"),
                             f"{filename} has drifted from {resource}'s inline "
                             "DefinitionString")

    def test_no_definition_token_survives_anywhere(self):
        """
        The tokens are gone and must stay gone. A reintroduced `%%DEFINITION%%`
        would now be substituted by nothing — the build no longer injects — and
        CloudFormation would be handed the literal `%%DEFINITION%%` as a state
        machine body, which it rejects well into a stack update.
        """
        for token in ("%%DEFINITION%%", "%%DEFINITION_SYNC%%",
                      "%%DEFINITION_BATCH%%"):
            self.assertNotIn(token, self.raw, token)

    def test_every_template_token_is_substituted_by_the_build(self):
        script = _read(ROOT, "build-deploy-s3-dist.sh")
        tokens = set(re.findall(r"%%[A-Z_]+%%", self.raw))
        self.assertTrue(tokens)
        for token in tokens:
            self.assertIn(token, script,
                          f"{token} appears in cfnstack.yaml but the build never "
                          "substitutes it — it would ship literally")


# ─────────────────────────────────────────────────────────────────────────────
# The state machine definitions — THREE files since the merge
#
# `definitions.json` is the orchestrator (the machine callers start);
# `definitions_sync.json` and `definitions_batch.json` are the two standalone
# children. Everything universal is asserted once, over all three; each machine's
# own shape gets its own class. A test written against one file only is how the
# `Release lease on failure` removal could have been left half-done — deleted from
# the sync child while an orphaned Catch still pointed at it.
# ─────────────────────────────────────────────────────────────────────────────

_ASL_FILES = {
    "orchestrator": "definitions.json",
    "sync": "definitions_sync.json",
    "batch": "definitions_batch.json",
}


class _AslCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {}
        cls.asl = {}
        for machine, filename in _ASL_FILES.items():
            cls.raw[machine] = _read(ROOT, filename)
            cls.asl[machine] = json.loads(cls.raw[machine])

    def states(self, machine):
        return self.asl[machine]["States"]

    def all_states(self):
        """`(machine, name, state)` for every state in all three machines."""
        for machine in _ASL_FILES:
            for name, state in self.states(machine).items():
                yield machine, name, state


class TestAllDefinitions(_AslCase):
    """Invariants that must hold in every one of the three machines."""

    def test_start_and_every_next_target_exists(self):
        for machine in _ASL_FILES:
            states = self.states(machine)
            names = set(states)
            for name, state in states.items():
                targets = []
                if "Next" in state:
                    targets.append(state["Next"])
                for branch in state.get("Choices") or []:
                    targets.append(branch["Next"])
                if "Default" in state:
                    targets.append(state["Default"])
                for catch in state.get("Catch") or []:
                    targets.append(catch["Next"])
                for target in targets:
                    self.assertIn(target, names, f"{machine}: {name} → {target}")
            self.assertIn(self.asl[machine]["StartAt"], names, machine)

    def test_each_parent_timeout_covers_its_own_child_worst_case(self):
        """
        The one piece of arithmetic that spans two files, and the repair tail is what
        made it non-obvious: the batch child's worst case is now `Submit batch`'s
        24 h park PLUS the repair child's own 28 h, so the orchestrator's batch branch
        has to clear ~52 h rather than the ~24 h it needed when the child was three
        states. Derived here instead of asserted as a literal, because a future
        `TimeoutSeconds` bump inside a child is exactly the edit that would leave the
        parent cutting the child off mid-repair — and a parent `States.Timeout` is the
        one error that carries no `Cause` to explain itself.
        """
        def worst_case(machine):
            # Sequential by construction: no `Parallel`, no `Map` (contract 6), and
            # the repair is reachable only from `Collect batch output`, so the longest
            # path is the sum of every timeout in the machine.
            return sum(state.get("TimeoutSeconds", 0)
                       for state in self.states(machine).values())

        for branch, child in (("Run sync inference", "sync"),
                              ("Run batch inference", "batch")):
            parent = self.states("orchestrator")[branch]["TimeoutSeconds"]
            self.assertGreater(
                parent, worst_case(child),
                f"{branch} ({parent}s) must exceed the {child} child's own worst "
                f"case ({worst_case(child)}s)")

        # And the repair state's own bound IS the orchestrator's sync bound, because
        # it is the same machine being started. Two different numbers here would mean
        # a repair is allowed more or less time than a first-class sync run.
        self.assertEqual(
            self.states("batch")["Run repair inference"]["TimeoutSeconds"],
            self.states("orchestrator")["Run sync inference"]["TimeoutSeconds"])

    def test_every_operation_the_asl_sends_is_routable(self):
        """
        The ASLs and the router are one contract in four files. An operation the
        router does not know raises a ValueError at runtime — after the lease is
        held. `Acquire` used to be exempt here because it belonged to the quota
        stack's zip; the semaphore is in-stack now, so there is no exemption and
        `StateAcquireLease` must be a real router key.
        """
        for machine, name, state in self.all_states():
            payload = (state.get("Parameters") or {}).get("Payload") or {}
            operation = payload.get("operation")
            if not operation:
                continue
            self.assertIn(operation, router._STATES, f"{machine}: {name}")

    def test_the_wait_for_task_token_states_use_the_context_object(self):
        """
        TWO dollar signs. `$.Task.Token` is a path into the state input and resolves
        to nothing; `$$.Task.Token` is the context object. The wrong one parks the
        execution on a token that was never issued, for the full TimeoutSeconds.

        The two waiters live in two DIFFERENT machines, which is the point of checking
        all three: the orchestrator holds none. Exactly TWO, and that is the whole
        saving of the nested-child repair — the batch machine used to carry a third,
        its own `Acquire repair lease`, duplicating the sync child's spine.
        """
        waiters = [(machine, name) for machine, name, state in self.all_states()
                   if state.get("Resource", "").endswith(".waitForTaskToken")]
        self.assertEqual(sorted(waiters),
                         [("batch", "Submit batch"),
                          ("sync", "Acquire lease")])
        for machine, name in waiters:
            state = self.states(machine)[name]
            self.assertEqual(state["Resource"],
                             "arn:aws:states:::lambda:invoke.waitForTaskToken")
            payload = state["Parameters"]["Payload"]
            self.assertEqual(payload["task_token.$"], "$$.Task.Token", name)
            # A token that outlives its TimeoutSeconds is a dead waiter someone has
            # to reap; an absent timeout is a year-long park.
            self.assertIn("TimeoutSeconds", state, name)

    def test_execution_arn_always_comes_from_the_context_object(self):
        """
        Never reconstructed from env + uuid. The semaphore's identity for a holder IS
        the execution ARN the terminal EventBridge event will carry, and a
        reconstructed one differing by a character makes trigger 3 a no-op.

        Note each machine reports its OWN `$$.Execution.Id`: the sync child's holder
        is keyed on the CHILD's arn, and the terminal rule fires on all three
        machines precisely so that arn is the one reclaimed.
        """
        seen = set()
        for machine, name, state in self.all_states():
            payload = (state.get("Parameters") or {}).get("Payload") or {}
            if "execution_arn.$" in payload:
                self.assertEqual(payload["execution_arn.$"], "$$.Execution.Id",
                                 f"{machine}: {name}")
                seen.add((machine, name))
        # An exact set, not a count floor: the loop above passes vacuously if a state
        # stops passing `execution_arn` at all. **No batch state is in it**, and that is
        # the shape of the repair: the batch machine starts the sync child rather than
        # leasing anything itself, so the holder is keyed on the REPAIR execution's own
        # arn — a grandchild — which the terminal rule covers because it watches the
        # sync machine, not because the batch machine named itself.
        self.assertEqual(seen, {
            ("orchestrator", "Validate & plan"),
            ("orchestrator", "Usage tracking"),
            ("sync", "Acquire lease"),
            ("sync", "Run inference"),
            ("sync", "Release lease"),
        })
        # `Usage tracking` no longer takes a child arn at all: with a repair the tree is
        # orchestrator → batch → sync, so a single scalar cannot name it. The states
        # walker discovers descendants from the history instead.
        usage = self.states("orchestrator")["Usage tracking"]
        self.assertNotIn("child_execution_arn.$", usage["Parameters"]["Payload"])

    def test_no_map_state_anywhere_in_any_machine(self):
        """
        ONE in-flight call per lease is contract, not guidance (§10 item 1): a Map
        with MaxConcurrency 4 inside a held span turns a cap of 3 into 12, silently,
        and the only symptom is Bedrock throttling. Checked in all three because the
        span the cap governs is now a child execution, and a Map in the ORCHESTRATOR
        would multiply whole child executions rather than Converse calls.
        """
        self.assertEqual([(machine, name) for machine, name, state
                          in self.all_states() if state.get("Type") == "Map"], [])
        for machine, raw in self.raw.items():
            self.assertNotIn("MaxConcurrency", raw, machine)

    def test_no_release_lease_on_failure_state_survives_anywhere(self):
        """
        The removed-behaviour audit for README.md §M3.2. Trigger 2 — the ASL
        `Catch` that released in-band — is deleted, following face-recognition. The
        invariant it enforced is re-established by the terminal-state EventBridge
        rule (trigger 3) plus the holder TTL (trigger 4); a leftover Catch pointing
        at a state that no longer exists is a `States.Runtime` failure at deploy.

        Asserted over state NAMES and `Catch` targets, not the raw text: the sync
        child's own Comment explains at length why the state is gone, so a substring
        test would fail on the accurate documentation and pass once it was deleted.
        """
        for machine in _ASL_FILES:
            self.assertNotIn("Release lease on failure", self.states(machine),
                             machine)
        for machine, name, state in self.all_states():
            for catch in state.get("Catch") or []:
                self.assertNotEqual(catch["Next"], "Release lease on failure",
                                    f"{machine}: {name}")

    def test_fail_states_use_static_causes(self):
        """A States.Timeout carries no Cause, and an intrinsic referencing an absent
        path fails the state — replacing an honest failure with a confusing one."""
        for machine, name, state in self.all_states():
            if state.get("Type") != "Fail":
                continue
            self.assertNotIn("States.", state.get("Cause", ""),
                             f"{machine}: {name}")
            self.assertNotIn(".$", json.dumps(list(state)), f"{machine}: {name}")

    def test_error_equals_never_names_a_raw_service_error_code(self):
        """
        Step Functions matches `errorType`, which for a Python Lambda is the
        exception CLASS name. `ErrorEquals: ["ThrottlingException"]` can never match
        a botocore ClientError — it is dead configuration that reads, in the console,
        as working back-pressure.

        `StepFunctions.` joins the reserved prefixes because the orchestrator's two
        `.sync:2` states are SDK integrations, not Lambda invocations, and their
        service-prefixed names ARE the errors Step Functions raises for them.
        `QuotaSaturated` is gone from the raised set: the semaphore never raises on
        a full cap now, it queues the task token (README.md §M2.1).
        """
        raised = {"ValidationException", "LambdaResourceTimeoutException",
                  "BedrockBatchQuotaExceeded", "BedrockInferenceFailed",
                  "BatchRecordsIncompleteException"}
        for machine, name, state in self.all_states():
            for entry in (state.get("Retry") or []) + (state.get("Catch") or []):
                for err in entry["ErrorEquals"]:
                    if err.startswith(("States.", "Lambda.", "Sandbox.",
                                       "StepFunctions.")):
                        continue
                    self.assertIn(err, raised, f"{machine}: {name}: {err}")

    def test_every_named_error_class_actually_exists(self):
        """The other half: a class that no longer exists makes the Retry dead in the
        opposite direction, and nothing raises."""
        from lib import custom_exceptions
        for machine, name, state in self.all_states():
            for entry in (state.get("Retry") or []) + (state.get("Catch") or []):
                for err in entry["ErrorEquals"]:
                    if err.startswith(("States.", "Lambda.", "Sandbox.",
                                       "StepFunctions.")):
                        continue
                    self.assertTrue(
                        hasattr(custom_exceptions, err),
                        f"{machine}: {name}: {err} is not in lib/custom_exceptions")
        for machine, name, state in self.all_states():
            if state.get("Type") == "Fail":
                self.assertTrue(hasattr(custom_exceptions, state["Error"]),
                                f"{machine}: {name}")

    def test_every_task_has_a_retry_ladder_for_its_own_integration(self):
        """
        A Lambda task retries `Lambda.*`; a `states:startExecution.sync:2` task
        cannot — Step Functions never raises a `Lambda.` error for it, so a Lambda
        ladder there is dead configuration on the two states that carry the whole
        run. Each integration gets the ladder that can actually fire.
        """
        for machine, name, state in self.all_states():
            if state.get("Type") != "Task":
                continue
            # The ErrorEquals entries alone, NOT the whole Retry dump: the
            # orchestrator's Comment explains why ExecutionAlreadyExists is absent,
            # so a text search would find it in the prose that documents its absence.
            named = {err for entry in state.get("Retry") or []
                     for err in entry["ErrorEquals"]}
            where = f"{machine}: {name}"
            if "states:startExecution" in state["Resource"]:
                self.assertIn("StepFunctions.SdkClientException", named, where)
                self.assertIn("StepFunctions.ThrottlingException", named, where)
                # A child that fails is retried at most twice under the SAME name,
                # and ExecutionAlreadyExists is deliberately NOT in the ladder — a
                # retry under a new name would pay for the work twice.
                self.assertNotIn("StepFunctions.ExecutionAlreadyExists", named,
                                 where)
            else:
                self.assertIn("Lambda.ServiceException", named, where)
                self.assertIn("Lambda.TooManyRequestsException", named, where)


class TestOrchestratorDefinition(_AslCase):
    """`definitions.json` — plans, routes, and collects usage. Runs no inference."""

    def setUp(self):
        self.st = self.states("orchestrator")

    def test_validate_and_plan_starts_it_and_has_no_catch(self):
        """Nothing has been acquired and no child exists yet, so a failure here
        should surface as itself rather than through a usage-tracking state."""
        self.assertEqual(self.asl["orchestrator"]["StartAt"], "Validate & plan")
        self.assertNotIn("Catch", self.st["Validate & plan"])

    def test_the_choice_routes_to_the_two_children_and_has_no_default(self):
        """
        No `Default`: an unmatched mode raises `States.NoChoiceMatched` and fails the
        execution, which is exactly what a `Fail` state here would produce and needs
        no state to say it.

        What is asserted is the absence of a `Default` **pointing at a child**. A
        fall-through to sync would run a 5000-record set through a serial lease-held
        loop that cannot finish inside the child's retry budget — a silent wrong-path
        run, which is strictly worse than the execution failing.
        """
        choice, = [state for state in self.st.values()
                   if state.get("Type") == "Choice"]
        branches = {entry["StringEquals"]: entry["Next"]
                    for entry in choice["Choices"]}
        self.assertEqual(branches, {"sync": "Run sync inference",
                                    "batch": "Run batch inference"})
        self.assertNotIn("Default", choice)

    def test_choose_mode_returns_only_the_two_modes_the_choice_handles(self):
        """
        The other half of having no `Default`: the Choice is exhaustive only because
        `choose_mode` is. A third return value would become a `NoChoiceMatched` in
        production on whichever request first triggered it, so the ASL's exhaustiveness
        is pinned to the function's, not assumed.
        """
        from lib.input_schema import choose_mode

        seen = set()
        for use_batch in (True, False, None):
            for can_batch in (True, False):
                for saving in (None, 0.0, 0.01, 5.0):
                    for tokens in (1, 10 ** 9):
                        mode, _ = choose_mode(
                            use_batch, can_batch, estimated_saving=saving,
                            estimated_input_tokens=tokens)
                        seen.add(mode)
        choice = self.states("orchestrator")["Choose mode"]
        handled = {c["StringEquals"] for c in choice["Choices"]}
        self.assertEqual(seen, handled)

    def test_the_two_branch_states_name_the_two_child_machines(self):
        """
        One substitution each, and nothing else: a branch pointing at the
        orchestrator's own ARN is an infinite recursion that bills per execution,
        and one pointing at the wrong child runs the wrong path with a valid plan.
        """
        arns = {name: state["Parameters"]["StateMachineArn"]
                for name, state in self.st.items()
                if "states:startExecution" in state.get("Resource", "")}
        self.assertEqual(arns, {
            "Run sync inference": "${SyncInferenceStateMachineArn}",
            "Run batch inference": "${BatchInferenceStateMachineArn}",
        })

    def test_the_children_are_named_with_a_mode_suffix(self):
        """
        `workflow_id` deliberately EXCLUDES mode (§3), so the same records through
        sync and through batch share one id. An unsuffixed child name would collide
        with its sibling's history on a mode flip, and `ExecutionAlreadyExists` on
        the child would look like the idempotency guard working.
        """
        self.assertEqual(
            self.st["Run sync inference"]["Parameters"]["Name.$"],
            "States.Format('{}-sync', $.data.workflow_id)")
        self.assertEqual(
            self.st["Run batch inference"]["Parameters"]["Name.$"],
            "States.Format('{}-batch', $.data.workflow_id)")

    def test_both_children_are_synchronous_and_bounded(self):
        """
        `.sync:2` because the orchestrator's own history is what `usage_tracking`
        walks, and a fire-and-forget child would leave it nothing to walk and the
        caller nothing to read. The timeout must EXCEED the child's own worst case
        so the child reports its own bound rather than being cut off mid-collect.
        """
        for name, floor in (("Run sync inference", 86400),
                            ("Run batch inference", 86400)):
            state = self.st[name]
            self.assertEqual(state["Resource"],
                             "arn:aws:states:::states:startExecution.sync:2")
            self.assertGreater(state["TimeoutSeconds"], floor, name)
            self.assertEqual(state["ResultPath"], "$.child", name)

    def test_the_child_result_is_selected_down_to_three_fields(self):
        """
        `$.Output.data` is the child's Summarize output, which is what makes
        `Usage tracking`'s return carry `records_succeeded` / `keys.response`
        unchanged — a consumer's `ResultSelector` on `$.Output.data` keeps working
        across the split. The raw `.sync:2` result would nest it a level deeper.
        """
        for name in ("Run sync inference", "Run batch inference"):
            selector = self.st[name]["ResultSelector"]
            self.assertEqual(selector, {"data.$": "$.Output.data",
                                        "execution_arn.$": "$.ExecutionArn",
                                        "status.$": "$.Status"}, name)

    def test_the_orchestrator_catches_nothing_and_has_no_fail_state(self):
        """
        Failures propagate as themselves. A `Catch` here could only re-raise through a
        `Fail`, whose `Error`/`Cause` must be STATIC — a `States.Timeout` carries no
        `Cause` and an intrinsic over an absent path fails the state — so catching in
        order to re-raise replaces a specific error (`BedrockInferenceFailed` with the
        per-record message, a plan `ValidationException`) with a generic one.

        The `Catch` that used to be here existed to record a failed run's Lambda cost:
        well under a cent, an estimate from history timestamps rather than billing
        data, and redundant with `_progress.json`, which already holds a priced
        response line per record before any failure. Lease cleanup never needed it —
        `QuotaTerminalRule` reclaims on the child's own terminal event and covers
        ABORTED, which no `Catch` can see.
        """
        for name, state in self.st.items():
            self.assertNotIn("Catch", state, name)
        self.assertEqual(
            [name for name, state in self.st.items()
             if state.get("Type") == "Fail"], [])

    def test_a_real_child_failure_is_not_retried_into_a_name_collision(self):
        """
        `States.TaskFailed` must stay OUT of both `.sync:2` retry ladders.

        It is what a genuine child failure raises. Retrying it re-issues
        `startExecution` under the SAME name — the name is
        `States.Format('{}-sync', workflow_id)`, fixed — so the retry returns
        `StepFunctions.ExecutionAlreadyExists`, and with no `Catch` to launder it that
        is what the caller sees instead of the child's real error. Only transport
        failures, where no execution was created, are safe to retry under a fixed name.
        """
        for name in ("Run sync inference", "Run batch inference"):
            named = {err for entry in self.st[name]["Retry"]
                     for err in entry["ErrorEquals"]}
            self.assertEqual(named, {"StepFunctions.SdkClientException",
                                     "StepFunctions.ThrottlingException"}, name)

    def test_usage_tracking_reads_the_child_summary(self):
        """
        `$.child.data` is the child's `Summarize` output. With the failure-path twin
        gone this is the only call site, and it always has a child summary to read —
        which is what lets `_bedrock_usage` treat a missing one as a reporting fault
        (`complete: False`) rather than a normal path.

        It takes **no** child arn, and that is the nested repair's one cost to this
        state: with a repair the descendant set is a TREE (orchestrator → batch → sync),
        so a single scalar cannot name it and a payload field would have to grow a level
        every time the graph does. The handler discovers descendants from the
        orchestrator's own history instead — `TaskSubmitted` events carry the child arn
        — which needs no ASL change to follow a deeper graph.
        """
        payload = self.st["Usage tracking"]["Parameters"]["Payload"]
        self.assertEqual(payload["data.$"], "$.child.data")
        self.assertNotIn("child_execution_arn.$", payload)
        self.assertNotIn("error.$", payload)

    def test_the_orchestrator_runs_no_inference_and_takes_no_lease(self):
        """
        It plans and routes. An `operation` here that touched Bedrock or the
        semaphore would put the cap and the call back in one execution, which is the
        split the three machines exist to make.
        """
        operations = {(state.get("Parameters") or {}).get("Payload", {})
                      .get("operation")
                      for state in self.st.values()}
        self.assertEqual(operations - {None},
                         {"StateValidateAndPlan", "StateUsageTracking"})

    def test_exactly_one_terminal_success_state(self):
        ends = [name for name, state in self.st.items() if state.get("End")]
        self.assertEqual(ends, ["Usage tracking"])


class TestSyncDefinition(_AslCase):
    """`definitions_sync.json` — the only machine that holds a lease."""

    def setUp(self):
        self.st = self.states("sync")

    def test_it_starts_at_the_acquire(self):
        """
        Acquired LATE by construction: every S3 read, record parse and validation
        happened in the orchestrator, so a queued execution holds no slot while it
        waits and contended work still makes progress.
        """
        self.assertEqual(self.asl["sync"]["StartAt"], "Acquire lease")

    def test_the_acquire_state_targets_this_stacks_own_function(self):
        """
        The merge's headline change. It used to be `${AcquireLambdaArn}` — a foreign
        stack's function, because a local acquire handler needs the unscopeable
        `states:SendTaskSuccess` grant. It is on the one workflow role now, which
        after the 2026-08-12 tier collapse CAN also call Bedrock — the isolation
        README.md §M2.4 narrowed to a role is gone entirely.
        """
        state = self.st["Acquire lease"]
        self.assertEqual(state["Parameters"]["FunctionName"],
                         "${InferenceLambdaArn}")
        payload = state["Parameters"]["Payload"]
        self.assertEqual(payload["operation"], "StateAcquireLease")
        # The scope comes from the plan, so the cap is per model and the ASL holds no
        # hardcoded resource name.
        self.assertEqual(payload["service.$"], "$.data.quota.service")
        self.assertEqual(payload["resource.$"], "$.data.quota.resource")
        self.assertEqual(payload["max_concurrent.$"],
                         "$.data.quota.max_concurrent")

    def test_the_acquire_discards_its_own_result(self):
        """
        `ResultPath: null` — the SendTaskSuccess payload is `{}` and the plan is the
        state input. Any other ResultPath would write an empty dict over a branch of
        the plan the next state reads.
        """
        self.assertIsNone(self.st["Acquire lease"]["ResultPath"])

    def test_no_state_in_this_machine_catches(self):
        """
        Trigger 2 is gone (README.md §M3.2), following face-recognition. A
        failure propagates to the orchestrator and the holder is reclaimed by the
        terminal-state rule on THIS execution's own FAILED event. A `Catch` added
        back here would release in-band and re-introduce the fourth path that has to
        agree with the other three.
        """
        self.assertEqual([name for name, state in self.st.items()
                          if "Catch" in state], [])

    def test_the_re_entry_retry_is_configured_for_the_checkpoint_loop(self):
        retries = self.st["Run inference"]["Retry"]
        reentry, = [entry for entry in retries
                    if "LambdaResourceTimeoutException" in entry["ErrorEquals"]]
        self.assertGreaterEqual(reentry["MaxAttempts"], 10)
        # BackoffRate 1: the raise is a deliberate yield, not a failure to back off
        # from, and an exponential delay would idle a held lease.
        self.assertEqual(reentry["BackoffRate"], 1)

    def test_the_span_is_acquire_run_release_summarize_in_that_order(self):
        """
        `Summarize` runs AFTER the release: summarizing is S3 and arithmetic, and
        holding a Bedrock concurrency slot through it caps throughput on work that
        needs no slot.
        """
        self.assertEqual(self.st["Acquire lease"]["Next"], "Run inference")
        self.assertEqual(self.st["Run inference"]["Next"], "Release lease")
        self.assertEqual(self.st["Release lease"]["Next"], "Summarize")

    def test_only_the_run_inference_state_uses_the_bedrock_function(self):
        """
        One ASL token, `${RunInferenceLambdaArn}`, and exactly one state uses it.
        This used to be a capability statement — that tier was the only holder of
        `bedrock:InvokeModel` — and since 2026-08-12 every tier can call Bedrock, so
        what is left is the routing: the serial Converse loop is the one state that
        runs on the `1-core` tier, and the rest of this machine is on `2-cores`.
        Still worth pinning, because a second state pointed at that token is a
        900-second timeout applied to something that does not need one.
        """
        targets = {name: state["Parameters"]["FunctionName"]
                   for name, state in self.st.items()
                   if state.get("Type") == "Task"}
        self.assertEqual(
            {name for name, arn in targets.items()
             if arn == "${RunInferenceLambdaArn}"}, {"Run inference"})
        self.assertEqual(set(targets.values()),
                         {"${InferenceLambdaArn}", "${RunInferenceLambdaArn}"})

    def test_exactly_one_terminal_success_state(self):
        ends = [name for name, state in self.st.items() if state.get("End")]
        self.assertEqual(ends, ["Summarize"])


class TestBatchDefinition(_AslCase):
    """`definitions_batch.json` — no lease anywhere on this path."""

    def setUp(self):
        self.st = self.states("batch")

    def test_it_starts_at_the_submit(self):
        self.assertEqual(self.asl["batch"]["StartAt"], "Submit batch")

    def test_it_is_four_states_and_the_repair_is_a_nested_child(self):
        """
        §8 as of 2026-08-12. The repair pass IS the sync child machine, started from
        here — not a second copy of its lease/checkpoint/release spine, which is what
        the previous shape carried (seven states, four of them duplicating the sync
        child). Structurally that means: this machine invokes NO run-inference
        function and holds NO semaphore state, and the whole repair is one Task.

        Asserted as an exact state set, not a subset: the failure mode being pinned is
        a spine growing back here state by state.
        """
        self.assertEqual(set(self.st), {"Submit batch", "Collect batch output",
                                        "Run repair inference", "Summarize"})
        self.assertEqual(self.st["Submit batch"]["Next"], "Collect batch output")

        repair = self.st["Run repair inference"]
        self.assertEqual(repair["Resource"],
                         "arn:aws:states:::states:startExecution.sync:2")
        self.assertEqual(repair["Parameters"]["StateMachineArn"],
                         "${SyncInferenceStateMachineArn}")
        # The plan is handed over whole — the child is startable given exactly this.
        self.assertEqual(set(repair["Parameters"]["Input"]),
                         {"uuid.$", "input.$", "data.$"})
        # A distinct name from the -sync sibling a mode flip on the same workflow_id
        # would create, and the console then shows which pass an execution was.
        self.assertIn("-repair", repair["Parameters"]["Name.$"])

        # No lease, no bedrock function, no Map: every task here is on the shared
        # workflow Lambda or is the nested child.
        functions = {(state.get("Parameters") or {}).get("FunctionName")
                     for state in self.st.values()}
        self.assertNotIn("${RunInferenceLambdaArn}", functions)
        self.assertNotIn("Lease", json.dumps(list(self.st)))

    def test_the_repair_is_reachable_only_by_a_catch_and_runs_exactly_once(self):
        """
        A `Catch`, not a `Choice`, and the difference is deliberate. A `.sync:2` child
        failure reaches its parent as `States.TaskFailed`, never as the child's own
        error name, so an equivalent Catch one level up in the ORCHESTRATOR could not
        tell a repairable batch from any other failure — only a Catch on the Lambda
        task can, because Step Functions matches the Python exception class name.

        That it renders as a red state in the console is the accepted cost and half the
        reason: a run that had a problem stays legible in the execution graph months
        later, where a boolean on a payload has to be dug out of the history.

        ONCE by construction: nothing has an edge back to `Collect batch output`, so
        there is no path that can repair twice. A record that fails the repair too
        stays a FAILED line.
        """
        collect = self.st["Collect batch output"]
        catch, = collect["Catch"]
        self.assertEqual(catch["ErrorEquals"], ["BatchRecordsIncompleteException"])
        self.assertEqual(catch["Next"], "Run repair inference")
        self.assertEqual(collect["Next"], "Summarize")

        # No Choice state decides this, and no state routes back to Collect.
        self.assertEqual([n for n, s in self.st.items()
                          if s.get("Type") == "Choice"], [])
        targets = [t for name, state in self.st.items()
                   for t in ([state.get("Next")] +
                             [c["Next"] for c in state.get("Catch") or []])
                   if t and name != "Submit batch"]
        self.assertNotIn("Collect batch output", targets)

    def test_both_catches_carry_an_explicit_nested_result_path(self):
        """
        The design breaks without this. A `Catch`'s default `ResultPath` is `$`, which
        replaces the whole state input with the error object — so the repair would be
        started with no plan, no keys and no `workflow_id`. A NESTED path merges
        instead, and there is something to merge into because a caught error replaces
        the state's RESULT, not its INPUT: `Submit batch` used `ResultPath`, not
        `OutputPath`, so `$.data` is still the plan on the caught path.
        """
        self.assertEqual(self.st["Submit batch"]["ResultPath"], "$.batch_result")
        for name, expected in (("Collect batch output", "$.data.batch_incomplete"),
                               ("Run repair inference", "$.data.repair_error")):
            catch, = self.st[name]["Catch"]
            self.assertEqual(catch.get("ResultPath"), expected, name)
            self.assertNotEqual(catch.get("ResultPath"), "$", name)

    def test_a_failed_repair_degrades_and_never_reaches_a_fail_state(self):
        """
        The restated invariant. "No `Catch` anywhere" was about not routing every state
        into a generic `Fail` terminal, which trades a specific error for a static one;
        both `Catch`es here route to `Summarize`, so the property that actually matters
        is that no `Catch` target is a `Fail` — and there are no `Fail` states at all.

        Degrading rather than failing is a perverse-incentive fix: with repair OFF a
        95/100 batch is a SUCCEEDED execution with five FAILED lines, so a failed
        repair must not turn that same run into a FAILED execution when
        `response.jsonl` is already complete on S3 either way.
        """
        catch, = self.st["Run repair inference"]["Catch"]
        self.assertEqual(catch["ErrorEquals"], ["States.ALL"])
        self.assertEqual(catch["Next"], "Summarize")
        for machine, name, state in self.all_states():
            for entry in state.get("Catch") or []:
                target = self.states(machine)[entry["Next"]]
                self.assertNotEqual(target.get("Type"), "Fail",
                                    f"{machine}: {name} → {entry['Next']}")

    def test_no_retry_ladder_anywhere_can_swallow_a_control_flow_exception(self):
        """
        Step Functions evaluates `Retry` BEFORE `Catch`, so a ladder naming
        `States.ALL` would re-read the whole batch output three times before the repair
        it exists to trigger ever fired — and on the third failure the caller would see
        a retry exhaustion instead of a repaired run.

        Self-enforcing as written, which is why this is cheap to keep: every ladder in
        all three machines names only prefixed integration errors plus
        `LambdaResourceTimeoutException`, and a Python class name can never collide
        with `Lambda.` or `StepFunctions.`. `States.ALL` in a ladder is the only way to
        break it, so that is what is banned.
        """
        control_flow = {"BatchRecordsIncompleteException", "States.ALL"}
        for machine, name, state in self.all_states():
            for entry in state.get("Retry") or []:
                for err in entry["ErrorEquals"]:
                    self.assertNotIn(err, control_flow, f"{machine}: {name}: {err}")

    def test_the_submit_parks_for_a_full_day(self):
        """A quota-shaped refusal returns normally with `PENDING` and the task stays
        parked until a drain submits it, so the timeout is sized to the QUEUE, not to
        a job's runtime."""
        self.assertEqual(self.st["Submit batch"]["TimeoutSeconds"], 86400)
        self.assertEqual(self.st["Submit batch"]["ResultPath"], "$.batch_result")

    def test_collect_reads_the_notifiers_payload(self):
        """The notifier DELETES the row after sending, so by the time this state runs
        the table lookup is usually a miss — `job_arn` has to arrive in the token
        payload."""
        payload = self.st["Collect batch output"]["Parameters"]["Payload"]
        self.assertEqual(payload["batch_result.$"], "$.batch_result")
        self.assertEqual(self.st["Submit batch"]["Next"], "Collect batch output")

    def test_exactly_one_terminal_success_state(self):
        """
        ONE terminal state for all three outcomes — clean batch, repaired batch, failed
        repair — and that is safe only because `Summarize` is a pure idempotent function
        of `response.jsonl` plus the plan. The repair child writes `summary.json` too,
        from its own Summarize; this one runs after it over the same `response.jsonl`
        and is the only one of the two that can see `$.data.repair`. Last writer wins,
        and both writers agree on everything but the repair fields.
        """
        ends = [name for name, state in self.st.items() if state.get("End")]
        self.assertEqual(ends, ["Summarize"])
        # Every non-terminal path converges here, which is what makes the sentence above
        # the whole correctness argument rather than one case of three.
        self.assertEqual(
            {state.get("Next") for state in self.st.values()} |
            {c["Next"] for state in self.st.values()
             for c in state.get("Catch") or []},
            {"Collect batch output", "Run repair inference", "Summarize", None})


# ─────────────────────────────────────────────────────────────────────────────
# micro-workflow-bundle contract adoption (2026-08-11) — the shared identity module, the
# idempotency token, and handle-only submits
# ─────────────────────────────────────────────────────────────────────────────

class TestSharedIdentityAdoption(unittest.TestCase):
    """
    `lib/identity.py` delegates to the copied `workflow_identity.py`
    (micro-workflow-bundle/README.md §C2). Determinism and key-order independence hold equally well
    for a WRONG preimage — only a fixed input mapped to a fixed output string
    catches "the preimage changed", the failure mode that is both silent and
    expensive: old and new code disagree about the identity of every request,
    and both runs SUCCEED.
    """

    # Known-answer vector. If this fails: an unintended change means fix the
    # code, not the vector; an intended change means update the vector AND bump
    # `.version`, because deployed code now disagrees with this code about the
    # identity of the same request. (Re-pinned once, 2026-08-18, riding the
    # max_tokens retirement's bump: the field left both the request options
    # and every parsed record, so the fixtures — which mirror what the
    # gateway actually hashes — changed shape with it.)
    GOLDEN_REQUEST = {
        "model_chain": ["us.anthropic.claude-sonnet-5-v1:0"],
        "options": {"temperature": 0.0},
        "tool_version": "ignored",
        "output": {"bucket": "b", "prefix": "p"},
        "mode": "sync",
    }
    GOLDEN_RECORDS = [
        {"record_id": "r0", "system": None,
         "messages": [{"role": "user", "content": [{"text": "Hi"}]}]},
    ]
    GOLDEN_DIGEST = ("871a628355bc0c15298bcb5870af5a7c"
                     "ffb8409caeaee55cba42a3274cde2445")
    GOLDEN_ID = "8a26d612-9dcd-5b4e-8251-199f07160682"
    GOLDEN_ID_TOKENED = "d17cb5ee-36bd-5514-92c1-f890a21aa6ce"  # nosemgrep: generic.secrets.gitleaks.generic-api-key

    def test_the_preimage_is_pinned(self):
        self.assertEqual(identity.digest_records(self.GOLDEN_RECORDS),
                         self.GOLDEN_DIGEST)
        self.assertEqual(
            identity.compute_workflow_id(
                self.GOLDEN_REQUEST, records_digest=self.GOLDEN_DIGEST,
                tool_version="1.0.0-test"),
            self.GOLDEN_ID)

    def test_a_token_narrows_the_id_and_a_fresh_token_is_the_rerun(self):
        tokened = identity.compute_workflow_id(
            self.GOLDEN_REQUEST, records_digest=self.GOLDEN_DIGEST,
            tool_version="1.0.0-test", idempotency_token="rerun-1")  # nosec B106
        self.assertEqual(tokened, self.GOLDEN_ID_TOKENED)
        self.assertNotEqual(tokened, self.GOLDEN_ID)
        # Same token, different records: still re-runs — the token NARROWS the
        # parameter hash rather than replacing it, so a reused token can never
        # hand back the wrong run's outputs.
        other = identity.compute_workflow_id(
            self.GOLDEN_REQUEST, records_digest="d" * 64,
            tool_version="1.0.0-test", idempotency_token="rerun-1")  # nosec B106
        self.assertNotEqual(tokened, other)

    def test_the_two_copies_are_byte_identical(self):
        """The copies ARE the distribution mechanism (flat gateway zip, no lib/).
        The bundle suite compares across stacks; this compares this stack's own
        two — the gateway's flat copy and lib/'s container copy."""
        gateway = _read(ROOT, "gateway_tool", "workflow_identity.py")
        container = _read(ROOT, "lib", "workflow_identity.py")
        self.assertEqual(gateway, container)

    def test_identity_delegates_rather_than_hand_rolling(self):
        source = _code(ROOT, "lib", "identity.py")
        self.assertIn("workflow_identity.compute_workflow_id", source)
        self.assertNotIn("uuid.uuid5", source)
        self.assertNotIn("NAMESPACE_URL", source)

    def test_the_gateway_zip_members_are_guarded(self):
        """`zip` exits 0 when only SOME members are missing (micro-workflow-bundle/README.md §C5) —
        the build script must check each member with [ -f ] before zipping.
        output_contract.json joined the members with the 2026-08-14 registration:
        the gateway degrades gracefully without it, so THIS guard is the only
        thing standing between a dropped contract and a fleet-wide silent
        `describe_tool` regression."""
        script = _read(ROOT, "build-deploy-s3-dist.sh")
        self.assertIn("members=(lambda_function.py callback_waker.py "
                      "callback_registry.py workflow_identity.py "
                      "output_contract.json input_examples.json)", script)
        self.assertIn('[ ! -f "gateway_tool/$f" ]', script)


class _GatewayCase(unittest.TestCase):
    """Imports the gateway the way its flat zip resolves it: gateway_tool/ on the
    path, `workflow_identity` a top-level module. Loaded once; per-test state is
    reset through the mocked clients."""

    @classmethod
    def setUpClass(cls):
        # gateway_tool/ itself goes on the path — the zip is flat, so at runtime
        # `import workflow_identity` is a TOP-LEVEL import (the trap the bundle
        # README.md documents: a suite with only ROOT on the path dies on the
        # gateway's own import while the deployed Lambda works fine). The module
        # has no package, so it is loaded by file path.
        gateway_dir = os.path.join(ROOT, "gateway_tool")
        if gateway_dir not in sys.path:
            sys.path.insert(0, gateway_dir)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gateway_lambda", os.path.join(gateway_dir, "lambda_function.py"))
        cls.gateway = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gateway)

    def setUp(self):
        self.s3_calls = []
        self.sfn = mock.MagicMock()
        self.sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123456789012:execution:sm:x"}
        self.s3 = mock.MagicMock()
        self._patches = [
            mock.patch.object(self.gateway, "_sfn", self.sfn),
            mock.patch.object(self.gateway, "_s3", self.s3),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def submit(self, **overrides):
        event = {
            "records": [{"record_id": "r0",
                         "messages": [{"role": "user",
                                       "content": [{"text": "Hi"}]}]}],
            "model_chain": ["us.anthropic.claude-sonnet-5-v1:0"],
            "use_batch": False,
        }
        event.update(overrides)
        return self.gateway._run_inference(event)

    def absorb(self, status="SUCCEEDED", **overrides):
        """A submit whose StartExecution collides with an existing run."""
        self.sfn.start_execution.side_effect = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "taken"}},
            "StartExecution")
        self.sfn.describe_execution.return_value = {
            "status": status, "executionArn":
                "arn:aws:states:us-east-1:123456789012:execution:sm:x",
            "cause": "boom" if status in ("FAILED", "ABORTED") else None,
        }
        # current_state walks execution history on a RUNNING absorb.
        self.sfn.get_execution_history.return_value = {"events": []}
        return self.submit(**overrides)


class TestGatewayTokenContract(_GatewayCase):
    """micro-workflow-bundle/README.md §C2 — the three ids at this gateway's submit."""

    def test_the_retired_uuid_is_rejected_on_submit(self):
        result = self.submit(uuid="some-correlation-id")
        self.assertTrue(result.get("error"))
        self.assertEqual(result["errorCode"], "INVALID_PARAMETER")
        self.assertIn("idempotency_token", result["message"])
        self.sfn.start_execution.assert_not_called()
        self.s3.put_object.assert_not_called()   # rejected BEFORE any write

    def test_a_malformed_token_is_rejected(self):
        result = self.submit(idempotency_token="bad token with spaces")  # nosec B106
        self.assertTrue(result.get("error"))
        self.assertEqual(result["errorCode"], "INVALID_PARAMETER")
        self.sfn.start_execution.assert_not_called()

    def test_a_fresh_token_changes_the_id_and_absence_dedups_on_parameters(self):
        base = self.submit()
        tokened = self.submit(idempotency_token="rerun-20260811")  # nosec B106
        again = self.submit()
        self.assertNotEqual(base["workflow_id"], tokened["workflow_id"])
        self.assertEqual(base["workflow_id"], again["workflow_id"])

    def test_the_token_never_reaches_the_state_machine_and_is_never_echoed(self):
        result = self.submit(idempotency_token="rerun-20260811")  # nosec B106
        self.assertNotIn("idempotency_token", json.dumps(result))
        self.assertNotIn("rerun-20260811", json.dumps(result))
        _, kwargs = self.sfn.start_execution.call_args
        self.assertNotIn("rerun-20260811", kwargs["input"])
        self.assertNotIn("idempotency_token", kwargs["input"])
        # ...and not into the S3 request objects either.
        for call in self.s3.put_object.call_args_list:
            self.assertNotIn(b"rerun-20260811", call.kwargs["Body"])

    def test_the_sfn_uuid_is_the_workflow_id(self):
        """The third id: the SFN input's `uuid` is this execution's LOCAL id —
        always the derived workflow_id, never a caller-supplied value."""
        result = self.submit(idempotency_token="rerun-20260811")  # nosec B106
        _, kwargs = self.sfn.start_execution.call_args
        sfn_input = json.loads(kwargs["input"])
        self.assertEqual(sfn_input["uuid"], result["workflow_id"])
        self.assertEqual(kwargs["name"], result["workflow_id"])

    def test_a_poll_accepts_a_stray_token_instead_of_failing_the_caller(self):
        self.sfn.describe_execution.side_effect = None
        self.sfn.describe_execution.return_value = {
            "status": "RUNNING",
            "executionArn":
                "arn:aws:states:us-east-1:123456789012:execution:sm:x"}
        self.sfn.get_execution_history.return_value = {"events": []}
        arn = "arn:aws:states:us-east-1:123456789012:execution:sm:x"
        for stray in ({"idempotency_token": "tok-1"}, {"uuid": "u-1"}):  # nosec B105
            with self.assertLogs(self.gateway.logger, level="WARNING") as logs:
                result = self.gateway._get_job_status({"job_id": arn, **stray})
            self.assertFalse(result.get("error"), stray)
            self.assertEqual(result["status"], "RUNNING")
            self.assertTrue(any("ignored" in line for line in logs.output))


class TestStagedRecords(_GatewayCase):
    """
    `records_s3uri` — the large-set transport (added 2026-08-14, from review):
    inline records ride the Lambda invoke event, capped at 6 MB, which is ~20
    base64 images TOTAL per request. The staged file must be TRANSPORT ONLY:
    identical records inline and staged must produce the same workflow_id, or
    the transport choice would re-identify (and re-pay for) identical work.
    """
    RECORDS = [{"record_id": "r0",
                "messages": [{"role": "user", "content": [{"text": "Hi"}]}]},
               {"record_id": "r1",
                "messages": [{"role": "user", "content": [{"text": "Yo"}]}]}]

    def _stage(self, body: bytes):
        # A .read(amt)-capable body — the streaming reader consumes the
        # object in chunks; BytesIO honours the same contract as
        # botocore's StreamingBody.
        import io
        self.s3.get_object.return_value = {"Body": io.BytesIO(body)}
        self.s3.create_multipart_upload.return_value = {"UploadId": "mpu-1"}
        self.s3.upload_part.return_value = {"ETag": '"etag"'}

    def test_staged_and_inline_records_are_one_workflow_id(self):
        inline = self.submit(records=self.RECORDS)
        jsonl = "".join(json.dumps(r) + "\n" for r in self.RECORDS)
        self._stage(jsonl.encode())
        staged = self.submit(records=None, records_s3uri="s3://b/micro-workflow/x/r.jsonl")
        self.assertEqual(staged["workflow_id"], inline["workflow_id"])
        self.assertEqual(staged["records_total"], 2)
        self.s3.get_object.assert_called_once_with(
            Bucket="b", Key="micro-workflow/x/r.jsonl")

    def test_a_json_array_body_is_accepted_too(self):
        self._stage(json.dumps(self.RECORDS).encode())
        result = self.submit(records=None, records_s3uri="s3://b/k.json")
        self.assertEqual(result["records_total"], 2)

    def test_both_sources_at_once_is_an_error_not_a_preference(self):
        result = self.submit(records_s3uri="s3://b/k.jsonl")
        self.assertEqual(result["errorCode"], "INVALID_PARAMETER")
        self.sfn.start_execution.assert_not_called()

    def test_an_unreadable_staged_object_names_the_grant_constraint(self):
        self.s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject")
        result = self.submit(records=None, records_s3uri="s3://other/k.jsonl")
        self.assertEqual(result["errorCode"], "S3_READ_FAILED")
        self.assertIn("micro-workflow/*", result["message"])

    def test_malformed_staged_bodies_fail_loudly(self):
        for name, body, uri in (
                ("not jsonl", b"{broken\n", "s3://b/k.jsonl"),
                ("empty", b"   ", "s3://b/k.jsonl"),
                ("bad uri", b"", "not-an-s3-uri")):
            with self.subTest(name):
                self._stage(body)
                result = self.submit(records=None, records_s3uri=uri)
                self.assertEqual(result["errorCode"], "INVALID_PARAMETER")

    # ── The streaming staged path (2026-08-23: a real 175 MB records.jsonl
    # pinned the old whole-file path at its memory ceiling until timeout) ──

    def _streamed_bytes(self) -> bytes:
        """Reassemble what the streaming pass uploaded, in part order."""
        parts = sorted(self.s3.upload_part.call_args_list,
                       key=lambda c: c.kwargs["PartNumber"])
        return b"".join(c.kwargs["Body"] for c in parts)

    def test_streamed_bytes_equal_the_inline_serialization(self):
        """The staging object's bytes must be EXACTLY what the inline path
        would have put at request.jsonl — the server-side copy makes them the
        final object, so a drift here is a corrupted audit record."""
        jsonl = "".join(json.dumps(r) + "\n" for r in self.RECORDS)
        self._stage(jsonl.encode())
        result = self.submit(records=None,
                             records_s3uri="s3://b/micro-workflow/x/r.jsonl")
        self.assertNotIn("errorCode", result)
        expected = "".join(self.gateway._record_line(
            {"record_id": r["record_id"], "system": r.get("system"),
             "messages": r["messages"]}) for r in self.RECORDS)
        self.assertEqual(self._streamed_bytes(), expected.encode("utf-8"))
        # ...and the final object is a server-side COPY of the staging key,
        # which is then deleted — the gateway never holds the file.
        copy = self.s3.copy_object.call_args.kwargs
        self.assertTrue(copy["Key"].endswith("/request.jsonl"))
        self.assertEqual(copy["CopySource"]["Key"],
                         self.s3.delete_object.call_args.kwargs["Key"])
        self.assertIn("_staging/", copy["CopySource"]["Key"])

    def test_a_mid_stream_refusal_aborts_the_staging_upload(self):
        dupe = [self.RECORDS[0], self.RECORDS[0]]
        self._stage("".join(json.dumps(r) + "\n" for r in dupe).encode())
        result = self.submit(records=None,
                             records_s3uri="s3://b/micro-workflow/x/r.jsonl")
        self.assertEqual(result["errorCode"], "INVALID_PARAMETER")
        self.assertIn("duplicate record_id", result["message"])
        self.s3.abort_multipart_upload.assert_called_once()
        self.s3.complete_multipart_upload.assert_not_called()
        self.sfn.start_execution.assert_not_called()

    def test_lines_split_across_stream_chunks_reassemble(self):
        """A record boundary never aligns with an S3 read boundary in
        practice — force pathological 7-byte reads and require the identical
        workflow_id the inline submission produces."""
        inline = self.submit(records=self.RECORDS)
        jsonl = "".join(json.dumps(r) + "\n" for r in self.RECORDS)
        self._stage(jsonl.encode())
        with mock.patch.object(self.gateway, "_STAGED_CHUNK_BYTES", 7):
            staged = self.submit(
                records=None,
                records_s3uri="s3://b/micro-workflow/x/r.jsonl")
        self.assertEqual(staged["workflow_id"], inline["workflow_id"])
        self.assertEqual(staged["records_total"], 2)


class _AgentToolContext:
    """The slice of Lambda context _resolve_tool reads: the AgentCore tool name
    under client_context.custom, target-prefixed the way the gateway ships it."""

    def __init__(self, tool: str):
        custom = {"bedrockAgentCoreToolName": f"bedrock-inference___{tool}"}
        self.client_context = type("Ctx", (), {"custom": custom})()


class TestOutputContractSurface(_GatewayCase):
    """
    The output-contract rollout (served from gateway_tool/output_contract.json),
    joined with registration: describe_tool serves gateway_tool/output_contract.json
    whole, each outputs[] entry carries the contract's one-line description as
    `desc` (omitted, never fabricated), and the contract's keys are exactly the
    label set _parse_succeeded_output can emit — the bundle's TestOutputContract
    pins the same parity from the outside.
    """

    SUCCEEDED_OUTPUT = {"data": {
        "workflow_id": "wid", "mode": "sync", "records_total": 1,
        "records_succeeded": 1, "records_failed": 0,
        "output": {"bucket": "b", "prefix": "micro-workflow/bedrock-inference/wid"},
        "keys": {"response": "micro-workflow/bedrock-inference/wid/response.jsonl",
                 "summary": "micro-workflow/bedrock-inference/wid/summary.json"},
        "usage_tracking": {
            "usage_key": "micro-workflow/bedrock-inference/wid/usage_tracking.json"},
    }}

    def _status(self):
        self.sfn.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "executionArn": ("arn:aws:states:us-east-1:123456789012:"
                             "execution:sm:wid"),
            "output": json.dumps(self.SUCCEEDED_OUTPUT),
        }
        return self.gateway._get_job_status(
            {"job_id": "arn:aws:states:us-east-1:123456789012:execution:sm:wid"})

    def test_describe_tool_serves_the_contract_it_shipped_with(self):
        resp = self.gateway.lambda_handler(
            {}, _AgentToolContext("describe_tool"))
        contract = resp["output_contract"]
        self.assertEqual(contract["target"], "bedrock-inference")
        self.assertEqual(contract["contract_version"], 1)
        # version and contract ride the same zip, so they cannot drift apart
        self.assertEqual(resp["version"], self.gateway.TOOL_VERSION)

    def test_contract_keys_are_exactly_the_emitted_label_set(self):
        result = self._status()
        emitted = {entry["label"] for entry in result["outputs"]}
        self.assertEqual(emitted, {"response", "summary", "usage"})
        self.assertEqual(
            set(self.gateway._OUTPUT_CONTRACT["outputs"]), emitted,
            "contract keys and emitted labels must be the SAME set — an "
            "unadvertised output and a contract entry for a removed one are "
            "both drift")

    def test_every_entry_joins_its_desc_and_the_status_carries_usage_uri(self):
        result = self._status()
        for entry in result["outputs"]:
            self.assertIn("desc", entry, entry["label"])
            self.assertNotIn("\n", entry["desc"], entry["label"])
        self.assertTrue(result["usage_uri"].endswith("usage_tracking.json"))
        self.assertTrue(result["output_s3_prefix"].endswith("/wid"))

    def test_a_missing_contract_degrades_desc_and_describe_never_the_status(self):
        with mock.patch.object(self.gateway, "_OUTPUT_CONTRACT", None):
            result = self._status()
            self.assertEqual({e["label"] for e in result["outputs"]},
                             {"response", "summary", "usage"})
            self.assertTrue(all("desc" not in e for e in result["outputs"]))
            resp = self.gateway.lambda_handler(
                {}, _AgentToolContext("describe_tool"))
            self.assertNotIn("output_contract", resp)


class TestSubmitReturnsAHandleNotResults(_GatewayCase):
    """micro-workflow-bundle/README.md §C3 — an absorbed resubmit differs from a fresh start only in
    `status` (+ message, + error on terminal failure), and NEVER carries the
    existing run's results. The old behaviour inlined
    `_parse_succeeded_output(existing)` on the SUCCEEDED branch — the exact
    shape tts reverted on 2026-08-07."""

    # `mode` left the handle with §5.6: it is DECIDED at plan time now (the
    # dollar guard needs the registry), so a submit cannot truthfully echo it —
    # the summary's inference_mode/mode_reason is where it lives.
    HANDLE_KEYS = {"job_id", "status", "workflow_id", "records_total",
                   "output_s3_prefix", "request_s3uri"}
    RESULT_KEYS = {"records_succeeded", "records_failed", "outputs",
                   "summary_s3uri", "response_s3uri", "bedrock_usage",
                   "inference_mode", "failures"}

    def test_a_fresh_submit_returns_a_handle(self):
        result = self.submit()
        self.assertEqual(set(result), self.HANDLE_KEYS)
        self.assertEqual(result["status"], "RUNNING")

    def test_an_absorbed_success_is_a_handle_with_a_message_and_no_results(self):
        result = self.absorb("SUCCEEDED")
        self.assertEqual(result["status"], "SUCCEEDED")
        # Byte-pinned across every bundle gateway; asserted verbatim there.
        self.assertIn("this call did not resubmit", result["message"])
        self.assertFalse(set(result) & self.RESULT_KEYS,
                         f"result fields leaked: {set(result) & self.RESULT_KEYS}")
        self.assertNotIn("error", result)

    def test_an_absorbed_terminal_failure_carries_its_cause(self):
        result = self.absorb("FAILED")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"], "boom")
        self.assertIn("message", result)
        self.assertFalse(set(result) & self.RESULT_KEYS)

    def test_no_submit_branch_carries_current_state(self):
        """Changed 2026-08-14 with bundle registration: an absorbed resubmit
        must carry EXACTLY a fresh start's fields (+ message, + error) — the
        bundle pins the set equality across every gateway — and a fresh start
        has no current_state, so neither may an absorbed one. The poll is
        where current_state lives."""
        for status in ("RUNNING", "SUCCEEDED", "FAILED"):
            self.sfn.start_execution.side_effect = None
            self.assertNotIn("current_state", self.absorb(status))

    def test_an_absorbed_resubmit_differs_from_fresh_only_in_status_extras(self):
        fresh = self.submit()
        self.sfn.start_execution.side_effect = None
        absorbed = self.absorb("SUCCEEDED")
        self.assertEqual(set(absorbed) - set(fresh), {"message"})
        for key in self.HANDLE_KEYS - {"status", "job_id"}:
            self.assertEqual(fresh[key], absorbed[key], key)

    def test_no_response_carries_a_uuid_field(self):
        """The retired spelling must not come back as an echo either — the
        workflow_id is the run's one name."""
        for result in (self.submit(),):
            self.assertNotIn("uuid", result)
        self.sfn.start_execution.side_effect = None
        self.assertNotIn("uuid", self.absorb("SUCCEEDED"))

    def test_inference_region_reaches_the_request_and_the_hash(self):
        """FOUND LIVE (2026-08-11): _build_request never forwarded the field, so
        an overridden submit hashed identically to the un-overridden one and was
        silently ABSORBED into it — the exact silent-global outcome the override
        exists to prevent. The field must land in input_params.json (the states read
        it from there) and therefore in the workflow_id."""
        base = self.submit()
        routed = self.submit(inference_region="eu")
        self.assertNotEqual(base["workflow_id"], routed["workflow_id"])
        body = json.loads(self.s3.put_object.call_args_list[-1]
                          .kwargs["Body"].decode())
        self.assertEqual(body.get("inference_region"), "eu")

    def test_chainless_default_is_mode_aware(self):
        """Chain-less requests default by requested mode (user, 2026-09-23):
        an EXPLICIT use_batch false gets the cheaper Sonnet-5-led sync chain;
        absent (auto may still batch) and true keep the batch-capable 4.6
        default — choose_mode keys on the PRIMARY model's batch capability,
        so a Sonnet-5-led auto default would silently resolve every auto
        request to sync and kill the auto->batch savings path."""
        def chain_of(**ov):
            self.submit(model_chain=None, **ov)
            body = json.loads(self.s3.put_object.call_args_list[-1]
                              .kwargs["Body"].decode())
            return body["model_chain"]
        self.assertEqual(chain_of(use_batch=False),
                         ["anthropic.claude-sonnet-5-5",
                          "anthropic.claude-sonnet-4-6"])
        self.assertEqual(chain_of(use_batch=None),
                         ["anthropic.claude-sonnet-4-6"])
        self.assertEqual(chain_of(use_batch=True),
                         ["anthropic.claude-sonnet-4-6"])


# ─────────────────────────────────────────────────────────────────────────────
# The seeded model registry (README.md §P5.1/§P5.5) — snake_case
# shape, bare ids, inference_region routing, capabilities, and the seeder
# ─────────────────────────────────────────────────────────────────────────────

# The 2026-08-11 seeded shape, exercised alongside the legacy camelCase REGISTRY
# above (which every older test still uses — that IS the compat coverage).
SEEDED_REGISTRY = [
    {
        "model_name": "Anthropic Claude Sonnet 4.6",
        "model_id": "anthropic.claude-sonnet-4-6",
        "model_type": ["vision", "text"],
        "inference_region": "global",
        "supported_regions": ["global"],
        "model_pricing": {"input_per_token_cost": 3e-06,  # nosec B105
                          "output_per_token_cost": 1.5e-05},  # nosec B105
        "batch_job": {"can_support": True},
        "capabilities": {
            "temperature": True, "thinking": "optional", "effort": True,
            "image": {
                "limits": {"max_images": 20, "max_payload_mb": None},
                "downscales_above": {"long_side_px": 1568, "total_px": 1150000},
                "recommendations": {"min_short_side_px": 328,
                                    "max_grid_images": 12},
            },
        },
    },
    {
        "model_name": "Anthropic Claude Sonnet 5",
        "model_id": "anthropic.claude-sonnet-5",
        "model_type": ["vision", "text"],
        "inference_region": "global",
        "supported_regions": ["global"],
        "model_pricing": {"input_per_token_cost": 2e-06,  # nosec B105
                          "output_per_token_cost": 1e-05},  # nosec B105
        "batch_job": {"can_support": False},
        "capabilities": {"temperature": False, "thinking": "required",
                         "thinking_default": "disabled",
                         "effort": True, "effort_default": "low"},
    },
    {
        "model_name": "Anthropic Claude Fable 5",
        "model_id": "anthropic.claude-fable-5",
        "model_type": ["vision", "text"],
        "inference_region": "global",
        "supported_regions": ["global"],
        "model_pricing": {"input_per_token_cost": 1.5e-05,  # nosec B105
                          "output_per_token_cost": 7.5e-05},  # nosec B105
        "batch_job": {"can_support": False},
        # thinking_default "adaptive", NOT "disabled": Fable 5 rejects an
        # explicit disabled outright (live 2026-08-14) — see ModelSpec.
        "capabilities": {"temperature": False, "thinking": "required",
                         "thinking_default": "adaptive",
                         "effort": True, "effort_default": "low"},
    },
    {
        "model_name": "Amazon Nova Pro",
        "model_id": "amazon.nova-pro-v1:0",
        "model_type": ["vision", "text"],
        "inference_region": "eu",
        "supported_regions": ["eu", "eu-west-1", "eu-central-1"],
        "model_pricing": {"input_per_token_cost": 8e-07,  # nosec B105
                          "output_per_token_cost": 3.2e-06},  # nosec B105
        "batch_job": {"can_support": True},
        "requires_inference_profile": True,
        "capabilities": {"temperature": True, "thinking": "absent",
                         "effort": False,
                         "image": {"limits": {"max_images": None,
                                              "max_payload_mb": 25}}},
    },
]


class TestGatewayModelTools(_GatewayCase):
    """list_models / get_model — thin reads over the seeded artifact, so what the
    tools advertise and what the plan accepts cannot disagree."""

    def _stub_registry(self):
        body = json.dumps({"registry_version": "alpha.test", "geo": "eu",
                           "models": SEEDED_REGISTRY}).encode()
        self.s3.get_object.return_value = {
            "Body": mock.MagicMock(read=lambda: body)}

    def test_list_models_summarizes_without_capabilities(self):
        self._stub_registry()
        result = self.gateway._list_models({})
        self.assertEqual(result["geo"], "eu")
        self.assertEqual(result["models_total"], len(SEEDED_REGISTRY))
        nova = next(m for m in result["models"]
                    if m["model_id"] == "amazon.nova-pro-v1:0")
        self.assertEqual(nova["inference_region"], "eu")
        self.assertTrue(nova["can_batch"])
        self.assertNotIn("capabilities", nova,
                         "the listing stays summary-thin; detail is get_model's")

    def test_get_model_returns_the_full_entry_by_bare_or_prefixed_id(self):
        self._stub_registry()
        for query in ("anthropic.claude-sonnet-4-6",
                      "global.anthropic.claude-sonnet-4-6"):
            result = self.gateway._get_model({"model_id": query})
            self.assertEqual(result["model"]["model_id"],
                             "anthropic.claude-sonnet-4-6", query)
            self.assertIn("image", result["model"]["capabilities"])

    def test_get_model_names_the_miss(self):
        self._stub_registry()
        result = self.gateway._get_model({"model_id": "meta.llama4-maverick"})
        self.assertTrue(result.get("error"))
        self.assertEqual(result["errorCode"], "MODEL_NOT_REGISTERED")


class TestInputExamples(_GatewayCase):
    """input_examples.json — the INPUT twin of the output contract, served
    by describe_tool. The run_inference example round-trips through the REAL
    lambda_handler with the mocked sfn/s3 clients, and the registry reads
    round-trip against the seeded-shape fixture, so an example that drifts
    from the real contract fails this build, not an agent's first call."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            cls.examples = json.load(f)["tools"]

    def _call(self, tool, event):
        return self.gateway.lambda_handler(dict(event),
                                           _AgentToolContext(tool))

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        self.assertEqual(sorted(self.examples),
                         ["describe_tool", "get_job_status", "get_model",
                          "list_models", "run_inference"])
        for tool, entry in self.examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)
            # every PLAIN required name must appear in the example; a prose
            # one-of entry (contains a space or "|") states a rule the
            # schema cannot express and is exempt
            for name in entry["required"]:
                if " " in name or "|" in name:
                    continue
                self.assertIn(name, entry["example"],
                              f"{tool}: required field {name!r} missing "
                              f"from its example")

    def test_describe_tool_carries_input_examples(self):
        out = self._call("describe_tool", {})
        self.assertIn("run_inference", out["input_examples"]["tools"])

    def test_submit_example_passes_the_gateway_validation(self):
        example = self.examples["run_inference"]["example"]
        out = self._call("run_inference", example)
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        self.assertEqual(out["records_total"], len(example["records"]))
        self.sfn.start_execution.assert_called_once()

    def test_staged_alternative_passes_the_gateway_validation(self):
        # the records_s3uri form reads the staged JSONL at submit, so the
        # alternative round-trips with a stubbed object body
        alt = self.examples["run_inference"]["alternatives"]["records_s3uri"]
        line = json.dumps({"record_id": "r0", "system": "Be terse.",
                           "messages": [{"role": "user",
                                         "content": [{"text": "hi"}]}]})
        # the staged reader streams the object in chunks (body.read(n)),
        # so the stub must be a real file-like body
        self.s3.get_object.return_value = {
            "Body": io.BytesIO((line + "\n").encode())}
        out = self._call("run_inference", alt)
        self.assertNotIn("error", out, f"alternative refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        self.assertEqual(out["records_total"], 1)

    def test_registry_examples_round_trip_against_the_seeded_shape(self):
        body = json.dumps({"registry_version": "alpha.test", "geo": "eu",
                           "models": SEEDED_REGISTRY}).encode()
        self.s3.get_object.return_value = {
            "Body": mock.MagicMock(read=lambda: body)}
        listed = self._call("list_models",
                            self.examples["list_models"]["example"])
        self.assertNotIn("error", listed, f"example refused: {listed}")
        self.assertEqual(listed["models_total"], len(SEEDED_REGISTRY))
        got = self._call("get_model", self.examples["get_model"]["example"])
        self.assertNotIn("error", got, f"example refused: {got}")
        # the example's model_id must stay a registered id, bare-spelled
        self.assertEqual(got["model"]["model_id"],
                         self.examples["get_model"]["example"]["model_id"])


class TestSeededRegistryShape(unittest.TestCase):
    """ModelSpec over the snake_case shape: bare ids, routing resolution, the
    capability block beating the id-substring tuples."""

    def spec(self, model_id):
        return bedrock_lib.resolve_model(SEEDED_REGISTRY, model_id)

    def test_a_prefixed_caller_id_resolves_to_its_bare_entry(self):
        """The prefix names ROUTING, not identity — a caller still sending the
        old `global.`-prefixed id must land on the bare entry, or every existing
        caller breaks the day the registry is reseeded."""
        bare = self.spec("anthropic.claude-sonnet-4-6")
        prefixed = self.spec("global.anthropic.claude-sonnet-4-6")
        self.assertEqual(bare.model_id, prefixed.model_id)

    def test_runtime_id_resolution_per_scope(self):
        spec = self.spec("anthropic.claude-sonnet-4-6")
        self.assertEqual(spec.resolve_runtime_model_id(),
                         "global.anthropic.claude-sonnet-4-6")
        self.assertEqual(spec.resolve_runtime_model_id("eu"),
                         "eu.anthropic.claude-sonnet-4-6")
        # A concrete region routes the CLIENT there. Sonnet has no
        # requires_inference_profile in this fixture, so the id goes bare...
        self.assertEqual(spec.resolve_runtime_model_id("eu-west-1"),
                         "anthropic.claude-sonnet-4-6")
        # ...while a profile-only model keeps its GEO profile (bare on-demand is
        # rejected by Bedrock — measured live 2026-08-11 on Nova Pro): the
        # client pins to the region, the id stays profiled through its geo.
        nova = self.spec("amazon.nova-pro-v1:0")
        self.assertEqual(nova.resolve_runtime_model_id("eu-west-1"),
                         "eu.amazon.nova-pro-v1:0")
        self.assertEqual(nova.resolve_client_region("eu-west-1"), "eu-west-1")
        self.assertEqual(spec.resolve_client_region("eu-west-1"), "eu-west-1")
        self.assertIsNone(spec.resolve_client_region("global"))

    def test_the_override_gate_is_literal_membership(self):
        """The EU-residency rule: a geo profile existing says nothing about
        in-region availability, so no inference from geo membership."""
        nova = self.spec("amazon.nova-pro-v1:0")
        self.assertTrue(nova.allows_inference_region("eu-west-1"))
        self.assertTrue(nova.allows_inference_region("eu"))
        self.assertFalse(nova.allows_inference_region("eu-north-1"))
        sonnet = self.spec("anthropic.claude-sonnet-4-6")
        self.assertFalse(sonnet.allows_inference_region("eu"),
                         "supports only ['global'] — an eu override must refuse "
                         "loudly, never silently run global")

    def test_capabilities_are_the_only_authority(self):
        """The id-substring family tuples are RETIRED (2026-10-15): a model's
        request contract comes from its entry's capability block and nowhere
        else — Sonnet 4.6's `temperature: true`, Sonnet 5's
        `thinking: required` and `temperature: false` are data, not id
        matching."""
        s46 = self.spec("anthropic.claude-sonnet-4-6")
        self.assertTrue(s46.accepts_temperature())
        self.assertFalse(s46.requires_thinking_field())
        s5 = self.spec("anthropic.claude-sonnet-5")
        self.assertFalse(s5.accepts_temperature())
        self.assertTrue(s5.requires_thinking_field())

    def test_pads_are_equal_but_never_aliased(self):
        """Shaping happens once; each pad gets its own deep copy, so a
        consumer mutating one pad cannot corrupt its siblings."""
        gen = bedrock_lib.padding_records(3, None)
        first = next(gen)
        clean = copy.deepcopy(first["modelInput"])
        # a consumer mutating pad 0 BETWEEN pulls must not poison later
        # pads — the template is snapshotted before the first yield
        first["modelInput"]["messages"].append({"role": "user",
                                                "content": [{"text": "X"}]})
        pads = [first] + list(gen)
        self.assertEqual([p["recordId"] for p in pads],
                         ["pad_0000", "pad_0001", "pad_0002"])
        self.assertEqual(pads[1]["modelInput"], clean)
        self.assertEqual(pads[2]["modelInput"], clean)
        self.assertIsNot(pads[1]["modelInput"], pads[2]["modelInput"])
        self.assertEqual(list(bedrock_lib.padding_records(0, None)), [])

    def test_pads_are_shaped_by_the_batch_models_contract(self):
        """A pad hardcoding temperature + no thinking was a LIVE
        all-or-nothing landmine: opus-5 batches AND is thinking-required,
        and the 2026-10-15 probes measured both directions — old shape
        REJECTED ('temperature is deprecated'), spec shape ACCEPTED
        (stopReason max_tokens, 1 output token)."""
        spec = bedrock_lib.ModelSpec(
            {"model_id": "anthropic.claude-test-6",
             "capabilities": {"temperature": False, "thinking": "required",
                              "thinking_default": "adaptive",
                              "effort": True, "effort_default": "low"}})
        pad = bedrock_lib.padding_record(0, spec)["modelInput"]
        self.assertEqual(pad["inferenceConfig"], {"maxTokens": 1})
        self.assertEqual(pad["additionalModelRequestFields"]["thinking"],
                         {"type": "adaptive"})
        # spec-less callers get the SAFE minimal shape: maxTokens only —
        # the historical temperature: 0 is the half the probe showed fatal
        bare = bedrock_lib.padding_record(0, None)["modelInput"]
        self.assertEqual(bare["inferenceConfig"], {"maxTokens": 1})
        self.assertNotIn("additionalModelRequestFields", bare)

    def test_temperature_keys_on_the_emitted_thinking_not_the_ask(self):
        """An entry's adaptive thinking_default turns reasoning on for a
        QUIET caller, and Bedrock rejects sampling params beside reasoning
        regardless of who asked — so temperature suppression must key on
        the type the request will carry, never on the caller's ask."""
        spec = bedrock_lib.ModelSpec(
            {"model_id": "anthropic.claude-test-6",
             "capabilities": {"temperature": True, "thinking": "required",
                              "thinking_default": "adaptive",
                              "effort": True}})
        params = bedrock_lib.build_converse_params(
            spec, None, [{"role": "user", "content": [{"text": "hi"}]}],
            thinking="disabled", temperature=0.5)
        self.assertEqual(params["additionalModelRequestFields"]["thinking"],
                         {"type": "adaptive"})
        self.assertNotIn("inferenceConfig", params)

    def test_value_level_slips_warn_at_resolve_time(self):
        """A hand-edit can carry what the seeder would refuse: a thinking
        typo coerces to 'absent' and a required entry without a default
        coerces to 'disabled' — each named once, where the entry is read."""
        entry = {"model_id": "anthropic.claude-handedit-6",
                 "model_type": ["text"],
                 "capabilities": {"temperature": False,
                                  "thinking": "require",   # typo
                                  "effort": True}}
        bedrock_lib._WARNED_CONTRACT.clear()
        self.addCleanup(bedrock_lib._WARNED_CONTRACT.clear)
        with self.assertLogs(bedrock_lib.log, level="WARNING") as captured:
            bedrock_lib.resolve_model([entry], "anthropic.claude-handedit-6")
        self.assertIn("coerces to 'absent'", "\n".join(captured.output))
        # the warnings dedup per entry per container — a re-resolve is silent
        with mock.patch.object(bedrock_lib.log, "warning") as warned:
            bedrock_lib.resolve_model([entry], "anthropic.claude-handedit-6")
        warned.assert_not_called()
        entry["capabilities"]["thinking"] = "required"
        bedrock_lib._WARNED_CONTRACT.clear()
        self.addCleanup(bedrock_lib._WARNED_CONTRACT.clear)
        with self.assertLogs(bedrock_lib.log, level="WARNING") as captured:
            bedrock_lib.resolve_model([entry], "anthropic.claude-handedit-6")
        self.assertIn("thinking_default", "\n".join(captured.output))

    def test_a_string_model_type_still_classifies_sanely(self):
        """A hand-edited 'model_type': 'embedding' (string, not list) must
        read as a pure embedder, and 'text,vision' as generative."""
        emb = {"model_id": "v.embed", "model_type": "embedding",
               "model_pricing": {}}
        logger = bedrock_lib.log
        with mock.patch.object(logger, "warning") as warned:
            bedrock_lib.resolve_model([emb], "v.embed")
        warned.assert_not_called()
        gen = {"model_id": "v.gen", "model_type": "text,vision",
               "model_pricing": {}}
        bedrock_lib._WARNED_CONTRACT.clear()
        self.addCleanup(bedrock_lib._WARNED_CONTRACT.clear)
        with self.assertLogs(logger, level="WARNING") as captured:
            bedrock_lib.resolve_model([gen], "v.gen")
        self.assertIn("lacks capability keys", "\n".join(captured.output))

    def test_effort_vocabulary_has_one_home(self):
        """input_schema RE-EXPORTS bedrock's VALID_EFFORT_VALUES (identity,
        not a pinned copy) — the dependency runs this direction because the
        reverse would drag s3_utils' module-level boto3 client into the
        stdlib-only vendorable module."""
        self.assertIs(input_schema.VALID_EFFORTS,
                      bedrock_lib.VALID_EFFORT_VALUES)

    def test_a_dead_effort_default_is_a_contract_finding(self):
        entry = {"model_id": "anthropic.claude-test-6", "model_type": ["text"],
                 "capabilities": {"temperature": True, "thinking": "absent",
                                  "effort": False, "effort_default": "low"}}
        findings = bedrock_lib.contract_findings(entry)
        self.assertTrue(any("can never be emitted" in f for f in findings),
                        findings)

    def test_non_dict_capabilities_coerces_to_defaults_not_a_crash(self):
        """A hand-edit can write capabilities as a string/list; the spec
        must shape with the key-less defaults (which the contract finding
        names) rather than AttributeError on every request."""
        spec = bedrock_lib.ModelSpec({"model_id": "x",
                                      "capabilities": "oops"})
        self.assertTrue(spec.accepts_temperature())
        self.assertEqual(spec.thinking_support, "absent")
        self.assertFalse(spec.accepts_effort())
        findings = bedrock_lib.contract_findings(
            {"model_id": "x", "model_type": ["text"],
             "capabilities": "oops"})
        self.assertTrue(any("JSON object" in f for f in findings))

    def test_an_embedding_model_cannot_enter_a_converse_chain(self):
        """Every record (and every pad) in this workflow is Converse-shaped;
        an embedding model in the chain would 400 each sync record and fail
        a batch job all-or-nothing — refused at plan time, by name."""
        entry = {"model_id": "amazon.nova-2-multimodal-embeddings-v1:0",
                 "model_type": ["embedding"],
                 "model_pricing": {"input_per_token_cost": 1e-7,
                                   "output_per_token_cost": 0.0}}
        with self.assertRaises(bedrock_lib.BedrockError) as raised:
            bedrock_lib.build_chain([entry],
                                    ["amazon.nova-2-multimodal-embeddings-v1:0"])
        self.assertIn("non-Converse", str(raised.exception))

    def test_only_recognized_non_converse_types_are_exempt(self):
        """Exemption demands a RECOGNIZED non-Converse type: a typo'd
        generative type ("Text", "txt") must stay on the hook, a missing
        model_type too, and a genuinely new non-Converse type (image
        generation) earns its exemption by joining NON_CONVERSE_TYPES —
        a deliberate one-line vocabulary edit, never an accident."""
        self.assertTrue(bedrock_lib.is_converse_exempt(
            {"model_id": "x", "model_type": ["embedding"]}))
        for types in (["image"], ["Text"], ["txt"], [], None):
            entry = {"model_id": "x"}
            if types is not None:
                entry["model_type"] = types
            self.assertFalse(bedrock_lib.is_converse_exempt(entry), types)

    def test_a_typod_generative_sibling_does_not_silence_the_contract(self):
        """["embedding", "Text"] must NOT be exempt — every declared type
        has to be a recognized non-Converse type, and case is normalized
        so "Text" reads as the generative type it was meant to be."""
        self.assertFalse(bedrock_lib.is_converse_exempt(
            {"model_id": "x", "model_type": ["embedding", "Text"]}))
        self.assertFalse(bedrock_lib.is_converse_exempt(
            {"model_id": "x", "model_type": ["embedding", "txt"]}))
        self.assertTrue(bedrock_lib.is_converse_exempt(
            {"model_id": "x", "model_type": ["Embedding"]}))

    def test_spec_less_pads_are_refused_when_pads_would_be_emitted(self):
        with self.assertRaises(ValueError):
            batch_client.iter_pads(1, None)   # EAGER: refuses at the call,
            #                                   never on the first pull
        # no pads needed -> nothing to shape -> None is fine
        self.assertEqual(
            list(batch_client.iter_pads(input_schema.MIN_BATCH_RECORDS,
                                        None)), [])

    def test_compound_defects_are_all_named_in_one_pass(self):
        """A thinking typo must not suppress the temperature/adaptive
        contradiction — the editor fixes everything on deploy one."""
        findings = bedrock_lib.contract_findings(
            {"model_id": "x", "model_type": ["text"],
             "capabilities": {"temperature": True, "thinking": "optionall",
                              "thinking_default": "adaptive",
                              "effort": True}})
        text = "\n".join(findings)
        self.assertIn("is not one of", text)        # the typo
        self.assertIn("contradicts", text)          # AND the contradiction
        findings = bedrock_lib.contract_findings(
            {"model_id": "x", "model_type": ["text"],
             "capabilities": {"temperature": False, "thinking": "absent",
                              "effort": 0, "effort_default": "low"}})
        text = "\n".join(findings)
        self.assertIn("JSON boolean", text)         # effort: 0 typed wrong
        self.assertIn("can never be emitted", text)  # AND the dead default
        # a MISSING thinking key emits no thinking field, so temperature
        # genuinely applies — the contradiction must NOT fire beside the
        # lacks-keys finding with a claim the shaper makes false
        findings = bedrock_lib.contract_findings(
            {"model_id": "x", "model_type": ["text"],
             "capabilities": {"temperature": True,
                              "thinking_default": "adaptive",
                              "effort": True}})
        text = "\n".join(findings)
        self.assertIn("lacks capability keys", text)
        self.assertNotIn("contradicts", text)

    def test_a_warm_container_rewarns_when_the_entry_changes(self):
        """The dedup is keyed on the FINDINGS, not the model_id alone — a
        hand-edit that breaks an entry a warm container already saw healthy
        must still be named."""
        entry = {"model_id": "anthropic.claude-warm-6", "model_type": ["text"],
                 "capabilities": {"temperature": False, "thinking": "optional",
                                  "effort": True}}
        bedrock_lib._WARNED_CONTRACT.clear()
        self.addCleanup(bedrock_lib._WARNED_CONTRACT.clear)
        with mock.patch.object(bedrock_lib.log, "warning") as warned:
            bedrock_lib.resolve_model([entry], "anthropic.claude-warm-6")
        warned.assert_not_called()                    # healthy: no warning
        del entry["capabilities"]                     # the hand-edit breaks it
        with self.assertLogs(bedrock_lib.log, level="WARNING") as captured:
            bedrock_lib.resolve_model([entry], "anthropic.claude-warm-6")
        self.assertIn("lacks capability keys", "\n".join(captured.output))

    def test_a_contract_less_generative_entry_warns_at_resolve_time(self):
        """A hand-edited artifact bypasses the seeder; the key-less defaults
        400 on a thinking-required family with nothing naming the cause —
        resolve_model names it, once, where the entry is first read."""
        bare = [{"model_id": "anthropic.claude-handedit-6",
                 "model_type": ["text"],
                 "model_pricing": {"input_per_token_cost": 1e-6,
                                   "output_per_token_cost": 1e-6}}]
        bedrock_lib._WARNED_CONTRACT.clear()
        self.addCleanup(bedrock_lib._WARNED_CONTRACT.clear)
        with self.assertLogs(bedrock_lib.log, level="WARNING") as captured:
            spec = bedrock_lib.resolve_model(
                bare, "anthropic.claude-handedit-6")
        self.assertIn("lacks capability keys", "\n".join(captured.output))
        self.assertEqual(spec.model_id, "anthropic.claude-handedit-6")

    def test_thinking_required_always_emits_the_field(self):
        """Sonnet 5+: additionalModelRequestFields.thinking must be PRESENT even
        when the caller did not ask — {"type": "disabled"} for Sonnet 5, plus
        its effort_default riding along since 2026-08-14."""
        s5 = self.spec("anthropic.claude-sonnet-5")
        params = bedrock_lib.build_converse_params(
            s5, None, [{"role": "user", "content": [{"text": "Hi"}]}],
            thinking="disabled")
        self.assertEqual(params["additionalModelRequestFields"]["thinking"],
                         {"type": "disabled"})
        self.assertNotIn("inferenceConfig", params)

    def test_the_claude_5_family_quiet_request_shapes(self):
        """
        THE 2026-08-14 SPEC, pinned per model, for a request whose caller asked
        for nothing: Fable 5 and Opus 5 run `thinking.type=adaptive` +
        `output_config.effort=low`; Sonnet 5 runs `thinking.type=disabled` +
        `effort=low`. The split exists because "required" only says the field
        must be present — Fable 5 REJECTS an explicit disabled
        (`"thinking.type.disabled" is not supported for this model`, the
        model_matrix run's only red), so its quiet value must be adaptive.

        And the 4.6 family stays byte-identical to before defaults existed:
        thinking disabled, NO output_config at all.
        """
        def additional(model_id, **kwargs):
            params = bedrock_lib.build_converse_params(
                self.spec(model_id), None,
                [{"role": "user", "content": [{"text": "Hi"}]}],
                **kwargs)
            return params.get("additionalModelRequestFields") or {}

        fable = additional("anthropic.claude-fable-5")
        self.assertEqual(fable["thinking"], {"type": "adaptive"})
        self.assertEqual(fable["output_config"], {"effort": "low"})

        s5 = additional("anthropic.claude-sonnet-5")
        self.assertEqual(s5["thinking"], {"type": "disabled"})
        self.assertEqual(s5["output_config"], {"effort": "low"})

        s46 = additional("anthropic.claude-sonnet-4-6")
        self.assertEqual(s46["thinking"], {"type": "disabled"})
        self.assertNotIn("output_config", s46)

        # A caller's explicit effort beats the entry's default...
        high = additional("anthropic.claude-fable-5", effort="high")
        self.assertEqual(high["output_config"], {"effort": "high"})
        # ...and asking for thinking keeps adaptive on a default-adaptive model.
        on = additional("anthropic.claude-fable-5", thinking="adaptive")
        self.assertEqual(on["thinking"], {"type": "adaptive"})

    def test_claude_5_family_master_entries_default_to_adaptive(self):
        """Opus 5 TOLERATES an explicit disabled (it passed the 2026-08-13
        matrix that way), so only the registry data says it must run adaptive —
        nothing at runtime would catch this entry regressing. Sonnet 5.5
        JOINED this row 2026-10-15: unlike Sonnet 5 (which wanted an explicit
        disabled), 5.5 runs adaptive by default and REJECTS disabled — the
        entry kept Sonnet 5's thinking_default across the id swap and every
        quiet request 400'd (SME field report; AWS model card
        model-card-anthropic-claude-sonnet-5-5 confirms "adaptive thinking is
        on by default"). Read from the MASTER, which is what actually ships."""
        with open(os.path.join(ROOT, "models",
                               "availablemodels_master.json")) as f:
            master = {m["model_id"]: m for m in json.load(f)["models"]}
        for model_id, thinking_default in (
                ("anthropic.claude-fable-5-1", "adaptive"),
                ("anthropic.claude-opus-5", "adaptive"),
                ("anthropic.claude-sonnet-5-5", "adaptive"),
                # Haiku 5.5 joined 2026-10-16: TOLERATES an explicit
                # disabled (probed live, like Opus 5) but the card and the
                # entry say adaptive; downscales_above COPIED from the
                # family — the measured token plateau (onset bracketed
                # between 2.36 and 4.19 Mpx) is consistent with 3.75 Mpx,
                # not a measurement of it
                ("anthropic.claude-haiku-5-5", "adaptive")):
            entry = master[model_id]
            caps = entry["capabilities"]
            self.assertEqual(caps["thinking_default"], thinking_default,
                             model_id)
            self.assertEqual(caps["effort_default"], "low", model_id)
            # the rest of the entry contract, pinned per the GPT-6 pattern:
            # nothing at runtime catches a hand-flip of these
            self.assertEqual(bedrock_lib.contract_findings(entry), [],
                             model_id)
            self.assertFalse(caps["temperature"], model_id)
            self.assertIn("eu", entry["geos"], model_id)
            self.assertEqual(entry["inference_region"], "global", model_id)
            self.assertEqual(
                entry["batch_job"]["can_support"],
                {"anthropic.claude-opus-5": True}.get(model_id, False),
                model_id)
            # The 5 family also downscales later: 2576 px long edge / 3.75 MP,
            # against the 4.6 family's 1568 / 1.15 MP (updated 2026-08-14 —
            # and the old registry said 1538, a TYPO of 1568, on every entry).
            self.assertEqual(caps["image"]["downscales_above"],
                             {"long_side_px": 2576, "total_px": 3750000},
                             model_id)
        for model_id in ("anthropic.claude-sonnet-4-6",
                         "anthropic.claude-haiku-4-5-20251001-v1:0"):
            self.assertEqual(
                master[model_id]["capabilities"]["image"]["downscales_above"],
                {"long_side_px": 1568, "total_px": 1150000}, model_id)

    def test_batch_follows_inference_region(self):
        """Settled 2026-08-11: no separate batch region — a run pinned to `eu`
        for residency must not have its batch half silently route global."""
        nova = self.spec("amazon.nova-pro-v1:0")
        self.assertEqual(bedrock_lib.resolve_batch_model_id(nova),
                         "eu.amazon.nova-pro-v1:0")
        self.assertEqual(bedrock_lib.resolve_batch_model_id(nova, "eu-west-1"),
                         "eu.amazon.nova-pro-v1:0")

    def test_snake_case_pricing_and_the_envelope_are_read(self):
        spec = self.spec("anthropic.claude-sonnet-4-6")
        cost = spec.estimate_cost({"inputTokens": 1000, "outputTokens": 100})
        self.assertAlmostEqual(cost, 1000 * 3e-06 + 100 * 1.5e-05)
        body = json.dumps({"registry_version": "v", "geo": "eu",
                           "models": SEEDED_REGISTRY}).encode()
        s3 = mock.MagicMock()
        s3.get_object.return_value = {"Body": mock.MagicMock(read=lambda: body)}
        registry = bedrock_lib.load_registry(s3, "b", "k")
        self.assertEqual(len(registry), len(SEEDED_REGISTRY))


class TestEffortWire(unittest.TestCase):
    """The OpenAI GPT-6 family carries the effort knob on a DIFFERENT wire
    (additionalModelRequestFields.reasoning.effort — flat reasoning_effort is
    REJECTED, probed live 2026-10-16) and widens the value vocabulary with
    registry-default-only values ("none" is Luna-only; Sol/Astra reject it).
    The wire is per-entry data (capabilities.effort_wire), the caller
    vocabulary stays the cross-wire intersection, and the seed-time battery
    validates each entry's default against ITS wire."""

    LUNA = {"model_id": "openai.gpt-6-luna", "model_type": ["vision", "text"],
            "inference_region": "global", "supported_regions": ["global"],
            "batch_job": {"can_support": False},
            "capabilities": {"temperature": False, "thinking": "absent",
                             "effort": True, "effort_default": "none",
                             "effort_wire": "reasoning"}}

    def _additional(self, entry, **kwargs):
        params = bedrock_lib.build_converse_params(
            bedrock_lib.ModelSpec(entry), None,
            [{"role": "user", "content": [{"text": "Hi"}]}], **kwargs)
        self.assertNotIn("inferenceConfig", params)   # temperature rejected
        return params.get("additionalModelRequestFields") or {}

    def test_reasoning_wire_quiet_request_emits_the_default_on_its_wire(self):
        additional = self._additional(self.LUNA)
        self.assertEqual(additional["reasoning"], {"effort": "none"})
        self.assertNotIn("output_config", additional)
        self.assertNotIn("thinking", additional)      # thinking absent

    def test_caller_effort_beats_the_default_on_the_reasoning_wire(self):
        additional = self._additional(self.LUNA, effort="high")
        self.assertEqual(additional["reasoning"], {"effort": "high"})

    def test_keyless_entries_stay_on_the_anthropic_wire(self):
        entry = dict(self.LUNA, model_id="anthropic.claude-wiretest-5")
        entry["capabilities"] = {"temperature": False, "thinking": "absent",
                                 "effort": True, "effort_default": "low"}
        additional = self._additional(entry)
        self.assertEqual(additional["output_config"], {"effort": "low"})
        self.assertNotIn("reasoning", additional)

    def test_caller_vocabulary_is_the_cross_wire_intersection(self):
        """A chain may mix wires, so every caller value must be valid on
        BOTH — "none"/"xhigh"/"max" are reachable only as registry defaults
        (Sol and Astra reject "none" by name; probed live 2026-10-16)."""
        self.assertEqual(bedrock_lib.VALID_EFFORT_VALUES,
                         ("low", "medium", "high"))
        for wire, vocab in bedrock_lib.EFFORT_WIRE_VOCABULARY.items():
            for value in bedrock_lib.VALID_EFFORT_VALUES:
                self.assertIn(value, vocab, wire)

    def test_an_unknown_wire_is_a_finding_and_coerces_to_output_config(self):
        entry = {"model_id": "v.wire", "model_type": ["text"],
                 "capabilities": {"temperature": False, "thinking": "absent",
                                  "effort": True, "effort_default": "low",
                                  "effort_wire": "reasoning_effort"}}
        findings = bedrock_lib.contract_findings(entry)
        self.assertTrue(any("effort_wire" in f for f in findings), findings)
        self.assertEqual(bedrock_lib.ModelSpec(entry).effort_wire(),
                         "output_config")

    def test_effort_default_is_validated_against_the_entries_own_wire(self):
        base = {"model_id": "v.wire", "model_type": ["text"],
                "capabilities": {"temperature": False, "thinking": "absent",
                                 "effort": True}}
        ok = dict(base); ok["capabilities"] = dict(
            base["capabilities"], effort_default="none",
            effort_wire="reasoning")
        self.assertEqual(bedrock_lib.contract_findings(ok), [])
        bad = dict(base); bad["capabilities"] = dict(
            base["capabilities"], effort_default="none")
        findings = bedrock_lib.contract_findings(bad)
        self.assertTrue(any("effort_default" in f for f in findings),
                        findings)

    def test_an_unhashable_effort_wire_warns_instead_of_type_erroring(self):
        """A hand-edited artifact can write effort_wire as a LIST — the same
        hand-edit class the capabilities-not-a-dict coercion exists for.
        contract_findings runs on the resolve warn-path, so it must name the
        defect, never TypeError on a dict lookup (review catch 2026-10-16)."""
        entry = {"model_id": "v.wire", "model_type": ["text"],
                 "capabilities": {"temperature": False, "thinking": "absent",
                                  "effort": True, "effort_default": "low",
                                  "effort_wire": ["reasoning"]}}
        findings = bedrock_lib.contract_findings(entry)
        self.assertTrue(any("effort_wire" in f for f in findings), findings)
        self.assertEqual(bedrock_lib.ModelSpec(entry).effort_wire(),
                         "output_config")

    def test_the_wire_set_is_the_vocabulary_key_set(self):
        self.assertEqual(set(bedrock_lib.EFFORT_WIRES),
                         set(bedrock_lib.EFFORT_WIRE_VOCABULARY))

    def test_a_dead_effort_wire_is_a_contract_finding(self):
        entry = {"model_id": "v.wire", "model_type": ["text"],
                 "capabilities": {"temperature": False, "thinking": "absent",
                                  "effort": False, "effort_wire": "reasoning"}}
        findings = bedrock_lib.contract_findings(entry)
        self.assertTrue(any("effort_wire" in f and "can never be emitted" in f
                            for f in findings), findings)

    def test_the_three_gpt6_master_entries_are_contract_clean(self):
        """Pinned against the MASTER (what ships): Converse-capable vision
        entries, no batch, no temperature, thinking absent, the reasoning
        wire with the user-decided defaults (none for Luna — Sol/Astra
        reject it, so their floor "low"), global routing reachable from eu,
        and the MEASURED image limits (100 images; 22 MiB payload passes,
        23 fails, 20 is the conservative registry pin; auto drops to low detail above ~1000 32px patches ≈
        1,024,000 px — fine structure aliases with no error)."""
        with open(os.path.join(ROOT, "models",
                               "availablemodels_master.json")) as f:
            master = {m["model_id"]: m for m in json.load(f)["models"]}
        expected_defaults = {"openai.gpt-6-luna": "none",
                             "openai.gpt-6.1-sol": "low",
                             "openai.gpt-6-astra": "low"}
        # Cache pricing pinned from the three MODEL CARDS (2026-10-16),
        # Global CRIS short-context, per token: write = 1.25x input on all
        # three; read is NOT uniform — Sol's card bills 0.05x where Luna and
        # Astra bill 0.10x — which is why these are per-entry pins, never a
        # family formula. Long context (>272k input) doubles input/cache and
        # bills 1.5x output; deliberately unmodeled, nothing this stack sends
        # goes near it.
        expected_cache_pricing = {
            "openai.gpt-6-luna": (1.25e-07, 1e-08),
            "openai.gpt-6.1-sol": (2.5e-06, 1e-07),
            "openai.gpt-6-astra": (1.25e-05, 1e-06)}
        for model_id, effort_default in expected_defaults.items():
            entry = master[model_id]
            self.assertEqual(bedrock_lib.contract_findings(entry), [],
                             model_id)
            caps = entry["capabilities"]
            self.assertEqual(caps["effort_wire"], "reasoning", model_id)
            self.assertEqual(caps["effort_default"], effort_default, model_id)
            # implicit caching is AUTOMATIC on GPT-6 and billed writes at
            # 1.25x input with ZERO reads in this stack's workloads — every
            # entry opts out and prices the traffic the opt-out cannot stop
            self.assertEqual(caps["prompt_cache_mode"], "explicit", model_id)
            write_cost, read_cost = expected_cache_pricing[model_id]
            pricing = entry["model_pricing"]
            self.assertEqual(pricing["cache_write_per_token_cost"],
                             write_cost, model_id)
            self.assertEqual(pricing["cache_read_per_token_cost"],
                             read_cost, model_id)
            self.assertAlmostEqual(
                write_cost, pricing["input_per_token_cost"] * 1.25,
                msg=model_id)
            self.assertFalse(caps["temperature"], model_id)
            self.assertEqual(caps["thinking"], "absent", model_id)
            self.assertFalse(entry["batch_job"]["can_support"], model_id)
            self.assertIn("eu", entry["geos"], model_id)
            self.assertEqual(entry["inference_region"], "global", model_id)
            self.assertEqual(caps["image"]["limits"],
                             {"max_images": 100, "max_image_mb": None,
                              "max_payload_mb": 20}, model_id)
            self.assertEqual(caps["image"]["downscales_above"],
                             {"long_side_px": None, "total_px": 1024000},
                             model_id)
            spec = bedrock_lib.ModelSpec(entry)
            self.assertEqual(spec.resolve_runtime_model_id(None),
                             f"global.{model_id}", model_id)
            self.assertFalse(spec.can_batch())


class TestPromptCacheAccounting(unittest.TestCase):
    """GPT-6 caches IMPLICITLY over Converse (automatic, no opt-in) and its
    usage itemization is NONDETERMINISTIC: the identical request sometimes
    returns full cache fields and sometimes NO cache keys at all, the cached
    input counted only in totalTokens while inputTokens collapses to a
    near-zero remainder (24 live calls 2026-10-16, incl. a 2k-20k ladder at
    one size — size-gating refuted). Two defenses, both registry-driven:
    entries declare `prompt_cache_mode: explicit` (emitted as
    prompt_cache_options — probed accepted on all three GPT-6 models, full
    inputTokens restored on fresh prefixes) and `estimate_cost` reconstructs
    the hidden write term from AWS's own identity
    total = input + output + cacheRead + cacheWrite, ONLY when the write
    field carries no value. Hidden tokens that were really cache READS
    (repeat requests) price as writes — errs HIGH, never low."""

    LUNA_LIKE = {
        "model_id": "openai.gpt-test", "model_type": ["text"],
        "inference_region": "global", "supported_regions": ["global"],
        "model_pricing": {"input_per_token_cost": 1e-07,
                          "output_per_token_cost": 5e-07,
                          "cache_write_per_token_cost": 1.25e-07,
                          "cache_read_per_token_cost": 1e-08},
        "batch_job": {"can_support": False},
        "capabilities": {"temperature": False, "thinking": "absent",
                         "effort": True, "effort_default": "none",
                         "effort_wire": "reasoning",
                         "prompt_cache_mode": "explicit"}}

    # ── the opt-out emission ────────────────────────────────────────────

    def _additional(self, entry, **kwargs):
        params = bedrock_lib.build_converse_params(
            bedrock_lib.ModelSpec(entry), None,
            [{"role": "user", "content": [{"text": "Hi"}]}], **kwargs)
        return params.get("additionalModelRequestFields") or {}

    def test_a_declaring_entry_emits_the_opt_out_on_every_request(self):
        additional = self._additional(self.LUNA_LIKE)
        self.assertEqual(additional["prompt_cache_options"],
                         {"mode": "explicit"})

    def test_keyless_entries_emit_no_cache_field(self):
        """The field's absence IS the API's implicit default — emitting
        anything for the families that never declared the knob would change
        every Anthropic/Nova request byte for no reason."""
        entry = copy.deepcopy(self.LUNA_LIKE)
        del entry["capabilities"]["prompt_cache_mode"]
        self.assertNotIn("prompt_cache_options", self._additional(entry))

    def test_an_unknown_mode_is_a_finding_and_emits_nothing(self):
        for bad in ("implicit", ["explicit"]):   # typo'd and unhashable
            entry = copy.deepcopy(self.LUNA_LIKE)
            entry["capabilities"]["prompt_cache_mode"] = bad
            findings = bedrock_lib.contract_findings(entry)
            self.assertTrue(any("prompt_cache_mode" in f for f in findings),
                            (bad, findings))
            self.assertIsNone(
                bedrock_lib.ModelSpec(entry).prompt_cache_mode(), bad)
            self.assertNotIn("prompt_cache_options", self._additional(entry))

    def test_batch_records_carry_the_opt_out_too(self):
        """Shaping is shared (build_converse_params feeds to_batch_record),
        so if a cache-opting model ever batches, both paths stay identical."""
        params = bedrock_lib.build_converse_params(
            bedrock_lib.ModelSpec(self.LUNA_LIKE), None,
            [{"role": "user", "content": [{"text": "Hi"}]}])
        record = bedrock_lib.to_batch_record("r0", params)
        self.assertEqual(
            record["modelInput"]["additionalModelRequestFields"]
            ["prompt_cache_options"], {"mode": "explicit"})

    # ── the reconstruction identity ─────────────────────────────────────

    def test_a_reported_write_field_means_no_reconstruction(self):
        """User decision 2026-10-16: reconstruct ONLY when the field is
        absent — a reported value is trusted as reported."""
        self.assertEqual(bedrock_lib.reconstructed_cache_write(
            {"inputTokens": 144, "outputTokens": 2270,
             "cacheReadInputTokens": 0, "cacheWriteInputTokens": 11450,
             "totalTokens": 13864}), 0)

    def test_a_silent_response_reconstructs_the_hidden_input(self):
        """The measured probe shape: no cache keys, inputTokens 2,
        totalTokens carrying the truth."""
        self.assertEqual(bedrock_lib.reconstructed_cache_write(
            {"inputTokens": 2, "outputTokens": 5, "totalTokens": 3137}),
            3130)

    def test_every_term_defaults_to_zero_when_absent_or_null(self):
        """Including inputTokens itself (user ask 2026-10-16 — never seen
        absent in 24 live calls, but the arithmetic must not care), and the
        explicit-null write the Opus 5 / Nova batch dialect carries counts
        as ABSENT, not as reported-zero."""
        self.assertEqual(bedrock_lib.reconstructed_cache_write(
            {"outputTokens": 5, "totalTokens": 100,
             "cacheWriteInputTokens": None, "cacheReadInputTokens": None,
             "inputTokens": None}), 95)
        self.assertEqual(bedrock_lib.reconstructed_cache_write({}), 0)
        self.assertEqual(bedrock_lib.reconstructed_cache_write(None), 0)

    def test_no_total_reconstructs_to_zero_never_negative(self):
        """Dialects that omit totalTokens (normalize_usage synthesizes
        in+out AFTER pricing) must not reconstruct phantom negative
        tokens into the max-guard."""
        self.assertEqual(bedrock_lib.reconstructed_cache_write(
            {"inputTokens": 1000, "outputTokens": 100}), 0)

    # ── the pricing ─────────────────────────────────────────────────────

    def test_hidden_write_tokens_are_priced_at_the_cache_write_rate(self):
        """The Luna probe call, priced end to end: 2 input + 5 output +
        3130 hidden, snake_case pricing keys (the master's spelling)."""
        spec = bedrock_lib.ModelSpec(self.LUNA_LIKE)
        cost = spec.estimate_cost(
            {"inputTokens": 2, "outputTokens": 5, "totalTokens": 3137})
        self.assertAlmostEqual(
            cost, 2 * 1e-07 + 5 * 5e-07 + 3130 * 1.25e-07, places=12)

    def test_reported_cache_fields_are_priced_as_reported(self):
        """The SME record shape — itemization complete, reconstruction a
        no-op by the identity (total - in - out - read == reported write)."""
        spec = bedrock_lib.ModelSpec(self.LUNA_LIKE)
        cost = spec.estimate_cost(
            {"inputTokens": 144, "outputTokens": 2270,
             "cacheReadInputTokens": 0, "cacheWriteInputTokens": 11450,
             "totalTokens": 13864})
        self.assertAlmostEqual(
            cost, 144 * 1e-07 + 2270 * 5e-07 + 11450 * 1.25e-07, places=12)

    def test_reads_price_at_the_read_rate_when_itemized(self):
        spec = bedrock_lib.ModelSpec(self.LUNA_LIKE)
        cost = spec.estimate_cost(
            {"inputTokens": 100, "outputTokens": 5,
             "cacheReadInputTokens": 2000, "cacheWriteInputTokens": 0,
             "totalTokens": 2105})
        self.assertAlmostEqual(
            cost, 100 * 1e-07 + 5 * 5e-07 + 2000 * 1e-08, places=12)

    def test_entries_without_cache_pricing_still_price_hidden_tokens_at_zero(self):
        """The pre-2026-10-16 behavior, unchanged for every non-GPT entry:
        reconstruction finds the tokens but a priceless registry entry
        bills them at 0 — cost_is_complete semantics deliberately
        untouched (user decision)."""
        entry = copy.deepcopy(self.LUNA_LIKE)
        entry["model_pricing"] = {"input_per_token_cost": 1e-07,
                                  "output_per_token_cost": 5e-07}
        spec = bedrock_lib.ModelSpec(entry)
        cost = spec.estimate_cost(
            {"inputTokens": 2, "outputTokens": 5, "totalTokens": 3137})
        self.assertAlmostEqual(cost, 2 * 1e-07 + 5 * 5e-07, places=12)

    # ── the telling ─────────────────────────────────────────────────────

    def test_sync_notes_name_the_under_itemized_tokens(self):
        runtime = _FakeRuntime(script=[{
            "output": {"message": {"content": [{"text": "OK"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 5,
                      "totalTokens": 3137},
            "metrics": {"latencyMs": 7}}])
        result = bedrock_lib.converse(
            runtime, bedrock_lib.ModelSpec(self.LUNA_LIKE), None,
            [{"role": "user", "content": [{"text": "Hi"}]}])
        self.assertTrue(any("under-itemized" in n and "3130" in n
                            for n in result.notes), result.notes)
        self.assertAlmostEqual(
            result.estimated_cost,
            2 * 1e-07 + 5 * 5e-07 + 3130 * 1.25e-07, places=12)

    def test_sync_notes_stay_silent_when_itemization_is_complete(self):
        runtime = _FakeRuntime()   # default response: in 10, out 5, total 15
        result = bedrock_lib.converse(
            runtime, bedrock_lib.ModelSpec(self.LUNA_LIKE), None,
            [{"role": "user", "content": [{"text": "Hi"}]}])
        self.assertFalse([n for n in result.notes if "under-itemized" in n])


class TestImagePayloadEnforcement(unittest.TestCase):
    """validate_plan refuses every DECLARED image limit before any spend —
    max_images per request, max_image_mb per inline image, max_payload_mb as
    the record's inline-byte total — and skips each when the entry carries
    null (NOT DECLARED, nothing pre-checked: the get_model vocabulary). All
    values measured live 2026-10-16: Anthropic refuses any image over 5 MiB
    of base64 = exactly 3.75 MiB raw (identical across all five entries); pixtral and
    llama4 bound the request at ~17.9 MiB raw (registry pins 16); Anthropic
    and GPT-6 at ~23 (pins 20); pixtral caps at 8 images, llama4 at 3, nova
    took 100 without refusal (pinned 100 for fleet consistency, user
    decision) and refuses payloads at the same ~17.9 MiB as pixtral/llama4
    (its documented 25 was never the real inline ceiling — pinned 16)."""

    @staticmethod
    def _spec(limits):
        return bedrock_lib.ModelSpec({
            "model_id": "v.imgtest", "model_type": ["vision", "text"],
            "capabilities": {"temperature": True, "thinking": "absent",
                             "effort": False, "image": {"limits": limits}}})

    @staticmethod
    def _record(image_sizes_bytes, s3_images=0):
        content = []
        for size in image_sizes_bytes:
            payload = base64.b64encode(b"x" * size).decode()
            content.append({"image": {"format": "png",
                                      "source": {"bytes": payload}}})
        for _ in range(s3_images):
            content.append({"image": {"format": "png", "source": {
                "s3Location": {"uri": "s3://b/frame.png"}}}})
        content.append({"text": "describe"})
        return {"record_id": "r1",
                "messages": [{"role": "user", "content": content}]}

    def _enforce(self, limits, record):
        spec = self._spec(limits)
        validate_plan._enforce_image_limit(
            record, spec, (spec.image_rules or {}).get("limits") or {})

    def test_an_oversized_inline_image_is_refused_naming_the_condition(self):
        with self.assertRaises(input_schema.ValidationException) as raised:
            self._enforce({"max_image_mb": 3.5},
                          self._record([4 * 1024 * 1024]))
        message = str(raised.exception)
        self.assertIn("per-image limit of 3.5 MiB", message)
        self.assertIn("r1", message)

    def test_the_inline_total_is_refused_against_max_payload_mb(self):
        with self.assertRaises(input_schema.ValidationException) as raised:
            self._enforce({"max_payload_mb": 16},
                          self._record([6 * 1024 * 1024] * 3))
        self.assertIn("payload limit of 16 MiB", str(raised.exception))

    def test_under_every_declared_limit_passes(self):
        self._enforce({"max_images": 8, "max_image_mb": 3.5,
                       "max_payload_mb": 16},
                      self._record([1024 * 1024] * 4))

    def test_null_limits_skip_their_checks(self):
        # a 30 MiB image sails through when nothing is declared — null means
        # NOT DECLARED, never "zero"
        self._enforce({"max_images": None, "max_image_mb": None,
                       "max_payload_mb": None},
                      self._record([30 * 1024 * 1024]))

    def test_s3location_images_count_toward_the_count_but_never_the_bytes(self):
        # count limit sees them (Bedrock counts images however sourced) ...
        with self.assertRaises(input_schema.ValidationException):
            self._enforce({"max_images": 3}, self._record([], s3_images=4))
        # ... but the size limits do not: they are not request payload
        self._enforce({"max_image_mb": 1, "max_payload_mb": 1},
                      self._record([], s3_images=4))

    def test_base64_size_arithmetic_matches_the_decoded_bytes(self):
        for size in (1, 2, 3, 1024, 5 * 1024 * 1024 - 1):
            encoded = base64.b64encode(b"x" * size).decode()
            block = {"image": {"source": {"bytes": encoded}}}
            self.assertEqual(validate_plan._inline_image_bytes(block), size)

    def test_a_declared_zero_means_zero_not_undeclared(self):
        """0 and null must read differently: a declared max_images 0 refuses
        ANY image (the stricter-fallback warning already reads 0 as
        declared, so the two readers now agree — review catch)."""
        with self.assertRaises(input_schema.ValidationException):
            self._enforce({"max_images": 0}, self._record([1024]))
        with self.assertRaises(input_schema.ValidationException):
            self._enforce({"max_payload_mb": 0}, self._record([1024]))

    def test_wrapped_base64_counts_payload_not_whitespace(self):
        """MIME/CLI-style base64 (76-char lines) is tolerated by b64decode
        and by Bedrock — whitespace must not inflate the computed size, and
        padding must be read past a trailing newline (review catch)."""
        raw = b"x" * (2 * 1024 * 1024 + 7)
        encoded = base64.b64encode(raw).decode()
        wrapped = "\n".join(encoded[i:i + 76]
                            for i in range(0, len(encoded), 76)) + "\n"
        block = {"image": {"source": {"bytes": wrapped}}}
        self.assertEqual(validate_plan._inline_image_bytes(block), len(raw))

    def test_a_quoted_limit_is_a_seed_time_finding_not_a_type_error(self):
        """A hand-edited '"max_image_mb": "3.5"' would string-multiply into
        a TypeError on every request — the battery names it instead."""
        entry = {"model_id": "v.img", "model_type": ["vision", "text"],
                 "capabilities": {
                     "temperature": True, "thinking": "absent",
                     "effort": False,
                     "image": {"limits": {"max_images": 100,
                                          "max_image_mb": "3.5",
                                          "max_payload_mb": True}}}}
        findings = bedrock_lib.contract_findings(entry)
        self.assertTrue(any("max_image_mb" in f for f in findings), findings)
        self.assertTrue(any("max_payload_mb" in f for f in findings),
                        findings)
        clean = {"model_id": "v.img", "model_type": ["vision", "text"],
                 "capabilities": {
                     "temperature": True, "thinking": "absent",
                     "effort": False,
                     "image": {"limits": {"max_images": 100,
                                          "max_image_mb": 3.5,
                                          "max_payload_mb": None}}}}
        self.assertEqual(bedrock_lib.contract_findings(clean), [])

    def test_a_malformed_limits_container_is_a_finding_and_degrades(self):
        """The round-1 guard covered VALUES; the container is the same
        hand-edit class (review catch): a non-dict image or limits must be
        a NAMED seed finding, and at plan time degrade to 'nothing
        declared' — never AttributeError on every request."""
        for caps_image in ("none", {"limits": "none"}):
            entry = {"model_id": "v.img", "model_type": ["vision", "text"],
                     "capabilities": {"temperature": True,
                                      "thinking": "absent", "effort": False,
                                      "image": caps_image}}
            findings = bedrock_lib.contract_findings(entry)
            self.assertTrue(any("must be a JSON object" in f
                                for f in findings), (caps_image, findings))
            spec = bedrock_lib.ModelSpec(entry)
            self.assertEqual(validate_plan._declared_limits(spec), {})
            # the full enforcement path survives the malformed entry
            validate_plan._enforce_image_limit(
                self._record([1024]), spec,
                validate_plan._declared_limits(spec))

    def test_declared_limits_drops_non_numeric_values(self):
        """A hand-edited artifact bypasses the seeder (resolve warns, never
        refuses), so quoted numbers must degrade to not-declared at the
        plan — including in the stricter-fallback comparison."""
        spec = self._spec({"max_images": "3", "max_image_mb": True,
                           "max_payload_mb": 16})
        self.assertEqual(validate_plan._declared_limits(spec),
                         {"max_payload_mb": 16})

    def test_the_measured_master_limits_are_pinned(self):
        with open(os.path.join(ROOT, "models",
                               "availablemodels_master.json")) as f:
            master = {m["model_id"]: m for m in json.load(f)["models"]}

        def limits(mid):
            return master[mid]["capabilities"]["image"]["limits"]

        for mid in ("anthropic.claude-fable-5-1", "anthropic.claude-opus-5",
                    "anthropic.claude-sonnet-5-5",
                    "anthropic.claude-sonnet-4-6",
                    "anthropic.claude-haiku-5-5"):
            self.assertEqual(limits(mid), {"max_images": 100,
                                           "max_image_mb": 3.5,
                                           "max_payload_mb": 20}, mid)
        self.assertEqual(
            limits("anthropic.claude-haiku-4-5-20251001-v1:0"),
            {"max_images": 20, "max_image_mb": 3.5, "max_payload_mb": 20})
        self.assertEqual(limits("mistral.pixtral-large-2502-v1:0"),
                         {"max_images": 8, "max_image_mb": None,
                          "max_payload_mb": 16})
        self.assertEqual(limits("meta.llama4-maverick-17b-instruct-v1:0"),
                         {"max_images": 3, "max_image_mb": None,
                          "max_payload_mb": 16})
        for mid in ("amazon.nova-pro-v1:0", "amazon.nova-2-lite-v1:0"):
            self.assertEqual(limits(mid), {"max_images": 100,
                                           "max_image_mb": None,
                                           "max_payload_mb": 16}, mid)


class TestSeededRegistryInThePlan(_WorkflowCase):
    """validate_plan over the seeded shape: the override gate, image limits, and
    the runtime quota scope."""

    def seeded_plan(self, request=None, records=None):
        with mock.patch.object(validate_plan.bedrock_lib, "load_registry",
                               return_value=SEEDED_REGISTRY):
            return self.plan(request=request, records=records)

    def test_an_unsupported_override_fails_the_plan_loudly(self):
        request = _request(model_chain=["anthropic.claude-sonnet-4-6"])
        request["inference_region"] = "eu"
        with self.assertRaises(ValidationException) as ctx:
            self.seeded_plan(request=request)
        self.assertIn("eu", str(ctx.exception))
        self.assertIn("supported regions", str(ctx.exception))

    def test_a_supported_override_routes_quota_batch_and_config(self):
        request = _request(model_chain=["amazon.nova-pro-v1:0"], mode="sync")
        request["inference_region"] = "eu-west-1"
        plan = self.seeded_plan(request=request)
        data = plan["data"]
        # Quota scope is the RUNTIME id — the geo profile for a profile-only
        # model even when pinned to a region.
        self.assertEqual(data["quota"]["resource"], "eu.amazon.nova-pro-v1:0")
        self.assertEqual(data["config"]["inference_region"], "eu-west-1")
        self.assertEqual(data["batch"]["batch_region"], "eu-west-1")

    def test_the_default_scope_reaches_quota_as_the_runtime_id(self):
        plan = self.seeded_plan(
            request=_request(model_chain=["anthropic.claude-sonnet-4-6"],
                             mode="sync"))
        self.assertEqual(plan["data"]["quota"]["resource"],
                         "global.anthropic.claude-sonnet-4-6")

    def test_the_override_is_in_the_hash_not_excluded(self):
        """Where a run is ALLOWED to execute is part of what the run is — an
        EU-residency rerun of a global run must be different work."""
        self.assertNotIn("inference_region", identity.EXCLUDED_KEYS)
        base = _request(model_chain=["amazon.nova-pro-v1:0"])
        routed = dict(base, inference_region="eu-west-1")
        digest = "d" * 64
        self.assertNotEqual(
            identity.compute_workflow_id(base, records_digest=digest,
                                         tool_version="t"),
            identity.compute_workflow_id(routed, records_digest=digest,
                                         tool_version="t"))

    def test_image_limits_reject_before_a_token_is_spent(self):
        """Bedrock-enforced `limits` fail the plan; `downscales_above` and
        `recommendations` never do — rejecting a request Bedrock would accept is
        a capability regression, not a guard."""
        # Real jpeg magic: parse_records validates the head since 2026-08-14,
        # and this test must fail on the COUNT limit, not the block shape.
        image = {"image": {"format": "jpeg", "source": {"bytes": _JPEG_B64}}}
        over = [{"record_id": "r0",
                 "messages": [{"role": "user",
                               "content": [image] * 21 + [{"text": "hi"}]}]}]
        with self.assertRaises(ValidationException) as ctx:
            self.seeded_plan(
                request=_request(model_chain=["anthropic.claude-sonnet-4-6"]),
                records=over)
        self.assertIn("21 images", str(ctx.exception))
        at_limit = [{"record_id": "r0",
                     "messages": [{"role": "user",
                                   "content": [image] * 20 + [{"text": "hi"}]}]}]
        plan = self.seeded_plan(
            request=_request(model_chain=["anthropic.claude-sonnet-4-6"],
                             mode="sync"),
            records=at_limit)
        self.assertEqual(plan["data"]["records_total"], 1)

    def test_a_malformed_inference_region_is_rejected_at_parse(self):
        for bad in ("moon", "eu_west_1", "EU-WEST-1!", "us-east"):
            with self.assertRaises(ValidationException):
                input_schema.parse_request(
                    {**_request(), "inference_region": bad})


class TestModelRegistrySeeder(unittest.TestCase):
    """The seeder's pure core (_resolve) plus the version-skip discipline."""

    @classmethod
    def setUpClass(cls):
        from states.seed_model_registry import lambda_function as seeder
        cls.seeder = seeder
        with open(os.path.join(ROOT, "models",
                               "availablemodels_master.json")) as f:
            cls.master = json.load(f)

    def test_a_generative_entry_missing_its_contract_keys_fails_the_seed(self):
        """The family tuples are retired, so an entry without explicit
        capabilities would silently get the module defaults — the seeder is
        the only moment the data changes, so it refuses there, naming the
        entry and the missing keys. Validation runs BEFORE the geo filter: a
        bad entry fails every deploy, not only the geos that carry it."""
        bad = {"models": [{"model_id": "anthropic.claude-test-6",
                           "model_type": ["text"], "geos": ["us"],
                           "capabilities": {"temperature": False}}]}
        with self.assertRaises(ValueError) as raised:
            self.seeder._resolve(bad, "eu", "v1")       # eu: geo-EXCLUDED
        self.assertIn("claude-test-6", str(raised.exception))
        self.assertIn("thinking", str(raised.exception))

    def test_thinking_required_demands_a_thinking_default(self):
        bad = {"models": [{"model_id": "anthropic.claude-test-6",
                           "model_type": ["text"], "geos": ["us"],
                           "capabilities": {"temperature": False,
                                            "thinking": "required",
                                            "effort": True}}]}
        with self.assertRaises(ValueError) as raised:
            self.seeder._resolve(bad, "us", "v1")
        self.assertIn("thinking_default", str(raised.exception))

    def test_embedding_entries_are_exempt_from_the_contract_keys(self):
        emb = {"models": [{"model_id": "vendor.embed-v1",
                           "model_type": ["embedding"], "geos": ["us"]}]}
        out = self.seeder._resolve(emb, "us", "v1")
        self.assertEqual(len(out["models"]), 1)


    def test_a_present_default_must_be_valid_in_any_mode(self):
        """default_thinking_type() silently coerces an invalid value to
        "disabled" and effort_default is emitted verbatim — a plausible typo
        ('adptive', 'lo') must die at seed time, the only moment the data
        changes, in EVERY mode (not just thinking: required)."""
        base = {"model_id": "anthropic.claude-test-6",
                "model_type": ["text"], "geos": ["us"]}
        bad = dict(base, capabilities={"temperature": False,
                                       "thinking": "optional",
                                       "thinking_default": "adptive",
                                       "effort": True})
        with self.assertRaises(ValueError) as raised:
            self.seeder._resolve({"models": [bad]}, "us", "v1")
        self.assertIn("thinking_default", str(raised.exception))
        bad = dict(base, capabilities={"temperature": False,
                                       "thinking": "optional",
                                       "effort": True,
                                       "effort_default": "lo"})
        with self.assertRaises(ValueError) as raised:
            self.seeder._resolve({"models": [bad]}, "us", "v1")
        self.assertIn("effort_default", str(raised.exception))

    def test_only_pure_embedders_are_exempt_from_the_contract_keys(self):
        """camelCase spelling is honored, but a DUAL-type
        ["embedding", "text"] entry is resolvable as a generative model and
        must carry the keys — exempting it would reopen the silent-defaults
        hole this validator closes."""
        pure = {"modelId": "v.embed2", "modelType": ["embedding"],
                "geos": ["us"]}
        out = self.seeder._resolve({"models": [pure]}, "us", "v1")
        self.assertEqual(len(out["models"]), 1)
        dual = {"model_id": "v.embed", "model_type": ["embedding", "text"],
                "geos": ["us"]}
        with self.assertRaises(ValueError) as raised:
            self.seeder._resolve({"models": [dual]}, "us", "v1")
        self.assertIn("v.embed", str(raised.exception))


    def test_embedder_classifier_agrees_between_lib_and_seeder(self):
        """One classifier (the seeder imports lib's is_converse_exempt) —
        this pins the seeder's refusal/exemption decision to it end-to-end
        over the shapes a hand edit can produce."""
        battery = [
            {"model_type": ["embedding"]},
            {"model_type": ["embedding", "text"]},
            {"model_type": "embedding"},
            {"model_type": "text,vision"},
            {"modelType": ["embedding"]},
            {"model_type": ["text"]},
            {},
        ]
        for entry in battery:
            lib_says = bedrock_lib.is_converse_exempt(entry)
            # the seeder exempts (never raises) iff the classifier says
            # exempt — a caps-less probe shows whether it demands them
            probe = dict(entry, model_id="x")
            try:
                self.seeder._require_capabilities(probe)
                seeder_exempts = True
            except ValueError:
                seeder_exempts = False
            self.assertEqual(lib_says, seeder_exempts, entry)


    def test_a_stray_default_on_a_thinking_absent_entry_still_seeds(self):
        """With thinking "absent" the default is never read and temperature
        is legitimately emitted — the contradiction check must not reject a
        runtime-valid entry over a leftover key."""
        entry = {"model_id": "amazon.nova-test-v1:0", "model_type": ["text"],
                 "geos": ["us"],
                 "capabilities": {"temperature": True, "thinking": "absent",
                                  "thinking_default": "adaptive",
                                  "effort": False}}
        out = self.seeder._resolve({"models": [entry]}, "us", "v1")
        self.assertEqual(len(out["models"]), 1)


    def test_the_seeder_imports_its_vocabularies_from_lib(self):
        """The gateway-tool zip carries lib/ at its root (the dispatcher
        already imports lib.identity the same way), so the seeder IMPORTS
        the contract vocabularies — identity, not mirrored copies, so a
        vocabulary change cannot fork between seed time and resolve time.
        The import is LAZY (inside the handler's try): a module-level
        import failure would skip _respond and hang the stack for the
        full CFN custom-resource timeout."""
        self.assertIs(self.seeder._bedrock_vocab(), bedrock_lib)
        import ast
        src = open(os.path.join(ROOT, "states", "seed_model_registry",
                                "lambda_function.py")).read()
        tree = ast.parse(src)
        top_imports = [n for n in tree.body
                       if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in top_imports:
            mods = ([a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""])
            for mod in mods:
                self.assertFalse(mod.startswith("lib"),
                                 f"module-level lib import: {mod}")

    def test_boolean_capability_keys_must_be_json_booleans(self):
        """bool("false") is True — a quoted boolean would silently flip a
        contract at runtime, and a wrong effort flag has no runtime
        recovery at all."""
        bad = {"model_id": "anthropic.claude-test-6", "model_type": ["text"],
               "geos": ["us"],
               "capabilities": {"temperature": "false",
                                "thinking": "optional", "effort": True}}
        with self.assertRaises(ValueError) as raised:
            self.seeder._resolve({"models": [bad]}, "us", "v1")
        self.assertIn("JSON boolean", str(raised.exception))

    def test_temperature_true_contradicts_an_adaptive_default(self):
        """A quiet request on such an entry carries thinking.type=adaptive,
        and Bedrock rejects sampling params beside reasoning — the
        contradiction dies at seed time instead of misleading the editor."""
        bad = {"model_id": "anthropic.claude-test-6", "model_type": ["text"],
               "geos": ["us"],
               "capabilities": {"temperature": True, "thinking": "required",
                                "thinking_default": "adaptive",
                                "effort": True}}
        with self.assertRaises(ValueError) as raised:
            self.seeder._resolve({"models": [bad]}, "us", "v1")
        self.assertIn("contradicts", str(raised.exception))

    def test_the_shipped_master_passes_the_seed_validation(self):
        for geo in ("us", "eu", "apac"):
            self.seeder._resolve(self.master, geo, "v1")

    def test_the_ap_region_prefix_aliases_to_the_apac_geo(self):
        """The template derives Geo from the deploy region's first hyphen-segment,
        and for every APAC region that segment is "ap" (ap-southeast-1) while the
        geo vocabulary says "apac" — so before the alias, an APAC deploy failed
        the whole stack on the seeder's own validator. us/eu pass through, which
        is why a us/eu test bed never caught it."""
        self.assertEqual(self.seeder._normalize_geo("ap"), "apac")
        self.assertEqual(self.seeder._normalize_geo("AP"), "apac")
        for geo in ("us", "eu", "apac"):
            self.assertEqual(self.seeder._normalize_geo(geo), geo)
        # ...and the aliased geo actually selects the apac models.
        apac = self.seeder._resolve(self.master,
                                    self.seeder._normalize_geo("ap"), "v1")
        self.assertEqual(apac["geo"], "apac")
        self.assertTrue(apac["models"])

    def test_region_families_without_a_model_list_fail_the_stack(self):
        """ca-central-1, sa-east-1, me-south-1, af-south-1, il-central-1: no
        model list exists for these families, and the right failure is loud, at
        deploy time, in the seeder — not a mystery empty registry later."""
        for prefix in ("ca", "sa", "me", "af", "il", "", None, "moon"):
            with self.assertRaises(ValueError, msg=prefix):
                self.seeder._normalize_geo(prefix)

    def test_geo_filtering_is_the_region_story(self):
        """Llama 4 is us-only in the master and must vanish from an EU
        artifact MECHANICALLY — ModelNotRegistered at plan time is the answer
        an EU deployment gives for it. (Nova Premier was the other us-only
        entry until 2026-08-18, when Amazon marked it Legacy —
        ResourceNotFoundException on invoke — and it left the master
        entirely.)"""
        eu = self.seeder._resolve(self.master, "eu", "v1")
        us = self.seeder._resolve(self.master, "us", "v1")
        eu_ids = {m["model_id"] for m in eu["models"]}
        us_ids = {m["model_id"] for m in us["models"]}
        self.assertNotIn("meta.llama4-maverick-17b-instruct-v1:0", eu_ids)
        self.assertNotIn("amazon.nova-premier-v1:0", eu_ids | us_ids)
        self.assertIn("meta.llama4-maverick-17b-instruct-v1:0", us_ids)
        self.assertIn("anthropic.claude-sonnet-4-6", eu_ids & us_ids)
        with self.assertRaises(bedrock_lib.ModelNotRegistered):
            bedrock_lib.resolve_model(
                eu["models"], "meta.llama4-maverick-17b-instruct-v1:0")

    def test_geo_scope_resolves_and_the_envelope_is_stamped(self):
        eu = self.seeder._resolve(self.master, "eu", "alpha.test")
        self.assertEqual(eu["registry_version"], "alpha.test")
        self.assertEqual(eu["geo"], "eu")
        nova = next(m for m in eu["models"]
                    if m["model_id"] == "amazon.nova-pro-v1:0")
        self.assertEqual(nova["inference_region"], "eu")
        self.assertNotIn("geos", nova)
        # ...and the resolved artifact is loadable end to end.
        spec = bedrock_lib.resolve_model(eu["models"], "amazon.nova-pro-v1:0")
        self.assertEqual(spec.resolve_runtime_model_id(),
                         "eu.amazon.nova-pro-v1:0")

    def test_every_master_entry_is_well_formed(self):
        """The master is data someone will edit by hand; malformed entries must
        fail HERE, not at the first plan against the deployed artifact."""
        geos = {"us", "eu", "apac"}
        scopes = {"global", "geo", "in-region", "us", "eu", "apac"}
        for entry in self.master["models"]:
            mid = entry["model_id"]
            for prefix in ("us.", "eu.", "apac.", "global."):
                self.assertFalse(mid.startswith(prefix),
                                 f"{mid}: master ids are BARE — routing lives "
                                 "in inference_region")
            self.assertTrue(set(entry["geos"]) <= geos, mid)
            self.assertTrue(entry["supported_regions"], mid)
            scope = entry["inference_region"]
            self.assertTrue(scope in scopes
                            or re.match(r"^[a-z]{2}-[a-z]+-\d$", scope), mid)
            self.assertNotIn("region", entry.get("batch_job") or {},
                             f"{mid}: batch follows inference_region — no "
                             "separate batch region (settled 2026-08-11)")
            pricing = entry.get("model_pricing") or {}
            # the master pins snake_case; camelCase is the legacy-artifact
            # spelling (estimate_cost reads both, the master writes one)
            self.assertNotIn("cacheReadPerTokenCost", pricing, mid)
            self.assertNotIn("cacheWritePerTokenCost", pricing, mid)
            # Cache pricing stopped being banned 2026-10-16 (GPT-6 caches
            # implicitly, so its entries price the traffic), but it stays
            # SHAPED: both keys or neither, positive numbers, and an entry
            # declaring the explicit-mode opt-out must price what the
            # opt-out cannot stop (repeats are still served from cache).
            cache_keys = [key for key in ("cache_write_per_token_cost",
                                          "cache_read_per_token_cost")
                          if key in pricing]
            if cache_keys:
                self.assertEqual(len(cache_keys), 2, mid)
                for key in cache_keys:
                    self.assertGreater(pricing[key], 0, mid)
            if (entry.get("capabilities") or {}).get("prompt_cache_mode"):
                self.assertEqual(len(cache_keys), 2, mid)
            if "vision" in entry.get("model_type", []) or \
                    "text" in entry.get("model_type", []):
                self.assertIn("capabilities", entry, mid)
                self.assertIn(entry["capabilities"]["thinking"],
                              ("absent", "optional", "required"), mid)
                if "thinking_default" in entry["capabilities"]:
                    self.assertIn(entry["capabilities"]["thinking_default"],
                                  ("adaptive", "disabled"), mid)
                if "effort_default" in entry["capabilities"]:
                    # per-WIRE vocabulary: the reasoning wire (GPT-6) adds
                    # registry-default-only values the Anthropic wire rejects
                    wire = entry["capabilities"].get("effort_wire",
                                                     "output_config")
                    self.assertIn(wire, bedrock_lib.EFFORT_WIRES, mid)
                    self.assertIn(
                        entry["capabilities"]["effort_default"],
                        bedrock_lib.EFFORT_WIRE_VOCABULARY[wire], mid)
                # The prefill ban is a request-shape rule, not a capability —
                # a flag here would imply some model accepts one.
                self.assertNotIn("assistant_prefill", entry["capabilities"], mid)

    def test_matching_installed_version_skips_the_write(self):
        s3 = mock.MagicMock()
        s3.get_object.return_value = {"Body": mock.MagicMock(
            read=lambda: json.dumps({"registry_version": "same"}).encode())}
        self.assertEqual(
            self.seeder._installed_version(s3, "b", "k"), "same")
        # A legacy bare-list artifact has no envelope — reports "" so the first
        # seed after cutover ALWAYS overwrites it.
        s3.get_object.return_value = {"Body": mock.MagicMock(
            read=lambda: json.dumps([{"modelId": "x"}]).encode())}
        self.assertEqual(self.seeder._installed_version(s3, "b", "k"), "")

    def test_delete_never_deletes_and_always_succeeds(self):
        """A missing registry bricks every in-flight execution, and the artifact
        may be shared with a replacement stack — Delete is a no-op SUCCESS."""
        responses = []
        with mock.patch.object(self.seeder, "_respond",
                               side_effect=lambda *a, **k: responses.append(a)):
            result = self.seeder.lambda_handler({"RequestType": "Delete"}, None)
        self.assertTrue(result["handled"])
        self.assertEqual(responses[0][2], "SUCCESS")


# ─────────────────────────────────────────────────────────────────────────────
# parse_partial_output — the copied salvage module (README.md §P6.2).
# These double as its GOLDEN VECTORS: fixed input -> fixed output, because the
# module will be byte-copied into consumer workflows and a drifted repair
# re-identifies work there. An intended behaviour change updates these AND
# bumps every consumer's .version.
# ─────────────────────────────────────────────────────────────────────────────

from lib.parse_partial_output import (  # noqa: E402
    escape_inner_quotes,
    parse_jsonl_line,
    parse_partial_json,
    parse_partial_xml,
)


class TestParsePartialJson(unittest.TestCase):

    def test_clean_json_is_not_marked_repaired(self):
        for text in ('{"a": 1, "b": [2, 3]}',
                     'Here is the answer:\n```json\n{"a": 1}\n```\nDone.',
                     '[{"x": "y"}] trailing prose'):
            value, repaired = parse_partial_json(text)
            self.assertIsNotNone(value, text)
            self.assertFalse(repaired,
                             "wrapping alone is not repair — the JSON was intact")

    def test_truncated_output_completes_to_the_coherent_prefix(self):
        """The max_tokens shape: cut mid-string, mid-key, or mid-array. The
        salvage completes, never rewrites — a dangling key or half-written
        string is DROPPED, not guessed at."""
        cases = [
            ('{"items": [{"id": 1, "name": "one"}, {"id": 2, "name": "tw',
             {"items": [{"id": 1, "name": "one"}, {"id": 2}]}),
            ('{"a": 1, "b":', {"a": 1}),
            ('{"a": 1, "b": [1, 2,', {"a": 1, "b": [1, 2]}),
            ('[{"scene": 1}, {"scene": 2}, {"sce', [{"scene": 1}, {"scene": 2}]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                value, repaired = parse_partial_json(text)
                self.assertEqual(value, expected)
                self.assertTrue(repaired)

    def test_unsalvageable_input_is_none_false_not_an_exception(self):
        """A failed salvage is a per-record outcome, not an exception path."""
        for text in ("", None, "no json here at all", "}{", "]["):
            with self.subTest(text=text):
                self.assertEqual(parse_partial_json(text), (None, False))

    def test_escaped_quotes_and_brackets_inside_strings_do_not_confuse_it(self):
        text = '{"quote": "she said \\"hi\\"", "path": "a{b[c"}'
        value, repaired = parse_partial_json(text)
        self.assertEqual(value["quote"], 'she said "hi"')
        self.assertEqual(value["path"], "a{b[c")
        self.assertFalse(repaired)


class TestParsePartialXml(unittest.TestCase):

    def test_clean_xml_is_not_marked_repaired(self):
        for text in ("<result><item>a</item></result>",
                     "Sure!\n<result><item>a</item></result>\nHope that helps.",
                     '<result attr="v"><empty/></result>'):
            value, repaired = parse_partial_xml(text)
            self.assertIsNotNone(value, text)
            self.assertIn("<result", value)
            self.assertFalse(repaired)

    def test_truncated_xml_closes_open_elements_in_stack_order(self):
        cases = [
            # Text nodes are UNDELIMITED (unlike JSON strings), so the salvage
            # cannot tell a complete "2" from a truncated "25" — it keeps the
            # text and lets `repaired` flag the uncertainty. The cut-off close
            # tag itself is discarded and the stack closes cleanly.
            ("<results><scene><id>1</id></scene><scene><id>2</i",
             "<results><scene><id>1</id></scene>"
             "<scene><id>2</id></scene></results>"),
            ("<results><scene><id>1</id>",
             "<results><scene><id>1</id></scene></results>"),
            ("<a><b>text</b><c>part",
             "<a><b>text</b><c>part</c></a>"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                value, repaired = parse_partial_xml(text)
                self.assertEqual(value, expected)
                self.assertTrue(repaired)

    def test_unsalvageable_input_is_none_false(self):
        for text in ("", None, "no xml here", "just < a less-than"):
            with self.subTest(text=text):
                self.assertEqual(parse_partial_xml(text), (None, False))

    def test_a_mismatched_close_stops_at_the_damage(self):
        """A wrong close tag is model confusion, not truncation — salvage keeps
        what was coherent BEFORE it rather than papering over the mismatch."""
        value, repaired = parse_partial_xml("<a><b>ok</b></wrong>")
        self.assertEqual(value, "<a><b>ok</b></a>")
        self.assertTrue(repaired)


class TestEscapeInnerQuotes(unittest.TestCase):
    """The third verb (2026-08-26): the pass only ever ADDS a backslash.
    Golden vectors — the module is byte-copied into consumers."""

    CHYRON = ('{"shot_id": 7, "texts_on_screen": ["POPBEAT", '
              '"MIRAVOX RETURNS WITH "STILL GOLD""], "description": "chyron"}')

    def test_the_measured_malformation_is_recovered_exactly(self):
        fixed = escape_inner_quotes(self.CHYRON)
        self.assertEqual(
            json.loads(fixed)["texts_on_screen"],
            ["POPBEAT", 'MIRAVOX RETURNS WITH "STILL GOLD"'])

    def test_already_valid_lines_come_back_byte_identical(self):
        for line in ('{"a": "she said \\"hi\\"", "b": [1, "x, y"]}',
                     '{"a": 1}',
                     '{"quote_at_end": "ends with \\""}'):
            with self.subTest(line=line):
                self.assertEqual(escape_inner_quotes(line), line)

    def test_it_only_ever_adds_backslashes(self):
        for line in (self.CHYRON, '{"broken": "a " b " c"}', '{"a": "trunc'):
            with self.subTest(line=line):
                out = escape_inner_quotes(line)
                self.assertEqual(out.replace("\\", ""),
                                 line.replace("\\", ""))
                self.assertGreaterEqual(len(out), len(line))


class TestParseJsonlLine(unittest.TestCase):
    """The JSONL ladder. `required_keys` is the caller's MINIMUM — join ids,
    never the full schema (user decision 2026-08-26: a recovered partial
    that lost optional metadata is kept, not discarded)."""

    def test_a_valid_line_is_strict_tier(self):
        obj, tier = parse_jsonl_line('{"shot_id": 3, "a": 1}', ("shot_id",))
        self.assertEqual((obj["shot_id"], tier), (3, "strict"))

    def test_the_chyron_line_is_escaped_tier(self):
        obj, tier = parse_jsonl_line(TestEscapeInnerQuotes.CHYRON,
                                     ("shot_id",))
        self.assertEqual(tier, "escaped")
        self.assertEqual(obj["texts_on_screen"][1],
                         'MIRAVOX RETURNS WITH "STILL GOLD"')

    def test_a_truncated_line_keeps_the_honest_prefix(self):
        obj, tier = parse_jsonl_line(
            '{"shot_id": 3, "description": "ok", "time_of_day": "indetermina',
            ("shot_id",))
        self.assertEqual(tier, "prefix")
        # The incomplete tail value is DROPPED WHOLE, never trimmed — the
        # measured difference from jsonrepair AND partial-json, which both
        # return the fabricated "indetermina".
        self.assertEqual(obj, {"shot_id": 3, "description": "ok"})

    def test_mid_line_damage_stops_at_the_damage_and_keeps_the_prefix(self):
        obj, tier = parse_jsonl_line(
            '{"shot_id": 3, "a": "x" garbage, "b": 1}', ("shot_id",))
        self.assertEqual((obj, tier), ({"shot_id": 3, "a": "x"}, "prefix"))

    def test_a_prefix_that_lost_a_required_key_stays_a_bad_line(self):
        obj, error = parse_jsonl_line(
            '{"description": "ok", "people": ["cut before the id',
            ("shot_id",))
        self.assertIsNone(obj)
        self.assertIn("prefix lost required shot_id", error)

    def test_alternative_key_sets_admit_either_document_shape(self):
        gate = ({"overall_content_type"}, {"start_id", "end_id"})
        header, tier = parse_jsonl_line(
            '{"overall_content_type": "News", "top_3_genres": ["News", "x',
            gate)
        self.assertEqual(tier, "prefix")
        self.assertEqual(header["overall_content_type"], "News")
        segment, tier = parse_jsonl_line(
            '{"start_id": 0, "end_id": 4, "segment_topic": "cut mid-topi',
            gate)
        self.assertEqual(tier, "prefix")
        self.assertEqual((segment["start_id"], segment["end_id"]), (0, 4))
        refused, error = parse_jsonl_line('{"start_id": 0, "end_i', gate)
        self.assertIsNone(refused)
        self.assertIn("prefix lost required", error)

    def test_nothing_salvageable_returns_the_strict_error(self):
        obj, error = parse_jsonl_line("not json at all", ("shot_id",))
        self.assertIsNone(obj)
        self.assertIn("unrecovered", error)

    def test_a_number_cut_mid_digits_is_dropped_never_trimmed(self):
        # 2026-08-26 review find, verified by execution BEFORE the fix:
        # '"end_id": 1780' cut after 178 salvaged to end_id=178 — a
        # fabricated value through the anti-fabrication gate. The tail
        # number is now dropped whole, so the id is honestly LOST and
        # the gate refuses the prefix.
        obj, error = parse_jsonl_line(
            '{"start_id": 120, "end_id": 178', ({"start_id", "end_id"},))
        self.assertIsNone(obj)
        self.assertIn("prefix lost required end_id", error)
        # Same class on a leading join key: no wrong-shot document.
        obj, error = parse_jsonl_line('{"shot_id": 13', ("shot_id",))
        self.assertIsNone(obj)
        # A number FOLLOWED by structure was completed by the model and
        # must survive — the drop is one-shot on the raw tail only.
        obj, tier = parse_jsonl_line('{"a": 1, "b": [1, 2,', ("a",))
        self.assertEqual((obj, tier), ({"a": 1, "b": [1, 2]}, "prefix"))

    def test_a_bare_string_gate_is_refused_not_exploded(self):
        # list("shot_id") is seven single-character keys — a natural typo
        # that silently disabled rung 3. Loud, immediate, and typed.
        with self.assertRaises(TypeError):
            parse_jsonl_line('{"broken": "x', "shot_id")


class TestCallbackToken(_GatewayCase):
    """
    `callback_task_token` (README.md §11; designed jointly with the
    conversation-analysis workflow): the service-side
    completion notification llm-prompt-driven workflows wait on. The rules
    pinned here, each load-bearing:

      * TRANSPORT ONLY — never in the hash (a waiting and a non-waiting
        submit of the same records are ONE workflow_id);
      * registered BEFORE StartExecution (the waker can never fire first);
      * absorbed-FINISHED completes synchronously (the paid-work-reuse path);
      * with a token present, failures RAISE — a waitForTaskToken caller
        never reads the return value, so a returned error dict is a silent
        hang until the consumer's deadman.
    """

    TOKEN = "AAAAKgAAAAIAAAAAAAAAA-fake-task-token"  # nosec B105

    def setUp(self):
        super().setUp()
        self.ddb = mock.MagicMock()
        for patch in (mock.patch.object(self.gateway, "_ddb", self.ddb),
                      mock.patch.object(self.gateway, "CALLBACK_TABLE",
                                        "cb-table")):
            patch.start()
            self.addCleanup(patch.stop)

    def test_token_is_transport_only_never_in_the_hash(self):
        plain = self.submit()
        waiting = self.submit(callback_task_token=self.TOKEN)
        self.assertEqual(plain["workflow_id"], waiting["workflow_id"])

    def test_without_a_token_the_table_is_never_touched(self):
        self.submit()
        self.ddb.put_item.assert_not_called()

    def test_registration_happens_before_start_execution(self):
        order = []
        self.ddb.put_item.side_effect = \
            lambda **kw: order.append(("put", kw)) or {}

        def start(**kw):
            order.append(("start", kw))
            return {"executionArn": "arn:x"}
        self.sfn.start_execution.side_effect = start

        result = self.submit(callback_task_token=self.TOKEN)
        self.assertEqual([step for step, _ in order], ["put", "start"])
        item = order[0][1]["Item"]
        self.assertTrue(item["execution_arn"]["S"]
                        .endswith(f":{result['workflow_id']}"))
        handle = json.loads(item["handle"]["S"])
        self.assertEqual(
            set(handle), {"job_id", "workflow_id", "records_total",
                          "output_s3_prefix", "request_s3uri"})
        self.assertIn("ttl", item)

    def test_absorbed_running_keeps_the_row_and_completes_nothing(self):
        self.absorb(status="RUNNING", callback_task_token=self.TOKEN)
        self.ddb.put_item.assert_called_once()
        self.sfn.send_task_success.assert_not_called()
        self.ddb.delete_item.assert_not_called()

    def test_absorbed_finished_completes_synchronously(self):
        """The paid-work-reuse path: identical rendered bytes absorb into a
        finished run and the consumer wakes IN THIS CALL with the verdict."""
        result = self.absorb(status="SUCCEEDED",
                             callback_task_token=self.TOKEN)
        call = self.sfn.send_task_success.call_args.kwargs
        self.assertEqual(call["taskToken"], self.TOKEN)
        payload = json.loads(call["output"])
        self.assertEqual(payload["status"], "SUCCEEDED")
        self.assertIn("message", payload)
        self.assertEqual(payload["workflow_id"], result["workflow_id"])
        self.ddb.delete_item.assert_called_once()

    def test_absorbed_failed_is_a_verdict_not_an_exception(self):
        """FAILED completes via SendTaskSUCCESS carrying error — the verdict
        is DATA for the consumer's Choice state (§7d honest partials)."""
        self.absorb(status="FAILED", callback_task_token=self.TOKEN)
        self.sfn.send_task_failure.assert_not_called()
        payload = json.loads(
            self.sfn.send_task_success.call_args.kwargs["output"])
        self.assertEqual(payload["status"], "FAILED")
        self.assertIn("error", payload)

    def test_start_failure_with_a_token_raises_and_cleans_the_row(self):
        self.sfn.start_execution.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow"}},
            "StartExecution")
        with self.assertRaises(RuntimeError):
            self.submit(callback_task_token=self.TOKEN)
        self.ddb.delete_item.assert_called_once()

    def test_start_failure_without_a_token_still_returns_the_error_dict(self):
        self.sfn.start_execution.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow"}},
            "StartExecution")
        result = self.submit()
        self.assertEqual(result["errorCode"], "SFN_START_FAILED")

    def test_malformed_token_raises_before_any_write(self):
        for bad in ("", "   ", 42, "x" * 3000):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    self.submit(callback_task_token=bad)
        self.ddb.put_item.assert_not_called()
        self.s3.put_object.assert_not_called()

    def test_missing_table_config_raises(self):
        with mock.patch.object(self.gateway, "CALLBACK_TABLE", ""):
            with self.assertRaises(RuntimeError):
                self.submit(callback_task_token=self.TOKEN)

    def test_choke_point_raises_returned_refusals_with_a_token(self):
        # fi's dispatcher-level net, backported (review, 2026-09-06): any
        # `return _error(...)` — present or future — is converted to a
        # raise when a callback token rides the event. Through
        # lambda_handler, because _run_inference-level tests cannot see
        # the wrapper.
        with self.assertRaises(RuntimeError) as ctx:
            self.gateway.lambda_handler(
                {"callback_task_token": self.TOKEN},
                _AgentToolContext("no_such_tool"))
        self.assertIn("UNKNOWN_TOOL", str(ctx.exception))

    def test_choke_point_lets_absorbed_failed_data_through(self):
        # the discriminator is `error is True` (the _error refusal
        # shape): an absorbed-FAILED result carries its cause as a
        # STRING beside the handle, AFTER complete_now redeemed the
        # token — data, never a detonation
        self.sfn.start_execution.side_effect = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        self.sfn.describe_execution.return_value = {
            "status": "FAILED", "cause": "boom"}
        event = {"records": [{"record_id": "r0",
                              "messages": [{"role": "user",
                                            "content": [{"text": "Hi"}]}]}],
                 "model_chain": ["us.anthropic.claude-sonnet-5-v1:0"],
                 "use_batch": False,
                 "callback_task_token": self.TOKEN}
        out = self.gateway.lambda_handler(event,
                                          _AgentToolContext("run_inference"))
        self.assertEqual(out["error"], "boom")   # returned, not raised
        self.sfn.send_task_success.assert_called_once()

    def test_absorb_describe_failure_deletes_row_and_raises(self):
        # fi's StartExecution-failure symmetry: the raise tells the
        # holder the submit failed (it will consume its own token), so
        # the row THIS call registered must go
        self.sfn.start_execution.side_effect = ClientError(
            {"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}},
            "StartExecution")
        self.sfn.describe_execution.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow"}},
            "DescribeExecution")
        with self.assertRaises(ClientError):
            self.submit(callback_task_token=self.TOKEN)
        self.ddb.delete_item.assert_called_once()


class TestCallbackWaker(unittest.TestCase):
    """callback_waker.py — the EventBridge-invoked half of README.md §11."""

    @classmethod
    def setUpClass(cls):
        gateway_dir = os.path.join(ROOT, "gateway_tool")
        if gateway_dir not in sys.path:
            sys.path.insert(0, gateway_dir)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "callback_waker", os.path.join(gateway_dir, "callback_waker.py"))
        cls.waker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.waker)

    def setUp(self):
        self.ddb = mock.MagicMock()
        self.sfn = mock.MagicMock()
        for patch in (mock.patch.object(self.waker, "_ddb", self.ddb),
                      mock.patch.object(self.waker, "_sfn", self.sfn),
                      mock.patch.object(self.waker, "CALLBACK_TABLE",
                                        "cb-table")):
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def _row(token="tok-1"):  # nosec B107
        return {"execution_arn": {"S": "arn:exec:wf-1"},
                "token_hash": {"S": "h" * 32},
                "token": {"S": token},
                "handle": {"S": json.dumps(
                    {"job_id": "arn:exec:wf-1", "workflow_id": "wf-1",
                     "records_total": 3,
                     "output_s3_prefix": "s3://b/p",
                     "request_s3uri": "s3://b/p/input_params.json"})}}

    def _event(self, status="SUCCEEDED"):
        return {"detail": {"executionArn": "arn:exec:wf-1",
                           "status": status}}

    def test_completes_every_row_with_the_merged_payload(self):
        self.ddb.query.return_value = {
            "Items": [self._row("tok-1"), self._row("tok-2")]}
        result = self.waker.lambda_handler(self._event("FAILED"), None)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(self.ddb.delete_item.call_count, 2)
        payload = json.loads(
            self.sfn.send_task_success.call_args.kwargs["output"])
        self.assertEqual(payload["status"], "FAILED")     # verdict as DATA
        self.assertEqual(payload["workflow_id"], "wf-1")  # handle survives
        self.assertIn("summary.json", payload["message"])

    def test_token_gone_is_benign_and_the_row_is_still_cleaned(self):
        """The consumer's deadman fired first, or a duplicate event won —
        logged, counted, deleted, never raised."""
        self.ddb.query.return_value = {"Items": [self._row()]}
        self.sfn.send_task_success.side_effect = ClientError(
            {"Error": {"Code": "TaskTimedOut", "Message": "gone"}},
            "SendTaskSuccess")
        result = self.waker.lambda_handler(self._event(), None)
        self.assertEqual((result["completed"], result["gone"]), (0, 1))
        self.ddb.delete_item.assert_called_once()

    def test_access_denied_keeps_the_row_and_raises_for_redelivery(self):
        # EventBridge only redelivers on FUNCTION ERROR — a normal return
        # strands the kept row until the consumer's deadman (fi's fix,
        # backported 2026-09-05)
        self.ddb.query.return_value = {"Items": [self._row()]}
        self.sfn.send_task_success.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no"}},
            "SendTaskSuccess")
        with self.assertRaises(RuntimeError):
            self.waker.lambda_handler(self._event(), None)
        self.ddb.delete_item.assert_not_called()

    def test_foreign_execution_is_one_query_miss(self):
        """Agent-submitted runs have no waiter — the common case."""
        self.ddb.query.return_value = {"Items": []}
        result = self.waker.lambda_handler(self._event(), None)
        self.assertEqual(result["completed"], 0)
        self.sfn.send_task_success.assert_not_called()

    def test_query_pagination_is_followed(self):
        self.ddb.query.side_effect = [
            {"Items": [self._row("tok-1")], "LastEvaluatedKey": {"k": "v"}},
            {"Items": [self._row("tok-2")]}]
        result = self.waker.lambda_handler(self._event(), None)
        self.assertEqual(result["completed"], 2)

    def test_non_sfn_event_is_ignored(self):
        result = self.waker.lambda_handler({"detail": {}}, None)
        self.assertEqual(result["completed"], 0)
        self.ddb.query.assert_not_called()


class TestCallbackTemplate(_CfnLoader):
    """The §11 template surface, pinned."""

    def test_callback_table_shape(self):
        table = self.tpl["Resources"]["CallbackTokensTable"]["Properties"]
        self.assertEqual(
            [(k["AttributeName"], k["KeyType"]) for k in table["KeySchema"]],
            [("execution_arn", "HASH"), ("token_hash", "RANGE")])
        self.assertTrue(table["TimeToLiveSpecification"]["Enabled"])

    def test_completion_rule_covers_terminal_statuses_orchestrator_only(self):
        """Orchestrator ONLY, unlike QuotaTerminalRule's three machines:
        consumers hold orchestrator execution ARNs, and the children's
        terminal events would be pure Query misses."""
        pattern = (self.tpl["Resources"]["CallbackCompletionRule"]
                   ["Properties"]["EventPattern"])
        self.assertEqual(sorted(pattern["detail"]["status"]),
                         ["ABORTED", "FAILED", "SUCCEEDED", "TIMED_OUT"])
        machines = json.dumps(pattern["detail"]["stateMachineArn"])
        self.assertIn("InferenceStateMachine", machines)
        self.assertNotIn("SyncInferenceStateMachine", machines)
        self.assertNotIn("BatchInferenceStateMachine", machines)

    def test_send_task_success_is_account_region_scoped(self):
        """REVERSED 2026-10-09 (user correction of the 2026-08-17 record):
        SendTask* DOES accept state-machine ARN scoping. The token's
        machine is the CALLER's — one of ours, or a customer's own machine
        calling the gateway with a callback token — and unknowable at
        deploy time, so the honest bound is every machine in THIS
        account+region, never a bare "*" (cross-account/region reach) and
        never a specific machine (breaks the cross-stack callback
        pattern). Runtime proof rides the deploy's callback E2E leg."""
        want = {"Sub": "arn:aws:states:${AWS::Region}:${AWS::AccountId}"
                       ":stateMachine:*"}
        for role in ("GatewayToolRole", "CallbackWakerRole"):
            statements = [s for s in self.tpl["Resources"][role]["Properties"]
                          ["Policies"][0]["PolicyDocument"]["Statement"]
                          if "states:SendTaskSuccess" in json.dumps(
                              s.get("Action", []))]
            self.assertEqual(len(statements), 1, role)
            self.assertEqual(statements[0]["Resource"], want, role)
            nag = json.dumps(self.tpl["Resources"][role].get("Metadata", {}))
            self.assertIn("W11", nag, role)
            self.assertNotIn("bearer credential", nag,
                             f"{role}: the W11 reason still carries the "
                             "overturned cannot-be-scoped claim")

    def test_gateway_env_carries_the_table(self):
        self.assertIn("CALLBACK_TABLE", self.env("GatewayToolLambda"))
        self.assertIn("CALLBACK_TABLE", self.env("CallbackWakerLambda"))

    def test_waker_rule_target_has_an_invoke_permission(self):
        perm = (self.tpl["Resources"]["CallbackCompletionRulePermission"]
                ["Properties"])
        self.assertEqual(perm["Principal"], "events.amazonaws.com")
        self.assertIn("CallbackCompletionRule",
                      json.dumps(perm["SourceArn"]))

    def test_waker_handler_and_package_ride_the_gateway_zip(self):
        props = self.tpl["Resources"]["CallbackWakerLambda"]["Properties"]
        self.assertEqual(props["Handler"], "callback_waker.lambda_handler")
        self.assertIn("GatewayToolPackage", json.dumps(props["Code"]))


class TestStreamedBatchInput(unittest.TestCase):
    """put_jsonl_streamed / iter_jsonl — the 668 MB OOM fix: memory is
    bounded by one part, malformed input names its line, and a failed
    stream aborts the multipart upload instead of leaving a torso."""

    def setUp(self):
        import lib.s3_utils as s3u
        self.s3u = s3u
        self.fake = _FakeS3()
        self._orig = s3u.s3
        s3u.s3 = self.fake

    def tearDown(self):
        self.s3u.s3 = self._orig

    def test_multiple_parts_reassemble_in_order(self):
        # rows are OBJECTS; serialization is put_jsonl_streamed's own
        # (jsonl_line — one home for the byte convention)
        rows = [{"i": i} for i in range(100)]
        n = self.s3u.put_jsonl_streamed("b", "k", iter(rows),
                                        part_size=64)
        self.assertEqual(n, 100)
        parts = [c for c in self.fake.calls if c[0] == "upload_part"]
        self.assertGreater(len(parts), 1)     # actually multipart
        body = self.fake.objects[("b", "k")].decode()
        self.assertEqual(body.splitlines(),
                         [self.s3u.jsonl_line(r) for r in rows])

    def test_streamed_and_whole_file_writers_share_the_bytes(self):
        # put_jsonl and put_jsonl_streamed must write byte-identical
        # objects for the same rows — these bytes are identity-digested
        rows = [{"a": "é", "n": 1}, {"b": [1, 2]}]
        self.s3u.put_jsonl("b", "whole", rows)
        self.s3u.put_jsonl_streamed("b", "streamed", iter(rows))
        self.assertEqual(self.fake.objects[("b", "whole")],
                         self.fake.objects[("b", "streamed")])

    def test_a_failing_stream_aborts_the_upload(self):
        def boom():
            yield {"i": 0}
            raise RuntimeError("source died")
        with self.assertRaises(RuntimeError):
            self.s3u.put_jsonl_streamed("b", "k", boom())
        self.assertIn(("abort_multipart_upload", "b", "k"),
                      self.fake.calls)
        self.assertNotIn(("b", "k"), self.fake.objects)

    def test_iter_jsonl_names_the_bad_line(self):
        # ValidationException, same contract as get_jsonl — an anonymous
        # ValueError would not be classified as fail-immediately (§9)
        from lib.custom_exceptions import ValidationException
        self.fake.put("b", "k", '{"ok": 1}\nnot json\n')
        with self.assertRaises(ValidationException) as cm:
            list(self.s3u.iter_jsonl("b", "k"))
        self.assertIn("line 2", str(cm.exception))

    def test_iter_jsonl_reports_file_lines_but_yields_record_indices(self):
        # blank lines are skipped from the yielded index but still count
        # in the reported line number, matching get_jsonl's numbering
        from lib.custom_exceptions import ValidationException
        self.fake.put("b", "k", '{"a": 1}\n\n{"b": 2}\nnope\n')
        rows = []
        with self.assertRaises(ValidationException) as cm:
            for item in self.s3u.iter_jsonl("b", "k"):
                rows.append(item)
        self.assertEqual(rows, [(0, {"a": 1}), (1, {"b": 2})])
        self.assertIn("line 4", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
