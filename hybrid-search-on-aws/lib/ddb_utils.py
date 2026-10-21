"""
ddb_utils.py — shared DynamoDB client and helpers.

Single boto3 DynamoDB resource shared across all callers in the same
Lambda container. All table operations go through this module.
"""
from __future__ import annotations

import time
import logging

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from lib.input_schema import INDEX_STATE_TABLE, DEDUP_TABLE, SEARCH_HISTORY_TABLE, AWS_REGION

log = logging.getLogger(__name__)

_dynamodb             = boto3.resource("dynamodb", region_name=AWS_REGION)
_table                = None
_dedup_table          = None
_search_history_table = None


def _get_table():
    global _table
    if _table is None:
        _table = _dynamodb.Table(INDEX_STATE_TABLE)
    return _table


def _get_dedup_table():
    global _dedup_table
    if _dedup_table is None:
        _dedup_table = _dynamodb.Table(DEDUP_TABLE)
    return _dedup_table


def _get_search_history_table():
    global _search_history_table
    if _search_history_table is None:
        _search_history_table = _dynamodb.Table(SEARCH_HISTORY_TABLE)
    return _search_history_table


get_table                = _get_table
get_dedup_table          = _get_dedup_table
get_search_history_table = _get_search_history_table


# ── Generic helpers ────────────────────────────────────────────────────────────

def put_item(item: dict) -> None:
    _get_table().put_item(Item=item)


def get_item(pk: str, sk: str) -> dict | None:
    resp = _get_table().get_item(Key={"pk": pk, "sk": sk})
    return resp.get("Item")


def batch_get_items(pk: str, sk_values: list[str]) -> list[dict]:
    """
    BatchGetItem for multiple sk values sharing the same pk.
    Returns items in arbitrary order. Missing items are silently absent.
    """
    if not sk_values:
        return []
    table  = _get_table()
    items: list[dict] = []
    for i in range(0, len(sk_values), 100):
        chunk = sk_values[i:i + 100]
        request_items = {
            table.name: {
                "Keys": [{"pk": pk, "sk": sk} for sk in chunk],
            }
        }
        while request_items:
            resp = table.meta.client.batch_get_item(RequestItems=request_items)
            items.extend(resp.get("Responses", {}).get(table.name, []))
            request_items = resp.get("UnprocessedKeys") or {}
    return items


def delete_item(pk: str, sk: str) -> None:
    _get_table().delete_item(Key={"pk": pk, "sk": sk})


def conditional_delete_item(pk: str, sk: str, condition: str,
                            attr_values: dict | None = None) -> bool:
    """
    DeleteItem with a ConditionExpression.
    Returns True on success, False on ConditionalCheckFailedException.
    Raises on all other errors.
    """
    kwargs: dict = {
        "Key": {"pk": pk, "sk": sk},
        "ConditionExpression": condition,
    }
    if attr_values:
        kwargs["ExpressionAttributeValues"] = attr_values
    try:
        _get_table().delete_item(**kwargs)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def conditional_put_item(item: dict, condition: str, attr_names: dict | None = None,
                         attr_values: dict | None = None) -> bool:
    """
    PutItem with a ConditionExpression.
    Returns True on success, False on ConditionalCheckFailedException.
    Raises on all other errors.
    """
    kwargs = {
        "Item": item,
        "ConditionExpression": condition,
    }
    if attr_names:
        kwargs["ExpressionAttributeNames"] = attr_names
    if attr_values:
        kwargs["ExpressionAttributeValues"] = attr_values

    try:
        _get_table().put_item(**kwargs)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def query_pk(pk: str, scan_forward: bool = True, limit: int | None = None) -> list[dict]:
    """
    Return items with a given pk, ordered by sk (asc if scan_forward=True).
    Paginates automatically. Pass limit to stop after the first N items.
    """
    table = _get_table()
    kwargs: dict = {
        "KeyConditionExpression": Key("pk").eq(pk),
        "ScanIndexForward": scan_forward,
    }
    if limit is not None:
        kwargs["Limit"] = limit

    items: list[dict] = []
    while True:
        resp = table.query(**kwargs)
        items.extend(resp.get("Items", []))
        last = resp.get("LastEvaluatedKey")
        if not last or (limit is not None and len(items) >= limit):
            break
        kwargs["ExclusiveStartKey"] = last
    return items[:limit] if limit is not None else items


def batch_write(items: list[dict]) -> None:
    """BatchWrite put requests. Handles up to 25 items per batch."""
    table = _get_table()
    for i in range(0, len(items), 25):
        with table.batch_writer() as batch:
            for item in items[i:i + 25]:
                batch.put_item(Item=item)


def batch_delete(keys: list[dict]) -> None:
    """BatchWrite delete requests. keys is a list of {pk, sk} dicts."""
    table = _get_table()
    for i in range(0, len(keys), 25):
        with table.batch_writer() as batch:
            for key in keys[i:i + 25]:
                batch.delete_item(Key=key)


def ttl_from_now(seconds: int) -> int:
    """Return a Unix epoch TTL value seconds from now."""
    return int(time.time()) + seconds


# Single timestamp format for every DDB record — is_rebuild_stale() PARSES
# this exact format, so all writers must share one definition.
ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def now_iso() -> str:
    return time.strftime(ISO_FORMAT, time.gmtime())
