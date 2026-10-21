"""
AgentCore Gateway tool dispatcher — hybrid-search ADMIN target.

Separate Lambda + separate execution role + separate gateway target from
the normal-caller tools (README.md Design record §5, decision D3): the
admin/normal split is an IAM boundary, never a payload flag. Normal
agents' gateway target has no permission to invoke this function.

Tools:
  configure_search_backend — update an EXISTING collection's retrieval
      weights / scopes / schema_map. Weight edits set weights_locked
      (weight-freeze: normal callers can still ADD scopes, never edit
      weights).
  delete_collection — full sweep: S3 data prefix, S3 Vectors indexes,
      state/lock/pending/asset DDB keys, dedup records, registry record.
      The "default" collection requires force=true.
"""
from __future__ import annotations

import json
import logging
import os
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from common import error as _error, paths_to_ddb as _paths_to_ddb, \
    resolve_tool as _resolve_tool, unmarshal_paths as _unmarshal_paths, \
    valid_collection_id as _valid_collection_id, \
    validate_retrieval_paths as _validate_retrieval_paths, s3v_index_name, \
    warn_idempotency_token as _warn_idempotency_token

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_ddb = boto3.client("dynamodb")
_s3  = boto3.client("s3")

COLLECTION_TABLE  = os.environ["COLLECTION_TABLE"]
INDEX_STATE_TABLE = os.environ["INDEX_STATE_TABLE"]
DEDUP_TABLE       = os.environ["DEDUP_TABLE"]
PROXY_BUCKET      = os.environ["PROXY_BUCKET"]
VECTOR_BUCKET     = os.environ.get("VECTOR_BUCKET", "")
TOOL_VERSION      = os.environ.get("TOOL_VERSION", "unknown")

def lambda_handler(event: dict, context) -> dict:
    logger.info("Event keys: %s", sorted(event.keys()))
    tool = _resolve_tool(context)

    # Both admin tools are condition-guarded sync mutations that honour no
    # idempotency token; accept and log rather than reject (see common.py).
    _token_note = _warn_idempotency_token(event)
    if _token_note:
        logger.warning("%s: %s", tool, _token_note)

    if tool == "configure_search_backend":
        return _configure(event)
    if tool == "delete_collection":
        return _delete_collection(event)
    if tool == "describe_tool":
        return {"version": TOOL_VERSION,
                "description": "hybrid-search ADMIN tools (weights, deletion)"}
    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ── configure_search_backend ───────────────────────────────────────────────────

def _configure(event: dict) -> dict:
    collection_id = (event.get("collection_id") or "").strip()
    if not collection_id:
        return _error("collection_id is required", "MISSING_PARAMETER")

    sets: list[str] = ["updated_at = :now"]
    values: dict = {":now": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    weight_edit = False

    for field, placeholder in (("retrieval_paths", ":rp"),
                               ("image_retrieval_paths", ":irp")):
        paths = event.get(field)
        if paths is None:
            continue
        if (msg := _validate_retrieval_paths(paths)):
            return _error(f"{field}: {msg}", "INVALID_PARAMETER")
        sets.append(f"{field} = {placeholder}")
        values[placeholder] = _paths_to_ddb(paths)
        weight_edit = True
    scopes = event.get("active_scopes")
    if scopes is not None:
        # A bare string ("news") would iterate per-character and silently
        # corrupt the registry — an easy LLM-agent mistake.
        if (not isinstance(scopes, list)
                or not all(isinstance(s, str) and s for s in scopes)):
            return _error("active_scopes must be a list of doc_type strings",
                          "INVALID_PARAMETER")
        sets.append("active_scopes = :sc")
        values[":sc"] = {"L": [{"S": s} for s in scopes]}
    smap = event.get("schema_map")
    if smap is not None:
        if (not isinstance(smap, dict)
                or not all(isinstance(k, str) and isinstance(v, str) and v
                           for k, v in smap.items())):
            return _error("schema_map must map doc_type to a schema file key",
                          "INVALID_PARAMETER")
        sets.append("schema_map = :sm")
        values[":sm"] = {"M": {k: {"S": v} for k, v in smap.items()}}
    # Full-replacement guard (measured live 2026-10-13: a payload naming
    # only 'dialogue' silently wiped visual/entity_fts/text_overlay on two
    # collections).  EVERY field this tool writes is a full replacement, so
    # a payload that OMITS members the record already carries is almost
    # always a partial edit, not an intended narrowing — refuse and NAME
    # them, unless replace_all says the narrowing is deliberate (string
    # coercion like `force`: an agent's "false" must not read as true).
    # One consistent read; on any read error fall through with a log — the
    # guard is advisory, the write itself is unchanged.
    replace_all = str(event.get("replace_all", "")).lower() in ("true", "1",
                                                                "yes")
    guarded = [f for f in ("retrieval_paths", "image_retrieval_paths",
                           "active_scopes", "schema_map")
               if event.get(f) is not None]
    if guarded and not replace_all:
        try:
            cur = _ddb.get_item(TableName=COLLECTION_TABLE,
                                Key={"collection_id": {"S": collection_id}},
                                ConsistentRead=True).get("Item") or {}
        except (ClientError, BotoCoreError) as exc:
            logger.warning("full-replacement guard skipped (%s) — "
                           "writing without the omission check", exc)
            cur = {}
        def _members(field, paths):
            # Weights diff at SUBQUERY depth ("visual.shot_ann"), not just
            # path groups — forgetting one inner weight is the same silent
            # wipe one nesting level down.
            if field in ("retrieval_paths", "image_retrieval_paths"):
                return {f"{g}.{k}" for g, ks in (paths or {}).items()
                        for k in ks}
            if field == "active_scopes":
                return set(paths or [])
            return set((paths or {}).keys())          # schema_map

        def _stored_members(field):
            # A malformed stored attribute must REFUSE (we cannot verify
            # omissions against it), never silently compare against an
            # empty set — for EVERY guarded field, not just the weights.
            if field in ("retrieval_paths", "image_retrieval_paths"):
                return _members(field, _unmarshal_paths(cur, field))
            raw = cur.get(field)
            if raw is None:
                return set()
            expected = "L" if field == "active_scopes" else "M"
            if not isinstance(raw, dict) or expected not in raw:
                raise ValueError(f"{field}: expected a DDB {expected} value")
            if field == "active_scopes":
                return {s.get("S", "") for s in raw["L"]}
            return set(raw["M"])
        for field in guarded:
            try:
                stored = _stored_members(field)
            except ValueError as exc:
                return _error(
                    f"stored {exc} — the record needs repair; pass "
                    f"replace_all: true to overwrite it deliberately",
                    "MALFORMED_STORED_RECORD")
            omitted = sorted(stored - _members(field, event[field]))
            if omitted:
                return _error(
                    f"{field} omits existing member(s) {omitted} — this "
                    f"write is a FULL replacement and would wipe them; send "
                    f"the complete set (read the current one from "
                    f"list_collections) or pass replace_all: true to narrow "
                    f"deliberately", "PARTIAL_REPLACEMENT_PAYLOAD")

    if weight_edit:
        sets.append("weights_locked = :wl")
        values[":wl"] = {"BOOL": True}
    if len(sets) == 1:
        return _error("nothing to update — pass retrieval_paths, "
                      "image_retrieval_paths, active_scopes, or schema_map",
                      "INVALID_PARAMETER")
    try:
        resp = _ddb.update_item(
            TableName=COLLECTION_TABLE,
            Key={"collection_id": {"S": collection_id}},
            UpdateExpression="SET " + ", ".join(sets),
            ConditionExpression="attribute_exists(collection_id)",
            ExpressionAttributeValues=values,
            ReturnValues="ALL_NEW",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return _error(f"collection {collection_id!r} does not exist — "
                          f"create it first", "NOT_FOUND")
        raise
    # Report the collection's ACTUAL lock state from the updated item — a
    # scopes-only edit on an already-locked collection must still say True.
    locked = resp.get("Attributes", {}).get("weights_locked", {}).get("BOOL", False)
    return {"collection_id": collection_id, "updated": True,
            "weights_locked": locked}


# ── delete_collection ──────────────────────────────────────────────────────────

def _delete_prefix(bucket: str, prefix: str) -> int:
    deleted = 0
    kwargs: dict = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = _s3.list_objects_v2(**kwargs)
        keys = [{"Key": o["Key"]} for o in resp.get("Contents", [])]
        if keys:
            _s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            deleted += len(keys)
        if not resp.get("IsTruncated"):
            return deleted
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]


def _delete_collection(event: dict) -> dict:
    collection_id = (event.get("collection_id") or "").strip()
    if not collection_id:
        return _error("collection_id is required", "MISSING_PARAMETER")
    if not _valid_collection_id(collection_id):
        # Same rule as create_collection — a malformed id must never reach
        # the begins_with sweeps or the S3 prefix deletes.
        return _error(f"collection_id {collection_id!r} invalid",
                      "INVALID_PARAMETER")
    if not _ddb.get_item(TableName=COLLECTION_TABLE,
                         Key={"collection_id": {"S": collection_id}}).get("Item"):
        # A typo'd id would otherwise "succeed" with 0 deletions — false
        # confidence that a cleanup happened.
        return _error(f"collection {collection_id!r} does not exist",
                      "NOT_FOUND")
    force = str(event.get("force", "")).lower() in ("true", "1", "yes")
    if collection_id == "default" and not force:
        return _error("'default' is the seeded fallback collection — pass "
                      "force=true to delete it", "DEFAULT_COLLECTION")

    prefix = f"{collection_id}:"

    # 1. Enumerate this collection's state records (its physical tables).
    index_keys: list[str] = []
    kwargs: dict = {"TableName": INDEX_STATE_TABLE,
                    "KeyConditionExpression": "pk = :p AND begins_with(sk, :c)",
                    "ExpressionAttributeValues": {":p": {"S": "state"},
                                                  ":c": {"S": prefix}}}
    while True:
        resp = _ddb.query(**kwargs)
        index_keys += [it["sk"]["S"] for it in resp.get("Items", [])]
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    # 2. Dedup records (enumerate per table via the GSI-equivalent scan on
    #    table_name; dedup doc_ids also drive S3 Vectors deletes).
    for ikey in index_keys:
        # asset_id (the dedup GSI hash key) is unknown at collection scope, so
        # enumerate via a filtered scan — dedup items are tiny (KEYS-level
        # projection) and collection deletion is a rare admin operation.
        doc_ids: list[str] = []
        scan_kwargs: dict = {"TableName": DEDUP_TABLE,
                             "FilterExpression": "table_name = :t",
                             "ExpressionAttributeValues": {":t": {"S": ikey}},
                             "ProjectionExpression": "doc_id"}
        while True:
            resp = _ddb.scan(**scan_kwargs)
            doc_ids += [it["doc_id"]["S"] for it in resp.get("Items", [])]
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        for i in range(0, len(doc_ids), 25):
            _ddb.batch_write_item(RequestItems={DEDUP_TABLE: [
                {"DeleteRequest": {"Key": {"doc_id": {"S": d}}}}
                for d in doc_ids[i:i + 25]]})

    # 3. S3 Vectors indexes for this collection (whole-index deletes — no
    #    per-vector sweep needed once the index itself goes). Names derived
    #    EXACTLY from the enumerated state records — never prefix-matched
    #    against list_indexes: the stored name is sanitized/truncated
    #    ("_"→"-", 53-char prefix cap), so raw-prefix matching both misses
    #    this collection's indexes and can over-match a sibling collection
    #    (e.g. sweeping "news" would catch "news--x"). Every S3V index has a
    #    state record (_ensure_registered precedes ensure_index).
    if VECTOR_BUCKET:
        s3v = boto3.client("s3vectors")
        for ikey in index_keys:
            table_key = ikey[len(prefix):]
            name = s3v_index_name(collection_id, table_key)
            try:
                s3v.delete_index(vectorBucketName=VECTOR_BUCKET, indexName=name)
                logger.info("Deleted S3 Vectors index %s", name)
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "NotFoundException":
                    logger.warning("S3 Vectors delete %s failed: %s", name, exc)

    # 4. S3 data prefix (LanceDB tables + staged row groups).
    s3_deleted  = _delete_prefix(PROXY_BUCKET, f"_lancedb_index/{collection_id}/")
    s3_deleted += _delete_prefix(PROXY_BUCKET, f"_indexing/{collection_id}/")

    # 5. Pending FIRST (a resumed waiter would write into the half-deleted
    #    collection), then state/asset. Lock items are deliberately NOT
    #    deleted: deleting one fires a stream REMOVE that dequeues a waiter
    #    mid-sweep; in-flight holders' locks expire via their own TTL and a
    #    deleted collection's orphan lock is harmless.
    ddb_deleted = 0
    for pk in ("pending", "state", "asset"):
        kwargs = {"TableName": INDEX_STATE_TABLE,
                  "KeyConditionExpression": "pk = :p AND begins_with(sk, :c)",
                  "ExpressionAttributeValues": {":p": {"S": pk},
                                                ":c": {"S": prefix}}}
        while True:
            resp = _ddb.query(**kwargs)
            for it in resp.get("Items", []):
                _ddb.delete_item(TableName=INDEX_STATE_TABLE,
                                 Key={"pk": {"S": pk}, "sk": it["sk"]})
                ddb_deleted += 1
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    # 6. Registry record last — while it exists, retries can re-enumerate.
    _ddb.delete_item(TableName=COLLECTION_TABLE,
                     Key={"collection_id": {"S": collection_id}})

    return {"collection_id": collection_id, "deleted": True,
            "physical_tables": len(index_keys),
            "s3_objects_deleted": s3_deleted,
            "ddb_records_deleted": ddb_deleted}
