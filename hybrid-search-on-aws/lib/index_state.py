"""
index_state.py — DynamoDB index state helpers.

Every physical unit is identified by index_key = "{collection_id}:{table_key}"
(README.md Design record §1.1). All functions here take index_key where the old
design took a bare table_key — the table SCHEMAS are unchanged (constraint:
no DDB resource-definition change); only key VALUES gained the collection
prefix.

LanceDBIndexState  (state, locks, pending-queues, asset metadata)
  pk="lock", sk="{index_key}"
    One distributed lock per physical table (lib/lock_utils).

  pk="pending", sk="{index_key}#{sha256(task_token)}"
    Per-lock waiter queue (lib/lock_utils).

  pk="state", sk="{index_key}"
    Per-index record. Conditional-put first-write-wins. Contains:
    schema_file pointer, s3v_index_name (S3 Vectors schemas), rebuild
    state, pending doc counter.
    index_key format (unchanged):
      FTS-capable:  "{doc_type}:{embedding_model}:{iso_language}"
      Non-FTS:      "{doc_type}:{embedding_model}"

  pk="asset", sk="{collection_id}:{asset_id}"
    Per-asset metadata (title, genres, casts, content_category), scoped
    per collection. GSI title-index (pk="asset" + title range) still
    serves @ operator resolution; results are post-filtered by the
    collection prefix on sk (titles are few — post-filter beats a GSI
    change, which the DDB constraint forbids).

LanceDBDedupState  (per-document dedup — isolated to avoid hot partition)
  doc_id (hash key, no sort key)
    { doc_id, asset_id, table_name=index_key, indexed_at, ttl }
    GSI asset_id-table_name-index — per-table refresh enumeration.
    doc_id itself embeds collection_id (md5 over
    "{collection_id}:{asset_id}:{doc_type}:{start_ms}:{end_ms}:{chunk_idx}")
    so cross-collection duplicates cannot collide.

LanceDBSearchHistoryTable  (search request/response history)
  request_id (hash key) + sk="search" — unchanged.
"""
from __future__ import annotations

import calendar
import logging
import time
from decimal import Decimal

from lib.ddb_utils import (
    ISO_FORMAT,
    now_iso,
    get_item,
    batch_get_items,
    delete_item,
    conditional_put_item,
    query_pk,
    ttl_from_now,
    get_table as _get_table,
    get_dedup_table as _get_dedup_table,
    get_search_history_table as _get_search_history_table,
)
from boto3.dynamodb.conditions import Key
from lib.input_schema import SEARCH_HISTORY_TTL_DAYS

_STALE_REBUILD_SECS = 1800


def is_rebuild_stale(started_at: str | None) -> bool:
    """Return True if rebuild_started_at is absent or older than _STALE_REBUILD_SECS."""
    if not started_at:
        return True
    try:
        ts  = time.strptime(started_at, ISO_FORMAT)
        age = time.time() - calendar.timegm(ts)
        return age > _STALE_REBUILD_SECS
    except Exception:
        return True

log = logging.getLogger(__name__)

_STATE_PK = "state"
_ASSET_PK = "asset"


# ── Per-index state ────────────────────────────────────────────────────────────

def register_index(index_key: str, schema_file: str) -> bool:
    """
    Write the state record for a new index on first table creation.
    Uses conditional put (attribute_not_exists) — atomic first-write-wins.
    Returns True if the record was written (new index), False if it already existed.
    """
    now = now_iso()
    item = {
        "pk":                         _STATE_PK,
        "sk":                         index_key,
        "schema_file":                schema_file,
        "pending_docs_since_rebuild": Decimal(0),
        "rebuild_in_progress":        False,
        "last_indexed_at":            now,
        "created_at":                 now,
    }
    written = conditional_put_item(item, "attribute_not_exists(pk)")
    if written:
        log.info("Registered new index index_key=%s schema_file=%s", index_key, schema_file)
    return written


def get_table_state(index_key: str) -> dict:
    """Return the state item for an index_key. Returns {} if missing."""
    return get_item(_STATE_PK, index_key) or {}


def update_last_indexed_at(index_key: str) -> None:
    """Update last_indexed_at on every successful index() call.

    ConditionExpression guards against creating a corrupt item when called
    under DDB eventual consistency right after a concurrent register_index()
    win — if the item isn't visible yet, this silently no-ops rather than
    creating a record with only last_indexed_at and missing all other fields.
    """
    from botocore.exceptions import ClientError
    now = now_iso()
    try:
        _get_table().update_item(
            Key={"pk": _STATE_PK, "sk": index_key},
            UpdateExpression="SET last_indexed_at = :now",
            ConditionExpression="attribute_exists(pk)",
            ExpressionAttributeValues={":now": now},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            log.warning(
                "update_last_indexed_at: state record for %s not visible yet — skipping timestamp update",
                index_key,
            )
        else:
            raise


def list_active_schemas() -> list[dict]:
    """
    Return all state records (all registered indices).
    Each record contains a schema_file pointer used by rebuild_index.
    """
    return query_pk(_STATE_PK)


def increment_pending(index_key: str, n: int) -> None:
    """Atomically increment pending_docs_since_rebuild by n."""
    _get_table().update_item(
        Key={"pk": _STATE_PK, "sk": index_key},
        UpdateExpression="ADD pending_docs_since_rebuild :n",
        ExpressionAttributeValues={":n": Decimal(n)},
    )


def mark_rebuild_started(index_key: str) -> None:
    now = now_iso()
    _get_table().update_item(
        Key={"pk": _STATE_PK, "sk": index_key},
        UpdateExpression="SET rebuild_in_progress = :t, rebuild_started_at = :now",
        ExpressionAttributeValues={":t": True, ":now": now},
    )


def mark_rebuild_complete(index_key: str) -> None:
    now = now_iso()
    _get_table().update_item(
        Key={"pk": _STATE_PK, "sk": index_key},
        UpdateExpression=(
            "SET rebuild_in_progress = :f, "
            "last_ann_rebuild_at = :now, "
            "pending_docs_since_rebuild = :zero"
        ),
        ExpressionAttributeValues={":f": False, ":now": now, ":zero": Decimal(0)},
    )


def mark_rebuild_failed(index_key: str) -> None:
    """
    Clear rebuild_in_progress without resetting pending_docs_since_rebuild.
    The pending counter stays intact so the next scheduled run retries.
    """
    now = now_iso()
    _get_table().update_item(
        Key={"pk": _STATE_PK, "sk": index_key},
        UpdateExpression="SET rebuild_in_progress = :f, last_rebuild_failed_at = :now",
        ExpressionAttributeValues={":f": False, ":now": now},
    )


# ── Per-document dedup (LanceDBDedupState table) ───────────────────────────────

def batch_is_indexed(doc_ids: list[str]) -> set[str]:
    """
    Return the subset of doc_ids that are already indexed.
    Uses BatchGetItem in chunks of 100 against the dedup table.
    doc_id is the hash key — no sort key.
    """
    if not doc_ids:
        return set()

    table   = _get_dedup_table()
    already = set()

    for i in range(0, len(doc_ids), 100):
        chunk = doc_ids[i:i + 100]
        request_items = {
            table.name: {
                "Keys": [{"doc_id": d} for d in chunk],
                "ProjectionExpression": "doc_id",
            }
        }
        while request_items:
            resp = table.meta.client.batch_get_item(RequestItems=request_items)
            for item in resp.get("Responses", {}).get(table.name, []):
                already.add(item["doc_id"])
            request_items = resp.get("UnprocessedKeys") or {}

    return already


def mark_docs_indexed(docs: list[dict], index_key: str) -> None:
    """Write one dedup item per document to the dedicated dedup table."""
    if not docs:
        return
    now   = now_iso()
    table = _get_dedup_table()
    items = [
        {
            "doc_id":     d["doc_id"],
            "asset_id":   d.get("asset_id", ""),
            "table_name": index_key,
            "indexed_at": now,
            "ttl":        ttl_from_now(86400 * 365),
        }
        for d in docs
    ]
    # boto3's BatchWriter buffers, flushes at 25, and retries UnprocessedItems.
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)
    log.info("Marked %d docs as indexed in index_key=%s", len(items), index_key)


# ── Per-asset metadata (collection-scoped: sk = "{collection_id}:{asset_id}") ──

def _asset_sk(collection_id: str, asset_id: str) -> str:
    return f"{collection_id}:{asset_id}"


def upsert_asset_metadata(
    collection_id: str,
    asset_id: str,
    genres: list[str],
    content_category: str,
    title: str = "",
    casts: list[str] | None = None,
) -> None:
    """
    Write or overwrite the asset-level metadata record for one collection.
    genres/casts stored as comma-separated strings; title stored normalised
    (lowercase, stripped) for the title-index GSI range key.
    """
    now            = now_iso()
    title_norm     = title.lower().strip() if title else ""
    expr           = "SET genres = :g, content_category = :c, updated_at = :now, title = :t"
    attr_values: dict = {
        ":g":   ",".join(genres) if genres else "",
        ":c":   content_category or "",
        ":now": now,
        ":t":   title_norm,
    }
    if casts:
        expr += ", casts = :casts"
        attr_values[":casts"] = ",".join(casts)
    else:
        expr += " REMOVE casts"
    _get_table().update_item(
        Key={"pk": _ASSET_PK, "sk": _asset_sk(collection_id, asset_id)},
        UpdateExpression=expr,
        ExpressionAttributeValues=attr_values,
    )
    log.info("Upserted asset metadata %s:%s title=%r genres=%s",
             collection_id, asset_id, title_norm, genres)


def query_assets_by_title(collection_id: str, title_prefix: str,
                          limit: int = 20) -> list[dict]:
    """
    @ operator resolution: title-index GSI prefix match, post-filtered to
    one collection by the sk prefix. Post-filter (not a key condition)
    because the GSI keys on (pk="asset", title) — unchanged per the DDB
    resource constraint; titles are few, so the filter is cheap.
    Returned items carry `asset_id` (sk with the collection prefix stripped).
    """
    if not title_prefix:
        return []
    norm   = title_prefix.lower().strip()
    prefix = f"{collection_id}:"
    # The title GSI spans ALL collections; paginate until `limit` matches
    # survive the collection post-filter (a fixed pre-filter Limit could be
    # consumed entirely by other collections' rows and silently drop the
    # @Title filter).
    kwargs: dict = {
        "IndexName": "title-index",
        "KeyConditionExpression": (
            Key("pk").eq(_ASSET_PK) & Key("title").begins_with(norm)
        ),
    }
    items: list[dict] = []
    while len(items) < limit:
        resp = _get_table().query(**kwargs)
        for it in resp.get("Items", []):
            sk = it.get("sk", "")
            if sk.startswith(prefix):
                items.append({**it, "asset_id": sk[len(prefix):]})
                if len(items) >= limit:
                    break
        last = resp.get("LastEvaluatedKey")
        if not last:
            break
        kwargs["ExclusiveStartKey"] = last
    log.info("query_assets_by_title %s prefix=%r → %d results",
             collection_id, norm, len(items))
    return items


# ── Search history ─────────────────────────────────────────────────────────────

_SEARCH_SK = "search"


def _to_ddb(obj):
    """Recursively convert float → Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_ddb(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_ddb(v) for v in obj]
    return obj


def write_search_history_initial(
    request_id: str,
    search_term: str,
    workflow_config_uri: str,
    decomposition: dict,
    k: int,
    created_at: int,
) -> None:
    """
    Write the initial search history record at the start of Decompose.
    decomposition is stored without ann_vector fields (vectors are large and
    not useful for history analysis).
    """
    ttl = created_at + SEARCH_HISTORY_TTL_DAYS * 86400
    clean = {
        field: val for field, val in decomposition.items()
        if field != "queries"
    }
    if "queries" in decomposition:
        clean["queries"] = [
            {fk: fv for fk, fv in q.items() if fk != "ann_vector"}
            for q in decomposition["queries"]
        ]
    _get_search_history_table().put_item(Item=_to_ddb({
        "request_id":          request_id,
        "sk":                  _SEARCH_SK,
        "search_term":         search_term,
        "workflow_config_uri": workflow_config_uri,
        "decomposition":       clean,
        "k":                   k,
        "created_at":          created_at,
        "ttl":                 ttl,
    }))
    log.info("Wrote initial search history request_id=%s", request_id)


def update_search_history_results(
    request_id: str,
    search_results: list[dict],
    duration_ms: int,
) -> None:
    """
    Append search_results and duration_ms to an existing search history record.
    Called by Reranker after merging results.
    """
    _get_search_history_table().update_item(
        Key={"request_id": request_id, "sk": _SEARCH_SK},
        UpdateExpression="SET search_results = :r, duration_ms = :d",
        ExpressionAttributeValues={
            ":r": _to_ddb(search_results),
            ":d": duration_ms,
        },
    )
    log.info(
        "Updated search history request_id=%s results=%d duration_ms=%d",
        request_id, len(search_results), duration_ms,
    )


def batch_get_asset_metadata(collection_id: str,
                             asset_ids: list[str]) -> dict[str, dict]:
    """
    Batch-read asset metadata for multiple asset_ids within one collection.
    Returns {asset_id: metadata_item} keyed by the BARE asset_id (collection
    prefix stripped). Missing items are absent from the dict.
    """
    prefix = f"{collection_id}:"
    sks    = [_asset_sk(collection_id, a) for a in asset_ids]
    items  = batch_get_items(_ASSET_PK, sks)
    return {item["sk"][len(prefix):]: item
            for item in items if item.get("sk", "").startswith(prefix)}


def list_doc_ids_for_asset_and_table(asset_id: str, index_key: str) -> list[str]:
    """
    Enumerate one asset's doc_ids in a single index table via the
    asset_id-table_name-index GSI. Used to sweep stale S3 Vectors keys on
    refresh BEFORE the dedup records (the only inventory) are deleted.
    """
    table  = _get_dedup_table()
    kwargs = {
        "IndexName": "asset_id-table_name-index",
        "KeyConditionExpression": (
            Key("asset_id").eq(asset_id) & Key("table_name").eq(index_key)
        ),
        "ProjectionExpression": "doc_id",
    }
    doc_ids: list[str] = []
    while True:
        resp = table.query(**kwargs)
        doc_ids += [it["doc_id"] for it in resp.get("Items", [])]
        last = resp.get("LastEvaluatedKey")
        if not last:
            return doc_ids
        kwargs["ExclusiveStartKey"] = last


def delete_docs_for_asset_and_table(asset_id: str, index_key: str) -> None:
    """
    Delete dedup records for one asset scoped to a single index table.
    Safe to call concurrently from parallel Map iterations — each call only
    touches its own table's dedup records.
    """
    table   = _get_dedup_table()
    doc_ids = list_doc_ids_for_asset_and_table(asset_id, index_key)
    with table.batch_writer() as batch:
        for doc_id in doc_ids:
            batch.delete_item(Key={"doc_id": doc_id})
    log.info(
        "Deleted %d dedup records for asset_id=%s index_key=%s",
        len(doc_ids), asset_id, index_key,
    )
