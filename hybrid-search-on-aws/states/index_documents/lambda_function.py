"""
StateIndexDocuments — write one prepared row group to its physical table.

Runs inside the Index SFN Map, one iteration per (collection, table),
under that table's own lock (acquired by the preceding Acquire-lock state
with sk=index_key — README.md Design record §1.4).

Input (Map item, produced by StatePrepareDocuments):
  index_key       — "{collection_id}:{table_key}" (lock/state/dedup identity)
  rows_s3_key     — S3 key of {schema_file_uri, collection_id,
                    refresh_asset_ids, documents:[typed rows]}
  collection_id   — scopes the physical LanceDB location / S3V index name

Flow:
  1. Load row group + schema; build the collection-scoped backend(s).
  2. register_index first-write-wins; schema_file mismatch fails hard.
  3. Per-asset refresh: clear dedup + LanceDB rows for refresh_asset_ids.
  4. Defensive dedup (Prepare already deduped — cheap second check).
  5. Write: LanceDB rows (FTS fields or full row) and/or S3 Vectors puts,
     per the schema's vector_store. FTS rebuild when applicable.
  6. mark_docs_indexed AFTER all stores succeed (dual-store consistency:
     a crash between writes re-attempts both on retry; S3V put is a true
     upsert, LanceDB refresh-delete keeps re-runs idempotent).
"""
from __future__ import annotations

import logging
import traceback

from lib.backends.lancedb_backend import LanceDBBackend, load_schema
from lib.backends.s3vectors_backend import S3VectorsBackend
from lib.collections import s3v_index_name
from lib.index_state import (
    batch_is_indexed,
    delete_docs_for_asset_and_table,
    list_doc_ids_for_asset_and_table,
    get_table_state,
    increment_pending,
    mark_docs_indexed,
    register_index,
    update_last_indexed_at,
)
from lib.input_schema import PROXY_BUCKET
from lib.s3_utils import get_json

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _ensure_registered(index_key: str, schema_file: str) -> None:
    """First-write-wins registration; schema_file mismatch fails hard."""
    state = get_table_state(index_key)
    if not state:
        if not register_index(index_key, schema_file):
            state = get_table_state(index_key)
    stored = (state or {}).get("schema_file", "")
    if stored and stored != schema_file:
        raise ValueError(
            f"Index {index_key!r} was registered with schema_file={stored!r}, "
            f"but this request uses {schema_file!r}. Schema changes require a "
            f"new table_key — never an in-place edit on a table with rows.")
    update_last_indexed_at(index_key)


def lambda_handler(event: dict, context) -> dict:
    index_key     = event.get("index_key", "")
    rows_s3_key   = event.get("rows_s3_key", "")
    collection_id = event.get("collection_id", "")
    if not index_key or not rows_s3_key or not collection_id:
        raise ValueError("index_key, rows_s3_key, collection_id are required")

    log.info("StateIndexDocuments: %s <- %s", index_key, rows_s3_key)
    try:
        payload           = get_json(PROXY_BUCKET, rows_s3_key)
        schema_file_uri   = payload["schema_file_uri"]
        documents         = payload.get("documents") or []
        refresh_asset_ids = payload.get("refresh_asset_ids") or []
        if not documents:
            return {**event, "indexed": 0}

        schema_def = load_schema(schema_file_uri)
        use_s3v    = schema_def.vector_store == "s3vectors"
        # LanceDB holds the full row (lancedb store) or the FTS-bearing row
        # (s3vectors + FTS). ANN-only s3vectors schemas never create a
        # LanceDB table at all.
        use_lance  = (not use_s3v) or bool(schema_def.fts_fields)

        _ensure_registered(index_key, schema_file_uri)

        lance = (LanceDBBackend(schema_def, collection_id, bucket=PROXY_BUCKET)
                 if use_lance else None)
        s3v   = (S3VectorsBackend(schema_def, collection_id,
                                  s3v_index_name(collection_id, schema_def.table_key))
                 if use_s3v else None)

        # ── Per-asset refresh: clear rows + dedup ONLY for assets that asked ──
        if refresh_asset_ids:
            for aid in refresh_asset_ids:
                # Enumerate the asset's OLD doc_ids from dedup BEFORE deleting
                # the dedup records — they are the only inventory of stale S3V
                # vector keys (shifted timespans mint new doc_ids; without
                # this, old vectors would ghost in ANN results forever).
                old_ids = list_doc_ids_for_asset_and_table(aid, index_key)
                if s3v and old_ids:
                    s3v.delete_keys(old_ids)
                delete_docs_for_asset_and_table(aid, index_key)
            if lance:
                lance.load()
                for aid in refresh_asset_ids:
                    lance.delete_by_asset_id(aid)
        refreshing = set(refresh_asset_ids)
        check_ids  = [d["doc_id"] for d in documents
                      if d.get("asset_id") not in refreshing]
        already    = batch_is_indexed(check_ids)
        new_docs   = [d for d in documents if d["doc_id"] not in already]

        if not new_docs:
            return {**event, "indexed": 0}

        # ── Write ────────────────────────────────────────────────────────────
        if lance:
            # Idempotent re-entry: a crash after this append but before
            # mark_docs_indexed re-runs this block on SFN retry — clear any
            # rows from the failed attempt so the append never duplicates.
            lance.delete_by_doc_ids([d["doc_id"] for d in new_docs])
            lance.index(new_docs)
            if schema_def.fts_fields:
                lance.build_fts_index()
        if s3v:
            s3v.ensure_index()
            s3v.put(new_docs)

        mark_docs_indexed(new_docs, index_key=index_key)
        increment_pending(index_key, len(new_docs))

        log.info("Indexed %d docs to %s (lance=%s s3v=%s)",
                 len(new_docs), index_key, bool(lance), bool(s3v))
        return {**event, "indexed": len(new_docs)}

    except Exception as exc:
        log.error("StateIndexDocuments failed — %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise
