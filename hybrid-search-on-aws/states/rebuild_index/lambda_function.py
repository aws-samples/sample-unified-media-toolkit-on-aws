"""
StateRebuildIndex — LanceDB optimize + ANN rebuild for ONE physical table.

Runs inside the Rebuild SFN Map under that table's own lock (acquired by
the preceding per-iteration Acquire-lock state, sk=index_key).

index_key = "{collection_id}:{table_key}" — the collection_id prefix
resolves the physical LanceDB location.
"""
from __future__ import annotations

import logging
import traceback

from lib.backends.lancedb_backend import LanceDBBackend, load_schema
from lib.index_state import (
    get_table_state,
    is_rebuild_stale,
    mark_rebuild_complete,
    mark_rebuild_failed,
    mark_rebuild_started,
)
from lib.input_schema import PROXY_BUCKET

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    index_key = event.get("index_key", "")
    force     = bool(event.get("force", False))
    if not index_key:
        raise ValueError("Missing required field: index_key")

    collection_id, _, _ = index_key.partition(":")
    state = get_table_state(index_key)
    if not state:
        return {"index_key": index_key, "skipped": True, "reason": "missing_state_record"}
    schema_file = state.get("schema_file", "")
    if not schema_file:
        return {"index_key": index_key, "skipped": True, "reason": "missing_schema_file"}
    if bool(state.get("rebuild_in_progress")) and not is_rebuild_stale(
            state.get("rebuild_started_at")):
        return {"index_key": index_key, "skipped": True, "reason": "in_progress"}

    log.info("[%s] starting rebuild (pending=%s, force=%s)",
             index_key, state.get("pending_docs_since_rebuild"), force)
    mark_rebuild_started(index_key)
    try:
        schema_def = load_schema(schema_file)
        backend    = LanceDBBackend(schema_def, collection_id, bucket=PROXY_BUCKET)
        backend.load()
        if not backend.is_loaded:
            mark_rebuild_complete(index_key)
            return {"index_key": index_key, "skipped": True, "reason": "empty_table"}
        backend.optimize()
        if schema_def.vector_store == "lancedb":
            backend.build_ann_index()
        mark_rebuild_complete(index_key)
    except Exception as exc:
        try:
            mark_rebuild_failed(index_key)
        except Exception as cleanup_exc:
            log.warning("[%s] mark_rebuild_failed threw: %s", index_key, cleanup_exc)
        log.error("[%s] rebuild failed — %s: %s", index_key, type(exc).__name__, exc)
        traceback.print_exc()
        raise

    log.info("[%s] rebuild complete", index_key)
    return {"index_key": index_key, "skipped": False, "rows": backend.count_rows()}
