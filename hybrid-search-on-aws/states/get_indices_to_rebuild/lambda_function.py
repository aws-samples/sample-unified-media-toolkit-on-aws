"""
StateGetIndicesToRebuild — list index_keys needing an ANN rebuild.

Returns a bare list of index_key strings ("{collection_id}:{table_key}")
for the Rebuild SFN Map. Each Map iteration acquires ITS OWN table's lock
(D9: rebuilds across collections/tables run concurrently; only
same-physical-table operations serialize).

Candidates: force=True, or pending_docs_since_rebuild >= REBUILD_THRESHOLD.
In-progress-and-not-stale rebuilds are excluded. S3 Vectors ANN-only
indexes are excluded entirely — S3V has no client-side rebuild step
(rebuild covers LanceDB optimize + IVF_PQ only; FTS rebuilds happen at
index time).
"""
from __future__ import annotations

import json
import logging

from lib.backends.lancedb_backend import load_schema
from lib.index_state import is_rebuild_stale, list_active_schemas
from lib.input_schema import REBUILD_THRESHOLD

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> list:
    log.info("StateGetIndicesToRebuild: force=%s", event.get("force"))
    force  = bool(event.get("force", False))
    states = list_active_schemas()
    candidates = []
    for state in states:
        index_key = state.get("sk", "")
        if not index_key:
            continue
        if bool(state.get("rebuild_in_progress")) and not is_rebuild_stale(
                state.get("rebuild_started_at")):
            log.info("[%s] rebuild in progress — skipping", index_key)
            continue
        schema_file = state.get("schema_file", "")
        if schema_file:
            try:
                schema_def = load_schema(schema_file)
                if schema_def.vector_store == "s3vectors" and not schema_def.fts_fields:
                    log.info("[%s] ANN-only s3vectors index — no rebuild step", index_key)
                    continue
            except Exception as exc:
                log.warning("[%s] schema load failed (%s) — including anyway", index_key, exc)
        pending = int(state.get("pending_docs_since_rebuild", 0))
        if force or pending >= REBUILD_THRESHOLD:
            candidates.append(index_key)
    log.info("Returning %d rebuild candidates", len(candidates))
    return candidates
