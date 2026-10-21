"""
StateRerankResults — merge per-index results, attach asset metadata, return top-k.

Receives the full search_results array from the Map (one item per index).
Filters skipped items. Merges across indices using index_weight × rrf_score.
Batch-reads asset metadata from DDB for all unique asset_ids in top-k results
to attach title. Slices to k. Updates search history record with results.
"""
from __future__ import annotations

import logging
import time
import traceback

from lib.index_state import batch_get_asset_metadata, update_search_history_results
from lib.s3_utils import delete_object
from lib.input_schema import PROXY_BUCKET

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _cross_index_rrf(index_results: list[dict]) -> list[dict]:
    """
    Flatten and rank results from multiple indices using index_weight × rrf_score.
    Doc IDs are structurally unique per index type so no dedup is needed.
    Normalises final_score to [0, 1] with the top result at 1.0.
    """
    all_docs = []
    for item in index_results:
        index_weight = float(item.get("index_weight", 1.0))
        index_scope  = item.get("index_scope", "")
        for doc in item.get("results") or []:
            final_score = index_weight * doc.get("rrf_score", 0.0)
            all_docs.append({**doc, "source_index": index_scope, "final_score": final_score})

    if not all_docs:
        return []

    max_score = max(d["final_score"] for d in all_docs) or 1.0
    for d in all_docs:
        d["final_score"] = round(d["final_score"] / max_score, 6)

    return sorted(all_docs, key=lambda x: -x["final_score"])


def _attach_asset_metadata(collection_id: str, results: list[dict]) -> list[dict]:
    """
    Batch-read collection-scoped asset metadata for unique asset_ids in
    results, attach title/genres to each result doc.
    """
    unique_ids = list({r.get("asset_id", "") for r in results if r.get("asset_id")})
    if not unique_ids or not collection_id:
        return results

    meta_map = batch_get_asset_metadata(collection_id, unique_ids)

    enriched = []
    for r in results:
        aid  = r.get("asset_id", "")
        meta = meta_map.get(aid, {})
        doc  = {**r}
        if meta.get("title"):
            doc["title"] = meta["title"]
        if meta.get("genres"):
            doc["genres"] = meta["genres"]
        enriched.append(doc)
    return enriched


def lambda_handler(event: dict, context) -> dict:
    log.info("StateRerankResults: search_term length=%d",
             len(event.get("search_term") or ""))

    search_term     = event.get("search_term", "")
    k               = int(event.get("k", 10))
    decomposition   = event.get("decomposition") or {}
    search_results  = event.get("search_results") or []
    request_id      = decomposition.get("request_id", "")
    collection_id   = decomposition.get("collection_id", "")
    vectors_s3_key  = decomposition.get("vectors_s3_key", "")
    start_time      = int(time.time() * 1000)

    try:
        # ── Enrich each index result with index_weight from decomposition ──────
        queries_by_scope = {
            q["index_scope"]: q
            for q in decomposition.get("queries", [])
            if "index_scope" in q
        }
        enriched_index_results = []
        # Scopes decompose dropped never reached the Map — surface them here
        # so a caller can tell "no matches" from "never searched". Entries
        # share the Map skip items' vocabulary ({index_scope, skip_reason}),
        # and the two sets are disjoint (a dropped scope has no Map item).
        skipped_indices        = [d["index_scope"] for d in
                                  decomposition.get("dropped_scopes") or []]

        for item in search_results:
            index_scope = item.get("index_scope", "")
            if item.get("skipped"):
                skipped_indices.append(index_scope)
                continue
            query_meta   = queries_by_scope.get(index_scope, {})
            index_weight = float(query_meta.get("index_weight", 1.0))
            enriched_index_results.append({**item, "index_weight": index_weight})

        log.info(
            "Merging %d index results, skipped=%s",
            len(enriched_index_results), skipped_indices,
        )

        if not enriched_index_results:
            log.warning("All indices skipped or empty — returning empty results")
            response = {
                "results":         [],
                "total":           0,
                "k":               k,
                "search_term":     search_term,
                "collection_id":   collection_id,
                "skipped_indices": skipped_indices,
            }
            _write_history(request_id, [], start_time)
            _delete_vectors(vectors_s3_key)
            return response

        # ── Cross-index RRF merge ──────────────────────────────────────────────
        merged = _cross_index_rrf(enriched_index_results)

        # ── Slice first, then attach metadata only for the top-k docs ─────────
        top_k    = _attach_asset_metadata(collection_id, merged[:k])

        log.info("Rerank complete: returning %d / %d total results", len(top_k), len(merged))

        response = {
            "results":         top_k,
            "total":           len(merged),
            "k":               k,
            "search_term":     search_term,
            "collection_id":   collection_id,
            "skipped_indices": skipped_indices,
        }
        _write_history(request_id, top_k, start_time)
        _delete_vectors(vectors_s3_key)
        return response

    except Exception as exc:
        log.error("StateRerankResults failed — %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise


def _delete_vectors(vectors_s3_key: str) -> None:
    if not vectors_s3_key:
        return
    try:
        delete_object(PROXY_BUCKET, vectors_s3_key)
    except Exception as exc:
        log.warning("Failed to delete vectors object %s: %s", vectors_s3_key, exc)


def _write_history(request_id: str, results: list[dict], start_ms: int) -> None:
    if not request_id:
        return
    duration_ms = int(time.time() * 1000) - start_ms
    try:
        update_search_history_results(
            request_id=request_id,
            search_results=results,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        # History write failure must not fail the search response
        log.warning("Failed to update search history request_id=%s: %s", request_id, exc)
