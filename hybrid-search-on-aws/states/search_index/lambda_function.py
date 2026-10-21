"""
StateSearchIndex — run sub-queries against one LanceDB index, return ranked results.

Receives one query item from the Map fan-out. Each item is self-contained:
table_key, schema_key, asset_ids filter, sub_queries spec, candidate_k, ann_vector.

Graceful miss: if the index is not yet registered or has no indexed docs,
returns {index_scope, results: [], skipped: True} — no error raised.

Sub-queries run in parallel via ThreadPoolExecutor.
Results merged within-index via weighted RRF, returning up to candidate_k items.
"""
from __future__ import annotations

import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from lib.backends.lancedb_backend import LanceDBBackend, load_schema
from lib.backends.s3vectors_backend import S3VectorsBackend
from lib.collections import s3v_index_name
from lib.index_state import get_table_state
from lib.input_schema import PROXY_BUCKET
from lib.s3_utils import get_json

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

RRF_K = 60

# Warm backend cache keyed by (index_key, schema_key) — persists across
# invocations; the schema_key component invalidates stale entries when a
# collection is deleted and re-created against a different schema file.
_backend_cache: dict[tuple, tuple] = {}


def _get_backends(index_key: str, collection_id: str, schema_key: str):
    """
    Returns (lance_or_None, s3v_or_None) per the schema's vector_store.
    ANN goes to S3 Vectors when the schema opts in; FTS is always LanceDB.
    """
    cache_key = (index_key, schema_key)
    if cache_key in _backend_cache:
        return _backend_cache[cache_key]
    schema_def = load_schema(f"s3://{PROXY_BUCKET}/{schema_key}")
    use_s3v    = schema_def.vector_store == "s3vectors"
    use_lance  = (not use_s3v) or bool(schema_def.fts_fields)
    lance = None
    if use_lance:
        lance = LanceDBBackend(schema_def, collection_id, bucket=PROXY_BUCKET)
        lance.load()
    s3v = None
    if use_s3v:
        table_key = index_key.partition(":")[2]
        s3v = S3VectorsBackend(schema_def, collection_id,
                               s3v_index_name(collection_id, table_key))
    _backend_cache[cache_key] = (lance, s3v)
    return lance, s3v


def _highlight(text: str, query: str) -> str | None:
    """Case-insensitive regex highlight of query tokens in text, wrapped in <em>.</em>"""
    if not text or not query:
        return None
    tokens = [re.escape(t) for t in query.split() if t]
    if not tokens:
        return None
    return re.sub(r'(' + '|'.join(tokens) + r')', r'<em>\1</em>', text, flags=re.IGNORECASE)


def _weighted_rrf(ranked_lists: list[tuple[str, float, list[dict]]]) -> list[dict]:
    scores:        dict[str, float] = {}
    match_sources: dict[str, dict]  = {}
    docs:          dict[str, dict]  = {}

    for subq_key, weight, results in ranked_lists:
        if not results or weight == 0.0:
            continue
        max_score = max((d.get("score", 0.0) for d in results), default=1.0) or 1.0
        for rank, doc in enumerate(results):
            doc_id     = doc["doc_id"]
            norm_score = doc.get("score", 0.0) / max_score
            contrib    = weight * norm_score / (RRF_K + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + contrib
            match_sources.setdefault(doc_id, {})[subq_key] = round(contrib, 8)
            if doc_id not in docs:
                docs[doc_id] = doc
            elif doc.get("highlights"):
                # FTS result for a doc already stored from ANN — merge highlights in
                existing = docs[doc_id].get("highlights") or {}
                docs[doc_id] = {**docs[doc_id], "highlights": {**existing, **doc["highlights"]}}

    return [
        {**docs[doc_id], "rrf_score": round(score, 8), "match_sources": match_sources[doc_id]}
        for doc_id, score in sorted(scores.items(), key=lambda x: -x[1])
    ]


def lambda_handler(event: dict, context) -> dict:
    query = event.get("query") or {}
    log.info("StateSearchIndex: index_scope=%s index_key=%s",
             query.get("index_scope"), query.get("index_key"))

    index_scope     = query.get("index_scope", "")
    index_key       = query.get("index_key", "")
    collection_id   = query.get("collection_id", "")
    schema_key      = query.get("schema_key", "")
    asset_ids       = query.get("asset_ids") or []
    candidate_k     = int(query.get("candidate_k", 50))
    vectors_s3_key  = query.get("vectors_s3_key", "")
    sub_queries     = query.get("sub_queries") or []

    # Load vector for this scope from S3 — avoids embedding inline in SFN state
    ann_vector = None
    if vectors_s3_key:
        vectors_map = get_json(PROXY_BUCKET, vectors_s3_key)
        ann_vector  = vectors_map.get(index_scope)

    if not index_key or not schema_key or not collection_id:
        log.warning("Missing index_key/schema_key/collection_id — skipping index_scope=%s", index_scope)
        return {"index_scope": index_scope, "results": [], "skipped": True,
                "skip_reason": "missing index_key, schema_key, or collection_id"}

    try:
        # ── Graceful miss: index not registered or no docs indexed ─────────────
        state = get_table_state(index_key)
        if not state or not state.get("last_indexed_at"):
            log.info("Index %s not registered or empty — skipping", index_key)
            return {"index_scope": index_scope, "results": [], "skipped": True,
                    "skip_reason": "index not registered"}

        lance, s3v = _get_backends(index_key, collection_id, schema_key)

        # ── Build tasks ────────────────────────────────────────────────────────
        ids_filter: list[str] | None = asset_ids if asset_ids else None

        tasks: list[tuple[str, float, Any]] = []
        for sq in sub_queries:
            sq_type  = sq.get("type")
            sq_field = sq.get("field", "")
            sq_w     = float(sq.get("weight", 0.0))
            sq_query = sq.get("query", "")
            subq_key = f"{sq_type}:{sq_field}"

            if sq_w == 0.0:
                continue

            if sq_type == "ann":
                if not ann_vector:
                    continue
                if s3v is not None:
                    tasks.append((
                        subq_key,
                        sq_w,
                        lambda v=ann_vector, ids=ids_filter:
                            s3v.query(v, candidate_k, asset_ids=ids),
                    ))
                else:
                    tasks.append((
                        subq_key,
                        sq_w,
                        lambda v=ann_vector, ids=ids_filter:
                            lance.search_ann(v, candidate_k, asset_ids=ids),
                    ))
            elif sq_type == "fts":
                if not sq_query or lance is None:
                    continue
                tasks.append((
                    subq_key,
                    sq_w,
                    lambda q=sq_query, f=sq_field, ids=ids_filter: [
                        {**doc, "highlights": {f: h}}
                        if (h := _highlight(doc.get(f, ""), q)) else doc
                        for doc in lance.search_fts(q, f, candidate_k, asset_ids=ids)
                    ],
                ))

        if not tasks:
            log.warning("No active sub-queries for %s", index_key)
            return {"index_scope": index_scope, "results": [], "skipped": False}

        # ── Run sub-queries in parallel ────────────────────────────────────────
        ranked_lists:   list[tuple[str, float, list[dict]]] = []
        failed_queries: list[str]                      = []

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): (name, w) for name, w, fn in tasks}
            for future in as_completed(futures):
                name, w = futures[future]
                try:
                    results = future.result()
                    ranked_lists.append((name, w, results))
                    log.info("Sub-query %s: %d results", name, len(results))
                except Exception as exc:
                    log.error("Sub-query %s failed: %s", name, exc)
                    failed_queries.append(name)

        if not ranked_lists:
            raise RuntimeError(
                f"All sub-queries failed for {index_key}: {failed_queries}"
            )

        merged  = _weighted_rrf(ranked_lists)
        results = merged[:candidate_k]

        response: dict[str, Any] = {
            "index_scope": index_scope,
            "index_key":   index_key,
            "results":     results,
            "skipped":     False,
        }
        if failed_queries:
            response["partial_failure"] = True
            response["failed_queries"]  = failed_queries
        return response

    except Exception as exc:
        log.error("StateSearchIndex failed index_scope=%s — %s: %s",
                  index_scope, type(exc).__name__, exc)
        traceback.print_exc()
        raise
