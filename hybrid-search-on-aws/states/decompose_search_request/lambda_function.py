"""
StateDecomposeSearchRequest — parse search query, embed, build query items.

Collection-scoped (README.md Design record §3): TENANT settings (active scopes,
schema_map, retrieval weights) come from the collection record — missing
collection_id resolves to "default", seeded at deploy time. ENGINE settings
(SLM models, prompt) still come from the global workflow config.

Two entry points:
  _decompose_text_search  — SLM query parsing + text embedding
  _decompose_image_search — image download + image embedding, no SLM

Both return a shared decomposition dict consumed by StateSearchIndex (Map fan-out).
"""
from __future__ import annotations

import logging
import re
import time
import traceback
from typing import Any

from lib.input_schema import (
    PROXY_BUCKET,
    CANDIDATE_K_MULTIPLIER,
    CANDIDATE_K_MIN,
)
from lib.s3_utils import get_json, get_object_text, get_image_base64, parse_s3_uri, put_json
from lib.collections import index_key as make_index_key, resolve_collection
from lib.index_state import (
    get_table_state,
    query_assets_by_title,
    write_search_history_initial,
)
from lib.bedrock_utils import call_slm_with_fallback, embed, embed_image

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ── Operator parsing ───────────────────────────────────────────────────────────

def _parse_operators(search_term: str) -> dict:
    asset_filter  = ""
    index_scopes  = []
    dialog_query  = ""
    quoted_phrase = ""

    text = search_term

    # says:"..."
    m = re.search(r'says:"([^"]*)"', text)
    if m:
        dialog_query = m.group(1).strip()
        text = text[:m.start()] + text[m.end():]

    # bare "..."
    m = re.search(r'"([^"]*)"', text)
    if m:
        quoted_phrase = m.group(1).strip()
        text = text[:m.start()] + m.group(1) + text[m.end():]

    # @Title
    m = re.search(r'@([^@#]+?)(?=\s*[@#]|$)', text)
    if m:
        asset_filter = m.group(1).strip()
        text = text[:m.start()] + text[m.end():]

    # #scope tokens
    for scope_m in re.finditer(r'#(\w+)', text):
        index_scopes.append(scope_m.group(1).lower())
    text = re.sub(r'#\w+', '', text)

    clean_query = re.sub(r'\s+', ' ', text).strip()
    return {
        "clean_query":   clean_query,
        "asset_filter":  asset_filter,
        "index_scopes":  index_scopes,
        "dialog_query":  dialog_query,
        "quoted_phrase": quoted_phrase,
    }


# ── XML parsing ───────────────────────────────────────────────────────────────

def _xml_field(xml: str, tag: str) -> str:
    m = re.search(rf'<{tag}>(.*?)</{tag}>', xml, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_slm_xml(xml: str) -> dict:
    retrieval_paths = [p.strip() for p in _xml_field(xml, "retrieval_paths").split(",") if p.strip()]
    entities        = [e.strip() for e in _xml_field(xml, "entities").split(",") if e.strip()]
    index_scopes_s  = _xml_field(xml, "index_scopes")
    # lowercase to match _parse_operators — a case-variant scope name from
    # the model must restrict, not silently match nothing
    index_scopes    = [s.strip().lower() for s in index_scopes_s.split(",") if s.strip()] if index_scopes_s else []
    return {
        "retrieval_paths": retrieval_paths,
        "entities":        entities,
        "dialog_query":    _xml_field(xml, "dialog_query"),
        "quoted_phrase":   _xml_field(xml, "quoted_phrase"),
        "asset_filter":    _xml_field(xml, "asset_filter"),
        "index_scopes":    index_scopes,
        "clean_query":     _xml_field(xml, "clean_query"),
    }


# ── Subquery spec derivation ──────────────────────────────────────────────────

def _derive_subquery_spec(schema_defs: dict[str, dict]) -> dict:
    spec: dict = {}
    for scope, sd in schema_defs.items():
        for field_name, field_def in sd.get("mappings", {}).items():
            if field_def.get("ann"):
                spec[f"{scope}_ann"] = {"scope": scope, "type": "ann", "field": field_name}
            elif field_def.get("fts") not in (None, False):
                spec[f"{scope}_{field_name}_fts"] = {"scope": scope, "type": "fts", "field": field_name}
    return spec


# ── Weight computation ────────────────────────────────────────────────────────

def _compute_weights(retrieval_paths: list[str], path_weights: dict) -> dict[str, float]:
    """Max-pool weights across all active retrieval paths."""
    merged: dict[str, float] = {}
    for path in retrieval_paths:
        for subq, w in path_weights.get(path, {}).items():
            merged[subq] = max(merged.get(subq, 0.0), w)
    return merged


def _backfill_scope_weights(
    weights: dict[str, float],
    restricted_scopes: list[str],
    subquery_spec: dict,
    path_weights: dict,
) -> dict[str, float]:
    """A user-typed scope restriction (#utterance) is the strongest statement
    of intent in a query. The path classifier chooses what to EMPHASIZE; it
    must not veto a scope the caller named — the two used to intersect to a
    silent empty result ('#utterance' + a visually-classified query searched
    nothing). For the named scopes, backfill zero weights by max-pooling over
    EVERY configured path, so the scope runs with the collection's own
    weights rather than an invented constant.

    restricted_scopes must be exactly the user-restricted set (already
    intersected with the collection's schemas) — the caller decides WHEN a
    restriction exists; this function only applies it. Pure: the input dict
    is never modified."""
    out = dict(weights)
    everything = _compute_weights(list(path_weights), path_weights)
    for subq_key, sq in subquery_spec.items():
        if sq["scope"] in restricted_scopes and not out.get(subq_key):
            w = everything.get(subq_key, 0.0)
            if w:
                out[subq_key] = w
                log.info("Scope restriction backfilled weight %s=%s "
                         "(paths gave it none)", subq_key, w)
    return out


# ── Query item builder ────────────────────────────────────────────────────────

def _build_query_items(
    active_scopes: list[str],
    weights: dict[str, float],
    has_vectors: set[str],
    vectors_s3_key: str,
    asset_ids: list[str],
    k: int,
    candidate_k: int,
    entities: list[str],
    dialog_query: str,
    quoted_phrase: str,
    schema_map: dict[str, str],
    state_records: dict[str, dict],
    subquery_spec: dict,
    collection_id: str,
) -> tuple[list[dict], list[dict]]:
    """Returns (query items, dropped scopes). A dropped scope will not be
    searched; entries use the SAME vocabulary as StateSearchIndex's skip
    items ({index_scope, skip_reason}). The CALLER decides which drops to
    surface — for a restricted query they distinguish "no matches" from
    "never searched"; for an unrestricted one, classifier-de-emphasized
    scopes are normal and not reported."""
    items = []
    dropped: list[dict] = []
    for scope in active_scopes:
        schema_key   = schema_map.get(scope, "")
        # state record sk IS the index_key ("{collection}:{table_key}")
        index_key    = state_records.get(scope, {}).get("sk", "")
        scope_has_v  = scope in has_vectors

        sub_queries: list[dict] = []
        scope_weight = 0.0

        for subq_key, sq in subquery_spec.items():
            sq_scope, sq_type, sq_field = sq["scope"], sq["type"], sq["field"]
            if sq_scope != scope:
                continue
            w = weights.get(subq_key, 0.0)
            if w == 0.0:
                continue
            if sq_type == "ann" and not scope_has_v:
                continue
            entry: dict[str, Any] = {"type": sq_type, "field": sq_field, "weight": w}
            if sq_type == "fts":
                if sq_field == "named_entities" and entities:
                    entry["query"] = " ".join(entities)
                elif sq_field == "text":
                    if dialog_query:
                        entry["query"] = dialog_query
                    elif quoted_phrase:
                        entry["query"] = quoted_phrase
                    else:
                        continue  # no text to search
            if "query" in entry or sq_type == "ann":
                sub_queries.append(entry)
                scope_weight = max(scope_weight, w)

        if not sub_queries:
            log.warning("Scope %s has no runnable sub-queries — dropped", scope)
            dropped.append({"index_scope": scope,
                            "skip_reason": "no_subqueries"})
            continue

        if not index_key:
            log.warning("No registered DDB state for scope=%s — skipping from query fan-out", scope)
            dropped.append({"index_scope": scope,
                            "skip_reason": "no_state_record"})
            continue

        item: dict[str, Any] = {
            "index_scope":    scope,
            "index_key":      index_key,
            "collection_id":  collection_id,
            "schema_key":     schema_key,
            "asset_ids":      asset_ids,
            "k":              k,
            "candidate_k":    candidate_k,
            "index_weight":   round(scope_weight, 4),
            "sub_queries":    sub_queries,
        }
        if scope_has_v:
            item["vectors_s3_key"] = vectors_s3_key

        items.append(item)
    return items, dropped


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_schema_defs(active_scopes: list[str], schema_map: dict[str, str]) -> dict[str, dict]:
    schema_defs: dict[str, dict] = {}
    for scope in active_scopes:
        rel_key = schema_map.get(scope, "")
        if rel_key:
            schema_defs[scope] = get_json(PROXY_BUCKET, rel_key)
    return schema_defs


def _load_state_records(schema_defs: dict[str, dict],
                        collection_id: str) -> dict[str, dict]:
    state_records: dict[str, dict] = {}
    for scope, sd in schema_defs.items():
        doc_type = sd.get("doc_type", scope)
        em       = sd.get("embedding_model", "")
        lang     = sd.get("language", "")
        tkey     = f"{doc_type}:{em}:{lang}" if lang else f"{doc_type}:{em}"
        st       = get_table_state(make_index_key(collection_id, tkey))
        if st:
            state_records[scope] = st
    return state_records


def _embed_specs(schema_defs: dict[str, dict]) -> dict[str, tuple[str, int | None]]:
    return {
        scope: (sd.get("embedding_model", ""), sd.get("dims") or None)
        for scope, sd in schema_defs.items()
    }


def _write_vectors(vectors: dict[str, list[float]], request_id: str) -> str:
    if not vectors or not request_id:
        return ""
    key = f"_search/{request_id}/vectors.json"
    put_json(PROXY_BUCKET, key, vectors)
    log.info("Wrote vectors to s3://%s/%s", PROXY_BUCKET, key)
    return key


# ── Text search ───────────────────────────────────────────────────────────────

def _decompose_text_search(
    search_term: str,
    workflow_cfg: dict,
    collection: dict,
    all_schemas: list[str],
    schema_map: dict[str, str],
    request_id: str,
    k: int,
    candidate_k: int,
) -> dict:
    collection_id  = collection["collection_id"]
    primary_model  = workflow_cfg.get("primaryModel", "")
    fallback_model = workflow_cfg.get("secondaryModel", "") or None
    prompt_s3_key  = workflow_cfg.get("promptS3Key", "")
    path_weights   = collection.get("retrieval_paths") or {}

    if not primary_model:
        raise ValueError("workflow config missing primaryModel")
    if not prompt_s3_key:
        raise ValueError("workflow config missing promptS3Key")
    if not path_weights:
        raise ValueError(
            f"collection {collection_id!r} has no retrieval_paths configured")

    # Call SLM
    prompt_text           = get_object_text(PROXY_BUCKET, prompt_s3_key)
    system_prompt, user_t = _split_prompt(prompt_text)
    user_prompt           = user_t.replace("{{search_query}}", search_term)

    raw_xml, model_used, fallback_triggered, fallback_reason = call_slm_with_fallback(
        primary_model, fallback_model, system_prompt, user_prompt,
    )
    log.info("SLM raw output: %d chars", len(raw_xml))

    slm = _parse_slm_xml(raw_xml)
    op  = _parse_operators(search_term)

    retrieval_paths = slm["retrieval_paths"] or ["visual"]
    entities        = slm["entities"]
    clean_query     = op["clean_query"] or slm["clean_query"] or search_term
    dialog_query    = op["dialog_query"] or slm["dialog_query"]
    quoted_phrase   = op["quoted_phrase"] or slm["quoted_phrase"]
    asset_filter    = op["asset_filter"] or slm["asset_filter"]
    # restriction: [] means unrestricted (search everything). op scopes are
    # user-typed # tokens; slm scopes are the model's inference — both narrow
    # the fan-out, but only USER-TYPED scopes arm the weight backfill below
    # (a hallucinated slm scope list must not neuter the path classification).
    restriction     = op["index_scopes"] or slm["index_scopes"]
    index_scopes    = restriction or all_schemas
    # A restricted name that matches no schema would otherwise vanish in the
    # active_scopes intersection — the silent-empty trap, one layer up.
    unknown_scopes  = [{"index_scope": s, "skip_reason": "unknown_scope"}
                       for s in dict.fromkeys(restriction)
                       if s not in all_schemas]

    # Resolve @ operator
    asset_ids: list[str] = []
    if asset_filter:
        matches   = query_assets_by_title(collection_id, asset_filter)
        asset_ids = [m["asset_id"] for m in matches if "asset_id" in m]
        log.info("@ operator %r resolved to %d asset_ids", asset_filter, len(asset_ids))

    active_scopes = [s for s in all_schemas if s in index_scopes]
    schema_defs   = _load_schema_defs(active_scopes, schema_map)
    specs         = _embed_specs(schema_defs)

    # Embed clean_query once per unique (model_id, dims)
    unique_specs = {spec for spec in specs.values() if spec[0]}
    vectors_by_spec: dict[tuple[str, int | None], list[float]] = {}
    for em, dims in unique_specs:
        vectors_by_spec[(em, dims)] = embed(em, clean_query, dims)
        log.info("Embedded with model %s dims=%d", em, len(vectors_by_spec[(em, dims)]))

    vectors = {scope: vectors_by_spec[spec] for scope, spec in specs.items() if spec in vectors_by_spec}

    subquery_spec  = _derive_subquery_spec(schema_defs)
    weights        = _compute_weights(retrieval_paths, path_weights)
    # backfill arms on USER-TYPED # tokens only — the restricted set is
    # derived here, independently of active_scopes, and passed explicitly
    user_restricted = [s for s in all_schemas if s in op["index_scopes"]]
    if user_restricted:
        weights = _backfill_scope_weights(weights, user_restricted,
                                          subquery_spec, path_weights)
    state_records  = _load_state_records(schema_defs, collection_id)
    vectors_s3_key = _write_vectors(vectors, request_id)

    queries, dropped = _build_query_items(
        active_scopes=active_scopes,
        weights=weights,
        has_vectors=set(vectors.keys()),
        vectors_s3_key=vectors_s3_key,
        asset_ids=asset_ids,
        k=k,
        candidate_k=candidate_k,
        entities=entities,
        dialog_query=dialog_query,
        quoted_phrase=quoted_phrase,
        schema_map=schema_map,
        state_records=state_records,
        subquery_spec=subquery_spec,
        collection_id=collection_id,
    )
    # Only a RESTRICTED query reports builder drops — on an unrestricted one,
    # scopes the classifier de-emphasized to zero are normal, not "never
    # searched" noise. Unknown restricted names are always reported.
    dropped_scopes = unknown_scopes + (dropped if restriction else [])

    return {
        "collection_id":       collection_id,
        "original_query":      search_term,
        "clean_query":         clean_query,
        "retrieval_paths":     retrieval_paths,
        "entities":            entities,
        "dialog_query":        dialog_query,
        "quoted_phrase":       quoted_phrase,
        "asset_filter":        asset_filter,
        "asset_ids":           asset_ids,
        "index_scopes":        index_scopes,
        "language_model_used": model_used,
        "fallback_triggered":  fallback_triggered,
        "fallback_reason":     fallback_reason,
        "vectors_s3_key":      vectors_s3_key,
        "queries":             queries,
        "dropped_scopes":      dropped_scopes,
    }


# ── Image search ──────────────────────────────────────────────────────────────

def _decompose_image_search(
    search_image_s3uri: str,
    workflow_cfg: dict,
    collection: dict,
    all_schemas: list[str],
    schema_map: dict[str, str],
    request_id: str,
    k: int,
    candidate_k: int,
) -> dict:
    collection_id = collection["collection_id"]
    path_weights  = collection.get("image_retrieval_paths") or {}
    if not path_weights:
        raise ValueError(
            f"collection {collection_id!r} has no image_retrieval_paths configured")

    retrieval_paths = list(path_weights.keys())

    # Only load schemas for scopes that have ANN weight in imageRetrievalPaths.
    # Avoids calling embed_image on text-only models that have no image support.
    image_scopes = {
        subq_key.rsplit("_ann", 1)[0]
        for path_w in path_weights.values()
        for subq_key in path_w
        if subq_key.endswith("_ann")
    }
    active_scopes = [s for s in all_schemas if s in image_scopes]
    # imageRetrievalPaths is an explicit operator statement of what to search
    # — a configured scope the collection no longer activates must be
    # reported, not silently thinned out of the fan-out (admin tooling
    # validates the two settings independently, so they CAN disagree).
    inactive_scopes = [{"index_scope": s, "skip_reason": "inactive_scope"}
                       for s in sorted(image_scopes)
                       if s not in all_schemas]
    schema_defs   = _load_schema_defs(active_scopes, schema_map)
    specs           = _embed_specs(schema_defs)

    # Download image once; embed per unique (model_id, dims)
    img_bucket, img_key = parse_s3_uri(search_image_s3uri)
    image_b64 = get_image_base64(img_bucket, img_key)
    log.info("Downloaded image from s3://%s/%s", img_bucket, img_key)

    unique_specs = {spec for spec in specs.values() if spec[0]}
    vectors_by_spec: dict[tuple[str, int | None], list[float]] = {}
    for em, dims in unique_specs:
        vectors_by_spec[(em, dims)] = embed_image(em, image_b64, dims)
        log.info("Image-embedded with model %s dims=%d", em, len(vectors_by_spec[(em, dims)]))

    vectors = {scope: vectors_by_spec[spec] for scope, spec in specs.items() if spec in vectors_by_spec}

    subquery_spec  = _derive_subquery_spec(schema_defs)
    weights        = _compute_weights(retrieval_paths, path_weights)
    state_records  = _load_state_records(schema_defs, collection_id)
    vectors_s3_key = _write_vectors(vectors, request_id)

    queries, dropped = _build_query_items(
        active_scopes=active_scopes,
        weights=weights,
        has_vectors=set(vectors.keys()),
        vectors_s3_key=vectors_s3_key,
        asset_ids=[],
        k=k,
        candidate_k=candidate_k,
        entities=[],
        dialog_query="",
        quoted_phrase="",
        schema_map=schema_map,
        state_records=state_records,
        subquery_spec=subquery_spec,
        collection_id=collection_id,
    )
    # every image scope was explicitly configured — all drops are reportable
    dropped_scopes = inactive_scopes + dropped

    return {
        "collection_id":       collection_id,
        "original_query":      search_image_s3uri,
        "clean_query":         "",
        "retrieval_paths":     retrieval_paths,
        "entities":            [],
        "dialog_query":        "",
        "quoted_phrase":       "",
        "asset_filter":        "",
        "asset_ids":           [],
        "index_scopes":        active_scopes,
        "language_model_used": None,
        "fallback_triggered":  False,
        "fallback_reason":     None,
        "vectors_s3_key":      vectors_s3_key,
        "queries":             queries,
        "dropped_scopes":      dropped_scopes,
        "search_image_s3uri":    search_image_s3uri,
    }


# ── Main handler ──────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    search_term      = event.get("search_term", "")
    search_image_s3uri = event.get("search_image_s3uri", "")
    is_image_search  = bool(search_image_s3uri) and not search_term

    log.info(
        "StateDecomposeSearchRequest: search_term=%r search_image_s3uri=%r",
        search_term, search_image_s3uri,
    )

    workflow_config_uri = event.get("workflow_config_uri", "")
    k                   = int(event.get("k", 10))
    request_id          = event.get("request_id", "")

    if not search_term and not search_image_s3uri:
        raise ValueError("Missing required field: search_term or search_image_s3uri")
    if not workflow_config_uri:
        raise ValueError("Missing required field: workflow_config_uri")

    candidate_k = max(CANDIDATE_K_MIN, k * CANDIDATE_K_MULTIPLIER)
    created_at  = int(time.time())

    try:
        # ENGINE settings (SLM models, prompt) — global workflow config.
        cfg_bucket, cfg_key = parse_s3_uri(workflow_config_uri)
        workflow_cfg = get_json(cfg_bucket, cfg_key)

        # TENANT settings (scopes, schema_map, weights) — collection record.
        # Missing collection_id -> "default" (seeded at deploy); unknown ->
        # fail hard.
        collection  = resolve_collection(event.get("collection_id"))
        all_schemas = list(collection.get("active_scopes") or [])
        schema_map  = dict(collection.get("schema_map") or {})

        if is_image_search:
            result = _decompose_image_search(
                search_image_s3uri=search_image_s3uri,
                workflow_cfg=workflow_cfg,
                collection=collection,
                all_schemas=all_schemas,
                schema_map=schema_map,
                request_id=request_id,
                k=k,
                candidate_k=candidate_k,
            )
        else:
            result = _decompose_text_search(
                search_term=search_term,
                workflow_cfg=workflow_cfg,
                collection=collection,
                all_schemas=all_schemas,
                schema_map=schema_map,
                request_id=request_id,
                k=k,
                candidate_k=candidate_k,
            )

        decomposition = {"request_id": request_id, **result}

        if request_id:
            write_search_history_initial(
                request_id=request_id,
                search_term=decomposition["original_query"],
                workflow_config_uri=workflow_config_uri,
                decomposition=decomposition,
                k=k,
                created_at=created_at,
            )

        log.info(
            "Decompose complete: %d queries, paths=%s entities=%s image_search=%s",
            len(decomposition["queries"]), decomposition["retrieval_paths"],
            decomposition["entities"], is_image_search,
        )
        return decomposition

    except Exception as exc:
        log.error("StateDecomposeSearchRequest failed — %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise


# ── Prompt splitting ──────────────────────────────────────────────────────────

def _split_prompt(text: str) -> tuple[str, str]:
    sys_m  = re.search(r'^#\s*System Prompt\s*$', text, re.MULTILINE)
    user_m = re.search(r'^#\s*User Prompt\s*$',   text, re.MULTILINE)
    if not sys_m or not user_m:
        return "", text
    system = text[sys_m.end():user_m.start()].strip()
    user   = text[user_m.end():].strip()
    system = re.sub(r'\n---\s*$', '', system).strip()
    user   = re.sub(r'\n---\s*$', '', user).strip()
    return system, user
