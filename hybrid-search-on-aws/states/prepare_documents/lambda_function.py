"""
StatePrepareDocuments — contracts in, prepared row groups out (lock-free).

The single entry state of the Index SFN (README.md Design record §2.4). Turns
content contracts (lib/contracts.py) into the typed row groups
StateIndexDocuments consumes, one S3 object per physical table:

  1. Load contracts: inline `documents` or `documents_s3uri` (JSONL).
  2. Resolve the collection (missing collection_id -> "default"; unknown ->
     fail hard). Validate every contract against its doc_type's schema.
  3. Compute doc_ids (backend-derived, D6) and dedup against the dedup
     table BEFORE any Bedrock call — extraction/embedding are the expensive
     steps; refresh=True documents bypass dedup.
  4. Process surviving content items:
       ANN field items are COLLECTED (text and/or ONE caller-selected
       image) then embedded once via embed_document() with the schema's
       own model/dims — text+image yields a JOINT (mixture) embedding
       where the provider supports it (Cohere/Titan; Nova: image wins)
       text -> other field: verbatim write; extract_entities=true also runs
                            the SLM entity pass, dedupe-append merged (D7)
  5. Upsert asset metadata (optional `asset_metadata` on the submission).
  6. Write one row-group object per index_key; return Map items
     [{index_key, rows_s3_key, collection_id, table_key, schema_file}].

Needs no lock — S3-index writes happen downstream; this state only reads
config + calls Bedrock. Mirrors decompose_search_request's lock-free
position in the Search SFN.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from botocore.exceptions import BotoCoreError, ClientError

from lib.backends.lancedb_backend import load_schema
from lib.bedrock_utils import call_slm_with_fallback, embed_document
from lib.collections import index_key as make_index_key, resolve_collection
from lib.contracts import merge_entities, validate_contract
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.index_state import batch_is_indexed, upsert_asset_metadata
from lib.input_schema import (MAX_INLINE_DOCUMENTS, MAX_MANIFEST_BYTES,
                              PROXY_BUCKET, REMAINING_TIME_BUFFER_MS)
from lib.s3_utils import (get_image_base64, get_json, get_object_text,
                          object_size, parse_s3_uri, put_json)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_ENTITY_FIELD = "named_entities"

# Two asymmetric classifications, deliberately:
# - The pre-embed HEAD sweep REFUSES only on codes known to mean "this object
#   will not appear on retry" (wrong-bucket included: NoSuchBucket and
#   301/PermanentRedirect for a bucket in another region) and lets anything
#   else through — a refusal there blocks the whole request, so it must be
#   certain.
# - The fetch-time wrap NAMES the document and key for every code EXCEPT the
#   known transients — an anonymous failure gives the caller nothing to fix,
#   and retryability is the wrong discriminator for whether to name it
#   (measured class: Glacier InvalidObjectState HEADs fine, fails on GET).
_PERMANENT_S3_CODES = ("404", "NoSuchKey", "403", "AccessDenied",
                       "NoSuchBucket", "301", "PermanentRedirect")
_TRANSIENT_S3_CODES = ("SlowDown", "ThrottlingException", "InternalError",
                       "ServiceUnavailable", "RequestTimeout", "503", "500")


def _sweep_image_uris(contracts: list[dict]) -> None:
    """Refuse-before-spend: HEAD every unique image URI in these contracts
    BEFORE any of them is embedded — a wrong path must cost zero model spend
    (measured live 2026-10-13: one guessed grid path failed a 93-doc scene
    batch whose text siblings had already embedded).  Permanent codes collect
    into ONE ValueError naming every bad URI with its owning documents
    (capped at 5 URIs / 3 docs each, with +N tallies); transient errors
    (throttles, connection failures) are ignored — the fetch path keeps its
    own handling.  Callers pass only PENDING contracts (post-checkpoint), so
    a retry never re-HEADs or re-judges images whose rows are already paid
    for and persisted."""
    owners: dict[str, list[str]] = {}
    for c in contracts:
        for item in c["content"]:
            if item["kind"] == "image":
                owners.setdefault(item["value"], []).append(c["doc_id"])
    if not owners:
        return

    def _entry(uri: str, why: str) -> str:
        docs = owners[uri]
        shown = ", ".join(docs[:3])
        more = f" +{len(docs) - 3} docs" if len(docs) > 3 else ""
        return f"doc(s) {shown}{more}: {uri} ({why})"

    def head(uri: str):
        try:
            bucket, key = parse_s3_uri(uri)
            object_size(bucket, key)
        except ValueError:
            # malformed uri (validate_contract only checks the s3:// prefix)
            # — as permanent as a missing object, and it must join the one
            # aggregated report, not abort the sweep anonymously
            return _entry(uri, "invalid s3 uri")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _PERMANENT_S3_CODES:
                return _entry(uri, code)
        except BotoCoreError:
            pass
        return None

    with ThreadPoolExecutor(max_workers=16) as ex:
        bad = [r for r in ex.map(head, owners) if r]
    if bad:
        extra = f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""
        raise ValueError(
            "image content item(s) unreadable — fix the URIs and resubmit "
            "(nothing from this attempt was embedded); if a PRIOR attempt "
            "checkpointed rows, restore the object(s) — or fix a "
            "documents_s3uri manifest in place — and REDRIVE instead, which "
            "keeps them: " + "; ".join(bad[:5]) + extra)


def _load_contracts(event: dict) -> list[dict]:
    docs = event.get("documents")
    uri  = (event.get("documents_s3uri") or "").strip()
    if bool(docs) == bool(uri):
        raise ValueError("exactly one of documents | documents_s3uri required")
    if docs:
        if not isinstance(docs, list) or not all(isinstance(d, dict) for d in docs):
            raise ValueError("documents must be a list of contract objects")
        if len(docs) > MAX_INLINE_DOCUMENTS:
            raise ValueError(
                f"{len(docs)} inline documents exceeds {MAX_INLINE_DOCUMENTS} — "
                f"use documents_s3uri (JSONL manifest) for bulk submissions")
        return list(docs)
    bucket, key = parse_s3_uri(uri)
    try:
        size = object_size(bucket, key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            # HeadObject 404s carry no bucket/key — name the URI ourselves
            # so the agent-facing error is actionable.
            raise ValueError(f"documents_s3uri not found: {uri}")
        raise
    if size > MAX_MANIFEST_BYTES:
        raise ValueError(
            f"{uri} is {size} bytes — exceeds the {MAX_MANIFEST_BYTES}-byte "
            f"manifest cap; split the submission")
    lines = get_object_text(bucket, key).splitlines()
    out = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{uri} line {n}: invalid JSON ({exc})")
    if not out:
        raise ValueError(f"{uri} contains no documents")
    return out


def _extract_entities(text: str, engine_cfg: dict) -> list[str]:
    """
    SLM entity extraction for callers WITHOUT pre-extracted entities.
    Reuses the same models the search decomposition uses (engine settings
    from the global workflow config).
    """
    primary  = engine_cfg.get("primaryModel", "")
    fallback = engine_cfg.get("secondaryModel") or None
    if not primary:
        log.warning("extract_entities requested but no primaryModel configured — skipping")
        return []
    system = ("You extract named entities (people, organizations, places, "
              "titles) from media text. Return ONLY an XML tag: "
              "<entities>comma, separated, names</entities>. "
              "No roles (mayor, patient), no generic nouns.")
    text_out, _, _, _ = call_slm_with_fallback(
        primary, fallback, system, text[:4000])
    m = re.search(r"<entities>(.*?)</entities>", text_out, re.DOTALL)
    if not m:
        return []
    return [e.strip() for e in m.group(1).split(",") if e.strip()]


def _process_document(contract: dict, schema_def, engine_cfg: dict,
                      embed_usage: dict | None = None) -> dict:
    """Contract -> row dict shaped for SchemaDefinition.to_row()."""
    row: dict = {
        "doc_id":        contract["doc_id"],
        "asset_id":      contract["asset_id"],
        "collection_id": contract["collection_id"],
        "start_ms":      contract["start_ms"],
        "end_ms":        contract["end_ms"],
        **contract["fields"],
    }
    ann_field = schema_def.ann_field
    entities: list[str] = []

    # ANN inputs are COLLECTED, then embedded once: text + image both
    # targeting the vector field produce a JOINT (mixture) embedding where
    # the provider supports it — never a silent last-writer-wins overwrite.
    ann_text: str | None = None
    ann_image_b64: str | None = None

    for item in contract["content"]:
        kind, field, value = item["kind"], item["field"], item["value"]
        if field == ann_field:
            if kind == "text":
                ann_text = f"{ann_text}\n{value}" if ann_text else value
            else:
                if ann_image_b64 is not None:
                    raise ValueError(
                        f"doc {contract['doc_id']}: multiple image items "
                        f"target {ann_field!r} — pre-select ONE representative "
                        f"frame upstream")
                bucket, key = parse_s3_uri(value)
                try:
                    ann_image_b64 = get_image_base64(bucket, key)
                except ClientError as exc:
                    code = exc.response.get("Error", {}).get("Code", "")
                    if code in _TRANSIENT_S3_CODES:
                        # Known-transient: stay a ClientError so the group
                        # checkpoints and a redrive resumes.
                        raise
                    # Everything else is permanent for THIS run (NoSuchKey,
                    # AccessDenied, Glacier InvalidObjectState, KMS denials,
                    # …): fail terminally and NAME the document and key — a
                    # bare anonymous ClientError gives the caller nothing to
                    # fix, and retryability is the wrong discriminator for
                    # whether to name it.  (The group's except-ValueError arm
                    # still checkpoints the rows embedded before this doc.)
                    raise ValueError(
                        f"doc {contract['doc_id']} "
                        f"(asset {contract.get('asset_id')!r}): image "
                        f"content item unreadable — {code or 'ClientError'} "
                        f"on {value}") from exc
        else:
            existing = row.get(field)
            row[field] = f"{existing}\n{value}" if existing else value
            if field == _ENTITY_FIELD:
                # Pre-structured entities (e.g. face-recognition people[]) —
                # comma-separated string, dedupe-append with any extracted set.
                entities = merge_entities(
                    entities, [e for e in str(value).split(",") if e.strip()])
        if item.get("extract_entities") and kind == "text" and field != _ENTITY_FIELD:
            entities = merge_entities(entities, _extract_entities(value, engine_cfg))

    if ann_text is not None or ann_image_b64 is not None:
        row[ann_field] = embed_document(
            schema_def.embedding_model, text=ann_text,
            image_b64=ann_image_b64, dims=schema_def.dim,
            usage=embed_usage)

    if entities and _ENTITY_FIELD in schema_def._mappings:
        row[_ENTITY_FIELD] = ", ".join(entities)
    elif entities:
        log.warning("entities extracted but schema %s has no %s field — dropped",
                    schema_def.doc_type, _ENTITY_FIELD)
    return row


def _load_checkpoint(rows_key: str) -> dict:
    """Row-group file from a previous attempt of THIS execution, or {}.
    request_id is the execution name, so cross-execution reuse is impossible."""
    try:
        return get_json(PROXY_BUCKET, rows_key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return {}
        raise


def _merge_usage(into: dict, delta: dict) -> None:
    for model_id, counts in delta.items():
        u = into.setdefault(model_id, {})
        for k, v in counts.items():
            u[k] = u.get(k, 0) + v


def lambda_handler(event: dict, context) -> dict:
    log.info("StatePrepareDocuments: collection_id=%r", event.get("collection_id"))
    try:
        collection = resolve_collection(event.get("collection_id"))
        collection_id = collection["collection_id"]
        schema_map    = collection.get("schema_map") or {}

        # Engine settings (SLM models for entity extraction) come from the
        # global workflow config — same source the Search SFN uses.
        engine_cfg = {}
        engine_uri = (event.get("workflow_config_uri") or "").strip()
        if engine_uri:
            b, k = parse_s3_uri(engine_uri)
            engine_cfg = get_json(b, k)

        contracts = _load_contracts(event)

        # ── Validate + group by doc_type ─────────────────────────────────────
        by_type: dict[str, list[dict]] = defaultdict(list)
        for i, raw in enumerate(contracts):
            doc_type = (raw.get("doc_type") or "").strip()
            if not doc_type:
                # Absent and empty are the same mistake but read differently:
                # name the missing field, not an empty-string comparison.
                raise ValueError(
                    f"document {i} has no doc_type — each document requires "
                    f"one matching an active scope of collection "
                    f"{collection_id!r}: {sorted(schema_map)}")
            schema_key = schema_map.get(doc_type, "")
            if not schema_key:
                raise ValueError(
                    f"document {i}: doc_type {doc_type!r} is not an active "
                    f"scope of collection {collection_id!r} — active: "
                    f"{sorted(schema_map)}")
            schema_def = load_schema(f"s3://{PROXY_BUCKET}/{schema_key}")
            by_type[doc_type].append(
                validate_contract(raw, schema_def, collection_id))

        # ── Dedup BEFORE any Bedrock spend ───────────────────────────────────
        non_refresh_ids = [c["doc_id"] for docs in by_type.values()
                           for c in docs if not c["refresh"]]
        already = batch_is_indexed(non_refresh_ids)

        # ── Process + write one row group per physical table ─────────────────
        # Checkpoint/resume: request_id = $$.Execution.Name is STABLE across
        # SFN Retry attempts, so the row-group keys double as resume markers.
        # A throttle mid-group checkpoints completed rows (partial=true) and
        # re-raises for the SFN Retry; re-entry reuses complete groups whole
        # and resumes partial groups after their last finished document —
        # Bedrock spend from failed attempts is never repeated.
        request_id = event.get("request_id") or "prepare"

        # ── Plan every group ONCE: survivors, checkpoint, pending ────────────
        # One structure read by both the sweep and the embed loop below — the
        # sweep's correctness rests on judging exactly the docs the embed
        # loop will read, so the two must never recompute the filter or the
        # checkpoint key independently.
        plans: dict[str, dict] = {}
        sweep_pending: list[dict] = []
        skipped = 0
        for doc_type, docs in by_type.items():
            survivors = [c for c in docs
                         if c["refresh"] or c["doc_id"] not in already]
            skipped += len(docs) - len(survivors)
            if not survivors:
                continue
            rows_key = (f"_indexing/{collection_id}/{request_id}/"
                        f"{doc_type}.json")
            checkpoint = _load_checkpoint(rows_key)
            rows = checkpoint.get("documents") or []
            complete = bool(checkpoint) and not checkpoint.get("partial")
            done = {r["doc_id"] for r in rows}
            pending = ([] if complete else
                       [c for c in survivors if c["doc_id"] not in done])
            plans[doc_type] = {"survivors": survivors, "rows_key": rows_key,
                               "checkpoint": checkpoint, "pending": pending,
                               "complete": complete, "rows": rows}
            sweep_pending += pending

        # ── Refuse-before-spend: sweep EVERY group's images before ANY embed ──
        # A per-group sweep would embed earlier groups in full before a later
        # group's bad path is discovered (the live incident was multi-scope:
        # text groups embedded, then the scene group's one wrong grid path
        # failed the run).  Nothing is embedded before this point, so the
        # error's "resubmit" remedy costs nothing.  Checkpoints are honored —
        # only PENDING docs' images are HEADed, so a vanished image behind
        # already-paid rows never kills a resume.
        _sweep_image_uris(sweep_pending)

        map_items: list[dict] = []
        embed_usage: dict = {}   # per-model Bedrock accounting (embed_document)
        total_rows = 0
        for doc_type, plan in plans.items():
            schema_key = schema_map[doc_type]
            schema_def = load_schema(f"s3://{PROXY_BUCKET}/{schema_key}")
            survivors  = plan["survivors"]
            rows_key   = plan["rows_key"]
            checkpoint = plan["checkpoint"]
            ikey       = make_index_key(collection_id, schema_def.table_key)

            rows        = plan["rows"]   # derived ONCE in the plan pass —
            #                              pending is defined against exactly
            #                              this list, so no re-derivation here
            # Each group's file stores ONLY its own usage delta — merging on
            # resume must never double-count another group's spend.
            group_usage = checkpoint.get("embed_usage") or {}
            _merge_usage(embed_usage, group_usage)
            if plan["complete"]:
                # Tier 1: this group finished on a previous attempt.
                log.info("[%s] resume: reusing complete group (%d rows)",
                         doc_type, len(rows))
            else:
                if rows:
                    # Tier 2: resume after the last checkpointed document.
                    log.info("[%s] resume: %d/%d rows from checkpoint",
                             doc_type, len(rows), len(survivors))
                pending = plan["pending"]
                group   = {
                    "schema_file_uri":   f"s3://{PROXY_BUCKET}/{schema_key}",
                    "collection_id":     collection_id,
                    # Per-ASSET refresh: only assets whose contracts asked for
                    # it get their existing rows cleared — a multi-asset
                    # submission must not wipe a bystander asset's rows.
                    "refresh_asset_ids": sorted({c["asset_id"] for c in survivors
                                                 if c["refresh"]}),
                    "documents":         rows,
                }
                delta: dict = {}

                def _checkpoint_partial(reason: str) -> None:
                    # Best-effort write; the caller re-raises UNCONDITIONALLY
                    # so the SFN Retry fires. A failed checkpoint write
                    # degrades to re-embedding the group, never masks the
                    # original error.
                    _merge_usage(group_usage, delta)
                    try:
                        put_json(PROXY_BUCKET, rows_key, {
                            **group, "partial": True,
                            "embed_usage": group_usage})
                        log.info("[%s] %s at %d/%d — checkpointed",
                                 doc_type, reason, len(rows), len(survivors))
                    except Exception as ckpt_exc:
                        log.warning("[%s] checkpoint write failed (%s) — "
                                    "retry restarts the group", doc_type, ckpt_exc)

                try:
                    for c in pending:
                        # Lambda-timeout guard: a Sandbox.Timedout kills the
                        # process mid-loop with NO checkpoint — exit early
                        # instead, so large assets (3k+ docs) resume across
                        # attempts rather than restarting from zero.
                        if (context is not None
                                and context.get_remaining_time_in_millis()
                                    < REMAINING_TIME_BUFFER_MS):
                            _checkpoint_partial("time budget")
                            raise LambdaResourceTimeoutException(
                                f"{doc_type}: {len(rows)}/{len(survivors)} done — "
                                f"resuming on next attempt")
                        rows.append(_process_document(
                            c, schema_def, engine_cfg, delta))
                except ClientError as exc:
                    # Checkpoint on ANY Bedrock/S3 ClientError, not just the
                    # two throttle codes — the write is best-effort and only
                    # ever helps (fewer avoidable re-embeds on redrive), and
                    # narrowing to specific codes previously meant a
                    # transient InternalServerException/ModelTimeoutException
                    # lost every row in the group with no way to resume.
                    code = exc.response.get("Error", {}).get("Code", "")
                    _checkpoint_partial(f"error ({code or type(exc).__name__})")
                    raise
                except BotoCoreError as exc:
                    # Transport blip (ReadTimeout/ConnectionClosed) mid-group:
                    # same treatment as ClientError — the rows embedded this
                    # attempt must survive for the redrive (these errorTypes
                    # match no ASL Retry rule either).
                    _checkpoint_partial(f"error ({type(exc).__name__})")
                    raise
                except ValueError:
                    # Terminal per-document failure named upstream — the
                    # fetch-time wrap (object vanished between the pre-embed
                    # HEAD sweep and this GET) or any other per-doc contract/
                    # provider refusal; checkpoint-then-raise is right for
                    # all of them.  The checkpoint pays off on REDRIVE (same
                    # input — object restored, or a documents_s3uri manifest
                    # fixed in place); an inline RESUBMIT mints a new
                    # request_id and cannot see these checkpoints — only
                    # dedup of fully-INDEXED docs survives that route.
                    _checkpoint_partial("permanent document error")
                    raise
                _merge_usage(group_usage, delta)
                _merge_usage(embed_usage, delta)
                # Complete: partial marker cleared.
                put_json(PROXY_BUCKET, rows_key, {
                    **group, "embed_usage": group_usage})

            map_items.append({
                "index_key":     ikey,
                "rows_s3_key":   rows_key,
                "collection_id": collection_id,
                "table_key":     schema_def.table_key,
            })
            total_rows += len(rows)

        # ── Asset metadata (optional, once per submission) ───────────────────
        meta = event.get("asset_metadata") or {}
        if meta and contracts:
            asset_id = contracts[0].get("asset_id", "")
            if asset_id:
                upsert_asset_metadata(
                    collection_id, asset_id,
                    genres=meta.get("genres") or [],
                    content_category=meta.get("content_category") or "",
                    title=meta.get("title") or "",
                    casts=meta.get("casts") or [],
                )

        log.info("Prepared %d rows in %d group(s), %d deduped, collection=%s",
                 total_rows, len(map_items), skipped, collection_id)
        return {
            "collection_id": collection_id,
            "map_items":     map_items,
            "prepared":      total_rows,
            "skipped":       skipped,
            "embed_usage":   embed_usage,
        }

    except LambdaResourceTimeoutException as exc:
        # Expected control flow on large assets, not a failure: progress is
        # already checkpointed and the SFN Retry resumes it. Logging this
        # as an ERROR with a traceback would page on-call for every
        # multi-attempt run of a normal large asset.
        log.info("StatePrepareDocuments checkpointed — %s", exc)
        raise
    except Exception as exc:
        log.error("StatePrepareDocuments failed — %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise
