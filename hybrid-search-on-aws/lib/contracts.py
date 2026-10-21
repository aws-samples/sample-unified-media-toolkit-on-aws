"""
contracts.py — the content contract: what callers hand to index_content.

One contract object per DOCUMENT (README.md Design record §2.2):

  {
    "collection_id": "news",          // omitted -> "default" (resolved upstream)
    "asset_id": "asset:uuid:abc123",
    "doc_type": "scene",
    "start_ms": 12000, "end_ms": 18500,
    "chunk_idx": 0,                   // optional; same-timespan multi-doc
    "refresh": false,                 // optional; delete this asset's rows first
    "fields": { "segment_parent_id": "..." },   // verbatim scalars
    "content": [                      // items needing backend PROCESSING
      {"kind": "text",  "field": "text", "value": "...",
       "extract_entities": true},
      {"kind": "text",  "field": "named_entities",
       "value": "Jane Doe, Richard Roe"},             // pre-structured, no SLM
      {"kind": "image", "field": "vector",
       "value": "s3://bucket/frames/frame_0045.jpg"}  // caller-selected frame
    ]
  }

Division of responsibility (locked): the CALLER selects/composes content
(representative frame, text blob); the BACKEND does all processing
(embedding via the schema's own model/dims, entity extraction, FTS/ANN
writes). `kind` is a closed enum — text | image. Adding a kind is a backend
code change, never caller-extensible.

doc_id is backend-derived (locked decision D6):
  md5("{collection_id}:{asset_id}:{doc_type}:{start_ms}:{end_ms}:{chunk_idx}")
collection_id in the hash prevents cross-collection dedup collisions;
computable BEFORE any Bedrock call so dedup-before-extraction works.
"""
from __future__ import annotations

import hashlib
import logging

log = logging.getLogger(__name__)

VALID_KINDS = ("text", "image")


def compute_doc_id(collection_id: str, asset_id: str, doc_type: str,
                   start_ms: int, end_ms: int, chunk_idx: int = 0) -> str:
    raw = f"{collection_id}:{asset_id}:{doc_type}:{start_ms}:{end_ms}:{chunk_idx}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


def validate_contract(doc: dict, schema_def, collection_id: str) -> dict:
    """
    Validate one contract document against its target schema. Fail hard on
    unknown fields / kinds (a typo silently dropping content is worse than
    a crash). Returns a normalized copy with doc_id computed.
    """
    asset_id = (doc.get("asset_id") or "").strip()
    doc_type = (doc.get("doc_type") or "").strip()
    if not asset_id:
        raise ValueError("contract missing asset_id")
    if doc_type != schema_def.doc_type:
        raise ValueError(
            f"contract doc_type {doc_type!r} does not match schema "
            f"doc_type {schema_def.doc_type!r}")
    try:
        start_ms = int(doc["start_ms"])
        end_ms   = int(doc["end_ms"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"contract for asset {asset_id!r} missing/invalid start_ms/end_ms")
    chunk_idx = int(doc.get("chunk_idx") or 0)

    mappings  = schema_def._mappings
    ann_field = schema_def.ann_field

    fields = doc.get("fields") or {}
    for f in fields:
        if f not in mappings:
            raise ValueError(
                f"contract fields[{f!r}] not in schema mappings for "
                f"doc_type {doc_type!r} — known: {sorted(mappings)}")
        if f == ann_field:
            raise ValueError(
                f"contract fields[{f!r}] targets the ANN vector column — "
                f"vectors are backend-computed; use a content item instead")

    content = doc.get("content") or []
    if not content and not fields:
        raise ValueError(f"contract for asset {asset_id!r} has no content or fields")
    if not any(item.get("field") == ann_field for item in content):
        raise ValueError(
            f"contract for asset {asset_id!r} has no content targeting the "
            f"vector field {ann_field!r} — every document needs at least one "
            f"text or image item to embed")
    for i, item in enumerate(content):
        kind  = item.get("kind")
        field = item.get("field", "")
        value = item.get("value")
        if kind not in VALID_KINDS:
            raise ValueError(
                f"content[{i}].kind {kind!r} invalid — must be one of {VALID_KINDS}")
        if field not in mappings:
            raise ValueError(
                f"content[{i}].field {field!r} not in schema mappings for "
                f"doc_type {doc_type!r} — known: {sorted(mappings)}")
        if not value:
            raise ValueError(f"content[{i}] ({field!r}) has empty value")
        if kind == "image":
            if field != ann_field:
                raise ValueError(
                    f"content[{i}]: kind=image must target the ANN field "
                    f"{ann_field!r}, got {field!r}")
            if not str(value).startswith("s3://"):
                raise ValueError(
                    f"content[{i}]: image value must be an s3:// uri")

    return {
        "collection_id": collection_id,
        "asset_id":      asset_id,
        "doc_type":      doc_type,
        "start_ms":      start_ms,
        "end_ms":        end_ms,
        "chunk_idx":     chunk_idx,
        "refresh":       bool(doc.get("refresh", False)),
        "fields":        dict(fields),
        "content":       list(content),
        "doc_id":        compute_doc_id(collection_id, asset_id, doc_type,
                                        start_ms, end_ms, chunk_idx),
    }


def merge_entities(existing: list[str], new: list[str]) -> list[str]:
    """
    Dedupe-append (locked decision D7): normalize (lowercase+strip) for
    comparison only; keep first-seen original casing for storage. Matches
    named_entities' query-time lower_case FTS tokenization.
    """
    seen = {e.lower().strip() for e in existing}
    out  = list(existing)
    for e in new:
        norm = e.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(e.strip())
    return out
