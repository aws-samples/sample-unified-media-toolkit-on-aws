"""
collections.py — collection registry (LanceDBCollections table).

A collection is a named tenant grouping (news, sport, episodic…) with
level-2 storage isolation (README.md Design record §1): per-collection physical
LanceDB tables and S3 Vectors indexes, shared compute and DynamoDB tables.

Registry item (pk = collection_id, hash key only):
  name                   display name
  active_scopes          list<doc_type>
  schema_map             map<doc_type, schema_file_key> (shared
                         _lancedb_index/schema/ directory — definitions are
                         shared; only DATA locations are per-collection)
  retrieval_paths        text-search sub-query weights (per-collection)
  image_retrieval_paths  image-search sub-query weights (per-collection)
  weights_locked         True once an admin edited weights; blocks further
                         normal-caller writes to the two weight maps ONLY
                         (weight-freeze, not full-freeze)
  created_by / created_at / updated_at

index_key = "{collection_id}:{table_key}" is the single physical-unit
identity used for: LanceDB S3 location, S3 Vectors index name, the state
record sk, the lock sk, and the dedup table_name attribute.

The "default" collection is seeded at deploy time by the CopySchema custom
resource from workflow_config/hybrid-search.json (skip-if-exists). Callers
that omit collection_id resolve to it — one defaulting line, no alternate
code path.

Ownership split: this module is the CONTAINER-side READ path (resolve/
derive). Registry WRITES (create/configure/delete) live in the gateway
Lambdas (gateway_tool/), which ship as a flat zip without lib/ and use the
low-level ddb client directly.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time

import boto3

from lib.input_schema import AWS_REGION, COLLECTION_TABLE, DEFAULT_COLLECTION_ID

log = logging.getLogger(__name__)

_dynamodb         = boto3.resource("dynamodb", region_name=AWS_REGION)
_collection_table = None

# lowercase alnum + hyphen/underscore, 3–48 chars. Must never collide with
# the shared schema directory name or a bare doc_type (defense-in-depth —
# no un-prefixed doc_type locations exist anymore, but keep the rule cheap).
_COLLECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,47}$")
_RESERVED_IDS     = {"schema"}


def _table():
    global _collection_table
    if _collection_table is None:
        _collection_table = _dynamodb.Table(COLLECTION_TABLE)
    return _collection_table


def validate_collection_id(collection_id: str, known_doc_types: set[str] | None = None) -> str:
    """Validate and return collection_id. Raises ValueError on bad input."""
    cid = (collection_id or "").strip()
    if not _COLLECTION_ID_RE.match(cid):
        raise ValueError(
            f"collection_id {cid!r} invalid — must match [a-z0-9][a-z0-9_-]{{2,47}}")
    if cid in _RESERVED_IDS or (known_doc_types and cid in known_doc_types):
        raise ValueError(f"collection_id {cid!r} is a reserved name")
    return cid


def index_key(collection_id: str, table_key: str) -> str:
    """The single physical-unit identity: lock sk, state sk, dedup table_name."""
    return f"{collection_id}:{table_key}"


def lancedb_location(collection_id: str, schema_location: str) -> str:
    """
    Per-collection LanceDB data location. schema_location is the
    SchemaDefinition-derived path ("_lancedb_index/{doc_type}/{model}/{dims}[/{lang}]");
    the collection_id is spliced in after the root prefix:
      _lancedb_index/{collection_id}/{doc_type}/{model}/{dims}[/{lang}]
    """
    root, _, rest = schema_location.partition("/")
    return f"{root}/{collection_id}/{rest}"


def s3v_index_name(collection_id: str, table_key: str) -> str:
    """
    Deterministic S3 Vectors index name for one (collection, table).
    Names cap at 63 chars, lowercase alnum + hyphen/dot — model IDs and
    language make raw concatenation overflow, so: prefix + short hash.
    The state record stores the exact name; the hash never needs reversing.
    """
    doc_type = table_key.split(":", 1)[0]
    h8       = hashlib.sha256(index_key(collection_id, table_key).encode()).hexdigest()[:8]
    # Truncate the PREFIX, never the hash — the hash is what disambiguates
    # same-doc_type schemas (different model/dims/language); cutting it would
    # collide two different tables onto one physical index.
    prefix   = f"{collection_id}--{doc_type}".lower().replace("_", "-")[:63 - 10]
    return f"{prefix}--{h8}"


def _plain(obj):
    """Decimal -> float recursively. Collection records are read via the
    boto3 resource (Decimal weights); decompose embeds them in its Lambda
    response, and json.dumps rejects Decimal."""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def get_collection(collection_id: str) -> dict:
    """Return the collection record (Decimals as floats), or {} if absent."""
    resp = _table().get_item(Key={"collection_id": collection_id})
    return _plain(resp.get("Item") or {})


def resolve_collection(collection_id: str | None) -> dict:
    """
    The single entry point states use: default the id, load the record,
    fail hard on an unknown collection (a typo'd collection silently
    indexing/searching nothing is worse than a crash).
    """
    cid    = (collection_id or "").strip() or DEFAULT_COLLECTION_ID
    record = get_collection(cid)
    if not record:
        raise ValueError(
            f"Unknown collection {cid!r} — create it first (create_collection) "
            f"or check the deployment seeded the '{DEFAULT_COLLECTION_ID}' collection")
    return record

