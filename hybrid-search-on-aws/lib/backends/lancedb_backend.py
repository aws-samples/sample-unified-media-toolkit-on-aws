"""
lancedb_backend.py — LanceDB backend backed by S3.

Index layout is determined entirely by an operator-supplied schema file.
The schema file (a JSON document on S3) declares:
  - doc_type:        logical name for this index type (e.g. "scene", "utterance")
  - embedding_model: model that produced the vectors
  - dims:            vector dimension (positive int, top-level)
  - language:        ISO 639-1 code (required if any field has "fts": true or "fts": {...})
  - mappings:        OpenSearch-style field definitions
  - result_fields:   schema-specific fields to include in search results (base fields
                     doc_id, asset_id, start_ms, end_ms are always returned)

Physical DATA locations are collection-scoped (README.md Design record §1.1) —
LanceDBBackend splices the collection_id into the schema-derived location:
  _lancedb_index/{collection_id}/{doc_type}/{embedding_model}/{dims}[/{language}]
Schema DEFINITION files stay in the single shared _lancedb_index/schema/
directory — two collections pointing at the same schema file get identical
table shapes in separate physical locations.

Schema mapping changes are immutable per registered index (enforced by
_ensure_registered in StateIndexDocuments): a mapping/model/dims change
always requires a new table_key (e.g. a _v2 schema file with a different
doc_type or model) — never an in-place edit on a table with rows.

vector_store ("lancedb" default | "s3vectors") selects where the ANN vector
lives. FTS always stays in LanceDB. For s3vectors schemas WITH FTS fields,
the LanceDB row holds everything except ANN search happens against S3
Vectors; for ANN-only s3vectors schemas no LanceDB table exists at all.

Supported field types in mappings:
  keyword         → pa.string()        row converter: str or ""
  text            → pa.string()        row converter: str or ""
  long            → pa.int64()         row converter: int(v or default)
  integer         → pa.int32()         row converter: int(v or default)
  float           → pa.float32()       row converter: float(v or default)
  double          → pa.float64()       row converter: float(v or default)
  float32_vector  → pa.list_(pa.float32(), dims)
                                       row converter: cast + dim validate;
                                       only one per schema; dims from top-level field

Field flags:
  "required": true  — field must be present and non-empty; missing/empty raises ValueError
  "ann": true       — marks the vector column (exactly one required)
  "fts": true       — FTS with language-default tokenizer options
  "fts": {...}      — FTS with explicit option overrides (stem, remove_stop_words, etc.)

Fields with "fts" drive build_fts_index().
The single field with "ann": true is the vector column for build_ann_index().

New index types need only a schema file — no code change required.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

import pyarrow as pa
import lancedb

from lib.s3_utils import get_json, parse_s3_uri
from lib.input_schema import AWS_REGION, PROXY_BUCKET

log = logging.getLogger(__name__)

# ── ISO 639-1 → LanceDB language name ─────────────────────────────────────────
# LanceDB requires full English names for stemming/stop-word removal.
_ISO_TO_LANCEDB: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "no": "Norwegian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "ta": "Tamil",
    "tr": "Turkish",
}

_ARROW_SCALAR_TYPES: dict[str, pa.DataType] = {
    "keyword": pa.string(),
    "text":    pa.string(),
    "long":    pa.int64(),
    "integer": pa.int32(),
    "float":   pa.float32(),
    "double":  pa.float64(),
}

TABLE_NAME = "documents"

_BASE_RESULT_FIELDS = ["doc_id", "asset_id", "start_ms", "end_ms"]

# ── Safety helpers ─────────────────────────────────────────────────────────────

_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def validate_asset_id(asset_id: str | None) -> None:
    if asset_id is None:
        return
    bare = asset_id.split(":")[-1] if ":" in asset_id else asset_id
    if not _UUID_RE.match(bare):
        raise ValueError(f"Invalid asset_id: {asset_id!r}")


def _sanitise_query(text: str) -> str:
    return re.sub(r'["\\\']', "", text)


def _resolve_language(iso: str) -> str:
    lang = _ISO_TO_LANCEDB.get(iso.lower())
    if lang is None:
        raise ValueError(
            f"Unsupported language code {iso!r}. "
            f"Supported: {sorted(_ISO_TO_LANCEDB)}"
        )
    return lang


# ── Schema definition ──────────────────────────────────────────────────────────

class SchemaDefinition:
    """
    Parsed and validated representation of a schema JSON file.

    Required top-level fields: doc_type, embedding_model, dims, mappings.
    language is required if any field has "fts": true or "fts": {...}.
    location is derived from doc_type, embedding_model, dims, and language.
    Exactly one field must have "ann": true (the vector column).
    """

    def __init__(self, schema_file_uri: str, raw: dict) -> None:
        self.schema_file_uri = schema_file_uri

        self.doc_type        = (raw.get("doc_type") or "").strip()
        self.embedding_model = (raw.get("embedding_model") or "").strip()
        self.language_iso    = (raw.get("language") or "").strip() or None
        dims_raw             = raw.get("dims")
        mappings             = raw.get("mappings") or {}
        self.result_fields   = raw.get("result_fields") or []
        # Where the ANN vector lives: "lancedb" (default) or "s3vectors".
        # FTS always stays in LanceDB — S3 Vectors has no FTS capability.
        self.vector_store    = (raw.get("vector_store") or "lancedb").strip().lower()
        if self.vector_store not in ("lancedb", "s3vectors"):
            raise ValueError(
                f"Schema {schema_file_uri!r}: vector_store must be "
                f"'lancedb' or 's3vectors', got {self.vector_store!r}")

        if not self.doc_type:
            raise ValueError(f"Schema {schema_file_uri!r} missing 'doc_type'")
        if not self.embedding_model:
            raise ValueError(f"Schema {schema_file_uri!r} missing 'embedding_model'")
        if not dims_raw or not isinstance(dims_raw, int) or dims_raw <= 0:
            raise ValueError(f"Schema {schema_file_uri!r} missing or invalid 'dims' (positive int required)")
        if not mappings:
            raise ValueError(f"Schema {schema_file_uri!r} has empty 'mappings'")

        # fts_fields: list of (field_name, fts_options_dict) pairs
        self.fts_fields:  list[tuple[str, dict]] = []
        self.ann_field:   str | None             = None
        self.dim:         int                    = dims_raw
        self._required_fields: set[str]          = set()
        arrow_fields:     list[pa.Field]         = []

        for field_name, spec in mappings.items():
            ftype = spec.get("type", "")

            if ftype == "float32_vector":
                if self.ann_field is not None:
                    raise ValueError(
                        f"Schema {schema_file_uri!r} has multiple float32_vector fields — "
                        f"only one vector column is supported"
                    )
                if not spec.get("ann"):
                    raise ValueError(
                        f"Schema {schema_file_uri!r} field {field_name!r}: "
                        f"float32_vector field must have \"ann\": true"
                    )
                self.ann_field = field_name
                arrow_fields.append(pa.field(field_name, pa.list_(pa.float32(), self.dim)))

            elif ftype in _ARROW_SCALAR_TYPES:
                arrow_fields.append(pa.field(field_name, _ARROW_SCALAR_TYPES[ftype]))
                fts_val = spec.get("fts")
                if fts_val is True:
                    self.fts_fields.append((field_name, {}))
                elif isinstance(fts_val, dict):
                    self.fts_fields.append((field_name, fts_val))

            else:
                raise ValueError(
                    f"Schema {schema_file_uri!r} field {field_name!r}: "
                    f"unknown type {ftype!r}. Supported: "
                    f"{sorted(_ARROW_SCALAR_TYPES) + ['float32_vector']}"
                )

            if spec.get("required"):
                self._required_fields.add(field_name)

        if self.ann_field is None:
            raise ValueError(
                f"Schema {schema_file_uri!r}: no field with \"ann\": true found. "
                f"Exactly one float32_vector field must be declared."
            )
        if self.fts_fields and not self.language_iso:
            raise ValueError(
                f"Schema {schema_file_uri!r}: FTS fields {[f for f, _ in self.fts_fields]!r} "
                f"require 'language' to be set."
            )

        mapping_names = set(mappings.keys())
        bad = [f for f in self.result_fields if f not in mapping_names]
        if bad:
            raise ValueError(
                f"Schema {schema_file_uri!r}: result_fields references fields not in mappings: {bad!r}"
            )

        self.arrow_schema = pa.schema(arrow_fields)
        self._mappings    = mappings  # kept for row converter

        # Derive location from schema identity fields — no s3:// prefix, no trailing slash
        if self.language_iso:
            self.location = f"_lancedb_index/{self.doc_type}/{self.embedding_model}/{self.dim}/{self.language_iso}"
        else:
            self.location = f"_lancedb_index/{self.doc_type}/{self.embedding_model}/{self.dim}"

        # Resolve LanceDB language name now (fail fast on bad ISO code)
        self.lancedb_lang = _resolve_language(self.language_iso) if self.language_iso else None

    @property
    def table_key(self) -> str:
        """Stable DynamoDB key for this index."""
        if self.language_iso:
            return f"{self.doc_type}:{self.embedding_model}:{self.language_iso}"
        return f"{self.doc_type}:{self.embedding_model}"

    def to_row(self, doc: dict) -> dict:
        """
        Convert a raw document dict to a typed row dict matching arrow_schema.
        Raises ValueError on dim mismatch or missing required fields.
        """
        for field_name in self._required_fields:
            val = doc.get(field_name)
            if val is None or val == "" or val == []:
                raise ValueError(f"doc missing required field {field_name!r}: {doc.get('doc_id', '<no doc_id>')!r}")

        row = {}
        for field_name, spec in self._mappings.items():
            ftype = spec.get("type", "")
            raw   = doc.get(field_name)

            if ftype == "float32_vector":
                vec = [float(v) for v in (raw or [])]
                if len(vec) != self.dim:
                    raise ValueError(
                        f"doc_id={doc.get('doc_id')!r}: vector dim={len(vec)}, "
                        f"expected {self.dim}"
                    )
                row[field_name] = vec

            elif ftype in ("long", "integer"):
                default = int(spec.get("default", 0))
                row[field_name] = int(raw) if raw is not None and raw != "" else default

            elif ftype in ("float", "double"):
                default = float(spec.get("default", 0.0))
                row[field_name] = float(raw) if raw is not None and raw != "" else default

            else:  # keyword / text
                row[field_name] = str(raw) if raw is not None else ""

        return row

    @classmethod
    def from_s3(cls, schema_file_uri: str) -> "SchemaDefinition":
        """Load and parse a schema file from S3."""
        bucket, key = parse_s3_uri(schema_file_uri)
        raw = get_json(bucket, key)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Schema file {schema_file_uri!r} must be a JSON object, "
                f"got {type(raw).__name__!r}"
            )
        return cls(schema_file_uri, raw)


# ── Schema file cache (per Lambda container) ──────────────────────────────────
# Keyed by schema_file_uri. One S3 read per cold start per distinct schema file.

_schema_cache: dict[str, SchemaDefinition] = {}


def load_schema(schema_file_uri: str) -> SchemaDefinition:
    """Return cached SchemaDefinition, loading from S3 on first call."""
    if schema_file_uri not in _schema_cache:
        log.info("Loading schema from %s", schema_file_uri)
        _schema_cache[schema_file_uri] = SchemaDefinition.from_s3(schema_file_uri)
    return _schema_cache[schema_file_uri]



# ── Backend ────────────────────────────────────────────────────────────────────

class LanceDBBackend:
    """
    One backend instance = one LanceDB index described by a SchemaDefinition.

    The index S3 location, field layout, FTS configuration, and ANN column
    all come from the schema definition — nothing is hardcoded here.
    """

    def __init__(self, schema_def: SchemaDefinition, collection_id: str,
                 bucket: str = "") -> None:
        """
        collection_id is REQUIRED — every physical table is collection-scoped
        (README.md Design record §1.1): _lancedb_index/{collection_id}/{doc_type}/...
        There is no un-scoped namespace.
        """
        if not collection_id:
            raise ValueError(
                f"LanceDBBackend for {schema_def.table_key!r}: collection_id is required")
        self._schema_def    = schema_def
        self._collection_id = collection_id
        _bucket             = bucket or PROXY_BUCKET
        if not _bucket:
            raise ValueError(
                f"LanceDBBackend for {schema_def.table_key!r}: no bucket provided "
                "and PROXY_BUCKET env var is not set"
            )
        from lib.collections import lancedb_location
        self._s3_uri  = f"s3://{_bucket}/{lancedb_location(collection_id, schema_def.location)}/"
        self._db:    lancedb.DBConnection | None = None
        self._table: Any = None
        self._result_fields: list[str] = list(dict.fromkeys(
            _BASE_RESULT_FIELDS + schema_def.result_fields
        ))

    # ── Public accessors ───────────────────────────────────────────────────────

    @property
    def collection_id(self) -> str:
        return self._collection_id

    @property
    def index_key(self) -> str:
        """The physical-unit identity: '{collection_id}:{table_key}'."""
        return f"{self._collection_id}:{self._schema_def.table_key}"

    @property
    def table_key(self) -> str:
        return self._schema_def.table_key

    @property
    def doc_type(self) -> str:
        return self._schema_def.doc_type

    @property
    def embedding_model(self) -> str:
        return self._schema_def.embedding_model

    @property
    def language_iso(self) -> str | None:
        return self._schema_def.language_iso

    @property
    def dim(self) -> int:
        return self._schema_def.dim

    @property
    def has_fts(self) -> bool:
        return bool(self._schema_def.fts_fields)

    @property
    def schema_def(self) -> SchemaDefinition:
        return self._schema_def

    @property
    def is_loaded(self) -> bool:
        """True once load() or index() has successfully opened/created the table."""
        return self._table is not None

    # ── Connection ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self._db is None:
            self._db = lancedb.connect(
                self._s3_uri,
                storage_options={"region": AWS_REGION},
            )

    def is_ready(self) -> bool:
        """Return True if the table exists. Avoids second round-trip if already loaded."""
        if self._table is not None:
            return True
        self._connect()
        return TABLE_NAME in self._db.list_tables().tables

    def load(self) -> None:
        self._connect()
        if TABLE_NAME not in self._db.list_tables().tables:
            log.info("Table not found at %s — index is empty", self._s3_uri)
            self._table = None
            return
        self._table = self._db.open_table(TABLE_NAME)
        log.info("Loaded index %s — %d rows", self.table_key, self._table.count_rows())

    # ── Write ──────────────────────────────────────────────────────────────────

    def index(self, docs: list[dict]) -> int:
        """
        Append documents to the table.
        Creates the table on first call; appends on subsequent calls.
        Returns the number of rows appended.
        """
        if not docs:
            return 0

        self._connect()
        rows   = [self._schema_def.to_row(d) for d in docs]
        schema = self._schema_def.arrow_schema

        if self._table is not None:
            # Table already open — skip list_tables() S3 round-trip.
            self._table.add(rows)
            log.info("Appended %d rows to index %s", len(rows), self.table_key)
        elif TABLE_NAME not in self._db.list_tables().tables:
            self._table = self._db.create_table(TABLE_NAME, data=rows, schema=schema)
            log.info("Created index %s with %d rows", self.table_key, len(rows))
        else:
            self._table = self._db.open_table(TABLE_NAME)
            self._table.add(rows)
            log.info("Appended %d rows to index %s", len(rows), self.table_key)

        return len(rows)

    def build_fts_index(self) -> None:
        """Rebuild FTS index on all FTS fields. No-op if schema has no FTS fields."""
        fts_fields = self._schema_def.fts_fields
        if not fts_fields:
            return
        if self._table is None:
            raise RuntimeError(f"Call load() or index() before build_fts_index() [{self.table_key}]")
        lang = self._schema_def.lancedb_lang
        for field, overrides in fts_fields:
            opts = {
                "language":          lang,
                "stem":              True,
                "remove_stop_words": True,
                "lower_case":        True,
                "ascii_folding":     True,
            }
            opts.update(overrides)
            self._table.create_fts_index(field, replace=True, **opts)
        log.info(
            "FTS index rebuilt on %s fields=%s language=%s",
            self.table_key, [f for f, _ in fts_fields], lang,
        )

    def build_ann_index(self) -> None:
        """Rebuild IVF_PQ ANN index on the vector column."""
        if self._table is None:
            raise RuntimeError(f"Call load() or index() before build_ann_index() [{self.table_key}]")
        num_rows = self._table.count_rows()
        if num_rows < 256:
            log.info("Skipping ANN index — only %d rows (need ≥256) [%s]", num_rows, self.table_key)
            return
        dim = self._schema_def.dim
        if dim % 16 == 0:
            num_sub_vectors = dim // 16
        elif dim % 8 == 0:
            num_sub_vectors = dim // 8
        else:
            num_sub_vectors = 1
        # num_partitions ≈ sqrt(num_rows), clamped to [16, 512].
        # IVF_PQ recall degrades sharply when partitions >> sqrt(rows) — each
        # partition contains too few vectors for the quantizer to learn.
        num_partitions = max(16, min(512, round(math.sqrt(num_rows))))
        self._table.create_index(
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name=self._schema_def.ann_field,
            replace=True,
            index_type="IVF_PQ",
        )
        log.info(
            "ANN index rebuilt on %s (%d rows, partitions=%d, sub_vectors=%d)",
            self.table_key, num_rows, num_partitions, num_sub_vectors,
        )

    def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """Idempotency helper: remove rows with these doc_ids (no-op when the
        table doesn't exist yet). Used before re-appending on SFN retry."""
        if not doc_ids:
            return
        if self._table is None:
            self.load()
        if self._table is None:
            return
        quoted = ", ".join("'" + d.replace("'", "''") + "'" for d in doc_ids)
        self._table.delete(f"doc_id IN ({quoted})")

    def delete_by_asset_id(self, asset_id: str) -> None:
        if self._table is None:
            return
        safe_id = asset_id.replace("'", "''")
        self._table.delete(f"asset_id = '{safe_id}'")
        log.info("Deleted rows for asset_id=%s from %s", asset_id, self.table_key)

    def optimize(self) -> None:
        if self._table is None:
            raise RuntimeError(f"Call load() or index() before optimize() [{self.table_key}]")
        self._table.optimize()
        log.info("Optimized %s", self.table_key)

    def count_rows(self) -> int:
        if self._table is None:
            return 0
        return self._table.count_rows()

    # ── Search ─────────────────────────────────────────────────────────────────

    def search_ann(
        self,
        query_vector: list[float],
        k: int,
        asset_id: str | None = None,
        asset_ids: list[str] | None = None,
    ) -> list[dict]:
        if self._table is None:
            self.load()
        if self._table is None:
            return []
        validate_asset_id(asset_id)

        self._table.checkout_latest()

        q = self._table.search(query_vector).metric("cosine").limit(k)
        if asset_id or asset_ids:
            q = q.minimum_nprobes(20)
        q = self._apply_filters(q, asset_id, asset_ids=asset_ids, prefilter=True)
        return [self._to_result(r) for r in q.to_list()]

    def search_fts(
        self,
        query_text: str,
        field: str,
        k: int,
        asset_id: str | None = None,
        asset_ids: list[str] | None = None,
    ) -> list[dict]:
        fts_field_names = [f for f, _ in self._schema_def.fts_fields]
        if not fts_field_names:
            raise RuntimeError(f"FTS not available for doc_type={self.doc_type!r}")
        if field not in fts_field_names:
            raise ValueError(f"Field {field!r} is not an FTS field for doc_type={self.doc_type!r}")
        if self._table is None:
            self.load()
        if self._table is None:
            return []
        validate_asset_id(asset_id)

        self._table.checkout_latest()

        safe = _sanitise_query(query_text).strip()
        if not safe:
            return []
        q = self._table.search(safe, query_type="fts", fts_columns=field).limit(k)
        q = self._apply_filters(q, asset_id, asset_ids=asset_ids, prefilter=True)
        return [self._to_result(r) for r in q.to_list()]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _apply_filters(self, query, asset_id, asset_ids=None, prefilter=True):
        if asset_ids:
            ids_quoted = ", ".join(f"'{a.replace(chr(39), chr(39)+chr(39))}'" for a in asset_ids)
            query = query.where(f"asset_id IN ({ids_quoted})", prefilter=prefilter)
        elif asset_id:
            safe_id = asset_id.replace("'", "''")
            query = query.where(f"asset_id = '{safe_id}'", prefilter=prefilter)
        return query

    def _to_result(self, row: dict) -> dict:
        result = {f: row.get(f) for f in self._result_fields}
        if row.get("_relevance_score") is not None:
            result["score"] = round(float(row["_relevance_score"]), 6)
        elif row.get("_distance") is not None:
            result["score"] = round(min(1.0, max(0.0, 1.0 - float(row["_distance"]))), 6)
        else:
            result["score"] = round(float(row.get("_score", 0.0)), 6)
        return result

    def info(self) -> dict:
        return {
            "doc_type":        self.doc_type,
            "embedding_model": self.embedding_model,
            "language":        self.language_iso,
            "s3_uri":          self._s3_uri,
            "dim":             self.dim,
            "row_count":       self.count_rows(),
        }
