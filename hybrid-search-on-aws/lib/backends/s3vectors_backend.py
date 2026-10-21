"""
s3vectors_backend.py — ANN vector storage on Amazon S3 Vectors.

Used when a schema declares vector_store: "s3vectors". Holds ONLY the ANN
vector + minimal filterable metadata (asset_id) + result_fields as
non-filterable metadata. FTS stays in LanceDB (S3 Vectors has no FTS).

One vector index per index_key ("{collection_id}:{table_key}"), created at
RUNTIME on first write (CreateIndex, first-write-wins — "already exists" is
success). Deploy-time CFN declaration is impossible: per-collection indexes
are unbounded and agent-created. The ONE vector bucket is a deploy-time CFN
resource (AWS::S3Vectors::VectorBucket).

Verified service facts (README.md Design record §4):
  - PutVectors on an existing key OVERWRITES — true upsert, retry-safe.
  - 500 vectors max per PutVectors/DeleteVectors call (S3V_BATCH_SIZE).
  - Metadata: 40 KB/vector total, 2 KB filterable, ≤10 non-filterable keys
    declared at CreateIndex, ≤50 keys total.
  - QueryVectors: topK ≤ 10,000, single response (no pagination in the
    API output shape — verified against the service model).
"""
from __future__ import annotations

import logging

import boto3

from lib.input_schema import (AWS_REGION, CANDIDATE_K_CEILING,
                              S3V_BATCH_SIZE, VECTOR_BUCKET)

log = logging.getLogger(__name__)

_s3v = None


def _client():
    global _s3v
    if _s3v is None:
        _s3v = boto3.client("s3vectors", region_name=AWS_REGION)
    return _s3v


class S3VectorsBackend:
    """
    ANN operations for one (collection, table) pair against one S3 Vectors
    index. Mirrors the vector-path subset of LanceDBBackend's interface:
    ensure_index / put / query / delete_keys.
    """

    def __init__(self, schema_def, collection_id: str, index_name: str) -> None:
        if not VECTOR_BUCKET:
            raise ValueError("S3VectorsBackend: VECTOR_BUCKET env var is not set")
        self._schema_def = schema_def
        self._collection_id = collection_id
        self._index_name = index_name

    @property
    def index_name(self) -> str:
        return self._index_name

    def ensure_index(self) -> None:
        """
        CreateIndex, first-write-wins. Dimension/metric are fixed by the
        service for the index's lifetime — the same immutability
        _ensure_registered enforces for LanceDB tables.
        Non-filterable metadata keys = the schema's result_fields (they ride
        along for result display, never for filtering; asset_id stays
        filterable by omission from this list).
        """
        non_filterable = [f for f in self._schema_def.result_fields
                          if f != "asset_id"][:10]
        try:
            kwargs: dict = {
                "vectorBucketName": VECTOR_BUCKET,
                "indexName":        self._index_name,
                "dataType":         "float32",
                "dimension":        self._schema_def.dim,
                "distanceMetric":   "cosine",
            }
            if non_filterable:
                kwargs["metadataConfiguration"] = {
                    "nonFilterableMetadataKeys": non_filterable}
            _client().create_index(**kwargs)
            log.info("Created S3 Vectors index %s (dim=%d)",
                     self._index_name, self._schema_def.dim)
        except _client().exceptions.ConflictException:
            # First-write-wins is only safe if the winner used OUR dimension.
            # An in-place dims edit on the schema JSON (same schema_file URI,
            # so _ensure_registered passes) would otherwise silently bind this
            # writer to a mismatched index and fail per-batch at put() —
            # after LanceDB rows were already appended.
            try:
                existing = _client().get_index(
                    vectorBucketName=VECTOR_BUCKET,
                    indexName=self._index_name)["index"]
            except _client().exceptions.NotFoundException:
                # Conflict then NotFound = a concurrent delete (admin sweep)
                # raced us between the two calls. Take the create path again;
                # a second conflict here is genuinely unexpected — let it
                # propagate.
                _client().create_index(**kwargs)
                log.info("Created S3 Vectors index %s after delete race "
                         "(dim=%d)", self._index_name, self._schema_def.dim)
                return
            if existing["dimension"] != self._schema_def.dim:
                raise ValueError(
                    f"S3 Vectors index {self._index_name} exists with "
                    f"dimension {existing['dimension']}, schema wants "
                    f"{self._schema_def.dim} — dims are immutable; changing "
                    f"them requires a new table_key/schema file, never an "
                    f"in-place edit")
            log.info("S3 Vectors index %s already exists (dim=%d)",
                     self._index_name, existing["dimension"])

    def put(self, rows: list[dict]) -> int:
        """
        Upsert vectors keyed by doc_id. Metadata: asset_id (filterable) +
        start_ms/end_ms + result_fields (non-filterable per ensure_index).
        Chunked at the 500-vector service cap.
        """
        ann_field = self._schema_def.ann_field
        meta_fields = ["asset_id", "start_ms", "end_ms"] + [
            f for f in self._schema_def.result_fields if f != "asset_id"]
        vectors = []
        for r in rows:
            metadata = {}
            for k in meta_fields:
                v = r.get(k)
                if v is None:
                    continue
                if not isinstance(v, (str, int, float, bool)):
                    raise ValueError(
                        f"doc {r.get('doc_id')}: metadata field {k!r} is "
                        f"{type(v).__name__} — S3 Vectors metadata supports "
                        f"only string/number/boolean")
                metadata[k] = v
            vectors.append({
                "key":      r["doc_id"],
                "data":     {"float32": [float(v) for v in r[ann_field]]},
                "metadata": metadata,
            })
        for i in range(0, len(vectors), S3V_BATCH_SIZE):
            _client().put_vectors(
                vectorBucketName=VECTOR_BUCKET,
                indexName=self._index_name,
                vectors=vectors[i:i + S3V_BATCH_SIZE],
            )
        log.info("Put %d vectors to S3V index %s", len(vectors), self._index_name)
        return len(vectors)

    def query(self, query_vector: list[float], k: int,
              asset_ids: list[str] | None = None) -> list[dict]:
        """
        ANN query. Returns rows shaped like LanceDBBackend results:
        {doc_id, asset_id, ..., score} with score = 1 - cosine distance.
        """
        kwargs: dict = {
            "vectorBucketName": VECTOR_BUCKET,
            "indexName":        self._index_name,
            "queryVector":      {"float32": [float(v) for v in query_vector]},
            "topK":             min(k, CANDIDATE_K_CEILING),
            "returnMetadata":   True,
            "returnDistance":   True,
        }
        if asset_ids:
            kwargs["filter"] = ({"asset_id": asset_ids[0]} if len(asset_ids) == 1
                                else {"asset_id": {"$in": asset_ids}})
        try:
            resp = _client().query_vectors(**kwargs)
        except _client().exceptions.NotFoundException:
            log.info("S3V index %s does not exist yet — empty results",
                     self._index_name)
            return []
        results = []
        for v in resp.get("vectors", []):
            row = dict(v.get("metadata") or {})
            row["doc_id"] = v["key"]
            dist = float(v.get("distance", 1.0))
            row["score"] = round(min(1.0, max(0.0, 1.0 - dist)), 6)
            results.append(row)
        return results

    def delete_keys(self, doc_ids: list[str]) -> None:
        """Delete vectors by key, chunked at the 500-key service cap."""
        for i in range(0, len(doc_ids), S3V_BATCH_SIZE):
            _client().delete_vectors(
                vectorBucketName=VECTOR_BUCKET,
                indexName=self._index_name,
                keys=doc_ids[i:i + S3V_BATCH_SIZE],
            )
        log.info("Deleted %d vectors from S3V index %s",
                 len(doc_ids), self._index_name)

    def delete_index(self) -> None:
        """Remove the whole index (collection deletion sweep)."""
        try:
            _client().delete_index(vectorBucketName=VECTOR_BUCKET,
                                   indexName=self._index_name)
            log.info("Deleted S3V index %s", self._index_name)
        except _client().exceptions.NotFoundException:
            pass
