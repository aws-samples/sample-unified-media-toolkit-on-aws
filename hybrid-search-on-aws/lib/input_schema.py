"""
input_schema.py — environment variable constants.

Single source of truth for all env-var-driven configuration.
Imports nothing from this project — no circular import risk.
Each constant is read once at module load time.
"""
from __future__ import annotations

import os

# ── DynamoDB ───────────────────────────────────────────────────────────────────
INDEX_STATE_TABLE        = os.environ.get("INDEX_STATE_TABLE", "LanceDBIndexState")
DEDUP_TABLE              = os.environ.get("DEDUP_TABLE", "LanceDBDedupState")
SEARCH_HISTORY_TABLE     = os.environ.get("SEARCH_HISTORY_TABLE", "LanceDBSearchHistory")
COLLECTION_TABLE         = os.environ.get("COLLECTION_TABLE", "LanceDBCollections")
SEARCH_HISTORY_TTL_DAYS  = int(os.environ.get("SEARCH_HISTORY_TTL_DAYS", "7"))

# ── AWS ────────────────────────────────────────────────────────────────────────
AWS_REGION    = os.environ.get("AWS_REGION", "us-east-1")  # Lambda injects AWS_REGION
PROXY_BUCKET  = os.environ.get("PROXY_BUCKET", "")
# S3 Vectors bucket (one per deployment; vector INDEXES are created at runtime,
# one per index_key, for schemas with vector_store="s3vectors")
VECTOR_BUCKET = os.environ.get("VECTOR_BUCKET", "")

# ── Collections ────────────────────────────────────────────────────────────────
# Every physical unit (LanceDB table location, S3 Vectors index, lock, state
# record, dedup table_name) is scoped by index_key = "{collection_id}:{table_key}".
# There is NO un-scoped legacy namespace (README.md Design record §1.1).
DEFAULT_COLLECTION_ID = "default"

# ── Lock ───────────────────────────────────────────────────────────────────────
# One lock item per physical table: pk="lock", sk=index_key.
# Writers to DIFFERENT tables (across or within collections) never contend.
LOCK_TTL_SECS = 3600

# ── Rebuild ────────────────────────────────────────────────────────────────────
# Minimum number of docs added since last ANN rebuild before triggering a new one
REBUILD_THRESHOLD = 500

# ── Indexing ───────────────────────────────────────────────────────────────────
# Max inline contract documents per index_content submission (manifest mode for more)
MAX_INLINE_DOCUMENTS = 100
# Max JSONL manifest size — checked via head_object BEFORE download. 256 MB
# of contracts (~50k scene docs) is far beyond any single-asset submission
# and comfortably inside the 3 GB IndexerLambda; beyond it, fail with
# guidance instead of an OOM kill mid-parse.
MAX_MANIFEST_BYTES = 256 * 1024 * 1024
# S3 Vectors PutVectors/DeleteVectors hard cap per call (verified AWS limit)
S3V_BATCH_SIZE = 500
# Below this much remaining Lambda time, checkpoint and raise
# LambdaResourceTimeoutException instead of risking a Sandbox.Timedout
# (which leaves no checkpoint). Mirrors face-recognition-on-aws's constant
# of the same name/purpose.
REMAINING_TIME_BUFFER_MS = 90_000

# ── Search ─────────────────────────────────────────────────────────────────────
# Multiplier applied to k to get per-index candidate count for RRF merging
CANDIDATE_K_MULTIPLIER = 5
CANDIDATE_K_MIN        = 50
# Hard ceiling on any single ANN query's result count. The gateway clamps
# k<=50, so candidate_k tops out at 250 — 500 leaves 2x headroom for direct
# SFN callers while keeping a runaway k from becoming an expensive S3 Vectors
# query (cost scales with topK) or an oversized LanceDB scan.
CANDIDATE_K_CEILING    = 500
