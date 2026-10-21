# hybrid-search-on-aws

Hybrid (semantic + keyword) search over media metadata, exposed to AI
agents as MCP tools through the toolkit's gateway. Indexing is
**contract-driven**: callers declare WHAT they hand over (`kind:
text|image`) and the backend does all embedding, full-text indexing,
and entity extraction. Storage is LanceDB on S3 (FTS + default ANN)
with optional Amazon S3 Vectors per schema (`vector_store:
"s3vectors"`). **Collections** (news, sport, …) give per-tenant
isolation inside one stack: per-collection physical tables/indexes
under `_lancedb_index/{collection_id}/...`, shared compute (see
[Design record §1](#1-collections)).

A search runs as a three-state Step Functions pipeline: an SLM
decomposes the query (operators: `@Title`, `"quoted"`, `says:"…"`,
`#scope`), every active index runs ANN + FTS sub-queries in parallel,
and results merge via score-weighted reciprocal-rank fusion before a
cross-index rerank.

Five index types ship by default, one per media granularity, all
schema-driven from `schema/*.json` — a new index type is a new schema
file, no code change:

| scope | ANN | FTS fields |
|---|---|---|
| `shot` | vector | — |
| `scene` | vector | `text`, `named_entities` |
| `segment` | vector | `text` |
| `utterance` | vector | `text` |
| `topic` | vector | `text`, `named_entities` |

## Gateway tools

| Target | Tools |
|---|---|
| `hybrid-search` | `index_content`, `search_content`, `create_collection`, `list_collections`, `get_job_status`, `describe_tool` |
| `hybrid-search-admin` | `configure_search_backend`, `delete_collection`, `describe_tool` — separate Lambda and IAM role; attach only for trusted admin agents |

Tool contracts in [Design record §5](#5-gateway-tools). The admin/normal
split is an IAM boundary, never a payload flag (decision D3).

### describe_tool output contract

The gateway's `describe_tool` returns `{version,
index_state_machine_arn, search_state_machine_arn, description,
output_contract}`. `output_contract` is
[gateway_tool/output_contract.json](gateway_tool/output_contract.json),
parsed at cold start and served verbatim: the field-level contract for
this target's two **inline** result shapes — `search-results` (the
per-document fields `search_content` returns: the invariant scoring
fields below plus the schema-dependent `result_fields` payload) and
`index-status` (the `prepared`/`skipped`/`indexed` counts
`get_job_status` returns for an `index_content` job). Both entries are
`kind: "inline"` because this target returns results inline and writes
no consumer-facing output files — there is no `outputs[]` list. Static
per `TOOL_VERSION`: the file ships in the gateway zip and
`build-deploy-s3-dist.sh` fails the build if it is missing, so version
and contract cannot drift; a caller caches it once per session.
Anything not listed in it (`_lancedb_index/` storage layout, `_search/`
transients like `vectors.json`, row-group objects) is internal and may
change without notice. Only the NORMAL target serves it —
`admin_function.py` is untouched.

## Repository layout

```
hybrid-search-on-aws/
├── lambda_function.py              Root dispatcher (routes by "operation" field)
├── definitions_search.json         Search SFN definition (readability copy of the inline DefinitionString in cfnstack.yaml)
├── definitions_index.json          Index SFN definition (Prepare → per-table Map)
├── definitions_rebuild.json        Rebuild SFN definition
├── cfnstack.yaml                   CloudFormation stack (CodeBuild, ECR, Lambda, SFN, DDB, EventBridge)
├── Dockerfile                      Container image (Python 3.13, LanceDB, PyArrow)
│
├── custom_resource/
│   └── lambda_function.py          CopySchema custom resource — seeds schema/prompt/workflow
│                                   config to S3 + the "default" collection record
│
├── gateway_tool/                   AgentCore Gateway MCP tools (flat zip, NO lib/)
│   ├── lambda_function.py          Normal target (six tools above)
│   ├── admin_function.py           Admin target — configure_search_backend, delete_collection
│   ├── common.py                   Shared: error/resolve_tool/validation; MIRRORS
│   │                               lib/collections id-rule + s3v_index_name (keep in sync)
│   └── output_contract.json        Field-level contract for the two inline result shapes
│
├── schema/                         Five default index schemas (shot/scene/segment/utterance/topic)
├── workflow_config/
│   └── hybrid-search.json          Seeded to s3://{ProxyBucket}/_workflowconfig/ on Create
│                                   only (skip if exists — operator controls retrieval settings)
├── prompt_config/
│   └── default.md                  Hybrid-search SLM prompt — seeded to
│                                   s3://{ProxyBucket}/_promptconfig/baseline/hybrid-search/
│                                   on every deploy (always overwritten)
│
├── lib/
│   ├── input_schema.py             Env-var constants + limits (MAX_INLINE_DOCUMENTS,
│   │                               CANDIDATE_K_CEILING, S3V_BATCH_SIZE, …)
│   ├── ddb_utils.py / s3_utils.py  DynamoDB and S3 helpers
│   ├── lock_utils.py               Per-table distributed lock (acquire/release_if_owner/
│   │                               store_pending/pop_next_pending/requeue_pending/
│   │                               release_all_for_execution)
│   ├── index_state.py              DDB state helpers (register_index, asset metadata,
│   │                               dedup, search history)
│   ├── collections.py              Collection registry READ path (resolve_collection,
│   │                               index_key, lancedb_location, s3v_index_name)
│   ├── contracts.py                Content contract validation + doc_id derivation
│   ├── custom_exceptions.py        LambdaResourceTimeoutException (checkpoint/resume
│   │                               near the Lambda deadline)
│   ├── bedrock_utils.py            Bedrock client — call_slm_with_fallback, embed /
│   │                               embed_image / embed_document (dispatches to providers)
│   ├── providers/                  cohere_embed_v4 / titan_multimodal_embed / nova_multimodal_embed
│   └── backends/                   lancedb_backend (FTS + default ANN), s3vectors_backend
│
├── states/                         One Lambda state per folder:
│   ├── decompose_search_request/     SLM parse, embed, build query items
│   ├── search_index/                 ANN + FTS sub-queries, per-index RRF merge
│   ├── rerank_results/               cross-index flatten, normalize, top-k
│   ├── prepare_documents/            contracts → typed row groups (lock-free)
│   ├── acquire_lock/                 get one table's lock or queue the waiter
│   ├── index_documents/              write one row group (LanceDB / S3 Vectors)
│   ├── get_indices_to_rebuild/       rebuild candidates per index_key
│   ├── rebuild_index/                ANN rebuild + optimize
│   ├── release_lock/                 drop lock, fire DDB stream
│   ├── handle_lock_release/          DDB stream handler, resumes waiters
│   └── handle_execution_failure/     EventBridge; releases dead executions' locks
│
└── test/
    ├── test_units.py               Mocked-boto3 unit tests (contracts, locks, naming,
    │                               backends, template assertions)
    └── harness/                    Local tooling: prepare_test_payload.py,
                                    test_decompose_search_request.py
```

ASL lives inline in `cfnstack.yaml` with readability copies in
`definitions_*.json`; builds fail on drift — change both or neither.

## Step Functions workflows

### SearchSFN — `{ResourcePrefix}-hybrid-search-search`

```
DecomposeSearchRequest
  → SFN INLINE Map (per active index, MaxConcurrency=0)
      → SearchIndex
  → RerankResults
```

Vectors are written to S3 by DecomposeSearchRequest (avoids the SFN
256 KB Map payload limit), read per-scope by each SearchIndex Map
iteration, deleted by RerankResults.

### IndexSFN — `{ResourcePrefix}-hybrid-search-indexer`

```
Set input defaults (Pass, JsonMerge)
  → Prepare documents (StatePrepareDocuments — LOCK-FREE: contracts in,
      typed row groups out; one S3 object per physical table; dedup BEFORE
      any Bedrock call; embedding + entity extraction here)
  → SFN INLINE Map (per physical table, MaxConcurrency=8)
      → Acquire table lock (waitForTaskToken, sk=index_key)
      → Index table (LanceDB append / S3V put + FTS rebuild)
      → Release table lock          (Catch → release → Fail)
```

Input: `{collection_id?, documents[] | documents_s3uri (JSONL),
asset_metadata?, workflow_config_uri?}` — exactly one of `documents |
documents_s3uri`; inline capped at 100 (`MAX_INLINE_DOCUMENTS`, stays
far under the 256 KB SFN input limit).

**Prepare documents checkpoints and resumes near the Lambda deadline**:
below `REMAINING_TIME_BUFFER_MS` (90 s) of remaining time, or on any
Bedrock/S3 `ClientError`, it writes its completed rows for the current
doc_type to the row-group S3 key with a `partial` marker and raises
`LambdaResourceTimeoutException` (`lib/custom_exceptions.py`). The ASL
`Retry` on that error (and on `Sandbox.Timedout`/`Lambda.Unknown` as a
belt-and-suspenders backstop, and on Bedrock transient codes)
re-invokes the state; re-entry reuses complete doc_type groups whole
and resumes a partial group after its last checkpointed document — no
re-embedding, no double Bedrock spend. A large asset (thousands of
contracts) legitimately shows several "Prepare documents" entries in
the SFN execution history; this is expected, not a stuck execution.

### RebuildSFN — `{ResourcePrefix}-hybrid-search-rebuild`

```
Get indices to rebuild (per-index_key candidates; skips ANN-only s3vectors)
  → SFN INLINE Map (per physical table, MaxConcurrency=8)
      → Acquire table lock (LockHeldError retry × 10 @ 60 s)
      → Rebuild table (optimize + IVF_PQ if vector_store=lancedb)
      → Release table lock          (Catch → release → Fail)
```

Rebuilds across collections/tables run CONCURRENTLY; only
same-physical-table operations serialize (per-table locks).

## Lambda functions

| Function | Memory | Timeout | Purpose |
|---|---|---|---|
| `SearchLambda` | 1769 MB (1 vCPU) | 2 min | ALL search states: Decompose + SearchIndex + Rerank |
| `IndexerLambda` | 3008 MB | 15 min | StatePrepareDocuments (checkpoints/resumes near the deadline) + StateAcquireLock + StateIndexDocuments + StateReleaseLock |
| `RebuildLambda` | 3008 MB | 15 min | StateGetIndicesToRebuild + StateAcquireLock + StateRebuildIndex + StateReleaseLock |
| `StreamHandlerLambda` | 256 MB | 2 min | HandleLockRelease — DDB stream trigger |
| `HandleExecutionFailureLambda` | 256 MB | 1 min | EventBridge failure handler — release_all_for_execution |
| `GatewayToolLambda` / admin variant | 256 MB | 90 s | MCP tools (flat zip, python3.13) |

DDB stream filter on `StreamHandlerLambda`: REMOVE events on
`pk="lock"` only. SearchLambda is set to 1769 MB (1 full vCPU) to avoid
thread contention between the parallel ANN and FTS sub-queries;
provisioned concurrency is the highest-leverage latency improvement if
sub-2s search is required.

## Schema files

Schema files live in `schema/` and are seeded to S3 at deploy time by
the `CopySchema` custom resource. Runtime path:
`s3://{PROXY_BUCKET}/_lancedb_index/schema/<filename>.json`.

| Field | Required | Description |
|---|---|---|
| `doc_type` | yes | Index identity (`shot`, `scene`, `segment`, `utterance`, `topic`) |
| `embedding_model` | yes | Model ID used to produce vectors (e.g. `cohere.embed-v4:0`) |
| `dims` | yes | Vector dimension (positive int) — passed to `embed()` to control output size |
| `language` | if FTS | ISO 639-1 code; required if any field has `"fts"` |
| `mappings` | yes | OpenSearch-style field definitions |
| `result_fields` | yes | Schema-specific fields returned in search results (base fields `doc_id`, `asset_id`, `start_ms`, `end_ms` always returned) |

`location` is derived automatically:
`_lancedb_index/{doc_type}/{embedding_model}/{dims}[/{language}]` (the
collection id is spliced in by the backend — see Design record §1.1).

Supported mapping types: `keyword`, `text`, `long`, `integer`, `float`,
`double`, `float32_vector`. Exactly one field must have `"ann": true`
(the vector column). FTS fields: `"fts": true` uses the
language-default tokenizer; `"fts": {"stem": false,
"remove_stop_words": false}` overrides options (use for
`named_entities` — proper nouns must not be stemmed).

Schemas are design-time artifacts: authored by humans, seeded at
deploy, immutable per physical table once a document lands — never
edited via tools (decision D1).

## Embedding providers

`lib/bedrock_utils.embed(model_id, text, dims)` dispatches by model_id
substring to the matching provider in `lib/providers/`. Each provider
owns `build_request` and `parse_response`.

| Provider module | Matched by | Supported dims |
|---|---|---|
| `cohere_embed_v4.py` | `cohere.embed` | 256, 512, 1024, 1536 (via `output_dimension`) |
| `titan_multimodal_embed.py` | `titan-embed-image` | 256, 384, 1024 (via `embeddingConfig`) |
| `nova_multimodal_embed.py` | `nova-2-multimodal-embeddings` | 256, 384, 1024, 3072 (via `embeddingDimension`) |

`dims` from the schema is always passed through so the returned vector
matches the index dimension. To add a new embedding model: create a
provider module, add an import + `if` branch in
`_resolve_embed_provider`. Nova Multimodal Embeddings is only available
in `us-east-1`; the others are available in `eu-west-1`.

**Profile-only models invoke through an inference profile.** Some
Bedrock embedding models refuse on-demand invocation ("Retry your
request with the ID or ARN of an inference profile").
`bedrock_utils.invocation_id()` maps the LOGICAL model id to
`{prefix}.{model_id}` for the models named in
`EMBED_PROFILE_ONLY_MODELS` (default `cohere.embed-v4:0`), at the three
`invoke_model` sites ONLY. The prefix defaults to **`global`** (one
invocation id in every region, maximum routing headroom);
`EMBED_INFERENCE_PROFILE_PREFIX` overrides — a deployment that must
keep embedding in-geography for data residency sets its geo
(`us`/`eu`/`apac`; geo profiles only work from their own geography).
The schema keeps the bare id everywhere else on purpose: it names the
vector space, is baked into every `_lancedb_index/.../{model}/{dims}`
table path, keys the usage rows, and the profile routes to the SAME
model — putting the profile id in a schema would orphan every existing
index for zero benefit. The IAM policy already grants
`inference-profile/*.cohere.embed-*` alongside the wildcard-region
foundation-model ARNs a cross-region profile needs.

## Search pipeline

### StateDecomposeSearchRequest

Two mutually exclusive modes:

**Text search** (`search_term` is set):
1. Loads workflow config from S3 (`workflow_config_uri`)
2. Calls Bedrock SLM (`primaryModel` → fallback `secondaryModel`) with the hybrid-search prompt
3. Parses SLM XML output: `retrieval_paths`, `entities`, `clean_query`, operators
4. Derives `subquery_spec` dynamically from schema field annotations (`ann: true` → `{scope}_ann`, `fts` → `{scope}_{field}_fts`)
5. One `embed()` call per unique `(model_id, dims)` combination — vectors keyed by scope
6. Writes vectors to `s3://{PROXY_BUCKET}/_search/{request_id}/vectors.json`
7. Builds one query item per active scope; writes search history

**Image search** (`search_image_s3uri` is set, `search_term` empty):
1. Skips the SLM entirely
2. Uses `imageRetrievalPaths` from workflow config (instead of `retrievalPaths`)
3. Downloads the image from S3 and calls `embed_image()` per unique `(model_id, dims)` combination
4. Builds ANN-only query items (no FTS — no text to search)
5. Writes vectors and search history as normal

For Cohere: image queries use `{"images":
["data:image/jpeg;base64,<b64>"]}` — the data URI prefix is required.
The `images` array format is distinct from the mixture mode used when
indexing shots; both produce vectors in the same Cohere embedding
space.

The workflow config (`hybrid-search.json`) controls ENGINE settings:
`primaryModel`/`secondaryModel` (full Bedrock model IDs), `promptS3Key`,
and the seed values for the `default` collection. TENANT settings
(active scopes, schema_map, retrieval weights) come from the collection
record (Design record §1.3). `subquerySpec` is in neither — it is
derived from schema field annotations at runtime.

**A user-typed scope restriction guarantees the scope is searched**
(`_backfill_scope_weights`). The SLM's `retrieval_paths` choose what to
emphasize, but a `#scope` token must not be vetoed by them — the two
used to intersect to a SILENT empty result: a dialogue phrase tagged
`#utterance` but classified visual gave every utterance sub-query
weight 0, so the one index the filter kept had no sub-queries and
total=0 with empty `skipped_indices`. Now the user-named scopes' zero
weights are backfilled by max-pooling over EVERY configured path (the
collection's own weights, never an invented constant). The SLM's own
`index_scopes` still only NARROW the fan-out and never arm the backfill
— a hallucinated scope list from a small model must not neuter the path
classification (they are lowercased, so a case-variant restricts
instead of matching nothing).

Reporting: on a RESTRICTED query, every scope that cannot run is
returned in `dropped_scopes` — `unknown_scope` (name matches no schema,
e.g. a typo, caught BEFORE the active_scopes intersection would
silently eat it), `no_subqueries` (no configured weights anywhere),
`no_state_record` (index never registered) — using the same
`{index_scope, skip_reason}` vocabulary as StateSearchIndex's Map-level
skips, and StateRerankResults folds both into `skipped_indices`, so "no
matches" and "never searched" are distinguishable. Image search reports
`inactive_scope` for scopes named in `imageRetrievalPaths` that the
collection no longer activates (admin tooling validates the two
settings independently, so they can disagree). On an UNRESTRICTED
query, builder drops are NOT reported — a scope the classifier
de-emphasized to zero is normal, not "never searched" noise.

Unchanged on purpose: plain unquoted text never feeds text-FTS (only
`says:"…"` / `"quoted"` do), so a backfilled scope with no quote runs
ANN-only.

**The `@Title` operator is greedy by design** (titles are multi-word):
put `@Title` last, or terminate it with another operator. A title-first
query swallows trailing free text into the title lookup, which then
matches nothing, and an unmatched filter silently drops (search runs
cross-asset). The agent prompt documents the construction rule and a
detection tip (cross-asset asset_ids after a filtered search = the
filter did not bind). Resolution: `LanceDBIndexStateTable` GSI
`title-index`, `pk="asset"`, `begins_with(title,
normalized(asset_filter))` (lowercase, strip); no match → no filter. A
louder longest-prefix resolution ladder was built, reviewed, and
deliberately reverted in favor of the documented contract.

### StateSearchIndex (Map fan-out per index)

- Loads the ANN vector for its scope from S3 (`vectors_s3_key`)
- Runs ANN and FTS sub-queries in parallel via `ThreadPoolExecutor`
- **Score-weighted RRF** within index: `contrib = sub_weight ×
  (norm_score / (RRF_K + rank))` where `norm_score = score / max_score`
  normalizes ANN cosine and FTS relevance to the same scale
- Returns per-doc `rrf_score`, `match_sources` dict (sub-query →
  contribution), and `highlights`
- FTS results carry `highlights`: `{field: text_with_<em>tokens</em>}`
  (case-insensitive regex); if a doc appears in both ANN and FTS,
  highlights from FTS are merged into the stored doc
- Graceful miss: an index with no state record / nothing indexed
  returns `{results: [], skipped: True}`; the reranker folds these into
  `skipped_indices`

### StateRerankResults

- Flattens all per-index results (no cross-index dedup needed — doc IDs
  are structurally unique per index type)
- `final_score = index_weight × rrf_score`, then normalized to `[0, 1]`
  (top result = 1.0)
- Batch-reads asset metadata from DDB for top-k docs only
  (`batch_get_asset_metadata`)
- Deletes `vectors.json` from S3 on both normal and empty-result paths

### Scoring fields on each result doc

| Field | Set by | Meaning |
|---|---|---|
| `score` | `lancedb_backend` | Raw ANN cosine similarity or FTS relevance |
| `rrf_score` | `search_index` | Score-weighted within-index RRF aggregate |
| `match_sources` | `search_index` | `{"ann:vector": 0.008, "fts:named_entities": 0.005}` — per sub-query contribution |
| `highlights` | `search_index` | `{"text": "...Jane <em>Doe</em>..."}` — FTS only |
| `source_index` | `rerank_results` | Index scope that surfaced the doc (`scene`, `shot`, etc.) |
| `final_score` | `rerank_results` | Normalized [0,1] cross-index score; sort key for the final list |

### Retrieval weights

Decompose maps the SLM's `retrieval_paths` classification to per-sub-query
weights from the collection record — the SLM never assigns floats. Shape
convention (the seeded defaults):

```python
# Base weights per retrieval path; contributions from all active paths
# are summed, then normalised to 1.0.
{
    "visual":       {"shot_ann": 0.7, "scene_ann": 0.5, "segment_ann": 0.3},
    "entity_fts":   {"scene_fts": 0.9, "topic_fts": 0.8, "segment_fts": 0.4},
    "dialogue":     {"utterance_fts": 0.8, "utterance_ann": 0.4},
    "text_overlay": {"scene_fts": 0.7, "segment_fts": 0.6},
}
# index_weight per scope = max of active sub-query weights for that scope
```

### Search history

Table `LanceDBSearchHistoryTable`: hash key `request_id` =
`$$.Execution.Name` (SFN execution name, already unique), sort key
`sk="search"`. Fields: `search_term`, `workflow_config_uri`,
`decomposition` (sans vectors), `search_results`, `k`, `created_at`,
`duration_ms`, `ttl` (`created_at + SEARCH_HISTORY_TTL_DAYS × 86400`).
GSI `created_at-index` for time-range queries. Decompose writes the
initial record; the reranker appends `search_results` + `duration_ms`.
`candidate_k = max(50, k × 5)` is computed in Decompose.

## DynamoDB tables

### LanceDBIndexState — state, locks, pending queues, asset metadata

Everything is keyed by `index_key = "{collection_id}:{table_key}"` —
the single physical-unit identity (lock, state record, dedup
table_name, S3 location derivation, S3 Vectors index name).

| pk | sk | Contents |
|---|---|---|
| `lock` | `{index_key}` | ONE lock per physical table. Writers to different tables never contend. TTL = 3600 s. |
| `pending` | `{index_key}#{sha256(task_token)}` | Per-lock waiter queue. FIFO via `created_at` within a lock. TTL = 24 h. |
| `state` | `{index_key}` | Per-index state: `schema_file`, `pending_docs_since_rebuild`, `rebuild_in_progress`, timestamps. |
| `asset` | `{collection_id}:{asset_id}` | Per-asset metadata: `title`, `genres` (CSV), `content_category`, `casts` (CSV). |

`table_key` format: `{doc_type}:{embedding_model}:{iso_language}` (FTS)
or `{doc_type}:{embedding_model}` (non-FTS).

### LanceDBCollections — collection registry

`pk = collection_id` (hash only). Per collection: `name`,
`active_scopes` (doc_types), `schema_map` (doc_type → schema file key;
definitions stay in the shared `_lancedb_index/schema/` directory —
only DATA locations are per-collection), `retrieval_paths`,
`image_retrieval_paths`, `weights_locked` (weight-freeze: set on any
admin weight edit; normal callers may still ADD scopes, never edit
weights). The `default` collection is seeded at deploy by the
CopySchema custom resource from `workflow_config/hybrid-search.json`
(skip-if-exists). Callers that omit `collection_id` resolve to
`default`.

### LanceDBDedupState — per-document dedup records

A separate table to avoid hot-partition throttling at scale.

| doc_id (hash key) | Contents |
|---|---|
| `{doc_id}` | `{ asset_id, table_name, indexed_at, ttl }` |

GSI `asset_id-table_name-index`: `asset_id` (hash) + `table_name`
(range) — used by `delete_docs_for_asset_and_table` on refresh.
KEYS_ONLY projection. `doc_id` is backend-derived (decision D6):
`md5("{collection_id}:{asset_id}:{doc_type}:{start_ms}:{end_ms}:{chunk_idx}")`
— collection_id in the hash prevents cross-collection dedup collisions;
`chunk_idx` (default 0) covers same-timespan multi-doc cases. Dedup runs
BEFORE any Bedrock call, so a resubmitted asset costs nothing to skip.

## Distributed lock design (per physical table)

- ONE lock item per `index_key` — `pk="lock",
  sk="{collection_id}:{table_key}"`. Different tables (across or within
  collections) never contend.
- **acquire(lock_sk, …)**: conditional PutItem; re-entrant per execution_arn
- **release_if_owner(lock_sk, arn)**: conditional delete → stream REMOVE event
- **queue**: `pk="pending", sk="{lock_sk}#{sha256(token)}"` — per-lock
  `begins_with` dequeue, FIFO by `created_at` WITHIN a lock
- **HandleLockRelease**: extracts the released sk from the stream
  record's Keys, `pop_next_pending(sk)` re-acquires (strict,
  non-re-entrant) + `SendTaskSuccess`; DEAD token
  (TaskDoesNotExist/TaskTimedOut/InvalidToken) → release again to
  cascade to the next waiter; TRANSIENT SFN failure → `requeue_pending`
  (waiter keeps its queue position) then release
- **HandleExecutionFailure**: `release_all_for_execution(arn)` — an
  Index/Rebuild execution holds one lock per Map iteration and may hold
  several when it dies
- **TTL safety net**: lock TTL = 3600 s; pending item TTL = 24 h

## Env vars

| Variable | Source | Purpose |
|---|---|---|
| `INDEX_STATE_TABLE` | CFN `!Ref LanceDBIndexStateTable` | DDB state/lock/asset/search-history table |
| `DEDUP_TABLE` | CFN `!Ref LanceDBDedupTable` | DDB dedup table |
| `SEARCH_HISTORY_TABLE` | CFN `!Ref LanceDBSearchHistoryTable` | DDB search history table |
| `SEARCH_HISTORY_TTL_DAYS` | CFN parameter (default 7) | Search history TTL |
| `COLLECTION_TABLE` | CFN `!Ref LanceDBCollectionTable` | Collection registry |
| `VECTOR_BUCKET` | CFN, sliced from `!GetAtt HybridSearchVectorBucket.VectorBucketArn` | S3 Vectors bucket (one per deployment) — named `{ResourcePrefix}-{AccountId}-{Region}-vectors`. There is no `...VectorBucketName` attribute to `GetAtt`; see Gotchas |
| `PROXY_BUCKET` | CFN `!Ref ProxyBucket` | S3 bucket for LanceDB index files, schema, vectors |
| `AWS_REGION` | Lambda runtime | AWS region for boto3 clients and LanceDB S3 storage options |
| `REBUILD_THRESHOLD` | hardcoded in `input_schema.py` (500) | Min new docs before ANN rebuild triggers |
| `LOCK_TTL_SECS` | hardcoded in `input_schema.py` (3600) | Lock expiry |

## Build, deploy, test

```bash
cd hybrid-search-on-aws
python3 -m unittest discover -s test -p 'test_*.py'
bash ./build-deploy-s3-dist.sh --build-only                  # local, no AWS
bash ./build-deploy-s3-dist.sh --bucket <BUCKET> [--prefix <PREFIX>] [--version <VERSION>] [--single-region]
```

Normally deployed as a child of
[micro-workflow-bundle](../micro-workflow-bundle/) (one stack for the
whole toolkit — see the repo root README). The stack is self-contained.
On stack create/update:

1. **CodeBuild** builds and pushes the Docker image to ECR
2. **CopySchema** custom resource seeds files from the deployment zip
   to `ProxyBucket`:

| Source in zip | Destination in S3 | Update strategy |
|---|---|---|
| `schema/*.json` | `_lancedb_index/schema/` | skip — preserves operator edits |
| `prompt_config/default.md` | `_promptconfig/baseline/hybrid-search/` | overwrite — always reset to packaged version |
| `workflow_config/hybrid-search.json` | `_workflowconfig/` | skip — preserves operator-controlled retrieval settings |

3. Lambda functions reference the ECR image by tag

**`workflow_config/hybrid-search.json` is skip-on-update:** once the
file exists in S3, stack updates never touch it. This keeps operator
customisations (model overrides, retrieval weights, active schemas)
stable across deploys. To reset to the packaged defaults: `aws s3 rm
s3://{bucket}/_workflowconfig/hybrid-search.json`, then trigger a stack
update.

**Changing the packaged `retrievalPaths` therefore reaches NO existing
deployment on its own** — and not only because of skip-on-update: every
collection record (including the seeded `default`) copied the weights
into DynamoDB at create time, and new collections inherit from the LIVE
`default` record, not the config file. Migrating a weights change to a
live stack means one `hybrid-search-admin___configure_search_backend`
call per existing collection (`default` included). That call is a FULL
replacement — read the collection's current `retrieval_paths` from
`list_collections` first (returned exactly as the read half of this
write), edit, and send the complete set: a partial payload wipes the
paths you didn't mention (measured live on a deployment that predated
the read path).

### Local test harness

```bash
# Generate + upload a StateIndexDocuments test payload (content contracts)
python test/harness/prepare_test_payload.py \
  --proxy-bucket <bucket> --uuid <uuid> --genres "news,politics" --content-category broadcast

# Invoke StateDecomposeSearchRequest locally (text search)
python test/harness/test_decompose_search_request.py \
  --proxy-bucket <bucket> \
  --index-state-table <table> \
  --search-history-table <table> \
  --dedup-table <table> \
  --search-term "Jane Doe on a stormy beach" \
  [--embed-model <model-id>] [--embed-dims <int>]

# Invoke StateDecomposeSearchRequest locally (image search)
python test/harness/test_decompose_search_request.py \
  --proxy-bucket <bucket> \
  --index-state-table <table> \
  --search-history-table <table> \
  --dedup-table <table> \
  --search-image-s3uri "s3://<bucket>/<uuid>/transcode/proxy/frames/frame_0001.jpg"
```

## Gotchas

Things that look like bugs but aren't, and things that look fine but are.

- **The S3 Vectors bucket is `AWS::S3Vectors::VectorBucket` with `DeletionPolicy: Retain` and `UpdateReplacePolicy: Retain`.** Retain because re-embedding a corpus costs real Bedrock spend, so the bucket deliberately outlives the stack. The reason a plain native resource is nonetheless safe is worth knowing, because the type has a sharp edge: **it cannot adopt an existing bucket** — measured against real CloudFormation, a same-named bucket fails the create with `409 AlreadyExists` / `HandlerErrorCode: AlreadyExists` and rolls back the **whole stack**, and a CFN `Condition` cannot rescue it (Conditions evaluate from parameters and pseudo-parameters only, never a runtime lookup). That collision is unreachable here only because `ResourcePrefix` is derived from the **stack id** in the shipped deployment paths ([micro-workflow-bundle](../micro-workflow-bundle/)): a delete + re-create yields a new stack id, hence a new bucket name, and can never land on the bucket it just retained. **If `ResourcePrefix` ever becomes operator-supplied and stable across a delete+recreate, this resource breaks** and needs an adopting custom resource.
- **`ResourcePrefix` is capped at 27 characters, and the S3 Vectors bucket is what caps it** — not, as looks likelier, a Lambda `FunctionName`. S3 Vectors allows 3–63 chars; the longest AWS region name is 14 (`ap-southeast-N`) and an account id is always 12, so `-{account}-{region}-vectors` spends 36 before the prefix contributes anything. That leaves **27**, tighter than the 30 a `FunctionName` allows. In the micro-workflow-bundle, `ResourcePrefix = ${Prefix}-${12-char stack-id slice}`, so the user-settable `Prefix` is capped at **14**. `test_vector_bucket_name_fits_the_63_char_limit` encodes the arithmetic; an earlier hand calculation of this budget got it wrong in the safe-looking direction, which is why it is now a test.
- **`!GetAtt HybridSearchVectorBucket.VectorBucketName` does not exist.** `VectorBucketName` is a `createOnlyProperty`, not a `readOnlyProperty`, on this type (`aws cloudformation describe-type`) — only `VectorBucketArn` and `CreationTime` are GetAtt-able, and `!Ref` returns the **ARN**, not the name. The six Lambda env vars and the `VectorBucketName` output that need the NAME slice it out of the ARN: `!Select [1, !Split ["bucket/", !GetAtt HybridSearchVectorBucket.VectorBucketArn]]` (verified on a live stack). Recomposing it with a second `!Sub` would work, but then the template holds two independent definitions of one string; `test_no_consumer_recomposes_the_vector_bucket_name` asserts the name pattern appears exactly **once**.
- **Any change to `ResourcePrefix` — including a stack delete + re-create — orphans the old vector bucket, with its vectors, silently.** `VectorBucketName` is a `createOnlyProperty`, so a renamed bucket is a **replacement**, and `UpdateReplacePolicy: Retain` leaves the old one standing. This is the accepted trade: an orphan is recoverable, a deleted vector corpus is not. Sweep stale `*-vectors` buckets manually.
- **No IAM role here may hold `logs:CreateLogGroup`** (except CodeBuild's, whose group the service creates). Every log group is a declared stack resource. With the permission, a function that cannot reach its declared group silently creates an unmanaged one — the failure mode disappears instead of surfacing. `!GetAtt SomeLogGroup.Arn` already yields the `...:*` form `logs:PutLogEvents` needs; verified against the deployed policy.
- **Four Lambdas reach their log group via `LoggingConfig.LogGroup`; five declare `/aws/lambda/{FunctionName}` directly. The split is NOT "unnamed vs named" — do not tidy it into one rule.** `StartBuildLambda` and `CopySchemaLambda` are unnamed, so `/aws/lambda/{name}` is unknowable at author time. `GatewayToolLambda` and `AdminToolLambda` *are* named, but their canonical groups **already exist** in deployed stacks — Lambda auto-created them before this template declared them — so declaring the canonical name fails the update. They use a `-log` suffixed group instead.
- **A pre-existing log group fails the update at PRE-FLIGHT VALIDATION, and the error names nothing.** Measured on a live stack: `Validation failed with 2 error(s). Call DescribeEvents to retrieve the full list of issues` — and `DescribeEvents` has no per-resource event, because the stack never got as far as creating resources. Count the errors and compare against the log groups you just added; that is the fastest diagnosis. CFN cannot skip an existing resource (Conditions see only parameters and pseudo-parameters, never a runtime lookup), and a custom resource that deletes the old group would destroy real logs and leave a deletion path in every future update.
- **NEVER rename one of the five canonical log groups (Indexer, Rebuild, Search, StreamHandler, HandleExecutionFailure).** `LogGroupName` is a `createOnlyProperty`, so a rename is a **replacement** — measured on a throwaway stack, CFN deletes the old group and every log in it, leaving nothing behind. These five are already stack-owned, so consistency with the `-log` convention would cost production logs. `test_the_canonical_five_keep_their_group_names` blocks it, and also blocks adding a `LoggingConfig` to them (which is how such a rename would start).
- **Every Lambda, whichever pattern, must `DependsOn` its group** or Lambda's first invocation wins the race and creates an unmanaged one.
- **`gateway_tool/admin_function.py` calls `boto3.client("s3vectors")` with no vendored SDK.** That is safe today — the `python3.13` runtime bundles a boto3 (measured: 1.42.97 on a live function) that knows the service natively. If S3 Vectors support ever needs a newer SDK than the runtime ships, this breaks with an opaque `UnknownServiceError`, not a clear version complaint.
- **The test suite needs PyYAML, and it is declared in `test/requirements.txt`, NOT the root `requirements.txt`.** The root one is `COPY`d into the Lambda container image by the Dockerfile, so anything added there ships to production. `import yaml` in the suite is wrapped in a `try` and `TestCfnTemplate` is `skipIf`-guarded: without PyYAML the template tests are skipped and the rest of the suite still runs, instead of a module-level `ImportError` that takes the whole suite down. Run `pip install -r test/requirements.txt` before relying on the template tests — a silent skip is not a pass.
- **Template tests assert against the PARSED template, never the raw text.** `cfnstack.yaml` documents the things it deliberately omits (`logs:CreateLogGroup`, `!GetAtt ...VectorBucketName`), so a text search for an "absent" property matches the comment explaining its absence and the test fails on the template's own documentation.
- **`test/` has no `__init__.py`**, so `python3 -m unittest test.test_units.Class.test_name` raises `ModuleNotFoundError` — which prints as `FAILED (errors=1)` and reads exactly like a mutation test that worked. Use `discover -s test -k <name>`, and check the counter name is `failures`, not `errors`.
- **LanceDB skips IVF_PQ below 256 rows** (`build_ann_index`), so small collections brute-force scan. Correct behavior at that scale; a stated consequence of per-collection tables vs. one big shared table.

---

## Design record

The numbered design, kept because code comments cite it (`README.md
Design record §N`). Decisions carry their WHY; superseded alternatives
are named so they are not re-proposed.

### 1. Collections

A collection is a named tenant grouping (news, sport, episodic…) with
**level-2 isolation: per-collection physical storage, shared compute**
(decision D2) — one CFN stack, shared Lambdas/SFNs/DDB tables;
per-collection S3 locations, per-collection S3 Vectors indexes,
per-physical-table locks.

#### 1.1 Single namespace — every physical location is collection-scoped

| Unit | Scheme |
|---|---|
| LanceDB table | `_lancedb_index/{collection_id}/{doc_type}/{embedding_model}/{dims}[/{lang}]` |
| S3 Vectors index | `{collection_id}--{doc_type}--{hash8}` (hash8 = deterministic hash of `{collection_id}:{table_key}`; 63-char lowercase name cap) |
| Index-state record | `pk="state"`, `sk="{collection_id}:{table_key}"` |
| Lock item | `pk="lock"`, `sk="{collection_id}:{table_key}"` |
| Pending-queue item | `pk="pending"`, `sk="{lock_sk}#{sha256(task_token)}"` |
| Dedup record | `table_name` attribute = `{collection_id}:{table_key}` |
| Asset-metadata record | `pk="asset"`, `sk="{collection_id}:{asset_id}"` |

`table_key` itself stays `{doc_type}:{embedding_model}[:{iso_language}]`.
There is NO un-prefixed legacy namespace and NO empty-collection_id
branch anywhere in the runtime. An earlier draft preserved a parallel
"legacy mode" (un-prefixed S3 locations, a global lock, a runtime
config-file fallback) for a consumer that turned out not to be live;
the dual-mode design was deliberately cut — single namespace, single
lock scheme, single document pipeline. A hard constraint that did
survive: the pre-existing DynamoDB tables' RESOURCE DEFINITIONS must
not change (no GSI add/remove, no pk/sk change) — key VALUES are free
(the collection prefixes are data, not schema); NEW tables are fine.

Deletion story: delete a collection = delete one S3 prefix
(`_lancedb_index/{collection_id}/`), its S3 Vectors indexes (enumerate
via `sk begins_with "{collection_id}:"` state records), and its DDB
keys. No row-level surgery.

Accepted isolation cost (stated, not discovered later): LanceDB skips
IVF_PQ below 256 rows, so small collections brute-force scan.
Per-collection tables also mean N× FTS/ANN rebuild work in aggregate —
but rebuilds are per-table-lock parallel (D9), so wall-clock is
unaffected.

#### 1.2 The `default` collection replaces any runtime fallback

At deploy time the `CopySchema` custom resource seeds a collection
record `pk="default"` into `LanceDBCollectionTable`, derived from
`workflow_config/hybrid-search.json` (active_scopes from `schemas`,
schema_map from the per-scope keys, retrieval paths from
`retrievalPaths`/`imageRetrievalPaths`). Seed strategy: skip-if-exists
(operator edits survive redeploys). Callers that omit `collection_id`
resolve to the literal string `"default"` in the gateway/Decompose
entry point — ONE line of defaulting, zero alternate code paths. The
global workflow config remains the SOURCE for seeding the default
collection, but the runtime reads collections only.

#### 1.3 Collection registry — `LanceDBCollectionTable`

Attributes listed under [DynamoDB tables](#lancedbcollections--collection-registry).
Rules:

- `collection_id` validation at creation: `[a-z0-9_-]{3,48}`; MUST NOT
  equal `schema` or any doc_type name (keeps the S3 namespace
  `_lancedb_index/{collection_id}/` disjoint from
  `_lancedb_index/schema/`; the doc_type rule is defense-in-depth now
  that no un-prefixed doc_type locations exist). The 48-char cap leaves
  room in the 63-char S3 Vectors index-name budget.
- Engine vs. tenant settings split: `primaryModel`/`secondaryModel`/
  `promptS3Key` are ENGINE settings — they stay in the global workflow
  config, read by Decompose. The collection record carries only TENANT
  settings (scopes, schema_map, weights). One source each, no
  duplication.
- `create_collection_if_absent`: conditional put
  (`attribute_not_exists(pk)`); `ConditionalCheckFailedException` =
  someone else created it = success. Same idiom as `register_index`.
- Retrieval-path keys are validated against the collection's
  active_scopes at write time (a weight for a scope the collection
  doesn't index is a typo — reject, don't ignore).

#### 1.4 Per-table lock — replaces the global lock everywhere

`LOCK_SK` stopped being a constant; every acquire/release/queue call
takes `lock_sk = "{collection_id}:{table_key}"`. The parts that are the
difference between working and deadlocked:

1. `lock_utils.acquire(lock_sk, holder, execution_arn)` /
   `release_if_owner(lock_sk, execution_arn)` — sk parameterized.
2. `store_pending(lock_sk, ...)` — pending item keyed
   `sk = "{lock_sk}#{sha256(token)}"` and carrying `lock_sk` as an
   attribute.
3. `pop_next_pending(lock_sk)` — a `begins_with(sk, "{lock_sk}#")`
   query, so the handler only considers waiters for the lock that was
   actually released. FIFO stays `created_at`-ordered WITHIN a lock
   (global FIFO across locks is meaningless once locks are
   independent).
4. `HandleLockRelease` accepts any REMOVE on `pk="lock"`, extracts the
   removed `sk` from the stream record's Keys, and passes it to
   `pop_next_pending(sk)`. (A sk-equality check against a global
   constant would SILENTLY DROP collection-scoped releases — waiters
   would hang until heartbeat timeout.) The CFN `FilterCriteria`
   (`pk="lock"`, REMOVE) matches any sk.
5. IndexSFN ASL: `StatePrepareDocuments` computes `lock_sk` and puts it
   in the state payload; `Acquire lock`, `Release lock`, and the Catch
   release path all reference it.
6. RebuildSFN: each Map iteration acquires its own table's lock;
   `index_name` input is the collection-scoped
   `{collection_id}:{table_key}`.

### 2. One document pipeline — the content contract

#### 2.1 Upstream producers are contract emitters

Before the contract, the upstream media-analysis pipeline embedded
documents itself in a second, parallel (Node) embedding-provider
implementation that had to track the Python providers forever. That
implementation was deleted: upstream keeps its domain logic (walking
analysis structure, selecting representative frames, composing text
blobs, supplying PRE-extracted entities) but emits content contracts
(§2.2, JSONL) to S3. All embedding happens in ONE place — this stack's
Python providers, inside `StatePrepareDocuments`. Consequence:
`StateIndexDocuments` accepts exactly ONE input shape (rows produced by
`StatePrepareDocuments`) — never a mix of pre-embedded and contract
inputs.

The backend's `extract_entities: true` capability exists for callers
WITHOUT pre-extracted entities (e.g. a raw speech-diarization
transcript); callers whose pipeline already extracted entities emit
them as pre-structured `{kind:"text", field:"named_entities",
extract_entities:false}` items — no SLM call, no cost.

#### 2.2 Contract shape

One JSON object per DOCUMENT (decision D4: caller declares WHAT via a
closed `kind` enum — `text`, `image`; backend does HOW; caller owns
content SELECTION — representative frame, best crop; no
`image_sequence` kind):

```jsonc
{
  "collection_id": "news",            // omitted -> "default"
  "asset_id": "asset:uuid:abc123",
  "doc_type": "scene",
  "start_ms": 12000, "end_ms": 18500,
  "chunk_idx": 0,                     // optional; D6
  "refresh": false,                   // optional; true = delete this asset's
                                      // rows in this table first
  "fields": {                         // verbatim scalar pass-through: values
    "segment_parent_id": "..."        // copied into the row as-is (validated
  },                                  // against schema mappings, no processing)
  "content": [                        // items that need backend PROCESSING
    {
      "kind": "text",
      "field": "text",
      "value": "A man walks into a stormy beach house...",
      "extract_entities": true        // -> named_entities, dedupe-append (D7)
    },
    {
      "kind": "text",
      "field": "named_entities",
      "value": "Jane Doe, John Roe"   // pre-structured (e.g. face-rec
                                      // people[] — D8); no extraction
    },
    {
      "kind": "image",
      "field": "vector",              // must be the schema's ann field
      "value": "s3://proxy-bucket/.../frame_0045.jpg"
    }
  ]
}
```

There is deliberately no `fts: true` flag on content items (dead weight
— the schema decides FTS; a text item targeting an FTS field is just a
write), and verbatim scalars live in `fields{}`, not `content[]`, so
`content[]` contains only items requiring embedding or extraction.

#### 2.3 Two submission modes (bulk is mandatory, not a nice-to-have)

- **Inline**: `index_content(documents: [<contract>, ...])` — small
  interactive/agent submissions. Hard cap 100 documents
  (`MAX_INLINE_DOCUMENTS` — stay far under the 256 KB SFN input limit).
- **Manifest**: `index_content(documents_s3uri:
  "s3://.../contracts.jsonl")` — JSONL of contract objects. This is how
  pipelines submit (a 2-hour asset = thousands of utterance/shot docs;
  per-doc calls are absurd and inline would blow the SFN limit).
  Mirrors face-recognition's `batch_index_faces` manifest precedent.
  Exactly one of `documents | documents_s3uri`.

All documents in one submission MUST share one `collection_id`
(validated) — a submission maps to one SFN execution whose lock set is
computed up front. Documents MAY span doc_types;
`StatePrepareDocuments` groups by doc_type and the SFN Map runs one
lock/index cycle per physical table.

#### 2.4 IndexSFN shape

```
Prepare documents (StatePrepareDocuments — lock-free)
  1. Load contracts (inline payload or S3 manifest)
  2. Resolve collection record (missing collection_id -> "default";
     unknown collection_id -> fail hard)
  3. Per doc_type: resolve schema via collection.schema_map; validate
     every contract field/content.field against schema mappings
     (unknown field -> ValueError, fail hard)
  4. Compute doc_ids (D6); batch_is_indexed dedup BEFORE any Bedrock
     call (skip already-indexed unless refresh)
  5. Embed (schema's own model/dims — never caller-supplied) +
     extract entities (D7 merge) for survivors
  6. Write prepared row groups to S3, one object per physical table;
     emit Map items [{lock_sk, rows_s3_key, collection_id, table_key}]
Map (per physical table):
  Acquire lock (lock_sk)  → Index documents → Release lock (lock_sk)
```

`StateIndexDocuments` keeps register/verify schema state,
refresh-delete, defensive dedup, LanceDB append / S3 Vectors put, FTS
rebuild, mark indexed, pending-rebuild counter — parameterized by
`{collection_id}:{table_key}` state keys and collection-scoped
locations.

#### 2.5 Asset metadata

`index_content` accepts optional `asset_metadata: {title, casts,
genres, content_category}` (top level, once per submission). Written as
`pk="asset"`, `sk="{collection_id}:{asset_id}"` — required for `@Title`
resolution and result enrichment; without it those degrade gracefully
(no title match, empty metadata in results). `@Title` resolution
(`title-index` GSI) and `batch_get_asset_metadata` are
collection-scoped by key prefix — no cross-collection title leakage.
The `title-index` GSI keys on `(pk="asset", title)`; with
collection-scoped sk values the GSI still works — query `pk="asset" AND
begins_with(title, ...)` then post-filter items whose sk starts with
`{collection_id}:` (title values are per-collection few; the
post-filter avoids any GSI change, honoring the no-resource-definition-
change constraint in §1.1).

### 3. Search is collection-scoped

SearchSFN input: `{collection_id?, search_term | search_image_s3uri,
k?, workflow_config_uri?}` — `workflow_config_uri` carries engine
settings (SLM models, prompt); tenant settings (active scopes,
schema_map, retrieval weights) come from the collection record.

- Decompose: loads the collection record; everything else (operator
  parsing, SLM, embed-per-(model,dims), vectors.json) is
  collection-independent.
- SearchIndex: `LanceDBBackend(schema_def, collection_id=...)` — the
  constructor derives the collection-scoped location. **Physical
  isolation IS the tenant filter; no `WHERE collection_id` clause.**
  The `collection_id` column on rows stays as defense-in-depth/audit
  (D6 keeps it in doc_id too).
- Rerank: `batch_get_asset_metadata(collection_id, asset_ids)` — key
  prefix per §2.5.

### 4. S3 Vectors — verified service facts

- `AWS::S3Vectors::VectorBucket` — native CFN, ONE bucket per
  deployment, `DeletionPolicy: Retain` (see the DeletionPolicy decision
  below).
- Vector INDEXES are created at RUNTIME (first write per
  `(collection_id, table_key)`) via `CreateIndex` — per-collection
  indexes are unbounded/agent-created, so deploy-time CFN declaration
  is impossible. First-write-wins, "already exists" = success. IAM:
  `s3vectors:CreateIndex/GetIndex/PutVectors/QueryVectors/DeleteVectors`
  scoped to the one bucket ARN.
- PutVectors = true upsert by key (verbatim doc-quote verified) —
  retry-safe.
- Limits that shape code: 500 vectors/Put+Delete call (`S3V_BATCH_SIZE`
  chunking loop); 40 KB metadata/vector + ≤50 keys (caps
  `result_fields` on ANN-only schemas); 2 KB filterable budget
  (asset_id + doc_id only; the collection is the INDEX, not metadata —
  one filterable field saved); ≤10 non-filterable keys declared per
  index at CreateIndex from the schema; QueryVectors pages at 100
  results (paging loop if candidate_k > 100); dimension 1–4096 (all
  current providers fit).
- Dual-store write (FTS fields → LanceDB, vector → S3 Vectors, same
  doc_id): mark_docs_indexed only after BOTH succeed; retry re-attempts
  both (LanceDB delete-then-append on refresh; S3V upsert is naturally
  idempotent). DeleteVectors fan-out uses the dedup GSI.

### 5. Gateway tools

**`hybrid-search` target (normal callers):**

| Tool | Contract |
|---|---|
| `index_content` | `(collection_id?, asset_id, documents \| documents_s3uri, asset_metadata?)` → async job_id; poll via `get_job_status` |
| `search_content` | `(collection_id?, search_term \| search_image_s3uri, k?)` → sync results |
| `create_collection` | `(collection_id, name, active_scopes[], schema_map?)` → create-if-absent; schema_map defaults by convention |
| `list_collections` | `()` → registry listing (names, scopes, retrieval paths, row counts from state records) — a cheap DDB-only summary, no live `count_rows()` S3 round-trips; a `describe_collection` with live counts can be added if a real need appears |
| `get_job_status` / `describe_tool` | standard micro-workflow contract |

**`hybrid-search-admin` target (separate Lambda + role + gateway
target — decision D3):**

| Tool | Contract |
|---|---|
| `configure_search_backend` | `(collection_id, retrieval_paths?, image_retrieval_paths?, active_scopes?)` → update existing; FULL replacement of the maps it carries; sets `weights_locked` on weight edits |
| `delete_collection` | `(collection_id, force?)` → S3 prefix + S3V indexes + DDB keys sweep; `default` requires force |

Schema authoring stays deploy-time (edit `schema/*.json`, redeploy) —
never a tool (D1).

### Decisions D1–D9

| # | Decision |
|---|---|
| D1 | Schema files are design-time artifacts: authored by humans, seeded at deploy, immutable per physical table once a document lands (`_ensure_registered`). Never edited via tools. |
| D2 | Collections = named tenant groupings. **Level-2 isolation: per-collection physical storage, shared compute** — one CFN stack, shared Lambdas/SFNs/DDB tables; per-collection S3 locations, per-collection S3 Vectors indexes, per-physical-table locks. |
| D3 | Admin vs. normal caller = IAM boundary (separate gateway target + Lambda + role), never a payload flag. Weight-freeze, not full-freeze: an admin edit locks `retrieval_paths`/`image_retrieval_paths` only; `active_scopes` additions stay open to normal callers. |
| D4 | Content contract: caller declares WHAT (closed `kind` enum: `text`, `image`); backend does HOW (embedding dispatch, FTS, entity extraction). Caller owns content SELECTION. No `image_sequence` kind. |
| D5 | S3 Vectors = per-schema opt-in (`vector_store: "s3vectors"`); FTS stays in LanceDB; new schemas only, no migration tooling. |
| D6 | doc_id backend-derived: `md5("{collection_id}:{asset_id}:{doc_type}:{start_ms}:{end_ms}:{chunk_idx}")`. collection_id in the hash prevents cross-collection dedup collisions; chunk_idx (default 0) covers same-timespan multi-doc cases. |
| D7 | Entity merge: dedupe-append (normalize lowercase+strip for comparison, keep first-seen casing — consistent with `named_entities`' lower_case FTS tokenization). |
| D8 | Face-recognition integration = search-by-NAME only (names into `named_entities` FTS via the contract; no new doc_type, no face vectors). Search-by-face-image is a separate future feature, not built; its doc_id formula, pre-corrected for D6 consistency: `md5("{collection_id}:{asset_id}:face:{face_id}:{start_ms}")`. |
| D9 | Concurrency: per-physical-table atomic lock. Concurrent indexing across collections, concurrent rebuilds across collections, concurrent different-doc_type indexing within a collection. Only same-physical-table writers serialize. |

### DeletionPolicy decision

The DynamoDB tables carry `DeletionPolicy: Delete` with default SSE —
sample posture: they hold **rebuildable derived state** (locks, dedup
records, index state, search history), so losing them costs a re-index,
not data. The S3 Vectors bucket is the opposite case — `DeletionPolicy:
Retain` + `UpdateReplacePolicy: Retain` — because re-embedding a corpus
costs real Bedrock spend; see the first Gotcha for why the retained
bucket can never collide with a re-created stack, and the sharp edge if
`ResourcePrefix` ever becomes operator-stable.

### P. Prototype lessons (pre-workflow LanceDB benchmark)

A standalone prototype evaluated LanceDB as a serverless hybrid-search
engine before the workflow existed. The code is gone; the lessons that
shipped:

- **LanceDB on S3, native object_store** — byte-range GET per query, no
  full index download. The storage layout became
  `_lancedb_index/{collection}/...`.
- **Cohere Embed v4 via Bedrock**, `output_dimension: 1024`, documents
  as `search_document`, queries as `search_query`.
- **Queries embed ONE text per invoke_model call** — batching produces
  slightly different vectors, which silently degrades ANN agreement
  with the indexed space. `lib/providers/cohere_embed_v4.py` still
  builds single-text requests for queries.
- **< 256 rows → skip the ANN index, flat scan** — carried verbatim
  into `lancedb_backend.rebuild_ann()`.
- **`create_fts_index(..., replace=True)` after appends** — safe to
  rebuild repeatedly.
- **Score scales**: LanceDB `_distance = 1 − cosine_sim` (range [0,2]);
  OpenSearch nmslib `cosinesimil = (1 + cosine_sim) / 2`. Conversions,
  not disagreements — results aligned on identical query vectors.
- **Memvid: refused.** memvid-sdk v2 is incompatible with Lambda arm64
  on AL2023; dropped entirely in the workflow.
- The prototype's ANN was IVF_HNSW_SQ with `refine_factor=3` to
  re-score past scalar-quantization distortion; the workflow uses
  IVF_PQ with partitions capped near sqrt(rows) (see `rebuild_ann`).
