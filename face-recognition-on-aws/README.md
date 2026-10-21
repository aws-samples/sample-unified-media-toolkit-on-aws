# face-recognition-on-aws

Shot-driven face recognition for video, packaged as a standalone AWS micro-workflow.
An AWS Lambda container pipeline on Amazon Rekognition image APIs: it extracts frames
through the media-processing sibling workflow (gateway submit + callback token), detects
faces on batched grid images with a probe-then-fill cost strategy, builds temporal face
tracks, recognizes celebrities, clusters identities via a transient per-run sandbox
collection, and matches/indexes against a persistent collection. A second, smaller state
machine provides batch face-index management (index / update / delete / collection
lifecycle). A DynamoDB counting semaphore bounds concurrent Rekognition usage
account-wide.

**No local ML models** — all inference is Rekognition (DetectFaces, RecognizeCelebrities,
collections). The container carries only pyvips (imaging) + numpy. No ONNX runtime, no
model seeding.

This file is the single developer document for the folder: architecture and operations
first, then a numbered **Design record** (§1–§23) that preserves the section numbers the
code cites (`README.md §N` in docstrings and comments).

## Workflow states

Defined in [definitions.json](definitions.json) — the single source of truth; the build
script injects it into cfnstack.yaml's `DefinitionString`. Entry point:
[lambda_function.py](lambda_function.py) dispatches on `event["operation"]`.

```
Prepare media processing       (builds the GATEWAY request)
  └─ Run media processing        (lambda:invoke.waitForTaskToken — HELD: submits to MP's gateway with a callback token; jpeg frames @ frame_interval_sec under MP's own run prefix)
       └─ Extract processed frames (enforces MP's terminal status BY NAME — the family-wide fail-by-name convention: MP refusing the jpeg face_frames group as NoCompatibleOutputGroups raises lib.exceptions.NoVideoStream carrying MP's own message, the class name being the stable code in the execution's error field; every other failure re-raises with MP's forwarded error/cause passed through; normalizes the handle to {frames_bucket, manifest_key, probe_key, duration_sec, media_processing_arn} — the ONLY state that knows about the handle; non-handle shapes raise)
            └─ Prepare probe iterators (frame triage + patch dHashes — S3/CPU only)
            └─ Acquire semaphore  (waitForTaskToken; Rekognition span starts)
                 └─ Map probe detection        (MaxConcurrencyPath → detect_map_concurrency, default 4)
                      └─ Plan dense detection  (skip/dense gap classification)
                           └─ Map detect faces (MaxConcurrencyPath → detect_map_concurrency, default 4)
                                └─ Build face tracks
                                     └─ Map recognize celebrities (MaxConcurrencyPath → detect_map_concurrency, default 4)
                                          └─ Cluster identities   (sandbox collection)
                                               └─ Search and index faces
                                                    └─ Release semaphore  (span ends)
                                                         └─ Finalize outputs
                                                              └─ Store debug images
                                                                   └─ Usage tracking
```

Out-of-band handlers (same container image):
- **StateHandleSemaphoreRelease** — DDB stream (REMOVE on `pk=semaphore, sk=holder#*`):
  exactly-once counter decrement + FIFO pending drain.
- **StateHandleExecutionTerminal** — EventBridge on ALL terminal states (incl.
  SUCCEEDED): semaphore holder delete + `sandbox-{workflow_id}` collection delete (fire,
  ignore ResourceNotFound).

A second state machine, `${ResourcePrefix}-face-index`
([definitions_index.json](definitions_index.json)), handles batch index management —
see §19.

## Detection strategy — probe-then-fill

1. **Triage** every frame (threaded pyvips shrink-on-load decode, every frame — no
   zero-decode byte-size pre-filter; removed, see Gotchas): luma stats (black/mono/blur).
   Each kept frame gets a **3×3 patch + global dHash**.
2. **Probe pass**: DetectFaces on every `probe_stride`-th frame (default 4), packed ≤20
   frames per grid image (dimension-agnostic layout solver, 2 px cell separators,
   submitted by S3Object reference — 15 MB / 10000 px bounds).
3. **Plan**: an inter-probe gap is skipped iff both endpoint probes found no faces AND
   stable-patch hashes are adjacent-static AND endpoint-static (catches zoom/pan drift)
   AND the global hash agrees. Chronically-churning patches (lower-third tickers,
   scoreboards — >70 % change frequency, ≤3 of 9) are excluded from the stability test.
4. **Dense pass** over face regions (dilated ±probe_stride) with near-duplicate dedup;
   skipped frames get interpolated boxes tagged `source: "interpolated"`.

Every failure mode degrades toward MORE detection: crowd overflow → recursive grid
split (≥95 faces); cross-cell face bleed → separator lines + >15 % overhang rejection;
zoom drift → endpoint distance; entrances → patch hash break + probe cadence bound.

## Identity trust hierarchy

Two recognizers can name a cluster: **RecognizeCelebrities** (a vote tallied per cluster)
and the **persistent collection** (SearchFacesByImage on the cluster's best face). The
pipeline's measured trust ordering, and why (§15, §23):

- Celebrity recognition produces confident false positives on non-celebrities — wrong
  names arrive scattered (dozens of distinct real public figures, a different one almost
  every look) at a mean confidence well below a genuine match's (~83 vs ~99). Naming is
  therefore a **plurality vote with mean confidence**, never the single
  highest-confidence sample (§15).
- A collection match is a curated reference, so **collection wins at contradictions
  only** — `_resolve_name` prefers `face_match`, but when the collection name disagrees
  with a validated celebrity plurality, a second witness from a DIFFERENT track must
  corroborate it, and the published person carries `name_conflict` +
  `name_conflict_detail` (both candidates, the resolution) either way (§23). Higher
  similarity is deliberately NOT the tie-breaker: similarity measures photo-to-reference
  quality, not which witness belongs to the cluster.
- `rep_face_key` vs `thumbnail_key` (§22): `thumbnail_key` is the per-person debug
  MONTAGE; `rep_face_key` is the single representative face crop (3:4, native resolution
  ≤512 px long edge) — the face-matching reference. Consumers must never feed the montage
  to a matcher.

## Recognition policy (GDPR use cases)

Three orthogonal toggles; enforcement in the state handlers (empty work lists, no ASL
branches):

| Toggle | Effect when off/empty/zero |
|---|---|
| `celebrity_recognition: false` | no RecognizeCelebrities calls |
| `collection_id: ""` | no persistent-collection ops (SearchFacesByImage AND IndexFaces) |
| `max_faces_to_index: 0` | auto-index off; persistent search still on — **the default** (auto-indexing is opt-in: pass >0) |
| `index_named_faces_only: true` | auto-index restricted to high-confidence-named faces |

The per-execution sandbox collection (`sandbox-{workflow_id}`) is **transient processing
state** — created if absent, deleted by the terminal-state EventBridge handler — and runs
in every use case, so clustering quality and the MDS visualization are identical across
policies. GDPR storage-limitation governs persistent templates, not transient processing
(§2b).

**Index budget is ranked, not first-come**: `search_index_faces` runs two passes — pass 1
searches every validated cluster against the persistent collection; pass 2 ranks the
unmatched, policy-eligible candidates by `_index_rank` (screen_time × sqrt(num_tracks) ×
best-face fusion_rank) and spends the `max_faces_to_index` slots on the top-ranked
groups. Previously the budget was consumed in cluster-iteration order (track-count only),
so large-but-brief fragment clusters could exhaust it before a person with real screen
presence and a good face crop was reached.

## Semaphore (DynamoDB, counting)

Per-holder items + transactional counter in the stack's single table (NOT the reference
pipeline's counter-only style — that shares one TTL and cannot reclaim an individual
crashed holder). Key invariants:

1. **Single decrement path**: normal release, EventBridge cleanup, and TTL expiry all
   just DELETE the holder item; only the stream handler decrements the counter.
2. **Exactly-once decrement**: streams are at-least-once — each decrement is a
   transaction with a per-eventID dedup marker.
3. **TTL refresh**: every Rekognition-span state refreshes its holder's TTL on entry
   (3600 s bounds a *state*, not the span); crashed jobs reclaim ≤1 h after their last
   state entry.
4. **`max_holders` is request-level, up-only**: `input_config.max_concurrent_executions`
   (default 5, gateway param, NOT a stack parameter) is applied on every acquire via a
   raise-only conditional update — a request at or below the currently-stored value is a
   no-op. The effective cap can only grow across the table's lifetime; there is no
   per-request lowering (a single low-value caller must not throttle every other caller).
5. **Counter self-heals via TTL, anchored to holder liveness**: the counter item also
   carries a `ttl` (3600 s), refreshed by `try_acquire`, `decrement_once`, and
   `refresh_holder` — the self-heal for a permanently-lost stream decrement
   (event-source-mapping outage, exhausted retries), which would otherwise leave
   `holders` stuck too high forever. `refresh_holder` only refreshes the counter's ttl
   after confirming its OWN holder item still exists, so a zombie execution (already
   TTL-reclaimed) can't resurrect the counter. Refreshing only on acquire/decrement (the
   reference pipeline's pattern) would falsely expire the counter under a saturated,
   long-lived queue — see §14.

Acquire is `waitForTaskToken`: slot free → immediate SendTaskSuccess; full → pending item
(FIFO by `created_at`, 24 h TTL, carries the waiter's `requested_max`), drained by the
stream handler. The semaphore is held ONLY across the Rekognition span —
media-processing frame extraction of queued jobs proceeds concurrently.

## Table items (single DDB table)

| pk | sk | Purpose |
|---|---|---|
| `face#{face_id}` | `record` | persistent face records (GSIs `celeb-index`, `collection-index`) |
| `deleted#{face_id}` | `record` | hard-delete tombstones (TTL 30 d, audit only) |
| `semaphore` | `counter` | holders count (TTL, refreshed alongside holder liveness) |
| `semaphore` | `holder#{execution_arn}` | per-holder slot (TTL) |
| `pending` | sha256(task_token) | queued waiters (TTL 24 h) |
| `stream-dedup` | {eventID} | exactly-once markers (TTL 1 h) |

`collection-index` GSI (pk `collection_id`, sk `celeb_created` = `{celeb}#{epoch_ms}`,
unnamed faces sort last via the `~unnamed` sentinel) drives `list_faces` pagination. Face
records carry `origin` ("user"|"auto"), `source_key`/`source_etag` (batch-index
idempotency is keyed on content ETag, not filename), and normalized `coord` `[x,y,w,h]`
locating the face in the original AND in the stored `full_image` (fit-inside downscale of
the WHOLE source image — not a face crop).

## Output files (S3)

All under `micro-workflow/face-recognition/{workflow_id}/`:

| File | Contents |
|---|---|
| `facemetadata.json` | flat `people[]` sorted by screen time: name/named/name_source, appearances (begin_ms/end_ms), total_screen_time_sec, thumbnail_key (montage), rep_face_key (single reference face), face_id / celebrity_id (+ celebrity_urls/known_gender via GetCelebrityInfo) — the recognizer's own identity keys, gated on name agreement, and — when the collection match disagreed with a validated celebrity plurality (§23) — `name_conflict` + `name_conflict_detail` (both candidates, the resolution); `stats`; `usage` |
| `facegroup.json` | full detail: per-face boxes with `source: detected\|interpolated`, tracks, triaged frames, detection-plan summary |
| `face_clusters.json` | MDS 2D coords per person (from the clustering similarity matrix — zero extra API calls) |
| `usage_tracking.json` | per-state Lambda cost + child MP cost + Rekognition API counts |
| `debug/people/person_NNNN.jpg` | per-person montage (doubles as thumbnail_key target) |
| `debug/people/person_NNNN_face.jpg` | representative face crop, 3:4 native res ≤512px long edge (rep_face_key target — the face-matching reference; §22) |
| `grids/` | detection/recognition grid JPEGs + raw Rekognition responses |
| `ephemeral/` | intermediates: frame_index, shot_boundaries (shots mode), iterators, detection_plan, face_tracks, cluster_state, identity_clusters, search_index_faces |
| *(MP's own run prefix)* | frame output (frame_manifest.json + frames + probe.json + MP's usage_tracking.json) — `micro-workflow/media-processing/{workflow_id}/` in the proxy bucket |
| `_facestore/{collection_id}/` | indexed-face thumbnails (96×128 face crops) + full images (fit-inside downscale of the WHOLE source frame/photo, ≤1280px long edge; face located via the record's normalized `coord`) — bucket-root level, shared across runs |

## Key conventions

- **Frame identity**: source frame number parsed from MP's `frame_{n:07d}.jpg` names;
  `timestamp = frame_num / fps` with exact fps (Fraction from probe.json `r_frame_rate`).
- **Frame extraction**: fit-inside 1920×1080, never upscale, portrait auto-handled — all
  inherited from MP's `scale_dimension`.
- **Tracking**: gender + gap-dilated IoU (`1 + 0.5·gap_sec`) + **cut-break veto**.
  Default cut evidence: the dHash data doubles as a cut detector (any stable-patch break
  between two frames blocks track continuation). With the optional
  `shot_segmentation_s3uri` supplement (§21 + §23), DINOv2 shot boundaries from a
  shot-segmentation run on the same asset JOIN the dHash veto as a UNION — either
  detector's break vetoes continuation (boundaries add slow dissolves; dHash stays as the
  local fallback for boundaries the upstream list is missing, which is how the §23
  mislabel welded two actors into one track). Boundaries additionally veto dense-skip
  gaps containing a cut. Validated fail-hard in `prepare_probe_iterators` (schema +
  implied-fps ±1% + duration ±max(2 s, 2%) same-asset cross-checks) before any
  Rekognition spend; `facemetadata.json` records `stats.cut_detector: "shots+dhash" |
  "dhash"`.
- **Checkpoint/re-enter**: long-running states persist partial results, watch
  `get_remaining_time_in_millis()` (60 s buffer), raise
  `LambdaResourceTimeoutException`; the ASL retries (10×) and re-entry skips completed
  work. No content-length ceiling.
- **Grid submission by S3Object reference** — grids land in `grids/` for audit anyway;
  15 MB bound verified at compose time (quality steps 85→75→65).
- **Sandbox reuse on re-enter**: `cluster_identities` never deletes the sandbox; a
  `ResourceAlreadyExists` means THIS execution created it (workflow_id names the
  execution; SFN name-uniqueness).

## AgentCore gateway tool

[gateway_tool/lambda_function.py](gateway_tool/lambda_function.py) — flat zip, no
Docker. Core tools: `recognize_faces`, `get_job_status`, `describe_tool`. Standard
micro-workflow contract: workflow_id = uuid5(canonical input + idempotency token +
TOOL_VERSION).

- **An ABSENT `idempotency_token` means "do not deduplicate me"** here — one is
  generated, so every call gets its own run (see Gotchas), unlike the siblings which
  dedup on parameters alone. The token is folded into the hash preimage as extra material
  and then **discarded**: never forwarded to the state machine, never stored, **never
  echoed by any response** (see Gotchas). The event `uuid` in the SFN input equals the
  execution name, because in-workflow states reconstruct the execution ARN from it —
  that is this execution's local id, not the caller's token.
- `ExecutionAlreadyExists` → the call is **absorbed** into the existing run (no second
  execution) and returns a **HANDLE, never results**: the same fields as a fresh start —
  `job_id`, `workflow_id`, `status`, `asset_s3uri`, `output_s3_prefix` — with `status`
  set to the existing run's, plus `message` (the only signal the call was absorbed) and,
  on a terminal failure, `error` + `error_code` compacted exactly like
  `get_job_status`'s. `current_state` is present on a fresh start and a RUNNING absorb
  only — a stopped execution is in no state, and reading one costs a
  `GetExecutionHistory` paging pass for an `"unknown"`. A `SUCCEEDED` submit therefore
  carries no faces: results come from `get_job_status` alone, which is what makes the
  declared `OutputSchema` equal the reachable field set. Cross-stack regression:
  `../micro-workflow-bundle/test/test_units.py::TestSubmitReturnsAHandleNotResults`.
- `get_job_status`'s **every** branch carries `job_id`, `workflow_id`, `asset_s3uri`,
  and `output_s3_prefix` (the prefix read back from the execution's *input*, so a
  RUNNING or failed run still has one — partial results already live under it); terminal
  failures add `error` — `code: message` when the cause parsed as a Lambda error JSON,
  the raw cause (truncated 600) otherwise — and `error_code`: the stable code alone,
  i.e. the raised exception's class name (`NoVideoStream`) or, when the cause carried no
  parseable type, Step Functions' own error field (`Lambda.Unknown`).
- outputs[] `{uri, mime, label, desc}` with labels `face-metadata`, `face-groups`,
  `face-clusters`, `people-images` (prefix), `debug-grids` (prefix), `usage`. `desc` is
  the one-line role description joined verbatim from
  [gateway_tool/output_contract.json](gateway_tool/output_contract.json) — the same file
  `describe_tool` serves in full, so the two surfaces cannot disagree; it is omitted,
  never fabricated, for a label the contract lacks.
- `describe_tool` returns `{version, state_machine_arn, description, output_contract}`.
  `output_contract` is the field-level contract for every `outputs[]` label: basename,
  kind (`file`/`prefix`), mime, the one-line `description` that doubles as the entries'
  `desc`, and a compact single-element `example` of the file's contents. It is static per
  `TOOL_VERSION` — the file ships in the gateway zip (`build-deploy-s3-dist.sh` pins the
  member, and a missing file fails the build), so version and contract cannot drift —
  which is why it lives here and not in `get_job_status`: a caller caches it once per
  session, and CI can pin it without running a job. Contract keys == emitted labels,
  exactly. **The contract also publishes the integration boundary**: anything not
  listed — everything under `ephemeral/` (`face_tracks.json`, `identity_clusters.json`,
  …), `media_processing/`, `_facestore/` — is internal and may change without notice;
  integrating against those paths is unsupported (a real consumer once built against
  `ephemeral/` files instead of `facegroup.json`/`face_clusters.json` and broke
  silently). The examples were authored from the producing code
  (`states/finalize_outputs`, `states/usage_tracking`), not the docs — note
  `facegroup.json` people[] entries repeat the facemetadata person fields and add
  `tracks[]` with per-face boxes (`{l,t,w,h}` normalized frame fractions,
  `source: detected|interpolated`), and `usage_tracking.json`'s `states` is a dict keyed
  by state name, not the siblings' list.

Phase 2 tools ([gateway_tool/index_tools.py](gateway_tool/index_tools.py), §19) — sync:
`create_collection` (foreign-collection refusal + drift warning), `list_collections`
(FaceCount vs record count = reconciliation probe; sandbox-* hidden), `list_faces`
(collection-index GSI, filters origin/named/workflow_id/celebrity_name, base64
next_token), `get_face` (late-binding name resolver; answers from the tombstone for
deleted faces), `batch_update_faces` (rename ≤500/call, DDB-only). Async (start the
face-index SM, poll via `get_job_status`): `batch_index_faces` (s3_prefix XOR
manifest_s3uri; prefix mode generates a manifest — the audit record),
`batch_delete_faces` (face_ids XOR celebrity_name), `delete_collection` (default
collection requires `force=true`). `get_job_status` detects index-SM executions by ARN
and parses their report shape (`num_indexed/num_skipped/num_failed/num_deleted`,
`report_uri`).

Agent system prompt:
[../media-toolkit-gateway/prompts/face_recognition_agent.md](../media-toolkit-gateway/prompts/face_recognition_agent.md).

## CloudFormation stack

[cfnstack.yaml](cfnstack.yaml). The `%%STATE_MACHINE_DEFINITION%%` /
`%%INDEX_STATE_MACHINE_DEFINITION%%` tokens are replaced by the build script with
definitions.json / definitions_index.json (indented into the `!Sub |-` blocks).

### Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `ResourcePrefix` | — | Resource name prefix |
| `ProxyBucket` | — | Frames + outputs bucket (validated/created by ProxyBucketValidator) |
| `MediaProcessingGatewayLambdaArn` | — | MP's gateway tool Lambda; the held `Run media processing` state submits `process_media` through it with a callback token + `idempotency_token=facerec-{uuid}` (MP names its own run). MP's cost is read from its published `usage_tracking.json` |

`DefaultCollectionId` (`face-recognition-on-aws`) is a template Mapping, not a stack
Parameter — baked in at build time, not overridable per deployment (callers override
per-request via `collection_id`, including `""` to disable). `MaxConcurrentExecutions`
and `DetectMapConcurrency` are not stack-level at all — see
`input_config.max_concurrent_executions` / `input_config.detect_map_concurrency` (§13);
the Map states read `detect_map_concurrency` at runtime via ASL `MaxConcurrencyPath`.

### Lambda variants

| Variant | Memory | States |
|---|---|---|
| `1-core` | 1792 MB | control states, semaphore handlers, terminal handler, finalize, usage, **build_face_tracks** (pure dict/list CPU), **search_index_faces** (1 pyvips crop/cluster; API waits are serial except the bounded 5-thread GetCelebrityInfo fan-out) |
| `3-cores` | 5376 MB + 2 GB ephemeral | triage, probe/detect/recognize Map workers, clustering, debug images — states that THREAD CPU-bound pyvips work across cores (libvips releases the GIL; CPU-bound Lambda cost is ~size-neutral, but paying 5376 MB for API-wait-dominated states is not) |
| `gateway-tool` | 256 MB | AgentCore dispatcher (python3.13 zip, no container) |

### No IngestBucket parameter

Deliberate deviation from siblings: face-recognition Lambdas never read the source
asset — the media-processing child does all source I/O under its own role.

### DeletionPolicy decision

The DDB table deliberately carries **no `DeletionPolicy: Retain` and default SSE**
(AWS-owned key): it holds rebuildable derived state. Recognition-run records are
re-derivable by re-running the workflow, and curated index batches re-create their
records from their manifests (the manifest is the audit record). This is sample posture —
a deployment whose persistent collection becomes a system of record should add `Retain`
and a CMK.

### Stack outputs

| Output | Purpose |
|---|---|
| `GatewayToolLambdaArn` | Pass as `FaceRecognitionToolLambdaArn` to media-toolkit-gateway |
| `StateMachineArn` / `StateMachineName` | Direct SFN access |
| `IndexStateMachineArn` | Face index management state machine (§19) |
| `FaceTableName` | DDB table (faces + semaphore) |
| `ImageUri` | ECR image |

## Build & deploy

```bash
bash ./build-deploy-s3-dist.sh --bucket MY_BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]
bash ./create_update_stack.sh --stack-name my-face-recognition \
    --resource-prefix myprefix --proxy-bucket my-proxy-bucket \
    --media-processing-gateway-arn arn:aws:lambda:...:function:my-media-processing-gateway-tool
```

Deploy order: media-processing stack (callback-capable — the submit handler refuses when
`describe_tool.features` lacks `callback_task_token`) → this stack →
media-toolkit-gateway update with `--face-recognition-lambda-arn`.

Artefacts: `face-recognition-on-aws-{VERSION}.zip` (CodeBuild source),
`face-recognition-gateway-tool-{VERSION}.zip` (flat zip incl. bucket_validator.py),
`face-recognition-stack.template` (tokens substituted + ASL injected).

## Testing

```bash
python3 test/test_units.py      # the suite: schema, hashing, tracking, quality,
                                # MDS, semaphore transactions, policy matrix,
                                # delete-and-ignore, celebrity plurality vote,
                                # two-tier merge — mocked boto3, no pyvips
python3 test/test_imaging.py    # layout solver + cell mapping (pure math);
                                # compose/crop tests skip without pyvips
```

Gateway semantics (workflow_id idempotency, GDPR toggles, the token contract) are
covered by the unit suite; end-to-end verification is deployment-only (§10 dev
checklist). `TestGatewayNoTokenEcho` pins the ABSENCE of any echo across all three
response shapes — including that `_correlation_uuid()` no longer exists — and
`TestGatewayIdempotencyToken` pins the token semantics (generate-when-absent,
hash-not-name, never forwarded).

## Gotchas

- **`execution_arn` comes from the ASL** (`$$.Execution.Id` in every Rekognition-span
  state's Payload) — never reconstructed from env + uuid. The workflow Lambdas carry no
  `STATE_MACHINE_ARN` env var; only the gateway tool does (StartExecution). The sandbox
  collection name still derives from the event `uuid`, which the gateway guarantees
  equals the execution name.
- **An absent `idempotency_token` is generated here, so the default is a fresh run.**
  (The parameter was once spelled `uuid`; the retired spelling is a rejection on
  submit.) Every sibling workflow's parameters *name the asset* it processes, so they
  fully determine its output and hashing the parameters alone is the right dedup key — a
  token there only distinguishes two otherwise-identical requests. This tool is not tied
  to one asset: it reads and writes a **persistent, mutable face collection**, so
  identical parameters can legitimately need to run again — *"recognize this video again
  now that the collection knows more faces"* is a real request, and under parameter-only
  dedup it was impossible to express without perturbing an unrelated parameter. So
  `recognize_faces` generates a uuid4 when the caller supplies none (**no token means
  "do not deduplicate me"**), folds the token into the execution-name hash, and a caller
  who wants retry safety supplies a stable token and gets exactly that. Two consequences
  worth stating plainly: the token **narrows** dedup rather than replacing the parameter
  hash (same token + different parameters still re-runs, so a reused token can never
  return the wrong video's faces), and **no token is ever echoed**, supplied or
  generated — a caller who supplied one already has it, handing back a generated one
  would be correlating on noise, and `workflow_id` already identifies the run. The cost
  this exposes is real and accepted: a fresh run burns a semaphore slot and billed
  Rekognition calls; a caller who cares supplies a token, and a caller who supplied none
  has asked for a new run. Regression: `test/test_units.py: TestGatewayIdempotencyToken`.
- **The token echo is gone entirely, and that is the fix — do not "restore" it.**
  Removed after a live bug whose whole blast radius was the echo itself:
  `_parse_succeeded_output` read `payload.get("correlation_uuid")` from the execution
  OUTPUT, a key no state ever writes, so it always resolved `None` and fell through to
  `payload.get("uuid")` — which must equal the execution name here, so a SUCCEEDED
  **poll** echoed the derived `workflow_id` instead of the caller's token. Two things
  hid it: `_attach_workflow_id` (RUNNING/FAILED) already read the input and was correct,
  and the submit path patched the parser's gap with a `setdefault` backstop
  `get_job_status` doesn't have. The repair first routed all four shapes through one
  `_correlation_uuid(resp)` reader; the follow-up deleted the reader and the echo with
  it, so the three surviving response paths agree by construction rather than by sharing
  a reader. Regression: `test/test_units.py: TestGatewayNoTokenEcho`, which asserts no
  response carries a `uuid`, that a stale `correlation_uuid` planted in either the input
  or the output is not surfaced, and that `_correlation_uuid` no longer exists as an
  attribute.
- **Rekognition TPS**: defaults sized for 100 TPS DetectFaces / 50 TPS SearchFaces
  (us-east-1). Low-limit regions: pass a lower `detect_map_concurrency` per request
  (default 4, no redeploy needed).
- **`max_holders` only ratchets up**: `max_concurrent_executions` (default 5) is a
  per-request value, but the stored DDB cap it feeds never decreases — once any caller
  requests a higher value, that's the new floor for everyone until the table is
  recreated (or goes idle past the counter TTL, §14).
- **RecognizeCelebrities is not available in every region** —
  `celebrity_recognition: false` is the escape hatch.
- **pyvips wheel**: `pyvips[binary]` must resolve for cp313/aarch64 in the container.
  Fallback documented in §3 (Pillow `draft()`), requires swapping
  frame_triage/grid_image imaging calls.
- **Semaphore admission is size-blind**: five long jobs head-of-line-block shorter ones
  (accepted for v1; §12).
- **Re-runs re-index**: since an omitted or changed `idempotency_token` starts a fresh
  run, a re-run with indexing enabled may re-index already-indexed people (mitigated by
  search-before-index). Exploratory runs should pass `max_faces_to_index: 0` — and
  supply a stable token if the intent is a retry rather than a re-run.
- **Usage-tracking cost model: a state with `estimated_cost: 0.0` is not a bug.** An
  earlier state-span model priced `TaskStateEntered`→`TaskStateExited` wall time for
  EVERY state at the largest memory seen in the run, over-reporting Lambda cost more
  than 2× on real runs via three distinct inflations: (1) `Map` envelopes charged on top
  of the iterator invocations they contain; (2) the held `Run media processing` state
  priced as Lambda compute, double-counting the child whose cost is already added as
  `child_cost`; (3) `Acquire semaphore` is `waitForTaskToken`, so its span is queue
  wait, not compute — unbounded under contention, which is the semaphore's whole job.
  The fix tags every state with a `kind` (`_KIND_LAMBDA` / `_KIND_CHILD` /
  `_KIND_TOKEN` / `_KIND_MAP`) and bills only `lambda`, each attempt at its OWN
  function's memory; the `_KIND_*` block in `states/usage_tracking/lambda_function.py`
  records why each non-Lambda kind is not this stack's compute. Every state still
  appears in the report with its span, because that span is the only measurement those
  states have — a model with no representation for a reported-but-unbilled state would
  have deleted the child's and the semaphore wait's only numbers. The
  `waitForTaskToken` compute is genuinely unrecoverable from execution history (no
  event for the Lambda's return), so `Acquire semaphore` under-states by a negligible
  amount — reporting the wait honestly and billing 0 is the accurate choice, not an
  oversight.
- **`wall_sec` is span-derived, a deliberate divergence from the siblings.** They sum
  attempts per `entered_id`; this stack takes the enclosing
  `TaskStateEntered`→`Exited` span, which also covers the gap between a failure and its
  re-schedule — both correctly cover all attempts, the span just reads slightly longer.
  This is why the cross-stack test (`../micro-workflow-bundle/test/test_units.py:
  TestUsageTrackingBillsFailedAttempts`) asserts `wall_sec >=` the attempt sum rather
  than an equality. Do not "fix" the inequality into an equality.
- **Usage-tracking cost constants** are pinned us-east-1 arm64 figures — re-check on
  pricing changes.
- **mark_in_out frame numbering**: MP's clipped output does NOT restart frame numbers at
  0 — both pipelines speak source-absolute frame numbers (verified; §21), so no mark_in
  offset is applied.

### Silent-default bugs (found on the first real deployment)

Both were caught by inspecting actual S3 output against a real asset, not by unit tests
(the tests that would have caught them were written afterward as regressions — see
`test/test_units.py: TestResolveFps`, `TestRecognizeUsageAggregation`). The common
shape: a `try/except Exception` swallowed a real failure and fell through to a
plausible-looking wrong value with no warning visible anywhere in the output.

1. **fps silently defaulted to 30000/1001 for every asset.** `_resolve_fps` read
   `probe.get("streams")`, but media-processing's actual probe.json schema nests ffprobe
   output under `probe.get("probe_info").get("streams")`. The top-level lookup always
   returned nothing, the bare `except Exception` swallowed the resulting no-op, and the
   function fell through to a hardcoded `Fraction(30000, 1001)`. On a 24000/1001 asset
   every `timestamp`/`begin_ms`/`end_ms` in facemetadata.json was wrong by 1.25×.
   **Fixed**: `_resolve_fps` reads `probe_info.streams` and raises `ValueError` (no
   probe_key, no video stream, unparseable rate) instead of ever returning a default — a
   wrong timestamp is worse than a failed job. `frame_meta` also stores the exact
   `"frame_rate": "24000/1001"` string alongside the `"fps"` float convenience value —
   any future arithmetic consumer must parse `frame_rate` via `Fraction()`, never trust
   the float.
2. **`recognize_celebrities_api` always reported 0 despite real, billed
   RecognizeCelebrities calls.** The count lived only in `event["data"]["iterators"]` →
   `recognize_N.json` files, and `cluster_identities` nulls that list before
   `usage_tracking` ever runs. **Fixed**: `cluster_identities._collect_celebrity_tags`
   also sums and returns the recognize iterators' usage, folded into
   `cluster_identities`'s own counter exactly once on the FIRST attempt (not on
   checkpoint resume, where it would double-count — the recognize iterators are a fixed,
   already-completed input that never changes across retries of this state).

**Broader fix**: every `except Exception` doing checkpoint/resume duty
(`prepare_probe_iterators`, `cluster_identities`, `search_index_faces`) was narrowed to
`lib.s3_utils.get_json_if_exists`, which returns `None` ONLY for `NoSuchKey`/`404` and
re-raises everything else (throttling, access denied, corrupted JSON). A broad catch
there previously meant a real S3 failure would be misread as "no checkpoint yet,"
silently restarting already-billed work. `usage_tracking._sum_api_counts` had its
try/except removed entirely — its three keys are only ever added after the producing
state confirmed a successful write, so a read failure there means real usage data is
missing from the cost report, and that must raise, not log-and-continue.

### Celebrity naming picked the luckiest wrong guess, not the honest one

Found on a 2-hour rolling-news broadcast test. `cluster_identities._cluster_celebrity`
tracked the single **highest-confidence** `RecognizeCelebrities` tag across all of a
cluster's tracks. For a non-celebrity news anchor tracked across hundreds of fragmented
track observations, RecognizeCelebrities returned a WRONG celebrity name for most calls —
almost never the same wrong name twice (dozens of distinct wrong names, all real public
figures unconnected to the broadcast), at a mean confidence around 83 — well below a
genuine match's mean (~99: a correctly-recognized anchor in the same broadcast landed
the SAME right name in nearly every look). With hundreds of independent noisy guesses,
one lucky outlier cleared `NAME_VOTE_INSTANT_SCORE` (95.0) and got permanently
auto-indexed into the persistent collection under a stranger's name — even though
"unrecognized" was the actual plurality outcome. Visual audit of the run's other
high-vote-share "matches" confirmed the same failure mode across every non-white
demographic in the broadcast, consistent with the well-documented finding (Buolamwini &
Gebru, "Gender Shades") that commercial face-recognition reference sets skew toward
lighter skin tones, so both confidence and specificity degrade for every
under-represented group. **Fixed**: `_cluster_celebrity` tallies votes per name across
the cluster and returns the **plurality** name's **mean** confidence, not the single
luckiest sample — `_vote_name`'s existing thresholds
(`NAME_VOTE_INSTANT_SCORE`/`NAME_VOTE_MIN_SCORE`) were already correctly calibrated to
reject ~83-mean-confidence noise; the bug was only ever in what confidence value reached
them. No threshold changed. Verified by replaying real runs' recorded data: every
confirmed false positive dropped, every high-confidence genuine match preserved.
Regression: `test/test_units.py: TestClusterCelebrity`. (§15.)

### Cross-person cluster contamination: single weak edge merged two people

Union-find in `cluster_identities` merged identity groups on ANY single SearchFaces edge
≥ `SANDBOX_SEARCH_THRESHOLD` (75) — and union-find merges are transitive and
irreversible. On real content, an unrelated person's track reached a large genuine
cluster through a chain of three 75-77-similarity noise links stitched between genuine
97-100 links, putting his face in her montage; the same defect fused two genuine actors'
clusters (both sides carrying hundreds/dozens of same-name votes at 98-99 mean) on
episodic content. Edge-score distributions across all replayed runs separate cleanly:
genuine same-footage edges overwhelmingly score ≥95, while every observed cross-person
bridge lived in the 75-85 band. **Fixed** (two-tier merge): a single edge unions only at
≥ `SANDBOX_MERGE_SIMILARITY` (90); weaker edges [75, 90) union only when ≥
`SANDBOX_WEAK_CORROBORATION` (2) independent weak edges connect the same two components
(`_merge_corroborated_weak_edges`, iterated to fixpoint), and skip-if-grouped no longer
lets a weakly-linked face skip its own search. Replaying all recorded runs' edges
through the production merge code: zero contaminated components, zero fragmented genuine
identities. Regression: `test/test_units.py: TestTwoTierMerge` (includes the exact real
chain shape). (§16.)

### Face image resolutions and aspect ratio

- **Stored derivatives** (§19 semantics): the thumbnail is a 3:4 portrait face crop
  (`FACE_THUMBNAIL_DIM` 96×128 — the initial port used 64×64 squares, the wrong call:
  face boxes are naturally portrait). The full_image is NOT a face crop — it is a
  fit-inside downscale of the WHOLE source image (`FULLIMAGE_MAX_EDGE_PX` 1280), with
  the record's normalized `coord` `[x,y,w,h]` locating the face in it; one downscaled
  frame can serve several faces, and it survives frame-prefix expiry. `crop_headshot` is
  aspect-preserving (expands the crop rect to the target ratio — never distorts).
- **The persistent IndexFaces image is native resolution** (`crop_face_native`, cap
  `INDEX_CROP_MAX_EDGE_PX` 1024, never upscale): the old fixed 180×240 tile resize threw
  away ~4× linear resolution on close-up faces before Rekognition ever saw them — the
  API takes 15 MB via S3Object, so shrinking bought nothing. The 180×240 tiles remain
  for the *search* grids (16 tiles must fit one image; SearchFacesByImage only needs to
  find the match).
- **Don't inflate stored sizes beyond source information**: derivatives crop from the
  proxy frame (960×540 for SD/720p sources; up to 1920×1080 per locked decision #11 for
  HD+). A 640-long-edge full image is already at/above the information ceiling of a
  median face crop — bigger numbers would just upscale blur.

### Rejected: pre-index sharpen/brighten enhancement (reference-pipeline feature, empirically useless-to-harmful)

The reference pipeline sharpened (sigma from Rekognition sharpness ≤30) and
brightness-modulated (brightness ≤30 or ≥70) face crops, then percentile-normalized,
before indexing/searching. The exact formulas were ported and A/B-tested against real
Rekognition with real broadcast faces at three difficulty levels: same-source
poor→clean pairs were identical; on the stress tier — the pipeline's worst faces (~40 px
wide, near the sharpness floor) vs a studio-headshot collection — enhancement made
similarity WORSE in nearly every case, and would have fired on the large majority of
index picks: near-constant distortion for negative return. Modern Rekognition embeddings
are robust to blur/exposure; artificial sharpening adds halo artifacts the embedder
reads as face detail. NOT ported; if a future model version changes this, re-run the A/B
(protocol in §17) before enabling.

### Pose gate for indexed faces (soft rank penalty lost to area dominance)

`fusion_rank = w*h*(1-pose_dist)` treats pose as a soft multiplicative penalty — on real
content the top index candidate for one cluster was a **yaw=-25° 3/4 profile** because
its box area outweighed a frontal alternative's. `best_faces` now hard-gates on
`|yaw| ≤ 20°` and `|pitch| ≤ 20°` (`POSE_MAX_ABS_YAW`/`POSE_MAX_ABS_PITCH`), plus
non-occluded and not-eyes-closed, before ranking — with fallback to the ungated pool
when a cluster has no frontal face (measured pass rates showed the gate narrows without
emptying). Roll is deliberately not gated (in-plane rotation is normalized by
Rekognition). For `limit=1` (the persistent-index pick) the top-3-by-rank are refined by
sharpness, mirroring `choose_representative` — biggest is not always crispest.
Regression: `TestFaceQuality.test_pose_gate_*`.

### Removed: absolute brightness/sharpness quality windows (miscalibrated AND skin-tone-correlated)

Measured against a real broadcast's full face population: the `SHARPNESS_RANGE` floor
(≥30) rejected the overwhelming majority of all faces (population median far below it),
and the `BRIGHTNESS_RANGE` ceiling (≤70) rejected nearly all of the *lightest-lit*
people's faces while passing most of a dark-skinned anchor's — so when the window bound
at all, `best_faces` selected light-skinned people's darkest frames. Rekognition's
`Quality.Brightness` measures illumination but correlates with skin reflectance (same
set, same lighting: measurably higher means for lighter-skinned subjects), so ANY
absolute brightness gate treats equally-well-lit faces differently by skin tone. The
`CENTROID_QUALITY_TIERS` were equally miscalibrated (even the loosest tier passed only a
small fraction), meaning the fall-through path fired almost always. **Fixed**: selection
ranks RELATIVELY within the candidate pool (area × frontality, then sharpness) —
skin-tone-neutral by construction; the only absolute gates kept are extreme floors
(`QUALITY_MIN_BRIGHTNESS` 10 for near-black crops, `CENTROID_MIN_SHARPNESS` 2.4), plus
Rekognition's own `QualityFilter` at IndexFaces time, which is the real quality gate.
`CENTROID_QUALITY_TIERS`/`SHARPNESS_RANGE`/`BRIGHTNESS_RANGE` are gone, not just unused.

### Removed: byte-size frame pre-filter (unvalidated heuristic, not a silent-except bug)

The original triage had a zero-decode tier 0: drop a frame whose JPEG byte size (scaled
by pixel count) fell below a threshold, on the theory that black/near-solid frames
compress far smaller. **Removed after inspecting real S3 output**: a frame with a clear
in-focus face fell under the scaled threshold and was silently dropped before triage or
detection ever ran, while genuinely-black frames from the *same dark scene* were far
smaller still — no byte-count threshold separates those two cases, because JPEG size
tracks scene complexity (texture, grain, motion blur), not face presence. The threshold
value itself also had no empirical basis — an invented constant, not a measured one.
Every frame is now decoded and classified from actual pixel content
(`lib/frame_triage.triage_luma` — mean/p99/stddev/Laplacian variance); the removed
`TRIAGE_MIN_JPEG_BYTES_AT_REF`/`TRIAGE_REF_PIXELS` constants and
`size_below_threshold()` are gone, not just unused.

---

# Design record

The port's design history: decisions with their reasons, review findings, incident
root-cause analyses, and rejected alternatives. Section numbers are stable — code
comments cite them as `README.md §N`.

## §1 Port context and repo layout

Ported from a ~4,600-line Node.js reference pipeline (Rekognition-based face detection →
celebrity recognition → identity clustering → persistent-collection search/index) to a
standalone Python micro-workflow following the family patterns, exposed via AgentCore
Gateway MCP tools. The old pipeline was coupled to its host system's ingest
(pre-extracted frames with shot IDs, a backlog system, webapp-shaped outputs); the port
is fully standalone. Objectives: scalability (internal throttle handling — semaphore),
fast once started (Map parallelism), cost-optimized (minimize Rekognition calls WITHOUT
reducing output quality/quantity — grid images + probe-then-fill), data visualization
(face-cluster layout), resiliency, serverless.

Locked decisions:
1. No shot dependency. Frames extracted at 1 fps (configurable) by a media-processing
   child; record actual source frame number + timestamp.
2. ML backbone stays 100% Amazon Rekognition. No local models — imaging via **pyvips**
   (`pyvips[binary]` wheel; Pillow fallback documented) + numpy for hashing/MDS.
3. Outputs: facemetadata.json (flat `people[]` — §6), facegroup.json,
   usage_tracking.json, face images, debug montages, face-cluster visualization (MDS
   from SearchFaces similarities — replaces the old t-SNE). No VTT, no legacy
   raw/redirect/timeseries shapes.
4. Stack owns its identity store: its own DynamoDB table + its own Rekognition
   collection (auto-created).
5. DynamoDB counting semaphore limits concurrent executions (default 5), modeled on a
   sibling's lock (waitForTaskToken + pending queue + stream drain + EventBridge failure
   backstop).
6. Face-indexing workflow = phase 2 (now implemented; §19–§20).
7. **Probe-then-fill detection**: detect every Kth frame first; skip confirmed-empty
   static regions; dense-detect face regions with near-duplicate dedup + box
   interpolation.
8. **Semaphore held only around Rekognition work** — acquire AFTER the media-processing
   child (MP handles its own throttling; holding a slot through a transcode wastes
   capacity).
9. **Failure release via EventBridge + TTL only** — no per-state Catch blocks (cleaner
   ASL; async-delivery tradeoff accepted, TTL backstop).
10. **Self-checkpoint re-enter**: workers monitor remaining Lambda time, persist partial
    results, throw `LambdaResourceTimeoutException`; SFN Retry re-invokes and completed
    work is skipped. Removes any content-length ceiling.
11. **Frame extraction targets 1920×1080, fit-inside, never upscale** (Rekognition bills
    per image, not per pixel — higher-res cells are free on the API side and preserve
    40–79 px faces a 960×540 downscale would lose). MP's `scale_dimension` implements
    the exact policy, portrait handled via axis swap.
12. **3×3 patch + 1 global dHash per frame**: per-patch change tracking lets the skip
    predicate discount chronically-noisy patches (lower-third tickers, scoreboards,
    clocks) that would otherwise force dense detection of static news/sports anchor
    shots.

Rekognition constraints (verified): image ≤ 10000×10000 px and ≤ 15 MB via S3Object
reference (5 MB only for inline bytes — so grids are uploaded to S3 and passed by
reference); DetectFaces ≤ 100 faces/image; RecognizeCelebrities ≤ 64 celebrities/image;
IndexFaces ≤ 100 faces indexed/image; min detectable face ~40 px (relative to the
image's long edge — §20 addendum). TPS defaults: DetectFaces 100, SearchFaces 50,
IndexFaces 50, RecognizeCelebrities 50. Cost $0.001/image API call.

Layout: `states/<name>/lambda_function.py` per state; `lib/` is the shared package;
`gateway_tool/` ships its own flat zip; mocked-boto3 unit tests under `test/`.

## §2 State machine and failure paths

One linear ASL (see the state list above) — no Choice states. **Zero-face early exit**:
if the probe pass + plan find no faces anywhere (pure B-roll), the remaining iterator
lists are empty and every downstream state short-circuits (states handle empty input);
whole job ≈ probe cost only.

**Failure paths — EventBridge + TTL only, NO per-state Catch blocks**:
- EventBridge rule on ALL terminal states — SUCCEEDED, FAILED, TIMED_OUT, ABORTED →
  `handle_execution_terminal`: (1) conditional-delete the holder item (no-ops when
  already released or when failure preceded acquire); (2) delete `sandbox-{workflow_id}`
  — the single, uniform sandbox cleanup path: fire `DeleteCollection` directly and
  catch/ignore `ResourceNotFoundException` (no existence pre-check, saves an API call).
  One rule covers success, failure, and abort identically.
- Per-holder TTL (3600 s, refreshed on Rekognition-state entry — §5) is the backstop if
  EventBridge delivery itself fails.
- Accepted tradeoff: async delivery means ~seconds of slot lag after a failure.

**Map retry + self-checkpoint re-enter**: workers checkpoint per-grid results as they go
(batched — a per-grid PUT of a growing JSON would be O(n²) S3 bytes; superseded by O(n)
side objects, §18), monitor `get_remaining_time_in_millis()` (60 s buffer), and on
approaching timeout persist state and raise `LambdaResourceTimeoutException`. ASL Retry:
`IntervalSeconds: 2, MaxAttempts: 10, BackoffRate: 1` — re-entry skips completed work.
Rekognition throttle/5xx retry 6× @2s backoff 1.2; `Sandbox.Timedout, Lambda.Unknown`
retry 2× backoff 2.0. The same treatment applies to the long single Tasks
(`cluster_identities`, `search_index_faces`): cluster_identities checkpoints its
union-find state and re-enters WITHOUT recreating the sandbox collection;
search_index_faces checkpoints per-group results (DDB registrations are naturally
idempotent by face_id).

### §2b Recognition policy matrix (GDPR use cases)

Deployments under GDPR may not persist biometric data (auto-index into a durable
collection). **Compliance line adopted: GDPR storage-limitation governs PERSISTENT
biometric templates, not transient processing** — DetectFaces itself derives facial
geometry transiently, and the sandbox collection (created per execution, deleted at
terminal state) is transient processing state in the same category as mid-pipeline
frames on S3. The sandbox therefore stays available in ALL use cases; only writes to the
PERSISTENT collection are policy-gated. Three orthogonal toggles compose every use case;
enforcement is inside the state handlers (no ASL branches):

| Use case | celebrity | collection_id | max_index | Result |
|---|---|---|---|---|
| 1. Talent tracking | **false** | customer's | **0** | pre-indexed talent matched via SearchFacesByImage; nothing new indexed; no celebrity calls |
| 2. No persistent collection | true | **`""`** | (ignored) | detect + track + full sandbox clustering + celebrity names; zero persistent biometric writes |
| 3. Celebrities + pre-indexed | true | customer's | **0** | full pipeline, read-only persistent collection |
| 4. Everything | true | any | >0 (+ `index_named_faces_only` optional) | full pipeline + curated auto-index |
| 5. Detection-only | **false** | **`""`** | — | faces, tracks, anonymous `Person_N` clusters, screen time, boxes; zero names, zero persistent writes. Pairs with `dedup_hamming_threshold=0` for redaction prep |

Because the sandbox is always available, clustering quality and the MDS
`face_clusters.json` visualization are identical across all use cases — no degraded
name-only fallback needed. **Transient-by-design guarantee**: sandbox deletion is the
EventBridge terminal handler; residual risk is a delivery failure on a never-resubmitted
workflow_id — under this posture a daily `sandbox-*` sweeper is required, not
nice-to-have (still deferred; the gap is the EventBridge-delivery-failure window only).
Validation: `index_named_faces_only=true` + `max_faces_to_index=0` → warning in the
response (no-op combination). GDPR Art. 17 erasure: §19's `batch_delete_faces`.

## §3 Probe-then-fill detection (cost core)

Two-pass design; detection results are the skip-oracle, hashes are the safety net.
Skipping is aggressive only where it is free (no-face regions); face-bearing regions
always get dense detection. See the Detection strategy section above for the predicate.

- Triage decodes EVERY frame — the one hot loop where the imaging library matters.
  Mitigations, all three: (a) pyvips `thumbnail_buffer` (shrink-on-load JPEG DCT fast
  path, demand-driven, several times faster than full Pillow decode with lower peak
  memory); (b) ThreadPoolExecutor over frames (S3 GET + decode overlap — libvips
  releases the GIL, so threads scale); (c) checkpoint/re-enter as the backstop.
  Fallback: Pillow `draft()` keeps the budget acceptable, so pyvips is an optimization
  choice, not a dependency risk.
- Patch noise classification: per patch, change frequency across the whole asset;
  patches changing in > ~70 % of adjacent pairs are chronically noisy and excluded from
  the stability predicate. Guards: never discount more than 3 of 9 patches (more churn =
  genuinely dynamic content); classification recorded in detection_plan.json; probe
  endpoints' NO-FACE verdicts are unaffected — the discount only relaxes hash stability,
  and a face appearing inside a discounted patch is still caught by the probe cadence.
- Near-duplicate dedup inside dense regions: skip frames within
  `dedup_hamming_threshold` (default 4, 0=off) of the last kept frame (stable-patch
  distance), max skip-run capped at probe_stride. Probe results are reused — **worst
  case total calls = single-pass baseline**; typical content lands well under it.
- Skipped-frame box reconstruction: within a track, linear interpolation across
  dedup-skipped runs (copy-forward at track end); every box carries
  `source: detected|interpolated`. Valid by construction — frames are only skipped when
  hash distance says the scene barely moved. Consumers needing exactness (redaction) set
  dedup 0.
- Dropped frames recorded (frame_num + reason) in facegroup.json for auditability.

## §4 Per-state responsibilities and data contract

Event shape: `{uuid, input: {s3uri, input_config}, output: {s3uri}, data: {}}`.
`output.s3uri` = service-derived prefix. Intermediates under `{prefix}/ephemeral/`;
grids under `{prefix}/grids/`.

| State | Role |
|---|---|
| prepare_media_processing | builds the MP gateway request (jpeg `face_frames` group @ `frame_interval_sec`, fit-inside 1920×1080) |
| run media processing (held) | submits via MP's gateway tool with a callback token; `extract_processed_frames` normalizes the result handle |
| prepare_probe_iterators | manifest + probe.json (exact fps as Fraction) → triage + patch hashes → `ephemeral/frame_index.json`, probe grids, `data.frame_meta`; validates the optional shots supplement (§21) |
| probe_detect_faces (Map) | grid → S3 → DetectFaces(S3Object) → faces mapped to cells |
| plan_dense_detection | skip/dense classification (§3) → dense grids (may be []); skip map in `ephemeral/detection_plan.json` |
| detect_faces (Map) | as probe worker; grid-split safeguard: ≥95 faces → split grid in half, re-detect |
| build_face_tracks | merge faces; tracking = same gender AND motion-tolerant IoU (dilate previous box by `1 + 0.5·gap_sec`) AND gap ≤ `max_track_gap_sec` (3.0 s) AND no cut break between the frames (dHash veto; shots UNION — §21/§23). Strict overlap breaks tracks for moving subjects at 1 fps sampling, and gender-only matching over a 3 s gap can chain two people through a cut — the hash veto restores the old shot-boundary containment. Box interpolation; noise filter (single-appearance tiny/dark/blurry, thresholds scaled to frame dimension) |
| recognize_celebrities (Map) | headshot grid (120×160 tiles, ≤64) → 1× RecognizeCelebrities; raw response archived under `grids/`; tags tracks |
| cluster_identities | group by celebrity name (or singleton); one qualified centroid/group (fusion rank `w*h*(1-pose_dist)` + pose gate); sandbox collection `sandbox-{workflow_id}` — create-if-absent, reuse on re-enter (workflow_id IS the execution name; SFN name-uniqueness), NO in-state deletion ever (cleanup is exclusively the EventBridge terminal handler). IndexFaces grids (≤100, QualityFilter MEDIUM); SearchFaces per faceId (threshold 75) → two-tier merge (§16) → `face_group_id`. Captures the pairwise similarity matrix from SearchFaces responses (zero extra API calls) for the viz |
| search_index_faces | gated on collection_id. Per group ≥ `min_appearances` (default 3) distinct tracks: plurality-vote name (§15); best-face SearchFacesByImage (native-res single query — §18) + conditional second witness (§23); non-matches auto-index ranked by `_index_rank` (QualityFilter HIGH, cap `max_faces_to_index`), register DDB, store derivatives under `_facestore/{collection_id}/` |
| release_semaphore | conditional DeleteItem of the holder ONLY — placed immediately after the last Rekognition state |
| finalize_outputs | `facemetadata.json`, `facegroup.json`; classical MDS on the similarity matrix (pure numpy) → `face_clusters.json` |
| store_debug_images | per-person montages + `person_NNNN_face.jpg` rep crops (§22) + indexed-faces grid; no Rekognition — post-release |
| usage_tracking | per-attempt Lambda cost + child MP cost (from MP's published usage_tracking.json) + API counters → `usage_tracking.json` |

## §5 Semaphore — per-holder items + transactional counter

Per-holder items, NOT the reference pipeline's counter-only style (one shared TTL
refreshed on every acquire/release — a crashed holder is never individually reclaimed;
release-by-owner impossible). API-level rate limiting was considered and rejected:
concurrency is statically bounded by construction (semaphore × Map concurrency ×
per-API pool bounds, §7), so per-call coordination buys nothing.

Flows — **single decrement path** (key invariant):
- **Acquire**: `TransactWriteItems([Update counter: holders+1, Condition holders <
  max_holders], [Put holder, Condition attribute_not_exists(sk)])`. Success →
  SendTaskSuccess immediately. TransactionCanceled → store pending, wait on token.
  Re-entrant: Put-condition failure on OWN arn → success.
- **Holder TTL refresh** (without this, over-admission): TTL 3600 s bounds a single
  STATE, not the whole Rekognition span — a long asset with re-enters can legitimately
  hold for hours. Every Rekognition-span state refreshes its holder's `ttl` on entry;
  a live job never expires, a crashed one is reclaimed ≤ 1 h after its last state entry.
- **Release** (normal end-state, EventBridge): conditional DeleteItem of holder only —
  no counter decrement here. Idempotent.
- **TTL expiry**: DDB deletes stalled holders.
- **All three deletions converge on the stream**: REMOVE on holder keys →
  handle_semaphore_release. **Exactly-once decrement** (streams are at-least-once; a
  handler crash after decrement + retry would double-decrement and permanently drift
  the counter): the decrement is a transaction with a per-eventID dedup marker
  (`stream-dedup`, TTL 1 h, Condition attribute_not_exists); a retried record trips the
  condition and skips. Then drain loop: query pending, pop min created_at FIFO, attempt
  acquire for that waiter; success → SendTaskSuccess + delete pending (dead token →
  delete pending + release slot to cascade); repeat while slots and waiters remain.
- EventSourceMapping FilterCriteria: `eventName=REMOVE AND pk="semaphore" AND sk prefix
  "holder#"` (stream KEYS_ONLY).

Known bounded unfairness: fresh direct-acquire can race a queued waiter's drain
(accepted; family precedent). Counter TTL self-heal: §14. Request-level `max_holders`:
§13.

## §6 Grids, images, outputs

- **Grid submission via S3Object reference** (10000×10000 px / 15 MB bounds; the 5 MB
  cap applies only to inline bytes). Required for debuggability anyway.
- **Dimension-agnostic grid packer**: cell size = the actual extracted frame dimension.
  Layout chosen at runtime: `cols × rows` maximized such that `cols·cell_w ≤ 9800` and
  `rows·cell_h ≤ 9800`, **capped at `frames_per_grid` = 20** (a review correction from
  an earlier 16 cap, which threw away a quarter of the batching win; the 100-face cap at
  20 cells ≈ 5 faces/cell average is amply guarded by the grid split). JPEG size
  estimated conservatively, hard-verified < 14 MB at compose time, quality stepped down
  (85→75→65) then cells reduced — 15 MB is a hard API bound, not a target. Memory note:
  compose grids sequentially per worker thread, don't hold N canvases.
- **Grid-split safeguard**: ≥95 faces returned → split grid in half, re-detect both
  halves (recursive, min 4 frames/grid). Handles crowd scenes the old pipeline silently
  truncated.
- **Cell separators + bleed rejection**: 2 px black lines between cells (breaks
  cross-cell face continuity so the detector cannot see one face spanning two cells);
  mapped boxes crossing their cell boundary by > 15 % rejected as bleed artifacts
  (§12 — the face is re-detected cleanly in its own cell on a neighbouring frame).
- **Cell mapping**: bbox center → cell index; recalibrate to frame-relative l/t/w/h.
- **Headshot grids**: face box + 30 % margin, resized 120×160, ≤64 tiles.
- **facemetadata.json — reshaped for downstream consumption.** The old shape split
  people across `recognized`/`unrecognized` arrays with per-shot nesting, forcing
  consumers to merge two lists and walk nested faces to answer "who appears when". New
  shape: one flat `people[]` sorted by screen time, appearances first-class, per-frame
  detail relegated to facegroup.json:

```json
{
  "version": 1,
  "collection_id": "...",
  "duration_sec": 7200.0,
  "people": [
    {
      "person_id": 0,
      "name": "Jane Doe",
      "named": true,
      "name_source": "celebrity",
      "confidence": 97.2,
      "face_id": "...",
      "gender": "Female", "age_range": [29, 41],
      "total_screen_time_sec": 312.4,
      "appearances": [ {"begin_ms": 1200, "end_ms": 8400, "representative_frame": 3} ],
      "thumbnail_key": "debug/people/person_0000.jpg",
      "rep_face_key": "debug/people/person_0000_face.jpg",
      "num_tracks": 14
    }
  ],
  "stats": {"frames_analyzed": 5400, "frames_skipped": 1800, "faces_detected": 8123},
  "usage": {}
}
```

- A VTT output was deliberately dropped: timeline data lives in
  `people[].appearances`; any consumer can derive a VTT in a few lines.

## §7 Stack shape and TPS budget

See the CloudFormation section above for the current parameter set (post-§13). Workflow
Lambda role: Rekognition detection APIs (non-resource), collection APIs (Resource `*` —
§13 explains the loosening), DDB CRUD + Query on table + GSIs, S3 rw on ProxyBucket,
states Get/Describe + SendTaskSuccess/Failure, lambda:GetFunctionConfiguration.

**TPS budget** (defaults; `detect_map_concurrency` is the relief valve): 5 executions ×
4 workers × ~1 in-flight ≈ 20 concurrent ≈ 40 TPS peak DetectFaces (limit 100 ✓).
SearchFaces clustering bounded per execution → well under the 50 TPS limit at 5
executions. Three defense layers: semaphore (admission) → static per-API pool bounds →
boto3 `adaptive` retry as shock absorber. The semaphore is held only across the
Rekognition span, so MP transcodes of queued jobs proceed concurrently and a released
slot goes straight to a job whose frames are already staged.

Lambda variant sizing is a measured decision (§18): states that THREAD CPU-bound pyvips
work ride 3-cores (CPU-bound Lambda work is ~cost-neutral across sizes, so cores buy
wall-clock speed for free); API-wait-dominated states ride 1-core (paying 5376 MB for
idle waits is pure waste).

## §8 Gateway tools (target `face-recognition`)

`recognize_faces` — required: `asset_s3_uri`. Optional: `idempotency_token`,
`frame_interval_sec` (default 1.0, 0.2–10), `mark_in_out`, `celebrity_recognition`
(default true), `collection_id` (default from stack; `""` disables persistent ops —
§2b), `max_faces_to_index` (default 0 — auto-index is opt-in), `index_named_faces_only`
(default false), `min_appearances` (default 3), `min_confidence` (default 80),
`probe_stride` (default 4, 0 = single-pass dense), `dedup_hamming_threshold` (default 4,
0 = off; set 0 for frame-exact boxes, e.g. redaction), `shot_segmentation_s3uri` (§21),
`max_concurrent_executions`, `detect_map_concurrency` (§13). Width/height not exposed
(fixed 1920×1080 fit-inside). The tool description documents the §2b use-case recipes so
agents can map constraints to toggles.

`get_job_status` SUCCEEDED fields: `workflow_id, asset_s3uri, output_s3_prefix,
num_people_recognized, num_people_unrecognized, num_faces_indexed, collection_id,
duration_sec, frames_analyzed, frames_skipped, pipeline_wall_sec, total_estimated_cost,
usage_key` + outputs[] (labels in the gateway section above). Registration in
media-toolkit-gateway follows the optional-target precedent: `FaceRecognitionToolLambdaArn`
parameter default `""` + Condition, conditional target + permission + output.

## §9 Phase 2 — original scope sketch

The first phase-2 sketch (synchronous `index_face`/`list_faces`/`delete_face`, no second
state machine) was superseded by the reviewed spec in §19 — batching, idempotency, and
right-to-erasure ordering all demanded a state machine. Bulk import from an existing
external collection: out of scope.

## §10 Build, deploy, verification

- Build: docker zip = `.version lib states Dockerfile lambda_function.py
  requirements.txt`; gateway zip = `lambda_function.py index_tools.py
  bucket_validator.py output_contract.json`.
- Deploy order: media-processing stack → this stack → media-toolkit-gateway.
- Local tests (mocked boto3): semaphore transactions (acquire-to-max, FIFO drain,
  double-release, TTL-reclaim, re-entrant, stream-retry dedup, TTL refresh); grid layout
  solver + cell mapping round-trip; bleed rejection; grid-split; patch dHash (zoom-drift
  promotes, static skips, entrance breaks, ticker discount, >3 noisy patches → no
  discount); skip-predicate conditions; box interpolation; tracking (gap-dilated IoU,
  gender, hash-break veto); plurality vote; MDS sanity; workflow_id idempotency;
  `LambdaResourceTimeoutException` re-enter; facemetadata shape; the §2b policy matrix
  (each toggle combination → correct API-call set, asserted via a mocked call recorder).
- Dev end-to-end checklist: multi-person asset; oversubscribed concurrency (queue +
  drain); kill one mid-run (EventBridge reclaim); B-roll-only asset (early exit);
  zoom-in clip (drift promotion); news clip with lower-third ticker (anchor shots skip
  despite churn); portrait 9:16 asset; policy use cases 1 and 2 (verify zero
  IndexFaces / zero collection calls in usage counters); outputs[] presign.

## §11 Explicitly rejected alternatives

- **API-level rate limiter**: concurrency is statically bounded by semaphore × Map ×
  pool bounds; per-call DDB coordination adds cost and a failure mode without changing
  the worst case.
- ~~pyvips rejected~~ — **reversed on review**: the original rejection assumed triage
  was seconds of work; the wall-clock finding (every-frame 1080p decode is the
  pipeline's hottest CPU loop) changed the math. pyvips adopted (§3); grid compositing
  also benefits. Pillow remains the documented fallback.
- **t-SNE on 8 handcrafted features** (old pipeline): replaced by MDS on the SearchFaces
  similarity matrix — identity-meaningful layout (the old one clustered by image
  quality), zero extra API calls, no JS lib port.
- **Per-grid feedback chaining** (skip decisions fed grid-to-grid): the
  detection-as-skip-oracle insight is adopted, but per-grid feedback serializes the
  dominant Map stage into sequential rounds; probe-then-fill captures the same insight
  with one barrier.
- **Local ONNX face prefilter**: excluded by locked decision 2; a small per-asset saving
  at the cost of a model dependency.

## §12 Risks / accepted tradeoffs

- **Grid-cell bleed**: bbox center in wrong cell mis-assigns a face. Mitigated by 2 px
  separators + the >15 % overhang rejection (§6); gutters would shrink cells, deferred.
- **Sub-40 px faces**: undetectable until large enough (Rekognition floor; same as the
  old pipeline). Track start = first detectable frame.
- **Short appearances in hash-static scenes**: bounded at probe_stride seconds by
  construction; `probe_stride=0` escape hatch.
- **Patch-discount blind spot**: a face appearing INSIDE a chronically-noisy patch
  region is caught only by the probe cadence. Judged acceptable: faces inside ticker
  graphics are rarely editorial subjects; the ≤3-patch cap and global-hash backstop
  bound the exposure.
- **Sandbox collection leak**: residual window = EventBridge delivery failure only;
  self-healing on resubmit of the same workflow_id; sweeper deferred (§2b).
- **Re-run re-indexing**: mitigated by search-before-index; documented in the agent
  prompt.
- **Semaphore fairness**: direct-acquire vs drain race (bounded, accepted).
- **RecognizeCelebrities region availability**: `celebrity_recognition=false` escape
  hatch.
- **Long content**: no hard ceiling — checkpoint re-enter lets an iterator span multiple
  invocations; the practical bound is the SFN 25 K history-event ceiling (sizing keeps
  expected re-enters per state low; Distributed Map is the fix if it ever binds).
- **Semaphore-after-MP failure window**: a job that fails before acquire never holds a
  slot; a job aborted between acquire and release relies on EventBridge (seconds) or TTL
  (≤1 h). Accepted with the no-Catch decision.
- **Semaphore admission is size-blind**: five long jobs head-of-line-block short ones.
  Accepted for v1; the cheap fix if it bites is a reserved short-job slot keyed on
  duration_sec (known before acquire).
- **1080p frame extraction cost**: heavier transcode + S3 storage than 540p; frames are
  intermediates under the MP run prefix (lifecycle expiry deliberately skipped — §18).

## §13 Post-deployment knob moves (stack parameters → request level)

`MaxConcurrentExecutions` and `DetectMapConcurrency` were stack Parameters at initial
implementation (§7); moved to request-level `input_config` fields after real deployment
feedback — a per-account concurrency knob and a per-region TPS knob both belong to the
caller, not the deployer, since they can legitimately differ per job/region without a
redeploy.

- **`max_concurrent_executions`** (default 5): the stored `max_holders` attribute is
  updated via a raise-only conditional `UpdateItem` on every acquire/enqueue, separate
  from the increment transaction; the increment's condition compares against the live
  attribute. The account-wide cap only ever grows — a smaller request never lowers it
  (a single low-value caller must not throttle every other caller sharing the table).
- **`detect_map_concurrency`** (default 4): the three Map states use ASL
  `MaxConcurrencyPath` instead of a literal, making the low-TPS-region relief valve
  (§12) per-request instead of a redeploy.
- **`DefaultCollectionId`** moved from a Parameter to a template Mapping
  (`Collection.Defaults.Id` — note `Default` is a reserved second-level key inside
  CloudFormation Mappings and fails validation). No longer deploy-time overridable;
  per-job override remains via the `collection_id` gateway parameter.
- **IAM collection-action wildcard**: collection actions were scoped to
  `collection/${DefaultCollectionId}` + `collection/sandbox-*`; loosened to
  `Resource: "*"` since §19 lets callers create/delete their own named collections,
  making the narrower scope pointless. Blast radius: the Lambda role can touch any
  Rekognition collection in the account.

## §14 Counter TTL — self-heal against a permanently-lost stream decrement

§5 originally gave the counter NO ttl — only holder items had one. The reference
pipeline refreshes its single counter item's TTL on every acquire/release (safe there:
the counter's own mutation cadence is its only liveness signal). **A literal port is
unsafe here**: if refresh were tied only to `try_acquire`/`decrement_once`, a saturated
queue with 5 long-running holders and no new arrivals or completions for the full TTL
window (plausible — Rekognition spans can run past an hour on long assets, §12) would
see zero counter mutations even though every holder is alive. DynamoDB would delete the
counter mid-flight; the next `try_acquire` would see `attribute_not_exists(holders)` and
admit immediately — an over-admission bug worse than the gap being fixed.

**Adopted**: the counter carries `ttl` (`SEMAPHORE_COUNTER_TTL_SEC`, 3600 s), refreshed
in three places: `try_acquire`, `decrement_once`, and `refresh_holder` — the last only
AFTER confirming (ConditionExpression attribute_exists) that the caller's OWN holder
item still exists. That guard is critical: a zombie execution whose holder was already
TTL-reclaimed must not resurrect the counter for a slot it no longer owns. Net effect: a
permanently-lost stream decrement (event-source-mapping outage, exhausted retries) is
bounded — worst case, the counter and its ratcheted `max_holders` expire and reset once
the table goes genuinely idle for the TTL window, rather than drifting upward forever.
Accepted side effect: a fully idle table for > 3600 s resets `max_holders` too — fine,
since "idle for an hour" means no concurrent callers remain whose cap could be violated.
Regressions: `test_counter_ttl_refreshed_on_acquire_and_decrement`,
`test_refresh_holder_refreshes_counter_ttl_when_holder_alive`,
`test_refresh_holder_does_not_resurrect_counter_for_zombie`.

## §15 Celebrity naming: plurality vote over max-confidence (fairness audit)

A fairness audit on a real 2-hour rolling-news broadcast — seeded with a reference set
specifically to test recognition of a Black news anchor — verified detection, tracking,
and triage clean for her (an independent CompareFaces sweep against every output
person-cluster found exactly one 100%-similarity match: the #1 most-screen-time person
in the video), yet she was auto-indexed into the persistent collection under the wrong
name. Traced to `RecognizeCelebrities` itself: most of her looks came back AS a
celebrity — but a DIFFERENT wrong name almost every time (dozens of distinct wrong
names, all real public figures), at ~83 mean confidence; a light-skinned anchor in the
same broadcast landed her one correct name on nearly every look at ~99 mean.
`_cluster_celebrity` picked the single highest-confidence tag across a cluster's
tracks — with hundreds of independent noisy guesses, a lucky outlier clearing
`NAME_VOTE_INSTANT_SCORE` (95.0) is statistically near-certain, even when
"unrecognized" is the true plurality outcome. Re-auditing the run's other
high-vote-share matches confirmed the same failure mode across every non-white
demographic present — consistent with Buolamwini & Gebru ("Gender Shades"): commercial
face-analysis reference data skews toward lighter skin tones, degrading both confidence
and specificity for under-represented groups. A real, external bias — not one introduced
by this pipeline's own image processing, which is exactly what the vote structure has to
absorb.

**Fixed**: `_cluster_celebrity` tallies `name -> [confidences]` across the cluster and
returns the PLURALITY name's MEAN confidence. No change to `_vote_name` or its
thresholds — they were already correctly calibrated to reject ~83-mean noise; the bug
was purely in what confidence value reached them. Verified by replaying the real runs'
recorded recognize/search data: every visually-confirmed false positive dropped, every
high-confidence genuine match (the broadcast's own anchors, the politicians and sports
figures in its packages) unchanged. Regression: `TestClusterCelebrity`.

## §16 Two-tier cluster merge + relative quality ranking (audit continued)

Both §15 findings were cross-checked against every prior run (talk-show, episodic drama,
and feature-film assets) before implementing.

**The naming fix generalizes.** Replaying every run's recognize tags: the max-confidence
pick had named many clusters off single outlier samples with weak plurality support and
~80-85 mean confidence; genuine celebrities (100% vote share, 99+ mean) were untouched;
and the plurality vote corrected two episodic leads' clusters outright, where a lucky
outlier had overridden an overwhelming same-name majority.

**Cross-person contamination (distinct bug).** Union-find merged on ANY single
SearchFaces edge ≥ `SANDBOX_SEARCH_THRESHOLD` (75); merges are transitive and
irreversible. An unrelated man's track bridged into a large genuine cluster through a
chain of three 75-77 noise links stitched between genuine 97-100 links; the same defect
had shipped on episodic content, fusing two genuine actors' clusters (which is also what
fed the naming mislabels above). Edge-score stats across all replayed runs: genuine
same-footage edges overwhelmingly ≥95; every observed cross-person bridge in 75-85.
**Fix (two-tier corroborated merge)**: single edge unions only at ≥
`SANDBOX_MERGE_SIMILARITY` (90); weak edges [75, 90) union only with ≥
`SANDBOX_WEAK_CORROBORATION` (2) independent weak edges connecting the same two
components (`_merge_corroborated_weak_edges`, fixpoint-iterated; weak_edges persist in
the checkpoint). skip-if-grouped now applies only to strong merges — a weakly-linked
face still runs its own search so its edges can corroborate independently. Replaying all
runs' recorded edges through the production merge: zero contaminated components, zero
fragmented genuine identities (a parameter sweep showed strong=95 starts fragmenting
genuine clusters; 90/2 is the validated operating point). Trade-off accepted: more small
`Person_N` singletons instead of wrong merges — fragmentation is honest and recoverable
(sandbox re-merges on better evidence); contamination poisons montages, names, and the
persistent collection irreversibly.

**Absolute quality windows removed**: see the Gotchas entry — measured on the same
broadcast, `SHARPNESS_RANGE`/`BRIGHTNESS_RANGE`/`CENTROID_QUALITY_TIERS` were both
miscalibrated (rejecting most real faces) and skin-tone-correlated (Quality.Brightness
tracks skin reflectance under identical lighting). Replaced with relative ranking inside
the candidate pool; only extreme floors kept. `NOISE_LOW_BRIGHTNESS/SHARPNESS` in the
single-appearance noise filter are unchanged — that is an AND of two extreme floors for
single-frame noise, not a selection window.

## §17 Face image resolution, aspect ratio, and the rejected enhancement port

Three questions, all answered with measurements:

- **Stored derivatives → 3:4 portrait** (96×128 thumbnail; full image = whole-source
  downscale). The initial port stored squares — a silent deviation from the reference
  pipeline's 3:4. 3:4 wins: face boxes are portrait, square output pads background or
  cuts forehead/chin. Derivatives crop from the proxy frame — output size beyond the
  source crop's pixel budget is upscaled blur, so the fix for review-quality images on
  HD sources is the decision-#11 1920×1080 extraction target, not bigger output numbers.
- **Persistent IndexFaces image → native resolution** (`crop_face_native`, cap 1024,
  never upscale). The old fixed 180×240 resize discarded ~4× linear resolution on
  close-up faces before Rekognition saw them; the API takes 15 MB via S3Object, so
  shrinking bought nothing. Search grids keep 180×240 tiles (16 must fit one image;
  search only needs to find the match).
- **Pre-index sharpen/brighten enhancement → tested and REJECTED.** The reference
  formulas were ported and A/B-tested against real Rekognition (scratch collections,
  QualityFilter NONE to isolate the effect) at three difficulty tiers. Same-source and
  cross-source pairs: identical similarity. Stress tier (the pipeline's absolute worst
  faces, ~40 px / near the sharpness floor): enhancement REDUCED similarity in nearly
  every case and would have triggered on the large majority of real index picks —
  near-constant distortion for a negative return. The old intuition (help a weak
  embedder see edges) is inverted by modern embeddings: unsharp-mask halos read as false
  face detail. Keep crops faithful to the source; re-run the A/B if Rekognition's face
  model version changes materially.

## §18 Full-workflow code review (8-angle recall pass)

Eight parallel review angles (line-by-line, resume-invariants, data-contracts, reuse,
simplification, efficiency, altitude, conventions) over the whole repo; all
CRITICAL/MEDIUM findings fixed in place.

Correctness fixes:
1. *Reciprocal weak edges counted twice* — SearchFaces is symmetric, so one
   bidirectional noise match satisfied `SANDBOX_WEAK_CORROBORATION=2` alone, defeating
   the two-tier merge for singleton pairs. Fixed: dedup by unordered face pair.
   Regression: `test_reciprocal_weak_edge_counts_once`.
2. *drain_pending token discipline* — ANY SendTaskSuccess ClientError was treated as a
   dead token: an SFN throttle deleted a LIVE waiter's pending item; worse, a dead token
   after a REENTRANT acquire released the live holder of an already-admitted execution
   (over-admission). Fixed: `try_acquire` returns "acquired"/"reentrant"/""; only DEAD
   codes (TaskDoesNotExist/TaskTimedOut) consume the pending item; the slot is released
   only when freshly acquired; transient errors re-raise and keep the pending item.
   Regressions: `TestDrainPendingTokenHandling`.
3. *Stream handler skipped the drain when its decrement deduped* — a retried stream
   batch after a mid-drain crash never drained again. Fixed: drain unconditionally (a
   no-op drain is one cheap Query).
4. *IndexFaces/register_face non-atomic* — a crash between the two left an orphaned
   collection face AND re-indexed a duplicate on re-entry. Fixed twice over: checkpoint
   immediately after any cluster that indexed, and self-repair on resume (a search match
   with no DDB record whose ExternalImageId embeds THIS run's uuid is our own orphan;
   register it with the current vote instead of indexing a duplicate).
5. *No periodic checkpoints in recognize_celebrities / cluster search loop* — a
   non-timeout crash re-billed every call since the last write and lost usage counts.
   Fixed: checkpoint every `CHECKPOINT_EVERY_N_GRIDS` / `CHECKPOINT_EVERY_N_SEARCHES`.
6. *finalize_outputs defaulted duration_sec to 0* — the same forbidden shape as the fps
   bug (probe-derived field silently defaulted). Now raises.

Altitude/efficiency fixes:
7. *Execution-ARN reconstruction deleted* — all Rekognition-span states receive
   `$$.Execution.Id` via the ASL Payload; the reconstruction produced a wrong holder key
   whenever uuid != execution name (console re-drives, direct StartExecution).
8. *Single-face native-resolution search query* — SearchFacesByImage only searches the
   LARGEST face in the query image; the old 16-tile grid paid 16 serial
   GETs+crops+compose per cluster for an API that used one arbitrary tile. Now: one best
   face, native resolution (better match quality too).
9. *store_debug_images*: person-level parallelism + ONE prefix listing replaces N
   head_object probes.
10. *History walks*: `includeExecutionData=False` (payloads are tens of MB on long
    runs; only names/types/timestamps are read).
11. *Cluster expansion de-quadraticized* — rep_tid computed once per group; expansion is
    a dict lookup.

Reuse/dead-code: shared `grid_image.match_tile`, shared `rekognition_utils.add_usage`
(was 4 hand-rolled copies — one shipped bug came from exactly this drift), named
margin constants, removed dead constants and an unused thread pool. `face_store`
docstring corrected: this ExternalImageId format is the same SHAPE as the reference
pipeline's but NOT compatible — standalone by design.

Follow-ups resolved: O(n²) checkpoint rewrites → one side object per completed grid,
consolidated with a single write + batch delete (O(n) total bytes; regression:
`TestDetectSideObjectCheckpoints`); Lambda variant right-sizing applied for
`build_face_tracks` and `search_index_faces` (measured: pure-CPU and API-wait states
don't need 3 cores); S3 lifecycle expiry on frame prefixes deliberately skipped.
Deferred: gateway parameter bounds re-typed as literals (input_schema is the truth);
dispatcher structural sniffing for stream/EventBridge events.

## §19 Phase 2 — face index management (locked spec)

Second state machine `${ResourcePrefix}-face-index`
([definitions_index.json](definitions_index.json), injected via
`%%INDEX_STATE_MACHINE_DEFINITION%%`), same container/roles/semaphore as recognition.
Routes on `$.action`.

Tool placement — gateway for sync reads/renames/create, state machine for anything
resumable or Rekognition-heavy:

| Tool | Placement | Notes |
|---|---|---|
| `create_collection` | gateway | existence rule below |
| `delete_collection(collection_id, force?)` | state machine | resumable sweep; order: DDB → S3 derivatives → Rekognition collection LAST (a dangling-empty collection is self-describing; the reverse poisons readers); `force` required for the stack default |
| `list_collections()` | gateway | per collection: Rekognition FaceCount + DDB record count — drift is always visible (doubles as reconciliation probe); hides `sandbox-*` |
| `list_faces(collection_id, …)` | gateway | `collection-index` GSI; filters origin/named/workflow_id/celebrity_name |
| `get_face(face_id)` | gateway | late-binding resolver: past facemetadata.json readers query the LIVE name by face_id; answers from the tombstone for deleted faces (readers learn WHY an id is gone, not a bare 404) |
| `batch_update_faces` | gateway (≤500/call) | rename = DDB update only, GSI re-keys itself; NO historical-output rewrite (late binding is the design) |
| `batch_delete_faces(face_ids \| celebrity_name)` | state machine | one source of truth, no gateway/SM split by size; hard delete + tombstone (`deleted#{face_id}`, own TTL) — NO soft-delete flag on live records (right-to-erasure doesn't tolerate lag) |
| `batch_index_faces(s3_prefix \| manifest_s3_uri, collection_id)` | state machine | must acquire THE EXISTING semaphore (IndexFaces shares the Rekognition TPS pool with recognize_faces runs) |

Key decisions:
- **Schema**: GSI `collection-index` (pk `collection_id`, sk `celeb_created` =
  `{celeb}#{epoch_ms}`) — ONE canonical sort order; more GSIs only when a real consumer
  demands them. Record attributes `origin`, `source_key`, `source_etag` — provenance
  lives in DDB, NOT in ExternalImageId.
- **ExternalImageId is per-CALL by API constraint** (every face in a tiled IndexFaces
  grid shares it), so it is a coarse batch-level tag (`user:{batch_id}:{epoch}`); DDB is
  the source of truth for per-face provenance.
- **Idempotency keyed on content, not name**: (collection_id, source ETag). Same ETag →
  `skipped/already_indexed` (zero API calls). Same ETag + different manifest name →
  rename via DDB (`skipped/renamed`). New ETag under an existing name → index (multiple
  reference faces per celebrity is by design). Crashed batches resume by re-running.
- **Tiling**: fit-inside tiles, one IndexFaces per grid (orders of magnitude cheaper
  than solo calls); phantoms (background faces in a tile): ONLY the largest mapped face
  per tile wins, the rest are deleted from the collection; tile→source coordinate
  recalibration before storing. Tiny-face fallback: a line whose tile yields no mappable
  face gets ONE solo full-resolution IndexFaces before being reported failed. Geometry:
  §20 addendum.
- **QualityFilter AUTO** for curated user images (not the runtime auto-index path's
  HIGH).
- **Image formats**, gated at manifest validation by magic bytes, never extension:
  JPEG/PNG/WebP/TIFF/GIF accepted; HEIC/HEIF/AVIF accepted with per-file
  fallback-to-fail (wheel builds can lack the HEVC decoder even when the loader
  registers — decode errors surface as `failed/decode_error`, never a crash); BMP /
  JPEG 2000 rejected (loaders not built in); SVG / PDF rejected by policy.
- **Derivatives** (semantic change adopted by the auto-index path too, for uniform
  records): `thumbnail` = 96×128 3:4 face crop; `full_image` = fit-inside downscale of
  the WHOLE source; `coord` = normalized `[x,y,w,h]`, resolution-agnostic, matching the
  Rekognition BoundingBox convention — the source of truth locating the face.
- **Per-line report, fail-hard**: every manifest line ends in EXACTLY one entry —
  `indexed | skipped | failed` + reason; the finalize state FAILS if entry count ≠
  manifest count (no silent drops). Validation also fails hard when EVERY line
  head-fails — a foreign/inaccessible bucket must not "succeed with 0 indexed". Source
  images must live under the proxy bucket (IAM boundary — the gateway rejects foreign
  buckets with FOREIGN_BUCKET).
- **create_collection existence rule**: Rekognition collection exists with FaceCount 0 →
  adopt; has faces with zero DDB overlap → **fail** (foreign/unmanaged collection); all
  in DDB → succeed (idempotent, it's ours); partial overlap → succeed + drift warning
  with both counts (a crashed delete leaves asymmetry; failing would lock the user out
  of their own collection).
- `delete_collection` in-flight risk (a running recognize_faces using that collection
  mid-search): DOCUMENTED, not detected — collection_id is per-request, so in-flight
  users are unknowable.
- NOT ported from the reference face-indexer: the update-propagation state machine
  (rewrites historical analysis outputs — wrong for the immutable-outputs +
  late-binding model) and bulk import-from-collection.

## §20 Phase 2 implementation deltas

Every §19 decision held. Deltas:

- `collection-index` added as an online GSI (CFN update + a one-off backfill stamping
  `celeb_created` + `origin` on pre-existing records) — no stack deletion.
- Second state machine instead of new states in the recognition ASL — separate concerns,
  same container image, same roles, same semaphore, one shared terminal EventBridge
  rule (the sandbox delete no-ops for index executions).
- `delete_collection` and `batch_delete` share one implementation
  (`states/batch_delete_faces`) — validate resolves the work list either way; only the
  "delete the Rekognition collection last" tail differs.
- Rename-on-reindex happens in validate (DDB-only, pre-semaphore), so a manifest that
  only renames existing content never holds a Rekognition slot.
- Gateway zip stays flat: `index_tools.py` re-declares five constants from
  `lib/input_schema`; `TestGatewayIndexToolConstants` guards the duplication.
- `list_collections`' reconciliation probe proved itself immediately: legacy collections
  in the dev account showed face_count>0 / record_count=0 — exactly the "foreign
  collection" case `create_collection` refuses to adopt.
- Live-verified end-to-end: create (new/adopt-empty/adopt-ours), prefix batch → ONE
  IndexFaces call with tile→source coords + derivatives, idempotent rerun (all skipped,
  zero Rekognition calls), paginated celeb-sorted listing, rename re-keying both GSIs,
  tombstone read-through, default-collection force guard, full collection sweep.

### §20 addendum: batch tile/grid sizing — measured, bigger-grid variant reverted

The initial 480×640 tile was a derived guess; a 1080×1080/9×9 variant (maximize the
pixel budget) was deployed and A/B-measured on the same real staged headshots: the small
grid indexed everything in ONE IndexFaces call; the large grid needed several solo-retry
rescues. Root cause (the wrong assumption in "pixels are free"): Rekognition's ~40 px
minimum face is RELATIVE — it scales with the image's long edge (floor ≈ 40 ×
long_edge / 1920). Consequences:
- For sources DOWNSCALED into tiles, tile size cancels out of the face-to-floor ratio
  entirely — bigger tiles buy nothing; giant group shots are the solo-retry fallback's
  job at ANY tile size.
- For small NATIVE-size sources (tiny avatars, low-res web stills), the face stays at
  native pixels while a bigger grid RAISES the absolute floor: they pass in small grids
  and miss in large ones. Empirically confirmed — exactly the smallest native images
  were the solo-retries at 1080.

Final geometry: **480×640 tiles, 8×6 grid (≈3850 px, square-ish)** — minimizes the grid
LONG edge (the floor driver) instead of maximizing pixel usage. Rows capped at 6: a tall
8×10 grid's longer edge would re-break the small avatars the 8×6 grid passes. The
compose over-bound fallback (split batch in half, re-grid, bottoms out at solo) is
retained — degrade toward MORE calls, never toward failed lines.

## §21 Shot-segmentation supplement

Accept an OPTIONAL `shots.json` produced by the shot-segmentation sibling and use its
DINOv2 shot boundaries as superior cut evidence. A QUALITY feature, not a cost one:
shot-seg's archived frames are low-res proxies (useless against Rekognition's face
floor), so the media-processing child, probe-then-fill, and every Rekognition call run
unchanged. The only thing shots touch is cut evidence.

**Interface**: `input_config.shot_segmentation_s3uri` points at the shots.json FILE, not
the job prefix — face-recognition depends on the documented shot schema
(`start_frame`/`end_frame`/`start_time`/`end_time`), never on shot-seg's internal path
layout; any producer with a conforming file works (hand-corrected lists included).
Chains naturally: shot-seg's get_job_status advertises the file as
outputs[label="shot-list"]. `asset_s3_uri` stays required (we always need the media for
1080p frames) — no overloading, no cross-workflow indirection. Absent → dHash cut
detector. The field folds into the workflow_id hash like every other input_config field.

**Timeline alignment** (verified against both codebases, incl. mark_in_out): both
pipelines speak SOURCE-ABSOLUTE frame numbers — MP's clipped output does not restart at
0, and shot-seg uses exact integer PTS arithmetic. No offset math anywhere; a shots.json
spanning the full asset supplements a clipped recognition run for free.

**Plug-in points**:
1. `build_face_tracks`: shot boundaries join the cut veto. (Originally REPLACE —
   superseded by §23's UNION after an incident; boundaries add slow dissolves and heal
   flash-frame false breaks, dHash catches boundaries the upstream list is missing.)
   Explicitly NOT doing per-shot face grouping (the old pipeline's approach): shots
   cannot separate two people alternating within one shot; gap-dilated IoU + gender +
   boundary containment can. Shots are a containment constraint on tracking, not a
   replacement for it. dHash is NOT removed from the pipeline — triage, noise-patch
   classification, and dedup all still use it.
2. `plan_dense_detection`: a skippable gap additionally requires NO shot boundary inside
   it — an AND (more detection, never less); hash-static tests stay because they detect
   within-shot motion and entrances, which boundaries cannot.

**Validation — fail-hard at `prepare_probe_iterators`** (before any Rekognition spend);
the normalized boundary list persists to `ephemeral/shot_boundaries.json`; downstream
states consume the validated list, never the raw file. Checks (a wrong shots file
silently fragmenting or gluing tracks is worse than a crash): schema (non-empty `shots`,
integer frames, strictly-increasing starts); same-asset fps cross-check (per-shot
implied fps vs the probe's exact fps within 1%); same-asset duration cross-check (last
`end_time` vs MP's full-source duration within max(2 s, 2%)). Bucket boundary: the
gateway rejects non-proxy-bucket uris with FOREIGN_BUCKET.

**Resolved questions**: every shot start is a boundary regardless of `shot_type`
(`content`|`black`|`static`) — a black sliver between scenes is two boundaries, correct
containment. Skipping DetectFaces on `black` shots was considered and rejected: per-frame
triage already drops black/mono frames at full resolution before probe selection (zero
API cost already), is more granular (individual black frames inside content shots), and
the proxy-resolution "black" verdict isn't ours to trust — a near-black frame with a dim
real face is this pipeline's to judge at 1080p. `static` shots are never skipped:
freeze-frame ≠ faceless. Validation failure FAILS the run — a caller who supplied shots
wants shots; warn-and-fall-back would silently hide a wrong-asset bug.

Zero ASL changes; zero new states (precedent: the recognition-policy toggles — behavior
differences live in state handlers as data, not workflow topology). No IAM changes.

## §22 Representative face crop per person (`rep_face_key`)

Driver: the shot-metadata-enrichment sibling binds face NAMES to bodies in multi-person
shots via a per-window reference gallery, and its probes found a name list alone binds
by LIST ORDER (confidently wrong half the time) while one labeled reference crop per
name fixes the binding. Measured constraints from the same probes: gallery cells render
at 192×256 (3:4); the 96×128 thumbnail scale FAILED a re-identification probe (the model
wavered between two faces and picked wrong), so the reference must carry more resolution
than `FACE_THUMBNAIL_DIM`.

The crop lives HERE (frames in hand at emit time; every future consumer gets it; the
artifact survives frame-prefix expiry) rather than each consumer re-walking
facegroup.json → tracks → centroid face → frame fetch → crop. What ships:

- `debug/people/person_{NNNN}_face.jpg` — written by `store_debug_images` next to the
  montage: the person's representative face as a single 3:4 crop at NATIVE resolution,
  downscale-only cap 512 px long edge (`REP_FACE_RATIO`/`REP_FACE_MAX_EDGE_PX`). New
  `grid_image.crop_headshot_native` = `crop_headshot`'s aspect-matched rectangle (incl.
  edge letterboxing) + `crop_face_native`'s never-upscale policy — the existing helpers
  each had exactly half the needed contract.
- `rep_face_key` in facemetadata people[] (and facegroup's repeated person fields),
  forward-pointer pattern like `thumbnail_key`. The rep pick (max `appearance_count`
  track → its centroid face) tie-breaks deterministically on earliest `first_ts` in
  BOTH `finalize_outputs` and `store_debug_images`, so `rep_face_key` provably points at
  the face `gender`/`age_range` were read from.
- `output_contract.json` updated: thumbnail_key is the debug MONTAGE, rep_face_key is
  the face-matching reference — consumers must never feed the montage to a matcher.
- Re-enter safety: montage and face crop are checkpointed independently against the
  prefix listing (a run that wrote montages before this change fills in only the missing
  `_face.jpg`s on retry).

Tests: `test_imaging.py::test_headshot_native_aspect_and_no_upscale` /
`test_headshot_native_caps_long_edge`.

## §23 The cross-cut mislabel — veto union + witness corroboration (incident RCA)

Found on an E2E run over a feature-film test asset with a two-person reference
collection: facemetadata named a 46-track person — visually actor_M1 in every montage
tile — **as actor_M2, name_source=collection at 99.95**, and actor_M1 appeared NOWHERE
in the output despite a 41-track celebrity plurality of "actor_M1" at ~99 mean recorded
in the ephemeral checkpoint. The collection itself was verified correct. Every model
call returned a locally-correct answer; **the failure is compositional**. The chain,
each link proven from the run's own artifacts:

1. **The supplied shot list missed real cuts** inside one shot spanning a cross-cut
   close-up conversation (actor_M1 → actor_M2 → actor_M1). (Postscript: the upstream
   embedding did NOT miss them — the boundaries were discarded by a shot-segmentation
   bookkeeping rule, fixed in that repo. For THIS repo nothing changes: an upstream
   boundary list can be incomplete for any reason, which is exactly why fix A refuses to
   depend on it alone.)
2. **Shots-mode had REPLACED the dHash cut veto**, so the supplied list was the
   tracker's ONLY cut evidence; the missed cut removed the veto entirely.
3. **The tracker bridged the cut by a hair**: gap-dilated IoU across the box jump
   cleared `MIN_TRACK_IOU` (0.10) by 0.018, same gender → one track with 3 actor_M2
   faces + 7 actor_M1 faces. NOTE: the contaminated track carries the MAJORITY's tag, so
   track-level agreement filters cannot exclude it.
4. **`_search_persistent` picked ONE query face per cluster by photo quality alone**
   (area × frontality, then sharpness). actor_M1 spends the film hatted/backlit/low-key;
   actor_M2's 3 contaminated frames are bright frontal close-ups — the foreign face is
   the best photo in the cluster BY CONSTRUCTION. The 99.95 was Rekognition truthfully
   matching actor_M2 to actor_M2: right answer, wrong question.
5. **`_resolve_name` preferred face_match unconditionally and published no trace of the
   losing candidate** — the correct identification existed only in the ephemeral
   checkpoint, which the output contract marks internal. Blast radius = the whole person
   (naming is per-cluster).

**Fix A — veto union in shots mode** (`build_face_tracks`): `cut_break_between` =
`shot_boundary OR local_dhash_break` instead of replace. The patch dHashes are computed
for every frame anyway (they drive probe-then-fill), so this is one OR, not new compute.
Cost asymmetry decides the false-break tradeoff: a false break splits a track and
sandbox clustering re-merges the pieces; a false continuation contaminates an identity
irreversibly. Protects every policy configuration, including celebrity-off runs where
fixes B/C have no conflict signal.

**Fix B — conditional two-witness corroboration** (`_search_persistent`): witness 1 =
today's best face (unchanged, zero cost change). ONLY when its match name disagrees with
a validated celebrity plurality: witness 2 = best face from a DIFFERENT track (the
different-track constraint is what defeats this incident — all contaminated faces live
in one track, so witness 2 is guaranteed genuine; a naive top-2 would have picked the
foreign face twice). W2 agrees with W1 → accept the collection name (the §15 scenario:
celebrity confidently wrong, collection right). W2 disagrees → prefer the witness
consistent with the plurality; name_source stays "collection". Cost: +1
SearchFacesByImage per CONFLICTED cluster only; the no-conflict majority spends nothing.

**Fix C — publish the conflict** (`finalize_outputs`): precedence is UNCHANGED —
collection-first is right, per §15's finding that RecognizeCelebrities misidentifies at
>99 confidence. When the resolved name disagreed with a validated celebrity plurality
(or witnesses disagreed), the person carries `name_conflict: true` plus the losing
candidate and its confidence. A future novel contamination path becomes self-announcing
instead of something a human catches by recognizing an actor.

**Decision — face-rec keeps its LOCAL dHash veto** and will NOT consume
shot-segmentation's own planned per-boundary dHash evidence. Three reasons: (1) the
local hashes are computed regardless — load-bearing for probe-then-fill, nothing saved;
(2) different frame streams — shot-seg hashes its own coarse transcode, the tracker asks
about ITS OWN frames at frame_interval_sec, and local hashes answer that exact question;
(3) it would re-create the single-point-of-failure this incident is about, plus a
version coupling (old shots.json files carry no flag, so the fallback IS the local
veto). The correct relationship is defense in depth: shot-seg's second opinion reduces
missed boundaries at the source; the local union catches the residue.

**REJECTED: "higher similarity wins" on a name conflict.** The counter-example is this
incident's own numbers: the contaminated witness matched at 99.95, the genuine one at
99.4 — higher-similarity picks the wrong name, BECAUSE the foreign face is the best
photo in the cluster by construction. Both matches are TRUE (each face genuinely matches
its own reference); similarity measures photo-to-reference quality and cannot arbitrate
which witness belongs to the cluster. Weight of evidence (plurality/corroboration) stays
the arbiter; similarity is reported, never decisive.

**Speced follow-up (not yet implemented)**: per-face celebrity vote retention (zero new
API cost — the recognize state already receives a result PER FACE and discards the
detail on aggregation). Two consumers: a mixed-name track is a CONTAMINATION TRIPWIRE
(flag it, exclude its minority-name faces from witness duty); cluster-level mixed votes
across tracks reveal multi-track contamination that two witnesses cannot (both witnesses
foreign → wrong 2-1 majority — the known residual limit of best-2-of-3).
Identity-consistent witness selection — excluding faces whose OWN per-face tag disagrees
with the validated cluster plurality — would have prevented this incident's conflict
entirely with zero extra calls.

**Regressions**: all three fixes replay the incident run's recorded artifacts as
fixtures — the contaminated track must SPLIT at the dHash break (A); the query selector
must not choose a face from a track whose faces straddle a cut, witness-2 must come from
a different track, and the corroboration matrix must resolve to the genuine name (B);
the conflicted person must carry both candidates (C). See
`test/test_units.py` (shots-mode cut veto UNION + search-stage replay tests).
