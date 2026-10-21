# shot-segmentation-on-aws

Serverless video shot segmentation and recurrence evidence. A seven-state AWS
Step Functions pipeline transcodes a source video (any format ffmpeg reads —
MXF, MP4, MOV, …) to a native-aspect-ratio all-intra proxy, extracts DINOv2
frame embeddings, detects shot boundaries, identifies recurring shot groups
(act-break cards, idents), and publishes frame-exact shot metadata plus
structural evidence (programme tail, burst runs) for agent consumption.

The workflow's posture is **evidence, not verdicts**: it emits shots and
recurrence facts with the measurements that justify them; scene-boundary and
editorial decisions belong to its consumers.

## Model

**DINOv2 ViT-S/14-reg** (ONNX FP32, opset 17, ORT CPU-optimised)

- Input: `[B, 3, 224, 224]` float32, ImageNet-normalised
- Output: `[B, 384]` float32 CLS token (L2-normalised downstream)
- Baked into the Docker image at `/opt/models/dinov2_vits14_reg_cls.onnx` — no
  cold-start S3 download. `_get_session()` in `run_dino_v2_model` loads it once
  and caches the ORT session in a module-level global.

## Architecture

One Step Functions state machine (ASL inline in [cfnstack.yaml](cfnstack.yaml),
readability copy in [definitions.json](definitions.json) — builds fail on
drift). Entry point [lambda_function.py](lambda_function.py) dispatches on
`event["operation"]`.

```
StateProbeVideoForChunkTranscode
  └─ Map: StateRunFFmpegTranscode          (MaxConcurrency=40, 20-min segments)
       └─ StateDynamicChunkingForFrameEmbedding
            └─ Map: StateExtractFrameFeatures    (MaxConcurrency=40, 6-min windows)
                 └─ StateMergeAndReconcileShots
                      └─ StateStoreDebugShotImages
                           └─ StateUsageTracking
```

**State responsibilities:**

1. **StateProbeVideoForChunkTranscode** — `ffprobe` for
   codec/framerate/timecode/interlace; seek-capability test via the exact
   ffmpeg command; `is_all_intra` detection; builds 20-min segment iterator.
2. **StateRunFFmpegTranscode** — transcodes one segment to an all-intra
   OpenH264 MP4 fitting within 512×512 at the source aspect ratio (no
   letterbox padding — e.g. 16:9 → 512×288); captures exact source frame
   numbers from showinfo PTS via integer fraction arithmetic; writes
   `_pts.json`.
3. **StateDynamicChunkingForFrameEmbedding** — filters `EMPTY` segments;
   verifies contiguity; builds 6-min windows with 30 s overlap.
4. **StateExtractFrameFeatures** — runs DINOv2 on each frame in the window
   (4 fps coarse + zoom-in at boundaries); detects black frames; runs shot
   boundary detection; archives all model-input frames at native proxy
   resolution as `_frames.tar.gz`. Checkpoints and resumes across the Lambda
   900 s ceiling (§9.13).
5. **StateMergeAndReconcileShots** — merges shots across window boundaries via
   coverage-tracking dedup; noise-merges short shots between similar
   neighbours; runs recurrence detection (§9.1–§9.6); selects up to 16
   representative frames per shot; repackages per-shot frame archives;
   hash-confirms surviving recurrence groups against those repacked frames;
   writes final `embeddings.npz`, `shots.json`, `recurrence.json`.
6. **StateStoreDebugShotImages** — reads per-shot frame archives from
   `shot_frames/`; downscales to 112×112 and composes per-shot JPEG grids
   (no ffmpeg re-seeking — grids show the exact pixels DINOv2 processed).
7. **StateUsageTracking** — walks the Step Functions execution history;
   estimates per-state Lambda cost.

The state machine fronts six Lambda size variants of the same container image
(896 MB half-core up to 10240 MB six-core); CodeBuild (ARM_CONTAINER) bakes the
model at build time. The gateway dispatcher is a separate flat-zip Lambda (no
container).

## Input Schema

Defined and validated by [lib/input_schema.py](lib/input_schema.py). All
`input_config` fields are optional — defaults are applied by
`validate_and_normalize()`.

```json
{
  "uuid": "<UUID>",
  "input": {
    "s3uri":  "s3://ingest-bucket/path/to/video.mxf",
    "bucket": "ingest-bucket",
    "key":    "path/to/video.mxf",
    "input_config": {
      "segment_duration_secs":       1200,
      "coarse_fps":                  4,
      "cosdist_threshold":           0.10,
      "shot_boundary_threshold":     0.20,
      "black_pixel_threshold":       0.04,
      "black_coverage_pct":          99.5,
      "black_entropy_max":           0.20,
      "max_recurrence_threshold":    0.07,
      "min_recurrence_count":        3,
      "min_occurrence_gap_sec":      60.0,
      "max_recurrence_count":        8,
      "adjacent_centroid_threshold": 0.20,
      "max_adjacent_merge_duration": 30.0,
      "near_black_entropy_max":      1.5,
      "min_shot_duration":           0.25,
      "noise_distance_threshold":    0.18,
      "fade_split_threshold":        0.45,
      "fade_split_min_duration_sec": 8.0,
      "drift_split_threshold":       0.40,
      "drift_split_min_duration_sec": 60.0,
      "embed_geometry":              "letterbox"
    }
  },
  "output": {
    "s3uri": "s3://proxy-bucket/output/prefix"
  },
  "data": {}
}
```

`bucket` and `key` are derived from `s3uri` by `validate_and_normalize()` in
the probe state — the gateway tool sends only `s3uri`. (They exist as separate
fields because the ASL's SDK HeadObject state reads `$.input.bucket` /
`$.input.key` before any Lambda runs.)

## Output Files (S3)

```
{prefix}/
├── probe.json                        probe state output (codec, timecode, seek capability)
├── mxf_info.json                     MXF only — bmx metadata
├── proxy/
│   ├── segment_0000.mp4              all-intra OpenH264 proxy, native AR within 512×512
│   ├── segment_0000_pts.json         frame index: source frame numbers ↔ PTS times
│   └── segment_0000.json             transcode metrics
├── embeddings/
│   ├── seg0000_win0000.npy           [N, 384] float32 per-window embeddings
│   ├── seg0000_win0000.json          local_shots, diagnostics
│   ├── seg0000_win0000_frames.tar.gz model-input JPEGs at proxy native resolution
│   ├── embeddings.npz                final: embeddings (float16 — cast up before matmuls) + frame_number + source_time + entropy
│   ├── shot_centroids.npy            [n_shots, 384] float16 per-shot centroids; row index == shot_id
│   ├── shots.json                    global shot list + frames_key (no embeddings — read shot_centroids.npy)
│   └── recurrence.json               recurring shot groups
├── shot_frames/
│   └── shot_0000_frames.tar.gz       selected frames (≤16) per shot, for debug grids
├── shot_images/
│   └── shot_000000.jpg               per-shot debug grid (112×112 px cells)
└── usage_tracking.json               per-state cost estimates
```

`scenes.json` does not exist: per-window scene clustering was removed from the
pipeline. Consumers derive scenes from `shots.json`.

### Shot schema (`shots.json` entry)

```json
{
  "shot_id": 0,
  "shot_type": "content",
  "start_frame": 0,
  "end_frame": 299,
  "start_time": 0.0,
  "end_time": 11.96,
  "start_timecode": "09:59:43:14",
  "end_timecode": "10:00:00:05",
  "frame_count": 50,
  "sync_pop": null,
  "flash_frame": null,
  "entropy_mean": 4.21,
  "entropy_min": 3.10,
  "entropy_max": 5.87,
  "frame_coherence": 0.9412,
  "frame_dispersion_max": 0.1837,
  "frame_motion_mean": 0.0264,
  "frame_motion_max": 0.1102,
  "selected_frames": [0, 45, 120, 299],
  "frames_key": "{prefix}/shot_frames/shot_0000_frames.tar.gz"
}
```

The four `frame_*` fields are **intra-shot dispersion**: how much the shot's
own frames vary, which `centroid_embedding` discards by being a mean. They are
computed in the same final pass as the entropy aggregation, from frame
embeddings already in RAM.

| field | meaning |
|---|---|
| `frame_coherence` | ‖mean of the unit frame embeddings‖, in [0, 1]. 1.0 = every frame identical. The mean frame-to-centroid cosine distance is **exactly** `1 - frame_coherence`, which is why that field is not published — it would be the same number twice. |
| `frame_dispersion_max` | furthest single frame from the centroid — the tail the mean hides (one dissolve or wipe frame on an otherwise static shot) |
| `frame_motion_mean` / `frame_motion_max` | cosine distance between **adjacent** frames. An embedding-space *proxy* for motion, not optical flow. High coherence with high adjacent motion = a shot that keeps returning to the same look (flashing graphic); low coherence with low adjacent motion = a slow drift. `frame_motion_max` flags a within-shot boundary the detector did not take. |

All four are `null` where undefined — all of them for a shot with no embedded
frames, the two motion fields for a single-frame shot. Deliberate: `0.0` would
read as "measured, and perfectly still". Pinned by
[test/shot_dispersion_test.py](test/shot_dispersion_test.py) (the
`1 - coherence` identity, coherence/motion independence, null-not-zero).

### Programme-tail evidence (`shots.json` envelope)

The envelope carries `programme_tail` — evidence about material after the last
long black (`null` when the programme runs to the end of the file):

```json
"programme_tail": {
  "separator_shot_id": 525, "separator_start_time": 1803.0,
  "separator_end_time": 1866.2, "separator_black_sec": 63.2,
  "tail_content_sec": 138.2, "match_fraction": 0.96,
  "match_duration_fraction": 0.96, "match_monotone": 0.98,
  "match_fraction_duration_corroborated": 0.96
}
```

The separator is a maximal RUN of consecutive black shots spanning ≥30 s,
starting in the second half, followed by any content at all (among candidates,
the one with the *most* content after — a multi-part master has several
inter-part blacks). A run, not a single shot: segmentation legitimately splits
one physical black into several black shots, and a single long black is a run
of one. There is no minimum-content floor on the tail — a floor was measured
to suppress a real ~56 s textless block, and an end-board tail reads honestly
in the numbers instead (seconds-long `tail_content_sec`, match ~0).

- `match_fraction` — share of tail content shots whose nearest pre-separator
  shot sits within cosine 0.05 (recurrence's own near-identical-render
  distance).
- `match_duration_fraction` — share of those matches whose durations agree
  within 1 s. A textless element is the SAME CUT as its source, so matched
  textless pairs measure frame-exact, while a same-set look-alike is a
  different cut and fails the agreement.
- `match_monotone` — share of matched sources advancing in order.
- `match_fraction_duration_corroborated` — widens the cosine band to 0.10 but
  demands the duration agreement; heavy burned graphics (a credit roll) push a
  clean copy just past 0.05 where burned subtitles stay inside it. Measured
  across the validation corpus it reads 0.0 on every non-textless tail and
  well above it on every real textless block.

Reading, measured on the validation corpus: **matched (or corroborated) AND
duration-agreeing AND ordered ≈ textless block** (clean copies appended for
localisation — the programme effectively ends at the separator, and
`actual_duration_sec` overstates the runtime by the tail); **low
`match_fraction` with duration-agreement ~1.0 ≈ a partial textless block**
(sourced only partly from this episode); **unmatched with
duration-disagreement ≈ a promo/recap or a further programme part**.
Deliberately no verdict field — the numbers are the contract
([lib/tail_evidence.py](lib/tail_evidence.py)). Why not audio: a textless
block carries full M&E and is audio-indistinguishable from programme; the moat
is already a published black run (§9.7).

### Burst-run evidence (`shots.json` envelope)

`burst_runs` — contiguous intervals of very short shots (each < 1 s, runs of
≥ 6), `[]` when none:

```json
"burst_runs": [{"start_shot_id": 3, "end_shot_id": 37}]
```

An interval and nothing else, on the recurrence trim discipline: a burst run
is contiguous by definition, so member count, duration, and character all join
from the shot list — the field's whole value is the *shared definition*. Shot
**type is ignored** on purpose: jingle reels separate their cards with black
micro-shots, and a content-only rule breaks every run at each separator. The
boundaries inside a run are real cuts, correctly detected; the field exists so
consumers (shot-metadata-enrichment above all) can **fold each run into one
inference** instead of paying per member — measured, an asset that *is* a
jingle reel has an order of magnitude more of its shots in bursts than
ordinary content with montage moments. Thresholds live in
[lib/burst_evidence.py](lib/burst_evidence.py) / `input_schema.py` and are
fitted on a small sample (§9.14 is the plan to grow the n); deliberately no
genre verdict in the payload (§9.15).

### Recurrence schema (`recurrence.json` entry)

```json
{
  "group_id": 0,
  "recurrence_count": 4,
  "occurrence_count": 4,
  "max_distance": 0.032,
  "hash_confirmed": true,
  "shots": [
    {"shot_id": 4, "position": "act_boundary"}
  ]
}
```

Member entries are `{shot_id, position}` **only** — times, durations,
timecodes, entropies and the `frame_*` dispersion fields come from joining
`shots.json` by `shot_id`. The two files are a pair, and every copied field
was a chance for them to disagree. Group-level numbers are kept only where the
join can't produce them: `occurrence_count` (needs the §9.1 gap rule),
`max_distance` (needs the centroids), and `recurrence_count` as the single
convenience exception (`== len(shots)`). Two consumer-side verdicts replace
old hard drops: **exceeds the set ceiling** =
`recurrence_count > max_recurrence_count` (both published — deliberately no
flag), and **`hash_confirmed`** = `true` (the same rendered image genuinely
recurs) / `false` (checked: same set, not same render) / `null` (unverifiable
— nothing hashable, which must not read as checked-and-failed).

The envelope carries a `gates` object attributing the only drops that still
exist:

```json
"gates": {
  "content_shots": 812, "excluded_non_content": 43, "components": 37,
  "min_members_dropped": 24, "occurrence_gate_dropped": 4, "groups": 4
}
```

Counts are GROUPS except `content_shots` / `excluded_non_content`, which are
shots; `components` and `min_members_dropped` exclude singletons (most shots
on a 1000-shot asset are unique, so counting them would bury the numbers that
matter). Since §9.3 annotates and §9.6 flags, nothing downstream of these
counters deletes: **`gates.groups == total_groups`** is the invariant.
`detect_recurrence()` fills these into an optional `stats` out-parameter — an
out-parameter rather than a second return value because the merge state, the
unit tests and the offline replay scripts all unpack a bare list (§9.10).

### Recurrence gates

Grouping is **canonical complete linkage**: `max_recurrence_threshold` bounds
group **diameter**, so every published group satisfies
`max_distance <= max_recurrence_threshold` by construction (§9.11; the earlier
Union-Find/single-linkage grouping violated this in shipped output).
Implementation is Union-Find as a connectivity screen, then canonical complete
linkage per component — provably identical to the global algorithm at equal
thresholds. The default is **0.07**, from a corpus sweep against visually
verified true cards: the safe window is (0.0634, ~0.075] — the widest-diameter
verified card cannot form below 0.0634, and a verified card splits at 0.080.
**A caller passing 0.05 explicitly gets the stricter complete-linkage
semantics at the old number and silently loses the widest-diameter card
class.**

A group must then survive two deleting gates, and carries two published
verdicts a consumer applies itself ([lib/recurrence.py](lib/recurrence.py),
Design record §9.x below). Each targets a false-group shape measured on real
assets, and none subsumes another:

| Gate | Test | Disposition | The shape it targets |
|---|---|---|---|
| §9.2 | content shots only enter the similarity matrix | **drops** | black shots share the identical sentinel centroid (distance 0.0) → one giant fake group; bars/slates recur trivially |
| §9.1 | `min_recurrence_count` applies to temporally-separated OCCURRENCES, gap measured END-to-START | **drops** | dialogue coverage (A/B/A/B inside one scene) — nearly every raw group on scripted drama |
| §9.3 | raw member count vs `max_recurrence_count` (8) | **annotate-only**: consumer derives `recurrence_count > max_recurrence_count` | recurring SETS — news anchor cameras (dozens of members measured), confessional chairs — pass every similarity gate because they ARE near-identical |
| §9.6 | a frame pair from two DIFFERENT members must match on the patch/global dHash family | **flags** `hash_confirmed` true/false/null | same-SET-but-different-render pairs that embeddings alone group |

Blacks are excluded from the **matrix** but kept in the sorted shot list,
because `act_boundary` classification is precisely "has a black neighbour" —
dropping them would silently relabel every card as `interior`. A
`content_rows` index map keeps the two views aligned.

§9.6 also carries the §9.4 burn-in hygiene: a frame is unhashable when its
entropy is below `RECURRENCE_HASH_MIN_ENTROPY` **or** when ≥
`RECURRENCE_MAX_BLACK_FRACTION` of it is black, and a group with fewer than
two hashable members gets `hash_confirmed: null` — unverifiable, deliberately
distinct from `false`.

## Key Design Decisions

### Frame number authority

Frame numbers are computed once in the transcode state via integer PTS
arithmetic:

```python
frame_number = round_to_nearest(pts × Fraction(time_base) × Fraction(frame_rate))
```

Rounded to NEAREST, never truncated (§9.8): containers store integer PTS on
their own timebase, and when ticks-per-frame is fractional (90000 ticks @
24000/1001 fps = 3753.75) every 4th frame's stored PTS sits just below the
exact boundary — truncation duplicated one frame number and skipped the next
on ~21% of positions (measured on a real 23.976 asset), making any cut at
those positions invisible to boundary detection. A contiguity audit WARNs when
the reconstruction is not +1-contiguous. The value flows unchanged through
every downstream state via `_pts.json`. Float arithmetic is never used for
frame numbering.

### Timecode derivation

Shot timecodes are derived in ONE place — the merge state's
`_derive_timecodes()`, just before shots.json is serialized — as pure
arithmetic from frame numbers: `frame_to_smpte(frame, fps, drop_frame,
tc_origin)` over the container's first-frame timecode (the only timecode value
ever READ from the media, in the probe state; a source with no timecode track
counts from `00:00:00:00` NDF). The embedding windows ship no per-frame
timecode strings — they were derived data carried as data, which is how
post-hoc fade/drift splits ended up with blank timecodes and how a drop-frame
arithmetic bug propagated.
[test/timecode_derivation_test.py](test/timecode_derivation_test.py) pins the
arithmetic across 23.976/24/25/29.97DF/30/50/59.94DF/60 and the derivation
semantics. `end_timecode` describes the nominal `end_frame` — consistent with
`frame_range` — which on PTS-hole boundaries differs from `end_time`'s
last-sampled basis. The `sync_pop`/`flash_frame` anomaly records get their
`timecode` from their own `frame` in the same pass.

### Seek capability test

The probe state runs the exact ffmpeg command used for real transcoding on a
1-second slice. This is the only faithful way to detect unseekable formats
(e.g., Sony eVTR Class B MXF). A timeout (>300 s) is treated as seekable —
ffmpeg actively seeking indicates the file is seekable.

| `can_seek` | `is_all_intra` | Result |
|------------|----------------|--------|
| `true` | any | proceed |
| `false` | `true` | proceed (all-intra path) |
| `false` | `false` or `null` | **FAIL** |

### Black frame sentinel

Black frames receive the sentinel embedding `np.full(384, 1/sqrt(384))`
instead of a real DINOv2 embedding. This vector is equidistant from all real
embeddings and prevents black frames from anchoring shot boundaries or
recurrence groups.

### Shot boundary detection

[lib/shot_pipeline.py](lib/shot_pipeline.py) evaluates cosine distances
between adjacent frames. For a run of frames exceeding `cosdist_threshold`,
the interior is scanned for single-frame anomalies (frames whose two
neighbours are similar to each other: distance < 0.10). If found, the anomaly
frame is isolated as a single-frame shot and later healed by merging its
neighbours; the merged shot records it as `sync_pop` when the frame is
monochrome-solid (luma entropy < 1.0 — a true broadcast sync-pop/2-pop) or
`flash_frame` otherwise (lightning, exposure flicker, glitch). Otherwise, peak
absorption places the boundary at the most dissimilar pair — always the first
frame of new content.

### Frame archive

Every frame that receives a DINOv2 embedding is saved at the proxy's native
resolution as JPEG and bundled into a per-window `_frames.tar.gz` archive on
S3. The merge state reads these archives to repackage selected frames (≤16 per
shot) into per-shot `shot_frames/` archives; the debug state extracts the
per-shot archives rather than re-seeking into the proxy with ffmpeg. Why:
timestamp-based ffmpeg seeking at non-integer fps consistently returned the
frame BEFORE the target (seek time lands fractionally below the target PTS),
so every first frame of every debug grid showed the previous shot's content.
Archiving the model-input frames gives integer-exact frame identity and a
visual audit trail of exactly what the model saw.

### Known limitation: dissolves

A dissolve produces 3 shots (shot A, transition, shot B). DINOv2 embeddings
encode *what* is visible, not *how* it changed: a dissolve's embedding
trajectory is indistinguishable from rack focus, camera movement through a
doorway, or lighting changes, so trajectory-shape classifiers false-positive
on legitimate content — worse than the extra transition shot. Impact is
cosmetic (a handful of extra shots per episode; recurrence at worst misses a
blend centroid). A boundary-continuity test (near-zero discontinuity at BOTH
boundaries of the middle shot) is the identified future approach; deferred.

### ORT CPU fix

Lambda removes `/sys/devices/system/cpu/possible` and `/present`. Without
these, ORT crashes on arm64 before the first inference call. The Dockerfile
copies [patch.txt](patch.txt) (`"0-5"`) to both paths;
[lib/ort_init.py](lib/ort_init.py) recreates them at runtime if absent and
derives thread count from Lambda memory tier (1792 MB → 1 vCPU … 10240 MB →
6 vCPU).

## AgentCore Gateway Tool

[gateway_tool/lambda_function.py](gateway_tool/lambda_function.py) is the
AgentCore Gateway dispatcher that wraps this workflow as three MCP tools. It
is deployed as a separate Lambda (flat zip, no Docker), so it cannot import
`lib/`. The helpers it needs are **copied** into the zip as
[gateway_tool/workflow_identity.py](gateway_tool/workflow_identity.py) and
[gateway_tool/bucket_validator.py](gateway_tool/bucket_validator.py); the
copies are the distribution mechanism, byte-pinned across all sibling
workflows by the bundle test suite (`TestWorkflowIdentityParity`,
`TestBucketValidatorParity`) — drift in `workflow_identity.py` silently
re-runs every request at a fresh prefix with nothing erroring.

### Tools exposed

**`segment_shots`** — validates parameters, constructs the nested SFN input,
derives the workflow_id, and calls `StartExecution`.

- All `input_config` fields from the workflow schema are exposed as flat
  top-level parameters.
- workflow_id strategy: `workflow_id = uuid5(canonical JSON of the SFN input
  WITHOUT output.s3uri + "|" + TOOL_VERSION + "|" + idempotency_token when one
  is supplied)` — derived by the shared
  [gateway_tool/workflow_identity.py](gateway_tool/workflow_identity.py). It
  NAMES the execution, so identical requests are idempotent: on
  `ExecutionAlreadyExists` the call is **absorbed** into the existing run — no
  second execution — while any parameter, token, or tool-version change
  re-runs to a fresh prefix. Outputs land at
  `s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{workflow_id}`.
- `idempotency_token` (optional, 1-80 chars of `A-Z a-z 0-9 _ . -`) is the
  caller's assertion "this is the same request I sent before". It rides in the
  hash preimage as extra material and is then **discarded** — never forwarded
  to the state machine, never stored, never echoed. Omitting it still dedups
  on the parameters alone. The retired `uuid` parameter is **rejected** on
  submit (it used to mean both "which asset" and "same request", so a caller
  still sending it cannot be assumed to mean the surviving half) and
  **accepted with a log warning** on `get_job_status` / `describe_tool`, where
  neither meaning could have changed the answer.
- Returns a **HANDLE, never results**: `{"job_id": <execution_arn>, "status":
  "RUNNING", "current_state": <active stage>, "workflow_id": <derived id>,
  "output_s3_prefix": <derived prefix>, "asset_s3uri": <source s3uri>}`. An
  absorbed resubmit returns *the same fields* — only `status` differs (the
  existing run's, which may be terminal), plus `message` (present only when
  absorbed, and the only thing distinguishing the two cases — the `job_id` is
  a hash of the same input either way) and `error` on a terminal failure.
  `current_state` is dropped once the run has stopped, because a stopped
  execution is in no state and reading one costs a `GetExecutionHistory`
  paging pass for an `"unknown"`. Results come from `get_job_status` alone;
  that is what makes the declared `OutputSchema` equal to the reachable field
  set. The SFN input still carries a `uuid` key (set to the workflow_id)
  because every state derives its output prefix from it; that is this
  execution's local id, not the caller's token.

**`get_job_status`** — calls `DescribeExecution`; when RUNNING, pages
`GetExecutionHistory` to find the active pipeline stage. When SUCCEEDED,
returns `workflow_id`, `output_s3_prefix`, the primary output keys
(`shots_key`, `embeddings_key`, `centroids_key`, `recurrence_key`), content
facts (`n_shots`, `n_recurrence_groups`, `total_frames`,
`actual_duration_sec`), cost/timing (`pipeline_wall_sec`,
`total_estimated_cost`, `usage_key`), and a self-describing `outputs` list of
`{uri, mime, label, desc}` entries (labels: `shot-list`, `frame-embeddings`,
`shot-centroids`, `recurrence`, `usage`, plus the `debug-grids` prefix).
`desc` is the one-line role description joined verbatim from
[gateway_tool/output_contract.json](gateway_tool/output_contract.json) — the
same file `describe_tool` serves in full, so the two surfaces cannot disagree;
it is omitted, never fabricated, for a label the contract lacks. **Every
branch** — RUNNING, SUCCEEDED, every failure, and the fallback for a status
Step Functions has not shipped yet — carries `job_id`, `workflow_id`,
`asset_s3uri`, and `output_s3_prefix`, so a status reply always names its own
run and its output location even when there is nothing to report yet.
(`output_s3_prefix` is read back from the execution's *input* whenever the
workflow's own `proxy_bucket`/`proxy_prefix` report is missing, which is why a
RUNNING or failed run still has one — partial results already live under it.)
Terminal failures add `error`.

**`describe_tool`** — returns `{"version": <TOOL_VERSION>,
"state_machine_arn": <ARN>, "description": <short backend description>,
"output_contract": <parsed output_contract.json>}`. Version is stamped at
deploy time via the `TOOL_VERSION` env var (set from the `.version` file by
`build-deploy-s3-dist.sh`); the ARN gives callers a console/CLI entry point
without parsing a job_id.

`output_contract` (~4.5 KB) is the field-level contract for every `outputs[]`
label: basename, kind (`file`/`prefix`), mime, the one-line `description` that
doubles as the entries' `desc`, and a compact single-element `example` of the
file's contents (`contents` block for the binary .npz/.npy instead). It is
static per `TOOL_VERSION` — the file ships in the gateway zip
(`build-deploy-s3-dist.sh` pins the member), so version and contract cannot
drift — which is why it lives here and not in `get_job_status`: a caller
caches it once per session, and CI can pin it without running a job. Contract
keys == emitted labels, exactly; anything not listed (per-window
intermediates, `proxy/` segments, `probe.json`) is internal and may change
without notice. The examples were authored from the producing code, not the
docs — note `shots.json`/`recurrence.json` carry envelopes (`uuid`,
`n_shots`, `validation`, … / `total_groups`, …) around the entry arrays; the
schema blocks earlier in this file show only the entries. The contract's
`recurrence` notes also carry the genre posture (§9.5) and the
`recurrence_count` vs `occurrence_count` semantics, so a consumer receives the
reading guidance with the data.

### Tool routing

AgentCore passes `context.client_context.custom["bedrockAgentCoreToolName"]`
as `"{target_name}___{tool_name}"`. `_resolve_tool()` strips the target prefix
and returns the bare tool name.

### Agent system prompt

The companion system prompt for an agent using this target lives at
[../media-toolkit-gateway/prompts/shot_segmentation_agent.md](../media-toolkit-gateway/prompts/shot_segmentation_agent.md).

## Build & Deploy

- [build-deploy-s3-dist.sh](build-deploy-s3-dist.sh) — runs the unit tests
  (packaging is gated on them), builds the Docker source zip + gateway tool
  zip + processed CFN template, and deploys them to S3. Token substitutions in
  cfnstack.yaml: `%%PKG_DOCKER_SOURCE%%`, `%%PKG_GATEWAY_TOOL%%`,
  `%%BUCKET_NAME%%`, `%%KEY_PREFIX%%`, `%%VERSION%%`.
- [create_update_stack.sh](create_update_stack.sh) — create-or-update with 3
  required params: `--resource-prefix`, `--ingest-bucket`, `--proxy-bucket`.
- `.version` stamps `TOOL_VERSION`.

### Model export (offline)

[export/export_dinov2_onnx.py](export/export_dinov2_onnx.py) converts the
DINOv2 ViT-S/14-reg PyTorch checkpoint (via `torch.hub`) to ORT-optimised ONNX
FP32. Run outside the container in a GPU-capable environment:

```bash
python export_dinov2_onnx.py
# produces model.tar.gz containing dinov2_vits14_reg_cls.onnx
```

Upload the tarball to
`s3://{model_bucket}/{model_prefix}/dino_v2/vits14_reg/model.tar.gz` before
building (the Docker build bakes it into the image).

## Tests

```bash
cd shot-segmentation-on-aws
python3 -m unittest discover -s test -p '*test*.py'
```

`test/` is not a package — run with discovery or directly
(`python test/recurrence_gates_test.py`), not `-m unittest test.<module>`.
`test/stress_test.py` drives a live deployed stack and needs AWS credentials
and a region. `test/harness/` holds offline replay/measurement tooling (replay
recurrence against a finished run's own artifacts, threshold sweeps, feature
scoring) — local tooling, never shipped.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENV_PROXY_BUCKET` | — | S3 proxy bucket; defined in `validate_and_normalize()` as a fallback but unreachable — `output.s3uri` is required |
| `SF_EXECUTION_ARN` | — | Step Functions execution ARN; injected by Step Functions context |

## Gotchas

- **Input can be any format ffmpeg supports, but unseekable non-all-intra
  files are rejected.** The probe state sets `can_seek=false,
  is_all_intra=false` and the pipeline fails. There is no workaround short of
  re-muxing the source.
- **`Extract frame features` checkpoints across Lambda's hard 900 s ceiling —
  a `LambdaResourceTimeoutException` is a NORMAL path, not a failure.** One
  window on very high-cut-rate content can need more than 900 s of DINOv2 work
  and the ceiling cannot be raised, so the batch loop self-times: before each
  ~120 s mini-batch it reserves room for one more batch (2× the slowest batch
  this invocation) plus the checkpoint upload (90 s), and when short it packs
  all accumulated state — embeddings, per-frame metadata, the persisted JPEGs
  the frames archive needs, the seam-holdover frames — into
  `{window}_checkpoint.tar` on S3 and raises. The ASL retries 10×; each retry
  resumes from the checkpoint and spends its budget on NEW batches. Three
  things not to "fix": the exception class name in
  [lib/custom_exceptions.py](lib/custom_exceptions.py) is matched BY NAME in
  both definition copies (renaming it silently disables the retry);
  `Lambda.Unknown` is `MaxAttempts: 2` *because* the checkpoint exists (it was
  0 when a retry could only repeat identical work — do not raise it further, a
  deterministically oversized single batch would burn 900 s per attempt); and
  the checkpoint tar is ephemeral internal state, deleted on success and
  absent from the output contract. Splice correctness (interrupted twice ≡
  uninterrupted, bit-exact) is pinned by
  [test/extract_resume_splice_test.py](test/extract_resume_splice_test.py).
- **`EMPTY` segments are normal**, not errors. A seek beyond the actual EOF
  returns 0 bytes from ffmpeg. The dynamic chunking state filters them
  silently.
- **30 s window overlap produces duplicate frame files in the per-window
  archive.** The merge state uses a frame-to-window index that assigns each
  frame to the first window that persisted it — duplicates in the overlap
  region are never repackaged twice.
- **Map state concurrency is hardcoded to 40** in `definitions.json`. If AWS
  Lambda concurrency limits are lower in a given account, transcode or
  embedding Map invocations will queue. Adjust `MaxConcurrency` in the state
  machine JSON, not in code.
- **Usage tracking bills timed-out invocations.**
  `states/usage_tracking/lambda_function.py` once dropped any invocation with
  `succeeded_ts is None`, so a `TaskFailed` — a Lambda killed at its timeout,
  the most expensive invocation possible — contributed 0 s while
  `invocations` stayed correct, so the output read as precise. This stack's
  checkpoint/resume states make a timeout a normal path, so it fired on
  ordinary runs. **Two invariants to preserve:** `_billed_sec` falls back to
  `failed_ts`, and 0.0 is correct **only** when an attempt has NO terminal
  event at all — that case is what `incomplete` means. `wall_sec` sums
  attempts per `entered_id` *before* taking the max across occurrences,
  because every retry attempt chains back to the same `TaskStateEntered` while
  concurrent Map iterators each get their own. The sibling `tts-on-aws`
  workflow holds the reference copy; the behaviour is pinned cross-stack by
  the bundle test suite (`TestUsageTrackingBillsFailedAttempts`) — a
  regression fails the bundle suite, not this stack's.
- **Usage tracking cost constants** in
  `states/usage_tracking/lambda_function.py` are us-east-1 arm64 figures.
  Update if deploying to another region or after pricing changes. The
  estimates also include ~10–50 ms of Step Functions scheduling overhead per
  invocation.
- **MXF files run `mxf2raw`** (BMX tool, bundled in the image) for a more
  accurate `is_all_intra` determination. If `mxf2raw` is absent or fails, the
  pipeline falls back to codec-tag heuristics.
- **Black off a tape is not low-entropy.** The default `black_entropy_max`
  (0.20 bits) is right for clean digital sources but no frame off a real tape
  can satisfy it: 0.20 requires ~97% of pixels in a single luma code — true
  only of synthetically generated black. Black recorded to tape and
  redigitised is dithered across several low codes by transport and encoder
  noise, measuring roughly 0.4–2.8 bits while every pixel still sits below
  luma 17. The symptom is flat-black shots typed `content`, each dragging the
  start of whatever segment it heads. **The default stays 0.20 — the
  validated per-request override for redigitised-tape sources is
  `black_entropy_max: 3.5`** (in `input_config` or as the flat `segment_shots`
  parameter). Raising it is safe because blackness is really decided by
  `black_coverage_pct` + `black_max_bright_pixel`; entropy is a backstop with
  no separating power left once those two agree (criteria 1+2 cap entropy at
  log2(31) = 4.95, so 3.5 still binds while anything ≥4.95 is a silent no-op).
  Pinned by [test/black_detection_test.py](test/black_detection_test.py).
  Related: `near_black_entropy_max` (1.5) is a safety margin, not a classifier
  — it cuts through the middle of the real-black entropy distribution, and
  within-shot spread dwarfs the margin, so it is left at 1.5 deliberately
  (retuning trades missed blacks for content misread as black; the durable fix
  is persisting a per-frame luma statistic alongside `entropy`).
  `SYNC_POP_ENTROPY_MAX` (1.0) is probably too low for the same reason, so
  genuine 2-pops may be labeled `flash_frame`; left alone — both labels heal
  the anomaly identically, so no boundary accuracy is at stake.
- **A single-frame shot bounded by black is not a continuity violation.** The
  continuity validator skips a single-frame shot when either neighbour is
  black. Two things collide without this guard: black shots all carry the
  identical sentinel centroid, so the neighbour distance is identically 0 and
  always trips the sync-pop threshold; and every merge pass deliberately
  refuses to merge across black, so a merged shot cannot come out `content`
  while spanning black frames. The check otherwise demanded a heal the
  pipeline is designed never to perform — and failed black-separator-dense
  broadcast reels outright. The check still fires for a genuine unhealed
  sync-pop between two *similar content* shots, which is what it was written
  to catch.
- **The §9.6 confirmation statistic is global-OR-median, never max-of-patches
  — and that is measurement, not taste.** `lib/frame_hash.py` came from
  face-recognition-on-aws where the *max* stable-patch distance is the
  cut-break veto, so reusing it as the confirmation threshold looks like the
  obvious port. It false-rejects genuine re-renders: measured on a real
  act-break card, cross-member stable-patch **max** runs 7–13 — and reaches
  **19 between adjacent frames of a single shot** — while whole-frame global
  reads 0–7 and the median 1–6. The first implementation used max ≤ 6 and
  dropped the one group the whole gate exists to keep. A threshold validated
  for "is this a CUT?" does not transfer to its inverse "is this the SAME
  image?" even on the same hash family. Constants and the measurement table
  live in [lib/input_schema.py](lib/input_schema.py).
- **`lib/frame_hash.py` is a BYTE-IDENTICAL copy of face-recognition-on-aws'
  file, and its constants deliberately live elsewhere.** The four module-level
  values it imports (`PATCH_GRID`, `PATCH_NOISE_CHANGE_FREQ`,
  `PATCH_NOISE_MAX_DISCOUNT`, `GLOBAL_HASH_SANITY_FACTOR`) are re-declared in
  `lib/input_schema.py` so the copied file needs no edit to travel between
  workflows — parity stays an `md5` comparison instead of a diff review. Fix a
  bug in the hash family and copy the whole file across; do not patch one
  side.
- **The §9.3 ceiling never had a safety margin, which is why it no longer
  drops.** When it deleted at `> max_recurrence_count` (8), a genuine title
  card in the validation corpus sat at **exactly 8 members** and survived by
  one, while the smallest false group anywhere in the corpus was n=9 — and
  verified cards reach 24 members on 2-hour blocks, so the drop destroyed half
  of them. Nothing is deleted for size any more: the knob is a published
  reference value and the consumer derives the verdict. The retune rule
  survives the change: never move `min_recurrence_count` or
  `max_recurrence_threshold` without re-running the verified-card corpus
  (§9.7).
- **A timecode-burn window is the most confirmable thing in the corpus, which
  is why §9.4 exists.** Measured on a real burn master: with black-dominance
  disabled the burn windows cross-compare at **global hamming 0, median 0** —
  tighter than any genuine card (0–3). The timecode increments every frame, so
  the glyphs are the *only* difference, and an 8×8 whole-frame dHash cannot
  resolve a glyph that small. §9.6 would have rated a fake group its most
  confident. With the test enabled those windows yield zero hashable frames
  and the group gets `hash_confirmed: null`. Never reason about this class
  from "surely the hash will notice".
- **The entropy floor does NOT cover timecode-burn black, and cannot.** Those
  masters are ~99.2% black with burned-in timecode, and the burned glyphs are
  genuine texture, so entropy reads *above* any useful floor while every frame
  in the burn window hashes identically — that inversion IS the defect (§9.4).
  The gate therefore applies two independent unhashable tests, entropy **and**
  black dominance (`RECURRENCE_MAX_BLACK_FRACTION`, measured from the luma
  already decoded for hashing, so it costs no I/O). Removing either one leaves
  a real class of fake groups; both directions are pinned by unit tests
  (`test_timecode_burn_black_drops_despite_high_entropy`,
  `test_dark_but_not_black_dominated_card_still_confirms`). On the measured
  burn master §9.4 is defence-in-depth behind §9.2 (the burn windows classify
  `black` upstream and never enter the matrix) — but a **content**-classified
  shot that is nonetheless black-dominated (a near-black glitch/fade shot) is
  reachable and only §9.4 covers it. The margins on both constants are thin
  (a few percent on measured real shots) and the two tests fail in opposite
  directions on purpose; that redundancy is the design.
- **`confirm_groups_by_hash()` takes I/O as a callable on purpose.**
  `load_lumas(shot_id)` returns `(frame_num, 2-D uint8 luma)` pairs; the merge
  state owns the S3 read and the tar/PIL decode, the lib owns only the
  decision. That is what makes the gate unit-testable with synthetic frames
  and replayable offline against a finished run's own artifacts — which is how
  the max-of-patches error was caught before deployment rather than after.
  Keep S3 out of `lib/recurrence.py`.
- **The OR in the §9.6 predicate is load-bearing; median-only would reject the
  disqualifying positive.** Measured live: a genuine act-break card whose art
  is identical but whose small digit and lower-third change scores global
  **0** but stable-patch median **10**, above `RECURRENCE_HASH_MAX_MEDIAN` (6)
  — an 8×8 whole-frame dHash cannot resolve a small glyph while a 3×3 patch
  containing it can. The median exists for the *complementary* case, tickers
  that corrupt the global hash. Neither statistic alone passes the corpus.
- **Intra-shot global distance is LARGER than the inter-member rejection band,
  so "11 is nearly 10" is not an argument.** On live news, adjacent frames of
  a single member reach global 34–36. The groups §9.6 rejects score 11–23 —
  which looks marginal against a threshold of 10 until you notice one shot's
  own frames scatter three times as far. Any proposal to loosen
  `RECURRENCE_HASH_MAX_GLOBAL` has to answer this; §9.9 explains why
  animated-furniture false flags need a different mechanism instead.
- **The two gates that still drop do so silently in the Lambda logs; the
  `gates` object is the only attribution.** The lib-level `log.info` lines do
  not propagate out of the Lambda (only the state's own summary lines do),
  which is why the merge state re-logs the counters itself and publishes them
  in `recurrence.json`. Since §9.3 annotates and §9.6 flags, only min-members
  and §9.1 delete, and `gates.groups == total_groups`. `TestGateAttribution`
  pins the per-gate counts. Attributing a drop still needs an offline replay
  when you want the *members* rather than the count; that replay reproduces
  live group counts exactly and is how §9.9 was found.
- **A recurrence gate can be a no-op on an asset and still be correct.**
  Locked-off confessional-interview groups on reality content genuinely ARE
  the same render — confirming them is right; the genre posture (§9.5), not a
  gate, is what tells consumers to treat them as candidates. Do not "fix" a
  gate because it dropped nothing on the asset in front of you.
- **Internal docs stay internal.** `GOTCHAS.md` (untracked) and
  `test/harness/runs/` carry identities the code and this README deliberately
  refer to only by generic corpus label; `assert_internal_docs_excluded` in
  the build script fails the build if a package ever includes them.

---

## Design record

Numbered decisions cited from code and tests (`README.md §N`). Each entry
carries the decision and its why; the operational constants are pins —
measured on a multi-genre validation corpus; do not change them without
re-measuring.

### §3 — Frame-hash family: 3×3 patch dHashes + global dHash, with a noisy-patch discount

[lib/frame_hash.py](lib/frame_hash.py) computes, per frame, nine 64-bit patch
dHashes (row-major 3×3 grid) plus one whole-frame global dHash on downsampled
luma — each dHash is a 9×8 resize followed by horizontal gradient sign.
Microseconds per frame, numpy-only. Patches whose hash churns in >
`PATCH_NOISE_CHANGE_FREQ` (0.70) of adjacent pairs across the asset
(lower-third tickers, scoreboards, clocks, burned timecode) are classified
chronically noisy and excluded from stability comparisons, capped at
`PATCH_NOISE_MAX_DISCOUNT` (3) of 9 — more churn than that means genuinely
dynamic content and no discount. The skip predicate (used by
face-recognition's probe-then-fill, where this file originates) requires every
adjacent stable-patch distance in a gap, the endpoint-to-endpoint stable
distance (slow drift), and the endpoint global distance (< 2× threshold,
backstop when patches are discounted) all under threshold. The file is a
byte-identical copy shared with face-recognition-on-aws; its constants live in
`lib/input_schema.py` as the import surface so the file travels unedited.

### §9.1 — Temporal-separation gate: occurrences, not raw members

`detect_recurrence()` counting raw shots forms false groups from dialogue
coverage (A/B/A/B shot-reverse-shot inside one scene) — on a measured scripted
episode, nearly every raw group was dialogue coverage. Fix: members closer
than `min_occurrence_gap_sec` (60.0) collapse into ONE occurrence cluster;
`min_recurrence_count` (3) is enforced on CLUSTERS. The gap is measured
**END-to-START** (previous member's end to next member's start), not
start-to-start — a long dialogue shot must not promote its own coverage into a
second occurrence by being long (regression:
`test_gap_is_end_to_start_not_start_to_start`). This gate removes an order of
magnitude more groups than any other; published groups carry
`occurrence_count` so consumers can apply the distinction themselves.

### §9.2 — Content shots only in the similarity matrix

All `shot_type == "black"` shots share the identical sentinel centroid
(distance 0.0), so any `min_count` of them forms one giant fake group
(verified numerically); `static` shots (bars/slates) recur trivially. Both are
excluded from the centroid matrix — but NOT from position classification,
which needs black neighbours to label `act_boundary`.

### §9.3 — Set-recurrence ceiling: annotate, never drop

Recurring SETS — news anchor cameras (measured at dozens of members),
confessional chairs — pass every similarity gate because they ARE
near-identical content; structural furniture (act-break cards) recurs 3–7
times. `max_recurrence_count` (8) marks the boundary. As a hard drop it had
**zero safety margin**: a genuine verified title card sat at exactly 8 members
while the smallest false group measured was n=9 — adjacent values, no gap —
and verified cards on 2-hour blocks reach 24 members, so the drop destroyed
half of the corpus's verified furniture. Member count is also the weakest
measured discriminator of furniture vs set (9th of 20 features scored;
`ent_range` and `coverage` separate far better and are derivable from the
shots.json join). The ceiling therefore **annotates only**: the knob is
published in the envelope and the consumer derives
`recurrence_count > max_recurrence_count`. Retune rule: never move
`min_recurrence_count` or `max_recurrence_threshold` without re-running the
verified-card corpus.

### §9.4 — Burn-in / black-dominance hygiene (the texture floor's blind spot)

Timecode-burn masters defeat boolean black detection: ~99.2% black pixels plus
burned-in timecode text reads entropy ABOVE any useful floor (the glyphs are
genuine texture — that inversion IS the defect) while every frame in a burn
window hashes identically, manufacturing one fake group per window — measured
as a *perfect* false confirmation (global hamming 0), tighter than any genuine
card. Mitigation: a frame with ≥ `RECURRENCE_MAX_BLACK_FRACTION` (0.90) black
pixels is UNHASHABLE for confirmation purposes, measured from the luma already
decoded for hashing (zero extra I/O); "black" reuses
`DEFAULT_BLACK_PIXEL_THRESHOLD` on the 0-255 scale so both detectors agree. A
group with fewer than two hashable members gets `hash_confirmed: null`.
Usually §9.2 removes burn windows upstream (they classify `black`), making
§9.4 defence-in-depth — but a content-classified yet black-dominated shot (a
near-black glitch/fade) is reachable and only §9.4 covers it. The entropy
floor and black-dominance test fail in opposite directions on purpose; both
are pinned by mutation-verified tests, and neither subsumes the other.

### §9.5 — Genre posture (consumer guidance, not a gate)

Position labels are `head/tail/act_boundary/interior`. Recurrence output is
trustworthy structural evidence on scripted episodic content; on news/reality,
small set-recurrence groups (n ≤ 8) can pass all gates — statistically
indistinguishable from cards by visual+temporal evidence alone — so consumers
should treat interior groups on such content as candidates, not verdicts. The
posture ships as a `notes` field on the `recurrence` label in
[gateway_tool/output_contract.json](gateway_tool/output_contract.json)
(served by `describe_tool`, joined per-output by `get_job_status`), together
with the `recurrence_count` vs `occurrence_count` semantics — the distinction
a caller needs to read a reality-TV group correctly.

### §9.6 — Hash confirmation gate: same-RENDERED-IMAGE, not same-set

Embeddings alone confirm same-SET content; a hash family confirms
same-RENDERED-IMAGE — which is what separates an act-break card (the same
render coming back) from a recurring location. The gate hashes up to
`RECURRENCE_HASH_MAX_FRAMES` (8) textured frames per member (§3 family;
noisy-patch discount handles tickers/clocks on broadcast cards) and flags
`hash_confirmed` true/false/null (see §9.9 for the flag semantics).

The confirmation STATISTIC was corrected by measurement: the first
implementation used the stable-patch MAX (≤ 6) by analogy with
face-recognition's cut-break veto, and offline replay against a real verified
act-break card dropped the one group the gate exists to keep. Measured on that
card: cross-member global hamming 0–7, stable-patch median 1–6, stable-patch
max 7–13 (and up to 19 between adjacent frames of ONE shot) — max-of-patches
is a BREAK statistic, not a confirmation statistic. Corrected predicate: a
pair confirms on global ≤ `RECURRENCE_HASH_MAX_GLOBAL` (10) **OR**
stable-patch median ≤ `RECURRENCE_HASH_MAX_MEDIAN` (6). Live corpus
separation: confirmed groups' best pair reads global 0–3, rejected groups
11–23 — the threshold of 10 sits in an observed-empty band. The OR is
load-bearing in both directions: a card whose art is identical but whose small
digit/lower-third changes confirms on global while its patch median exceeds 6
(an 8×8 whole-frame dHash cannot resolve a small glyph; a 3×3 patch can), and
a ticker-corrupted global is rescued by the discounted-patch median. The
general lesson: a threshold validated for one decision ("is this a CUT?") does
not transfer to its inverse ("is this the SAME image?") even on the same hash
family. The gate's honest boundary: it separates same-set-DIFFERENT-render; it
correctly confirms locked-off same-render shots (confessionals), which §9.5's
posture — not this gate — tells consumers how to read. Text animating over a
fixed plate confirms across animation phases; what defeats the gate is the
whole PLATE moving (a pan or re-crop) — see §9.9.

### §9.7 — Validation corpus and the picture-only decision

Acceptance for the recurrence gates runs against a multi-genre ground-truthed
corpus: scripted episodic drama (clean single-card positives — any regression
on those is disqualifying), a short-film **negative control** (0 groups),
reality with confessional set-recurrence (adversarial: passes similarity gates
under the ceiling; posture test), a timecode-burn master (the §9.4 test), news
studio masters (anchor-camera set recurrence over the ceiling; large counts of
small groups that are a posture test, not a gating test), documentaries,
sports, and feature films. Acceptance: exact-match on the verified true cards,
zero groups on the negative control, anchor/confessional set groups over the
ceiling annotated, dialogue-coverage groups absent everywhere.

The same corpus settled tail detection as **picture-only**: a loudness/silence
pass looked like a cheap, definitional discriminator for textless elements
("no text and no audio") and was measured and refuted —
[test/harness/audio_probe.py](test/harness/audio_probe.py) is the evidence. A
textless title block ships WITH its music bed, making it audio-level
indistinguishable from programme, so "programme ends at last non-silent audio"
lands late; meanwhile the silent moat is already published as a black shot, so
audio adds nothing picture does not have. (The one place audio WOULD add
signal is head leader — bars+tone and slate classify as `content` — and that
needs a 1 kHz tone test belonging to whatever consumes audio, not here.)

### §9.8 — PTS → frame-number rounding, and small-gap boundaries

A live fused-shot incident (a cross-cut conversation bridged by a downstream
face tracker) was root-caused NOT to the embedding model — both missed cuts
produced adjacent-sample distance spikes at ~2× the 0.20 threshold — but to
the frame-number reconstruction:

1. **Round to nearest, never truncate** (the root cause).
   `_parse_showinfo_log` computed source frame number as
   `int(pts × time_base × frame_rate)` — truncation. With fractional
   ticks-per-frame (90000 ticks @ 24000/1001 fps = 3753.75), muxers store
   integer PTS that sit just below the exact boundary on every 4th frame, so
   truncation produced duplicate frame numbers and holes on ~21% of positions
   in a strict every-4th-frame pattern — each one a blind spot where no
   gap==1 sample pair can exist for boundary detection. Fix: round to nearest
   (`int(fn + Fraction(1, 2))`), one summary log, plus a contiguity audit
   that WARNs loudly when the reconstruction is not +1-contiguous.
   Regression: [test/pts_frame_numbers_test.py](test/pts_frame_numbers_test.py)
   synthesizes muxer-rounded showinfo logs (the 23.976 case fails under
   truncation, mutation-verified).
2. **Small-gap boundary acceptance** (defence-in-depth for mapping defects
   and genuinely-VFR input). `_find_shot_boundaries` originally discarded any
   spike over a non-unit frame gap, whatever its magnitude. Now boundaries
   are accepted for gaps 0..`SMALL_GAP_BOUNDARY_MAX` (3) when the distance
   exceeds the threshold, placed at the right-hand sample with the ±(gap−1)
   imprecision logged; gap-0 duplicate pairs (two DIFFERENT images sharing a
   frame number — the distance proves it) are the same signal in disguise. 3
   is safe because the smallest DELIBERATE sampling gap is the coarse stride
   (≥ 6). A closing shot's `end_frame` is `frame_index[boundary_row] - 1` —
   byte-identical for gap-1 boundaries and absorbing the unsampled hole into
   the LEFT shot for small-gap ones (found live: a hole frame belonging to no
   shot fails the merge state's continuity validator). Pinned by
   [test/small_gap_boundary_test.py](test/small_gap_boundary_test.py).
3. **A dHash second opinion for sub-threshold cuts was measured and REFUTED —
   do not build it.** Sampled adjacent pairs across distance classes show the
   distributions overlap where it matters: the bottom decile of confirmed
   cuts sits under the top decile of same-shot motion, and the target class
   (similar-framing cross-cuts just over threshold) measures INSIDE the
   motion band. The strictest viable joint rule measured ~6% precision —
   at corpus scale it would inject on the order of a thousand false
   boundaries to recover a handful of real cuts. Structural conclusion, same
   as §9.6's: a dHash answers "is this the same RENDERED IMAGE?", and
   "different image" does not mean "cut" — fast motion produces different
   images inside one shot at exactly the hash distances a soft cut produces
   across two.

### §9.9 — Report, don't drop (flags over deletion)

Two groups the hash gate rejected on live news were genuinely structural
furniture, just not *identically rendered* furniture — a day-of-week
transition graphic sampled at different phases of its pan, and a recurring
still used as a full-frame backdrop at different zoom/crops. Both are the
specified behaviour ("is this the same rendered image?" — honestly no), both
are exactly what a furniture-seeking consumer wants, and under drop semantics
the failure was silent. Do NOT fix this by raising
`RECURRENCE_HASH_MAX_GLOBAL`: the live separation (confirmed 0–3, rejected
11–23, these two at 12) means any threshold admitting them also admits the
correct rejections, i.e. disables the gate. Three options, resolved:

- **Option 1 — crop/scale-invariant second opinion: DECLINED (do not
  build).** Both cases differ by an affine transform a dHash cannot see
  through by construction; the two known cases stay `hash_confirmed: false`
  by design, visible to consumers.
- **Option 2 — even-span hash sampling: IMPLEMENTED.** Hash frames are
  sampled evenly across each member's archived span instead of first-N, at
  the same frame budget. Measured NOT to rescue the two known cases (a
  continuous pan/re-crop moves the whole plate, so no phase pair is the same
  render) but kept as a strict coverage improvement: it fixes the adjacent
  class where one member archives only the tail half of an animation cycle
  (`test_animated_card_confirms_across_phase_windows` — first-N structurally
  could not match that shape).
- **Option 3 — report-not-drop: IMPLEMENTED,** and applied to §9.3 in the
  same change. `confirm_groups_by_hash` decides but never deletes: every
  group publishes with `hash_confirmed` true / false / null (null =
  unverifiable, deliberately distinct from false), group ids stable through
  the gate; the ceiling annotates (§9.3); member entries trimmed to
  `{shot_id, position}` since every other per-member field was a copy of
  shots.json joinable by shot_id. This is the option most consistent with
  the workflow's posture (emit evidence, do not make scene decisions), and
  it would have made the finding visible in output instead of requiring an
  offline replay.

Acceptance for any future change here is judged on the FLAGS, not group
counts: verified cards stay `hash_confirmed: true`, the correct rejections
stay `false`, the negative control stays at 0 groups.

Useful consequence, measured: **`hash_confirmed: true` + over-the-ceiling is a
usable ad-break-countdown-slate signature with zero new code** — the slate
plate IS the same render across breaks and the churning teaser/countdown
regions are exactly what the noisy-patch discount exists for. Visual audits of
the published pool also showed furniture is two classes with opposite coverage
behaviour (cards/idents at very low coverage; break slates at set-like
coverage), that low coverage is a ranking signal rather than a verdict, and
that furniture recall needs the flag + frames, not scalars — which is why the
consumer-facing filter guidance lives in
`gateway_tool/output_contract.json`'s recurrence notes.

### §9.10 — Per-gate observability

A group can die at four places inside `detect_recurrence()` and a fifth in
`confirm_groups_by_hash()`, and originally only the last was reportable; the
lib's `log.info` lines do not propagate out of the Lambda, so "why is my card
missing?" required an offline replay just to learn WHICH gate ate it — the
cost that hid §9.9 for a full batch. `detect_recurrence()` takes an optional
`stats` out-parameter (`content_shots`, `excluded_non_content`, `components`,
`min_members_dropped`, `occurrence_gate_dropped`, and survivors `groups`); the
merge state re-logs the counters and publishes them as `recurrence.json`'s
`gates` object. Singletons are excluded from `components` /
`min_members_dropped` — on a 1000-shot asset most shots are unique, and
counting them buries the numbers that matter. The ceiling check also moved to
run AFTER §9.1: both are conjunctive predicates over the same component, so
the surviving set is identical either way (verified corpus-wide), but with the
ceiling first its counter claimed every oversized component §9.1 was about to
kill anyway — an over-attributing counter is worse than no counter, because it
points a debugging session at the wrong gate. `TestGateAttribution` pins the
per-gate counts and the order-independence of the output. Corpus shape worth
remembering: §9.1 removes an order of magnitude more than the ceiling marks,
and min-members (pairs) dwarfs both — most near-recurrence in real programming
is pairs.

### §9.11 — Complete linkage: the threshold bounds group DIAMETER

Union-Find over pairwise distance ≤ threshold is **single-linkage**
clustering: membership needs *some* pair within threshold, so group diameter
is unbounded and groups chain transitively into mixed bags. The contract is
"one recurring element per group" — a diameter constraint — and the mismatch
shipped: published groups measured diameters up to 2.6× the threshold, and one
recurrence.json carried `max_recurrence_threshold: 0.05` with
`max_distance: 0.1315` in the same document. Verified true cards are tight
(diameter up to 0.0634); ceiling-culled false groups are not.

The fix is **canonical complete linkage** — merge the pair of clusters whose
complete-linkage distance is smallest each round, stop when the smallest
exceeds t. Canonical, not greedy: merging the *first* admissible pair is
order-dependent and gave a materially wrong answer in the first harness
implementation (it reported no viable threshold where a clean window exists);
[test/harness/linkage.py](test/harness/linkage.py) is the single shared
implementation for that reason. Union-Find is retained as a lossless
connectivity screen (a diameter-bounded cluster is a clique, hence inside one
component; cross-component cluster pairs always contain a >threshold pair), so
clustering runs per component — but only at the SAME t: refining a component
found at a lower t silently undercounts, and `min_recurrence_count` must relax
until after refinement (a subset can have MORE occurrences than its parent).

Tightening "some pair within t" to "every pair within t" makes the same number
stricter, so the default moved 0.05 → **0.07**: the corpus sweep
([test/harness/sweep.py](test/harness/sweep.py)) puts the safe window at
(0.0634, ~0.075] — the widest verified card has diameter 0.0634 and cannot
form below it; another verified card splits at 0.080; 0.065 sits only 0.0016
above the failure point. Group count goes UP under the constraint, because
chained components break into several legitimate groups — the intended effect,
not a regression. Complete linkage does NOT solve the recurring-set problem
(the largest set cluster is genuinely tight and inside any threshold), and
over-ceiling group counts grow at the wider threshold — chaining and the
ceiling are independent defects, which is why §9.9's annotate-instead-of-drop
followed.

Performance: `complete_linkage` is masked-in-place with a per-row minima cache
(each row caches its minimum and first-attaining column; a merge recomputes
only its own row plus rows whose cached minimum sat in a merged column) —
identical output to the naive form on thousands of matrices including every
real candidate matrix, and orders of magnitude faster on the degenerate
every-pair-within-threshold shape (a locked-camera lecture). One measured dead
end recorded here so it is not retried: masking alone without the row cache
bought NOTHING — the per-merge rescans, not the allocations, were the cost. Do
not switch algorithms or add a SciPy dependency: the canonical tie-break order
is load-bearing (replays and recorded measurements depend on it).
`TestDiameterConstraint` pins chain-does-not-bridge, the published-diameter
invariant, and canonical-vs-greedy.

### §9.12 — Scope boundary: intra-asset recurrence cannot see cross-episode furniture (do not "fix")

A scripted drama's title sequence, production idents and end board appear once
per episode, so its furniture recurs *across* episodes; a detector whose
entire input is one asset structurally cannot see it, and series episodes
scoring **zero groups is the correct answer**, not a miss. Two consequences:
do not loosen `min_recurrence_count` to chase these (at 2 the corpus gains
every two-shot coincidence in every asset — §9.10's pair count); and
cross-episode furniture is a different product needing a series-level index
over `shot_centroids.npy` from multiple runs, belonging to whatever consumes
this workflow. Corollary for judging output quality: an asset legitimately
scoring 0 groups is not evidence the gates are too tight, and a dense factual
asset scoring many is not evidence they are too loose — genre determines the
available recurrence, which is what §9.5's posture note tells consumers.

### §9.13 — Extract-frame-features checkpoint/resume

One window of very high-cut-rate content can deterministically exceed Lambda's
hard 900 s ceiling (the zoom-in densifier samples thousands of extra frames),
and the timeout cannot be raised. The fix makes a timeout a NORMAL path:

- **Self-timeout, never the hard kill.** Before each ~120 s mini-batch the
  loop checks the remaining budget against ONE MORE BATCH — estimated as
  `EXTRACT_BATCH_TIME_FACTOR` (2.0) × the slowest batch measured THIS
  invocation, so a cheap batch can never talk the gate into fatal optimism —
  plus `EXTRACT_CHECKPOINT_BUFFER_MS` (90 s) for the upload. When short, it
  packs everything accumulated (embeddings bit-exact, per-frame metadata, the
  persisted model-input JPEGs, the cross-batch seam-holdover frames) into ONE
  uncompressed tar at `{dest_prefix}/embeddings/{window}_checkpoint.tar` and
  raises `LambdaResourceTimeoutException` — Step Functions matches the CLASS
  NAME, so renaming it disables the retry.
- **ASL:** the exception retries 10× at 2 s (each retry makes real progress);
  `Lambda.Unknown` is 2 attempts because with a checkpoint on S3 even a hard
  kill mid-batch loses only the interrupted batch. Both definition copies
  carry the retry policy.
- **Resume** restores the tar, splices the accumulators (REPLACE, not
  append), re-roots holdover frame paths into the new invocation's /tmp, and
  continues at the recorded batch position. The checkpoint is deleted after
  the window's real outputs are written (the Lambda role carries
  s3:DeleteObject for exactly this; a failed delete logs a warning rather
  than being swallowed).
- **Tests:** [test/extract_checkpoint_test.py](test/extract_checkpoint_test.py)
  pins the lossless pack/unpack round trip, the gate arithmetic, and that a
  corrupt checkpoint is a hard error (a silent fresh start would re-embed and
  DUPLICATE every frame processed so far).
  [test/extract_resume_splice_test.py](test/extract_resume_splice_test.py)
  runs the real `_process_window` (real ffmpeg on a synthesised colour-block
  video, deterministic fake DINOv2 session, S3 faked to a local dir)
  interrupted TWICE and asserts the three-invocation output is bit-identical
  to an uninterrupted run — and that each hop advanced by exactly one batch.
  The gate reads the clock exactly once per pass; the splice test's stub
  schedule would catch a second read regressing.
- **Bounds:** /tmp is 2 GB against a worst case of about half that; a SINGLE
  batch that cannot finish inside a fresh budget still fails after the retry
  caps — the residual class, to be met with adaptive mini-batch sizing if it
  ever occurs. Cost accounting was already ready: usage tracking bills failed
  attempts via the `failed_ts` fallback.

The process lesson this carries: a fix's own asset class must run end-to-end
before the fix is called done — the first live run surfaced an orphaned-
checkpoint IAM gap a unit replay structurally could not see.

### §9.14 — Full-ingest robustness sweep (planned)

Run the pipeline over a far more diverse uncurated ingest set (commercials,
sports, conference talks, dual-purpose burn-in masters, MXF deliveries,
foreign-language content, short clips) to measure solidity beyond the curated
corpus: triage every non-SUCCEEDED terminal state into a failure taxonomy
(probe rejects for unseekable formats are EXPECTED failures, not bugs), verify
the published invariants corpus-wide (trimmed schema, diameter ≤ threshold,
`gates.groups == total_groups`, offline replay reproducing live memberships,
contiguity-audit warnings absent, checkpoint tars retired, programme_tail
sane), and report per-genre recurrence/burst/tail statistics — which doubles
as the measurement base that grows the n behind every fitted threshold
(notably §9.15's burst constants).

### §9.15 — Burst/montage evidence

Bursty content (jingle reels, montages) correctly produces many sub-second
shots, and the cost lands DOWNSTREAM: shot-metadata-enrichment infers per
shot, so a jingle reel burns tokens on near-useless per-card metadata.
Measured with data already published (shot durations only): an asset that IS a
jingle reel reads an order of magnitude more of its shots inside bursts than
ordinary content containing montage moments — a usable gap. Implemented as
`burst_runs` in the shots.json envelope: contiguous intervals
`{start_shot_id, end_shot_id}` of consecutive sub-second shots
(`BURST_MAX_SHOT_SEC` 1.0 strict `<` — a shot at exactly 1.0 s breaks the run;
`BURST_MIN_RUN` 6; shot type ignored so black micro-separators ride along,
because a content-only rule breaks every run at each card separator). The
`frame_*` dispersion fields characterise a burst's NATURE: jingle burst shots
are internally frozen (rapid cuts between static cards) where action-montage
shots carry high intra-shot motion. Decisions: consumers fold each run into
one inference (that is where the cost lives); the producer publishes the runs
so every consumer shares one definition — evidence, not a "jingle" verdict;
and the producer never merges bursts — the boundaries are real cuts, and
recurrence/face-tracking legitimately want them. The run thresholds are fitted
on a small sample; §9.14 grows the n before any retune.
