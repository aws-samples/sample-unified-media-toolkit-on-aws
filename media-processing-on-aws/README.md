# media-processing-on-aws

AWS Lambda container image for media pre-processing, driven by one Step
Functions state machine and exposed to agents as an AgentCore Gateway
(MCP) target. It does three jobs:

- **`process_media`** — probe a source file, chunk-transcode it in
  parallel, and merge the chunks into one or more output groups (mp4
  proxy, wav/m4a audio, jpeg/png frame sets, technical `metadata`, a
  ducked audio-description `ad_mix`). A metadata-only request skips the
  transcode entirely.
- **`quick_edit`** — compose ONE output video from an inline JSON cut
  list over one or more sources: sequential segments, cut/fade/crossfade
  transitions, black generator segments, normalized crops, audio
  overlays, two-pass loudness normalization.
- **`capture_frames`** — one still per millisecond marker across one or
  more sources, snapped to each source's own frame grid.

All three run on the same state machine and the same container image;
quick_edit and capture_frames reuse the chunk pipeline's frame-accuracy
machinery rather than adding states.

## Container tools

Built from source in the Dockerfile (two-stage, `linux/arm64`):

| Tool | Version | Purpose |
|------|---------|---------|
| ffmpeg | n6.1.2 | Transcode, probe, audio extraction |
| ffprobe | n6.1.2 | Stream/format metadata |
| mxf2raw (BMX) | latest `main` | MXF-accurate duration, timecode, essence type |
| openH264 | 2.6.0 | H.264 encode (LGPL codec) |
| YASM | 1.3.0 | Assembler required by ffmpeg optimised routines |

LGPL source archives (`source_ffmpeg.zip`, `source_openh264.zip`) are
included in the image at `/opt/sources/` for distribution compliance.
All binaries land on `PATH` via `/opt/bin`; shared libs at `/opt/lib`
on `LD_LIBRARY_PATH`.

## Workflow

```
"Probe media file"   [3-cores / 5376 MB]
    ↓
"Metadata only?"     [Choice]  ──data.metadata_only == true──┐
    ↓ default                                                │
"Map transcode segment chunk iterators"  [MaxConcurrency=40] │
    ↓  (Sandbox.Timedout | Lambda.Unknown → retry on 6-cores)│
"Merge segment chunks"  [$.data.function_arn]                │
    ↓                                                        │
"Usage tracking"     [half-core / 896 MB]  ←─────────────────┘
```

State names above are the ASL keys in [definitions.json](definitions.json) — they are
what `get_job_status` reports as `current_state`, and they differ from the `operation`
values in the payload.

A metadata-only request is fully served by the probe state, so the Choice skips the
Map and merge. Usage tracking still runs — it is the `End: true` state whose payload
the gateway reads.

Entry point: [lambda_function.py](lambda_function.py) dispatches on the `operation` field.

| ASL state | `operation` | File | Lambda |
|-----------|-------------|------|--------|
| `Probe media file` | `StateProbeMediaFile` | [states/probe_media_file/lambda_function.py](states/probe_media_file/lambda_function.py) | `Lambda3Cores` (5376 MB) |
| `Transcode segment chunk (X mins)` | `StateChunkTranscode` | [states/chunk_transcode/lambda_function.py](states/chunk_transcode/lambda_function.py) | `$.item_data.function_arn` — 3-cores default, 5-cores for large inputs |
| `Transcode segment chunk (Retry larger memory)` | `StateChunkTranscode` | (same file) | `Lambda6Cores` (10240 MB), hardcoded |
| `Merge segment chunks` | `StateMergeChunks` | [states/merge_chunks/lambda_function.py](states/merge_chunks/lambda_function.py) | `$.data.function_arn` — inherits probe's stamped ARN (3- or 5-cores) |
| `Usage tracking` | `StateUsageTracking` | [states/usage_tracking/lambda_function.py](states/usage_tracking/lambda_function.py) | `LambdaHalfCore` (896 MB) |

Three states name a Lambda literally in the ASL: probe → 3-cores, the timeout retry →
6-cores, usage tracking → half-core. Chunk transcode and merge instead resolve theirs
from the payload at runtime, so their memory tier is a function of the input, not of the
definition.

Both iterator states share the one `operation` value: the Map's `ItemSelector` sets
`operation: StateChunkTranscode` and `item_data.$: $$.Map.Item.Value`, and each state
forwards the whole payload (`Payload.$: "$"`). The retry state is a different Lambda
size, not different code.

## Input schema

```json
{
  "operation": "StateProbeMediaFile",
  "uuid": "<UUID>",
  "input": {
    "s3uri": "s3://ingest-bucket/path/to/source.mxf",
    "input_config": {
      "use_channel_mapping": false,
      "audio_channel_mapping": [1, 2],
      "use_partial": false,
      "mark_in_out": [0, 1800],
      "crop_x": 0,
      "crop_y": 0,
      "width": 960,
      "height": 540
    }
  },
  "output": {
    "s3uri": "s3://proxy-bucket/output/prefix",
    "output_groups": [
      {"group_name": "proxy",         "format": "mp4"},
      {"group_name": "audio_only",    "format": "wav"},
      {"group_name": "audio_m4a",     "format": "m4a"},
      {"group_name": "frame_capture", "format": "jpeg"},
      {"group_name": "metadata",      "format": "metadata", "detail": "default"}
    ]
  },
  "data": {}
}
```

`output.output_groups` is optional — defaults to a single `mp4` proxy group if omitted.
All `input_config` fields are optional. `crop_x`/`crop_y`/`width`/`height` are shared
across video and frame groups, not per-group, and are ignored by the `metadata` group.

## State: StateProbeMediaFile

**File:** [states/probe_media_file/lambda_function.py](states/probe_media_file/lambda_function.py)
**Helpers:** [states/probe_media_file/probe_helper.py](states/probe_media_file/probe_helper.py), [lib/input_schema.py](lib/input_schema.py)

Header-only probe — no full decode. Rejects archive-tier sources (`archive_tier`,
reusing the file-size `head_object`), resolves duration, detects `is_all_intra`,
selects audio streams, validates output groups, and builds chunk iterators.

### Output groups

| Format | Source requirement | Output |
|--------|-------------------|--------|
| `mp4`  | video stream required | `.ts` video chunks + `.flac` audio chunks |
| `wav`  | audio stream required | `.flac` audio chunks → WAV at merge |
| `m4a`  | audio stream required | `.flac` audio chunks → M4A at merge |
| `jpeg` | video stream required | JPEG frames uploaded by ChunkTranscode; manifest written at merge |
| `png`  | video stream required | PNG frames uploaded by ChunkTranscode; manifest written at merge |
| `metadata` | none | `metadata.json` written **by this state**; no chunks |
| `ad_mix` | audio stream required | `.flac` audio chunks → ducked AD mix WAV at merge (requires `render_manifest_s3uri`) |

Audio-only source (no video, audio groups only): time-based 30-min segmentation.
Multiple audio groups (wav + m4a) share the same `.flac` intermediate chunks.

Only the **first** `mp4` group and **first** frame (`jpeg`/`png`) group are used —
subsequent ones of the same category are warned and dropped. Multiple audio groups
(wav + m4a) are all processed.

### The `metadata` output group

Resolved technical metadata, produced entirely inside `StateProbeMediaFile` — ffprobe
has already run there, so the group costs no extra decode. Written to
`{prefix}/{group_name}/{name}.json` (default `metadata.json`), and `data.output_keys`
is stamped at probe time so it is uniform with every other group. Design: §7.

Two `detail` levels, both served by the **same single ffprobe call**:

| `detail` | Adds |
|----------|------|
| `default` | `format` + `streams` + `chapters` + `programs` |
| `full`    | additionally per-stream `disposition` and `tags` |

`full` is a filtering decision, not a second probe — `disposition` and `tags` are
already present in `-show_streams` output. The gate is per-**stream** only:
`format.tags`, `chapters[].tags` (where a chapter's title lives) and `programs[].tags`
(`service_name`) are the substance of those sections and are present at both levels.
`-show_programs` nests a full copy of each program's streams under
`programs[].streams[]`, so `_filter_probe_info` strips both lists — filtering only the
top-level one leaked `disposition`/`tags` at `default` on every MPEG-TS source (the
only container that reports programs). Frame- and packet-level detail is **not offered
at any level** — §7.3 has the measurements and the reason.

Payload (top-level fields are the contract; `probe_info` is the verbatim ffprobe dict):

```json
{
  "format_name": "mxf", "duration_sec": 6.04, "duration_source": "ffprobe_ts",
  "file_size": 89412378, "has_video": true, "has_audio": true, "is_all_intra": false,
  "video": {"codec": "h264", "width": 1920, "height": 1080, "fps": "25/1",
            "interlaced": true, "sample_aspect_ratio": "1:1", "field_order": "tt"},
  "audio_streams": [{"mapid": "0:a:0", "codec": "pcm_s24le", "channels": 1, "sample_rate": 48000}],
  "mxf": {"fps": 25.0, "drop_frame": false, "essence_type": "AVCI_100_1080i",
          "frame_count": 151, "duration_sec": 6.04,
          "start_timecode": "09:59:30:00", "mxf_info": {}},
  "probe_info": {}
}
```

`duration_sec` + `duration_source` and `is_all_intra` are the reason this group exists
— none is a passthrough of an ffprobe field.

**`mxf2raw` runs automatically, never on request.** `is_mxf_format(format_name)` on the
ffprobe result is the gate. MXF → the `mxf` block is present; non-MXF → the key is
**absent entirely** (no `is_mxf: false`, no zeroed fields). MXF where `mxf2raw` fails →
**raises**; a caller handed `frame_count: -1` has been handed a plausible-looking wrong
number. The invariant: the presence of `mxf` means MXF, and its contents are always
real measurements (§8).

`frame_count` and `mxf.duration_sec` are **omitted** when the MXF header records no
duration (`count="-1"` — observed on Sony eVTR D10 material). That is a valid MXF
shape, not a failure, so the fields are absent rather than sentinel-valued; `fps <= 0`
is the failure sentinel and raises (§8.3.1).

A metadata-**only** request (every group is a `metadata` group) sets
`data.metadata_only`, produces zero iterators, and the ASL Choice routes it straight to
usage tracking — roughly two Lambda invocations total. A mixed request (`metadata` +
`mp4`) is handled the same way at probe time; the merge state never learns about
metadata groups, it only inherits the already-populated `data.output_keys`.

### `is_all_intra` detection

| Result | Condition |
|--------|-----------|
| `True` | codec in `{prores, dnxhd, dnxhr, mjpeg, dvvideo, ffv1, huffyuv, v210, r10k, cineform}`, D10/IMX codec tag (`mx5p` etc.), or mxf2raw `essence_type` contains `d10`/`imx`/`intra` |
| `False` | codec in `{h264, hevc, vp9, av1, mpeg4, …}`, or mxf2raw `essence_type` contains `long_gop` |
| `None` | `mpeg2video` without D10 tag, or unrecognised codec → `scan_is_all_intra` (first 20s frame scan) |

### Duration priority

`resolve_duration` ([lib/ffprobe.py](lib/ffprobe.py)) has exactly two rungs, for MXF
and non-MXF alike:

1. `ffprobe duration_ts × time_base` → `duration_source: "ffprobe_ts"`
2. `ffprobe fmt.duration` → `duration_source: "ffprobe_fmt"` (logged as a last resort)

Neither yields a positive duration → raises.

There is **no `mxf2raw frame_count / fps` rung** in `resolve_duration`, despite it
being the intuitive first choice for MXF. `data.duration_sec` — which drives segment
boundaries — is always ffprobe-derived. The MXF-accurate number is available separately
as `mxf.duration_sec` in the `metadata` group payload, so the two can be compared
without changing transcode behaviour. Wiring the rung in would shift `duration_sec`
for every MXF asset and needs its own verification (§8.4).

### ARN routing

Probe stamps `function_arn` on each iterator. Files ≥ 20 GB or duration > 2 hours use
`CHUNK_TRANSCODE_FUNCTION_ARN_LARGE` (5-cores); smaller files use
`CHUNK_TRANSCODE_FUNCTION_ARN` (3-cores).

`CHUNK_TRANSCODE_FUNCTION_ARN` is **not** set on the 3-cores Lambda that actually runs
probe — only `..._LARGE` is. The default falls back to `context.invoked_function_arn`,
i.e. probe's own 3-cores function, which happens to be the intended default tier. That
is why it works, not a coincidence worth relying on: adding the env var to 3-cores or
moving probe to another tier are both fine, but changing probe's tier *alone* would
silently change the default chunk-transcode tier with it.

### Output (added to `data`)

```
data.destination    — {bucket, prefix} resolved from output.s3uri
data.output_groups  — reconciled list with all defaults filled in
data.probe_s3_key   — S3 key of probe.json
data.duration_sec   — resolved duration
data.has_video      — bool
data.has_audio      — bool
data.is_all_intra   — bool
data.metadata_only  — bool; every requested group is a `metadata` group. Read by the
                      "Metadata only?" Choice state to skip Map + merge
data.output_keys    — {group_name: s3_key} for groups already written here (metadata
                      groups only; {} otherwise). StateMergeChunks seeds its own
                      output_keys from this, so a mixed request keeps both
data.function_arn   — ARN stamped on iterators (large or default; used by StateMergeChunks via data passthrough)
data.iterators      — list of iterator dicts (consumed by Map state); `[]` when metadata_only
```

`metadata_only` is computed from the group list, never inferred from `iterators == []`
— an empty iterator list is also what a bug looks like, and that must still fail loudly
(§7.2).

## State: StateChunkTranscode

**File:** [states/chunk_transcode/lambda_function.py](states/chunk_transcode/lambda_function.py)
**Logic:** [states/chunk_transcode/transcode_helper.py](states/chunk_transcode/transcode_helper.py)

Invoked per-iterator by the Step Functions Map state. Each invocation:
1. Resolves seek position — keyframe scan (`find_prior_keyframe`) for inter-frame codecs; direct seek for all-intra; no seek for first chunk.
2. Substitutes `{start_frame}`, `{end_frame}`, `{start_sample}`, `{end_sample}` tokens into `filter_complex`.
3. Runs single ffmpeg invocation producing all outputs (TS + FLAC + JPEGs/PNGs) for the chunk.
4. Validates: `validate_frames` (±1) for `.ts`; `validate_duration` (±10 ms) for `.flac`; `validate_jpeg_count` (±1) for frame images.
5. Uploads results and returns `{item_id, outputs}`.

Audio-only chunks (no vfilters) seek to `max(0, nominal_start_sec - 3.0)` as a
pre-roll, then use sample-accurate `atrim` to clip the desired range.

Return shape:
```json
{
  "item_id": 0,
  "outputs": [
    {"s3_prefix": "segment_video", "s3_key": "...video_0.ts"},
    {"s3_prefix": "segment_audio", "s3_key": "...audio_0.flac"},
    {"s3_prefix": "<frame_group_name>", "frames": {"start": 0, "step": 30, "total": 600}}
  ]
}
```

`frames.total` is the actual count produced — the last chunk may produce fewer than
expected.

`ValidationException` is the exact error class name matched by Step Functions Retry
(2 attempts, `BackoffRate` 1.0). The Catch is on `Sandbox.Timedout` **and**
`Lambda.Unknown`, and routes to the 6-cores retry state. Those two errors also carry an
explicit `Retry` with `MaxAttempts: 0`, which is what stops the default retry from
firing before the Catch can hand the chunk to a larger Lambda.

The quick-edit renderers ([states/chunk_transcode/edit_render.py](states/chunk_transcode/edit_render.py))
also live in this state: a cut-list plan entry (segment body, junction clip, or capture
batch) is just another Map iteration (§14).

## State: StateMergeChunks

**File:** [states/merge_chunks/lambda_function.py](states/merge_chunks/lambda_function.py)
**Helpers:** [states/merge_chunks/merge_helper.py](states/merge_chunks/merge_helper.py)

Sorts ChunkTranscode outputs by `item_id`, collects keys by `s3_prefix`, then processes
each output group:

| Format | Steps | Output key |
|--------|-------|-----------|
| `mp4`  | single ffmpeg: concat `.ts` (codec copy) + concat `.flac` (AAC 96 kbps stereo) → MP4; video-only MP4 if no audio chunks | `{prefix}/{group_name}/{name}.mp4` |
| `wav`  | ffmpeg: concat `.flac` → PCM WAV | `{prefix}/{group_name}/{name}.wav` |
| `m4a`  | ffmpeg: concat `.flac` → AAC M4A | `{prefix}/{group_name}/{name}.m4a` |
| `jpeg`/`png` | reconstruct key list from `{start, step, total}` descriptors → write manifest JSON | `{prefix}/{group_name}/frame_manifest.json` |
| `metadata` | none — already written by `StateProbeMediaFile`; skipped here | (inherited from `data.output_keys`) |
| `ad_mix` | concat `.flac` bed + fetch AD render manifest → render each narration take (`af_chain` + `atempo`) → adelay/amix bus (batched past 150 inputs) → sidechain-duck the bed under it → one whole-programme `loudnorm`+`alimiter` pass → stereo WAV ([lib/ad_mix.py](lib/ad_mix.py); built for the audio-description workflow, which renders no audio itself and hands over `render_manifest.json`) | `{prefix}/{group_name}/{name}.wav` |

Manifest JSON structure:
```json
{
  "total": 1200,
  "prefix": "{out_prefix}/{group_name}",
  "keys": ["frame_0000000.jpg", ...],
  "dimension": [960, 540]
}
```

All `.ts` and `.flac` intermediates are deleted from S3 after all groups succeed. Set
`KEEP_SEGMENT_CHUNKS` env var to skip deletion (useful for debugging).

`/tmp` peak: final MP4 only (~1.8 GB for a 2-hour file at ~2 Mbps) — within the 4 GB
`EphemeralStorage` carried by 3-, 4-, 5-, and 6-cores. Half-core, 1-core, and 2-cores
have the 512 MB default, so merge must never be routed to one of them.

### Output (returned in `data`)

```
data.output_keys   — {group_name: s3_key} for each completed output group. Seeded from
                     the probe state's output_keys, so metadata groups survive a mixed
                     request rather than being overwritten
data.iterators     — removed from data
```

## State: StateUsageTracking

**File:** [states/usage_tracking/lambda_function.py](states/usage_tracking/lambda_function.py)

Walks the Step Functions execution history, computes per-state Lambda invocation
counts, durations, and estimated cost (a pinned us-east-1 pricing table; per-invocation
ARN for accurate cost on mixed-size chunk transcode retries), then writes
`usage_tracking.json` to S3. Pipeline wall time is measured from `ExecutionStarted`
timestamp. Billing rules for failed and retried attempts: §15.

## Library modules

| Module | Purpose |
|--------|---------|
| [lib/s3_utils.py](lib/s3_utils.py) | Shared boto3 S3 client; `presign`, `put_json`, `head_object`, `archive_tier` (cold-storage guard), `upload_file` (multipart for files ≥ 16 MB; 64 MB parts; 8 concurrent; 60 pool connections), `copy_object`, `delete_object` |
| [lib/proc_utils.py](lib/proc_utils.py) | `run(cmd, timeout)` — shared subprocess runner for ffmpeg/ffprobe |
| [lib/ffprobe.py](lib/ffprobe.py) | `run_ffprobe`, `is_mxf_format` (canonical — the MXF predicate lives here, not in `mxf2raw.py`), `scan_is_all_intra`, `find_prior_keyframe`, `fps_from_str`, `resolve_duration` |
| [lib/input_schema.py](lib/input_schema.py) | `parse_s3uri`, `parse_output_groups`, `compute_frame_interval`; the format registry (`VALID_FORMATS`, `VIDEO_FORMATS`, `AUDIO_FORMATS`, `FRAME_FORMATS`, `METADATA_FORMATS`, `METADATA_DETAIL_LEVELS`) |
| [lib/mxf2raw.py](lib/mxf2raw.py) | `run_mxf2raw`, `parse_mxf2raw_info`, `build_mxf_block` — used by the `metadata` output group, where absent-or-broken mxf2raw on an MXF source **raises**. `is_all_intra` still consumes it best-effort |
| [lib/audio_command.py](lib/audio_command.py) | `select_audio_streams` — Dolby-E filter, channel mapping cap at 2, fallback to auto-select |
| [lib/video_command.py](lib/video_command.py) | `determine_is_all_intra`, `scale_dimension`, `sec_to_frame` (**the** timestamp→frame conversion, shared by segment boundaries and keyframe preroll), `build_segment_boundaries`, `compute_bitrate` |
| [lib/cut_list.py](lib/cut_list.py) | Quick-edit cut-list validation, exact-Fraction frame mapping, record-timeline derivation, Map plan builder — pure, no AWS (§14). Byte-identical copy ships in the gateway zip |
| [lib/frame_capture.py](lib/frame_capture.py) | capture_frames marker validation and batching — pure, no AWS (§14.4). Byte-identical copy ships in the gateway zip |
| [lib/edit_filters.py](lib/edit_filters.py) | Quick-edit ffmpeg filter builders (the §14.2 normalization matrix); `is_hdr` refusal predicate |
| [lib/ad_mix.py](lib/ad_mix.py) | The ad_mix render: per-take fitting, ducking bus, whole-mix mastering |
| [lib/wav_utils.py](lib/wav_utils.py) | Pure-Python WAV header parser (used internally; WAV output now produced by ffmpeg in merge) |

## AgentCore Gateway tool

[gateway_tool/lambda_function.py](gateway_tool/lambda_function.py) is the AgentCore
Gateway dispatcher that wraps this workflow as MCP tools. It is deployed as a separate
Lambda (not a container — flat zip, no Docker).

### `process_media`

Validates parameters, constructs the nested SFN input, derives the workflow_id, and
calls `StartExecution`.

- `output_groups` maps directly to `output.output_groups` in the SFN input; omit for
  the default single mp4 proxy. Valid formats: `mp4`, `wav`, `m4a`, `jpeg`, `png`,
  `metadata`, `ad_mix` (requires `render_manifest_s3uri` — the audio-description
  workflow's render manifest; rejected on any other format).
- `metadata` groups accept `detail` (`default` | `full`, case-insensitively normalized;
  anything else → `INVALID_PARAMETER`). A metadata-only request is the cheap path — two
  Lambda invocations, no Map, no merge — so an agent that only needs technical facts
  should ask for `metadata` rather than transcoding a throwaway proxy.
  `width`/`height`/`crop_x`/`crop_y` are accepted but ignored for it.
- All `input_config` fields (`use_partial`, `mark_in_out`, `crop_x`/`crop_y`/`width`/`height`,
  `audio_channel_mapping`) are exposed as flat top-level parameters.
- workflow_id strategy: `workflow_id = uuid5(canonical JSON of the SFN input WITHOUT
  output.s3uri + "|" + TOOL_VERSION + "|" + idempotency_token when one is supplied)` —
  derived by the shared [gateway_tool/workflow_identity.py](gateway_tool/workflow_identity.py),
  which documents the three ids. It NAMES the execution, so identical requests are
  idempotent: on `ExecutionAlreadyExists` the call is **absorbed** into the existing
  run — no second execution — while any parameter, token, or tool-version change
  re-runs to a fresh prefix. Outputs land at
  `s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{workflow_id}`; a caller-supplied
  `output_s3_prefix` is deprecated and ignored.
- `idempotency_token` (optional, 1-80 chars of `A-Z a-z 0-9 _ . -`) is the caller's
  assertion "this is the same request I sent before". It rides in the hash preimage as
  extra material and is then **discarded** — never forwarded to the state machine,
  never stored, never echoed. Omitting it still dedups on the parameters alone, since
  they name the asset and therefore determine the output; a token only distinguishes
  two otherwise-identical requests. The retired `uuid` parameter is **rejected** on
  submit (it used to mean both "which asset" and "same request", so a caller still
  sending it cannot be assumed to mean the surviving half) and **accepted with a log
  warning** on `get_job_status` / `describe_tool`, where neither key could have changed
  the answer.
- Returns a **HANDLE, never results**: `{"job_id": <execution_arn>, "status":
  "RUNNING", "current_state": <active stage>, "workflow_id": <derived id>,
  "output_s3_prefix": <derived prefix>, "asset_s3uri": <source s3uri>}`. An absorbed
  resubmit returns *the same fields* — only `status` differs (the existing run's, which
  may be terminal), plus `message` (present only when absorbed, and the only thing
  distinguishing the two cases — the `job_id` is a hash of the same input either way)
  and `error` on a terminal failure. `current_state` is dropped once the run has
  stopped, because a stopped execution is in no state and reading one costs a
  `GetExecutionHistory` paging pass for a `"unknown"`. Results come from
  `get_job_status` alone; that is what makes the declared `OutputSchema` equal to the
  reachable field set. No `uuid`, deliberately — the SFN input still carries a `uuid`
  key (set to the workflow_id) because every state derives its output prefix from it;
  that is this execution's local id, not the caller's token.

### `get_job_status`

Calls `DescribeExecution`; when RUNNING, pages `GetExecutionHistory` to find the active
pipeline stage. The `"Metadata only?"` Choice state is invisible to this scan — it
emits `ChoiceStateEntered`, not `TaskStateEntered`, so a metadata-only run reports
`StateProbeMediaFile` then `StateUsageTracking` with nothing in between. When
SUCCEEDED, returns `workflow_id`, `output_keys` (nested object `{group_name: s3_key}`),
`asset_s3uri`, `usage_key`, `total_estimated_cost`, and a self-describing `outputs`
list of `{uri, mime, label, desc}` entries (labels: each `group_name`, plus `probe` and
`usage`; mime by extension). `desc` is the one-line role description joined verbatim
from [gateway_tool/output_contract.json](gateway_tool/output_contract.json) — the same
file `describe_tool` serves in full, so the two surfaces cannot disagree; it is
omitted, never fabricated, when the contract lacks the entry. Uniquely on this stack
the join key is the group's **format kind** (`mp4`, `wav`, `m4a`, `jpeg`, `png`,
`metadata`), read from the reconciled `data.output_groups` — not the label (a
caller-chosen group name the contract cannot enumerate) and not the key's file
extension (a jpeg group's output object is `frame_manifest.json`, so the extension
would join the wrong entry); the fixed `probe`/`usage` labels join directly.
`output_keys` is kept nested because group names are runtime-dynamic and cannot be
enumerated in the OutputSchema. **Every branch** — RUNNING, SUCCEEDED, every failure,
and the fallback for a status Step Functions has not shipped yet — carries `job_id`,
`workflow_id`, `asset_s3uri`, and `output_s3_prefix`, so a status reply always names
its own run and its output location even when there is nothing to report yet.
(`output_s3_prefix` is read back from the execution's *input*, not its output, which is
why a RUNNING or failed run still has one — partial results already live under it.)
Terminal failures add `error`.

### `quick_edit`

Compose ONE output video from an inline JSON cut list over 1+ sources. Validation is
the same `cut_list` module probe runs (flat byte-identical copy in the zip,
parity-pinned), so a bad request refuses at the gateway with the named condition.
Segments are sequential (order = output timeline; Σ(out−in) − Σcrossfades is the
output duration); transitions ride the incoming segment (`transition_in`, types
cut/fade/crossfade); `"black"` is a generator source (EDL BL); `channel: av|v|a`;
`crop` is NORMALIZED 0–1 (face boxes are already in that space); `audio_overlays`
replace/mix with record- or boundary-anchored (`audio_bed`) placement; `loudnorm`
overrides the multi-source default. HDR sources refuse at probe. Returns the standard
handle (+`segment_count`/`source_count`); outputs: the stitched mp4 (group `edit`) +
`edit_manifest` (requested vs actual per cut, record timeline, crops in ratios AND
pixels, measured loudness, source ETags). Full design record: §14.

### `capture_frames`

One still per ms-marker across 1+ sources, snapped to each SOURCE's frame grid (no
target fps exists for stills); optional per-marker `crop`, optional `resize` canvas
(fit+pad, never stretch; native otherwise). Caps 500 markers / 25 sources; markers
batch ≤100 per Map iteration grouped by source. Outputs:
`frames/frame_{index:04d}.{ext}` + `capture_manifest` (index → key, requested vs
actual ms, frame). Design record: §14.4.

Both tools take `callback_task_token` with the exact process_media contract (§11):
validated loudly by the shared `_read_callback_token`, registered before
StartExecution, completed synchronously on an absorbed-into-finished run, and covered
by the same raise-vs-return choke point — refusals RAISE when a token is present.

### `describe_tool`

Returns `{"version": <TOOL_VERSION>, "state_machine_arn": <ARN>, "description": <short
backend description>, "output_contract": <parsed output_contract.json>}`. Version is
stamped at deploy time via the `TOOL_VERSION` env var (set from the `.version` file by
`build-deploy-s3-dist.sh`); the ARN comes from the state machine env var, giving
callers a console/CLI entry point without parsing a job_id.

`output_contract` (replicating the family's shot-segmentation pilot) is the field-level
contract for the workflow's output files: basename pattern, kind, mime, the one-line
`description` that doubles as the outputs[] entries' `desc`, and a compact
single-element `example` of the file's contents (`contents` block for the binary media
files instead). It is static per `TOOL_VERSION` — the file ships in the gateway zip
(`build-deploy-s3-dist.sh` pins the member), so version and contract cannot drift —
which is why it lives here and not in `get_job_status`: a caller caches it once per
session, and CI can pin it without running a job. This is the one stack where contract
keys are **not** the emitted labels: labels are runtime group names, so the keys are
the format kinds (`mp4`, `wav`, `m4a`, `jpeg`, `png`, `metadata`, `ad_mix`) plus the
fixed `probe`/`usage`, and a top-level `note` in the file says so in-band. Anything not
listed (`segment_video/` `.ts` and `segment_audio/` `.flac` chunk intermediates,
anything else under the prefix) is internal and may change without notice. The examples
were authored from the producing code, not the docs — note `probe.json` is an envelope
(`uuid`, `source_*`, resolved duration, `iterators`) around the verbatim ffprobe dict
in `probe_info`, and this stack's `usage_tracking.json` carries
`source_key`/`retried`/`incomplete` rather than the sibling stacks' fields.

`describe_tool` also advertises `features: ["callback_task_token"]` (§11).

### Tool routing

AgentCore passes `context.client_context.custom["bedrockAgentCoreToolName"]` as
`"{target_name}___{tool_name}"`. `_resolve_tool()` strips the target prefix and returns
the bare tool name.

### Agent system prompt

The companion system prompt for a Strands agent using this target lives at
[../media-toolkit-gateway/prompts/media_processing_agent.md](../media-toolkit-gateway/prompts/media_processing_agent.md).

## CloudFormation stack

[cfnstack.yaml](cfnstack.yaml) is a self-contained standalone stack. The state machine
definition is **inlined in the template** as a `DefinitionString` + `!Sub`, and that is
deliberate: `!Sub` resolves every `${MediaProcessingLambda3Cores.Arn}` against the
template, so a typo or a deleted Lambda is a template-validation error. The alternative
— uploading definitions.json and referencing it via `DefinitionS3Location` +
`DefinitionSubstitutions` — was tried and reverted, because substitutions have no
equivalent check: an unmatched token is not an error, it is passed through literally
and only surfaces when the state machine runs.

The consequence of staying inline:

- **The definition has two copies: [definitions.json](definitions.json) and the inline
  block in cfnstack.yaml.** definitions.json is the readable one and is what the docs
  above cite; cfnstack.yaml's copy is what actually deploys. Keeping the definition
  inline is the tradeoff that buys `!Sub` validation, and the cost is that **an ASL
  change has to be made in both files**. Nothing enforces the match — they drifted once
  already (the inline copy was missing `"Metadata only?"` while definitions.json had
  it, so the Choice state was documented but never deployed). After editing either,
  diff them.

`$$.Execution.Id` and `$$.Map.Item.Value` are written with two dollar signs in both
copies — that is the Step Functions context-object spelling, and `!Sub` passes them
through untouched because it only expands `${...}`. The two copies stay
byte-comparable.

Token substitutions applied by the build script:

| Token | Value |
|-------|-------|
| `%%VERSION%%` | `.version` file |
| `%%PKG_MEDIA_PROCESSING%%` | `media-processing-on-aws-{VERSION}.zip` |
| `%%PKG_GATEWAY_TOOL%%` | `media-processing-gateway-tool-{VERSION}.zip` |
| `%%BUCKET_NAME%%` | deployment bucket basename |
| `%%KEY_PREFIX%%` | `{solution}/{version}/media-processing` |

### Lambda variants

All seven share one container image and a 900 s timeout; memory comes from the
`Lambda` mapping and is the only thing that differs (plus `/tmp`).

| Logical ID | Memory | `/tmp` | Used for |
|------------|--------|--------|----------|
| `MediaProcessingLambdaHalfCore` | 896 MB   | 512 MB | **Usage tracking** |
| `MediaProcessingLambda1Core`    | 1792 MB  | 512 MB | Available |
| `MediaProcessingLambda2Cores`   | 3584 MB  | 512 MB | Available |
| `MediaProcessingLambda3Cores`   | 5376 MB  | 4 GB   | **Probe**; chunk transcode (default tier); merge chunks |
| `MediaProcessingLambda4Cores`   | 7168 MB  | 4 GB   | Available |
| `MediaProcessingLambda5Cores`   | 8960 MB  | 4 GB   | Chunk transcode + merge for large inputs (≥ 20 GB or > 2 hr) |
| `MediaProcessingLambda6Cores`   | 10240 MB | 4 GB   | **Chunk transcode timeout retry** |
| `MediaProcessingGatewayToolLambda` | 256 MB | 512 MB | AgentCore Gateway dispatcher (Python 3.13, arm64, no container, 30 s timeout) |

Bold entries are named literally in [definitions.json](definitions.json). The rest are
reached only through a payload-resolved `function_arn`, so "used for" describes runtime
routing rather than anything the ASL states.

The two chunk-transcode ARN env vars are distributed unevenly, and neither list matches
"the tier that runs probe":

| Env var | Set on |
|---------|--------|
| `CHUNK_TRANSCODE_FUNCTION_ARN` | half-core, 1-core, 2-cores |
| `CHUNK_TRANSCODE_FUNCTION_ARN_LARGE` | half-core, 1-core, 2-cores, **3-cores** |

Probe runs on 3-cores, so it reads `..._LARGE` from the environment but resolves the
default tier from its own `invoked_function_arn`. See ARN routing above for why that
still lands on the intended 3-cores. 4-, 5-, and 6-cores get neither, which is correct
— nothing running there stamps iterators.

### Stack parameters

| Parameter | Purpose |
|-----------|---------|
| `ResourcePrefix` | Prefix applied to all resource names |
| `IngestBucket` | S3 bucket containing source media files (read-only) |
| `ProxyBucket` | S3 bucket for outputs and intermediates |

### Stack outputs

| Output | Purpose |
|--------|---------|
| `GatewayToolLambdaArn` | Imported by `media-toolkit-gateway/cfnstack.yaml` as `MediaProcessingToolLambdaArn` |

## Key constants

| Constant | Value | Location |
|----------|-------|---------|
| `VIDEO_SEGMENT_SECS` | 600 s (10 min) | lib/video_command.py |
| `VIDEO_MIN_SECS` | 60 s | lib/video_command.py |
| `AUDIO_SEGMENT_SECS` | 1800 s (30 min) | probe_media_file/lambda_function.py |
| `AUDIO_MIN_SECS` | 180 s (3 min) | probe_media_file/lambda_function.py |
| `LARGE_FILE_BYTES` | 20 GB | probe_media_file/lambda_function.py |
| `LARGE_DURATION_SECS` | 7200 s (2 hr) | probe_media_file/lambda_function.py |
| `FFPROBE_TIMEOUT` | 840 s | lib/ffprobe.py |
| `MXF2RAW_TIMEOUT` | 840 s | lib/mxf2raw.py |
| `PRESIGN_EXPIRY` | 1800 s | lib/s3_utils.py |
| `JPEG_UPLOAD_WORKERS` | 50 | chunk_transcode/transcode_helper.py |

## Tests

```
cd media-processing-on-aws && python3 -m unittest discover -s test -p 'test_*.py'
```

The `test_*_units.py` suites are offline (no AWS, no ffmpeg) and gate packaging —
`build-deploy-s3-dist.sh` runs them before producing artifacts.
`test/test_quick_edit_gateway_units.py` additionally pins the gateway copies of
`cut_list.py`/`frame_capture.py` byte-identical to the `lib/` originals.
`test/test_probe.py` and `test/test_chunk_transcode.py` are live CLI scripts needing
real S3. The usage-tracking billing invariants (§15) are pinned cross-stack by
`micro-workflow-bundle/test/test_units.py: TestUsageTrackingBillsFailedAttempts`.

## Build & deploy

`build-deploy-s3-dist.sh` packages the container build context, the workflow zip, and
the flat gateway zip, substitutes the template tokens, and uploads to the deployment
bucket. Packaging functions are explicit allowlists, and
`assert_internal_docs_excluded()` fails the build if internal-only files (local
gotchas, harness run manifests) ever land in an artifact. The container image is built
by CodeBuild from source.

## Gotchas

- **Usage tracking must bill failed attempts** — see §15 for the invariants
  (`_billed_sec` falls back to `failed_ts`; 0.0 only when an attempt has NO terminal
  event; per-`entered_id` wall-time summation). A `TaskFailed` Lambda killed at its
  timeout is the most expensive invocation possible and used to contribute 0 s.
- **`output.output_groups` defaults to mp4 proxy** — if `output_groups` is omitted or
  empty, `parse_output_groups` inserts a single default `mp4` group. Explicitly pass
  groups to get audio or frame outputs.
- **`audio_channel_mapping` is 1-based** — indices reference the 1-based position in
  the file's audio stream list. Any out-of-range index in the capped-to-2 list causes
  full fallback to auto-select.
- **`audio_channel_mapping` capped at 2** — only the first two indices are used.
- **`crop_x`/`crop_y` are shared** — apply to both the mp4 video chain and the
  jpeg/png frame chain. There is no per-group crop.
- **FLAC is the shared audio intermediate** — all audio output groups (wav, m4a, and
  the AAC track in mp4) are derived from the same `.flac` chunks. The FLAC intermediate
  is always forced to 2-channel stereo (`-ac 2`); the final per-group channel count
  (`audio_channels`) is applied at merge time.
- **The intermediates are not master-referenced — the planned analysis filters
  (loudness etc., §3–§6) must not measure them. Implementation parked.** The `.flac`
  chunks are channel-selected and stereo-downmixed (previous bullet), so
  loudness/silence measured on them reflects the downmix, not the program's channel
  layout — wrong for exactly the multichannel/multi-mono broadcast MXF this stack
  handles. The `.ts` proxy is scaled, deinterlaced, and re-encoded. A single ffmpeg
  pass over the master source doesn't fit a Lambda either: a >50 GB MXF is 9–12 min of
  S3 streaming alone (no staging — `/tmp` is 4 GB on every tier), before any decode.
  The design (§4) therefore taps the filters into the existing chunk-transcode
  invocations — source essence, pre-downmix/pre-scale, decode already paid — with
  per-filter seam reducers at merge. Do not revive the superseded "single pass at merge
  over concatenated FLAC" design; its fidelity premise ("FLAC is lossless, so
  measurements match the source") is false for any multichannel source (§4.0).
- **All-intra seek** — for ProRes, DNxHD, D10/IMX: direct `-ss nominal_start_sec`
  (exact, no preroll). For inter-frame (H.264 etc.): keyframe scan via
  `find_prior_keyframe`, with the keyframe's frame number derived from its
  **timestamp** via `sec_to_frame`.
- **Never derive a frame number from `pts / duration` — ffmpeg 6.1's MOV demuxer
  reports per-frame `duration` as 1 for any media timescale below 1000.**
  `transcode_helper` once computed `keyframe_frame = int(kf_pts / kf_dur)`. On a
  QuickTime file with the classic `1/600` timescale at 25 fps (24 ticks/frame) ffprobe
  6.1.6 prints `pts 5040000, duration 1`, so an 8400 s keyframe came out as frame
  5040000 instead of 210000 and `preroll_frames` went to −4830000, tripping the
  `ValidationException` guard on every non-first chunk. Verified against the deployed
  binary (`/opt/sources/source_ffmpeg.zip` → `FFmpeg-n6.1.2`), reproduced on synthetic
  MOVs, and swept: wrong at timescale 300/400/500/600/750/900, correct at 1000 and
  above. `-show_packets` shows the same `duration=1`, so it is the demuxer, not
  ffprobe's printing. It hid because only chunked (>600 s) **inter-frame** video with a
  sub-1000 timescale reaches the path — all-intra sources, audio-only chunks and
  single-chunk files never do. The fix is version-independent:
  `round(sec_to_frame(kf_ts, fps))` uses only `best_effort_timestamp_time`, which both
  6.1 and 9.x report correctly. **`round`, not `floor`** — a keyframe timestamp *is* a
  frame boundary, and ffprobe prints it to 6 dp, so frame 28753 at 24000/1001 arrives
  as `1199.239708` and would floor to 28752, buying one frame of phantom preroll.
  Pinned by `test/test_video_command_units.py`.
- **Archive-tier sources fail at probe, by design, with the tier named** —
  `head_object` is already called for the file size, so `archive_tier` costs no extra
  API call. S3 Intelligent-Tiering keeps `StorageClass` at `INTELLIGENT_TIERING` and
  signals cold storage through **`ArchiveStatus`** (`ARCHIVE_ACCESS` @ 90 d,
  `DEEP_ARCHIVE_ACCESS` @ 180 d when `EnableArchiveTiers` is on), so both fields must
  be checked. `GLACIER_IR` is deliberately *not* blocked — it reads like STANDARD. A
  completed restore (`ongoing-request="false"`) clears the block; one still running
  does not, and gets "retry once it completes" rather than "restore it". The message
  names the tier but deliberately carries no AWS CLI command — it surfaces through the
  gateway to arbitrary callers, and how to restore is the operator's business.
  `ValueError` is the right class: the probe state's ASL `Retry` covers only `Lambda.*`
  errors, so a permanent condition is not retried. Pinned by
  `test/test_s3_utils_units.py`.
- **`run_ffprobe` uses `-v error`, never `-v quiet`** — on failure the stderr text is
  the entire diagnosis (S3 `InvalidObjectState`, 403, `moov atom not found`).
  `-v quiet` reduced a batch of archive-tier failures to
  `Command exited 1 after 0.04s:` with an empty body. `run()` captures stdout and
  stderr separately, so the JSON on stdout stays clean.
- **Start timecode is not `start_time`, and only a large `start_time` breaks the
  keyframe scan.** QuickTime carries the first-frame timecode in a separate `tmcd`
  track (ffprobe surfaces it as a `timecode` tag), which does not shift presentation
  timestamps — a file can carry `timecode 14:00:13;10` with `start_time=0.000000`.
  Where a real origin offset exists, `best_effort_timestamp_time` is absolute while
  `nominal_sec` is zero-based, but an offset that is an **integer number of frame
  periods** is harmless: the same value feeds both `-ss` and `sec_to_frame`, `-ss` is
  itself relative to `start_time`, and the errors cancel exactly (measured — on a
  `start_time=0.04` file at 25 fps, i.e. exactly 1 frame, `-ss X` lands on the frame
  `sec_to_frame(X)` names; every non-zero offset found in the test corpus is
  frame-aligned like this). A *fractional*-frame offset would NOT cancel exactly —
  `round()` vs ffmpeg's seek-discard cut can differ by one silent frame — which is one
  more reason the eventual fix must subtract `start_time` rather than lean on
  cancellation. An offset larger than the 30 s search window makes the
  `<= nominal_sec` filter reject every keyframe and `find_prior_keyframe` raises.
  Observed in the wild: an MPEG-TS with `start_time` ≈ 49070 s (13.6 h — common for
  broadcast transport streams, whose PTS origin is the station clock) and inter-frame
  mpeg2video fails every chunk whose keyframe scan window the offset exceeds. The fix
  is to subtract the stream `start_time` before both uses (and before the
  `<= nominal_sec` filter); output-neutral on frame-aligned offsets. **Not yet
  applied** — deferred deliberately: zero exposure in the current content, and it
  touches the seek path.
- **`find_prior_keyframe` must not be called for chunk 0** — raises `ValueError` if
  `nominal_sec == 0`. ChunkTranscode guards this with `nominal_start_sec > 0`.
- **Token substitution uses `str.replace`** — not `.format()`. Avoids clashes with
  literal `{}` in ffmpeg filter expressions. Tokens: `{start_frame}`, `{end_frame}`,
  `{start_sample}`, `{end_sample}`.
- **`frame_interval = math.ceil(fps × frame_interval_sec)`** — ceiling, not round. A
  29.97 fps file with `frame_interval_sec=1.0` gives `frame_interval=30`.
- **Intermediate cleanup** — `.ts` and `.flac` chunks are deleted by StateMergeChunks
  on success. Set `KEEP_SEGMENT_CHUNKS=1` to suppress for debugging.
- **mxf2raw is optional for transcode, mandatory for `metadata` on MXF** —
  `run_mxf2raw` returns `None` if the binary is absent, times out, or emits unparseable
  XML. On the transcode path that degrades quietly: `is_all_intra` falls back to the
  ffprobe-only path (`None` for ambiguous codecs → `scan_is_all_intra`). On the
  `metadata` path `build_mxf_block` **raises** instead, because the group's contract is
  that a present `mxf` block holds real measurements (§8.3).
- **`mxf2raw` output is snake_case** — real bmx 1.6.0 emits `clip`,
  `start_timecodes`, `edit_rate`, `essence_type`. `parse_mxf2raw_info` reads through
  `_pick()`, which tries snake_case **first** and keeps the CamelCase spellings only as
  a cross-version fallback. Checking CamelCase first silently yields `{}` on every real
  file.
- **`Fraction("0/0")` raises, it does not return zero** — `ZeroDivisionError`, which
  will abort a whole surrounding `try`. `_fps_from_edit_rate` catches `ValueError` and
  `ZeroDivisionError` and returns `0.0`; `fps <= 0.0` is then the single sentinel that
  covers every parse failure.
- **`mxf.frame_count == -1` is valid MXF, not a failure** — an OP1A header that
  records no duration (observed on Sony eVTR D10 material) with a perfectly good
  `edit_rate`. `build_mxf_block` **omits** `frame_count` and `duration_sec` in that
  case rather than emitting the sentinel; absent means "the header did not say". Only
  `fps <= 0.0` raises (§8.3.1).
- **Multiple video/frame groups** — only the first `mp4` group and first `jpeg`/`png`
  group are processed; extras are logged as warnings and ignored. Multiple `wav`/`m4a`
  audio groups are all processed.
- **Frame-only mode** — an input with only `jpeg`/`png` output groups (no `mp4`) is
  valid. In that case `build_vfilters` skips the `[vts]` split and feeds `[vjpeg]`
  directly, avoiding a dangling label that ffmpeg would reject.
- **vfilter chain is conditional** — `yadif` is inserted only for interlaced sources
  (`field_order` not `progressive`/`unknown`); the `crop` filter is inserted only when
  `coded_crop_w > 0` or `coded_crop_h > 0`.
- **Manifest key is always `frame_manifest.json`** — the per-group frame manifest is
  written to `{prefix}/{group_name}/frame_manifest.json` regardless of the group's
  `name` field.
- **CodeBuild timeout is 90 minutes** — ffmpeg + openH264 + YASM compile from source
  takes 60–75 min on the ARM CodeBuild instance. Do not reduce the timeout.

---

# Design record

Numbered design decisions, kept because code comments and docstrings cite them by
section number. Entries marked **parked** are settled designs with no code behind them
yet.

## §1 Analysis-filter groups — why they land in this stack (parked)

A caller asks for the output of an ffmpeg *analysis* filter (one that writes
measurements to stderr rather than producing media) and receives parsed JSON as a
normal output group. Motivating case: EBU R128 momentary loudness for ad-break scoring
— but loudness is one instance of a recurring shape (`silencedetect`, `blackdetect`,
`astats`, `signalstats` are all the same request), so it is designed as a general
mechanism, not a `loudness` special case.

It belongs in media-processing rather than a stack of its own because this container
already has ffmpeg and already decodes the source audio — a separate stack would
re-download and re-decode the same audio to run one filter — and because analysis
filters produce *no media*: there is nothing for a dedicated workflow to own, the
output is a small JSON file.

## §2 Rejected: caller-supplied ffmpeg command

The original proposal was a generic "custom command" parameter letting the caller pass
their own ffmpeg argv. Rejected, and recorded because the idea will recur:

- **2a. It breaks the `workflow_id` contract.** A free-form command string hashes fine
  but makes idempotency meaningless: the same semantic request written two ways
  (`-vf scale=...` vs `-filter_complex [0:v]scale=...`) yields two prefixes and two
  full transcodes.
- **2b. It collides with the chunking machinery.** `build_ff_cmd`
  (`states/chunk_transcode/transcode_helper.py`) does not wrap a command — it assembles
  one from probe-computed state: `-ss` fast-seek with keyframe pre-roll, and
  `{start_frame}`/`{end_frame}`/`{start_sample}`/`{end_sample}` token substitution that
  makes `trim`/`atrim` correct *relative to the seek point* (and `afilters` gets
  different token semantics than `vfilters` on the audio-only path, where the seek is
  pre-rolled 3 s and sample tokens are rebased). A caller-supplied command either
  bypasses all of this — producing silently wrong output on any chunked asset — or must
  participate in a token protocol it cannot see.
- **2c. It is an arbitrary-execution surface.** `lib/proc_utils.run()` uses
  `subprocess.run(cmd_list)` with no shell, which is correct and worth keeping. But
  ffmpeg itself reads and writes files and opens network URLs. A caller-supplied
  command, in a Lambda holding the proxy-bucket role, is a filesystem and network
  primitive wearing an ffmpeg costume — and this is a gateway-exposed MCP tool
  reachable by an LLM agent.

**Kept from the idea:** the generalization. Callers should not need a code change per
filter; §3 delivers that declaratively.

## §3 The `analysis` output group (parked)

A third group kind alongside video / audio / frame. The caller names a filter from an
allowlist and supplies validated options; they never write argv.

```json
{"group_name": "loudness", "format": "json",
 "analysis": {"filter": "ebur128", "options": {"peak": "true"}}}
```

- `format: "json"` — new entry in `VALID_FORMATS`, with
  `ANALYSIS_FORMATS = frozenset(("json",))`.
- `analysis.filter` — must be in the allowlist. Unknown → `ValueError` at probe time.
- `analysis.options` — validated per filter against a small schema (type + range +
  allowed values). Unknown option keys are an error, not ignored: a silently-dropped
  option would produce plausible-looking wrong measurements.
- Canonical serialization (sorted option keys) so `workflow_id` idempotency holds.

### §3.1 Allowlist, initial

Only `ebur128` is required by the motivating consumer; the other four are specified to
confirm the mechanism generalizes — implement on demand, not upfront.

| Filter | Stream | Options (allowed keys) | Parser output shape |
|---|---|---|---|
| `ebur128` | audio | `peak`, `target`, `dualmono`, `panlaw` (**not** `framelog` — §5a) | time series `[{t, M}]` + classified groups |
| `silencedetect` | audio | `noise`, `duration`, `mono` | interval list `[{start, end, duration}]` |
| `blackdetect` | video | `black_min_duration`, `picture_black_ratio_th`, `pixel_black_th` | interval list |
| `astats` | audio | `length`, `reset`, `measure_perchannel`, `measure_overall` | per-channel **summary** block — no time axis |
| `signalstats` | video | `stat`, `out` | per-frame series (`YAVG`, `YDIF`, …) |

Adding a filter = one allowlist entry (filter name, stream kind, option schema) plus
one parser function and one seam reducer (§4.3). Per-filter quirks — none of the four
is a copy of the `ebur128` work:

- **`astats` is a summary, not a series** — one multi-line block per channel at end of
  stream; its reducer recombines per-chunk summaries with sample-count weighting and
  omits the stats that genuinely cannot combine.
- **`signalstats` does not log to stderr by itself** — it sets frame metadata, so the
  effective chain is `signalstats,metadata=print` and the parsed lines carry the
  `metadata` filter's prefix. Exactly why the allowlist is a registry mapping a
  caller-facing name to a chain, not a passthrough of the caller's string.
- **`silencedetect`/`blackdetect` emit paired start/end lines**, not self-contained
  records; the parser must pair them and close an unterminated final interval at
  stream duration, not drop it.
- **Log-prefix form is not uniform across filters or releases** — confirm the exact
  prefix against the pinned ffmpeg version per filter at implementation time.
- **Option units are not obvious** — `silencedetect`'s `noise` accepts dB or linear
  amplitude and silently means different things. The option schema must pin the
  accepted form, and parsers must normalize all emitted timestamps to integer ms.

### §3.2 Parsers

Each allowlist entry owns a `stderr → JSON` parser in a new `lib/analysis_filters.py`.
The `ebur128` parser is a port of an existing upstream producer's log parser (match
`[Parsed_ebur128_0 @ ` lines, keep `t` and `M`, `t` → integer ms, `nan` → `-100.0`),
plus its LUFS bucketing/contiguity-grouping classifier (`TAG_ABSOLUTESILENT` …
`TAG_VERYLOUD`, 100 ms gap grouping, small-group reduction, boundary padding for the
100 ms-discrete momentaries). **Do not port** the upstream abnormality-detection
functions — they sit behind a disabled flag there and have never run in production; if
wanted, specify and validate on real content first, not inherit switched-off.

### §3.3 Emitted shape

One file per analysis group. A common envelope (`filter`, `options`, `kind`) wraps a
per-filter payload — three payload kinds, not one:

| `kind` | Payload | Filters |
|---|---|---|
| `series` | `datapoints[]`, each with integer-ms `t` | `ebur128`, `signalstats` |
| `intervals` | `intervals[]` of `{start, end, duration}`, integer ms | `silencedetect`, `blackdetect` |
| `summary` | `channels[]` / `overall` — no time axis | `astats` |

`ebur128` additionally carries the classified `groups` reduction; both are needed —
onset scans read the raw series, silence classification reads groups. The `kind`
discriminator exists so a consumer can tell a series from a summary without knowing the
filter; flattening all three into one shape would let an `astats` payload with a
fabricated `t: 0` read as a one-sample series and quietly survive being fed to interval
logic.

## §4 Placement — chunk-parallel over the master source, reduced at merge (parked)

Driven by one requirement: **measurements must be master-referenced** — taken from the
source essence, not from anything this pipeline re-encoded. That puts two constraints
in tension:

- **A single continuous pass over the source does not fit one Lambda** for the assets
  this stack exists to handle: a >50 GB MXF is 9–12 minutes of S3 streaming alone at
  realistic presigned-URL throughput (`/tmp` is 4 GB on every tier, so no staging),
  before any decode. Audio-only analysis does not escape — ffmpeg still demuxes the
  whole container.
- **Naive chunked analysis produces wrong answers** (§4.1).

Resolution: analysis filters run inside the existing chunk-transcode invocations,
tapping the decoded source stream (§4.2); a per-filter reducer stitches the per-chunk
measurements in `StateMergeChunks` (§4.3). The Map has already solved the hard problems
— keyframe pre-roll, sample-accurate `atrim`, the 3 s audio pre-roll, 5-cores routing,
the 6-cores timeout retry — and already pays for the source decode.

### §4.0 Superseded ruling — one pass at merge, over the intermediates

The original ruling ran the filters once over the whole file in `StateMergeChunks`,
reading the concatenated `.flac` chunks ("FLAC is lossless, so measurements match the
source") and the `.ts` proxy. Single-pass placement correctly avoided every seam
problem — but its fidelity premise was false:

- **The FLAC intermediate is not master audio.** Channel-selected and forced to
  2-channel stereo; losslessness is per-sample — of a *downmix*. Loudness measured on
  it is not the program's loudness for any multichannel or multi-mono source, which is
  exactly what a large broadcast MXF carries.
- **The proxy is not master video.** Scaled, deinterlaced when the source is
  interlaced, re-encoded at proxy bitrate. `blackdetect` would mostly survive;
  `signalstats` would measure our transcode, not the source.

It also gave the merge state a full-duration decode against its 15-minute wall; the
revision reduces merge's addition to JSON reduction. Do not revive the superseded
design.

### §4.1 Why per-chunk results cannot simply be concatenated

`ebur128` is a stateful integrating filter: run per chunk and concatenated, its
momentary series carries a discontinuity at every chunk seam, the classifier (which
groups on >100 ms gaps) fabricates a group boundary at every seam, and silence-onset
consumers find onsets at transcode seams — artifacts of our chunking, not the content.
Every allowlist filter fails in its own way: a straddling `silencedetect`/`blackdetect`
interval splits into two halves that can each fall under the duration threshold and
vanish (a 4 s silence split 2 s/2 s against a 3 s threshold disappears); `astats` RMS
and DC offset need sample-count weighting the summary block doesn't carry;
`signalstats` `YDIF` is frame-to-frame and produces garbage at every seam's first
frame. Any filter whose output depends on temporal context beyond one frame, or that
aggregates over the stream, needs seam-aware reduction.

### §4.2 The chunk tap

Filters are grafted into the ffmpeg invocation `build_ff_cmd` already assembles, via
`split`/`asplit` where the stream is still master-referenced:

- **Audio filters** tap after decode, **before** channel mapping and the `-ac 2`
  downmix. Open design point for multi-mono sources: define "the program audio" for
  measurement (the streams `select_audio_streams` selects, pre-downmix, or an explicit
  stream selection on the group). Never silently measure the downmix.
- **Video filters** tap after decode, **before** `scale`/`yadif`/`crop`.
  Master-referenced means the interlaced source for interlaced content — record in the
  payload that `signalstats` saw fields, not deinterlaced frames.
- **Seam-exactness via the existing pre-roll.** Filters measure the *pre-rolled*
  stream (before `trim`/`atrim`); the per-chunk parser discards datapoints outside the
  chunk's nominal range. Warm-up for windowed filters (`ebur128`'s 400 ms momentary)
  happens during pre-roll, so the stitched series is exact at seams. Chunk-local
  timestamps are rebased to asset time (integer ms).
- **Measurements travel via S3, not the payload.** A per-frame `signalstats` series
  for one chunk is megabytes — over the 256 KB Step Functions payload limit. Each
  chunk writes its parsed measurement JSON next to the media intermediates and returns
  the key; the reducer reads the keys. These are deleted with the `.ts`/`.flac`
  chunks, so **reduction must run inside the merge group loop, before intermediate
  deletion**.
- **Analysis-only requests still run the Map.** Probe must force iterator creation for
  the stream kinds the filters need even with no media group — the most likely silent
  failure in this feature (verify on the first dev run with a loudness-only request).
  Such chunks are *measure-only*: decode + filters, `-f null -`, no encode, no media
  upload. The `metadata_only` shortcut does not apply; a combined request rides the
  invocation it would have run anyway.

### §4.3 Per-filter reduction

Reducers are pure JSON transforms (no media I/O) in the merge state's group loop; each
allowlist entry owns one, next to its parser:

| Filter | Seam problem | Reduction |
|---|---|---|
| `ebur128` | 400 ms warm-up at chunk start | none beyond concatenation once the pre-roll trim holds; classification runs once, globally, over the stitched series |
| `silencedetect` / `blackdetect` | straddling interval splits; each half can fall under the duration threshold | run per-chunk with the threshold at ~0; merge intervals touching a seam; apply the caller's threshold **after** merging |
| `astats` | RMS / DC offset not combinable from summaries alone | sample-count weighting from chunk durations (peak = max; RMS = √(Σnᵢ·rmsᵢ²/Σnᵢ); DC = weighted mean). Stats that genuinely do not combine (flat factor, peak count) are **omitted**, explicitly — not approximated |
| `signalstats` | `YDIF` garbage at each chunk's first frame | decode already starts at a prior keyframe, so emit from one frame before the chunk boundary and drop the first datapoint |

### §4.4 One decode, N filters

Within a chunk invocation the filters share the decode by construction — the taps live
in the same filter graph as the transcode chains. Audio and video filters cannot share
one tap; group by stream kind. And **the stderr prefix index is positional**:
`[Parsed_<filter>_<n>]`'s `n` is the filter's index in the whole chain, which varies
with the transcode chains present — demultiplex by the index assigned when the chain is
built, never a hardcoded `_0` (the ported parser matches the literal
`[Parsed_ebur128_0 @ ` and would silently return an empty series the moment `ebur128`
is not first).

### §4.5 Runtime

Audio filters add negligible cost to a chunk — the decode dominates and is already
paid. `signalstats` is the exception: full-source-resolution per-frame stats are
roughly decode-order cost; measure it on a 10-minute 1080i chunk before enabling it.
The merge state gains only JSON reduction.

## §5 Verified ffmpeg facts (measured, not assumed)

Each is an instance of the §12 failure mode — a silently empty series, no crash:

- **5a. `framelog=verbose` suppresses the entire series at default log level.**
  `framelog` sets the log *level* the filter emits at, not the level of detail; with
  ffmpeg's default `-v info`, per-frame lines logged at verbose are filtered before
  reaching stderr (measured: 30 `t:` lines at default framelog, **0** with
  `framelog=verbose`). Ruling: the analysis runner pins `-v info` and does **not**
  expose `framelog` as a caller option — log level is a transport detail of how the
  measurements are recovered, not a property of the measurement.
- **5b. `peak` changes the field count.** The upstream parser required exactly 12
  fields; `peak=true` yields 18 (mono) and `peak=sample+true` 21 (stereo) — every line
  rejected, empty series. `peak` appends per-channel columns, so a fixed count cannot
  hold. Ruling: **parse by field name, not field count** — walk the `key value` pairs,
  pick `t` and `M`, treat a line as malformed only when those keys are absent or
  non-numeric (with a sanity floor so genuinely corrupt lines still raise). That also
  makes per-release log-format drift survivable, and lets `peak` stay allowlisted.
- **5c. A JS truthiness guard drops the first datapoint.** The upstream parser ends
  with `if (datapoint.t && datapoint.M)` — in JS `0` is falsy, so the momentary at
  `t: 0` is discarded, and so is any datapoint whose `M` is exactly `0.0`. Port the
  *intent* (both keys present) as an explicit `is not None` check, not the bug. The
  `t: 0` sample is the one a scene at the top of the asset would look for.

## §6 Gateway impact of analysis groups

Minimal by construction: `output_keys[group_name]` → the analysis JSON key (no new
field); `outputs[]` labels already use the caller-chosen `group_name` and `.json`
already maps to `application/json`; the gateway InputSchema needs the `analysis` object
described on `output_groups` (the schema language supports no `enum`, so the allowlist
is documented in prose); the agent prompt gains one chaining recipe.

## §7 The `metadata` output group

A caller wants container/codec/stream detail and nothing else. Without this group the
only way to get it was to run a full transcode and read `data.probe_s3_key` — paying a
chunked ffmpeg pipeline for metadata the probe state already produced in its first few
seconds.

### §7.0 One auto-detecting format, not two

An earlier draft had `probe` and `mxfinfo` as sibling formats, with `mxfinfo` returning
`{"is_mxf": false}` on non-MXF sources. Collapsed into one auto-detecting `metadata`
format because:

- **`is_mxf: false` was a wart** — it existed only to answer "why is this empty?", a
  question the caller should never have to ask. With auto-detection the `mxf` block is
  present when meaningful and absent when not.
- **It removes a decision the caller cannot make.** An agent that does not yet know
  the container cannot know whether to request `mxfinfo`; its only safe strategy was
  "always request both, ignore one" — which is what auto-detection does, centrally.
- **The discriminator already exists and already runs first.** `is_mxf_format` reads
  `format_name` from the ffprobe result, and ffprobe already runs before group
  processing. That ordering is now load-bearing rather than incidental.
- **It generalizes.** "One `metadata` group, N tools, each contributing when it has
  something to say" accommodates a future tool without another format.

Cost of the collapse: `mxf2raw` runs on every MXF request whether or not the caller
wanted MXF detail — one extra bounded header read (it reads the footer index, not the
essence) of a file already being probed. Not a cost worth a caller-facing option.

### §7.1 What it emits — and why it is not just `probe.json`

The value added over raw `probe.json` is **surfacing the resolved facts** rather than
making an agent fetch and re-derive them from raw ffprobe JSON. Two fields are the
reason the group exists:

- **`duration_sec` + `duration_source`** — not a passthrough of `format.duration`,
  which is wrong or absent on exactly the broadcast formats this container exists to
  handle, but the `resolve_duration` priority chain, with the rung it landed on made
  visible (so the logged-last-resort `ffprobe_fmt` is visible rather than silently
  trusted).
- **`is_all_intra`** — not an ffprobe field at all: a codec heuristic with an
  empirical 20-second frame-scan fallback. No caller can derive it from raw ffprobe
  output.

`probe_info` is kept verbatim so nothing is lost, but the top-level fields are the
contract. See the payload example in the probe-state section above.

### §7.2 Control flow — metadata-only skips the Map and merge

A metadata-only request has nothing to chunk; before this group, the handler raised
"no compatible output_groups". Two constraints shaped the fix:

- A Map over an empty `ItemsPath` is legal, but `StateMergeChunks` raises on empty
  iterators, so falling through would fail the execution *after* probe succeeded.
- `StateUsageTracking` must still run — it is the `End: true` state whose payload the
  gateway reads; skipping to a `Succeed` state would break `get_job_status` for
  metadata-only jobs, and cost accounting should still work.

Ruling: a `Choice` state after "Probe media file" branching on an explicit
`data.metadata_only` boolean stamped by the probe handler, straight to "Usage
tracking". **Branch on the explicit boolean, never on `data.iterators` being empty**:
an empty iterator list is also what a bug looks like — a video request whose boundary
computation produced nothing — and that must fail loudly. The flag says "this was asked
for"; emptiness says "something went wrong". The flag is named for the group *kind*,
not a specific tool — §8 adds mxf2raw inside the same group without touching it.

Because merge is skipped, **probe writes the group copy and stamps
`data.output_keys` itself**; a metadata group in a *mixed* request is likewise handled
at probe time (one code path), and merge seeds its own `output_keys` from probe's so
the mixed request keeps both.

### §7.3 Granularity — the dividing line is "does the mode read the whole file"

**Ruling: two `detail` levels — `default` (format + streams + chapters + programs) and
`full` (adds `disposition` and stream `tags`). Frame- and packet-level granularity is
not offered at any level.**

Measured on a 35 MB / 5-minute H.264 MP4 served over HTTP with range support, counting
bytes the server actually delivered (the presigned-URL case this container runs in):

| Mode | Bytes read | JSON size | Whole file? |
|---|---|---|---|
| `-show_format -show_streams` | **3.7 MB** | 4.4 KB | no — bounded by `probesize` |
| `+ -show_chapters -show_programs` | 3.7 MB | 4.4 KB | no — same read |
| `-show_packets` | **35.5 MB** | 1.6 MB | **yes** |
| `-skip_frame nokey -show_frames` | **35.5 MB** | 28 KB | **yes** |
| `-show_frames` | 35.5 MB | 3.2 MB | **yes** |

Two things this measurement settles that intuition gets wrong:

1. **`-show_chapters` and `-show_programs` are free** — identical byte count. They go
   in `default`; a separate detail level for something that costs nothing would be a
   knob with no purpose.
2. **Keyframe-only probing is *not* a cheap middle ground.** `-skip_frame nokey`
   produces a small output and looks like the natural compromise — and reads **the
   entire file**. Small output, full cost. Anyone proposing "just keyframes" later
   should be pointed here.

Frame level is excluded in the units that matter: at the measured ~1.8 KB/frame of
`-show_frames` JSON, a 2-hour asset @ 30 fps is ~390 MB of JSON — beyond a sane single
S3 object, far beyond the 256 KB Step Functions payload limit, and on a 20 GB MXF the
input side means downloading the whole asset inside a 15-minute Lambda.
`-count_frames`/`-count_packets` are the same exclusion (full decode / full read). For
MXF the frame-accurate count is free anyway: `mxf2raw`'s `frame_count` from the footer
index (§8). `full` is not a second ffprobe invocation — `disposition` and `tags` are
already in the `-show_streams` output; it is a filtering decision, which is why it is
safe to expose and costs nothing. If a frame-accurate count is ever genuinely needed
for non-MXF, the honest shape is a separate state with its own Lambda tier and timeout
— not a `detail` level that silently turns a 2-invocation job into a full-asset decode.

### §7.4 Cost

Metadata-only is the cheapest possible execution — roughly two Lambda invocations
against the dozens a chunked transcode costs. Stated in the agent prompt so an agent
that needs only duration or codec stops requesting a throwaway proxy.

### §7.5 Gateway impact

`metadata` in the gateway's `_VALID_FORMATS`; `detail` validated (case-normalized, bad
value → `INVALID_PARAMETER`); `get_job_status` already surfaces `duration_sec` /
`has_video` / `has_audio` / `is_all_intra` for every job, so a metadata-only job
answers the most common questions **without an S3 fetch at all**; the Choice state
needed no name mapping (it emits `ChoiceStateEntered`, invisible to the history scan);
the InputSchema says `width`/`height`/`crop_*` are ignored for it.

## §8 The MXF contribution — `mxf2raw`, run automatically

Not a separate group: when the source is MXF, `mxf2raw` runs as part of the same
`metadata` group and its results land in the `mxf` block. The tool was already in the
image (added for MXF duration accuracy) and the wrapper module was already written but
imported by nothing — the cost had been paid and returned nothing, which is most of the
argument for wiring it in.

### §8.1 The detection gate

- `METADATA_FORMATS` stays a *set* rather than an equality check because §7.2's
  control flow keys off "is every requested group a metadata group", and a set keeps a
  future second member from reopening that logic.
- `is_mxf_format(format_name)` — a substring match over
  `("mxf", "mxf_d10", "mxf_opatom")` against ffprobe's `format_name` — is the gate.
  **`lib/ffprobe.py` is its canonical home** (the predicate reads ffprobe's output);
  a verbatim duplicate in `lib/mxf2raw.py` was deleted during implementation.
- Non-MXF → no `mxf2raw` call, no `mxf` key. Nothing to explain, nothing to branch on.

### §8.2 What the `mxf` block contains

`frame_count`, `fps`, `drop_frame`, `essence_type`, plus two additions over the raw
parse: **`duration_sec` = `frame_count / fps`** (the MXF-accurate duration, published
here rather than wired into `resolve_duration` — §8.4) and **`start_timecode`** (the
material timecode string, previously parsed only to derive `drop_frame` and then
discarded; broadcast callers care about it directly). The former standalone
`mxf_info.json` writer was deleted rather than left as a second, divergent copy of the
same facts.

### §8.3 The one case that must raise

Auto-detection removes the "empty means not MXF" ambiguity, but not the case where the
tool genuinely fails on a real MXF. `run_mxf2raw` returns `None` on every failure path,
and the parser's tolerant sentinel tuple is correct for its *original* role as an
optional accuracy hint — and wrong here: a caller handed `frame_count: -1` has been
handed a plausible-looking wrong number. So: MXF source + mxf2raw failure → **raise**;
non-MXF → no `mxf` key; the tolerant behaviour stays available for the internal
accuracy-hint use. The invariant: **the presence of the `mxf` key means MXF, and its
contents are always real measurements.** There is no third state.

#### §8.3.1 Superseded by real data — `frame_count == -1` is not a failure

"Never emit the `-1 / 0.0` sentinels, raise on either" conflated two different things.
Real `mxf2raw` output on Sony eVTR D10 material has `duration.count = "-1"` together
with a perfectly good `edit_rate` of `25/1` — a **valid OP1A header that records no
duration**, not a tool failure; raising would reject a readable file. The implemented
rule splits the sentinels:

- **`fps <= 0.0` raises.** It is the single value every parse failure lands on, and
  `duration_sec` divides by it, so the check comes first regardless.
- **`frame_count < 0` omits** `frame_count` and `duration_sec` from the block, with a
  warning. Absent means "the header did not say" — the honest answer. The block still
  carries `fps`, `drop_frame`, `essence_type`, `start_timecode`, `mxf_info`.

Two real parser bugs surfaced during verification, both of which had made the parse
return the sentinel tuple on **every** genuine MXF: bmx 1.6.0 emits snake_case keys
(the code tried CamelCase first and always fell through to `{}`), and
`Fraction("0/0")` raises `ZeroDivisionError` (aborting the surrounding `try`, so one
unusable edit rate discarded every other field). Both fixes are in the Gotchas above.

### §8.4 The `duration_source` divergence

The intuitive `mxf2raw frame_count / fps` rung in `resolve_duration` deliberately does
not exist. Wiring it in changes `duration_sec` — which feeds segment boundary
computation for every MXF asset — and needs its own before/after verification on real
MXF content; a transcode-affecting change must not ride along with read-only features.
Instead the MXF-accurate number is published as `mxf.duration_sec` in the same payload
as the ffprobe-derived `duration_sec`, so the two can be compared for free. Record the
deltas: they are the evidence a future decision about `resolve_duration` would need.

### §8.5 Gateway impact

**Nothing** — the point of auto-detection: no extra format, no extra InputSchema field,
no decision procedure in the prompt. One descriptive prompt line: for MXF sources the
metadata payload also carries an `mxf` block with frame-accurate count and start
timecode.

## §9 Rejected for this container: `exiftool`

ExifTool is still the best tool in its space — and that is precisely why it should not
go in this image. There is no credible Python reimplementation: the PyPI packages are
subprocess wrappers around the same Perl binary, and the pure-Python alternatives cover
a fraction of ExifTool's two-decades-deep per-vendor tag knowledge. So the real choice
is "add Perl + ExifTool to this image" or "don't". Don't:

- **9a.** It means adding a Perl toolchain (and its `make test`) to a Python image
  whose runtime stage installs exactly one system package — a large, permanent cost
  across all seven Lambda variants for a feature with no identified consumer.
- **9b.** The build has no headroom: ffmpeg + openH264 + YASM already take 60–75 min
  against the 90-minute CodeBuild ceiling. This is the constraint that turns "some
  extra image size" into "the build starts failing".
- **9c.** A deployed ExifTool container already exists elsewhere in the fleet,
  already consumed by an image-ingest pipeline. Duplicating it here creates a second
  implementation on a different runtime, version-synced by hand.

**The distinction that decides it:** ffprobe and mxf2raw belong here because **they are
already in this image for transcode reasons** — exposing them costs a wrapper. ExifTool
is in no way a transcode dependency; adding it would build a new capability into this
container rather than surface one it already has (§1's test, and ExifTool fails it). If
ExifTool is wanted through the agent gateway, give the existing ExifTool container its
own gateway target — one ExifTool in the fleet, not two.

## §10 `proc_utils.run()` must return stderr (parked, with §3–§6)

`run()` returns `result.stdout` and discards stderr on success (it is only logged, on
failure). Analysis filters write their measurements to stderr and exit 0, so the
current runner throws away exactly the data §3 needs; the capture site is the
chunk-transcode invocation (§4.2). Add a stderr-returning path (a `capture_stderr`
parameter or a sibling function) without changing `run()`'s existing return type —
`run_ffprobe` and `run_mxf2raw` both consume stdout. Not needed by the metadata group.

## §11 Callback mechanism — `callback_task_token`

The family consumer pattern, ported from the bedrock-inference workflow's callback
design. `process_media` (and `quick_edit`/`capture_frames`) accept an optional
`callback_task_token` from a consumer workflow's ONE held `waitForTaskToken` state. The
gateway registers the token in `MediaProcessingCallbackTokensTable` BEFORE
`StartExecution` (the row exists before the run can possibly finish), and
[gateway_tool/callback_waker.py](gateway_tool/callback_waker.py) — fired by one
EventBridge rule on this machine's native status-change events — completes every
registered token with the submit handle + terminal `status` AS DATA (a FAILED verdict
is the consumer's to handle via its own Choice state, never a mid-machine detonation).

- On FAILED the payload also carries the execution's `error` (the raising **class
  name** — e.g. `NoCompatibleOutputGroups` when every requested group needs a stream
  the source lacks, [lib/input_schema.py](lib/input_schema.py)) and a truncated `cause`
  (the Lambda error JSON whose `errorMessage` names the condition); TIMED_OUT/ABORTED
  status-change details carry neither, and the payload message says so. The
  dispatcher's synchronous absorbed-into-finished completion emits the SAME shape —
  deliberately: `error` is the class name on both paths and the sentence lives in
  `cause`, so consumers should read both fields. The split exists because a consumer
  that could only echo `error` once had nothing better than "see the run prefix" to
  report over an empty prefix.
- Absorbed-into-finished runs complete the token synchronously (paid-work reuse).
- With a token present, every gateway refusal RAISES (the dispatcher's raise-vs-return
  choke point) — a waiting caller cannot read a returned error dict.
- `describe_tool` advertises `features: ["callback_task_token"]`.
- Token row format: `(execution_arn, token_hash)` with `token_hash =
  sha256(token)[:32]` — a ROW KEY (multiple waiters absorbed into one run), never an
  identity; the dispatcher module is pinned to contain no hashing so the workflow-id
  hash cannot be quietly reimplemented beside its one true home
  ([gateway_tool/callback_registry.py](gateway_tool/callback_registry.py)).
- **Table posture / DeletionPolicy decision:** the DynamoDB table holds rebuildable
  derived state — transient task-token rows with a TTL on every item, no customer
  data. Default SSE is deliberate (nothing worth a CMK's key-management cost), no
  point-in-time recovery (a restored row's token belongs to a wait state that has
  already timed out), and the table is designed to delete with the stack. Its
  `TableName` is createOnly, so renaming it REPLACES the table — safe only while no
  callback run is in flight.

This exists so consumer workflows (speech-diarization, audiostem-separation,
face-recognition) can stop invoking this stack's STATE MACHINE directly
(`states:startExecution.sync:2` — media-processing's internal execution output as their
API, no idempotency, extra IAM) and use the standard sibling pattern: invoke the
GATEWAY with a token, read outputs from the run prefix via the output contract.
Notification is a SERVICE concern owned here once — never a table+rule+waker per
consumer stack, which would multiply by N workflows and fan every completion out to N
rules for N−1 misses. Modules `callback_registry.py` (byte-identical family copy) +
`callback_waker.py` ride the gateway zip.

## §12 Verification pins

The failure mode this stack is most exposed to is **a plausible-looking wrong number
produced by a swallowed error** — wrong loudness momentaries would place ad breaks at
wrong times with no crash and no log line. Parsers must raise on unparseable input
rather than returning an empty series, and no `except Exception` may wrap a parse path.

### §12.1 Analysis groups (when implemented)

Unit fixtures: real captured ffmpeg stderr (never hand-written lines — the log prefix
and field layout are what varies), including a `nan` line, a malformed line, `peak`
lines at both field counts (§5b), a surviving `t: 0` datapoint (§5c), `framelog`
rejected as unknown (§5a), a straddling interval that must merge and re-threshold, RMS
recombination over unequal chunk lengths, warm-up trim at a chunk start, the `YDIF`
first-frame drop, timestamp rebase. Deployment-only checks: an analysis-only request
produces iterators and measure-only chunks (the silent-failure case); the stitched
momentary series has uniform ~100 ms gaps across a multi-chunk asset (pre-roll warm-up
and rebase verified together); parity against the upstream producer on a stereo asset,
then on a **multichannel** asset confirm the measurement *differs* from a
FLAC-intermediate measurement — identical numbers would mean the tap landed downstream
of the downmix; chunk duration and Map payload size before/after taps.

### §12.2 The `metadata` group

Unit-pinned: metadata-only yields zero iterators with `data.metadata_only` true only
when *every* group is metadata; **a video request that legitimately produces zero
iterators still raises** (the regression the explicit flag exists to prevent — a
refactor switching the branch to "iterators is empty" passes every other test); MXF →
`mxf` key present, non-MXF → key **absent entirely** (assert absence, not falsiness);
MXF where `run_mxf2raw` returns `None` raises — keep this test adjacent to the non-MXF
one, they are the same "no result" internally and only one may quietly succeed;
`fps <= 0.0` raises clearly; `frame_count < 0` omits without raising, rest of block
populated (keep both MXF fixtures — one with a real frame count, one D10 with `-1`);
`mxf.duration_sec == frame_count / fps` when present; `start_timecode` present as a
string; `detail: "full"` adds `disposition`/`tags` **with no additional ffprobe
invocation** (assert on call count — a re-probing implementation would be invisible in
the payload and double the cost); no `detail` value produces `frames`/`packets` keys.
Deployment checks: metadata-only reaches Usage tracking and SUCCEEDS with a full
`get_job_status` payload; stays cheap on a large MXF (two invocations, bounded read,
for both ffprobe and mxf2raw's footer-index read); mixed request writes both outputs;
`duration_sec` vs `mxf.duration_sec` deltas recorded (§8.4); the
`-show_chapters -show_programs` addition — the one non-additive edit, it changes
`run_ffprobe` for every execution — leaves probe payloads and segment boundaries
unchanged on assets that already worked.

## §13 Not in scope

Sound-event detection (separate workflow, separate model); upstream abnormality
detection (dead code there — specify and validate separately if wanted); parsers for
the four non-ebur128 filters (registry designed to hold them; each needs its own parser
and seam reducer once a consumer exists); caller-supplied argv (§2); frame/packet-level
ffprobe detail (§7.3); `mxf2raw` options beyond `--info --info-format xml`; replacing
the upstream loudness producer (depends on §12.1's parity check); an ExifTool gateway
target (§9 rejects ExifTool *in this container*, not as a capability — the existing
container should get its own target); wiring the mxf2raw rung into `resolve_duration`
(§8.4 — changes segment boundaries, needs its own verification, and only if the
recorded deltas matter).

## §14 Quick edit and frame capture

Two tools on the existing gateway target — `quick_edit` (compose ONE output video from
an inline JSON cut list over 1..N sources) and `capture_frames` (one still per
millisecond marker). Same state machine, no new states: probe gained the
cut-list/marker branches, chunk-transcode gained the renderers, merge is unchanged for
edits (dumb concat of the same `segment_video/.ts` + `segment_audio/.flac` pairs) and
writes `capture_manifest.json` for captures. Shipped and live-validated (frame-exact on
multi-source stress edits spanning interlaced MPEG-2 MXF, ProRes SD, sub-SD upscales,
and mixed 23.976/25/29.97/50 fps sources).

### §14.1 Contract decisions (each argued and settled)

- **Gateway surface: separate TOOLS, not process_media output groups.** process_media's
  contract hangs off ONE `asset_s3uri` (the identity anchor); a cut list over N sources
  has no honest value for it, and the gateway's SchemaDefinition has no `oneOf` to
  express conditional requiredness. The family answer to a second input shape is a
  second tool.
- **Inline JSON cut list, never an EDL file.** EDL *semantics* are borrowed
  selectively: source in/out = `in_ms`/`out_ms`; record in/out is DERIVED and reported
  in the manifest, never required (cumulative timecode arithmetic is what agent callers
  get wrong; record≠source duration would open the retime door); transitions ride the
  INCOMING segment (`transition_in`); record gaps are never inferred.
- **Segment order IS the output timeline, and the segment list IS the caller's
  duration budget**: output = Σ(out−in) − Σ(crossfade D).
- **Transitions: cut | fade | crossfade.** Crossfades use OVERLAP-TRIM semantics
  (consume D from each neighbour — only marked material, validates locally) and render
  as JUNCTION clips: tail-D of N and head-D of N+1 blend via xfade/acrossfade in their
  own Map iteration, bodies trimmed, merge still a dumb concat. NLE-style "handles"
  (using material beyond the marks) is a named non-feature. `fade_black` is NOT a type:
  **black is a SEGMENT** with the reserved source `"black"` (EDL's BL reel — keeps
  items homogeneous for the schema, keeps black inside the duration budget), and
  fade–hold–fade is composition: black between two crossfades. Plain `fade` is kept as
  ergonomics.
- **Per-segment `crop` in NORMALIZED 0–1 ratios** (reframing, e.g. vertical social
  clips): face-recognition boxes are already in that space, and the evidence runs on
  proxies while the edit cuts sources — pixel coordinates would crop the wrong region.
  WHERE to crop is the agent's job (deterministic-first: no in-workflow auto-reframe);
  executing it is one `crop` filter ahead of the fit. Static per segment; subject
  movement is solved by cutting the segment.
- **`channel: av|v|a`** — what the segment CONTRIBUTES; the timeline stays sequential
  ("v" = silence under, fillable by overlays; "a" = black picture). Parallel placement
  is exclusively the overlays' job; channel never creates a second video stream.
- **Audio overlays**: `replace` (voiceover/J-L cuts) and `mix` (beds, static `gain_db`
  −60..+12 — deliberately NO auto-ducking; a full-programme ducked mix is ad_mix's job
  and ad_mix stays: hundreds of takes, sidechain ducking, per-take fitting, post-mix
  mastering, audio-only output anchored to one asset). Anchors: `at_record_ms`, or
  boundary-glued via `transition_in.audio_bed` with `lead_ms`.
- **HDR sources REFUSE by name** (v1): the LGPL ffmpeg build carries no zscale
  (libzimg not enabled; adding it is a deliberate license call), and native filters
  cannot linearize PQ/HLG. Verified against the deployed binary; every other needed
  filter is present.
- **Caps (deliberate ratchets)**: 100 segments (confirmed vs the Lambda envelope:
  merge is a byte-bound remux governed by the 1 h cap, piece count is nearly free), 25
  distinct sources (overlay/bed sources count), 25 overlays, 1 h Σ(out−in) (authored
  black counts; crossfades shorten), 5 s transitions, 500 markers. Constants in
  [lib/cut_list.py](lib/cut_list.py) and [lib/frame_capture.py](lib/frame_capture.py).
- Final encode = the existing mp4 policy (compute_bitrate + faststart); target
  resolution must be even in both dimensions (4:2:0); mod-16 is NOT required (H.264
  crops internally — 1080 itself is not mod-16).

### §14.2 Frame accuracy — the chunk pipeline's machinery, reused

Contract math is exact-Fraction integer-ms (`lib/cut_list.ms_to_frame`, round-half-up
on the target grid; in-inclusive/out-exclusive so butt joints never dup or drop). Float
timestamps go through `video_command.sec_to_frame` — the one timestamp→frame
conversion. Rendering NEVER trusts seek precision: all-intra sources seek the frame
directly; inter-frame sources seek the PRIOR KEYFRAME (`find_prior_keyframe`) and pick
frames by COUNTING — `trim=start_frame` on the SOURCE grid, then the normalization
matrix (yadif → crop → fit + letterbox, never stretch → setsar 1:1 → fps dup/drop, no
motion interpolation → yuv420p SDR BT.709), then the TARGET-grid pin
(`trim=end_frame=n`). **`select` is banned for frame picking** (frame-inaccurate
history here, and select never EOFs its branch — a post-select yadif starves and the
graph reads to EOF; a unit test refuses any `select=` in a capture graph). Audio
follows the house conform (`aresample=48000:async=1 … apad …` then SAMPLE-based atrim
against a 3 s pre-roll; `apad` keeps a short audio stream from desyncing the concat),
and `adelay` uses the sample suffix, never rounded milliseconds. Composition rules,
pinned: identical mezzanine params on every plan entry (junction splices must concat
seamlessly); micro-fades on CUT edges only; a caller fade crossing a slice boundary
truncates there (v1).

### §14.3 Loudness (two-pass loudnorm, −23 LUFS EBU R128)

Pass 1 (measurement) runs at PROBE, **once per unique SOURCE** over a fixed sampled
window (300 s starting at 180 s in — skips bars/tone/slates/cold opens; whole file when
shorter), thread-pooled (I/O-bound presigned reads). This superseded the first shipped
design (per segment over its exact in/out, serial), for three reasons: a per-SOURCE
gain corrects the master-level mismatch between sources while PRESERVING intra-source
dynamics (per-segment normalization pushed a quiet scene and a loud scene from the same
master both to −23 — flattening intent); cost dedupes from N segments + M elements to U
unique sources; and the fixed shallow window removes deep seeks from measurement
entirely.

Pass 2 applies the source's gain as a PLAIN `volume=` filter — deliberately never
loudnorm: loudnorm's linear mode silently FALLS BACK to dynamic processing when the
gain would exceed its TP target, and dynamic mode outputs 192 kHz, which quarters every
sample-based atrim downstream (measured live: "Expect 7200 ms. Received 1800 ms").
Asked and answered: loudnorm has NO output-rate parameter (no way to refuse the
fallback), and appending `aresample=48000` would fix the duration while silently
shipping the WRONG processing — dynamic mode is time-varying, which breaks junction
gain-consistency and re-flattens the intra-source dynamics the per-source design
preserves; the loud duration failure was the honest form of the bug. If loudnorm is
ever reintroduced as a PROCESSOR in a trim-bearing graph, `aresample` after it is
mandatory — but prefer `volume=` for constant gain. loudnorm's two legitimate seats:
pass-1 measurement, and ad_mix's final whole-mix mastering (dynamic is the point there;
WAV out; nothing sample-trims after).

Junction sides, bodies and overlay elements all key by SOURCE, so junction splices are
gain-consistent by construction. TP SAFETY: linear gain is clamped against the sampled
WINDOW's true peak, so a hotter-than-window passage could clip after an upward gain —
every entry's audio chain therefore ends with `alimiter=limit=0.98` when loudnorm is on
(the ad_mix precedent). Mixing order: programme envelopes first, replace-mutes, then
overlays with only their own gain/envelopes. Default ON when segments span >1 distinct
source; `loudnorm` parameter overrides. The manifest reports measured loudness per
SOURCE (under `sources`), so quiet-source vs dropped-audio stays attributable.

### §14.4 Captures (the companion mode)

Markers in ms over one or more sources; one still per marker, snapped to each SOURCE's
own frame grid (there is no target fps for stills); optional normalized `crop` per
marker (same reframe window and validator as quick_edit); optional `resize` canvas
(fit + pad, never stretch; native otherwise); deterministic basenames
`frames/frame_{index:04d}.{ext}`. Probe groups markers by source, SORTS by position,
and span-caps batches at 600 s of content; the renderer runs ONE ffmpeg per batch — one
keyframe scan, one seek, one sequential read, K trim branches (deint once, pre-split) —
because deep RANDOM access into a large long-GOP MXF over presigned HTTPS costs minutes
per seek while sequential read is cheap (measured live; per-marker seek pairs overran
the 900 s Lambda before this shape). `capture_manifest.json` is the join: index → key,
requested vs actual ms, frame.

### §14.5 Evidence outputs

`edit_manifest.json` (fixed label, written at probe): requested vs actual per cut in ms
AND frames, the derived record timeline per segment, crop in ratios AND pixels,
normalizations applied (deinterlace / scale_pad / silence_insert), transitions,
overlays with measured loudness, source ETags (the same-URI-new-content idempotency
caveat, evidenced), `crossfade_mode: "overlap-trim"`. Silence is always attributable: a
dropped track shows `silence_insert`; a quiet source shows its measured input loudness.

### §14.6 Deferred, by name

`crossfade_mode: "handles"`; `at_segment` overlay anchor; overlapping `mix` overlays;
animated/keyframed crop; HDR output and HDR-input tonemapping (the libzimg license
call); audio-only quick edit (podcast cuts — merge already speaks wav/m4a);
per-boundary transition defaults beyond `transition_in`.

## §15 Usage tracking must bill failed attempts

`usage_tracking.json` once built `durations` with `if i["succeeded_ts"] is not None`,
so a `TaskFailed` — a Lambda killed at its timeout, the most expensive invocation
possible — contributed 0 s to `total_billed_sec` and `wall_sec`, and the report
under-reported *worst* exactly when the workflow behaved worst. Chunk-transcode retries
are routine in this stack, so it matters here. Invariants to preserve:

- `_billed_sec` falls back to `failed_ts`, and 0.0 is correct **only** when an attempt
  has NO terminal event at all — that case is what `incomplete` means.
- `wall_sec` sums attempts per `entered_id` *before* taking the max across occurrences,
  because every retry attempt chains back to the same `TaskStateEntered` while
  concurrent Map iterators each get their own.
- A `waitForTaskToken` state's billed runtime ends at `TaskSubmitted` (when its Lambda
  returns) — `TaskSucceeded` fires when the held token completes, potentially a whole
  sibling run later, and the sibling's own bill already covers that time. Plain
  `lambda:invoke` emits no `TaskSubmitted`, so submit time is preferred only where it
  exists.
- The per-invocation cost loop iterates *all* invocations with per-invocation ARN
  resolution, so mixed-size chunk-transcode retries are charged at the right tier.

Pinned cross-stack by `micro-workflow-bundle/test/test_units.py:
TestUsageTrackingBillsFailedAttempts` — a regression fails the bundle suite.
