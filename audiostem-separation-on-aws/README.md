# audiostem-separation-on-aws

AWS Lambda container for audio stem separation using the Bandit v2 ONNX model. Separates audio into three stems — **speech**, **music**, and **sfx** — via a Step Functions workflow. Accepts any media file; non-WAV sources (and WAVs that are not 48 kHz mono) are converted before separation.

## Workflow States

Defined in [definitions.json](definitions.json). Entry point for every state: [lambda_function.py](lambda_function.py) dispatches on the `operation` field.

```
StateCheckAndPreprocess
  ├─ (needs_conversion=false) ────────────────────────────────────────────┐
  └─ (needs_conversion=true)                                               │
       └─ Run media processing  (lambda:invoke.waitForTaskToken)          │
            └─ StateExtractProcessedAudio                                  │
                 └─ StateFixedChunkingForAudioStemSeparation  ←────────────┘
                      └─ Map: StateRunBanditV2Model  (MaxConcurrency=40)
                           └─ StateConcatenateAudioChunksByStems
                                └─ StateUsageTracking
```

**State responsibilities:**

1. **StateCheckAndPreprocess** — Validates the event via `input_schema`, probes the source file, and routes to one of three paths:
   - **Fast-path**: source is a WAV, exactly 48 kHz, mono — used as-is.
   - **WAV-normalise path**: source is a WAV but wrong sample rate or multi-channel — resampled to 48 kHz mono int16 in-Lambda by [states/check_and_preprocess/wav_to_wav.py](states/check_and_preprocess/wav_to_wav.py) using concurrent S3 byte-range reads + multipart upload. `needs_conversion=false`.
   - **Media-processing path**: source is not a WAV (MP4, MXF, MKV, …) or header parse failed — builds `data.media_processing_request` (gateway params for `media-processing___process_media`) and sets `needs_conversion=true`.
2. **Run media processing** — HELD `lambda:invoke.waitForTaskToken` task (the family sibling pattern — replaced a direct `startExecution.sync:2` of MP's state machine; Design record §2). [states/submit_media_processing/lambda_function.py](states/submit_media_processing/lambda_function.py) probes MP's `describe_tool` for the `callback_task_token` feature (refuse-before-spend), then submits `process_media` with `idempotency_token=audiostem-{uuid}` (a redrive absorbs into the same MP run) and this state's task token. MP's callback waker completes the token with the submit handle + terminal `status` AS DATA — always `SendTaskSuccess`, never a detonation; enforcing the verdict is the consumer's job. MP is a HARD dependency, so every submit-path failure raises rather than self-skips.
3. **StateExtractProcessedAudio** — Reads `data.media_processing_output`. Gateway handle (has `output_s3_prefix`): enforces `status == "SUCCEEDED"` — a FAILED verdict is enforced BY NAME (Design record §3): MP refusing the wav group as `NoCompatibleOutputGroups` raises [`lib.exceptions.NoAudioStream`](lib/exceptions.py) carrying MP's own message (the class name is the stable code in the execution's error field, matched across both completion shapes plus the legacy plain-ValueError wording), and every other failure re-raises with MP's forwarded error/cause passed through instead of an opaque "see the run prefix". Derives the wav key deterministically from MP's output contract — `{output_s3_prefix}/audiostem_sep/audio_48k_mono.wav`; `media_processing_arn` is the handle's `job_id`. A non-handle shape raises (the legacy nested-execution shape was retired — pre-conversion FAILED runs cannot be redriven anyway, their old ASL's direct-start IAM is gone; resubmit through the gateway instead). Probes the WAV header, writes `data.audio_bucket / audio_key / audio_cfg / media_processing_arn`, clears both `media_processing_request` and `media_processing_output`.
4. **StateFixedChunkingForAudioStemSeparation** — Reads WAV header (byte-range GET if `audio_cfg.duration_sec` is null); aligns `fixed_chunk_size_secs` to nearest multiple of 8 s; builds a narrow `item_cfg` (16 fields) and emits `data.iterators`. `audio_cfg` is intentionally excluded from iterators — `duration_sec` is passed to `_build_iterators` directly.
5. **StateRunBanditV2Model** — Map state. Each iterator fetches its audio slice from S3 (byte-range GET), runs the Bandit v2 encoder once, then up to three decoder heads per 8 s internal chunk. Resamples each stem to its target frequency; uploads per-chunk WAVs and a `chunk_NNN.json` metrics file. When `speech_safe` is enabled it also builds the gated-mix `speech_safe` chunk here ([lib/speech_safe.py](lib/speech_safe.py), Design record §5) — the mix slice and the raw speech stem coexist sample-aligned pre-resample, so the patch costs no extra inference or IO; the chunk metrics record the patch spans and every gate knob.
6. **StateConcatenateAudioChunksByStems** — Determines active stems via hard-bracket read of `data.{stem}_stem` (plus the `speech_safe` lane when enabled — it concatenates exactly like a stem). For each active stem, streams chunk PCM into an S3 multipart upload. Total WAV size is pre-calculated from iterator metadata so the header is written upfront — no fixup pass. Uploads aggregate `audiostem_sep.json`. Also owns the two AUDIO-ANALYSIS products ([lib/audio_analysis.py](lib/audio_analysis.py), Design record §4): per-lane RMS envelopes ride the same concatenation streams (free — the PCM is already in memory), the ORIGINAL mix gets its own windowed byte-range pass (envelope + line-up tone detection; tones are measured on the original only, since which stem a pure sine lands in after separation is a model artifact), and the state writes `stem_activity.json` (activity spans per lane + `tones[]`, `original` = non-silence of the mix, plus the `speech_stem_gaps` lane whenever `speech_stem` was enabled — mix active while the raw stem is near-silent, an upper bound on deleted speech; Design record §5) and `stem_waveforms.json` (peak-normalized rendering envelopes). Deliberately NO bars_and_tone verdict here — audiostem reports the measured tone; naming what a span means is the fusion layer's job.
7. **StateUsageTracking** — Walks the parent execution history to compute per-state cost. Reads `data.media_processing_usage_key` (written by `StateExtractProcessedAudio`) — if set, reads MP's own published `usage_tracking.json` from that key and adds its `total_estimated_cost` to this run's. Writes `{output_prefix}/usage_tracking.json`.

## Input Schema

Defined and validated by [lib/input_schema.py](lib/input_schema.py). All `input_config` and `output_config` fields are optional — defaults are applied by `validate_and_normalize()`.

```json
{
  "uuid": "<UUID>",
  "input": {
    "s3uri": "s3://bucket/path/to/source.mp4",
    "input_config": {
      "model_id": "bandit_v2",
      "fixed_chunk_size_secs": 180
    }
  },
  "output": {
    "s3uri": "s3://bucket/output/prefix",
    "output_config": {
      "speech_stem": true,
      "speech_resampl_freq": 16000,
      "music_stem": true,
      "music_resampl_freq": 48000,
      "sfx_stem": true,
      "sfx_resampl_freq": 32000,
      "speech_safe": false,
      "speech_safe_resampl_freq": 48000
    }
  },
  "data": {}
}
```

`model_id` is fixed to `"bandit_v2"` — not configurable. `fixed_chunk_size_secs` minimum is 60 s. Valid resampling frequencies: 8000, 16000, 32000, 44100, 48000, 96000.

## Internal Data Fields

`StateCheckAndPreprocess` flattens all config into `data` for downstream states:

| Field | Written by | Read by |
|-------|-----------|---------|
| `model_id`, `fixed_chunk_size_secs` | CheckAndPreprocess | FixedChunking |
| `speech_stem`, `speech_resampl_freq` | CheckAndPreprocess | FixedChunking, RunBanditV2, Concatenate |
| `music_stem`, `music_resampl_freq` | CheckAndPreprocess | FixedChunking, RunBanditV2, Concatenate |
| `sfx_stem`, `sfx_resampl_freq` | CheckAndPreprocess | FixedChunking, RunBanditV2, Concatenate |
| `speech_safe`, `speech_safe_resampl_freq` | CheckAndPreprocess | FixedChunking, RunBanditV2, Concatenate |
| `audio_bucket`, `audio_key` | CheckAndPreprocess (all paths) | FixedChunking, Concatenate, UsageTracking |
| `output_bucket`, `output_prefix` | CheckAndPreprocess | FixedChunking, Concatenate, UsageTracking |
| `needs_conversion` | CheckAndPreprocess | definitions.json Choice state |
| `audio_cfg` | CheckAndPreprocess (WAV paths); `None` on conversion path | FixedChunking |
| `media_processing_request` | CheckAndPreprocess (conversion path); `None` on WAV paths | SubmitMediaProcessing (gateway params); cleared by ExtractProcessedAudio |
| `media_processing_output` | MP's callback waker via `SendTaskSuccess` — the submit handle `{job_id, workflow_id, asset_s3uri, output_s3_prefix}` + terminal `status` | ExtractProcessedAudio; cleared after extraction |
| `media_processing_arn` | ExtractProcessedAudio (the handle's `job_id`) | — (traceability only) |
| `media_processing_usage_key` | ExtractProcessedAudio (derived: `{MP run prefix}/usage_tracking.json`) | UsageTracking (child-cost read) |
| `aligned_chunk_secs` | FixedChunking | — (observability only) |
| `iterators` | FixedChunking → Map (replaced by Map output) | Concatenate |
| `audiostem_sep_output` | Concatenate `{speech_key, music_key, sfx_key, speech_safe_key, metrics_key, stem_activity_key, stem_waveforms_key}` (stem keys: active stems only; `speech_safe_key` only when enabled) | — (terminal SFN output for external callers) |
| `usage_tracking` | UsageTracking | — (terminal SFN output) |

`media_processing_arn` and `media_processing_usage_key` are only present in `data` on the conversion path. Fast-path and WAV-normalise paths never write them; `StateUsageTracking` reads the usage key with `.get()` and skips child cost tracking if `None`.

## Output Files (S3)

All written under `{output_prefix}/`:

| File | Contents |
|------|----------|
| `audio_speech.wav` | Concatenated speech stem (16 kHz default) |
| `audio_music.wav` | Concatenated music stem (48 kHz default) |
| `audio_sfx.wav` | Concatenated SFX stem (32 kHz default) |
| `audio_speech_safe.wav` | Gated-mix speech (48 kHz default; only when `speech_safe=true`): the speech stem with attenuated original mix patched into separator-deleted spans, crossfaded (Design record §5) — additive, never replaces `audio_speech.wav` |
| `audiostem_sep.json` | Aggregate metrics: per-stem summary, timings, per-chunk summaries |
| `stem_activity.json` | Semantic timelines: activity spans per lane (`original` = non-silence of the mix; stems = energy ≥ threshold) + `speech_stem_gaps` lane (mix active ∧ speech stem < −55 dBFS — audio the stem does not cover, an upper bound on deleted speech; present when `speech_stem` was enabled) + `tones[]` measured on the original (`freq_hz` measured, `nominal_hz` = nearest of 400/1000/2000 within ±2%, `kind` steady/pop); knobs recorded in `params` |
| `stem_waveforms.json` | Rendering envelopes: peak-normalized per-bin RMS for original + active stems at `params.bin_ms` (50 ms default) — timeline waveform lanes without reading WAVs |
| `usage_tracking.json` | Per-state invocation counts, durations, estimated cost; `total_estimated_cost` covers both this pipeline and media-processing child (if run) |
| `segment_chunks/{stem}/chunk_NNN.wav` | Per-chunk intermediate stem WAVs |
| `segment_chunks/chunk_NNN.json` | Per-chunk metrics |
| `resampled/audio_48k_mono.wav` | WAV-normalise path output (WAV sources that needed resampling) |

The media-processing path's intermediate wav is **not** under this prefix: MP writes to its own service-derived run prefix (`micro-workflow/media-processing/{workflow_id}/audiostem_sep/audio_48k_mono.wav` in the proxy bucket) — house rule, callers never pass output prefixes.

## Key Conventions

### Source Format Handling
[states/check_and_preprocess/check_helper.py](states/check_and_preprocess/check_helper.py) probes the source. Three outcomes:

- **Fast-path** — WAV, exactly 48 kHz, mono: used as-is.
- **WAV-normalise path** — WAV but wrong sample rate or multi-channel: [states/check_and_preprocess/wav_to_wav.py](states/check_and_preprocess/wav_to_wav.py) converts in-Lambda. Output at `{output_prefix}/resampled/audio_48k_mono.wav`. Files ≤ 10 MB use a single `put_object`; larger files use concurrent multipart (up to 8 threads).
- **Media-processing path** — not a WAV: submitted to the media-processing GATEWAY (held task token). Output at `{MP's own run prefix}/audiostem_sep/audio_48k_mono.wav`.

### Chunk-Size Alignment
`fixed_chunk_size_secs` is rounded to the nearest multiple of 8 s (Bandit's internal chunk size):

```
aligned = ceil((requested - 4) / 8) * 8
```

Examples: 180 s → 176 s, 238 s → 240 s, 244 s → 240 s. Minimum is 8 s. The last chunk is clamped to file end — not padded.

### Model Loading
ONNX artefacts are baked into the container image at build time and extracted flat to `/opt/models/`: `encoder.int8.onnx`, `speech_decoder.int8.onnx`, `music_decoder.int8.onnx`, `sfx_decoder.int8.onnx`, `model_config.json`. Cold start loads sessions from `/opt/models/` with no S3 download. Warm container: module-level `_sessions` cache hit → < 1 ms.

### Split Encoder/Decoder
The model is split into one encoder and up to three decoder heads. The encoder runs once per 8 s chunk producing shared buffers `(q, x)`; each **active** decoder head consumes them. Disabling a stem flag (`speech_stem: false`) skips loading that decoder, skips its inference, and skips its upload — disabled stems cost nothing.

### WAV Utilities
[lib/wav_utils.py](lib/wav_utils.py) has no external media dependencies (no ffmpeg, libav, soundfile). Handles `WAVE_FORMAT_EXTENSIBLE`, non-standard chunk ordering, multi-channel downmix to mono. Header reads use 4 KB first; falls back to 64 KB. All S3 access uses the shared client from [lib/s3_utils.py](lib/s3_utils.py).

Key functions:
- `get_wav_info(bucket, key)` — byte-range header read; returns `WavInfo`
- `read_wav_window(bucket, key, info, start_sec, duration_sec, target_sr)` — single byte-range GET; returns float32 mono array resampled to `target_sr`
- `encode_wav_header_mono(n_samples, sample_rate)` — 44-byte RIFF header (takes sample count, not byte count)
- `encode_wav_mono(audio, sample_rate)` — complete in-memory WAV (header + PCM); use only for chunks that fit in RAM
- `rms_db(audio)` — RMS level in dBFS; returns -120.0 for silence
- `resample(audio, src_sr, tgt_sr)` — anti-aliased polyphase (scipy `resample_poly`; replaced a linear-interp resampler that aliased every 16 kHz speech / 32 kHz sfx stem ever shipped)

### S3 Utilities
[lib/s3_utils.py](lib/s3_utils.py) provides a single `boto3.client("s3")` with `max_pool_connections=60` shared by all callers via `from lib.s3_utils import s3`. `get_json` / `put_json` handle all JSON reads and writes. `upload_file`, `copy_object`, `delete_object`, `s3_uri`, `presign`, and `head_size` are defined but have no current callers.

### Multipart Concatenation
`StateConcatenateAudioChunksByStems` always uses multipart upload. Total PCM bytes are pre-calculated from iterator metadata (duration × sample_rate × 2), the WAV header is built upfront, then chunks stream as multipart parts. Peak memory is O(one chunk) regardless of total duration. Small chunks (< 5 MB PCM, the S3 per-part minimum) are accumulated into a `pending` buffer until threshold is met before uploading as one part — the final part is exempt from the 5 MB minimum.

### ORT CPU Fix
Lambda removes `/sys/devices/system/cpu/possible` and `/present`. The Dockerfile copies `patch.txt` ("0-5") to both paths. [lib/ort_init.py](lib/ort_init.py) creates those files at runtime if somehow absent. `get_num_cpus(memory_mb)` derives vCPU count from the Lambda memory allocation table (1792 MB → 1, 3584 MB → 2, … 10240 MB → 6). Call `ensure_cpuinfo_files()` before `import onnxruntime`.

### Usage Tracking and Child Execution Cost
`StateUsageTracking` reads `data.media_processing_usage_key` (written by `StateExtractProcessedAudio` on both handle shapes). If set, it `get_json`s media-processing's **own published** `usage_tracking.json` from that key and embeds `{execution_arn, pipeline_wall_sec, estimated_cost, states}` as the `media_processing` block — no walking of MP's execution history, no cross-machine IAM (that is what retired the `MediaProcessingStateMachineArn` parameter; Design record §2). MP's `states` entries follow MP's usage shape, embedded verbatim. A missing file after a SUCCEEDED handle is a pipeline bug and raises. `__init__.py` is an empty package marker.

## AgentCore Gateway Tool

[gateway_tool/lambda_function.py](gateway_tool/lambda_function.py) is the AgentCore Gateway dispatcher that wraps this workflow as three MCP tools. It is deployed as a separate Lambda (not a container — flat zip, no Docker).

### Tools exposed

**`separate_audio_stems`** — validates parameters, constructs the nested SFN input, derives the workflow_id, and calls `StartExecution`.

- All `input_config` / `output_config` fields from the workflow schema are exposed as flat top-level parameters.
- workflow_id strategy: `workflow_id = uuid5(canonical JSON of the SFN input WITHOUT output.s3uri + "|" + TOOL_VERSION + "|" + idempotency_token when one is supplied)` — derived by the shared [gateway_tool/workflow_identity.py](gateway_tool/workflow_identity.py), which documents the three ids. It NAMES the execution, so identical requests are idempotent: on `ExecutionAlreadyExists` the call is **absorbed** into the existing run — no second execution — while any parameter, token, or tool-version change re-runs to a fresh prefix. Outputs land at `s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{workflow_id}`; a caller-supplied `output_s3_prefix` is deprecated and ignored.
- `idempotency_token` (optional, 1-80 chars of `A-Z a-z 0-9 _ . -`) is the caller's assertion "this is the same request I sent before". It rides in the hash preimage as extra material and is then **discarded** — never forwarded to the state machine, never stored, never echoed. Omitting it still dedups on the parameters alone, since they name the asset and therefore determine the output; a token only distinguishes two otherwise-identical requests. The retired `uuid` parameter is **rejected** on submit (it used to mean both "which asset" and "same request", so a caller still sending it cannot be assumed to mean the surviving half) and **accepted with a log warning** on `get_job_status` / `describe_tool`, where neither key could have changed the answer.
- Returns a **HANDLE, never results** (Design record §1): `{"job_id": <execution_arn>, "status": "RUNNING", "current_state": <active stage>, "workflow_id": <derived id>, "output_s3_prefix": <derived prefix>, "asset_s3uri": <source s3uri>}`. An absorbed resubmit returns *the same fields* — only `status` differs (the existing run's, which may be terminal), plus `message` (present only when absorbed, and the only thing distinguishing the two cases — the `job_id` is a hash of the same input either way) and, on a terminal failure, `error` + `error_code` — compacted exactly like `get_job_status`'s (the stable class-name code as a field, never a raw stack-trace blob; Design record §3). `current_state` is dropped once the run has stopped, because a stopped execution is in no state and reading one costs a `GetExecutionHistory` paging pass for a `"unknown"`. Results come from `get_job_status` alone; that is what makes the declared `OutputSchema` equal to the reachable field set. No `uuid`, deliberately — the SFN input still carries a `uuid` key (set to the workflow_id) because every state derives its output prefix from it; that is this execution's local id, not the caller's token.

**`get_job_status`** — calls `DescribeExecution`; when RUNNING, pages `GetExecutionHistory` to find the active pipeline stage. When SUCCEEDED, returns structured fields including `workflow_id`, `output_s3_prefix`, per-stem output keys (`speech_key`, `music_key`, `sfx_key`, `speech_safe_key`), `asset_s3uri`, `total_estimated_cost`, `usage_key`, and a self-describing `outputs` list of `{uri, mime, label, desc}` entries (labels: `speech`, `music`, `sfx`, `speech-safe`, `metrics`, `stem-activity`, `stem-waveforms`, `usage`). `desc` is the one-line role description joined verbatim from [gateway_tool/output_contract.json](gateway_tool/output_contract.json) — the same file `describe_tool` serves in full, so the two surfaces cannot disagree; it is omitted, never fabricated, for a label the contract lacks. **Every branch** — RUNNING, SUCCEEDED, every failure, and the fallback for a status Step Functions has not shipped yet — carries `job_id`, `workflow_id`, `asset_s3uri`, and `output_s3_prefix`, so a status reply always names its own run and its output location even when there is nothing to report yet. (`output_s3_prefix` is read back from the execution's *input*, not its output, which is why a RUNNING or failed run still has one — partial results already live under it.) Terminal failures add `error` — `code: message` when the cause parsed as a Lambda error JSON, the raw cause (truncated 600) otherwise — and `error_code`: the stable code alone, i.e. the raised exception's class name (`NoAudioStream`) or, when the cause carried no parseable type, Step Functions' own error field (`Lambda.Unknown`, `States.Timeout`).

**`describe_tool`** — returns `{"version": <TOOL_VERSION>, "state_machine_arn": <ARN>, "description": <short backend description>, "output_contract": <parsed output_contract.json>}`. Version is stamped at deploy time via the `TOOL_VERSION` env var (set from the `.version` file by `build-deploy-s3-dist.sh`); the ARN comes from the state machine env var, giving callers a console/CLI entry point without parsing a job_id.

`output_contract` (~7 KB) is the field-level contract for every `outputs[]` label: basename, kind (`file`/`prefix`), mime, the one-line `description` that doubles as the entries' `desc`, and a compact single-element `example` of the file's contents (`contents` block for the binary stem WAVs instead — format, default/configurable sample rate). It is static per `TOOL_VERSION` — the file ships in the gateway zip (`build-deploy-s3-dist.sh` pins the member), so version and contract cannot drift — which is why it lives here and not in `get_job_status`: a caller caches it once per session, and CI can pin it without running a job. Contract keys == emitted labels, exactly; anything not listed (per-chunk intermediates under `segment_chunks/`, the `resampled/` and `media_processing/` conversion intermediates) is internal and may change without notice. The examples were authored from the producing code, not the docs — note `audiostem_sep.json`'s `output_keys`/`stem_summary`/`chunks[].stems` are keyed per *active* stem, and `usage_tracking.json`'s `media_processing` block is null unless the conversion path ran.

### Tool routing

AgentCore passes `context.client_context.custom["bedrockAgentCoreToolName"]` as `"{target_name}___{tool_name}"`. `_resolve_tool()` strips the target prefix and returns the bare tool name.

### Agent system prompt

The companion system prompt for a Strands agent using this target lives in the family gateway repo at [media-toolkit-gateway/prompts/audio_stem_agent.md](../media-toolkit-gateway/prompts/audio_stem_agent.md).

---

## CloudFormation Stack

[cfnstack.yaml](cfnstack.yaml) is a self-contained standalone stack. The custom resource Lambda is inlined.

### Resources created

| Resource | Name / Value |
|----------|-------------|
| ECR Repository | `audiostem-separation-on-aws` (retained on stack delete) |
| CodeBuild Project | `${ResourcePrefix}-audiostem-separation-on-aws` |
| Start-build Lambda | `${ResourcePrefix}-audiostem-separation-start-build` (inlined Python) |
| Lambda functions (×7) | `${ResourcePrefix}-audiostem-separation-{variant}` |
| Step Functions state machine | `${ResourcePrefix}-audiostem-separation` |

### Lambda variants

| Variant | Memory | vCPUs | States |
|---------|--------|-------|--------|
| `half-core` | 896 MB | 0.5 | StateFixedChunkingForAudioStemSeparation, StateUsageTracking |
| `1-core` | 1792 MB | 1 | StateConcatenateAudioChunksByStems |
| `2-cores` | 3584 MB | 2 | StateCheckAndPreprocess, StateExtractProcessedAudio |
| `3-cores` | 5376 MB | 3 | — |
| `4-cores` | 7168 MB | 4 | — |
| `5-cores` | 8960 MB | 5 | — |
| `6-cores` | 10240 MB | 6 | StateRunBanditV2Model (Map state inference, default) |
| `gateway-tool` | 256 MB | — | AgentCore Gateway dispatcher (Python 3.13, arm64, no container) |

### Stack parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `ResourcePrefix` | — | Prefix applied to all resource names |
| `IngestBucket` | — | S3 bucket containing source media files (read-only) |
| `ProxyBucket` | — | S3 bucket Lambda functions read/write audio and outputs to |
| `MediaProcessingGatewayLambdaArn` | — | ARN of media-processing's gateway tool Lambda; `Run media processing` submits through it (held task token). There is deliberately NO `MediaProcessingStateMachineArn` parameter: MP's cost is read from its published `usage_tracking.json`, so no cross-machine IAM remains (Design record §2) |

### Stack outputs

| Output | Purpose |
|--------|---------|
| `GatewayToolLambdaArn` | Imported by `media-toolkit-gateway/cfnstack.yaml` as `AudioStemToolLambdaArn` |

### ECR image build and skip logic
On every stack create or update a Lambda custom resource starts a CodeBuild job. `pre_build` runs `aws ecr describe-images` — if the image tag already exists it writes `/tmp/skip_build` and the `build` phase is skipped. `post_build` always signals CloudFormation via `curl` to the pre-signed `CFN_ResponseURL`. Stack updates are fast (~30 s) when the image version has not changed.

## Build & Deploy

[build-deploy-s3-dist.sh](build-deploy-s3-dist.sh) packages and deploys the stack in one step. Run from this workflow's folder:

```bash
bash ./build-deploy-s3-dist.sh --bucket MY_BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--bucket` | _(required)_ | Deployment bucket basename |
| `--solution` | `unified-media-toolkit` | S3 key prefix namespace |
| `--version` | `.version` file | Overrides the version string |
| `--single-region` | off | Deploy to `--bucket` as-is; otherwise appends region suffix |
| `--acl` | `bucket-owner-full-control` | S3 object ACL |

Produces three artefacts under `dist/`:
- `regional-s3-assets/audiostem-separation-on-aws-{VERSION}.zip` — Docker source context (uploaded to S3; used by CodeBuild)
- `regional-s3-assets/audiostem-separation-gateway-tool-{VERSION}.zip` — flat zip of the pinned `gateway_tool/` members (dispatcher, `bucket_validator.py`, `workflow_identity.py`, `output_contract.json`), deployed directly to the gateway tool Lambda
- `global-s3-assets/audiostem-separation-stack.template` — processed CloudFormation template with all `%%TOKEN%%` substitutions applied

## Tests

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

Tests gate packaging — `build-deploy-s3-dist.sh` refuses to package on a failing suite.

## Gotchas

- **The speech stem is a FOREGROUND-DIALOGUE product — it deletes low-prominence speech, and no inference-time knob changes that (measured).** Sentences present in the mix transcript but absent from the stem transcript split into two classes: low-prominence speech — whispers, phone-futzed voices, background talkers, dialogue shouted over loud music — where the stem is near-silent, and timing artifacts where the energy survived. On labeled low-prominence sentences the speech-stem/mix RMS survival ratio was zero in **every** cell of a {int8, fp32} × {raw, +gain} matrix while foreground controls survived essentially intact: the model *semantically* classifies that audio as not-speech regardless of level or precision — the bias is in the weights. Three consequences: (1) **never feed the speech stem to transcription or diarization** — a mix-vs-stem A/B against human-validated speaker labels showed clustering retention collapse on the stem, with losses concentrated exactly on the quiet side of conversations; (2) word timing corrupts near deletions — forced alignment has nothing to lock onto in hollowed-out regions, producing sentence starts that drift tens of seconds (ends stay correct); (3) the shipped remediation is the opt-in `speech_safe` output (gated mix substitution, Design record §5) plus the `speech_stem_gaps` lane that makes the stem's coverage gaps visible on every run; the only model-side fix would be fine-tuning with prominence-augmented targets — input AGC is a no-op (the model normalizes internally). Corollary worth keeping: **INT8 quantization costs nothing** — int8 equals fp32 to three decimals across every class, so the Lambda-friendly quantized graphs are free.
- **Usage tracking must bill failed attempts.** An earlier version of `states/usage_tracking/lambda_function.py` dropped any invocation with `succeeded_ts is None`, so a `TaskFailed` — a Lambda killed at its timeout, the most expensive invocation possible — contributed 0 s while `invocations` stayed correct, so the output read as precise (a real run under-reported severalfold). The `failed` flag it already recorded was never read; only the timestamp was missing. **Two invariants to preserve:** `_billed_sec` falls back to `failed_ts`, and 0.0 is correct **only** when an attempt has NO terminal event at all — that case is what `incomplete` means. `wall_sec` sums attempts per `entered_id` *before* taking the max across occurrences, because every retry attempt chains back to the same `TaskStateEntered` while concurrent Map iterators each get their own. `tts-on-aws` is the family reference copy; pinned cross-stack by `micro-workflow-bundle/test/test_units.py: TestUsageTrackingBillsFailedAttempts` — a regression fails the bundle suite, not this stack's.
- **media-processing must be deployed first, and at callback-capable version** — `cfnstack.yaml` requires `MediaProcessingGatewayLambdaArn`, and the submit handler refuses (before spending anything) when MP's `describe_tool.features` lacks `callback_task_token`. The inline `DefinitionString` in `cfnstack.yaml` is what deploys; `definitions.json` is the reference copy — `test/test_gateway_callback.py::TestAslParity` pins whole-document equality, so edit BOTH or the suite fails.
- **Disabling a stem skips its decoder and upload** — `speech_stem: false` means that decoder is not loaded, not run, and no WAV is uploaded. Only active stems incur inference cost. One exception: `speech_safe: true` runs the speech decoder even when `speech_stem` is false (the patch is built from its output); the raw stem WAV is then still not uploaded.
- **Minimum effective chunk size is 8 s** — values below 8 are rounded up to 8.
- **Cold start is ~1–2 s** (model baked into `/opt/models/`, no download). Warm container inference: ~13–20 s per chunk on 6-vCPU Lambda.
- **Usage tracking cost estimates are slightly high** (~10–50 ms systematic overestimate) because Step Functions scheduling latency is included in the billed-duration timestamps.
- **Lambda pricing constants** in [states/usage_tracking/lambda_function.py](states/usage_tracking/lambda_function.py) are us-east-1 figures; update for other regions or when pricing changes.

## Dead Code

| Item | Location | Notes |
|------|----------|-------|
| `lib/s3_utils.py` | `s3_uri`, `presign`, `head_size`, `upload_file`, `copy_object`, `delete_object` | No callers; utility stubs for future use |
| `lib/model_store.py` | `ensure_in_tmp`, `ensure_in_tmp_with_fallback`, `ensure_in_s3`, `cleanup_models` | Not called at inference time — model is baked into `/opt/models/` |

## Model Export

[export/export_bandit_v2.py](export/export_bandit_v2.py) converts Bandit PyTorch checkpoints to split ONNX INT8 graphs (offline, run outside the container). [export/test_bandit_v2.py](export/test_bandit_v2.py) validates the exported ONNX graphs with shape, energy, and parity checks.

## Design record

Numbered decisions cited from code comments (`README.md Design record §N`). Each carries the decision and its why; the measured evidence behind them lives in the project's internal records.

1. **Submit returns a HANDLE, never results.** The gateway `separate_audio_stems` reply is a handle (`job_id`, `workflow_id`, `output_s3_prefix`, `status`, …); all results come from `get_job_status`. This keeps the declared `OutputSchema` equal to the reachable field set and makes an absorbed resubmit indistinguishable from a fresh submit except for `message`.
2. **Media-processing is a gateway sibling behind a held task token — no cross-machine IAM.** The direct `startExecution.sync:2` of MP's state machine was replaced by submitting through MP's gateway Lambda with this state's task token; MP's callback waker completes the token with the handle + terminal status AS DATA. MP's cost is read from its own published `usage_tracking.json`, which is what removed the `MediaProcessingStateMachineArn` parameter and every `states:*`/`events:*` cross-machine grant.
3. **Failures are enforced BY NAME.** A child/sibling verdict is data; the consumer enforces it, raising an exception whose class name is the stable error code in the execution's error field (`NoAudioStream` for MP's `NoCompatibleOutputGroups` refusal), and passing MP's own error/cause through instead of an opaque pointer at the run prefix. The gateway status surfaces the same code as `error_code`.
4. **Audio-analysis products ride the concatenation state; audiostem measures, the fusion layer names.** `stem_activity.json` (activity spans + line-up tones measured on the ORIGINAL mix — which stem a pure sine lands in after separation is a model artifact) and `stem_waveforms.json` (peak-normalized envelopes) are produced in `StateConcatenateAudioChunksByStems`, where the PCM is already in memory. There is deliberately NO bars_and_tone verdict here: audiostem reports the measured tone; naming what a span MEANS (e.g. bars-and-tone) is the downstream fusion layer's job, made by joining `tones[]` against visual slate/bars evidence. Every analysis knob is an ENV constant (never a per-request parameter — it would enter the workflow_id preimage) and is recorded in the product's `params` block.
5. **speech_safe is gated mix SUBSTITUTION, additive and opt-in; the speech_stem_gaps lane ships regardless.** The separation model deletes low-prominence speech outright — a training-distribution property (its training data defines "dialogue" as loud, clean read speech), measured: deleted spans are digital zero at the decoder output, and no inference-time knob (gain, AGC, precision, VAD tuning) recovers them. Rather than fix the model, the product guarantee is "never silence where the source had sound": where the raw stem is near-silent (< −55 dBFS per 50 ms bin) while the mix carries energy, attenuated mix is ADDED under the stem with crossfaded edges, built in the Map state where mix and stem coexist sample-aligned pre-resample (zero extra inference/IO). The shape has published support: observation adding provably bounds artifact damage, and remixing attenuated background under separated dialogue measures better than either the raw mix or the bare stem — production dialogue-enhancement systems ship exactly this, never a bare stem. Patched spans are mix-flavored, not separated, which is why (a) `audio_speech_safe.wav` is a NEW opt-in output and `audio_speech.wav` is untouched, and (b) `stem_activity.json` carries the `speech_stem_gaps` lane on every run where `speech_stem` was enabled (the safe lane is patched, so the lane needs the raw stem envelope) — computed at concatenate time from the absolute linear envelopes (the delivered waveform JSON is peak-normalized, so this comparison exists only there), making the stem's losses visible without opting into the patched audio. A gap is measurement, not verdict: mix-active ∧ stem-silent includes music/sfx-only passages (an upper bound on deleted speech), and naming what a gap means is the fusion layer's job. Gate knobs are ENV constants recorded in the chunk metrics' `params` (per §4's rule); the gate mask is edge-extended before crossfade smoothing so a patch crossing a chunk boundary holds full level at the seam. speech_safe is consumed by playback and mixing (AD mixes, review/QC playback, dialogue-forward renders) and defaults to 48 kHz accordingly; as an ASR input the mix measured better pooled and stays the validated default, but the seat is content-class-sensitive — speech_safe measured ahead of the mix on music-heavy episodes (sung-lyric insertion suppression) while recovering most of the bare stem's deleted words, so it is a measured alternative to compare, not a prohibited one.
