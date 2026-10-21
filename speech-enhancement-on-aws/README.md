# speech-enhancement-on-aws

AWS Lambda container for speech enhancement using the DeepFilterNet3 ONNX model (runtime-only re-engineering — no cargo/rust; upstream: github.com/rikorose/deepfilternet). Noise-suppresses the dialogue in any media file's audio **in place** — one enhanced WAV out, no stem extraction, everything keeps its timeline position. Accepts any media file; non-WAV sources (and WAVs that are not 48 kHz mono) are converted before enhancement.

**Positioning vs audiostem-separation:** audiostem's speech stem *extracts* (and semantically deletes low-prominence speech); this workflow *attenuates noise around* the dialogue. It is a listening-quality product — **measured and REFUTED as an ASR pre-processor, and never measured as a diarization one** (on a multi-episode caption ground-truth corpus the enhanced track lost to the original mix on WER, the mix winning every episode; diarization pre-processing remains unvalidated — see the gotcha; the describe_tool description states the verdict).

## Workflow States

Defined in [definitions.json](definitions.json). Entry point for every state: [lambda_function.py](lambda_function.py) dispatches on the `operation` field.

```
StateCheckAndPreprocess
  ├─ (needs_conversion=false) ────────────────────────────────────────────┐
  └─ (needs_conversion=true)                                               │
       └─ Run media processing  (lambda:invoke.waitForTaskToken)          │
            └─ StateExtractProcessedAudio                                  │
                 └─ StateFixedChunkingForSpeechEnhancement  ←──────────────┘
                      └─ Map: StateRunSpeechEnhancementModel  (MaxConcurrency=40)
                           └─ StateConcatenateEnhancedAudio
                                └─ StateUsageTracking
```

**State responsibilities** (the preprocess → MP-submit → extract → usage spine is the audiostem family pattern, adapted):

1. **StateCheckAndPreprocess** — Validates via `input_schema`, probes the source, routes: **fast-path** (WAV, 48 kHz, mono — used as-is), **WAV-normalise path** ([states/check_and_preprocess/wav_to_wav.py](states/check_and_preprocess/wav_to_wav.py) resamples in-Lambda to `{output_prefix}/resampled/audio_48k_mono.wav`), or **media-processing path** (non-WAV → `data.media_processing_request`, `needs_conversion=true`). Flattens config into `data`: `output_resampl_freq`, `atten_lim_db`, `fixed_chunk_size_secs`, `model_id`, buckets/prefixes.
2. **Run media processing** — HELD `waitForTaskToken` task. [states/submit_media_processing/lambda_function.py](states/submit_media_processing/lambda_function.py) probes MP's `describe_tool` for `callback_task_token` (refuse-before-spend), submits `process_media` with `idempotency_token=speech-enhancement-{uuid}`. MP's callback waker completes the token with the handle + terminal `status` AS DATA; enforcing the verdict is the consumer's job. MP is a HARD dependency — submit-path failures raise, never self-skip.
3. **StateExtractProcessedAudio** — Enforces handle `status == "SUCCEEDED"` — a FAILED verdict is enforced BY NAME (the family fail-by-name contract): MP refusing the wav group as `NoCompatibleOutputGroups` raises [`lib.exceptions.NoAudioStream`](lib/exceptions.py) carrying MP's own message (the class name is the stable code in the execution's error field, matched across both completion shapes plus the legacy plain-ValueError wording), and every other failure re-raises with MP's forwarded error/cause passed through instead of an opaque "see the run prefix". Derives the wav key from MP's output contract — `{output_s3_prefix}/speech_enhancement/audio_48k_mono.wav` — probes the WAV header, writes `data.audio_bucket / audio_key / audio_cfg / media_processing_arn / media_processing_usage_key`, clears the request/output fields.
4. **StateFixedChunkingForSpeechEnhancement** — Aligns `fixed_chunk_size_secs` to the nearest multiple of **10 s** (ties down, floor 10 — a whole multiple of DFN3's STFT hop), emits `data.iterators`. Iterator `output_prefix` is `{output_prefix}/enhanced_chunks` — intermediates stay out of the run root. Boundaries decide parallelism only, not audio quality (seams are the model state's job).
5. **StateRunSpeechEnhancementModel** — Map state. Each iterator fetches its window **±1 s context** (`_CONTEXT_LEFT_SEC`/`_CONTEXT_RIGHT_SEC`) via byte-range GET conformed to 48 kHz, runs DFN3 ([states/run_speech_enhancement_model/model_inference.py](states/run_speech_enhancement_model/model_inference.py) — pure numpy + onnxruntime: sr 48000, fft 960, hop 480, nb_erb 32, nb_df 96, df_order 5, lookahead 2), trims back to the exact window, resamples to `output_resampl_freq` (family polyphase resampler), uploads `chunk_NNN.wav` + `chunk_NNN.json`. `atten_lim_db` caps attenuation via a full-band post-mix of the original over both stages (~0 = bypass, >=100 = unlimited, null = full suppression). Logging is keys-only — values can carry customer-named S3 keys.
6. **StateConcatenateEnhancedAudio** — Streams chunk PCM in order into ONE S3 multipart upload (header pre-calculated from iterator metadata, 5 MB pending-buffer parts, abort on error, O(one chunk) memory). Concatenation is byte-plain — context padding upstream already settled seams. Writes `audio_enhanced.wav` + aggregate `speech_enhancement.json`; `data` gains `enhancement_output {enhanced_key, resampl_freq, rms_db_in, rms_db_out, duration_sec, size_bytes, metrics_key}`.
7. **StateUsageTracking** — Walks the execution history for per-state cost; adds MP's published `usage_tracking.json` cost when `data.media_processing_usage_key` is set. Writes `{output_prefix}/usage_tracking.json`.

## Input Schema

[lib/input_schema.py](lib/input_schema.py). All `input_config` / `output_config` fields optional — defaults merged by `validate_and_normalize()`.

```json
{
  "uuid": "<UUID>",
  "input": {
    "s3uri": "s3://bucket/path/to/source.mp4",
    "input_config": { "model_id": "deepfilternet3", "fixed_chunk_size_secs": 360 }
  },
  "output": {
    "s3uri": "s3://bucket/output/prefix",
    "output_config": { "output_resampl_freq": 16000, "atten_lim_db": null }
  },
  "data": {}
}
```

`model_id` is fixed to `"deepfilternet3"`. `fixed_chunk_size_secs` minimum 10 s, default 360 s. Valid `output_resampl_freq`: 8000, 16000, 32000, 44100, 48000, 96000 (default 16000). `atten_lim_db`: number in [0, 100] or null; bools rejected.

## Output Files (S3)

All written under `{output_prefix}/`:

| File | Contents |
|------|----------|
| `audio_enhanced.wav` | Concatenated enhanced audio (16 kHz mono default) |
| `speech_enhancement.json` | Aggregate metrics: output summary (RMS in/out — the drop is the noise removed), timings, per-chunk summaries |
| `usage_tracking.json` | Per-state cost; includes media-processing child when conversion ran |
| `enhanced_chunks/chunk_NNN.wav` / `.json` | Per-chunk intermediates (internal, not contract) |
| `resampled/audio_48k_mono.wav` | WAV-normalise path intermediate |

MP-path intermediate wav is NOT under this prefix — it lives under MP's own run prefix (`…/speech_enhancement/audio_48k_mono.wav`), house rule.

## Key Conventions

- **Model loading** — ONNX artefacts (`enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`, `config.ini`) are baked into the image at `/opt/models/`, **sha256-pinned in the Dockerfile** (`DFN3_MODEL_SHA256`). Provenance verified against the upstream Rikorose checkpoint (re-exported; all 55 tensors match on size/mean/std, config.ini byte-identical) → MIT OR Apache-2.0. Warm container: module-level singleton. An ETag-checked S3 model_store from the legacy implementation is deliberately retired (`lib/model_store.py` keeps only `find_file`).
- **Seam handling** — ±1 s context padding in the model state, trimmed before upload; concat is plain. S3 range reads beyond EOF return available bytes silently, so edge chunks need no special case.
- **Resampling** — `lib/wav_utils.resample` is anti-aliased polyphase (scipy `resample_poly`). The legacy linear-interp resampler was NOT ported — that is the aliasing bug class fixed family-wide.
- **WAV utilities / S3 utilities / ORT cpuinfo fix / multipart concat** — identical family modules to audiostem ([lib/wav_utils.py](lib/wav_utils.py), [lib/s3_utils.py](lib/s3_utils.py), [lib/ort_init.py](lib/ort_init.py)); see audiostem-separation-on-aws's README for the mechanics.

## AgentCore Gateway Tool

[gateway_tool/lambda_function.py](gateway_tool/lambda_function.py) — flat-zip dispatcher, three MCP tools:

- **`enhance_speech`** — params: `asset_s3uri` (required), `fixed_chunk_size_secs` (≥10), `output_resampl_freq`, `atten_lim_db` (0–100 or null), `idempotency_token`. Family contract: `workflow_id` = uuid5 of the request ([gateway_tool/workflow_identity.py](gateway_tool/workflow_identity.py)), idempotent resubmit absorbs into the existing run, returns a HANDLE never results, outputs land at `s3://{PROXY_BUCKET}/micro-workflow/speech-enhancement/{workflow_id}`.
- **`get_job_status`** — on SUCCEEDED returns `enhanced_key`, `metrics_key`, RMS in/out, `duration_sec`, cost, and `outputs[]` with labels **`enhanced` / `metrics` / `usage`** (descs joined from [gateway_tool/output_contract.json](gateway_tool/output_contract.json); pinned by `test_units.py::TestOutputContractPin`). Terminal failures add `error` — `code: message` when the cause parsed as a Lambda error JSON, the raw cause (truncated 600) otherwise — and `error_code`: the stable code alone, i.e. the raised exception's class name (`NoAudioStream`) or, when the cause carried no parseable type, Step Functions' own error field (`Lambda.Unknown`).
- **`describe_tool`** — version, state machine ARN, description (carries the measured mix-first verdict — enhanced not recommended as an ASR input today), `output_contract`, `input_examples`.

## CloudFormation Stack

[cfnstack.yaml](cfnstack.yaml), self-contained. ECR repo `speech-enhancement-on-aws`; Lambda functions `${ResourcePrefix}-speech-enhancement-{variant}`; state machine `${ResourcePrefix}-speech-enhancement`.

**Lambda tier routing:**

| Variant | Memory | States |
|---------|--------|--------|
| `half-core` | 896 MB | SubmitMediaProcessing, ExtractProcessedAudio, FixedChunking, UsageTracking |
| `1-core` | 1792 MB | ConcatenateEnhancedAudio |
| `2-cores` | 3584 MB | CheckAndPreprocess |
| `5-cores` | 8960 MB | RunSpeechEnhancementModel (Map inference) |
| `gateway-tool` | 256 MB | AgentCore dispatcher (Python 3.13, arm64, flat zip) |

Parameters: `ResourcePrefix`, `IngestBucket`, `ProxyBucket`, `MediaProcessingGatewayLambdaArn` (MP must be deployed first at callback-capable version — the submit handler refuses before spending when `describe_tool.features` lacks `callback_task_token`). Output `GatewayToolLambdaArn` is imported by `media-toolkit-gateway/cfnstack.yaml` as `SpeechEnhancementToolLambdaArn`.

## Build & Deploy

[build-deploy-s3-dist.sh](build-deploy-s3-dist.sh) — same flags as the family. Produces `speech-enhancement-on-aws-{VERSION}.zip` (CodeBuild context), `speech-enhancement-gateway-tool-{VERSION}.zip` (pinned members: dispatcher, `bucket_validator.py`, `workflow_identity.py`, `output_contract.json`, `input_examples.json`), and `speech-enhancement-stack.template`.

Tests gate packaging:

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

## Gotchas

- **The inline `DefinitionString` in cfnstack.yaml is what deploys; definitions.json is the reference copy** — `test/test_gateway_callback.py::TestAslParity` pins whole-document equality; edit BOTH or the suite fails.
- **Never claim ASR benefit — the transcription gate RAN and refuted it** (multi-episode caption ground-truth corpus at the tool's default/uncapped suppression: the enhanced track lost to the original mix on WER, the mix winning every episode). The clustering-retention and RMS-survival halves of the original gate design have NOT been run for DFN3, and no capped-suppression configuration has been measured — so diarization benefit is UNVALIDATED (never measured), not refuted; absent evidence, the recommendation is the same: the mix is the default; feeding the enhanced track into speech analysis is not recommended until a re-measurement says otherwise. Audiostem's speech stem taught the family that a model can *semantically delete* quiet speech while sounding fine on foreground — judge any future re-measurement POOLED (single-episode recall swings several points at identical code).
- **The mix is the validated transcription input** family-wide — the gate measured every alternative tried so far (speech stem, enhanced track) losing to it; alternatives are not recommended until a re-measurement says otherwise.
- **Minimum effective chunk size is 10 s**; values are aligned to the nearest 10 s multiple, ties down (360 stays 360; 355 → 350; 356 → 360).
- **`atten_lim_db` is a cap, not a gain** — enforced as the upstream post-mix `lim·noisy + (1−lim)·enhanced` AFTER both stages, so the guarantee is full-band; < 0.01 dB = full bypass, ≥ 100 dB = "no limit". (An earlier implementation used a mask floor that the DF stage could exceed.)
- **Usage tracking invariants** (billed-failed-attempts fix, `wall_sec` per-`entered_id` summation) are the family's — see the audiostem-separation-on-aws README gotcha; pinned cross-stack by `micro-workflow-bundle/test/test_units.py`.
- **Lambda pricing constants** in [states/usage_tracking/lambda_function.py](states/usage_tracking/lambda_function.py) are us-east-1 figures.

## Dead Code

| Item | Location | Notes |
|------|----------|-------|
| `lib/s3_utils.py` | `s3_uri`, `presign`, `head_size`, `upload_file`, `copy_object`, `delete_object` | No callers; family utility stubs |
| `lib/model_store.py` | S3/tmp ensure functions | Model is baked; only `find_file` is called |
