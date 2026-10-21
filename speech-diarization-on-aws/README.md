# speech-diarization-on-aws

AWS Lambda container for speech diarization. Accepts any media file (video, audio, or WAV), optionally converts it via a media-processing child execution, then produces word-level transcripts with speaker labels, WebVTT captions, and (optionally) per-speaker audio channels.

## Model Stack

| Model | Format | Purpose |
|-------|--------|---------|
| Whisper large-v3-turbo (or `cy`/`ar` variant) | CTranslate2 INT8 | ASR |
| Pyannote community-1 | ONNX | Voice activity detection (also loaded by StateGlobalSpeakersClustering for the splice binder) |
| WeSpeaker ResNet34 | ONNX | Speaker embedding (256-dim) |
| wav2vec2 XLSR-53 | ONNX INT8 | Forced alignment (27 languages) |
| wtpsplit SaT | ONNX | Sentence segmentation (85+ languages) |

The four baseline models (Pyannote, Whisper turbo, alignment/en, wtpsplit) are seeded to `s3://{ProxyBucket}/_model_artefacts/` at CFN stack creation time by CodeBuild. Additional language-specific alignment and Whisper models are downloaded on-demand by `StateDetectLanguageDownloadModels` at runtime (S3 first, CloudFront fallback).

## Workflow States

Defined in [definitions.json](definitions.json). Entry point for every state: [lambda_function.py](lambda_function.py) dispatches on the `operation` field.

```
StateCheckAndPreprocess
  ├─ (needs_conversion=false) ──────────────────────────────────────────────────┐
  └─ (needs_conversion=true)                                                     │
       └─ Run media processing  (lambda:invoke.waitForTaskToken)                 │
            └─ StateExtractProcessedAudio                                        │
                 └─ StateFixedChunkingForVAD  ←──────────────────────────────────┘
                      └─ Map: StateRunVADModel  (MaxConcurrency=40, 2-core Lambda)
                           └─ StateDynamicChunkingBySpeechTurns
                                └─ StateDetectLanguageDownloadModels  (3-core)
                                     └─ Map: StateRunWhisperAlignWtpsplitModels  (4-core)
                                              + StateRunSpeechEmbeddingModel     (2-core, chained within Map)
                                                   └─ StateGlobalSpeakersClustering  (1-core)
                                                        └─ StateMergeSpeakersTranscripts
                                                             └─ StateSeparateSpeakerChannels  (2-core)
                                                                  └─ StateUsageTracking
```

**State responsibilities:**

1. **StateCheckAndPreprocess** — Validates the event via `input_schema.validate_and_normalize()`, probes the source file, and routes to one of three paths. Writes all config sections flat into `data` (`model_cfg`, `vad_cfg`, `asr_cfg`, `embed_cfg`, `clu_cfg`, `output_bucket`, `output_prefix`, `speaker_channels`, `channel_mode`). Snapshots the full `workflow_params` to `{output_prefix}/workflow_params.json`.
2. **Run media processing** — HELD `lambda:invoke.waitForTaskToken` task (the family sibling pattern — replaced a direct `startExecution.sync:2` of MP's state machine). [states/submit_media_processing/lambda_function.py](states/submit_media_processing/lambda_function.py) probes MP's `describe_tool` for the `callback_task_token` feature (refuse-before-spend), then submits `process_media` with `idempotency_token=diar-{uuid}` (a redrive absorbs into the same MP run) and this state's task token; `use_partial`/`mark_in_out` ride as flat gateway params. MP's callback waker completes the token with the submit handle + terminal `status` AS DATA — always `SendTaskSuccess`; enforcing the verdict is the consumer's job. MP is a HARD dependency, so every submit-path failure raises.
3. **StateExtractProcessedAudio** — Reads `data.media_processing_output`. Gateway handle (has `output_s3_prefix`): enforces `status == "SUCCEEDED"` and derives the wav key from MP's output contract. A source with **no audio stream** fails BY NAME (decided over a succeed-with-empty-transcript alternative that was built and reverted — the agent probes first and handles the named condition): MP's probe refuses the wav group as `NoCompatibleOutputGroups`, its waker forwards `error`/`cause`, and this state raises [`lib.exceptions.NoAudioStream`](lib/exceptions.py) carrying MP's own message — the class name is the stable code in the execution's error field, matched across BOTH completion shapes (waker and absorbed-synchronous) plus the legacy plain-ValueError wording. Every OTHER failure raises **with the forwarded error/cause passed through** (the earlier behavior reported only "see the run prefix" over an empty prefix — a measured incident). The sibling content fact: zero usable speech (music-only input) raises `NoSpeechDetected` at `StateGlobalSpeakersClustering` with the embedding-count evidence, replacing an opaque "No high-quality embeddings found". The wav key is `{output_s3_prefix}/diarization_audio/audio_16k_mono.wav`; `media_processing_arn` is the handle's `job_id`. A non-handle shape raises (the legacy nested-execution shape was retired — pre-conversion FAILED runs cannot be redriven anyway, their old ASL's direct-start IAM is gone; resubmit through the gateway instead). Writes `data.audio_bucket / audio_key / audio_cfg / media_processing_arn / media_processing_usage_key`, clears `media_processing_request` and `media_processing_output`.
4. **StateFixedChunkingForVAD** — Reads WAV header; emits fixed chunks (default 720 s, 30 s overlap) as `data.iterators`.
5. **StateRunVADModel** — Map state. Each iterator runs Pyannote sliding-window segmentation; returns speech turns and writes `segments_key` per chunk.
6. **StateDynamicChunkingBySpeechTurns** — Flattens all VAD turns, forms speech groups (adjacent turns with gap < `speech_group_max_gap_sec`), partitions into ≤360 s transcription iterators. Replaces `data.iterators` with the new dynamic-chunk iterator set.
7. **StateDetectLanguageDownloadModels** — Majority-vote language detection across up to `language_detection_num_probes` dynamic chunks; resolves Whisper/alignment model variant; ensures all required models are present in S3. Enriches every iterator with `language`, `asr_cfg`, and `model_cfg`.
8. **StateRunWhisperAlignWtpsplitModels** — Runs Whisper on batched speech groups (28 s encoder windows), forced alignment on low-confidence words, iterator-level wtpsplit sentence segmentation. Writes `transcript_key` per iterator.
9. **StateRunSpeechEmbeddingModel** — Chained within the same Map, receives the Whisper output directly. Extracts WeSpeaker ResNet34 embeddings per sentence (HIGH ≥ 0.75 s, LOW 0.30–0.75 s, SKIPPED < 0.30 s). Writes `embeddings_key` per iterator.
10. **StateGlobalSpeakersClustering** — AHC clustering on HIGH embeddings with **cross-talk cannot-link constraints** (Design record P3, **default OFF** — the span→sentence mapping is imprecise and hard links regressed DER: [lib/constraints.py](lib/constraints.py) maps the VAD's channel-resolved overlap evidence to sentence pairs; linked pairs get distance 1e9 in the LINKAGE matrix only, so proven-distinct voices merge LAST — not never: a user `num_speakers` override or a low sweep K can still cut below the constraints, counted report-only in `cannot_link_violations`; silhouette/CH scoring stays on honest distances; supplied constraints suppress the single-speaker fast-path and are respected by the small-cluster mop-up); dual K-selection (elbow + combined scoring, guardrailed — Design record P1); outlier removal then locality-constrained reassignment; assigns LOW via local match-or-UNKNOWN ([lib/assign.py](lib/assign.py), Design record P5); SKIPPED via diarization overlap (midpoint fallback window-capped); ALWAYS emits identity-resolved two-speaker `overlap_segments` (additive output; identity precision stated in the contract — 81%, zh 56-69% — consumers decide trust, per the house evidence-not-verdict rule).
11. **StateMergeSpeakersTranscripts** — Merges per-iterator transcripts with speaker labels; builds `final_transcript.json` (including `overlap_spans`, the cross-talk ranges collected from the consolidated `dynamic_segments_{n}.json` turn flags — auxiliary: a missing doc degrades with a warning, never fails the merge) and `final_transcript.vtt`. Clears `data.iterators`.
12. **StateSeparateSpeakerChannels** — Writes one WAV per speaker (aligned or compact mode). Reads hidden config (`merge_gap_sec`, `fade_ms`, `compact_gap_sec`) from `event.output.output_config` via `.get()` with module-level defaults.
13. **StateUsageTracking** — Walks the parent execution history to compute per-state cost. Reads `data.media_processing_usage_key` — if set, reads MP's own published `usage_tracking.json` from that key and adds its `total_estimated_cost` to this run's.

## Input Schema

Defined and validated by [lib/input_schema.py](lib/input_schema.py). All `input_config` and `output_config` fields are optional — defaults are applied by `validate_and_normalize()`.

```json
{
  "uuid": "<UUID>",
  "input": {
    "s3uri": "s3://bucket/path/to/source.mp4",
    "input_config": {
      "language": null,
      "use_partial": false,
      "mark_in_out": [0, 0],
      "vad": {
        "fixed_chunk_size_secs": 720,
        "dynamic_speech_chunk_secs": 360,
        "min_speech_duration_sec": 0.0,
        "min_silence_duration_sec": 0.3,
        "speech_group_max_gap_sec": 1.0
      },
      "asr": {
        "model_variant": "turbo",
        "num_workers": 1,
        "beam_size": 2,
        "initial_prompt": "",
        "merge_gap_threshold_sec": 0.20,
        "language_detection_num_probes": 5,
        "language_detection_min_sec": 10.0
      },
      "embed": {
        "min_embedding_duration_sec": 0.75,
        "soft_embedding_min_sec": 0.30
      },
      "clustering": {
        "algorithm": "ahc",
        "num_speakers": null,
        "max_k": 50,
        "min_cluster_size": 2,
        "min_turns_per_speaker": 3,
        "k_selection_method": "elbow",
        "sil_weight": 0.6,
        "ch_weight": 0.4,
        "outlier_global_percentile": 97.0,
        "single_speaker_threshold": 0.2,
        "apply_snorm": false
      }
    }
  },
  "output": {
    "s3uri": "s3://bucket/output/prefix",
    "output_config": {
      "speaker_channels": true,
      "channel_mode": "aligned"
    }
  },
  "data": {}
}
```

`language: null` triggers auto-detection. Supply a BCP-47 code (`"en"`, `"cy"`) to skip detection.

`num_speakers: null` means auto-detect K. Supply an integer to override clustering.

`use_partial: true` + `mark_in_out: [start_sec, end_sec]` clips the audio before processing. The clip window must be ≥ 10 s and `start_sec` must be < content duration. If the content is ≤ 10 s `use_partial` is silently ignored. For WAV inputs the clip is applied in-Lambda by `wav_to_wav.convert()`; for non-WAV inputs the window is forwarded to the media-processing child SM via `use_partial`/`mark_in_out` in its `input_config`. `mark_in_out` is validated by `_validate_input_config()` only when `use_partial=true`.

`model_bucket`, `model_prefix`, and `cf_host` live in `input_config.model` and are populated from env vars (`MODEL_BUCKET`, `MODEL_ARTEFACTS_PREFIX`, `CLOUDFRONT_URL`) by `validate_and_normalize()`.

### Hidden output_config fields

These are not part of the public schema (excluded from `_merge_output_config`) but are read by `StateSeparateSpeakerChannels` if passed:

| Field | Default | Purpose |
|-------|---------|---------|
| `merge_gap_sec` | 0.5 | Merge adjacent same-speaker regions within this gap |
| `fade_ms` | 10 | Linear fade at segment boundaries (ms) |
| `compact_gap_sec` | 0.1 | Silence between segments in compact mode |

Pass them in `output.output_config` to override. They are not validated by `validate_and_normalize()`.

## Internal Data Fields

`StateCheckAndPreprocess` flattens all config into `data` for downstream states:

| Field | Written by | Read by |
|-------|-----------|---------|
| `language` | CheckAndPreprocess | DetectLanguage, MergeSpeakers |
| `use_partial` | CheckAndPreprocess | — (informational; resolved at CheckAndPreprocess time) |
| `mark_in_out` | CheckAndPreprocess | — (informational; resolved at CheckAndPreprocess time) |
| `model_cfg` | CheckAndPreprocess | Fixed/DynamicChunking, DetectLanguage, RunVAD, RunWhisper, RunEmbedding |
| `vad_cfg` | CheckAndPreprocess | FixedChunking, RunVAD, DynamicChunking |
| `asr_cfg` | CheckAndPreprocess → DetectLanguage (enriched) | RunWhisper |
| `embed_cfg` | CheckAndPreprocess | RunEmbedding |
| `clu_cfg` | CheckAndPreprocess | GlobalSpeakersClustering |
| `audio_bucket`, `audio_key` | CheckAndPreprocess (all paths) | All downstream states |
| `audio_cfg` | CheckAndPreprocess (WAV paths) | FixedChunking, RunVAD |
| `output_bucket`, `output_prefix` | CheckAndPreprocess | All downstream states |
| `speaker_channels`, `channel_mode` | CheckAndPreprocess (from output_config) | SeparateSpeakerChannels |
| `needs_conversion` | CheckAndPreprocess | definitions.json Choice state |
| `media_processing_request` | CheckAndPreprocess (conversion path) | SubmitMediaProcessing (gateway params); cleared by ExtractProcessedAudio |
| `media_processing_output` | MP's callback waker via `SendTaskSuccess` — the submit handle `{job_id, workflow_id, asset_s3uri, output_s3_prefix}` + terminal `status` | ExtractProcessedAudio; cleared after extraction |
| `media_processing_arn` | ExtractProcessedAudio (the handle's `job_id`) | — (traceability only) |
| `media_processing_usage_key` | ExtractProcessedAudio (derived: `{MP run prefix}/usage_tracking.json`) | UsageTracking (child-cost read) |
| `iterators` | FixedChunking → VAD Map → DynamicChunking → Transcribe+Embed Map | All Map states; cleared by MergeSpeakers |
| `language_confidence` | DetectLanguage | MergeSpeakers |
| `global_speakers` | GlobalSpeakersClustering | MergeSpeakers |
| `final_transcript_key` | MergeSpeakers | SeparateSpeakerChannels |
| `num_global_speakers` | MergeSpeakers | SeparateSpeakerChannels |

Note: `data.clu_cfg` is written from `input_config["clustering"]`. The abbreviation is intentional — it follows the same `*_cfg` pattern as `model_cfg`, `vad_cfg`, `asr_cfg`, `embed_cfg`.

### Hidden clu_cfg fields

These are not in the public input schema but are read by `StateGlobalSpeakersClustering` via `.get()` with defaults from `input_schema.py`:

| Field | Default | Purpose |
|-------|---------|---------|
| `centroid_trim_percentile` | 90.0 | Per-cluster periphery trimming before centroid computation |
| `min_turns_for_trim` | 5 | Minimum cluster size before trimming is applied |
| `k_penalty` | 0.02 | Penalty applied to combined score per unit K/k_max |
| `unreliable_silhouette` | 0.20 | Flag K selection as unreliable below this silhouette |
| `elbow_plateau_pct` | 0.03 | Silhouette within this fraction of sil_max is "plateau" |
| `assign_local_window_sec` | 30.0 | Locality window for LOW/outlier assignment and the SKIPPED midpoint fallback (lib/assign.py) |
| `assign_margin` | 0.05 | Cosine-similarity lead over the runner-up required to assign; below it → UNKNOWN |
| `use_crosstalk_constraints` | False | P3 cross-talk cannot-links ONLY (EXPERIMENTAL — off by default: hard links from the imprecise span→sentence mapping regressed DER; Design record P3). The `overlap_segments` output is ungated |

Hidden `vad_cfg` fields, same pattern (read by `StateRunVADModel`):

| Field | Default | Purpose |
|-------|---------|---------|
| `seg_threshold` | 0.30 | Powerset silence-class probability threshold (exposed for VAD sweeps; single source of truth `input_schema.DEFAULT_SEG_THRESHOLD`; Design record P2/P6) |
| `use_powerset_guides` | False | Powerset guides ([lib/powerset_guides.py](lib/powerset_guides.py)) — see the Key Conventions section below. Default OFF until the multilingual DER harness + KPI replay gates pass |

## Output Files (S3)

All written under `{output_prefix}/`:

| File | Contents |
|------|----------|
| `workflow_params.json` | Snapshot of the full normalized event at CheckAndPreprocess time |
| `transcripts/final_transcript.json` | Sentences + words with speaker labels, confidence, word timestamps; top-level `overlap_spans` = cross-talk ranges (VAD turns where ≥2 speakers were simultaneously active — turn-granular); top-level `word_timing_source` = word-timestamp provenance (`aligned:<lang>` \| `aligned:en_fallback` \| `whisper_native` — promotes the previously silent alignment fallback to a recorded fact); per-sentence `speech_spans` = word-tight spans split at intra-sentence pauses >0.5 s, but ONLY when alignment ran — `whisper_native` timings are too coarse to trim against (multilingual guard), so the sentence span stays unsplit; top-level `overlap_segments` = identity-resolved cross-talk spans with contract-stated identity precision (see step 10) |
| `transcripts/final_transcript.vtt` | WebVTT captions (Netflix style: 42 chars/line, 7 s max cue) |
| `speakers/SPEAKER_NN.wav` | Per-speaker audio (if `speaker_channels: true`) |
| `speakers/speaker_manifest.json` | Speaker list, durations, mode, compact time map |
| `usage_tracking.json` | Per-state cost breakdown; `total_estimated_cost` covers parent + child |
| `embeddings/global_speakers.json` | Clustering result: speaker models, sentence-speaker map, diagnostics |
| `segments/segments_{n}.json` | Per-fixed-chunk VAD turns (intermediate) |
| `powerset/powerset_{n}.json` | Per-fixed-chunk powerset guides — fences, overlap spans, tracks, co-active pairs + durable seconds (`co_active_secs`), clique floor, local speaker floor (intermediate; only when `use_powerset_guides`) |
| `dynamic/dynamic_segments_{n}.json` | Per-dynamic-chunk speech groups + VAD turns (intermediate) |
| `transcripts/transcript_{n}.json` | Per-chunk word-level transcript (intermediate) |
| `embeddings/embeddings_{n}.json` | Per-chunk speaker embeddings (intermediate) |

## Key Conventions

### Source Format Handling
[states/check_and_preprocess/check_helper.py](states/check_and_preprocess/check_helper.py) probes the source. Three outcomes:

- **Fast-path** — WAV, mono, exactly 16 kHz: used as-is when `use_partial=false`. When `use_partial=true`, `wav_to_wav.convert()` clips the window and writes to `{output_prefix}/clipped/audio_16k_mono.wav`. `needs_conversion=false`.
- **WAV-normalise path** — WAV but non-16 kHz or multi-channel: `wav_to_wav.convert()` resamples (and optionally clips) to 16 kHz mono int16 in-Lambda. Output at `{output_prefix}/resampled/audio_16k_mono.wav`. `needs_conversion=false`.
- **Media-processing path** — not a WAV (MP4, MXF, MKV, …): submitted to the media-processing GATEWAY (held task token). `use_partial` and `mark_in_out` are forwarded as flat gateway params; MP applies the clip. Output at `{MP's own run prefix}/diarization_audio/audio_16k_mono.wav`. `needs_conversion=true`.

`resolve_mark_in_out()` in [check_helper.py](states/check_and_preprocess/check_helper.py) normalises the clip window against the actual content duration: silently disables `use_partial` for content ≤ 10 s, clamps `mark_out` to the content duration, and raises `ValueError` if the resulting window is < 10 s. Called for both WAV paths (duration known from the header); skipped for the media-processing path (duration unknown at this stage).

### Config Flattening
`StateCheckAndPreprocess` flattens `input_config` subsections into `data` using the `*_cfg` naming convention (`model_cfg`, `vad_cfg`, `asr_cfg`, `embed_cfg`, `clu_cfg`). `output_config` is spread directly into `data` via `**output_config` (adds `speaker_channels` and `channel_mode`). Downstream states read all config from `data`, never from `event.input.input_config` or `event.output.output_config`.

### S3 Utilities
All S3 access goes through [lib/s3_utils.py](lib/s3_utils.py). Single `boto3.client("s3")` with `max_pool_connections=60` shared via `from lib.s3_utils import s3`. `get_json` / `put_json` used throughout. `upload_file`, `copy_object`, `delete_object`, `s3_uri`, `presign`, `head_size` are defined but have no callers.

### Speech Groups
Adjacent VAD turns with an inter-turn gap < `speech_group_max_gap_sec` (default 1.0 s) are merged into one speech group for Whisper. A 0.5 s synthetic silence pad is inserted between groups before batching.

### Three-Tier Embedding Quality
[lib/embedding.py](lib/embedding.py) classifies each sentence by its longest contiguous speech segment:
- **HIGH** (≥ 0.75 s) — full embedding; AHC clustering input. Outlier turns removed pre-clustering are re-assigned via [lib/assign.py](lib/assign.py); no confident local match → UNKNOWN.
- **LOW** (0.30–0.75 s) — embedding extracted; assigned via **locality-constrained matching** ([lib/assign.py](lib/assign.py)): candidates are speakers with a HIGH anchor within `assign_local_window_sec`, winner must lead the runner-up by `assign_margin`, else UNKNOWN. (Global nearest-centroid was measured at/below chance on short embeddings, which encode phonetics rather than voice; locality scored far above chance — Design record P5.)
- **SKIPPED** (< 0.30 s or empty) — no embedding; assigned via diarization overlap; the nearest-midpoint fallback applies only within `assign_local_window_sec`, else UNKNOWN.

`global_speakers.json` carries an `assignment` block (low/skipped assigned-vs-UNKNOWN counts, outliers rejected, the window/margin used) for KPI tracking.

### Dual K-Selection
[lib/cluster_sweep.py](lib/cluster_sweep.py) runs two scoring methods per candidate K:
- **Elbow**: lowest K whose silhouette is within `elbow_plateau_pct` (3%) of the global maximum
- **Combined**: weighted silhouette (0.6) + Calinski-Harabász (0.4) minus a K penalty (0.02)

If both agree, that K is used. On disagreement, `k_selection_method` picks the winner (`"elbow"` default) — **unless a guardrail fires** (Design record P1, validated by offline replay of production sweeps): a weak curve (best silhouette < `unreliable_silhouette`) or a flat curve (silhouette range < 0.05) resolves the disagreement to the **larger of the two candidate K's** ("prefer-split"): a signal-free curve makes the elbow plateau rule collapse to the lowest k (one measured run selected k=2 with the overwhelming majority of speech in one cluster), and the downstream cost is asymmetric — cast-resolution welds over-split clusters back together, but an over-merged cluster can only have its names stripped (a measured lower-K pick cost one run most of its named speech). Additionally, when the sweep is still climbing at a `max_k`-imposed ceiling (last combined ≥ 98% of max), the ceiling is raised once to 2×`max_k` (never past `N // min_turns_per_speaker`) and the sweep extends; `selection_reason` records `ceiling_raised` and, if the curve still climbs after that, `⚠still_climbing_at_ceiling`. Guardrail constants live at the top of `lib/cluster_sweep.py`; unit tests in [test/test_k_sweep_guardrails.py](test/test_k_sweep_guardrails.py).

The two outlier-related percentile parameters have different scopes:
- `outlier_global_percentile` (97.0) — pre-clustering: removes turns whose mean pairwise distance to all others exceeds the 97th percentile. Capped at 3% of N.
- `centroid_trim_percentile` (90.0, hidden) — post-clustering: trims periphery embeddings within each speaker cluster before computing the centroid.

### Powerset Guides (experimental, `use_powerset_guides`, default OFF)

[lib/powerset_guides.py](lib/powerset_guides.py): keep what the segmentation model knows instead of collapsing it to plain VAD. ONE flag, read in `StateRunVADModel` only — every downstream state keys off `powerset/powerset_{n}.json` artifact PRESENCE (three-state: absent, present, present-but-degenerate). Default OFF until the multilingual DER harness + KPI replay gates pass.

- **Fences** — in-speech speaker-change fences split sentences and embedding slices across sub-0.3 s handovers the VAD/wtpsplit/slicer are all blind to.
- **Overlap excision** — overlap spans are excised from embedding slices (pyannote's own `exclude_overlap` parity).
- **Tracks** — stitched local speaker tracks (per-row bind-or-break across the 9 s shared by adjacent windows) are embedded as CONTIGUOUS segments and clustered as units, with co-activity cannot-links + a max-clique K floor. Decisive measured constraint: track segments must be embedded as contiguous audio — averaging per-sentence vectors does NOT substitute (a mega-cluster asset collapsed under pooling in offline rehearsal).
- **Post-clustering split pass** — any cluster whose sentence-bearing member chains 2-split at cosine silhouette ≥ `POWERSET_SPLIT_SIL` divides; catches same-register never-co-active voices the sweep merges. Runs after the knife-edge K selection, prefer-split direction; standard Rousseeuw silhouette, singleton side scores 0.
- **Small-N guard** — the tracks path requires the K sweep to have ≥ `POWERSET_MIN_SWEEP_CANDIDATES` (4) candidate values — floor = 5×`min_turns_per_speaker` chains (15 at defaults; scales with config) — else sentence-unit fallback (diag `smalln_track_fallback`); a tracks sweep that self-reports `fallback_no_valid_candidates` is rerun on sentence units regardless of count (diag `tracks_degenerate_fallback`). Measured: a sparse-dialogue short with only a handful of chains forced a single speaker on both mix and stem audio while sentence units give K=2; healthy assets carry an order of magnitude more chains.
- **K-trust package** (floors up, splits calibrated down — added after an ear review found main characters split across clusters and two same-family voices welded):
  1. **Durable local-speaker floor** — a window's 0/1/2 labels are distinct people by model semantics, so the max per-window durable count (≥ `POWERSET_LOCAL_FLOOR_MIN_SEC` active, corroborated by ≥ `POWERSET_LOCAL_FLOOR_MIN_WINDOWS` windows) floors K alongside the clique floor; precedence user `num_speakers` > floor (conflict reported via `floor_user_conflict`, never enforced against the user).
  2. **Violation veto** — in the sweep's ⚠unreliable/flat regime, candidate K's violating cannot-links are dropped ([lib/cluster_sweep.py](lib/cluster_sweep.py)).
  3. **Cross-chunk binds** — chunk marks clip tracks, so uncontested continuity at a mark emits bind CANDIDATES (`xchunk_bind_candidates`); the clustering state applies one only when chain voices agree at ≥ `POWERSET_XCHUNK_BIND_SIM` (continuity alone welded a rapid-exchange handover).
  4. **Reconcile pass** — clusters at centroid sim ≥ `POWERSET_RECONCILE_SIM` (0.85; one person's fragments measure well above it, different-people pooled similarity well below) MUST merge unless evidence blocks: durable co-presence ≥ `POWERSET_RECONCILE_LINK_MIN_SEC`, quick alternation corroborated by ≥ `POWERSET_RECONCILE_CORROB_SEC` of co-presence, or a cross-register F0 gap (one predicate `_cross_register` for all three F0 gates — ratio AND straddle; band-only misread a single narrator's emphatic range as two registers). Co-presence comes from the artifact's `co_active_secs` (pair-class frames deduped on the grid) — NEVER from track-segment intersection, which is 0 by construction (track segments are disjoint; both gates were silently dead until this was found). Skipped entirely under user `num_speakers`.
  5. **Dust absorption** — a final cluster under `min_turns_per_speaker` is residue: members fold into the nearest survivor within `assign_margin`, else UNKNOWN.

  Post-pass violations + any floor crossing are recounted on shipped labels (`post_pass_cannot_link_violations`, `k_below_floor`, report-only). Offline referee: [test/replay_acceptance.py](test/replay_acceptance.py) replays cached corpus runs against ear-labeled MUST_MERGE/MUST_STAY_APART pairs.
- **Splice binder** ([lib/splice_binder.py](lib/splice_binder.py)) — additional cross-chunk bind CANDIDATES judged by the segmentation model itself: a 4.5 s clean solo span per side + 1 s silence spliced into one 10 s window; ONE sustained solo slot across the pause = same voice. Runs in the clustering state (same container image; pyannote session bootstrapped via the registry, ≤3×3 probes per adjacent-chunk boundary — minutes at feature-film length, O(1) memory, state peak well under the 1792 MB tier). A bind needs the probe verdict AND embedding agreement ≥ 0.80 AND row/column consistency (a side with >1 surviving one-slot verdict abstains) — the detectors' blind regimes are complementary (the probe welds same-register non-interacting voices when trusted alone; embeddings weld same-register family voices), so only the AND is safe. Same-chunk binding was REFUTED (no boundary prior — retention collapsed); adjacent boundaries only. Candidates flow through the SAME voice-gated application as the continuity binder; degrades to zero candidates on any failure. Measured win: on a flat silhouette curve, ONE accepted bind moved the sweep from a collapsed K to the source-fed landing; healthy assets unchanged. Diagnostics under `clustering_diagnostics.splice_probe`.

### Usage Tracking and Child Execution Cost
`StateUsageTracking` reads `data.media_processing_usage_key` (written by `StateExtractProcessedAudio`). If set, it `get_json`s media-processing's **own published** `usage_tracking.json` from that key and embeds `{execution_arn, pipeline_wall_sec, estimated_cost, states}` as the `media_processing` block — no walking of MP's execution history, no cross-machine IAM (that is what retired the `MediaProcessingStateMachineArn` parameter). MP's `states` entries follow MP's usage shape, embedded verbatim. A missing file after a SUCCEEDED handle is a pipeline bug and raises.

### ORT CPU Fix
Lambda removes `/sys/devices/system/cpu/possible` and `/present`. The Dockerfile copies `patch.txt` ("0-5") to both paths. [lib/ort_init.py](lib/ort_init.py) creates them at runtime if absent. `get_num_cpus(memory_mb)` derives vCPU count from the Lambda allocation table (1792 MB → 1, …, 10240 MB → 6).

### WAV Handling
[lib/wav_utils.py](lib/wav_utils.py) has no external media dependencies. Header parsed from a 4 KB byte-range GET (falls back to 64 KB). Handles `WAVE_FORMAT_EXTENSIBLE`, non-standard chunk ordering, multi-channel downmix. `encode_wav_mono()` for small in-memory WAVs; streaming path (`encode_wav_header_mono()` + chunked int16) in `StateSeparateSpeakerChannels` for large files.

## AgentCore Gateway Tool

[gateway_tool/lambda_function.py](gateway_tool/lambda_function.py) is the AgentCore Gateway dispatcher that wraps this workflow as MCP tools. It is deployed as a separate Lambda (not a container — flat zip, no Docker).

### Tools exposed

**`diarize_speech`** — validates and normalises all parameters, constructs the nested SFN input, derives the workflow_id, and calls `StartExecution`.

- All 19 `input_config` / `output_config` fields from the workflow schema are exposed as flat top-level parameters.
- `mark_in_out` sets `use_partial=True` automatically; the agent never sees `use_partial`.
- workflow_id strategy: `workflow_id = uuid5(canonical JSON of the SFN input WITHOUT output.s3uri + "|" + TOOL_VERSION + "|" + idempotency_token when one is supplied)` — derived by the shared [gateway_tool/workflow_identity.py](gateway_tool/workflow_identity.py), which documents the three ids. It NAMES the execution, so identical requests are idempotent: on `ExecutionAlreadyExists` the call is **absorbed** into the existing run — no second execution — while any parameter, token, or tool-version change re-runs to a fresh prefix. Outputs land at `s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{workflow_id}`; a caller-supplied `output_s3_prefix` is deprecated and ignored.
- `idempotency_token` (optional, 1-80 chars of `A-Z a-z 0-9 _ . -`) is the caller's assertion "this is the same request I sent before". It rides in the hash preimage as extra material and is then **discarded** — never forwarded to the state machine, never stored, never echoed. Omitting it still dedups on the parameters alone, since they name the asset and therefore determine the output; a token only distinguishes two otherwise-identical requests. The retired `uuid` parameter is **rejected** on submit (it used to mean both "which asset" and "same request", so a caller still sending it cannot be assumed to mean the surviving half) and **accepted with a log warning** on `get_job_status` / `describe_tool`, where neither key could have changed the answer.
- Returns a **HANDLE, never results**: `{"job_id": <execution_arn>, "status": "RUNNING", "current_state": <active stage>, "workflow_id": <derived id>, "output_s3_prefix": <derived prefix>, "asset_s3uri": <source s3uri>}`. An absorbed resubmit returns *the same fields* — only `status` differs (the existing run's, which may be terminal), plus `message` (present only when absorbed, and the only thing distinguishing the two cases — the `job_id` is a hash of the same input either way) and, on a terminal failure, `error` + `error_code` — compacted exactly like `get_job_status`'s (the stable class-name code as a field, never a raw stack-trace blob). `current_state` is dropped once the run has stopped, because a stopped execution is in no state and reading one costs a `GetExecutionHistory` paging pass for a `"unknown"`. Results come from `get_job_status` alone; that is what makes the declared `OutputSchema` equal to the reachable field set. No `uuid`, deliberately — the SFN input still carries a `uuid` key (set to the workflow_id) because every state derives its output prefix from it; that is this execution's local id, not the caller's token.

**`get_job_status`** — calls `DescribeExecution`; when RUNNING, pages `GetExecutionHistory` to find the active pipeline stage (state with `entered > exited`, lowest entry count wins so Map sub-states are suppressed). When SUCCEEDED, parses `_parse_succeeded_output` and returns structured fields:

| Group | Fields |
|-------|--------|
| Correlation | `job_id`, `workflow_id` (an `execution_arn` echo was removed — it was byte-identical to `job_id`) |
| Source | `asset_s3uri` |
| Output location | `output_s3_prefix` |
| Self-describing outputs | `outputs` list of `{uri, mime, label, desc}` (labels: `transcript`, `caption`, `speaker-manifest`, `global-speakers`, `speaker-audio` prefix, `usage`) |
| Primary output keys | `transcript_json_key`, `transcript_vtt_key`, `speaker_manifest_key`, `global_speakers_key` |
| Content facts | `language`, `language_confidence`, `num_sentences`, `duration_sec` |
| Speaker results | `num_speakers`, `clustering_method`, `single_speaker_detected`, `speaker_channels_status`, `speaker_channels_num_files` |
| Cost / timing | `pipeline_wall_sec`, `total_estimated_cost`, `usage_key` |
| Child execution | `media_processing_arn` (absent if no conversion) |

Internal plumbing (`*_cfg`, `model_manifest`, `iterators`, intermediate S3 keys, `language_probe_results`, `usage_tracking.states`) is excluded — the full detail is available on S3 via `usage_key` and `transcript_json_key`.

`desc` is the one-line role description joined verbatim from [gateway_tool/output_contract.json](gateway_tool/output_contract.json) — the same file `describe_tool` serves in full, so the two surfaces cannot disagree; it is omitted, never fabricated, for a label the contract lacks.

**Every branch** — RUNNING, SUCCEEDED, every failure, and the fallback for a status Step Functions has not shipped yet — carries `job_id`, `workflow_id`, `asset_s3uri`, and `output_s3_prefix`, so a status reply always names its own run and its output location even when there is nothing to report yet. (`output_s3_prefix` is read back from the execution's *input*, not its output, which is why a RUNNING or failed run still has one — partial results already live under it.) Terminal failures add `error` — `code: message` when the cause parsed as a Lambda error JSON, the raw cause (truncated 600) otherwise — and `error_code`: the stable code alone, i.e. the raised exception's class name (`NoAudioStream`, `NoSpeechDetected`) or, when the cause carried no parseable type, Step Functions' own error field (`Lambda.Unknown`, `States.Timeout`).

**`describe_tool`** — returns `{"version": <TOOL_VERSION>, "state_machine_arn": <ARN>, "description": <short backend description>, "output_contract": <parsed output_contract.json>}`. Version is stamped at deploy time via the `TOOL_VERSION` env var (set from the `.version` file by `build-deploy-s3-dist.sh`); the ARN comes from the state machine env var, giving callers a console/CLI entry point without parsing a job_id.

`output_contract` is the field-level contract for every `outputs[]` label: basename, kind (`file`/`prefix`), mime, the one-line `description` that doubles as the entries' `desc`, and a compact single-element `example` of the file's contents (`null` for the VTT and the WAV prefix, whose formats live in `notes`). It is static per `TOOL_VERSION` — the file ships in the gateway zip (`build-deploy-s3-dist.sh` pins the member), so version and contract cannot drift — which is why it lives here and not in `get_job_status`: a caller caches it once per session, and CI can pin it without running a job. Contract keys == emitted labels, exactly; anything not listed (`segments/`, `dynamic/`, per-chunk `transcripts/transcript_{n}.json` and `embeddings/embeddings_{n}.json` intermediates, `workflow_params.json`) is internal and may change without notice. The examples were authored from the producing code, not the docs. Note the transcript's top-level confidence field is `language_confidence`, and the per-speaker usage entries carry `retried`/`incomplete` while the self-estimated `Usage tracking` entry does not.

### Tool routing

AgentCore passes `context.client_context.custom["bedrockAgentCoreToolName"]` as `"{target_name}___{tool_name}"`. `_resolve_tool()` strips the target prefix and returns the bare tool name.

### Agent system prompt

The companion system prompt for a Strands agent using this target lives in the family gateway repo at [media-toolkit-gateway/prompts/speech_diarization_agent.md](../media-toolkit-gateway/prompts/speech_diarization_agent.md). It covers parameter selection heuristics, polling strategy, output field reference, and cross-agent correlation patterns — content that belongs in the agent's reasoning context, not the tool schema.

---

## CloudFormation Stack

[cfnstack.yaml](cfnstack.yaml) is a self-contained standalone stack. Custom resource Lambda is inlined.

### Resources created

| Resource | Name / Value |
|----------|-------------|
| CodeBuild Project | `${ResourcePrefix}-speech-diarization-on-aws` |
| Start-build Lambda | inlined Python (StartBuildDelayResponse pattern) |
| Lambda functions (×7) | `${ResourcePrefix}-speech-diarization-{variant}` |
| Step Functions state machine | `${ResourcePrefix}-speech-diarization` |
| Gateway tool log group | `/aws/lambda/${ResourcePrefix}-speech-diarization-gateway-tool` (7-day retention) |
| Gateway tool IAM role | `${ResourcePrefix}-speech-diarization-gateway-tool` |
| Gateway tool Lambda | `${ResourcePrefix}-speech-diarization-gateway-tool` (Python 3.13, arm64, 256 MB, 30 s) |
| Gateway tool Lambda permission | `bedrock-agentcore.amazonaws.com` invoke principal |

### Lambda variants

| Variant | Memory | vCPUs | States |
|---------|--------|-------|--------|
| `half-core` | 896 MB | 0.5 | StateExtractProcessedAudio, StateFixedChunkingForVAD, StateDynamicChunkingBySpeechTurns, StateUsageTracking |
| `1-core` | 1792 MB | 1 | StateGlobalSpeakersClustering, StateMergeSpeakersTranscripts |
| `2-cores` | 3584 MB | 2 | StateCheckAndPreprocess, StateRunVADModel, StateRunSpeechEmbeddingModel, StateSeparateSpeakerChannels |
| `3-cores` | 5376 MB | 3 | StateDetectLanguageDownloadModels |
| `4-cores` | 7168 MB | 4 | StateRunWhisperAlignWtpsplitModels |
| `5-cores` | 8960 MB | 5 | — |
| `6-cores` | 10240 MB | 6 | — |
| `gateway-tool` | 256 MB | — | AgentCore Gateway dispatcher (Python 3.13, arm64, no container) |

### Stack parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `ResourcePrefix` | — | Prefix applied to all resource names |
| `IngestBucket` | — | S3 bucket containing source media files (read-only) |
| `ProxyBucket` | — | S3 bucket Lambda functions read/write audio and models to |
| `MediaProcessingGatewayLambdaArn` | — | ARN of media-processing's gateway tool Lambda; `Run media processing` submits through it (held task token). There is deliberately NO `MediaProcessingStateMachineArn` parameter |

### Stack outputs

| Output | Purpose |
|--------|---------|
| `GatewayToolLambdaArn` | Imported by `media-toolkit-gateway/cfnstack.yaml` as `SpeechDiarizationToolLambdaArn` |

### IAM notes
The state machine role only invokes this stack's own Lambdas — the gateway-sibling conversion removed `states:StartExecution/StopExecution/DescribeExecution` on MP and the `events:*` managed-rule grants that `.sync:2` required. The workflow Lambda role has:
- `s3:ListBucket` + `s3:GetObject` + `s3:HeadObject` on `IngestBucket` (read-only; fast-path source audio)
- `s3:ListBucket` + `s3:GetObject` + `s3:PutObject` on `ProxyBucket`
- `states:GetExecutionHistory` on this stack's own executions only (MP's cost comes from its published usage file)
- `lambda:GetFunctionConfiguration` on this stack's own functions
- `lambda:InvokeFunction` on `MediaProcessingGatewayLambdaArn` (StateSubmitMediaProcessing's gateway submit)

The gateway tool Lambda role has:
- `logs:*` on its own log group
- `states:StartExecution` on the speech-diarization state machine ARN
- `states:DescribeExecution` + `states:GetExecutionHistory` on execution ARNs under that state machine

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
- `regional-s3-assets/speech-diarization-on-aws-{VERSION}.zip` — Docker source context (CodeBuild input)
- `regional-s3-assets/speech-diarization-gateway-tool-{VERSION}.zip` — flat zip of the `gateway_tool/` members (dispatcher, `bucket_validator.py`, `workflow_identity.py`, `output_contract.json`; deployed directly to the gateway tool Lambda)
- `global-s3-assets/speech-diarization-stack.template` — processed CloudFormation template

The `%%PKG_GATEWAY_TOOL%%` token in `cfnstack.yaml` is substituted by the build script to the gateway tool zip filename.

### Fast code-only redeploy (no CFN update)

The CodeBuild project pins `IMAGE_TAG` as a project env var, and the buildspec **skips the docker build when that tag already exists in ECR** — re-triggering the build with only a new source zip silently rebuilds nothing (measured: BUILD phase 0 s, Lambdas "updated" to the same old digest). The working recipe:

```bash
bash ./build-deploy-s3-dist.sh --bucket BUCKET --solution SOLUTION --version NEWVER --single-region
aws codebuild start-build --project-name ${ResourcePrefix}-speech-diarization-on-aws \
  --source-location-override BUCKET/SOLUTION/NEWVER/speech-diarization/speech-diarization-on-aws-NEWVER.zip \
  --environment-variables-override name=IMAGE_TAG,value=NEWVER,type=PLAINTEXT
# after SUCCEEDED:
python3 test/update_lambda.py --image-uri speech-diarization-on-aws \
  --prefix ${ResourcePrefix}-speech-diarization --region REGION
```

Verify with `aws lambda get-function … --query Code.ImageUri` — the digest must match the newly pushed tag. Note the gateway `TOOL_VERSION` is untouched by this path, so identical tool requests still absorb; use a fresh `idempotency_token` to force reruns.

## Tests

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

Tests gate packaging — `build-deploy-s3-dist.sh` refuses to package on a failing suite. [test/diagnose_workflow.py](test/diagnose_workflow.py) is a read-only field-diagnostic harness for a single run; [test/replay_acceptance.py](test/replay_acceptance.py) is the offline clustering referee over cached corpus runs.

## Dead Code

| Item | Location | Notes |
|------|----------|-------|
| `lib/s3_utils.py` | `s3_uri`, `presign`, `head_size`, `upload_file`, `copy_object`, `delete_object` | No callers |
| `lib/model_registry.py` | `get_concurrency()`, `get_lambda_memory_mb()` | No callers |
| `lib/model_registry.py` | `CHAR_BASED_LANGUAGES`, `NO_ENGLISH_ALIGN_FALLBACK` | No callers (the CTC charset is vocab-driven, language-agnostic); kept as recorded language knowledge — `test/diagnose_workflow.py` carries its own private copy |
| `lib/model_store.py` | `ensure_in_tmp_with_fallback()` | No callers; all states use `ensure_in_tmp()` |
| `lib/ort_init.py` | `_count_cpus_from_proc()` | No callers |
| `lib/input_schema.py` | `DEFAULT_ELBOW_PLATEAU_PCT` (first definition) | Shadowed by a second definition later in the file; remove the first |
| `lib/cluster_pipeline.py` | `_SINGLE_SPEAKER_MERGE_THRESHOLD`, `_SINGLE_SPEAKER_MEAN_DIST`, `_OUTLIER_GLOBAL_PERCENTILE`, `_MAX_OUTLIER_FRACTION` | Module-level constants now superseded by `DEFAULT_*` imports from `input_schema.py` |
| `lib/input_schema.py` | VBx constants (`VBX_DEFAULT_FA/FB/LDA_DIM/MAX_ITER/INIT_THRESHOLD`) | VBx permanently disabled by `_VALID_ALGORITHMS = frozenset(("ahc",))` |
| `states/global_speakers_clustering` | `_derive_output_prefix()` | No callers |

## Gotchas

- **Fetched audio buffers are ALWAYS `TARGET_SR` (16 kHz) — never do sample math on them with `audio_cfg["sample_rate"]` (the FILE's rate).** Measured incident: a 48 kHz WAV admitted by an older `>=16 kHz` fast-path reached the whisper/embedding states, whose group/sentence slicing used the file rate over 16 kHz buffers — indexes ran 3x deep, every speech group past 1/3 of each fetch window was silently skipped (`skip empty group`), and runs reported SUCCEEDED with two-thirds of their speech missing. Routing now converts non-16 kHz WAVs first (`check_helper` requires the exact rate), and both states pin their sample math to `TARGET_SR` with a warning if `audio_cfg` disagrees. `separate_speaker_channels` is exempt: it deliberately reads at native rate (`target_sr=None`) and keeps its math at the same rate.
- **`algorithm: "ahc"` only** — `"vbx"` is implemented in `lib/clustering.py` but gated out by `_VALID_ALGORITHMS = frozenset(("ahc",))` in `lib/input_schema.py`. **Do not re-enable as-is** (Design record P4): with the shipped defaults it collapses every asset to K=1.
- **media-processing must be deployed first, and at callback-capable version** — `cfnstack.yaml` requires `MediaProcessingGatewayLambdaArn`, and the submit handler refuses (before spending anything) when MP's `describe_tool.features` lacks `callback_task_token`. The inline `DefinitionString` in `cfnstack.yaml` is what deploys; `definitions.json` is the reference copy — `test/test_gateway_callback.py::TestAslParity` pins whole-document equality, so edit BOTH or the suite fails.
- **K ceiling** — `max_k` defaults to 50. AHC is O(N²) — setting it too high significantly increases clustering wall-clock time. The sweep may extend itself once to 2×`max_k` when the score is still climbing at the ceiling (see Dual K-Selection) — budget wall-clock for up to 100 candidates.
- **Language fallback degrades alignment — RECORDED, not silent** — if a language has no alignment model, forced-alignment falls back to the English model (Latin-script) or is skipped, without error. The fact is recorded as `word_timing_source` in `final_transcript.json` (`lib/model_registry.py: word_timing_source()`); word-timing-dependent consumers (speech-span trimming, embedding slices) must gate on it rather than on language lists.
- **Welsh (`cy`) alignment model may not exist on CloudFront** — if `align/cy/model.tar.gz` is missing from both S3 and CloudFront, `StateDetectLanguageDownloadModels` falls back to the English alignment model (safe for Latin-script languages). If `model_variant=turbo` is explicitly set for `language=cy`, the English fallback is applied pre-emptively without attempting the download.
- **Welsh (`cy`) and Arabic (`ar`) use dedicated Whisper models** — the `cy`/`ar` tarballs must be in S3 or downloadable via CloudFront.
- **Usage tracking must bill failed attempts.** An earlier version of [states/usage_tracking/lambda_function.py](states/usage_tracking/lambda_function.py) dropped any invocation with `succeeded_ts is None`, so a `TaskFailed` — a Lambda killed at its timeout, the most expensive invocation possible — contributed 0 s while `invocations` stayed correct, so the output read as precise (a real sibling-stack run under-reported severalfold). **Two invariants to preserve:** `_billed_sec` falls back to `failed_ts`, and 0.0 is correct **only** when an attempt has NO terminal event at all — that case is what `incomplete` means. `wall_sec` sums attempts per `entered_id` *before* taking the max across occurrences, because every retry attempt chains back to the same `TaskStateEntered` while concurrent Map iterators each get their own. `tts-on-aws` is the family reference copy; pinned cross-stack by `micro-workflow-bundle/test/test_units.py: TestUsageTrackingBillsFailedAttempts` — a regression fails the bundle suite, not this stack's.
- **[test/diagnose_workflow.py](test/diagnose_workflow.py) shared that attempt map but never priced it — its fix is a DISPLAY fix.** It builds the same `scheduled_ts`/`succeeded_ts` records, and a `FAILED` attempt likewise recorded `status`/`error`/`cause` but no end timestamp. The only consumer is the per-state failure listing (`duration_sec` elsewhere in the file comes from S3 group stats, `wall_sec` from `DescribeExecution`), so there was no second cost bug. It records `failed_ts` and prints `ran_for=` on a failed attempt — which is the fact that separates "ran to its 900 s timeout" from "raised in 2 s", since both print the same `error` string.
- **Usage tracking cost constants** in [states/usage_tracking/lambda_function.py](states/usage_tracking/lambda_function.py) are us-east-1 arm64 figures; update for other regions or when pricing changes.
- **Step Functions scheduling latency (~10–50 ms)** is included in billed-duration estimates — small systematic overestimate.
- **CTranslate2 and ORT are CPU-only** — do not use GPU instance types.
- **`data.clu_cfg` not `data.clustering`** — the clustering config section is stored as `clu_cfg` in `data`, not `clustering`. Consistent with the `*_cfg` convention but different from the schema key name.

## Model Preparation (Offline)

Scripts in [export/](export/) convert HuggingFace checkpoints to the ONNX/CTranslate2 formats expected by the container:

| Script | Output |
|--------|--------|
| `export_whisper_models.py` | CTranslate2 INT8 Whisper tarballs |
| `export_alignment_models.py` | ONNX INT8 wav2vec2 per-language tarballs |
| `export_pyannote_community1.py` | ONNX Pyannote community-1 tarball |
| `export_wtpsplit.py` | ONNX wtpsplit SaT tarball |

## Design record

Numbered decisions cited from code comments (`README.md Design record P<n>`). Numbering is preserved from the original clustering working plan; each entry carries the decision and its why — the full measured evidence lives in the project's internal records.

- **P1 — K-selection guardrails: prefer-split in the no-signal regime; raise the ceiling once when still climbing.** A weak (best silhouette < `unreliable_silhouette`) or flat (range < 0.05) curve carries no elbow signal, and the plateau rule then collapses to the lowest k — one measured production run selected k=2 and put the overwhelming majority of an episode's speech in one cluster. Because the downstream cost is asymmetric (cast-resolution welds over-split clusters back together, while an over-merged cluster can only have its names stripped — a measured lower-K pick cost one run most of its named speech), an elbow/combined disagreement under a weak or flat curve resolves to the LARGER candidate K. When the combined score is still climbing at the `max_k` ceiling (last ≥ 98% of max), the ceiling is raised once to 2×`max_k`, capped at `N // min_turns_per_speaker`; `selection_reason` records what happened. Validated by replaying production sweeps offline (a minority of picks moved, the rest unchanged). Constants at the top of [lib/cluster_sweep.py](lib/cluster_sweep.py); unit tests in [test/test_k_sweep_guardrails.py](test/test_k_sweep_guardrails.py).
- **P2 — Miss decomposition: the dominant DER miss class is VAD solo-span misses, and `seg_threshold` is the suspect.** Decomposing measured misses showed a few percent of SOLO speech missed at the VAD stage. `seg_threshold` (the powerset silence-class probability threshold) had been hardcoded in `lib/segmentation.py` with no recorded rationale; it is now a hidden `vad_cfg` knob with ONE source of truth (`input_schema.DEFAULT_SEG_THRESHOLD`) precisely so a harness can sweep it — which is how the default moved to 0.30 (P6).
- **P3 — Cross-talk cannot-link constraints: span evidence is sound, the sentence mapping is not — default OFF; the `overlap_segments` OUTPUT stays on.** Powerset overlap evidence proves per-window that two solo voices are different people (span-level precision measured at 100%), but mapping spans to SENTENCES is only ~77–93% precise under every rule swept (worse for zh, where sentences are coarser), and a hard cannot-link built on a wrong mapping shreds a true cluster — DER regressed on every corpus asset, so `use_crosstalk_constraints` defaults OFF. When enabled, linked pairs get distance 1e9 in the LINKAGE matrix only (silhouette/CH stay on honest distances), so proven-distinct voices merge LAST, not never; violations are counted report-only. The identity-resolved `overlap_segments` output ships regardless, with its identity precision stated in the output contract — evidence over verdict; consumers decide trust.
- **P4 — VBx is refuted with the shipped defaults and stays disabled.** Measured offline on stored production embeddings: VBx collapses every asset to K=1 — the `fa/D` scaling makes the acoustic term ~0.001 so the `fb·log(Nk)` prior always wins — and no cross-content (fa, fb) regime was found; the docstring claim that the defaults suit WeSpeaker 256-d embeddings is false. `_VALID_ALGORITHMS = frozenset(("ahc",))` in `lib/input_schema.py` is the gate; do not re-enable without a recalibration that generalizes across content.
- **P5 — Post-clustering mop-ups are locality-constrained with a rejection option.** Short (< 0.75 s) embeddings encode phonetics more than voice: measured on a labeled short-slice set, GLOBAL nearest-centroid assignment scored at or below chance (it maps systematically wrong, not just noisily), while restricting candidates to speakers with a clustered HIGH anchor within ±`assign_local_window_sec` (30 s) scored several times chance. Therefore every mop-up (LOW sentences, outlier turns, the SKIPPED midpoint fallback) goes through [lib/assign.py](lib/assign.py): local candidates only, and the winner must lead the runner-up by `assign_margin` or the item is UNKNOWN — never fattening the nearest big centroid.
- **P6 — VAD defaults carry the measured WER knee: `seg_threshold` 0.30, `min_speech_duration_sec` 0.** The 0.5 s minimum-duration discard deleted real short utterances (single words, names, interjections) before ASR ever saw them — a structural deletion class, with VAD-threshold misses as the sibling class. On the pooled caption-ground-truth battery, 0.30/0 was the knee bracketed from BOTH sides: 0.35/0 recovers fewer words, 0.25/0 buys back deletions at a worse insertion price; at 0.30/0 insertions FELL below the old defaults (fuller speech groups give the ASR better context at boundaries) at equal cost. Admitted short runs join speech groups normally but NEVER enter clustering (the P5 tiers: <0.75 s sentences are LOW/SKIPPED — locality-assigned or UNKNOWN), so recall recovery cannot pollute AHC. The same battery re-refuted the speech stem as a transcription input at the stem's own best cell (its floor is separator-deleted audio, not VAD). Retune ONLY against the pooled battery — single episodes swing on identical code; a finer-grained threshold is a fitted scalar.
