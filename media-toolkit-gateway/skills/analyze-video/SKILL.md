---
name: analyze-video
description: Run the end-to-end media analysis pipeline over the media-toolkit MCP tools for one asset — audio-stem separation, speech diarization, conversation analysis, shot segmentation, optional face recognition, shot-metadata enrichment, and program-structure analysis — then print a summary of the metadata. Use when the user asks to analyze a video end-to-end, run the full/complete analysis pipeline, or resume/replay a previous analyze-video run. To turn the results into an HTML page use the visualize-metadata skill; to export an editorial timeline use export-package.
metadata:
  version: 2.0.0
compatibility: Requires the media-toolkit MCP server to be connected. Reads source media from the ingest bucket and writes outputs to the proxy bucket; writes a state file into the workspace.
---

# Analyze video (end-to-end pipeline)

Run every micro-workflow against ONE asset, maximally parallel, persist a state
file, and finish with an in-chat SUMMARY of the metadata. This skill does NOT
build the HTML page or any export — it produces the metadata and the state file
that `visualize-metadata` and `export-package` consume.

All workflow tools are ASYNC: each returns a `job_id`; poll
`<workflow>___get_job_status` until `SUCCEEDED`/`COMPLETED` (or `FAILED`).
Per-tool parameter detail, output labels, idempotency semantics, and
troubleshooting live in the workflow references at
[../video-processing/references/](../video-processing/references/) — read the
relevant one when a step needs more than the step table below shows; this file
carries only THIS pipeline's wiring and composition.

Locate output files by their LABEL in the completed status's `outputs` entries —
read the labels, never guess S3 keys. If an expected label is missing, list the
`outputs` and pick by filename.

> ⚠️ ALWAYS go through the media-toolkit tools for every output —
> both reading result JSON and presigning/downloading media. Presign with
> `media-outputs___get_download_url(s)`; fetch JSON from those presigned URLs.
> Do NOT use `aws s3 cp` / `aws s3 presign` / `aws s3 ls`: the gateway is the
> contract (scoped access, correct output labels). The raw AWS CLI is an
> EMERGENCY fallback only when the gateway is unreachable (e.g. an expired
> token you cannot refresh), and any CLI-minted URLs MUST be replaced with
> gateway-issued ones once the gateway is back. If a gateway call fails with an
> auth error, ask the user to refresh the gateway token and reload — do not
> silently switch to the CLI.

## Run-or-replay (the top-level behavior)

On invocation, resolve `ASSET_ID` and look for
`outputs/analyze-video/{ASSET_ID}.state.json`:

- **No state file → RUN.** Execute the full pipeline below, writing state as you
  go, then print the summary.
- **State file exists → REPLAY.** Say so, then re-submit each recorded step with
  its original `RUN_TOKEN` + params (each workflow absorbs the call and returns
  existing outputs at no cost), refresh the state file, and print the summary.
  Steps that never finished are run to completion. This is cheap and safe.
- **User says "fresh"/"from scratch"/"re-run"**, or the face collection / prompt
  changed: increment the generation N → new `RUN_TOKEN` → genuinely new run.
  Preserve the prior state as `{ASSET_ID}.state.r{N-1}.json`.

Either way, the skill ENDS at the summary. Building the page or the FCPXML is a
separate, explicit step the user triggers via the other two skills.

> ⚠️ IDs FLOW, they are never retyped: copy every workflow_id / run prefix VERBATIM from the sibling run's response (`output_s3_prefix`) or the recorded state file — never from memory. A mistyped-but-EXISTING prefix binds against the wrong run and nothing will refuse it; only a nonexistent one fails validation.

## Input

Accept either:
- an **`s3://` uri** — use directly as `asset_s3uri`; or
- a **shorthand name** (e.g. "Love and Peace") — resolve it to an `s3://` uri by
  calling `media-ingest___list_files` (extension `mp4`, then `mov` if needed)
  and matching the name in the key. If several match, prefer a canonical-library-style
  canonical copy or ask which one; if none match, ask for the `s3://` uri.

`$ARGUMENTS` is the asset (uri or name) when present.

Derive once and reuse:
- `ASSET_ID` = source basename, sanitized to `[A-Za-z0-9_.-]`, max 40 chars.
- `HISTORY_SCOPE` = `asset.{ASSET_ID}` — pass to conversation-analysis AND
  shot-metadata-enrichment so their inference histories group per asset.
- `RUN_TOKEN` = `analyze-{ASSET_ID}-r{N}` (N = generation, 1 if none). Pass as
  `idempotency_token` on EVERY submit, face-recognition included — this is what
  makes replay free and keeps face-recognition from paying twice.

## Fixed decisions (do not re-ask)

- **Conversation analysis** runs after the transcript (step 4A).
- **Batch inference always**: `use_batch=true` for BOTH conversation-analysis
  and shot-metadata-enrichment (half cost, minutes-to-hours).
- **celebrity_recognition is always ON** when face recognition runs.
- The ONLY runtime confirm is the face-recognition collection (see below).

## Audio tool routing (three different jobs — don't substitute)

- **Use the ORIGINAL MIX** (the step-1 process_media wav) **when the audio
  feeds analysis**: transcription, diarization, conversation analysis,
  language detection — always, no exceptions. The speech stem silently
  deletes low-prominence speech, and both cleaners measured WORSE than
  the mix as transcription inputs. The mix is the validated default for
  all speech analysis.
- **Use `audio-stem___separate_audio_stems` when the deliverable is the
  audio taken APART**: a music bed, an effects lane, per-stem remix
  sources, or the stem activity / waveform timeline products this pipeline
  consumes (PSA input). N outputs, one per component. Its speech stem is a
  foreground-listening product — fine for review lanes, never an input to
  further speech analysis.
- **Use `speech-enhancement___enhance_speech` when the deliverable is the
  SAME audio with clearer dialogue**: a screener, a review copy,
  accessibility listening, a noisy interview made watchable. One WAV,
  enhanced in place and sample-aligned with the source; `atten_lim_db`
  caps the suppression when ambience is part of the content. It runs on
  the ORIGINAL MIX (standalone — never on a stem, never chained after
  audiostem), and it is not part of this analysis pipeline: it is a
  deliverable, not an analysis input.

## Pipeline graph — parallelize exactly this way

```
STEP 1  process_media (ONE job: mp4 proxy + wav output groups)
          ├─ on wav   ─→ 2A separate_audio_stems(wav)
          ├─ on wav   ─→ 3A diarize_speech(wav — the MIX, never a stem)
          │                └─ on transcript ─→ 4A analyze_conversation  [batch]
          └─ on proxy ─→ 2B segment_shots(proxy)
                           └─ on shot-list ─→ [CONFIRM] 3B recognize_faces(proxy, shots.json)
STEP 5  enrich_shot_metadata — JOIN: 2B shots + 3A transcript + 3B faces (if run)  [batch]
STEP 6  analyze_program_structure — 5 shot-metadata + 2B shots + 4A conversation + 3A transcript + 2A audiostem
```

Rules:
- **Gate each chain on step 1's own evidence**: if the wav
  output key is absent (media-processing drops the wav group on a source
  with no audio stream — check `output_keys`/the probe's `has_audio`), SKIP
  2A→3A→4A entirely and report "source has no audio — audio chain skipped".
  Submitting anyway fails the audio workflows by name (`NoAudioStream`,
  both audio-stem and diarization); audio that carries no speech
  (music-only) fails later as `NoSpeechDetected`. Likewise an absent proxy
  key (`has_video` false — an audio-only source) means SKIP 2B→3B
  (face-recognition fails by name as `NoVideoStream`) and run the audio
  chain on the wav alone. All are terminal content facts — never resubmit
  on them, just proceed with the chains the source supports.
- 2A, 2B and 3A start **together** the moment step 1 completes.
- The audio chain (2A→3A→4A) and the visual chain (2B→3B) never wait on each
  other. Poll all outstanding jobs in ONE parallel block per round.
- Step 5 is a join: it waits for 2B AND 3A (and 3B when opted in). If the audio
  chain failed, run 5 with shots only — partial success IS success.
- Step 6 (PSA) needs BOTH the shot-metadata output (5) AND the shot-list (2B);
  carry the shots.json uri all the way through. Feed it 4A, 3A, and 2A too when
  they completed — they sharpen its mode detection, gap map, and break analysis.

## Step reference

| # | Tool | Input (from) | Key params | Output labels |
|---|------|--------------|------------|---------------|
| 1 | `media-processing___process_media` | source uri | `output_groups: [{group_name:"proxy",format:"mp4"},{group_name:"audio",format:"wav"}]` | proxy mp4, wav |
| 2A | `audio-stem___separate_audio_stems` | wav (1) | defaults (all 3 stems) | speech/music/sfx WAVs, `stem-activity`, `stem-waveforms` |
| 2B | `shot-segmentation___segment_shots` | proxy (1) | defaults | `shot-list` (shots.json), recurrence |
| 3A | `speech-diarization___diarize_speech` | wav (1) — the MIX | defaults; `language` if user stated one | `transcript` (final_transcript.json), WebVTT |
| 4A | `conversation-analysis___analyze_conversation` | `transcript_s3uri` (3A) | `use_batch:true`, `history_scope` | analysis JSON |
| 3B | `face-recognition___recognize_faces` | proxy (1) + `shot_segmentation_s3uri` = the FULL shots.json URI (2B) | `collection_id` (confirmed), `celebrity_recognition:true`, `idempotency_token` | facemetadata.json |
| 5 | `shot-metadata-enrichment___enrich_shot_metadata` | `shot_segmentation_s3uri` (2B), `transcript_s3uri` (3A), `face_recognition_s3uri` (3B) | `use_batch:true`, `history_scope`, `program_names`/`overall_content_type` (single strings) if known; **news routing**: when 4A has completed and its `overall_content_type` says news, also pass `prompt_version: "default_news"` — that is what produces `supers`/`news_setting`/`live_bug_present` on shot docs (4A usually finishes well before this step; worth the wait on likely-news content) | `shots` (shot_metadata.jsonl), `scenes`, `window-manifest`, `usage` |
| 6 | `program-structure-analysis___analyze_program_structure` | `shot_metadata_s3uri` (5), `shot_segmentation_s3uri` (2B), `conversation_analysis_s3uri` (4A), `transcript_s3uri` (3A), `audiostem_s3uri` (2A) | `mode:auto` | `structure`, `rundown` (news mode), `manifest` |

> ⚠️ Wiring gotcha: face-recognition's `shot_segmentation_s3uri` takes the
> FULL `shots.json` object URI (the 2B `shot-list` entry's `uri`), never the
> run prefix — details in the face-recognition reference. Other joins (SME,
> PSA) accept the run prefix; face-recognition does not.

## Face-recognition confirm (the one runtime question)

Use your harness's question primitive if it has one; otherwise ASK IN
CHAT and WAIT for the reply. When the shot-list (2B) is ready:
1. Call `face-recognition___list_collections`.
2. Ask the user, in chat: "Run face recognition? If so, which collection?"
   — list each collection by name with its FaceCount, plus a "Skip" option.
   State that `celebrity_recognition` will be ON.
3. On a collection choice: submit 3B with that `collection_id`,
   `celebrity_recognition: true`, and the `RUN_TOKEN`.
4. On skip: proceed; step 5 runs without `face_recognition_s3uri` and people are
   described generically.

On REPLAY, do NOT re-ask — reuse the collection recorded in the state file.

## Polling and progress

- Poll all outstanding jobs in one parallel block, then wait between rounds:
  ~30 s while GPU steps (1, 2A, 2B, 3A, 3B) are running; stretch to ~120 s once
  only batch jobs (4A, 5) remain.
- Report one short line per running job each round.
- If a batch job is still running after ~20 min, offer to keep waiting or stop
  and resume later (the state file covers resumption).

## State file — the contract with the other skills

Persist to `outputs/analyze-video/{ASSET_ID}.state.json` after every submit
and completion. `visualize-metadata` and `export-package` READ this file to
locate the run's outputs, so it must record, per step: `job_id`, `status`, the
output `prefix` and the key labeled uris (proxy, wav, speech/music/sfx stems,
`stem-waveforms`, `shot-list` shots.json, `transcript`, conversation `analysis`,
face `facemetadata`, SME `shots`/`scenes`, PSA `structure`). Also record:
`source_uri`, `ASSET_ID`, generation `N`, `RUN_TOKEN`, `HISTORY_SCOPE`,
`proxy_bucket`, the face `collection` (or "skipped"), and `content_mode` /
`psa_mode`. Write s3:// uris (NOT presigned URLs — those expire; the consumers
presign fresh).

## Summary / highlights (how this skill ENDS)

After the pipeline (or replay) completes, print an in-chat digest — text and
small markdown tables, NO HTML. Include:

- **Header**: title/ASSET_ID, duration, `overall_content_type` + top genres
  (from conversation-analysis), and the resolved `psa_mode`.
- **Structure**: scene count, shot count, PSA sequence count; list the scenes
  briefly (id · timecode range · setting · one-line description) or a compact
  table for short programs.
- **People**: named cast with screen-time and confidence; count of unknown
  faces. (Omit if face-rec was skipped.)
- **Dialogue**: speaker count, sentence count, language; the conversation
  chapters (topic · sentiment · time range) and any PSA break candidates.
- **Run health / QC**: `shots_gap` / `segments_gap` (say "complete" when empty),
  `pairing_repair_rate`, bars-and-tone / line-up tones if any, per-step
  status, and total estimated cost from the usage outputs.
- **Next steps**: tell the user they can now run **`visualize-metadata {ASSET_ID}`**
  to build the interactive HTML page, **`export-package {ASSET_ID}`** to emit
  an FCPXML editorial timeline, or **`index-video {ASSET_ID}`** to make the
  metadata searchable via hybrid-search (then query with `search-media`). Note the state file path.

## Reporting and failure handling

- After each phase, report one line: what finished, wall time, cost if the
  usage output carries one.
- A failed step kills only its dependents: no wav → skip the audio chain, still
  do visual + SME (shots-only) + PSA; no transcript → SME without spoken
  evidence and PSA without the gap map. Say plainly what failed (with its status
  message) and what was skipped; the summary reflects only what completed.

## Notes

- If any submit errors because the gateway isn't connected, tell the user to
  connect the `media-toolkit` MCP server (reload the window or use the
  MCP Servers view) and retry.
- Before composing program-structure or enrichment calls, you may call the
  relevant `___describe_tool` to confirm the CURRENT output contract rather than
  trusting this file — schemas can evolve.
