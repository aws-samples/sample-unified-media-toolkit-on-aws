---
name: video-processing
description: "Route media requests to the right Unified Media Toolkit micro-workflow tool and load only the per-workflow reference you need. Use for ANALYSING or TRANSFORMING media that already exists in the toolkit's buckets — probe/trim/transcode/extract (process_media), compose a new video from a cut list (quick_edit), stills at moments (capture_frames), stem separation, in-place dialogue enhancement, transcription/diarization, conversation analysis, shots, faces, shot enrichment, program structure, cast resolution, audio description, text-to-speech, and hybrid search/indexing. This is the ROUTER: pick the row, read its reference, then call the tool. Not for generating video/images from scratch."
metadata:
  version: 1.0.0
---

# Video & Audio Processing — tool router

Keep this file as the router. Before calling a micro-workflow tool, read the
ONE applicable reference under [references/](references/) — they carry
parameter choices, output labels, idempotency rules, progress states, and
troubleshooting. For a compound request, read each reference whose tool you
will actually call; do not load unrelated ones. The live gateway tool schema
is authoritative where it differs from a reference. If an optional target is
absent from the connected catalog, say it is unavailable — never substitute
another tool.

## Intent → tool

| User intent | Gateway submit tool | Reference |
|---|---|---|
| Technical metadata — codec, resolution, fps, duration, tracks, timecode, HDR check (cheap: no transcode) | `media-processing___process_media` with a `metadata` output group | [Media processing](references/media_processing_agent.md) |
| Render ONE whole asset: proxy, audio extraction, periodic frame sampling, metadata — including a simple single-range trim via `mark_in_out` | `media-processing___process_media` | [Media processing](references/media_processing_agent.md) |
| Compose a NEW video from marked moments: in/out cuts over one or MORE sources, transitions, fade-to-black, vertical/social reframes, voiceover or music beds, channel-only segments | `media-processing___quick_edit` | [Media processing](references/media_processing_agent.md) |
| Stills at SPECIFIC moments (thumbnails, artwork candidates, QC frames) — snapped to each source's frame grid | `media-processing___capture_frames` | [Media processing](references/media_processing_agent.md) |
| Separate mixed audio into speech / music / SFX stems; stem activity and waveform timeline products | `audio-stem___separate_audio_stems` | [Audio stem separation](references/audio_stem_agent.md) |
| Make dialogue easier to hear IN PLACE (screener, review copy, accessibility listening) — one enhanced WAV, not stems | `speech-enhancement___enhance_speech` | [Speech enhancement](references/speech_enhancement_agent.md) |
| Transcribe, identify speakers, captions, per-speaker audio — on the original mix audio (the validated default; stems/enhanced tracks measured worse today) | `speech-diarization___diarize_speech` | [Speech diarization](references/speech_diarization_agent.md) |
| Segment a transcript into topics with tone, sentiment, key events (needs a completed diarization) | `conversation-analysis___analyze_conversation` | [Conversation analysis](references/conversation_analysis_agent.md) |
| Detect shots, scenes, cuts, recurrence; shot/frame embeddings | `shot-segmentation___segment_shots` | [Shot segmentation](references/shot_segmentation_agent.md) |
| Detect, track, identify people; screen time; manage face collections | `face-recognition___recognize_faces` + the `face-recognition___*` management tools | [Face recognition](references/face_recognition_agent.md) |
| Per-shot descriptions, objects, setting, on-screen text, brands (needs shots; optional faces/transcript joins) | `shot-metadata-enrichment___enrich_shot_metadata` | [Shot metadata enrichment](references/shot_metadata_enrichment_agent.md) |
| Program structure: sequences, mode detection, news rundown, break candidates (needs shot metadata) | `program-structure-analysis___analyze_program_structure` | [Program structure analysis](references/program_structure_analysis_agent.md) |
| Bind speakers + faces + on-screen text into NAMED person entities with evidence | `cast-resolution___resolve_cast` | [Cast resolution](references/cast_resolution_agent.md) |
| Audio-description script + narration for an analyzed video (render the mix via `process_media` `ad_mix`) | `audio-description___generate_ad` | [Audio description](references/audio_description_agent.md) |
| Artwork / thumbnail / poster candidate frames ranked per archetype, honoring an editorial brief (needs completed sibling runs) | `artwork-selection___select_artwork` | *(no per-workflow reference yet — the `select-artworks` playbook skill covers usage)* |
| Synthesize speech from text; clone a voice; manage the voice catalog | `text-to-speech___synthesize_speech` / `register_voice` | [Text-to-speech](references/tts_agent.md) |
| Upload a LOCAL file into the ingest bucket (presigned single PUT or parallel multipart, max 5 GiB; bytes go client-to-S3, never through the gateway) | `media-uploads___start_upload` (+ `complete_upload`) | [Media uploads](references/media_uploads_agent.md) |
| Search indexed media by meaning/keyword; index content; collections | `hybrid-search___search_content` / `index_content` | [Hybrid search](references/hybrid_search_agent.md) |
| Run rendered prompts through Bedrock models (sync or half-cost batch) | `bedrock-inference___run_inference` | [Bedrock inference](references/bedrock_inference_agent.md) |
| Save/fetch/promote versioned prompts | `prompt-registry___*` | [Prompt registry](references/prompt_registry_agent.md) |

Two targets have no reference file because every reference already covers
them in its downloads section: `media-outputs___get_download_url[s]`
(presign outputs — the ONLY way to HTTP-fetch results) and
`media-ingest___list_files` (list source media; its `s3uri` feeds the
submit tools directly). The SUPERUSER surfaces `hybrid-search-admin` and
`prompt-registry-admin` are deliberately NOT routed here — they exist only
for trusted admin agents and most gateways do not attach them.

## Cross-tool routing rules (do not re-derive these)

- **Speech analysis reads the ORIGINAL MIX by default.** Transcription,
  diarization, conversation analysis, language detection take the plain
  extracted wav — the validated input. The speech stem silently deletes
  low-prominence speech and the enhanced track measured worse as an STT
  input on the current models, so feeding either is NOT recommended today
  (both are primarily deliverables); if a caller insists, compare the
  result against a mix-based one.
- **`process_media` vs `quick_edit`**: one whole asset (even a trimmed
  range) → `process_media`; anything composed — multiple ranges, multiple
  sources, transitions, reframes, overlays — → `quick_edit`.
- **Periodic frames vs marker stills**: every-N-seconds sampling is a
  `process_media` frames group; specific moments are `capture_frames`.
- **speech-enhancement is standalone** on the original mix — it is not
  chained after audiostem, and as an analysis input it is not recommended
  today (see the rule above).
- **A source missing the needed stream FAILS BY NAME, and the verdict is
  sticky**: the audio workflows (diarization, audio-stem,
  speech-enhancement) fail `NoAudioStream` on a video-only source,
  face-recognition fails `NoVideoStream` on an audio-only one, and
  diarization fails `NoSpeechDetected` when audio carries no usable
  speech. All are terminal CONTENT facts — never resubmit on them. Skip
  the probe when evidence already exists (a workflow-output input, a
  prior probe's `has_audio`/`has_video`, a stream-native container);
  probe unknown video containers before stream-specific submits.

## Harness lifecycle (default) — purpose-built harnesses override THIS section only

The submit tools are asynchronous: submit returns a handle
(`job_id`, `workflow_id`, `output_s3_prefix`), never results. By default,
poll `{target}___get_job_status` every ~30 s (sooner for metadata-only
requests, which finish in seconds) and report `current_state` as progress.
Read results only from a SUCCEEDED status response's `outputs[]` list,
selecting by `label`, and presign via `media-outputs` tools. Resubmitting
identical parameters is free (absorbed into the existing run) — never
"retry" a terminal failure without changing the parameter the `error`
names or supplying a new `idempotency_token`.

A purpose-built harness (jobs panel, project bindings, in-turn probes) may
replace this section with its own lifecycle; the intent table, references,
and cross-tool rules above are canonical and must not be edited downstream
— they are synchronized from the unified-media-toolkit-on-aws monorepo via
the gateway client kit.
