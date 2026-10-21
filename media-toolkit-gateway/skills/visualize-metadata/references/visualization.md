# Visualization build contract

The page UI is FROZEN in ONE template beside this file:
`template_unified.html`. **Start from a copy of it; never rebuild or
restyle the page** — the template IS the approved format, and every run
must render identically. Your entire job is building the DATA object
and substituting the tokens.

One chassis, two content modes, both rendered by the same file:

| `DATA.mode` | hero tab / hero lanes | chosen when |
|---|---|---|
| `news` | Rundown (stories) / STORIES + CONVERSATION ANALYSIS | PSA `manifest.json` says `"mode": "news"` |
| `narrative` | Scenes + Chapters / SCENES + CHAPTERS | PSA says `"mode": "narrative"`, or PSA absent |

**PSA `manifest.mode` is the mode authority.** Do NOT infer the mode
from news fields on CA segments or SME docs — since alpha.20260918e
PSA derives news structure itself from default prompt runs, so those
fields are absent on runs the page must still treat as news.

Every OTHER module is presence-driven, not mode-driven: omit a DATA
key and its tab / timeline group simply does not render. Never
fabricate an empty document for a run that did not happen.

## Tokens

| token | replace with |
|---|---|
| `/*__DATA__*/` | the DATA JSON (one `json.dumps`, no pretty-print) inside the `<script type="application/json" id="DATA">` block |
| `__ASSET__` | display name of the asset (title + header) |

Substitute with exact string replacement. Validate afterwards: the
embedded JSON must parse, and no `__ASSET__` / `/*__DATA__*/` token may
survive.

## DATA contract (one shape, both modes)

Top-level keys (all embedded; only `media.*` values are presigned):

```
asset str · asset_label str · mode "news"|"narrative"
media    {video, speech?, music?, sfx?, faces: {<face key>: url},
          artwork?: {<frame basename>: url}}   presigned urls
rundown  the FUSED spine (see fusion below):
         {duration_sec, people, speakers, credits, program_brands,
          stories?}           — stories only in news mode
conversation_analysis · final_transcript · shots · manifest ·
recurrence · scenes          each output document, verbatim
envelopes  {original?, speech?, music?, sfx?: {bin_sec, rms[0..1]}}
           — stem_waveforms.json downsampled to ~0.5 s bins,
           peak-normalized per stem
usage_summary  [{workflow, cost, wall_sec, detail?}] rows from each
           run's usage output (total_estimated_cost /
           pipeline_wall_sec, or summed inference_runs)
shot_metadata_jsonl  str — the raw jsonl text (the page parses lines)
vtt      str — the diarization caption output verbatim (subtitle track)
structure?  PSA structure.json verbatim (markers / break_candidates /
           sequences) → the PROGRAM STRUCTURE timeline group
artwork?    artwork_candidates.json verbatim (slates per archetype with
           vision ranking) → the Artwork tab + ARTWORK lane; presign
           each candidate's `frame` uri into media.artwork keyed by
           the frame BASENAME
```

`manifest` must carry `fps` (compute `actual_frame_count /
actual_duration_sec` from shots.json — never hardcode) and `stems_s3`
({speech/music/sfx: truthy} presence flags for the stem selector).

## The rundown fusion (builder's job — no workflow emits this doc)

PSA's news rundown carries `stories` + `shot_roles` only; the page
needs the fused spine. Build it:

- **stories** (news only): PSA rundown stories as the base; overlay
  each with its max-time-overlap CA segment (named_entities,
  sentiment, tone, conversation_type); enrich from SME docs in the
  story span when the run carried news fields (headline_banners /
  locators from supers, presentation_verdict from live-bug + `EARLIER`
  evidence × interaction — with only interaction available the verdict
  degrades to live_unbugged / produced_or_studio).
- **people** (both modes): cast-resolution persons (named, PLUS any
  person carrying a critic_review flag — machine-doubted persons are
  exactly what a review page exists for; an unnamed flagged person
  displays its first aliases) —
  {name, person_type, character_name?, role?, name_mechanisms,
  aliases, face, screen_time_sec, speech_time_sec, speaker_ids,
  appearances[[s,e],…], review_flag?: {field, reason}}. `review_flag`
  comes from persons.json critic_review (tooth-2, report-only): the
  template outlines the tile red with a ⚠ mark and puts the cited
  reason in the hover title — reviews start where the machine's
  suspicion ends. `face` = the face-recognition rep_face_key
  reached via the person's face_ids; presign it into media.faces.
  `character_name` AND `role` come from cast-resolution VERBATIM —
  never join credits pairings and never re-derive roles in the builder
  (the POC role ladder is retired 2026-09-23; an absent role renders
  no badge). News mode EXCLUDES interstitial-only people: a person
  whose appearance time sits majority outside the story spans lives in
  the commercial pods, not the programme. Aggregate duplicate names
  (screen/speech sums, appearance union). Fallback when
  cast-resolution is absent: FR face_metadata people (celebrity names
  + confidence); omit the key entirely when face-recognition also
  skipped.
- **speakers** (both modes): one turn per transcript sentence
  {speaker, start, end} + {name, person_type} from cast-resolution's
  speaker_map when present.
- **credits** (both modes): cast-resolution credits.json blocks →
  {start_sec, end_sec, kinds, lines[≤12 deduped]}.
- **program_brands** (news): program_brand supers ranked by 10-min
  bucket spread; empty list when the SME run carried no supers.
- **duration_sec**: shots.json actual_duration_sec.

## Build rules

- **Embed every metadata document inline; presign only media** — a
  `file://` page cannot fetch JSON from S3 (CORS). Face crops and
  artwork frames are `<img>` on presigned urls.
- Presign with `media-outputs___get_download_urls` (batch ≤50,
  `expiry_seconds: 3600`); objects only, never prefix entries.
- A missing input drops out by omission — leave its DATA key absent
  (the chrome degrades silently); never fabricate an empty document.
- Media links expire in ~1 hour; re-running the skill re-presigns and
  rewrites the page — that IS the refresh mechanism.
