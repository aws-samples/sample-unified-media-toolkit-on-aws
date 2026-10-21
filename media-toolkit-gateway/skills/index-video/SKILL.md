---
name: index-video
description: Index a completed analyze-video run into hybrid-search so its stories, dialogue, topics, and scenes become searchable — builds timespan documents (news rundown segments, speaker-attributed utterances, topic chapters, scenes with on-screen text and representative frames), indexes them into a collection, verifies with a smoke search. Use when the user asks to index a video, add it to search, make it searchable, or build a search index for analyzed content. To query afterwards use the search-media skill.
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server and a completed analyze-video run (its state file). For news content run program-structure-analysis in news mode (part of analyze-video) and ideally cast-resolution first — speaker-attributed search (says:"Name") needs resolved names.
---

# Index video metadata into hybrid-search

Turn one analyzed asset's metadata into hybrid-search documents: one
document per timespan, indexed per scope, verified by a smoke search.
This file carries the composition only — the indexing contract, query
operators, and field reference live in
[../video-processing/references/hybrid_search_agent.md](../video-processing/references/hybrid_search_agent.md)
(read it first).

> ⚠️ ALWAYS go through the media-toolkit tools — presign + fetch artifacts
> via `media-outputs___get_download_url(s)`, never `aws s3 cp`.

## Input

`$ARGUMENTS` is an `ASSET_ID` (or shorthand name), optionally with a
collection (`index-video "AI News Channel" into news`). Resolve the run from
`outputs/analyze-video/{ASSET_ID}.state.json`; without a state file, tell
the user to run `analyze-video {ASSET_ID}` first. Default collection:
ask once if not given — collections are tenant groupings (`news`,
`episodic`), not per-asset.

## What gets indexed (scope map)

| Source artifact (state file step) | doc_type | Document per | Content |
|---|---|---|---|
| PSA `structure`/`rundown` (news mode) | `segment` | story / sequence | summary text (`extract_entities: true`), span |
| conversation analysis | `topic` | chapter | topic + key events + sentiment text, span |
| diarization `transcript` | `utterance` | speaker turn (sentence) | the quote as `text`; speaker name prepended |
| SME `scenes` + `shots` | `scene` | scene | description + setting + ON-SCREEN TEXT (chyrons — index these verbatim) + brands; faces as `named_entities` (no extraction cost); the scene's representative shot image from shot-segmentation's `debug-grids` as the `image` content item |

Field names inside each artifact: read that workflow's reference /
`describe_tool` output contract rather than guessing — artifact schemas are
versioned there.

Rules that shape the documents:
- **Speaker names**: when the state file records a cast-resolution run, map
  `SPEAKER_NN` → person names via its speaker-map before building
  utterance docs. Without it, index anyway but WARN the user that
  `says:"Name"` queries will not work (speakers are anonymous labels).
- **Cross-talk**: apply the secondary-attribution rule from the
  hybrid-search reference for utterances intersecting `overlap_segments`.
- **Representative frames come from shot-segmentation — do NOT run a
  capture job.** The run already ships per-shot JPEGs: take the
  `debug-grids` output entry's own `uri` (a prefix) and append
  `shot_{NNNNNN}.jpg` — never rebuild the path from the run prefix (the
  grids live in a subdirectory under it, and a guessed path fails the
  whole scene batch with NoSuchKey). Already on the proxy bucket where
  `kind: image` values must live. For each scene pick its MIDDLE shot's image via the scene→shot
  ranges in the SME scenes artifact. They are 112 px tiled composites of
  the shot's selected frames — adequate for look-alike recall; only if a
  user explicitly wants full-resolution stills is a
  `media-processing___capture_frames` job the alternative.
- **`asset_metadata`** `{title, genres}` on every submission (title from
  the state file / user; genres from conversation analysis) — this is what
  makes `@Title` filtering work later.

## Steps

1. Read the state file; presign + fetch the artifacts in the table (skip a
   doc_type whose artifact is missing and say so — partial indexing IS
   indexing; news without PSA rundown loses story-level search).
2. `hybrid-search___list_collections`; `create_collection` if the target
   collection is absent (create-if-absent, immediately searchable;
   `active_scopes` is required — the doc_types from the scope map above).
3. Build the documents and submit via `hybrid-search___index_content` in
   INLINE batches of ≤100 `documents` (the cap) — a 30-minute programme is
   typically 500–900 documents, so expect several submissions; each is its
   own job and that is correct (no idempotency token exists or is needed —
   dedup is per-document). Re-indexing after a fresh analyze-video
   generation: set `refresh: true` on the documents to replace that
   asset's rows.
4. Poll every job; report Σ `prepared`/`skipped`/`indexed` against the
   count sent. Never report an indexing job you have not polled.
5. **Smoke search**: `search_content` with a distinctive quoted phrase
   from one indexed story plus `@{title}` (operator LAST), confirm the hit
   comes back with the right `asset_id` and span, and show it.
6. Record into the state file:
   `indexing: {collection, docs_sent, indexed, skipped, generation, date}`.

## Ending

Report the per-scope document counts, the smoke-search hit, and point the
user at **`search-media`** for querying. If cast-resolution was missing,
repeat the says:-queries warning and offer to run it and re-index
(`refresh: true` makes that cheap and safe).
