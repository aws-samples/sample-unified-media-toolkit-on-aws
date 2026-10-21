---
name: search-media
description: Search indexed media with hybrid-search and present timecoded, evidence-backed hits — "find the moment/story/scene where …", "who said …", "where does X appear", visual look-alike search by image. Composes the query operators (quoted phrases, says:, scope filters, @Title), groups results by asset with highlight snippets, and captures thumbnail stills of the top hits for visual confirmation. Requires content indexed by the index-video skill.
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server and a hybrid-search collection populated by index-video (or equivalent). Thumbnails need the asset's proxy (from the analyze-video state file) to be available.
---

# Search media

Query a hybrid-search collection and answer "where / who / which story"
with timecoded evidence. This file carries the composition only — operator
semantics, result fields, and pitfalls live in
[../video-processing/references/hybrid_search_agent.md](../video-processing/references/hybrid_search_agent.md)
(read it before composing the first query).

## Input

`$ARGUMENTS` is the user's question, optionally naming a collection or an
asset ("search news for …", "in Love and Peace, find …"). If the collection is
unknown, `list_collections` and pick/ask. An image path or s3:// frame
means visual search (`search_image_s3uri` instead of `search_term`).

## Composing the query

Translate the question, don't forward it verbatim:
- Exact wording wanted → `"quoted phrase"`; attribution questions
  ("what did NAME say about X") → `says:"…"` **with** `#utterance`.
- Story/structure questions → `#segment`; topic/sentiment → `#topic`;
  visual/who-is-on-screen → `#scene`.
- Asset-pinned questions → `@Title` LAST in the string (it is greedy).
- After a title-filtered search, CHECK the results' `asset_id`s — foreign
  assets mean the filter did not bind; fix the query rather than re-run.

## Presenting results

1. `search_content`; if empty, read `skipped_indices` first — empty results
   with empty `skipped_indices` genuinely means no matches; a skipped scope
   means the query could not run there (say which).
2. Group hits by asset; show a table per asset:
   `mm:ss–mm:ss · scope · score · highlight` (keep the `<em>` keyword
   highlights as **bold**). Answer the user's question from
   `asset_id` + `start_ms`/`end_ms` + the highlight text — quote evidence,
   never paraphrase beyond what a hit supports.
3. **Visual confirmation** (when the asset has an analyze-video state file
   with a proxy): one `media-processing___capture_frames` job with a marker
   at each top hit's midpoint (≤ ~8 markers), presign via
   `media-outputs___get_download_urls`, and show the stills inline next to
   the table rows.
4. Offer follow-ups, don't auto-run them: refine/narrow the query, play a
   region (presign the proxy and give the timecode), or cut the selected
   hits into a reel with `quick_edit` (segments from the hit spans — see
   the media-processing reference for the cut-list contract).

## Failure paths

- Collection exists but the asset is not in it → suggest
  **`index-video {ASSET_ID}`** (and note `says:"Name"` needs a
  cast-resolution run at index time).
- No collection at all → same, plus `create_collection` happens inside
  index-video; do not create empty collections from here.
- Expired thumbnail/proxy links → re-presign, never hand out `s3://` URIs
  as if they were links.
