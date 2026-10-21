---
name: review-compliance
description: Build the content-compliance review page for an analyzed video — visual depicted evidence, spoken discourse facts, and an optional caller-supplied word-list scan, on a synced-video timeline with a findings list and accessibility table. Evidence only, never a verdict. Use when the user asks for content moderation, compliance review, ratings evidence, or "what's objectionable in this video".
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server to be connected and a completed analyze-video run (its state file). The word list and any rating rulebook come from the caller — nothing is built in.
---

# Review compliance

Turn a completed `analyze-video` run into ONE self-contained
compliance review page: every flagged moment as a click-to-seek
finding (visual · spoken · word-list layers), a category-lane
timeline with a filmstrip, and the always-on accessibility table.
**The page shows evidence, never a verdict** — classification against
a rulebook is the reviewer's job, and the page says so on its face.

> ⚠️ ALWAYS go through the media-toolkit tools. Presign media and
> fetch every output JSON via `media-outputs___get_download_url(s)`.
> Do NOT use `aws s3 cp` / `aws s3 presign` — the gateway is the
> contract. On a gateway auth error, ask the user to refresh the
> gateway token.

> ⚠️ IDs FLOW, they are never retyped: copy every workflow_id / run prefix VERBATIM from the sibling run's response (`output_s3_prefix`) or the recorded state file — never from memory. A mistyped-but-EXISTING prefix binds against the wrong run and nothing will refuse it; only a nonexistent one fails validation.

## Input

`$ARGUMENTS` is an `ASSET_ID` or shorthand name, optionally with:

- a **word list** — a local file path or inline terms. NEVER invent
  or assume one: word lists are market/rating-specific and the
  caller owns them. No list → the word-list layer simply does not
  render (and is NOT listed in `attested_none`).
- a **rating profile/rulebook name** — free text for the header
  pills. Without one, the rating block reads "not assessed /
  evidence review only — no rulebook supplied".

Resolve the run from `outputs/analyze-video/{ASSET_ID}.state.json`
(rebuild each prefix as
`s3://{proxy_bucket}/micro-workflow/<target>/<workflow_id>`).
No state file → tell the user to run `analyze-video {ASSET_ID}`
first, or accept explicit run prefixes.

## Steps

1. **Fetch evidence FRESH** (presign + read each artifact):
   - `shot_metadata.jsonl` (shot-metadata-enrichment) — each shot's
     `depicted` entries are the VISUAL layer; timestamps from the
     matching `shots.json` (shot-segmentation) entry.
   - the conversation analysis document — `audio_segments[]`'s
     `discourse_facts` are the SPOKEN layer, verbatim.
   - `final_transcript.json` (speech-diarization) — the word-list
     scan source AND the inline VTT subtitle track. Its top-level
     `overlap_segments` (cross-talk spans naming both live voices —
     SUGGESTIVE, not ratings-grade: identity pairing in cross-talk is
     error-prone, more so outside English) feed the
     attribution-uncertainty badge below.
   - **cast-resolution speaker map, when a cast run exists** — name
     the speaker of every spoken/word-list finding
     (`src: "SPEAKER_03 (Character Name, on_camera_speaker)"`);
     without it, keep the raw SPEAKER_ID.
   - **program-structure `structure.json`, when a PSA run exists** —
     suppress word-scan hits inside bars/tone and slate spans, and
     note credits/textless regions in `meta.note` so a reviewer
     knows those spans carry non-program material.
2. **Build the DATA object** exactly per
   [references/page_contract.md](references/page_contract.md) — read
   it BEFORE writing. Load-bearing semantics:
   - `findings[]`: visual entries carry the shot span and
     `src: "shot N"`; spoken entries carry the verbatim quote,
     **re-verified against the transcript text before setting
     `verified: true`**; word-list entries only when a list was
     supplied.
   - **Attribution uncertainty**: a spoken/word-list finding whose
     time falls inside an `overlap_segments` span involving its
     speaker gets " — attribution uncertain: cross-talk with
     <other speaker>" appended to its `evidence` text (and prefer the
     conversation analysis's per-fact `attribution_overlapped_with`
     when present — same fact, already joined). Never move the
     finding to the other speaker: the overlap data is suggestive,
     the badge records ambiguity, the reviewer decides.
   - `categories` = only layers/categories with actual findings;
     `attested_none` = only layers that were ASKED and returned
     empty. An absent layer means *not assessed* — never claim
     "none" for it.
   - `rating`: the caller's rulebook context, or the explicit
     not-assessed block (the template renders these pills
     unconditionally — always provide the object).
   - `meta.note` carries the standing caveats: evidence-not-verdict,
     frame sampling can miss sub-second events, plus any PSA region
     notes.
3. **Filmstrip thumbnails**: middle frame per shot from
   shot-segmentation's frame archives, downscaled to ~40 px height
   as base64 data-URIs (macOS: `sips -Z 40`); a shot's `thumb` may
   be null when an archive is unreadable — never fail the page.
4. **Presign ONLY the video** — a proxy WITH an audio track
   (media-processing's transcoded proxy; shot-segmentation proxies
   are video-only by design). Everything else is embedded inline.
5. **Assemble from the FROZEN template**: COPY
   [references/template_compliance.html](references/template_compliance.html)
   verbatim, replace the single `/*__DATA__*/` marker, write to
   `outputs/review-compliance/{ASSET_ID}.html`. Never author or
   restyle page chrome — a page change is a reviewed template edit.
6. **Validate** (DATA parses, no marker survives, `file://` works)
   and open: `open -a "Google Chrome" <file>` (macOS). Media links
   expire in ~1 h; re-run this skill to refresh.

## Presenting the result

Summarize in chat AFTER the page opens: finding counts per category
and layer, the attested-none layers, whether the word-list layer ran,
and any PSA region caveats — then point at the page for the actual
review. Never rank severity and never conclude "compliant" /
"non-compliant": that call belongs to the reviewer and their
rulebook.

## Notes

- This skill runs NO workflows and pays only for presigns — it is a
  pure consumption pattern over the analyze-video outputs.
- Re-running is the refresh mechanism (same as visualize-metadata).
- If conversation analysis or SME was skipped in the source run, the
  corresponding layer is absent (not assessed) — say so in chat and
  in `meta.note`, and never fabricate an empty layer.
