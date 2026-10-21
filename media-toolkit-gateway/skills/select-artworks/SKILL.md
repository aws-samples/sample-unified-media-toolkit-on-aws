---
name: select-artworks
description: Select ranked artwork candidate frames for an analyzed video via the media-toolkit artwork-selection workflow, honoring an optional editorial brief, and present the slate per archetype with presigned frame images. Use when the user asks for artwork, thumbnails, key frames, poster candidates, or a hero image for a video.
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server to be connected and a completed analyze-video run (its state file). Optional but better with a completed cast-resolution run.
---

# Select artworks

Produce ranked, diverse artwork candidate slates (hero close-up,
two-shot, establishing vista) for an analyzed video. Code qualifies
candidates from the shot lattice; one vision pass orders each slate
and flags unusable frames. **The final pick is editorial — always
present the slate, never crown a single winner.**

> ⚠️ ALWAYS go through the media-toolkit tools. Presign
> frames with `media-outputs___get_download_url(s)`. Do NOT use
> `aws s3 cp` / `aws s3 presign` — the gateway is the contract. On a
> gateway auth error, ask the user to refresh the gateway token.

> ⚠️ IDs FLOW, they are never retyped: copy every workflow_id / run prefix VERBATIM from the sibling run's response (`output_s3_prefix`) or the recorded state file — never from memory. A mistyped-but-EXISTING prefix binds against the wrong run and nothing will refuse it; only a nonexistent one fails validation.

## Input

`$ARGUMENTS` is an `ASSET_ID` or shorthand name, optionally followed
by editorial guidance in plain words (e.g.
`select-artworks "Love and Peace" favor dusk exteriors, feature the lead`).

Resolve the run from
`outputs/analyze-video/{ASSET_ID}.state.json`:

- **State file found** → rebuild each sibling run prefix as
  `s3://{proxy_bucket}/micro-workflow/<target>/<workflow_id>` from the
  step's recorded `workflow_id`.
- **No state file** → tell the user to run `analyze-video {ASSET_ID}`
  first, or accept explicit run prefixes if they supply them.

## Steps

1. **Gather prefixes.** Required: `shot_segmentation_s3uri` and
   `shot_metadata_s3uri` (from the state file's steps). Identity:
   - a completed **cast-resolution** run prefix if one exists (ask the
     user, or check prior runs) — real names, best hero/two-shot
     targeting;
   - else pass `face_recognition_s3uri` from the state file — the
     workflow falls back to face-track identity (unnamed people appear
     as anonymous `person_*` labels; the slates still fill).
   Pass `program_structure_s3uri` too when the state file has a PSA
   step (credits/recap exclusion).
2. **Submit** `artwork-selection___select_artwork` with the prefixes.
   - Editorial guidance from `$ARGUMENTS` (or the conversation) goes
     verbatim into `selection_brief` — the workflow translates it into
     its closed vocabulary; it can re-weight but never bypass
     eligibility or the spoiler cut.
   - Leave other knobs at defaults unless asked:
     `candidates_per_archetype` 1–10 (default 4), `archetypes` subset
     (order = priority), `spoiler_cutoff` (default 0.75).
   - Use `idempotency_token: artsel-{ASSET_ID}-{brief-hash-or-none}`
     so a re-ask with identical parameters returns the existing run.
3. **Poll** `artwork-selection___get_job_status` every 30 s (sync runs
   finish in ~1 minute). The status header shows per-archetype counts
   and whether the vision rerank was `applied`.
4. **Fetch the product**: presign and read `artwork_candidates.json`
   from the run's outputs.
5. **Presign the frames** of every usable candidate (skip candidates
   with `unusable`) via `media-outputs___get_download_urls`
   (batch ≤50, `expiry_seconds: 3600`).

## Presenting the slate

Per archetype, IN ORDER (the list is already in final order — vision
order when `rerank.status` is `applied`, metadata order otherwise —
say which, and why, from `rerank.reason`):

- rank, shot id, timestamp, the qualifying evidence in one line
  (camera_shot_type · people · people_count), and the presigned frame
  link (or inline image where the client renders them);
- the judge's one-line `top_reason` for each archetype's winner;
- flagged-unusable candidates listed at the end with their reasons —
  they exist so the editor sees what was rejected, never as picks;
- `identity_source` when it is `face_recognition`: note that people
  are matched by face prominence, not name, and a cast-resolution run
  would enable name-targeted briefs (`must include <name>`).

The slate is deliberately diverse (one candidate per shot, no shot
shared across archetypes) — the runner-up is a genuine alternative.

## Notes

- Re-running with the same token re-presigns nothing by itself; frame
  links expire in ~1 h — re-run step 5 to refresh them.
- An empty hero/two-shot slate almost always means no identity source
  was passed — say so and offer the cast-resolution route.
- A `rerank.status` of `skipped`/`failed` still delivers slates
  (metadata order); report the reason from the product, don't retry
  blindly.
