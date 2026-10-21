---
name: create-audio-description
description: Generate an audio-description narration for an analyzed video via the media-toolkit audio-description workflow (script + synthesized narration + fit/repair), render the ducked AD mix with media-processing, and build the AD review bench HTML page (original/described audio toggle, cue ledger, violations). Use when the user asks for audio description, AD, described video, or an accessibility narration track.
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server to be connected, a completed analyze-video run (its state file), and a registered text-to-speech voice. LONG-RUNNING (roughly 15-60 minutes).
---

# Create audio description

Write, synthesize, fit and repair an AD script for a programme, then
render the final ducked, loudness-normalized AD mix. Two chained
workflows: `audio-description___generate_ad` (script + narration
takes + render manifest) and `media-processing___process_media`
(the actual mix — generate_ad renders NO audio itself).

> ⚠️ ALWAYS go through the media-toolkit tools. Presign and
> download outputs via `media-outputs___get_download_url(s)`. Do NOT
> use `aws s3 cp` / `aws s3 presign` — the gateway is the contract.
> On a gateway auth error, ask the user to refresh the gateway token.

Per-tool parameter and output-label detail lives in the workflow references
([audio-description](../video-processing/references/audio_description_agent.md),
[media-processing](../video-processing/references/media_processing_agent.md),
[text-to-speech](../video-processing/references/tts_agent.md)) — this file
carries only the chain's wiring.

> ⚠️ IDs FLOW, they are never retyped: copy every workflow_id / run prefix VERBATIM from the sibling run's response (`output_s3_prefix`) or the recorded state file — never from memory. A mistyped-but-EXISTING prefix binds against the wrong run and nothing will refuse it; only a nonexistent one fails validation.

## Input

`$ARGUMENTS` is an `ASSET_ID` or shorthand name, optionally with a
voice hint (e.g. `create-audio-description "Love and Peace" with a warm
female voice`).

Resolve the run from
`outputs/analyze-video/{ASSET_ID}.state.json`:

- **State file found** → rebuild each sibling run prefix as
  `s3://{proxy_bucket}/micro-workflow/<target>/<workflow_id>` from the
  step's recorded `workflow_id`.
- **No state file** → tell the user to run `analyze-video {ASSET_ID}`
  first, or accept explicit run prefixes.

## Steps

1. **Pick the voice — never guess.** Call
   `text-to-speech___list_voice_collection` and either match the
   user's hint or show the catalog and ask. The chosen `voice_id` is
   REQUIRED by generate_ad.
2. **Gather prefixes** from the state file: the four sibling runs
   (`shot_segmentation_s3uri`, `shot_metadata_s3uri`,
   `speech_diarization_s3uri`, `conversation_analysis_s3uri`), a
   people source (`cast_resolution_s3uri` when a cast run exists,
   else `face_recognition_s3uri`), and `audiostem_s3uri` when
   present.
3. **Set expectations BEFORE submitting**: generate_ad runs roughly
   **15–60 minutes** (model + TTS spend; the machine sleeps on push
   callbacks, polling costs nothing). Tell the user, then submit
   `audio-description___generate_ad` with the prefixes + `voice` +
   `idempotency_token: ad-{ASSET_ID}-{voice_id}` (a retry with the
   same token returns the existing run instead of paying twice).
   Optional passthroughs only when the user asks: `language`,
   `title`, `sung_vocal_policy`.
4. **Poll** `audio-description___get_job_status` every 60 s. Report
   the `current_state` as it advances (write → synthesize → fit →
   repair) so the wait is legible.
5. **On SUCCEEDED, present the SCRIPT FIRST**: presign and read the
   AD script artifact, summarize cue count / coverage, and link the
   script for review. The script is the reviewable product; offer the
   mix as the next step rather than rendering unasked.
6. **Render the mix when the user wants it**: call
   `media-processing___process_media` on the programme SOURCE
   (`source_s3_uri` from the state file) with
   `output_groups=[{"format": "ad_mix", "render_manifest_s3uri":
   <the render-manifest output uri from step 5>}]` — this is the
   documented chain; a `mix` parameter on generate_ad is refused
   with exactly this instruction. Poll it, then presign the mixed
   track and hand over the link.
7. **Build the AD review page** — read
   [references/ad_review_data.md](references/ad_review_data.md)
   BEFORE writing. COPY `references/template_ad_review.html`
   verbatim, build the DATA object per the contract (the joined view
   document + presigned video/mix + the VTT strings), substitute the
   tokens, and write to
   `outputs/create-audio-description/{ASSET_ID}.html`.
   Never author or restyle page chrome. Validate (DATA parses, no
   `__` token survives), then open it:
   `open -a "Google Chrome" <file>` (macOS). The page works without
   the mix (audio toggle reports "mix unavailable") — build it right
   after step 5 so script review happens ON the page, and re-run this
   step after step 6 to add the mix toggle and refresh expired links.

## Notes

- generate_ad is the most expensive skill in this set (model + TTS
  spend per run) — that is WHY the idempotency token matters and why
  the script review sits between generation and mix.
- A people source is required: without cast-resolution, face
  recognition serves — but named character references in the script
  are only as good as the people source. For scripted content,
  running cast-resolution first measurably improves naming.
- If the status reports a skipped/degraded stage, relay the named
  reason from the outputs — the workflow degrades deliberately,
  never silently.
