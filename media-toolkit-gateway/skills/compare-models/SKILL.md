---
name: compare-models
description: Run conversation-analysis or shot-metadata-enrichment on ONE asset under 2+ models and build a side-by-side comparison HTML page — scoreboard (cost/tokens/wall/coverage/stops), per-model boundary lanes with a divergence heat strip, a shots-ranked-by-disagreement list, and a per-shot field matrix (all models as columns, divergent cells highlighted). Use when the user asks to compare models, run a model bake-off, or evaluate N models on CA/SME output. Agreement evidence only — the page never declares a winner.
---

# Compare models (CA / SME)

Run the SAME asset through one LLM workflow under N models (N ≥ 2), join
the outputs by shot_id (SME) or segment span (CA), and build one
self-contained comparison page. Maintainer skill (repo-local; not in the
client kit yet).

> ⚠️ All submits/polls through the media-toolkit gateway; presign the
> proxy via `media-outputs___get_download_url`. IDs flow — copy
> workflow_ids verbatim from responses.

## Inputs

- Asset (name or s3:// uri) with a completed analyze-video run — the
  comparison REQUIRES identical upstream inputs (same shots run, same
  transcript, same face-recognition run) for every model, or the join is
  meaningless. Verify shot_id ranges match across runs before building.
- Models: 2+ registry model ids (`bedrock-inference___list_models`).
  Pick a baseline (default `anthropic.claude-sonnet-4-6`, the validated
  seat) — it renders as the first column/lane.
- Workflow: `shot-metadata-enrichment` (shot-level matrix) or
  `conversation-analysis` (segment-level; builder variant pending).

## Steps

1. For each model: submit the workflow with `model_chain: [model]`,
   `use_batch` per the model's registry capability (sonnet-4-6 batches;
   5.5s and GPT are sync-only), the SAME upstream input uris, and
   `idempotency_token: compare-{asset}-{model}-{gen}`. Re-running the
   skill with the same tokens absorbs idempotently at zero cost.
2. Poll to SUCCEEDED; fetch per run: `shot_metadata.jsonl`,
   `scenes.json`, `usage_tracking.json`. Also fetch once:
   shot-segmentation's `shots.json` (timings), its `shot_images/` grids
   (thumbnails), the diarized `final_transcript.vtt` (subtitles), and
   SME wall times (Step Functions describe-execution).
3. Write `config.json` (see `references/build_compare.py` header):
   models in display order with baseline FIRST, run dirs, wall_sec,
   paths, and a `media-outputs`-presigned proxy URL.
4. `python3 references/build_compare.py config.json out.html` — joins,
   normalizes (lists compared as sets; dict lists reduced to names),
   scores divergence per shot (count of structured fields with >1
   distinct value; `description` is never scored; a model's missing
   shot is an honest gap, not divergence), embeds thumbnails + VTT.
5. Open the page; verify: scoreboard rows match usage files, boundary
   tick counts per lane equal each run's scene count, both themes draw.

## Page semantics (do not blur)

- Agreement/divergence EVIDENCE, never a verdict or ranking.
- `—` = the model did not answer that shot (gap) — excluded from
  divergence; coverage shows it in the scoreboard instead.
- Presigned URLs expire — regenerate the page, never hand-edit it.
- Divergence weights are uniform across fields (v1); list-vocabulary
  fields (objects/actions) dominate raw counts — the ranked list is for
  triage, not scoring.

## Files

- `references/compare_template.html` — the page (house design system;
  one meaning for color: the warning hue marks divergence).
- `references/build_compare.py` — deterministic builder; DATA contract
  in its docstring.
