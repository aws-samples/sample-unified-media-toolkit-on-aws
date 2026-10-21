# artwork-selection-on-aws

Selects representative artwork frames for a media asset: code filters
and ranks candidate shots from sibling run outputs (shot-segmentation,
shot-metadata-enrichment; cast-resolution recommended), one vision pass
via bedrock-inference orders each pre-qualified slate, and the output
is a ranked DIVERSE slate per archetype — `artwork_candidates.json`
plus the grabbed frames. The model only ever chooses among candidates
code pre-qualified; it never scans the program and never touches time.

## Architecture

Two Lambdas out of ONE flat zip (the gateway dispatcher and the
operation-routed workflow Lambda) and ONE six-state machine:

```
ValidateAndCompilePlan            refuse-before-work: input shape, sibling
                                  artifacts exist and parse, archetype plan
                                  (explicit param, else the genre profile),
                                  prompt-registry categories bound
TranslateBriefToPlanAmendments    HELD waitForTaskToken: one small text
                                  call compiles selection_brief into the
                                  closed vocabulary; no brief = self-
                                  completed skip
FilterRankCandidateShots          pure code: Layer A eligibility + Layer B
                                  archetype queries over the shot lattice
GrabCandidateFrames               JPEGs from shot-segmentation's stored
                                  frame archives (proxy resolution)
RerankFramesByVision              HELD waitForTaskToken: the ONE vision
                                  seat — all slates in one staged record,
                                  the judge orders within each and flags
                                  unusable frames; failures self-complete
                                  as a skip with the reason as data
AssembleArtworkCandidates         artwork_candidates.json + frames + usage
```

Family conventions: every wait is a held `waitForTaskToken` state
completed by bedrock-inference's service-side waker; skips are
self-completed tokens; no Choice states, no polling. ASL lives inline
in `cfnstack.yaml` with a readability copy in `definitions.json` — edit
BOTH, the build fails on drift. Sibling gateway ARNs are explicit stack
parameters, never `Fn::ImportValue`. Both sibling ARNs are REQUIRED:
the vision rerank is the validated core of the product (metadata-only
ranking ships bad frames — measured on the prototype: the vision pass
changed most archetypes' top pick and flagged frames metadata ranked
first, eyes-closed and motion-blurred among them), so a deployment
without bedrock-inference would be a worse product masquerading as a
degraded mode. The RUNTIME still degrades per run: prompt or submission
failures skip with a named reason and Assemble ships the metadata order
flagged `rerank: skipped`.

## Gateway tool contract

Three tools: `select_artwork` / `get_job_status` / `describe_tool`.
`workflow_id = uuid5(request + tool_version [+ idempotency_token])`;
an idempotent resubmit returns the existing run. Output contract in
`gateway_tool/output_contract.json`.

Sibling run prefixes (`<workflow>_s3uri`, family convention):

| param | req? | why |
|---|---|---|
| shot_segmentation_s3uri | required | candidate universe: shot bounds, frame archives, stability stats |
| shot_metadata_s3uri | required | the semantic layer; without it ranking is entropy+motion only |
| cast_resolution_s3uri | recommended | top-cast-by-screen-time → hero/two-shot targeting + derived concentration |
| face_recognition_s3uri | optional | identity FALLBACK when cast is absent: top faces by tracked screen time, TEMPORAL track join (face clusters never string-match shot-metadata people), anonymous `person_*` labels for unnamed clusters — artwork needs "the same prominent person", not a name; must-include no-ops on anonymous ids. Cast wins when both present |
| program_structure_s3uri | optional | credits/recap/cold-open spans excluded IN ADDITION to the flat spoiler cut |

Steering parameters (defaults live in ONE home, `lib/wf_common.py`):

| param | type / default | notes |
|---|---|---|
| selection_brief | free text, optional | the one editorial seat (design record §5) |
| genre | string, optional | selects the Layer-B profile row; neutral profile when absent; never inferred |
| archetypes | list from the closed archetype vocabulary, optional | explicit subset; default = profile-driven mix |
| candidates_per_archetype | int, default 4 (bounded 1–10) | final slate size per archetype |
| spoiler_cutoff | float 0–1, default 0.75 | flat positional cut for reveal avoidance; program-structure spans excluded in addition when supplied |
| model_chain | 1–4 Bedrock model ids, primary first | applies to BOTH inference rounds (brief translation + vision rerank); ids must be registered with the bedrock-inference deployment |
| use_batch | bool, default false | sync by default (two single-record rounds); true batch-prices them |
| idempotency_token | optional | same token + same parameters returns the existing run |

Deliberately NOT parameters: per-field weight knobs (the brief's job),
cast concentration (derived, see §6), aspect ratios (a future version,
needs face boxes), model selection beyond the chain (prompt-registry +
bedrock-inference own it). Full-resolution source extraction
(`full_res` / `asset_s3uri`) is designed but NOT YET SUPPORTED — the
gateway refuses it; slates serve proxy-resolution frames from
shot-segmentation's stored archives.

## Build & deploy

```
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]
bash ./build-deploy-s3-dist.sh --build-only [--version VERSION]
bash ./create_update_stack.sh ...
```

No docker, no CodeBuild, no ECR: pure boto3+stdlib, one flat zip
carrying both handlers plus every `lib/` and `states/` module at the
zip root. BUMP `.version` ON EVERY DEPLOY — S3Key carries the version
and CloudFormation detects key changes, never content changes, so an
unbumped redeploy silently leaves old code running. Tests gate
packaging, and the build asserts internal docs are excluded from every
artifact.

## Tests

```
python3 -m unittest discover -s test -p 'test_*.py'
```

Hermetic (no AWS); includes the ASL no-drift pin between
`cfnstack.yaml` and `definitions.json`. Fixtures are invented.

## Gotchas

- Never commit real-content outputs: per-asset results and grabbed
  frames are customer-identifying and copyrighted (`.gitignore` guards
  `results/` and image files). Test fixtures are invented.
- Candidate `shot_id`s and cast names are only meaningful against the
  exact sibling runs they came from — the plan pins that provenance.
- An empty or thin slate is self-explaining: `eligibility` in the
  product carries per-clause rejection counts plus
  `eligible_clean`/`eligible_overlaid`, so "no candidates" always says
  why.
- `poc/recommend.py` and `poc/rerank_vision.py` are the validated
  prototypes (filter/rank and the rerank rubric that seeded the
  prompt-registry categories), kept for reference — not shipped code.

## Design record

Decisions that shape the workflow, with the why; code and comments cite
these sections by number.

1. **A separate micro-workflow, not a skill or a sibling's operation.**
   The selection criteria must be steerable per request — a free-text
   `selection_brief` — which is versioned, caller-facing judgment: the
   workflow-family + prompt-registry pattern. It consumes sibling runs
   by pointer (the cast-resolution shape); hybrid-search is
   deliberately NOT a dependency — the brief is served entirely by the
   plan state's translation into the closed vocabulary, not by
   free-text retrieval.

2. **Three-layer split.** Layer A eligibility: a deterministic,
   genre-independent hard filter (slates/graphics inserts, burned-in
   text, screen-in-screen depictions, motion/coherence floors,
   credits+recap territory, minimum shot duration, re-aired recurrence)
   — no taste, same code for every title. Layer B archetype profiles:
   DECLARATIVE DATA, the only place genre lives — a profile is a table
   of which archetypes to hunt, how many of each, and which evidence
   fields to weight. Adding a genre is a row, not a code path; profiles
   encode structural claims about composition, never scalars fitted to
   a title. Layer C: the ONE vision pass — it orders and explains
   pre-qualified candidates; it cannot invent a frame, pick a slate
   card, or violate the spoiler policy.

3. **Emit a slate, not a pick.** Click-through is unmeasurable here, so
   a single "best frame" is unfalsifiable; the deliverable is a ranked
   DIVERSE set with the qualifying evidence per candidate. Diversity is
   a first-class constraint, not a tiebreak: no two candidates from one
   shot, no shot shared across archetypes, spread across the timeline,
   distinct compositions and people.

4. **The vision rerank earns its seat.** Sharpness, motion blur,
   eyes-open, composition — the things the metadata genuinely does not
   carry (entropy and lighting are proxies, not judgments). On the
   validated prototype the vision winners reproduced the frames a human
   editor had independently curated from the same candidate pools,
   while metadata-only ranking shipped unusable frames. Output: ranked
   frame ids plus a one-line reason each, id-referenced, no timestamps.

5. **Editorial subjectivity gets exactly one seat**: `selection_brief`,
   compiled by the brief-translation call into the closed vocabulary
   (archetype weights, must-include people, must-avoid content). The
   brief can only tighten/re-weight within that vocabulary — it never
   invents fields and it NEVER reaches Layer A: an editor cannot brief
   their way into a credit roll. One known divergence class is by
   design: a story-value pick a human chooses for narrative can carry
   aesthetic flags — the brief arbitrates story-vs-aesthetics.

6. **Cast concentration is DERIVED, never asked.** cast-resolution's
   screen-time distribution decides star-vehicle (weight hero
   close-ups) vs ensemble (weight two-shots/groups) — self-correcting
   when the marketing lead isn't the screen-time lead. The hero
   archetype hunts the top cast members round-robin, so a two-anchor
   desk gets both anchors in the hero slate; on a true star vehicle the
   supporting cast rarely has qualifying solo close-ups, so the lead
   fills the slate anyway — no genre branch. Genre is the opposite:
   ACCEPTED as a parameter (the catalogue knows it), never inferred;
   neutral profile when absent.

7. **Two-tier eligibility for overlaid footage.** `shot_kind` admits
   `normal` AND `graphics_overlay` — real footage wearing channel
   furniture (ticker, bug, locator) is a legitimate artwork source on
   content classes that have no clean plates (news), but it ranks
   strictly BELOW the clean tier and carries its burden verbatim
   (`shot_kind`, `texts_on_screen`) so the burned-in graphics are
   reviewable. Content with clean plates behaves exactly as before.

8. **Identity fallback.** When cast-resolution is absent,
   face-recognition's top faces by tracked screen time stand in; the
   join to shots is temporal (face tracks never string-match
   shot-metadata people), unnamed clusters ride under anonymous
   `person_*` labels, and `identity_source` in the product names which
   path produced the identities. Acceptance for selection quality is
   editorial agreement on a labelled set, stated with the n — never a
   click metric.
