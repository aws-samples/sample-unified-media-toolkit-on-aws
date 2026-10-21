# program-structure-analysis-on-aws

Layer-2 CODE-ONLY rollup over sibling runs: CPL-style markers, graded
break candidates, narrative sequences, and (news mode) the
industrialized rundown — shot roles + stories with three start times +
first-class interstitials. No model calls anywhere: every judgment is
Layer-1 or conversation-analysis evidence, assembled and annotated.
Recomputable for free against already-paid inference.

## Architecture

Two Lambdas out of ONE flat zip (the gateway dispatcher and the
operation-routed workflow Lambda) and ONE two-state machine with NO
external waits:

```
ValidateAndPlan   refuse-before-work: inputs resolve, mode selection
                  (design record §4), the plan says what evidence is
                  present and what that disables
Assemble          the whole rollup in one pass — markers, break
                  candidates, sequences, gap map, roles, stories
```

Assemble is deliberately NOT chunked: the family chunking rule targets
long-running legs, and this one is seconds over a few MB of JSON (the
inputs are bounded by episode size, not catalogue size); a Map here
would be ceremony. A STANDALONE child: no sibling gateway-ARN
parameters — the inputs are s3 run pointers, read straight from the
proxy bucket. ASL lives inline in `cfnstack.yaml` with a readability
copy in `definitions.json` — edit BOTH, the build fails on drift.

## Gateway tool contract

Three tools: `analyze_program_structure` / `get_job_status` /
`describe_tool`. `workflow_id = uuid5(request + tool_version
[+ idempotency_token])`; an idempotent resubmit returns the existing
run. Output contract in `gateway_tool/output_contract.json`. The
dispatcher is THIN (identity only); PSA makes no model calls, so the
prompt-bound siblings' identity caveats do not apply — same inputs
really are the same work, and re-running after a SIBLING re-run means
new input uris, which re-identify naturally.

| param | req? | unlocks |
|---|---|---|
| shot_segmentation_s3uri | required | the shot lattice + recurrence |
| shot_metadata_s3uri | required | shot_kind, news_setting, scenes |
| conversation_analysis_s3uri | optional | the news rundown (stories) + mode auto-resolution |
| transcript_s3uri | optional | gap_map + orientation |
| audiostem_s3uri | optional | tone/silence/music corroboration |
| mode | optional | `auto` \| `narrative` \| `news`; absent → auto (decided from the evidence, recorded in `mode_evidence`) |
| idempotency_token | optional | same token + same parameters returns the existing run |

Products: `structure.json` (CPL markers, graded break candidates,
narrative sequences, gap map/orientation when transcript evidence is
present; `acts[]` deliberately ABSENT on masters — §8) and
`rundown.json` (news mode: shot roles + stories with three timestamps +
first-class interstitials).

## Build & deploy

```
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]
bash ./build-deploy-s3-dist.sh --build-only [--version VERSION]
bash ./create_update_stack.sh ...
```

No docker, no CodeBuild, no ECR: one flat zip, both handlers at the zip
root. BUMP `.version` ON EVERY DEPLOY — S3Key carries the version;
CloudFormation detects key changes, never content changes. Tests gate
packaging, and the build asserts internal docs are excluded from every
artifact.

## Tests

```
python3 -m unittest discover -s test -p 'test_*.py'
```

Hermetic (no AWS); includes the ASL no-drift pin between
`cfnstack.yaml` and `definitions.json`. Fixtures are invented.

## Gotchas

- News mode needs conversation-analysis output carrying the news-fork
  fields (news_form/slug/sig_out on segments) and enrichment's
  `news_setting` shot fields; without them the rundown is refused, not
  guessed.
- Four vocabularies at three levels around bars/slates, each
  deliberate, joined HERE and nowhere else: shot-level `bars`/`slate`
  (enrichment shot_kind) ↔ segment types `color_bars_and_tone` /
  `technical_slate` ↔ detector `known_type` values ↔ the fused span
  verdict `bars_and_tone`. (`bars`, not `bars_tone`, at shot level: a
  vision model cannot hear a tone, and the near-identical spelling
  against `bars_and_tone` was a measured review trap.) audiostem
  reports measured tones only; naming a tone span "bars-and-tone" is
  this layer's join of `tones[]` against the visual bars/slate
  evidence — evidence vs verdict, as everywhere in the family.
- Sequences are EVIDENCE-GRADED output (every join carries its via);
  consumers may ignore them. Music-saturated genres with a
  protagonist in every scene can still over-chain — the missing
  counter-force is a boundary penalty that cannot be calibrated
  without a chapters-style ground truth, so it is deliberately not
  tuned (§2).
- The soft-window (untranscribed-walla) predicate is mechanically
  sound but RARE on scripted content; it is expected to earn its keep
  on unscripted/news crowd material.

## Design record

Decisions that shape the workflow, with the why; code and comments cite
these sections by number.

1. **Layer-2, code-only, by pointer.** Models judge frames and language
   once (shot-metadata-enrichment, conversation-analysis); this layer
   assembles and annotates spans — runs, positions and joins. It reads
   sibling outputs by pointer, never copies, and that is why it is
   separate from enrichment (which runs in parallel with CA and must
   not gain a serializing dependency). A marker that cannot be
   assembled from Layer-1 evidence is fixed by a new ENRICHMENT field,
   never by an LLM inside PSA. Reading what a card says is
   frame-inspection's job — delegated, never embedded.

2. **Sequences: narrative movements as scene threads.** Every programme
   has two grouping axes — visual continuity (place, time, people) and
   discourse (what is said about what). Scripted narrative groups by
   visual continuity first. A sequence is a chain of scenes forming one
   narrative movement — the chapter-like unit between scene and act;
   parallel storylines CROSS-CUT, so sequences are THREADS of scenes
   that interleave. Assembly is greedy in time order over boolean
   continuity edges: music bridges (one continuous music span crossing
   a scene cut is the editor's own "same movement" mark — the strongest
   single cue), shared setting or a recurring shot group spanning both
   scenes, and cast overlap. Two structural rules came from
   measurement: a music bridge must be ANCHORED (the cue started during
   the earlier scene — an inherited bed is not an editorial mark), and
   cross-cut re-entry merges only past a SHORT intervening sequence (a
   true cross-cut alternates quickly). Micro-scenes ride as connective
   tissue. Further tuning is gated on a chapters-style ground truth,
   not taste.

3. **News shot ROLES, not scenes.** News has no scenes in the film
   sense: studio vs field vs graphics is a property of a SHOT. The
   primary evidence already ships (enrichment's `news_setting`);
   this layer normalizes it and corroborates the studio roles with
   anchor-set recurrence from shot-segmentation (a locked-off anchor
   camera is a very large recurring group; furniture is not).

4. **Mode selection.** Explicit `mode` param wins; else CA's
   `overall_content_type` resolves it; else `narrative`. The chosen
   mode and WHY land in the plan — evidence, not verdict. The two modes
   share one generalization: **language LOCATES a boundary; picture
   TIMES it** — a discourse boundary is always snapped to the nearest
   eligible visual seam, and a visual seam with no discourse support is
   structure (act/pod), not story.

5. **Products are annotated spans over the shot lattice** — inclusive
   shot-id ranges, frame-accurate and gapless by construction. The
   marker catalogue is CPL-style (SMPTE IMF): blackframes, technical
   slate, bars+tone, opening/main title, interior title cards, end
   credits + logos, textless tail. Internal `_plan/` is unstable; the
   manifest's input pointers are the provenance (no inference runs to
   join).

6. **Break candidates are graded, never silent.** An insertion point
   must NEVER split continuous dialogue — a diarization turn spanning
   the seam is a HARD VETO, as is a protected span (titles, credits,
   recap). Surviving seams are graded by converging evidence (scene +
   topic alignment, speech gaps, black/bumper adjacency, music
   bridging); the grade and its evidence ship with each candidate. The
   shape maps onto SCTE-35 splice points. No LLM anywhere — every
   field is a lattice or activity-span check.

7. **News stories: the promo-absorption rules.** CA's discourse
   segments are joined to the video with three measured boundary
   rules: (a) START SNAP — a rundown item starts at the DIRECTOR'S
   CUT, so each segment's start snaps back to the start of the shot
   containing its first cue, bounded, never at/before the previous
   item's end; (b) CUELESS video (promo tails, stingers, music-only
   beds own no cue range) is assigned by VISUAL ROLE, never adjacency
   — story-role shots join the story side, interstitial-role shots the
   interstitial side, and when both sides agree the material leads IN
   to the next item; (c) every story carries THREE timestamps —
   `start_sec` (editorial), `content_start_sec` (first story-role
   shot: "the actual conversation begins"), `speech_start_sec` (first
   word). Interstitials (tease/promo/headlines/handoff) are
   FIRST-CLASS rundown rows, never trimmed silently; a tease shares
   its story's slug (identity binding), but identity never merges
   spans.

8. **Scope, build order, and the acts[] negative result.** PSA's
   products are the CPL-marker catalogue plus the groupings above, all
   assembled CODE-ONLY from Layer-1 evidence (shot_kind gained `slate`
   and `bars` upstream for exactly this rollup). `lib/markers.py` is
   the ONE home of the nondiegetic-block predicate going forward
   (frame-inspection's `lib/spans.py` is demoted to a fallback: PSA
   marker spans arrive there as explicit `spans`), and the gap map +
   orientation machinery was migrated here from the audio-description
   runner — parity with the runner's output is the port's acceptance
   test, so the constants move together or not at all. The measured
   NEGATIVE result: retrospective act RECONSTRUCTION on masters is not
   mechanically recoverable — masters carry no black at broadcast
   break positions (the blacks were inserted downstream), editorial
   surface marks are heterogeneous, and the convergence detectors
   scored far too low against a single-series ground truth to ship;
   tuning further would fit thresholds to one series. PROSPECTIVE
   break-finding ("where COULD you break") is a different, solvable
   problem — any grade-A seam is a valid insertion point. So `acts[]`
   is emitted ONLY from delivery metadata or on as-aired evidence
   (blacks/bumpers present, where the detector works by construction);
   otherwise the product is `break_candidates[]` and NO act claims —
   honest abstention over confident reconstruction.
