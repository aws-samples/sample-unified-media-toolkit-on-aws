# shot-metadata-enrichment-on-aws

**Shot-segmentation run in, one validated metadata document per shot +
deterministically assembled scenes out.** An LLM-prompt-driven
micro-workflow on the conversation-analysis template: windows of labeled
frame images render through the prompt registry's
`shot-metadata-enrichment` prompt (seven tokens, including a
reference-face gallery) and submit as **one** bedrock-inference request —
**batch by default**, `anthropic.claude-sonnet-4-6` by default. The
per-shot documents feed scene assembly, program-structure analysis,
hybrid search, artwork selection and compliance review.

This README is the single developer document for the workflow. The
numbered **[Design record](#design-record)** at the end preserves the
section numbers that code comments cite (`README §3`, `§10`, …).

## Products

Everything lands under
`micro-workflow/shot-metadata-enrichment/{workflow_id}/`:

| label | basename | what |
|---|---|---|
| `shots` | shot_metadata.jsonl | THE product: one validated doc per shot; join to shots.json by shot_id (times never pass through a model) |
| `scenes` | scenes.json | flags → scenes; attributes AGGREGATED IN CODE (majority/union — never read off one shot); pairing_repairs[] + repair rate (an interleaved-genre detector) |
| `window-manifest` | window_manifest.json | window plan + POINTERS to the bedrock run (never copies) |
| `usage` | usage_tracking.json | aggregated cost/tokens; `cost_is_complete` discipline |

`_plan/`, `_records/` and `records.jsonl` are internal
([gateway_tool/output_contract.json](gateway_tool/output_contract.json)
is the field-level pin).

### The `depicted` field

The catalog includes `depicted`: policy-relevant OBSERVABLES under a
CLOSED vocabulary — never verdicts, no category words, no severity;
compliance consumers classify under their own rulebook (evidence is
ours, policy is theirs). The vocabulary authority is the BOUND PROMPT
ITSELF: `output_parse.depicted_vocabulary` parses the backticked act
tokens out of whatever version ran, and the prune conforms output to
THAT list — so a custom `prompt_version` brings its own vocabulary, a
section with no tokens is an OPEN vocabulary (shape-checked, never
act-pruned), and code carries no list to drift (conformance must not
cost prompt customization; a live trial caught the model inventing
acts, which is why the prune exists at all). A test pins the seed
prompt's section to the 21 validated acts; the catalog was widened
after an audit against professional describer records found observables
the old field set could not carry. Capture is NEVER OPTIONAL for this
set and plain-named. What frames don't show it cannot record:
dark/brief events are a frame-inspection escalation's job, not this
prompt's.

SEMANTICS: `[]` is an ATTESTATION ("none visible") and only a prompt
that ASKED may make it — collect_finalize injects `[]` for model
omissions only when the bound prompt text carries the section; a doc
from a prompt version with no section has NO `depicted` key at all,
which means "not assessed", never "none".

## Architecture

TWO Lambdas (gateway = flat zip; workflow = container image, for
pyvips), ONE state machine, NO tables, NO polling anywhere. The
conversation-analysis template with the render step widened to a Map:

```
Validate and plan     everything that can refuse refuses BEFORE any spend:
                      shots.json fetch+validate, optional face/transcript
                      joins, get_prompt (binds `default`, checks the
                      7-token values catalog), IMAGE-capable model-chain
                      validation, the §3 image-bounded window plan + Map
                      batches, history scope → _plan/plan.json
→ Render windows      MAP over window batches (Map + small thread pool —
  (Map, concurrency 4) OOM bounded per iteration). Each iteration:
                      RENDER_THREADS tarball fetches, frame picks by tier,
                      per-window face gallery (pyvips), transcript slice,
                      copied prompt_render → _records/window_NNN.json
                      CHECKPOINT. LambdaResourceTimeoutException → Retry
                      RESUMES (existing side objects are skipped, never
                      re-rendered)
→ Stage records       concatenate checkpoints in window order into
                      records.jsonl — STREAMED via S3 multipart (never
                      held whole); refuses on a missing checkpoint or an
                      image over-budget
→ Submit and wait     ONE waitForTaskToken state, NO Catch: inline invoke
                      of bedrock-inference's gateway (records_s3uri +
                      callback_task_token + idempotency_token = THIS run's
                      uuid), history pointer written from the TRUE submit
                      response, redrive IS the reconcile
→ Collect + finalize  parse per-window JSONL (strict first; a failed line
                      climbs output_parse's COUNTED salvage ladder —
                      escape pass, then gated prefix salvage, gate =
                      shot_id alone; unrecovered → honest gap),
                      coverage per shot_id, the §4 pairing repair
                      (`false` wins BOTH directions, repairs counted),
                      scene assembly from repaired flags → the 3 products
→ Usage tracking      aggregates the bedrock run's summaries; success-path
                      only, never raises
```

The workflow Lambda is operation-routed (top-level
[lambda_function.py](lambda_function.py) →
`states/<name>/lambda_function.py`, the family tree shape); `lib/` is
the shared package. The ASL lives inline in
[cfnstack.yaml](cfnstack.yaml) with a readability copy in
[definitions.json](definitions.json) — the build fails on drift.

## The decisions that shaped it (all measured — see the Design record)

- **The window is bounded by IMAGES, not shots (§3)**: 16-shot target
  inside a 100-image budget (the registry's Claude/GPT `max_images`
  pin), counting context-shot frames AND the gallery's 1 block. Shot
  count floats DOWNWARD only. Frame tier: 1 under 4 s, 2 under 12 s, 3
  at ≥ 12 s, capped by the archive's own `selected_frame_count`.
- **Frames pass through as ORIGINAL bytes** — no tiling (measured: a
  full-resolution tile breaks whole-frame judgments like
  `title_card`/`black`), no re-encode, one labeled image block per
  frame; the text label is the join key.
- **The face gallery is the ONE tiled image (§5)**: 192×256 3:4 cells
  from face-recognition's `rep_face_key` crops (NEVER `thumbnail_key` —
  that is a debug montage), 3px white dividers, cap 5 by within-window
  appearances, always exactly 1 image block. Names alone bind by LIST
  ORDER (measured wrong at coin-flip rate in two-person shots); the
  gallery is what binds. A facemetadata predating `rep_face_key` is
  refused by name.
- **Pairing repair: `false` wins, BOTH directions (§4)** — a false-list
  reason forces the flag false; a true-list reason on a false flag
  keeps it false (the reason can be a true observation that is not a
  boundary). Reasons stay VERBATIM; repairs are COUNTED (≈0% on linear
  content, a few percent on interleaved content — the rate is itself a
  genre signal).
- **Defaults land CONCRETE in plan.json**: `model_chain` →
  `anthropic.claude-sonnet-4-6` (the model the prompt was validated on;
  a versioned constant, deliberately not an env var), `use_batch` →
  **TRUE** (image-heavy volume, nothing interactive; sync is the
  explicit opt-out). Chain models must be IMAGE-capable — a text-only
  fallback would reject every record.
- **Partial success IS success**: unanswered shot_ids land in
  `shots_gap` honestly; the semantic retry is the caller's fresh
  `idempotency_token`.

## Key conventions

- **record_id**: `window-{index:03d}` — the join from response.jsonl to
  the plan's windows.
- **Checkpoint/re-enter**: the Map iterations watch
  `get_remaining_time_in_millis()` (60 s buffer) and raise
  `LambdaResourceTimeoutException`; the ASL retries (10×) and re-entry
  skips completed `_records/window_NNN.json` side objects.
- **Config knobs** (`WINDOW_SHOT_TARGET` 16, `WINDOW_IMAGE_BUDGET` 100,
  `WINDOWS_PER_BATCH` 10, `RENDER_THREADS` 3, `BURST_REPRESENTATIVES`
  3) are ENV, never per-request — a per-request knob would enter the
  workflow_id preimage. The template's env block pins the same values;
  edit BOTH or the deployment diverges from the tree.
- **pyvips, not PIL** (face-recognition's imaging stack) — in a
  CONTAINER IMAGE (CodeBuild+ECR, face-recognition's
  StartBuildDelayResponse pattern). Zip-vendoring the wheel was tried
  first and REJECTED live: the pyvips/pyvips-binary version pairing is
  load-bearing (2.1 + 8.18 imports and then dies at
  `ffi.callback('VipsCallbackFn')` on first use).
  `pyvips[binary]==3.0.0`, the same pin as face-recognition. The
  gateway stays a flat zip.
- **Copied modules** (byte-identical to their masters, pinned by
  tests — copy, never edit): `lib/prompt_render.py` +
  `lib/workflow_history.py` from prompt-registry-on-aws,
  `gateway_tool/workflow_identity.py` from
  conversation-analysis-on-aws, `lib/parse_partial_output.py` from
  bedrock-inference-on-aws (the salvage ladder; stdlib-only —
  json-repair was measured returning silently trimmed/fabricated values
  and rejected).
- **Slim shot_list (§5)**: id, REQUIRED/CONTEXT, duration_sec,
  frame_count — NO timecodes (the prompt forbids emitting times and the
  join is shot_id).

## Gateway tool

[gateway_tool/lambda_function.py](gateway_tool/lambda_function.py) —
flat zip, no Docker. Tools: `enrich_shot_metadata`, `get_job_status`,
`describe_tool`. Standard micro-workflow contract: `workflow_id =
uuid5(canonical request + TOOL_VERSION [+ idempotency_token])`; a
duplicate submit is ABSORBED (handle + the byte-pinned message, never
results); results come from `get_job_status` alone. Parameters:
`shot_segmentation_s3uri` (required; run prefix or shots.json),
`face_recognition_s3uri`, `transcript_s3uri`, `program_names`,
`overall_content_type`, `model_chain`, `use_batch`, `history_scope`,
`idempotency_token`.

## Build & deploy

```bash
bash ./build-deploy-s3-dist.sh --bucket MY_BUCKET [--version VERSION] [--single-region]
bash ./create_update_stack.sh --stack-name my-shot-metadata-enrichment \
    --resource-prefix myprefix --proxy-bucket my-proxy-bucket \
    --bedrock-inference-lambda-arn arn:... --prompt-registry-lambda-arn arn:...
```

A CONSUMER stack: the two sibling gateway-Lambda ARNs are required
parameters, so bedrock-inference and prompt-registry deploy first (and
the prompt-registry deploy must have seeded the
`shot-metadata-enrichment` prompt). BUMP `.version` ON EVERY DEPLOY
(S3Key carries it; CloudFormation detects key changes, not content
changes). Known-benign lint: W1034 on the `%%TOKEN%%` mappings
pre-substitution.

## Tests

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

No AWS needed. The windowing tests run against a real committed
97-shot shots.json fixture (the asset every prompt validation ran on);
the state tests run the whole pipeline against dict-backed fakes
(plan → render → stage → submit → collect) including the
checkpoint/resume, multipart-abort, gallery, pairing-repair and
honest-gap paths. Tests gate packaging.

## Gotchas

- **An unbumped redeploy leaves old code running with no error** —
  S3Key versioning, same as every sibling.
- **The seeder can rebuild the `default` prompt under this workflow**
  (PromptSeedMode: overwrite-builtins): prompt bytes are NOT in the
  workflow_id hash, so re-enriching after a prompt rebuild needs a
  fresh `idempotency_token`. Provenance records `source_md5` either
  way.
- **bedrock-inference enforces the registry's `max_images` per record**
  (100 on the Claude/GPT entries) before a token is spent — the §3
  planner exists so this guard never fires; Stage records re-checks the
  rendered bytes as defence in depth.
- **shots.json's `selected_frame_count` bounds the plan** — the planner
  never promises a frame the tarball does not hold, so the renderer's
  actual picks always equal the plan (the budget check is exact, not
  approximate).
- **A transcript's ASSET IDENTITY is the caller's responsibility** —
  nothing in the chain can catch a plausible-but-wrong
  `transcript_s3uri` join.

---

## Design record

The settled design, with the measurements that settled it. Section
numbers are stable — code comments cite them as `README §N`.

### §1 Shot-level, not scene-level (the founding decision)

Answered SHOT-LEVEL, for four reasons, none primarily cost:

1. **Shots are facts; scenes are opinions.** A cut is deterministic; a
   scene is a construct. Scene-first enrichment welds every piece of
   metadata to boundaries that cannot be revised without re-paying the
   whole inference bill. Shot-level metadata survives regrouping for
   free.
2. **The visual-evidence problem is only solvable at shot level.** A
   shot is one continuous take: 1–3 frames genuinely represent it and
   every claim is pointable at pixels. A scene is heterogeneous and
   needs grid images — exactly the "grids downscale small marks into
   illegibility" trap.
3. **It is CHEAPER.** Image tokens track total frames, not units;
   shot-level needs fewer, more legible frames than scene grids.
4. **The downstream consumers vote for shots.** Program-structure,
   ad-break, opening-title, next-episode and recap detection all key on
   TRANSITIONS, which live at cut granularity — and LLM-driven scene
   grouping replacing statistical clustering REQUIRES shot-level
   semantic metadata to exist first.

Scene-level description is not lost: it becomes a cheap text-only
rollup (§4).

### §2 The field set — every field named for its consumer

| Field | Shape | Consumer |
|---|---|---|
| `description` | one sentence, ≤25 words | scene rollup; hybrid-search FTS |
| `starts_new_scene` + `boundary_reason` | bool + one closed word | scene assembly (§4) |
| `shot_kind` | closed: normal, title_card, credit_roll, graphics_overlay, logo_bumper, black, slate, bars | opening-title, end-credit, program-structure |
| `setting`, `time_of_day`, `lighting` | closed vocab | scene grouping (setting continuity), search facets |
| `camera_shot_type` | closed vocab | ad-break (establishing shots), highlight |
| `texts_on_screen` | verbatim POST-PRODUCTION text only, when legible | credits/titles/next-episode; entity linking |
| `unreadable_text_present` | bool | the escalation trigger (§8) |
| `people_count` | closed bucket: 0, 1, 2, 3, several, crowd | grouping, search (model counting is approximate; buckets absorb it) |
| `people` | names from the face-recognition join ONLY (§5) | entity linking, search |
| `named_entities` | evidence-typed, from overlays/legible text only | entity linking, search |
| `objects` | searchable-specific terms | hybrid search |
| `brands` | carrier/evidence/confidence schema | monetization; the §9 comparison |
| `depicted` | closed-vocabulary observables (see above) | compliance review |
| `sentiment` | single word | ad-break, highlight |

Deliberately NOT produced by the model: timestamps (authoritative from
shots.json — the model references `shot_id` only, so hallucinated
timestamps are structurally impossible); people names from pixels (§5,
§7); scene ids (§4 — scenes are assembled downstream of the flags).

### §3 Windowing and record shape

One record = a window of consecutive shots, each shot 1–3 labeled
frames (the tier above). NOT one shot per record (the model needs the
neighborhood to judge relatedness) and NOT the whole program
(multi-image accuracy degrades measurably as image count grows).

- **THE WINDOW IS BOUNDED BY IMAGES, NOT BY SHOTS.** Pack consecutive
  shots into a record until either the shot target or the image budget
  is reached, whichever binds first — the budget counts the context
  shot's frames AND the face gallery's 1 block, so the shot count
  floats DOWNWARD when the 3-frame tier stacks up, never above the
  target. bedrock-inference hard-refuses any record over the primary
  model's registry `max_images` before a token is spent; the planner
  exists so that guard never fires. (A shots-only rule was a latent
  defect: at 1–3 frames per shot, long-take content overflows an image
  cap on real windows.)
- **The target is 16 shots inside a 100-image budget** (the registry's
  Claude/GPT `max_images` pin). Measured: coverage and the structural
  fields (`starts_new_scene`, `shot_kind`, `setting`, `people_count`)
  hold at 16 shots; the enumerated fields (`objects`, description
  length) show mild positional tail decay — accepted for the ~11%
  input amortization. The knee is ACCURACY, not platform limits:
  models accept far more images than the budget; positional decay in
  enumerative fields is what binds.
- **Context shots at the seams**: each window carries the previous
  window's last shot as a context-only entry — frames included,
  explicitly marked "context: do not emit an entry" — so every
  `starts_new_scene` decision, including the window's first real shot,
  has the preceding shot's actual visuals. Coverage validation expects
  exactly the non-context shot_ids back.
- **Labeling discipline** (both vendors' official guidance): every
  frame is preceded by a text label carrying the REAL join key —
  "Image 7 = shot_id 214 (2 of 2):" — images before instruction text.
  Bare ordering mis-indexes.
- **No grids, ever** (see §11 for the tiling probes). Single labeled
  frames at the proxy resolution (512×288 for 16:9), passed through as
  ORIGINAL bytes — no re-encode, no resize.
- **Output = one self-contained compact JSON object per shot keyed by
  shot_id** (JSONL per window), never positional arrays —
  count-check, then the retry unit is missing ids only. A window is
  one line of records.jsonl; the whole asset submits as ONE
  bedrock-inference request.

### §4 Scene assembly: local flags, deterministic grouping

The per-shot pass does NOT do global grouping (a window cannot see
scene structure beyond its edges; coupling grouping into the catalog
pass couples two failure modes). It emits a LOCAL judgment —
`starts_new_scene` against the previous shots' frames + the transcript
slice — and scene assembly is trivial deterministic code over the
flags. Statistical/agglomerative shots-to-scenes clustering is retired.

- **Linear content** (scripted drama, film) — flags alone are
  sufficient (measured: zero false seams across window boundaries).
- **Interleaved content** (news, magazine, sports, documentary) —
  flags are necessary but NOT sufficient. A story package deliberately
  alternates interview A-roll with B-roll, so a pairwise judgment sees
  a genuine place change at almost every cut while the true unit is the
  whole interleaving. This is a GLOBAL property of the sequence; no
  prompt iteration fixes a local judgment (two attempts, §11). A
  dedicated text-only grouping prompt over the complete shot-doc
  sequence is the designed second pass; the flags are its input —
  `intercut_return`, `graphics_boundary`, `new_conversation` and the
  per-shot settings are exactly the interleaving evidence it needs.

**The collect state MUST validate the flag/reason pairing and repair
it — and the repair rule is `false` WINS, not "the reason wins".**
`boundary_reason` draws from disjoint `true`/`false` lists and the
model crosses them at a low rate on interleaved content (zero on
linear). Prompting against it did not help: the crossing is not
disobedience but the model reporting a real local discontinuity while
naming the global structure correctly — and the direction is NOT
consistent (both `true`+false-reason and `false`+true-reason occur, the
latter because reasons like `after_black`/`new_conversation` describe a
TRANSITION, not a scene relationship). In every observed case `false`
was correct, which also matches the designed bias: a merge is
repairable downstream, a spurious split has already destroyed the
grouping. On a mismatch take `false`, keep the model's reason verbatim,
and COUNT the repairs — the rate is a usable interleaved-genre
detector.

Scene-level attributes are AGGREGATED IN CODE (majority /
first-non-indeterminate), never read off a member shot — per-window
independence causes attribute drift inside a scene (§11).

### §5 Inputs

| Input | Source | Required? |
|---|---|---|
| `shots.json` | shot-segmentation run (shot_id, exact timings, entropy, black/sync-pop, selected_frames, frames_key) | yes |
| per-shot frames | `shot_frames/shot_NNNN_frames.tar.gz` — distance-selected JPEGs at proxy res, already persisted | yes |
| transcript | speech-diarization `final_transcript.json`, sliced per window by timestamp | optional (big description-quality gain, cheap tokens) |
| face names | face-recognition run output, joined per shot by timestamp, injected as TEXT | optional |
| face gallery | ONE grid image per window from `rep_face_key` crops | optional, travels with face names |

- **The face-name rule (hard):** when a name is provided for a face
  tracked in the shot, the description MUST use it — generic labels
  generate no downstream value. Names come ONLY from the join: VLMs
  refuse to identify faces by policy, and a model pressed for names
  invents them from context priors. **A name list is not a join** —
  handed a bare cast list, the model distributes names by plausibility
  (measured, with invented names). Absent the join, the workflow
  degrades to unnamed people.
- **The gallery is what BINDS names to bodies** (§11 probes): names
  alone assign by LIST ORDER in multi-person shots — a coin flip
  wearing a confident face — and an anti-guess instruction does not
  survive contact with a plausible assignment. One labeled reference
  crop per name fixed the binding completely and order-stably, with no
  policy refusal (matching a provided labeled crop is
  re-identification, not identification).
- **Gallery format (settled by probes):** ONE grid image, not N crop
  blocks (as good or better at every cell size, and it makes the image
  budget deterministic: always exactly 1 block). Cells 192×256 (3:4) —
  96×128-scale sources measurably fail binding. Cell source:
  face-recognition's `rep_face_key` — the person's representative face
  cropped once per asset at native resolution, NEVER the stored
  `thumbnail_key` (a debug montage) and never per-window crops from
  512×288 shot frames (a face in a medium shot at proxy res is far
  below the failing tier). Cap 5 faces per window by within-window
  appearance count (tie-break `total_screen_time_sec`); overflow names
  still ship as text.
- **Slim `shot_list`**: id, role, duration_sec, frame_count — no
  timecodes (unusable by construction: the prompt forbids emitting
  times and the join is shot_id; dropping them was the single biggest
  free input win). Duration stays: a 40-second take and a 12-frame
  flash are different shots.

conversation-analysis output is deliberately NOT an input — the
dependency graph stays flat.

### §6 Models and cost posture

Default chain `["anthropic.claude-sonnet-4-6"]` — the model the prompt
was validated on, a versioned constant in `lib/sme_common.py`, never an
env var (a per-stack env could drift with nothing in the tree to show
it). `use_batch` defaults TRUE: an asset is ~100 windows of image-heavy
records, nothing downstream is interactive, and the batch discount is
the normal case (measured live at ~half the sync price).

The measured cost structure, which drives several design choices:

- **The system prompt is the largest input term** — flat per window,
  roughly as large as all the frames together, and batch cannot cache
  it. That is why window size amortization matters and why prompt-prose
  compression was probed (and rejected, §11).
- **Output tokens dominate at this image size** — the terse
  closed-vocabulary schema is a cost control, not just a hallucination
  control. Switching output to compact one-line-per-shot JSONL cut
  output ~34% with structurally identical results; every field added to
  §2 is paid once per shot.
- **Key order is recovery order.** The model follows the example
  line's key order essentially always, and the salvage ladder's prefix
  rung keeps everything BEFORE a damage point — so the quote-bearing
  transcription fields (`texts_on_screen`, `supers`), where all
  measured damage lives, sit at the very END of the document.
  `description` deliberately STAYS at position 1: it is generated
  immediately before `starts_new_scene`/`boundary_reason` and is the
  implicit reasoning step for the one judgment scene assembly hangs on.
- **512×288 is the pass-1 resolution** — the frames already exist at
  exactly that size, and scene gist, setting, people-count and boundary
  judgments survive it comfortably. Resolution ladders per model buy
  nothing (flat-rate image pricing exists in the chain's secondary
  candidates).

### §7 Hallucination controls (research-ranked, all adopted)

1. **Closed vocabularies everywhere a vocabulary can close** (§2),
   shared verbatim with brand-logo-detection so search joins across
   outputs.
2. **"Empty is correct"** stated explicitly per list field — VLMs are
   measurably biased toward affirming presence; most drama shots
   contain zero brands and zero readable text, and the prompt says so.
3. **Evidence-typed claims** for brands and entities (text / logotype /
   product_design / trade_dress), confidence as three words, never
   numbers.
4. **No face naming by the VLM** (§5) and no timestamp emission (§2).
5. **Counting in buckets** — models count small objects approximately;
   buckets absorb the error instead of recording it.
6. **temperature 0** (the workflow default).
7. **Draft-then-verify is NOT in pass 1** — it doubles the bill for
   fields that are mostly empty. Reserved for the escalation tier
   (§8), where the flagged subset is small and the fields (verbatim
   text, brands) are exactly the invention-prone ones.

### §8 High-res escalation (designed in, deferred)

Proxy-resolution frames read scene gist but not dense small text. What
survives is DETECTION; what doesn't is READING. So:

- **Pass 1 flags, never guesses**: `shot_kind`
  (title_card/credit_roll/…) and `unreadable_text_present: true` ARE
  the output when text is present but illegible.
- **The escalation is a SEPARATE registry prompt** (deferred), run only
  on flagged shots — typically a small fraction of a program: 1–3
  frames per shot at ~1280–1568px, single shot per record, verbatim
  transcription + brand verification, draft-then-verify ON, merged back
  into the shot docs. Escalation beats running everything high-res by a
  wide margin, because dense text mostly isn't there.
- **Measured caveat on the premise**: the "below the reading threshold"
  arithmetic is TYPOGRAPHY-DEPENDENT, not a law — a short centred cast
  list in a large face survives the proxy downscale (read verbatim,
  live) while the dense crew crawl below it honestly flagged
  `unreadable_text_present`. Real text plus an honest flag on the SAME
  shot is the behaviour the escalation contract must assume, so
  escalation MERGES into `texts_on_screen`, never replaces it.
- Frame source, settled as design: shot-segmentation grows a higher-res
  selected-frames archive (it holds the source during transcode and
  owns the exact PTS math). Unscheduled; v1 ships pass 1 only and
  measures how far proxy resolution actually gets first.

### §9 The brand/logo comparison (kept separate for now)

brand-logo-detection stays a separate prompt. Pass 1 attempts the same
three outputs (brands/objects/classification) with the same schema and
vocabularies verbatim at proxy resolution, and a same-assets comparison
through both decides whether a dedicated brand prompt survives, becomes
the escalation tier's brand half, or retires. Sharing the schema
verbatim is what makes the comparison apples-to-apples.

### §10 Architecture: the conversation-analysis template, widened

Everything the conversation-analysis template settled is inherited, not
re-derived — TWO Lambdas, linear states, ONE `waitForTaskToken` wait,
NO polling, NO Choice states:

- **Validate-and-plan refuses before any spend** and binds the
  registry `default` prompt; states pass URIs, never bodies (256 KB
  state I/O).
- **The render step is widened to a Map** over window batches with a
  small thread pool per iteration — the OOM-bounded unit, with
  per-window side-object checkpoints so a Lambda timeout retry RESUMES
  rather than restarts (face-recognition's pattern).
- **Siblings are invoked directly at the LAMBDA layer** — the state
  machine never starts a sibling's state machine: the sibling's gateway
  Lambda is its submit brain (identity, staging, absorption of closed
  runs), and bedrock-inference's `run_inference` is invoked inline with
  `callback_task_token`, whose waker completes the token with the
  terminal status as data. NO Catch on the wait state: a redrive
  resubmits, is absorbed, and completes synchronously — redrive IS the
  reconcile.
- **History pointers are written at submit time from the TRUE submit
  response** (under `micro-workflow/histories/`), so inference history
  survives even a later collect failure.
- **Output contract: parse and merge, never pass-through** — the
  collect state owns coverage per shot_id (the retryable unit is the
  shot_id; uncovered ids are reported honestly as partials) and the
  usage state aggregates the inference run's own usage/summary
  documents into the `usage` label (success-path only, never raises —
  a terminal-state exception would fail an execution whose products
  are already written).
- Shared plumbing lives in `lib/sme_common.py` (adapted from
  conversation-analysis's `ca_common`); the gateway dispatcher is
  adapted from conversation-analysis's, THIN by design — every check
  that costs a network call lives in Validate-and-plan, where a refusal
  is still before any inference spend.

Registration: a nested stack in the micro-workflow-bundle; a gateway
target with three tools (submit / get_job_status / describe_tool) + an
agent prompt.

### §11 Measured probes and later evolutions

The probes that settled (or re-settled) the numbers above. Full
provenance lives outside the public tree.

- **Boundary-flag validation** (linear content, full-asset): one prompt
  fix was needed, and the defect is the lesson — **a permissive
  boundary REASON is a boundary cause.** A `new_subject` reason
  LICENSED calling every cutaway a new scene (the subject changes on
  almost every cut), compounding with near-darkness where the model has
  no location evidence to argue itself out. The fix: three ordered
  rules (a cut is not a boundary; a change of subject is not a
  boundary; when you cannot tell, answer `false`), `new_subject`
  retired for `new_line_of_action`, and `no_location_evidence` added to
  the `false` list so "I can't see where this is" has somewhere honest
  to go. Residual, stated honestly: the fix biases toward `false`, so
  an evidence-poor montage under-segments rather than over-segments —
  the right direction (a merge is repairable, a split is not), but a
  real trade.
- **The interleaved direction**: on magazine/news content the flag
  fires on well over half of all cuts. Fix 1 (kept): *a RETURN is not a
  scene boundary*, judged against every earlier shot in the window, +
  `intercut_return` on the `false` list — content-type-agnostic on
  purpose (interview↔B-roll, crosscutting, phone-call ends,
  sports↔commentary are the same pattern), and linear content did not
  regress. Fix 2 (attempted, FAILED, do not retry in the prompt): an
  explicit pairing instruction made flag/reason crossings WORSE — it is
  repaired in code (§4). Conclusion: flags under-deliver on interleaved
  genres structurally, hence §4's grouping prompt.
- **JSONL output** (switched): truncated JSONL loses the last line and
  the coverage check names the missing ids; a truncated single JSON
  object loses the whole window. Compact one-line docs also cut output
  ~34%, and the model stopped fencing its output entirely (the parser
  stays fence-tolerant anyway).
- **Terse prompt rewrite: A/B'd and it LOST — do not ship it.** The
  saving is capped (~9% of the prompt; most of the prompt is field
  definitions and irreducible closed vocabulary) and on interleaved
  content it regressed FIELD ROUTING: filmed-environment signage leaked
  into `texts_on_screen` (defined as post-production text only) instead
  of `brands`. Misrouting, not hallucination — the subtler failure,
  because nothing looks invented. On linear content it was
  indistinguishable, which is also worth knowing.
- **Input-token levers, swept**: the prompt is only ~42% of the input;
  tables are a de-duplication device, not a compression device (they
  win where prose repeats a label per row and LOSE on a flat enum
  list); the controlled vocabulary is cheap and the wrong target
  (pruning terms deletes the coverage they exist for); the one free win
  was in the DATA (the slim shot_list, §5).
- **1 frame per shot: probed and REJECTED.** The boundary flag does not
  care, but everything that depends on motion or duration degrades —
  `camera_shot_type` changed on a quarter of shots (a single mid-shot
  frame cannot see a take that pushes in), and a one-frame sample of a
  dark moment flips `shot_kind` to `black`. One frame samples an
  instant; two frames sample a TAKE. Keep the adaptive tier.
- **Larger windows**: structural fields hold to 16 shots (coverage
  complete, boundary map identical); the enumerated fields show
  positional tail decay (head-heavy `objects`, thinning descriptions).
  The 16-shot target adopts that trade for the input amortization; the
  image budget still binds first on long-take content. A later brief
  drop to 12 (dodging a model family's truncating default output cap)
  was reverted same-day — the root fix was bedrock-inference emitting
  `maxTokens` on the affected registry entries, not smaller windows
  that tax every model more input.
- **Tiling shot frames: token-neutral, measured, rejected on quality.**
  A tile collapses N blocks into 1 but saves ~0% tokens, and
  whole-frame judgments (`title_card`, `insert_graphics`, `black`,
  `camera_shot_type`) break — "the WHOLE FRAME is X" stops being true
  the moment the frame is a pane, and a prompt variant that taught the
  tiles fixed the boundary fields and did NOT fix `camera_shot_type`.
  Perceptual artefact, not a labelling misunderstanding. The model
  never once mentioned the tiling; it understood the tile perfectly and
  answered a different question about it.
- **Face binding probes** (§5's evidence): window-level
  re-identification from frames alone is reliable without crops;
  identification is refused by policy; and the dangerous middle case —
  a bare cast list — gets distributed by plausibility (measured with
  invented names, every assignment made, flagged merely
  `confidence: low`). Names-only binding in two-person shots followed
  LIST ORDER both ways. One labeled reference crop per name bound
  correctly in both orderings; a grid gallery passed every run at both
  cell sizes while separate crops failed at thumbnail scale — pinning
  cell resolution, not shape, as the failure axis.
- **Run-to-run instability is itself a genre signal**: at temperature
  0, linear content's boundary map is reproducible across re-runs;
  interleaved content moves between runs because it sits ON the
  decision boundary. Do not chase those diffs as prompt bugs, and do
  not benchmark prompt changes on interleaved content alone.
- **Per-window independence causes attribute drift inside a scene**
  (same room, different windows, different `time_of_day`) — hence §4's
  aggregate-in-code rule.
- **The secondary-model degradation shape is known**: a weaker model
  preserves STRUCTURE (ids, flags, shot_kind) and loses DESCRIPTION
  richness and the enumerated fields — the right way round for a
  fallback, and worth stating to consumers. A smaller window fixes a
  weak model's image-ATTRIBUTION failures (the multi-image knee is
  LOWER for weaker models) but not its capability gaps — and a chain
  cannot have per-model window sizes anyway: a record's bytes go to
  every chain member unchanged.
- **Burst-run folding**: shot-segmentation publishes `burst_runs` —
  contiguous intervals of ≥6 consecutive sub-second shots (jingles,
  montages, promo stings). Every member keeps its own REQUIRED entry
  and its own output document, but only `BURST_REPRESENTATIVES` members
  per run (first/middle/last) carry frame images — the rest are
  frameless and the prompt tells the model to judge them from the
  run's representatives (the members are visually near-interchangeable;
  each frame is the dominant token cost). Set `BURST_REPRESENTATIVES=0`
  to disable. The `shot_list` also carries structural evidence
  annotations the model cannot see from one window: burst-run
  membership and programme-level recurrence.
- **Upstream dependency, pinned**: face-recognition emits
  `people/person_NNNN_face.jpg` (3:4 native-res representative-face
  crop) and `rep_face_key` in facemetadata.json — the gallery consumes
  the provided crop and reconstructs nothing; a facemetadata predating
  `rep_face_key` is refused by name. shot-segmentation needed no
  changes: the tarball frame archives cover the frame tiers, and the
  loose per-shot keyframe hybrid-search wants is this workflow's job
  if ever needed (it untars anyway).
