# frame-inspection-on-aws

Targeted frame inspection ("the witness"): pooled (asset, category,
span) requests → frame selection from shot-segmentation's OWN archives
(`_frames.tar.gz` @ 4 fps + `embeddings.npz`; embedding-distance change
gate + 6 s heartbeat; NO ffmpeg, NO source access, NO re-extraction) →
ONE pooled bedrock-inference submission → salvage-parsed,
schema-checked per-category evidence products (`credits.json` /
`titles.json` / `verdicts.json`). Evidence only — which cards deserve
voice, or what a verdict implies, is the consumer's judgment.

## Architecture

Gateway dispatcher + callback waker as flat zips; the WORKFLOW Lambda
is a Docker image on ECR (built by this stack's own CodeBuild +
StartBuild custom resource; the image tag IS the version, so an
unbumped redeploy skips the build). The image exists for ONE
dependency: `pyvips[binary]` (bundles libvips) powering
`lib/imaging.py` — caller-requested per-span `crop: "lower_third"` on
title-read spans, measured to cut image payload and input-token cost
substantially on chyron spans. The layout inside the image stays FLAT
(`lib/` + `states/` contents at the task root), byte-compatible with
the family's zip convention.

ONE state machine, the five-state template with the selection step
widened to a Map:

```
Validate and plan      refuse-before-spend: shot-segmentation runs
                       validated, spans resolved (auto-trigger or
                       explicit), categories bound in prompt-registry,
                       model chain IMAGE-capable; emits the bounded
                       selection-unit list (design record §10)
Select frames          Map over units: embedding change gate + heartbeat
                       over the window manifests, JPEGs pulled from the
                       tarballs, per-unit record checkpoints
Stage records          concatenate unit checkpoints in plan order into
                       ONE pooled records.jsonl — streamed via S3
                       multipart, never held whole (images inline)
Submit and wait        ONE held waitForTaskToken submit to
                       bedrock-inference; completion is PUSH (the
                       callback_task_token contract, §11), no polling
Collect and finalize   salvage-parse, schema-check, assemble per-asset
                       per-category products + inspection_manifest.json
Usage tracking         success-path only
```

ASL lives inline in `cfnstack.yaml` with a readability copy in
`definitions.json` — edit BOTH, the build fails on drift. Sibling
gateway-Lambda ARNs (bedrock-inference, prompt-registry) are explicit
stack parameters, never `Fn::ImportValue`; the machine never starts a
sibling's state machine directly (§11).

## Gateway tool contract

Three tools: `inspect_frames` / `get_job_status` / `describe_tool`.
`workflow_id = uuid5(request + tool_version [+ idempotency_token])`;
an idempotent resubmit returns the existing run. Output contract in
`gateway_tool/output_contract.json` (per-category product shapes).

`inspect_frames` accepts either the pooled form —
`assets: [{shot_segmentation_s3uri, shot_metadata_s3uri?, requests:
[{category, spans?}]}]` — or single-asset sugar
(`shot_segmentation_s3uri` + `category` [+ `shot_metadata_s3uri`,
`spans`]). Other parameters: `model_chain`, `use_batch` (absent → TRUE:
this workflow pools requests for batch economics; `false` is the sync
escape hatch for a waiting consumer), `history_scope` (your OWN asset
identity for the inference history — also the rerun mechanism after a
prompt change), `idempotency_token`, and `callback_task_token`
(validated loudly before any submit; requires the deployment's callback
table). `describe_tool` advertises `callback_task_token` and
`span_crop` as features.

The dispatcher is deliberately THIN: every check that costs a network
call lives in the plan state, where a refusal is still before any
inference spend. Even category names are not validated at the gateway —
the category table lives in the workflow's `lib/fi_common.py` (one
source of truth).

## Build & deploy

```
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]
bash ./build-deploy-s3-dist.sh --build-only [--version VERSION]
bash ./create_update_stack.sh ...
```

TWO artifacts: the gateway dispatcher's flat zip, and the workflow
image package (the CodeBuild SOURCE for the container image; the
Dockerfile owns the layout). BUMP `.version` ON EVERY DEPLOY — S3Key
and image tag carry the version; CloudFormation detects key changes,
never content changes. Tests gate packaging, and the build asserts
internal docs are excluded from every artifact.

## Tests

```
python3 -m unittest discover -s test -p 'test_*.py'
```

Hermetic (no AWS); includes copied-module byte-pins against their
master copies in sibling workflows and the ASL no-drift pin between
`cfnstack.yaml` and `definitions.json`. Fixtures are invented.

## Gotchas

- The embedding change gate's threshold is SMALL on purpose: semantic
  (DINO) embeddings see successive text cards as alike, so within-card
  distances sit far below cut distances — a shot-cut-scale threshold
  reads a credit roll as one frame. Measured live before shipping; the
  per-category values are ENV defaults.
- The time heartbeat is load-bearing, not a nicety: a slow scroll
  changes too little frame-to-frame and a change-only gate collapsed a
  real cast roll to nothing. Keep-on-change OR every 6 s.
- Usage lives under `summary.bedrock_usage` in the bedrock-inference
  run summary — not at the top level.
- KNOWN LIMIT (title-read auto-trigger): `unreadable_text_present`
  spans inherit the enrichment pass's recall on that flag — a super
  the sampled frames never SAW is neither read nor flagged, so no
  trigger fires. Explicit caller spans are the override; do not
  compensate with heuristics.
- `lib/prompt_render.py`, `lib/workflow_history.py`,
  `lib/parse_partial_output.py` and `gateway_tool/workflow_identity.py`
  are COPIED modules, byte-pinned to their masters in sibling
  workflows — never edit the copies.
- The `fi-*` prompt-registry categories are INTERNAL and served by the
  registry's admin surface, not its public one.
- The DynamoDB callback-token table holds only rebuildable, ephemeral
  derived state (TTL on every row): default SSE and
  delete-with-the-stack are deliberate.

## Design record

Decisions that shape the workflow, with the why; code and comments cite
these sections by number.

1. **Evidence and judgment stay split.** The witness reads and never
   selects: every card verbatim, every scroll's billing, an explicit
   `[UNREADABLE]` escape instead of a guess — and the CONSUMER decides
   what matters. The triggers that decide WHERE to look are mechanical
   code over shot metadata (`lib/spans.py`) — no model, no
   content-specific rules. Two rejected placements, so this is not
   re-litigated: inside program-structure-analysis (a code-only rollup
   that never touches pixels — giving it frame access would turn
   Layer-2 aggregation into Layer-1 extraction), and widening the
   per-shot enrichment pass (the limiting factor is the ACCESS PATTERN
   — a time range needs dense sampling at readable resolution, where
   enrichment feeds per-shot thumbnails under a fixed image budget).

2. **Placement: a separate "witness" micro-workflow.** The deciding
   question for the whole detection family (credits, titles, ad-break
   opportunity, highlights) is *does it need new pixels?* Ad breaks
   and highlights do not — they are Layer-2 queries over the existing
   shot lattice. Credits/title reading and claim-verification DO, and
   they share one access pattern: (t0, t1, question) → dense frames at
   readable resolution → evidence JSON. Build that once; a category is
   a prompt-registry entry, never a new service.

3. **Access: the proxy archives are sufficient.** Measured with the
   production vision model on real credit frames extracted across a
   resolution ladder: high-contrast credit text is fully readable at
   the 512-wide proxy — the a-priori minimum-cap-height rule was too
   conservative. The credits gap was never resolution; its real causes
   were sampling density (cards change every few seconds), a prompt
   that correctly refuses small text at proxy resolution, and task
   dilution. Consequence: the witness reads shot-segmentation's
   EXISTING archives (`_frames.tar.gz` @ 4 fps, `embeddings.npz`) —
   no source access, no deep-archive restore, nothing re-transcoded.
   Residual risk, stated both directions: static cards are measured;
   fine scrolls with motion blur, tickers, and stylized titles over
   busy picture are untested classes — the contingency is a higher
   ingest-time archive resolution applied selectively (e.g. head/tail
   segments), never globally.

4. **Consumers, by design**: audio-description's tail judgment (the
   card-selection model finally reads real text), the canonical cast
   name map, program-structure-analysis (block structure and billing
   as data), search/compliance (billing extraction independent of AD).

5. **Selection is embeddings + time, no pixel processing.** Keep a
   frame on CHANGE (cosine distance to the last KEPT frame's embedding
   above the category's gate — a static card run collapses to one
   frame) or on TIME (the ≥6 s heartbeat that keeps crawls sampled).
   Pure stdlib on purpose: one `.npy` parse and ~one cosine per
   archived frame; vendoring numpy for that would be the complexity,
   not the saving.

6. **One container, one reason.** Everything is stdlib+boto3 flat zips
   except the workflow image, which exists solely for pyvips cropping
   (§ Architecture). The gateway and waker stay zips.

7. **Call surface per category.** Invariant params plus:

   | category | spans | per-span payload | extra inputs | output |
   |---|---|---|---|---|
   | credits-read | AUTO (nondiegetic shot_kind runs bridged over ≤5 s black) or explicit | — | shot_metadata_s3uri for auto | credits.json (blocks → cards {t, lines}) |
   | title-read | AUTO (unreadable_text_present clusters OUTSIDE blocks) or explicit | optional `crop` | shot_metadata_s3uri for auto | titles.json [{t0, t1, shot_id, lines}] |
   | claim-verify | ALWAYS explicit | `question` per span (rendered as a workflow-provided prompt variable) | — | verdicts.json [{id, verdict ∈ supported\|contradicted\|indeterminate, evidence}] |

   Sampling knobs are category-keyed ENV defaults, never per-request
   (a per-request knob enters the workflow_id preimage); `spans` +
   `question` DO participate in identity (the same question absorbs, a
   new question is a new run); the category decides the output
   contract and `describe_tool` reports per-category shapes.

8. **Request pooling.** Single-episode request counts are tiny, so
   `inspect_frames` accepts a LIST of assets, pooling every rendered
   record across categories AND assets into ONE `records.jsonl` → one
   bedrock-inference job. Records are self-contained, so pooling is
   free; `record_id = {asset_idx}:{category}:{span_idx}` is the demux
   key and outputs land per asset per category, each still matching
   its contract. Identity covers the whole list (a repeated sweep
   absorbs); the natural call unit is "tonight's new arrivals".
   Pooling is what makes batch viable for this workload, not merely
   cheaper.

9. **Outputs are parsed contract products, never pass-through.** Each
   product is assembled in code from salvage-parsed records (strict
   first, counted recovery tiers), schema-checked (an out-of-enum
   verdict is a parse failure), and carries `gaps[]` for spans whose
   read did not survive — honest gap, reasoned, never fabricated;
   partial success IS success, semantic retry = fresh
   idempotency_token. Raw model output is NEVER copied into products:
   `inspection_manifest.json`'s `inference_runs[]` points at the
   bedrock run's own prefix. Four consumers must not each re-implement
   parsing, and parse accounting is a quality signal.

10. **Chunking rule.** EVERY long-running leg is a Map over bounded,
    idempotent, individually checkpointed units. The selection unit is
    bounded by IMAGES per unit (≤18, the record size) — a 120 s credit
    tail is many units, a 3 s super is one. Checkpoints are per-unit
    side objects; a re-entered iteration skips existing ones, never
    re-renders. Unit boundaries double as the parallelism knobs (Map
    concurrency is ENV, never per-request).

11. **Internal workflow; push-based completion.** frame-inspection is
    NOT agent-registered (pooling discipline and cost assume
    orchestrated callers; registration is additive later). The gateway
    Lambda is the SUBMIT BRAIN siblings invoke directly — identity,
    absorption, get_job_status; raw StartExecution would copy that
    logic and could not absorb closed runs. Completion is push: a
    consumer submits with `callback_task_token` and sleeps on ONE
    `waitForTaskToken` state; the callback waker — an EventBridge rule
    on this machine's terminal status-change events — completes every
    token registered for that execution. ALWAYS SendTaskSuccess: a
    FAILED verdict is DATA for the consumer, never a detonation.
    `gateway_tool/callback_registry.py` is the one home of the row
    format — `(execution_arn, sha256(token)[:32])`, a row key, not an
    identity (multiple waiters absorb into one run). Notification is a
    SERVICE concern owned once here; the consumer-side alternative
    multiplies tables/wakers/rules by N workflows. The shape is copied
    from bedrock-inference's callback-token contract so consumers see
    one convention everywhere.
