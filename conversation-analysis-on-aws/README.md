# conversation-analysis-on-aws

**Speech-diarization transcript in, merged coverage-validated topic
segments out.** The first LLM-prompt-driven micro-workflow in the family —
built on the prompt-registry + bedrock-inference pair. It chunks long
programs on natural boundaries, renders the registry's
`conversation-analysis` prompt per chunk, submits ONE bedrock-inference
request with push-based completion, validates coverage, and merges
deterministically — uncovered ranges are reported honestly, never
silently retried.

Each merged segment carries topic, reasoning, named entities, key events,
sentiment/tone/conversation-type classification, and `discourse_facts`:
policy-relevant SPOKEN observables — threats, self-harm intent, despair,
hate/slurs, sexual/substance references, crime instruction — each a
`{kind, cue_id, quote}` with the quote copied VERBATIM from the cue, so
any consumer can mechanically re-verify it against the transcript.
Evidence, never a verdict; plain profanity is deliberately NOT a fact
(word-lists are the consumer's mechanical job on the verbatim
transcript). No code carries the vocabulary — the parser's documented
pass-through IS the mechanism, so a custom `prompt_version` freely
declares its own kinds. `[]` means "attested: none"; an ABSENT key means
the bound prompt never asked or the model omitted it — treat absent as
NOT ASSESSED.

This document is the single developer doc for the folder. The numbered
**Design record** at the end preserves the section numbering that code
comments cite (`README.md §5` etc.).

## Shape (the template the other LLM-prompt workflows copy)

This folder is the TEMPLATE: the states are shaped so the
workflow-SPECIFIC part is exactly one stage ("prepare values" — here, a
cheap transcript-to-WebVTT render; in a vision workflow, a heavy frame
stage), and everything around it (plan, render via the copied module,
stage records, submit-and-wait, collect, validate, merge, usage) is the
reusable skeleton. Each workflow still owns its own machine; what is
shared is the ASL shape and the copied modules.

TWO Lambdas, TWO zips (flat gateway zip + tree workflow zip), ONE state
machine, NO tables, NO containers, NO polling anywhere. FIVE LINEAR
STATES, deliberately (the first draft carried a deadman Catch + reconcile
and an in-machine coverage-retry loop — both cut in review):

```
Validate and plan     everything that can refuse refuses BEFORE any spend:
                      transcript normalize (§7a), get_prompt binds `default`
                      (§1), model chain vs bedrock's registry (§7c), §5
                      chunk plan, §7b history scope → _plan/plan.json
→ Render and stage    per chunk: WebVTT envelope (§2) + copied prompt_render
                      → records.jsonl staged at OUR prefix (§8)
→ Submit and wait     ONE waitForTaskToken state (§8), NO Catch. The handler
                      invokes bedrock's gateway Lambda inline (records_s3uri
                      + callback_task_token + idempotency_token = THIS run's
                      uuid) and writes the §7 history pointer from the TRUE
                      submit response; the wake is bedrock's waker, carrying
                      the terminal status AS DATA. A 26 h timeout FAILS the
                      execution — and a REDRIVE of this state resubmits,
                      absorbs into the finished run, and is woken
                      SYNCHRONOUSLY by the gateway's absorbed-closed path:
                      redrive IS the reconcile.
→ Collect + finalize  one state, nothing branches between the halves: parse
                      JSONL (§4), validate coverage (§3), then pure-code
                      merge (§5-§6) → analysis + chunk-manifest (§7e).
                      Uncovered ranges land in segments_gap HONESTLY and the
                      run SUCCEEDS; the semantic retry is the CALLER's fresh
                      idempotency_token (bedrock owns transport retries —
                      throttles, fallbacks, batch repair — and no sibling
                      can validate OUR coverage semantics).
→ Usage tracking      aggregates the bedrock run's own summary/usage docs;
                      success-path only, never raises
```

The workflow Lambda is operation-routed (top-level `lambda_function.py` →
`states/<name>/lambda_function.py`, the family shape) — one artifact,
many states, no container; an unknown operation raises.

## Key design decisions (the WHY — details in the Design record)

- **Cue-id referencing (§3)**: the model outputs `start_id`/`end_id`,
  never timestamps — hallucinated timestamps are structurally impossible,
  coverage is machine-checkable (the retryable unit is an id range),
  merging is id arithmetic.
- **JSONL output (§4)**: truncation loses one line, a bad line damages
  only itself; the parser reads strictly first, and a failed line climbs
  the copied salvage ladder — every recovery COUNTED in `line_repairs`,
  unrecovered lines stay honest gaps.
- **Chunking (§5)**: four cut tiers validated by measurement and
  simulation on real long-form assets. Strong pause (≥10 s) cuts
  disjoint; weak pause (≥3 s), speaker-change and hard cuts carry context
  cues so the model can flag `continues_previous`. 3 s is enough to cut
  AT, not BLIND. Config is ENV, never per-request (it would enter the
  workflow_id preimage).
- **No polling (§8)**: bedrock-inference's `callback_task_token` (built
  for this workflow). One held state, EventBridge wake in seconds at any
  run length, constant history footprint.
- **Lambda-invoke, never sibling StartExecution (§8)**: the gateway
  Lambda is the submit brain (identity, staging, closed-run absorption —
  the paid-work-reuse path when a re-run renders byte-identical records).
- **No rollup inference (§6)**: header disagreement across chunks is
  reconciled by coverage-duration-weighted vote in code;
  `content_type_candidates` reports a no-majority split, never
  adjudicates.
- **Caller-absent defaults are CONCRETE, never null**: model_chain →
  `anthropic.claude-sonnet-4-6` (`ca_common.DEFAULT_MODEL_CHAIN`, a
  versioned constant — it moves only with a TOOL_VERSION bump, and it is
  validated through the same registry door as a caller's chain);
  use_batch → explicit `false` (never batch unless asked); max_tokens →
  REMOVED entirely, here AND in bedrock-inference — no cap is ever sent
  to Bedrock, so each model's own output maximum applies and segment
  output can never truncate; a future cap is a deliberate re-addition in
  that service, never a side door. plan.json and provenance always name
  the actual ids.

## Gateway tool contract

Three tools (`gateway_tool/lambda_function.py`;
`gateway_tool/output_contract.json` is the field-level pin,
`input_examples.json` the known-good request per tool):

- `analyze_conversation` — submit. Returns a HANDLE (`job_id`,
  `workflow_id`, `status`, `output_s3_prefix`), never results.
- `get_job_status` — DescribeExecution; `outputs[]` (`{uri, mime,
  label}`) when complete.
- `describe_tool` — version, state machine ARN, description, output
  contract, input examples.

`analyze_conversation` parameters:

| parameter | req | meaning |
|---|---|---|
| `transcript_s3uri` | yes | s3:// uri on the proxy bucket — a speech-diarization run's `final_transcript.json` (PREFERRED: lossless ids, true sentence boundaries, speakers) or a WebVTT file (accepted; caption-split cues and missing speakers reduce segmentation quality, §7a) |
| `transcript_format` | no | `diarization-json` \| `webvtt` — overrides content sniffing |
| `model_chain` | no | Bedrock model ids, primary first (list or comma-separated); absent → the versioned default; every id validated against bedrock-inference's registry before any spend (§7c) |
| `use_batch` | no | absent → `false`: never batch unless asked; `true` forwards the request, bedrock-inference may still downgrade with the reason |
| `history_scope` | no | YOUR asset identity for the inference-history prefix (§7b); absent → derived from the transcript |
| `prompt_version` | no | a NAMED registry version of the `conversation-analysis` prompt (e.g. `default_news`); absent → the category default; in the hash — a different version is different work |
| `idempotency_token` | no | same token + same parameters returns the EXISTING run; also the RERUN mechanism (below) |

**Identity, and the one-token rule.** `workflow_id = uuid5(request +
TOOL_VERSION [+ token])` via the copied `workflow_identity.py`; an
idempotent resubmit is ABSORBED at the gateway (same handle + a pinned
message). The prompt's CONTENT is deliberately NOT in the hash (fetching
it at submit would put a registry read on every submit): a rebuilt
`default` therefore needs a fresh `idempotency_token` to re-analyze an
already-analyzed transcript. Provenance records the actual `source_md5`,
so what ran is never ambiguous. The submit to bedrock always folds THIS
run's uuid in as its idempotency_token — one token, two properties: a
redriven/retried submit absorbs into its own bedrock run (idempotent),
and a caller rerun (fresh caller token → new uuid) re-renders
byte-identical records under a NEW bedrock identity and gets a genuinely
fresh inference, never an absorb into the flawed run it is escaping.

## Every file a run produces

Under the RUN PREFIX `micro-workflow/conversation-analysis/{workflow_id}/`:

| file | status | contents |
|---|---|---|
| `conversation_analysis.json` | **contract — label `analysis`** | THE product: merged `audio_segments[]` (each with `start_id`/`end_id` cue ids AND `start_time`/`end_time` resolved from the transcript — timestamps never pass through a model), header (`overall_content_type`, `top_3_genres`, `content_type_candidates` only on a no-majority split), `segments_gap[]` (uncovered cue-id ranges, honest partials), `provenance` (prompt category/version/`source_md5`, concrete `model_chain`, transcript uri/format, chunk count, history scope, tool_version) |
| `chunk_manifest.json` | **contract — label `chunk-manifest`** | traceability by POINTER: the chunk plan (per chunk: id range, `cut_tier`, `cut_gap_s`, token estimate, context start, `resolved`) and `inference_runs[]` — each entry names the bedrock run's `workflow_id`, `job_id`, `output_s3_prefix`, staged `records_s3uri`, `absorbed` flag, and `history_key` |
| `usage_tracking.json` | **contract — label `usage`** | aggregated spend: per-run mode/`mode_reason`/records/models_used/`estimated_cost` + `usage_uri`, and totals (`total_estimated_cost`, `cost_is_complete`, token counts) |
| `records.jsonl` | internal | the staged rendered records — one line per chunk: `record_id` (`chunk-{index:03d}`), `system`, Converse `messages` with the WebVTT envelope inside. What the model was ASKED |
| `_plan/plan.json` | internal | everything ValidateAndPlan decided: normalized-transcript facts, the prompt-artifact snapshot, the CONCRETE model chain, explicit `use_batch`, chunk plan, history scope. The one document every later state reads |
| `_plan/runs.json` | internal | the submit record (bedrock ids, absorbed flag, history key) — written by SubmitAndWait because a waitForTaskToken handler's return value is discarded |
| `_plan/chunk_results.json` | internal | per-chunk verdicts before merging: parsed header/segments, coverage `gaps`, structural `problems`, `bad_lines`, `ok` |

Internals may change without notice; the three labeled files are stable
API. Under the BEDROCK RUN PREFIX (`inference_runs[].output_s3_prefix` —
that stack's files, that stack's contract): `input_params.json` +
`request.jsonl` (the request pair; request.jsonl is the EXACT rendered
bytes and the run's identity), `response.jsonl` (raw model output per
record), `summary.json` (the authoritative outcome), and its own
`usage_tracking.json`.

And ONE ~500-byte HISTORY POINTER at
`micro-workflow/histories/{scope}/conversation-analysis/{prompt_version}/{ts}-{bedrock_workflow_id}.json`.

## Navigating the histories: from "what ran?" to the exact bytes

The history prefix is the asset-centric index; everything else is two
hops from it. The entry carries NO outcome and NO cost on purpose — a
pointer written once at submit can never disagree with the truth it
points at (`summary.json` stays the single authority).

```
"what ran against this transcript?"
  └─ LIST micro-workflow/histories/{scope}/conversation-analysis/{version}/
        keys are {timestamp}-{workflow_id}.json → chronological for free
     └─ READ one entry:
          prompt.category/version → WHICH template (immutable; get_prompt
                                     fetches its byte-frozen source)
          idempotency_token       → the conversation-analysis run's own
                                     uuid (the one-token rule) → hop to
                                     micro-workflow/conversation-analysis/
                                     {that uuid}/ for the MERGED products
          output_s3_prefix        → the bedrock run:
             ├─ request.jsonl    the EXACT rendered bytes (the question)
             ├─ response.jsonl   raw model output per record (the answer;
             │                   join to chunks by record_id chunk-NNN)
             └─ summary.json     the authoritative outcome and cost
          job_id                 → the execution ARN, for console debugging
```

`{scope}` defaults to `transcript.{diarization-workflow-id}` (the exact
transcript DOCUMENT — a re-diarization is a new scope, correctly), falls
back to `transcript.{sha1(uri)[:12]}` for non-diarization uris, and is
overridable per request via `history_scope` — bind to YOUR asset id
(`asset.{ingestAssetId}`) when you have one, and "everything ever run
against this asset" becomes one prefix list.

## The E2E swimlane

```
 CALLER/AGENT      GATEWAY (this stack)   STATE MACHINE        PROMPT-REGISTRY   BEDROCK-INFERENCE      EVENTBRIDGE
     │                    │                    │                     │                 │                    │
 (1) ├─ analyze_conv ───► │                    │                     │                 │                    │
     │                    ├ workflow_id =      │                     │                 │                    │
     │                    │  uuid5(request +   │                     │                 │                    │
     │                    │  version [+token]) │                     │                 │                    │
     │                    ├─ StartExecution ─► │                     │                 │                    │
     │ ◄── HANDLE ────────┤                    │                     │                 │                    │
     │                    │               Validate and plan          │                 │                    │
 (2) │                    │                    ├── get_prompt ─────► │ (binds default) │                    │
     │                    │                    ├── get_model ───────────────────────► │ (chain valid?)     │
     │                    │                    ├ chunk plan (§5) → _plan/plan.json    │                    │
     │                    │               Render and stage           │                 │                    │
 (3) │                    │                    ├ per chunk: VTT envelope + copied      │                    │
     │                    │                    │ prompt_render → records.jsonl (S3)    │                    │
     │                    │               Submit and wait (waitForTaskToken)           │                    │
 (4) │                    │                    ├── run_inference(records_s3uri, ─────► │ registers token,   │
     │                    │                    │    callback_task_token,               │ StartExecution     │
     │                    │                    │    idempotency_token = OUR uuid)      │                    │
     │                    │                    ├ write history pointer (from the TRUE  │                    │
     │                    │                    │ submit response) then PARK — no poll  │                    │
     │                    │                    ┆  ...sync seconds or batch hours...    ├─ terminal event ─► │
 (5) │                    │                    │ ◄────────────────── SendTaskSuccess ──┤ (waker)  ◄─────────┤
     │                    │                    │      (terminal status AS DATA)        │                    │
     │                    │               Collect and finalize       │                 │                    │
 (6) │                    │                    ├ read response.jsonl, parse JSONL (§4),│                    │
     │                    │                    │ validate coverage (§3), merge (§5-§6),│                    │
     │                    │                    │ write analysis + chunk-manifest       │                    │
     │                    │               Usage tracking             │                 │                    │
 (7) │                    │                    ├ aggregate run summaries → usage label │                    │
 (8) ├─ get_job_status ─► │ ── Describe ─────► │                     │                 │                    │
     │ ◄─ header + 3 outputs[] ┤               │                     │                 │                    │
```

Failure/rerun shapes worth knowing on this diagram: a duplicate (1) with
identical parameters is ABSORBED at the gateway (same handle + the pinned
message); a lost wake at (5) fails the execution at the 26 h timeout, and
a REDRIVE of Submit-and-wait resubmits → absorbs at bedrock → is
completed SYNCHRONOUSLY (redrive is the reconcile); a caller rerun (fresh
token at (1)) re-renders byte-identical records under a NEW bedrock
identity — a genuinely fresh inference, never an absorb into the flawed
run.

## Copied modules

`lib/prompt_render.py` and `lib/workflow_history.py` are byte-identical
copies of prompt-registry's masters; `lib/parse_partial_output.py` is
bedrock-inference's salvage ladder (stdlib-only; json-repair was measured
and rejected at the master); `gateway_tool/workflow_identity.py` is the
family-wide shared identity copy. Copies are `cp`'d verbatim, NEVER
edited in place — in-repo byte-parity tests compare against the sibling
folders when present, and the bundle's parity classes are the cross-repo
gate. Per-consumer variation lives in the CALL, never in the copy.

## Layout

```
cfnstack.yaml               2 Lambdas, 1 machine (ASL inline), 4 params
definitions.json            readability copy of the ASL — edit BOTH; the
                            build and a test fail on drift
build-deploy-s3-dist.sh     two zips (tree workflow + flat gateway);
                            tests gate packaging; ASL drift gate; token
                            guard; internal-docs exclusion gate
create_update_stack.sh      needs --bedrock-lambda-arn / --registry-lambda-arn
lib/                        core logic: transcript (§7a), chunker (§5),
                            envelope (§2), output_parse (§3-§4), merge
                            (§5-§6), ca_common (S3 + §8 sibling invoke),
                            + the three copied modules
states/                     five operation handlers, one folder each
                            (lambda_function.py per state); the router
                            is the top-level lambda_function.py
gateway_tool/               dispatcher (analyze_conversation/get_job_status/
                            describe_tool), output_contract.json,
                            input_examples.json, workflow_identity.py (copy)
test/test_units.py          unit suite (no AWS); fixtures are one full
                            episodic transcript plus two long-form news
                            shape-skeletons (real timings, invented text)
```

## Tests / build / deploy

```bash
python3 -m unittest discover -s test -p 'test_*.py'    # unit suite, no AWS
bash build-deploy-s3-dist.sh --build-only              # tests + ASL gate + zips
bash create_update_stack.sh --bedrock-lambda-arn <arn> \
     --registry-lambda-arn <arn> ...                   # stack create/update
```

The build fails on cfnstack.yaml / definitions.json ASL drift, on any
unit-test failure, and if internal docs leak into an artifact. The
bedrock-inference deployment it points at must support
`callback_task_token` on `run_inference`.

## Gotchas

- **A JSONPath in ASL `Parameters` to a MISSING field fails the state** —
  the trap that shaped the first draft's wiring; the five-state machine
  references only always-present fields, keep it that way.
- **The history pointer is written at SUBMIT, inside the waitForTaskToken
  handler** — the wake payload always carries a `message`, which the
  copied `workflow_history` module reads as "absorbed → skip". Writing
  history from the wake would skip every fresh run.
- **The submit state's retry ladder is safe by absorption**: a re-invoke
  after a successful-but-lost submit absorbs at bedrock (same
  uuid-token), re-registers the callback row, and rewrites runs.json
  identically.
- **`.version` must differ from every bundle version deployed against the
  bucket** — same S3Key: CloudFormation detects key changes, never
  content changes, so a collided version deploys STALE code (CodeSha256
  is the tell).
- **TWO zips**: the workflow zip keeps the lib/ + states/ TREE (imports
  are package-style, `from lib import x`); the gateway dispatcher ships
  its own flat zip and never imports workflow code.
- **Diarization emits NEGATIVE inter-sentence gaps on crosstalk**
  (overlapping speakers) — gap math must never assume monotonic
  timestamps.

---

# Design record

The workflow's design history, compressed from the original working plan.
Section numbers are stable — code comments cite them as `README.md §N`.

## §1. The prompt contract: the workflow binds `default`

The workflow calls `get_prompt(category="conversation-analysis")` with NO
version — it binds `default` (a caller's `prompt_version` overrides). The
prompt declares ONE variable, `{{transcript}}`; chunk/duration context
travels in-band in the transcript envelope (§2) instead of as template
variables, and the id scheme (§3) makes "never invent timestamps"
begging-prose unnecessary — hallucinated timestamps are structurally
impossible instead of pleaded against.

Registry versions are immutable (put_prompt always refuses
VERSION_EXISTS). The pre-GA escape hatch is the SEEDER, not the API: a
`PromptSeedMode` stack parameter — `write-if-absent` (the default and
only prod-sane value) | `overwrite-builtins`, which re-seeds a shipped
prompt in place only when it exists with `origin: builtin` and a
differing source md5, never touching `origin: user` versions. At GA the
parameter stays at write-if-absent: defaults freeze for real. Audit is
unharmed by a rebuilt default: run identity is the rendered bytes, and
`request.jsonl` at each run prefix holds them forever.

**The workflow-provided values catalog**: the render module raises on
missing values AND on undeclared-token values, so the workflow supplies
exactly the INTERSECTION of its catalog (`transcript`, plus explicit
`chunk_start_time`/`chunk_end_time`/`total_duration`/
`transcript_duration` for authors who want them) with
`artifact.variables`. Declare it and it is injected; declare something
the workflow cannot compute and render fails loudly at submit, naming
the token. The limit of genericity, stated honestly: metadata injects
generically; *behavioral* prose ("you are seeing a portion — segment
only within it") belongs to the prompt author.

## §2. `{{transcript}}` substitutes a self-describing WebVTT envelope

Generated per-request from the diarization `final_transcript.json` — one
cue per sentence, **cue id = the sentence's global integer `id`**,
speaker via `<v>` voice tag. NOT the shipped caption VTT
(caption-formatted; line limits split cues, so its boundaries do not map
1:1 to sentence ids).

```
WEBVTT

NOTE
program duration 00:12:04 | transcript covers 00:09:47 of speech
chunk 2 of 3 | coverage cues 40-79 (00:04:10.320-00:08:30.100)
cues 36-39 are CONTEXT from the previous chunk - do not segment them

36
00:03:58.100 --> 00:04:02.440
<v SPEAKER_01>...
```

Why VTT over JSON: roughly half the tokens (JSON repeats five key names
plus quoting per cue); massive subtitle representation in model training
data; native cue ids — which §3 needs. The metadata rides in a `NOTE`
block: legal WebVTT, in-band (the prompt author sees everything the model
sees; template + values = rendered bytes keeps the audit story), and
reduced to just the program duration on single-chunk runs (the common
case). The NOTE deliberately does NOT carry the previous chunk's closing
topic: all chunks ship as N records in ONE `run_inference` submit and run
in parallel — sequential topic-chaining would serialize the pipeline for
a hint the overlap cues already provide.

## §3. Output references cue ids, never timestamps

The model outputs `start_id`/`end_id` (sentence ids). Consequences, each
load-bearing:

- **Timestamps cannot be hallucinated** — they never pass through the
  model; the workflow re-derives authoritative `start`/`end` from the
  transcript by id at merge time.
- **Coverage is validatable**: ids exist, `start_id <= end_id`, ranges
  are disjoint, and their union tiles the chunk's coverage range; a gap
  or overlap has an exact id-range diagnosis.
- **Merge is id arithmetic**: ids are global across the program (a chunk
  IS an id range), so joining per-chunk responses is concatenation plus
  the §5 boundary stitch.
- Output tokens shrink: short ids replace two 12-char timestamps per
  segment.

## §4. Output format is JSONL — not XML, not a single JSON document

First line: header object (`overall_content_type`, `top_3_genres`); then
one line per segment:

```jsonl
{"overall_content_type":"Episodic","top_3_genres":["Drama","Mystery/Thriller","Supernatural"]}
{"start_id":0,"end_id":11,"segment_topic":"...","reasoning":"...","named_entities":[...],"key_events":[...],"sentiment":"...","tone":"...","conversation_type":"...","continues_previous":false}
```

Why not XML: XML's partial-recovery advantage pays off on DEEP NESTING,
and this output is a flat list; meanwhile `reasoning`/`named_entities`
quote transcript text full of `&`, `<`, quotes — models do not reliably
entity-escape, so the trade is "truncated JSON" for
"well-formed-looking XML no parser accepts". JSONL gets XML's recovery
property with `json.loads`: truncation loses at most the final line (the
§3 coverage check names the uncovered id tail); a malformed line damages
only itself; chunk merge is line concatenation. A failed line climbs the
copied salvage ladder (`lib/parse_partial_output.py`: escape pass, then
prefix salvage gated on the cue ids / content type a line is unplaceable
without); every recovery is counted in `line_repairs`. Forced tool-use at
the Converse layer was assessed and declined: one `toolUse` block is a
single nested JSON document — exactly the shape this section rejects —
with no partial recovery in batch.

**Key order is recovery order.** Models follow the example line's key
order (measured essentially universally on a sibling workflow's corpus),
and the salvage ladder's prefix rung keeps everything BEFORE a damage
point — so the example puts the cheap, placeable fields first and free
text last: `start_id`/`end_id`/`continues_previous` up front,
classification vocab next, then names and lists, with the verbatim-quote
field second-last and `reasoning` (the longest free text) LAST.

## §5. Chunking is the workflow's decision, on natural boundaries

Budget: per-chunk input tokens `B` = model context window (from the
bedrock-inference registry) − rendered-prompt overhead − output reserve,
capped by config; VTT tokens estimated conservatively as `chars / 3.5`.
Whole transcript fits in `B` → ONE chunk, no NOTE complexity, no merge —
the common case must stay this simple. Otherwise accumulate sentences to
the budget and pick the cut inside a search window (the last
`CUT_SEARCH_FRACTION` of the accumulated span):

| priority | boundary | chunk relation |
|---|---|---|
| a | largest inter-sentence gap ≥ `PAUSE_STRONG_S` | **disjoint** — a strong pause is non-speech content (music, action, scene change); next chunk starts at the next id with no context cues |
| b | largest inter-sentence gap ≥ `PAUSE_WEAK_S` | cut there, **with context cues** — a weak pause may be a dramatic pause inside one topic |
| c | last `speaker_id` change in the window | cut there; context cues walk back through the preceding complete speaker exchange, capped at `OVERLAP_MAX_S` |
| d | none of the above (single-speaker monologue) | hard cut at budget; context = trailing `OVERLAP_MAX_S` of cues |

**Why two pause tiers — measured on real assets spanning three content
shapes.** A business-news program's gap distribution is bimodal with a
clean silence band: talk gaps stay at or under ~3 s and real breaks
(segment/commercial) are well above 10 s — there, 3 s already clears the
conversational noise floor. A chatty morning news magazine is the
counterexample: a 3 s gap lands every couple of minutes (music stings,
handoffs) — far too frequent to all be topic boundaries. Episodic drama
adds 3-second-plus dramatic beats INSIDE one running conversation.
Conclusion: 3 s is enough to cut AT, not enough to cut BLIND — only a
strong pause earns a disjoint cut; a weak pause cuts with context, the
same safety net as a speaker-change cut. The tier decides the context
policy, not the cut. Simulation on the same assets showed the short-form
case staying a single chunk, chunk sizes within a few percent of budget,
every strong-pause cut landing on a verified real break, and tier d never
firing on real dialogue. One measured quirk the implementation tolerates:
diarization emits NEGATIVE inter-sentence gaps on crosstalk — gap math
never assumes monotonic timestamps.

Validated defaults (ENV-tunable, never per-request): `PAUSE_STRONG_S =
10.0`, `PAUSE_WEAK_S = 3.0`, `OVERLAP_MAX_S = 30`, `CUT_SEARCH_FRACTION
= 0.15`. Output truncation is structurally off the table: NO max_tokens
knob exists anywhere in the family — bedrock-inference never emits an
output cap, so each model's own maximum applies.

**Context cues are context-only — coverage stays disjoint by
construction.** The NOTE block marks them ("do not segment cues 36–39;
coverage begins at 40"), so overlap never produces duplicate segments and
merge needs no dedup. The model may set `continues_previous: true` on its
FIRST segment; the workflow stitches it to the previous chunk's last
segment (union the id ranges, keep the earlier
`segment_topic`/`reasoning`, union `named_entities`/`key_events`).

## §6. Header reconciliation: majority vote, deterministic rollup — NO extra inference

- One chunk → its header verbatim. No vote, no rollup.
- `overall_content_type`: coverage-duration-weighted vote across chunks.
  Strict majority → that single value. No majority →
  `overall_content_type` is still the top-weighted value, and the merged
  header ALSO carries `content_type_candidates`: the deduped distinct
  values, weight-ordered — the disagreement is reported, not adjudicated.
- `top_3_genres`: union all chunks' genres, dedup, weight each by the
  coverage duration of the chunks that named it, take the top 3.
- Reconciliation is pure code over per-chunk headers: deterministic,
  free, and auditable (each chunk's raw response survives at the run
  prefix).

## §7. Architecture: a Step Functions child, deliberately

Standard micro-workflow shape — gateway tool, ONE state machine, outputs
under `micro-workflow/conversation-analysis/{workflow_id}/`. Why a state
machine when the substitution would fit one Lambda:

1. **Duration makes it mandatory**: the pipeline WAITS on
   bedrock-inference, and a batch-mode run is hours — past any Lambda's
   15-minute cap.
2. **Traceability and per-state visibility**: plan / render / submit /
   collect+merge / usage each visible, each with its own retry ladder.
3. **This machine is the TEMPLATE for llm-prompt workflows** (see Shape
   above): the workflow-specific part is one stage; the skeleton is
   reusable.

Every schema description, refusal message, and prompt is written for an
AGENT reader — this target is driven by other agent systems: refusals
name what to do instead (the family's errorCode + instructive-message
convention).

### §7a. Input: a transcript URI, two accepted formats

The workflow takes a **transcript URI, never raw media** — it starts
where speech-diarization ends. Format detected by content (extension as
hint, sniff as authority), with an optional `transcript_format` override:

| format | ids | sentence boundaries | speakers | verdict |
|---|---|---|---|---|
| diarization `final_transcript.json` | native global integer `id` | true sentence units, float sec | per-sentence `speaker_id` + confidence | **preferred — lossless for every §5 signal** |
| generic WebVTT | cue identifier if present, else sequential position | CAPTION units — line/duration limits split sentences mid-clause | `<v>` tag if present, else a single UNKNOWN speaker | accepted compatibility on-ramp |

Both normalize into the internal sentence list `{id, start, end,
speaker_id, text}`. VTT degradation is graceful and DOCUMENTED, not
hidden: no `<v>` tags disables tier-c speaker-change cuts;
caption-split cues make `start_id`/`end_id` reference fragments rather
than sentences — resolved timestamps stay correct either way. (The
diarization VTT itself is the worst of both — caption-split AND id-less;
the JSON sibling is always the right handle.)

### §7b. Histories: scope defaults to the transcript's identity

- `idempotency_token` can NEVER be the query key — it is folded into the
  hash and discarded, never stored, never echoed (pinned family-wide).
- The upstream workflow_id CAN, and is the default binding: a
  `transcript_s3_uri` inside a diarization run prefix derives
  `transcript.{workflow_id}`; any other URI derives
  `transcript.{short-hash-of-uri}`. Deterministic, zero caller effort.
- Binding to the CALLER'S own asset identity is the caller's
  responsibility, via `history_scope` (scope-charset constrained) — the
  service cannot know a caller's asset naming.

"Everything run against this transcript" is ONE prefix list; "runs at
version X against this transcript" is the deeper prefix. A cross-asset
"every run at version X" walks scopes first — the accepted price of the
asset-first layout.

### §7c. Model chain is caller-driven, validated here

An optional `model_chain` (Bedrock model ids, primary first) passes to
bedrock-inference VERBATIM; omitted → the versioned default. Validated
before submitting, against bedrock-inference's registry (`get_model`):

1. every id exists (the refusal names the id and points at
   `list_models`);
2. every model is a TEXT model;
3. **the §5 chunk budget is computed from the MINIMUM context window
   across the chain** — a fallback that cannot fit the chunk is not a
   fallback.

Rule 3 is the non-obvious coupling: the caller's model choice changes
the chunking, which changes the rendered bytes, which changes the
bedrock-inference workflow_id — same transcript + different chain =
different run, correctly.

### §7d. Retries: bedrock-inference owns transport, we own semantics

bedrock-inference already runs a per-record retry ladder
(throttle/5xx), per-record cross-family model-chain fallback inside one
held slot, and a bounded batch repair tail — a FAILED line on a batch
run failed twice on two code paths. This workflow adds NO transport
retries. What remains OURS, because no sibling can judge it: a record
that SUCCEEDED whose text fails §3 coverage validation. There is NO
in-machine coverage-retry loop (cut in review — clean first pass is the
live norm, and the loop cost two states plus attempt bookkeeping for a
rare case): uncovered ranges land in `segments_gap` honestly and the run
SUCCEEDS; the semantic retry is the CALLER's fresh `idempotency_token`,
which works ONLY because of the one-token rule (a rerun re-renders
byte-identical records under a NEW bedrock identity instead of absorbing
into the flawed run). And read `records_failed` on every status check —
a SUCCEEDED run with failures is a documented shape, not an anomaly.

### §7e. Output contract: we parse and merge — never pass-through

bedrock-inference's `response.jsonl` carries raw `output_text` per
record; it never parses model output, by design. Parsing, validating,
and merging IS this workflow's product — a caller handed raw per-chunk
JSONL would have to re-implement the workflow. Three labeled outputs
(shared `outputs[]` contract, `{uri, mime, label}`): `analysis`,
`chunk-manifest`, `usage` (see the file table above). Cost/usage live in
the rollup because the caller's question is "what did this analysis
cost"; the chunk-manifest points at each bedrock run's own
`summary.json`/`usage_tracking.json` for the per-chunk answer.

## §8. Reaching bedrock-inference: direct, at the LAMBDA layer, push-woken

LLM-prompt workflows get direct access to bedrock-inference — never an
agent-gateway round-trip (that layer is transport + auth for AGENTS; a
state machine doing an OAuth dance mid-execution is wrong, and the
family already composes children directly).

**"Direct" means invoking bedrock-inference's gateway-tool LAMBDA**
(`lambda:InvokeFunction` with the same tool-call event an agent gateway
would send), never raw `StartExecution` on its machine. The Lambda IS
the submit brain — workflow_id derivation from the rendered bytes,
parameter validation, and ExecutionAlreadyExists absorption. Raw
StartExecution would force every consumer to re-implement that preimage,
and drift there is not a broken test — it is silent duplicate Bedrock
spend. A nested `startExecution.sync` was weighed and rejected: the
machine's real input is an internal S3 pair at a prefix derived from the
gateway-computed workflow_id (tools are the contract surface), and
`.sync` cannot absorb a CLOSED execution — which two real cases need:
(a) an at-least-once submit whose response was lost retries after a fast
sync run already finished; (b) cross-run dedup as a FEATURE — a re-run
that renders byte-identical records SHOULD absorb into the finished run,
reusing paid-for inference at zero cost. Both need
describe-the-closed-run-and-return-its-handle, which is the gateway's
absorb.

**NO POLLING — the wait is PUSH-BASED** (runs elapse from under a minute
to ~24 h). `run_inference` accepts an optional `callback_task_token`: on
the run's terminal event bedrock-inference calls SendTaskSuccess
CARRYING the terminal status (FAILED/TIMED_OUT too — the verdict is DATA
for the consumer, never a mid-machine detonation); if the submit absorbs
into an already-FINISHED run, the token is completed synchronously in
the same call. The notification lives in BEDROCK-INFERENCE, once —
N consumer workflows would otherwise each need a table, a rule, and a
waker, with every completion fanning out to all of them for N−1 misses.
Bedrock-inference owns one TTL'd token table (multiple waiters per run —
absorbed resubmits append), ONE EventBridge rule on its own machine's
native status-change events, and ONE waker Lambda. A consumer's whole
wait is one `lambda:invoke.waitForTaskToken` state:

- No registration race exists: the token rides inside the submit
  request, stored before the caller learns the execution name.
- Resume latency is EventBridge delivery (seconds) at any run length;
  constant history footprint.
- Deadman without polling: the wait state's 26 h TimeoutSeconds has NO
  Catch; a timed-out wait FAILS the execution, and a REDRIVE of the
  submit state resubmits, absorbs into the finished run, and is woken
  synchronously by the absorbed-closed path. Redrive IS the reconcile —
  and this stack needs no DescribeExecution IAM on the sibling at all.
- A token is transport, not knowledge — bedrock-inference learns nothing
  about callers. (`states:SendTask*` cannot be resource-scoped — a task
  token is an opaque bearer credential, not an ARN; the real controls
  are the unguessable token and the callback table's write policy.)

**Records transport: this workflow always STAGES.** `run_inference`
accepts inline `records` or `records_s3uri` — transport-only, one
workflow_id either way. Text-only records would usually fit inline, but
the 256 KB Step Functions payload limit binds long before the 6 MB
Lambda invoke cap (state I/O carries the whole set), so states pass
URIs, never bodies — and the sibling workflows this design templates
(vision workflows with base64 frames) cannot ride the invoke payload at
all. All chunks ride ONE submit — multi-chunk runs are exactly the shape
that benefits from batch mode when the caller opts in (bedrock-inference's
guard decides).

Wiring: the bundle passes the bedrock-inference gateway Lambda ARN into
consumer stacks as an explicit parameter (never Fn::ImportValue). IAM
per consumer: `lambda:InvokeFunction` on that one function,
`s3:GetObject` on `micro-workflow/bedrock-inference/*`, `s3:PutObject`
on this workflow's own prefix (for staging).
