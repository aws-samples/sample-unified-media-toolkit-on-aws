# prompt-registry-on-aws

The prompt registry (Design record §3, §6, §7): versioned, immutable prompt
templates on S3, parsed ONCE at save time, served to caller workflows that
render and submit to `bedrock-inference`. **No state machine** (Design record,
decision 8) — pure CRUD, gateway-target-only.

The seven gateway tools, the Custom::PromptSeeder, `cfnstack.yaml`, the
build/deploy scripts, the output contract and the unit suite all exist; the
frozen core (parser, render, history, vocabulary) was built first because of
the freeze bias below.

## The stack (cfnstack.yaml)

The smallest in the family: NO state machine, NO container/CodeBuild/ECR, NO
DynamoDB. ONE flat zip, TWO handlers (media-processing's pattern): the
gateway dispatcher and the seeder. The seeder goes through the SAME
`save_prompt` path as put_prompt (one parser, one pair-write — seeded and
user artifacts cannot drift), walks prompts/ from the zip, writes each pair
WRITE-IF-ABSENT (S3 conditional put as the race-proof half), lands everything
`stage: prod, origin: builtin`, verifies filename stem == header Version
BEFORE writing, and its Delete is a no-op success (shared bucket; prompts
outlive any one stack). **`PromptSeedMode: overwrite-builtins` is the DEV
BACKDOOR** (settled, for rebuilding category defaults while the
llm-prompt-workflows are developed): the seeder rewrites an existing
pair in place ONLY when it is `origin: builtin` AND the shipped source md5
differs; `origin: user` versions are never touched in any mode, and
put_prompt stays fully immutable either way (`save_prompt(overwrite=True)`
is reachable from the seeder alone). **The parameter DEFAULTS TO
overwrite-builtins PRE-GA** (settled decision: the prompts are
under heavy development, every deploy should reseed changed builtins, and
the bundle deliberately passes nothing so the posture lives in exactly ONE
place — this Default). AT GA flip the Default to write-if-absent;
`test_seed_mode_defaults_overwrite_builtins_UNTIL_GA` fails until that flip
is made in the same diff. Flipping the parameter changes the custom
resource's Properties, so the seed re-runs in the same stack update. Three
parameters (ResourcePrefix, ProxyBucket — no defaults; PromptSeedMode —
defaults overwrite-builtins pre-GA); gateway IAM scoped to
`_promptregistry/*` only; the seeder role deliberately cannot delete.
Known-benign lint: W1034 under --build-only, W3005 on the seeder's DependsOn
(the bundle's log-group hygiene rule wins over cfn-lint's redundancy claim —
same accepted tension as bedrock-inference's StartBuildLambda).

## The seven tools

| tool | returns | refusals |
|---|---|---|
| put_prompt(category, text, stage=dev) | identity + source_s3uri + token list + vocabulary WARNINGS (never blocking) | VERSION_EXISTS, INVALID_PROMPT, PROMPT_TOO_LARGE, INVALID_PARAMETER |
| get_prompt(category, version?) | the full artifact + source_s3uri; version optional → `default`; NAMED missing version never falls back | PROMPT_NOT_FOUND |
| list_prompts(category?) | categories → versions with lifecycle metadata (per-version GETs) | PROMPT_NOT_FOUND |
| promote_prompt(category, version, to_stage=prod) | stage + previous_stage + promoted_at; idempotent-with-message at/beyond target | PROMPT_NOT_FOUND, INVALID_PARAMETER |
| delete_prompt(category, version) | deleted + stage_was + name-freed message | PROD_UNDELETABLE, PROMPT_NOT_FOUND |
| deprecate_prompt(category, version) | deprecated + deprecated_at; idempotent | DEFAULT_PROTECTED, PROMPT_NOT_FOUND |
| describe_tool | version, description, output_contract (kind inline ×4), the WHOLE vocabulary — and NO state_machine_arn, honestly | — |

A stray `idempotency_token`/`uuid` is warned-and-ignored on EVERY tool
(nothing here starts an execution, so per the family gateway contract the
warning talks about the token — "nothing here reads it").

## How a user interacts with the service (the whole lifecycle, drawn)

Two phases, separated on purpose: authoring talks only to the registry;
inference never talks to the registry at run time — only before it.

```
════════════════════════════ PHASE A — AUTHORING ═══════════════════════════════
 (prompt engineer, or an admin agent over MCP)

   ┌──────────────┐          ┌─────────────────────┐          ┌─────────────────────────────┐
   │ Prompt       │          │ prompt-registry     │          │ S3 (proxy bucket)           │
   │ Engineer     │          │ (gateway tools)     │          │ _promptregistry/            │
   └──────┬───────┘          └──────────┬──────────┘          └──────────────┬──────────────┘
          │                             │                                    │
 (1) author the .md                     │                                    │
     <!-- Version: v2 -->               │                                    │
     # System Prompt ...                │                                    │
     # User Prompt ... {{tokens}}       │                                    │
          │                             │                                    │
 (2) ─── put_prompt(category, .md) ───► │                                    │
          │                   parse ONCE (grammar → variables[])             │
          │                   lint names (vocabulary.json):                  │
          │                     alias   → "prefer canonical X"   [warning]   │
          │                     novel   → "did you mean X?"      [warning]   │
          │                   version exists?  ──► REFUSED (immutable —      │
          │                                        an edit is a NEW version) │
          │                             ├── write {category}/v2/source.md ──►│
          │                             ├── write {category}/v2/artifact.json►│
          │ ◄── saved + warnings ────── │                                    │
          │                             │                                    │
 (3) ─── list_prompts / get_prompt ───► │ ◄──── read artifact.json ──────────│
          │ ◄── versions, variables ─── │                                    │
          │                             │                                    │
 (4) ─── deprecate_prompt ────────────► │  sets a flag; NO DELETE ever       │
          │                             │  (history entries must never dangle)│

════════════════════════════ PHASE B — INFERENCE ═══════════════════════════════
 (a caller workflow or agent; the registry is NOT in the hot path)

   ┌──────────────┐   ┌─────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
   │ Caller       │   │ prompt-registry │   │ copied modules       │   │ bedrock-inference  │
   │ workflow /   │   │                 │   │ (in the caller)      │   │                    │
   │ agent        │   │                 │   │                      │   │                    │
   └──────┬───────┘   └────────┬────────┘   └──────────┬───────────┘   └─────────┬──────────┘
          │                    │                       │                         │
 (5) ─── get_prompt(category, version) ──►             │                         │
          │ ◄── artifact {system, user_template,       │                         │
          │              variables[]} ──               │                         │
          │                    │                       │                         │
 (6) gather VALUES for each declared token             │                         │
     (fetch metadata, chunk transcript,                │                         │
      prepare frame/grid images)                       │                         │
          │                                            │                         │
 (7) ─── render(artifact, values) ─────────────────────►  prompt_render.py       │
          │            missing/unknown value? ──► RAISES (no 'n/a', ever)        │
          │ ◄── {system, messages[]}  = the RENDERED BYTES                       │
          │                                            │                         │
 (8) ─── run_inference(records=[{record_id, system, messages}], model_chain…) ──►│
          │                                            │       workflow_id =     │
          │                                            │       hash(rendered     │
          │ ◄── HANDLE {job_id, workflow_id, output_s3_prefix, status} ───────── │
          │                                            │                         │
 (9) ─── write_history_entry(submit_response, scope, category, version) ─►       │
          │                       workflow_history.py:                           │
          │                       absorbed (message present)? → SKIP             │
          │                       else write ~300-byte POINTER to                │
          │                       micro-workflow/histories/{scope}/{category}/   │
          │                                    {version}/{ts}-{workflow_id}.json │
          │                                            │                         │
 (10) ── get_job_status(job_id) … poll ─────────────────────────────────────────►│
          │ ◄── SUCCEEDED + outputs[{response, summary, usage}] ──────────────── │
          │      then read response.jsonl from S3                                │

════════════════════════ PHASE C — RERUN / COMPARE / AUDIT ═════════════════════

 (11) same prompt+data resubmitted            → ABSORBED into the existing run
      same prompt+data + fresh idempotency_token → NEW run, NEW prefix (diffable)
      prompt edited (= new version) → different rendered bytes → re-runs naturally

 (12) "what ran against this asset?" — LIST one prefix:
        micro-workflow/histories/{scope}/…      ← chronological, ts-first names
      each entry points at:
        prompt {category, version}  → the EXACT immutable template (Phase A)
        output_s3_prefix            → request.jsonl (exact rendered bytes)
                                      + summary.json (authoritative outcome)
```

Three properties the shape guarantees, step by step: **(2) vs (7)** — the
grammar is parsed once at save; render just substitutes, so a caller can never
disagree with the registry about what a token means. **(7) → (8)** — identity
is the RENDERED BYTES, so an edited prompt re-runs naturally and two routes to
identical bytes dedupe naturally; the prompt version travels in the history
entry (9), not the hash. **(9) + (12)** — the registry never learns about runs
and inference never learns what a prompt version is; the history prefix is the
only join, and it is a pointer, so it cannot disagree with the truth it points
at.

## Layout

```
cfnstack.yaml                        the stack: 2 Lambdas, 1 flat zip, nothing else
build-deploy-s3-dist.sh              flat-zip build; tests gate packaging; token guard
create_update_stack.sh               create-or-update; derives the template URL from the BUILT copy
prompts/{category}/{version}.md      the shipped builtins, one file per version —
                                     conversation-analysis + shot-metadata-
                                     enrichment (each with a news variant),
                                     and the workflow-owned fi-*, ad-*,
                                     cast-* and artwork-* categories
vocabulary.json                      the reviewed variable vocabulary (a lint)
gateway_tool/lambda_function.py      the seven-tool dispatcher (save_prompt is THE save path)
gateway_tool/seed_prompts.py         Custom::PromptSeeder — same save path, write-if-absent
gateway_tool/output_contract.json    kind-inline contract served by describe_tool
gateway_tool/prompt_md.py            save-time parser — REGISTRY-ONLY, never copied
gateway_tool/vocabulary.py           classify/suggest for put_prompt warnings — REGISTRY-ONLY
gateway_tool/prompt_render.py        COPIED-MODULE source of truth (render)
gateway_tool/workflow_history.py     COPIED-MODULE source of truth (§7 entries)
test/test_units.py                   unit suite incl. golden vectors, the real builtin corpus,
                                     all seven tools, the seeder, the template
```

S3 layout (directory-per-version, two fixed-name objects — settled):
`_promptregistry/{category}/{version}/source.md` + `artifact.json`. A version
is a PAIR (byte-preserved source + parsed artifact); the shared prefix keeps
the pair associated by location, makes immutability one prefix-existence
check, and lists versions with one delimiter LIST. The predecessor (v5)
`_promptconfig/{namespace}/{category}/{version}.md` maps: namespace →
`origin` metadata, filename stem → the version prefix.

## Why there is NO DynamoDB table (settled)

S3 is the store AND the index. Three reasons, recorded so this is not
re-litigated: (1) DDB's 400 KB item cap fails late — the largest builtin
today is ~16 KB but prompts grow by nature (few-shot examples), the item would
carry source + parsed artifact (≈2× the text), and the failure arrives as a
hard put error months in; (2) the whole query surface is
get-by-(category,version) / list-versions / list-categories, which the key
layout answers by prefix listing at a scale where latency is irrelevant;
(3) immutability is an S3 property (write-if-absent) but a DDB code
discipline. If a queryable index is ever needed it is built BEHIND the S3
source of truth (Design record, decision 5's pattern). `put_prompt` carries a
~1 MB sanity bound — not for S3's sake, but because a multi-megabyte "prompt"
is a paste accident that would ship to a model at real token cost.

## The freeze (read before touching gateway_tool/)

`prompt_render.py`'s output feeds `workflow_id` — the hash of rendered bytes —
so ANY behavior change here re-identifies every request of every consumer and
must bump every consumer's `.version` (the family contract's golden-vector
rule, micro-workflow-bundle/README.md §C2; Design record §6). `TestRenderGoldenVectors` pins fixed artifact+values
→ fixed sha256; a failure there is either an unintended change (fix the code)
or an intended one (update the vector AND sweep the consumers). The module is
deliberately dumb — no grammar, no metadata, no defaulting — so it never needs
to change:

- **The legacy `'n/a'` defaulting is deliberately killed.** The predecessor's
  `replaceVariables` substituted the literal string `n/a` for an undefined
  value; a prompt that silently ships "n/a" to a model is the
  invisible-behavior-change class this design exists to end. Missing values
  RAISE; so do values for undeclared tokens (a typo'd key that substitutes
  nothing is the same failure mode).
- **Values are keyed by the raw declared token string** (`"{{scene:transcript}}"`,
  braces and all) — exactly `artifact.variables[].token`. Two spellings of one
  variable (`{{ scene }}` vs `{{scene}}`) are two declared tokens; the parser
  dedups on raw inner text and render must find each spelling in the template.
- **Substitution hits BOTH sections** (system + user_template) at every
  occurrence — tokens are collected from both and real builtins use both.
- **Only a list of CONVERSE-SHAPED blocks is content BLOCKS** (settled:
  "blocks means image blocks; json means nested data") — every
  element a SINGLE-KEY dict keyed text|image|document|video. Spliced into the
  user message at each token site, blocks byte-untouched; blocks in the
  system prompt raise (Converse system is text). A list of PLAIN data dicts
  (scene_descriptions, shot_list) stringifies as JSON — under the earlier
  any-list-of-dicts rule it would have spliced scene payloads as if they
  were images. Any other dict/list renders
  as compact JSON — `ensure_ascii=False`, `(",", ":")` separators, insertion
  order (NOT sorted: author field order is often prompt-meaningful). The
  `ensure_ascii` choice is part of the freeze — escaping would change bytes,
  hence workflow_id, for every non-ASCII transcript.
- **bool and None are TypeErrors**, not stringified: there is no one true
  string for a bool, and None is a missing value wearing a key.

`workflow_history.py` (Design record §7): pointer entries at
`micro-workflow/histories/{scope}/{category}/{prompt_version}/{ts}-{workflow_id}.json`,
written once at submit by the CALLER, ~300 bytes, no outcome/usage duplication
(the reader follows `output_s3_prefix/summary.json`). It takes the whole
submit RESPONSE and returns None on an absorbed resubmit (`message` present) —
the skip decision lives in the module so nine callers cannot each forget it.
`submitted_at_ms` is injected, never read from a clock in the module.

`prompt_md.py` is registry-only by design: the grammar (`{{a:b+c}}`,
legacy `{{a+b}}`, HTML-comment headers, case-insensitive `# System Prompt` /
`# User Prompt` split) is parsed exactly once, at save time. Consumers get
pre-declared `variables` and never see the grammar — that is what keeps the
copied render module small enough to freeze. Stricter than the predecessor
where `put_prompt` must be strict: missing sections, missing `Version:`
header, and empty tokens all raise instead of half-parsing.

## The vocabulary (vocabulary.json, settled)

**The naming rule (settled once, so it is not re-litigated per token): plural
when the name is the ELEMENT noun, singular when it is a CONTAINER or mass
noun.** `scene_descriptions`/`tags`/`frame_images`/`program_names` are plural
because each element IS one scene description / tag / frame image / name.
`transcript`/`shot_list`/`gps_location`/`image` are singular because the name
names the container — one transcript document holds many timestamped cues,
one shot_list holds many shots (the corpus's own precedent: nobody writes
`shot_lists`). The crude form "plural iff the value is a list" is WRONG —
shot_list already violates it. And the plural of a container noun changes
MEANING, not number: `transcripts` means multiple transcript documents, which
is exactly what tts's `transcripts` parameter means (N texts, one per clip) —
a future prompt taking multiple transcripts uses that contract and that name.

A reviewed LINT over variable names, not a runtime gate: the per-prompt
contract stays the artifact's own parsed `variables` (render enforces it
exactly); the vocabulary exists for CROSS-prompt consistency, which nothing
was watching — measured drift in the ported corpus: `dialogue`/`dialogues`,
`camera_shot`/`camera_shot_type`, `sound_event`/`sound_events`,
`text_on_screen`/`texts_on_screen`, `tag`/`tags`, and three spellings of
"frames as blocks". Rules:

- `TestVocabulary` pins coverage over the REAL corpus in both directions: a
  novel name in any builtin fails until vocabulary.json is amended in the
  same diff (verified by mutation), and aliases may not collide or fork.
- `put_prompt` WARNS on novel names — never refuses.
  Prompts change weekly by design; the registry can judge a name's novelty,
  not its validity. The warning brain is `gateway_tool/vocabulary.py`
  (registry-only): classify → canonical | alias (prefer the canonical
  spelling) | novel (+ a "did you mean" suggestion via prefix,
  singular/plural flip, then difflib).
- **`provided_by: "workflow"` marks WORKFLOW-PROVIDED variables**:
  `transcript`, `chunk_start_time`, `chunk_end_time`,
  `total_duration`, `transcript_duration` are injected programmatically by
  the calling llm-prompt-workflow (the conversation-analysis values-catalog
  pattern) — prompt authors DECLARE them to
  receive the value and never compute them. The annotation rides to agents
  through describe_tool's vocabulary; it is documentation, not a gate.
- **Aliases are ATTESTED legacy spellings only — never guesses** (settled
  when asked about `scene`/`scenes`/`segment`): the parser normalizes
  NOTHING (names pass through verbatim; aliases exist only in this lint),
  guessable variants have no principled stopping point, and the suggestion
  mechanism catches them all anyway — `TestClassify` pins that `scene`,
  `scenes` and `segment` suggest their canonicals while staying out of the
  alias map.
- **The `SupportedVariables:` header is RETIRED** — a majority of the ported
  headers had drifted from actual usage (some headers parse prose like
  "optional", "or", "w/" as names). The parser carries it as inert `aux`;
  nothing validates against it and new prompts should not author it. The
  parsed `variables` supersede it.
- Consolidations are encoded as canonical-plus-aliases, NOT applied to the
  ported builtins (fidelity first; a rename ships as a new prompt version
  when each caller workflow is rebuilt): `latitude`+`longitude` →
  `gps_location`; `scene_frame_images`/`shot_frame_images` → `frame_images`;
  roots follow the predecessor's own VALUE names — `scene_description` →
  `scene_descriptions`, `segments` → `segment_descriptions` (the values are
  LISTS; the legacy structure mappers build exactly those
  names); the singular/plural leaf drift resolves toward the plural (or more
  precise) spelling; and `dialogue`/`dialogues` fold into `transcript` —
  the same spoken-text concept, spelled three ways across
  the composites, canonical to the dominant spelling and the upstream
  artifact's name; and `program_name`+`potential_program_names` →
  `program_names` — 1..N comma-separated names, the
  known-vs-candidate framing carried by the prompt's prose (no ported
  prompt used both, so the merge collides nowhere); and `tags`/`tag` →
  `relevant_tags` — the value-name precedent again:
  scene-metadata's output schema and the legacy value-builder both name the
  field `relevant_tags`, so the model already reads that key wherever
  the token leaf said `tags`.
- **`grid_images` is NOT an alias of `frame_images`** (corrected in review):
  frame_images is a list of SINGLE full frames, grid_images a list
  of TILED multi-frame grids — one image : many frames is a different
  contract, and the legacy shot/scene frame helpers say outright that grid
  tiling is not used on their paths (it downscales small marks into
  illegibility). `face_images` and `image` likewise stay distinct —
  contracts, not spellings.

## Category ownership (settled)

The registry holds TWO species of category and labels them instead of
hiding either:

- **Agent-facing**: a menu — an agent lists, renders, and submits via
  bedrock-inference itself (none shipped today; the ported menu prompts
  were deleted as unconsumed).
- **Workflow-owned** (`ad-*` → audio-description; `fi-*` →
  frame-inspection; `cast-*` → cast-resolution; conversation-analysis;
  shot-metadata-enrichment): the owning workflow's judgment CONFIG,
  bound by category name at its validate state. The `Audience:` header
  is MACHINE-READABLE (the SERVE_INTERNAL gate's
  input): `workflow-internal` (ad-*/fi-*/cast-*) hides a category from
  the standard agent surface entirely; `agent-tunable`
  (conversation-analysis, shot-metadata-enrichment) keeps it visible —
  those two expose caller `prompt_version` parameters and genre
  variants by design. Editing any of them is RETUNING that workflow,
  and the edit gate is that workflow's eval (audio-description: the
  reference-corpus diff).

Naming: workflow-owned categories are OWNER-PREFIXED (`ad-writer`,
`fi-credits-read`). frame-inspection's REQUEST parameter keeps the bare
task name (`category: "credits-read"`) — the API vocabulary — and maps
to the registry name via `fi_common.CATEGORIES[task]["prompt_category"]`
(renamed from the bare `credits-read`/`title-read`/`claim-verify`
while frame-inspection was still internal, so nothing external saw the
old names). Deploy note: a reseed only ADDS pairs; pairs seeded under
retired names linger in S3 (a `default` version is protected from
deprecate AND delete), harmless — removable only by a manual S3 sweep
of the old `_promptregistry/{category}/` prefixes.

## Lifecycle: stage marks (settled; diverges from the original plan — Design record §3.3)

The original plan said "there is no delete." That blanket rule
taxed exactly the weekly-iteration workflow the registry serves — trying
instructions burns version names into permanent junk — so it is REPLACED by a
staged lifecycle:

| stage | delete? | mutate content? | leaves |
|---|---|---|---|
| `dev` / `test` (default: dev) | yes — frees the name for reuse | never (edit = delete + re-put) | promote → prod |
| `prod` | NO — `delete_prompt` refuses loudly | never | `deprecate_prompt` flag only |

Rules that make it sound:

- **Promotion is a one-way ratchet** (dev/test → prod, never back): a
  demotable prod version is a deletable prod version, so demotion would
  reduce prod's guarantee to dev's.
- **Immutability splits precisely**: CONTENT (source.md, system,
  user_template, variables) is immutable at every stage; LIFECYCLE metadata
  (`stage`, `deprecated`) is mutable — deprecate's flag set that precedent.
- **The mechanism is an in-artifact field** (settled): `stage`
  lives in artifact.json beside `origin`/`deprecated`, with
  `promoted_at`/`deprecated_at` stamps. source.md is NEVER rewritten;
  artifact.json's content fields freeze at put and promote/deprecate
  read-modify-write ONLY the lifecycle fields (the implementation pins
  content fields byte-identical across a promote). Rejected alternatives:
  stage-in-key (promotion becomes copy+delete, breaking the permanent-prefix
  property mid-flight) and a sidecar marker object (every stage check pays a
  second read; promotion gets an inconsistency window). Consequences
  accepted: list_prompts pays per-version GETs (already implied — it returns
  origin/comment/created_at, which live in the artifact; stage rides free),
  and delete_prompt's read-then-check-then-delete has a theoretical
  promote/delete race, acceptable at human-and-agent editing traffic and
  documented rather than engineered away. get_prompt returns `stage`, so a
  caller can refuse to bind a dev prompt into a prod pipeline.
- **Why deleting dev versions is safe enough**: run identity is the RENDERED
  BYTES, never the version name. Every run's own prefix keeps request.jsonl
  (the exact bytes) and summary.json forever, so a deleted dev version
  dangles only the human-facing `prompt.version` pointer in history entries —
  and only for runs made against a version its author chose not to promote.
  Documented cost, accepted; `delete_prompt` does NOT scan for runs
  (histories are keyed by scope — a version-wide scan is unbounded).
- The seeded builtins land as `prod`, `origin: "builtin"`.
- **`default` is the category's guaranteed entry point** (settled):
  born `prod` (put_prompt forces the stage when the parsed version is
  `default`, noted in the response), never deprecatable
  (`DEFAULT_PROTECTED` — a version-less get_prompt resolves to it, so
  deprecating it orphans every such read), never deletable (falls out of
  prod). promote_prompt on it is the idempotent already-at-target success —
  no special case. And get_prompt's `version` is OPTIONAL → `default`, with
  one line drawn deliberately: an ABSENT version means default; a NAMED
  version that does not exist is PROMPT_NOT_FOUND, never a silent fallback
  (the predecessor substituted default.md silently — the
  invisible-behavior-change class this design kills).
- Tool surface: put_prompt (stage=dev default) / get_prompt / list_prompts /
  promote_prompt / delete_prompt / deprecate_prompt / describe_tool — seven.
  (Predecessor precedent: it HAD a delete, refusing only `default*`; the
  plan's blanket no-delete was the new strictness, now softened back with a
  principled boundary.)

## Two histories, and only one survives (common confusion, settled)

The predecessor had prompt EDIT history (`_prompt_history/.../versions/{ts}.md`
— text snapshots) and inference histories. The edit history has no successor
mechanism: versions are immutable and an edit is a new named version, so the
registry IS the edit history. Only the INFERENCE history (Design record §7)
exists in the new design, and it is the caller's, via `workflow_history.py`.
Old `_prompt_history/` objects are legacy and stay where they are.

## Gotchas

- **The golden-vector digests are load-bearing, not documentation.** Verified
  by mutation: changing the JSON separators fails the vector. If you "fix"
  formatting in `_scalar_text`, you are changing every consumer's request
  identity.
- **The builtin corpus doubles as the parser's regression suite** —
  `TestRealBuiltinsCorpus` parses every real builtin file and asserts each
  header `Version:` equals its filename stem. Editing a builtin here means
  the registry's seeded copy changes on next deploy (write-if-absent: only
  NEW version names land) — an edited prompt ships as a NEW version name,
  never an overwrite. ONE exception, deliberate and deploy-gated: under
  `PromptSeedMode: overwrite-builtins` (the dev backdoor above) an edited
  builtin IS rewritten in place on the next seed — which is exactly how a
  category default is rebuilt for a new caller workflow while staying
  `default`-bound.
- **`test/` has no `__init__.py`** (house convention): run a single test with
  `discover -s test -k <name>`, not `python3 -m unittest test.test_units....`.

## Tests

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

Tests gate packaging — `build-deploy-s3-dist.sh` refuses to package on a
failing suite.

## The admin surface

Shipped as designed, with two truths sharper than the original note:

- **The visibility boundary is per ARTIFACT, not just per category**:
  serve paths check the loaded artifact's own audience (save_prompt
  materializes an `internal` bool; `_artifact_internal` is the ONE
  rule with an aux-header fallback for older artifacts), so a
  default-less category cannot smuggle internal versions past the
  standard surface.
- **"Tweak without a code deploy" is bounded by default-binding**: the
  workflows bind the immutable `default` version, so an admin
  put_prompt parks a version the workflows never read — the LIVE edit
  path remains editing the repo file and reseeding (overwrite-builtins
  rides any prompt-registry redeploy; that is a registry redeploy, not
  a consumer code deploy). Registry-side stage/version resolution for
  workflow binds is the open follow-up if that loop ever feels heavy.

Original design note:

Internal prompt categories (`ad-*`, `fi-*`, and cast-resolution's three
enrichment prompts, which migrated INTO the registry under this
design) must stay registry-managed for superuser tweakability but be
INVISIBLE to agents. The settled design, mirroring hybrid-search-admin:

- The prompt header's `Audience:` field is machine-readable;
  `workflow-internal` marks a category internal. conversation-analysis
  and shot-metadata-enrichment stay visible (agent-tunable by design) —
  visible is the default, only explicit headers hide.
- ONE codebase, TWO Lambdas from the same zip: the standard
  gateway-tool function runs `SERVE_INTERNAL=false` — `list_prompts`
  omits internal categories, `get_prompt` on one refuses by name,
  `put/promote/delete` into one are refused (no agent clutter). A
  second function runs `SERVE_INTERNAL=true` (full surface).
- media-toolkit-gateway registers both: `prompt-registry` (every agent) and
  `prompt-registry-admin` (optional parameter, attach only for
  superusers — the hybrid-search-admin posture, same warning text).
- Workflow children (AD, fi, conversation-analysis, sme, and
  cast-resolution) get the ADMIN function's ARN
  as their `PROMPT_REGISTRY_ARN` stack parameter: internal callers are
  trusted by definition, zero consumer code change.
- cast-resolution migration: `adjudicate_system.md` /
  `critic2_system.md` / `vocatives_system.md` moved from lib/ into
  registry categories marked internal (superseding an earlier in-repo
  decision — its rationale was "not independently editable";
  the admin surface now provides exactly the gated editability that
  rationale lacked). Its states fetch at validate like AD does.
- Tests: filtering both directions, refusal wording, audience parsing,
  and the two-target coverage pins in the bundle harness.

## Design record

Decisions cited from code comments (`README.md Design record …`). Section
and decision numbering is preserved from the original inference-plan working
notes; entries not cited anywhere are not reproduced here.

- **§3 — The grammar is parsed EXACTLY ONCE, at save time, by the registry.**
  Templates are versioned, immutable S3 pairs; `put_prompt` parses the
  `{{a:b+c}}` grammar into pre-declared `variables[]`, and consumers only
  ever substitute those declared tokens. That split is what keeps the copied
  render module small enough to freeze and makes it impossible for a caller
  to disagree with the registry about what a token means.
- **§3.3 — The blanket "no delete" rule is replaced by the staged
  lifecycle.** dev/test versions are deletable (frees the name), prod is
  not; see "Lifecycle: stage marks" above for the full rules and the
  rejected alternatives.
- **§3.4 — Seeding semantics.** The seeder walks `prompts/` from the zip,
  goes through the SAME `save_prompt` path as `put_prompt`, writes each pair
  write-if-absent (S3 conditional put), lands builtins `stage: prod,
  origin: builtin`, verifies filename stem == header `Version:` before
  writing, and treats Delete as a no-op success (prompts outlive any one
  stack). The pre-GA `overwrite-builtins` mode is bounded to
  `origin: builtin` artifacts whose shipped md5 differs.
- **§6 — Freeze bias.** `prompt_render.py`'s output feeds `workflow_id` (the
  hash of rendered bytes), so every behavior change re-identifies every
  consumer's requests and must bump every consumer's `.version`. The module
  is designed to change never, not rarely — no grammar, no metadata, no
  defaulting; golden vectors pin its behavior.
- **§7 — Inference history is a caller-written pointer.** One ~300-byte
  entry per submit at
  `micro-workflow/histories/{scope}/{category}/{prompt_version}/{ts}-{workflow_id}.json`,
  written by the CALLER via the copied `workflow_history.py`; no
  outcome/usage duplication (the reader follows
  `output_s3_prefix/summary.json`); absorbed resubmits are skipped inside
  the module so no caller can forget the rule.
- **Decision 5 — Histories are the caller's responsibility; any index is
  built BEHIND the S3 source of truth.** The registry never learns about
  runs and inference never learns what a prompt version is; the history
  prefix is the only join, and because it is a pointer it cannot disagree
  with the truth it points at. The shared module exists so the one layout
  does not decay into nine private ones.
- **Decision 8 — No state machine.** The registry is pure synchronous CRUD
  served by a gateway target: no Step Functions, no container, no CodeBuild,
  no ECR, no DynamoDB — one flat zip, two Lambda handlers. `describe_tool`
  honestly returns no `state_machine_arn`.
