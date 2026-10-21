# micro-workflow-bundle

One build script, one deploy script, one CloudFormation stack that brings up
**every** micro-workflow in this repository as a nested stack, plus the
AgentCore Gateway that exposes them to agents as MCP tools.

This directory contains **no application code** — no Lambda source, no
container, no state machine. Its entire failure surface is "does the root
template agree with the children it nests, and does the build script compose
S3 keys that actually exist." Both are mechanically checkable offline, which
is what [test/test_units.py](test/test_units.py) is for.

Release template URLs and per-release notes live in the
[root README](../README.md). This file is the developer documentation: what
the bundle deploys and why it is shaped this way, how to build and deploy it,
the contract a new micro-workflow must implement to join
([§C0–§C12](#adding-a-new-micro-workflow--the-contract)), the prompt/inference
conventions the LLM-backed workflows share
([§PI](#prompt-and-inference-conventions)), the test suite, the gotchas, and
the numbered [design record](#design-record) that comments in the templates
and scripts cite (`§P3`, `§P10`, …).

---

## What it deploys

| Nested stack | Child directory | Depends on |
|---|---|---|
| `MediaProcessingStack` | [media-processing-on-aws/](../media-processing-on-aws/) | — |
| `ShotSegmentationStack` | [shot-segmentation-on-aws/](../shot-segmentation-on-aws/) | — |
| `HybridSearchStack` | [hybrid-search-on-aws/](../hybrid-search-on-aws/) | — |
| `TtsStack` | [tts-on-aws/](../tts-on-aws/) | — |
| `AudiostemSeparationStack` | [audiostem-separation-on-aws/](../audiostem-separation-on-aws/) | media-processing |
| `SpeechEnhancementStack` | [speech-enhancement-on-aws/](../speech-enhancement-on-aws/) | media-processing |
| `SpeechDiarizationStack` | [speech-diarization-on-aws/](../speech-diarization-on-aws/) | media-processing |
| `FaceRecognitionStack` | [face-recognition-on-aws/](../face-recognition-on-aws/) | media-processing |
| `PromptRegistryStack` | [prompt-registry-on-aws/](../prompt-registry-on-aws/) | — |
| `BedrockInferenceStack` | [bedrock-inference-on-aws/](../bedrock-inference-on-aws/) | — |
| `ConversationAnalysisStack` | [conversation-analysis-on-aws/](../conversation-analysis-on-aws/) | bedrock-inference + prompt-registry (gateway-Lambda ARN parameters) |
| `ShotMetadataEnrichmentStack` | [shot-metadata-enrichment-on-aws/](../shot-metadata-enrichment-on-aws/) | bedrock-inference + prompt-registry |
| `FrameInspectionStack` | [frame-inspection-on-aws/](../frame-inspection-on-aws/) | bedrock-inference + prompt-registry |
| `ProgramStructureAnalysisStack` | [program-structure-analysis-on-aws/](../program-structure-analysis-on-aws/) | — |
| `AudioDescriptionStack` | [audio-description-on-aws/](../audio-description-on-aws/) | bedrock-inference + prompt-registry + tts |
| `CastResolutionStack` | [cast-resolution-on-aws/](../cast-resolution-on-aws/) | frame-inspection + bedrock-inference + prompt-registry (the registry ARN is the ADMIN surface, required) |
| `ArtworkSelectionStack` | [artwork-selection-on-aws/](../artwork-selection-on-aws/) | bedrock-inference + prompt-registry |
| `M2cAgentToolsStack` | [media-toolkit-gateway/](../media-toolkit-gateway/) | every child above that is a gateway target (frame-inspection is deliberately NOT one — it exists for internal sibling callers such as cast-resolution) |

Every "depends on" is expressed as explicit sibling gateway-Lambda ARN
parameters, never `Fn::ImportValue` (the §C8 pattern), so CloudFormation
derives the ordering from the references.

**All of them, always.** There is no opt-out parameter, and adding one is not
a small change — see design record [§P1](#p1--all-children-always). Two
consequences fall out of it:

- `media-toolkit-gateway`'s optional-target `Conditions` are permanently true
  here, which is why every cross-reference in [cfnstack.yaml](cfnstack.yaml)
  is a plain `!GetAtt` with no `!If` wrapper. `TestGatewayHarnessCoverage`
  pins that set against the gateway harness's `OPTIONAL_TARGETS`, so the
  harness cannot treat a target it never verifies as merely "not deployed".
- Nested-stack ordering is derived **entirely** from those `!GetAtt`
  references. There is no `DependsOn` anywhere in the root template,
  deliberately.

---

## Build and deploy

```bash
cd micro-workflow-bundle
bash ./build-deploy-s3-dist.sh --bucket MY_BUCKET_BASENAME
bash ./create_update_stack.sh \
    --stack-name my-micro-workflows \
    --ingest-bucket MY_INGEST_BUCKET
```

- `--proxy-bucket` is **optional and best omitted** — the stack then creates
  and owns one named `{prefix}-{stack-id}-{account}-{region}-proxy`. Supply it
  only to reuse an existing bucket, including one **retained** from a deleted
  bundle (the only way to get those outputs back).
- `--prefix` sets the leading part of `ResourcePrefix` (default `umt`, max
  **14** chars — arithmetic, not taste; see §P10); the stack appends a 12-char
  stack-id slice itself.
- Two bundles in one account+region **must vary `--gateway-name` or
  `--environment`** — varying `--prefix` alone is not enough, because the
  gateway stack has no `ResourcePrefix` parameter (see
  [§P7](#p7--the-gateway-ignores-resourceprefix)).
- Expect **1–2 hours** for a first create. The container children each build
  an image in CodeBuild *inside* their nested stack (60- and 90-minute
  timeouts, running in parallel) and tts additionally seeds its model archives
  to S3. It is not hung.

### Versioning

Each child carries its own `.version`; the bundle's own `.version` covers only
the root template. The forwarding rule:

| Bundle invocation | Child gets | Effect |
|---|---|---|
| `--version` omitted (default) | the child's own `.version`, read once by the bundle and passed explicitly | An unchanged child keeps its S3 key → CFN sees no `Code`/`ImageTag` change → CodeBuild **skips** its image rebuild. Only what you touched rebuilds. |
| `--version X` | `--version X` | All children restage under `.../X/...`; every `ImageTag` changes, so **every image build re-runs**. One coherent snapshot, at that cost. |

The `.version` format is an enforced release grammar:
`{alpha|beta|preview|vNN}.YYYYMMDDx` (NN 1–2 digits, x a lowercase letter) or
plain `MAJOR.MINOR.PATCH` semver — pinned by
`test_every_version_matches_the_release_grammar`. Prefer the dated form: the
value is simultaneously an S3 key segment and the ECR `ImageTag`, so the one
question it must answer is "which build is this?" — a date answers it, and
semver implied a compatibility contract these stacks never made. Nothing
downstream parses the value, and `--version` on the command line overrides the
file.

**Bump `.version` before every deploy, no exceptions.** CloudFormation
detects a changed S3 **key**, never changed S3 **content**. Rewriting the same
key is a no-op update — a green deploy that shipped nothing. Per-child
versioning is also what makes incremental deploys cheap: an untouched child
keeps its key, so its CodeBuild step finds the existing image tag and skips a
~90-minute rebuild.

### Build script shape

One `for` loop over a declarative table (`WORKFLOWS` in
[build-deploy-s3-dist.sh](build-deploy-s3-dist.sh)) — the children's CLI is
uniform (`-b|--bucket -s|--solution -v|--version -r|--single-region -a|--acl`),
which is the precondition that makes a loop correct rather than merely
shorter. Table order is DEPLOY order (§P3). Per child, the loop verifies the
directory and script exist, reads `WORKFLOW_NAME`/`TEMPLATE_NAME` out of the
child's script (never derives them from the directory name), resolves the
version once, invokes the child, aborts the whole run on any non-zero exit,
and records the composed key. After staging, `verify_staged_children()` runs
`head-object` on every composed key before the root template is written — a
missing key must fail the build, not the deploy, because a wrong
`TemplateURL` otherwise surfaces only after real CodeBuild minutes.

Two guards worth knowing about:

- **Dirty-tree guard**: the script refuses to build from a dirty working tree
  unless `--allow-dirty` is given, and a failed `git status` is treated as
  dirty (fail closed) — `git status --porcelain` writes nothing to stdout both
  when the tree is clean and when the command fails. Building dirty is the
  normal dev loop; it just has to be *stated*. The resolved commit SHA is
  printed in the version table, suffixed `-dirty` when it applies.
- **Unsubstituted-token guard**: `%%[A-Z0-9_]*%%`, not `%%[A-Z_]*%%` — the
  narrow class cannot match a digit, so it silently misses
  `%%KEYPREFIX_M2C_AGENT_TOOLS%%` (the `2` in `M2C`). A surviving token
  deploys as a literal `TemplateURL` and fails the stack after the first
  child's long CodeBuild has already run.

### Deploy script

`create_update_stack.sh` derives the template URL from the mappings inside the
**built** template under `dist/global-s3-assets/`, so an un-built template
fails loudly on `%%BUCKET_NAME%%` instead of resolving to a wrong bucket.
Capabilities passed: `CAPABILITY_IAM CAPABILITY_NAMED_IAM` —
`CAPABILITY_NAMED_IAM` is defensive, not load-bearing (no child declares a
named IAM resource; `test_named_iam_is_defensive_not_load_bearing` pins the
fact), and `CAPABILITY_AUTO_EXPAND` is deliberately **not** granted (see
Gotchas). `ProxyBucket` is always passed, including as an explicit empty
string: on an *update*, an omitted parameter silently keeps the previous
value, so dropping `--proxy-bucket` from a stack created with an explicit
bucket would appear to do nothing.

---

## Adding a new micro-workflow — the contract

What a new `<name>-on-aws/` stack at the repository root must implement to be
a sibling of the existing workflow children. Written for the case that
actually happens: someone ports a working pipeline in from another repo and
wires a gateway onto it.

**Every rule below names the test that enforces it.** That is the point of
this section: prose drifts, so a rule here is only worth reading if you can
run the thing that checks it. Rules with no test are marked **[UNENFORCED]**,
which means: it will not fail your build, a reviewer has to catch it, and it
is a candidate for the next test someone writes.

All test paths are relative to the repository root. The bundle suite is the
one that spans stacks:

```bash
cd micro-workflow-bundle && python3 -m unittest discover -s test
```

Run it before you open a review. It imports and executes every child's
gateway, so it fails on your stack even if your own `test/` directory is
empty.

### §C0. The one-paragraph version

A micro-workflow is a Step Functions state machine fronted by one flat Lambda
zip that AgentCore Gateway invokes as an MCP target. A caller submits and gets
back a **handle**, never results. The handle's `workflow_id` is a `uuid5` of
the request, so an identical resubmit is absorbed instead of launching a
second run, and outputs land at a prefix derived from that same id. Results
come from `get_job_status` and nowhere else.

If you change nothing else after reading this section, get these three right:
`workflow_id` derivation (§C2), the handle-only submit (§C3), and the flat zip
(§C5). They are the ones that are expensive to retrofit, because callers and
deployed artifacts both depend on them.

### §C1. Layout

```
<name>-on-aws/
├── .version                  # release grammar, e.g. alpha.YYYYMMDD[a-z]
├── README.md                 # the single developer doc, incl. a "## Gotchas" section
├── cfnstack.yaml             # the whole stack, one file
├── build-deploy-s3-dist.sh   # builds + uploads; fails on unsubstituted tokens
├── create_update_stack.sh    # deploys; derives the template URL from the build
├── definitions.json          # ASL readability copy, or definitions/*.json
├── gateway_tool/             # the flat Lambda zip -> MCP target
│   ├── lambda_function.py
│   ├── workflow_identity.py  # COPIED VERBATIM, see §C2
│   └── bucket_validator.py   # if you validate the proxy bucket
├── states/<state_name>/lambda_function.py
│   └── ... one directory per SFN state
├── lib/                      # container-only helpers; NOT in the gateway zip
└── test/test_units.py
```

| Rule | Enforced by |
|---|---|
| `.version` exists, is non-empty, and matches the release grammar | `micro-workflow-bundle/test/test_units.py::test_every_child_has_a_version_file`, `::test_every_version_matches_the_release_grammar` |
| Template name follows the convention, workflow name parses | `::test_every_child_declares_parsable_workflow_and_template_names`, `::test_template_name_convention_holds_for_all_eight` |
| Build script accepts the flags the bundle passes | `::test_all_eight_children_accept_the_flags_the_bundle_passes` |
| `README.md` has a `## Gotchas` section | **[UNENFORCED]** — but reviewers read it, so an absent one costs you review depth |

**Bump `.version` on every deploy, without exception.** CloudFormation
detects a changed S3 *key*, never changed S3 *content*. Rebuilding at the same
version overwrites the same key, the stack update sees an unchanged `Code`
block, and your old code keeps running — a green deploy that shipped nothing.

### §C2. `workflow_id` — copy the file, do not reimplement it

`gateway_tool/workflow_identity.py` is **byte-identical across every child
that derives an id**. Copy it with `cp`. Do not "adapt" it, do not fix its
formatting.

```python
workflow_id = compute_workflow_id(
    payload,                        # the SFN input as it will be sent
    tool_version=TOOL_VERSION,      # from .version, via the CFN env var
    exclude=("output.s3uri",),      # paths derived FROM the id
    extra_material=(etag,),         # optional: content-addressing
)
```

| Rule | Enforced by |
|---|---|
| All copies byte-identical | `::TestWorkflowIdentityParity::test_all_six_copies_are_byte_identical` |
| The set of children deriving an id is declared, not inferred | `::test_the_children_that_derive_a_workflow_id_are_exactly_these` |
| No gateway hand-rolls the derivation | `::test_no_gateway_still_hand_rolls_the_derivation` |
| The module actually ships in the zip | `::test_the_module_ships_in_the_gateway_zip` |
| The hash **preimage** is pinned by known-answer vectors | `::TestWorkflowIdGoldenVectors::test_the_preimage_is_pinned` |

That last one is the important one and the one a new stack forgets.
Determinism and key-order independence hold just as well for a *wrong*
preimage; only a fixed input mapped to a fixed output string catches "the
preimage changed", which is the failure mode here that is both silent and
expensive — the `workflow_id` is the execution name *and* the output prefix,
so a changed preimage re-runs every request at a fresh prefix and **nothing
errors**. If a golden vector fails: an unintended change means fix the code,
not the vector; an intended change means update the vector **and bump every
affected `.version`**, because old and new code now disagree about the
identity of the same request. Per-gateway variation must live in the CALL
(`exclude=`, `extra_material=`), never in the module — that is what keeps
parity a file comparison instead of a diff review.

**Exclude anything derived from the id.** Your output prefix is
`micro-workflow/{target}/{workflow_id}/`, so it cannot be in the preimage —
that is circular. Pinned by
`::test_the_derived_output_prefix_cannot_influence_the_id`.

#### The three ids, which are not the same id

| Layer | Key | Meaning |
|---|---|---|
| Gateway **input** | `idempotency_token` | Caller's assertion "same request as before". Optional, 1–80 chars of `A-Za-z0-9_.-`. Folded into the hash, then **discarded** |
| Gateway **output** | `workflow_id` | The service's name for the run; the caller's handle |
| SFN input / state `event` | `uuid` | *This execution's* local id. Keep this name — states derive prefixes and reconstruct ARNs from it |

`uuid` is **not** always the `workflow_id`: face-recognition's
`prepare_media_processing` mints `facerec-{urandom(6).hex()}` for a child it
invokes directly. That is correct, because a directly-invoked child is not
named by a hash.

The token **names nothing**. It changes the request's *identity*, which is
why it is not a retry button: a plain retry with the same token is
indistinguishable from the original call and gets the original verdict. A
*fresh* token is also the rerun mechanism — "same prompt, same model, same
data, run it again" is a new token, never a `force_rerun` flag, because the
token-based rerun is itself idempotent under retry where a boolean flag can
never be.

| Rule | Enforced by |
|---|---|
| The retired `uuid` **parameter** is rejected on submits, not ignored | `::TestGatewayTokenContract::test_every_gateway_rejects_the_retired_uuid_parameter` |
| A poll accepts a stray token/`uuid` instead of failing the caller | `::test_a_poll_accepts_a_token_instead_of_failing_the_caller`, `::test_a_poll_accepts_the_retired_uuid_too` |
| The token never reaches the state machine | `::test_no_gateway_forwards_the_token_to_the_state_machine` |
| The token is never echoed in any response | `::test_no_gateway_echoes_the_token` |
| A malformed token is rejected everywhere | `::test_a_bad_token_is_rejected_everywhere` |
| Same token + different parameters still re-runs | `::test_the_same_token_with_different_parameters_still_re_runs` |
| An ignored token is logged, not silent | `::test_an_ignored_token_is_logged_not_silent` |

**Rejecting on a submit but warning on a poll is deliberate, not
inconsistency.** Your caller is an LLM, and a hallucinated parameter is a
routine failure mode. On a submit, silently ignoring `uuid` would start a
fresh run for a caller who believed they were protected from paying twice —
so it must fail loudly. On a poll the field could not have changed the
answer, so failing the request costs a working call and buys nothing.

**Decide whether an absent token means "dedup on parameters" or "do not dedup
me."** If your parameters fully name the output (they name an asset), an
absent token still dedups on the parameter hash — expensive work should
require an explicit act to re-pay for. If they do not — face-recognition
reads a mutable face collection, tts renders against a mutable voice catalog
— then identical parameters can legitimately need to run again, and an absent
token must *generate* one. Declare which you are in the test's
`_GENERATES_A_TOKEN`. Pinned by
`::test_the_token_changes_the_id_only_where_that_is_the_design`.

### §C3. A submit returns a handle, never results

This is the rule most likely to be violated by a port, because "return what
the caller asked for" feels helpful.

```
submit, fresh               job_id, workflow_id, status=RUNNING,
                            + handle fields (current_state, asset_s3uri,
                              output_s3_prefix)
submit, absorbed RUNNING    the same, + message
submit, absorbed SUCCEEDED  the same, + message               <- NO results
submit, absorbed terminal   the same, + message + error       <- NO results
get_job_status              the ONLY source of results
```

Enforced by `::TestSubmitReturnsAHandleNotResults`, which **imports and calls
your gateway** — its own battery plus everything it inherits from
`TestGatewayTokenContract`, each subTested across every child that starts an
execution.

| Rule | Enforced by |
|---|---|
| A fresh submit returns a handle, `status=RUNNING`, zero result fields | `::test_a_fresh_submit_returns_a_handle` |
| An absorbed resubmit differs from a fresh start **only** in `status` | `::test_an_absorbed_resubmit_differs_from_a_fresh_start_only_in_status` |
| An absorbed resubmit never returns the existing run's results | `::test_an_absorbed_resubmit_never_returns_the_existing_runs_results` |
| An absorbed resubmit says it did not resubmit (`message`) | `::test_an_absorbed_resubmit_says_it_did_not_resubmit` |
| A terminal absorb carries its `error` cause; a success carries none | `::test_an_absorbed_terminal_failure_carries_its_cause`, `::test_an_absorbed_success_carries_no_error` |
| `current_state` present while RUNNING, dropped once stopped | `::test_current_state_is_reported_while_running_and_dropped_once_stopped` |
| Every poll branch names the run | `::test_every_poll_branch_names_the_run` |

**Why, in one paragraph.** Before this rule, an absorbed resubmit handed back
the finished run's entire payload — many times the handful of fields declared
in `cfnstack.yaml`, dozens of undeclared fields across the submit tools.
Three defects in one: the reachable field set could not be *declared* (it was
the union of the handle shape and every result shape), the agent needed two
readers for one response with no way to know which it would get, and a submit
could dump an unbounded payload into an agent that asked for a handle. Making
the absorbed response a handle whose only variable is `status` fixes all
three by construction: reachable == declared.

**`message` is the only signal a submit was absorbed, and comparing `job_id`s
cannot replace it.** The execution name is a hash of the same input either
way, so a fresh start of a run that has since aged out returns a
*byte-identical* ARN.

**`current_state` is the one deliberate asymmetry.** Present on a RUNNING
absorb, because "which stage is it in" is the caller's real question and
worth a `GetExecutionHistory` paging pass. Absent once stopped, because a
stopped execution is in no state and `"unknown"` would buy a non-answer with
a paging pass.

**A terminal failure is sticky for ~90 days.** Step Functions reserves an
execution name for 90 days after the run stops, so an identical resubmit
keeps returning that FAILED verdict. Say 90 days in your schemas, not
"forever" — an agent told a state is permanent has no reason to look again.
Only a changed parameter, a different token, or a new tool version re-runs it.

**Never invent a status word.** Use Step Functions' vocabulary. One child
once answered `ALREADY_RUNNING` (a claim about the gateway's own bookkeeping,
not the job — a run that succeeded hours ago and one that FAILED both got it)
and `STARTED` (a seventh word for RUNNING). Both are gone.

### §C4. Output contract

On `SUCCEEDED`, `get_job_status` returns `outputs[]` of `{uri, mime, label,
desc}` — plain `s3://` URIs; an entry whose uri ends in `/` is a prefix and
carries no mime; `desc` is a one-line role description joined from the
stack's `gateway_tool/output_contract.json` (omitted, never fabricated, when
the contract lacks the label). **`label` is a role vocabulary**: one entry per
distinct KIND of file (`transcript`, `caption`, `speaker-manifest`, `usage`).
That is what makes "select by label, never parse filenames" meaningful.
Labels and output basenames are stable API — and so are the field-level
contents of each labeled file, recorded per label in `output_contract.json`
(basename, kind, mime, one-line description, compact example) and served in
full by `describe_tool` as `output_contract`. The file also defines the
contract BOUNDARY: an output not listed in it is internal and may change
without notice — a real consumer once integrated against a child's
`ephemeral/` intermediates and broke silently; the contract file is what
makes that distinguishable. Enforced by `::TestOutputContract` — contract
keys == the per-child set the suite declares (hand-maintained in
`_CONTRACTS`, exact in both directions), envelope invariants (single-line
descriptions, valid kinds — a multi-line description silently bloats every
status response, because the gateway injects it verbatim into every
`outputs[]` entry), a runtime import proving each gateway serves its on-disk
file and joins `desc` from it, and zip membership. Two checks are
deliberately NOT in the suite: parity against the label set the gateway's own
code can emit, and basename greppability in the producing state — the live
label-set check per target is the gateway harness's job
(`media-toolkit-gateway/scripts/test_gateway.py`), and a new label reaches
consumers desc-less rather than failing a build until those land.

**For N files of ONE role, emit a prefix entry plus a manifest — do not
enumerate.** This is the house answer and two stacks reached it
independently: speech-diarization's per-speaker WAVs (speaker ids can be
non-contiguous) and tts's per-clip outputs. Conforming would force either a
non-discriminating `label: "clip"` repeated N times (the rule becomes
vacuous) or dropping the per-item fields a `{uri, mime, label}` triple has
nowhere to hold.

Always include, even when your shape differs: **`workflow_id`** (an agent
resuming with only a `job_id` cannot name the run), **`output_s3_prefix`**,
and **`usage_uri`** if you write `usage_tracking.json`. One child shipped
without all three — plain gaps, not design.

**Never inline a presigned URL.** ~1.3 KB each (~350 context tokens); the
`X-Amz-Security-Token` dominates. HTTP access comes from
`media-outputs___get_download_url[s]` only.

### §C5. Gateway zip and the mirroring trap

**The zip is flat (`zip -j`)** so `Handler: lambda_function.handler`
resolves. Consequence: **`lib/` is not in it**, and helpers are *copied* into
`gateway_tool/`. Those copies ARE the distribution mechanism — a parity test
is mandatory, not optional, because nothing else keeps them in sync.

| Rule | Enforced by |
|---|---|
| Every zip member is checked with `[ -f ]` before `zip` | `::TestBuildScriptSafety::test_checks_each_child_path_explicitly` |
| Copied helpers stay byte-identical | `::TestWorkflowIdentityParity::test_all_six_copies_are_byte_identical`, `::TestBucketValidatorParity::test_all_six_copies_are_byte_identical` |
| Build fails on any unsubstituted `%%TOKEN%%` | `::test_build_script_substitutes_every_token_in_the_template` |
| Composed S3 keys are verified to exist before writing the root | `::test_verifies_every_composed_key_exists_before_writing_the_root` |

**`zip` exits 0 when only *some* members are missing.** It warns, packages
the rest, and returns success — it fails only when nothing matched. Trusting
the exit code ships a Lambda whose handler module simply is not in the
archive; it builds clean, deploys clean, and fails at **import on the first
invocation**. This really happened, more than once. Check each member up
front:

```bash
local members=(lambda_function.py bucket_validator.py workflow_identity.py)
local f
for f in "${members[@]}"; do
  [ ! -f "$f" ] && echo "error: gateway_tool/$f not found" && exit 1
done
zip -jq "$BUILD_DIST_DIR/$PKG_GATEWAY_TOOL" "${members[@]}" || exit 1
```

### §C6. MCP target and tool schemas

- Tool names over MCP are `{target}___{tool}` and **must be ≤ 64 chars** —
  Bedrock rejects the entire ConverseStream on an oversize name. With the
  longest verb in use (`separate_audio_stems`, 20) + `___`, target names must
  be ≤ 41 chars. Enforced at build time by `validate_tool_name_lengths` in
  `media-toolkit-gateway/build-deploy-s3-dist.sh`.
- `TARGET_NAME` on your gateway Lambda **must equal** the GatewayTarget
  `Name` in `media-toolkit-gateway/cfnstack.yaml`. **[UNENFORCED]** — a
  mismatch sends your outputs to a prefix nobody reads.
- Schemas are native YAML `InlinePayload` lists, PascalCase keys.
  `SchemaDefinition` supports only `Type`, `Description`, `Properties`,
  `Required`, `Items` — **no `enum`, `minimum`, `maximum`**. Bounds go in the
  `Description` prose and are enforced in code.
- Every tool needs a `describe_tool` returning `{version, state_machine_arn,
  description, output_contract}` — introspection only, deliberately without
  the ASL (token cost, internals leakage). prompt-registry, which has no
  state machine at all, honestly omits the arn. `output_contract` is the
  parsed `gateway_tool/output_contract.json` from the same zip, so it is
  static per `TOOL_VERSION` and a consumer's CI can pin it without running a
  job. Targets without labeled file outputs (tts, hybrid-search) carry
  `kind: "inline"` contracts documenting their inline result shapes instead.
- Declared `OutputSchema` should equal the reachable field set. §C3 is what
  makes that checkable rather than aspirational.

### §C7. Hostile input

Your caller is an LLM emitting arbitrary JSON. For every field assume the
wrong type, and note the two that bite hardest:

- **A bare string where a list is expected iterates per-character** — a
  silent wrong answer, not a crash.
- **Falsy-zero**: `if not x` rejects a legitimate `0`.
  `max_faces_to_index: 0` is a real, meaningful value.

Also: negative/NaN/huge numbers, quote injection into filter strings, path
traversal in S3 keys, and unbounded sizes (`head_object` before you
download). Errors returned to an agent must be sanitized (no stack traces, no
ARNs) and actionable **at submit time**, not asynchronously. **[UNENFORCED
across stacks]** — per-stack suites cover some of this; there is no shared
harness.

### §C8. CloudFormation

| Rule | Enforced by |
|---|---|
| Lambdas are unnamed and reach logs via `LoggingConfig` | `::TestChildLogGroupHygiene::test_start_build_lambdas_are_unnamed_and_use_logging_config` |
| Log groups declared with retention; functions `DependsOn` them | `::test_start_build_log_groups_are_declared_with_retention`, `::test_start_build_lambdas_depend_on_their_log_group` |
| No role may create its own log group | `::test_no_child_role_may_create_its_own_log_group` |
| No declared-group role carries `AWSLambdaBasicExecutionRole` | `::test_no_declared_group_lambda_role_carries_the_basic_execution_policy` |
| Function names fit 64 chars at the prefix ceiling | `::test_child_function_names_fit_64_chars_at_the_prefix_ceiling` |
| Proxy bucket passed conditionally, never a bare `!Ref` | `::TestProxyBucketOwnership::test_every_child_gets_the_conditional_bucket_not_a_bare_ref` |
| Memory tiers use the house vocabulary and memory | `::TestMemoryTierVocabulary::test_every_declared_tier_uses_the_house_memory_and_name` |
| No child declares a `Transform` | `::TestDeployScript::test_no_child_declares_a_transform` |
| No output is exported | `::TestOutputs::test_no_output_is_exported` |

**Never name a Lambda you also declare a log group for.** Declaring the
canonical `/aws/lambda/{FunctionName}` fails `create-stack` at pre-flight —
`Validation failed with N error(s)`, **no per-resource event**, nearly
undebuggable — if a previous standalone deploy left that group behind
unowned. CFN cannot adopt an existing group, and `Conditions` cannot branch
on a runtime lookup. `LogGroupName` is `createOnlyProperty`, so renaming
*deletes* the old group's contents. Leave functions unnamed, give the group a
`-log` suffix, wire it through `LoggingConfig.LogGroup`.

**`AWSLambdaBasicExecutionRole` grants `logs:CreateLogGroup` on every
group**, which lets a function that cannot reach its declared group silently
create an unmanaged one instead of failing visibly. That is exactly how
orphaned groups come to exist. Use an inline
`logs:CreateLogStream`/`PutLogEvents` scoped to `!GetAtt <Group>.Arn`.

**Export nothing.** Cross-stack values pass as explicit parameters. An
`Fn::ImportValue` hard-locks the producer: the value cannot change and the
stack cannot be deleted while any importer exists. (This is "the §C8
pattern" the root template's comments cite: sibling dependencies arrive as
explicit gateway-Lambda ARN parameters.)

Memory tier vocabulary — a tier named `3-cores` at anything but 5376 MB makes
a cross-workflow comparison silently wrong:

| Tier | MB | Name |
|---|---|---|
| `HalfCore` | 896 | `half-core` |
| `OneCore` | 1792 | `1-core` |
| `TwoCores` | 3584 | `2-cores` |
| `ThreeCores` | 5376 | `3-cores` |
| `FourCores` | 7168 | `4-cores` |
| `FiveCores` | 8960 | `5-cores` |
| `SixCores` | 10240 | `6-cores` |

Declaring a tier no state routes to is normal — the tiers are an interface,
not per-child taste, and unrouted tiers preserve the ability to re-point a
state during testing without a template change. Size per-role instead if
cores are the wrong axis (hybrid-search does), but then declare that you are
not tiered — `::test_the_tiered_children_are_exactly_these` fails a child
that *silently loses* its mapping, which is the failure a simple
`if absent: skip` would hide.

### §C9. Custom resources must always respond

A custom resource that raises before sending its response makes
CloudFormation wait a **full hour** before giving up, orphaning the Lambda
and its role — whose surviving `FunctionName` then blocks the next
`create-stack` with an `EarlyValidation ResourceExistenceCheck` failure.

**No statement that can raise may sit outside the `try`.** The real bug:
`ResourceProperties["AccountId"]` read outside it, and CFN's **Delete event
does not carry it**. It happened in production, twice.

| Rule | Enforced by |
|---|---|
| No property read precedes the `try` | `::TestBucketValidatorParity::test_no_property_read_precedes_the_try_block` |
| The failure path can always resolve a physical id | `::test_the_failure_path_can_always_resolve_a_physical_id` |
| Handler is named `handler` everywhere | `::test_the_handler_is_named_handler_everywhere` |
| One custom-resource `Type` across all children | `::TestCustomResourceTypeUniformity` |

A Delete must SUCCEED with no properties at all (a resource whose create
never completed). A **Create** missing a required property must FAIL loudly
with a named reason — the fix must not paper over a genuinely broken
template. Both directions are pinned in
`tts-on-aws/test/test_units.py::TestBucketValidatorAlwaysResponds`.

### §C10. `usage_tracking`, if you add one

If your last state prices the run, add your stack to
`USAGE_TRACKING_STACKS` in the bundle suite. It is **deliberately not** the
`CHILDREN` map — the set that matters is "has the code", not "is nested by
the bundle".

Enforced by `::TestUsageTrackingBillsFailedAttempts` and
`::test_the_usage_tracking_stacks_are_exactly_these`, which fails a stack
that gains the code without joining the list — that is how the original bug
reached every copy.

**Bill failed attempts.** AWS bills a Lambda that ran to its timeout and was
retried; it is the most expensive invocation a workflow can have. Reading
only `succeeded_ts` charges it 0 s and under-reports by multiples — worst
exactly when the workflow behaved worst. Any state using checkpoint/resume
makes a timeout a *normal* path. (The inverse defect also existed: pricing
the whole state span counts concurrent Map iterations as sequential time and
over-reports.)

**Key on the state occurrence, not the attempt.** An ASL `Retry` does
**not** re-enter the state, so every attempt of one occurrence chains back to
the same `TaskStateEntered` — group on its event id. Retry attempts are
*sequential* and must be summed before the max; Map iterations are
*concurrent* and must not be. Getting this backwards reports a state retried
3x as its longest single attempt.

**Zero is right in exactly one place**: an attempt with no terminal event at
all (this state's own in-flight record). Nowhere else.

### §C11. Deliberate exceptions

Cite one of these only if your reasoning matches. They are exceptions with
arguments, not precedent for skipping the contract.

**Idempotency at the document layer instead of the execution layer**
(hybrid-search `index_content`): an accumulating write to a shared
collection. Two calls with different documents must both run, so collapsing
on an input hash would silently drop the second. It names its execution
`uuid4()` on purpose and dedups one layer down, content-addressed:
`doc_id = md5(collection:asset:doc_type:start:end:chunk)`, filtered
**before** any Bedrock spend — verified live: a full resubmit skips every
document without spending a token. If you claim this exception you must have
a content-addressed dedup key that survives a re-run without the caller
remembering anything.

**Dedup-before-spend is locked.** Expensive calls (Bedrock, Rekognition)
come *after* all validation and dedup. One bad document must not fail a batch
after money was spent on the others.

**No `workflow_identity.py` at all** (hybrid-search): it derives no
`workflow_id`, so the rest of the module would be dead code in the zip. It
carries only `warn_idempotency_token()` in `common.py`.

**"Does not honour a token" is not "starts no execution."** Three groups,
and only the first starts nothing: status polls and sync reads; accumulating
writes named `uuid4()` (`index_content`, `batch_index_faces`,
`batch_delete_faces`, `delete_collection`) which DO start executions; and
`register_voice`, which starts one AND dedups on its parameter hash — it
refuses a token because there is no dedup at the record layer beneath it, so
a token could only ever force a duplicate, permanently unremovable catalog
entry. Your warning message must talk about the *token* ("nothing here reads
it"), never about what the tool does. Pinned by
`::test_the_warning_makes_no_claim_about_starting_an_execution` and
`::test_a_tool_that_ignores_a_token_may_still_start_an_execution`.

### §C12. Pre-review checklist

```bash
# 1. the cross-stack suite — fails on YOUR stack even with an empty test/
cd micro-workflow-bundle && python3 -m unittest discover -s test

# 2. your own suite
cd <name>-on-aws && python3 -m unittest discover -s test

# 3. syntax, everything you touched
python3 -m py_compile $(git diff --name-only HEAD | grep '\.py$')
bash -n build-deploy-s3-dist.sh create_update_stack.sh

# 4. the template parses, and every condition is wired
#    (intrinsic-tolerant parse; validate-template needs credentials)
```

Then, by hand, the four things no test covers:

1. **Every `!If` names a declared condition, and every conditional resource
   is referenced only from equally-conditional places.** An unguarded
   reference fails at deploy time *only for customers who omit that
   parameter*.
2. **`TARGET_NAME` == the GatewayTarget `Name`.**
3. **`.version` bumped.** Every deploy. No exceptions.
4. **The child's `README.md` still true after your diff** — a stale doc your
   commit should have updated is a review finding.

When this section and the suite disagree, **the suite is right** — it runs.
Fix this section.

---

## Prompt and inference conventions

The LLM-backed workflows (conversation-analysis, shot-metadata-enrichment,
frame-inspection, audio-description, cast-resolution, artwork-selection)
share two services — `prompt-registry-on-aws` (templates) and
`bedrock-inference-on-aws` (execution) — plus three copied modules. The
organizing principle, in two rules:

- **Split by who changes it and how often**, not by code layer: prompts
  change often (prompt engineers), the model list changes when a model ships,
  workflow bindings change per deployment, inference mechanics change rarely.
  Each caller workflow owns its own binding (prompt ref + version, model
  chain, temperature, batch preference) inline in its own directory; there is
  no shared workflow-config subsystem.
- **Resolve everything before submit, never inside the run.** The dataflow
  is resolve → render → submit, and the rendered bytes on S3 are the
  complete, immutable audit record of what ran. bedrock-inference's
  `workflow_id` (hash of the rendered bytes) enforces it: an edited prompt
  renders different bytes and re-runs naturally; two routes to identical
  bytes dedupe naturally. Prompt *references* are deliberately not in the
  hash — the prompt version travels in the history entry (§PI7).

Each service is ignorant of the others' internals: the registry never
fetches metadata or learns about runs; inference never learns what a
transcript or a prompt version is.

### §PI3. prompt-registry

Pure CRUD on S3, gateway-target-only, **no state machine** — the §C2/§C3
rules about `workflow_id` and handle-only submits do not apply to a tool that
starts no executions; the token-warning rule (§C11) does.

- Storage: `s3://{proxy}/_promptregistry/{category}/{version}/` holding
  `source.md` (the authored template, byte-preserved) and `artifact.json`
  (parsed at save time; what `get_prompt` returns).
- `{category}` is the prompt's name; prompt categories are kebab-case;
  `{version}` is a snake_case name, not a number. **Versions are immutable
  once written**: an edit is a new version — that is what makes a history
  entry's version reference permanently meaningful (the same immutability
  argument as §C3's sticky terminal verdicts).
- Template variables (`{{a:b+c}}` grammar) are parsed ONCE, at save time, by
  the registry and declared in the artifact. Consumers never parse the
  grammar; they substitute pre-declared tokens.
- Lifecycle is staged: versions carry `stage: dev|test|prod` (default dev).
  dev/test versions MAY be deleted, freeing the name for iteration; prod
  versions can only be deprecated; promotion is a one-way ratchet. Safe
  because run identity is the rendered bytes and every run's prefix keeps its
  request record — a deleted dev version dangles only the human-facing
  version pointer. See `prompt-registry-on-aws/README.md` "Lifecycle".
- Builtin prompts are seeded by a custom resource from files packaged in the
  zip; the custom resource follows §C8/§C9 to the letter.
- Genre and content-class knowledge lives in prompts, never in workflow code.

### §PI5. Model registry: facts as data, rules as code

bedrock-inference seeds its own `availablemodels.json` from one packaged
master list, geo-filtered at install time, with a version-skip so an unbumped
stack update cannot clobber an operator-edited registry.

- **Per-model FACTS live in the registry** as a `capabilities` block
  (temperature support, thinking required/optional, effort, image limits);
  **cross-field RULES live in code, in exactly one shaping function**
  (`build_converse_params`). The failure mode this prevents: rules scattered
  across a class hierarchy until they contradict.
- `model_id` is BARE (`anthropic.claude-…`, never `global.anthropic.…`): the
  prefix names *routing*, not identity. `inference_region` is the routing
  scope (`global` | a geo | a concrete region), request-overrideable for
  residency, and the override is IN the `workflow_id` hash — where a run is
  allowed to execute is part of what the run *is*. An override must be
  literally in the model's `supported_regions`, checked against every model
  in the chain, at plan time.
- Image `limits` (service-enforced; exceeding one is a 400 or a silent
  server-side downscale) are kept distinguishable from `recommendations`
  (quality-tuning guidance; exceeding one degrades results, nothing errors).
  Limits are enforced at `validate_plan`, before any token is spent;
  recommendations are data for the caller's grid-preparation step.
- Batch-vs-sync is one optional boolean, `use_batch`, resolved against
  capabilities. An unsatisfiable `use_batch: true` downgrades to sync **with
  the reason reported** (`mode_reason`), never a hard error. The absent-flag
  default is denominated in DOLLARS, not record count: batch runs iff the
  estimated input-cost saving clears the `BatchMinSavings` stack parameter
  (default $0.25) — count is a proxy for the wrong quantity (a few huge
  records deserve batch; a hundred tiny ones deserve sync-in-seconds). Every
  auto decision lands in `mode_reason`, auditable after the fact.
- Batch jobs declare `modelInvocationType: "Converse"`, so sync and batch
  share one request AND response shape; a batch job that fails records gets a
  bounded, once-only **sync repair tail** under a lease — missing / errored /
  refused records re-run through the normal sync chain walk (which restores
  chain-fallback semantics to batch); truncated records are NOT repaired
  (re-running truncates again; salvage is the caller's job, §PI6.2). Repair
  runs only when `0 < failed ≤ repair_max_fraction × total` — all-failed is
  systemic and would re-pay the whole job to fail again.
- The runtime recovery in the converse path (strip a rejected `temperature`
  once, record a note) stays: capability flags save a wasted call; recovery
  is the correctness backstop for a stale registry.

### §PI6. The copied modules

Cross-repo sharing is **copy-the-file + byte-parity + golden vectors** — the
`workflow_identity.py` mechanism (§C2, §C5). There is no shared package
across the workflow directories, and prose drifts; only the bundle suite
holds copies together. **Freeze bias:** every behavior change to a copied
module bumps every consumer's `.version` (old and new code disagree about the
identity of the same request). Design them so small and stable they change
never, not rarely.

- **§PI6.1 `prompt_render.py`** (source of truth: prompt-registry-on-aws).
  `render(artifact, values)` substitutes pre-declared tokens into the
  template. Deliberately dumb: no grammar parsing (the registry did that at
  save time), no metadata fetching, no defaulting; a missing required
  variable raises. Golden vectors matter more than parity here: its output
  feeds the `workflow_id` hash, so two repos disagreeing about rendering is
  two repos disagreeing about identity — a silent re-run at a fresh prefix.
  Pinned by `TestPromptRenderParity`.
- **§PI6.2 `parse_partial_output.py`** (source of truth:
  bedrock-inference-on-aws). The model-output salvage ladder:
  `parse_partial_json(text)` / `parse_partial_xml(text)`, each returning
  `(value | None, was_repaired)` — recover the largest well-formed prefix of
  a `max_tokens` truncation or a prose-wrapped answer, where `was_repaired`
  must survive into the record so a repaired answer is never mistaken for a
  clean one. The invariant: the repair only ever **completes, trims, or
  escapes — never rewrites**. External repair packages were measured and
  rejected: they silently trim a string cut mid-word (`"indeterm"` for
  `"indeterminate"`) — fabrication — where this module drops the incomplete
  tail value whole. The JSONL helpers (`escape_inner_quotes`,
  `parse_jsonl_line(line, required_keys)`) keep salvage schema-agnostic
  through the **mandatory** `required_keys` gate: the caller states the
  minimum its code cannot join without, so a recovered partial that lost
  optional metadata is kept, not discarded, and an ungated rewrite is
  unreachable by omission. The module is stdlib-only BY DECISION — a copy
  that grows an import has been edited, not copied. Pinned by
  `TestParsePartialOutputParity` (exact consumer membership + byte-identity
  against the master).
- **§PI6.3 `workflow_history.py`** (source of truth:
  prompt-registry-on-aws). Writes the §PI7 history entry, so "caller
  responsibility" does not decay into N layouts. Pinned by
  `TestWorkflowHistoryParity`.

Each parity class carries an exact-membership test of which children hold the
copy (the `_VALIDATOR_CHILDREN` pattern) — a child that gains a copy without
joining the list must fail.

### §PI7. Inference histories: caller-written, house-standardized

```
s3://{proxy}/micro-workflow/histories/{scope}/{category}/{prompt_version}/{submitted_at_ms}-{workflow_id}.json
```

- `{scope}` is a required, caller-chosen grouping key
  (`[A-Za-z0-9_.-]{1,64}`, the execution-name charset): an asset id where one
  exists, a project/collection name for ad-hoc agent work. It is the
  standardized answer to "which runs concern this thing" — what lets a
  *stateless agent*, with no memory of last week's `workflow_id`s,
  rediscover prior runs by listing one prefix.
- The record is a **pointer, not a copy** (~300 bytes): `{workflow_id,
  job_id, output_s3_prefix, scope, prompt: {category, version}, model_chain,
  idempotency_token, submitted_at}`. Outcome and usage are deliberately NOT
  duplicated in — the reader follows `output_s3_prefix/summary.json`, which
  is authoritative and exists exactly once. Duplicating summaries is how a
  record and the truth come to disagree.
- **Written once, at submit, by the caller** via `workflow_history.py` —
  every field is known the moment `run_inference` returns; no completion
  hook, no second writer. **Skip the write on an absorbed resubmit** (the
  submit response's `message` field is the signal, §C3) — otherwise a retried
  submit manufactures a phantom second run pointing at the same execution.
- Rerun/comparison composes: same prompt+data with a different model or a
  fresh token lands a sibling entry under the same prefix; listing that
  prefix IS the comparison view. S3-list answers only key-path queries, and
  that limit is accepted; a queryable index, if ever needed, is the caller's,
  built behind this convention.

---

## Tests

```bash
cd micro-workflow-bundle
python3 -m unittest discover -s test        # the whole suite, offline
```

Traps when editing the suite:

- **`TestUsageTrackingBillsFailedAttempts` IMPORTS its sibling stacks' Lambda
  code and must leave `sys.modules` exactly as it found it.** It is the one
  class here that executes another directory's module rather than parsing a
  file, because the invariant is arithmetic (a `TaskFailed` attempt is
  billed) and each copy spells the history walk differently — a text or AST
  assertion would pin the spelling instead. Three things this forces. (1)
  Every one of those modules builds boto3 clients **at import time**, so
  `_load_usage_module` installs `boto3`/`lib`/`lib.s3_utils` stubs *before*
  `exec_module` and restores the previous entries in a `finally` — a leaked
  `boto3` stub would silently disarm every other test in the repo that
  imports one, passing loudly. (2) The stacks expose **two different
  shapes** — most have `_parse_execution(events, cache) -> list` keyed by
  `state_name` with `total_billed_sec`; face-recognition has
  `_execution_cost(arn) -> (dict, float)` keyed by state with `billed_sec`,
  and it fetches history itself so `_sfn` must be monkey-patched
  (bedrock-inference shares that shape). One `summarize()` helper normalises
  them; do not split the class in two, or the next stack to diverge gets its
  own copy of the assertions. (3) `wall_sec` is asserted as `>=`, not `==`,
  because the sum-per-`entered_id` stacks and the state-span stack cover the
  same two attempts with different arithmetic; the companion test pins the
  concurrent case exactly, which is what stops "just sum everything" from
  passing. **`USAGE_TRACKING_STACKS` must be exactly what is checked out in
  this tree** — `_load_usage_module` *executes* each entry's module, so an
  entry naming an absent directory is a `FileNotFoundError` in `setUpClass`,
  not a skip, and it takes the whole class down.
  `test_the_usage_tracking_stacks_are_exactly_these` scans the repo root and
  fails if the tuple goes stale in **either** direction — an unlisted copy
  would otherwise be silently exempt from every other assertion, which is
  exactly how the original billing bug reached every copy.
- **A "must be absent" assertion can fail in BOTH directions, and the second
  one is silent.** The documented trap is a false *failure*: a text search
  hits the comment that explains the omission. The inverse is a false
  *pass*: `test_the_module_ships_in_the_gateway_zip` once searched the build
  script's text for `workflow_identity.py`, and every build script names
  that file in the comment explaining its existence guard — so deleting the
  actual zip member left the test green. Found by mutation. It now parses the
  `members=()` array. Assert against the parsed structure whether you are
  proving presence or absence. `test_no_gateway_still_hand_rolls_the_derivation`
  walks the **AST** for the same reason — every gateway documents the
  derivation in prose, so a text search for `uuid5` fails on the docstring
  describing the very thing being asserted.
- **Assert against the parsed template, not the raw file text, for any "X
  must be absent" property.** The template and scripts *document* the things
  they deliberately omit (`DependsOn`, `bSingleRegion`,
  `CAPABILITY_AUTO_EXPAND`), so a text search fails on its own explanatory
  comment. `_capabilities()` parses the `CAPABILITIES=` assignment alone for
  the same reason.
- **A child test suite must put `gateway_tool/` on `sys.path`, not just the
  child root.** The gateway zip is flat (`zip -j`), so at runtime
  `workflow_identity` is a **top-level** module — not
  `gateway_tool.workflow_identity`. A suite that loads `lambda_function.py`
  by file path with only the child root on the path dies with
  `ModuleNotFoundError` on the gateway's own import while the Lambda works
  perfectly.
- **`test/` has no `__init__.py`** (matching every child suite), so
  `python3 -m unittest test.test_units.Class.test_name` raises
  `ModuleNotFoundError` — which prints as `FAILED (errors=1)` and reads
  exactly like a mutation test that worked. Use `discover -s test -k <name>`
  to run a single test.
- **Walk `self.nested()`, not `self.root["Resources"]`, for anything
  per-child.** The root is no longer nothing-but-nested-stacks — it also
  declares `BundleProxyBucket` and its policy — so an unfiltered walk crashes
  looking those up in `CHILDREN`. Filter by
  `Type == AWS::CloudFormation::Stack`.
- **Assert the exact condition SET, never the absence of a `Conditions`
  block.** "No `Conditions` at all" as a proxy for "no region split" died the
  moment the optional proxy bucket needed one. Pin the set, which still fails
  if a region-splitting condition appears without banning conditions
  outright.
- **`TestBucketValidatorParity` and `TestCustomResourceTypeUniformity` read
  the SIBLING directories off disk**, not this one. The paths hang off the
  repo root derived from `__file__`, not the cwd, so the suite runs from
  anywhere. What matters is that it reads files this directory does not own:
  a legitimate fix to the validator fails the byte-equality test until the
  fix is copied to **every child that carries it**. That failure is the test
  doing its job — copy the file, do not relax the assertion.
  `_VALIDATOR_CHILDREN` lists them deliberately rather than globbing, and
  `test_the_children_that_declare_a_validator_are_exactly_these` pins the
  set.
- **Do not pin the tier COUNT per child, only the tier vocabulary.** Counts
  differ legitimately and change whenever a workflow gains a tier for testing
  headroom; a count assertion fails on a correct change, which trains people
  to edit the test instead of reading it.
- **Pin SETS and CEILINGS, never TALLIES — in the tests and in this file.**
  A set (`CHILDREN`, `USAGE_TRACKING_STACKS`, `_VALIDATOR_CHILDREN`,
  `_TIERED_CHILDREN`, `OPTIONAL_TARGETS`) fails in both directions and
  *names* what changed, so it is worth the edit a new child costs. A tally
  ("all twelve children", "nine container images") grows by design, is true
  of nothing in particular, and rots without a single test noticing. Derive
  the number at assert time from the set, and write prose against the set
  ("every child") rather than its size. Two numbers legitimately stay
  numeric: a **ceiling that IS the invariant** (the root's parameter list is
  the contract), and a recorded measurement, which is evidence and must not
  be edited to match today's tree.
- **The suite iterates `CHILDREN`, so an unregistered child is INVISIBLE to
  it.** A new workflow directory that builds and deploys but never joins
  `CHILDREN`/`TOKEN_SUFFIXES`/`WORKFLOWS`/the root template passes the whole
  suite, and the omission first surfaces as a missing gateway target on a
  live bundle deploy. `test_every_deployable_stack_in_the_tree_is_a_registered_child`
  closes that direction by discovering from disk (any sibling with both a
  `cfnstack.yaml` and a `build-deploy-s3-dist.sh`) and naming the omission.
  Exempting a stack is possible but must be *stated* — add it to
  `NOT_BUNDLE_CHILDREN`.
- **Check `assertIn` before subscripting a dict in a test.** A missing key
  raises `KeyError`, which unittest reports as `errors=1` — indistinguishable
  at a glance from the `ModuleNotFoundError` trap above.

`_CfnLoader` preserves CFN short-form intrinsics as `("!Tag", value)` tuples
rather than flattening them, so a test can assert on *which* intrinsic was
used and not merely on the resolved string.

**When mutation-testing, restore with `cp` from a backup — never
`git checkout --`.** Files in a working tree can carry uncommitted work from
other streams, and `git checkout` on one discards it silently with no reflog
entry to recover from. Take the backup first (`cp f /tmp/f.bak`), restore
from it, and `diff -q` at the end to prove the file came back byte-identical.
**Also verify the mutation actually applied** — a target string with wrong
indentation makes the mutation a silent no-op, and the test then reports
`OK`, which reads exactly like a weak assertion. Assert the target was found
before writing.

---

## Gotchas

Things that look like bugs but aren't, and things that look fine but are.

- **Bump `.version` before every deploy.** See
  [Versioning](#versioning) — this applies to each child's `.version` too.
- **Two bundles in one account+region MUST vary `--gateway-name` or
  `--environment`.** Varying `--prefix` alone is **not enough**:
  `media-toolkit-gateway` has no `ResourcePrefix` parameter and names its
  Lambdas, log groups, secret, gateway, and Cognito domain off
  `GatewayName`/`Environment`. A second bundle differing only in prefix
  brings up every other child without collision and then fails on the
  gateway stack. §P7 prices this out; it is accepted, not overlooked.
- **The children are only self-consistent under `--single-region`.** Every
  child substitutes `%%BUCKET_NAME%%` with the bucket *basename* while
  uploading artifacts to `${BUCKET_NAME}-${BUCKET_REGION}` when that flag is
  absent, and no child template appends `${AWS::Region}` to compensate. This
  bundle resolves the staging bucket **once** and always passes
  `--bucket <resolved> --single-region` to every child so each child's `sed`
  and its own upload agree. That is also why the root template has **no
  `bSingleRegion` condition** — copying one in would add a second independent
  way to name the wrong bucket. §P3.
- **`aws cloudformation validate-template` on the root asks for
  `CAPABILITY_AUTO_EXPAND`. Do not grant it.** That response is a
  conservative superset for any template containing a nested stack — CFN
  cannot see inside a child at validate time. Measured: a minimal template
  whose only resource is an `AWS::CloudFormation::Stack` with no IAM and no
  transform reports the same, yet `create-stack` on it succeeds with
  `CAPABILITY_IAM` alone and with no capabilities at all. No child declares a
  transform. `CAPABILITY_NAMED_IAM` is likewise passed **defensively, not
  because a child needs it** — no child declares
  `RoleName`/`ManagedPolicyName`; every named-looking hit is an inline
  `PolicyName` inside a role's `Policies:` list.
- **The unsubstituted-token guard must be `%%[A-Z0-9_]*%%`, not
  `%%[A-Z_]*%%`.** The narrow class cannot match a digit, so it silently
  misses `%%KEYPREFIX_M2C_AGENT_TOOLS%%` (the `2` in `M2C`). A surviving
  token deploys as a literal `TemplateURL` and fails the stack after the
  first child's long CodeBuild has already run.
- **The children's DynamoDB tables have no `DeletionPolicy: Retain`, by
  design.** (hybrid-search, bedrock-inference, face-recognition and tts are
  the children that declare tables.) Stack delete is meant to delete them —
  their contents are derived from media that lives elsewhere and are
  rebuildable, and `Retain` would trade a recoverable loss for a permanent
  `already exists` blocker on the next create. Do not "fix" this. **One
  exception exists and is not a DDB table:** hybrid-search's
  `HybridSearchVectorBucket` (`AWS::S3Vectors::VectorBucket`, both retain
  policies) deliberately survives stack deletion, because re-embedding costs
  real Bedrock spend. **Deleting a bundle therefore strands buckets — two
  classes of them.** hybrid-search's `*-vectors` bucket always, plus
  `BundleProxyBucket` (`*-proxy`) whenever `ProxyBucket` was left blank.
  Both names embed a stack-id-derived slice, so a re-create never reuses
  either. Read the `ProxyBucketName` and `ResourcePrefix` outputs **before**
  deleting — they are the only record of what to sweep — then either sweep by
  hand or pass the proxy bucket back via `--proxy-bucket`. The proxy bucket
  also carries `UpdateReplacePolicy: Retain`, and *that* is the policy which
  actually fires in normal use: `BucketName` is replacement-forcing, so
  without it editing `Prefix` on a live stack would **delete every workflow
  output**. Measured: flipping the condition off logs `DELETE_SKIPPED`
  rather than deleting, and a full stack delete leaves the bucket in place.
  That stack-id derivation is also load-bearing: the native type **cannot
  adopt** an existing bucket (measured: `409 AlreadyExists` rolls back the
  whole stack), so a stable operator-supplied prefix would make re-create
  impossible. §P9/§P10 item 11.
- **`IngestBucket` has no default on purpose; `ProxyBucket` deliberately
  does (`""`).** They are not symmetric. `IngestBucket` holds source media
  this stack did not produce and cannot recreate, so a typo would deploy
  clean and then find nothing to process — no default is the only guard, and
  it must not gain one. `ProxyBucket` used to be mandatory for a *different*
  reason: several children run a custom resource that **creates** the proxy
  bucket when absent, so a typo silently made a stray bucket and wrote every
  output somewhere nobody was looking. That is now fixed at the source
  rather than guarded — blank means **this stack declares the bucket itself**
  (`BundleProxyBucket`), and CFN orders it ahead of every child through the
  `!If`'s reference, so the validators' create path is unreachable inside
  the bundle (measured on a live create: the bucket completed before the
  first child started, with no `DependsOn` anywhere). **The validators are
  still required** — a child deployed standalone has no bundle to declare
  its bucket. The resource type is `Custom::ProxyBucketValidator` in every
  child that has one; `TestCustomResourceTypeUniformity` pins the single
  spelling.
- **Every child must receive `ProxyBucket` as the `!If`, never a bare
  `!Ref`.** A single missed call site passes `""` to that one child, so one
  workflow writes to a bucket named empty-string — and it deploys clean
  whenever a bucket *is* supplied, i.e. it survives exactly the testing a
  deployer is most likely to do.
  `test_every_child_gets_the_conditional_bucket_not_a_bare_ref` pins them
  against `CHILDREN` as a set.
- **`Prefix` is capped at 14 characters, and that number is arithmetic, not
  taste.** `ResourcePrefix` = `{Prefix}-{12-char stack-id slice}` ≤ 27, and
  27 is the `min()` over every name derived from it: hybrid-search's
  `-{account 12}-{region 14}-vectors` spends 36 of S3's 63 (**binds at 27**),
  this template's `-…-proxy` spends 34 (allows 29), and
  `-audiostem-separation-bucket-check` spends 34 of Lambda's 64 (allows 30).
  So `Prefix` ≤ 27 − 1 − 12 = 14. Verified live: CFN rejects 15 chars with
  *"must contain at most 14 characters"*, and 14 renders at exactly 27.
  **The `-bucket-check` rename was a hard prerequisite:** the old 44-char
  validator suffix bound the ceiling at 20, which would have left `Prefix`
  at 7. §P10 item 10.
- **`AllowedPattern` on `Prefix` is load-bearing, not cosmetic.** It lands
  inside S3 bucket names, where uppercase and underscores are illegal.
  Without it the failure surfaces as an S3 error on a *nested* stack instead
  of as parameter validation before anything is built. All three rejections
  (uppercase, underscore, leading hyphen) verified live.
- **The gateway is an admin gateway.** `HybridSearchAdminToolLambdaArn` is
  wired unconditionally, and that target grants `configure_search_backend`
  and `delete_collection` (which sweeps LanceDB tables, S3 Vectors indexes,
  DDB records, and the registry). The security boundary is the credentials
  secret at `/{GatewayName}/{Environment}/gateway-credentials`, **not** the
  target list. Serving untrusted agents means deploying
  `media-toolkit-gateway` standalone with `HybridSearchAdminToolLambdaArn=""`.
- **Deleting the stack leaves Lambda-auto-created log groups behind**, and
  they block the next `create-stack` with `already exists`. Any child whose
  custom-resource function is invoked at deploy time needs a `DependsOn` on
  its declared log group, or Lambda's first invocation wins the race and
  creates an unmanaged one.
- **An UNNAMED Lambda reaches its declared group through
  `LoggingConfig.LogGroup`, not `/aws/lambda/{FunctionName}`.** CFN generates
  the physical name, so the canonical group name is unknowable at author
  time; `LoggingConfig` decouples the two. Every `StartBuildLambda` works
  this way. Verified live: the function logs into the declared group, **no**
  auto group appears for the generated name, the role needs no
  `logs:CreateLogGroup`, and the group dies with the stack. Skipping this is
  what left a pile of orphaned `*-StartBuildLambda-<random>` groups behind —
  one per stack lifecycle, retention `None`, forever.
- **Never declare a log group at a name a previous deploy already
  auto-created.** It fails at pre-flight validation — `Validation failed
  with N error(s)` with **no per-resource event naming the culprit**, because
  the stack never got far enough to touch a resource. **This kills a
  `create-stack`, not just an update**, and an orphan outlives the stack
  that made it, so a *first* deploy into a "clean" account can fail on
  groups left by a stack deleted months ago — measured in exactly that form,
  with both colliding groups left by earlier standalone deploys whose
  Lambdas no longer existed. Diagnose by counting the errors and comparing
  against the groups just added. CFN cannot skip a resource that already
  exists: `Conditions` evaluate from parameters and pseudo-parameters only,
  never a runtime lookup. Two ways out — rename the function so the group
  name is new (what `-bucket-check` did), or point a `LoggingConfig.LogGroup`
  at a `-log`-suffixed group and leave the orphan to be swept. Prefer the
  `-log` route when the orphan holds logs worth keeping: renaming is a
  `createOnlyProperty` replacement that **deletes** them. `retentionInDays`
  tells you which groups are already stack-owned: declared ones read a
  number, Lambda-auto-created ones read `None`.
- **`LoggingConfig` is not enough on its own — drop
  `AWSLambdaBasicExecutionRole` in the same edit.** That managed policy
  grants `logs:CreateLogGroup` on *every* group, so a function that cannot
  reach its declared group silently creates an unmanaged one rather than
  failing visibly. It is invisible to a Statement-scanning test because a
  managed policy ARN carries no statements. Replace it with an inline
  `logs:CreateLogStream`/`PutLogEvents` scoped to `!GetAtt <Group>.Arn`.
  `test_no_declared_group_lambda_role_carries_the_basic_execution_policy`
  pins it; tts's `TtsLambdaRole`/`GatewayToolRole` are knowingly excluded as
  pre-existing tier-role debt.
- **Renaming a resource's LOGICAL ID while it keeps an explicit physical
  name cannot be done in one update.** CFN treats a logical-id change as a
  replacement and performs it **create-before-delete**, so the new id tries
  to create a name the old id still owns. It fails at *pre-flight*:
  `Validation failed with 1 error(s)`, **no per-resource event naming the
  culprit**, then `UPDATE_ROLLBACK_COMPLETE`. Measured live on a throwaway
  stack holding two log groups that differed only in logical id. The escape
  is a **two-step deploy** — drop the resource in one update, re-add it
  under the new id in the next (verified) — and for a log group the contents
  do not survive the gap. Resources with **no** explicit physical name
  rename cleanly in one update. This is a *different* failure from the
  `createOnlyProperty` one above: there the name itself changes, here only
  the logical id does.
- **`gateway_tool/workflow_identity.py` is the second copied-not-imported
  module** — see §C2 for the full rules. It exists because the derivation it
  replaced was hand-copied into the gateways and had **already drifted three
  ways**.
- **A gateway module is only distributed if it is in the zip, and `zip` will
  not tell you.** See §C5. Every child lists members in a `members=()` array
  and checks each with `[ -f ]` before zipping;
  `test_the_module_ships_in_the_gateway_zip` pins both the membership and
  the guard.
- **`gateway_tool/bucket_validator.py` is byte-identical in every child that
  declares a validator and is COPIED, not imported.** This is not hygiene:
  five siblings once read `ResourceProperties["AccountId"]` *outside* their
  `try`, and CFN's `Delete` event does not carry it — the `KeyError`
  escaped, no response was sent, and the stack sat in `DELETE_IN_PROGRESS`
  for a full hour, twice, each time orphaning the validator Lambda whose
  surviving `FunctionName` then blocked the next `create-stack`. The bug hid
  behind a file that *looked* shared. `TestBucketValidatorParity` asserts
  byte-equality, the guarded read, the handler name, and the
  `PhysicalResourceId` fallback. Fix a bug here and copy it to every child
  that carries it (`_VALIDATOR_CHILDREN` is the list).
- **A tier name means the same memory in every child, and declaring a tier
  no state routes to is normal.** See §C8. The *count* differs freely and
  unrouted tiers are the norm rather than dead weight.
  `TestMemoryTierVocabulary` pins the memory/name pairs and the allowed key
  set in both directions, but deliberately does **not** pin the count.
- **Never rename a stack-owned log group that holds logs you want.**
  `LogGroupName` is a `createOnlyProperty`, so a rename is a *replacement*
  and CFN **deletes the old group with everything in it** — measured,
  nothing is left behind. Check the group is empty first; consistency is not
  worth deleting production logs for.
- **The proxy-bucket validator Lambdas are named `-bucket-check`, not
  `-proxy-bucket-validator`.** The logical ids are still
  `ProxyBucketValidator*`. The old physical name sat at exactly **64**
  chars — the hard `FunctionName` limit — so one more character in
  `ResourcePrefix` would have failed the deploy. §P10 item 10 has the
  budget.
- **A nested-stack failure is reported only as the child's logical id**
  (e.g. `MediaProcessingStack`). Open that nested stack in the console for
  the underlying resource error; the root stack's events will not name it.

---

## Design record

Numbered decisions the templates, scripts, and tests cite (`§P…` for bundle
design, `§PI…` for prompt/inference — see above). Each is a settled decision
with its argument; re-opening one means answering the argument, not just
preferring the alternative.

### §P1 — all children, always

No per-workflow opt-out flags. An earlier design had them, with
media-processing and the gateway exempted; the exemption list was the
symptom that killed it:

1. **media-processing is not optional at the CFN level.** Several children
   declare `MediaProcessingStateMachineArn` with no `Default`; CFN rejects
   the deploy outright if it is absent.
2. **Opting a workflow out of an already-deployed bundle destroys its
   registry.** The stateful DDB tables (face records, voice catalog,
   search-collection registry) have no `DeletionPolicy` **by design** —
   their contents are derived from media that lives elsewhere and are
   rebuildable, and `Retain` would trade a recoverable loss for a permanent
   `already exists` blocker on the next create. Removing a nested stack on
   update deletes the child and takes the tables with it, under any opt-out
   mechanism. Deploying everything unconditionally means no update can ever
   remove a child stack — a stronger guarantee than a guard flag, for free.
   Anyone who needs one workflow alone still has that workflow's own
   `build-deploy-s3-dist.sh` and `cfnstack.yaml`, untouched.
3. **The gateway gets simpler, not harder.** With everything deployed, its
   optional-target conditions are permanently true and no cross-reference
   needs an `!If` — which is exactly where that pattern breaks in practice.

### §P3 — deploy order and `TemplateURL` construction

The dependency order is forced by `MediaProcessingStateMachineArn` having no
default and by the gateway consuming every target ARN:

```
              MediaProcessingStack
                      │
    ┌──────────┬──────┴─────┬──────────────┐
AudioStem  SpeechDiar  SpeechEnhance  FaceRecognition      (also: ShotSegmentation,
    │          │            │            │                  HybridSearch, Tts,
    └──────────┴─────┬──────┴────────────┘                  PromptRegistry,
                     │                                      BedrockInference — no
   consumer children (conversation-analysis, …)             MP dependency, start in
                     │                                      parallel)
              M2cAgentToolsStack        (last — consumes every target ARN)
```

`DependsOn` is **implicit and sufficient** via `!GetAtt` references — CFN
derives ordering from the reference; adding explicit `DependsOn` would be
redundant noise. The build script's `WORKFLOWS` table is kept in deploy
order: build order does not have to match (children are independent at build
time), but matching means a media-processing build break stops the run
before every other build reports success, which is the more useful failure.

Each child template lives at its own version prefix, so the root cannot build
the URLs from one `KeyPrefix`: the `Mappings` carry one full
`{solution}/{version}/{workflow}/{template-name}` key per child, all written
by `sed` at build time, and the staging bucket is resolved **once** (see the
`--single-region` gotcha — there is deliberately no `bSingleRegion`
condition; the children substitute the bucket basename but upload regionally,
so the bundle always passes `--single-region` and writes the resolved name).

The logical resource names (`MediaProcessingStack`, …) are permanent API —
renaming one deletes and recreates the child stack (and its DDB tables).

### §P7 — the gateway ignores `ResourcePrefix`

The workflow children name every resource `${ResourcePrefix}-…`;
`media-toolkit-gateway` has no `ResourcePrefix` parameter at all and names
resources off `${GatewayName}-…-${Environment}` instead. **Decision: leave
as is**, with the downside priced: it is a naming-collision cost only — no
data risk, fails fast at create. Two bundles in one account+region differing
only in prefix produce non-colliding child stacks and one colliding gateway
stack (functions, log groups, the credentials secret, the AgentCore gateway,
and the Cognito domain — which already appends `${AWS::AccountId}`, so no
cross-account collision is reachable). The rule that follows:
**`GatewayName` + `Environment` is the gateway stack's uniqueness key**, and
`Environment` (`dev`/`staging`/`prod`) exists precisely to be varied.

### §P9 — the retained vector bucket

hybrid-search's vector bucket is named
`${ResourcePrefix}-${AWS::AccountId}-${AWS::Region}-vectors` with both
retain policies — the only resource that deliberately survives stack
deletion, because re-embedding costs real Bedrock spend. The sharp edge:
`AWS::S3Vectors::VectorBucket` **cannot adopt** an existing same-named
bucket (measured: `409 AlreadyExists` rolls back the whole stack, and a CFN
`Condition` cannot rescue it). It does not bite only because
`ResourcePrefix` embeds the stack id, so a re-create gets a new name and a
retained orphan — recoverable, where a deleted vector corpus is not. **The
dependency runs the other way from how it looks**: the stack-id derivation
is what makes the native resource safe. If `ResourcePrefix` ever becomes
operator-supplied and stable across a delete+recreate, this resource breaks
and needs an adopting `Custom::` resource. Also measured:
`VectorBucketName` is a `createOnlyProperty`, not a `readOnlyProperty`, so
`!Ref` returns the **ARN** and the name is sliced out of it
(`!Select`/`!Split`) — one definition in the template.

### §P10 — the character budget and the proxy bucket

**Item 9 / item 10 — the budget.** The ceiling on `ResourcePrefix` is the *minimum*
over every name derived from it, and which constraint binds changed with the
validator rename:

| Constraint | Limit | Suffix | Ceiling on `ResourcePrefix` |
|---|---|---|---|
| vector bucket `-{account 12}-{region 14}-vectors` | 63 | 36 | **27** ← binds |
| proxy bucket `-{account 12}-{region 14}-proxy` | 63 | 34 | 29 |
| `-audiostem-separation-bucket-check` (post-rename) | 64 | 34 | 30 |
| `-audiostem-separation-proxy-bucket-validator` (pre-rename) | 64 | 44 | 20 |

Worst-case region is 14 chars (`ap-southeast-N`); an account id is always
12; Lambda's 64-char `FunctionName` limit is measured, not read. So the
validator rename to `-bucket-check` was a **prerequisite** for the 14-char
`Prefix` cap (`27 − 1 − 12`), not a consistency nicety — before it the same
arithmetic gives 7. The direction of that comparison was gotten backwards
twice during design (dismissing the tighter constraint as "not binding"),
and was caught by a unit test, not by re-reading the numbers — which is why
`test_units.py` asserts every name still fits at `ResourcePrefix=27` instead
of trusting the prose.

**Item 10(d) — `Prefix`, not `ResourcePrefix`, is the parameter.** The root
exposes `Prefix` (`MaxLength: 14`, `AllowedPattern: ^[a-z0-9][a-z0-9-]*$`,
default `umt`) and derives
`ResourcePrefix = ${Prefix}-${12-char stack-id slice}` for every workflow
child. The gateway stack receives no `ResourcePrefix` — it has no such
parameter (§P7); `test_no_child_receives_the_bare_prefix` asserts the others
get the `!Sub` and pins that one exception by name. The deploy script
rejects the retired `--resource-prefix` flag **explicitly** — its argument
parser ends in a catch-all, so the old flag would otherwise be silently
ignored and the stack would deploy under the default prefix, colliding with
the bundle it was meant to be isolated from. The script also validates
`Prefix` ahead of the API call, because CFN's own rejection arrives without
the arithmetic that explains the ceiling.

**Item 10(a–c) — log-group ownership.** The named validator Lambdas were
renamed (`-bucket-check`) so their declared log groups could be created
fresh (declaring the canonical name of a previously auto-created group
fails pre-flight — see Gotchas); the unnamed `StartBuildLambda`s route
through `LoggingConfig` to declared groups with retention and `DependsOn`
edges; both sets of roles dropped `AWSLambdaBasicExecutionRole` for inline
grants scoped to the declared group. All verified end-to-end on throwaway
stacks, not by inspection.

**Item 11 — the bundle owns the proxy bucket when not given one.**
`ProxyBucket` was once mandatory purely to guard against the child
validators' create-on-404 path (a typo silently made a stray bucket and
sent every output somewhere nobody was looking). Better than guarding the
failure mode is removing it: blank means the bundle declares
`BundleProxyBucket` (`${ResourcePrefix}-${AWS::AccountId}-${AWS::Region}-proxy`,
both retain policies, public-access-block + AES256 + `DenyInsecureTransport`
— the security floor, deliberately nothing more: no access logging, CORS,
acceleration, or versioning, each priced and rejected), and CFN orders it
ahead of every child through the `!If` reference alone. The validators stay
— a child deployed standalone still needs one — but their create path is
unreachable inside the bundle. `IngestBucket` stays mandatory: it holds
source media the bundle did not produce and cannot recreate, and
auto-creating an empty one would turn a typo into a stack that deploys
clean and finds nothing to process. The recovery path for a stranded proxy
bucket is pointing `--proxy-bucket` at it explicitly — which works
precisely because the parameter stays optional rather than being removed.

### Root parameters and outputs

Seven parameters: `Prefix`, `IngestBucket` (no default, on purpose),
`ProxyBucket` (default `""` = stack-owned), `GatewayName`, `Environment`,
`OAuthScope`, `SecretsKmsKeyId` — everything else the children need is
either defaulted at the root or wired from a sibling's output; nothing is
asked twice. The parameter list IS the contract
(`test_exactly_seven_parameters` — a deliberate ceiling, see "pin sets and
ceilings"). Outputs answer "what do I hand my agent?" (`GatewayUrl`,
`GatewayCredentialsSecretName`, `GatewayIdentifier`) and "what do I sweep
after a delete?" (`ProxyBucketName`, `ResourcePrefix`,
`HybridSearchVectorBucketName`, plus the stateful table names). **No
`Export:` on any output** — an exported value cannot change and the stack
cannot be deleted while any importer exists; the bundle would be the worst
possible thing to hard-lock.
