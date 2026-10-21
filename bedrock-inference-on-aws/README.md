# bedrock-inference-on-aws

**N already-rendered prompt records in, N responses out.** Three Step Functions state
machines, seven Lambda memory tiers over one container image, three DynamoDB tables.
Takes an `input_params.json` + `request.jsonl` pair on S3 and returns `response.jsonl` +
`summary.json` + `usage_tracking.json`, running either **synchronously under a
concurrency lease** or through a **Bedrock batch job**, with per-record status,
cross-family model fallback, and token accounting.

It does **not** render prompts. Templates, variable substitution, and prompt selection
belong to the caller (`prompt-registry`); this workflow's input is already-final
`system` + `messages` blocks (§10 rule 2).

This README is the single developer document for the workflow. The numbered
**[Design record](#design-record)** at the end preserves the section numbers that code
comments cite (`§N` for the original design plan, `§MN` for the semaphore-merge plan,
`§PN` for the prompt-inference integration records).

## Self-contained: it CONTAINS the semaphore, it does not consume one

There is no sibling stack and no deploy order. A standalone `service-quotas-on-aws`
stack used to own the concurrency cap and **no longer exists** — §M records the merge.
`lib/semaphore_client.py` and `states/acquire_lease/` are the same code, in-stack.

**What the split bought, and what is now gone.** `states:SendTaskSuccess` was once
believed unscopeable (corrected later: it accepts machine-ARN scoping, now bound to this
account+region), so the standalone design confined that grant to one function per stack,
neither of which could call Bedrock. The merge narrowed that isolation from one stack to
two roles (§M2.4); the tier collapse narrowed it to **nothing**. One
`BedrockInferenceLambdaRole` now holds `states:SendTask*` **and** `bedrock:InvokeModel`,
so the function that redeems a token is the same function that can call the model. See
[One image, seven memory tiers, one role](#one-image-seven-memory-tiers-one-role) — the
separation survives only as a convention of the code, and
`test_exactly_one_role_sends_task_results_and_it_can_also_call_bedrock` is named to say
so.

**The one semantic loss: the cap is per PREFIX, not per region.** The singleton was
enforced by the table NAME — one quota table per region, so a second stack failed at
CREATE. Both tables now carry this stack's prefix, so two deployments hold two
independent caps against one real Bedrock quota and can jointly exceed it (§M2.3).
Nothing prevents it; `summary.quota_table` reports the table name so the split shows up
in the output rather than only in the throttling.

## Three machines: an orchestrator and two standalone children

```
ORCHESTRATOR ── Validate & plan ─► Choose mode ─┬─ Run sync inference  ─┐
                                                └─ Run batch inference ─┴─► Usage tracking
                                                    (.sync:2, one per branch)

SYNC CHILD    ── Acquire lease ──► Run inference ─────────► Release lease ──► Summarize
                  (waitForTaskToken)  (checkpoint/re-enter)

BATCH CHILD   ── Submit batch ──► Collect batch output ───────────────────► Summarize
                  (waitForTaskToken,      │ Catch                            ▲     ▲
                   up to 24 h)            │ BatchRecordsIncompleteException  │     │
                                          └─► Run repair inference ──────────┘     │
                                                (.sync:2 → the SYNC CHILD)         │
                                                │ Catch States.ALL ────────────────┘
```

**The batch child's third state IS the sync child** (§P8). A batch job has no chain
fallback and no retry ladder, so a handful of per-record misses used to be permanent
`FAILED` lines. `Collect batch output` now writes `response.jsonl`, seeds
`_progress.json` with the batch's *successes*, and **raises**
`BatchRecordsIncompleteException`; the `Catch` on that name routes to a nested
`startExecution.sync:2` of the sync child named `{workflow_id}-repair`, which resumes
from that checkpoint and therefore infers **exactly the gaps** through the normal chain
walk. There is no duplicated lease/checkpoint/release spine — the repair is the same
machine, so either child is still startable on its own given a plan.

**Plan in the orchestrator, Summarize in each child** (§M decision 1). Two call sites
for `build_summary`, deliberately: each child ends at its own `Summarize`, so **either
child is startable on its own given a plan**. A single summarize state in the
orchestrator would have made them un-runnable alone. Planning in the orchestrator turns
"plan before you lease" from a state ordering into a machine boundary — no future edit
can reorder planning into the held span.

**`Usage tracking` walks the whole execution TREE, not two histories.** The
orchestrator's own history is three short Lambda calls and one `.sync:2` span; every
Converse retry, every checkpoint re-entry and the whole lease wait live in the child —
and since the repair tail, a **grandchild** can live under the batch child. Reading only
`$$.Execution.Id` would attribute a long run to a few seconds of compute. So it
**recurses**, discovering children from `TaskSubmitted` events whose
`taskSubmittedEventDetails.resourceType == "states"` by pulling `ExecutionArn` out of
the event's `output` — which is why `includeExecutionData=True` is not optional
decoration but the discovery mechanism itself. `child_wall_sec` is a **max, not a sum**,
because siblings overlap in wall-clock; the walk dedups on arn and bounds itself against
a cycle. `retried` counts `invocations - 1`, not failed attempts — the two differ the
moment a state can fail and be caught, and the wrong one reports a phantom flake on
every repaired batch run. It is the **success path only** and **never raises** — as the
terminal state, an exception here would fail an execution whose `response.jsonl` and
`summary.json` are already written.

**There is no failure-path cost report, deliberately.** A `Track usage on failure` state
used to exist. What it captured was Lambda time — well under a cent (the tiers in use
are 896–3584 MB; even ten 900 s timeout retries is ~$0.12), and an *estimate* from
history timestamps rather than billing data. The money that matters is Bedrock tokens,
and those already survive any failure: `run_inference` checkpoints a **fully priced
response line per record** into `_progress.json` before it can fail. The state's price
was a `Catch` that laundered every distinct child error into one static `Fail`, so it is
gone. Use Cost Explorer for aggregate spend.

**The orchestrator is four states with NO `Catch` anywhere.** Failures propagate as
themselves — `BedrockInferenceFailed` with `Summarize`'s per-record message, a plan
`ValidationException`, a genuine `States.Timeout`. Catching could only re-raise through
a `Fail`, whose `Error`/`Cause` must be **static** (a `States.Timeout` carries no
`Cause`; an intrinsic over an absent path fails the state), i.e. trade a specific error
for a generic one. `Choose mode` has **no `Default`**: an unmatched mode is
`States.NoChoiceMatched`, the same outcome a `Fail` state would give, and a `Default`
falling through to sync would run a 5000-record set through a serial lease-held loop
that cannot finish inside the child's retry budget.

**`States.TaskFailed` is deliberately absent from both `.sync:2` retry ladders.** It is
what a genuine child failure raises, and the child name is fixed
(`States.Format('{}-sync', workflow_id)`) — so retrying it re-issues `startExecution`
under the same name and returns `ExecutionAlreadyExists`, which with no `Catch` to
launder it is what the caller would see *instead of* the real error. Only transport
failures, where no execution was created, are safe to retry under a fixed name.

**Sync takes a lease; batch does not.** The Bedrock in-progress-job limit is
account-level and service-enforced, so a local cap would serialize submissions Bedrock
is willing to accept in parallel (§1.1). The batch child touches the semaphore nowhere
at all.

**No state in any machine has a `Catch` for lease cleanup.** `Release lease on failure`
is **deleted** (§M3.2): the failure path is the terminal-state EventBridge rule, which
covers strictly more cases — a `Catch` cannot fire for an ABORTED execution or a Lambda
killed between releasing and returning, which is exactly what a cleanup path exists for.

**The two `Catch`es in the batch child are the only ones, and the invariant they respect
is `Catch[].Next` is never a `Fail`.** There are still **no `Fail` states in any machine
at all** — that is the property the old "no `Catch` anywhere" shorthand was really
about, and it is now pinned by name
(`test_a_failed_repair_degrades_and_never_reaches_a_fail_state`). Both `Catch`es carry
an **explicit nested `ResultPath`** (`$.data.batch_incomplete`, `$.data.repair_error`)
rather than the default `$`, which would replace the whole input and leave the repair
with no plan to run on; a nested path *merges*. The second one catches `States.ALL` and
degrades to a normal `Summarize`: repair-off would have SUCCEEDED with the same `FAILED`
lines over a `response.jsonl` that is already complete, so repair-on must not do worse.
The reason lands in `summary.repair_error`, which is the only trace a caught error
leaves that a caller can read.

**Which is also why the routing signal is a `Catch` and not a `Choice`.** A `.sync:2`
child failure reaches its parent as `States.TaskFailed` and *never* as the child's own
error name, so an orchestrator-level catch could not discriminate the repair signal —
but a `Catch` on a **Lambda** task can, because Step Functions matches the Python
exception **class name**. The cost is that a normal outcome renders in the console as a
failed state, and that is the accepted trade: a batch run that left records behind
*should* look like it had a problem rather than silently taking a `Choice` branch. It
also means `Retry` must never see this exception (`Retry` precedes `Catch`) —
self-enforcing today, because every ladder in the repo names only
`Lambda.*`/`StepFunctions.*` prefixes that a Python class name cannot collide with, and
`test_no_retry_ladder_anywhere_can_swallow_a_control_flow_exception` fails the build on
a `States.ALL` that would break it.

## `max_tokens`: retired as a caller knob, re-added as registry data

**`max_tokens` is REMOVED outright as a caller parameter** — one more parameter is one
more maintenance cost, so the simple contract won: `inferenceConfig.maxTokens` is never
emitted on a caller's say-so and Bedrock's own default applies. The knob went through
three designs: a flat default (which capped the 128k-output Claude families at ~6% —
the probed output ceilings: Claude 4.6/5 families 128k, nova-2-lite 65,535, pixtral
131,072, llama4-maverick 8,192), then per-model registry resolution with clamping, then
removal. Consequences, each pinned by `TestMaxTokensIsRetired`: `inferenceConfig`
appears only when it has something to say (temperature, for models that take it and only
with reasoning off) and is omitted entirely otherwise; a stray request-level or
per-record `max_tokens` is warned-and-ignored (the retired-`mode` treatment) and
stripped before the records digest, so two requests differing only in the retired field
are ONE `workflow_id`; the registry carries no `max_output_tokens` column; and the batch
PADS keep their internal `maxTokens: 1` — machinery, not a caller knob (60–99 pads per
padded job must stay nearly free), overriding any entry value.

**The deliberate re-addition is ENTRY data (`capabilities.emit_max_tokens`), never a
caller knob:** the "Bedrock default = model max" premise was REFUTED for the
thinking-required 5 family — their omitted-field Converse default is an undocumented
flat 4096, and adaptive thinking (billed as output) pushes long windows past it
(stop_reason `max_tokens` at exactly 4096, probed per model on fable-5-1 / opus-5 /
sonnet-5-5 / haiku-5-5; sonnet-4-6, haiku-4-5 and the GPT-6 trio measured clean,
finishing past 4096). Those four entries carry `emit_max_tokens: 16384` (4× the
truncating default, well under the 128K card max so a runaway record stays bounded),
emitted verbatim into `inferenceConfig.maxTokens` on every request. The
temperature-rejection recovery in `_converse_once` strips ONLY temperature and keeps the
block when it still carries the entry's maxTokens. The caller-side retirement is
unchanged.

## Mode selection: the dollar guard

Absent `use_batch`, the **dollar guard** decides: batch runs when the estimated
input-cost saving clears `BATCH_MIN_SAVINGS` ($0.25 default) — denominated in dollars,
not records, because 3 transcript chunks at ~100k input tokens deserve batch while 100
greetings deserve sync-in-seconds, and a count rule gets both wrong. `use_batch: true`
on a model that cannot batch **downgrades to sync with the reason reported** in
`mode_reason` and the summary — the old hard error retired; the "don't bill full price
without telling" rationale lives in the telling now. `use_batch: false` always means
sync. The request-level `mode` parameter is retired (warned-and-ignored).

## `workflow_id` is the idempotency key, and what it excludes is the design

**On the SHARED derivation** (micro-workflow-bundle/README.md §C2): [lib/identity.py](lib/identity.py)
delegates to the copied `workflow_identity.py` — byte-identical in
[gateway_tool/](gateway_tool/workflow_identity.py) (the flat zip's copy) and
[lib/](lib/workflow_identity.py) (the container's), pinned by
`TestSharedIdentityAdoption` with a golden known-answer vector. **The adoption changed
the hash preimage** (shared canonical form is `ensure_ascii=True`, material order
`payload | tool_version | digest [| token]`), which is why it shipped with a `.version`
bump and before any external caller existed. The records digest itself
(`digest_records`, `ensure_ascii=False`) is bedrock-inference-specific and did not move
— it enters the shared derivation as opaque `extra_material`. The execution is *named*
`workflow_id`, so a resubmit hits `ExecutionAlreadyExists` and is **absorbed** (§3).

`EXCLUDED_KEYS = ("output", "mode", "use_batch", "tool_version")` — the same records
through sync and through batch are the **same work**, so they collide, and the output
prefix is *derived* from the id rather than being an input to it. `tool_version` is
excluded from the request dict because it is hashed separately as its own term.

**`idempotency_token` narrows the dedup and IS the rerun feature.** Folded into the hash
as trailing extra material, then discarded — never forwarded to the state machine, never
stored in `input_params.json`, never echoed (all three pinned by
`TestGatewayTokenContract`). Absent token → dedup on the parameter hash (these records
are final rendered bytes and fully name the work); fresh token → new `workflow_id` →
fresh run at a fresh prefix, itself idempotent under retry, which a `force_rerun` flag
could never be. Same token + different records still re-runs, so a reused token cannot
hand back the wrong run's outputs. The retired `uuid` parameter is **rejected on
submit** (before any S3 write) and warned-and-ignored on polls, per the contract's
LLM-caller asymmetry.

Two properties, both with named tests:

- **Stable** across key insertion order, `output`, and `mode`. A hash that moves when
  nothing semantic moved re-pays for work already done, and **both executions succeed**
  — there is no failure to notice.
- **Sensitive** to any prompt byte, the model chain, record *order*, `TOOL_VERSION`, and
  the token. Record order matters because `response.jsonl` is in request order: sorting
  the digest would hand back a result whose lines pair with the wrong inputs.

The gateway tool **imports** `lib/identity.py` rather than reimplementing it, and
`StateValidateAndPlan` recomputes the **parameter hash** and compares it against the
execution name — the name wins (it is the batch job name and the token-table sort key; a
tokened run keyed on the recomputed hash would collide with the token-less run of the
same parameters and overwrite its parked task token). A mismatch on a token-less submit
means the request object was overwritten after submission; on a tokened one it is
expected, and the warning text says both.

## The registry's `batchJob.region` is a GEO, not an AWS region

`"us"`, `"eu"`, `"apac"`, `"global"`. Its **only** legitimate use is prefixing the batch
model id: a model invoked at runtime through `us.` may only batch through `global.`.

Passing it to `boto3.client(region_name=...)` builds a client for a region that does not
exist, and the failure is a DNS error that reads as a Bedrock outage. Worse,
`submit_batch` **persists** the region into the token row's `model_params`, so every
subsequent drain replays the poison and it outlives the execution.

The plan therefore emits **both**, and the names are not interchangeable:

| Key | Value | Consumed by |
|-----|-------|-------------|
| `batch_geo` | the registry geo (`"global"`) | reporting only — nothing downstream reads it |
| `batch_region` | `AWS_REGION` | `boto3.client(region_name=...)` |

Fixed at the producer, in `validate_plan`, deliberately leaving `submit_batch`/`drain`
unchanged — a per-consumer fix would leave the next consumer to rediscover it. Two tests
guard it: one asserts `batch_region` matches an AWS-region regex and is not in
`_BATCH_GEO_PREFIXES`, and one greps every other state module to assert `batch_geo`
appears nowhere.

## Sync: checkpoint and re-enter

40 serial Converse calls do not fit in one 15-minute invocation. The loop checks
`context.get_remaining_time_in_millis()` **before** each record, and with less than
`REMAINING_TIME_BUFFER_MS` (60 s) left it writes `_progress.json` and raises
`LambdaResourceTimeoutException` — which the ASL retries with
`MaxAttempts: 10, BackoffRate: 1`.

The raise is a **deliberate yield, not a failure**: `BackoffRate: 1` because an
exponential delay would idle a held lease.

Three consequences worth holding on to:

- **`refresh_holder()` once on state entry, not per record.** The re-entry *is* the
  state entry, which is what makes the holder TTL bound a state rather than a span
  (semaphore invariant 3, §5.1).
- **The progress read must distinguish absent from failed.** `get_json_if_exists`
  returns `None` **only** on a genuine 404 and re-raises everything else. A broad
  `except` reads a throttled or denied GET as "no checkpoint yet" and re-inferences
  every completed record — so the failure gets *more* expensive under load, which is
  exactly when it happens (§12.3).
- **Response lines are re-ordered from the record list**, not from the checkpoint.
  `done` is a dict keyed by `record_id`; a caller zipping its own inputs positionally
  would otherwise find them mis-paired after a re-entry.

## The model chain is per record, and `ValidationException` never advances it

`converse_with_chain` walks a chain of up to four models. The escalation rules are in
[lib/bedrock.py](lib/bedrock.py) and each has a test:

| Class | Codes | Behaviour |
|-------|-------|-----------|
| Retry same model | `ThrottlingException`, `InternalServerException`, `ModelErrorException`, `ModelNotReadyException`, `ModelTimeoutException`, `ServiceUnavailableException` | exponential backoff, then advance |
| Escalate immediately | `ServiceQuotaExceededException`, `AccessDeniedException`, `ResourceNotFoundException` | advance without retrying — backoff cannot fix an exhausted quota |
| **Never advance** | `ValidationException` | a malformed request is malformed for every model; walking the chain multiplies the error, the latency, and the input-token bill |

**The chain does not multiply the cap.** `quota.resource` is the *primary* model id:
fallbacks are per-record escalations inside one held slot.

**A poisoned record does not downgrade its neighbours.** One bad record among 40
produces one `FAILED` line; the other 39 stay on the primary model.

Cross-family fallbacks are preferable and same-family pairs **warn**: a same-family pair
shares a quota pool and shares regional capacity events, so the event the backup exists
to survive takes out both.

**No model takes an assistant-role prefill, in sync or in batch** — a request-shape rule
for the whole family set, not a per-model capability (§9.2 records the
narrowing-then-widening history). `build_converse_params` raises `BedrockError` when the
last message is `role: "assistant"`; the rejection resolves to
`error_code() == "BedrockError"`, not `"ValidationException"`, so it advances the chain
like any other model-specific refusal rather than halting it.

## Partial success IS success

39/40 → a **successful execution**, one `FAILED` line in `response.jsonl`, and the
reason in `summary.failures[]`. 0/40 → `BedrockInferenceFailed`, naming up to five
reasons.

`summarize` writes `summary.json` **first and raises second**, so the per-record reasons
survive the failure — a summary written after the raise is a summary that never exists
on the run that most needs it.

The counts are always reported, including on a SUCCEEDED run, and the gateway tool
surfaces `records_failed` rather than hiding it: a caller that reads only the status
would treat 1/40 as a complete result.

**On batch, 39/40 gets a second chance before it becomes the answer** (§P8) — the
repair tail re-runs the gap through the sync chain, so a `FAILED` line on a batch run
means it failed *twice*, on two different code paths, with the chain's fallbacks
available the second time. Three summary fields report the tail and they are mutually
exclusive by construction:

| Field | Written by | Means |
|---|---|---|
| `records_repaired` + `repair_execution_arn` | `summarize`, from `$.data.repair` | the tail ran; `records_succeeded − records_resumed` is what it **recovered**, not what it attempted |
| `repair_error` | `summarize`, from `$.data.repair_error` | the tail ran and failed; degraded to a normal summary |
| `mode_reason` `"+ no sync repair (…)"` | `collect_batch`'s own return, via `data.repair_skipped` | there **were** failed records and the tail was refused — systemic (over `REPAIR_MAX_FRACTION`), or `options.batch_repair: false` |

`repair_skipped` exists because `collect_batch` must not write `$.data.repair` itself:
that key is the ASL `ResultSelector`'s output and only the ASL may set it. Returning a
`needed: False` verdict under that name made `summarize` read a refusal as a repair that
happened, so **every clean batch run** reported a `records_repaired` equal to its own
successes — caught by `TestModeParity`, and the reason the two names are split.

**The bound is deliberate and `REPAIR_MAX_FRACTION` (0.5) is a flagged design item, not
a settled one.** All-failed is systemic — a shape or access defect — and re-running it
through sync re-pays the entire job at on-demand rate to fail again; it falls through to
`Summarize`, which raises `BedrockInferenceFailed` with the reasons as ever. Truncated
lines (`stop_reason: max_tokens`) are SUCCESS and therefore in the seeded checkpoint:
re-running them truncates again, and salvage is the caller's `parse_partial_output`.

## The semaphore: three holder-disappearance triggers, one decrement

`${ResourcePrefix}-sync-infer-quotas`. Per-holder items plus a transactional counter,
scoped by an opaque `{service}#{resource}` string where `resource` is the **primary**
model id — so each model is its own cap and a fallback chain does not multiply it.

```
semaphore#{service}#{resource}  counter            count, max_holders
semaphore#{service}#{resource}  holder#{exec_arn}  ttl
lease#{execution_arn}           {service}#{res}    ttl        ← reverse pointer
semaphore#{service}#{resource}  pending#{...}      task_token, created_at, ttl
stream-dedup                    {eventID}          ttl
```

A slot comes back three ways, **down from four** — the ASL `Catch` was the deleted one:

| Trigger | Path |
|---|---|
| Normal release | sync child's `Release lease` state |
| Terminal execution | EventBridge rule on **all three** machines × all four terminal statuses → `handle_execution_terminal` |
| Holder TTL | DynamoDB expiry |

All three converge on **one** `decrement_once`, deduped on the DDB stream record's own
`eventID`. `handle_execution_terminal` **deletes and only deletes** — the delete reaches
the stream, which decrements and drains, which is what makes an aborted execution wake a
waiter exactly as a clean release does. Since IAM can no longer enforce that convergence
(the one workflow role needs `dynamodb:UpdateItem` now), a test pins `decrement_once(`
callers to exactly the definition plus that one handler.

All three machines are on the terminal rule even though only the sync child ever holds a
lease: an orchestrator ABORTED mid-`.sync:2` does not necessarily terminate its child,
so the child's own terminal event is the one that reclaims. `{"handled": True,
"released": 0}` is the expected common case.

The reverse pointer exists because an EventBridge terminal event carries **only
`executionArn`**, and the scope is part of the partition key — so the holder is found
through `lease#{execution_arn}` rather than a GSI.

**`try_acquire` reads `CancellationReasons[1]` alone.** Reading reason 0 too would
report a cap-*filling* holder's own retry as saturated and deadlock it behind its own
slot. `max_holders` ratchets **up only**: a later caller's smaller request must not
throttle everyone already admitted.

## Batch: the token table, and why a completion can find its row

`${ResourcePrefix}-batch-infer-tokens` — a **second** table, not a partition of the
first. Disjoint access patterns (a GSI and a `Scan` here, a transactional counter there)
and disjoint roles; a shared table would grant every semaphore reader the task tokens.

One row per job, `pk="batch#{model_id}"`, `sk=workflow_id`, plus **one GSI** on
`job_arn`:

```
pk="batch#{model_id}"  sk={workflow_id}   status, task_token, job_arn, request_s3uri,
                                          model_params, created_at, ttl
GSI job_arn-index      job_arn HASH       INCLUDE [status, model_id]
```

The GSI exists because the EventBridge completion event carries a **`batchJobArn` and
nothing else**. `INCLUDE [status, model_id]` rather than `KEYS_ONLY` because
`states/batch_status` needs `model_id` to rebuild the pk for its `UpdateItem`, and a
second `GetItem` would defeat the index. The prior-generation implementation answers the
same question with a **second table** and a jobName join (§6.1).

`StreamViewType` is `NEW_AND_OLD_IMAGES`: the notifier reads the OLD image to make
itself a **transition** trigger rather than a state trigger.

**Statuses:** `PENDING → SUBMITTING → SUBMITTED → COMPLETED | FAILED`.
`PartiallyCompleted` maps to **`COMPLETED`** on purpose — per-record failures are
`FAILED` lines, which is partial success, not a failed job.

### Submit classifies its refusal, and the classification is load-bearing

| Outcome | Action | Why the alternative is worse |
|---------|--------|------------------------------|
| One of the 7 `BATCH_QUOTA_EXCEPTIONS` | `mark_pending`, return `PENDING`, stay parked | Failing throws away a submit the drain would have completed |
| Anything else (e.g. `ValidationException`) | log and **raise**; the row is left `SUBMITTING` | A `PENDING` row here is retried forever by every future drain, invisibly, until the caller's token times out |

The row is written **before** `create_model_invocation_job`, so a Lambda death in
between still leaves a record of the attempt. `SUBMITTING` is neither retried nor
notified; it expires on its own TTL.

### The drain, and the asymmetric pair of EventBridge rules

Two rules on the same `aws.bedrock` event, scoped oppositely, for opposite reasons
(§6.5):

| Rule | Scope | Target | Reason |
|------|-------|--------|--------|
| `BatchTokenRule` | this account + region + **job-arn prefix** | `half-core` (marker `notify`) | a foreign job's completion is never news about one of *our* task tokens |
| `BatchCapacityRule` | **unscoped** — no region, no arn filter | `1-core` (marker `drain`) | the job limit is per *account* and this stack is not a singleton, so a foreign completion is the only signal that capacity freed |

Plus `DrainScheduleRule`, whose payload is `{"operation": "Drain"}` with **no model
image** — the drain then falls back to `models_with_pending()` (a Scan), which is the
backstop for TTL expiry, manual deletion, and any missed event.

The drain **stops on the first quota wall** and leaves the remainder `PENDING`, drains
**oldest-first by `created_at`** (the sort key is a uuid5, so key order is arbitrary and
would starve whichever id hashes low), and drains **only the models named in the stream
images** — capacity is accounted per model, and draining B on A's news submits B's queue
against a limit nothing told us about.

A permanent submit failure inside the drain is `mark_terminal(FAILED)`, not left
`PENDING`: the stream then tells the notifier, which fails the parked caller with the
reason.

### The notifier redeems batch task tokens

A `FAILED` job goes out through **`send_task_failure`**, not a success payload with
`{"status": "FAILED"}` inside it. A caller handed an error through the success channel
has to remember to check, and the one that forgets treats an error payload as a result
(§12.3).

Idempotency is two-layered: the OLD-image transition check is the cheap half, and
`mark_terminal`'s compound `ConditionExpression`
(`attribute_exists(pk) AND #s <> :terminal_a AND #s <> :terminal_b`) is the half that
survives a redelivered stream record.

`TaskDoesNotExist` / `TaskTimedOut` are **swallowed** and the row cleaned up — the token
can never be redeemed. **Every other `ClientError` re-raises**, because the token is
still live and swallowing it strands the caller for its full 24 h.

## Four sequence diagrams

The four paths a run can take. Every arrow is a real call in the deployed stack; the
legend at the end names each resource and what it is doing. Two things are worth reading
these for, because neither is visible in any single file: **nothing polls anything** —
every wait is a parked task token woken by a stream or an event — and **no caller ever
decrements the counter or drains its own queue**; the table's own stream does both,
which is what makes a crash, an abort and a clean release converge on one code path.

Notation: `--->` is a call, `<---` its return or a call in the other direction, and `*`
is work a resource does to itself with no other lifeline involved. A lifeline that is a
`|` throughout a step is not participating in it — in particular, a state machine
sitting parked on a task token looks exactly like one that is idle, which is the point.

### Case 1 — sync, slot free (no wait)

```
(Z) caller  (A) gateway  (B) orchestrator SM  (C) sync SM  (D) acquire
(E) quota table  (F) run-inference  (G) release  (I) Bedrock  (J) stream handler

(Z) (A) (B) (C) (D) (E) (F) (G) (I) (J)
|   |   |   |   |   |   |   |   |   |
|--->   |   |   |   |   |   |   |   |    1. run_inference(payload)
|   *   |   |   |   |   |   |   |   |    2. compute workflow_id = sha256(model+params+records)
|   *   |   |   |   |   |   |   |   |    3. write input_params.json + request.jsonl to S3
|   |--->   |   |   |   |   |   |   |    4. StartExecution, execution NAME = workflow_id
<---|   |   |   |   |   |   |   |   |    5. {job_id: executionArn} — returns at once, no wait
|   |   |   |   |   |   |   |   |   |
|   |   *   |   |   |   |   |   |   |    6. Validate & plan (H): recompute workflow_id and compare,
|   |   |   |   |   |   |   |   |   |       resolve the model chain, emit quota{service,resource,
|   |   |   |   |   |   |   |   |   |       max_concurrent} and batch_geo / batch_region
|   |   *   |   |   |   |   |   |   |    7. Choose mode -> sync
|   |   |--->   |   |   |   |   |   |    8. startExecution.sync:2 — the parent BLOCKS on the child
|   |   |   |--->   |   |   |   |   |    9. Acquire lease .waitForTaskToken, $$.Task.Token
|   |   |   |   |--->   |   |   |   |    10. ratchet max_holders: READ the counter item FIRST
|   |   |   |   <---|   |   |   |   |    11. already 3 == requested -> NO WRITE
|   |   |   |   |--->   |   |   |   |    12. TransactWriteItems, ONE transaction:
|   |   |   |   |   |   |   |   |   |          counter +1     cond: holders < max_holders
|   |   |   |   |   |   |   |   |   |          Put holder#{execution_arn}
|   |   |   |   |   |   |   |   |   |          Put lease#{execution_arn}   (reverse pointer)
|   |   |   |   <---|   |   |   |   |    13. OK  -> try_acquire returns "acquired"
|   |   |   <---|   |   |   |   |   |    14. SendTaskSuccess — the only thing that ends the wait
|   |   |   |----------->   |   |   |    15. Run inference
|   |   |   |   |   <---|   |   |   |    16. refresh_holder(arn) ONCE on entry, extends the TTL
|   |   |   |   |   |   |------->   |    17. Converse, record 1..N, SERIAL (one model, one call)
|   |   |   |   |   |   <-------|   |    18. one response per record
|   |   |   |   |   |   *   |   |   |    19. checkpoint _progress.json per record; at < 60 s
|   |   |   |   |   |   |   |   |   |        remaining, raise LambdaResourceTimeoutException
|   |   |   |   |   |   |   |   |   |        -> ASL Retry, 10 attempts, resumes from the
|   |   |   |   |   |   |   |   |   |        checkpoint. The lease is NOT released between.
|   |   |   <-----------|   |   |   |    20. response.jsonl + per-record status
|   |   |   |--------------->   |   |    21. Release lease
|   |   |   |   |   <-------|   |   |    22. Delete holder#{arn} and lease#{arn}
|   |   |   |   |   |   |   |   |   |        ** no decrement here — see below **
|   |   |   <---------------|   |   |    23. returns the lease status; NEVER raises
|   |   |   |   |   |   |   |   |   |
|   |   |   |   |   |--------------->    24. stream: REMOVE on holder#  (FilterCriteria matched)
|   |   |   |   |   <---------------|    25. decrement_once(eventID): dedup marker + counter -1
|   |   |   |   |   <---------------|    26. drain_pending(scope), UNCONDITIONAL -> queue empty,
|   |   |   |   |   |   |   |   |   |        0 waiters admitted
|   |   |   |   |   |   |   |   |   |
|   |   |   *   |   |   |   |   |   |    27. Summarize (K): summary.json, partial success IS success
|   |   <---|   |   |   |   |   |   |    28. child SUCCEEDED, $.sync_result
|   |   *   |   |   |   |   |   |   |    29. Usage tracking (L): recurses the execution TREE, parent+child(+grandchild)
|   |   |   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |   |   |    30. (M) terminal rule fires -> (N): list_leases(arn) -> []
|   |   |   |   |   |   |   |   |   |        -> {"released": 0}   <- the expected common case
```

Steps 10–11 are the burst fix: the ratchet reads before it writes, because the steady
state is "the cap is already correct" and an unconditional write put every execution in
a burst onto the hottest item in the table to change nothing. Step 22 not decrementing
is invariant 1 — the counter belongs to the stream, so steps 25–26 are identical whether
the holder vanished by release, by abort, or by TTL. Step 30 is not redundant with step
22: it covers a Lambda that released and then died before returning, and it is the same
code path for an `ABORTED` or `TIMED_OUT` execution that never reached `Release lease`
at all.

### Case 2 — sync, cap full (park, then resume)

Identical to case 1 through step 9. This execution is the 4th against a cap of 3.

```
(C) sync SM  (D) acquire  (E) quota table  (J) stream handler  (F) run-inference

(C) (D) (E) (J) (F)
|   |   |   |   |
|--->   |   |   |    1. Acquire lease .waitForTaskToken  (4th vs a cap of 3)
|   |--->   |   |    2. ratchet: read, already 3, NO WRITE
|   |--->   |   |    3. TransactWriteItems: counter +1 cond holders < max
|   <---|   |   |    4. TransactionCanceledException
|   |   |   |   |          reasons[0] = ConditionalCheckFailed  (cap full)
|   |   |   |   |          reasons[1] = None                   (not re-entrant)
|   |   |   |   |        -> try_acquire returns ""  (saturated, not an error)
|   |--->   |   |    5. store_pending: Put pending#{scope} / sha256(token)
|   |   |   |   |          task_token, execution_arn, requested_max,
|   |   |   |   |          created_at in MILLIseconds, ttl = 24 h
|   *   |   |   |    6. return {"acquired": false, "queued": true}
|   |   |   |   |       ** NO SendTaskSuccess. The execution stays parked on its
|   |   |   |   |          token for up to TimeoutSeconds: 86400. Nothing polls. **
|   |   |   |   |
|   |   |   |   |       ... an unrelated holder finishes and deletes its own row ...
|   |   |   |   |
|   |   |--->   |    7. REMOVE on holder#  (eventName=REMOVE, pk prefix
|   |   |   |   |       semaphore#, sk prefix holder#)
|   |   <---|   |    8. decrement_once(eventID), ONE transaction:
|   |   |   |   |          Put stream-dedup/{eventID}  cond: not exists
|   |   |   |   |          counter -1                  cond: holders > 0
|   |   <---|   |    9. drain_pending(scope), UNCONDITIONAL:
|   |   |   |   |          Query pending#{scope}, sort by created_at, oldest first
|   |   <---|   |    10. try_acquire with the WAITER's execution_arn -> acquired
<-----------|   |    11. SendTaskSuccess(the waiter's token)
|   |   <---|   |    12. Delete the pending item; loop while slots remain
|--------------->    13. resumed -> Run inference
|   |   |   |   |        ... identical to case 1 from step 15: refresh, Converse,
|   |   |   |   |            release, stream decrement, drain, Summarize, usage ...
```

Step 5's sort key is `sha256(task_token)`, so an ASL retry of the same acquire
overwrites its own pending item instead of queueing twice. `created_at` is in
**milliseconds** because a burst of waiters inside one second would otherwise have
arbitrary FIFO order. Step 9 runs **unconditionally**, not only when step 8 decremented:
a redelivered record's decrement dedups to `False`, but the crashed attempt may have
died before draining — and if the removed holder was the last one, no future `REMOVE`
will ever fire and every waiter sits until its token times out a day later. Step 10
acquires on the **waiter's** ARN, not the stream handler's, so the holder row names the
execution that will actually hold it. In step 11 a dead token
(`TaskDoesNotExist`/`TaskTimedOut`) releases the slot only if this call *freshly*
acquired it — a `"reentrant"` hold means a previous drain already admitted this waiter
and died before step 12, and that execution is live.

### Case 3 — batch, job created immediately

No lease anywhere on this path: the in-progress-job limit is account-level and
service-enforced, so a local cap would serialize submissions Bedrock is willing to
accept in parallel.

```
(B) orchestrator SM  (O) batch SM  (P) submit-batch  (Q) token table  (I) Bedrock
(R) batch-status  (S) notify  (T) collect  (U) drain

(B) (O) (P) (Q) (I) (R) (S) (T) (U)
|   |   |   |   |   |   |   |   |
*   |   |   |   |   |   |   |   |    1. Choose mode -> batch (>= 100 records, model supports it)
|--->   |   |   |   |   |   |   |    2. startExecution.sync:2
|   |--->   |   |   |   |   |   |    3. Submit batch .waitForTaskToken
|   |   *   |   |   |   |   |   |    4. render request.jsonl through the SAME
|   |   |   |   |   |   |   |   |       build_converse_params the sync loop uses, then
|   |   |   |   |   |   |   |   |       to_batch_record, then pad up to 100 lines -> S3
|   |   |--->   |   |   |   |   |    5. Put row status=SUBMITTING with task_token and
|   |   |   |   |   |   |   |   |       model_params   ** BEFORE the API call **
|   |   |------->   |   |   |   |    6. CreateModelInvocationJob, job_name = workflow_id
|   |   <-------|   |   |   |   |    7. job_arn
|   |   |--->   |   |   |   |   |    8. status=SUBMITTED, job_arn
|   <---|   |   |   |   |   |   |    9. return normally — the task STAYS PARKED (24 h)
|   |   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |   |       ... later, Bedrock finishes the job ...
|   |   |   |   |   |   |   |   |
|   |   |   |   |--->   |   |   |    10. Batch Inference Job State Change (BatchTokenRule,
|   |   |   |   |   |   |   |   |        prefix-scoped to OUR account's job arns)
|   |   |   <-------|   |   |   |    11. find_by_job_arn via the job_arn-index GSI — the event
|   |   |   |   |   |   |   |   |        carries only the arn, never our workflow_id
|   |   |   <-------|   |   |   |    12. mark_terminal(COMPLETED), cond attribute_exists AND
|   |   |   |   |   |   |   |   |        status not already terminal.  ** no SendTask* here **
|   |   |   |----------->   |   |    13. stream MODIFY -> status COMPLETED
|   |   |   |   |   |   *   |   |    14. OLD image not terminal + NEW image terminal = a
|   |   |   |   |   |   |   |   |        TRANSITION, not a state -> this record is the one to send
|   <-------------------|   |   |    15. SendTaskSuccess{status, job_arn}
|   |   |   <-----------|   |   |    16. delete_row (best effort)
|   |----------------------->   |    17. resumed with $.batch_result -> Collect batch output
|   |   |   |   |   |   |   *   |    18. read the job output, map recordId -> record_id BY ID,
|   |   |   |   |   |   |   |   |        never by position (pads and reordering)
|   |   |   |   |   |   |   *   |    18a. every record answered -> write response.jsonl and
|   |   |   |   |   |   |   |   |         RETURN. If any record is missing or errored and
|   |   |   |   |   |   |   |   |         failed <= REPAIR_MAX_FRACTION x total, instead:
|   |   |   |   |   |   |   |   |         seed _progress.json with the SUCCESSES, then RAISE
|   |   |   |   |   |   |   |   |         BatchRecordsIncompleteException -> the Catch runs
|   |   |   |   |   |   |   |   |         the SYNC child as {workflow_id}-repair (case 1),
|   |   |   |   |   |   |   |   |         which infers only the gaps and rewrites the file.
|   |   |   |   |   |   |   |   |         Repair failure -> Catch States.ALL -> step 19 anyway
|   *   |   |   |   |   |   |   |    19. Summarize (K) -> summary.json
<---|   |   |   |   |   |   |   |    20. child SUCCEEDED
*   |   |   |   |   |   |   |   |    21. Usage tracking (L), recursing into the repair
|   |   |   |   |   |   |   |   |        grandchild if there was one
|   |   |   |   |   |   |   |   |
|   |   |   |------------------->    22. stream REMOVE (from step 16)
|   |   |   |   |   |   |   |   *    23. drain the model named in the OLD image
```

Step 5 before step 6 is the load-bearing order: reversed, a Lambda death between them
leaves a running job whose completion event finds no row and no token, so the caller
parks for its full 24 h with the work already paid for and the answer sitting in S3.
Step 12 writing a status rather than sending a token is why the notifier is a **stream**
consumer — a stream handler failure is retried by DynamoDB, whereas a failed inline send
would need the next completion event to recover, and if that was the last job in flight,
nothing would ever arrive. Step 15 uses `send_task_failure` for a `FAILED` job rather
than a success payload carrying a failure flag. Step 18 maps by `record_id` because
Bedrock does not guarantee output order and interleaves the padding records from step 4.
Step 18a is **write-then-raise**, in that order: both S3 objects the repair depends on
are durable before the exception, so the raise is a routing signal over committed state
rather than a lost result — raise first and the repair has no checkpoint to resume from
and the run has no `response.jsonl`.

### Case 4 — batch, throttled, queued, drained later

Diverges from case 3 at step 6.

```
(P) submit-batch  (I) Bedrock  (Q) token table  (U) drain  (S) notify  (O) batch SM

(P) (I) (Q) (U) (S) (O)
|   |   |   |   |   |
|------->   |   |   |    1. Put row status=SUBMITTING  (as case 3 step 5)
|--->   |   |   |   |    2. CreateModelInvocationJob
<---|   |   |   |   |    3. ServiceQuotaExceededException — one of the 7
|   |   |   |   |   |       BATCH_QUOTA_EXCEPTIONS in lib/bedrock.py
|------->   |   |   |    4. mark_pending(reason): status=PENDING
|------------------->    5. return {"status": "PENDING"} — NOT an error. The task
|   |   |   |   |   |       stays parked on the SAME token; a drain owns it now.
|   |   |   |   |   |
|   |   |   |   |   |       ** three independent triggers can drain it: **
|   |   |--->   |   |       (a) the token table's own stream — REMOVE on any batch#
|   |   |   |   |   |           row, or MODIFY to COMPLETED/FAILED. NOT this PENDING
|   |   |   |   |   |           write: it drains when a SIBLING job of ours ends,
|   |   |   |   |   |           and only for the model named in the image.
|   |------->   |   |       (b) BatchCapacityRule: ANY aws.bedrock job completion in
|   |   |   |   |   |           the account, including a FOREIGN stack's. Deliberately
|   |   |   |   |   |           UNSCOPED -> models_with_pending() (Scan)
|   |   |   *   |   |       (c) DrainScheduleRule, rate(6 hours), payload
|   |   |   |   |   |           {"operation": "Drain"}, no image
|   |   |   |   |   |           -> models_with_pending()
|   |   |   |   |   |
|   |   <---|   |   |    6. query_pending(model), OLDEST FIRST by created_at — the
|   |   |   |   |   |       sort key is a uuid5, so key order would starve whichever
|   |   |   |   |   |       workflow_id happens to hash low
|   <-------|   |   |    7. CreateModelInvocationJob from the row's own model_params
|   |   |   |   |   |       — no need to re-read the plan or the request from S3
|   |   <---|   |   |    8. status=SUBMITTED, job_arn
|   |   |   |   |   |       ** the drain still does not call SendTask* — but that is
|   |   |   |   |   |          a CODE convention since the tier collapse, not an IAM
|   |   |   |   |   |          wall: the one workflow role holds both grants **
|   |   |   |   |   |       stop on the FIRST quota wall; the rest stays PENDING
|   |   |   |   |   |
|   |   |   |   |   |       ... from here identical to case 3 step 10: completion
|   |   |   |   |   |           event -> (R) BatchStatus -> terminal row -> stream ->
|   |   |   |   |   |           (S) Notify -> SendTaskSuccess -> Collect -> Summarize
```

Trigger (b) is unscoped on purpose and the comment in the template says not to add a
filter: the job limit is per **account** while this stack is not a singleton, so if
another stack fills the limit, our `PENDING` rows would drain only when one of *our own*
jobs completes — and we have nothing in flight, so no event ever arrives. Deadlock
bounded only by TTL. The inverse asymmetry is `BatchTokenRule`, which *is*
prefix-scoped, because a foreign completion is real news about capacity but never news
about one of our tokens. In step 7 a permanent failure (not quota-shaped) is
`mark_terminal(FAILED)` rather than left `PENDING`: the stream then tells the notifier,
which fails the parked caller with the reason, instead of the row being retried forever
by every future drain until the token times out.

### How the stream handler processes pending items

One function, `(J)`, subscribed to the quota table with `FilterCriteria` narrowing it to
`eventName=REMOVE` AND `pk` prefix `semaphore#` AND `sk` prefix `holder#`. Per record:

1. **Re-check both key prefixes in code** and `continue` on a mismatch rather than
   raising. The record this guard exists for is a `stream-dedup` marker's own TTL
   expiry, which also arrives as a `REMOVE` and must never read as a released holder. A
   raise would fail the whole batch and the stream would redeliver it forever.
2. **Take the scope from the record's own keys**, never from an event field — the keys
   are the only source that cannot disagree with the item that was actually removed.
3. **`decrement_once(service, resource, eventID)`.** Dedup marker `Put` (conditional on
   non-existence) and `counter - 1` (conditional on `holders > 0`) in **one**
   transaction, so a redelivered record returns `False` instead of double-decrementing.
   The `holders > 0` guard is what stops a counter TTL reset whose holders had not yet
   been reclaimed from driving the counter negative and permanently inflating the cap.
4. **`drain_pending` once per distinct scope, unconditionally** — see case 2 step 9 for
   why not only when the decrement succeeded. Per scope: `Query pending#{scope}` (its
   own partition, not a sort key under the semaphore partition — a trap a load harness
   fell into once), sort by `created_at`, then for each waiter `try_acquire` →
   `SendTaskSuccess` → `delete` the pending item, stopping the moment `try_acquire`
   returns falsy because there is no free slot.
5. **The drain is per scope, not global.** This table holds one cap per model, so a
   single argless drain would only ever service one of them.
6. **A drain failure fails the stream record**, which DynamoDB redelivers; the
   redelivered decrement dedups to `False` while the drain runs again. That is the
   intended shape — an idempotent decrement in front of a retryable drain. A transient
   `SendTaskSuccess` failure re-raises **without** deleting the pending item, so the
   retry admits that waiter; deleting would strand a live execution on its token until
   its acquire state's own timeout.

### Legend

| | Resource | What it does here |
|---|---|---|
| (Z) | the calling agent / consumer | Invokes the gateway tool; never talks to the state machines directly |
| (A) | Lambda `{prefix}-bedrock-inference-gateway-tool` | AgentCore dispatcher. Computes `workflow_id` by **importing** `lib/identity.py` (which delegates to the shared `workflow_identity.py`), writes `input_params.json` + `request.jsonl`, and `StartExecution` **named** `workflow_id` — so a resubmit hits `ExecutionAlreadyExists` and is **absorbed into a handle** (status + `message`, never results) |
| (B) | SFN `{prefix}-bedrock-inference` | Orchestrator. Plan → route → usage tracking. Runs each child with `startExecution.sync:2` |
| (C) | SFN `{prefix}-bedrock-inference-sync` | Sync child. `Acquire lease` → `Run inference` → `Release lease` → `Summarize`. **No `Catch` anywhere** |
| (D) | Lambda `{prefix}-bedrock-inference-2-cores`, op `StateAcquireLease` | `try_acquire` on the quota table; on success `SendTaskSuccess`, on saturation `store_pending` and stay silent |
| (E) | DynamoDB `{prefix}-sync-infer-quotas` | The semaphore. `counter` / `holder#{arn}` / `lease#{arn}` / `pending#{scope}` / `stream-dedup`, plus the stream that drives (J) |
| (F) | Lambda `{prefix}-bedrock-inference-1-core`, op `StateRunInference` | `refresh_holder` once on entry, then N serial Converse calls, checkpointing `_progress.json` per record and yielding with `LambdaResourceTimeoutException` when < 60 s remain |
| (G) | Lambda `…-2-cores`, op `StateReleaseLease` | Deletes the holder row and its pointer. **Never raises, never decrements** |
| (H) | Lambda `…-2-cores`, op `StateValidateAndPlan` | Recomputes `workflow_id` and compares, resolves the model chain, emits `quota{}` and both `batch_geo` (reporting) and `batch_region` (the boto3 client) |
| (I) | Amazon Bedrock | `Converse` on the sync path; `CreateModelInvocationJob` + the async job on the batch path |
| (J) | Lambda `…-2-cores`, `STREAM_HANDLER=semaphore` | The quota table's stream handler: exactly-once decrement, then drain per scope. See the section above |
| (K) | Lambda `…-2-cores`, op `StateSummarize` | Writes `summary.json` **first** and raises second, so per-record reasons survive an all-failed run. Called from **both** children |
| (L) | Lambda `…-2-cores`, op `StateUsageTracking` | Walks the whole execution tree (parent, child, grandchild) for cost |
| (M) | EventBridge `{prefix}-bedrock-inference-quota-terminal` | `aws.states` terminal statuses (SUCCEEDED/FAILED/ABORTED/TIMED_OUT) on **all three** machines |
| (N) | Lambda `…-2-cores`, `source=aws.states` branch | `list_leases(arn)` via the `lease#` reverse pointer (the event carries only the ARN), then **delete only** — the stream owns the decrement |
| (O) | SFN `{prefix}-bedrock-inference-batch` | Batch child, **four** states: `Submit batch` → `Collect batch output` → `Summarize`, plus `Run repair inference` reachable **only** by the `BatchRecordsIncompleteException` `Catch` and itself a `.sync:2` of (C). Takes no lease of its own; the repair tail takes (C)'s |
| (P) | Lambda `{prefix}-bedrock-inference-2-cores`, op `StateSubmitBatch` | Renders the batch input, writes the row `SUBMITTING` **before** calling Bedrock, then `SUBMITTED` or `PENDING` |
| (Q) | DynamoDB `{prefix}-batch-infer-tokens` | One row per job, `pk=batch#{model_id}` / `sk=workflow_id`, GSI on `job_arn`, `NEW_AND_OLD_IMAGES` stream feeding both (S) and (U) |
| (R) | Lambda `{prefix}-bedrock-inference-half-core`, `aws.bedrock` + `STREAM_HANDLER=notify` | Prefix-scoped `aws.bedrock` handler. GSI lookup by arn → `mark_terminal`. Calls neither `SendTask*` nor Bedrock — a code convention now, not an IAM fact |
| (S) | Lambda `{prefix}-bedrock-inference-half-core`, `STREAM_HANDLER=notify` | Token-table stream, INSERT/MODIFY to a terminal status. Redeems the parked batch token. Shares the half-core tier with (R), which is exactly why the marker matters |
| (T) | Lambda `…-2-cores`, op `StateCollectBatch` | Reads the job output, maps `recordId` → `record_id` **by id**, writes `response.jsonl` through the same builders the sync loop uses |
| (U) | Lambda `{prefix}-bedrock-inference-1-core`, `STREAM_HANDLER=drain` | Submits `PENDING` rows oldest-first, stopping on the first quota wall. Calls no Step Functions API; shares the 1-core tier with (F) |

**Every lifeline from (D) to (U) is the same container image and the same IAM role** —
they differ only in memory tier and in which router branch the event shape picks. `(D)`,
`(G)`, `(H)`, `(J)`, `(K)`, `(L)`, `(N)`, `(P)` and `(T)` are all
`{prefix}-bedrock-inference-2-cores`; `(R)` and `(S)` are both `-half-core`; `(F)` and
`(U)` are both `-1-core`.

**Read the "no Bedrock" / "no Step Functions" notes above as statements about the CODE,
not about permissions.** They used to be IAM facts — six functions, six roles,
`states:SendTask*` confined to the roles that could not call a model. That is gone (see
[One image, seven memory tiers, one role](#one-image-seven-memory-tiers-one-role)), so a
defect in the notifier could now do what a defect in the notifier previously could not.

## Layout

| Path | Purpose |
|------|---------|
| [lambda_function.py](lambda_function.py) | Router. **Nine** `operation` values, plus the DDB-stream, `aws.states` and `aws.bedrock` branches |
| [lib/identity.py](lib/identity.py) | `workflow_id`. Imported by the gateway tool, never reimplemented |
| [lib/input_schema.py](lib/input_schema.py) | Every constant and every rejection. One parser, shared by the gateway tool and the plan state |
| [lib/bedrock.py](lib/bedrock.py) | Registry, `ModelSpec`, param shaping, `converse`, the chain walk, cost |
| [lib/batch_client.py](lib/batch_client.py) | Control plane, padding, output reading |
| [lib/token_table.py](lib/token_table.py) | The batch row. Hand-rolled attr marshalling, no `lib-dynamodb` |
| [lib/response.py](lib/response.py) | `success_line` / `failure_line` / `build_summary` — the one place the response shape is decided |
| [lib/s3_utils.py](lib/s3_utils.py) | JSON/JSONL helpers. `get_json_if_exists` is the checkpoint read |
| [lib/semaphore_client.py](lib/semaphore_client.py) | The counting semaphore. Merged in from the standalone quota stack; still imports nothing from this project, so it stays vendorable |
| [states/](states/) | The state handlers: `validate_plan`, `usage_tracking`, `acquire_lease`, `run_inference`, `release_lease`, `submit_batch`, `collect_batch`, `summarize`, `drain`, `notify`, `batch_status`, `handle_semaphore_release`, `handle_execution_terminal`, `seed_model_registry` |
| [gateway_tool/](gateway_tool/) | AgentCore dispatcher: `run_inference`, `get_job_status`, `list_models`, `get_model`, `describe_tool`. Writes the request objects |
| [models/](models/) | `availablemodels_master.json` — the seeded model registry data |
| [definitions.json](definitions.json) | The **orchestrator** ASL — a readability copy of the inline `DefinitionString`; the template is what deploys |
| [definitions_sync.json](definitions_sync.json) | The sync child, same arrangement |
| [definitions_batch.json](definitions_batch.json) | The batch child, same arrangement |

## One image, seven memory tiers, one role

**Conformance with the sibling workflows, and a real reversal.** Every other workflow in
this repo shapes its Lambdas as a `HalfCore … SixCores` ladder over one container image
and one IAM role. This stack was the exception: six purpose-named functions on a flat
zip, each with its own role. It is not any more.

| Tier | Memory | `STREAM_HANDLER` | Reached by | Replaced |
|---|---|---|---|---|
| `half-core` | 896 | `notify` | `NotifyStreamMapping`, `BatchTokenRule` (prefix-scoped `aws.bedrock`) | notify (256), batch-status (256) |
| `1-core` | 1792 | `drain` | `DrainStreamMapping`, `DrainScheduleRule`, `BatchCapacityRule` (unscoped), `RunInferenceLambdaArn` | drain (512), run-inference (1024) |
| `2-cores` | 3584 | `semaphore` | `SemaphoreStreamMapping`, `QuotaTerminalRule` (`aws.states`), `InferenceLambdaArn`, `SubmitBatchLambdaArn` | workflow (1024), submit-batch (2048) |
| `3-cores` … `6-cores` | 5376–10240 | none | declared, unrouted | — |

**The packing was forced, not chosen.** There are three markers, one per tier, and
`notify` was needed by two of the old functions while `semaphore` and `drain` were
needed by one each — so given six functions and three markers there is exactly one legal
assignment. That is why the marker table is the thing to read before moving a state.
**Every routed tier went UP and none went down**, which is the direction that matters:
896 would have been an OOM waiting to happen for `collect_batch` at its old 1024.

**A state landing on a marked tier is harmless.** The router picks its branch by event
shape — `operation` for an SFN task, `Records[]` for a stream, `source` for EventBridge
— and reads `STREAM_HANDLER` *only* for a DynamoDB record or an `aws.bedrock` event,
neither of which carries `operation`. The four unrouted tiers set no marker at all, and
an unset marker raises.

Only three Lambda ARNs appear in the three ASLs (`InferenceLambdaArn`,
`RunInferenceLambdaArn`, `SubmitBatchLambdaArn`), so re-pointing a state at a different
tier is a one-line change to a `DefinitionSubstitutions` entry and the matching
`lambda:InvokeFunction` grant. **Every tier carries the whole environment**,
deliberately: a per-tier subset would make that move two changes with `KeyError` as the
failure mode for forgetting the second.

### Three roles, and the compartments that are gone

| Role | Holds | Notably lacks |
|------|-------|---------------|
| `BedrockInferenceLambdaRole` | everything the workflow does: S3, both tables incl. `Scan`, both streams, `bedrock:InvokeModel` **and** the batch control plane, `PassRole`, **`SendTaskSuccess`/`SendTaskFailure`**, `states:DescribeExecution`/`GetExecutionHistory`, `lambda:GetFunctionConfiguration` | nothing this workflow needs — that is the point and the cost |
| `GatewayToolRole` | `StartExecution`, `DescribeExecution`, S3 | Bedrock, both tables, `SendTask*` |
| `SeedModelRegistryRole` | `Get`/`PutObject` on the ONE registry key | Bedrock, both tables, `SendTask*` |

**It used to be six roles and the split was the design.** §6.3's asymmetry was enforced
by IAM: `RunInferenceRole` held `bedrock:InvokeModel` and no `CreateModelInvocationJob`,
so the sync loop **could not** submit a batch job inside a held lease; `SubmitBatchRole`
held the control plane and no `InvokeModel`; `NotifyRole` held `states:SendTask*` and no
Bedrock while `DrainRole` held Bedrock and no Step Functions, so a drained submit
structurally could not notify. Every one of those pairwise impossibilities is now a
**code convention that IAM will not catch**, and six compartmentalization tests were
deleted with them.

The trade was made knowingly: the tier ladder exists so a state can change memory size
without a template change, and a per-state role makes that move a permissions change too
— which is exactly the coupling the ladder removes. `GatewayToolRole` and
`SeedModelRegistryRole` stayed split because collapsing *those* would put Bedrock and
both tables behind an internet-facing dispatcher; the siblings keep the same split.

**What survives, and survives only in code:** exactly one path decrements the semaphore
counter — `states/handle_semaphore_release`, on the quota table's stream, deduped per
`eventID`. Every holder-disappearance path reaches it by DELETING a holder row, never by
decrementing. `test_the_workflow_role_writes_the_counter_but_only_one_handler_does`
walks the whole tree and pins the caller set to the definition plus that one handler; it
is now the entire guarantee rather than a belt beside IAM's braces.
`test_exactly_one_role_sends_task_results_and_it_can_also_call_bedrock` asserts the
*co-residency*, so re-splitting the role fails a test instead of leaving a stale claim
of isolation.

Three machine roles, each scoped to only the **tiers** its own ASL names —
`StateMachineRole` → 2-cores; `SyncStateMachineRole` → 2-cores + 1-core;
`BatchStateMachineRole` → 2-cores. That grant used to read as capability (the
orchestrator could not reach the only holder of `bedrock:InvokeModel`); with one role,
reaching any tier is reaching Bedrock, so what it buys now is narrower: a state moved to
a tier without granting it fails with `AccessDenied` rather than silently working.
`BatchStateMachineRole` still does **not** get 1-core, because the repair tail starts
the sync child rather than invoking its functions.

**The `StartExecution` invariant is now a graph, and it is asserted as one.** "Only the
orchestrator may start a child" stopped being true the moment the repair became a nested
`.sync:2`, so `test_the_start_graph_is_exactly_this_and_is_acyclic` builds the edge set
**from the IAM policies** and derives the rest: orchestrator → {sync, batch}, batch →
{sync}, sync → {}, depth exactly 3, no cycles. A future grant is then checked by the
rule instead of by a reviewer — which is the point, because a machine that could start
itself is a recursion with a real Bedrock bill.

**`.sync:2` needs FIVE grants, not one.** `states:StartExecution` on the machine arn,
`states:DescribeExecution` + `states:StopExecution` on its execution arn pattern, and
`events:PutRule` / `events:PutTargets` / `events:DescribeRule` on the literal
`StepFunctionsGetEventsForStepFunctionsExecutionRule`. Omit the `events:*` three and the
child runs to **success** while the parent hangs to its own `TimeoutSeconds` — a failure
mode with no error anywhere.

## `STREAM_HANDLER` is what distinguishes three tiers running one image

Three stream consumers across **two** tables, and the marker is what tells them apart:

| Marker | Tier | Table | Handler |
|---|---|---|---|
| `notify` | half-core | batch-infer-tokens | `states/notify` |
| `drain` | 1-core | batch-infer-tokens | `states/drain` |
| `semaphore` | 2-cores | sync-infer-quotas | `states/handle_semaphore_release` |

The half-core tier's value is **`"notify"`, NOT `"drain"`** — that is precisely what
makes the router's `aws.bedrock` branch resolve to `states.batch_status` for the
prefix-scoped token rule. Setting it to `"drain"` would silently turn that rule into a
second capacity sweep and no task token would ever be redeemed. That branch tests for
`"drain"` **explicitly** rather than "not notify", so the 2-cores tier's `"semaphore"`
lands on `batch_status` if a Bedrock event ever reaches it, never on a second sweep.

An unset or unrecognised marker **raises**, rather than defaulting to a branch — which
is why the four unrouted tiers set none: a stream pointed at one of them should fail
loudly rather than pick a handler nobody decided on.

## Gateway tool and callbacks

The gateway dispatcher (`gateway_tool/`) serves five tools — `run_inference`,
`get_job_status`, `list_models`, `get_model`, `describe_tool`. It ships
`gateway_tool/output_contract.json` (labels `response`/`summary`/`usage` — keys must
equal the emitted label set exactly), joins each outputs[] entry's `desc` from it, and
serves it whole as `describe_tool.output_contract`. The absorb `message` is BYTE-PINNED
bundle-wide ("identical request already submitted — this call did not resubmit; …") — do
not reword locally. No submit branch carries `current_state` (the poll owns it); every
poll branch — `PENDING_REDRIVE` included — names its run (`job_id` + `workflow_id`).

**`callback_task_token`** (§11): `run_inference` accepts an optional Step Functions task
token from a consumer workflow's ONE `waitForTaskToken` submit state (the
llm-prompt-driven workflows — conversation-analysis first; the design was worked out
jointly with that workflow). The gateway registers the token in
`${ResourcePrefix}-bi-callback-tokens` BEFORE `StartExecution` (the waker can never fire
first — no describe-and-recheck race exists), and `CallbackCompletionRule` — ONE
EventBridge rule on the ORCHESTRATOR's native status-change events, the same mechanism
`.sync` uses internally — invokes `CallbackWakerLambda`, which completes every
registered token with `SendTaskSuccess` carrying the terminal status: a FAILED verdict
is DATA for the consumer's Choice state, never a mid-machine detonation. A submit that
absorbs into an already-FINISHED run completes the token synchronously in the same call
(the paid-work-reuse path). Both paths are live-verified: a fresh submit wakes its
consumer via EventBridge in seconds with the full handle payload, and an identical
resubmit absorbs into the finished run (same workflow_id, NO new execution) and
completes the token synchronously; the table drains to empty both times.

The token is TRANSPORT ONLY — never in the hash, so a waiting and a non-waiting submit
of identical records are ONE `workflow_id`. With a token present, failures RAISE instead
of returning error dicts: a `waitForTaskToken` caller never reads the return value, so a
returned error would be a silent hang until the consumer's deadman. Notification lives
HERE, once, because the consumer-side alternative multiplies a table+rule+waker by N
workflows and fans every completion out to N rules for N−1 misses. The row-format home
is `gateway_tool/callback_registry.py` — a separate module because the dispatcher is
pinned hashlib-free (the workflow-id discipline); the waker is
`gateway_tool/callback_waker.py`, a flat-zip function riding the gateway zip. IAM:
`SendTaskSuccess` (never `SendTaskFailure`) scoped to this account+region's state
machines (SendTask* accepts machine-ARN scoping; the token's machine is the caller's and
unknowable at deploy time); neither new holder can call Bedrock.

Note the callback-token table name carries the `bi-` prefix to match the sibling
convention; renaming it is a table REPLACEMENT on deploy, safe only with no callback run
in flight.

## Consumer contract

1. **Prompts must already be rendered.** No template resolution here (§10 rule 2).
2. **One prefix holds one request.** The pointer is `{prefix}/input_params.json`;
   `request.jsonl` is *derived*, never accepted as a second uri — two uris admits the
   failure mode where the hash is computed over one request and the work done on the
   other.
3. **`record_id` is yours and must be unique.** `pad_` is reserved: batch padding
   records use it and `read_job_output` drops them, so a real record named `pad_000`
   would vanish from the response with no error anywhere.
4. **`response.jsonl` is in request order**, one line per input record, always the same
   length.
5. **Check `records_failed`, not just the status.** Partial success is success.
6. **One in-flight Converse call per lease.** No `Map` in **any of the three** ASLs — a
   `MaxConcurrency: 4` inside a held span turns a cap of 3 into 12, silently, and the
   only symptom is Bedrock throttling. A test asserts it across all three machines,
   which is also why `usage_tracking` carries an unused `map` kind: adding one would
   show up as an unbilled envelope rather than a silent gap.
7. **Start the ORCHESTRATOR, never a child.** `InferenceStateMachineArn` is the
   orchestrator; the two children are started by it with a `-sync` / `-batch` name
   suffix, and the batch child starts the sync child a third time as `-repair`. They are
   standalone by design (either runs alone given a plan), but a consumer that starts one
   directly gets no usage rollup and bypasses mode selection. **`-repair` is reserved**:
   it is the batch child's own name for its tail, so a consumer that starts a sync
   execution under that name will collide with a repair that has not happened yet.
8. **The terminal rule is already yours if you wrap this one.** This stack owns
   `QuotaTerminalRule`, scoped to its own three `stateMachineArn`s — a wrapper's own
   executions are not on it and do not need to be, since only this stack's machines hold
   leases.

## Stack

[cfnstack.yaml](cfnstack.yaml). All **three** ASLs are **inline** in the template, in
`DefinitionString: |-` blocks — the template is the single source of truth for what
deploys, and there is no injection step. `definitions*.json` are readability copies of
those three blocks; **edit both sides**. `verify_sfn_definition` in the build fails the
build on a drift (compared as parsed JSON) and a unit test compares them as exact
strings, because a copy that diverges silently means the reviewed diff is not what
shipped.

| Token | Value |
|-------|-------|
| `%%VERSION%%` | `.version` — also `TOOL_VERSION` **and the ECR image tag** |
| `%%PKG_BEDROCK_INFERENCE%%` | `bedrock-inference-{VERSION}.zip` — the CodeBuild **source** archive, not a Lambda zip |
| `%%PKG_GATEWAY_TOOL%%` | `bedrock-inference-gateway-tool-{VERSION}.zip` — a real Lambda zip |
| `%%BUCKET_NAME%%` / `%%KEY_PREFIX%%` | deployment bucket basename / `{solution}/{version}` |

Five tokens, all `sed`-substituted. The drift check runs against the **source**
`cfnstack.yaml`, before that pass: `sed` rewrites the tokens everywhere including inside
the ASL bodies, so checking the built copy would read a substituted `%%VERSION%%` as
drift.

**The seven tiers deploy as a container image, and the stack builds it itself** (matching
every sibling workflow). `%%PKG_BEDROCK_INFERENCE%%` still names one S3 zip, but it is
the *whole repo* — `CodeBuildProject` takes it as its S3 source and builds `Dockerfile`
with an **inline** buildspec, and `StartBuild` (`Custom::StartBuildDelayResponse`,
backed by the unnamed `StartBuildLambda`) blocks the stack until the push finishes so no
`Function` resource can reference a tag that is not in ECR yet. ARM: `ARM_CONTAINER` /
`BUILD_GENERAL1_LARGE` / `amazonlinux2-aarch64-standard:3.0`, matching the tiers'
`arm64`.

**The image tag IS the version, and that is what makes an unbumped redeploy cheap:** the
buildspec's `pre_build` asks ECR whether `bedrock-inference-on-aws:{VERSION}` already
exists and skips the `docker build` entirely if it does. So a stack update with no
`.version` bump is a no-op build — and, symmetrically, **a code change with no bump
ships nothing**, because the tag it would push to is already there. Bump `.version` for
every code change; that has always been true for `workflow_id` stability and is now also
true for what is in the image.

**Four functions stay flat zips, and none is a workflow state:** the gateway dispatcher,
`CallbackWakerLambda` (rides the gateway zip, `Handler: callback_waker.lambda_handler`),
`StartBuildLambda` (it builds the image), and `SeedModelRegistryLambda` — which ships
*inside the gateway-tool zip*, the way media-processing's `bucket-check` custom resource
does. No sibling puts a CloudFormation custom resource in its image, and on the image
this one needed `DependsOn: StartBuild`, so the registry could not be seeded until a
docker build finished. It also turned out to be the thing that **blocked the zip→image
migration**: `PackageType` requires replacement and CloudFormation refuses to replace a
*custom-named* resource, so the first `update-stack` rolled back in seconds on `Rename
{prefix}-bedrock-inference-seed-model-registry and update the stack again`. The seven
tiers are new logical ids with new names and migrate cleanly; only the function keeping
its name across the change was stuck. Consequence: **no `ImageConfig.Command`
anywhere**, so all seven tiers run the `Dockerfile`'s one `CMD`.

The `Dockerfile` pins a digest-dated `public.ecr.aws/lambda/python:3.13.*` base and
**installs nothing** — no `pip install`, deliberately. The image's own botocore is the
one that parses `TransactionCanceledException.CancellationReasons`, which is what
`try_acquire` reads to tell "cap full" from a real conflict; vendoring a newer botocore
over it has broken that shape before. `requirements.txt` is therefore **not an install
list — it is the floor the `FROM` tag must clear** (`>=1.42.59`), and
`TestBotocoreIsNewEnoughForTheParametersWeSend` reads both files and fails if they drift
apart.

**That pin is a floor as much as a ceiling, and the floor is the half that bit.** On the
managed runtime AWS keeps botocore moving under you; in an image the `FROM` line freezes
it. The first pinned tag shipped a botocore that predates `modelInvocationType` on
`CreateModelInvocationJob` — the §P5.4 parameter that stopped Bedrock judging this
stack's Converse records malformed. So the zip→image move broke **every** batch submit
with a client-side `Unknown parameter in input: "modelInvocationType"`, stranding rows
in `SUBMITTING`, while sync, the semaphore, cfn-lint, cfn_nag and the whole offline unit
suite stayed green. The unit test for that very parameter asserted against a
`MagicMock`, which accepts any kwarg. It surfaced only because the load harness had just
been fixed to actually exercise the batch arm — a bug found by a test fix, in the window
where nothing else was watching. Two tests now close it: botocore's own `ParamValidator`
run against its own bedrock service model on the exact kwargs `create_job` builds
(skipped, loudly, if the local botocore is below the floor), and the hermetic
`requirements.txt`↔`Dockerfile` cross-check. **When bumping the base tag, check
`botocore.__version__` in the candidate image and confirm `CancellationReasons` is still
on DynamoDB's `TransactionCanceledException`** — the third test in that class pins the
second half.

**Two parameters**, and neither names another stack: `ResourcePrefix` and `ProxyBucket`.
Neither has a default, which is the test for belonging there — a parameter names
something *outside* this stack that only the deployer knows.

Five others were removed (`test_no_request_default_is_a_stack_parameter`,
`test_the_two_deploy_time_constants_are_read_where_a_json_cannot_reach`):

| Was a parameter | Now | Why not a parameter |
|---|---|---|
| `DefaultModelChain` | `lib/input_schema.py` `DEFAULT_MODEL_CHAIN` | a request default |
| `MaxConcurrentDefault` | `lib/input_schema.py` `MAX_CONCURRENT_DEFAULT` | a request default |
| `BatchMinSavings` | `lib/input_schema.py` `BATCH_MIN_SAVINGS` | a request default |
| `ModelRegistryKey` | `Mappings.ModelRegistry.Object.Key` | interpolated into an IAM resource ARN |
| `DrainScheduleExpression` | `Mappings.Drain.Schedule.Expression` | an `AWS::Events::Rule` property |

The first three were **request defaults, and the code already carried every one**
(`os.environ.get(..., "3")`, `..., "0.25"`) — so parameter and constant were two
spellings of one number, and only the parameter could drift per-stack with nothing in
the tree to show it. `UNPRICED_BATCH_MIN_TOKENS` was already a bare constant with no
parameter; that was the pattern to follow.

`DefaultModelChain` is the one that **proves** the cost rather than merely risking it.
The gateway writes `model_chain` into the request, every key in that request is inside
`workflow_id`, and the template's shipped default differed from the value a deployed
stack ran — so every chain-less request against that stack hashed differently from the
same request against the tree: silent re-identification of finished work. A
`workflow_config/` JSON would not have fixed it; an operator edit re-identifies work
just as silently. As a versioned constant the value moves only with a `.version` bump,
which is when a hash change is expected and reviewable.

The shipped default is a **bare** id, no geo prefix: `resolve_model` strips the prefix
and `resolve_runtime_model_id` re-derives routing from the registry entry's
`inference_region`, so a geo prefix on this default would be inert on every deployment —
decoration that reads like residency control.

The last two are read by CloudFormation *itself* while building a resource, so neither
can live in `input_schema.py` or a runtime JSON. An S3 grant must name its object at
deploy time; a key read at runtime could not be scoped at all, which inverts the point
of the grant. Both the grant and the functions' `MODEL_REGISTRY_KEY` come from the one
mapping entry, so a function cannot read key A while its role is scoped to key B.

**DynamoDB posture:** all three tables hold rebuildable derived state (holders, queue
rows, callback tokens), so default SSE and no `DeletionPolicy: Retain` are deliberate —
there is nothing in them worth keeping past the stack.

**Seven outputs** (`InferenceStateMachineArn`, `InferenceStateMachineName`,
`GatewayToolLambdaArn`, `SyncInferQuotasTableName`, `BatchInferTokensTableName`,
`CallbackTokensTableName`, `BedrockBatchServiceRoleArn`) — **no `Export`s**,
deliberately: an Export cannot be changed or deleted while any importer exists, which
would weld every consumer stack's lifecycle to this one. The two **children are
deliberately not output**: publishing their ARNs invites a consumer to start one
directly.

## Build and test

```bash
cd bedrock-inference-on-aws
bash ./build-deploy-s3-dist.sh --build-only            # no AWS credentials needed
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--single-region]

python3 -m unittest discover -s test -p 'test_*.py'    # 1 expected skip on an old local botocore

# NOT run by the build script — run them by hand after --build-only:
cfn-lint     dist/global-s3-assets/bedrock-inference-stack.template   # 0 errors
cfn_nag_scan --input-path dist/global-s3-assets/bedrock-inference-stack.template

# LIVE, spends real Bedrock money, needs a deployed stack:
RUN_TAG=$(date +%s) python3 test/burst_semaphore.py    # 24 sync + 4 batch at once

# LIVE, every REGISTRY model x every legal mode, with a real image for vision
# models (MODEL_REGISTRY_KEY must name the key the DEPLOYED stack reads):
RUN_TAG=$(date +%s) IMAGE=/path/to/test.png python3 test/model_matrix.py
```

**cfn_nag is 0 failures, and both warning kinds are expected.** `W58` fires on all nine
functions because the role grants `logs:CreateLogStream`/`PutLogEvents` scoped to the
declared groups and **deliberately withholds `logs:CreateLogGroup`** —
`AWSLambdaBasicExecutionRole` would let a function that cannot reach its declared group
silently create an unmanaged one instead of failing visibly. `W76` (policy complexity >
25) is suppressed with its reason in the template: it *is* the tier collapse, since one
role is now the union of six. cfn-lint's one remaining warning, `W1034` on the source
bucket name, only appears under `--build-only`, where `%%BUCKET_NAME%%` is substituted
with the placeholder `BUCKET_NAME_NOT_SET`.

**`test/burst_semaphore.py` is checked in and deliberately not named `test_*.py`**, so
`discover` cannot pick it up. It exists because contention is the one condition a
hermetic double cannot simulate, and it is where a real `TransactionConflictException`
bug was found — the offline suite was fully green at the time. It asserts seven things,
and the three that make it non-vacuous are `cap was actually REACHED` (a run that never
reached the cap proves nothing and reports INCONCLUSIVE rather than passing), `waiters
queued` (a cap that admits everyone was never contended), and `the batch requests
actually went to BATCH` (a token row observed — a silent downgrade to sync passes every
other check). Read its docstring before changing it: **four** separate ways to make it
silently prove nothing are recorded there, all four hit in practice. The fourth: the
harness once sent the **retired** `mode` field, which `validate_plan` warns about and
ignores, so every batch request auto-routed to sync (short records save $0.0000 against
the $0.25 floor) and every check still passed — a downgraded request SUCCEEDS.
`use_batch` now, plus a check that a token row was actually observed. Identical payloads
hash to the same `workflow_id`, so re-running the harness unchanged returns instant hits
on the *previous* run's completed executions over a semaphore nothing touched — hence
`RUN_TAG`.

The build **runs the tests first and refuses to package on failure.** The failures this
suite exists to catch are the ones production hides: a `workflow_id` that is not stable
re-pays for work already done, a checkpoint that is read as absent re-inferences every
completed record, and an `ErrorEquals` naming a service error code is dead retry
configuration that reads as working back-pressure. None of those raise anywhere.

Tests are hermetic and **moto is deliberately not used** — it is not installed here, and
the load-bearing behaviours are specific enough that a general fake would hide them. The
doubles model the three that matter: the `job_arn-index` GSI lookup, `mark_terminal`'s
compound `ConditionExpression`, and the `Scan` behind `models_with_pending()`.
Unmodelled expressions **raise** rather than defaulting to `True`; a silently ignored
guard would make a broken idempotency check look tested.

**Two `_FakeDdb` instances, not one** — `semaphore_client._DDB` and `token_table._DDB` —
so a semaphore key can never collide with a batch row and make a test pass for the wrong
reason. The token-table double carries three behaviours the semaphore's does not, each
forced by that client: a **`scan`**; a `query` that honours `IndexName` and
`FilterExpression`; and an `_apply_update` that resolves `#k{i}` through
`ExpressionAttributeNames`, because `token_table._update` builds a *generic* `SET #k0 =
:v0, …` expression rather than the literal ones the semaphore client uses.

The suite also parses `cfnstack.yaml` and **all three** ASLs to assert cross-file
contracts no runtime test can see: all three stream `FilterCriteria` against the
clients' key and status constants, `SendTask*` confined to the **one** workflow role —
which **can** call Bedrock, and the test says so in its name so the collapse cannot be
mistaken for the old compartmentalization — the seven tiers byte-identical apart from
`MemorySize` and `STREAM_HANDLER`, the memory ladder against the house figures, which
functions are images and which are zips, each machine's `DefinitionSubstitutions`
balanced against its **own** ASL's `${...}` tokens, each machine's role scoped to only
the functions its own ASL names, the terminal rule's three ARNs against the actual set
of machine resources, every `%%TOKEN%%` actually substituted by the build script, and —
trivially, and the reason it is first — **that the template parses at all**.

**Absence is asserted against parsed structure, never raw text.** These artifacts
document their own invariants — the sync child's `Comment` explains at length why
`Release lease on failure` is gone, and the orchestrator's Retry `Comment` explains why
`ExecutionAlreadyExists` is deliberately not retried. A substring search for either
fails while the documentation is accurate and passes once someone deletes it. So:
state-name keys and `Catch[].Next` targets, `ErrorEquals` sets, `json.dumps(self.tpl)`
(comments already stripped by the parser), and `_code()` for the Python capability
tests. Related trap: `"InferenceStateMachine"` is a **suffix** of both child names, so
an `assertNotIn` guarding against self-recursion passes on the recursion it forbids —
use an exact set from `re.findall(r"(\w+StateMachine)\.Arn", …)`.

## Bundle registration

This stack is a bundle child and a media-toolkit-gateway **optional target**
(`BedrockInferenceToolLambdaArn`, condition-gated like shot-segmentation). Three shape
concessions the bundle forced on this repo: `Mappings.Lambda` may hold ONLY {tiers,
Function, GatewayTool}, so the seeder's entry lives under `Mappings.CustomResource` (the
callback waker's entry joined it there); `StartBuildLambda` carries an explicit
`DependsOn: StartBuildLambdaLogGroup` (bundle log-group hygiene) that cfn-lint flags as
W3005 — accepted, the comment on the resource says why; and the build script may contain
exactly ONE literal `members=(` (the bundle regex-extracts the first one), so the
CodeBuild source array is named `image_sources`. The bundle's loaders put this child's
ROOT on sys.path (unique among children: this gateway imports the workflow's own
`lib.identity`/`lib.input_schema`), and its identity-derivation check accepts the `from
lib import identity` delegation. `prompts/bedrock_inference_agent.md` in
media-toolkit-gateway is the agent-facing doc; the target declares five tools
(run_inference, get_job_status, list_models, get_model, describe_tool).

## Gotchas

- **Every state that touches request.jsonl STREAMS; never materialise the record set.**
  `validate_plan` was the LAST materialiser (memory measured at roughly four times the
  request size on a large multi-image request — one pass via `iter_jsonl` +
  `parse_record` feeding the image-limit check, `record_text_chars` accumulation,
  `identity.RecordsDigest`, and the count;
  `test_plan_recomputes_the_same_workflow_id` pins the streamed digest equal to the list
  form, so the hash preimage did not move). `_write_batch_input` reads request.jsonl
  line by line (`iter_jsonl` + `parse_record` with a shared seen-set) and
  multipart-writes the shaped batch input (`put_jsonl_streamed`), so memory is bounded
  by one 16 MB part. The whole-file form held three full copies and a measured
  hundreds-of-MB frame-image request OOM-killed the Lambda before the job existed (found
  by a consumer workflow's corpus sweep). The COLLECT side has the same trap post-spend,
  twice over: each `.out` row echoes its full `modelInput`, so `read_job_output` streams
  the `.out` objects and keeps three fields per row, and `collect_batch` re-reads the
  request via `iter_jsonl`+per-line `parse_record` (never `parse_records`, which would
  materialise every image payload again) and streams response.jsonl out.
  `get_jsonl`/`put_jsonl` remain for the small artifacts.
- **Image bytes: the two transports want OPPOSITE things, and a str reaching a boto3
  blob is re-encoded, not rejected.** A record's image travels as a base64 STRING
  (records are JSON). Batch forwards the string verbatim — its input JSONL is JSON and
  Bedrock decodes there. Sync must hand boto3 RAW BYTES: a str hitting the blob
  parameter is treated as the payload and re-encoded, so the model receives
  base64-of-base64 and answers `detected file MIME type text/plain does not match the
  expected type image/png` (observed live on nova-pro — every vision model failed sync
  while the identical record PASSED via batch; `test/model_matrix.py` is what caught it,
  on its first run). The decode lives in `decode_blob_sources`, inside `converse()`
  AFTER the shared shaping, copy-on-write; `s3Location` sources (the Nova family accepts
  `source.s3Location.uri`) pass through untouched on both paths. `parse_records` also
  pre-validates every image block at plan time — format ∈ {png,jpeg,gif,webp}, exactly
  one of bytes|s3Location, and the first 24 base64 chars must decode to the declared
  format's file magic — so a double-encoded payload or a format/bytes mismatch is named
  before a token is spent (a 24-char head decode; the payload is never re-decoded and s3
  uris are never fetched).
- **A child `.version` equal to an already-deployed BUNDLE version ships stale code on a
  direct child update.** The bundle re-uploads every child's artifacts under the
  BUNDLE's version path with the bundle version substituted into the child template — so
  a child repo later building at that same version writes the SAME S3Key the stack
  already references: the S3 object is overwritten but the template's Code block is
  byte-unchanged, and CloudFormation detects key changes, never content changes.
  Measured live: env vars updated, code silently stale, mismatched CodeSha256 the tell.
  Rule: a direct child update needs a child version that differs from every bundle
  version deployed against that bucket.
- **The region prefix is NOT the geo: `ap-southeast-1` splits to `ap`, and the geo
  vocabulary says `apac`.** The seeder's `Geo` property is
  `!Select [0, !Split ["-", !Ref AWS::Region]]`, which matches the vocabulary for us and
  eu and silently doesn't for APAC — before the `_GEO_ALIASES` fix, an APAC deploy
  failed the whole stack on the seeder's own validator, and a us/eu test bed can never
  catch it. The alias lives in the seeder because `Fn::FindInMap` accepts only
  `Ref`/`FindInMap` as keys, never `Fn::Select`, so a template-side translation would
  mean hand-enumerating every APAC region.
- **The registry lives at `_model_configs/availablemodels.json`, and `_settings/`
  belongs to the prior-generation deployment.** The proxy bucket can be shared with a
  prior-generation stack whose core stack writes its own bare-list
  `availablemodels.json` at `_settings/`. The original design was a deliberate cutover —
  this stack's seeder overwrote that file in place — replaced by separate paths so
  neither stack can clobber the other's. The key exists in FOUR places (the mapping,
  plus offline fallbacks in `lib/bedrock.py`, `lib/input_schema.py`,
  `gateway_tool/lambda_function.py`); a test pins all three fallbacks to the mapping.
- **`ErrorEquals` matches the exception CLASS NAME, not an AWS error code.** A botocore
  `ClientError` arrives at Step Functions as `"ClientError"`, so
  `ErrorEquals: ["ThrottlingException"]` can never match — dead configuration that
  reads, in the console, as working back-pressure. This is why
  `LambdaResourceTimeoutException`, `BedrockBatchQuotaExceeded`, and
  `BedrockInferenceFailed` exist as named classes. Two tests: one asserts every
  non-`States.`/`Lambda.` name is in the raised set, the other that each is a real
  attribute of `lib/custom_exceptions.py`.
- **`$$.Task.Token` and `$$.Execution.Id` — two dollar signs.** `$.Task.Token` is a path
  into the state *input* and resolves to nothing, parking the execution on a token that
  was never issued for the full `TimeoutSeconds`. `execution_arn` is **always**
  `$$.Execution.Id`, never reconstructed from env + uuid: the holder's identity *is* the
  ARN the terminal EventBridge event will carry.
- **`Fn::FindInMap` takes EXACTLY three arguments, and the two-argument form is green
  until CREATE.** `Mappings.ModelRegistry` and `Mappings.Drain` each hold one scalar, so
  the natural spelling is `ModelRegistry: {Key: …}` + `!FindInMap [ModelRegistry, Key]`
  — which parses, passes the drift check, and packages, then fails at deploy. Hence the
  filler middle keys `Object` and `Schedule`, and
  `test_every_findinmap_names_three_levels_that_exist`, which also resolves each lookup
  against `Mappings` so a renamed entry fails offline too (the `OneCore` vs `1Core` trap
  below). **cfn-lint and cfn_nag are NOT run by the build script** — run them by hand on
  `dist/global-s3-assets/`; E1011 is what found this one.
- **A CFN logical id may not start with a digit, so the tier suffixes and the mapping
  keys are spelled differently.** `BedrockInferenceLambda1Core` reads
  `!FindInMap [Lambda, OneCore, Memory]`; `2Cores` reads `TwoCores`. A test carries the
  suffix→key ladder explicitly for this reason.
- **A plain YAML scalar may not END with a colon.**
  `- prefix: !Sub arn:aws:bedrock:${AWS::Region}:${AWS::AccountId}:` unquoted makes the
  **entire template unparseable** — a real defect found here, whose only symptom was a
  CloudFormation error hours after a clean build. Hence `test_the_template_parses_at_all`.
- **`budget_tokens` is never emitted and has no numeric form.** Deprecated on the 4.6
  family and **rejected with a 400** on later families. `thinking` is `"disabled"` or
  `"adaptive"`; a number is a `ValidationException` at parse time.
- **`thinking: "required"` says the FIELD must be present — the entry's
  `thinking_default` says what a quiet request's field CARRIES, and ONLY the registry
  data enforces it.** Fable 5 **rejects** an explicit `{"type": "disabled"}`
  (`"thinking.type.disabled" is not supported for this model` — observed live), and so
  did a 5.5-family entry that kept a stale `thinking_default: disabled` across a model
  id swap: every quiet request 400'd — the incident that retired the id-substring
  capability tuples. The entry's `capabilities` block is now the SOLE request contract,
  seed-time enforced by `_require_capabilities`. Quiet-request shapes: the current 5
  family → `thinking.type=adaptive` + `output_config.effort=low`; the 4.6 family →
  `thinking.type=disabled`, no `output_config` (`effort_default` absent). A caller's
  explicit `effort` beats the default. If a quiet-disabled 5-family entry ever returns,
  that contract lives in its entry, not in code.
- **The effort knob rides a per-entry WIRE (`capabilities.effort_wire`), and the caller
  vocabulary is deliberately the cross-wire intersection.** The GPT-6 family
  (`openai.gpt-6-luna` / `gpt-6.1-sol` / `gpt-6-astra`, Global-CRIS-only, no batch,
  temperature rejected by name) takes `additionalModelRequestFields.reasoning.effort` —
  flat `reasoning_effort` is REJECTED (probed live) — while the Claude 5 family keeps
  `output_config.effort`. Callers may pass only `low/medium/high` (valid on BOTH wires,
  so a mixed-wire fallback chain never 400s on the knob); the wider reasoning-wire
  values (`none`/`xhigh`/`max`) are reachable only as registry `effort_default`s,
  validated per wire at seed time — and `none` is **Luna-only** (Sol/Astra reject it by
  name; their quiet default is `low`, the floor). Luna's quiet default is `none` —
  reasoning off for bulk extraction, because a quiet `medium` billed several times the
  output tokens on a trivial prompt. Image facts, all measured: 100 images/request max,
  single-image payload passes at 22 MiB and fails at 23, and **auto detail drops to LOW
  above ~1,000 32px patches (≈1 Mpx)** — fine structure aliases with no error and a
  *cheaper* bill as the only tell (Bedrock's Converse exposes no detail override; the
  entries' `downscales_above.total_px: 1024000` records the cliff). Probe method:
  token-plateau over generated noise/checker PNGs
  ([test/probe_image_payload.py](test/probe_image_payload.py) — self-contained, runs on
  a remote US box for geo-locked models).
- **`capabilities.image.limits` is ENFORCED at plan time, all three fields, null-skip —
  and the two size fields mean different things.** `max_images` per request,
  `max_image_mb` per inline image, `max_payload_mb` as the record's inline-byte total
  (s3Location images count toward the COUNT, never the bytes — they are not request
  payload). A null means NOT DECLARED, nothing pre-checked — the `get_model` description
  says so. All values measured live: the Anthropic family refuses any single image over
  **5 MiB of base64** (5,242,880 bytes, named in its own error; = exactly 3.75 MiB raw,
  identical across the family; pinned 3.5) with a request total of ~23 MiB (pinned 20);
  pixtral and llama4 bound the REQUEST at ~17.9 MiB (pinned 16) and cap counts at 8 and
  3; GPT-6 totals ~23 (pinned 20); nova took 100 images without refusal (pinned 100 for
  fleet consistency). Registry pins deliberately ride one step conservative of the
  measured pass/fail boundary — **do not change a pin without re-measuring**.
- **Temperature is omitted whenever reasoning is on**, and for the families that reject
  it outright. `_converse_once` recovers from a rejected temperature **once** and
  *records a note* — the reference implementation deletes `temperature`/`topP` before
  every retry regardless of cause, so a run configured at 0.0 silently finishes at the
  model's default and the log says only "RETRY AGAIN".
- **`cache_system_prompt` is opt-in.** Nova rejects an explicit system `cachePoint` with
  a `ValidationException` about conflicting keys, and there is **no prompt cache in
  batch at all** — `to_batch_record` strips it. A rejected batch job is a 24 h park
  before anyone notices.
- **GPT-6 caches IMPLICITLY and its usage itemization is unreliable — the registry opts
  out and the cost math reconstructs.** Measured over live probe ladders: every
  ≥1024-token prefix cache-writes automatically (billed 1.25× input, and this stack's
  record-shaped prefixes got ZERO reads back), and the SAME request sometimes returns
  full `cacheRead/WriteInputTokens` and sometimes NO cache keys, the cached input
  counted only in `totalTokens` while `inputTokens` collapses to a near-zero remainder —
  size-gating refuted, `outputTokens` trustworthy in every call. Three consequences, all
  registry-driven: (1) the GPT-6 entries carry `capabilities.prompt_cache_mode:
  "explicit"`, emitted as `prompt_cache_options` on every request (probed accepted on
  all three over Converse — the model cards say Converse lacks *explicit caching*, but
  the opt-out passes through; fresh prefixes then bill honest `inputTokens`, repeats can
  still be served from cache); (2) `estimate_cost` prices the write term as
  `reconstructed_cache_write` — `max(0, total − input − output − cacheRead)`, every term
  `or 0` — ONLY when the write field carries no value (missing or explicit-null; a
  reported value is trusted), with hidden READS deliberately priced as writes (errs
  high, never low) and a record-line note naming the under-itemization; (3) the entries
  pin cache pricing from the MODEL CARDS, per entry never a formula — the write
  multiplier is shared across the family but the read multiplier differs per entry,
  short-context rates (the long-context input threshold bills higher and nothing here
  sends a request that long). `cost_is_complete` semantics deliberately untouched —
  downstream reads it.
- **`modelInput` IS the Converse request, because `create_job` declares
  `modelInvocationType: "Converse"`** — the job-level parameter that decides which
  dialect Bedrock validates records in (§P5.4). An early all-rejected job
  (`messages.0.content.0.type: Field required`, 0/100, job "Completed") was diagnosed as
  a record-shape defect and "fixed" with a Messages-API translation
  (`_to_anthropic_body`, family-shaped pads, dual-dialect readers) — the actual defect
  was the MISSING PARAMETER: the API defaults to `InvokeModel`. The reference
  implementation passes `'Converse'` on all its LLM batch paths and was never buggy; the
  whole translation layer is deleted, and sync and batch share one request AND response
  dialect. Re-adding a per-family body here would now be the bug it was thought to fix.
- **A batch job validates ALL-OR-NOTHING.** One unscorable record — a pad included —
  converts a padded job into a wholly failed one, which is how the dialect defect
  announced itself: 100 rejected, 99 of them option-free pads.
- **A batch job reports `Completed` with `successRecordCount: 0`.** The job succeeded;
  every record in it failed. Nothing upstream rejects anything, the drain sees a healthy
  `SUBMITTED` row, and the only signal is the manifest — so read `successRecordCount`,
  never the job status.
- **`outputDataConfig.s3OutputDataConfig.s3Uri` must END WITH `/`** — Bedrock's own
  pattern requires it and rejects a bare prefix. Normalised in `create_job`, at the API
  boundary, because there are two callers (`submit_batch` and the drain's resubmit,
  which replays the *persisted* `model_params`) so a per-caller fix leaves the drain to
  rediscover it. The **input** uri is deliberately not normalised: it names a single
  object.
- **The batch job invokes the model under `BedrockBatchServiceRole`, not the workflow
  role.** S3 i/o alone is not enough. Both arn forms are required — a cross-region
  inference profile resolves to a foundation-model arn (with either an empty region or a
  real one) *plus* the account-scoped profile arn. Without the grant the job is
  accepted, validated, scheduled, runs, and fails minutes later with "Customer doesn't
  have permissions to invokeModel".
- **Padding is applied once, after all real records.** 40 records → 100 lines, 60 pads,
  one job. Per-record padding would multiply the floor by the record count.
- **Batch output is listed from `{s3Uri}/{jobId}/`, not the shared parent.** Every
  execution writes under the same `micro-workflow/` root, so listing the parent picks up
  another job's partially-written output.
- **Batch cost is half the on-demand rate** (`BATCH_COST_MULTIPLIER = 0.5`), and
  `estimate_cost` returns **`None`** — not `0.0` — for an unpriced model.
  `summary.bedrock_usage` carries `rounds_unpriced` and `cost_is_complete` so a partial
  figure never reads as a complete one.
- **A failed record still reports the rounds it burned.** A record that failed after
  three throttled attempts cost real money; dropping its rounds under-reports the bill
  for exactly the runs that went worst.
- **A record with no output row is priced at `0.0`, not `None`.** It produced no tokens,
  and unpriced would make a complete cost figure look partial.
- **`release_lease` never raises, but the REASON changed.** It used to be that a raise
  would be caught by `Release lease on failure`'s own `Catch` and replace the original
  error. There is no `Catch` anywhere now — the reason is that every record is already
  answered and `response.jsonl` already written, so raising would turn a successful run
  into a failed one for a slot the terminal rule and the TTL both reclaim anyway. It
  returns `{"released": bool, "detail": str}` and logs `RELEASE FAILED` at ERROR.
- **There are no `Fail` states in any of the three machines, and no `Catch` whose
  `Next` is one.** `Release lease on failure` and `Track usage on failure`/`Failed` are
  all gone, for the same underlying reason: a `Fail`'s `Error`/`Cause` must be **static**
  (a `States.Timeout` carries no cause, and an intrinsic referencing an absent path
  fails the state), so a catch-in-order-to-re-raise can only ever trade a specific error
  for a generic one. Cleanup that genuinely must happen is on the terminal EventBridge
  rule, which also covers ABORTED — a status no `Catch` can see. **The batch child's two
  `Catch`es are not that shape**: one routes *forward* to the repair child and the other
  routes *forward* to `Summarize`, so neither launders an error and both leave the run
  with a real result. That is the checkable version of the invariant, and it is what the
  test asserts.
- **A `Catch` replaces the state's RESULT, not its INPUT — and `OutputPath` is not
  applied on that path.** Both matter for the repair. `Submit batch` uses
  `ResultPath: "$.batch_result"` (a merge, not a replace), so `$.data` — the whole plan
  — survives into `Collect batch output` and therefore into its catch path, which is the
  only reason the repair has something to run on. And `Collect batch output` is a
  `payloadResponseOnly`-style task with `OutputPath: "$.Payload"`; on the caught-error
  path that projection is skipped, because there is no `Payload` in an error object to
  project.
- **A Lambda task retries `Lambda.*`; a `.sync:2` task cannot.** The orchestrator's two
  branch states are SDK integrations, so their ladders name
  `StepFunctions.SdkClientException` / `ThrottlingException` — a `Lambda.ServiceException`
  there is dead configuration. **`States.TaskFailed` is not in them either**, and that
  one is subtle: it is what a genuine child failure raises, the child name is fixed, so
  retrying it re-issues `startExecution` under the same name and gets
  `ExecutionAlreadyExists` — the caller would see a name collision instead of the real
  error. `StepFunctions.ExecutionAlreadyExists` is likewise **not** retried: it is the
  idempotency contract working, not a transient fault. Only transport failures, where no
  execution was created, are safe to retry under a fixed name.
- **`_check_execution_name` warns, never raises.** A human starting an execution from
  the console has a legitimate reason to name it something else; refusing would make the
  workflow undebuggable by hand. From the gateway tool the two are the same string by
  construction, so a mismatch there means the request object changed after submission.
- **`refresh_holder()` returning `False` is best-effort by design**, logged at ERROR
  (`HOLDER LOST mid-hold … the cap is being exceeded`) because it means the cap is being
  exceeded *right now*. The caller is mid-span and cannot usefully abort.
- **Nothing retries `TransactionConflictException` — not botocore, not the ASL.**
  Verified in `botocore/data/_retry.json`: the dynamodb policy retries
  `ReplicatedWriteConflictException` and `TransactionInProgressException`, and the
  string `TransactionConflict` appears nowhere in botocore; the `Acquire lease` Retry
  names only Lambda-level errors. Every writer in `semaphore_client.py` contends on ONE
  item (`semaphore#{service}#{resource} / counter`), so under a burst a conflict is the
  **steady state of a working semaphore**, and unhandled it reaches the caller: under a
  measured burst well past the cap, nearly half the submissions failed outright on an
  unretried conflict. Hence `_conflict_retry` around every counter write. The cap itself
  was never the problem — it reached its limit exactly and never exceeded it.
- **A conflicted transaction and a legitimately-refused one are the SAME exception
  class.** `TransactionCanceledException` for both; the difference is only
  `CancellationReasons[i].Code` — `TransactionConflict` (another writer held the item,
  **nothing was evaluated**) vs `ConditionalCheckFailed` (the condition was evaluated
  and was false). Every caller here reads a cancellation as a meaningful *answer* —
  "saturated", "no holder", "already released", "duplicate stream record" — so a misread
  conflict is a **wrong answer, not a retry**. Worst is `decrement_once`: a
  holder-REMOVE record gets exactly one decrement, and dropping it leaves the counter
  holding a slot with no holder behind it until the counter's own TTL an hour later,
  with nothing in any log. `is_transaction_conflict` is what separates them, and it is
  checked **before** the `!= "TransactionCanceledException"` guard wherever the
  top-level shape also has to be handled — gating on the cancellation code first lets
  `TransactionConflictException` straight past the classifier.
- **Exhausting the conflict budget is not an answer, and each caller's fallback
  differs.** `try_acquire`, `release` and `decrement_once` **raise** (`""` would park a
  waiter on a queue whose only drain trigger is another holder's release — with slots
  free, no release is coming; `False` from `decrement_once` acks a decrement that never
  happened, so the shard must retry it). `refresh_holder` returns the **truthy** value:
  a conflict is no evidence the holder is gone, `False` would fire the caller's HOLDER
  LOST alarm falsely, and raising would fail an execution that legitimately holds its
  slot. The `max_holders` ratchet and the counter-ttl bump **swallow** it — both are
  already no-ops on `ConditionalCheckFailed`.
- **`_raise_max_holders` reads before it writes, and the read is the point.** The steady
  state is "the cap is already correct": every execution in a burst asks for the same
  `max_concurrent`, so the first ratchets and the other N-1 have nothing to do — yet the
  unconditional write made all N contend on the hottest item in the table to change
  nothing. That was the *entire* source of the burst failures. Eventual consistency is
  safe **here specifically** because the ratchet is monotonic and the write is still
  conditionally guarded, so a stale low read leads to a `ConditionalCheckFailed` that is
  already the ignored case. That reasoning does **not** transfer to `holders`, which is
  why nothing else in the module reads before writing.
- **Pending items live under their own partition key** `pending#{scope}`, not as a
  `pending#` sort key under `semaphore#{scope}`. A probe on the semaphore partition
  reports "0 waiters queued" no matter how many are parked — that turned a harness bug
  into a fake product finding once already. (Two more harness lessons, both unit-pinned:
  read the holder rows BEFORE the counter — `try_acquire` moves both in one transaction,
  so counter-behind-rows is impossible by construction, and that read order puts the
  unavoidable non-atomicity in the benign direction; and identical payloads hash to the
  same `workflow_id`, so an unchanged re-run reports instant hits on the previous run's
  executions — hence `RUN_TAG`.)
- **`states/batch_status` returns `{"handled": False}` rather than raising** on a
  foreign arn, a non-terminal status, or a missing arn — EventBridge would retry a
  lookup that can never succeed.
- **`bedrock_control(region)` / `bedrock_runtime(region)` return an *uncached* client
  whenever `region` is passed.** `submit_batch` always passes an explicit region equal
  to the Lambda's own, so it builds a fresh client per call — correct, marginally
  wasteful, and the price of never letting a geo reach a cached client.
- **Table-name helpers are functions, not module globals**, and `_ddb()` / `_bedrock()`
  are lazy. A client built at import time would make an offline suite pass or fail
  according to the developer's shell.
- **The capability tests read `_code()`, not the raw file.** These modules' docstrings
  *discuss* the grants they must not hold (`batch_status.py` says "**No `SendTask*`
  here.**"), so a substring test against the raw source asserts against the prose: it
  fails when the comment is accurate and passes when the comment is deleted.
- **`boto3`/`botocore` are deliberately NOT vendored — `requirements.txt` declares a
  FLOOR the base image must clear, not something to install.** A vendored botocore
  shadows the image's and goes stale. But the floor is real and enforced (`>=1.42.59`),
  because in a container the `FROM` tag is the only thing choosing the botocore version,
  and one too old rejects a needed parameter *client-side*, before the request leaves
  the process.
- **A `MagicMock` client accepts any kwarg; real botocore validates the input shape.**
  So "does the API take this parameter" is not testable with a mock — assert with
  `botocore.validate.ParamValidator` against `get_session().get_service_model(...)`,
  which reads the packaged JSON off disk and needs no network or credentials.

## Open

- **§12.2, the remainder.** The golden paths are done, and a live burst closed the sync
  half of the contention story: the cap, the pending queue, the FIFO drain, and a full
  waiter cohort admitted and released under real load. What is still unreached: **batch
  saturation where the last completion drains the tail** — the one that matters most,
  and the only test of the `BatchCapacityRule`/drain pair under real contention; the
  drain stopping on a genuine quota wall rather than a faked one; the holder TTL
  reclaiming a slot from a Lambda killed mid-hold; and a sync run long enough to
  actually checkpoint and re-enter. The burst's batch jobs never contended — they all
  fit under the account's in-progress limit, which is exactly why batch saturation needs
  a purpose-built test rather than a bigger burst.
- **`REPAIR_MAX_FRACTION` is a flagged design item, not a settled constant.** 0.5 with
  no way to override it per request: `options.batch_repair` is only a boolean opt-out,
  so a caller who wants the tail on a 70%-failed job, or off above 10%, cannot say so.
  The bound is also a *fraction* rather than a dollar figure, which is inconsistent with
  `BATCH_MIN_SAVINGS` — the mode decision was moved to dollars for exactly the reason a
  count rule gets both ends wrong, and "50% of records" has the same defect: 1 of 2 tiny
  records is under the bound while 2000 of 5000 huge ones is not. Deliberately parked,
  not overlooked.
- **The repair tail's own failure path is unverified live.** The `States.ALL` `Catch`
  that degrades a failed repair into a normal summary is unit-pinned only; forcing it
  means making the sync child fail *after* `collect_batch` has already committed
  `response.jsonl`, which the offline suite can model and a live run cannot easily. Same
  for a repair that partially succeeds — a record that fails the sync chain too, and so
  stays a `FAILED` line after two attempts on two code paths.
- **The conflict budget is untested at its limit in production.** The jittered backoff
  absorbed a cap-sized burst without approaching exhaustion, so the ceiling is unproven
  by anything but unit tests. A cap materially larger than the default, or a much wider
  burst, is where `_CONFLICT_MAX_ATTEMPTS` would first need revisiting — and the
  exhaustion paths, which differ per caller by design, are the least-exercised code in
  the module.

---

## Design record

The numbered records below preserve the section anchors cited throughout the code.
Plain `§N` anchors are the original design plan; `§MN` anchors are the plan for the
merge that absorbed the standalone semaphore stack; `§PN` anchors are the
prompt-inference integration records. Where a record is superseded, the entry says what
replaced it — the point of keeping it is that the current shape only makes sense against
the argument it displaced.

### §1 — What it is

One Step Functions execution per **request**, where a request is N records sharing a
model chain and a mode. **One execution carries all N records**, not one per chunk: sync
runs N serial Converse calls under one lease; batch submits one job containing N
records. This is what makes the two modes the same shape and the same contract, and what
makes batch's padding floor (§5.2) amortize once rather than N times.

### §1.1 — The two paths

Sync and batch touch **different tables**, and never both. Sync: validate & plan →
acquire lease (waitForTaskToken) → serial Converse loop with checkpoint/re-enter →
release → summarize. Batch: validate & plan → submit (row written before the API call;
quota-shaped refusals go `PENDING` and stay parked) → park on the token up to 24 h →
EventBridge completion → stream-driven notify → collect → summarize. **Batch takes no
lease**: the account-level limit is service-enforced, so a lease would serialize
submissions Bedrock is willing to accept in parallel. That asymmetry is the whole reason
the two tables exist separately.

### §1.2 — Two tables, and why they were once two STACKS

The original design put the sync semaphore in a standalone `service-quotas-on-aws` stack
and the batch token table here, arguing from IAM: `states:SendTaskSuccess` was believed
unscopeable (a task token is an opaque bearer credential), so the token grant was
confined to one function per stack, neither able to call Bedrock. The batch **queue**
follows the token on ownership grounds too — a drain submits a job *on behalf of a
parked caller*, using a payload only this workflow understands; sync is a proactive
semaphore over a soft cap, batch a reactive queue over a hard service-enforced limit,
and only the former needed a singleton table. **This split is history** — the merge
(§M) absorbed the semaphore and the tier collapse removed the remaining role isolation.
The two-tables decision itself survives (§M decision 2).

### §2 — What is dropped from the reference implementation

The reference (a prior-generation web-inference service; reference only, deliberately
not a port) contributed the `waitForTaskToken` batch wait, the model-chain fallback
idea, the retryable-vs-permanent exception gate, costless batch padding records, and the
DDB-stream drain trigger. Dropped, with reasons: nested-execution-only input (forbids
Map-over-requests and unit testing; replaced by the S3-mediated request, §4);
error-as-success returns (`throwOnError` defaulting off — every caller must remember to
check a field; replaced by per-record status with an explicit all-failed hard failure,
§7); SDK-error-name retry ladders (Node error names can never match a Python
`ClientError`; replaced by named exception classes, §8); media-domain helpers (a shared
inference service must not know what a scene is); a per-chunk cache written but never
read back (retry N re-paid for chunks 0..N−1; replaced by a checkpoint that is actually
read, §5.1); sync/batch param-shaping divergence (replaced by one shaping function used
by both paths, §9); and a two-table jobName join (replaced by one row keyed by
`workflow_id`, §6.1).

### §3 — Identity and idempotency

```
workflow_id = uuid5(canonical_json(request − output − mode − use_batch − tool_version)
                    + "|" + records_digest + "|" + TOOL_VERSION)
```

A supplied correlation id names nothing and is not in the hash. `ExecutionAlreadyExists`
on `StartExecution` is the **idempotent resubmit path**, not an error. Output at
`s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{workflow_id}/`. The records are a
separate digest term because the request is split across two objects (§4.1);
`records_digest` is over the canonicalized *parsed* records, so serialization
differences cannot move the id — only semantic content can. Record **order** is
significant and not sorted away: `response.jsonl` is in request order, so the same
records in a different order are different work. It is deliberately **not** the S3 ETag:
the gateway must know `workflow_id` before writing (the prefix derives from it), and an
ETag under SSE-KMS is opaque — enabling bucket encryption would silently re-identify
every completed run. One module (`lib/identity.py`), imported by both the gateway tool
and `StateValidateAndPlan` — which recomputes and compares; two copies of the rule is a
hash that silently disagrees, and the symptom is a resubmit that re-pays. **`mode` and
`use_batch` are excluded on purpose**: sync and batch for identical work are the same
work and must collapse to one execution; the mode that ran is reported as
`inference_mode`. `workflow_id` is also the batch job name (§6.1), so job identity, row
identity, and caller identity are one string. Prompt content **is** in the hash because
the caller renders prompts before submitting — an edited prompt changes the request
bytes and therefore the id.

### §4 — S3-mediated request and response

Step Functions payloads cap at 256 KB and one transcript chunk can exceed that, so the
SFN input is a pointer, **always** — not "a pointer when large", which would give two
code paths with one rarely exercised.

### §4.1 — Request

`{prefix}/input_params.json` (control: `tool_version`, `model_chain`, `max_concurrent`,
`options`, `output`) + `{prefix}/request.jsonl` (one record per line: `record_id`,
`system`, `messages`). `system` and `messages` arrive **already rendered and separate**
— separate because concatenating them destroys prompt caching on the stable system
half, the cheapest optimisation for a per-chunk workload; already rendered because the
caller owns prompt resolution and the rendered bytes must be what the hash sees (§3).
`max_concurrent` defaults to **3**, from shot/scene metadata extraction experience.

### §4.2 — Response

`{prefix}/response.jsonl` (one line per record, in request order: `record_id`, `status`,
`model_id`, `output_text`, `stop_reason`, `usage`, `estimated_cost`) +
`{prefix}/summary.json` (`workflow_id`, `tool_version`, `inference_mode`, `models_used`,
`records_total/succeeded/failed`, `failures[]`, `bedrock_usage`, `quota_table`). Fields
that exist for specific reasons: **`models_used`** — what actually ran, not what was
available (on a clean run the fallback never fires, so `model_chain` overstates; spend
reporting needs the former). **`rounds_unpriced`** — rounds whose model had no pricing
entry, reported *alongside* `estimated_cost` so a partial figure is never mistaken for
the whole bill; `estimate_cost` returns `None`, not `0.0`, precisely to keep this
distinction alive. **`inference_mode`** — because mode is excluded from `workflow_id`
(§3), the summary is the only place a caller learns which path ran. **`quota_table`** —
the cheap split-brain detector: two runs reporting two table names is the only visible
symptom of two independent caps (§M2.3).

### §5 — The state machine

Originally one machine: Validate & plan → Choose mode → {sync: Acquire lease → Run
inference → Release lease | batch: Submit batch → Collect batch output} → Summarize,
with a `Catch` on post-acquire sync states routing to Release. **Superseded by the
three-machine split (§M1)** — the per-state reasoning (acquire late, acquire on the sync
branch only, release backstopped by the terminal rule) survives in the current shape.

### §5.1 — Sync: checkpoint and re-enter

The loop skips completed records from `_progress.json`, checkpoints per record, and
yields with `LambdaResourceTimeoutException` near the deadline; the ASL retries
(`MaxAttempts: 10, BackoffRate: 1`) and the re-entry resumes. Load-bearing details:
checkpoint granularity is per record (coarser re-pays completed Converse calls — the
reference implementation's defect, §2); `get_json_if_exists` returns `None` only for a
genuine 404 and re-raises everything else (§12.3); **`refresh_holder` is called once per
invocation, on entry, not per record** — the holder TTL bounds a *state entry* and the
re-entry IS the state entry. Why once is provably enough: `SEMAPHORE_HOLDER_TTL_SEC =
3600` against Lambda's 900 s hard ceiling leaves 2700 s of margin, and the next re-entry
refreshes again — the bound is structural, with no per-call duration to measure. It is
still a correctness requirement because the span outlives the TTL: a multi-hour run
acquired at t=0 has its holder expire at t=3600 while work continues; DDB would reclaim
the holder, the stream would decrement, and a waiter would be admitted **while this loop
is still calling Bedrock** — the cap silently exceeded, no error anywhere.
`MaxAttempts: 10` × 15 min ≈ 2.5 h of wall clock exceeds the 3600 s holder TTL, so the
run survives only because each re-entry refreshes; that coupling is stated as an
invariant. (`drain_pending`'s injected `send_task_success` sender is part of the same
record: the module builds no stepfunctions client at import, which keeps it importable
and testable offline.)

### §5.2 — Batch: one job, all N records

One `CreateModelInvocationJob` containing every record, job name = `workflow_id` (so the
completion lookup is a direct key read, §6.1). Bedrock requires `MIN_RECORDS = 100`;
below that, pad with `maxTokens: 1` records that cost nothing. Padding is applied
**once, after** all real records. The residual floor: the 100-record minimum is
amortized per *request*, so a caller issuing five separate requests for five prompt
categories pays five padded jobs — noted, not acted on. The batch wait is
`waitForTaskToken` with `TimeoutSeconds: 86400`. `Collect batch output` maps `recordId`
back to `record_id` and normalizes into `response.jsonl`, so a caller's parsing code is
mode-independent; any per-record failure becomes a `FAILED` line, not a workflow
failure.

### §5.3 — Model chain

Retryable failures retry on the current model with backoff, then advance; a
`ValidationException` never advances (malformed for every model). The chain walk is
**per record**, not per request — one poisoned record must not downgrade the other 39.

### §6 — The batch queue and the token table

Everything that holds a task token for a Bedrock batch job lives here (§1.2). Adapted
from the reference implementation's backlog/service-token pair; the simplification is
one table instead of two.

### §6.1 — One row per batch job

One item keyed `pk="batch#{model_id}"` / `sk=workflow_id` carrying `status`,
`task_token`, `job_arn`, `request_s3uri`, `model_params`, `ttl`; GSI `job_arn-index`
because the completion event carries a `batchJobArn` and nothing else. The reference
implementation splits this across a backlog table and a token table joined on `jobName`
— a split forced by its cross-stack Lambda layout; here both halves are in one stack, so
the token and the queue entry are one item and the join disappears.

### §6.2 — Submit

Write the row `SUBMITTING` (with token, request pointer, and the `model_params` a drain
needs to resubmit) **before** `CreateModelInvocationJob`; success → `SUBMITTED` +
`job_arn`; a quota-shaped exception (`ServiceQuotaExceededException`,
`ThrottlingException`, `InternalServerException`, `ServiceUnavailable`,
`RequestTimeoutException`, and kin — the 7 `BATCH_QUOTA_EXCEPTIONS`) → `PENDING`, **not
an error to the caller** (the task stays parked). Only quota-shaped exceptions queue: a
`ValidationException` on a malformed record set must fail immediately — queueing it
retries a permanent error forever.

### §6.3 — One holder of `states:SendTask*`

The original handler layout made completion handlers and the drain pure **row writers**
and a single stream-triggered **notifier** the only sender of task results —
`DrainFunction` got Bedrock and no SFN, `NotifyFunction` SFN and no Bedrock, so neither
could do the other's damage. Streams are at-least-once, so the notifier swallows exactly
`TaskDoesNotExist`/`TaskTimedOut` and re-raises everything else — a broad `except` turns
a genuine SFN outage into a silently lost completion. **The IAM half of this is gone**
(§M2.4 and the tier collapse): the layout and the swallow-exactly-two discipline
survive, as code convention pinned by tests.

### §6.4 — Completion and drain

The completion rule (`aws.bedrock`, `Batch Inference Job State Change`, all terminal
statuses) is **prefix-scoped to this deployment's jobs** — the reference
implementation's rule has no scoping filter and fires on every batch job in the account.
`BatchStatusFunction` looks the row up via the GSI and writes the terminal status; the
stream then triggers the notifier. The drain is on the **table stream**, not inline in
the EventBridge handler, for three load-bearing reasons: it must also fire on TTL expiry
and manual deletion; stream handlers are retried by DynamoDB while an EventBridge
handler failure loses the drain silently; and an inline drain throttled partway would
need the *next* completion event to resume — if the completing job was the last in
flight, nothing would ever drain. It stops on the first quota exception, leaving the
remainder `PENDING`.

### §6.5 — The multi-stack drain gap

This stack is not a singleton and the batch limit is per **account**: if stack A fills
the account's in-progress jobs, stack B's `PENDING` rows would drain only when one of
B's own jobs completes — and B has nothing in flight, so no event ever arrives; deadlock
bounded only by TTL. Two fixes, both applied: (1) the capacity rule listens for **any**
Bedrock batch completion in the account — no prefix filter — while the token rule stays
prefix-scoped (two rules, opposite scoping, for opposite reasons); (2) a scheduled sweep
(rate 6 hours) invoking the drain with no image, falling back to
`models_with_pending()` — the backstop for TTL expiry, manual deletion, and any missed
event, and the thing that makes the queue self-healing rather than event-dependent. Six
hours because completions drive normal drainage; the sweep's interval is only the
worst-case latency of the missed-event path, and a stranded row still gets three more
sweeps inside the 24 h token budget. With both, a single-stack deployment behaves as the
reference does, and a multi-stack one degrades to at-worst hours of latency instead of
deadlocking.

### §7 — Partial success

Per-record status. One failed record out of 40 is a **successful execution** with 39
results, one `FAILED` line, and the reason in `summary.failures[]`. The single hard
failure is **every** record failing: that is not a partial result, it is no result
wearing one, and an empty `response.jsonl` with a 0% success rate looks
indistinguishable from a model that legitimately produced nothing. The error message
surfaces up to the first five reasons. A caller decides whether 39/40 is usable; failing
here would turn a partially-useful result into no result.

### §8 — Error contract

Step Functions matches `errorType` = the Python exception **class name**. This is the
single most easily broken thing in the design, because breaking it produces a retry
ladder that silently never fires — which reads as "the workflow just gave up".
`ValidationException` (malformed request, fail immediately);
`BedrockBatchQuotaExceeded` (batch submit refused; row queued PENDING, task still
parked); `LambdaResourceTimeoutException` (sync loop checkpointed; retried ×10);
`BedrockInferenceFailed` (every record failed, §7). The design's original
`QuotaSaturated` no longer exists — a full cap queues the task token instead of raising
(§M2.1), so an exception name for it would be dead `ErrorEquals` configuration.

### §9 — The model matrix lives in the caller

`lib/bedrock.py` is **vendored** — copied into each consuming container and into this
workflow. Pure facts and pure shaping, no I/O beyond the Bedrock call itself. The caller
owns model ids, capability flags, pricing, token sizing; this workflow owns the Converse
call, retry ladder, chain walk, per-record accounting, and the batch queue. Why the
caller and not a service: the caller must shape params *before* it can ask for a lease
or build a request — a network round trip to learn a static fact is backwards; the file
is copied, so drift is visible in a diff and testable entirely offline. The alternative
(this workflow owns the matrix, callers pass an opaque id) sounds cleaner but means a
caller cannot decide batch-eligibility or size a chunk without asking someone.
**Amended:** the id-substring family tuples (`NO_TEMPERATURE_MODELS`,
`ADAPTIVE_THINKING_MODELS`, `EFFORT_MODELS`) are RETIRED — the registry entry's
`capabilities` block is the sole request contract, enforced at seed time
(`states/seed_model_registry._require_capabilities`). The rest of this section is design
history from when the tuples were live. One durable lesson stays: the reference
implementation deletes `temperature`/`topP`/`topK` silently before every retry, so
nobody learns which parameter the model actually rejected — this stack's strip is
targeted and **recorded**.

### §9.1 — Known model-matrix facts

Bedrock Converse only — no vendor-specific API; Message Batches, server-side
`fallbacks`, and first-party prefill rules do not apply. `budget_tokens` 400s on the
current reasoning families. Sampling params are rejected across the 4.7+ family. On
Bedrock, a forced `tool_choice` interacts with `thinking` per family. Bedrock model ids
carry the `anthropic.` prefix. `max_tokens` caps thinking **and** response text together
— a fact about the Converse API, not about this workflow (the caller knob is retired;
see the main section).

### §9.2 — No prefill, no sampling params — sync or batch

The reference implementation contains a base class that skips the assistant prefill when
reasoning is enabled, and a subclass that overrides it and pushes the prefill
unconditionally — a contradiction living inside a single runtime. **The override is the
defect, not the rule it overrides.** Resolution: no assistant-role prefill, no sampling
params, for the reasoning families — full stop, independent of the `thinking` setting;
later widened to the whole family set as a request-shape rule. `build_converse_params`
raises `BedrockError` when the last message's role is `"assistant"` — raises rather than
silently dropping, because a caller who built a prefill-dependent prompt (a forced JSON
opener) would otherwise get an unrelated parse failure downstream with no indication
why. It applies to batch for free, not as a second rule: `to_batch_record` shapes every
record through the same `build_converse_params` (the §12.1 "param shaping parity"
property). The rejection is model-specific, not chain-halting: `error_code()` resolves
to `"BedrockError"`, not `"ValidationException"`, so the chain advances — a fallback
model may accept the very prompt the primary refused. This contradiction having lived
inside a single runtime is the duplication argument in miniature, and the reason the
matrix is vendored as **data** rather than a class hierarchy with overriding behaviour.

### §10 — Consumer contract (original form)

Six rules, now expanded to the eight in the main section. Rule 1: a lease permits one
concurrent Converse call — concurrency inside a held lease is a defect. Rule 2: render
prompts before submitting; this workflow never calls the prompt registry. Rule 3: read
the summary, not just the responses. Rule 4: `ExecutionAlreadyExists` is success. Rule
5: vendor the model registry. Rule 6: never batch a latency-sensitive path — batch is
hours, and a caller that hardcodes it on an interactive path gets what it asked for.
§10.1: a caller needing ONE small Converse call on a synchronous path should not use
this workflow at all — a nested execution plus two S3 round trips for a 2-second call is
worse in every dimension. The line: N records or minutes of work → this workflow; one
call, seconds → call Bedrock directly under your own throttling discipline.

### §11 — Consumer callback tokens (`callback_task_token`)

The service-side completion notification that llm-prompt-driven consumer workflows wait
on (designed jointly with the conversation-analysis workflow, this stack's first
callback consumer). One table (`${ResourcePrefix}-bi-callback-tokens`, DynamoDB #3 —
disjoint access pattern and roles from both others: the gateway writes, the waker
completes), one EventBridge rule on the orchestrator's terminal events, one waker
Lambda. Register-before-start eliminates the race by construction; synchronous
completion on absorb is the paid-work-reuse path; FAILED is delivered as
`SendTaskSuccess` DATA, never `SendTaskFailure`, so the consumer's Choice state decides.
The full rationale is in [Gateway tool and callbacks](#gateway-tool-and-callbacks).
(This section number originally described the pre-merge single-machine stack inventory,
which §M1/§M3 superseded; the number was reused for the callback design, which is what
every `§11` citation in the code means.)

### §12 — Verification

### §12.1 — Unit-testable (stubbed Bedrock)

The checklist the offline suite implements: `workflow_id` stability and sensitivity;
checkpoint/re-enter (a stubbed clock forces a checkpoint; the re-entry must skip
completed records and total Converse calls must equal the record count exactly); a 500
on the checkpoint GET must raise, not read as "no checkpoint"; chain walk (exhaust then
advance; `ValidationException` never advances; a poisoned record does not downgrade
neighbours); partial success (§7); batch padding (below the floor → padded to 100;
above → no pads); batch classify (quota list → PENDING; `ValidationException` → raise,
no row); batch drain (stops on the first quota exception; a stream event for model A
does not drain model B; the argless sweep drains every model); notifier idempotency (two
deliveries send once effectively; exactly two dead-token errors swallowed); token lookup
by GSI; batch normalization (per-record error → one `FAILED` line and a successful
execution; `recordId` → `record_id` mapping order-independent); **mode parity** (the
same request through sync and batch produces identically-shaped `response.jsonl` and the
same `bedrock_usage` keys); **param shaping parity** (sync and batch build the same
`inferenceConfig` and `additionalModelRequestFields` for the same model — §2's
divergence defect); release reached on failure (superseded by the terminal-rule design,
§M3.2); IAM/code shape (the drain contains no `SendTask*` call and the notifier no
Bedrock call — now a `_code()` convention test, §M2.4).

### §12.2 — Requires a deployment

The ten live tests: (1) long sync run with ≥2 checkpoint/re-enter cycles and the lease
surviving throughout; (2) holder TTL — a run longer than the TTL with `refresh_holder`
disabled must reproduce the early-admission bug, with it enabled must not (the one test
that proves §5.1's invariant, and it can only run live); (3) batch end-to-end through
the completion-event → status-row → stream → notify chain; (4) batch saturation past the
account limit, with the **last** completion draining the tail; (5) the scheduled sweep
recovering a hand-broken completion within one interval; (6) EventBridge scope — a
foreign batch job must NOT invoke the status handler but MUST invoke the drain (both
halves, since §6.5 depends on the asymmetry); (7) idempotent resubmit — same request
twice, one set of Bedrock charges; (8) contention — more concurrent executions than the
cap, the overflow parked and admitted on the first release; (9) prefill/adaptive
thinking probed live before the registry encodes either comment of a contradictory pair
(§9.2); (10) cost reconciliation of `estimated_cost` against a real bill with
`rounds_unpriced` at 0. The Open section records which remain unexercised.

### §12.3 — Silent-failure sites

| Site | Silent symptom |
|---|---|
| `ErrorEquals` on a botocore error code instead of a class name | Retry never fires; reads as a workflow that gave up |
| No `refresh_holder` in the sync loop | Cap silently exceeded mid-run |
| Broad `except` on the checkpoint read | Every completed record re-inferred and re-paid, worst under load |
| Checkpoint written but never read | Same, permanently — the reference implementation's actual behaviour |
| Broad `except` in the notifier | A genuine SFN outage looks like a delivered completion; caller waits out its 24 h timeout |
| Drain rule scoped by prefix | Multi-stack deadlock (§6.5) — invisible until two stacks exist |
| No scheduled sweep | Queue depends entirely on events; one missed event parks a caller until TTL |
| Error-as-success returns | Caller treats an error payload as a result |
| `mode` included in `workflow_id` | Sync-then-batch retry pays twice for identical work |
| Sync/batch param shaping divergence | Same request, different model behaviour, depending on mode |
| Concatenating `system` + `user` | Prompt caching silently lost; only the bill shows it |
| `estimated_cost` without `rounds_unpriced` | Partial figure read as the whole bill |
| Non-quota exception queued as PENDING | Permanent error retried forever, invisible until the token times out |
| Unrendered `{{var}}` reaching the model | Model invents content around the literal token |

### §13 — Not in scope

Prompt resolution (the prompt-registry workflow; callers render before submitting).
Chunking (caller domain knowledge). Result validation or parsing — this workflow returns
`output_text`; XML/JSON parsing, schema validation and merging belong to the caller
(salvage is `parse_partial_output`, §P6.2). Streaming responses (every consumer is
batch-shaped; would need a different contract than "write the response to S3"). Tool use
/ structured outputs (no consumer needs them yet, and `tool_choice` interacts with
thinking in ways worth designing deliberately). Embeddings (different service shape,
different quota profile). A global batch queue across inference stacks — §6.5 mitigates
the multi-stack gap with an unscoped drain rule and a sweep rather than a shared table;
a shared queue would need the singleton property this stack deliberately gave up
(§M2.3); revisit only if multiple inference stacks become normal rather than possible.

---

### §M — The semaphore merge

The merge made this stack **self-contained**: no quota-stack parameters, no sibling
stack to deploy first; the semaphore moved in following face-recognition's pattern, and
the standalone `service-quotas-on-aws` stack was deleted. Four decisions were settled up
front and not re-opened:

| # | Decision |
|---|----------|
| 1 | **Plan in the orchestrator, `Summarize` in each child.** Both children return a complete `summary.json` + `response.jsonl` and are independently startable given a plan. |
| 2 | **Two DynamoDB tables** — `sync-infer-quotas` (semaphore) and `batch-infer-tokens` (batch rows). A straight rename of both pre-merge tables; no item-shape or stream-filter changes. |
| 3 | **Follow face-recognition literally** for `states:SendTask*` — the semaphore handlers collapse onto the shared workflow function. |
| 4 | **Delete the standalone quota stack**, porting its valuable tests into this suite. |

### §M1 — The three state machines

An orchestrator (`Validate & plan` → `Choose mode` → one of two `.sync:2` children →
`Usage tracking`) plus two standalone children: the sync child (`Acquire lease` → `Run
inference` → `Release lease` → `Summarize`) and the batch child (`Submit batch` →
`Collect batch output` → `Summarize`, later + the repair tail, §P8). The orchestrator
keeps the pre-merge machine's name so `InferenceStateMachineArn` — which consumers call
at `.sync:2` sites and derive a name from — keeps meaning "the machine a caller
starts"; no consumer change was needed. The parameter surface shrank with the merge and
again afterwards: two parameters remain (`ResourcePrefix`, `ProxyBucket`).

### §M1.1 — What leaves the ASL

`Release lease on failure` and `Failed` are deleted from the sync machine, leaving
exactly face-recognition's three holder-disappearance triggers (normal release, terminal
rule, holder TTL). Two accepted costs: reclaim latency on the failure path grows from
in-band to when-the-terminal-event-lands; and the `Acquire lease` `Catch` goes too — its
comment documented a real window where `try_acquire` succeeds and the `SendTaskSuccess`
after it fails, leaving a live holder on a token nobody will redeem. That case is now
reclaimed by the terminal rule when the state times out; the execution sits for the
state's `TimeoutSeconds` first — but it sat holding a slot the drain could not reuse
either way, so the in-band release only ever shortened the window, never closed it.
**Recorded as a known regression, not a silent one.** Both `Fail` states went with it,
and every `Catch` in the orchestrator: `Track usage on failure` (sub-cent estimate
bought by laundering every child error into one static `Fail`), `Unresolved mode`
(`Choose mode` has no `Default`; an unmatched mode raises `States.NoChoiceMatched`, the
same outcome, and a `Default` falling through to sync would be worse), and
`States.TaskFailed` out of both `.sync:2` retry ladders (retrying it re-issues
`startExecution` under a fixed name and the caller sees `ExecutionAlreadyExists` instead
of the real error — the `Catch` had been masking this). The terminal rule also covers
ABORTED — a status no `Catch` can see.

### §M1.2 — `.sync:2`, and what the orchestrator reads back

Both child calls are `states:startExecution.sync:2` (parses `Output` as JSON, so the
orchestrator reads the child's `Summarize` return directly). Child execution names are
`{workflow_id}-sync` / `{workflow_id}-batch` — not bare `workflow_id`, because mode is
deliberately excluded from the hash, so with two child machines an unsuffixed name would
be reusable but ambiguous, and a mode flip on a retried `workflow_id` would collide with
the sibling's history. The suffix keeps the idempotency property (the orchestrator
execution is still named exactly `workflow_id`) while giving each child a distinct name.
`ResultSelector` lifts the child's data into `$.data` so `Usage tracking` sees one shape
regardless of branch.

### §M1.3 — Three definition files: inline, no tokens

Originally this called for `%%DEFINITION*%%` tokens and an injection loop; **that is not
what shipped**. Each ASL is inline in `cfnstack.yaml` (the single source of truth for
what deploys) and the matching `definitions*.json` is a readability copy — lintable,
diffable, openable in a state machine editor. Both sides get edited together, and two
guards make a divergence loud: `verify_sfn_definition` fails the build (parsed-JSON
compare, so a re-indent is not drift) and
`test_each_inline_definition_still_matches_its_readability_copy` compares exact strings.
The build's check runs against the **source** template, before the `sed` token pass —
that pass rewrites `%%VERSION%%` everywhere, ASL comments included, so checking the
built copy would report a substitution as drift. (The arrangement grew out of an earlier
defect in the standalone stack where a built template had been copied back over the
source; inline-with-drift-guards is the fix that keeps the two sides honest.)

### §M2 — The semaphore moves in

The vendored `lib/semaphore_client.py` was kept over face-recognition's reference copy —
it is strictly better in three ways that had to survive "follow face-recognition
literally" (§M2.1). The table rename is the only semantic change (§M2.3). The item
shapes are unchanged by the merge: `counter`, `holder#{arn}`, the `lease#{arn}` reverse
pointer, `pending#{scope}` waiters, and `stream-dedup` markers (drawn in
[The semaphore](#the-semaphore-three-holder-disappearance-triggers-one-decrement)).

### §M2.1 — Keep the vendored client, not the reference's

Three reasons: (1) `try_acquire` reads `CancellationReasons[1]` alone — the reference
*also* requires `reasons[0]["Code"] == "None"`, which reports a live holder's own retry
as saturated and deadlocks it behind its own slot; the reentrant-at-cap test (a holder
re-acquiring at a full cap must get `"reentrant"`, not `""`) is the regression pin and
the single most valuable test ported. (2) Multi-scope keys plus the `lease#{execution}`
reverse pointer — a terminal event carries only `executionArn` while the scope is in the
partition key, and this stack really is multi-scope (`quota.resource` is the primary
model id, so each model is its own cap). (3) `drain_pending` takes an injected
`send_task_success` sender, so the client builds no stepfunctions client at import.
`acquire_or_raise` — a non-parking acquire with **no production callers ever** — was
deleted with its tests rather than carried as a dead public entry point; `QuotaSaturated`
went with it (a full cap queues the task token, so an exception name for it would be
dead `ErrorEquals` configuration).

### §M2.2 — The semaphore state modules

`states/acquire_lease` (keeps the deliberate re-raise of an unexpected `ClientError`
after `SendTaskSuccess` — swallowing a transient send failure strands the caller for the
full `TimeoutSeconds`; keeps `max_concurrent` with **no default** — the ASL always
passes `$.data.quota.max_concurrent`, and a default here would silently substitute a cap
nobody asked for), `states/release_lease`, `states/handle_semaphore_release` (the stream
handler), `states/handle_execution_terminal` (the `aws.states` handler). The stream
handler calls `drain_pending()` **directly** (decision 3) — the standalone stack's
out-of-band async self-invoke, which existed only so the decrement function could hold
no `SendTask*`, is deleted along with its env var and invoke grant: a hop that could
fail silently, protecting nothing. The **per-scope** drain loop is kept (derive each
record's scope from its own keys, dedup scopes, drain each once) — the reference's
argless `drain_pending()` is single-scope and would only ever drain one cap. The
unconditional drain is kept with the reference's reasoning verbatim: a crashed attempt
may have died mid-drain, and if the removed holder was the last active one, no future
`REMOVE` will fire.

### §M2.3 — The cap is per PREFIX now

**The singleton property is deliberately destroyed, and this is the one semantic change
in the whole merge.** The standalone stack's cap was region-wide *because* the table
name was fixed: a second stack with the same name failed at CREATE. Now the cap is per
deployment prefix; two stacks with different prefixes in one region hold two independent
caps against one real Bedrock quota and can jointly exceed it. That is the accepted
price of "self-contained" — a workflow that needs a region-wide cap must share a
`ResourcePrefix`. `summary.quota_table` reports the table name as the split-brain
detector. (A third `STREAM_HANDLER` value, `semaphore`, arrived with the move; the
router distinguishes the workflow function's three trigger shapes — `operation`,
`Records[]`, `source` — by event shape, with `_is_eventbridge_terminal` ahead of the
fallthrough raise.)

### §M2.4 — What the workflow role gained, and the invariant that broke

The record of a cost being weighed and accepted, in two steps. Pre-merge, the workflow
role deliberately had no `UpdateItem` on the quota table, so IAM itself proved "exactly
one decrement path". The merge gave the role the table writes, stream read, and
`states:SendTask*` — the convergence property **survived** (all triggers reach exactly
one `decrement_once`, deduped on the stream `eventID`) but its *IAM proof* was lost,
replaced by a tree-walking test; the role still could not call Bedrock, so the letter of
§1.2 held while its spirit thinned. **Superseded in its conclusion, not its reasoning**,
by the tier collapse: the six workflow functions became seven memory tiers over one
container image and one `BedrockInferenceLambdaRole`, so `SendTask*` and
`bedrock:InvokeModel` now sit on the **same** role and the last of §1.2's isolation is
gone. What survives, and survives only in code: exactly one caller of `decrement_once`,
pinned by a test that walks the tree; and the co-residency itself is asserted by name so
a re-split fails a test instead of leaving a stale claim of isolation.

### §M3 — Template changes

§M3.1: the quota-stack parameters deleted; the remaining request/deploy defaults later
moved to `lib/input_schema.py` and `Mappings` (see [Stack](#stack)). §M3.2: one table
mapping became two (`sync-infer-quotas`, `batch-infer-tokens`), same key schemas, same
GSI, same streams — and `Release lease on failure` left the ASL (§M1.1); the
removed-behaviour audit test pins that trigger 2 stays gone. §M3.3: three machines,
three log groups, three machine roles; the orchestrator inherits the pre-merge name;
`.sync:2` needs the managed-rule `events:*` grants, whose omission is the classic
failure that reads as an IAM error on a resource nobody wrote. §M3.4: `QuotaTerminalRule`
retargeted to the workflow function and listing **all three** machine ARNs — all three
even though only the sync child holds leases, because an orchestrator ABORTED
mid-`.sync:2` does not necessarily terminate its child, and the batch machine is listed
so a future lease on that path is not a silent leak; stream `FilterCriteria` copied
verbatim (a correctness requirement, not an optimization: a `stream-dedup` marker's own
TTL expiry arrives as a `REMOVE`, the one event that must never read as a released
holder). §M3.5: the standalone stack's stub-consumer apparatus dropped entirely — the
real sync machine is the verification vehicle.

### §M4 — `usage_tracking`: the kind-tagged model, not the state-span model

Two cost-report shapes existed among the sibling workflows. The state-span model bills
state spans — and the orchestrator's history is almost entirely `.sync:2` child spans
and a `waitForTaskToken` park, so billing those as Lambda time materially over-reports
(the reference module's own docstring records the over-report measured on a real
execution). The kind-tagged model tags each span `lambda`/`child`/`token`/`map` and
bills **only** `lambda`, treating waits as queue time — precisely the orchestrator's
shape. Ported accordingly, rolling up the whole tree. A `child_execution_arn` scalar
parameter was later dropped deliberately: a repaired batch run is a TREE — orchestrator
→ batch child → sync repair grandchild — and a scalar could name only one of the two
edges, while the grandchild is where the repair's Converse calls and lease wait are
billed; the handler discovers descendants from the `TaskSubmitted` events instead and
recurses.

### §M5 — Tests ported and added

§M5.1 (ported from the standalone quota stack, decision 4): the five semaphore
invariants as named tests; the `_FakeDdb` transaction double with its
`TransactWriteItems` `CancellationReasons` modelling (the two suites' doubles merged —
each contributed behaviours the other had no need for; unmodelled expressions keep
**raising** rather than defaulting to `True`); the reentrant-at-cap regression test
(§M2.1). §M5.2 (new cross-file contract tests): every `${...}` in each ASL balanced
against that machine's `DefinitionSubstitutions`; `SendTask*` confined to an exact role
set (now the one union role, with co-residency asserted); the terminal rule's
`detail.stateMachineArn` lists exactly the three machines; the orchestrator's `.sync:2`
states name the two children and nothing else; no `Map` anywhere in any ASL;
`STREAM_HANDLER` takes exactly `notify`/`drain`/`semaphore` with the `aws.bedrock`
handler on `notify`; both tables' stream `FilterCriteria` match the clients' key and
status constants.

---

### §P — Prompt-inference integration records

### §P5.1 — The seeded model registry and runtime routing

The stack seeds its **own** registry (`Custom::ModelRegistrySeeder`, from
`models/availablemodels_master.json` to `_model_configs/availablemodels.json` on the
proxy bucket), ending any dependency on an external model list. The registry is
snake_case, bare ids (no geo prefixes stored), with per-entry `inference_region`
routing, `capabilities`, and pricing. `resolve_runtime_model_id` derives the id handed
to Converse/batch from the routing scope: `global`/geo scopes prefix the bare id
(`global.{id}`), a concrete region uses the bare id in that region; legacy prefixed ids
pass through untouched when no override is given. A request-time routing override exists
for the residency case — a model may support `global` while local law requires the run
to stay regional; the override must be LITERALLY in the entry's `supported_regions`, and
refusing loudly is the correct answer.

### §P5.4 — `modelInvocationType: "Converse"` — one dialect for both paths

`create_job` declares `modelInvocationType: "Converse"`, the job-level parameter that
decides which dialect Bedrock validates batch records in; without it the API defaults to
`InvokeModel` and judges Converse-shaped records malformed. An entire per-family
translation layer was built to "fix" the resulting all-rejected jobs before the missing
parameter was identified; it is deleted, and sync and batch share one request AND
response dialect. Re-adding a per-family body would now be the bug it was thought to
fix. (This parameter is also why `requirements.txt` pins a botocore floor — see
[Stack](#stack).)

### §P5.5 — Enforced image limits (`capabilities.image.limits`)

The Bedrock-enforced half of the image rules, rejected at plan time before a token is
spent: `max_images` per request, `max_image_mb` per inline image, `max_payload_mb` as
the record's inline-byte total; null means not-declared, nothing pre-checked. A
4-image record on a hard-cap-3 model otherwise fails after submit — or in batch, after
the whole job validated and parked. All pinned values are measured live and ride one
step conservative of the measured pass/fail boundary (the Gotchas entry has the
numbers); do not change a pin without re-measuring.

### §P6.2 — `parse_partial_output`: copied, not imported

The salvage module for truncated or malformed model responses. The source of truth
lives in `bedrock-inference-on-aws/lib/`; consumer workflows copy the file verbatim, and
the bundle suite pins byte-parity plus golden known-answer vectors the day a second copy
exists. Per-consumer variation must live in the CALL, never in the module — the same
vendor-as-data discipline as `lib/bedrock.py` (§9).

### §P8 — The batch repair tail

Repairing a batch job's failed records IS sync inference — lease, chain walk,
checkpoint, release — so the batch child invokes the sync CHILD MACHINE for it
(`{workflow_id}-repair`) instead of duplicating four of its states. What that buys:
`run_inference` and the sync ASL are untouched, there is exactly one
lease/checkpoint/release spine in the stack, and "either child is startable on its own
given a plan" still holds. What it costs: the start graph gains an edge, so it is stated
as an invariant — orchestrator → {sync, batch}, batch → sync, sync → nothing, ACYCLIC,
and a test pins it. The route in is a `Catch` on `Collect batch output`, not a `Choice`
(the main section records why); `collect_batch` raises
`BatchRecordsIncompleteException` only over already-committed S3 state
(write-then-raise), and only when `0 < failed <= REPAIR_MAX_FRACTION × total` — above
the bound is systemic and a sync re-run would re-pay the whole job at on-demand rate to
fail again; below it, failures are per-record weather the chain walk repairs in seconds.
`options.batch_repair: false` opts out. A repaired record costs the on-demand rate
against its siblings' 50% batch discount — a repaired run is visibly more expensive, by
design.
