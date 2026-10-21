# cast-resolution-on-aws

Resolves WHO is in a media asset. Speech diarization emits anonymous
`SPEAKER_NN` labels; face recognition emits person/track ids; on-screen
text (chyrons, credits, titles) carries names nobody has joined up.
This workflow binds them into resolved **person entities** — who is in
the video, what they said, what they're called, and what their role is
— for every genre with one code path:

- **Episodic/film**: celebrity → actor → character name; role = character.
- **News**: face + lower-third + speech → named person with role
  (anchor / correspondent / reporter / guest).
- **Sports**: commentators, athletes (a prompt variant; same machinery).

It consumes sibling micro-workflow run prefixes (speech-diarization,
face-recognition, shot-segmentation, shot-metadata-enrichment) and emits
`persons.json` plus flat remap tables (`SPEAKER_03 → person_2 → "Dan
Reyes"`), and — when a credits read ran — a verbatim `credits.json`
witness product that downstream consumers (e.g. audio description) use
directly. Typical consumers: audio description (name the character, not
"a man"), search (real names into entity indexes), moderation context
(WHO did the flagged act), news QC (speaker turns tagged with role).

Sharp edges live in [GOTCHAS.md](GOTCHAS.md). The numbered **Design
record** at the end of this file is the design history code comments
cite (`README.md §N`).

## Architecture

Two Lambdas out of one flat zip (a gateway dispatcher and an
operation-routed workflow handler) and ONE Step Functions machine with
SIX linear states — no Choice states, no polling anywhere. Every wait
is a single held `waitForTaskToken` state completed by a service-side
waker; every skip is a SELF-completed token carrying its reason as
data; redrive is the reconcile. Deadman timeouts: witness held state
93600 s (batch SLA), enrichment held states 3600 s, machine 108000 s.

```
 caller (agent / sibling workflow)
   │  resolve_cast {sibling run prefixes, flags?, idempotency_token?}
   │    boolean flags, ALL default TRUE, normalized into the hash
   │    (omitted == default == same work): read_onscreen_text |
   │    llm_enrichment | use_batch (witness-read economics)
   ▼
 GATEWAY  workflow_id = uuid5(request + tool_version [+ token])
   │      resubmit of an identical request → the existing run; an
   │      unknown boolean parameter is warned and ignored, never an error
   ▼
 [1] ValidateSources        refuse early: input shape, own-bucket rule,
       │                    sibling artifacts exist; pin the exact sibling
       │                    workflow_ids + normalized flags → _plan/plan.json
       ▼
 [2] ResolveAddressees      LLM vocative round 1 (transcript only, so it
     (HELD, bedrock         runs FIRST). Skips self-complete: enrichment
      callback)             off / no transcript / no cues / nothing
       │                    unresolved. Else: adjacency-unresolved vocative
       │                    sites → bedrock-inference (tiny text records,
       ▼                    sync); its waker completes the token
 [3] ReadOnscreenText       the witness. Skips self-complete: reads off /
     (HELD, frame-          no witness ARN / probe says no callback token /
      inspection callback)  nothing to read; all-ledgered completes with
       │                    the already-paid-for product prefixes. Else:
       │                    plan reads (evidence triggers + need checks +
       │                    crop-aware ledger), submit ONE pooled request
       ▼                    + token; the waker returns terminal status AS DATA
 [4] BindPersons            THE durability line — ONE bind, never repeated:
       │                    sources ► transcript cues (language-gated) ►
       │                    vocative adjacency voting (round-1 addressees
       │                    join as ORDINARY cues) ► co-occurrence ►
       │                    class-1 matching ► assemble persons [tooth-1
       │                    critic inside: veto impossible merges, demote
       │                    collisions — free, always on] ► witness
       │                    products ► title cues + cast pairing ► roles ►
       │                    append read ledger ► persons.json + provenance
       ▼                    pin + remap tables + usage
 [5] AdjudicateCast         cast adjudication + tooth-2 critique in one
     (HELD, bedrock         pooled submission (independent records pool —
      callback)             the state-count rule). Skips: enrichment off /
       │                    no dossiers. Reads the SHIPPED persons.json,
       ▼                    never re-binds
 [6] ApplyEnrichment        AMEND-only: firewalled verdicts join persons
                            (mechanism cast_adjudication, never an
                            overwrite); tooth-2 flags land in
                            critic_review (report-only); usage recomputed
                            from the pre-enrichment base, never +=
```

Why this order: ResolveAddressees needs only the transcript, so it runs
before the witness wait and BindPersons consumes both. ONE bind makes
three failure classes structurally impossible: enrichment losing
witness title cues, losing the provenance pin, and discarding round-1
results/cost when round 2 skips.

### Layout

```
cfnstack.yaml               2 Lambdas, 1 machine (ASL inline), params
definitions.json            readability copy of the ASL — edit BOTH; the
                            build and a test fail on drift
build-deploy-s3-dist.sh     flat zip; tests gate packaging; ASL drift gate
create_update_stack.sh      --resource-prefix / --proxy-bucket
lambda_function.py          the operation router (top level)
lib/                        binding logic: sources (parsers + S3 loaders),
                            transcript_cues, vocatives (+ vocatives_llm),
                            cooccur, class1, persons, critic (demote-only),
                            critic2 (report-only dossiers), adjudicate,
                            reads (focus-read scheduling + product
                            parsing), ledger (cross-run never-re-pay),
                            bind_pipeline (THE bind pipeline, one copy for
                            state/harness/gate), pb_common (rule shapes +
                            gateway invoke), s3io
states/<name>/lambda_function.py   the six state handlers, one folder each:
                            validate_sources, resolve_addressees (held),
                            read_onscreen_text (held), bind_persons,
                            adjudicate_cast (held), apply_enrichment
gateway_tool/               dispatcher (resolve_cast / get_job_status /
                            describe_tool), output_contract.json,
                            input_examples.json, workflow_identity.py
                            (byte-identical family copy)
test/test_units.py          the suite — binding rules, critic, gateway
                            validation, ASL parity
test/harness/               local tooling (local bind harness, corpus
                            input extractor, review-page builder, gate and
                            critic scorers) — tooling lives here, never a
                            scripts/ folder
viz/                        persons_template.html (house review-page look)
```

Zip shape: the workflow zip keeps the `lib/` + `states/` TREE — imports
are package-style (`from lib import x`) — with the top-level
`lambda_function.py` router; the gateway dispatcher ships its own flat
zip plus `lib/pb_common.py`, the one lib module it needs for
`DEFAULT_FLAGS` at the door.

## Inputs and outputs

**Mandatory: at least ONE identity source** — a speech-diarization
prefix OR a face-recognition prefix. With neither there is nothing to
bind and the gateway refuses at submit time (the error names the
condition). Nothing else is required — not even the asset uri: frames
come from shot-segmentation's retained archives, never from the source
media.

| Input | Enables | Absent ⇒ |
|---|---|---|
| speech-diarization | speakers, utterance attribution, transcript cues (self-ID / vocatives / introductions) | silent-face persons only; no quotes |
| face-recognition | face tracks, celebrity names, appearances | speaker-centric persons; chyrons bind to the ACTIVE SPEAKER at display time; `face_map.json` ABSENT (not empty) |
| shot-segmentation | the dense frame archives behind ALL witness reads, plus the shot lattice | free-evidence-only mode: no OCR reads at all |
| shot-metadata-enrichment | overlay-text evidence, depicted-face guard, people joins, read TRIGGERS | weaker gating; depicted guard degraded (co-occurrence corroboration bar rises) |

Three-state semantics throughout: an ABSENT source makes its dependent
outputs ABSENT ("not assessed"), never empty ("attested none").
`persons.json` carries a `sources` block naming which inputs were
present, so a consumer can tell "no faces bound" from "faces were never
available". Note on privacy postures: cast-resolution does no biometric
processing itself — it only consumes face-recognition output when a
deployment provides it (see §16); name↔speaker binding from
transcript/chyron text is still personal data, and the
evidence-with-provenance shape is what lets a deployment filter to its
own policy.

Outputs (labels and basenames are stable API; see
`gateway_tool/output_contract.json`):

- `persons.json` — the resolved graph: person types, names (only from
  independent naming mechanisms), per-binding evidence + confidence,
  critic vetoes, unresolved residue listed explicitly, and the
  provenance pin naming the exact sibling workflow_ids the binding is
  valid against.
- `speaker_map.json`, `face_map.json` — flat remap tables, pinned to
  source run ids; `face_map.json` is conditional on a face source.
- `on_screen_text.json` — every OCR read's raw yield `{text, region,
  t0, t1, purpose}` (reusable evidence, independent of this run's
  bindings).
- `credits.json` — verbatim witness product when a credits read ran.
- `usage_tracking.json` — standard cost/usage accounting; every step
  that spends model money is priced here.

Person **types** make the edge cases expressible instead of failures:
`on_camera_speaker | voice_only | silent_face | depicted_only |
uncertain` (`voice_only` = narrator/announcer/phone; `depicted_only` =
photos, file footage, monitors — faces that must NEVER bind to
concurrent speech).

## Gateway tool contract

Standard three-tool target `cast-resolution`:

- `resolve_cast` — submit. Sibling run prefixes
  (`speech_diarization_s3uri`, `face_recognition_s3uri`,
  `shot_segmentation_s3uri`, `shot_metadata_s3uri`), optional boolean
  flags (`read_onscreen_text`, `llm_enrichment`, `use_batch` — all
  default true and are normalized into the identity hash), optional
  `credits_hint` and `idempotency_token`. Returns a HANDLE: `job_id`,
  `workflow_id`, `status`, `output_s3_prefix`.
- `get_job_status` — poll the handle; terminal responses enumerate the
  outputs present.
- `describe_tool` — self-description, including the capability
  features consumers probe.

`workflow_id` = uuid5(request + tool_version [+ token]). The request
names sibling RUN PREFIXES, so identical prefixes = identical work: a
resubmit is absorbed into the existing run, and a sibling re-run
changes its prefix and correctly produces a NEW binding run (see §1).
Outputs land at `micro-workflow/cast-resolution/{workflow_id}/`.
Example requests: `gateway_tool/input_examples.json` (every value an
invented placeholder; each example passes the gateway's full input
validation, pinned by a unit test).

## Build & deploy

```bash
# build only (tests + ASL drift + zip-member gates must pass to package)
bash ./build-deploy-s3-dist.sh --build-only

# build and publish to your deployment bucket(s)
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution NS] [--version V] [--single-region]

# create/update the stack from the BUILT template
bash ./create_update_stack.sh --stack-name my-cast-resolution \
    --resource-prefix myprefix --proxy-bucket MY_PROXY_BUCKET \
    --fi-lambda-arn <frame-inspection gateway ARN> \
    --bedrock-lambda-arn <bedrock-inference gateway ARN>
```

No docker, no CodeBuild, no ECR: pure boto3 + stdlib. This is a
CONSUMER stack — the frame-inspection and bedrock-inference gateway
Lambda ARNs are parameters, so those stacks deploy first; it is also
wired as a child of the micro-workflow family bundle. **Bump `.version`
on every deploy**: the S3 key carries the version and CloudFormation
detects key changes, never content changes — an unbumped redeploy
leaves old code running with no error to notice.

The three enrichment system prompts (adjudicate / critic2 / vocatives)
live in prompt-registry as INTERNAL admin-surface categories
(`cast-adjudicate` / `cast-critic2` / `cast-vocatives`): agents never
see them, and editing one is a registry reseed, not a cast-resolution
deploy. Genre prompt versions are `default` (episodic) and
`default_news` (see §4).

## Tests

```bash
cd cast-resolution-on-aws && python3 -m unittest discover -s test -p 'test_*.py'
```

Mocked-boto3 unit suite (`test/test_units.py`): binding rules, critic
teeth, gateway validation, ASL parity between `cfnstack.yaml` and
`definitions.json` — template invariants are parsed, never grepped.
The build script runs the suite before packaging; a red test means no
zip. Local tooling under `test/harness/` binds real sibling outputs
offline (the harness and the BindPersons state share
`lib/bind_pipeline.py`, one copy).

## Gotchas

- **The ASL lives twice** — inline in `cfnstack.yaml` and as a
  readability copy in `definitions.json`. Edit BOTH; the build and a
  unit test fail on drift.
- **Remap tables are join-validity-scoped**: `speaker_map.json` is
  only meaningful against the one diarization run whose transcript
  used those labels. Consumers must check both artifacts name the same
  diarization `workflow_id` (§1).
- **Absent ≠ empty**: `face_map.json` missing means "no face source
  was provided", not "no faces found". Guard for absence, don't
  default to `{}`.
- **The read ledger is crop-aware**: a full-frame entry satisfies any
  need, a cropped entry satisfies only a cropped need (§9). Span
  merges keep the WIDER frame need.
- **Capability probing is mandatory**: the caller strips the `crop`
  key (request AND ledger candidates) when the deployed witness does
  not advertise `span_crop` — otherwise an accept-and-ignore witness
  would make the ledger claim a crop that never happened (§10).
- **Flags are hashed**: omitting a flag, passing its default, and
  passing it explicitly are the SAME workflow identity. Unknown
  boolean parameters are warned and ignored, never an error.
- **Error messages name the condition**, never prescribe CLI commands
  (family rule).
- **Captions (.scc/.itt) are validation-only** — they never feed the
  workflow.

---

## Design record

The numbered design history. Code comments cite these sections as
`README.md §N`.

### §1 — Identity is an overlay, never a rewrite

Source outputs stay immutable; consumers relabel via remap tables, and
sources stay independently re-runnable. Bindings are only valid
against the exact sibling workflow_ids they were computed from —
`SPEAKER_ID`s are per-run labels, and a diarization re-run renumbers
its clusters. The **provenance pin** in `persons.json` is a
join-validity label, not a cache: reading sibling outputs is trivially
cheap and always done fresh; the pin exists for CONSUMERS, so that
"SPEAKER_03 → person_2" can be checked against the one run it is true
of. A sibling re-run changes its prefix, changes the input hash, and
yields a NEW binding run; a cast-resolution re-run on the SAME sibling
runs is absorbed (gateway idempotency does both naturally).

### §2 — Evidence over verdict; abstention over fabrication

Every binding carries mechanism, timespan(s), confidence, and
evidence; consumers see WHY. A wrong name is worse than "the
detective", so the locked binding rules are:

- only CONFIRMED bindings merge identities; proposals are reported,
  never acted on (wrong merges misattribute quotes — as bad as wrong
  names);
- a person is NAMED only by an independent naming mechanism (face
  source name, self-ID/sign-off cue, overlay name at the person's own
  timespan) — never by co-occurrence alone;
- name CONFLICTS abstain: `name = null`, both candidates kept in
  aliases, type `uncertain`;
- unresolved residue is first-class output, never dropped;
- three-state sources: absent input ⇒ dependent fields absent.

### §3 — Naming

The workflow is `cast-resolution`: "cast" is unambiguously content
metadata (EPG/streaming vocabulary that covers presenters and reality
participants, not just scripted cast). Rejected alternatives: a
"person-binding"-style name reads as people-tracking to security
review; `talent` excludes news subjects; `identity`/`entity` collide
with IAM / AWS Entity Resolution; `role-*` reads as RBAC. The DATA
nouns stay `person` (`persons.json`, `person_id`, person types): a
character, an anchor, a guest, an athlete, and a face in file footage
are all persons, and the noun identifies nobody.

### §4 — Genre lives in prompts, never code

Binding code never branches on genre; genre is a prompt version
(`default` for episodic, `default_news` for news) resolved through
prompt-registry. The three enrichment system prompts are internal
admin-surface registry categories (§ Build & deploy). ValidateSources
binds all prompts into the plan WHEN enrichment can run and DEGRADES
on any bind failure — enrichment is an enhancement, never a
dependency; the enrichment states then skip with reason
`prompts_not_bound`.

### §5 — Co-occurrence proposes, corroboration confirms

Instantaneous speaker↔face co-occurrence is the weakest evidence and
confidently WRONG for whole classes (an anchor speaking over a photo
of the subject is perfect co-occurrence and the wrong answer — see the
§18 inventory). Therefore:

- **Aggregate across the whole asset**: a true pair co-occurs across
  many separate turns in different scenes; shot/reverse-shot noise
  averages out. Exclusivity weighting: a face alone on screen during
  speech earns full credit; concurrent faces divide it.
- **Variance is a classifier**: a voice whose co-occurring faces are
  diverse with no stable partner IS a narrator/VO/announcer
  (`voice_only`) — detected rather than mis-bound.
- **Negative evidence**: a face frequently on screen WITHOUT this
  voice weakens the pair.
- **Naming requires a second mechanism**: chyron-while-speaking,
  self-ID/vocative, credits, or sustained single-face exclusivity.
- **Depicted-face guard**: faces inside photos/monitors/file footage
  are excluded from speech binding. The guard credits LIVE spans only
  (face spans minus depicted shots, for EVERY face in the depicted
  shot): a depicted shot's face-speech pairing is untrustworthy for
  all faces in it, and excluding only the depicted face would hand a
  co-present live face undivided credit and a false confirm (the
  monitor-guest trap — a verified failure mode). The depicted-share
  threshold survives only for the `depicted_only` TYPE label, where
  majority-of-total is what the label means; a speaker whose only
  faces are fully depicted types `voice_only` (phone interview over
  their photo), never `on_camera_speaker`.

### §6 — Class-1 hierarchical matching: one binding engine

Exclusivity share structurally fails the two-anchor desk: every
co-shot divides its credit, so a share-based classifier can eat both
anchors' voices. Top speakers and top faces are CLASS-1 citizens,
matched FIRST as a small K×K assignment; everything else is class 2 on
the §5 mechanisms.

- **Selection is structural**: every voice/face carrying ≥ 4% of total
  speech / screen time (`CLASS1_MIN_SHARE` — a fraction of the total,
  never a fixed count; self-bounds K at 25).
- **Two angles: what it IS and what it is NOT.** Exclusions are
  near-certain and weightless, so they strike pairs from the matrix
  FIRST; assertion evidence is noisy, aggregates, and only ranks the
  survivors.
- **Eliminations**: F0×gender contradiction (diarization's per-cluster
  F0 median vs the face registry's gender; male/female bands with an
  overlap band that abstains, confidence-gated on voiced-sentence
  count), zero co-presence, and taken (one-to-one). Struck pairs carry
  their reason into the dossier.
- **Assertions**: FLOOR-holding share (during maximal single-speaker
  blocks ≥ 8 s the reverse-shot grammar quiets and the camera favors
  the speaker; a voice whose blocks show NO dominant face is
  narrator-shaped), ONSET following (the edit cuts TO the speaker as
  they start talking — a face appearance beginning within 4 s of turn
  starts; strong on scripted, weaker on news desk two-shots), presence
  LIFT (P(face | voice speaks) vs the face's base rate — immune to
  crowd division), SOLO cutaways, NAME join (token-subset; a mismatch
  is NOT a penalty — scripted content legitimately splits
  actor/character names), and F0 agreement as a small boost.
- **Assignment is one-to-one over the survivors**, confident evidence
  first (highest score with a real margin over the voice's next-best
  FREE face; co-anchor complementarity falls out of the structure). A
  forced-by-elimination endgame existed briefly and was DELETED: it
  produced the one wrong tier-1 assignment during development —
  eliminations act through strikes and margins, never a discounted
  bar.
- **Tier 2**: after the heads lock, voices with ≥ 5 turns (the
  corroboration constant) plus tier-1 abstainers run against unmatched
  class-1 faces and unmatched NAMED live faces — same confident bar,
  one-to-one, no forced rung, no dossiers. The class-1 floor means
  ORDERING, not eligibility. Matches carry `tier` 1|2.
- **One engine**: three subsumed mechanisms were deleted once the
  matcher proved itself (top-talker recovery, the co-occurrence
  CONFIRM path — cooccur now emits proposals and the narrator
  classification only — and the forced rung). Known hold: binding
  voices to UNNAMED faces waits on a twin-aware rule (people often
  exist as a named track plus an uncoalesced unnamed twin; the twin
  can outrank the named track and one-to-one then orphans the named
  face).
- **No silent exit**: every unmatched class-1 voice leaves a RECORDED
  abstention (reason + top candidates + vetoed pairs) in
  `persons.json`'s `class1` section — the adjudication round's
  dossier.
- **Heads fixed downstream**: cooccur discounts a matched face's
  credit toward other voices (×0.25 — its presence is explained by its
  own voice) and never classifies a class-1 voice `voice_only`.
- **Registry precedence**: a class-1 person whose merged cluster
  absorbed other spoken/overlay names keeps the sole registry name;
  the rest demote to aliases (mechanism `class1_name_precedence`). Two
  registry names disagreeing still abstains.

Known limit: on dramas whose shots carry overlay text, chrome leakage
can let the role-threading rung crown scripted leads "anchor" —
pre-existing chrome-gate behavior.

### §7 — Corroboration thresholds: rule shapes, judged pooled

Which evidence combinations suffice to EMIT a name vs mark `uncertain`
is decided by measurement against ground truth, and encoded as RULES
(evidence-combination requirements), never as a tuned scalar score —
the validation corpus is small and scalars would overfit it. All
decision constants in `lib/pb_common.py` are PROVISIONAL rule shapes
until that corpus gate; every constant states why it exists. Judging
is POOLED, never per-asset (single-asset recall swings make per-asset
numbers noise). Truth lists are positive-only oracles: a name they
don't cover is "uncovered", not "wrong" — the headline metric is
corroboration, deliberately not precision. Expected shape of the
answer: "self-ID or credits-corroborated bindings name; co-occurrence
alone proposes, never names."

### §8 — Focus reads: the uniform activation rule

Frame OCR goes through the frame-inspection witness service: `(t0, t1,
question) → dense frames at readable resolution → evidence JSON`,
reading the dense per-shot frame archives shot-segmentation retains —
no new extraction, no source access. `credits-read` and `title-read`
are its prompt categories; chyron reads are title-read WITH EXPLICIT
SPANS, not a new category. Cast-resolution is the CALLER: it owns
triggers, budget, the ledger, and evidence integration.

**The uniform activation rule: every read class needs (a) an
activation TRIGGER — free evidence that the target exists — and (b) an
unmet NEED — the names aren't already in hand. Never blind-read.**

Shipped read classes (`lib/reads.py:plan_reads`; every trigger an
observed event, no fixed intervals anywhere):

- **Chyron/title spans, three trigger classes**: transcript
  `introduction` cues and unnamed face debuts with ≥ 10 s screen time
  (`MIN_DEBUT_SCREEN_SEC`) each open an 8 s window
  (`CHYRON_WINDOW_SEC`), lower-third CROPPED; enrichment text-shot
  evidence outside the tail is read at the shot's own bounds, FULL
  frame — its text can sit anywhere. All pass the need-check
  (satisfied spans), the crop-aware ledger, and a `MAX_READ_SPANS`
  (40) budget keeping the head of the priority order.
- **Credits — evidence-or-hint, never blind**: text evidence in the
  tail 20% of the runtime (`TAIL_FRACTION`), or an explicit caller
  `credits_hint` (informed caller knowledge is a legitimate trigger —
  the rule forbids BLIND reads). The request goes SPANLESS: the
  witness derives credit spans itself from nondiegetic shot kinds
  (credit_roll / title_card / logo_bumper / slate, bridged across
  ≤ 5 s black). Only when shot metadata is absent entirely does a
  hinted read fall back to an explicit tail-20% span.
- **Transcript cues are both scheduler and evidence**
  (`lib/transcript_cues.py`): self-introduction ("I'm Miriam Okafor")
  binds a name to the CURRENT speaker with zero frames; sign-off
  ("Miriam Okafor, Metro News, Dallas") the same plus org/location;
  introduction ("joining us now is…") names the NEXT new speaker and
  predicts a lower-third at that speaker's first turn → schedules a
  read there; vocatives/handoffs ("back to you, Tom") are
  cross-speaker name evidence. Deterministic vocative resolution is
  adjacent-turn voting (≥ 2 votes, 2× lead, ≤ 10 s gap); capitalized
  interjections are dropped, and honorifics/role terms ("Detective",
  "Your Honor") become kind `address_term` — role evidence, never
  person names. The LLM round answers only "which window speaker is
  addressed?" for adjacency-unresolved sites, firewalled to window
  speakers and aggregated under the SAME vote thresholds; ASR spelling
  variants of one address are clustered before voting so variants
  don't split the tally below the vote floor. Single-site vocatives
  stay unresolved BY DESIGN — the vote floor is the precision guard.
- Credits sampling is a frame SELECTION question (the archives are
  dense), default interval ~3 s as a parameter, with a
  **name-continuity check**: consecutive credit frames must share ≥ 1
  OCR'd name; zero overlap ⇒ selection too coarse for this asset ⇒
  halve the interval over the gap and re-read. Self-correcting, no
  fitted constant.

Deliberate divergences from the fuller design: a program-structure
part-inventory trigger is not wired (the tail-evidence fallback IS the
shipped credits trigger), and uncertainty-driven escalation reads are
design, not code — today's budget is the span cap, not a marginal-gain
stop.

### §9 — The read ledger: no run ever re-pays

Dedup is on **evidence obtained, not frames seen**: a frame enrichment
saw for DESCRIPTION is not a frame "read" — a read is skipped only
when the NEED for that timespan is already satisfied (name + role
evidenced). The ledger (`lib/ledger.py`) is one JSON file per
UNDERLYING VIDEO, keyed by the shot-segmentation workflow_id (the
stable per-video handle every read is scheduled against) and shared by
all binding runs of that video, so idempotent re-runs and retries
never re-pay OCR. It is **crop-aware**: a full-frame entry satisfies
any need; a cropped entry satisfies only a cropped need — otherwise a
lower-third read would permanently satisfy a full-frame text-shot need
over the same window and never-re-pay would become never-read. Span
merges union time bounds and keep the WIDER frame need.

### §10 — Read cost posture

- **Frame stacking is the dominant saving and ships in the witness**:
  frames are packed many-per-record (IMAGES_PER_RECORD = 18), so the
  per-record prompt — the fixed cost — is amortized; task-dilution
  risk is handled by a per-frame keyed JSONL contract (one line per
  frame).
- **Batch is the default posture**: the witness defaults
  `use_batch=true` and pools `assets[]` to clear the Bedrock batch
  minimum across assets/categories in one job. Cast-resolution hands
  it ONE pooled request per run. `use_batch=false` is for a
  latency-sensitive run a caller is waiting on — at single-episode
  scale batch saves pennies and costs hours, so interactive runs go
  sync.
- **Lower-third cropping**: chyron-class spans request the witness's
  lower-third crop; measured materially smaller payloads and cheaper
  reads on the layout-class trial. The 0.40 crop fraction is
  STRUCTURAL, not fitted: broadcast lower-thirds nominally occupy the
  bottom quarter-to-third, plus a safety band — the failure modes are
  asymmetric (a too-tall crop costs a few extra tokens; a too-short
  crop clips the TOP line of a two-line super and fails SILENTLY,
  returning the role without the name). Opt-in per span, only for the
  trigger classes where the convention is strongest (introduction,
  face_debut); text-shot and credits spans stay full frame.
  Capability-gated on the witness advertising `span_crop` (see
  Gotchas). Parked ambition: a LEARNED per-asset region (split
  screens, non-standard bands via static-overlay temporal
  differencing) passed per span.
- **OCR resolution**: 512 px gives full token recovery on real credit
  frames — historical credits gaps were sampling density, prompt
  self-censorship, and task dilution, never resolution. Higher
  resolution at ingest is the recorded contingency for untested
  classes (fine scrolls with motion blur, stylized titles over busy
  picture).

### §11 — Credit pairing: witness-structured, never title-specific

Two locked rules:

1. **Credit-layout interpretation must never be title-specific.** The
   class of credit layouts is open (stacked pairs, reversed order,
   centered columns, non-English forms), and text-side column parsing
   survives only when the model happens to preserve the column gap as
   whitespace — the GEOMETRY is what pairs the names, and text lines
   lose it. So pairing is a WITNESS-level structured output: the
   credits read emits the NEUTRAL pairing the card visually shows,
   `{name, paired_with, header}` — never "character", which would bake
   a VERDICT into the witness (credits also pair crew roles: "Directed
   by" → MARK ELLISON). CLASSIFICATION is caller-side
   (`lib/reads.py`), from the header vocabulary ("Starring /
   Co-Starring / Guest Starring / Cast" ⇒ character; "Directed by / …
   Producer / Casting By" ⇒ crew role) plus billing connectors and
   case direction; crew entries become a contributors-with-roles
   product instead of being discarded. The two conservative text-side
   forms ("ACTOR as CHARACTER"; case-direction two-column) survive as
   FALLBACK for old products only — no further text-side layout forms
   will be added; a third would be the start of a per-title parser
   museum.
2. **Character names come only from evidence fusion, never from the
   actor's name.** actor→character requires either a STATED mapping
   (credits / cast sheet) or adjudication over the person's full
   dossier. Name similarity between an actor and a character is NEVER
   evidence — a first-name coincidence ("Billy" the character,
   "Billy Ray Marsh" the actor) would be right locally and wrong in
   general; a critic rule vetoes a `character_name` whose only support
   is similarity to the actor name.

The structural gap this design bridges: celebrity recognition names
PRINCIPALS (whose credit cards are name-only), while paired credit
cards name CO-STARS — two mostly disjoint populations. Bridges, in
preference order: opening-title principal cards; the default-disabled
external cast-sheet lookup (§17); and cast adjudication over vocative
cross-evidence.

### §12 — The critic: tooth 1 demotes, tooth 2 reports

**Tooth 1** (`lib/critic.py`, free, always on) is demote-only by
design: it never renames and never promotes — it vetoes a merge before
it happens or demotes a name to `uncertain` after assembly, always
with a machine-readable reason in `vetoes` and on the persons touched.
Checks: `speech_self_overlap` (a person cannot talk over themselves —
same-name speaker merges blocked above 1 s overlap),
`face_self_overlap` (looser 5 s bar — one person CAN be on screen
twice: live + monitor), `name_collision` (two distinct persons
finalizing with one name → both demoted). Some vetoes demote TRUE
names (file footage of a person who is also live) — that is the
designed trade under "wrong name is worse than none", and the real fix
is the depicted-face guard signal, not loosening the veto.

**Tooth 2** (`lib/critic2.py`) is a report-only LLM critique of person
dossiers, restructured after a first version failed exactly as
predicted: a general skeptic mostly emitted absence-based doubt
re-auditing an external oracle. The working shape:
contradiction-only prompt; evidence classed observed/inferred with
"never re-audit observed"; dossiers only for persons with something
that CAN contradict; a flag must cite ≥ 2 conflicting evidence ids or
the parser drops it. Flags land in `critic_review`. GRADUATION (flag →
demote) is deliberately not taken: tooth 2 stays report-only until
flags-on-corroborated-names is ~zero POOLED (a critic that vetoes true
names is worse than no critic), and adjudication can RESOLVE a
conflict instead of only demoting on it.

### §13 — Phases: free evidence first

P0 binds from FREE evidence only — transcript cues, face-source names,
enrichment overlay names, co-occurrence; no new reads, no LLM. It is
the mandatory core: the output contract is wrong without it, and it
sets the baseline every paid step must beat. P1 adds the witness focus
reads (credits, lower-thirds) with the §8 scheduler and §9 ledger. P2
adds the LLM seats: vocative resolution and cast adjudication. The
shaping rule: a step is MANDATORY only if the output contract is wrong
without it; everything that spends model money is skippable by flag
and priced in `usage_tracking.json`. Each LLM seat was built
harness-first and wired into the state machine only after it earned
its keep pooled (the same gate discipline as tooth 2's graduation).

Cast adjudication (`lib/adjudicate.py`) weighs each person's
`addressed_as` (vocatives/self-IDs), overlay names, and quotes against
the credits cast universe, behind a three-layer deterministic
fabrication firewall: (1) names verbatim from the universe, characters
verbatim from the pairs; (2) every assignment cites evidence ids; (3)
the AGREEMENT check — the assignment must share a token with the
person's OWN evidence names (verbatim-alone cannot see a wrong join;
quotes are excluded because a name in a person's own speech marks the
ADDRESSEE, not the speaker).

### §14 — Recognizer duplicate coalescing and the id trust hierarchy

Face recognizers can emit one person as two parallel entries (a
tracker-duplicate artifact: two tracks sharing the recognizer's own
collection face UUID with near-identical spans), which inflates
co-occurrence crowds, double-counts screen time, and triggers false
self-overlap vetoes. Fix at the producer boundary: `parse_faces`
COALESCES person entries sharing a non-empty collection face UUID —
union spans, screen seconds recomputed single-coverage,
`coalesced_ids` + typed `recognizer_coalesce` evidence as the audit
trail — so no consumer ever sees the duplicate. (A silent merge-veto
exemption at the consumer was rejected: person-wide, unaudited, and
blind to every cost except the veto.)

Coalescing runs over BOTH key spaces as a union — collection face UUID
and celebrity id, namespaced, never an or-precedence single key (which
silently hides a shared celebrity id behind the distinct fresh UUIDs
auto-indexing mints). The two key spaces are NOT equally trustworthy:
celebrity detection has a higher false-positive rate than SearchFaces
against a curated collection, so at a contradiction the collection
wins — a shared celebrity id never welds two entries the collection
distinguishes, and on a weld the collection-sourced name replaces a
celebrity-detected one, never the reverse. An uncontradicted celebrity
id still welds and still names. Products that carry no `celebrity_id`
make no claim (absent ≠ empty). The genuinely-depicted class (file
footage of a real person who is not in the room) is the depicted-face
guard's job (§5) — that person IS two distinct presences.

Same-name merging across mechanisms is separate and deliberate:
persons whose SOLE resolved name normalizes equal merge (mechanism
`name_agreement`) — e.g. an overlay-named `voice_only` speaker and a
collection-named `on_camera_speaker` both called "Mia Chen" are one
person surfaced by two independent mechanisms.

### §15 — Two-field attribution (interpreters, dubbing, narration)

Voice identity and content attribution are DIFFERENT questions, so the
utterance-level mapping carries both: `voice_person` (who produced the
audio) and `attributed_person` (whose words they are), with an
`interprets_for` / `dubs_for` binding between the two persons. Search
("what did the minister say") wants attributed; audio description and
speaker analytics want voice. One shape covers interpreters, dubbed
content, animation voice actors, and a narrator reading a character's
letter — no per-case policy needed; consumers pick the field.

### §16 — Faceless operation

Some privacy postures forbid facial recognition entirely, so
cast-resolution runs without a face-recognition input — verified
end-to-end, not just designed: the gateway and ValidateSources require
only one identity source; every consumer guards absent faces
(three-state); read triggers merely lose the face-debut class; chyron
overlay names bind to the single active speaker; `face_map.json` is
ABSENT (not empty); and the provenance pin records only the sources
actually used. Residual observation: speaker-only persons type as
`uncertain` rather than `voice_only` — the classifier needs face
evidence to assert voice_only. Acceptable, noted for the
corroboration-threshold work.

### §17 — External lookup for unknown faces (default DISABLED)

Gate: a face person with legitimate screen time (share-of-runtime
threshold, not a passer-by) still unnamed after all free evidence and
focus reads. Mechanism when enabled: NOT reverse face search on the
open web (policy-fraught, unreliable) — a TITLE-DRIVEN cast-sheet
fetch: resolve the program via a metadata provider, pull the cast list
and headshots, index them into a scoped face-recognition collection,
and let the EXISTING face-match machinery do the naming. Evidence
records `mechanism: external_lookup` with the provider, so consumers
can filter it out wholesale. Config: `external_lookup: {enabled:
false, providers: [...], min_screen_share}` — `enabled: false` is the
shipped default; enabling is a deployment decision (egress and
third-party terms are theirs). No silent network egress anywhere else:
the core binder touches only the deployment's own S3/DDB/Bedrock.

### §18 — Appendix: speaker↔face edge-case inventory

Grouped by failure class; §5 exists because of these. **A. Voice
without bindable face**: narrator/VO; announcer over set shots;
shot/reverse-shot (listener on camera); J/L-cuts (audio leads/trails
picture); phone/radio/intercom; reporter VO over b-roll (THE news
case); archival audio over unrelated footage; diegetic media (TV in
scene, PA); interpreter VO; dubbing/ADR; singing. **B. Face without
voice**: reaction shots; faces in photos/posters/monitors (perfect
co-occurrence, wrong — `depicted_only`); file footage of the discussed
person; sign-language interpreter inset; extras; press-conference
crowds. **C. Multi-face**: split screen; panels/roundtables;
two-shots; walk-and-talks. **D. Person ≠ one identity**: one actor
many characters (twins/disguise/flashback); one character many actors
(ages/recast); masks/prosthetics (face hidden while speaking);
animation/mocap. **E. Source-ID errors**: speaker cluster splits
(phone-filter voice, shouting/whisper); cluster merges; overlapping
speech; face-track fragmentation (profiles, low light — correlates
with dialogue scenes); celebrity false positives / look-alikes;
in-plot impersonation. **F. News-structural**: chyron names the
story/location/sponsor, persists past the speaker, lags, or is absent
on repeat soundbites; remote frozen video/photo cards; studio
monitors; green-screen presenters. **G. Episodic-structural**:
"previously on" recaps (other episodes' faces/voices); dream/flashback
recasting; cold-open one-offs never named; credits over theme (names
without faces or speech).
