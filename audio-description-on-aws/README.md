# audio-description-on-aws

Micro-workflow: four sibling run pointers plus ONE people source
(face-recognition, or a cast-resolution run — which also anchors the
name map with adjudicated identity bindings) in, a narrated
audio-description (AD) script out — a timed cue script with synthesized
narration takes, delivered as srt/vtt/DAPT/review.html plus a
`render_manifest.json` that media-processing's `ad_mix` output group
turns into the ducked, loudness-normalized AD track. **This workflow
renders no audio and runs no ffmpeg** — synthesis is text-to-speech's
job, the mix is media-processing's, and the fit in between is pure
arithmetic over tts's own reported `duration_s`.

The pipeline is: understand (name map + watch-through notes) → plan
narration windows (pure code) → chunked writer + critic → TTS
synthesis/fit with a bounded reask loop → a sequential defect-repair
chain → coverage oracles + a derived two-axis cue ledger. The stage
graph was validated by repeated zero-hand-edit runs scored against a
corpus of professionally authored AD cues before being ported here;
measured lessons travel as comments on the rules they produced, and the
full rationale is the numbered [Design record](#design-record) below.

## What audio description is

Narration describing on-screen visual content for blind and low-vision
audiences, placed so it never competes with dialogue. Two delivery
models coexist:

- **Broadcast-mix (the streaming standard)**: the deliverable is a
  COMPLETE alternative audio track — original mix plus narration,
  professionally mixed. The original track is never modified; the AD
  track is an additional rendition the player exposes (HLS
  `CHARACTERISTICS="public.accessibility.describes-video"`, DASH
  `Role=description`, SMPTE MCA **VIN** label in IMF/MXF).
- **Receiver-mix (the DVB broadcast model)**: narration travels as a
  separate low-bitrate stream plus fade/pan control metadata and the
  receiver performs the mix.

Placement and ducking solve different problems and both are standard
practice: narration goes in DIALOGUE-free windows (dialogue is never
talked over), but real programs are almost never *silent* in those
windows — so the music/effects bed is dipped a few dB under each cue
and restored after, inside the AD rendition only. Style guides (ITC,
Netflix) treat the dip as normal mixing craft.

Specs followed: W3C **DAPT** (TTML2 AD profile) for script
interchange; the **Netflix AD Style Guide**, UK **ITC** guidance and
**ISO/IEC 20071-21** for content rules; EBU **R128** / ATSC **A/85**
loudness on the final mix. Core style rules the prompts encode:
present tense; describe only in dialogue gaps; what/who/where at scene
entry; read on-screen text; name characters once established; never
interpret, never spoil ahead of the picture.

## The workflow

```
  INPUTS (4 pointers + 1 people source + opt)                      PROMPTS (bound by name
  ─ shot-segmentation      ─ conversation-analysis                  from prompt-registry)
  ─ shot-metadata-enrich   ─ face-rec OR cast-resolution            ad-name-map  ad-writer
  ─ speech-diarization     [audiostem] [decisions]                  ad-watch-notes
                           [cast-resolution persons + credits — name map + tail cards]
                                                                    ad-critic    ad-repair
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. VALIDATE AND PLAN                                     refuse-before-spend        │
 │    probe supplied artifacts (cast: one GET) · bind 5 prompts · pin voice            │
 │    probe tts describe_tool.push_completion  ──►  refuse by name, never a hang       │
 │    writes _plan/plan.json — every later state reads only this                       │
 └───────────────┬────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌────────────────────────────────────────────┐
 │ 2. BUILD NAME MAP + WATCH NOTES  (loop) ⟳  │   bedrock #1: NAME MAP
 │    face keys → speakable character names,  │    (evidence → character_map.json)
 │    then the describer's first full watch   │   bedrock #2: WATCH-THROUGH
 │    (beats, reveals, tone, synopsis)        │    (record built WITH the name map,
 └───────────────┬────────────────────────────┘     so #2 must follow #1)
                 ▼
 ┌────────────────────────────────────────────┐
 │ 3. PLAN NARRATION WINDOWS      (pure code) │   no model: gap map between speech,
 │    speech gaps → windows, word budgets,    │   entry-duty ladder, sung-vocal
 │    scene-entry duties, reveal snapping     │   policy (narrate|avoid)
 └───────────────┬────────────────────────────┘
                 ▼
 ┌────────────────────────────────────────────┐
 │ 4. WRITE SCRIPT                  (loop) ⟳  │   bedrock: WRITER — all chunks as
 │    chunked draft → critic audit →          │   records in ONE run (parallel inside
 │    prune hallucinated ids → cues.json      │   bedrock-inference) → CRITIC pass
 └───────────────┬────────────────────────────┘
                 ▼
 ┌────────────────────────────────────────────┐
 │ 5. SYNTHESIZE AND FIT     (loop ⟳ + Map ⇉) │   tts: shards of ≤90 cues, ALL
 │    tts shards in PARALLEL → fit each take  │   shards in parallel via the
 │    into its window (dur = raw/tempo,       │   fan-out Map — wall-clock is max,
 │    tempo-capped); over-budget → bounded    │   not sum. fit is ARITHMETIC over
 │    REASK (rewrite shorter, re-fan) →       │   tts's own duration_s — no audio
 │    collide → placed.json                   │   opened. unplaceable → drop
 └───────────────┬────────────────────────────┘   register with its measurement
                 ▼
 ┌────────────────────────────────────────────┐
 │ 6. REPAIR DEFECTS 1-BY-1  (loop ⟳ + Map ⇉) │   ORACLES (code) find defects:
 │    one defect at a time, PASSES stay       │   false text claims · name leaks ·
 │    sequential: oracle finds → bedrock      │   redundancy · uncovered beats ·
 │    rewrites under a HARD word cap → code   │   tail blocks · descriptor joins
 │    re-fits, re-collides, checks coverage   │   unfit/harmful answer → reverted
 │    neutrality → accept | revert | drop     │   or dropped IN CODE. a pass's
 │    → re-tts (shards fanned in parallel)    │   re-tts shards fan out like #5's
 └───────────────┬────────────────────────────┘
                 ▼
 ┌────────────────────────────────────────────┐
 │ 7. FINALIZE → 8. USAGE TRACKING            │   srt · vtt · DAPT · review.html ·
 │    two-axis cue ledger (every cue, every   │   cue-ledger · gaps ·
 │    violation, disposition derived)         │   render-manifest ──────┐
 └────────────────────────────────────────────┘                          │
                                                                         ▼
  THE MIX IS NOT THIS WORKFLOW'S JOB (renders no audio):   media-processing
  the render-manifest output feeds a chained process_media  `ad_mix` output group
  → ducked, loudness-normalized narration+bed wav           (concat FLAC bed, per-take
                                                             atempo, sidechain duck)
```

The stage graph is **data-dependency-serial, not lazily serial**: the
watch-through record is built with the name map, windows take their
duties from the watch-through's beats, the critic reads the draft, and
the repair chain's PASSES are deliberately sequential (each accepted
repair changes the placements the next repair is collided against).
Fan-out lives everywhere the data allows: the chunked writer sends
every chunk as a record in one bedrock run, tts Maps 40-wide inside
each shard, and the tts SHARDS themselves run in parallel through the
two fan-out Map states (measured on a long episode, back-to-back
serial shards had put nearly half the run's wall clock in SynthFit).
The four token-heavy bedrock legs (namemap / watch / writer / critic)
pass `use_batch: True` — the batch-inference token discount, and the
26 h state timeouts cover a batch job's worst case. Small serial legs
(repairs, reasks) omit the flag so bedrock-inference's dollar guard
keeps them sync — a padded batch job per tiny record would trade hours
for sub-cent savings.

**Judgment lives in the prompts, verification lives in code.** Every
model answer (name map, cues, repairs) passes a mechanical gate — parse
salvage, id pruning, arithmetic fit, collision, coverage neutrality —
before it can affect the script. A bad model day degrades quality; it
cannot corrupt the run.

## Layout

```
cfnstack.yaml         deploy truth (inline DefinitionString) + resources
definitions.json      state machine readability copy — edit BOTH, the
                      build fails on drift (TestStateMachineParity)
lib/                  PURE modules (parsed data in, data out — tests need
                      no AWS): ad_common (constants, one home), windows
                      (gap map, entry ladder, reveals, sung-vocal
                      policy), oracles (mechanical repair triggers),
                      coverage (the deletion-neutrality guard), fitting
                      (tts fit, collide, collateral), ledger (two-axis,
                      disposition derived), deliver (srt/vtt/DAPT/
                      review), evidence (sibling fetch), submit
                      (sibling submissions + write-through register),
                      loop_state (the re-entrant loop contract)
states/               operation-routed handlers behind one workflow
                      Lambda; the multi-submission stages are LOOP
                      states (see the loop contract)
gateway_tool/         generate_ad / get_job_status / describe_tool
prompts               live in prompt-registry-on-aws as five builtin
                      categories: ad-writer, ad-critic, ad-repair,
                      ad-name-map, ad-watch-notes
test/                 measured incidents as synthetic fixtures
```

## The loop contract

Every multi-submission stage is ONE `waitForTaskToken` Task + ONE
Choice:

```
        ┌─────────────────────────── re-enter ───────────────────────────┐
        │                                                                ▼
   ┌────┴─────┐  callback payload   ┌──────────────┐   no ad_loop_done  ┌──────────┐
   │  Choice  │◄────────────────────│ Task (state)  │◄──────────────────│ sibling  │
   │ ad_loop_ │                     │ waitForTask-  │  token completed  │ callback │
   │ done?    │                     │ Token, 26h    │  by tts/bedrock   │ (waker)  │
   └────┬─────┘                     └──────┬───────┘   WAKER            └──────────┘
        │ present                          │
        ▼                                  │ Lambda submits with callback_task_token,
   next stage                              │ checkpoints to S3, returns {waiting}.
                                           │ When ITS OWN work is done it calls
                                           │ SendTaskSuccess({ad_loop_done: true}).
                                           └─ S3 checkpoints are the truth; the
                                              callback payload is a signal, never data
```

- A sibling callback re-enters the same state (its payload carries no
  `ad_loop_done`); the Lambda reads its S3 checkpoint to know where it
  is. No Choice ever depends on a sibling's payload shape.
- Results are read from S3 only — tts persists `outputs.json` at the
  submit handle's `output_s3_prefix` (its callback waker writes it, a
  FAILED run gets a `{status, error}` doc at the same pointer), bedrock
  at its `output_s3_prefix/response.jsonl`.
- Redrive IS the reconcile: a re-entered finished loop self-completes
  immediately (`phase: done` checkpoint).
- `_plan/submits/{kind}-{tag}.json` is the replay-safe write-through
  submit register, ONE ROW PER FILE — the fan-out Map registers shards
  concurrently, so rows never share a document (a single-list
  read-modify-write loses rows), and a whole-file overwrite per tag
  makes a replay free (an absorbed-terminal submit resolves its token
  synchronously inside the gateway call, so a phase can re-run).

**Fan-out extension.** A loop with N independent tts shards cannot wait
for N callbacks on one token (the first callback consumes it; the rest
are benign `TOKEN_GONE` no-ops and the state sleeps to its 26 h
deadman). So the loop resolves its OWN token with
`{tts_shard_tags: [...]}` — tags only; every shard's `{order, seed}`
spec is already in the checkpoint — and the Choice routes through an
inline Map (`Synthesize shards in parallel` /
`Re-synthesize repaired shards`, `MaxConcurrency: 0`; tts is
Lambda-driven with no service throttle). Each iteration is its own
`waitForTaskToken` Task: the shared `SubmitTtsShard` handler re-reads
the checkpoint by tag and submits ONE shard against the ITERATION's
token. Iterations write nothing shared (only their own register row).
When the Map completes, its ResultPath is discarded and the loop state
re-enters. `ad_loop_done` is checked BEFORE `tts_shard_tags` in both
Choices, and sibling payloads carry neither key, so a sibling callback
still re-enters the loop state.

The re-entered fit/apply leg is TOLERANT by design
(`submit.collect_fanout` — a strict read wedged every redrive when the
fan-out itself was lost):

- **Missing register row / outputs doc, or a bumped attempt that never
  landed** → the loop re-issues the SAME fan-out. Landed shards absorb
  at the gateway and complete their iteration tokens synchronously, so
  re-fanning everything is free. This is what makes the
  checkpoint-then-resolve write order safe (the specs MUST be durable
  before iterations read them) and closes the register-row race (the
  row is written after the gateway call that can consume the token).
- **A FAILED shard** → resubmitted under a fresh `-r{n}` idempotency
  token (the plain token absorbs into the sticky failure forever),
  bounded by `TTS_SHARD_ATTEMPTS`; the bumped attempt is checkpointed
  BEFORE the re-fan and recorded on the register row, so a lost send
  re-fans without double-bumping. Exhausted attempts raise BY NAME —
  then, as for any sticky terminal failure, a `.version` bump is the
  retry. An absorbed-failure submit response (handle + error) does NOT
  raise in the iteration: the gateway already persisted the verdict and
  completed the token; the fit leg owns retry-vs-raise.
- **A pre-fan-out checkpoint** (serial `shards`/`shard` keys — an
  execution that crossed the fan-out deploy mid-loop) → refused by
  name; those runs are pinned to an ASL without the Maps and must be
  re-run, not redriven.

## Gateway tool contract

Three tools: `generate_ad`, `get_job_status`, `describe_tool`.
`workflow_id = uuid5(request + TOOL_VERSION [+ idempotency_token])`;
an idempotent resubmit returns the existing run. The output contract is
`gateway_tool/output_contract.json` (cue-ledger, windows, script
srt/vtt/DAPT, review page, gaps, render-manifest; everything under
`_plan/`, `_records/` and `takes/` is internal). The gateway is THIN by
design (identity only): validation that costs a read lives in
ValidateAndPlan. AD is prompt-bound, so a category prompt rebuild
re-runs via a fresh `idempotency_token` — prompt bytes are not in the
hash.

Request parameters:

- Required: `shot_segmentation_s3uri`, `shot_metadata_s3uri`,
  `speech_diarization_s3uri`, `conversation_analysis_s3uri`, plus ONE
  people source — `face_recognition_s3uri` or `cast_resolution_s3uri`
  (one home for the rule: `ad_common.people_source_error`, enforced at
  submit and validate) — and `voice` (a text-to-speech voice_id).
- Optional: `audiostem_s3uri` (music/sfx gating evidence — supplying
  it is a REQUEST for music gating, so a pointed-at run that cannot
  deliver `stem_activity.json` is refused by name), `decisions_s3uri`
  (reviewer-accepted absences stop re-reporting), and
  `cast_resolution_s3uri` as the sixth input even when face-recognition
  is supplied (adjudicated person bindings anchor the name map; its
  conditional `credits` output feeds the tail-card reads).
- `sung_vocal_policy`: `narrate` (default) | `avoid` — `avoid` closes
  windows over untranscribed sung vocals (speech-stem activity with no
  transcript words). An output-changing knob, so it enters the
  workflow_id preimage. Default `narrate` because the WORD-level gap
  map already closes windows over transcribed lyrics, and a hard veto
  starves windows on music-heavy programmes.
- `language` (default: the program's language as diarization reports
  it), `title` (delivery basename), `idempotency_token`.
- A `mix` parameter (or `audio_proxy_s3uri`) is REFUSED at the gateway
  with the media-processing `ad_mix` instruction — never a started
  execution doomed to fail. Likewise `frame_inspection_s3uri` was
  retired and is refused by name: frame-inspection is internal, and
  cast-resolution is both the only orderer of its reads and the
  publisher of the `credits` product.

## Key conventions

- **Consumer child**: three sibling gateway-Lambda ARNs arrive as
  required template parameters (`BedrockInferenceGatewayLambdaArn`,
  `PromptRegistryGatewayLambdaArn`, `TtsGatewayLambdaArn`). Sibling run
  ARTIFACTS are read straight from the proxy bucket by pointer; the
  ARNs are for submissions only. The submit brain stays the sibling's
  gateway — never `StartExecution` on a sibling directly.
- **lib/ is PURE** (parsed data in, data out — tests need no AWS);
  states/ are operation-routed handlers behind one workflow Lambda.
- **Constants have one home**: `lib/ad_common.py` (TTS_SHARD_SIZE=90,
  ONSET_PAD, WPS/WPS_REPAIR_CAP/WPS_REASK, TEMPO_CAP, REASK_ROUNDS,
  DEFAULT_MODEL_CHAIN env-overridable, SUNG_VOCAL_DEFAULT="narrate").
  Each constant carries its measured provenance as a comment.
- **Prompts are five prompt-registry categories** (`ad-*`,
  workflow-owned judgment CONFIG, bound by name at ValidateAndPlan — a
  missing category refuses the run, never falls back). Editing one is
  retuning this workflow; the edit gate is the corpus eval.
- **Copied modules** (`prompt_render.py`, `workflow_history.py`,
  `parse_partial_output.py`, `workflow_identity.py`): copied verbatim
  from their source-of-truth repos, never imported; byte-parity is
  pinned by the bundle suite. Per-consumer variation lives in the
  CALL, never in the file.
- **Generalize, never content-specific**: the validation corpus is
  tiny. Structural arguments over fitted scalars; the `_COMMON`
  stoplist is structural only (corpus-fitted entries were removed, and
  the incident is named at the definition).
- **The reference corpus's professional cues are the measurement
  oracle only** — cue text never feeds a workflow or a prompt
  (captions are validation-only, never a workflow input).

## Build & Deploy

```bash
bash ./build-deploy-s3-dist.sh --bucket <dist-bucket> --single-region
bash ./create_update_stack.sh --stack-name ad-dev --resource-prefix ad-dev \
  --proxy-bucket <PROXY> --bedrock-gateway-arn <ARN> \
  --prompt-registry-arn <ARN> --tts-gateway-arn <ARN> [--region eu-west-1]
```

Flat zip, no docker (pure boto3+stdlib). `--build-only` deliberately
leaves `%%BUCKET_NAME%%` in the template so `create_update_stack.sh`
refuses a never-deployed build by name. Bump `.version` (8-digit
`alpha.YYYYMMDD[a-z]`) before every deploy. Deployable standalone
(`ad-dev`) or as a bundle child; registered as a media-toolkit-gateway
target (`audio-description`, 3 tools).

## Tests

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

Fixtures are SYNTHETIC encodings of measured incidents — each names
the incident it pins. The suite also carries the review regressions,
the tts callback contract, and the ASL parity check. Tests gate
packaging (`build-deploy-s3-dist.sh` runs them first).

## Gotchas

- **tts GATEWAY parameter shape ≠ tts state-machine input shape.** The
  gateway takes `transcripts` as a list of plain STRINGS with flat
  `flow_steps`/`seed` keys; the state machine takes
  `[{"text", "target_duration_s"}]` under a `config` dict. An early
  live run failed at SynthFit because `tts_submit` sent the
  state-machine shape — and the nested `config` had been silently
  ignored, so seed/flow_steps were never reaching the model. When
  adding a sibling submission, read the sibling's *gateway* docstring,
  not its input schema.
- **A pre-callback tts deployment silently IGNORES
  `callback_task_token`** — the wait state would sleep to its 26 h
  deadman. ValidateAndPlan probes `describe_tool().push_completion`
  (capability, before any bedrock spend) and `tts_submit` checks the
  handle's `output_s3_prefix` (belt-and-braces). Requires a
  text-to-speech build with callback support.
- **Both the state machine and `definitions.json` define the ASL.**
  The template's inline `DefinitionString` is the deploy truth;
  `definitions.json` is the readability copy. `TestStateMachineParity`
  fails the build on drift — edit BOTH.
- **`workflow_id` is sticky on failure** — a failed execution name is
  reserved for 90 days, so a retry of the same request against the
  same TOOL_VERSION absorbs into the failure; bump `.version` (or
  change the idempotency token) after a code fix.
- **Loop states must register takes only on ACCEPT** (repair) and drop
  registers must carry the measurement — `cue_ledger.json` derives
  dispositions from the registers; a cue "not in the mix" with no
  register entry is the bug the ledger's `unplaced` invariant catches.
- **Every raw model answer is salvage-parsed** (`parse_jsonl_cues`,
  name-map salvage): null/array/non-numeric deviations land in `bad`,
  never raise. Hallucinated window_ids are pruned before they can
  KeyError downstream — after the spend is the wrong place to find out.

## Design record

The numbered sections below are the design history code comments cite
(`README.md Design record #N`). Numbers are stable — they predate this
README and are referenced from docstrings.

### #1 — The two delivery models

See [What audio description is](#what-audio-description-is).
Broadcast-mix first: this family produces the complete alternative
track; receiver-mix fade/pan metadata is a possible later emission.

### #2 — Specs and references

See the spec table in
[What audio description is](#what-audio-description-is). The script
interchange format is W3C DAPT; Netflix TTAL is the industry-adjacent
equivalent. Regulatory drivers: FCC CVAA quotas (US), Ofcom quotas
(UK).

### #3 — Why this family is well positioned

Every input the script needs is already a sibling workflow output, and
placement is exactly the id/time-referenced shape the family enforces
("times never pass through a model"): speech-diarization turns give
dialogue-free windows (confirmed speech-silent against the audiostem
speech stem); shot-metadata-enrichment gives per-shot description,
people, setting, on-screen text (reading on-screen text is an explicit
AD duty); scenes give the "what/who/where" re-establishment beats;
face-recognition / cast-resolution give who people are;
conversation-analysis gives program context; text-to-speech gives
seeded, reproducible synthesis; media-processing mixes.

### #4 — Window planning (pure code)

- **WORD-level transcript timings, not sentence-level** — this is
  load-bearing: a single ASR "sentence" can span tens of seconds of
  silence and hide an entire scene from a sentence-level gap map.
  Windows get a WORD BUDGET (~2.5 words/sec).
- **Scene-entry priority**: every scene start searches [-1 s, +10 s]
  for a usable window and accepts a MICRO window (≥ ~0.7 s usable, a
  2-word fragment cue: "Dim office.") that ordinary windows reject
  (≥ 1.5 s). Measured, this took scene-entry coverage from a small
  minority to nearly all scenes.
- **Fast-cut rule** (ITC/Netflix montage guidance — describe the unit,
  not each cut): cuts closer together than a speakable window share
  ONE entry cue; a burst of cuts collapses to a single thematic window
  ("a series of quick glimpses…"), never machine-gunned cues.

### #5 — QC oracles, all programmatic

ZERO overlap between any narration cue and any diarization speech turn
(the hard rule, framed on the id lattice, not on floats); every
rendered cue fits inside its window post-fit; loudness within spec on
the final mix (media-processing's job); coverage reported HONESTLY,
never padded — scene entries with no adequate window land in `ad_gaps`
(dense-dialogue programs legitimately have sparse AD).

### #6 — Script and fit decisions (each from a measured incident)

- **The script is the product as much as the audio** — it is what a
  human reviewer edits, so it ships as a labeled output, and the mix
  is reproducible from script + seed (tts determinism).
- **Extended AD** (pausing the video to fit longer narration) is out
  of scope — it alters program timing.
- **The model never touches time**: windows come from data; the model
  fills text into windows it is handed — it never emits a timestamp,
  never invents a window. Cues are window-id-referenced JSONL; empty
  windows are legal (nothing worth saying is a valid answer). Overflow
  is fixed by shortening text, never by moving or stretching audio.
- **Character names EARLY, pronouns almost never across cue
  boundaries.** The he/she ambiguity is a known AD failure mode and
  the industry answer is uniform (DCMP: "when in doubt, repeat the
  name"; Netflix permits naming before dialogue introduces a character
  when withholding would confuse; ITC audience research found blind
  viewers prefer early naming). As implemented: actor names from face
  recognition are identity KEYS, never narration text; the prompt
  supplies a NAME MAP with evidence; dialogue-given names beat
  credits-only names; a character whose anonymity is the plot keeps a
  consistent descriptor, never an invented name; and EVERY CUE
  RE-ANCHORS its subject — cues are heard in isolation against program
  audio, so none may open with a bare pronoun.
- **Reveals under dialogue → describe in the NEXT gap**: a window only
  "sees" the shots it overlaps, so a plot-critical event entirely
  under speech was invisible. Windows carry `recent_shot_ids` (shots
  that ended within ~3 s before the window) tagged JUST BEFORE, with
  the instruction that a plot-critical recent event is described FIRST
  — slightly late is correct; missing it is not (for a reveal, late is
  the only spec-legal placement anyway).
- **Anchor rule — the picture leads the voice**: placing cues at
  window start let audio lead the visuals. Every window gets an
  `anchor` = scene cut + 0.1 s for entry windows; a cue never starts
  before its anchor. If the anchored remainder cannot legally hold any
  synthesized cue, the window is DROPPED and the next window carries
  the scene — never a lead, never an overlap.
- **TTS text normalization is its own stage, split from display
  text**: measured failures — a verbatim ALL-CAPS production card made
  the TTS spell letters, and a bare year was read as a cardinal
  number. The cue carries TWO texts: `text` (display — subtitles keep
  the year as digits) and a derived `tts_text` (recased caps-runs,
  years to spoken form, likely abbreviations), applied after scripting
  so the reviewable script stays verbatim.
- **Shot descriptions are WHOLE-SHOT summaries — a cue at a shot's
  start must describe the shot's INITIAL state.** Enrichment watches
  the full shot, so a long shot's description can state a development
  that happens well into it; a cue placed at the shot's start voices
  the future. Hence a FRAME-VERIFICATION QC pass: a frame at each
  cue's playback time checked against the text ("is every claim
  visible now?") — in prototyping, the cheapest QC in the design and
  the one that caught the most defects. Credit rolls are the extreme
  case: read each card during the window it is ON SCREEN (card times
  verified from frames), never as an early summary block.
- **In long dialogue-free stretches, window granularity = the SHOT**,
  not a fixed subdivision — fixed-length windows merge shots and a cue
  read at the window start narrates cuts the viewer has not reached;
  the drift compounds across a long music-only span. Placement
  principle, stated once: **description syncs to its subject's ONSET —
  as-it-happens, never before.** One cue per shot, placed at that
  shot's cut; shots under ~2.5 s attach to the previous window as
  "cutaway just after" evidence; leftover silence under the score is
  deliberate, not a gap. END-alignment (start = span_end − cue_dur)
  was rejected as the default — it inverts the lead into a lag — but
  remains right for exactly two cases: anticipatory placement of a
  scene-entry cue in the PREVIOUS gap's tail when no gap exists after
  a cut (ITC-sanctioned, never for reveals), and finishing a cue flush
  against resuming dialogue.
- **Atmosphere is describable appearance**: enrichment already emits
  shot-level atmosphere signals (`sentiment`, `lighting`) — pass them
  into the script evidence (free, the data exists). What metadata
  genuinely lacks (appearance-as-atmosphere: pallor, a smile) is NOT a
  new per-shot enrichment field — enrichment is the family's cost
  center — the frame-verification pass DOUBLES as atmosphere
  enrichment, paid only for windows AD actually describes.
- **Honest-evidence boundary on identity claims**: a cue may only
  claim identities the evidence establishes for THAT frame;
  cross-scene callbacks stay generic ("a figure on a beach"), even
  when a human viewer would infer the connection.
- **Fragment cues are a real tier**: TTS has a fixed utterance
  overhead, so word budgets alone under-predict duration on short cues
  — the fit stage measures real reported durations and sends
  over-tempo cues (> TEMPO_CAP) back for SHORTER text (bounded,
  counted). The AD track ships subtitle siblings (`.srt`/`.vtt`,
  `[AD]`-prefixed) so a sighted reviewer can see placement.

### #6b — Generalization lessons (second and third series)

- **The priority ladder is CONTEXT-SPLIT, not global**: who-first is
  right mid-scene and WRONG at a scene cut — entries lead with PLACE
  ("The restaurant."). Enforced mechanically: a REQUIRED-OPENER
  evidence line carries the setting into the entry window, and a
  CRITIC pass audits the finished script (entries not leading with
  place; filler cues that only restate what the audio conveys →
  emptied). Dense drama correctly loses a large share of candidate
  cues to the filler rule.
- **LATE ENTRY**: a scene whose cut has no usable gap grounds its
  place at the scene's FIRST window instead of never.
- **Gaps split at scene cuts** — a window spanning a cut hides the
  entry (the anchor rule needs a boundary to anchor to).
- **character_map.json derives cleanly** with evidence — dialogue
  mentions, on-screen supers, credit cards, known-role — including
  honest nulls for unnamed minor cast.
- **Scale pins**: tts caps 100 transcripts per batch (shards are ≤90
  cues); the script stage batches ~18 windows per record; wall clock
  is dominated by TTS.
- **Masks/costumes stale the face evidence** — identity claims under
  masks must degrade to descriptors; the frame-verification pass is
  the systematic catch.

### #6c — The watch-through pass

The one thing the human workflow has that a per-window pipeline lacks:
the describer watches the whole film BEFORE writing a word. The
equivalent is a stage-0 comprehension pass — one long-context LLM call
over the timecode-INTERLEAVED record (shot descriptions × dialogue ×
scene boundaries × music texture, "reading the film like a shooting
script"). It emits the DESCRIBER'S NOTES, shared context for every
downstream stage: synopsis + act structure; character dossier;
a REVEAL LEDGER (what the audience does not yet know at each point —
the only way "never foreshadow" can be enforced rather than hoped); a
PLOT-CRITICAL BEAT LIST with timestamps powering a completeness ORACLE
(every beat covered by some cue, else `ad_gaps` with a reason — a
structural check, not a per-video patch); motif/continuity sheet; tone
register. Where the text stream is SUSPICIOUS — an activity spike in
metadata the description does not explain — the notes mark beat
candidates for targeted VISION SPOT-CHECKS (frame pulls at those
timestamps only). That is how "watching" gets eyes without paying
vision over the whole program.

### #6d — Evidence sufficiency: consume what siblings already ship

Measured finding: AD mostly UNDER-CONSUMES existing outputs rather
than needing new ones. shot-segmentation's per-shot motion/entropy/
dispersion stats make intra-shot events visible for free (event
candidates, long-shot subdivision); face-recognition's UNNAMED cluster
people give consistent descriptors for uncredited characters across
the program; WHEN an on-screen card shows within a shot is the
frame-verification pass's job, not a new enrichment field. The ONE
enrichment addition that passed measurement is `state_change`, bounded
three ways: (1) GATED EMISSION — the field exists only when the shot's
begin and end states differ materially, null otherwise, costing zero
output tokens for most shots (protects the cost center); (2) FREE
TEXT, word-capped (~15 words) — no downstream consumer needs to BRANCH
on change kind, they need the words (vocabularies are for joins; no
join wants this); (3) ONE controlled slot that IS for joining: `when`
in {early, mid, late} — honest to sampled-frames evidence (the model
cannot emit reliable intra-shot timecodes). A motion-stat hint added
nothing measurable (motion flags kinetic shots; the burn cases are
slow changes) and was deferred: slow changes are carried by
first/last-frame comparison, kinetic events falling BETWEEN frame
samples remain the watch-pass + vision spot-check's job.

### #6e — cast-resolution as the sixth input

The name map was re-deriving person identity from raw hints — face
keys, name-like dialogue lines, on-screen text — evidence the
cast-resolution workflow now adjudicates deterministically
(corroboration-gated bindings: face + speech + chyron/credits reads,
demote-only critic). Wired as an OPTIONAL pointer:

- **The LLM round stays load-bearing.** persons.json carries WHO; the
  name map still decides speakable names, concealment, and the naming
  policy (`speakable_from_sec`). Bindings enter the record as an
  "ADJUDICATED PERSON BINDINGS" section ABOVE the raw hints,
  instructed as established identities — add freely, contradict only
  on explicit on-screen proof. Named persons only, capped at
  CAST_BINDINGS_CAP (40) by screen+speech time (long assets ship
  hundreds of persons, most unnamed silent faces the name map has no
  use for).
- **Provenance is refused, not warned** (refuse-before-spend):
  bindings are only valid against the exact sibling runs they were
  computed from — SPEAKER_IDs and face person_ids are per-run labels.
  validate_plan downloads persons.json (the one validate-time GET) and
  pins on the producer's DOCUMENTED consumer key,
  `provenance.sibling_workflow_ids`. uri string-compare was reviewed
  and rejected: containment heuristics false-pass parent-folder
  spellings and false-refuse mirrored buckets; the workflow id has
  neither hole. A doc without the pin passes — absence of the claim is
  not a contradiction — and a cross-repo lock-step test parses
  cast-resolution's output contract, pinning every consumed field, so
  a producer field rename fails a build instead of failing open
  (belt-and-braces with the producer's own always-run pins; this test
  skips on a single-repo checkout). The refusal happens AT VALIDATION
  inside the run; the gateway is thin by design, so the submit still
  returns a handle (prompt and schema say so explicitly).
- **In-flight cast runs are narrowed, not closed**: persons.json
  exists mid-pipeline and is REWRITTEN by cast-resolution's enrichment
  state, and nothing the producer writes today is truly terminal. The
  probe requires persons.json AND usage_tracking.json — refusing
  pre-Bind pointers — and the schema/prompt instruct callers to poll
  the cast run to SUCCEEDED before submitting. A producer-side
  terminal marker is the eventual fix if the window ever bites.
- **One GET, forwarded**: the filtered payload (named persons, top
  CAST_BINDINGS_CAP by screen+speech, evidence rows stripped) rides
  plan.json as `cast_bindings` — no later state re-downloads or
  re-filters the full doc. The parse side prunes name-map characters
  without a face_key (the hallucinated-id KeyError class), and every
  OPTIONAL pointer is probed for the artifact THIS workflow consumes
  (a typo'd pointer used to sail through validate and silently degrade
  after full spend).
- **Absent = exactly the old behavior** (three-state: the section
  never renders, no probe, no GET).
- **cast-resolution can be the SOLE people source**: the producer
  exports person-level `appearances` (member faces' spans merged, the
  facemetadata ms shape), AD synthesizes the facemetadata-people shape
  from persons.json (appearances required; a producer too old to
  export them is refused by name when it is the sole source), and the
  cast block keys faces by the binding's own name. A FACELESS cast run
  (its provenance pins carry no face_recognition) is also accepted:
  fetch degrades presence timelines HONESTLY to merged speaker-turn
  spans (`_speech_basis_people`), keys ride the named voices, windows'
  people_on_screen goes near-empty by design (narration windows are
  speech gaps — in a faceless world nobody knows who is on screen in
  silence), and visuals ride enrichment's descriptors. This is the
  GDPR posture: deployments where no facial recognition may run still
  get AD.

Rejected: replacing the name-map LLM round with a deterministic
character_map derivation from persons.json. For episodic content the
map's real work — concealment detection and early-naming policy — is
judgment over the WATCH evidence, not identity resolution; and
persons.json abstains by design where AD still wants a descriptor
("the detective"). Revisit only if the corpus gate shows the anchored
LLM adding errors to adjudicated identities.

### #7 — Settled parameters and parked questions

Settled:

- **Script format: both, JSONL canonical.** JSONL is what the pipeline
  stages read/write and what the eval harness diffs; DAPT is an EXPORT
  in deliver.py alongside srt/vtt, for interchange with professional
  post tooling.
- **Voice: caller parameter** with a catalog default. **Language:
  caller parameter**, defaulting to the program's language as
  diarization reports it.
- **Sung-vocal window veto: configurable, default narrate-over** (see
  the gateway contract for the rationale).
- State names are semi-API (agents read `current_state` in polls), so
  loop-plumbing states carry readable names (`Build name map and watch
  notes`, `Plan narration windows`, `Repair defects one at a time`,
  `Synthesize shards in parallel`, `Re-synthesize repaired shards`,
  question-named Choices); renames land as their own change — they
  touch both ASL homes, the router's mapping, and agent-facing docs.

Parked (wait for a real occurrence before building):

- **Credits-only acceptance**: credits.json is run-independent card
  text, yet it rides a pointer whose provenance gate covers the
  per-run-label bindings — re-running diarization strands an existing
  cast run's credits for AD. If it bites: scope the provenance refusal
  to the bindings and accept the credits rider regardless.
- Exact dip depth / narration-vs-bed loudness offsets live in
  media-processing's `ad_mix` group, taken from current delivery
  specs, not here.

### #8 — The workflow shape

The stage graph is validated; three internals are deliberately NOT
frozen and the design must not lock them in:

1. **Graded speech-clearance** will change window generation
   (candidate mechanical predicate: speech-stem energy with NO
   transcript words = untranscribed walla that professional describers
   talk over).
2. **Windows/orientation may migrate to program-structure-analysis**
   (an ad break wants the same seams a scene entry wants) — they live
   in lib/, not spread through states, so the later move is a lift.
   Likewise `oracles.nondiegetic_blocks` is a temporary copy: the ONE
   home for that predicate is program-structure-analysis'
   `lib/markers.py`, and this copy serves the tail oracle until the
   consumer migrates.
3. **Watch-notes churn** between identical rolls is a prompt-stability
   problem the registry freeze addresses, not the shell.

Prompts are FIVE registry categories: `ad-name-map`, `ad-watch-notes`,
`ad-writer`, `ad-critic`, `ad-repair` (ONE category — the repair
step's tag parameterizes the ask:
beat/reveal/descriptor/textclaim/nameleak/tailblock/dedupe).
`credits-read` lives in frame-inspection and is consumed here via the
cast-resolution run's conditional `credits` output — frame-inspection
is internal, and cast-resolution is both the only orderer of its reads
and the publisher of the product.

Design decisions carried from measurement (each traces to a recorded
incident): evidence/verdict split everywhere; deletions only through
the coverage guard (`lib/coverage.py`, the neutrality check every
deleting stage must pass); ONE repair failure policy (`drop_on_fail`
governs "unfit" AND "longer-with-collateral"); `drops.json`
write-through so the ledger re-derives from a finished prefix;
`decisions.json` input so reviewer-accepted absences stop
re-reporting; density/overlap calibration cites the professional-
practice envelope (2.7–4.7 cues/min, 21–39 % of runtime), never a
fitted scalar.

The EVALUATION harness (professional-cue extraction + diff against
professionally described references) stays offline tooling by design —
it judges the workflow and must not live inside it.

### #9 — Acceptance gate

The port's acceptance gate is a reference-corpus diff of the
WORKFLOW's output against the validated runner's (recall within the
measured noise floor = ported correctly); graded speech-clearance then
lands as a windows.py change behind the same gate. Judge the gate
POOLED across the corpus — single-episode recall is noisy.

### #10 — Chunking rule

Every long-running leg splits into bounded, checkpointed, parallel
units:

- `write_script`: a Map over window CHUNKS (~18 windows/record); the
  critic is one record but chunkable by the same key if scripts grow.
- `synth_fit`: TTS shards ≤90 cues per submission, shards run as
  parallel submissions; fit is per-cue and embarrassingly parallel
  within an iteration.
- `repair_chain`: SEQUENTIAL BY DESIGN across steps (each step reads
  the previous step's placed set — parallelizing steps would race the
  coverage guard), but WITHIN a step all targets go as one batched
  submission and one TTS shard set; the checkpoint is per STEP, so a
  re-entry resumes mid-chain, never re-runs completed repairs.
- `understand` (watch notes): the one deliberately UNCHUNKED call —
  the long-context read is the point (the describer's single sitting);
  bounded instead by the interleaved record's token budget.
