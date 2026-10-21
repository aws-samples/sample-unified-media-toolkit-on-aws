# Phase 9 measurement harness

Offline replay scripts for the recurrence gates. **Not unit tests** — they run
against production artefacts pulled from S3, need AWS credentials, and take
minutes. The unit tests are `test/recurrence_gates_test.py`; these are the
*method of record* for the numbers behind the README.md §9 defaults, checked in because a
previous set was lost to `/tmp`.

`test/` is not a package and there is no `__init__.py` here, so
`python -m unittest discover -s test -p 'recurrence*'` does not touch these.

## Why offline replay

Every script imports `lib.recurrence` and calls `detect_recurrence` on
production `shot_centroids.npy`, so the replay runs *the shipped code path* on
*the shipped data*. That lets a gate be re-run with its threshold changed
without re-transcoding a 25 GiB mezzanine. The one thing replay cannot
reproduce is per-frame entropy (not published), so 9.4's entropy floor is
absent offline and only black-dominance filters frames — which leaves *more*
hashable frames than production and biases 9.6 replays toward CONFIRMING. A
rejection offline is therefore a fortiori a rejection in production.

## Setup

Artefacts are large and untracked. Work in a scratch directory, never in the
repo:

    mkdir -p ~/scratch/phase9 && cd ~/scratch/phase9
    <repo>/test/harness/setup_workdir.sh          # all three batches
    <repo>/test/harness/setup_workdir.sh paytv-batch   # or just one

That produces the layout the replays expect:

    <batch>/runs.txt
    <batch>/<run>.shots.json
    <batch>/<run>.recurrence.json
    <batch>/<run>.centroids.npy

Corpus manifests (`<run_name> <workflow_id>` per line) live in `runs/`:

All manifests were **re-baselined** to fresh post-swap runs
(post: complete linkage at 0.07, annotate-not-drop, trimmed contract,
`hash_confirmed`, dispersion fields). The pre-swap artefacts referenced by the
historical findings below came from the previous manifests (git history).

| manifest | assets | character |
| --- | --- | --- |
| `phase9-batch.txt` | 8 | the original trial — drama_e, drama_a, reality_a, reality_c, news_b, markets_a |
| `paytv-batch.txt` | 6 | pay-TV broadcast masters, incl. timecode-burn and 2-hour blocks |
| `studio-batch.txt` | 14 | two studio ingest prefixes; premium scripted, features, game show, textless deliveries. **film_a re-added** — it completed via 9.13's checkpoint/resume (8 windows checkpointed and resumed; 2483 shots, 143 min) |
| `docu-batch.txt` | 1 | an air-accident-investigation series — factual series, 44:46, 1020 shots, the densest recurrence in the corpus. The asset that proved chaining reached shipped output |
| `meridian.txt` | 1 | the Phase 9 negative control — **0 groups live post-swap** |

## Scripts, and the finding each produced

### Orchestration
| script | purpose |
| --- | --- |
| `poll.sh <manifest>` | poll a batch to settlement, showing the current state for RUNNING jobs |
| `collect.sh <manifest>` | pull `recurrence.json` / `shots.json` / `shot_centroids.npy` per run |
| `setup_workdir.sh [batch…]` | build the scratch layout from the checked-in manifests |
| `fetch_frames.sh <run> <wid> <shot_id>…` | pull and untar per-shot frame tarballs into `frames/<run>/shot_NNNN/` |

### The 9.3 ceiling — retired as a drop

The ceiling no longer deletes: groups over `max_recurrence_count` publish, and
the consumer derives the verdict (`recurrence_count > max_recurrence_count`,
plus `coverage`/`ent_range` from the shots.json join). The evidence chain that
retired it, in reading order: `replay_93.py` (deleted) found the 25 culled
groups; `sheet_all.py` put eyes on them; `truth.CULLED_LABELS` recorded 16 SET
/ 6 FURN / 3 PROMO — i.e. the drop was destroying genuine cards; `features.py`
showed member count is the WEAKEST discriminator available; and `ceiling_96.py`
(deleted) proved 9.6 alone rejects **0 of 25**, so the answer was
annotate-not-delete rather than delete-the-gate. Deleted scripts:
`replay_93.py` and `diff_96.py` replayed drop-diffs that no longer exist
(nothing is dropped, and new runs publish `hash_confirmed` directly);
`ceiling_96.py`'s decisive 0/25 is recorded here and in README.md §9.9.

The **published-pool visual audit** (`truth.AUDIT_LABELS`): all
53 groups in the over-ceiling/set-signature strata were eyeballed from contact
sheets, and the labels corrected by a second review — **9 FURN / 40 SET
/ 1 PROMO / 3 TMPL** (TMPL is new: a branded graphic template re-composited per
airing — market-chart panels, "DATA CHECK", a weather map — neither a place nor
a same-render card). Furniture splits into **two classes with opposite coverage
behaviour**: cards/idents/wipes (6, all at coverage ≤ 0.0066 — markets_a wipe +
title card, network ident + logo card, streamer end cards, network PSA
end-card ident) and **ad-break countdown slates** (3, all news_b: a teaser
plate with "WILL RETURN IN N:NN" ticking, ~15 s per break, coverage 0.018–0.021
— set-like, not low). Both audit passes initially mislabelled the slates as
re-aired promos, because a mid-frame thumbnail shows the teaser footage and the
countdown is only legible zoomed — the class that most directly timestamps ad
breaks was the easiest to misread. Counter-finding: 17 of 40 SETs sit at
coverage ≤ 0.016, so low coverage alone must never be read as "furniture".
`features.py` re-run on the corrected labels (15 FURN / 40 SET): **ent_range
leads at 0.945, coverage falls to 0.833** (the slate class is why), member
count — the retired ceiling's feature — stays near-noise at **0.612**.

The **kept-pool sampling audit** (`truth.AUDIT_SAMPLE_LABELS`,
53 of the 356 groups outside the strata, seeded shuffle, every asset
represented): **28 SET / 17 PROMO / 5 FURN / 3 TMPL**. The ~9% furniture rate
is the strata's measured false-negative rate — all five are LOW-coverage
families the signatures cannot see (an "<NETWORK> DAILY" logo family at
coverage ~0.0004, another break-countdown-slate variant, a break teaser, a
second network-PSA instance set, and **reality_b's own title card**, sibling
of the verified reality_a card) — and all five are published with
`hash_confirmed: true`, so nothing is lost. PROMO concentration is the other
finding: 13 of news_a's 15 sampled groups are streamer promo spots re-airing
~6× across the block. The flag tracked semantics perfectly across the sample:
every FURN and every re-aired PROMO reads true; the 8 false flags are roaming
set cameras and one reconstruction-footage variant.

| script | finding |
| --- | --- |
| `enumerate_audit.py` | *(replaces `enumerate_culled.py`)* writes `audit_groups.json` — published groups selected for eyeballing (default: over the reference ceiling OR carrying the set signature `coverage>0.016 AND ent_range<1.0`; `--all` for sampling), with batch/run/wid/features/shots so frames can be fetched |
| `sheet_all.py` | contact sheets `audit_page{N}.jpg` from `audit_groups.json` — the visual ground-truth mechanism (it produced the original 25-group audit and the 53-group published-pool audit) |
| `features.py` | **the feature study.** Scores every property computable from `shots.json` + centroids for its ability to separate genuine furniture from a recurring set, then scores candidate keep-rules. Replaces `separate.py`, `verdict.py`, `rule_test.py` and `guardrails.py`, which each answered a slice of this over 2 batches instead of 4 — and therefore disagreed on how many genuine cards exist. **Headline: `n`, the shipped ceiling's feature, ranks 9th of 20 (AUC 0.799), and `n<=8` keeps only 6 of 12 genuine furniture groups. `coverage` (member seconds / programme seconds) ranks 1st at 0.953 and is duration-normalised by construction; `ent_range` (mean per-shot `entropy_max - entropy_min`, already published and previously read only as a 9.4 floor) ranks 2nd at 0.948, high for animated idents and low for a locked-off anchor camera. NO feature separates cleanly — the `gap?` column is `-` for all 20.** Also: Pearson r(hours, n) over the 12 FURN groups is only **+0.348**, so runtime barely predicts a card's member count and the previously proposed `ceil(15*hours)` ceiling had almost no support. Retains guardrails' refutation of a minimum-gap floor (furniture bottoms out at 1.09 min between occurrences, sets at 1.01; an 8-min floor kills 8/12 furniture) and its confirmation that all 6 false culls are in the two >=2 h assets. |

### The 9.6 hash confirmation — flags, never drops

`hash_confirmed` is true / false / null (unverifiable) per published group, so
"what did 9.6 reject" is read straight from the output — which is what deleted
`diff_96.py` (it reverse-engineered the rejects by diffing an offline replay
against the published file).

| script | finding |
| --- | --- |
| `measure_96.py` | best cross-member pair statistics per rejected group against the shipped thresholds, so a `hash_confirmed: false` reads as decisive or marginal |
| `match_graph.py` | is 9.6's **existential** rule (any one matching pair confirms the whole group) the real problem? Builds the member-match graph; a same-render group should be near-complete, a same-set group confirmed by one lucky pair nearly empty. n=52 still yields a 14-member clique, so clique-pruning does not rescue it. |

### Clustering

**The swap shipped**: production grouping is now canonical complete
linkage at default threshold **0.07** (`lib/recurrence.py`,
`_group_by_complete_linkage`), restoring the invariant
`max_distance <= max_recurrence_threshold` on every published group. The rows
below are the diagnosis and calibration that justified it; findings quoting
single-linkage behaviour describe the **pre-swap** pipeline and are preserved
as the reasoning of record, not as reproducible numbers.

| script | finding |
| --- | --- |
| `linkage.py` | **re-exports `lib.recurrence.complete_linkage`** — the shared primitive moved into production with the swap, and the harness copy was deleted after the two were verified identical (clusters, member order, tie-breaks) on 4930 matrices: random tie-heavy sets plus every real candidate-group matrix at 0.05/0.07/0.10. Canonical (merge the *closest* admissible pair each round) is not a detail — the first, greedy *first-admissible-pair* version is order-dependent and gave a materially wrong answer, reporting no viable threshold at all where one exists. |
| `truth.py` | **the only ground truth.** The six visually verified true cards as exact **shot-id sets**, plus `CULLED_LABELS` (FURN / SET / PROMO) for the 25 ceiling-culled groups. Identity by `(run, group_id)` breaks the moment a replay moves the threshold — that artefact made the drama_e card read as LOST at 0.07 when it had survived intact — so there is deliberately no id-keyed accessor. This lived in seven places, including two that `exec()`'d a third file's source by string-splitting to borrow its tables. |
| `corpus.py` | **shared corpus access**: asset loading, cosine distance matrices, occurrence clustering, and `labelled_groups()` — the load/replay/diff/label loop that five scripts each carried their own copy of. Re-exports the schema defaults rather than restating them, so a replay cannot measure against a threshold the pipeline no longer uses, and its occurrence clustering **calls** `lib.recurrence.occurrence_labels` rather than replicating gate 9.1 — the pipeline refactor made that possible by publishing the rule as labels (which the replays need, to measure gaps between occurrences) with the count derived from it. The copies had drifted: only one of two loaders noticed film_a's missing artefacts, one of three distance builders never zeroed the diagonal, and the three occurrence-clustering copies disagreed on whether the running end-time is the maximum so far or the previous member's — harmless in practice, since shots are disjoint and sorted so their ends are monotone, but the kind of thing that stops being harmless the day a caller passes overlapping spans. |
| `chaining.py` | *(deleted with the annotate change — both halves of its comparison now replay post-swap code, so neither its culled bucket nor its published-diameter violations can reproduce; this row is the finding of record.)* Union-Find at distance ≤ 0.05 is **single-linkage**, so group diameter is unbounded. **True cards: diameter 0.0048–0.0634 (median 0.014), one complete-linkage subcluster each. Culled groups: 0.032–0.325 (median 0.120), 21 of 25 exceed the threshold and split into up to 6 subclusters.** This is the root defect: groups chain into mixed bags. **And it is not confined to the discard pile — on docu-batch, 9 of 19 post-ceiling groups have a diameter above the threshold (up to 2.6×), three dissolve completely under complete linkage, and six more shed members.** Note the contract consequence: `recurrence.json` publishes `max_recurrence_threshold: 0.05` and `max_distance: 0.1315` in the same document. |
| `sweep.py` | **which threshold? This is the calibration the 0.07 default rests on.** Tightening "some pair within t" to "every pair within t" makes the same t stricter, so 0.05 is the wrong number under complete-linkage semantics. Sweeps t against the six true cards: **safe window (0.0634, ~0.075], shipped default 0.07** — `markets_a:g3` has diameter 0.0634 so it cannot form below that, and `drama_e:g0` splits at 0.080. Post-swap it measures the shipped path directly and reads **415 groups / 41 >8-member at 0.070** (up to 4 more per row than the pre-swap simulation, which pruned single-occurrence-chain parents before refinement could rescue their subsets — the card verdicts are identical at every threshold; see its docstring). Second finding, equally important: **`maxn` stays 52 at every threshold and >8-member groups grow 26 → 41 between 0.05 and 0.07.** Chaining and the recurring-set ceiling are independent problems, and fixing the first makes the second's arithmetic worse. |
| `refine.py` | *(retired with the swap — its whole subject was the before/after between shipped single-linkage output and diameter-constrained refinement, and the "before" column can no longer be produced by any code in the repo.)* Finding of record: **322 → 364 groups at 0.05 (5/6 true cards intact, the markets_a card lost); 355 → 411 at 0.07 (6/6 intact)**; group count goes UP because chained components break into several legitimate ones; >8-member groups roughly halve at a fixed threshold but do not disappear, so the ceiling was not retired by the change. |

### Programme boundaries

**Shipped** as the `programme_tail` envelope field in shots.json
(`lib/tail_evidence.py` = moat.py's separator + textless.py's match test,
generalised). Full-corpus measurement that shaped it: the moat alone fires on
4/29 — but on the pay-TV masters the "tail" is a further programme PART after an
inter-part black, not textless; the tail-to-body match stats separate the two
(cooking_a 0.96 matched / sports_doc_a 0.88 matched + 0.98 ordered = textless,
both visually verified — sports_doc_a's body carries burned-in subtitles and its
22-minute tail is the same shots clean; title_card_a 0.01 / paytv_a 0.00 =
part boundaries). The verification order matters: sports_doc_a's 0.88 initially
looked explainable as "part 2 revisiting the same interview setups", and only
frames settled it — always eyeball a new firing asset before trusting the
numbers on it.

| script | finding |
| --- | --- |
| `textless.py` | textless elements — clean copies of in-programme shots appended after end credits for localisation. Discriminator is a conjunction: contiguous tail run, source shots monotone increasing, `src_span == run length`, tight distances, block at end of file. **cooking_a `_dualpurpose_`: two aligned blocks (24 shots → src 3–26, d_max 0.0239; 17 shots → src 27–44, d_max 0.0186). quiz_a: 9 candidate runs, all correctly rejected** (non-monotone, src span 120 for len 10) — it is the game-show set recurring. |
| `moat.py` | the shipped-signal version of the same boundary: a long black shot in the second half followed by ≥60 s of content. **Fires on 1 of 13 studio assets — the one marked `_dualpurpose_` — with zero false positives.** Uses only `shot_type` and timestamps already in `shots.json`. |
| `audio_probe.py` | one-off: would loudness/silence beat picture? **No.** The textless title block carries M&E and is audio-indistinguishable from programme, so `last non-silent audio` puts the boundary 2 minutes late; and the silent moat is already published as a black shot. Records the one real audio-only gap: 57 s of bars-and-slate at the head classified as `content`. |

## Scope boundary worth not "fixing"

studio_a1–104 and drama_b produce **zero candidates pre-ceiling**, and that is
correct. A scripted drama's title sequence appears once per episode, so its
furniture recurs *across* episodes; intra-asset recurrence structurally cannot
see it. Loosening `min_recurrence_count` to chase these will flood the output.
