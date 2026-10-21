"""
StateCheckAndPreprocess — first state of the synthesis workflow.

Validates the batch request and expands it into Map iterators. Fails
fast on unknown voice_ids or empty transcripts (README.md §10:
"else fail fast, do not silently proceed").

style_hint language gate (README.md §7a): a live multilingual E2E
test found `style` causes real content corruption on non-English
voices — e.g. "fast" on German/Spanish dropped QC similarity from
~0.95-1.0 (unstyled baseline, same voice/text) to 0.44-0.67, while the
same style stayed clean (1.0) on the one Chinese voice tested. Every
prior verification of style_hint (the 26-template research, the seed
sweep, this session's earlier Tier1/Tier2 confirmation) was English-
only — that verification does not transfer to other languages, and
this gate exists so an unverified combination fails loudly at request
time instead of silently shipping degraded audio (the same standard
already applied to unknown voice_ids/style names above).

Expected input:
{
  "operation": "StateCheckAndPreprocess",
  "uuid": "...",
  "input": {
    "batches": [{"voice_id": "...",
                 "transcripts": [{"text": "...",
                                  "target_duration_s": null}, ...]}, ...],
    "out_bucket": "...",           # optional; default proxy bucket
    "out_prefix": "...",           # optional; default derived from uuid
    "config": {"flow_steps": 6, "sampling_k": 25, "seed": null,
               "style": null}      # style: one of lib.style_hints.ALL_STYLES
  }
}
Output adds data.iterators, each carrying everything the per-batch states
need. **One caller batch becomes SEVERAL iterators**, split by estimated
work so the Map actually runs in parallel — see
split_into_iterator_groups for the measurements behind that, and for why
the split preserves transcript order.
"""

from __future__ import annotations

import logging
import os

from lib import style_hints, voice_store

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MAX_TRANSCRIPTS_PER_BATCH = 100  # coarse per-batch cap; the real bound is
                                 # the payload estimate below, which is
                                 # tighter for long texts or best-of-N
MAX_TEXT_CHARS = 1000

# How many Map iterators to aim for. This MUST equal the `MaxConcurrency` on
# `Map batch synthesis` in definitions.json / cfnstack.yaml: fewer groups
# leaves Map slots idle, and more than MaxConcurrency queues the surplus into
# a second serial wave that pays every cold model load again (48.8 s LLM +
# 35.3 s Flow + 9.4 s QC per invocation). That second wave is why "smaller
# chunks are always faster" does not hold here: modelled on the reference run,
# 13 groups against MaxConcurrency 10 is SLOWER than 10 groups for more money.
# test_units.py pins these two values equal, in BOTH copies of the definition.
MAP_MAX_CONCURRENCY = 40

# Step Functions hard-limits state input/output to 256 KiB (README.md §10).
# Two boundaries in this workflow carry the whole batch at once:
#   1. StateSynthesizeLLM → StateSynthesizeFlowHift, which carries every
#      transcript's speech-token array for every best-of-N candidate.
#   2. The Map's aggregate ResultPath, which carries every batch's outputs.
# README.md §10 derived ~49,000 tokens of headroom and required that
# "whatever default [transcripts_per_batch] gets must be checked against
# this ceiling, not assumed safe by extrapolation". A fixed count cannot
# do that: measured with the real JSON envelope, 100 transcripts fits only
# at ~230-char texts and best-of-1 (257 KiB — already at the edge); at
# MAX_TEXT_CHARS it fits 24, and at emotion-tier best-of-3 it fits 36 at
# 230 chars and 8 at 1000. So estimate the bytes and fail here, at the
# only point where the caller can act on it, instead of after paying for
# the whole LLM decode and dying on States.DataLimitExceeded.
SFN_PAYLOAD_LIMIT_BYTES = 256 * 1024
# 80% — leaves room for the envelope fields not modeled below (Map
# metadata, config, voice record) and for token counts above the estimate.
SFN_PAYLOAD_BUDGET_BYTES = int(SFN_PAYLOAD_LIMIT_BYTES * 0.80)
# CosyVoice3 speech tokens run at 25 Hz; English speech averages ~15
# chars/s, so ~1.67 tokens per character. Matches README.md's measured
# longest clip (390 tokens for a ~15.6 s / ~234-char utterance).
TOKENS_PER_CHAR = 25.0 / 15.0
# Serialized cost of one token in the Lambda runtime's JSON output:
# up to 4 digits plus the ", " separator json.dumps emits by default.
BYTES_PER_TOKEN = 6
# Per-transcript JSON overhead beyond the text itself: the transcripts[]
# entry keys and the llm_results[] wrapper keys.
BYTES_PER_TRANSCRIPT_OVERHEAD = 120


def estimate_batch_payload_bytes(texts: list[str], n_candidates: int) -> int:
    """Estimated size of the largest single state payload this batch will
    produce (the LLM → Flow+HiFT handoff, which carries the text AND every
    candidate's token array)."""
    total = 0
    for text in texts:
        tokens = int(len(text) * TOKENS_PER_CHAR)
        total += (len(text) + BYTES_PER_TRANSCRIPT_OVERHEAD
                  + n_candidates * tokens * BYTES_PER_TOKEN)
    return total


# The Map's aggregate ResultPath carries EVERY batch's post-QC outputs in one
# payload, so it is bounded across batches, not per batch. It is also the one
# payload bound that splitting a batch into more iterators cannot relieve: the
# total is the sum over every transcript in the request, however they are
# grouped. Token arrays are gone by then (synthesize_flow_hift pops
# llm_results).
#
# Each transcript's text used to appear THREE times here — transcripts[].text,
# qc.expected, qc.transcribed — which at MAX_TEXT_CHARS made the aggregate
# 15.9x larger than the information in it needs and capped a request at 63
# transcripts. verify_output now drops transcripts[] and qc.expected outright,
# and keeps qc.transcribed only on clips that FAILED or errored (see
# _prune_qc_evidence there). So the multiplier is 1 in the worst case rather
# than 3, and 0 when everything passes:
#
#   1000-char texts, budget 204 KiB:   63 transcripts before ->  ~1000 after
#    230-char texts:                  215 transcripts before ->  ~1000 after
#
# TEXT_COPIES_IN_AGGREGATE is 1, not 0, deliberately: a run where every clip
# fails QC keeps one copy of every text, and that is a real workload (a wrong
# `language` fails the lot), so it is what the budget must survive. The gate
# must stay pessimistic — being wrong the other way means paying for a full
# synthesis and then dying on States.DataLimitExceeded.
TEXT_COPIES_IN_AGGREGATE = 1
# Measured with real json.dumps against the pruned shape, worst entry first: a
# clip that FAILED QC keeps qc.transcribed and costs 258 bytes before its text,
# and one whose QC could not RUN also carries a qc.error string ("passed": None,
# similarity -1.0) for 326. 320 was calibrated for the old 3x shape, where the
# text dominated and the envelope did not have to be tight; with the text gone
# this term IS the estimate for short transcripts, so it is measured against the
# 326-byte case and rounded up rather than to the typical entry. Checked against
# real envelopes over 10-1000 chars x 1-1000 transcripts x 0/5/100 % failure,
# with and without qc.error: no combination under-estimates. Confirmed against
# live execution 5512a1af (2026-08-10): 301 bytes on a passing clip, 324 on a
# failed one that keeps qc.transcribed, so 400 is ~76 bytes conservative on the
# worst real entry. Left there rather than tightened — the slack is the margin
# against a longer wav_key or a new per-clip field, and this gate must fail
# safe.
BYTES_PER_OUTPUT_OVERHEAD = 400
# The aggregate also carries each ITERATOR's own envelope — job_id, voice_id,
# out_bucket, out_prefix, config, batch_index, source_batch_index, and the
# `voice` record — times however many iterators the Map ran. Counted as a fixed
# term rather than amortized into the per-entry figure above: it scales with the
# iterator count, not the transcript count, so folding it into a per-transcript
# constant would over-charge a 1000-transcript request and under-charge a
# 40-transcript one, which is the direction that fails a job.
#
# **Measured against execution 5512a1af (eu-west-1, 2026-08-10): 751 bytes per
# iterator, not the 380 this constant first held.** That first figure was
# derived from a hand-built envelope that omitted `voice`, which is the largest
# member at 334 bytes — it carries `ref_transcript` verbatim, and nothing in
# register_voice bounds that string (363 chars is the longest in the live
# 46-voice catalog, versus 112 on the reference run's voice). The error was
# masked because it cancelled against BYTES_PER_OUTPUT_OVERHEAD being ~96 bytes
# high per entry: on the reference shape the two nearly annihilated and the
# total landed 1.1 % LOW, i.e. on the unsafe side, by luck rather than
# calibration. Both terms are now measured separately so neither hides the
# other.
#
# 1100 rather than 751: the measured figure came from a voice whose
# ref_transcript is 112 chars, and the longest of the 46 in the live catalog is
# 363, which alone pushes the envelope to ~1002. The headroom is the whole
# margin against a chatty reference clip, since register_voice does not bound
# that string at all.
BYTES_PER_ITERATOR_ENVELOPE = 1100


def estimate_map_aggregate_bytes(all_texts: list[str],
                                 n_iterators: int | None = None) -> int:
    """Bytes the Map's aggregate ResultPath will carry.

    `n_iterators` is the real iterator count when the caller knows it (the gate
    in lambda_handler runs after the split, so it does). It defaults to the
    worst case, MAP_MAX_CONCURRENCY, for callers estimating a ceiling before
    scheduling — over-charging there is the safe direction.
    """
    if n_iterators is None:
        n_iterators = MAP_MAX_CONCURRENCY
    return (n_iterators * BYTES_PER_ITERATOR_ENVELOPE
            + sum(TEXT_COPIES_IN_AGGREGATE * len(text)
                  + BYTES_PER_OUTPUT_OVERHEAD for text in all_texts))


# ───────────────────────────────────────────────────────────────────────────
# Iterator scheduling
#
# A caller's batch is NOT one iterator. It used to be, and that made the Map
# useless: execution 630f729f-59b0-57f2-8bcb-6a42ecedc987 (eu-west-1,
# 2026-08-06) submitted 73 transcripts as one batch, so `Map batch synthesis`
# ran at concurrency 1 for 52 minutes — MaxConcurrency 10 with one item to
# give it. The gateway always submits exactly ONE batch, so the Map's
# parallelism has to come from inside a batch or it does not exist.
#
# The split is by ESTIMATED WORK, not by transcript count, and the difference
# is not cosmetic. Both models decode token-by-token, so per-transcript cost
# is roughly linear in text length, and a fixed count puts however many long
# sentences happen to be adjacent into one iterator while another gets the
# short ones. Modelled on that run's own coefficients (73 clips at ~42 chars,
# every iterator paying its own cold loads, a stage over the invocation budget
# charged a full 900 s per resume):
#
#   1 iterator, as it ran      56.9 min   (measured: 52.2 — the model is
#                                          slightly pessimistic, not optimistic)
#   fixed 8 per iterator        7.1 min
#   fixed 10 per iterator       8.5 min   <- MORE per iterator is not monotone
#   work-balanced               2.9 min
#
# On a bimodal input (60 short + 13 near-MAX_TEXT_CHARS) the same comparison is
# 62.3 min for fixed-8 against 8.7 work-balanced — 7x — because a count split
# hands one iterator eight 950-char transcripts and everything else idles.
# Balanced chunks also keep the checkpoint/resume path from firing on ordinary
# runs: zero resumes on every modelled workload, where the count split needed
# several, and each resume burns a whole 900 s invocation for the ~110-230 s of
# forward progress it makes before the buffer trips.
#
# Coefficients are fitted against all 73 transcripts of that execution
# (CloudWatch per-transcript log lines, `/aws/lambda/*-tts-{2-cores,4-cores,
# 1-core}`). Reproducing that run from these numbers lands within 0.2 % of
# the measured 52 min, so they are measurements, not guesses:
#
#   LLM        4.81 s + 0.1815 s/char   (R2 0.87)
#   Flow+HiFT 18.61 s + 0.1819 s/char   (R2 0.87)   <- 63 % of all compute
#   QC                 0.0648 s/char    (fitted from the same run)
#
# Characters are a PROXY, and a known-imperfect one. The actual driver is
# speech tokens (R2 0.98 against the same timings — far better than chars at
# 0.87 or words at 0.83), but a token count does not exist until the LLM
# stage has already decoded the transcript, so it cannot schedule anything
# here. Chars beat words because a 12-char word costs more than a 3-char one
# and word count discards exactly that. Do not "improve" this to words.
WORK_FIXED_SEC = 4.81 + 18.61          # per transcript, both model stages
WORK_PER_CHAR_SEC = 0.1815 + 0.1819 + 0.0648
# Flow+HiFT alone, for the per-invocation timeout cap below. It is the
# expensive stage and the one that actually times out (it needed 3
# invocations on the reference run where the LLM needed 2), so the cap is
# derived from it rather than from the pipeline total.
FLOW_FIXED_SEC = 18.61
FLOW_PER_CHAR_SEC = 0.1819
# Lambda's 900 s ceiling less checkpoint.REMAINING_TIME_BUFFER_MS (60 s):
# work beyond this point does not get done in this invocation, it gets
# checkpointed and re-invoked. Kept as a literal rather than imported from
# lib.checkpoint so this module's scheduling has no import-time dependency
# on the state modules; the two must agree, and the test suite pins that.
USABLE_INVOCATION_SEC = 900.0 - 60.0


def estimate_transcript_work_sec(text: str) -> float:
    """Modelled wall-clock cost of one transcript through LLM + Flow/HiFT +
    QC, excluding the per-invocation cold model load (which is a property of
    the iterator, not of the transcript)."""
    return WORK_FIXED_SEC + WORK_PER_CHAR_SEC * len(text)


def split_into_iterator_groups(transcripts: list[dict],
                               target_groups: int) -> list[list[dict]]:
    """Split transcripts into contiguous runs that minimize the work of the
    HEAVIEST run, using at most `target_groups` of them — more only when a
    single Lambda invocation could not hold the Flow+HiFT work otherwise.

    The heaviest run is the only figure that matters: every iterator runs
    concurrently, so the Map costs what its slowest iterator costs, and a short
    final run is free.

    **Order is preserved, and that is a correctness requirement, not a
    preference.** The obvious algorithm here is LPT (sort longest-first, deal
    into the emptiest bin), which balances better — by ~1 % on the reference
    run, 6 % worst case. It is wrong anyway: `outputs` is built by positional
    `append` in synthesize_flow_hift, and the gateway derives the caller-
    facing `transcript_index` from a running counter over the iterators in
    order. Reordering transcripts would hand clip 40's audio back as clip 3,
    with nothing raising. The margin LPT buys is not worth a silently wrong
    join key, so this walks the list in order and only chooses where to cut.
    """
    if not transcripts:
        return []
    # A transcript is indivisible, so more slots than transcripts is not a
    # finer split, and clamping keeps `max(work)` a valid lower bound for the
    # bisection below.
    target_groups = max(1, min(target_groups, len(transcripts)))
    work = [estimate_transcript_work_sec(t["text"]) for t in transcripts]

    # Minimize the WORK OF THE HEAVIEST GROUP by bisecting on that value.
    # `_cut` is greedy-optimal for a *contiguous* split once the ceiling is
    # fixed, and "can it be done under L groups" is monotone in L, so bisecting
    # L converges on the exact optimum rather than approximating it.
    #
    # Do not replace this with a running "fill to total/target_groups" target,
    # which is the intuitive version and is wrong at the tail. Each cut lands
    # slightly under its target, the shortfalls accumulate, and whatever is
    # left is stranded in the last group once no further cut is permitted.
    # Measured on the reference run's 73 transcripts into 20 groups: the last
    # group came out at 8 items / 286 s against ~145 s for the other 19 — a
    # 2.2x straggler, i.e. the exact imbalance this function exists to remove,
    # reintroduced at the tail. Re-deriving the target from the remaining work
    # after every cut does NOT fix it (measured: same 286 s), because the
    # binding constraint is the cut being refused, not the target being stale.
    # The bisection has no tail case at all: on that input it gives 134-173 s,
    # within 1.15x of the theoretical floor, and it is exactly optimal on 7 of
    # the 9 modelled workloads in test_units.py.
    #
    # The timeout cap rides along in the same walk rather than as a second pass
    # over the finished groups. Equal work per group says nothing about
    # whether a group fits ONE invocation — 4 transcripts at MAX_TEXT_CHARS are
    # ~200 s each in Flow alone — and checkpoint/resume handles the overflow
    # correctly but serially, where iterators run in parallel, so an extra
    # iterator beats an extra invocation. Capping afterwards wasted iterators:
    # re-cutting a finished group left a remainder each time, which turned the
    # worst legal batch (100 transcripts at MAX_TEXT_CHARS) into 40 iterators
    # where 25 covers it. One walk that respects both ceilings gives 25.
    flow = [FLOW_FIXED_SEC + FLOW_PER_CHAR_SEC * len(t["text"])
            for t in transcripts]

    def _cut(limit: float) -> list[list[dict]]:
        """Walk in order, closing the current group when the next transcript
        would push it past `limit` of total work or past one invocation of
        Flow+HiFT. A single transcript is never split — it is the indivisible
        unit, so one pathological near-MAX_TEXT_CHARS transcript still overruns
        on its own, the residual caveat lib/checkpoint.py documents."""
        out: list[list[dict]] = []
        cur: list[dict] = []
        acc = facc = 0.0
        for t, w, f in zip(transcripts, work, flow):
            if cur and (acc + w > limit or facc + f > USABLE_INVOCATION_SEC):
                out.append(cur)
                cur, acc, facc = [], 0.0, 0.0
            cur.append(t)
            acc += w
            facc += f
        if cur:
            out.append(cur)
        return out

    # The Flow ceiling can force more groups than the Map has slots. Bisecting
    # against target_groups then would never terminate on a feasible limit, so
    # the floor for the search is however many groups the cap alone demands
    # (`_cut(inf)`, i.e. work-limit never binds) — greedy is optimal for that.
    target_groups = max(target_groups, len(_cut(float("inf"))))
    lo, hi = max(work), sum(work)
    for _ in range(64):      # ~1e-19 relative; float64 converged long before
        mid = (lo + hi) / 2
        if len(_cut(mid)) <= target_groups:
            hi = mid
        else:
            lo = mid
    # `hi` is feasible and `lo` is not. Nudge up so the final walk cannot lose
    # a cut to the last bit of rounding and return target_groups + 1 groups.
    return _cut(hi * (1 + 1e-9))


# Languages where style_hint has actually been live-verified content-clean
# (see module docstring). Chinese passed its one test but isn't backed by
# the same repeated-verification depth English got this session — kept
# out until it gets that, rather than assumed safe from a single data
# point.
STYLE_VERIFIED_LANGUAGES = frozenset({"en"})


def lambda_handler(event: dict, context) -> dict:
    uuid = event["uuid"]
    inp = event["input"]
    batches = inp.get("batches") or []
    if not batches:
        raise ValueError("input.batches is empty")

    config = inp.get("config") or {}
    out_bucket = inp.get("out_bucket") or os.environ["PROXY_BUCKET"]
    out_prefix = (inp.get("out_prefix")
                  or f"micro-workflow/tts-on-aws/output/{uuid}").rstrip("/")

    style = config.get("style")
    if style is not None and style not in style_hints.ALL_STYLES:
        raise ValueError(
            f"config.style unknown: {style!r}; must be one of "
            f"{sorted(style_hints.ALL_STYLES)} or omitted")

    # Map slots are shared across the caller's batches, so divide them by
    # estimated work rather than evenly. An even split (MAP_MAX_CONCURRENCY //
    # len(batches)) gives a 90-transcript batch and a 1-transcript batch ten
    # slots each: nine wasted on one side while the other runs 9 transcripts
    # per iterator. Every batch keeps at least one slot, so the shares can add
    # up to more than MAP_MAX_CONCURRENCY when there are many tiny batches;
    # that is the same benign overshoot as the timeout cap, and the Map queues
    # the surplus. Computed here from raw text lengths before validation, which
    # is safe because a batch that fails validation raises out of the loop
    # anyway and its share is never used.
    batch_work = [sum(estimate_transcript_work_sec(t.get("text") or "")
                      for t in (b.get("transcripts") or []))
                  for b in batches]
    all_work = sum(batch_work) or 1.0
    batch_slots = [max(1, round(MAP_MAX_CONCURRENCY * w / all_work))
                   for w in batch_work]

    iterators = []
    for bi, batch in enumerate(batches):
        voice_id = batch.get("voice_id")
        if not voice_id:
            raise ValueError(f"batches[{bi}].voice_id missing")
        voice = voice_store.get_voice(voice_id)
        if voice is None:
            raise ValueError(f"batches[{bi}].voice_id unknown: {voice_id}")
        if style is not None and voice.get("language") not in STYLE_VERIFIED_LANGUAGES:
            raise ValueError(
                f"batches[{bi}]: style_hint is only verified for "
                f"{sorted(STYLE_VERIFIED_LANGUAGES)} voices; "
                f"voice_id {voice_id} is language={voice.get('language')!r} "
                "(see README.md §7a for the live multilingual test "
                "that found real content corruption outside English)")

        transcripts = batch.get("transcripts") or []
        if not transcripts:
            raise ValueError(f"batches[{bi}].transcripts is empty")
        if len(transcripts) > MAX_TRANSCRIPTS_PER_BATCH:
            raise ValueError(
                f"batches[{bi}] has {len(transcripts)} transcripts; max "
                f"{MAX_TRANSCRIPTS_PER_BATCH} per batch (state-payload bound) — "
                "split into multiple batches")
        for ti, t in enumerate(transcripts):
            text = (t.get("text") or "").strip()
            if not text:
                raise ValueError(f"batches[{bi}].transcripts[{ti}].text is empty")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(
                    f"batches[{bi}].transcripts[{ti}].text is {len(text)} chars; "
                    f"max {MAX_TEXT_CHARS}")
            t["text"] = text

        # Payload ceiling (see SFN_PAYLOAD_BUDGET_BYTES above). n_candidates
        # mirrors synthesize_llm's own seed selection exactly: an explicit
        # caller seed always means single-shot regardless of style.
        n_candidates = (1 if config.get("seed") is not None
                        else style_hints.candidates_for(style))
        est = estimate_batch_payload_bytes(
            [t["text"] for t in transcripts], n_candidates)
        if est > SFN_PAYLOAD_BUDGET_BYTES:
            raise ValueError(
                f"batches[{bi}] would produce an estimated {est // 1024} KiB "
                f"Step Functions state payload (budget "
                f"{SFN_PAYLOAD_BUDGET_BYTES // 1024} KiB of the 256 KiB "
                f"hard limit) from {len(transcripts)} transcripts x "
                f"{n_candidates} candidate(s) — split into smaller batches"
                + (f". style={style!r} generates {n_candidates} candidates "
                   "per transcript; passing an explicit seed forces "
                   "single-shot and cuts the payload accordingly"
                   if n_candidates > 1 else ""))

        # One caller batch becomes as many iterators as it takes to fill the
        # Map, split by estimated work (see split_into_iterator_groups). The
        # payload gate above stays PER CALLER BATCH and is now deliberately
        # conservative: the LLM→Flow handoff it bounds is really per iterator,
        # and iterators are smaller than the batch they came from, so a batch
        # rejected here would in fact have fitted once split. Left strict
        # because loosening it is a capability change — it would let callers
        # submit shapes that have never run — and the aggregate gate below
        # would still have to hold.
        for group in split_into_iterator_groups(transcripts, batch_slots[bi]):
            iterators.append({
                "job_id": uuid,
                # Unique per ITERATOR, not per caller batch: it is the
                # checkpoint key's discriminator (lib/checkpoint.py) and part
                # of every wav_key (synthesize_flow_hift), so two iterators
                # sharing one index would resume from each other's checkpoint
                # and overwrite each other's audio.
                "batch_index": len(iterators),
                "source_batch_index": bi,
                "voice_id": voice_id,
                "transcripts": group,
                "config": config,
                "out_bucket": out_bucket,
                "out_prefix": out_prefix,
            })

    # Cross-batch bound on the Map's aggregate result (see
    # estimate_map_aggregate_bytes) — each batch can be individually legal
    # and the aggregate still blow the 256 KiB limit. Splitting a batch into
    # more iterators does NOT change this total: it is the sum over every
    # transcript in the request, however they are grouped. So it is the one
    # gate the scheduling above cannot relieve, and it is counted over the
    # caller's batches, which is what the caller can actually act on.
    all_texts = [t["text"] for it in iterators for t in it["transcripts"]]
    agg = estimate_map_aggregate_bytes(all_texts, len(iterators))
    if agg > SFN_PAYLOAD_BUDGET_BYTES:
        raise ValueError(
            f"the {len(batches)} batches together ({len(all_texts)} "
            f"transcripts) would produce an estimated {agg // 1024} KiB Map "
            f"result payload (budget {SFN_PAYLOAD_BUDGET_BYTES // 1024} KiB "
            "of the 256 KiB Step Functions hard limit) — submit fewer "
            "transcripts per request")

    event.setdefault("data", {})["iterators"] = iterators
    log.info("validated %d batches, %d transcripts total -> %d Map iterators "
             "(est. work %.0fs longest / %.0fs total; "
             "est. Map aggregate %d KiB of %d KiB budget)",
             len(batches), len(all_texts), len(iterators),
             max((sum(estimate_transcript_work_sec(t["text"])
                      for t in it["transcripts"]) for it in iterators),
                 default=0.0),
             sum(estimate_transcript_work_sec(t) for t in all_texts),
             agg // 1024, SFN_PAYLOAD_BUDGET_BYTES // 1024)
    return event
