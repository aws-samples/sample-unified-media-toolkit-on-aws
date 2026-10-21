"""
pb_common.py — span math and the P0 decision constants.

All thresholds here are PROVISIONAL until the corpus gate (README.md §7
"Corroboration threshold"): they encode the RULE SHAPES we intend to
measure, not tuned values. Every constant states why it exists.
"""
from __future__ import annotations

import base64
import json
import os
import re
import unicodedata


# ── co-occurrence rule shapes ─────────────────────────────────────────────────

# Enough separate turns for aggregate statistics to mean anything —
# tier-2 voice eligibility in class1 (the co-occurrence CONFIRM path
# this bar once served was retired 2026-09-24: the matcher is the one
# binding engine; cooccur emits proposals and the narrator verdict).
CONFIRM_MIN_TURNS = 5     # distinct speech turns with the face on screen

# A voice whose co-occurring faces are diverse with NO stable partner IS
# a narrator/VO/announcer (README.md §5: variance as a classifier).
VOICE_ONLY_MAX_SHARE   = 0.2   # nobody owns even this much -> no partner
VOICE_ONLY_MIN_FACES   = 3     # seen over at least this many faces

# Below this many turns, co-occurrence says nothing either way.
MIN_TURNS_TO_JUDGE = 2


# ── critic rule shapes ────────────────────────────────────────────────────────

# A person cannot talk over themselves: same-name speaker merges are
# vetoed above this much overlapping speech. Tolerance absorbs
# diarization boundary jitter, nothing more.
MERGE_SPEECH_OVERLAP_TOLERANCE_SEC = 1.0
# One real person CAN be on screen twice (live + photo, monitor, split
# screen replay), so face merges get a looser bar.
MERGE_FACE_OVERLAP_TOLERANCE_SEC = 5.0

# ── over-clustering accommodation (user direction, 2026-09-09) ───────────────
# Diarization over-merge is an INPUT CONDITION, not a defect to wait
# out. Same-name absorption (persons.accommodate_overclustering) welds
# round-2-named voices into their recognizer-named persons. The
# RECOVERY_* top-talker thresholds that lived here were deleted
# 2026-09-24 with recovery pass 2 (subsumed by the class-1/2 matcher:
# 34/35 welds, and the pass caused the one proven misbinding).


# ── class-1 hierarchical matching rule shapes (user direction, 2026-09-23) ───

# Class-1 citizens are the voices/faces carrying the programme: every
# voice/face whose INDIVIDUAL share of total speech / screen time
# clears this floor. A fraction of the total, never a fixed count — a
# two-hander and a 2.5h broadcast both pick their actual leads — and
# the floor is its own K bound (at most 1/0.04 = 25 citizens,
# structurally). A cumulative-coverage rule was considered and
# rejected: a 60%-speech lead would have covered the line alone and
# excluded the 35% co-lead — the exact desk pair the tier exists for.
CLASS1_MIN_SHARE = 0.04
# Score shape: lift term in [0,1] (excess lift over 1.0, capped),
# solo term in [0,1] (solo seconds as a share of the voice's speech,
# CLASS1_SOLO_REF = full marks), plus the name-join weight. A match
# needs a real score AND a real margin over the voice's next-best FREE
# face — thin margin is a recorded abstention, never a guess.
CLASS1_LIFT_CAP = 4.0
CLASS1_SOLO_REF = 0.2
CLASS1_NAME_WEIGHT = 1.5
CLASS1_MIN_SCORE = 0.5
CLASS1_MIN_MARGIN = 0.2
# Onset following (user direction, 2026-09-24: temporal grammar — the
# edit CUTS TO the speaker as they start talking): a face whose
# appearance BEGINS within this window after a voice's turn start earns
# a cut-to vote. Window swept on 11 known pairs across the 10-asset
# corpus: top-1 flat from 2-5s, median lead peaks at 4s and decays by
# 8s (reaction-shot cuts creep in). ONSET_REF = the follow rate earning
# full marks — the cut follows the voice at least half the time
# (measured true pairs: 34-63% news, 68-95% scripted).
ONSET_WINDOW_SEC = 4.0
ONSET_REF = 0.5
# Floor-holding blocks (user direction, 2026-09-24: freeze one
# speaker's segments and ask who owns the screen while ONLY they hold
# the floor): maximal runs of one speaker's turns with no other
# speaker intervening, joined across pauses <= the gap, long enough
# for face statistics to mean anything. During a monologue the
# reverse-shot grammar quiets and the camera favors the speaker —
# measured on the 11 known corpus pairs: the true face tops the block
# share 10/11 (leads 2.2-34x; the miss is a news anchor narrating INTO
# package b-roll, which lift and name-join settle). FLOOR_REF = the
# block share earning full marks — the face owns about a third of the
# floor-holding time, inserts and b-roll take the rest (measured true
# pairs: 14-52%).
FLOOR_BLOCK_MIN_SEC = 8.0
FLOOR_BLOCK_JOIN_GAP_SEC = 5.0
FLOOR_REF = 0.3
# The forced-by-elimination rung (sole survivor at a relaxed floor)
# was DELETED 2026-09-24: with onset + floor-holding in the assertion
# set it fired ZERO times corpus-wide, and it produced the one wrong
# tier-1 assignment during development. Eliminations act through
# strikes and margins, never through a discount on the evidence bar.
# A class-1 MATCHED face's presence is already explained by its own
# voice: class-2 co-occurrence divides its credit by this factor (a
# tail voice riding an anchor's face is shot/reverse-shot editing, not
# the anchor). Discount, not exclusion — a diarization SPLIT of the
# matched voice still binds to the same face through the vocative and
# subset-name mechanisms.
CLASS2_MATCHED_FACE_DISCOUNT = 0.25

# F0×gender veto bands, from the 2026-09-23 corpus calibration (66
# bound voice+face pairs across 10 assets, 41 male / 24 female faces,
# clusters with >= F0_MIN_SENTENCES voiced sentences): male face F0
# p05-p95 = 89-178 Hz, female = 143-233 Hz. Outside the band the pair
# is VETOED; inside the 140-180 overlap the check abstains; agreement
# earns a small boost, never a decision. Of the corpus's three
# band-violators, two were verified misbindings (the veto's target
# class) — the third was an 80-year-old actor whose 200 Hz voice is
# REAL (human review, 2026-09-24): the veto struck the true pair and
# the elimination rung then bound the wrong survivor. Hence the AGE
# gates below — physiology, not tuning: aging shifts F0 toward the
# shared band in BOTH directions (presbyphonia — male rises, female
# falls), so an elderly face's cross-register reading abstains instead
# of striking; a child's face makes F0 unassessable entirely
# (prepubescent F0 is high regardless of sex). Age comes from the
# recognizer's estimate (age_range midpoint), which skews young on
# well-preserved faces — the gate catches what the estimator admits.
F0_MALE_MAX_HZ = 180.0
F0_FEMALE_MIN_HZ = 140.0
F0_MIN_SENTENCES = 3
F0_MATCH_BONUS = 0.25
F0_ELDERLY_AGE = 60.0
F0_CHILD_AGE = 15.0


# ── vocative resolution rule shapes ───────────────────────────────────────────

# One "Tom," is noise; a speaker consistently addressed by the same name
# across separate exchanges is a name. The lead requirement kills the
# case where two names get voted onto one speaker.
VOCATIVE_MIN_VOTES   = 2
VOCATIVE_MIN_LEAD    = 2.0
# an adjacent turn only counts as the same exchange within this gap
VOCATIVE_MAX_GAP_SEC = 10.0


# ── focus-read scheduler rule shapes (P1) ─────────────────────────────────────

# A lower-third appears within seconds of its trigger (an introduction
# cue, a face's first soundbite); the window is the read span. The
# trigger classes that read LOWER THIRDS (and therefore opt into the
# witness's crop) are declared HERE, beside the window they share — a
# new chyron-class trigger joins this tuple once, instead of span
# assembly re-deriving the class by name matching.
CHYRON_WINDOW_SEC = 8.0

# Depicted-face guard (enrichment media_depictions join, 2026-09-06): a
# face whose screen time is MAJORITY inside shots the enrichment attests
# contain displayed media (photo/screen/poster/mirror/projection) is
# treated as depicted — excluded from speaker co-occurrence and typed
# depicted_only. 0.5 is the structural majority rule, not a fitted
# scalar: below it the face demonstrably appears live more than
# depicted, and binding on the live majority is the designed behavior.
DEPICTED_SHARE_THRESHOLD = 0.5
CHYRON_CLASS_REASONS = ("introduction", "face_debut")
# One-off b-roll faces never get a lower-third worth a read.
MIN_DEBUT_SCREEN_SEC = 10.0
# Per-run read budget: STRUCTURAL, not fitted — an order of magnitude
# above what the trigger classes produce on the measured assets. Priority
# order (intro cues, then face debuts, then unreadable-text shots)
# decides what a smaller budget keeps.
MAX_READ_SPANS = 40
# "Tail" for the credits activation gate: text evidence concentrated in
# the last fifth of the runtime reads as credits, not chyrons.
TAIL_FRACTION = 0.2
# UNHINTED credits reads are a PROBE, not a purchase (spend audit,
# 2026-09-30: full-tail credits reads on four 2h news assets cost
# ~$10-13 of a $20.69 corpus bill and bought channel logos and
# "WILL RETURN IN:" countdown cards — 1,387 frames of one logo on one
# asset — because tail tickers/bumpers satisfy the text-in-tail gate
# on content that HAS no credit roll). Without a caller hint the read
# covers only the final PROBE window — standard episodic rolls run
# ~30-90s and sit at the very end, so the probe still captures them
# (the corpus' real rolls all fit); a caller who knows better
# (feature films, front-loaded rolls) passes credits_hint for the
# full-tail read.
CREDITS_PROBE_SEC = 240.0
# Reads bought PER unnamed major head (2026-10-05): a head present
# for most of the programme otherwise legitimizes every overlapping
# span up to MAX_READ_SPANS — on 2h news the title spend was
# cap-bound, not need-bound (~$4.30 of a $6.28 corpus estimate). The
# planner's class-priority order means the quota buys the head's own
# face-debut span first, then further sightings; the consumer needs a
# name repeated on >= VOCATIVE_MIN_VOTES frames to trust it, and a
# recurring lower third satisfies that within one read — three read
# moments per head is naming budget, not surveillance.
READS_PER_HEAD = 3


# ── name-token matching, ONE home (2026-10-05) ───────────────────────────────
# The lookup rule for corroborating a PROPOSED name against on-screen
# text (and anywhere else two spellings of one name must join): whole
# tokens only, contiguous runs, symmetric containment. Never characters
# inside a token (short names live inside ordinary words in every
# language), never fuzzy repair, no diacritic stripping (combining
# marks are letters in some scripts). A miss is always safe: the
# corroboration stays with the witness, which is today's behavior.

_HYPHENS = re.compile(r"[\s‐-―\-]+")


def name_tokens(s: str) -> list[str]:
    """NFKC + casefold, split on whitespace/hyphens, strip edge
    punctuation. The ONE normalization both sides of every name
    comparison go through."""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    out = []
    for t in _HYPHENS.split(s):
        while t and unicodedata.category(t[0]).startswith("P"):
            t = t[1:]
        while t and unicodedata.category(t[-1]).startswith("P"):
            t = t[:-1]
        if t:
            out.append(t)
    return out


def name_match(a: str, b: str) -> bool:
    """True when the shorter token sequence appears as a CONTIGUOUS
    run inside the longer — 'Joanne' in 'JOANNE SMITH', 'ROMAN' in
    'Rowan Kemble'; never 'Anne' in 'JOANNE' (token-internal), never
    'Jane Doe' in 'JANE AND JOHN DOE' (scattered)."""
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return any(long_[i:i + len(short)] == short
               for i in range(len(long_) - len(short) + 1))


# ── enrichment system prompts ─────────────────────────────────────────────────

# ── span helpers ──────────────────────────────────────────────────────────────

def overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap of [a0,a1] and [b0,b1] in seconds; 0.0 when disjoint."""
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)


def spans_total_sec(spans: list[tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in spans)


def spans_overlap_sec(a: list, b: list) -> float:
    """Total pairwise overlap between two span lists (seconds). ONE home
    (moved from critic, 2026-09-06): the merge veto and the depicted-face
    share both compute it, and two copies of overlap math drift."""
    return sum(overlap_sec(a0, a1, b0, b1) for a0, a1 in a for b0, b1 in b)


def subtract_spans(spans: list, cuts: list) -> list:
    """`spans` minus `cuts` — the remaining live segments. Cuts are
    merged first so overlapping cut spans never double-subtract."""
    if not cuts:
        return list(spans)
    merged: list = []
    for c0, c1 in sorted(cuts):
        if merged and c0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], c1)
        else:
            merged.append([c0, c1])
    out = []
    for a, b in spans:
        cur = a
        for c0, c1 in merged:
            if c1 <= cur or c0 >= b:
                continue
            if c0 > cur:
                out.append((cur, min(c0, b)))
            cur = max(cur, c1)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


# ── caller flags, ONE home for the defaults ──────────────────────────────────
# The gateway normalizes these into the identity hash; ValidateSources
# writes them into the plan; consuming states read plan[flag] directly
# (a missing key is a bug, not a choice — never re-default downstream).
DEFAULT_FLAGS = {
    "use_batch": True,           # batch-price the witness reads
    "read_onscreen_text": True,  # credits/chyrons/titles channel
    "llm_enrichment": True,     # LLM enrichment (both rounds)
}


# ── family gateway invocation ─────────────────────────────────────────────────

def invoke_gateway_tool(lam, function_arn: str, target: str, tool: str,
                        payload: dict) -> dict:
    """Invoke a sibling gateway Lambda directly (the ad_e2e pattern:
    tool routing rides base64 ClientContext). The ONE copy — states and
    scripts alike call this.

    Raises RuntimeError on FunctionError: an unhandled exception in the
    gateway returns HTTP 200 with an error body, and parsing that body
    as a tool response silently misreads a crash as a refusal."""
    ctx = base64.b64encode(json.dumps(
        {"custom": {"bedrockAgentCoreToolName":
                    f"{target}___{tool}"}}).encode()).decode()
    r = lam.invoke(FunctionName=function_arn, ClientContext=ctx,
                   Payload=json.dumps(payload).encode())
    body = r["Payload"].read()
    if r.get("FunctionError"):
        raise RuntimeError(
            f"{target}___{tool} invocation failed "
            f"({r['FunctionError']}): "
            f"{body[:300].decode('utf-8', 'replace')}")
    return json.loads(body)


# benign SendTaskSuccess races: the token was already completed (an
# absorbed-closed run, a duplicate delivery) or the state moved on —
# the machine is where we wanted it. ONE list, shared by every held
# state (the code list and the swallow-vs-raise rule are one fact).
TOKEN_GONE = ("TaskDoesNotExist", "TaskTimedOut", "InvalidToken")


def complete_task_token(sfn, task_token: str, payload: dict) -> dict:
    """Self-complete a held state's own token with `payload` as data;
    TOKEN_GONE races are benign, anything else raises."""
    import json as _json
    from botocore.exceptions import ClientError
    try:
        sfn.send_task_success(taskToken=task_token,
                              output=_json.dumps(payload))
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in TOKEN_GONE:
            raise
    return payload


def submit_or_park(lam, gateway_arn: str, target: str, tool: str,
                   payload: dict, skip):
    """Submit to a sibling gateway with a callback token and PARK, or
    degrade via `skip(reason)`. Encodes the hard-won discriminator
    (review F3): a response WITH a job_id means the service OWNS the
    token — park even if error text rides alongside; only a handle-less
    response is a refusal."""
    try:
        sub = invoke_gateway_tool(lam, gateway_arn, target, tool, payload)
    except Exception as exc:                         # noqa: BLE001 — degrade
        return skip(f"submit_failed: {str(exc)[:150]}")
    if not (isinstance(sub, dict) and sub.get("job_id")):
        msg = sub.get("message") if isinstance(sub, dict) else str(sub)
        return skip(f"submit_refused: {msg}")
    return {"parked": True, "job_id": sub["job_id"]}
