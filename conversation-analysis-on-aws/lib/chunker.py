"""
Chunk planning (README.md §5) — validated on three real assets; the
pinned chunk plans in test/test_units.py are the executable record.

The cut-tier ladder, in priority order inside a search window at the end of
each budget-full chunk:

    a  strong pause  gap >= PAUSE_STRONG_S   disjoint (no context cues)
    b  weak pause    gap >= PAUSE_WEAK_S     cut + context cues
    c  speaker change                        cut + context cues
    d  hard cut (single-speaker, no pause)   cut + context cues

Two pause tiers because 3 s is enough to cut AT but not BLIND: markets_a's
gaps are bimodal (nothing in [5 s, 10 s); breaks are 17-20 s) but news_a has a
3 s gap every two minutes — music stings inside one topic. Only a strong
pause is trusted as a topic boundary; every other cut carries context so
the model can flag `continues_previous`.

Gap math tolerates NEGATIVE gaps (diarization emits them on crosstalk —
measured -0.38 s on markets_a); nothing here assumes monotonic timestamps.

Positions vs ids: the planner walks LIST POSITIONS (always contiguous) and
reports both positions and the sentences' ids. For diarization input they
coincide; for VTT-derived ids they may not, and coverage semantics follow
ids because that is what the model sees.
"""
from __future__ import annotations

# Flat-zip top-level import (the family's copied-module layout): lib/
# ships at the zip ROOT, so siblings import as top-level modules.
from lib.envelope import estimate_cue_tokens

PAUSE_STRONG_S = 10.0
PAUSE_WEAK_S = 3.0
OVERLAP_MAX_S = 30.0
CUT_SEARCH_FRACTION = 0.15

TIER_STRONG_PAUSE = "strong-pause"
TIER_WEAK_PAUSE = "weak-pause"
TIER_SPEAKER_CHANGE = "speaker-change"
TIER_HARD_CUT = "hard-cut"
TIER_END = "end"

# Tiers whose cut carries context cues into the NEXT chunk.
_CONTEXT_TIERS = (TIER_WEAK_PAUSE, TIER_SPEAKER_CHANGE, TIER_HARD_CUT)


def plan_chunks(sentences: list[dict], budget_tokens: float, *,
                pause_strong_s: float = PAUSE_STRONG_S,
                pause_weak_s: float = PAUSE_WEAK_S,
                overlap_max_s: float = OVERLAP_MAX_S,
                search_fraction: float = CUT_SEARCH_FRACTION) -> list[dict]:
    """
    -> [{"index", "start_pos", "end_pos", "start_id", "end_id",
         "context_start_pos" (None on disjoint/first), "cut_tier",
         "cut_gap_s", "est_tokens"}, ...]

    `cut_tier` describes how THIS chunk ends (TIER_END on the last);
    `context_start_pos` describes how it begins (context inherited from
    the PREVIOUS chunk's cut).
    """
    if not sentences:
        raise ValueError("plan_chunks: sentences is empty")
    if budget_tokens <= 0:
        raise ValueError("plan_chunks: budget_tokens must be positive")

    tokens = [estimate_cue_tokens(s) for s in sentences]
    n = len(sentences)
    chunks: list[dict] = []
    pos = 0
    context_start: int | None = None
    while pos < n:
        total, j = 0.0, pos
        while j < n and total < budget_tokens:
            total += tokens[j]
            j += 1
        if j >= n:
            chunks.append(_chunk(len(chunks), pos, n - 1, context_start,
                                 TIER_END, 0.0, sentences, tokens))
            break

        cut_pos, tier, gap = _pick_cut(sentences, pos, j, search_fraction,
                                       pause_strong_s, pause_weak_s)
        chunks.append(_chunk(len(chunks), pos, cut_pos, context_start,
                             tier, gap, sentences, tokens))
        if tier in _CONTEXT_TIERS:
            context_start = _context_start(sentences, cut_pos, overlap_max_s)
        else:
            context_start = None
        pos = cut_pos + 1
    return chunks


def _chunk(index, start_pos, end_pos, context_start_pos, tier, gap,
           sentences, tokens) -> dict:
    return {
        "index": index,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "start_id": sentences[start_pos]["id"],
        "end_id": sentences[end_pos]["id"],
        "context_start_pos": context_start_pos,
        "cut_tier": tier,
        "cut_gap_s": round(gap, 2),
        "est_tokens": round(sum(tokens[start_pos:end_pos + 1]), 1),
    }


def _pick_cut(sentences, start, budget_end, search_fraction,
              pause_strong_s, pause_weak_s):
    """Choose the cut position in the window ending at budget_end (excl)."""
    span = budget_end - start
    window_start = max(start + 1, budget_end - max(2, int(span
                                                          * search_fraction)))
    gaps = []               # (gap_s, position of the cue the chunk ENDS on)
    for k in range(window_start - 1, budget_end - 1):
        gaps.append((sentences[k + 1]["start"] - sentences[k]["end"], k))

    strong = [g for g in gaps if g[0] >= pause_strong_s]
    if strong:
        gap, k = max(strong)
        return k, TIER_STRONG_PAUSE, gap
    weak = [g for g in gaps if g[0] >= pause_weak_s]
    if weak:
        gap, k = max(weak)
        return k, TIER_WEAK_PAUSE, gap
    changes = [k for g, k in gaps
               if sentences[k + 1]["speaker_id"] != sentences[k]["speaker_id"]]
    if changes:
        k = changes[-1]
        return k, TIER_SPEAKER_CHANGE, \
            sentences[k + 1]["start"] - sentences[k]["end"]
    return budget_end - 2, TIER_HARD_CUT, 0.0


def _context_start(sentences, cut_pos, overlap_max_s) -> int:
    """Context cues for the NEXT chunk: walk back from the cut through the
    preceding complete speaker exchange (until the speaker changes a second
    time), capped at overlap_max_s of timeline."""
    cut_end = sentences[cut_pos]["end"]
    k = cut_pos
    changes_seen = 0
    while k > 0:
        if cut_end - sentences[k - 1]["start"] > overlap_max_s:
            break
        if sentences[k - 1]["speaker_id"] != sentences[k]["speaker_id"]:
            changes_seen += 1
            if changes_seen >= 2:
                break
        k -= 1
    return k
