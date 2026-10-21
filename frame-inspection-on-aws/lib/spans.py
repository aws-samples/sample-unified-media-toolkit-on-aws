"""
Span triggers: WHERE to look, decided mechanically from shot metadata —
no model, no content-specific rules (README.md §1, §7).

A span is {"start": s, "end": s} in programme seconds, optionally with an
"id" and a "question" (claim-verify supplies both). Two auto-triggers:

- nondiegetic_blocks: contiguous runs of credit/title/logo/slate shots,
  bridged across short black — where credits-read looks.
- unreadable_text_spans: shots flagged unreadable_text_present OUTSIDE
  those blocks — supers over picture, where title-read looks.

Both predicates were validated in the AD prototype across 31 episodes of
three series before landing here.

KNOWN LIMIT (measured, mitj404 shot 371): unreadable_text_spans inherits
the per-shot pass's recall on the flag — a super the enrichment's 1-3
sampled frames never SAW is neither read nor flagged, so no trigger
fires. Explicit caller spans are the override; do not compensate here
with heuristics.
"""
from __future__ import annotations

NONDIEGETIC = {"credit_roll", "title_card", "logo_bumper", "slate"}
BLACK_BRIDGE_MAX = 5.0   # black this long is a beat inside a block;
                         # longer and the block has ended


def nondiegetic_blocks(shot_docs: list, shot_times: dict) -> list:
    """Contiguous nondiegetic runs as spans.

    shot_docs: shot_metadata.jsonl rows (dicts with shot_id, shot_kind).
    shot_times: {shot_id: (start_time, end_time)} from shots.json.
    """
    kinds = {d["shot_id"]: d.get("shot_kind") for d in shot_docs}
    blocks, current = [], []
    for sid in sorted(shot_times):
        kind = kinds.get(sid)
        span = shot_times[sid][1] - shot_times[sid][0]
        if kind in NONDIEGETIC:
            current.append(sid)
        elif kind == "black" and current and span <= BLACK_BRIDGE_MAX:
            current.append(sid)
        else:
            if _has_nondiegetic(current, kinds):
                blocks.append(current)
            current = []
    if _has_nondiegetic(current, kinds):
        blocks.append(current)
    return [{
        "start": shot_times[b[0]][0],
        "end": shot_times[b[-1]][1],
        "shot_ids": b,
        "kinds": sorted({kinds[s] for s in b if kinds.get(s) in NONDIEGETIC}),
    } for b in blocks]


def unreadable_text_spans(shot_docs: list, shot_times: dict) -> list:
    """Shots carrying text the per-shot pass could not read, outside the
    nondiegetic blocks: supers, location cards, chapter titles over
    picture (the mitj404 'Based on the book...' case)."""
    blocked = set()
    for blk in nondiegetic_blocks(shot_docs, shot_times):
        blocked.update(blk["shot_ids"])
    spans = []
    for d in shot_docs:
        sid = d["shot_id"]
        if sid in blocked or sid not in shot_times:
            continue
        if d.get("unreadable_text_present"):
            spans.append({
                "start": shot_times[sid][0],
                "end": shot_times[sid][1],
                "shot_ids": [sid],
                "kinds": [d.get("shot_kind") or "normal"],
            })
    return spans


def _has_nondiegetic(shot_ids: list, kinds: dict) -> bool:
    return any(kinds.get(s) in NONDIEGETIC for s in shot_ids)
