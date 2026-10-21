"""
CPL-style programme markers, assembled CODE-ONLY from Layer-1 evidence
(README.md §8): blackframes, technical slate, bars+tone,
opening/main title, interior title cards, end credits + logos, textless
tail. Every judgment here already happened upstream (shot_kind is
enrichment's model call; tones are audiostem's detector); this module is
runs, positions and joins — recomputable for free.

This is the ONE home for the nondiegetic-block predicate going forward
(frame-inspection's lib/spans.py demotes to a fallback per
README.md §8's build-order note).
"""
from __future__ import annotations

NONDIEGETIC = {"credit_roll", "title_card", "logo_bumper", "slate"}
BLACK_BRIDGE_MAX = 5.0     # black this long is a beat inside a block
HEAD_ZONE_FRAC = 0.25      # opening titles live in the first quarter
MIN_BLACK_MARKER = 0.4     # shorter black is a cut artifact, not a marker


def shot_kinds(shot_docs: list) -> dict:
    return {d["shot_id"]: d.get("shot_kind") or "normal" for d in shot_docs}


def runs_of(kinds: dict, shot_times: dict, wanted: set,
            bridge_black: bool = False) -> list:
    """Contiguous runs of the wanted shot kinds, optionally bridged over
    short black. The universal assembly primitive here."""
    out, cur = [], []
    for sid in sorted(shot_times):
        kind = kinds.get(sid)
        span = shot_times[sid][1] - shot_times[sid][0]
        if kind in wanted:
            cur.append(sid)
        elif bridge_black and kind == "black" and cur \
                and span <= BLACK_BRIDGE_MAX:
            cur.append(sid)
        else:
            if any(kinds.get(s) in wanted for s in cur):
                out.append(cur)
            cur = []
    if any(kinds.get(s) in wanted for s in cur):
        out.append(cur)
    return [{"start": shot_times[r[0]][0], "end": shot_times[r[-1]][1],
             "shot_ids": r,
             "kinds": sorted({kinds[s] for s in r if kinds.get(s) in wanted})}
            for r in out]


def markers(shot_docs: list, shot_times: dict, stem_activity: dict | None,
            duration: float) -> dict:
    """The marker catalogue. Evidence in, labeled spans out."""
    kinds = shot_kinds(shot_docs)

    blacks = [r for r in runs_of(kinds, shot_times, {"black"})
              if r["end"] - r["start"] >= MIN_BLACK_MARKER]
    slates = runs_of(kinds, shot_times, {"slate"})
    bars = runs_of(kinds, shot_times, {"bars"})
    for b in bars:                              # audiostem's tone detector
        b["tone_confirmed"] = _tones_overlap(stem_activity, b)

    nondiegetic = runs_of(kinds, shot_times, NONDIEGETIC, bridge_black=True)
    head_zone = duration * HEAD_ZONE_FRAC
    opening = [b for b in nondiegetic
               if b["start"] < head_zone and "title_card" in b["kinds"]
               and "credit_roll" not in b["kinds"]]
    # a credit_roll is end-credits material wherever it sits; a LOGO-only
    # block is only end-credits in the tail (symmetric to the head zone) —
    # without the positional test a head studio logo becomes end_credits[0]
    # and poisons everything anchored to it (content_end, textless tail)
    credits = [b for b in nondiegetic
               if "credit_roll" in b["kinds"]
               or ("logo_bumper" in b["kinds"]
                   and b["start"] >= duration - head_zone)]
    interior_titles = [b for b in nondiegetic
                       if b not in opening and b not in credits]

    # textless tail: normal picture AFTER the last credits block reaching
    # (near) the end of programme — IMF textless material at end of tape
    textless = []
    if credits and credits[-1]["end"] >= duration * 0.8:
        after = [sid for sid in sorted(shot_times)
                 if shot_times[sid][0] >= credits[-1]["end"]
                 and kinds.get(sid) == "normal"]
        if after:
            docs = {d["shot_id"]: d for d in shot_docs}

            def _has_post_text(d):
                # texts_on_screen is the UNCLAIMED remainder since SME
                # 2026-09-09 — an extracted title-card name lives only in
                # named_entities as overlay_text, and text over the tail
                # disqualifies it as textless either way
                return bool(d.get("texts_on_screen")) or any(
                    isinstance(e, dict)
                    and e.get("evidence") == "overlay_text"
                    for e in d.get("named_entities") or [])

            if not any(_has_post_text(docs.get(s, {})) for s in after):
                textless.append({"start": shot_times[after[0]][0],
                                 "end": shot_times[after[-1]][1],
                                 "shot_ids": after})

    return {"blackframes": blacks, "slate": slates, "bars_and_tone": bars,
            "opening_title": opening, "interior_title_cards": interior_titles,
            "end_credits": credits, "textless_tail": textless}


def silence_gaps(stem_activity: dict, min_gap: float = 1.0) -> list:
    """(start, end) seconds where the ORIGINAL mix is silent — the seams
    act breaks and pod boundaries live in."""
    spans = [(a / 1000.0, b / 1000.0)
             for a, b in (stem_activity or {}).get("activity", {})
                                              .get("original", [])]
    gaps, prev_end = [], 0.0
    for a, b in spans:
        if a - prev_end >= min_gap:
            gaps.append((prev_end, a))
        prev_end = max(prev_end, b)
    return gaps


def _tones_overlap(stem_activity: dict | None, span: dict) -> bool:
    for t in (stem_activity or {}).get("tones", []):
        if t.get("nominal_hz") is None:
            continue
        a, b = t["start_ms"] / 1000.0, t["end_ms"] / 1000.0
        if a < span["end"] and b > span["start"]:
            return True
    return False
