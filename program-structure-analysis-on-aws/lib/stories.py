"""
News stories: CA's discourse segments joined to the video with the three
boundary rules from the measured promo-absorption defect
(README.md §7):

1. START SNAP (fuse_rundown's proven rule): a rundown item starts at the
   DIRECTOR'S CUT — snap each segment's start back to the start of the
   shot containing its first cue, <=SNAP_WINDOW_SEC, never at/before the
   previous item's end.
2. CUELESS video between segments is assigned by VISUAL ROLE, never by
   adjacency: story-role shots (anchor desk, field, interview) join the
   story side; interstitial-role shots (full-screen graphics) join the
   interstitial side; when both sides are the same class, broadcast
   grammar says the material leads IN — it goes to the NEXT item.
3. Every story carries THREE timestamps: start_sec (editorial),
   content_start_sec (first story-role shot — "the actual conversation
   begins", the timestamp the POC lacked), speech_start_sec (first word).

Interstitials (tease/promo/headlines/handoff) are FIRST-CLASS rows — a
tease shares its story's slug (identity), but identity never merges spans.
"""
from __future__ import annotations

from lib import roles as roles_mod

SNAP_WINDOW_SEC = 6.0
INTERSTITIAL_FORMS = {"tease", "promo", "headlines_block", "handoff_banter"}


def _hms(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def stories(ca: dict, shot_docs: list, shot_times: dict,
            recurrence: dict | None, role_of: dict | None = None,
            fences: list | None = None) -> list:
    """CA audio_segments -> rundown rows with role-resolved video spans.
    A caller that already computed shot_roles passes it in (assemble does)
    so the rundown's roles and the segmentation's roles are ONE map.
    `fences` are the break-candidate times (assemble passes structure's
    own break_candidates): rule 2 never assigns video ACROSS a fence, so
    a commercial pod between two stories stays uncovered instead of being
    absorbed into the next story's span (measured on a 2 h broadcast:
    12/19 break candidates sat strictly inside story spans)."""
    if role_of is None:
        role_of = roles_mod.shot_roles(shot_docs, recurrence)
    ordered = sorted(shot_times.items(), key=lambda kv: kv[1][0])

    rows = []
    for seg in ca.get("audio_segments", []):
        speech_start = _hms(seg["start_time"])
        speech_end = _hms(seg["end_time"])
        form = seg.get("news_form") or seg.get("conversation_type") or ""
        rows.append({
            "slug": seg.get("slug"),
            "headline": seg.get("segment_topic"),
            "news_form": form,
            "interstitial": form in INTERSTITIAL_FORMS,
            "interaction": seg.get("interaction"),
            "correspondent": seg.get("correspondent"),
            "sig_out": seg.get("sig_out"),
            "speech_start_sec": round(speech_start, 2),
            "speech_end_sec": round(speech_end, 2),
        })

    # rule 1: START SNAP against the shot lattice. prev_end is None for
    # the FIRST item — there is nothing to cross, so a shot starting at
    # 0.0 is a legal snap target (0.0 initial refused it).
    prev_end = None
    for row in rows:
        snapped = row["speech_start_sec"]
        for sid, (a, b) in ordered:
            if a <= row["speech_start_sec"] <= b:
                if row["speech_start_sec"] - a <= SNAP_WINDOW_SEC \
                        and (prev_end is None or a > prev_end):
                    snapped = a
                break
        row["start_sec"] = round(snapped, 2)
        prev_end = row["speech_end_sec"]

    # rule 2: the cueless video between consecutive items — assign each
    # gap shot by role, then move the receiving item's boundary. A BREAK
    # FENCE overrides the role vote: a break candidate is the director's
    # own seam (black + silence + bumper evidence), stronger than any
    # per-shot role guess. Shots before the gap's first fence trail the
    # previous item, shots after its last fence lead the next in, and
    # shots between two fences (the pod itself) join NEITHER — the pod
    # stays uncovered, described by structure.json's markers and breaks.
    for prev, nxt in zip(rows, rows[1:]):
        gap_shots = [(sid, ab) for sid, ab in ordered
                     if ab[0] >= prev["speech_end_sec"] - 0.5
                     and ab[1] <= nxt["start_sec"] + 0.5]
        cuts = sorted(f for f in (fences or [])
                      if prev["speech_end_sec"] < f < nxt["start_sec"])
        if cuts:
            first, last = cuts[0], cuts[-1]
            for sid, (a, b) in gap_shots:
                mid = (a + b) / 2
                if mid < first:
                    prev["end_sec"] = round(
                        max(prev.get("end_sec") or b, b), 2)
                elif mid > last and a < nxt["start_sec"]:
                    nxt["start_sec"] = round(a, 2)
            continue
        for sid, (a, b) in gap_shots:
            to_next = _assign_to_next(role_of.get(sid), prev, nxt)
            if to_next and a < nxt["start_sec"]:
                nxt["start_sec"] = round(a, 2)
            elif not to_next:
                prev["end_sec"] = round(b, 2)

    # spans + rule 3: content_start = first STORY-role shot in the span
    for i, row in enumerate(rows):
        nxt_start = rows[i + 1]["start_sec"] if i + 1 < len(rows) else None
        # explicit None test, NOT truthiness: 0.0 is a real timestamp
        end = row.get("end_sec")             # rule 2 may have set it
        row["end_sec"] = (row["speech_end_sec"] if end is None
                          else max(end, row["speech_end_sec"]))
        if nxt_start is not None:
            row["end_sec"] = min(max(row["end_sec"], row["start_sec"]),
                                 nxt_start)
        row["content_start_sec"] = _content_start(row, ordered, role_of)
    return rows


def _assign_to_next(role_entry: dict | None, prev: dict, nxt: dict) -> bool:
    """Rule 2's vote for ONE gap shot. Story-role video belongs with the
    story side; graphics with the interstitial side; ties lead IN."""
    if role_entry is None:
        return True
    story_shot = roles_mod.is_story_role(role_entry)
    if story_shot and not nxt["interstitial"]:
        return True
    if story_shot and not prev["interstitial"]:
        return False
    if not story_shot and nxt["interstitial"]:
        return True
    if not story_shot and prev["interstitial"]:
        return False
    return True                      # tie: broadcast material leads in


def _content_start(row: dict, ordered: list, role_of: dict) -> float:
    for sid, (a, b) in ordered:
        if b <= row["start_sec"] or a >= row["end_sec"]:
            continue
        if roles_mod.is_story_role(role_of.get(sid, {"role": "", "anchor_set": False})):
            return round(max(a, row["start_sec"]), 2)
    return row["speech_start_sec"]
