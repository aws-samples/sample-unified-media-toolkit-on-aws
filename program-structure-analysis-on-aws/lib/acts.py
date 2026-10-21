"""
Break seams for linear narrative — PROSPECTIVE, never reconstructive
(README.md §8, a measured negative result).

On AS-AIRED captures an act boundary is a production artifact (black for
the ad pod + silence + bumpers) and the convergence detector below reads
it directly. On MASTERS there is nothing reliable to read: the drama_a
screeners carry NO black at their broadcast break positions, fades are
inconsistent, and silence/scene/music signals scored 9/27 recall at
top-8 candidates per episode against the 10-episode ground truth
(the AD corpus's align-map lag jumps). So:

- acts[] is emitted ONLY from delivery metadata or on as-aired evidence
  (blacks/bumpers present);
- otherwise the product is break_candidates[] — graded seams where a
  break COULD cleanly go (the ad-break §2 posture) — and NO act claims.
  Honest abstention over confident reconstruction.
"""
from __future__ import annotations

from lib import markers as markers_mod

EDGE_ZONE = 120.0        # seams this close to head/tail are titles/credits
SILENCE_NEAR = 3.0       # a silence gap within this of the black corroborates
RECURRENCE_NEAR = 8.0    # an act_boundary recurring shot within this
ACCEPT_SCORE = 2.0       # accept a seam at/above this


def candidate_seams(marks: dict, shot_times: dict, stem_activity: dict,
                    recurrence: dict | None, duration: float) -> list:
    """Interior black runs scored by converging evidence."""
    silences = markers_mod.silence_gaps(stem_activity)
    boundary_shots = _act_boundary_shots(recurrence)
    bumpers = marks.get("interior_title_cards") or []

    seams = []
    for blk in marks["blackframes"]:
        mid = (blk["start"] + blk["end"]) / 2
        if mid < EDGE_ZONE or mid > duration - EDGE_ZONE:
            continue
        evidence = {"black_sec": round(blk["end"] - blk["start"], 2)}
        score = 1.0
        if any(a - SILENCE_NEAR <= mid <= b + SILENCE_NEAR
               for a, b in silences):
            score += 1.0
            evidence["silent"] = True
        near_rec = [s for s in boundary_shots
                    if abs(_shot_mid(shot_times, s) - mid) <= RECURRENCE_NEAR]
        if near_rec:
            score += 1.0
            evidence["act_boundary_shots"] = near_rec[:4]
        near_bump = [b for b in bumpers
                     if abs((b["start"] + b["end"]) / 2 - mid) <= 15.0]
        if near_bump:
            score += 1.0
            evidence["bumper_card"] = round(near_bump[0]["start"], 1)
        seams.append({"at": round(mid, 2), "score": score,
                      "evidence": evidence})
    return sorted(seams, key=lambda s: s["at"])


def acts(marks: dict, shot_times: dict, stem_activity: dict,
         recurrence: dict | None, duration: float,
         accept_score: float = ACCEPT_SCORE) -> dict:
    seams = candidate_seams(marks, shot_times, stem_activity, recurrence,
                            duration)
    accepted = [s for s in seams if s["score"] >= accept_score]

    # content extent: after the head matter, before the credits
    content_start = 0.0
    if marks["opening_title"]:
        content_start = 0.0        # the teaser usually PRECEDES the title
    content_end = (marks["end_credits"][0]["start"]
                   if marks["end_credits"] else duration)

    bounds = [content_start] + [s["at"] for s in accepted] + [content_end]
    spans = []
    for i in range(len(bounds) - 1):
        spans.append({"start": round(bounds[i], 2),
                      "end": round(bounds[i + 1], 2),
                      "kind": "act",
                      "index": i + 1})
    # teaser/tag are positional refinements a consumer can derive from the
    # opening-title marker; kept out of v1 on purpose (KISS).
    return {"seams": seams, "accepted": accepted, "acts": spans}


def _act_boundary_shots(recurrence: dict | None) -> list:
    groups = (recurrence or {}).get("recurrence_groups") or []
    return [m["shot_id"] for g in groups for m in g.get("shots", [])
            if m.get("position") == "act_boundary"]


def _shot_mid(shot_times: dict, sid: int) -> float:
    a, b = shot_times.get(sid, (0.0, 0.0))
    return (a + b) / 2
