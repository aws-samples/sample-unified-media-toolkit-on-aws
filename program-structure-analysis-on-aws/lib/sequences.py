"""
Sequences for linear narrative: chains of scenes forming one narrative
movement (README.md §2) — the chapter-like unit between
scene and act.

Greedy in time order (KISS — no graph library): a scene JOINS the current
sequence when any continuity edge binds it to the sequence's recent
scenes:
  music bridge   one music span crossing the boundary — the editor's own
                 "same movement" mark, the strongest single cue
  same place     shared setting label, or a recurring shot group spanning
                 both scenes (DINO recurrence = "same location" where the
                 setting vocabulary is too coarse)
  same people    face-cluster overlap
CROSS-CUTS (A/B/A/B) are handled by offering a window of the last
RECALL_SCENES + 1 scenes (the immediate predecessor plus RECALL_SCENES
back), not one: a scene binding to the sequence a few scenes back
re-enters it, and the
interleaved foreign scenes become a parallel THREAD of the same sequence
(the scene-level intercut_return pattern, one level up).

Evidence rides on every join so a consumer can see WHY two scenes share a
sequence. No thresholds fitted to any programme: the edges are boolean
joins over shipped labels.
"""
from __future__ import annotations

RECALL_SCENES = 2      # how far back a cross-cut may re-enter


def sequences(scenes: list, shot_docs: list, shot_times: dict,
              recurrence: dict | None, stem_activity: dict | None) -> list:
    music = [(a / 1000.0, b / 1000.0) for a, b in
             (stem_activity or {}).get("activity", {}).get("music", [])]
    scene_groups = _recurrence_by_scene(scenes, recurrence)

    seqs = []
    for sc in sorted(scenes, key=lambda s: s["start_time"]):
        # micro-scenes (an insert, a beat of black, a 1-2 s transition —
        # no setting, no people) are connective tissue, not units: they
        # ride with the running sequence instead of breaking it
        if seqs and sc["end_time"] - sc["start_time"] < 4.0 \
                and not sc.get("setting") and not sc.get("people"):
            seqs[-1]["scenes"].append(sc)
            seqs[-1]["evidence"].append({"scene_id": sc["scene_id"],
                                         "via": "micro_scene"})
            continue
        joined = False
        # offer the last TWO sequences: a cross-cut's return scene binds to
        # the sequence one back (A/B/A) — binding there MERGES the
        # interleave into one sequence with threads (measured by the
        # synthetic cross-cut fixture: last-only can never re-enter A)
        offers = [-1] if seqs else []
        # a TRUE cross-cut alternates quickly: re-entry one sequence back
        # is only offered while the intervening sequence is still short —
        # binding back past a LONG sequence is a return to a location
        # (a new movement), not an interleave (measured: unconstrained
        # merging chained drama_a-102 12 -> 38 scenes)
        if len(seqs) >= 2 and len(seqs[-1]["scenes"]) <= RECALL_SCENES:
            offers.append(-2)
        for si in offers:
            # the joinable window: the immediate predecessor PLUS
            # RECALL_SCENES back — RECALL_SCENES + 1 members
            recent = seqs[si]["scenes"][-RECALL_SCENES - 1:]
            hit = _edge(sc, recent, music, scene_groups)
            if hit:
                if si == -2:
                    merged = seqs[-2]["scenes"] + seqs[-1]["scenes"]
                    merged.sort(key=lambda s: s["start_time"])
                    seqs[-2]["scenes"] = merged
                    seqs[-2]["evidence"] += seqs[-1]["evidence"]
                    seqs.pop()
                seqs[-1]["scenes"].append(sc)
                seqs[-1]["evidence"].append({"scene_id": sc["scene_id"],
                                             **hit})
                joined = True
                break
        if not joined:
            seqs.append({"scenes": [sc], "evidence": []})

    out = []
    for i, seq in enumerate(seqs):
        members = seq["scenes"]
        threads = _threads(members, music, scene_groups)
        out.append({
            "sequence_id": i,
            "start": members[0]["start_time"],
            "end": members[-1]["end_time"],
            "scene_ids": [s["scene_id"] for s in members],
            "threads": threads,
            "settings": sorted({s.get("setting") for s in members
                                if s.get("setting")}),
            "evidence": seq["evidence"],
        })
    return out


def _edge(sc: dict, recent: list, music: list, scene_groups: dict):
    """The first continuity edge binding sc to any recent member, with its
    evidence — None when nothing binds."""
    for other in reversed(recent):
        if _music_bridge(other, sc, music):
            return {"joins": other["scene_id"], "via": "music_bridge"}
        if sc.get("setting") and sc.get("setting") == other.get("setting"):
            return {"joins": other["scene_id"], "via": "setting",
                    "setting": sc["setting"]}
        shared_group = (scene_groups.get(sc["scene_id"], set())
                        & scene_groups.get(other["scene_id"], set()))
        if shared_group:
            return {"joins": other["scene_id"], "via": "recurrence",
                    "groups": sorted(shared_group)[:3]}
        cast = set(sc.get("people") or []) & set(other.get("people") or [])
        if cast:
            return {"joins": other["scene_id"], "via": "cast",
                    "people": sorted(cast)[:3]}
    return None


def _music_bridge(a: dict, b: dict, music: list) -> bool:
    """One music span covering the boundary between scene a's end and
    scene b's start — AND anchored to the movement: the cue must have
    STARTED during scene a. Without the anchor, a score-heavy programme
    (drama_b: 42% music coverage) chains transitively into one 60-scene
    mega-sequence (measured); an inherited bed is not an editorial mark,
    a cue placed inside the movement is."""
    lo, hi = a["end_time"] - 0.5, b["start_time"] + 0.5
    return any(ma < lo and mb > hi and ma >= a["start_time"] - 2.0
               for ma, mb in music)


def _threads(members: list, music: list, scene_groups: dict) -> list:
    """Partition a sequence's scenes into interleaved threads: consecutive
    members that DON'T bind pairwise belong to alternating threads (the
    cross-cut). Single-thread sequences return one thread."""
    threads = []
    for sc in members:
        placed = False
        for th in threads:
            if _edge(sc, th[-1:], music, scene_groups):
                th.append(sc)
                placed = True
                break
        if not placed:
            threads.append([sc])
    return [[s["scene_id"] for s in th] for th in threads]


def _recurrence_by_scene(scenes: list, recurrence: dict | None) -> dict:
    """scene_id -> set of recurring group_ids whose members fall inside."""
    spans = {s["scene_id"]: (s["start_shot_id"], s["end_shot_id"])
             for s in scenes}
    out = {sid: set() for sid in spans}
    for g in (recurrence or {}).get("recurrence_groups", []):
        for m in g.get("shots", []):
            for sid, (a, b) in spans.items():
                if a <= m["shot_id"] <= b:
                    out[sid].add(g["group_id"])
    return out
