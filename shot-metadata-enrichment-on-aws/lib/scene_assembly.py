"""
Scene assembly (README.md §4): trivial deterministic grouping over the
repaired `starts_new_scene` flags — a scene runs from one `true` to the
shot before the next `true`. AHC/statistical clustering is retired; the
interleaved-genre grouping prompt (§4's second path) slots in LATER
without touching this pass, so this module stays flags-only.

Scene-level attributes are AGGREGATED IN CODE, never read off a single
shot (GOTCHAS: per-window independence causes attribute drift INSIDE a
scene — the same room answered `day` in one window and `indeterminate` in
the next; majority vote across members is the stable answer).

A shot missing from the model output (an honest coverage gap) BREAKS
nothing: it is carried as a member without evidence, listed in the
scene's `shots_gap`, and never starts a scene.
"""
from __future__ import annotations

from collections import Counter


def assemble_scenes(docs: dict, shots_by_id: dict,
                    ordered_ids: list) -> list:
    """docs: shot_id -> normalized doc (gaps absent). shots_by_id: the
    shot-segmentation records (authoritative times). ordered_ids: every
    enriched shot_id in order. -> scenes[]"""
    groups: list = []
    for sid in ordered_ids:
        doc = docs.get(sid)
        starts = bool(doc and doc.get("starts_new_scene"))
        if not groups or (starts and groups[-1]):
            groups.append([])
        groups[-1].append(sid)
    # The very first group opens implicitly even when its first shot's doc
    # is a gap — an empty leading group can only appear if ordered_ids is
    # empty, guarded by the caller.

    scenes = []
    for idx, members in enumerate(groups):
        member_docs = [docs[s] for s in members if s in docs]
        first, last = shots_by_id[members[0]], shots_by_id[members[-1]]
        opener = docs.get(members[0]) or {}
        scenes.append({
            "scene_id": idx,
            "start_shot_id": members[0],
            "end_shot_id": members[-1],
            "shot_count": len(members),
            "start_time": first["start_time"],
            "end_time": last["end_time"],
            "start_timecode": first.get("start_timecode"),
            "end_timecode": last.get("end_timecode"),
            "boundary_reason": opener.get("boundary_reason"),
            # Majority vote across members — never the first shot's answer.
            "setting": _majority(member_docs, "setting"),
            "time_of_day": _majority(member_docs, "time_of_day"),
            "sentiment": _majority(member_docs, "sentiment"),
            # Unions keep first-seen order (stable, join-friendly).
            "people": _union(member_docs, "people"),
            "notable": _union(member_docs, "notable"),
            "brands": _union_named(member_docs, "brands"),
            "named_entities": _union_named(member_docs, "named_entities"),
            "shot_descriptions": [
                {"shot_id": d["shot_id"], "description": d["description"]}
                for d in member_docs],
            "shots_gap": [s for s in members if s not in docs],
        })
    return scenes


def _majority(docs: list, field: str):
    votes = Counter(d[field] for d in docs
                    if isinstance(d.get(field), str) and d[field])
    if not votes:
        return None
    top = votes.most_common()
    # Deterministic tie-break: count desc, then value — same inputs, same
    # scene document, byte for byte.
    return sorted(top, key=lambda kv: (-kv[1], kv[0]))[0][0]


def _union(docs: list, field: str) -> list:
    seen, out = set(), []
    for d in docs:
        for v in d.get(field) or []:
            if isinstance(v, str) and v not in seen:
                seen.add(v)
                out.append(v)
    return out


def _union_named(docs: list, field: str) -> list:
    """Dict-valued lists (brands, named_entities) dedupe on `name` — the
    first occurrence keeps its evidence fields."""
    seen, out = set(), []
    for d in docs:
        for v in d.get(field) or []:
            name = v.get("name") if isinstance(v, dict) else None
            if name and name not in seen:
                seen.add(name)
                out.append(v)
    return out
