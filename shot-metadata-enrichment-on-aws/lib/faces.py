"""
The face join (README.md §5): face-recognition's facemetadata.json → per-shot
name lists + the per-window reference gallery plan.

Two rules, both measured (README §11, GOTCHAS):

- **A name list is not a join.** `{{face_names}}` must arrive already
  resolved per shot — a bare cast list invites the model to distribute
  names by plausibility. The join here is appearance-span x shot-span
  overlap on the timeline both services share.
- **Names alone cannot BIND once a shot holds two named people** — the
  model assigns by list order. The gallery (one labeled reference crop per
  named person, composed into ONE image) is what binds; `rep_face_key` is
  face-recognition's single representative face crop (its PLAN §22 —
  NEVER `thumbnail_key`, which is a 16-cell debug montage).

Only NAMED people participate: an unnamed cluster contributes nothing the
prompt may use (generic labels are the model's own job).
"""
from __future__ import annotations


def named_people(facemetadata: dict) -> list:
    """people[] entries that carry a real name AND a rep-face pointer,
    with appearance spans in seconds."""
    people = []
    for p in facemetadata.get("people") or []:
        if not p.get("named") or not p.get("name"):
            continue
        spans = [(a["begin_ms"] / 1000.0, a["end_ms"] / 1000.0)
                 for a in p.get("appearances") or []]
        if not spans:
            continue
        people.append({
            "person_id": p["person_id"],
            "name": p["name"],
            "rep_face_key": p.get("rep_face_key"),
            "total_screen_time_sec": p.get("total_screen_time_sec") or 0.0,
            "spans": spans,
        })
    return people


def shots_with_faces(shots: list, people: list) -> dict:
    """shot_id -> [name, ...] for every shot whose span overlaps a named
    person's appearance span. Deterministic name order: screen time desc,
    then name."""
    ranked = sorted(people, key=lambda p: (-p["total_screen_time_sec"],
                                           p["name"]))
    out = {}
    for shot in shots:
        s0, s1 = float(shot["start_time"]), float(shot["end_time"])
        names = [p["name"] for p in ranked
                 if any(b < s1 and e > s0 for b, e in p["spans"])]
        if names:
            out[shot["shot_id"]] = names
    return out


def window_gallery(window_shot_ids: list, shot_names: dict,
                   people: list, max_faces: int) -> list:
    """The window's gallery roster: unique named people appearing in this
    window, ranked by within-window appearance count (README §5), tie-break
    total screen time then name. Returns the people entries, capped."""
    counts = {}
    for sid in window_shot_ids:
        for name in shot_names.get(sid) or []:
            counts[name] = counts.get(name, 0) + 1
    by_name = {p["name"]: p for p in people}
    roster = sorted(
        (name for name in counts if name in by_name),
        key=lambda n: (-counts[n],
                       -by_name[n]["total_screen_time_sec"], n))
    return [by_name[n] for n in roster[:max_faces]]


def face_names_text(window_shot_ids: list, context_shot_id,
                    shot_names: dict) -> str:
    """The {{face_names}} value: per-shot lines, resolved — or the sentinel
    (GOTCHAS: optional inputs are still REQUIRED tokens; the sentinel is a
    value, never an empty string the model could misread)."""
    ids = ([context_shot_id] if context_shot_id is not None else []) \
        + list(window_shot_ids)
    lines = [f"shot {sid}: {', '.join(shot_names[sid])}"
             for sid in ids if sid in shot_names]
    if not lines:
        return "(no face recognition supplied for this window)"
    lines.append("(shots not listed carry no recognized faces)")
    return "\n".join(lines)
