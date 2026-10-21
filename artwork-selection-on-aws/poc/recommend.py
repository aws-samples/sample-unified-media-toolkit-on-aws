#!/usr/bin/env python3
"""Thumbnail candidates per ARTWORK_SELECTION_PLAN stages 1-3:
code filters/ranks shots from existing metadata, grabs frames locally.
Usage: recommend.py WORKDIR VIDEO   (WORKDIR holds shots.json,
shot_metadata.jsonl, cast_persons.json; frames land beside them)."""
import json, subprocess, sys, os

wd, video = sys.argv[1], sys.argv[2]
os.chdir(wd)
shots = {s["shot_id"]: s for s in json.load(open("shots.json"))["shots"]}
docs = {d["shot_id"]: d for d in map(json.loads, open("shot_metadata.jsonl"))}
cast = json.load(open("cast_persons.json"))["persons"]
top_cast = [p["name"] for p in sorted(
    (p for p in cast if p.get("name")),
    key=lambda p: -(p.get("screen_sec") or 0))][:5]
print("top cast:", top_cast)
duration = max(s["end_time"] for s in shots.values())
spoiler_cut = duration * 0.75

def disqualified(sid):
    d, s = docs.get(sid), shots.get(sid)
    if not d or not s: return True
    return (d.get("shot_kind") != "normal" or d.get("texts_on_screen")
            or d.get("unreadable_text_present") or d.get("brands")
            or d.get("depicted")
            or s["end_time"] - s["start_time"] < 1.2
            or s["start_time"] > spoiler_cut)

def rank_key(sid):
    s = shots[sid]
    return (-min(s["end_time"] - s["start_time"], 8.0),
            s.get("frame_motion_mean") or 0.0,
            -(s.get("entropy_mean") or 0.0))

def pick(pred, n=4):
    cands = sorted((sid for sid in docs
                    if not disqualified(sid) and pred(docs[sid])),
                   key=rank_key)[:n]
    out = []
    for sid in cands:
        s, d = shots[sid], docs[sid]
        out.append({"shot": sid,
                    "t": round((s["start_time"] + s["end_time"]) / 2, 2),
                    "camera": d.get("camera_shot_type"),
                    "desc": d.get("description", "")[:110]})
    return out

lead = top_cast[0] if top_cast else ""
arch = {
  "hero": pick(lambda d: d.get("camera_shot_type")
               in ("close_up", "medium_close_up")
               and d.get("people_count") == "1"
               and lead in (d.get("people") or [])),
  "two": pick(lambda d: d.get("people_count") == "2"
              and d.get("camera_shot_type")
              in ("medium", "medium_wide", "over_the_shoulder")
              and any(n in (d.get("people") or []) for n in top_cast)),
  "vista": pick(lambda d: d.get("camera_shot_type")
                in ("wide", "extreme_wide", "aerial")
                and d.get("people_count") in ("0", "several", None)),
}
json.dump(arch, open("candidates.json", "w"), indent=1)
for k, v in arch.items():
    print(f"== {k}: {len(v)}")
    for c in v:
        print(f"   shot {c['shot']:4} @{c['t']:7.1f}s "
              f"{str(c['camera']):>15} {c['desc']}")
        subprocess.run(["ffmpeg", "-y", "-v", "error",  # nosec B603
                        "-ss", str(c["t"]), "-i", video,
                        "-frames:v", "1", "-q:v", "2",
                        f"{k}_shot{c['shot']:04d}.jpg"])
