#!/usr/bin/env python3
"""
build_compare.py — SME model-comparison page builder.

Joins N shot-metadata-enrichment runs over the SAME asset (same shot ids)
into the compare_template.html DATA contract: per-shot field matrix,
divergence scoring, per-model boundary lanes, scoreboard, thumbnails.

Usage:
  python3 build_compare.py config.json out.html

config.json:
{
  "title": "...", "subtitle": "...", "note": "...",
  "video_url": "https://presigned...",      # presign via media-outputs tools
  "shots_json": "path/shots.json",          # shot-segmentation output
  "vtt": "path/transcript.vtt",             # optional, "" to skip
  "shot_images_dir": "path/shot_images",    # shot_{NNNNNN}.jpg grids
  "shot_frames_dir": "path/shot_frames",    # shot_{NNNN}_frames.tar.gz
  "baseline": "sonnet46",
  "models": [                               # display order; baseline first
    {"key": "sonnet46", "label": "sonnet-4-6", "dir": "runs/sonnet46",
     "wall_sec": 512.9}
  ]
}

Deterministic: same inputs, same page (thumbnails re-encode stably with
a fixed quality; the presigned URL is the only varying input).
"""
import base64
import io
import json
import sys

# Structured fields compared for divergence, in matrix display order.
# `description` is deliberately NOT scored (free text always differs);
# it renders as the stacked block under the matrix.
FIELDS = [
    "people_count", "sentiment", "starts_new_scene", "boundary_reason",
    "shot_kind", "camera_shot_type", "time_of_day", "lighting",
    "people", "named_entities", "brands", "objects", "actions",
    "expressions", "notable", "depicted", "media_depictions",
    "texts_on_screen", "unreadable_text_present",
]
LIST_AS_SET = {"lighting", "people", "named_entities", "brands", "objects",
               "actions", "expressions", "notable", "depicted",
               "media_depictions"}  # order-insensitive compare
THUMB_H = 44
THUMB_Q = 60


def norm(field, val):
    """(canonical-for-compare, display) for one field value."""
    if val is None:
        return "", "—"
    if isinstance(val, bool):
        return str(val).lower(), str(val).lower()
    if isinstance(val, (int, float)):
        return str(val), str(val)
    if isinstance(val, str):
        s = val.strip()
        return s, (s if s else "—")
    if isinstance(val, list):
        items = []
        for x in val:
            if isinstance(x, dict):
                items.append(str(x.get("name") or x.get("text") or
                                 json.dumps(x, sort_keys=True)))
            else:
                items.append(str(x))
        if field in LIST_AS_SET:
            items = sorted(set(items))
        disp = ", ".join(items) if items else "—"
        return json.dumps(items), disp
    return json.dumps(val, sort_keys=True), json.dumps(val)


def load_run(d):
    docs = {}
    for line in open(f"{d}/shot_metadata.jsonl"):
        doc = json.loads(line)
        docs[int(doc["shot_id"])] = doc
    scenes = json.load(open(f"{d}/scenes.json"))
    usage = json.load(open(f"{d}/usage_tracking.json"))
    return docs, scenes, usage


def main(cfg_path, out_path):
    cfg = json.load(open(cfg_path))
    tmpl_dir = __file__.rsplit("/", 1)[0]
    template = open(f"{tmpl_dir}/compare_template.html").read()

    shots_env = json.load(open(cfg["shots_json"]))
    shot_times = {int(s["shot_id"]): (s["start_time"], s["end_time"])
                  for s in shots_env["shots"]}

    runs, models_out = {}, []
    for m in cfg["models"]:
        docs, scenes, usage = load_run(m["dir"])
        runs[m["key"]] = docs
        gaps = len(scenes.get("shots_gap") or [])
        total = scenes.get("shots_total") or len(docs)
        failed = sum(r.get("records_failed") or 0
                     for r in usage.get("inference_runs") or [])
        mode = (usage.get("inference_runs") or [{}])[0].get(
            "inference_mode", "?")
        models_out.append({
            "key": m["key"], "label": m["label"], "mode": mode,
            "cost": usage.get("total_estimated_cost") or 0.0,
            "tokens_in": usage.get("total_input_tokens") or 0,
            "tokens_out": usage.get("total_output_tokens") or 0,
            "wall_sec": m.get("wall_sec") or 0,
            "coverage": f"{total - gaps}/{total}", "gaps": gaps,
            "scenes": len(scenes.get("scenes") or []),
            "stops": ("ok" if not failed and not gaps
                      else f"{failed} failed · {gaps} gap"),
            "stops_ok": not failed and not gaps,
        })
    # baseline first
    models_out.sort(key=lambda m: m["key"] != cfg["baseline"])
    mkeys = [m["key"] for m in models_out]

    def hms(t):
        return f"{int(t // 60):02d}:{int(t % 60):02d}"

    def thumb(sid):
        """One representative frame per shot for the strip: the MIDDLE
        selected frame from the shot's frame tarball (first frames are
        often black/transitional). Falls back to the grid composite."""
        from PIL import Image
        img = None
        try:
            import tarfile
            t = tarfile.open(
                f'{cfg["shot_frames_dir"]}/shot_{sid:04d}_frames.tar.gz')
            names = sorted(m.name for m in t.getmembers()
                           if m.name.endswith(".jpg"))
            if names:
                img = Image.open(t.extractfile(names[len(names) // 2]))
                img.load()
        except Exception:
            img = None
        if img is None:
            try:
                img = Image.open(
                    f'{cfg["shot_images_dir"]}/shot_{sid:06d}.jpg')
            except Exception:
                return None
        w = max(1, round(img.width * THUMB_H / img.height))
        img = img.convert("RGB").resize((w, THUMB_H))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=THUMB_Q)
        return ("data:image/jpeg;base64," +
                base64.b64encode(buf.getvalue()).decode())

    def grid(sid):
        """Full-size frame grid for the selected-shot film strip."""
        try:
            raw = open(f'{cfg["shot_images_dir"]}/shot_{sid:06d}.jpg',
                       "rb").read()
            return ("data:image/jpeg;base64," +
                    base64.b64encode(raw).decode())
        except Exception:
            return None

    shots_out = []
    for sid in sorted(shot_times):
        start, end = shot_times[sid]
        v, div = {}, []
        for f in FIELDS:
            cell, canon = {}, set()
            missing = False
            for k in mkeys:
                doc = runs[k].get(sid)
                if doc is None or f not in doc:
                    cell[k] = "—"
                    missing = True
                    continue
                c, disp = norm(f, doc[f])
                cell[k] = disp
                canon.add(c)
            v[f] = cell
            # a shot absent from one run (honest gap) is not field divergence
            if not missing and len(canon) > 1:
                div.append(f)
        shots_out.append({
            "id": sid, "start": start, "end": end, "ts": hms(start),
            "thumb": thumb(sid), "grid": grid(sid),
            "div": div, "score": len(div), "v": v,
            "desc": {k: (runs[k].get(sid) or {}).get("description", "—")
                     for k in mkeys},
        })

    data = {
        "meta": {
            "title": cfg["title"], "subtitle": cfg["subtitle"],
            "duration_sec": shots_env.get("actual_duration_sec") or
                            max(e for _, e in shot_times.values()),
            "note": cfg.get("note", ""),
        },
        "media": {"video": cfg["video_url"]},
        "vtt": (open(cfg["vtt"]).read() if cfg.get("vtt") else None),
        "models": models_out,
        "baseline": cfg["baseline"],
        "fields": FIELDS,
        "shots": shots_out,
    }
    page = template.replace("/*__DATA__*/",
                            json.dumps(data, ensure_ascii=False))
    open(out_path, "w").write(page)
    n_div = sum(1 for s in shots_out if s["div"])
    print(f"wrote {out_path}: {len(models_out)} models, "
          f"{len(shots_out)} shots, {n_div} divergent, "
          f"{sum(s['score'] for s in shots_out)} field disagreements")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
