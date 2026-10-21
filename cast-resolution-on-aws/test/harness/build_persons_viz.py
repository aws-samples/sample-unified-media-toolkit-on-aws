#!/usr/bin/env python3
"""Build the cast-resolution review page (house metadata-viz look).

One self-contained HTML from file://, everything inlined, no video (P0
runs on local corpus inputs; a media pane can come later with presigned
proxies). Deterministic: same inputs, same page.

Usage:
    python3 scripts/build_persons_viz.py --local /tmp/gt/in_<asset> \\
        --persons out/persons.json -o persons.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from lib import sources  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LANES = 30          # context budget for the canvas; table has everyone


def _presign(uri: str, expiry: int) -> str:
    import boto3
    from botocore.config import Config
    s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
    bucket, key = uri[5:].split("/", 1)
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry)


def _vtt_ts(t: float) -> str:
    ms = int(t * 1000)
    return (f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:"
            f"{ms % 60000 // 1000:02d}.{ms % 1000:03d}")


def _build_vtt(transcript, persons) -> str:
    """Subtitles voiced with the RESOLVED person names — binding quality
    is readable straight off the picture. Unresolved speakers keep their
    SPEAKER_ID so gaps are visible too, not hidden."""
    label = {}
    for p in persons:
        for sid in p["speaker_ids"]:
            label[sid] = p["name"] or sid
    out = ["WEBVTT", ""]
    for i, t in enumerate(transcript["turns"]):
        who = label.get(t["speaker_id"], t["speaker_id"])
        out += [str(i), f"{_vtt_ts(t['start'])} --> {_vtt_ts(t['end'])}",
                f"<v {who}>{t['text']}", ""]
    return "\n".join(out)


def _merge_spans(spans, gap=1.0):
    """Merge adjacent/overlapping spans so the canvas draws O(scenes),
    not O(sentences)."""
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 2), round(b, 2)] for a, b in out]


def _evidence_rows(p, doc):
    """Flatten a person's evidence into display rows, most telling first."""
    rows = []
    for n in p.get("name_mechanisms") or []:
        rows.append({"layer": "name", "mechanism": n,
                     "text": f"named “{p['name']}”" if p.get("name")
                             else "name evidence (conflicted)", "t": None})
    for e in p.get("evidence") or []:
        mech = e.get("mechanism") or e.get("kind") or "?"
        if mech == "cooccurrence":
            rows.append({"layer": "face", "mechanism": "cooccurrence",
                         "text": f"speaker↔face share {e.get('share')}, "
                                 f"{e.get('turns_together')} turns, "
                                 f"lead {e.get('lead')}", "t": None})
        elif e.get("kind") in ("self_id", "sign_off", "introduction"):
            rows.append({"layer": "name", "mechanism": e["kind"],
                         "text": f"“{e.get('name')}”", "t": e.get("t")})
        elif e.get("kind") == "vocative_resolved":
            rows.append({"layer": "name", "mechanism": "vocative",
                         "text": f"addressed as “{e.get('name')}” "
                                 f"×{e.get('votes')} by "
                                 f"{', '.join(e.get('addressers') or [])}",
                         "t": None})
        elif e.get("kind") == "overlay_name":
            rows.append({"layer": "name", "mechanism": "overlay",
                         "text": f"on-screen text “{e.get('name')}” "
                                 f"(shot {e.get('shot_id')})",
                         "t": e.get("t0")})
        elif mech == "name_agreement":
            rows.append({"layer": "name", "mechanism": "name agreement",
                         "text": f"independent mechanisms agree on "
                                 f"“{e.get('name')}” — merged", "t": None})
        elif mech == "critic_veto":
            what = {"speech_self_overlap":
                        f"overlapping speech ({e.get('overlap_sec')}s) — "
                        f"a person cannot talk over themselves",
                    "face_self_overlap":
                        f"faces co-present {e.get('overlap_sec')}s",
                    "name_collision":
                        f"“{e.get('name')}” claimed by persons "
                        f"{e.get('person_ids')}"}.get(e.get("check"),
                                                      e.get("check"))
            rows.append({"layer": "name", "mechanism": "critic VETO",
                         "text": f"{what} — demoted, never guessed",
                         "t": None})
    return rows


def build(local_dir, persons_path, video_uri=None, expiry=43200):
    doc = json.load(open(persons_path))

    def j(name):
        p = os.path.join(local_dir, name)
        return json.load(open(p)) if os.path.exists(p) else None
    tx = j("transcript.json")
    fm = j("facemetadata.json")
    transcript = sources.parse_transcript(tx) if tx else None
    faces = {f["face_person_id"]: f
             for f in (sources.parse_faces(fm) if fm else [])}

    turns_by_speaker: dict[str, list] = {}
    dur = 0.0
    if transcript:
        for t in transcript["turns"]:
            turns_by_speaker.setdefault(t["speaker_id"], []).append(
                (t["start"], t["end"]))
            dur = max(dur, t["end"])
    for f in faces.values():
        for a, b in f["spans"]:
            dur = max(dur, b)

    cues_by_speaker: dict[str, list] = {}
    for c in doc.get("cues") or []:
        if c.get("t") is not None and c["kind"] in (
                "self_id", "sign_off", "introduction"):
            cues_by_speaker.setdefault(c["speaker_id"], []).append(
                {"t": round(c["t"], 2), "name": c["name"]})

    persons = []
    for p in doc["persons"]:
        speech, face, names = [], [], []
        for sid in p["speaker_ids"]:
            speech += turns_by_speaker.get(sid, [])
            names += cues_by_speaker.get(sid, [])
        for fid in p["face_ids"]:
            face += faces.get(fid, {}).get("spans", [])
        for e in p.get("evidence") or []:
            if e.get("kind") == "overlay_name" and e.get("t0") is not None:
                names.append({"t": round(e["t0"], 2), "name": e["name"]})
        persons.append({**p,
                        "spans": {"speech": _merge_spans(speech),
                                  "face": _merge_spans(face),
                                  "names": names},
                        "evidence_rows": _evidence_rows(p, doc)})

    # lanes: named first, then by activity; canvas budget MAX_LANES
    ranked = sorted(persons,
                    key=lambda p: (bool(p["name"]),
                                   p["speech_sec"] + p["screen_sec"]),
                    reverse=True)
    lanes = [p["person_id"] for p in ranked[:MAX_LANES]]

    src = doc.get("sources") or {}
    present = [k for k in ("speech_diarization", "face_recognition",
                           "shot_segmentation", "shot_metadata_enrichment")
               if src.get(k)]
    named = sum(1 for p in persons if p["name"])
    media = {"video": _presign(video_uri, expiry)} if video_uri else None
    vtt = _build_vtt(transcript, persons) if (transcript and media) else None
    return {
        "media": media,
        "vtt": vtt,
        "vetoes": doc.get("vetoes") or [],
        "meta": {
            "title": os.path.basename(local_dir.rstrip("/")).removeprefix("in_"),
            "subtitle": f"{len(persons)} persons · {named} named · "
                        f"P0 free evidence only (no OCR reads, no LLM)",
            "duration_sec": round(dur, 2),
            "sources": present,
            "note": ("Evidence, never a verdict: a person is named only by "
                     "an independent naming mechanism; co-occurrence alone "
                     "proposes, never names. Absent sources make their "
                     "fields absent, not empty. Unnamed residue is "
                     "first-class output — a wrong name is worse than "
                     "none."),
        },
        "persons": persons,
        "lanes": lanes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, help="corpus input dir")
    ap.add_argument("--persons", required=True, help="persons.json path")
    ap.add_argument("--video", help="s3:// source/proxy to presign")
    ap.add_argument("--expiry", type=int, default=43200)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    data = build(args.local, args.persons, args.video, args.expiry)
    tpl = open(os.path.join(HERE, "..", "..", "viz",
                            "persons_template.html")).read()
    out = tpl.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    open(args.out, "w").write(out)
    print(f"wrote {args.out} "
          f"({os.path.getsize(args.out) // 1024} KB, "
          f"{len(data['persons'])} persons, {len(data['lanes'])} lanes)")


if __name__ == "__main__":
    main()
