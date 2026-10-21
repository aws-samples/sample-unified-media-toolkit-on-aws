#!/usr/bin/env python3
"""P0 harness: bind persons from EXISTING sibling runs, locally.

No new reads, no LLM — free evidence only (README.md §13 P0): transcript cues +
face-source names + enrichment overlay names + co-occurrence. Measures how
far free evidence gets us per genre before any OCR is built.

Usage:
    python3 scripts/p0_bind.py --config asset.json [-o outdir]

asset.json (any prefix may be omitted — at least one identity source):
    {"title": "...",
     "speech_diarization": "s3://.../micro-workflow/speech-diarization/<wid>",
     "face_recognition":   "s3://.../micro-workflow/face-recognition/<wid>",
     "shot_segmentation":  "s3://.../micro-workflow/shot-segmentation/<wid>",
     "shot_metadata_enrichment": "s3://.../shot-metadata-enrichment/<wid>"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from lib import bind_pipeline  # noqa: E402
from lib import persons  # noqa: E402
from lib import sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="json of sibling S3 prefixes")
    g.add_argument("--local", help="dir with transcript.json etc.")
    ap.add_argument("--title", default=None)
    ap.add_argument("-o", "--outdir", default=".")
    args = ap.parse_args()

    if args.local:
        (transcript, faces, _shots, overlay, credits, src, speaker_f0,
         screen_texts) = bind_pipeline.load_local_asset(args.local)
        cfg = {"title": args.title or os.path.basename(
            args.local.rstrip("/"))}
    else:
        cfg = json.load(open(args.config))
        src = {k: cfg.get(k) for k in
               ("speech_diarization", "face_recognition",
                "shot_segmentation", "shot_metadata_enrichment")}
        transcript = sources.load_transcript(cfg.get("speech_diarization"))
        speaker_f0 = sources.load_speaker_f0(cfg.get("speech_diarization"))
        faces = sources.load_faces(cfg.get("face_recognition"))
        shots = sources.load_shots(cfg.get("shot_segmentation"))
        # mirror the state's wiring (states/bind_persons.py): one fetch
        # of shot_metadata.jsonl feeds overlay names AND the
        # depicted-face guard — a harness that skips the marking pass
        # measures a pipeline that never runs in production
        md_docs = sources.load_shot_metadata_docs(
            cfg.get("shot_metadata_enrichment"))
        overlay = (sources.parse_overlay_names(md_docs, shots)
                   if md_docs and shots is not None else None)
        screen_texts = (sources.parse_screen_texts(md_docs, shots)
                        if md_docs and shots is not None else None)
        sources.mark_depicted_faces(
            faces or [],
            sources.parse_media_depiction_spans(md_docs or [], shots or {}))
        credits = None

    if transcript is None and faces is None:
        raise SystemExit("at least one identity source required "
                         "(speech_diarization or face_recognition)")

    doc = bind_pipeline.bind(transcript, faces, overlay, src, credits,
                             speaker_f0=speaker_f0,
                             screen_texts=screen_texts)
    smap, fmap = persons.remap_tables(doc)

    os.makedirs(args.outdir, exist_ok=True)
    for name, d in (("persons.json", doc), ("speaker_map.json", smap),
                    ("face_map.json", fmap)):
        if name == "face_map.json" and faces is None:
            continue                      # absent source -> absent output
        if name == "speaker_map.json" and transcript is None:
            continue
        path = os.path.join(args.outdir, name)
        json.dump(d, open(path, "w"), indent=1)
        print("wrote", path)

    named = [p for p in doc["persons"] if p["name"]]
    print(f"\n{cfg.get('title', '?')}: {len(doc['persons'])} persons, "
          f"{len(named)} named, "
          f"{len(doc['unresolved']['speakers'])} unresolved speakers, "
          f"{len(doc['unresolved']['faces'])} unresolved faces")
    for p in doc["persons"]:
        print(f"  [{p['person_id']}] {p['type']:18s} "
              f"name={p['name'] or '—':24s} "
              f"speakers={p['speaker_ids']} faces={p['face_ids']} "
              f"via={','.join(p['name_mechanisms']) or '-'}")


if __name__ == "__main__":
    main()
