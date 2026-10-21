#!/usr/bin/env python3
"""Extract cast-resolution P0 inputs to a local corpus dir.

Downloads the four input files (transcript.json, facemetadata.json,
shots.json, shot_metadata.jsonl) so `p0_bind.py --local DIR` runs without
S3. Two ways to name the runs:

  --config asset.json     the p0_bind config shape (sibling prefixes)
  --ad-run s3://…/audio-description/<wid>
                          read the sibling prefixes from the AD run's
                          _plan/plan.json (the AD corpus episodes)

Missing sources are skipped with a note — absence is a legitimate state
(three-state semantics), extraction mirrors it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

_s3 = boto3.client("s3")

# local filename -> (config key, key suffix under the run prefix)
FILES = {
    "transcript.json": ("speech_diarization",
                        "transcripts/final_transcript.json"),
    "facemetadata.json": ("face_recognition", "facemetadata.json"),
    "shots.json": ("shot_segmentation", "embeddings/shots.json"),
    "shot_metadata.jsonl": ("shot_metadata_enrichment",
                            "shot_metadata.jsonl"),
}
# AD plan.json input key -> p0_bind config key
AD_PLAN_KEYS = {
    "speech_diarization_s3uri": "speech_diarization",
    "face_recognition_s3uri": "face_recognition",
    "shot_segmentation_s3uri": "shot_segmentation",
    "shot_metadata_s3uri": "shot_metadata_enrichment",
}


def _get(uri: str) -> bytes:
    bucket, key = uri[5:].split("/", 1)
    return _s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def prefixes_from_ad_run(ad_prefix: str) -> dict:
    plan = json.loads(_get(f"{ad_prefix.rstrip('/')}/_plan/plan.json"))
    inputs = plan.get("inputs") or {}
    return {ours: inputs[theirs]
            for theirs, ours in AD_PLAN_KEYS.items() if inputs.get(theirs)}


def extract(cfg: dict, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for fname, (ckey, suffix) in FILES.items():
        prefix = cfg.get(ckey)
        if not prefix:
            print(f"  {fname}: source absent — skipped")
            continue
        try:
            data = _get(f"{prefix.rstrip('/')}/{suffix}")
        except Exception as exc:                     # noqa: BLE001 — report
            print(f"  {fname}: fetch failed — {exc}")
            continue
        with open(os.path.join(outdir, fname), "wb") as f:
            f.write(data)
        print(f"  {fname}: {len(data)} bytes")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="json of sibling S3 prefixes")
    g.add_argument("--ad-run", help="audio-description run prefix")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    cfg = (prefixes_from_ad_run(args.ad_run) if args.ad_run
           else json.load(open(args.config)))
    print(f"extracting to {args.outdir}")
    extract(cfg, args.outdir)


if __name__ == "__main__":
    main()
