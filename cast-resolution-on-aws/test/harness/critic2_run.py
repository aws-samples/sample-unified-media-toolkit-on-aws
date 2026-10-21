#!/usr/bin/env python3
"""Tooth-2 harness: run the report-only LLM critique for one local asset.

Binds the asset fresh, assembles dossiers, submits ONE bedrock-inference
run (inline records, sync), parses flags, writes persons.json with
critic_review attached. Report-only: nothing is renamed or demoted.

Usage:
    python3 scripts/critic2_run.py --local /tmp/gt/in_tdy -o outdir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root: test/harness/ is two levels below it
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from lib import critic2  # noqa: E402
from lib import pb_common  # noqa: E402

import harness_prompts

REGION = "eu-west-1"
# account-numbered ARNs are LOCAL config, never code (repo rule)
BEDROCK_GW = os.environ.get("PB_BEDROCK_GW") or \
    open("/tmp/gt/_bedrock_gw").read().strip()
_lam = boto3.client("lambda", region_name=REGION)
_s3 = boto3.client("s3", region_name=REGION)


def tool(name, payload):
    return pb_common.invoke_gateway_tool(_lam, BEDROCK_GW,
                                         "bedrock-inference", name, payload)


def bind(local_dir):
    from lib import bind_pipeline
    (transcript, faces, _shots, overlay, credits, src, speaker_f0,
     screen_texts) = bind_pipeline.load_local_asset(local_dir)
    return bind_pipeline.bind(transcript, faces, overlay, src, credits,
                              speaker_f0=speaker_f0,
                              screen_texts=screen_texts), transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    ep = os.path.basename(args.local.rstrip("/")).removeprefix("in_")

    doc, transcript = bind(args.local)
    dossiers = critic2.build_dossiers(
        doc, turns=(transcript or {}).get("turns"))
    print(f"{ep}: {len(dossiers)} reviewable persons")
    if not dossiers:
        return
    records = critic2.build_records(dossiers, harness_prompts.system("cast-critic2"))
    sub = tool("run_inference", {
        "records": records,
        "use_batch": False,
        "idempotency_token": f"pb-critic2v2-{ep}-20260904"})
    print("submit:", json.dumps({k: sub.get(k) for k in
          ("status", "workflow_id", "errorCode", "message")}))
    if sub.get("error"):
        sys.exit(1)
    while True:
        time.sleep(15)
        st = tool("get_job_status", {"job_id": sub["job_id"]})
        print(" ", st.get("status"), st.get("current_state", ""))
        if st.get("status") in ("SUCCEEDED", "FAILED", "TIMED_OUT",
                                "ABORTED"):
            break
    if st["status"] != "SUCCEEDED":
        print("error:", str(st.get("error"))[:300])
        sys.exit(1)

    prefix = st["output_s3_prefix"][5:]
    bucket, key = prefix.split("/", 1)
    body = _s3.get_object(Bucket=bucket,
                          Key=f"{key}/response.jsonl")["Body"].read()
    texts = [json.loads(ln).get("output_text", "")
             for ln in body.decode().splitlines() if ln.strip()]
    flags = critic2.parse_reviews(texts)
    doc = critic2.attach_flags(doc, flags)
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(doc, open(os.path.join(args.outdir, "persons.json"), "w"),
              indent=1)
    print(f"\n{len(flags)} suspect flags (report-only):")
    by_id = {p["person_id"]: p for p in doc["persons"]}
    for f in flags:
        p = by_id.get(f["person_id"], {})
        print(f"  [{f['person_id']}] {p.get('name') or p.get('aliases')} "
              f"({f.get('suspect_field')}): {f['reason'][:110]}")


if __name__ == "__main__":
    main()
