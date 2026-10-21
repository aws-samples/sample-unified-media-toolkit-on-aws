#!/usr/bin/env python3
"""P2 cast-adjudication harness: adjudicate one local asset, sync.

Binds the asset fresh (THE pipeline), builds adjudication dossiers
against its credits cast universe, submits ONE bedrock-inference run,
applies the firewalled verdicts, re-runs the critic (demote-only — a
bad assignment gets demoted, not shipped), writes persons.json.

Usage:
    python3 scripts/adjudicate_run.py --local /tmp/gt/in_<asset> -o out
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

from lib import adjudicate  # noqa: E402
from lib import bind_pipeline  # noqa: E402
from lib import critic  # noqa: E402
from lib import pb_common  # noqa: E402
from lib import reads  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    ep = os.path.basename(args.local.rstrip("/")).removeprefix("in_")

    (transcript, faces, _shots, overlay, credits, src, speaker_f0,
     screen_texts) = bind_pipeline.load_local_asset(args.local)
    doc = bind_pipeline.bind(transcript, faces, overlay, src, credits,
                             speaker_f0=speaker_f0,
                             screen_texts=screen_texts)
    cast = reads.cast_from_credits(credits) if credits else {}
    dossiers = adjudicate.build_dossiers(
        doc, cast, turns=(transcript or {}).get("turns"))
    print(f"{ep}: {len(dossiers)} adjudicable persons, "
          f"{len(cast.get('pairs') or [])} pairs in the universe")
    if not dossiers:
        return

    records = adjudicate.build_records(dossiers, cast, harness_prompts.system("cast-adjudicate"))
    sub = tool("run_inference", {
        "records": records,
        "use_batch": False,
        "idempotency_token": f"pb-adjudicate-{ep}-20260905"})
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
    verdicts = adjudicate.parse_verdicts(texts, cast, dossiers)
    doc = adjudicate.apply_verdicts(doc, verdicts)
    doc = critic.critique(doc)            # demote-only, runs AFTER apply

    os.makedirs(args.outdir, exist_ok=True)
    json.dump(doc, open(os.path.join(args.outdir, "persons.json"), "w"),
              indent=1)
    print(f"\n{len(verdicts)} assignment(s) survived the firewall:")
    by_id = {p["person_id"]: p for p in doc["persons"]}
    for v in verdicts:
        p = by_id.get(v["person_id"], {})
        print(f"  [{v['person_id']}] name={p.get('name')} "
              f"character={p.get('character_name')} "
              f"via={v['evidence_ids']} — {v['reason'][:90]}")


if __name__ == "__main__":
    main()
