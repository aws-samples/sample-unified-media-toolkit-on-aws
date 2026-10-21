#!/usr/bin/env python3
"""P1 harness: run the focus-read chain for one asset, locally.

plan_reads() on the local corpus inputs -> ONE pooled inspect_frames
submit to the frame-inspection gateway (driven directly by ARN, the
ad_e2e pattern — no MCP token needed) -> poll -> download the witness
products (credits.json / titles.json) into the corpus dir, where
p0_bind --local picks them up as evidence. Trial-then-wire: the SFN
gains a read state only after this chain is proven on the corpus.

Usage:
    python3 scripts/p1_reads.py --local /tmp/gt/in_<asset> \\
        --config /tmp/gt/pbgate_<asset>_cfg.json [--sync]

--config names the sibling S3 prefixes (shot_segmentation_s3uri and
shot_metadata_s3uri are what the witness needs). --sync passes
use_batch=false for interactive validation; the default leaves the
witness's batch pooling on.
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

from lib import pb_common  # noqa: E402
from lib import reads  # noqa: E402
from lib import sources  # noqa: E402
from lib import transcript_cues  # noqa: E402

REGION = "eu-west-1"
FI_GW_FILE = "/tmp/gt/_fi_gw"
_lam = boto3.client("lambda", region_name=REGION)


def tool(fn_arn, name, payload):
    return pb_common.invoke_gateway_tool(_lam, fn_arn, "frame-inspection",
                                         name, payload)


def s3_json(uri):
    s3 = boto3.client("s3", region_name=REGION)
    bucket, key = uri[5:].split("/", 1)
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    ep = os.path.basename(args.local.rstrip("/")).removeprefix("in_")

    def j(name):
        p = os.path.join(args.local, name)
        return json.load(open(p)) if os.path.exists(p) else None

    tx = j("transcript.json")
    transcript = sources.parse_transcript(tx) if tx else None
    fm = j("facemetadata.json")
    faces = sources.parse_faces(fm) if fm else None
    sh = j("shots.json")
    shots = sources.parse_shots(sh) if sh else {}
    docs = []
    mdp = os.path.join(args.local, "shot_metadata.jsonl")
    if os.path.exists(mdp):
        docs = [json.loads(ln) for ln in open(mdp) if ln.strip()]
    text_shots = sources.parse_text_signals(docs, shots)
    duration = max((b for _, b in shots.values()), default=0.0)
    cues = transcript_cues.extract_cues(transcript["turns"]) \
        if transcript else []

    plan = reads.plan_reads(cues, faces, text_shots, duration)
    print(f"{ep}: duration={duration:.0f}s text_shots={len(text_shots)} "
          f"-> credits={plan['credits']} spans={len(plan['spans'])} "
          f"skipped={plan['skipped']}")
    if not plan["requests"]:
        print("nothing to read")
        return

    gw = open(FI_GW_FILE).read().strip()
    req = {"shot_segmentation_s3uri": cfg["shot_segmentation"]
           if "shot_segmentation" in cfg
           else cfg["shot_segmentation_s3uri"]}
    md = cfg.get("shot_metadata_enrichment") or cfg.get("shot_metadata_s3uri")
    if md:
        req["shot_metadata_s3uri"] = md
    # one pooled submit per run — the witness stacks frames per record and
    # pools records per batch job (README.md §10 cost posture)
    req["assets"] = [{"shot_segmentation_s3uri":
                      req.pop("shot_segmentation_s3uri"),
                      **({"shot_metadata_s3uri": req.pop(
                          "shot_metadata_s3uri")} if md else {}),
                      "requests": plan["requests"]}]
    if args.sync:
        req["use_batch"] = False
    req["idempotency_token"] = args.token or f"pb-p1-{ep}"

    sub = tool(gw, "inspect_frames", req)
    print("submit:", json.dumps({k: sub.get(k) for k in
          ("status", "workflow_id", "errorCode", "message")}))
    if sub.get("error"):
        sys.exit(1)
    while True:
        time.sleep(20)
        st = tool(gw, "get_job_status", {"job_id": sub["job_id"]})
        print(" ", st.get("status"), st.get("current_state", ""))
        if st.get("status") in ("SUCCEEDED", "FAILED", "TIMED_OUT",
                                "ABORTED"):
            break
    if st["status"] != "SUCCEEDED":
        print("error:", str(st.get("error"))[:400])
        sys.exit(1)

    got = []
    for o in st.get("outputs") or []:
        base = o["uri"].rsplit("/", 1)[-1]
        if base in ("credits.json", "titles.json"):
            doc = s3_json(o["uri"])
            json.dump(doc, open(os.path.join(args.local, base), "w"))
            got.append(base)
    print(f"fetched {got} into {args.local} — rerun p0_bind --local to "
          f"fold the evidence in")


if __name__ == "__main__":
    main()
