#!/usr/bin/env python3
"""P1 corpus sweep: ONE pooled inspect_frames submit for every local
corpus asset — the batch-economics posture in practice (fi defaults
use_batch=TRUE and pools assets[] into one Bedrock batch job).

Per asset: shot-seg prefix recovered from the local shots.json `uuid`;
read plan from local evidence; enrichment prefix attached when a cfg/
state file records it (enables fi's credits auto-trigger), else the
credits read rides an explicit tail span (credits_hint=True — the corpus
IS episodic drama; caller knowledge is a legitimate trigger).

Usage:
    python3 scripts/p1_sweep.py [--submit | --fetch] [--sync]

--submit builds the pooled request and submits (idempotent token).
--fetch polls once and, on SUCCEEDED, downloads each asset's products
into its /tmp/gt/in_* dir.
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
PROXY = os.environ.get("PB_PROXY_BUCKET") or \
    open("/tmp/gt/_proxy_bucket").read().strip()
P = f"s3://{PROXY}/micro-workflow"
FI_GW_FILE = "/tmp/gt/_fi_gw"
STATE = "/tmp/gt/pb_sweep_submit.json"
RUN_ID = "pb-sweep-20260904"
# local corpus assets known to carry end credits (asset names identify
# titles, so the list is LOCAL config, never code) — loaded lazily so a
# purged /tmp fails with a named condition, not an import crash
def hint_prefixes():
    path = "/tmp/gt/credits_hint_prefixes.json"
    if not os.path.exists(path):
        raise SystemExit(f"local corpus config missing: {path}")
    return tuple(json.load(open(path)))

_lam = boto3.client("lambda", region_name=REGION)
_s3 = boto3.client("s3", region_name=REGION)


def tool(name, payload):
    gw = open(FI_GW_FILE).read().strip()
    return pb_common.invoke_gateway_tool(_lam, gw, "frame-inspection",
                                         name, payload)


def known_enrichment(ep):
    """shot_metadata prefix when some record names it."""
    for path in (f"/tmp/gt/pbgate_{ep}_cfg.json",
                 f"/tmp/gt/pb_p1_{ep}_cfg.json"):
        if os.path.exists(path):
            cfg = json.load(open(path))
            uri = cfg.get("shot_metadata_enrichment") or \
                cfg.get("shot_metadata_s3uri")
            if uri:
                return uri.rstrip("/")
    return None


def build_assets():
    assets, plans = [], {}
    for d in sorted(os.listdir("/tmp/gt")):
        if not d.startswith("in_"):
            continue
        ep = d[3:]
        base = f"/tmp/gt/{d}"

        def j(name):
            p = os.path.join(base, name)
            return json.load(open(p)) if os.path.exists(p) else None
        sh_doc = j("shots.json")
        if not sh_doc or not sh_doc.get("uuid"):
            print(f"  {ep}: no shots.json uuid — skipped")
            continue
        if os.path.exists(f"{base}/credits.json"):
            print(f"  {ep}: witness products already fetched — skipped")
            continue
        shots = sources.parse_shots(sh_doc)
        duration = max((b for _, b in shots.values()), default=0.0)
        tx = j("transcript.json")
        transcript = sources.parse_transcript(tx) if tx else None
        fm = j("facemetadata.json")
        faces = sources.parse_faces(fm) if fm else None
        docs = []
        mdp = f"{base}/shot_metadata.jsonl"
        if os.path.exists(mdp):
            docs = [json.loads(ln) for ln in open(mdp) if ln.strip()]
        text_shots = sources.parse_text_signals(docs, shots)
        cues = []
        if transcript:
            cues = transcript_cues.extract_cues(
                transcript["turns"], transcript.get("language") or "en")

        md_prefix = known_enrichment(ep)
        plan = reads.plan_reads(
            cues, faces, text_shots, duration,
            credits_hint=any(ep.startswith(h) for h in hint_prefixes()),
            has_shot_metadata=bool(md_prefix))
        if not plan["requests"]:
            print(f"  {ep}: nothing to read")
            continue
        entry = {"shot_segmentation_s3uri":
                 f"{P}/shot-segmentation/{sh_doc['uuid']}",
                 "requests": plan["requests"]}
        if md_prefix:
            entry["shot_metadata_s3uri"] = md_prefix
        assets.append(entry)
        plans[ep] = {"asset_idx": len(assets) - 1,
                     "spans": len(plan["spans"]),
                     "credits": plan["credits"]}
        print(f"  {ep}: idx={plans[ep]['asset_idx']} "
              f"spans={plans[ep]['spans']} credits={plan['credits']} "
              f"md={'yes' if md_prefix else 'no'}")
    return assets, plans


# One giant pool OOM'd bedrock-inference's batch submitter (measured
# 2026-09-04: 34 assets -> 1531 records -> 668 MB request.jsonl ->
# Runtime.OutOfMemory before the batch job launched). Chunked pools keep
# each staged file ~150 MB while every chunk still clears the Bedrock
# batch minimum.
ASSETS_PER_SUBMIT = 8


def submit(sync=False):
    assets, plans = build_assets()
    if not assets:
        print("nothing to submit")
        return
    subs = []
    for ci in range(0, len(assets), ASSETS_PER_SUBMIT):
        chunk = assets[ci:ci + ASSETS_PER_SUBMIT]
        req = {"assets": chunk,
               "idempotency_token": f"{RUN_ID}-c{ci // ASSETS_PER_SUBMIT}"}
        if sync:
            req["use_batch"] = False
        sub = tool("inspect_frames", req)
        print(f"chunk {ci // ASSETS_PER_SUBMIT}: " + json.dumps(
            {k: sub.get(k) for k in ("status", "workflow_id",
                                     "errorCode", "message")}))
        if sub.get("error"):
            sys.exit(1)
        chunk_plans = {ep: {**info,
                            "asset_idx": info["asset_idx"] - ci}
                       for ep, info in plans.items()
                       if ci <= info["asset_idx"] < ci + len(chunk)}
        subs.append({"job_id": sub["job_id"],
                     "workflow_id": sub["workflow_id"],
                     "output_s3_prefix": sub["output_s3_prefix"],
                     "plans": chunk_plans})
    json.dump({"submits": subs}, open(STATE, "w"), indent=1)


def fetch():
    st = json.load(open(STATE))
    all_done = True
    for sub in st["submits"]:
        if sub.get("done"):
            continue
        job = tool("get_job_status", {"job_id": sub["job_id"]})
        status = job.get("status")
        print(f"{sub['workflow_id'][:8]}: {status} "
              f"{job.get('current_state', '')}")
        if status in ("FAILED", "TIMED_OUT", "ABORTED"):
            sub["done"] = "failed"
            print("  error:", str(job.get("error"))[:300])
            continue
        if status != "SUCCEEDED":
            all_done = False
            continue
        prefix = sub["output_s3_prefix"][5:]
        bucket, keyprefix = prefix.split("/", 1)
        for ep, info in sub["plans"].items():
            got = []
            for basename in ("credits.json", "titles.json"):
                key = f"{keyprefix}/{info['asset_idx']}/{basename}"
                try:
                    body = _s3.get_object(Bucket=bucket,
                                          Key=key)["Body"].read()
                except _s3.exceptions.NoSuchKey:
                    continue
                open(f"/tmp/gt/in_{ep}/{basename}", "wb").write(body)
                got.append(basename)
            print(f"  {ep}: {got}")
        sub["done"] = "fetched"
    json.dump(st, open(STATE, "w"), indent=1)
    return all_done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    if args.submit:
        submit(sync=args.sync)
    elif args.fetch:
        sys.exit(0 if fetch() else 3)   # 3 = still waiting (poller loops)
    else:
        ap.error("--submit or --fetch")
