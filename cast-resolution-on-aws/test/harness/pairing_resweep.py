#!/usr/bin/env python3
"""Corpus credits re-read with the PAIRING prompt (fi-credits-read v2,
rolled in 2026-09-04) — reuses the P1 sweep's staged frames, batch mode.

For every corpus asset: stream its fi run's records.jsonl, keep the
credits-read records, swap in the CURRENT registry prompt, remap
record_id to `{ep}|{original}`, pool ~12 assets per chunk (~150MB —
inside the measured submit ceiling, above the batch minimum), submit
via bedrock-inference use_batch=true. Fetch assembles each asset's
product through frame-inspection's OWN `_assemble` (the shipping code,
not a scorer-side variant) and rewrites /tmp/gt/in_{ep}/credits.json
(old file kept as credits.pre-pairing.json).

Usage:
  python3 scripts/pairing_resweep.py --submit
  python3 scripts/pairing_resweep.py --fetch    # exit 3 while batch runs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root
sys.path.insert(0, ROOT)
FI = os.path.join(os.path.dirname(ROOT), "frame-inspection-on-aws")
sys.path.insert(0, os.path.join(FI, "states"))
sys.path.insert(0, os.path.join(FI, "lib"))

from lib import pb_common  # noqa: E402

REGION = "eu-west-1"
BUCKET = os.environ.get("PB_PROXY_BUCKET") or \
    open("/tmp/gt/_proxy_bucket").read().strip()
# account-numbered ARNs are LOCAL config, never code (repo rule)
BEDROCK_GW = os.environ.get("PB_BEDROCK_GW") or \
    open("/tmp/gt/_bedrock_gw").read().strip()
SCRATCH = "micro-workflow/bedrock-inference/_staged/pb-pairing-resweep"
STATE = "/tmp/gt/pairing_resweep.json"
TOKEN = "pb-pairing-resweep-20260904"  # nosec B105
ASSETS_PER_CHUNK = 12

_lam = boto3.client("lambda", region_name=REGION)
_s3 = boto3.client("s3", region_name=REGION)


def _registry_system() -> str:
    """The CURRENT fi-credits-read default — the prompt production
    binds, so the resweep measures exactly what ships."""
    body = _s3.get_object(
        Bucket=BUCKET,
        Key="_promptregistry/fi-credits-read/default/artifact.json"
    )["Body"].read()
    art = json.loads(body)
    assert "pairings" in art["system"], "registry still has the v1 prompt"
    return art["system"]


def _sweep():
    return json.load(open("/tmp/gt/pb_sweep_submit.json"))["submits"]


def submit():
    system = _registry_system()
    # (fi_run_wid, ep, asset_idx) for every corpus asset
    jobs = [(s["workflow_id"], ep, p["asset_idx"])
            for s in _sweep() for ep, p in sorted(s["plans"].items())]
    chunks, state = [jobs[i:i + ASSETS_PER_CHUNK]
                     for i in range(0, len(jobs), ASSETS_PER_CHUNK)], {}
    for ci, chunk in enumerate(chunks):
        lines, per_ep = [], {}
        for fi_wid, ep, asset_idx in chunk:
            key = f"micro-workflow/frame-inspection/{fi_wid}/records.jsonl"
            body = _s3.get_object(Bucket=BUCKET, Key=key)["Body"]
            prefix = f"{asset_idx}:credits-read:"
            n = 0
            for raw in body.iter_lines():
                if not raw.strip():
                    continue
                head = raw[:200].decode("utf-8", "replace")
                if f'"record_id": "{prefix}' not in head \
                        and f'"record_id":"{prefix}' not in head:
                    continue
                rec = json.loads(raw)
                rec["system"] = system
                rec["record_id"] = f"{ep}|{rec['record_id']}"
                lines.append(json.dumps(rec, ensure_ascii=False,
                                        separators=(",", ":")))
                n += 1
            per_ep[ep] = {"fi_wid": fi_wid, "asset_idx": asset_idx,
                          "records": n}
            print(f"chunk {ci}: {ep} {n} record(s)")
        out_key = f"{SCRATCH}/chunk{ci}/records.jsonl"
        _s3.put_object(Bucket=BUCKET, Key=out_key,
                       Body=("\n".join(lines) + "\n").encode("utf-8"))
        sub = pb_common.invoke_gateway_tool(
            _lam, BEDROCK_GW, "bedrock-inference", "run_inference", {
                "records_s3uri": f"s3://{BUCKET}/{out_key}",
                "model_chain": ["anthropic.claude-sonnet-4-6"],
                "use_batch": True,
                "idempotency_token": f"{TOKEN}-c{ci}"})
        if sub.get("error"):
            print(f"chunk {ci}: REFUSED — {sub.get('message')}")
            sys.exit(1)
        state[f"chunk{ci}"] = {"job_id": sub["job_id"],
                               "output_s3_prefix": sub["output_s3_prefix"],
                               "eps": per_ep}
        print(f"chunk {ci}: submitted {sub['workflow_id'][:8]} "
              f"({len(lines)} records, batch)")
    json.dump(state, open(STATE, "w"), indent=1)


def _credits_span(fi_wid: str, asset_idx: int) -> dict:
    plan = json.loads(_s3.get_object(
        Bucket=BUCKET,
        Key=f"micro-workflow/frame-inspection/{fi_wid}/_plan/plan.json"
    )["Body"].read())
    for asset in plan["assets"]:
        if asset["asset_idx"] != asset_idx:
            continue
        for req in asset["requests"]:
            if req["category"] == "credits-read":
                return req["spans"][0]
    raise SystemExit(f"no credits-read span for {fi_wid} #{asset_idx}")


def fetch():
    import collect_finalize  # fi's shipping assembler
    state = json.load(open(STATE))
    sfn = boto3.client("stepfunctions", region_name=REGION)
    for cname, c in state.items():
        st = sfn.describe_execution(executionArn=c["job_id"])["status"]
        if st == "RUNNING":
            print(f"{cname}: still RUNNING (batch)")
            sys.exit(3)
        if st != "SUCCEEDED":
            raise SystemExit(f"{cname}: {st}")

    for cname, c in state.items():
        key = c["output_s3_prefix"][5:].split("/", 1)[1] + "/response.jsonl"
        body = _s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        outputs_by_ep: dict = {}
        for ln in body.decode("utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            if row.get("status") != "SUCCESS":
                continue
            ep, orig = row["record_id"].split("|", 1)
            outputs_by_ep.setdefault(ep, {})[orig] = \
                row.get("output_text") or ""
        for ep, meta in c["eps"].items():
            outputs = outputs_by_ep.get(ep) or {}
            parsed, why = collect_finalize._parse_span(
                outputs, meta["asset_idx"], "credits-read", 0)
            if parsed is None:
                print(f"{ep}: GAP — {why}")
                continue
            span = _credits_span(meta["fi_wid"], meta["asset_idx"])
            product = collect_finalize._assemble(
                "credits-read", [(span, 0, parsed)], [])
            dst = f"/tmp/gt/in_{ep}/credits.json"
            if os.path.exists(dst) \
                    and not os.path.exists(dst + ".pre-pairing"):
                shutil.copy(dst, dst + ".pre-pairing")
            json.dump(product, open(dst, "w"))
            n_pairs = sum(len(card.get("pairings") or [])
                          for b in product["blocks"]
                          for card in b["cards"])
            print(f"{ep}: {sum(len(b['cards']) for b in product['blocks'])} "
                  f"cards, {n_pairs} pairing entries")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    submit() if args.submit else fetch()
