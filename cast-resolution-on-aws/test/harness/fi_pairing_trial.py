#!/usr/bin/env python3
"""fi-credits-read structured-pairing PROMPT TRIAL (gate/drafts/
fi_credits_pairing.md).

Reuses the EXACT frames the P1 sweep already paid to stack: streams each
fi run's staged records.jsonl, keeps the credits-read records of the
eval assets, swaps in the trial system prompt (adds `pairings[]` to the
per-frame line), stages the result, and submits ONE sync
bedrock-inference run per asset. No frame-inspection code or registry
default is touched — the trial rides beside production.

Eval (the draft's gate): pairing precision against the text-side
extractors' agreement + hand-check, across every measured layout class:
two-column cast (ALLCAPS actor | TitleCase character), 'ACTOR as
CHARACTER' lines, crew cards (role above/beside name), brief reality
credits, and one non-English asset (the multilingual path the text
side abstains on). The asset map (local corpus name -> fi sweep run +
asset_idx) is LOCAL config, not code:
/tmp/gt/pairing_trial_assets.json

Usage:
  python3 scripts/fi_pairing_trial.py --submit   # stage + submit all
  python3 scripts/fi_pairing_trial.py --fetch    # collect + score
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

REGION = "eu-west-1"
BUCKET = os.environ.get("PB_PROXY_BUCKET") or \
    open("/tmp/gt/_proxy_bucket").read().strip()
# account-numbered ARNs are LOCAL config, never code (repo rule)
BEDROCK_GW = os.environ.get("PB_BEDROCK_GW") or \
    open("/tmp/gt/_bedrock_gw").read().strip()
SCRATCH = "micro-workflow/bedrock-inference/_staged/pb-pairing-trial"
STATE = "/tmp/gt/pairing_trial.json"
TOKEN = "pb-pairing-trial-20260904"  # nosec B105

# eval asset -> [fi sweep run workflow_id prefix, asset_idx]; the map
# names local corpus assets, so it lives in local config, never code
ASSETS = {k: tuple(v) for k, v in json.load(
    open("/tmp/gt/pairing_trial_assets.json")).items()}

_lam = boto3.client("lambda", region_name=REGION)
_s3 = boto3.client("s3", region_name=REGION)

# The trial prompt: the production fi-credits-read system verbatim, plus
# the pairings extension (draft rules: VISUAL pairing only, verbatim
# text, NEUTRAL — never 'character'; classification is the caller's).
TRIAL_SYSTEM = """You read the text on television credit and title cards, verbatim.

You receive consecutive DISTINCT frames from one credits/title block of a
programme, each preceded by a label naming its time (`Frame at t=...`).
Consecutive frames are different cards, or successive views of a scrolling
roll; overlapping content between adjacent scroll frames is normal.

Rules:

- Report text EXACTLY as displayed: spelling, capitalization, diacritics,
  punctuation. Names are the payload — never normalize or correct one.
- One JSON line per frame, in order:
  `{"t": <the frame's labeled time>, "lines": ["...", "..."],
  "pairings": [...]}` — `lines` has one entry per visual text line, top
  to bottom, left column before right where a card pairs roles with
  names.
- `pairings` reports what the card's LAYOUT pairs together, one entry
  per visually paired name:
  `{"name": "...", "paired_with": "...", "header": "..."}`
  * Emit a pairing ONLY when the card visually pairs a person's name
    with other text — side-by-side columns, an 'as'-line, a stacked
    name-under-role pair, a connector glyph. NEVER infer a pairing from
    mere adjacency in a list.
  * `name` is the person's name; `paired_with` is the text it is paired
    to (verbatim), or null when the name sits under a heading with no
    per-name pairing; `header` is the section heading governing the
    name (verbatim), or null when there is none.
  * Copy text verbatim into all three fields. Do NOT interpret roles,
    do NOT translate, do NOT decide whether `paired_with` is a
    character, a job, or anything else.
  * A frame whose cards pair nothing gets `"pairings": []`.
- A frame with no readable text gets `"lines": []` and `"pairings": []`.
- A line you cannot read with confidence is the literal string
  `"[UNREADABLE]"` — never a guess, never a plausible reconstruction. An
  invented name is worse than a gap.
- For scrolling rolls, report every line legible in THAT frame; do not
  de-duplicate against other frames — the consumer joins overlaps.

Output: raw JSONL only — one line per frame, no commentary, no code
fences. Escape inner double quotes as \\"."""


def _sweep_runs():
    d = json.load(open("/tmp/gt/pb_sweep_submit.json"))
    return {s["workflow_id"][:8]: s["workflow_id"] for s in d["submits"]}


def _stage(ep: str, fi_wid: str, asset_idx: int) -> str | None:
    """Filter this asset's credits-read records out of the fi run's
    staged records.jsonl (streamed — the file is ~180MB), swap the
    system prompt, stage the trial records. -> records_s3uri | None."""
    key = f"micro-workflow/frame-inspection/{fi_wid}/records.jsonl"
    body = _s3.get_object(Bucket=BUCKET, Key=key)["Body"]
    out_key = f"{SCRATCH}/{ep}/records.jsonl"
    lines, count = [], 0
    prefix = f"{asset_idx}:credits-read:"
    for raw in body.iter_lines():
        if not raw.strip():
            continue
        head = raw[:200].decode("utf-8", "replace")
        if f'"record_id": "{prefix}' not in head \
                and f'"record_id":"{prefix}' not in head:
            continue
        rec = json.loads(raw)
        rec["system"] = TRIAL_SYSTEM
        lines.append(json.dumps(rec, ensure_ascii=False,
                                separators=(",", ":")))
        count += 1
    if not count:
        print(f"{ep}: no credits-read records in {fi_wid[:8]} — skipped")
        return None
    _s3.put_object(Bucket=BUCKET, Key=out_key,
                   Body=("\n".join(lines) + "\n").encode("utf-8"))
    print(f"{ep}: staged {count} record(s)")
    return f"s3://{BUCKET}/{out_key}"


def submit():
    runs = _sweep_runs()
    state = {}
    for ep, (short, asset_idx) in ASSETS.items():
        uri = _stage(ep, runs[short], asset_idx)
        if not uri:
            continue
        sub = pb_common.invoke_gateway_tool(
            _lam, BEDROCK_GW, "bedrock-inference", "run_inference", {
                "records_s3uri": uri,
                "model_chain": ["anthropic.claude-sonnet-4-6"],
                "use_batch": False,
                "idempotency_token": f"{TOKEN}-{ep}"})
        if sub.get("error"):
            print(f"{ep}: REFUSED — {sub.get('message')}")
            continue
        state[ep] = {"job_id": sub["job_id"],
                     "output_s3_prefix": sub["output_s3_prefix"]}
        print(f"{ep}: submitted {sub['workflow_id'][:8]}")
    json.dump(state, open(STATE, "w"), indent=1)


def _pairings(prefix: str) -> tuple[list, int]:
    key = prefix[5:].split("/", 1)[1] + "/response.jsonl"
    body = _s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    pairs, frames = [], 0
    for ln in body.decode("utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        if row.get("status") != "SUCCESS":
            continue
        for out in (row.get("output_text") or "").splitlines():
            out = out.strip().strip("`")
            if not out.startswith("{"):
                continue
            try:
                frame = json.loads(out)
            except json.JSONDecodeError:
                continue
            frames += 1
            for p in frame.get("pairings") or []:
                if isinstance(p, dict) and p.get("name"):
                    pairs.append({"name": p["name"],
                                  "paired_with": p.get("paired_with"),
                                  "header": p.get("header"),
                                  "t": frame.get("t")})
    return pairs, frames


def fetch():
    state = json.load(open(STATE))
    sfn = boto3.client("stepfunctions", region_name=REGION)
    for ep, s in state.items():
        st = sfn.describe_execution(executionArn=s["job_id"])["status"]
        if st == "RUNNING":
            print(f"{ep}: still RUNNING")
            sys.exit(3)
        if st != "SUCCEEDED":
            print(f"{ep}: {st}")
            continue
        pairs, frames = _pairings(s["output_s3_prefix"])
        # de-dup verbatim repeats across scroll frames
        seen, uniq = set(), []
        for p in pairs:
            k = (p["name"], p["paired_with"], p["header"])
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        # the text-side extractor on the SAME asset, for agreement
        text_pairs = []
        cpath = f"/tmp/gt/in_{ep}/credits.json"
        if os.path.exists(cpath):
            cast = reads.cast_from_credits(json.load(open(cpath)))
            text_pairs = cast["pairs"]
        witness = {(p["name"].casefold(),
                    (p["paired_with"] or "").casefold()) for p in uniq}
        text = {(p["actor"].casefold(), p["character"].casefold())
                for p in text_pairs}
        print(f"\n=== {ep}: {frames} frames, {len(uniq)} distinct "
              f"pairings (text-side had {len(text_pairs)} pairs; "
              f"agree on {len(witness & text)})")
        for p in uniq:
            print(f"  {p['name']!r} -> {p['paired_with']!r} "
                  f"[header {p['header']!r}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    submit() if args.submit else fetch()
