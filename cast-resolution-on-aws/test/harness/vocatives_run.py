#!/usr/bin/env python3
"""P2-b harness: LLM vocative resolution for one local asset, sync.

Pipeline: bind free evidence -> collect UNRESOLVED vocative sites ->
one bedrock-inference run over their conversation windows -> firewall +
vote aggregation (same thresholds as the deterministic voter) ->
RE-BIND with the new resolved cues injected -> cast adjudication reuses
the richer addressed_as evidence -> persons.json.

Prints the deltas that answer the ledger question (named persons,
resolved vocatives, adjudication assignments) so the sweep scorer can
pool them.

Usage:
    python3 scripts/vocatives_run.py --local /tmp/gt/in_<asset> -o out
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
from lib import transcript_cues  # noqa: E402
from lib import vocatives  # noqa: E402
from lib import vocatives_llm  # noqa: E402

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


def _run_records(records, token, deadline_sec=1800):
    sub = tool("run_inference", {"records": records, "use_batch": False,
                                 "idempotency_token": token})
    if sub.get("error"):
        raise SystemExit(f"refused: {sub.get('message')}")
    t0, wait = time.time(), 5
    while True:
        st = tool("get_job_status", {"job_id": sub["job_id"]})
        if st.get("status") in ("SUCCEEDED", "FAILED", "TIMED_OUT",
                                "ABORTED"):
            break
        if time.time() - t0 > deadline_sec:
            raise SystemExit(f"inference still RUNNING after "
                             f"{deadline_sec}s — giving up on "
                             f"{sub['job_id']}")
        time.sleep(wait)
        wait = min(wait * 2, 30)
    if st["status"] != "SUCCEEDED":
        raise SystemExit(f"inference {st['status']}: "
                         f"{str(st.get('error'))[:200]}")
    # the ONE response parser (drops non-SUCCESS records, same as the
    # states) — a second inline copy diverged on exactly that filter
    from lib import s3io
    return s3io.get_response_texts(st["output_s3_prefix"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    ep = os.path.basename(args.local.rstrip("/")).removeprefix("in_")

    (transcript, faces, _shots, overlay, credits, src, speaker_f0,
     _screen_texts) = bind_pipeline.load_local_asset(args.local)
    if not transcript:
        print(f"{ep}: no transcript — nothing to resolve")
        return
    lang = transcript.get("language") or "en"
    cues = transcript_cues.extract_cues(transcript["turns"], lang)
    if not cues:
        print(f"{ep}: no cues (language={lang}) — nothing to resolve")
        return
    resolved = vocatives.resolve_vocatives(transcript["turns"], cues)
    windows = vocatives_llm.build_windows(transcript["turns"], cues,
                                          resolved)
    print(f"{ep}: {len(resolved)} adjacency-resolved, "
          f"{len(windows)} unresolved sites")
    llm_cues = []
    if windows:
        texts = _run_records(vocatives_llm.build_records(windows, harness_prompts.system("cast-vocatives")),
                             f"pb-vocllm-{ep}-20260905")
        votes = vocatives_llm.parse_answers(texts, windows)
        llm_cues = vocatives_llm.aggregate(votes)
        print(f"{ep}: {len(votes)} firewalled votes -> "
              f"{len(llm_cues)} resolved names")

    # baseline vs enriched bind, same pipeline
    base = bind_pipeline.bind(transcript, faces, overlay, dict(src),
                              credits, speaker_f0=speaker_f0)
    doc = bind_pipeline.bind(transcript, faces, overlay, src, credits,
                             extra_cues=llm_cues, speaker_f0=speaker_f0)

    # adjudication rides the richer addressed_as evidence
    cast = reads.cast_from_credits(credits) if credits else {}
    assignments = []
    dossiers = adjudicate.build_dossiers(
        doc, cast, turns=transcript["turns"])
    if dossiers:
        texts = _run_records(adjudicate.build_records(dossiers, cast, harness_prompts.system("cast-adjudicate")),
                             f"pb-vocadj-{ep}-20260905")
        assignments = adjudicate.parse_verdicts(texts, cast, dossiers)
        doc = adjudicate.apply_verdicts(doc, assignments)
        doc = critic.critique(doc)

    os.makedirs(args.outdir, exist_ok=True)
    json.dump(doc, open(os.path.join(args.outdir, "persons.json"), "w"),
              indent=1)
    named = lambda d: sum(1 for p in d["persons"] if p.get("name"))  # noqa
    print(f"{ep}: named {named(base)} -> {named(doc)}, "
          f"llm_resolved={len(llm_cues)}, "
          f"adjudicated={len(assignments)}")
    for c in llm_cues:
        print(f"  vocative: {c['name']} -> {c['speaker_id']} "
              f"({c['votes']} votes)")
    by_id = {p["person_id"]: p for p in doc["persons"]}
    for v in assignments:
        p = by_id.get(v["person_id"], {})
        print(f"  adjudicated: [{v['person_id']}] name={p.get('name')} "
              f"character={p.get('character_name')}")


if __name__ == "__main__":
    main()
