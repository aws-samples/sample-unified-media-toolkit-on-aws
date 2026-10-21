"""
v2 of the adaptation study — two changes from dubbing_adaptation_study.py,
tested together so we can isolate each effect from one run:

1. Model: Sonnet 4.5 instead of Haiku 4.5 (eu.anthropic.claude-sonnet-4-5-
   20250929-v1:0) — does a stronger model adapt closer to the target
   length on its own, single-shot?
2. Multi-candidate: prompt for 3 candidates at explicitly DIFFERENT
   lengths (short/medium/long around the target, not 3 independent
   attempts at the same target — asking for identical-target attempts
   from one model call tends to produce near-duplicate lengths, which
   would make best-of-3 selection pointless) — does synthesizing all 3
   and picking the closest-fitting one meaingfully beat a single shot?

Reports THREE numbers per language so the two effects are separable:
  - candidate_1 only  (comparable to the v1 Haiku single-shot baseline,
                       isolates the model-quality effect alone)
  - best_of_3         (isolates the added value of generating and
                       measuring multiple candidates)
  - all 3 raw ratios   (so the underlying spread is visible, not just
                       the two summary picks)

Same EN source durations/ids as the v1 studies (loaded from
dubbing_duration_study_results.json) so results are directly comparable
pair-for-pair across all three studies (literal / Haiku-1x / Sonnet-3x).
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid as uuid_mod

import os

import boto3

_sfn = boto3.client("stepfunctions", region_name="eu-west-1")
_bedrock = boto3.client("bedrock-runtime", region_name="eu-west-1")

SYNTH_ARN = os.environ["TTS_SYNTH_ARN"]  # tts-synthesis state machine
MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

VOICE_BY_LANG = {
    "it": "c300471c-2048-4a95-b6cd-b2bae41c2584",
    "de": "96203e29-e903-46d6-9834-931d1490ecca",
    "es": "329d9db5-866c-441d-87d2-6bbc24a4398e",
    "ja": "f9a0a4db-f433-43af-b83d-b698b1e1b34d",
}

# Same empirically-measured medians as v1 (chars / synthesized_duration_s
# from the literal-translation study's own output) — unchanged, still the
# only real signal we have for this pipeline's per-language speaking rate.
CHARS_PER_SEC = {"it": 12.97, "de": 14.75, "es": 12.73, "ja": 4.19}

LANG_NAME = {"it": "Italian", "de": "German", "es": "Spanish", "ja": "Japanese"}

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\):]\s*(.+?)\s*$")


def adapt_3(en_text, target_lang, target_duration_s):
    target_chars = round(target_duration_s * CHARS_PER_SEC[target_lang])
    short_chars = round(target_chars * 0.85)
    long_chars = round(target_chars * 1.15)
    prompt = f"""You are a professional dubbing adaptor (not a literal translator). Rewrite the following English line in {LANG_NAME[target_lang]} three DIFFERENT ways, so that together they give a range of lengths around a target — this is for a dubbing pipeline that will pick whichever candidate best fits a timing window.

All three must preserve the core meaning and sound natural when spoken aloud. You may restructure, compress, or expand as needed — this is dubbing adaptation, not literal translation.

Target speaking time: approximately {target_duration_s:.1f} seconds at a natural pace (a real empirically-measured speaking rate for the TTS voice, not a guess).

Produce exactly three candidates, each on its own line, numbered 1-3:
1. A SHORTER candidate, approximately {short_chars} characters in {LANG_NAME[target_lang]}
2. A candidate at the TARGET length, approximately {target_chars} characters in {LANG_NAME[target_lang]}
3. A LONGER candidate, approximately {long_chars} characters in {LANG_NAME[target_lang]}

English line: "{en_text}"

Respond with ONLY the three numbered lines, no explanation, no extra text."""

    resp = _bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }))
    body = json.loads(resp["body"].read())
    raw = body["content"][0]["text"].strip()

    candidates = []
    for line in raw.splitlines():
        m = _NUMBERED_LINE_RE.match(line)
        if m:
            candidates.append(m.group(1).strip().strip('"'))
    if len(candidates) != 3:
        raise ValueError(f"expected 3 numbered candidates, got {len(candidates)}: {raw!r}")
    return candidates


def start_synth(voice_id, text, tag):
    sfn_input = {
        "uuid": str(uuid_mod.uuid4()),
        "input": {
            "batches": [{"voice_id": voice_id, "transcripts": [{"text": text}]}],
            "config": {"flow_steps": 6},
        },
        "data": {},
    }
    resp = _sfn.start_execution(
        stateMachineArn=SYNTH_ARN, name=f"dubv2-{tag}-{int(time.time()*1000)}"[:80],
        input=json.dumps(sfn_input))
    return resp["executionArn"]


def poll(arn, timeout_s=240):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = _sfn.describe_execution(executionArn=arn)
        if resp["status"] != "RUNNING":
            if resp["status"] != "SUCCEEDED":
                return {"status": resp["status"], "error": (resp.get("cause") or "")[:200]}
            out = json.loads(resp["output"])
            for it in out["data"].get("iterators", []):
                item = it.get("item_data") or it
                o = item["outputs"][0]
                if "error" in o:
                    return {"status": "SYNTH_ERROR", "error": o["error"][:200]}
                return {"status": "SUCCEEDED", "duration_s": o["duration_s"]}
        time.sleep(2)
    return {"status": "TIMEOUT"}


def main():
    literal = json.load(open("dubbing_duration_study_results.json"))
    langs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["it", "de", "es", "ja"]
    n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    all_results = {}
    for lang in langs:
        print(f"\n{'='*70}\n{lang.upper()} (Sonnet 4.5, 3 candidates)\n{'='*70}")
        cases = literal[lang][:n_pairs]

        per_case = []
        for c in cases:
            try:
                candidates = adapt_3(c["en_text"], lang, c["en_duration_s"])
            except ValueError as exc:
                print(f"  id={c['id']} ADAPT FAILED: {exc}")
                continue
            print(f"  id={c['id']} en_dur={c['en_duration_s']:.2f}s")
            for i, cand in enumerate(candidates, 1):
                print(f"    [{i}] \"{cand[:70]}\"")
            per_case.append((c["id"], c["en_text"], c["en_duration_s"], candidates))

        # fire all 3 candidates x all cases concurrently
        pending = []
        for sid, en_text, en_dur, candidates in per_case:
            for i, cand in enumerate(candidates, 1):
                arn = start_synth(VOICE_BY_LANG[lang], cand, f"{lang}-{sid}-c{i}")
                pending.append((sid, en_text, en_dur, i, cand, arn))
                time.sleep(0.05)

        by_case = {}
        for sid, en_text, en_dur, cand_i, cand_text, arn in pending:
            r = poll(arn)
            by_case.setdefault(sid, {"en_text": en_text, "en_duration_s": en_dur, "candidates": {}})
            if r["status"] == "SUCCEEDED":
                ratio = r["duration_s"] / en_dur
                by_case[sid]["candidates"][cand_i] = {
                    "text": cand_text, "duration_s": r["duration_s"], "ratio": round(ratio, 3)}
                print(f"  id={sid} c{cand_i}: dur={r['duration_s']:.2f}s ratio={ratio:.3f}")
            else:
                print(f"  id={sid} c{cand_i} FAILED: {r}")

        all_results[lang] = by_case

    json.dump(all_results, open("dubbing_adaptation_study_v2_results.json", "w"),
             indent=2, ensure_ascii=False)

    print(f"\n\n{'='*70}\nSUMMARY (Sonnet 4.5, 3 candidates)\n{'='*70}")
    import statistics
    for lang, by_case in all_results.items():
        cand1_ratios = []
        best_ratios = []
        for sid, case in by_case.items():
            cands = case["candidates"]
            if 1 in cands:
                cand1_ratios.append(cands[1]["ratio"])
            if cands:
                best = min(cands.values(), key=lambda c: abs(c["ratio"] - 1.0))
                best_ratios.append(best["ratio"])

        for label, ratios in [("candidate_1_only", cand1_ratios), ("best_of_3", best_ratios)]:
            if not ratios:
                print(f"{lang} [{label}]: no data")
                continue
            within_10 = sum(1 for r in ratios if abs(r - 1.0) <= 0.10)
            within_20 = sum(1 for r in ratios if abs(r - 1.0) <= 0.20)
            print(f"{lang} [{label}]: n={len(ratios)} mean={statistics.mean(ratios):.3f} "
                 f"median={statistics.median(ratios):.3f} "
                 f"±10%: {within_10}/{len(ratios)} ({within_10/len(ratios)*100:.0f}%) "
                 f"±20%: {within_20}/{len(ratios)} ({within_20/len(ratios)*100:.0f}%)")


if __name__ == "__main__":
    main()
