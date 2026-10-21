"""
LLM-based dubbing ADAPTATION (not literal translation) vs. literal
FLEURS translation — does length/timing-aware rewriting actually close
the duration-fit gap measured in dubbing_duration_study.py?

Reuses the same parallel EN/target sentence pairs and the same
synthesis+measurement mechanics; the only difference is the text fed
to TTS: instead of FLEURS' literal human translation, an LLM (Bedrock
Claude Haiku 4.5) is prompted to ADAPT the English source to the
target language at a specific target duration — same meaning, but
free to restructure/compress/expand like a professional dubbing
adaptor would, per the user's explicit instruction that this is an
LLM's job, not literal translation's.

Target duration = the real EN source clip's own measured duration
(same "the video segment defines the slot" framing as the dubbing
design discussion). Speaking-rate constants (chars/sec per language)
are the ACTUAL medians measured from dubbing_duration_study.py's own
literal-translation synthesis output — not assumed from any style
guide — used only to give the LLM a concrete character-count target
in its prompt; it has no other way to know CosyVoice3's real rate.
"""
from __future__ import annotations

import json
import sys
import time
import uuid as uuid_mod

import os

import boto3

_sfn = boto3.client("stepfunctions", region_name="eu-west-1")
_bedrock = boto3.client("bedrock-runtime", region_name="eu-west-1")

BUCKET = os.environ["TTS_PROXY_BUCKET"]
SYNTH_ARN = os.environ["TTS_SYNTH_ARN"]  # tts-synthesis state machine
MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

VOICE_BY_LANG = {
    "it": "c300471c-2048-4a95-b6cd-b2bae41c2584",
    "de": "96203e29-e903-46d6-9834-931d1490ecca",
    "es": "329d9db5-866c-441d-87d2-6bbc24a4398e",
    "ja": "f9a0a4db-f433-43af-b83d-b698b1e1b34d",
}

# Measured medians from dubbing_duration_study.py's own literal-translation
# output (chars / synthesized_duration_s) — the real empirical CosyVoice3
# speaking rate per language on this pipeline, not a generic estimate.
CHARS_PER_SEC = {"it": 12.97, "de": 14.75, "es": 12.73, "ja": 4.19}

LANG_NAME = {"it": "Italian", "de": "German", "es": "Spanish", "ja": "Japanese"}


def adapt(en_text, target_lang, target_duration_s):
    target_chars = round(target_duration_s * CHARS_PER_SEC[target_lang])
    prompt = f"""You are a professional dubbing adaptor (not a literal translator). Your job is to rewrite the following English line in {LANG_NAME[target_lang]} so that it:
1. Preserves the core meaning
2. Sounds natural when spoken aloud
3. Takes approximately {target_duration_s:.1f} seconds to speak at a natural pace — target length is about {target_chars} characters in {LANG_NAME[target_lang]} (this is a real empirically-measured speaking rate for the TTS voice, not a guess)

You may restructure, compress, or expand the sentence as needed — this is dubbing adaptation, not translation. Prioritize hitting the target duration over literal accuracy, as long as the core meaning survives.

English line: "{en_text}"

Respond with ONLY the {LANG_NAME[target_lang]} adapted line, no explanation, no quotes."""

    resp = _bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }))
    body = json.loads(resp["body"].read())
    return body["content"][0]["text"].strip().strip('"')


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
        stateMachineArn=SYNTH_ARN, name=f"dubadapt-{tag}-{int(time.time()*1000)}"[:80],
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
    # reuse the literal-translation study's already-measured EN durations
    # and ids so results are directly comparable pair-for-pair
    literal = json.load(open("dubbing_duration_study_results.json"))
    langs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["it", "de", "es", "ja"]
    n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    all_results = {}
    for lang in langs:
        print(f"\n{'='*70}\n{lang.upper()} (LLM-adapted)\n{'='*70}")
        cases = literal[lang][:n_pairs]

        adapted_cases = []
        for c in cases:
            adapted_text = adapt(c["en_text"], lang, c["en_duration_s"])
            adapted_cases.append((c["id"], c["en_text"], c["en_duration_s"], adapted_text))
            print(f"  id={c['id']} en_dur={c['en_duration_s']:.2f}s "
                 f"adapted=\"{adapted_text[:60]}\"")

        arns = []
        for sid, en_text, en_dur, adapted_text in adapted_cases:
            arn = start_synth(VOICE_BY_LANG[lang], adapted_text, f"{lang}-{sid}")
            arns.append((sid, en_text, en_dur, adapted_text, arn))
            time.sleep(0.05)

        lang_results = []
        for sid, en_text, en_dur, adapted_text, arn in arns:
            r = poll(arn)
            if r["status"] == "SUCCEEDED":
                ratio = r["duration_s"] / en_dur
                lang_results.append({
                    "id": sid, "en_text": en_text, "en_duration_s": round(en_dur, 2),
                    "adapted_text": adapted_text, "tgt_duration_s": r["duration_s"],
                    "ratio": round(ratio, 3),
                })
                print(f"  id={sid} en={en_dur:.2f}s tgt={r['duration_s']:.2f}s ratio={ratio:.3f}")
            else:
                print(f"  id={sid} FAILED: {r}")
        all_results[lang] = lang_results

    json.dump(all_results, open("dubbing_adaptation_study_results.json", "w"),
             indent=2, ensure_ascii=False)

    print(f"\n\n{'='*70}\nSUMMARY (LLM-adapted)\n{'='*70}")
    import statistics
    for lang, results in all_results.items():
        ratios = [r["ratio"] for r in results]
        if not ratios:
            print(f"{lang}: no successful measurements")
            continue
        within_10 = sum(1 for r in ratios if abs(r - 1.0) <= 0.10)
        within_20 = sum(1 for r in ratios if abs(r - 1.0) <= 0.20)
        within_30 = sum(1 for r in ratios if abs(r - 1.0) <= 0.30)
        print(f"{lang}: n={len(ratios)} mean={statistics.mean(ratios):.3f} "
             f"median={statistics.median(ratios):.3f} "
             f"stdev={statistics.stdev(ratios):.3f} "
             f"min={min(ratios):.3f} max={max(ratios):.3f}")
        print(f"       within ±10%: {within_10}/{len(ratios)} ({within_10/len(ratios)*100:.0f}%)  "
             f"within ±20%: {within_20}/{len(ratios)} ({within_20/len(ratios)*100:.0f}%)  "
             f"within ±30%: {within_30}/{len(ratios)} ({within_30/len(ratios)*100:.0f}%)")


if __name__ == "__main__":
    main()
