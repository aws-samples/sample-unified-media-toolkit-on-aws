"""
Directly re-test the 3 clips that flipped pass/fail in the earlier
session (fr_1913, ko_1883, ko_1929) — N repeated register_voice
executions each, on the SAME uploaded audio, to settle whether the
earlier flip was genuine run-to-run non-determinism in this deployed
environment or something else (e.g. a since-fixed bug, a transient
container anomaly, or operator error).
"""
import json
import sys
import time
import uuid as uuid_mod

import os

import boto3

_sfn = boto3.client("stepfunctions", region_name="eu-west-1")

BUCKET = os.environ["TTS_PROXY_BUCKET"]
ARN = os.environ["TTS_REGISTER_VOICE_ARN"]  # register-voice state machine

CASES = [
    {"key": "micro-workflow/tts-on-aws/catalog_sources/recheck/fr_1913.wav",
     "text": "Le rugissement du tigre ne ressemble pas au rugissement ample du lion, mais plutôt à une phrase dont les mots seraient des cris et des grondements.",
     "lang": "fr", "name": "recheck-fr-1913"},
    {"key": "micro-workflow/tts-on-aws/catalog_sources/recheck/ko_1883.wav",
     "text": "다리 밑 수직 간격은 15미터이며, 공사는 2011년 8월에 마무리되었으며, 해당 다리의 통행금지는 2017년 3월까지이다.",
     "lang": "ko", "name": "recheck-ko-1883"},
    {"key": "micro-workflow/tts-on-aws/catalog_sources/recheck/ko_1929.wav",
     "text": "델 포트로가 2세트에서 먼저 어드밴티지를 얻었지만 6 대 6이 된 후 다시 타이 브레이크가 필요했다.",
     "lang": "ko", "name": "recheck-ko-1929"},
]


def start_one(case, attempt):
    sfn_input = {
        "uuid": str(uuid_mod.uuid4()),
        "input": {
            "ref_bucket": BUCKET, "ref_key": case["key"],
            "ref_transcript": case["text"],
            "voice": {"name": f"{case['name']}-a{attempt}", "gender": "female",
                     "language": case["lang"], "catalog": "recheck",
                     "origin": "FLEURS", "license": "CC-BY-4.0"},
        },
        "data": {},
    }
    resp = _sfn.start_execution(
        stateMachineArn=ARN, name=f"recheck-{case['name']}-a{attempt}-{int(time.time())}",
        input=json.dumps(sfn_input))
    return resp["executionArn"]


def poll(arn, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = _sfn.describe_execution(executionArn=arn)
        if resp["status"] != "RUNNING":
            if resp["status"] == "SUCCEEDED":
                out = json.loads(resp["output"])
                sim = out.get("data", {}).get("ref_info", {}).get("transcript_similarity")
                return {"status": "SUCCEEDED", "similarity": sim}
            cause = resp.get("cause") or ""
            heard = None
            if "Heard:" in cause:
                heard = cause.split("Heard:", 1)[1].split(". A mismatched")[0].strip()
            return {"status": resp["status"], "heard": heard}
        time.sleep(2)
    return {"status": "TIMEOUT"}


def main():
    n_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    results = {c["name"]: [] for c in CASES}

    for attempt in range(1, n_attempts + 1):
        print(f"\n--- attempt {attempt}/{n_attempts} ---")
        arns = {c["name"]: start_one(c, attempt) for c in CASES}
        for name, arn in arns.items():
            r = poll(arn)
            results[name].append(r)
            print(f"  {name}: {r}")

    print(f"\n{'='*60}\nSUMMARY over {n_attempts} attempts\n{'='*60}")
    for name, attempts in results.items():
        sims = [a.get("similarity") for a in attempts]
        statuses = [a["status"] for a in attempts]
        print(f"{name}:")
        print(f"  statuses: {statuses}")
        print(f"  similarities: {sims}")
        heards = {a.get("heard") for a in attempts if a.get("heard")}
        if len(heards) > 1:
            print(f"  MULTIPLE DISTINCT TRANSCRIPTIONS ACROSS ATTEMPTS:")
            for h in heards:
                print(f"    - {h}")
        elif heards:
            print(f"  consistent transcription: {list(heards)[0]}")

    json.dump(results, open("recheck_flippers_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
