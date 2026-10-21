"""
StateValidateReference ASR reliability test — run the same 46-candidate
registration batch N times (cleaning the DynamoDB catalog between runs,
NOT re-uploading — same S3 audio bytes every run) and track per-clip
pass/fail + similarity across runs.

Answers: does whisper-base's ASR transcription flip verdict on the SAME
audio across repeated runs, and how often — real data instead of the
two anecdotal flips found manually in the prior session.

Usage: python3 reliability_test.py <bucket> <register-arn> <n_runs>
"""
import json
import os
import subprocess
import sys
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.environ["TTS_VOICE_REGISTRY_TABLE"]


def clean_catalog(bucket):
    ddb = boto3.client("dynamodb", region_name="eu-west-1")
    s3 = boto3.client("s3", region_name="eu-west-1")
    resp = ddb.query(TableName=TABLE, IndexName="catalog-index",
        KeyConditionExpression="#c = :c", ExpressionAttributeNames={"#c": "catalog"},
        ExpressionAttributeValues={":c": {"S": "default"}})
    items = resp["Items"]
    for it in items:
        voice_id = it["voice_id"]["S"]
        for field in ("s3_conditioning_key", "s3_preview_key", "s3_source_key"):
            key = it.get(field, {}).get("S")
            if key:
                s3.delete_object(Bucket=bucket, Key=key)
        ddb.delete_item(TableName=TABLE, Key={"pk": {"S": f"voice#{voice_id}"}, "sk": {"S": "record"}})
    return len(items)


def main():
    bucket, arn, n_runs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    all_runs = []

    for run_i in range(n_runs):
        print(f"\n{'='*70}\nRUN {run_i+1}/{n_runs}\n{'='*70}")
        n_deleted = clean_catalog(bucket)
        print(f"cleaned {n_deleted} existing records")

        t0 = time.time()
        proc = subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            [sys.executable, f"{HERE}/register_all.py", bucket, arn],  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args
            capture_output=True, text=True)
        print(proc.stdout[-2000:])
        if proc.returncode != 0:
            print("(non-zero exit, some failures — expected, continuing)")

        results = json.load(open(f"{HERE}/register_all_results.json"))
        run_data = {r["name"]: {"status": r["status"],
                                "similarity": r.get("transcript_similarity")}
                   for r in results}
        all_runs.append(run_data)
        print(f"run {run_i+1} done in {time.time()-t0:.1f}s: "
             f"{sum(1 for r in results if r['status']=='SUCCEEDED')}/{len(results)} passed")

    # aggregate: per-voice, across all runs
    all_names = sorted(set(n for run in all_runs for n in run))
    print(f"\n\n{'='*70}\nAGGREGATE ACROSS {n_runs} RUNS\n{'='*70}")
    flippers = []
    always_pass = 0
    always_fail = 0
    for name in all_names:
        statuses = [run.get(name, {}).get("status", "MISSING") for run in all_runs]
        sims = [run.get(name, {}).get("similarity") for run in all_runs]
        n_pass = sum(1 for s in statuses if s == "SUCCEEDED")
        if n_pass == n_runs:
            always_pass += 1
        elif n_pass == 0:
            always_fail += 1
        else:
            flippers.append((name, statuses, sims))

    print(f"\nalways passed: {always_pass}/{len(all_names)}")
    print(f"always failed: {always_fail}/{len(all_names)}")
    print(f"FLIPPED (passed some runs, failed others): {len(flippers)}/{len(all_names)}")
    for name, statuses, sims in flippers:
        print(f"  {name}: statuses={statuses} similarities={sims}")

    if always_fail:
        print("\nconsistently-failing voices:")
        for name in all_names:
            statuses = [run.get(name, {}).get("status", "MISSING") for run in all_runs]
            if all(s != "SUCCEEDED" for s in statuses):
                print(f"  {name}")

    json.dump({"runs": all_runs, "flippers": [f[0] for f in flippers],
              "always_pass": always_pass, "always_fail": always_fail,
              "n_runs": n_runs},
             open(f"{HERE}/reliability_test_results.json", "w"), indent=2)
    print(f"\nfull data written to reliability_test_results.json")


if __name__ == "__main__":
    main()
