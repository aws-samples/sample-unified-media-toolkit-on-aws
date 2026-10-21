"""
Register every catalog_sources/{fleurs,librispeech} reference as a voice,
all in parallel — concurrency test for StateValidateReference /
StateExtractSpeechToken / StateExtractSpeakerEmbedding /
StateExtractSpeechFeat / StateWriteVoiceRecord running N-wide.

Two phases:
  1. upload every wav to S3 (sequential — cheap, no concurrency signal needed)
  2. fire all StartExecution calls concurrently via a thread pool, poll all
     in parallel, report per-voice pass/fail + timing.

Usage: python3 register_all.py <proxy-bucket> <register-voice-state-machine-arn>
"""
import concurrent.futures
import json
import os
import sys
import time
import uuid as uuid_mod

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_SOURCES = os.path.join(HERE, "..", "catalog_sources")
_s3 = boto3.client("s3", region_name="eu-west-1")
_sfn = boto3.client("stepfunctions", region_name="eu-west-1")

UPLOAD_PREFIX = "micro-workflow/tts-on-aws/catalog_sources"


def load_candidates():
    items = []
    fleurs = json.load(open(f"{CATALOG_SOURCES}/fleurs/manifest.json"))
    for r in fleurs:
        items.append({
            "local_path": f"{CATALOG_SOURCES}/fleurs/{r['wav']}",
            "s3_key": f"{UPLOAD_PREFIX}/fleurs/{r['wav']}",
            "ref_transcript": r["ref_text"],
            "name": f"FLEURS-{r['lang']}-{r['gender']}-{r['source_id']}",
            "gender": r["gender"],
            "language": r["lang"],
            "catalog": "default",
            "origin": "FLEURS",
        })
    librispeech = json.load(open(f"{CATALOG_SOURCES}/librispeech/manifest.json"))
    for r in librispeech:
        items.append({
            "local_path": f"{CATALOG_SOURCES}/librispeech/{r['wav']}",
            "s3_key": f"{UPLOAD_PREFIX}/librispeech/{r['wav']}",
            "ref_transcript": r["ref_text"],
            "name": f"LibriSpeech-{r['gender']}-{r['speaker_id']}",
            "gender": r["gender"],
            "language": "en",
            "catalog": "default",
            "origin": "LibriSpeech",
        })
    return items


def upload_all(bucket, items):
    print(f"uploading {len(items)} files to s3://{bucket}/{UPLOAD_PREFIX}/ ...")
    for it in items:
        _s3.upload_file(it["local_path"], bucket, it["s3_key"])
    print("upload done")


def start_one(bucket, state_machine_arn, item):
    sfn_input = {
        "uuid": str(uuid_mod.uuid4()),
        "input": {
            "ref_bucket": bucket,
            "ref_key": item["s3_key"],
            "ref_transcript": item["ref_transcript"],
            "voice": {
                "name": item["name"],
                "gender": item["gender"],
                "language": item["language"],
                "catalog": item["catalog"],
                "origin": item["origin"],
                "license": "CC-BY-4.0",
            },
        },
        "data": {},
    }
    t0 = time.time()
    resp = _sfn.start_execution(
        stateMachineArn=state_machine_arn,
        name=f"catalog-{item['name']}-{int(t0)}"[:80].replace(" ", "_"),
        input=json.dumps(sfn_input))
    return {"name": item["name"], "arn": resp["executionArn"], "start_t": t0}


def poll_one(entry, timeout_s=180):
    deadline = entry["start_t"] + timeout_s
    while time.time() < deadline:
        resp = _sfn.describe_execution(executionArn=entry["arn"])
        status = resp["status"]
        if status != "RUNNING":
            elapsed = time.time() - entry["start_t"]
            result = {"name": entry["name"], "status": status,
                     "elapsed_s": round(elapsed, 1)}
            if status == "SUCCEEDED":
                out = json.loads(resp["output"])
                voice = out.get("data", {}).get("voice", {})
                result["voice_id"] = voice.get("voice_id")
                result["transcript_similarity"] = (
                    out.get("data", {}).get("ref_info", {}).get("transcript_similarity"))
            else:
                result["error"] = (resp.get("cause") or resp.get("error") or "")[:300]
            return result
        time.sleep(3)
    return {"name": entry["name"], "status": "TIMEOUT_POLLING",
            "elapsed_s": round(time.time() - entry["start_t"], 1)}


def main():
    bucket, state_machine_arn = sys.argv[1], sys.argv[2]
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None

    items = load_candidates()
    if only:
        items = [it for it in items if it["name"] in only]
        missing = only - {it["name"] for it in items}
        if missing:
            sys.exit(f"--only names not found: {missing}")
    print(f"{len(items)} candidates loaded "
         f"({sum(1 for i in items if i['origin']=='FLEURS')} FLEURS, "
         f"{sum(1 for i in items if i['origin']=='LibriSpeech')} LibriSpeech)")

    upload_all(bucket, items)

    print(f"\nstarting {len(items)} executions IN PARALLEL...")
    t_batch_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        entries = list(pool.map(lambda it: start_one(bucket, state_machine_arn, it), items))
    print(f"all {len(entries)} StartExecution calls issued in "
         f"{time.time()-t_batch_start:.1f}s")

    print("\npolling all executions concurrently...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(entries)) as pool:
        results = list(pool.map(poll_one, entries))
    t_batch_total = time.time() - t_batch_start

    ok = [r for r in results if r["status"] == "SUCCEEDED"]
    failed = [r for r in results if r["status"] != "SUCCEEDED"]

    print(f"\n=== RESULTS: {len(ok)}/{len(results)} succeeded, "
         f"total wall time {t_batch_total:.1f}s ===")
    for r in sorted(results, key=lambda r: r["elapsed_s"]):
        if r["status"] == "SUCCEEDED":
            print(f"  OK  {r['elapsed_s']:5.1f}s  sim={r.get('transcript_similarity')}  "
                 f"{r['name']}  -> {r['voice_id']}")
        else:
            print(f"  FAIL {r['elapsed_s']:5.1f}s  {r['name']}: "
                 f"{r['status']} {r.get('error','')}")

    json.dump(results, open(f"{HERE}/register_all_results.json", "w"), indent=2)
    print(f"\nfull results written to register_all_results.json")
    if failed:
        print(f"\n{len(failed)} FAILURES — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
