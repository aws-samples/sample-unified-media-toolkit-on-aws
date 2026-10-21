"""
Live model-matrix harness: EVERY model in the DEPLOYED registry, through the real
gateway, on the real stack — batch AND sync for models whose registry entry says
`batch_job.can_support`, sync only for the rest, all with a real image for vision
models and a text prompt for text-only ones. Embedding models are reported as
NOT TESTABLE and skipped: this workflow is Converse-only — there is no embedding
operation anywhere in lib/ or states/ — so "tested" would be a lie.

Run:
    RUN_TAG=$(date +%s) IMAGE=~/Downloads/m2cv5.png python3 test/model_matrix.py

RUN_TAG is required and goes INSIDE the prompt: workflow_id is a hash over the
rendered records, so a rerun without a fresh tag is answered idempotently from
the previous run's artifacts and proves nothing.

SUBMISSION IS DIRECT Step Functions, not the gateway Lambda (2026-08-14: the
tool is not registered to media-toolkit-gateway yet, so direct execution is the
integration path this harness must prove). The harness therefore does the
gateway's job itself, through the SAME shared modules the gateway uses —
`lib.identity` for the workflow_id/prefix and `lib.input_schema` for parsing —
so the id it derives is the id the gateway WOULD have derived, and a later
gateway submit of the same request is absorbed idempotently rather than re-run.
The jsonl is written first and input_params.json second, preserving the
gateway's ordering guarantee (an input_params.json with no sibling jsonl is the
one combination that fails at read time). PROXY_BUCKET, TOOL_VERSION, the registry key and the
state-machine arn are all read from the DEPLOYED gateway function's env — the
harness tests the stack's own configuration, not the tree's.

What a per-run PASS requires — all of it, because each clause has a silent
failure mode it exists to catch:
  * the execution reached SUCCEEDED;
  * summary.records_succeeded == 1 and records_failed == 0 (a FAILED line in a
    SUCCEEDED execution is the partial-success contract doing its job — for
    this harness that is a model that cannot actually serve the request);
  * response.jsonl exists and its one record carries non-empty output_text;
  * the mode that RAN (summary.inference_mode) is the mode this run asked for —
    a use_batch:true request that quietly downgraded to sync passes every other
    check (the burst harness learned this on 2026-08-12).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import identity                                # noqa: E402
from lib.custom_exceptions import ValidationException  # noqa: E402
from lib.input_schema import parse_records, parse_request  # noqa: E402

REGION = os.environ.get("AWS_REGION", "eu-west-1")
PREFIX = os.environ.get("PREFIX", "infertext")
RUN_TAG = os.environ.get("RUN_TAG") or sys.exit("RUN_TAG is required (see docstring)")
IMAGE = os.path.expanduser(os.environ.get("IMAGE", "~/Downloads/m2cv5.png"))
GATEWAY = f"{PREFIX}-bedrock-inference-gateway-tool"
# Batch jobs of 100 records (1 real + 99 pads) usually land in 15-60 min;
# 3 h is the give-up line, far past "slow" and well short of the 24 h token.
DEADLINE_SEC = int(os.environ.get("DEADLINE_SEC", str(3 * 3600)))
POLL_SEC = 30

lam = boto3.client("lambda", region_name=REGION)
sfn = boto3.client("stepfunctions", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

# The deployed stack's own configuration — bucket, version, registry key, arn —
# so the harness tests what is RUNNING, not what the tree says should be.
_ENV = lam.get_function_configuration(FunctionName=GATEWAY)[
    "Environment"]["Variables"]
PROXY_BUCKET = _ENV["PROXY_BUCKET"]
TOOL_VERSION = _ENV["TOOL_VERSION"]
TARGET_NAME = _ENV.get("TARGET_NAME", "bedrock-inference")
REGISTRY_KEY = os.environ.get("MODEL_REGISTRY_KEY",
                              _ENV["MODEL_REGISTRY_KEY"])
STATE_MACHINE_ARN = _ENV["INFERENCE_STATE_MACHINE_ARN"]


def load_registry() -> tuple[str, list]:
    """The DEPLOYED registry — the harness must test the models the stack
    believes it has, not the tree's master."""
    body = s3.get_object(Bucket=PROXY_BUCKET, Key=REGISTRY_KEY)["Body"].read()
    registry = json.loads(body)
    models = registry["models"] if isinstance(registry, dict) else registry
    print(f"registry s3://{PROXY_BUCKET}/{REGISTRY_KEY}: {len(models)} models, "
          f"version {registry.get('registry_version') if isinstance(registry, dict) else '(bare list)'}, "
          f"stack tool_version {TOOL_VERSION}")
    return PROXY_BUCKET, models


def plan_runs(models: list) -> tuple[list, list]:
    """(runs, skipped) — one run per (model, mode) the registry says is legal."""
    with open(IMAGE, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    runs, skipped = [], []
    for entry in models:
        model_id = entry["model_id"]
        types = entry.get("model_type") or []
        if "embedding" in types:
            skipped.append((model_id, "embedding — Converse-only workflow, "
                            "no embedding path exists"))
            continue
        vision = "vision" in types
        content = [{"text": f"Run {RUN_TAG}: in one short sentence, "
                            + ("describe what this image shows."
                               if vision else "say hello and name yourself.")}]
        if vision:
            content.append({"image": {"format": "png",
                                      "source": {"bytes": image_b64}}})
        modes = ["sync"]
        if (entry.get("batch_job") or {}).get("can_support"):
            modes.append("batch")
        for mode in modes:
            runs.append({
                "model": model_id, "mode": mode, "vision": vision,
                "payload": {
                    "records": [{
                        "record_id": f"mm-{mode}-0",
                        "messages": [{"role": "user", "content": content}],
                    }],
                    "model_chain": [model_id],
                    "use_batch": mode == "batch",
                },
            })
    return runs, skipped


# At most 2 concurrent submits: a vision payload is ~290 KB, and 12 of them
# pushed through one client at once saturated the uplink into per-write SSL
# timeouts on the FIRST run of this harness — 7 of 17 made it, the dead threads
# left no "state", and the poller crashed on the survivors. Submission is not
# what this harness measures, so it gets no concurrency it does not need.
_submit_gate = threading.Semaphore(2)


def _start_direct(payload: dict) -> dict:
    """
    The gateway's run_inference, replayed faithfully outside the Lambda:
    parse → workflow_id → jsonl-then-input_params.json → start_execution named workflow_id.
    Every derivation goes through the SAME shared modules the gateway imports,
    so the execution this starts is the one a later gateway submit would find.
    """
    request = {"tool_version": TOOL_VERSION,
               "model_chain": payload["model_chain"],
               "use_batch": payload["use_batch"]}
    parse_request({**request, "output": {"bucket": PROXY_BUCKET,
                                         "prefix": "placeholder"}})
    records = parse_records(payload["records"])

    workflow_id = identity.compute_workflow_id(
        request, records_digest=identity.digest_records(records),
        tool_version=TOOL_VERSION)
    prefix = identity.output_prefix(TARGET_NAME, workflow_id)
    request["output"] = {"bucket": PROXY_BUCKET, "prefix": prefix}
    request_s3uri = f"s3://{PROXY_BUCKET}/{prefix}/input_params.json"

    # jsonl FIRST, request.json second — the gateway's ordering guarantee:
    # StateValidateAndPlan derives the records key from the request key, so a
    # request.json with no sibling jsonl is the one combination that fails.
    records_jsonl = "".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
        for r in records)
    s3.put_object(Bucket=PROXY_BUCKET, Key=f"{prefix}/request.jsonl",
                  Body=records_jsonl.encode("utf-8"),
                  ContentType="application/jsonl")   # match the gateway's _MIME_JSONL
    s3.put_object(Bucket=PROXY_BUCKET, Key=f"{prefix}/input_params.json",
                  Body=json.dumps(request, indent=2,
                                  ensure_ascii=False).encode("utf-8"),
                  ContentType="application/json")

    workflow_input = {"operation": "StateValidateAndPlan", "uuid": workflow_id,
                      "input": {"request_s3uri": request_s3uri}, "data": {}}
    try:
        resp = sfn.start_execution(stateMachineArn=STATE_MACHINE_ARN,
                                   name=workflow_id,
                                   input=json.dumps(workflow_input))
        arn = resp["executionArn"]
    except sfn.exceptions.ExecutionAlreadyExists:
        # Identical request → the existing run IS the answer (same absorb the
        # gateway performs) — what makes a rerun under the same RUN_TAG safe.
        arn = (STATE_MACHINE_ARN.replace(":stateMachine:", ":execution:")
               + f":{workflow_id}")
    return {"job_id": arn, "workflow_id": workflow_id,
            "output_s3_prefix": f"s3://{PROXY_BUCKET}/{prefix}"}


def submit(run: dict) -> None:
    run["state"] = "SUBMIT_ERROR"   # overwritten on success; never absent
    for attempt in (1, 2, 3):
        try:
            with _submit_gate:
                body = _start_direct(run["payload"])
            break
        except ValidationException as exc:
            run["handle"] = {"error": str(exc)}
            print(f"  submit {run['model']} {run['mode']}: REJECTED {exc}")
            return
        except Exception as exc:                                   # noqa: BLE001
            print(f"  submit {run['model']} {run['mode']}: "
                  f"attempt {attempt} failed ({type(exc).__name__})")
            if attempt == 3:
                run["handle"] = {"error": str(exc)}
                return
            time.sleep(5 * attempt)
    run["handle"] = body
    run["state"] = "RUNNING"
    print(f"  submit {run['model']} {run['mode']}: {body['workflow_id']}")


def poll(run: dict) -> None:
    if run["state"] != "RUNNING":
        return
    desc = sfn.describe_execution(executionArn=run["handle"]["job_id"])
    if desc["status"] == "RUNNING":
        return
    run["state"] = desc["status"]
    run["output"] = json.loads(desc.get("output") or "{}")
    if desc["status"] != "SUCCEEDED":
        run["cause"] = json.loads(desc.get("cause") or "{}") if desc.get(
            "cause") else desc.get("error", "")
    print(f"  {time.strftime('%H:%M:%S')}  {run['model']} {run['mode']}: "
          f"{desc['status']}")


def s3_json(bucket: str, key: str):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:                                              # noqa: BLE001
        return None


def verdict(run: dict, bucket: str) -> dict:
    """The four-clause PASS from the module docstring, plus the receipts."""
    result = {"model": run["model"], "mode": run["mode"],
              "vision": run["vision"], "state": run["state"]}
    if run["state"] == "SUBMIT_ERROR":
        result.update(ok=False, why=str(run["handle"]))
        return result
    if run["state"] != "SUCCEEDED":
        result.update(ok=False, why=json.dumps(run.get("cause") or {})[:300])
        return result

    prefix = run["handle"]["output_s3_prefix"].split(f"{bucket}/", 1)[1]
    summary = json.loads(s3_json(bucket, f"{prefix}/summary.json") or "{}")
    lines = (s3_json(bucket, f"{prefix}/response.jsonl") or b"").decode()
    rows = [json.loads(l) for l in lines.splitlines() if l.strip()]
    text = (rows[0].get("output_text") or "") if rows else ""

    ran_mode = summary.get("inference_mode")
    checks = {
        "succeeded_1_of_1": summary.get("records_succeeded") == 1
                            and summary.get("records_failed") == 0,
        "output_text": bool(text.strip()),
        "mode_as_requested": ran_mode == run["mode"],
    }
    result.update(
        ok=all(checks.values()), checks=checks, ran_mode=ran_mode,
        mode_reason=summary.get("mode_reason"),
        cost=summary.get("estimated_cost_usd") or summary.get("cost_usd"),
        text=text[:120],
        failed_lines=[r.get("error") for r in rows if r.get("error")][:1])
    return result


def main() -> int:
    bucket, models = load_registry()
    runs, skipped = plan_runs(models)
    print(f"\n=== plan: {len(runs)} runs "
          f"({sum(r['mode'] == 'batch' for r in runs)} batch, "
          f"{sum(r['mode'] == 'sync' for r in runs)} sync), "
          f"{len(skipped)} skipped ===")
    for model_id, why in skipped:
        print(f"  skip {model_id}: {why}")

    threads = [threading.Thread(target=submit, args=(r,)) for r in runs]
    print("\n=== submit ===")
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n=== poll ===")
    start = time.time()
    while time.time() - start < DEADLINE_SEC:
        for run in runs:
            poll(run)
        pending = [r for r in runs if r["state"] == "RUNNING"]
        if not pending:
            break
        time.sleep(POLL_SEC)
    else:
        for run in runs:
            if run["state"] == "RUNNING":
                run["state"] = "DEADLINE"

    print("\n=== verdicts ===")
    results = [verdict(r, bucket) for r in runs]
    width = max(len(r["model"]) for r in results) + 2
    for r in sorted(results, key=lambda x: (x["ok"], x["model"], x["mode"])):
        flag = "PASS" if r["ok"] else "FAIL"
        detail = (f"'{r.get('text', '')}'" if r["ok"] else
                  r.get("why") or json.dumps({k: v for k, v in
                                              (r.get("checks") or {}).items()
                                              if not v})
                  + " " + json.dumps(r.get("failed_lines")))
        print(f"  [{flag}] {r['model']:<{width}} {r['mode']:<6} "
              f"reason={r.get('mode_reason')!s:<40.40} {detail}")

    passed = sum(r["ok"] for r in results)
    print(f"\n{passed}/{len(results)} runs passed; {len(skipped)} models "
          "skipped as untestable (embedding).")
    print(json.dumps({"results": results, "skipped": skipped},
                     default=str), file=open("/tmp/model_matrix_results.json", "w"))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
