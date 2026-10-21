#!/usr/bin/env python3
"""
LIVE load test — not part of the offline suite, and deliberately NOT named `test_*.py`
so `unittest discover` cannot pick it up: it needs real AWS credentials, a deployed
stack, and about ten minutes, and it spends real Bedrock money.

    RUN_TAG=$(date +%s) python3 test/burst_semaphore.py

The RUN_TAG is not optional in practice — see point 1 below.

It is checked in because it found a production bug the 291-test offline suite could
not: `TransactionConflictException` killing 11 of 24 sync executions while the cap
itself held perfectly. Contention is the one condition a hermetic double cannot
simulate, and the one the semaphore exists to manage.

Semaphore burst: 24 sync + 4 batch requests fired at once against stack `infertext`,
with the semaphore polled DURING the run.

Why it is shaped this way — three things that would each make the test vacuous:

  1. **Every request must be semantically distinct — across RUNS as well as within
     one.** `workflow_id` is a hash of the request, so 24 identical submissions
     collide into ONE execution and the cap is never contended. Each prompt carries
     its own index AND a `RUN_TAG`: without the tag, a second run of this harness
     re-submits the first run's exact payloads, gets 28 instant hits on already-
     completed executions, and reports "28 SUCCEEDED" over a semaphore that was never
     touched. Measured — that is exactly what the first re-run did.
  2. **Each hold must outlast the poll interval.** A one-record run at max_tokens=1
     finishes in ~2 s, so 24 of them can pass through a cap of 3 without two ever
     being observed together. Each request carries RECORDS_PER_REQUEST records, and
     the sync loop is serial (one Converse call per record, no Map anywhere), so a
     hold lasts ~N × latency.
  3. **"No violation" only means something if the cap was REACHED.** The test asserts
     max_observed == MAX_HOLDERS, not merely <=. A run that never reached 3 proves
     nothing about the cap and is reported as INCONCLUSIVE, not as a pass.
  4. **The batch arm must be seen to have gone to BATCH.** It sent a retired `mode`
     field, which validate_plan warns about and IGNORES, so all four batch requests
     auto-routed to sync — 8 records of ~20 tokens saves $0.0000 against a $0.25
     floor — and every check still passed, because a downgraded request SUCCEEDS.
     `use_batch` now, and a token row must be observed. Same failure shape as 1-3:
     the arm reported PASS while exercising nothing.

The counter and the actual holder-row count are polled separately and compared,
because reading only the counter would hide a holder the counter is not counting. Only
ONE direction of divergence is a bug, though: `release()` deletes the holder row and
the STREAM owns the decrement (invariant 1), so counter AHEAD of rows is the design's
own mid-handoff state. Counter BEHIND rows is the defect — an uncounted holder means
the cap can be exceeded.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import boto3

# Overridable, because the defaults name one developer's dev stack: the cap is per
# PREFIX, so pointing this at a shared prefix bursts someone else's cap too.
REGION = os.environ.get("AWS_REGION_OVERRIDE", "eu-west-1")
PREFIX = os.environ.get("RESOURCE_PREFIX", "infertext")
GATEWAY = f"{PREFIX}-bedrock-inference-gateway-tool"
QUOTA_TABLE = f"{PREFIX}-sync-infer-quotas"
TOKEN_TABLE = f"{PREFIX}-batch-infer-tokens"
MODEL = os.environ.get("MODEL_ID", "global.anthropic.claude-sonnet-4-6")
SCOPE_PK = f"semaphore#bedrock#{MODEL}"
# Pending items live under their OWN partition key, not as a `pending#` SORT key under
# the semaphore partition. Probing `SCOPE_PK` for them — as the first version of this
# harness did — reports "0 waiters queued" no matter how many are actually parked,
# which turned a harness bug into a fake product finding.
PENDING_PK = f"pending#bedrock#{MODEL}"

N_SYNC = 24
N_BATCH = 4
RECORDS_PER_REQUEST = 8
# Must equal lib.input_schema.MAX_CONCURRENT_DEFAULT, or "cap was REACHED" is
# asserted against the wrong number and a healthy run reports INCONCLUSIVE. That
# was the stack's `MaxConcurrentDefault` parameter until 2026-08-12; the constant
# is the single source now, so this can only drift by editing one of the two.
MAX_HOLDERS = int(os.environ.get("MAX_HOLDERS", "3"))
POLL_SECONDS = 1.0
TIMEOUT_SECONDS = 1800
# Pass a fresh tag on every run: `RUN_TAG=$(date +%s) python3 test/burst_semaphore.py`. Defaulted
# rather than required so a forgotten tag at least fails LOUDLY as an inconclusive
# verdict (cap never reached) instead of a quiet fake pass.
RUN_TAG = os.environ.get("RUN_TAG", "r0")

_ctx = base64.b64encode(json.dumps(
    {"custom": {"bedrockAgentCoreToolName":
                "bedrock-inference___run_inference"}}).encode()).decode()

lam = boto3.client("lambda", region_name=REGION,
                   config=boto3.session.Config(read_timeout=300,
                                               retries={"max_attempts": 0}))
sfn = boto3.client("stepfunctions", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)


def payload(mode: str, index: int) -> dict:
    # The index is inside the PROMPT, not just alongside it: the hash is computed over
    # the records, so a distinguishing field the digest does not cover would still
    # collide.
    return {
        "records": [
            {"record_id": f"{mode}{index:02d}r{r}",
             "messages": [{"role": "user", "content": [
                 {"text": f"Hi, there. This is {mode} request {index} record {r} "
                          f"of run {RUN_TAG}."}]}]}
            for r in range(RECORDS_PER_REQUEST)
        ],
        "model_chain": [MODEL],
        # `use_batch`, NOT `mode`. `mode` is RETIRED — validate_plan warns and
        # IGNORES it (lib/input_schema.py) — so the four "batch" requests fell
        # through to the auto guard, and at 8 records of ~20 tokens the estimated
        # saving is $0.0000 against a $0.25 floor, i.e. sync. This harness spent
        # runs reporting "4 batch SUCCEEDED" over four sync executions, a whole
        # third of it proving nothing. Found 2026-08-12 by the batch arm finishing
        # in 113 s with `max batch rows 0` — the number was already printed, just
        # not asserted. It is asserted now.
        "use_batch": mode == "batch",
        "max_tokens": 1,
    }


def submit(mode: str, index: int) -> dict:
    response = lam.invoke(FunctionName=GATEWAY, ClientContext=_ctx,
                          Payload=json.dumps(payload(mode, index)).encode())
    body = json.loads(response["Payload"].read())
    return {"mode": mode, "index": index, "body": body}


# ── the observer ──────────────────────────────────────────────────────────────

samples: list = []
stop = threading.Event()


def snapshot() -> dict:
    """
    A sample is TWO calls, so it is never atomic — which means the read ORDER decides
    which direction a mid-flight acquire shows up as, and getting it backwards
    manufactures the exact defect this harness is looking for.

    `try_acquire` increments the counter and puts the holder row in ONE transaction, so
    counter-behind-rows is impossible by construction. Read counter-then-rows, though,
    and an acquire landing between the two reads yields a stale-low counter beside a
    fresh row count — reported as the bug. Reading ROWS FIRST puts that same race in the
    benign direction (rows stale-low, counter fresh = "counter ahead", which release
    produces legitimately anyway). `ConsistentRead` removes the other source: an
    eventually-consistent counter read is stale-low all on its own.
    """
    rows = ddb.query(
        TableName=QUOTA_TABLE, ConsistentRead=True,
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": {"S": SCOPE_PK}})["Items"]
    holder_rows = [r["sk"]["S"] for r in rows if r["sk"]["S"].startswith("holder#")]
    counter = ddb.get_item(TableName=QUOTA_TABLE, ConsistentRead=True,
                           Key={"pk": {"S": SCOPE_PK}, "sk": {"S": "counter"}}
                           ).get("Item") or {}
    pending_rows = ddb.query(
        TableName=QUOTA_TABLE, ConsistentRead=True,
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": {"S": PENDING_PK}})["Items"]
    tokens = ddb.scan(TableName=TOKEN_TABLE)["Items"]
    return {
        "t": round(time.time() - t0, 1),
        "counter": int(counter.get("holders", {}).get("N", -1)),
        "max_holders": int(counter.get("max_holders", {}).get("N", -1)),
        "holder_rows": len(holder_rows),
        "pending_rows": len(pending_rows),
        "batch_rows": len(tokens),
        "batch_statuses": sorted({i.get("status", {}).get("S", "?") for i in tokens}),
    }


def observe() -> None:
    while not stop.is_set():
        try:
            samples.append(snapshot())
        except Exception as exc:                                    # noqa: BLE001
            samples.append({"t": round(time.time() - t0, 1), "error": str(exc)})
        stop.wait(POLL_SECONDS)


# ── run ───────────────────────────────────────────────────────────────────────

t0 = time.time()
observer = threading.Thread(target=observe, daemon=True)
observer.start()

jobs = [("sync", i) for i in range(N_SYNC)] + [("batch", i) for i in range(N_BATCH)]
with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
    submitted = list(pool.map(lambda job: submit(*job), jobs))

print(f"submitted {len(submitted)} in {time.time() - t0:.1f}s")
arns, failed_submits = {}, []
for item in submitted:
    body = item["body"]
    arn = body.get("job_id")
    if not arn:
        failed_submits.append(item)
        continue
    arns[arn] = item
print(f"  accepted={len(arns)} distinct={len(set(arns))} rejected={len(failed_submits)}")
for item in failed_submits:
    print("  REJECTED", item["mode"], item["index"], json.dumps(item["body"])[:300])

# ── block until every execution is terminal ───────────────────────────────────

terminal, deadline = {}, time.time() + TIMEOUT_SECONDS
while len(terminal) < len(arns) and time.time() < deadline:
    for arn in arns:
        if arn in terminal:
            continue
        status = sfn.describe_execution(executionArn=arn)["status"]
        if status != "RUNNING":
            terminal[arn] = status
    if len(terminal) < len(arns):
        live = [s for s in samples[-1:]]
        print(f"  {time.time() - t0:6.0f}s done={len(terminal)}/{len(arns)} "
              f"{live[0] if live else ''}")
        time.sleep(10)

stop.set()
observer.join(timeout=5)

# ── report ────────────────────────────────────────────────────────────────────

print("\n=== execution outcomes ===")
by_mode: dict = {}
for arn, item in arns.items():
    status = terminal.get(arn, "STILL_RUNNING")
    by_mode.setdefault((item["mode"], status), []).append(item["index"])
for (mode, status), indexes in sorted(by_mode.items()):
    print(f"  {mode:6s} {status:14s} {len(indexes):3d}  {sorted(indexes)}")

# Tally the CAUSE, not just the count. A cap that holds while callers die of
# contention is the bug this harness found the first time, and "11 FAILED" alone does
# not distinguish it from a Bedrock throttle or a bad payload.
failed = [arn for arn, status in terminal.items() if status == "FAILED"]
if failed:
    print("\n=== failure causes ===")
    causes: dict = {}
    for arn in failed:
        events = sfn.get_execution_history(
            executionArn=arn, reverseOrder=True, maxResults=20)["events"]
        detail = next(
            (e[k] for e in events for k in
             ("executionFailedEventDetails", "taskFailedEventDetails",
              "lambdaFunctionFailedEventDetails") if k in e), {})
        cause = (detail.get("cause") or detail.get("error") or "?")
        line = next((ln.strip() for ln in cause.split("\\n")
                     if "Exception" in ln and "Traceback" not in ln), cause[:160])
        causes[line[:200]] = causes.get(line[:200], 0) + 1
    for line, count in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}  {line}")

good = [s for s in samples if "error" not in s]
print(f"\n=== semaphore, {len(good)} samples over "
      f"{good[-1]['t'] if good else 0}s ===")
if good:
    max_counter = max(s["counter"] for s in good)
    max_rows = max(s["holder_rows"] for s in good)
    max_pending = max(s["pending_rows"] for s in good)
    max_batch = max(s["batch_rows"] for s in good)
    # Direction matters, and only ONE direction is a bug. `release()` deletes the
    # holder row and the STREAM owns the decrement (invariant 1), so counter AHEAD of
    # rows is the design's own intermediate state — every sample in the first clean run
    # was exactly 3/2, mid-handoff. Counter BEHIND rows is the real defect: it means a
    # holder exists that the counter is not counting, so the cap can be exceeded.
    # (The two reads are also separate calls, so a sample is never atomic anyway.)
    drift = [s for s in good if s["counter"] >= 0
             and s["counter"] < s["holder_rows"]]
    lagging = [s for s in good if s["counter"] > s["holder_rows"] >= 0]
    over = [s for s in good if s["counter"] > MAX_HOLDERS
            or s["holder_rows"] > MAX_HOLDERS]
    print(f"  max counter        {max_counter}   (cap {MAX_HOLDERS})")
    print(f"  max holder rows    {max_rows}")
    print(f"  max pending rows   {max_pending}")
    print(f"  max batch rows     {max_batch}  statuses "
          f"{sorted({st for s in good for st in s['batch_statuses']})}")
    print(f"  counter BEHIND rows {len(drift)} sample(s)   <- the bug")
    print(f"  counter ahead rows {len(lagging)} sample(s)   "
          f"(stream decrement in flight; expected)")
    print(f"  OVER CAP           {len(over)} sample(s)")
    for s in over[:5]:
        print("    ", s)
    for s in drift[:5]:
        print("     drift:", s)
    print("\n  timeline (counter/rows/pending/batch):")
    print("   ", " ".join(
        f"{s['counter']}/{s['holder_rows']}/{s['pending_rows']}/{s['batch_rows']}"
        for s in good))

print("\n=== final state ===")
print("  ", json.dumps(snapshot()))

print("\n=== verdict ===")
checks = []
checks.append(("every execution SUCCEEDED",
               bool(terminal) and set(terminal.values()) == {"SUCCEEDED"}
               and len(terminal) == len(arns) == len(jobs)))
checks.append((f"cap of {MAX_HOLDERS} never exceeded",
               bool(good) and not over))
checks.append((f"cap of {MAX_HOLDERS} was actually REACHED "
               "(otherwise the test proves nothing)",
               bool(good) and max(max_counter, max_rows) == MAX_HOLDERS))
checks.append(("waiters queued (pending rows observed)",
               bool(good) and max_pending > 0))
# The batch arm has to be seen to have RUN. A row in the token table is the only
# proof: a silent downgrade to sync still returns SUCCEEDED on every execution, and
# every other check here passes on a run that never submitted a Bedrock batch job.
checks.append((f"the {N_BATCH} batch requests actually went to BATCH "
               "(token rows observed — a downgrade to sync passes every other check)",
               bool(good) and max_batch > 0))
checks.append(("counter never fell BEHIND its holder rows "
               "(ahead is the stream handoff, not drift)",
               bool(good) and not drift))
final = snapshot()
checks.append(("holders back to 0, no holder rows, token table empty",
               final["counter"] == 0 and final["holder_rows"] == 0
               and final["batch_rows"] == 0))
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
