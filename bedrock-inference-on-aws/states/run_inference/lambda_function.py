"""
StateRunInference — the sync loop (README.md §5.1).

Forty serial Converse calls will not fit in one 15-minute invocation, so the loop
checkpoints per record and re-enters. Verbatim the pattern in
`face-recognition-on-aws/states/recognize_celebrities/lambda_function.py:55-61`.

Three details are load-bearing, and each one fails *silently* if dropped (§12.3):

1. **`refresh_holder()` once, on entry** — not per record. The holder TTL bounds a
   *state entry*, and the re-entry **is** the state entry. `HOLDER_TTL_SEC = 3600`
   against Lambda's 900 s hard ceiling leaves 2700 s of margin, so no single
   invocation can outlive its own refresh; the bound is structural, not measured.
   It is still a *correctness* requirement, because the span outlives the TTL: a
   2.5 h run (10 re-entries) acquired at t=0 has its holder expire at t=3600 while
   work continues to t=9000. DynamoDB then reclaims the holder, the quota stack's
   stream decrements, and a waiter is admitted **while this loop is still calling
   Bedrock** — the cap silently exceeded, no error anywhere.

2. **Checkpoint granularity is per record.** Coarser re-pays for completed Converse
   calls, which is exactly `web-inference`'s defect: `helper.js:runInference` writes
   `chunk_N.json` per chunk and never reads it back, so retry attempt 12
   re-inferences chunks 0–10 — paid for twelve times.

3. **`get_json_if_exists` returns `None` only for a 404.** A broad `except` reads a
   throttled S3 GET as "no checkpoint yet" and re-pays for every completed record, so
   the failure gets *more* expensive under load — which is when it happens.

**One in-flight Converse call per holder.** A lease permits exactly one concurrent
call (§10 item 1). This loop is serial for that reason; parallelising it inside the
held span would turn a cap of 3 into 3×N with no error anywhere. Parallelism means
multiple leases, which means multiple executions.
"""
from __future__ import annotations

import logging

from lib import bedrock as bedrock_lib
from lib import response as response_lib
from lib import s3_utils, semaphore_client
from lib.batch_client import bedrock_runtime
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.input_schema import (
    DEFAULT_MAX_RETRIES,
    MODEL_REGISTRY_KEY,
    PROXY_BUCKET,
    REMAINING_TIME_BUFFER_MS,
    parse_records,
    records_key,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    data = event["data"]
    quota = data["quota"]
    config = data["config"]
    request = data["request"]
    output = data["output"]
    progress_key = data["keys"]["progress"]
    execution_arn = event.get("execution_arn") or ""

    # ONCE, on entry — invariant 3. Best-effort by design: the caller is mid-span
    # and cannot usefully abort, and the counter's own TTL is the backstop. Logged at
    # ERROR because a lost holder means the cap is being exceeded right now.
    if execution_arn:
        if not semaphore_client.refresh_holder(
                quota["service"], quota["resource"], execution_arn):
            logger.error(
                "HOLDER LOST mid-hold on %s#%s for %s — the cap is being exceeded",
                quota["service"], quota["resource"], execution_arn)

    # Re-parsed rather than trusted from the plan: the plan carries counts, not the
    # records, because 40 rendered prompts do not fit in a 256 KB Step Functions
    # payload (§4). `parse_records` is the same function `StateValidateAndPlan` ran,
    # so record validation cannot differ between plan and run.
    records = parse_records(
        s3_utils.get_jsonl(
            request["bucket"],
            request.get("records_key") or records_key(request["control_key"])))

    registry = bedrock_lib.load_registry(
        s3_utils.s3, PROXY_BUCKET, MODEL_REGISTRY_KEY)
    chain = bedrock_lib.build_chain(registry, config["model_chain"])
    # The plan's resolved routing scope (default or request override). A concrete
    # region routes the runtime CLIENT there; global/geo scopes prefix the model
    # id inside build_converse_params and run from the Lambda's own region.
    inference_region = config.get("inference_region") or None
    client = bedrock_runtime(
        chain[0].resolve_client_region(inference_region))

    state = s3_utils.get_json_if_exists(output["bucket"], progress_key) or {}
    done = state.get("done") or {}
    resumed = len(done)

    for index, record in enumerate(records):
        record_id = record["record_id"]
        if record_id in done:
            continue                       # already inferred — never re-paid

        if context is not None and \
                context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
            _checkpoint(output["bucket"], progress_key, done, len(records))
            raise LambdaResourceTimeoutException(
                f"checkpointed at record {index} of {len(records)} "
                f"({len(done)} complete)")

        done[record_id] = _infer_one(client, chain, record, config,
                                     inference_region)

    _checkpoint(output["bucket"], progress_key, done, len(records))

    # `response.jsonl` in **request order**, not completion order (§4.2). The
    # checkpoint is a dict keyed by `record_id`, so the order has to be re-imposed
    # from the record list — a caller zipping its own inputs against the response
    # lines positionally would otherwise silently mis-pair them after a re-entry.
    lines = [done[record["record_id"]] for record in records]
    s3_utils.put_jsonl(output["bucket"], data["keys"]["response"], lines)

    succeeded = sum(1 for line in lines
                    if line.get("status") == response_lib.STATUS_SUCCESS)
    logger.info("%d/%d records succeeded (%d resumed from the checkpoint)",
                succeeded, len(lines), resumed)

    return {
        "operation": "StateRunInference",
        "status": "COMPLETED",
        "uuid": event.get("uuid") or "",
        "input": event.get("input") or {},
        "data": {
            **data,
            "records_total": len(records),
            "records_resumed": resumed,
            "records_succeeded": succeeded,
        },
    }


def _checkpoint(bucket: str, key: str, done: dict, total: int) -> None:
    """
    Persist what is finished. Written before *and* after the deadline raise, and once
    more at the end — the last write costs one PUT and removes the case where a clean
    finish leaves the final record unrecorded.
    """
    s3_utils.put_json(bucket, key, {"done": done, "records_total": total})


def _infer_one(client, chain: list, record: dict, config: dict,
               inference_region: str | None = None) -> dict:
    """
    One record through the model chain. **Never raises for a model-quality reason.**

    A failed record is a `FAILED` line, not a failed execution (§7) — the chain walk
    is per record, so one poisoned record must not downgrade the other 39. The single
    hard failure is *every* record failing, and `StateSummarize` owns that decision
    because only it can see the whole set.

    A saturated quota is recorded as a failure here rather than raised. That differs
    from `conversation-analysis`, which escalates it, and the reason is the lease:
    this loop already holds a slot, so throttling means the *service* is refusing,
    not that the local cap is full. Failing the execution would abandon the completed
    records and release a slot the next execution would immediately re-acquire and
    hit the same wall with.
    """
    record_id = record["record_id"]
    try:
        result = bedrock_lib.converse_with_chain(
            client, chain, record.get("system"), record["messages"],
            temperature=config["temperature"],
            thinking=config.get("thinking") or "disabled",
            effort=config.get("effort"),
            max_retries=DEFAULT_MAX_RETRIES,
            inference_region=inference_region)
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("record %s failed: %s", record_id, exc)
        return response_lib.failure_line(
            record_id=record_id,
            reason=f"{bedrock_lib.error_code(exc)}: {exc}",
            model_id=chain[0].model_id,
            # Rounds the failure actually cost. `BedrockError` carries them from the
            # chain walk; a non-Bedrock failure (an S3 error, say) cost none, and
            # counting it as one would inflate the bill rather than under-report it.
            rounds=getattr(exc, "rounds", 0))

    if result.refused:
        # Every model in the chain declined. A refusal is a failure of this record,
        # not of the request — and it is reported with the stopReason so a caller can
        # tell it apart from an error.
        return response_lib.failure_line(
            record_id=record_id,
            reason=f"refused: stopReason={result.stop_reason}",
            model_id=result.model_id, usage=result.usage,
            estimated_cost=result.estimated_cost, rounds=result.rounds)

    return response_lib.success_line(
        record_id=record_id, model_id=result.model_id,
        output_text=result.text, stop_reason=result.stop_reason,
        usage=result.usage, estimated_cost=result.estimated_cost,
        rounds=result.rounds, notes=result.notes)
