"""
StateSubmitBatch — one job, all N records, parked on a task token (README.md §5.2, §6.2).

**No lease is taken.** The account-level in-progress-job limit is service-enforced, so
a local cap would serialize submissions Bedrock is willing to accept in parallel — the
§1.1 asymmetry expressed in code, and the reason `Acquire lease` is on the sync branch
only.

The ordering is the load-bearing part, in this exact sequence (§6.2):

  1. Write the row `status=SUBMITTING` with the task token.
  2. `CreateModelInvocationJob`, job name = `{workflow_id}`.
  3. Success → `status=SUBMITTED`, `job_arn` set. Return nothing; the task stays parked.
  4. A **quota-shaped** refusal → `status=PENDING`. Also not an error: a drain submits
     it when capacity frees, and the task is still parked on the same token.

Row first, then Bedrock. Reversed, a Lambda death in between leaves a running job whose
completion event finds no row and no token — so the caller parks for its full 24 h with
the work already paid for and the answer sitting in S3.

Only quota-shaped exceptions queue. A `ValidationException` on a malformed record set
must fail the caller **immediately**: queueing a permanent error retries it forever,
invisibly, until the token times out. That classification lives in
`lib/bedrock.py:BATCH_QUOTA_EXCEPTIONS`, and `batch_client.create_job` applies it.
"""
from __future__ import annotations

import logging

from lib import batch_client, bedrock as bedrock_lib, s3_utils, token_table
from lib.custom_exceptions import BedrockBatchQuotaExceeded, ValidationException
from lib.input_schema import (
    MODEL_REGISTRY_KEY,
    PROXY_BUCKET,
    iter_parse_records,
    records_key,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    data = event["data"]
    config = data["config"]
    request = data["request"]
    output = data["output"]
    batch = data["batch"]
    workflow_id = data["workflow_id"]
    keys = data["keys"]

    task_token = (event.get("task_token") or "").strip()
    if not task_token:
        raise ValidationException(
            "task_token is required — this state must be invoked with "
            "'.waitForTaskToken'; without it the completion has nowhere to go and "
            "the execution hangs until its own timeout")

    if not batch.get("can_batch"):
        raise ValidationException(
            f"{config['model_chain'][0]} does not support batch inference")

    model_id = batch["batch_model_id"]
    input_s3uri = _write_batch_input(data, request, output, config, keys)

    # Step 1 — the row, BEFORE the call. `model_params` is what a drain needs to
    # resubmit without re-reading the plan.
    token_table.put_submitting(
        model_id=model_id, workflow_id=workflow_id, task_token=task_token,
        request_s3uri=request["s3uri"],
        model_params={
            "job_name": workflow_id,
            "model_id": model_id,
            "input_s3uri": input_s3uri,
            "output_s3uri": f"s3://{output['bucket']}/{keys['batch_output_prefix']}",
            "region": batch.get("batch_region") or "",
        })

    # Step 2.
    try:
        job_arn = batch_client.create_job(
            job_name=workflow_id, model_id=model_id, input_s3uri=input_s3uri,
            output_s3uri=f"s3://{output['bucket']}/{keys['batch_output_prefix']}",
            region=batch.get("batch_region") or None)
    except BedrockBatchQuotaExceeded as exc:
        # Step 4 — queued, and NOT an error to the caller. Returning normally is the
        # point: the token stays parked, the row is PENDING, and a drain owns it now.
        token_table.mark_pending(model_id, workflow_id, str(exc))
        logger.info("queued %s on %s: %s", workflow_id, model_id, exc)
        return {"status": token_table.STATUS_PENDING, "reason": str(exc)}
    except Exception:
        # Anything else — a ValidationException above all — propagates. The row is
        # left SUBMITTING rather than deleted: it is the only record that a submit was
        # attempted, it expires on its own TTL, and a PENDING row here would be
        # retried forever by the drain.
        logger.exception("submit failed permanently for %s on %s",
                         workflow_id, model_id)
        raise

    # Step 3.
    token_table.mark_submitted(model_id, workflow_id, job_arn)
    logger.info("submitted %s as %s", workflow_id, job_arn)
    return {"status": token_table.STATUS_SUBMITTED, "job_arn": job_arn}


def _write_batch_input(data: dict, request: dict, output: dict, config: dict,
                       keys: dict) -> str:
    """
    Render `request.jsonl` into a Bedrock batch input file and return its uri —
    STREAMED end to end (2026-09-05): the request set is read line by line and
    the batch input is written as multipart parts, so memory is bounded by one
    part, never by the record set. The whole-file form held three full copies
    at once (parsed records, shaped rows, the joined body) and a measured
    34-asset frame-image pool — 1,531 records, 668 MB request.jsonl — killed
    the Lambda with Runtime.OutOfMemory before the job existed.

    Every record still goes through `iter_parse_records` (the streamed
    parser — ONE home for the seen-set/ceiling/emptiness policy) and
    `build_converse_params` — the **same** function the sync loop calls — then
    `to_batch_record`. §12.1's "param shaping parity" holds by construction:
    one shaping function, no second code path to drift.
    """
    registry = bedrock_lib.load_registry(
        s3_utils.s3, PROXY_BUCKET, MODEL_REGISTRY_KEY)
    spec = bedrock_lib.build_chain(registry, config["model_chain"])[0]

    # The chain's fallbacks are dropped here, and that is a real limitation stated
    # rather than hidden: a batch job runs on exactly one model, so there is nothing to
    # fall back *to* within a job. A failed record comes back as a per-record error and
    # becomes a FAILED line (§5.2).
    if len(config["model_chain"]) > 1:
        logger.info(
            "batch runs on %s only; the %d fallback model(s) in the chain do not "
            "apply to a batch job", spec.model_id,
            len(config["model_chain"]) - 1)

    def shape(record: dict) -> dict:
        return bedrock_lib.to_batch_record(
            record["record_id"],
            bedrock_lib.build_converse_params(
                spec, record.get("system"), record["messages"],
                temperature=config["temperature"],
                thinking=config.get("thinking") or "disabled",
                effort=config.get("effort"),
                # Never for batch: there is no prompt cache, and the block is
                # rejected. `to_batch_record` strips it too — belt and braces, because
                # a rejected job is a 24 h park before anyone notices.
                # Batch follows the plan's routing scope, same as sync — the
                # job-level modelId comes from resolve_batch_model_id, but the
                # shaping stays identical so a repair pass re-running these
                # records through the sync chain shapes the same call.
                inference_region=config.get("inference_region") or None))

    src_key = request.get("records_key") or records_key(request["control_key"])
    # a mutable cell, not a generator-function attribute: an attribute set
    # at exhaustion is invisible until the consumer finishes, and reads as
    # shared state across calls — the cell is per-invocation by construction
    real_count = [0]

    def shaped_rows():
        """One shaped batch record per source record, then the pads —
        row OBJECTS; put_jsonl_streamed owns the byte convention.
        iter_parse_records owns seen-set/ceiling/emptiness (one home)."""
        count = 0
        for record in iter_parse_records(request["bucket"], src_key):
            count += 1
            yield shape(record)
        real_count[0] = count
        # padding once, after all real records — batch_client.iter_pads is
        # the ONE home for the floor rule; generated, never materialised.
        # spec-shaped, so pads honor the batch model's request contract
        yield from batch_client.iter_pads(count, spec)

    total = s3_utils.put_jsonl_streamed(
        output["bucket"], keys["batch_input"], shaped_rows())
    real = real_count[0]
    logger.info("wrote %d records (+%d padding) to s3://%s/%s",
                real, total - real, output["bucket"], keys["batch_input"])
    return f"s3://{output['bucket']}/{keys['batch_input']}"
