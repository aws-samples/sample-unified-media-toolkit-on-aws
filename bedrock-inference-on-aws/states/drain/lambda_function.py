"""
DrainFunction — submit PENDING batch jobs when capacity frees (README.md §6.4, §6.5).

`bedrock:*` and **no Step Functions**. A drained submit does not notify anyone: it writes
`status=SUBMITTED` and the job's own completion event drives the rest. That is what keeps
`states:SendTask*` confined to `NotifyFunction` (§6.3).

Three triggers, each covering a case the others miss:

1. **The table stream** — a REMOVE (notified or TTL-expired) or a status change on this
   deployment's own rows. Model id comes from the stream image.
2. **The unscoped `aws.bedrock` rule** — *any* batch completion in the account, including
   another stack's. This is §6.5's fix: the batch limit is per **account** while this
   stack is not a singleton, so if stack A fills the account's job limit, stack B's
   PENDING rows would drain only when one of *B's own* jobs completes — and B has nothing
   in flight, so no event ever arrives. Deadlock bounded only by TTL. A foreign
   completion is legitimate news that capacity freed.
3. **The 6-hourly scheduled sweep** — no event image, so no model id; `models_with_pending()`
   supplies the work list. Backstop for TTL expiry, manual deletion, and any missed
   event, and the thing that makes the queue self-healing rather than event-dependent.
   Six hours because it is the backstop, not the mechanism: its interval is only the
   worst-case latency of the missed-event path, and a stranded row still gets three
   more sweeps inside the 24 h token budget.

Why the drain is on the stream rather than inline in the EventBridge handler: if it were
inline and got throttled partway, it would need the *next* completion event to resume —
and if the completing job was the last one, nothing would ever drain (§6.4).

**Stop on the first quota exception.** v5's `return metrics` in `fetchAndStartJobs`. The
remainder stays PENDING for the next trigger; continuing would burn the whole queue
against a wall that is not going to move within this invocation.
"""
from __future__ import annotations

import logging

from lib import batch_client, token_table
from lib.custom_exceptions import BedrockBatchQuotaExceeded

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Per invocation, per model. A bound, not a target: a stream REMOVE re-triggers on
# every deletion — completions drive drainage, not the clock — so an unfinished queue
# is picked up on the next completion, with the 6-hourly sweep behind it.
MAX_SUBMITS_PER_MODEL = 25


def lambda_handler(event: dict, context) -> dict:
    models = _models_to_drain(event)
    if not models:
        return {"models": [], "submitted": 0, "reason": "nothing to drain"}

    results, submitted = {}, 0
    for model_id in models:
        count, stopped = _drain_model(model_id)
        results[model_id] = {"submitted": count, "stopped_on_quota": stopped}
        submitted += count

    logger.info("drained %d job(s) across %d model(s): %s",
                submitted, len(models), results)
    return {"models": sorted(models), "submitted": submitted, "detail": results}


def _models_to_drain(event: dict) -> list:
    """
    Which models this trigger implicates.

    A stream event drains **only the model in its image** — a completion on model A
    freed capacity that is, for quota purposes, accounted per model, and draining B on
    A's news would submit B's queue against a limit nothing told us about. A sweep or a
    foreign Bedrock completion carries no usable model, so both fall back to every
    model with a PENDING row.
    """
    records = event.get("Records") or []
    if records and records[0].get("eventSource") == "aws:dynamodb":
        models = set()
        for record in records:
            image = ("OldImage" if record.get("eventName") == "REMOVE"
                     else "NewImage")
            row = token_table.stream_row(record, image)
            model_id = row.get("model_id") or token_table.model_id_from_pk(
                row.get("pk", ""))
            if model_id:
                models.add(model_id)
        return sorted(models)

    if event.get("source") == "aws.bedrock":
        # Unscoped by design: this may be a foreign job, and its arn is not in our
        # table. All we learn is "the account has capacity now" — so drain everything
        # queued (§6.5 fix 1).
        logger.info("aws.bedrock completion (possibly foreign) — draining all models")
        return token_table.models_with_pending()

    # Scheduled sweep, or a manual invocation. Same answer (§6.5 fix 2).
    return token_table.models_with_pending()


def _drain_model(model_id: str) -> tuple[int, bool]:
    """
    Submit PENDING rows for one model, oldest first, until a quota wall or the bound.

    Returns `(submitted, stopped_on_quota)`.
    """
    rows = token_table.query_pending(model_id, limit=MAX_SUBMITS_PER_MODEL)
    if not rows:
        return 0, False

    submitted = 0
    for row in rows:
        workflow_id = row.get("sk") or ""
        params = row.get("model_params") or {}
        if not params.get("input_s3uri") or not params.get("output_s3uri"):
            # A row written by an older tool version, or a partial write. Left PENDING
            # to expire on its TTL rather than deleted: the task token on it is still
            # parked, and deleting the row would strand the caller with no record of
            # why.
            logger.error("row %s has no usable model_params; skipping", workflow_id)
            continue

        try:
            job_arn = batch_client.create_job(
                job_name=params.get("job_name") or workflow_id,
                model_id=params.get("model_id") or model_id,
                input_s3uri=params["input_s3uri"],
                output_s3uri=params["output_s3uri"],
                region=params.get("region") or None)
        except BedrockBatchQuotaExceeded as exc:
            # The wall. Stop — the remainder stays PENDING for the next trigger.
            logger.info("quota wall on %s after %d submit(s): %s",
                        model_id, submitted, exc)
            return submitted, True
        except Exception as exc:                                   # noqa: BLE001
            # A permanent failure for THIS row only. Marked FAILED so the stream tells
            # the notifier, which fails the parked caller with the reason — rather than
            # leaving it PENDING to be retried forever by every future drain (§6.2).
            logger.exception("permanent submit failure for %s", workflow_id)
            token_table.mark_terminal(
                model_id, workflow_id, token_table.STATUS_FAILED,
                message=f"drain submit failed permanently: {exc}")
            continue

        token_table.mark_submitted(model_id, workflow_id, job_arn)
        submitted += 1
        logger.info("drained %s → %s", workflow_id, job_arn)

    return submitted, False
