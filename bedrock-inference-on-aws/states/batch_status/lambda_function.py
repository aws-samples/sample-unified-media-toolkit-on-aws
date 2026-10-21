"""
BatchStatusFunction — a Bedrock batch completion becomes a terminal row (README.md §6.4).

Triggered by the **prefix-scoped** EventBridge rule: `source: aws.bedrock`,
`detail-type: Batch Inference Job State Change`, scoped to this deployment's job-name
prefix. Contrast the *drain* rule, which is deliberately **unscoped** (§6.5) — two rules
with opposite scoping for opposite reasons: a foreign job's completion is legitimate news
that capacity freed, but it is never news about one of our tokens.

The live v5 rule (`deployment/media2cloud-backend-stack.yaml:2401`) has no scoping filter
at all, so it fires on every Bedrock batch job in the account regardless of origin —
while MediaConvert at :2289 filters `userMetadata.solutionUuid` and Transcribe at :2325
filters a job-name prefix. This handler is defensive about it anyway: an event whose arn
finds no row is logged and ignored, never an error.

**No `SendTask*` here.** This function writes a status and stops; the row's stream
record then triggers `NotifyFunction`, the only holder of that grant in the deployment
(§6.3). And no Bedrock call either — the event carries everything needed.
"""
from __future__ import annotations

import logging

from lib import token_table

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Statuses the rule subscribes to. `PartiallyCompleted` maps to COMPLETED in
# `BEDROCK_STATUS_MAP`: per-record failures become FAILED *lines*, not a failed job
# (§5.2, §7), so the caller gets its 39 results and a summary that says so.
_TERMINAL_EVENT_STATUSES = tuple(token_table.BEDROCK_STATUS_MAP)


def lambda_handler(event: dict, context) -> dict:
    detail = event.get("detail") or {}
    job_arn = (detail.get("batchJobArn") or detail.get("jobArn") or "").strip()
    raw_status = (detail.get("status") or "").strip()
    message = detail.get("message") or detail.get("statusMessage") or ""

    if not job_arn:
        logger.warning("aws.bedrock event with no batchJobArn: %s", detail)
        return {"handled": False, "reason": "no batchJobArn in the event detail"}

    status = token_table.BEDROCK_STATUS_MAP.get(raw_status)
    if status is None:
        # Scheduled, InProgress, Stopping — a state change this workflow does not act
        # on. Not an error; the rule may be broader than the map.
        logger.info("ignoring non-terminal Bedrock status %r for %s",
                    raw_status, job_arn)
        return {"handled": False, "reason": f"non-terminal status {raw_status!r}"}

    row = token_table.find_by_job_arn(job_arn)
    if row is None:
        # Either a foreign job (the v5 rule's unscoped-fire defect) or a row already
        # notified and deleted. Neither is this function's problem, and raising would
        # make EventBridge retry a lookup that will never succeed.
        logger.info("no row for %s — foreign job or already notified", job_arn)
        return {"handled": False, "reason": "no row for this job arn"}

    model_id = row.get("model_id") or token_table.model_id_from_pk(row.get("pk", ""))
    workflow_id = row.get("sk") or ""

    wrote = token_table.mark_terminal(
        model_id, workflow_id, status, job_arn=job_arn,
        message=f"{raw_status}: {message}" if message else raw_status)

    if not wrote:
        # The guard on `mark_terminal` refused: the row is already terminal, so this is
        # a redelivered event. Expected — EventBridge is at-least-once — and skipping
        # it avoids emitting a second stream record and a second SendTaskSuccess.
        logger.info("%s already terminal; skipping the redelivered %s event",
                    workflow_id, raw_status)

    return {
        "handled": True,
        "workflow_id": workflow_id,
        "model_id": model_id,
        "status": status,
        "bedrock_status": raw_status,
        "wrote": wrote,
    }
