"""
StateSubmitAndWait's Lambda half — conversation-analysis's shape verbatim
(that module's docstring carries the full reasoning; it is load-bearing
and not repeated here): the bedrock submit happens INLINE so this function
sees the TRUE submit response (absorbed-vs-fresh), the history pointer is
written exactly when a fresh run starts, run bookkeeping goes to S3 (there
is no state output — waitForTaskToken discards the return), FAILURES
RAISE, and idempotency_token = this run's own uuid (redrive absorbs;
caller rerun re-identifies).

One frame-inspection wrinkle: a run has SEVERAL prompt categories, so one
history pointer is written PER CATEGORY under the same scope — each
category's prompt version travels with the runs it identified.
"""
from __future__ import annotations

import logging
import time

from lib import fi_common
from lib import workflow_history

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = fi_common.read_json(*fi_common.parse_s3_uri(event["plan_s3uri"]))

    submit = {
        "records_s3uri": event["records_s3uri"],
        "callback_task_token": event["task_token"],
        "idempotency_token": event["uuid"],
        "model_chain": plan["model_chain"],
        "use_batch": plan["use_batch"],
    }
    response = fi_common.invoke_gateway_tool(
        fi_common.BEDROCK_GATEWAY_ARN, "bedrock-inference___run_inference",
        submit)
    if response.get("error"):
        raise RuntimeError(
            f"bedrock-inference refused the submit "
            f"({response.get('errorCode')}): {response.get('message')}")

    history_keys = {}
    for category, artifact in plan["prompts"].items():
        history_keys[category] = workflow_history.write_history_entry(
            fi_common._s3, fi_common.PROXY_BUCKET, response,
            scope=plan["scope"], category=category,
            prompt_version=artifact.get("version", "default"),
            model_chain=plan["model_chain"],
            submitted_at_ms=int(time.time() * 1000),
            idempotency_token=submit["idempotency_token"])

    runs_key = f"{plan['output_prefix']}/_plan/runs.json"
    fi_common.write_json(fi_common.PROXY_BUCKET, runs_key, [{
        "workflow_id": response["workflow_id"],
        "job_id": response["job_id"],
        "output_s3_prefix": response["output_s3_prefix"],
        "records_s3uri": event["records_s3uri"],
        "record_count": event.get("record_count"),
        "absorbed": bool(response.get("message")),
        "history_keys": history_keys,
    }])
    logger.info("submitted: bedrock %s (absorbed=%s) — parked on the token",
                response["workflow_id"], bool(response.get("message")))
    return {"submitted": True}     # discarded by waitForTaskToken
