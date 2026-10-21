"""
StateSubmitAndWait's Lambda half — invoked with `.waitForTaskToken`, so
this function's RETURN VALUE IS DISCARDED and the state's real output
arrives later as bedrock-inference's SendTaskSuccess payload.
conversation-analysis's shape verbatim (its PLAN §7-§8); the deltas here
are the defaults already resolved into plan.json (sonnet-4-6 chain,
use_batch TRUE — user decisions 2026-08-19).

- The bedrock submit happens INLINE (direct gateway-Lambda invoke) with
  `callback_task_token` passed through — this function sees the TRUE
  submit response, the only place absorbed-vs-fresh can be told apart,
  and the §7 history pointer is written exactly when a fresh run starts.
- Run bookkeeping goes to S3 (`_plan/runs.json`) — there is no state
  output to put it in.
- FAILURES RAISE; a returned error dict is invisible by construction.
- EVERY submit carries `idempotency_token = this run's own uuid`: a
  redriven submit re-derives the SAME bedrock workflow_id and absorbs;
  a caller rerun (fresh token → new uuid) re-renders byte-identical
  records under a NEW token and gets a genuinely fresh inference.
"""
from __future__ import annotations

import logging
import time

from lib import sme_common
from lib import workflow_history

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = sme_common.read_json(
        *sme_common.parse_s3_uri(event["plan_s3uri"]))

    submit = {
        "records_s3uri": event["records_s3uri"],
        "callback_task_token": event["task_token"],
        "idempotency_token": event["uuid"],
        "model_chain": plan["model_chain"],     # always concrete ids
        "use_batch": plan["use_batch"],         # always an explicit bool
        # NO max_tokens, by family decision: bedrock-inference never emits
        # an output cap, so each model's own maximum applies and the JSONL
        # cannot truncate by configuration.
    }

    response = sme_common.invoke_gateway_tool(
        sme_common.BEDROCK_GATEWAY_ARN, "bedrock-inference___run_inference",
        submit)
    if response.get("error"):
        raise RuntimeError(
            f"bedrock-inference refused the submit "
            f"({response.get('errorCode')}): {response.get('message')}")

    history_key = workflow_history.write_history_entry(
        sme_common._s3, sme_common.PROXY_BUCKET, response,
        scope=plan["scope"], category=plan["prompt"]["category"],
        prompt_version=plan["prompt"]["version"],
        model_chain=plan["model_chain"],
        submitted_at_ms=int(time.time() * 1000),
        idempotency_token=submit["idempotency_token"])

    runs_key = f"{plan['output_prefix']}/_plan/runs.json"
    runs = [{
        "workflow_id": response["workflow_id"],
        "job_id": response["job_id"],
        "output_s3_prefix": response["output_s3_prefix"],
        "records_s3uri": event["records_s3uri"],
        "record_count": event.get("record_count"),
        "absorbed": bool(response.get("message")),
        "history_key": history_key,
    }]
    sme_common.write_json(sme_common.PROXY_BUCKET, runs_key, runs)
    logger.info("submitted: bedrock %s (absorbed=%s, batch=%s) — parked on "
                "the task token", response["workflow_id"],
                bool(response.get("message")), plan["use_batch"])
    return {"submitted": True}     # discarded by waitForTaskToken
