"""
StateSubmitAndWait's Lambda half (README.md §7, §8) — invoked with
`.waitForTaskToken`, so this function's RETURN VALUE IS DISCARDED and the
state's real output arrives later as bedrock-inference's SendTaskSuccess
payload. Consequences, each load-bearing:

- The bedrock submit happens INLINE here (direct gateway-Lambda invoke,
  §8) with `callback_task_token` passed through — so this function sees
  the TRUE submit response, which is the only place absorbed-vs-fresh can
  be told apart. The waker's payload always carries a `message`, so the
  history write CANNOT happen after the wake: the copied
  `workflow_history` module reads `message` as "absorbed → skip", and the
  §7 pointer must be written exactly when a fresh run starts.
- Run bookkeeping goes to S3 (`_plan/runs.json`), not to state output —
  there is no state output to put it in.
- FAILURES RAISE. A returned error dict here is invisible by construction
  (the same rule bedrock's gateway applies to callback submits).

EVERY submit carries `idempotency_token = this run's own uuid` — one
token, two properties (settled 2026-08-18 when the in-machine retry loop
was dropped): a redriven submit state re-derives the SAME bedrock
workflow_id and absorbs (idempotent), while a caller rerun (fresh
conversation-analysis token → new uuid) re-renders byte-identical records
under a NEW bedrock token and gets a genuinely fresh inference — without
this, a rerun would absorb into the very run whose output the caller is
trying to escape.
"""
from __future__ import annotations

import logging
import time

from lib import ca_common
from lib import workflow_history

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = ca_common.read_json(
        *ca_common.parse_s3_uri(event["plan_s3uri"]))

    submit = {
        "records_s3uri": event["records_s3uri"],
        "callback_task_token": event["task_token"],
        "idempotency_token": event["uuid"],
    }
    submit["model_chain"] = plan["model_chain"]      # always concrete ids
    submit["use_batch"] = plan["use_batch"]          # always an explicit bool
    # NO max_tokens, by decision (2026-08-18): the knob exists NOWHERE
    # in the family any more — bedrock-inference never emits an output
    # cap, so each model's own maximum applies (128k for the Claude
    # families; 65,535 for nova-2-lite) and truncation is structurally
    # off the table for the segment format. A cap would be a deliberate
    # re-addition in that service, never a side door here.

    response = ca_common.invoke_gateway_tool(
        ca_common.BEDROCK_GATEWAY_ARN, "bedrock-inference___run_inference",
        submit)
    if response.get("error"):
        raise RuntimeError(
            f"bedrock-inference refused the submit "
            f"({response.get('errorCode')}): {response.get('message')}")

    history_key = workflow_history.write_history_entry(
        ca_common._s3, ca_common.PROXY_BUCKET, response,
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
    ca_common.write_json(ca_common.PROXY_BUCKET, runs_key, runs)
    logger.info("submitted: bedrock %s (absorbed=%s, history=%s) — parked "
                "on the task token", response["workflow_id"],
                bool(response.get("message")), history_key)
    return {"submitted": True}     # discarded by waitForTaskToken
