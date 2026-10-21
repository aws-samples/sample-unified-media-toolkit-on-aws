"""
StateUsageTracking (README.md §7e): the aggregated `usage` label. The
caller's question is "what did this ANALYSIS cost", so the rollup sums
every chunk inference run's own usage/summary documents (the
chunk-manifest points at each run for the per-chunk breakdown) plus this
machine's own wall-clock. Family rules apply verbatim: SUCCESS PATH ONLY,
and NEVER RAISES — as the terminal state, an exception here would fail an
execution whose products are already written.
"""
from __future__ import annotations

import logging
import time

from lib import ca_common

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    try:
        return _track(event)
    except Exception as exc:                                   # noqa: BLE001
        logger.error("USAGE TRACKING FAILED: %s — the run's outcome is "
                     "unaffected; each inference run's own usage_tracking"
                     ".json remains authoritative", exc)
        return {"uuid": event.get("uuid"), "usage": "unavailable"}


def _track(event: dict) -> dict:
    plan = ca_common.read_json(
        *ca_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    runs = ca_common.read_json(
        ca_common.PROXY_BUCKET, f"{prefix}/_plan/runs.json")

    total_cost = 0.0
    cost_complete = True
    tokens_in = tokens_out = 0
    per_run = []
    for run in runs:
        _, run_key = ca_common.parse_s3_uri(run["output_s3_prefix"])
        summary = ca_common.read_json_if_exists(
            ca_common.PROXY_BUCKET, f"{run_key}/summary.json") or {}
        usage = summary.get("bedrock_usage") or {}
        cost = usage.get("estimated_cost")
        if cost is None or not usage.get("cost_is_complete", True):
            cost_complete = False
        if cost is not None:
            total_cost += float(cost)
        tokens_in += int(usage.get("input_tokens") or 0)
        tokens_out += int(usage.get("output_tokens") or 0)
        per_run.append({
            "workflow_id": run["workflow_id"],
            "inference_mode": summary.get("inference_mode"),
            "mode_reason": summary.get("mode_reason"),
            "records_succeeded": summary.get("records_succeeded"),
            "records_failed": summary.get("records_failed"),
            "models_used": summary.get("models_used"),
            "estimated_cost": cost,
            "usage_uri": f"{run['output_s3_prefix']}/usage_tracking.json",
        })

    doc = {
        "workflow_id": plan["workflow_id"],
        "tool_version": plan["tool_version"],
        "written_at_ms": int(time.time() * 1000),
        "inference_runs": per_run,
        "total_estimated_cost": round(total_cost, 8),
        "cost_is_complete": cost_complete,
        "total_input_tokens": tokens_in,
        "total_output_tokens": tokens_out,
    }
    ca_common.write_json(ca_common.PROXY_BUCKET,
                         f"{prefix}/usage_tracking.json", doc)
    logger.info("usage: %d run(s), est cost %.6f (complete=%s)",
                len(per_run), total_cost, cost_complete)
    return {"uuid": event["uuid"],
            "total_estimated_cost": doc["total_estimated_cost"]}
