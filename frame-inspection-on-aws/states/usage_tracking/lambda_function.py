"""
StateUsageTracking: the aggregated `usage` label — sums the (single,
pooled) bedrock-inference run's own summary with the cost_is_complete
discipline. Family rules verbatim: SUCCESS PATH ONLY, and NEVER RAISES —
as the terminal state, an exception here would fail an execution whose
products are already written.
"""
from __future__ import annotations

import logging
import time

from lib import fi_common

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    try:
        return _track(event)
    except Exception as exc:                                   # noqa: BLE001
        logger.error("USAGE TRACKING FAILED: %s — the run's outcome is "
                     "unaffected; the inference run's own usage_tracking"
                     ".json remains authoritative", exc)
        return {"uuid": event.get("uuid"), "usage": "unavailable"}


def _track(event: dict) -> dict:
    plan = fi_common.read_json(*fi_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    runs = fi_common.read_json(fi_common.PROXY_BUCKET,
                               f"{prefix}/_plan/runs.json")

    total_cost, cost_is_complete = 0.0, True
    tokens_in = tokens_out = 0
    entries = []
    for run in runs:
        summary = fi_common.read_json_if_exists(
            *fi_common.parse_s3_uri(
                run["output_s3_prefix"].rstrip("/") + "/summary.json")) or {}
        # bedrock-inference nests usage under `bedrock_usage` (measured on
        # the fi-dev validation run — the top-level keys the first draft
        # read do not exist and reported an honest-looking $0.00).
        usage = summary.get("bedrock_usage") or {}
        cost = usage.get("estimated_cost")
        if cost is None:
            cost_is_complete = False
        else:
            total_cost += float(cost)
            # the producer's OWN flag: bedrock-inference sums only PRICED
            # rounds into estimated_cost and sets cost_is_complete=false
            # when any round used an unpriced model — a partial float
            # must not be reported as complete (absent flag on older
            # summaries counts as complete).
            if usage.get("cost_is_complete") is False:
                cost_is_complete = False
        tokens_in += int(usage.get("input_tokens") or 0)
        tokens_out += int(usage.get("output_tokens") or 0)
        entries.append({"workflow_id": run["workflow_id"],
                        "record_count": run.get("record_count"),
                        "absorbed": run.get("absorbed"),
                        "estimated_cost": cost,
                        "usage_uri": run["output_s3_prefix"].rstrip("/")
                                     + "/usage_tracking.json"})

    usage = {
        "workflow_id": plan["workflow_id"],
        "tool_version": plan["tool_version"],
        "written_at_ms": int(time.time() * 1000),
        "inference_runs": entries,
        "total_estimated_cost": round(total_cost, 4),
        "cost_is_complete": cost_is_complete,
        "total_input_tokens": tokens_in,
        "total_output_tokens": tokens_out,
    }
    fi_common.write_json(fi_common.PROXY_BUCKET,
                         f"{prefix}/usage_tracking.json", usage)
    return {"uuid": event["uuid"], "usage_s3uri":
            f"s3://{fi_common.PROXY_BUCKET}/{prefix}/usage_tracking.json"}
