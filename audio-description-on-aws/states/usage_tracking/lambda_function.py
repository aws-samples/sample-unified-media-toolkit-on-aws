"""
StateUsageTracking: aggregate every bedrock-inference submission's own
summary (checkpointed submit records name the run prefixes) plus a count
of tts submissions. Family rules verbatim: SUCCESS PATH ONLY, NEVER
RAISES, and the producer's cost_is_complete flag is PROPAGATED — a
partial float is never reported as complete (the frame-inspection
code-review lesson, 2026-08-31).
"""
from __future__ import annotations

import logging
import time

from lib import ad_common as C
from lib import submit

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    try:
        return _track(event)
    except Exception as exc:                               # noqa: BLE001
        logger.error("USAGE TRACKING FAILED: %s — the run's outcome is "
                     "unaffected", exc)
        return {"uuid": event.get("uuid"), "usage": "unavailable"}


def _track(event: dict) -> dict:
    plan = C.read_json(*C.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]

    # loop checkpoints get overwritten as phases advance, so every
    # sibling submission is ALSO registered write-through (one row per
    # file — the fan-out Map registers shards concurrently). A run that
    # registered under the pre-fan-out SINGLE-FILE register and finishes
    # under this code (cross-deploy redrive) must not report $0-complete
    # (review 2026-09-02), so legacy rows are folded in:
    submits = submit.read_submits(prefix)
    seen = {(r.get("kind"), r.get("tag")) for r in submits}
    submits += [r for r in (C.read_json_if_exists(
        C.PROXY_BUCKET, f"{prefix}/_plan/submits.json") or [])
        if (r.get("kind"), r.get("tag")) not in seen]

    total_cost, complete = 0.0, True
    tokens_in = tokens_out = 0
    entries = []
    for run in submits:
        if run.get("kind") != "bedrock":
            continue
        summary = C.read_json_if_exists(
            *C.parse_s3_uri(run["output_s3_prefix"].rstrip("/")
                            + "/summary.json")) or {}
        usage = summary.get("bedrock_usage") or {}
        cost = usage.get("estimated_cost")
        if cost is None:
            complete = False
        else:
            total_cost += float(cost)
            if usage.get("cost_is_complete") is False:
                complete = False
        tokens_in += int(usage.get("input_tokens") or 0)
        tokens_out += int(usage.get("output_tokens") or 0)
        entries.append({"tag": run.get("tag"),
                        "workflow_id": run.get("workflow_id"),
                        "estimated_cost": cost})
    tts_count = sum(1 for r in submits if r.get("kind") == "tts")

    usage = {"workflow_id": plan["workflow_id"],
             "tool_version": plan["tool_version"],
             "written_at_ms": int(time.time() * 1000),
             "bedrock_runs": entries, "tts_submissions": tts_count,
             "total_estimated_cost": round(total_cost, 4),
             "cost_is_complete": complete,
             "total_input_tokens": tokens_in,
             "total_output_tokens": tokens_out}
    C.write_json(C.PROXY_BUCKET, f"{prefix}/usage_tracking.json", usage)
    return {"uuid": event["uuid"],
            "usage_s3uri": f"s3://{C.PROXY_BUCKET}/{prefix}"
                           "/usage_tracking.json"}
