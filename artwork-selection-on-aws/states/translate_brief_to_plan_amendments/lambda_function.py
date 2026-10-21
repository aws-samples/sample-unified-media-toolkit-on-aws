"""
TranslateBriefToPlanAmendments — HELD: the one editorial seat. The
free-text selection_brief goes to bedrock-inference (sync, one tiny
text record) to be translated into the CLOSED vocabulary: archetype
counts, must-include people, avoid notes, a one-line rerank emphasis.
FilterRankCandidateShots validates the amendments — the brief can only
re-weight among pre-qualified candidates, never reach eligibility.
Self-completed skips: empty brief, no bedrock gateway, prompts not
bound, submit failure — every path reaches the filter.
"""
from __future__ import annotations

import json
import logging
import os

import boto3

from lib import s3io
from lib import wf_common
from lib.profiles import ARCHETYPES

logger = logging.getLogger(__name__)

BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")

_lam = boto3.client("lambda")
_sfn = boto3.client("stepfunctions")


def lambda_handler(event: dict, context) -> dict:
    task_token = event["task_token"]
    plan = s3io.get_json(event["plan_s3uri"])

    def skip(reason: str) -> dict:
        logger.info("brief translation skipped: %s", reason)
        return wf_common.complete_task_token(
            _sfn, task_token, {"skipped": reason})

    brief = (plan.get("selection_brief") or "").strip()
    if not brief:
        return skip("no_brief")
    if not BEDROCK_GATEWAY_ARN:
        return skip("no_bedrock_gateway")
    system = (plan.get("prompts") or {}).get("artwork-brief")
    if not system:
        return skip("prompts_not_bound")

    ask = {
        "brief": brief,
        "archetype_vocabulary": sorted(ARCHETYPES),
        "current_archetype_counts": plan["archetype_counts"],
    }
    records = [{
        "record_id": "brief-000",
        "system": system,
        "messages": [{"role": "user", "content": [
            {"text": json.dumps(ask, ensure_ascii=False)}]}],
        "max_tokens": 1000,
    }]
    return wf_common.submit_or_park(
        _lam, BEDROCK_GATEWAY_ARN, "bedrock-inference", "run_inference",
        {"records": records,
         "model_chain": plan["model_chain"],
         "use_batch": plan["use_batch"],
         "idempotency_token": f"artwork-brief-{plan['uuid']}",
         "callback_task_token": task_token},
        skip)
