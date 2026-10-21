"""
StateValidateAndPlan: everything that can refuse, refuses here — cheap by
construction (PSA spends no inference dollar anywhere, so this state
guards only against wasted Lambda seconds and, more importantly, against
producing structure from HALF an evidence set without saying so).

Mode selection (README.md §4): explicit `mode` param
wins; else CA's overall_content_type; else `narrative`. The chosen mode
and WHY land in the plan — evidence, not verdict.
"""
from __future__ import annotations

import logging

from lib import psa_common

logger = logging.getLogger()
logger.setLevel(logging.INFO)

NEWS_TYPES = {"News Program"}
MODES = ("auto", "narrative", "news")


class ProgramStructureValidationError(Exception):
    """Refusal; the message names what to fix."""


def lambda_handler(event: dict, context) -> dict:
    workflow_id = event["uuid"]
    request = event.get("input") or {}

    inputs = {}
    for name, required in (("shot_segmentation_s3uri", True),
                           ("shot_metadata_s3uri", True),
                           ("conversation_analysis_s3uri", False),
                           ("transcript_s3uri", False),
                           ("audiostem_s3uri", False)):
        uri = (request.get(name) or "").strip()
        if required and not uri:
            raise ProgramStructureValidationError(f"{name} is required")
        if uri:
            psa_common.parse_s3_uri(uri)      # malformed uris refuse here
            inputs[name] = uri.rstrip("/")

    mode = (request.get("mode") or "auto").strip()
    if mode not in MODES:
        raise ProgramStructureValidationError(
            f"mode must be one of {MODES}, got {mode!r}")
    mode_evidence = {"requested": mode}
    if mode == "auto":
        content_type = None
        if "conversation_analysis_s3uri" in inputs:
            ca = _read_run_file(inputs["conversation_analysis_s3uri"],
                                "conversation_analysis.json")
            content_type = (ca or {}).get("overall_content_type")
        mode = "news" if content_type in NEWS_TYPES else "narrative"
        mode_evidence["overall_content_type"] = content_type
    if mode == "news" and "conversation_analysis_s3uri" not in inputs:
        raise ProgramStructureValidationError(
            "news mode needs conversation_analysis_s3uri (the stories ARE "
            "CA's discourse segments; without them there is no rundown)")

    prefix = psa_common.run_prefix(workflow_id)
    plan = {"workflow_id": workflow_id,
            "tool_version": psa_common.TOOL_VERSION,
            "output_prefix": prefix,
            "inputs": inputs,
            "mode": mode,
            "mode_evidence": mode_evidence}
    plan_key = f"{prefix}/_plan/plan.json"
    psa_common.write_json(psa_common.PROXY_BUCKET, plan_key, plan)
    logger.info("mode=%s inputs=%s", mode, sorted(inputs))
    return {"uuid": workflow_id,
            "plan_s3uri": f"s3://{psa_common.PROXY_BUCKET}/{plan_key}"}


def _read_run_file(run_uri: str, basename: str):
    bucket, key = psa_common.parse_s3_uri(run_uri)
    if not key.endswith(basename):
        key = f"{key}/{basename}"
    return psa_common.read_json_if_exists(bucket, key)
