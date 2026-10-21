"""
ValidateSources — everything that can refuse refuses HERE, before any
work: input shape, own-bucket rule, source artifact existence, and the
provenance pin (README.md §1: SPEAKER_IDs are per-run labels, so the plan
records the exact sibling workflow_ids this binding will be valid
against). Persists _plan/plan.json; later states read the plan by URI.
"""
from __future__ import annotations

import logging
import os
import time

import boto3

from lib import pb_common
from lib import s3io

logger = logging.getLogger(__name__)

PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
PROMPT_REGISTRY_ARN = os.environ.get("PROMPT_REGISTRY_ARN", "")
# the three enrichment system prompts, migrated from in-repo lib/*.md
# to registry INTERNAL categories (user, 2026-09-08 — the admin surface
# gives superusers gated editability without a code deploy). Bound HERE,
# refuse-before-spend, exactly like audio-description binds its five.
PROMPT_CATEGORIES = ("cast-adjudicate", "cast-critic2", "cast-vocatives")
TARGET_NAME = os.environ.get("TARGET_NAME", "cast-resolution")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")

# input key -> the artifact whose absence makes the source unusable
_SOURCES = {
    "speech_diarization_s3uri": "transcripts/final_transcript.json",
    "face_recognition_s3uri": "facemetadata.json",
    "shot_segmentation_s3uri": "embeddings/shots.json",
    "shot_metadata_s3uri": "shot_metadata.jsonl",
}
_IDENTITY_SOURCES = ("speech_diarization_s3uri", "face_recognition_s3uri")


def lambda_handler(event: dict, context) -> dict:
    uuid = event["uuid"]
    inp = event.get("input") or {}

    inputs, sibling_ids = {}, {}
    for key, artifact in _SOURCES.items():
        prefix = (inp.get(key) or "").strip().rstrip("/")
        if not prefix:
            continue
        own = f"s3://{PROXY_BUCKET}/micro-workflow/"
        if not prefix.startswith(own):
            raise ValueError(
                f"{key} must be a sibling run prefix on this deployment's "
                f"proxy bucket (under {own})")
        artifact_uri = f"{prefix}/{artifact}"
        if not s3io.exists(artifact_uri):
            raise ValueError(
                f"{key}: expected artifact not found — {artifact_uri}. "
                f"The sibling run may be incomplete or the prefix wrong")
        inputs[key] = prefix
        sibling_ids[key.removesuffix("_s3uri")] = prefix.rsplit("/", 1)[-1]

    if not any(k in inputs for k in _IDENTITY_SOURCES):
        raise ValueError(
            "at least one identity source is required: "
            "speech_diarization_s3uri or face_recognition_s3uri")

    plan = {
        "uuid": uuid,
        # caller flags, normalized (absent == default == same identity;
        # pb_common.DEFAULT_FLAGS is the ONE home for the defaults).
        # use_batch prices the WITNESS reads — enrichment rounds are
        # tiny text records and always run sync (delta < $0.01)
        **{k: bool(inp.get(k, d))
           for k, d in pb_common.DEFAULT_FLAGS.items()},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_version": TOOL_VERSION,
        "inputs": inputs,
        # the provenance pin: remap tables are only meaningful joined to
        # these exact runs — a sibling re-run renumbers its local ids
        "sibling_workflow_ids": sibling_ids,
        "sources_present": sorted(k.removesuffix("_s3uri") for k in inputs),
    }
    # Enrichment is an ENHANCEMENT, never a dependency (the template's
    # own invariant): bind the prompts only when the enrichment rounds
    # can actually run, and DEGRADE on any binding failure — the
    # downstream states already skip with a named reason when prompts
    # are absent, which is where an operator sees it. A hard refusal
    # here would fail every run against an unseeded/rolled-back
    # registry, including runs that would never consume the prompts.
    if plan.get("llm_enrichment") and PROMPT_REGISTRY_ARN \
            and os.environ.get("BEDROCK_GATEWAY_ARN"):
        lam = boto3.client("lambda")
        prompts = {}
        for category in PROMPT_CATEGORIES:
            try:
                resp = pb_common.invoke_gateway_tool(
                    lam, PROMPT_REGISTRY_ARN, "prompt-registry",
                    "get_prompt", {"category": category})
            except RuntimeError as exc:
                logger.warning("prompt bind failed (%s): %s — "
                               "enrichment will skip", category, exc)
                prompts = None
                break
            if resp.get("error") or not resp.get("system"):
                logger.warning(
                    "prompt category %r unavailable (%s): %s — "
                    "enrichment will skip", category,
                    resp.get("errorCode"), resp.get("message"))
                prompts = None
                break
            prompts[category] = resp["system"]
        if prompts:
            plan["prompts"] = prompts

    prefix = f"s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{uuid}"
    plan_s3uri = f"{prefix}/_plan/plan.json"
    s3io.put_json(plan_s3uri, plan)
    return {"uuid": uuid, "plan_s3uri": plan_s3uri}
