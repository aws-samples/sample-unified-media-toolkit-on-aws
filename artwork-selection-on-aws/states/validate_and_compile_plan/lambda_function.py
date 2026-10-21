"""
ValidateAndCompilePlan — everything that can refuse refuses HERE,
before any work: input shape, own-bucket rule, source artifact
existence, the provenance pin (candidate shot_ids and cast names are
only meaningful against the exact sibling runs they came from), and
flag normalization. Compiles the archetype plan (explicit archetypes
param, else the genre profile) and binds the two prompt-registry
categories. Persists _plan/plan.json; later states read it by URI.
"""
from __future__ import annotations

import logging
import os
import time

import boto3

from lib import s3io
from lib import wf_common
from lib.profiles import ARCHETYPES, profile_for

logger = logging.getLogger(__name__)

PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
PROMPT_REGISTRY_ARN = os.environ.get("PROMPT_REGISTRY_ARN", "")
BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "artwork-selection")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")

# input key -> the artifact whose absence makes the source unusable
_SOURCES = {
    "shot_segmentation_s3uri": "embeddings/shots.json",
    "shot_metadata_s3uri": "shot_metadata.jsonl",
    "cast_resolution_s3uri": "persons.json",
    "face_recognition_s3uri": "facemetadata.json",
    "program_structure_s3uri": "structure.json",
}
_REQUIRED = ("shot_segmentation_s3uri", "shot_metadata_s3uri")


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
                f"{key} must be a sibling run prefix on this "
                f"deployment's proxy bucket (under {own})")
        artifact_uri = f"{prefix}/{artifact}"
        if not s3io.exists(artifact_uri):
            raise ValueError(
                f"{key}: expected artifact not found — {artifact_uri}. "
                f"The sibling run may be incomplete or the prefix wrong")
        inputs[key] = prefix
        sibling_ids[key.removesuffix("_s3uri")] = prefix.rsplit("/", 1)[-1]

    for key in _REQUIRED:
        if key not in inputs:
            raise ValueError(f"{key} is required — it defines the "
                             f"candidate universe")

    flags = {k: inp.get(k, d) if inp.get(k) is not None else d
             for k, d in wf_common.DEFAULT_FLAGS.items()}

    # the archetype plan: explicit list wins, else the genre profile.
    # Names were validated at the gateway; an unknown name HERE means
    # the deployed vocabulary changed under a started run — refuse.
    wanted = flags["archetypes"] or list(profile_for(flags["genre"]))
    unknown = [a for a in wanted if a not in ARCHETYPES]
    if unknown:
        raise ValueError(f"unknown archetype(s): {unknown} — the "
                         f"closed vocabulary is {sorted(ARCHETYPES)}")
    archetype_counts = {a: int(flags["candidates_per_archetype"])
                        for a in wanted}

    plan = {
        "uuid": uuid,
        **flags,
        "archetype_counts": archetype_counts,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_version": TOOL_VERSION,
        "inputs": inputs,
        "sibling_workflow_ids": sibling_ids,
        "sources_present": sorted(k.removesuffix("_s3uri")
                                  for k in inputs),
    }

    # Bind the prompts, DEGRADE on failure: the held states already
    # skip with a named reason when prompts are absent — a hard refusal
    # here would fail every run against an unseeded registry.
    if PROMPT_REGISTRY_ARN and BEDROCK_GATEWAY_ARN:
        lam = boto3.client("lambda")
        prompts = {}
        for category in wf_common.PROMPT_CATEGORIES:
            try:
                resp = wf_common.invoke_gateway_tool(
                    lam, PROMPT_REGISTRY_ARN, "prompt-registry",
                    "get_prompt", {"category": category})
            except RuntimeError as exc:
                logger.warning("prompt bind failed (%s): %s — the "
                               "consuming state will skip", category, exc)
                prompts = None
                break
            if resp.get("error") or not resp.get("system"):
                logger.warning("prompt category %r unavailable: %s — "
                               "the consuming state will skip",
                               category, resp.get("message"))
                prompts = None
                break
            prompts[category] = resp["system"]
        if prompts:
            plan["prompts"] = prompts

    prefix = f"s3://{PROXY_BUCKET}/micro-workflow/{TARGET_NAME}/{uuid}"
    plan_s3uri = f"{prefix}/_plan/plan.json"
    s3io.put_json(plan_s3uri, plan)
    return {"uuid": uuid, "plan_s3uri": plan_s3uri}
