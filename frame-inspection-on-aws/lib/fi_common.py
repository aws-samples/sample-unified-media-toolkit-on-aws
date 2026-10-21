"""
Shared plumbing for the frame-inspection states and gateway — the
ca_common shape, small on purpose (README.md §5).

Run prefix layout (everything under micro-workflow/frame-inspection/
{workflow_id}/):

    _plan/plan.json            assets/requests/spans + prompt snapshots + config
    _plan/runs.json            the bedrock-inference submit record
    _records/unit_NNNNN.json   per-unit rendered-record checkpoints
    records.jsonl              the staged pooled records
    {asset_idx}/{product}      per-asset per-category products
                               (credits.json | titles.json | verdicts.json)
    inspection_manifest.json   traceability (spans, units, inference_runs)
    usage_tracking.json        aggregated usage
"""
from __future__ import annotations

import base64
import json
import os

import boto3
from botocore.exceptions import ClientError

_s3 = boto3.client("s3")
_lambda = boto3.client("lambda")

PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "frame-inspection")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")
PROMPT_REGISTRY_ARN = os.environ.get("PROMPT_REGISTRY_ARN", "")

# Map sizing. Env-tunable per deployment, NEVER per request (a per-request
# knob would enter the workflow_id preimage — enrichment's rule).
SPANS_PER_BATCH = int(os.environ.get("SPANS_PER_BATCH", "4"))
IMAGES_PER_RECORD = int(os.environ.get("IMAGES_PER_RECORD", "18"))

# Category table: VERSIONED CONSTANTS, deliberately not env (the
# DEFAULT_MODEL_CHAIN lesson — a default change must re-identify work,
# which a TOOL_VERSION bump does and an env drift does not). Each category
# names its auto-trigger, its selection knobs, whether spans carry a
# question, its product basename, and its REGISTRY category (the
# owner-prefixed fi-{task} taxonomy name — the request keeps the bare
# task name as the API vocabulary). Adding a category = one row here +
# one registry prompt + one schema check in collect_finalize.
CATEGORIES = {
    # change_distance is MEASURED, not guessed (fi-dev live validation,
    # mitj404 tail, 2026-08-29): DINO cosine distance within one static card
    # is <=0.026, while real card CUTS range 0.035-0.26 — three of seven
    # measured cuts sat under the first-guess 0.25 and were lost (three
    # principal cards skipped, caught only when the heartbeat landed on
    # them). 0.03 splits noise from cut with margin on both sides. Text
    # cards look alike to a semantic embedding; they do not look alike to
    # a reader.
    "credits-read": {
        "prompt_category": "fi-credits-read",
        "trigger": "nondiegetic_blocks",
        "change_distance": 0.03,
        "heartbeat_sec": 6.0,
        "needs_question": False,
        "product": "credits.json",
    },
    "title-read": {
        "prompt_category": "fi-title-read",
        "trigger": "unreadable_text_spans",
        "change_distance": 0.03,
        "heartbeat_sec": None,       # a super is one short shot; over moving
                                     # picture the gate keeps more frames,
                                     # which is acceptable at that length
        "needs_question": False,
        "product": "titles.json",
    },
    "claim-verify": {
        "prompt_category": "fi-claim-verify",
        "trigger": None,             # explicit spans ONLY, each with a question
        "change_distance": 0.15,     # the moment is the point: keep more
        "heartbeat_sec": 2.0,
        "needs_question": True,
        "product": "verdicts.json",
    },
}

# Per-span crop modes a caller may request on title-read spans —
# contract vocabulary, so it lives HERE (not in imaging, whose pyvips
# import must never be a precondition for validating a plan: the build
# gate runs on machines without libvips — review, 2026-09-05).
CROP_MODES = ("lower_third",)

DEFAULT_MODEL_CHAIN = ["anthropic.claude-sonnet-4-6"]   # image-capable
# Sync reads default to the cheaper Sonnet-5-led chain (on-demand $2/$10
# vs 4.6's $3/$15 per M tokens; equally image-capable per the registry —
# 100 images, same downscale envelope). Batch keeps 4.6: Sonnet 5 cannot
# batch (registry batch_job.can_support: false). Mirrors
# bedrock-inference's mode-aware chain-less default (user, 2026-09-23);
# mirrored HERE because this workflow always materializes the chain into
# its hashed plan and history, so it cannot ride the platform default.
DEFAULT_SYNC_MODEL_CHAIN = ["anthropic.claude-sonnet-5-5",
                            "anthropic.claude-sonnet-4-6"]


def default_model_chain(use_batch: bool) -> list:
    """A COPY per plan (a shared list would let one plan's mutation leak)."""
    return list(DEFAULT_MODEL_CHAIN if use_batch else DEFAULT_SYNC_MODEL_CHAIN)

# Runaway guard, versioned like CATEGORIES (a cost knob, not a tuning
# knob): a span over strobing/fast-cutting footage can exceed the change
# gate on nearly every archived frame, and every kept frame is staging +
# image tokens in the batch job. The cap is STRUCTURAL, not fitted — an
# order of magnitude above the largest real need measured so far (a full
# cast roll at the 6 s heartbeat is tens of frames) and ~13 records at
# the default IMAGES_PER_RECORD. Over the cap we keep an evenly spaced
# subset: coverage over density.
MAX_FRAMES_PER_SPAN = 240


def run_prefix(workflow_id: str) -> str:
    return f"micro-workflow/{TARGET_NAME}/{workflow_id}"


def read_json(bucket: str, key: str):
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body.decode("utf-8"))


def read_json_if_exists(bucket: str, key: str):
    """None ONLY on a genuine missing key; everything else re-raises."""
    try:
        return read_json(bucket, key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def write_json(bucket: str, key: str, payload) -> None:
    _s3.put_object(Bucket=bucket, Key=key,
                   Body=json.dumps(payload, ensure_ascii=False,
                                   indent=1).encode("utf-8"),
                   ContentType="application/json")


def read_bytes(bucket: str, key: str) -> bytes:
    return _s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// uri: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 uri: {uri!r}")
    return bucket, key


def list_keys(bucket: str, prefix: str) -> list:
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        page = _s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            return keys
        token = page["NextContinuationToken"]


def invoke_gateway_tool(function_arn: str, tool_name: str, event: dict):
    """Invoke a sibling's gateway-tool Lambda exactly as AgentCore would —
    the tool name rides the client context; the submit brain stays the
    sibling's. An `{error: true}` response is returned, not raised."""
    ctx = base64.b64encode(json.dumps(
        {"custom": {"bedrockAgentCoreToolName": tool_name}}
    ).encode("utf-8")).decode("ascii")
    resp = _lambda.invoke(FunctionName=function_arn, ClientContext=ctx,
                          Payload=json.dumps(event).encode("utf-8"))
    payload = json.loads(resp["Payload"].read().decode("utf-8"))
    if resp.get("FunctionError"):
        raise RuntimeError(f"{tool_name} raised: {json.dumps(payload)[:400]}")
    return payload
