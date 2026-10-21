"""
Shared plumbing for the shot-metadata-enrichment states and gateway
(README.md §10 — conversation-analysis-on-aws's ca_common, adapted). Small on
purpose: S3 I/O against the proxy bucket, the direct gateway-Lambda invoke,
and the run prefix layout. Everything here is import-shared inside ONE flat
zip; nothing is copied to other repos.

Run prefix layout (everything under micro-workflow/shot-metadata-enrichment/
{workflow_id}/):

    _plan/plan.json            window plan + prompt snapshot + config
    _plan/runs.json            the bedrock-inference submit record
    _plan/window_results.json  per-window parse/validation state
    _records/window_NNN.json   per-window rendered record (Map checkpoint)
    records.jsonl              the staged rendered records (concatenated)
    shot_metadata.jsonl        the product         (label: shots)
    scenes.json                assembled scenes    (label: scenes)
    window_manifest.json       traceability        (label: window-manifest)
    usage_tracking.json        aggregated usage    (label: usage)
"""
from __future__ import annotations

import base64
import json
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

_s3 = boto3.client("s3")
# The sync gateway invokes must outlive the slowest sibling gateway:
# bedrock-inference's dispatcher runs up to its 300 s Timeout while
# streaming a large staged records file (its cfnstack GatewayTool entry).
# botocore's default 60 s read timeout would hang up mid-work and then
# auto-retry the invoke into a concurrent duplicate submit. One retry
# stays: a duplicated run_inference is absorbed (same records -> same
# workflow_id), and the quick get_prompt/get_model invokes keep their
# self-healing on transient throttles — the ASL retries transport
# errors, never a RuntimeError raised from a failed sibling call.
_lambda = boto3.client("lambda", config=Config(
    read_timeout=330, retries={"max_attempts": 2}))

PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "shot-metadata-enrichment")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")
PROMPT_REGISTRY_ARN = os.environ.get("PROMPT_REGISTRY_ARN", "")

PROMPT_CATEGORY = "shot-metadata-enrichment"

# The default model chain when the caller sets none (user decision
# 2026-08-19: sonnet-4-6 — the model the prompt was validated on, README §6).
# A VERSIONED CONSTANT, deliberately not an env var — bedrock-inference's
# DefaultModelChain lesson: the chain lands in plan.json and provenance,
# and a per-stack env could drift with nothing in the tree to show it.
# Validated through the same registry door as a caller's chain.
DEFAULT_MODEL_CHAIN = ["anthropic.claude-sonnet-4-6"]

# use_batch defaults TRUE here (user decision 2026-08-19), diverging from
# conversation-analysis's explicit-false: an hour of content is ~100-125
# windows of image-heavy records — the 50% batch discount is the normal
# case, and nothing downstream is interactive. A caller who needs sync
# latency passes use_batch=false explicitly.
DEFAULT_USE_BATCH = True

# §3 windowing configuration. Env-tunable per deployment, NEVER per request
# (per-request knobs would enter the workflow_id preimage). 16-shot target
# (user decision 2026-08-23, adopting the README §11 16-shot probes: coverage
# 16/16 and structural fields identical to the 10-shot reference on n=4
# windows, ~11% cheaper per shot, accepting the measured tail decay in the
# enumerative fields). Briefly 12 on 2026-10-16 to dodge the Claude 5
# family's truncating 4096 Converse output default; reverted same day —
# the root fix is bedrock-inference's registry-driven emit_max_tokens on
# the affected entries, and shrinking windows taxed every model ~20% more
# input to shield two. The 100-image budget matches the registry's raised
# Claude max_images; at 16 shots x 1-3 frames + context + gallery the shot
# target binds (<= 53 blocks), so the budget is a backstop, not the packer.
WINDOW_SHOT_TARGET = int(os.environ.get("WINDOW_SHOT_TARGET", "16"))
WINDOW_IMAGE_BUDGET = int(os.environ.get("WINDOW_IMAGE_BUDGET", "100"))
# Windows per Map iteration — the OOM-bounded unit (user decision
# 2026-08-19: Map + small thread pool).
WINDOWS_PER_BATCH = int(os.environ.get("WINDOWS_PER_BATCH", "10"))
RENDER_THREADS = int(os.environ.get("RENDER_THREADS", "3"))

# Frame tier (README §5): 1 frame under 4 s, 2 under 12 s, 3 at >= 12 s.
FRAME_TIER = ((4.0, 1), (12.0, 2), (float("inf"), 3))

# Burst-run folding: shot-segmentation publishes `burst_runs` — contiguous
# intervals of >= 6 consecutive sub-second shots (jingles, montages, promo
# stings; a 35 s jingle measures 94.8% burst shots). Every member still gets
# its own REQUIRED entry and its own output document, but only
# BURST_REPRESENTATIVES members per run (first / middle / last) carry frame
# images — the rest are frameless, and the prompt tells the model to judge
# them from the run's representatives. Measured on jingle.mp4: 55 member
# frames fold to 9. Set BURST_REPRESENTATIVES=0 to disable folding.
BURST_REPRESENTATIVES = int(os.environ.get("BURST_REPRESENTATIVES", "3"))

# Face gallery (README §5): 192x256 3:4 cells, max 5 faces per window.
GALLERY_CELL_W = 192
GALLERY_CELL_H = 256
GALLERY_MAX_FACES = 5
GALLERY_DIVIDER_PX = 3

# The checkpoint/re-enter deadline guard (face-recognition's pattern).
REMAINING_TIME_BUFFER_MS = 60_000


def run_prefix(workflow_id: str) -> str:
    return f"micro-workflow/{TARGET_NAME}/{workflow_id}"


def read_json(bucket: str, key: str):
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body.decode("utf-8"))


def read_json_if_exists(bucket: str, key: str):
    """None ONLY on a genuine missing key; everything else re-raises —
    bedrock-inference's checkpoint-read lesson: a broad except turns a
    throttled GET into 'no data yet' exactly when it hurts most."""
    try:
        return read_json(bucket, key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def read_bytes(bucket: str, key: str) -> bytes:
    return _s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def write_json(bucket: str, key: str, payload) -> None:
    _s3.put_object(Bucket=bucket, Key=key,
                   Body=json.dumps(payload, ensure_ascii=False,
                                   indent=1).encode("utf-8"),
                   ContentType="application/json")


def read_text(bucket: str, key: str) -> str:
    return _s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode(
        "utf-8")


def write_text(bucket: str, key: str, text: str,
               content_type: str = "application/jsonl") -> None:
    _s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"),
                   ContentType=content_type)


def list_keys(bucket: str, prefix: str) -> list:
    keys, token = [], None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = _s3.list_objects_v2(**kwargs)
        keys.extend(o["Key"] for o in resp.get("Contents") or [])
        token = resp.get("NextContinuationToken")
        if not token:
            return keys


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// uri: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 uri: {uri!r}")
    return bucket, key


def invoke_gateway_tool(function_arn: str, tool_name: str, event: dict):
    """Invoke a sibling's gateway-tool Lambda exactly as AgentCore would —
    the tool name rides the client context, the event is the flat tool
    parameters, and the submit brain (identity, absorption, validation)
    stays the sibling's. An `{error: true}` response is returned, not
    raised — the caller decides whether its context can surface a dict."""
    ctx = base64.b64encode(json.dumps(
        {"custom": {"bedrockAgentCoreToolName": tool_name}}
    ).encode("utf-8")).decode("ascii")
    resp = _lambda.invoke(FunctionName=function_arn, ClientContext=ctx,
                          Payload=json.dumps(event).encode("utf-8"))
    payload = json.loads(resp["Payload"].read().decode("utf-8"))
    if resp.get("FunctionError"):
        raise RuntimeError(
            f"{tool_name} raised: {json.dumps(payload)[:400]}")
    return payload
