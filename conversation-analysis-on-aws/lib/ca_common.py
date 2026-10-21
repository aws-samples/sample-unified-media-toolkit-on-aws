"""
Shared plumbing for the conversation-analysis states and gateway (README.md §7).

Small on purpose: S3 JSON/text I/O against the proxy bucket, the direct
gateway-Lambda invoke (§8 — `lambda:InvokeFunction` with the AgentCore
client-context header, the pattern live-proven 2026-08-17), and the run
prefix layout. Everything here is import-shared inside ONE flat zip;
nothing is copied to other repos.

Run prefix layout (everything under micro-workflow/conversation-analysis/
{workflow_id}/):

    _plan/plan.json           chunk plan + prompt artifact snapshot + config
    _plan/runs.json           the bedrock-inference submit record
    _plan/chunk_results.json  per-chunk parse/validation state
    records.jsonl             the staged rendered records
    conversation_analysis.json  the product        (label: analysis)
    chunk_manifest.json         traceability       (label: chunk-manifest)
    usage_tracking.json         aggregated usage   (label: usage)
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
TARGET_NAME = os.environ.get("TARGET_NAME", "conversation-analysis")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")
PROMPT_REGISTRY_ARN = os.environ.get("PROMPT_REGISTRY_ARN", "")

# §5 chunking configuration. Env-tunable per deployment, NEVER per request:
# per-request knobs would enter the workflow_id preimage and turn every
# operator experiment into a re-identification of finished work.
CHUNK_BUDGET_TOKENS = float(os.environ.get("CHUNK_BUDGET_TOKENS", "15000"))

PROMPT_CATEGORY = "conversation-analysis"

# The default model chain when the caller sets none (user decision
# 2026-09-03: sonnet-4-6, replacing 2026-08-18's nova-2-lite —
# discourse_facts made the task judgment-heavier and nova-2-lite was
# measured failing it: header-only output on a full-episode transcript
# under the first section draft, ONE lazy segment for 48 minutes even
# at baseline; sonnet on the same episode: 15 segments, 0 gaps, 18
# facts with 18/18 verbatim quotes, $0.13). A VERSIONED CONSTANT,
# deliberately not an env var — bedrock-inference's DefaultModelChain
# lesson: the chain lands in plan.json and provenance, and a per-stack
# env could drift with nothing in the tree to show it. As a constant it
# moves only with a TOOL_VERSION bump, which is already in the
# workflow_id hash — so a default change re-identifies work exactly
# when it should. Validated against the deployment's registry at plan
# time like any caller-supplied chain.
DEFAULT_MODEL_CHAIN = ["anthropic.claude-sonnet-4-6"]


def run_prefix(workflow_id: str) -> str:
    return f"micro-workflow/{TARGET_NAME}/{workflow_id}"


def read_json(bucket: str, key: str):
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body.decode("utf-8"))


def read_json_if_exists(bucket: str, key: str):
    """None ONLY on a genuine missing key; everything else re-raises —
    bedrock-inference's checkpoint-read lesson (§12.3 there): a broad except
    turns a throttled GET into 'no data yet' exactly when it hurts most."""
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


def read_text(bucket: str, key: str) -> str:
    return _s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode(
        "utf-8")


def write_text(bucket: str, key: str, text: str,
               content_type: str = "application/jsonl") -> None:
    _s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"),
                   ContentType=content_type)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// uri: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 uri: {uri!r}")
    return bucket, key


def invoke_gateway_tool(function_arn: str, tool_name: str, event: dict):
    """§8: invoke a sibling's gateway-tool Lambda exactly as AgentCore would
    — the tool name rides the client context, the event is the flat tool
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
