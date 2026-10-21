"""
gateway.py — the family's sibling-gateway invocation (proven copy of
cast-resolution's pb_common.invoke_gateway_tool) plus the callback
feature probe (audio-description's idiom).
"""
from __future__ import annotations

import base64
import json


def invoke_gateway_tool(lam, function_arn: str, target: str, tool: str,
                        payload: dict) -> dict:
    """Invoke a sibling gateway Lambda directly (tool routing rides
    base64 ClientContext). Raises RuntimeError on FunctionError: an
    unhandled exception in the gateway returns HTTP 200 with an error
    body, and parsing that as a tool response silently misreads a
    crash as a refusal."""
    ctx = base64.b64encode(json.dumps(
        {"custom": {"bedrockAgentCoreToolName":
                    f"{target}___{tool}"}}).encode()).decode()
    r = lam.invoke(FunctionName=function_arn, ClientContext=ctx,
                   Payload=json.dumps(payload).encode())
    body = r["Payload"].read()
    if r.get("FunctionError"):
        raise RuntimeError(
            f"{target}___{tool} invocation failed "
            f"({r['FunctionError']}): "
            f"{body[:300].decode('utf-8', 'replace')}")
    return json.loads(body)


def require_callback_feature(lam, function_arn: str, target: str) -> None:
    """Refuse-before-spend: the deployment must complete callback
    tokens, or this workflow would sleep to its deadman."""
    resp = invoke_gateway_tool(lam, function_arn, target,
                               "describe_tool", {})
    if "callback_task_token" not in (resp.get("features") or []):
        raise RuntimeError(
            f"{target} deployment predates callback support "
            "(describe_tool.features lacks callback_task_token) — "
            "update the sibling stack before running this workflow")
