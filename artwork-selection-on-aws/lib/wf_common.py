"""
wf_common.py — the family's gateway-invocation and task-token plumbing
(proven copies from cast-resolution's pb_common) plus this workflow's
caller-flag defaults, ONE home.
"""
from __future__ import annotations

import base64
import json

# The default chain for BOTH inference rounds (brief translation and
# vision rerank) when the caller names none — Sonnet 4.6 primary,
# Nova 2 Lite backup (user decision, 2026-09-17). The stage-4 rubric
# was validated on a larger judge; callers wanting that exact judge
# pass model_chain themselves.
DEFAULT_MODEL_CHAIN = ["anthropic.claude-sonnet-4-6",
                       "amazon.nova-2-lite-v1:0"]

# ── caller flags, ONE home for the defaults ──────────────────────────
# The gateway normalizes these into the identity hash; the plan state
# writes them into the plan; consuming states read plan[flag] directly
# (a missing key is a bug, not a choice — never re-default downstream).
DEFAULT_FLAGS = {
    "selection_brief": "",   # editorial free text -> plan amendments
    "genre": "",             # selects the Layer-B profile row
    "archetypes": None,      # explicit subset; None = profile-driven
    "candidates_per_archetype": 4,   # the validated prototype's n
    "spoiler_cutoff": 0.75,  # positional reveal-avoidance cut
    "model_chain": DEFAULT_MODEL_CHAIN,  # both inference rounds
    "use_batch": False,      # sync by default: two single-record
                             # rounds; batch is the caller's economics
}

# The two prompt-registry INTERNAL categories this workflow binds.
PROMPT_CATEGORIES = ("artwork-rerank", "artwork-brief")


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


# benign SendTaskSuccess races: the token was already completed or the
# state moved on — the machine is where we wanted it.
TOKEN_GONE = ("TaskDoesNotExist", "TaskTimedOut", "InvalidToken")


def complete_task_token(sfn, task_token: str, payload: dict) -> dict:
    """Self-complete a held state's own token with `payload` as data;
    TOKEN_GONE races are benign, anything else raises."""
    from botocore.exceptions import ClientError
    try:
        sfn.send_task_success(taskToken=task_token,
                              output=json.dumps(payload))
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in TOKEN_GONE:
            raise
    return payload


def submit_or_park(lam, gateway_arn: str, target: str, tool: str,
                   payload: dict, skip):
    """Submit to a sibling gateway with a callback token and PARK, or
    degrade via `skip(reason)`. A response WITH a job_id means the
    service OWNS the token — park even if error text rides alongside;
    only a handle-less response is a refusal."""
    try:
        sub = invoke_gateway_tool(lam, gateway_arn, target, tool,
                                  payload)
    except Exception as exc:                 # noqa: BLE001 — degrade
        return skip(f"submit_failed: {str(exc)[:150]}")
    if not (isinstance(sub, dict) and sub.get("job_id")):
        msg = sub.get("message") if isinstance(sub, dict) else str(sub)
        return skip(f"submit_refused: {msg}")
    return {"parked": True, "job_id": sub["job_id"]}
