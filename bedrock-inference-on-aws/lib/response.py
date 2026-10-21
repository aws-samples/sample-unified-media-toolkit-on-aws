"""
The response contract: one record line, one summary (README.md §4.2).

Both paths write through here, which is what makes `response.jsonl` byte-shape
identical whether a request ran sync or batch — the property §12.1's "mode parity"
test asserts, and the reason a caller's parsing code never branches on mode.

```json
{"record_id": "chunk_000", "status": "SUCCESS", "model_id": "...",
 "output_text": "...", "stop_reason": "end_turn",
 "usage": {"inputTokens": 0, "outputTokens": 0, "cacheReadInputTokens": 0,
           "cacheWriteInputTokens": 0, "totalTokens": 0},
 "estimated_cost": 0.0}
```
"""
from __future__ import annotations

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"

# Always present, always in this order, always numeric. A caller summing
# `usage.outputTokens` across a response file must not have to handle a missing key
# on the one record that failed.
USAGE_KEYS = ("inputTokens", "outputTokens", "cacheReadInputTokens",
              "cacheWriteInputTokens", "totalTokens")

MAX_REPORTED_FAILURES = 5


def normalize_usage(usage: dict | None) -> dict:
    usage = usage or {}
    normalized = {key: int(usage.get(key) or 0) for key in USAGE_KEYS}
    if not normalized["totalTokens"]:
        normalized["totalTokens"] = (normalized["inputTokens"]
                                    + normalized["outputTokens"])
    return normalized


def success_line(*, record_id: str, model_id: str, output_text: str,
                 stop_reason: str, usage: dict | None,
                 estimated_cost: float | None, rounds: int = 1,
                 notes: list | None = None) -> dict:
    line = {
        "record_id": record_id,
        "status": STATUS_SUCCESS,
        "model_id": model_id,
        "output_text": output_text or "",
        "stop_reason": stop_reason or "",
        "usage": normalize_usage(usage),
        # `None` is preserved rather than coerced to 0.0: an unpriced model and a
        # free call are different facts, and the summary counts the former as
        # `rounds_unpriced` (§4.2).
        "estimated_cost": estimated_cost,
        # Billable model calls behind this line — retries and chain fallbacks
        # included, so `summary.bedrock_usage.rounds` can exceed `records_total`.
        # Batch lines carry 1; there are no retries inside a batch job.
        "rounds": int(rounds),
    }
    if notes:
        line["notes"] = list(notes)
    return line


def failure_line(*, record_id: str, reason: str, model_id: str = "",
                 usage: dict | None = None,
                 estimated_cost: float | None = None,
                 rounds: int = 1) -> dict:
    """
    A failed record. `usage` and `rounds` are still reported: a record that failed
    after burning input tokens on two models cost real money, and dropping it
    under-reports the bill for exactly the runs that went worst.
    """
    return {
        "record_id": record_id,
        "status": STATUS_FAILED,
        "model_id": model_id,
        "output_text": "",
        "stop_reason": "",
        "reason": reason,
        "usage": normalize_usage(usage),
        "estimated_cost": estimated_cost,
        "rounds": int(rounds),
    }


def count_rounds(lines: list) -> tuple[int, int]:
    """
    `(rounds, rounds_unpriced)` from the response lines.

    Derived here rather than accumulated by the caller: `rounds_unpriced` is only
    meaningful next to `estimated_cost`, and the two are computed from the same list
    in the same pass so they cannot disagree. A line whose `estimated_cost` is `None`
    contributes its rounds to `rounds_unpriced` — that is the whole distinction
    `estimate_cost` returns `None` to preserve (§4.2).
    """
    rounds = unpriced = 0
    for line in lines:
        spent = int(line.get("rounds") or 1)
        rounds += spent
        if line.get("estimated_cost") is None:
            unpriced += spent
    return rounds, unpriced


def collect_models(lines: list) -> list:
    """
    Models that actually answered, in first-seen order.

    First-seen order rather than sorted: it puts the primary first on a clean run, so
    a reader sees at a glance whether the fallback fired at all. `model_chain`
    overstates — on a clean run the fallback never ran (§4.2).
    """
    models = []
    for line in lines:
        model_id = line.get("model_id") or ""
        if model_id and model_id not in models:
            models.append(model_id)
    return models


def build_summary(*, workflow_id: str, tool_version: str, inference_mode: str,
                  mode_reason: str, lines: list, models_used: list | None = None,
                  rounds: int | None = None, rounds_unpriced: int | None = None,
                  quota_table: str = "", extra: dict | None = None) -> dict:
    """
    `summary.json` (§4.2). Mirrors
    `conversation-analysis/states/aggregate_results/lambda_function.py`, which is the
    shape any extracted service has to honour because it is what its callers already
    report.

    Four fields exist for specific reasons:

    - **`models_used`** — what actually ran, not what was available. On a clean run
      the fallback never fires, so `model_chain` overstates. Spend reporting needs
      the former.
    - **`rounds_unpriced`** — rounds whose model had no pricing entry, reported
      *alongside* `estimated_cost` so a partial figure is never mistaken for the whole
      bill.
    - **`inference_mode`** — mode is excluded from `workflow_id` (§3), so the summary
      is the only place a caller learns which path ran.
    - **`quota_table`** — the cheap split-brain detector. Two runs reporting two table
      names is the only visible symptom of two quota stacks, and the singleton table
      *is* the cap.
    """
    succeeded = [line for line in lines if line.get("status") == STATUS_SUCCESS]
    failed = [line for line in lines if line.get("status") != STATUS_SUCCESS]

    derived_rounds, derived_unpriced = count_rounds(lines)
    rounds = derived_rounds if rounds is None else int(rounds)
    rounds_unpriced = (derived_unpriced if rounds_unpriced is None
                       else int(rounds_unpriced))
    models_used = collect_models(lines) if models_used is None else list(models_used)

    usage_total = {key: 0 for key in USAGE_KEYS}
    estimated_cost = 0.0
    for line in lines:
        for key in USAGE_KEYS:
            usage_total[key] += int((line.get("usage") or {}).get(key) or 0)
        if line.get("estimated_cost") is not None:
            estimated_cost += float(line["estimated_cost"])

    summary = {
        "workflow_id": workflow_id,
        "tool_version": tool_version,
        "inference_mode": inference_mode,
        "mode_reason": mode_reason,
        "models_used": list(models_used),
        "records_total": len(lines),
        "records_succeeded": len(succeeded),
        "records_failed": len(failed),
        "failures": [
            {"record_id": line.get("record_id"),
             "reason": line.get("reason") or "unknown"}
            for line in failed
        ],
        "bedrock_usage": {
            "input_tokens": usage_total["inputTokens"],
            "output_tokens": usage_total["outputTokens"],
            "cache_read_input_tokens": usage_total["cacheReadInputTokens"],
            "cache_write_input_tokens": usage_total["cacheWriteInputTokens"],
            "total_tokens": usage_total["totalTokens"],
            "rounds": rounds,
            "rounds_unpriced": rounds_unpriced,
            "estimated_cost": estimated_cost,
            # False whenever any round was unpriced. The flag to check before
            # quoting the figure — `estimate_cost` returns None rather than 0.0
            # precisely to keep this distinction alive.
            "cost_is_complete": rounds_unpriced == 0,
        },
        "quota_table": quota_table,
    }
    summary.update(extra or {})
    return summary


def failure_message(lines: list, limit: int = MAX_REPORTED_FAILURES) -> str:
    """
    The message for `BedrockInferenceFailed` when **every** record failed (§7).

    Surfaces up to the first five reasons rather than a count, taken from
    `conversation-analysis/states/aggregate_results/lambda_function.py:122-131`: "40
    records failed" sends a reader to CloudWatch, while the first reason usually
    identifies the cause outright.
    """
    reasons = [line.get("reason") or "unknown" for line in lines][:limit]
    suffix = "" if len(lines) <= limit else f" (+{len(lines) - limit} more)"
    return (f"every record failed ({len(lines)} of {len(lines)}): "
            + " | ".join(reasons) + suffix)
