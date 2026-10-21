"""
workflow_history.py — the §7 inference-history entry, written by the CALLER.

**COPIED MODULE — source of truth: prompt-registry-on-aws/gateway_tool/.**
Same distribution discipline as `prompt_render.py` (byte-parity + a shape test
in the bundle suite). This module exists so "histories are the caller's
responsibility" (prompt-registry-on-aws/README.md Design record, decision 5) does not decay into nine
private layouts — the layout IS the query interface, and a stateless agent
rediscovers last week's runs by listing one prefix.

The key:

    micro-workflow/histories/{scope}/{category}/{prompt_version}/
        {submitted_at_ms}-{workflow_id}.json

  * `scope` is the caller-chosen grouping key — an asset id where one exists,
    a project/collection name for ad-hoc work. Charset [A-Za-z0-9_.-]{1,64}
    (the execution-name charset); anything else raises.
  * Timestamp-first filename keeps v5's list-in-chronological-order property;
    the workflow_id suffix is the join back to the run.

The record is a POINTER, not a copy (~300 bytes). Outcome and usage are
deliberately NOT in it — the reader follows `output_s3_prefix/summary.json`,
which exists exactly once and is authoritative. v5's history records duplicate
response summaries, which is how a record and the truth learn to disagree.

Written ONCE, at submit — every field is known the moment `run_inference`
returns, so there is no completion hook and no second writer. And SKIPPED on
an absorbed resubmit: the submit response's `message` field is the absorption
signal (micro-workflow-bundle/README.md §C3), and writing anyway would manufacture a phantom second
run pointing at the same execution. That skip is why this function takes the
whole submit RESPONSE rather than picked-out fields — the decision belongs to
the module, or nine callers re-implement it and one forgets.

`submitted_at_ms` is REQUIRED and injected by the caller (no clock in here):
the timestamp is part of the key, golden-vector tests need it fixed, and the
caller may want its own submit-time rather than write-time.
"""
from __future__ import annotations

import json
import re

__all__ = ["write_history_entry", "history_key", "HistoryError"]

_SCOPE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_PREFIX = "micro-workflow/histories"


class HistoryError(ValueError):
    """The entry cannot be written as asked; the message says why."""


def history_key(*, scope: str, category: str, prompt_version: str,
                submitted_at_ms: int, workflow_id: str) -> str:
    for name, value in (("scope", scope), ("category", category),
                        ("prompt_version", prompt_version)):
        if not value or not _SCOPE_RE.match(value):
            raise HistoryError(
                f"{name} must match [A-Za-z0-9_.-]{{1,64}}, got {value!r}")
    if not workflow_id:
        raise HistoryError("workflow_id is required")
    return (f"{_PREFIX}/{scope}/{category}/{prompt_version}/"
            f"{int(submitted_at_ms)}-{workflow_id}.json")


def write_history_entry(s3_client, bucket: str, submit_response: dict, *,
                        scope: str, category: str, prompt_version: str,
                        model_chain: list, submitted_at_ms: int,
                        idempotency_token: str | None = None):
    """
    Write the pointer entry for one submit; returns the key, or None when the
    submit was ABSORBED into an existing run (`message` present) — the caller
    needs no branch of its own.
    """
    response = submit_response or {}
    if response.get("message"):
        return None
    workflow_id = response.get("workflow_id")
    if not workflow_id:
        raise HistoryError(
            "submit response carries no workflow_id — refusing to write an "
            "entry that cannot name its run")

    key = history_key(scope=scope, category=category,
                      prompt_version=prompt_version,
                      submitted_at_ms=submitted_at_ms,
                      workflow_id=workflow_id)
    record = {
        "workflow_id": workflow_id,
        "job_id": response.get("job_id"),
        "output_s3_prefix": response.get("output_s3_prefix"),
        "scope": scope,
        "prompt": {"category": category, "version": prompt_version},
        "model_chain": list(model_chain or []),
        "idempotency_token": idempotency_token,
        "submitted_at": int(submitted_at_ms),
    }
    s3_client.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(record, ensure_ascii=False,
                        separators=(",", ":")).encode("utf-8"),
        ContentType="application/json")
    return key
