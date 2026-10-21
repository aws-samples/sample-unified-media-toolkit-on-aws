"""
StateSummarize — write `summary.json`, and decide whether the run failed (README.md §7).

The last state on both branches, so it is the one place that sees the whole result set —
which is why the all-records-failed decision lives here and nowhere else.

**Partial success is success.** 39 of 40 is a successful execution with 39 results, one
`FAILED` line, and the reason in `summary.failures[]`. A caller decides whether that is
usable; failing here would turn a partially-useful result into no result.

**The one hard failure is every record failing.** That is not a partial result, it is no
result wearing one — an empty `response.jsonl` at a 0% success rate is indistinguishable
from a model that legitimately produced nothing. `BedrockInferenceFailed` names up to five
reasons in the message rather than a count, taken from
`conversation-analysis/states/aggregate_results/lambda_function.py:122-131`: "40 records
failed" sends a reader to CloudWatch, while the first reason usually identifies the cause.

**Called from BOTH children, never from the orchestrator** (README.md §M decision 1).
`build_summary` therefore has two call sites, and that is deliberate: each child ends at
its own `Summarize`, so either one is startable on its own given a plan. A single
summarize in the orchestrator would have made the children un-runnable alone.

**It is also the only state that can report the batch repair tail** (§8). `collect_batch`
signals a repairable gap by *raising*, so its return value is discarded by the `Catch`
that routes to the repair child — every count here is recomputed from `response.jsonl`,
which the repair child rewrote, and the repair's own numbers arrive on `$.data.repair`
through the ASL `ResultSelector`. That is why `mode_reason`'s `"+ sync repair"` append
lives here and not upstream, and why the arithmetic is `succeeded - resumed` rather than
anything `collect_batch` measured.

This also makes the last-writer-wins on `summary.json` safe: both the batch child's
Summarize and the repair child's own Summarize write the same key, and both are pure
functions of `response.jsonl` plus the plan. The batch child's runs last, and it is the
only one of the two that can see `$.data.repair`.

`quota_table` goes in the summary as the cheap split-brain detector, and the merge is
what made it worth reading. The table NAME used to enforce the singleton — one
`${ResourcePrefix}-service-quotas` per region, so a second stack failed at CREATE. Both
tables now carry this stack's own prefix, so two deployments hold two independent caps
against one real Bedrock quota and can jointly exceed it (README.md §M2.3). Nothing
prevents that any more; reporting the table name is what makes it visible in the output
rather than only in the throttling.
"""
from __future__ import annotations

import logging

from lib import response as response_lib
from lib import s3_utils, semaphore_client
from lib.custom_exceptions import BedrockInferenceFailed

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    data = event["data"]
    output = data["output"]
    keys = data["keys"]

    lines = s3_utils.get_jsonl(output["bucket"], keys["response"])

    # Both are written by the batch child's ASL and only by it: `$.data.repair` is the
    # repair child's ResultSelector output, `$.data.repair_error` the Catch that degrades
    # a failed repair into a normal summary (§8). Absent on every sync run and on any
    # batch run that needed no repair, so `mode_reason` is unchanged there.
    repair = data.get("repair") or {}
    repair_error = data.get("repair_error") or {}
    # `repair_skipped` is the opposite case and comes from `collect_batch`'s own return:
    # there WERE failed records and the repair was refused (systemic, over the bound, or
    # opted out). Reported because a caller staring at 3 failed lines is owed the reason
    # the tail did not run, and this is the only object it can read.
    repair_skipped = data.get("repair_skipped") or {}

    summary = response_lib.build_summary(
        workflow_id=data["workflow_id"],
        tool_version=data.get("tool_version") or "",
        inference_mode=data.get("mode") or "",
        mode_reason=_mode_reason(data, repair, repair_error, repair_skipped),
        lines=lines,
        # `models_used`, `rounds` and `rounds_unpriced` are derived from the lines
        # themselves rather than passed in: they must agree with `estimated_cost`, and
        # computing all four from one list in one pass is what guarantees they do.
        quota_table=_quota_table(data, repair),
        extra={
            "request_s3uri": data["request"]["s3uri"],
            "response_s3uri": f"s3://{output['bucket']}/{keys['response']}",
            "output_prefix": f"s3://{output['bucket']}/{output['prefix']}",
            "model_chain": list(data["config"]["model_chain"]),
            "records_resumed": _records_resumed(data, repair),
        })

    if repair:
        # What the repair actually RECOVERED, not what it attempted. `records_resumed`
        # is the repair child's own count of records it SKIPPED because the seeded
        # checkpoint already had them — i.e. the batch successes — so succeeded minus
        # resumed is exactly what the sync chain won back. Derived from the child's
        # number rather than from `collect_batch`'s candidate count because nothing
        # collect_batch returned survives the caught error that routes here (§8).
        summary["records_repaired"] = max(
            0, summary["records_succeeded"] - _records_resumed(data, repair))
        summary["repair_execution_arn"] = repair.get("execution_arn") or ""

    if repair_error:
        # The repair failed and was degraded to a normal summary rather than failing an
        # execution whose response.jsonl is already complete — repair-off would have
        # SUCCEEDED with the same FAILED lines, so repair-on must not do worse. The
        # reason goes in the summary because a caught error leaves no other trace a
        # caller can read.
        summary["repair_error"] = str(repair_error.get("Cause")
                                      or repair_error.get("Error") or repair_error)

    s3_utils.put_json(output["bucket"], keys["summary"], summary)
    logger.info("summary: %d/%d succeeded, %d round(s), est $%.4f%s",
                summary["records_succeeded"], summary["records_total"],
                summary["bedrock_usage"]["rounds"],
                summary["bedrock_usage"]["estimated_cost"],
                "" if summary["bedrock_usage"]["cost_is_complete"]
                else f" (partial: {summary['bedrock_usage']['rounds_unpriced']} "
                     "unpriced round(s))")

    if summary["records_total"] and not summary["records_succeeded"]:
        # Written first, raised second — the summary is on S3 either way, so a caller
        # debugging the failure has the per-record reasons even though the execution
        # failed. Raising before the write would leave nothing behind but a log group.
        raise BedrockInferenceFailed(
            response_lib.failure_message(
                [line for line in lines
                 if line.get("status") != response_lib.STATUS_SUCCESS]))

    return {
        "operation": "StateSummarize",
        "status": "COMPLETED",
        "uuid": event.get("uuid") or "",
        "input": event.get("input") or {},
        "data": {
            **data,
            "summary_s3uri": f"s3://{output['bucket']}/{keys['summary']}",
            "records_total": summary["records_total"],
            "records_succeeded": summary["records_succeeded"],
            "records_failed": summary["records_failed"],
        },
    }


def _records_resumed(data: dict, repair: dict) -> int:
    """
    Records this run did not re-pay for.

    On the sync path that is the checkpoint's own `done` set, carried on `$.data` by
    `run_inference`. On the batch repair path the repair child is a *separate*
    execution, so its count arrives on `$.data.repair` through the ASL ResultSelector
    and is the number of batch successes the seeded checkpoint made it skip.
    """
    if repair:
        return int(repair.get("records_resumed") or 0)
    return int(data.get("records_resumed") or 0)


def _mode_reason(data: dict, repair: dict, repair_error: dict,
                 repair_skipped: dict) -> str:
    """
    The plan's reason, plus what the repair tail did to it.

    Appended HERE rather than in `collect_batch`, which is where it used to live: that
    state's return value is discarded by the `Catch` that routes to the repair, so the
    only state that can report the repair having happened is the one that runs after it.
    The `repair_skipped` case is the one exception — nothing was caught, so that state's
    return DID survive, and it carries the reason the tail was refused.
    """
    reason = data.get("mode_reason") or ""
    if repair_error:
        return f"{reason} + sync repair (failed, degraded)".lstrip(" +")
    if repair:
        return f"{reason} + sync repair".lstrip(" +")
    if repair_skipped:
        return (f"{reason} + no sync repair "
                f"({repair_skipped.get('reason') or 'refused'})").lstrip(" +")
    return reason


def _quota_table(data: dict, repair: dict) -> str:
    """
    The quota table this run leased from, or `""` when nothing leased.

    Empty rather than the table name for batch, and that is the accurate answer: batch
    takes no lease (§5), so no table governed the run and naming one would imply a cap
    that was never consulted. **A batch run whose repair tail ran is the exception** —
    the repair IS the sync child, lease and all, so a table did govern part of that run
    and reporting `""` would hide it from the split-brain check.
    """
    if (data.get("mode") or "") != "sync" and not repair:
        return ""
    try:
        return semaphore_client.table_name()
    except Exception as exc:                                       # noqa: BLE001
        # A reporting field must not fail a completed run.
        logger.warning("could not resolve the quota table name: %s", exc)
        return ""
