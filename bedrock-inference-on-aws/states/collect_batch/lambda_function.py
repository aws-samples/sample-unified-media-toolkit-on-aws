"""
StateCollectBatch — a batch job's output becomes `response.jsonl` (README.md §5.2).

Runs after the task token is resumed, so the job is terminal. Reads the job's output,
maps `recordId` back to `record_id`, and writes lines through `lib/response.py` — the
**same** builders the sync loop uses, which is what makes `response.jsonl` shape-identical
across modes (§12.1's "mode parity") and why a caller's parsing code never branches on
mode.

Two properties worth stating:

- **Any per-record failure becomes a `FAILED` line, not a workflow failure** (§5.2, §7).
  A batch job that reports `PartiallyCompleted` is a successful execution with per-record
  reasons in the summary.
- **The mapping is by `record_id`, not by position.** Bedrock does not guarantee output
  order, and it writes padding records interleaved with real ones. Zipping positionally
  would mis-pair every record after the first pad — §12.1 tests exactly this.

A record present in the request but **absent** from the output gets a `FAILED` line here
rather than silently vanishing: the response must have one line per request record, or
`records_total` disagrees with the file and the summary's arithmetic — the caller's only
partial-success signal — becomes wrong.
"""
from __future__ import annotations

import logging

from lib import batch_client, bedrock as bedrock_lib
from lib import response as response_lib
from lib import s3_utils, token_table
from lib.custom_exceptions import (
    BatchRecordsIncompleteException,
    ValidationException,
)
from lib.input_schema import (
    MODEL_REGISTRY_KEY,
    PROXY_BUCKET,
    REPAIR_MAX_FRACTION,
    iter_parse_records,
    records_key,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    data = event["data"]
    request = data["request"]
    output = data["output"]
    config = data["config"]
    keys = data["keys"]
    batch = data["batch"]
    workflow_id = data["workflow_id"]

    job_arn = _job_arn(event, batch, workflow_id)
    output_s3uri = f"s3://{output['bucket']}/{keys['batch_output_prefix']}"

    manifest, rows = batch_client.read_job_output(job_arn, output_s3uri)
    by_id = {row["record_id"]: row for row in rows}
    logger.info("read %d output record(s) for %s; manifest=%s",
                len(rows), job_arn, bool(manifest))

    # The spec, for pricing. The batch job ran on the primary; `batch_model_id` carries
    # a rewritten geo prefix (§9's `resolve_batch_model_id`), so pricing is looked up
    # from the chain's own id rather than the job's.
    registry = bedrock_lib.load_registry(
        s3_utils.s3, PROXY_BUCKET, MODEL_REGISTRY_KEY)
    spec = bedrock_lib.build_chain(registry, config["model_chain"])[0]
    model_id = batch.get("batch_model_id") or spec.model_id

    # The request is re-read STREAMED (iter_parse_records — one home for
    # the parse policy) — parse_records would materialise every record's
    # image payloads a second time, post-spend. Each record shrinks to
    # its response line immediately; the lines list holds output text
    # only, never inputs.
    lines = [
        _to_line(record, by_id.get(record["record_id"]), model_id, spec)
        for record in iter_parse_records(
            request["bucket"],
            request.get("records_key") or records_key(request["control_key"]))]

    s3_utils.put_jsonl_streamed(output["bucket"], keys["response"], lines)

    succeeded = sum(1 for line in lines
                    if line.get("status") == response_lib.STATUS_SUCCESS)
    logger.info("%d/%d records succeeded from batch %s",
                succeeded, len(lines), job_arn)

    repair = _repair_decision(lines, config)
    if repair["needed"]:
        # The repair pass IS the existing sync loop: `run_inference` resumes from
        # `_progress.json` and skips whatever is `done`, so seeding the checkpoint
        # with the SUCCESSFUL batch lines makes it infer exactly the gaps — through
        # the normal chain walk, which is what restores chain-fallback semantics to
        # batch (a single-model job has nothing to fall back to; the repair does,
        # and the primary keeps first shot: a batch miss was transient flakiness,
        # not a model verdict, and even a refusal legitimately deserves the
        # primary's second look before the chain escalates it — refusals are not
        # deterministic). Truncated lines (stop_reason max_tokens) are SUCCESS and
        # therefore in `done`: re-running them truncates again; salvage is the
        # caller's parse_partial_output.
        done = {line["record_id"]: line for line in lines
                if line.get("status") == response_lib.STATUS_SUCCESS}
        s3_utils.put_json(output["bucket"], keys["progress"],
                          {"done": done, "records_total": len(lines)})
        logger.info("repair: %d candidate(s), checkpoint seeded with %d done",
                    repair["candidates"], len(done))
        # WRITE-THEN-RAISE, in that order and for the same reason `summarize` does it:
        # both S3 objects the repair depends on are already durable, so the raise is a
        # routing signal over committed state rather than a lost result. Raising first
        # would leave the repair with no checkpoint to resume from and the run with no
        # response.jsonl.
        #
        # The ASL Catch on this name routes to the sync child (§8). Nothing in this
        # return path survives a caught error — Step Functions replaces the state's
        # RESULT with the error and combines it with the state's raw INPUT — so the
        # counts below are deliberately NOT what downstream reads: `Summarize`
        # recomputes them from response.jsonl, and `records_repaired` comes from the
        # repair child's own `records_resumed` via the ASL ResultSelector.
        raise BatchRecordsIncompleteException(
            f"{repair['candidates']}/{len(lines)} record(s) incomplete from batch "
            f"{job_arn} — {repair['reason']}; response.jsonl written and the sync "
            f"checkpoint seeded with {len(done)} completed record(s)")

    return {
        "operation": "StateCollectBatch",
        "status": "COMPLETED",
        "uuid": event.get("uuid") or "",
        "input": event.get("input") or {},
        "data": {
            **data,
            "records_total": len(lines),
            "records_succeeded": succeeded,
            "batch": {**batch, "job_arn": job_arn},
            # **`repair` is NOT ours to write.** `$.data.repair` is the repair child's
            # ResultSelector output and only the ASL may set it — returning the decision
            # under that name would make `summarize` read a `needed: False` verdict as a
            # repair that happened, and every clean batch run would report a
            # `records_repaired` of its own successes. Reaching this return AT ALL means
            # no repair was needed, so the only thing worth carrying is WHY it was
            # refused when there were failures to refuse over.
            **({"repair_skipped": {"reason": repair["reason"],
                                   "candidates": repair["candidates"]}}
               if repair["candidates"] else {}),
        },
    }


def _repair_decision(lines: list, config: dict) -> dict:
    """
    Whether the sync repair tail runs (README.md §P8). Guards, both
    load-bearing:

    - BOUNDED: only when `0 < failed <= REPAIR_MAX_FRACTION x total`. All-failed
      is systemic — a shape or access defect — and a sync re-run would re-pay the
      whole job at on-demand rate to fail again; it falls through to Summarize,
      which raises BedrockInferenceFailed with the reasons, as ever.
    - ONCE: the ASL runs the repair exactly one time by construction — the Catch
      that routes to it is on THIS state, and the repair state's only exits are
      `Summarize` and its own degrade-to-`Summarize` Catch, so there is no edge back
      here. A record that fails the repair too stays a FAILED line.

    `options.batch_repair: false` is the caller's opt-out for cheap FAILED lines.
    """
    failed = sum(1 for line in lines
                 if line.get("status") != response_lib.STATUS_SUCCESS)
    total = len(lines)
    if not config.get("batch_repair", True):
        reason = "disabled by options.batch_repair"
        needed = False
    elif failed == 0:
        reason = "nothing to repair"
        needed = False
    elif failed > REPAIR_MAX_FRACTION * total:
        reason = (f"{failed}/{total} failed exceeds the "
                  f"{REPAIR_MAX_FRACTION:.0%} repair bound — systemic, not "
                  "per-record; failing loudly instead of re-paying at "
                  "on-demand rate")
        needed = False
    else:
        reason = f"{failed}/{total} failed — re-running through the sync chain"
        needed = True
    logger.info("repair decision: needed=%s (%s)", needed, reason)
    return {"needed": needed, "candidates": failed, "reason": reason}


def _job_arn(event: dict, batch: dict, workflow_id: str) -> str:
    """
    The job arn, from the resumed token's payload if present, else from the row.

    The row read is the fallback rather than the primary path because the notifier
    **deletes** the row after sending — so by the time this state runs the row is
    usually gone. That is why `NotifyFunction` puts `job_arn` in the task output at all.
    """
    resumed = event.get("batch_result") or event.get("taskResult") or {}
    job_arn = (resumed.get("job_arn") or batch.get("job_arn") or "").strip()
    if job_arn:
        return job_arn

    model_id = batch.get("batch_model_id") or ""
    row = token_table.get_row(model_id, workflow_id) if model_id else None
    job_arn = ((row or {}).get("job_arn") or "").strip()
    if not job_arn:
        raise ValidationException(
            f"no job_arn for {workflow_id}: not in the resumed task output and no row "
            f"under {model_id!r} — the batch output prefix cannot be located")
    return job_arn


def _to_line(record: dict, row: dict | None, model_id: str,
             spec: bedrock_lib.ModelSpec) -> dict:
    """
    One request record + its batch output line → one response line.

    `rounds=1` for every batch line: a batch job makes exactly one model call per
    record. There is no retry ladder and no chain fallback inside a job — a distinction
    the summary's `rounds` figure is meant to carry honestly rather than paper over.

    Costs are priced with `batch=True`, halving the registry's on-demand rate. An
    absent record gets `rounds=0` so it contributes nothing to either figure.
    """
    record_id = record["record_id"]

    if row is None:
        return response_lib.failure_line(
            record_id=record_id, model_id=model_id, rounds=0,
            estimated_cost=0.0,
            reason="no output record for this record_id in the batch job output")

    error = row.get("error")
    if error:
        code = (error.get("errorCode") if isinstance(error, dict) else "") or ""
        message = (error.get("errorMessage") if isinstance(error, dict)
                   else str(error)) or "unspecified batch record error"
        return response_lib.failure_line(
            record_id=record_id, model_id=model_id,
            # A rejected batch record produced no tokens, so it is priced at zero
            # rather than left unpriced — an unpriced line would inflate
            # `rounds_unpriced` and make a complete cost figure look partial.
            estimated_cost=0.0,
            reason=f"{code}: {message}" if code else str(message))

    # `model_output` is a CONVERSE response — `create_job` declares
    # `modelInvocationType: "Converse"`, so batch rows come back in the same
    # dialect the sync path reads (§5.4, resolved 2026-08-11). The dual-dialect
    # translators (`normalize_batch_usage`, `batch_stop_reason`) that used to sit
    # here died with the InvokeModel path; `extract_text` below is the same
    # reader the sync loop uses.
    model_output = row.get("model_output") or {}
    usage = dict(model_output.get("usage") or {})
    stop_reason = model_output.get("stopReason") or ""

    if stop_reason in bedrock_lib.REFUSAL_STOP_REASONS:
        # `content_filtered` / `guardrail_intervened`. Sync escalates these to the next
        # model; batch has no next model, so it is recorded against this record and the
        # rest of the request still completes.
        return response_lib.failure_line(
            record_id=record_id, model_id=model_id,
            reason=f"refused: stopReason={stop_reason}", usage=usage,
            estimated_cost=spec.estimate_cost(usage, batch=True))

    # same telling as the sync path: tokens the row's usage hid are priced
    # (reconstructed_cache_write inside estimate_cost) and must say so
    hidden = bedrock_lib.reconstructed_cache_write(usage)
    return response_lib.success_line(
        record_id=record_id, model_id=model_id,
        output_text=bedrock_lib.extract_text(model_output),
        stop_reason=stop_reason, usage=usage,
        estimated_cost=spec.estimate_cost(usage, batch=True),
        rounds=1,
        notes=[f"usage under-itemized: {hidden} input tokens counted only "
               f"in totalTokens — priced at the cache-write rate"]
        if hidden else None)
