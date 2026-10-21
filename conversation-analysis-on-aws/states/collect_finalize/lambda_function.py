"""
StateCollectAndFinalize (README.md §3-§6, §7e): one state, because nothing
branches between its halves (the in-machine coverage-retry loop was
deliberately dropped, 2026-08-18 — see README.md §7d). Reads the bedrock
run's response.jsonl, parses the JSONL output (§4), validates coverage per
chunk (§3), then merges in pure code (§5-§6) and writes the two products:

    conversation_analysis.json   label: analysis        THE product
    chunk_manifest.json          label: chunk-manifest  traceability

Partial coverage is reported HONESTLY and the run still SUCCEEDS: an
unresolved chunk's uncovered ids land in `segments_gap` (partial success
IS success). The rerun path is the CALLER's fresh idempotency_token — our
submit folds this run's own uuid into the bedrock token, so a fresh
caller token is a genuinely fresh inference, never an absorb into the
flawed run.

A terminally FAILED bedrock run raises
ConversationAnalysisInferenceFailed and propagates as itself — no Catch,
no Fail state.
"""
from __future__ import annotations

import json
import logging

from lib import ca_common
from lib import merge
from lib import output_parse
from lib import transcript as transcript_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ConversationAnalysisInferenceFailed(Exception):
    """The whole bedrock run terminally failed — nothing to salvage."""


def lambda_handler(event: dict, context) -> dict:
    plan = ca_common.read_json(
        *ca_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]

    status = (event.get("wait_result") or {}).get("status")
    if status != "SUCCEEDED":
        raise ConversationAnalysisInferenceFailed(
            f"bedrock-inference ended {status} — see s3://"
            f"{ca_common.PROXY_BUCKET}/{prefix}/_plan/runs.json and the "
            "run's summary.json for per-record reasons")

    results = _collect(plan, prefix)
    ca_common.write_json(ca_common.PROXY_BUCKET,
                         f"{prefix}/_plan/chunk_results.json", results)
    return _finalize(event, plan, prefix, results)


def _collect(plan: dict, prefix: str) -> dict:
    runs = ca_common.read_json(
        ca_common.PROXY_BUCKET, f"{prefix}/_plan/runs.json")
    run = runs[-1]
    _, run_prefix_key = ca_common.parse_s3_uri(run["output_s3_prefix"])
    response_lines = ca_common.read_text(
        ca_common.PROXY_BUCKET,
        f"{run_prefix_key}/response.jsonl").splitlines()

    results: dict = {}
    for raw in response_lines:
        line = json.loads(raw)
        record_id = line.get("record_id") or ""
        if not record_id.startswith("chunk-"):
            continue
        index = int(record_id.split("-")[1])
        chunk = plan["chunks"][index]
        if line.get("status") != "SUCCESS":
            # §7d: a FAILED line on a SUCCEEDED run is the documented
            # shape — this chunk's whole range becomes an honest gap.
            logger.warning("chunk %d FAILED in bedrock: %s",
                           index, line.get("error"))
            continue
        parsed = output_parse.parse_response(line.get("output_text") or "")
        coverage = list(range(chunk["start_id"], chunk["end_id"] + 1))
        verdict = output_parse.validate_coverage(
            parsed["segments"], coverage)
        if parsed["line_repairs"]:
            # A recovery must never be silent: a rising count here is the
            # prompt regression signal the strict parser used to give by
            # dropping the line outright.
            logger.warning("chunk %d: %d line(s) recovered by the salvage "
                           "ladder %s", index,
                           sum(parsed["line_repairs"].values()),
                           parsed["line_repairs"])
        results[str(index)] = {
            "header": parsed["header"],
            "segments": output_parse.normalize_segments(
                [s for s in parsed["segments"] if _in_coverage(s, chunk)]),
            "gaps": verdict["gaps"],
            "problems": verdict["problems"],
            "bad_lines": len(parsed["bad_lines"]),
            "line_repairs": parsed["line_repairs"],
            "ok": verdict["ok"] and parsed["header"] is not None,
        }
    return results


def _finalize(event: dict, plan: dict, prefix: str, results: dict) -> dict:
    runs = ca_common.read_json(
        ca_common.PROXY_BUCKET, f"{prefix}/_plan/runs.json")
    t_bucket, t_key = ca_common.parse_s3_uri(plan["transcript"]["s3uri"])
    tdoc = transcript_mod.load_transcript(
        ca_common.read_text(t_bucket, t_key),
        transcript_format=plan["transcript"]["source_format"])
    sentences = tdoc["sentences"]

    chunk_results = []
    for chunk in plan["chunks"]:
        r = results.get(str(chunk["index"]))
        if r is None:
            # Never answered: the whole range is a gap.
            chunk_results.append({"header": None, "segments": [],
                                  "gaps": [[chunk["start_id"],
                                            chunk["end_id"]]]})
        else:
            chunk_results.append({"header": r["header"],
                                  "segments": r["segments"],
                                  "gaps": r["gaps"]})

    analysis = merge.merge_chunks(chunk_results, sentences,
                                  plan["transcript"]["duration_sec"])
    n_ovl = merge.annotate_overlapped_attribution(
        analysis, sentences, tdoc.get("overlap_segments") or [])
    if n_ovl:
        logger.info("attribution ambiguity: %d discourse facts fall "
                    "in cross-talk spans", n_ovl)
    analysis["provenance"] = {
        "prompt": {"category": plan["prompt"]["category"],
                   "version": plan["prompt"]["version"],
                   "source_md5": plan["prompt"]["source_md5"]},
        "model_chain": plan["model_chain"],
        "transcript_s3uri": plan["transcript"]["s3uri"],
        "transcript_format": plan["transcript"]["source_format"],
        "chunk_count": len(plan["chunks"]),
        "history_scope": plan["scope"],
        "tool_version": plan["tool_version"],
    }
    ca_common.write_json(ca_common.PROXY_BUCKET,
                         f"{prefix}/conversation_analysis.json", analysis)

    manifest = {
        "workflow_id": plan["workflow_id"],
        "budget_tokens": plan["budget_tokens"],
        "chunks": [
            {**{k: c[k] for k in ("index", "start_id", "end_id",
                                  "cut_tier", "cut_gap_s", "est_tokens")},
             "context_start_pos": c["context_start_pos"],
             "resolved": bool((results.get(str(c["index"])) or {})
                              .get("ok"))}
            for c in plan["chunks"]],
        # POINTERS, the history discipline: raw responses stay at each
        # bedrock run's own prefix — this is the join, never a copy.
        "inference_runs": runs,
    }
    ca_common.write_json(ca_common.PROXY_BUCKET,
                         f"{prefix}/chunk_manifest.json", manifest)

    segments = len(analysis["audio_segments"])
    gaps = len(analysis["segments_gap"])
    logger.info("finalized: %d segment(s), %d gap range(s), %s",
                segments, gaps, analysis.get("overall_content_type"))
    return {
        "uuid": event["uuid"],
        "plan_s3uri": event["plan_s3uri"],
        "segments": segments,
        "segments_gap": gaps,
        "overall_content_type": analysis.get("overall_content_type"),
    }


def _in_coverage(seg, chunk) -> bool:
    try:
        return (int(seg["start_id"]) >= chunk["start_id"]
                and int(seg["end_id"]) <= chunk["end_id"])
    except (KeyError, TypeError, ValueError):
        return False
