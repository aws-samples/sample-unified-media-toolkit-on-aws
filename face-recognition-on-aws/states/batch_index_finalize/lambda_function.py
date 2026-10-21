"""
StateBatchIndexFinalize — merge early-report + worker results into the full
per-line report (README.md §19: every manifest line ends in EXACTLY one entry;
a line with no entry means lost work and FAILS the execution — fail-hard).

Report: {summary: {indexed, skipped, failed, by_reason}, entries: [...]}
written to {output}/report.json. The response surface (num_indexed etc.)
lands in event.data for the gateway's get_job_status.
"""

from __future__ import annotations

import logging
from collections import Counter

from lib.s3_utils import get_json, parse_s3uri, put_json

logger = logging.getLogger(__name__)


def lambda_handler(event: dict, context) -> dict:
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")

    entries: list[dict] = list(get_json(
        out_bucket, f"{out_prefix}/ephemeral/early_report.json")["entries"])
    usage_total: dict = {}
    for it in event["data"].get("iterators") or []:
        iterator = get_json(out_bucket, it["key"])
        entries.extend((iterator.get("results") or {}).values())
        for k, v in (iterator.get("usage") or {}).items():
            usage_total[k] = usage_total.get(k, 0) + int(v)

    # Resolve deferred duplicate-in-batch lines from the FIRST line's actual
    # worker outcome (validate could not know it yet): indexed -> skipped/
    # already_indexed pointing at the same face_id; failed -> failed too, so
    # the caller knows this content was NOT covered.
    by_uri = {e["image_uri"]: e for e in entries
              if e.get("status") != "deferred"}
    for e in entries:
        if e.get("status") != "deferred":
            continue
        primary = by_uri.get(e.get("duplicate_of") or "")
        if primary and primary["status"] == "indexed":
            e.update(status="skipped", reason="duplicate_in_batch",
                     face_id=primary.get("face_id"))
        else:
            e.update(status="failed",
                     reason="duplicate_of_failed_line" if primary
                     else "duplicate_of_missing_line")

    expected = event["data"]["num_manifest_entries"]
    if len(entries) != expected:
        raise RuntimeError(
            f"report incomplete: {len(entries)} entries for {expected} "
            f"manifest lines — a worker lost results (fail-hard, no silent "
            f"drops)")

    statuses = Counter(e["status"] for e in entries)
    reasons = Counter(f"{e['status']}:{e['reason']}"
                      for e in entries if e.get("reason"))
    report = {
        "collection_id": event["input"]["collection_id"],
        "manifest_s3uri": event["input"]["manifest_s3uri"],
        "summary": {
            "total": len(entries),
            "indexed": statuses.get("indexed", 0),
            "skipped": statuses.get("skipped", 0),
            "failed": statuses.get("failed", 0),
            "by_reason": dict(reasons),
        },
        "usage": usage_total,
        "entries": sorted(entries, key=lambda e: e["image_uri"]),
    }
    report_key = f"{out_prefix}/report.json"
    put_json(out_bucket, report_key, report)

    event["data"]["report_key"] = report_key
    event["data"]["num_indexed"] = statuses.get("indexed", 0)
    event["data"]["num_skipped"] = statuses.get("skipped", 0)
    event["data"]["num_failed"] = statuses.get("failed", 0)
    event["data"]["usage"] = usage_total
    event["data"]["iterators"] = None
    logger.info("batch index report: %s", report["summary"])
    return event
