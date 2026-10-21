"""
StateValidateIndexBatch — first state of the index-management SM (README.md §19).

Dispatches on event["action"]:
  batch_index       — load + validate the manifest (the gateway already
                      normalized prefix-mode into a generated manifest),
                      resolve every line's source ETag (threaded HeadObject),
                      triage lines (skip / rename / index) against the
                      collection's existing records, partition the INDEX
                      work into worker iterators.
  batch_delete      — resolve face_ids | celebrity_name to concrete records.
  delete_collection — enumerate the collection's records for the sweep.

Triage semantics (README.md §19 — idempotency keyed on content, not name):
  object missing/unreadable       -> failed/head_error       (early report)
  ETag already indexed, same name -> skipped/already_indexed (early report)
  ETag already indexed, new name  -> rename via DDB now      (skipped/renamed)
  ETag duplicated within batch    -> skipped/duplicate_in_batch (first wins)
  else                            -> work item for the Map workers

Renames are DDB-only writes (no Rekognition TPS) so they are safe to apply
here, BEFORE the semaphore is acquired. Early-report entries are written to
S3; the finalize state merges them with worker results into the full
per-line report — every manifest line ends in exactly one report entry
(fail-hard: no silent drops).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import ClientError

from lib.face_store import TABLE_NAME, unmarshal, update_face_name
from lib.index_batch import parse_manifest
from lib.input_schema import (
    BATCH_TILES_PER_GRID,
    COLLECTION_INDEX_GSI,
)
from lib.s3_utils import get_bytes, parse_s3uri, put_json, s3

logger = logging.getLogger(__name__)

_ddb = boto3.client("dynamodb")
_rek = boto3.client("rekognition")

_MAX_ITERATORS = 10
_HEAD_THREADS = 16


def lambda_handler(event: dict, context) -> dict:
    action = event["action"]
    if action == "batch_index":
        return _validate_batch_index(event)
    if action == "batch_delete":
        return _resolve_batch_delete(event)
    if action == "delete_collection":
        return _enumerate_collection(event)
    raise ValueError(f"unknown action: {action!r}")


# ── batch_index ────────────────────────────────────────────────────────────────

def _validate_batch_index(event: dict) -> dict:
    cfg = event["input"]
    collection_id = cfg["collection_id"]
    manifest_uri = cfg["manifest_s3uri"]
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")

    # Normalize BEFORE any side effect: the Acquire state reads
    # $.input.max_concurrent_executions — a direct SFN caller omitting it
    # would otherwise fail with an opaque JSONPath error AFTER this state
    # already applied renames. Clamp to the same [1,50] the gateway enforces
    # (the semaphore max_holders ratchet is up-only and permanent).
    raw = cfg.get("max_concurrent_executions")
    mce = int(raw) if raw is not None else 5
    if not 1 <= mce <= 50:
        raise ValueError(
            f"input.max_concurrent_executions must be in [1, 50] (got {mce})")
    cfg["max_concurrent_executions"] = mce

    # The collection must already exist — batch indexing never creates one
    # implicitly (create_collection is its own explicit tool).
    _rek.describe_collection(CollectionId=collection_id)

    m_bucket, m_key = parse_s3uri(manifest_uri)
    entries = parse_manifest(get_bytes(m_bucket, m_key).decode("utf-8"))

    # ── Resolve every line's ETag (content identity) ──────────────────────────
    def _head(entry):
        b, k = parse_s3uri(entry["image_uri"])
        try:
            resp = s3.head_object(Bucket=b, Key=k)
            return entry, resp["ETag"].strip('"'), None
        except ClientError as exc:
            return entry, None, exc.response["Error"]["Code"]

    with ThreadPoolExecutor(max_workers=_HEAD_THREADS) as pool:
        headed = list(pool.map(_head, entries))

    # ── Existing records in this collection, by source ETag ──────────────────
    etag_to_record: dict[str, dict] = {}
    kwargs: dict = {
        "TableName": TABLE_NAME,
        "IndexName": COLLECTION_INDEX_GSI,
        "KeyConditionExpression": "collection_id = :c",
        "ExpressionAttributeValues": {":c": {"S": collection_id}},
    }
    # Only the three consumed fields are retained — full records would make
    # this pre-pass O(collection bytes) in memory for no reason.
    kwargs["ProjectionExpression"] = "face_id, source_etag, celeb"
    while True:
        resp = _ddb.query(**kwargs)
        for raw in resp.get("Items") or []:
            rec = unmarshal(raw)
            if rec.get("source_etag"):
                etag_to_record[rec["source_etag"]] = rec
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break
        kwargs["ExclusiveStartKey"] = lek

    # ── Triage ────────────────────────────────────────────────────────────────
    early_report: list[dict] = []
    work: list[dict] = []
    first_line_for_etag: dict[str, dict] = {}
    for entry, etag, err in headed:
        line = {"image_uri": entry["image_uri"],
                "celebrity_name": entry["celebrity_name"]}
        if err:
            early_report.append({**line, "status": "failed",
                                 "reason": f"head_error:{err}"})
            continue
        existing = etag_to_record.get(etag)
        if existing:
            if entry["celebrity_name"] != (existing.get("celeb") or ""):
                update_face_name(existing["face_id"], entry["celebrity_name"])
                early_report.append({**line, "status": "skipped",
                                     "reason": "renamed",
                                     "face_id": existing["face_id"]})
            else:
                early_report.append({**line, "status": "skipped",
                                     "reason": "already_indexed",
                                     "face_id": existing["face_id"]})
            continue
        if etag in first_line_for_etag:
            # Same content twice in one batch: DEFER the verdict to finalize
            # (worker outcome of the first line decides) — reporting
            # "skipped" now would be wrong if the first line later fails.
            early_report.append({**line, "status": "deferred",
                                 "reason": "duplicate_in_batch",
                                 "duplicate_of":
                                     first_line_for_etag[etag]["image_uri"]})
            continue
        first_line_for_etag[etag] = line
        work.append({**line, "source_etag": etag})

    # Fail-hard escape hatch: if EVERY line failed at HEAD (wrong bucket /
    # IAM boundary / typo'd prefix), a "SUCCEEDED with 0 indexed" execution
    # would mislead the caller — fail the run with the real reason instead.
    if entries and not work and early_report and all(
            e["status"] == "failed" for e in early_report):
        raise RuntimeError(
            f"all {len(early_report)} manifest lines failed at HeadObject "
            f"(first: {early_report[0]['reason']}) — likely an inaccessible "
            f"source bucket; the stack can only read "
            f"s3://{out_bucket}/")

    put_json(out_bucket, f"{out_prefix}/ephemeral/early_report.json",
             {"entries": early_report})

    # ── Partition index work into worker iterators ────────────────────────────
    per_iter = max(BATCH_TILES_PER_GRID,
                   -(-len(work) // _MAX_ITERATORS)) if work else 1  # ceil
    iterators = []
    for item_id, start in enumerate(range(0, len(work), per_iter)):
        chunk = work[start:start + per_iter]
        iter_key = f"{out_prefix}/ephemeral/index_batch_{item_id}.json"
        put_json(out_bucket, iter_key, {
            "entries": chunk,
            "results": {},          # image_uri -> report entry
            "usage": {},
        })
        iterators.append({"item_id": item_id, "key": iter_key,
                          "context": {
                              "out_bucket": out_bucket,
                              "output_s3uri": event["output"]["s3uri"],
                              "collection_id": collection_id,
                          }})
    event["data"]["iterators"] = iterators
    event["data"]["num_manifest_entries"] = len(entries)
    event["data"]["num_to_index"] = len(work)
    logger.info("batch_index validated: %d entries -> %d to index, "
                "%d early-report, %d iterators", len(entries), len(work),
                len(early_report), len(iterators))
    return event


# ── batch_delete ───────────────────────────────────────────────────────────────

def _resolve_batch_delete(event: dict) -> dict:
    cfg = event["input"]
    collection_id = cfg["collection_id"]
    face_ids = cfg.get("face_ids") or []
    celeb = (cfg.get("celebrity_name") or "").strip()
    if bool(face_ids) == bool(celeb):
        raise ValueError("exactly one of face_ids | celebrity_name required")

    records: list[dict] = []
    if celeb:
        kwargs: dict = {
            "TableName": TABLE_NAME,
            "IndexName": "celeb-index",
            "KeyConditionExpression": "celeb = :n",
            "FilterExpression": "collection_id = :c",
            "ExpressionAttributeValues": {":n": {"S": celeb},
                                          ":c": {"S": collection_id}},
        }
        while True:
            resp = _ddb.query(**kwargs)
            records.extend(unmarshal(i) for i in resp.get("Items") or [])
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                break
            kwargs["ExclusiveStartKey"] = lek
    else:
        from lib.face_store import batch_get_faces
        found = batch_get_faces(face_ids)
        missing = [fid for fid in face_ids if fid not in found]
        if missing:
            # Fail-hard: deleting a half-known list silently would leave the
            # caller believing all requested faces are gone.
            raise ValueError(f"{len(missing)} face_id(s) not found: "
                             f"{missing[:5]}")
        wrong = [fid for fid, r in found.items()
                 if r.get("collection_id") != collection_id]
        if wrong:
            raise ValueError(f"face_id(s) not in collection "
                             f"{collection_id!r}: {wrong[:5]}")
        records = list(found.values())

    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    work_key = f"{out_prefix.rstrip('/')}/ephemeral/delete_work.json"
    put_json(out_bucket, work_key, {"records": records})
    event["data"]["work_key"] = work_key
    event["data"]["num_records"] = len(records)
    logger.info("batch_delete resolved: %d records", len(records))
    return event


# ── delete_collection ──────────────────────────────────────────────────────────

def _enumerate_collection(event: dict) -> dict:
    cfg = event["input"]
    collection_id = cfg["collection_id"]

    records: list[dict] = []
    kwargs: dict = {
        "TableName": TABLE_NAME,
        "IndexName": COLLECTION_INDEX_GSI,
        "KeyConditionExpression": "collection_id = :c",
        "ExpressionAttributeValues": {":c": {"S": collection_id}},
    }
    while True:
        resp = _ddb.query(**kwargs)
        records.extend(unmarshal(i) for i in resp.get("Items") or [])
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break
        kwargs["ExclusiveStartKey"] = lek

    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    work_key = f"{out_prefix.rstrip('/')}/ephemeral/delete_work.json"
    put_json(out_bucket, work_key, {"records": records})
    event["data"]["work_key"] = work_key
    event["data"]["num_records"] = len(records)
    logger.info("delete_collection enumerated: %d records for %s",
                len(records), collection_id)
    return event
