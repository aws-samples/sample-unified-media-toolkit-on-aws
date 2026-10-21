"""
StateBatchDeleteFaces — resumable hard-delete sweep (README.md §19).

Serves BOTH batch_delete and delete_collection (identical mechanics; the
work list differs — validate_index_batch resolved it to concrete records).

Per record: DeleteFaces (Rekognition) in batches -> tombstone + DDB delete ->
S3 derivative delete. Order per README.md §19: the Rekognition faces stop
matching FIRST (right-to-erasure), then bookkeeping. For delete_collection,
the collection itself is deleted LAST, after every record is swept — a
dangling empty Rekognition collection is self-describing; DDB records
pointing at a deleted collection would poison batch_get_faces consumers.

Checkpoint/re-enter: progress persists to the work file every
_CHECKPOINT_EVERY records; all operations are idempotent (DeleteFaces on
gone ids is a no-op, tombstone Put overwrites, S3 delete of missing keys
succeeds), so re-processing a partially-swept chunk is safe.
"""

from __future__ import annotations

import logging

from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.face_store import delete_face_record
from lib.input_schema import (
    REKOGNITION_DELETE_FACES_MAX,
    REMAINING_TIME_BUFFER_MS,
)
from lib.rekognition_utils import (
    delete_collection_ignore_missing,
    delete_faces,
)
from lib.s3_utils import delete_keys, get_json, parse_s3uri, put_json

logger = logging.getLogger(__name__)

_CHECKPOINT_EVERY = 100


def lambda_handler(event: dict, context) -> dict:
    action = event["action"]
    cfg = event["input"]
    collection_id = cfg["collection_id"]
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    work_key = event["data"]["work_key"]

    work = get_json(out_bucket, work_key)
    records = work["records"]
    done: set[str] = set(work.get("done") or [])
    reason = ("collection_deleted" if action == "delete_collection"
              else "batch_delete")

    todo = [r for r in records if r["face_id"] not in done]
    logger.info("%s: %d records, %d remaining", action, len(records),
                len(todo))

    # ── 1. Rekognition DeleteFaces (batched, idempotent) ──────────────────────
    # Delete from the collection FIRST and in bulk — the faces stop matching
    # immediately even if bookkeeping below is interrupted and resumed.
    remaining_ids = [r["face_id"] for r in todo]
    for i in range(0, len(remaining_ids), REKOGNITION_DELETE_FACES_MAX):
        delete_faces(collection_id,
                     remaining_ids[i:i + REKOGNITION_DELETE_FACES_MAX])

    # ── 2. Per-record bookkeeping: tombstone + DDB delete + S3 derivatives ────
    since_checkpoint = 0
    for rec in todo:
        if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
            work["done"] = sorted(done)
            put_json(out_bucket, work_key, work)
            raise LambdaResourceTimeoutException(
                f"{action} checkpointed at {len(done)}/{len(records)}")

        delete_face_record(rec, reason)
        keys = [rec[k] for k in ("thumbnail_key", "full_image_key")
                if rec.get(k)]
        if keys:
            delete_keys(out_bucket, keys)
        done.add(rec["face_id"])
        since_checkpoint += 1
        if since_checkpoint >= _CHECKPOINT_EVERY:
            work["done"] = sorted(done)
            put_json(out_bucket, work_key, work)
            since_checkpoint = 0

    work["done"] = sorted(done)
    put_json(out_bucket, work_key, work)

    # ── 3. delete_collection only: remove the Rekognition collection LAST ─────
    if action == "delete_collection":
        delete_collection_ignore_missing(collection_id)
        logger.info("collection %s deleted", collection_id)

    event["data"]["num_deleted"] = len(done)
    report_key = f"{out_prefix.rstrip('/')}/report.json"
    put_json(out_bucket, report_key, {
        "action": action,
        "collection_id": collection_id,
        "num_deleted": len(done),
        "face_ids": sorted(done),
    })
    event["data"]["report_key"] = report_key
    return event
