"""
StateStageRecords — the small state after the Map barrier: concatenate the
per-window `_records/window_NNN.json` checkpoints, IN WINDOW ORDER, into
the one `records.jsonl` bedrock-inference stages from.

STREAMED via S3 multipart (the user's OOM discipline, applied here too):
an image-bearing record is ~1-2 MB of base64, so a long asset's record set
is hundreds of MB — never held whole. Parts flush at 8 MB; one window is
in memory at a time.

Refuses on a missing window (a Map iteration that "succeeded" without its
checkpoint is a defect, never a smaller job) and on an image over-budget
(defence in depth: bedrock's validate_plan enforces the model registry's
limit; OUR budget is the design's, checked against the bytes actually
rendered).
"""
from __future__ import annotations

import json
import logging

from lib import sme_common

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_PART_BYTES = 8 * 1024 * 1024        # S3 multipart minimum is 5 MB


def lambda_handler(event: dict, context) -> dict:
    plan = sme_common.read_json(
        *sme_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    records_key = f"{prefix}/records.jsonl"

    s3 = sme_common._s3
    upload = s3.create_multipart_upload(
        Bucket=sme_common.PROXY_BUCKET, Key=records_key,
        ContentType="application/jsonl")
    upload_id = upload["UploadId"]
    parts, buffer, part_no = [], bytearray(), 1

    def _flush(force=False):
        nonlocal buffer, part_no
        if not buffer or (not force and len(buffer) < _PART_BYTES):
            return
        resp = s3.upload_part(
            Bucket=sme_common.PROXY_BUCKET, Key=records_key,
            UploadId=upload_id, PartNumber=part_no, Body=bytes(buffer))
        parts.append({"ETag": resp["ETag"], "PartNumber": part_no})
        part_no += 1
        buffer = bytearray()

    try:
        for window in plan["windows"]:
            key = f"{prefix}/_records/window_{window['index']:03d}.json"
            record = sme_common.read_json_if_exists(
                sme_common.PROXY_BUCKET, key)
            if record is None:
                raise RuntimeError(
                    f"window {window['index']} has no rendered record at "
                    f"{key} — a Map iteration completed without its "
                    "checkpoint; redrive the render Map")
            images = sum(
                1 for message in record.get("messages") or []
                for block in message.get("content") or []
                if isinstance(block, dict) and "image" in block)
            if images > plan["image_budget"]:
                raise RuntimeError(
                    f"window {window['index']} rendered {images} image "
                    f"blocks against the {plan['image_budget']} budget — "
                    "the planner and the renderer disagree; fix before "
                    "spending")
            buffer += json.dumps(record, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")
            buffer += b"\n"
            _flush()
        _flush(force=True)
        s3.complete_multipart_upload(
            Bucket=sme_common.PROXY_BUCKET, Key=records_key,
            UploadId=upload_id, MultipartUpload={"Parts": parts})
    except BaseException:
        s3.abort_multipart_upload(Bucket=sme_common.PROXY_BUCKET,
                                  Key=records_key, UploadId=upload_id)
        raise

    logger.info("staged %d record(s) at %s (%d part(s))",
                len(plan["windows"]), records_key, len(parts))
    return {
        "uuid": event["uuid"],
        "plan_s3uri": event["plan_s3uri"],
        "records_s3uri": f"s3://{sme_common.PROXY_BUCKET}/{records_key}",
        "record_count": len(plan["windows"]),
    }
