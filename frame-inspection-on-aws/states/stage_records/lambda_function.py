"""
StateStageRecords: concatenate the unit checkpoints in plan order into ONE
pooled records.jsonl — streamed via S3 multipart, never held whole (the
records carry inline base64 images). Refuses on a missing checkpoint
(the Map claimed success it did not earn) — enrichment's rule.

A unit with ZERO records (its span selected no frames) is legal here and
becomes a gap at collect time.
"""
from __future__ import annotations

import json
import logging

import boto3

from lib import fi_common

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_PART_SIZE = 8 * 1024 * 1024


def lambda_handler(event: dict, context) -> dict:
    plan = fi_common.read_json(*fi_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    records_key = f"{prefix}/records.jsonl"

    s3 = boto3.client("s3")
    upload = s3.create_multipart_upload(
        Bucket=fi_common.PROXY_BUCKET, Key=records_key,
        ContentType="application/jsonl")
    parts, buffer, part_no, count = [], bytearray(), 1, 0
    try:
        for unit_index in range(len(plan["units"])):
            checkpoint = fi_common.read_json_if_exists(
                fi_common.PROXY_BUCKET,
                f"{prefix}/_records/unit_{unit_index:05d}.json")
            if checkpoint is None:
                raise RuntimeError(
                    f"unit {unit_index} has no checkpoint — the Map did not "
                    "finish; redrive Select frames")
            for record in checkpoint["records"]:
                buffer += (json.dumps(record, ensure_ascii=False) + "\n"
                           ).encode("utf-8")
                count += 1
            if len(buffer) >= _PART_SIZE:
                parts.append(_flush(s3, records_key, upload, part_no, buffer))
                buffer, part_no = bytearray(), part_no + 1
        if buffer or not parts:
            parts.append(_flush(s3, records_key, upload, part_no, buffer))
        s3.complete_multipart_upload(
            Bucket=fi_common.PROXY_BUCKET, Key=records_key,
            UploadId=upload["UploadId"],
            MultipartUpload={"Parts": parts})
    except Exception:
        s3.abort_multipart_upload(Bucket=fi_common.PROXY_BUCKET,
                                  Key=records_key,
                                  UploadId=upload["UploadId"])
        raise
    logger.info("staged %d record(s) from %d unit(s)", count,
                len(plan["units"]))
    return {"uuid": event["uuid"], "plan_s3uri": event["plan_s3uri"],
            "records_s3uri": f"s3://{fi_common.PROXY_BUCKET}/{records_key}",
            "record_count": count}


def _flush(s3, key: str, upload: dict, part_no: int, buffer: bytearray):
    part = s3.upload_part(
        Bucket=fi_common.PROXY_BUCKET, Key=key,
        UploadId=upload["UploadId"], PartNumber=part_no,
        Body=bytes(buffer))
    return {"PartNumber": part_no, "ETag": part["ETag"]}
