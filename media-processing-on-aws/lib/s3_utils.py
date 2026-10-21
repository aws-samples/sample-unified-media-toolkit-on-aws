from __future__ import annotations

import json

import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

PRESIGN_EXPIRY = 1800  # 30 min

# Multipart threshold: 16 MB — JPEG/PNG frames (~100–500 KB) stay as single PUTs;
# .ts/.flac/.mp4/.wav files go through multipart.
# 64 MB parts: a 1.8 GB MP4 becomes ~29 parts instead of ~230 at the boto3 default 8 MB.
# max_concurrency=4: conservative for Lambda; avoids contending with ffmpeg threads.
_MULTIPART_CONFIG = TransferConfig(
    multipart_threshold=16 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=8,
)

s3 = boto3.client("s3", config=Config(max_pool_connections=60))


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def presign(bucket: str, key: str, expires: int = PRESIGN_EXPIRY) -> str:
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)


def put_json(bucket: str, key: str, data: dict) -> None:
    body = json.dumps(data, indent=2, ensure_ascii=False).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def get_json(bucket: str, key: str) -> dict:
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def head_object(bucket: str, key: str) -> dict:
    """Raw HeadObject response — ContentLength, StorageClass, ArchiveStatus, Restore."""
    return s3.head_object(Bucket=bucket, Key=key)


# Cold tiers: GetObject fails with InvalidObjectState until a restore completes.
# GLACIER_IR is deliberately absent — it reads like STANDARD, no restore needed.
_UNREADABLE_STORAGE_CLASSES = frozenset(("GLACIER", "DEEP_ARCHIVE"))


def archive_tier(head: dict) -> str | None:
    """
    Name of the cold tier holding this object, or None if it can be read now.

    Intelligent-Tiering keeps StorageClass at INTELLIGENT_TIERING and signals the
    archive tiers through ArchiveStatus instead, so both fields have to be checked.
    A completed restore (ongoing-request="false") clears the block; one still
    running does not — callers distinguish the two on head["Restore"], since
    "wait" and "start a restore" are different instructions.
    """
    if 'ongoing-request="false"' in (head.get("Restore") or ""):
        return None                              # restored copy available

    tier = head.get("ArchiveStatus")
    if not tier and head.get("StorageClass") in _UNREADABLE_STORAGE_CLASSES:
        tier = head["StorageClass"]
    return tier or None


def upload_file(bucket: str, key: str, local_path: str, extra_args: dict | None = None) -> None:
    s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args or {}, Config=_MULTIPART_CONFIG)


def copy_object(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
    s3.copy_object(
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Bucket=dst_bucket,
        Key=dst_key,
    )


def delete_object(bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)
