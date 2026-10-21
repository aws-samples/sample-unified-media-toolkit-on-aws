from __future__ import annotations

import os
import json
import time
import logging
import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

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


# NOT USED — no callers in lib/ or states/
def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def presign(bucket: str, key: str, expires: int = PRESIGN_EXPIRY) -> str:
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)


def head_size(bucket: str, key: str) -> int:
    """Returns ContentLength in bytes."""
    return s3.head_object(Bucket=bucket, Key=key).get("ContentLength", 0)


def get_json(bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


def get_binary(bucket: str, key: str) -> bytes:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def put_json(bucket: str, key: str, data: dict) -> None:
    body = json.dumps(data, indent=2, ensure_ascii=False).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def put_binary(bucket: str, key: str, data: bytes, mime: str) -> None:
    s3.put_object(Bucket=bucket, Key=key,
                   Body=data,
                   ContentType=mime)


def download_file(bucket, key, local_path) -> float:
    t0 = time.monotonic()
    s3.download_file(bucket, key, local_path)
    elapsed = time.monotonic() - t0
    mb = os.path.getsize(local_path) / (1024 ** 2)
    log.info("Proxy: %.1f MB  %.1fs", mb, elapsed)
    return elapsed


def upload_file(bucket: str, key: str, local_path: str, extra_args: dict | None = None) -> float:
    size_mb = os.path.getsize(local_path) / (1024 ** 2)
    log.info("Uploading %.1f MB → s3://%s/%s", size_mb, bucket, key)
    t0 = time.monotonic()
    s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args or {}, Config=_MULTIPART_CONFIG)
    elapsed = time.monotonic() - t0
    log.info(
        "Upload complete: %.1fs  (%.1f MB/s)",
        elapsed, size_mb / elapsed if elapsed > 0 else 0,
    )
    return elapsed


# NOT USED — no callers in lib/ or states/
def copy_object(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
    s3.copy_object(
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Bucket=dst_bucket,
        Key=dst_key,
    )


# NOT USED — no callers in lib/ or states/
def delete_object(bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)
