"""
s3_utils.py — shared S3 client and helpers.

Single boto3 S3 client shared across all callers in the same Lambda container.
max_pool_connections=60 avoids connection exhaustion under concurrent requests.
"""
from __future__ import annotations

import base64
import json
import logging

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

s3 = boto3.client("s3", config=Config(max_pool_connections=60))


def get_json(bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


def put_json(bucket: str, key: str, data: dict) -> None:
    body = json.dumps(data, indent=2, ensure_ascii=False).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def get_object_bytes(bucket: str, key: str) -> bytes:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def get_object_text(bucket: str, key: str, encoding: str = "utf-8") -> str:
    return get_object_bytes(bucket, key).decode(encoding)


def put_object(bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def delete_object(bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)


def object_size(bucket: str, key: str) -> int:
    return s3.head_object(Bucket=bucket, Key=key)["ContentLength"]


def object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            return False
        raise  # AccessDenied, NoSuchBucket, etc. — propagate, don't mask


def get_image_base64(bucket: str, key: str) -> str:
    """Download an S3 object and return it as a base64-encoded string."""
    resp = s3.get_object(Bucket=bucket, Key=key)
    return base64.b64encode(resp["Body"].read()).decode("ascii")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse 's3://bucket/key' → ('bucket', 'key'). Raises ValueError on bad input."""
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    rest = uri[5:]
    if "/" not in rest:
        raise ValueError(f"Invalid S3 URI — missing key: {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket:
        raise ValueError(f"Invalid S3 URI — empty bucket: {uri!r}")
    key = key.lstrip("/")
    if not key:
        # Every caller treats the result as an OBJECT key; passing "" to
        # S3 raises a client-side ParamValidationError (a BotoCoreError),
        # which transient-tolerant paths misclassify as a blip.
        raise ValueError(f"Invalid S3 URI — missing key: {uri!r}")
    return bucket, key
