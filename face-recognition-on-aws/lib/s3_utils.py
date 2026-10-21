from __future__ import annotations

import json

import boto3
from botocore.config import Config

# Single shared client. max_pool_connections sized for the threaded states
# (triage: up to 32 concurrent GETs; grid upload + Rekognition pools).
s3 = boto3.client("s3", config=Config(max_pool_connections=60))


def get_json(bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


def get_json_if_exists(bucket: str, key: str) -> dict | None:
    """
    get_json, but returns None for a missing key instead of raising —
    for checkpoint/resume reads where "not written yet" is expected.
    Any OTHER failure (throttling, access denied, corrupted JSON) still
    raises: silently treating an unrelated error as "no checkpoint" would
    restart already-billed work and mask the real problem. No blanket
    except Exception here — see README.md Gotchas ("silent-default bugs")
    for the real incidents this guards against.
    """
    from botocore.exceptions import ClientError
    try:
        return get_json(bucket, key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def put_json(bucket: str, key: str, data: dict) -> None:
    body = json.dumps(data, ensure_ascii=False).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def get_bytes(bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def put_bytes(bucket: str, key: str, body: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def list_keys(bucket: str, prefix: str) -> set[str]:
    """All object keys under a prefix (paginated)."""
    keys: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            keys.add(obj["Key"])
    return keys


def delete_keys(bucket: str, keys: list[str]) -> None:
    """Batch-delete objects (1000 per call, the S3 API cap)."""
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={
            "Objects": [{"Key": k} for k in keys[i:i + 1000]],
            "Quiet": True,
        })


def parse_s3uri(s3uri: str) -> tuple[str, str]:
    """s3://bucket/key/parts → (bucket, key). Raises ValueError on bad input."""
    if not s3uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {s3uri!r}")
    rest = s3uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 uri: {s3uri!r}")
    return bucket, key
