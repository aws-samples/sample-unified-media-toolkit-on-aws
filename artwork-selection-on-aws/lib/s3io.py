"""
s3io.py — S3 read/write on full s3:// uris; states pass URIs, never
bodies (256 KB state I/O). Family copy (cast-resolution's s3io) plus
the byte helpers this workflow needs for frame tars and JPEGs.
"""
from __future__ import annotations

import json

import boto3

_s3 = boto3.client("s3")


def split(uri: str) -> tuple[str, str]:
    return tuple(uri[5:].split("/", 1))


def put_json(uri: str, doc) -> str:
    bucket, key = split(uri)
    _s3.put_object(Bucket=bucket, Key=key,
                   Body=json.dumps(doc, ensure_ascii=False).encode(),
                   ContentType="application/json")
    return uri


def get_json(uri: str):
    bucket, key = split(uri)
    return json.loads(_s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def get_jsonl(uri: str) -> list:
    bucket, key = split(uri)
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return [json.loads(ln) for ln in body.decode("utf-8").splitlines()
            if ln.strip()]


def get_bytes(uri: str) -> bytes:
    bucket, key = split(uri)
    return _s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def put_bytes(uri: str, body: bytes, content_type: str) -> str:
    bucket, key = split(uri)
    _s3.put_object(Bucket=bucket, Key=key, Body=body,
                   ContentType=content_type)
    return uri


def exists(uri: str) -> bool:
    from botocore.exceptions import ClientError
    bucket, key = split(uri)
    try:
        _s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey",
                                             "NotFound"):
            return False
        raise


def get_response_texts(output_s3_prefix: str) -> list[str]:
    """A bedrock-inference run's response.jsonl -> the output_text of
    every SUCCESS record."""
    bucket, key = split(output_s3_prefix.rstrip("/"))
    body = _s3.get_object(Bucket=bucket,
                          Key=f"{key}/response.jsonl")["Body"].read()
    out = []
    for ln in body.decode("utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        if row.get("status") == "SUCCESS":
            out.append(row.get("output_text") or "")
    return out
