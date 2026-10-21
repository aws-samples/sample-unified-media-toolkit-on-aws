"""
s3io.py — the write side (sources.py owns the read side). Full s3:// uris
in and out; states pass URIs, never bodies (256 KB state I/O).
"""
from __future__ import annotations

import json

import boto3

_s3 = boto3.client("s3")


def put_json(uri: str, doc) -> str:
    bucket, key = uri[5:].split("/", 1)
    _s3.put_object(Bucket=bucket, Key=key,
                   Body=json.dumps(doc, ensure_ascii=False).encode(),
                   ContentType="application/json")
    return uri


def get_json(uri: str):
    bucket, key = uri[5:].split("/", 1)
    return json.loads(_s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def exists(uri: str) -> bool:
    from botocore.exceptions import ClientError
    bucket, key = uri[5:].split("/", 1)
    try:
        _s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def get_response_texts(output_s3_prefix: str) -> list[str]:
    """A bedrock-inference run's response.jsonl -> the output_text of
    every SUCCESS record (the enrichment states' consumption shape)."""
    import json as _json
    bucket, key = output_s3_prefix[5:].rstrip("/").split("/", 1)
    body = _s3.get_object(Bucket=bucket,
                          Key=f"{key}/response.jsonl")["Body"].read()
    out = []
    for ln in body.decode("utf-8").splitlines():
        if not ln.strip():
            continue
        row = _json.loads(ln)
        if row.get("status") == "SUCCESS":
            out.append(row.get("output_text") or "")
    return out
