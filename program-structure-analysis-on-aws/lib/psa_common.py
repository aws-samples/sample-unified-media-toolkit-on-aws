"""
Shared plumbing for the program-structure-analysis states and gateway.
Smaller than every sibling ON PURPOSE: PSA makes NO model calls — no
bedrock, no tts, no prompt registry, no sibling gateway invocation. Pure
code over sibling run artifacts; the whole run is seconds of Lambda time.

Run prefix layout (micro-workflow/program-structure-analysis/{workflow_id}/):
    _plan/plan.json          resolved inputs + mode
    structure.json           markers + break candidates (+ sequences/acts)
    rundown.json             NEWS MODE ONLY (stories + interstitials + roles)
    psa_manifest.json        pointers + mode evidence
"""
from __future__ import annotations

import json
import os

import boto3
from botocore.exceptions import ClientError

_s3 = boto3.client("s3")

PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "program-structure-analysis")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")


def run_prefix(workflow_id: str) -> str:
    return f"micro-workflow/{TARGET_NAME}/{workflow_id}"


def read_json(bucket: str, key: str):
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body.decode("utf-8"))


def read_json_if_exists(bucket: str, key: str):
    """None ONLY on a genuine missing key; everything else re-raises."""
    try:
        return read_json(bucket, key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def read_jsonl(bucket: str, key: str) -> list:
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return [json.loads(line) for line in body.decode("utf-8").splitlines()
            if line.strip()]


def write_json(bucket: str, key: str, payload) -> None:
    _s3.put_object(Bucket=bucket, Key=key,
                   Body=json.dumps(payload, ensure_ascii=False,
                                   indent=1).encode("utf-8"),
                   ContentType="application/json")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// uri: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 uri: {uri!r}")
    return bucket, key
