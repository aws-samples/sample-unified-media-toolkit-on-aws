"""
Shared S3 client and helpers.

One boto3 client per container, matching `conversation-analysis/lib/s3_utils.py`
and `media-processing/lib/s3_utils.py`. Nothing here imports from this project
except `custom_exceptions` for the 404 discipline.

The whole request/response contract is S3-mediated (README.md §4), so JSONL read and
write live here rather than in the states — sync and batch must produce a
byte-identical `response.jsonl` shape, and two writers would not.
"""
from __future__ import annotations

import json
import logging

import boto3
from botocore.config import Config

log = logging.getLogger(__name__)

s3 = boto3.client("s3", config=Config(max_pool_connections=60))

_NOT_FOUND_CODES = ("404", "NoSuchKey", "NotFound")


def parse_s3uri(uri: str) -> tuple[str, str]:
    """Parse 's3://bucket/key/path' → ('bucket', 'key/path')."""
    from lib.custom_exceptions import ValidationException

    if not uri or not str(uri).startswith("s3://"):
        raise ValidationException(f"invalid s3uri — must start with 's3://': {uri!r}")
    rest = str(uri)[5:]
    if "/" not in rest:
        raise ValidationException(f"invalid s3uri — no key after bucket: {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket or not key.strip("/"):
        raise ValidationException(f"invalid s3uri — empty bucket or key: {uri!r}")
    return bucket, key.strip("/")


def _is_not_found(exc) -> bool:
    response = getattr(exc, "response", None) or {}
    code = (response.get("Error") or {}).get("Code", "")
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return code in _NOT_FOUND_CODES or status == 404


def get_json(bucket: str, key: str):
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def get_json_if_exists(bucket: str, key: str):
    """
    The parsed object, or `None` **only** when it does not exist.

    Everything else re-raises, and that is the point (README.md §5.1, §12.3). A broad
    `except` here reads a throttled or denied GET as "no checkpoint yet" and
    re-inferences every completed record — so the failure gets *more* expensive
    under load, which is exactly when it happens.
    """
    try:
        return get_json(bucket, key)
    except Exception as exc:                                       # noqa: BLE001
        if _is_not_found(exc):
            return None
        raise


def get_text(bucket: str, key: str, encoding: str = "utf-8") -> str:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode(encoding)


def put_json(bucket: str, key: str, data) -> None:
    body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body,
                  ContentType="application/json")


def put_text(bucket: str, key: str, text: str,
             content_type: str = "text/plain") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=(text or "").encode("utf-8"),
                  ContentType=content_type)


def jsonl_line(row) -> str:
    """THE compact JSONL serialization — one home (review, 2026-09-06).
    These bytes are billed, transferred, AND identity-digested: put_jsonl,
    put_jsonl_streamed and the gateway's staged-records writer must all
    serialize identically, or staged-vs-inline submissions of the same
    records would write different request.jsonl bytes."""
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def put_jsonl(bucket: str, key: str, rows: list) -> int:
    """
    Write one compact JSON document per line. Returns the row count.

    `separators` is explicit and `ensure_ascii` is off: Bedrock batch input is
    billed and transferred by byte, and a batch record set is the largest thing this
    workflow writes.
    """
    body = "".join(jsonl_line(row) + "\n" for row in rows)
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"),
                  ContentType="application/jsonl")
    return len(rows)


def get_jsonl(bucket: str, key: str) -> list:
    """
    Parse a JSONL object into a list, skipping blank lines.

    A malformed line raises rather than being skipped: silently dropping a record
    would under-report `records_total`, and the summary's arithmetic is how a caller
    detects partial success (§7).
    """
    rows = []
    for lineno, line in enumerate(get_text(bucket, key).split("\n"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            from lib.custom_exceptions import ValidationException
            raise ValidationException(
                f"s3://{bucket}/{key} line {lineno} is not valid JSON: {exc}")
    return rows


def object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:                                      # noqa: BLE001
        if _is_not_found(exc):
            return False
        # AccessDenied must not read as "absent" — that would silently skip a
        # checkpoint or re-run work that already exists.
        raise


def list_keys(bucket: str, prefix: str) -> list:
    """
    Every key under `prefix`, paginated, directory placeholders removed.

    Used to read a batch job's output. The caller scopes `prefix` to the job's own
    subfolder — Bedrock writes to `{s3Uri}/{jobId}/`, and listing the shared parent
    picks up another job's partially-written output (the fix already made in
    `core-lib/lib/segmentHelper/models/base.js:getBatchInferenceJobResponses`).
    """
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            if not obj["Key"].endswith("/"):
                keys.append(obj["Key"])
    return keys


def iter_jsonl(bucket: str, key: str):
    """Stream a JSONL object as (record_index, parsed_row) without holding
    the file — the submit path's 668 MB OOM fix (2026-09-05). The yielded
    index counts RECORDS (blank lines skipped); the error message reports
    the FILE line number, same contract as get_jsonl, and raises the same
    ValidationException so a malformed line refuses the run instead of
    surfacing as an anonymous ValueError."""
    from lib.custom_exceptions import ValidationException

    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    index = 0
    # 1 MiB chunks, NOT botocore's 1 KiB default: iter_lines re-copies its
    # pending buffer per chunk, so the default is quadratic in line length
    # — and this feeds ~440 KB image-bearing lines (review, 2026-09-06)
    for lineno, raw in enumerate(body.iter_lines(chunk_size=1 << 20),
                                 start=1):
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        try:
            yield index, json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationException(
                f"s3://{bucket}/{key} line {lineno} is not valid JSON: "
                f"{exc}") from exc
        index += 1


def put_jsonl_streamed(bucket: str, key: str, rows,
                       part_size: int = 16 * 1024 * 1024,
                       client=None) -> int:
    """Multipart-write an iterable of ROW OBJECTS as compact JSONL
    (serialized HERE, via jsonl_line — one home for the byte convention);
    memory is bounded by one part. Returns the row count. Aborts the
    upload on any failure — a raising source generator included — so no
    half-written object survives. `client` lets a caller with its own
    (or test-faked) S3 client inject it."""
    s3c = client or s3
    upload = s3c.create_multipart_upload(Bucket=bucket, Key=key,
                                         ContentType="application/jsonl")
    parts, buffer, part_no, count = [], bytearray(), 1, 0
    try:
        for row in rows:
            buffer += (jsonl_line(row) + "\n").encode("utf-8")
            count += 1
            if len(buffer) >= part_size:
                # boto3 takes any bytes-like Body; a bytes() copy here
                # doubles the peak per-part memory for nothing, and the
                # buffer is replaced (never mutated) after the call
                parts.append({"ETag": s3c.upload_part(
                    Bucket=bucket, Key=key, UploadId=upload["UploadId"],
                    PartNumber=part_no, Body=buffer)["ETag"],
                    "PartNumber": part_no})
                buffer, part_no = bytearray(), part_no + 1
        if buffer or not parts:
            parts.append({"ETag": s3c.upload_part(
                Bucket=bucket, Key=key, UploadId=upload["UploadId"],
                PartNumber=part_no, Body=buffer)["ETag"],
                "PartNumber": part_no})
        s3c.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload["UploadId"],
            MultipartUpload={"Parts": parts})
    except Exception:
        s3c.abort_multipart_upload(Bucket=bucket, Key=key,
                                   UploadId=upload["UploadId"])
        raise
    return count
