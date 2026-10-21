"""media-uploads target — presigned uploads INTO the ingest bucket.

Four tools, one Lambda, no state machine (synchronous control plane; the
data plane is the CLIENT PUTting bytes straight to S3 via the presigned
URLs — bytes never pass through the gateway):

  start_upload          one file: single presigned PUT (small) or a
                        multipart session with EVERY part URL (the 5 GiB
                        cap bounds the plan to <=50 parts, so the full
                        set always fits one response)
  get_upload_part_urls  re-presign named parts — the refresh tool for
                        URLs that outlived the Lambda's signing
                        credentials (HTTP 403 on PUT), never pagination
  complete_upload       CompleteMultipartUpload from the client's
                        collected part ETags; the object exists only now
  abort_upload          drop an abandoned multipart session (the ingest
                        bucket's AbortIncompleteMultipartUpload lifecycle
                        rule is the backstop — incomplete parts bill
                        until aborted)

The inverse of media-outputs, deliberately separate from media-ingest's
lister: this role WRITES (PutObject/AbortMultipartUpload + ListBucket for
the overwrite check) and still has no GetObject — one concern per policy.
"""
import logging
import math
import mimetypes
import os
import posixpath
import re
from datetime import datetime, timedelta, timezone

import boto3
from botocore.client import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

INGEST_BUCKET = os.environ["INGEST_BUCKET"]
# Optional operator-enforced key prefix ("" = whole bucket, the default:
# the ingest bucket is the caller's own). Used RAW — the IAM policy embeds
# the same parameter verbatim, and the template's AllowedPattern forbids a
# leading '/', so the two layers can never disagree (a mismatch would 403
# every PUT, which the retry guidance would then loop on forever).
ALLOWED_PREFIX = os.environ.get("UPLOAD_PREFIX", "")

_MIN_EXPIRY, _MAX_EXPIRY, _DEFAULT_EXPIRY = 60, 3600, 3600
MAX_SIZE_BYTES = 5 * 1024**3          # user decision 2026-10-14: cap at 5 GiB
SINGLE_PUT_MAX = 64 * 1024**2         # above this, multipart (speed + retry)
DEFAULT_PART_SIZE = 128 * 1024**2
MIN_PART_SIZE = 8 * 1024**2           # S3 minimum is 5 MiB; 8 keeps headroom
MAX_PART_URLS = 50                    # the house per-response context budget

# Overlay first, stdlib mimetypes second, octet-stream last; an explicit
# content_type parameter beats all three. The stdlib already covers common
# documents and images (.pdf/.docx/.json/.xml/.csv/.jpg/.png/.svg/...); the
# overlay carries what it gets wrong or misses — broadcast media wrappers,
# sidecar/caption formats, jsonl. The resolved type is FOLDED INTO THE
# SIGNATURE, so the client must echo it verbatim as the Content-Type header
# on a single PUT.
_MIME_OVERLAY = {
    # media wrappers / essence
    ".mxf": "application/mxf", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".ts": "video/mp2t", ".m2ts": "video/mp2t",
    ".mts": "video/mp2t", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".flac": "audio/flac", ".webm": "video/webm",
    ".gxf": "application/gxf", ".lxf": "application/lxf",
    # structured text / sidecars the stdlib misses
    ".jsonl": "application/jsonl", ".ndjson": "application/x-ndjson",
    ".md": "text/markdown", ".yaml": "application/yaml",
    ".yml": "application/yaml",
    # caption/subtitle formats (uploadable as VALIDATION references —
    # the family rule stands: captions never feed a workflow)
    ".vtt": "text/vtt", ".ttml": "application/ttml+xml",
    ".dfxp": "application/ttml+xml", ".itt": "application/ttml+xml",
    ".scc": "text/plain",
}

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"),
                  config=Config(signature_version="s3v4"))


def handler(event, context):
    tool = _resolve_tool(context)
    # No tool here starts an execution; presigning and the multipart calls
    # are deterministic in their inputs. Warn-and-ignore, never reject
    # (the caller is an LLM; a hallucinated field must not fail a request
    # it could not have changed) — same inline check as download_url.py.
    if "uuid" in event or event.get("idempotency_token"):
        logger.warning("%s: idempotency_token/uuid ignored — uploads are "
                       "identified by upload_id, not a run hash", tool)
    if tool == "start_upload":
        return _start_upload(event)
    if tool == "get_upload_part_urls":
        return _get_upload_part_urls(event)
    if tool == "complete_upload":
        return _complete_upload(event)
    if tool == "abort_upload":
        return _abort_upload(event)
    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ── start_upload ───────────────────────────────────────────────────────────────

def _start_upload(event):
    if not (event.get("key") or "").strip():
        return _error("key is required — the destination path in the ingest "
                      "bucket", "MISSING_PARAMETER")
    key, err = _resolve_key(event.get("key"))
    if err:
        return _error(err, "INVALID_PARAMETER")
    size = _coerce_int(event.get("size_bytes"))
    if size is None:
        return _error("size_bytes (integer) is required — stat the local "
                      "file first", "INVALID_PARAMETER")
    if size <= 0:
        return _error("size_bytes must be positive", "INVALID_PARAMETER")
    if size > MAX_SIZE_BYTES:
        return _error(
            f"size_bytes {size} exceeds the {MAX_SIZE_BYTES}-byte (5 GiB) "
            f"upload cap", "FILE_TOO_LARGE")

    content_type = (event.get("content_type") or "").strip() or _guess_mime(key)
    expiry, exp_err = _resolve_expiry(event)
    if exp_err:
        return exp_err

    # Overwrite guard via ListBucket (this role deliberately has no
    # GetObject, so no HeadObject): exact-key match only.
    existing = _existing_size(key)
    overwrite = str(event.get("overwrite", "")).lower() in ("true", "1", "yes")
    if existing is not None and not overwrite:
        return _error(
            f"s3://{INGEST_BUCKET}/{key} already exists ({existing} bytes) — "
            f"pass overwrite: true to replace it, or choose another key",
            "KEY_EXISTS")

    s3uri = f"s3://{INGEST_BUCKET}/{key}"
    if size <= SINGLE_PUT_MAX:
        # ContentLength is signed here too — a single PUT of any other size
        # than the declared one fails, same enforcement as the parts.
        url = s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": INGEST_BUCKET, "Key": key,
                    "ContentType": content_type, "ContentLength": size},
            ExpiresIn=expiry)
        return {"mode": "single", "s3uri": s3uri, "url": url,
                "content_type": content_type, "size_bytes": size,
                "expires_at": _expires_at(expiry),
                "note": "PUT the whole file to url with this exact "
                        "Content-Type header; the object exists on success "
                        "— no complete step"}

    # Multipart plan: default 128 MiB parts, floored so the count never
    # exceeds the per-response URL budget (the 5 GiB cap guarantees a
    # solution exists), and never below the S3 minimum.
    part_size = _coerce_int(event.get("part_size") or DEFAULT_PART_SIZE)
    if part_size is None:
        return _error("part_size must be an integer (bytes)",
                      "INVALID_PARAMETER")
    part_size = max(part_size, MIN_PART_SIZE,
                    math.ceil(size / MAX_PART_URLS))
    part_count = math.ceil(size / part_size)

    mpu = s3.create_multipart_upload(Bucket=INGEST_BUCKET, Key=key,
                                     ContentType=content_type)
    upload_id = mpu["UploadId"]
    try:
        parts = [{"part_number": n,
                  "url": _presign_part(key, upload_id, n, expiry,
                                       _part_len(size, part_size, n))}
                 for n in range(1, part_count + 1)]
    except Exception:
        # Never leak a session the caller got no handle for — an unaborted
        # multipart upload stores (and bills) its parts invisibly, and the
        # lifecycle backstop is operator-owned and optional. The abort is
        # best-effort: it must never MASK the original error, and if it
        # fails too the log carries the orphaned upload_id.
        try:
            s3.abort_multipart_upload(Bucket=INGEST_BUCKET, Key=key,
                                      UploadId=upload_id)
        except Exception as abort_exc:  # noqa: BLE001
            logger.error("orphaned multipart session %s on %s — abort "
                         "failed too (%s); the bucket lifecycle rule is "
                         "the remaining cleanup", upload_id, key, abort_exc)
        raise
    return {"mode": "multipart", "s3uri": s3uri, "upload_id": upload_id,
            "content_type": content_type, "size_bytes": size,
            "part_size": part_size,
            "part_count": part_count, "parts": parts,
            "expires_at": _expires_at(expiry),
            "note": "PUT byte range (n-1)*part_size..n*part_size-1 to part "
                    "n's url (parallel PUTs are the speedup; no Content-Type "
                    "header on parts); record each response's ETag; on HTTP "
                    "403 re-request that part via get_upload_part_urls; "
                    "finish with complete_upload — the object does not exist "
                    "until then"}


# ── get_upload_part_urls (refresh, not pagination) ────────────────────────────

def _get_upload_part_urls(event):
    if not (event.get("key") or "").strip():
        return _error("key is required — the destination path in the ingest "
                      "bucket", "MISSING_PARAMETER")
    key, err = _resolve_key(event.get("key"))
    if err:
        return _error(err, "INVALID_PARAMETER")
    upload_id = (event.get("upload_id") or "").strip()
    if not upload_id:
        return _error("upload_id is required — from start_upload",
                      "MISSING_PARAMETER")
    # size_bytes + part_size reconstruct the plan the fresh URLs must match
    # — every part URL is signed with its EXACT Content-Length, which is
    # what makes the 5 GiB cap enforced rather than advisory: without the
    # plan, any part number could be presigned and PUT at any size.
    size = _coerce_int(event.get("size_bytes"))
    part_size = _coerce_int(event.get("part_size"))
    if size is None or part_size is None:
        return _error("size_bytes and part_size (integers — size_bytes as "
                      "sent to start_upload, part_size from its response) "
                      "are required", "INVALID_PARAMETER")
    if not (0 < size <= MAX_SIZE_BYTES) or part_size < MIN_PART_SIZE:
        return _error(f"size_bytes must be 1..{MAX_SIZE_BYTES} and "
                      f"part_size at least {MIN_PART_SIZE}",
                      "INVALID_PARAMETER")
    part_count = math.ceil(size / part_size)
    nums = event.get("part_numbers")
    if (not isinstance(nums, list) or not nums
            or not all(_is_part_number(n) and n <= part_count
                       for n in nums)):
        return _error(f"part_numbers must be a list of integers within this "
                      f"upload's plan (1-{part_count})", "INVALID_PARAMETER")
    if len(nums) > MAX_PART_URLS:
        return _error(f"{len(nums)} part_numbers exceeds {MAX_PART_URLS} "
                      f"per call — batch the refreshes", "INVALID_PARAMETER")
    expiry, exp_err = _resolve_expiry(event)
    if exp_err:
        return exp_err
    return {"parts": [{"part_number": n,
                       "url": _presign_part(key, upload_id, n, expiry,
                                            _part_len(size, part_size, n))}
                      for n in nums],
            "expires_at": _expires_at(expiry)}


# ── complete_upload ────────────────────────────────────────────────────────────

def _complete_upload(event):
    if not (event.get("key") or "").strip():
        return _error("key is required — the destination path in the ingest "
                      "bucket", "MISSING_PARAMETER")
    key, err = _resolve_key(event.get("key"))
    if err:
        return _error(err, "INVALID_PARAMETER")
    upload_id = (event.get("upload_id") or "").strip()
    if not upload_id:
        return _error("upload_id is required — from start_upload",
                      "MISSING_PARAMETER")
    parts = event.get("parts")
    if (not isinstance(parts, list) or not parts
            or not all(isinstance(p, dict) and _is_part_number(p.get("part_number"))
                       and p.get("etag") for p in parts)):
        return _error("parts must be a non-empty list of {part_number, etag} "
                      "— the ETags come from each part PUT's response",
                      "INVALID_PARAMETER")
    # The AUTHORITATIVE cap check: per-part ContentLength signing limits
    # each URL, but plans are caller-reconstructed, so the sum is only
    # bounded here — against S3's own record. Summed over the COMPLETING
    # subset only: stray parts stored under other numbers are discarded by
    # S3 at complete and must not fail a legitimate assembly.
    completing = {p["part_number"] for p in parts}
    total, marker = 0, 0
    try:
        while True:
            listed = s3.list_parts(Bucket=INGEST_BUCKET, Key=key,
                                   UploadId=upload_id,
                                   PartNumberMarker=marker, MaxParts=1000)
            total += sum(p["Size"] for p in listed.get("Parts", [])
                         if p["PartNumber"] in completing)
            if not listed.get("IsTruncated"):
                break
            marker = listed["NextPartNumberMarker"]
    except s3.exceptions.NoSuchUpload:
        # the retry-after-lost-response path: the designed answer, not a
        # raw Lambda error
        return _error("no such multipart upload — already completed, "
                      "aborted, or swept by the bucket's lifecycle rule",
                      "NO_SUCH_UPLOAD")
    if total > MAX_SIZE_BYTES:
        # only reachable by exceeding the cap deliberately (per-part
        # signing bounds every honest plan); abort is best-effort and must
        # not mask the verdict
        try:
            s3.abort_multipart_upload(Bucket=INGEST_BUCKET, Key=key,
                                      UploadId=upload_id)
            swept = "the session was aborted"
        except Exception as abort_exc:  # noqa: BLE001
            logger.error("over-cap session %s on %s could not be aborted "
                         "(%s) — lifecycle rule is the remaining cleanup",
                         upload_id, key, abort_exc)
            swept = "aborting the session failed; it remains until swept"
        return _error(
            f"completing parts total {total} bytes, over the "
            f"{MAX_SIZE_BYTES}-byte (5 GiB) upload cap — {swept}",
            "FILE_TOO_LARGE")
    try:
        resp = s3.complete_multipart_upload(
            Bucket=INGEST_BUCKET, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": [
                {"PartNumber": p["part_number"],
                 "ETag": str(p["etag"]).strip('"')}
                for p in sorted(parts, key=lambda p: p["part_number"])]})
    except s3.exceptions.NoSuchUpload:
        return _error("no such multipart upload — already completed, "
                      "aborted, or swept by the bucket's lifecycle rule",
                      "NO_SUCH_UPLOAD")
    except s3.exceptions.ClientError as exc:  # EntityTooSmall, InvalidPart…
        code = exc.response.get("Error", {}).get("Code", "ClientError")
        return _error(f"complete failed ({code}): "
                      f"{exc.response.get('Error', {}).get('Message', '')}",
                      _snake_code(code))
    size = _existing_size(key)
    out = {"s3uri": f"s3://{INGEST_BUCKET}/{key}",
           "etag": resp.get("ETag", "").strip('"'),
           "note": "the s3uri feeds workflow submit tools directly"}
    if size is not None:
        out["size_bytes"] = size   # ListBucket verification, free
    return out


# ── abort_upload ───────────────────────────────────────────────────────────────

def _abort_upload(event):
    if not (event.get("key") or "").strip():
        return _error("key is required — the destination path in the ingest "
                      "bucket", "MISSING_PARAMETER")
    key, err = _resolve_key(event.get("key"))
    if err:
        return _error(err, "INVALID_PARAMETER")
    upload_id = (event.get("upload_id") or "").strip()
    if not upload_id:
        return _error("upload_id is required — from start_upload",
                      "MISSING_PARAMETER")
    try:
        s3.abort_multipart_upload(Bucket=INGEST_BUCKET, Key=key,
                                  UploadId=upload_id)
    except s3.exceptions.NoSuchUpload:
        return _error("no such multipart upload — already completed, "
                      "aborted, or swept by the bucket's lifecycle rule",
                      "NO_SUCH_UPLOAD")
    return {"aborted": True, "key": key, "upload_id": upload_id}


# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve_tool(context):
    try:
        name = context.client_context.custom["bedrockAgentCoreToolName"]
        return name.split("___", 1)[1] if "___" in name else name
    except (AttributeError, KeyError, TypeError):
        return ""


def _resolve_key(raw):
    """Validate one destination key. Returns (key, None) or (None, message)."""
    raw = (raw or "").strip()
    if not raw:
        return None, "key is required — the destination path in the ingest bucket"
    if raw.startswith("s3://"):
        rest = raw[5:]
        bucket, _, raw = rest.partition("/")
        if bucket != INGEST_BUCKET:
            return None, f"bucket must be {INGEST_BUCKET}"
    if raw.endswith("/"):
        return None, "key names an object, not a prefix — include the filename"
    key = posixpath.normpath(raw.lstrip("/"))
    # after normpath, traversal can only survive as an exact '..' SEGMENT —
    # a startswith check would false-reject keys like '..archive/f.mp4'
    if key in (".", "") or ".." in key.split("/"):
        return None, "key must not traverse outside the bucket"
    if ALLOWED_PREFIX and not key.startswith(ALLOWED_PREFIX):
        return None, f"key must be under the deployment's upload prefix {ALLOWED_PREFIX!r}"
    return key, None


def _guess_mime(key):
    ext = posixpath.splitext(key)[1].lower()
    return (_MIME_OVERLAY.get(ext)
            or mimetypes.guess_type(key)[0]
            or "application/octet-stream")


def _existing_size(key):
    """Exact-key existence via ListBucket (no GetObject on this role)."""
    resp = s3.list_objects_v2(Bucket=INGEST_BUCKET, Prefix=key, MaxKeys=1)
    for it in resp.get("Contents", []):
        if it["Key"] == key:
            return it["Size"]
    return None


def _coerce_int(value):
    """Strict integer for byte counts: int (not bool), integral float, or a
    digit string — anything else is None. JSON true would otherwise coerce
    to 1 and plan a 1-byte upload whose every signed PUT then 403s."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _is_part_number(n):
    # bool is an int subclass: JSON true would silently mean part 1
    return isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 10000


def _snake_code(code):
    """S3's CamelCase error codes -> the family's SNAKE_CASE errorCode
    grammar (EntityTooSmall -> ENTITY_TOO_SMALL)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", code).upper()


def _part_len(size, part_size, n):
    return min(part_size, size - (n - 1) * part_size)


def _presign_part(key, upload_id, part_number, expiry, content_length):
    # ContentLength is folded into the signature: a PUT of any other size
    # fails, which is what turns the per-upload size plan into an enforced
    # cap instead of a declared one.
    return s3.generate_presigned_url(
        "upload_part",
        Params={"Bucket": INGEST_BUCKET, "Key": key, "UploadId": upload_id,
                "PartNumber": part_number, "ContentLength": content_length},
        ExpiresIn=expiry)


def _resolve_expiry(event):
    expiry = event.get("expiry_seconds")
    try:
        expiry = int(expiry) if expiry is not None else _DEFAULT_EXPIRY
    except (TypeError, ValueError):
        return None, _error("expiry_seconds must be an integer",
                            "INVALID_PARAMETER")
    return max(_MIN_EXPIRY, min(_MAX_EXPIRY, expiry)), None


def _expires_at(expiry):
    return (datetime.now(timezone.utc)
            + timedelta(seconds=expiry)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _error(message, code):
    logger.warning("%s: %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
