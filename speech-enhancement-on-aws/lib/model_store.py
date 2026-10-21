# lib/model_store.py
"""
Model store — downloads and caches ONNX model archives for speech-enhancement.

The DeepFilterNet3 model artefacts are baked into the container image at build time
(extracted flat to /opt/models/ — no S3 or CloudFront fetch needed at inference).
This module provides the infrastructure used at image-build time and for any
runtime seeding path where models need to be fetched.

Download hierarchy
───────────────────
  1. Customer S3 bucket  (ProxyBucket/_model_artefacts/…)
     Copied there at CFN stack creation time.

  2. CloudFront CDN      (opensource_models/…)
     Fallback when the S3 key is absent — e.g. a model was accidentally deleted
     or a new model variant is requested before the stack was updated.

/tmp warm-container cache
──────────────────────────
  Every ensure_in_tmp() call checks whether the target directory already
  has files before doing any I/O. On warm Lambda invocations this costs
  < 1 ms and zero bytes transferred.

Atomicity
──────────
  Extraction goes to   tmp_dir + '.__staging'
  then os.rename() →   tmp_dir   on success.
  Partial extractions are never visible as tmp_dir.
  Safe for concurrent cold-starts on the same container.

Streaming extraction
──────────────────────
  The S3/CF archive is downloaded into RAM then extracted via tarfile mode="r|gz"
  (true streaming on a BytesIO buffer). Peak memory = 1× archive size — the
  decompressed content is written directly to disk rather than buffered a second time.

Public API — inference-time (S3 → /tmp)
─────────────────────────────────────────
  ensure_in_tmp(bucket, s3_key, tmp_dir, label="")
      Download model archive from customer S3 → extract to tmp_dir.
      No-op if tmp_dir already contains files (warm container).
      Raises RuntimeError on S3 failure — no CF fallback.

  ensure_in_tmp_with_fallback(bucket, s3_key, cf_key, tmp_dir, label="")
      Same as ensure_in_tmp but falls back to CloudFront if the S3 key is missing.
      Returns "cache", "s3", or "cloudfront" indicating the source used.

  find_file(tmp_dir, pattern, required=True) → str | None
      Return path of first file matching glob pattern under tmp_dir.
      Returns None if not found and required=False.
      Raises FileNotFoundError if not found and required=True (default).

Public API — model-availability gate (CloudFront → S3, no /tmp)
────────────────────────────────────────────────────────────────
  ensure_in_s3(bucket, s3_key, cf_key=None, label="", cf_url=None) → "exists" | "copied"
      Check whether the archive exists in the customer S3 bucket.
      If missing, streams it from CloudFront → put_object into S3.
      Nothing is written to /tmp.

      cf_url  : full CloudFront URL (preferred — caller builds from runtime cf_host)
      cf_key  : relative path appended to CLOUDFRONT_URL env var (legacy)
      One of cf_url or cf_key must be supplied when the S3 key is absent.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

_CF_URL = os.environ.get("CLOUDFRONT_URL", "").rstrip("/")
_DL_TIMEOUT = int(os.environ.get("MODEL_DL_TIMEOUT", "180"))
_DL_RETRIES = int(os.environ.get("MODEL_DL_RETRIES", "3"))


# =============================================================================
# PUBLIC — inference-time  (S3 → /tmp)
# =============================================================================

def ensure_in_tmp(
    bucket:  str,
    s3_key:  str,
    tmp_dir: str,
    label:   str = "",
) -> None:
    """
    Ensure model archive is extracted to tmp_dir.
    Downloads from customer S3 — no CloudFront fallback.
    No-op on warm Lambda containers.
    """
    if _dir_has_files(tmp_dir):
        log.info(f"[/tmp ✅] {label or s3_key}")
        return

    log.info(f"[s3→/tmp] {label or s3_key}  s3://{bucket}/{s3_key}")
    t0 = time.monotonic()
    raw = _s3_get_bytes(bucket, s3_key)
    _extract_bytes_to_dir(raw, tmp_dir)
    log.info(
        f"[s3→/tmp ✅] {label or s3_key}  "
        f"{len(raw)/1_048_576:.1f} MB  {time.monotonic()-t0:.1f}s"
    )


def ensure_in_tmp_with_fallback(
    bucket:  str,
    s3_key:  str,
    cf_key:  str,
    tmp_dir: str,
    label:   str = "",
) -> str:
    """
    Ensure model archive is extracted to tmp_dir.
    Tries customer S3 first; falls back to CloudFront if key is missing.
    Returns "cache", "s3", or "cloudfront" indicating the source used.
    """
    if _dir_has_files(tmp_dir):
        log.info(f"[/tmp ✅] {label or s3_key}")
        return "cache"

    # ── Try S3 ────────────────────────────────────────────────────────────────
    try:
        log.info(
            f"[s3→/tmp] {label or s3_key}  s3://{bucket}/{s3_key}"
        )
        t0 = time.monotonic()
        raw = _s3_get_bytes(bucket, s3_key)
        _extract_bytes_to_dir(raw, tmp_dir)
        log.info(
            f"[s3→/tmp ✅] {label or s3_key}  "
            f"{len(raw)/1_048_576:.1f} MB  {time.monotonic()-t0:.1f}s"
        )
        return "s3"

    except _S3NotFoundError:
        log.warning(
            f"[s3 miss] s3://{bucket}/{s3_key} — "
            f"falling back to CloudFront"
        )

    # ── CloudFront fallback ───────────────────────────────────────────────────
    url = f"{_cf_url()}/{cf_key}"
    log.info(f"[cf→/tmp] {label or cf_key}  {url}")
    t0 = time.monotonic()
    raw = _cf_get_bytes(url)
    _extract_bytes_to_dir(raw, tmp_dir)
    log.info(
        f"[cf→/tmp ✅] {label or cf_key}  "
        f"{len(raw)/1_048_576:.1f} MB  {time.monotonic()-t0:.1f}s"
    )
    return "cloudfront"


def find_file(tmp_dir: str, pattern: str, required: bool = True) -> Optional[str]:
    """
    Return path of first file matching glob `pattern` under `tmp_dir`.
    Returns None if not found and required=False.
    Raises FileNotFoundError if not found and required=True (default).
    """
    hits = sorted(Path(tmp_dir).rglob(pattern))
    if not hits:
        if not required:
            return None
        raise FileNotFoundError(
            f"No file matching '{pattern}' under {tmp_dir}"
        )
    if len(hits) > 1:
        log.warning(f"{len(hits)} matches for '{pattern}'; using {hits[0]}")
    return str(hits[0])

# =============================================================================
# PUBLIC — model-availability gate  (CloudFront → S3, no /tmp)
# =============================================================================


def ensure_in_s3(
    bucket:  str,
    s3_key:  str,
    cf_key:  str | None = None,
    label:   str = "",
    cf_url:  str | None = None,
) -> str:
    """
    Ensure the model archive exists in the customer S3 bucket.
    If missing, streams it from CloudFront → s3.put_object.
    Nothing is written to /tmp.

    Parameters
    ──────────
    cf_url  : full CloudFront URL (preferred).  Caller builds this from the
              runtime cf_host value in $.data — no env var required.
    cf_key  : relative path appended to the CLOUDFRONT_URL env var.
              Legacy interface retained for backwards compatibility.

    Exactly one of cf_url or cf_key must be supplied when the S3 key is
    absent; raise ValueError if both are None at that point.

    Returns "exists" or "copied".
    """
    if _s3_key_exists(bucket, s3_key):
        log.info(f"[s3 ✅] {label or s3_key}")
        return "exists"

    log.info(f"[s3 miss] {label or s3_key} — streaming from CloudFront")

    if cf_url:
        url = cf_url
    elif cf_key:
        url = f"{_cf_url()}/{cf_key}"
    else:
        raise ValueError(
            f"ensure_in_s3: s3://{bucket}/{s3_key} is missing and neither "
            f"cf_url nor cf_key was supplied — cannot download"
        )
    url = _normalise_url(url)

    t0 = time.monotonic()
    raw = _cf_get_bytes(url)

    log.info(
        f"  {len(raw)/1_048_576:.1f} MB downloaded in "
        f"{time.monotonic()-t0:.1f}s — uploading to S3"
    )
    t1 = time.monotonic()
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=raw,
        ContentType="application/gzip",
    )
    log.info(
        f"[cf→s3 ✅] {label or s3_key}  "
        f"upload {time.monotonic()-t1:.1f}s  →  s3://{bucket}/{s3_key}"
    )
    return "copied"


# =============================================================================
# PRIVATE — S3
# =============================================================================

class _S3NotFoundError(Exception):
    pass


def _s3_key_exists(bucket: str, key: str) -> bool:
    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _s3_get_bytes(bucket: str, key: str) -> bytes:
    try:
        resp = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            raise _S3NotFoundError(f"s3://{bucket}/{key} not found") from exc
        raise RuntimeError(
            f"S3 error downloading s3://{bucket}/{key}: {code}"
        ) from exc


# =============================================================================
# PRIVATE — CloudFront
# =============================================================================

def _normalise_url(url: str) -> str:
    """
    Ensure a CloudFront URL has an https:// scheme.

    cf_host values sometimes arrive as bare hostnames
    (e.g. "dx2y1cac29mt3.cloudfront.net") when the CFN parameter was
    stored without a scheme.  urllib.request refuses schemeless URLs
    with "unknown url type" — normalise here so every caller is safe.
    """
    if url and not url.startswith(("https://", "http://")):
        return "https://" + url
    return url


def _cf_url() -> str:
    if not _CF_URL:
        raise EnvironmentError(
            "CLOUDFRONT_URL environment variable is not set"
        )
    return _CF_URL


def _cf_get_bytes(
    url:     str,
    retries: int = _DL_RETRIES,
    timeout: int = _DL_TIMEOUT,
) -> bytes:
    """
    Download a CloudFront URL into memory with retries.
    Returns raw bytes.
    """
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            log.info(f"[cf dl] attempt {attempt}/{retries}  {url}")
            req = urllib.request.Request(
                url,
                headers={
                    "Accept-Encoding": "identity",
                    "User-Agent":      "Lambda-ModelStore/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                cl = resp.headers.get("Content-Length")
                if cl:
                    log.info(f"  size: {int(cl)/1_048_576:.1f} MB")
                return resp.read()

        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning(
                f"[cf ⚠️ ] attempt {attempt}/{retries} failed "
                f"({type(exc).__name__}: {exc}) — retry in {wait}s"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to download '{url}' after {retries} attempts"
    ) from last_exc


# =============================================================================
# PRIVATE — extraction
# =============================================================================

def _extract_bytes_to_dir(raw: bytes, dest: str) -> None:
    """
    Atomically extract an in-memory .tar.gz archive to dest.

    Uses a staging directory to ensure dest is never left in a partial
    state.  Strips a single top-level directory from the archive if
    present (standard convention for model tarballs).

    Uses tarfile mode="r|gz" (streaming) on the BytesIO buffer — this
    avoids loading the decompressed content a second time, keeping peak
    RAM at 1× archive size.
    """
    staging = dest + ".__staging"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r|gz") as tf:
            _safe_extract(tf, staging)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(staging, dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _safe_extract(tf: tarfile.TarFile, dest: str) -> None:
    """
    Extract TarFile members to dest.

    - Strips a single top-level directory if the first member is a
      directory whose name matches the prefix of all other members.
    - Skips any member whose resolved path escapes dest (path-traversal
      guard) with a warning.
    """
    dest = os.path.realpath(dest)
    strip_pfx: str | None = None

    for member in tf:
        name = member.name.strip()
        if not name or name == ".":
            continue

        # Determine strip prefix from the very first member
        if strip_pfx is None:
            first_part = name.split("/")[0]
            if member.isdir() and name.rstrip("/") == first_part:
                strip_pfx = first_part.rstrip("/") + "/"
                continue   # skip the top-level dir entry itself
            else:
                strip_pfx = ""   # no stripping

        if strip_pfx and name.startswith(strip_pfx):
            name = name[len(strip_pfx):]
        if not name:
            continue

        # Path-traversal guard
        resolved = os.path.realpath(os.path.join(dest, name))
        if not (resolved == dest or resolved.startswith(dest + os.sep)):
            log.warning(
                f"[security] Skipping unsafe tar member: {member.name!r}"
            )
            continue

        member.name = name
        tf.extract(member, dest, set_attrs=False)


def _dir_has_files(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and any(f.is_file() for f in p.rglob("*"))


# =============================================================================
# MODEL CLEANUP
# =============================================================================

def cleanup_models() -> None:
    """
    Wipe /tmp/models to free space between invocations.

    Not called automatically — invoke explicitly when /tmp housekeeping is needed
    (e.g. in a maintenance or seeding Lambda, not in the inference hot path where
    the model is baked into /opt/models/ and /tmp is not used).
    """
    import shutil

    target = "/tmp/models"
    if os.path.exists(target):
        size_mb = sum(
            f.stat().st_size
            for f in Path(target).rglob("*")
            if f.is_file()
        ) // 1_048_576
        shutil.rmtree(target)
        log.info(f"cleanup_models: removed {target}  ({size_mb} MB freed)")
    else:
        log.debug("cleanup_models: /tmp/models not present — skip")
