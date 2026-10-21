import logging
import os
import posixpath
from datetime import datetime, timedelta, timezone

import boto3
from botocore.client import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROXY_BUCKET = os.environ["PROXY_BUCKET"]
# Workflow outputs, plus the tts voice-catalog previews (voice records carry
# a preview under _model_artefacts/voice_catalog/ — the IAM policy grants the
# same two prefixes; widen both together or the grant is unreachable).
_ALLOWED_PREFIXES = ("micro-workflow/", "_model_artefacts/voice_catalog/")
_MIN_EXPIRY, _MAX_EXPIRY, _DEFAULT_EXPIRY = 60, 3600, 3600
# A presigned URL here is ~1.3 KB (the assumed-role session
# token in the query string dominates), so a full batch is
# ~16 K tokens of agent context. The cap is a context budget,
# not a compute one — generate_presigned_url is local crypto.
_MAX_URIS = 50

s3 = boto3.client(
    "s3",
    region_name=os.environ.get("AWS_REGION"),
    config=Config(signature_version="s3v4"),
)

# Two tools, one Lambda, deliberately different response shapes:
#
#   get_download_url   one uri  -> {"url": ..., "expires_at": ...}
#   get_download_urls  N uris   -> {"urls": [...], "count": N, "expires_at": ...}
#
# The singular tool is the original contract and is frozen. Batching is a
# SECOND tool rather than a widening of the first, because a caller reading
# result["url"] cannot be rescued by a compatible request parameter — the
# break is in the response, not the request.

def handler(event, context):
    tool = _resolve_tool(context)

    # Neither tool starts an execution, so neither honours an idempotency token
    # (nor the retired `uuid` spelling): presigning is a pure function of the key
    # and the expiry, and calling it twice costs nothing to deduplicate.  Accept
    # and log rather than reject — the caller is an LLM, a hallucinated parameter
    # name is a routine failure mode, and failing an otherwise-correct request
    # over a field that could not have changed the URL is the worse trade.  This
    # file carries the check inline because the media-toolkit-gateway gateway zip has
    # no shared module; the sibling workflows use
    # `workflow_identity.warn_idempotency_token`.
    if "uuid" in event or event.get("idempotency_token"):
        logger.warning(
            "%s: idempotency_token/uuid was ignored — presigning starts no "
            "execution and is deterministic in the key and expiry", tool)

    if tool == "get_download_url":
        return _get_download_url(event)
    if tool == "get_download_urls":
        return _get_download_urls(event)
    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")

def _resolve_key(raw):
    """Validate one uri/key. Returns (key, None) or (None, message)."""
    raw = (raw or "").strip()
    if not raw:
        return None, "empty uri"

    if raw.startswith("s3://"):
        rest = raw[5:]
        bucket, _, key = rest.partition("/")
        if bucket != PROXY_BUCKET:
            return None, f"bucket must be {PROXY_BUCKET}"
    else:
        key = raw

    # prefix check BEFORE normpath — normpath strips the
    # trailing slash that marks a prefix entry
    if key.endswith("/"):
        return None, "cannot presign a prefix — pass an object key"
    key = posixpath.normpath(key.lstrip("/"))
    if (key.startswith("..") or "/../" in key
            or not key.startswith(_ALLOWED_PREFIXES)):
        return None, ("key must be under one of: "
                      + ", ".join(_ALLOWED_PREFIXES))
    return key, None

def _resolve_expiry(event):
    """Clamp expiry_seconds. Returns (expiry, None) or (None, error_dict)."""
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

def _presign(key, expiry):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": PROXY_BUCKET, "Key": key},
        ExpiresIn=expiry,
    )

def _get_download_url(event):
    """One uri -> flat {url, expires_at}. FROZEN CONTRACT.

    Do not accept a uris/batch parameter here and do not nest the
    response: existing callers read result["url"] directly. Anything
    batch-shaped belongs in _get_download_urls.
    """
    raw = event.get("s3uri_or_key")
    if raw is None or raw == "":
        return _error("s3uri_or_key is required", "MISSING_PARAMETER")
    # Type-check before .strip() — an agent that has several uris often
    # passes a list here, and an AttributeError would surface as an
    # unhandled invocation error with a stack trace instead of a message
    # the agent can act on.
    if not isinstance(raw, str):
        return _error(
            f"s3uri_or_key must be a string, not {type(raw).__name__}. "
            "For several uris use get_download_urls.",
            "INVALID_PARAMETER")
    raw = raw.strip()
    if not raw:
        return _error("s3uri_or_key is required", "MISSING_PARAMETER")

    key, why = _resolve_key(raw)
    if why:
        return _error(why, "INVALID_PARAMETER")

    expiry, err = _resolve_expiry(event)
    if err:
        return err

    logger.info("presigned %s expiry=%ds", key, expiry)
    return {"url": _presign(key, expiry), "expires_at": _expires_at(expiry)}

def _get_download_urls(event):
    """N uris -> {urls: [{uri, url | error+message}], count, expires_at}."""
    uris = event.get("uris")

    if isinstance(uris, str):
        # a single string is an easy agent slip — accept it
        uris = [uris]
    elif uris is None:
        return _error(
            f"uris is required (a list of 1-{_MAX_URIS} output object "
            "uris). For exactly one uri, get_download_url is simpler.",
            "MISSING_PARAMETER")
    elif not isinstance(uris, list):
        return _error("uris must be a list of strings", "INVALID_PARAMETER")

    if not uris:
        return _error("uris must not be empty", "INVALID_PARAMETER")
    if len(uris) > _MAX_URIS:
        return _error(
            f"too many uris: {len(uris)} (max {_MAX_URIS}). "
            "Presign only the files you need.",
            "INVALID_PARAMETER")

    # clamp once — every URL in the batch shares one expiry
    expiry, err = _resolve_expiry(event)
    if err:
        return err

    # Per-item errors: one bad uri must not discard the
    # good presigns alongside it.
    urls, ok = [], 0
    for raw in uris:
        if isinstance(raw, str):
            key, why = _resolve_key(raw)
        else:
            key, why = None, f"not a string: {type(raw).__name__}"
        if why:
            logger.warning("skipped %r: %s", raw, why)
            urls.append({"uri": raw, "error": True, "message": why})
            continue
        urls.append({"uri": raw, "url": _presign(key, expiry)})
        ok += 1

    logger.info("presigned %d/%d uris expiry=%ds", ok, len(uris), expiry)
    return {"urls": urls, "count": ok, "expires_at": _expires_at(expiry)}

def _resolve_tool(context):
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        return raw.split("___", 1)[1] if "___" in raw else raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("bedrockAgentCoreToolName not found")
        return ""

def _error(message, code="ERROR"):
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
