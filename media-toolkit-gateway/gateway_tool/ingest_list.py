import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

INGEST_BUCKET = os.environ["INGEST_BUCKET"]

# Page-size defaults are a CONTEXT budget, not a compute one (the same
# reasoning as download_url's _MAX_URIS): every row lands in the calling
# agent's context, and the typical question — "find my file", "what is
# under this folder" — is answered inside the first 20 once prefix and
# extension filtering are applied. next_token pages through the rest.
_DEFAULT_RESULTS, _MAX_RESULTS = 20, 200
# Extension filtering is CLIENT-side (S3 cannot match suffixes), so a
# sparse match over a huge bucket could walk forever. Cap the keys
# scanned per call; the caller continues from next_token.
_MAX_KEYS_SCANNED = 10_000

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))


def handler(event, context):
    tool = _resolve_tool(context)

    # Listing starts no execution and must reflect current state — an
    # idempotency token could not change the answer. Accept and log
    # rather than reject (a hallucinated parameter name is a routine
    # agent failure mode); same inline convention as download_url.py.
    if "uuid" in event or event.get("idempotency_token"):
        logger.warning(
            "%s: idempotency_token/uuid was ignored — listing starts no "
            "execution and reflects current bucket state", tool)

    if tool == "list_files":
        return _list_files(event)
    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


def _list_files(event):
    prefix, err = _resolve_prefix(event.get("prefix"))
    if err:
        return err
    extensions, err = _resolve_extensions(event.get("extension"))
    if err:
        return err
    limit, err = _resolve_max_results(event.get("max_results"))
    if err:
        return err
    start_after = event.get("next_token")
    if start_after is not None and not isinstance(start_after, str):
        return _error("next_token must be the string returned by the "
                      "previous call", "INVALID_PARAMETER")

    # next_token is the last key the previous call RETURNED or SCANNED,
    # resumed via StartAfter — exact even when a page was cut mid-way
    # by max_results, which S3's own ContinuationToken cannot express.
    files, scanned, last_key, exhausted = [], 0, None, True
    kwargs = {"Bucket": INGEST_BUCKET, "Prefix": prefix}
    if start_after:
        kwargs["StartAfter"] = start_after
    while True:
        page = s3.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            key = obj["Key"]
            scanned += 1
            last_key = key
            if key.endswith("/"):
                continue          # zero-byte folder placeholder
            if extensions and not any(key.lower().endswith(e)
                                      for e in extensions):
                continue
            files.append({
                "s3uri": f"s3://{INGEST_BUCKET}/{key}",
                "key": key,
                "size": obj["Size"],
                "last_modified":
                    obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            if len(files) >= limit:
                exhausted = False
                break
        else:
            if page.get("IsTruncated") and scanned < _MAX_KEYS_SCANNED:
                kwargs = {"Bucket": INGEST_BUCKET, "Prefix": prefix,
                          "ContinuationToken":
                              page["NextContinuationToken"]}
                continue
            exhausted = not page.get("IsTruncated")
        break

    result = {"files": files, "count": len(files)}
    if not exhausted and last_key:
        result["next_token"] = last_key
    logger.info("list_files prefix=%r ext=%r -> %d files (%d keys scanned)",
                prefix, extensions, len(files), scanned)
    return result


def _resolve_prefix(raw):
    """Optional key prefix; a full s3:// uri on the ingest bucket is
    accepted and stripped (agents paste uris). Returns (prefix, None)
    or (None, error_dict)."""
    if raw is None:
        return "", None
    if not isinstance(raw, str):
        return None, _error("prefix must be a string", "INVALID_PARAMETER")
    raw = raw.strip()
    if raw.startswith("s3://"):
        bucket, _, key = raw[5:].partition("/")
        if bucket != INGEST_BUCKET:
            return None, _error(f"bucket must be {INGEST_BUCKET}",
                                "INVALID_PARAMETER")
        raw = key
    return raw.lstrip("/"), None


def _resolve_extensions(raw):
    """Optional extension filter: one string or a list; '.mp4', 'mp4'
    and 'MP4' all match the same files. Returns (tuple, None) or
    (None, error_dict)."""
    if raw is None or raw == "" or raw == []:
        return (), None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(e, str) for e in raw):
        return None, _error("extension must be a string like 'mp4' (or a "
                            "list of them)", "INVALID_PARAMETER")
    out = []
    for e in raw:
        e = e.strip().lower().lstrip(".")
        if not e:
            return None, _error("extension must not be empty",
                                "INVALID_PARAMETER")
        out.append("." + e)
    return tuple(out), None


def _resolve_max_results(raw):
    if raw is None:
        return _DEFAULT_RESULTS, None
    try:
        raw = int(raw)
    except (TypeError, ValueError):
        return None, _error("max_results must be an integer",
                            "INVALID_PARAMETER")
    return max(1, min(_MAX_RESULTS, raw)), None


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
