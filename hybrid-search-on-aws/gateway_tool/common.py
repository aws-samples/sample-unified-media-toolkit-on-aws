"""
common.py — shared helpers for the two gateway dispatchers.

Both lambda_function.py (normal target) and admin_function.py (admin target)
ship in ONE flat zip (build-deploy-s3-dist.sh); this module is the single
home for the pieces they share. The collection-id rule and the S3 Vectors
index-name derivation mirror lib/collections (the container-side copies) —
keep them in sync; the gateway zip deliberately excludes lib/.
"""
from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger()

TOOL_DELIMITER = "___"

# Mirrors lib/collections._COLLECTION_ID_RE / _RESERVED_IDS. Doc-type names
# are reserved to keep the per-collection S3 namespace unambiguous.
COLLECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,47}$")
RESERVED_IDS     = {"schema", "shot", "scene", "segment", "utterance", "topic"}

# Mirrors lib/input_schema.MAX_INLINE_DOCUMENTS — enforced at submit time so
# an over-cap batch never comes back as an async failure.
MAX_INLINE_DOCUMENTS = 100


def error(message: str, code: str) -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}


def resolve_tool(context) -> str:
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        return raw.split(TOOL_DELIMITER, 1)[1] if TOOL_DELIMITER in raw else raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("No bedrockAgentCoreToolName in client context")
        return ""


def warn_idempotency_token(event) -> str | None:
    """Log line for a tool handed an idempotency token it cannot honour.

    NO tool on either hybrid-search target honours one, which is why this file
    carries the check rather than the sibling workflows' `workflow_identity.py`
    (installed and then removed here in Phase 1 — this target derives no
    `workflow_id`, so the rest of that module would be dead code in the zip).

    `index_content` names its execution `uuid4()` on purpose: indexing is an
    accumulating write to a shared collection, so two calls with different
    documents must both run, and collapsing them on an input hash would silently
    drop the second.  Its dedup is content-addressed one layer down —
    `doc_id = md5(collection:asset:doc_type:start:end:chunk)` — which already
    makes a retry safe without the caller remembering anything.  `search_content`
    is a query and must reflect the collection as it is now;
    `create_collection`, `configure_search_backend` and `delete_collection` are
    condition-guarded sync mutations.

    A WARNING, deliberately not an error: the caller is an LLM, a hallucinated
    parameter name is one of its routine failure modes, and failing an otherwise
    correct request over a field that could not have changed the answer is the
    worse trade.  This exists to make "ignored" auditable in CloudWatch rather
    than invisible, and covers the retired `uuid` spelling for the same reason.
    """
    if "uuid" in event:
        return ("`uuid` is not a parameter of this tool and was ignored; it "
                "names nothing here and no response echoes it")
    if event.get("idempotency_token"):
        return ("idempotency_token was ignored: no hybrid-search tool "
                "deduplicates on one — indexing dedups per-document on a "
                "content hash, and reads reflect current state")
    return None


def valid_collection_id(collection_id: str) -> bool:
    return bool(COLLECTION_ID_RE.match(collection_id)) and collection_id not in RESERVED_IDS


# One message beside the one rule — index_content and create_collection both
# refuse with it, and a hand-copied pair would drift the moment the grammar
# or the reserved set changes; deriving the list from RESERVED_IDS keeps the
# message honest ('schema' is reserved too, and is not a doc_type).
COLLECTION_ID_RULE_MSG = ("collection_id must match [a-z0-9][a-z0-9_-]{2,47} "
                          "and not be a reserved name "
                          f"({', '.join(sorted(RESERVED_IDS))})")


def sanitize_cause(cause: str, error: str = "") -> str:
    """
    A failed Lambda task's SFN `cause` is JSON with errorMessage, errorType
    and the FULL Python stackTrace (module paths, table names). Surface only
    type + message to the calling agent; the trace stays in CloudWatch.
    NEVER falls back to the raw cause — SFN truncates cause at 32 KB, so a
    truncated-mid-JSON trace would fail parsing and leak through a raw
    fallback. Unparseable cause → the SFN error code alone.
    """
    import json as _json
    try:
        c = _json.loads(cause or "")
        msg = c.get("errorMessage") or ""
        typ = c.get("errorType") or ""
        if msg or typ:
            out = f"{typ}: {msg}" if typ and msg else (typ or msg)
            return out[:500]
    except (ValueError, TypeError, AttributeError):
        pass
    return error or "execution failed — see CloudWatch logs"


def validate_retrieval_paths(paths) -> str | None:
    """
    Shape-check agent-supplied retrieval paths BEFORE they reach DynamoDB:
    {"visual": {"shot_ann": 0.7}} — outer dict, inner dicts, finite
    non-negative numeric weights. Returns an error message, or None if valid.
    """
    import math
    if not isinstance(paths, dict):
        return "retrieval paths must be an object: {path: {sub_query: weight}}"
    for p, weights in paths.items():
        if not isinstance(weights, dict):
            return f"retrieval path {p!r} must map sub-queries to weights"
        for k, v in weights.items():
            # isfinite rejects inf (json.loads accepts "Infinity") and NaN;
            # the 1000 cap keeps str(v) inside DynamoDB's N-type range and
            # weights are [0,1]-scale by convention anyway.
            if (isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) or v < 0 or v > 1000):
                return (f"weight {p}.{k} = {v!r} invalid — must be a "
                        f"finite number in [0, 1000]")
    return None


def paths_to_ddb(paths: dict) -> dict:
    """{"visual": {"shot_ann": 0.7}} -> DynamoDB M/N attribute value."""
    return {"M": {p: {"M": {k: {"N": str(v)} for k, v in w.items()}}
                  for p, w in (paths or {}).items()}}


def unmarshal_paths(item: dict, attr: str) -> dict:
    """The READ half of paths_to_ddb — DynamoDB M/N retrieval-paths attribute
    -> {"visual": {"shot_ann": 0.7}}.  Lives here beside the write half so
    both dispatchers share ONE shape definition.  An absent attribute is the
    documented empty fallback; a PRESENT-but-malformed one raises ValueError
    — tolerance (degrade + mark) is a per-caller decision, never silent."""
    raw = item.get(attr)
    if raw is None:
        return {}
    if not isinstance(raw, dict) or "M" not in raw:
        raise ValueError(f"{attr}: expected a DDB M map, got "
                         f"{sorted(raw)[:3] if isinstance(raw, dict) else type(raw).__name__}")
    try:
        return {p: {k: float(v["N"]) for k, v in wm["M"].items()}
                for p, wm in raw["M"].items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{attr}: malformed entry — {exc!r}") from exc


def s3v_index_name(collection_id: str, table_key: str) -> str:
    """
    MUST mirror lib/collections.s3v_index_name byte-for-byte (the gateway
    zip excludes lib/). Deterministic 63-char-capped S3 Vectors index name:
    sanitized prefix truncated, hash never truncated.
    """
    doc_type = table_key.split(":", 1)[0]
    h8       = hashlib.sha256(f"{collection_id}:{table_key}".encode()).hexdigest()[:8]
    prefix   = f"{collection_id}--{doc_type}".lower().replace("_", "-")[:63 - 10]
    return f"{prefix}--{h8}"
