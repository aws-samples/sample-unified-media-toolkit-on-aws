"""
AgentCore Gateway tool dispatcher — hybrid-search NORMAL-caller target.

Tools (README.md Design record §5):
  index_content     — contracts (inline | S3 JSONL manifest) → Index SFN;
                      async, returns job_id, poll via get_job_status
  search_content    — sync: starts the Search SFN and polls to completion
                      (same pattern the M2C API Lambda uses)
  create_collection — conditional create-if-absent (normal callers may
                      CREATE a new collection, never edit an existing one)
  list_collections  — registry listing + cheap DDB state summary
  get_job_status    — poll an Index SFN execution
  describe_tool     — version + ARNs + the output contract (field-level
                      examples of the two INLINE result shapes — the
                      search-result docs and the index-status counts —
                      from output_contract.json in this zip)

Admin tools (configure_search_backend, delete_collection) live in a
SEPARATE Lambda (admin_function.py) behind a separate gateway target and
IAM role — the admin/normal split is an IAM boundary, never a payload flag.

AgentCore passes the tool name via
context.client_context.custom["bedrockAgentCoreToolName"] as
"{target}___{tool}".
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid as _uuid_mod

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from common import COLLECTION_ID_RULE_MSG, MAX_INLINE_DOCUMENTS, \
    error as _error, \
    paths_to_ddb as _paths_to_ddb, resolve_tool as _resolve_tool, \
    sanitize_cause as _sanitize_cause, unmarshal_paths as _unmarshal_paths, \
    valid_collection_id, \
    warn_idempotency_token as _warn_idempotency_token

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_ddb = boto3.client("dynamodb")

INDEX_SFN_ARN     = os.environ["INDEX_STATE_MACHINE_ARN"]
SEARCH_SFN_ARN    = os.environ["SEARCH_STATE_MACHINE_ARN"]
COLLECTION_TABLE  = os.environ["COLLECTION_TABLE"]
INDEX_STATE_TABLE = os.environ["INDEX_STATE_TABLE"]
PROXY_BUCKET      = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION      = os.environ.get("TOOL_VERSION", "unknown")
WORKFLOW_CONFIG_KEY = os.environ.get(
    "WORKFLOW_CONFIG_KEY", "_workflowconfig/hybrid-search.json")

_SEARCH_POLL_SECS = 0.6
_SEARCH_TIMEOUT_SECS = 55  # stay under typical gateway/Lambda budgets
_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$")

# ── Output contract ─────────────────────────────────────────────────────────
# output_contract.json ships in this zip (the build script pins the member) and
# is the field-level contract for this target's INLINE result shapes: the
# per-document fields search_content returns and the index-status counts
# get_job_status returns for an index_content job.  Unlike the file-emitting
# siblings there is no outputs[] `desc` join here — this target writes no
# consumer-facing output files, so describe_tool is the contract's only
# surface.  Static per TOOL_VERSION, loaded once at cold start.  A missing or
# unparseable file must not take search/index/status down with it, so it
# degrades to None with an ERROR log; the build script's member check is what
# makes that state unreachable in a real deployment.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "output_contract.json")) as _f:
        _OUTPUT_CONTRACT = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("output_contract.json unavailable: %s", _exc)
    _OUTPUT_CONTRACT = None

# input_examples.json is the INPUT twin of the output contract: one minimal
# valid request per tool, served by describe_tool so an agent gets a
# known-good starting call instead of guessing shapes from prose. Unit
# tests round-trip the index_content and search_content examples through
# this handler's own validation, so an example that drifts from the real
# contract fails the build. Same degradation contract as the output
# contract: never takes index/search/status down with it.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "input_examples.json")) as _f:
        _INPUT_EXAMPLES = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("input_examples.json unavailable: %s", _exc)
    _INPUT_EXAMPLES = None


def lambda_handler(event: dict, context) -> dict:
    # keys only — the values can carry callback task tokens and
    # customer-named S3 uris, neither of which belongs in a log group
    logger.info("Event keys: %s", sorted(event.keys()))
    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    # NO tool on this target honours an idempotency token — indexing dedups
    # per-document on a content hash, searches and reads must reflect current
    # state.  Accept and log rather than reject: the caller is an LLM and a
    # hallucinated parameter must not fail an otherwise-correct request.
    _token_note = _warn_idempotency_token(event)
    if _token_note:
        logger.warning("%s: %s", tool, _token_note)

    if tool == "index_content":
        return _index_content(event)
    if tool == "search_content":
        return _search_content(event)
    if tool == "create_collection":
        return _create_collection(event)
    if tool == "list_collections":
        return _list_collections(event)
    if tool == "get_job_status":
        return _get_job_status(event)
    if tool == "describe_tool":
        resp = {
            "version": TOOL_VERSION,
            "index_state_machine_arn": INDEX_SFN_ARN,
            "search_state_machine_arn": SEARCH_SFN_ARN,
            "description": ("Hybrid (semantic + keyword) search over media "
                            "metadata: LanceDB FTS + LanceDB/S3-Vectors ANN, "
                            "multi-collection, contract-driven indexing"),
        }
        # Field-level contract for this target's INLINE result shapes — the
        # per-document fields search_content returns and the index-status
        # counts get_job_status returns (compact example per entry).  Static
        # per TOOL_VERSION, so a caller caches it for the session and CI can
        # pin it without running a job.  Deliberately here and not in
        # search_content/get_job_status: examples are per-version data and
        # would otherwise be re-delivered with every call.
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp
    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ── index_content ──────────────────────────────────────────────────────────────

def _index_content(event: dict) -> dict:
    documents = event.get("documents") or []
    manifest  = (event.get("documents_s3uri") or "").strip()
    if bool(documents) == bool(manifest):
        return _error("exactly one of documents | documents_s3uri required",
                      "INVALID_PARAMETER")
    if documents and (not isinstance(documents, list)
                      or not all(isinstance(d, dict) for d in documents)):
        # Reject at submit time — otherwise the SFN fails async and the agent
        # only learns via get_job_status.
        return _error("documents must be a JSON array of contract objects",
                      "INVALID_PARAMETER")
    if documents and len(documents) > MAX_INLINE_DOCUMENTS:
        # Same submit-time principle; StatePrepareDocuments re-checks, but an
        # over-cap batch must not come back as an async failure.
        return _error(
            f"{len(documents)} inline documents exceeds {MAX_INLINE_DOCUMENTS} "
            f"— use documents_s3uri (JSONL manifest) for bulk submissions",
            "INVALID_PARAMETER")
    if manifest and not manifest.startswith(f"s3://{PROXY_BUCKET}/"):
        return _error(
            f"documents_s3uri must be under s3://{PROXY_BUCKET}/ — the stack "
            f"only reads its proxy bucket", "FOREIGN_BUCKET")

    collection_id = (event.get("collection_id") or "").strip()
    if collection_id and not valid_collection_id(collection_id):
        return _error(COLLECTION_ID_RULE_MSG, "INVALID_PARAMETER")
    # Same submit-time principle: an unknown collection must not come back as
    # N async execution failures minutes later.  One registry read; on any
    # DDB/transport error fall through — the async path still enforces it,
    # and a registry hiccup must not block an otherwise-valid submit.
    # ConsistentRead: create_collection → index_content is a scripted
    # back-to-back sequence (the index-video skill), and an eventually
    # consistent read of the row that was just put would hard-refuse a
    # collection that exists.
    try:
        resolved = collection_id or "default"
        item = _ddb.get_item(TableName=COLLECTION_TABLE,
                             Key={"collection_id": {"S": resolved}},
                             ConsistentRead=True).get("Item")
        if not item:
            # the seeding clause only applies to the default collection — on
            # a named id it sends the caller down an irrelevant remedy
            hint = (" or check the deployment seeded the 'default' "
                    "collection" if resolved == "default" else "")
            return _error(
                f"Unknown collection {resolved!r} — create it first "
                f"(create_collection){hint}", "UNKNOWN_COLLECTION")
    except (ClientError, BotoCoreError) as exc:
        # Fail-open by design, but never silently: a PERMANENT misconfig
        # (e.g. lost dynamodb:GetItem) would otherwise disable this guard
        # forever with nothing distinguishing "passed" from "never ran".
        logger.warning("collection existence check skipped (%s) — "
                       "deferring to the async path", exc)
    sfn_input: dict = {
        "collection_id":       collection_id,          # "" -> default (resolved downstream)
        "documents":           documents,
        "documents_s3uri":    manifest,
        "asset_metadata":      event.get("asset_metadata") or {},
        "workflow_config_uri": f"s3://{PROXY_BUCKET}/{WORKFLOW_CONFIG_KEY}",
    }
    name = str(_uuid_mod.uuid4())
    try:
        resp = _sfn.start_execution(
            stateMachineArn=INDEX_SFN_ARN, name=name,
            input=json.dumps(sfn_input))
    except ClientError as exc:
        return _error(f"Failed to start indexing: "
                      f"{exc.response['Error']['Message']}", "SFN_START_FAILED")
    return {"job_id": resp["executionArn"], "status": "RUNNING",
            "collection_id": collection_id or "default",
            "num_inline_documents": len(documents),
            "documents_s3uri": manifest}


# ── search_content (sync) ──────────────────────────────────────────────────────

def _search_content(event: dict) -> dict:
    search_term = (event.get("search_term") or "").strip()
    image_uri   = (event.get("search_image_s3uri") or "").strip()
    if bool(search_term) == bool(image_uri):
        return _error("exactly one of search_term | search_image_s3uri required",
                      "INVALID_PARAMETER")
    if image_uri and not image_uri.startswith("s3://"):
        return _error("search_image_s3uri must be an s3:// uri", "INVALID_PARAMETER")
    try:
        k = max(1, min(50, int(event.get("k") or 10)))
    except (TypeError, ValueError):
        return _error("k must be an integer", "INVALID_PARAMETER")

    sfn_input = {
        "collection_id":       (event.get("collection_id") or "").strip(),
        "search_term":         search_term,
        "search_image_s3uri":    image_uri,
        "k":                   k,
        "workflow_config_uri": f"s3://{PROXY_BUCKET}/{WORKFLOW_CONFIG_KEY}",
    }
    try:
        resp = _sfn.start_execution(
            stateMachineArn=SEARCH_SFN_ARN,
            input=json.dumps(sfn_input))
    except ClientError as exc:
        return _error(f"Failed to start search: "
                      f"{exc.response['Error']['Message']}", "SFN_START_FAILED")

    arn      = resp["executionArn"]
    deadline = time.time() + _SEARCH_TIMEOUT_SECS
    while time.time() < deadline:
        d = _sfn.describe_execution(executionArn=arn)
        status = d["status"]
        if status == "SUCCEEDED":
            return json.loads(d["output"])
        if status in ("FAILED", "TIMED_OUT", "ABORTED"):
            return _error(f"Search execution {status}: "
                          f"{_sanitize_cause(d.get('cause', ''), d.get('error', ''))}",
                          "SEARCH_FAILED")
        time.sleep(_SEARCH_POLL_SECS)
    return _error(f"Search timed out after {_SEARCH_TIMEOUT_SECS}s "
                  f"(execution still running: {arn})", "SEARCH_TIMEOUT")




# ── create_collection (create-if-absent — normal-caller exception) ─────────────

def _default_paths_from_default_collection(scopes: list[str]) -> tuple[dict, dict]:
    """
    New collections inherit retrieval weights from the seeded 'default'
    collection record, filtered to their active scopes — ONE source of
    truth (workflow_config/hybrid-search.json -> CopySchema seed -> here),
    no hardcoded copy. Falls back to empty maps if the default record is
    missing (search then fails loudly with 'no retrieval_paths configured').
    """
    resp = _ddb.get_item(TableName=COLLECTION_TABLE,
                         Key={"collection_id": {"S": "default"}})
    item = resp.get("Item") or {}

    def unmarshal_paths(attr):
        # strict by nature: unmarshal_paths raises on a malformed attribute,
        # and the create path converts that into MALFORMED_DEFAULT_RECORD —
        # inheriting {} would birth a weightless, searchable-in-name-only
        # collection.
        return _unmarshal_paths(item, attr)

    def filter_paths(paths):
        out = {}
        for path, weights in paths.items():
            kept = {k: v for k, v in weights.items()
                    if k.split("_", 1)[0] in scopes}
            if kept:
                out[path] = kept
        return out

    return (filter_paths(unmarshal_paths("retrieval_paths")),
            filter_paths(unmarshal_paths("image_retrieval_paths")))


def _create_collection(event: dict) -> dict:
    collection_id = (event.get("collection_id") or "").strip()
    if not valid_collection_id(collection_id):
        return _error(COLLECTION_ID_RULE_MSG, "INVALID_PARAMETER")

    active_scopes = event.get("active_scopes") or []
    if not isinstance(active_scopes, list) or not active_scopes:
        return _error("active_scopes (non-empty list of doc_types) required",
                      "INVALID_PARAMETER")

    # schema_map defaults by convention: scope -> default_{scope}.json.
    # Explicit override allowed for non-default schemas.
    schema_map = dict(event.get("schema_map") or {})
    for scope in active_scopes:
        schema_map.setdefault(scope, f"_lancedb_index/schema/default_{scope}.json")

    # A freshly created collection must be SEARCHABLE without waiting for an
    # admin weight edit (weight-freeze governs EDITS, not creation defaults):
    # inherit weights from the seeded default collection, scope-filtered.
    try:
        retrieval_paths, image_retrieval_paths = \
            _default_paths_from_default_collection(active_scopes)
    except ValueError as exc:
        # Name the condition and the right escalation — repair lives on the
        # ADMIN target, which this caller's gateway may not even attach, so
        # the message must not read like an instruction to call a tool.
        return _error(
            f"the seeded 'default' collection record is malformed ({exc}) — "
            f"an operator must repair it on the admin target before new "
            f"collections can inherit weights from it",
            "MALFORMED_DEFAULT_RECORD")
    if not retrieval_paths:
        # An ABSENT default record (or one with no weights for these scopes)
        # is the same outcome as a malformed one — a weightless collection
        # that is searchable in name only — and must refuse the same way.
        return _error(
            f"no retrieval weights to inherit for scopes {active_scopes} — "
            f"the seeded 'default' collection record is missing, carries no "
            f"retrieval_paths, or has no weights for these scopes; the "
            f"deployment seed (or an operator repair on the admin target) "
            f"must provide them first", "MISSING_DEFAULT_WEIGHTS")

    now  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    item = {
        "collection_id":         {"S": collection_id},
        "name":                  {"S": (event.get("name") or collection_id)},
        "active_scopes":         {"L": [{"S": s} for s in active_scopes]},
        "schema_map":            {"M": {k: {"S": v} for k, v in schema_map.items()}},
        "retrieval_paths":       _paths_to_ddb(retrieval_paths),
        "image_retrieval_paths": _paths_to_ddb(image_retrieval_paths),
        "weights_locked":        {"BOOL": False},
        "created_by":            {"S": "gateway"},
        "created_at":            {"S": now},
        "updated_at":            {"S": now},
    }
    try:
        _ddb.put_item(TableName=COLLECTION_TABLE, Item=item,
                      ConditionExpression="attribute_not_exists(collection_id)")
        created = True
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        created = False  # losing the race = collection exists = success
    return {"collection_id": collection_id, "created": created,
            "active_scopes": active_scopes, "schema_map": schema_map,
            "retrieval_paths": retrieval_paths,
            "note": ("" if created else
                     "already existed — no changes applied; weight edits "
                     "require the admin tool")}


# ── list_collections ───────────────────────────────────────────────────────────

def _list_collections(event: dict) -> dict:
    collections = []
    kwargs: dict = {"TableName": COLLECTION_TABLE}
    while True:
        resp = _ddb.scan(**kwargs)
        for it in resp.get("Items", []):
            rec = {
                "collection_id":  it["collection_id"]["S"],
                "name":           it.get("name", {}).get("S", ""),
                "active_scopes":  [s["S"] for s in it.get("active_scopes", {}).get("L", [])],
                "weights_locked": it.get("weights_locked", {}).get("BOOL", False),
                "created_at":     it.get("created_at", {}).get("S", ""),
                # schema_map rides along for the same reason as the paths:
                # the admin guard's remedy says "read the current set from
                # list_collections", so every guarded field must be readable
                "schema_map":     {k: v.get("S", "") for k, v in
                                   it.get("schema_map", {}).get("M", {}).items()},
            }
            # The read half of configure_search_backend's full-replacement
            # write: without these, a weights migration cannot compose the
            # complete set and a partial payload silently wipes the paths it
            # omits (measured live 2026-10-13).  Tolerant with a VISIBLE
            # marker: one malformed record must not fail the listing, but a
            # silent {} would dead-end the migration recipe that reads here.
            for attr in ("retrieval_paths", "image_retrieval_paths"):
                try:
                    rec[attr] = _unmarshal_paths(it, attr)
                except ValueError as exc:
                    logger.warning("collection %r: %s — degraded to {} in "
                                   "the listing and marked %s_malformed",
                                   rec["collection_id"], exc, attr)
                    rec[attr] = {}
                    rec[f"{attr}_malformed"] = True
            collections.append(rec)
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    # Cheap DDB-only index summary (resolved decision: no live count_rows()).
    states: dict[str, list[dict]] = {}
    kwargs = {"TableName": INDEX_STATE_TABLE,
              "KeyConditionExpression": "pk = :p",
              "ExpressionAttributeValues": {":p": {"S": "state"}}}
    while True:
        resp = _ddb.query(**kwargs)
        for it in resp.get("Items", []):
            sk  = it["sk"]["S"]
            cid = sk.partition(":")[0]
            states.setdefault(cid, []).append({
                "index_key":       sk,
                "last_indexed_at": it.get("last_indexed_at", {}).get("S", ""),
                "pending_docs":    int(it.get("pending_docs_since_rebuild", {}).get("N", "0")),
            })
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    for c in collections:
        c["indexes"] = states.get(c["collection_id"], [])
    return {"collections": collections}


# ── get_job_status ─────────────────────────────────────────────────────────────

def _get_job_status(event: dict) -> dict:
    job_id = (event.get("job_id") or "").strip()
    if not _SFN_ARN_RE.match(job_id):
        return _error("job_id must be a Step Functions execution ARN",
                      "INVALID_JOB_ID")
    try:
        d = _sfn.describe_execution(executionArn=job_id)
    except _sfn.exceptions.ExecutionDoesNotExist:
        return _error(f"No execution found: {job_id}", "JOB_NOT_FOUND")
    except ClientError as exc:
        return _error(f"Failed to describe job: "
                      f"{exc.response['Error']['Message']}", "SFN_DESCRIBE_FAILED")
    status = d["status"]
    result: dict = {"job_id": job_id, "status": status}
    if status == "SUCCEEDED":
        try:
            out = json.loads(d.get("output") or "{}")
            result["collection_id"] = out.get("collection_id")
            result["prepared"]      = out.get("prepared")
            result["skipped"]       = out.get("skipped")
            result["indexed"]       = sum(
                int(r.get("index_result", {}).get("indexed", 0))
                for r in out.get("index_results") or [])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    elif status in ("FAILED", "TIMED_OUT", "ABORTED"):
        result["error"] = _sanitize_cause(d.get("cause", ""),
                                          d.get("error", "")) or status
    return result
