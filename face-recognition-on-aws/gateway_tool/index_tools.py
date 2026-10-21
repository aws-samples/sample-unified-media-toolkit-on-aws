"""
Phase 2 gateway tools — face index management (README.md §19).

Synchronous (this Lambda): create_collection, list_collections, list_faces,
get_face, batch_update_faces.
Asynchronous (start the face-index SM): batch_index_faces,
batch_delete_faces, delete_collection. Async responses return a job_id
(execution ARN) polled via the existing get_job_status tool.

Shipped in the gateway zip alongside lambda_function.py (flat, no lib/) —
constants that must match the container are re-declared here and guarded by
test_units.TestGatewayIndexToolConstants.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid as _uuid_mod

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()

_ddb = boto3.client("dynamodb")
_rek = boto3.client("rekognition")
_s3 = boto3.client("s3")
_sfn = boto3.client("stepfunctions")

FACE_TABLE_NAME = os.environ.get("FACE_TABLE_NAME", "")
INDEX_STATE_MACHINE_ARN = os.environ.get("INDEX_STATE_MACHINE_ARN", "")

# definitions_index.json StartAt — a fresh execution is always in this state, so
# the submit can report it without paying for a GetExecutionHistory call.  Named
# separately from the recognition SM's first state in lambda_function.py: these
# are two different state machines.
_INDEX_FIRST_STATE = "Validate index batch"
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "face-recognition")
DEFAULT_COLLECTION_ID = os.environ.get("DEFAULT_COLLECTION_ID", "")

# Mirrors lib/input_schema.py (guarded by a unit test).
COLLECTION_INDEX_GSI = "collection-index"
UNNAMED_CELEB_SENTINEL = "~unnamed"
MAX_BATCH_UPDATE_FACES = 500
LIST_FACES_MAX_PAGE_SIZE = 100
LIST_FACES_DEFAULT_PAGE_SIZE = 25
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".jfif", ".png", ".webp",
                    ".tif", ".tiff", ".heic", ".heif", ".avif", ".gif")


def celeb_created_sk(celeb: str | None, created_at_ms: int) -> str:
    """MUST stay byte-identical to lib/face_store.celeb_created_sk (guarded
    by TestGatewayIndexToolConstants) — list_faces begins_with filtering and
    rename correctness both depend on this exact format."""
    name = celeb if celeb else UNNAMED_CELEB_SENTINEL
    return f"{name}#{created_at_ms:015d}"


def _parse_bool(value):
    """MCP layers commonly deliver booleans as strings — accept both.
    Returns (parsed_or_None, error_or_None)."""
    if value is None or isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        if value.lower() in ("true", "1", "yes"):
            return True, None
        if value.lower() in ("false", "0", "no"):
            return False, None
    return None, f"expected a boolean, got {value!r}"


def _error(message: str, code: str) -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}


def _unmarshal(item: dict) -> dict:
    return {k: (v.get("S") if "S" in v else
                v.get("N") if "N" in v else v.get("BOOL"))
            for k, v in item.items()}


def _public_record(rec: dict) -> dict:
    """The face record shape returned by list_faces/get_face."""
    out = {k: rec.get(k) for k in (
        "face_id", "collection_id", "celeb", "origin", "workflow_id",
        "source_key", "source_etag", "gender", "age_range", "confidence",
        "external_image_id") if rec.get(k) is not None}
    out["created_at"] = int(rec["created_at"]) if rec.get("created_at") else None
    coord = rec.get("coord")
    if coord:
        out["coord"] = [float(x) for x in str(coord).split(",")]
    for k in ("thumbnail_key", "full_image_key"):
        if rec.get(k):
            out[k.replace("_key", "_uri")] = f"s3://{PROXY_BUCKET}/{rec[k]}"
    return out


# ── create_collection ──────────────────────────────────────────────────────────

def create_collection(event: dict) -> dict:
    collection_id = (event.get("collection_id") or "").strip()
    if not collection_id:
        return _error("collection_id is required", "MISSING_PARAMETER")

    try:
        _rek.create_collection(CollectionId=collection_id)
        return {"collection_id": collection_id, "created": True,
                "face_count": 0, "record_count": 0}
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise

    # Exists: apply the README.md §19 rule against BOTH stores.
    face_count = _rek.describe_collection(
        CollectionId=collection_id)["FaceCount"]
    record_count = _count_records(collection_id)
    if face_count == 0:
        return {"collection_id": collection_id, "created": False,
                "adopted": True, "face_count": 0,
                "record_count": record_count}
    if record_count == 0:
        return _error(
            f"collection {collection_id!r} exists with {face_count} faces "
            f"but none are managed by this stack — refusing to adopt a "
            f"foreign collection", "FOREIGN_COLLECTION")
    result = {"collection_id": collection_id, "created": False,
              "adopted": True, "face_count": face_count,
              "record_count": record_count}
    if face_count != record_count:
        result["warning"] = (
            f"drift: Rekognition has {face_count} faces but DynamoDB has "
            f"{record_count} records — a crashed delete/purge may have left "
            f"asymmetry")
    return result


def _count_records(collection_id: str) -> int:
    total = 0
    kwargs: dict = {
        "TableName": FACE_TABLE_NAME,
        "IndexName": COLLECTION_INDEX_GSI,
        "KeyConditionExpression": "collection_id = :c",
        "ExpressionAttributeValues": {":c": {"S": collection_id}},
        "Select": "COUNT",
    }
    while True:
        resp = _ddb.query(**kwargs)
        total += resp["Count"]
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            return total
        kwargs["ExclusiveStartKey"] = lek


# ── list_collections ───────────────────────────────────────────────────────────

def list_collections(event: dict) -> dict:
    ids: list[str] = []
    kwargs: dict = {}
    while True:
        resp = _rek.list_collections(**kwargs)
        ids.extend(resp.get("CollectionIds") or [])
        token = resp.get("NextToken")
        if not token:
            break
        kwargs["NextToken"] = token

    collections = []
    for cid in ids:
        if cid.startswith("sandbox-"):
            continue  # transient clustering sandboxes are not user-facing
        face_count = _rek.describe_collection(CollectionId=cid)["FaceCount"]
        record_count = _count_records(cid)
        entry = {"collection_id": cid, "face_count": face_count,
                 "record_count": record_count,
                 "is_default": cid == DEFAULT_COLLECTION_ID}
        if face_count != record_count:
            entry["drift"] = True
        collections.append(entry)
    return {"collections": collections}


# ── list_faces / get_face ──────────────────────────────────────────────────────

def list_faces(event: dict) -> dict:
    collection_id = ((event.get("collection_id") or "").strip()
                     or DEFAULT_COLLECTION_ID)
    if not collection_id:
        return _error("collection_id is required", "MISSING_PARAMETER")
    page_size = min(int(event.get("page_size")
                        or LIST_FACES_DEFAULT_PAGE_SIZE),
                    LIST_FACES_MAX_PAGE_SIZE)

    kwargs: dict = {
        "TableName": FACE_TABLE_NAME,
        "IndexName": COLLECTION_INDEX_GSI,
        "KeyConditionExpression": "collection_id = :c",
        "ExpressionAttributeValues": {":c": {"S": collection_id}},
        "Limit": page_size,
    }

    celeb = (event.get("celebrity_name") or "").strip()
    if celeb:
        kwargs["KeyConditionExpression"] += \
            " AND begins_with(celeb_created, :n)"
        kwargs["ExpressionAttributeValues"][":n"] = {"S": f"{celeb}#"}

    filters = []
    origin = (event.get("origin") or "").strip()
    if origin:
        if origin not in ("user", "auto"):
            return _error("origin must be 'user' or 'auto'",
                          "INVALID_PARAMETER")
        filters.append("origin = :o")
        kwargs["ExpressionAttributeValues"][":o"] = {"S": origin}
    named, bool_err = _parse_bool(event.get("named"))
    if bool_err:
        return _error(f"named: {bool_err}", "INVALID_PARAMETER")
    # celeb_created is the GSI SORT KEY — DynamoDB rejects key attributes in
    # FilterExpression (found live by E2E: the earlier string-bool bug had
    # masked it by never activating the filter). Named-ness is equivalent to
    # the celeb attribute existing (register_face omits it when unnamed).
    if named is True:
        filters.append("attribute_exists(celeb)")
    elif named is False:
        filters.append("attribute_not_exists(celeb)")
    workflow_id = (event.get("workflow_id") or "").strip()
    if workflow_id:
        filters.append("workflow_id = :w")
        kwargs["ExpressionAttributeValues"][":w"] = {"S": workflow_id}
    if filters:
        kwargs["FilterExpression"] = " AND ".join(filters)

    token = (event.get("next_token") or "").strip()
    if token:
        try:
            kwargs["ExclusiveStartKey"] = json.loads(
                base64.b64decode(token.encode()).decode())
        except Exception:                                     # noqa: BLE001
            return _error("invalid next_token", "INVALID_PARAMETER")

    resp = _ddb.query(**kwargs)
    faces = [_public_record(_unmarshal(i)) for i in resp.get("Items") or []]
    out: dict = {"collection_id": collection_id, "faces": faces}
    lek = resp.get("LastEvaluatedKey")
    if lek:
        out["next_token"] = base64.b64encode(
            json.dumps(lek).encode()).decode()
    return out


def get_face(event: dict) -> dict:
    face_id = (event.get("face_id") or "").strip()
    if not face_id:
        return _error("face_id is required", "MISSING_PARAMETER")
    resp = _ddb.get_item(
        TableName=FACE_TABLE_NAME,
        Key={"pk": {"S": f"face#{face_id}"}, "sk": {"S": "record"}})
    item = resp.get("Item")
    if not item:
        # Late-binding consumers benefit from knowing WHY a face is gone.
        tomb = _ddb.get_item(
            TableName=FACE_TABLE_NAME,
            Key={"pk": {"S": f"deleted#{face_id}"}, "sk": {"S": "record"}})
        if tomb.get("Item"):
            t = _unmarshal(tomb["Item"])
            return {"face_id": face_id, "deleted": True,
                    "reason": t.get("reason"),
                    "celeb": t.get("celeb"),
                    "deleted_at": int(t["deleted_at"])
                    if t.get("deleted_at") else None}
        return _error(f"face {face_id} not found", "FACE_NOT_FOUND")
    return _public_record(_unmarshal(item))


# ── batch_update_faces (rename/tag — late binding) ─────────────────────────────

def batch_update_faces(event: dict) -> dict:
    updates = event.get("updates")
    if not isinstance(updates, list) or not updates:
        return _error("updates must be a non-empty list of "
                      "{face_id, celebrity_name}", "MISSING_PARAMETER")
    if len(updates) > MAX_BATCH_UPDATE_FACES:
        return _error(f"at most {MAX_BATCH_UPDATE_FACES} updates per call",
                      "INVALID_PARAMETER")
    for i, u in enumerate(updates):
        if not (isinstance(u, dict) and (u.get("face_id") or "").strip()
                and (u.get("celebrity_name") or "").strip()):
            return _error(f"updates[{i}] must have face_id and "
                          f"celebrity_name", "INVALID_PARAMETER")

    results = []
    for u in updates:
        face_id = u["face_id"].strip()
        name = u["celebrity_name"].strip()
        rec = _ddb.get_item(
            TableName=FACE_TABLE_NAME,
            Key={"pk": {"S": f"face#{face_id}"}, "sk": {"S": "record"}}
        ).get("Item")
        if not rec:
            results.append({"face_id": face_id, "updated": False,
                            "reason": "not_found"})
            continue
        created_at = int(_unmarshal(rec).get("created_at") or 0)
        try:
            _ddb.update_item(
                TableName=FACE_TABLE_NAME,
                Key={"pk": {"S": f"face#{face_id}"}, "sk": {"S": "record"}},
                UpdateExpression="SET celeb = :c, celeb_created = :cc",
                ConditionExpression="attribute_exists(pk)",
                ExpressionAttributeValues={
                    ":c": {"S": name},
                    ":cc": {"S": celeb_created_sk(name, created_at)},
                })
        except ClientError as exc:
            # Read-then-update race: the face can be deleted between GetItem
            # and UpdateItem (e.g. a concurrent batch_delete for the same
            # celebrity). Per-item outcome, never abort the batch with
            # earlier updates already committed and the results list lost.
            if exc.response["Error"]["Code"] == \
                    "ConditionalCheckFailedException":
                results.append({"face_id": face_id, "updated": False,
                                "reason": "deleted_concurrently"})
                continue
            raise
        results.append({"face_id": face_id, "updated": True,
                        "celebrity_name": name})
    updated = sum(1 for r in results if r["updated"])
    return {"updated": updated, "failed": len(results) - updated,
            "results": results}


# ── Async ops: start the face-index state machine ──────────────────────────────

def batch_index_faces(event: dict) -> dict:
    collection_id = ((event.get("collection_id") or "").strip()
                     or DEFAULT_COLLECTION_ID)
    if not collection_id:
        return _error("collection_id is required", "MISSING_PARAMETER")
    prefix_uri = (event.get("s3_prefix") or "").strip()
    manifest_uri = (event.get("manifest_s3uri") or "").strip()
    if bool(prefix_uri) == bool(manifest_uri):
        return _error("exactly one of s3_prefix | manifest_s3uri required",
                      "INVALID_PARAMETER")

    # The stack's IAM only grants source-image access on the proxy bucket —
    # reject foreign buckets HERE with a usable message instead of letting
    # every line fail downstream with head_error:403 (or the listing crash
    # with a raw AccessDenied).
    for uri, param in ((prefix_uri, "s3_prefix"),
                       (manifest_uri, "manifest_s3uri")):
        if uri and not uri.startswith(f"s3://{PROXY_BUCKET}/"):
            return _error(
                f"{param} must be under s3://{PROXY_BUCKET}/ — the stack "
                f"only has read access to its proxy bucket; copy the images "
                f"there first", "FOREIGN_BUCKET")

    # Clamp like recognize_faces does (1-50): the semaphore max_holders
    # ratchet is up-only and PERMANENT — an unvalidated large value here
    # would permanently raise the shared Rekognition concurrency cap.
    raw_max = event.get("max_concurrent_executions")
    if raw_max is not None:
        try:
            raw_max = int(raw_max)
        except (TypeError, ValueError):
            return _error("max_concurrent_executions must be an integer",
                          "INVALID_PARAMETER")
        if not 1 <= raw_max <= 50:
            return _error("max_concurrent_executions must be in [1, 50]",
                          "INVALID_PARAMETER")
    max_concurrent = raw_max if raw_max is not None else 5

    job_uuid = str(_uuid_mod.uuid4())
    out_prefix = (f"micro-workflow/{TARGET_NAME}/_index_jobs/{job_uuid}")

    if prefix_uri:
        manifest_uri, count, err = _manifest_from_prefix(
            prefix_uri, out_prefix)
        if err:
            return err
        logger.info("generated manifest for %s: %d entries",
                    prefix_uri, count)

    sfn_input = {
        "uuid": job_uuid,
        "action": "batch_index",
        "input": {
            "collection_id": collection_id,
            "manifest_s3uri": manifest_uri,
            "max_concurrent_executions": max_concurrent,
        },
        "output": {"s3uri": f"s3://{PROXY_BUCKET}/{out_prefix}"},
        "data": {},
    }
    return _start_index_execution(job_uuid, sfn_input, {
        "collection_id": collection_id,
        "manifest_s3uri": manifest_uri,
    })


def _manifest_from_prefix(prefix_uri: str,
                          out_prefix: str) -> tuple[str, int, dict | None]:
    rest = prefix_uri[5:]
    bucket, _, key_prefix = rest.partition("/")

    keys: list[str] = []
    kwargs: dict = {"Bucket": bucket, "Prefix": key_prefix}
    try:
        while True:
            resp = _s3.list_objects_v2(**kwargs)
            keys.extend(o["Key"] for o in resp.get("Contents") or []
                        if o["Key"].lower().endswith(IMAGE_EXTENSIONS))
            if not resp.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
    except ClientError as exc:
        return "", 0, _error(
            f"cannot list {prefix_uri}: "
            f"{exc.response['Error']['Code']}", "LIST_FAILED")
    if not keys:
        return "", 0, _error(
            f"no image files under {prefix_uri} (extensions: "
            f"{', '.join(IMAGE_EXTENSIONS)})", "EMPTY_PREFIX")

    lines = []
    for key in sorted(keys):
        base = os.path.splitext(os.path.basename(key))[0]
        name = " ".join(base.split())
        if name:
            lines.append(json.dumps(
                {"image_uri": f"s3://{bucket}/{key}",
                 "celebrity_name": name}, ensure_ascii=False))
    manifest_key = f"{out_prefix}/manifest.jsonl"
    _s3.put_object(Bucket=PROXY_BUCKET, Key=manifest_key,
                   Body="\n".join(lines).encode("utf-8"),
                   ContentType="application/jsonl")
    return f"s3://{PROXY_BUCKET}/{manifest_key}", len(lines), None


def batch_delete_faces(event: dict) -> dict:
    collection_id = ((event.get("collection_id") or "").strip()
                     or DEFAULT_COLLECTION_ID)
    face_ids = event.get("face_ids") or []
    celeb = (event.get("celebrity_name") or "").strip()
    if bool(face_ids) == bool(celeb):
        return _error("exactly one of face_ids | celebrity_name required",
                      "INVALID_PARAMETER")

    job_uuid = str(_uuid_mod.uuid4())
    sfn_input = {
        "uuid": job_uuid,
        "action": "batch_delete",
        "input": {
            "collection_id": collection_id,
            **({"face_ids": face_ids} if face_ids else
               {"celebrity_name": celeb}),
        },
        "output": {"s3uri": f"s3://{PROXY_BUCKET}/micro-workflow/"
                            f"{TARGET_NAME}/_index_jobs/{job_uuid}"},
        "data": {},
    }
    return _start_index_execution(job_uuid, sfn_input,
                                  {"collection_id": collection_id})


def delete_collection(event: dict) -> dict:
    collection_id = (event.get("collection_id") or "").strip()
    if not collection_id:
        return _error("collection_id is required", "MISSING_PARAMETER")
    force, bool_err = _parse_bool(event.get("force"))
    if bool_err:
        return _error(f"force: {bool_err}", "INVALID_PARAMETER")
    if collection_id == DEFAULT_COLLECTION_ID and not force:
        return _error(
            f"{collection_id!r} is the stack default collection — pass "
            f"force=true to delete it. In-flight recognize_faces runs using "
            f"it will have undefined naming results.", "DEFAULT_COLLECTION")

    job_uuid = str(_uuid_mod.uuid4())
    sfn_input = {
        "uuid": job_uuid,
        "action": "delete_collection",
        "input": {"collection_id": collection_id},
        "output": {"s3uri": f"s3://{PROXY_BUCKET}/micro-workflow/"
                            f"{TARGET_NAME}/_index_jobs/{job_uuid}"},
        "data": {},
    }
    return _start_index_execution(job_uuid, sfn_input,
                                  {"collection_id": collection_id})


def _start_index_execution(job_uuid: str, sfn_input: dict,
                           echo: dict) -> dict:
    if not INDEX_STATE_MACHINE_ARN:
        return _error("service misconfigured: INDEX_STATE_MACHINE_ARN unset",
                      "CONFIG_ERROR")
    resp = _sfn.start_execution(
        stateMachineArn=INDEX_STATE_MACHINE_ARN,
        name=job_uuid,
        input=json.dumps(sfn_input),
    )
    # A handle, matching every other submit in this repo: an index job's results
    # come from get_job_status, which routes on the state machine name and parses
    # the index output shape.  No absorbed branch here — these executions are
    # named with a random uuid4, not a work-hash, so a name collision is
    # impossible and every call starts a new run.
    return {"job_id": resp["executionArn"], "status": "RUNNING",
            "current_state": _INDEX_FIRST_STATE,
            "workflow_id": job_uuid, "action": sfn_input["action"],
            "output_s3_prefix": sfn_input["output"]["s3uri"], **echo}
