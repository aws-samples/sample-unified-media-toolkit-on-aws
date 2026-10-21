"""
Voice registry — DynamoDB records + S3 conditioning/preview/source storage.

Modeled on face-recognition-on-aws/lib/face_store.py (the house pattern
for "persistent named entity with derivative artifacts"). Schema per
README.md §5:

DDB item: pk="voice#{voice_id}", sk="record"
    voice_id, name, gender, language, origin, catalog, comment,
    s3_conditioning_key, s3_preview_key, s3_source_key?, duration_sec,
    created_at, catalog_created
GSI:
    catalog-index (catalog, catalog_created) — list_voices pagination;
        catalog_created = "{catalog}#{created_at_ms:015d}"

origin: dataset name for baked-in voices ("FLEURS", "LibriSpeech", ...)
or the literal "user-upload" for cloned voices — the single field that
distinguishes the two registration paths.

Conditioning blob (~1 MB npz: speech_tokens + spk_embedding + prompt_mel)
lives in S3 at s3_conditioning_key — DynamoDB's 400 KB item limit rules
out inline storage.

Deletes are HARD (DDB + S3 derivatives) with a tombstone item
pk="deleted#{voice_id}" (own TTL) as audit trail — same rationale as
face_store: no soft-delete flags on live records.
"""

from __future__ import annotations

import os
import time

import boto3

TABLE_NAME = os.environ.get("VOICE_TABLE_NAME", "")
TOMBSTONE_TTL_SEC = 90 * 24 * 3600

USER_UPLOAD_ORIGIN = "user-upload"

# Voice license terms — RECORDED at registration, never derived at read
# time (the origin→license mapping is a fact about the source dataset at
# the moment of registration, not a stable rule).
#   IC          — Individual Contributor: the uploader attests they hold
#                 the rights to the voice (default for user uploads)
#   CC-BY-4.0   — LibriSpeech / FLEURS baked-in voices
LICENSE_IC = "IC"
_KNOWN_ORIGIN_LICENSES = {
    "librispeech": "CC-BY-4.0",
    "fleurs": "CC-BY-4.0",
}


def default_license(origin: str) -> str | None:
    """License to record when the caller supplies none. User uploads
    default to IC; known datasets to their published license; unknown
    origins get NO default — an explicit license must be supplied, so a
    new dataset can't silently inherit a wrong term."""
    if origin == USER_UPLOAD_ORIGIN:
        return LICENSE_IC
    return _KNOWN_ORIGIN_LICENSES.get(origin.lower())


_ddb = boto3.client("dynamodb")


def catalog_created_sk(catalog: str, created_at_ms: int) -> str:
    """catalog-index sort key — zero-padded epoch keeps within-catalog
    ordering stable (same shape as face_store's celeb_created)."""
    return f"{catalog}#{created_at_ms:015d}"


def register_voice(voice_id: str, fields: dict) -> None:
    """Write one voice record. Derives catalog_created from
    catalog + created_at. `fields` must include catalog; None values
    are skipped."""
    created_at = int(time.time() * 1000)
    catalog = fields.get("catalog") or "default"
    item = {
        "pk": {"S": f"voice#{voice_id}"},
        "sk": {"S": "record"},
        "voice_id": {"S": voice_id},
        "created_at": {"N": str(created_at)},
        "catalog_created": {"S": catalog_created_sk(catalog, created_at)},
    }
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            item[k] = {"BOOL": v}
        elif isinstance(v, (int, float)):
            item[k] = {"N": str(v)}
        else:
            item[k] = {"S": str(v)}
    _ddb.put_item(TableName=TABLE_NAME, Item=item)


def unmarshal(item: dict) -> dict:
    return {k: (v.get("S") if "S" in v else
                v.get("N") if "N" in v else v.get("BOOL"))
            for k, v in item.items()}


def get_voice(voice_id: str) -> dict | None:
    resp = _ddb.get_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": f"voice#{voice_id}"}, "sk": {"S": "record"}})
    item = resp.get("Item")
    return unmarshal(item) if item else None


def list_voices(catalog: str = "default", gender: str | None = None,
                language: str | None = None, limit: int = 50,
                next_token: str | None = None) -> dict:
    """List voices in a catalog, newest last (catalog_created ordering).
    gender/language are post-query filters (small result sets — a voice
    catalog is hundreds of entries, not millions; no dedicated GSI
    justified yet)."""
    kwargs = {
        "TableName": TABLE_NAME,
        "IndexName": "catalog-index",
        # "catalog" is a DynamoDB reserved keyword — an unaliased reference
        # is a ValidationException at query time, not a no-op. Same alias as
        # gateway_tool/lambda_function.py's _list_voice_collection.
        "KeyConditionExpression": "#c = :c",
        "ExpressionAttributeNames": {"#c": "catalog"},
        "ExpressionAttributeValues": {":c": {"S": catalog}},
        "Limit": limit,
    }
    if next_token:
        import base64
        import json
        kwargs["ExclusiveStartKey"] = json.loads(
            base64.b64decode(next_token).decode())
    resp = _ddb.query(**kwargs)

    voices = [unmarshal(i) for i in resp.get("Items", [])]
    if gender:
        voices = [v for v in voices if v.get("gender") == gender]
    if language:
        voices = [v for v in voices if v.get("language") == language]

    out = {"voices": voices}
    if "LastEvaluatedKey" in resp:
        import base64
        import json
        out["next_token"] = base64.b64encode(
            json.dumps(resp["LastEvaluatedKey"]).encode()).decode()
    return out


def delete_voice(voice_id: str, reason: str) -> bool:
    """Hard-delete the record + write a tombstone. S3 derivative cleanup
    is the caller's job (needs the record's keys — fetch before calling).
    Returns False if the record didn't exist."""
    record = get_voice(voice_id)
    if record is None:
        return False
    tomb = {
        "pk": {"S": f"deleted#{voice_id}"},
        "sk": {"S": "record"},
        "voice_id": {"S": voice_id},
        "deleted_at": {"N": str(int(time.time() * 1000))},
        "reason": {"S": reason},
        "ttl": {"N": str(int(time.time()) + TOMBSTONE_TTL_SEC)},
    }
    for k in ("catalog", "name", "origin"):
        if record.get(k):
            tomb[k] = {"S": str(record[k])}
    _ddb.put_item(TableName=TABLE_NAME, Item=tomb)
    _ddb.delete_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": f"voice#{voice_id}"}, "sk": {"S": "record"}})
    return True
