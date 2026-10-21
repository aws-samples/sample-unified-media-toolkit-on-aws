"""
Persistent face records — DynamoDB items + S3 thumbnail/full-image storage.

DDB item (README.md §6, §19): pk="face#{face_id}", sk="record"
    face_id, collection_id, external_image_id, celeb?, gender, age_range,
    confidence, coord, thumbnail_key, full_image_key, workflow_id, created_at,
    origin ("user"|"auto"), source_key?, source_etag?, celeb_created
GSIs:
    celeb-index       (celeb, created_at)      — lookups by name
    collection-index  (collection_id, celeb_created) — list_faces pagination;
        celeb_created = "{celeb}#{created_at}" with unnamed faces sorting
        last via UNNAMED_CELEB_SENTINEL ("~unnamed" — 0x7E > letters/digits).

Deletes are HARD (Rekognition + DDB + S3 derivatives) with a tombstone item
pk="deleted#{face_id}" (own TTL) as the audit trail — never a soft-delete
flag on live records (README.md §19: right-to-erasure doesn't tolerate lag,
and a marker breaks the FaceCount == record-count reconciliation invariant).

ExternalImageId format: "{version}:{uuid-hex32}:{epoch-ms}" — the same SHAPE
as the old pipeline's createExternalImageId but NOT compatible with it (the
old pipeline pins Version 'v4' and regex-validates '^v4:'; ours is '1').
This store is standalone — nothing here feeds core-lib's resolveExternalImageId.
The hex32 segment embeds the workflow uuid, which search_index_faces relies
on to identify (and repair) its own orphaned faces after a crash between
IndexFaces and register_face.

full_image derivative semantics (README.md §19 — CHANGED from the initial
port): fit-inside downscale of the WHOLE source image (video frame or user
photo), NOT a face-region crop. `coord` (normalized [x, y, w, h], 0-1,
resolution-agnostic) locates the face within it and within the original.
One downscaled frame can therefore serve several faces from the same frame.
"""

from __future__ import annotations

import os
import time

import boto3

from lib.input_schema import (
    FACE_THUMBNAIL_DIM,
    FULLIMAGE_MAX_EDGE_PX,
    HEADSHOT_CROP_MARGIN,
    TOMBSTONE_PK_PREFIX,
    TOMBSTONE_TTL_SEC,
    UNNAMED_CELEB_SENTINEL,
)
from lib.s3_utils import put_bytes

TABLE_NAME = os.environ.get("FACE_TABLE_NAME", "")
_EXTERNAL_IMAGE_ID_VERSION = "1"

_ddb = boto3.client("dynamodb")


def create_external_image_id(workflow_id: str) -> str:
    hex32 = workflow_id.replace("-", "")[:32].ljust(32, "0")
    return f"{_EXTERNAL_IMAGE_ID_VERSION}:{hex32}:{int(time.time() * 1000)}"


def celeb_created_sk(celeb: str | None, created_at_ms: int) -> str:
    """collection-index sort key: named faces sort alphabetically, unnamed
    last (sentinel). Zero-padded epoch keeps within-name ordering stable."""
    name = celeb if celeb else UNNAMED_CELEB_SENTINEL
    return f"{name}#{created_at_ms:015d}"


def register_face(face_id: str, fields: dict) -> None:
    """Write one face record. Derives the collection-index sort key from
    celeb + created_at so every record is listable by collection."""
    created_at = int(time.time() * 1000)
    item = {
        "pk": {"S": f"face#{face_id}"},
        "sk": {"S": "record"},
        "face_id": {"S": face_id},
        "created_at": {"N": str(created_at)},
        "celeb_created": {"S": celeb_created_sk(fields.get("celeb"),
                                                created_at)},
    }
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            item[k] = {"BOOL": v}
        elif isinstance(v, (int, float)):
            item[k] = {"N": str(v)}
        elif isinstance(v, (list, tuple)):
            item[k] = {"S": ",".join(str(x) for x in v)}
        else:
            item[k] = {"S": str(v)}
    _ddb.put_item(TableName=TABLE_NAME, Item=item)


def unmarshal(item: dict) -> dict:
    return {k: (v.get("S") if "S" in v else
                v.get("N") if "N" in v else v.get("BOOL"))
            for k, v in item.items()}


def get_face(face_id: str) -> dict | None:
    resp = _ddb.get_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": f"face#{face_id}"}, "sk": {"S": "record"}})
    item = resp.get("Item")
    return unmarshal(item) if item else None


def batch_get_faces(face_ids: list[str]) -> dict[str, dict]:
    """face_id → record (unmarshalled, string values). Missing ids omitted.
    Deduplicates input (BatchGetItem rejects duplicate keys) and retries
    UnprocessedKeys — throttled keys silently treated as "missing" would
    fail delete resolution with a bogus "face not found"."""
    import time as _time

    unique = list(dict.fromkeys(face_ids))
    out: dict[str, dict] = {}
    for i in range(0, len(unique), 100):
        keys = [{"pk": {"S": f"face#{fid}"}, "sk": {"S": "record"}}
                for fid in unique[i:i + 100]]
        for attempt in range(8):
            resp = _ddb.batch_get_item(
                RequestItems={TABLE_NAME: {"Keys": keys}})
            for item in resp.get("Responses", {}).get(TABLE_NAME, []):
                rec = unmarshal(item)
                out[rec["face_id"]] = rec
            keys = (resp.get("UnprocessedKeys") or {}).get(
                TABLE_NAME, {}).get("Keys") or []
            if not keys:
                break
            _time.sleep(0.2 * (attempt + 1))
        if keys:
            raise RuntimeError(
                f"batch_get_faces: {len(keys)} keys still unprocessed after "
                f"retries (DDB throttling) — failing rather than reporting "
                f"live faces as missing")
    return out


def update_face_name(face_id: str, celebrity_name: str) -> bool:
    """Rename/tag one face (late binding — historical outputs resolve the
    live name via get_face). Re-keys BOTH name GSIs (celeb, celeb_created).
    Returns False when the record does not exist; never creates one."""
    rec = get_face(face_id)
    if rec is None:
        return False
    created_at = int(rec.get("created_at") or 0)
    _ddb.update_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": f"face#{face_id}"}, "sk": {"S": "record"}},
        UpdateExpression="SET celeb = :c, celeb_created = :cc",
        ConditionExpression="attribute_exists(pk)",
        ExpressionAttributeValues={
            ":c": {"S": celebrity_name},
            ":cc": {"S": celeb_created_sk(celebrity_name, created_at)},
        })
    return True


def delete_face_record(record: dict, reason: str) -> None:
    """Hard-delete one record + write its tombstone (audit, own TTL)."""
    face_id = record["face_id"]
    tomb = {
        "pk": {"S": f"{TOMBSTONE_PK_PREFIX}{face_id}"},
        "sk": {"S": "record"},
        "face_id": {"S": face_id},
        "deleted_at": {"N": str(int(time.time() * 1000))},
        "reason": {"S": reason},
        "ttl": {"N": str(int(time.time()) + TOMBSTONE_TTL_SEC)},
    }
    for k in ("collection_id", "celeb", "origin", "workflow_id"):
        if record.get(k):
            tomb[k] = {"S": str(record[k])}
    _ddb.put_item(TableName=TABLE_NAME, Item=tomb)
    _ddb.delete_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": f"face#{face_id}"}, "sk": {"S": "record"}})


def store_face_images(bucket: str, prefix: str, face_id: str,
                      source_image: bytes, box: dict) -> tuple[str, str]:
    """
    Store the two derivatives for an indexed face (README.md §19 semantics):
      thumbnail  — 3:4 face crop (96x128)
      full_image — fit-inside downscale of the WHOLE source image
    Returns (thumbnail_key, full_image_key). `box` (normalized) locates the
    face in both the original and the downscaled full image.
    """
    from lib.grid_image import crop_headshot, downscale_fit

    hex_id = face_id.replace("-", "")
    thumb_key = f"{prefix}/{hex_id}.jpg"
    full_key = f"{prefix}/fullimage/{hex_id}.jpg"

    thumb = crop_headshot(source_image, box, margin=HEADSHOT_CROP_MARGIN,
                          out_w=FACE_THUMBNAIL_DIM[0],
                          out_h=FACE_THUMBNAIL_DIM[1])
    put_bytes(bucket, thumb_key, thumb, "image/jpeg")

    full = downscale_fit(source_image, FULLIMAGE_MAX_EDGE_PX)
    put_bytes(bucket, full_key, full, "image/jpeg")
    return thumb_key, full_key
