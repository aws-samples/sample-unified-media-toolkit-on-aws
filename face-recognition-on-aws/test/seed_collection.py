#!/usr/bin/env python3
"""
Seed a face collection from a directory of celebrity photos where each
filename is the person's name (e.g. "Jane Doe.jpeg").

For each photo: IndexFaces (MaxFaces=1 — the photos are single-subject
headshots) into the collection with a compact ExternalImageId, then register
face_id → name in the stack's DynamoDB table (the pipeline resolves matched
face_ids to names through DDB, mirroring what phase-2's index_face will do).

Usage:
    python3 seed_collection.py --dir ~/Downloads/NABFaceCollection \
        --collection face-recognition-on-aws \
        --table unified-media-toolkit-face-recognition \
        [--region eu-west-1] [--dry-run]

Idempotent: skips photos whose name already has a DDB record.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

VALID_EXT = (".jpg", ".jpeg", ".png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-1"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rek = boto3.client("rekognition", region_name=args.region)
    ddb = boto3.client("dynamodb", region_name=args.region)

    photos = sorted(
        f for f in os.listdir(os.path.expanduser(args.dir))
        if f.lower().endswith(VALID_EXT))
    print(f"{len(photos)} photos in {args.dir}")
    if args.dry_run:
        for f in photos[:10]:
            print("  ", os.path.splitext(f)[0])
        return 0

    try:
        rek.create_collection(CollectionId=args.collection)
        print(f"created collection {args.collection}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
        print(f"collection {args.collection} exists")

    # Existing names (idempotency)
    existing: set[str] = set()
    scan_kwargs = {
        "TableName": args.table,
        "FilterExpression": "begins_with(pk, :p)",
        "ExpressionAttributeValues": {":p": {"S": "face#"}},
        "ProjectionExpression": "celeb",
    }
    while True:
        resp = ddb.scan(**scan_kwargs)
        existing.update(i["celeb"]["S"] for i in resp.get("Items", [])
                        if "celeb" in i)
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    if existing:
        print(f"{len(existing)} names already registered — skipping those")

    ok = failed = skipped = 0
    for fname in photos:
        name = os.path.splitext(fname)[0].strip()
        if name in existing:
            skipped += 1
            continue
        path = os.path.join(os.path.expanduser(args.dir), fname)
        with open(path, "rb") as fh:
            image_bytes = fh.read()
        if len(image_bytes) > 5 * 1024 * 1024:
            print(f"  SKIP >5MB inline limit: {fname}")
            failed += 1
            continue
        try:
            resp = rek.index_faces(
                CollectionId=args.collection,
                Image={"Bytes": image_bytes},
                ExternalImageId=f"seed:{int(time.time() * 1000)}",
                MaxFaces=1,
                QualityFilter="AUTO",
                DetectionAttributes=["DEFAULT"],
            )
        except ClientError as exc:
            print(f"  FAIL {fname}: {exc.response['Error']['Code']}")
            failed += 1
            continue
        records = resp.get("FaceRecords") or []
        if not records:
            print(f"  NO FACE {fname}")
            failed += 1
            continue
        face = records[0]["Face"]
        detail = records[0].get("FaceDetail") or {}
        gender = ((detail.get("Gender") or {}).get("Value"))
        age = detail.get("AgeRange") or {}
        item = {
            "pk": {"S": f"face#{face['FaceId']}"},
            "sk": {"S": "record"},
            "face_id": {"S": face["FaceId"]},
            "collection_id": {"S": args.collection},
            "external_image_id": {"S": face.get("ExternalImageId", "")},
            "celeb": {"S": name},
            "created_at": {"N": str(int(time.time() * 1000))},
        }
        if gender:
            item["gender"] = {"S": gender}
        if age:
            item["age_range"] = {"S": f"{age.get('Low', 0)},{age.get('High', 0)}"}
        ddb.put_item(TableName=args.table, Item=item)
        ok += 1
        if ok % 25 == 0:
            print(f"  ... {ok} indexed")

    print(f"done: {ok} indexed, {skipped} skipped (existing), {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
