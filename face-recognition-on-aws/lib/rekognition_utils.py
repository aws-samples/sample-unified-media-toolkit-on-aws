"""
Rekognition API wrappers — adaptive retry, explicit throttle retry, and
per-invocation call counters (README.md §7 three-layer throttle defense:
semaphore admission → static per-API pool bounds → adaptive retry).

All image APIs take S3Object references (grids are uploaded first) — the
15 MB / 10000×10000 px bounds apply, not the 5 MB inline-bytes cap.

Call counters accumulate per CallCounter instance; states persist them into
iterator JSONs / usage payloads for usage_tracking.
"""

from __future__ import annotations

import logging
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from lib.input_schema import API_COUNTER_KEYS

logger = logging.getLogger(__name__)

# adaptive: client-side token bucket that reacts to throttles account-wide.
_rek = boto3.client("rekognition", config=Config(
    retries={"mode": "adaptive", "max_attempts": 5},
    max_pool_connections=30,
))

_THROTTLE_CODES = ("ProvisionedThroughputExceededException", "ThrottlingException")
# GetCelebrityInfo fan-out width: the API is 50 TPS and uncharged;
# 5 concurrent per run stays far under it even with several runs in
# flight (the semaphore bounds those), and _run absorbs bursts.
CELEBRITY_INFO_THREADS = 5
_EXPLICIT_RETRIES = 10
_RETRY_PAUSE_SEC = 1.0


class CallCounter:
    """Mutable per-state API call tally. Fields derive from the ONE key
    list in input_schema so a new API cannot be counted here yet
    silently dropped by usage_tracking's sum."""

    def __init__(self) -> None:
        for key in API_COUNTER_KEYS:
            setattr(self, key, 0)

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def add(self, other: dict) -> None:
        for k, v in (other or {}).items():
            if hasattr(self, k):
                setattr(self, k, getattr(self, k) + int(v))


def add_usage(total: dict, usage: dict) -> None:
    """
    Accumulate a usage dict into a running total, in place. The single
    canonical form of the sum-iterator-usage loop — one shipped bug
    (recognize_celebrities_api=0 in the cost report) came from one of
    several hand-rolled copies silently dropping a key.
    """
    for k, v in (usage or {}).items():
        total[k] = total.get(k, 0) + int(v)


def _run(fn, **kwargs):
    """Explicit throttle retry on top of the SDK's adaptive layer."""
    for attempt in range(_EXPLICIT_RETRIES):
        try:
            return fn(**kwargs)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in _THROTTLE_CODES and attempt < _EXPLICIT_RETRIES - 1:
                time.sleep(_RETRY_PAUSE_SEC)
                continue
            raise
    raise RuntimeError("unreachable")


def _s3_object(bucket: str, key: str) -> dict:
    return {"S3Object": {"Bucket": bucket, "Name": key}}


# ── Detection ──────────────────────────────────────────────────────────────────

def detect_faces(bucket: str, key: str, counter: CallCounter) -> list[dict]:
    resp = _run(_rek.detect_faces,
                Image=_s3_object(bucket, key), Attributes=["ALL"])
    counter.detect_faces_api += 1
    return resp.get("FaceDetails") or []


def recognize_celebrities(bucket: str, key: str, counter: CallCounter) -> dict:
    resp = _run(_rek.recognize_celebrities, Image=_s3_object(bucket, key))
    counter.recognize_celebrities_api += 1
    return {
        "CelebrityFaces": resp.get("CelebrityFaces") or [],
        "UnrecognizedFaces": resp.get("UnrecognizedFaces") or [],
    }


def get_celebrity_info(celebrity_id: str, counter: CallCounter) -> dict:
    """Canonical info for one celebrity id -> {"urls": [...],
    "known_gender": str|None}. No per-call charge; the constraint is
    the API's TPS (50/s) — callers dedup per DISTINCT id and fan out at
    most CELEBRITY_INFO_THREADS wide (user guidance 2026-09-07), and
    _run's throttle retry absorbs bursts from concurrent runs."""
    resp = _run(_rek.get_celebrity_info, Id=celebrity_id)
    counter.get_celebrity_info_api += 1
    return {
        "urls": resp.get("Urls") or [],
        "known_gender": (resp.get("KnownGender") or {}).get("Type"),
    }


# ── Collections ────────────────────────────────────────────────────────────────

def create_collection_if_absent(collection_id: str) -> None:
    try:
        _rek.create_collection(CollectionId=collection_id)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise


def delete_collection_ignore_missing(collection_id: str) -> None:
    """Fire delete; swallow ResourceNotFound (user decision — no pre-check)."""
    try:
        _rek.delete_collection(CollectionId=collection_id)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise


def index_faces(collection_id: str, bucket: str, key: str,
                external_image_id: str, max_faces: int,
                quality_filter: str, counter: CallCounter) -> list[dict]:
    resp = _run(_rek.index_faces,
                CollectionId=collection_id,
                Image=_s3_object(bucket, key),
                ExternalImageId=external_image_id,
                MaxFaces=max_faces,
                QualityFilter=quality_filter,
                DetectionAttributes=["DEFAULT"])
    counter.index_faces_api += 1
    return resp.get("FaceRecords") or []


def search_faces(collection_id: str, face_id: str, threshold: float,
                 max_faces: int, counter: CallCounter) -> list[dict]:
    resp = _run(_rek.search_faces,
                CollectionId=collection_id,
                FaceId=face_id,
                FaceMatchThreshold=threshold,
                MaxFaces=max_faces)
    counter.search_faces_api += 1
    return resp.get("FaceMatches") or []


def search_faces_by_image(collection_id: str, bucket: str, key: str,
                          threshold: float, counter: CallCounter) -> list[dict]:
    try:
        resp = _run(_rek.search_faces_by_image,
                    CollectionId=collection_id,
                    Image=_s3_object(bucket, key),
                    FaceMatchThreshold=threshold,
                    MaxFaces=1,
                    QualityFilter="LOW")
    except ClientError as exc:
        # No face found in the query image is a soft miss, not an error.
        if exc.response["Error"]["Code"] == "InvalidParameterException":
            counter.search_faces_api += 1
            return []
        raise
    counter.search_faces_api += 1
    return resp.get("FaceMatches") or []


def delete_faces(collection_id: str, face_ids: list[str]) -> None:
    if face_ids:
        _run(_rek.delete_faces, CollectionId=collection_id, FaceIds=face_ids)
