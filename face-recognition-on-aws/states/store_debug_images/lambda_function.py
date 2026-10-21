"""
StateStoreDebugImages — per-person montages + indexed-faces grid (README.md §4).
Post-release; no Rekognition calls. The per-person montage doubles as the
thumbnail_key target referenced from facemetadata.json.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.grid_image import compose_tile_grid, crop_headshot, crop_headshot_native
from lib.input_schema import (
    HEADSHOT_CROP_MARGIN,
    REMAINING_TIME_BUFFER_MS,
    REP_FACE_MAX_EDGE_PX,
    REP_FACE_RATIO,
)
from lib.s3_utils import get_bytes, get_json, list_keys, parse_s3uri, put_bytes

logger = logging.getLogger(__name__)

_CELL = 112
_COLS = 8
_MAX_FACES_PER_MONTAGE = 16
_PERSON_WORKERS = 8


def lambda_handler(event: dict, context) -> dict:
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")
    meta = event["data"]["frame_meta"]
    frames_bucket = event["data"]["media_processing_output"]["frames_bucket"]
    frames_prefix = meta["frames_prefix"]

    tracks = {tr["track_id"]: tr
              for tr in get_json(out_bucket,
                                 event["data"]["tracks_key"])["tracks"]}
    clusters = get_json(out_bucket, event["data"]["clusters_key"])["clusters"]

    # One prefix listing replaces N head_object probes — on a clean first run
    # every probe was a guaranteed miss paid once per person (hundreds of
    # clusters on real broadcasts).
    people_prefix = f"{out_prefix}/debug/people/"
    existing = list_keys(out_bucket, people_prefix)

    def _montage(cluster) -> None:
        gid = cluster["face_group_id"]
        key = f"{people_prefix}person_{gid:04d}.jpg"
        face_key = f"{people_prefix}person_{gid:04d}_face.jpg"
        if key in existing and face_key in existing:
            return  # re-enter: already written
        # Sort by time first: cluster["track_ids"] comes from a Python
        # set (identity_clusters.json), so its order is incidental
        # numeric track_id order — i.e. roughly the person's FIRST
        # appearances only. Sampling that directly showed only the
        # earliest few seconds of screen time in the montage, hiding
        # later instances entirely. Spread the sample evenly across the
        # person's full timespan instead.
        person_tracks = sorted(
            (tracks[tid] for tid in cluster["track_ids"] if tid in tracks),
            key=lambda tr: tr["first_ts"])
        if not person_tracks:
            return
        if key not in existing:
            faces = [tr["faces"][tr["centroid_idx"]] for tr in
                     _evenly_sample(person_tracks, _MAX_FACES_PER_MONTAGE)]
            tiles = []
            for face in faces:
                jpeg = get_bytes(frames_bucket,
                                 f"{frames_prefix}/{face['frame_key']}")
                tiles.append(crop_headshot(jpeg, face["box"],
                                           HEADSHOT_CROP_MARGIN, _CELL, _CELL))
            montage, _ = compose_tile_grid(tiles, _CELL, _CELL, _COLS)
            put_bytes(out_bucket, key, montage, "image/jpeg")
        if face_key not in existing:
            # The representative face as a SINGLE consumer-facing crop —
            # the montage above is a debug contact sheet, not a reference
            # face. Same rep selection as finalize_outputs' gender/age
            # source (max appearance_count → that track's centroid face),
            # so facemetadata's rep_face_key points at the face its own
            # demographics were read from. 3:4 at native resolution
            # (never the 96×128 thumbnail scale — measured too small to
            # re-identify against; shot-metadata-enrichment PLAN §5).
            rep_track = max(person_tracks,
                            key=lambda tr: (tr["appearance_count"],
                                            -tr["first_ts"]))
            rep = rep_track["faces"][rep_track["centroid_idx"]]
            jpeg = get_bytes(frames_bucket,
                             f"{frames_prefix}/{rep['frame_key']}")
            put_bytes(out_bucket, face_key,
                      crop_headshot_native(jpeg, rep["box"],
                                           HEADSHOT_CROP_MARGIN,
                                           *REP_FACE_RATIO,
                                           REP_FACE_MAX_EDGE_PX),
                      "image/jpeg")

    # Parallelize at the PERSON level — each montage's work (16 GETs, crops,
    # compose, PUT) is independent; the old inner-tile-only pool serialized
    # hundreds of persons behind each other. Deadline check between waves.
    with ThreadPoolExecutor(max_workers=_PERSON_WORKERS) as pool:
        for wi in range(0, len(clusters), _PERSON_WORKERS):
            if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
                raise LambdaResourceTimeoutException(
                    f"debug images checkpointed at person wave {wi}")
            wave = clusters[wi:wi + _PERSON_WORKERS]
            # list() propagates the first worker exception (fail hard).
            list(pool.map(_montage, wave))

    event["data"]["debug_people_prefix"] = people_prefix
    return event


def _evenly_sample(items: list, n: int) -> list:
    """n items spread evenly across the sequence (first and last included)."""
    if len(items) <= n:
        return items
    if n <= 1:
        return items[:1]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]
