"""
StateBuildFaceTracks — merge probe + dense detections into temporal tracks
(README.md §4): gap-dilated IoU + gender + cut-break veto, box interpolation
for dedup-skipped frames, representative-centroid selection, noise
filtering, and recognize-iterator packing.

Cut evidence (README.md §21 + §23): the 3×3 patch dHash break test is the
cut veto in EVERY mode. When a validated shot-segmentation supplement is
present (frame_meta.shot_boundaries_key), its DINOv2 shot boundaries JOIN
the dHash test as a UNION — a break from either vetoes continuation. The
boundaries add what dHash cannot see (slow dissolves) but never replace it:
a boundary list is an upstream product that can be incomplete, and a missed
hard cut with no local fallback welds two people into one track (the
Meridian mislabel, README.md §23). The union re-admits dHash's false breaks
on flashes/flicker — accepted, because a false break only splits a track
and sandbox clustering re-merges the pieces, while a false continuation
contaminates an identity irreversibly.
"""

from __future__ import annotations

import bisect
import logging
import math

from lib import frame_hash, shots as shots_lib, tracks as tracks_lib
from lib.face_quality import choose_representative
from lib.input_schema import (
    HEADSHOTS_PER_GRID,
    MAX_TRACKS_PER_ITERATOR,
    NOISE_LOW_BRIGHTNESS,
    NOISE_LOW_SHARPNESS,
    NOISE_MIN_FACE_PX_AT_REF,
    get_input_config,
)
from lib.rekognition_utils import add_usage
from lib.s3_utils import get_json, parse_s3uri, put_json

logger = logging.getLogger(__name__)

_MAX_ITERATORS = 10


def _dhash_cut_test(frame_index: dict, noisy: list, threshold: int):
    """The local cut detector: True iff any adjacent kept-frame pair between
    frame_a and frame_b breaks on stable-patch dHash distance. Runs in every
    mode — the hashes are computed for probe-then-fill regardless, so this
    costs one closure, not new work."""
    kept = sorted((f for f in frame_index["frames"]
                   if not f.get("dropped")),
                  key=lambda f: f["frame_num"])
    nums = [f["frame_num"] for f in kept]
    hashes = [f["hashes"] for f in kept]

    def dhash_break(frame_a: int, frame_b: int) -> bool:
        ia = bisect.bisect_left(nums, frame_a)
        ib = bisect.bisect_right(nums, frame_b) - 1
        for i in range(max(ia, 0), min(ib, len(hashes) - 1)):
            if frame_hash.stable_distance(
                    hashes[i], hashes[i + 1], noisy) >= threshold:
                return True
        return False

    return dhash_break


def _cut_veto(dhash_break, starts=None):
    """Compose the tracking cut veto (README.md §23): dHash always; a shot-
    boundary list is UNIONED in when supplied, never substituted — an
    upstream boundary list can be incomplete, and a missed hard cut with no
    local fallback welds two people into one track."""
    if starts is None:
        return dhash_break

    def cut_break_between(frame_a: int, frame_b: int) -> bool:
        return (shots_lib.boundary_between(starts, frame_a, frame_b)
                or dhash_break(frame_a, frame_b))

    return cut_break_between


def lambda_handler(event: dict, context) -> dict:
    cfg = get_input_config(event)
    meta = event["data"]["frame_meta"]
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")

    # ── Gather all detections (probe + dense iterators) ──────────────────────
    faces: list[dict] = []
    counter_totals: dict = {}
    iter_refs = list(event["data"].get("probe_iterators") or [])
    iter_refs += [{"key": it["key"]} for it in (event["data"]["iterators"] or [])]
    for ref in iter_refs:
        iterator = get_json(out_bucket, ref["key"])
        for grid_faces in (iterator.get("results") or {}).values():
            faces.extend(grid_faces)
        add_usage(counter_totals, iterator.get("usage") or {})

    min_conf = cfg["min_confidence"]
    faces = [f for f in faces if f["confidence"] >= min_conf]
    logger.info("%d detections above confidence %.0f", len(faces), min_conf)

    # ── Cut-break veto: dHash always; shot boundaries UNION in (README.md §23) ──
    plan = get_json(out_bucket, event["data"]["detection_plan_key"])
    frame_index = get_json(out_bucket, meta["frame_index_key"])
    dhash_break = _dhash_cut_test(
        frame_index, plan.get("noisy_patches") or [],
        max(cfg["dedup_hamming_threshold"], 4))

    starts = None
    if meta.get("shot_boundaries_key"):
        starts = get_json(out_bucket, meta["shot_boundaries_key"])["starts"]
    cut_break_between = _cut_veto(dhash_break, starts)

    # ── Tracking ──────────────────────────────────────────────────────────────
    all_tracks = tracks_lib.build_tracks(
        faces, cfg["max_track_gap_sec"], cut_break_between)

    # Box interpolation across dedup-skipped frames
    skipped = plan.get("dedup_skipped") or []
    for tr in all_tracks:
        tracks_lib.interpolate_skipped(tr, skipped)

    # ── Noise filter + centroid selection ────────────────────────────────────
    w, h = meta["wxh"]
    scale = math.sqrt((w * h) / (960 * 540)) if w and h else 1.0
    min_px = NOISE_MIN_FACE_PX_AT_REF * scale
    qualified = []
    for tr in all_tracks:
        detected = [f for f in tr["faces"] if f.get("source") != "interpolated"]
        if len(detected) == 1:
            f = detected[0]
            face_px_w = f["box"]["w"] * w
            face_px_h = f["box"]["h"] * h
            if (face_px_w < min_px or face_px_h < min_px
                    or (f.get("brightness", 99) < NOISE_LOW_BRIGHTNESS
                        and f.get("sharpness", 99) < NOISE_LOW_SHARPNESS)):
                continue  # single-appearance noise
        centroid = choose_representative(detected)
        if centroid is None:
            continue
        tr["centroid_idx"] = tr["faces"].index(detected[centroid])
        qualified.append(tr)

    for i, tr in enumerate(qualified):
        tr["track_id"] = i
    tracks_key = f"{out_prefix}/ephemeral/face_tracks.json"
    put_json(out_bucket, tracks_key, {"tracks": qualified,
                                      "usage": counter_totals})
    logger.info("%d tracks (%d dropped as noise/unqualified)",
                len(qualified), len(all_tracks) - len(qualified))

    # ── Recognize-celebrities iterator packing ────────────────────────────────
    event["data"]["tracks_key"] = tracks_key
    event["data"]["num_tracks"] = len(qualified)
    event["data"]["probe_iterators"] = None

    if not cfg["celebrity_recognition"] or not qualified:
        event["data"]["iterators"] = []
        return event

    worker_ctx = {
        "out_bucket": out_bucket,
        "output_s3uri": event["output"]["s3uri"],
        "frames_bucket": event["data"]["media_processing_output"]["frames_bucket"],
        "frames_prefix": meta["frames_prefix"],
        "tracks_key": tracks_key,
    }
    track_ids = [tr["track_id"] for tr in qualified]
    batches = [track_ids[i:i + HEADSHOTS_PER_GRID]
               for i in range(0, len(track_ids), HEADSHOTS_PER_GRID)]
    # MAX_TRACKS_PER_ITERATOR bounds work per Map item
    per_iter = max(1, math.ceil(
        len(batches) / max(1, min(_MAX_ITERATORS,
                                  math.ceil(len(track_ids)
                                            / MAX_TRACKS_PER_ITERATOR)))))
    iterators = []
    for item_id, bi in enumerate(range(0, len(batches), per_iter)):
        iter_key = f"{out_prefix}/ephemeral/recognize_{item_id}.json"
        put_json(out_bucket, iter_key, {
            "batches": batches[bi:bi + per_iter],
            "results": {},
            "usage": {},
        })
        iterators.append({"item_id": item_id, "key": iter_key,
                          "context": worker_ctx})
    event["data"]["iterators"] = iterators
    return event
