"""
StatePlanDenseDetection — pure compute: classify inter-probe gaps as
skippable or dense, then pack dense grids (README.md §3).

Patch noise classification first (ticker discount), then per-gap:
  skippable iff (1) both endpoint probes found NO faces, AND
                (2-4) frame_hash.gap_is_skippable (stable-patch adjacency,
                endpoint drift, global sanity), AND
                (5) shots mode only: no shot boundary strictly inside the
                gap (README.md §21 — an AND: more detection, never less;
                the hash tests stay because they catch within-shot motion
                and entrances that shot boundaries cannot).

Dense regions = face-bearing gaps dilated ±probe_stride frames + hash-break
frames, minus near-duplicate dedup (stable-patch distance <
dedup_hamming_threshold to the last kept frame, skip-run capped at
probe_stride). Skipped-dedup frames are recorded for box interpolation.

probe_stride == 0 → every kept frame was already probed; nothing to plan.
"""

from __future__ import annotations

import logging
import math

from lib import frame_hash
from lib.input_schema import FRAMES_PER_GRID, get_input_config
from lib.s3_utils import get_json, parse_s3uri, put_json

logger = logging.getLogger(__name__)

_GRIDS_PER_ITERATOR = 40
_MAX_ITERATORS = 10


def lambda_handler(event: dict, context) -> dict:
    cfg = get_input_config(event)
    meta = event["data"]["frame_meta"]
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")

    frame_index = get_json(out_bucket, meta["frame_index_key"])
    frames = sorted((f for f in frame_index["frames"] if not f.get("dropped")),
                    key=lambda f: f["frame_num"])
    by_num = {f["frame_num"]: f for f in frames}

    # Collect probe results: frame_num → face count
    probe_faces: dict[int, int] = {}
    probed_nums: set[int] = set()
    for it in event["data"]["iterators"] or []:
        iterator = get_json(out_bucket, it["key"])
        for grid, faces in iterator.get("results", {}).items():
            for f in faces:
                probe_faces[f["frame_num"]] = probe_faces.get(
                    f["frame_num"], 0) + 1
        for grid in iterator["grids"]:
            probed_nums.update(fr["frame_num"] for fr in grid["frames"])

    stride = cfg["probe_stride"]
    threshold = max(cfg["dedup_hamming_threshold"], 4)
    plan = {"noisy_patches": [], "skipped_gaps": [], "dense_frames": [],
            "dedup_skipped": []}

    shot_starts = None
    if meta.get("shot_boundaries_key"):
        from lib import shots as shots_lib
        shot_starts = get_json(
            out_bucket, meta["shot_boundaries_key"])["starts"]

    if stride == 0 or len(frames) == 0:
        dense: list[dict] = []          # everything already probed
    else:
        hashes = [f["hashes"] for f in frames]
        noisy = frame_hash.classify_noisy_patches(hashes, threshold)
        plan["noisy_patches"] = noisy

        probes = frames[::stride]
        dense = []
        for a, b in zip(probes, probes[1:]):
            ia = frames.index(a)  # contiguous slices — index is fine at 1 fps scale
            ib = frames.index(b)
            gap = frames[ia + 1:ib]
            a_empty = probe_faces.get(a["frame_num"], 0) == 0
            b_empty = probe_faces.get(b["frame_num"], 0) == 0
            no_cut_inside = (shot_starts is None
                             or not shots_lib.boundary_inside(
                                 shot_starts, a["frame_num"], b["frame_num"]))
            if (a_empty and b_empty and no_cut_inside
                    and frame_hash.gap_is_skippable(
                        [g["hashes"] for g in gap], a["hashes"], b["hashes"],
                        noisy, threshold)):
                plan["skipped_gaps"].append(
                    [a["frame_num"], b["frame_num"]])
                continue
            dense.extend(gap)
        # Tail beyond the last probe is always dense (no closing endpoint).
        if probes:
            last = frames.index(probes[-1])
            dense.extend(frames[last + 1:])

        # Near-duplicate dedup inside dense regions
        dedup_t = cfg["dedup_hamming_threshold"]
        if dedup_t > 0:
            kept_dense = []
            last_kept = None
            run = 0
            for f in dense:
                if (last_kept is not None and run < stride
                        and frame_hash.stable_distance(
                            last_kept["hashes"], f["hashes"], noisy) < dedup_t):
                    plan["dedup_skipped"].append(
                        {"frame_num": f["frame_num"],
                         "timestamp": f["timestamp"]})
                    run += 1
                    continue
                kept_dense.append(f)
                last_kept = f
                run = 0
            dense = kept_dense

    dense = [f for f in dense if f["frame_num"] not in probed_nums]
    plan["dense_frames"] = [f["frame_num"] for f in dense]
    put_json(out_bucket, f"{out_prefix}/ephemeral/detection_plan.json", plan)
    logger.info("plan: %d dense frames, %d gaps skipped, %d dedup-skipped, "
                "noisy patches %s", len(dense), len(plan["skipped_gaps"]),
                len(plan["dedup_skipped"]), plan["noisy_patches"])

    # ── Pack dense grids (same iterator shape as the probe pass) ─────────────
    cell_w, cell_h = meta["wxh"]
    layout = meta["grid_layout"]
    per_grid = min(layout["cols"] * layout["rows"], FRAMES_PER_GRID)
    grids = [dense[i:i + per_grid] for i in range(0, len(dense), per_grid)]

    worker_ctx = {
        "out_bucket": out_bucket,
        "output_s3uri": event["output"]["s3uri"],
        "frames_bucket": event["data"]["media_processing_output"]["frames_bucket"],
        "frames_prefix": meta["frames_prefix"],
    }
    per_iter = max(_GRIDS_PER_ITERATOR,
                   math.ceil(len(grids) / _MAX_ITERATORS)) if grids else 1
    iterators = []
    for item_id, gi in enumerate(range(0, len(grids), per_iter)):
        iter_key = f"{out_prefix}/ephemeral/detect_faces_{item_id}.json"
        put_json(out_bucket, iter_key, {
            "grids": [
                {"frames": [{"key": f["key"], "frame_num": f["frame_num"],
                             "timestamp": f["timestamp"]}
                            for f in grid]}
                for grid in grids[gi:gi + per_iter]
            ],
            "layout": {"cols": layout["cols"], "rows": layout["rows"],
                       "cell_w": cell_w, "cell_h": cell_h},
            "results": {},
            "usage": {},
        })
        iterators.append({"item_id": item_id, "key": iter_key,
                          "context": worker_ctx})

    # Preserve probe iterator pointers for the track builder.
    event["data"]["probe_iterators"] = [
        {"item_id": it["item_id"], "key": it["key"]}
        for it in (event["data"]["iterators"] or [])]
    event["data"]["iterators"] = iterators
    event["data"]["detection_plan_key"] = (
        f"{out_prefix}/ephemeral/detection_plan.json")
    return event
