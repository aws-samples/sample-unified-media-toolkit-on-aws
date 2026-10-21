# lib/shot_pipeline.py
"""
Shot segmentation and representative embedding computation.

Design principles
─────────────────
Shot boundaries are detected purely from cosine distances between
consecutive DINOv2 embeddings. No event classification is used.
The DINOv2 handler ensures frame-accurate zoom-in at all visual
transitions (via cosdist_threshold and black frame zoom), so the
frame index already contains exactly the frames needed for accurate
boundary placement.

Boundary detection algorithm
─────────────────────────────
1. Compute cosine distances for all N-1 consecutive frame pairs.

2. Evaluate only gap=1 pairs (zoomed-in regions).
   Coarse-stride pairs (gap = coarse_stride, e.g. 6 at 25fps/4fps)
   are not evaluated — the DINOv2 handler confirmed their cosine
   distance was ≤ cosdist_threshold and they belong to the same shot.

3. Find contiguous runs of gap=1 pairs where distance > threshold.

4. For each run, first scan interior frames for the sync-pop signature
   (see Sync-pop detection below). If found at interior frame b:

     a. Process left sub-run [run_start, b-2] with peak absorption
        (if non-empty, i.e. b >= run_start + 2).
     b. Place two boundaries at b and b+1, isolating the sync-pop
        as a single-frame shot.
     c. Continue outer scan from b+1 (right sub-run handled naturally
        by subsequent iterations of the outer loop).

   If no sync-pop: apply standard peak absorption to the full run.
   Find peak pair k, absorb peak frame into most similar side:

     dist_to_left ≤ dist_to_right  → frame[k] → Shot A, boundary at k+1
     dist_to_right < dist_to_left  → frame[k] → Shot B, boundary at k

   Each run with no sync-pop produces exactly one boundary.

5. Gap guarantee: every detected boundary satisfies
       shot[N+1].start_frame − shot[N].end_frame = 1
   — for gap-1 boundaries because that is where they sit, and for
   small-gap boundaries (9.8 fix 1: a super-threshold spike across a
   PTS-mapping hole/duplicate or VFR gap of <= SMALL_GAP_BOUNDARY_MAX)
   because the closing shot's end_frame is frame_index[boundary_row]-1,
   absorbing the unsampled hole frames into the LEFT shot. end_time still
   describes the last SAMPLED frame; timecodes are derived at merge time
   from frame numbers (2026-08-25), so end_timecode describes the nominal
   end_frame — consistent with frame_range.

Sync-pop detection
──────────────────
Sync-pop frames are used in broadcast production to verify audio/video
synchronisation. They are always exactly one frame and appear between
two highly similar content frames. They may be black, white, grey, any
solid/monochrome colour, or contain minimal graphics (e.g. the digit
"2", sometimes called "2-pop").

Two-stage detection:

  Stage 1 — in _find_shot_boundaries:
    For each run interior frame b, compute cosine distance between
    frame[b-1] and frame[b+1]. If < SYNC_POP_THRESHOLD (0.10), the
    two adjacent frames are highly similar → isolate frame b as a
    single-frame shot by placing boundaries at b and b+1.

  Stage 2 — in _merge_sync_pop_shots (called from build_shots):
    After segmentation and per-shot embedding computation, scan for
    shots with frame_count == 1. For each, if both neighbours exist
    and cosine distance between their centroids < SYNC_POP_THRESHOLD,
    merge the three shots into one. The merged shot records the
    anomaly frame's number, time, and timecode in the sync_pop field
    when the frame is monochrome-solid (luma entropy <
    SYNC_POP_ENTROPY_MAX), or in the flash_frame field otherwise
    (lightning, exposure flicker, glitch — healed identically but not
    a broadcast sync-pop). Centroid is recomputed from the combined
    content frames only (prev + next, excluding the anomaly frame).

Shot classification
───────────────────
  "content"  — one or more non-black frames, visually varying
  "black"    — all frames have frame_type == "black" (sentinel embeddings)
  "static"   — all adjacent frame embeddings are visually identical
               (SMPTE bars, slates, freeze-frames, static title cards)

Shot dict schema
────────────────
{
    "shot_id":                   0,
    "shot_type":                 "content",
    "start_frame":               0,
    "end_frame":                 299,
    "start_time":                0.0,
    "end_time":                  11.96,
    "start_timecode":            "09:59:43:14",
    "end_timecode":              "10:00:00:05",
    "frame_count":               50,
    "sync_pop":                  null,   # monochrome-solid healed anomaly, or absent
    "flash_frame":               null,   # high-entropy healed anomaly, or absent
    "centroid_embedding":        [...],
    "representative_embeddings": [...]
}
"""
from __future__ import annotations

import logging
import numpy as np
from lib.input_schema import (
    DEFAULT_SHOT_BOUNDARY_THRESHOLD,
    SMALL_GAP_BOUNDARY_MAX,
    SYNC_POP_THRESHOLD,
    SYNC_POP_ENTROPY_MAX,
    FREEZE_SIMILARITY_THRESHOLD,
    OUTLIER_DISTANCE_THRESHOLD,
)

log = logging.getLogger(__name__)


# =============================================================================
# Cosine distances
# =============================================================================

def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    normed = embeddings / norms
    dots = (normed[:-1] * normed[1:]).sum(axis=1)
    return (1.0 - dots).astype(np.float32)


# =============================================================================
# Shot boundary detection
# =============================================================================

def _peak_absorb(
    run_start:  int,
    run_end:    int,
    frame_index: list[int],
    distances:  np.ndarray,
    boundaries: list[tuple[int, float]],
    N:          int,
) -> None:
    if run_end <= run_start:
        return

    run_dists = distances[run_start:run_end]
    peak_offset = int(np.argmax(run_dists))
    k = run_start + peak_offset
    peak_dist = float(distances[k])

    if k == 0:
        boundary_row = k + 1
    else:
        dist_to_left = float(distances[k - 1])
        dist_to_right = float(distances[k])
        boundary_row = (k + 1) if dist_to_left <= dist_to_right else k

    if 0 < boundary_row < N:
        boundaries.append((boundary_row, peak_dist))


def _find_shot_boundaries(
    frame_index: list[int],
    embeddings:  np.ndarray,
    threshold:   float,
) -> list[tuple[int, float]]:
    N = len(frame_index)
    if N < 2:
        return []

    distances:  np.ndarray = _cosine_distances(embeddings)
    boundaries: list[tuple[int, float]] = []

    i = 0
    while i < len(distances):
        gap = frame_index[i + 1] - frame_index[i]
        dist = float(distances[i])

        if gap != 1:
            # 9.8 fix 1: a super-threshold spike over a SMALL gap is a cut
            # with +-(gap-1)-frame localisation error, not noise — discarding
            # it is how Meridian fused three shots and face-recognition
            # renamed a person for 207 s. gap 0 is a PTS-mapping DUPLICATE
            # pair (two different images sharing a frame number: d>threshold
            # proves they differ); gaps 2-3 are mapping holes or VFR. The
            # boundary goes at the right-hand sample; deliberate sampling
            # gaps (coarse stride >= 6) stay excluded, so detection
            # sensitivity on well-formed input is unchanged.
            if dist > threshold and gap <= SMALL_GAP_BOUNDARY_MAX:
                boundaries.append((i + 1, dist))
                log.info(
                    "Small-gap boundary (9.8 fix 1): frames %d->%d gap=%d "
                    "dist=%.4f — accepted at right-hand sample, +-%d frame(s)",
                    frame_index[i], frame_index[i + 1], gap, dist,
                    max(gap - 1, 0))
            i += 1
            continue
        if dist <= threshold:
            i += 1
            continue

        j = i
        while j < len(distances):
            if (frame_index[j + 1] - frame_index[j] == 1
                    and float(distances[j]) > threshold):
                j += 1
            else:
                break

        sync_pop_b: int | None = None
        for b in range(i + 1, j):
            n1 = embeddings[b - 1].astype(np.float32)
            n2 = embeddings[b + 1].astype(np.float32)
            n1 = n1 / max(float(np.linalg.norm(n1)), 1e-8)
            n2 = n2 / max(float(np.linalg.norm(n2)), 1e-8)
            neighbour_dist = float(1.0 - n1 @ n2)
            if neighbour_dist < SYNC_POP_THRESHOLD:
                sync_pop_b = b
                log.debug(
                    "Sync-pop candidate: row %d (frame %d)  "
                    "neighbour_dist=%.4f",
                    b, frame_index[b], neighbour_dist,
                )
                break

        if sync_pop_b is not None:
            b = sync_pop_b
            if b >= i + 2:
                _peak_absorb(i, b - 1, frame_index, distances, boundaries, N)
            boundaries.append((b,     float(distances[b - 1])))
            boundaries.append((b + 1, float(distances[b])))
            i = b + 1
        else:
            _peak_absorb(i, j, frame_index, distances, boundaries, N)
            i = j

    return boundaries


# =============================================================================
# Shot segmentation
# =============================================================================

def _segment_shots(
    frame_index: list[int],
    frame_type:  list[str],
    source_time: list[float],
    embeddings:  np.ndarray,
    threshold:   float,
) -> list[dict]:
    if not frame_index:
        return []

    boundaries = _find_shot_boundaries(frame_index, embeddings, threshold)

    boundary_set: set[int] = set()
    for row, _ in boundaries:
        boundary_set.add(row)

    shots:         list[dict] = []
    shot_start_row = 0

    for row in range(len(frame_index)):
        if row > 0 and row in boundary_set:
            rows = list(range(shot_start_row, row))
            first, last = rows[0], rows[-1]
            shot_type = (
                "black"
                if all(frame_type[r] == "black" for r in rows)
                else "content"
            )
            shots.append({
                "start_frame":    frame_index[first],
                # The shot ends one source frame BEFORE the next shot starts,
                # not at its own last SAMPLE. For a gap-1 boundary the two are
                # the same number (frame_index[row]-1 == frame_index[last]);
                # for a small-gap boundary (9.8 fix 1: a cut detected across a
                # mapping hole or duplicate) the unsampled hole frames are
                # absorbed into the LEFT shot, matching the right-hand
                # boundary placement and its documented +-(gap-1) frame
                # uncertainty. Without this, the hole belongs to NO shot and
                # the merge state's continuity validator (correctly) fails
                # the asset — the first live film_b run with fix 1
                # active failed exactly here, on its two rescued gap-2 cuts.
                "end_frame":      frame_index[row] - 1,
                "start_time":     source_time[first] if source_time else 0.0,
                "end_time":       source_time[last] if source_time else 0.0,
                "shot_type":      shot_type,
                "frame_rows":     rows,
            })
            shot_start_row = row

    rows = list(range(shot_start_row, len(frame_index)))
    if rows:
        first, last = rows[0], rows[-1]
        shot_type = (
            "black"
            if all(frame_type[r] == "black" for r in rows)
            else "content"
        )
        shots.append({
            "start_frame":    frame_index[first],
            "end_frame":      frame_index[last],
            "start_time":     source_time[first] if source_time else 0.0,
            "end_time":       source_time[last] if source_time else 0.0,
            "shot_type":      shot_type,
            "frame_rows":     rows,
        })

    for shot_id, shot in enumerate(shots):
        shot["shot_id"] = shot_id

    black_count = sum(1 for s in shots if s["shot_type"] == "black")
    log.info(
        "Shot segmentation: %d frames → %d shots  "
        "(%d boundaries  %d black shots)",
        len(frame_index), len(shots), len(boundary_set), black_count,
    )
    return shots


# =============================================================================
# Representative embedding computation
# =============================================================================

def _is_freeze_sequence(embeddings: np.ndarray) -> bool:
    if len(embeddings) < 2:
        return True
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-8)
    normed = embeddings / norms
    sims = (normed[:-1] * normed[1:]).sum(axis=1)
    return bool(np.all(sims >= FREEZE_SIMILARITY_THRESHOLD))


def _compute_shot_embedding(
    shot:       dict,
    embeddings: np.ndarray,
) -> dict:
    rows = shot.get("frame_rows", [])
    is_black_shot = shot.get("shot_type", "content") == "black"

    if not rows:
        zero = np.zeros(embeddings.shape[1], dtype=np.float32)
        shot["centroid_embedding"] = zero.tolist()
        shot["representative_embeddings"] = [
            {"type": "centroid", "embedding": zero.tolist()}
        ]
        shot["frame_count"] = 0
        return shot

    X = embeddings[rows].astype(np.float32)
    M = len(X)

    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
    normed = X / norms

    raw_centroid = normed.mean(axis=0)
    c_norm = np.linalg.norm(raw_centroid)
    centroid = (raw_centroid / c_norm) if c_norm > 1e-8 else normed[0].copy()

    representatives = [{"type": "centroid", "embedding": centroid.tolist()}]

    if is_black_shot:
        shot["centroid_embedding"] = centroid.tolist()
        shot["representative_embeddings"] = representatives
        shot["frame_count"] = M
        return shot

    is_freeze = _is_freeze_sequence(normed)
    if is_freeze:
        shot["shot_type"] = "static"

    if not is_freeze and M >= 2:
        boundary_start = normed[0]
        boundary_end = normed[-1]

        if float(1.0 - boundary_start @ centroid) > OUTLIER_DISTANCE_THRESHOLD:
            representatives.append({
                "type":      "boundary_start",
                "embedding": boundary_start.tolist(),
            })
        if float(1.0 - boundary_end @ centroid) > OUTLIER_DISTANCE_THRESHOLD:
            representatives.append({
                "type":      "boundary_end",
                "embedding": boundary_end.tolist(),
            })

        if M >= 4:
            cos_dists = 1.0 - (normed @ centroid)
            interior_dists = cos_dists[1:M - 1]
            if len(interior_dists) > 0:
                outlier_idx = int(np.argmax(interior_dists))
                outlier_dist = float(interior_dists[outlier_idx])
                if outlier_dist > OUTLIER_DISTANCE_THRESHOLD:
                    representatives.append({
                        "type":      "outlier",
                        "embedding": normed[1 + outlier_idx].tolist(),
                    })

    shot["centroid_embedding"] = centroid.tolist()
    shot["representative_embeddings"] = representatives
    shot["frame_count"] = M
    return shot


# =============================================================================
# Sync-pop merge  (Stage 2)
# =============================================================================

def _merge_sync_pop_shots(
    shots:      list[dict],
    embeddings: np.ndarray,
    frame_entropy: list[float] | None = None,
) -> list[dict]:
    if len(shots) < 3:
        return shots

    result:   list[dict] = []
    n_merged: int = 0
    i = 0

    while i < len(shots):
        shot = shots[i]

        is_candidate = (
            shot.get("frame_count", 0) == 1
            and shot.get("shot_type") != "black"
            and i > 0
            and i < len(shots) - 1
        )

        if is_candidate:
            prev_shot = shots[i - 1]
            next_shot = shots[i + 1]

            prev_c = np.array(
                prev_shot["centroid_embedding"], dtype=np.float32)
            next_c = np.array(
                next_shot["centroid_embedding"], dtype=np.float32)
            prev_norm = prev_c / max(float(np.linalg.norm(prev_c)), 1e-8)
            next_norm = next_c / max(float(np.linalg.norm(next_c)), 1e-8)
            dist = float(1.0 - prev_norm @ next_norm)

            if dist < SYNC_POP_THRESHOLD:
                if (result
                        and result[-1].get("shot_id") == prev_shot.get("shot_id")):
                    result.pop()

                    # A true sync-pop is a monochrome-solid frame (black,
                    # white, grey, or minimal graphic like a "2") — near-
                    # zero luma entropy.  A high-entropy single-frame
                    # anomaly (lightning flash, exposure flicker, glitch)
                    # is healed identically but labeled flash_frame so
                    # downstream consumers don't mistake it for broadcast
                    # leader structure.
                    pop_rows = shot.get("frame_rows", [])
                    pop_entropy = None
                    if frame_entropy is not None and pop_rows:
                        pop_entropy = float(frame_entropy[pop_rows[0]])
                    is_mono = (
                        pop_entropy is None
                        or pop_entropy < SYNC_POP_ENTROPY_MAX
                    )
                    anomaly_key = "sync_pop" if is_mono else "flash_frame"

                    log.info(
                        "%s: shot %d  frame=%d  entropy=%s  "
                        "neighbour_dist=%.4f  merging shots [%d+%d+%d]",
                        "Sync-pop" if is_mono else "Flash-frame",
                        shot["shot_id"],
                        shot["start_frame"],
                        f"{pop_entropy:.3f}" if pop_entropy is not None else "n/a",
                        dist,
                        prev_shot["shot_id"],
                        shot["shot_id"],
                        next_shot["shot_id"],
                    )

                    content_rows = (
                        prev_shot.get("frame_rows", [])
                        + next_shot.get("frame_rows", [])
                    )

                    merged_type = (
                        "content"
                        if (prev_shot.get("shot_type") == "content"
                            or next_shot.get("shot_type") == "content")
                        else "black"
                    )

                    merged = {
                        "start_frame":    prev_shot["start_frame"],
                        "end_frame":      next_shot["end_frame"],
                        "start_time":     prev_shot["start_time"],
                        "end_time":       next_shot["end_time"],
                        "shot_type":      merged_type,
                        "frame_rows":     content_rows,
                        anomaly_key: {
                            # timecode is derived at merge time from `frame`,
                            # like every shot timecode — see the merge state's
                            # _derive_timecodes
                            "frame":    shot["start_frame"],
                            "time":     shot.get("start_time",     0.0),
                        },
                    }

                    _compute_shot_embedding(merged, embeddings)

                    merged["frame_count"] = (
                        prev_shot.get("frame_count", 0)
                        + 1
                        + next_shot.get("frame_count", 0)
                    )

                    result.append(merged)
                    n_merged += 1
                    i += 2
                    continue

        result.append(shot)
        i += 1

    if n_merged > 0:
        log.info(
            "Sync-pop merge: %d sync-pop(s) resolved  "
            "%d shots → %d shots",
            n_merged, len(shots), len(result),
        )
        for shot_id, shot in enumerate(result):
            shot["shot_id"] = shot_id

    return result


# =============================================================================
# Public entry point
# =============================================================================

def build_shots(
    embeddings:              np.ndarray,
    frame_index:             list[int],
    frame_type:              list[str],
    source_time:             list[float],
    shot_boundary_threshold: float = DEFAULT_SHOT_BOUNDARY_THRESHOLD,
    frame_entropy:           list[float] | None = None,
) -> list[dict]:
    shots = _segment_shots(
        frame_index=frame_index,
        frame_type=frame_type,
        source_time=source_time,
        embeddings=embeddings,
        threshold=shot_boundary_threshold,
    )

    if not shots:
        log.warning("No shots produced from segmentation")
        return []

    for shot in shots:
        _compute_shot_embedding(shot, embeddings)

    shots = _merge_sync_pop_shots(shots, embeddings, frame_entropy)

    for shot in shots:
        shot.pop("frame_rows", None)

    black_count  = sum(1 for s in shots if s.get("shot_type") == "black")
    static_count = sum(1 for s in shots if s.get("shot_type") == "static")
    syncpop_count = sum(1 for s in shots if s.get("sync_pop"))
    flash_count = sum(1 for s in shots if s.get("flash_frame"))
    empty_count = sum(1 for s in shots if s.get("frame_count", 0) == 0)

    log.info(
        "Shot pipeline complete: %d shots  "
        "%d black  %d static  %d sync-pop  %d flash-frame  %d empty",
        len(shots), black_count, static_count, syncpop_count,
        flash_count, empty_count,
    )
    return shots
