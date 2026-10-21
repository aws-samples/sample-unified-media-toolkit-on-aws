# merge_reconcile_shots_handler.py
"""
Step Function State: Merge and Reconcile Shots
------------------------------------------------
Single Lambda that reads per-window .json + .npy produced by the
Extract Frame Features Map state, merges shots across window
boundaries, detects and resolves dissolve transitions and noise
shots, runs recurrence detection, validates invariants, and writes
final outputs.

No scene clustering — shots are the terminal structural unit.
Scene clustering is deferred to a downstream pipeline with
VLM/SLM classification.

Shot merge algorithm
────────────────────
1. Frame ownership deduplication (window A owns frames < window B's
   first frame).

2. Collect all local_shots from all windows.

3. Sort by (start_frame, -frame_count).

4. Coverage-tracking deduplication:
     - end_frame <= coverage_end → fully covered duplicate, skip
     - start_frame <= coverage_end and end_frame > coverage_end
       → cross-segment boundary stitch (extend last shot)
     - start_frame > coverage_end → new shot, append

5. Sequential reindex of shot_ids.

Post-processing
───────────────
6. Noise merge — find very short shots between similar neighbours,
   merge all three into one.

7. Per-shot entropy statistics from frame-level entropy.

8. Recurrence detection (lib/recurrence.py).

9. Validation (6 invariants — no scene invariants).

Outputs (S3)
────────────
  embeddings.npz   — float32 [N, 384] + int64 frame_number
                     + float64 source_time + float32 entropy
  shots.json       — global shot list; each shot carries
                     frames_key → per-shot _frames.tar.gz S3 key
  recurrence.json  — recurring shot groups

Step Functions output keys (data.*)
────────────────────────────────────
  embeddings_key, shots_key, recurrence_key — S3 keys for the
  three output files above.
  n_shots, n_recurrence_groups, n_recurring_shots, total_frames,
  embed_dim, validation_ok, n_shot_frames_ok, n_shot_frames_errors.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import tarfile as _tarfile
import threading
import time
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import boto3
from PIL import Image

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
_h = logging.StreamHandler()
_h.setLevel(logging.DEBUG)
_h.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
if not log.handlers:
    log.addHandler(_h)

from lib.input_schema import (
    SYNC_POP_THRESHOLD,
    DEFAULT_FRAME_SELECT_MIN_DIST,
    DEFAULT_NOISE_DISTANCE_THRESHOLD,
    DEFAULT_MAX_NOISE_DURATION,
    DEFAULT_ADJACENT_CENTROID_THRESHOLD,
    DEFAULT_MAX_ADJACENT_MERGE_DURATION,
    DEFAULT_NEAR_BLACK_ENTROPY_MAX,
    DEFAULT_FADE_SPLIT_THRESHOLD,
    DEFAULT_FADE_SPLIT_MIN_DURATION_SEC,
    DEFAULT_FADE_SPLIT_MIN_SIDE_FRAMES,
    DEFAULT_DRIFT_SPLIT_THRESHOLD,
    DEFAULT_DRIFT_SPLIT_MIN_DURATION_SEC,
    DEFAULT_DRIFT_SPLIT_MIN_SIDE_SEC,
    DEFAULT_MIN_SHOT_DURATION,
    DEFAULT_MAX_RECURRENCE_THRESHOLD,
    DEFAULT_MIN_RECURRENCE_COUNT,
    DEFAULT_MIN_OCCURRENCE_GAP_SEC,
    DEFAULT_MAX_RECURRENCE_COUNT,
)
from lib.s3_utils import (
    get_json,
    get_binary,
    put_json,
    put_binary,
)
from lib.burst_evidence import detect_burst_runs
from lib.misc_utils import frame_to_smpte
from lib.recurrence import confirm_groups_by_hash, detect_recurrence
from lib.tail_evidence import analyse_programme_tail
_SENTINEL_DIM = 384
_SENTINEL_VALUE = 1.0 / math.sqrt(_SENTINEL_DIM)
_SENTINEL = np.full(_SENTINEL_DIM, _SENTINEL_VALUE, dtype=np.float32)
_SENTINEL_DIST_MIN = 0.01


# =============================================================================
# S3 helpers
# =============================================================================

def _read_npy(bucket, key):
    return np.load(io.BytesIO(get_binary(bucket, key)))


def _write_json(bucket, key, data):
    put_json(bucket, key, data)
    log.info("Written: s3://%s/%s", bucket, key)


def _write_npz(bucket, key, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    buf.seek(0)
    put_binary(bucket, key, buf.getvalue(), "application/octet-stream")
    mb = buf.getbuffer().nbytes / (1024 ** 2)
    log.info("Written: s3://%s/%s  (%.2f MB)", bucket, key, mb)


# =============================================================================
# Frame ownership
# =============================================================================

def _owned_row_range(frame_index, next_first_frame):
    n = len(frame_index)
    if n == 0:
        return 0, -1
    if next_first_frame is None:
        return 0, n - 1
    end = -1
    for i, f in enumerate(frame_index):
        if f < next_first_frame:
            end = i
        else:
            break
    return 0, end


# =============================================================================
# Cosine distance helper
# =============================================================================

def _cosine_dist(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - (a / na) @ (b / nb))


# =============================================================================
# Shot merge (coverage tracking)
# =============================================================================

def _merge_shots(window_metas):
    """
    Collect all local_shots from all windows, dedup via coverage tracking.
    Returns deduplicated shot list with sequential shot_ids.
    """
    all_shots: list[dict] = []
    for meta in window_metas:
        wid = meta["window_id"]
        for s in meta.get("local_shots", []):
            all_shots.append({**s, "_window_id": wid})

    if not all_shots:
        return []

    all_shots.sort(
        key=lambda s: (s["start_frame"], -(s.get("frame_count", 0))))

    deduped:      list[dict] = []
    coverage_end: int = -1

    for shot in all_shots:
        sf = shot["start_frame"]
        ef = shot["end_frame"]

        if ef <= coverage_end:
            continue

        if sf <= coverage_end and deduped:
            last = deduped[-1]
            if ef > last["end_frame"]:
                last["end_frame"] = ef
                last["end_time"] = shot.get(
                    "end_time", last.get("end_time", 0.0))
                if (shot.get("frame_count", 0)
                        > last.get("frame_count", 0)):
                    last["centroid_embedding"] = shot["centroid_embedding"]
                    last["frame_count"] = shot["frame_count"]
                    last["shot_type"] = shot.get(
                        "shot_type", last.get("shot_type"))
            coverage_end = max(coverage_end, ef)
            continue

        deduped.append(shot)
        coverage_end = ef

    for i, shot in enumerate(deduped):
        shot["shot_id"] = i

    log.info(
        "Shot merge: %d collected → %d deduplicated",
        len(all_shots), len(deduped),
    )
    return deduped


# =============================================================================
# Centroid recomputation
# =============================================================================

def _shot_frame_slice(shot, frame_numbers):
    """Half-open row range [lo, hi) of the per-frame arrays belonging to a shot.

    One source of truth: the centroid recompute, the entropy pass and the
    dispersion pass all address the same rows, and three copies of a
    searchsorted pair is three chances to disagree on `side=`.
    """
    return (int(np.searchsorted(frame_numbers, shot["start_frame"],
                                side="left")),
            int(np.searchsorted(frame_numbers, shot["end_frame"],
                                side="right")))


def _unit_rows(X):
    """Rows of X L2-normalised to unit length, float32."""
    return X.astype(np.float32) / np.linalg.norm(
        X, axis=1, keepdims=True).clip(1e-8).astype(np.float32)


def _recompute_shot_centroid(shot, embeddings, frame_numbers):
    """Recompute centroid from all embeddings in shot's frame range."""
    lo, hi = _shot_frame_slice(shot, frame_numbers)
    X = embeddings[lo:hi].astype(np.float32)

    if len(X) == 0:
        shot["centroid_embedding"] = [0.0] * int(embeddings.shape[1])
        shot["frame_count"] = 0
        return

    normed = _unit_rows(X)
    raw = normed.mean(axis=0)
    c_norm = float(np.linalg.norm(raw))
    centroid = (raw / c_norm) if c_norm > 1e-8 else normed[0].copy()
    shot["centroid_embedding"] = centroid.tolist()
    shot["frame_count"] = int(hi - lo)


# =============================================================================
# Fade split (runs before adjacent drift merge)
# =============================================================================

def _split_halves(shot: dict, right_start: int) -> tuple[dict, dict]:
    """
    Copy a shot into left/right halves for a split at right_start.

    Frame-located fields (sync_pop, flash_frame, noise_frames) are kept only in the
    half that actually contains their frames — duplicating them would
    misreport their location in shots.json.  transition_to_next
    describes the boundary AFTER the shot, so only the right half
    (which still ends at that boundary) keeps it; a stale copy on the
    left half would suppress legitimate merges at the new interior
    boundary.
    """
    left = dict(shot)
    right = dict(shot)

    for key in ("sync_pop", "flash_frame"):
        marker = shot.get(key)
        if marker is not None:
            in_right = marker.get("frame", -1) >= right_start
            (left if in_right else right).pop(key, None)

    nf = shot.get("noise_frames")
    if nf is not None:
        nf_start = (nf.get("frames") or [right_start])[0]
        in_right = nf_start >= right_start
        (left if in_right else right).pop("noise_frames", None)

    left.pop("transition_to_next", None)
    return left, right

def _best_split(normed: np.ndarray, min_side: int) -> tuple[int, float]:
    """
    Find the split index k (0 < min_side <= k <= N-min_side) that maximises
    cosine_dist(centroid(normed[:k]), centroid(normed[k:])).

    Uses prefix sums — O(N·D), no pairwise matrix.

    Returns (k, dist).  Returns (-1, 0.0) when the array is too short.
    """
    N = len(normed)
    if N < 2 * min_side:
        return -1, 0.0

    prefix = np.cumsum(normed, axis=0)         # (N, D) — sum of first k rows
    total  = prefix[-1]                        # (D,)

    ks = np.arange(min_side, N - min_side + 1, dtype=np.int64)  # candidate split indices
    left_sums  = prefix[ks - 1]               # (M, D)
    right_sums = total - prefix[ks - 1]       # (M, D)

    counts_l = ks[:, None].astype(np.float32)
    counts_r = (N - ks)[:, None].astype(np.float32)
    cl = left_sums  / counts_l                # mean of left half
    cr = right_sums / counts_r                # mean of right half

    nl = np.linalg.norm(cl, axis=1, keepdims=True).clip(1e-8)
    nr = np.linalg.norm(cr, axis=1, keepdims=True).clip(1e-8)
    cl /= nl
    cr /= nr

    dists = 1.0 - np.einsum("md,md->m", cl, cr)
    bm    = int(np.argmax(dists))
    return int(ks[bm]), float(dists[bm])


def _split_fades(
    shots:         list[dict],
    embeddings:    np.ndarray,
    frame_numbers: np.ndarray,
    source_times:  np.ndarray,
    cfg:           dict,
) -> list[dict]:
    """
    Detect slow fade-to-dark / fade-in transitions inside long content shots
    using a centroid-of-halves split sweep.

    For each content shot whose duration >= fade_split_min_duration_sec:
      1. Find the split point k that maximises dist(centroid_left, centroid_right).
      2. If that distance > fade_split_threshold, split the shot at k.
      3. Recurse independently on both halves — handles multi-fade shots.

    Guards:
      - Only content shots (shot_type == "content").
      - Minimum duration guard prevents wasted work on short shots.
      - Minimum side size (DEFAULT_FADE_SPLIT_MIN_SIDE_FRAMES) prevents
        degenerate single-frame splits.
      - frame_numbers must be sorted (guaranteed by the merge step).

    After all splits, shot_ids are NOT reindexed — the caller (lambda_handler)
    reindexes the full list once after all post-processing passes.
    """
    threshold = float(cfg.get("fade_split_threshold",        DEFAULT_FADE_SPLIT_THRESHOLD))
    min_dur   = float(cfg.get("fade_split_min_duration_sec", DEFAULT_FADE_SPLIT_MIN_DURATION_SEC))
    min_side  = int(DEFAULT_FADE_SPLIT_MIN_SIDE_FRAMES)

    def _split_shot(shot: dict) -> list[dict]:
        dur = shot.get("end_time", 0.0) - shot.get("start_time", 0.0)
        if dur < min_dur:
            return [shot]
        if shot.get("shot_type") != "content":
            return [shot]

        lo = int(np.searchsorted(frame_numbers, shot["start_frame"], side="left"))
        hi = int(np.searchsorted(frame_numbers, shot["end_frame"],   side="right"))
        sub = embeddings[lo:hi].astype(np.float32)
        if len(sub) < 2 * min_side:
            return [shot]

        norms = np.linalg.norm(sub, axis=1, keepdims=True).clip(1e-8)
        normed = sub / norms

        k, dist = _best_split(normed, min_side)
        if k < 0 or dist <= threshold:
            return [shot]

        # Split between embedding rows (lo+k-1) and (lo+k).
        # frame_numbers[split_row] - frame_numbers[split_row-1] can be > 1
        # (embeddings are sampled, not every frame).  Set left.end_frame to
        # frame_numbers[split_row]-1 so that right.start_frame - left.end_frame == 1,
        # satisfying the gap=1 continuity invariant.  _recompute_shot_centroid uses
        # searchsorted(side="right") for end_frame, so the left slice still covers
        # exactly rows [lo, split_row) even though frame_numbers[split_row]-1 is not
        # in the array.
        split_row    = lo + k
        right_start  = int(frame_numbers[split_row])

        left, right = _split_halves(shot, right_start)

        left["end_frame"]    = right_start - 1
        left["end_time"]     = float(source_times[split_row - 1])

        right["start_frame"]    = right_start
        right["start_time"]     = float(source_times[split_row])

        _recompute_shot_centroid(left,  embeddings, frame_numbers)
        _recompute_shot_centroid(right, embeddings, frame_numbers)

        # Abort if either half centroid collapsed to sentinel — this happens when
        # the split point lands inside a black region of a fade-through-black shot.
        # A content shot with a sentinel centroid fails validation; the unsplit
        # original is valid, so bail out rather than producing an invalid state.
        d_left_sentinel  = _cosine_dist(left["centroid_embedding"],  _SENTINEL.tolist())
        d_right_sentinel = _cosine_dist(right["centroid_embedding"], _SENTINEL.tolist())
        if d_left_sentinel < _SENTINEL_DIST_MIN or d_right_sentinel < _SENTINEL_DIST_MIN:
            log.info(
                "Fade split: shot %d aborted — split at frame %d|%d produced "
                "sentinel centroid (d_left=%.4f, d_right=%.4f)",
                shot["shot_id"], right_start - 1, right_start,
                d_left_sentinel, d_right_sentinel,
            )
            return [shot]

        log.info(
            "Fade split: shot %d (%.1fs) split at frame %d|%d "
            "(dist=%.4f > thr=%.4f) → %d + %d frames",
            shot["shot_id"], dur,
            right_start - 1, right_start,
            dist, threshold,
            left["frame_count"], right["frame_count"],
        )

        return _split_shot(left) + _split_shot(right)

    result: list[dict] = []
    n_splits = 0
    for shot in shots:
        parts = _split_shot(shot)
        result.extend(parts)
        if len(parts) > 1:
            n_splits += len(parts) - 1

    if n_splits > 0:
        log.info("Fade split: %d shot(s) split into %d additional shots",
                 n_splits, n_splits)

    return result


# =============================================================================
# Drift split (runs after fade split — catches slow scene drift in
# very long shots where no adjacent-frame or centroid-of-halves signal
# is strong enough)
# =============================================================================

def _split_drift_shots(
    shots:         list[dict],
    embeddings:    np.ndarray,
    frame_numbers: np.ndarray,
    source_times:  np.ndarray,
    cfg:           dict,
) -> list[dict]:
    """
    Split very long content shots at slow scene-drift transitions.

    Rationale: a gradual transition (camera move between sets, live
    broadcast studio→exterior) never produces an adjacent-frame cosine
    distance above shot_boundary_threshold, and its centroid-of-halves
    signal is diluted when the shot is much longer than the transition.
    But the distance measured across a 1-second stride peaks sharply at
    the transition (measured 0.41–0.94 on confirmed false merges vs
    <0.40 on almost all legitimate long shots).

    For each content shot longer than drift_split_min_duration_sec:
      1. Compute cosine distance between each embedding row and the row
         ~1s later (vectorised, O(N·D)).
      2. Find the peak.  If peak > drift_split_threshold, split at the
         largest adjacent-pair distance inside that 1s span (most
         concentrated single step — best frame-accurate estimate of the
         transition midpoint).
      3. Recurse on both halves.

    Sentinel (black) rows are excluded from stride pairs.  Split aborts
    if either half would collapse to a sentinel centroid, mirroring the
    fade-split guard.
    """
    threshold = float(
        cfg.get("drift_split_threshold", DEFAULT_DRIFT_SPLIT_THRESHOLD))
    min_dur = float(
        cfg.get("drift_split_min_duration_sec",
                DEFAULT_DRIFT_SPLIT_MIN_DURATION_SEC))
    min_side_sec = float(DEFAULT_DRIFT_SPLIT_MIN_SIDE_SEC)

    def _split_shot(shot: dict) -> list[dict]:
        dur = shot.get("end_time", 0.0) - shot.get("start_time", 0.0)
        if dur < min_dur or shot.get("shot_type") != "content":
            return [shot]

        lo = int(np.searchsorted(frame_numbers, shot["start_frame"], side="left"))
        hi = int(np.searchsorted(frame_numbers, shot["end_frame"],   side="right"))
        X = embeddings[lo:hi].astype(np.float32)
        T = source_times[lo:hi]
        N = len(X)
        if N < 8:
            return [shot]

        norms = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
        Xn = X / norms
        is_sent = (1.0 - Xn @ _SENTINEL) < _SENTINEL_DIST_MIN

        # 1s-stride pairs
        jj = np.searchsorted(T, T + 1.0, side="left")
        valid = jj < N
        ii = np.nonzero(valid)[0]
        jj = jj[valid]
        mask = ~(is_sent[ii] | is_sent[jj])
        ii, jj = ii[mask], jj[mask]
        if len(ii) == 0:
            return [shot]

        # Drop stride pairs whose span cannot contain a margin-legal
        # split row — otherwise a peak hugging the shot edge wins the
        # argmax and suppresses a genuine mid-shot drift.  A split row r
        # (right side starts at r) is legal iff T[r-1]-t0 >= min_side
        # and t1-T[r] >= min_side; a pair's candidate rows are
        # (ii[p], jj[p]], so the pair is viable iff jj >= first legal r
        # and ii + 1 <= last legal r.
        t0, t1 = float(T[0]), float(T[-1])
        legal = np.nonzero(
            (T[:-1] - t0 >= min_side_sec)
            & (t1 - T[1:] >= min_side_sec))[0] + 1   # legal split rows r
        if len(legal) == 0:
            return [shot]
        first_legal, last_legal = int(legal[0]), int(legal[-1])
        viable = (jj >= first_legal) & (ii + 1 <= last_legal)
        ii, jj = ii[viable], jj[viable]
        if len(ii) == 0:
            return [shot]

        stride_d = 1.0 - np.einsum("nd,nd->n", Xn[ii], Xn[jj])
        p = int(np.argmax(stride_d))
        if float(stride_d[p]) <= threshold:
            return [shot]

        # Split at the largest adjacent-pair step inside the peak span,
        # respecting the min-side margin on both ends.  The viability
        # filter above guarantees at least one legal row in this span.
        span_lo, span_hi = int(ii[p]), int(jj[p])
        adj = 1.0 - np.einsum(
            "nd,nd->n", Xn[span_lo:span_hi], Xn[span_lo + 1:span_hi + 1])
        best_row, best_d = -1, -1.0
        for off in range(len(adj)):
            row = span_lo + off + 1     # right side starts at this row
            if T[row - 1] - t0 < min_side_sec or t1 - T[row] < min_side_sec:
                continue
            if float(adj[off]) > best_d:
                best_d, best_row = float(adj[off]), row
        if best_row < 0:
            return [shot]

        split_row   = lo + best_row
        right_start = int(frame_numbers[split_row])

        left, right = _split_halves(shot, right_start)
        left["end_frame"]       = right_start - 1
        left["end_time"]        = float(source_times[split_row - 1])
        right["start_frame"]    = right_start
        right["start_time"]     = float(source_times[split_row])

        _recompute_shot_centroid(left,  embeddings, frame_numbers)
        _recompute_shot_centroid(right, embeddings, frame_numbers)

        d_l = _cosine_dist(left["centroid_embedding"],  _SENTINEL.tolist())
        d_r = _cosine_dist(right["centroid_embedding"], _SENTINEL.tolist())
        if d_l < _SENTINEL_DIST_MIN or d_r < _SENTINEL_DIST_MIN:
            log.info(
                "Drift split: shot %d aborted — sentinel centroid "
                "(d_left=%.4f, d_right=%.4f)",
                shot["shot_id"], d_l, d_r)
            return [shot]

        log.info(
            "Drift split: shot %d (%.1fs) split at frame %d|%d "
            "(stride_peak=%.4f > thr=%.4f, step=%.4f) → %.1fs + %.1fs",
            shot["shot_id"], dur, right_start - 1, right_start,
            float(stride_d[p]), threshold, best_d,
            left["end_time"] - left["start_time"],
            right["end_time"] - right["start_time"],
        )
        return _split_shot(left) + _split_shot(right)

    result: list[dict] = []
    n_splits = 0
    for shot in shots:
        parts = _split_shot(shot)
        result.extend(parts)
        n_splits += len(parts) - 1

    if n_splits > 0:
        log.info("Drift split: %d additional shot(s) created", n_splits)
    return result


# =============================================================================
# Minimum-duration absorb (runs before adjacent drift merge)
# =============================================================================

def _absorb_minimum_shots(
    shots:         list[dict],
    embeddings:    np.ndarray,
    frame_numbers: np.ndarray,
    cfg:           dict,
) -> list[dict]:
    """
    Unconditionally absorb shots shorter than min_shot_duration into the
    more similar of their two neighbours.

    These sub-threshold shots (e.g. 0.08 s, 0.16 s, 0.20 s) cannot be real
    editorial shots at any frame rate.  They arise from motion-blur boundary
    frames, film-leader artefacts, or transition slivers that noise_merge
    cannot handle because their neighbours are genuinely dissimilar scenes.

    Unlike noise_merge, this pass applies NO similarity gate — duration alone
    decides WHETHER to absorb.  Similarity only decides the DIRECTION: the
    sliver's centroid is compared against both neighbours and it merges into
    the closer one.  A sliver at a genuine scene boundary (e.g. a lightning
    flash opening the next shot) otherwise pollutes the wrong shot's frame
    range and centroid.  min_shot_duration must be < max_noise_duration
    (enforced by input_schema validation) so this pass only removes artefact
    slivers, not the noise shots handled by noise_merge.

    Absorbing backward extends the previous shot's end boundary; absorbing
    forward extends the next shot's start boundary.  Either way the absorbing
    shot's centroid is recomputed to include the absorbed frames.
    """
    min_dur = float(cfg.get("min_shot_duration", DEFAULT_MIN_SHOT_DURATION))

    if len(shots) < 2:
        return shots

    result: list[dict] = []
    n_absorbed = 0

    for i, shot in enumerate(shots):
        dur = shot.get("end_time", 0.0) - shot.get("start_time", 0.0)
        if dur >= min_dur or shot.get("shot_type") == "black":
            result.append(shot)
            continue

        prev = result[-1] if result and result[-1].get("shot_type") != "black" else None
        nxt  = (shots[i + 1]
                if i + 1 < len(shots) and shots[i + 1].get("shot_type") != "black"
                else None)

        if prev is None and nxt is None:
            result.append(shot)
            continue

        if prev is not None and nxt is not None:
            d_prev = _cosine_dist(shot["centroid_embedding"], prev["centroid_embedding"])
            d_next = _cosine_dist(shot["centroid_embedding"], nxt["centroid_embedding"])
            into_prev = d_prev <= d_next
        else:
            into_prev = prev is not None

        if into_prev:
            merged = dict(prev)
            merged["end_frame"]    = shot["end_frame"]
            merged["end_time"]     = shot["end_time"]
            _recompute_shot_centroid(merged, embeddings, frame_numbers)
            result[-1] = merged
            absorbed_into = prev["shot_id"]
        else:
            # Extend the next shot backward in place; the loop appends it
            # on a later iteration (and may absorb it again if still short).
            nxt["start_frame"]    = shot["start_frame"]
            nxt["start_time"]     = shot["start_time"]
            _recompute_shot_centroid(nxt, embeddings, frame_numbers)
            absorbed_into = nxt["shot_id"]

        n_absorbed += 1
        log.info(
            "Min-duration absorb: shot %d (%.3fs) absorbed into shot %d (%s)",
            shot["shot_id"], dur, absorbed_into,
            "prev" if into_prev else "next",
        )

    if n_absorbed > 0:
        log.info("Min-duration absorb: %d shot(s) absorbed, %d → %d shots",
                 n_absorbed, len(shots), len(result))

    return result


# =============================================================================
# Adjacent drift merge (second pass — runs before noise merge)
# =============================================================================

def _merge_adjacent_drift_shots(
    shots:         list[dict],
    embeddings:    np.ndarray,
    frame_numbers: np.ndarray,
    cfg:           dict,
) -> list[dict]:
    """
    Second-pass merge for motion-blur-induced false cuts.

    Adjacent shot pairs are merged when:
      1. cosine_dist(A.centroid, B.centroid) < adjacent_centroid_threshold
      2. Neither shot is a black shot or freeze frame
      3. Neither shot carries a transition_to_next marker

    Runs before _merge_noise_shots so that noise detection operates on
    corrected centroids.  The centroid of the merged shot is recomputed
    from the full frame range, averaging out the blurred boundary frames.
    """
    threshold = float(
        cfg.get("adjacent_centroid_threshold", DEFAULT_ADJACENT_CENTROID_THRESHOLD))
    max_merged_sec = float(
        cfg.get("max_adjacent_merge_duration", DEFAULT_MAX_ADJACENT_MERGE_DURATION))

    if len(shots) < 2:
        return shots

    result: list[dict] = []
    n_merged = 0

    for shot in shots:
        if (
            result
            and not result[-1].get("transition_to_next")
            and not shot.get("transition_to_next")
            and result[-1].get("shot_type") not in ("black", "static")
            and shot.get("shot_type") not in ("black", "static")
        ):
            merged_dur = shot.get("end_time", 0.0) - result[-1].get("start_time", 0.0)
            if merged_dur > max_merged_sec:
                result.append(shot)
                continue
            dist = _cosine_dist(
                result[-1]["centroid_embedding"],
                shot["centroid_embedding"])
            if dist < threshold:
                merged = dict(result[-1])
                merged["end_frame"]    = shot["end_frame"]
                merged["end_time"]     = shot["end_time"]
                if shot.get("shot_type") == "content":
                    merged["shot_type"] = "content"
                _recompute_shot_centroid(merged, embeddings, frame_numbers)
                result[-1] = merged
                n_merged += 1
                log.info(
                    "Adjacent drift merge: shot %d + %d "
                    "(dist=%.4f < thr=%.4f)",
                    result[-1]["shot_id"], shot["shot_id"], dist, threshold,
                )
                continue
        result.append(shot)

    if n_merged > 0:
        log.info("Adjacent drift merge: %d pairs merged, %d → %d shots",
                 n_merged, len(shots), len(result))

    return result


# =============================================================================
# Noise merge
# =============================================================================

def _noise_merge_blocked(prev, shot, nxt, cfg, dist=None):
    """
    Why `_merge_noise_shots` will refuse to heal `shot`, or None if it would heal it.

    Single source of truth, shared with `_validate_continuity`'s check 2. That check declares a
    single-frame shot between two near-identical neighbours a violation, on the premise that the
    noise merge should have healed it. Any condition under which the merge deliberately declines
    must therefore appear here, or validation demands a heal the pipeline is designed never to
    perform and the run fails on correct output. Deriving both from one predicate is what stops
    them drifting apart again — the first version of this guard covered only the black-neighbour
    case and a black *self* still failed the whole run.
    """
    if shot.get("shot_type") == "black":
        # A black shot is excluded from candidacy outright: merging it would produce a
        # "content" shot spanning black frames, which every merge pass exists to prevent.
        # So a one-frame black flash inside a continuous scene is unhealable BY DESIGN.
        return "shot is black"
    if prev.get("shot_type") == "black" or nxt.get("shot_type") == "black":
        # Black carries the sentinel centroid, so the neighbour distance is identically 0 and
        # always trips the threshold; and a reclassified near-black neighbour has a real
        # centroid that can sit close to a dark one. Either way the merge refuses.
        return "neighbour is black"
    if shot.get("transition_to_next"):
        return "resolved dissolve"
    dur = shot.get("end_time", 0.0) - shot.get("start_time", 0.0)
    if dur >= float(cfg.get("max_noise_duration", DEFAULT_MAX_NOISE_DURATION)):
        return "longer than max_noise_duration"
    if dist is not None and dist >= float(
            cfg.get("noise_distance_threshold", DEFAULT_NOISE_DISTANCE_THRESHOLD)):
        # SYNC_POP_THRESHOLD (0.10) is below the 0.18 default, so normally a shot that trips
        # check 2 also clears the merge. A caller who sets noise_distance_threshold under 0.10
        # inverts that, and without this the tighter setting would fail validation.
        return "neighbours not similar enough for the merge threshold"
    return None


def _merge_noise_shots(
    shots:         list[dict],
    embeddings:    np.ndarray,
    frame_numbers: np.ndarray,
    cfg:           dict,
) -> list[dict]:
    """
    Detect and merge noise shots — very short artefact shots between
    visually similar neighbours.

    Detection criteria for shot[i]:
      1. duration < max_noise_duration (default 0.5s)
      2. Both neighbours exist
      3. Not a resolved dissolve (no transition_to_next field)
      4. cosine_dist(prev.centroid, nxt.centroid) < noise_distance_threshold

    Resolution: merge [prev, noise, nxt] into one shot.
    Record noise frame range in noise_frames field.
    """
    similar_thr = float(
        cfg.get("noise_distance_threshold", DEFAULT_NOISE_DISTANCE_THRESHOLD))

    if len(shots) < 3:
        return shots

    result: list[dict] = []
    n_merged = 0
    i = 0

    while i < len(shots):
        shot = shots[i]

        is_candidate = i > 0 and i < len(shots) - 1

        if is_candidate and result:
            prev = result[-1]
            nxt = shots[i + 1]
            # Candidacy conditions live in _noise_merge_blocked so that
            # _validate_continuity cannot demand a heal that is refused here.
            if not _noise_merge_blocked(prev, shot, nxt, cfg):
                d = _cosine_dist(
                    prev["centroid_embedding"],
                    nxt["centroid_embedding"])

                if d < similar_thr:
                    merged = dict(prev)
                    merged["end_frame"] = nxt["end_frame"]
                    merged["end_time"] = nxt["end_time"]
                    prev_type = prev.get("shot_type", "content")
                    nxt_type  = nxt.get("shot_type",  "content")
                    if prev_type == "content" or nxt_type == "content":
                        merged["shot_type"] = "content"
                    elif prev_type == "static" or nxt_type == "static":
                        merged["shot_type"] = "static"
                    else:
                        merged["shot_type"] = "black"
                    merged["noise_frames"] = {
                        "frames": [shot["start_frame"], shot["end_frame"]],
                        "start_time": shot.get("start_time", 0.0),
                        "end_time": shot.get("end_time", 0.0),
                    }
                    # Carry forward anomaly markers from prev if present
                    if prev.get("sync_pop"):
                        merged["sync_pop"] = prev["sync_pop"]
                    if prev.get("flash_frame"):
                        merged["flash_frame"] = prev["flash_frame"]
                    # Carry forward transition_to_next from prev if present
                    if prev.get("transition_to_next"):
                        merged["transition_to_next"] = prev[
                            "transition_to_next"]

                    _recompute_shot_centroid(
                        merged, embeddings, frame_numbers)

                    result[-1] = merged
                    n_merged += 1
                    i += 2
                    log.info(
                        "Noise merge: shot %d (frames %d-%d, %.3fs) "
                        "neighbour_dist=%.4f",
                        shot["shot_id"], shot["start_frame"],
                        shot["end_frame"],
                        shot.get("end_time", 0.0) - shot.get("start_time", 0.0), d,
                    )
                    continue

        result.append(shot)
        i += 1

    if n_merged > 0:
        log.info("Noise merge: %d resolved, %d → %d shots",
                 n_merged, len(shots), len(result))

    return result


# =============================================================================
# Near-black reclassification (runs after early entropy pass, before merges)
# =============================================================================

def _reclassify_near_black_shots(shots, cfg):
    """
    Reclassify content/static shots whose entropy_mean is below
    near_black_entropy_max as 'black'.

    This catches shots that the per-frame black detector missed — typically
    because the sentinel embedding threshold was not breached at the frame
    level (e.g. near-black rather than pure-black frames), or because a merge
    in a prior window produced a shot whose inherited label is stale.

    Must run AFTER _compute_shot_entropy so entropy_mean is populated,
    and BEFORE the drift/noise merge passes so near-black shots cannot be
    absorbed as content.
    """
    threshold = float(
        cfg.get("near_black_entropy_max", DEFAULT_NEAR_BLACK_ENTROPY_MAX))

    # Legacy window metadata without an "entropy" field is zero-filled
    # upstream; reclassifying on all-zero entropy would silently flip
    # every shot to black.  Real content always has some entropy.
    if not any(shot.get("entropy_max", 0.0) > 0.0 for shot in shots):
        log.warning(
            "Near-black reclassify: skipped — no entropy data "
            "(all entropy_max == 0.0)")
        return shots

    n_reclassified = 0
    for shot in shots:
        if shot.get("shot_type") == "black":
            continue
        if shot.get("entropy_mean", 999.0) < threshold:
            shot["shot_type"] = "black"
            n_reclassified += 1
    if n_reclassified > 0:
        log.info("Near-black reclassify: %d shot(s) → black", n_reclassified)
    return shots


# =============================================================================
# Per-shot entropy
# =============================================================================

def _compute_shot_entropy(shots, frame_numbers, entropy_array):
    for shot in shots:
        lo, hi = _shot_frame_slice(shot, frame_numbers)
        se = entropy_array[lo:hi]
        if len(se) > 0:
            shot["entropy_mean"] = round(float(np.mean(se)), 4)
            shot["entropy_min"] = round(float(np.min(se)), 4)
            shot["entropy_max"] = round(float(np.max(se)), 4)
        else:
            shot["entropy_mean"] = 0.0
            shot["entropy_min"] = 0.0
            shot["entropy_max"] = 0.0


# =============================================================================
# Per-shot intra-shot dispersion
# =============================================================================

def _compute_shot_dispersion(shots, frame_numbers, embeddings):
    """How much a shot's own frames VARY, in embedding space.

    The centroid is a mean, and averaging discards the spread — so every shot
    property downstream describes the shot's average appearance and nothing
    describes how uniform it is. That spread is the difference between a locked-
    off anchor camera and an animated ident that happen to share a centroid, and
    it costs nothing to keep: the frame embeddings are already in RAM here, and
    `frame_coherence` is a number `_recompute_shot_centroid` already computes
    (as the pre-normalisation length of the mean) and throws away.

    Four fields, chosen so none is derivable from another:

      frame_coherence       ||mean of the unit frame vectors||, in [0, 1]. 1.0
                            means every frame is identical; lower means the
                            frames fan out. Note the mean frame-to-centroid
                            cosine distance is EXACTLY 1 - this, so publishing
                            both would publish one number twice.
      frame_dispersion_max  the furthest single frame from the centroid. The
                            tail the mean hides — one dissolve or wipe frame on
                            an otherwise static shot.
      frame_motion_mean     mean cosine distance between ADJACENT frames. A
                            proxy for motion, not a measurement of it: no
                            optical flow is computed, and a hard cut inside a
                            missed boundary reads the same as fast pans. High
                            coherence with high adjacent motion is a shot that
                            keeps returning to the same look (a flashing
                            graphic); low coherence with low adjacent motion is
                            a slow drift across the whole shot.
      frame_motion_max      the largest adjacent-frame step — a within-shot
                            boundary candidate the shot detector did not take.

    Undefined rather than zero: a shot with no embedded frames reports null for
    all four, and a single-frame shot reports null for the two motion fields.
    Zero would read as "measured, and perfectly still".
    """
    for shot in shots:
        lo, hi = _shot_frame_slice(shot, frame_numbers)
        X = embeddings[lo:hi]
        if len(X) == 0:
            shot["frame_coherence"] = None
            shot["frame_dispersion_max"] = None
            shot["frame_motion_mean"] = None
            shot["frame_motion_max"] = None
            continue

        normed = _unit_rows(X.astype(np.float32))
        raw = normed.mean(axis=0)
        coherence = float(np.linalg.norm(raw))
        shot["frame_coherence"] = round(min(coherence, 1.0), 4)

        if coherence > 1e-8:
            to_centroid = 1.0 - np.clip(normed @ (raw / coherence), -1.0, 1.0)
            shot["frame_dispersion_max"] = round(float(to_centroid.max()), 4)
        else:
            # frames cancel out entirely — no centroid direction to measure
            # against, so the spread is maximal by construction
            shot["frame_dispersion_max"] = 2.0

        if len(normed) >= 2:
            step = 1.0 - np.clip(
                np.sum(normed[:-1] * normed[1:], axis=1), -1.0, 1.0)
            shot["frame_motion_mean"] = round(float(step.mean()), 4)
            shot["frame_motion_max"] = round(float(step.max()), 4)
        else:
            shot["frame_motion_mean"] = None
            shot["frame_motion_max"] = None


# =============================================================================
# Continuity validation (6 invariants — no scene checks)
# =============================================================================

def _validate_continuity(shots, frame_index, fps, cfg):
    violations = []

    if not shots and not frame_index:
        return

    shots_sorted = sorted(shots, key=lambda s: s["start_frame"])

    # Check 1: geometry
    for s in shots_sorted:
        if s["end_frame"] < s["start_frame"]:
            violations.append(
                f"[1] Shot {s['shot_id']}: end < start")

    # Check 2: unresolved sync-pop
    #
    # A single-frame shot between two near-identical neighbours should have been healed by the
    # noise merge, so its survival means a bug — EXCEPT where the merge deliberately declines.
    # Those exceptions are not restated here: _noise_merge_blocked owns them, because a partial
    # restatement is exactly the bug this check keeps producing. A single frame bounded by black
    # was the first case found; a single frame that IS black (a one-frame black flash inside a
    # continuous scene, unmergeable by design) was the second, and it failed a whole asset.
    for i, s in enumerate(shots_sorted):
        if s["end_frame"] != s["start_frame"]:
            continue
        if i == 0 or i == len(shots_sorted) - 1:
            continue
        pc = np.array(shots_sorted[i-1]["centroid_embedding"], np.float32)
        nc = np.array(shots_sorted[i+1]["centroid_embedding"], np.float32)
        pn = pc / max(float(np.linalg.norm(pc)), 1e-8)
        nn = nc / max(float(np.linalg.norm(nc)), 1e-8)
        d = float(1.0 - pn @ nn)
        if d >= SYNC_POP_THRESHOLD:
            continue
        blocked = _noise_merge_blocked(
            shots_sorted[i-1], s, shots_sorted[i+1], cfg, dist=d)
        if blocked:
            log.debug("shot %s: single frame, unhealable (%s) — not a violation",
                      s["shot_id"], blocked)
            continue
        violations.append(
            f"[2] Shot {s['shot_id']}: unresolved sync-pop")

    # Check 3: boundary coverage
    if shots_sorted and frame_index:
        if shots_sorted[0]["start_frame"] != frame_index[0]:
            violations.append(
                f"[3] First shot start {shots_sorted[0]['start_frame']} "
                f"!= frame_index[0] {frame_index[0]}")
        if shots_sorted[-1]["end_frame"] != frame_index[-1]:
            violations.append(
                f"[3] Last shot end {shots_sorted[-1]['end_frame']} "
                f"!= frame_index[-1] {frame_index[-1]}")

    # Check 4: shot_id sequential
    for i, s in enumerate(shots_sorted):
        if s["shot_id"] != i:
            violations.append(
                f"[4] Position {i} has shot_id={s['shot_id']}")

    # Check 5: gap=1 between consecutive shots
    for i in range(1, len(shots_sorted)):
        gap = shots_sorted[i]["start_frame"] - shots_sorted[i-1]["end_frame"]
        if gap != 1:
            violations.append(
                f"[5] Shot {shots_sorted[i-1]['shot_id']}→"
                f"{shots_sorted[i]['shot_id']}: gap={gap}")

    # Check 6: no sentinel in content shot
    for s in shots:
        if s.get("shot_type") != "content":
            continue
        c = np.array(s["centroid_embedding"], np.float32)
        cn = float(np.linalg.norm(c))
        if cn < 1e-8:
            violations.append(f"[6] Shot {s['shot_id']}: zero centroid")
            continue
        d = float(1.0 - (c / cn) @ _SENTINEL)
        if d < _SENTINEL_DIST_MIN:
            violations.append(
                f"[6] Shot {s['shot_id']}: sentinel centroid (d={d:.4f})")

    if violations:
        msg = (f"Validation FAILED ({len(violations)}):\n"
               + "\n".join(f"  {v}" for v in violations))
        log.error(msg)
        raise ValueError(msg)

    log.info("Validation PASSED: %d shots", len(shots))


# =============================================================================
# Frame-to-window index
# =============================================================================

def _build_frame_to_window_index(
    window_metas: list[dict],
    windows:      list[dict],
) -> dict[int, str]:
    """
    Build a mapping from source frame number to the S3 frames_key of the
    window archive that persisted it.

    Uses exact frame_index lists from window metadata (not start/end range)
    to correctly handle the 30-second overlap region between windows.
    First-window-wins on duplicates — consistent with the frame ownership
    deduplication logic already applied during the merge step.

    Returns: {source_frame_number: frames_key}
    """
    frame_to_window: dict[int, str] = {}
    for meta, window in zip(window_metas, windows):
        frames_key = window.get("frames_key")
        if not frames_key:
            continue
        for fn in meta.get("frame_index", []):
            if fn not in frame_to_window:
                frame_to_window[fn] = frames_key
    log.info(
        "Frame-to-window index: %d frames across %d windows",
        len(frame_to_window), len(windows),
    )
    return frame_to_window


# =============================================================================
# Frame selection — distance matrix
# =============================================================================

def _select_shot_frames(
    shot:          dict,
    embeddings:    np.ndarray,
    frame_numbers: np.ndarray,
    cfg:           dict | None = None,
) -> list[int]:
    """
    Select up to 16 representative source frame numbers for a shot.

    Special cases (evaluated in order):
      1. shot_type == "static"  → 4 equally distributed frames by position
      2. frame_count <= 4 → all frames

    General case — N×N cosine distance matrix:
      - Flatten upper triangle to (distance, i, j) pairs
      - Sort descending by distance
      - Iterate pairs, accumulate unique frame indices
      - Stop when 16 selected or current pair distance < min_dist
      - Return selected frame numbers in temporal order

    cfg key: frame_select_min_dist (default 0.05)

    Returns list of source frame numbers (not embedding row indices).
    """
    min_dist = float(
        (cfg or {}).get("frame_select_min_dist", DEFAULT_FRAME_SELECT_MIN_DIST)
    )
    lo = int(np.searchsorted(frame_numbers, shot["start_frame"], side="left"))
    hi = int(np.searchsorted(frame_numbers, shot["end_frame"],   side="right"))
    shot_fns  = frame_numbers[lo:hi].tolist()   # source frame numbers
    shot_embs = embeddings[lo:hi]               # [N, 384]
    N = len(shot_fns)

    if N == 0:
        return []

    # ── Special case 1: static / freeze shot ─────────────────────────
    if shot.get("shot_type") == "static":
        if N <= 4:
            return list(shot_fns)
        # 4 equally distributed by position
        indices = [round(i * (N - 1) / 3) for i in range(4)]
        return [shot_fns[i] for i in indices]

    # ── Special case 2: very short shot ───────────────────────────────
    if N <= 4:
        return list(shot_fns)

    # ── General case: distance matrix selection ────────────────────────
    # Normalise embeddings for cosine similarity
    norms = np.linalg.norm(shot_embs, axis=1, keepdims=True).clip(1e-8)
    normed = (shot_embs / norms).astype(np.float32)

    # Full N×N cosine distance matrix (upper triangle only needed)
    sim = normed @ normed.T                       # [N, N] similarities
    dist = (1.0 - sim).clip(0.0)                 # [N, N] distances

    # Collect upper-triangle pairs (i < j) sorted by distance descending
    ii, jj = np.triu_indices(N, k=1)
    pair_dists = dist[ii, jj]
    order = np.argsort(pair_dists)[::-1]          # descending

    selected: list[int] = []   # embedding-local indices, in order added
    seen: set[int] = set()

    for idx in order:
        d  = float(pair_dists[idx])
        if d < min_dist and len(selected) >= 4:
            break
        i, j = int(ii[idx]), int(jj[idx])
        for k in (i, j):
            if k not in seen:
                seen.add(k)
                selected.append(k)
            if len(selected) == 16:
                break
        if len(selected) == 16:
            break

    # Return source frame numbers in temporal order
    selected_sorted = sorted(selected)
    return [shot_fns[k] for k in selected_sorted]


# =============================================================================
# Per-shot frame repackaging (streaming, parallel upload)
# =============================================================================

def _repack_shot_frames_streaming(
    shots:            list[dict],
    frame_to_window:  dict[int, str],
    proxy_bucket:     str,
    proxy_prefix:      str,
    max_workers:      int = 10,
) -> tuple[int, int]:
    """
    Stream-oriented repack: download each window archive once, extract
    only needed frames, and upload per-shot tars as soon as all frames
    for a shot are collected.  On completion, each shot in `shots`
    has `frames_key` set to the uploaded S3 key, or None on failure.

    Peak memory ≈ one archive + buffered ready-to-upload shot tars.

    Returns (n_ok, n_errors).
    """
    _local = threading.local()

    def _get_client():
        if not hasattr(_local, "s3"):
            _local.s3 = boto3.session.Session().client("s3")
        return _local.s3

    # ── Build lookup structures ───────────────────────────────────────
    # frame_name → list of shot_ids that need it
    frame_to_shots: dict[str, list[int]] = defaultdict(list)
    # shot_id → set of frame_names still pending
    shot_pending:   dict[int, set[str]] = {}
    # shot_id → {frame_name: jpeg_bytes}
    shot_frames:    dict[int, dict[str, bytes]] = {}
    # shot_id → shot dict
    shot_by_id:     dict[int, dict] = {}

    for shot in shots:
        sid = shot["shot_id"]
        shot_by_id[sid] = shot
        needed = set()
        for fn in shot.get("selected_frames", []):
            name = f"frame_{fn:010d}.jpg"
            frame_to_shots[name].append(sid)
            needed.add(name)
        shot_pending[sid] = needed
        shot_frames[sid] = {}

    # Which archives do we need?
    archive_to_names: dict[str, set[str]] = defaultdict(set)
    for shot in shots:
        for fn in shot.get("selected_frames", []):
            fk = frame_to_window.get(fn)
            if fk:
                archive_to_names[fk].add(f"frame_{fn:010d}.jpg")

    log.info(
        "Streaming repack: %d unique frames from %d archives "
        "for %d shots",
        len(frame_to_shots), len(archive_to_names), len(shots),
    )

    # ── Upload helper ─────────────────────────────────────────────────
    def _pack_and_upload(sid: int, collected: dict[str, bytes]):
        buf = io.BytesIO()
        with _tarfile.open(fileobj=buf, mode="w:gz",
                           compresslevel=1) as tar:
            for name in sorted(collected):
                data = collected[name]
                info = _tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        payload = buf.getvalue()

        fk = (f"{proxy_prefix}/shot_frames/"
              f"shot_{sid:04d}_frames.tar.gz")
        _get_client().put_object(
            Bucket=proxy_bucket, Key=fk,
            Body=payload, ContentType="application/gzip",
        )
        return sid, fk

    # ── Stream archives, dispatch uploads as shots become ready ───────
    # Downloads are prefetched a bounded number of archives ahead so the
    # (serial) tar-extract/distribute loop never waits on S3.  Extraction
    # stays in submission order — required by first-window-wins frame
    # ownership.  Peak extra memory = _PREFETCH_DEPTH × archive size
    # (~30 MB each), all in RAM — no /tmp usage.
    _PREFETCH_DEPTH = 4
    n_ok = n_err = 0
    pool = ThreadPoolExecutor(max_workers=max_workers)
    dl_pool = ThreadPoolExecutor(max_workers=_PREFETCH_DEPTH)
    pending_futures: dict = {}

    archive_items = list(archive_to_names.items())
    dl_futures = deque()
    next_submit = 0

    def _submit_downloads():
        nonlocal next_submit
        while (next_submit < len(archive_items)
               and len(dl_futures) < _PREFETCH_DEPTH):
            fk, names = archive_items[next_submit]
            dl_futures.append(
                (fk, names, dl_pool.submit(get_binary, proxy_bucket, fk)))
            next_submit += 1

    try:
        _submit_downloads()
        while dl_futures:
            frames_key, needed_names, fut = dl_futures.popleft()
            try:
                raw = fut.result()
            except Exception as exc:
                log.error(
                    "Failed to download archive s3://%s/%s: %s",
                    proxy_bucket, frames_key, exc,
                )
                raise
            _submit_downloads()

            remaining = set(needed_names)
            with _tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not remaining:
                        break
                    name = os.path.basename(member.name)
                    if name not in remaining:
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    jpeg = f.read()
                    remaining.discard(name)

                    # Distribute to every shot that needs this frame
                    for sid in frame_to_shots.get(name, []):
                        if name in shot_pending.get(sid, set()):
                            shot_frames[sid][name] = jpeg
                            shot_pending[sid].discard(name)

                            # Shot complete? → submit upload immediately
                            if not shot_pending[sid]:
                                collected = shot_frames.pop(sid)
                                fut = pool.submit(
                                    _pack_and_upload, sid, collected)
                                pending_futures[fut] = sid

            del raw  # free archive bytes

            if remaining:
                log.warning(
                    "Archive %s: %d frame(s) not found: %s",
                    frames_key, len(remaining), sorted(remaining)[:5],
                )

        # ── Handle shots with remaining frames (partial or empty) ─────────
        for sid in list(shot_frames.keys()):
            collected = shot_frames.pop(sid, {})
            if collected:
                # Partial — upload what we have
                log.warning(
                    "Shot %d: %d of %d frames found (partial)",
                    sid, len(collected),
                    len(collected) + len(shot_pending.get(sid, set())),
                )
                fut = pool.submit(_pack_and_upload, sid, collected)
                pending_futures[fut] = sid
            else:
                shot_by_id[sid]["frames_key"] = None
                n_err += 1
                log.error("Shot %d: no frames collected from any archive", sid)

        # ── Collect results ───────────────────────────────────────────────
        for fut in as_completed(pending_futures):
            sid = pending_futures[fut]
            try:
                _, fk = fut.result()
                shot_by_id[sid]["frames_key"] = fk
                n_ok += 1
            except Exception as exc:
                log.error("Shot %d upload failed: %s", sid, exc)
                shot_by_id[sid]["frames_key"] = None
                n_err += 1
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        dl_pool.shutdown(wait=False, cancel_futures=True)

    log.info(
        "Streaming repack done: %d ok  %d errors", n_ok, n_err)
    return n_ok, n_err


# =============================================================================
# Lambda handler
# =============================================================================

def _derive_timecodes(shots: list[dict], fps_str: str | None,
                      drop_frame: bool, tc_origin: str | None) -> None:
    """Derive EVERY shot timecode from frame numbers, in one place.

    Timecode in this pipeline is arithmetic — frame_to_smpte(frame, fps,
    drop_frame, origin) over the container's first-frame timecode (or
    00:00:00:00 NDF when the source carries none).  Until 2026-08-25 the
    strings were computed per sampled frame in the windows and carried
    through every merge/split as data, which is how post-hoc splits ended
    up with blanks and how a drop-frame arithmetic bug propagated.  Now the
    windows ship no strings at all and this single pass, just before
    shots.json is serialized, is the only producer: start/end timecodes
    from start_frame/end_frame, and the sync_pop/flash_frame anomaly
    record's timecode from its own frame.  end_timecode therefore
    describes the nominal end_frame (consistent with frame_range), which
    on PTS-hole boundaries differs from end_time's last-sampled basis.
    """
    if not fps_str:
        for shot in shots:
            shot.setdefault("start_timecode", "")
            shot.setdefault("end_timecode", "")
        return
    for shot in shots:
        shot["start_timecode"] = frame_to_smpte(
            shot["start_frame"], fps_str, drop_frame, tc_origin)
        shot["end_timecode"] = frame_to_smpte(
            shot["end_frame"], fps_str, drop_frame, tc_origin)
        for key in ("sync_pop", "flash_frame"):
            anomaly = shot.get(key)
            if anomaly and anomaly.get("frame") is not None:
                anomaly["timecode"] = frame_to_smpte(
                    anomaly["frame"], fps_str, drop_frame, tc_origin)


def lambda_handler(event: dict, context) -> dict:
    log.info("StateMergeAndReconcileShots event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        data = event.get("data", {})
        uuid_val = event.get("uuid", "unknown")
        video = data.get("video", {})
        iterators = data.get("iterators", [])
        cfg = data.get("frames_to_shots_cfg", {})

        proxy_bucket = data["proxy_bucket"]
        proxy_prefix = data["proxy_prefix"]

        fps = float(video.get("frame_rate_fps", 25.0))
        actual_frame_count = video.get("actual_frame_count")
        actual_duration_sec = video.get("actual_duration_sec")

        # ── Sort windows, filter empty ────────────────────────────────
        windows = sorted(
            [w for w in iterators if w.get("frame_count", 0) > 0],
            key=lambda w: w["window_id"])

        if not windows:
            raise ValueError("No windows with frames")

        n_windows = len(windows)
        log.info("Merging %d windows", n_windows)

        # ── Load window metadata + .npy (parallel, order-preserving) ──
        t_load = time.monotonic()

        def _load_window(w):
            return (get_json(proxy_bucket, w["metadata_key"]),
                    _read_npy(proxy_bucket, w["embeddings_key"]))

        with ThreadPoolExecutor(max_workers=min(8, n_windows)) as pool:
            loaded = list(pool.map(_load_window, windows))
        window_metas = [m for m, _ in loaded]
        window_npys = [n for _, n in loaded]
        load_ms = round((time.monotonic() - t_load) * 1000, 1)
        log.info("Loaded %d windows in %.0fms", n_windows, load_ms)

        # ── Frame ownership ───────────────────────────────────────────
        owned_ranges = []
        for i, meta in enumerate(window_metas):
            fi = meta.get("frame_index", [])
            if i + 1 < n_windows:
                nxt_fi = window_metas[i + 1].get("frame_index", [])
                nff = nxt_fi[0] if nxt_fi else None
            else:
                nff = None
            owned_ranges.append(_owned_row_range(fi, nff))

        # ── Concatenate arrays ────────────────────────────────────────
        t_emb = time.monotonic()
        emb_slices = []
        all_fnums:     list[int] = []
        all_stimes:    list[float] = []
        all_entropy:   list[float] = []

        for meta, npy, (rs, re) in zip(
                window_metas, window_npys, owned_ranges):
            if re < rs:
                continue
            emb_slices.append(npy[rs:re + 1])
            fi = meta.get("frame_index", [])
            st = meta.get("source_time", [])
            ent = meta.get("entropy", [])
            n_rows = re - rs + 1
            all_fnums.extend(fi[rs:re + 1])
            all_stimes.extend(
                st[rs:re + 1] if st else [0.0] * n_rows)
            all_entropy.extend(
                ent[rs:re + 1] if ent else [0.0] * n_rows)

        embeddings = (np.concatenate(emb_slices).astype(np.float32)
                      if emb_slices
                      else np.zeros((0, 384), dtype=np.float32))
        frame_numbers = np.array(all_fnums, dtype=np.int64)
        source_times = np.array(all_stimes, dtype=np.float64)
        entropy_array = np.array(all_entropy, dtype=np.float32)
        emb_ms = round((time.monotonic() - t_emb) * 1000, 1)
        log.info("Embeddings: %s  %.0fms", embeddings.shape, emb_ms)

        frame_index_list = all_fnums

        # ── Merge shots (coverage tracking) ───────────────────────────
        t_merge = time.monotonic()
        merged_shots = _merge_shots(window_metas)
        merge_ms = round((time.monotonic() - t_merge) * 1000, 1)
        log.info("Merged: %d shots  %.0fms",
                 len(merged_shots), merge_ms)

        # ── Strip internal fields ─────────────────────────────────────
        for s in merged_shots:
            s.pop("_window_id", None)
            s.pop("local_scene_id", None)

        # ── Minimum-duration absorb ───────────────────────────────────
        t_absorb = time.monotonic()
        merged_shots = _absorb_minimum_shots(
            merged_shots, embeddings, frame_numbers, cfg)
        absorb_ms = round((time.monotonic() - t_absorb) * 1000, 1)

        # ── Early entropy + near-black reclassification ───────────────
        # Run before merge passes so near-black shots are black-protected
        # and cannot be absorbed as content by drift/noise merge.
        _compute_shot_entropy(merged_shots, frame_numbers, entropy_array)
        _reclassify_near_black_shots(merged_shots, cfg)

        # ── Fade split ────────────────────────────────────────────────
        t_fade = time.monotonic()
        merged_shots = _split_fades(
            merged_shots, embeddings, frame_numbers, source_times, cfg)
        fade_ms = round((time.monotonic() - t_fade) * 1000, 1)

        # ── Adjacent drift merge ──────────────────────────────────────
        t_adj = time.monotonic()
        merged_shots = _merge_adjacent_drift_shots(
            merged_shots, embeddings, frame_numbers, cfg)
        adj_ms = round((time.monotonic() - t_adj) * 1000, 1)

        # ── Noise merge ───────────────────────────────────────────────
        t_noise = time.monotonic()
        merged_shots = _merge_noise_shots(
            merged_shots, embeddings, frame_numbers, cfg)
        noise_ms = round((time.monotonic() - t_noise) * 1000, 1)

        # ── Second fade split (catches shots created by merge passes) ──
        t_fade2 = time.monotonic()
        merged_shots = _split_fades(
            merged_shots, embeddings, frame_numbers, source_times, cfg)
        fade2_ms = round((time.monotonic() - t_fade2) * 1000, 1)

        # ── Drift split (very long shots with slow scene drift) ───────
        t_drift = time.monotonic()
        merged_shots = _split_drift_shots(
            merged_shots, embeddings, frame_numbers, source_times, cfg)
        drift_ms = round((time.monotonic() - t_drift) * 1000, 1)

        # ── Re-index shot_ids ─────────────────────────────────────────
        for i, s in enumerate(merged_shots):
            s["shot_id"] = i
        n_shots = len(merged_shots)

        # ── Per-shot entropy ──────────────────────────────────────────
        _compute_shot_entropy(
            merged_shots, frame_numbers, entropy_array)

        # ── Per-shot intra-shot dispersion ────────────────────────────
        # Here and not next to the centroid recompute: that one runs only on
        # shots a merge or split TOUCHED, so half the output would be missing
        # the fields. This is the same final pass the entropy aggregation uses.
        _compute_shot_dispersion(
            merged_shots, frame_numbers, embeddings)

        # ── Final near-black reclassification ─────────────────────────
        # The merge/split passes can create new low-entropy "content"
        # shots (split halves inherit the parent type; the sentinel
        # abort only catches pure-black halves).  Re-run on the final
        # entropy values so the output invariant holds.
        _reclassify_near_black_shots(merged_shots, cfg)

        # ── Validation ────────────────────────────────────────────────
        _validate_continuity(merged_shots, frame_index_list, fps, cfg)

        # ── Recurrence ────────────────────────────────────────────────
        t_rec = time.monotonic()

        rec_stats = {}
        recurrence_groups = detect_recurrence(
            shots=merged_shots,
            cfg=cfg,
            actual_duration_sec=actual_duration_sec,
            stats=rec_stats,
        )
        n_rec_groups = len(recurrence_groups)
        n_rec_shots = sum(
            g["recurrence_count"] for g in recurrence_groups)
        rec_ms = round((time.monotonic() - t_rec) * 1000, 1)
        log.info("Recurrence: %d groups  %d shots  %.0fms",
                 n_rec_groups, n_rec_shots, rec_ms)
        # Re-log the lib's per-gate counters from the STATE, because
        # lib.recurrence's own log.info lines do not propagate out of the
        # Lambda — only this module's do. Without this line the only way to
        # learn which gate dropped a group was an offline replay.
        log.info(
            "Recurrence gates: content=%d (excluded %d)  components=%d  "
            "min_members_dropped=%d  occurrence_gate_dropped=%d  "
            "-> %d groups (9.3 annotates, never drops)",
            rec_stats.get("content_shots", 0),
            rec_stats.get("excluded_non_content", 0),
            rec_stats.get("components", 0),
            rec_stats.get("min_members_dropped", 0),
            rec_stats.get("occurrence_gate_dropped", 0),
            rec_stats.get("groups", 0))

        # ── Build frame-to-window index ───────────────────────────────
        t_fidx = time.monotonic()
        frame_to_window = _build_frame_to_window_index(
            window_metas, windows)
        fidx_ms = round((time.monotonic() - t_fidx) * 1000, 1)
        log.info("Frame-to-window index: %.0fms", fidx_ms)

        # ── Frame selection per shot ───────────────────────────────────
        t_sel = time.monotonic()
        for shot in merged_shots:
            shot["selected_frames"] = _select_shot_frames(
                shot, embeddings, frame_numbers, cfg)
        sel_ms = round((time.monotonic() - t_sel) * 1000, 1)
        log.info(
            "Frame selection: %d shots  %.0fms  "
            "avg=%.1f frames/shot",
            len(merged_shots), sel_ms,
            sum(len(s["selected_frames"]) for s in merged_shots)
            / max(len(merged_shots), 1),
        )

        # ── Repack selected frames per shot (streaming + parallel) ────
        t_repack = time.monotonic()
        n_repacked, n_repack_errors = _repack_shot_frames_streaming(
            merged_shots, frame_to_window, proxy_bucket, proxy_prefix,
            max_workers=max(1, min(20, len(merged_shots))),
        )
        repack_ms = round((time.monotonic() - t_repack) * 1000, 1)
        log.info(
            "Repack: %d ok  %d errors  %.0fms",
            n_repacked, n_repack_errors, repack_ms,
        )

        # ── Recurrence hash confirmation (README.md 9.6 + the 9.4 texture
        # floor). Runs AFTER repack so member frames are one small
        # per-shot tar away. Report-not-drop: every group comes back with
        # hash_confirmed true/false/null; nothing is deleted here.
        if recurrence_groups:
            t_conf = time.monotonic()
            ent_by_frame = {int(fn): float(e)
                            for fn, e in zip(frame_numbers, entropy_array)}
            shot_fk = {s["shot_id"]: s.get("frames_key")
                       for s in merged_shots}

            def _load_lumas(sid):
                fk = shot_fk.get(sid)
                if not fk:
                    return []
                out = []
                with _tarfile.open(
                        fileobj=io.BytesIO(get_binary(proxy_bucket, fk)),
                        mode="r:gz") as tar:
                    for m in sorted(tar.getmembers(),
                                    key=lambda m: m.name):
                        f = tar.extractfile(m)
                        if f is None:
                            continue
                        fn = int(m.name.replace("frame_", "")
                                 .replace(".jpg", ""))
                        img = Image.open(io.BytesIO(f.read())).convert("L")
                        out.append((fn, np.asarray(img)))
                return out

            recurrence_groups = confirm_groups_by_hash(
                recurrence_groups, _load_lumas, ent_by_frame)
            conf_ms = round((time.monotonic() - t_conf) * 1000, 1)
            verdicts = [g.get("hash_confirmed") for g in recurrence_groups]
            log.info(
                "Recurrence hash confirmation: %d confirmed  %d unconfirmed  "
                "%d unverifiable  %.0fms",
                sum(v is True for v in verdicts),
                sum(v is False for v in verdicts),
                sum(v is None for v in verdicts), conf_ms)

        # ── Write embeddings.npz ──────────────────────────────────────
        t_write = time.monotonic()
        npz_key = f"{proxy_prefix}/embeddings/embeddings.npz"
        # Embeddings archived as float16: on L2-normalised vectors the
        # cosine-distance error is ~1e-3 — far below every threshold in
        # the pipeline (min 0.05) — and the file halves.  Consumers must
        # cast up with .astype(np.float32) before matmuls.
        _write_npz(
            proxy_bucket, npz_key,
            embeddings=embeddings.astype(np.float16),
            frame_number=frame_numbers,
            source_time=source_times,
            entropy=entropy_array,
        )

        # ── Write shot_centroids.npy ──────────────────────────────────
        # Per-shot centroids as [n_shots, 384] float16, row index ==
        # shot_id.  Precomputed for downstream consumers so they don't
        # have to recompute from embeddings.npz frame slices.
        centroids_key = f"{proxy_prefix}/embeddings/shot_centroids.npy"
        centroid_matrix = np.array(
            [s["centroid_embedding"] for s in merged_shots],
            dtype=np.float16,
        )
        buf = io.BytesIO()
        np.save(buf, centroid_matrix)
        put_binary(proxy_bucket, centroids_key, buf.getvalue(),
                   "application/octet-stream")
        log.info("Written: s3://%s/%s  %s",
                 proxy_bucket, centroids_key, centroid_matrix.shape)

        # ── Write shots.json ──────────────────────────────────────────
        # centroid_embedding (384 floats ≈ 8 KB of JSON text per shot,
        # ~90% of the file) is not serialized — consumers read
        # shot_centroids.npy (row == shot_id) instead.
        shots_key = f"{proxy_prefix}/embeddings/shots.json"
        _derive_timecodes(
            merged_shots,
            video.get("frame_rate"),
            bool(video.get("drop_frame", False)),
            video.get("timecode_first_frame"),
        )
        shots_for_json = []
        for s in merged_shots:
            d = dict(s)
            d.pop("centroid_embedding", None)
            d.pop("representative_embeddings", None)
            shots_for_json.append(d)
        # Programme-tail evidence: null when the programme runs to the end of
        # the file; otherwise the separator black plus tail-to-body match
        # stats. Consumers read match_fraction/match_monotone to tell a
        # textless block (matched AND ordered) from a further programme part
        # (unmatched) — the reading is contract documentation, not a verdict
        # field (see lib/tail_evidence.py for why).
        programme_tail = analyse_programme_tail(merged_shots)
        # Burst-run evidence (9.15): intervals of consecutive sub-second
        # shots, so a consumer can fold each run into one inference instead
        # of paying per member. Empty list = no bursts.
        burst_runs = detect_burst_runs(merged_shots)
        _write_json(proxy_bucket, shots_key, {
            "uuid":                    uuid_val,
            "n_shots":                 n_shots,
            "actual_frame_count":      actual_frame_count,
            "actual_duration_sec":     actual_duration_sec,
            "shot_boundary_threshold": float(cfg.get("shot_boundary_threshold", 0.20)),
            "validation":              {"passed": True},
            "programme_tail":          programme_tail,
            "burst_runs":              burst_runs,
            "shots":                   shots_for_json,
        })

        # ── Write recurrence.json ─────────────────────────────────────
        recurrence_key = f"{proxy_prefix}/embeddings/recurrence.json"
        _write_json(proxy_bucket, recurrence_key, {
            "uuid":                     uuid_val,
            "actual_duration_sec":      actual_duration_sec,
            # Defaults from the schema constants, never literals: these four
            # tell a consumer which knobs PRODUCED the groups below, and a
            # restated fallback goes silently stale the day the default moves
            # (0.05 sat here for a day after the real default became 0.07).
            "max_recurrence_threshold": float(
                cfg.get("max_recurrence_threshold",
                        DEFAULT_MAX_RECURRENCE_THRESHOLD)),
            "min_recurrence_count":     int(
                cfg.get("min_recurrence_count",
                        DEFAULT_MIN_RECURRENCE_COUNT)),
            "min_occurrence_gap_sec":   float(
                cfg.get("min_occurrence_gap_sec",
                        DEFAULT_MIN_OCCURRENCE_GAP_SEC)),
            "max_recurrence_count":     int(
                cfg.get("max_recurrence_count",
                        DEFAULT_MAX_RECURRENCE_COUNT)),
            "total_groups":             n_rec_groups,
            "total_recurring_shots":    n_rec_shots,
            # Per-gate attribution for the two gates that still DROP (9.1 and
            # min-members), so a consumer can tell a missing card's cause
            # apart without an offline replay. Since 9.3 annotates and 9.6
            # flags (hash_confirmed per group), nothing downstream of these
            # counters deletes: gates.groups == total_groups is the
            # invariant, and the old hash_confirmation_dropped counter is
            # just count(hash_confirmed == false) — derivable, so removed.
            "gates":                    rec_stats,
            "recurrence_groups":        recurrence_groups,
        })

        write_ms = round((time.monotonic() - t_write) * 1000, 1)
        elapsed = round(time.monotonic() - t0, 2)

        log.info(
            "DONE: shots=%d  recurrence=%d  "
            "load=%.0fms  merge=%.0fms  "
            "absorb=%.0fms  fade=%.0fms  adj=%.0fms  noise=%.0fms  fade2=%.0fms  drift=%.0fms  rec=%.0fms  "
            "fidx=%.0fms  sel=%.0fms  repack=%.0fms  "
            "write=%.0fms  total=%.1fs",
            n_shots, n_rec_groups,
            load_ms, merge_ms,
            absorb_ms, fade_ms, adj_ms, noise_ms, fade2_ms, drift_ms, rec_ms,
            fidx_ms, sel_ms, repack_ms,
            write_ms, elapsed)

        if n_repack_errors > 0:
            raise ValueError(
                f"Repack error counts: {n_repack_errors}"
            )

        return {
            **event,
            "data": {
                **data,
                "embeddings_key":      npz_key,
                "shots_key":           shots_key,
                "centroids_key":       centroids_key,
                "recurrence_key":      recurrence_key,
                "n_shots":             n_shots,
                "n_recurrence_groups": n_rec_groups,
                "n_recurring_shots":   n_rec_shots,
                "total_frames":        int(embeddings.shape[0]),
                "embed_dim":           int(embeddings.shape[1])
                if embeddings.ndim == 2 else 0,
                "validation_ok":       True,
                "n_shot_frames_ok":    n_repacked,
                "n_shot_frames_errors": n_repack_errors,
            },
        }

    except Exception as exc:
        log.error("[ERR] %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise
