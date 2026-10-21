#!/usr/bin/env python3
"""
test_global_cluster.py  —  v3
──────────────────────────────
Local smoke-test for the GLOBAL CLUSTER state logic (B2).

Mirrors b2_global_cluster_handler.py v3 exactly:
  • Intra-file duplicate turn_idx detection — hard error, fail immediately
  • Deduplication on (start, end) timestamp pair (turn_idx is local/resets)
  • _cluster_with_fallback — 3-stage cascade for degenerate results
  • id_map via time-key lookup covering all registrations (overlap turns
    get entries under both chunk slots)
  • merge_gap uses <= (inclusive)
  • Speaker models with two-pass robust centroid
  • n_resolved counter (not len(id_map)) for the coverage assertion

Usage
─────
    # Single chunk from local output dir:
    python tests/test_global_cluster.py \\
        --bucket        my-bucket \\
        --prefix        jobs/job-abc \\
        --embeddings-dir test_output \\
        --chunk-indices  0 \\
        --method        ahc \\
        --threshold     0.715 \\
        --min-cluster-size 2 \\
        --verbose

    # Multiple chunks from local dir:
    python tests/test_global_cluster.py \\
        --bucket        my-bucket \\
        --prefix        jobs/job-abc \\
        --embeddings-dir test_output \\
        --chunk-indices  0 1 2 \\
        --verbose

    # Explicit S3 keys (skip local upload):
    python tests/test_global_cluster.py \\
        --bucket        my-bucket \\
        --prefix        jobs/job-abc \\
        --embeddings-keys \\
            jobs/job-abc/embeddings/embeddings_0.json \\
            jobs/job-abc/embeddings/embeddings_1.json \\
        --verbose

Arguments
─────────
  --bucket          S3 bucket containing embeddings files
  --prefix          S3 key prefix (e.g. jobs/job-abc)

  Embeddings source (one of):
  --embeddings-dir  Local dir with embeddings_{n}.json
                    (default: test_output/)
  --chunk-indices   Which chunk indices to load from --embeddings-dir
                    (default: 0)
  --embeddings-keys One or more explicit S3 keys — skips local lookup

  Clustering:
  --method            "ahc" or "vbx"  (default: ahc)
  --threshold         AHC cosine-distance cut       (default: 0.715)
  --fa                VBx acoustic weight            (default: 0.3)
  --fb                VBx speaker prior              (default: 17.0)
  --lda-dim           VBx LDA projection dim, 0=skip (default: 0)
  --max-iter          VBx EM iterations              (default: 40)
  --init-threshold    VBx AHC init threshold         (default: 0.5)
  --min-cluster-size  Merge clusters smaller than N  (default: 2)

  Speaker models:
  --outlier-percentile  Trim embeddings beyond this cosine-distance
                        percentile (default: 90.0)
  --min-turns-for-trim  Skip trimming for clusters smaller than N
                        (default: 5)

  --output-dir      Directory to write global_speakers.json
                    (default: test_output/)
  --handler-dir     Project root containing lib/
                    (default: parent of tests/)
  --verbose         Print diarization segment table

What it tests
─────────────
  1. Embeddings loading       — reads embeddings_{n}.json from S3
  2. Intra-file integrity     — HARD ERROR if any file has duplicate turn_idx
  3. Embedding matrix         — shape, dtype, L2 norms ≈ 1.0, no NaN/Inf
  4. Deduplication            — (start,end) dedup, overlap-dup count logged
  5. cluster_with_fallback()  — AHC or VBx + 3-stage fallback cascade
  6. Label validity           — contiguous 0-indexed, count matches turns
  7. Speaker models           — robust centroid, count matches n_spk
  8. id_map construction      — n_resolved == n_total_reg (not len(id_map))
                                overlap turns covered under both chunk slots
  9. Diarization timeline     — sorted, no overlapping segments
  10. JSON output             — writes global_speakers.json locally

Exit codes
──────────
  0  all assertions passed
  1  one or more assertions failed
  2  hard error (duplicate turn_idx, missing file, S3 error)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

TimeKey = Tuple[float, float]

_MIN_TURNS_FOR_FALLBACK = 20
_MAX_SPEAKERS_FALLBACK = 10


# =============================================================================
# PATH SETUP
# =============================================================================

def setup_path(handler_dir: str) -> None:
    for p in [handler_dir, os.path.join(handler_dir, "lib")]:
        if p not in sys.path:
            sys.path.insert(0, p)


# =============================================================================
# ASSERTIONS
# =============================================================================

def check(failures: list, condition: bool, msg: str) -> None:
    if not condition:
        failures.append(msg)
        log.error(f"  FAIL: {msg}")
    else:
        log.info(f"  OK  : {msg}")


# =============================================================================
# EMBEDDINGS RESOLUTION
# =============================================================================

def resolve_embeddings_keys(
    args:   argparse.Namespace,
    bucket: str,
    prefix: str,
) -> List[Tuple[int, str]]:
    """
    Return list of (chunk_idx, s3_key) for all requested chunks.

    Priority:
      1. --embeddings-keys  (explicit S3 keys — no upload)
      2. --embeddings-dir + --chunk-indices  (local JSON → upload)
    """
    import boto3
    s3 = boto3.client("s3")

    if args.embeddings_keys:
        result = []
        for key in args.embeddings_keys:
            basename = os.path.basename(key)
            try:
                idx = int(basename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                idx = len(result)
            result.append((idx, key))
            log.info(f"  Using explicit key: {key}")
        return result

    result = []
    for idx in args.chunk_indices:
        local_path = os.path.join(
            args.embeddings_dir, f"embeddings_{idx}.json"
        )
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"embeddings_{idx}.json not found in '{args.embeddings_dir}'. "
                f"Run test_extract_embeddings.py first, or pass --embeddings-keys."
            )
        s3_key = f"{prefix}/embeddings/embeddings_{idx}.json"
        log.info(f"  Uploading {local_path} → s3://{bucket}/{s3_key}")
        s3.upload_file(local_path, bucket, s3_key)
        result.append((idx, s3_key))

    return result


# =============================================================================
# LOAD & DEDUPLICATE
# =============================================================================

def load_embeddings(
    bucket:   str,
    emb_keys: List[Tuple[int, str]],
) -> Tuple[List[Dict], List[Tuple[str, TimeKey]]]:
    """
    Load per-chunk embeddings from S3, validate intra-file integrity,
    and deduplicate on (start, end) timestamp pair.

    Intra-file duplicate turn_idx — HARD ERROR
    ───────────────────────────────────────────
    turn_idx must be unique within a single embedding file.  A duplicate
    means B1 wrote the same local index twice (e.g. a boundary turn was
    appended to both the main list and the overlap list under the same
    index).  This is a B1 bug.  We raise ValueError immediately with a
    full description so it is fixed at the source, not papered over here.

    Cross-file same turn_idx — expected, handled by (start,end) dedup
    ───────────────────────────────────────────────────────────────────
    turn_idx is LOCAL to each embedding file and resets when a dynamic
    chunk crosses a VAD source-window boundary.  turn_idx=5 in chunk 0
    and turn_idx=5 in chunk 3 are completely different speech segments.
    Using turn_idx as a global dedup key would silently drop all turns in
    later chunks whose local index collides with an earlier chunk's range.

    (start, end) timestamps are assigned by the VAD model and are byte-
    identical for the same segment wherever it appears.  The only
    duplicates that share a time key are the 2-3 overlap-zone turns at
    each chunk boundary — exactly what we want to deduplicate.

    First-occurrence wins — the earlier chunk's embedding is kept because
    that segment is interior to that window (more surrounding audio context).

    Returns
    ───────
    unique_turns
        Deduplicated turns sorted by start time.  Each dict has
        "_chunk_idx" (int) and "_turn_key" (str) injected.

    all_registrations
        (turn_key, time_key) for every occurrence including overlap dups.
        Used to build id_map with entries under both chunk slots for
        boundary turns, without a second S3 round-trip.

    Raises
    ──────
    ValueError
        If any embedding file contains two turns with the same turn_idx.
    """
    import boto3
    s3c = boto3.client("s3")

    seen: set = set()
    unique_t: List[Dict] = []
    all_regs: List[Tuple[str, TimeKey]] = []

    for chunk_idx, key in sorted(emb_keys, key=lambda x: x[0]):
        try:
            obj = s3c.get_object(Bucket=bucket, Key=key)
            doc = json.loads(obj["Body"].read())
        except Exception as exc:
            log.error(f"  chunk {chunk_idx:>2}: load failed {key!r} — {exc}")
            raise

        turns = doc.get("speaker_turns", [])

        # ── Intra-file duplicate turn_idx check ───────────────────────────────
        # Fail immediately — this is a B1 bug, not something to work around.
        _seen_local: Dict[int, Dict] = {}
        _duplicates: List[str] = []

        for turn in turns:
            tid = turn.get("turn_idx")
            if tid is None:
                continue
            if tid in _seen_local:
                prev = _seen_local[tid]
                _duplicates.append(
                    f"  turn_idx={tid} "
                    f"first=({float(prev['start']):.3f}s"
                    f"–{float(prev['end']):.3f}s) "
                    f"duplicate=({float(turn['start']):.3f}s"
                    f"–{float(turn['end']):.3f}s)"
                )
            else:
                _seen_local[tid] = turn

        if _duplicates:
            lines = "\n".join(_duplicates)
            raise ValueError(
                f"\n\nDuplicate turn_idx found in chunk {chunk_idx} "
                f"({key}):\n{lines}\n\n"
                f"This is a B1 (embedding extractor) bug — the same local "
                f"turn_idx was written more than once to the same file.  "
                f"Fix the extractor before re-running global clustering."
            )

        log.info(
            f"  chunk {chunk_idx:>2}: "
            f"{len(_seen_local)} unique turn_idx values — no duplicates"
        )

        # ── Normal loading & cross-file dedup ─────────────────────────────────
        n_new = n_dup = n_skip = 0

        for turn in turns:
            if "embedding" not in turn:
                n_skip += 1
                continue

            start = round(float(turn["start"]), 3)
            end = round(float(turn["end"]),   3)
            time_key: TimeKey = (start, end)
            turn_key: str = f"{chunk_idx}:{turn.get('turn_idx', 0)}"

            # Always register — id_map must cover both slots for overlap turns
            all_regs.append((turn_key, time_key))

            if time_key in seen:
                n_dup += 1
                continue

            seen.add(time_key)
            t = dict(turn)
            t["_chunk_idx"] = chunk_idx
            t["_turn_key"] = turn_key
            unique_t.append(t)
            n_new += 1

        log.info(
            f"  chunk {chunk_idx:>2}:  "
            f"+{n_new:>3} new   "
            f"{n_dup:>2} overlap-dup   "
            f"{n_skip:>2} no-emb   "
            f"s3://{bucket}/{key}"
        )

    unique_t.sort(key=lambda t: float(t["start"]))
    return unique_t, all_regs


# =============================================================================
# FALLBACK CLUSTERING
# =============================================================================

def cluster_with_fallback(
    emb_matrix: np.ndarray,
    cfg:        dict,
) -> Tuple[np.ndarray, str]:
    """
    Cluster with degenerate-result detection and 3-stage fallback.

    The fallback is only triggered when:
      • The initial result has <= 1 speaker  AND
      • N >= _MIN_TURNS_FOR_FALLBACK
        (a genuine 1-speaker recording is valid; only recordings large
        enough to be suspicious trigger the cascade)

    Stage 1 — AHC maxclust search
    ──────────────────────────────
    Compute the linkage dendrogram (centroid linkage, cosine distance)
    once and reuse it.  fcluster with criterion="maxclust" cuts the
    dendrogram to produce exactly k clusters — no threshold guessing.
    Walk k=2,3,...,max_k; apply _merge_small_clusters; accept the first
    k that yields >= 2 valid speakers.

    Also logs the "natural" AHC threshold for k=2,3,4 so the operator
    can tune clustering.threshold in the config and avoid the fallback
    on subsequent runs.

    Stage 2 — VBx
    ──────────────
    Variational-Bayes EM with AHC initialisation.  Designed for long
    recordings; automatically prunes speakers with negligible weight.

    Stage 3 — last resort
    ──────────────────────
    Force a 2-way split at height Z[-1,2]*0.9999 (just below the final
    dendrogram merge).  min_cluster_size=1 so even tiny speakers survive.
    """
    from lib.clustering import cluster, cluster_ahc, cluster_vbx

    method = cfg["method"]
    N = len(emb_matrix)
    min_cs = cfg.get("min_cluster_size", 2)

    # ── Initial attempt ───────────────────────────────────────────────────────
    labels = cluster(emb_matrix, **cfg)
    n_spk = int(labels.max()) + 1 if len(labels) else 0
    log.info(
        f"Initial {method}(threshold={cfg.get('threshold')}) "
        f"→ {n_spk} speaker(s)"
    )

    if n_spk > 1 or N < _MIN_TURNS_FOR_FALLBACK:
        return labels, method

    # ── Degenerate — build dendrogram for diagnostics and fallback ────────────
    log.warning(
        f"Degenerate clustering: {n_spk} speaker(s) for {N} turns. "
        f"AHC threshold={cfg.get('threshold')} is too high for this "
        f"recording. Starting fallback cascade."
    )

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True).clip(1e-10)
    normed = (emb_matrix / norms).astype(np.float64)

    cos_dist = np.clip(1.0 - normed @ normed.T, 0.0, 2.0)
    np.fill_diagonal(cos_dist, 0.0)
    condensed = cos_dist[np.triu_indices(N, k=1)]

    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(condensed, method="centroid")
    heights = Z[:, 2]

    def _ht(idx: int) -> str:
        return f"{heights[idx]:.4f}" if abs(idx) <= len(heights) else "n/a"

    log.info(
        "Dendrogram natural thresholds (centroid AHC, cosine distance): "
        f"2-spk → threshold ∈ ({_ht(-2)}, {_ht(-1)}), "
        f"3-spk → threshold ∈ ({_ht(-3)}, {_ht(-2)}), "
        f"4-spk → threshold ∈ ({_ht(-4)}, {_ht(-3)})"
    )
    log.info(
        f"Config recommendation: set clustering.threshold < {_ht(-1)} "
        f"(currently {cfg.get('threshold')}) to avoid fallback on this "
        f"recording."
    )

    # ── Stage 1: AHC maxclust k = 2 … max_k ──────────────────────────────────
    from lib.clustering import _merge_small_clusters, _reindex

    max_k = min(_MAX_SPEAKERS_FALLBACK, max(N // max(min_cs, 1), 2), N - 1)

    for k in range(2, max_k + 1):
        raw = fcluster(Z, t=k, criterion="maxclust").astype(np.int32) - 1
        merged = _reindex(_merge_small_clusters(raw, normed, min_cs))
        n = int(merged.max()) + 1 if len(merged) else 0
        log.info(f"  [Stage 1] maxclust k={k} → {n} valid speaker(s)")
        if n >= 2:
            log.info(f"  [Stage 1] accepted k={k}")
            return merged, f"ahc(maxclust={k},auto)"

    # ── Stage 2: VBx ──────────────────────────────────────────────────────────
    log.info("[Stage 2] AHC maxclust exhausted — trying VBx...")
    vbx_labels = cluster_vbx(
        emb_matrix,
        fa=cfg.get("fa",             0.3),
        fb=cfg.get("fb",             17.0),
        max_iter=cfg.get("max_iter",       40),
        init_threshold=cfg.get("init_threshold", 0.5),
        min_cluster_size=min_cs,
    )
    n_vbx = int(vbx_labels.max()) + 1 if len(vbx_labels) else 0
    log.info(f"  [Stage 2] VBx → {n_vbx} speaker(s)")
    if n_vbx >= 2:
        return vbx_labels, "vbx(auto)"

    # ── Stage 3: last resort — force 2-way split ──────────────────────────────
    log.warning("[Stage 3] VBx still degenerate — forcing 2-cluster split...")
    forced_thresh = float(heights[-1]) * 0.9999
    forced_labels = cluster_ahc(
        emb_matrix,
        threshold=forced_thresh,
        method="centroid",
        min_cluster_size=1,
    )
    n_forced = int(forced_labels.max()) + 1 if len(forced_labels) else 0
    log.warning(
        f"  [Stage 3] threshold={forced_thresh:.6f} → {n_forced} speaker(s)"
    )
    if n_forced >= 2:
        return forced_labels, f"ahc(forced,threshold={forced_thresh:.4f})"

    log.error(
        "All fallback stages failed — returning 1-speaker result. "
        "Check embedding quality or reduce audio duration per chunk."
    )
    return labels, f"{method}(degenerate)"


# =============================================================================
# ID_MAP
# =============================================================================

def build_id_map(
    unique_turns: List[Dict],
    labels: np.ndarray,
    all_registrations: List[Tuple[str, TimeKey]],
) -> Tuple[Dict[str, str], int]:
    """
    Build id_map covering ALL registrations including overlap slots.

    Step A  Build (start,end) → speaker from unique-turn assignments.

    Step B  Walk all_registrations — overlap turns get entries under both
            chunk N and chunk N+1 keys, both pointing to the same speaker.

    Returns
    ───────
    id_map      turn_key → "SPEAKER_XX"
                Last writer wins on duplicate turn_key.  After the
                intra-file duplicate check in load_embeddings, the only
                remaining case that produces a duplicate turn_key is a
                genuine B1 bug; we log it but do not raise here.

    n_resolved  Count of registrations that successfully mapped to a
                speaker.  Use this for the coverage assertion, not
                len(id_map), because a duplicate turn_key reduces
                len(id_map) by 1 even though both registrations resolved.
    """
    time_to_speaker: Dict[TimeKey, str] = {
        (round(float(t["start"]), 3), round(float(t["end"]), 3)):
            f"SPEAKER_{int(lbl):02d}"
        for t, lbl in zip(unique_turns, labels)
    }

    id_map: Dict[str, str] = {}
    n_resolved: int = 0
    n_unresolved: int = 0

    for turn_key, time_key in all_registrations:
        spk = time_to_speaker.get(time_key)
        if spk is not None:
            id_map[turn_key] = spk
            n_resolved += 1
        else:
            n_unresolved += 1
            log.warning(f"  id_map: cannot resolve {turn_key} @ {time_key}")

    if n_unresolved:
        log.warning(f"  {n_unresolved} registrations unresolved")

    n_dup_keys = n_resolved - len(id_map)
    if n_dup_keys > 0:
        log.warning(
            f"  {n_dup_keys} duplicate turn_key(s) in id_map — "
            f"two turns in the same chunk share the same turn_idx. "
            f"This should have been caught by the intra-file check; "
            f"investigate the embedding extractor (B1)."
        )

    log.info(
        f"  id_map: {len(id_map)} unique keys  "
        f"{n_resolved} resolved  "
        f"{n_unresolved} unresolved  "
        f"{n_dup_keys} dup-key overwrites"
    )

    return id_map, n_resolved


# =============================================================================
# DIARIZATION
# =============================================================================

def build_diarization(
    unique_turns: List[Dict],
    labels:       np.ndarray,
    merge_gap:    float = 0.3,
) -> List[Dict]:
    """
    Sorted, gap-merged diarization timeline built from unique turns only.

    Adjacent same-speaker segments separated by <= merge_gap seconds are
    fused into one to reduce fragmentation in downstream transcript assembly.

    No zone filtering — all unique turns are included.  The test script
    does not currently receive zone annotations; if zone is added to B1
    output in future, add a filter here to match the handler.
    """
    segments = [
        {
            "start":   float(t["start"]),
            "end":     float(t["end"]),
            "speaker": f"SPEAKER_{int(lbl):02d}",
        }
        for t, lbl in zip(unique_turns, labels)
    ]
    if not segments:
        return []

    segments.sort(key=lambda s: s["start"])

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            seg["speaker"] == prev["speaker"]
            and seg["start"] - prev["end"] <= merge_gap
        ):
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


# =============================================================================
# SPEAKER MODELS
# =============================================================================

def compute_speaker_models(
    emb_matrix: np.ndarray,
    labels: np.ndarray,
    unique_turns: List[Dict],
    centroid_trim_percentile: float = 90.0,
    min_turns_for_trim: int = 5,
) -> Dict[str, dict]:
    """
    Two-pass robust centroid per speaker cluster.

    pass 1  μ₀ = L2-norm( mean(X_k) )
            d_i = 1 − x_i · μ₀        cosine distance to draft centroid

    pass 2  inliers = { x_i : d_i <= percentile(d, centroid_trim_percentile) }
            μ* = L2-norm( mean(inliers) )

    Trimming is skipped when M < min_turns_for_trim — removing 10% of a
    3-turn cluster risks discarding its only clean embedding.
    """
    models: Dict[str, dict] = {}

    for k in sorted(np.unique(labels).tolist()):
        mask = labels == k
        X_k = emb_matrix[mask].astype(np.float64)
        turns_k = [t for t, m in zip(unique_turns, mask) if m]
        M = len(X_k)

        total_speech = sum(
            float(t["end"]) - float(t["start"]) for t in turns_k
        )

        # Pass 1 — draft centroid
        raw_mu0 = X_k.mean(axis=0)
        norm0 = np.linalg.norm(raw_mu0)
        mu0 = (raw_mu0 / norm0) if norm0 > 1e-10 else X_k[0].copy()

        if M < min_turns_for_trim:
            centroid = mu0
            n_inliers = M
            n_outliers = 0
        else:
            # Pass 2 — trim periphery, recompute centroid
            cos_dist = 1.0 - (X_k @ mu0)
            dist_cut = np.percentile(cos_dist, centroid_trim_percentile)
            inlier_mask = cos_dist <= dist_cut

            if not inlier_mask.any():
                inlier_mask[np.argmin(cos_dist)] = True

            X_in = X_k[inlier_mask]
            n_inliers = int(inlier_mask.sum())
            n_outliers = M - n_inliers

            raw_mu1 = X_in.mean(axis=0)
            norm1 = np.linalg.norm(raw_mu1)
            centroid = (raw_mu1 / norm1) if norm1 > 1e-10 else mu0

            if n_outliers:
                log.debug(
                    f"  SPEAKER_{k:02d}: trimmed {n_outliers}/{M} outliers  "
                    f"dist_cut={dist_cut:.4f}"
                )

        models[f"SPEAKER_{k:02d}"] = {
            "num_turns":        M,
            "num_inliers":      n_inliers,
            "num_outliers":     n_outliers,
            "total_speech_sec": round(total_speech, 3),
        }

    return models


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(args: argparse.Namespace) -> bool:
    setup_path(args.handler_dir)

    bucket = args.bucket
    prefix = args.prefix.strip("/")

    print("=" * 64)
    print("  GLOBAL CLUSTER — local smoke test  v3")
    print("=" * 64)
    print(f"  Bucket            : {bucket}")
    print(f"  Prefix            : {prefix}")
    print(f"  Method            : {args.method.upper()}")
    if args.method.lower() == "ahc":
        print(f"  Threshold         : {args.threshold}")
    else:
        print(f"  fa / fb           : {args.fa} / {args.fb}")
        print(f"  lda_dim           : {args.lda_dim}")
        print(f"  max_iter          : {args.max_iter}")
        print(f"  init_threshold    : {args.init_threshold}")
    print(f"  min_cluster_size  : {args.min_cluster_size}")
    print(f"  centroid_trim_percentile: {args.centroid_trim_percentile}")
    print(f"  min_turns_for_trim: {args.min_turns_for_trim}")
    print()

    failures = []

    # ── 1. Resolve embeddings keys ────────────────────────────────────────────
    log.info("Step 1/6 — resolving embeddings keys...")
    try:
        emb_keys = resolve_embeddings_keys(args, bucket, prefix)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(2)

    check(failures, len(emb_keys) > 0,
          f"At least 1 embeddings file resolved (got {len(emb_keys)})")

    # ── 2. Load, validate & deduplicate ──────────────────────────────────────
    log.info("Step 2/6 — loading, validating & deduplicating embeddings...")
    t_load = time.monotonic()
    try:
        unique_turns, all_registrations = load_embeddings(bucket, emb_keys)
    except ValueError as exc:
        # Duplicate turn_idx — hard error, stop immediately
        log.error(str(exc))
        print()
        print("  HARD ERROR — duplicate turn_idx detected (see above).")
        print("  Fix the embedding extractor (B1) before re-running.")
        print()
        sys.exit(2)

    load_sec = time.monotonic() - t_load
    n_unique = len(unique_turns)
    n_total_reg = len(all_registrations)
    n_overlap = n_total_reg - n_unique

    log.info(
        f"  {n_unique} unique turns  "
        f"{n_overlap} overlap-dups removed  "
        f"{n_total_reg} total registrations  "
        f"loaded in {load_sec:.2f}s"
    )

    check(failures, n_unique > 0,
          f"At least 1 unique turn loaded (got {n_unique})")

    if not unique_turns:
        log.error("No turns to cluster — aborting")
        sys.exit(2)

    n_chunks = len(set(t["_chunk_idx"] for t in unique_turns))
    if n_chunks == 1:
        log.warning(
            "  Only 1 chunk loaded — overlap-anchor path not exercised. "
            "Pass multiple --chunk-indices to test cross-chunk behaviour."
        )

    # ── 3. Validate embedding matrix ──────────────────────────────────────────
    log.info("Step 3/6 — validating embedding matrix...")
    emb_matrix = np.array(
        [t["embedding"] for t in unique_turns], dtype=np.float32
    )
    N, D = emb_matrix.shape
    log.info(f"  Embedding matrix: {N}×{D}")

    check(failures, emb_matrix.ndim == 2,
          f"Embedding matrix is 2-D (shape {emb_matrix.shape})")
    check(failures, np.isfinite(emb_matrix).all(),
          "No NaN/Inf in embedding matrix")

    norms = np.linalg.norm(emb_matrix, axis=1)
    bad_norm = int(np.sum((norms < 0.99) | (norms > 1.01)))
    if bad_norm > 0:
        log.warning(
            f"  {bad_norm}/{N} embeddings are not unit-norm "
            f"(min={norms.min():.4f}, max={norms.max():.4f}) — "
            f"re-normalising"
        )
        emb_matrix = emb_matrix / norms[:, None].clip(1e-10)

    check(failures, bad_norm == 0,
          f"All embeddings L2-normalised ≈ 1.0 ({bad_norm} out of range)")

    # ── 4. Cluster with fallback ──────────────────────────────────────────────
    log.info(f"Step 4/6 — clustering with fallback ({args.method.upper()})...")
    cfg = {
        "method":           args.method,
        "threshold":        args.threshold,
        "fa":               args.fa,
        "fb":               args.fb,
        "lda_dim":          args.lda_dim,
        "max_iter":         args.max_iter,
        "init_threshold":   args.init_threshold,
        "min_cluster_size": args.min_cluster_size,
    }
    t_clu = time.monotonic()
    labels, method_used = cluster_with_fallback(emb_matrix, cfg)
    clu_sec = time.monotonic() - t_clu

    n_spk = int(labels.max()) + 1 if len(labels) else 0
    counts = dict(sorted(Counter(int(l) for l in labels).items()))
    log.info(
        f"  {n_spk} global speakers  "
        f"method={method_used}  "
        f"in {clu_sec:.2f}s  "
        f"turns_per_speaker={counts}"
    )

    check(failures, len(labels) == n_unique,
          f"Label count matches unique turns ({len(labels)} vs {n_unique})")
    check(failures, labels.dtype == np.int32,
          f"Labels dtype is int32 (got {labels.dtype})")
    check(failures, n_spk > 0,
          f"At least 1 global speaker (got {n_spk})")

    unique_labels = sorted(np.unique(labels).tolist())
    check(failures, unique_labels == list(range(n_spk)),
          f"Labels contiguous 0-indexed (got {unique_labels})")

    # ── 5. Speaker models ─────────────────────────────────────────────────────
    log.info("Step 5/6 — computing speaker models...")
    speaker_models = compute_speaker_models(
        emb_matrix, labels, unique_turns,
        centroid_trim_percentile=args.centroid_trim_percentile,
        min_turns_for_trim=args.min_turns_for_trim,
    )
    check(failures, len(speaker_models) == n_spk,
          f"Speaker model count matches n_spk "
          f"({len(speaker_models)} vs {n_spk})")

    # ── 6. id_map and diarization ─────────────────────────────────────────────
    log.info("Step 6/6 — building id_map and diarization timeline...")
    id_map, n_resolved = build_id_map(unique_turns, labels, all_registrations)

    # Every registration must resolve — this is the real invariant.
    # Use n_resolved, not len(id_map): duplicate turn_keys reduce len(id_map)
    # by 1 per collision even when both registrations resolved correctly.
    check(failures, n_resolved == n_total_reg,
          f"All registrations resolved to a speaker "
          f"({n_resolved}/{n_total_reg})")

    check(failures, len(id_map) > 0,
          f"id_map non-empty (got {len(id_map)} unique entries)")

    bad_spk = [v for v in id_map.values() if not v.startswith("SPEAKER_")]
    check(failures, not bad_spk,
          f"All id_map values are SPEAKER_XX (bad: {bad_spk})")

    # Overlap turns must appear under BOTH chunk slots
    if n_overlap > 0:
        check(failures, len(id_map) > n_unique,
              f"id_map has entries beyond unique-turn count "
              f"({len(id_map)} > {n_unique})")

    # Diarization
    diarization = build_diarization(unique_turns, labels)
    total_speech = sum(s["end"] - s["start"] for s in diarization)

    check(failures, len(diarization) > 0,
          f"Diarization non-empty (got {len(diarization)} segments)")

    starts = [s["start"] for s in diarization]
    check(failures, starts == sorted(starts),
          "Diarization sorted by start time")

    overlaps_found = [
        f"seg[{i}] {diarization[i]['end']:.2f} > "
        f"seg[{i+1}] {diarization[i+1]['start']:.2f}"
        for i in range(len(diarization) - 1)
        if diarization[i]["end"] > diarization[i + 1]["start"]
    ]
    check(failures, not overlaps_found,
          f"No overlapping segments (found: {overlaps_found[:3]})")

    # ── Write output ──────────────────────────────────────────────────────────
    doc = {
        "num_global_speakers": n_spk,
        "clustering_method":   method_used,
        "num_turns_total":     n_unique,
        "num_overlap_removed": n_overlap,
        "total_speech_sec":    round(total_speech, 2),
        "speaker_models":      speaker_models,
        "id_map":              id_map,
        "diarization":         diarization,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "global_speakers.json")
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    log.info(f"  Written: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    speaker_counts = Counter(s["speaker"] for s in diarization)

    print()
    print("=" * 64)
    print(f"  Chunks            : {n_chunks}")
    print(f"  Unique turns      : {n_unique}  "
          f"({n_overlap} overlap-dups removed)")
    print(f"  id_map entries    : {len(id_map)} unique keys  "
          f"({n_resolved} resolved  "
          f"{n_total_reg - n_resolved} unresolved)")
    print(f"  Global speakers   : {n_spk}  (method: {method_used})")
    for spk, cnt in sorted(speaker_counts.items()):
        m = speaker_models.get(spk, {})
        print(
            f"    {spk}: {cnt:>3} diar-segs  "
            f"{m.get('num_turns', 0):>3} turns  "
            f"{m.get('total_speech_sec', 0.0):>8.1f}s  "
            f"outliers={m.get('num_outliers', 0)}"
        )
    print(f"  Total speech      : {total_speech:.1f}s")
    print(f"  Timing            : load={load_sec:.2f}s  "
          f"cluster={clu_sec:.2f}s")
    print(f"  Output JSON       : {out_path}")

    if args.verbose and diarization:
        print()
        print(f"  {'#':>4}  {'start':>8}  {'end':>8}  "
              f"{'dur':>6}  speaker")
        print("  " + "─" * 44)
        for i, seg in enumerate(diarization):
            dur = seg["end"] - seg["start"]
            print(
                f"  {i:>4}  {seg['start']:>8.2f}  {seg['end']:>8.2f}  "
                f"{dur:>6.2f}  {seg['speaker']}"
            )

    print()
    if failures:
        print(f"  FAILED — {len(failures)} assertion(s):")
        for f in failures:
            print(f"    ✗ {f}")
        print()
        return False

    print("  PASSED — all assertions OK  🎉")
    print()
    return True


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local smoke-test for GLOBAL CLUSTER state (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            ────────
            # Single chunk, AHC, default threshold:
            python tests/test_global_cluster.py \\
                --bucket my-bucket --prefix jobs/job-abc \\
                --chunk-indices 0 --method ahc --threshold 0.715 --verbose

            # Multi-chunk (recommended — exercises overlap-anchor path):
            python tests/test_global_cluster.py \\
                --bucket my-bucket --prefix jobs/job-abc \\
                --chunk-indices 0 1 2 --min-cluster-size 2 --verbose

            # Explicit S3 keys (no local upload):
            python tests/test_global_cluster.py \\
                --bucket my-bucket --prefix jobs/job-abc \\
                --embeddings-keys \\
                    jobs/job-abc/embeddings/embeddings_0.json \\
                    jobs/job-abc/embeddings/embeddings_1.json \\
                --method ahc --threshold 0.4 --verbose
        """),
    )

    parser.add_argument("--bucket",  required=True,
                        help="S3 bucket containing embeddings files")
    parser.add_argument("--prefix",  required=True,
                        help="S3 key prefix (e.g. jobs/job-abc)")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--embeddings-dir",  default="test_output",
                     help="Local dir with embeddings_{n}.json "
                          "(default: test_output/)")
    src.add_argument("--embeddings-keys", nargs="+",
                     help="Explicit S3 keys — skips local upload")

    parser.add_argument("--chunk-indices", type=int, nargs="+", default=[0],
                        help="Chunk indices to load from --embeddings-dir "
                             "(default: 0)")

    # Clustering
    parser.add_argument("--method",           default="ahc",
                        choices=["ahc", "vbx"],
                        help="Clustering method (default: ahc)")
    parser.add_argument("--threshold",        type=float, default=0.715,
                        help="AHC cosine-distance threshold (default: 0.715)")
    parser.add_argument("--fa",               type=float, default=0.3,
                        help="VBx acoustic weight (default: 0.3)")
    parser.add_argument("--fb",               type=float, default=17.0,
                        help="VBx Dirichlet prior (default: 17.0)")
    parser.add_argument("--lda-dim",          type=int,   default=0,
                        help="VBx LDA projection dim, 0=skip (default: 0)")
    parser.add_argument("--max-iter",         type=int,   default=40,
                        help="VBx EM iterations (default: 40)")
    parser.add_argument("--init-threshold",   type=float, default=0.5,
                        help="VBx AHC init threshold (default: 0.5)")
    parser.add_argument("--min-cluster-size", type=int,   default=2,
                        help="Merge clusters smaller than N "
                             "(default: 2 — matches state config)")

    # Speaker models
    parser.add_argument("--outlier-percentile", type=float, default=90.0,
                        help="Trim embeddings beyond this cosine-distance "
                             "percentile (default: 90.0)")
    parser.add_argument("--min-turns-for-trim", type=int,   default=5,
                        help="Skip outlier trimming for clusters smaller "
                             "than N (default: 5)")

    parser.add_argument("--output-dir",  default="test_output",
                        help="Directory to write global_speakers.json "
                             "(default: test_output/)")
    parser.add_argument("--handler-dir",
                        default=os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))),
                        help="Project root containing lib/ "
                             "(default: parent of tests/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full diarization segment table")

    args = parser.parse_args()

    ok = run_test(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
