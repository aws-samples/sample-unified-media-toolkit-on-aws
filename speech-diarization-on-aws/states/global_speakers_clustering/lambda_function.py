# global_speakers_clustering.py
"""
StateGlobalSpeakersClustering  (1-core, 1792 MB)

Loads all embeddings_{n}.json files, clusters HIGH quality embeddings, and
produces global_speakers.json with speaker labels for every sentence.

Three-tier assignment (mop-ups are LOCAL with a rejection option —
lib/assign.py; measured rationale in README.md Design record P5)
─────────────────────
  HIGH  (quality="high")  — AHC clustering via _select_optimal_clustering.
                             Outlier turns are removed before clustering and
                             re-assigned afterwards via locality-constrained
                             matching; no confident local match → UNKNOWN.

  LOW   (quality="low")   — locality-constrained centroid match: candidates
                             are speakers with a HIGH anchor within
                             assign_local_window_sec, winner must lead the
                             runner-up by assign_margin, else UNKNOWN.
                             (Global nearest-centroid measured at/below
                             chance on short embeddings — they encode
                             phonetics, not voice.)

  SKIPPED                 — assigned via diarization overlap: find the labeled
                             HIGH sentence that maximally overlaps in time; the
                             nearest-midpoint fallback only applies within
                             assign_local_window_sec, else UNKNOWN.

Output — global_speakers.json:
  {
    "num_global_speakers":    int,
    "clustering_method":      str,   — e.g. "ahc(k_enum  k=3)"
    "sentence_speakers":      { "dyn_chunk_idx:sentence_idx": "SPEAKER_NN", ... },
    "diarization":            [ { "start", "end", "speaker" }, ... ],
    "speaker_models":         { "SPEAKER_NN": { "embedding", "num_turns", ... }, ... },
    "clustering_diagnostics": { k_selected, selection_reason, sweep_log, ... }
  }
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections import Counter
from typing import Any

import numpy as np

from lib.input_schema import (
    DEFAULT_ASSIGN_LOCAL_WINDOW_SEC,
    DEFAULT_USE_CROSSTALK_CONSTRAINTS,
    DEFAULT_ASSIGN_MARGIN,
    DEFAULT_CENTROID_TRIM_PERCENTILE,
    DEFAULT_MIN_TURNS_FOR_TRIM,
    DEFAULT_K_PENALTY,
    DEFAULT_UNRELIABLE_SIL,
    DEFAULT_ELBOW_PLATEAU_PCT,
    POWERSET_F0_SPLIT_RATIO,
    POWERSET_F0_STRADDLE_HIGH,
    POWERSET_F0_STRADDLE_LOW,
    VBX_DEFAULT_FA,
    VBX_DEFAULT_FB,
    VBX_DEFAULT_LDA_DIM,
    VBX_DEFAULT_MAX_ITER,
    VBX_DEFAULT_INIT_THRESHOLD,
)
from lib.assign import assign_speaker_local
from lib.constraints import (evidence_to_cannot_links,
                             overlap_segments, resolve_evidence)
from lib.cluster_pipeline import (
    _select_optimal_clustering,
)
from lib.cluster_score import _compute_cluster_confidence, _compute_speaker_models
from lib.f0 import aggregate_f0
from lib.s3_utils import s3 as _s3, get_json

log = logging.getLogger(__name__)
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


# ─────────────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    log.info("StateGlobalSpeakersClustering event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── unpack event ──
        data = event["data"]
        output_bucket = data["output_bucket"]
        output_prefix = data["output_prefix"]
        iterators = data["iterators"]

        clu_cfg = data["clu_cfg"]

        # ── User configurable ──────────────────────────────────────────────────────────
        num_speakers = clu_cfg.get("num_speakers", None)
        algorithm = str(
            clu_cfg["algorithm"])
        max_k = int(
            clu_cfg["max_k"])
        min_cluster_size = int(
            clu_cfg["min_cluster_size"])
        min_turns_per_spk = int(
            clu_cfg["min_turns_per_speaker"])
        k_selection_method = str(
            clu_cfg["k_selection_method"])
        sil_weight = float(
            clu_cfg["sil_weight"])
        ch_weight = float(
            clu_cfg["ch_weight"])
        outlier_global_percentile = float(
            clu_cfg["outlier_global_percentile"])
        single_spk_thr = float(
            clu_cfg["single_speaker_threshold"])
        apply_snorm = bool(
            clu_cfg["apply_snorm"])

        # ── Hidden configurations ────────────────────────────────────────────────────
        # AHC
        centroid_trim_percentile = float(
            clu_cfg.get("centroid_trim_percentile", DEFAULT_CENTROID_TRIM_PERCENTILE))
        min_turns_for_trim = int(
            clu_cfg.get("min_turns_for_trim", DEFAULT_MIN_TURNS_FOR_TRIM))
        k_penalty = float(
            clu_cfg.get("k_penalty", DEFAULT_K_PENALTY))
        unreliable_sil = float(
            clu_cfg.get("unreliable_silhouette", DEFAULT_UNRELIABLE_SIL))
        elbow_plateau_pct = float(
            clu_cfg.get("elbow_plateau_pct", DEFAULT_ELBOW_PLATEAU_PCT))
        # post-clustering assignment (lib/assign.py)
        assign_window = float(
            clu_cfg.get("assign_local_window_sec",
                        DEFAULT_ASSIGN_LOCAL_WINDOW_SEC))
        assign_margin = float(
            clu_cfg.get("assign_margin", DEFAULT_ASSIGN_MARGIN))
        # VBx
        fa = float(
            clu_cfg.get("fa", VBX_DEFAULT_FA))
        fb = float(
            clu_cfg.get("fb", VBX_DEFAULT_FB))
        lda_dim = int(
            clu_cfg.get("lda_dim", VBX_DEFAULT_LDA_DIM))
        max_iter = int(
            clu_cfg.get("max_iter", VBX_DEFAULT_MAX_ITER))
        init_threshold = float(
            clu_cfg.get("init_threshold", VBX_DEFAULT_INIT_THRESHOLD))

        if num_speakers is not None:
            num_speakers = int(num_speakers)

        log.info(
            "StateGlobalSpeakersClustering START | "
            "chunks=%d  algorithm=%s  k_selection=%s",
            len(iterators), algorithm, k_selection_method,
        )

        # ── 1. Load embeddings (high + low + skipped) ────────────────────────────
        t_io = time.monotonic()
        high_sents, low_sents, all_skipped, embedding_dim, track_segs = (
            _load_all_sentence_embeddings(output_bucket, iterators))
        io_sec = time.monotonic() - t_io

        log.info(
            "Sentences: %d high  %d low  %d skipped  loaded in %.2fs",
            len(high_sents), len(low_sents), len(all_skipped), io_sec,
        )

        if not high_sents:
            from lib.exceptions import NoSpeechDetected
            # a terminal content fact (music-only, ambience) with the
            # evidence attached — the counts are what distinguish a
            # genuinely speechless source from an upstream regression
            # that silently dropped speech (the 48 kHz incident class)
            raise NoSpeechDetected(
                f"no usable speech reached clustering "
                f"— 0 high-quality embeddings "
                f"(low={len(low_sents)}, skipped={len(all_skipped)}, "
                f"iterators={len(iterators)})")

        # ── 2. Clustering on HIGH quality only ────────────────────────────────────
        emb_matrix = np.array(
            [s["embedding"] for s in high_sents], dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True).clip(1e-10)
        emb_matrix = emb_matrix / norms
        N, D = emb_matrix.shape

        total_speech_sec = sum(
            float(s.get("duration", float(s["end"]) - float(s["start"])))
            for s in high_sents)

        cfg = {
            "algorithm":                 algorithm,
            "num_speakers":              num_speakers,
            "max_k":                     max_k,
            "min_cluster_size":          min_cluster_size,
            "min_turns_per_speaker":     min_turns_per_spk,
            "k_selection_method":        k_selection_method,
            "sil_weight":                sil_weight,
            "ch_weight":                 ch_weight,
            "outlier_global_percentile": outlier_global_percentile,
            "apply_snorm":               apply_snorm,
            "single_speaker_threshold":  single_spk_thr,
            "k_penalty":                 k_penalty,
            "unreliable_silhouette":     unreliable_sil,
            "elbow_plateau_pct":         elbow_plateau_pct,
            # VBx
            "fa": fa, "fb": fb, "lda_dim": lda_dim,
            "max_iter": max_iter, "init_threshold": init_threshold,
            # assignment
            "assign_local_window_sec": assign_window,
            "assign_margin": assign_margin,
        }

        # ── P3: cross-talk evidence (two voices proven distinct) ─────────
        # ALWAYS resolved: the identity-labelled overlap_segments output
        # is additive evidence with its precision stated in the contract
        # (81%, zh 56-69% — the span->sentence mapping); trusting it is
        # the CONSUMER's call (user decision 2026-09-18). The flag gates
        # ONLY the cannot-links half: hard links from that mapping
        # regressed DER on every corpus (8af3cbc) and stay OFF.
        use_constraints = bool(clu_cfg.get(
            "use_crosstalk_constraints", DEFAULT_USE_CROSSTALK_CONSTRAINTS))
        evidence = _load_overlap_evidence(output_bucket, output_prefix)
        n_evidence = len(evidence)
        evidence_records = resolve_evidence(evidence, high_sents)
        if use_constraints:
            cannot_links = evidence_to_cannot_links(evidence_records)
            log.info("P3: %d evidence records -> %d cannot-links "
                     "(EXPERIMENTAL constraints ON)",
                     n_evidence, len(cannot_links))
        else:
            cannot_links = []
            log.info("P3: %d evidence records -> overlap_segments "
                     "output only (cannot-links off, the default)",
                     n_evidence)

        t_clu = time.monotonic()
        guides = None
        if track_segs:
            try:
                from lib.powerset_guides import load_guides
                guides = load_guides(output_bucket, output_prefix)
            except Exception as exc:             # noqa: BLE001 — degrade
                log.warning("powerset guide load failed — sentence "
                            "clustering fallback: %s", exc)
        tracks_ok, n_chains = _tracks_viable(track_segs, cfg)
        splice_diag = None
        if guides and tracks_ok:
            # splice-binder candidates (lib/splice_binder.py): the
            # segmentation model judges <=3x3 cross-boundary track
            # pairs; verdicts + row/column veto + embedding agreement
            # emit ADDITIONAL bind candidates through the same
            # voice-gated application as the continuity binder.
            # Degrades to no candidates on any failure — the probe is
            # an upgrade, never a dependency.
            try:
                from lib.splice_binder import probe_bind_candidates
                model_cfg = data.get("model_cfg") or {}
                extra, splice_diag = probe_bind_candidates(
                    output_bucket=output_bucket,
                    output_prefix=output_prefix,
                    audio_bucket=data.get("audio_bucket"),
                    audio_key=data.get("audio_key"),
                    model_bucket=model_cfg.get("model_bucket"),
                    model_prefix=model_cfg.get("model_prefix",
                                               "_model_artefacts"),
                    track_segs=track_segs,
                    existing=dict(
                        guides.get("xchunk_bind_candidates") or []),
                    memory_mb=getattr(context,
                                      "memory_limit_in_mb", None),
                )
                if extra:
                    cands = dict(
                        guides.get("xchunk_bind_candidates") or [])
                    for right, left in extra.items():
                        cands.setdefault(right, left)
                    guides["xchunk_bind_candidates"] = sorted(
                        cands.items())
                log.info("splice binder: %s", splice_diag)
            except Exception as exc:        # noqa: BLE001 — degrade
                log.warning("splice binder failed (continuing "
                            "without): %s", exc)
                splice_diag = {"error": str(exc)[:200]}
            labels, final_sents, method_used, clu_diagnostics = (
                _cluster_tracks(track_segs, guides, high_sents, cfg,
                                assign_margin))
            # constant-free second net: a tracks sweep that reports
            # its own degeneracy is rerun on sentence units (the floor
            # above covers the trivial-candidate regime; this covers
            # pathological geometry at any chain count)
            if (clu_diagnostics.get("selection_reason")
                    == "fallback_no_valid_candidates"):
                log.warning("tracks sweep degenerate "
                            "(fallback_no_valid_candidates, %d chains) "
                            "— sentence-unit rerun", n_chains)
                labels, final_sents, method_used, clu_diagnostics = (
                    _select_optimal_clustering(
                        emb_matrix, high_sents, cfg, total_speech_sec,
                        cannot_links=cannot_links))
                clu_diagnostics = dict(clu_diagnostics)
                clu_diagnostics["tracks_degenerate_fallback"] = n_chains
            # attach AFTER the possible degenerate rerun so the probe
            # record survives either path
            if splice_diag is not None:
                clu_diagnostics = dict(clu_diagnostics)
                clu_diagnostics["splice_probe"] = splice_diag
        else:
            if guides and n_chains:
                log.warning(
                    "small-N guard: %d track chains cannot support the "
                    "K sweep — sentence-unit fallback", n_chains)
            labels, final_sents, method_used, clu_diagnostics = (
                _select_optimal_clustering(emb_matrix, high_sents, cfg,
                                           total_speech_sec,
                                           cannot_links=cannot_links))
            if guides and n_chains:
                clu_diagnostics = dict(clu_diagnostics)
                clu_diagnostics["smalln_track_fallback"] = n_chains
        clu_sec = time.monotonic() - t_clu

        final_emb = np.array(
            [s["embedding"] for s in final_sents], dtype=np.float32)
        norms_f = np.linalg.norm(
            final_emb, axis=1, keepdims=True).clip(1e-10)
        final_emb = final_emb / norms_f

        # Rejected outlier turns carry label -1 (lib/assign.py): rendered
        # UNKNOWN downstream, excluded from models and diarization
        kept = labels >= 0
        kept_emb = final_emb[kept]
        kept_labels = labels[kept]
        kept_sents = [s for s, k in zip(final_sents, kept) if k]

        n_spk = int(kept_labels.max()) + 1 if len(kept_labels) else 0
        counts = dict(sorted(Counter(int(l) for l in labels).items()))
        log.info("Clustering → %d speakers  method=%s  in %.2fs | %s",
                 n_spk, method_used, clu_sec, counts)

        # ── 3. Speaker models + confidence ────────────────────────────────────────
        speaker_models = _compute_speaker_models(
            kept_emb, kept_labels, kept_sents,
            centroid_trim_percentile=centroid_trim_percentile,
            min_turns_for_trim=min_turns_for_trim)
        confidence_map = _compute_cluster_confidence(
            kept_emb.astype(np.float64), kept_labels)
        for spk, conf in confidence_map.items():
            if spk in speaker_models:
                speaker_models[spk]["confidence"] = conf

        # per-cluster F0 rollup (lib/f0) — additive fields on the
        # speaker model, consumed downstream as a voice↔face
        # compatibility signal; never an input to clustering. Sentences
        # from pre-F0 embedding files simply lack the fields and the
        # rollup abstains (three-state doctrine).
        f0_by_label: dict[int, list] = {}
        for s, l in zip(kept_sents, kept_labels):
            f0_by_label.setdefault(int(l), []).append(
                (s.get("f0_hz"), s.get("f0_voiced_ratio") or 0.0))
        for k, stats_in in f0_by_label.items():
            spk = f"SPEAKER_{k:02d}"
            stats = aggregate_f0(stats_in)
            if stats and spk in speaker_models:
                speaker_models[spk].update(stats)

        # ── 4. Build centroids for assignment ─────────────────────────────────────
        centroids: dict[str, np.ndarray] = {}
        for spk_name, model in speaker_models.items():
            c = np.array(model["embedding"], dtype=np.float64)
            norm = np.linalg.norm(c)
            centroids[spk_name] = c / norm if norm > 1e-10 else c

        # ── 5. Assign HIGH quality sentences ──────────────────────────────────────
        sentence_speakers: dict[str, str] = {}
        for sent, lbl in zip(final_sents, labels):
            key = f"{sent['dyn_chunk_idx']}:{sent['sentence_idx']}"
            sentence_speakers[key] = (f"SPEAKER_{int(lbl):02d}"
                                      if lbl >= 0 else "UNKNOWN")

        # ── 5b. Identity-resolved two-speaker overlap segments (P3) ─────────────
        # the transcript emits ONE speaker per instant; these segments
        # carry the proven SECOND voice of each cross-talk span
        _label_of = [sentence_speakers.get(
            f"{s['dyn_chunk_idx']}:{s['sentence_idx']}")
            for s in high_sents]
        overlap_segs = overlap_segments(evidence_records, _label_of)
        log.info("  overlap segments: %d identity-resolved", len(overlap_segs))

        # ── 6. Assign LOW quality sentences (local match or UNKNOWN) ─────────────
        # Short embeddings are phonetics-dominated: measured, GLOBAL
        # nearest-centroid scores at/below chance while candidates
        # restricted to speakers anchored nearby score 7x chance
        # (README.md Design record P5).
        anchor_times = np.array(
            [float(s["start"]) for s in kept_sents], dtype=np.float64)
        anchor_names = [f"SPEAKER_{int(l):02d}" for l in kept_labels]
        n_low_assigned = n_low_unknown = 0
        for sent in low_sents:
            key = f"{sent['dyn_chunk_idx']}:{sent['sentence_idx']}"
            emb = np.array(sent["embedding"], dtype=np.float64)
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            spk = assign_speaker_local(
                emb, float(sent["start"]), anchor_times, anchor_names,
                centroids, assign_window, assign_margin)
            sentence_speakers[key] = spk or "UNKNOWN"
            if spk:
                n_low_assigned += 1
            else:
                n_low_unknown += 1

        log.info("  Low-quality assignment: %d/%d local-matched, "
                 "%d UNKNOWN", n_low_assigned, len(low_sents),
                 n_low_unknown)

        # ── 7. Diarization for skipped sentences ──────────────────────────────────
        diarization = _build_diarization(kept_sents, kept_labels)
        total_speech = sum(s["end"] - s["start"] for s in diarization)

        n_skip_assigned = 0
        for skip in all_skipped:
            key = f"{skip['dyn_chunk_idx']}:{skip['sentence_idx']}"
            spk = _find_speaker_by_overlap(
                float(skip["start"]), float(skip["end"]), diarization,
                max_gap_sec=assign_window)
            sentence_speakers[key] = spk or "UNKNOWN"
            if spk:
                n_skip_assigned += 1

        log.info("  Skipped assignment: %d/%d via diarization, %d UNKNOWN",
                 n_skip_assigned, len(all_skipped),
                 len(all_skipped) - n_skip_assigned)

        # ── 8. Output ─────────────────────────────────────────────────────────────
        out_key = f"{output_prefix}/embeddings/global_speakers.json"

        elapsed = round(time.monotonic() - t0, 2)

        payload = {
            "num_global_speakers":    n_spk,
            "clustering_method":      method_used,
            "num_sentences_total":    len(high_sents) + len(low_sents) + len(all_skipped),
            "num_sentences_high":     len(high_sents),
            "num_sentences_low":      len(low_sents),
            "num_sentences_skipped":  len(all_skipped),
            "total_speech_sec":       round(total_speech, 2),
            "elapsed_sec_clustering": elapsed,
            "sentence_speakers":      sentence_speakers,
            "diarization":            diarization,
            "overlap_segments":       overlap_segs,
            "speaker_models":         speaker_models,
            "clustering_diagnostics": {
                "k_selected":              clu_diagnostics.get("k_selected"),
                "selection_reason":        clu_diagnostics.get("selection_reason"),
                "single_speaker_detected": clu_diagnostics.get("single_speaker_detected"),
                "n_outliers_removed":      clu_diagnostics.get("n_outliers_removed"),
                "n_cannot_links":          clu_diagnostics.get("n_cannot_links", 0),
                "n_overlap_evidence":      n_evidence,
                "cannot_link_violations":  clu_diagnostics.get(
                    "cannot_link_violations", 0),
                "dendrogram_final_height": clu_diagnostics.get("dendrogram_final_height"),
                "k_sweep_log":             clu_diagnostics.get("sweep_log", []),
                # powerset tracks mode (absent = sentence path)
                "unit":                    clu_diagnostics.get("unit"),
                "n_tracks":                clu_diagnostics.get("n_tracks"),
                "n_f0_cannot_links":       clu_diagnostics.get("n_f0_cannot_links"),
                "bimodal_tracks":          clu_diagnostics.get("bimodal_tracks"),
                "n_post_splits":           clu_diagnostics.get("n_post_splits"),
                "n_reconciled":            clu_diagnostics.get("n_reconciled"),
                # the WHY of every refused reconcile merge — the live
                # referee needs it (2026-09-30: three blocked merges
                # were undiagnosable from the artifact alone)
                "reconcile_blocked":       clu_diagnostics.get("reconcile_blocked"),
                "post_pass_cannot_link_violations": clu_diagnostics.get(
                    "post_pass_cannot_link_violations"),
                "k_below_floor":           clu_diagnostics.get("k_below_floor"),
                "xchunk_bind_candidates":  clu_diagnostics.get("xchunk_bind_candidates"),
                "splice_probe":            clu_diagnostics.get("splice_probe"),
                "n_dust_absorbed":         clu_diagnostics.get("n_dust_absorbed"),
                "smalln_track_fallback":   clu_diagnostics.get("smalln_track_fallback"),
                "xchunk_binds":            clu_diagnostics.get("xchunk_binds"),
                "local_speaker_floor":     clu_diagnostics.get("local_speaker_floor"),
                "floor_user_conflict":     clu_diagnostics.get("floor_user_conflict"),
                "tracks_degenerate_fallback": clu_diagnostics.get("tracks_degenerate_fallback"),
                "clique_floor":            clu_diagnostics.get("clique_floor"),
                "trackless_high_sentences": clu_diagnostics.get("trackless_high_sentences"),
            },
            "assignment": {
                "low_assigned":      n_low_assigned,
                "low_unknown":       n_low_unknown,
                "skipped_assigned":  n_skip_assigned,
                "skipped_unknown":   len(all_skipped) - n_skip_assigned,
                "outliers_rejected": clu_diagnostics.get(
                    "n_outliers_rejected", 0),
                "local_window_sec":  assign_window,
                "margin":            assign_margin,
            },
        }

        _s3.put_object(
            Bucket=output_bucket, Key=out_key,
            Body=json.dumps(payload, separators=(",", ":")).encode(),
            ContentType="application/json")
        log.info("Written → s3://%s/%s", output_bucket, out_key)

        global_speakers = {
            "num_global_speakers":     n_spk,
            "clustering_method":       method_used,
            "num_sentences_total":     payload["num_sentences_total"],
            "num_sentences_high":      len(high_sents),
            "num_sentences_low":       len(low_sents),
            "k_selected":              clu_diagnostics.get("k_selected"),
            "single_speaker_detected": clu_diagnostics.get("single_speaker_detected"),
            "n_outliers_removed":      clu_diagnostics.get("n_outliers_removed"),
            "global_speakers_key":     out_key,
            "elapsed_sec_clustering":  elapsed,
            "elapsed_sec_embedding":   max(float(it.get("elapsed_sec", 0.0)) for it in iterators),
        }

        output = {
            **event,
            "data": {
                **data,
                "global_speakers": global_speakers,
            }
        }
        log.info("StateGlobalSpeakersClustering output =\n%s",
                 json.dumps(output, indent=2)[:2000])
        return output

    except Exception as exc:
        log.error(
            "StateGlobalSpeakersClustering failed — %s: %s",
            type(exc).__name__, exc
        )
        traceback.print_exc()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def _cross_register(fa: float | None, fb: float | None) -> bool:
    """Do two F0 medians (Hz) prove DIFFERENT vocal registers?

    One definition for every F0 gate in this state — track bimodality,
    the nearest-neighbor cannot-link, and the reconcile veto. Both
    conditions are required: a ratio wide enough to leave one voice's
    own range AND modes that straddle the male/female divide. The
    ratio alone reads excited same-register prosody as two people
    (round-3 live failure, 2026-09-25); the straddle band alone reads
    an emphatic passage as a register change, because the band is only
    170-195 Hz wide and a single voice's medians land either side of it
    (2026-09-29: a lone-narrator documentary's two fragments sat at
    174/235 Hz — ratio 1.35, no register change — and the reconcile
    pass refused to reunify them).
    """
    if not fa or not fb:
        return False
    lo, hi = min(fa, fb), max(fa, fb)
    return (hi / max(lo, 1e-9) >= POWERSET_F0_SPLIT_RATIO
            and lo <= POWERSET_F0_STRADDLE_LOW
            and hi >= POWERSET_F0_STRADDLE_HIGH)


def _best_k_at_or_above(sweep_log, floor):
    """The sweep's own best-scoring candidate K at or above the floor
    (combined score; vetoed candidates carry combined=None and are
    skipped, falling back to silhouette, then to the floor itself).
    The floor is proof of a MINIMUM speaker count — never an answer."""
    at_or_above = [c for c in sweep_log
                   if int(c.get("k_act") or 0) >= floor]
    scored = [c for c in at_or_above if c.get("combined") is not None]
    if scored:
        return int(max(scored, key=lambda c: c["combined"])["k_act"])
    if at_or_above:
        return int(max(at_or_above,
                       key=lambda c: c.get("sil") or 0.0)["k_act"])
    return int(floor)


def _tracks_viable(track_segs, cfg):
    """Small-N guard: with tracks as the clustering unit the K sweep's
    candidate range is 2..n_chains // min_turns_per_speaker; the guard
    requires that range to hold at least POWERSET_MIN_SWEEP_CANDIDATES
    values, so the floor scales with min_turns_per_speaker instead of
    being an absolute count (measured collapse regime: 6-8 chains on a
    sparse-dialogue short, forced single speaker on BOTH mix and stem
    audio, while sentence units give K=2). Returns (viable, n_chains).
    """
    from lib.input_schema import (DEFAULT_MIN_TURNS_PER_SPEAKER,
                                  POWERSET_MIN_SWEEP_CANDIDATES)
    n = len({t["chain"] for t in track_segs}) if track_segs else 0
    min_turns = int(cfg.get("min_turns_per_speaker",
                            DEFAULT_MIN_TURNS_PER_SPEAKER) or
                    DEFAULT_MIN_TURNS_PER_SPEAKER)
    floor = (POWERSET_MIN_SWEEP_CANDIDATES + 1) * max(1, min_turns)
    return n >= floor, n


def _cluster_tracks(track_segs, guides, high_sents, cfg,
                    assign_margin):
    """Powerset tracks as the clustering unit (2026-09-25 rehearsal:
    reality_a mega-cluster ARI -0.03 -> +0.40 on the production clusterer
    with raw embeddings — the unit change alone). Pools segment
    embeddings per chain (duration-weighted), clusters tracks with
    co-activity cannot-links and the max-clique K floor, then maps the
    labels back onto HIGH sentences by slice-midpoint containment.
    Trackless sentences take the nearest track-cluster centroid within
    the assignment margin, else UNKNOWN (-1). Returns the same
    (labels, final_sents, method, diagnostics) shape as
    _select_optimal_clustering so everything downstream is unchanged.
    """
    from lib.input_schema import (POWERSET_F0_MIN_SENTENCES,
                                  POWERSET_F0_MINORITY,
                                  POWERSET_SPLIT_MIN_CHAINS,
                                  POWERSET_SPLIT_SIL)
    # voice-gated cross-chunk binds: structural candidates from the
    # guides (uncontested continuity at a chunk mark) apply only when
    # the two chains' pooled embeddings agree — continuity alone also
    # matches a handover at the mark (measured weld on rapid-exchange
    # content). Applied before pooling so a bound chain embeds as one.
    from lib.input_schema import POWERSET_XCHUNK_BIND_SIM
    cands = dict(guides.get("xchunk_bind_candidates") or [])
    n_binds_applied = 0
    if cands:
        tmp: dict = {}
        for seg in track_segs:
            tmp.setdefault(seg["chain"], []).append(seg)

        def _chain_vec(ch):
            segs2 = tmp.get(ch)
            if not segs2:
                return None
            ws = np.array([sg["duration"] for sg in segs2])
            vs = np.array([sg["embedding"] for sg in segs2])
            v = np.average(vs, axis=0, weights=np.maximum(ws, 1e-6))
            n = np.linalg.norm(v)
            return v / n if n > 1e-10 else None

        rename: dict = {}
        for right, left in cands.items():
            left = rename.get(left, left)
            va, vb = _chain_vec(left), _chain_vec(right)
            if va is None or vb is None:
                continue
            if float(va @ vb) >= POWERSET_XCHUNK_BIND_SIM:
                rename[right] = left
        if rename:
            track_segs = [dict(t, chain=rename.get(t["chain"], t["chain"]))
                          for t in track_segs]
            guides = dict(guides)
            guides["co_active_pairs"] = [
                [rename.get(a, a), rename.get(b, b)]
                for a, b in (guides.get("co_active_pairs") or [])]
            n_binds_applied = len(rename)

    by_chain: dict = {}
    for seg in track_segs:
        by_chain.setdefault(seg["chain"], []).append(seg)

    # per-track F0 from member sentences (octave-hygienic, voiced
    # floor): QUARANTINE bimodal tracks — a track claiming two
    # registers is a stitching weld, and its pooled embedding is the
    # exact bridge that merges clusters (round-2 forensics)
    def _track_f0(segs):
        vals = []
        for sent in high_sents:
            f0 = sent.get("f0_hz")
            if not f0 or (sent.get("f0_voiced_ratio") or 0) < 0.35:
                continue
            mid = (float(sent.get("segment_start", sent["start"]))
                   + float(sent.get("segment_end", sent["end"]))) / 2
            if any(a - 0.2 <= mid <= b + 0.2 for a, b in
                   ((sg["start"], sg["end"]) for sg in segs)):
                vals.append(f0)
        if len(vals) < POWERSET_F0_MIN_SENTENCES:
            return None, False
        med = sorted(vals)[len(vals) // 2]
        vals = [v for v in vals if abs(v / 2 - med) > 0.25 * med]
        if len(vals) < POWERSET_F0_MIN_SENTENCES:
            return med, False
        v = sorted(vals)
        c1, c2 = v[0], v[-1]
        a = b = None
        for _ in range(30):
            a = [x for x in v if abs(x - c1) <= abs(x - c2)]
            b = [x for x in v if abs(x - c1) > abs(x - c2)]
            if not a or not b:
                break
            c1, c2 = sum(a) / len(a), sum(b) / len(b)
        bimodal = bool(
            a and b
            and min(len(a), len(b)) / len(v) >= POWERSET_F0_MINORITY
            # the two modes must be different REGISTERS, not one voice's
            # emphatic range (see _cross_register)
            and _cross_register(c1, c2))
        return sorted(vals)[len(vals) // 2], bimodal

    # NOTE (2026-09-26): quarantine of bimodal tracks was measured and
    # REMOVED — dropping units perturbs the knife-edge K sweep more
    # than the welds it removes (reality_a offline: retention 1.000
    # without quarantine, 0.570 with). A mixed track's correct
    # treatment is SPLITTING at its register boundary (future work);
    # until then the impurity scoreboard still detects them, and the
    # nearest-neighbor link below caps their cluster-level damage.
    f0_of_chain: dict = {}
    n_bimodal_tracks = 0
    for ch in sorted(by_chain):
        med, bimodal = _track_f0(by_chain[ch])
        f0_of_chain[ch] = None if bimodal else med
        if bimodal:
            n_bimodal_tracks += 1
    chains = sorted(by_chain)
    mats, turns = [], []
    for ch in chains:
        segs = by_chain[ch]
        ws = np.array([s["duration"] for s in segs], dtype=np.float64)
        vs = np.array([s["embedding"] for s in segs], dtype=np.float64)
        v = np.average(vs, axis=0, weights=ws)
        n = np.linalg.norm(v)
        mats.append((v / n if n > 1e-10 else v).astype(np.float32))
        turns.append({"start": min(s["start"] for s in segs),
                      "end": max(s["end"] for s in segs),
                      "duration": float(ws.sum())})
    track_matrix = np.stack(mats)
    idx_of = {ch: i for i, ch in enumerate(chains)}
    cannot = [(idx_of[a], idx_of[b])
              for a, b in (guides.get("co_active_pairs") or [])
              if a in idx_of and b in idx_of]
    # per-link durable co-activity seconds (guides co_active_secs,
    # aligned with co_active_pairs; 0.0 = artifact predates the field
    # = not measured). Consumed by the reconcile evidence gates —
    # NEVER measure co-presence from track segments, which are
    # disjoint by construction (the frame grid votes one chain per
    # frame; measured 2026-09-29: 434/434 corpus links intersected
    # 0.00 s, silently disabling every duration-gated consumer).
    link_sec: dict = {}
    _secs = guides.get("co_active_secs") or []
    for _k, (_a, _b) in enumerate(guides.get("co_active_pairs") or []):
        if _a in idx_of and _b in idx_of:
            _key = (min(idx_of[_a], idx_of[_b]),
                    max(idx_of[_a], idx_of[_b]))
            _v = float(_secs[_k]) if _k < len(_secs) else 0.0
            link_sec[_key] = max(link_sec.get(_key, 0.0), _v)
    # cross-register CANNOT-LINKS, NEAREST-NEIGHBOR ONLY (round-4
    # lesson, offline-selected 2026-09-26: linking EVERY cross-register
    # pair floods mixed-gender content with O(n²/4) poisoned distances
    # — 2,606 links on one asset — deforming the dendrogram and
    # re-collapsing the knife-edge K selection. A wrong merge is only
    # imminent between embedding NEIGHBORS, so each track vetoes just
    # its nearest other track, and only when that pair provably
    # straddles the register divide. Offline replay: reality_a retention
    # ARI 1.000 with 0 links, the mitj M+F merge split with exactly 1)
    n_f0_links = 0
    if len(track_matrix) >= 2:
        dmat = 1.0 - track_matrix @ track_matrix.T
        np.fill_diagonal(dmat, np.inf)
        for x in range(len(chains)):
            fx = f0_of_chain.get(chains[x])
            if not fx:
                continue
            y = int(np.argmin(dmat[x]))
            fy = f0_of_chain.get(chains[y])
            if not fy:
                continue
            if (_cross_register(fx, fy)
                    and (min(x, y), max(x, y)) not in cannot):
                cannot.append((min(x, y), max(x, y)))
                n_f0_links += 1
    # K floor = the strongest durable proof: co-active clique OR the
    # sequential local-speaker count (2026-09-29 — a proven-3 asset
    # shipped K=2 because its co-active shards died at the track
    # filter while the per-window count evidence was discarded).
    # Precedence: an explicit user num_speakers outranks the floor —
    # the conflict is REPORTED, never enforced against the user.
    floor = max(int(guides.get("clique_floor") or 0),
                int(guides.get("local_speaker_floor") or 0))

    tcfg = dict(cfg)
    tcfg["min_cluster_size"] = 2
    total = sum(t["duration"] for t in turns)
    t_labels, t_turns, method, diag = _select_optimal_clustering(
        track_matrix, turns, tcfg, total, cannot_links=cannot)
    k = int(t_labels.max()) + 1 if len(t_labels) else 0
    user_k = cfg.get("num_speakers")
    floor_user_conflict = bool(user_k is not None and floor > int(user_k))
    if floor > k and user_k is None:
        tcfg2 = dict(tcfg)
        # the floor is a MINIMUM, not a target: pick the sweep's own
        # best-scoring candidate at or above it (2026-09-30 DER gate:
        # forcing K=floor exactly turned a k=2 trap on a 6-speaker
        # meeting into K=3 — the floor rescued the collapse but then
        # capped the answer)
        tcfg2["num_speakers"] = _best_k_at_or_above(
            diag.get("sweep_log") or [], floor)
        t_labels, t_turns, method, diag = _select_optimal_clustering(
            track_matrix, turns, tcfg2, total, cannot_links=cannot)
        method += "+k_floor"

    # chain -> label (t_turns order == turns order == chains order:
    # the track path skips outlier removal only when N small; map via
    # the returned turn list by (start,end) identity to stay safe)
    key_of = {(round(t["start"], 3), round(t["end"], 3)): i
              for i, t in enumerate(t_turns)}
    label_of_chain = {}
    for ch, t in zip(chains, turns):
        i = key_of.get((round(t["start"], 3), round(t["end"], 3)))
        if i is not None:
            label_of_chain[ch] = int(t_labels[i])

    # centroids per label for trackless assignment
    cents: dict = {}
    for ch, lb in label_of_chain.items():
        if lb >= 0:
            cents.setdefault(lb, []).append(track_matrix[idx_of[ch]])
    centroids = {lb: (lambda m: m / max(np.linalg.norm(m), 1e-10))(
        np.mean(vs, axis=0)) for lb, vs in cents.items()}

    # sentence labels by slice-midpoint containment
    seg_index = []
    for ch, segs in by_chain.items():
        lb = label_of_chain.get(ch, -1)
        for sg in segs:
            seg_index.append((sg["start"] - 0.2, sg["end"] + 0.2, lb, ch))
    labels = np.full(len(high_sents), -1, dtype=np.int64)
    voiced_chains: set = set()
    trackless = 0
    for i, sent in enumerate(high_sents):
        mid = (float(sent.get("segment_start", sent["start"]))
               + float(sent.get("segment_end", sent["end"]))) / 2
        hit = next(((lb, ch) for a, b, lb, ch in seg_index
                    if a <= mid <= b), None)
        if hit is not None and hit[0] >= 0:
            labels[i] = hit[0]
            voiced_chains.add(hit[1])
            continue
        trackless += 1
        v = np.asarray(sent["embedding"], dtype=np.float64)
        n = np.linalg.norm(v)
        if n > 1e-10 and centroids:
            v = v / n
            sims = sorted(((float(v @ c), lb)
                           for lb, c in centroids.items()), reverse=True)
            if sims and (len(sims) == 1
                         or sims[0][0] - sims[1][0] >= assign_margin):
                labels[i] = sims[0][1]
    # POST-CLUSTERING SPLIT PASS (2026-09-28): a cluster whose member
    # chains 2-split with cosine silhouette >= POWERSET_SPLIT_SIL
    # divides. Runs AFTER the sweep and the sentence mapping so the
    # knife-edge K selection is untouched, and errs prefer-split
    # (over-split is recoverable downstream; over-merge is not).
    # Motivating defect: two clean same-register anchor chain groups
    # from different show blocks — zero co-activity, minutes apart, no
    # F0 straddle, so no existing guard applies — merged by the sweep
    # while plainly embedding-separable (cross vs within cosine
    # AUC 0.94). Trackless (centroid-assigned) sentences keep their
    # cluster: only sentences owned by a moved chain move with it.
    # Only SENTENCE-BEARING chains participate: a chain owning no
    # HIGH sentence is a short interjection track whose noisy pooled
    # embedding can manufacture a "separable" subgroup, and moving one
    # relabels no sentences — pure K inflation. This is also the
    # population the threshold was swept on offline.
    n_post_splits = 0
    members_of: dict = {}
    for ch, lb in label_of_chain.items():
        if lb >= 0 and ch in voiced_chains:
            members_of.setdefault(lb, []).append(ch)
    nxt = max(label_of_chain.values(), default=-1) + 1
    for lb in sorted(members_of):
        members = sorted(members_of[lb])
        if len(members) < POWERSET_SPLIT_MIN_CHAINS:
            continue
        X = track_matrix[[idx_of[ch] for ch in members]].astype(
            np.float64)
        # 2-means on the unit sphere, seeded from the extremes of the
        # projection onto the cluster mean (deterministic — no RNG in
        # a Lambda that must reproduce offline)
        proj = X @ X.mean(axis=0)
        c0, c1 = X[int(np.argmin(proj))], X[int(np.argmax(proj))]
        g = np.zeros(len(members), dtype=np.int64)
        for _ in range(30):
            g = (X @ c1 > X @ c0).astype(np.int64)
            if g.min() == g.max():
                break
            m0 = X[g == 0].mean(axis=0)
            m1 = X[g == 1].mean(axis=0)
            c0 = m0 / max(np.linalg.norm(m0), 1e-10)
            c1 = m1 / max(np.linalg.norm(m1), 1e-10)
        if g.min() == g.max():
            continue
        # standard (Rousseeuw) silhouette: a singleton side scores 0,
        # NOT 1 — lib.cluster_score's a=0 shortcut inflates one-chain
        # splits by up to +0.14, which is what separates the swept
        # threshold from firing on borderline clusters (measured: the
        # inflated score doubled the split count on the 3-asset replay)
        d = 1.0 - X @ X.T
        s = np.zeros(len(g), dtype=np.float64)
        for side in (0, 1):
            own = np.where(g == side)[0]
            oth = np.where(g != side)[0]
            if len(own) > 1:
                a = d[np.ix_(own, own)].sum(axis=1) / (len(own) - 1)
                b = d[np.ix_(own, oth)].mean(axis=1)
                s[own] = (b - a) / np.maximum(np.maximum(a, b), 1e-10)
        sil = float(s.mean())
        if sil < POWERSET_SPLIT_SIL:
            continue
        n_post_splits += 1
        moved = {members[i] for i in range(len(members)) if g[i] == 1}
        spans = [(sg["start"] - 0.2, sg["end"] + 0.2)
                 for ch in moved for sg in by_chain[ch]]
        for i, sent in enumerate(high_sents):
            if labels[i] != lb:
                continue
            mid = (float(sent.get("segment_start", sent["start"]))
                   + float(sent.get("segment_end", sent["end"]))) / 2
            if any(a <= mid <= b for a, b in spans):
                labels[i] = nxt
        for ch in moved:
            label_of_chain[ch] = nxt
        nxt += 1
    if n_post_splits:
        method += f"+post_split({n_post_splits})"

    # RECONCILE PASS (2026-09-29): fragments of one voice must not
    # survive as separate speakers. Two clusters above
    # POWERSET_RECONCILE_SIM merge UNLESS evidence separates them:
    # a co-activity cannot-link between their chains (you cannot
    # overlap yourself) or an F0 register straddle. Skipped entirely
    # under an explicit user num_speakers (user K outranks all
    # mechanisms). Proof > similarity > scores.
    n_reconciled = 0
    reconcile_blocked = []
    if cfg.get("num_speakers") is None:
        from lib.input_schema import (POWERSET_RECONCILE_CORROB_SEC,
                                      POWERSET_RECONCILE_HANDOVER_MAX,
                                      POWERSET_RECONCILE_LINK_MIN_SEC,
                                      POWERSET_RECONCILE_QUICK_SEC,
                                      POWERSET_RECONCILE_SIM)
        _timeline = sorted(
            (float(high_sents[i].get("segment_start",
                                     high_sents[i]["start"])),
             float(high_sents[i].get("segment_end",
                                     high_sents[i]["end"])), i)
            for i in range(len(high_sents)) if labels[i] >= 0)

        def _quick_handovers(a, b):
            n = 0
            for k in range(len(_timeline) - 1):
                s0, e0, i0 = _timeline[k]
                s1, e1, i1 = _timeline[k + 1]
                la, lb = int(labels[i0]), int(labels[i1])
                if {la, lb} == {a, b} \
                        and 0 <= s1 - e0 <= POWERSET_RECONCILE_QUICK_SEC:
                    n += 1
            return n

        def _co_sec(chs_a, chs_b):
            """Measured co-active seconds between two chain groups,
            summed over their pair-class links (link_sec; artifacts
            predating co_active_secs contribute the 0.25 s emission
            lower bound per link, filled at load_guides)."""
            return sum(
                link_sec.get((min(idx_of[ca], idx_of[cb]),
                              max(idx_of[ca], idx_of[cb])), 0.0)
                for ca in chs_a for cb in chs_b)
        while True:
            members_r: dict = {}
            for ch, lb in label_of_chain.items():
                if lb >= 0:
                    members_r.setdefault(lb, []).append(ch)
            labs = sorted(members_r)
            if len(labs) < 2:
                break
            cents, regs = {}, {}
            for lb in labs:
                idxs = [idx_of[ch] for ch in members_r[lb]]
                ws = np.array([sum(sg["duration"] for sg in by_chain[ch])
                               for ch in members_r[lb]])
                v = np.average(track_matrix[idxs], axis=0,
                               weights=np.maximum(ws, 1e-6))
                n = np.linalg.norm(v)
                cents[lb] = v / n if n > 1e-10 else v
                f0s = [f0_of_chain.get(ch) for ch in members_r[lb]]
                f0s = [f for f in f0s if f]
                regs[lb] = sorted(f0s)[len(f0s) // 2] if f0s else None
            best = None
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    a, b = labs[i], labs[j]
                    sim = float(cents[a] @ cents[b])
                    if sim < POWERSET_RECONCILE_SIM:
                        continue
                    # evidence gate 1: durable co-presence — you
                    # cannot overlap yourself. Durable proof only:
                    # flickery false overlap (applause/laughter) must
                    # not veto a merge, so the pair-class seconds must
                    # accumulate to the bar. (F0 nearest-neighbor
                    # links deliberately don't participate — register
                    # is gate 2's job, at cluster-median level.)
                    co = _co_sec(members_r[a], members_r[b])
                    if co >= POWERSET_RECONCILE_LINK_MIN_SEC:
                        reconcile_blocked.append(
                            [a, b, round(sim, 3),
                             f"co_active({co:.1f}s)"])
                        continue
                    # evidence gate 3: conversation veto — quick
                    # A<->B alternation CORROBORATED by co-presence
                    # means two people. Handover count alone cannot
                    # tell dialogue from a flap-fragment (a cluster
                    # assignment alternating mid-monologue also "hands
                    # over" sub-second) — but a monologue never
                    # overlaps itself. The corroboration bar sits
                    # above one emission-floor blip (a lone narrator's
                    # music transient must not veto its own
                    # reunification) and far below gate 1's
                    # standalone-proof bar.
                    if (_quick_handovers(a, b)
                            >= POWERSET_RECONCILE_HANDOVER_MAX
                            and co >= POWERSET_RECONCILE_CORROB_SEC):
                        reconcile_blocked.append(
                            [a, b, round(sim, 3),
                             f"conversation({co:.1f}s)"])
                        continue
                    # evidence gate 2: the clusters are different vocal
                    # registers (same predicate as the track-level
                    # gates — the band alone cannot tell a register
                    # change from an emphatic passage)
                    ra, rb = regs[a], regs[b]
                    if _cross_register(ra, rb):
                        reconcile_blocked.append(
                            [a, b, round(sim, 3),
                             f"register({ra:.0f}/{rb:.0f})"])
                        continue
                    if best is None or sim > best[0]:
                        best = (sim, a, b)
            if best is None:
                break
            _, keep, drop = best
            for ch, lb in list(label_of_chain.items()):
                if lb == drop:
                    label_of_chain[ch] = keep
            labels[labels == drop] = keep
            n_reconciled += 1
    if n_reconciled:
        method += f"+reconcile({n_reconciled})"


    # DUST ABSORPTION (2026-09-29): the schema's min_turns_per_speaker
    # declares the minimum evidence for asserting a distinct speaker;
    # a final cluster below it is not a speaker, it is residue (a
    # 2-turn shard of a narrator survived as its own K). Members fold
    # into the nearest surviving cluster within the assignment margin,
    # else UNKNOWN — absorption with abstention, never a forced home.
    n_dust = 0
    if cfg.get("num_speakers") is None:
        from lib.input_schema import DEFAULT_MIN_TURNS_PER_SPEAKER
        min_turns = int(cfg.get("min_turns_per_speaker",
                                DEFAULT_MIN_TURNS_PER_SPEAKER) or
                        DEFAULT_MIN_TURNS_PER_SPEAKER)
        counts = Counter(int(l) for l in labels.tolist() if l >= 0)
        dust = {lb for lb, n in counts.items() if n < min_turns}
        survivors = [lb for lb in counts if lb not in dust]
        if dust and survivors:
            cents_d = {}
            for lb in survivors:
                idxs = [i for i in range(len(high_sents))
                        if labels[i] == lb]
                v = np.mean([np.asarray(high_sents[i]["embedding"],
                                        dtype=np.float64)
                             for i in idxs], axis=0)
                n = np.linalg.norm(v)
                cents_d[lb] = v / n if n > 1e-10 else v
            for i in range(len(high_sents)):
                if int(labels[i]) not in dust:
                    continue
                v = np.asarray(high_sents[i]["embedding"],
                               dtype=np.float64)
                n = np.linalg.norm(v)
                if n <= 1e-10:
                    labels[i] = -1
                    continue
                v = v / n
                sims = sorted(((float(v @ c), lb)
                               for lb, c in cents_d.items()),
                              reverse=True)
                if sims and (len(sims) == 1
                             or sims[0][0] - sims[1][0] >= assign_margin):
                    labels[i] = sims[0][1]
                else:
                    labels[i] = -1
            for ch, lb in list(label_of_chain.items()):
                if lb in dust:
                    label_of_chain[ch] = -1
            n_dust = len(dust)
    if n_dust:
        method += f"+dust({n_dust})"

    # The post-passes run AFTER K selection, so the sweep's own
    # violation count no longer describes the SHIPPED labels. Recount
    # on the final assignment (report-only, like the sweep's): a
    # reconcile merge may drop K below the co-active clique floor, but
    # only because that floor can rest on a 0.25 s pair-class blip
    # while vetoing a merge needs POWERSET_RECONCILE_LINK_MIN_SEC of
    # real co-presence. Where the two instruments disagree, the shipped
    # K and the floor it crossed are both named — evidence, not a
    # silent override.
    final_violations = sum(
        1 for x, y in cannot
        if 0 <= label_of_chain.get(chains[x], -1)
        == label_of_chain.get(chains[y], -1))
    k_final = len({lb for lb in label_of_chain.values() if lb >= 0})
    k_below_floor = [k_final, int(floor)] if k_final < floor else None

    diag = dict(diag)
    diag.update({"unit": "powerset_tracks", "n_tracks": len(chains),
                 "xchunk_binds": n_binds_applied,
                 "xchunk_bind_candidates": len(cands),
                 "local_speaker_floor": guides.get("local_speaker_floor"),
                 "floor_user_conflict": floor_user_conflict,
                 "n_post_splits": n_post_splits,
                 "n_reconciled": n_reconciled,
                 "n_dust_absorbed": n_dust,
                 # dedup: the merge loop re-enumerates surviving pairs
                 # after every merge, re-blocking the same pair
                 "reconcile_blocked": [list(t) for t in dict.fromkeys(
                     tuple(b) for b in reconcile_blocked)][:12],
                 "post_pass_cannot_link_violations": final_violations,
                 "k_below_floor": k_below_floor,
                 "n_cannot_links": len(cannot),
                 "n_f0_cannot_links": n_f0_links,
                 "bimodal_tracks": n_bimodal_tracks,
                 "clique_floor": floor,
                 "trackless_high_sentences": trackless})
    return labels, list(high_sents), f"tracks({method})", diag


def _load_all_sentence_embeddings(
    bucket:    str,
    iterators: list[dict],
) -> tuple[list[dict], list[dict], list[dict], int, list[dict]]:
    """
    Returns (high_sents, low_sents, all_skipped, embedding_dim,
    track_segs — powerset track-segment embeddings, [] when the
    guides flag was off).
    """
    high:    list[dict] = []
    low:     list[dict] = []
    skipped: list[dict] = []
    tracks:  list[dict] = []
    embedding_dim: int | None = None

    for it in sorted(iterators, key=lambda x: x["dyn_chunk_idx"]):
        if it.get("status") != "COMPLETED":
            continue
        emb_key = it.get("embeddings_key")
        if not emb_key:
            continue

        doc = get_json(bucket, emb_key)

        file_dim = doc.get("embedding_dim")
        if file_dim is not None:
            file_dim = int(file_dim)
            if embedding_dim is None:
                embedding_dim = file_dim
            elif file_dim != embedding_dim:
                raise ValueError(
                    f"embedding_dim mismatch chunk {it['dyn_chunk_idx']}")

        for sent in doc.get("sentence_embeddings", []):
            if "embedding" not in sent:
                continue
            if sent.get("quality") == "low":
                low.append(sent)
            else:
                high.append(sent)

        for skip in doc.get("skipped_sentences", []):
            skipped.append(skip)

        tracks.extend(doc.get("track_embeddings") or [])

        log.info("  chunk %2d: %d high  %d low  %d skip",
                 it["dyn_chunk_idx"],
                 doc.get("num_high", 0),
                 doc.get("num_low", 0),
                 doc.get("num_skipped", 0))

    high.sort(key=lambda s: float(s["start"]))
    low.sort(key=lambda s: float(s["start"]))
    skipped.sort(key=lambda s: float(s["start"]))

    if embedding_dim is None and high:
        embedding_dim = len(high[0]["embedding"])

    return high, low, skipped, (embedding_dim or 0), tracks


def _load_overlap_evidence(bucket: str, output_prefix: str) -> list[dict]:
    """Concatenate the per-fixed-chunk cross-talk evidence written by
    StateRunVADModel (segments_{n}.json "overlap_evidence" — P3)."""
    evidence: list[dict] = []
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket,
                                   Prefix=f"{output_prefix}/segments/"):
        for obj in page.get("Contents") or []:
            try:
                doc = get_json(bucket, obj["Key"])
            except Exception as exc:                 # noqa: BLE001
                log.warning("overlap_evidence: could not read %s (%s)",
                            obj["Key"], exc)
                continue
            evidence.extend(doc.get("overlap_evidence") or [])
    return evidence


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _find_speaker_by_overlap(
    start: float, end: float, diarization: list[dict],
    max_gap_sec: float = float("inf"),
) -> str | None:
    best_spk, best_ov = None, 0.0
    for seg in diarization:
        ov = max(0.0, min(end, seg["end"]) - max(start, seg["start"]))
        if ov > best_ov:
            best_ov = ov
            best_spk = seg["speaker"]
    if best_spk:
        return best_spk
    # Nearest by midpoint — but only within max_gap_sec: a sentence far
    # from any labeled speech must stay UNKNOWN, not adopt whoever
    # happens to be nearest in a silent stretch
    center = (start + end) / 2
    best_dist = float("inf")
    for seg in diarization:
        dist = abs((seg["start"] + seg["end"]) / 2 - center)
        if dist < best_dist:
            best_dist = dist
            best_spk = seg["speaker"]
    return best_spk if best_dist <= max_gap_sec else None


def _build_diarization(
    sentences: list[dict], labels: np.ndarray, merge_gap: float = 0.5,
) -> list[dict]:
    segments = [
        {"start": float(s["start"]), "end": float(s["end"]),
         "speaker": f"SPEAKER_{int(lbl):02d}"}
        for s, lbl in zip(sentences, labels)
    ]
    if not segments:
        return []
    segments.sort(key=lambda s: s["start"])
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if (seg["speaker"] == prev["speaker"]
                and seg["start"] - prev["end"] <= merge_gap):
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged
