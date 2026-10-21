# embedding_handler.py
"""
StateRunSpeechEmbeddingModel  (chained within Map after Whisper, 3584 MB / 2-cores)

Reads sentences from transcript_{n}.json, slices the audio per sentence
(using the longest contiguous speech segment within the sentence's word list),
and extracts WeSpeakerResNet34 ONNX speaker embeddings at two quality tiers:

  high  (segment_duration ≥ min_embedding_duration_sec, default 0.75 s)
      Included in the AHC clustering input.

  low   (≥ soft_embedding_min_sec (default 0.30 s) but < high threshold)
      Extracted but NOT used for clustering.  After clustering, these are
      assigned to the nearest speaker centroid.  Short replies and
      back-channels often fall here.

  skipped  (< soft_embedding_min_sec, or no words, or empty audio slice)
      No embedding.  After clustering, speaker is assigned via diarization
      overlap with the surrounding labeled sentences.

Output — embeddings_{n}.json:
  {
    "sentence_embeddings": [
      { "dyn_chunk_idx": int, "sentence_idx": int, "start": float, "end": float,
        "duration": float, "segment_start": float, "segment_end": float,
        "segment_duration": float, "quality": "high"|"low", "embedding": [float...] }
    ],
    "skipped_sentences": [
      { "dyn_chunk_idx": int, "sentence_idx": int, ..., "reason": str }
    ]
  }

Lambda configuration
─────────────────────
  Memory  : 3584 MB (TwoCores)
  Timeout : 900 s
  Arch    : arm64
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback

import numpy as np
import onnxruntime as ort

from lib.model_registry import PYANNOTE_MODEL
from lib.model_store import ensure_in_tmp, find_file, cleanup_models
from lib.wav_utils import TARGET_SR, get_wav_info, read_wav_window
from lib.embedding import probe_embedding_model, extract_embedding
from lib.f0 import slice_f0
from lib.ort_init import ensure_cpuinfo_files, get_num_cpus
from lib.s3_utils import s3 as _s3, get_json, put_json

ensure_cpuinfo_files()

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False

_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
if not log.handlers:
    log.addHandler(_handler)

_EMBEDDING_INTERNAL_GAP_SEC = 0.5


def _progress(step: str, detail: str = "", t_start: float = 0.0) -> None:
    elapsed = f"  (+{time.monotonic() - t_start:.2f}s)" if t_start else ""
    log.info(f"  ▶ {step:<30s}  {detail}{elapsed}")


# =============================================================================
# ONNX SESSION — module-level cache
# =============================================================================

_session:     ort.InferenceSession | None = None
_input_name:  str | None = None
_output_name: str | None = None
_emb_dim:     int | None = None


def _get_session(
    model_bucket: str,
    model_prefix: str = "_model_artefacts",
    memory_mb:    int | None = None,
) -> tuple[ort.InferenceSession, str, str, int]:
    global _session, _input_name, _output_name, _emb_dim
    if _session is not None:
        log.info("[cache ✅] ONNX session already loaded")
        return _session, _input_name, _output_name, _emb_dim

    s3_key = PYANNOTE_MODEL["s3_key"]
    if model_prefix and not s3_key.startswith(model_prefix):
        s3_key = model_prefix.strip("/") + "/" + s3_key.split("/", 1)[1]

    ensure_in_tmp(
        bucket=model_bucket, s3_key=s3_key,
        tmp_dir=PYANNOTE_MODEL["tmp_dir"],
        label="pyannote/community-1",
    )

    for pattern in ("embedding.int8.onnx", "embedding.onnx"):
        try:
            onnx_path = find_file(PYANNOTE_MODEL["tmp_dir"], pattern)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(
            f"No embedding ONNX found in {PYANNOTE_MODEL['tmp_dir']}")

    _num_cpus = get_num_cpus(memory_mb)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _num_cpus
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    t0 = time.monotonic()
    _session = ort.InferenceSession(
        onnx_path, opts, providers=["CPUExecutionProvider"])
    _input_name, _output_name, _emb_dim = probe_embedding_model(_session)
    log.info(
        f"ONNX session ready: {time.monotonic() - t0:.2f}s  dim={_emb_dim}")
    return _session, _input_name, _output_name, _emb_dim


# =============================================================================
# HELPERS
# =============================================================================

def _load_sentences(bucket: str, transcript_key: str) -> list[dict]:
    return get_json(bucket, transcript_key).get("sentences", [])


def _find_longest_speech_segment(
    words:       list[dict],
    max_gap_sec: float = _EMBEDDING_INTERNAL_GAP_SEC,
    fences:      list[float] | None = None,
) -> tuple[float, float] | None:
    """A fence (powerset in-speech speaker change) is a hard
    break whatever the word gap: a slice spanning it mixes two voices
    into one embedding — the measured cluster-bridge mechanism."""
    if not words:
        return None

    segments: list[tuple[float, float]] = []
    seg_start_idx = 0

    for i in range(1, len(words)):
        gap = float(words[i]["start"]) - float(words[i - 1]["end"])
        fenced = any(float(words[i - 1]["end"]) - 0.05 <= t
                     <= float(words[i]["start"]) + 0.05
                     for t in (fences or []))
        if gap > max_gap_sec or fenced:
            segments.append((
                float(words[seg_start_idx]["start"]),
                float(words[i - 1]["end"]),
            ))
            seg_start_idx = i

    segments.append((
        float(words[seg_start_idx]["start"]),
        float(words[-1]["end"]),
    ))
    return max(segments, key=lambda s: s[1] - s[0])


# =============================================================================
# HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    cleanup_models()

    log.setLevel(logging.INFO)
    log.info("StateRunSpeechEmbeddingModel event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Unpack item_data ───────────────────────────────────────────────────
        item: dict = event.get("item_data", event)

        required = {
            "dyn_chunk_idx", "fetch_start_sec", "fetch_end_sec",
            "transcript_key", "audio_bucket", "audio_key", "output_bucket", "output_prefix",
            "audio_cfg", "model_cfg", "embed_cfg",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")

        dyn_chunk_idx:   int = int(item["dyn_chunk_idx"])
        fetch_start_sec: float = float(item["fetch_start_sec"])
        fetch_end_sec:   float = float(item["fetch_end_sec"])
        fetch_duration:  float = fetch_end_sec - fetch_start_sec
        transcript_key:  str = item["transcript_key"]

        audio_bucket:     str = item["audio_bucket"]
        audio_key:        str = item["audio_key"]
        output_bucket:    str = item["output_bucket"]
        output_prefix:    str = item["output_prefix"]

        # ── model config ─────────────────────────────────────────────────────────
        model_cfg:       dict = item["model_cfg"]
        model_bucket:    str = model_cfg["model_bucket"]
        model_prefix:    str = model_cfg["model_prefix"]

        # ── audio config ─────────────────────────────────────────────────────────
        audio_cfg:       dict = item["audio_cfg"]
        # buffers from read_wav_window are TARGET_SR — slice and
        # embed at the PIPELINE rate, never the file's (see the
        # 2026-09-29 note in the whisper state)
        sample_rate:     int = TARGET_SR
        file_sr = int(audio_cfg["sample_rate"])
        if file_sr != TARGET_SR:
            log.warning(
                "audio_cfg sample_rate=%d != pipeline %d — fetched "
                "buffers are resampled to the pipeline rate", file_sr,
                TARGET_SR)

        # ── embed config ─────────────────────────────────────────────────────────
        embed_cfg:                  dict = item["embed_cfg"]
        min_embedding_duration_sec: float = float(embed_cfg["min_embedding_duration_sec"])
        soft_embedding_min_sec:     float = float(embed_cfg["soft_embedding_min_sec"])

        memory_mb = getattr(context, "memory_limit_in_mb", None)
        if memory_mb:
            memory_mb = int(memory_mb)

        log.info(
            "  Chunk %d: fetch=[%.1fs, %.1fs]  "
            "min_embed=%.2fs  soft_min=%.2fs",
            dyn_chunk_idx, fetch_start_sec, fetch_end_sec,
            min_embedding_duration_sec, soft_embedding_min_sec,
        )
        _progress("START", f"chunk={dyn_chunk_idx}", t_start=t0)

        # ── ONNX session ─────────────────────────────────────────────────────
        session, input_name, output_name, emb_dim = _get_session(
            model_bucket, model_prefix, memory_mb)

        # ── Load sentences ────────────────────────────────────────────────────
        sentences = _load_sentences(output_bucket, transcript_key)
        log.info("  %d sentences in chunk %d", len(sentences), dyn_chunk_idx)
        if not sentences:
            # a chunk can legitimately end up speechless: its only words
            # were pad-boundary duplicates the resolution ladder dropped.
            # Emit an empty embeddings doc; the clustering state's readers
            # all default on absent lists.
            log.warning("  chunk %d has no sentences — emitting empty "
                        "embeddings doc", dyn_chunk_idx)
            embeddings_key = (f"{output_prefix}/embeddings/"
                              f"embeddings_{dyn_chunk_idx}.json")
            put_json(output_bucket, embeddings_key, {
                "dyn_chunk_idx":       dyn_chunk_idx,
                "fetch_start_sec":     fetch_start_sec,
                "fetch_end_sec":       fetch_end_sec,
                "num_embedded":        0,
                "num_high":            0,
                "num_low":             0,
                "num_skipped":         0,
                "sentence_embeddings": [],
                "skipped_sentences":   [],
                "track_embeddings":    [],
            })
            return {
                **item,
                "embeddings_key": embeddings_key,
                "num_embedded":   0,
                "num_high":       0,
                "num_low":        0,
                "num_skipped":    0,
                "status":         "COMPLETED",
                "elapsed_sec":    round(time.monotonic() - t0, 2),
            }

        # ── Fetch audio ───────────────────────────────────────────────────────
        wav_info = get_wav_info(audio_bucket, audio_key)
        audio = read_wav_window(
            audio_bucket, audio_key, wav_info,
            start_sec=fetch_start_sec, duration_sec=fetch_duration)
        if len(audio) == 0:
            raise RuntimeError(f"Empty audio for chunk {dyn_chunk_idx}")

        # ── Extract embeddings ────────────────────────────────────────────────
        _progress("embeddings: extracting", f"{len(sentences)} sentences",
                  t_start=t0)
        t_emb = time.monotonic()
        # powerset guides (2026-09-25): artifact presence = assessed
        guides = None
        chunk_fences: list[float] = []
        chunk_ovl: list[list[float]] = []
        try:
            from lib.powerset_guides import load_guides, subtract_spans
            guides = load_guides(output_bucket, output_prefix)
        except Exception as exc:                 # noqa: BLE001 — degrade
            log.warning("powerset guide load failed (continuing "
                        "unguided): %s", exc)
        if guides:
            chunk_fences = [t for t in guides["fences"]
                            if fetch_start_sec <= t <= fetch_end_sec]
            chunk_ovl = [sp for sp in guides["overlap_spans"]
                         if sp[1] > fetch_start_sec
                         and sp[0] < fetch_end_sec]

        embedded: list[dict] = []
        skipped:  list[dict] = []

        for sent_idx, sent in enumerate(sentences):
            words = sent.get("words", [])
            sent_start = float(sent["start"])
            sent_end = float(sent["end"])
            sent_dur = round(sent_end - sent_start, 3)

            seg = _find_longest_speech_segment(words,
                                               fences=chunk_fences)
            if seg is None:
                skipped.append({
                    "dyn_chunk_idx": dyn_chunk_idx,
                    "sentence_idx":  sent_idx,
                    "start":         sent_start,
                    "end":           sent_end,
                    "duration":      sent_dur,
                    "reason":        "no words in sentence",
                })
                continue

            seg_start, seg_end = seg
            if chunk_ovl:
                # cross-talk excision (pyannote's own pipeline sets
                # exclude_overlap=true for embeddings): keep the
                # longest single-voice piece of the slice
                pieces = subtract_spans(seg_start, seg_end, chunk_ovl)
                if pieces:
                    seg_start, seg_end = max(pieces,
                                             key=lambda p: p[1] - p[0])
            seg_dur = round(seg_end - seg_start, 3)

            # Determine quality tier
            if seg_dur >= min_embedding_duration_sec:
                quality = "high"
            elif seg_dur >= soft_embedding_min_sec:
                quality = "low"
            else:
                skipped.append({
                    "dyn_chunk_idx": dyn_chunk_idx,
                    "sentence_idx":  sent_idx,
                    "start":         sent_start,
                    "end":           sent_end,
                    "duration":      sent_dur,
                    "segment_duration": seg_dur,
                    "reason": (
                        f"too short ({seg_dur:.2f}s < {soft_embedding_min_sec:.2f}s)"),
                })
                continue

            # Slice audio
            s = max(0, int((seg_start - fetch_start_sec) * sample_rate))
            e = min(len(audio),
                    int((seg_end - fetch_start_sec) * sample_rate))
            if e <= s:
                skipped.append({
                    "dyn_chunk_idx": dyn_chunk_idx,
                    "sentence_idx":  sent_idx,
                    "start":         sent_start,
                    "end":           sent_end,
                    "duration":      sent_dur,
                    "reason":        "empty audio slice",
                })
                continue

            # Extract — use soft_min_sec as floor so low-quality extractions
            # aren't rejected by extract_embedding's internal length check
            emb = extract_embedding(
                audio[s:e], session, input_name, output_name,
                sr=sample_rate,
                min_segment_sec=soft_embedding_min_sec,
            )

            if emb is None:
                skipped.append({
                    "dyn_chunk_idx": dyn_chunk_idx,
                    "sentence_idx":  sent_idx,
                    "start":         sent_start,
                    "end":           sent_end,
                    "duration":      sent_dur,
                    "reason":        "embedding extraction returned None",
                })
                continue

            # per-sentence F0 statistic (lib/f0) — an ADDITIVE column for
            # downstream voice↔face compatibility matching; deliberately
            # outside every clustering input (2026-09-23)
            f0_hz, f0_voiced = slice_f0(audio[s:e], sample_rate)

            embedded.append({
                "dyn_chunk_idx":    dyn_chunk_idx,
                "sentence_idx":     sent_idx,
                "start":            sent_start,
                "end":              sent_end,
                "duration":         sent_dur,
                "segment_start":    round(seg_start, 3),
                "segment_end":      round(seg_end, 3),
                "segment_duration": seg_dur,
                "quality":          quality,
                "f0_hz":            (round(f0_hz, 1)
                                     if f0_hz is not None else None),
                "f0_voiced_ratio":  round(f0_voiced, 3),
                "embedding":        emb.tolist(),
            })

        # ── track-segment embeddings (2026-09-25) ─────────────────────
        # The rehearsal's decisive negative: averaging per-sentence
        # vectors does NOT substitute for embedding long CONTIGUOUS
        # powerset-pure segments (reality_a collapsed under pooling,
        # rescued under segment embeddings — embedding(concat) !=
        # mean(embedding(pieces))). So track segments are embedded
        # directly; clustering pools per chain, duration-weighted.
        track_embedded: list[dict] = []
        if guides:
            for tr in guides["tracks"]:
                for a, b in tr["segs"]:
                    a2 = max(float(a), fetch_start_sec)
                    b2 = min(float(b), fetch_end_sec)
                    if b2 - a2 < min_embedding_duration_sec:
                        continue
                    ts = max(0, int((a2 - fetch_start_sec) * sample_rate))
                    te = min(len(audio),
                             int((b2 - fetch_start_sec) * sample_rate))
                    if te <= ts:
                        continue
                    emb_t = extract_embedding(
                        audio[ts:te], session, input_name, output_name,
                        sr=sample_rate,
                        min_segment_sec=min_embedding_duration_sec)
                    if emb_t is None:
                        continue
                    track_embedded.append({
                        "chain":     tr["id"],
                        "start":     round(a2, 3),
                        "end":       round(b2, 3),
                        "duration":  round(b2 - a2, 3),
                        "embedding": emb_t.tolist(),
                    })
            if track_embedded:
                log.info("  track embeddings: %d segments over %d chains",
                         len(track_embedded),
                         len({t["chain"] for t in track_embedded}))

        emb_sec = time.monotonic() - t_emb
        n_high = sum(1 for e in embedded if e["quality"] == "high")
        n_low = sum(1 for e in embedded if e["quality"] == "low")
        log.info(
            "  embeddings: %d high  %d low  %d skipped  %.1fs",
            n_high, n_low, len(skipped), emb_sec,
        )

        # ── Store to S3 ──────────────────────────────────────────────────────
        embeddings_key = f"{output_prefix}/embeddings/embeddings_{dyn_chunk_idx}.json"

        emb_doc = {
            "dyn_chunk_idx":       dyn_chunk_idx,
            "fetch_start_sec":     fetch_start_sec,
            "fetch_end_sec":       fetch_end_sec,
            "embedding_dim":       emb_dim,
            "num_embedded":        len(embedded),
            "num_high":            n_high,
            "num_low":             n_low,
            "num_skipped":         len(skipped),
            "sentence_embeddings": embedded,
            "skipped_sentences":   skipped,
            "track_embeddings":    track_embedded,
        }
        put_json(output_bucket, embeddings_key, emb_doc)

        elapsed = round(time.monotonic() - t0, 2)
        output = {
            **item,
            "embeddings_key": embeddings_key,
            "num_embedded":   len(embedded),
            "num_high":       n_high,
            "num_low":        n_low,
            "num_skipped":    len(skipped),
            "status":         "COMPLETED",
            "elapsed_sec":    elapsed,
        }
        _progress("DONE",
                  f"high={n_high}  low={n_low}  skip={len(skipped)}  "
                  f"elapsed={elapsed}s", t_start=t0)
        return output

    except Exception as exc:
        log.error("[ERR] Embedding chunk %s — %s: %s",
                  event.get("item_data", {}).get("dyn_chunk_idx", "?"),
                  type(exc).__name__, exc)
        traceback.print_exc()
        raise
