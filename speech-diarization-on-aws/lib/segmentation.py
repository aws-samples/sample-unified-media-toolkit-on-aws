# lib/segmentation.py
"""
Pyannote segmentation ONNX — sliding-window VAD runner.

Model contract
──────────────
  Input  : "waveform"  float32[1, 1, window_samples]
  Output : "logits"    float32[1, num_frames, num_powerset_classes]

  The community-1 model outputs log-softmax probabilities over 7 powerset
  classes (LogSoftmax final layer — values in (-inf, 0)).

Powerset encoding (community-1, pipeline_config.json)
──────────────────────────────────────────────────────
  num_powerset_classes = 7
  max_speakers_per_chunk = 3, max_speakers_per_frame = 2

  class 0        → silence
  classes 1,2,3  → single active speaker (spk0, spk1, spk2)
  classes 4,5,6  → overlapping pairs (spk0+spk1, spk0+spk2, spk1+spk2)

  Speech activity per frame = 1 - P(class_0)
  Overlap                   = max(P(class_4), P(class_5), P(class_6)) > threshold

Sliding window & aggregation
─────────────────────────────
  step=1.0s (pipeline_config), window=10.0s → 10x overlap per interior frame.
  Aggregation uses Hamming window weighting, matching pyannote inference.py.

Role in the pipeline
─────────────────────
  This is a pure VAD step. It answers "when is someone speaking, and is
  there overlap?" It does NOT identify who is speaking — that is done by:
    B1 (extract_embeddings) + B2 (global_cluster / VBx).
  The `has_overlap` flag is carried forward to help B1/B2 optionally
  down-weight unreliable overlap-region embeddings.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import onnxruntime as ort
from lib.input_schema import (
    DEFAULT_SEG_THRESHOLD,
    DEFAULT_MIN_SPEECH_DURATION_SEC,
    DEFAULT_MIN_SILENCE_DURATION_SEC,
)

log = logging.getLogger(__name__)

# Default model parameters — from pipeline_config.json (pyannote community-1)
SEG_WINDOW_SEC   = 10.0   # model receptive field
SEG_STEP_SEC     = 1.0    # sliding step  (pipeline_config: step=1.0)
# silence-class probability threshold: single source of truth is
# input_schema.DEFAULT_SEG_THRESHOLD (caller-tunable via hidden
# vad_cfg.seg_threshold)


# =============================================================================
# MODEL METADATA PROBE
# =============================================================================

def probe_model(
    session:     ort.InferenceSession,
    sample_rate: int = 16_000,
    window_sec:  float = SEG_WINDOW_SEC,
) -> tuple[str, int, int, float]:
    """
    Return (input_name, frames_per_window, num_powerset_classes, frame_rate_hz).

    Tries the static output shape first; falls back to a dummy inference
    for models with fully dynamic output shapes (e.g. INT8 quantised).
    """
    input_name = session.get_inputs()[0].name
    output_shape = session.get_outputs()[0].shape
    window_samples = int(window_sec * sample_rate)

    frames_per_window: Optional[int] = None
    num_ps_classes:    Optional[int] = None

    if (output_shape and len(output_shape) == 3
            and isinstance(output_shape[1], int) and output_shape[1] > 0
            and isinstance(output_shape[2], int) and output_shape[2] > 0):
        frames_per_window = output_shape[1]
        num_ps_classes = output_shape[2]

    if frames_per_window is None or num_ps_classes is None:
        log.info("Dynamic output shape — running dummy inference to probe model")
        dummy = np.zeros((1, 1, window_samples), dtype=np.float32)
        out = session.run(["logits"], {input_name: dummy})[0]
        frames_per_window = out.shape[1]
        num_ps_classes = out.shape[2]

    frame_rate = frames_per_window / window_sec
    log.info(
        f"Model probe: input='{input_name}'  "
        f"frames_per_window={frames_per_window}  "
        f"num_speakers={num_ps_classes}  "
        f"frame_rate={frame_rate:.2f} Hz  "
        f"frame_shift={1000/frame_rate:.1f} ms"
    )
    return input_name, frames_per_window, num_ps_classes, frame_rate


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_segmentation(
    audio:                    np.ndarray,
    session:                  ort.InferenceSession,
    fetch_start_sec:          float = 0.0,
    sample_rate:              int = 16_000,
    window_sec:               float = SEG_WINDOW_SEC,
    step_sec:                 float = SEG_STEP_SEC,
    threshold:                float = DEFAULT_SEG_THRESHOLD,
    min_speech_duration_sec:  float = DEFAULT_MIN_SPEECH_DURATION_SEC,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_DURATION_SEC,
    collect_overlap_evidence: bool = False,
    powerset_collector=None,
):
    """
    Run pyannote segmentation ONNX over `audio` with a sliding window.

    Pure VAD — outputs speech segments with overlap flag. Does not assign
    speaker identities (that is done by the downstream B1/B2 states).

    Parameters
    ──────────
    audio            : float32 mono waveform at `sample_rate` Hz
    session          : ONNX InferenceSession for the segmentation model
    fetch_start_sec  : absolute time of audio[0] in the original file

    Returns
    ───────
    list of turn dicts sorted by start time:
      {
        "start":       float   — absolute seconds in original file
        "end":         float
        "duration":    float
        "has_overlap": bool    — True if 2+ speakers active simultaneously
      }
    With collect_overlap_evidence=True, returns (turns, evidence) where
    evidence is the per-window channel-resolved cross-talk list from
    extract_overlap_evidence() (P3 cannot-link input). Evidence is
    collected from EVERY window (duplicates from the 1 s stride dedupe
    downstream; half-window sampling measured 3-7x worse recall).
    """
    n_samples = len(audio)
    window_samples = int(window_sec * sample_rate)
    step_samples = int(step_sec * sample_rate)

    if n_samples == 0:
        log.warning("run_segmentation: empty audio — returning []")
        return ([], []) if collect_overlap_evidence else []

    # ── 1. Probe model ────────────────────────────────────────────────────────
    input_name, frames_per_window, num_ps_classes, frame_rate = probe_model(
        session, sample_rate, window_sec
    )

    # ── 2. Build window start positions ───────────────────────────────────────
    starts = _compute_window_starts(n_samples, window_samples, step_samples)
    log.info(
        f"Segmentation: {n_samples/sample_rate:.1f}s audio  "
        f"{len(starts)} windows  (window={window_sec}s, step={step_sec}s)"
    )

    # ── 3. Global accumulator (Hamming-weighted overlap-add) ──────────────────
    total_frames = max(1, int(math.ceil(n_samples / sample_rate * frame_rate)))
    # Accumulate powerset probabilities across overlapping windows
    ps_sum = np.zeros((total_frames, num_ps_classes), dtype=np.float64)
    ps_count = np.zeros(total_frames,                 dtype=np.float64)

    # Hamming window — matches pyannote inference.py aggregate(hamming=True)
    hamming_window = np.hamming(frames_per_window).reshape(-1, 1)  # [F, 1]

    # ── 4. Sliding window inference ───────────────────────────────────────────
    from lib.input_schema import (OVERLAP_EVIDENCE_MIN_SOLO_SEC,
                                  OVERLAP_EVIDENCE_MIN_OVERLAP_SEC)
    evidence: list[dict] = []
    # evidence collected from EVERY window: near-duplicates from the
    # 1 s stride are deduped downstream (constraints.resolve_evidence
    # keys by sentence pair), and collection is pure numpy on
    # already-computed probabilities. EVERY window on purpose —
    # measured on sim dialogues (2026-09-16): half-window sampling
    # cost 3-7x recall for zero benefit (an overlap event only yields
    # evidence in the few windows where BOTH participants also have
    # solo spans).

    for win_idx, start in enumerate(starts):
        chunk = _get_window(audio, start, window_samples)
        inp = chunk[np.newaxis, np.newaxis, :].astype(np.float32)  # [1,1,W]
        try:
            raw = session.run(["logits"], {input_name: inp})[0]   # [1,F,C]
        except Exception as exc:
            log.warning(f"Window {win_idx} inference failed: {exc} — skipping")
            continue

        probs = _to_probabilities(raw[0])   # [F, num_ps_classes]

        if powerset_collector is not None:
            # powerset guides (lib/powerset_guides.py): the probs are
            # already in memory — this costs one argmax per window
            powerset_collector.append(
                fetch_start_sec + start / sample_rate, probs)

        if collect_overlap_evidence:
            evidence.extend(extract_overlap_evidence(
                probs,
                fetch_start_sec + start / sample_rate,
                frame_rate, threshold,
                OVERLAP_EVIDENCE_MIN_SOLO_SEC,
                OVERLAP_EVIDENCE_MIN_OVERLAP_SEC))

        frame_start = int(start * frames_per_window / window_samples)
        frame_end = min(frame_start + frames_per_window, total_frames)
        n_frames = frame_end - frame_start

        w = hamming_window[:n_frames]
        ps_sum[frame_start:frame_end] += (probs[:n_frames]
                                          * w).astype(np.float64)
        ps_count[frame_start:frame_end] += w[:, 0]

    # ── 5. Hamming-weighted average ───────────────────────────────────────────
    covered = ps_count > 0
    if not np.any(covered):
        log.warning(
            "run_segmentation: no windows produced output — returning []")
        return ([], []) if collect_overlap_evidence else []

    ps_global = np.zeros((total_frames, num_ps_classes), dtype=np.float32)
    ps_global[covered] = (
        ps_sum[covered] / ps_count[covered, np.newaxis]
    ).astype(np.float32)

    # ── 6. Extract VAD turns ──────────────────────────────────────────────────
    turns = _extract_turns(
        ps_global, frame_rate, fetch_start_sec,
        threshold, min_speech_duration_sec, min_silence_duration_sec,
    )

    durations = [t["duration"] for t in turns]
    if durations:
        log.info(
            f"Turn durations: mean={np.mean(durations):.1f}s  "
            f"median={np.median(durations):.1f}s  "
            f"max={np.max(durations):.1f}s  "
            f">5s={sum(d>5 for d in durations)}  "
            f">10s={sum(d>10 for d in durations)}  "
            f">30s={sum(d>30 for d in durations)}"
        )
    log.info(
        f"Segmentation done: {len(turns)} turns found  "
        f"(threshold={threshold}, min_speech_duration_sec={min_speech_duration_sec}s)"
    )
    if collect_overlap_evidence:
        log.info(f"Overlap evidence: {len(evidence)} channel-resolved "
                 f"cross-talk records")
        return turns, evidence
    return turns


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _compute_window_starts(
    n_samples:      int,
    window_samples: int,
    step_samples:   int,
) -> list[int]:
    """
    Regular grid: 0, step, 2*step, …, up to n_samples - window_samples.
    If the audio is shorter than one window, returns [0].
    An extra window is added at max(0, n_samples - window_samples) to ensure
    the final samples are always covered.
    """
    if n_samples <= window_samples:
        return [0]
    last_valid = n_samples - window_samples
    starts = list(range(0, last_valid + 1, step_samples))
    if not starts or starts[-1] < last_valid:
        starts.append(last_valid)
    return starts


def _get_window(audio: np.ndarray, start: int, window_samples: int) -> np.ndarray:
    """Slice audio[start : start+window_samples], zero-padding if needed."""
    chunk = audio[start: start + window_samples]
    if len(chunk) < window_samples:
        chunk = np.pad(chunk, (0, window_samples - len(chunk)))
    return chunk


# Powerset class -> local speaker pair (community-1 ordering):
# 0 silence; 1,2,3 solo spk0/1/2; 4 spk0+1; 5 spk0+2; 6 spk1+2
_PAIR_OF_CLASS = {4: (0, 1), 5: (0, 2), 6: (1, 2)}
_SOLO_CLASS_OF_SPK = {0: 1, 1: 2, 2: 3}


def extract_overlap_evidence(
    window_probs:     np.ndarray,   # [F, 7] one window's probabilities
    window_start_sec: float,        # absolute time of frame 0
    frame_rate:       float,
    threshold:        float,
    min_solo_sec:     float,
    min_overlap_sec:  float,
) -> list[dict]:
    """
    Channel-resolved cross-talk evidence from ONE segmentation window.

    Powerset local-speaker indices are only consistent WITHIN a window
    (each window permutes freely), so this is deliberately per-window:
    when an overlap class (two speakers at once) is active AND both
    participants also have solo spans in the same window, those two
    solo spans provably belong to DIFFERENT people — regardless of how
    similar their embeddings are. Downstream turns each record into a
    cannot-link between the sentences overlapping the two solo spans
    (README.md Design record P3).

    Returns [{"a": [t0, t1], "b": [t0, t1], "overlap": [t0, t1]}, ...]
    (absolute seconds; longest solo span per participant).
    """
    # this function hardcodes the 7-class community-1 powerset layout
    # (classes 1-3 solo, 4-6 pairs); an export with a DIFFERENT class
    # count — fewer OR more (a >3-local-speaker export permutes the
    # pair-class positions) — yields NO evidence rather than an
    # IndexError or silently misread columns
    if window_probs.ndim != 2 or window_probs.shape[1] != 7:
        return []
    def _spans(mask):
        out, i = [], 0
        while i < len(mask):
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        return out

    def _t(frame):
        return round(window_start_sec + frame / frame_rate, 3)

    evidence = []
    for cls, (sa, sb) in _PAIR_OF_CLASS.items():
        ov_mask = window_probs[:, cls] > threshold
        ov_spans = [s for s in _spans(ov_mask)
                    if (s[1] - s[0]) / frame_rate >= min_overlap_sec]
        if not ov_spans:
            continue
        solo = {}
        for spk in (sa, sb):
            m = window_probs[:, _SOLO_CLASS_OF_SPK[spk]] > threshold
            spans = [s for s in _spans(m)
                     if (s[1] - s[0]) / frame_rate >= min_solo_sec]
            if spans:
                solo[spk] = max(spans, key=lambda s: s[1] - s[0])
        if len(solo) == 2:
            ov = max(ov_spans, key=lambda s: s[1] - s[0])
            a, b = solo[sa], solo[sb]
            evidence.append({
                "a":       [_t(a[0]), _t(a[1])],
                "b":       [_t(b[0]), _t(b[1])],
                "overlap": [_t(ov[0]), _t(ov[1])],
            })
    return evidence


def _to_probabilities(raw: np.ndarray) -> np.ndarray:
    """
    Convert model output to per-class probabilities.

    community-1 last layer is LogSoftmax → output is log-probabilities.
    softmax(log_probs) == exp(log_probs) since they already sum to 1.
    Using softmax is safe as a defensive implementation.
    """
    x = raw - raw.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _close_gaps(mask: np.ndarray, gap_frames: int) -> np.ndarray:
    """Fill gaps of silence shorter than gap_frames within speech regions."""
    result = mask.copy()
    i = 0
    while i < len(mask):
        if mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and not mask[j]:
            j += 1
        if j < len(mask) and (j - i) < gap_frames:
            result[i:j] = True
        i = j + 1
    return result


def _extract_turns(
    ps_probs:                 np.ndarray, # [total_frames, num_powerset_classes]
    frame_rate:               float,
    fetch_start_sec:          float,
    threshold:                float,
    min_speech_duration_sec:  float,
    min_silence_duration_sec: float,
) -> list[dict]:
    """
    Convert powerset probability matrix into a list of speech turns.

    Pure VAD — each turn records only timing and whether overlap occurred.

    Algorithm
    ─────────
    1. Speech activity  = 1 - P(silence_class=0) > threshold
    2. Close short gaps (< min_gap_sec) — community-1: min_gap=0, no-op
    3. Find contiguous speech runs
    4. Discard runs shorter than min_duration_sec
    5. Overlap flag = any frame where max(P(overlap classes 4,5,6)) > threshold
    """
    num_frames = ps_probs.shape[0]

    min_gap_frames = max(1, int(min_silence_duration_sec * frame_rate)
                         ) if min_silence_duration_sec > 0 else 0
    min_dur_frames = max(1, int(min_speech_duration_sec * frame_rate))

    # Speech activity: anything that isn't silence (class 0)
    speech = (1.0 - ps_probs[:, 0]) > threshold          # [F] bool
    if min_gap_frames > 0:
        speech = _close_gaps(speech, min_gap_frames)

    # Overlap: any of the 3 overlap-pair classes (indices 4, 5, 6) active
    # num_powerset_classes=7: classes 0-3 are silence/single, 4-6 are pairs
    overlap_mask = np.max(ps_probs[:, 4:], axis=1) > threshold  # [F] bool

    turns: list[dict] = []
    i = 0
    while i < num_frames:
        if not speech[i]:
            i += 1
            continue

        j = i + 1
        while j < num_frames and speech[j]:
            j += 1

        if (j - i) >= min_dur_frames:
            t_start = fetch_start_sec + i / frame_rate
            t_end = fetch_start_sec + j / frame_rate
            turns.append({
                "start":       round(t_start, 3),
                "end":         round(t_end,   3),
                "duration":    round(t_end - t_start, 3),
                "has_overlap": bool(np.any(overlap_mask[i:j])),
            })

        i = j

    return turns
