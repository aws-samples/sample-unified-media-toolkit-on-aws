# lib/language_detect.py
"""
Language detection using dynamic chunk probes.

Algorithm
─────────
  1. Caller selects up to DEFAULT_NUM_PROBES evenly-spaced dynamic chunks
     (≥ DEFAULT_MIN_PROBE_SEC speech) via _select_probe_chunks() in the handler.
  2. Byte-range GET each chunk's audio from S3 (capped at MAX_FEED_SEC).
  3. Run Whisper detect_language() on each (30 s window, encode only).
  4. Majority vote across probes → final language + mean confidence.

The Whisper model used here is always the turbo model (large-v3-turbo),
loaded temporarily with compute_type="int8" then discarded before
Whisper variant + alignment models are downloaded.
"""

from __future__ import annotations

import gc
import logging
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)

MIN_FEED_SEC = 5.0
MAX_FEED_SEC = 30.0    # Whisper's fixed context window

# =============================================================================
# AUDIO FETCH
# =============================================================================


def fetch_chunk_audio(
    bucket:   str,
    key:      str,
    wav_info,           # WavInfo from lib.wav_utils
    chunk:    dict,
    max_sec:  float = MAX_FEED_SEC,
) -> np.ndarray | None:
    """Byte-range fetch audio for one dynamic chunk probe (capped at max_sec).

    Accepts a dynamic chunk dict with 'fetch_start_sec' / 'fetch_end_sec'
    keys (StateDynamicChunkingBySpeechTurns output format).
    """
    from lib.wav_utils import read_wav_window

    start = float(chunk["fetch_start_sec"])
    end = min(float(chunk["fetch_end_sec"]), start + max_sec)
    duration = end - start

    if duration < MIN_FEED_SEC:
        log.debug(f"Chunk probe too short after cap ({duration:.1f}s) — skip")
        return None
    try:
        return read_wav_window(bucket, key, wav_info,
                               start_sec=start, duration_sec=duration)
    except Exception as exc:
        log.warning(f"fetch_chunk_audio [{start:.1f}s,{end:.1f}s]: {exc}")
        return None


# =============================================================================
# WHISPER DETECT_LANGUAGE
# =============================================================================

def _run_detect_language(
    audio: np.ndarray,
    model,
) -> tuple[str | None, float]:
    """
    Run Whisper detect_language() on a 30-second mono float32 window.
    Returns (lang_code, probability) or (None, 0.0) on failure.
    """
    try:
        audio = audio.astype(np.float32)
        target = 16_000 * 30
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]

        result = model.detect_language(audio)

        # faster-whisper has changed its return type across versions:
        #
        #   v0.x / v1.0  : List[Tuple[str, float]]
        #                  [(lang, prob), ...]  ordered best-first
        #
        #   v1.1+        : Tuple[str, float, List[Tuple[str, float]]]
        #                  (best_lang, best_prob, [(lang, prob), ...])
        #
        #   Some builds  : Dict[str, float]  {lang: prob}
        #
        # Detect which shape we have by inspecting result[0].

        if isinstance(result, dict):
            # {lang: prob} mapping
            best = max(result, key=result.get)
            return str(best), float(result[best])

        if isinstance(result, (list, tuple)) and result:
            first = result[0]
            if isinstance(first, str):
                # New-style: (best_lang, best_prob, all_probs)
                # result[0] is already the best language string
                lang = first
                prob = float(result[1]) if len(result) > 1 else 0.0
                return lang, prob
            if isinstance(first, (list, tuple)) and len(first) == 2:
                # Old-style: [(lang, prob), ...]
                lang, prob = first
                return str(lang), float(prob)

        return None, 0.0

    except Exception as exc:
        log.warning(f"detect_language error: {exc}")
        return None, 0.0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def detect_language(
    audio_bucket:       str,
    audio_key:          str,
    probes:             list[dict],
    whisper_model_path: str,
    asr_cfg:            dict,
    cpu_threads:        int | None = None,
) -> dict:
    """
    Detect the language of an audio file using Whisper majority vote.

    Parameters
    ──────────
    probes  : list of dynamic chunk dicts (StateDynamicChunkingBySpeechTurns
              output) already filtered and selected by the caller.  Each dict
              must have 'fetch_start_sec' and 'fetch_end_sec'.  Probe
              selection (evenly-spaced, speech-duration gate) is the caller's
              responsibility — see _select_probe_chunks() in the handler.

    num_probes / min_probe_sec are kept for informational logging only;
    actual filtering was done upstream.

    cpu_threads : number of CTranslate2 inter/intra threads.  Should be
                  derived from Lambda memory_limit_in_mb via get_num_cpus()
                  in the handler.  When None, CTranslate2 defaults to
                  os.cpu_count() which returns the host's physical core count
                  on Lambda — causing thread over-subscription on the actual
                  vCPU allocation.

    Returns
    ───────
    {
        "language":        "cy",
        "confidence":      0.94,
        "method":          "whisper_majority_vote",
        "num_probes_used": 5,
        "probe_results":   [{"start": 5234.6, "end": 5264.6,
                             "lang": "cy", "prob": 0.97}, ...],
    }
    """
    from faster_whisper import WhisperModel
    from lib.wav_utils import get_wav_info

    if not probes:
        log.warning("detect_language: no probes supplied — defaulting to 'en'")
        return _default_result()

    # ── Load Whisper temporarily (detect only, no decoding) ───────────────────
    log.info(
        f"Loading Whisper for language detection: {whisper_model_path}  "
        f"({len(probes)} probe(s))  cpu_threads={cpu_threads}"
    )
    model = WhisperModel(
        whisper_model_path,
        device="cpu",
        compute_type="int8",
        num_workers=1,
        cpu_threads=cpu_threads,
    )
    wav_info = get_wav_info(audio_bucket, audio_key)

    # ── Probe ─────────────────────────────────────────────────────────────────
    probe_results = []
    votes: Counter = Counter()

    for chunk in probes:
        audio = fetch_chunk_audio(audio_bucket, audio_key, wav_info, chunk)
        if audio is None:
            continue
        lang, prob = _run_detect_language(audio, model)
        if lang is None:
            continue
        start = float(chunk["fetch_start_sec"])
        end = float(chunk["fetch_end_sec"])
        probe_results.append({
            "start": round(start, 2),
            "end":   round(end,   2),
            "lang":  lang,
            "prob":  round(prob, 4),
        })
        votes[lang] += 1
        log.info(
            f"  probe [{start:.1f}s, {end:.1f}s] → {lang} ({prob:.3f})"
        )

    # ── Free model before Whisper variant loads its own instance ─────────────
    del model
    gc.collect()

    if not votes:
        log.warning("detect_language: all probes failed — defaulting to 'en'")
        return _default_result()

    # ── Majority vote ─────────────────────────────────────────────────────────
    best_lang = votes.most_common(1)[0][0]
    confidence = round(
        sum(r["prob"] for r in probe_results if r["lang"] == best_lang)
        / max(1, votes[best_lang]),
        4,
    )

    log.info(
        f"Language detected: {best_lang}  confidence={confidence:.3f}  "
        f"votes={dict(votes)}"
    )
    return {
        "language":        best_lang,
        "confidence":      confidence,
        "method":          "whisper_majority_vote",
        "num_probes_used": len(probe_results),
        "probe_results":   probe_results,
    }


def _default_result() -> dict:
    return {
        "language":        "en",
        "confidence":      0.0,
        "method":          "default_fallback",
        "num_probes_used": 0,
        "probe_results":   [],
    }
