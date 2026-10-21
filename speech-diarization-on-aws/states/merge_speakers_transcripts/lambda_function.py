# merge_speakers_transcripts.py
"""
StateMergeSpeakersTranscripts — arm64 Docker Lambda.

Responsibilities
────────────────
  • Load all transcript_N.json files (absorbs old merge_transcripts state)
  • Load global_speakers.json
  • Assign speaker to each sentence via sentence_speakers lookup
  • Build final_transcript.json (sentences + words arrays)
  • Build final_transcript.vtt (Netflix Timed Text Style Guide)

Speaker assignment
──────────────────
  Primary : direct lookup in sentence_speakers["{dyn_chunk_idx}:{sentence_idx}"]
  Fallback: diarization overlap matching (for any edge case where the key
            is missing — should not happen in normal operation)

Cross-talk (overlap_spans, added 2026-08-26)
────────────────────────────────────────────
  The VAD's powerset model flags every speech turn where ≥2 speakers were
  simultaneously active (segmentation.py classes 4/5/6). Until now that
  flag died in the internal segments files — computed, persisted, consumed
  by nothing. final_transcript.json now carries a top-level
  `overlap_spans: [[start, end], …]` built from the CONSOLIDATED turn list
  in dynamic_segments_{n}.json (boundary-merged, absolute times, no
  double-counting from the 30 s fixed-chunk overlaps). TURN-granular, not
  frame-exact: a span is a whole speech turn in which cross-talk occurred
  for at least one frame. Consumers: ad-break `dialogue_bridged`
  hardening, CA's interaction judgment, per-sentence flags by time-join.
  AUXILIARY by design — a missing dynamic-segments doc degrades to the
  spans of the docs that did load (logged), never a failed merge.

WebVTT (Netflix Timed Text Style Guide)
───────────────────────────────────────
  • Max 42 chars per line, max 2 lines per cue
  • Max 7 s cue duration
  • Min 83 ms gap between cues
  • Cues are split at word boundaries by BOTH duration and text length —
    no truncation with "…" ever occurs
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
import unicodedata
from typing import Any
from lib.input_schema import validate_and_normalize
from lib.s3_utils import s3 as _s3, get_json

log = logging.getLogger(__name__)
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ── Netflix Timed Text Style Guide ────────────────────────────────────────────
_VTT_MAX_CHARS_PER_LINE = 42
_VTT_MAX_CUE_CHARS = _VTT_MAX_CHARS_PER_LINE * 2   # 84
_VTT_MAX_CUE_SEC = 7.0
_VTT_MIN_CUE_SEC = 1.5
_VTT_MIN_GAP_SEC = 0.083   # 2 frames @ 25 fps

# ── Unicode punctuation ───────────────────────────────────────────────────────
_BREAK_AFTER_CHARS = frozenset({
    ",", ";", ":", "—", "–",
    "、", "，", "；", "：",
})
_PUNCT_CATEGORIES = frozenset({"Po", "Pd", "Pe"})


# ─────────────────────────────────────────────────────────────────────────────
# S3 HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_s3_json(bucket: str, key: str) -> dict:
    return get_json(bucket, key)


def _put_s3(bucket: str, key: str, body: str | bytes,
            content_type: str) -> None:
    if isinstance(body, str):
        body = body.encode("utf-8")
    _s3.put_object(Bucket=bucket, Key=key, Body=body,
                          ContentType=content_type)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPT LOADING + SPEAKER ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def _overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    return max(0.0, min(e1, e2) - max(s1, s2))


def _find_speaker(start: float, end: float,
                  diarization: list[dict]) -> str | None:
    """Fallback: find speaker with max diarization overlap."""
    best_spk:  str | None = None
    best_ov:   float = 0.0
    for seg in diarization:
        ov = _overlap(start, end, seg["start"], seg["end"])
        if ov > best_ov:
            best_ov = ov
            best_spk = seg["speaker"]
    return best_spk if best_ov > 0.0 else None


def _collect_overlap_spans(
    bucket:    str,
    iterators: list[dict],
    load=_load_s3_json,
) -> list[list[float]]:
    """Cross-talk spans for final_transcript.json (module docstring).
    Merges touching/overlapping flagged turns; sorted, rounded to ms."""
    raw: list[list[float]] = []
    for item in sorted(iterators, key=lambda x: int(x["dyn_chunk_idx"])):
        key = item.get("dyn_segment_key")
        if not key:
            continue
        try:
            doc = load(bucket, key)
        except Exception as exc:                 # noqa: BLE001 — auxiliary
            log.warning("overlap_spans: could not read s3://%s/%s (%s) — "
                        "spans from this chunk are omitted",
                        bucket, key, exc)
            continue
        for t in doc.get("turns", []):
            if t.get("has_overlap"):
                raw.append([float(t["start"]), float(t["end"])])
    raw.sort()
    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [[round(s0, 3), round(e0, 3)] for s0, e0 in merged]


# Word-tight speech spans: split a sentence's time span where the
# inter-word gap exceeds this (same 0.5 s "contiguous speech" semantic
# as the embedding state's internal gap). Measured 2026-09-16 (DER
# harness): sentence spans bridge intra-sentence pauses and account
# for the 2-13% false-alarm term.
SPEECH_SPAN_GAP_SEC = 0.5


def _speech_spans(words, start, end, gap):
    """Word-tight speech spans inside one sentence.

    gap=None means the run had NO usable forced alignment
    (word_timing_source="whisper_native"): Whisper-native word
    timestamps are too coarse to trim against, so the whole sentence
    span is returned unsplit — the multilingual guard (a language
    without an alignment model must not get its speech cut by
    misplaced word boundaries).
    """
    if gap is None or not words:
        return [[round(float(start), 3), round(float(end), 3)]]
    spans = []
    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        if spans and ws - spans[-1][1] <= gap:
            spans[-1][1] = max(spans[-1][1], we)
        else:
            spans.append([ws, we])
    return [[round(a, 3), round(b, 3)] for a, b in spans]


def _load_and_label_sentences(
    bucket:           str,
    iterators:        list[dict],
    sentence_speakers: dict[str, str],
    diarization:      list[dict],
    span_gap:         float | None = SPEECH_SPAN_GAP_SEC,
) -> tuple[list[dict], list[dict], dict]:
    """
    Load all transcript_N.json files, concatenate sentences with global IDs,
    and assign speaker labels.

    Returns
    ───────
    out_sentences : list of sentence dicts with id, speaker_id, word_id_range,
                    speech_spans (word-tight; see _speech_spans)
    out_words     : flat list of word dicts with global id
    chunk_metrics : per-chunk stats
    """
    ordered = sorted(iterators, key=lambda x: int(x["dyn_chunk_idx"]))

    out_sentences: list[dict] = []
    out_words:     list[dict] = []
    chunk_metrics: dict[str, dict] = {}
    sentence_id = 0
    word_id = 0
    n_primary = n_fallback = n_unknown = 0

    for item in ordered:
        idx = int(item["dyn_chunk_idx"])
        key = item.get("transcript_key")
        if not key:
            log.warning("  chunk %d: transcript_key is null — skipping", idx)
            chunk_metrics[str(idx)] = {"num_sentences": 0, "skipped": True}
            continue

        doc = _load_s3_json(bucket, key)
        chunk_sents = doc.get("sentences", [])

        for local_idx, sent in enumerate(chunk_sents):
            # ── Speaker assignment ────────────────────────────────────────────
            spk_key = f"{idx}:{local_idx}"
            speaker = sentence_speakers.get(spk_key)

            if speaker:
                n_primary += 1
            else:
                speaker = _find_speaker(
                    float(sent["start"]), float(sent["end"]), diarization)
                if speaker:
                    n_fallback += 1
                else:
                    n_unknown += 1

            # ── Emit words ────────────────────────────────────────────────────
            w_start_id = word_id
            for w in sent.get("words", []):
                wrow = {
                    "id":          word_id,
                    "word":        w["word"],
                    "start":       float(w["start"]),
                    "end":         float(w["end"]),
                    "probability": w.get("probability"),
                }
                if w.get("pad_resolution"):
                    # provenance: this word was relocated/kept by the
                    # pad-boundary ladder (see lib/transcribe.py)
                    wrow["pad_resolution"] = w["pad_resolution"]
                out_words.append(wrow)
                word_id += 1
            w_end_id = word_id - 1

            out_sentences.append({
                "id":            sentence_id,
                "text":          sent.get("text", ""),
                "start":         float(sent["start"]),
                "end":           float(sent["end"]),
                "confidence":    sent.get("confidence"),
                "speaker_id":    speaker,
                "word_id_range": [w_start_id, w_end_id],
                "speech_spans":  _speech_spans(
                    sent.get("words") or [], sent["start"], sent["end"],
                    span_gap),
            })
            sentence_id += 1

        chunk_metrics[str(idx)] = {
            "num_sentences": len(chunk_sents),
            "fetch_sec": round(
                float(item["fetch_end_sec"]) - float(item["fetch_start_sec"]),
                3),
            "elapsed_sec": float(item.get("elapsed_sec", 0.0)),
        }

    log.info(
        "Speaker assignment: %d primary  %d fallback  %d unknown  "
        "→ %d sentences  %d words",
        n_primary, n_fallback, n_unknown,
        len(out_sentences), len(out_words),
    )
    return out_sentences, out_words, chunk_metrics


# ─────────────────────────────────────────────────────────────────────────────
# WEBVTT
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    total_ms = round(seconds * 1_000)
    h, total_ms = divmod(total_ms, 3_600_000)
    m, total_ms = divmod(total_ms,    60_000)
    s, ms = divmod(total_ms,     1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _is_punct_break(word: str) -> bool:
    if not word:
        return False
    last = word[-1]
    return (last in _BREAK_AFTER_CHARS
            or unicodedata.category(last) in _PUNCT_CATEGORIES)


def _wrap_text(text: str,
               max_chars: int = _VTT_MAX_CHARS_PER_LINE) -> str:
    """
    Wrap cue text to at most 2 lines per Netflix Timed Text Style Guide.

    Callers guarantee len(text) ≤ 2 * max_chars, so truncation never occurs.
    """
    if len(text) <= max_chars:
        return text

    words = text.split()
    if len(words) == 1:
        return text   # single long word — cannot wrap

    candidates: list[tuple[int, int, int]] = []
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        if len(l1) > max_chars or len(l2) > max_chars:
            continue
        score = 0
        if _is_punct_break(words[i - 1]):
            score += 20
        if len(l2) >= len(l1):
            score += 10
        diff = abs(len(l1) - len(l2))
        candidates.append((score, -diff, i))

    if candidates:
        candidates.sort(reverse=True)
        _, _, best_i = candidates[0]
        return f"{' '.join(words[:best_i])}\n{' '.join(words[best_i:])}"

    # Fallback — force break at max_chars (no truncation)
    line1 = text[:max_chars]
    sp = line1.rfind(" ")
    if sp > 0:
        line1 = line1[:sp]
    line2 = text[len(line1):].strip()
    return f"{line1}\n{line2}"


def _split_cue(
    sentence:   dict,
    sent_words: list[dict],
    max_sec:    float = _VTT_MAX_CUE_SEC,
    max_chars:  int = _VTT_MAX_CUE_CHARS,
) -> list[dict]:
    """
    Split a sentence into cues that satisfy both duration and text length
    constraints.  Splits at word boundaries — never truncates.

    Each cue's text is guaranteed to be ≤ max_chars, so _wrap_text never
    needs to truncate.
    """
    if not sent_words:
        return [{
            "text":       sentence["text"],
            "start":      sentence["start"],
            "end":        sentence["end"],
            "speaker_id": sentence["speaker_id"],
        }]

    cues:      list[dict] = []
    cue_words: list[dict] = []
    cue_start: float = float(sent_words[0]["start"])

    for w in sent_words:
        test_text = " ".join(x["word"] for x in cue_words + [w]).strip()
        test_dur = float(w["end"]) - cue_start

        if cue_words and (test_dur > max_sec or len(test_text) > max_chars):
            cues.append({
                "text":       " ".join(
                    x["word"] for x in cue_words).strip(),
                "start":      cue_start,
                "end":        float(cue_words[-1]["end"]),
                "speaker_id": sentence["speaker_id"],
            })
            cue_start = float(w["start"])
            cue_words = [w]
        else:
            cue_words.append(w)

    if cue_words:
        cues.append({
            "text":       " ".join(x["word"] for x in cue_words).strip(),
            "start":      cue_start,
            "end":        float(sentence["end"]),
            "speaker_id": sentence["speaker_id"],
        })

    return cues


def _enforce_min_gap(cues: list[dict],
                     min_gap: float = _VTT_MIN_GAP_SEC) -> list[dict]:
    for i in range(len(cues) - 1):
        gap = cues[i + 1]["start"] - cues[i]["end"]
        if gap < min_gap:
            trimmed = cues[i + 1]["start"] - min_gap
            floor = cues[i]["start"] + _VTT_MIN_CUE_SEC
            cues[i]["end"] = max(floor, trimmed)
    return cues


def _build_vtt(sentences: list[dict], words: list[dict]) -> str:
    """Build full WebVTT string."""
    word_by_id: dict[int, dict] = {w["id"]: w for w in words}

    raw_cues: list[dict] = []
    for sent in sentences:
        lo, hi = sent["word_id_range"]
        sent_words = [word_by_id[i]
                      for i in range(lo, hi + 1) if i in word_by_id]
        raw_cues.extend(_split_cue(sent, sent_words))

    _enforce_min_gap(raw_cues)

    lines = ["WEBVTT", ""]
    for cue in raw_cues:
        speaker = cue["speaker_id"] or "UNKNOWN"
        text = _wrap_text(cue["text"])
        lines.append(f"{_fmt_ts(cue['start'])} --> {_fmt_ts(cue['end'])}")
        lines.append(f"<v {speaker}>{text}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    log.info("StateMergeSpeakersTranscripts event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── unpack event ──
        data = event["data"]

        output_bucket = data["output_bucket"]
        output_prefix = data["output_prefix"]

        global_speakers = data["global_speakers"]
        global_speakers_key = global_speakers["global_speakers_key"]
        num_global_speakers = int(global_speakers["num_global_speakers"])

        language = data["language"]
        language_confidence = data["language_confidence"]

        asr_cfg = data["asr_cfg"]
        model_variant = asr_cfg["model_variant"]

        audio_cfg = data["audio_cfg"]
        duration_sec = audio_cfg["duration_sec"]

        iterators = data["iterators"]

        log.info(
            "StateMergeSpeakersTranscripts START | "
            "chunks=%d  num_global_speakers=%d  language=%s",
            len(iterators), num_global_speakers, language,
        )

        # ── Load speaker assignments ──────────────────────────────────────────
        log.info("Loading speakers  s3://%s/%s",
                 output_bucket, global_speakers_key)
        speakers_doc = _load_s3_json(output_bucket, global_speakers_key)
        sentence_speakers = speakers_doc.get("sentence_speakers", {})
        diarization = speakers_doc.get("diarization", [])
        overlap_segments = speakers_doc.get("overlap_segments", [])
        log.info("  sentence_speakers=%d entries  diarization=%d segments",
                 len(sentence_speakers), len(diarization))

        # ── Load transcripts + assign speakers ────────────────────────────────
        # multilingual guard: only split sentence spans at word gaps when
        # forced alignment actually ran (see _speech_spans)
        word_timing_source = (data.get("word_timing_source")
                              or "whisper_native")
        span_gap = (SPEECH_SPAN_GAP_SEC
                    if str(word_timing_source).startswith("aligned")
                    else None)
        out_sentences, out_words, chunk_metrics = _load_and_label_sentences(
            output_bucket, iterators, sentence_speakers, diarization,
            span_gap=span_gap,
        )

        # ── Cross-talk spans (module docstring: turn-granular, auxiliary) ─────
        overlap_spans = _collect_overlap_spans(output_bucket, iterators)
        log.info("Cross-talk: %d overlap span(s)", len(overlap_spans))

        if not out_sentences:
            raise RuntimeError(
                "No sentences produced — check transcript files")

        # ── Output S3 paths ───────────────────────────────────────────────────
        final_json_key = f"{output_prefix}/transcripts/final_transcript.json"
        final_vtt_key = f"{output_prefix}/transcripts/final_transcript.vtt"

        # ── Write final_transcript.json ───────────────────────────────────────
        final_doc = {
            "language":            language,
            "language_confidence": language_confidence,
            "word_timing_source":  word_timing_source,
            "model_variant":       model_variant,
            "audio_duration_sec":  duration_sec,
            "num_global_speakers": num_global_speakers,
            "num_sentences":       len(out_sentences),
            "num_overlap_spans":   len(overlap_spans),
            "overlap_spans":       overlap_spans,
            # P3 (2026-09-16): identity-resolved cross-talk — each entry
            # names BOTH proven speakers of an overlap span; the second
            # voice is otherwise structurally absent from sentences[]
            "overlap_segments":    overlap_segments,
            "sentences":           out_sentences,
            "words":               out_words,
        }
        _put_s3(
            output_bucket, final_json_key,
            json.dumps(final_doc, separators=(",", ":"), ensure_ascii=False),
            "application/json",
        )
        log.info("Written → s3://%s/%s  sentences=%d  words=%d",
                 output_bucket, final_json_key,
                 len(out_sentences), len(out_words))

        # ── Write final_transcript.vtt ────────────────────────────────────────
        vtt_str = _build_vtt(out_sentences, out_words)
        _put_s3(output_bucket, final_vtt_key,
                vtt_str, "text/vtt; charset=utf-8")
        log.info("Written → s3://%s/%s", output_bucket, final_vtt_key)

        # ── Return ────────────────────────────────────────────────────────────
        output = {
            **event,
            "data": {
                **data,
                "final_transcript_key": final_json_key,
                "final_transcript_vtt": final_vtt_key,
                "num_global_speakers":  num_global_speakers,
                "num_sentences":        len(out_sentences),
                "iterators":            None,
            },
        }

        log.info("StateMergeSpeakersTranscripts output keys = %s",
                 sorted(output.keys()))
        return output

    except Exception as exc:
        log.error("[ERR] StateMergeSpeakersTranscripts — %s: %s",
                  type(exc).__name__, exc)
        traceback.print_exc()
        raise
