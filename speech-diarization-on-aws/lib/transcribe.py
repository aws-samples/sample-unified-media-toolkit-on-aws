# lib/transcribe.py
"""
Transcription worker logic.

Architecture
────────────
  Speech groups are the atomic unit for Whisper inference and word alignment.
  Sentence splitting runs ONCE per iterator (after all batches complete),
  giving wtpsplit full visibility across group boundaries.

  batch_worker() runs Whisper + alignment and returns timestamped words.
  split_words_into_sentences() runs wtpsplit on the full word sequence and
  returns finished sentences.

Threading model
───────────────
  ThreadPoolExecutor — one thread per concurrent slot (asr_cfg.num_workers).
  Lambda's sandbox has no /dev/shm, so ProcessPoolExecutor fails at
  construction.  ThreadPoolExecutor uses only pthread mutexes, which are
  always available.

  Thread safety:
    • worker_init() is called ONCE in the main thread before the pool starts,
      loading all model globals into the module's namespace.
    • After init all globals are read-only; each thread reads them concurrently.
    • CTranslate2 (faster-whisper) and onnxruntime both release the GIL during
      C++ inference, so N threads give genuine N-core parallelism.
    • cpu_threads=1 in WhisperModel is intentional: with N concurrent threads
      each requesting 1 OS thread from CTranslate2, the N cores are used
      without contention.

Kwargs passthrough
──────────────────
  faster-whisper WhisperModel.transcribe() accepts many keyword args.
  Any key in event["whisper_kwargs"] is forwarded verbatim, except for
  a small set of keys that the pipeline controls itself (audio, language,
  word_timestamps, vad_filter).

Alignment
─────────
  Word timestamps from Whisper are refined by a quantized wav2vec2 ONNX model
  for words below the confidence threshold.  The alignment model is
  language-specific; if unavailable the Whisper timestamps are used as-is
  (still accurate to ~50 ms).

Inter-group padding and word assignment
───────────────────────────────────────
  INTER_GROUP_PAD_SEC (500 ms) is inserted between groups in the combined
  batch audio.  Inter-group gaps in the original audio are always ≥ 1.0s
  (the speech group formation threshold), so 500ms of synthetic silence is
  a conservative signal — Whisper reliably creates segment boundaries at
  gaps ≥ 300ms.

  Word assignment uses soft boundaries: each group owns audio from its
  local_offset_sec up to the start of the next group.  A nearest-group
  fallback handles the rare edge case of words outside all soft windows.

Sentence splitting
──────────────────
  Sentence splitting (wtpsplit) runs ONCE on the full word sequence after
  all batch_worker calls complete, via split_words_into_sentences().

  This gives wtpsplit full context across group boundaries, fixing the
  trailing-fragment problem where words like "Pride" at the end of one
  group were incorrectly merged backward into the previous sentence
  ("...answers. Pride") instead of forward into the next ("Pride before
  a fall.").

  wtpsplit internally uses a 512-token sliding window with stride 64,
  so long texts are handled efficiently without any external windowing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
import onnxruntime as ort

from lib.sentence_split import SentenceAssembler, load_wtpsplit

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Keys that the pipeline always controls — never overridden by whisper_kwargs
_RESERVED_KWARGS = {"audio", "language", "word_timestamps", "vad_filter"}
_OUTPUT_WORD_FIELDS = frozenset({"word", "start", "end", "probability",
                                 "pad_resolution"})

# 500 ms synthetic silence between groups in a Whisper batch.
# Inter-group gaps are ≥ 1.0s in reality; 500ms is a clear signal for Whisper
# to segment without wasting batch budget.
INTER_GROUP_PAD_SEC = 0.5

# Pack groups until this many seconds of wall-clock audio per batch.
# Whisper's encoder processes 30s windows internally; keeping batches ≤ 28s
# means most batches fit in a single encoder pass.  Long monologue groups
# (> 28s) are their own batch — faster-whisper handles sequential decoding.
BATCH_TARGET_SEC = 28.0


# =============================================================================
# VOCAB LOADER
# =============================================================================

def load_vocab(vocab_path: str) -> dict[int, str]:
    """
    Load vocab.json → {token_index: token_str}.

    Handles three formats:
      flat dict  {"A": 4, "B": 5}
      nested     {"vocab": {"A": 4, ...}}
      list       ["<pad>", "<s>", "A", ...]
    """
    if not vocab_path or not os.path.exists(vocab_path):
        return {}
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        log.warning(f"Could not load vocab {vocab_path}: {exc}")
        return {}

    if isinstance(raw, list):
        return {i: tok for i, tok in enumerate(raw)}
    if isinstance(raw, dict):
        vocab = raw.get("vocab", raw)
        return {
            int(idx): tok
            for tok, idx in vocab.items()
            if isinstance(idx, (int, float))
        }
    return {}


# =============================================================================
# WORKER GLOBALS
# =============================================================================

_whisper_model = None
_align_sess = None
_align_available = False
_inv_vocab:      dict[int, str] = {}
_vocab_chars:    set | None = None     # lazy; reset when a vocab is loaded
_language = "en"
_whisper_kwargs: dict[str, Any] = {}
_sample_rate = 16_000
_align_conf_thresh = 0.8
_word_align_padding = 0.1
_max_word_window = 2.0

# Cache key for the loaded models — survives across warm invocations.
# Encodes (whisper_model_path, align_model_path, threads_per_worker).
# If the key matches at worker_init time, the expensive model instantiation
# is skipped and only the cheap per-invocation config is updated.
_worker_init_key: str | None = None


def worker_init(
    whisper_model_path: str,
    align_model_path:   str,         # "" = no alignment
    vocab_path:         str,         # "" = no vocab
    wtpsplit_model_dir: str,
    language:           str,
    whisper_kwargs:     dict[str, Any],
    sample_rate:        int = 16_000,
    align_conf_thresh:  float = 0.8,
    word_align_padding: float = 0.1,
    max_word_window:    float = 2.0,
    num_workers:        int = 1,
) -> None:
    """
    Initialise all worker-level globals.
    Called once in the main thread before the ThreadPoolExecutor starts.

    Loads Whisper, alignment ONNX, and wtpsplit models.  wtpsplit is used
    later by split_words_into_sentences() (called in the main thread after
    all batches complete).

    On warm containers, if the model key matches the previously loaded models,
    only the cheap per-invocation config (_language, _whisper_kwargs, etc.) is
    updated — the expensive WhisperModel and ORT session instantiation is skipped.

    num_workers is the ThreadPoolExecutor size.  threads_per_worker is derived
    as max(1, total_cores // num_workers) and applied to all three inference
    engines so each concurrent thread gets an equal share of the vCPU budget.
    """
    global _whisper_model, _align_sess, _align_available
    global _inv_vocab, _vocab_chars, _language, _whisper_kwargs
    global _sample_rate, _align_conf_thresh, _word_align_padding, _max_word_window
    global _worker_init_key

    from faster_whisper import WhisperModel

    pid = os.getpid()
    total_cores = os.cpu_count() or 1
    threads_per_worker = max(1, total_cores // num_workers)

    # ── Per-invocation config (always updated) ────────────────────────────────
    _language = language.lower()
    _whisper_kwargs = {k: v for k, v in whisper_kwargs.items()
                       if k not in _RESERVED_KWARGS}
    _sample_rate = sample_rate
    _align_conf_thresh = align_conf_thresh
    _word_align_padding = word_align_padding
    _max_word_window = max_word_window

    new_key = f"{whisper_model_path}|{align_model_path}|{threads_per_worker}"

    log.info(
        f"[PID {pid}] worker_init: lang={_language}  "
        f"align={'yes' if align_model_path else 'no'}  "
        f"num_workers={num_workers}  total_cores={total_cores}  "
        f"threads_per_worker={threads_per_worker}  "
        f"kwargs={list(_whisper_kwargs.keys())}"
    )

    if _worker_init_key == new_key:
        log.info(f"[PID {pid}] Worker ready ✅  (warm — models reused)")
        return

    # ── Whisper ───────────────────────────────────────────────────────────────
    t_w = time.monotonic()
    _whisper_model = WhisperModel(
        whisper_model_path,
        device="cpu",
        compute_type="int8",
        cpu_threads=threads_per_worker,
        num_workers=1,
    )
    whisper_load_sec = round(time.monotonic() - t_w, 2)
    log.info(f"[PID {pid}] Whisper model loaded  {whisper_load_sec}s")

    # ── Alignment ONNX ────────────────────────────────────────────────────────
    align_load_sec = 0.0
    _align_available = bool(
        align_model_path and os.path.exists(align_model_path)
    )
    if _align_available:
        t_a = time.monotonic()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads_per_worker
        _align_sess = ort.InferenceSession(
            align_model_path, opts,
            providers=["CPUExecutionProvider"],
        )
        _inv_vocab = load_vocab(vocab_path)
        _vocab_chars = None            # recompute for the new vocab
        align_load_sec = round(time.monotonic() - t_a, 2)
        log.info(
            f"[PID {pid}] Align model loaded  vocab={len(_inv_vocab)} tokens  "
            f"intra_op_threads={threads_per_worker}  {align_load_sec}s"
        )
    else:
        log.info(f"[PID {pid}] No alignment model — using Whisper timestamps")

    # ── wtpsplit ──────────────────────────────────────────────────────────────
    # Loaded here so that split_words_into_sentences() can use it later
    # in the main thread after all batches complete.
    t_wtp = time.monotonic()
    load_wtpsplit(wtpsplit_model_dir, intra_op_num_threads=threads_per_worker)
    wtpsplit_load_sec = round(time.monotonic() - t_wtp, 2)
    log.info(f"[PID {pid}] wtpsplit model loaded  {wtpsplit_load_sec}s")

    _worker_init_key = new_key
    log.info(
        f"[PID {pid}] Worker ready ✅  "
        f"whisper={whisper_load_sec}s  "
        f"align={align_load_sec if _align_available else 0.0}s  "
        f"wtpsplit={wtpsplit_load_sec}s"
    )


# =============================================================================
# ALIGNMENT — single word refinement
# =============================================================================

def _ctc_greedy_chars(segment: np.ndarray) -> list[dict] | None:
    """Run the align model on an audio segment and greedy-decode to
    [{"char", "t"}] with t relative to the segment start (0.02 s frames).
    None on model failure. Shared by _refine_word and _pad_word_support."""
    try:
        normed = (segment - np.mean(segment)) / (np.std(segment) + 1e-7)
        emissions = _align_sess.run(
            None,
            {_align_sess.get_inputs()[0].name:
             normed[np.newaxis, :].astype(np.float32)},
        )[0]                                    # [1, T, vocab]
    except Exception:
        return None
    indices = np.argmax(emissions[0], axis=-1)
    chars = []
    prev_idx = None
    for i, idx in enumerate(indices):
        if idx == prev_idx or idx <= 3:
            prev_idx = idx
            continue
        char = _inv_vocab.get(int(idx), "")
        if char:
            chars.append({"char": char.lower(), "t": i * 0.02})
        prev_idx = idx
    return chars


def _target_charset(word: str) -> set:
    """Target character set for CTC coverage/matching: the word's characters
    the loaded align model can actually EMIT. Language-agnostic — the only
    legitimate filter is the model's own vocab (digits/punctuation fall out
    for every script because no wav2vec vocab emits them). Empty set = the
    probe cannot judge this word."""
    global _vocab_chars
    if _vocab_chars is None:
        _vocab_chars = {c.lower() for c in _inv_vocab.values() if c}
    return {c.lower() for c in word if not c.isspace()} & _vocab_chars


def _refine_word(
    word:    str,
    w_start: float,
    w_end:   float,
    audio:   np.ndarray,
) -> tuple[float, float]:
    """
    Refine word timestamps using the wav2vec2 align model.
    Returns original timestamps if alignment is unreliable.

    w_start and w_end are relative to the start of `audio`.
    """
    if _align_sess is None:
        return w_start, w_end

    pad_start = max(0.0, w_start - _word_align_padding)
    pad_end = w_end + _word_align_padding
    s = int(pad_start * _sample_rate)
    e = int(pad_end * _sample_rate)
    segment = audio[s:e]

    min_s = int(0.1 * _sample_rate)
    max_s = int(_max_word_window * _sample_rate)
    if len(segment) < min_s or len(segment) > max_s:
        return w_start, w_end

    chars = _ctc_greedy_chars(segment)
    if not chars:
        return w_start, w_end

    target = _target_charset(word)
    matching = [{"char": c["char"], "t": pad_start + c["t"]}
                for c in chars if c["char"] in target]
    if not matching:
        return w_start, w_end

    r_start = matching[0]["t"]
    r_end = matching[-1]["t"] + 0.02

    # Sanity guard — reject large deviations
    if abs(r_start - w_start) > 0.15:
        r_start = w_start
    if abs(r_end - w_end) > 0.15:
        r_end = w_end
    if r_start >= r_end:
        # the two guards act independently and can cross on sub-0.15 s
        # slots — an inverted interval is worse than no refinement
        return w_start, w_end

    return round(r_start, 3), round(r_end, 3)


# =============================================================================
# PAD-BOUNDARY WORD RESOLUTION (2026-10-12)
# =============================================================================
# Groups 50+ s apart in real time sit INTER_GROUP_PAD_SEC apart in the
# combined audio, so a sub-second Whisper word-timestamp slop at a group
# boundary used to map a word tens of seconds off (measured: a 47 ms slop
# became a 52 s sentence-start error). The pads are synthetic zeros — a word
# whose span leaves every group's real audio is provably mislocated. The
# resolution ladder (each layer fires only where the previous abstained):
#   1. detection — exact construction geometry, no thresholds
#   2. interval feasibility — reject sides where the word cannot fit between
#      its would-be neighbours even after duration shrink (an EMPTY side
#      always fits: feasibility is a rejector, never a confirmer)
#   3. continuity vote — Whisper EMISSION ORDER (same-segment neighbours),
#      gated: the span must be within PAD_PROXIMITY_GATE_SEC of (or straddle)
#      the absorbing side's audio; gate failure escalates, never convicts
#   4. BOTH sides packed → DROP: the slot is already owned, which is the
#      boundary-DUPLICATE signature (Whisper re-emits a word across the
#      pad; the probe saturates on the twin's audio and cannot help —
#      live-measured 2026-10-12: probing here readmitted ~700 insertions
#      vs ~20 recovered deletions). Otherwise: aligner support probe (CTC
#      char coverage) on both raw sides; drop when every side was judged
#      and none supports the word (pad hallucinations must not be
#      laundered by relocation); a probe that cannot judge (no emittable
#      charset, degenerate window, model failure) or cannot separate the
#      sides (lead < margin: short words saturate coverage) abstains →
#      keep + flag. With no aligner (multilingual degraded mode) nothing
#      is EVER dropped: keep + flag.
#   5. sentence-mates override (rehome_straddle_strays, after wtpsplit) —
#      a FLAGGED word stranded in a group holding no clean word of its own
#      sentence is re-homed to the sentence's clean-majority group; clean
#      words are never second-guessed, so honest cross-group sentences stay.

PAD_PROXIMITY_GATE_SEC = 0.1   # continuity vote only this close to a boundary
PAD_MIN_WORD_SEC = 0.1         # smallest slot a relocated word may shrink into
PAD_SUPPORT_FLOOR = 0.5        # CTC char coverage below this on BOTH sides = no support
PAD_PROBE_WINDOW_SEC = 2.0     # audio window probed per candidate side
# Required lead of the best side over the runner-up before the probe may
# adjudicate: char coverage saturates on short words (both sides of normal
# speech decode {t,o} for "to"), so a near-tie is non-evidence. A-priori
# value, not fitted — any tie-break below it falls back to keep + flag.
PAD_SUPPORT_MARGIN = 0.25


def _pad_word_support(word: str, audio: np.ndarray,
                      lo_sec: float, hi_sec: float) -> float | None:
    """CTC support score for `word` inside audio[lo:hi]: the fraction of the
    word's character set recovered by greedy decode over the window.

    Returns None when the probe CANNOT JUDGE (no aligner, degenerate window,
    model failure, or a word with no alphabetic target such as "911.") —
    abstention, never evidence. 0.0 means the probe ran and found nothing."""
    target = _target_charset(word)
    if _align_sess is None or not target:
        return None
    s = max(0, int(lo_sec * _sample_rate))
    e = min(len(audio), int(hi_sec * _sample_rate))
    segment = audio[s:e]
    if len(segment) < int(PAD_MIN_WORD_SEC * _sample_rate):
        return None
    chars = _ctc_greedy_chars(segment)
    if chars is None:
        return None
    found = {c["char"] for c in chars}
    return len(found & target) / len(target)


def _resolve_word_groups(words: list[dict], groups: list[dict],
                         support_fn=None) -> list[dict]:
    """Assign emission-ordered words to groups; resolve pad-boundary words.

    words : [{t_s, t_e, seg_id}] combined-local times, emission order
    groups: [{off, end, win_hi}] in batch order — off/end bound the REAL
            audio; win_hi is the legacy soft-window end (next group's off,
            i.e. the pad belongs to the left group's window)
    support_fn(word_idx, group_pos, lo_sec, hi_sec) -> float | None (None =
            the probe cannot judge this word/window — abstention), or
            support_fn itself None in the degraded (no-aligner) mode

    Returns one dict per word: {gpos, rel_s, rel_e, flag} where gpos is the
    batch position (None = dropped) and rel times are group-relative. flag ∈
    {None, "pad_moved", "pad_unresolved", "pad_dropped"}.
    """
    n = len(groups)

    def owner_by_window(t):
        for i, g in enumerate(groups):
            if g["off"] <= t < g["win_hi"]:
                return i
        return None

    # Phase A — classify. A word is CLEAN when its span lies inside one
    # group's real audio; everything else is a pad case.
    cls = []
    for w in words:
        gpos = None
        for i, g in enumerate(groups):
            if w["t_s"] >= g["off"] and w["t_e"] <= g["end"]:
                gpos = i
                break
        cls.append(gpos)

    # Per-group occupied bounds from CLEAN words (for feasibility slots)
    first_clean = [None] * n   # earliest clean word start (rel) per group
    last_clean = [None] * n    # latest clean word end (rel) per group
    for w, gpos in zip(words, cls):
        if gpos is None:
            continue
        rs = w["t_s"] - groups[gpos]["off"]
        re_ = w["t_e"] - groups[gpos]["off"]
        if first_clean[gpos] is None or rs < first_clean[gpos]:
            first_clean[gpos] = rs
        if last_clean[gpos] is None or re_ > last_clean[gpos]:
            last_clean[gpos] = re_

    out = []
    for wi, (w, gpos) in enumerate(zip(words, cls)):
        if gpos is not None:
            out.append({"gpos": gpos,
                        "rel_s": w["t_s"] - groups[gpos]["off"],
                        "rel_e": w["t_e"] - groups[gpos]["off"],
                        "flag": None})
            continue

        # ── Pad case. Candidate sides = the groups around the word. ─────────
        left = None
        for i in range(n - 1, -1, -1):
            if groups[i]["off"] <= w["t_s"]:
                left = i
                break
        right = left + 1 if (left is not None and left + 1 < n) else (
            0 if left is None else None)

        def feasible(side):
            if side is None:
                return None
            g = groups[side]
            gdur = g["end"] - g["off"]
            if side == left:
                lo = last_clean[side] if last_clean[side] is not None else 0.0
                avail = gdur - lo
            else:
                hi = first_clean[side] if first_clean[side] is not None else gdur
                avail = hi
            return avail if avail >= PAD_MIN_WORD_SEC else None

        avail_l, avail_r = feasible(left), feasible(right)
        survivors = [s for s, a in ((left, avail_l), (right, avail_r))
                     if s is not None and a is not None]

        chosen, flag = None, "pad_moved"
        if survivors:
            # ── Layer 3: continuity vote, proximity-gated. Runs for ANY
            # survivor set — a same-segment neighbour is stronger evidence
            # than the acoustic probe, so it may confirm a lone survivor. ──
            votes = []
            prev = next((j for j in range(wi - 1, -1, -1)
                         if cls[j] is not None), None)
            nxt = next((j for j in range(wi + 1, len(words))
                        if cls[j] is not None), None)
            if right in survivors:
                d_right = max(0.0, groups[right]["off"] - w["t_e"])
                if (nxt is not None and cls[nxt] == right
                        and words[nxt]["seg_id"] == w["seg_id"]
                        and d_right <= PAD_PROXIMITY_GATE_SEC):
                    votes.append((right, words[nxt]["t_s"] - w["t_e"]))
            if left in survivors:
                d_left = max(0.0, w["t_s"] - groups[left]["end"])
                if (prev is not None and cls[prev] == left
                        and words[prev]["seg_id"] == w["seg_id"]
                        and d_left <= PAD_PROXIMITY_GATE_SEC):
                    votes.append((left, w["t_s"] - words[prev]["t_e"]))
            if len(votes) == 1:
                chosen = votes[0][0]
            elif len(votes) == 2:
                chosen = min(votes, key=lambda v: v[1])[0]
            elif support_fn is None and len(survivors) == 1:
                # degraded mode takes the lone feasible side as-is; with an
                # aligner it must still earn acoustic support below —
                # feasibility is a rejector, never a confirmer
                chosen = survivors[0]

        if chosen is None and support_fn is not None:
            if not survivors:
                # BOTH sides packed: the word's slot is already owned —
                # overwhelmingly a boundary DUPLICATE (Whisper re-emits a
                # word across the pad; its twin occupies the real audio, so
                # the char-coverage probe saturates and CANNOT tell the copy
                # from the original). Live-measured 2026-10-12 (19 episodes,
                # mix): probing instead of dropping here readmitted ~700
                # insertions while recovering ~20 deletions — feasibility IS
                # the duplicate suppressor, so it convicts. Degraded mode
                # (no aligner) still never drops.
                out.append({"gpos": None, "rel_s": None, "rel_e": None,
                            "flag": "pad_dropped",
                            "near": owner_by_window(w["t_s"])})
                continue
            # ── Layer 4: acoustic adjudication over every raw candidate
            # side (the infeasible one still contributes COMPARISON
            # evidence for the margin). A probe that cannot judge a side
            # (empty charset, degenerate window, model failure) returns
            # None and ABSTAINS. The drop below requires every existing
            # side judged and none supporting the word; a best side within
            # PAD_SUPPORT_MARGIN of the runner-up cannot adjudicate either
            # (short words saturate char coverage on both sides).
            # DELIBERATE asymmetry: one judged side >= floor is POSITIVE
            # evidence and may relocate even when the other side abstained
            # — only a drop verdict requires every side judged. ─────────────
            sides = [s for s in (left, right) if s is not None]
            scores = {}
            abstained = False
            for side in sides:
                g = groups[side]
                gdur = g["end"] - g["off"]
                if side == left:
                    lo, hi = max(0.0, gdur - PAD_PROBE_WINDOW_SEC), gdur
                else:
                    lo, hi = 0.0, min(PAD_PROBE_WINDOW_SEC, gdur)
                sc = support_fn(wi, side, lo, hi)
                if sc is None:
                    abstained = True
                else:
                    scores[side] = sc
            if scores:
                ranked = sorted(scores, key=lambda s: -scores[s])
                best = ranked[0]
                if scores[best] >= PAD_SUPPORT_FLOOR:
                    lead = (scores[best] - scores[ranked[1]]
                            if len(ranked) > 1 else 1.0)
                    if lead >= PAD_SUPPORT_MARGIN:
                        chosen = best
                    else:
                        # saturated tie between SUPPORTED sides: feasibility
                        # (a rejector) may break it — a side that passed the
                        # floor AND has room wins when it is the only one;
                        # a tie with room on both sides keeps + flags below
                        feas = [s for s in ranked if s in survivors
                                and scores[s] >= PAD_SUPPORT_FLOOR]
                        if len(feas) == 1:
                            chosen = feas[0]
                elif not abstained:
                    # every existing side judged, none supports the word
                    # (near = stats attribution only, never a placement)
                    out.append({"gpos": None, "rel_s": None, "rel_e": None,
                                "flag": "pad_dropped", "near": best})
                    continue

        if chosen is None:
            # ── Keep + flag: nothing is ever dropped without full probe
            # evidence. A LONE feasible side is the keep side (the legacy
            # window owner is always left by construction — don't bake that
            # bias in when feasibility points the other way); otherwise the
            # legacy owner (nearest group when the word sits past every
            # window) absorbs it through the SAME occupancy-updating clamp
            # as every relocation below. ────────────────────────────────────
            if len(survivors) == 1:
                side = survivors[0]
            else:
                side = owner_by_window(w["t_s"])
                if side is None:
                    side = left if left is not None else right
            chosen, flag = side, "pad_unresolved"

        # ── Absorb: minimal-displacement clamp into the side's free slot ─────
        # A word mostly inside the group keeps its position and only the
        # protruding part is clipped; a fully-outside word slides to the edge.
        # Accepted residual: at a FULLY packed edge every absorbed word gets
        # the same bounded PAD_MIN sliver — words claiming audio past the
        # group's end have no distinct honest home, and the flag records it.
        g = groups[chosen]
        gdur = g["end"] - g["off"]
        ws_rel, we_rel = w["t_s"] - g["off"], w["t_e"] - g["off"]
        if chosen == left:
            lo = last_clean[chosen] if last_clean[chosen] is not None else 0.0
            rel_e = min(max(we_rel, lo + PAD_MIN_WORD_SEC), gdur)
            rel_s = min(max(ws_rel, lo), rel_e - PAD_MIN_WORD_SEC)
            last_clean[chosen] = rel_e
        else:
            hi = first_clean[chosen] if first_clean[chosen] is not None else gdur
            rel_s = max(min(ws_rel, hi - PAD_MIN_WORD_SEC), 0.0)
            rel_e = max(min(we_rel, hi), rel_s + PAD_MIN_WORD_SEC)
            first_clean[chosen] = rel_s
        out.append({"gpos": chosen, "rel_s": round(rel_s, 3),
                    "rel_e": round(rel_e, 3), "flag": flag})
    return out


# =============================================================================
# BATCH BUILDING
# =============================================================================

def build_batches(
    groups:     list[dict],
    target_sec: float = BATCH_TARGET_SEC,
) -> list[list[dict]]:
    """
    Pack speech groups into batches whose combined wall-clock duration
    stays within *target_sec*.

    Each group must carry an "audio" field (np.ndarray float32 PCM) and
    "duration" (wall-clock seconds).

    Each group in the output batch gains two extra fields:
        local_offset_sec : where the group's audio starts in the combined buffer
        local_end_sec    : where the group's audio ends in the combined buffer

    Combined audio layout per batch:
        [group_0_audio | pad | group_1_audio | pad | ... | group_N_audio | pad]
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_sec = 0.0

    for group in groups:
        dur = group["duration"]
        slot = dur + INTER_GROUP_PAD_SEC

        # Flush current batch when adding this group would exceed target
        if current and current_sec + slot > target_sec:
            batches.append(current)
            current = []
            current_sec = 0.0

        current.append({
            **group,
            "local_offset_sec": current_sec,
            "local_end_sec":    current_sec + dur,
        })
        current_sec += slot

    if current:
        batches.append(current)

    return batches


# =============================================================================
# BATCH WORKER
# =============================================================================

def batch_worker(batch: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Transcribe a batch of speech groups with a SINGLE Whisper call,
    assign words to groups, and refine low-confidence timestamps.

    Sentence splitting is NOT performed here — it runs once per iterator
    in split_words_into_sentences() after all batches complete.

    Steps
    ─────
    1. Concatenate group audio with INTER_GROUP_PAD_SEC gaps → combined audio
    2. One _whisper_model.transcribe() call on the combined audio
    3. Assign each word to its owning group using soft boundaries
    4. Refine low-confidence words with the alignment model (per group,
       with group-relative timestamps)
    5. Collect words and build per-group stats

    Returns
    ───────
    (all_words, all_stats)
      all_words : flat list of word dicts with absolute timestamps and _raw
      all_stats : per-group stats dicts
    """
    pid = os.getpid()
    t0 = time.monotonic()

    if _whisper_model is None:
        return [], [
            {"group_idx": g["group_idx"], "error": "worker not initialised"}
            for g in batch
        ]

    # ── 1. Build combined audio ───────────────────────────────────────────────
    silence = np.zeros(
        int(INTER_GROUP_PAD_SEC * _sample_rate), dtype=np.float32)
    parts = []
    for group in batch:
        parts.append(group["audio"].astype(np.float32))
        parts.append(silence)
    combined = np.concatenate(parts)
    combined_dur = len(combined) / _sample_rate

    # ── 2. Whisper — single call ──────────────────────────────────────────────
    t_w = time.monotonic()
    try:
        raw_segs, _ = _whisper_model.transcribe(
            combined,
            **{
                **_whisper_kwargs,
                "language":        _language,
                "word_timestamps": True,
                "vad_filter":      False,
            },
        )
        segments = list(raw_segs)
        whisper_sec = round(time.monotonic() - t_w, 2)
    except Exception as exc:
        log.error("[PID %d] batch_worker transcribe error: %s", pid, exc)
        return [], [
            {
                "group_idx":    g["group_idx"],
                "pid":          pid,
                "abs_start":    g["start"],
                "abs_end":      g["end"],
                "duration_sec": round(g["duration"], 3),
                "error":        f"transcribe failed: {exc}",
            }
            for g in batch
        ]

    log.info(
        "[PID %d] batch  groups=%s  combined=%.1fs  segs=%d  whisper=%.2fs",
        pid,
        [g["group_idx"] for g in batch],
        combined_dur,
        len(segments),
        whisper_sec,
    )

    # ── 3. Assign words to groups: emission order + pad-boundary ladder ───────
    # (_resolve_word_groups; see the PAD-BOUNDARY WORD RESOLUTION block above —
    # the legacy soft-window lookup mapped a pad-straddling word through the
    # WRONG group, turning sub-second Whisper slop into a 50 s absolute error.)
    group_words: dict[int, list[dict]] = {
        g["group_idx"]: [] for g in batch
    }

    t_align = time.monotonic()
    n_orphaned = 0
    pad_counts = {"pad_moved": 0, "pad_unresolved": 0, "pad_dropped": 0}
    per_group_pad: dict[int, dict] = {
        g["group_idx"]: dict(pad_counts) for g in batch
    }

    flat_words: list[dict] = []
    raw_refs: list[Any] = []
    for si, seg in enumerate(segments):
        if not seg.words:
            continue
        for w in seg.words:
            flat_words.append({"t_s": float(w.start), "t_e": float(w.end),
                               "seg_id": si})
            raw_refs.append(w)

    groups_meta = [{
        "off": g["local_offset_sec"],
        "end": g["local_offset_sec"] + g["duration"],
        "win_hi": (batch[i + 1]["local_offset_sec"] if i + 1 < len(batch)
                   else g["local_end_sec"] + INTER_GROUP_PAD_SEC),
    } for i, g in enumerate(batch)]

    def _support(word_idx, group_pos, lo_sec, hi_sec):
        return _pad_word_support(raw_refs[word_idx].word.strip(),
                                 batch[group_pos]["audio"], lo_sec, hi_sec)

    resolved = _resolve_word_groups(
        flat_words, groups_meta,
        support_fn=_support if _align_available else None)

    for w, r in zip(raw_refs, resolved):
        flag = r["flag"]
        if r["gpos"] is None:
            n_orphaned += 1
            pad_counts["pad_dropped"] += 1
            near = r.get("near")
            if near is not None:
                per_group_pad[batch[near]["group_idx"]]["pad_dropped"] += 1
            log.info("[PID %d] pad word '%s' dropped — no acoustic support "
                     "on either candidate side", pid, w.word.strip())
            continue
        owning = batch[r["gpos"]]
        if flag:
            pad_counts[flag] += 1
            per_group_pad[owning["group_idx"]][flag] += 1
            log.info("[PID %d] pad word '%s' → group %d (%s)",
                     pid, w.word.strip(), owning["group_idx"], flag)

        abs_ws = round(owning["start"] + r["rel_s"], 3)
        abs_we = round(owning["start"] + r["rel_e"], 3)

        wdict = {
            "word":        w.word.strip(),
            "_raw":        w.word,
            "start":       abs_ws,
            "end":         abs_we,
            "probability": round(w.probability, 3),
        }
        if flag:
            wdict["pad_resolution"] = flag

        # ── 4. Alignment refinement (relocated words always re-timed) ────────
        if _align_available and (flag or wdict["probability"]
                                 < _align_conf_thresh):
            rs, re = _refine_word(
                wdict["word"], r["rel_s"], r["rel_e"],
                owning["audio"],
            )
            wdict["start"] = round(owning["start"] + rs, 3)
            wdict["end"]   = round(owning["start"] + re, 3)
            if abs(rs - r["rel_s"]) > 5e-4 or abs(re - r["rel_e"]) > 5e-4:
                wdict["_refined"] = True   # the aligner actually moved it

        group_words[owning["group_idx"]].append(wdict)

    if any(pad_counts.values()):
        log.info("[PID %d] pad resolution: moved=%d unresolved=%d dropped=%d",
                 pid, pad_counts["pad_moved"], pad_counts["pad_unresolved"],
                 pad_counts["pad_dropped"])

    align_sec = round(time.monotonic() - t_align, 2)

    # ── 5. Collect words and build per-group stats ────────────────────────────
    all_words: list[dict] = []
    all_stats: list[dict] = []

    for group in batch:
        gidx = group["group_idx"]
        words = group_words[gidx]
        dur = round(group["duration"], 3)

        if not words:
            log.warning(
                "[PID %d] group %d [%.2fs→%.2fs  dur=%.2fs  turns=%d]: "
                "no words assigned — Whisper may have skipped this group",
                pid, gidx, group["start"], group["end"],
                dur, group["num_turns"],
            )
            all_stats.append({
                "group_idx":       gidx,
                "pid":             pid,
                "abs_start":       round(group["start"], 3),
                "abs_end":         round(group["end"],   3),
                "duration_sec":    dur,
                "num_turns":       group["num_turns"],
                "whisper_sec":     0.0,
                "align_sec":       0.0,
                "segments_found":  0,
                "words_total":     0,
                "words_refined":   0,
                "words_kept":      0,
                "align_available": _align_available,
                **per_group_pad[gidx],
                "error":           "no words assigned to group",
            })
            continue

        all_words.extend(words)

        all_stats.append({
            "group_idx":       gidx,
            "pid":             pid,
            "abs_start":       round(group["start"], 3),
            "abs_end":         round(group["end"],   3),
            "duration_sec":    dur,
            "num_turns":       group["num_turns"],
            "whisper_sec":     round(whisper_sec / len(batch), 2),
            "align_sec":       round(align_sec / len(batch), 2),
            "segments_found":  len(segments),
            "words_total":     len(words),
            "words_refined":   sum(1 for w in words if w.get("_refined")),
            "words_kept":      sum(1 for w in words if not w.get("_refined")),
            "align_available": _align_available,
            **per_group_pad[gidx],
            "error":           None,
        })

        log.info(
            "[PID %d] group %3d  turns=%d  words=%d",
            pid, gidx, group["num_turns"], len(words),
        )

    # ── Diagnostic: surface zero-word groups ──────────────────────────────────
    n_zero_word = sum(
        1 for s in all_stats
        if s.get("error") == "no words assigned to group"
    )
    if n_zero_word:
        log.warning(
            "[PID %d] batch: %d/%d group(s) had no words assigned  "
            "(orphaned_words=%d)",
            pid, n_zero_word, len(batch), n_orphaned,
        )

    log.info(
        "[PID %d] batch done  groups=%d  words=%d  "
        "whisper=%.2fs  align=%.2fs  total=%.2fs",
        pid,
        len(batch),
        len(all_words),
        whisper_sec,
        align_sec,
        round(time.monotonic() - t0, 2),
    )

    return all_words, all_stats


# =============================================================================
# ITERATOR-LEVEL SENTENCE SPLITTING
# =============================================================================

def _dedup_seam_twins(words: list[dict]) -> tuple[list[dict], int]:
    """Drop boundary TWIN words: when a VAD group boundary lands mid-word,
    both adjacent group windows decode the word and the aligner anchors
    both copies to the SAME audio event (near-identical spans). The pad
    ladder only adjudicates words inside the synthetic pad, so twins in
    owned audio pass it. Trigger (language-agnostic by construction):
    adjacent after time-sort, casefold-equal text (NO linguistic
    normalization), span-overlap IoU > 0.5. Verdict: keep the higher
    decode probability (tie keeps the earlier copy). Abstention: a word
    missing timestamps is never judged. Genuine repetitions ("no, no")
    carry disjoint advancing spans and can never trigger - one voice
    cannot say two words in the same span. Measured before the fix:
    ~0.3% of words on an English caption corpus, ~6% on a Korean asset
    (character-based segmentation straddles seams more often).
    """
    out: list[dict] = []
    dropped = 0
    for w in words:
        if out:
            p = out[-1]
            ws, we = w.get("start"), w.get("end")
            ps, pe = p.get("start"), p.get("end")
            if (None not in (ws, we, ps, pe)
                    and w["word"].casefold() == p["word"].casefold()):
                inter = min(we, pe) - max(ws, ps)
                union = max(we, pe) - min(ws, ps)
                if union > 0 and inter / union > 0.5:
                    dropped += 1
                    if ((w.get("probability") or 0.0)
                            > (p.get("probability") or 0.0)):
                        out[-1] = w
                    continue
        out.append(w)
    return out, dropped


def split_words_into_sentences(

    all_words: list[dict],
    language:  str,
    merge_gap_threshold_sec: float = 0.20,
) -> list[dict]:
    """
    Split a flat list of timestamped words into sentences.

    Called once per iterator after all batch_worker calls complete.
    wtpsplit must already be loaded (via worker_init) before calling.

    Steps
    ─────
    1. Sort words by timestamp
    2. Ensure _raw spacing consistency across batch boundaries
    3. Reconstruct full text from _raw tokens
    4. Run wtpsplit sentence splitting (single call on full text)
    5. Map each sentence to its constituent words
    6. Merge short sentences

    Why iterator-level splitting
    ────────────────────────────
    Per-group splitting cannot see across group boundaries, causing words
    at group edges (e.g. "Pride" at the end of one group, "before a fall"
    at the start of the next) to be incorrectly merged backward into the
    previous sentence.  Iterator-level splitting gives wtpsplit the full
    text, so it correctly identifies "Pride before a fall." as one sentence.

    wtpsplit internally uses a 512-token sliding window with stride 64,
    so long texts (5000+ words from a 360s chunk) are handled efficiently
    without any external windowing.

    Returns a list of sentence dicts ready for output.
    """
    if not all_words:
        return []

    # ── 1. Sort by timestamp ──────────────────────────────────────────────
    all_words.sort(key=lambda w: w["start"])

    # ── 1b. Seam-twin dedup (see _dedup_seam_twins) ───────────────────────
    all_words, twins = _dedup_seam_twins(all_words)
    if twins:
        log.info("Seam-twin dedup: dropped %d duplicate boundary words",
                 twins)

    # ── 2. Ensure _raw spacing at batch boundaries ───────────────────────
    # Within a batch, _raw tokens have correct spacing from a single Whisper
    # call (e.g. " Pride", " before").  Across batches, the first token of
    # a new batch may lack a leading space since it was the start of a
    # separate Whisper decoding pass.  Insert a space to ensure the full
    # reconstructed text has consistent single-space word separation,
    # matching what split_sentences() produces after whitespace normalisation.
    for i in range(1, len(all_words)):
        raw = all_words[i].get("_raw", all_words[i]["word"])
        prev_raw = all_words[i - 1].get("_raw", all_words[i - 1]["word"])
        if (prev_raw and raw
                and not prev_raw[-1].isspace()
                and not raw[0].isspace()):
            all_words[i] = {**all_words[i], "_raw": " " + raw}

    # ── 3. Reconstruct full text ──────────────────────────────────────────
    text = "".join(w.get("_raw", w["word"]) for w in all_words).strip()
    if not text:
        return []

    # ── 4. wtpsplit — single call ─────────────────────────────────────────
    assembler = SentenceAssembler(
        language=language,
        merge_gap_threshold_sec=merge_gap_threshold_sec,
    )
    raw_sents = assembler.split(
        text=text,
        start=all_words[0]["start"],
        end=all_words[-1]["end"],
        whisper_words=all_words,
    )

    # ── 5. Map sentences to words ─────────────────────────────────────────
    # Use the assembler's char-level PARTITION (word_ids) when present:
    # re-deriving membership from TIME spans double-assigned boundary
    # words whenever sentence spans overlap (one word object in two
    # sentences -> emitted twice in the final transcript; measured at
    # ~0.64% of words on an English caption corpus, every build since
    # the original pipeline). The time filter remains only for the
    # time-ratio mapper (no word timestamps), guarded by a used-set so
    # a word can never join two sentences.
    sentences: list[dict] = []
    used: set[int] = set()
    for sent in raw_sents:
        ids = sent.get("word_ids")
        if ids is not None:
            sent_words = [all_words[i] for i in ids
                          if 0 <= i < len(all_words)]
        else:
            sent_words = []
            for i, w in enumerate(all_words):
                if i in used:
                    continue
                if sent["start"] <= w["start"] <= sent["end"]:
                    sent_words.append(w)
                    used.add(i)
        if not sent_words:
            continue
        conf = round(
            sum(w["probability"] for w in sent_words) / len(sent_words), 3
        )
        sentences.append({
            "text":       sent["text"],
            "start":      sent_words[0]["start"],
            "end":        sent_words[-1]["end"],
            "confidence": conf,
            "words":      [
                {k: v for k, v in w.items() if k in _OUTPUT_WORD_FIELDS}
                for w in sent_words
            ],
        })

    # ── 6. Merge short sentences ──────────────────────────────────────────
    sentences.sort(key=lambda s: s["start"])
    return assembler.merge_short(sentences)


def rehome_straddle_strays(sentences: list[dict],
                           groups: list[dict]) -> int:
    """Layer 5 of the pad ladder: sentence-mates override weak placements.

    A pad word adjudicated by the aligner probe alone can land on the wrong
    side — short function words give weak char-coverage evidence (measured:
    a relocated "You" stretched its sentence 26.6 s across an inter-group
    gap on stem audio). Once sentences exist there is stronger evidence than
    the probe had: the word's own sentence-mates. Any pad-FLAGGED word
    stranded in a group holding NO clean word of its sentence is re-homed
    into the group where the sentence's clean majority sits, re-timed there
    when the aligner is available, and flagged "pad_rehomed".

    Clean words are never touched, so a genuine cross-group sentence
    ("Pride ... before a fall") keeps its honest placement: its words are
    clean on both sides. Nothing is ever dropped here — the majority group
    always exists as a destination.

    sentences: output of split_words_into_sentences(), mutated in place
    groups:    [{start, end, audio}] this iterator's speech groups
               (absolute seconds)
    Returns the number of words moved.
    """
    spans = sorted(groups, key=lambda g: (g["start"], g["end"]))

    def owner(w):
        mid = (w["start"] + w["end"]) / 2.0
        for gi, g in enumerate(spans):
            if g["start"] - 0.05 <= mid <= g["end"] + 0.05:
                return gi
        return None

    moved = 0
    for sent in sentences:
        words = sent.get("words") or []
        if len(words) < 2:
            continue
        own = [owner(w) for w in words]
        clean = [i for i, w in enumerate(words)
                 if not w.get("pad_resolution") and own[i] is not None]
        if not clean:
            continue
        clean_groups = {own[i] for i in clean}
        strays = [i for i, w in enumerate(words)
                  if w.get("pad_resolution") and own[i] is not None
                  and own[i] not in clean_groups]
        if not strays:
            continue

        # majority group of the CLEAN words (count, then summed duration)
        votes: dict[int, list[float]] = {}
        for i in clean:
            votes.setdefault(own[i], []).append(
                words[i]["end"] - words[i]["start"])
        maj = max(votes, key=lambda g: (len(votes[g]), sum(votes[g])))
        g = spans[maj]
        g_start, g_end = g["start"], g["end"]
        maj_idx = [i for i in clean if own[i] == maj]
        left_edge = min(words[i]["start"] for i in maj_idx)
        right_edge = max(words[i]["end"] for i in maj_idx)

        # OTHER sentences' words already occupying the majority group bound
        # the placement too — a rehomed stray must not double-book their
        # audio (overlap stays capped at PAD_MIN_WORD_SEC via the squeeze)
        # not-yet-final flags (pad_moved/pad_unresolved — may be rehomed
        # later in this pass) are excluded; observed words AND already-
        # rehomed words (final) bind. Membership is by OVERLAP with the
        # group, not midpoint, so edge-straddling occupants stay visible.
        occ = [(w2["start"], w2["end"])
               for s2 in sentences if s2 is not sent
               for w2 in (s2.get("words") or [])
               if w2.get("pad_resolution") in (None, "pad_rehomed")
               and w2["end"] > g_start - 0.05
               and w2["start"] < g_end + 0.05]
        # a straddling occupant counts too; its bound is capped at the edge
        floor_bound = min(max([g_start] + [o_e for o_s, o_e in occ
                                           if o_s < left_edge]), left_edge)
        ceil_bound = max(min([g_end] + [o_s for o_s, o_e in occ
                                        if o_e > right_edge]), right_edge)

        # When the slot before/after the anchors is too small (the stray is
        # usually the group's TRUE first word, and anchors often start within
        # ~0.1 s of the group edge), the nearest stray squeezes to the group
        # edge with anchor overlap capped at PAD_MIN_WORD_SEC; farther strays
        # on that side then keep their ladder placement (and flag) — never
        # stacked over the anchors or each other.
        lead_blocked = tail_blocked = False
        for i in sorted(strays, key=lambda i: -i if i < maj_idx[0] else i):
            w = words[i]
            dur = max(min(w["end"] - w["start"], 1.0), PAD_MIN_WORD_SEC)
            if i < maj_idx[0]:        # textually before the anchor words
                if lead_blocked:
                    continue
                we = left_edge - 0.02
                ws = max(we - dur, floor_bound)
                if we - ws < PAD_MIN_WORD_SEC:
                    ws, we = floor_bound, floor_bound + PAD_MIN_WORD_SEC
                    lead_blocked = True
                else:
                    left_edge = ws
            elif i > maj_idx[-1]:     # textually after
                if tail_blocked:
                    continue
                ws = right_edge + 0.02
                we = min(ws + dur, ceil_bound)
                if we - ws < PAD_MIN_WORD_SEC:
                    ws, we = ceil_bound - PAD_MIN_WORD_SEC, ceil_bound
                    tail_blocked = True
                else:
                    right_edge = we
            else:                     # textually between anchors
                prev_maj = max(j for j in maj_idx if j < i)
                next_maj = min(j for j in maj_idx if j > i)
                ws = words[prev_maj]["end"] + 0.001
                if any(o_s - 1e-6 < ws < o_e for o_s, o_e in occ):
                    continue          # slot start sits inside an occupant
                we = min([ws + dur, words[next_maj]["start"], g_end]
                         + [o_s for o_s, o_e in occ if o_s >= ws])
                if we - ws < PAD_MIN_WORD_SEC:
                    continue
            if _align_available:
                rs, re_ = _refine_word(
                    w["word"], ws - g_start, we - g_start, g["audio"])
                # accept only a refinement that stays INSIDE the placed slot
                # (shrink-only) — the slot already encodes every occupancy
                # bound; refine may move ±0.15 s and must not escape it
                if ws - 1e-6 <= g_start + rs < g_start + re_ <= we + 1e-6:
                    ws, we = g_start + rs, g_start + re_
            w["start"], w["end"] = round(ws, 3), round(we, 3)
            w["pad_resolution"] = "pad_rehomed"
            moved += 1

        sent["start"] = min(w["start"] for w in words)
        sent["end"] = max(w["end"] for w in words)
    return moved
