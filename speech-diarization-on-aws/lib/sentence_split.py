# lib/sentence_split.py
"""
Sentence boundary splitting — wtpsplit / SaT only.

Why wtpsplit replaces NLTK entirely
─────────────────────────────────────
  • Single sat-3l-sm model (~100 MB) covers 85+ languages.
  • Native support for CJK, Arabic, Welsh, all Indic scripts.
  • No per-language punkt models, no per-language downloads.
  • More accurate than punkt on most languages it covers.

Whisper→sentence timestamp mapping
────────────────────────────────────
  Space-separated languages (en, fr, de, cy, ar, …):
    Sequential word index — sentence i gets words[word_cursor : word_cursor+n]
    where n = len(sentence.split()).

  Character-based languages (zh, ja, ko, th):
    Character-ratio mapping — sentence i starts at
    int(char_offset / total_chars * n_words) and ends at
    int((char_offset + len(sent)) / total_chars * n_words).
    Whisper produces sub-word tokens for CJK so word boundaries
    are meaningless; character position is the only stable proxy.

Merge strategy — gap-aware merge_short
───────────────────────────────────────
  Short sentences (< min_sentence_words or < min_sentence_duration) are
  merged into adjacent neighbours ONLY when the inter-sentence gap is
  ≤ merge_gap_threshold_sec (default 0.20s, from DEFAULT_MERGE_GAP_THRESHOLD_SEC
  in lib/input_schema.py).

  This prevents cross-speaker merging in rapid exchanges (quiz Q&A,
  interviews) where a contestant's 1-word answer ("Family.") is separated
  from the host's question by 0.2-1.0s.  Without gap awareness, these
  answers get merged into the host's sentence, contaminating the speaker
  embedding with the wrong speaker's audio.

  No pre-merge is performed on text strings before word mapping.  All
  merging is deferred to merge_short() which has full timestamp context.

Fallback when wtpsplit is unavailable
──────────────────────────────────────
  regex patterns per script family.  This should never happen in
  production (wtpsplit is in the Lambda layer) but prevents hard
  crashes during local testing without the full Lambda environment.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# Languages that have no word-boundary spaces.
CHAR_BASED = {"zh", "ja", "ko", "th"}

# Regex patterns used only when wtpsplit is unavailable.
_RE_CJK = re.compile(r'(?<=[。！？…‥])\s*')
_RE_RTL = re.compile(r'(?<=[.!?؟])\s+')
_RE_GENERIC = re.compile(r'(?<=[.!?\u0964\u0965])\s+')

# Module-level model cache — loaded once per container.
_wtp_model = None


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_wtpsplit(model_dir: str, intra_op_num_threads: int = 1) -> None:
    """
    Load the SaT ONNX model from `model_dir` into the module cache.
    Should be called once during Lambda init (worker_init in transcribe.py).

    model_dir is /tmp/models/wtpsplit/, extracted by ensure_in_tmp().
    It must contain:
      model_optimized.onnx  — INT8 quantized SaT model
      config.json           — model config (use_fp16=False)
      tokenizer.json        — XLM-RoBERTa fast tokenizer

    intra_op_num_threads controls how many OS threads each ORT inference call
    may use.  Set to max(1, total_cores // num_workers) so that N concurrent
    threads each use an equal share of the available vCPUs.

    Uses lib.wtpsplit_sat.SaT — no torch, transformers, or huggingface_hub.
    """
    global _wtp_model
    if _wtp_model is not None:
        return

    try:
        from lib.wtpsplit_sat import SaT

        log.info(
            f"Loading SaT ONNX from {model_dir}  "
            f"(intra_op_threads={intra_op_num_threads})")
        _wtp_model = SaT(
            model_dir,
            ort_providers=["CPUExecutionProvider"],
            intra_op_num_threads=intra_op_num_threads,
        )
        log.info("SaT ONNX loaded ✅")
    except Exception as exc:
        log.warning(
            f"SaT failed to load ({exc}) — regex fallback will be used"
        )
        _wtp_model = None


# =============================================================================
# SPLITTING
# =============================================================================

def split_sentences(text: str, language: str) -> list[str]:
    """
    Split `text` into sentences for the given ISO-639-1 `language`.

    Uses wtpsplit if available, regex otherwise.
    Returns a list of non-empty sentence strings.
    """
    text = " ".join(text.split())   # normalise whitespace
    if not text:
        return []

    if _wtp_model is not None:
        try:
            lang = language.lower()
            sents = _wtp_model.split(text)  # -sm models are language-agnostic
            result = [s.strip() for s in sents if s.strip()]
            if result:
                return result
        except Exception as exc:
            log.warning(f"wtpsplit.split() failed for lang={language}: {exc}")

    # Regex fallback
    return _regex_split(text, language)


def _regex_split(text: str, language: str) -> list[str]:
    lang = language.lower()
    if lang in CHAR_BASED:
        pattern = _RE_CJK
    elif lang in {"ar", "fa", "he", "ur"}:
        pattern = _RE_RTL
    else:
        pattern = _RE_GENERIC
    parts = pattern.split(text)
    return [p.strip() for p in parts if p.strip()]


# =============================================================================
# SENTENCE ASSEMBLER
# =============================================================================

class SentenceAssembler:
    """
    Splits Whisper text into sentences and maps them to word timestamps.

    Mirrors the logic from the reference TranscriptionPipeline but uses
    wtpsplit for all tokenization — no NLTK dependency.
    """

    def __init__(
        self,
        language:                str,
        min_sentence_duration:   float = 1.0,
        min_sentence_words:      int = 3,
        min_sentence_chars:      int = 10,
        merge_gap_threshold_sec: float = 0.20,
    ):
        self.language = language.lower()
        self.is_char_based = self.language in CHAR_BASED
        self.min_sentence_duration = min_sentence_duration
        self.min_sentence_words = min_sentence_words
        self.min_sentence_chars = min_sentence_chars
        self.merge_gap_threshold_sec = merge_gap_threshold_sec

    # ── unit helpers ──────────────────────────────────────────────────────────

    def _count_units(self, text: str) -> int:
        if self.is_char_based:
            return len(text.replace(" ", ""))
        return len(text.split())

    def _min_units(self) -> int:
        return (self.min_sentence_chars if self.is_char_based
                else self.min_sentence_words)

    # ── sentence → word mapping ───────────────────────────────────────────────

    def _map_word_index(
        self,
        sents:         list[str],
        whisper_words: list[dict],
    ) -> list[dict]:
        """
        Assign word-level timestamps to sentences via character-position
        alignment.

        Each word carries a '_raw' field (the original faster-whisper token
        string, which may have a leading space).  Concatenating all '_raw'
        values reproduces the exact text that was passed to split_sentences(),
        so every sentence string is a guaranteed substring of that text.

        We compute the content span of each word within the reconstructed text
        (skipping any leading whitespace, which is inter-word spacing rather
        than word content), then assign each word to whichever sentence's
        character range contains it.
        """
        if not sents or not whisper_words:
            return []

        # ── 1. Reconstruct the full text ─────────────────────────────────────
        raw_tokens = [w.get("_raw", w["word"]) for w in whisper_words]
        raw_concat = "".join(raw_tokens)
        full_text = raw_concat.strip()
        left_trim = len(raw_concat) - len(raw_concat.lstrip())

        # ── 2. Compute each word's content span within full_text ─────────────
        content_starts: list[int] = []
        content_ends:   list[int] = []
        raw_pos = 0
        for tok in raw_tokens:
            leading = len(tok) - len(tok.lstrip())
            content_start = max(0, raw_pos + leading - left_trim)
            content_end = max(0, raw_pos + len(tok) - left_trim)
            content_starts.append(content_start)
            content_ends.append(content_end)
            raw_pos += len(tok)

        # ── 3. Map each sentence to the words whose content falls within it ──
        # The char ranges are DISJOINT (search_from advances past each
        # sentence), so the word_ids below are a true PARTITION: a word
        # belongs to at most ONE sentence. This ownership is the fix for
        # the double-assignment defect (one boundary word in two
        # overlapping sentence TIME spans -> emitted twice downstream);
        # consumers must use word_ids, never re-derive membership from
        # time spans.
        results:     list[dict] = []
        search_from: int = 0

        for sent in sents:
            s = sent.strip()
            idx = full_text.find(s, search_from)
            if idx == -1:
                log.debug(
                    "_map_word_index: sentence not found in full_text: %r", s)
                continue
            end = idx + len(s)

            owned_ids = [
                i for i, (cs, ce) in enumerate(
                    zip(content_starts, content_ends))
                if cs >= idx and ce <= end
            ]
            if not owned_ids:
                # Looser pass: word content merely starts within the range
                owned_ids = [
                    i for i, cs in enumerate(content_starts)
                    if idx <= cs < end
                ]
            if not owned_ids:
                continue

            results.append({
                "text":  s,
                "start": round(whisper_words[owned_ids[0]]["start"], 3),
                "end":   round(whisper_words[owned_ids[-1]]["end"],  3),
                "word_ids": owned_ids,
            })
            search_from = end

        return results

    def _map_char_ratio(
        self,
        sents:         list[str],
        whisper_words: list[dict],
    ) -> list[dict]:
        """Character-ratio mapping for CJK languages."""
        n_words = len(whisper_words)
        total_chars = sum(len(s) for s in sents)
        if n_words == 0 or total_chars == 0:
            return []

        results = []
        char_offset = 0
        prev_ei = -1    # enforce a PARTITION: consecutive ratio ranges may
        for sent in sents:   # share a boundary index - never assign it twice
            s_len = len(sent)
            start_ratio = char_offset / total_chars
            end_ratio = min((char_offset + s_len) / total_chars, 1.0)
            si = min(int(start_ratio * n_words), n_words - 1)
            ei = min(int(end_ratio * n_words), n_words - 1)
            ei = max(ei, si)
            si = max(si, prev_ei + 1)
            char_offset += s_len
            if si > ei:
                continue
            prev_ei = ei
            results.append({
                "text":  sent.strip(),
                "start": round(whisper_words[si]["start"], 3),
                "end":   round(whisper_words[ei]["end"],   3),
                "word_ids": list(range(si, ei + 1)),
            })
        return results

    def _map_time_ratio(
        self,
        sents: list[str],
        start: float,
        end:   float,
    ) -> list[dict]:
        """Proportional time estimate when no word timestamps are available."""
        units = [self._count_units(s) for s in sents]
        total = sum(units)
        if total == 0:
            return []
        duration = end - start
        results = []
        offset = 0
        for sent, u in zip(sents, units):
            ss = start + (offset / total) * duration
            se = ss + (u / total) * duration
            results.append({
                "text":  sent.strip(),
                "start": round(ss, 3),
                "end":   round(se, 3),
            })
            offset += u
        return results

    # ── public API ────────────────────────────────────────────────────────────

    def split(
        self,
        text:          str,
        start:         float,
        end:           float,
        whisper_words: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Split `text` into sentences and map each to a time range.

        Returns list of {"text", "start", "end"}.

        No pre-merge of short fragments is performed here.  All merging is
        deferred to merge_short() which has full timestamp context and can
        check inter-sentence gaps to avoid cross-speaker contamination.
        """
        sents = split_sentences(text, self.language)
        if not sents:
            return []

        # Filter empty strings (no pre-merge — merge_short handles it)
        sents = [s for s in sents if s.strip()]

        if whisper_words:
            if self.is_char_based:
                return self._map_char_ratio(sents, whisper_words)
            return self._map_word_index(sents, whisper_words)

        return self._map_time_ratio(sents, start, end)

    def merge_short(self, sentences: list[dict]) -> list[dict]:
        """
        Merge sentences that are too short (by duration or unit count)
        into adjacent neighbours, subject to gap constraints.

        Gap-aware merging
        ─────────────────
        A short sentence is merged into its neighbour ONLY when the gap
        between them is ≤ merge_gap_threshold_sec (default 0.15s).

        This prevents cross-speaker contamination in rapid exchanges:
          "...abbreviated to fam?"   [host, ends 179.4s]
          "Family."                  [contestant, starts 180.0s, gap=0.56s]
          "Correct."                 [host, starts 180.7s, gap=0.20s]

        With merge_gap_threshold_sec=0.15s, none of these merge — each remains
        a separate sentence.  "Family." gets its own embedding (or is
        skipped if too short and assigned via diarization).  The host
        cluster stays clean.

        With the old threshold of 0.5s, "Family." + "Correct." would merge,
        contaminating the embedding with both speakers.
        """
        if not sentences:
            return sentences

        merged = []
        sep = "" if self.is_char_based else " "
        i = 0

        while i < len(sentences):
            cur = dict(sentences[i])
            duration = cur["end"] - cur["start"]
            units = self._count_units(cur["text"])
            is_short = (
                duration < self.min_sentence_duration
                or units < self._min_units()
            )

            # Merge backwards into previous
            if is_short and merged:
                gap = cur["start"] - merged[-1]["end"]
                if gap <= self.merge_gap_threshold_sec:
                    prev = merged[-1]
                    merged[-1] = {
                        "text":        prev["text"] + sep + cur["text"],
                        "start":       prev["start"],
                        "end":         cur["end"],
                        "confidence":  round(
                            (prev.get("confidence", 1.0)
                             + cur.get("confidence", 1.0)) / 2, 3),
                        "words":       prev.get("words", [])
                                       + cur.get("words", []),
                    }
                    i += 1
                    continue

            # Merge forwards into next
            if is_short and i + 1 < len(sentences):
                nxt = sentences[i + 1]
                gap = nxt["start"] - cur["end"]
                if gap <= self.merge_gap_threshold_sec:
                    sentences[i + 1] = {
                        "text":        cur["text"] + sep + nxt["text"],
                        "start":       cur["start"],
                        "end":         nxt["end"],
                        "confidence":  round(
                            (cur.get("confidence", 1.0)
                             + nxt.get("confidence", 1.0)) / 2, 3),
                        "words":       cur.get("words", [])
                                       + nxt.get("words", []),
                    }
                    i += 1
                    continue

            merged.append(cur)
            i += 1

        return merged