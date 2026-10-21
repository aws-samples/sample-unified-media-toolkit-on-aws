"""
ASR-roundtrip QC gate (README.md §8) — synthesize → transcribe →
fuzzy-compare against the requested text.

Model: Systran/faster-whisper-base — the bare multilingual `base`, NOT
`base.en` (verified: two genuinely distinct repos; the .en variant is an
English-only distillation that silently breaks every non-English QC
check). Loaded via the same model_store S3→/tmp mechanism as everything
else.

Also used at voice-registration time to verify the caller's reference
transcript matches the reference audio (a mismatch doesn't error — it
silently corrupts every synthesis against that voice; found the hard way
during local verification).

Similarity metric — two real bugs found and fixed while registering the
default catalog (46 real references, not synthetic test cases):

1. Character-level SequenceMatcher.ratio() on long strings is NOT robust
   to a single early character-count shift (e.g. "fine clear" vs
   "fineclear" — one dropped space from a hyphen-to-nothing whisper
   quirk). One early 1-character desync drags a single "replace" block
   across the ENTIRE rest of the string, tanking the score even when
   99% of the words match. Measured: a LibriSpeech clip differing by
   two words out of ~45 scored 0.567 (below the 0.70 register
   threshold) on/character-level comparison; the same pair scores 0.925
   at word-level. Fixed by tokenizing on whitespace before diffing —
   but ONLY for scripts that actually use whitespace word boundaries
   (see _tokenize below); CJK text has no such boundaries and
   word-level diffing on a single CJK "word" (the whole string) would
   be strictly worse than character-level.
2. Whisper transcribes Chinese in whatever script variant it infers
   (often Traditional) regardless of the reference transcript's script.
   A perfectly correct FLEURS Simplified-Chinese transcript then scores
   ~0.6 against a Traditional-script transcription of IDENTICAL
   content. Fixed by normalizing both sides through OpenCC's t2s
   (Traditional-to-Simplified) conversion before comparing — a no-op
   for already-Simplified text, so this is safe to always apply.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except ImportError:  # pragma: no cover - opencc optional at import time
    _T2S = None

# Unicode ranges where individual characters are the natural comparison
# unit (no whitespace word boundaries) — CJK unified ideographs plus the
# common extension block, AND Japanese kana. Whisper's Chinese/Japanese
# output is character-dense with no spaces, so word-tokenizing would
# collapse the whole string into one token and make comparison strictly
# worse.
#
# Kana are not optional here: Japanese sentences can be entirely kana with
# no ideograph at all. Measured with the ideograph-only pattern,
# _tokenize("こんにちはせかい") returned ONE token (the whole string) and
# similarity("ありがとうございます", "ありがとございます") scored 0.0 —
# a single dropped character read as total content failure, which is
# exactly the collapse this branch exists to prevent.
#   3040-309F hiragana, 30A0-30FF katakana, 31F0-31FF katakana phonetic
#   extensions, FF66-FF9D halfwidth katakana
_CJK_RE = re.compile(
    r"[一-鿿㐀-䶿"
    r"぀-ゟ゠-ヿㇰ-ㇿｦ-ﾝ]")

# Pin the exact repo id — a one-word mistake here (base.en) silently breaks
# non-English QC without ever raising an error.
WHISPER_REPO_ID = "Systran/faster-whisper-base"

DEFAULT_PASS_THRESHOLD = 0.80      # synthesis QC — coarse content check
DEFAULT_REGISTER_THRESHOLD = 0.70  # register-voice transcript check (looser:
                                   # whisper drift on names/rare words is
                                   # expected; catching wholesale mismatch
                                   # is the goal, not exactness)

_model = None
_model_dir = None


def _get_model(model_dir: str):
    """Cached WhisperModel, keyed on model_dir — a warm container asked for a
    different directory must not silently get the previously-loaded model
    back. Only one directory is in use today (WHISPER_TMP_DIR), so this never
    reloads in practice; it just stops the cache from lying if that changes."""
    global _model, _model_dir
    if _model is None or _model_dir != model_dir:
        from faster_whisper import WhisperModel
        _model = WhisperModel(model_dir, device="cpu", compute_type="int8")
        _model_dir = model_dir
    return _model


def normalize_language(language: str | None) -> str | None:
    """Map a caller-supplied language to a whisper code, or None to let
    whisper auto-detect.

    Necessary because `language` reaches us from an LLM through the gateway's
    register_voice InputSchema, which declares it as free-form prose ("Language
    of the reference audio") — CFN SchemaDefinition has no enum, so nothing
    upstream can constrain it. faster-whisper does not tolerate a near-miss:
    Tokenizer.__init__ raises ValueError("'%s' is not a valid language code")
    for anything outside its 100-entry _LANGUAGE_CODES tuple, so "English",
    "en-US" or "eng" would surface as a raw internal traceback from a state
    that has already downloaded a model and read the audio.

    Codes are checked against faster_whisper's own tuple rather than a copy
    kept here, so this cannot drift from the installed library. An
    unrecognized value degrades to auto-detect (with a warning) instead of
    failing: the whole point of passing `language` is a modest accuracy gain
    on a known-language clip, and losing that gain beats losing the run."""
    if not language:
        return None
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    code = language.strip().lower().replace("_", "-")
    if code in _LANGUAGE_CODES:
        return code
    # "en-US" / "zh-CN" → primary subtag
    base = code.split("-", 1)[0]
    if base in _LANGUAGE_CODES:
        return base
    log.warning("language %r is not a whisper language code — falling back to "
                "auto-detect", language)
    return None


def transcribe(wav_path: str, model_dir: str, language: str | None = None) -> dict:
    model = _get_model(model_dir)
    language = normalize_language(language)
    segments, info = model.transcribe(wav_path, beam_size=5, language=language)
    text = "".join(seg.text for seg in segments).strip()
    return {"text": text, "language": info.language,
            "language_probability": round(float(info.language_probability), 3)}


def _normalize(text: str) -> str:
    """Lowercase, script-normalize (Traditional->Simplified Chinese, a
    no-op for non-Chinese or already-Simplified text), strip
    punctuation/extra whitespace — compare content, not formatting."""
    if _T2S is not None:
        text = _T2S.convert(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    """Word tokens for whitespace-delimited scripts; individual
    characters for CJK text, which has no whitespace word boundaries
    (see module docstring for why word-splitting CJK would be worse,
    not better)."""
    if _CJK_RE.search(text):
        return list(text.replace(" ", ""))
    return text.split()


def similarity(expected: str, actual: str) -> float:
    a, b = _normalize(expected), _normalize(actual)
    return SequenceMatcher(None, _tokenize(a), _tokenize(b)).ratio()


def verify_content(wav_path: str, expected_text: str, model_dir: str,
                   language: str | None = None,
                   threshold: float = DEFAULT_PASS_THRESHOLD) -> dict:
    """Transcribe wav and fuzzy-match against expected_text.
    Returns pass/fail + evidence — never raises on content mismatch
    (the state machine decides what a failure means)."""
    result = transcribe(wav_path, model_dir, language=language)
    score = similarity(expected_text, result["text"])
    passed = score >= threshold
    if not passed:
        # score + lengths only — the full expected/transcribed texts are
        # already in the qc evidence fields of the output record; script
        # text does not belong in a log group
        log.warning("QC FAIL (%.2f < %.2f): expected %d chars, got %d",
                    score, threshold, len(expected_text), len(result["text"]))
    return {
        "passed": passed,
        "similarity": round(score, 3),
        "threshold": threshold,
        "expected": expected_text,
        "transcribed": result["text"],
        "detected_language": result["language"],
    }
