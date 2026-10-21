# lib/model_registry.py
"""
Model registry — single source of truth for all model paths.

Called by:
  states/detect_language_download_models/lambda_function.py  — model S3 availability gate
  states/run_vad_model/lambda_function.py                    — PYANNOTE_MODEL paths
  states/run_speech_embedding_model/lambda_function.py       — PYANNOTE_MODEL paths
  states/run_whisper_align_wtpsplit_models/lambda_function.py — reads manifest built here
  lib/transcribe.py                                          — (no imports since the vocab-driven charset, 2026-10-12)

All model entries share three keys:
  cf_key  : path under the CloudFront origin  (opensource_models/…)
  s3_key  : path in the customer S3 bucket    (_model_artefacts/…)
  tmp_dir : extraction target in Lambda /tmp  (/tmp/models/…)

Whisper entries additionally carry:
  concurrency      : Step Functions Map MaxConcurrency for the transcription Map state
  lambda_memory_mb : Lambda memory for the transcription Lambda
  description      : human-readable label

Storage layout
──────────────
  CloudFront  : {CF_URL}/opensource_models/{model}/{variant}/model.tar.gz
  Customer S3 : s3://{bucket}/_model_artefacts/{model}/{variant}/model.tar.gz
  Lambda /tmp : /tmp/models/{model}/{variant}/

Models are seeded to customer S3 at CFN stack creation by CodeBuild.
Language-specific align and non-turbo Whisper models are seeded on-demand by
StateDetectLanguageDownloadModels (ensure_in_s3 → CloudFront fallback).

Adding a specialised Whisper model
───────────────────────────────────
  1. Add an entry to WHISPER_MODELS keyed by ISO-639-1 code.
  2. Follow the cf_key / s3_key / tmp_dir / concurrency / lambda_memory_mb convention.
  3. Add a dedicated entry to ALIGN_MODELS if one exists for that language.
  Nothing else changes — StateDetectLanguageDownloadModels reads the registry directly.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# PYANNOTE MODEL
# =============================================================================
# Used by:
#   StateRunVADModel              — segmentation.onnx
#   StateRunSpeechEmbeddingModel  — embedding.int8.onnx (or embedding.onnx)
#   StateDetectLanguageDownloadModels — ensures it exists in customer S3
# =============================================================================

PYANNOTE_MODEL: dict[str, str] = {
    "cf_key":  "opensource_models/pyannote/community-1/model.tar.gz",
    "s3_key":  "_model_artefacts/pyannote/community-1/model.tar.gz",
    "tmp_dir": "/tmp/models/pyannote/community-1",
}

# =============================================================================
# WTPSPLIT MODEL
# =============================================================================
# Used by:
#   StateRunWhisperAlignWtpsplitModels    — sentence splitting, all languages
#   StateDetectLanguageDownloadModels     — ensures it exists in customer S3
# =============================================================================

WTPSPLIT_MODEL: dict[str, str] = {
    "cf_key":  "opensource_models/wtpsplit/model.tar.gz",
    "s3_key":  "_model_artefacts/wtpsplit/model.tar.gz",
    "tmp_dir": "/tmp/models/wtpsplit",
}

# =============================================================================
# WHISPER MODEL REGISTRY
# =============================================================================
# Used by:
#   StateDetectLanguageDownloadModels  — turbo model for detect_language(); ensures
#                                        correct variant is in customer S3
#   StateRunWhisperAlignWtpsplitModels — loads model for transcription
# =============================================================================

WHISPER_MODELS: dict[str, dict[str, Any]] = {
    # ── Base: large-v3-turbo — fallback for all unlisted languages ────────────
    "turbo": {
        "cf_key":           "opensource_models/whisper/turbo/model.tar.gz",
        "s3_key":           "_model_artefacts/whisper/turbo/model.tar.gz",
        "tmp_dir":          "/tmp/models/whisper/turbo",
        "concurrency":      2,
        "lambda_memory_mb": 6144,
        "description":      "large-v3-turbo (multilingual)",
    },

    # ── Welsh: specialised large-v3 ───────────────────────────────────────────
    "cy": {
        "cf_key":           "opensource_models/whisper/cy/model.tar.gz",
        "s3_key":           "_model_artefacts/whisper/cy/model.tar.gz",
        "tmp_dir":          "/tmp/models/whisper/cy",
        "concurrency":      1,
        "lambda_memory_mb": 6144,
        "description":      "large-v3 Welsh-specialised",
    },

    # ── Arabic placeholder — uncomment when model URL is confirmed ────────────
    # "ar": {
    #     "cf_key":           "opensource_models/whisper/ar/model.tar.gz",
    #     "s3_key":           "_model_artefacts/whisper/ar/model.tar.gz",
    #     "tmp_dir":          "/tmp/models/whisper/ar",
    #     "concurrency":      1,
    #     "lambda_memory_mb": 6144,
    #     "description":      "large-v3 Arabic-specialised",
    # },
}

# =============================================================================
# ALIGNMENT MODEL REGISTRY
# =============================================================================
# Used by:
#   StateDetectLanguageDownloadModels      — ensures model exists in customer S3
#   StateRunWhisperAlignWtpsplitModels     — loads model for word-timestamp refinement
# =============================================================================

_ALIGN_CF = "opensource_models/align"
_ALIGN_S3 = "_model_artefacts/align"

ALIGN_MODELS: dict[str, dict[str, str]] = {
    lang: {
        "cf_key":  f"{_ALIGN_CF}/{lang}/model.tar.gz",
        "s3_key":  f"{_ALIGN_S3}/{lang}/model.tar.gz",
        "tmp_dir": f"/tmp/models/align/{lang}",
    }
    for lang in [
        "en", "cy", "ar", "zh", "ja", "ko",
        "fr", "de", "es", "pt", "nl", "it",
        "pl", "ru", "sv", "fi", "no",
        "cs", "sk", "ro", "tr", "el", "uk",
        "hi", "id", "ms",
        # "da" and "vi" REMOVED 2026-10-06 (license audit,
        # THIRD_PARTY_LICENSES.md): the Danish fine-tune carries
        # Danish Parliament no-derivatives terms and the Vietnamese
        # one is CC-BY-NC — both degrade to the English Latin-script
        # fallback like any other unsupported language.
    ]
}

# Languages where the English align model is an acceptable phoneme fallback.
# CJK and RTL excluded — an English acoustic model is meaningless for
# non-Latin scripts.
ENGLISH_ALIGN_FALLBACK = {
    "cy", "fr", "de", "es", "it", "pt", "nl", "pl", "ro",
    "sv", "da", "fi", "no", "cs", "sk", "tr", "el",
    "ca", "eu", "gl", "lv", "hr", "sl",
    "vi",   # Latin script; dedicated model removed 2026-10-06 (CC-BY-NC)
}

# Languages with no word-boundary spaces (character-based segmentation).
CHAR_BASED_LANGUAGES = {"zh", "ja", "ko", "th"}

# Languages for which the English align fallback is unsafe (non-Latin script).
NO_ENGLISH_ALIGN_FALLBACK = {
    "zh", "ja", "ko", "th", "ar", "fa", "he", "ur",
    "hi", "te", "ml", "bn", "ka", "am",
}

# =============================================================================
# PUBLIC API
# =============================================================================


def get_whisper_entry(lang: str) -> dict[str, Any]:
    """
    Return the WHISPER_MODELS entry for `lang`.
    Falls back to "turbo" for any language without a specialised model.
    Injects "_model_key" indicating which registry key resolved.
    """
    key = lang.lower() if lang.lower() in WHISPER_MODELS else "turbo"
    return {**WHISPER_MODELS[key], "_model_key": key}


def get_align_entry(lang: str) -> dict[str, str]:
    """
    Return the ALIGN_MODELS entry for `lang`.

    Fallback chain:
      1. Dedicated model for lang          → best accuracy
      2. English model (Latin-script only) → acceptable for European langs
      3. {}                                → skip alignment entirely
    """
    lang = lang.lower()
    if lang in ALIGN_MODELS:
        return ALIGN_MODELS[lang]
    if lang in ENGLISH_ALIGN_FALLBACK:
        return {**ALIGN_MODELS["en"], "_fallback": True}
    return {}


def resolve_manifest(lang: str, variant: str | None = None) -> dict[str, Any]:
    """
    Return the complete model manifest for a detected language.
    Consumed by StateDetectLanguageDownloadModels (S3 availability check)
    and StateRunWhisperAlignWtpsplitModels (load from /tmp).

    Parameters
    ──────────
    lang    : ISO-639-1 language code (drives alignment model selection)
    variant : Whisper model variant to use.  If None, resolved from lang
              via get_whisper_entry(lang) as before.  Pass an explicit
              value (e.g. "turbo", "cy") when the handler has already
              resolved the variant to avoid double-resolution.

    Shape:
    {
        "language": "cy",
        "whisper":  { cf_key, s3_key, tmp_dir, concurrency,
                      lambda_memory_mb, description, _model_key },
        "align":    { cf_key, s3_key, tmp_dir } | {},
        "pyannote": { cf_key, s3_key, tmp_dir },
        "wtpsplit": { cf_key, s3_key, tmp_dir },
    }
    """
    whisper_entry = (
        get_whisper_entry(variant)
        if variant and str(variant).strip()
        else get_whisper_entry(lang)
    )
    return {
        "language": lang.lower(),
        "whisper":  whisper_entry,
        "align":    get_align_entry(lang),
        "pyannote": PYANNOTE_MODEL,
        "wtpsplit": WTPSPLIT_MODEL,
    }


# NOT USED — no callers in lib/ or states/
def get_concurrency(lang: str) -> int:
    """Step Functions Map MaxConcurrency for the A3 Transcribe state."""
    return int(get_whisper_entry(lang).get("concurrency", 1))


# NOT USED — no callers in lib/ or states/
def get_lambda_memory_mb(lang: str) -> int:
    """Lambda memory in MB for the A3 Transcribe function."""
    return int(get_whisper_entry(lang).get("lambda_memory_mb", 6144))


def word_timing_source(align_entry: dict | None, language: str) -> str:
    """Provenance of this run's word timestamps (multilingual
    requirement, 2026-09-16 — promotes the silent alignment fallback
    to a recorded fact):

      aligned:<lang>       forced alignment with the language's model
      aligned:en_fallback  forced alignment with the ENGLISH model
                           (Latin-script fallback — degraded)
      whisper_native       no alignment ran; Whisper's own coarse
                           timestamps

    Consumers gate word-timing-dependent behavior on this (speech-span
    trimming, embedding slices, AD windows, DER scoring splits).
    """
    if not align_entry:
        return "whisper_native"
    if align_entry.get("fallback") or align_entry.get("_fallback"):
        return "aligned:en_fallback"
    return f"aligned:{language.lower()}"
