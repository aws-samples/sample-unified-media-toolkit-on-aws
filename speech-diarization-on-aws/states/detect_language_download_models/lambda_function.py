# detect_download_language_handler.py
"""
StateDetectLanguageDownloadModels  (3-cores, 5376 MB)

Two responsibilities in one Lambda invocation:

  1. DETECT LANGUAGE
     If input_config.language is set, skip detection.  Otherwise sample up to
     language_detection_num_probes evenly-spaced dynamic chunks as Whisper probes
     (turbo model, detect_language() only — no transcription), take a majority vote,
     and record the result.

  2. ENSURE MODELS IN CUSTOMER S3
     For the resolved language and model_variant: check whether each model archive
     exists in the customer S3 bucket; if not, stream it from CloudFront (cf_host)
     directly into S3 via ensure_in_s3().  Nothing is written to /tmp here.
     Models checked: Whisper variant, alignment (if applicable), wtpsplit.

  3. ENRICH ITERATORS
     Injects language, asr_cfg (with model_variant and model_manifest), and model_cfg
     into every dynamic-chunk iterator so the downstream Map states have everything
     they need without further S3 lookups.

Model variant resolution (priority order)
─────────────────────────────────────────
  1. asr.model_variant if explicitly set (non-null, non-empty)
  2. Language-specific variant from WHISPER_MODELS registry
  3. "turbo" fallback

Alignment model skip logic
──────────────────────────
  get_align_entry() returns {} for languages with no usable model (CJK, RTL, and
  languages in NO_ENGLISH_ALIGN_FALLBACK without a dedicated native model).  An empty
  dict → skip the S3 check entirely.  "_fallback": True entries (English model used
  for Latin-script European languages) are still downloaded.

cleanup_models() call strategy
───────────────────────────────
  Called at invocation start (clear previous job's /tmp) and again after language
  detection completes (free the turbo Whisper model before the potentially large
  variant + align model bodies are buffered in RAM during S3 put_object).

Lambda configuration
─────────────────────
  Memory  : 5376 MB (ThreeCores) — peak: one ~500 MB Whisper model body in RAM
  Timeout : 10 min  — worst case: detection 30 s + 3 missing models × ~2 min each
  Arch    : arm64
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback

from lib.language_detect import detect_language
from lib.model_registry import (
    ENGLISH_ALIGN_FALLBACK,
    WHISPER_MODELS,
    ALIGN_MODELS,
    get_align_entry,
    resolve_manifest,
)
from lib.model_store import cleanup_models, ensure_in_s3, ensure_in_tmp
from lib.ort_init import get_num_cpus


# ── Module logger with its own explicit handler ───────────────────────────────
# We attach a StreamHandler directly to the module logger and set
# log.propagate = False so Lambda's root logger cannot interfere.
# Lambda's runtime pre-configures the root logger at WARNING and may
# reset it between warm invocations — bypassing it entirely is the
# only reliable way to guarantee DEBUG records reach CloudWatch.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False

_handler = logging.StreamHandler()
# handler is always permissive; logger is the gate
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
if not log.handlers:               # guard against duplicate handlers on warm containers
    log.addHandler(_handler)


# =============================================================================
# HELPERS
# =============================================================================

def _progress(step: str, detail: str = "", t_start: float = 0.0) -> None:
    """Emit a structured progress line visible in CloudWatch."""
    elapsed = f"  (+{time.monotonic() - t_start:.2f}s)" if t_start else ""
    log.info(f"  ▶ {step:<30s}  {detail}{elapsed}")


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

def _select_probe_chunks(
    iterators: list[dict],
    asr_cfg:   dict,
) -> list[dict]:
    """
    Select up to num_probes evenly-spaced dynamic chunks that meet the
    minimum speech duration gate.

    Dynamic chunks are already speech-dense, so we use their fetch windows
    as probes directly instead of re-reading VAD segments files.

    If fewer qualifying chunks exist than num_probes, all qualifying chunks
    are returned.  If none meet the gate, all chunks are used as a fallback.
    """
    num_probes = asr_cfg["language_detection_num_probes"]
    min_probe_sec = asr_cfg["language_detection_min_sec"]

    qualifying = [
        c for c in iterators
        if float(c.get("speech_sec", 0.0)) >= min_probe_sec
    ]
    if not qualifying:
        log.warning(
            f"No chunks with speech_sec >= {min_probe_sec}s — "
            f"falling back to all {len(iterators)} chunks"
        )
        qualifying = list(iterators)

    n = len(qualifying)
    if n <= num_probes:
        return qualifying

    # Evenly-spaced linspace indices, deduplicated while preserving order
    indices = [round(i * (n - 1) / (num_probes - 1))
               for i in range(num_probes)]
    seen: set[int] = set()
    probes = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            probes.append(qualifying[idx])
    return probes


def _run_language_detection(
    audio_bucket:  str,
    audio_key:     str,
    model_cfg:     dict,
    asr_cfg:       dict,
    iterators:     list[dict],
    memory_mb:     int | None,
    t0:            float,
) -> dict:
    """
    Download Whisper turbo model to /tmp and run majority-vote language detection
    across evenly-spaced dynamic chunk probes.

    Returns a dict with: language, confidence, method, num_probes_used,
    probe_results.
    """

    model_bucket = model_cfg.get("model_bucket") or os.environ["MODEL_BUCKET"]
    model_prefix = model_cfg.get(
        "model_prefix") or os.environ["MODEL_ARTEFACTS_PREFIX"]

    base_entry = WHISPER_MODELS["turbo"]

    # Apply model_prefix override — same pattern as vad_handler / embedding_handler
    s3_key = base_entry["s3_key"]
    if model_prefix and not s3_key.startswith(model_prefix):
        s3_key = model_prefix.strip("/") + "/" + s3_key.split("/", 1)[1]

    _progress("detect: downloading Whisper turbo",
              f"s3://{model_bucket}/{s3_key}", t_start=t0)
    ensure_in_tmp(
        bucket=model_bucket,
        s3_key=s3_key,
        tmp_dir=base_entry["tmp_dir"],
        label="whisper/turbo",
    )

    probe_chunks = _select_probe_chunks(iterators=iterators, asr_cfg=asr_cfg)
    min_speech = asr_cfg["language_detection_min_sec"]
    log.info(
        f"  {len(probe_chunks)} probe chunk(s) selected from "
        f"{len(iterators)} iterators  (min_speech={min_speech}s)"
    )

    _progress("detect: running Whisper probes",
              f"{len(probe_chunks)} probes  audio=s3://{audio_bucket}/{audio_key}",
              t_start=t0)
    result = detect_language(
        audio_bucket=audio_bucket,
        audio_key=audio_key,
        probes=probe_chunks,
        whisper_model_path=base_entry["tmp_dir"],
        asr_cfg=asr_cfg,
        cpu_threads=get_num_cpus(memory_mb),
    )

    log.info(
        f"  detected language='{result['language']}'  "
        f"confidence={result.get('confidence', 0):.3f}  "
        f"method={result.get('method', '?')}  "
        f"probes_used={result.get('num_probes_used', '?')}"
    )
    return result


# =============================================================================
# MODEL VARIANT RESOLUTION
# =============================================================================

def _resolve_model_variant(language: str, requested_variant: str | None) -> str:
    """
    Resolve the Whisper model variant to use for transcription.

    Priority:
      1. requested_variant if explicitly set (non-null, non-empty string)
      2. Language-specific variant if one exists in WHISPER_MODELS
      3. "turbo" fallback

    Examples:
      language=cy, requested=None    → "cy"   (dedicated Welsh model)
      language=en, requested=None    → "turbo" (no dedicated English variant)
      language=cy, requested="turbo" → "turbo" (explicit override respected)
    """
    if requested_variant and str(requested_variant).strip():
        variant = str(requested_variant).strip()
        log.info(f"  model_variant: using explicitly requested '{variant}'")
        return variant

    if language in WHISPER_MODELS:
        log.info(f"  model_variant: resolved '{language}' from WHISPER_MODELS")
        return language

    log.info(f"  model_variant: no variant for '{language}' — using 'turbo'")
    return "turbo"


# =============================================================================
# MODEL AVAILABILITY GATE
# =============================================================================

def _ensure_models(
    model_cfg: dict,
    language:  str,
    variant:   str,
    t0:        float,
) -> tuple[dict, int, int]:
    """
    Ensure whisper (variant), align (if applicable), and wtpsplit models
    are present in the customer S3 bucket.  Missing models are streamed
    from CloudFront (cf_host) directly into S3 — nothing touches /tmp.

    Alignment model skip logic:
      get_align_entry() returns {} for languages where no suitable model
      exists (CJK, RTL, and languages in NO_ENGLISH_ALIGN_FALLBACK that
      have no dedicated native model).  An empty dict means skip.

      For languages where get_align_entry() returns an entry with
      "_fallback": True, the English alignment model is used — this is
      acceptable for Latin-script European languages.

    Returns:
      (clean_manifest, num_checked, num_copied)

    clean_manifest shape:
      {
        "whisper": { "s3_key": "...", "tmp_dir": "..." },
        "align":   { "s3_key": "...", "tmp_dir": "..." },   ← {} if skipped
        "wtpsplit":{ "s3_key": "...", "tmp_dir": "..." },
      }
    """
    manifest = resolve_manifest(language, variant)

    # align: {} means get_align_entry() found no suitable model — skip entirely.
    # When the user explicitly chose a non-native variant (e.g. variant="turbo"
    # for language="cy"), redirect alignment to English now rather than
    # discovering at download time that the language-specific model isn't seeded.
    align_entry = manifest.get("align") or None
    # Only redirect alignment when the variant is non-native AND the language
    # does not have its own alignment model (i.e. it would normally fall back
    # to English anyway). Languages with a native align model (e.g. "en", "fr")
    # keep their own model regardless of which Whisper variant was chosen.
    lang_lower = language.lower()
    if align_entry and variant and variant != lang_lower and lang_lower not in ALIGN_MODELS:
        if lang_lower in ENGLISH_ALIGN_FALLBACK:
            align_entry = {**get_align_entry("en"), "_fallback": True}
            log.info(
                f"  align: variant='{variant}' is non-native for '{language}' "
                f"— redirecting to English alignment model"
            )
        else:
            align_entry = None
            log.info(
                f"  align: variant='{variant}' is non-native for '{language}' "
                f"and no English fallback is safe — skipping alignment"
            )

    checks: list[tuple[str, dict | None]] = [
        ("whisper",  manifest.get("whisper")),
        ("align",    align_entry),
        ("wtpsplit", manifest.get("wtpsplit")),
    ]

    num_checked = 0
    num_copied = 0
    clean: dict[str, dict] = {}

    model_bucket = model_cfg.get("model_bucket") or os.environ["MODEL_BUCKET"]
    model_prefix = model_cfg.get(
        "model_prefix") or os.environ["MODEL_ARTEFACTS_PREFIX"]
    cf_host = model_cfg.get("cf_host") or os.environ["CLOUDFRONT_URL"]

    for label, entry in checks:
        if not entry:
            log.info(
                f"  [{label}] skipped — no model for language '{language}'")
            clean[label] = {}
            continue

        # Apply model_prefix override — same pattern as vad_handler / embedding_handler
        s3_key = entry["s3_key"]
        if model_prefix and not s3_key.startswith(model_prefix):
            s3_key = model_prefix.strip("/") + "/" + s3_key.split("/", 1)[1]

        # Build full CloudFront URL from cf_host and the registry cf_key
        cf_relative = entry.get("cf_key", entry["s3_key"].split("/", 1)[-1])
        cf_url = f"{cf_host}/{cf_relative}"

        _progress(f"models: checking [{label}]",
                  f"s3://{model_bucket}/{s3_key}", t_start=t0)
        try:
            outcome = ensure_in_s3(
                bucket=model_bucket,
                s3_key=s3_key,
                cf_url=cf_url,
                label=label,
            )
        except RuntimeError as exc:
            # Alignment model not on CloudFront — fall back to English if safe,
            # otherwise skip. Never fall back for whisper or wtpsplit.
            if label == "align" and language.lower() in ENGLISH_ALIGN_FALLBACK:
                en_entry = get_align_entry("en")
                en_s3_key = en_entry["s3_key"]
                if model_prefix and not en_s3_key.startswith(model_prefix):
                    en_s3_key = (
                        model_prefix.strip("/") + "/"
                        + en_s3_key.split("/", 1)[1]
                    )
                en_cf_url = f"{cf_host}/{en_entry['cf_key']}"
                log.warning(
                    f"  [align] download failed ({exc}) — "
                    f"falling back to English alignment model"
                )
                outcome = ensure_in_s3(
                    bucket=model_bucket,
                    s3_key=en_s3_key,
                    cf_url=en_cf_url,
                    label="align/en-fallback",
                )
                s3_key = en_s3_key
                entry = {**en_entry, "_fallback": True}
            elif label == "align":
                log.warning(
                    f"  [align] download failed and no English fallback is safe "
                    f"for '{language}' — skipping alignment.  ({exc})"
                )
                clean[label] = {}
                continue
            else:
                raise

        num_checked += 1
        if outcome == "copied":
            num_copied += 1
        log.info(f"  [{label}] {outcome}  s3://{model_bucket}/{s3_key}")

        clean[label] = {
            "s3_key":  s3_key,
            "tmp_dir": entry["tmp_dir"],
        }
        # preserve the fallback fact — downstream provenance
        # (word_timing_source) must be able to tell "aligned with the
        # language's own model" from "aligned with the English model"
        if label == "align" and entry.get("_fallback"):
            clean[label]["fallback"] = True

    return clean, num_checked, num_copied


# =============================================================================
# ITERATOR ENRICHMENT
# =============================================================================

def _enrich_iterators(
    iterators:  list[dict],
    language:   str,
    asr_cfg:    dict,
    model_cfg:  dict,
) -> list[dict]:
    """
    Return a new list with language, asr_cfg, model_manifest, model_bucket,
    and model_prefix injected into every dynamic chunk iterator item.

    The downstream Map state passes each item directly as item_data to the
    Whisper Lambda — no further S3 lookups needed.
    """
    return [
        {
            **item,
            "language":  language,
            "asr_cfg": {
                **item.get("asr_cfg", {}),
                **asr_cfg,
            },
            "model_cfg": {
                **item.get("model_cfg", {}),
                **model_cfg,
            },
        }
        for item in iterators
    ]


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    # Always wipe /tmp at invocation start. This state downloads turbo for
    # language detection then unconditionally wipes again before S3 model
    # availability checks — no warm-cache reuse opportunity here.
    cleanup_models()

    log.setLevel(logging.INFO)
    log.info("StateDetectLanguageDownloadModels event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Unpack event ──────────────────────────────────────────────────────
        data = event["data"]

        audio_bucket = data["audio_bucket"]
        audio_key = data["audio_key"]
        output_bucket = data["output_bucket"]
        output_prefix = data["output_prefix"]

        if not (audio_bucket and audio_key and output_bucket and output_prefix):
            raise ValueError(
                f"data.audio_bucket, data.audio_key, data.output_bucket, data.output_prefix are required "
                f"(got audio_bucket={audio_bucket!r}, audio_key={audio_key!r})"
                f"(got output_bucket={output_bucket!r}, output_prefix={output_prefix!r})"
            )

        model_cfg = data["model_cfg"]
        asr_cfg = data["asr_cfg"]

        iterators: list[dict] = data["iterators"]
        if not iterators:
            raise ValueError("data.iterators is empty — nothing to process")

        language = (data.get("language") or "").strip().lower() or None
        model_variant = (asr_cfg.get("model_variant") or "").strip() or None

        memory_mb: int | None = getattr(context, "memory_limit_in_mb", None)
        if memory_mb:
            memory_mb = int(memory_mb)

        log.info(
            f"  language_input='{language or 'auto-detect'}'  "
            f"model_variant_input='{model_variant or 'auto'}'  "
            f"iterators={len(iterators)}"
        )
        _progress("START", f"iterators={len(iterators)}", t_start=t0)

        # ── Step 1: Determine language ────────────────────────────────────────
        if language:
            # User provided the language — skip detection entirely
            language_confidence = 1.0
            language_method = "user_provided"
            language_num_probes = 0
            probe_results: list = []

            log.info(
                f"  language='{language}' supplied by caller — skipping detection")
            _progress("language: user-provided", f"'{language}'", t_start=t0)

        else:
            # Auto-detect from Whisper probes over dynamic chunks
            _progress("language: auto-detecting", "", t_start=t0)
            detect_result = _run_language_detection(
                audio_bucket=audio_bucket,
                audio_key=audio_key,
                model_cfg=model_cfg,
                asr_cfg=asr_cfg,
                iterators=iterators,
                memory_mb=memory_mb,
                t0=t0,
            )

            language = detect_result["language"]
            language_confidence = float(detect_result.get("confidence", 0.0))
            language_method = detect_result.get(
                "method", "auto_detect")
            language_num_probes = int(detect_result.get("num_probes_used", 0))
            probe_results = detect_result.get("probe_results", [])

            # Whisper turbo is no longer needed — wipe /tmp before model downloads
            # to free RAM and /tmp space before the (potentially large) Whisper
            # variant body is buffered in RAM during the S3 put_object calls.
            cleanup_models()
            _progress(
                "detect: /tmp cleared",
                f"language='{language}'  confidence={language_confidence:.3f}",
                t_start=t0,
            )

        # ── Step 2: Resolve model variant ─────────────────────────────────────
        model_variant = _resolve_model_variant(
            language=language, requested_variant=model_variant)
        log.info(f"  resolved model_variant='{model_variant}'")
        _progress("models: variant resolved", f"'{model_variant}'", t_start=t0)

        # ── Step 3: Ensure models present in customer S3 ─────────────────────
        model_manifest, num_checked, num_copied = _ensure_models(
            model_cfg=model_cfg,
            language=language,
            variant=model_variant,
            t0=t0,
        )
        # word-timing provenance: promotes the previously SILENT alignment
        # fallback to a recorded fact (multilingual requirement,
        # 2026-09-16). Word timestamps come from forced alignment where a
        # model exists, else are Whisper-native and coarse — consumers
        # (span trimming, embedding slices, AD windows) gate on this.
        from lib.model_registry import word_timing_source as _wts
        word_timing_source = _wts(model_manifest.get("align"), language)
        log.info(f"  word_timing_source='{word_timing_source}'")

        log.info(
            f"  models: checked={num_checked}  copied={num_copied}  "
            f"present={[k for k, v in model_manifest.items() if v]}"
        )
        _progress("models: all present",
                  f"checked={num_checked}  copied={num_copied}", t_start=t0)

        # ── Step 4: Enrich iterators ──────────────────────────────────────────
        # Store model_variant and model_manifest to asr_cfg as it is specific to asr processing
        asr_cfg["model_variant"] = model_variant
        asr_cfg["model_manifest"] = model_manifest

        _progress("iterators: enriching",
                  f"{len(iterators)} items  language='{language}'", t_start=t0)
        enriched_iterators = _enrich_iterators(
            iterators=iterators,
            language=language,
            asr_cfg=asr_cfg,
            model_cfg=model_cfg,
        )
        log.info(f"  enriched {len(enriched_iterators)} iterator items")

        # ── Build output ──────────────────────────────────────────────────────
        elapsed_sec_langdetect = round(
            time.monotonic() - t0, 2) if language_method != "user_provided" else 0.0

        output = {
            **event,
            "data": {
                **data,
                "language":                 language,
                "language_confidence":      round(language_confidence, 4),
                "language_method":          language_method,
                "word_timing_source":       word_timing_source,
                "language_num_probes":      language_num_probes,
                "elapsed_sec_langdetect":   elapsed_sec_langdetect,
                "language_probe_results":   probe_results if probe_results else None,
                # "model_variant":            model_variant,
                # "models_checked":           num_checked,
                # "models_copied":            num_copied,
                "asr_cfg":                  asr_cfg,
                "iterators":                enriched_iterators,
            }
        }

        log.info("StateDetectLanguageDownloadModels output keys = %s",
                 sorted(output.keys()))
        _progress(
            "DONE",
            f"language='{language}'  variant='{model_variant}'  "
            f"iterators={len(enriched_iterators)}  elapsed={elapsed_sec_langdetect}s",
            t_start=t0,
        )
        return output

    except Exception as exc:
        log.error(
            "[ERR] StateDetectLanguageDownloadModels — %s: %s",
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise
