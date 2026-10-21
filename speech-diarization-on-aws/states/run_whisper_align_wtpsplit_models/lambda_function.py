# run_whisper_align_wtpsplit_model_handle.py
"""
StateRunWhisperAlignWtpsplitModels  (Map — one Lambda per dynamic chunk, 7168 MB / 4-cores)

Downloads Whisper, alignment (optional), and wtpsplit models to /tmp, fetches the
audio window from S3, transcribes via batched Whisper + alignment, and runs
sentence splitting once on the full word sequence.

Speech groups
─────────────
  The dynamic_segments_{n}.json file pre-groups temporally close VAD turns into
  speech groups.  For each group, the handler slices the contiguous real audio
  (including natural intra-group silences) and feeds it to build_batches /
  batch_worker.  Audio is the real signal — no synthetic padding within groups.

Batching
─────────
  build_batches() packs groups into ≤ 28 s wall-clock batches.  Each batch is
  processed by one batch_worker() call, which submits a single
  WhisperModel.transcribe() call on the concatenated audio and assigns words back
  to their owning groups via soft time boundaries.

Sentence splitting
──────────────────
  split_words_into_sentences() runs ONCE after all batches complete, on the full
  flat word list.  This gives wtpsplit visibility across speech group and batch
  boundaries, preventing the trailing-fragment problem.

Worker pool
───────────
  ThreadPoolExecutor with num_workers threads.  CTranslate2 and onnxruntime
  release the GIL during inference, giving genuine N-core parallelism.

Output enriches the iterator item:
  transcript_key  — S3 key of transcript_{dyn_chunk_idx}.json
  num_sentences
  status          — "COMPLETED"
  elapsed_sec

transcript_{n}.json shape:
  {
    "dyn_chunk_idx": int, "language": str,
    "fetch_start_sec": float, "fetch_end_sec": float,
    "num_sentences": int, "num_words": int, "wtpsplit_sec": float,
    "sentences": [{"text", "start", "end", "confidence", "words": [...]}],
    "worker_stats": [...]
  }

Lambda configuration
─────────────────────
  Memory  : 7168 MB (FourCores)
  Timeout : 900 s
  Arch    : arm64
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.model_store import cleanup_models, ensure_in_tmp, find_file, write_models_cache_key
from lib.wav_utils import TARGET_SR, get_wav_info, read_wav_window
from lib.transcribe import (
    batch_worker, build_batches, BATCH_TARGET_SEC,
    rehome_straddle_strays, split_words_into_sentences, worker_init,
)
from lib.s3_utils import s3 as _s3, get_json, put_json

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


# =============================================================================
# HELPERS
# =============================================================================

def _progress(step: str, detail: str = "", t_start: float = 0.0) -> None:
    elapsed = f"  (+{time.monotonic() - t_start:.2f}s)" if t_start else ""
    log.info(f"  ▶ {step:<30s}  {detail}{elapsed}")


# =============================================================================
# MODEL DOWNLOAD
# =============================================================================

def _download_models(
    model_bucket:   str,
    model_manifest: dict,
    t0:             float,
) -> tuple[str, str, str, str]:
    """
    Ensure whisper, align (optional), and wtpsplit models are extracted to /tmp.
    Downloads all three archives concurrently — they write to separate /tmp dirs.

    Returns:
        (whisper_tmp_dir, align_model_path, vocab_path, wtpsplit_tmp_dir)

    align_model_path and vocab_path are empty strings when no align model
    is available — worker_init() treats "" as "no alignment".
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    whisper  = model_manifest["whisper"]
    align    = model_manifest.get("align", {})
    wtpsplit = model_manifest["wtpsplit"]

    has_align    = bool(align.get("s3_key") and align.get("tmp_dir"))
    is_fallback  = align.get("_fallback", False) if has_align else False
    align_label  = "align (en fallback)" if is_fallback else "align"

    log.info(
        "  downloading models concurrently: whisper + %s + wtpsplit",
        align_label if has_align else "no-align",
    )

    def _dl_whisper():
        _progress("model: whisper",
                  f"s3://{model_bucket}/{whisper['s3_key']}", t_start=t0)
        ensure_in_tmp(
            bucket=model_bucket,
            s3_key=whisper["s3_key"],
            tmp_dir=whisper["tmp_dir"],
            label="whisper",
        )

    def _dl_align():
        if not has_align:
            return
        _progress(f"model: {align_label}",
                  f"s3://{model_bucket}/{align['s3_key']}", t_start=t0)
        ensure_in_tmp(
            bucket=model_bucket,
            s3_key=align["s3_key"],
            tmp_dir=align["tmp_dir"],
            label=align_label,
        )

    def _dl_wtpsplit():
        _progress("model: wtpsplit",
                  f"s3://{model_bucket}/{wtpsplit['s3_key']}", t_start=t0)
        ensure_in_tmp(
            bucket=model_bucket,
            s3_key=wtpsplit["s3_key"],
            tmp_dir=wtpsplit["tmp_dir"],
            label="wtpsplit",
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_dl_whisper):  "whisper",
            pool.submit(_dl_align):    align_label,
            pool.submit(_dl_wtpsplit): "wtpsplit",
        }
        for fut in as_completed(futures):
            label = futures[fut]
            exc = fut.exception()
            if exc:
                raise RuntimeError(f"model download failed [{label}]: {exc}") from exc

    whisper_tmp_dir = whisper["tmp_dir"]

    align_model_path = ""
    vocab_path = ""
    if has_align:
        try:
            align_model_path = find_file(align["tmp_dir"], "model_quantized.onnx")
            vocab_path = find_file(align["tmp_dir"], "vocab.json")
            log.info(
                f"  align model ({'fallback=en' if is_fallback else 'native'})"
                f": {align_model_path}")
        except FileNotFoundError as exc:
            log.warning(
                f"  align model file not found: {exc} — skipping alignment")
    else:
        log.info(
            "  align: no suitable model for this language — "
            "Whisper timestamps used as-is")

    wtpsplit_tmp_dir = wtpsplit["tmp_dir"]

    return whisper_tmp_dir, align_model_path, vocab_path, wtpsplit_tmp_dir


# =============================================================================
# SPEECH GROUP LOADING
# =============================================================================

def _load_speech_groups(bucket: str, dyn_segment_key: str) -> list[dict]:
    """Read speech groups from a dynamic_segments_{n}.json file."""
    log.debug("  reading s3://%s/%s", bucket, dyn_segment_key)
    return get_json(bucket, dyn_segment_key)["speech_groups"]


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def _make_cache_key(model_manifest: dict) -> str:
    """Build a cache key from the whisper and align s3_keys."""
    whisper_key = model_manifest.get("whisper", {}).get("s3_key", "")
    align_key   = model_manifest.get("align",   {}).get("s3_key") or "none"
    return f"{whisper_key}|{align_key}"


def lambda_handler(event: dict, context) -> dict:
    log.setLevel(logging.INFO)
    log.info("StateRunWhisperAlignWtpsplitModels event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Unpack item_data ───────────────────────────────────────────────────
        item: dict = event.get("item_data", event)

        required = {
            "dyn_chunk_idx", "fetch_start_sec", "fetch_end_sec",
            "dyn_segment_key",
            "audio_bucket", "audio_key", "output_bucket", "output_prefix",
            "language", "asr_cfg", "audio_cfg", "model_cfg",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"Missing required item_data fields: {sorted(missing)}")

        dyn_chunk_idx:    int = int(item["dyn_chunk_idx"])
        fetch_start_sec:  float = float(item["fetch_start_sec"])
        fetch_end_sec:    float = float(item["fetch_end_sec"])
        fetch_duration:   float = fetch_end_sec - fetch_start_sec
        dyn_segment_key:  str = item["dyn_segment_key"]

        language:         str = item["language"].lower()

        audio_bucket:     str = item["audio_bucket"]
        audio_key:        str = item["audio_key"]
        output_bucket:    str = item["output_bucket"]
        output_prefix:    str = item["output_prefix"]

        # ── asr config ────────────────────────────────────────────────────────────
        asr_cfg:        dict = item["asr_cfg"]
        model_manifest: dict = asr_cfg["model_manifest"]
        num_workers:    int = int(asr_cfg["num_workers"])
        beam_size:      int = int(asr_cfg["beam_size"])
        initial_prompt: str = str(asr_cfg["initial_prompt"])
        merge_gap_threshold_sec: float = float(
            asr_cfg["merge_gap_threshold_sec"])
        condition_on_previous_text: bool = bool(
            asr_cfg.get("condition_on_previous_text", True))
        no_speech_threshold: float = float(
            asr_cfg.get("no_speech_threshold", 0.6))
        log_prob_threshold: float = float(
            asr_cfg.get("log_prob_threshold", -1.0))

        # ── audio config ─────────────────────────────────────────────────────────
        audio_cfg:      dict = item["audio_cfg"]
        # read_wav_window returns TARGET_SR audio regardless of the
        # file's rate, so ALL sample math on fetched buffers uses the
        # PIPELINE rate. (2026-09-29: slicing at audio_cfg's file rate
        # silently dropped every speech group past 1/3 of each fetch
        # window when a 48 kHz WAV reached this state; the routing fix
        # converts such files first — this is defense in depth.)
        sample_rate:    int = TARGET_SR
        file_sr = int(audio_cfg["sample_rate"])
        if file_sr != TARGET_SR:
            log.warning(
                "audio_cfg sample_rate=%d != pipeline %d — fetched "
                "buffers are resampled to the pipeline rate", file_sr,
                TARGET_SR)

        # ── model config ─────────────────────────────────────────────────────────
        model_cfg:      dict = item["model_cfg"]

        log.info(
            "  Chunk %d: fetch=[%.1fs, %.1fs]  (%.1fs)  lang=%s  beam=%d",
            dyn_chunk_idx, fetch_start_sec, fetch_end_sec,
            fetch_duration, language, beam_size,
        )
        _progress("START",
                  f"chunk={dyn_chunk_idx}  fetch={fetch_duration:.0f}s  "
                  f"lang={language}", t_start=t0)

        # ── Step 1: Download models ───────────────────────────────────────────
        model_bucket = model_cfg.get("model_bucket", os.environ["MODEL_BUCKET"])
        cache_key = _make_cache_key(model_manifest)
        cache_hit = cleanup_models(cache_key)

        if cache_hit:
            _progress("models: cache hit", f"key={cache_key}", t_start=t0)
            # Models already in /tmp — resolve paths without re-downloading.
            # If any expected file is missing (e.g. OS reclaimed /tmp under
            # memory pressure), invalidate the cache and re-download.
            whisper  = model_manifest["whisper"]
            align    = model_manifest.get("align", {})
            wtpsplit = model_manifest["wtpsplit"]
            whisper_tmp_dir  = whisper["tmp_dir"]
            wtpsplit_tmp_dir = wtpsplit["tmp_dir"]
            align_model_path = ""
            vocab_path       = ""
            cache_valid = True
            if align.get("s3_key") and align.get("tmp_dir"):
                try:
                    align_model_path = find_file(align["tmp_dir"], "model_quantized.onnx")
                    vocab_path       = find_file(align["tmp_dir"], "vocab.json")
                except FileNotFoundError as exc:
                    log.warning(
                        "  cache hit but align model missing in /tmp (%s) "
                        "— invalidating cache and re-downloading", exc)
                    cache_valid = False

            if not cache_valid:
                cleanup_models()   # unconditional wipe — sentinel is stale
                _progress("models: downloading (cache invalidated)", "", t_start=t0)
                whisper_tmp_dir, align_model_path, vocab_path, wtpsplit_tmp_dir = (
                    _download_models(model_bucket, model_manifest, t0)
                )
                write_models_cache_key(cache_key)
        else:
            _progress("models: downloading", "", t_start=t0)
            whisper_tmp_dir, align_model_path, vocab_path, wtpsplit_tmp_dir = (
                _download_models(model_bucket, model_manifest, t0)
            )
            write_models_cache_key(cache_key)

        _progress("models: ready", "", t_start=t0)

        # ── Step 2: Load speech groups ────────────────────────────────────────
        _progress("groups: loading from dynamic segments",
                  f"key=…/{dyn_segment_key.split('/')[-1]}", t_start=t0)
        speech_groups = _load_speech_groups(output_bucket, dyn_segment_key)
        log.info(
            "  %d speech groups in fetch window [%.1fs, %.1fs]",
            len(speech_groups), fetch_start_sec, fetch_end_sec,
        )
        if not speech_groups:
            raise RuntimeError(
                f"No speech groups in dynamic segment: {dyn_segment_key}")

        # ── Step 3: Fetch audio window ────────────────────────────────────────
        _progress("wav: reading header",
                  f"s3://{audio_bucket}/{audio_key}", t_start=t0)
        wav_info = get_wav_info(audio_bucket, audio_key)
        log.info(
            "  WAV: duration=%.1fs  sr=%d  ch=%d  bits=%d",
            wav_info.duration_sec, wav_info.sample_rate,
            wav_info.num_channels, wav_info.bits_per_sample,
        )

        fetch_bytes = int(
            fetch_duration * wav_info.sample_rate
            * wav_info.num_channels * (wav_info.bits_per_sample // 8)
        )
        _progress("audio: byte-range fetch",
                  f"[{fetch_start_sec:.1f}s → {fetch_end_sec:.1f}s]  "
                  f"~{fetch_bytes / 1_048_576:.1f} MB", t_start=t0)
        t_audio = time.monotonic()
        audio_full = read_wav_window(
            audio_bucket, audio_key, wav_info,
            start_sec=fetch_start_sec,
            duration_sec=fetch_duration,
        )
        log.info(
            "  audio fetch: %.2fs  %d samples  %.1fs  min=%.3f  max=%.3f",
            time.monotonic() - t_audio, len(audio_full),
            len(audio_full) / sample_rate,
            audio_full.min(), audio_full.max(),
        )
        if len(audio_full) == 0:
            raise RuntimeError(
                f"read_wav_window returned empty array for chunk "
                f"{dyn_chunk_idx}")

        # ── Step 4: Slice group audio ─────────────────────────────────────────
        _progress("audio: slicing per group",
                  f"{len(speech_groups)} groups", t_start=t0)
        groups_with_audio: list[dict] = []
        for group in speech_groups:
            s = max(0, int(
                (group["start"] - fetch_start_sec) * sample_rate))
            e = min(len(audio_full), int(
                (group["end"] - fetch_start_sec) * sample_rate))
            if e <= s:
                log.warning(
                    "  skip empty group[%d] %.2f→%.2f",
                    group["group_idx"], group["start"], group["end"])
                continue
            group["audio"] = audio_full[s:e].copy()
            groups_with_audio.append(group)

        log.info("  %d groups with audio from %d speech groups",
                 len(groups_with_audio), len(speech_groups))
        if not groups_with_audio:
            raise RuntimeError(
                f"All groups produced empty audio slices for chunk "
                f"{dyn_chunk_idx}")

        # ── Step 5: Init workers + build batches + transcribe ─────────────────
        whisper_kwargs: dict = {
            "beam_size": beam_size,
            "condition_on_previous_text": condition_on_previous_text,
            "no_speech_threshold": no_speech_threshold,
            "log_prob_threshold": log_prob_threshold,
        }
        # Explicitly set temperature to 0.0 and best_of to 1
        if beam_size == 1:
            whisper_kwargs = {**whisper_kwargs, "best_of": 1, "temperature": 0.0}
        if initial_prompt:
            whisper_kwargs = {**whisper_kwargs, "initial_prompt": initial_prompt}

        _progress("transcribe: worker init",
                  f"num_workers={num_workers}", t_start=t0)
        worker_init(
            whisper_model_path=whisper_tmp_dir,
            align_model_path=align_model_path,
            vocab_path=vocab_path,
            wtpsplit_model_dir=wtpsplit_tmp_dir,
            language=language,
            whisper_kwargs=whisper_kwargs,
            sample_rate=sample_rate,
            align_conf_thresh=0.8,
            num_workers=num_workers,
        )

        # For short clips (≤ 30s) use the actual fetch duration as the batch
        # target so all groups fit in one Whisper call instead of being split
        # by the 28s safety margin. For longer audio the default 28s applies.
        batch_target = fetch_duration if fetch_duration <= 30.0 else BATCH_TARGET_SEC
        batches = build_batches(groups_with_audio, target_sec=batch_target)
        log.info(
            "  %d groups → %d batches  (target=%.0fs wall-clock each)",
            len(groups_with_audio), len(batches), batch_target,
        )
        _progress(
            "transcribe: dispatching",
            f"{len(groups_with_audio)} groups → {len(batches)} batches",
            t_start=t0,
        )

        all_raw_words: list[dict] = []
        all_stats:     list[dict] = []

        t_pool = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(batch_worker, b): (bidx, b)
                for bidx, b in enumerate(batches)
            }
            for future in as_completed(futures):
                bidx, b = futures[future]
                try:
                    b_words, b_stats = future.result()
                    all_raw_words.extend(b_words)
                    all_stats.extend(b_stats)
                    errors = [s for s in b_stats if s.get("error")]
                    n_errors = len(errors)
                    status = "✅" if not n_errors else "❌"

                    log.info(
                        "  %s batch %3d  groups=%d  words=%d  "
                        "whisper=%.2fs  align=%.2fs  errors=%d",
                        status, bidx,
                        len(b),
                        sum(s.get("words_total", 0) for s in b_stats),
                        sum(s.get("whisper_sec", 0.0) for s in b_stats),
                        sum(s.get("align_sec",   0.0) for s in b_stats),
                        n_errors,
                    )
                    for s in errors:
                        log.warning(
                            "    group %d [%.2fs→%.2fs]: %s",
                            s.get("group_idx", -1),
                            s.get("abs_start", 0),
                            s.get("abs_end",   0),
                            s.get("error", "unknown"),
                        )
                except Exception as exc:
                    log.error("  batch %d failed: %s", bidx, exc)
                    for group in b:
                        all_stats.append({
                            "group_idx": group["group_idx"],
                            "error": str(exc),
                        })

        pool_sec = time.monotonic() - t_pool
        log.info(
            "  thread pool done in %.1fs  (workers=%d  batches=%d)",
            pool_sec, num_workers, len(batches),
        )
        _progress("transcribe: batches done",
                  f"{pool_sec:.1f}s  words={len(all_raw_words)}",
                  t_start=t0)

        # ── Step 6: Sentence splitting — single wtpsplit call ─────────────────
        # All batches have returned timestamped words.  Run sentence splitting
        # once on the full word sequence so wtpsplit has complete visibility
        # across speech group boundaries.
        _progress("wtpsplit: splitting",
                  f"{len(all_raw_words)} words", t_start=t0)
        t_wtp = time.monotonic()
        sentences = split_words_into_sentences(
            all_raw_words, language,
            merge_gap_threshold_sec=merge_gap_threshold_sec,
        )
        # pad ladder layer 5: a pad-flagged word stranded away from every
        # clean sentence-mate is re-homed to its sentence's majority group
        n_rehomed = rehome_straddle_strays(sentences, groups_with_audio)
        if n_rehomed:
            log.info("  straddle rehoming: %d pad-flagged word(s) moved "
                     "to their sentence's group", n_rehomed)
        # powerset fences (2026-09-25): split sentences at in-speech
        # speaker changes the text model cannot see (a <0.3s pause
        # never splits VAD; wtpsplit is text-only). ONE hook AFTER
        # merge_short covers both lumping paths. Keyed on artifact
        # presence — absent guides (flag off) = no-op.
        try:
            from lib.powerset_guides import apply_fences, load_guides
            guides = load_guides(item["output_bucket"],
                                 item["output_prefix"])
            if guides and guides["fences"]:
                chunk_fences = [t for t in guides["fences"]
                                if fetch_start_sec <= t <= fetch_end_sec]
                before = len(sentences)
                sentences = apply_fences(sentences, chunk_fences)
                if len(sentences) != before:
                    log.info("  powerset fences: %d sentences -> %d "
                             "(%d in-chunk fences)", before,
                             len(sentences), len(chunk_fences))
        except Exception as exc:                 # noqa: BLE001 — degrade
            log.warning("powerset fence application failed "
                        "(continuing unfenced): %s", exc)
        wtpsplit_sec = round(time.monotonic() - t_wtp, 2)

        worker_errors = [s for s in all_stats if s.get("error")]
        if worker_errors:
            log.warning(
                "  %d worker error(s): %s",
                len(worker_errors),
                [e["error"] for e in worker_errors],
            )

        log.info(
            "  sentence splitting: %d words → %d sentences  "
            "wtpsplit=%.2fs  worker_errors=%d",
            len(all_raw_words), len(sentences),
            wtpsplit_sec, len(worker_errors),
        )
        _progress("wtpsplit: done",
                  f"sentences={len(sentences)}  wtpsplit={wtpsplit_sec}s",
                  t_start=t0)

        # ── Step 7: Write transcript to S3 ────────────────────────────────────
        transcript_key = f"{output_prefix}/transcripts/transcript_{dyn_chunk_idx}.json"
        transcript_doc = {
            "dyn_chunk_idx":   dyn_chunk_idx,
            "language":        language,
            "fetch_start_sec": fetch_start_sec,
            "fetch_end_sec":   fetch_end_sec,
            "num_sentences":   len(sentences),
            "num_words":       len(all_raw_words),
            "wtpsplit_sec":    wtpsplit_sec,
            "sentences":       sentences,
            "worker_stats":    sorted(
                all_stats, key=lambda x: x.get("group_idx", 0)
            ),
        }
        put_json(output_bucket, transcript_key, transcript_doc)
        _progress("s3: writing transcript",
                  f"s3://{output_bucket}/{transcript_key}", t_start=t0)
        log.info("  stored → s3://%s/%s  (%d sentences, %d words)",
                 output_bucket, transcript_key, len(sentences), len(all_raw_words))

        # ── Return enriched iterator item ─────────────────────────────────────
        elapsed = round(time.monotonic() - t0, 2)
        output = {
            **item,
            "transcript_key": transcript_key,
            "num_sentences":  len(sentences),
            "status":         "COMPLETED",
            "elapsed_sec":    elapsed,
        }
        _progress(
            "DONE",
            f"chunk={dyn_chunk_idx}  sentences={len(sentences)}  "
            f"words={len(all_raw_words)}  elapsed={elapsed}s",
            t_start=t0,
        )
        return output

    except Exception as exc:
        log.error(
            "[ERR] StateRunWhisperAlignWtpsplitModels chunk %s — %s: %s",
            event.get("item_data", {}).get("dyn_chunk_idx", "?"),
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise
