# vad_handler.py
"""
StateRunVADModel  (Map state — one Lambda per fixed chunk, 3584 MB / 2-cores)

Receives one fixed-chunk iterator from StateFixedChunkingForVAD, fetches the audio
window from S3 via byte-range GET, runs pyannote community-1 ONNX segmentation,
and writes segments_{chunk_idx}.json to S3.

No zone assignment is done here — the fixed chunks have no core/overlap concept.
Turn indices assigned are local to this chunk; they are re-numbered globally by
StateDynamicChunkingBySpeechTurns once all VAD outputs are merged.

Output enriches the iterator item (via Map ResultPath):
  segments_key  — S3 key of segments_{chunk_idx}.json
  num_turns     — number of turns found
  status        — "COMPLETED"
  elapsed_sec

segments_{chunk_idx}.json shape:
  {
    "chunk_idx":          int,
    "fetch_start_sec":    float,
    "fetch_end_sec":      float,
    "fetch_duration_sec": float,
    "num_turns":          int,
    "turns": [
      { "turn_idx": int, "start": float, "end": float,
        "duration": float, "has_overlap": bool }
    ]
  }
  All timestamps are absolute seconds in the original audio file.
  No "zone" field — assigned by StateDynamicChunkingBySpeechTurns.

Model warm-cache:
  Cold container: S3 download → /tmp extraction (~2 s for the 17 MB archive)
  Warm container: _dir_has_files() check → < 1 ms, zero network calls

Lambda configuration
─────────────────────
  Memory  : 3584 MB (TwoCores)
  Timeout : 900 s
  Arch    : arm64
"""
# ── Defaults ──────────────────────────────────────────────────────────────────

from __future__ import annotations

from lib.s3_utils import put_json
from lib.wav_utils import get_wav_info, read_wav_window
from lib.model_store import ensure_in_tmp, find_file, cleanup_models
from lib.model_registry import PYANNOTE_MODEL
from lib.segmentation import run_segmentation
import onnxruntime as ort
import traceback
import time
import logging
import json
# ── ORT cpuinfo fix (arm64 Lambda) ───────────────────────────────────────────
# Must be imported and called before onnxruntime — see lib/ort_init.py.
# Writes correct CPU range from /proc/cpuinfo so ORT gets the right vCPU
# count regardless of which Lambda function (vad/whisper/embed) uses this image.
from lib.ort_init import ensure_cpuinfo_files, get_num_cpus
ensure_cpuinfo_files()
# ─────────────────────────────────────────────────────────────────────────────


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
# PROGRESS LOGGING
# =============================================================================

def _progress(step: str, detail: str = "", t_start: float = 0.0) -> None:
    """Emit a structured progress line visible in CloudWatch."""
    elapsed = f"  (+{time.monotonic() - t_start:.2f}s)" if t_start else ""
    log.info(f"  ▶ {step:<30s}  {detail}{elapsed}")


# =============================================================================
# ONNX SESSION — module-level cache
# Persists across warm invocations of the same container.
# =============================================================================

_session: ort.InferenceSession | None = None


def _get_session(
    model_bucket:  str,
    model_prefix:  str = "_model_artefacts",
    memory_mb:     int | None = None,
) -> ort.InferenceSession:
    """
    Load (or reuse) the ONNX segmentation session.

    The session is stored in a module-level variable so it survives
    across warm invocations — model weights stay in RAM, avoiding
    repeated disk reads.

    model_prefix overrides the registry default so the key resolves to
    {model_prefix}/pyannote/community-1/model.tar.gz in the customer bucket.
    """
    global _session
    if _session is not None:
        log.info("[cache ✅] ONNX session already loaded")
        return _session

    # Build s3_key from model_prefix, falling back to registry default
    s3_key = PYANNOTE_MODEL["s3_key"]
    if model_prefix and not s3_key.startswith(model_prefix):
        # Registry key starts with "_model_artefacts/..." — replace prefix
        s3_key = model_prefix.strip("/") + "/" + s3_key.split("/", 1)[1]

    ensure_in_tmp(
        bucket=model_bucket,
        s3_key=s3_key,
        tmp_dir=PYANNOTE_MODEL["tmp_dir"],
        label="pyannote/community-1",
    )
    onnx_path = find_file(PYANNOTE_MODEL["tmp_dir"], "segmentation*.onnx")
    log.info(f"Loading ONNX session from {onnx_path}")
    t0 = time.monotonic()

    opts = ort.SessionOptions()
    # Lambda vCPU count is determined by memory size, not os.cpu_count().
    # os.cpu_count() returns the host's physical core count (often 2+), which
    # causes ORT to spawn more threads than the function actually has — leading
    # to context switching overhead and slower inference.
    #
    # get_num_cpus() reads /proc/cpuinfo — correct for any Lambda memory size
    _num_cpus = get_num_cpus(memory_mb)
    log.info(f"ORT intra_op_num_threads = {_num_cpus} (memory_mb={memory_mb})")

    opts.intra_op_num_threads = _num_cpus
    opts.inter_op_num_threads = 1
    # Avoid spawning extra processes inside Lambda
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    _session = ort.InferenceSession(
        onnx_path, opts,
        providers=["CPUExecutionProvider"],
    )
    log.info(f"ONNX session ready: {time.monotonic() - t0:.2f}s")
    return _session


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    # Wipe all models from /tmp at invocation start — prevents /tmp filling up
    # across warm container reuse with different jobs / languages.
    cleanup_models()

    log.info("VAD event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── Unpack item_data ───────────────────────────────────────────────────
        # The Map state wraps each iterator item under "item_data".
        # fallback to event for direct testing
        log.setLevel(logging.INFO)
        item = event.get("item_data", event)

        required = {
            "chunk_idx", "fetch_start_sec", "fetch_end_sec",
            "audio_bucket", "audio_key",
            "output_bucket", "output_prefix",
            "audio_cfg", "model_cfg", "vad_cfg",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"Missing item_data fields: {missing}")

        chunk_idx = int(item["chunk_idx"])
        fetch_start_sec = float(item["fetch_start_sec"])
        fetch_end_sec = float(item["fetch_end_sec"])
        fetch_duration = fetch_end_sec - fetch_start_sec

        audio_bucket = item["audio_bucket"]
        audio_key = item["audio_key"]

        output_bucket = item["output_bucket"]
        output_prefix = item["output_prefix"]

        model_cfg = item["model_cfg"]
        model_bucket = model_cfg["model_bucket"]
        model_prefix = model_cfg["model_prefix"]

        # VAD config — with defaults
        vad_cfg = item["vad_cfg"]
        min_speech_duration_sec:  float = float(
            vad_cfg["min_speech_duration_sec"])
        min_silence_duration_sec: float = float(
            vad_cfg["min_silence_duration_sec"])
        # hidden knob (2026-09-16): the previously hardcoded VAD speech
        # threshold — exposed so the DER harness can sweep the measured
        # 3-10% SOLO miss (see input_schema.DEFAULT_SEG_THRESHOLD)
        from lib.input_schema import DEFAULT_SEG_THRESHOLD
        seg_threshold = float(
            vad_cfg.get("seg_threshold", DEFAULT_SEG_THRESHOLD))

        log.info(
            f"Chunk {chunk_idx}: "
            f"fetch=[{fetch_start_sec:.1f}s, {fetch_end_sec:.1f}s]  "
            f"({fetch_duration:.1f}s)  "
            f"min_speech={min_speech_duration_sec}s  min_silence={min_silence_duration_sec}s"
        )
        log.debug(
            f"  audio_bucket  : {audio_bucket}\n"
            f"  audio_key     : {audio_key}\n"
            f"  model_bucket  : {model_bucket}\n"
            f"  model_prefix  : {model_prefix}\n"
            f"  output_bucket : {output_bucket}\n"
            f"  output_prefix : {output_prefix}"
        )
        _progress("START", f"chunk={chunk_idx}  fetch={fetch_duration:.0f}s")

        # ── Ensure ONNX session (warm-cached) ─────────────────────────────────
        _progress("model: load/check cache",
                  f"s3://{model_bucket}/{model_prefix}/pyannote/...", t0)
        session = _get_session(model_bucket, model_prefix,
                               context.memory_limit_in_mb)
        _progress("model: ready", "", t0)

        # ── Read WAV header (cheap byte-range GET) ────────────────────────────
        t_hdr = time.monotonic()
        _progress("wav: reading header",
                  f"s3://{audio_bucket}/{audio_key}", t0)
        info = get_wav_info(audio_bucket, audio_key)
        log.info(
            f"  WAV header: {time.monotonic() - t_hdr:.2f}s  "
            f"duration={info.duration_sec:.1f}s  "
            f"sr={info.sample_rate}  ch={info.num_channels}  "
            f"bits={info.bits_per_sample}  data_offset={info.data_offset}"
        )

        # ── Fetch audio window from S3 (byte-range GET) ───────────────────────
        t_dl = time.monotonic()
        fetch_bytes = int(fetch_duration * info.sample_rate * info.num_channels
                          * (info.bits_per_sample // 8))
        _progress(
            "audio: byte-range fetch",
            f"[{fetch_start_sec:.1f}s → {fetch_end_sec:.1f}s]  "
            f"~{fetch_bytes/1_048_576:.1f} MB",
            t0,
        )
        audio = read_wav_window(
            audio_bucket, audio_key, info,
            start_sec=fetch_start_sec,
            duration_sec=fetch_duration,
        )
        log.info(
            f"  audio fetch: {time.monotonic() - t_dl:.2f}s  "
            f"{len(audio):,} samples  "
            f"{len(audio)/info.sample_rate:.1f}s  "
            f"min={audio.min():.3f}  max={audio.max():.3f}"
        )

        if len(audio) == 0:
            raise RuntimeError(
                f"read_wav_window returned empty array for chunk {chunk_idx}"
            )

        # ── Run pyannote segmentation ─────────────────────────────────────────
        t_seg = time.monotonic()
        _progress(
            "vad: running segmentation",
            f"{len(audio)/info.sample_rate:.1f}s audio  "
            f"threshold={seg_threshold}  min_speech={min_speech_duration_sec}s",
            t0,
        )
        # powerset guides (2026-09-25): ONE flag, read HERE only —
        # downstream states key off artifact presence (three-state)
        from lib.input_schema import DEFAULT_USE_POWERSET_GUIDES
        from lib.powerset_guides import PowersetCollector, build_guides
        use_guides = bool(vad_cfg.get("use_powerset_guides",
                                      DEFAULT_USE_POWERSET_GUIDES))
        collector = PowersetCollector() if use_guides else None
        turns, overlap_evidence = run_segmentation(
            audio=audio,
            session=session,
            fetch_start_sec=fetch_start_sec,
            threshold=seg_threshold,
            min_speech_duration_sec=min_speech_duration_sec,
            min_silence_duration_sec=min_silence_duration_sec,
            collect_overlap_evidence=True,
            powerset_collector=collector,
        )
        seg_sec = time.monotonic() - t_seg
        speech_sec = sum(t["duration"] for t in turns)
        overlap_count = sum(1 for t in turns if t.get("has_overlap"))
        log.info(
            f"  segmentation: {seg_sec:.2f}s  "
            f"→ {len(turns)} turns  "
            f"speech={speech_sec:.1f}s  "
            f"overlap_turns={overlap_count}  "
            f"RTF={seg_sec / max(fetch_duration, 1e-6):.3f}"
        )
        if log.isEnabledFor(logging.DEBUG) and turns:
            log.debug("  Turn breakdown:")
            for t in turns:
                log.debug(
                    f"    [{t['start']:>8.2f}s → {t['end']:>8.2f}s]  "
                    f"dur={t['duration']:>6.2f}s  "
                    f"overlap={'Y' if t.get('has_overlap') else 'N'}"
                )

        # ── Assign turn indices ───────────────────────────────────────────────
        # No zone assignment here — done by StateDynamicChunkingBySpeechTurns
        # once the full cross-chunk timeline is available.
        for idx, turn in enumerate(turns):
            turn["turn_idx"] = idx

        # ── Build segments document ───────────────────────────────────────────
        segments_doc = {
            "chunk_idx":          chunk_idx,
            "fetch_start_sec":    fetch_start_sec,
            "fetch_end_sec":      fetch_end_sec,
            "fetch_duration_sec": round(fetch_duration, 3),
            "num_turns":          len(turns),
            "turns":              turns,
            # P3: channel-resolved cross-talk evidence (two solo spans
            # provably from DIFFERENT people) — consumed as cannot-link
            # constraints by global clustering
            "overlap_evidence":   overlap_evidence,
            "seg_threshold":      seg_threshold,
        }

        # ── Store to S3 ───────────────────────────────────────────────────────
        segments_key = f"{output_prefix}/segments/segments_{chunk_idx}.json"
        put_json(output_bucket, segments_key, segments_doc)

        if collector is not None:
            from lib.segmentation import SEG_WINDOW_SEC, SEG_STEP_SEC
            guides = build_guides(collector, SEG_WINDOW_SEC, SEG_STEP_SEC,
                                  fetch_start_sec, fetch_end_sec)
            powerset_key = (f"{output_prefix}/powerset/"
                            f"powerset_{chunk_idx}.json")
            put_json(output_bucket, powerset_key, guides)
            log.info("  powerset guides: %d fences, %d overlap spans, "
                     "%d tracks, %d co-active pairs, clique floor %d "
                     "→ s3://%s/%s",
                     len(guides["fences"]), len(guides["overlap_spans"]),
                     len(guides["tracks"]),
                     len(guides["co_active_pairs"]),
                     guides["clique_floor"], output_bucket, powerset_key)
        _progress("s3: writing segments",
                  f"s3://{output_bucket}/{segments_key}", t0)
        log.info(
            f"  stored → s3://{output_bucket}/{segments_key}  ({len(turns)} turns)")

        # ── Return enriched iterator item ─────────────────────────────────────
        # This replaces the iterator item in data.iterators via ResultPath.
        elapsed = round(time.monotonic() - t0, 2)
        output = {
            **item,
            "segments_key": segments_key,
            "num_turns":    len(turns),
            "status":       "COMPLETED",
            "elapsed_sec":  elapsed,
        }
        _progress(
            "DONE", f"chunk={chunk_idx}  turns={len(turns)}  elapsed={elapsed}s", t0)
        log.debug("  output keys: %s", sorted(output.keys()))

        return output

    except Exception as exc:
        log.error(
            "[ERR] VAD chunk %s — %s: %s",
            event.get("item_data", {}).get("chunk_idx", "?"),
            type(exc).__name__, exc
        )
        traceback.print_exc()
        raise
