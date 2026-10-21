#!/usr/bin/env python3
"""
test_transcribe.py
───────────────────
Local smoke-test for the TRANSCRIBE state logic (A2).

Reads one segments_{n}.json (from test_segment.py or S3), downloads the
Whisper + alignment + wtpsplit models, fetches the audio window from S3,
and runs the full transcription pipeline — exactly as the Lambda would,
but locally for easy debugging.

Usage
─────
    python tests/test_transcribe.py \\
        --model-bucket  model-bucket \\
        --bucket        my-bucket \\
        --prefix        jobs/job-abc \\
        --name          audio.wav \\
        --language      en \\
        --chunk-idx     0 \\
        --fetch-start   0.0 \\
        --fetch-end     510.0 \\
        [--core-start   0.0] \\
        [--core-end     510.0] \\
        [--segments-dir test_output] \\
        [--segments-key jobs/job-abc/segments/segments_0.json] \\
        [--beam-size    5] \\
        [--initial-prompt "Transcript of a meeting."] \\
        [--output-dir   test_output] \\
        [--handler-dir  .] \\
        [--verbose]

Arguments
─────────
  --model-bucket    S3 bucket containing model artefacts
  --bucket          S3 bucket containing the audio file
  --prefix          S3 key prefix (e.g. jobs/job-abc)
  --name            Audio filename (e.g. audio.wav)
  --language        ISO-639-1 language code (e.g. en, cy)
  --chunk-idx       Chunk index — used for output filename (default: 0)
  --fetch-start     Fetch window start in seconds
  --fetch-end       Fetch window end in seconds
  --core-start      Core ownership start (default: same as fetch-start)
  --core-end        Core ownership end   (default: same as fetch-end)

  Segments source (one of):
  --segments-dir    Local dir with segments_{chunk_idx}.json written by
                    test_segment.py  (default: test_output/)
  --segments-key    Explicit S3 key for the segments file — skips
                    local lookup and upload

  Whisper passthrough:
  --beam-size       Beam size for decoding (default: 5)
  --initial-prompt  Optional conditioning prompt for Whisper

  --output-dir      Directory to write transcript_{chunk_idx}.json
                    (default: test_output/)
  --handler-dir     Project root containing lib/  (default: parent
                    of the tests/ directory)
  --verbose         Print every sentence in a table after the run

What it tests
─────────────
  1. cleanup_models()      — wipes /tmp/models before starting
  2. ensure_in_tmp()       — downloads Whisper, align, wtpsplit models
  3. find_file()           — locates model files inside extracted dirs
  4. Segments JSON         — loads turns for this chunk from S3
  5. Audio fetch           — byte-range GET of full chunk window
  6. ProcessPoolExecutor   — spawn pool with worker_init()
  7. chunk_worker()        — Whisper transcribe + alignment per turn
  8. post_process()        — merge short sentences, sort
  9. Zone tagging          — core / left_overlap / right_overlap per sentence
 10. Output validation     — sentence schema, timing, confidence range
 11. JSON output           — writes transcript_{chunk_idx}.json locally

Exit codes
──────────
  0  all assertions passed
  1  one or more assertions failed
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

EXPECTED_SENTENCE_FIELDS = {"text", "start",
                            "end", "confidence", "zone", "words"}
VALID_ZONES = {"core", "left_overlap", "right_overlap"}


# =============================================================================
# PATH SETUP
# =============================================================================

def setup_path(handler_dir: str) -> None:
    for p in [handler_dir, os.path.join(handler_dir, "lib")]:
        if p not in sys.path:
            sys.path.insert(0, p)


# =============================================================================
# ASSERTIONS
# =============================================================================

def check(failures: list, condition: bool, msg: str) -> None:
    if not condition:
        failures.append(msg)
        log.error(f"  FAIL: {msg}")
    else:
        log.info(f"  OK  : {msg}")


# =============================================================================
# ZONE TAGGING  (mirrors handler logic)
# =============================================================================

def _classify_zone(
    start:          float,
    end:            float,
    core_start_sec: float,
    core_end_sec:   float,
) -> str:
    mid = (start + end) / 2.0
    if mid < core_start_sec:
        return "left_overlap"
    if mid > core_end_sec:
        return "right_overlap"
    return "core"


# =============================================================================
# SEGMENTS RESOLUTION
# =============================================================================

def resolve_segments_key(
    args:      argparse.Namespace,
    bucket:    str,
    prefix:    str,
    chunk_idx: int,
) -> str:
    """
    Return the S3 key for the segments file.

    Priority:
      1. --segments-key  (explicit S3 key — no upload)
      2. --segments-dir  (local JSON → upload then return S3 key)
    """
    import boto3
    s3 = boto3.client("s3")

    if args.segments_key:
        log.info(f"Using explicit segments key: {args.segments_key}")
        return args.segments_key

    local_path = os.path.join(
        args.segments_dir, f"segments_{chunk_idx}.json"
    )
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"segments_{chunk_idx}.json not found in '{args.segments_dir}'. "
            f"Run test_segment.py first, or pass --segments-key."
        )

    s3_key = f"{prefix}/segments/segments_{chunk_idx}.json"
    log.info(f"Uploading {local_path} → s3://{bucket}/{s3_key}")
    s3.upload_file(local_path, bucket, s3_key)
    return s3_key


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(args: argparse.Namespace) -> bool:
    multiprocessing.set_start_method("spawn", force=True)
    setup_path(args.handler_dir)

    from lib.model_registry import (
        WHISPER_MODELS, ALIGN_MODELS, WTPSPLIT_MODEL, PYANNOTE_MODEL,
        get_whisper_entry, get_align_entry,
    )
    from lib.model_store import ensure_in_tmp, find_file, cleanup_models
    from lib.transcribe import worker_init, chunk_worker, post_process, load_vocab
    from lib.wav_utils import get_wav_info, read_wav_window

    import boto3
    s3 = boto3.client("s3")

    bucket = args.bucket
    prefix = args.prefix.strip("/")
    audio_key = f"{prefix}/{args.name}" if prefix else args.name
    chunk_idx = args.chunk_idx
    fetch_start = args.fetch_start
    fetch_end = args.fetch_end
    core_start = args.core_start if args.core_start is not None else fetch_start
    core_end = args.core_end if args.core_end is not None else fetch_end
    language = args.language.lower()

    print("=" * 64)
    print("  TRANSCRIBE — local smoke test")
    print("=" * 64)
    print(f"  Model bucket  : {args.model_bucket}")
    print(f"  Audio         : s3://{bucket}/{audio_key}")
    print(f"  Language      : {language}")
    print(f"  Max workers   : {args.max_workers}")
    print(f"  Chunk idx     : {chunk_idx}")
    print(f"  Fetch window  : {fetch_start}s → {fetch_end}s  "
          f"({fetch_end - fetch_start:.1f}s)")
    print(f"  Core window   : {core_start}s → {core_end}s")
    print()

    failures = []

    # ── 1. Resolve segments key ───────────────────────────────────────────────
    log.info("Step 1/6 — resolving segments key...")
    try:
        segments_key = resolve_segments_key(args, bucket, prefix, chunk_idx)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    # ── 2. Download models ────────────────────────────────────────────────────
    log.info("Step 2/6 — cleaning /tmp and downloading models...")
    cleanup_models()

    whisper_entry = get_whisper_entry(language)
    align_entry = get_align_entry(language)
    wtpsplit_entry = WTPSPLIT_MODEL

    t0 = time.monotonic()
    ensure_in_tmp(
        bucket=args.model_bucket,
        s3_key=whisper_entry["s3_key"],
        tmp_dir=whisper_entry["tmp_dir"],
        label=f"whisper/{whisper_entry['_model_key']}",
    )
    ensure_in_tmp(
        bucket=args.model_bucket,
        s3_key=align_entry["s3_key"],
        tmp_dir=align_entry["tmp_dir"],
        label=f"align/{language}",
    )
    ensure_in_tmp(
        bucket=args.model_bucket,
        s3_key=wtpsplit_entry["s3_key"],
        tmp_dir=wtpsplit_entry["tmp_dir"],
        label="wtpsplit",
    )
    download_sec = time.monotonic() - t0
    log.info(f"  Models ready in {download_sec:.1f}s")

    whisper_path = whisper_entry["tmp_dir"]

    align_model_path = ""
    vocab_path = ""
    try:
        align_model_path = find_file(
            align_entry["tmp_dir"], "model_quantized.onnx")
        vocab_path = find_file(align_entry["tmp_dir"], "vocab.json")
        log.info(f"  Align model : {align_model_path}")
    except FileNotFoundError as exc:
        log.warning(
            f"  Align model not found: {exc} — Whisper timestamps used as-is")

    check(failures, os.path.isdir(whisper_path),
          f"Whisper model dir exists: {whisper_path}")
    check(failures, os.path.isdir(wtpsplit_entry["tmp_dir"]),
          f"wtpsplit model dir exists: {wtpsplit_entry['tmp_dir']}")

    # ── 3. Load segment turns ─────────────────────────────────────────────────
    log.info("Step 3/6 — loading segment turns from S3...")
    t_seg = time.monotonic()
    seg_obj = s3.get_object(Bucket=bucket, Key=segments_key)
    seg_doc = json.loads(seg_obj["Body"].read())
    all_turns = seg_doc.get("turns", [])
    log.info(
        f"  {len(all_turns)} turns loaded in {time.monotonic()-t_seg:.2f}s")

    check(failures, len(all_turns) > 0,
          f"Segments file has at least 1 turn (got {len(all_turns)})")

    # ── 4. Fetch audio window ─────────────────────────────────────────────────
    log.info("Step 4/6 — fetching audio window from S3...")
    t_audio = time.monotonic()
    wav_info = get_wav_info(bucket, audio_key)
    audio_full = read_wav_window(
        bucket, audio_key, wav_info,
        start_sec=fetch_start,
        duration_sec=fetch_end - fetch_start,
    )
    fetch_sec = time.monotonic() - t_audio
    log.info(
        f"  {len(audio_full):,} samples  "
        f"({len(audio_full)/16_000:.1f}s)  "
        f"fetched in {fetch_sec:.2f}s"
    )

    check(failures, len(audio_full) > 0,     "Audio array non-empty")
    check(failures, audio_full.dtype.name ==
          "float32", "Audio dtype is float32")

    # ── 5. Build worker args ──────────────────────────────────────────────────
    sr = 16_000
    worker_args = []
    for i, turn in enumerate(all_turns):
        t_start = float(turn["start"])
        t_end = float(turn["end"])
        s = max(0, int((t_start - fetch_start) * sr))
        e = min(len(audio_full), int((t_end - fetch_start) * sr))
        if e <= s:
            continue
        worker_args.append((i, t_start, t_end, audio_full[s:e].copy()))

    check(failures, len(worker_args) > 0,
          f"At least 1 worker task built from turns (got {len(worker_args)})")
    log.info(f"  {len(worker_args)} worker tasks dispatched")

    # ── 6. Run worker pool ────────────────────────────────────────────────────
    log.info("Step 5/6 — running transcription worker pool (spawn)...")

    whisper_kwargs = {"beam_size": args.beam_size}
    if args.initial_prompt:
        whisper_kwargs["initial_prompt"] = args.initial_prompt

    t_transcribe = time.monotonic()
    results_map: Dict[int, list] = {}
    all_stats:   List[Dict] = []

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=worker_init,
        initargs=(
            whisper_path,
            align_model_path,
            vocab_path,
            wtpsplit_entry["tmp_dir"],
            language,
            whisper_kwargs,
            sr,
            0.8,   # align_conf_thresh
        ),
    ) as pool:
        futures = {
            pool.submit(chunk_worker, wa): wa[0]
            for wa in worker_args
        }
        for future in as_completed(futures):
            tidx = futures[future]
            try:
                c_idx, sents, stats = future.result()
                results_map[c_idx] = sents
                all_stats.append(stats)
                status = "✅" if not stats.get("error") else "❌"
                log.info(
                    f"  {status} turn {tidx:>3}  "
                    f"{len(sents)} sentences  "
                    f"{stats.get('total_sec', 0):.2f}s"
                )
            except Exception as exc:
                log.error(f"  Turn {tidx} future failed: {exc}")
                results_map[tidx] = []
                all_stats.append({"chunk_idx": tidx, "error": str(exc)})

    transcribe_sec = time.monotonic() - t_transcribe
    log.info(f"  Pool done in {transcribe_sec:.1f}s")

    # ── Post-process ──────────────────────────────────────────────────────────
    log.info("Step 6/6 — post-processing and validating...")
    raw_sentences: list = []
    for i in range(len(worker_args)):
        raw_sentences.extend(results_map.get(i, []))

    sentences = post_process(raw_sentences, language)

    for sent in sentences:
        sent["zone"] = _classify_zone(
            sent["start"], sent["end"], core_start, core_end
        )

    # ── Validate sentences ────────────────────────────────────────────────────
    check(failures, len(sentences) > 0,
          f"At least 1 sentence produced (got {len(sentences)})")

    sentence_failures = []
    for i, sent in enumerate(sentences):
        missing = EXPECTED_SENTENCE_FIELDS - set(sent.keys())
        if missing:
            sentence_failures.append(f"sent[{i}] missing fields: {missing}")
        if float(sent.get("start", 0)) >= float(sent.get("end", 0)):
            sentence_failures.append(
                f"sent[{i}] start >= end  "
                f"({sent.get('start')} >= {sent.get('end')})"
            )
        if not (0.0 <= float(sent.get("confidence", -1)) <= 1.0):
            sentence_failures.append(
                f"sent[{i}] confidence out of [0,1]: "
                f"{sent.get('confidence')}"
            )
        if sent.get("zone") not in VALID_ZONES:
            sentence_failures.append(
                f"sent[{i}] invalid zone: {sent.get('zone')!r}"
            )
        if not isinstance(sent.get("words"), list):
            sentence_failures.append(f"sent[{i}] words is not a list")
        if not sent.get("text", "").strip():
            sentence_failures.append(f"sent[{i}] text is empty")

    if sentence_failures:
        for sf in sentence_failures:
            failures.append(sf)
            log.error(f"  FAIL: {sf}")
    elif sentences:
        log.info(f"  OK  : all {len(sentences)} sentences valid")

    # Check no worker errors
    worker_errors = [s for s in all_stats if s.get("error")]
    check(failures, len(worker_errors) == 0,
          f"No worker errors (got {len(worker_errors)}: "
          f"{[e['error'] for e in worker_errors]})")

    # ── Write output JSON ─────────────────────────────────────────────────────
    transcript_doc = {
        "chunk_idx":       chunk_idx,
        "language":        language,
        "fetch_start_sec": fetch_start,
        "fetch_end_sec":   fetch_end,
        "core_start_sec":  core_start,
        "core_end_sec":    core_end,
        "num_sentences":   len(sentences),
        "sentences":       sentences,
        "worker_stats":    sorted(all_stats, key=lambda x: x.get("chunk_idx", 0)),
        "elapsed_sec":     round(download_sec + transcribe_sec, 2),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"transcript_{chunk_idx}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(transcript_doc, fh, indent=2, ensure_ascii=False)
    log.info(f"  Written: {out_path}")

    # ── Zone breakdown ────────────────────────────────────────────────────────
    zone_counts = {"core": 0, "left_overlap": 0, "right_overlap": 0}
    for sent in sentences:
        zone_counts[sent.get("zone", "core")] += 1

    check(failures, zone_counts["core"] > 0,
          f"At least 1 core-zone sentence (zone counts: {zone_counts})")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_sec = download_sec + fetch_sec + transcribe_sec
    print()
    print("=" * 64)
    print(f"  Sentences     : {len(sentences)}")
    print(f"  Zone counts   : {zone_counts}")
    print(f"  Worker errors : {len(worker_errors)}")
    print(f"  Timing        : download={download_sec:.1f}s  "
          f"fetch={fetch_sec:.2f}s  "
          f"transcribe={transcribe_sec:.1f}s  "
          f"total={total_sec:.1f}s")
    print(f"  Output JSON   : {out_path}")

    if args.verbose and sentences:
        print()
        print(f"  {'#':>4}  {'start':>8}  {'end':>8}  {'dur':>6}  "
              f"{'conf':>5}  {'zone':<14}  text")
        print("  " + "─" * 72)
        for i, sent in enumerate(sentences):
            dur = round(sent["end"] - sent["start"], 1)
            text = sent["text"]
            if len(text) > 45:
                text = text[:42] + "..."
            print(
                f"  {i:>4}  {sent['start']:>8.2f}  {sent['end']:>8.2f}  "
                f"{dur:>6.1f}  {sent['confidence']:>5.3f}  "
                f"{sent['zone']:<14}  {text}"
            )

    print()
    if failures:
        print(f"  FAILED — {len(failures)} assertion(s):")
        for f in failures:
            print(f"    ✗ {f}")
        print()
        return False
    else:
        print("  PASSED — all assertions OK  🎉")
        print()
        return True


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local smoke-test for TRANSCRIBE state logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            ────────
            # Use segments from test_segment.py (most common):
            python tests/test_transcribe.py \\
                --model-bucket model-bucket \\
                --bucket       my-bucket \\
                --prefix       jobs/job-abc \\
                --name         audio.wav \\
                --language     en \\
                --chunk-idx    0 \\
                --fetch-start  0 \\
                --fetch-end    510 \\
                --segments-dir test_output \\
                --verbose

            # Point directly at an S3 segments key:
            python tests/test_transcribe.py \\
                --model-bucket model-bucket \\
                --bucket       my-bucket \\
                --prefix       jobs/job-abc \\
                --name         audio.wav \\
                --language     cy \\
                --chunk-idx    0 \\
                --fetch-start  0 \\
                --fetch-end    510 \\
                --segments-key jobs/job-abc/segments/segments_0.json \\
                --initial-prompt "Cyfarfod cyngor." \\
                --verbose
        """),
    )
    parser.add_argument("--model-bucket",    required=True,
                        help="S3 bucket containing model artefacts")
    parser.add_argument("--bucket",          required=True,
                        help="S3 bucket containing the audio file")
    parser.add_argument("--prefix",          required=True,
                        help="S3 key prefix (e.g. jobs/job-abc)")
    parser.add_argument("--name",            required=True,
                        help="Audio filename (e.g. audio.wav)")
    parser.add_argument("--language",        required=True,
                        help="ISO-639-1 language code (e.g. en, cy)")
    parser.add_argument("--chunk-idx",       type=int, default=0,
                        help="Chunk index (default: 0)")
    parser.add_argument("--fetch-start",     type=float, required=True,
                        help="Fetch window start in seconds")
    parser.add_argument("--fetch-end",       type=float, required=True,
                        help="Fetch window end in seconds")
    parser.add_argument("--core-start",      type=float, default=None,
                        help="Core ownership start (default: same as fetch-start)")
    parser.add_argument("--core-end",        type=float, default=None,
                        help="Core ownership end   (default: same as fetch-end)")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--segments-dir",  default="test_output",
                     help="Local dir with segments_{chunk_idx}.json "
                          "(default: test_output/)")
    src.add_argument("--segments-key",
                     help="Explicit S3 key for segments_{chunk_idx}.json")

    parser.add_argument("--max-workers",     type=int,   default=1,
                        help="Worker pool size for transcription "
                             "(default: 1 — safe for all model sizes at 6144 MB; "
                             "set to 2 only for turbo model on chunks "
                             "with many short turns)")
    parser.add_argument("--beam-size",       type=int,   default=5,
                        help="Whisper beam size (default: 5)")
    parser.add_argument("--initial-prompt",  default=None,
                        help="Optional Whisper initial prompt string")
    parser.add_argument("--output-dir",      default="test_output",
                        help="Directory to write transcript_{chunk_idx}.json "
                             "(default: test_output/)")
    parser.add_argument("--handler-dir",
                        default=os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))),
                        help="Project root containing lib/ "
                             "(default: parent of tests/)")
    parser.add_argument("--verbose",         action="store_true",
                        help="Print every sentence in a table")

    args = parser.parse_args()

    if args.fetch_end <= args.fetch_start:
        parser.error(
            f"--fetch-end ({args.fetch_end}) must be > "
            f"--fetch-start ({args.fetch_start})"
        )

    ok = run_test(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
