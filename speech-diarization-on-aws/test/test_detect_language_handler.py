#!/usr/bin/env python3
"""
test_detect_language.py
────────────────────────
Local smoke-test for the DETECT LANGUAGE state logic.

Reads one or more segments_{n}.json files (produced by test_segment.py or
the real VAD Lambda) from a local directory or S3, builds synthetic dynamic-
chunk probe dicts from the turns, and runs detect_language() directly — no
Lambda invocation needed.

The probe shape mirrors what StateDynamicChunkingBySpeechTurns produces:
each probe has fetch_start_sec / fetch_end_sec / speech_sec, which is what
detect_language() now consumes via fetch_chunk_audio().

Usage
─────
    python tests/test_detect_language.py \\
        --model-bucket  model-bucket \\
        --bucket        my-bucket \\
        --prefix        jobs/job-abc \\
        --name          audio.wav \\
        --segments-dir  test_output \\
        [--num-probes   5] \\
        [--min-probe-sec 10.0] \\
        [--handler-dir  .] \\
        [--verbose]

    # Or point directly at S3 segments keys:
    python tests/test_detect_language.py \\
        --model-bucket  model-bucket \\
        --bucket        my-bucket \\
        --prefix        jobs/job-abc \\
        --name          audio.wav \\
        --segments-keys jobs/job-abc/segments/segments_0.json \\
                        jobs/job-abc/segments/segments_1.json

Arguments
─────────
  --model-bucket   S3 bucket containing model artefacts
  --bucket         S3 bucket containing the audio file
  --prefix         S3 key prefix  (e.g. jobs/job-abc)
  --name           Audio filename (e.g. audio.wav)

  Source of segments (one of):
  --segments-dir   Local directory containing segments_*.json files
                   written by test_segment.py  (default: test_output/)
  --segments-keys  One or more S3 keys for segments_{n}.json files

  --num-probes     Number of probe chunks to use (default: 5)
  --min-probe-sec  Minimum speech duration for a probe chunk in seconds
                   (default: 10.0)
  --handler-dir    Project root containing lib/  (default: parent of tests/)
  --verbose        Print per-probe results table

What it tests
─────────────
  1. cleanup_models()       — wipes /tmp/models before starting
  2. ensure_in_tmp()        — downloads Whisper turbo model from S3
  3. build_probe_chunks()   — loads segments, converts turns to chunk-shaped probes
  4. select_probes (inline) — even distribution, speech_sec duration gate
  5. detect_language()      — Whisper majority vote over dynamic chunk probes
  6. Output validation      — language code, confidence range, probe count

Exit codes
──────────
  0  all assertions passed
  1  one or more assertions failed
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import textwrap
import time

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

VALID_LANGUAGE_CODES = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
    "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fo", "fr", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id",
    "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln",
    "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa",
    "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta",
    "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi",
    "yo", "zh",
}


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
# PROBE CHUNK BUILDER
# =============================================================================

def build_probe_chunks(
    args:   argparse.Namespace,
    bucket: str,
    prefix: str,
) -> list[dict]:
    """
    Load segments from local dir or S3 and convert them into synthetic
    dynamic-chunk-shaped probe dicts that detect_language() expects.

    Each probe has the shape produced by StateDynamicChunkingBySpeechTurns:
      { fetch_start_sec, fetch_end_sec, speech_sec, dyn_chunk_idx, ... }

    Strategy: group all turns from each segments file into one probe chunk
    covering the file's full fetch window.  This gives detect_language() the
    same granularity as real dynamic chunks (one chunk per VAD file).
    """
    import boto3
    s3 = boto3.client("s3")

    # ── Resolve S3 keys ───────────────────────────────────────────────────────
    if args.segments_keys:
        log.info(f"Using {len(args.segments_keys)} explicit S3 segments keys")
        s3_keys = args.segments_keys
    else:
        pattern = os.path.join(args.segments_dir, "segments_*.json")
        local_files = sorted(glob.glob(pattern))
        if not local_files:
            raise FileNotFoundError(
                f"No segments_*.json found in '{args.segments_dir}'. "
                f"Run test_segment.py first, or pass --segments-keys."
            )
        log.info(
            f"Found {len(local_files)} local segments file(s) in "
            f"'{args.segments_dir}' — uploading to S3..."
        )
        s3_keys = []
        for fpath in local_files:
            fname = os.path.basename(fpath)
            s3_key = f"{prefix}/segments/{fname}"
            s3.upload_file(fpath, bucket, s3_key)
            log.info(f"  Uploaded: s3://{bucket}/{s3_key}")
            s3_keys.append(s3_key)

    # ── Build one probe chunk per segments file ───────────────────────────────
    chunks = []
    for idx, key in enumerate(s3_keys):
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            doc = json.loads(obj["Body"].read())
        except Exception as exc:
            log.warning(f"  Could not load {key}: {exc}")
            continue

        turns = doc.get("turns", [])
        fetch_start = float(doc.get("fetch_start_sec", 0.0))
        fetch_end = float(doc.get("fetch_end_sec",
                                  max((float(t["end"]) for t in turns), default=0.0)))
        speech_sec = sum(float(t.get("duration", 0.0)) for t in turns)

        chunks.append({
            "dyn_chunk_idx":   idx,
            "fetch_start_sec": fetch_start,
            "fetch_end_sec":   fetch_end,
            "speech_sec":      round(speech_sec, 3),
            "_segments_key":   key,   # informational only
        })
        log.info(
            f"  chunk[{idx}]: [{fetch_start:.1f}s, {fetch_end:.1f}s]  "
            f"speech={speech_sec:.1f}s  turns={len(turns)}"
        )

    return chunks


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(args: argparse.Namespace) -> bool:
    setup_path(args.handler_dir)

    from lib.model_registry import WHISPER_MODELS
    from lib.model_store import ensure_in_tmp, cleanup_models
    from lib.language_detect import detect_language

    bucket = args.bucket
    prefix = args.prefix.strip("/")
    audio_key = f"{prefix}/{args.name}" if prefix else args.name

    print("=" * 64)
    print("  DETECT LANGUAGE — local smoke test")
    print("=" * 64)
    print(f"  Model bucket  : {args.model_bucket}")
    print(f"  Audio         : s3://{bucket}/{audio_key}")
    print(f"  Segments from : {args.segments_dir or 'S3 keys (explicit)'}")
    print(f"  Num probes    : {args.num_probes}")
    print(f"  Min probe     : {args.min_probe_sec}s")
    print()

    failures = []

    # ── 1. Build probe chunks from segments ───────────────────────────────────
    log.info("Step 1/4 — building probe chunks from segments...")
    try:
        all_chunks = build_probe_chunks(args, bucket, prefix)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    check(failures, len(all_chunks) > 0, "At least one probe chunk built")

    # Select evenly-spaced probes (mirrors _select_probe_chunks in the handler)
    qualifying = [c for c in all_chunks
                  if float(c.get("speech_sec", 0.0)) >= args.min_probe_sec]
    if not qualifying:
        log.warning(
            f"No chunks with speech_sec >= {args.min_probe_sec}s — "
            f"using all {len(all_chunks)} chunks"
        )
        qualifying = all_chunks

    n = len(qualifying)
    num = args.num_probes
    if n <= num:
        probes = qualifying
    else:
        indices = [round(i * (n - 1) / (num - 1)) for i in range(num)]
        seen: set = set()
        probes = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                probes.append(qualifying[idx])

    log.info(
        f"  {len(all_chunks)} chunk(s) → {len(probes)} probe(s) selected "
        f"(speech_sec ≥ {args.min_probe_sec}s)"
    )
    check(failures, len(probes) > 0,
          f"At least 1 probe chunk selected (got 0 — "
          f"try --min-probe-sec with a lower value)")

    # ── 2. Download Whisper turbo model ────────────────────────────────────────
    log.info("Step 2/4 — cleaning /tmp and downloading Whisper turbo model...")
    cleanup_models()

    base_entry = WHISPER_MODELS["turbo"]
    t0 = time.monotonic()
    ensure_in_tmp(
        bucket=args.model_bucket,
        s3_key=base_entry["s3_key"],
        tmp_dir=base_entry["tmp_dir"],
        label="whisper/turbo",
    )
    download_sec = time.monotonic() - t0
    log.info(f"  Whisper turbo ready in {download_sec:.1f}s")
    check(failures, os.path.isdir(base_entry["tmp_dir"]),
          f"Whisper turbo model dir exists: {base_entry['tmp_dir']}")

    # ── 3. Run detect_language ────────────────────────────────────────────────
    log.info("Step 3/4 — running detect_language()...")
    t0 = time.monotonic()
    result = detect_language(
        bucket=bucket,
        audio_key=audio_key,
        probes=probes,
        whisper_model_path=base_entry["tmp_dir"],
        num_probes=args.num_probes,
        min_probe_sec=args.min_probe_sec,
    )
    detect_sec = time.monotonic() - t0
    log.info(f"  detect_language() completed in {detect_sec:.1f}s")

    # ── 4. Validate result ────────────────────────────────────────────────────
    log.info("Step 4/4 — validating result...")

    lang = result.get("language", "")
    confidence = result.get("confidence", -1.0)
    method = result.get("method", "")
    num_used = result.get("num_probes_used", 0)
    probe_results = result.get("probe_results", [])

    check(failures, isinstance(lang, str) and len(lang) >= 2,
          f"Language is a string (got {lang!r})")
    check(failures, lang in VALID_LANGUAGE_CODES,
          f"Language code is valid ISO-639-1 (got '{lang}')")
    check(failures, 0.0 <= confidence <= 1.0,
          f"Confidence in [0, 1] (got {confidence:.4f})")
    check(failures, method != "",
          f"Method field present (got {method!r})")
    check(failures, num_used > 0,
          f"At least 1 probe used (got {num_used} — check audio has speech)")
    check(failures, isinstance(probe_results, list),
          "probe_results is a list")

    # Validate each probe result
    probe_failures = []
    for i, pr in enumerate(probe_results):
        if not isinstance(pr.get("lang"), str):
            probe_failures.append(f"probe[{i}] missing 'lang'")
        if not (0.0 <= float(pr.get("prob", -1)) <= 1.0):
            probe_failures.append(
                f"probe[{i}] prob {pr.get('prob')} out of [0, 1]")
        if float(pr.get("start", 0)) >= float(pr.get("end", 0)):
            probe_failures.append(
                f"probe[{i}] start >= end")

    if probe_failures:
        for pf in probe_failures:
            failures.append(pf)
            log.error(f"  FAIL: {pf}")
    elif probe_results:
        log.info(f"  OK  : all {len(probe_results)} probe results valid")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"  Language      : {lang}")
    print(f"  Confidence    : {confidence:.4f}")
    print(f"  Method        : {method}")
    print(f"  Probes used   : {num_used}")
    print(f"  Timing        : download={download_sec:.1f}s  "
          f"detect={detect_sec:.1f}s  "
          f"total={download_sec + detect_sec:.1f}s")
    if args.verbose and probe_results:
        print()
        print(f"  {'#':>3}  {'start':>8}  {'end':>8}  {'dur':>6}  "
              f"{'lang':<6}  prob")
        print("  " + "─" * 46)
        for i, pr in enumerate(probe_results):
            dur = round(float(pr["end"]) - float(pr["start"]), 1)
            print(
                f"  {i:>3}  {pr['start']:>8.2f}  {pr['end']:>8.2f}  "
                f"{dur:>6.1f}  {pr['lang']:<6}  {pr['prob']:.4f}"
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
        description="Local smoke-test for DETECT LANGUAGE state logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            ────────
            # Use segments written by test_segment.py (most common):
            python tests/test_detect_language.py \\
                --model-bucket model-bucket \\
                --bucket       my-bucket \\
                --prefix       jobs/job-abc \\
                --name         audio.wav \\
                --segments-dir test_output \\
                --verbose

            # Point directly at S3 segments keys:
            python tests/test_detect_language.py \\
                --model-bucket model-bucket \\
                --bucket       my-bucket \\
                --prefix       jobs/job-abc \\
                --name         audio.wav \\
                --segments-keys jobs/job-abc/segments/segments_0.json \\
                                jobs/job-abc/segments/segments_1.json
        """),
    )
    parser.add_argument("--model-bucket",   required=True,
                        help="S3 bucket containing model artefacts")
    parser.add_argument("--bucket",         required=True,
                        help="S3 bucket containing the audio file")
    parser.add_argument("--prefix",         required=True,
                        help="S3 key prefix (e.g. jobs/job-abc)")
    parser.add_argument("--name",           required=True,
                        help="Audio filename (e.g. audio.wav)")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--segments-dir",  default="test_output",
                     help="Local dir with segments_*.json from test_segment.py "
                          "(default: test_output/)")
    src.add_argument("--segments-keys", nargs="+",
                     help="Explicit S3 keys for segments_{n}.json files")

    parser.add_argument("--num-probes",     type=int,   default=5,
                        help="Number of turns to probe (default: 5)")
    parser.add_argument("--min-probe-sec",  type=float, default=10.0,
                        help="Minimum turn duration for probing (default: 10.0)")
    parser.add_argument("--handler-dir",
                        default=os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))),
                        help="Project root containing lib/ "
                             "(default: parent of tests/)")
    parser.add_argument("--verbose",        action="store_true",
                        help="Print per-probe results table")
    args = parser.parse_args()

    ok = run_test(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
