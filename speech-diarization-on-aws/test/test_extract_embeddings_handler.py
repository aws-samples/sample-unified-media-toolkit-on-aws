#!/usr/bin/env python3
"""
test_extract_embeddings.py
───────────────────────────
Local smoke-test for the EXTRACT EMBEDDINGS state logic (B1).

Reads one segments_{n}.json (from test_segment.py or S3), downloads the
WeSpeaker ResNet34 ONNX model, fetches the audio window from S3, and
runs the full embedding extraction pipeline — exactly as the Lambda would,
but locally for easy debugging.

Usage
─────
    python tests/test_extract_embeddings.py \\
        --model-bucket  model-bucket \\
        --bucket        my-bucket \\
        --prefix        jobs/job-abc \\
        --name          audio.wav \\
        --chunk-idx     0 \\
        --fetch-start   0.0 \\
        --fetch-end     510.0 \\
        [--core-start   0.0] \\
        [--core-end     510.0] \\
        [--segments-dir test_output] \\
        [--segments-key jobs/job-abc/segments/segments_0.json] \\
        [--min-duration 1.5] \\
        [--use-int8     1] \\
        [--output-dir   test_output] \\
        [--handler-dir  .] \\
        [--verbose]

Arguments
─────────
  --model-bucket   S3 bucket containing model artefacts
  --bucket         S3 bucket containing the audio file
  --prefix         S3 key prefix (e.g. jobs/job-abc)
  --name           Audio filename (e.g. audio.wav)
  --chunk-idx      Chunk index — used for output filename (default: 0)
  --fetch-start    Fetch window start in seconds
  --fetch-end      Fetch window end in seconds
  --core-start     Core ownership start (default: same as fetch-start)
  --core-end       Core ownership end   (default: same as fetch-end)

  Segments source (one of):
  --segments-dir   Local dir with segments_{chunk_idx}.json written by
                   test_segment.py  (default: test_output/)
  --segments-key   Explicit S3 key for the segments file — skips
                   local lookup and upload

  --min-duration   Minimum turn duration in seconds to embed (default: 1.5)
  --use-int8       Use INT8 quantised model: 1=yes, 0=no  (default: 1)
  --output-dir     Directory to write embeddings_{chunk_idx}.json
                   (default: test_output/)
  --handler-dir    Project root containing lib/  (default: parent of tests/)
  --verbose        Print per-turn embedding stats table

What it tests
─────────────
  1. cleanup_models()            — wipes /tmp/models before starting
  2. ensure_in_tmp()             — downloads pyannote model bundle from S3
  3. find_file()                 — locates embedding[.int8].onnx in model dir
  4. probe_embedding_model()     — verifies input name and embedding dim=256
  5. Segments JSON loading       — reads turns from S3
  6. Audio fetch                 — byte-range GET of full chunk window
  7. extract_embedding_for_turn()— fbank → ONNX → L2-normalised vector
  8. Quality gate                — turns < min-duration are skipped
  9. Embedding validation        — shape, dtype, L2 norm ≈ 1.0, no NaN/Inf
 10. JSON output                 — writes embeddings_{chunk_idx}.json locally

Exit codes
──────────
  0  all assertions passed
  1  one or more assertions failed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

VALID_ZONES = {"core", "left_overlap", "right_overlap"}
EXPECTED_EMB_DIM = 256


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
# SEGMENTS RESOLUTION
# =============================================================================

def resolve_segments_key(
    args:      argparse.Namespace,
    bucket:    str,
    prefix:    str,
    chunk_idx: int,
) -> str:
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
    setup_path(args.handler_dir)

    import boto3
    import onnxruntime as ort

    from lib.model_registry import PYANNOTE_MODEL
    from lib.model_store import ensure_in_tmp, find_file, cleanup_models
    from lib.wav_utils import get_wav_info, read_wav_window
    from lib.embedding import (
        probe_embedding_model,
        extract_embedding_for_turn,
    )

    s3 = boto3.client("s3")

    bucket = args.bucket
    prefix = args.prefix.strip("/")
    audio_key = f"{prefix}/{args.name}" if prefix else args.name
    chunk_idx = args.chunk_idx
    fetch_start = args.fetch_start
    fetch_end = args.fetch_end
    core_start = args.core_start if args.core_start is not None else fetch_start
    core_end = args.core_end if args.core_end is not None else fetch_end
    min_dur = args.min_duration
    use_int8 = args.use_int8 == 1

    print("=" * 64)
    print("  EXTRACT EMBEDDINGS — local smoke test")
    print("=" * 64)
    print(f"  Model bucket  : {args.model_bucket}")
    print(f"  Audio         : s3://{bucket}/{audio_key}")
    print(f"  Chunk idx     : {chunk_idx}")
    print(f"  Fetch window  : {fetch_start}s → {fetch_end}s  "
          f"({fetch_end - fetch_start:.1f}s)")
    print(f"  Core window   : {core_start}s → {core_end}s")
    print(f"  Min duration  : {min_dur}s")
    print(f"  Model variant : {'INT8' if use_int8 else 'FP32'}")
    print()

    failures = []

    # ── 1. Resolve segments key ───────────────────────────────────────────────
    log.info("Step 1/6 — resolving segments key...")
    try:
        segments_key = resolve_segments_key(args, bucket, prefix, chunk_idx)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    # ── 2. Download model ─────────────────────────────────────────────────────
    log.info("Step 2/6 — cleaning /tmp and downloading embedding model...")
    cleanup_models()

    t0 = time.monotonic()
    ensure_in_tmp(
        bucket=args.model_bucket,
        s3_key=PYANNOTE_MODEL["s3_key"],
        tmp_dir=PYANNOTE_MODEL["tmp_dir"],
        label="pyannote/community-1",
    )
    download_sec = time.monotonic() - t0
    log.info(f"  Model ready in {download_sec:.1f}s")

    check(failures, os.path.isdir(PYANNOTE_MODEL["tmp_dir"]),
          f"Pyannote model dir exists: {PYANNOTE_MODEL['tmp_dir']}")

    # ── 3. Load ONNX session ──────────────────────────────────────────────────
    log.info("Step 3/6 — loading embedding ONNX session...")
    pattern = "embedding.int8.onnx" if use_int8 else "embedding.onnx"
    try:
        onnx_path = find_file(PYANNOTE_MODEL["tmp_dir"], pattern)
        log.info(f"  Found: {onnx_path}")
    except FileNotFoundError:
        # fall back to the other variant
        fallback = "embedding.onnx" if use_int8 else "embedding.int8.onnx"
        log.warning(f"  {pattern} not found — trying {fallback}")
        onnx_path = find_file(PYANNOTE_MODEL["tmp_dir"], fallback)
        log.info(f"  Found fallback: {onnx_path}")

    check(failures, os.path.exists(onnx_path),
          f"Embedding ONNX file exists: {onnx_path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        onnx_path, opts, providers=["CPUExecutionProvider"]
    )

    input_name, output_name, emb_dim = probe_embedding_model(session)
    log.info(f"  input='{input_name}'  output='{output_name}'  dim={emb_dim}")

    check(failures, session is not None,    "ONNX session loaded")
    check(failures, emb_dim == EXPECTED_EMB_DIM,
          f"Embedding dim == {EXPECTED_EMB_DIM} (got {emb_dim})")

    # ── 4. Load segment turns ─────────────────────────────────────────────────
    log.info("Step 4/6 — loading segment turns from S3...")
    t_seg = time.monotonic()
    seg_obj = s3.get_object(Bucket=bucket, Key=segments_key)
    seg_doc = json.loads(seg_obj["Body"].read())
    turns = seg_doc.get("turns", [])
    log.info(f"  {len(turns)} turns loaded in {time.monotonic()-t_seg:.2f}s")

    check(failures, len(turns) > 0,
          f"Segments file has at least 1 turn (got {len(turns)})")

    qualifying = [t for t in turns
                  if float(t.get("duration", 0)) >= min_dur]
    log.info(
        f"  {len(qualifying)}/{len(turns)} turns qualify "
        f"(duration ≥ {min_dur}s)"
    )
    check(failures, len(qualifying) > 0,
          f"At least 1 turn qualifies for embedding "
          f"(got 0 — try --min-duration with a lower value)")

    # ── 5. Fetch audio window ─────────────────────────────────────────────────
    log.info("Step 5/6 — fetching audio window from S3...")
    t_audio = time.monotonic()
    wav_info = get_wav_info(bucket, audio_key)
    audio = read_wav_window(
        bucket, audio_key, wav_info,
        start_sec=fetch_start,
        duration_sec=fetch_end - fetch_start,
    )
    fetch_sec = time.monotonic() - t_audio
    log.info(
        f"  {len(audio):,} samples  "
        f"({len(audio)/16_000:.1f}s)  "
        f"fetched in {fetch_sec:.2f}s"
    )

    check(failures, len(audio) > 0,
          "Audio array non-empty")
    check(failures, audio.dtype.name == "float32",
          f"Audio dtype is float32 (got {audio.dtype})")
    check(failures, np.isfinite(audio).all(),
          "Audio contains no NaN/Inf")

    # ── 6. Extract embeddings ─────────────────────────────────────────────────
    log.info("Step 6/6 — extracting embeddings...")
    t_emb = time.monotonic()
    embedded:  List[Dict] = []
    num_skipped = 0
    emb_failures = []

    for turn in turns:
        duration = float(turn.get("duration", 0.0))

        if duration < min_dur:
            num_skipped += 1
            continue

        emb = extract_embedding_for_turn(
            wav=audio,
            turn_start_abs=float(turn["start"]),
            turn_end_abs=float(turn["end"]),
            fetch_start_sec=fetch_start,
            session=session,
            input_name=input_name,
            output_name=output_name,
        )

        tidx = turn.get("turn_idx", "?")
        if emb is None:
            num_skipped += 1
            emb_failures.append(f"turn[{tidx}] embedding returned None "
                                f"({duration:.2f}s)")
            continue

        # Validate this embedding
        if emb.shape != (emb_dim,):
            emb_failures.append(
                f"turn[{tidx}] wrong shape: {emb.shape} != ({emb_dim},)")
        if not np.isfinite(emb).all():
            emb_failures.append(f"turn[{tidx}] contains NaN/Inf")
        norm = float(np.linalg.norm(emb))
        if not (0.99 <= norm <= 1.01):
            emb_failures.append(
                f"turn[{tidx}] L2 norm {norm:.4f} not ≈ 1.0")

        embedded.append({
            "turn_idx":    tidx,
            "start":       turn["start"],
            "end":         turn["end"],
            "duration":    round(duration, 3),
            "zone":        turn.get("zone", "core"),
            "has_overlap": turn.get("has_overlap", False),
            "embedding":   emb.tolist(),
        })

    emb_sec = time.monotonic() - t_emb
    log.info(
        f"  {len(embedded)} embedded, {num_skipped} skipped  "
        f"in {emb_sec:.1f}s  "
        f"({emb_sec/max(1,len(embedded)):.2f}s/turn avg)"
    )

    check(failures, len(embedded) > 0,
          f"At least 1 embedding extracted (got {len(embedded)})")

    if emb_failures:
        for ef in emb_failures:
            failures.append(ef)
            log.error(f"  FAIL: {ef}")
    else:
        log.info(f"  OK  : all {len(embedded)} embeddings valid "
                 f"(shape, dtype, L2 norm ≈ 1.0)")

    # Zone breakdown
    zone_counts = {"core": 0, "left_overlap": 0, "right_overlap": 0}
    for rec in embedded:
        zone_counts[rec.get("zone", "core")] += 1

    # ── Write output JSON ─────────────────────────────────────────────────────
    emb_doc = {
        "chunk_idx":       chunk_idx,
        "fetch_start_sec": fetch_start,
        "fetch_end_sec":   fetch_end,
        "core_start_sec":  core_start,
        "core_end_sec":    core_end,
        "embedding_dim":   emb_dim,
        "num_embedded":    len(embedded),
        "num_skipped":     num_skipped,
        "speaker_turns":   embedded,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"embeddings_{chunk_idx}.json")
    with open(out_path, "w") as fh:
        json.dump(emb_doc, fh, indent=2)
    log.info(f"  Written: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_sec = download_sec + fetch_sec + emb_sec
    print()
    print("=" * 64)
    print(f"  Turns total   : {len(turns)}")
    print(f"  Embedded      : {len(embedded)}")
    print(f"  Skipped       : {num_skipped}  "
          f"(< {min_dur}s or embedding failed)")
    print(f"  Zone counts   : {zone_counts}")
    print(f"  Timing        : download={download_sec:.1f}s  "
          f"fetch={fetch_sec:.2f}s  "
          f"embed={emb_sec:.1f}s  "
          f"total={total_sec:.1f}s")
    print(f"  Avg per turn  : {emb_sec/max(1,len(embedded)):.2f}s")
    print(f"  Output JSON   : {out_path}")

    if args.verbose and embedded:
        print()
        print(f"  {'#':>4}  {'start':>8}  {'end':>8}  {'dur':>6}  "
              f"{'norm':>6}  {'ovlp':>5}  zone")
        print("  " + "─" * 56)
        for rec in embedded:
            emb_arr = np.array(rec["embedding"], dtype=np.float32)
            norm = float(np.linalg.norm(emb_arr))
            print(
                f"  {rec['turn_idx']:>4}  "
                f"{rec['start']:>8.2f}  "
                f"{rec['end']:>8.2f}  "
                f"{rec['duration']:>6.2f}  "
                f"{norm:>6.4f}  "
                f"{str(rec['has_overlap']):>5}  "
                f"{rec['zone']}"
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
        description="Local smoke-test for EXTRACT EMBEDDINGS state logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            ────────
            # Use segments from test_segment.py (most common):
            python tests/test_extract_embeddings.py \\
                --model-bucket model-bucket \\
                --bucket       my-bucket \\
                --prefix       jobs/job-abc \\
                --name         audio.wav \\
                --chunk-idx    0 \\
                --fetch-start  0 \\
                --fetch-end    510 \\
                --segments-dir test_output \\
                --verbose

            # Point directly at an S3 segments key:
            python tests/test_extract_embeddings.py \\
                --model-bucket model-bucket \\
                --bucket       my-bucket \\
                --prefix       jobs/job-abc \\
                --name         audio.wav \\
                --chunk-idx    0 \\
                --fetch-start  0 \\
                --fetch-end    510 \\
                --segments-key jobs/job-abc/segments/segments_0.json \\
                --use-int8     0 \\
                --verbose
        """),
    )
    parser.add_argument("--model-bucket",  required=True,
                        help="S3 bucket containing model artefacts")
    parser.add_argument("--bucket",        required=True,
                        help="S3 bucket containing the audio file")
    parser.add_argument("--prefix",        required=True,
                        help="S3 key prefix (e.g. jobs/job-abc)")
    parser.add_argument("--name",          required=True,
                        help="Audio filename (e.g. audio.wav)")
    parser.add_argument("--chunk-idx",     type=int,   default=0,
                        help="Chunk index (default: 0)")
    parser.add_argument("--fetch-start",   type=float, required=True,
                        help="Fetch window start in seconds")
    parser.add_argument("--fetch-end",     type=float, required=True,
                        help="Fetch window end in seconds")
    parser.add_argument("--core-start",    type=float, default=None,
                        help="Core ownership start (default: same as fetch-start)")
    parser.add_argument("--core-end",      type=float, default=None,
                        help="Core ownership end   (default: same as fetch-end)")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--segments-dir", default="test_output",
                     help="Local dir with segments_{chunk_idx}.json "
                          "(default: test_output/)")
    src.add_argument("--segments-key",
                     help="Explicit S3 key for segments_{chunk_idx}.json")

    parser.add_argument("--min-duration",  type=float, default=1.5,
                        help="Minimum turn duration to embed in seconds "
                             "(default: 1.5)")
    parser.add_argument("--use-int8",      type=int,   default=1,
                        choices=[0, 1],
                        help="Use INT8 quantised model: 1=yes 0=no "
                             "(default: 1)")
    parser.add_argument("--output-dir",    default="test_output",
                        help="Directory to write embeddings_{chunk_idx}.json "
                             "(default: test_output/)")
    parser.add_argument("--handler-dir",
                        default=os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))),
                        help="Project root containing lib/ "
                             "(default: parent of tests/)")
    parser.add_argument("--verbose",       action="store_true",
                        help="Print per-turn embedding stats table")

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
