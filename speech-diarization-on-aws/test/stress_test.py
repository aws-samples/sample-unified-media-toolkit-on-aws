#!/usr/bin/env python3
"""
stress_test_pipeline.py
========================
Stress test the whisper stt Step Functions pipeline by running
N parallel executions against copies of the same input audio file.

What it does
────────────
  1. Parses the source S3 URI and resolves the state machine ARN
  2. Copies the audio file to {prefix}/{n}/{filename} for each iteration
  3. Starts all N state machine executions in parallel (one per copy)
  4. Polls execution status until all complete or fail
  5. Prints a summary table with status, duration, and estimated cost

Usage
──────
  python stress_test_pipeline.py \\
      --s3uri       s3://my-bucket/path/to/audio_en.wav \\
      --state_machine_arn arn:aws:states:us-east-1:123456789:stateMachine:AudioStemSeparation \\
      [--n          5] \\
      [--prefix     _testing] \\
      [--model_bucket   my-bucket] \\
      [--model_prefix   _model_artefacts] \\
      [--poll_interval  15]

Output
──────
  Console: live status polling + final summary table
  s3://{bucket}/{prefix}/{n}/{filename}  ← copied audio files
  Each execution writes its own outputs under the same prefix.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

_s3 = boto3.client("s3")
_sfn = boto3.client("stepfunctions")


# =============================================================================
# HELPERS
# =============================================================================

def parse_s3_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/key' → (bucket, key)"""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"S3 URI must include a key: {uri}")
    return parts[0], parts[1]


def s3_copy(src_bucket: str, src_key: str,
            dst_bucket: str, dst_key: str) -> None:
    _s3.copy_object(
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Bucket=dst_bucket,
        Key=dst_key,
    )


def _execution_name(base: str, n: int, total: int) -> str:
    """stress-prefix-3of5-20250315T160423"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    width = len(str(total))
    return f"stress-{base}-{n:0{width}d}of{total}-{ts}"


def _build_execution_input(
    bucket:         str,
    audio_key:      str,
    model_bucket:   str,
    model_prefix:   str,
    cf_host:        str,
    model_variant:  str,
    language:       str | None,
    iteration:      int,
) -> dict:
    """Build the Step Functions input payload for one execution."""
    input_params = {
        "uuid":   f"stress-test-{iteration:03d}",
        "input": {
            "bucket":    bucket,
            "audio_key": audio_key,
        },
        "data": {
            "model_bucket": model_bucket,
            "model_prefix": model_prefix,
            "cf_host": cf_host,
        },
    }

    if language is not None:
        input_params["data"]["language"] = language

    if model_variant != "turbo":
        input_params["data"]["asr"] = {
            "model_id": "whisper",
            "model_variant": model_variant,
        }
    return input_params


# =============================================================================
# COPY STAGE
# =============================================================================

def copy_files(
    src_bucket: str,
    src_key:    str,
    dst_bucket: str,
    prefix:     str,
    n:          int,
) -> list[str]:
    """
    Copy audio file to {prefix}/{i}/{filename} for i in range(n).
    Returns list of destination keys.
    """
    filename = src_key.split("/")[-1]
    dst_keys = [
        f"{prefix.strip('/')}/{i}/{filename}"
        for i in range(n)
    ]

    print(
        f"\n── Copying {n} file(s) to s3://{dst_bucket}/{prefix.strip('/')}/")

    def _copy(i: int) -> str:
        dst_key = dst_keys[i]
        s3_copy(src_bucket, src_key, dst_bucket, dst_key)
        print(f"   [{i}] s3://{dst_bucket}/{dst_key}")
        return dst_key

    with ThreadPoolExecutor(max_workers=min(n, 10)) as pool:
        futures = {pool.submit(_copy, i): i for i in range(n)}
        results = [None] * n
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    print(f"   ✅ {n} file(s) copied")
    return results


# =============================================================================
# EXECUTION STAGE
# =============================================================================

def start_executions(
    state_machine_arn:      str,
    dst_bucket:             str,
    dst_keys:               list[str],
    model_bucket:           str,
    model_prefix:           str,
    cf_host:                str,
    model_variant:          str,
    language:               str | None,
    base_name:              str,
) -> list[dict]:
    """
    Start one execution per destination key.
    Returns list of { index, execution_arn, name, audio_key, started_at }.
    """
    n = len(dst_keys)
    print(f"\n── Starting {n} execution(s)")

    executions = []
    for i, audio_key in enumerate(dst_keys):
        name = _execution_name(base_name, i, n)
        payload = _build_execution_input(
            bucket=dst_bucket,
            audio_key=audio_key,
            model_bucket=model_bucket,
            model_prefix=model_prefix,
            cf_host=cf_host,
            model_variant=model_variant,
            language=language,
            iteration=i,
        )
        resp = _sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=name,
            input=json.dumps(payload),
        )
        arn = resp["executionArn"]
        started_at = resp["startDate"]
        executions.append({
            "index":         i,
            "execution_arn": arn,
            "name":          name,
            "audio_key":     audio_key,
            "started_at":    started_at,
            "status":        "RUNNING",
            "stopped_at":    None,
            "duration_sec":  None,
            "estimated_cost": None,
            "error":         None,
        })
        print(f"   [{i}] {name}")
        print(f"        {arn}")

    print(f"   ✅ {n} execution(s) started")
    return executions


# =============================================================================
# POLLING STAGE
# =============================================================================

def poll_executions(
    executions:    list[dict],
    poll_interval: int,
) -> list[dict]:
    """
    Poll all executions until they all reach a terminal state.
    Updates each execution dict in-place with final status/duration/cost.
    Returns the updated list.
    """
    terminal = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
    n = len(executions)

    print(f"\n── Polling {n} execution(s) every {poll_interval}s ...")
    print(f"   (Ctrl+C to stop polling — executions will continue running)\n")

    while True:
        pending = [e for e in executions if e["status"] not in terminal]
        if not pending:
            break

        for exe in pending:
            try:
                resp = _sfn.describe_execution(
                    executionArn=exe["execution_arn"])
                status = resp["status"]
                exe["status"] = status

                if status in terminal:
                    stopped_at = resp.get("stopDate")
                    exe["stopped_at"] = stopped_at
                    started_ts = exe["started_at"].timestamp() \
                        if hasattr(exe["started_at"], "timestamp") \
                        else float(exe["started_at"])
                    stopped_ts = stopped_at.timestamp() \
                        if hasattr(stopped_at, "timestamp") \
                        else float(stopped_at)
                    exe["duration_sec"] = round(stopped_ts - started_ts, 1)

                    # Pull estimated cost from usage_tracking if SUCCEEDED
                    if status == "SUCCEEDED":
                        try:
                            out = json.loads(resp.get("output", "{}"))
                            exe["estimated_cost"] = (
                                out.get("data", {})
                                   .get("usage_tracking", {})
                                   .get("total_estimated_cost")
                            )
                        except (json.JSONDecodeError, AttributeError):
                            pass

                    # Capture error cause for failed executions
                    if status in {"FAILED", "TIMED_OUT", "ABORTED"}:
                        exe["error"] = resp.get("cause", status)

            except ClientError as exc:
                print(
                    f"   ⚠️  describe_execution error [{exe['index']}]: {exc}")

        # Print current status line
        counts = {}
        for e in executions:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        status_str = "  ".join(f"{s}={c}" for s, c in sorted(counts.items()))
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"   {ts}  {status_str}")

        if not [e for e in executions if e["status"] not in terminal]:
            break

        time.sleep(poll_interval)

    return executions


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(executions: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("STRESS TEST SUMMARY")
    print("=" * 80)

    col_w = [5, 10, 14, 14, 16, 30]
    header = (
        f"{'IDX':<{col_w[0]}}"
        f"{'STATUS':<{col_w[1]}}"
        f"{'DURATION(s)':<{col_w[2]}}"
        f"{'EST. COST($)':<{col_w[3]}}"
        f"{'AUDIO KEY':<{col_w[4]}}"
        f"{'EXECUTION NAME'}"
    )
    print(header)
    print("-" * 80)

    total_cost = 0.0
    total_duration = 0.0
    succeeded = 0

    for e in executions:
        dur = f"{e['duration_sec']:.1f}" if e["duration_sec"] else "—"
        cost = f"{e['estimated_cost']:.6f}" if e["estimated_cost"] else "—"
        audio_short = "/".join(e["audio_key"].split("/")[-3:])
        name_short = e["name"]

        status_icon = {
            "SUCCEEDED": "✅",
            "FAILED":    "❌",
            "TIMED_OUT": "⏱️",
            "ABORTED":   "🚫",
            "RUNNING":   "🔄",
        }.get(e["status"], "?")

        print(
            f"{e['index']:<{col_w[0]}}"
            f"{status_icon} {e['status']:<{col_w[1]-2}}"
            f"{dur:<{col_w[2]}}"
            f"{cost:<{col_w[3]}}"
            f"{audio_short:<{col_w[4]}}"
            f"{name_short}"
        )

        if e["status"] == "SUCCEEDED":
            succeeded += 1
            if e["duration_sec"]:
                total_duration += e["duration_sec"]
            if e["estimated_cost"]:
                total_cost += e["estimated_cost"]

        if e["error"]:
            # Print truncated error on next line
            err = e["error"][:120] + ("..." if len(e["error"]) > 120 else "")
            print(f"  {'':>{col_w[0]}}  ⤷ {err}")

    n = len(executions)
    print("-" * 80)
    print(f"  {succeeded}/{n} succeeded")
    if succeeded:
        print(f"  avg duration : {total_duration / succeeded:.1f}s")
        print(f"  total cost   : ${total_cost:.6f}")
        print(f"  cost/run     : ${total_cost / succeeded:.6f}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress test the STT Whisper Step Functions pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--s3uri", required=True,
        help="S3 URI of the source audio WAV file  e.g. s3://bucket/path/audio.wav",
    )
    parser.add_argument(
        "--state_machine_arn", required=True,
        help="ARN of the Step Functions state machine",
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of parallel test executions (default: 5)",
    )
    parser.add_argument(
        "--prefix", default="_testing",
        help="S3 prefix for copied test files (default: _testing)",
    )
    parser.add_argument(
        "--model_bucket", default=None,
        help="S3 bucket containing the model artefacts "
             "(default: same bucket as audio)",
    )
    parser.add_argument(
        "--model_prefix", default="_model_artefacts",
        help="S3 prefix of model artefacts (default: _model_artefacts)",
    )
    parser.add_argument(
        "--cf_host", default="1234.cloudfront.net",
        help="Host url to download models if not present in model_bucket (default: 1234.cloudfront.net)",
    )
    parser.add_argument(
        "--model_variant", default="turbo",
        help="Model variant [turbo|cy|ar]. turbo for multilingual, or specialized language Welsh [cy] or Arabic [ar] (default: turbo)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Specify ISO 2-letter code language. Skip the option to auto detect language",
    )
    parser.add_argument(
        "--poll_interval", type=int, default=15,
        help="Polling interval in seconds (default: 15)",
    )
    parser.add_argument(
        "--no_poll", action="store_true",
        help="Start executions and exit without polling",
    )
    args = parser.parse_args()

    # ── Parse source URI ──────────────────────────────────────────────────────
    src_bucket, src_key = parse_s3_uri(args.s3uri)
    dst_bucket = src_bucket
    model_bucket = args.model_bucket or src_bucket
    filename = src_key.split("/")[-1]
    base_name = args.prefix[:20]
    # base_name           = filename.rsplit(".", 1)[0][:20]  # trim for execution name

    print(f"\n{'='*60}")
    print(f"  STT Whisper/Pyannote — stress test")
    print(f"{'='*60}")
    print(f"  source        : s3://{src_bucket}/{src_key}")
    print(f"  runs          : {args.n}")
    print(f"  prefix        : {args.prefix}")
    print(f"  variant       : {args.model_variant}")
    print(f"  language      : {args.language or 'auto-detect'}")
    print(f"  sm arn        : {args.state_machine_arn}")
    print(f"{'='*60}")

    # ── Stage 1: Copy files ───────────────────────────────────────────────────
    dst_keys = copy_files(
        src_bucket=src_bucket,
        src_key=src_key,
        dst_bucket=dst_bucket,
        prefix=args.prefix,
        n=args.n,
    )

    # ── Stage 2: Start executions ─────────────────────────────────────────────
    executions = start_executions(
        state_machine_arn=args.state_machine_arn,
        dst_bucket=dst_bucket,
        dst_keys=dst_keys,
        model_bucket=model_bucket,
        model_prefix=args.model_prefix,
        cf_host=args.cf_host,
        model_variant=args.model_variant,
        language=args.language,
        base_name=base_name,
    )

    if args.no_poll:
        print("\n  --no_poll set — exiting without waiting for results.")
        print("  Execution ARNs:")
        for e in executions:
            print(f"    [{e['index']}] {e['execution_arn']}")
        sys.exit(0)

    # ── Stage 3: Poll ─────────────────────────────────────────────────────────
    try:
        executions = poll_executions(executions, args.poll_interval)
    except KeyboardInterrupt:
        print("\n\n  Polling interrupted — executions are still running.")

    # ── Stage 4: Summary ──────────────────────────────────────────────────────
    print_summary(executions)


if __name__ == "__main__":
    main()
