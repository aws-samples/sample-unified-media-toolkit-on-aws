#!/usr/bin/env python3
"""
stress_test_pipeline.py
========================
Stress test the DinoV2 frame embedding Step Functions pipeline by running
N parallel executions against the same source video file in the ingest bucket.

What it does
------------
  1. Parses the source S3 URI and validates the file exists
  2. Starts N state machine executions in parallel, all pointing at the same
     source video — each with a unique uuid so proxy outputs never collide
  3. Polls execution status until all complete or fail
  4. Prints a summary table with status, duration, and estimated cost

Usage
-----
  python stress_test_pipeline.py \\
      --s3uri               s3://ingest-bucket/path/to/source.mxf \\
      --state_machine_arn   arn:aws:states:us-east-1:123456789:stateMachine:DinoV2FrameEmbedding \\
      [--n                      3] \\
      [--proxy_bucket           proxy-bucket] \\
      [--proxy_prefix           _testing/proxy] \\
      [--model_bucket           model-bucket] \\
      [--model_prefix           _model_artefacts] \\
      [--segment_duration_secs  1800] \\
      [--overlap_secs           2.0] \\
      [--crop_x                 0] \\
      [--crop_y                 0] \\
      [--poll_interval          20] \\
      [--no_poll]

Output
------
  Console: live status polling + final summary table
  Each execution writes proxy/probe/embeddings under:
    s3://{proxy_bucket}/{proxy_prefix}/{uuid}/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

_s3 = boto3.client("s3")
_sfn = boto3.client("stepfunctions")


# =============================================================================
# HELPERS
# =============================================================================

def parse_s3_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/key' -> (bucket, key)"""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"S3 URI must include a key: {uri}")
    return parts[0], parts[1]


def s3_object_exists(bucket: str, key: str) -> bool:
    """Return True if the S3 object exists."""
    try:
        _s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _execution_name(base: str, n: int, total: int) -> str:
    """stress-base-02of05-20260319T143022"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    width = len(str(total))
    slug = base.strip("/").replace("/", "-")[:20]
    return f"stress-{slug}-{n:0{width}d}of{total}-{ts}"


def _build_execution_input(
    src_bucket:            str,
    video_key:             str,
    proxy_bucket:          str,
    proxy_prefix:          str,
    model_bucket:          str,
    model_prefix:          str,
    segment_duration_secs: int,
    overlap_secs:          float,
    crop_x:                int,
    crop_y:                int,
    uuid:                  str,
) -> dict:
    """Build the Step Functions input payload for one execution."""
    payload: dict = {
        "uuid":  uuid,
        "input": {
            "bucket":    src_bucket,
            "video_key": video_key,
        },
        "data": {
            "model_bucket": model_bucket,
            "model_prefix": model_prefix,
            "transcode": {
                "segment_duration_secs":       segment_duration_secs,
                "overlap_secs":                overlap_secs,
                "max_keyframe_drift_secs":     5,
                "wide_overlap_threshold_secs": 15,
            },
            "iterators": [],
        },
    }

    # Optional input cropping — only injected when non-zero
    if crop_x > 0 or crop_y > 0:
        payload["data"]["media_preprocess"] = {
            "input_cropping": {
                "crop_x":            crop_x,
                "crop_y":            crop_y,
                "keep_aspect_ratio": True,
            }
        }

    return payload


# =============================================================================
# EXECUTION STAGE
# =============================================================================

def start_executions(
    state_machine_arn:     str,
    src_bucket:            str,
    video_key:             str,
    proxy_bucket:          str,
    proxy_prefix:          str,
    model_bucket:          str,
    model_prefix:          str,
    segment_duration_secs: int,
    overlap_secs:          float,
    crop_x:                int,
    crop_y:                int,
    n:                     int,
) -> list[dict]:
    """
    Start N executions all pointing at the same source video.
    Each gets a unique uuid so proxy outputs are isolated.
    Returns list of execution tracking dicts.
    """
    print(f"\n-- Starting {n} execution(s)")

    executions = []
    for i in range(n):
        uuid = f"stress-{i:03d}-{int(time.time())}"
        name = _execution_name("dinov2", i, n)
        payload = _build_execution_input(
            src_bucket=src_bucket,
            video_key=video_key,
            proxy_bucket=proxy_bucket,
            proxy_prefix=proxy_prefix,
            model_bucket=model_bucket,
            model_prefix=model_prefix,
            segment_duration_secs=segment_duration_secs,
            overlap_secs=overlap_secs,
            crop_x=crop_x,
            crop_y=crop_y,
            uuid=uuid,
        )

        resp = _sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=name,
            input=json.dumps(payload),
        )
        arn = resp["executionArn"]
        started_at = resp["startDate"]

        executions.append({
            "index":          i,
            "uuid":           uuid,
            "execution_arn":  arn,
            "name":           name,
            "video_key":      video_key,
            "started_at":     started_at,
            "status":         "RUNNING",
            "stopped_at":     None,
            "duration_sec":   None,
            "segment_count":  None,
            "probe_s3_uri":   None,
            "estimated_cost": None,
            "error":          None,
        })
        print(f"   [{i}] {name}")
        print(f"        uuid={uuid}")
        print(f"        arn={arn}")

    print(f"   OK  {n} execution(s) started")
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
    Updates each execution dict in-place.
    Returns the updated list.
    """
    terminal = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
    n = len(executions)

    print(f"\n-- Polling {n} execution(s) every {poll_interval}s ...")
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

                    started_ts = (
                        exe["started_at"].timestamp()
                        if hasattr(exe["started_at"], "timestamp")
                        else float(exe["started_at"])
                    )
                    stopped_ts = (
                        stopped_at.timestamp()
                        if hasattr(stopped_at, "timestamp")
                        else float(stopped_at)
                    )
                    exe["duration_sec"] = round(stopped_ts - started_ts, 1)

                    if status == "SUCCEEDED":
                        try:
                            out = json.loads(resp.get("output", "{}"))
                            data = out.get("data", {})

                            # Probe URI written by StateProbeVideoForChunkTranscode
                            exe["probe_s3_uri"] = data.get("probe_s3_uri")

                            # Segment count from iterators after transcode map
                            exe["segment_count"] = len(
                                data.get("iterators", []))

                            # Estimated cost written by StateUsageTracking
                            exe["estimated_cost"] = (
                                data.get("usage_tracking", {})
                                    .get("total_estimated_cost")
                            )
                        except (json.JSONDecodeError, AttributeError):
                            pass

                    if status in {"FAILED", "TIMED_OUT", "ABORTED"}:
                        exe["error"] = resp.get("cause", status)

            except ClientError as exc:
                print(f"   WARNING describe_execution [{exe['index']}]: {exc}")

        # Live status line
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
    print("\n" + "=" * 100)
    print("STRESS TEST SUMMARY — DinoV2 Frame Embedding Pipeline")
    print("=" * 100)

    col = [5, 12, 13, 13, 10, 24]
    print(
        f"{'IDX':<{col[0]}}"
        f"{'STATUS':<{col[1]}}"
        f"{'DURATION(s)':<{col[2]}}"
        f"{'EST.COST($)':<{col[3]}}"
        f"{'SEGMENTS':<{col[4]}}"
        f"{'UUID':<{col[5]}}"
        f"VIDEO KEY"
    )
    print("-" * 100)

    total_cost = 0.0
    total_duration = 0.0
    succeeded = 0

    for e in executions:
        dur = f"{e['duration_sec']:.1f}" if e["duration_sec"] else "--"
        cost = f"{e['estimated_cost']:.6f}" if e["estimated_cost"] else "--"
        segments = str(e["segment_count"]) if e["segment_count"] else "--"
        vid_short = "/".join(e["video_key"].split("/")[-3:])

        icon = {
            "SUCCEEDED": "OK ",
            "FAILED":    "ERR",
            "TIMED_OUT": "TMO",
            "ABORTED":   "ABT",
            "RUNNING":   "...",
        }.get(e["status"], "?  ")

        print(
            f"{e['index']:<{col[0]}}"
            f"{icon} {e['status']:<{col[1]-4}}"
            f"{dur:<{col[2]}}"
            f"{cost:<{col[3]}}"
            f"{segments:<{col[4]}}"
            f"{e['uuid']:<{col[5]}}"
            f"{vid_short}"
        )

        if e.get("probe_s3_uri"):
            print(f"{'':>{col[0]+2}}probe  -> {e['probe_s3_uri']}")

        if e.get("error"):
            err = e["error"][:120] + ("..." if len(e["error"]) > 120 else "")
            print(f"{'':>{col[0]+2}}error  -> {err}")

        if e["status"] == "SUCCEEDED":
            succeeded += 1
            if e["duration_sec"]:
                total_duration += e["duration_sec"]
            if e["estimated_cost"]:
                total_cost += e["estimated_cost"]

    n = len(executions)
    print("-" * 100)
    print(f"  {succeeded}/{n} succeeded")
    if succeeded:
        print(f"  avg duration : {total_duration / succeeded:.1f}s")
        if total_cost > 0:
            print(f"  total cost   : ${total_cost:.6f}")
            print(f"  cost/run     : ${total_cost / succeeded:.6f}")
    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress test the DinoV2 frame embedding Step Functions pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--s3uri", required=True,
        help="S3 URI of the source video  e.g. s3://ingest-bucket/path/source.mxf",
    )
    parser.add_argument(
        "--state_machine_arn", required=True,
        help="ARN of the Step Functions state machine",
    )
    parser.add_argument(
        "--n", type=int, default=3,
        help="Number of parallel executions (default: 3)",
    )
    parser.add_argument(
        "--proxy_bucket", default=None,
        help="Proxy bucket for all pipeline outputs "
             "(default: ENV_PROXY_BUCKET env var or source bucket)",
    )
    parser.add_argument(
        "--proxy_prefix", default="_testing/proxy",
        help="S3 prefix for proxy outputs (default: _testing/proxy)",
    )
    parser.add_argument(
        "--model_bucket", default=None,
        help="S3 bucket containing model artefacts (default: source bucket)",
    )
    parser.add_argument(
        "--model_prefix", default="_model_artefacts",
        help="S3 prefix of model artefacts (default: _model_artefacts)",
    )
    parser.add_argument(
        "--segment_duration_secs", type=int, default=1800,
        help="Transcode segment duration in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--overlap_secs", type=float, default=2.0,
        help="Right-side segment overlap in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--crop_x", type=int, default=0,
        help="Horizontal input crop in pixels (default: 0 = disabled)",
    )
    parser.add_argument(
        "--crop_y", type=int, default=0,
        help="Vertical input crop in pixels e.g. 32 for VBI (default: 0 = disabled)",
    )
    parser.add_argument(
        "--poll_interval", type=int, default=20,
        help="Polling interval in seconds (default: 20)",
    )
    parser.add_argument(
        "--no_poll", action="store_true",
        help="Start executions and exit without polling",
    )
    args = parser.parse_args()

    # ── Resolve buckets ───────────────────────────────────────────────────────
    src_bucket, video_key = parse_s3_uri(args.s3uri)
    model_bucket = args.model_bucket or src_bucket
    proxy_bucket = (
        args.proxy_bucket
        or os.environ.get("ENV_PROXY_BUCKET")
        or src_bucket
    )

    # ── Validate source exists ────────────────────────────────────────────────
    if not s3_object_exists(src_bucket, video_key):
        print(f"ERROR: Source file not found: s3://{src_bucket}/{video_key}")
        sys.exit(1)

    # ── Print config ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DinoV2 Frame Embedding — stress test")
    print(f"{'='*70}")
    print(f"  source          : s3://{src_bucket}/{video_key}")
    print(f"  runs            : {args.n}")
    print(f"  proxy_bucket    : {proxy_bucket}")
    print(f"  proxy_prefix    : {args.proxy_prefix}/{{uuid}}/")
    print(f"  model_bucket    : {model_bucket}")
    print(f"  model_prefix    : {args.model_prefix}")
    print(f"  segment_secs    : {args.segment_duration_secs}")
    print(f"  overlap_secs    : {args.overlap_secs}")
    print(f"  crop_x/y        : {args.crop_x}/{args.crop_y}")
    print(f"  state machine   : {args.state_machine_arn}")
    print(f"{'='*70}")

    # ── Start executions ──────────────────────────────────────────────────────
    executions = start_executions(
        state_machine_arn=args.state_machine_arn,
        src_bucket=src_bucket,
        video_key=video_key,
        proxy_bucket=proxy_bucket,
        proxy_prefix=args.proxy_prefix,
        model_bucket=model_bucket,
        model_prefix=args.model_prefix,
        segment_duration_secs=args.segment_duration_secs,
        overlap_secs=args.overlap_secs,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        n=args.n,
    )

    if args.no_poll:
        print("\n  --no_poll set — exiting without waiting for results.")
        print("  Execution ARNs:")
        for e in executions:
            print(f"    [{e['index']}] {e['execution_arn']}")
        sys.exit(0)

    # ── Poll ──────────────────────────────────────────────────────────────────
    try:
        executions = poll_executions(executions, args.poll_interval)
    except KeyboardInterrupt:
        print("\n\n  Polling interrupted — executions are still running.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(executions)


if __name__ == "__main__":
    main()
