#!/usr/bin/env python3
"""
mediaconvert_probe_test.py — Test AWS MediaConvert Probe API against MXF files.

Usage
-----
  python mediaconvert_probe_test.py \
      s3://bucket/path/to/file1.mxf \
      s3://bucket/path/to/file2.mxf

  # Save full JSON response to files
  python mediaconvert_probe_test.py --save s3://bucket/file.mxf

  # Specify region
  python mediaconvert_probe_test.py --region eu-west-1 s3://bucket/file.mxf

Output
------
For each file, prints a summary:
  - Container format + duration
  - Frame count (if available)
  - Video tracks: codec, fps, resolution, timecode
  - Audio tracks: codec, sample rate, channels
  - Warnings if duration/frame count are missing

Full JSON response saved to {filename}_probe.json if --save is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


# =============================================================================
# MediaConvert client helpers
# =============================================================================

_mc_endpoint_cache: dict[str, str] = {}


def _get_mediaconvert_client(session: boto3.Session, region: str):
    """
    MediaConvert requires a per-account regional endpoint.
    Discovers and caches it on first call.
    """
    if region not in _mc_endpoint_cache:
        mc = session.client("mediaconvert", region_name=region)
        try:
            resp = mc.describe_endpoints(Mode="DEFAULT")
            endpoint_url = resp["Endpoints"][0]["Url"]
            _mc_endpoint_cache[region] = endpoint_url
            print(f"  MediaConvert endpoint: {endpoint_url}")
        except ClientError as exc:
            raise RuntimeError(
                f"Failed to discover MediaConvert endpoint: {exc}"
            ) from exc

    return session.client(
        "mediaconvert",
        region_name=region,
        endpoint_url=_mc_endpoint_cache[region],
    )


# =============================================================================
# Probe
# =============================================================================

def _probe_file(
    mc_client,
    s3_url: str,
) -> dict:
    """Call MediaConvert Probe API for a single S3 URL. Returns raw response."""
    t0 = time.monotonic()
    resp = mc_client.probe(InputFiles=[{"FileUrl": s3_url}])
    elapsed = round(time.monotonic() - t0, 2)
    print(f"  Probe completed in {elapsed}s")
    return resp


# =============================================================================
# Result parsing
# =============================================================================

def _fmt_fps(fps_obj: dict | None) -> str:
    if not fps_obj:
        return "unknown"
    num = fps_obj.get("Numerator", 0)
    den = fps_obj.get("Denominator", 1)
    if num and den:
        fps = num / den
        return f"{num}/{den} ({fps:.4f} fps)"
    return "unknown"


def _summarise(s3_url: str, probe_results: list[dict]) -> None:
    """Print a human-readable summary of probe results."""
    filename = s3_url.split("/")[-1]
    print()
    print(f"{'=' * 60}")
    print(f"File: {filename}")
    print(f"URL:  {s3_url}")
    print(f"{'=' * 60}")

    if not probe_results:
        print("  ⚠️  No probe results returned")
        return

    for result_idx, result in enumerate(probe_results):
        container = result.get("Container", {})
        tracks = result.get("Tracks", [])

        # ── Container ─────────────────────────────────────────────────────────
        fmt = container.get("Format", "unknown")
        duration = container.get("Duration")

        print(f"\nContainer:")
        print(f"  Format:   {fmt}")

        if duration is not None:
            print(
                f"  Duration: {duration:.3f}s  ({_seconds_to_tc(duration, 25)})  ✅")
        else:
            print(f"  Duration: ⚠️  NOT AVAILABLE")

        # ── Tracks ─────────────────────────────────────────────────────────────
        video_tracks = [t for t in tracks if t.get("TrackType") == "Video"]
        audio_tracks = [t for t in tracks if t.get("TrackType") == "Audio"]
        data_tracks = [t for t in tracks if t.get(
            "TrackType") not in ("Video", "Audio")]

        print(f"\nTracks: {len(tracks)} total  "
              f"({len(video_tracks)} video  "
              f"{len(audio_tracks)} audio  "
              f"{len(data_tracks)} data)")

        for i, vt in enumerate(video_tracks):
            vd = vt.get("VideoTrack", {})
            print(f"\n  Video Track #{i}:")
            print(f"    Codec:       {vd.get('Codec', 'unknown')}")
            print(f"    Profile:     {vd.get('CodecProfile', 'unknown')}")
            print(f"    FPS:         {_fmt_fps(vd.get('FrameRate'))}")
            print(
                f"    Resolution:  {vd.get('Width', '?')} × {vd.get('Height', '?')}")
            print(f"    Interlaced:  {vd.get('Interlaced', 'unknown')}")
            print(f"    Bit depth:   {vd.get('BitDepth', 'unknown')}")
            print(f"    Bitrate:     {_fmt_bitrate(vd.get('BitRate'))}")
            print(f"    Color space: {vd.get('ColorSpace', 'unknown')}")
            print(f"    Pixel fmt:   {vd.get('PixelFormat', 'unknown')}")

            # Frame count — not always present
            frame_count = vd.get("FrameCount") or vd.get("NumberOfFrames")
            if frame_count:
                print(f"    Frame count: {frame_count}  ✅")
            else:
                print(f"    Frame count: ⚠️  NOT AVAILABLE")

            # Timecode
            tc = vd.get("StartTimecode") or vd.get("Timecode")
            if tc:
                print(f"    Timecode:    {tc}  ✅")
            else:
                print(f"    Timecode:    ⚠️  NOT AVAILABLE")

            # All remaining fields for debugging
            known = {
                "Codec", "CodecProfile", "FrameRate", "Width", "Height",
                "Interlaced", "BitDepth", "BitRate", "ColorSpace",
                "PixelFormat", "FrameCount", "NumberOfFrames",
                "StartTimecode", "Timecode",
            }
            extra = {k: v for k, v in vd.items() if k not in known}
            if extra:
                print(f"    Extra fields: {json.dumps(extra, indent=6)}")

        for i, at in enumerate(audio_tracks):
            ad = at.get("AudioTrack", {})
            print(f"\n  Audio Track #{i}:")
            print(f"    Codec:       {ad.get('Codec', 'unknown')}")
            print(f"    Sample rate: {ad.get('SampleRate', 'unknown')} Hz")
            print(f"    Channels:    {ad.get('Channels', 'unknown')}")
            print(f"    Bit depth:   {ad.get('BitDepth', 'unknown')}")
            print(f"    Bitrate:     {_fmt_bitrate(ad.get('BitRate'))}")

    # ── Assessment ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Assessment:")

    all_results_have_duration = all(
        r.get("Container", {}).get("Duration") is not None
        for r in probe_results
    )
    all_results_have_frames = any(
        t.get("VideoTrack", {}).get("FrameCount")
        or t.get("VideoTrack", {}).get("NumberOfFrames")
        for r in probe_results
        for t in r.get("Tracks", [])
        if t.get("TrackType") == "Video"
    )

    if all_results_have_duration:
        print("  ✅ Duration available — fast probe path viable")
    else:
        print("  ❌ Duration unavailable — full scan still required")

    if all_results_have_frames:
        print("  ✅ Frame count available — no scan needed")
    else:
        print("  ⚠️  Frame count not in response — may need to derive from duration × fps")


def _seconds_to_tc(seconds: float, fps: float) -> str:
    """Convert seconds to approximate HH:MM:SS:FF timecode for display."""
    total_frames = round(seconds * fps)
    f = total_frames % int(fps)
    s = (total_frames // int(fps)) % 60
    m = (total_frames // (int(fps) * 60)) % 60
    h = total_frames // (int(fps) * 3600)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def _fmt_bitrate(br) -> str:
    if br is None:
        return "unknown"
    mbps = br / 1_000_000
    return f"{mbps:.1f} Mbps"


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test AWS MediaConvert Probe API against MXF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "s3_urls",
        nargs="+",
        metavar="S3_URL",
        help="S3 URLs to probe (e.g. s3://bucket/path/to/file.mxf)",
    )
    parser.add_argument(
        "--region", "-r",
        default=None,
        help="AWS region (default: from boto3 session)",
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save full JSON response to {filename}_mc_probe.json",
    )
    args = parser.parse_args()

    session_kwargs = {}
    if args.region:
        session_kwargs["region_name"] = args.region
    session = boto3.Session(**session_kwargs)
    region = args.region or session.region_name

    if not region:
        print("ERROR: Could not determine AWS region.", file=sys.stderr)
        return 1

    print(f"Region: {region}")

    try:
        mc = _get_mediaconvert_client(session, region)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    any_error = False

    for s3_url in args.s3_urls:
        if not s3_url.startswith("s3://"):
            print(f"\nERROR: Not an S3 URL: {s3_url}", file=sys.stderr)
            any_error = True
            continue

        print(f"\nProbing: {s3_url}")
        try:
            resp = _probe_file(mc, s3_url)
            probe_results = resp.get("ProbeResults", [])
            _summarise(s3_url, probe_results)

            if args.save:
                # Strip ResponseMetadata for cleaner output
                clean = {"ProbeResults": probe_results}
                filename = s3_url.split(
                    "/")[-1].replace(".mxf", "").replace(".MXF", "")
                output_path = f"{filename}_mc_probe.json"
                with open(output_path, "w") as f:
                    json.dump(clean, f, indent=2, default=str)
                print(f"\n  Full response saved: {output_path}")

        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            msg = exc.response["Error"]["Message"]
            print(f"\n  ❌ AWS error [{code}]: {msg}", file=sys.stderr)
            any_error = True
        except Exception as exc:
            print(f"\n  ❌ Error: {exc}", file=sys.stderr)
            any_error = True

    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
