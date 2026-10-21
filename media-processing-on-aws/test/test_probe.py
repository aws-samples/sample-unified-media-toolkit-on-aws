"""
Local test script for StateProbeMediaFile.

Usage:
    cd docker/media-processing
    python test/test_probe.py \
        --s3uri s3://<your-ingest-bucket>/path/to/source.mxf \
        --output-s3uri s3://<your-proxy-bucket>/output/test-run \
        [--groups mp4,wav,m4a,jpeg,metadata] \
        [--detail default|full] \
        [--width 960] [--height 540] \
        [--crop-x 0] [--crop-y 0] \
        [--mark-in 0] [--mark-out 1800] \
        [--channel-mapping 1,2] \
        [--pretty]

The script calls StateProbeMediaFile directly (no Lambda / Step Functions needed)
and prints the returned event to stdout. Use --pretty for readable JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid


def _build_event(args: argparse.Namespace) -> dict:
    input_config: dict = {}
    if args.crop_x:
        input_config["crop_x"] = args.crop_x
    if args.crop_y:
        input_config["crop_y"] = args.crop_y
    if args.width:
        input_config["width"] = args.width
    if args.height:
        input_config["height"] = args.height
    if args.mark_in is not None or args.mark_out is not None:
        input_config["use_partial"] = True
        input_config["mark_in_out"] = [args.mark_in or 0, args.mark_out or 0]
    if args.channel_mapping:
        input_config["use_channel_mapping"] = True
        input_config["audio_channel_mapping"] = [int(c) for c in args.channel_mapping.split(",")]

    output_groups = []
    if args.groups:
        fmt_to_defaults = {
            "mp4":      {"group_name": "proxy",         "format": "mp4"},
            "wav":      {"group_name": "audio_only",    "format": "wav"},
            "m4a":      {"group_name": "audio_m4a",     "format": "m4a"},
            "jpeg":     {"group_name": "frame_capture", "format": "jpeg"},
            "png":      {"group_name": "frame_capture", "format": "png"},
            "metadata": {"group_name": "metadata",      "format": "metadata"},
        }
        for fmt in args.groups.split(","):
            fmt = fmt.strip()
            if fmt not in fmt_to_defaults:
                print(f"Unknown format {fmt!r} — valid: {', '.join(fmt_to_defaults)}",
                      file=sys.stderr)
                sys.exit(1)
            group = dict(fmt_to_defaults[fmt])
            if fmt == "metadata" and args.detail:
                group["detail"] = args.detail
            output_groups.append(group)

    event: dict = {
        "operation": "StateProbeMediaFile",
        "uuid":      args.run_uuid,
        "input": {
            "s3uri": args.s3uri,
        },
        "output": {
            "s3uri": args.output_s3uri,
        },
        "data": {},
    }
    if input_config:
        event["input"]["input_config"] = input_config
    if output_groups:
        event["output"]["output_groups"] = output_groups

    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Test StateProbeMediaFile locally")
    parser.add_argument("--s3uri",        required=True, help="Source s3uri (s3://bucket/key)")
    parser.add_argument("--output-s3uri", required=True, help="Output s3uri prefix (s3://bucket/prefix)")
    parser.add_argument("--groups",       default=None,  help="Comma-separated output formats: mp4,wav,m4a,jpeg,png,metadata")
    parser.add_argument("--detail",       default=None,  choices=["default", "full"],
                        help="metadata group detail level")
    parser.add_argument("--width",        type=int, default=0)
    parser.add_argument("--height",       type=int, default=0)
    parser.add_argument("--crop-x",       type=int, default=0)
    parser.add_argument("--crop-y",       type=int, default=0)
    parser.add_argument("--mark-in",      type=float, default=None)
    parser.add_argument("--mark-out",     type=float, default=None)
    parser.add_argument("--channel-mapping", default=None, help="e.g. 1,2")
    parser.add_argument("--uuid",         dest="run_uuid", default=str(uuid.uuid4()))
    parser.add_argument("--pretty",       action="store_true", help="Pretty-print output JSON")
    args = parser.parse_args()

    event = _build_event(args)
    print("=== Input event ===", file=sys.stderr)
    print(json.dumps(event, indent=2), file=sys.stderr)
    print("===================", file=sys.stderr)

    from states.probe_media_file.lambda_function import lambda_handler
    result = lambda_handler(event, None)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
