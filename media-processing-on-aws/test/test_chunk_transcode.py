"""
Local test script for StateChunkTranscode.

Two modes:
  1. --from-probe  Feed output from test_probe.py directly and run one chunk iterator.
                  Useful to test the full probe→transcode hand-off without Step Functions.

  2. --iterator   Pass a raw iterator JSON file (or '-' to read stdin).
                  Useful to replay a specific iterator from a previous probe run.

Usage:
    cd docker/media-processing

    # Mode 1: pipe probe output, pick chunk index 0
    python test/test_probe.py --s3uri s3://bucket/video.mxf \
        --output-s3uri s3://proxy/out --pretty \
        | python test/test_chunk_transcode.py --from-probe --chunk 0

    # Mode 2: replay a saved iterator JSON
    python test/test_chunk_transcode.py --iterator /tmp/iterator_0.json

Options:
    --chunk N       Which iterator index to run in --from-probe mode (default: 0)
    --pretty        Pretty-print the returned output JSON
    --save-iterator Save the selected iterator to a file for reuse
"""
from __future__ import annotations

import argparse
import json
import sys


def _load_iterator_from_probe(data: dict, chunk_index: int) -> dict:
    iterators = data.get("data", {}).get("iterators")
    if not iterators:
        print("ERROR: probe output has no data.iterators", file=sys.stderr)
        sys.exit(1)
    if chunk_index >= len(iterators):
        print(
            f"ERROR: --chunk {chunk_index} out of range (probe produced {len(iterators)} iterators)",
            file=sys.stderr,
        )
        sys.exit(1)
    return iterators[chunk_index]


def _build_event(iterator: dict) -> dict:
    return {
        "operation": "StateChunkTranscode",
        "item_data": iterator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test StateChunkTranscode locally")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-probe",
        action="store_true",
        help="Read probe output JSON from stdin",
    )
    src.add_argument(
        "--iterator",
        metavar="FILE",
        help="Path to a saved iterator JSON file, or '-' for stdin",
    )

    parser.add_argument("--chunk",          type=int, default=0, help="Iterator index (--from-probe mode, default 0)")
    parser.add_argument("--pretty",         action="store_true", help="Pretty-print output JSON")
    parser.add_argument("--save-iterator",  metavar="FILE", default=None, help="Save the selected iterator to FILE")
    args = parser.parse_args()

    if args.from_probe:
        raw = sys.stdin.read()
        try:
            probe_output = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"ERROR: failed to parse probe output JSON: {e}", file=sys.stderr)
            sys.exit(1)
        iterator = _load_iterator_from_probe(probe_output, args.chunk)
    else:
        src_file = args.iterator
        if src_file == "-":
            raw = sys.stdin.read()
        else:
            with open(src_file) as f:
                raw = f.read()
        try:
            iterator = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"ERROR: failed to parse iterator JSON: {e}", file=sys.stderr)
            sys.exit(1)

    if args.save_iterator:
        with open(args.save_iterator, "w") as f:
            json.dump(iterator, f, indent=2)
        print(f"Iterator saved to {args.save_iterator}", file=sys.stderr)

    event = _build_event(iterator)
    print("=== Input event ===", file=sys.stderr)
    print(json.dumps(event, indent=2), file=sys.stderr)
    print("===================", file=sys.stderr)

    from states.chunk_transcode.lambda_function import lambda_handler
    result = lambda_handler(event, None)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
