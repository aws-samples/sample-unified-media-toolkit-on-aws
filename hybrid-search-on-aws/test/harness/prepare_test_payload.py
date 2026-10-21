#!/usr/bin/env python3
"""
prepare_test_payload.py — generate a test index_content submission
(content contracts, README.md Design record §2.2) and print the SFN input.

Emits a JSONL manifest of contract documents to S3 and prints the
HybridSearchIndexSFN execution input referencing it. Embedding happens
in the backend (StatePrepareDocuments) — this script never calls Bedrock.

Usage:
  python test/harness/prepare_test_payload.py \
    --proxy-bucket <bucket> --asset-id asset:uuid:test-001 \
    [--collection-id default] [--doc-type scene] [--num-docs 5] \
    [--title "Test Asset"] [--genres "news,politics"]
"""
from __future__ import annotations

import argparse
import json

import boto3


def build_contracts(args) -> list[dict]:
    docs = []
    for i in range(args.num_docs):
        start = i * 10_000
        docs.append({
            "collection_id": args.collection_id,
            "asset_id":      args.asset_id,
            "doc_type":      args.doc_type,
            "start_ms":      start,
            "end_ms":        start + 10_000,
            "content": [
                {"kind": "text", "field": "text",
                 "value": f"Test document {i}: a reporter interviews "
                          f"Jane Doe about storm damage on the coast.",
                 "extract_entities": False},
                {"kind": "text", "field": "vector",
                 "value": f"Test document {i}: storm damage interview"},
                {"kind": "text", "field": "named_entities",
                 "value": "Jane Doe"},
            ],
        })
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-bucket", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--collection-id", default="default")
    ap.add_argument("--doc-type", default="scene")
    ap.add_argument("--num-docs", type=int, default=5)
    ap.add_argument("--title", default="Test Asset")
    ap.add_argument("--genres", default="news")
    args = ap.parse_args()

    contracts = build_contracts(args)
    key = f"_indexing/{args.collection_id}/test/{args.asset_id.split(':')[-1]}.jsonl"
    body = "\n".join(json.dumps(c) for c in contracts)
    boto3.client("s3").put_object(Bucket=args.proxy_bucket, Key=key,
                                  Body=body.encode(), ContentType="application/jsonl")
    print(f"Wrote {len(contracts)} contracts to s3://{args.proxy_bucket}/{key}\n")

    sfn_input = {
        "collection_id":    args.collection_id,
        "documents_s3uri": f"s3://{args.proxy_bucket}/{key}",
        "asset_metadata": {
            "title":  args.title,
            "genres": [g.strip() for g in args.genres.split(",") if g.strip()],
        },
        "workflow_config_uri": f"s3://{args.proxy_bucket}/_workflowconfig/hybrid-search.json",
    }
    print("SFN execution input:")
    print(json.dumps(sfn_input, indent=2))


if __name__ == "__main__":
    main()
