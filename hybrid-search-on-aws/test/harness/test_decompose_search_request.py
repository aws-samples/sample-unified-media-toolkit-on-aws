"""
test_decompose_search_request.py — invoke StateDecomposeSearchRequest locally.

Calls the Lambda handler in-process (no Lambda invoke needed) after setting
the required environment variables.  All real AWS calls are made:
  - S3  : reads workflow config + prompt, writes vectors to S3
  - DDB : reads table state, writes search history
  - Bedrock : calls SLM + embed

Prerequisites:
  - AWS credentials in environment / ~/.aws/credentials
  - Pass table/bucket names via CLI flags or env-file

Usage:
    cd hybrid-search-on-aws
    python test/harness/test_decompose_search_request.py \\
        --proxy-bucket <proxy-bucket> \\
        --index-state-table <prefix>-lancedb-index-state \\
        --search-history-table <prefix>-lancedb-search-history \\
        --dedup-table <prefix>-lancedb-dedup-state \\
        [--search-term "Jane Doe on a stormy beach"] \\
        [--k 10] \\
        [--workflow-config-uri s3://<bucket>/_workflowconfig/hybrid-search.json] \\
        [--request-id <id>] \\
        [--dry-run]

With --dry-run the event is printed but the handler is not called.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid as _uuid_mod

# ── ensure the package root is on sys.path so lib.* imports work ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_HERE))   # repo root
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

_DEFAULT_SEARCH_TERM  = "Jane Doe on a stormy beach"
_DEFAULT_K            = 10
_DEFAULT_IMAGE_URI    = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proxy-bucket",
                        default=os.environ.get("PROXY_BUCKET", ""),
                        help="S3 proxy bucket name (no s3:// prefix)")
    parser.add_argument("--index-state-table",
                        default=os.environ.get("INDEX_STATE_TABLE", "LanceDBIndexState"),
                        help="DynamoDB index-state table name")
    parser.add_argument("--search-history-table",
                        default=os.environ.get("SEARCH_HISTORY_TABLE", "LanceDBSearchHistory"),
                        help="DynamoDB search-history table name")
    parser.add_argument("--dedup-table",
                        default=os.environ.get("DEDUP_TABLE", "LanceDBDedupState"),
                        help="DynamoDB dedup table name")
    parser.add_argument("--search-history-ttl-days",
                        default=os.environ.get("SEARCH_HISTORY_TTL_DAYS", "7"),
                        help="Search history TTL in days")
    parser.add_argument("--workflow-config-uri",
                        help="Full s3:// URI to workflow config (default: s3://<proxy-bucket>/_workflowconfig/hybrid-search.json)")
    parser.add_argument("--search-term",       default=_DEFAULT_SEARCH_TERM,
                        help="Text search query (mutually exclusive with --search-image-s3uri)")
    parser.add_argument("--search-image-s3uri",  default=_DEFAULT_IMAGE_URI,
                        help="S3 URI of image to search by (e.g. s3://bucket/key.jpg). "
                             "When set, --search-term is ignored and imageRetrievalPaths are used.")
    parser.add_argument("--k",            type=int, default=_DEFAULT_K)
    parser.add_argument("--embed-model",  default="",
                        help="Force all embed() calls to use this model ID regardless of schema "
                             "(e.g. amazon.titan-embed-image-v1, amazon.nova-2-multimodal-embeddings-v1:0)")
    parser.add_argument("--embed-dims",   type=int, default=None,
                        help="Force embedding output dimension (required for Titan/Nova when using --embed-model)")
    parser.add_argument("--request-id",   default="",
                        help="SFN execution name used as request_id (default: auto-generated)")
    parser.add_argument("--region",       default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print the event and exit without calling the handler")
    args = parser.parse_args()

    if not args.proxy_bucket:
        parser.error("--proxy-bucket is required (or set PROXY_BUCKET env var)")

    # ── Set env vars before importing handler (constants are read at module load) ─
    os.environ["PROXY_BUCKET"]              = args.proxy_bucket
    os.environ["INDEX_STATE_TABLE"]         = args.index_state_table
    os.environ["SEARCH_HISTORY_TABLE"]      = args.search_history_table
    os.environ["DEDUP_TABLE"]               = args.dedup_table
    os.environ["SEARCH_HISTORY_TTL_DAYS"]   = args.search_history_ttl_days
    os.environ["AWS_REGION"]                = args.region

    workflow_config_uri = (
        args.workflow_config_uri
        or f"s3://{args.proxy_bucket}/_workflowconfig/hybrid-search.json"
    )
    request_id = args.request_id or f"test-{_uuid_mod.uuid4()}"

    # image search takes priority over text search
    if args.search_image_s3uri:
        event = {
            "workflow_config_uri": workflow_config_uri,
            "search_term":         "",
            "search_image_s3uri":    args.search_image_s3uri,
            "k":                   args.k,
            "request_id":          request_id,
        }
    else:
        event = {
            "workflow_config_uri": workflow_config_uri,
            "search_term":         args.search_term,
            "search_image_s3uri":    "",
            "k":                   args.k,
            "request_id":          request_id,
        }

    print("=" * 60)
    print("Event:")
    print(json.dumps(event, indent=2))
    print("=" * 60)

    if args.dry_run:
        print("--dry-run: skipping handler invocation.")
        return

    # Import after env vars are set so constants in input_schema.py pick them up
    from states.decompose_search_request.lambda_function import lambda_handler  # noqa: E402
    import lib.bedrock_utils as _bedrock_utils  # noqa: E402

    if args.embed_model:
        _forced_model = args.embed_model
        _forced_dims  = args.embed_dims
        _orig_embed       = _bedrock_utils.embed
        _orig_embed_image = _bedrock_utils.embed_image
        _bedrock_utils.embed = lambda model_id, text, dims=None: _orig_embed(
            _forced_model, text, _forced_dims if _forced_dims is not None else dims
        )
        _bedrock_utils.embed_image = lambda model_id, image_b64, dims=None: _orig_embed_image(
            _forced_model, image_b64, _forced_dims if _forced_dims is not None else dims
        )
        print(f"embed() / embed_image() forced → model={_forced_model!r} dims={_forced_dims}\n")

    print("Invoking StateDecomposeSearchRequest …\n")
    try:
        result = lambda_handler(event, context=None)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))

    # ── Summary ────────────────────────────────────────────────────────────────
    queries = result.get("queries", [])
    print(f"\n{'='*60}")
    print(f"request_id       : {result.get('request_id')}")
    if result.get('search_image_s3uri'):
        print(f"search_image_s3uri : {result.get('search_image_s3uri')}")
    else:
        print(f"clean_query      : {result.get('clean_query')}")
    print(f"retrieval_paths  : {result.get('retrieval_paths')}")
    print(f"entities         : {result.get('entities')}")
    print(f"model_used       : {result.get('language_model_used')}")
    print(f"fallback         : {result.get('fallback_triggered')} ({result.get('fallback_reason')})")
    print(f"vectors_s3_key   : {result.get('vectors_s3_key')}")
    print(f"query items      : {len(queries)}")
    for q in queries:
        sq_summary = [(s["type"], s["field"], s.get("weight")) for s in q.get("sub_queries", [])]
        print(f"  [{q['index_scope']}] index_key={q['index_key']!r}  sub_queries={sq_summary}")
    print("=" * 60)


if __name__ == "__main__":
    main()
