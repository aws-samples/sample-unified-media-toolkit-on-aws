#!/usr/bin/env python3
"""
update_lambdas.py — Update Lambda functions with a new ECR image.

Usage
-----
  # Minimal — repo name only, auto-detects account/region/latest digest
  python update_lambdas.py --image-uri dino-v2-on-aws

  # Override prefix and region
  python update_lambdas.py --image-uri dino-v2-on-aws --prefix my-prefix --region eu-west-1

  # Full URI (bypasses auto-detection)
  python update_lambdas.py --image-uri 123456789.dkr.ecr.eu-west-1.amazonaws.com/dino-v2-on-aws:latest

Auto-detection
--------------
If --image-uri contains no '.' (i.e. it's a bare repo name like "dino-v2-on-aws"):
  1. Calls sts.get_caller_identity() to get the AWS account ID
  2. Uses the region from boto3 session (or --region override)
  3. Calls ecr.describe_images() to find the most recently pushed image digest
  4. Constructs the full URI:
       {account}.dkr.ecr.{region}.amazonaws.com/{repo}@{digest}
"""

from __future__ import annotations

import argparse
import sys
from datetime import timezone

import boto3
from botocore.exceptions import ClientError


# =============================================================================
# Constants
# =============================================================================

DEFAULT_PREFIX = "dino-v2-on-aws"

SUFFIXES = [
    "half-core",
    "1-core",
    "2-cores",
    "3-cores",
    "4-cores",
    "5-cores",
    "6-cores",
]


# =============================================================================
# ECR helpers
# =============================================================================

def _get_account_id(session: boto3.Session) -> str:
    sts = session.client("sts")
    return sts.get_caller_identity()["Account"]


def _get_latest_image_digest(session: boto3.Session, repo_name: str) -> str:
    """
    Return the digest of the most recently pushed image in the ECR repo.
    Uses imagesPushedAt to determine recency — more reliable than 'latest' tag
    which may not be updated on every push.
    """
    ecr = session.client("ecr")
    paginator = ecr.get_paginator("describe_images")

    images = []
    for page in paginator.paginate(repositoryName=repo_name):
        images.extend(page["imageDetails"])

    if not images:
        raise RuntimeError(f"No images found in ECR repository '{repo_name}'")

    # Sort by push time descending — most recent first
    images.sort(
        key=lambda img: img.get("imagePushedAt", 0),
        reverse=True,
    )

    latest = images[0]
    digest = latest["imageDigest"]
    tags = latest.get("imageTags", [])
    pushed = latest.get("imagePushedAt", "unknown")

    print(f"  Latest image: digest={digest}")
    print(f"  Tags:         {tags or '(none)'}")
    print(f"  Pushed at:    {pushed}")

    return digest


def _resolve_image_uri(image_uri: str, region: str | None, session: boto3.Session) -> str:
    """
    If image_uri looks like a full ECR URI (contains '.'), return it as-is.
    Otherwise treat it as a bare repo name and construct the full URI.
    """
    if "." in image_uri:
        # Already a full URI — use as provided
        return image_uri

    # Bare repo name — auto-detect account, region, latest digest
    repo_name = image_uri
    account_id = _get_account_id(session)
    aws_region = region or session.region_name

    if not aws_region:
        raise RuntimeError(
            "Could not determine AWS region. "
            "Set --region or configure AWS_DEFAULT_REGION."
        )

    print(f"Auto-detecting image URI for repo '{repo_name}'...")
    print(f"  Account: {account_id}")
    print(f"  Region:  {aws_region}")

    digest = _get_latest_image_digest(session, repo_name)
    full_uri = (
        f"{account_id}.dkr.ecr.{aws_region}.amazonaws.com"
        f"/{repo_name}@{digest}"
    )
    return full_uri


# =============================================================================
# Lambda update
# =============================================================================

def _update_function(
    lambda_client,
    function_name: str,
    image_uri:     str,
) -> bool:
    """Update a single Lambda function. Returns True on success."""
    try:
        lambda_client.update_function_code(
            FunctionName=function_name,
            ImageUri=image_uri,
        )
        return True
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        msg = exc.response["Error"]["Message"]
        print(f"    AWS error [{code}]: {msg}")
        return False
    except Exception as exc:
        print(f"    Unexpected error: {exc}")
        return False


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update Lambda functions with a new ECR image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--image-uri",
        required=True,
        help=(
            "ECR image URI or bare repo name. "
            "e.g. 'dino-v2-on-aws' or "
            "'123456789.dkr.ecr.eu-west-1.amazonaws.com/dino-v2-on-aws:latest'"
        ),
    )
    parser.add_argument(
        "-p", "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Lambda function name prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "-r", "--region",
        default=None,
        help="AWS region (default: from boto3 session / AWS_DEFAULT_REGION)",
    )
    args = parser.parse_args()

    # ── Build boto3 session ────────────────────────────────────────────────────
    session_kwargs = {}
    if args.region:
        session_kwargs["region_name"] = args.region
    session = boto3.Session(**session_kwargs)

    # ── Resolve full image URI ─────────────────────────────────────────────────
    try:
        image_uri = _resolve_image_uri(args.image_uri, args.region, session)
    except Exception as exc:
        print(f"ERROR: Failed to resolve image URI — {exc}", file=sys.stderr)
        return 1

    aws_region = args.region or session.region_name
    lambda_client = session.client("lambda", region_name=aws_region)

    print()
    print("===========================================")
    print(f"Prefix:    {args.prefix}")
    print(f"Image URI: {image_uri}")
    print(f"Region:    {aws_region}")
    print("===========================================")
    print()

    # ── Update each Lambda ─────────────────────────────────────────────────────
    succeeded: list[str] = []
    failed:    list[str] = []

    for suffix in SUFFIXES:
        function_name = f"{args.prefix}-{suffix}"
        print(f"Updating: {function_name} ...")
        if _update_function(lambda_client, function_name, image_uri):
            print(f"  ✅ Success")
            succeeded.append(function_name)
        else:
            print(f"  ❌ Failed")
            failed.append(function_name)

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("===========================================")
    print("Summary")
    print("===========================================")
    print(f"✅ Succeeded: {len(succeeded)}")
    for fn in succeeded:
        print(f"   - {fn}")

    print(f"❌ Failed:    {len(failed)}")
    for fn in failed:
        print(f"   - {fn}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
