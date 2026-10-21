#!/usr/bin/env python3
"""
bedrock_pricing.py — Fetch Amazon Bedrock foundation model pricing for a given
region, optionally filtered by model family, output as JSON.

Data source: AWS Price List Bulk API (service code AmazonBedrockFoundationModels).
This is the same file format the AWS pricing calculator page itself uses.

KNOWN GAP: this offer code was verified (against a real us-west-2 pull) to have
NO attribute distinguishing model provider — every servicename ends in the
generic "(Amazon Bedrock Edition)" suffix, including third-party models. Family
filtering below uses a curated keyword map instead of substring-matching on
that suffix, to avoid "amazon" matching every model in the file.

It was also verified that this offer code does NOT contain Amazon's own
Titan/Nova models in at least one region. If --family amazon returns zero
rows, that's not necessarily a bug in this script — Amazon's own models may
be published under a different offer code (AmazonBedrock or
AmazonBedrockService). Use --service-code to try one of those if you hit this.

Adopted from the V5 reference implementation (2026-10-13); reviewed for
sensitive content (none — public Pricing API only) and the requests
dependency replaced with stdlib urllib so it runs wherever boto3 does.
NOT named test_*.py on purpose: unittest discover must not pick it up
(live network calls). Use it to verify models/availablemodels_master.json
pricing before a deploy seeds the live registry.

Requires:
    pip install boto3

IAM permissions needed on whatever credentials you run this with:
    pricing:ListPriceLists
    pricing:GetPriceListFileUrl

Note: the AWS Pricing API control plane only exists in us-east-1 and
ap-south-1 — that's independent of which Bedrock region you're pricing.
This script always calls the Pricing API in us-east-1 regardless of
--region (--region is the Bedrock region whose prices you want, not where
the API call goes).

Usage:
    python bedrock_pricing.py --region us-west-2
    python bedrock_pricing.py --region us-west-2 --family anthropic
    python bedrock_pricing.py --region eu-central-1 --family anthropic meta --output prices.json
    python bedrock_pricing.py --region us-east-1 --family amazon --service-code AmazonBedrockService
"""

import argparse
import gzip
import json
import re
import sys
from datetime import datetime, timezone

import urllib.request

import boto3

DEFAULT_SERVICE_CODE = "AmazonBedrockFoundationModels"
PRICING_API_REGION = "us-east-1"  # control-plane region for the Pricing service itself

# Curated family -> keyword map. Matched (case-insensitive, substring) against
# the servicename with the universal "(Amazon Bedrock Edition)" suffix
# stripped off first. If a family name isn't in this map, it's used directly
# as a raw keyword against the (suffix-stripped) servicename as a fallback,
# so new/unlisted providers still work, just less precisely.
FAMILY_KEYWORDS = {
    "anthropic": ["claude"],
    "meta": ["llama"],
    "cohere": ["cohere"],
    "ai21": ["jurassic", "jamba"],
    "amazon": ["titan", "nova"],
    "stability": ["stable diffusion", "stable image"],
    "writer": ["palmyra"],
    "luma": ["luma"],
    "twelvelabs": ["twelvelabs"],
    "mistral": ["mistral", "pixtral"],
    "deepseek": ["deepseek"],
    "qwen": ["qwen"],
}

_SUFFIX_RE = re.compile(r"\s*\(Amazon Bedrock Edition\)\s*$", re.I)


def strip_suffix(servicename: str) -> str:
    return _SUFFIX_RE.sub("", servicename or "").strip()


def matches_family(servicename: str, families: list[str]) -> bool:
    name = strip_suffix(servicename).lower()
    for fam in families:
        keywords = FAMILY_KEYWORDS.get(fam.lower(), [fam.lower()])
        if any(kw in name for kw in keywords):
            return True
    return False


def get_price_list_url(region_code: str, service_code: str, currency: str) -> str:
    client = boto3.client("pricing", region_name=PRICING_API_REGION)
    resp = client.list_price_lists(
        ServiceCode=service_code,
        RegionCode=region_code,
        CurrencyCode=currency,
        EffectiveDate=datetime.now(timezone.utc),
    )
    price_lists = resp.get("PriceLists", [])
    if not price_lists:
        raise RuntimeError(
            f"No price list found for service={service_code}, region={region_code}, "
            f"currency={currency}. Check the region code is valid, and that this "
            f"service code publishes pricing for it."
        )
    arn = price_lists[0]["PriceListArn"]
    url_resp = client.get_price_list_file_url(PriceListArn=arn, FileFormat="json")
    return url_resp["Url"]


def download_price_list(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as resp:  # nosec B310 — https URL from the Pricing API
        raw = resp.read()
    # AWS sometimes serves these gzip-compressed regardless of URL/extension.
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def extract_price(sku: str, terms: dict):
    term = terms.get("OnDemand", {}).get(sku)
    if not term:
        return None, None, None
    for term_detail in term.values():
        for dim in term_detail.get("priceDimensions", {}).values():
            price = dim.get("pricePerUnit", {}).get("USD")
            unit = dim.get("unit")
            desc = dim.get("description")
            return price, unit, desc
    return None, None, None


def build_pricing_rows(data: dict, families: list[str] | None) -> list[dict]:
    products = data.get("products", {})
    terms = data.get("terms", {})

    rows = []
    for sku, product in products.items():
        attrs = product.get("attributes", {})
        servicename = attrs.get("servicename", "")

        if families and not matches_family(servicename, families):
            continue

        price, unit, desc = extract_price(sku, terms)
        if price is None:
            continue

        rows.append({
            "sku": sku,
            "model": servicename,
            "usagetype": attrs.get("usagetype"),
            "region": attrs.get("regionCode"),
            "location": attrs.get("location"),
            "price_usd": price,
            "unit": unit,
            "description": desc,
        })

    rows.sort(key=lambda r: (r["model"] or "", r["usagetype"] or ""))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Amazon Bedrock foundation model pricing for a region as JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--region", required=True, help="Bedrock region code, e.g. us-west-2")
    parser.add_argument(
        "--family", nargs="*", default=None,
        help="Model family filter, e.g. --family anthropic meta. "
             "Known: " + ", ".join(sorted(FAMILY_KEYWORDS)) + ". "
             "Unknown names are used as a raw keyword filter. Omit for all models.",
    )
    parser.add_argument("--currency", default="USD", help="Currency code (default: USD)")
    parser.add_argument(
        "--service-code", default=DEFAULT_SERVICE_CODE,
        help=f"AWS Price List service code (default: {DEFAULT_SERVICE_CODE}). "
             "Try AmazonBedrockService if you're looking for Amazon's own Titan/Nova "
             "models and they're missing from the default.",
    )
    parser.add_argument("--output", default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    try:
        url = get_price_list_url(args.region, args.service_code, args.currency)
        raw_data = download_price_list(url)
        rows = build_pricing_rows(raw_data, args.family)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.family and not rows:
        print(
            f"Warning: 0 results for family={args.family} in service-code "
            f"{args.service_code}. That family may be published under a different "
            f"service code (see --help).",
            file=sys.stderr,
        )

    output = {
        "region": args.region,
        "service_code": args.service_code,
        "currency": args.currency,
        "family_filter": args.family,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "prices": rows,
    }

    text = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Wrote {len(rows)} price entries to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()