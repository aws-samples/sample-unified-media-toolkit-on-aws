#!/usr/bin/env bash
# package_catalog.sh — export the live default voice catalog (DynamoDB
# records + S3 conditioning/preview blobs) into one voice_catalog.tar.gz,
# for hosting on CloudFront and importing into a customer's stack at
# CodeBuild time (same seed/host/import shape as package_models.sh's
# model artefacts, applied to catalog data instead of model weights).
#
# Real catalog records are batch-write-item'd wholesale, not
# re-registered through register_voice at deploy time — registration
# (ASR transcript-match validation + speech_tokenizer_v3/CAMPPlus
# extraction) is deterministic given the same reference audio and was
# already validated once when these 46 voices were first registered;
# re-running it on every deploy would just re-pay that cost for the
# same result.
#
# voice_id is fixed at export time (baked into the manifest, not
# regenerated) — re-importing the same bundle on a stack update must
# write the SAME DynamoDB items every time, not create duplicates.
#
# Output layout (unpacked voice_catalog.tar.gz) — flattened to
# {catalog}/{voice_id}/, NOT the live stack's own
# micro-workflow/tts-on-aws/voices/{voice_id}/ path (that path is an
# artifact of THIS proxy bucket's layout, not something a customer's
# bucket needs to reproduce):
#   {catalog}/manifest.json              — [{ddb_item...}, ...] one per
#                                           voice, in DynamoDB's Item
#                                           wire format (S/N/BOOL-typed),
#                                           ready for BatchWriteItem
#                                           with no reshaping.
#                                           s3_conditioning_key/
#                                           s3_preview_key are REWRITTEN
#                                           to the destination key
#                                           (_model_artefacts/voice_catalog/
#                                           {catalog}/{voice_id}/...) —
#                                           NOT the source key on this
#                                           export's proxy bucket, which
#                                           a customer's bucket has no
#                                           reason to match.
#   {catalog}/{voice_id}/conditioning.npz
#   {catalog}/{voice_id}/preview.wav
# {catalog} is each record's own `catalog` field (currently always
# "default" for the baked-in set) — not a hardcoded constant, so a
# future non-default catalog exports to its own top-level directory
# automatically.
#
# Import side (CodeBuild, see cfnstack.yaml): extract the tar, copy
# {catalog}/{voice_id}/ trees to
# s3://{ProxyBucket}/_model_artefacts/voice_catalog/{catalog}/{voice_id}/
# (a direct, unmodified recursive copy — the manifest's rewritten keys
# already point exactly there), then BatchWriteItem each catalog/
# manifest.json into VoiceTable.
#
# Usage: ./package_catalog.sh <proxy-bucket> <voice-table-name> [cloudfront-upload-bucket]
#   proxy-bucket        the live stack's ProxyBucket (source of the real
#                        conditioning/preview S3 objects to export)
#   voice-table-name     the live stack's VoiceTable (source of the real
#                        DynamoDB records to export)
#   cloudfront-upload-bucket  [optional] if given, also uploads the
#                        resulting tar.gz to
#                        s3://<bucket>/opensource_models/voice_catalog/voice_catalog.tar.gz
#                        (the house CloudFront origin bucket) — omit to
#                        just produce the local tar.gz for manual review
#                        first.
set -euo pipefail

PROXY_BUCKET="${1:?usage: $0 <proxy-bucket> <voice-table-name> [cloudfront-upload-bucket]}"
VOICE_TABLE="${2:?usage: $0 <proxy-bucket> <voice-table-name> [cloudfront-upload-bucket]}"
CF_UPLOAD_BUCKET="${3:-}"

# Must match cfnstack.yaml's Models.S3.Prefix-equivalent for the catalog
# import — the destination prefix under the CUSTOMER's ProxyBucket, not
# this export's source bucket.
DEST_PREFIX="_model_artefacts/voice_catalog"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "== Exporting DynamoDB records from $VOICE_TABLE =="
# Only the "default" catalog's baked-in voices — never export a
# customer's own user-uploaded voices (origin != FLEURS/LibriSpeech)
# into a bundle that gets shipped to every OTHER customer's stack.
aws dynamodb scan --table-name "$VOICE_TABLE" \
  --filter-expression "#c = :c" \
  --expression-attribute-names '{"#c":"catalog"}' \
  --expression-attribute-values '{":c":{"S":"default"}}' \
  --output json > "$STAGE/scan_raw.json"

echo "== Rewriting manifest, downloading conditioning + preview blobs =="
python3 - "$STAGE/scan_raw.json" "$STAGE" "$DEST_PREFIX" "$PROXY_BUCKET" << 'PYEOF'
import glob
import json
import os
import sys

import boto3

raw_path, stage, dest_prefix, bucket = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
raw = json.load(open(raw_path))
items = raw["Items"]

# only ship real baked-in provenance, never a user-upload that slipped
# into the "default" catalog by mistake
known_origins = {"FLEURS", "LibriSpeech"}
kept = [it for it in items if it.get("origin", {}).get("S") in known_origins]
skipped = len(items) - len(kept)

s3 = boto3.client("s3")
by_catalog = {}
for it in kept:
    catalog = it["catalog"]["S"]
    voice_id = it["voice_id"]["S"]
    voice_dir = os.path.join(stage, catalog, voice_id)
    os.makedirs(voice_dir, exist_ok=True)

    # download from the ORIGINAL source key BEFORE rewriting it
    source_cond_key = it["s3_conditioning_key"]["S"]
    s3.download_file(bucket, source_cond_key, os.path.join(voice_dir, "conditioning.npz"))
    source_preview_key = it.get("s3_preview_key", {}).get("S")
    if source_preview_key:
        s3.download_file(bucket, source_preview_key, os.path.join(voice_dir, "preview.wav"))

    # then rewrite to the DESTINATION key a customer's bucket will
    # actually have after import — this export's own source key (this
    # proxy bucket's micro-workflow/... layout) is not meaningful to
    # any other bucket.
    it["s3_conditioning_key"] = {"S": f"{dest_prefix}/{catalog}/{voice_id}/conditioning.npz"}
    if source_preview_key:
        it["s3_preview_key"] = {"S": f"{dest_prefix}/{catalog}/{voice_id}/preview.wav"}

    by_catalog.setdefault(catalog, []).append(it)
    print(f"  downloaded {voice_id} ({it['name']['S']})")

for catalog, catalog_items in by_catalog.items():
    catalog_dir = os.path.join(stage, catalog)
    # ensure_ascii=False — ref_transcript is real non-English text
    # (Japanese/Chinese/etc.) for most of this catalog; \uXXXX escapes
    # are technically valid JSON but unreadable to a human reviewing
    # the manifest, and there is no reason to force ASCII here.
    json.dump(catalog_items, open(f"{catalog_dir}/manifest.json", "w"),
              indent=2, ensure_ascii=False)

print(f"{len(kept)} voices exported across {len(by_catalog)} catalog(s) "
      f"({skipped} skipped: not a known baked-in origin)")
for catalog, catalog_items in by_catalog.items():
    print(f"  {catalog}: {len(catalog_items)} voices")
PYEOF

rm -f "$STAGE/scan_raw.json"

echo "== Packing voice_catalog.tar.gz =="
OUT="$(pwd)/voice_catalog.tar.gz"
# BSD find (macOS) has no -printf; ls suffices since each catalog dir is
# a single path component with no special characters (voice_id/catalog
# strings from our own DynamoDB records, not arbitrary user input).
tar -czf "$OUT" -C "$STAGE" $(cd "$STAGE" && ls)
echo "wrote: $OUT ($(du -h "$OUT" | cut -f1))"

if [[ -n "$CF_UPLOAD_BUCKET" ]]; then
  DEST="s3://$CF_UPLOAD_BUCKET/opensource_models/voice_catalog/voice_catalog.tar.gz"
  echo "== Uploading to $DEST =="
  aws s3 cp "$OUT" "$DEST"
  echo "uploaded: $DEST"
fi
