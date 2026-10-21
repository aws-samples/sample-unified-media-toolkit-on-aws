#!/bin/bash
# collect.sh <manifest>   — pull the three artefacts every replay needs into
# the current directory, named <run_name>.<artefact>.
set -u
MANIFEST=${1:?usage: collect.sh <manifest>}
P=${PROXY_PREFIX:?set PROXY_PREFIX to s3://<proxy-bucket>/micro-workflow/shot-segmentation}
REGION=${AWS_REGION:-eu-west-1}
while read -r name id; do
  [ -z "${name:-}" ] && continue
  aws s3 cp "$P/$id/embeddings/recurrence.json"    "$name.recurrence.json" --region "$REGION" --quiet
  aws s3 cp "$P/$id/embeddings/shots.json"         "$name.shots.json"      --region "$REGION" --quiet
  aws s3 cp "$P/$id/embeddings/shot_centroids.npy" "$name.centroids.npy"   --region "$REGION" --quiet
  printf '%-18s %s bytes\n' "$name" "$(wc -c < "$name.recurrence.json" 2>/dev/null || echo MISSING)"
done < "$MANIFEST"
