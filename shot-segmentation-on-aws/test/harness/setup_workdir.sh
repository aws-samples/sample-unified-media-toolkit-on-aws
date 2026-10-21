#!/bin/bash
# setup_workdir.sh [batch ...]   — build the scratch layout the replay scripts
# expect, from the manifests checked in under runs/. Run it FROM the scratch
# directory (never from the repo — these artefacts are large and untracked):
#
#   mkdir -p ~/scratch/phase9 && cd ~/scratch/phase9
#   <repo>/test/harness/setup_workdir.sh
#
# Produces:  <batch>/runs.txt  +  <batch>/<run>.{shots.json,recurrence.json,centroids.npy}
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
BATCHES=("$@")
[ ${#BATCHES[@]} -eq 0 ] && BATCHES=(phase9-batch paytv-batch studio-batch)
for b in "${BATCHES[@]}"; do
  m="$HERE/runs/$b.txt"
  [ -f "$m" ] || { echo "no manifest for $b at $m" >&2; exit 1; }
  mkdir -p "$b"
  cp "$m" "$b/runs.txt"
  echo "=== $b"
  ( cd "$b" && "$HERE/collect.sh" runs.txt )
done
