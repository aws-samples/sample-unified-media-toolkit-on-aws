#!/bin/bash
# fetch_frames.sh <run_name> <workflow_id> <shot_id>...
P=${PROXY_PREFIX:?set PROXY_PREFIX to s3://<proxy-bucket>/micro-workflow/shot-segmentation}
name=$1; id=$2; shift 2
for sid in "$@"; do
  d=frames/$name/shot_$(printf %04d $sid)
  [ -d "$d" ] && continue
  mkdir -p "$d"
  aws s3 cp "$P/$id/shot_frames/shot_$(printf %04d $sid)_frames.tar.gz" "$d.tar.gz" --region eu-west-1 --quiet \
    && tar xzf "$d.tar.gz" -C "$d" && rm -f "$d.tar.gz" || { rmdir "$d"; echo "FAIL $name $sid"; }
done
