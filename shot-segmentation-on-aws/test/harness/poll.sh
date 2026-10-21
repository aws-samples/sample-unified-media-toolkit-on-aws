#!/bin/bash
# poll.sh <manifest>   — poll every run in a manifest until all settle,
# reporting the current state machine step for those still RUNNING.
set -u
MANIFEST=${1:?usage: poll.sh <manifest>}
SM=${SM_ARN:?set SM_ARN to the shot-segmentation execution ARN prefix}
REGION=${AWS_REGION:-eu-west-1}
while :; do
  done_all=1; out=""
  while read -r name id; do
    [ -z "${name:-}" ] && continue
    st=$(aws stepfunctions describe-execution --execution-arn "$SM:$id" \
         --region "$REGION" --query 'status' --output text 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      done_all=0
      cs=$(aws stepfunctions get-execution-history --execution-arn "$SM:$id" \
          --region "$REGION" --reverse-order --max-items 12 \
          --query "events[?type=='TaskStateEntered'||type=='MapStateEntered'||type=='PassStateEntered'].stateEnteredEventDetails.name | [0]" \
          --output text 2>/dev/null)
      out+=$(printf '%-18s %-10s %s\n' "$name" "$st" "$cs")$'\n'
    else
      out+=$(printf '%-18s %-10s\n' "$name" "$st")$'\n'
    fi
  done < "$MANIFEST"
  printf '=== %s\n%s' "$(date -u +%H:%M:%S)" "$out"
  [ $done_all -eq 1 ] && { echo "ALL SETTLED"; break; }
  sleep 120
done
