#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Create or update the media-processing CloudFormation stack.
# Run from media-processing-on-aws/ or any directory — template path is relative
# to the script location.
#
# Usage:
#   bash ./create_update_stack.sh \
#       --stack-name my-media-processing \
#       --resource-prefix myprefix \
#       --ingest-bucket my-ingest-bucket \
#       --proxy-bucket my-proxy-bucket

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use the processed template from the build output — cfnstack.yaml still contains
# raw %%TOKEN%% placeholders that are substituted by build-deploy-s3-dist.sh.
TEMPLATE="$SCRIPT_DIR/dist/global-s3-assets/media-processing-stack.template"

######################################################################
# Usage
######################################################################
function usage() {
  echo "
------------------------------------------------------------------------------
Create or update the media-processing CloudFormation stack.

bash ./create_update_stack.sh \\
    --stack-name STACK_NAME \\
    --resource-prefix RESOURCE_PREFIX \\
    --ingest-bucket INGEST_BUCKET \\
    --proxy-bucket PROXY_BUCKET \\
    [--region REGION] \\
    [--profile PROFILE]

Required:
  --stack-name      CloudFormation stack name
  --resource-prefix Prefix applied to all resource names
  --ingest-bucket   S3 bucket containing source media files
  --proxy-bucket    S3 bucket for outputs and intermediates

Optional:
  --region    AWS region (defaults to AWS CLI default)
  --profile   AWS CLI profile
------------------------------------------------------------------------------
"
  return 0
}

######################################################################
# Parse arguments
######################################################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack-name)      STACK_NAME="$2";      shift 2 ;;
    --resource-prefix) RESOURCE_PREFIX="$2"; shift 2 ;;
    --ingest-bucket)   INGEST_BUCKET="$2";   shift 2 ;;
    --proxy-bucket)    PROXY_BUCKET="$2";    shift 2 ;;
    --region)          REGION="$2";          shift 2 ;;
    --profile)         PROFILE="$2";         shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

######################################################################
# Validate
######################################################################
ERRORS=0
[ -z "$STACK_NAME" ]      && echo "error: --stack-name is required"      && ERRORS=$((ERRORS+1))
[ -z "$RESOURCE_PREFIX" ] && echo "error: --resource-prefix is required" && ERRORS=$((ERRORS+1))
[ -z "$INGEST_BUCKET" ]   && echo "error: --ingest-bucket is required"   && ERRORS=$((ERRORS+1))
[ -z "$PROXY_BUCKET" ]    && echo "error: --proxy-bucket is required"    && ERRORS=$((ERRORS+1))
[ ! -f "$TEMPLATE" ]      && echo "error: template not found: $TEMPLATE — run build-deploy-s3-dist.sh first" && ERRORS=$((ERRORS+1))
[ "$ERRORS" -gt 0 ]       && usage && exit 1

######################################################################
# Build AWS CLI flags
######################################################################
AWS_FLAGS=()
[ -n "$REGION" ]  && AWS_FLAGS+=(--region "$REGION")
[ -n "$PROFILE" ] && AWS_FLAGS+=(--profile "$PROFILE")

######################################################################
# Resolve template S3 URL from embedded Mappings
# (template exceeds the 51200-byte --template-body limit)
######################################################################
T_BUCKET=$(grep -m1 'Bucket:' "$TEMPLATE" | sed 's/.*"\(.*\)".*/\1/')
T_KEY_PREFIX=$(grep -m1 'KeyPrefix:' "$TEMPLATE" | sed 's/.*"\(.*\)".*/\1/')
T_NAME=$(basename "$TEMPLATE")

if [ -z "$T_BUCKET" ] || [ -z "$T_KEY_PREFIX" ]; then
  echo "error: could not parse Bucket/KeyPrefix from ${TEMPLATE}"
  echo "       Re-run build-deploy-s3-dist.sh to regenerate the processed template."
  exit 1
fi

DEPLOY_REGION="${REGION:-$(aws configure get region 2>/dev/null)}"
DEPLOY_REGION="${DEPLOY_REGION:-us-east-1}"
TEMPLATE_URL="https://${T_BUCKET}.s3.${DEPLOY_REGION}.amazonaws.com/${T_KEY_PREFIX}/${T_NAME}"
echo "Template URL: ${TEMPLATE_URL}"
echo ""

PARAMETERS=(
  "ParameterKey=ResourcePrefix,ParameterValue=${RESOURCE_PREFIX}"
  "ParameterKey=IngestBucket,ParameterValue=${INGEST_BUCKET}"
  "ParameterKey=ProxyBucket,ParameterValue=${PROXY_BUCKET}"
)

CAPABILITIES="CAPABILITY_IAM CAPABILITY_NAMED_IAM"

######################################################################
# Detect: create or update?
######################################################################
echo "Checking stack '${STACK_NAME}'..."

STACK_STATUS=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  "${AWS_FLAGS[@]}" \
  --query "Stacks[0].StackStatus" \
  --output text 2>/dev/null)

if [ -z "$STACK_STATUS" ] || [ "$STACK_STATUS" = "None" ]; then
  ACTION="create"
elif [ "$STACK_STATUS" = "ROLLBACK_COMPLETE" ]; then
  echo "Stack is in ROLLBACK_COMPLETE — delete it first before re-creating."
  exit 1
else
  ACTION="update"
fi

echo "Action: ${ACTION}-stack"
echo "Stack:  ${STACK_NAME}"
echo "Region: ${REGION:-$(aws configure get region "${AWS_FLAGS[@]}" 2>/dev/null || echo 'default')}"
echo ""

######################################################################
# Create or update
######################################################################
if [ "$ACTION" = "create" ]; then
  aws cloudformation create-stack \
    --stack-name "$STACK_NAME" \
    --template-url "${TEMPLATE_URL}" \
    --parameters "${PARAMETERS[@]}" \
    --capabilities $CAPABILITIES \
    "${AWS_FLAGS[@]}"

  if [ $? -ne 0 ]; then
    echo "error: create-stack failed"
    exit 1
  fi

  echo "Waiting for stack creation to complete..."
  aws cloudformation wait stack-create-complete \
    --stack-name "$STACK_NAME" \
    "${AWS_FLAGS[@]}"
  WAIT_EXIT=$?

else
  # Attempt update; treat "No updates are to be performed" as success.
  UPDATE_OUTPUT=$(aws cloudformation update-stack \
    --stack-name "$STACK_NAME" \
    --template-url "${TEMPLATE_URL}" \
    --parameters "${PARAMETERS[@]}" \
    --capabilities $CAPABILITIES \
    "${AWS_FLAGS[@]}" 2>&1)
  UPDATE_EXIT=$?

  if [ $UPDATE_EXIT -ne 0 ]; then
    if echo "$UPDATE_OUTPUT" | grep -q "No updates are to be performed"; then
      echo "Stack is already up to date — no changes."
      exit 0
    fi
    echo "error: update-stack failed"
    echo "$UPDATE_OUTPUT"
    exit 1
  fi

  echo "Waiting for stack update to complete..."
  aws cloudformation wait stack-update-complete \
    --stack-name "$STACK_NAME" \
    "${AWS_FLAGS[@]}"
  WAIT_EXIT=$?
fi

######################################################################
# Result
######################################################################
if [ "$WAIT_EXIT" -ne 0 ]; then
  echo ""
  echo "Stack ${ACTION} FAILED. Last events:"
  aws cloudformation describe-stack-events \
    --stack-name "$STACK_NAME" \
    "${AWS_FLAGS[@]}" \
    --query "StackEvents[?ResourceStatus=='CREATE_FAILED'||ResourceStatus=='UPDATE_FAILED'||ResourceStatus=='ROLLBACK_IN_PROGRESS'].[LogicalResourceId,ResourceStatus,ResourceStatusReason]" \
    --output table
  exit 1
fi

echo ""
echo "======================================================================"
echo " Stack ${ACTION} COMPLETE: ${STACK_NAME}"
echo "======================================================================"

# Print the key outputs
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  "${AWS_FLAGS[@]}" \
  --query "Stacks[0].Outputs[*].[OutputKey,OutputValue]" \
  --output table
