#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Create or update the conversation-analysis stack.
#
# bash ./create_update_stack.sh \
#     --stack-name my-conversation-analysis \
#     --resource-prefix myprefix \
#     --proxy-bucket MY_PROXY_BUCKET \
#     --bedrock-lambda-arn <bedrock-inference GatewayToolLambdaArn> \
#     --registry-lambda-arn <prompt-registry GatewayToolLambdaArn> \
#     [--template-url URL] [--region eu-west-1] [--profile my-profile]
#
# --template-url is optional: it is derived from the BUILT template under
# dist/global-s3-assets/ (reading the built copy rather than cfnstack.yaml so
# an un-built template fails loudly on the %%BUCKET_NAME%% placeholder
# instead of resolving to a wrong bucket — media-toolkit-gateway' pattern).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_NAME="conversation-analysis-stack.template"

while [[ $# -gt 0 ]]; do
  case $1 in
    --stack-name)      STACK_NAME="$2"; shift; shift ;;
    --resource-prefix) RESOURCE_PREFIX="$2"; shift; shift ;;
    --proxy-bucket)    PROXY_BUCKET="$2"; shift; shift ;;
    --bedrock-lambda-arn)  BEDROCK_ARN="$2"; shift; shift ;;
    --registry-lambda-arn) REGISTRY_ARN="$2"; shift; shift ;;
    --template-url)    TEMPLATE_URL="$2"; shift; shift ;;
    --region)          REGION="$2"; shift; shift ;;
    --profile)         PROFILE="$2"; shift; shift ;;
    *) shift ;;
  esac
done

[ -z "$STACK_NAME" ] && echo "error: --stack-name is required" && exit 1
[ -z "$RESOURCE_PREFIX" ] && echo "error: --resource-prefix is required" && exit 1
[ -z "$PROXY_BUCKET" ] && echo "error: --proxy-bucket is required" && exit 1
[ -z "$BEDROCK_ARN" ] && echo "error: --bedrock-lambda-arn is required" && exit 1
[ -z "$REGISTRY_ARN" ] && echo "error: --registry-lambda-arn is required" && exit 1

AWS_ARGS=()
[ -n "$REGION" ] && AWS_ARGS+=(--region "$REGION")
[ -n "$PROFILE" ] && AWS_ARGS+=(--profile "$PROFILE")

if [ -z "$TEMPLATE_URL" ]; then
  BUILT="$SCRIPT_DIR/dist/global-s3-assets/$TEMPLATE_NAME"
  [ ! -f "$BUILT" ] && \
    echo "error: no built template at $BUILT — run build-deploy-s3-dist.sh first" && exit 1
  if grep -q '%%BUCKET_NAME%%' "$BUILT"; then
    echo "error: built template still carries %%BUCKET_NAME%% — it was built with --build-only and never deployed"
    exit 1
  fi
  BUCKET=$(grep -A2 'Source:' "$BUILT" | grep 'Bucket:' | head -1 | awk '{print $2}' | tr -d '"')
  KEY_PREFIX=$(grep -A3 'Source:' "$BUILT" | grep 'KeyPrefix:' | head -1 | awk '{print $2}' | tr -d '"')
  LOCATION=$(aws s3api get-bucket-location --bucket "$BUCKET" "${AWS_ARGS[@]}" 2>/dev/null | jq -r '.LocationConstraint // "us-east-1"')
  [ "$LOCATION" == "null" ] && LOCATION="us-east-1"
  DOMAIN="s3.amazonaws.com"
  [ "$LOCATION" != "us-east-1" ] && DOMAIN="s3.${LOCATION}.amazonaws.com"
  TEMPLATE_URL="https://${BUCKET}.${DOMAIN}/${KEY_PREFIX}/${TEMPLATE_NAME}"
fi

echo "Template URL: $TEMPLATE_URL"

PARAMS=(
  ParameterKey=ResourcePrefix,ParameterValue="$RESOURCE_PREFIX"
  ParameterKey=ProxyBucket,ParameterValue="$PROXY_BUCKET"
  ParameterKey=BedrockInferenceGatewayLambdaArn,ParameterValue="$BEDROCK_ARN"
  ParameterKey=PromptRegistryGatewayLambdaArn,ParameterValue="$REGISTRY_ARN"
)

if aws cloudformation describe-stacks --stack-name "$STACK_NAME" "${AWS_ARGS[@]}" > /dev/null 2>&1; then
  ACTION=update-stack
else
  ACTION=create-stack
fi
echo "Running $ACTION on $STACK_NAME ..."
aws cloudformation $ACTION \
  --stack-name "$STACK_NAME" \
  --template-url "$TEMPLATE_URL" \
  --parameters "${PARAMS[@]}" \
  --capabilities CAPABILITY_IAM \
  "${AWS_ARGS[@]}" \
  --query StackId --output text
