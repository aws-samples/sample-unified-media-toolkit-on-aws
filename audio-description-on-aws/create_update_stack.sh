#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Create or update the audio-description stack.
#
# bash ./create_update_stack.sh \
#     --stack-name my-audio-description \
#     --resource-prefix myprefix \
#     --proxy-bucket MY_PROXY_BUCKET \
#     --bedrock-gateway-arn ARN \
#     --prompt-registry-arn ARN \
#     --tts-gateway-arn ARN \
#     [--template-url URL] [--region eu-west-1] [--profile my-profile]
#
# A CONSUMER child: the three sibling gateway-Lambda ARNs (each sibling
# stack's GatewayToolLambdaArn output) are REQUIRED template parameters.
#
# --template-url is optional: it is derived from the BUILT template under
# dist/global-s3-assets/ (reading the built copy rather than cfnstack.yaml so
# an un-built template fails loudly on the %%BUCKET_NAME%% placeholder
# instead of resolving to a wrong bucket — media-toolkit-gateway' pattern).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_NAME="audio-description-stack.template"

while [[ $# -gt 0 ]]; do
  case $1 in
    --stack-name)      STACK_NAME="$2"; shift; shift ;;
    --resource-prefix) RESOURCE_PREFIX="$2"; shift; shift ;;
    --proxy-bucket)    PROXY_BUCKET="$2"; shift; shift ;;
    --bedrock-gateway-arn) BEDROCK_GATEWAY_ARN="$2"; shift; shift ;;
    --prompt-registry-arn) PROMPT_REGISTRY_ARN="$2"; shift; shift ;;
    --tts-gateway-arn)     TTS_GATEWAY_ARN="$2"; shift; shift ;;
    --template-url)    TEMPLATE_URL="$2"; shift; shift ;;
    --region)          REGION="$2"; shift; shift ;;
    --profile)         PROFILE="$2"; shift; shift ;;
    *) shift ;;
  esac
done

[ -z "$STACK_NAME" ] && echo "error: --stack-name is required" && exit 1
[ -z "$RESOURCE_PREFIX" ] && echo "error: --resource-prefix is required" && exit 1
[ -z "$PROXY_BUCKET" ] && echo "error: --proxy-bucket is required" && exit 1
# the template declares these three with no Default — a launch without
# them is rejected by CloudFormation, so refuse here by name instead
[ -z "$BEDROCK_GATEWAY_ARN" ] && echo "error: --bedrock-gateway-arn is required (bedrock-inference stack's GatewayToolLambdaArn output)" && exit 1
[ -z "$PROMPT_REGISTRY_ARN" ] && echo "error: --prompt-registry-arn is required (prompt-registry stack's AdminToolLambdaArn output — the ad-* categories are internal, served on the admin surface only)" && exit 1
[ -z "$TTS_GATEWAY_ARN" ] && echo "error: --tts-gateway-arn is required (tts-on-aws stack's GatewayToolLambdaArn output)" && exit 1

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
  ParameterKey=BedrockInferenceGatewayLambdaArn,ParameterValue="$BEDROCK_GATEWAY_ARN"
  ParameterKey=PromptRegistryGatewayLambdaArn,ParameterValue="$PROMPT_REGISTRY_ARN"
  ParameterKey=TtsGatewayLambdaArn,ParameterValue="$TTS_GATEWAY_ARN"
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
