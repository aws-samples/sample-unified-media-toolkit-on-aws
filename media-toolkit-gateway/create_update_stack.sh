#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Create or update the media-toolkit-gateway CloudFormation stack.
# Run from media-toolkit-gateway/ or any directory — template path is relative
# to the script location.
#
# Prerequisites: the required micro-workflow stacks must already be deployed so
# their gateway tool Lambda ARNs are available:
#   - media-processing        → GatewayToolLambdaArn output
#   - audiostem-separation    → GatewayToolLambdaArn output
#   - speech-diarization      → GatewayToolLambdaArn output
#   - shot-segmentation       → GatewayToolLambdaArn output (optional)
#   - face-recognition        → GatewayToolLambdaArn output (optional)
#   - hybrid-search           → GatewayToolLambdaArn output (optional)
#   - hybrid-search (admin)   → AdminToolLambdaArn output (optional)
#   - tts-on-aws              → GatewayToolLambdaArn output (optional)
#
# Usage:
#   bash ./create_update_stack.sh \
#       --stack-name my-agent-tools \
#       --audio-stem-lambda-arn arn:aws:lambda:... \
#       --speech-diarization-lambda-arn arn:aws:lambda:... \
#       --media-processing-lambda-arn arn:aws:lambda:... \
#       --shot-segmentation-lambda-arn arn:aws:lambda:... \
#       --face-recognition-lambda-arn arn:aws:lambda:... \
#       --hybrid-search-lambda-arn arn:aws:lambda:... \
#       --hybrid-search-admin-lambda-arn arn:aws:lambda:... \
#       --text-to-speech-lambda-arn arn:aws:lambda:... \
#       --proxy-bucket my-proxy-bucket

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/dist/global-s3-assets/media-toolkit-gateway-stack.template"

######################################################################
# Usage
######################################################################
function usage() {
  echo "
------------------------------------------------------------------------------
Create or update the media-toolkit-gateway CloudFormation stack.

bash ./create_update_stack.sh \\
    --stack-name STACK_NAME \\
    --audio-stem-lambda-arn AUDIO_STEM_LAMBDA_ARN \\
    --speech-diarization-lambda-arn SPEECH_DIARIZATION_LAMBDA_ARN \\
    --media-processing-lambda-arn MEDIA_PROCESSING_LAMBDA_ARN \\
    --shot-segmentation-lambda-arn SHOT_SEGMENTATION_LAMBDA_ARN \\
    --face-recognition-lambda-arn FACE_RECOGNITION_LAMBDA_ARN \\
    --proxy-bucket PROXY_BUCKET \\
    [--gateway-name GATEWAY_NAME] \\
    [--environment ENVIRONMENT] \\
    [--oauth-scope OAUTH_SCOPE] \\
    [--secrets-kms-key-id SECRETS_KMS_KEY_ID] \\
    [--template-url TEMPLATE_URL] \\
    [--region REGION] \\
    [--profile PROFILE]

Required:
  --stack-name                    CloudFormation stack name
  --audio-stem-lambda-arn         GatewayToolLambdaArn from the audiostem-separation stack
  --speech-diarization-lambda-arn GatewayToolLambdaArn from the speech-diarization stack
  --media-processing-lambda-arn   GatewayToolLambdaArn from the media-processing stack
  --shot-segmentation-lambda-arn  GatewayToolLambdaArn from the shot-segmentation stack
  --proxy-bucket                  S3 proxy bucket the micro-workflows write outputs to (required)

Optional targets (omit to deploy without them):
  --face-recognition-lambda-arn   GatewayToolLambdaArn from the face-recognition stack
  --hybrid-search-lambda-arn      GatewayToolLambdaArn from the hybrid-search stack
  --hybrid-search-admin-lambda-arn AdminToolLambdaArn from the hybrid-search stack (admin target)
  --text-to-speech-lambda-arn     GatewayToolLambdaArn from the tts-on-aws stack

Optional (CFN parameter defaults apply when omitted):
  --gateway-name         AgentCore Gateway name (default: media-toolkit-gateway)
  --environment          Deployment environment: dev | staging | prod (default: dev)
  --oauth-scope          Cognito resource server Identifier (default: media-workflows)
  --secrets-kms-key-id   KMS key for the credentials secret (default: alias/aws/secretsmanager)
  --template-url         [optional] HTTPS S3 URL of the uploaded template. Defaults to the
                         URL derived from the built template's Bucket/KeyPrefix mappings.
                         The template always comes from S3 — it exceeds the 51200-byte
                         --template-body limit.
  --region               AWS region (defaults to AWS CLI default)
  --profile              AWS CLI profile
------------------------------------------------------------------------------
"
  return 0
}

######################################################################
# Parse arguments
######################################################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack-name)                    STACK_NAME="$2";                    shift 2 ;;
    --audio-stem-lambda-arn)         AUDIO_STEM_LAMBDA_ARN="$2";        shift 2 ;;
    --speech-diarization-lambda-arn) SPEECH_DIARIZATION_LAMBDA_ARN="$2"; shift 2 ;;
    --media-processing-lambda-arn)   MEDIA_PROCESSING_LAMBDA_ARN="$2";  shift 2 ;;
    --shot-segmentation-lambda-arn)  SHOT_SEGMENTATION_LAMBDA_ARN="$2"; shift 2 ;;
    --face-recognition-lambda-arn)   FACE_RECOGNITION_LAMBDA_ARN="$2";  shift 2 ;;
    --hybrid-search-lambda-arn)      HYBRID_SEARCH_LAMBDA_ARN="$2";     shift 2 ;;
    --hybrid-search-admin-lambda-arn) HYBRID_SEARCH_ADMIN_LAMBDA_ARN="$2"; shift 2 ;;
    --text-to-speech-lambda-arn)     TEXT_TO_SPEECH_LAMBDA_ARN="$2";      shift 2 ;;
    --proxy-bucket)                  PROXY_BUCKET="$2"; shift 2 ;;
    --gateway-name)                  GATEWAY_NAME="$2";                  shift 2 ;;
    --environment)                   ENVIRONMENT="$2";                   shift 2 ;;
    --oauth-scope)                   OAUTH_SCOPE="$2";                   shift 2 ;;
    --secrets-kms-key-id)            SECRETS_KMS_KEY_ID="$2";           shift 2 ;;
    --template-url)                  TEMPLATE_URL="$2";                  shift 2 ;;
    --region)                        REGION="$2";                        shift 2 ;;
    --profile)                       PROFILE="$2";                       shift 2 ;;
    -h|--help)                       usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

######################################################################
# Validate
######################################################################
ERRORS=0
[ -z "$STACK_NAME" ]                    && echo "error: --stack-name is required"                    && ERRORS=$((ERRORS+1))
[ -z "$AUDIO_STEM_LAMBDA_ARN" ]         && echo "error: --audio-stem-lambda-arn is required"         && ERRORS=$((ERRORS+1))
[ -z "$SPEECH_DIARIZATION_LAMBDA_ARN" ] && echo "error: --speech-diarization-lambda-arn is required" && ERRORS=$((ERRORS+1))
[ -z "$MEDIA_PROCESSING_LAMBDA_ARN" ]   && echo "error: --media-processing-lambda-arn is required"   && ERRORS=$((ERRORS+1))
[ -z "$SHOT_SEGMENTATION_LAMBDA_ARN" ]  && echo "error: --shot-segmentation-lambda-arn is required"  && ERRORS=$((ERRORS+1))
[ -z "$PROXY_BUCKET" ]                  && echo "error: --proxy-bucket is required"                  && ERRORS=$((ERRORS+1))
[ -z "$TEMPLATE_URL" ] && [ ! -f "$TEMPLATE" ] && echo "error: ${TEMPLATE} not found — run build-deploy-s3-dist.sh first, or pass --template-url" && ERRORS=$((ERRORS+1))
[ "$ERRORS" -gt 0 ]                     && usage && exit 1

######################################################################
# Build AWS CLI flags
######################################################################
AWS_FLAGS=()
[ -n "$REGION" ]  && AWS_FLAGS+=(--region "$REGION")
[ -n "$PROFILE" ] && AWS_FLAGS+=(--profile "$PROFILE")

######################################################################
# Resolve template URL
# The template exceeds the 51200-byte --template-body limit, so it must
# be fetched from S3. An explicit --template-url wins; otherwise derive
# it from the Bucket/KeyPrefix in the BUILT template's CodeBuild Mappings
# (the same approach media-processing/create_update_stack.sh uses).
#
# Deriving from the built template — not cfnstack.yaml — is what makes
# this safe: the source carries %%BUCKET_NAME%% placeholders, so an
# unbuilt template yields a bucket named "%%BUCKET_NAME%%" rather than a
# silently wrong one, and the S3Key for the gateway_tool zip inside it
# would be an unsubstituted token too.
######################################################################
if [ -z "$TEMPLATE_URL" ]; then
  T_BUCKET=$(grep -m1 'Bucket:' "$TEMPLATE" | sed 's/.*"\(.*\)".*/\1/')
  T_KEY_PREFIX=$(grep -m1 'KeyPrefix:' "$TEMPLATE" | sed 's/.*"\(.*\)".*/\1/')

  if [ -z "$T_BUCKET" ] || [ -z "$T_KEY_PREFIX" ] || \
     [ "${T_BUCKET#%%}" != "$T_BUCKET" ]; then
    echo "error: could not resolve Bucket/KeyPrefix from ${TEMPLATE}"
    echo "       Re-run build-deploy-s3-dist.sh to regenerate the processed template,"
    echo "       or pass --template-url explicitly."
    exit 1
  fi

  DEPLOY_REGION="${REGION:-$(aws configure get region 2>/dev/null)}"
  DEPLOY_REGION="${DEPLOY_REGION:-us-east-1}"
  TEMPLATE_URL="https://${T_BUCKET}.s3.${DEPLOY_REGION}.amazonaws.com/${T_KEY_PREFIX}/$(basename "$TEMPLATE")"
fi

echo "Template URL: ${TEMPLATE_URL}"
TEMPLATE_ARG=(--template-url "${TEMPLATE_URL}")
echo ""

# Required parameters
PARAMETERS=(
  "ParameterKey=AudioStemToolLambdaArn,ParameterValue=${AUDIO_STEM_LAMBDA_ARN}"
  "ParameterKey=SpeechDiarizationToolLambdaArn,ParameterValue=${SPEECH_DIARIZATION_LAMBDA_ARN}"
  "ParameterKey=MediaProcessingToolLambdaArn,ParameterValue=${MEDIA_PROCESSING_LAMBDA_ARN}"
  "ParameterKey=ShotSegmentationToolLambdaArn,ParameterValue=${SHOT_SEGMENTATION_LAMBDA_ARN}"
  "ParameterKey=ProxyBucket,ParameterValue=${PROXY_BUCKET}"
)

# Optional targets — pass only when provided (CFN default "" omits the target)
[ -n "$FACE_RECOGNITION_LAMBDA_ARN" ] && PARAMETERS+=("ParameterKey=FaceRecognitionToolLambdaArn,ParameterValue=${FACE_RECOGNITION_LAMBDA_ARN}")
[ -n "$HYBRID_SEARCH_LAMBDA_ARN" ] && PARAMETERS+=("ParameterKey=HybridSearchToolLambdaArn,ParameterValue=${HYBRID_SEARCH_LAMBDA_ARN}")
[ -n "$HYBRID_SEARCH_ADMIN_LAMBDA_ARN" ] && PARAMETERS+=("ParameterKey=HybridSearchAdminToolLambdaArn,ParameterValue=${HYBRID_SEARCH_ADMIN_LAMBDA_ARN}")
[ -n "$TEXT_TO_SPEECH_LAMBDA_ARN" ] && PARAMETERS+=("ParameterKey=TextToSpeechToolLambdaArn,ParameterValue=${TEXT_TO_SPEECH_LAMBDA_ARN}")

# Optional parameters — only override when explicitly provided, otherwise use CFN defaults
[ -n "$GATEWAY_NAME" ]       && PARAMETERS+=("ParameterKey=GatewayName,ParameterValue=${GATEWAY_NAME}")
[ -n "$ENVIRONMENT" ]        && PARAMETERS+=("ParameterKey=Environment,ParameterValue=${ENVIRONMENT}")
[ -n "$OAUTH_SCOPE" ]        && PARAMETERS+=("ParameterKey=OAuthScope,ParameterValue=${OAUTH_SCOPE}")
[ -n "$SECRETS_KMS_KEY_ID" ] && PARAMETERS+=("ParameterKey=SecretsKmsKeyId,ParameterValue=${SECRETS_KMS_KEY_ID}")

CAPABILITIES="CAPABILITY_IAM"

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
    "${TEMPLATE_ARG[@]}" \
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
    "${TEMPLATE_ARG[@]}" \
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

# Print the key outputs (GatewayUrl, CredentialsSecretArn, etc.)
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  "${AWS_FLAGS[@]}" \
  --query "Stacks[0].Outputs[*].[OutputKey,OutputValue]" \
  --output table
