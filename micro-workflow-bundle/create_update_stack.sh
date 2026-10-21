#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Create or update the micro-workflow-bundle CloudFormation stack — one stack
# that nests every micro-workflow child.
#
# Run from micro-workflow-bundle/ or any directory — the template path is
# relative to the script location.
#
# Prerequisite: bash ./build-deploy-s3-dist.sh --bucket BUCKET
# (stages every child template plus this root template to S3).
#
# Usage:
#   bash ./create_update_stack.sh \
#       --stack-name my-micro-workflows \
#       --ingest-bucket my-ingest-bucket \
#       [--proxy-bucket my-proxy-bucket]   # omit to let the stack create one

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/dist/global-s3-assets/micro-workflow-bundle-stack.template"

######################################################################
# Usage
######################################################################
function usage() {
  echo "
------------------------------------------------------------------------------
Create or update the micro-workflow-bundle CloudFormation stack.

Deploys every micro-workflow stack as a nested stack: media-processing,
audiostem-separation, speech-diarization, shot-segmentation, face-recognition,
hybrid-search, text-to-speech, bedrock-inference, prompt-registry,
conversation-analysis, shot-metadata-enrichment, and the media-toolkit-gateway
AgentCore Gateway that exposes them all as MCP tools.

bash ./create_update_stack.sh \\
    --stack-name STACK_NAME \\
    --ingest-bucket INGEST_BUCKET \\
    [--proxy-bucket PROXY_BUCKET] \\
    [--prefix PREFIX] \\
    [--gateway-name GATEWAY_NAME] \\
    [--environment ENVIRONMENT] \\
    [--oauth-scope OAUTH_SCOPE] \\
    [--secrets-kms-key-id SECRETS_KMS_KEY_ID] \\
    [--template-url TEMPLATE_URL] \\
    [--region REGION] \\
    [--profile PROFILE]

where
  --stack-name        Name of the CloudFormation stack.

  --ingest-bucket     S3 bucket holding source media. Must already exist. No
                      default, and it must not gain one: it holds source media
                      this stack did not produce and cannot recreate, so a typo
                      would deploy clean and then find nothing to process.

  --proxy-bucket      [optional] S3 bucket workflows write outputs to. OMIT IT
                      and the stack creates and owns one named
                      {prefix}-{stack-id}-{account}-{region}-proxy. That is the
                      recommended path: it used to be mandatory only to guard
                      against the child validators' own create-on-404 behaviour
                      (a typo silently made a stray bucket), and declaring the
                      bucket removes that failure mode instead of guarding it.
                      Supply a name to reuse an existing bucket — including a
                      bucket RETAINED from a deleted bundle, which is the only
                      way to get those outputs back.

  --prefix            [optional] Prefix for resource names in the workflow
                      children (media-toolkit-gateway has no ResourcePrefix param).
                      Defaults to 'umt'. Max 14 chars, lowercase
                      alphanumerics and hyphens (it lands in S3 bucket names). A
                      12-char slice of the stack id is appended automatically,
                      so the effective prefix is {prefix}-{stack-id}.

  --gateway-name      [optional] AgentCore Gateway name. Omitted, the gateway
                      derives {prefix}-{stack-id} — unique per bundle, nothing
                      to vary. Set it to PIN a stable gateway identity: the
                      secret path follows it, so a pinned name survives a
                      stack re-creation with zero client reconfiguration.

  --environment       [optional] dev | staging | prod. Defaults to 'dev'.

  --oauth-scope       [optional] Cognito resource server Identifier. Defaults to
                      'media-workflows'.

  --secrets-kms-key-id [optional] KMS key for the credentials secret. Defaults to
                      'alias/aws/secretsmanager'.

  --template-url      [optional] Root template S3 URL. Derived from the built
                      template's Mappings when omitted.

  --region            [optional] AWS region.
  --profile           [optional] AWS CLI profile.

IMPORTANT — running two bundles in one account (any region):
  With --gateway-name OMITTED (the default) bundles never collide: the gateway
  stack derives {prefix}-{stack-id}, unique per stack, same as the workflow
  children. Only PINNED names carry a uniqueness duty: a pinned
  --gateway-name + --environment pair must be unique per ACCOUNT — the Cognito
  domain it seeds is a global namespace — or the gateway stack fails with
  'already exists' after the workflow children have built.

IMPORTANT — two resources OUTLIVE a stack delete, by design:
  A stack-created proxy bucket and hybrid-search's S3 Vectors bucket both carry
  DeletionPolicy: Retain, because their contents are expensive or impossible to
  rebuild. Both names embed a stack-id slice, so a delete + re-create strands
  them and the new stack makes different ones. Read ProxyBucketName and
  ResourcePrefix from the stack outputs BEFORE deleting, then either sweep the
  buckets by hand or pass the proxy bucket back via --proxy-bucket.

Expect a LONG first create (typically 1-2 hours). Most children build a
container image in CodeBuild inside their nested stack (60- and 90-minute
timeouts, running in parallel) and text-to-speech additionally seeds six model
archives to S3. The stack is not hung.
"
}

######################################################################
# Parse arguments
######################################################################
while [[ $# -gt 0 ]]; do
  case $1 in
    --stack-name)          STACK_NAME="$2";          shift 2 ;;
    --ingest-bucket)       INGEST_BUCKET="$2";       shift 2 ;;
    --proxy-bucket)        PROXY_BUCKET="$2";        shift 2 ;;
    --prefix)              PREFIX="$2";              shift 2 ;;
    --gateway-name)        GATEWAY_NAME="$2";        shift 2 ;;
    --environment)         ENVIRONMENT="$2";         shift 2 ;;
    --oauth-scope)         OAUTH_SCOPE="$2";         shift 2 ;;
    --secrets-kms-key-id)  SECRETS_KMS_KEY_ID="$2";  shift 2 ;;
    --template-url)        TEMPLATE_URL="$2";        shift 2 ;;
    --region)              REGION="$2";              shift 2 ;;
    --profile)             PROFILE="$2";             shift 2 ;;
    -h|--help)             usage; exit 0 ;;
    # --resource-prefix was renamed to --prefix when the stack started
    # appending a stack-id slice itself. Rejected explicitly rather than
    # falling through to the catch-all below, which would silently ignore it
    # and deploy under the DEFAULT prefix — i.e. a second bundle intended to
    # be isolated would collide with the first, ~90 minutes in.
    --resource-prefix)
      echo "error: --resource-prefix was renamed to --prefix."
      echo "       The stack now appends a 12-char stack-id slice itself, so"
      echo "       this flag sets only the leading part (max 14 chars)."
      exit 1 ;;
    *)                     shift ;;
  esac
done

######################################################################
# Validate
######################################################################
ERRORS=0
[ -z "$STACK_NAME" ]    && echo "error: --stack-name is required"    && ERRORS=$((ERRORS+1))
[ -z "$INGEST_BUCKET" ] && echo "error: --ingest-bucket is required" && ERRORS=$((ERRORS+1))

# --proxy-bucket is NOT validated for presence — omitting it is the recommended
# path and means "let the stack create and own one".
#
# --prefix IS validated here, ahead of the API call, because CloudFormation's own
# rejection arrives without context ("must contain at most 14 characters") and
# the reason for the ceiling is arithmetic worth stating: ResourcePrefix becomes
# {prefix}-{12-char stack-id} = up to 27 chars, and hybrid-search's S3 Vectors
# bucket name spends the remaining 36 of S3's 63-char limit.
if [ -n "$PREFIX" ]; then
  if [ "${#PREFIX}" -gt 14 ]; then
    echo "error: --prefix is ${#PREFIX} chars; max is 14"
    echo "       A 12-char stack-id slice is appended, and the resulting 27-char"
    echo "       prefix is what fits hybrid-search's S3 Vectors bucket name"
    echo "       inside S3's 63-char limit."
    ERRORS=$((ERRORS+1))
  fi
  if ! echo "$PREFIX" | grep -qE '^[a-z0-9][a-z0-9-]*$'; then
    echo "error: --prefix must be lowercase alphanumerics/hyphens starting with"
    echo "       an alphanumeric (got '${PREFIX}') — it lands in S3 bucket names."
    ERRORS=$((ERRORS+1))
  fi
fi

[ -z "$TEMPLATE_URL" ] && [ ! -f "$TEMPLATE" ] && \
  echo "error: ${TEMPLATE} not found — run build-deploy-s3-dist.sh first, or pass --template-url" && \
  ERRORS=$((ERRORS+1))

if [ -n "$ENVIRONMENT" ]; then
  case "$ENVIRONMENT" in
    dev|staging|prod) ;;
    *) echo "error: --environment must be dev, staging, or prod (got '${ENVIRONMENT}')"
       ERRORS=$((ERRORS+1)) ;;
  esac
fi

[ "$ERRORS" -gt 0 ] && usage && exit 1

######################################################################
# Build AWS CLI flags
######################################################################
AWS_FLAGS=()
[ -n "$REGION" ]  && AWS_FLAGS+=(--region "$REGION")
[ -n "$PROFILE" ] && AWS_FLAGS+=(--profile "$PROFILE")

######################################################################
# Resolve template URL
#
# The root template must be fetched from S3 rather than passed inline. Its own
# body is small enough for --template-body, but every child template it
# references exceeds the 51200-byte limit and CloudFormation fetches those by
# URL regardless — so there is no version of this that avoids
# S3, and staying consistent with the children is worth more than saving a
# fetch.
#
# Deriving from the BUILT template under dist/ — not from cfnstack.yaml — is
# what makes this safe: the source carries a %%BUCKET_NAME%% placeholder, so an
# unbuilt template resolves to a bucket literally named "%%BUCKET_NAME%%" and
# fails loudly, instead of silently resolving to a wrong-but-plausible bucket.
######################################################################
if [ -z "$TEMPLATE_URL" ]; then
  # PREFER the last build's own record of where its root template went
  # (written by build-deploy-s3-dist.sh next to the built template). The
  # bundle's .version FILE is NOT that record: --version overrides it at
  # build time without updating it, so a URL composed from the stale file
  # can name a DIFFERENT, older object that happens to exist. Measured
  # 2026-08-20: an update composed from .version=alpha.0807a silently
  # deployed a 2026-08-15 9-child root over the live 12-child
  # alpha.20260819a bundle, DELETING three nested stacks.
  T_MARKER="$SCRIPT_DIR/dist/global-s3-assets/.root_template_url"
  if [ -s "$T_MARKER" ]; then
    TEMPLATE_URL=$(cat "$T_MARKER")
    echo "Template URL recorded by the last build: ${TEMPLATE_URL}"
  else
    # Fallback for a dist/ produced by an older build script. S3Bucket, not
    # Bucket: this template's mapping key differs from the children's.
    T_BUCKET=$(grep -m1 'S3Bucket:' "$TEMPLATE" | sed 's/.*"\(.*\)".*/\1/')

    if [ -z "$T_BUCKET" ] || [ "${T_BUCKET#%%}" != "$T_BUCKET" ]; then
      echo "error: could not resolve S3Bucket from ${TEMPLATE}"
      echo "       Re-run build-deploy-s3-dist.sh to regenerate the processed template,"
      echo "       or pass --template-url explicitly."
      exit 1
    fi

    T_SOLUTION=$(grep -m1 -oE '"[^"/]+/[^"/]+/[^"/]+/[^"]+-stack\.template"' "$TEMPLATE" \
      | tr -d '"' | cut -d/ -f1)
    T_VERSION=$(cat "$SCRIPT_DIR/.version" 2>/dev/null)

    if [ -z "$T_SOLUTION" ] || [ -z "$T_VERSION" ]; then
      echo "error: could not derive the root template key from ${TEMPLATE}"
      echo "       Pass --template-url explicitly."
      exit 1
    fi

    DEPLOY_REGION="${REGION:-$(aws configure get region "${AWS_FLAGS[@]}" 2>/dev/null)}"
    DEPLOY_REGION="${DEPLOY_REGION:-us-east-1}"
    TEMPLATE_URL="https://${T_BUCKET}.s3.${DEPLOY_REGION}.amazonaws.com/${T_SOLUTION}/${T_VERSION}/micro-workflow-bundle/$(basename "$TEMPLATE")"
  fi

  # Cross-check: whatever URL was derived, the S3 object there must BE the
  # locally built template — byte-for-byte. A derivation can only compose a
  # plausible key; only a content match proves it names THIS build. The root
  # template is far below the multipart threshold, so ETag == md5.
  if [[ "$TEMPLATE_URL" =~ ^https://(.+)\.s3[^/]*\.amazonaws\.com/(.+)$ ]]; then
    V_BUCKET="${BASH_REMATCH[1]}"
    V_KEY="${BASH_REMATCH[2]}"
    REMOTE_ETAG=$(aws s3api head-object --bucket "$V_BUCKET" --key "$V_KEY" \
      "${AWS_FLAGS[@]}" --query ETag --output text 2>/dev/null | tr -d '"')
    LOCAL_MD5=$(md5 -q "$TEMPLATE" 2>/dev/null || md5sum "$TEMPLATE" | cut -d' ' -f1)
    if [ -z "$REMOTE_ETAG" ]; then
      echo "error: no object found at the derived template URL:"
      echo "       ${TEMPLATE_URL}"
      echo "       Re-run build-deploy-s3-dist.sh, or pass --template-url explicitly."
      exit 1
    fi
    if [[ "$REMOTE_ETAG" == *-* ]]; then
      echo "warning: multipart ETag at ${TEMPLATE_URL} — content check skipped"
    elif [ "$REMOTE_ETAG" != "$LOCAL_MD5" ]; then
      echo "error: the object at the derived template URL is NOT this build's template."
      echo "       ${TEMPLATE_URL}"
      echo "       S3 ETag ${REMOTE_ETAG} != local md5 ${LOCAL_MD5}."
      echo "       Deploying it would apply a DIFFERENT (possibly older) bundle —"
      echo "       nested stacks absent from it would be DELETED. Re-run"
      echo "       build-deploy-s3-dist.sh, or pass --template-url explicitly."
      exit 1
    else
      echo "Verified: staged root template matches the local build (md5 ${LOCAL_MD5})"
    fi
  fi
fi

echo "Template URL: ${TEMPLATE_URL}"
TEMPLATE_ARG=(--template-url "${TEMPLATE_URL}")
echo ""

######################################################################
# Parameters
######################################################################
#
# ProxyBucket is passed ALWAYS, including as an explicit empty value when
# --proxy-bucket was omitted. It must not be dropped from the list the way the
# other optionals are: on an UPDATE, an omitted parameter without
# UsePreviousValue keeps the value already on the stack. So a stack created with
# an explicit bucket, then updated without the flag, would silently keep that
# bucket instead of switching to a stack-owned one — the omission would appear to
# do nothing. Sending "" makes the intent explicit in both directions.
#
PARAMETERS=(
  "ParameterKey=IngestBucket,ParameterValue=${INGEST_BUCKET}"
  "ParameterKey=ProxyBucket,ParameterValue=${PROXY_BUCKET}"
)

# Optional parameters — only override when explicitly provided, otherwise use
# the template's own defaults.
[ -n "$PREFIX" ]             && PARAMETERS+=("ParameterKey=Prefix,ParameterValue=${PREFIX}")
[ -n "$GATEWAY_NAME" ]       && PARAMETERS+=("ParameterKey=GatewayName,ParameterValue=${GATEWAY_NAME}")
[ -n "$ENVIRONMENT" ]        && PARAMETERS+=("ParameterKey=Environment,ParameterValue=${ENVIRONMENT}")
[ -n "$OAUTH_SCOPE" ]        && PARAMETERS+=("ParameterKey=OAuthScope,ParameterValue=${OAUTH_SCOPE}")
[ -n "$SECRETS_KMS_KEY_ID" ] && PARAMETERS+=("ParameterKey=SecretsKmsKeyId,ParameterValue=${SECRETS_KMS_KEY_ID}")

# CAPABILITY_IAM is required: the children create IAM roles.
#
# CAPABILITY_NAMED_IAM is passed defensively, not because anything needs it
# today. Measured against real CloudFormation: no child declares RoleName,
# ManagedPolicyName, InstanceProfileName, or a standalone AWS::IAM::Policy —
# every named-looking hit is an inline PolicyName inside a role's Policies list
# (validate-template returns CAPABILITY_IAM for that shape, CAPABILITY_NAMED_IAM
# only for an explicit RoleName), or CodeBuild's LogsConfig GroupName, which is
# not IAM at all. Several of the children's own deploy scripts pass NAMED_IAM
# anyway; matching them costs nothing and means a child that later adds a
# RoleName does not fail the bundle ~90 minutes into a create.
#
# CAPABILITY_AUTO_EXPAND is deliberately NOT passed, even though
# "aws cloudformation validate-template" on this file ASKS for it. That response
# is a conservative superset for any template containing a nested stack —
# CloudFormation cannot see inside a child at validate time, so it names every
# capability the child might need. Verified by experiment: a minimal template
# whose only resource is an AWS::CloudFormation::Stack (no IAM, no transform)
# also reports NAMED_IAM + AUTO_EXPAND from validate-template, yet create-stack
# on it succeeds with CAPABILITY_IAM alone and with no capabilities at all.
# AUTO_EXPAND is required only for transforms and macros, and no child declares
# Transform, Fn::Transform, or AWS::Serverless. Do not "fix" this by copying
# validate-template's output — an unnecessary capability is a granted permission
# nobody audited.
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
# Report the proxy-bucket decision BEFORE acting on it.
#
# The stack-created name embeds a 12-char slice of the stack id, which does not
# exist until the stack does — so on a create the exact name genuinely cannot be
# printed here, only its shape. It is surfaced as the ProxyBucketName output
# instead. Saying which BRANCH will fire is the part that matters: the operator
# who meant to reuse a bucket and mistyped the flag name sees "creating a new
# one" here rather than discovering it an hour later.
######################################################################
if [ -z "$PROXY_BUCKET" ]; then
  echo "Proxy bucket: this stack will CREATE and OWN one, named"
  echo "              ${PREFIX:-umt}-{stack-id}-{account}-{region}-proxy"
  echo "              It is RETAINED on stack delete — read the ProxyBucketName"
  echo "              output before deleting, or the name is lost."
else
  echo "Proxy bucket: ${PROXY_BUCKET} (supplied; this stack will not own it)"
fi
echo ""

if [ "$ACTION" = "create" ]; then
  # COUNTED from the template, not hardcoded: this number grows with every new
  # child and nothing fails when the message goes stale. Read from cfnstack.yaml
  # rather than $TEMPLATE because --template-url is a valid path through this
  # script with no dist/ build present, and counting resources — unlike
  # resolving the bucket above — involves no %%TOKEN%% placeholder.
  nested=$(grep -c "Type: AWS::CloudFormation::Stack" "${SCRIPT_DIR}/cfnstack.yaml")
  echo "----------------------------------------------------------------------"
  echo "This create brings up ${nested} nested stacks. Most build a container image"
  echo "in CodeBuild (60-90 min timeouts, in parallel) and text-to-speech also"
  echo "seeds six model archives. Typically 1-2 hours. It is not hung."
  echo "----------------------------------------------------------------------"
  echo ""
fi

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

  echo "$UPDATE_OUTPUT"
  echo "Waiting for stack update to complete..."
  aws cloudformation wait stack-update-complete \
    --stack-name "$STACK_NAME" \
    "${AWS_FLAGS[@]}"
  WAIT_EXIT=$?
fi

######################################################################
# Report
######################################################################
if [ $WAIT_EXIT -ne 0 ]; then
  echo ""
  echo "error: stack ${ACTION} did not complete successfully."
  echo "Most recent failure events:"
  aws cloudformation describe-stack-events \
    --stack-name "$STACK_NAME" \
    "${AWS_FLAGS[@]}" \
    --query "StackEvents[?contains(ResourceStatus,'FAILED')].[Timestamp,LogicalResourceId,ResourceStatusReason]" \
    --output table 2>/dev/null | head -40
  echo ""
  echo "A nested stack failure is reported here only as the child's logical id"
  echo "(e.g. MediaProcessingStack). Open that nested stack in the console for"
  echo "the underlying resource error."
  exit 1
fi

echo ""
echo "======================================================================"
echo " Stack ${ACTION} complete: ${STACK_NAME}"
echo "======================================================================"
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  "${AWS_FLAGS[@]}" \
  --query "Stacks[0].Outputs[*].[OutputKey,OutputValue]" \
  --output table

echo ""
echo "Read the gateway credentials bundle with:"
echo "  aws secretsmanager get-secret-value --secret-id <GatewayCredentialsSecretName>"
