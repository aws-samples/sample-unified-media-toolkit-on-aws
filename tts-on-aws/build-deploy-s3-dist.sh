#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Run this script from the repo root (tts-on-aws/).
#
# bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]
# bash ./build-deploy-s3-dist.sh --build-only [--version VERSION]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKFLOW_NAME="tts"
TEMPLATE_NAME="${WORKFLOW_NAME}-stack.template"
DOCKER_NAME="tts-on-aws"

TEMPLATE_DIST_DIR="$SCRIPT_DIR/dist/global-s3-assets"
BUILD_DIST_DIR="$SCRIPT_DIR/dist/regional-s3-assets"

#
# @function usage
#
function usage() {
  echo -e "
------------------------------------------------------------------------------

Build and deploy the CosyVoice3 TTS (ONNX Runtime) CloudFormation stack.
Run from the repo root (tts-on-aws/).

------------------------------------------------------------------------------
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region] [--build-only]

where
  --bucket BUCKET     S3 bucket name to deploy to. Without --single-region this is
                      treated as a basename and the region is appended, e.g. BUCKET-us-east-1.

  --build-only        Build packages only; skip deploy to S3 (no AWS calls).

  --solution SOLUTION [optional] S3 key prefix namespace. Defaults to 'unified-media-toolkit'.

  --version VERSION   [optional] Image/stack version. Defaults to contents of .version file.

  --single-region     [optional] Deploy to exactly the bucket specified in --bucket (no region suffix).

  --acl ACL           [optional] S3 object ACL. Defaults to 'bucket-owner-full-control'.
"
  return 0
}

######################################################################
# Parse arguments
######################################################################
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -b|--bucket)
      BUCKET_NAME="$2"
      shift; shift ;;
    --build-only)
      BUILD_ONLY=true
      shift ;;
    -s|--solution)
      SOLUTION="$2"
      shift; shift ;;
    -v|--version)
      VERSION="$2"
      shift; shift ;;
    -r|--single-region)
      SINGLE_REGION=true
      shift ;;
    -a|--acl)
      ACL_SETTING="$2"
      shift; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      shift ;;
  esac
done

######################################################################
# Validate prerequisites and arguments
######################################################################
[ -z "$BUILD_ONLY" ] && \
  BUILD_ONLY=false

[ "$(which jq)" == "" ] && \
  echo "error: jq is required" && exit 1

if [ "$BUILD_ONLY" == "false" ]; then
  [ "$(which aws)" == "" ] && \
    echo "error: AWS CLI is required" && exit 1

  [ -z "$BUCKET_NAME" ] && \
    echo "error: missing --bucket parameter" && usage && exit 1
fi

[ -z "$VERSION" ] && \
  VERSION=$(cat "$SCRIPT_DIR/.version" 2>/dev/null)

[ -z "$VERSION" ] && \
  echo "error: cannot determine version — use --version or create a .version file" && \
  usage && exit 1

[ -z "$SOLUTION" ] && \
  SOLUTION="unified-media-toolkit"

[ -z "$SINGLE_REGION" ] && \
  SINGLE_REGION=false

[ -z "$ACL_SETTING" ] && \
  ACL_SETTING="bucket-owner-full-control"

if [ "$BUILD_ONLY" == "false" ]; then
  ACCOUNTID=$(aws sts get-caller-identity | jq -r .Account)
  [ -z "$ACCOUNTID" ] && \
    echo "error: failed to get AWS Account ID" && exit 1

  aws s3api get-bucket-location \
    --bucket "${BUCKET_NAME}" \
    --expected-bucket-owner "${ACCOUNTID}" > /dev/null 2>&1
  [ $? -ne 0 ] && \
    echo "error: bucket '${BUCKET_NAME}' does not exist or does not belong to account '${ACCOUNTID}'" && \
    exit 1
fi

[ -z "$BUCKET_NAME" ] && \
  BUCKET_NAME="BUCKET_NAME_NOT_SET"

######################################################################
# Derived variables
######################################################################
PKG_TTS_ON_AWS="${DOCKER_NAME}-${VERSION}.zip"
PKG_GATEWAY_TOOL="tts-on-aws-gateway-tool-${VERSION}.zip"
KEY_PREFIX="${SOLUTION}/${VERSION}/${WORKFLOW_NAME}"

echo "======================================================================"
if [ "$BUILD_ONLY" == "true" ]; then
  echo " CosyVoice3 TTS on AWS — Build Only"
else
  echo " CosyVoice3 TTS on AWS — Build & Deploy"
fi
echo "======================================================================"
echo "  solution  : $SOLUTION"
echo "  version   : $VERSION"
echo "  bucket    : $BUCKET_NAME"
echo "  key prefix: $KEY_PREFIX"
echo "  package   : $PKG_TTS_ON_AWS"
echo "  template  : $TEMPLATE_NAME"
echo "======================================================================"

######################################################################
# @function clean_start
######################################################################
function clean_start() {
  echo "------------------------------------------------------------------------------"
  echo "Cleaning dist directories"
  echo "------------------------------------------------------------------------------"
  rm -rf "$SCRIPT_DIR/dist"
  mkdir -p "$TEMPLATE_DIST_DIR" "$BUILD_DIST_DIR"

  # remove macOS cruft
  find "$SCRIPT_DIR" -name '.DS_Store' -type f -delete
}

######################################################################
# @function run_unit_tests
# Refuses to package on failure. Discovers test/test_*.py — the suite
# (payload-ceiling estimators, iterator scheduling, gateway contracts,
# ASL-vs-cfnstack.yaml parity); the other files in test/ are one-off
# research scripts and stay out of the build gate.
######################################################################
function run_unit_tests() {
  echo "------------------------------------------------------------------------------"
  echo "Running unit tests"
  echo "------------------------------------------------------------------------------"
  pushd "$SCRIPT_DIR" > /dev/null
  python3 -m unittest discover -s test -p 'test_*.py' || {
    echo "error: unit tests failed — refusing to build"
    exit 1
  }
  popd > /dev/null
}

######################################################################
# @function build_docker_package
# Zip the entire docker context (excluding dist/, __pycache__, .git)
# and place it in BUILD_DIST_DIR. CodeBuild will use this as its S3 source.
######################################################################
function build_docker_package() {
  echo "------------------------------------------------------------------------------"
  echo "Building Docker source package: $PKG_TTS_ON_AWS"
  echo "------------------------------------------------------------------------------"
  pushd "$SCRIPT_DIR" > /dev/null

  # clean up Python byte-code before packaging
  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
  find . -name '*.pyc' -type f -delete 2>/dev/null || true

  zip -rq "$BUILD_DIST_DIR/$PKG_TTS_ON_AWS" \
    .version \
    assets \
    lib \
    states \
    Dockerfile \
    lambda_function.py \
    patch.txt \
    requirements.txt

  echo ">> Built: $BUILD_DIST_DIR/$PKG_TTS_ON_AWS"
  popd > /dev/null
}

######################################################################
# @function build_gateway_tool_zip
# Packages gateway_tool/*.py (dispatcher + bucket_validator custom
# resource handler) plus output_contract.json and input_examples.json
# (the output contract and per-tool example requests served by
# describe_tool) as a standalone zip. This is a
# pure-Python zip (no Docker) deployed to the tool Lambda AND the
# ProxyBucketValidatorFunction (cfnstack.yaml) — it does NOT get lib/
# (see gateway_tool/lambda_function.py's own docstring: style-name/
# business-rule validation lives in StateCheckAndPreprocess, which does
# have lib/, specifically so the gateway tool never needs it).
######################################################################
function build_gateway_tool_zip() {
  echo "------------------------------------------------------------------------------"
  echo "Building gateway tool zip: $PKG_GATEWAY_TOOL"
  echo "------------------------------------------------------------------------------"
  local members=(lambda_function.py bucket_validator.py workflow_identity.py
                 callback_registry.py callback_waker.py synthesis_outputs.py
                 output_contract.json input_examples.json)

  pushd "$SCRIPT_DIR/gateway_tool" > /dev/null

  # zip exits 0 when only SOME members are missing (it warns and packages the
  # rest), so a missing handler would ship as a silently broken Lambda —
  # here that means the ProxyBucketValidator custom resource never returns
  # and the stack hangs an hour before rolling back. Check each up front.
  local f
  for f in "${members[@]}"; do
    [ ! -f "$f" ] && \
      echo "error: gateway_tool/$f not found" && exit 1
  done

  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

  zip -jq "$BUILD_DIST_DIR/$PKG_GATEWAY_TOOL" "${members[@]}" || exit 1

  echo ">> Built: $BUILD_DIST_DIR/$PKG_GATEWAY_TOOL"
  popd > /dev/null
}

######################################################################
# @function build_cloudformation_template
# Copy cfnstack.yaml to TEMPLATE_DIST_DIR, substitute tokens, rename
# to .template, and also place a copy in BUILD_DIST_DIR (regional).
# Both state machines are defined INLINE in cfnstack.yaml (DefinitionString:
# !Sub), so definitions.json/definitions_register_voice.json are NOT
# deployment artifacts — they ship only inside source_code.zip as the
# editable source of truth, and nothing at build time checks that the two
# copies still agree. Keep them in sync by hand when editing either.
######################################################################
function build_cloudformation_template() {
  echo "------------------------------------------------------------------------------"
  echo "Building CloudFormation template: $TEMPLATE_NAME"
  echo "------------------------------------------------------------------------------"
  local src="$SCRIPT_DIR/cfnstack.yaml"
  local dst="$TEMPLATE_DIST_DIR/$TEMPLATE_NAME"

  [ ! -f "$src" ] && \
    echo "error: cfnstack.yaml not found at $src" && exit 1

  cp "$src" "$dst"

  sed -i'.bak' -e "s|%%BUCKET_NAME%%|${BUCKET_NAME}|g"       "$dst" || exit 1
  sed -i'.bak' -e "s|%%KEY_PREFIX%%|${KEY_PREFIX}|g"         "$dst" || exit 1
  sed -i'.bak' -e "s|%%VERSION%%|${VERSION}|g"               "$dst" || exit 1
  sed -i'.bak' -e "s|%%PKG_TTS_ON_AWS%%|${PKG_TTS_ON_AWS}|g" "$dst" || exit 1
  sed -i'.bak' -e "s|%%PKG_GATEWAY_TOOL%%|${PKG_GATEWAY_TOOL}|g" "$dst" || exit 1

  rm -f "${dst}.bak"

  # A leftover %%TOKEN%% deploys as a literal string — an S3Key pointing at
  # "%%PKG_TTS_ON_AWS%%" fails the stack minutes in, after CodeBuild has
  # already started. Fail here instead.
  local leftover
  leftover=$(grep -o '%%[A-Z_]*%%' "$dst" | sort -u | tr '\n' ' ')
  [ -n "$leftover" ] && \
    echo "error: unsubstituted token(s) in $TEMPLATE_NAME: $leftover" && exit 1

  # also copy processed template to regional assets so both buckets have it
  cp "$dst" "$BUILD_DIST_DIR/$TEMPLATE_NAME"

  echo ">> Built: $dst"
}

######################################################################
# @function deploy_to_bucket
# Upload regional-s3-assets (packages + template copy) to the given bucket.
######################################################################
function deploy_to_bucket() {
  local bucket=$1
  local dest="s3://${bucket}/${KEY_PREFIX}/"

  local location
  location=$(aws s3api get-bucket-location \
    --bucket "${bucket}" \
    --expected-bucket-owner "${ACCOUNTID}" \
    | jq -r '.LocationConstraint // "us-east-1"')

  [ -z "$location" ] && \
    echo "error: bucket '${bucket}' not found or not owned by account '${ACCOUNTID}'" && \
    return 1

  local region="us-east-1"
  [ "$location" != "null" ] && region="$location"

  local acl_flag="--acl ${ACL_SETTING}"
  local region_flag=""
  [ "$region" != "us-east-1" ] && region_flag="--region ${region}"

  echo ""
  echo "== Deploying to s3://${bucket}/${KEY_PREFIX}/ (${region}) =="
  aws s3 cp "$BUILD_DIST_DIR" "$dest" --recursive $acl_flag $region_flag
  echo "== Deploy to ${bucket} [COMPLETED] =="

  # also upload the template to global-s3-assets path
  aws s3 cp "$TEMPLATE_DIST_DIR/$TEMPLATE_NAME" \
    "s3://${bucket}/${KEY_PREFIX}/${TEMPLATE_NAME}" \
    $acl_flag $region_flag

  local domain="s3.amazonaws.com"
  [ "$region" != "us-east-1" ] && domain="s3.${region}.amazonaws.com"

  local url="https://${bucket}.${domain}/${KEY_PREFIX}/${TEMPLATE_NAME}"
  echo ""
  echo "========================================"
  echo "  Template URL:"
  echo "  $url"
  echo ""
  echo "  One-click deploy:"
  echo "  https://console.aws.amazon.com/cloudformation/home?region=${region}#/stacks/quickcreate?templateURL=${url}&stackName=tts-on-aws"
  echo "========================================"
}

######################################################################
# @function on_complete
######################################################################
function on_complete() {
  echo ""
  echo "======================================================================"
  echo " Build & Deploy Complete — ${SOLUTION} ${VERSION}"
  echo "======================================================================"
  echo "  PKG_TTS_ON_AWS   : $PKG_TTS_ON_AWS"
  echo "  PKG_GATEWAY_TOOL : $PKG_GATEWAY_TOOL"
  echo "  TEMPLATE         : $TEMPLATE_NAME"
  echo "======================================================================"
}

######################################################################
# @function assert_internal_docs_excluded
# GOTCHAS.md and the harness run manifests are INTERNAL ONLY -- they carry
# the customer, programme and asset identities that the code and
# README.md deliberately refer to only by generic corpus label. Every
# packaging function above is an explicit allowlist, so today they are
# excluded by construction; this asserts it, because "the allowlist happens
# not to name it" is a property one `test \` or `. \` added to a zip -r
# erases silently. Deliberately FAILS the build rather than quietly
# excluding: a build that starts shipping internal docs is a scrub
# regression someone needs to see, not something to paper over with -x.
######################################################################
function assert_internal_docs_excluded() {
  echo "------------------------------------------------------------------------------"
  echo "Verifying internal docs are excluded from every artifact"
  echo "------------------------------------------------------------------------------"

  if ! command -v unzip > /dev/null 2>&1; then
    echo "error: unzip not found -- cannot verify that internal docs stay out of"
    echo "       the artifacts. Install unzip; this check does not get skipped."
    exit 1
  fi

  local violations=0
  local archive member
  for archive in "$BUILD_DIST_DIR"/*.zip; do
    [ -f "$archive" ] || continue
    while IFS= read -r member; do
      case "$member" in
        */) continue ;;                       # directory entry, not a file
        GOTCHAS.md|*/GOTCHAS.md|test/harness/runs/*|*/test/harness/runs/*)
          echo "  [LEAK] $(basename "$archive"): $member"
          violations=$((violations+1))
          ;;
      esac
    done < <(unzip -Z1 "$archive")
  done

  if [ $violations -ne 0 ] ; then
    echo ""
    echo "error: ${violations} internal file(s) would ship in the source packages."
    echo "       Remove them from the zip member list above and re-run."
    exit 1
  fi

  echo "  no internal docs in any artifact"
}

######################################################################
# Main
######################################################################
clean_start
run_unit_tests
build_docker_package
build_gateway_tool_zip
build_cloudformation_template
assert_internal_docs_excluded

if [ "$BUILD_ONLY" == "true" ]; then
  on_complete
  exit 0
fi

if [ "$SINGLE_REGION" == "true" ]; then
  deploy_to_bucket "${BUCKET_NAME}"
else
  # derive region from bucket location and append as suffix
  BUCKET_REGION=$(aws s3api get-bucket-location \
    --bucket "${BUCKET_NAME}" \
    --expected-bucket-owner "${ACCOUNTID}" \
    | jq -r '.LocationConstraint // "us-east-1"')
  [ "$BUCKET_REGION" == "null" ] && BUCKET_REGION="us-east-1"

  deploy_to_bucket "${BUCKET_NAME}-${BUCKET_REGION}"
fi

on_complete
