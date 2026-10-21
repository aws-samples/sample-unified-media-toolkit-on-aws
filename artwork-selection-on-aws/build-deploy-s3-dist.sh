#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Run this script from artwork-selection-on-aws/
#
# bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]
# bash ./build-deploy-s3-dist.sh --build-only [--version VERSION]
#
# No docker, no CodeBuild, no ECR: pure boto3+stdlib, so ONE flat Lambda zip
# carries BOTH handlers (the gateway dispatcher and the operation-routed
# workflow Lambda) plus every lib/ and states/ module at the zip ROOT —
# flat because the handlers resolve top-level module paths
# (lambda_function.lambda_handler, router.lambda_handler).
#
# BUMP .version ON EVERY DEPLOY. S3Key carries the version, CloudFormation
# detects key changes and never content changes, so an unbumped redeploy
# leaves the old code running with no error to notice.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKFLOW_NAME="artwork-selection"
TEMPLATE_NAME="${WORKFLOW_NAME}-stack.template"

TEMPLATE_DIST_DIR="$SCRIPT_DIR/dist/global-s3-assets"
BUILD_DIST_DIR="$SCRIPT_DIR/dist/regional-s3-assets"

function usage() {
  echo -e "
------------------------------------------------------------------------------

Build and deploy the Artwork Selection CloudFormation stack.
Run from artwork-selection-on-aws/

------------------------------------------------------------------------------
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region] [--build-only]

where
  --bucket BUCKET     S3 bucket name to deploy to. Without --single-region this is
                      treated as a basename and the region is appended.
  --build-only        Build packages only; skip deploy to S3.
  --solution SOLUTION [optional] S3 key prefix namespace. Defaults to 'unified-media-toolkit'.
  --version VERSION   [optional] Stack version. Defaults to contents of .version file.
  --single-region     [optional] Deploy to exactly the bucket specified (no region suffix).
  --acl ACL           [optional] S3 object ACL. Defaults to 'bucket-owner-full-control'.

A CONSUMER stack: BedrockInferenceGatewayLambdaArn and
PromptRegistryGatewayLambdaArn are required parameters (the siblings'
GatewayToolLambdaArn outputs), so those two stacks deploy first.
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

[ -z "$BUILD_ONLY" ] && BUILD_ONLY=false

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

[ -z "$SOLUTION" ] && SOLUTION="unified-media-toolkit"
[ -z "$SINGLE_REGION" ] && SINGLE_REGION=false
[ -z "$ACL_SETTING" ] && ACL_SETTING="bucket-owner-full-control"

if [ "$BUILD_ONLY" == "false" ]; then
  ACCOUNTID=$(aws sts get-caller-identity | jq -r .Account)
  [ -z "$ACCOUNTID" ] && \
    echo "error: failed to get AWS Account ID" && exit 1
fi

[ -z "$BUCKET_NAME" ] && BUCKET_NAME="BUCKET_NAME_NOT_SET"

PKG_GATEWAY_TOOL="${WORKFLOW_NAME}-gateway-tool-${VERSION}.zip"
PKG_WORKFLOW="${WORKFLOW_NAME}-${VERSION}.zip"
KEY_PREFIX="${SOLUTION}/${VERSION}/${WORKFLOW_NAME}"

echo "======================================================================"
if [ "$BUILD_ONLY" == "true" ]; then
  echo " Artwork Selection — Build Only"
else
  echo " Artwork Selection — Build & Deploy"
fi
echo "======================================================================"
echo "  solution  : $SOLUTION"
echo "  version   : $VERSION"
echo "  bucket    : $BUCKET_NAME"
echo "  key prefix: $KEY_PREFIX"
echo "  gateway   : $PKG_GATEWAY_TOOL"
echo "  workflow  : $PKG_WORKFLOW"
echo "  template  : $TEMPLATE_NAME"
echo "======================================================================"

function clean_start() {
  echo "------------------------------------------------------------------------------"
  echo "Cleaning dist directories"
  echo "------------------------------------------------------------------------------"
  rm -rf "$SCRIPT_DIR/dist"
  mkdir -p "$TEMPLATE_DIST_DIR" "$BUILD_DIST_DIR"
  find "$SCRIPT_DIR" -name '.DS_Store' -type f -delete
}

######################################################################
# @function run_unit_tests
# Refuses to package on failure. The suite pins the chunker against the
# three real assets, the copied modules byte-identical to their masters,
# the ASL inline/copy parity, and every state handler against fakes.
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
# @function build_gateway_tool_package
# ONE zip, TWO handlers (gateway dispatcher + operation-routed workflow),
# flat at the root — gateway, states and lib modules are all TOP-LEVEL at
# runtime.
######################################################################
function build_gateway_tool_package() {
  echo "------------------------------------------------------------------------------"
  echo "Building Lambda package: $PKG_GATEWAY_TOOL"
  echo "------------------------------------------------------------------------------"
  pushd "$SCRIPT_DIR" > /dev/null

  # `zip` exits 0 when only SOME members are missing (micro-workflow-bundle/README.md §C5)
  # — check each member up front. THREE member groups, one zip, all flat.
  # verify_members DIR MEMBERS... checks BOTH directions (review,
  # 2026-09-06): every listed member exists on disk, and every real file
  # in the dir is listed — the zip ships FROM these lists, so an
  # unlisted file silently never ships and the deployed Lambda
  # ImportErrors (or FileNotFoundErrors) at cold start. nullglob keeps
  # an empty dir from iterating literal glob strings.
  function verify_members() {
    local dir="$1"; shift
    local f on_disk base listed
    for f in "$@"; do
      if [ ! -f "$dir/$f" ]; then
        echo "error: $dir/$f not found"
        exit 1
      fi
    done
    shopt -s nullglob
    for on_disk in "$dir"/*; do
      [ -f "$on_disk" ] || continue
      base="${on_disk#"$dir"/}"
      case "$base" in .DS_Store|*.pyc) continue ;; esac
      listed=false
      for f in "$@"; do
        [ "$f" == "$base" ] && listed=true && break
      done
      if [ "$listed" == "false" ]; then
        echo "error: $on_disk exists but is not in $dir's member list —"
        echo "       it would silently not ship; list it (or remove it)"
        exit 1
      fi
    done
    shopt -u nullglob
  }

  local members=(lambda_function.py workflow_identity.py output_contract.json
                 input_examples.json)
  verify_members gateway_tool "${members[@]}"

  # + the lib modules the dispatcher imports (verified live 2026-10-06:
  # the first trim shipped only profiles.py and the deployed gateway
  # ImportError'd on wf_common — the smoke caught it): profiles.py
  # (archetype validation at the door) and wf_common.py (shared
  # gateway plumbing). Flat beside the dispatcher.
  local gw_lib=(lib/profiles.py lib/wf_common.py)
  local gf
  for gf in "${gw_lib[@]}"; do
    if [ ! -f "$gf" ]; then
      echo "error: $gf not found"
      exit 1
    fi
  done
  zip -jq "$BUILD_DIST_DIR/$PKG_GATEWAY_TOOL" \
    "${members[@]/#/gateway_tool/}" \
    "${gw_lib[@]}" || exit 1

  echo ">> Built: $BUILD_DIST_DIR/$PKG_GATEWAY_TOOL"
  popd > /dev/null
}


######################################################################
# @function build_workflow_package
# The workflow Lambda's TREE zip (the family shape; shot-segmentation
# is the canonical copy): top-level lambda_function.py routes by
# `operation` into states/<name>/lambda_function.py; lib/ is the
# shared package; imports are package-style (from lib import x).
######################################################################
function build_workflow_package() {
  echo "------------------------------------------------------------------------------"
  echo "Building Lambda package: $PKG_WORKFLOW"
  echo "------------------------------------------------------------------------------"
  pushd "$SCRIPT_DIR" > /dev/null

  local workflow_members=(lambda_function.py \
                          lib/filter_rank.py \
                          lib/profiles.py \
                          lib/s3io.py \
                          lib/wf_common.py \
                          states/assemble_artwork_candidates/lambda_function.py \
                          states/filter_rank_candidate_shots/lambda_function.py \
                          states/grab_candidate_frames/lambda_function.py \
                          states/rerank_frames_by_vision/lambda_function.py \
                          states/translate_brief_to_plan_amendments/lambda_function.py \
                          states/validate_and_compile_plan/lambda_function.py)
  local f
  for f in "${workflow_members[@]}"; do
    if [ ! -f "$f" ]; then
      echo "error: $f not found"
      exit 1
    fi
  done
  zip -rq "$BUILD_DIST_DIR/$PKG_WORKFLOW" \
    lambda_function.py lib states \
    -x "*__pycache__*" || exit 1

  echo ">> Built: $BUILD_DIST_DIR/$PKG_WORKFLOW"
  popd > /dev/null
}

function verify_sfn_definition() {
  echo "------------------------------------------------------------------------------"
  echo "Verifying ASL inline/copy parity"
  echo "------------------------------------------------------------------------------"
  python3 - <<'PYEOF' || exit 1
import json, sys, yaml
class L(yaml.SafeLoader): pass
def unknown(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode): return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node)
    return loader.construct_mapping(node)
L.add_multi_constructor("!", unknown)
tpl = yaml.load(open("cfnstack.yaml"), Loader=L)
inline = json.loads(tpl["Resources"]["ArtworkSelectionStateMachine"]
                    ["Properties"]["DefinitionString"])
copy = json.load(open("definitions.json"))
if inline != copy:
    sys.exit("error: the template's inline ASL drifted from definitions.json — edit BOTH")
print(">> ASL inline == definitions.json")
PYEOF
}

function build_cloudformation_template() {
  echo "------------------------------------------------------------------------------"
  echo "Building CloudFormation template: $TEMPLATE_NAME"
  echo "------------------------------------------------------------------------------"
  local src="$SCRIPT_DIR/cfnstack.yaml"
  local dst="$TEMPLATE_DIST_DIR/$TEMPLATE_NAME"

  [ ! -f "$src" ] && \
    echo "error: cfnstack.yaml not found at $src" && exit 1

  cp "$src" "$dst"

  sed -i'.bak' -e "s|%%BUCKET_NAME%%|${BUCKET_NAME}|g"             "$dst" || exit 1
  sed -i'.bak' -e "s|%%KEY_PREFIX%%|${KEY_PREFIX}|g"               "$dst" || exit 1
  sed -i'.bak' -e "s|%%VERSION%%|${VERSION}|g"                     "$dst" || exit 1
  sed -i'.bak' -e "s|%%PKG_GATEWAY_TOOL%%|${PKG_GATEWAY_TOOL}|g"   "$dst" || exit 1
  sed -i'.bak' -e "s|%%PKG_WORKFLOW%%|${PKG_WORKFLOW}|g"           "$dst" || exit 1
  rm -f "${dst}.bak"

  # Digit-tolerant guard: a surviving token deploys as a literal S3Key and
  # fails the stack after the build was green.
  if grep -qE '%%[A-Z0-9_]*%%' "$dst"; then
    echo "error: unsubstituted token(s) survive in $dst:"
    grep -nE '%%[A-Z0-9_]*%%' "$dst"
    exit 1
  fi

  cp "$dst" "$BUILD_DIST_DIR/$TEMPLATE_NAME"
  echo ">> Built: $dst"
}

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
  local region_flag=""
  [ "$region" != "us-east-1" ] && region_flag="--region ${region}"

  echo ""
  echo "== Deploying to ${dest} (${region}) =="
  aws s3 cp "$BUILD_DIST_DIR" "$dest" --recursive --acl ${ACL_SETTING} $region_flag
  aws s3 cp "$TEMPLATE_DIST_DIR/$TEMPLATE_NAME" \
    "s3://${bucket}/${KEY_PREFIX}/${TEMPLATE_NAME}" \
    --acl ${ACL_SETTING} $region_flag
  echo "== Deploy to ${bucket} [COMPLETED] =="

  local domain="s3.amazonaws.com"
  [ "$region" != "us-east-1" ] && domain="s3.${region}.amazonaws.com"
  local url="https://${bucket}.${domain}/${KEY_PREFIX}/${TEMPLATE_NAME}"
  echo ""
  echo "========================================"
  echo "  Template URL:"
  echo "  $url"
  echo ""
  echo "  Parameters:"
  echo "    ResourcePrefix  names the two Lambdas and log groups (no default)"
  echo "    ProxyBucket     sibling outputs in, artwork-selection outputs out (no default)
    BedrockInferenceGatewayLambdaArn / PromptRegistryGatewayLambdaArn
                    the siblings' gateway ARNs (required; deploy those first)"
  echo "========================================"
}

function on_complete() {
  echo ""
  echo "======================================================================"
  echo " Build & Deploy Complete — ${SOLUTION} ${VERSION}"
  echo "======================================================================"
  echo "  PKG_GATEWAY_TOOL : $PKG_GATEWAY_TOOL"
  echo "  PKG_WORKFLOW     : $PKG_WORKFLOW"
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
verify_sfn_definition
build_gateway_tool_package
build_workflow_package
build_cloudformation_template
assert_internal_docs_excluded

if [ "$BUILD_ONLY" == "true" ]; then
  on_complete
  exit 0
fi

if [ "$SINGLE_REGION" == "true" ]; then
  deploy_to_bucket "${BUCKET_NAME}"
else
  BUCKET_REGION=$(aws s3api get-bucket-location \
    --bucket "${BUCKET_NAME}" \
    --expected-bucket-owner "${ACCOUNTID}" \
    | jq -r '.LocationConstraint // "us-east-1"')
  [ "$BUCKET_REGION" == "null" ] && BUCKET_REGION="us-east-1"
  deploy_to_bucket "${BUCKET_NAME}-${BUCKET_REGION}"
fi

on_complete
