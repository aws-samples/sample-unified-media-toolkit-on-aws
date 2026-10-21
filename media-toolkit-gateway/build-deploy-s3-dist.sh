#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Run this script from media-toolkit-gateway/
#
# bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKFLOW_NAME="media-toolkit-gateway"
TEMPLATE_NAME="${WORKFLOW_NAME}-stack.template"

TEMPLATE_DIST_DIR="$SCRIPT_DIR/dist/global-s3-assets"
BUILD_DIST_DIR="$SCRIPT_DIR/dist/regional-s3-assets"

#
# @function usage
#
function usage() {
  echo -e "
------------------------------------------------------------------------------

Build and deploy the AgentCore Gateway + workflow agent tools CloudFormation stack.
Run from media-toolkit-gateway/

This stack has no Docker image. Its only artifact is the gateway_tool zip, which
carries the four Lambda handlers (download_url, ingest_list, secrets_populator,
kit_publisher) plus the nested client_kit.zip.
It must be deployed AFTER the three micro-workflow stacks it wraps:
  - media-processing
  - audiostem-separation-on-aws
  - speech-diarization-on-aws

------------------------------------------------------------------------------
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]

where
  --bucket BUCKET     S3 bucket name to deploy to. Without --single-region this is
                      treated as a basename and the region is appended, e.g. BUCKET-us-east-1.

  --solution SOLUTION [optional] S3 key prefix namespace. Defaults to 'unified-media-toolkit'.

  --version VERSION   [optional] Stack version. Defaults to contents of .version file.

  --single-region     [optional] Deploy to exactly the bucket specified in --bucket (no region suffix).

  --build-only        [optional] Build packages and the template only; skip deploy to S3.

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
    -s|--solution)
      SOLUTION="$2"
      shift; shift ;;
    -v|--version)
      VERSION="$2"
      shift; shift ;;
    -r|--single-region)
      SINGLE_REGION=true
      shift ;;
    --build-only)
      BUILD_ONLY=true
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
[ "$(which jq)" == "" ] && \
  echo "error: jq is required" && exit 1

[ -z "$BUILD_ONLY" ] && BUILD_ONLY=false

if [ "$BUILD_ONLY" == "false" ]; then
  [ "$(which aws)" == "" ] && \
    echo "error: AWS CLI is required" && exit 1
  [ -z "$BUCKET_NAME" ] && \
    echo "error: missing --bucket parameter" && usage && exit 1
fi
[ -z "$BUCKET_NAME" ] && BUCKET_NAME="BUCKET_NAME_NOT_SET"

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

######################################################################
# Derived variables
######################################################################
PKG_GATEWAY_TOOL="media-toolkit-gateway-tool-${VERSION}.zip"
KEY_PREFIX="${SOLUTION}/${VERSION}/${WORKFLOW_NAME}"

echo "======================================================================"
if [ "$BUILD_ONLY" == "true" ]; then
  echo " AgentCore Gateway / Workflow Agent Tools — Build Only"
else
  echo " AgentCore Gateway / Workflow Agent Tools — Build & Deploy"
fi
echo "======================================================================"
echo "  solution  : $SOLUTION"
echo "  version   : $VERSION"
echo "  bucket    : $BUCKET_NAME"
echo "  key prefix: $KEY_PREFIX"
echo "  package   : $PKG_GATEWAY_TOOL"
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

  find "$SCRIPT_DIR" -name '.DS_Store' -type f -delete
}

######################################################################
# @function build_gateway_tool_zip
# Packages gateway_tool/*.py as the single Lambda artifact for this stack:
#   download_url.py       — media-outputs___get_download_url tool handler
#   ingest_list.py        — media-ingest___list_files tool handler
#   upload_tool.py        — media-uploads___* presigned-upload tool handlers
#   secrets_populator.py  — Secrets Manager custom resource handler
# All are referenced by cfnstack.yaml via Code.S3Bucket/S3Key. A zip is
# required rather than merely tidier because CFN names inline ZipFile code
# "index" — both handlers would have to be index.handler. The zip must stay
# flat (zip -j) so Handler: <module>.handler resolves.
######################################################################
######################################################################
# @function build_client_kit_zip
# Assembles the customer-facing client kit — skills/, the two scripts
# and the README — as gateway_tool/client_kit.zip so the kit ships
# INSIDE the Lambda package and kit_publisher.py can write it to the
# proxy bucket at deploy time (no dist-bucket access needed by the
# customer's account). Assembled at build time, never committed: one
# source of truth in the repo. NOTE the standing S3Key convention
# still applies — kit content ships inside the Lambda zip, so a
# skills/README/script change deploys only with a .version bump,
# exactly like a handler change.
######################################################################
function build_client_kit_zip() {
  echo "------------------------------------------------------------------------------"
  echo "Building client kit: gateway_tool/client_kit.zip"
  echo "------------------------------------------------------------------------------"
  pushd "$SCRIPT_DIR" > /dev/null
  local f
  for f in README.md scripts/get_gateway_token.sh scripts/configure.sh scripts/get_gateway_token.ps1 scripts/configure.ps1; do
    [ ! -f "$f" ] && echo "error: $f not found" && exit 1
  done
  [ ! -d skills ] && echo "error: skills/ not found" && exit 1
  rm -f gateway_tool/client_kit.zip
  find skills -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
  zip -rq gateway_tool/client_kit.zip \
    README.md \
    skills \
    scripts/get_gateway_token.sh \
    scripts/configure.sh \
    scripts/get_gateway_token.ps1 scripts/configure.ps1 \
    -x '*.DS_Store' || exit 1
  # the routing skill's references/ is a SYMLINK to ../../prompts; zip
  # dereferences it so the kit carries real copies of the per-workflow
  # references. A broken link would zip to nothing and ship a router
  # whose every reference 404s — check one member landed.
  if ! unzip -l gateway_tool/client_kit.zip \
      | grep -q 'skills/video-processing/references/media_processing_agent.md'; then
    echo "error: client kit is missing the dereferenced routing references" \
         "(skills/video-processing/references symlink broken?)" && exit 1
  fi
  echo ">> Built: gateway_tool/client_kit.zip ($(du -h gateway_tool/client_kit.zip | cut -f1))"
  popd > /dev/null
}

function build_gateway_tool_zip() {
  echo "------------------------------------------------------------------------------"
  echo "Building gateway tool zip: $PKG_GATEWAY_TOOL"
  echo "------------------------------------------------------------------------------"
  local members=(download_url.py ingest_list.py upload_tool.py \
                 secrets_populator.py kit_publisher.py client_kit.zip)

  pushd "$SCRIPT_DIR/gateway_tool" > /dev/null

  # zip exits 0 when only SOME members are missing (it warns and packages the
  # rest), so a missing handler would ship as a silently broken Lambda.
  # Check each member up front instead of trusting the exit code.
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
# @function validate_tool_name_lengths
# Bedrock rejects MCP tool names ({target}___{tool}) longer than 64
# characters — an oversize name fails the entire ConverseStream.
# Extract every target Name and tool Name from cfnstack.yaml and fail
# the build on any cross-product combination over the limit.  Also
# fail if extraction comes back empty (regex rot on template layout).
######################################################################
function validate_tool_name_lengths() {
  echo "------------------------------------------------------------------------------"
  echo "Validating MCP tool name lengths (limit 64 chars)"
  echo "------------------------------------------------------------------------------"
  python3 - "$SCRIPT_DIR/cfnstack.yaml" <<'PYEOF'
import re, sys

text = open(sys.argv[1]).read()

# Target Description hard cap (AgentCore: maxLength 200). Checked here because
# an oversize one fails at CREATE/UPDATE after a green build — measured
# 2026-08-15 when a 220-char description rolled back the whole m2cv6
# gateway-stack update, cancelling every sibling target's update with it.
too_long = [(m.group(1), len(m.group(1)))
            for m in re.finditer(r'^ {12}Description: "([^"]+)"$', text, re.M)
            if len(m.group(1)) > 200]
if too_long:
    for desc, n in too_long:
        print(f"FAIL: GatewayTarget Description is {n} chars (max 200): "
              f"{desc[:60]}...")
    sys.exit(1)

# Target names: "Name: <kebab-case>" at the GatewayTarget property indent
targets = re.findall(r'^ {12}Name:\s*([a-z0-9-]+)\s*$', text, re.M)
# Tool names: "- Name: <snake_case>" inside InlinePayload lists
tools = re.findall(r'^\s+-\s+Name:\s*([a-z0-9_]+)\s*$', text, re.M)

if not targets or not tools:
    print(f"FAIL: name extraction empty (targets={targets}, tools={tools}) — "
          "template layout changed; update the regexes in "
          "validate_tool_name_lengths")
    sys.exit(1)

# Conservative cross-product: every target x every tool errs safe.
bad = [f"{t}___{v}" for t in sorted(set(targets)) for v in sorted(set(tools))
       if len(f"{t}___{v}") > 64]
if bad:
    print("FAIL: MCP tool name(s) exceed the 64-char Bedrock limit:")
    for n in bad:
        print(f"  {len(n):3d}  {n}")
    sys.exit(1)

longest = max((f"{t}___{v}" for t in set(targets) for v in set(tools)), key=len)
print(f"OK: {len(set(targets))} targets x {len(set(tools))} tools; "
      f"longest = {longest} ({len(longest)}/64)")
PYEOF
  [ $? -ne 0 ] && echo "error: tool name validation failed" && exit 1
  return 0
}

######################################################################
# @function build_cloudformation_template
# Copy cfnstack.yaml to TEMPLATE_DIST_DIR, substitute tokens, rename to
# .template, and also place a copy in BUILD_DIST_DIR (regional).
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

  sed -i'.bak' -e "s|%%BUCKET_NAME%%|${BUCKET_NAME}|g"             "$dst" || exit 1
  sed -i'.bak' -e "s|%%KEY_PREFIX%%|${KEY_PREFIX}|g"               "$dst" || exit 1
  sed -i'.bak' -e "s|%%VERSION%%|${VERSION}|g"                     "$dst" || exit 1
  sed -i'.bak' -e "s|%%PKG_GATEWAY_TOOL%%|${PKG_GATEWAY_TOOL}|g"   "$dst" || exit 1

  rm -f "${dst}.bak"

  # A leftover %%TOKEN%% deploys as a literal string — an S3Key pointing at
  # "%%PKG_GATEWAY_TOOL%%" fails the stack minutes in. Fail here instead.
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
# Upload regional-s3-assets (source zip + template copy) to the given bucket.
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
  echo "  https://console.aws.amazon.com/cloudformation/home?region=${region}#/stacks/quickcreate?templateURL=${url}&stackName=media-toolkit-gateway"
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
  echo "  PKG_GATEWAY_TOOL : $PKG_GATEWAY_TOOL"
  echo "  TEMPLATE         : $TEMPLATE_NAME"
  echo "======================================================================"
}

######################################################################
# @function assert_internal_docs_excluded
# GOTCHAS.md and the harness run manifests are INTERNAL ONLY -- they carry
# the customer, programme and asset identities that the code and README.md
# deliberately refer to only by generic corpus label. Every
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

  # The client kit is the MOST exposed artifact (published to every
  # deployment's proxy bucket) and rides NESTED inside the gateway
  # zip, where the member-name pass above cannot see into it. It is
  # also the one zip built with a recursive `zip -rq skills` rather
  # than an allowlist, so a stray file beside a SKILL.md would ship.
  # Check its members by NAME (internal-doc patterns) and its text
  # content for deployment values.
  if [ -f "$SCRIPT_DIR/gateway_tool/client_kit.zip" ]; then
    while IFS= read -r member; do
      case "$member" in
        */) continue ;;
        GOTCHAS.md|*/GOTCHAS.md|*.bak.*|*.state.json|*/runs/*)
          echo "  [LEAK] client_kit.zip: $member"
          violations=$((violations+1))
          ;;
      esac
    done < <(unzip -Z1 "$SCRIPT_DIR/gateway_tool/client_kit.zip")
    # Content scan. The committed patterns are STRUCTURAL on purpose —
    # naming the actual account id / bucket / usernames here would BE
    # the leak this check exists to prevent (this script is committed
    # and ships in the repo-root source package). Structural: any 12-digit AWS
    # account id that is not a documentation-fake, local /Users/ paths,
    # and stack-suffixed bucket shapes. The sharp deployment-specific
    # literals live in the UNTRACKED .scrub-patterns.local (gitignored,
    # one regex per line) — present on maintainer machines, absent and
    # silently skipped on customer checkouts.
    local kit_text kit_leak=0
    kit_text=$(unzip -p "$SCRIPT_DIR/gateway_tool/client_kit.zip" '*.md' '*.sh' '*.html' 2>/dev/null)
    # any 12-digit account id that is not a documentation fake
    if echo "$kit_text" | grep -oE '(^|[^0-9])[0-9]{12}([^0-9]|$)' \
        | grep -qvE '111122223333|123456789012|000000000000'; then
      kit_leak=1
    fi
    # local user paths and stack-suffixed bucket/resource shapes
    if echo "$kit_text" \
        | grep -qE '/Users/[a-zA-Z]|so0050-[0-9a-f]{12}|[a-z0-9]+-[0-9a-f]{12}-[0-9]{12}-'; then
      kit_leak=1
    fi
    if [ "$kit_leak" -ne 0 ]; then
      echo "  [LEAK] client_kit.zip: deployment-specific value inside kit content"
      violations=$((violations+1))
    fi
    if [ -f "$SCRIPT_DIR/.scrub-patterns.local" ]; then
      while IFS= read -r pat; do
        [ -z "$pat" ] && continue
        if echo "$kit_text" | grep -qE "$pat"; then
          echo "  [LEAK] client_kit.zip: matches local scrub pattern"
          violations=$((violations+1))
        fi
      done < "$SCRIPT_DIR/.scrub-patterns.local"
    fi
  fi

  if [ $violations -ne 0 ] ; then
    echo ""
    echo "error: ${violations} internal file(s) would ship in the source packages."
    echo "       Remove them from the zip member list above and re-run."
    exit 1
  fi

  echo "  no internal docs in any artifact (client kit inspected inside)"
}

######################################################################
# Main
######################################################################
clean_start
validate_tool_name_lengths
build_client_kit_zip
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
  BUCKET_REGION=$(aws s3api get-bucket-location \
    --bucket "${BUCKET_NAME}" \
    --expected-bucket-owner "${ACCOUNTID}" \
    | jq -r '.LocationConstraint // "us-east-1"')
  [ "$BUCKET_REGION" == "null" ] && BUCKET_REGION="us-east-1"

  deploy_to_bucket "${BUCKET_NAME}-${BUCKET_REGION}"
fi

on_complete
