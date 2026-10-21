#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Runs from any working directory — every path, including the git provenance
# check, is derived from this file's own location, not the cwd.
#
# Builds and stages EVERY micro-workflow child plus this bundle's root
# template, so one create-stack brings up the whole set.
#
# bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION]
#                               [--single-region] [--allow-dirty]
#                               [--skip-children] [--acl ACL] [--build-only]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKFLOW_NAME="micro-workflow-bundle"
TEMPLATE_NAME="${WORKFLOW_NAME}-stack.template"

TEMPLATE_DIST_DIR="$SCRIPT_DIR/dist/global-s3-assets"
BUILD_DIST_DIR="$SCRIPT_DIR/dist/regional-s3-assets"

# dir|keyprefix-token-suffix
#
# Order is DEPLOY order (README.md §P3): media-processing first so that a break in
# the one child three others depend on surfaces before they are built;
# media-toolkit-gateway last because it consumes every other child's gateway-tool
# ARNs (more references than children — hybrid-search supplies two). Build
# order does not have to match deploy order — the children are independent at
# build time — but matching means the most consequential failure arrives first.
#
# The logical resource names in cfnstack.yaml are derived from these suffixes
# and are permanent API: renaming one deletes and recreates that child stack
# (and its DynamoDB tables). Do not rename to tidy them up.
WORKFLOWS=(
  "media-processing-on-aws|MEDIA_PROCESSING"
  "audiostem-separation-on-aws|AUDIOSTEM_SEPARATION"
  "speech-enhancement-on-aws|SPEECH_ENHANCEMENT"
  "speech-diarization-on-aws|SPEECH_DIARIZATION"
  "shot-segmentation-on-aws|SHOT_SEGMENTATION"
  "face-recognition-on-aws|FACE_RECOGNITION"
  "hybrid-search-on-aws|HYBRID_SEARCH"
  "tts-on-aws|TTS"
  "bedrock-inference-on-aws|BEDROCK_INFERENCE"
  "prompt-registry-on-aws|PROMPT_REGISTRY"
  "conversation-analysis-on-aws|CONVERSATION_ANALYSIS"
  "shot-metadata-enrichment-on-aws|SHOT_METADATA_ENRICHMENT"
  "frame-inspection-on-aws|FRAME_INSPECTION"
  "program-structure-analysis-on-aws|PROGRAM_STRUCTURE_ANALYSIS"
  "audio-description-on-aws|AUDIO_DESCRIPTION"
  "cast-resolution-on-aws|CAST_RESOLUTION"
  "artwork-selection-on-aws|ARTWORK_SELECTION"
  "media-toolkit-gateway|M2C_AGENT_TOOLS"
)

#
# @function usage
#
function usage() {
  echo -e "
------------------------------------------------------------------------------

Build and stage the Unified Media Toolkit micro-workflow bundle: every child stack
plus the root template that nests them. Runs from any directory.

------------------------------------------------------------------------------
bash ./build-deploy-s3-dist.sh --bucket BUCKET [options]

where
  --bucket BUCKET     S3 bucket name to stage to. Without --single-region this is
                      treated as a basename and the region is appended, e.g.
                      BUCKET-us-east-1.

  --solution SOLUTION [optional] S3 key prefix namespace. Defaults to 'unified-media-toolkit'.

  --version VERSION   [optional] Forces this version on EVERY child and on
                      the root template. Omit to let each child read its own
                      .version — which is what lets an untouched child keep its S3
                      key so CloudFormation sees no ImageTag change and CodeBuild
                      SKIPS its image rebuild. Passing --version rebuilds every
                      container image (~60-90 min each, in parallel) plus tts's
                      model seeding, in exchange for one coherent snapshot.

  --single-region     [optional] Stage to exactly the bucket given by --bucket
                      (no region suffix).

  --allow-dirty       [optional] Permit building from a dirty working tree.
                      Required in that case, because the artifact a dirty build
                      produces is indistinguishable from a clean one by its S3 key
                      alone. Commit, or pass this to proceed deliberately.

  --skip-children     [optional] Build and stage the root template ONLY. For
                      iterating on root wiring without re-zipping and re-uploading
                      every child. The composed child template keys must already
                      be in S3 — this script verifies they are.

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
    --allow-dirty)
      ALLOW_DIRTY=true
      shift ;;
    --skip-children)
      SKIP_CHILDREN=true
      shift ;;
    -o|--build-only)
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

[ "$(which aws)" == "" ] && \
  echo "error: AWS CLI is required" && exit 1

[ "$(which git)" == "" ] && \
  echo "error: git is required" && exit 1

[ -z "$BUILD_ONLY" ] && BUILD_ONLY=false
[ "$BUILD_ONLY" != "true" ] && [ -z "$BUCKET_NAME" ] && \
  echo "error: missing --bucket parameter (or pass --build-only)" && usage && exit 1

# The bundle's own .version stamps ONLY the root template. It is independent of
# the children's own versions.
BUNDLE_VERSION="$VERSION"
[ -z "$BUNDLE_VERSION" ] && \
  BUNDLE_VERSION=$(cat "$SCRIPT_DIR/.version" 2>/dev/null)

[ -z "$BUNDLE_VERSION" ] && \
  echo "error: cannot determine version — use --version or create a .version file" && \
  usage && exit 1

[ -z "$SOLUTION" ] && \
  SOLUTION="unified-media-toolkit"

[ -z "$SINGLE_REGION" ] && \
  SINGLE_REGION=false

[ -z "$ACL_SETTING" ] && \
  ACL_SETTING="bucket-owner-full-control"

[ -z "$SKIP_CHILDREN" ] && \
  SKIP_CHILDREN=false

if [ "$BUILD_ONLY" == "true" ]; then
  # ── local pipeline only: no STS, no bucket checks, no uploads. The
  # placeholder bucket makes an accidentally deployed artifact fail
  # LOUDLY instead of resolving to someone's real bucket. ──
  ACCOUNTID="000000000000"
  STAGING_BUCKET="BUCKET_NAME_NOT_SET"
  BUCKET_REGION="us-east-1"
  KEY_PREFIX="${SOLUTION}/${BUNDLE_VERSION}/${WORKFLOW_NAME}"
  REGION_FLAG=""
else
  ACCOUNTID=$(aws sts get-caller-identity | jq -r .Account)
  [ -z "$ACCOUNTID" ] && \
    echo "error: failed to get AWS Account ID" && exit 1

  aws s3api get-bucket-location \
    --bucket "${BUCKET_NAME}" \
    --expected-bucket-owner "${ACCOUNTID}" > /dev/null 2>&1
  [ $? -ne 0 ] && \
    echo "error: bucket '${BUCKET_NAME}' does not exist or does not belong to account '${ACCOUNTID}'" && \
    exit 1

  ######################################################################
  # Resolve the staging bucket ONCE
  #
  # This is the single most load-bearing decision in this script, and it exists to
  # route around a latent inconsistency shared by every child:
  #
  #   Every child substitutes %%BUCKET_NAME%% with the bucket BASENAME, but when
  #   --single-region is absent it uploads its artifacts to
  #   "${BUCKET_NAME}-${BUCKET_REGION}". So a child staged WITHOUT --single-region
  #   bakes a CodeBuild "Source Bucket" mapping pointing at a bucket that holds
  #   none of its artifacts. The children are only self-consistent under
  #   --single-region.
  #
  # So: resolve the regional name here, then pass it to every child as
  # "--bucket <resolved> --single-region". The child's sed and the child's upload
  # then agree, and this bundle's %%BUCKET_NAME%% is the same resolved value —
  # which is also why cfnstack.yaml has no bSingleRegion condition appending
  # "-${AWS::Region}" the way deployment/media2cloud.yaml does. One resolved
  # bucket name, written everywhere, with no second chance to disagree.
  ######################################################################
  BUCKET_REGION=$(aws s3api get-bucket-location \
    --bucket "${BUCKET_NAME}" \
    --expected-bucket-owner "${ACCOUNTID}" \
    | jq -r '.LocationConstraint // "us-east-1"')
  [ "$BUCKET_REGION" == "null" ] && BUCKET_REGION="us-east-1"

  if [ "$SINGLE_REGION" == "true" ]; then
    STAGING_BUCKET="${BUCKET_NAME}"
  else
    STAGING_BUCKET="${BUCKET_NAME}-${BUCKET_REGION}"
  fi

  aws s3api get-bucket-location \
    --bucket "${STAGING_BUCKET}" \
    --expected-bucket-owner "${ACCOUNTID}" > /dev/null 2>&1
  [ $? -ne 0 ] && \
    echo "error: staging bucket '${STAGING_BUCKET}' does not exist or does not belong to account '${ACCOUNTID}'" && \
    echo "       (derived from --bucket '${BUCKET_NAME}'; pass --single-region to use it verbatim)" && \
    exit 1

  KEY_PREFIX="${SOLUTION}/${BUNDLE_VERSION}/${WORKFLOW_NAME}"

  REGION_FLAG=""
  [ "$BUCKET_REGION" != "us-east-1" ] && REGION_FLAG="--region ${BUCKET_REGION}"
fi

######################################################################
# Guard a dirty build
#
# Always builds from the working tree. That is the normal dev loop and must stay
# possible — but a dirty build has to be STATED, because the S3 key an artifact
# lands under cannot distinguish a dirty build from a clean one. A stale .version
# plus a dirty tree produces a key that actively lies about its contents.
#
# --allow-dirty is the ONLY escape, which makes this guard load-bearing: it must
# never pass by accident. `git status --porcelain` writes nothing to stdout when
# it FAILS (not a repo, bad pathspec) as well as when the tree is clean, so its
# empty output alone is not evidence of cleanliness — check the exit code too, or
# a build outside a git checkout silently reports itself clean.
#
# Two repo topologies are in use and the guard answers on both:
#   MONOREPO — the repo root holds every child. One status call over
#   DOCKER_ROOT answers for everything; BUILD_SHA is that repo's HEAD.
#   PER-CHILD REPOS — the parent is not a repo and each child is its own, so
#   the whole-tree call is 'fatal: outside repository' BY CONSTRUCTION, not
#   evidence of anything. Fall back to asking every child repo individually:
#   a clean multi-repo checkout then builds WITHOUT --allow-dirty, dirty
#   files are named with their child prefixed, and the summary prints a
#   per-child commit column instead of 'unknown'.
######################################################################
# DOCKER_ROOT is the directory holding the children.
DOCKER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DIRTY=$(git -C "$SCRIPT_DIR" status --porcelain -- "$DOCKER_ROOT" 2>/dev/null)
GIT_STATUS_RC=$?
BUILD_SHA=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null)
GIT_MODE="monorepo"
CHILD_COMMITS=()

if [ $GIT_STATUS_RC -ne 0 ]; then
  # Whole-tree status is unanswerable — per-child fallback. Every child must
  # answer for the verdict to count: one non-repo child and we are back to
  # "cannot determine", because that child's code is in the artifact too.
  GIT_MODE="per-child"
  GIT_STATUS_RC=0
  GIT_FAILED_DIR=""
  DIRTY=""

  # The bundle's own repo (the root template and scripts are artifacts too).
  cdirty=$(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null)
  crc=$?
  if [ $crc -ne 0 ] || [ -z "$BUILD_SHA" ]; then
    GIT_STATUS_RC=128
    GIT_FAILED_DIR="$(basename "$SCRIPT_DIR")"
  elif [ -n "$cdirty" ]; then
    BUILD_SHA="${BUILD_SHA}-dirty"
    DIRTY="${DIRTY}$(echo "$cdirty" | sed "s|^\(.. \)|\1$(basename "$SCRIPT_DIR")/|")"$'\n'
  fi

  for entry in "${WORKFLOWS[@]}"; do
    [ $GIT_STATUS_RC -ne 0 ] && break
    cdir="${entry%%|*}"
    cpath="${DOCKER_ROOT}/${cdir}"
    csha=$(git -C "$cpath" rev-parse HEAD 2>/dev/null)
    cdirty=$(git -C "$cpath" status --porcelain 2>/dev/null)
    crc=$?
    if [ $crc -ne 0 ] || [ -z "$csha" ]; then
      GIT_STATUS_RC=128
      GIT_FAILED_DIR="$cdir"
      break
    fi
    if [ -n "$cdirty" ]; then
      csha="${csha}-dirty"
      DIRTY="${DIRTY}$(echo "$cdirty" | sed "s|^\(.. \)|\1${cdir}/|")"$'\n'
    fi
    CHILD_COMMITS+=("${cdir}|${csha}")
  done
  DIRTY="${DIRTY%$'\n'}"
fi

if [ $GIT_STATUS_RC -ne 0 ]; then
  if [ "$ALLOW_DIRTY" != "true" ]; then
    echo "------------------------------------------------------------------------------"
    echo "error: cannot determine whether the tree is clean — 'git status' failed"
    echo "       for ${GIT_FAILED_DIR:-$DOCKER_ROOT}."
    echo ""
    echo "       Neither the whole tree nor every child directory is a git checkout."
    echo "       An empty 'git status --porcelain' means BOTH 'clean' and 'the command"
    echo "       failed', so this cannot be treated as clean. Either build from a git"
    echo "       checkout, or pass --allow-dirty to proceed deliberately."
    echo "------------------------------------------------------------------------------"
    exit 1
  fi
  BUILD_SHA="unknown"
  DIRTY=""
  CHILD_COMMITS=()
elif [ -n "$DIRTY" ] && [ "$ALLOW_DIRTY" != "true" ]; then
  echo "------------------------------------------------------------------------------"
  echo "error: uncommitted changes in the repo and --allow-dirty was not given."
  echo ""
  echo "$DIRTY" | sed 's/^/       /'
  echo ""
  echo "       The S3 key these artifacts land under cannot tell a dirty build from"
  echo "       a clean one. Either commit, or pass --allow-dirty to proceed"
  echo "       deliberately."
  echo "------------------------------------------------------------------------------"
  exit 1
fi

# Per-child mode already suffixed BUILD_SHA (the bundle repo's own state);
# a dirty CHILD must not mislabel the bundle's SHA.
[ "$GIT_MODE" = "monorepo" ] && [ -n "$DIRTY" ] && BUILD_SHA="${BUILD_SHA}-dirty"
[ -z "$BUILD_SHA" ] && BUILD_SHA="unknown"

######################################################################
# Resolve every child's version BEFORE building anything
#
# Read each .version exactly once here and pass it to the child explicitly as
# --version, so the value used for this bundle's %%KEYPREFIX_*%% mapping and the
# value the child uses for its own KEY_PREFIX are the same read. Letting both
# sides read .version independently allows the two reads to straddle an edit and
# produce a root template pointing at a key that was never staged.
######################################################################
declare -a RESOLVED_DIRS=()
declare -a RESOLVED_SUFFIXES=()
declare -a RESOLVED_VERSIONS=()
declare -a RESOLVED_KEYS=()

for entry in "${WORKFLOWS[@]}"; do
  IFS='|' read -r cdir csuffix <<< "$entry"
  cpath="${DOCKER_ROOT}/${cdir}"
  cscript="${cpath}/build-deploy-s3-dist.sh"

  # Check explicitly rather than trusting a later command's exit code. The
  # lesson generalizes from `zip`, which exits 0 when only SOME members are
  # missing: "the command succeeded" is not evidence that all inputs existed.
  [ ! -d "$cpath" ] && \
    echo "error: child directory not found: ${cpath}" && exit 1
  [ ! -f "$cscript" ] && \
    echo "error: child build script not found: ${cscript}" && exit 1
  [ ! -f "${cpath}/cfnstack.yaml" ] && \
    echo "error: child template not found: ${cpath}/cfnstack.yaml" && exit 1

  # Read WORKFLOW_NAME and TEMPLATE_NAME FROM THE CHILD'S SCRIPT rather than
  # deriving them from the directory name. Both follow a convention today
  # (WORKFLOW_NAME is the directory minus "-on-aws"; TEMPLATE_NAME is
  # "${WORKFLOW_NAME}-stack.template") but nothing enforces either, and tts
  # diverged on TEMPLATE_NAME through alpha.0803c. Reading the real values means
  # a future divergence fails this build's head-object check instead of the
  # root stack's TemplateURL ~90 minutes into a deploy.
  cwf=$(grep -m1 '^WORKFLOW_NAME=' "$cscript" | sed 's/^WORKFLOW_NAME=//; s/^"//; s/"$//')
  [ -z "$cwf" ] && \
    echo "error: cannot parse WORKFLOW_NAME from ${cscript}" && exit 1

  ctpl=$(grep -m1 '^TEMPLATE_NAME=' "$cscript" | sed 's/^TEMPLATE_NAME=//; s/^"//; s/"$//')
  [ -z "$ctpl" ] && \
    echo "error: cannot parse TEMPLATE_NAME from ${cscript}" && exit 1
  # Resolve the one variable reference the convention uses.
  ctpl="${ctpl//\$\{WORKFLOW_NAME\}/$cwf}"
  case "$ctpl" in
    *'$'*)
      echo "error: unresolved variable in TEMPLATE_NAME from ${cscript}: ${ctpl}"
      exit 1 ;;
  esac

  cver="$VERSION"
  [ -z "$cver" ] && cver=$(cat "${cpath}/.version" 2>/dev/null)
  [ -z "$cver" ] && \
    echo "error: cannot determine version for ${cdir} (no --version and no .version)" && exit 1

  RESOLVED_DIRS+=("$cdir")
  RESOLVED_SUFFIXES+=("$csuffix")
  RESOLVED_VERSIONS+=("$cver")
  RESOLVED_KEYS+=("${SOLUTION}/${cver}/${cwf}/${ctpl}")
done

######################################################################
# Print the resolved plan before doing any work
#
# Per-child versioning means "which version did this bundle actually deploy?"
# has one answer per child. One table up front is the whole answer.
######################################################################
echo "======================================================================"
echo " Unified Media Toolkit — Micro-Workflow Bundle Build & Stage"
echo "======================================================================"
echo "  solution       : $SOLUTION"
echo "  bundle version : $BUNDLE_VERSION"
echo "  bucket (given) : $BUCKET_NAME"
echo "  bucket (staged): $STAGING_BUCKET   [region ${BUCKET_REGION}]"
echo "  key prefix     : $KEY_PREFIX"
echo "  template       : $TEMPLATE_NAME"
echo "  commit         : $BUILD_SHA"
echo "  source         : working tree$([ "$ALLOW_DIRTY" == "true" ] && echo ' (--allow-dirty)')"
if [ -z "$VERSION" ]; then
  echo "  child versions : per-child .version (untouched children skip image rebuild)"
else
  echo "  child versions : FORCED to ${VERSION} — all 7 container images will rebuild"
fi
echo "----------------------------------------------------------------------"
printf "  %-30s %-16s %s\n" "CHILD" "VERSION" "TEMPLATE KEY"
for i in "${!RESOLVED_DIRS[@]}"; do
  printf "  %-30s %-16s %s\n" \
    "${RESOLVED_DIRS[$i]}" "${RESOLVED_VERSIONS[$i]}" "${RESOLVED_KEYS[$i]}"
done
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

  # The kit is BUILT by the media-toolkit-gateway child at ITS resolved
  # version, which equals BUNDLE_VERSION only under a --version override —
  # per-child versioning is the design, so the root script must name the
  # staged kit by the gateway's version, not the bundle's.
  GATEWAY_VERSION=""
  local gi
  for gi in "${!RESOLVED_DIRS[@]}"; do
    [ "${RESOLVED_DIRS[$gi]}" == "media-toolkit-gateway" ] && \
      GATEWAY_VERSION="${RESOLVED_VERSIONS[$gi]}"
  done

  # Hand the resolved staging values to the root build script, which merges
  # every source zip into the single package. One resolution, one writer —
  # the root never re-derives the bucket (see "Resolve the staging bucket
  # ONCE" above).
  cat > "$SCRIPT_DIR/dist/.staging.env" <<EOF
STAGING_BUCKET="$STAGING_BUCKET"
BUCKET_REGION="$BUCKET_REGION"
SOLUTION="$SOLUTION"
KEY_PREFIX="$KEY_PREFIX"
REGION_FLAG="$REGION_FLAG"
ACL_SETTING="$ACL_SETTING"
BUNDLE_VERSION="$BUNDLE_VERSION"
GATEWAY_VERSION="$GATEWAY_VERSION"
BUILD_ONLY="$BUILD_ONLY"
EOF

  find "$SCRIPT_DIR" -name '.DS_Store' -type f -delete
}

######################################################################
# @function build_children
# One loop over the WORKFLOWS table — deliberately not one
# build_<workflow>() function per child. The children's CLI is uniform (the
# same six flags everywhere), which is the precondition that makes a loop
# correct rather than merely shorter: a function per child would be one place
# per child to forget a flag.
######################################################################
function build_children() {
  local total="${#RESOLVED_DIRS[@]}"
  local i
  for i in "${!RESOLVED_DIRS[@]}"; do
    local cdir="${RESOLVED_DIRS[$i]}"
    local cver="${RESOLVED_VERSIONS[$i]}"
    local cpath="${DOCKER_ROOT}/${cdir}"

    echo ""
    echo "=============================================================================="
    echo " [$((i+1))/${total}] ${cdir} @ ${cver}"
    echo "=============================================================================="

    # --single-region is passed ALWAYS, with the already-resolved regional
    # bucket name. See the "Resolve the staging bucket ONCE" block: it is what
    # makes each child's %%BUCKET_NAME%% substitution agree with the bucket it
    # actually uploads to.
    (
      cd "$cpath" || exit 1
      if [ "$BUILD_ONLY" == "true" ]; then
        bash ./build-deploy-s3-dist.sh \
          --build-only \
          --solution "$SOLUTION" \
          --version "$cver"
      else
        bash ./build-deploy-s3-dist.sh \
          --bucket "$STAGING_BUCKET" \
          --solution "$SOLUTION" \
          --version "$cver" \
          --single-region \
          --acl "$ACL_SETTING"
      fi
    )

    # Abort the whole run on any child failure. A partially staged bundle whose
    # root template points at a template that was never uploaded fails at
    # DEPLOY time instead — after real CodeBuild minutes have been spent.
    [ $? -ne 0 ] && \
      echo "" && \
      echo "error: child build failed: ${cdir} @ ${cver} — aborting the bundle" && \
      exit 1
  done
}

######################################################################
# @function verify_staged_children
# head-object every composed child template key.
#
# This is the mechanical guard against a child renaming WORKFLOW_NAME or
# TEMPLATE_NAME, and against a --single-region mismatch between what was staged
# earlier and what this run composes. Without it, either mistake surfaces as a
# failed TemplateURL fetch ~90 minutes into a create, after media-processing's
# CodeBuild has already run.
######################################################################
function verify_staged_children() {
  echo ""
  echo "------------------------------------------------------------------------------"
  echo "Verifying all ${#RESOLVED_DIRS[@]} child templates are staged"
  echo "------------------------------------------------------------------------------"
  local missing=0
  local i
  for i in "${!RESOLVED_DIRS[@]}"; do
    local k="${RESOLVED_KEYS[$i]}"
    if aws s3api head-object \
         --bucket "$STAGING_BUCKET" \
         --key "$k" \
         --expected-bucket-owner "$ACCOUNTID" \
         $REGION_FLAG > /dev/null 2>&1; then
      printf "  [ok]      %s\n" "$k"
    else
      printf "  [MISSING] %s\n" "$k"
      missing=$((missing+1))
    fi
  done

  [ $missing -ne 0 ] && \
    echo "" && \
    echo "error: ${missing} child template(s) not found in s3://${STAGING_BUCKET}/" && \
    echo "       The root template would reference them and fail at deploy time." && \
    if [ "$SKIP_CHILDREN" == "true" ]; then
      echo "       --skip-children was given: re-run without it to stage them."
    fi && \
    exit 1

  echo "  all ${#RESOLVED_DIRS[@]} present"
}

######################################################################
# @function build_cloudformation_template
######################################################################
function build_cloudformation_template() {
  echo ""
  echo "------------------------------------------------------------------------------"
  echo "Building root template: $TEMPLATE_NAME"
  echo "------------------------------------------------------------------------------"
  local src="${SCRIPT_DIR}/cfnstack.yaml"
  local dst="$TEMPLATE_DIST_DIR/$TEMPLATE_NAME"

  [ ! -f "$src" ] && \
    echo "error: cfnstack.yaml not found at $src" && exit 1

  cp "$src" "$dst"

  sed -i'.bak' -e "s|%%BUCKET_NAME%%|${STAGING_BUCKET}|g" "$dst" || exit 1
  sed -i'.bak' -e "s|%%VERSION%%|${BUNDLE_VERSION}|g"     "$dst" || exit 1

  local i
  for i in "${!RESOLVED_SUFFIXES[@]}"; do
    sed -i'.bak' \
      -e "s|%%KEYPREFIX_${RESOLVED_SUFFIXES[$i]}%%|${RESOLVED_KEYS[$i]}|g" \
      "$dst" || exit 1
  done

  rm -f "${dst}.bak"

  # A leftover %%TOKEN%% deploys as a literal string — a TemplateURL pointing at
  # "%%KEYPREFIX_TTS%%" fails the stack after media-processing's 90-minute
  # CodeBuild has already run. Fail here instead.
  #
  # The character class MUST include 0-9. The two children that carry this guard
  # (tts, media-toolkit-gateway) both use '%%[A-Z_]*%%', which silently cannot match
  # %%KEYPREFIX_M2C_AGENT_TOOLS%% — the digit in "M2C" falls outside the class.
  # No child template has a digit-bearing token, so their pattern happens to be
  # sufficient there; this template is the first with one.
  local leftover
  leftover=$(grep -o '%%[A-Z0-9_]*%%' "$dst" | sort -u | tr '\n' ' ')
  [ -n "$leftover" ] && \
    echo "error: unsubstituted token(s) in $TEMPLATE_NAME: $leftover" && exit 1

  cp "$dst" "$BUILD_DIST_DIR/$TEMPLATE_NAME"

  echo ">> Built: $dst"
}

######################################################################
# @function deploy_root_template
######################################################################
function deploy_root_template() {
  echo ""
  echo "------------------------------------------------------------------------------"
  echo "Uploading root template to s3://${STAGING_BUCKET}/${KEY_PREFIX}/"
  echo "------------------------------------------------------------------------------"

  aws s3 cp "$TEMPLATE_DIST_DIR/$TEMPLATE_NAME" \
    "s3://${STAGING_BUCKET}/${KEY_PREFIX}/${TEMPLATE_NAME}" \
    --acl "${ACL_SETTING}" $REGION_FLAG || exit 1

  local domain="s3.amazonaws.com"
  [ "$BUCKET_REGION" != "us-east-1" ] && domain="s3.${BUCKET_REGION}.amazonaws.com"

  ROOT_TEMPLATE_URL="https://${STAGING_BUCKET}.${domain}/${KEY_PREFIX}/${TEMPLATE_NAME}"

  # Record where THIS build's root template actually went, for
  # create_update_stack.sh to consume. The bundle's .version FILE is not
  # that record: --version overrides it at build time without updating it,
  # so a URL derived from the file can name a DIFFERENT, older object that
  # happens to exist — measured 2026-08-20: an update composed from
  # .version=alpha.0807a silently deployed a 2026-08-15 9-child root over
  # the live 12-child alpha.20260819a bundle, DELETING three nested stacks.
  echo "$ROOT_TEMPLATE_URL" > "$TEMPLATE_DIST_DIR/.root_template_url"

  echo ""
  echo "========================================"
  echo "  Template URL:"
  echo "  $ROOT_TEMPLATE_URL"
  echo ""
  echo "  One-click deploy:"
  echo "  https://console.aws.amazon.com/cloudformation/home?region=${BUCKET_REGION}#/stacks/quickcreate?templateURL=${ROOT_TEMPLATE_URL}&stackName=unified-media-toolkit"
  echo "========================================"
}

######################################################################
# @function container_child_count
#
# DERIVED from the children's own templates, never hardcoded. A tally in
# an operator message goes stale the first time a child gains or drops a
# container image, and nothing fails when it does — the message just
# lies. Reading PackageType off the templates cannot drift.
######################################################################
function container_child_count() {
  local n=0 cdir
  for cdir in "${RESOLVED_DIRS[@]}"; do
    grep -q "PackageType: Image" "${DOCKER_ROOT}/${cdir}/cfnstack.yaml" 2>/dev/null \
      && n=$((n + 1))
  done
  echo "$n"
}

######################################################################
# @function child_commit
# Look up a child's HEAD from the per-child provenance pass, shortened to
# 7 hex chars but keeping the -dirty marker. "-" under monorepo mode (one
# whole-tree SHA covers every child) or when provenance was waived.
######################################################################
function child_commit() {
  local dir="$1" entry sha
  for entry in "${CHILD_COMMITS[@]}"; do
    if [ "${entry%%|*}" = "$dir" ]; then
      sha="${entry#*|}"
      if [ "$sha" != "${sha%-dirty}" ]; then
        echo "${sha:0:7}-dirty"
      else
        echo "${sha:0:7}"
      fi
      return
    fi
  done
  echo "-"
}

######################################################################
# @function on_complete
######################################################################
function on_complete() {
  echo ""
  echo "======================================================================"
  echo " Bundle staged — ${SOLUTION} ${BUNDLE_VERSION}"
  echo "======================================================================"
  if [ "$GIT_MODE" = "per-child" ] && [ "$BUILD_SHA" != "unknown" ]; then
    echo "  commit    : $BUILD_SHA (bundle repo; children below)"
  else
    echo "  commit    : $BUILD_SHA"
  fi
  echo "  bucket    : $STAGING_BUCKET"
  echo "  template  : ${KEY_PREFIX}/${TEMPLATE_NAME}"
  echo "----------------------------------------------------------------------"
  printf "  %-32s %-18s %s\n" "CHILD" "VERSION" "COMMIT"
  local i
  for i in "${!RESOLVED_DIRS[@]}"; do
    printf "  %-32s %-18s %s\n" "${RESOLVED_DIRS[$i]}" "${RESOLVED_VERSIONS[$i]}" \
      "$(child_commit "${RESOLVED_DIRS[$i]}")"
  done
  echo "----------------------------------------------------------------------"
  echo "  Next: bash ./create_update_stack.sh --stack-name NAME \\"
  echo "            --ingest-bucket BUCKET --proxy-bucket BUCKET"
  echo ""
  echo "  The first create is LONG: $(container_child_count) container images build in"
  echo "  CodeBuild inside their nested stacks (60-90 min timeouts, in parallel)"
  echo "  and tts additionally seeds six model archives. It is not hung."
  echo "======================================================================"
}

######################################################################
# Main
######################################################################
clean_start

if [ "$SKIP_CHILDREN" == "true" ]; then
  echo ""
  echo "######################################################################"
  echo "# --skip-children: NOT building or staging any of the ${#RESOLVED_DIRS[@]} children."
  echo "# The root template will reference the keys listed above, which must"
  echo "# already be in S3. Verifying that now."
  echo "######################################################################"
else
  build_children
fi

if [ "$BUILD_ONLY" == "true" ]; then
  build_cloudformation_template
  echo ""
  echo "Build-only complete — nothing uploaded. Artifacts in $BUILD_DIST_DIR"
else
  verify_staged_children
  build_cloudformation_template
  deploy_root_template
  on_complete
fi
