#!/bin/bash

########################################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
########################################################################################

# Run this script from bedrock-inference-on-aws/
#
# bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region]
# bash ./build-deploy-s3-dist.sh --build-only [--version VERSION]
#
# DOCKER, CODEBUILD AND ECR since 2026-08-12, adopted for CONSISTENCY with the rest of
# the repo and not because this workflow needs a container: it decodes no media and loads
# no model — it is boto3, JSON, and the Bedrock Converse API — so a flat zip on the
# managed python3.13 arm64 runtime was and remains technically sufficient. The siblings
# that need a container need it for open-source model weights; matching them means one
# deployment story to learn and one build script shape to read.
#
# Nothing is pip-installed into the image. requirements.txt pins boto3/botocore for the
# record; they are deliberately NOT vendored (a vendored botocore shadows the runtime's
# and goes stale, and the runtime's is what carries new Bedrock model ids and Converse
# fields). See the Dockerfile.
#
# TWO PACKAGES, and only the second is a Lambda deployment zip:
#
#   {WORKFLOW}-{VERSION}.zip               CodeBuild SOURCE archive — the Dockerfile
#                                          plus the code it copies. CodeBuild builds it
#                                          into the ONE image all seven memory tiers and
#                                          the registry seeder run from.
#   {WORKFLOW}-gateway-tool-{VERSION}.zip  the AgentCore dispatcher, still a flat zip,
#                                          as it is in every sibling.
#
# The gateway tool carries lib/ — unlike the flat gateway zips elsewhere in the repo,
# this dispatcher imports `lib.identity` and `lib.input_schema` because it computes
# workflow_id and validates with the workflow's OWN parser. Two copies of the hash rule
# would be a hash that silently disagrees, and the symptom is a resubmit that re-pays
# for work already done (README.md §3).
#
# BUMP .version ON EVERY DEPLOY. The version is the ECR image tag, the buildspec skips
# the docker build when that tag already exists, and Lambda resolves ImageUri to a
# DIGEST at update time — so re-deploying an unbumped version leaves all seven tiers
# pinned to the old digest and the code change silently does not ship. There is no error
# to notice; the stack reports UPDATE_COMPLETE.
#
# NOT A SINGLETON, and the merge is what made that true of the semaphore too. When the
# concurrency cap lived in its own stack it was one table per region by name; two tables
# now carry `${ResourcePrefix}`, so two deployments hold two independent caps
# (README.md §M2.3). Many inference stacks per region are fine: the account-level
# batch job limit is service-enforced, and the drain rule is deliberately unscoped so a
# foreign completion still wakes this deployment's queue.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKFLOW_NAME="bedrock-inference"
TEMPLATE_NAME="${WORKFLOW_NAME}-stack.template"

TEMPLATE_DIST_DIR="$SCRIPT_DIR/dist/global-s3-assets"
BUILD_DIST_DIR="$SCRIPT_DIR/dist/regional-s3-assets"

#
# @function usage
#
function usage() {
  echo -e "
------------------------------------------------------------------------------

Build and deploy the Bedrock Inference CloudFormation stack.
Run from bedrock-inference-on-aws/

------------------------------------------------------------------------------
bash ./build-deploy-s3-dist.sh --bucket BUCKET [--solution SOLUTION] [--version VERSION] [--single-region] [--build-only]

where
  --bucket BUCKET     S3 bucket name to deploy to. Without --single-region this is
                      treated as a basename and the region is appended, e.g. BUCKET-us-east-1.

  --build-only        Build packages only; skip deploy to S3 Bucket.
                      Use this when the M2C build script handles deployment.

  --solution SOLUTION [optional] S3 key prefix namespace. Defaults to 'unified-media-toolkit'.

  --version VERSION   [optional] Stack version. Defaults to contents of .version file.

  --single-region     [optional] Deploy to exactly the bucket specified in --bucket (no region suffix).

  --acl ACL           [optional] S3 object ACL. Defaults to 'bucket-owner-full-control'.

SELF-CONTAINED — no sibling stack, no deploy order. It owns its own concurrency
semaphore and its own batch queue, so ResourcePrefix and ProxyBucket are the only two
parameters without a default (README.md §M1).
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
[ -z "$BUILD_ONLY" ] && BUILD_ONLY=false

[ "$(which jq)" == "" ] && \
  echo "error: jq is required" && exit 1

# --build-only produces artefacts in dist/ and touches no AWS API. Requiring the
# CLI, credentials, and a live bucket for it would make the package unbuildable in
# CI and offline, which is exactly where a build gets verified.
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

# Substituted into the template's CodeBuild.Source mapping; CodeBuild reads the source
# archive from here at deploy time, so a build-only run still needs a name in the
# template even though nothing is uploaded.
[ -z "$BUCKET_NAME" ] && BUCKET_NAME="BUCKET_NAME_NOT_SET"

######################################################################
# Derived variables
######################################################################
PKG_BEDROCK_INFERENCE="${WORKFLOW_NAME}-${VERSION}.zip"
PKG_GATEWAY_TOOL="${WORKFLOW_NAME}-gateway-tool-${VERSION}.zip"
KEY_PREFIX="${SOLUTION}/${VERSION}/${WORKFLOW_NAME}"

echo "======================================================================"
if [ "$BUILD_ONLY" == "true" ]; then
  echo " Bedrock Inference — Build Only"
else
  echo " Bedrock Inference — Build & Deploy"
fi
echo "======================================================================"
echo "  solution  : $SOLUTION"
echo "  version   : $VERSION"
echo "  bucket    : $BUCKET_NAME"
echo "  key prefix: $KEY_PREFIX"
echo "  package   : $PKG_BEDROCK_INFERENCE"
echo "  gateway   : $PKG_GATEWAY_TOOL"
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
# Refuses to package on failure. The failures this suite exists to catch are the
# ones production hides: a workflow_id that is not stable re-pays for work already
# done, a checkpoint that is read as absent re-inferences every completed record,
# and an ErrorEquals naming a service error code is dead retry configuration that
# reads as working back-pressure. None of those raise anywhere.
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
# @function build_codebuild_source_package
# The CodeBuild SOURCE archive — NOT a Lambda deployment zip. CodeBuild
# unpacks it, runs `docker build .` in the result, and pushes the image to
# ECR; the seven tier functions and the registry seeder all point at that
# one image.
#
# The Dockerfile must therefore be at the ARCHIVE ROOT, which is what the
# guard below is for: CodeBuild's docker build would fail with a bare
# "Dockerfile: no such file or directory" and the stack would sit in
# CREATE_IN_PROGRESS until the custom resource's own timeout, because the
# CFN response comes from the buildspec's post_build and a failed build
# never reaches it.
#
# Members are exactly what the Dockerfile COPYs plus the Dockerfile itself.
# Test files, the plans and the definitions copies are deliberately absent —
# they belong in the source-code zip below, not in a layer that ships to
# every Lambda invocation.
######################################################################
function build_codebuild_source_package() {
  echo "------------------------------------------------------------------------------"
  echo "Building CodeBuild source archive: $PKG_BEDROCK_INFERENCE"
  echo "------------------------------------------------------------------------------"
  pushd "$SCRIPT_DIR" > /dev/null

  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
  find . -name '*.pyc' -type f -delete 2>/dev/null || true

  # `zip` exits 0 when only SOME members are missing — it warns, packages the
  # rest, and succeeds. Every one of these is fatal LATER and elsewhere (in a
  # CodeBuild log, or on a Lambda cold start), so check them here.
  #
  # models/availablemodels_master.json is what the Custom::ModelRegistrySeeder
  # installs; without it the stack fails at CREATE, not in a test.
  # named image_sources, deliberately nothing like the gateway array's name:
  # the bundle suite regex-extracts the FIRST such array in this script, and a
  # second one earlier in the file would shadow the one it exists to check.
  local image_sources=(Dockerfile lambda_function.py requirements.txt .version \
                       models/availablemodels_master.json)
  local f
  for f in "${image_sources[@]}"; do
    if [ ! -e "$SCRIPT_DIR/$f" ]; then
      echo "error: $f not found — CodeBuild source archive would build nothing"
      exit 1
    fi
  done

  zip -rq "$BUILD_DIST_DIR/$PKG_BEDROCK_INFERENCE" \
    Dockerfile \
    .version \
    lambda_function.py \
    lib \
    models \
    states \
    requirements.txt \
    -x '*/__pycache__/*' \
    || exit 1

  echo ">> Built: $BUILD_DIST_DIR/$PKG_BEDROCK_INFERENCE"
  popd > /dev/null
}

######################################################################
# @function build_gateway_tool_package
# The dispatcher's own zip, with gateway_tool/lambda_function.py hoisted to
# the archive ROOT and lib/ alongside it.
#
# Carrying lib/ is what makes this different from the flat gateway zips
# elsewhere in the repo: this dispatcher IMPORTS lib.identity and
# lib.input_schema rather than reimplementing them, because it computes
# workflow_id before writing the request objects (the prefix is derived from
# the id) and StateValidateAndPlan recomputes the same hash and compares. A
# flat zip would force a second copy of the rule, and a hash that silently
# disagrees breaks idempotent resubmit (README.md §3).
#
# TWO FUNCTIONS COME OUT OF THIS ONE ZIP, which is media-processing's pattern
# exactly (its `bucket-check` custom resource reads GatewayToolPackage too):
# the gateway dispatcher at the root, and the Custom::ModelRegistrySeeder at
# states/seed_model_registry/. The seeder is deliberately NOT in the container
# image — no sibling puts a CloudFormation custom resource there, and on the
# image it would have to wait for the docker build before the registry could be
# seeded. `states/` and `models/` keep their directory structure (no -j) because
# the handler is the dotted path `states.seed_model_registry.lambda_function` and
# the seeder locates its master list relative to its own __file__.
######################################################################
function build_gateway_tool_package() {
  echo "------------------------------------------------------------------------------"
  echo "Building Gateway tool Lambda package: $PKG_GATEWAY_TOOL"
  echo "------------------------------------------------------------------------------"

  if [ ! -d "$SCRIPT_DIR/gateway_tool" ]; then
    echo ">> gateway_tool/ absent — skipping"
    return 0
  fi

  pushd "$SCRIPT_DIR" > /dev/null

  # `zip` exits 0 when only SOME members are missing — it warns, packages the
  # rest, and succeeds, so a dropped module builds clean and fails at import on
  # the first invocation (micro-workflow-bundle/README.md §C5; it really happened, twice). Check each
  # member up front.
  local members=(lambda_function.py callback_waker.py callback_registry.py workflow_identity.py output_contract.json input_examples.json)
  local f
  for f in "${members[@]}"; do
    if [ ! -f "gateway_tool/$f" ]; then
      echo "error: gateway_tool/$f not found"
      exit 1
    fi
  done

  # Same check for the seeder's members, and `models/` is the one that matters:
  # a zip missing the master list builds clean, deploys clean, and fails the
  # stack at CREATE inside a custom resource — the worst place to find it.
  local seeder=(
    states/__init__.py
    states/seed_model_registry/__init__.py
    states/seed_model_registry/lambda_function.py
    models/availablemodels_master.json
  )
  for f in "${seeder[@]}"; do
    if [ ! -f "$f" ]; then
      echo "error: $f not found (Custom::ModelRegistrySeeder ships in this zip)"
      exit 1
    fi
  done

  # -j flattens: gateway_tool/lambda_function.py lands at the root, where
  # Handler: lambda_function.lambda_handler resolves.
  zip -jq "$BUILD_DIST_DIR/$PKG_GATEWAY_TOOL" gateway_tool/*.py \
    gateway_tool/output_contract.json gateway_tool/input_examples.json || exit 1
  zip -rq "$BUILD_DIST_DIR/$PKG_GATEWAY_TOOL" lib \
    -x '*/__pycache__/*' || exit 1
  # NOT -j: the seeder's handler is a dotted path and its master list is found
  # relative to __file__, so both need their real paths in the archive. Only the
  # ONE state directory — the other fourteen belong to the image.
  zip -rq "$BUILD_DIST_DIR/$PKG_GATEWAY_TOOL" \
    states/__init__.py states/seed_model_registry models \
    -x '*/__pycache__/*' || exit 1

  echo ">> Built: $BUILD_DIST_DIR/$PKG_GATEWAY_TOOL"
  popd > /dev/null
}

######################################################################
# @function verify_sfn_definition
# The three ASL definitions live INLINE in cfnstack.yaml, and the template is
# the single source of truth for what deploys. `definitions*.json` are
# readability copies — lintable, diffable, and openable in a state machine
# editor on their own — so this function's whole job is to prove the copy has
# not drifted from the template.
#
# THREE state machines, three files (README.md §M1.3): the orchestrator that
# callers start, and the two standalone children it routes between.
#
# A drift is a HARD BUILD ERROR rather than a warning, and it has to be: a
# reader who edits definitions_sync.json and deploys would ship the template's
# older body while the diff they reviewed says otherwise — the failure mode a
# readability copy exists to avoid, not to create. Compared as PARSED JSON, so
# re-indenting either side is not drift; only the machine's meaning is.
#
# There is no token substitution here any more. The `${...}` ARNs in each body
# are supplied by that template resource's own DefinitionSubstitutions, exactly
# as before.
######################################################################
function verify_sfn_definition() {
  local dst=$1

  python3 - "$dst" "$SCRIPT_DIR" << 'PYVERIFY'
import json, os, re, sys

dst, script_dir = sys.argv[1], sys.argv[2]
src = open(dst, encoding="utf-8").read()

# (CFN logical resource id, readability copy). The resource id is what anchors
# the search: matching on the file's own content would make a drifted copy
# unfindable, which is the one case this check exists for.
MACHINES = (
    ("InferenceStateMachine", "definitions.json"),
    ("SyncInferenceStateMachine", "definitions_sync.json"),
    ("BatchInferenceStateMachine", "definitions_batch.json"),
)


def embedded(resource):
    """
    The `DefinitionString: |-` block scalar under `resource`, dedented.

    Hand-parsed rather than read through a YAML loader: this template is full of
    short-form intrinsics (`!Sub`, `!GetAtt`) that SafeLoader rejects outright,
    and pulling in a CFN-aware loader would make the build depend on one.
    """
    anchor = re.search(rf"(?m)^([ \t]*){re.escape(resource)}:[ \t]*$", src)
    if not anchor:
        raise SystemExit(f"error: resource {resource} not found in {dst}")

    start = re.compile(r"(?m)^([ \t]*)DefinitionString:[ \t]*\|-[ \t]*$")
    header = start.search(src, anchor.end())
    if not header:
        raise SystemExit(
            f"error: {resource} has no 'DefinitionString: |-' block in {dst}")

    indent = header.group(1) + "    "
    lines = []
    for line in src[header.end():].split("\n")[1:]:
        if line.strip() and not line.startswith(indent):
            break
        lines.append(line[len(indent):] if line.strip() else "")
    return "\n".join(lines).rstrip("\n")


for resource, fname in MACHINES:
    body = embedded(resource)
    try:
        inline = json.loads(body)
    except ValueError as exc:
        raise SystemExit(
            f"error: {resource}'s inline DefinitionString is not valid JSON in "
            f"{dst}: {exc}")

    with open(os.path.join(script_dir, fname), encoding="utf-8") as handle:
        copy = json.load(handle)

    if inline != copy:
        raise SystemExit(
            f"error: {fname} has DRIFTED from {resource}'s inline "
            f"DefinitionString. The TEMPLATE is what deploys; the file is the "
            f"readability copy. Reconcile them — do not deploy a template whose "
            f"reviewable copy says something else.")
    print(f">> verified {fname} matches {resource}")

print(f">> {len(MACHINES)} SFN definitions verified against their copies")
PYVERIFY
}

######################################################################
# @function build_cloudformation_template
######################################################################
function build_cloudformation_template() {
  echo "------------------------------------------------------------------------------"
  echo "Building CloudFormation template: $TEMPLATE_NAME"
  echo "------------------------------------------------------------------------------"
  local src="$SCRIPT_DIR/cfnstack.yaml"
  local dst="$TEMPLATE_DIST_DIR/$TEMPLATE_NAME"

  [ ! -f "$src" ] && \
    echo "error: cfnstack.yaml not found at $src" && exit 1

  # Against the SOURCE, before the sed pass: the tokens below are substituted
  # everywhere in the file, so a %%VERSION%% inside an ASL comment would leave the
  # built copy legitimately different from definitions*.json and this check would
  # read that as drift. Comparing the source instead keeps the guard about the two
  # things it is meant to compare.
  # `|| exit 1` because there is no `set -e` here: without it a drift would print
  # its error and the build would carry on and ship, which is the opposite of a guard.
  verify_sfn_definition "$src" || exit 1

  cp "$src" "$dst"

  sed -i'.bak' -e "s|%%BUCKET_NAME%%|${BUCKET_NAME}|g"                     "$dst" || exit 1
  sed -i'.bak' -e "s|%%KEY_PREFIX%%|${KEY_PREFIX}|g"                       "$dst" || exit 1
  sed -i'.bak' -e "s|%%VERSION%%|${VERSION}|g"                             "$dst" || exit 1
  sed -i'.bak' -e "s|%%PKG_BEDROCK_INFERENCE%%|${PKG_BEDROCK_INFERENCE}|g" "$dst" || exit 1
  sed -i'.bak' -e "s|%%PKG_GATEWAY_TOOL%%|${PKG_GATEWAY_TOOL}|g"           "$dst" || exit 1

  rm -f "${dst}.bak"

  cp "$dst" "$BUILD_DIST_DIR/$TEMPLATE_NAME"

  echo ">> Built: $dst"
}

######################################################################
# @function deploy_to_bucket
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
  echo "  https://console.aws.amazon.com/cloudformation/home?region=${region}#/stacks/quickcreate?templateURL=${url}&stackName=bedrock-inference"
  echo ""
  echo "  No sibling stack to deploy first — this one is self-contained."
  echo "  Two parameters have no default:"
  echo "    ResourcePrefix  names both tables; the sync cap is per PREFIX"
  echo "    ProxyBucket     model registry + request/response + batch i/o"
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
  echo "  CODEBUILD SOURCE      : $PKG_BEDROCK_INFERENCE"
  echo "  PKG_GATEWAY_TOOL      : $PKG_GATEWAY_TOOL"
  echo "  TEMPLATE              : $TEMPLATE_NAME"
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
build_codebuild_source_package
build_gateway_tool_package
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
