#!/bin/bash
###############################################################################
# build-deploy-s3-dist.sh (repo root) — build the whole toolkit and produce
# the single source package.
#
#   bash ./build-deploy-s3-dist.sh --bucket BUCKET_BASE_NAME
#                                  [--solution SOLUTION] [--version VERSION]
#                                  [--skip-children] [--acl ACL] [--build-only]
#
# Two stages, in order:
#
#   1. Delegate to micro-workflow-bundle/build-deploy-s3-dist.sh with the
#      same options — it builds every workflow, the gateway, and the root
#      CloudFormation template. The bundle folder owns that logic, its
#      tests, and its README.md.
#
#   2. Package the TRACKED TREE (git ls-files, working-tree content) into
#      ONE flat source zip under ./dist/. The tracked tree is the curated
#      artifact — the no-customer-identifiers rule, .gitignore, and the
#      per-folder leak gates police what gets tracked — so the package is
#      complete and drift-proof by construction. A final leak sweep over
#      the result stays as a belt.
#
# Under --build-only nothing touches AWS; otherwise the source package is
# uploaded to the same staging prefix the bundle resolved (it records the
# resolved values in micro-workflow-bundle/dist/.staging.env — one bucket
# resolution, one writer).
###############################################################################
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

(cd "$ROOT_DIR/micro-workflow-bundle" && bash ./build-deploy-s3-dist.sh "$@") || exit 1

# shellcheck source=/dev/null
source "$ROOT_DIR/micro-workflow-bundle/dist/.staging.env"
[ -z "$BUNDLE_VERSION" ] && \
  echo "error: the bundle build left no staging record" && exit 1

SRC_PACKAGE="unified-media-toolkit-sourcecode-${BUNDLE_VERSION}.zip"
ROOT_DIST_DIR="$ROOT_DIR/dist"

echo ""
echo "------------------------------------------------------------------------------"
echo "Packaging the tracked tree: $SRC_PACKAGE"
echo "------------------------------------------------------------------------------"
rm -rf "$ROOT_DIST_DIR"
mkdir -p "$ROOT_DIST_DIR"

# git ls-files is the member list; file CONTENT comes from the working tree,
# so a --allow-dirty build packages the same code the child artifacts were
# just built from. Untracked and ignored files can never join.
(cd "$ROOT_DIR" && git ls-files -z | xargs -0 zip -q "$ROOT_DIST_DIR/$SRC_PACKAGE") || exit 1

TRACKED_COUNT=$(cd "$ROOT_DIR" && git ls-files | wc -l | tr -d ' ')
ZIPPED_COUNT=$(unzip -l "$ROOT_DIST_DIR/$SRC_PACKAGE" | tail -1 | awk '{print $2}')
[ "$TRACKED_COUNT" != "$ZIPPED_COUNT" ] && \
  echo "error: package member count ($ZIPPED_COUNT) != tracked file count ($TRACKED_COUNT)" && \
  exit 1

# Final leak sweep — a belt, since tracked content should never match these
# LOCAL-ONLY artifact classes (sharp-edges journals, identity-carrying
# working files, agent/IDE config, generated per-deployment helpers,
# bytecode). A hit means something slipped past .gitignore into the index.
leaks=$(unzip -l "$ROOT_DIST_DIR/$SRC_PACKAGE" \
        | grep -E 'GOTCHAS\.md|do-not-commit/|\.mcp\.json|gateway_headers\.|__pycache__|\.pyc$' || true)
[ -n "$leaks" ] && \
  echo "error: source package carries excluded content:" && \
  echo "$leaks" | sed 's/^/       /' && exit 1

echo ">> Built: $ROOT_DIST_DIR/$SRC_PACKAGE ($(du -h "$ROOT_DIST_DIR/$SRC_PACKAGE" | cut -f1), ${ZIPPED_COUNT} files)"

# The client kit the gateway build just assembled (skills + configure/token
# scripts + Consumer Guide). This staged copy is the GENERIC kit — at deploy
# time the KitPublisher injects a per-deployment kit.env and publishes to the
# proxy bucket's _toolkit/ prefix; this one defaults to the documented
# secret path.
# named by the GATEWAY's resolved version (the kit's builder), which only
# equals BUNDLE_VERSION under a --version override — matching the
# KitPublisher's deploy-time naming convention
KIT_PACKAGE="client-kit-${GATEWAY_VERSION:-$BUNDLE_VERSION}.zip"
KIT_SRC="$ROOT_DIR/media-toolkit-gateway/gateway_tool/client_kit.zip"
[ ! -f "$KIT_SRC" ] && \
  echo "error: the gateway build left no client kit at media-toolkit-gateway/gateway_tool/client_kit.zip" && exit 1
cp "$KIT_SRC" "$ROOT_DIST_DIR/$KIT_PACKAGE"
echo ">> Built: $ROOT_DIST_DIR/$KIT_PACKAGE ($(du -h "$ROOT_DIST_DIR/$KIT_PACKAGE" | cut -f1))"

if [ "$BUILD_ONLY" == "true" ]; then
  echo "Build-only: the source package was not uploaded."
  exit 0
fi

# The source package describes the whole release, so it lives at the
# solution/version ROOT — beside the per-workflow folders, not inside the
# bundle's.
RELEASE_PREFIX="${SOLUTION}/${BUNDLE_VERSION}"
aws s3 cp "$ROOT_DIST_DIR/$SRC_PACKAGE" \
  "s3://${STAGING_BUCKET}/${RELEASE_PREFIX}/${SRC_PACKAGE}" \
  --acl "${ACL_SETTING}" $REGION_FLAG || exit 1
aws s3 cp "$ROOT_DIST_DIR/$KIT_PACKAGE" \
  "s3://${STAGING_BUCKET}/${RELEASE_PREFIX}/${KIT_PACKAGE}" \
  --acl "${ACL_SETTING}" $REGION_FLAG || exit 1

# ── release summary: the three things to copy ──
# The template URL is read from the bundle's record of where THIS build's
# template actually went (.root_template_url), never recomposed from
# .version — see the bundle script for the incident that rule encodes.
ROOT_TEMPLATE_URL=$(cat "$ROOT_DIR/micro-workflow-bundle/dist/global-s3-assets/.root_template_url" 2>/dev/null)
[ -z "$ROOT_TEMPLATE_URL" ] && \
  echo "error: the bundle build left no .root_template_url record" && exit 1

S3_DOMAIN="s3.amazonaws.com"
[ "$BUCKET_REGION" != "us-east-1" ] && S3_DOMAIN="s3.${BUCKET_REGION}.amazonaws.com"
SRC_PACKAGE_URL="https://${STAGING_BUCKET}.${S3_DOMAIN}/${RELEASE_PREFIX}/${SRC_PACKAGE}"
KIT_PACKAGE_URL="https://${STAGING_BUCKET}.${S3_DOMAIN}/${RELEASE_PREFIX}/${KIT_PACKAGE}"

echo ""
echo "=============================================================================="
echo " Release ${BUNDLE_VERSION} — copy and paste"
echo "=============================================================================="
echo ""
echo "  CFN template URL:"
echo "  ${ROOT_TEMPLATE_URL}"
echo ""
echo "  One-click deploy:"
echo "  https://console.aws.amazon.com/cloudformation/home?region=${BUCKET_REGION}#/stacks/quickcreate?templateURL=${ROOT_TEMPLATE_URL}&stackName=unified-media-toolkit"
echo ""
echo "  Source code package:"
echo "  ${SRC_PACKAGE_URL}"
echo ""
echo "  Client kit (generic — deployments publish their own at _toolkit/):"
echo "  ${KIT_PACKAGE_URL}"
echo ""
echo "=============================================================================="
