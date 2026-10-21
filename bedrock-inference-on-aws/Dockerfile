################################################################################
#
# bedrock-inference-on-aws — Lambda container image (linux/arm64).
#
# THE THINNEST IMAGE IN docker/, and deliberately so: this workflow decodes no
# media and loads no model. It is boto3, JSON, and the Bedrock Converse API. The
# image exists for CONSISTENCY with the sibling workflows — one deployment story,
# one build script shape, one place to look when a build breaks — not because a
# flat zip on the managed python3.13 runtime stopped being sufficient. It was
# sufficient until 2026-08-12 and it still would be.
#
# Built by CodeBuild on ARM_CONTAINER (see cfnstack.yaml), so the platform is
# native and no --platform flag is involved. Building by hand on an x86 machine
# needs one:
#
#   docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
#     -t bedrock-inference-on-aws .
#
# (--provenance/--sbom: Lambda rejects buildx's default OCI attestation
#  manifests — the siblings all learned this the hard way.)
#
# For testing locally:
#   docker run --platform linux/arm64 -it --entrypoint /bin/bash \
#     bedrock-inference-on-aws
#
################################################################################
# Pinned to a dated tag rather than floating on :3.13 — a base image that moves
# under an unchanged .version would put a different digest behind the same ECR
# tag, and the buildspec's skip-if-exists would then be skipping a build that
# was NOT the build that produced the running image.
#
# A PIN IS A FLOOR AS WELL AS A CEILING, and that half of it is the half that
# bites. On the managed runtime AWS keeps botocore moving under you; in an image
# this line is the only thing that decides which botocore the code gets, and it
# freezes it forever. The tag before this one (3.13.2026.02.18.08) shipped
# botocore 1.40.4, which predates `modelInvocationType` on
# CreateModelInvocationJob by two minor versions — so on 2026-08-12 the zip→image
# move silently broke every batch submit with a CLIENT-SIDE "Unknown parameter in
# input", while sync, the semaphore and the whole offline suite stayed green
# (the unit test for that parameter asserts against a MagicMock, which accepts
# any kwarg). This tag ships 1.42.97. requirements.txt records the 1.42.59 floor
# and a test cross-checks the two files.
#
# So: when bumping this, check botocore, do not just take the newest tag —
#   docker run --rm --platform linux/arm64 --entrypoint python3 <tag> \
#     -c 'import botocore; print(botocore.__version__)'
# and confirm `CancellationReasons` is still a member of DynamoDB's
# TransactionCanceledException, because try_acquire reads it.
FROM public.ecr.aws/lambda/python:3.13.2026.08.07.17

# ── NO pip install, and that is a decision, not an omission ───────────────────
# The Lambda python base image already ships boto3/botocore, and
# `pip install -r requirements.txt` would put a second copy over the top. A
# vendored botocore goes stale and takes the base image's TransactWriteItems
# error shape with it, and try_acquire's CancellationReasons parsing depends on
# that shape. There is no third-party dependency to install: this is DynamoDB
# transactions and the Converse API.
#
# requirements.txt is therefore not an install list — it is the FLOOR the FROM
# tag above has to clear, and it is copied in so the running image records what
# it was built against. Satisfying that floor is this line's job, not pip's; if a
# future parameter needs a botocore newer than any published base tag, raise the
# floor and vendor deliberately, accepting the CancellationReasons risk. Don't
# quietly `pip install -U` and leave the two copies to fight.

# ── Application code ──────────────────────────────────────────────────────────
# Explicit chmod: Docker COPY preserves host permissions, and a 0600 source file
# (an editor artifact) breaks the non-root Lambda runtime user. NOTE --chmod
# applies to DIRECTORIES too — 0644 on a directory strips the traversal bit and
# silently turns packages into namespace packages (Python can list lib/ but not
# open lib/__init__.py). Copy first, then fix files and dirs together with
# chmod's X flag: execute on directories only, read on everything.
COPY requirements.txt .version lambda_function.py ./
COPY lib/ ./lib/
COPY states/ ./states/
# models/ carries availablemodels_master.json — the master list the
# Custom::ModelRegistrySeeder installs. The SEEDER does not run from this image
# (it is a flat zip out of the gateway-tool package, as in every sibling), so
# this copy is not what feeds it; `lib/bedrock.py` reads the INSTALLED registry
# from S3 at runtime and never touches the master. It is copied anyway, for one
# reason: `states/` is copied whole and the seeder module comes with it, so
# leaving models/ out would make that module importable but broken. Cheap
# (20 KB) and it keeps the image a complete copy of the tree.
COPY models/ ./models/
RUN chmod -R u=rwX,go=rX /var/task

USER 1001

# ONE entry point, used by all seven memory tiers, and nothing in the template
# overrides it: there is no `ImageConfig.Command` anywhere. An image function has
# no `Handler` property, so this CMD is the only entry point the image has, and
# the router inside lambda_function dispatches on the payload's `operation`.
CMD [ "lambda_function.lambda_handler" ]
