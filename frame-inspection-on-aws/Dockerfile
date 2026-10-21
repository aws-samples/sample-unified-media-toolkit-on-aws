# Python 3.13

################################################################################
#
# Build for Lambda (linux/arm64). Always pass the platform flag explicitly
# so builds on x86 machines (CI, Intel Mac) produce the correct arch:
#   docker buildx build --platform linux/arm64 -t frame-inspection-on-aws .
#
# For testing locally
#   docker run --platform linux/arm64 -it --entrypoint /bin/bash frame-inspection-on-aws
#
# The image exists for ONE dependency: pyvips (libvips bundled in the
# pyvips[binary] wheel — no system install), used by lib/imaging.py to
# crop lower-third chyron frames before they become Bedrock image tokens.
# The layout stays FLAT (lib/ and states/ contents copied to the task
# root) so imports are top-level, byte-identical to the old zip's -j
# convention — the gateway zip keeps that convention unchanged.
#
################################################################################
FROM public.ecr.aws/lambda/python:3.13.2026.02.18.08


# Dependencies first: requirements.txt changes rarely, code changes every
# build — this order lets Docker reuse the pip layer across code-only builds.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code, the family TREE shape ──────────────────────────────────
# lambda_function.py (the operation router) at the task root; lib/ and
# states/ are PACKAGES (imports are package-style, `from lib import x`)
COPY lambda_function.py ./
COPY lib/ ./lib/
COPY states/ ./states/

USER 1001

CMD [ "lambda_function.lambda_handler" ]
