# Python 3.12 — REQUIRED by the onnxruntime==1.18.0 pin (no cp313 wheels;
# see requirements.txt for why 1.18.0 is non-negotiable).

################################################################################
#
# Build for Lambda (linux/arm64). Always pass the platform flag explicitly
# so builds on x86 machines produce the correct arch:
#   docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
#     -t tts-on-aws .
#
# (--provenance/--sbom: Lambda rejects buildx's default OCI attestation
#  manifests — found the hard way in Phase 1.)
#
# For testing locally:
#   docker run --platform linux/arm64 -it --entrypoint /bin/bash tts-on-aws
#
# Models are NOT baked into the image — they are seeded to S3 at stack
# creation (CodeBuild, see cfnstack.yaml) and downloaded to /tmp at runtime
# via lib/model_store.py. Image stays small; model updates don't need an
# image rebuild.
#
################################################################################
FROM public.ecr.aws/lambda/python:3.12.2026.02.18.08

# ── Application code ──────────────────────────────────────────────────────────
# Explicit chmod: Docker COPY preserves host permissions; a 0600 source file
# (editor artifact) breaks the non-root Lambda runtime user (Phase 1 lesson).
# NOTE --chmod applies to DIRECTORIES too — 0644 on a directory strips the
# traversal bit and silently turns packages into namespace packages (Python
# can list lib/ but not open lib/__init__.py). Copy, then fix files/dirs
# separately.
COPY requirements.txt lambda_function.py ./
COPY lib/ ./lib/
COPY states/ ./states/
COPY assets/ ./assets/
# (`find` is absent from the slim base — chmod's X flag does the same job:
#  execute bit on directories only, read on everything.)
RUN chmod -R u=rwX,go=rX /var/task

# ── System deps + Python deps ─────────────────────────────────────────────────
# libgomp: OpenMP runtime required by onnxruntime and ctranslate2.
RUN mkdir -p /sys/devices/system/cpu && \
  microdnf install -y libgomp && \
  microdnf clean all && \
  pip install --no-cache-dir -r requirements.txt

# ── ORT cpuinfo fix ───────────────────────────────────────────────────────────
# Lambda removes /sys/devices/system/cpu/possible and /present from its
# sandbox. ORT's cpuinfo hard-crashes on arm64 when these are missing.
# patch.txt contains "0-5" — covers all Lambda vCPU sizes up to 10240 MB.
# lib/ort_init.py must run before the first onnxruntime import (enforced
# by lib/sessions.py being the only ORT entry point).
COPY patch.txt /sys/devices/system/cpu/possible
COPY patch.txt /sys/devices/system/cpu/present

USER 1001

CMD [ "lambda_function.lambda_handler" ]
