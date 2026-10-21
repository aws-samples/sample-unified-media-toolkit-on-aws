# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Search — Lambda arm64 image
#
# Base: public.ecr.aws/lambda/python:3.13 (Amazon Linux 2023, glibc 2.34)
#   ✓ LanceDB manylinux_2_17_aarch64 (needs ≥ 2.17)
#   ✓ awslambdaric already bundled in base image — not in requirements.txt
#
# Single image serves all three Lambda functions.
# Operation routing is via the "operation" field in the event payload,
# handled by lambda_function.lambda_handler at runtime.
#
# Build:
#   docker build --platform linux/arm64 -t hybrid-search-on-aws .
# ─────────────────────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.13.2026.02.18.08
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION}

# ── Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# ── Application source
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py
COPY lib/               ${LAMBDA_TASK_ROOT}/lib/
COPY states/            ${LAMBDA_TASK_ROOT}/states/

WORKDIR ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
