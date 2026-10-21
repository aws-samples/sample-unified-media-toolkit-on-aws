# Python 3.13

################################################################################
#
# Build for Lambda (linux/arm64). Always pass the platform flag explicitly
# so builds on x86 machines (CI, Intel Mac) produce the correct arch:
#   docker buildx build --platform linux/arm64 -t speech-enhancement-on-aws .
#
# For testing locally
#   docker run --platform linux/arm64 -it --entrypoint /bin/bash speech-enhancement-on-aws
#
################################################################################
FROM public.ecr.aws/lambda/python:3.13.2026.02.18.08


# ── Application code ───────────────────────────────────────────────────────────
COPY requirements.txt lambda_function.py ./
COPY lib/ ./lib/
COPY states/ ./states/

# ── System deps ───────────────────────────────────────────────────────────────
# libgomp: OpenMP runtime required by onnxruntime and ctranslate2 for
#          intra-op parallelism. Present in amazonlinux:2023 but not always
#          in the slim Lambda base — install explicitly to be safe.
# RUN microdnf install -y libgomp && microdnf clean all
RUN mkdir -p /sys/devices/system/cpu && \
  microdnf install -y libgomp tar gzip && \
  microdnf clean all && \
  pip install --no-cache-dir -r requirements.txt

# ── Model artefacts ───────────────────────────────────────────────────────────
# DeepFilterNet3 — upstream Rikorose/DeepFilterNet checkpoint
# (MIT OR Apache-2.0), re-exported to ONNX; provenance verified
# 2026-10-10 (all 55 tensors match upstream on size/mean/std;
# config.ini byte-identical). sha256-pinned per the supply-chain rule.
ARG DFN3_MODEL_SHA256=f6fcc88ddbcacac10c6c62b939cc989b2b72d313157150bae30ff534f81240c6
RUN mkdir -p /opt/models && \
  curl -fsSL https://dx2y1cac29mt3.cloudfront.net/opensource_models/deepfilternet/model.tar.gz \
  -o /tmp/model.tar.gz && \
  echo "${DFN3_MODEL_SHA256}  /tmp/model.tar.gz" | sha256sum -c - && \
  tar -xzf /tmp/model.tar.gz -C /opt/models/ && \
  rm /tmp/model.tar.gz

# ── ORT cpuinfo fix ───────────────────────────────────────────────────────────
# Lambda removes /sys/devices/system/cpu/possible and /present from its
# sandbox. ORT's pytorch/cpuinfo hard-crashes on arm64 when these files are
# missing. patch.txt contains "0-5" — covers all Lambda vCPU sizes up to
# 10240 MB (6 vCPU). The actual thread count is determined at runtime by
# lib/ort_init.py reading /proc/cpuinfo, so the range here just needs to
# exist and be parseable — it doesn't need to be exact.
# Reference: https://github.com/DiscreteTom/onnx-on-aws-lambda-arm64
# RUN mkdir -p /sys/devices/system/cpu
COPY patch.txt /sys/devices/system/cpu/possible
COPY patch.txt /sys/devices/system/cpu/present

USER 1001

# ── Set default command ───────────────────────────────────────────────────────────
CMD [ "lambda_function.lambda_handler" ]
