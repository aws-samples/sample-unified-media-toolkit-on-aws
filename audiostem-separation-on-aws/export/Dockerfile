# Python 3.13

################################################################################
#
# Build for Lambda (linux/arm64). Always pass the platform flag explicitly
# so builds on x86 machines (CI, Intel Mac) produce the correct arch:
#   docker buildx build --platform linux/arm64 -t onnx-export-image .
#
# For testing locally
#   docker run --platform linux/arm64 -it --entrypoint /bin/bash onnx-export-image
#
################################################################################
FROM public.ecr.aws/lambda/python:3.13.2026.02.18.08

# ── System deps ───────────────────────────────────────────────────────────────
# libgomp: OpenMP runtime required by onnxruntime and ctranslate2 for
#          intra-op parallelism. Present in amazonlinux:2023 but not always
#          in the slim Lambda base — install explicitly to be safe.
RUN microdnf install -y libgomp && microdnf clean all

# ── Python deps — layered for cache efficiency ────────────────────────────────

# 1. Standard deps (numpy, scipy, onnxruntime, boto3, faster-whisper,
#    tokenizers, huggingface-hub)

# ── Application code ──────────────────────────────────────────────────────────
COPY \
  requirements.txt \
  download_nltk_models.py \
  download_silera_model.py \
  export_alignment_models.py \
  export_bandit_v2.py \
  export_pyannote_community1.py \
  export_whisper_models.py \
  export_wtpsplit.py \
  ./

# RUN pip install --no-cache-dir -r requirements.txt
