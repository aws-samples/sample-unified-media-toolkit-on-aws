# Python 3.13

################################################################################
#
# Build for Lambda (linux/arm64). Always pass the platform flag explicitly
# so builds on x86 machines (CI, Intel Mac) produce the correct arch:
#   docker buildx build --platform linux/arm64 -t face-recognition-on-aws .
#
# For testing locally
#   docker run --platform linux/arm64 -it --entrypoint /bin/bash face-recognition-on-aws
#
# No onnxruntime / no local models: all ML is Amazon Rekognition APIs.
# No ORT cpuinfo patch needed. Imaging is pyvips (libvips bundled in the
# pyvips[binary] wheel).
#
################################################################################
FROM public.ecr.aws/lambda/python:3.13.2026.02.18.08


# ── Application code ───────────────────────────────────────────────────────────
COPY requirements.txt lambda_function.py ./
COPY lib/ ./lib/
COPY states/ ./states/

RUN pip install --no-cache-dir -r requirements.txt

USER 1001

CMD [ "lambda_function.lambda_handler" ]
