# Python 3.13

################################################################################
#
# Build for Lambda (linux/arm64). Always pass the platform flag explicitly
# so builds on x86 machines (CI, Intel Mac) produce the correct arch:
#   docker buildx build --platform linux/arm64 -t shot-metadata-enrichment-on-aws .
#
# For testing locally
#   docker run --platform linux/arm64 -it --entrypoint /bin/bash shot-metadata-enrichment-on-aws
#
# The container exists for ONE dependency: pyvips (the reference-face gallery
# composition, PLAN §5). Zip-vendoring the pyvips[binary] wheel was tried
# first and REJECTED (user decision 2026-08-19): the wheel pairing is
# version-sensitive (pyvips 2.1 + pyvips-binary 8.18 imports but fails at
# ffi.callback('VipsCallbackFn') — measured live on the first face-joined
# ladder rung), and the family already has a container pattern that pins the
# pairing in one requirements line. lambda_function.py (the operation
# router) sits at the task root; lib/ and states/ are PACKAGES (the
# family tree shape; imports are package-style, `from lib import x`).
#
################################################################################
FROM public.ecr.aws/lambda/python:3.13.2026.02.18.08


# ── Application code ───────────────────────────────────────────────────────────
COPY requirements.txt ./
COPY lambda_function.py ./
COPY lib/ ./lib/
COPY states/ ./states/

RUN pip install --no-cache-dir -r requirements.txt

USER 1001

CMD [ "lambda_function.lambda_handler" ]
