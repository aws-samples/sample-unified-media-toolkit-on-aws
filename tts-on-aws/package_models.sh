#!/usr/bin/env bash
# package_models.sh — assemble the verified ONNX models into per-component
# model.tar.gz archives and upload to the model bucket.
#
# Sources: the files already downloaded + verified during local ONNX
# verification (onnx_verify/models/) plus the Qwen2 tokenizer files.
# The faster-whisper model is fetched fresh from HF (Systran/faster-whisper-base
# — the multilingual base, NOT base.en; see lib/qc.py).
#
# S3 layout (matches states/*/lambda_function.py *_MODEL_S3_KEY constants):
#   _model_artefacts/cosyvoice3/llm/model.tar.gz             (~1.9 GB: LLM fp16 + text-embed + tokenizer files)
#   _model_artefacts/cosyvoice3/flow_hift/model.tar.gz       (~960 MB: flow fp16 + hift fp32)
#   _model_artefacts/cosyvoice3/speech_tokenizer/model.tar.gz (~925 MB)
#   _model_artefacts/cosyvoice3/campplus/model.tar.gz        (~27 MB)
#   _model_artefacts/faster-whisper-base/model.tar.gz        (~145 MB)
#
# Usage: ./package_models.sh <model-bucket> [source-dir]
#
# NOTE: onnx_verify/ (the local verification workspace these files were
# originally downloaded/verified into) has been removed from the repo
# — its findings are recorded in README.md, the files themselves were
# scratch/local-only. [source-dir] must now be supplied explicitly
# (re-download/re-verify from scratch, or point at wherever you kept a
# copy) — there is no default location anymore.
set -euo pipefail

BUCKET="${1:?usage: $0 <model-bucket> [source-dir]}"
SRC="${2:?usage: $0 <model-bucket> <source-dir> — no default; see NOTE above}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

pack_and_upload() {
  local name="$1"; shift
  local dir="$STAGE/$name"
  mkdir -p "$dir"
  for f in "$@"; do cp "$f" "$dir/"; done
  tar -czf "$STAGE/$name.tar.gz" -C "$dir" .
  aws s3 cp "$STAGE/$name.tar.gz" "s3://$BUCKET/_model_artefacts/$name/model.tar.gz"
  echo "uploaded: $name ($(du -h "$STAGE/$name.tar.gz" | cut -f1))"
}

# LLM: INT4 backbone pair + INT4 decoder/speech-embed heads (adopted after
# live measurement — FP16 was ~13x slower per decode token on Lambda and
# needed a 7168 MB tier; INT4 passed the linux/arm64 ASR gate, see
# lib/llm.py) + text-embedding + Qwen2 tokenizer files
pack_and_upload cosyvoice3/llm \
  "$SRC/llm_backbone_initial_int4.onnx" \
  "$SRC/llm_backbone_decode_int4.onnx" \
  "$SRC/llm_decoder_int4.onnx" \
  "$SRC/llm_speech_embedding_int4.onnx" \
  "$SRC/text_embedding_fp32.onnx" \
  "$SRC/tokenizer_config.json" \
  "$SRC/vocab.json" \
  "$SRC/merges.txt"

# estimator: official FP32, NOT community FP16 — the FP16 export produced
# platform-dependent garbled output in the Lambda container (see lib/flow.py)
pack_and_upload cosyvoice3/flow_hift \
  "$SRC/flow_token_embedding_fp16.onnx" \
  "$SRC/flow_pre_lookahead_fp16.onnx" \
  "$SRC/flow_speaker_projection_fp16.onnx" \
  "$SRC/flow.decoder.estimator.fp32.onnx" \
  "$SRC/hift_f0_predictor_fp32.onnx" \
  "$SRC/hift_source_generator_fp32.onnx" \
  "$SRC/hift_decoder_fp32.onnx"

pack_and_upload cosyvoice3/speech_tokenizer \
  "$SRC/speech_tokenizer_v3.onnx"

pack_and_upload cosyvoice3/campplus \
  "$SRC/campplus.onnx"

# faster-whisper base (multilingual) — fetch via huggingface_hub.
# Tar the directory directly (snapshot_download leaves .cache metadata
# subdirs that the flat pack_and_upload cp can't handle; exclude them).
WHISPER_DIR="$STAGE/whisper_src"
python3 - "$WHISPER_DIR" << 'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-base", local_dir=sys.argv[1])
EOF
tar -czf "$STAGE/faster-whisper-base.tar.gz" -C "$WHISPER_DIR" \
  --exclude='.cache' --exclude='.gitattributes' .
aws s3 cp "$STAGE/faster-whisper-base.tar.gz" \
  "s3://$BUCKET/_model_artefacts/faster-whisper-base/model.tar.gz"
echo "uploaded: faster-whisper-base ($(du -h "$STAGE/faster-whisper-base.tar.gz" | cut -f1))"

echo "all model archives uploaded to s3://$BUCKET/_model_artefacts/"
