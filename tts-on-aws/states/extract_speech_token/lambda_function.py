"""
StateExtractSpeechToken — Parallel branch 1 of register-voice.

Reference WAV → discrete speech tokens (speech_tokenizer_v3, the ~924 MB
model — the reason this branch gets its own Lambda tier). Writes tokens
to a per-execution S3 staging key; the join state assembles the blob.
"""

from __future__ import annotations

import json
import logging
import os

import boto3

from lib.audio_io import load_wav
from lib.frontend import extract_speech_tokens
from lib.model_store import ensure_in_tmp

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
TOKENIZER_MODEL_S3_KEY = "_model_artefacts/cosyvoice3/speech_tokenizer/model.tar.gz"
TOKENIZER_TMP_DIR = "/tmp/models/cosyvoice3_speech_tokenizer"

_s3 = boto3.client("s3")


def lambda_handler(event: dict, context) -> dict:
    inp = event["input"]
    uuid = event["uuid"]

    local = "/tmp/register_ref.wav"
    _s3.download_file(inp["ref_bucket"], inp["ref_key"], local)
    audio16 = load_wav(local, 16000)

    ensure_in_tmp(MODEL_BUCKET, TOKENIZER_MODEL_S3_KEY, TOKENIZER_TMP_DIR,
                  "speech-tokenizer-v3")
    tokens = extract_speech_tokens(audio16, TOKENIZER_TMP_DIR)

    staging_key = f"micro-workflow/tts-on-aws/staging/{uuid}/speech_tokens.json"
    _s3.put_object(Bucket=os.environ["PROXY_BUCKET"], Key=staging_key,
                   Body=json.dumps(tokens), ContentType="application/json")
    log.info("extracted %d speech tokens → %s", len(tokens), staging_key)  # nosemgrep: python.lang.security.audit.logging.python-logger-credential-disclosure
    return {"staging_key": staging_key, "n_tokens": len(tokens)}
