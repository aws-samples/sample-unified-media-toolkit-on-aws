"""
StateExtractSpeakerEmbedding — Parallel branch 2 of register-voice.

Reference WAV → [1, 192] CAMPPlus x-vector (27 MB model — smallest tier).
Embedding is small enough to return inline (192 floats ≈ 4 KB JSON).
"""

from __future__ import annotations

import logging
import os

import boto3

from lib.audio_io import load_wav
from lib.frontend import extract_spk_embedding
from lib.model_store import ensure_in_tmp

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
CAMPPLUS_MODEL_S3_KEY = "_model_artefacts/cosyvoice3/campplus/model.tar.gz"
CAMPPLUS_TMP_DIR = "/tmp/models/cosyvoice3_campplus"

_s3 = boto3.client("s3")


def lambda_handler(event: dict, context) -> dict:
    inp = event["input"]

    local = "/tmp/register_ref.wav"
    _s3.download_file(inp["ref_bucket"], inp["ref_key"], local)
    audio16 = load_wav(local, 16000)

    ensure_in_tmp(MODEL_BUCKET, CAMPPLUS_MODEL_S3_KEY, CAMPPLUS_TMP_DIR,
                  "campplus")
    embedding = extract_spk_embedding(audio16, CAMPPLUS_TMP_DIR)

    log.info("extracted speaker embedding %s", embedding.shape)
    return {"spk_embedding": embedding.flatten().tolist()}
