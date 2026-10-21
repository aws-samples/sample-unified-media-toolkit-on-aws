"""
StateExtractSpeechFeat — Parallel branch 3 of register-voice.

Reference WAV → matcha mel [1, T, 80] for Flow conditioning. Pure DSP
(lib.dsp.matcha_mel), no model download at all — cheapest branch,
smallest tier.

Staged to S3, not returned inline: a 30 s reference is ~1500 frames x 80
x 4 bytes ≈ 480 KB, over the 256 KB state-payload ceiling.
"""

from __future__ import annotations

import io
import logging
import os

import boto3
import numpy as np

from lib.audio_io import load_wav
from lib.dsp import matcha_mel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_s3 = boto3.client("s3")


def lambda_handler(event: dict, context) -> dict:
    inp = event["input"]
    uuid = event["uuid"]

    local = "/tmp/register_ref.wav"
    _s3.download_file(inp["ref_bucket"], inp["ref_key"], local)
    audio24 = load_wav(local, 24000)

    prompt_mel = matcha_mel(audio24).T[np.newaxis, :, :]  # [1, T, 80]

    staging_key = f"micro-workflow/tts-on-aws/staging/{uuid}/prompt_mel.npy"
    buf = io.BytesIO()
    np.save(buf, prompt_mel)
    _s3.put_object(Bucket=os.environ["PROXY_BUCKET"], Key=staging_key,
                   Body=buf.getvalue(), ContentType="application/octet-stream")
    log.info("extracted prompt mel %s → %s", prompt_mel.shape, staging_key)
    return {"staging_key": staging_key, "mel_frames": int(prompt_mel.shape[1])}
