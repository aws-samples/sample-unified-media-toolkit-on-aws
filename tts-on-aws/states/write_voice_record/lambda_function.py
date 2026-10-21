"""
StateWriteVoiceRecord — join state of register-voice.

Assembles the three Parallel branch outputs into one conditioning blob
(applying upstream's mel/token length alignment — mel frames = 2x tokens,
the same hard truncation frontend_zero_shot does), writes it plus the
preview/source derivatives to S3, and registers the DynamoDB record.

Expected event: the Parallel state's output array threaded through
ResultPath, plus the original input:
{
  "uuid": "...",
  "input": {...},                # original register request
  "data": {"ref_info": {...}},   # from StateValidateReference
  "branches": [
    {"staging_key": ".../speech_tokens.json", "n_tokens": N},
    {"spk_embedding": [...192 floats...]},
    {"staging_key": ".../prompt_mel.npy", "mel_frames": M}
  ]
}
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid as uuid_mod

import boto3
import numpy as np

from lib import voice_store
from lib.frontend import save_conditioning

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_s3 = boto3.client("s3")

VOICE_PREFIX = "micro-workflow/tts-on-aws/voices"


def lambda_handler(event: dict, context) -> dict:
    inp = event["input"]
    branches = event["branches"]
    proxy_bucket = os.environ["PROXY_BUCKET"]

    tokens_out, embedding_out, mel_out = branches

    body = _s3.get_object(Bucket=proxy_bucket,
                          Key=tokens_out["staging_key"])["Body"].read()
    speech_tokens = json.loads(body)

    spk_embedding = np.array([embedding_out["spk_embedding"]], dtype=np.float32)

    body = _s3.get_object(Bucket=proxy_bucket,
                          Key=mel_out["staging_key"])["Body"].read()
    prompt_mel = np.load(io.BytesIO(body))

    # upstream frontend_zero_shot alignment: force mel % tokens = 2
    token_len = min(prompt_mel.shape[1] // 2, len(speech_tokens))
    prompt_mel = prompt_mel[:, :2 * token_len, :]
    speech_tokens = speech_tokens[:token_len]

    voice_id = str(uuid_mod.uuid4())
    cond_key = f"{VOICE_PREFIX}/{voice_id}/conditioning.npz"
    preview_key = f"{VOICE_PREFIX}/{voice_id}/preview.wav"
    source_key = f"{VOICE_PREFIX}/{voice_id}/source.wav"

    local_npz = "/tmp/conditioning.npz"
    save_conditioning({"speech_tokens": speech_tokens,
                       "spk_embedding": spk_embedding,
                       "prompt_mel": prompt_mel}, local_npz)
    _s3.upload_file(local_npz, proxy_bucket, cond_key)

    # preview + source: the reference itself (a future preview could be a
    # synthesized sample once a house sample-sentence is chosen)
    voice_meta = inp.get("voice") or {}
    origin = voice_meta.get("origin") or voice_store.USER_UPLOAD_ORIGIN
    license_term = voice_meta.get("license") or voice_store.default_license(origin)
    if not license_term:
        # unknown origin with no explicit license — refuse rather than
        # record a voice with no license term (or silently guess one)
        raise ValueError(
            f"no license term for origin {origin!r} — supply voice.license "
            f"explicitly for origins outside the known-dataset table")
    _s3.copy_object(Bucket=proxy_bucket, Key=preview_key,
                    CopySource={"Bucket": inp["ref_bucket"], "Key": inp["ref_key"]})
    fields = {
        "name": voice_meta.get("name") or voice_id[:8],
        "gender": voice_meta.get("gender"),
        # Always the NORMALIZED whisper code, never the caller's raw string:
        # the gateway's register_voice InputSchema declares language as
        # free-form prose, so an agent can send "English" or "en-US", and this
        # field is compared for EQUALITY downstream (StateCheckAndPreprocess's
        # STYLE_VERIFIED_LANGUAGES == {"en"}, list_voice_collection's language
        # filter) besides being handed to whisper on every QC pass. Storing
        # "English" would silently deny style hints to a genuinely English
        # voice. StateValidateReference does the normalizing (it already has
        # faster-whisper loaded — importing it here would cost this 896 MB
        # tier ~200 MB and ~1.3 s for a lookup table) and leaves both its
        # answer and whisper's own detection in ref_info.
        "language": (event["data"]["ref_info"].get("language")
                     or event["data"]["ref_info"].get("detected_language")),
        "origin": origin,
        "license": license_term,
        "catalog": voice_meta.get("catalog") or "default",
        "comment": voice_meta.get("comment"),
        "ref_transcript": inp["ref_transcript"],
        "s3_conditioning_key": cond_key,
        "s3_preview_key": preview_key,
        "duration_sec": event["data"]["ref_info"]["duration_s"],
    }
    if origin == voice_store.USER_UPLOAD_ORIGIN:
        # audit trail — only for the live user-upload path (§5)
        _s3.copy_object(Bucket=proxy_bucket, Key=source_key,
                        CopySource={"Bucket": inp["ref_bucket"], "Key": inp["ref_key"]})
        fields["s3_source_key"] = source_key

    voice_store.register_voice(voice_id, fields)

    # clean the staging area
    for key in (tokens_out["staging_key"], mel_out["staging_key"]):
        _s3.delete_object(Bucket=proxy_bucket, Key=key)

    log.info("registered voice %s (%d tokens, %d mel frames)",
             voice_id, len(speech_tokens), prompt_mel.shape[1])
    event.setdefault("data", {})["voice"] = {
        "voice_id": voice_id, **{k: v for k, v in fields.items() if v is not None}}
    return event
