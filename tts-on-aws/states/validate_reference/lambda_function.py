"""
StateValidateReference — first state of the register-voice workflow.

Two checks before any extraction work runs:

  1. Format contract: WAV, 8-48 kHz, 1-30 s (lib.audio_io.validate_ref_wav).
  2. Transcript match: ASR-transcribe the reference and fuzzy-compare
     against the caller's ref_transcript. A mismatched transcript doesn't
     error downstream — it silently corrupts every synthesis against the
     voice (demonstrated during local ONNX verification: word-salad
     output). Reject at the only moment the caller can fix it.

Expected input:
{
  "operation": "StateValidateReference",
  "uuid": "...",
  "input": {
    "ref_bucket": "...", "ref_key": "...",
    "ref_transcript": "...",           # what is spoken in the reference
    "voice": {"name": "...", "gender": "...", "language": "en",
              "catalog": "default", "origin": "user-upload",
              "comment": "..."}
  }
}
"""

from __future__ import annotations

import logging
import os

import boto3

from lib import qc
from lib.audio_io import validate_ref_wav
from lib.model_store import ensure_in_tmp

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
WHISPER_MODEL_S3_KEY = "_model_artefacts/faster-whisper-base/model.tar.gz"
WHISPER_TMP_DIR = "/tmp/models/faster_whisper_base"

_s3 = boto3.client("s3")


def lambda_handler(event: dict, context) -> dict:
    inp = event["input"]
    ref_transcript = (inp.get("ref_transcript") or "").strip()
    if not ref_transcript:
        raise ValueError("input.ref_transcript is required — it is fed to the "
                         "model as ground truth for the reference audio")

    local = "/tmp/register_ref.wav"
    _s3.download_file(inp["ref_bucket"], inp["ref_key"], local)
    info = validate_ref_wav(local)

    ensure_in_tmp(MODEL_BUCKET, WHISPER_MODEL_S3_KEY, WHISPER_TMP_DIR,
                  "faster-whisper-base")
    # Normalize here, once, and hand the result to StateWriteVoiceRecord in
    # ref_info: this is the only state in the register-voice pipeline that
    # already has faster-whisper loaded, and the persisted voice record must
    # carry a real whisper code (see that state's `language` field for why).
    language = qc.normalize_language((inp.get("voice") or {}).get("language"))
    check = qc.verify_content(local, ref_transcript, WHISPER_TMP_DIR,
                              language=language,
                              threshold=qc.DEFAULT_REGISTER_THRESHOLD)
    if not check["passed"]:
        raise ValueError(
            f"reference transcript does not match the reference audio "
            f"(similarity {check['similarity']} < {check['threshold']}). "
            f"Supplied: {ref_transcript!r}. Heard: {check['transcribed']!r}. "
            "A mismatched transcript silently corrupts every synthesis "
            "against this voice — fix the transcript or the audio.")

    event.setdefault("data", {})["ref_info"] = {
        **info,
        "transcript_similarity": check["similarity"],
        # "language" is the caller's request normalized to a whisper code
        # (None if they gave none or gave something unrecognizable);
        # "detected_language" is what whisper actually heard. The record
        # prefers the former and falls back to the latter.
        "language": language,
        "detected_language": check["detected_language"],
    }
    log.info("reference validated: %.1fs %dHz, transcript sim %.2f",
             info["duration_s"], info["samplerate"], check["similarity"])
    return event
