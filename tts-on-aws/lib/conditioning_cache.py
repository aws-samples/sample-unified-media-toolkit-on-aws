"""
Voice-conditioning /tmp cache — S3 npz blob → loaded dict, warm-reused.

The plan's original StateLoadVoiceConditioning ("fetch conditioning once
per iterator") can't exist as a separate Step Functions state: the ~1 MB
blob exceeds the 256 KB state-payload limit, so a loader state could only
pass a pointer — pure overhead. Instead both synthesis states call
get_conditioning() directly: S3 GET on first use per container, /tmp +
in-memory cache after. Same net effect (one fetch per batch on a cold
container, zero on warm), one less state.
"""

from __future__ import annotations

import logging
import os

import boto3

from lib.frontend import load_conditioning

log = logging.getLogger(__name__)

_s3 = boto3.client("s3")
_cache: dict[str, dict] = {}

_TMP_DIR = "/tmp/voice_conditioning"


def get_conditioning(bucket: str, key: str) -> dict:
    """Fetch + load a conditioning npz, cached per container."""
    if key in _cache:
        return _cache[key]

    os.makedirs(_TMP_DIR, exist_ok=True)
    local = os.path.join(_TMP_DIR, key.replace("/", "_"))
    if not os.path.exists(local):
        log.info("Fetching conditioning s3://%s/%s", bucket, key)
        _s3.download_file(bucket, key, local)

    cond = load_conditioning(local)
    _cache[key] = cond
    return cond
