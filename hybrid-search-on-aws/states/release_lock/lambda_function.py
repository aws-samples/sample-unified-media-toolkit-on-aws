"""
StateReleaseLock — release ONE physical table's lock (sk = index_key).

Final per-iteration state in the Index SFN Map and Rebuild SFN Map,
including on error via Catch. The DDB stream REMOVE event triggers
HandleLockRelease, which resumes the oldest waiter queued for THIS lock.
"""
from __future__ import annotations

import logging
import traceback

from lib.lock_utils import release_if_owner

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    index_key     = event.get("index_key", "")
    execution_arn = event.get("execution_arn", "")
    if not index_key:
        raise ValueError("StateReleaseLock: index_key is required")

    try:
        release_if_owner(index_key, execution_arn)
        return {**event, "lock_released": True}
    except Exception as exc:
        log.error("StateReleaseLock failed — %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise
