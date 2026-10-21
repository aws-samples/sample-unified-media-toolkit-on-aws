"""
StateReleaseSemaphore — delete the holder item ONLY (README.md §5 invariant 1:
the counter is decremented exclusively by the DDB stream handler processing
the holder REMOVE). Idempotent — EventBridge terminal cleanup may race this.
"""

from __future__ import annotations

import logging

from lib import semaphore

logger = logging.getLogger(__name__)


def lambda_handler(event: dict, context) -> dict:
    execution_arn = event["execution_arn"]
    released = semaphore.release_holder(execution_arn)
    logger.info("release %s: %s", execution_arn,
                "released" if released else "already released")
    return {"released": released}
