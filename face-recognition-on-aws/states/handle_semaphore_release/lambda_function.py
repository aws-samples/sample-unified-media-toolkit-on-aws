"""
StateHandleSemaphoreRelease — DDB stream handler (README.md §5).

Triggered by REMOVE events on holder items (EventSourceMapping FilterCriteria:
eventName=REMOVE, pk="semaphore", sk begins_with "holder#"). All three holder
deletion paths (normal release, EventBridge terminal cleanup, TTL expiry)
converge here.

Per record: exactly-once counter decrement (transactional dedup marker on the
stream record's eventID — streams are at-least-once), then FIFO drain of
pending waiters into freed slots.
"""

from __future__ import annotations

import logging

from lib import semaphore

logger = logging.getLogger(__name__)


def lambda_handler(event: dict, context) -> dict:
    decremented = 0
    for record in event.get("Records") or []:
        event_id = record.get("eventID") or ""
        if semaphore.decrement_once(event_id):
            decremented += 1
    # Drain UNCONDITIONALLY — not only when this invocation decremented.
    # Streams retry the same records after a crash: the retry's decrements
    # all dedup to False, but the crashed attempt may have died mid-drain
    # (or before draining at all) with free slots and queued waiters. If the
    # removed holder was the LAST active one, no future REMOVE will ever fire
    # — skipping the drain here would strand every waiter until its 3600 s
    # acquire heartbeat kills the workflow. drain_pending on a full/empty
    # semaphore is one cheap Query returning 0.
    admitted = semaphore.drain_pending()
    logger.info("stream: %d decremented, %d admitted", decremented, admitted)
    return {"decremented": decremented, "admitted": admitted}
