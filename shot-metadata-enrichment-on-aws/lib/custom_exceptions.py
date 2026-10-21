"""Exceptions with load-bearing NAMES — the ASL retries on them by string.
face-recognition's pattern: the render Map's iterations watch
`get_remaining_time_in_millis()` and raise LambdaResourceTimeoutException
to checkpoint; the state's Retry re-enters and completed windows are
skipped via their `_records/window_NNN.json` side objects."""
from __future__ import annotations


class LambdaResourceTimeoutException(Exception):
    """Deadline guard tripped — partial work is checkpointed; retry me."""


class ShotMetadataValidationError(Exception):
    """Refusal before any spend; the message names what to fix."""


class ShotMetadataInferenceFailed(Exception):
    """The whole bedrock run terminally failed — nothing to salvage."""
