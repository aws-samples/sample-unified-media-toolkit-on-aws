"""Custom exceptions shared between the dispatcher and state handlers.

Kept in lib/ (same convention as face-recognition-on-aws, where this pattern
comes from) so state modules can import it at the top of the file without
touching the lazily-importing dispatcher.
"""

from __future__ import annotations


class LambdaResourceTimeoutException(Exception):
    """
    Raised by a state handler NEAR the Lambda deadline, after checkpointing
    partial results to S3 — i.e. before the hard kill, so the raise itself is
    the signal that a checkpoint exists. The ASL Retry rule on this error
    name re-invokes the state and the handler resumes from the checkpoint,
    skipping completed work. Step Functions matches the Python exception
    CLASS NAME, so renaming this class silently disables every retry rule
    that names it.
    """
