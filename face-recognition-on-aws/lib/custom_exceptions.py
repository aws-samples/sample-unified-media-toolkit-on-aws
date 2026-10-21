"""
Custom exceptions shared between lambda_function.py (dispatcher) and the
state handlers.

Kept in lib/ rather than lambda_function.py so state modules can import it
at the top of the file. lambda_function.py imports each state module lazily
inside lambda_handler() (one state's dependencies never load another's); a
state importing FROM lambda_function.py at top level would therefore try to
import a module that is still mid-import — a circular-import deadlock. That
was previously worked around with `from lambda_function import
LambdaResourceTimeoutException` inside each handler() function; this module
removes the need for that workaround.
"""

from __future__ import annotations


class LambdaResourceTimeoutException(Exception):
    """
    Raised by a state handler near the Lambda deadline after checkpointing
    partial results to S3. The ASL Retry rule on this error re-invokes the
    state and the handler resumes from the checkpoint, skipping completed
    work.
    """
