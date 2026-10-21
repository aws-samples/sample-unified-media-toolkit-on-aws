"""
Custom exceptions shared between lambda_function.py (dispatcher) and the
state handlers. Same rationale as face-recognition-on-aws's module of the
same name: kept out of lambda_function.py so state modules can import it
at top level without a circular-import deadlock (lambda_function.py
imports each state module lazily inside lambda_handler()).
"""

from __future__ import annotations


class LambdaResourceTimeoutException(Exception):
    """
    Raised by a state handler near the Lambda deadline after checkpointing
    partial batch results to S3. The ASL Retry rule on this error
    re-invokes the state and the handler resumes from the checkpoint,
    skipping already-completed transcripts.
    """
