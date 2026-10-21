"""
Batch-resume checkpointing for the per-transcript loops in
StateSynthesizeLLM / StateSynthesizeFlowHift / StateVerifyOutput.

A batch can hold up to MAX_TRANSCRIPTS_PER_BATCH=100 transcripts
(check_and_preprocess.py); at measured per-transcript cost (~20-24s)
that is up to ~2400s of work inside a 900s Lambda — timeout is the
expected outcome at that batch size, not an edge case.

Pattern (mirrors face-recognition-on-aws's StateClusterIdentities /
StateSynthesizeLLM et al.): before each unit of work, check the
Lambda's remaining time; if under REMAINING_TIME_BUFFER_MS, persist
partial progress to S3 and raise LambdaResourceTimeoutException. The
ASL Retry rule on that exception re-invokes the SAME state, which loads
the checkpoint and resumes — completed transcripts are never redone.

Checkpoint key is job_id + batch_index + state name, all stable across
retries of the same batch (set once by check_and_preprocess.py and
passed through item_data unchanged) — never collides across batches or
states.
"""

from __future__ import annotations

from lib.s3_utils import delete_object, get_json_if_exists, put_json

REMAINING_TIME_BUFFER_MS = 60_000


def checkpoint_key(out_prefix: str, state_name: str, job_id: str, batch_index: int) -> str:
    return f"{out_prefix.rstrip('/')}/ephemeral/{state_name}_checkpoint_{job_id}_{batch_index}.json"


def load(bucket: str, key: str) -> dict | None:
    return get_json_if_exists(bucket, key)


def save(bucket: str, key: str, data: dict) -> None:
    put_json(bucket, key, data)


def clear(bucket: str, key: str) -> None:
    """Best-effort — S3 delete_object on an already-absent key is a
    no-op success (not an error), so no existence check is needed."""
    delete_object(bucket, key)
