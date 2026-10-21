"""
The error contract (README.md §8).

Step Functions matches a Lambda failure on `errorType`, which the Python runtime
sets to the **exception class name** — not to a botocore error code. So
`ErrorEquals: ["ThrottlingException"]` can never match a `ClientError`, and every
retry ladder that needs to fire must name a class defined here.

| Class | Retryable in ASL | Meaning |
|---|---|---|
| `ValidationException` | no | malformed request; fail immediately |
| `BedrockBatchQuotaExceeded` | no (parked) | batch submit refused; row queued PENDING, task still parked |
| `LambdaResourceTimeoutException` | yes, ×10 | sync loop checkpointed; re-enter |
| `BedrockInferenceFailed` | no | every record failed (§7) |
| `BatchRecordsIncompleteException` | **never** | control flow: route to the sync repair tail (§8) |

**`BatchRecordsIncompleteException` is CONTROL FLOW, not a fault**, and it is the one
class here that a `Retry` must never name. Step Functions evaluates `Retry` before
`Catch`, so listing it would burn the ladder — and re-read the whole batch output on
each attempt — before the repair it exists to trigger ever fires. Nothing guards that
by convention: every ladder in all three ASLs names only `Lambda.*`-prefixed
integration errors and `LambdaResourceTimeoutException`, and a Python class name can
never collide with the `Lambda.` prefix. `test_no_retry_ladder_names_a_control_flow_exception`
pins it anyway, because `States.ALL` in a ladder would swallow it.

`ValidationException` is **re-exported from `lib.semaphore_client`, not
redefined**. Two classes with the same name in two modules would satisfy
`ErrorEquals` identically, so the duplication would never show up as a failure — it
would show up as an `except ValidationException` that does not catch the one the
semaphore raised.

There is no `QuotaSaturated`. It existed for `semaphore_client.acquire_or_raise`, a
non-parking acquire for callers that could not wait on a task token; that function
had no production callers in the standalone quota stack and none here, and both were
deleted with the merge (README.md §M2.1). A saturated acquire parks on its task
token and is drained in FIFO order — there is no code path in this stack where a full
cap is an error.

Kept in `lib/` rather than in `lambda_function.py` so state modules can import it
at the top of the file: the router imports each state module lazily inside
`lambda_handler`, so a state importing FROM the router at module level would
import a module that is still mid-import.
"""

from __future__ import annotations

from lib.semaphore_client import ValidationException  # noqa: F401


class LambdaResourceTimeoutException(Exception):
    """
    Raised by the sync loop near the Lambda deadline, after checkpointing completed
    records to S3. The ASL Retry on this name re-invokes the state, which refreshes
    the holder and resumes from the checkpoint (README.md §5.1).
    """


class BedrockBatchQuotaExceeded(Exception):
    """
    Bedrock refused a `CreateModelInvocationJob` for a quota-shaped reason.

    Not raised to the caller on the normal path — the row is written `PENDING` and
    the execution stays parked on its task token until a drain submits it (§6.2).
    Defined because the classification is the load-bearing part: anything that is
    NOT one of these must fail the caller immediately rather than being queued and
    retried forever.
    """


class BedrockInferenceFailed(Exception):
    """
    Every record failed. Not a partial result (§7).

    39/40 is a *successful* execution with one FAILED line. Only a 0% success rate
    raises, because an empty `response.jsonl` is indistinguishable from a model that
    legitimately produced nothing.
    """


class BatchRecordsIncompleteException(Exception):
    """
    A batch job came back with a repairable gap: `0 < failed <= REPAIR_MAX_FRACTION x
    total` (README.md §P8). Raised by `Collect batch output` AFTER
    `response.jsonl` and the seeded `_progress.json` are both on S3, and caught by the
    batch child's ASL, which routes to the sync child as a repair pass.

    **A raise rather than a returned flag, deliberately.** The alternative was a
    boolean plus a Choice state, and it was rejected on visibility: a partial batch is
    a run that had a problem, and an exception marks the execution graph permanently
    and legibly where a payload field has to be dug out of the history months later.
    The cost is honest and accepted — a normal outcome renders as a red state in the
    console and emits a `LambdaFunctionsFailed` datapoint on `Collect batch output`.

    It is NOT a failure: the caught path summarizes normally and the execution
    SUCCEEDS. Nothing about the stack's "partial success IS success" contract changes.

    See the module docstring for why this must never appear in a `Retry` ladder.
    """
