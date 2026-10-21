"""
Bedrock batch job submission and output collection (README.md §5.2, §6.2).

Two boto3 clients, and the split is not cosmetic: `CreateModelInvocationJob` /
`GetModelInvocationJob` are on the **control-plane** `bedrock` client, while
`Converse` is on `bedrock-runtime`. Calling one on the other's client fails with an
`UnknownOperationException` that reads like a permissions problem.

Adapted from `core-lib/lib/segmentHelper/models/base.js`
(`makeBatchCommandInput` / `batchInferenceJob` / `getBatchInferenceJobResponses`).
The record shaping itself is in `lib/bedrock.py`, so sync and batch cannot diverge.
"""
from __future__ import annotations

import logging
import os

import boto3
from botocore.config import Config

from lib import bedrock as bedrock_lib
from lib.custom_exceptions import BedrockBatchQuotaExceeded, ValidationException
from lib.input_schema import BATCH_TIMEOUT_HOURS, MIN_BATCH_RECORDS
from lib import s3_utils

logger = logging.getLogger(__name__)

BATCH_ROLE_ARN = os.environ.get("BATCH_SERVICE_ROLE_ARN", "")

_MANIFEST_SUFFIX = "manifest.json.out"

_BEDROCK = None
_RUNTIME = None


def bedrock_control(region: str | None = None):
    """Control plane — Create/Get/StopModelInvocationJob."""
    global _BEDROCK
    if _BEDROCK is None or region:
        client = boto3.client(
            "bedrock", region_name=region,
            config=Config(retries={"max_attempts": 4, "mode": "standard"}))
        if region:
            return client
        _BEDROCK = client
    return _BEDROCK


def bedrock_runtime(region: str | None = None):
    """Data plane — Converse."""
    global _RUNTIME
    if _RUNTIME is None or region:
        client = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(
                # A single Converse call on a large record can take minutes; the
                # default 60 s read timeout turns a slow-but-succeeding call into a
                # retry, which pays for the same tokens twice.
                read_timeout=900, connect_timeout=10,
                retries={"max_attempts": 0}))
        if region:
            return client
        _RUNTIME = client
    return _RUNTIME


def iter_pads(real_count: int, spec: "bedrock_lib.ModelSpec | None"):
    """Yield the pad records that bring `real_count` up to the
    MIN_BATCH_RECORDS floor — the ONE home for the padding rule (the old
    list-materialising `pad_records` wrapper is deleted: its only callers
    were tests, and materialising a padded record set is the OOM class the
    streamed submit path exists to kill). `spec` is
    REQUIRED (None allowed only as a conscious choice): a spec-less pad
    carries no thinking field, which fails a thinking-required batch job
    ALL-OR-NOTHING — the probed opus-5 landmine. The modelInput is shaped
    once and DEEP-COPIED per pad (never shared, so a consumer mutating one
    pad cannot corrupt its siblings)."""
    # NOT a generator function on purpose: the spec-None refusal must fire
    # AT THE CALL, not on the consumer's first pull mid-stream
    count = max(0, MIN_BATCH_RECORDS - real_count)
    if count and spec is None:
        # a spec-less pad omits the thinking field a thinking-required
        # batch model demands on EVERY record, and a batch job validates
        # all-or-nothing — the probed opus-5 loss. Refuse by name; the
        # spec-less minimal shape stays available via padding_record for
        # callers who have decided they want it.
        raise ValueError(
            "iter_pads needs the batch model's ModelSpec — pads must carry "
            "the entry's request contract or a thinking-required batch job "
            "fails all-or-nothing")
    return bedrock_lib.padding_records(count, spec)


def create_job(*, job_name: str, model_id: str, input_s3uri: str,
               output_s3uri: str, region: str | None = None,
               role_arn: str | None = None,
               timeout_hours: int = BATCH_TIMEOUT_HOURS) -> str:
    """
    Submit one job for the whole record set; returns the job arn.

    Job name is `{workflow_id}`, so job identity, row identity, and caller identity
    are one string (§3) and the completion lookup needs no join table.

    Quota-shaped refusals are re-raised as `BedrockBatchQuotaExceeded` and everything
    else propagates unchanged. That classification is the load-bearing part of §6.2:
    queueing a `ValidationException` on a malformed record set retries a permanent
    error forever, invisibly, until the caller's token times out.

    **`output_s3uri` is normalised to a TRAILING SLASH here**, at the API boundary,
    because Bedrock's `outputDataConfig.s3OutputDataConfig.s3Uri` pattern ends in a
    literal `/` and rejects a prefix without one. A real deploy-time failure: the
    submit raised `ValidationException`, which is correctly classified as permanent,
    so the row was left `SUBMITTING` and the caller's token stayed parked for the full
    24 h with the reason visible only in the submit function's own log.

    It is fixed on the producer side rather than at each call site because there are
    two — `submit_batch` and the drain's resubmit, which replays the *persisted*
    `model_params` — so a per-caller fix leaves the drain to rediscover it. The input
    uri is deliberately NOT normalised: that one names a single object, not a prefix.
    `read_job_output` already tolerates either form (it `rstrip('/')`s before joining
    the job id), so this changes nothing downstream.
    """
    role = role_arn or BATCH_ROLE_ARN
    if not role:
        raise ValidationException(
            "BATCH_SERVICE_ROLE_ARN is not set — Bedrock needs a role to read the "
            "input and write the output")
    if not job_name or not model_id or not input_s3uri or not output_s3uri:
        raise ValidationException(
            "job_name, model_id, input_s3uri and output_s3uri are all required")

    params = {
        "jobName": job_name,
        "roleArn": role,
        "modelId": model_id,
        # The parameter that decides which dialect Bedrock validates every
        # record's modelInput in — and whose ABSENCE (the API defaults to
        # "InvokeModel") judged this stack's Converse-shaped records malformed
        # on 2026-08-09: 0/100, `messages.0.content.0.type: Field required`,
        # job "Completed" (§5.4, job 731t09iortbs). Declared here, at the one
        # place jobs are created, so submit_batch and the drain's replay of
        # persisted model_params both get it.
        "modelInvocationType": "Converse",
        "timeoutDurationInHours": int(timeout_hours),
        "inputDataConfig": {
            "s3InputDataConfig": {"s3Uri": input_s3uri, "s3InputFormat": "JSONL"},
        },
        "outputDataConfig": {
            "s3OutputDataConfig": {"s3Uri": output_s3uri.rstrip("/") + "/"},
        },
    }
    try:
        response = bedrock_control(region).create_model_invocation_job(**params)
    except Exception as exc:                                       # noqa: BLE001
        code = bedrock_lib.error_code(exc)
        if code in bedrock_lib.BATCH_QUOTA_EXCEPTIONS:
            raise BedrockBatchQuotaExceeded(f"{code}: {exc}") from exc
        raise
    return response["jobArn"]


def get_job(job_arn: str, region: str | None = None) -> dict:
    return bedrock_control(region).get_model_invocation_job(jobIdentifier=job_arn)


def read_job_output(job_arn: str, output_s3uri: str) -> tuple[dict | None, list]:
    """
    Read a completed job's output: `(manifest, records)`.

    The listing is scoped to the job's **own** subfolder. Bedrock writes to
    `{s3Uri}/{jobId}/`, and listing the shared parent picks up another job's
    partially-written output — a real fix already made in
    `base.js:getBatchInferenceJobResponses`, and one that matters more here because
    every execution of this workflow writes under the same `micro-workflow/` root.

    Padding records are dropped (`recordId` starting `pad_`), which is why
    `parse_records` rejects that prefix on a real record.

    The `.out` objects are STREAMED line by line, never materialised: each
    output row echoes its full `modelInput` — for an image-heavy job that
    echo is the entire request set over again, and holding one `.out` whole
    re-creates the submit path's 668 MB OOM on the collect side, AFTER the
    inference spend. Only the three fields below survive per row.
    """
    bucket, prefix = s3_utils.parse_s3uri(output_s3uri)
    job_id = str(job_arn).split("/")[-1]
    list_prefix = f"{prefix.rstrip('/')}/{job_id}" if job_id and job_id != job_arn \
        else prefix

    manifest, records = None, []
    for key in s3_utils.list_keys(bucket, list_prefix):
        if key.endswith(_MANIFEST_SUFFIX):
            manifest = s3_utils.get_json(bucket, key)
            continue
        if not key.endswith(".out"):
            continue
        for _, row in s3_utils.iter_jsonl(bucket, key):
            record_id = row.get("recordId")
            if not record_id or bedrock_lib.is_padding(record_id):
                continue
            records.append({
                "record_id": record_id,
                "model_output": row.get("modelOutput"),
                # Per-record failure shape:
                # {"errorCode": 400, "errorMessage": "..."}
                "error": row.get("error"),
            })
    return manifest, records
