"""
AgentCore Gateway tool dispatcher for the bedrock-inference micro-workflow.

Five tools:
  run_inference    → write input_params.json + request.jsonl, then StartExecution
  get_job_status   → DescribeExecution; parses summary.json when complete
  list_models      → the seeded registry's models, summarized per entry
  get_model        → one model's full registry entry, capabilities and image
                     rules included — "what can this model take" as data
  describe_tool    → version, state machine arn, one-line backend description

AgentCore passes the selected tool name in Lambda context:
  context.client_context.custom["bedrockAgentCoreToolName"]
  → "{target_name}___{tool_name}"   e.g. "bedrock-inference___run_inference"

The event object is a flat dict of that tool's inputSchema properties.

**This dispatcher writes the request objects.** Unlike the other gateway tools, which
pass s3uris through, this one *creates* `{prefix}/input_params.json` and
`{prefix}/request.jsonl` at a prefix derived from the `workflow_id` it just computed.
That ordering is forced: the prefix is `micro-workflow/{target}/{workflow_id}`, so the
hash has to exist before the objects can be written, and the hash is computed over the
request body the caller supplied rather than over its S3 bytes.

`lib/identity.py` is **imported, not reimplemented** — `StateValidateAndPlan` recomputes
the same hash and compares. Two copies of the rule is a hash that silently disagrees,
and the symptom is a resubmit that re-pays for work already done (README.md §3).
That module delegates to the shared `workflow_identity.py` (micro-workflow-bundle/README.md §C2),
whose flat copy also ships at this zip's root for the bundle's parity suite.

Idempotency: the execution is named `workflow_id`, so a resubmit hits
`ExecutionAlreadyExists` and is ABSORBED — the caller gets a handle (status +
`message`), never the existing run's results (micro-workflow-bundle/README.md §C3). An optional
`idempotency_token` narrows the dedup: it folds into the hash and is discarded, so a
fresh token is the caller's rerun mechanism — same prompt, same model, same data,
new run at a new prefix. The retired `uuid` parameter is rejected on submits and
warned-and-ignored on polls.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid

import boto3
from botocore.exceptions import ClientError

import callback_registry
import workflow_identity
from workflow_identity import warn_idempotency_token  # noqa: F401  (bundle contract:
# every gateway exposes the shared warn helper as a module attribute)
from lib import identity, s3_utils
from lib.custom_exceptions import ValidationException
from lib.input_schema import (DEFAULT_MODEL_CHAIN, DEFAULT_SYNC_MODEL_CHAIN,
                              MAX_RECORDS_CEILING,
                              parse_record, parse_records, parse_request)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_s3 = boto3.client("s3")
_ddb = boto3.client("dynamodb")

STATE_MACHINE_ARN = os.environ["INFERENCE_STATE_MACHINE_ARN"]
# The callback-token table (README.md §11): rows this dispatcher registers at
# submit (via callback_registry — the row format's one home) and
# callback_waker completes at the execution's terminal event.
CALLBACK_TABLE = os.environ.get("CALLBACK_TABLE", "")
# Must equal the GatewayTarget Name in media-toolkit-gateway/cfnstack.yaml — it is the
# {target} segment of the derived output prefix.
TARGET_NAME = os.environ.get("TARGET_NAME", "bedrock-inference")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
MODEL_REGISTRY_KEY = os.environ.get(
    "MODEL_REGISTRY_KEY", "_model_configs/availablemodels.json")

_TOOL_DELIMITER = "___"

_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$")
_SFN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")

_MIME_JSON = "application/json"
_MIME_JSONL = "application/jsonl"

# The output contract (gateway_tool/output_contract.json): field-level shapes for every
# outputs[] label, served whole by describe_tool and joined per-entry as `desc`.
# Loaded at cold start; degrades to None with an ERROR log rather than taking
# submit/status down — the build script's zip-member check is what makes that
# state unreachable in a real deployment.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "output_contract.json")) as _f:
        _OUTPUT_CONTRACT = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("output_contract.json unavailable: %s", _exc)
    _OUTPUT_CONTRACT = None

# input_examples.json is the INPUT twin of the output contract: one minimal
# valid request per tool, served by describe_tool so an agent gets a
# known-good starting call instead of guessing shapes from prose. A unit
# test round-trips the submit example through this handler's own
# validation, so an example that drifts from the real contract fails the
# build. Same degradation contract as the output contract: never takes
# submit/status down with it.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "input_examples.json")) as _f:
        _INPUT_EXAMPLES = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("input_examples.json unavailable: %s", _exc)
    _INPUT_EXAMPLES = None


def _contract_desc(label: str):
    if not _OUTPUT_CONTRACT:
        return None
    entry = (_OUTPUT_CONTRACT.get("outputs") or {}).get(label) or {}
    return entry.get("description")


def _output_entry(bucket: str, key: str, label: str, mime: str | None) -> dict:
    """One outputs[] entry — `desc` joined from the contract when the label has
    one, omitted (never fabricated) when it does not."""
    entry = {"uri": f"s3://{bucket}/{key}", "label": label}
    if mime is not None:
        entry["mime"] = mime
    desc = _contract_desc(label)
    if desc:
        entry["desc"] = desc
    return entry


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    # THE raise-vs-return choke point, backported from frame-inspection
    # (review, 2026-09-06): a waitForTaskToken caller never reads a
    # returned error dict, so any refusal that RETURNS while a callback
    # token is present strands the caller to its deadman — fi's per-site
    # convention audit found sites this file had already missed
    # (UNKNOWN_TOOL, staged-records refusals, S3_WRITE_FAILED,
    # SFN_DESCRIBE_FAILED). The discriminator is `error is True` — the
    # _error() REFUSAL shape — never mere truthiness: an absorbed
    # already-FAILED run carries its cause as a STRING under the same
    # key, as DATA beside a job handle, after complete_now already
    # redeemed the token with that verdict.
    result = _dispatch(event or {}, context)
    token = (event or {}).get("callback_task_token")
    if token and isinstance(result, dict) and result.get("error") is True:
        raise RuntimeError(
            f"[{result.get('errorCode')}] {result.get('message')} — "
            "raised instead of returned because a waitForTaskToken "
            "caller never reads a return value")
    return result


def _dispatch(event: dict, context) -> dict:
    logger.info("Event keys: %s", sorted((event or {}).keys()))

    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    if tool == "run_inference":
        return _run_inference(event or {})

    if tool == "get_job_status":
        return _get_job_status(event or {})

    if tool == "list_models":
        return _list_models(event or {})

    if tool == "get_model":
        return _get_model(event or {})

    if tool == "describe_tool":
        resp = {
            "version": TOOL_VERSION,
            "state_machine_arn": STATE_MACHINE_ARN,
            "default_model_chain": _default_chain(),
            "default_sync_model_chain": _default_chain(False),
            "description": (
                "Bedrock inference micro-workflow: takes N already-rendered prompt "
                "records and returns N responses, synchronously under a concurrency "
                "lease or through a Bedrock batch job, with per-record status, model "
                "fallback, and token accounting. Prompts must be rendered before "
                "submitting — this workflow does not resolve prompt templates."),
            # capability advertisement (the fi probe pattern, 2026-09-05): a
            # consumer parking on callback_task_token must be able to ask
            # whether THIS build will ever complete the token, instead of
            # sleeping into its deadman against an older deployment
            "features": (["callback_task_token"] if CALLBACK_TABLE else []),
        }
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# Tool: run_inference
# ---------------------------------------------------------------------------

def _run_inference(event: dict) -> dict:
    """
    Validate the request, write it to S3, and start an execution.

    Required:
      records          List of {record_id, system, messages}. `system` and
                       `messages` must already be rendered — this workflow never calls
                       the prompt registry (§10 rule 2).

    Optional:
      records_s3uri    Staged alternative to inline records, MUTUALLY EXCLUSIVE
                       with them: an s3:// uri to a JSONL (or JSON-array) file
                       under {ProxyBucket}/micro-workflow/*. Use it when the set
                       exceeds the Lambda invoke payload (6 MB — about 20
                       base64 images total). Transport only: the parsed records
                       are hashed and rewritten identically either way, so the
                       same records inline and staged are ONE workflow_id.
      model_chain      List of Bedrock model ids, primary first. Falls back to
                       lib.input_schema.DEFAULT_MODEL_CHAIN. Cross-family fallbacks are
                       preferable: a same-family pair shares a quota pool and shares
                       regional capacity events, so the event the backup exists to
                       survive takes out both.
      use_batch        Optional boolean (§5.6). Absent: the dollar guard decides —
                       batch when the estimated input-cost saving clears
                       BATCH_MIN_SAVINGS, reported in mode_reason. true: batch,
                       downgraded to sync WITH the reason when the model cannot.
                       false: always sync. (`mode` is retired; warned and ignored.)
      max_concurrent   Sync concurrency cap for this model. Default 3.
      temperature      0.0-1.0, default 0.0.
      max_tokens       RETIRED (2026-08-18) — warned and ignored, never
                       rejected. maxTokens is never sent to any model;
                       Bedrock's own default (the model's maximum output)
                       applies. stop_reason `max_tokens` remains a normal
                       outcome at the model's own ceiling.
      thinking         'disabled' (default) | 'adaptive'.
      effort           'low' | 'medium' | 'high', on models that accept it.
      idempotency_token  The caller's assertion "same request as before"
                       (micro-workflow-bundle/README.md §C2). Folded into the hash, then discarded —
                       never forwarded, never echoed. ABSENT means "dedup on the
                       parameter hash": these records are final rendered bytes and
                       fully name the work, so identical parameters are identical
                       work. A FRESH token is the rerun mechanism — same prompt,
                       model and data, new run at a new prefix.

      callback_task_token  A Step Functions task token from a consumer
                       workflow's `waitForTaskToken` submit state (README.md
                       §11). Registered BEFORE StartExecution; the run's
                       terminal event completes it (SendTaskSuccess carrying
                       the terminal status — a FAILED verdict is DATA for the
                       consumer's Choice state). A submit that absorbs into an
                       already-FINISHED run completes the token synchronously
                       in this call. TRANSPORT ONLY — never in the hash, so a
                       waiting and a non-waiting submit of the same records
                       are ONE workflow_id. With this field present, delivery
                       failures RAISE instead of returning an error dict: a
                       waitForTaskToken caller never reads the return value,
                       so a returned error would be a silent hang.

    Returns a HANDLE — {"job_id", "status", "workflow_id", "records_total",
    "output_s3_prefix", "request_s3uri"} — never results (micro-workflow-bundle/README.md §C3). Results
    come from get_job_status alone; which mode ran (and why) is the summary's
    `inference_mode`/`mode_reason`, decided at plan time, not at submit.
    """
    if not PROXY_BUCKET or not TARGET_NAME:
        return _error(
            "PROXY_BUCKET and TARGET_NAME env vars must be set on this function",
            "CONFIG_ERROR")

    # The retired `uuid` is REJECTED here, not ignored: it used to mean both
    # "which asset" and "same request as before", and a caller still sending it
    # cannot be assumed to mean the surviving half — guessing wrong either pays
    # twice or hands back the wrong run. Polls warn instead (see _get_job_status):
    # there the field could not have changed the answer.
    token, token_error = workflow_identity.read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")

    records_in = event.get("records")
    records_uri = (event.get("records_s3uri") or "").strip()
    if records_uri and records_in:
        return _error(
            "records and records_s3uri are mutually exclusive — the request "
            "must have exactly one source of records, or the hash and the "
            "run could disagree about which set names the work",
            "INVALID_PARAMETER")
    if not records_uri:
        if isinstance(records_in, str):
            # An agent that JSON-encoded the list rather than passing it. Accepted, because
            # the alternative is an opaque type error on the caller's most complex field.
            try:
                records_in = json.loads(records_in)
            except ValueError as exc:
                return _error(f"records is a string but not valid JSON: {exc}",
                              "INVALID_PARAMETER")
        if not isinstance(records_in, list) or not records_in:
            return _error(
                "records is required: a non-empty list of "
                "{record_id, system, messages} objects with prompts already "
                "rendered — inline, or staged as records_s3uri when the set is "
                "too large for the invoke payload",
                "MISSING_PARAMETER")

    if (event.get("output_s3_prefix") or "").strip():
        # The prefix is derived from workflow_id, so it cannot be caller-chosen without
        # breaking idempotent resubmit: two callers naming different prefixes for
        # identical work would each expect their own copy.
        logger.warning("output_s3_prefix is deprecated and ignored; outputs are "
                       "written to the service-derived prefix")

    # ── Build input_params.json, then validate it with the WORKFLOW's own parser ──
    # `parse_request` / `parse_records` are the same functions StateValidateAndPlan
    # runs. Validating here means a malformed request is an immediate, actionable tool
    # error rather than a failed execution the agent has to poll to discover.
    request = _build_request(event)
    try:
        # `output` is not known yet — the prefix is derived from a hash of this very
        # request — but `parse_request` requires it. It is excluded from the hash
        # (`identity.EXCLUDED_KEYS`), so the placeholder changes nothing.
        config = parse_request({**request,
                                "output": {"bucket": PROXY_BUCKET,
                                           "prefix": "placeholder"}})
        records = parse_records(records_in) if not records_uri else None
    except ValidationException as exc:
        return _error(str(exc), "INVALID_PARAMETER")

    # ── callback_task_token: validated LOUDLY, before any S3 write ───────────
    # With a token present the caller is by definition a waitForTaskToken
    # workflow, and such a caller NEVER reads this function's return value —
    # so every failure on the callback path RAISES (failing the caller's wait
    # state fast) instead of returning an error dict nobody will see.
    callback_token = event.get("callback_task_token")
    if callback_token is not None:
        if (not isinstance(callback_token, str)
                or not callback_token.strip()
                or len(callback_token) > 2048):
            raise ValueError(
                "callback_task_token must be a non-empty Step Functions "
                "task token string (a waitForTaskToken caller cannot read "
                "a returned error, so this raises)")
        callback_token = callback_token.strip()
        if not CALLBACK_TABLE:
            raise RuntimeError(
                "callback_task_token was supplied but CALLBACK_TABLE is "
                "not configured on this deployment")

    # ── Staged records, ingested LAST among the refusals ─────────────────────
    # Every cheap validation above has passed, so the streaming pass (which
    # writes a staging object) cannot be orphaned by a later parameter error.
    staged = None
    if records_uri:
        records_in, staged, staged_error = _ingest_staged_records(records_uri)
        if staged_error:
            return staged_error
        if staged is None:
            # JSON-array body — the small-set convenience shape, in memory.
            try:
                records = parse_records(records_in)
            except ValidationException as exc:
                return _error(str(exc), "INVALID_PARAMETER")

    # ── workflow_id, then the prefix, then the objects ────────────────────────
    # Hashed from the parsed records rather than the object's ETag: the prefix to write
    # to is derived from the id, so there is no object to head yet. The *parsed* records
    # are what gets written, so StateValidateAndPlan re-parsing them is a no-op and it
    # recomputes the same digest. The token folds in as trailing extra material and is
    # discarded here — it is not in `request`, so it is not in input_params.json either.
    records_digest = (staged["digest"] if staged
                      else identity.digest_records(records))
    records_total = staged["count"] if staged else len(records)
    workflow_id = identity.compute_workflow_id(
        request,
        records_digest=records_digest,
        tool_version=TOOL_VERSION,
        idempotency_token=token)
    if staged is None:
        records_jsonl = "".join(_record_line(record) for record in records)

    prefix = identity.output_prefix(TARGET_NAME, workflow_id)
    request["output"] = {"bucket": PROXY_BUCKET, "prefix": prefix}
    request_key = f"{prefix}/input_params.json"
    request_s3uri = f"s3://{PROXY_BUCKET}/{request_key}"

    try:
        if staged:
            # The streamed pass already wrote the canonical bytes to the
            # staging key (the final key was unknowable before the digest
            # named the prefix); a server-side copy moves them without the
            # gateway ever holding the file.
            _s3.copy_object(
                Bucket=PROXY_BUCKET, Key=f"{prefix}/request.jsonl",
                CopySource={"Bucket": PROXY_BUCKET,
                            "Key": staged["temp_key"]},
                MetadataDirective="REPLACE", ContentType=_MIME_JSONL)
        else:
            _s3.put_object(
                Bucket=PROXY_BUCKET, Key=f"{prefix}/request.jsonl",
                Body=records_jsonl.encode("utf-8"), ContentType=_MIME_JSONL)
        _s3.put_object(
            Bucket=PROXY_BUCKET, Key=request_key,
            Body=json.dumps(request, indent=2, ensure_ascii=False).encode("utf-8"),
            ContentType=_MIME_JSON)
    except ClientError as exc:
        message = exc.response["Error"]["Message"]
        logger.exception("could not write the request objects: %s", message)
        if staged:
            _delete_staged_object(staged["temp_key"])
        return _error(f"Failed to write the request to S3: {message}", "S3_WRITE_FAILED")
    if staged:
        _delete_staged_object(staged["temp_key"])

    # jsonl first, request.json second, and the order matters: StateValidateAndPlan
    # derives the records key from the request key, so an input_params.json with no sibling
    # jsonl is the one combination that fails at read time. Writing it last means it
    # never exists alone.

    # `uuid` here is THIS EXECUTION'S LOCAL ID (the third of the three ids,
    # micro-workflow-bundle/README.md §C2) — always the derived workflow_id, never the caller's token,
    # which was consumed into the hash above and goes no further.
    workflow_input = {
        "operation": "StateValidateAndPlan",
        "uuid": workflow_id,
        "input": {"request_s3uri": request_s3uri},
        "data": {},
    }
    output_s3_prefix = f"s3://{PROXY_BUCKET}/{prefix}"
    logger.info("run_inference: workflow_id=%s records=%d mode=%s output=%s",
                workflow_id, records_total, request.get("mode"), output_s3_prefix)

    # ── Register the callback BEFORE StartExecution ───────────────────────────
    # The execution ARN is derivable from the name we are about to use, so the
    # row exists before the run can possibly finish — the waker can never fire
    # first, with no describe-and-recheck dance. A stale row from a failed
    # start is TTL-cleaned and its execution never emits an event.
    execution_arn = (
        f"{STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}"
        f":{workflow_id}")
    if callback_token:
        callback_registry.register(_ddb, CALLBACK_TABLE, execution_arn,
                                   callback_token, {
            "job_id": execution_arn,
            "workflow_id": workflow_id,
            "records_total": records_total,
            "output_s3_prefix": output_s3_prefix,
            "request_s3uri": request_s3uri,
        })

    try:
        resp = _sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN, name=workflow_id,
            input=json.dumps(workflow_input))
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ExecutionAlreadyExists":
            return _existing_execution(workflow_id, records_total,
                                       output_s3_prefix, request_s3uri,
                                       callback_token=callback_token)
        message = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s — %s", code, message)
        if callback_token:
            # No execution will ever emit an event for this row, and the
            # waiting caller cannot read a returned error — clean up and
            # fail the wait state loudly.
            callback_registry.delete(_ddb, CALLBACK_TABLE, execution_arn,
                                     callback_token)
            raise RuntimeError(
                f"Failed to start the inference job: {message}") from exc
        return _error(f"Failed to start the inference job: {message}",
                      "SFN_START_FAILED")

    return {
        "job_id": resp["executionArn"],
        "status": "RUNNING",
        "workflow_id": workflow_id,
        "records_total": records_total,
        "output_s3_prefix": output_s3_prefix,
        "request_s3uri": request_s3uri,
    }


def _build_request(event: dict) -> dict:
    """
    The `input_params.json` body from flat tool parameters (§4.1).

    Absent keys are **omitted rather than defaulted** to their known defaults. Two
    reasons: `parse_request` is the single place defaults are decided (so the plan and
    the run cannot disagree), and every key written here is inside `workflow_id` — a
    dispatcher that materialises defaults would change the hash for every request the
    day a default changes, re-running work that is already done.
    """
    request: dict = {"tool_version": TOOL_VERSION}

    chain = event.get("model_chain")
    if isinstance(chain, str):
        chain = [part.strip() for part in chain.split(",") if part.strip()]
    if not chain:
        chain = _default_chain(event.get("use_batch"))
    request["model_chain"] = chain

    if event.get("mode") is not None:
        logger.warning("`mode` is retired and was ignored — use `use_batch` "
                       "(true|false|absent); the summary reports what ran")
    if event.get("use_batch") is not None:
        request["use_batch"] = bool(event["use_batch"])
    # Routing scope override (the EU-residency case). In input_params.json and
    # therefore IN the hash — where a run is allowed to execute is part of what
    # the run is, so a residency rerun of a global run is different work.
    if (event.get("inference_region") or "").strip():
        request["inference_region"] = event["inference_region"].strip().lower()
    if event.get("max_concurrent") is not None:
        request["max_concurrent"] = event["max_concurrent"]

    if event.get("max_tokens") is not None:
        # RETIRED 2026-08-18 (user decision: one more parameter is one more
        # maintenance cost) — warned and ignored, never rejected, the same
        # treatment as the retired `mode`. maxTokens is never emitted;
        # Bedrock's model-maximum default applies.
        logger.warning("`max_tokens` is retired and was ignored — the "
                       "model's own maximum output limit applies")
    options = {}
    for key in ("temperature", "effort", "thinking"):
        if event.get(key) is not None:
            options[key] = event[key]
    if event.get("batch_repair") is not None:
        # The §8 repair tail's opt-out: false trades the on-demand-rate sync
        # repair of failed batch records for cheap FAILED lines.
        options["batch_repair"] = bool(event["batch_repair"])
    if options:
        request["options"] = options

    return request


def _default_chain(use_batch=None) -> list:
    """A COPY — the caller writes this into the request it is building, and a shared
    list would let one request's mutation become every later request's default.
    Only an EXPLICIT use_batch: false gets the cheaper sync-priced chain; absent
    (auto) may still resolve to batch, and choose_mode keys on the primary
    model's batch capability, so the batch-capable default must lead there."""
    return list(DEFAULT_SYNC_MODEL_CHAIN if use_batch is False
                else DEFAULT_MODEL_CHAIN)


def _existing_execution(workflow_id: str, records_total: int,
                        output_s3_prefix: str, request_s3uri: str,
                        callback_token: str | None = None) -> dict:
    """
    Idempotent resubmit, absorbed into the existing run — the caller gets a HANDLE
    whose only variable is `status`, never the run's results (micro-workflow-bundle/README.md §C3).

    This used to inline `_parse_succeeded_output(existing)` on the SUCCEEDED
    branch, handing a submit the finished run's whole payload. That shape is what
    the contract exists to forbid — the reachable field set could not be declared,
    and the agent needed two readers for one response — and tts reverted the same
    behaviour on 2026-08-07. `message` carries the one fact a status cannot: this
    call did NOT resubmit. Comparing job_ids cannot replace it — the execution
    name is a hash of the same input either way. A terminal failure additionally
    carries `error`, because "FAILED, and here is why, and a retry will keep
    answering this for ~90 days" is submit-time-actionable; a SUCCEEDED absorb
    carries none, and its results come from get_job_status alone.
    """
    execution_arn = (
        f"{STATE_MACHINE_ARN.replace(':stateMachine:', ':execution:')}:{workflow_id}")
    logger.info("execution already exists for workflow_id=%s: %s",
                workflow_id, execution_arn)
    try:
        existing = _sfn.describe_execution(executionArn=execution_arn)
    except ClientError as exc:
        message = exc.response["Error"]["Message"]
        logger.exception("DescribeExecution failed for the existing job: %s", message)
        if callback_token:
            # fi's StartExecution-failure symmetry (review, 2026-09-06):
            # the raise tells the holder the submit failed, so it will
            # consume its token itself — delete the row THIS call
            # registered, or the waker later fires on a spent token.
            callback_registry.delete(_ddb, CALLBACK_TABLE, execution_arn,
                                     callback_token)
            raise
        return _error(f"Job already exists but could not retrieve status: {message}",
                      "SFN_DESCRIBE_FAILED")

    status = existing.get("status", "RUNNING")
    result = {
        "job_id": execution_arn,
        "status": status,
        "workflow_id": workflow_id,
        "records_total": records_total,
        "output_s3_prefix": output_s3_prefix,
        "request_s3uri": request_s3uri,
        # Byte-pinned across every bundle gateway (the text an agent reads to
        # decide whether it just spent money) — do not reword locally.
        "message": ("identical request already submitted — this call did not "
                    "resubmit; call get_job_status with this job_id for "
                    "status and results"),
    }
    if status != "SUCCEEDED" and status != "RUNNING":
        result["error"] = (existing.get("cause") or existing.get("error")
                           or status)

    if callback_token and status != "RUNNING":
        # Absorbed into a run that already FINISHED: its terminal event fired
        # long ago, so no waker will ever see this token. Complete it NOW —
        # this is the paid-work-reuse path (identical rendered bytes absorb
        # into the finished run; the consumer wakes immediately with the
        # verdict and reads the existing results). RUNNING absorbs need
        # nothing: the row was registered before StartExecution and the
        # terminal event is still coming.
        callback_registry.complete_now(_sfn, callback_token, result)
        callback_registry.delete(_ddb, CALLBACK_TABLE, result["job_id"],
                                 callback_token)
    return result


# ---------------------------------------------------------------------------
# Tool: get_job_status
# ---------------------------------------------------------------------------

def _get_job_status(event: dict) -> dict:
    """
    Poll an inference execution by its arn.

    Required:
      job_id  Step Functions execution arn returned by run_inference

    Returns on RUNNING:   {"status": "RUNNING", "current_state": str, ...}
    Returns on SUCCEEDED: records_total/succeeded/failed, failures[], models_used,
                          bedrock_usage, inference_mode, and a self-describing
                          outputs[] list.

    **`records_failed` is surfaced on a SUCCEEDED run**, not hidden. Partial success is
    success here (§7), so a caller that reads only the status would treat 1/40 as a
    complete result. The failure count and reasons are in the response for that reason.

    The response *bodies* are deliberately not returned: N model outputs are unbounded.
    The agent reads `response_s3uri`.
    """
    # A stray token — either spelling — is warned about and IGNORED on a poll,
    # never rejected: the caller is an LLM, a hallucinated parameter is a routine
    # failure mode, and neither field could change what this poll answers
    # (micro-workflow-bundle/README.md §C2). Submits reject instead, where guessing wrong costs money.
    note = workflow_identity.warn_idempotency_token(event)
    if note:
        logger.warning("get_job_status: %s", note)

    job_id = (event.get("job_id") or "").strip()
    if not job_id:
        return _error("job_id is required", "MISSING_PARAMETER")
    if not _SFN_ARN_RE.match(job_id):
        return _error("job_id must be a Step Functions execution arn", "INVALID_JOB_ID")

    try:
        resp = _sfn.describe_execution(executionArn=job_id)
    except _sfn.exceptions.ExecutionDoesNotExist:
        return _error(f"No execution found: {job_id}", "JOB_NOT_FOUND")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        message = exc.response["Error"]["Message"]
        logger.exception("DescribeExecution failed: %s — %s", code, message)
        return _error(f"Failed to describe job: {message}", "SFN_DESCRIBE_FAILED")

    status = resp["status"]

    if status == "RUNNING":
        result = {"status": "RUNNING", "current_state": _get_current_state(job_id)}
        _attach_ids(result, resp)
        return result

    if status in ("FAILED", "TIMED_OUT", "ABORTED"):
        result = {"status": status,
                  "error": resp.get("cause") or resp.get("error") or status}
        _attach_ids(result, resp)
        return result

    if status == "SUCCEEDED":
        return _parse_succeeded_output(resp)

    # PENDING_REDRIVE and any status Step Functions invents later: still a
    # valid reply, and it still names its own run (bundle poll contract).
    result = {"status": status}
    _attach_ids(result, resp)
    return result


def _parse_succeeded_output(resp: dict) -> dict:
    """
    Extract structured results from a SUCCEEDED execution.

    `StateSummarize` is the final state and returns the whole event, so `resp["output"]`
    carries `data.*`. The summary itself is read from `data`, not re-fetched from S3 —
    one fewer failure mode on the status path.
    """
    result: dict = {"status": "SUCCEEDED"}
    _set(result, "job_id", resp.get("executionArn"))
    try:
        payload = json.loads(resp.get("output") or "{}")
        data = payload.get("data") or {}
        output = data.get("output") or {}
        keys = data.get("keys") or {}

        _set(result, "workflow_id", data.get("workflow_id")
             or (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)
        _set(result, "inference_mode", data.get("mode"))
        _set(result, "mode_reason", data.get("mode_reason"))

        # Partial success is success (§7), so these three are always reported.
        _set(result, "records_total", data.get("records_total"))
        _set(result, "records_succeeded", data.get("records_succeeded"))
        _set(result, "records_failed", data.get("records_failed"))
        _set(result, "records_resumed", data.get("records_resumed"))

        _set(result, "summary_s3uri", data.get("summary_s3uri"))

        # A self-describing outputs[] — labels and basenames are API, so an agent can
        # pick the artifact it wants without knowing this workflow's key layout. The
        # response bodies themselves are NOT inlined: N model outputs are unbounded.
        # `usage` joined the list with the output-contract rollout (micro-workflow-bundle/README.md §C4:
        # a stack that writes usage_tracking.json must surface it, plus usage_uri).
        bucket = output.get("bucket")
        if bucket:
            usage_key = (data.get("usage_tracking") or {}).get("usage_key")
            outputs = [_output_entry(bucket, key, label, mime)
                       for key, label, mime in (
                           (keys.get("response"), "response", _MIME_JSONL),
                           (keys.get("summary"), "summary", _MIME_JSON),
                           (usage_key, "usage", _MIME_JSON))
                       if key]
            if outputs:
                result["outputs"] = outputs
            _set(result, "response_s3uri",
                 f"s3://{bucket}/{keys['response']}" if keys.get("response") else None)
            _set(result, "usage_uri",
                 f"s3://{bucket}/{usage_key}" if usage_key else None)
            _set(result, "output_s3_prefix",
                 f"s3://{bucket}/{output['prefix']}" if output.get("prefix")
                 else None)

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not parse execution output: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Tools: list_models / get_model
# ---------------------------------------------------------------------------

def _load_registry():
    """The seeded artifact, envelope and all — this is a thin READ over the same
    object `StateValidateAndPlan` treats as the allowlist, so what these tools
    advertise and what the plan accepts cannot disagree."""
    import json as _json

    body = _s3.get_object(Bucket=PROXY_BUCKET,
                          Key=MODEL_REGISTRY_KEY)["Body"].read()
    artifact = _json.loads(body)
    if isinstance(artifact, dict):
        return artifact
    return {"registry_version": "", "geo": "", "models": artifact}


def _list_models(event: dict) -> dict:
    """
    The deployment's models, one summary line each. `capabilities` stays out of
    the listing on purpose — an agent choosing a model needs id/type/routing/
    batchability, and the full per-model detail is one `get_model` away rather
    than N copies here.
    """
    note = workflow_identity.warn_idempotency_token(event)
    if note:
        logger.warning("list_models: %s", note)
    try:
        artifact = _load_registry()
    except ClientError as exc:
        return _error(f"could not read the model registry: "
                      f"{exc.response['Error']['Message']}", "REGISTRY_READ_FAILED")
    models = [{
        "model_id": entry.get("model_id") or entry.get("modelId") or "",
        "model_name": entry.get("model_name") or entry.get("modelName") or "",
        "model_type": entry.get("model_type") or entry.get("modelType") or [],
        "inference_region": entry.get("inference_region") or "",
        "supported_regions": entry.get("supported_regions")
                             or entry.get("supportedRegions") or [],
        "can_batch": bool((entry.get("batch_job")
                           or entry.get("batchJob") or {}).get("can_support")
                          or (entry.get("batchJob") or {}).get("canSupport")),
    } for entry in artifact.get("models") or []]
    return {"registry_version": artifact.get("registry_version") or "",
            "geo": artifact.get("geo") or "",
            "models": models,
            "models_total": len(models)}


def _get_model(event: dict) -> dict:
    """
    One model's FULL registry entry — pricing, capabilities, image limits and
    recommendations. This is where a caller preparing grid images reads
    `capabilities.image` (§5.5): `limits` are what validate_plan will reject
    (max_images per request, max_image_mb per inline image, max_payload_mb as
    the record's inline-byte total; a NULL limit means none declared — nothing
    is pre-checked), `downscales_above` degrades silently server-side,
    `recommendations` are quality guidance. Accepts bare or geo-prefixed ids,
    same as the plan does.
    """
    note = workflow_identity.warn_idempotency_token(event)
    if note:
        logger.warning("get_model: %s", note)
    model_id = (event.get("model_id") or "").strip()
    if not model_id:
        return _error("model_id is required", "MISSING_PARAMETER")
    try:
        artifact = _load_registry()
    except ClientError as exc:
        return _error(f"could not read the model registry: "
                      f"{exc.response['Error']['Message']}", "REGISTRY_READ_FAILED")

    from lib import bedrock as bedrock_lib
    try:
        spec = bedrock_lib.resolve_model(artifact.get("models") or [], model_id)
    except bedrock_lib.ModelNotRegistered as exc:
        return _error(str(exc), "MODEL_NOT_REGISTERED")
    return {"registry_version": artifact.get("registry_version") or "",
            "geo": artifact.get("geo") or "",
            "model": spec.config}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach_ids(result: dict, resp: dict) -> None:
    # job_id echoed on EVERY branch (house status contract: a status reply
    # always names its own run) plus workflow_id. The SFN input's `uuid` is the
    # execution's LOCAL id — the same string as workflow_id on every
    # gateway-started run — and reading ids back out of an execution's input is
    # the pattern that produced the correlation_uuid bug this module's history
    # warns about.
    _set(result, "job_id", resp.get("executionArn"))
    _set(result, "workflow_id",
         (resp.get("executionArn") or "").rsplit(":", 1)[-1] or None)


def _get_current_state(execution_arn: str) -> str:
    """
    Name of the currently active state for a RUNNING execution.

    Counts TaskStateEntered against TaskStateExited per state name; more entries than
    exits means still active. `Run inference` re-enters on every checkpoint, so its
    counts climb through a long run — which is why the *lowest* entry count is
    returned, surfacing the outer pipeline stage rather than the re-entry count.
    """
    try:
        events: list = []
        kwargs: dict = {"executionArn": execution_arn, "maxResults": 1000}
        while True:
            page = _sfn.get_execution_history(**kwargs)
            events.extend(page["events"])
            token = page.get("nextToken")
            if not token:
                break
            kwargs["nextToken"] = token

        entered: dict = {}
        exited: dict = {}
        order: list = []
        for item in events:
            if item["type"] == "TaskStateEntered":
                name = item["stateEnteredEventDetails"]["name"]
                if name not in entered:
                    order.append(name)
                entered[name] = entered.get(name, 0) + 1
            elif item["type"] == "TaskStateExited":
                name = item["stateExitedEventDetails"]["name"]
                exited[name] = exited.get(name, 0) + 1

        active = [name for name in order
                  if entered.get(name, 0) > exited.get(name, 0)]
        if not active:
            return "unknown"
        return min(active, key=lambda name: entered[name])

    except Exception as exc:                                       # noqa: BLE001
        logger.warning("Could not determine current state: %s", exc)
        return "unknown"


def _resolve_tool(context) -> str:
    """
    Tool name from the AgentCore client context. The raw value is
    "{target_name}___{tool_name}"; the target prefix is stripped.
    """
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        if _TOOL_DELIMITER in raw:
            return raw.split(_TOOL_DELIMITER, 1)[1]
        return raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("bedrockAgentCoreToolName not found in the Lambda context")
        return ""


def _set(target: dict, key: str, value) -> None:
    """Set `key` only when `value` is not None."""
    if value is not None:
        target[key] = value


_STAGED_CHUNK_BYTES = 1 << 20    # one S3 stream read
_STAGED_PART_BYTES = 8 << 20     # multipart part buffer (S3's minimum is 5 MiB)


def _record_line(record: dict) -> str:
    """ONE canonical request.jsonl line — delegates to s3_utils.jsonl_line,
    the one home for the byte convention: the inline path, the staged path
    and the workflow's own writers MUST serialize identically, or
    staged-vs-inline submissions of the same records would write different
    request.jsonl bytes."""
    return s3_utils.jsonl_line(record) + "\n"


def _ingest_staged_records(uri: str):
    """
    `records_s3uri` → `(records, staged, error)` — exactly one is non-None.

    THE TRANSPORT FOR LARGE RECORD SETS (added 2026-08-14, from review): inline
    `records` ride the Lambda invoke event, which caps the request at 6 MB —
    with base64 images at ~290 KB each that is a ceiling of ~20 images TOTAL
    per request, regardless of record count. A staged file has no such cap;
    `MAX_RECORDS_CEILING` (10k) and Bedrock's own limits take over.

    A JSONL body is STREAMED (2026-08-23): validated record by record, hashed
    incrementally, and multipart-uploaded to a staging key, so memory stays
    bounded by one record plus a part buffer regardless of file size. The
    whole-file path before it pinned this function's 1024 MB at the ceiling
    until the 60 s timeout on a real 175 MB records.jsonl (a 199-minute
    asset's enrichment run) — three-plus full copies of the body coexisted.
    A JSON-array body (the small-set convenience shape) keeps the in-memory
    path; `staged` comes back as {"count", "digest", "temp_key"}. Bound:
    the staging object lands at the final key via server-side CopyObject,
    which caps at 5 GB — far above any record set Bedrock itself accepts.

    The staged object is TRANSPORT ONLY. The parsed records are hashed and
    re-written to the derived prefix exactly as inline records are, so the same
    records submitted inline and staged produce the SAME workflow_id and are
    absorbed into one run — the file's key, bucket and ETag never reach the
    hash, and editing the staged file in place does not re-identify past work.

    Constraints, stated where the caller will read them:
      * the uri must be readable by THIS function's role, whose S3 read grant
        is `{ProxyBucket}/micro-workflow/*` plus the registry key — stage the
        file under `micro-workflow/` in the proxy bucket;
      * the body is JSONL (one record object per line, blank lines ignored) or
        a single JSON array — the same two shapes `records` accepts inline.
    """
    try:
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        if not uri.startswith("s3://") or not bucket or not key:
            raise ValueError
    except ValueError:
        return None, None, _error(
            f"records_s3uri must be a full s3://bucket/key uri, got {uri!r}",
            "INVALID_PARAMETER")
    try:
        body = _s3.get_object(Bucket=bucket, Key=key)["Body"]
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        return None, None, _error(
            f"could not read records_s3uri ({code}): this tool's role reads "
            f"only {PROXY_BUCKET}/micro-workflow/* — stage the records there",
            "S3_READ_FAILED")

    # Peek far enough to see the first non-whitespace byte: it decides the
    # body's shape without committing the whole file to memory.
    carry = b""
    while not carry.strip():
        chunk = body.read(_STAGED_CHUNK_BYTES)
        if not chunk:
            break
        carry += chunk
    if not carry.strip():
        return None, None, _error(f"records_s3uri object is empty: {uri}",
                                  "INVALID_PARAMETER")
    if carry.lstrip().startswith(b"["):
        # JSON-array body: the small-set convenience shape, held in memory.
        text = (carry + body.read()).decode("utf-8", errors="replace")
        try:
            return json.loads(text), None, None
        except ValueError as exc:
            return None, None, _error(
                f"records_s3uri object is neither JSONL nor a JSON array: "
                f"{exc}", "INVALID_PARAMETER")
    return _stream_staged_jsonl(body, carry)


def _stream_staged_jsonl(body, carry: bytes):
    """The streaming half of `_ingest_staged_records`: one pass over a JSONL
    body — per-line `parse_record`, incremental `identity.RecordsDigest` —
    with the multipart mechanics DELEGATED to lib/s3_utils.put_jsonl_streamed
    (ONE writer, one byte convention, abort-on-failure included; review,
    2026-09-06). The final request.jsonl key is derived from the digest this
    pass is still computing, hence the staging key + the server-side copy in
    run_inference. This validation loop stays separate from
    input_schema.iter_parse_records deliberately: it reads an already-open
    body with a peeked carry, not a bucket/key.
    """
    temp_key = (f"micro-workflow/{TARGET_NAME}/_staging/"
                f"{uuid.uuid4().hex}.request.jsonl")
    digest = identity.RecordsDigest()
    seen: set = set()
    count = 0

    def _records():
        nonlocal count
        for line in _iter_staged_lines(body, carry):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValidationException(
                    f"records_s3uri line {count + 1} is not valid JSON "
                    f"(JSONL: one record object per line): {exc}")
            record = parse_record(row, count, seen)
            count += 1
            if count > MAX_RECORDS_CEILING:
                raise ValidationException(
                    f"records_s3uri holds over {MAX_RECORDS_CEILING} "
                    f"records — the ceiling refuses here, before a token "
                    f"is spent")
            digest.update(record)
            yield record
        if not count:
            raise ValidationException(
                "request.jsonl is empty — refusing to start a run")

    try:
        s3_utils.put_jsonl_streamed(PROXY_BUCKET, temp_key, _records(),
                                    part_size=_STAGED_PART_BYTES,
                                    client=_s3)
    except ValidationException as exc:
        return None, None, _error(str(exc), "INVALID_PARAMETER")
    except ClientError as exc:
        # An S3 failure mid-stream gets the same polite dict every other S3
        # failure in this dispatcher returns — raising a raw ClientError
        # would hand a tokenless agent caller an opaque Lambda error. (The
        # writer already aborted its own multipart upload before re-raising.)
        message = exc.response["Error"]["Message"]
        logger.exception("staging upload failed: %s", message)
        return None, None, _error(
            f"Failed to stage the records to S3: {message}",
            "S3_WRITE_FAILED")
    return None, {"count": count, "digest": digest.hexdigest(),
                  "temp_key": temp_key}, None


def _iter_staged_lines(body, carry: bytes):
    """Complete lines from an S3 streaming body, one at a time — the caller's
    peek may already hold any amount of it in `carry`, so the carry is split
    BEFORE the first read. Splitting on b"\\n" is UTF-8-safe (a newline byte
    never occurs inside a multi-byte sequence), so each yielded line decodes
    independently."""
    while True:
        lines = carry.split(b"\n")
        carry = lines.pop()
        for line in lines:
            yield line.decode("utf-8", errors="replace")
        chunk = body.read(_STAGED_CHUNK_BYTES)
        if not chunk:
            break
        carry += chunk
    if carry:
        yield carry.decode("utf-8", errors="replace")



def _delete_staged_object(temp_key: str) -> None:
    try:
        _s3.delete_object(Bucket=PROXY_BUCKET, Key=temp_key)
    except ClientError as exc:
        logger.warning("staging object %s not deleted (%s) — harmless "
                       "but not free; delete it manually", temp_key,
                       exc.response["Error"]["Code"])


def _error(message: str, code: str = "ERROR") -> dict:
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
