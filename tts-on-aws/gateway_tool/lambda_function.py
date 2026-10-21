"""
AgentCore Gateway tool dispatcher for the tts-on-aws micro-workflow.

Six tools (README.md §11):
  synthesize_speech      → StartExecution on the synthesis state machine
  register_voice         → StartExecution on the register-voice state machine
  get_job_status         → DescribeExecution; parses output when complete
  list_voice_collection  → DynamoDB catalog-index query
  get_voice              → DynamoDB single-record fetch
  describe_tool          → static metadata

AgentCore passes the selected tool name in Lambda context:
  context.client_context.custom["bedrockAgentCoreToolName"]
  → "{target_name}___{tool_name}"

workflow_id / idempotency:
  workflow_id = uuid5(canonical JSON of the SFN input + tool version
  + the idempotency token), so identical requests collide on the execution
  name and the existing run is returned instead of paying twice.

  `idempotency_token` (optional, 1-80 chars of A-Z a-z 0-9 _ . -) is the
  caller's assertion "this is the same request I sent before".  The sibling
  workflows dedup on parameters alone, because their `uuid` names the asset
  being processed and the parameters fully determine the output.  This workflow
  has no single asset: it synthesizes text against a MUTABLE voice catalog and,
  absent an explicit `seed`, samples freshly, so the same parameters can
  legitimately need to run again.  So an ABSENT token means "do not deduplicate
  me" — `synthesize_speech` generates one, and the call gets its own run.  A
  caller who DOES want retry safety supplies a stable token and gets it.

  The token rides in the hash preimage as extra material and is DISCARDED —
  never forwarded to the state machine, never stored, never echoed back.  A
  caller that supplied one already has it; the service answers with its own
  name for the run, `workflow_id`.  The retired `uuid` parameter is REJECTED
  here, not ignored: it used to mean both "which asset" and "same request", and
  a caller still sending it cannot be assumed to mean the surviving half.

  That rejection is scoped to the two tools that start an execution.
  `get_job_status`, `list_voice_collection`, `get_voice` and `describe_tool`
  ACCEPT both spellings and merely log: the caller is an LLM, a hallucinated
  parameter name is a routine failure mode, and neither key could have changed
  what a poll or a catalog read returns.

  `register_voice` is deliberately exempt and keeps pure parameter-hash
  dedup: `voice_id` is a fresh uuid4 written with an unconditional
  `put_item` and there is no delete-voice tool, so a resubmitted identical
  registration would create a second permanent, unremovable catalog entry.
  Absorbing that retry is worth more than a caller's ability to force a
  duplicate registration.  See README.md.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid as _uuid_mod

from workflow_identity import (compute_workflow_id,
                               read_idempotency_token,
                               warn_idempotency_token)

import boto3
from botocore.exceptions import ClientError

import callback_registry
import synthesis_outputs

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_sfn = boto3.client("stepfunctions")
_ddb = boto3.client("dynamodb")
_s3 = boto3.client("s3")     # absorbed-terminal outputs.json persistence

# push-completion support (2026-08-31, for audio-description's synth and
# repair loops — bedrock-inference's mechanism, ported): a consumer
# workflow submits with callback_task_token and sleeps on ONE
# waitForTaskToken state; the waker completes the token at this
# machine's terminal event and persists outputs.json first.
CALLBACK_TABLE = os.environ.get("CALLBACK_TABLE", "")

SYNTHESIS_STATE_MACHINE_ARN = os.environ["SYNTHESIS_STATE_MACHINE_ARN"]
REGISTER_VOICE_STATE_MACHINE_ARN = os.environ["REGISTER_VOICE_STATE_MACHINE_ARN"]
VOICE_TABLE_NAME = os.environ["VOICE_TABLE_NAME"]
PROXY_BUCKET = os.environ["PROXY_BUCKET"]
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
# NOTE: no TARGET_NAME read here, unlike every sibling gateway tool. Those
# derive the output prefix as micro-workflow/{TARGET_NAME}/{workflow_id};
# this workflow's states hardcode micro-workflow/tts-on-aws/... instead
# (check_and_preprocess / write_voice_record / extract_speech_*), and the
# SYNTHESIS submit handle mirrors that hardcoded convention when it
# derives output_s3_prefix (added 2026-08-31 for callback consumers).
# cfnstack.yaml still sets TARGET_NAME on the function for cross-stack
# consistency — it is not read.

_TOOL_DELIMITER = "___"
_SFN_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:states:[a-z0-9-]+:\d{12}:execution:[^:]+:[^:]+$")

# The only signal that a submit was absorbed into an existing run rather than
# starting a new one.  Every field except `status` is identical to a fresh
# start's, so without this the caller cannot tell the two apart.  Wording is
# kept byte-identical across every micro-workflow gateway.
_ABSORBED_MESSAGE = (
    "identical request already submitted — this call did not resubmit; "
    "call get_job_status with this job_id for status and results"
)
# ONE s3-uri parser for the zip — synthesis_outputs owns it so the waker
# never blind-slices a prefix beside a validated regex here.
_S3URI_RE = synthesis_outputs.S3URI_RE

MAX_FLOW_STEPS = 10

# ── Output contract ─────────────────────────────────────────────────────────
# output_contract.json ships in this zip (the build script pins the member) and
# is the field-level contract for the INLINE result shapes get_job_status
# returns — the per-transcript outputs[] entries, batch_qc_summary, the
# register_voice voice record — plus the one real file, usage_tracking.json
# (reached via the top-level usage_uri).  Unlike the siblings there is no
# per-label `desc` join into outputs[]: this target has no label vocabulary
# (see _parse_succeeded_output), so describe_tool is the contract's only
# surface.  Static per TOOL_VERSION, loaded once at cold start.  A missing or
# unparseable file must not take submit/status down with it, so it degrades to
# None with an ERROR log; the build script's member check is what makes that
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
# test round-trips the examples through this handler's own validation, so
# an example that drifts from the real contract fails the build. Same
# degradation contract as the output contract: never takes submit/status
# down with it.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "input_examples.json")) as _f:
        _INPUT_EXAMPLES = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("input_examples.json unavailable: %s", _exc)
    _INPUT_EXAMPLES = None


def lambda_handler(event: dict, context) -> dict:
    # keys only — the values can carry callback task tokens and
    # customer-named S3 uris, neither of which belongs in a log group
    logger.info("Event keys: %s", sorted(event.keys()))
    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    # `synthesize_speech` is the ONLY tool here that a token changes anything
    # for.  `register_voice` is in the ignoring set deliberately, not by
    # oversight: it keeps pure parameter-hash dedup because a duplicate
    # registration is a permanent, unremovable catalog entry, so absorbing the
    # retry beats letting a token force one (see the header).  Everywhere else —
    # polls, catalog reads — the field is accepted and IGNORED, never rejected:
    # the caller is an LLM, a hallucinated parameter name is one of its routine
    # failure modes, and failing an otherwise-correct request over a field this
    # tool was always going to discard is the worse trade.  The warning is here
    # so "ignored" is auditable in CloudWatch instead of invisible.
    if tool != "synthesize_speech":
        _token_note = warn_idempotency_token(event)
        if _token_note:
            logger.warning("%s: %s", tool, _token_note)

    if tool == "synthesize_speech":
        return _synthesize_speech(event)
    if tool == "register_voice":
        return _register_voice(event)
    if tool == "get_job_status":
        return _get_job_status(event)
    if tool == "list_voice_collection":
        return _list_voice_collection(event)
    if tool == "get_voice":
        return _get_voice(event)
    if tool == "describe_tool":
        resp = {
            "version": TOOL_VERSION,
            # Mechanical capability flag, NOT a version string to compare:
            # a consumer that will sleep on callback_task_token (e.g.
            # audio-description's ValidateAndPlan) probes THIS before any
            # spend — a deployment without it would silently ignore the
            # token and leave the consumer sleeping to its deadman.
            "push_completion": True,
            "state_machine_arns": [
                {"name": "synthesis", "arn": SYNTHESIS_STATE_MACHINE_ARN},
                {"name": "register_voice", "arn": REGISTER_VOICE_STATE_MACHINE_ARN},
            ],
            "description": (
                "CosyVoice3 text-to-speech Step Functions pipeline (ONNX, CPU): "
                "batch synthesis with zero-shot voice cloning, per-clip ASR QC, "
                "voice catalog with user-registered voices"),
        }
        # Field-level contract for the inline result shapes (per-clip
        # outputs[] entry, batch_qc_summary, voice record) and
        # usage_tracking.json.  Static per TOOL_VERSION, so a caller caches it
        # for the session and CI can pin it without running a job.
        # Deliberately here and not in get_job_status: the shapes are
        # per-version data and would otherwise be re-delivered with every job.
        if _OUTPUT_CONTRACT is not None:
            resp["output_contract"] = _OUTPUT_CONTRACT
        if _INPUT_EXAMPLES is not None:
            resp["input_examples"] = _INPUT_EXAMPLES
        return resp
    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


# ---------------------------------------------------------------------------
# synthesize_speech
# ---------------------------------------------------------------------------

def _synthesize_speech(event: dict) -> dict:
    """
    Required:
      voice_id     registered voice to synthesize with
      transcripts  list of strings (or one string) to synthesize
    Optional:
      flow_steps           6 (default) or up to 10
      seed                 int — reproducible sampling
      target_durations_s   list of floats aligned with transcripts (±10%
                           bounded fit; item fails if outside the bound)
      style                pace/emotion hint applied to every transcript
                           in this batch (e.g. "fast", "slow", "happy",
                           "sad", "angry", "cute", "loud", "soft") — exact
                           name checked against the trained instruct2
                           template set by StateCheckAndPreprocess, not
                           here (see README.md §7a); unknown names
                           fail the workflow, not silently ignored
    """
    voice_id = (event.get("voice_id") or "").strip()
    if not voice_id:
        return _error("voice_id is required", "MISSING_PARAMETER")

    transcripts = event.get("transcripts")
    if isinstance(transcripts, str):
        transcripts = [transcripts]
    if not transcripts or not isinstance(transcripts, list):
        return _error("transcripts is required (string or list of strings)",
                      "MISSING_PARAMETER")
    if not all(isinstance(t, str) and t.strip() for t in transcripts):
        return _error("every transcript must be a non-empty string",
                      "INVALID_PARAMETER")

    flow_steps = event.get("flow_steps", 6)
    if not isinstance(flow_steps, int) or not 1 <= flow_steps <= MAX_FLOW_STEPS:
        return _error(f"flow_steps must be an int 1-{MAX_FLOW_STEPS}",
                      "INVALID_PARAMETER")

    targets = event.get("target_durations_s")
    if targets is not None:
        if not isinstance(targets, list) or len(targets) != len(transcripts):
            return _error("target_durations_s must align 1:1 with transcripts",
                          "INVALID_PARAMETER")

    items = []
    for i, text in enumerate(transcripts):
        item = {"text": text.strip()}
        if targets is not None and targets[i]:
            item["target_duration_s"] = float(targets[i])
        items.append(item)

    config = {"flow_steps": flow_steps}
    if event.get("seed") is not None:
        config["seed"] = int(event["seed"])
    style = event.get("style")
    if style is not None:
        if not isinstance(style, str) or not style.strip():
            return _error("style must be a non-empty string",
                          "INVALID_PARAMETER")
        config["style"] = style.strip()

    # An absent token means "do not deduplicate me": one is generated so the
    # hash is unique and this call gets its own run.  Synthesis reads a mutable
    # voice catalog and (without an explicit seed) samples freshly, so
    # identical parameters do NOT imply an identical result — silently
    # returning an earlier run's audio would be the wrong default.  A caller
    # who wants retry safety supplies a stable token and gets exactly that.
    idempotency_token, token_error = read_idempotency_token(event)
    if token_error:
        return _error(token_error, "INVALID_PARAMETER")

    # callback_task_token: validated LOUDLY, before any submit — a
    # waitForTaskToken caller never reads this function's return value,
    # so failures on the callback path RAISE (bedrock-inference's rule).
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

    sfn_input = {
        "input": {
            "batches": [{"voice_id": voice_id, "transcripts": items}],
            "config": config,
        },
        "data": {},
    }
    return _start_execution(SYNTHESIS_STATE_MACHINE_ARN, sfn_input,
                            idempotency_token or str(_uuid_mod.uuid4()),
                            callback_token=callback_token,
                            with_output_prefix=True)


# ---------------------------------------------------------------------------
# register_voice
# ---------------------------------------------------------------------------

def _register_voice(event: dict) -> dict:
    """
    Required:
      ref_s3uri       s3://bucket/key of the reference WAV (1-30 s, clean speech)
      ref_transcript  exact transcript of the reference audio (validated by
                      ASR at registration — a mismatch is rejected)
    Optional:
      name, gender, language, catalog, comment
      license  license term for the voice; defaults to "IC" (Individual
               Contributor — the uploader attests they hold the rights)
               for this user-upload path
    """
    ref_s3uri = (event.get("ref_s3uri") or "").strip()
    m = _S3URI_RE.match(ref_s3uri)
    if not m:
        return _error("ref_s3uri must be s3://bucket/key", "INVALID_PARAMETER")
    # TtsLambdaRole grants s3:GetObject on ${ProxyBucket}/* ONLY, so a
    # reference in any other bucket cannot be read by StateValidateReference.
    # Without this check the workflow starts, the agent gets a job_id, and
    # the failure surfaces as an opaque AccessDenied a minute later — reject
    # it here, where the caller can still act on it.
    if m.group(1) != PROXY_BUCKET:
        return _error(
            f"ref_s3uri must live in the proxy bucket ({PROXY_BUCKET}); "
            f"got bucket {m.group(1)!r}. Upload the reference audio there "
            "first — this workflow has no read access to other buckets.",
            "INVALID_PARAMETER")
    ref_transcript = (event.get("ref_transcript") or "").strip()
    if not ref_transcript:
        return _error("ref_transcript is required — the exact words spoken in "
                      "the reference audio", "MISSING_PARAMETER")

    sfn_input = {
        "input": {
            "ref_bucket": m.group(1),
            "ref_key": m.group(2),
            "ref_transcript": ref_transcript,
            "voice": {
                "name": event.get("name"),
                "gender": event.get("gender"),
                "language": event.get("language"),
                "catalog": event.get("catalog") or "default",
                "origin": "user-upload",
                "license": event.get("license"),  # None → IC default in write_voice_record
                "comment": event.get("comment"),
            },
        },
        "data": {},
    }
    # No idempotency token, deliberately — unlike synthesize_speech, this
    # dedups on the parameter hash alone, so re-registering the same clip with
    # the same metadata is ABSORBED rather than run again. `write_voice_record`
    # mints voice_id = uuid4() and writes it with an unconditional put_item,
    # and no tool deletes a voice, so a duplicate run means a second permanent
    # catalog entry nobody can remove. Do not add a token here without first
    # adding dedup at the record layer (on the reference ETag + metadata).
    return _start_execution(REGISTER_VOICE_STATE_MACHINE_ARN, sfn_input)


# ---------------------------------------------------------------------------
# get_job_status
# ---------------------------------------------------------------------------

def _get_job_status(event: dict) -> dict:
    job_id = (event.get("job_id") or "").strip()
    if not job_id:
        return _error("job_id is required", "MISSING_PARAMETER")
    if not _SFN_ARN_RE.match(job_id):
        return _error("job_id must be a Step Functions execution ARN",
                      "INVALID_JOB_ID")

    try:
        resp = _sfn.describe_execution(executionArn=job_id)
    except _sfn.exceptions.ExecutionDoesNotExist:
        return _error(f"No execution found: {job_id}", "JOB_NOT_FOUND")
    except ClientError as exc:
        msg = exc.response["Error"]["Message"]
        logger.exception("DescribeExecution failed: %s", msg)
        return _error(f"Failed to describe job: {msg}", "SFN_DESCRIBE_FAILED")

    # The execution NAME is the workflow_id (see _start_execution), so every
    # branch can name the run from the job_id alone — no branch may return a
    # bare status, or an agent that resumed a conversation holding only a
    # job_id would have nothing to report the run by.
    workflow_id = job_id.rsplit(":", 1)[-1] or None
    status = resp["status"]
    if status == "RUNNING":
        result = {"job_id": job_id, "status": "RUNNING"}
        _set(result, "workflow_id", workflow_id)
        return result
    if status in ("FAILED", "TIMED_OUT", "ABORTED"):
        result = {"job_id": job_id, "status": status,
                  "error": resp.get("cause") or resp.get("error") or status}
        _set(result, "workflow_id", workflow_id)
        return result
    if status == "SUCCEEDED":
        result = _parse_succeeded_output(resp)
        result["job_id"] = job_id
        return result
    # Any status Step Functions adds later (e.g. PENDING_REDRIVE) lands here.
    result = {"job_id": job_id, "status": status}
    _set(result, "workflow_id", workflow_id)
    return result


def _parse_succeeded_output(resp: dict) -> dict:
    """The SUCCEEDED flattener — moved to synthesis_outputs (2026-08-31)
    so callback_waker reaches it without importing this module's hard
    env reads. This delegation keeps the gateway's one name for it."""
    return synthesis_outputs.parse_succeeded_output(resp, PROXY_BUCKET)


# ---------------------------------------------------------------------------
# voice catalog
# ---------------------------------------------------------------------------

def _number(raw: str):
    """DDB "N" values arrive as strings. The gateway OutputSchema declares
    duration_sec as Type: number, so hand back a real int/float — a quoted
    "1.86" reads to an agent as a string and breaks any arithmetic on it."""
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


# Stored as DDB "N" (epoch ms) but declared Type: string in the gateway
# OutputSchema — left as a string so the response matches its own contract.
_STRING_TYPED_NUMERIC = frozenset({"created_at"})


def _unmarshal(item: dict) -> dict:
    return {k: (v["S"] if "S" in v else
                (v["N"] if k in _STRING_TYPED_NUMERIC else _number(v["N"]))
                if "N" in v else v.get("BOOL"))
            for k, v in item.items()}

_PUBLIC_FIELDS = ("voice_id", "name", "gender", "language", "origin",
                  "license", "catalog", "comment", "duration_sec",
                  "created_at", "s3_preview_key")


def _public(record: dict) -> dict:
    return {k: record[k] for k in _PUBLIC_FIELDS if record.get(k) is not None}


def _list_voice_collection(event: dict) -> dict:
    catalog = (event.get("catalog") or "default").strip()
    try:
        resp = _ddb.query(
            TableName=VOICE_TABLE_NAME,
            IndexName="catalog-index",
            # "catalog" is a DynamoDB reserved keyword — must be aliased
            KeyConditionExpression="#c = :c",
            ExpressionAttributeNames={"#c": "catalog"},
            ExpressionAttributeValues={":c": {"S": catalog}},
            Limit=min(int(event.get("limit") or 50), 100),
        )
    except ClientError as exc:
        return _error(exc.response["Error"]["Message"], "DDB_QUERY_FAILED")

    voices = [_public(_unmarshal(i)) for i in resp.get("Items", [])]
    gender = event.get("gender")
    language = event.get("language")
    if gender:
        voices = [v for v in voices if v.get("gender") == gender]
    if language:
        voices = [v for v in voices if v.get("language") == language]
    return {"catalog": catalog, "voices": voices, "count": len(voices)}


def _get_voice(event: dict) -> dict:
    voice_id = (event.get("voice_id") or "").strip()
    if not voice_id:
        return _error("voice_id is required", "MISSING_PARAMETER")
    try:
        resp = _ddb.get_item(
            TableName=VOICE_TABLE_NAME,
            Key={"pk": {"S": f"voice#{voice_id}"}, "sk": {"S": "record"}})
    except ClientError as exc:
        return _error(exc.response["Error"]["Message"], "DDB_GET_FAILED")
    item = resp.get("Item")
    if not item:
        return _error(f"No voice found: {voice_id}", "VOICE_NOT_FOUND")
    record = _public(_unmarshal(item))
    if record.get("s3_preview_key"):
        record["preview_uri"] = f"s3://{PROXY_BUCKET}/{record.pop('s3_preview_key')}"
    return record


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

def _start_execution(state_machine_arn: str, sfn_input: dict,
                     idempotency_token: str | None = None,
                     callback_token: str | None = None,
                     with_output_prefix: bool = False) -> dict:
    """Idempotent StartExecution: name = uuid5(canonical input + version).

    `idempotency_token`, when given, is folded into the hash as EXTRA MATERIAL
    rather than into `sfn_input`, so two calls dedup only if their parameters
    AND their token match — and "the token is never forwarded, never stored,
    never echoed" holds by construction instead of by remembering to pop it.
    `synthesize_speech` passes a generated uuid4 when the caller supplied none,
    which makes the hash unique and the run fresh — the deliberate default for
    a workflow whose inputs are not fully captured by its parameters (see the
    module docstring).  `register_voice` passes None and keeps pure
    parameter-hash dedup.
    """
    # `uuid` is excluded because it IS the workflow_id (set just below), so
    # including it would be circular.  Stating it here rather than relying on
    # "assign uuid after hashing" keeps the derivation order-independent: the
    # old invariant was enforced only by statement order, which is invisible to
    # a reader and broken by any edit that moves a line.
    workflow_id = compute_workflow_id(
        sfn_input, tool_version=TOOL_VERSION, exclude=("uuid",),
        extra_material=(idempotency_token,) if idempotency_token else ())
    # The event `uuid` must equal the execution name: the states derive
    # out_prefix (micro-workflow/tts-on-aws/output/{uuid}) from it, so two
    # runs sharing a uuid would overwrite each other's audio.
    sfn_input["uuid"] = workflow_id

    # the synthesis run's service-derived prefix (check_and_preprocess
    # derives the same value from the uuid) — part of the SYNTHESIS
    # handle so a callback consumer knows where outputs.json will land;
    # a voice registration has no output prefix and gets none
    output_s3_prefix = (f"s3://{PROXY_BUCKET}/micro-workflow/tts-on-aws/"
                        f"output/{workflow_id}") if with_output_prefix \
        else None

    # Rebuild the execution ARN from the state machine ARN's own parts —
    # partition included, so this holds in aws-cn/aws-us-gov too (the
    # same partitions _SFN_ARN_RE accepts on the get_job_status path).
    # Derived BEFORE StartExecution (it equals what StartExecution would
    # return: same machine, same name) so the callback row can exist first.
    parts = state_machine_arn.split(":")
    partition, region, account = parts[1], parts[3], parts[4]
    sm_name = parts[-1]
    arn = (f"arn:{partition}:states:{region}:{account}"
           f":execution:{sm_name}:{workflow_id}")

    # "RUNNING", not the "STARTED" this used to return.  A fresh start and
    # an absorbed resubmit now differ only in `status`, so `status` has to
    # come from one vocabulary — Step Functions' own, which every sibling
    # and every poll already speaks.  "STARTED" was a seventh word for a
    # state SFN calls RUNNING.
    handle = {"status": "RUNNING", "job_id": arn,
              "workflow_id": workflow_id}
    if output_s3_prefix:
        handle["output_s3_prefix"] = output_s3_prefix
    if callback_token:
        # Register BEFORE StartExecution — bedrock-inference's settled
        # ordering ("the waker can never fire first"), which the first
        # cut of this port inverted (review 2026-08-31): a run reaching
        # its terminal event before the row exists wakes nobody, ever,
        # and the consumer sleeps to its wait-state deadman. A row
        # registered for a start that then fails is deleted below; a
        # row for an absorbed run is exactly what the waker needs.
        callback_registry.register(_ddb, CALLBACK_TABLE, arn,
                                   callback_token, handle)

    try:
        _sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=workflow_id,
            input=json.dumps(sfn_input))
        return handle
    except _sfn.exceptions.ExecutionAlreadyExists:
        # DESCRIBE the existing run rather than reporting the name collision
        # and stopping.  This used to return {"status": "ALREADY_RUNNING"} with
        # no results, which was wrong twice over: the caller got nothing back
        # for work that was already paid for and possibly finished, and the
        # status word was a claim about the RUN that nothing here had checked —
        # an execution that succeeded hours ago, or failed, both answered
        # "ALREADY_RUNNING".  The status now describes the existing execution,
        # matching the four sibling workflows; `message` carries the one fact
        # the status can't, namely that this call did not resubmit.
        logger.info("Execution already exists: %s", arn)
        try:
            existing = _sfn.describe_execution(executionArn=arn)
        except ClientError as exc:
            msg = exc.response["Error"]["Message"]
            logger.exception("DescribeExecution failed for existing job: %s",
                             msg)
            if callback_token:
                callback_registry.delete(_ddb, CALLBACK_TABLE, arn,
                                         callback_token)
            return _error(
                f"Job already exists but could not retrieve status: {msg}",
                "SFN_DESCRIBE_FAILED")
        # A HANDLE, exactly like a fresh start — only `status` varies, plus
        # `message`.  It deliberately does NOT return the existing run's
        # results, even when that run already SUCCEEDED: this used to inline
        # _parse_succeeded_output(existing), which meant a submit could return
        # a full per-transcript outputs[] the caller never declared or asked
        # for.  Results come from get_job_status and only from there, so a
        # submit's reachable field set equals its declared field set by
        # construction and the agent has one rule to follow instead of two.
        status = existing.get("status", "RUNNING")
        result = {"status": status, "job_id": arn,
                  "workflow_id": workflow_id,
                  "message": _ABSORBED_MESSAGE}
        if output_s3_prefix:
            result["output_s3_prefix"] = output_s3_prefix
        if status not in ("RUNNING", "SUCCEEDED"):
            # The agent needs the reason to choose between "change a parameter"
            # and "resubmit with a different idempotency_token" — a terminal
            # failure is sticky for as long as Step Functions reserves the
            # execution name, so a plain retry cannot clear it.
            result["error"] = (existing.get("cause")
                               or existing.get("error") or status)
        if callback_token and status != "RUNNING":
            # Already terminal: its event predates our row, so the waker
            # will never fire. Persist the verdict ourselves — SUCCEEDED
            # gets the poll shape, a failure gets the failure doc at the
            # SAME pointer (so the woken consumer reads a named cause,
            # never a NoSuchKey) — then complete synchronously and clean
            # up the row registered above. A terminal event that raced
            # in between register and describe makes complete_now a
            # benign TOKEN_GONE and this delete a no-op.
            try:
                if status == "SUCCEEDED":
                    synthesis_outputs.persist_outputs(_s3, existing,
                                                      PROXY_BUCKET)
                elif output_s3_prefix:
                    synthesis_outputs.persist_failure(
                        _s3, output_s3_prefix, status,
                        existing.get("cause") or existing.get("error"),
                        arn)
            except Exception as exc:                       # noqa: BLE001
                logger.error("outputs.json persist failed on absorbed "
                             "run %s: %s — completing the token anyway",
                             arn, exc)
            callback_registry.complete_now(_sfn, callback_token, result)
            callback_registry.delete(_ddb, CALLBACK_TABLE, arn,
                                     callback_token)
        # RUNNING with a token needs nothing here: the row was registered
        # BEFORE the StartExecution attempt, so even a run terminating
        # this instant fires the waker against an existing row — the
        # describe-and-recheck race is closed by the ordering alone.
        return result
    except ClientError as exc:
        msg = exc.response["Error"]["Message"]
        logger.exception("StartExecution failed: %s", msg)
        if callback_token:
            callback_registry.delete(_ddb, CALLBACK_TABLE, arn,
                                     callback_token)
        return _error(f"Failed to start job: {msg}", "SFN_START_FAILED")


def _resolve_tool(context) -> str:
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        if _TOOL_DELIMITER in raw:
            return raw.split(_TOOL_DELIMITER, 1)[1]
        return raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("bedrockAgentCoreToolName not found in Lambda context")
        return ""


def _set(d: dict, key: str, value) -> None:
    """Set `key` only when `value` is not None — omit rather than null, matching
    every sibling gateway_tool."""
    if value is not None:
        d[key] = value


def _error(message: str, code: str = "ERROR") -> dict:
    """Shared error shape across every docker/*/gateway_tool dispatcher —
    `error: True` + `errorCode` + `message`, and always logged. Do not
    change it here alone: the agent prompts and the gateway InputSchema/
    OutputSchema blocks in media-toolkit-gateway/cfnstack.yaml document this
    exact shape, and `error` is a boolean there, not the message string."""
    logger.error("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}
