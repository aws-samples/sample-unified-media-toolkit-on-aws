"""
StateValidateAndPlan — the single entry state (README.md §5).

Everything that can fail deterministically fails here, **before a token is spent**:
the request parses, the records parse, every model in the chain is registered, and
the mode is chosen and *reported*.

Two structural properties:

- **It runs before the lease, and now in a different machine from it.** Acquire is
  late on purpose (§5, semaphore invariant: plan outside the span): S3 reads, record
  parsing, and planning happen while queued rather than while holding a slot, so
  queued work still makes progress. The merge hardened this from a state ordering
  into a machine boundary — this state is the ORCHESTRATOR's, and `Acquire lease` is
  the sync child's first state (README.md §M decision 1), so no future edit can
  reorder planning into the held span.
- **It recomputes `workflow_id` and compares.** The gateway tool computed it to name
  the execution; recomputing here from the same module (`lib/identity.py`) proves the
  request at that prefix is the request the id was derived from. A mismatch means the
  request object was overwritten after submission, in which case the output prefix
  would attribute one run's result to another run's inputs.
"""
from __future__ import annotations

import logging

from lib import bedrock as bedrock_lib
from lib import identity, s3_utils
from lib.custom_exceptions import ValidationException
from lib.input_schema import (
    AWS_REGION,
    MODEL_REGISTRY_KEY,
    PROXY_BUCKET,
    QUOTA_SERVICE,
    TARGET_NAME,
    TOOL_VERSION,
    choose_mode,
    iter_parse_records,
    parse_request,
    record_text_chars,
    records_key,
    request_pointer,
    tokens_from_chars,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    bucket, request_key, prefix = request_pointer(event)

    request = s3_utils.get_json(bucket, request_key)
    config = parse_request(request)

    jsonl_key = records_key(prefix)

    # The registry is the actual allowlist: Bedrock IAM cannot be scoped to the
    # chosen model, because a cross-region inference profile needs the profile ARN
    # plus the foundation-model ARN in every region it routes to, none of them
    # knowable at deploy time. An unregistered model raises here rather than running
    # uncosted.
    registry = bedrock_lib.load_registry(
        s3_utils.s3, PROXY_BUCKET, MODEL_REGISTRY_KEY)
    chain = bedrock_lib.build_chain(registry, config["model_chain"])
    primary = chain[0]

    # Request-time routing override (README.md §P5.1) — the
    # EU-residency case: a model may support `global` while local law requires
    # the run to stay regional. The override must be LITERALLY in the model's
    # supported_regions; refusing loudly here is the legally-correct answer,
    # never silently running global. Checked against EVERY model in the chain:
    # a fallback that cannot honour the residency constraint is not a fallback.
    inference_region = (request.get("inference_region") or "").strip()
    if inference_region:
        for spec in chain:
            if not spec.allows_inference_region(inference_region):
                raise ValidationException(
                    f"inference_region {inference_region!r} is not in "
                    f"{spec.model_id}'s supported regions "
                    f"{spec.supported_regions} — the request cannot be routed "
                    "there. Drop the override or choose a model that supports "
                    "that region")

    # ONE streamed pass over request.jsonl (iter_parse_records owns the
    # seen-set/ceiling/emptiness policy): image limits, token chars, the
    # records digest, and the count — the record set is never
    # materialised. This state held 2651 MB for a measured 668 MB
    # request (live, 2026-09-06) as the LAST materialiser on the batch
    # path after submit and collect were streamed; a ~1 GB request would
    # have OOMed the run at plan time, before any spend.
    image_limits = _declared_limits(primary)
    # Limits are enforced against the PRIMARY only — a fallback is a
    # per-record escalation nothing at plan time can predict, and refusing
    # every record for a limit that binds only on escalation would be a
    # capability regression. The honest middle: name the exposure once per
    # plan — but only when the run actually CARRIES images (the dominant
    # workload is text-only; a warning on every such plan trains operators
    # to ignore it on the one run where it matters). An escalated
    # over-limit record fails at Bedrock as a FAILED line —
    # ValidationException never advances the chain.
    stricter_notes = _stricter_fallback_notes(chain, image_limits)
    records_digest = identity.RecordsDigest()
    chars = 0
    records_total = 0
    for record in iter_parse_records(bucket, jsonl_key):
        if stricter_notes and _record_has_image(record):
            for note in stricter_notes:
                logger.warning("%s", note)
            stricter_notes = []
        _enforce_image_limit(record, primary, image_limits)
        chars += record_text_chars(record)
        records_digest.update(record)
        records_total += 1

    # The dollar guard (§5.6): the saving is estimated input cost x the batch
    # discount, both operands already in hand — the records are final rendered
    # bytes and the registry pricing is loaded. None for an unpriced model, so
    # choose_mode falls back to token volume rather than reading "no price" as
    # "no saving".
    est_tokens = tokens_from_chars(chars)
    in_cost = bedrock_lib._get(primary.pricing, "input_per_token_cost",
                               "inputPerTokenCost", 0.0)
    est_saving = (est_tokens * in_cost * (1 - bedrock_lib.BATCH_COST_MULTIPLIER)
                  if in_cost else None)
    mode, mode_reason = choose_mode(
        config["use_batch"], primary.can_batch(),
        estimated_saving=est_saving, estimated_input_tokens=est_tokens)
    logger.info("mode=%s (%s) for %d records on %s",
                mode, mode_reason, records_total, primary.model_id)

    # Hashed from the *parsed* records, not from the object's ETag: the gateway tool
    # has to know `workflow_id` before it can write the object (the prefix is derived
    # from the id), and an ETag is not the body MD5 under SSE-KMS. See
    # `lib/identity.py`'s docstring.
    #
    # This recompute is the PARAMETER hash — no idempotency_token, because the token
    # never leaves the gateway. The run's authoritative workflow_id is therefore the
    # EXECUTION NAME (the gateway derived it, token folded in), and the recompute is
    # the integrity check against it: on a token-less submit the two are equal by
    # construction, and on a tokened one they differ by exactly the token's material.
    # The distinction is load-bearing downstream, not cosmetic — `data.workflow_id`
    # becomes the batch JOB NAME and the token-table SORT KEY, so two runs that
    # differ only by token must carry two different ids or the second run's
    # `put_submitting` overwrites the first's parked task token.
    parameter_hash = identity.compute_workflow_id(
        request,
        records_digest=records_digest.hexdigest(),
        tool_version=TOOL_VERSION)
    workflow_id = _authoritative_workflow_id(event, parameter_hash)

    output = config["output"]
    return {
        "operation": "StateValidateAndPlan",
        "status": "COMPLETED",
        "uuid": event.get("uuid") or "",
        "input": event.get("input") or {},
        "data": {
            "workflow_id": workflow_id,
            "tool_version": TOOL_VERSION,
            "target_name": TARGET_NAME,
            "mode": mode,
            "mode_reason": mode_reason,
            "request": {
                "bucket": bucket,
                "control_key": request_key,
                "records_key": jsonl_key,
                "s3uri": f"s3://{bucket}/{request_key}",
            },
            "output": output,
            "config": {
                "model_chain": [spec.model_id for spec in chain],
                # The resolved routing scope, default or request-overridden —
                # "" means each spec's own default. Both children hand it to
                # build_converse_params/resolve_batch_model_id, so sync and
                # batch route identically and neither re-derives it.
                "inference_region": inference_region,
                "max_concurrent": config["max_concurrent"],
                "temperature": config["temperature"],
                "effort": config["effort"],
                "thinking": config["thinking"],
                "batch_repair": config["batch_repair"],
            },
            # The lease scope. `resource` is the PRIMARY model's RUNTIME id —
            # runtime, not bare, because `global.X` and `eu.X` draw on different
            # service quota pools, so a per-region override gets its own cap for
            # free. The chain's fallbacks are per-record escalations inside one
            # held slot, not separate leases, so a chain does not multiply it.
            "quota": {
                "service": QUOTA_SERVICE,
                "resource": primary.resolve_runtime_model_id(
                    inference_region or None),
                "max_concurrent": config["max_concurrent"],
            },
            "batch": {
                "can_batch": primary.can_batch(),
                # Batch FOLLOWS inference_region (settled 2026-08-11): a run
                # pinned to `eu` for residency must not have its batch half
                # silently route global. Same scope, same resolution as sync.
                "batch_model_id": (bedrock_lib.resolve_batch_model_id(
                                       primary, inference_region or None)
                                   if primary.can_batch() else ""),
                # The scope batch actually submits under (override wins, else
                # the model's own — which on a legacy registry entry is the old
                # batchJob.region). A routing NAME, never handed to boto3 as-is;
                # `batch_region` is the concrete boto3 region for the batch
                # control-plane client — a scoped override routes there,
                # global/geo run from the Lambda's own.
                "batch_geo": (inference_region or primary.batch_scope or ""),
                "batch_region": (primary.resolve_client_region(
                                     inference_region or None) or AWS_REGION),
            },
            "records_total": records_total,
            "keys": {
                "progress": f"{output['prefix']}/_progress.json",
                "response": f"{output['prefix']}/response.jsonl",
                "summary": f"{output['prefix']}/summary.json",
                "batch_input": f"{output['prefix']}/batch/input.jsonl",
                "batch_output_prefix": f"{output['prefix']}/batch/output",
            },
        },
    }


def _stricter_fallback_notes(chain, image_limits) -> list:
    """One note per fallback whose DECLARED limits are stricter than the
    primary's — emitted lazily, on the first image-bearing record, by the
    caller's streaming loop."""
    notes = []
    for fallback in chain[1:]:
        fb_limits = _declared_limits(fallback)
        stricter = [key for key in ("max_images", "max_image_mb",
                                    "max_payload_mb")
                    if fb_limits.get(key) is not None
                    and (image_limits.get(key) is None
                         or fb_limits[key] < image_limits[key])]
        if stricter:
            notes.append(
                f"fallback {fallback.model_id} declares stricter image "
                f"limits than primary ({', '.join(stricter)}) — records "
                f"exceeding them fail per record IF they escalate; the "
                f"plan enforces the primary's limits only")
    return notes


def _record_has_image(record: dict) -> bool:
    return any(isinstance(block, dict) and "image" in block
               for message in record.get("messages") or []
               for block in message.get("content") or [])


def _declared_limits(spec) -> dict:
    """The entry's enforceable image limits: a well-shaped dict with NUMERIC
    values only. A hand-edited artifact bypasses the seeder (resolve_model
    WARNS, never refuses), so a quoted number or a non-dict container must
    degrade to 'not declared' here — the contract finding already named it
    at resolve time; crashing the plan over a log-fixable data defect would
    fail every request, text-only runs included."""
    limits = (spec.image_rules or {}).get("limits")
    if not isinstance(limits, dict):
        return {}
    return {key: value for key, value in limits.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)}


def _inline_image_bytes(block: dict) -> int:
    """Decoded byte size of an inline image source, 0 for anything else.
    Records carry image bytes as a base64 STRING (JSON), so the decoded size
    is arithmetic over the string length — the payload is never decoded
    here. `s3Location` sources are deliberately 0: they are not request
    payload, which is what the size limits bound."""
    source = (block.get("image") or {}).get("source") or {}
    data = source.get("bytes")
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    if isinstance(data, str):
        # MIME/CLI-style wrapped base64 is tolerated by b64decode and by
        # Bedrock, so whitespace must not count as payload — and padding
        # must be read past a trailing newline
        length = (len(data) - data.count("\n") - data.count("\r")
                  - data.count(" "))
        return (length * 3) // 4 - data.rstrip()[-2:].count("=")
    return 0


def _enforce_image_limit(record: dict, primary, limits: dict) -> None:
    """
    `capabilities.image.limits` — the Bedrock-ENFORCED half of the image rules
    (README.md §P5.5), rejected here before a token is spent. A
    4-image record on Llama 4 (hard cap 3) otherwise fails after submit — or in
    batch, after the whole job validated and parked. Per record, so the
    handler's one streamed pass can call it without holding the set.

    Three declared limits, each skipped when the entry carries none (null
    means NOT DECLARED, nothing pre-checked — the get_model contract says so):
    `max_images` per request, `max_image_mb` per inline image (the Anthropic
    family refuses any image over 5 MiB of base64 (5,242,880 bytes, its own
    error names it) = exactly 3.75 MiB raw — measured
    identical across all five entries, 2026-10-16), and `max_payload_mb` as
    the record's inline-byte total (pixtral/llama4/nova refuse ~17.9 MiB,
    Anthropic/GPT-6 ~23 MiB — all measured; the registry pins ride a step
    conservative). Every *_mb value is MEBIBYTES (×1048576), stated in
    get_model's contract. The size bounds were measured on the sync Converse
    endpoint and are enforced on BATCH records too, deliberately: the
    per-image bound is the MODEL's own refusal (its error names it) and in
    batch one such record fails the whole all-or-nothing job, and any batch
    failure is repaired through the SYNC chain, where the request bound
    certainly holds. Limits bind the PRIMARY; stricter fallbacks are warned
    at plan time (see the caller) — an escalated over-limit record fails per
    record, never the run.

    Deliberately NOT enforced here: `downscales_above` (accepted, silently
    degraded server-side) and `recommendations` (caller-side grid-prep guidance)
    — both are data for the caller, surfaced through the registry, and rejecting
    a request Bedrock would accept is a capability regression, not a guard.
    """
    max_images = limits.get("max_images")
    max_image_mb = limits.get("max_image_mb")
    max_payload_mb = limits.get("max_payload_mb")
    # `is None` throughout, never falsy: a declared 0 MEANS 0 ("no images
    # at all"), and the stricter-fallback warning reads it the same way
    if max_images is None and max_image_mb is None and max_payload_mb is None:
        return
    blocks = [block
              for message in record.get("messages") or []
              for block in message.get("content") or []
              if isinstance(block, dict) and "image" in block]
    if max_images is not None and len(blocks) > max_images:
        raise ValidationException(
            f"record {record['record_id']}: {len(blocks)} images exceeds "
            f"{primary.model_id}'s limit of {max_images} per request — "
            "Bedrock enforces this; combine frames into grid images or "
            "split the record")
    if max_image_mb is None and max_payload_mb is None:
        return
    sizes = [_inline_image_bytes(block) for block in blocks]
    if max_image_mb is not None:
        cap = max_image_mb * 1024 * 1024
        for size in sizes:
            if size > cap:
                raise ValidationException(
                    f"record {record['record_id']}: an inline image of "
                    f"{size / 1048576:.2f} MiB exceeds {primary.model_id}'s "
                    f"per-image limit of {max_image_mb} MiB — Bedrock "
                    "enforces this; re-encode or downscale the image")
    if (max_payload_mb is not None
            and sum(sizes) > max_payload_mb * 1024 * 1024):
        raise ValidationException(
            f"record {record['record_id']}: {sum(sizes) / 1048576:.2f} MiB "
            f"of inline image bytes exceeds {primary.model_id}'s request "
            f"payload limit of {max_payload_mb} MiB — Bedrock enforces "
            "this; shrink or split the record")


def _authoritative_workflow_id(event: dict, parameter_hash: str) -> str:
    """
    The run's workflow_id, which is the EXECUTION NAME — warn when it is not the
    parameter hash this request recomputes to.

    The name wins because it is what everything observable is keyed on: it is the
    id the gateway handed the caller, the last segment of the output prefix, the
    batch job name, and the token-table sort key. A tokened submit makes the two
    legitimately different (the token is folded into the name's hash and never
    reaches this state), so equality cannot be asserted — but the warning is still
    the integrity signal for the OTHER divergence, a request object overwritten at
    its prefix after submission, which breaks the idempotent-resubmit guarantee
    (§3). Human console starts also legitimately mismatch; refusing to run would
    make the workflow undebuggable by hand, so this warns and proceeds.

    A gateway-less start with no execution_arn on the event falls back to the
    parameter hash — the same value the name would have carried.
    """
    execution_arn = (event.get("execution_arn") or "").strip()
    if not execution_arn:
        return parameter_hash
    name = execution_arn.split(":")[-1]
    if name != parameter_hash:
        logger.warning(
            "execution name %s != the parameter hash %s recomputed from the "
            "request. Expected when the submit carried an idempotency_token; "
            "otherwise the request object at this prefix changed after "
            "submission, and idempotent resubmit (§3) is broken for it",
            name, parameter_hash)
    return name


__all__ = ["lambda_handler", "ValidationException"]
