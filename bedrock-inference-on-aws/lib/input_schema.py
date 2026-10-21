"""
Environment constants and request parsing for the bedrock-inference workflow.

Two jobs, both deliberately kept out of the state handlers:

  • env-var constants — single source of truth, read once at module load
  • `parse_request()` / `parse_records()` — normalise `input_params.json` and
    `request.jsonl` (README.md §4.1) and **fail on anything malformed**, in
    `StateValidateAndPlan`, before a token is spent

The Step Functions input is a **pointer, always** — never "a pointer when large"
(§4). A single scene's transcript can exceed the 256 KB payload cap, and two code
paths with one of them rarely exercised is how the rare one breaks.

```json
{"operation": "StateValidateAndPlan",
 "uuid": "<correlation only>",
 "input": {"request_s3uri": "s3://bucket/micro-workflow/.../input_params.json"},
 "data": {}}
```

There is no `workflowConfig` counterpart to M2C's `core-lib/lib/workflowConfig/`:
everything that changes the result is in the request bytes, so it is inside
`workflow_id` (§3). An out-of-band config store means an edited config silently
returns the previous run's cached result.
"""
from __future__ import annotations

import base64
import binascii
import logging
import os
import re

from lib.custom_exceptions import ValidationException
from lib.s3_utils import parse_s3uri

log = logging.getLogger(__name__)

# ── AWS ──────────────────────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
TARGET_NAME = os.environ.get("TARGET_NAME", "bedrock-inference")

MODEL_REGISTRY_KEY = os.environ.get(
    "MODEL_REGISTRY_KEY", "_model_configs/availablemodels.json")

# ── Quota (sync path only) ───────────────────────────────────────────────────
# The scope handed to `lib/semaphore_client`. `resource` is the model id, so two
# models are two independent caps. Still an opaque `{service}#{resource}` string
# even though only "bedrock" is ever passed now: the client was written to be
# vendored and its keys carry no knowledge of what they scope.
QUOTA_SERVICE = "bedrock"

# `lib/semaphore_client` has NO default of its own, deliberately: no single number
# is right for both Bedrock (3) and Rekognition (5), so a client-level default
# would be the wrong cap applied silently whenever a caller omitted the field
# (semaphore invariant 4). The default therefore lives HERE, and ONLY here as of
# 2026-08-12 — the stack's `MaxConcurrentDefault` parameter was a second spelling
# of this same number that could drift per-stack with nothing in the tree to show
# it. It is 3 — from shot/scene metadata extraction experience (§4.1).
MAX_CONCURRENT_DEFAULT = int(os.environ.get("MAX_CONCURRENT_DEFAULT", "3"))

# ── Batch ────────────────────────────────────────────────────────────────────
# Bedrock's floor for a model invocation job. Below it, pad (§5.2).
MIN_BATCH_RECORDS = 100
BATCH_TIMEOUT_HOURS = 24

# The auto-mode guard is denominated in DOLLARS, not records (§5.6, settled
# 2026-08-11): what batch buys is 50 % of token cost and what it costs is hours
# of latency, so the guard is the saving itself. Count is a proxy for the wrong
# quantity — 3 transcript chunks at ~100k input tokens each save ~$0.45 and
# deserve batch; 100 greeting-sized records save a fraction of a cent and
# deserve sync-in-seconds. A count rule gets both wrong.
BATCH_MIN_SAVINGS = float(os.environ.get("BATCH_MIN_SAVINGS", "0.25"))

# The unpriced-model fallback: a missing price must not read as "no saving"
# (the stack's None-not-zero discipline), so an unpriced model batches on token
# volume instead. 200k input tokens is roughly where a mid-tier model's saving
# clears BATCH_MIN_SAVINGS.
UNPRICED_BATCH_MIN_TOKENS = int(
    os.environ.get("UNPRICED_BATCH_MIN_TOKENS", "200000"))

# chars-per-token heuristic for the pre-run input estimate. An underestimate
# errs toward sync — the fast direction.
_CHARS_PER_TOKEN = 4.0

# The repair tail's bound (README.md §P8): failed fractions above
# this are systemic — a shape or access defect — and a sync re-run would re-pay
# the whole job at on-demand rate to fail again. Below it, failures are
# per-record weather and the sync chain walk repairs them in seconds.
REPAIR_MAX_FRACTION = float(os.environ.get("REPAIR_MAX_FRACTION", "0.5"))

# ── Sync loop ────────────────────────────────────────────────────────────────
# Below this much remaining Lambda time, checkpoint and raise
# LambdaResourceTimeoutException rather than starting another Converse call
# (§5.1). A Sandbox.Timedout mid-Converse loses that record's spend with nothing
# recorded. 60 s is face-recognition-on-aws's figure.
REMAINING_TIME_BUFFER_MS = int(os.environ.get("REMAINING_TIME_BUFFER_MS", "60000"))

# In-Lambda backoff attempts per model before the chain advances (§5.3).
DEFAULT_MAX_RETRIES = 3

# ── Ceilings ─────────────────────────────────────────────────────────────────
# A caller submitting 200k records has made a mistake, and the cost of honouring
# it is unbounded. The sync path has a second, tighter bound: 10 re-entries of a
# 15-minute Lambda is ~2.5 h of wall clock, so a record count whose serial
# Converse time exceeds that will exhaust the ASL Retry and fail — which is why
# `auto` sends large sets to batch.
MAX_RECORDS_CEILING = int(os.environ.get("MAX_RECORDS_CEILING", "10000"))
MAX_MODEL_CHAIN = 4

# The chain the gateway substitutes when an agent names no model. `model_chain` is
# REQUIRED below, so without this a caller would have to resolve a model itself
# (list_models first) before it could submit anything.
#
# It lives HERE, as a versioned code constant, and NOT as a stack parameter — which
# is what it was until 2026-08-12. `model_chain` is written into the request by the
# gateway and every key in that request is inside `workflow_id`, so changing this
# value re-identifies every chain-less request ever submitted and re-runs work that
# is already done. As a deploy-time parameter that could happen per-stack with
# nothing in the tree to show it, and it DID: the template shipped haiku-4-5 while
# the eu-west-1 stack ran `global.anthropic.claude-sonnet-4-6`. As a constant the
# value moves only with a code change and a .version bump — which is exactly when a
# hash change is expected and reviewable. A workflow_config file would NOT fix this;
# an operator edit re-identifies work just as silently.
#
# Bare id, no geo prefix: the prefix names ROUTING, not identity. `resolve_model`
# strips it and `resolve_runtime_model_id` re-derives it from the registry entry's
# `inference_region`, so the `us.` prefix this default used to carry was inert on
# every deployment — decoration that read like residency control.
DEFAULT_MODEL_CHAIN = [
    part.strip()
    for part in os.environ.get(
        "DEFAULT_MODEL_CHAIN",
        "anthropic.claude-sonnet-4-6").split(",")
    if part.strip()
]

# The chain-less default for requests that EXPLICITLY force sync
# (use_batch: false — and only that; user decision 2026-09-23). Sonnet 5
# leads because its on-demand pricing undercuts Sonnet 4.6 ($2/$10 vs
# $3/$15 per M tokens); 4.6 stays as the in-chain fallback. Absent
# (auto) and true keep DEFAULT_MODEL_CHAIN: Sonnet 5 cannot batch
# (registry batch_job.can_support: false), and choose_mode keys on the
# PRIMARY model's batch capability — a Sonnet-5-led default under auto
# would silently resolve every auto request to sync and forfeit the
# batch discount exactly where it is largest.
DEFAULT_SYNC_MODEL_CHAIN = [
    part.strip()
    for part in os.environ.get(
        "DEFAULT_SYNC_MODEL_CHAIN",
        "anthropic.claude-sonnet-5-5,anthropic.claude-sonnet-4-6").split(",")
    if part.strip()
]

DEFAULT_TEMPERATURE = 0.0

# ONE home for the vocabulary: lib/bedrock.py owns it (that module stays
# stdlib-only, so this import direction is free of the boto3 cost that
# reverted the opposite one on 2026-10-15)
from lib.bedrock import VALID_EFFORT_VALUES as VALID_EFFORTS  # noqa: E402

# ONE home (lib/bedrock.py), same import direction as VALID_EFFORTS above.
# There is no numeric form on purpose: `budget_tokens` is deprecated on 4.6
# and **400s** on 4.7+ (§9.1), so accepting a number here would only let a
# caller build a request Bedrock rejects.
from lib.bedrock import VALID_THINKING  # noqa: E402


def _positive_int(value, name: str, default: int, ceiling: int | None = None) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValidationException(f"{name} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValidationException(f"{name} must be an integer, got {value!r}")
    if parsed < 1:
        raise ValidationException(f"{name} must be >= 1, got {parsed}")
    if ceiling is not None and parsed > ceiling:
        raise ValidationException(f"{name} must be <= {ceiling}, got {parsed}")
    return parsed


def request_pointer(event: dict) -> tuple[str, str, str]:
    """
    Resolve `input.request_s3uri` → (bucket, request_key, prefix).

    `request.jsonl` is a sibling of `input_params.json` (the control parameters;
    renamed from `request.json` 2026-08-14 — two objects both named `request.*`
    read as one artifact in two formats, when they are parameters vs records) by
    construction: one prefix
    holds one request, and deriving the records key rather than accepting a second
    uri removes the failure mode where the two point at different requests.
    """
    inp = event.get("input") or {}
    uri = (inp.get("request_s3uri") or "").strip()
    if not uri:
        raise ValidationException("input.request_s3uri is required")
    bucket, key = parse_s3uri(uri)
    if not key.endswith("input_params.json"):
        raise ValidationException(
            f"input.request_s3uri must name an 'input_params.json' object, "
            f"got {uri!r}")
    prefix = key[: -len("input_params.json")].rstrip("/")
    return bucket, key, prefix


def records_key(prefix: str) -> str:
    return f"{prefix}/request.jsonl"


def parse_request(request: dict) -> dict:
    """
    Normalise `input_params.json` (§4.1) into the flat config every state reads.

    Raises `ValidationException` on anything that cannot produce a run. This is the
    only place that decides defaults, so the plan and the run are computed from the
    same values.
    """
    if not isinstance(request, dict):
        raise ValidationException(
            f"input_params.json must be an object, got {type(request).__name__}")

    chain = request.get("model_chain")
    if isinstance(chain, str):
        chain = [chain]
    if not isinstance(chain, list) or not chain:
        raise ValidationException("model_chain must be a non-empty list of model ids")
    if len(chain) > MAX_MODEL_CHAIN:
        raise ValidationException(
            f"model_chain must hold at most {MAX_MODEL_CHAIN} models, got {len(chain)}")
    model_chain = []
    for entry in chain:
        if not isinstance(entry, str) or not entry.strip():
            raise ValidationException(f"model_chain entries must be model ids, got {entry!r}")
        if entry.strip() not in model_chain:
            model_chain.append(entry.strip())

    # `mode: sync|batch|auto` retired 2026-08-11 (§5.6) in favour of the optional
    # boolean `use_batch`. A stray `mode` is warned about and IGNORED, never
    # rejected — it was a legitimate parameter until now, and both spellings were
    # always excluded from the hash, so ignoring it cannot re-identify work.
    if request.get("mode") is not None:
        log.warning("`mode` is retired and was ignored — use `use_batch`: "
                    "true forces batch (downgraded with a reason when the model "
                    "cannot), false forces sync, absent lets the estimated "
                    "saving decide")
    use_batch = request.get("use_batch")
    if use_batch is not None and not isinstance(use_batch, bool):
        raise ValidationException(
            f"use_batch must be a boolean when present, got {use_batch!r}")

    # Request-time routing override: "global", a geo (us|eu|apac), or a concrete
    # region (eu-west-1). Whether the CHOSEN MODELS support it is validate_plan's
    # check (it has the registry); this only rejects shapes that can never be a
    # routing scope. Deliberately NOT in EXCLUDED_KEYS: where a run is allowed to
    # execute is part of what the run IS (the EU-residency case), so an override
    # changes the workflow_id and a residency rerun is different work.
    inference_region = (request.get("inference_region") or "").strip().lower()
    if inference_region and not re.match(
            r"^(global|us|eu|apac|[a-z]{2}(-gov)?-[a-z]+-\d)$", inference_region):
        raise ValidationException(
            f"inference_region must be 'global', a geo (us|eu|apac), or an AWS "
            f"region (e.g. eu-west-1), got {request.get('inference_region')!r}")

    options = request.get("options") or {}
    if not isinstance(options, dict):
        raise ValidationException("options must be an object")

    temperature = options.get("temperature")
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    if isinstance(temperature, bool):
        raise ValidationException(f"temperature must be a number, got {temperature!r}")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        raise ValidationException(f"temperature must be a number, got {temperature!r}")
    if not 0.0 <= temperature <= 1.0:
        raise ValidationException(f"temperature must be within 0.0-1.0, got {temperature}")

    effort = options.get("effort")
    if effort is not None:
        effort = str(effort).strip().lower()
        if effort not in VALID_EFFORTS:
            raise ValidationException(
                f"options.effort must be one of {VALID_EFFORTS}, got {effort!r}")

    # Defaults to disabled, and the default is a decision rather than an
    # omission: reasoning spends output tokens the caller may not expect, so
    # a caller that wants it asks for it.
    thinking = options.get("thinking")
    if thinking is None:
        thinking = "disabled"
    thinking = str(thinking).strip().lower()
    if thinking not in VALID_THINKING:
        raise ValidationException(
            f"options.thinking must be one of {VALID_THINKING}, got "
            f"{options.get('thinking')!r} — `budget_tokens` is not accepted for any "
            "model (it 400s on 4.7+), so there is no numeric form")

    # The batch repair tail (§8) is ON by default; `false` opts a caller out who
    # prefers cheap FAILED lines over an on-demand-rate repair pass.
    batch_repair = options.get("batch_repair")
    if batch_repair is None:
        batch_repair = True
    elif not isinstance(batch_repair, bool):
        raise ValidationException(
            f"options.batch_repair must be a boolean, got {batch_repair!r}")

    output = request.get("output") or {}
    if not isinstance(output, dict):
        raise ValidationException("output must be an object")
    out_bucket = (output.get("bucket") or "").strip()
    out_prefix = (output.get("prefix") or "").strip().strip("/")
    if not out_bucket or not out_prefix:
        raise ValidationException("output.bucket and output.prefix are both required")

    return {
        "tool_version": (request.get("tool_version") or "").strip() or None,
        "model_chain": model_chain,
        "use_batch": use_batch,
        "inference_region": inference_region,
        # Forwarded verbatim to `semaphore_client`, which clamps rather than rejects
        # anything over its own ceiling.
        "max_concurrent": _positive_int(
            request.get("max_concurrent"), "max_concurrent", MAX_CONCURRENT_DEFAULT),
        "temperature": temperature,

        "effort": effort,
        "thinking": thinking,
        "batch_repair": batch_repair,
        "output": {"bucket": out_bucket, "prefix": out_prefix},
    }


# Converse's four image formats, and the file magic each one's bytes must open
# with. WEBP is RIFF-framed, so its magic is split (bytes 0-3 / 8-11).
_IMAGE_MAGIC = {
    "gif": (b"GIF87a", b"GIF89a"),
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "webp": (b"RIFF",),
}


def _detect_image_format(head: bytes) -> str | None:
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    for fmt, magics in _IMAGE_MAGIC.items():
        if fmt != "webp" and any(head.startswith(m) for m in magics):
            return fmt
    return None


def _validate_image_block(record_id: str, image) -> None:
    """
    Cheap pre-flight on an image block: format is one Converse accepts, the
    source is exactly one of `bytes` | `s3Location`, and for `bytes` the FIRST
    24 base64 chars decode to that format's file magic. Only the head is
    decoded — the check costs a 24-char slice per image, not a re-decode of the
    payload — so a corrupt tail still surfaces only at the model. What this
    catches at plan time instead of N failed records later:

      * a declared format that does not match the actual bytes;
      * bytes that are not an image at all — DOUBLE-ENCODED BASE64 above all,
        whose head decodes to ASCII (`iVBO...`) and no magic. Live 2026-08-13:
        every vision model failed sync with `The detected file MIME type
        text/plain does not match the expected type image/png` and the defect
        was in the transport, not the caller — this check would have named it
        before a token was spent.

    `s3Location` (which the Nova family accepts as a source) is passed through
    with only a scheme check: the object is not fetched here, deliberately —
    validation must not cost an S3 GET per record.
    """
    if not isinstance(image, dict):
        raise ValidationException(
            f"record {record_id}: image block must be an object")
    fmt = image.get("format")
    if fmt not in _IMAGE_MAGIC:
        raise ValidationException(
            f"record {record_id}: image format must be one of "
            f"{sorted(_IMAGE_MAGIC)}, got {fmt!r}")
    source = image.get("source")
    if not isinstance(source, dict) \
            or ("bytes" in source) == ("s3Location" in source):
        raise ValidationException(
            f"record {record_id}: image source must carry exactly one of "
            "'bytes' (base64 string) or 's3Location' (Nova family)")
    if "s3Location" in source:
        uri = (source.get("s3Location") or {}).get("uri")
        if not str(uri or "").startswith("s3://"):
            raise ValidationException(
                f"record {record_id}: image s3Location.uri must be an "
                f"s3:// uri, got {uri!r}")
        return
    data = source.get("bytes")
    if not isinstance(data, str) or not data:
        raise ValidationException(
            f"record {record_id}: image bytes must be a non-empty base64 "
            "string — records travel as JSON, so raw bytes cannot appear here")
    try:
        head = base64.b64decode(data[:24], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationException(
            f"record {record_id}: image bytes are not base64 ({exc})") from exc
    detected = _detect_image_format(head)
    if detected != fmt:
        raise ValidationException(
            f"record {record_id}: image format says {fmt!r} but the bytes "
            + (f"look like {detected!r}" if detected else
               "open with no known image magic — base64 that decodes to "
               "base64 again (double-encoding) is the classic cause"))


def parse_records(rows: list) -> list:
    """
    Normalise `request.jsonl` (§4.1). One line per record:

    ```json
    {"record_id": "chunk_000", "system": "...", "messages": [...]}
    ```

    `system` and `messages` arrive **already rendered and separate**. Separate
    because concatenating them destroys prompt caching on the stable system half —
    the cheapest available optimisation for a per-chunk workload, and one whose loss
    shows up only on the bill (§12.3). Already rendered because the caller owns
    prompt resolution and, per §3, the rendered bytes must be what the hash sees.

    `record_id` must be unique: it is the join key between `request.jsonl`,
    `response.jsonl`, and a batch job's `recordId`, so a duplicate silently
    overwrites another record's result.
    """
    if not rows:
        raise ValidationException("request.jsonl is empty — refusing to start a run")
    if len(rows) > MAX_RECORDS_CEILING:
        raise ValidationException(
            f"request.jsonl holds {len(rows)} records, over the "
            f"{MAX_RECORDS_CEILING} ceiling")

    records, seen = [], set()
    for index, row in enumerate(rows):
        records.append(parse_record(row, index, seen))
    return records


def iter_parse_records(bucket: str, key: str):
    """Streamed `parse_records`: yield each normalised record from a JSONL
    object without holding the set — ONE home for the shared seen-set, the
    MAX_RECORDS_CEILING refusal, and the emptiness refusal (review,
    2026-09-06: the streaming rewrite had inlined these per state, and the
    copies had already started to drift). validate_plan, submit_batch and
    collect_batch all iterate this; the gateway's staged-records path stays
    separate deliberately — it reads an already-open body with a peeked
    carry, not a bucket/key.

    Emptiness raises at EXHAUSTION, so a caller must consume the generator
    fully for the guarantee — every current caller does."""
    from lib import s3_utils

    seen: set = set()
    count = 0
    for index, row in s3_utils.iter_jsonl(bucket, key):
        if count >= MAX_RECORDS_CEILING:
            raise ValidationException(
                f"request.jsonl holds more than the "
                f"{MAX_RECORDS_CEILING} record ceiling")
        yield parse_record(row, index, seen)
        count += 1
    if count == 0:
        raise ValidationException(
            "request.jsonl is empty — refusing to start a run")


def parse_record(row, index: int, seen: set) -> dict:
    """Validate and normalise ONE request.jsonl row (the §4.1 shape above).

    The single-record form exists for the gateway's STREAMING staged-records
    path, which must validate line by line without holding the file — `seen`
    is the caller's record_id set and is mutated here, so the duplicate check
    costs the ids alone, never the records."""
    if not isinstance(row, dict):
        raise ValidationException(
            f"request.jsonl line {index + 1} must be an object, "
            f"got {type(row).__name__}")

    record_id = str(row.get("record_id") or "").strip()
    if not record_id:
        raise ValidationException(
            f"request.jsonl line {index + 1} has no record_id")
    if record_id.startswith("pad_"):
        # Reserved: `Collect batch output` drops pad_* records, so a real
        # record named pad_000 would vanish from the response with no error.
        raise ValidationException(
            f"record_id {record_id!r} uses the reserved 'pad_' prefix")
    if record_id in seen:
        raise ValidationException(f"duplicate record_id {record_id!r}")
    seen.add(record_id)

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValidationException(
            f"record {record_id}: messages must be a non-empty list of "
            "Converse message objects")
    for message in messages:
        if not isinstance(message, dict) or "role" not in message \
                or not isinstance(message.get("content"), list):
            raise ValidationException(
                f"record {record_id}: each message needs a role and a content list")
        for block in message["content"]:
            if isinstance(block, dict) and "image" in block:
                _validate_image_block(record_id, block["image"])

    system = row.get("system")
    if system is not None and not isinstance(system, (str, list)):
        raise ValidationException(
            f"record {record_id}: system must be a string or a list of "
            f"Converse system blocks, got {type(system).__name__}")

    # `max_tokens` is RETIRED outright (2026-08-18, user decision: one
    # more parameter is one more maintenance cost). A row still carrying
    # it is ignored — never rejected (the house stray-field rule) and
    # never digested, so two requests differing only in a stray retired
    # field are ONE run. `maxTokens` is never emitted; Bedrock's
    # model-maximum default applies (probed across every reachable
    # model the day it was removed).
    if row.get("max_tokens") is not None:
        log.warning("record %s: `max_tokens` is retired and was "
                    "ignored — the model's own maximum applies",
                    record_id)
    return {
        "record_id": record_id,
        "system": system,
        "messages": messages,
    }


def estimate_input_tokens(records: list) -> int:
    """Pre-run input-token estimate over the RENDERED records: every text block
    in system and messages, at ~4 chars/token. Output tokens are unknowable
    pre-run and deliberately ignored — input dominates exactly the workloads
    where batch matters, and an underestimate errs toward sync."""
    return tokens_from_chars(sum(record_text_chars(r) for r in records or []))


def record_text_chars(record: dict) -> int:
    """Text chars of ONE record — the streamed form of the accumulation
    above. A streaming caller sums these and converts ONCE via
    tokens_from_chars: per-record integer division would drift from the
    list form's total."""
    chars = 0
    system = record.get("system")
    if isinstance(system, str):
        chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                chars += len(block.get("text") or "")
            elif isinstance(block, str):
                chars += len(block)
    for message in record.get("messages") or []:
        for block in message.get("content") or []:
            if isinstance(block, dict):
                chars += len(block.get("text") or "")
    return chars


def tokens_from_chars(chars: int) -> int:
    return int(chars / _CHARS_PER_TOKEN)


def choose_mode(use_batch, can_batch: bool, *, estimated_saving,
                estimated_input_tokens: int) -> tuple[str, str]:
    """
    Resolve `use_batch` to `"sync"` or `"batch"`, with the reason (§5.6, settled
    2026-08-11). The chosen mode is **reported, never silently assumed** — it is
    excluded from `workflow_id`, so `summary.inference_mode` / `mode_reason` are
    where a caller learns which path ran and why.

    | use_batch | model can batch | result                                  |
    |-----------|-----------------|-----------------------------------------|
    | absent    | yes             | batch iff the saving clears the guard   |
    | absent    | no              | sync                                    |
    | true      | yes             | batch                                   |
    | true      | no              | **sync, downgrade REPORTED** — not an   |
    |           |                 | error (the old hard ValidationException |
    |           |                 | retired; the telling moved to the       |
    |           |                 | reason and the summary)                 |
    | false     | —               | sync, always: the caller can always opt |
    |           |                 | OUT of batch, never force it into being |

    `estimated_saving` is `None` for an unpriced model — the token-volume
    fallback decides then, because a missing price must not read as "no saving".
    """
    if use_batch is False:
        return "sync", "requested (use_batch: false)"
    if use_batch is True:
        if can_batch:
            return "batch", "requested (use_batch: true)"
        return "sync", ("use_batch requested but the primary model does not "
                        "support batch inference — downgraded to sync at the "
                        "on-demand rate")
    if not can_batch:
        return "sync", "auto: primary model cannot batch"
    if estimated_saving is None:
        if estimated_input_tokens >= UNPRICED_BATCH_MIN_TOKENS:
            return "batch", (f"auto: unpriced model, est. "
                             f"{estimated_input_tokens} input tokens at or over "
                             f"the {UNPRICED_BATCH_MIN_TOKENS}-token fallback")
        return "sync", (f"auto: unpriced model, est. {estimated_input_tokens} "
                        f"input tokens under the {UNPRICED_BATCH_MIN_TOKENS}-"
                        "token fallback")
    if estimated_saving >= BATCH_MIN_SAVINGS:
        return "batch", (f"auto: est. batch saving ${estimated_saving:.2f} "
                         f"clears the ${BATCH_MIN_SAVINGS:.2f} guard")
    return "sync", (f"auto: est. batch saving ${estimated_saving:.4f} under "
                    f"the ${BATCH_MIN_SAVINGS:.2f} guard — sync answers in "
                    "seconds; a fresh idempotency_token with use_batch true "
                    "forces batch anyway")
