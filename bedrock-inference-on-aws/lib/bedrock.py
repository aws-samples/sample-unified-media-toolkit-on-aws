"""
The model matrix, the Converse call, the retry ladder, and the chain walk.

**Vendored** — this file is copied into each consuming container as well as living
here (README.md §9). Pure facts and pure shaping, no I/O beyond the Bedrock call
itself: model ids, `accepts_temperature`, thinking/effort shaping, `can_batch` /
`batch_region`, pricing.

Why the caller and not a service: a caller must shape params and decide
batch-eligibility *before* it can ask for a lease or build a request. A network
round trip to learn a static fact is backwards. The cost is accepted — the file is
copied, so drift is visible in a diff and testable entirely offline. (Since
2026-08-18 `max_tokens` is NOT a caller knob; since 2026-10-16 an entry MAY
declare `capabilities.emit_max_tokens` — the deliberate re-addition as
registry data, forced by the 5 family's truncating 4096 Converse default —
see build_converse_params.)

Adapted from `conversation-analysis-on-aws/lib/bedrock.py`, which is the canonical
shape, with three changes this workflow needs:

  1. **`system` and `messages` arrive in Converse shape**, already rendered, rather
     than as two strings. This workflow never renders a prompt (§13).
  2. **One shaping function serves both paths.** `build_converse_params` is used by
     the sync loop and by `to_batch_record`, so the same request cannot produce
     differently-shaped calls depending on mode — `web-inference`'s divergence
     defect (§2), where `run-inference` range-checked temperature and called
     `useExtendedReasoning('low')` and `batch-run-inference` did neither.
  3. **Thinking/effort shaping is data**, not a class hierarchy. §9.2 records a
     contradiction that exists *inside* `core-lib`'s model hierarchy today, which is
     the duplication argument in miniature.

Five divergences from `core-lib/lib/segmentHelper/models/base.js` (the first
four carried over from conversation-analysis and documented there in full;
the fifth, 2026-10-15, also diverges from conversation-analysis's own copy —
anyone syncing the vendored file in either direction must carry it
knowingly):

  1. `ThrottlingException` retries on the same model; only
     `ServiceQuotaExceededException` escalates to the next.
  2. `temperature` is never silently dropped on retry — only when Bedrock
     explicitly rejects it, and the strip is recorded. (Shaping-time
     suppression beside reasoning is a different thing: it is the documented
     contract when the CALLER asks for adaptive, and when an entry's
     adaptive DEFAULT does it the condition is named by resolve_model's
     contract warning — data the seeder refuses to seed.)
  3. `content_filtered` / `guardrail_intervened` escalate rather than being
     returned as the result.
  4. An unregistered model raises rather than running uncosted.
  5. A model's request contract (temperature/thinking/effort) lives ONLY on
     its registry entry's `capabilities` — the id-substring family tuples
     (NO_TEMPERATURE_MODELS etc.) are retired here while the reference
     copies still carry them. Re-adopting a tuple version silently flips
     every 5-family contract; copying this version over a registry whose
     entries lack `capabilities` blocks mis-shapes them.

`max_tokens` is not an error anywhere in this module — it is returned as a normal
outcome with the partial text intact.
"""
from __future__ import annotations

import base64
import copy
import logging
import time

log = logging.getLogger(__name__)

# _model_configs/, not _settings/ — v5's core stack owns _settings/ in a shared
# proxy bucket. Must match Mappings.ModelRegistry.Object.Key in cfnstack.yaml,
# which is where the IAM grant and the env var both come from; this is only the
# offline fallback.
DEFAULT_REGISTRY_KEY = "_model_configs/availablemodels.json"

# Retry taxonomy, from source/layers/core-lib/lib/exceptionHelper.js.
THROTTLING_EXCEPTIONS = frozenset(("ThrottlingException",))
QUOTA_EXCEPTIONS = frozenset(("ServiceQuotaExceededException",))
TRANSIENT_EXCEPTIONS = frozenset((
    "InternalServerException", "ModelErrorException", "ModelNotReadyException",
    "ModelTimeoutException", "ServiceUnavailableException",
))
# Model is unusable for this caller; another model may not be.
UNAVAILABLE_EXCEPTIONS = frozenset((
    "AccessDeniedException", "ResourceNotFoundException",
))

RETRY_ON_SAME_MODEL = THROTTLING_EXCEPTIONS | TRANSIENT_EXCEPTIONS
ESCALATE_TO_NEXT_MODEL = QUOTA_EXCEPTIONS | UNAVAILABLE_EXCEPTIONS

# Quota-shaped names for a batch `CreateModelInvocationJob` refusal (§6.2). Only
# these queue the row PENDING; anything else — a ValidationException on a malformed
# record set above all — must fail the caller immediately, because queueing a
# permanent error retries it forever, invisibly, until the token times out.
BATCH_QUOTA_EXCEPTIONS = frozenset((
    "ServiceQuotaExceededException", "ThrottlingException",
    "InternalServerException", "ServiceUnavailable", "ServiceUnavailableException",
    "RequestTimeoutException", "TooManyRequestsException",
))

# stopReasons that mean "this model will not answer" rather than "answer truncated".
REFUSAL_STOP_REASONS = frozenset(("content_filtered", "guardrail_intervened"))

# A model's request contract (temperature / thinking / effort) lives ONLY on
# its registry entry's `capabilities` — there are no id-substring family
# tuples any more (NO_TEMPERATURE_MODELS / ADAPTIVE_THINKING_MODELS /
# EFFORT_MODELS, retired 2026-10-15). They were fallbacks that went stale as
# models shipped (the sonnet-5 -> sonnet-5-5 swap proved the entry is the
# contract), were already inert — every generative entry carries the keys,
# the seeder refuses to SEED one that does not, and `resolve_model` refuses
# unregistered ids — and made every model update a two-place edit. Adding a
# model is now ONE registry entry. The runtime net is NARROW: `_converse_once`
# recovers a rejected temperature only — a wrong thinking/effort contract
# fails every record as ValidationException, which is precisely why the keys
# are mandatory at seed time.
#
# The one path the seeder cannot guard: a HAND-EDITED installed artifact
# (`_model_configs/availablemodels.json`) never passes through it. An entry
# added by hand without its capability keys reads as thinking "absent" /
# effort off — on a thinking-required family every request then 400s. If you
# must hand-edit, copy a master entry whole; the clean path is editing the
# master and redeploying.
#
# `budget_tokens` is never emitted for ANY model: deprecated on 4.6 and
# rejected with a 400 on 4.7+ (§9.1) — thinking takes only
# {"type": "adaptive"|"disabled"}.

# Assistant-role prefill is rejected for EVERY model — a request-shape rule
# (settled 2026-08-11), not a per-model capability, so the registry carries no
# flag and there is no tuple. History of the narrowing-then-widening: §9.2
# first banned it for Opus 5 / Fable 5 (core-lib's Sonnet5 subclass pushes a
# prefill unconditionally, overriding base.js's own reasoning-mode skip — the
# override was the defect); then Claude turned out to reject an assistant-role
# LAST message across the family, and Llama 4 does too (v5 works around it
# with a user-role trailer). A "capability" that is false or useless for every
# model that matters is not a capability. Loud rejection rather than v5-style
# silent dropping: a caller who built a prefill-dependent prompt (a forced
# JSON opener) gets an unrelated parse failure downstream if the message just
# vanishes. "Applies to batch too" falls out of the plumbing:
# `build_converse_params` is the one function both paths call.

VALID_THINKING = ("disabled", "adaptive")

# the values ModelSpec.thinking_support may answer. The seeder IMPORTS this
# and the three constants below (its zip carries lib/ at the root) — one
# object, no mirrored copy to drift
VALID_THINKING_SUPPORT = ("absent", "optional", "required")

# model_type values KNOWN to never reach Converse — the ONLY vocabulary
# the exemption consults: an entry is exempt from the capability contract
# iff every declared type is in this tuple, so an unknown or typo'd type
# ("Text", "txt") stays on the hook (the safe direction) and a
# legitimately new non-Converse type (image/video generation) earns its
# exemption as a deliberate one-line edit here
NON_CONVERSE_TYPES = ("embedding",)

# the per-entry request-contract keys a generative entry must carry
CONTRACT_KEYS = ("temperature", "thinking", "effort")

# the boolean subset of CONTRACT_KEYS ("thinking" is the enum-valued one)
BOOL_CONTRACT_KEYS = ("temperature", "effort")

# effort vocabulary — THE one home: lib/input_schema re-exports this as
# VALID_EFFORTS. The dependency runs this direction deliberately — the
# reverse (bedrock importing input_schema, even lazily) executes
# lib/s3_utils' module-level boto3.client("s3"), breaking this module's
# stdlib-only vendoring property (tried and reverted 2026-10-15).
# These three are the CALLER vocabulary: the values every effort-accepting
# wire accepts, so a chain mixing wires never 400s on a caller's knob.
VALID_EFFORT_VALUES = ("low", "medium", "high")

# effort WIRE shapes — which additionalModelRequestFields key carries the
# knob, per entry (`capabilities.effort_wire`, default "output_config"):
#   output_config → {"output_config": {"effort": ...}}   (Anthropic Claude 5)
#   reasoning     → {"reasoning": {"effort": ...}}       (OpenAI GPT-6 —
#                   flat `reasoning_effort` is REJECTED, probed 2026-10-16)
# Each wire's REGISTRY vocabulary (what an entry's effort_default may say)
# is wider than the caller's: "none" is reachable only as a default, and
# only on the reasoning wire — Sol/Astra reject it, Luna accepts it, so it
# is per-entry data, never a caller value.
EFFORT_WIRE_VOCABULARY = {
    "output_config": ("low", "medium", "high"),
    "reasoning": ("none", "low", "medium", "high", "xhigh", "max"),
}
# ONE source of truth: the wire set IS the vocabulary's key set — a wire
# added to only one of two parallel constants would drift silently in
# both directions (seed check vs runtime coercion).
EFFORT_WIRES = tuple(EFFORT_WIRE_VOCABULARY)

# `capabilities.prompt_cache_mode` — what `prompt_cache_options.mode` a
# request carries, emitted verbatim when the entry declares it. "explicit"
# is the ONLY member because it is the only value worth emitting: implicit
# is the API default when the field is absent, and GPT-6's implicit caching
# bills every >=1024-token prefix as a cache WRITE at 1.25x the input rate
# with zero reads in this stack's record-shaped workloads (measured
# 2026-10-16, SME bake-off: 6/6 records wrote, 0 read). Explicit mode with
# no breakpoints stops the write charge and restores honest inputTokens on
# fresh prefixes (probed on all three GPT-6 models; repeats can still be
# served from cache — reconstructed_cache_write catches those).
PROMPT_CACHE_MODES = ("explicit",)


# Region-scoped inference-profile prefixes, stripped before family detection.
_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "us-gov.", "global.")

# Geo prefixes a batch job id carries. Distinct from _PROFILE_PREFIXES: batch takes
# a bare geo (`us.`), and `resolve_batch_model_id` re-derives it from the registry's
# `batchJob.region` rather than reusing the runtime profile's.
_BATCH_GEO_PREFIXES = ("us", "eu", "apac", "global")

_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0

_PAD_PREFIX = "pad_"

# Bedrock batch inference is billed at 50% of the on-demand rate, and the registry
# carries one price per model (the on-demand one). Applied in `estimate_cost` when
# `batch=True` so a batch summary reports a real number instead of `None` — which
# would surface as `estimated_cost: 0.0` and read as "this run was free".
#
# If a model ever ships with a different batch discount, this is the single constant
# to change, and `summary.bedrock_usage.cost_is_complete` is not the flag that would
# catch it — a wrong multiplier is a wrong number, not an incomplete one.
BATCH_COST_MULTIPLIER = 0.5


def reconstructed_cache_write(usage: dict) -> int:
    """Cache-write tokens a response declined to itemize.

    GPT-6 over Converse caches implicitly (automatic, no opt-in) and its
    usage block is NONDETERMINISTIC about saying so: the identical request
    sometimes comes back fully itemized and sometimes with the cache keys
    ABSENT while the cached input is counted only in totalTokens —
    inputTokens collapses to a near-zero remainder (measured 2026-10-16,
    24 live calls incl. a 2k-20k size ladder; size-gating refuted). AWS's
    own identity `total = input + output + cacheRead + cacheWrite` is the
    reconstruction: solve for the write term. Applied ONLY when the write
    field carries no value (missing key, or the explicit-null the Opus 5 /
    Nova batch dialect writes) — a reported value is trusted as reported.
    When the hidden tokens were really cache READS (a repeat request within
    the cache TTL), pricing them as writes errs HIGH, never low. Every term
    defaults to 0 when absent or null, including inputTokens itself; a
    response with no totalTokens reconstructs to 0 (max-guard), so entries
    whose dialects omit totals never gain phantom tokens.
    """
    usage = usage or {}
    # counts only HIDDEN tokens, never the reported field: a reported value
    # (even 0) means nothing was hidden — estimate_cost prices the reported
    # field directly, and the note-writers must stay silent on honest rows
    if usage.get("cacheWriteInputTokens") is not None:
        return 0
    return max(0, int(
        (usage.get("totalTokens") or 0)
        - (usage.get("inputTokens") or 0)
        - (usage.get("outputTokens") or 0)
        - (usage.get("cacheReadInputTokens") or 0)))


class BedrockError(Exception):
    """
    Inference could not be completed by any model in the chain.

    Carries the underlying service error `codes` because the caller has to decide
    between "record this record as failed" and "propagate for the Step Functions
    Retry", and a flattened message string cannot answer that.
    """

    def __init__(self, message, codes=None, rounds=0):
        super().__init__(message)
        self.codes = list(codes or [])
        # Model calls actually made before giving up. A record that burned three
        # throttled attempts on model A and two on model B cost five rounds of input
        # tokens, and the summary under-reports the bill if a failure contributes
        # zero (§4.2).
        self.rounds = int(rounds or 0)


class ModelNotRegistered(BedrockError):
    """Named model is absent from the deployed model registry."""


class BedrockQuotaSaturated(BedrockError):
    """
    Every model in the chain is throttled or out of quota after the in-Lambda
    backoff ladder.

    A distinct class rather than a re-raised ClientError because Step Functions
    matches a Lambda failure on `errorType`, which the Python runtime sets to the
    **exception class name** — not the Bedrock error code.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def load_registry(s3_client, bucket: str, key: str = DEFAULT_REGISTRY_KEY) -> list:
    """
    Load the deployed model list — written by this stack's own
    Custom::ModelRegistrySeeder since 2026-08-11 (before that, by the v5 core
    stack's CreateModelList, whose artifact still lives separately at
    `_settings/`).

    Reused as-is rather than duplicated: it already carries per-model pricing,
    `batchJob.canSupport`, and region scoping, and a second copy would drift. It is
    also the **actual allowlist** — Bedrock IAM cannot be scoped to the chosen model,
    because a cross-region inference profile needs the profile ARN plus the
    foundation-model ARN in every region it routes to, none knowable at deploy time.
    """
    import json

    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    registry = json.loads(body)
    # The seeded artifact (2026-08-11) is an ENVELOPE — {registry_version, geo,
    # models: [...]} — so the installed list is self-describing; the legacy core-
    # stack artifact is the bare list. Callers always get the list.
    if isinstance(registry, dict):
        registry = registry.get("models")
    if not isinstance(registry, list) or not registry:
        raise BedrockError(
            f"model registry s3://{bucket}/{key} holds no model list")
    return registry


def model_family(model_id: str) -> str:
    """
    Provider family for a model id, used for the cross-family fallback check.

    `us.anthropic.claude-haiku-4-5-20251001-v1:0` → `anthropic`.
    """
    identifier = (model_id or "").strip().lower()
    for prefix in _PROFILE_PREFIXES:
        if identifier.startswith(prefix):
            identifier = identifier[len(prefix):]
            break
    return identifier.split(".")[0] if "." in identifier else identifier


def _get(config: dict, snake: str, camel: str, default=None):
    """Read a registry field in either spelling. The seeded registry is
    snake_case (2026-08-11 shape); the transition tolerance for the old camelCase
    artifact costs one line per field and keeps a straggler deployment working."""
    if snake in config:
        return config[snake]
    if camel in config:
        return config[camel]
    return default


class ModelSpec:
    """One registry entry, with the behaviour that depends on it.

    Reads BOTH registry shapes: the seeded snake_case one (bare `model_id`,
    `inference_region`, `capabilities`) and the legacy camelCase artifact the m2c
    core stack wrote (prefixed `modelId`, CSV `supportedRegions`). Capability
    questions are answered by the entry's `capabilities` block ALONE — the
    id-substring family tuples that used to back it are retired (2026-10-15;
    see the retirement note above). A generative entry missing the block gets
    fixed defaults (temperature on, thinking absent, effort off) and a
    resolve-time warning naming it; the seeder refuses to seed one.
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self.model_id = _get(self.config, "model_id", "modelId") or ""
        self.model_name = (_get(self.config, "model_name", "modelName")
                           or self.model_id)
        self.family = model_family(self.model_id)

    @property
    def pricing(self) -> dict:
        return _get(self.config, "model_pricing", "modelPricing") or {}

    @property
    def capabilities(self) -> dict:
        value = self.config.get("capabilities")
        # a hand-edit can write a string/list here — coerce to {} so the
        # key-less defaults apply (named by contract_findings) instead of
        # every request dying on an AttributeError deep in the shaper
        return value if isinstance(value, dict) else {}

    @property
    def image_rules(self) -> dict:
        """`capabilities.image` — {} for text-only and legacy entries. The
        `limits` half is enforced at validate_plan; `downscales_above` and
        `recommendations` are caller-side data surfaced through get_model.
        A hand-edited non-dict coerces to {} (no limits declared) — the
        shape finding names it; a crash here would fail every request."""
        value = self.capabilities.get("image")
        return value if isinstance(value, dict) else {}

    @property
    def supported_regions(self) -> list:
        """May hold `global`, geos (`us`|`eu`|`apac`), and concrete regions —
        all three legitimately coexist. Legacy artifacts carry a CSV string."""
        raw = _get(self.config, "supported_regions", "supportedRegions") or []
        if isinstance(raw, str):
            return [region.strip() for region in raw.split(",") if region.strip()]
        return [str(region) for region in raw if str(region).strip()]

    @property
    def inference_region(self) -> str:
        """Default routing scope: `global`, a geo, or a concrete region. The
        seeder resolved `"geo"` to the deploy geo at install time. Legacy
        entries carry the scope inside their prefixed modelId instead, so the
        prefix (if any) is the answer there."""
        value = self.config.get("inference_region")
        if value:
            return str(value)
        for prefix in _PROFILE_PREFIXES:
            if self.model_id.startswith(prefix):
                return prefix.rstrip(".")
        return ""

    def resolve_runtime_model_id(self, inference_region: str | None = None) -> str:
        """The id handed to Converse/batch for a routing scope (README.md
        §P5.1): `global`/geo scopes prefix the bare id (`global.{id}`), a
        concrete region uses the bare id in that region. Legacy prefixed ids pass
        through untouched when no override is given.

        A concrete-region scope on a `requires_inference_profile` model keeps the
        model's GEO prefix: Bedrock rejects bare on-demand invocation of
        profile-only models ("Invocation of model ID X with on-demand throughput
        isn't supported" — measured live, 2026-08-11, Nova Pro pinned to
        eu-west-1). The residency property survives — the geo profile is invoked
        FROM the pinned region's client and a geo profile routes within its geo —
        it is the id that must stay profiled, not the routing that changes."""
        scope = (inference_region or self.inference_region or "").strip()
        model_id = self.model_id
        for prefix in _PROFILE_PREFIXES:
            if model_id.startswith(prefix):
                model_id = model_id[len(prefix):]
                break
        if scope in _BATCH_GEO_PREFIXES:
            return f"{scope}.{model_id}"
        if scope and self.config.get("requires_inference_profile"):
            # Concrete region + profile-only model: profile through the region's
            # own geo (eu-west-1 -> eu.{id}), client still pinned to the region.
            geo = scope.split("-", 1)[0]
            if geo in _BATCH_GEO_PREFIXES:
                return f"{geo}.{model_id}"
        return model_id

    def resolve_client_region(self, inference_region: str | None = None) -> str | None:
        """The boto3 region for a scope — a concrete region routes the client
        there; `global`/geo scopes run from the Lambda's own region (None)."""
        scope = (inference_region or self.inference_region or "").strip()
        if scope and scope not in _BATCH_GEO_PREFIXES:
            return scope
        return None

    def allows_inference_region(self, inference_region: str) -> bool:
        """A request-time override must be LITERALLY in supported_regions — no
        inference from geo membership, because a geo profile existing says
        nothing about in-region FM availability. This is the EU-residency gate:
        a model supporting only ["global"] refuses an `eu` override loudly at
        plan time rather than silently running global (§5.1)."""
        return inference_region in self.supported_regions

    def can_batch(self) -> bool:
        return bool(_get(self.config.get("batch_job")
                         or self.config.get("batchJob") or {},
                         "can_support", "canSupport"))

    @property
    def batch_scope(self) -> str:
        """The routing scope batch submits under when the request carries no
        override. On the seeded shape batch FOLLOWS inference_region — there is
        no separate batch region (settled 2026-08-11). On a legacy camelCase
        entry the old `batchJob.region` geo is still honoured: those ids are
        pre-prefixed per deployment, and e.g. a `us.`-invoked Claude may only
        batch through `global.` — dropping that on a straggler artifact would
        submit mis-prefixed jobs."""
        legacy = (self.config.get("batchJob") or {}).get("region") or ""
        if legacy and "batch_job" not in self.config:
            return str(legacy)
        return self.inference_region

    @property
    def batch_pricing_override(self) -> dict:
        """Per-model batch pricing for models whose batch rate is NOT 50 % of
        on-demand (Nova MM Embeddings). Preferred over BATCH_COST_MULTIPLIER."""
        return (self.config.get("batch_job")
                or self.config.get("batchJob") or {}).get("pricing") or {}

    def accepts_temperature(self) -> bool:
        # absent (an embedding entry, or a legacy artifact that predates the
        # capability keys) reads as "accepts it" — the pre-capability default,
        # and _converse_once recovers at runtime if the data is wrong
        return bool(self.capabilities.get("temperature", True))

    @property
    def thinking_support(self) -> str:
        """"absent" | "optional" | "required". `required` (the 5 family)
        means the request must ALWAYS carry
        additionalModelRequestFields.thinking; WHAT a quiet request's field
        carries is the entry's thinking_default — "adaptive" across every
        shipped 5-family entry since the 2026-10-15 sonnet-5-5 fix (retired
        Sonnet 5 was the one quiet-disabled entry)."""
        value = self.capabilities.get("thinking")
        return value if value in VALID_THINKING_SUPPORT else "absent"

    def accepts_adaptive_thinking(self) -> bool:
        return self.thinking_support != "absent"

    def requires_thinking_field(self) -> bool:
        return self.thinking_support == "required"

    def accepts_effort(self) -> bool:
        return bool(self.capabilities.get("effort"))

    def default_thinking_type(self) -> str:
        """
        What `additionalModelRequestFields.thinking` carries when the caller did
        not ask for reasoning. "disabled" unless the entry overrides it —
        because "required" only says the FIELD must be present, and the Claude 5
        family split on what value a quiet request may carry: Sonnet 5 accepts
        (and wants) an explicit disabled, while FABLE 5 REJECTS IT outright —
        `"thinking.type.disabled" is not supported for this model` (live,
        2026-08-14, the model_matrix run's only red) — and Opus 5, though it
        tolerates disabled, is SPECIFIED to run adaptive. Data on the entry,
        not id-matching in code: the registry is where model contracts live.
        """
        value = self.capabilities.get("thinking_default")
        return value if value in VALID_THINKING else "disabled"

    def default_effort(self) -> str | None:
        """The emitted effort when the caller did not pass one (the Claude 5
        family runs at "low" by default — settled 2026-08-14; GPT-6 Luna at
        "none" — user decision 2026-10-16); None emits no effort field at
        all, which keeps 4.6-family requests byte-identical to what they
        were before effort defaults existed."""
        return self.capabilities.get("effort_default") or None

    def effort_wire(self) -> str:
        """Which additionalModelRequestFields key carries the effort knob —
        see EFFORT_WIRES. Key-less entries shape as before (output_config,
        the Anthropic wire)."""
        value = self.capabilities.get("effort_wire")
        return value if value in EFFORT_WIRES else "output_config"

    def prompt_cache_mode(self) -> str | None:
        """The `prompt_cache_options.mode` every request to this entry
        carries, or None to emit no field (the API's implicit default) —
        see PROMPT_CACHE_MODES. Entry data, not a caller knob: whether a
        model's implicit cache helps or just bills write premiums is a
        property of the model's billing contract, not of any one request."""
        value = self.capabilities.get("prompt_cache_mode")
        return value if value in PROMPT_CACHE_MODES else None

    def emit_max_tokens(self) -> int | None:
        """The `inferenceConfig.maxTokens` every request to this entry
        carries, or None to omit the field. The DELIBERATE re-addition of
        the retired knob, as entry data and never a caller parameter
        (2026-10-16): Bedrock's Converse default for an omitted maxTokens
        is NOT always the model maximum — the thinking-required 5 family
        truncates at an undocumented flat 4096 (adaptive-thinking tokens
        bill as output and push structured answers past it; measured:
        fable-5-1/opus-5/sonnet-5-5/haiku-5-5 all stop `max_tokens` at
        exactly 4096 on a >4k-token ask, while sonnet-4-6, haiku-4-5 and
        the GPT-6 trio finish naturally past 4096), and an explicit value
        sails past the default (haiku-5-5 at 8192 → end_turn at 5,154).
        Tolerates hand-edits: a bool, non-number, or non-positive value
        reads as None (and contract_findings names it)."""
        value = self.capabilities.get("emit_max_tokens")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value) if value > 0 else None

    def estimate_cost(self, usage: dict, *, batch: bool = False) -> float | None:
        """
        Cost in USD from token usage, or None when pricing is unknown.

        None rather than 0.0: a missing price and a free call are different facts,
        and 0.0 reads as "this run cost nothing". It surfaces in the summary as
        `bedrock_usage.rounds_unpriced`, reported alongside `estimated_cost` so a
        partial figure is never mistaken for the whole bill (§4.2).

        `batch=True` applies `BATCH_COST_MULTIPLIER`. The registry carries the
        on-demand price only, so without it a batch run reports double — and returning
        None instead would report `0.0`, which is worse than an approximation because
        nothing marks it as one.
        """
        # `or 0`, NEVER a .get() default: a present-but-NULL key returns the null,
        # not the default, and Nova and Opus 5 batch output really does carry
        # `"cacheReadInputTokens": null` — which made this line the TypeError that
        # failed four otherwise-complete batch executions on 2026-08-13, AFTER
        # Bedrock had already run and billed the job. The 2026-08-14 "cache
        # tokens do not apply here" settlement died 2026-10-16: GPT-6 caches
        # IMPLICITLY (no opt-in exists for the write; the entries' explicit-mode
        # emission curbs it but repeats still read), so its entries carry real
        # cache pricing and the write term falls back to the reconstructed count
        # when the response hid it — see reconstructed_cache_write. Entries
        # without cache pricing still price those tokens at 0 by design.
        pricing = self.pricing
        in_cost = _get(pricing, "input_per_token_cost",
                       "inputPerTokenCost") or 0.0
        out_cost = _get(pricing, "output_per_token_cost",
                        "outputPerTokenCost") or 0.0
        if not in_cost and not out_cost:
            return None
        write_tokens = usage.get("cacheWriteInputTokens")
        if write_tokens is None:
            write_tokens = reconstructed_cache_write(usage)
        cost = (
            ((usage.get("inputTokens") or 0) * in_cost)
            + ((usage.get("outputTokens") or 0) * out_cost)
            + ((usage.get("cacheReadInputTokens") or 0)
               * (_get(pricing, "cache_read_per_token_cost",
                       "cacheReadPerTokenCost") or 0.0))
            + ((write_tokens or 0)
               * (_get(pricing, "cache_write_per_token_cost",
                       "cacheWritePerTokenCost") or 0.0))
        )
        return cost * BATCH_COST_MULTIPLIER if batch else cost

    def __repr__(self):
        return f"<ModelSpec {self.model_id}>"


def is_converse_exempt(config: dict) -> bool:
    """True when the entry can never reach build_converse_params — the one
    exemption from the capability contract. Exempt requires a DECLARED
    model_type with no generative member and at least one RECOGNIZED
    non-Converse member (NON_CONVERSE_TYPES): a missing/empty model_type
    is not exempt (a hand-added Claude entry usually omits it) and neither
    is an unknown value ("Text", "txt" — a typo must stay on the hook).
    Tolerates a hand-edit's string model_type ("embedding",
    "text,vision"). The seeder IMPORTS this function — one implementation,
    no copy to drift."""
    model_type = config.get("model_type") or config.get("modelType") or []
    if isinstance(model_type, str):
        model_type = [t.strip() for t in model_type.split(",") if t.strip()]
    types = {str(t).lower() for t in model_type}
    # EVERY declared type must be a recognized non-Converse type: a typo'd
    # generative sibling beside "embedding" must not silence the contract
    return bool(types) and types <= set(NON_CONVERSE_TYPES)



_WARNED_CONTRACT: dict = {}   # model_id -> last-warned findings signature


def contract_findings(entry: dict) -> list:
    """Every way a GENERATIVE entry's capability contract can be wrong, in
    ONE place: the seeder RAISES on these at deploy time and resolve_model
    WARNS on them at runtime (the net for hand-edited artifacts the seeder
    never sees). A pure embedder has no contract. Each message names the
    condition and its runtime consequence."""
    if is_converse_exempt(entry):
        return []
    caps = entry.get("capabilities") or {}
    if not isinstance(caps, dict):
        return [f"capabilities must be a JSON object, got "
                f"{type(caps).__name__} — the contract cannot be read, so "
                f"requests shape with the key-less defaults"]
    findings = []
    missing = [key for key in CONTRACT_KEYS if key not in caps]
    if missing:
        consequence = {
            "temperature": "temperature on (unless an adaptive "
                           "thinking_default suppresses sampling)",
            "thinking": "thinking absent — 400s on a thinking-required "
                        "family",
            "effort": "effort off — a caller's effort knob is silently "
                      "dropped"}
        findings.append(
            f"lacks capability keys {missing} — those keys shape with "
            f"their key-less defaults: "
            + "; ".join(consequence[key] for key in missing))
    if ("thinking" in caps
            and caps["thinking"] not in VALID_THINKING_SUPPORT):
        findings.append(
            f"thinking {caps['thinking']!r} is not one of "
            f"{VALID_THINKING_SUPPORT} — coerces to 'absent', so no "
            f"thinking field is emitted (400s on a thinking-required "
            f"family)")
    if caps.get("thinking") == "required" and "thinking_default" not in caps:
        findings.append(
            "thinking is required but the entry carries no "
            "thinking_default — quiet requests coerce to 'disabled', "
            "which the adaptive-only families reject")
    if ("thinking_default" in caps
            and caps["thinking_default"] not in VALID_THINKING):
        findings.append(
            f"thinking_default {caps['thinking_default']!r} is not one of "
            f"{VALID_THINKING} — quiet requests coerce to 'disabled'")
    if ("effort_wire" in caps and caps["effort_wire"] not in EFFORT_WIRES):
        findings.append(
            f"effort_wire {caps['effort_wire']!r} is not one of "
            f"{EFFORT_WIRES} — coerces to 'output_config', so the knob is "
            f"emitted on the Anthropic wire and the model will reject it")
    # tuple membership, not a dict lookup: a hand-edited effort_wire can be
    # an unhashable list, and this layer warns — it must never TypeError
    wire_vocab = (EFFORT_WIRE_VOCABULARY[caps["effort_wire"]]
                  if caps.get("effort_wire") in EFFORT_WIRES
                  else EFFORT_WIRE_VOCABULARY["output_config"])
    if ("effort_default" in caps
            and caps["effort_default"] not in wire_vocab):
        findings.append(
            f"effort_default {caps['effort_default']!r} must be one of "
            f"{wire_vocab} for this entry's effort wire — it is emitted "
            f"VERBATIM and the model will reject it")
    if "effort" in caps and not caps.get("effort"):
        for dead in ("effort_default", "effort_wire"):
            if dead in caps:
                findings.append(
                    f"{dead} is set but effort: false — it can never be "
                    f"emitted; drop one to say what the entry means")
    # tuple membership for the same reason as effort_wire: a hand-edited
    # value can be unhashable, and this layer warns — it must never TypeError
    if ("prompt_cache_mode" in caps
            and caps["prompt_cache_mode"] not in PROMPT_CACHE_MODES):
        findings.append(
            f"prompt_cache_mode {caps['prompt_cache_mode']!r} is not one of "
            f"{PROMPT_CACHE_MODES} — coerces to no emission, so the model's "
            f"implicit cache stays on and bills write premiums the entry "
            f"meant to opt out of")
    if "emit_max_tokens" in caps:
        value = caps["emit_max_tokens"]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or value <= 0):
            findings.append(
                f"emit_max_tokens {value!r} must be a positive JSON number "
                f"— coerces to no emission, so Bedrock's omitted-field "
                f"default applies and the truncation the entry meant to "
                f"lift comes back")
    for key in BOOL_CONTRACT_KEYS:
        if key in caps and not isinstance(caps[key], bool):
            findings.append(
                f"capabilities.{key} {caps[key]!r} must be a JSON boolean "
                f"— it reads as "
                f"{'TRUE (knob emitted)' if caps[key] else 'FALSE (knob suppressed)'}")
    # image.limits is plan-ENFORCED arithmetic — the whole container AND
    # its values must be well-shaped, or a hand-edit turns the named
    # plan-time refusal into a bare TypeError/AttributeError per request
    image = caps.get("image")
    if image is not None and not isinstance(image, dict):
        findings.append(
            f"capabilities.image {image!r} must be a JSON object — a "
            f"non-object coerces to 'no image rules', so every declared "
            f"limit silently stops being enforced")
    image_limits = (image or {}).get("limits") if isinstance(image, dict) \
        else None
    if image_limits is not None and not isinstance(image_limits, dict):
        findings.append(
            f"image.limits {image_limits!r} must be a JSON object — a "
            f"non-object coerces to 'no limits declared', so nothing is "
            f"pre-checked")
    for key in ("max_images", "max_image_mb", "max_payload_mb"):
        value = (image_limits or {}).get(key) \
            if isinstance(image_limits, dict) else None
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, float))):
            findings.append(
                f"image.limits.{key} {value!r} must be a JSON number or "
                f"null — the plan multiplies and compares it, so a "
                f"quoted value fails every request instead of refusing "
                f"the oversized one by name")
    if (caps.get("temperature") and "thinking" in caps
            and caps["thinking"] != "absent"
            and caps.get("thinking_default") == "adaptive"):
        # raw declared-value gate, not the coerced thinking_support: a
        # typo'd thinking fires BESIDE its typo finding (one deploy names
        # both defects — hence the conditional phrasing), while a MISSING
        # thinking key stays with the lacks-keys finding alone (its entry
        # emits no thinking field, so temperature genuinely applies)
        findings.append(
            "temperature: true contradicts thinking_default: adaptive — "
            "once thinking resolves adaptive on a quiet request the shaper "
            "suppresses sampling, so the flag can never take effect; set "
            "temperature: false to say what the entry means")
    return findings


def _warn_missing_contract(spec: "ModelSpec") -> "ModelSpec":
    """The runtime half of contract_findings: a hand-edited (or legacy)
    artifact bypasses the seeder, and the key-less defaults 400 every
    request on a thinking-required family with nothing naming the cause —
    so name it here, at resolve time, once per entry per Lambda container
    (`_WARNED_CONTRACT`). A warning rather than a refusal: legacy artifacts
    shape correctly for every family that ever appeared in a seeded
    artifact (4.6-and-earlier tolerate both an omitted thinking field and
    temperature; no 5-family entry ever shipped without its keys), and
    refusing would take out a running deployment over a log-fixable data
    gap. A hand-ADDED key-less entry for a temperature-rejecting family
    (opus-4-7/4-8, the 5s) does mis-shape — that is this warning's
    audience."""
    findings = contract_findings(spec.config)
    signature = tuple(findings)
    # keyed on CONTENT, not just id: a warm container that saw the entry
    # healthy must still warn when a later hand-edit breaks it (and warns
    # again if the findings CHANGE, never for mere repetition)
    if _WARNED_CONTRACT.get(spec.model_id) == signature:
        return spec
    _WARNED_CONTRACT[spec.model_id] = signature
    for message in findings:
        log.warning(
            "registry entry %s: %s — the entry does not carry the contract "
            "the seeder enforces (hand-edited artifact?); the seeded master "
            "is the source of truth", spec.model_id, message)
    return spec


def resolve_model(registry: list, model_id: str) -> ModelSpec:
    """
    Find a registry entry for `model_id`. Exact match first, then substring, the
    same two-tier lookup as modelFactory.js `_getModel`.

    Raises rather than fabricating a spec. An unregistered model would run with no
    pricing (cost silently reported as unknown or zero) and `can_batch()` silently
    false — two wrong answers that look like working ones.
    """
    if not model_id:
        raise ModelNotRegistered("no model id supplied")

    def _mid(config):
        return _get(config, "model_id", "modelId") or ""

    # Exact match — and, since the seeded registry stores BARE ids, a
    # geo-prefixed caller id (`global.anthropic.claude-sonnet-4-6`) matches its
    # bare entry exactly too: the prefix names routing, not identity, and the
    # request-level `inference_region` is where routing is decided now.
    stripped = model_id
    for prefix in _PROFILE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    for config in registry:
        if _mid(config) in (model_id, stripped):
            return _warn_missing_contract(ModelSpec(config))
    matches = [config for config in registry if stripped in _mid(config)]
    if len(matches) == 1:
        return _warn_missing_contract(ModelSpec(matches[0]))
    if len(matches) > 1:
        raise ModelNotRegistered(
            f"model {model_id!r} matches {len(matches)} registry entries "
            f"({', '.join(_mid(config) for config in matches)}); use a full model id")
    raise ModelNotRegistered(
        f"model {model_id!r} is not in the deployed model registry. Add it to "
        f"models/availablemodels_master.json (with this deployment's geo in its "
        f"`geos`) and redeploy — running an unregistered model would report no "
        f"cost and no batch capability.")


def build_chain(registry: list, model_chain: list) -> list:
    """
    Resolve an ordered list of model ids to specs, warning when a pair buys no
    independence.

    Same-family pairs are **allowed but warned about**, not rejected: an operator
    may legitimately have only one family enabled, and refusing to run is worse than
    running with a weaker fallback. M2C's shipped default (`nova-2-lite` →
    `nova-pro`) is exactly the pair this warning exists to catch — shared quota pool,
    shared regional capacity events, so the event the backup exists to survive takes
    out both.
    """
    if not model_chain:
        raise BedrockError("model chain is empty")

    chain: list = []
    for model_id in model_chain:
        spec = resolve_model(registry, model_id)
        if is_converse_exempt(spec.config):
            # an embedding (or other non-Converse) model can never answer a
            # Converse request — sync would 400 every record and a batch job
            # (Converse-dialect, job-level) would fail ALL-OR-NOTHING after
            # the park. Refuse by name at plan time, before any spend.
            raise BedrockError(
                f"model {spec.model_id} is a non-Converse model "
                f"(model_type {spec.config.get('model_type')}) — it cannot "
                f"serve in a run_inference model chain")
        if any(spec.model_id == existing.model_id for existing in chain):
            log.warning("model %s already in the chain; dropping the duplicate",
                        spec.model_id)
            continue
        if chain and spec.family == chain[0].family:
            log.warning(
                "%s and %s are both %s models: same quota pool and same regional "
                "capacity events, so a capacity failure takes out both. Prefer a "
                "cross-family fallback.", chain[0].model_id, spec.model_id, spec.family)
        chain.append(spec)
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Request shaping — ONE function, both paths
# ─────────────────────────────────────────────────────────────────────────────

def normalize_system(system) -> list:
    """
    Accept a plain string or a list of Converse system blocks; emit blocks.

    Kept separate from `messages` and never concatenated into the user turn. That is
    what makes the system prompt a stable cache prefix across every record of a
    request — the per-record content changes, the instructions do not — and its loss
    is visible only on the bill (§12.3).
    """
    if system is None:
        return []
    if isinstance(system, str):
        return [{"text": system}] if system.strip() else []
    blocks = []
    for block in system:
        if isinstance(block, str):
            if block.strip():
                blocks.append({"text": block})
        elif isinstance(block, dict):
            blocks.append(dict(block))
        else:
            raise BedrockError(
                f"system block must be a string or an object, got {type(block).__name__}")
    return blocks


def build_converse_params(spec: ModelSpec, system, messages: list, *,
                          temperature: float = 0.0,
                          thinking: str = "disabled", effort: str | None = None,
                          inference_region: str | None = None) -> dict:
    """
    Build a Converse request. **Used by the sync loop and by `to_batch_record`**, so
    a request cannot be shaped one way in sync and another in batch (§2, §12.1).

    Four model-matrix facts are applied here rather than at the call site:

    - `temperature` is omitted for models whose entry rejects it (`capabilities.temperature: false`)
      **and** whenever thinking is enabled — Bedrock rejects sampling params
      alongside reasoning, and shaping it here is what keeps the two paths identical.
    - `budget_tokens` is never emitted for any model. It is deprecated on 4.6 and
      **400s** on 4.7+ (§9.1).
    - **`maxTokens` is never a CALLER knob** (removed outright 2026-08-18,
      user decision: "one more parameter; one more maintenance cost — keep
      it simple"), **and since 2026-10-16 an entry MAY declare
      `capabilities.emit_max_tokens`** — the deliberate re-addition the
      removal reserved, as registry data. The removal rested on "Bedrock's
      default = the model's maximum output", probed true for every model
      reachable that day — and REFUTED for the thinking-required 5 family
      on 2026-10-16: their omitted-field Converse default is an
      undocumented flat 4096, which adaptive thinking (billed as output)
      pushes structured answers past — every 16-shot SME window truncated
      at stop_reason max_tokens == 4096 exactly. An entry-declared value
      is emitted verbatim on every request (see emit_max_tokens); entries
      without the key emit nothing, exactly as before. Caller-side
      max_tokens stays warned-and-ignored and digest-stripped. The batch
      PADS' internal maxTokens:1 overrides any entry value — 60-99 pads
      per padded job must stay nearly free. stop_reason `max_tokens`
      remains a normal outcome: a model hitting a ceiling still reports it.
    - **No assistant-role prefill for ANY model** — a request-shape rule, not a
      capability (settled 2026-08-11: Claude rejects it family-wide, Llama 4 too,
      and Nova merely tolerates it — a flag that is false or useless everywhere is
      not a flag). Raises rather than silently dropping the message, because a
      caller who built a prefill-dependent prompt (a forced JSON opener) gets an
      unrelated parse failure downstream if the message is just dropped. Applies
      to batch too, for free: `to_batch_record` shapes through this function.

    No OPT-IN prompt caching: the `cache_system_prompt` option was REMOVED
    2026-08-14 (the use cases never cache, Nova rejects an explicit system
    `cachePoint` outright, and batch has no prompt cache at all).
    `to_batch_record` still strips stray `cachePoint` blocks because a CALLER
    may embed one in their own system list. The OPT-OUT is new (2026-10-16):
    entries declaring `prompt_cache_mode` emit `prompt_cache_options` on every
    request, because GPT-6's implicit caching is automatic and bills cache
    writes at 1.25x the input rate for prefixes this stack never re-reads —
    see PROMPT_CACHE_MODES.
    """
    if not messages:
        raise BedrockError("messages is empty; refusing to spend a call on it")
    if thinking not in VALID_THINKING:
        raise BedrockError(
            f"thinking must be one of {VALID_THINKING}, got {thinking!r}")
    if messages[-1].get("role") == "assistant":
        raise BedrockError(
            "the last message must not be assistant-role: Claude and Llama 4 "
            "reject a prefill outright and no supported model needs one — move "
            "the instruction into a user turn or the system prompt")

    # the type the request will actually CARRY — the entry's default can turn
    # reasoning on for a quiet caller (the 5 family defaults adaptive), and
    # Bedrock rejects sampling params beside reasoning regardless of who
    # asked for it, so temperature must key on THIS, not on the caller's ask.
    # thinking="disabled" is also the schema DEFAULT, indistinguishable from
    # an explicit ask at this layer — the entry contract defines what a quiet
    # request means, which is the whole registry-first design
    if spec.accepts_adaptive_thinking():
        emitted_thinking = ("adaptive" if thinking == "adaptive"
                            else spec.default_thinking_type())
    else:
        emitted_thinking = None

    params = {
        # The RUNTIME id under the resolved routing scope — bare registry ids
        # prefixed per inference_region (`global.{id}`, geo, or bare-in-region).
        # Resolved HERE, in the one shaping function, so sync and batch route
        # identically and neither call site re-derives it. On a legacy prefixed
        # id with no override this is the id unchanged.
        "modelId": spec.resolve_runtime_model_id(inference_region),
        "messages": [dict(message) for message in messages],
    }
    # inferenceConfig appears ONLY when it has something to say (probed
    # 2026-08-18: omitting it entirely is accepted by every reachable
    # model). maxTokens only from ENTRY data, never a caller — see the
    # docstring.
    if spec.accepts_temperature() and emitted_thinking != "adaptive":
        params["inferenceConfig"] = {"temperature": float(temperature)}
    emit_max = spec.emit_max_tokens()
    if emit_max:
        params.setdefault("inferenceConfig", {})["maxTokens"] = emit_max
    # (an entry whose adaptive DEFAULT suppresses temperature — data the
    # seeder refuses — is warned at resolve_model, not per record)

    system_blocks = normalize_system(system)
    if system_blocks:
        params["system"] = system_blocks

    additional = {}
    if spec.accepts_adaptive_thinking():
        # "required" implies accepts_adaptive_thinking(), so this one branch
        # covers optional and required alike; "absent" families emit no field.
        # `thinking_support == "required"` (Sonnet 5+ family) means the field
        # must be PRESENT on every request — not merely accepted. WHAT it says
        # when the caller did not ask is the entry's `thinking_default`:
        # "adaptive" across the current 5 family (Sonnet 5.5 joined
        # 2026-10-15; Fable 5 REJECTS an explicit disabled; Opus 5 is
        # specified adaptive), "disabled" for the optional 4.6 family.
        # See default_thinking_type.
        additional["thinking"] = {"type": emitted_thinking}
    # Caller-passed effort wins; the entry's `effort_default` fills otherwise
    # (Claude 5 family: "low"; GPT-6 Luna: "none"). Entries without a default
    # emit no effort field, exactly as before defaults existed. WHICH field
    # carries it is the entry's effort_wire — the caller vocabulary
    # (low/medium/high) is valid on every wire, so a mixed-wire chain never
    # 400s on the knob; the wire-only values (e.g. "none") arrive solely as
    # registry defaults, validated per wire at seed time.
    chosen_effort = effort or spec.default_effort()
    if chosen_effort and spec.accepts_effort():
        # the wire name IS the request-field key — a future third wire added
        # to EFFORT_WIRE_VOCABULARY emits correctly with no edit here
        additional[spec.effort_wire()] = {"effort": chosen_effort}
    # Cache opt-out rides every request for entries that declare it (GPT-6:
    # implicit caching is automatic, writes bill at 1.25x input, and this
    # stack's record-shaped prefixes never get read back). With no
    # breakpoints in the request, explicit mode means "no prompt caching,
    # no cache-write charges" — probed accepted on all three GPT-6 models
    # over Converse 2026-10-16, full inputTokens restored on fresh prefixes.
    cache_mode = spec.prompt_cache_mode()
    if cache_mode:
        additional["prompt_cache_options"] = {"mode": cache_mode}
    if additional:
        params["additionalModelRequestFields"] = additional

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Batch record shaping
# ─────────────────────────────────────────────────────────────────────────────

def to_batch_record(record_id: str, params: dict) -> dict:
    """
    A `build_converse_params` result → one line of a batch job's input JSONL.

    **`modelInput` IS the Converse request** (minus the job-level `modelId`),
    because `create_job` declares `modelInvocationType: "Converse"` — the job
    parameter that decides which dialect Bedrock validates the records in.
    RESOLVED (README.md §P5.4): this function used to
    translate Anthropic records into the native Messages API body, `family`-typed
    pads and all, because a job submitted WITHOUT that parameter defaults to
    `InvokeModel` and judges Converse-shaped records malformed
    (`messages.0.content.0.type: Field required`, 0/100, job "Completed") —
    measured on job 731t09iortbs. The record shape was never the defect; the
    missing job parameter was. v5 passes `'Converse'` on all its LLM batch paths
    (`batch-infer-scene-metadata/index.js:312`).

    Still true, and unchanged:

    1. `modelId` is a **job-level** field, not a per-record one.
    2. No `cachePoint` — batch has no prompt cache, and the block is rejected.
       A rejected job is a 24 h park before anyone notices.
    3. Reasoning config is **kept**. Dropping it makes the same model and the same
       prompt behave differently by path: a reasoning-enabled model would run with
       reasoning in sync and without it in batch.
    """
    model_input = {key: value for key, value in params.items() if key != "modelId"}
    if model_input.get("system"):
        model_input["system"] = _strip_cache_points(model_input["system"])
    # cachePoint can ride in MESSAGE CONTENT too — a caller's own blocks arrive
    # through the request bytes untouched — and batch rejects it there just as
    # it does in system. Copy-on-write: untouched messages pass by reference.
    messages = model_input.get("messages") or []
    if any(isinstance(block, dict) and "cachePoint" in block
           for message in messages
           for block in (message.get("content") or [])):
        model_input["messages"] = [
            {**message, "content": _strip_cache_points(message["content"])}
            if any(isinstance(b, dict) and "cachePoint" in b
                   for b in (message.get("content") or []))
            else message
            for message in messages
        ]
    return {"recordId": record_id, "modelInput": model_input}


def _strip_cache_points(blocks: list) -> list:
    """Drop pure-cachePoint blocks and the `cachePoint` key from mixed ones —
    WITHOUT taking the rest of the block list with it (a dropped list would
    silently unsystem-prompt, or empty, every record it touched)."""
    return [
        {key: value for key, value in block.items() if key != "cachePoint"}
        if isinstance(block, dict) else block
        for block in blocks
        if not (isinstance(block, dict)
                and len(block) == 1 and "cachePoint" in block)
    ]


def padding_record(index: int, spec: ModelSpec | None) -> dict:
    """
    A record that satisfies Bedrock's 100-record job floor at no cost (§5.2).

    `max_tokens: 1`, and the `pad_` prefix is what `Collect batch output` filters on —
    which is why `parse_records` rejects a real `record_id` using that prefix.

    Converse-shaped like every real record (`modelInvocationType: "Converse"` is
    job-level, and a job validates ALL-OR-NOTHING — a pad that cannot be scored
    converts a padded job into a wholly failed one, which is how the 2026-08-09
    dialect defect announced itself: 100 rejected, 99 of them option-free pads).

    With a `spec`, the pad is shaped through the SAME build_converse_params the
    real records use, so it honors the entry's contract — no temperature on a
    family that rejects it, the thinking field where it is required. This was
    a LIVE landmine, not a hypothetical: opus-5 batches (can_support: true)
    and is thinking-required, and both shapes were probed on it 2026-10-15 —
    the historical hardcoded pad (temperature: 0, no thinking) is REJECTED
    ("`temperature` is deprecated for this model"), so any sub-100-record
    opus-5 batch job failed ALL-OR-NOTHING on its own pads; the spec-shaped
    pad (thinking adaptive + effort low + maxTokens 1) is ACCEPTED and
    answers stopReason max_tokens at 1 output token. The new shapes were
    then validated with REAL CreateModelInvocationJob runs (2026-10-15)
    against EVERY batchable generative model the eu deployment serves —
    sonnet-4-6, opus-5, haiku-4-5 via global., nova-2-lite via global.,
    nova-pro and nova-micro via eu. (also live-confirming batch-follows-
    inference_region in the eu direction): all six jobs Completed 100/100
    with errorRecordCount 0.
    The spec-less form survives for callers without one, but emits ONLY
    maxTokens — the historical temperature: 0 is exactly what the probe
    showed fatal, and no model requires a temperature, so omitting it is
    strictly more compatible. (A thinking-REQUIRED model still needs the
    spec-shaped form; the spec-less pad carries no thinking field.)
    """
    if spec is not None:
        params = build_converse_params(
            spec, None,
            [{"role": "user", "content": [{"text": "Hi"}]}])
        config = dict(params.get("inferenceConfig") or {})
        # pads need no sampling at 1 output token, and temperature is the
        # knob the probe showed fatal where mis-declared — drop it even
        # when the entry accepts it: maximally compatible for free
        config.pop("temperature", None)
        config["maxTokens"] = 1          # machinery, never a caller knob
        params["inferenceConfig"] = config
    else:
        params = {"messages": [{"role": "user", "content": [{"text": "Hi"}]}],
                  "inferenceConfig": {"maxTokens": 1}}
    return {
        "recordId": _pad_record_id(index),
        "modelInput": to_batch_record("pad", params)["modelInput"],
    }


def _pad_record_id(index: int) -> str:
    """The one home for the pad id format — padding_record and
    padding_records both derive from it, so the grammar cannot fork
    mid-job."""
    return f"{_PAD_PREFIX}{index:04d}"


def padding_records(count: int, spec: ModelSpec | None):
    """Yield `count` pads, SHAPING the modelInput once (the expensive half)
    and deep-copying it per pad — a tiny dict, so the copies are free and no
    two pads alias state a future consumer could mutate together."""
    if count <= 0:
        return
    first = padding_record(0, spec)
    # snapshot BEFORE yielding: a consumer that mutates pad 0 between
    # pulls must not have its mutation copied into pads 1..N
    template = copy.deepcopy(first["modelInput"])
    yield first
    for index in range(1, count):
        yield {"recordId": _pad_record_id(index),
               "modelInput": copy.deepcopy(template)}


def is_padding(record_id: str) -> bool:
    return str(record_id or "").lower().startswith(_PAD_PREFIX)


def resolve_batch_model_id(spec: ModelSpec,
                           inference_region: str | None = None) -> str:
    """
    The model id a batch job takes. **Batch follows `inference_region`** — the
    same routing scope, default or request-overridden, that sync uses; there is
    no separate batch region in the seeded registry (settled 2026-08-11: a
    caller who pinned the run to `eu` for residency must not have its batch half
    silently route `global`).

    v5 note, kept for the reader who remembers it: `base.js:resolveBatchJobModelId`
    re-derived a distinct geo from `batchJob.region` because its runtime ids were
    pre-prefixed per deployment. With bare ids and an explicit routing scope, the
    resolution is the same one sync uses.
    """
    if not spec.can_batch():
        from lib.custom_exceptions import ValidationException
        raise ValidationException(
            f"{spec.model_id} does not support batch inference")
    return spec.resolve_runtime_model_id(inference_region or spec.batch_scope)


# ─────────────────────────────────────────────────────────────────────────────
# Converse
# ─────────────────────────────────────────────────────────────────────────────

def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _backoff_seconds(attempt: int) -> float:
    return min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** attempt))


def error_code(exc) -> str:
    """Botocore ClientError code, falling back to the exception class name."""
    response = getattr(exc, "response", None) or {}
    code = (response.get("Error") or {}).get("Code")
    return code or type(exc).__name__


# Kept as a private alias: conversation-analysis's copy names it `_error_code`, and
# the two files are diffed against each other.
_error_code = error_code


def _stamp_rounds(exc, rounds: int) -> None:
    """
    Record how many model calls an exception cost.

    Guarded because not every exception object accepts arbitrary attributes — a
    `__slots__` type would raise here, and losing the round count must not turn a
    Bedrock error into an `AttributeError` from the accounting code.
    """
    try:
        exc.rounds = int(rounds)
    except Exception:                                              # noqa: BLE001
        pass


def _rounds_of(exc) -> int:
    try:
        return int(getattr(exc, "rounds", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _accumulate(total: dict, usage: dict) -> dict:
    for key, value in (usage or {}).items():
        if key.endswith("Tokens") and isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
    return total


def extract_text(response: dict) -> str:
    """
    Concatenate the text blocks of a Converse **or** an Anthropic Messages response.

    `reasoningContent` blocks are deliberately skipped: they are not the answer, and
    a caller parsing XML or JSON out of `output_text` would find the model's
    scratchpad prepended to it. The Messages equivalent is `type: "thinking"`, which
    the `"text" in block` test already excludes — a thinking block's payload lives
    under `thinking`, not `text`.

    Two shapes because batch `modelOutput` is an **InvokeModel** response, so an
    Anthropic batch row is `{"content": [...]}` at the top level while a Converse
    (sync, or Nova batch) row is `{"output": {"message": {"content": [...]}}}`. Both
    are read here rather than at the call site: `collect_batch` handles one row shape
    and cannot know the family's response dialect.
    """
    blocks = (((response.get("output") or {}).get("message") or {}).get("content")
              or [])
    if not blocks and isinstance(response.get("content"), list):
        blocks = response["content"]
    # isinstance, NOT `"text" in block`: Opus 5 batch output writes each content
    # block as an EXPLICIT-NULL UNION — every possible field present, the unused
    # ones null — so membership is true on every block, including the adaptive-
    # thinking reasoning block whose `text` is null. `None` reaching the join was
    # the 2026-08-14 matrix red (`sequence item 0: expected str instance`), the
    # third member of the explicit-null family after `cacheReadInputTokens: null`.
    return "".join(block["text"] for block in blocks
                   if isinstance(block, dict)
                   and isinstance(block.get("text"), str))


_extract_text = extract_text


class InferenceResult:
    """One completed model call, successful or refused."""

    def __init__(self, *, text: str, stop_reason: str, usage: dict,
                 model_id: str, spec: ModelSpec, attempts: int,
                 latency_ms: int, notes: list, rounds: int | None = None):
        self.text = text
        self.stop_reason = stop_reason
        self.usage = usage
        self.model_id = model_id
        self.spec = spec
        self.attempts = attempts
        self.latency_ms = latency_ms
        self.notes = notes
        # Billable model calls behind this result. Distinct from `attempts`, which
        # counts only the successful model's tries: a result reached after model A
        # burned three throttled calls cost four rounds, not one, and `rounds`
        # is what `summary.bedrock_usage.rounds` reports (§4.2). Only usage from the
        # answering model is available, so rounds is the honest count of what was
        # *called* even where token totals for the abandoned calls are lost.
        self.rounds = int(attempts if rounds is None else rounds)

    @property
    def truncated(self) -> bool:
        """`max_tokens` is a normal outcome, not a failure."""
        return self.stop_reason == "max_tokens"

    @property
    def refused(self) -> bool:
        return self.stop_reason in REFUSAL_STOP_REASONS

    @property
    def estimated_cost(self) -> float | None:
        return self.spec.estimate_cost(self.usage)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "stop_reason": self.stop_reason,
            "truncated": self.truncated,
            "usage": dict(self.usage),
            "estimated_cost": self.estimated_cost,
            "attempts": self.attempts,
            "rounds": self.rounds,
            "latency_ms": self.latency_ms,
            "notes": list(self.notes),
        }


def _converse_once(client, params: dict, notes: list) -> dict:
    """
    One Converse call, with a single ValidationException recovery for `temperature`.

    The strip is targeted and **recorded**. `core-lib`'s base.js deletes temperature
    and topP before *every* retry regardless of cause, so a run configured at 0.0
    silently finishes at the model's default sampling — and the log says only
    "RETRY AGAIN". Nobody learns which parameter the model actually rejected.
    """
    try:
        return client.converse(**params)
    except Exception as exc:
        if error_code(exc) != "ValidationException":
            raise
        message = str(exc).lower()
        if ("temperature" not in message
                or "temperature" not in params.get("inferenceConfig", {})):
            raise
        # Drop ONLY temperature — since emit_max_tokens (2026-10-16) the
        # block may also carry an entry's maxTokens lift, and dropping it
        # would silently restore the 4096-truncation the entry declared the
        # lift against. An emptied block is removed (the omitted shape the
        # retirement probe verified).
        config = dict(params.get("inferenceConfig") or {})
        config.pop("temperature", None)
        if config:
            params["inferenceConfig"] = config
        else:
            params.pop("inferenceConfig", None)
        note = (f"{params['modelId']} rejected `temperature`; retried without it — "
                f"sampling is at the model default, not the configured value")
        log.warning("%s", note)
        notes.append(note)
        return client.converse(**params)


# Content-block types whose `source` may carry media. `bytes` is decoded at the
# SYNC boundary only; `s3Location` (which the Nova family accepts as a source)
# passes through untouched on both paths — there is nothing to decode.
_BLOB_BLOCK_TYPES = ("image", "document", "video")


def decode_blob_sources(messages: list) -> list:
    """
    Base64 `source.bytes` strings → raw bytes, copy-on-write, for the SYNC path.

    THE TWO TRANSPORTS WANT OPPOSITE THINGS, and this function is the sync half
    of that asymmetry. A record travels as JSON (`request.jsonl`), so its image
    bytes can only be a base64 STRING. Batch forwards that string verbatim —
    the job's input JSONL is itself JSON and Bedrock decodes it there. Sync
    hands the block to boto3, whose blob type treats a str as the RAW payload
    and re-encodes it — so the model received base64-of-base64 and answered
    `The detected file MIME type text/plain does not match the expected type
    image/png` (nova-pro, live, 2026-08-13). Every vision model failed on sync
    while the identical record PASSED via batch; that asymmetry was the proof.

    Copy-on-write: only the message/block/source dicts on the path to a decoded
    value are copied, so the caller's records — which are re-read on checkpoint
    re-entry — are never mutated. Blocks without a str `source.bytes` (including
    `s3Location` sources) pass through by reference.
    """
    decoded_messages = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            decoded_messages.append(message)
            continue
        new_content = []
        touched = False
        for block in content:
            blob = None
            if isinstance(block, dict):
                for block_type in _BLOB_BLOCK_TYPES:
                    inner = block.get(block_type)
                    if isinstance(inner, dict) and isinstance(
                            (inner.get("source") or {}).get("bytes"), str):
                        blob = (block_type, inner)
                        break
            if blob is None:
                new_content.append(block)
                continue
            block_type, inner = blob
            new_content.append({**block, block_type: {
                **inner, "source": {
                    **inner["source"],
                    "bytes": base64.b64decode(inner["source"]["bytes"])}}})
            touched = True
        decoded_messages.append(
            {**message, "content": new_content} if touched else message)
    return decoded_messages


def converse(client, spec: ModelSpec, system, messages: list, *,
             temperature: float = 0.0,
             thinking: str = "disabled", effort: str | None = None,
             max_retries: int = 3, inference_region: str | None = None,
             sleep=_sleep) -> InferenceResult:
    """
    Call one model, retrying transient failures on that same model.

    Retries cost no output tokens, so a bounded backoff here is close to free and is
    the cheapest thing in the chain. Raises the last exception for anything the
    caller's model chain should handle.
    """
    params = build_converse_params(
        spec, system, messages, temperature=temperature,
        thinking=thinking, effort=effort, inference_region=inference_region)
    # AFTER shaping, at the transport boundary: shaping stays identical for sync
    # and batch (to_batch_record shares build_converse_params), and only the
    # sync wire format wants raw bytes — see decode_blob_sources.
    params["messages"] = decode_blob_sources(params["messages"])

    usage_total: dict = {}
    notes: list = []
    started = time.monotonic()
    attempt = 0

    while True:
        try:
            response = _converse_once(client, params, notes)
        except Exception as exc:
            code = error_code(exc)
            # Rounds spent on this model, stamped on the exception so the chain walk
            # can total them. An attribute rather than a wrapper class: the caller
            # dispatches on `error_code(exc)`, and re-wrapping would hide the code
            # one level down where `ESCALATE_TO_NEXT_MODEL` no longer matches it.
            _stamp_rounds(exc, attempt + 1)
            if code in ESCALATE_TO_NEXT_MODEL:
                # Backoff cannot fix an exhausted quota or a model this role cannot
                # invoke. Escalating immediately is both faster and correct.
                log.warning("%s: %s — escalating to the next model without retry",
                            spec.model_id, code)
                raise
            if code in RETRY_ON_SAME_MODEL and attempt < max_retries:
                delay = _backoff_seconds(attempt)
                attempt += 1
                log.warning("%s: %s — retry %d/%d in %.0fs",
                            spec.model_id, code, attempt, max_retries, delay)
                sleep(delay)
                continue
            raise

        _accumulate(usage_total, response.get("usage") or {})
        stop_reason = response.get("stopReason") or ""
        latency_ms = ((response.get("metrics") or {}).get("latencyMs")
                      or int((time.monotonic() - started) * 1000))

        # the telling: a bill that silently includes reconstructed tokens
        # would read as if the model reported them (same discipline as the
        # temperature-strip note). Checked on the RAW response usage —
        # _accumulate never invents keys, so absence here is Bedrock's.
        hidden = reconstructed_cache_write(usage_total)
        if hidden:
            notes.append(
                f"usage under-itemized: {hidden} input tokens counted only "
                f"in totalTokens — priced at the cache-write rate")

        return InferenceResult(
            text=extract_text(response), stop_reason=stop_reason,
            usage=usage_total, model_id=spec.model_id, spec=spec,
            attempts=attempt + 1, latency_ms=latency_ms, notes=notes)


def converse_with_chain(client, chain: list, system, messages: list, *,
                        temperature: float = 0.0,
                        thinking: str = "disabled", effort: str | None = None,
                        max_retries: int = 3,
                        inference_region: str | None = None,
                        sleep=_sleep) -> InferenceResult:
    """
    Walk the model chain until one answers. **Per record** (§5.3) — one poisoned
    record must not downgrade the other 39.

    Escalates on quota/unavailable errors, on any non-retryable error, and on a
    refusal stopReason — a different family may not filter the same content, which is
    the second reason a fallback should be cross-family.

    A `ValidationException` **never advances the chain**: a malformed request is
    malformed for every model, and walking the chain on it just multiplies the error
    and the latency. A refusal from the last model is returned, not raised: the
    caller records it against that record and continues with the request.
    """
    if not chain:
        raise BedrockError("model chain is empty")

    errors, codes = [], []
    spent = 0                      # billable rounds across every model tried so far
    for index, spec in enumerate(chain):
        last = index == len(chain) - 1
        try:
            result = converse(
                client, spec, system, messages,
                temperature=temperature, thinking=thinking, effort=effort,
                max_retries=max_retries,
                inference_region=inference_region, sleep=sleep)
        except Exception as exc:
            code = error_code(exc)
            spent += _rounds_of(exc)
            errors.append(f"{spec.model_id}: {code}: {exc}")
            codes.append(code)
            if code == "ValidationException":
                raise BedrockError(
                    "malformed request — not retried on any other model: "
                    + " | ".join(errors), codes=codes, rounds=spent) from exc
            if last:
                raise BedrockError(
                    "every model in the chain failed — " + " | ".join(errors),
                    codes=codes, rounds=spent) from exc
            log.warning("%s failed (%s); falling back to %s",
                        spec.model_id, code, chain[index + 1].model_id)
            continue

        spent += result.attempts
        result.rounds = spent

        if result.refused and not last:
            errors.append(f"{spec.model_id}: stopReason={result.stop_reason}")
            log.warning("%s returned %s; falling back to %s",
                        spec.model_id, result.stop_reason, chain[index + 1].model_id)
            continue

        if index > 0:
            result.notes.append(
                f"fell back from {chain[0].model_id} after: " + " | ".join(errors))
        if result.refused:
            # Last model also refused. Returned, not raised, so this record is
            # recorded as refused and the rest of the request still completes.
            log.error("all %d model(s) refused: %s", len(chain), "; ".join(errors))
        return result

    raise BedrockError("model chain exhausted without a result")


def is_saturated(exc) -> bool:
    """
    True when a failure means "the quota wall", not "this record is bad".

    Checks a raw botocore code **and** `BedrockError.codes`, because by the time the
    chain has exhausted every model the original exception is wrapped and the raw
    code is no longer on the outside.
    """
    codes = set(getattr(exc, "codes", None) or [])
    codes.add(error_code(exc))
    return bool(codes & (THROTTLING_EXCEPTIONS | QUOTA_EXCEPTIONS))
