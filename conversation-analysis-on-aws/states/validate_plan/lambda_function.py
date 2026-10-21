"""
StateValidateAndPlan (README.md §7): everything that can refuse, refuses HERE
— before any inference dollar. Fetches and normalizes the transcript (§7a),
fetches the prompt (get_prompt, version-less → `default`, §1), validates
the model chain against bedrock-inference's registry (§7c), plans the
chunks (§5), derives the history scope (§7b), and persists the whole plan
to `_plan/plan.json` so every later state reads ONE document instead of
threading a large payload through Step Functions state I/O (the 256 KB
limit is why states pass URIs, never bodies — §8).
"""
from __future__ import annotations

import logging
import re

from lib import ca_common
from lib import chunker
from lib import transcript as transcript_mod
from lib import workflow_history

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_DIARIZATION_RUN_RE = re.compile(
    r"micro-workflow/speech-diarization/([A-Za-z0-9-]+)/")


class ConversationAnalysisValidationError(Exception):
    """Refusal before any spend; the message names what to fix."""


def lambda_handler(event: dict, context) -> dict:
    workflow_id = event["uuid"]
    request = event.get("input") or {}

    uri = (request.get("transcript_s3uri") or "").strip()
    if not uri:
        raise ConversationAnalysisValidationError(
            "transcript_s3uri is required — a speech-diarization run's "
            "final_transcript.json (preferred) or a WebVTT file on the "
            "proxy bucket")
    bucket, key = ca_common.parse_s3_uri(uri)
    if bucket != ca_common.PROXY_BUCKET:
        raise ConversationAnalysisValidationError(
            f"transcript_s3uri must be on the proxy bucket "
            f"({ca_common.PROXY_BUCKET}) — got bucket {bucket!r}; copy the "
            "transcript there first")

    normalized = transcript_mod.load_transcript(
        ca_common.read_text(bucket, key),
        transcript_format=request.get("transcript_format") or None)
    sentences = normalized["sentences"]

    prompt = _fetch_prompt((request.get("prompt_version") or "").strip())
    # Caller-absent defaults are applied HERE and land in plan.json as the
    # concrete values (provenance never says "default"; it names the ids).
    model_chain = _validate_model_chain(
        request.get("model_chain") or list(ca_common.DEFAULT_MODEL_CHAIN))

    plan_chunks = chunker.plan_chunks(sentences,
                                      ca_common.CHUNK_BUDGET_TOKENS)
    scope = _derive_scope(request.get("history_scope"), uri)

    prefix = ca_common.run_prefix(workflow_id)
    plan = {
        "workflow_id": workflow_id,
        "tool_version": ca_common.TOOL_VERSION,
        "transcript": {
            "s3uri": uri,
            "source_format": normalized["source_format"],
            "sentences_count": len(sentences),
            "duration_sec": normalized["duration_sec"],
        },
        "prompt": {
            "category": ca_common.PROMPT_CATEGORY,
            "version": prompt["version"],
            "source_md5": prompt["source_md5"],
            "stage": prompt["stage"],
            "system": prompt["system"],
            "user_template": prompt["user_template"],
            "variables": prompt["variables"],
        },
        "model_chain": model_chain,
        # Explicit False, not null (user decision 2026-08-18): absent means
        # "never batch", not "let the dollar guard decide" — predictable
        # latency for the common interactive case; callers opt in.
        "use_batch": bool(request.get("use_batch")),
        "scope": scope,
        "budget_tokens": ca_common.CHUNK_BUDGET_TOKENS,
        "chunks": plan_chunks,
        "output_prefix": prefix,
    }
    plan_key = f"{prefix}/_plan/plan.json"
    ca_common.write_json(ca_common.PROXY_BUCKET, plan_key, plan)
    logger.info("plan: %d sentences -> %d chunk(s), tiers %s, scope %s",
                len(sentences), len(plan_chunks),
                [c["cut_tier"] for c in plan_chunks], scope)

    return {
        "uuid": workflow_id,
        "plan_s3uri": f"s3://{ca_common.PROXY_BUCKET}/{plan_key}",
        "chunk_count": len(plan_chunks),
    }


def _fetch_prompt(version: str = "") -> dict:
    """get_prompt: version-less binds `default` (§1); the caller's
    prompt_version binds a NAMED version (e.g. "default_news"), and a named
    version that does not exist refuses HERE, before any spend — the
    registry never falls back silently."""
    params = {"category": ca_common.PROMPT_CATEGORY}
    if version:
        params["version"] = version
    artifact = ca_common.invoke_gateway_tool(
        ca_common.PROMPT_REGISTRY_ARN, "prompt-registry___get_prompt",
        params)
    if artifact.get("error"):
        raise ConversationAnalysisValidationError(
            f"could not fetch the {ca_common.PROMPT_CATEGORY} prompt"
            f"{' version ' + version if version else ''}: "
            f"{artifact.get('message')}")
    declared = {v["token"] for v in artifact["variables"]}
    # The workflow-provided values catalog (§1): we can supply exactly
    # {{transcript}} today; a declared token outside the catalog must fail
    # HERE, loudly, naming it — never at render inside a paid submit.
    unsupplied = declared - {"{{transcript}}"}
    if unsupplied:
        raise ConversationAnalysisValidationError(
            f"the {artifact['version']} prompt declares variables this "
            f"workflow cannot supply: {sorted(unsupplied)} — the catalog "
            "provides only {{transcript}} (the self-describing envelope)")
    return artifact


def _validate_model_chain(chain) -> list:
    """§7c: caller-driven, validated here. The conversation-analysis
    default (ca_common.DEFAULT_MODEL_CHAIN, sonnet-4-6) arrives through
    the same door and is validated identically — a deployment whose
    registry lacks the default fails HERE, loudly, at plan time."""
    if isinstance(chain, str):
        chain = [part.strip() for part in chain.split(",") if part.strip()]
    if not isinstance(chain, list):
        raise ConversationAnalysisValidationError(
            "model_chain must be a list of Bedrock model ids (or a "
            "comma-separated string), primary first")
    for model_id in chain:
        entry = ca_common.invoke_gateway_tool(
            ca_common.BEDROCK_GATEWAY_ARN, "bedrock-inference___get_model",
            {"model_id": model_id})
        if entry.get("error"):
            raise ConversationAnalysisValidationError(
                f"model {model_id!r} is not in this deployment's registry "
                "— call bedrock-inference list_models for the legal ids")
        model = entry.get("model") or entry
        types = model.get("model_type") or []
        if types and "text" not in types:
            raise ConversationAnalysisValidationError(
                f"model {model_id!r} is not a text model "
                f"(model_type={types}) — this workflow renders text-only "
                "records")
    return chain


def _derive_scope(explicit, transcript_uri: str) -> str:
    """§7b: the caller's own asset binding wins; else the upstream
    diarization workflow_id; else a deterministic hash of the uri."""
    if explicit:
        scope = str(explicit).strip()
        if not workflow_history._SCOPE_RE.match(scope):
            # Reaching into the copied module's pinned charset on purpose:
            # ONE regex decides what a scope is, and it lives with the key
            # builder it protects.
            raise ConversationAnalysisValidationError(
                f"history_scope {scope!r} must match "
                f"{workflow_history._SCOPE_RE.pattern}")
        return scope
    m = _DIARIZATION_RUN_RE.search(transcript_uri)
    if m:
        return f"transcript.{m.group(1)}"
    import hashlib
    return "transcript." + hashlib.sha1(
        transcript_uri.encode("utf-8"),
        usedforsecurity=False).hexdigest()[:12]
