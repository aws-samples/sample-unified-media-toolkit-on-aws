"""
bedrock_utils.py — shared Bedrock client and helpers.

Single boto3 bedrock-runtime client shared across all callers in the same
Lambda container. Model IDs are passed through as-is from the workflow config.

Two call patterns:
  call_slm / call_slm_with_fallback — text generation via converse() API.
      Works across all generative model families (Nova, Claude, etc.) without
      caller knowledge of per-model request/response formats.

  embed — vector embedding via invoke_model().
      converse() is not supported by embedding models; invoke_model() is
      required. Dispatches request/response format to the appropriate provider
      module in lib/providers/.
"""
from __future__ import annotations

import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

from lib.input_schema import AWS_REGION
from lib.providers import cohere_embed_v4, titan_multimodal_embed, nova_multimodal_embed

log = logging.getLogger(__name__)

_bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# Models Bedrock serves ONLY through an inference profile (no on-demand
# throughput — the ValidationException names the condition). The SCHEMA
# keeps the BARE model id: it names the vector space and is baked into
# every LanceDB table path and usage row, so it must never change; the
# profile is applied HERE, at invoke time only. Default profile is
# `global.` (user decision 2026-09-03: one id in every region, maximum
# routing headroom; verified live from eu-west-1 and us-west-2). A
# deployment that must keep embedding IN-GEOGRAPHY (data residency)
# sets EMBED_INFERENCE_PROFILE_PREFIX to its geo (`us`/`eu`/`apac`) —
# geo profiles only work from their own geography. The IAM policy
# already grants inference-profile/*.cohere.embed-* plus
# wildcard-region foundation-model ARNs.
_PROFILE_PREFIX = os.environ.get("EMBED_INFERENCE_PROFILE_PREFIX",
                                 "global")
_PROFILE_ONLY_MODELS = frozenset(
    m.strip() for m in os.environ.get(
        "EMBED_PROFILE_ONLY_MODELS", "cohere.embed-v4:0").split(",")
    if m.strip())


def invocation_id(model_id: str) -> str:
    """The id to INVOKE with: the logical model id everywhere else, the
    geo inference profile for profile-only models."""
    if model_id in _PROFILE_ONLY_MODELS:
        return f"{_PROFILE_PREFIX}.{model_id}"
    return model_id

# ── converse() error codes that warrant trying the fallback model ──────────────
_CONTENT_FILTER_CODES = {"ContentFilteredException", "ValidationException",
                         "AccessDeniedException"}
_MODEL_ERROR_CODES    = {"ModelNotReadyException", "ModelErrorException",
                         "ModelStreamErrorException", "ModelTimeoutException"}


def call_slm(model_id: str, system_prompt: str, user_prompt: str) -> str:
    """
    Call a generative model via converse(). Returns the first text content block.
    Raises ClientError on any Bedrock error — caller decides whether to retry/fallback.
    """
    kwargs: dict = {
        "modelId":         model_id,
        "messages":        [{"role": "user", "content": [{"text": user_prompt}]}],
        "inferenceConfig": {"maxTokens": 512, "temperature": 0.0},
    }
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]
    resp = _bedrock.converse(**kwargs)
    return resp["output"]["message"]["content"][0]["text"]


def call_slm_with_fallback(
    primary: str,
    fallback: str | None,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str, bool, str | None]:
    """
    Call primary model; on content-filter or model error, retry with fallback.
    ThrottlingException is re-raised so SFN retry handles it.

    Returns (text, model_used, fallback_triggered, fallback_reason).
    """
    try:
        text = call_slm(primary, system_prompt, user_prompt)
        return text, primary, False, None
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in _CONTENT_FILTER_CODES or code in _MODEL_ERROR_CODES:
            if fallback:
                log.warning("Primary model %s failed (%s) — trying fallback %s",
                            primary, code, fallback)
                text = call_slm(fallback, system_prompt, user_prompt)
                return text, fallback, True, code
        raise


def _resolve_embed_provider(model_id: str):
    """Return the provider module that matches this model_id."""
    if titan_multimodal_embed.MODEL_PREFIX in model_id:
        return titan_multimodal_embed
    if nova_multimodal_embed.MODEL_PREFIX in model_id:
        return nova_multimodal_embed
    return cohere_embed_v4


def embed(model_id: str, text: str, dims: int | None = None) -> list[float]:
    """
    Generate a text embedding via invoke_model().
    Provider is selected by model_id; each provider module in lib/providers/
    owns its own request format and response parsing.
    """
    provider = _resolve_embed_provider(model_id)
    body     = provider.build_request(text, dims)
    resp     = _bedrock.invoke_model(
        modelId=invocation_id(model_id),
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return provider.parse_response(result)


def embed_document(model_id: str, text: str | None = None,
                   image_b64: str | None = None,
                   dims: int | None = None,
                   usage: dict | None = None) -> list[float]:
    """
    DOCUMENT-side embedding for indexing. Differs from embed()/embed_image()
    (query-side): Cohere gets input_type=search_document, and text+image
    together produce a JOINT (mixture) embedding where the provider supports
    it (Cohere, Titan; Nova falls back to image-only).

    usage: optional caller-owned accumulator, updated in place per model_id
    (invocations / text_characters / images / input_tokens where the model
    reports them). Feeds the per-asset indexer_embed_usage.json artifact —
    the only record of embedding spend now that indexing owns the Bedrock
    calls. Known limitation: the accumulator is in-memory per Lambda
    attempt — calls made by an attempt that later fails and retries are
    not counted (the artifact reflects the SUCCESSFUL attempt only; treat
    it as a lower bound under throttling, reconcile exact spend via
    CloudWatch/Bedrock invocation metrics).
    """
    provider = _resolve_embed_provider(model_id)
    body     = provider.build_document_request(text, image_b64, dims)
    resp     = _bedrock.invoke_model(
        modelId=invocation_id(model_id),
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    if usage is not None:
        # Count what was actually SENT, not what the caller supplied — Nova's
        # build_document_request drops text when an image is present
        # (image-wins), so len(text) would overstate text spend there.
        text_sent = bool(text) and not (
            image_b64 and provider is nova_multimodal_embed)
        u = usage.setdefault(model_id, {
            "invocations": 0, "text_characters": 0, "images": 0, "input_tokens": 0})
        u["invocations"]     += 1
        u["text_characters"] += len(text) if text_sent else 0
        u["images"]          += 1 if image_b64 else 0
        # Titan reports inputTextTokenCount; others report nothing — the
        # character/image counts above are the portable fallback.
        u["input_tokens"]    += int(result.get("inputTextTokenCount") or 0)
    return provider.parse_response(result)


def embed_image(model_id: str, image_b64: str, dims: int | None = None) -> list[float]:
    """
    Generate an image embedding via invoke_model().
    For Cohere: passes {"images": ["data:image/jpeg;base64,<b64>"]}, which produces
    vectors in the same embedding space as the indexed shot/scene vectors.
    """
    provider = _resolve_embed_provider(model_id)
    body     = provider.build_image_request(image_b64, dims)
    resp     = _bedrock.invoke_model(
        modelId=invocation_id(model_id),
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return provider.parse_response(result)
