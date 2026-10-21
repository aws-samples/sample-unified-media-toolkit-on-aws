"""
StateSubmitMediaProcessing — HELD: submits data.media_processing_request
to the media-processing GATEWAY with this state's task token (the
family sibling pattern, 2026-09-18 — replaces the direct
states:startExecution.sync:2 of MP's state machine). MP's waker
completes the token with the submit handle + terminal status AS DATA;
StateExtractProcessedAudio checks the status and derives the wav key
from MP's output contract.

Idempotency: the token is derived from THIS run's uuid, so a redriven
execution absorbs into the same MP run instead of re-transcoding.

MP is a HARD dependency here (no audio -> nothing to diarize), so
every failure on this path RAISES — failing the wait state fast —
rather than self-completing a skip.
"""
from __future__ import annotations

import logging
import os

import boto3

from lib.gateway import invoke_gateway_tool, require_callback_feature

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_lam = boto3.client("lambda")


def lambda_handler(event: dict, context) -> dict:
    uuid = event["uuid"]
    task_token = event["task_token"]
    gateway_arn = (event.get("mp_gateway_arn")
                   or os.environ.get("MEDIA_PROCESSING_GATEWAY_ARN", ""))
    if not gateway_arn:
        raise RuntimeError("media-processing gateway ARN not supplied "
                           "(ASL mp_gateway_arn or env)")
    request = (event.get("data") or {}).get("media_processing_request")
    if not request:
        raise RuntimeError("data.media_processing_request is missing — "
                           "StateCheckAndPreprocess did not prepare the "
                           "conversion request")

    require_callback_feature(_lam, gateway_arn, "media-processing")

    resp = invoke_gateway_tool(
        _lam, gateway_arn, "media-processing", "process_media",
        {**request,
         "idempotency_token": f"diar-{uuid}",
         "callback_task_token": task_token})
    if not (isinstance(resp, dict) and resp.get("job_id")):
        # a handle-less response is a refusal; a response WITH a job_id
        # means media-processing owns the token (absorbed runs included)
        raise RuntimeError(f"media-processing refused the submit: "
                           f"{str(resp)[:300]}")
    log.info("parked on media-processing %s (%s)",
             resp.get("workflow_id"), resp.get("status"))
    return {"parked": True, "job_id": resp["job_id"]}
