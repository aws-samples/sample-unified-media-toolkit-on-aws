"""
RerankFramesByVision — HELD: the ONE model seat. All grabbed frames go
to bedrock-inference as ONE vision record (the validated prototype
shape: the judge sees every archetype group in one request and ranks
WITHIN each; it orders and flags pre-qualified candidates, it cannot
add one). The record is STAGED as records_s3uri — base64 frames blow
the 256 KB invoke payload, and bedrock-inference's staged-records path
exists for exactly this. Self-completed skips: no frames, no bedrock
gateway, prompts not bound, submit failure — Assemble then ships the
metadata order flagged rerank=skipped.
"""
from __future__ import annotations

import base64
import json
import logging
import os

import boto3

from lib import s3io
from lib import wf_common

logger = logging.getLogger(__name__)

BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")

_lam = boto3.client("lambda")
_sfn = boto3.client("stepfunctions")


def build_record(doc: dict, system: str) -> dict | None:
    """One vision record over every candidate WITH a frame; None when
    no frame survived the grab."""
    content = []
    n_frames = 0
    for archetype, cands in doc["slates"].items():
        with_frames = [c for c in cands if c.get("frame")]
        if not with_frames:
            continue
        content.append({"text": f"--- archetype: {archetype} "
                                f"({len(with_frames)} candidates, "
                                f"metadata order; a candidate may "
                                f"contribute several frames of one "
                                f"shot) ---"})
        for cand in with_frames:
            # person candidates carry several frames of ONE shot
            # (FR-informed grab) — the judge ranks frames, assemble
            # keeps each candidate's best
            for uri in cand.get("frames") or [cand["frame"]]:
                fname = uri.rsplit("/", 1)[-1]
                data = base64.standard_b64encode(
                    s3io.get_bytes(uri)).decode()
                content.append({"text": f"file: {fname}"})
                content.append({"image": {"format": "jpeg",
                                          "source": {"bytes": data}}})
                n_frames += 1
    if not n_frames:
        return None
    emphasis = (doc.get("brief_amendments") or {}).get("emphasis")
    avoid = (doc.get("brief_amendments") or {}).get("avoid")
    if emphasis:
        content.append({"text": f"Editorial emphasis: {emphasis}"})
    if avoid:
        content.append({"text": f"Editorial avoid-list: {avoid}"})
    content.append({"text": "Rank now."})
    return {"record_id": "rerank-000", "system": system,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2000}


def lambda_handler(event: dict, context) -> dict:
    task_token = event["task_token"]
    plan = s3io.get_json(event["plan_s3uri"])
    doc = s3io.get_json(event["candidates_s3uri"])
    prefix = event["candidates_s3uri"].rsplit("/", 1)[0]

    def skip(reason: str) -> dict:
        logger.info("vision rerank skipped: %s", reason)
        return wf_common.complete_task_token(
            _sfn, task_token, {"skipped": reason})

    if not BEDROCK_GATEWAY_ARN:
        return skip("no_bedrock_gateway")
    system = (plan.get("prompts") or {}).get("artwork-rerank")
    if not system:
        return skip("prompts_not_bound")
    record = build_record(doc, system)
    if record is None:
        return skip("no_frames")

    records_s3uri = f"{prefix}/_rerank/records.jsonl"
    s3io.put_bytes(
        records_s3uri,
        (json.dumps(record, ensure_ascii=False) + "\n").encode(),
        "application/jsonl")
    return wf_common.submit_or_park(
        _lam, BEDROCK_GATEWAY_ARN, "bedrock-inference", "run_inference",
        {"records_s3uri": records_s3uri,
         "model_chain": plan["model_chain"],
         "use_batch": plan["use_batch"],
         "idempotency_token": f"artwork-rerank-{plan['uuid']}",
         "callback_task_token": task_token},
        skip)
