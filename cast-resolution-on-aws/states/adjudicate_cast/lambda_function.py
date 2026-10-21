"""
AdjudicateCast — enrichment round 2 (P2-a + tooth-2), ONE held state.

Post-review architecture: BindPersons is the ONLY pipeline — this state
never re-binds. It reads the SHIPPED persons.json (which already
carries the round-1 resolved addressees as ordinary evidence, every
witness cue, and the provenance pin), builds cast-adjudication dossiers
and tooth-2 critique records from it, and submits ONE pooled inference
run — the two families are independent of each other, so they share the
submission (the batching rule: dependent rounds get states, independent
records get pooled). Callback token; ApplyEnrichment consumes.

Nothing to ask self-completes the token with adjudication=skipped, and
the context artifacts are written ONLY when a submission happens — a
skipped run leaves no misleading intermediates.
"""
from __future__ import annotations

import os

import boto3

from lib import adjudicate
from lib import critic2
from lib import pb_common
from lib import reads
from lib import s3io
from lib import sources

BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")
_lam = boto3.client("lambda")
_sfn = boto3.client("stepfunctions")


def lambda_handler(event: dict, context) -> dict:
    plan_s3uri = event["plan_s3uri"]
    task_token = event["task_token"]

    def skip(reason: str) -> dict:
        return pb_common.complete_task_token(
            _sfn, task_token,
            {"adjudication": "skipped", "reason": reason})

    plan = s3io.get_json(plan_s3uri)
    uuid = plan["uuid"]
    prefix = plan_s3uri.rsplit("/_plan/", 1)[0]
    if not plan.get("llm_enrichment", True):
        return skip("llm_enrichment=false")
    if not BEDROCK_GATEWAY_ARN:
        return skip("no_bedrock_gateway")
    prompts = plan.get("prompts") or {}
    adj_system = prompts.get("cast-adjudicate")
    critic_system = prompts.get("cast-critic2")
    if not (adj_system and critic_system):
        # unbound prompts: pre-migration plan or a degraded bind —
        # skip BEFORE the S3/transcript loads below
        return skip("prompts_not_bound")

    doc = s3io.get_json(f"{prefix}/persons.json")
    credits_doc = None
    # root first, then the pre-promotion path (transition fallback: a
    # run whose Bind executed pre-2026-09-08 wrote _enrich/ — dropping
    # it would silently lose the paid cast universe on a deploy-crossing
    # or redriven run; remove once no such run can be redriven, 90-day
    # execution retention)
    for key in (f"{prefix}/credits.json", f"{prefix}/_enrich/credits.json"):
        try:
            credits_doc = s3io.get_json(key)
            break
        except Exception:                        # noqa: BLE001 — absent
            continue
    cast = reads.cast_from_credits(credits_doc) if credits_doc else {}
    transcript = sources.load_transcript(
        plan["inputs"].get("speech_diarization_s3uri"))
    turns = (transcript or {}).get("turns")

    dossiers = adjudicate.build_dossiers(doc, cast, turns=turns)
    critique = critic2.build_dossiers(doc, turns=turns)
    records = (adjudicate.build_records(dossiers, cast, adj_system)
               if dossiers else [])
    records += (critic2.build_records(critique, critic_system)
                if critique else [])
    if not records:
        return skip("nothing_to_adjudicate")

    # context only when a submission exists — ApplyEnrichment's parse
    # needs the dossiers (agreement firewall) and the cast universe
    s3io.put_json(f"{prefix}/_enrich/context.json", {
        "dossiers": dossiers, "cast": cast,
        "counts": {"adjudication": len(dossiers),
                   "critique": len(critique)}})
    return pb_common.submit_or_park(
        _lam, BEDROCK_GATEWAY_ARN, "bedrock-inference", "run_inference",
        {"records": records,
         "use_batch": False,
         "idempotency_token": f"pb-adj-{uuid}",
         "callback_task_token": task_token},
        skip)
