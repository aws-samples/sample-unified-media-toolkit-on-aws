"""
ResolveAddressees — enrichment round 1 (P2-b), ONE held state, and the
FIRST evidence state: it needs only the transcript, so it runs before
the witness reads and BindPersons consumes its output directly — one
bind, no re-bind (the post-review architecture: BindPersons is the only
pipeline; everything after it AMENDS).

Sends the UNRESOLVED vocative sites (adjacency voting runs first and
its names are never re-asked) to bedrock-inference with THIS state's
task token; its waker completes the token with the terminal status AS
DATA. Skip paths self-complete the token instantly: llm_enrichment
false, no transcript, no cues (language gate), nothing unresolved.

The windows are persisted (_enrich/windows.json) because BindPersons
parses the answers against them — a waitForTaskToken handler's return
value is discarded.

Enrichment rounds always run SYNC regardless of use_batch: the records
are tiny text (delta < $0.01), and use_batch prices the witness frame
reads where the real money is.
"""
from __future__ import annotations

import os

import boto3

from lib import pb_common
from lib import s3io
from lib import sources
from lib import transcript_cues
from lib import vocatives
from lib import vocatives_llm

BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")
_lam = boto3.client("lambda")
_sfn = boto3.client("stepfunctions")


def lambda_handler(event: dict, context) -> dict:
    plan_s3uri = event["plan_s3uri"]
    task_token = event["task_token"]

    def skip(reason: str) -> dict:
        return pb_common.complete_task_token(
            _sfn, task_token, {"addressees": "skipped", "reason": reason})

    plan = s3io.get_json(plan_s3uri)
    uuid = plan["uuid"]
    if not plan.get("llm_enrichment", True):
        return skip("llm_enrichment=false")
    if not BEDROCK_GATEWAY_ARN:
        return skip("no_bedrock_gateway")
    voc_system = (plan.get("prompts") or {}).get("cast-vocatives")
    if not voc_system:
        # unbound prompts: a pre-migration plan, or ValidateSources
        # degraded the bind (registry unavailable) — skip with the
        # reason, never KeyError
        return skip("prompts_not_bound")

    transcript = sources.load_transcript(
        plan["inputs"].get("speech_diarization_s3uri"))
    if not transcript:
        return skip("no_transcript")
    cues = transcript_cues.extract_cues(
        transcript["turns"], transcript.get("language") or "en")
    if not cues:
        return skip("no_cues")
    resolved = vocatives.resolve_vocatives(transcript["turns"], cues)
    # the LLM is asked only about candidates the structural screens
    # accept as NAMES — a rejected interjection/word is a settled
    # non-answer, not an unresolved question (user principle,
    # 2026-09-24: tokens chase the unconfirmed; measured, the screens
    # cut 48% of windows corpus-wide)
    screen = vocatives.name_screen(transcript["turns"])
    cues = [c for c in cues
            if c["kind"] != "vocative" or screen(c["name"])]
    windows = vocatives_llm.build_windows(transcript["turns"], cues,
                                          resolved)
    if not windows:
        return skip("nothing_unresolved")

    prefix = plan_s3uri.rsplit("/_plan/", 1)[0]
    s3io.put_json(f"{prefix}/_enrich/windows.json", windows)
    return pb_common.submit_or_park(
        _lam, BEDROCK_GATEWAY_ARN, "bedrock-inference", "run_inference",
        {"records": vocatives_llm.build_records(windows, voc_system),
         "use_batch": False,
         "idempotency_token": f"pb-voc-{uuid}",
         "callback_task_token": task_token},
        skip)
