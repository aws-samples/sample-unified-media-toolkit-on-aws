"""
ReadOnscreenText — the witness held state (CA §8; the first draft's
Wait/Check poll loop was the anti-pattern and is gone, user review
2026-09-04): credits/chyrons/titles get read by frame-inspection while
this state is parked.

The handler plans the focus reads and submits ONE pooled request to the
witness gateway WITH callback_task_token = this state's own task token;
fi's waker completes the token with the terminal status AS DATA. The
submit folds this run's uuid in as fi's idempotency token, so a retried
or REDRIVEN invocation absorbs into the same witness run — and an absorb
into a CLOSED run is completed synchronously by fi's gateway: redrive IS
the reconcile. Once a submit response carries a job handle, the token is
FI'S to complete — this handler must only park (completing it here too
would double-complete an absorbed-closed run, review F3).

The cross-run ledger (lib/ledger.py) is the never-re-pay filter: spans
another bind of the SAME video already read are planned out, and when
that leaves nothing to read the token completes with the ledgered
product prefixes so binding still consumes the paid-for evidence.

Degrade-by-design: no FI_GATEWAY_ARN, an empty read plan, or a submit
failure all SELF-COMPLETE the token immediately — reads are an
enhancement, and every path reaches binding. A waitForTaskToken
handler's RETURN VALUE IS DISCARDED; the token payload is the state's
only output.
"""
from __future__ import annotations

import os

import boto3

from lib import ledger
from lib import pb_common
from lib import reads
from lib import s3io
from lib import sources
from lib import transcript_cues
from lib import vocatives

FI_GATEWAY_ARN = os.environ.get("FI_GATEWAY_ARN", "")
_lam = boto3.client("lambda")
_sfn = boto3.client("stepfunctions")


def _complete(task_token: str, payload: dict) -> dict:
    # pb_common owns the benign-race code list (ONE fact, three states)
    return pb_common.complete_task_token(_sfn, task_token, payload)


def _skip(task_token: str, uuid: str, plan_s3uri: str, reason: str) -> dict:
    """Nothing to wait for: complete our own token NOW with the skip as
    data. The state wakes instantly and the machine proceeds to binding
    on free evidence."""
    return _complete(task_token, {"uuid": uuid, "plan_s3uri": plan_s3uri,
                                  "reads": "skipped",
                                  "reads_reason": reason})


def lambda_handler(event: dict, context) -> dict:
    uuid = event["uuid"]
    plan_s3uri = event["plan_s3uri"]
    task_token = event["task_token"]
    if not FI_GATEWAY_ARN:
        return _skip(task_token, uuid, plan_s3uri, "no_fi_gateway")
    plan_flags = s3io.get_json(plan_s3uri)
    if not plan_flags.get("read_onscreen_text", True):
        return _skip(task_token, uuid, plan_s3uri,
                     "read_onscreen_text=false")
    # capability probe (review F9): parking on a token a witness build
    # WITHOUT callback support will never complete means sleeping into
    # the 26h deadman. A witness that can't call back is degraded to
    # free-evidence binding, honestly and immediately.
    try:
        desc = pb_common.invoke_gateway_tool(
            _lam, FI_GATEWAY_ARN, "frame-inspection", "describe_tool", {})
    except Exception as exc:                         # noqa: BLE001 — degrade
        return _skip(task_token, uuid, plan_s3uri,
                     f"fi_describe_failed: {str(exc)[:150]}")
    features = desc.get("features") or []
    if "callback_task_token" not in features:
        return _skip(task_token, uuid, plan_s3uri, "fi_no_callback")

    plan = plan_flags
    inputs = plan["inputs"]
    seg = inputs.get("shot_segmentation_s3uri")
    if not seg:
        return _skip(task_token, uuid, plan_s3uri, "no_shot_segmentation")

    transcript = sources.load_transcript(
        inputs.get("speech_diarization_s3uri"))
    faces = sources.load_faces(inputs.get("face_recognition_s3uri"))
    shots = sources.load_shots(seg)
    duration = max((b for _, b in shots.values()), default=0.0)
    text_shots = []
    credit_rolls = []
    screen_texts = []
    md = inputs.get("shot_metadata_s3uri")
    if md:
        docs = sources._get_jsonl(f"{md.rstrip('/')}/shot_metadata.jsonl")
        text_shots = sources.parse_text_signals(docs, shots)
        # SME-typed credit rolls: the credits evidence anchor (user
        # design 2026-09-30 — news carries none, so it never pays)
        credit_rolls = [shots[d["shot_id"]] for d in docs
                        if d.get("shot_kind") == "credit_roll"
                        and d.get("shot_id") in shots]
        screen_texts = sources.parse_screen_texts(docs, shots)
    cues = []
    if transcript:
        cues = transcript_cues.extract_cues(
            transcript["turns"], transcript.get("language") or "en")
        # deterministic vocative resolution joins the forecast cues —
        # the bind's mine_cues does exactly this, and the plan-time
        # class-1 forecast must see what the bind will see
        cues = cues + vocatives.resolve_vocatives(
            transcript["turns"], cues)

    # the cross-run ledger, keyed by the shot-seg wid (the per-video
    # handle): spans a previous bind of this video already read are
    # never re-paid
    seg_wid = (plan.get("sibling_workflow_ids") or {}).get(
        "shot_segmentation") or seg.rstrip("/").rsplit("/", 1)[-1]
    ledger_uri = ledger.uri_for(plan_s3uri, seg_wid)
    led_entries = ledger.load(ledger_uri)

    # last-resort gate (user direction + spend audit, 2026-09-30):
    # reads exist to name unnamed MAJOR heads; spans serving none are
    # never paid for. The plan-time class-1 FORECAST (the same pure
    # function the bind runs, same inputs INCLUDING the F0 veto)
    # subtracts every major voice the free bind will resolve to a
    # named face — an asset whose majors are all resolvable plans
    # nothing and the state self-completes. A forecast failure
    # degrades to the larger need set, never to under-reading.
    bound = set()
    try:
        from lib import class1
        f0 = sources.load_speaker_f0(
            inputs.get("speech_diarization_s3uri"))
        c1 = class1.match(transcript, faces, cues=cues, speaker_f0=f0)
        if c1:
            by_fid = {}
            for f in faces or []:
                for fid in [f["face_person_id"],
                            *(f.get("coalesced_ids") or [])]:
                    by_fid[fid] = f
            bound = {m["speaker_id"] for m in c1.get("matches") or []
                     if (by_fid.get(m.get("face_person_id")) or {}
                         ).get("name")}
    except Exception as exc:                     # noqa: BLE001 — degrade
        print(f"plan-time class1 forecast failed (need stays "
              f"unsubtracted): {exc}")
    # voices whose proposed name SME's on-screen text already
    # corroborates will be named by the free bind — no witness need
    bound |= reads.screen_text_confirmed_speakers(
        transcript, cues, screen_texts)
    need = reads.need_windows_from_sources(faces, transcript,
                                           bound_speakers=bound)
    last_speech = max((t["end"] for t in (transcript or {}).get(
        "turns") or []), default=None)
    read_plan = reads.plan_reads(cues, faces, text_shots, duration,
                                 ledger=ledger.as_spans(led_entries),
                                 has_shot_metadata=bool(md),
                                 need_windows=need,
                                 credit_roll_spans=credit_rolls,
                                 last_speech_end=last_speech)
    if not read_plan["requests"]:
        if read_plan["skipped"]["ledgered"] and led_entries:
            # everything needed was already read by an earlier bind of
            # this video — hand binding the recorded product prefixes
            return _complete(task_token, {
                "uuid": uuid, "plan_s3uri": plan_s3uri,
                "reads": "ledgered",
                "read_prefixes": ledger.prefixes(led_entries)})
        return _skip(task_token, uuid, plan_s3uri, "nothing_to_read")

    if "span_crop" not in features:
        # an older witness accepts-and-ignores the crop key and reads
        # full frames while we ledger the span as cropped — strip the
        # key so the plan, the ledger and reality agree (review,
        # 2026-09-05: same accept-and-ignore class as the callback probe)
        for req in read_plan["requests"]:
            for sp in req.get("spans") or []:
                sp.pop("crop", None)
        for cand in read_plan["ledger_candidates"]:
            cand.pop("crop", None)

    asset = {"shot_segmentation_s3uri": seg,
             "requests": read_plan["requests"]}
    if md:
        asset["shot_metadata_s3uri"] = md
    try:
        sub = pb_common.invoke_gateway_tool(
            _lam, FI_GATEWAY_ARN, "frame-inspection", "inspect_frames", {
                "assets": [asset],
                # normalized by the gateway and written into the plan
                # by ValidateSources (default TRUE — batch pricing;
                # pb_common.DEFAULT_FLAGS is the one home). A missing
                # key is a bug, not a choice: KeyError over a silent
                # re-default that flips the run's economics.
                "use_batch": bool(plan["use_batch"]),
                "idempotency_token": f"pb-{uuid}",
                "callback_task_token": task_token})
    except Exception as exc:                         # noqa: BLE001 — degrade
        return _skip(task_token, uuid, plan_s3uri,
                     f"submit_failed: {str(exc)[:150]}")
    # ONE discriminator, checked first: a job handle means fi ACCEPTED
    # the token (registered it, or already completed it synchronously
    # for an absorbed-closed run) — park, never complete it ourselves.
    # An absorbed-FAILED response carries error text alongside its
    # handle; treating that as a refusal double-completed the token
    # (review F3).
    if not (isinstance(sub, dict) and sub.get("job_id")):
        msg = sub.get("message") if isinstance(sub, dict) else str(sub)
        return _skip(task_token, uuid, plan_s3uri,
                     f"submit_refused: {msg}")

    # observability + the ledger candidates BindPersons will append
    # once the products are actually fetched
    s3io.put_json(plan_s3uri.replace("plan.json", "reads.json"),
                  {"job_id": sub["job_id"],
                   "output_s3_prefix": sub["output_s3_prefix"],
                   "ledger_uri": ledger_uri,
                   "ledger_candidates": read_plan["ledger_candidates"],
                   "spans": len(read_plan["spans"]),
                   "credits": read_plan["credits"]})
    # PARK: fi's waker (or its absorbed-closed synchronous path) completes
    # the token with the terminal status as data.
    return {"parked": True, "reads_job_id": sub["job_id"]}
