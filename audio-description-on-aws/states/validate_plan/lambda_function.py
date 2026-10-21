"""
StateValidateAndPlan: refuse-before-spend. Every sibling pointer
resolved, all five registry prompts bound (a category missing at prod
stage refuses the run, never falls back), the voice pinned, and the
config validated — then ONE _plan/plan.json every later state reads.
"""
from __future__ import annotations

import logging

from lib import ad_common as C

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AudioDescriptionValidationError(Exception):
    """Named refusal; the ASL surfaces the message verbatim."""


REQUIRED = C.REQUIRED_INPUT_URIS
OPTIONAL = C.OPTIONAL_INPUT_URIS

# refuse-by-name probes, ONE table for required and optional alike,
# each naming an artifact this workflow needs: a typo'd OPTIONAL
# pointer silently degrading to None at fetch time is the same
# silent-loss class as a typo'd required one (review, 2026-09-07 —
# the optionals were never probed). Supplying an optional pointer is a
# REQUEST for its evidence, so audiostem is probed on the artifact AD
# consumes even though its producer tolerates a failed analysis on a
# completed run (user decision, 2026-09-08: a run that cannot deliver
# music gating is refused, not silently degraded). Two deliberate
# exceptions:
#  - decisions_s3uri is UNPROBED: an absent decisions.json is the
#    designed "nothing accepted yet" state of a real review, not a
#    caller error — refusing it would break standing re-run params.
#  - credits.json is unprobed on the cast pointer: a CONDITIONAL rider
#    (present only when that run performed a credits read); the
#    pointer's requested product is persons.json, read below — the
#    usage_tracking probe keeps pre-Bind pointers out, which NARROWS
#    (not closes) the in-flight window; callers poll to SUCCEEDED.
PROBES = {
    "shot_segmentation_s3uri": ("embeddings/shots.json",),
    "shot_metadata_s3uri": ("shot_metadata.jsonl",),
    "speech_diarization_s3uri": ("transcripts/final_transcript.json",),
    "conversation_analysis_s3uri": ("conversation_analysis.json",),
    "face_recognition_s3uri": ("facemetadata.json",),
    "audiostem_s3uri": ("stem_activity.json",),
    "cast_resolution_s3uri": ("usage_tracking.json",),
}


def lambda_handler(event: dict, context) -> dict:
    workflow_id = event["uuid"]
    request = event.get("input") or {}

    inputs = {}
    for name in REQUIRED:
        uri = (request.get(name) or "").strip()
        if not uri:
            raise AudioDescriptionValidationError(f"{name} is required")
        C.parse_s3_uri(uri)
        inputs[name] = uri.rstrip("/")
    for name in OPTIONAL:
        uri = (request.get(name) or "").strip()
        if uri:
            C.parse_s3_uri(uri)
            inputs[name] = uri.rstrip("/")
    rule = C.people_source_error(inputs)
    if rule:
        raise AudioDescriptionValidationError(rule)

    # a full-artifact spelling of the cast pointer (any of its contract
    # basenames — get_job_status hands out presignable artifact uris)
    # would break the probes and the GET; normalize to the run root
    # here, once, by SEGMENT (str.endswith on names is the banned
    # suffix idiom — artifact_key's docstring)
    cr = inputs.get("cast_resolution_s3uri")
    if cr:
        segs = cr.split("/")
        if segs[-1] in ("persons.json", "speaker_map.json",
                        "face_map.json", "credits.json",
                        "usage_tracking.json"):
            inputs["cast_resolution_s3uri"] = "/".join(segs[:-1])

    voice_id = (request.get("voice") or "").strip()
    if not voice_id:
        raise AudioDescriptionValidationError(
            "voice is required (a text-to-speech voice_id; "
            "list_voice_collection shows the catalog)")

    policy = (request.get("sung_vocal_policy")
              or C.SUNG_VOCAL_DEFAULT).strip()
    if policy not in C.SUNG_VOCAL_POLICIES:
        raise AudioDescriptionValidationError(
            f"sung_vocal_policy must be one of {C.SUNG_VOCAL_POLICIES}")

    # (no mix parameter: this workflow renders no audio — the gateway
    # refuses `mix` at submit and points at the media-processing ad_mix
    # chain, so no execution is ever started for a doomed request)

    # refuse-before-spend means the ARTIFACTS, not just the uri shapes:
    # a typo'd pointer must be refused by NAME here, never surface as a
    # NoSuchKey mid-run (review 2026-08-31 — the ASL comment promised
    # this and nothing implemented it). HEAD probes for every supplied
    # pointer; the one GET below (cast persons.json) is content the
    # provenance check and the plan payload both need.
    from lib import evidence
    for name, probe_list in PROBES.items():
        uri = inputs.get(name)
        if not uri:
            continue                    # absent optional, three-state
        b, k = C.parse_s3_uri(uri)
        for path in probe_list:
            key = evidence.artifact_key(k, *path.split("/"))
            if not C.exists(b, key):
                remedy = ("; if the evidence does not exist, omit the "
                          "optional pointer instead"
                          if name in OPTIONAL else "")
                raise AudioDescriptionValidationError(
                    f"{name}: s3://{b}/{key} not found — the pointer "
                    f"must name a run that produced this artifact"
                    f"{remedy}")

    # ── cast-resolution: the one validate-time GET, doubling as the
    # persons.json existence check (a HEAD first would just add a
    # race). Content is needed twice — the provenance pin and the
    # filtered name-map payload, which rides plan.json so no later
    # state re-downloads or re-filters the 600-person doc. ──
    cast_bindings = None
    if inputs.get("cast_resolution_s3uri"):
        cr_b, cr_k = C.parse_s3_uri(inputs["cast_resolution_s3uri"])
        key = evidence.artifact_key(cr_k, "persons.json")
        try:
            persons_doc = C.read_json_if_exists(cr_b, key)
        except ValueError as exc:       # json.JSONDecodeError subclass
            raise AudioDescriptionValidationError(
                "cast_resolution_s3uri persons.json is not readable "
                f"JSON: {str(exc)[:120]}")
        if not isinstance(persons_doc, dict):
            raise AudioDescriptionValidationError(
                f"cast_resolution_s3uri: s3://{cr_b}/{key} "
                + ("not found — the pointer must name a run that "
                   "produced it" if persons_doc is None else
                   "is not a persons document"))
        problem = evidence.cast_provenance_error(inputs, persons_doc)
        if problem:
            raise AudioDescriptionValidationError(problem)
        cast_bindings = evidence.filter_cast_bindings(
            persons_doc, C.CAST_BINDINGS_CAP)
        if "face_recognition_s3uri" not in inputs:
            persons = persons_doc.get("persons") or []
            # the PRODUCER declares presence: every >=2026-09-08 person
            # carries the appearances KEY (face- or speech-basis, its
            # appearances_basis says which — faceless runs degrade to
            # speaker turn spans producer-side). Key absence, not value
            # emptiness, is what identifies a pre-export run: a real
            # run may legitimately have empty timelines (nothing
            # detected), and inferring facelessness from emptiness
            # falsely refused it (review, 2026-09-08).
            if persons and not any("appearances" in p for p in persons):
                raise AudioDescriptionValidationError(
                    "cast_resolution_s3uri is the only people source "
                    "but its persons predate the appearance-timeline "
                    "export (2026-09-08 producer) — re-run the cast "
                    "run, or supply face_recognition_s3uri")

    # ── tts capability: the synth/repair loops sleep on
    # callback_task_token, and a pre-callback tts deployment IGNORES the
    # field — the wait state would sleep to its 26h deadman. Probe the
    # capability itself here, before any bedrock spend (review
    # 2026-08-31; tts_submit keeps its per-submit handle check too). ──
    from lib import submit as submit_mod
    try:
        submit_mod.assert_tts_push_completion()
    except RuntimeError as exc:
        raise AudioDescriptionValidationError(str(exc))

    # ── bind the five prompt categories NOW: a missing prompt must
    # refuse before any model spend, not fail the writer mid-run ──
    prompts = {}
    for category in C.PROMPT_CATEGORIES:
        resp = C.invoke_gateway_tool(
            C.PROMPT_REGISTRY_ARN, "prompt-registry___get_prompt",
            {"category": category})
        if resp.get("error"):
            raise AudioDescriptionValidationError(
                f"prompt category {category!r} unavailable "
                f"({resp.get('errorCode')}): {resp.get('message')}")
        prompts[category] = resp["artifact"] if "artifact" in resp else resp

    prefix = C.run_prefix(workflow_id)
    plan = {
        "workflow_id": workflow_id,
        "tool_version": C.TOOL_VERSION,
        "output_prefix": prefix,
        "inputs": inputs,
        "voice_id": voice_id,
        "language": (request.get("language") or "").strip() or None,
        "sung_vocal_policy": policy,
        "model_chain": C.DEFAULT_MODEL_CHAIN,
        "prompts": prompts,
        "scope": request.get("history_scope") or workflow_id,
        "title": (request.get("title") or "").strip() or workflow_id,
    }
    if cast_bindings is not None:
        plan["cast_bindings"] = cast_bindings
    C.write_json(C.PROXY_BUCKET, f"{prefix}/_plan/plan.json", plan)
    logger.info("plan written: %s inputs, policy=%s", len(inputs), policy)
    return {"uuid": workflow_id,
            "plan_s3uri": f"s3://{C.PROXY_BUCKET}/{prefix}/_plan/plan.json"}
