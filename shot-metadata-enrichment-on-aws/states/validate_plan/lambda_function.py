"""
StateValidateAndPlan (README.md §10): everything that can refuse, refuses
HERE — before any inference dollar. Fetches shots.json from the
shot-segmentation run, the optional facemetadata.json and diarization
transcript, binds the registry `default` prompt (checking its declared
tokens against THIS workflow's values catalog), validates the model chain
as IMAGE-capable against bedrock's registry, plans the image-bounded
windows (§3) and their Map batches, derives the history scope, and
persists everything to `_plan/plan.json` — states pass URIs, never bodies
(256 KB state I/O).
"""
from __future__ import annotations

import hashlib
import logging
import re

from lib import faces as faces_mod
from lib import sme_common
from lib import windowing
from lib import workflow_history
from lib.custom_exceptions import ShotMetadataValidationError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_SHOTSEG_RUN_RE = re.compile(
    r"micro-workflow/shot-segmentation/([A-Za-z0-9-]+)/")

# The workflow-provided values catalog (conversation-analysis §1 pattern):
# a declared token outside this set must fail HERE, loudly, naming it —
# never at render inside a paid submit.
VALUES_CATALOG = {"{{program_names}}", "{{overall_content_type}}",
                  "{{shot_list}}", "{{face_names}}", "{{face_images}}",
                  "{{transcript}}", "{{frame_images}}"}


def lambda_handler(event: dict, context) -> dict:
    workflow_id = event["uuid"]
    request = event.get("input") or {}

    shots_uri = (request.get("shot_segmentation_s3uri") or "").strip()
    if not shots_uri:
        raise ShotMetadataValidationError(
            "shot_segmentation_s3uri is required — a shot-segmentation "
            "run's shots.json (or its run prefix) on the proxy bucket")
    shots_uri = _resolve_shots_uri(shots_uri)
    bucket, shots_key = sme_common.parse_s3_uri(shots_uri)
    if bucket != sme_common.PROXY_BUCKET:
        raise ShotMetadataValidationError(
            f"shot_segmentation_s3uri must be on the proxy bucket "
            f"({sme_common.PROXY_BUCKET}) — got bucket {bucket!r}")

    doc = sme_common.read_json(bucket, shots_key)
    shots = _validate_shots(doc)

    face_uri, people = _load_faces(request)
    shot_names = faces_mod.shots_with_faces(shots, people) if people else {}

    transcript_uri = _resolve_transcript_uri(request)

    prompt = _fetch_prompt((request.get("prompt_version") or "").strip())
    model_chain = _validate_model_chain(
        request.get("model_chain") or list(sme_common.DEFAULT_MODEL_CHAIN))

    # Burst folding + recurrence hints (both tolerant of runs predating the
    # evidence fields — absent means empty, never an error).
    run_of, frameless = windowing.burst_fold(
        doc.get("burst_runs") or [], sme_common.BURST_REPRESENTATIVES)
    recurring = _load_recurrence(bucket, shots_key)

    windows = windowing.plan_windows(
        shots,
        shot_target=sme_common.WINDOW_SHOT_TARGET,
        image_budget=sme_common.WINDOW_IMAGE_BUDGET,
        tier=sme_common.FRAME_TIER,
        gallery_shots=set(shot_names),
        frameless=frameless)
    batches = [
        [w["index"] for w in windows[i:i + sme_common.WINDOWS_PER_BATCH]]
        for i in range(0, len(windows), sme_common.WINDOWS_PER_BATCH)]

    scope = _derive_scope(request.get("history_scope"), shots_uri)
    use_batch = request.get("use_batch")
    if use_batch is None:
        # DEFAULT TRUE (user decision 2026-08-19) — image-heavy volume,
        # nothing interactive downstream; sync is the explicit opt-out.
        use_batch = sme_common.DEFAULT_USE_BATCH

    prefix = sme_common.run_prefix(workflow_id)
    plan = {
        "workflow_id": workflow_id,
        "tool_version": sme_common.TOOL_VERSION,
        "shots": {
            "s3uri": shots_uri,
            "run_prefix": shots_uri.rsplit("/", 2)[0],
            "count": len(shots),
            "duration_sec": doc.get("actual_duration_sec"),
        },
        "faces": {
            "s3uri": face_uri,
            "people": people,           # named only, spans + rep_face_key
            "shot_names": {str(k): v for k, v in shot_names.items()},
        },
        "transcript": {"s3uri": transcript_uri},
        # Shot-segmentation's structural evidence, threaded to the renderer
        # as compact per-shot annotations (keys are strings — plan.json is
        # JSON round-tripped):
        #   burst:     shot_id -> run index; members beyond the per-run
        #              representatives carry no frames (see windowing)
        #   recurring: shot_id -> one terse string for the shot_list
        "burst": {str(k): v for k, v in run_of.items()},
        "recurring": recurring,
        "program_names": str(request.get("program_names")
                             or "(not supplied)"),
        "overall_content_type": str(request.get("overall_content_type")
                                    or "(not supplied)"),
        "prompt": {
            "category": sme_common.PROMPT_CATEGORY,
            "version": prompt["version"],
            "source_md5": prompt["source_md5"],
            "stage": prompt["stage"],
            "system": prompt["system"],
            "user_template": prompt["user_template"],
            "variables": prompt["variables"],
        },
        "model_chain": model_chain,
        "use_batch": bool(use_batch),
        "scope": scope,
        "shot_target": sme_common.WINDOW_SHOT_TARGET,
        "image_budget": sme_common.WINDOW_IMAGE_BUDGET,
        "windows": windows,
        "batches": batches,
        "output_prefix": prefix,
    }
    plan_key = f"{prefix}/_plan/plan.json"
    sme_common.write_json(sme_common.PROXY_BUCKET, plan_key, plan)
    logger.info(
        "plan: %d shots -> %d window(s) in %d batch(es), %d named "
        "face(s), batch=%s, scope=%s", len(shots), len(windows),
        len(batches), len(people), plan["use_batch"], scope)

    return {
        "uuid": workflow_id,
        "plan_s3uri": f"s3://{sme_common.PROXY_BUCKET}/{plan_key}",
        "window_count": len(windows),
        # The Map state's items — indices only, each iteration reads the
        # plan document (state payloads carry URIs, never bodies).
        "batches": [{"batch_index": i, "window_indices": b}
                    for i, b in enumerate(batches)],
    }


def _load_recurrence(bucket: str, shots_key: str) -> dict:
    """{shot_id(str): terse annotation} from the run's own recurrence.json.

    Recurring shots are programme-level structure the model cannot see from
    one window: an ident/title card re-airing (hash_confirmed true), a
    recurring set position, an act-boundary card flanked by black. One
    compact string per member shot keeps the shot_list slim:

        "recurs 8x across programme, hash_confirmed, act_boundary"

    Tolerant by design: recurrence.json absent (a run predating it, or a
    caller pointing at a bare shots.json copy) means {} — never an error.
    """
    key = shots_key.rsplit("/", 1)[0] + "/recurrence.json"
    try:
        rec = sme_common.read_json(bucket, key)
    except Exception:
        return {}
    out: dict = {}
    for g in rec.get("recurrence_groups") or []:
        parts = [f"recurs {g.get('recurrence_count')}x across programme"]
        hc = g.get("hash_confirmed")
        if hc is True:
            parts.append("hash_confirmed")   # the same rendered image
        elif hc is False:
            parts.append("similar_but_not_identical_render")
        for s in g.get("shots") or []:
            note = parts + ([s["position"]]
                            if s.get("position") not in (None, "interior")
                            else [])
            out[str(s["shot_id"])] = ", ".join(note)
    return out


def _resolve_shots_uri(uri: str) -> str:
    """Accept the run prefix or the embeddings/shots.json object itself —
    the run prefix is what shot-segmentation's outputs[] hands an agent."""
    if uri.endswith("shots.json"):
        return uri
    return uri.rstrip("/") + "/embeddings/shots.json"


def _validate_shots(doc: dict) -> list:
    shots = doc.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ShotMetadataValidationError(
            "shots.json carries no shots[] — is this a completed "
            "shot-segmentation run?")
    for field in ("shot_id", "start_time", "end_time", "frames_key"):
        missing = [s.get("shot_id") for s in shots if field not in s]
        if missing:
            raise ShotMetadataValidationError(
                f"shots.json entries missing {field!r} (e.g. shot "
                f"{missing[:3]}) — the schema drifted; refusing to plan")
    return sorted(shots, key=lambda s: int(s["shot_id"]))


def _load_faces(request: dict):
    """Optional face-recognition join. Absent → text sentinels downstream
    (optional means optional, family style). Present → NAMED people with
    rep_face_key pointers; a facemetadata predating face-recognition's
    rep_face_key change is refused by name, not silently gallery-less."""
    uri = (request.get("face_recognition_s3uri") or "").strip()
    if not uri:
        return None, []
    if not uri.endswith("facemetadata.json"):
        uri = uri.rstrip("/") + "/facemetadata.json"
    bucket, key = sme_common.parse_s3_uri(uri)
    if bucket != sme_common.PROXY_BUCKET:
        raise ShotMetadataValidationError(
            f"face_recognition_s3uri must be on the proxy bucket — got "
            f"{bucket!r}")
    people = faces_mod.named_people(sme_common.read_json(bucket, key))
    lacking = [p["name"] for p in people if not p.get("rep_face_key")]
    if lacking:
        raise ShotMetadataValidationError(
            f"facemetadata.json people lack rep_face_key (e.g. "
            f"{lacking[:3]}) — this run predates face-recognition's "
            "representative-face output (its PLAN §22); re-run "
            "face-recognition or omit face_recognition_s3uri")
    return uri, people


def _resolve_transcript_uri(request: dict):
    uri = (request.get("transcript_s3uri") or "").strip()
    if not uri:
        return None
    bucket, _ = sme_common.parse_s3_uri(uri)
    if bucket != sme_common.PROXY_BUCKET:
        raise ShotMetadataValidationError(
            f"transcript_s3uri must be on the proxy bucket — got "
            f"{bucket!r}")
    return uri


def _fetch_prompt(version: str = "") -> dict:
    """Bind the registry prompt: the category default, or the caller's
    NAMED version (prompt_version — e.g. "default_news"). A named version
    that does not exist refuses HERE, before any spend — the registry never
    falls back silently."""
    params = {"category": sme_common.PROMPT_CATEGORY}
    if version:
        params["version"] = version
    artifact = sme_common.invoke_gateway_tool(
        sme_common.PROMPT_REGISTRY_ARN, "prompt-registry___get_prompt",
        params)
    if artifact.get("error"):
        raise ShotMetadataValidationError(
            f"could not fetch the {sme_common.PROMPT_CATEGORY} prompt"
            f"{' version ' + version if version else ''}: "
            f"{artifact.get('message')}")
    declared = {v["token"] for v in artifact["variables"]}
    unsupplied = declared - VALUES_CATALOG
    if unsupplied:
        raise ShotMetadataValidationError(
            f"the {artifact['version']} prompt declares variables this "
            f"workflow cannot supply: {sorted(unsupplied)} — the catalog "
            f"provides {sorted(VALUES_CATALOG)}")
    return artifact


def _validate_model_chain(chain) -> list:
    """Caller-driven, validated here — and every model must be
    IMAGE-capable: this workflow renders image-bearing records, and a
    text-only fallback would receive bytes it must reject (the chain's
    constraints INTERSECT — GOTCHAS)."""
    if isinstance(chain, str):
        chain = [part.strip() for part in chain.split(",") if part.strip()]
    if not isinstance(chain, list) or not chain:
        raise ShotMetadataValidationError(
            "model_chain must be a list of Bedrock model ids (or a "
            "comma-separated string), primary first")
    for model_id in chain:
        entry = sme_common.invoke_gateway_tool(
            sme_common.BEDROCK_GATEWAY_ARN, "bedrock-inference___get_model",
            {"model_id": model_id})
        if entry.get("error"):
            raise ShotMetadataValidationError(
                f"model {model_id!r} is not in this deployment's registry "
                "— call bedrock-inference list_models for the legal ids")
        model = entry.get("model") or entry
        types = model.get("model_type") or []
        # bedrock's registry spells the capability "vision" (measured on
        # the live registry, first ladder rung 2026-08-19 — "image" was a
        # guess that refused sonnet-4-6 itself).
        if types and not ({"vision", "image"} & set(types)):
            raise ShotMetadataValidationError(
                f"model {model_id!r} is not a vision-capable model "
                f"(model_type={types}) — this workflow renders labeled "
                "frame images into every record")
    return chain


def _derive_scope(explicit, shots_uri: str) -> str:
    """The caller's own asset binding wins; else the upstream
    shot-segmentation workflow_id; else a deterministic hash."""
    if explicit:
        scope = str(explicit).strip()
        if not workflow_history._SCOPE_RE.match(scope):
            raise ShotMetadataValidationError(
                f"history_scope {scope!r} must match "
                f"{workflow_history._SCOPE_RE.pattern}")
        return scope
    m = _SHOTSEG_RUN_RE.search(shots_uri)
    if m:
        return f"shots.{m.group(1)}"
    return "shots." + hashlib.sha1(
        shots_uri.encode("utf-8"),
        usedforsecurity=False).hexdigest()[:12]
