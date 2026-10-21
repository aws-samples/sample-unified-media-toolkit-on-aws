"""
StateExtractProcessedFrames — normalizes the media-processing result to
the field shape every downstream state already reads (2026-09-18, the
family sibling pattern).

Accepts the GATEWAY handle (has output_s3_prefix) — MP's waker
completed the held submit state with the submit handle + terminal
status AS DATA; the FAILED verdict is enforced HERE (MP is a hard
dependency: no frames, nothing to recognize). The manifest and probe
keys are derived from MP's output contract
({prefix}/face_frames/frame_manifest.json and {prefix}/probe.json);
duration_sec is read from probe.json. The legacy nested-execution
shape was retired 2026-09-20; a non-handle shape raises.

Downstream states keep reading media_processing_output as before —
this is the ONLY place that knows about the handle.
"""
from __future__ import annotations

import json
import logging

from lib.exceptions import NoVideoStream
from lib.s3_utils import get_json

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _normalize_verdict(mp_out: dict) -> tuple:
    """(code, inner_message) for a FAILED handle, across BOTH completion
    shapes. The waker forwards error = the raising class name and cause =
    a Lambda error JSON (errorMessage is the sentence that matters,
    truncated). DEPLOY SKEW is why BOTH shapes are read: MP deployments
    at or after 2026-10-15 emit the one waker shape on every completion
    path, but an OLDER MP's absorbed-into-finished synchronous completion
    puts the whole cause JSON into `error` with no `cause` key — so a
    field that parses as that JSON yields errorType/errorMessage,
    whichever field it arrived in. Do not delete the dual parse until no
    pre-2026-10-15 MP deployment can call back. Older waker deploys carry
    neither field; then both halves are empty and the caller falls back
    honestly."""
    code = str(mp_out.get("error") or "")
    inner = ""
    parsed_a_dict = False
    for raw in (mp_out.get("cause"), code):
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            # errorMessage may be absent — the waker's compact cause keeps
            # whichever of errorType/errorMessage the execution carried,
            # and an exception raised with empty args has only the type
            parsed_a_dict = True
            inner = str(parsed.get("errorMessage") or "")
            if parsed.get("errorType"):
                code = str(parsed["errorType"])
            break
    if not inner and not parsed_a_dict:
        # only an UNPARSEABLE cause rides along raw (truncated); a parsed
        # dict with no errorMessage already gave everything it had in code
        inner = str(mp_out.get("cause") or "")[:300]
    return code, inner


def lambda_handler(event: dict, context) -> dict:
    data = event.get("data", {})
    mp_out = data.get("media_processing_output")
    if not mp_out:
        raise ValueError("data.media_processing_output is missing")

    if "output_s3_prefix" not in mp_out:
        # the legacy nested-execution shape retired 2026-09-20 — every
        # run now parks on MP's gateway and is completed with the handle
        raise ValueError(
            f"data.media_processing_output is not a gateway handle "
            f"(got: {mp_out})")
    # the FAILED verdict is enforced BY NAME: a source with no video
    # stream raises NoVideoStream (the stable code agents branch on),
    # everything else re-raises with MP's own error/cause passed through.
    status = mp_out.get("status")
    if status != "SUCCEEDED":
        code, inner = _normalize_verdict(mp_out)
        # this consumer's MP request is ALWAYS frames-only (the jpeg
        # face_frames group), so NoCompatibleOutputGroups can only ever
        # mean "no video stream" here — the class alone is decisive,
        # even when no message survived the completion path. The wording
        # guard remains only for the legacy ValueError shape (MP deploys
        # older than the exception rename).
        no_video = (code == "NoCompatibleOutputGroups"
                    or (code == "ValueError"
                        and "No output_groups are compatible" in inner
                        and "no video" in inner))
        if no_video:
            # a terminal content fact with a stable code: the agent
            # probes first next time and skips the visual chain —
            # failing HERE (before any Rekognition spend) is the
            # family's refuse-before-spend rule (2026-10-15, the
            # speech-diarization decision applied family-wide)
            raise NoVideoStream(
                "the source has no video stream — no frames to recognize"
                + (f" ({inner})" if inner else ""))
        detail = ": ".join(x for x in (code, inner) if x) \
            or "the waker carried no error detail — see the MP run"
        raise RuntimeError(
            f"media-processing run {mp_out.get('workflow_id')} "
            f"ended {status}: {detail}")
    prefix = mp_out["output_s3_prefix"]
    frames_bucket, _, key_prefix = prefix[5:].partition("/")
    # deterministic per MP's output contract
    manifest_key = f"{key_prefix}/face_frames/frame_manifest.json"
    probe_key    = f"{key_prefix}/probe.json"
    duration_sec = get_json(frames_bucket, probe_key).get("duration_sec")
    mp_out = {
        "frames_bucket":        frames_bucket,
        "manifest_key":         manifest_key,
        "probe_key":            probe_key,
        "duration_sec":         duration_sec,
        "media_processing_arn": mp_out.get("job_id"),
    }
    log.info("normalized gateway handle: s3://%s/%s (%.1fs)",
             frames_bucket, manifest_key, duration_sec or -1.0)

    return {
        **event,
        "data": {
            **data,
            "media_processing_output":  mp_out,
            "media_processing_request": None,
        },
    }
