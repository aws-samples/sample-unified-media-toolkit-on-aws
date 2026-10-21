"""
StateRenderWindows — the Map iteration (README.md §10; user decision
2026-08-19: Map + a SMALL thread pool, memory bounded per iteration).

Each iteration owns one batch of window indices. Per window: fetch the
member shots' frame tarballs (RENDER_THREADS-wide pool — the work is
S3-bound; the small pool is the OOM bound), pick the planned frames,
compose the window's face gallery when it has one, slice the transcript,
render through the COPIED prompt_render module (the identity-bearing
step), and checkpoint the finished record as `_records/window_NNN.json`.

Re-enter/resume (face-recognition's pattern): the iteration lists its
existing side objects first and SKIPS completed windows; between windows
it watches `get_remaining_time_in_millis()` and raises
LambdaResourceTimeoutException — the ASL Retry re-invokes and the finished
windows cost one LIST, not a re-render.

Rendering is deterministic (same plan → same bytes): a redriven submit
absorbs into its own bedrock run, a fresh caller token re-renders
byte-identical records under a new one.

record_id convention: `window-{index:03d}` — the join key from
response.jsonl back to plan windows.
"""
from __future__ import annotations

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from lib import frames_io
from lib import gallery as gallery_mod
from lib import prompt_render
from lib import sme_common
from lib.custom_exceptions import LambdaResourceTimeoutException

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = sme_common.read_json(
        *sme_common.parse_s3_uri(event["plan_s3uri"]))
    indices = event["window_indices"]
    prefix = plan["output_prefix"]

    existing = set(sme_common.list_keys(
        sme_common.PROXY_BUCKET, f"{prefix}/_records/"))
    transcript = _load_transcript(plan)
    face_cache: dict = {}
    declared = {v["token"] for v in plan["prompt"]["variables"]}

    done = skipped = 0
    for index in indices:
        key = f"{prefix}/_records/window_{index:03d}.json"
        if key in existing:
            skipped += 1
            continue
        if context.get_remaining_time_in_millis() \
                < sme_common.REMAINING_TIME_BUFFER_MS:
            raise LambdaResourceTimeoutException(
                f"render checkpointed at window {index} "
                f"({done} rendered, {skipped} already present)")
        record = _render_window(plan, plan["windows"][index], transcript,
                                face_cache, declared)
        sme_common.write_json(sme_common.PROXY_BUCKET, key, record)
        done += 1

    logger.info("batch %s: %d rendered, %d skipped",
                event.get("batch_index"), done, skipped)
    return {"batch_index": event.get("batch_index"),
            "rendered": done, "skipped": skipped}


def _render_window(plan: dict, window: dict, transcript,
                   face_cache: dict, declared: set) -> dict:
    shots_by_id = _shots_by_id(plan)
    ordered = ([window["context_shot_id"]]
               if window["context_shot_id"] is not None else []) \
        + list(window["shot_ids"])
    members = [shots_by_id[sid] for sid in ordered]

    # plan.json round-trips through JSON, so the frames map's shot_id keys
    # arrive as strings — normalize once.
    planned = {int(k): v for k, v in window["frames"].items()}

    # S3-bound fan-out, deliberately narrow (the OOM bound): each worker
    # fetches one shot's tarball and extracts only its planned picks.
    def _fetch(shot):
        # Frameless shots (non-representative burst-run members) cost no
        # tarball fetch — the fold's saving is images AND I/O.
        if planned[shot["shot_id"]] == 0:
            return []
        tar = sme_common.read_bytes(sme_common.PROXY_BUCKET,
                                    shot["frames_key"])
        return frames_io.pick_frames(tar, planned[shot["shot_id"]])

    with ThreadPoolExecutor(max_workers=sme_common.RENDER_THREADS) as pool:
        picked = list(pool.map(_fetch, members))

    burst = plan.get("burst") or {}
    recurring = plan.get("recurring") or {}
    shot_list, blocks, image_no = [], [], 0
    for shot, frames in zip(members, picked):
        sid = shot["shot_id"]
        role = "CONTEXT" if sid == window["context_shot_id"] else "REQUIRED"
        entry = {
            "shot_id": sid, "role": role,
            # Slim form (README §5): NO timecodes — 575 -> 132 tokens, and
            # the prompt forbids emitting times anyway.
            "duration_sec": round(float(shot["end_time"])
                                  - float(shot["start_time"]), 2),
            "frame_count": len(frames),
        }
        # Structural evidence annotations (README §11): burst-run membership
        # (frameless members are judged from the run's representatives) and
        # programme-level recurrence the model cannot see from one window.
        if str(sid) in burst:
            entry["burst_run"] = burst[str(sid)]
        if str(sid) in recurring:
            entry["recurring"] = recurring[str(sid)]
        shot_list.append(entry)
        for k, raw in enumerate(frames, start=1):
            image_no += 1
            blocks.append({"text": frames_io.frame_label(
                image_no, sid, k, len(frames), role)})
            blocks.append({"image": {"format": "jpeg", "source": {
                "bytes": base64.b64encode(raw).decode("ascii")}}})

    values = {
        "{{program_names}}": plan["program_names"],
        "{{overall_content_type}}": plan["overall_content_type"],
        "{{shot_list}}": shot_list,
        "{{face_names}}": _face_names(plan, window),
        "{{face_images}}": _gallery_blocks(plan, window, face_cache),
        "{{transcript}}": _transcript_slice(
            transcript, float(members[0]["start_time"]),
            float(members[-1]["end_time"])),
        "{{frame_images}}": blocks,
    }
    rendered = prompt_render.render(
        plan["prompt"],
        {k: v for k, v in values.items() if k in declared})
    return {"record_id": f"window-{window['index']:03d}",
            "system": rendered["system"],
            "messages": rendered["messages"]}


def _shots_by_id(plan: dict) -> dict:
    cached = plan.get("_shots_by_id")
    if cached is None:
        doc = sme_common.read_json(
            *sme_common.parse_s3_uri(plan["shots"]["s3uri"]))
        cached = {s["shot_id"]: s for s in doc["shots"]}
        plan["_shots_by_id"] = cached      # per-invocation memo only
    return cached


def _face_names(plan: dict, window: dict) -> str:
    from lib import faces as faces_mod
    shot_names = {int(k): v
                  for k, v in (plan["faces"]["shot_names"] or {}).items()}
    return faces_mod.face_names_text(
        window["shot_ids"], window["context_shot_id"], shot_names)


def _gallery_blocks(plan: dict, window: dict, face_cache: dict):
    """The {{face_images}} value: [label text, ONE gallery image] — or the
    text sentinel when this window carries no recognized faces."""
    from lib import faces as faces_mod
    shot_names = {int(k): v
                  for k, v in (plan["faces"]["shot_names"] or {}).items()}
    roster = faces_mod.window_gallery(
        ([window["context_shot_id"]]
         if window["context_shot_id"] is not None else [])
        + list(window["shot_ids"]),
        shot_names, plan["faces"]["people"], sme_common.GALLERY_MAX_FACES)
    if not roster:
        return "(no reference faces supplied for this window)"
    jpegs = []
    for person in roster:
        key = person["rep_face_key"]
        if key not in face_cache:
            face_cache[key] = sme_common.read_bytes(
                sme_common.PROXY_BUCKET, key)
        jpegs.append(face_cache[key])
    image = gallery_mod.compose_gallery(
        jpegs, cell_w=sme_common.GALLERY_CELL_W,
        cell_h=sme_common.GALLERY_CELL_H,
        divider_px=sme_common.GALLERY_DIVIDER_PX)
    return [
        {"text": gallery_mod.gallery_label([p["name"] for p in roster])},
        {"image": {"format": "jpeg", "source": {
            "bytes": base64.b64encode(image).decode("ascii")}}},
    ]


def _load_transcript(plan: dict):
    uri = plan["transcript"]["s3uri"]
    if not uri:
        return None
    doc = sme_common.read_json(*sme_common.parse_s3_uri(uri))
    sentences = doc.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError(
            f"{uri} carries no sentences[] — expected a speech-diarization "
            "final_transcript.json")
    return sentences


def _transcript_slice(sentences, t0: float, t1: float) -> str:
    if sentences is None:
        return "(no transcript supplied for this window)"
    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] "
             f"{s.get('speaker_id', 'spk')}: {s.get('text', '')}"
             for s in sentences
             if float(s["end"]) > t0 and float(s["start"]) < t1]
    return "\n".join(lines) or "(no speech in this window)"
