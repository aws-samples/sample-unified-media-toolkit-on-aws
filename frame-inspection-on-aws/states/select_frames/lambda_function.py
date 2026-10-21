"""
StateSelectFrames (one Map iteration, a batch of span units): for each
unit, pull the windows the span touches, select frames (embedding change
gate + heartbeat, lib/selection), extract ONLY the chosen JPEGs from the
window tarballs, render <=IMAGES_PER_RECORD-image records through the
copied prompt_render, and checkpoint _records/unit_NNNNN.json.

Resume semantics: an existing checkpoint is SKIPPED — a
LambdaResourceTimeoutException retry resumes, never restarts. The
timeout guard raises with 60 s left, enrichment's number.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import tarfile

from lib import fi_common
from lib import imaging
from lib import prompt_render
from lib import selection

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_TIMEOUT_BUFFER_MS = 60_000


class LambdaResourceTimeoutException(Exception):
    """Raised near the Lambda deadline so the ASL retry re-enters."""


def lambda_handler(event: dict, context) -> dict:
    plan = fi_common.read_json(*fi_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    done = []
    for unit_index in event["unit_indices"]:
        checkpoint_key = f"{prefix}/_records/unit_{unit_index:05d}.json"
        if fi_common.read_json_if_exists(
                fi_common.PROXY_BUCKET, checkpoint_key) is not None:
            done.append(unit_index)
            continue
        if context and context.get_remaining_time_in_millis() < _TIMEOUT_BUFFER_MS:
            raise LambdaResourceTimeoutException(
                f"{len(done)}/{len(event['unit_indices'])} units done — "
                "re-enter to continue")
        records = _render_unit(plan, plan["units"][unit_index])
        fi_common.write_json(fi_common.PROXY_BUCKET, checkpoint_key,
                             {"unit_index": unit_index, "records": records})
        done.append(unit_index)
    return {"batch_index": event["batch_index"], "units_done": done}


def _render_unit(plan: dict, unit: dict) -> list:
    asset = plan["assets"][unit["asset_idx"]]
    req = next(r for r in asset["requests"]
               if r["category"] == unit["category"])
    span = req["spans"][unit["span_idx"]]
    conf = fi_common.CATEGORIES[unit["category"]]

    frames = _select_span_frames(asset, span, conf)
    if not frames:
        return []          # collect_finalize reports the span as a gap
    if span.get("crop"):
        # caller-requested per-span crop (chyron reads want only the
        # lower band — ~60% fewer image tokens); validated at plan time
        frames = [(t, imaging.crop(jpeg, span["crop"]))
                  for t, jpeg in frames]
    artifact = plan["prompts"][unit["category"]]

    records = []
    per = plan["images_per_record"]
    for chunk_no, start in enumerate(range(0, len(frames), per)):
        chunk = frames[start:start + per]
        blocks = []
        for t, jpeg in chunk:
            blocks.append({"text": f"Frame at t={t}:"})
            blocks.append({"image": {
                "format": "jpeg",
                "source": {"bytes": base64.b64encode(jpeg).decode("ascii")},
            }})
        values = {
            "{{frame_images}}": blocks,
            "{{span_context}}": (
                f"Span {span.get('id') or unit['span_idx']}: "
                f"{span['start']:.2f}s to {span['end']:.2f}s"
                + (f", shot kinds {span['kinds']}" if span.get("kinds") else "")
                + f". Frames {start + 1}-{start + len(chunk)} "
                  f"of {len(frames)} selected."),
        }
        if conf["needs_question"]:
            values["{{question}}"] = span["question"]
        rendered = prompt_render.render(artifact, values)
        records.append({
            "record_id": (f"{unit['asset_idx']}:{unit['category']}:"
                          f"{unit['span_idx']}:{chunk_no}"),
            "system": rendered["system"],
            "messages": rendered["messages"],
        })
    return records


def _select_span_frames(asset: dict, span: dict, conf: dict) -> list:
    """(programme_second, jpeg_bytes) for every frame worth showing —
    selection first, tarball extraction ONLY for the chosen members."""
    heartbeat = conf["heartbeat_sec"]
    out = []
    for win in asset["windows"]:
        if win["end"] < span["start"] or win["start"] > span["end"]:
            continue
        manifest = fi_common.read_json(win["bucket"], win["manifest_key"])
        emb_blob = fi_common.read_bytes(win["bucket"], win["npy_key"])
        kept = selection.select(
            manifest, emb_blob, span,
            change_distance=conf["change_distance"],
            heartbeat_sec=heartbeat if heartbeat is not None else 1e9)
        if not kept:
            continue
        wanted = {name: t for t, name in
                  selection.frame_names(manifest, kept)}
        tar_bytes = fi_common.read_bytes(win["bucket"], win["tar_key"])
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                name = member.name.rsplit("/", 1)[-1]
                if name in wanted:
                    out.append((wanted[name],
                                tar.extractfile(member).read()))
    out.sort(key=lambda pair: pair[0])
    if len(out) > fi_common.MAX_FRAMES_PER_SPAN:
        logger.warning(
            "span %.2f-%.2f selected %d frames — capping to %d evenly "
            "spaced (fi_common.MAX_FRAMES_PER_SPAN)", span["start"],
            span["end"], len(out), fi_common.MAX_FRAMES_PER_SPAN)
        out = selection.evenly_spaced(out, fi_common.MAX_FRAMES_PER_SPAN)
    return out
