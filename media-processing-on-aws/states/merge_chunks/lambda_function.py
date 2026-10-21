"""
Step Function State: StateMergeChunks
--------------------------------------
Merges segment chunks produced by the Map state (StateChunkTranscode) into
final output files, one per output_group.

Event shape:
  See states/merge_chunks/input.jsonc

data.iterators contains ChunkTranscode return dicts [{item_id, outputs}, ...],
sorted by item_id before processing. Each output entry routes by s3_prefix:
  "segment_video" → .ts chunk key
  "segment_audio" → .flac chunk key
  "frames"        → {start, step, total} descriptor

Per output_group from data.output_groups:
  "mp4"  → concat .ts → MP4 (with AAC audio if audio chunks present)
  "wav"  → concat .flac → decode → WAV
  "m4a"  → concat .flac → re-encode → M4A
  "jpeg"/"png" → reconstruct key list → write manifest JSON
  "metadata"   → already written by StateProbeMediaFile; skipped here

FLAC intermediates are deleted after all audio groups complete (success only).

Returns the event enriched with:
  data.output_keys  — {group_name: s3_key} for each completed group
  data.iterators    — deleted
  status            — "COMPLETED"
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time

from lib.s3_utils import delete_object, put_json
from lib.ad_mix import merge_ad_mix
from lib.input_schema import (
    VIDEO_FORMATS,
    AUDIO_FORMATS,
    FRAME_FORMATS,
    AD_MIX_FORMATS,
)
from states.merge_chunks.merge_helper import (
    merge_audio_group,
    merge_mp4,
    write_frames_manifest,
)

KEEP_SEGMENT_CHUNKS = bool(os.environ.get("KEEP_SEGMENT_CHUNKS"))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)


def lambda_handler(event: dict, context) -> dict:
    log.info("StateMergeChunks event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    uuid = event["uuid"]
    data = event.get("data", {})

    dest       = data["destination"]
    out_bucket = dest["bucket"]
    out_prefix = dest["prefix"]

    output_groups = data.get("output_groups") or []
    if not output_groups:
        raise ValueError("No output_groups — nothing to produce")

    iterators = list(data.get("iterators") or [])
    if not iterators:
        raise ValueError("No iterators — nothing to merge")

    iterators.sort(key=lambda it: it["item_id"])

    # ── Collect chunk keys by s3_prefix ───────────────────────────────────────
    video_keys:   list[str]        = []
    flac_keys:    list[str]        = []
    frame_descs:  dict[str, list]  = {}   # keyed by group_name (s3_prefix)
    captures:     dict[str, list]  = {}   # frame-capture records

    for it in iterators:
        for out in it["outputs"]:
            prefix = out["s3_prefix"]
            if prefix == "segment_video":
                video_keys.append(out["s3_key"])
            elif prefix == "segment_audio":
                flac_keys.append(out["s3_key"])
            elif "capture" in out:
                captures.setdefault(prefix, []).append(
                    {**out["capture"], "s3_key": out["s3_key"]})
            elif "frames" in out:
                frame_descs.setdefault(prefix, []).append(out["frames"])

    log.info(
        "Collected: %d video chunks  %d audio chunks  %d frame group(s)",
        len(video_keys), len(flac_keys), len(frame_descs),
    )

    tmp_dir = tempfile.mkdtemp(prefix=f"{uuid}_merge_")
    # Seeded from the probe state — metadata groups are written there, so a mixed
    # request must not lose them when this state rewrites data.output_keys.
    output_keys: dict[str, str] = dict(data.get("output_keys") or {})

    try:
        for group in output_groups:
            fmt        = group["format"]
            group_name = group["group_name"]
            name       = group.get("name") or fmt

            if fmt in VIDEO_FORMATS:
                if not video_keys:
                    log.warning("mp4 group %r: no video chunks — skipping", group_name)
                    continue
                out_key = f"{out_prefix}/{group_name}/{name}.mp4"
                merge_mp4(out_bucket, video_keys, flac_keys, tmp_dir, out_key)
                output_keys[group_name] = out_key

            elif fmt in AUDIO_FORMATS:
                if not flac_keys:
                    log.warning("%s group %r: no audio chunks — skipping", fmt, group_name)
                    continue
                ext     = "wav" if fmt == "wav" else "m4a"
                out_key = f"{out_prefix}/{group_name}/{name}.{ext}"
                merge_audio_group(out_bucket, flac_keys, tmp_dir, out_key, group)
                output_keys[group_name] = out_key

            elif fmt in AD_MIX_FORMATS:
                if not flac_keys:
                    log.warning("ad_mix group %r: no audio chunks — skipping", group_name)
                    continue
                out_key = f"{out_prefix}/{group_name}/{name}.wav"
                merge_ad_mix(out_bucket, flac_keys, tmp_dir, out_key, group)
                output_keys[group_name] = out_key

            elif fmt in FRAME_FORMATS and captures.get(group_name):
                caps = sorted(captures[group_name], key=lambda c: c["index"])
                manifest_key = f"{out_prefix}/{group_name}/capture_manifest.json"
                put_json(out_bucket, manifest_key, {
                    "total": len(caps),
                    "prefix": f"{out_prefix}/{group_name}",
                    "captures": caps,
                })
                output_keys[group_name] = manifest_key

            elif fmt in FRAME_FORMATS:
                descs = frame_descs.get(group_name)
                if not descs:
                    log.warning("%s group %r: no frame descriptors — skipping", fmt, group_name)
                    continue
                frames_prefix = f"{out_prefix}/{group_name}"
                manifest_key  = f"{out_prefix}/{group_name}/frame_manifest.json"
                dimension     = [group.get("width", 0), group.get("height", 0)]
                write_frames_manifest(out_bucket, manifest_key, descs, frames_prefix,
                                      frame_format=fmt, dimension=dimension)
                output_keys[group_name] = manifest_key

        # ── Delete chunk intermediates after all groups succeed ────────────────
        if KEEP_SEGMENT_CHUNKS:
            log.info("KEEP_SEGMENT_CHUNKS set — skipping deletion of %d chunk(s)",
                     len(video_keys) + len(flac_keys))
        else:
            for key in video_keys + flac_keys:
                try:
                    delete_object(out_bucket, key)
                    log.info("Deleted intermediate: %s", key)
                except Exception as exc:
                    log.warning("Failed to delete intermediate %s: %s", key, exc)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info("Cleaned up %s", tmp_dir)

    elapsed = round(time.monotonic() - t0, 2)
    log.info("StateMergeChunks complete: %d outputs  elapsed=%.2fs",
             len(output_keys), elapsed)

    merged_data = {k: v for k, v in data.items() if k != "iterators"}
    merged_data["output_keys"] = output_keys

    return {
        **event,
        "status": "COMPLETED",
        "data":   merged_data,
    }