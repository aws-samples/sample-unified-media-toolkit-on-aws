"""
Step Function State: StateChunkTranscode
-----------------------------------------
Transcodes one chunk iterator produced by StateProbeMediaFile.

Event shape (from Map state ItemSelector):
  {
    "operation": "StateChunkTranscode",
    "item_data": <iterator dict from ProbeMediaFile>   # item_id is read from item_data.item_id
  }

Returns:
  {
    "item_id": <int>,
    "outputs": [
      {"s3_prefix": ..., "s3_key": ...},                             # segment_video / segment_audio
      {"s3_prefix": ..., "frames": {"start": N, "step": N, "total": N}}  # frames
    ]
  }
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile

from lib.s3_utils import upload_file
from states.chunk_transcode.transcode_helper import transcode, upload_frames

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)


def lambda_handler(event: dict, context) -> dict:
    log.info("StateChunkTranscode event keys = %s", sorted(event.keys()))

    iterator = event["item_data"]
    item_id  = iterator["item_id"]

    uuid       = iterator["uuid"]
    out        = iterator["output"]
    out_bucket = out["bucket"]
    out_prefix = out["prefix"]

    tmp_dir = tempfile.mkdtemp(prefix=f"{uuid}_")
    try:
        if iterator.get("type") == "edit":
            from states.chunk_transcode.edit_render import render_edit
            chunk_outputs = render_edit(iterator, tmp_dir)
        elif iterator.get("type") == "capture":
            from states.chunk_transcode.capture_render import render_captures
            chunk_outputs = render_captures(iterator, tmp_dir)
        else:
            chunk_outputs = transcode(iterator, tmp_dir)

        outputs = []
        for output in chunk_outputs:
            s3_prefix = output["s3_prefix"]
            nominal_start_frame = iterator.get("nominal_start_frame", 0)
            frame_interval      = iterator.get("frame_interval") or 1

            if output.get("local_pattern"):
                frame_keys = upload_frames(
                    out_bucket, out_prefix, s3_prefix,
                    tmp_dir, output["local_pattern"],
                    nominal_start_frame,
                    frame_interval,
                    frame_format=output.get("frame_format", "jpeg"),
                )
                outputs.append({
                    "s3_prefix": s3_prefix,
                    "frames": {
                        "start": nominal_start_frame,
                        "step":  frame_interval,
                        "total": len(frame_keys),
                    },
                })
            else:
                s3_key = f"{out_prefix}/{s3_prefix}/{output['s3_name']}"
                upload_file(out_bucket, s3_key, output["local_path"])
                log.info("Uploaded s3://%s/%s", out_bucket, s3_key)
                entry = {"s3_prefix": s3_prefix, "s3_key": s3_key}
                if "capture" in output:   # frame-capture record → merge
                    entry["capture"] = output["capture"]
                outputs.append(entry)

        return {
            "item_id": item_id,
            "outputs": outputs,
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info("Cleaned up %s", tmp_dir)