"""
Frame-capture branch of StateProbeMediaFile.

Probe each distinct source once (reusing the quick-edit source prober),
re-validate markers against the real files, snap each marker to the
SOURCE frame grid, and emit "capture" iterators (markers grouped by
source, batched). The Map iterations upload the stills; StateMergeChunks
writes capture_manifest.json from their returned capture records — no
concat happens.
"""
from __future__ import annotations

import logging
import time

from lib.cut_list import frame_to_ms, ms_to_frame, parse_fps
from lib.edit_filters import is_hdr
from lib.frame_capture import batch_markers, validate_frame_capture
from lib.input_schema import parse_output_groups
from lib.s3_utils import put_json
from states.probe_media_file.quick_edit_probe import (_probe_source,
                                                       probe_sources)

log = logging.getLogger(__name__)


def handle_frame_capture(event: dict, fc_raw: dict, uuid: str,
                         out_bucket: str, out_prefix: str,
                         output_groups_req, fn_arn: str) -> dict:
    t0 = time.monotonic()
    norm = validate_frame_capture(fc_raw)

    facts = probe_sources(norm["distinct_sources"], _probe_source)
    for m in norm["markers"]:
        f = facts[m["source_s3uri"]]
        if not f["has_video"]:
            raise ValueError(
                f"markers[{m['index']}]: the source has no video stream")
        if is_hdr(f.get("color_transfer")):
            raise ValueError(
                f"markers[{m['index']}]: the source is HDR "
                f"({f['color_transfer']}); HDR requires tonemapping, which "
                f"this workflow does not support — supply an SDR source")
        if m["at_ms"] > f["duration_sec"] * 1000:
            raise ValueError(
                f"markers[{m['index']}]: at_ms {m['at_ms']} is beyond the "
                f"source's {f['duration_sec']:.3f}s duration")
        # snap to the SOURCE grid — no target fps exists for stills
        src_fps = parse_fps(f["fps"])
        m["frame"] = ms_to_frame(m["at_ms"], src_fps)
        m["actual_ms"] = frame_to_ms(m["frame"], src_fps)
        # a marker within half a frame period of the end rounds PAST the
        # last frame — refuse here by name instead of letting the render
        # discover an empty trim after a full sequential read
        n_frames = int(f["duration_sec"] * float(src_fps))
        if n_frames > 0 and m["frame"] >= n_frames:
            raise ValueError(
                f"markers[{m['index']}]: at_ms {m['at_ms']} snaps to frame "
                f"{m['frame']}, past the source's last frame "
                f"({n_frames - 1} at {f['duration_sec']:.3f}s)")

    batches = batch_markers(norm["markers"])
    iterators = [{
        "function_arn": fn_arn,
        "uuid": uuid,
        "item_id": i,
        "output": {"bucket": out_bucket, "prefix": out_prefix},
        "type": "capture",
        "source": {k: facts[batch[0]["source_s3uri"]][k]
                   for k in ("bucket", "key", "fps", "field_order",
                             "width", "height", "is_all_intra")},
        "format": norm["format"],
        "resize": norm["resize"],
        "markers": batch,
    } for i, batch in enumerate(batches)]

    log.info("frame-capture probe: %d markers, %d sources, %d iterations  "
             "elapsed=%.2fs", len(norm["markers"]), len(facts),
             len(iterators), time.monotonic() - t0)

    output_groups = parse_output_groups(
        output_groups_req or [{"group_name": "frames",
                               "format": norm["format"]}],
        has_video=True, has_audio=False)

    probe_key = f"{out_prefix}/probe.json"
    put_json(out_bucket, probe_key, {
        "uuid": uuid,
        "frame_capture": True,
        "marker_count": len(norm["markers"]),
        "sources": facts,
    })

    return {
        **event,
        "status": "COMPLETED",
        "data": {
            **event.get("data", {}),
            "destination": {"bucket": out_bucket, "prefix": out_prefix},
            "output_groups": output_groups,
            "probe_s3_key": probe_key,
            "duration_sec": 0.0,
            "has_video": True,
            "has_audio": False,
            "is_all_intra": False,
            "metadata_only": False,
            "output_keys": {},
            "function_arn": fn_arn,
            "iterators": iterators,
        },
    }
