"""
Frame-capture renderer: one "capture" iterator (one source's marker
batch, sorted by position, span-capped at probe) → ONE ffmpeg
invocation producing every still in the batch.

Why one invocation (measured live 2026-10-09): deep RANDOM access into
a large long-GOP MXF over presigned HTTPS costs minutes PER SEEK, and
the first design paid it twice per marker (keyframe scan + render) —
two markers 600 s apart in a 2.5 h MXF overran the Lambda's 900 s
ceiling. SEQUENTIAL read is cheap, so the fix is the chunk pipeline's
economics: seek once (prior keyframe before the FIRST marker — or the
frame itself on all-intra), read forward through the batch's span, and
pick every wanted frame by COUNTING (per-branch
trim=start_frame=n:end_frame=n+1 — the house frame-exact mechanism;
select is banned here), never by seek precision. The batch span is capped at probe
(frame_capture.MAX_SPAN_MS) so one invocation's read stays
chunk-sized.
"""
from __future__ import annotations

import os
from fractions import Fraction

from lib.cut_list import parse_fps
from lib.edit_filters import crop_pixels, fit_dimensions
from lib.ffprobe import find_prior_keyframe
from lib.proc_utils import run
from lib.s3_utils import presign
from lib.video_command import sec_to_frame
from states.chunk_transcode.transcode_helper import (
    FFMPEG, THREADS, ValidationException,
)


def build_capture_vfilters(src: dict, crop: dict | None,
                           resize: tuple | list | None) -> list[str]:
    """[crop] → [fit+pad] (never stretch); native size when resize is
    absent. Rotation is ffmpeg's decoder auto-rotate. Deinterlacing is
    NOT here: yadif sits ONCE before the split in build_batch_command."""
    filters: list[str] = []
    w, h = src["width"], src["height"]
    if crop is not None:
        cw, ch, cx, cy = crop_pixels(crop, w, h)
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
        w, h = cw, ch
    if resize:
        canvas_w, canvas_h = resize
        ow, oh = fit_dimensions(w, h, canvas_w, canvas_h)
        filters.append(f"scale={ow}:{oh}")
        if (ow, oh) != (canvas_w, canvas_h):
            filters.append(f"pad={canvas_w}:{canvas_h}:"
                           f"{(canvas_w - ow) // 2}:{(canvas_h - oh) // 2}")
    return filters


def batch_seek_plan(src: dict, first_frame: int) -> tuple[float | None, int]:
    """(seek_ss, base_frame) for the batch — the chunk-transcode seek
    policy. Frames are then picked by COUNT relative to base_frame."""
    src_fps = parse_fps(src["fps"])
    if first_frame <= 0:
        return None, 0
    if src.get("is_all_intra"):
        return float(Fraction(first_frame) / src_fps), first_frame
    kf = find_prior_keyframe(src["bucket"], src["key"],
                             float(Fraction(first_frame) / src_fps))
    try:
        kf_ts = float(kf.get("best_effort_timestamp_time"))
    except (TypeError, ValueError):
        raise ValidationException(
            "find_prior_keyframe returned non-numeric "
            f"best_effort_timestamp_time={kf.get('best_effort_timestamp_time')!r}")
    base = round(sec_to_frame(kf_ts, src_fps))
    if base > first_frame:
        raise ValidationException(
            f"keyframe frame {base} is past the first marker frame "
            f"{first_frame}")
    return kf_ts, base


def build_batch_command(iterator: dict, url: str, seek: float | None,
                        base_frame: int, tmp_dir: str) -> tuple[list, list[dict]]:
    """One batch → (ffmpeg argv, per-marker outputs). K trim branches
    off one decoded stream; each trim closes after its one frame and
    ffmpeg stops when every output is done — the read never passes the
    last marker."""
    src = iterator["source"]
    fmt = iterator["format"]
    ext = "png" if fmt == "png" else "jpg"
    q_args = (["-compression_level", "6"] if fmt == "png"
              else ["-q:v", "2"])
    markers = iterator["markers"]
    k = len(markers)

    lines, outputs = [], []
    interlaced = src.get("field_order") not in (None, "", "progressive",
                                                "unknown")
    # deinterlace ONCE, pre-split (yadif 1:1 in frame mode — the
    # per-branch trim counts are unchanged)
    pre = "yadif=deint=interlaced," if interlaced else ""
    if k > 1:
        lines.append(f"[0:v]{pre}split={k}" +
                     "".join(f"[c{i}]" for i in range(k)))
    elif interlaced:
        lines.append(f"[0:v]{pre.rstrip(',')}[c0]")
    for i, m in enumerate(markers):
        n = m["frame"] - base_frame
        if n < 0:
            raise ValidationException(
                f"markers[{m['index']}]: frame {m['frame']} precedes the "
                f"batch seek landing {base_frame}")
        src_label = f"[c{i}]" if (k > 1 or interlaced) else "[0:v]"
        # frame picked by TRIM COUNT — the house frame-exact mechanism
        # (the chunk cut), never select: select is not frame-exact in
        # this stack's history and never EOFs its branch (a downstream
        # filter then waits forever and ffmpeg reads to EOF — the
        # 2026-10-09 interlaced-batch 900 s overrun). trim emits frame
        # n alone and closes the branch deterministically.
        chain = [f"trim=start_frame={n}:end_frame={n + 1}",
                 "setpts=PTS-STARTPTS"] + \
            build_capture_vfilters(src, m.get("crop"), iterator.get("resize"))
        lines.append(f"{src_label}{','.join(chain)}[o{i}]")
        name = f"frame_{m['index']:04d}.{ext}"
        outputs.append({"s3_prefix": "frames", "s3_name": name,
                        "local_path": os.path.join(tmp_dir, name),
                        "capture": {
                            "index": m["index"],
                            "source_s3uri": m["source_s3uri"],
                            "requested_ms": m["at_ms"],
                            "actual_ms": m["actual_ms"],
                            "frame": m["frame"],
                        }})

    cmd = [FFMPEG, "-threads", str(THREADS), "-v", "error", "-y"]
    if seek is not None:
        cmd += ["-ss", f"{seek:.6f}"]
    cmd += ["-i", url, "-filter_complex", ";".join(lines)]
    for i, out in enumerate(outputs):
        cmd += ["-map", f"[o{i}]", "-frames:v", "1"] + q_args + \
               [out["local_path"]]
    return cmd, outputs


def render_captures(iterator: dict, tmp_dir: str) -> list[dict]:
    """One iterator = one source's sorted, span-capped marker batch =
    one seek + one ffmpeg. Returns handler-shaped outputs."""
    src = iterator["source"]
    url = presign(src["bucket"], src["key"])
    first_frame = min(m["frame"] for m in iterator["markers"])
    seek, base_frame = batch_seek_plan(src, first_frame)
    cmd, outputs = build_batch_command(iterator, url, seek, base_frame,
                                       tmp_dir)
    run(cmd, timeout=840)
    for out in outputs:
        if not os.path.exists(out["local_path"]) or \
                os.path.getsize(out["local_path"]) == 0:
            raise ValueError(
                f"markers[{out['capture']['index']}]: no frame decoded at "
                f"{out['capture']['requested_ms']} ms "
                f"(frame {out['capture']['frame']})")
    return outputs
