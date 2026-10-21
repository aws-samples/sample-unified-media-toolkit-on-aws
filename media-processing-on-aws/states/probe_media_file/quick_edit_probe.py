"""
Quick-edit branch of StateProbeMediaFile.

One entry point, handle_quick_edit(): validate the cut list (pure rules
already enforced by lib/cut_list), probe each DISTINCT source once,
re-validate against the real files (durations, stream presence for the
channel selectors, the v1 HDR refusal), derive or check the target,
measure per-segment/per-element loudness (pass 1 of the two-pass
loudnorm — ONE measurement per segment; its junction shares reuse the
same gain), write the full plan as an S3 side object, and emit SLIM
"edit" iterators (one per plan entry) for the existing Map state.

The entries produce the same segment_video/*.ts + segment_audio/*.flac
chunk outputs as uniform chunking, so StateMergeChunks runs unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import time
from fractions import Fraction

from lib.cut_list import (BLACK_SOURCE, build_plan, frame_to_ms,  # noqa: F401
                          ms_to_frame, parse_fps, validate_quick_edit)
from lib.edit_filters import crop_pixels, is_hdr, loudnorm_measure_filter
from lib.ffprobe import resolve_duration, run_ffprobe
from lib.input_schema import parse_output_groups, parse_s3uri
from lib.proc_utils import run
from lib.s3_utils import archive_tier, head_object, presign, put_json
from lib.video_command import determine_is_all_intra

log = logging.getLogger(__name__)

# Per measured source. 840 s matches the house ffmpeg budget
# (FFPROBE_TIMEOUT/chunk transcode); with the fixed shallow window the
# budget is generous rather than load-bearing.
LOUDNORM_TIMEOUT = 840
# Loudness is sampled ONCE PER UNIQUE SOURCE over a fixed window
# (user decision 2026-10-10), not per segment over its exact in/out:
# a per-SOURCE gain corrects master-level mismatch between sources
# while PRESERVING intra-source dynamics (a quiet scene stays quieter
# than a loud one from the same master — per-segment normalization
# flattened that); it dedupes N-segment cost to U-source cost; and the
# fixed shallow window (start ~3:00 to skip bars/tone/cold opens)
# removes the deep-seek cost class from measurement entirely.
LOUDNESS_WINDOW_MS = 300_000
LOUDNESS_START_MS = 180_000


def _loudness_window(duration_sec: float) -> tuple[float, float]:
    dur_ms = int(duration_sec * 1000)
    win = min(LOUDNESS_WINDOW_MS, dur_ms)
    start = min(LOUDNESS_START_MS, max(0, dur_ms - win))
    return start / 1000.0, (start + win) / 1000.0


def _parse_resolution(res: str) -> tuple[int, int]:
    try:
        w, h = res.lower().split("x", 1)
        w, h = int(w), int(h)
    except (ValueError, AttributeError):
        raise ValueError(f"target.resolution is not WIDTHxHEIGHT: {res!r}")
    if w < 2 or h < 2:
        raise ValueError(f"target.resolution is not a usable canvas: {res!r}")
    if w % 2 or h % 2:
        raise ValueError(
            f"target.resolution must be even in both dimensions "
            f"(4:2:0 chroma subsampling): {res!r}")
    return w, h


def _probe_source(uri: str) -> dict:
    """head (archive gate) + ffprobe → the facts re-validation needs."""
    bucket, key = parse_s3uri(uri)
    head = head_object(bucket, key)
    tier = archive_tier(head)
    if tier:
        raise ValueError(
            f"{uri} is in the {tier} access tier and cannot be read.")
    probe = run_ffprobe(bucket, key)
    streams = probe.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = [s for s in streams if s.get("codec_type") == "audio"]
    # the stack's canonical two-rung resolve (duration_ts × time_base
    # first, fmt.duration as last resort) — same as the uniform-chunk path
    ref = v if v is not None else (a[0] if a else {})
    duration_sec, _ = resolve_duration(ref, probe.get("format", {}))
    facts = {
        "bucket": bucket, "key": key,
        "etag": (head.get("ETag") or "").strip('"'),
        "duration_sec": duration_sec,
        "has_video": v is not None,
        "has_audio": bool(a),
    }
    if v is not None:
        facts.update({
            "width": int(v.get("width") or 0),
            "height": int(v.get("height") or 0),
            "fps": v.get("r_frame_rate") or "",
            "field_order": v.get("field_order") or "progressive",
            "color_transfer": v.get("color_transfer") or "",
            # routes the renderer's seek: all-intra → direct frame seek;
            # inter-frame or unknown (None → False) → keyframe scan +
            # preroll COUNT, the proven chunk-transcode machinery
            "is_all_intra": bool(determine_is_all_intra(v)),
        })
    return facts


def _revalidate(norm: dict, facts: dict[str, dict]) -> None:
    """The checks only real files can answer. Conditions named."""
    for i, seg in enumerate(norm["segments"]):
        src = seg["source_s3uri"]
        if src == BLACK_SOURCE:
            continue
        f = facts[src]
        if seg["out_ms"] > f["duration_sec"] * 1000 + 1:
            raise ValueError(
                f"segments[{i}]: out_ms {seg['out_ms']} is beyond the "
                f"source's {f['duration_sec']:.3f}s duration")
        if seg["channel"] in ("av", "v") and not f["has_video"]:
            raise ValueError(
                f"segments[{i}]: channel {seg['channel']!r} needs a video "
                f"stream and the source has none")
        if seg["channel"] == "a" and not f["has_audio"]:
            raise ValueError(
                f"segments[{i}]: channel 'a' needs an audio stream and "
                f"the source has none")
        if f["has_video"] and is_hdr(f.get("color_transfer")):
            raise ValueError(
                f"segments[{i}]: the source is HDR "
                f"({f['color_transfer']}); HDR requires tonemapping, "
                f"which this workflow does not support — supply an SDR "
                f"source")
    for ov in norm["audio_overlays"] + [
            s["transition_in"]["audio_bed"] for s in norm["segments"]
            if "audio_bed" in s["transition_in"]]:
        f = facts[ov["source_s3uri"]]
        if not f["has_audio"]:
            raise ValueError(
                f"audio element {ov['source_s3uri']} has no audio stream")
        if ov["out_ms"] > f["duration_sec"] * 1000 + 1:
            raise ValueError(
                f"audio element {ov['source_s3uri']}: out_ms "
                f"{ov['out_ms']} is beyond the source's "
                f"{f['duration_sec']:.3f}s duration")


def _derive_target(norm: dict, facts: dict[str, dict]) -> tuple[int, int, str]:
    """Explicit target wins; else all video sources must agree."""
    target = norm["target"]
    res, fps = target.get("resolution"), target.get("fps")
    if res and fps:
        w, h = _parse_resolution(res)
        return w, h, str(parse_fps(fps))
    video_facts = [facts[s["source_s3uri"]] for s in norm["segments"]
                   if s["source_s3uri"] != BLACK_SOURCE
                   and facts[s["source_s3uri"]]["has_video"]
                   and s["channel"] in ("av", "v")]
    if not video_facts:
        raise ValueError("no video source to derive a target from; "
                         "specify target.resolution and target.fps")
    dims = {(f["width"], f["height"]) for f in video_facts}
    rates = {f["fps"] for f in video_facts}
    if res:
        w, h = _parse_resolution(res)
    elif len(dims) == 1:
        (w, h), = dims
    else:
        raise ValueError("sources disagree on resolution; specify "
                         "target.resolution")
    if fps:
        fps_str = str(parse_fps(fps))
    elif len(rates) == 1:
        fps_str = str(parse_fps(rates.pop()))
    else:
        raise ValueError("sources disagree on framerate; specify "
                         "target.fps")
    return w, h, fps_str


def _measure_loudness(bucket: str, key: str,
                      in_sec: float, out_sec: float) -> dict:
    """loudnorm pass 1 over one element's range — audio decode only."""
    url = presign(bucket, key)
    text = run(["ffmpeg", "-hide_banner", "-nostats",
                "-ss", f"{in_sec:.3f}", "-to", f"{out_sec:.3f}",
                "-i", url, "-vn",
                "-af", loudnorm_measure_filter(),
                "-f", "null", "-"],
               timeout=LOUDNORM_TIMEOUT, return_stderr=True)
    # loudnorm reports on stderr; the json block is its tail
    start = text.rfind("{")
    if start < 0:
        raise ValueError(
            f"loudness measurement produced no result for "
            f"s3://{bucket}/{key} [{in_sec:.3f}–{out_sec:.3f}s]")
    return json.loads(text[start:] if text.endswith("}")
                      else text[start:text.rfind("}") + 1])


def probe_sources(uris, probe) -> dict:
    """head+ffprobe each distinct source over a bounded pool. A deep MXF
    probe is seconds of network round-trips (header partition + footer
    index over presigned HTTPS), so 25 sources sequentially is minutes of
    pure latency; the work is I/O-bound, same as the loudness pool below.
    `probe` is passed in (not referenced) so each caller's module-local
    `_probe_source` — and the tests that patch it — keep working.
    pool.map preserves input order, so a refusal surfaces for the FIRST
    bad source exactly as the sequential loop did."""
    uris = list(uris)
    if len(uris) <= 1:
        return {u: probe(u) for u in uris}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(4, len(uris))) as pool:
        return dict(zip(uris, pool.map(probe, uris)))


def handle_quick_edit(event: dict, qe_raw: dict, uuid: str, out_bucket: str,
                      out_prefix: str, output_groups_req, fn_arn: str) -> dict:
    t0 = time.monotonic()
    norm = validate_quick_edit(qe_raw)

    facts = probe_sources(norm["distinct_sources"], _probe_source)
    _revalidate(norm, facts)
    canvas_w, canvas_h, fps_str = _derive_target(norm, facts)
    fps = parse_fps(fps_str)
    plan = build_plan(norm, fps)

    # Loudness: default ON when segments span >1 distinct source
    # (don't silently re-master a single-source edit); flag overrides.
    seg_sources = {s["source_s3uri"] for s in norm["segments"]
                   if s["source_s3uri"] != BLACK_SOURCE}
    loudnorm_on = qe_raw.get("loudnorm")
    if loudnorm_on is None:
        loudnorm_on = len(seg_sources) > 1
    loudness: dict[str, dict] = {}
    if loudnorm_on:
        audio_sources = {
            seg["source_s3uri"] for seg in norm["segments"]
            if seg["source_s3uri"] != BLACK_SOURCE
            and seg["channel"] != "v"
            and facts[seg["source_s3uri"]]["has_audio"]
        } | {ov["source_s3uri"] for ov in plan["overlays"]}
        # thread-pooled: the work is I/O-bound presigned reads
        from concurrent.futures import ThreadPoolExecutor

        def _measure(uri: str) -> tuple[str, dict]:
            f = facts[uri]
            lo, hi = _loudness_window(f["duration_sec"])
            return uri, _measure_loudness(f["bucket"], f["key"], lo, hi)

        with ThreadPoolExecutor(max_workers=4) as pool:
            loudness = dict(pool.map(_measure, sorted(audio_sources)))
    log.info("quick-edit probe: %d sources, %d entries, %d loudness "
             "measurements  elapsed=%.2fs", len(facts),
             len(plan["entries"]), len(loudness), time.monotonic() - t0)

    manifest = build_edit_manifest(norm, plan, facts, loudness,
                                   canvas_w, canvas_h, fps_str, loudnorm_on)
    manifest_key = f"{out_prefix}/edit/edit_manifest.json"
    put_json(out_bucket, manifest_key, manifest)

    plan_key = f"{out_prefix}/_edit/edit_plan.json"
    put_json(out_bucket, plan_key, {
        "uuid": uuid,
        "quick_edit": norm,
        "target": {"width": canvas_w, "height": canvas_h, "fps": fps_str},
        "plan": plan,
        "sources": facts,
        "loudness": loudness,
        "loudnorm_on": bool(loudnorm_on),
    })

    iterators = [{
        "function_arn": fn_arn,
        "uuid": uuid,
        "item_id": i,
        "output": {"bucket": out_bucket, "prefix": out_prefix},
        "type": "edit",
        "plan_bucket": out_bucket,
        "plan_key": plan_key,
        "entry_index": i,
    } for i in range(len(plan["entries"]))]

    output_groups = parse_output_groups(
        output_groups_req or [{"group_name": "edit", "format": "mp4"}],
        has_video=True, has_audio=True)

    probe_payload = {
        "uuid": uuid,
        "quick_edit": True,
        "total_frames": plan["total_frames"],
        "duration_sec": plan["total_ms"] / 1000.0,
        "target": {"width": canvas_w, "height": canvas_h, "fps": fps_str},
        "sources": facts,
        "plan_s3_key": plan_key,
    }
    probe_key = f"{out_prefix}/probe.json"
    put_json(out_bucket, probe_key, probe_payload)

    return {
        **event,
        "status": "COMPLETED",
        "data": {
            **event.get("data", {}),
            "destination": {"bucket": out_bucket, "prefix": out_prefix},
            "output_groups": output_groups,
            "probe_s3_key": probe_key,
            "duration_sec": plan["total_ms"] / 1000.0,
            "has_video": True,
            "has_audio": True,
            "is_all_intra": False,
            "metadata_only": False,
            "output_keys": {"edit_manifest": manifest_key},
            "function_arn": fn_arn,
            "iterators": iterators,
        },
    }


def build_edit_manifest(norm, plan, facts, loudness, canvas_w, canvas_h,
                        fps_str, loudnorm_on) -> dict:
    """The evidence output: requested vs actual per cut, derived record
    timeline, crops in ratios AND pixels, normalizations, transitions,
    overlays, loudness, source ETags."""
    fps = parse_fps(fps_str)
    segments = []
    for rec in plan["segment_records"]:
        i = rec["segment_index"]
        seg = norm["segments"][i]
        src = seg["source_s3uri"]
        f = facts.get(src, {})
        norms = []
        if f.get("field_order") not in (None, "", "progressive", "unknown"):
            norms.append("deinterlace")
        if src != BLACK_SOURCE and f.get("has_video") and (
                f.get("width"), f.get("height")) != (canvas_w, canvas_h):
            norms.append("scale_pad")
        if seg["channel"] == "v" or (
                src != BLACK_SOURCE and not f.get("has_audio")):
            norms.append("silence_insert")
        entry = {
            "segment_index": i,
            "source_s3uri": src,
            "channel": seg["channel"],
            "requested_in_ms": seg["in_ms"],
            "requested_out_ms": seg["out_ms"],
            "src_in_frame": ms_to_frame(seg["in_ms"], fps),
            "src_out_frame": ms_to_frame(seg["out_ms"], fps),
            "record_in_ms": rec["record_in_ms"],
            "record_out_ms": rec["record_out_ms"],
            "record_in_frame": rec["record_in_frame"],
            "record_out_frame": rec["record_out_frame"],
            "transition_in": {k: v for k, v in seg["transition_in"].items()
                              if k != "audio_bed"},
            "normalizations": norms,
        }
        if seg.get("crop"):
            entry["crop_requested"] = seg["crop"]
            cw, ch, cx, cy = crop_pixels(seg["crop"], f["width"], f["height"])
            entry["crop_actual_px"] = {"x": cx, "y": cy,
                                       "width": cw, "height": ch}
        segments.append(entry)
    return {
        "target": {"width": canvas_w, "height": canvas_h, "fps": fps_str},
        "total_frames": plan["total_frames"],
        "total_ms": plan["total_ms"],
        "total_authored_ms": norm["total_authored_ms"],
        "crossfade_mode": "overlap-trim",
        "loudnorm_on": bool(loudnorm_on),
        "segments": segments,
        "overlays": [
            {k: o[k] for k in ("kind", "element_key", "source_s3uri",
                               "mode", "gain_db", "record_in_frame",
                               "record_out_frame", "in_ms", "out_ms")}
            for o in plan["overlays"]],
        "sources": {uri: {"etag": f.get("etag"),
                          "duration_sec": f.get("duration_sec"),
                          **({"measured_loudness": loudness[uri]}
                             if uri in loudness else {})}
                    for uri, f in facts.items()},
    }
