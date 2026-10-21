"""
Quick-edit renderer: one plan entry → one ffmpeg invocation → the same
segment_video/*.ts + segment_audio/*.flac pair uniform chunks produce,
so StateMergeChunks concatenates edit entries unchanged.

Frame accuracy follows the PROVEN chunk-transcode machinery, not seek
precision (transcode_helper's rule: "Never make the frame selection
depend on seek precision instead of the trim count"):

- VIDEO: all-intra sources seek the target frame directly; inter-frame
  (or unknown) sources seek the PRIOR KEYFRAME (find_prior_keyframe)
  and the exact frame is then picked by COUNTING —
  trim=start_frame={preroll} on the SOURCE grid, before any scaling —
  with keyframe_frame derived via sec_to_frame (THE timestamp→frame
  conversion in this stack). Only after the source-grid cut does the
  chain normalize (yadif → crop → fit → pad → setsar → fps → format)
  and pin the TARGET-grid count (trim=end_frame=n).
- AUDIO: the house conform from probe_helper.build_afilters —
  aresample=48000:async=1 … apad … s16 — then SAMPLE-based atrim
  against a 3 s pre-roll seek (apad guarantees the end sample exists:
  audio running short near an out point pads with silence rather than
  shortening the chunk and desyncing the concat). adelay uses the
  sample suffix, never rounded milliseconds.

Composition rules (README.md §14.2–§14.3, pinned): identical mezzanine
params on every entry; loudnorm pass 2 uses the SOURCE's measured gain
(a junction side reuses its segment's gain); programme envelopes before
replace-mutes before the overlay mix; micro-fades on CUT edges only; a
caller fade crossing a slice boundary truncates there (v1).
"""
from __future__ import annotations

import math
import os
import shlex
from fractions import Fraction

from lib.cut_list import BLACK_SOURCE, parse_fps
from lib.edit_filters import (
    MICRO_FADE_SEC, build_audio_conform, build_fade_vfilters,
    build_junction_filters, build_normalize_vfilters, gain_filter,
    loudness_gain_filter,
)
from lib.ffprobe import find_prior_keyframe
from lib.proc_utils import run
from lib.s3_utils import get_json, presign
from lib.video_command import compute_bitrate, sec_to_frame
from states.chunk_transcode.transcode_helper import (
    FFMPEG, THREADS, ValidationException, validate_duration,
    validate_frames,
)

GOP_SECONDS = 2.0      # mezzanine GOP — matches the proxy default
MOTION = 1.6
SAMPLE_RATE = 48000
AUDIO_PREROLL_SEC = 3.0   # the audio-only chunk path's pre-roll


def _sec(frames: int, fps: Fraction) -> float:
    return float(Fraction(frames) / fps)


def _fmt(x: float) -> str:
    return f"{x:.6f}"


class _Graph:
    """Accumulates inputs and filter_complex lines. Flat, not a DSL —
    just bookkeeping for input indexes and label names."""

    def __init__(self):
        self.inputs: list[list[str]] = []
        self.lines: list[str] = []

    def add_input(self, args: list[str]) -> int:
        self.inputs.append(args)
        return len(self.inputs) - 1


def video_seek_plan(facts: dict, target_frame_in: int, n_target: int,
                    target_fps: Fraction) -> tuple[float | None, int, int]:
    """
    (seek_ss, preroll_frames, n_src_frames) for one video side.

    target_frame_in is on the TARGET grid; the cut lands on the nearest
    SOURCE frame (the plan's residual mixed-framerate case — manifest
    reports the delta). The source window covers the target span plus a
    2-frame margin; the target-grid trim downstream pins the exact
    output count.
    """
    src_fps = parse_fps(facts["fps"])
    t_in = Fraction(target_frame_in) / target_fps          # exact seconds
    f0 = math.floor(t_in * src_fps + Fraction(1, 2))       # nearest src frame
    n_src = math.ceil(Fraction(n_target) / target_fps * src_fps) + 2

    if f0 <= 0:
        return None, 0, n_src
    if facts.get("is_all_intra"):
        # every frame is a keyframe — direct seek is exact, no preroll
        # (the chunk path's all-intra convention)
        return float(Fraction(f0) / src_fps), 0, n_src

    kf = find_prior_keyframe(facts["bucket"], facts["key"],
                             float(Fraction(f0) / src_fps))
    try:
        kf_ts = float(kf.get("best_effort_timestamp_time"))
    except (TypeError, ValueError):
        raise ValidationException(
            "find_prior_keyframe returned non-numeric "
            f"best_effort_timestamp_time={kf.get('best_effort_timestamp_time')!r}")
    # round, not floor — a keyframe timestamp IS a frame boundary and
    # ffprobe prints it to 6 dp (the sec_to_frame contract)
    keyframe_frame = round(sec_to_frame(kf_ts, src_fps))
    preroll = f0 - keyframe_frame
    if preroll < 0:
        raise ValidationException(
            f"preroll_frames={preroll} < 0: keyframe_frame={keyframe_frame} "
            f"exceeds target source frame {f0}")
    return kf_ts, preroll, n_src


def audio_seek_plan(target_frame_in: int, n_target: int,
                    target_fps: Fraction) -> tuple[float | None, int, int]:
    """(seek_ss, start_sample, n_samples) for one audio element/side —
    the audio-only chunk path's recipe: pre-roll ~3 s, then
    sample-exact atrim relative to the seek point."""
    t_in = Fraction(target_frame_in) / target_fps
    n_samples = round(Fraction(n_target) / target_fps * SAMPLE_RATE)
    if t_in <= 0:
        return None, 0, n_samples
    seek = max(0.0, float(t_in) - AUDIO_PREROLL_SEC)
    seek_str = _fmt(seek)
    # offsets computed against the PRINTED seek value — that string is
    # exactly what ffmpeg parses, so the zero point matches to the sample
    start_sample = math.floor((t_in - Fraction(seek_str)) * SAMPLE_RATE)
    return float(seek_str), start_sample, n_samples


def _video_chain(g: _Graph, entry: dict, plan_doc: dict,
                 urls: dict, fps: Fraction) -> None:
    """Emit the [vts] video chain for a body or junction entry."""
    tgt = plan_doc["target"]
    n = entry["record_out_frame"] - entry["record_in_frame"]
    pin = f"trim=start_frame=0:end_frame={n},setpts=PTS-STARTPTS"

    def side_input(uri: str, src_in_frame: int, frames: int,
                   channel: str, crop: dict | None) -> tuple[int, str]:
        """Returns (input index, pre-pin chain) for one video side."""
        if uri == BLACK_SOURCE or channel == "a":
            idx = g.add_input(
                ["-f", "lavfi", "-i",
                 f"color=c=black:s={tgt['width']}x{tgt['height']}"
                 f":r={tgt['fps']}:d={_fmt(_sec(frames, fps))}"])
            return idx, ""
        f = plan_doc["sources"][uri]
        seek, preroll, n_src = video_seek_plan(f, src_in_frame, frames, fps)
        args = (["-ss", _fmt(seek)] if seek is not None else []) + \
               ["-i", urls[uri]]
        idx = g.add_input(args)
        # source-grid cut FIRST (the proven order: trim → setpts →
        # yadif → crop → scale …), then the normalize chain
        src_cut = (f"trim=start_frame={preroll}"
                   f":end_frame={preroll + n_src},setpts=PTS-STARTPTS")
        norm = ",".join(build_normalize_vfilters(
            f["width"], f["height"], f.get("field_order"),
            tgt["width"], tgt["height"], tgt["fps"], crop=crop))
        return idx, f"{src_cut},{norm}"

    if entry["kind"] == "junction":
        # each side's CHANNEL comes from its segment — an "a"-only
        # segment's junction share is black picture, same as its body
        segs = plan_doc["quick_edit"]["segments"]
        d = entry["d_frames"]
        li, lnorm = side_input(entry["left_source_s3uri"],
                               entry["left_src_in_frame"], d,
                               segs[entry["left_index"]]["channel"],
                               entry.get("left_crop"))
        ri, rnorm = side_input(entry["right_source_s3uri"],
                               entry["right_src_in_frame"], d,
                               segs[entry["right_index"]]["channel"],
                               entry.get("right_crop"))
        xfade, _ = build_junction_filters(d, float(fps))
        pin_d = f"trim=start_frame=0:end_frame={d},setpts=PTS-STARTPTS"
        g.lines.append(f"[{li}:v]{lnorm + ',' if lnorm else ''}{pin_d}[vl]")
        g.lines.append(f"[{ri}:v]{rnorm + ',' if rnorm else ''}{pin_d}[vr]")
        g.lines.append(f"[vl][vr]{xfade},{pin}[vts]")
        return

    idx, norm = side_input(entry["source_s3uri"], entry["src_in_frame"],
                           n, entry["channel"], entry.get("crop"))
    fades = build_fade_vfilters(entry["fade_in_frames"],
                                entry["fade_out_frames"], n)
    chain = ",".join(x for x in ([norm] if norm else []) + [pin] + fades)
    g.lines.append(f"[{idx}:v]{chain}[vts]")


def _element_afades(sl: dict, fps: Fraction) -> list[str]:
    """Slice-local envelope: an element fade applies in the slice that
    holds the element's start/end; crossing fades truncate at the
    boundary (v1 rule in the module docstring)."""
    slice_len = _sec(sl["record_out_frame"] - sl["record_in_frame"], fps)
    head_off = _sec(sl["record_in_frame"] - sl["element_record_in_frame"], fps)
    tail_gap = _sec(sl["element_record_out_frame"] - sl["record_out_frame"], fps)
    fades = []
    f_in = max(sl["fade_in_ms"] / 1000, MICRO_FADE_SEC)
    f_out = max(sl["fade_out_ms"] / 1000, MICRO_FADE_SEC)
    if head_off == 0:
        fades.append(f"afade=t=in:st=0:d={min(f_in, slice_len):g}")
    if tail_gap == 0:
        st = max(0.0, slice_len - f_out)
        fades.append(f"afade=t=out:st={st:g}:d={min(f_out, slice_len):g}")
    return fades


def _audio_chain(g: _Graph, entry: dict, plan_doc: dict,
                 urls: dict, fps: Fraction) -> None:
    """Emit the [aout] audio chain: base programme + overlay slices."""
    loud = plan_doc.get("loudness") or {}
    loudnorm_on = plan_doc.get("loudnorm_on", False)
    n = entry["record_out_frame"] - entry["record_in_frame"]
    dur = _sec(n, fps)
    n_samples_entry = round(Fraction(n) / fps * SAMPLE_RATE)

    def src_loudnorm(uri: str) -> str | None:
        # per-SOURCE constant gain (sampled window at probe): corrects
        # master mismatch between sources, preserves intra-source
        # dynamics. volume=, never loudnorm — see loudness_gain_filter
        m = loud.get(uri)
        return loudness_gain_filter(m) if (loudnorm_on and m) else None

    def base_side(uri: str, channel: str, has_audio: bool,
                  src_in_frame: int, frames: int,
                  segment_index: int) -> str:
        """One side's conformed, loudnormed audio → returns its label."""
        label = f"ab{len(g.lines)}"
        if uri == BLACK_SOURCE or channel == "v" or not has_audio:
            n_samp = round(Fraction(frames) / fps * SAMPLE_RATE)
            idx = g.add_input(["-f", "lavfi", "-i",
                               "anullsrc=r=48000:cl=stereo"])
            g.lines.append(
                f"[{idx}:a]aformat=sample_fmts=s16:channel_layouts=stereo,"
                f"atrim=end_sample={n_samp},asetpts=PTS-STARTPTS[{label}]")
            return label
        seek, start_sample, n_samp = audio_seek_plan(src_in_frame, frames, fps)
        args = (["-ss", _fmt(seek)] if seek is not None else []) + \
               ["-i", urls[uri]]
        idx = g.add_input(args)
        chain = build_audio_conform(loudnorm=src_loudnorm(uri))
        g.lines.append(
            f"[{idx}:a]{chain},"
            f"atrim=start_sample={start_sample}"
            f":end_sample={start_sample + n_samp},"
            f"asetpts=PTS-STARTPTS[{label}]")
        return label

    def has_audio(uri: str) -> bool:
        return uri == BLACK_SOURCE or \
            plan_doc["sources"].get(uri, {}).get("has_audio", False)

    if entry["kind"] == "junction":
        segs = plan_doc["quick_edit"]["segments"]
        d = entry["d_frames"]
        left = base_side(entry["left_source_s3uri"],
                         segs[entry["left_index"]]["channel"],
                         has_audio(entry["left_source_s3uri"]),
                         entry["left_src_in_frame"], d, entry["left_index"])
        right = base_side(entry["right_source_s3uri"],
                          segs[entry["right_index"]]["channel"],
                          has_audio(entry["right_source_s3uri"]),
                          entry["right_src_in_frame"], d,
                          entry["right_index"])
        _, acrossfade = build_junction_filters(d, float(fps))
        base_parts = [acrossfade,
                      f"atrim=end_sample={n_samples_entry}",
                      "asetpts=PTS-STARTPTS"]
        base_in = f"[{left}][{right}]"
    else:
        label = base_side(entry["source_s3uri"], entry["channel"],
                          has_audio(entry["source_s3uri"]),
                          entry["src_in_frame"], n, entry["segment_index"])
        env = []
        if entry["fade_in_frames"]:
            env.append(f"afade=t=in:st=0:"
                       f"d={_sec(entry['fade_in_frames'], fps):g}")
        elif entry["micro_in"]:
            env.append(f"afade=t=in:st=0:d={MICRO_FADE_SEC:g}")
        if entry["fade_out_frames"]:
            d_out = _sec(entry["fade_out_frames"], fps)
            env.append(f"afade=t=out:st={max(0.0, dur - d_out):g}:d={d_out:g}")
        elif entry["micro_out"]:
            env.append(f"afade=t=out:st={max(0.0, dur - MICRO_FADE_SEC):g}"
                       f":d={MICRO_FADE_SEC:g}")
        base_parts = env
        base_in = f"[{label}]"

    # replace-span mutes on the base, in entry-local time
    for sl in entry["audio_slices"]:
        if sl["mode"] == "replace":
            lo = _sec(sl["record_in_frame"] - entry["record_in_frame"], fps)
            hi = _sec(sl["record_out_frame"] - entry["record_in_frame"], fps)
            base_parts.append(
                f"volume=enable='between(t,{lo:g},{hi:g})':volume=0")
    base_chain = ",".join(base_parts) if base_parts else "anull"
    g.lines.append(f"{base_in}{base_chain}[abase]")

    # overlay/bed slices: conform (house chain incl. apad), element
    # loudnorm, gain, SAMPLE-trim, slice envelope, sample-exact delay
    mix_labels = ["[abase]"]
    for k, sl in enumerate(entry["audio_slices"]):
        seek, start_sample, n_samp = audio_seek_plan(
            sl["src_offset_frames"],
            sl["record_out_frame"] - sl["record_in_frame"], fps)
        args = (["-ss", _fmt(seek)] if seek is not None else []) + \
               ["-i", urls[sl["source_s3uri"]]]
        idx = g.add_input(args)
        chain = build_audio_conform(
            loudnorm=src_loudnorm(sl["source_s3uri"]),
            gain=gain_filter(sl["gain_db"]))
        parts = [chain,
                 f"atrim=start_sample={start_sample}"
                 f":end_sample={start_sample + n_samp}",
                 "asetpts=PTS-STARTPTS"]
        parts += _element_afades(sl, fps)
        delay_samples = round(
            Fraction(sl["record_in_frame"] - entry["record_in_frame"])
            / fps * SAMPLE_RATE)
        parts.append(f"adelay={delay_samples}S:all=1")
        g.lines.append(f"[{idx}:a]{','.join(parts)}[ov{k}]")
        mix_labels.append(f"[ov{k}]")

    # TP safety: the per-source gain was clamped against the sampled
    # WINDOW's true peak, so a hotter-than-window segment could clip
    # after an upward gain — the ad_mix precedent ends with alimiter
    tail = ",alimiter=limit=0.98" if loudnorm_on else ""
    if len(mix_labels) > 1:
        g.lines.append(f"{''.join(mix_labels)}"
                       f"amix=inputs={len(mix_labels)}"
                       f":duration=first:normalize=0,"
                       f"atrim=end_sample={n_samples_entry}{tail}[aout]")
    else:
        g.lines.append(f"[abase]{tail.lstrip(',') or 'anull'}[aout]")


def build_entry_command(entry: dict, plan_doc: dict, item_id: int,
                        tmp_dir: str, urls: dict) -> tuple[list, list[dict]]:
    """One entry → (ffmpeg argv, chunk_outputs with local names)."""
    tgt = plan_doc["target"]
    fps = Fraction(tgt["fps"])
    g = _Graph()
    _video_chain(g, entry, plan_doc, urls, fps)
    _audio_chain(g, entry, plan_doc, urls, fps)

    bitrate = compute_bitrate(tgt["width"], tgt["height"], fps,
                              GOP_SECONDS, MOTION)
    gopsize = round(float(fps) * GOP_SECONDS)
    outputs = [
        {"s3_prefix": "segment_video", "s3_name": f"video_{item_id}.ts",
         "map": f"[vts] -c:v libopenh264 -b:v {bitrate} -g {gopsize} "
                f"-f mpegts"},
        {"s3_prefix": "segment_audio", "s3_name": f"audio_{item_id}.flac",
         "map": "[aout] -c:a flac -ar 48000 -ac 2"},
    ]
    args = [FFMPEG, "-threads", str(THREADS), "-v", "error", "-y"]
    for inp in g.inputs:
        args += inp
    args += ["-filter_complex", ";".join(g.lines)]
    for output in outputs:
        local = os.path.join(tmp_dir, output["s3_name"])
        args += ["-map"] + shlex.split(output["map"]) + [local]
    return args, outputs


def render_edit(iterator: dict, tmp_dir: str) -> list[dict]:
    """Fetch the plan, render this iterator's entry, validate, return
    chunk_outputs with local_path (the handler uploads them)."""
    plan_doc = get_json(iterator["plan_bucket"], iterator["plan_key"])
    entry = plan_doc["plan"]["entries"][iterator["entry_index"]]
    fps = Fraction(plan_doc["target"]["fps"])

    urls = {}
    for uri, f in plan_doc["sources"].items():
        urls[uri] = presign(f["bucket"], f["key"])

    cmd, outputs = build_entry_command(entry, plan_doc,
                                       iterator["item_id"], tmp_dir, urls)
    run(cmd, timeout=840)

    n = entry["record_out_frame"] - entry["record_in_frame"]
    results = []
    for output in outputs:
        local_path = os.path.join(tmp_dir, output["s3_name"])
        if output["s3_name"].endswith(".ts"):
            validate_frames(local_path, n)
        else:
            validate_duration(local_path, str(_sec(n, fps)))
        results.append({**output, "local_path": local_path})
    return results
