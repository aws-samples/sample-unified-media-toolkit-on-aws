"""
Quick-edit cut list: validation, frame mapping, record-timeline
derivation, and the Map plan builder.

Everything here is PURE — no boto3, no ffmpeg, no filesystem. The
contract and every rule below are the settled design in
README.md §14. The input is
the inline JSON cut list ("quick_edit" tool argument), never an EDL
file.

The model, in one paragraph: segments are SEQUENTIAL (order = output
timeline; the segment list IS the caller's duration budget). Transitions
ride the INCOMING segment (`transition_in`), types cut | fade |
crossfade. A crossfade consumes its duration from BOTH neighbours
(overlap-trim — output = Σ(out−in) − ΣD) and renders as a JUNCTION plan
entry; fade is a within-segment envelope; black is a SEGMENT with the
reserved source "black" (EDL's BL reel), so fade-to-black-with-hold is
composition, not a type. Audio overlays (replace | mix) anchor to the
derived record timeline (`at_record_ms`) or to a boundary
(`transition_in.audio_bed`, `lead_ms` before the transition starts);
they are sliced per plan entry so every Map iteration stays independent.
"""
from __future__ import annotations

import math
from fractions import Fraction

# ── Caps (README.md §14.1 "Caps" — deliberate ratchets) ───────────────────────

MAX_SEGMENTS         = 100
MAX_DISTINCT_SOURCES = 25        # overlay/bed sources COUNT toward this
MAX_AUDIO_OVERLAYS   = 25
MAX_OUTPUT_MS        = 3_600_000  # Σ(out−in) over segments, authored black included
MAX_TRANSITION_MS    = 5_000
GAIN_DB_MIN          = -60.0
GAIN_DB_MAX          = 12.0

BLACK_SOURCE         = "black"   # the reserved generator source (EDL "BL")
CHANNELS             = frozenset(("av", "v", "a"))
TRANSITION_TYPES     = frozenset(("cut", "fade", "crossfade"))
OVERLAY_MODES        = frozenset(("replace", "mix"))

MICRO_FADE_MS        = 20        # non-optional boundary audio micro-fade
BED_DEFAULT_FADE_MS  = 500       # gentler default envelope for beds


class CutListError(ValueError):
    """A refused cut list. The message names the condition (never a CLI)."""


def validate_crop(crop, where: str) -> dict:
    """NORMALIZED 0–1 reframe window — shared by quick_edit segments
    and frame_capture markers."""
    if not isinstance(crop, dict):
        raise CutListError(f"{where}: crop must be an object")
    vals = {}
    for fld in ("x", "y", "width", "height"):
        v = crop.get(fld)
        if not isinstance(v, (int, float)) or isinstance(v, bool) \
                or not (0.0 <= float(v) <= 1.0):
            raise CutListError(
                f"{where}: crop.{fld} must be a ratio in [0, 1], "
                f"got {v!r}")
        vals[fld] = float(v)
    if vals["width"] <= 0 or vals["height"] <= 0:
        raise CutListError(f"{where}: crop width/height must be > 0")
    if vals["x"] + vals["width"] > 1.0 or vals["y"] + vals["height"] > 1.0:
        raise CutListError(
            f"{where}: crop extends beyond the frame "
            f"(x+width ≤ 1 and y+height ≤ 1)")
    return vals


# ── Frame mapping ──────────────────────────────────────────────────────────────

def parse_fps(fps) -> Fraction:
    """'25', '30000/1001', 25, 29.97 → Fraction. Refuses non-positive."""
    try:
        if isinstance(fps, str) and "/" in fps:
            num, den = fps.split("/", 1)
            f = Fraction(int(num), int(den))
        else:
            f = Fraction(str(fps)).limit_denominator(100_000)
    except (ValueError, ZeroDivisionError) as exc:
        raise CutListError(f"target fps is not a frame rate: {fps!r}") from exc
    if f <= 0:
        raise CutListError(f"target fps must be positive: {fps!r}")
    return f


def ms_to_frame(ms: int, fps: Fraction) -> int:
    """
    The documented mapping rule: frame = round(ms / 1000 × fps),
    half-up, exact Fraction arithmetic (no float accumulation).
    In-points are inclusive, out-points exclusive — adjacent segments
    cut at the same ms map to the same frame and butt together with no
    duplicated or dropped frame.

    Scope note: this is for the CONTRACT's integer milliseconds, which
    Fraction(ms, 1000) represents exactly — stricter than going through
    float. A FLOAT timestamp (ffprobe keyframe/packet times) must go
    through video_command.sec_to_frame instead, THE timestamp→frame
    conversion in this stack (it snaps 6-dp print noise; the renderers
    use it on the keyframe-seek path).
    """
    return math.floor(Fraction(ms, 1000) * fps + Fraction(1, 2))


def frame_to_ms(frame: int, fps: Fraction) -> float:
    """Exact position of a frame index in ms (for manifests)."""
    return float(Fraction(frame * 1000, 1) / fps)


# ── Validation ─────────────────────────────────────────────────────────────────

def _require_int_ms(obj: dict, field: str, where: str, minimum: int = 0) -> int:
    v = obj.get(field)
    if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
        raise CutListError(
            f"{where}: {field} must be an integer ≥ {minimum}, got {v!r}")
    return v


def _validate_transition(tr: dict, where: str, first_segment: bool) -> dict:
    if not isinstance(tr, dict):
        raise CutListError(f"{where}: transition must be an object")
    ttype = tr.get("type")
    if ttype not in TRANSITION_TYPES:
        raise CutListError(
            f"{where}: unknown transition type {ttype!r} "
            f"(supported: {', '.join(sorted(TRANSITION_TYPES))})")
    out = {"type": ttype}
    if ttype in ("fade", "crossfade"):
        d = _require_int_ms(tr, "duration_ms", where, minimum=1)
        if d > MAX_TRANSITION_MS:
            raise CutListError(
                f"{where}: transition duration {d} ms exceeds the "
                f"{MAX_TRANSITION_MS} ms cap")
        out["duration_ms"] = d
    elif "duration_ms" in tr:
        raise CutListError(f"{where}: a cut has no duration_ms")
    if first_segment and ttype == "crossfade":
        raise CutListError(
            f"{where}: the first segment cannot arrive on a crossfade — "
            "there is no outgoing side (open from a black segment instead)")
    bed = tr.get("audio_bed")
    if bed is not None:
        if first_segment:
            raise CutListError(
                f"{where}: the first segment's transition carries no "
                "boundary, so it cannot carry an audio_bed")
        out["audio_bed"] = _validate_overlay(
            bed, f"{where}.audio_bed", anchor="lead_ms", is_bed=True)
    return out


def _validate_overlay(ov: dict, where: str, anchor: str, is_bed: bool) -> dict:
    if not isinstance(ov, dict):
        raise CutListError(f"{where}: must be an object")
    src = ov.get("source_s3uri")
    if not isinstance(src, str) or not src.startswith("s3://"):
        raise CutListError(
            f"{where}: source_s3uri must be an s3:// uri "
            f"(the {BLACK_SOURCE!r} generator has no audio), got {src!r}")
    in_ms  = _require_int_ms(ov, "in_ms", where)
    out_ms = _require_int_ms(ov, "out_ms", where, minimum=1)
    if out_ms <= in_ms:
        raise CutListError(
            f"{where}: empty range — out_ms {out_ms} ≤ in_ms {in_ms}")
    pos = _require_int_ms(ov, anchor, where)
    mode = ov.get("mode", "mix" if is_bed else None)
    if mode not in OVERLAY_MODES:
        raise CutListError(
            f"{where}: mode must be one of "
            f"{', '.join(sorted(OVERLAY_MODES))}, got {mode!r}")
    gain = ov.get("gain_db", 0.0)
    if not isinstance(gain, (int, float)) or isinstance(gain, bool) or \
            not (GAIN_DB_MIN <= float(gain) <= GAIN_DB_MAX):
        raise CutListError(
            f"{where}: gain_db must be within "
            f"[{GAIN_DB_MIN:g}, {GAIN_DB_MAX:g}] dB, got {gain!r}")
    dur = out_ms - in_ms
    default_fade = BED_DEFAULT_FADE_MS if is_bed else MICRO_FADE_MS
    fade_in  = ov.get("fade_in_ms",  min(default_fade, dur // 2))
    fade_out = ov.get("fade_out_ms", min(default_fade, dur // 2))
    for name, v in (("fade_in_ms", fade_in), ("fade_out_ms", fade_out)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise CutListError(f"{where}: {name} must be an integer ≥ 0")
    if fade_in + fade_out > dur:
        raise CutListError(
            f"{where}: fade_in_ms + fade_out_ms ({fade_in}+{fade_out}) "
            f"exceed the element's {dur} ms")
    return {"source_s3uri": src, "in_ms": in_ms, "out_ms": out_ms,
            anchor: pos, "mode": mode, "gain_db": float(gain),
            "fade_in_ms": fade_in, "fade_out_ms": fade_out}


def validate_quick_edit(qe: dict) -> dict:
    """
    Validate and normalize a quick_edit request (defaults applied,
    every refusal a named condition). Pure — stream-presence checks for
    channel "v"/"a" and duration checks against real sources belong to
    probe's re-validation, not here.
    """
    if not isinstance(qe, dict):
        raise CutListError("quick_edit must be an object")
    segments = qe.get("segments")
    if not isinstance(segments, list) or not segments:
        raise CutListError("quick_edit.segments must be a non-empty list")
    if len(segments) > MAX_SEGMENTS:
        raise CutListError(
            f"{len(segments)} segments exceed the {MAX_SEGMENTS}-segment cap")

    default_tr = _validate_transition(
        qe.get("transition", {"type": "cut"}), "transition",
        first_segment=False)
    if "audio_bed" in default_tr:
        raise CutListError(
            "the global default transition cannot carry an audio_bed — "
            "beds anchor to one boundary (use segments[i].transition_in)")

    norm_segments: list[dict] = []
    total_ms = 0
    sources: set[str] = set()
    for i, seg in enumerate(segments):
        where = f"segments[{i}]"
        if not isinstance(seg, dict):
            raise CutListError(f"{where}: must be an object")
        src = seg.get("source_s3uri")
        if src != BLACK_SOURCE and (
                not isinstance(src, str) or not src.startswith("s3://")):
            raise CutListError(
                f"{where}: source_s3uri must be an s3:// uri or the "
                f"reserved {BLACK_SOURCE!r} generator, got {src!r}")
        in_ms  = _require_int_ms(seg, "in_ms", where)
        out_ms = _require_int_ms(seg, "out_ms", where, minimum=1)
        if out_ms <= in_ms:
            raise CutListError(
                f"{where}: empty range — out_ms {out_ms} ≤ in_ms {in_ms}")
        crop = seg.get("crop")
        if crop is not None:
            if src == BLACK_SOURCE:
                raise CutListError(
                    f"{where}: crop does not apply to the "
                    f"{BLACK_SOURCE!r} generator")
            if seg.get("channel", "av") == "a":
                raise CutListError(
                    f"{where}: crop does not apply to channel 'a' — "
                    "an audio-only segment carries no picture")
            crop = validate_crop(crop, where)
        channel = seg.get("channel", "av")
        if channel not in CHANNELS:
            raise CutListError(
                f"{where}: channel must be one of "
                f"{', '.join(sorted(CHANNELS))}, got {channel!r}")
        if src == BLACK_SOURCE and "channel" in seg:
            raise CutListError(
                f"{where}: channel does not apply to the {BLACK_SOURCE!r} "
                "generator (black is already picture with silence)")
        tr = seg.get("transition_in")
        tr = (_validate_transition(tr, f"{where}.transition_in",
                                   first_segment=(i == 0))
              if tr is not None else
              (dict(default_tr) if i > 0 else {"type": "cut"}))
        if src != BLACK_SOURCE:
            sources.add(src)
        total_ms += out_ms - in_ms
        norm_segments.append({"source_s3uri": src, "in_ms": in_ms,
                              "out_ms": out_ms, "channel": channel,
                              "crop": crop, "transition_in": tr})
        if "audio_bed" in tr:
            sources.add(tr["audio_bed"]["source_s3uri"])

    if total_ms > MAX_OUTPUT_MS:
        raise CutListError(
            f"Σ(out−in) = {total_ms} ms exceeds the {MAX_OUTPUT_MS} ms "
            "(1 hour) output cap")

    overlays = qe.get("audio_overlays", [])
    if not isinstance(overlays, list):
        raise CutListError("audio_overlays must be a list")
    if len(overlays) > MAX_AUDIO_OVERLAYS:
        raise CutListError(
            f"{len(overlays)} audio overlays exceed the "
            f"{MAX_AUDIO_OVERLAYS}-overlay cap")
    norm_overlays = [
        _validate_overlay(ov, f"audio_overlays[{i}]",
                          anchor="at_record_ms", is_bed=False)
        for i, ov in enumerate(overlays)]
    for ov in norm_overlays:
        sources.add(ov["source_s3uri"])

    if len(sources) > MAX_DISTINCT_SOURCES:
        raise CutListError(
            f"{len(sources)} distinct sources exceed the "
            f"{MAX_DISTINCT_SOURCES}-source cap (overlay and bed sources "
            "count — each is probed)")

    target = qe.get("target") or {}
    if not isinstance(target, dict):
        raise CutListError("target must be an object")

    return {"segments": norm_segments, "audio_overlays": norm_overlays,
            "transition": default_tr, "target": dict(target),
            "total_authored_ms": total_ms,
            "distinct_sources": sorted(sources)}


# ── Record timeline + plan ─────────────────────────────────────────────────────

def _crossfade_frames(tr: dict, fps: Fraction, where: str) -> int:
    if tr["type"] != "crossfade":
        return 0
    d = ms_to_frame(tr["duration_ms"], fps)
    if d < 1:
        raise CutListError(
            f"{where}: crossfade duration {tr['duration_ms']} ms is below "
            f"one frame at the target rate — use a cut")
    return d


def build_plan(norm: dict, fps: Fraction) -> dict:
    """
    normalized quick_edit (validate_quick_edit output) + target fps →
    the Map plan and the manifest skeleton. All positions in TARGET-grid
    frames (ms kept alongside for the manifest).

    Entries (each one independent Map iteration):
      {"kind": "body", "segment_index", "source_s3uri", "channel",
       "src_in_frame", "src_out_frame", "record_in_frame",
       "record_out_frame", "fade_in_frames", "fade_out_frames",
       "audio_slices": [...]}
      {"kind": "junction", "left_index", "right_index", "d_frames",
       "left_src_in_frame", "right_src_in_frame", "record_in_frame",
       "record_out_frame", "audio_slices": [...]}
    Zero-length bodies (a segment fully consumed by its crossfades,
    e.g. a black segment with no hold) emit NO entry.
    """
    segs = norm["segments"]
    n = len(segs)

    # Per-segment frame counts and per-boundary crossfade frames.
    frames = []
    for i, s in enumerate(segs):
        f_in, f_out = ms_to_frame(s["in_ms"], fps), ms_to_frame(s["out_ms"], fps)
        if f_out - f_in < 1:
            raise CutListError(
                f"segments[{i}]: shorter than one frame at the target rate "
                f"({s['out_ms'] - s['in_ms']} ms)")
        frames.append((f_in, f_out))
    d_head = [0] * n   # crossfade frames consumed from segment i's head
    for i in range(1, n):
        d_head[i] = _crossfade_frames(segs[i]["transition_in"], fps,
                                      f"segments[{i}].transition_in")
    d_tail = d_head[1:] + [0]  # consumed from segment i's tail

    # Fade envelopes (within-segment, no duration effect).
    fade_in_f = [0] * n
    fade_out_f = [0] * n
    for i in range(n):
        tr = segs[i]["transition_in"]
        if tr["type"] == "fade":
            fade_in_f[i] = ms_to_frame(tr["duration_ms"], fps)
            if i > 0:
                fade_out_f[i - 1] = fade_in_f[i]

    # Budget: everything a segment's two boundaries demand fits inside it.
    for i, (f_in, f_out) in enumerate(frames):
        dur = f_out - f_in
        left  = d_head[i] or fade_in_f[i]
        right = d_tail[i] or fade_out_f[i]
        if left + right > dur:
            raise CutListError(
                f"segments[{i}]: boundary demands ({left}+{right} frames) "
                f"exceed the segment's {dur} frames")

    # Walk the record timeline.
    entries: list[dict] = []
    seg_records: list[dict] = []
    pos = 0
    for i, (f_in, f_out) in enumerate(frames):
        rec_in_incl_junction = pos  # head junction starts where cursor is
        if d_head[i]:
            left_f_in, left_f_out = frames[i - 1]
            entries.append({
                "kind": "junction", "left_index": i - 1, "right_index": i,
                "d_frames": d_head[i],
                "left_source_s3uri": segs[i - 1]["source_s3uri"],
                "left_src_in_frame": left_f_out - d_head[i],
                "left_crop": segs[i - 1]["crop"],
                "right_source_s3uri": segs[i]["source_s3uri"],
                "right_src_in_frame": f_in,
                "right_crop": segs[i]["crop"],
                "record_in_frame": pos,
                "record_out_frame": pos + d_head[i],
                "audio_slices": [],
            })
            pos += d_head[i]
        body = (f_out - f_in) - d_head[i] - d_tail[i]
        if body > 0:
            entries.append({
                "kind": "body", "segment_index": i,
                "source_s3uri": segs[i]["source_s3uri"],
                "channel": segs[i]["channel"],
                "crop": segs[i]["crop"],
                "src_in_frame": f_in + d_head[i],
                "src_out_frame": f_out - d_tail[i],
                "record_in_frame": pos,
                "record_out_frame": pos + body,
                "fade_in_frames": fade_in_f[i],
                "fade_out_frames": fade_out_f[i],
                # micro-fades guard CUT boundaries only; a junction or
                # fade edge is a mid-segment splice / already enveloped
                "micro_in": d_head[i] == 0 and fade_in_f[i] == 0,
                "micro_out": d_tail[i] == 0 and fade_out_f[i] == 0,
                "audio_slices": [],
            })
        pos += body
        seg_records.append({
            "segment_index": i,
            "record_in_frame": rec_in_incl_junction,
            "record_out_frame": pos + d_tail[i],
            "record_in_ms": frame_to_ms(rec_in_incl_junction, fps),
            "record_out_ms": frame_to_ms(pos + d_tail[i], fps),
        })
    total_frames = pos

    # Resolve overlays and beds to record frames, then slice per entry.
    resolved: list[dict] = []
    for idx, ov in enumerate(norm["audio_overlays"]):
        start = ms_to_frame(ov["at_record_ms"], fps)
        length = ms_to_frame(ov["at_record_ms"] + ov["out_ms"] - ov["in_ms"],
                             fps) - start
        resolved.append(dict(ov, kind="overlay", overlay_index=idx,
                             element_key=f"overlay:{idx}",
                             record_in_frame=start,
                             record_out_frame=start + length))
    for i, s in enumerate(segs):
        bed = s["transition_in"].get("audio_bed")
        if bed is None:
            continue
        tr = s["transition_in"]
        boundary = seg_records[i]["record_in_frame"]  # junction/cut point
        if tr["type"] == "fade":
            boundary -= fade_in_f[i]  # transition starts at the fade-out
        start = boundary - ms_to_frame(bed["lead_ms"], fps)
        if start < 0:
            raise CutListError(
                f"segments[{i}].transition_in.audio_bed: lead_ms "
                f"{bed['lead_ms']} starts before the output begins")
        length = ms_to_frame(bed["out_ms"] - bed["in_ms"], fps)
        resolved.append(dict(bed, kind="bed", boundary_index=i,
                             element_key=f"bed:{i}",
                             record_in_frame=start,
                             record_out_frame=start + length))

    for ov in resolved:
        if ov["record_out_frame"] > total_frames:
            label = (f"audio_overlays[{ov['overlay_index']}]"
                     if ov["kind"] == "overlay" else
                     f"segments[{ov['boundary_index']}].transition_in.audio_bed")
            raise CutListError(
                f"{label}: extends to frame {ov['record_out_frame']}, "
                f"beyond the {total_frames}-frame output")
    replaces = sorted((o["record_in_frame"], o["record_out_frame"])
                      for o in resolved if o["mode"] == "replace")
    for (a0, a1), (b0, b1) in zip(replaces, replaces[1:]):
        if b0 < a1:
            raise CutListError(
                f"two replace overlays overlap on the record timeline "
                f"(frames {b0}–{a1})")

    for entry in entries:
        e0, e1 = entry["record_in_frame"], entry["record_out_frame"]
        for ov in resolved:
            lo, hi = max(e0, ov["record_in_frame"]), min(e1, ov["record_out_frame"])
            if lo < hi:
                entry["audio_slices"].append({
                    "kind": ov["kind"], "mode": ov["mode"],
                    "element_key": ov["element_key"],
                    "source_s3uri": ov["source_s3uri"],
                    "gain_db": ov["gain_db"],
                    "record_in_frame": lo, "record_out_frame": hi,
                    "src_offset_frames":
                        ms_to_frame(ov["in_ms"], fps) + (lo - ov["record_in_frame"]),
                    "element_record_in_frame": ov["record_in_frame"],
                    "element_record_out_frame": ov["record_out_frame"],
                    "fade_in_ms": ov["fade_in_ms"],
                    "fade_out_ms": ov["fade_out_ms"],
                })

    return {"target_fps": str(fps), "total_frames": total_frames,
            "total_ms": frame_to_ms(total_frames, fps),
            "entries": entries, "segment_records": seg_records,
            "overlays": resolved}
