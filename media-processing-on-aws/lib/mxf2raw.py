from __future__ import annotations

import logging
import subprocess
# defusedxml: the parsed XML comes from the mxf2raw tool run over
# CUSTOMER media — treat it as untrusted (CSR scan 2026-10-08).
# defusedxml exports the parse functions only, so the element walker
# below takes its parameter untyped — importing the stdlib Element
# class would re-flag the module for the import alone.
import defusedxml.ElementTree as ET
from fractions import Fraction

from lib.s3_utils import presign

log = logging.getLogger(__name__)

MXF2RAW_TIMEOUT = 840 # 300

# is_mxf_format / MXF_FORMAT_NAMES live in lib/ffprobe.py — the predicate reads
# ffprobe's format_name, so it belongs with the probe wrapper. Import it from there.


# ── XML parser ─────────────────────────────────────────────────────────────────

def _xml_elem_to_dict(elem):
    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
    children = list(elem)
    text = (elem.text or "").strip()
    if not children and not elem.attrib:
        return text
    result = dict(elem.attrib)
    if text:
        result["_text"] = text
    child_dict: dict = {}
    for child in children:
        ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        cval = _xml_elem_to_dict(child)
        if ctag in child_dict:
            existing = child_dict[ctag]
            child_dict[ctag] = existing + [cval] if isinstance(existing, list) else [existing, cval]
        else:
            child_dict[ctag] = cval
    result.update(child_dict)
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def run_mxf2raw(bucket: str, key: str) -> dict | None:
    """Run mxf2raw --info --info-format xml via presigned URL. Returns parsed dict or None."""
    url = presign(bucket, key)
    try:
        result = subprocess.run(  # nosec B603
            ["mxf2raw", "--info", "--info-format", "xml", url],
            capture_output=True, text=True, timeout=MXF2RAW_TIMEOUT,
        )
    except FileNotFoundError:
        log.warning("mxf2raw not found — skipping BMX metadata")
        return None
    except subprocess.TimeoutExpired:
        log.warning("mxf2raw timed out after %ds", MXF2RAW_TIMEOUT)
        return None
    if not result.stdout.strip():
        log.warning("mxf2raw produced no output (rc=%d): %s",
                    result.returncode, result.stderr[:200])
        return None
    try:
        root = ET.fromstring(result.stdout)
        info = {root.tag.split("}")[-1]: _xml_elem_to_dict(root)}
        log.info("mxf2raw parsed successfully")
        return info
    except ET.ParseError as exc:
        log.warning("mxf2raw XML parse error: %s", exc)
        return None


def _pick(node: dict, *names) -> object:
    """First present key among names. bmx 1.6.0 emits snake_case; the CamelCase
    spellings are kept as a defensive fallback across BMX versions."""
    for n in names:
        if n in node:
            return node[n]
    return None


def _fps_from_edit_rate(er) -> float:
    """
    Parse a bmx edit_rate ("25/1", "30000/1001") → float.
    Returns 0.0 on anything unusable — notably "0/0", which Fraction raises on
    rather than returning zero.
    """
    s = str(er or "").strip()
    if not s:
        return 0.0
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_mxf2raw_info(info: dict) -> tuple[int, float, bool, str | None, str]:
    """
    Returns (frame_count, fps, drop_frame, essence_type, start_timecode).
    frame_count=-1 and fps=0.0 signal parse failure — callers that surface the
    result to a caller must check the sentinels (see build_mxf_block).

    Shapes verified against real bmx 1.6.0 output: the clip node is `clip`,
    timecodes live under `start_timecodes` as plain strings, and `duration.count`
    is "-1" when the header has no duration (a truncated or growing file).
    """
    try:
        bmx  = info.get("bmx", info)
        clip = _pick(bmx, "clip", "Clip") or {}

        # frame count — header may legitimately say -1 (no duration recorded)
        dur_node = _pick(clip, "duration", "Duration") or {}
        frame_count = -1
        if isinstance(dur_node, dict):
            try:
                frame_count = int(dur_node.get("count", -1))
            except (TypeError, ValueError):
                frame_count = -1

        # edit rate → fps
        fps = _fps_from_edit_rate(_pick(clip, "edit_rate", "EditRate"))

        # material timecode → drop_frame. Real output is a plain string; the dict
        # form is tolerated in case a future bmx adds attributes to the element.
        tc_node = _pick(clip, "start_timecodes", "StartTimecodes") or {}
        material_tc = ""
        if isinstance(tc_node, dict):
            mat = _pick(tc_node, "material", "Material") or ""
            if isinstance(mat, dict):
                material_tc = mat.get("_text") or mat.get("value") or ""
            elif isinstance(mat, str):
                material_tc = mat
        elif isinstance(tc_node, str):
            material_tc = tc_node
        drop_frame = ";" in material_tc

        # essence type — first track wins; picture tracks are emitted first
        essence_type: str | None = None
        tracks = _pick(clip, "tracks", "Tracks") or {}
        if isinstance(tracks, dict):
            track = _pick(tracks, "track", "Track") or {}
            if isinstance(track, list):
                track = track[0]
            if isinstance(track, dict):
                essence_type = _pick(track, "essence_type", "EssenceType")
        if not essence_type:
            ei = _pick(clip, "essence_info", "EssenceInfo") or {}
            if isinstance(ei, dict):
                essence_type = _pick(ei, "essence_type", "EssenceType")

        log.info("mxf2raw: frame_count=%d  fps=%.4f  drop_frame=%s  essence_type=%s  "
                 "start_timecode=%s",
                 frame_count, fps, drop_frame, essence_type, material_tc or "-")
        return frame_count, fps, drop_frame, essence_type, material_tc

    except Exception as exc:
        log.warning("Failed to parse mxf2raw info: %s", exc)
        return -1, 0.0, False, None, ""


def build_mxf_block(bucket: str, key: str) -> dict:
    """
    Run mxf2raw on an MXF source and return the `mxf` block for metadata.json.

    Raises RuntimeError when mxf2raw cannot produce real measurements. The caller
    only reaches this for a source ffprobe already identified as MXF, so a failure
    here is a genuine failure, not "not applicable" — and emitting frame_count=-1
    would hand a caller a plausible-looking wrong number.
    """
    info = run_mxf2raw(bucket, key)
    if info is None:
        raise RuntimeError(
            f"mxf2raw produced no usable output for MXF source s3://{bucket}/{key}")

    frame_count, fps, drop_frame, essence_type, start_timecode = parse_mxf2raw_info(info)

    # fps == 0.0 is the sentinel every parse failure lands on, so this one check
    # covers "the parse failed" as well as "the edit rate is unusable" — and it
    # must come first regardless, since duration_sec divides by it.
    if fps <= 0.0:
        raise RuntimeError(
            f"mxf2raw returned no usable edit rate for MXF source "
            f"s3://{bucket}/{key} (fps={fps}, frame_count={frame_count})")

    block = {
        "fps":          round(fps, 6),
        "drop_frame":   drop_frame,
        "essence_type": essence_type,
        "mxf_info":     info,
    }
    # frame_count == -1 is a real, valid MXF shape — an OP1A header that records no
    # duration (observed on Sony eVTR D10 material). Omit the field rather than
    # emitting the sentinel: absent means "the header did not say", and there is no
    # frame count to divide into a duration either.
    if frame_count >= 0:
        block["frame_count"]  = frame_count
        block["duration_sec"] = round(frame_count / fps, 6)
    else:
        log.warning("mxf2raw: header records no duration (count=-1) — "
                    "omitting frame_count and duration_sec from the mxf block")
    if start_timecode:
        block["start_timecode"] = start_timecode
    return block
