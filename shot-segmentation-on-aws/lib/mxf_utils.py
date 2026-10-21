# lib.mxf_utils.py

import logging
import subprocess
# defusedxml: the parsed XML comes from the mxf2raw tool run over
# CUSTOMER media — treat it as untrusted (CSR scan 2026-10-08).
# defusedxml exports the parse functions only, so the element walker
# below takes its parameter untyped — importing the stdlib Element
# class would re-flag the module for the import alone.
import defusedxml.ElementTree as ET

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from lib.misc_utils import fps_from_str
from lib.s3_utils import presign, put_json

MXF2RAW_TIMEOUT = 300          # 5 min — 50GB+ over HTTP can be slow
MXF_FORMAT_NAMES = ("mxf", "mxf_d10", "mxf_opatom")

# D10/IMX codec tags — definitively all-intra MPEG-2
D10_CODEC_TAGS = frozenset((
    "mx5p", "mx4p", "mx3p",
    "mx5n", "mx4n", "mx3n",
    "mx5f", "mx4f", "mx3f",
))

# Codecs that are always all-intra by definition
ALWAYS_INTRA_CODECS = frozenset((
    "prores", "dnxhd", "dnxhr", "mjpeg", "dvvideo",
    "ffv1", "huffyuv", "v210", "r10k", "cineform",
))

# Codecs that are always long-GOP
ALWAYS_LONGOP_CODECS = frozenset((
    "h264", "hevc", "h265", "vp9", "av1", "mpeg4",
    "vp8", "theora", "wmv1", "wmv2", "wmv3",
))

# =============================================================================
# mxf2raw BMX integration
# =============================================================================

def is_mxf(format_name: str) -> bool:
    return any(m in (format_name or "").lower() for m in MXF_FORMAT_NAMES)


def _xml_elem_to_dict(elem) -> dict | str:
    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
    children = list(elem)
    attribs = dict(elem.attrib)
    text = (elem.text or "").strip()

    if not children and not attribs:
        return text

    result: dict = {}
    result.update(attribs)
    if text:
        result["_text"] = text

    child_dict: dict = {}
    for child in children:
        child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        child_val = _xml_elem_to_dict(child)
        if child_tag in child_dict:
            existing = child_dict[child_tag]
            if isinstance(existing, list):
                existing.append(child_val)
            else:
                child_dict[child_tag] = [existing, child_val]
        else:
            child_dict[child_tag] = child_val

    result.update(child_dict)
    return result


def _xml_to_dict(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    return {root.tag.split("}")[-1]: _xml_elem_to_dict(root)}


def run_mxf2raw(bucket: str, key: str) -> dict | None:
    """
    Run mxf2raw --info --info-format xml on the MXF file via pre-signed URL.
    Returns parsed dict on success, None if mxf2raw unavailable or fails.
    """
    url = presign(bucket, key)
    try:
        result = subprocess.run(  # nosec B603
            ["mxf2raw", "--info", "--info-format", "xml", url],
            capture_output=True,
            text=True,
            timeout=MXF2RAW_TIMEOUT,
        )
    except FileNotFoundError:
        log.warning("mxf2raw not found — skipping BMX metadata")
        return None
    except subprocess.TimeoutExpired:
        log.warning(
            "mxf2raw timed out after %ds — skipping BMX metadata",
            MXF2RAW_TIMEOUT,
        )
        return None

    if not result.stdout.strip():
        log.warning(
            "mxf2raw produced no output (rc=%d): %s",
            result.returncode, result.stderr[:200],
        )
        return None

    try:
        info = _xml_to_dict(result.stdout)
        log.info("mxf2raw parsed successfully")
        return info
    except ET.ParseError as exc:
        log.warning("mxf2raw XML parse error: %s", exc)
        return None


def write_mxf_info_to_s3(bucket: str, prefix: str, mxf_info: dict) -> str:
    key = f"{prefix}/mxf_info.json"
    put_json(bucket, key, mxf_info)
    log.info("mxf_info.json written: s3://%s/%s", bucket, key)
    return key


def parse_mxf2raw_info(info: dict) -> tuple[int, float, bool, str | None]:
    try:
        bmx = info.get("bmx", info)
        clip = bmx.get("Clip") or bmx.get("clip") or {}

        dur_node = clip.get("duration") or clip.get("Duration") or {}
        if isinstance(dur_node, str):
            frame_count = -1
        else:
            try:
                frame_count = int(dur_node.get("count", "-1"))
            except (ValueError, TypeError):
                frame_count = -1

        edit_rate_str = clip.get("edit_rate") or clip.get("EditRate") or "0/0"
        fps = fps_from_str(str(edit_rate_str))

        tc_node = clip.get("StartTimecodes") or {}
        material_tc = ""
        if isinstance(tc_node, dict):
            mat = tc_node.get("material") or tc_node.get("Material") or {}
            if isinstance(mat, dict):
                material_tc = mat.get("_text") or mat.get("value") or ""
            elif isinstance(mat, str):
                material_tc = mat
        drop_frame = ";" in material_tc

        essence_type: str | None = None
        tracks_node = clip.get("Tracks") or clip.get("tracks") or {}
        if isinstance(tracks_node, dict):
            track = tracks_node.get("Track") or tracks_node.get("track") or {}
            if isinstance(track, list):
                track = track[0]
            if isinstance(track, dict):
                essence_type = (
                    track.get("essence_type")
                    or track.get("EssenceType")
                    or None
                )

        if not essence_type:
            essence_info = clip.get(
                "essence_info") or clip.get("EssenceInfo") or {}
            if isinstance(essence_info, dict):
                essence_type = (
                    essence_info.get("essence_type")
                    or essence_info.get("EssenceType")
                    or None
                )

        log.info(
            "mxf2raw: frame_count=%d  fps=%.4f  drop_frame=%s  essence_type=%s",
            frame_count, fps, drop_frame, essence_type,
        )
        return frame_count, fps, drop_frame, essence_type

    except Exception as exc:
        log.warning("Failed to parse mxf2raw info: %s", exc)
        return -1, 0.0, False, None
