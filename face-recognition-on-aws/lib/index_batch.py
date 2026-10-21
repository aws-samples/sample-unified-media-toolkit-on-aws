"""
Batch face-indexing support (README.md §19): manifest parsing, magic-byte
format sniffing, tile preparation, and grid→source coordinate recalibration.

Input normalization: the gateway resolves prefix-mode into a GENERATED
manifest (basename → celebrity name); the state machine consumes ONLY
manifests (JSONL, one {"image_uri", "celebrity_name"} per line). The
generated manifest is the audit record of what the batch was asked to do.

Idempotency is keyed on (collection_id, source ETag) — content identity,
not filename (README.md §19): same ETag → skipped/already_indexed; same ETag
+ different name → rename via DDB (skipped/renamed); new ETag → index.

ExternalImageId is per-CALL (IndexFaces constraint): every face in a tiled
grid shares it, so it is a coarse batch-level tag "user:{batch_id}:{epoch}"
— per-face provenance (origin/source_key/source_etag) lives in DynamoDB.
"""

from __future__ import annotations

import json
import logging
import os
import time

from lib.input_schema import (
    BATCH_ACCEPTED_FORMATS,
    BATCH_TILE_DIM,
)

logger = logging.getLogger(__name__)


# ── Format sniffing (magic bytes — NEVER the file extension) ───────────────────

def sniff_format(data: bytes) -> str | None:
    """Return the canonical format name for the accepted list, or None.
    A .jpg-named HEIC (common via messaging apps) is detected as heif."""
    if len(data) < 16:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if data[4:8] == b"ftyp" and data[8:12] in (
            b"heic", b"heix", b"hevc", b"heim", b"heis", b"hevm", b"hevs",
            b"mif1", b"msf1", b"avif", b"avis"):
        return "heif"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def format_accepted(fmt: str | None) -> bool:
    return fmt in BATCH_ACCEPTED_FORMATS


# ── Manifest ───────────────────────────────────────────────────────────────────

def parse_manifest(text: str) -> list[dict]:
    """
    Parse a JSONL manifest into [{image_uri, celebrity_name}, ...].
    Fail-hard: a malformed line raises (a silently dropped line would be a
    face that never got indexed with no trace in the report).
    """
    lines = []
    for ln, raw in enumerate(text.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest line {ln}: invalid JSON: {exc}") from exc
        uri = (doc.get("image_uri") or "").strip()
        name = (doc.get("celebrity_name") or "").strip()
        if not uri.startswith("s3://"):
            raise ValueError(f"manifest line {ln}: image_uri must be s3:// "
                             f"(got {uri!r})")
        if not name:
            raise ValueError(f"manifest line {ln}: celebrity_name is required")
        lines.append({"image_uri": uri, "celebrity_name": name})
    if not lines:
        raise ValueError("manifest contains no entries")
    return lines


def create_batch_external_id(batch_id: str) -> str:
    """Batch-level ExternalImageId: user:{batch_id-hex32}:{epoch_ms}.
    Rekognition allows [a-zA-Z0-9_.\\-:]+ — the same charset as the auto
    path's '1:{hex32}:{epoch}'; 'user:' marks origin at the collection level."""
    hex32 = batch_id.replace("-", "")[:32].ljust(32, "0")
    return f"user:{hex32}:{int(time.time() * 1000)}"


# ── Tile preparation ───────────────────────────────────────────────────────────

def prepare_tile(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Fit-inside downscale one source image to the batch tile box
    (BATCH_TILE_DIM, never upscale), letterboxed onto a black canvas of
    exactly tile_w × tile_h (uniform cells are required by compose_grid).
    Returns (tile_jpeg, placement) where placement records how the source
    maps into the tile:
        {"scale": s, "off_x": px, "off_y": px, "src_w": W, "src_h": H}
    so a grid-detected face box can be recalibrated back to normalized
    source-image coordinates.
    """
    import pyvips

    from lib.grid_image import load_rgb

    tile_w, tile_h = BATCH_TILE_DIM
    img = load_rgb(image_bytes)
    W, H = img.width, img.height

    scale = min(tile_w / W, tile_h / H, 1.0)   # fit-inside, never upscale
    new_w = max(1, round(W * scale))
    new_h = max(1, round(H * scale))
    if scale < 1.0:
        img = img.thumbnail_image(new_w, height=new_h,
                                  size=pyvips.enums.Size.DOWN)
        # thumbnail may round differently; trust the actual result
        new_w, new_h = img.width, img.height
        scale = new_w / W

    off_x = (tile_w - new_w) // 2
    off_y = (tile_h - new_h) // 2
    canvas = img.embed(off_x, off_y, tile_w, tile_h, background=[0, 0, 0])
    return canvas.jpegsave_buffer(Q=90), {
        "scale": scale, "off_x": off_x, "off_y": off_y,
        "src_w": W, "src_h": H,
    }


def tile_box_to_source(tile_box: dict, placement: dict) -> dict:
    """
    Recalibrate a tile-relative face box ({l,t,w,h} ratios of the tile, as
    produced by map_face_to_cell) to normalized source-image coordinates,
    undoing the letterbox offsets and the fit-inside scale. Clamped to [0,1].
    """
    tile_w, tile_h = BATCH_TILE_DIM
    s = placement["scale"]
    px = tile_box["l"] * tile_w - placement["off_x"]
    py = tile_box["t"] * tile_h - placement["off_y"]
    pw = tile_box["w"] * tile_w
    ph = tile_box["h"] * tile_h
    x = max(0.0, px / s / placement["src_w"])
    y = max(0.0, py / s / placement["src_h"])
    w = min(1.0 - x, pw / s / placement["src_w"])
    h = min(1.0 - y, ph / s / placement["src_h"])
    return {"l": round(x, 6), "t": round(y, 6),
            "w": round(max(0.0, w), 6), "h": round(max(0.0, h), 6)}


def largest_face_per_cell(records: list[dict], n_tiles: int,
                          cols: int, rows: int) -> dict[int, dict]:
    """
    Phantom removal (README.md §19): per tile, ONLY the largest mapped face is
    valid; everything else in that tile is a phantom to be deleted. Returns
    {tile_index: {"record": IndexFaces FaceRecord, "box": tile-relative box}}.
    Records whose box maps to no cell (separator/bleed) are treated as
    phantoms too — the caller deletes every face id not in the result.
    """
    from lib.grid_image import map_face_to_cell

    tile_w, tile_h = BATCH_TILE_DIM
    best: dict[int, dict] = {}
    for rec in records:
        bbox = (rec.get("Face") or {}).get("BoundingBox") or {}
        mapped = map_face_to_cell(bbox, n_tiles, cols, rows, tile_w, tile_h)
        if mapped is None:
            continue
        idx, tile_box = mapped
        area = tile_box["w"] * tile_box["h"]
        if idx not in best or area > best[idx]["area"]:
            best[idx] = {"record": rec, "box": tile_box, "area": area}
    return best
