"""
Grid image compositing, layout solving, and face-to-frame cell mapping.

Grids are uploaded to S3 and passed to Rekognition by S3Object reference
(15 MB / 10000×10000 px bounds — NOT the 5 MB inline-bytes cap).

Layout (README.md §6): dimension-agnostic. cols × rows maximised such that
cols·cell_w <= 9800 and rows·cell_h <= 9800, capped at FRAMES_PER_GRID (20).
2 px black separator lines between cells break cross-cell face continuity;
mapped boxes crossing their cell boundary by > 15 % are rejected as bleed
artifacts.
"""

from __future__ import annotations

import logging

from lib.input_schema import (
    CELL_BLEED_MAX_OVERHANG,
    FRAMES_PER_GRID,
    GRID_CELL_SEPARATOR_PX,
    GRID_JPEG_QUALITY,
    GRID_MAX_EDGE_PX,
    GRID_MAX_JPEG_BYTES,
)

logger = logging.getLogger(__name__)

_SEP = GRID_CELL_SEPARATOR_PX


def solve_layout(cell_w: int, cell_h: int,
                 max_frames: int = FRAMES_PER_GRID) -> tuple[int, int]:
    """
    (cols, rows) maximising cols*rows <= max_frames within the pixel bound.
    Prefers the layout with the most cells; ties broken toward squarer grids.
    """
    max_cols = max(1, (GRID_MAX_EDGE_PX + _SEP) // (cell_w + _SEP))
    max_rows = max(1, (GRID_MAX_EDGE_PX + _SEP) // (cell_h + _SEP))
    best = (1, 1)
    for cols in range(1, max_cols + 1):
        rows = min(max_rows, max_frames // cols)
        if rows < 1:
            continue
        n, best_n = cols * min(rows, max_frames // cols), best[0] * best[1]
        if n > best_n or (n == best_n
                          and abs(cols - rows) < abs(best[0] - best[1])):
            best = (cols, rows)
    return best


def compose_grid(jpeg_frames: list[bytes], cols: int, rows: int,
                 cell_w: int, cell_h: int) -> bytes:
    """
    Composite up to cols*rows JPEG frames into one grid JPEG with black
    separators. Steps quality down (85 → 75 → 65) if the result exceeds
    GRID_MAX_JPEG_BYTES. Frames must all be cell_w × cell_h.
    """
    import pyvips

    grid_w = cols * cell_w + (cols - 1) * _SEP
    grid_h = rows * cell_h + (rows - 1) * _SEP
    canvas = pyvips.Image.black(grid_w, grid_h, bands=3)

    for i, jpg in enumerate(jpeg_frames):
        r, c = divmod(i, cols)
        img = pyvips.Image.jpegload_buffer(jpg)
        if img.bands == 1:
            img = img.colourspace("srgb")
        elif img.bands == 4:
            img = img[:3]
        if img.width != cell_w or img.height != cell_h:
            # Defensive: MP produces uniform frames, but never trust silently.
            img = img.thumbnail_image(cell_w, height=cell_h,
                                      size=pyvips.enums.Size.FORCE)
        canvas = canvas.insert(img, c * (cell_w + _SEP), r * (cell_h + _SEP))

    for quality in (GRID_JPEG_QUALITY, 75, 65):
        out = canvas.jpegsave_buffer(Q=quality)
        if len(out) <= GRID_MAX_JPEG_BYTES:
            return out
    raise RuntimeError(
        f"grid JPEG exceeds {GRID_MAX_JPEG_BYTES} bytes at Q=65 "
        f"({cols}x{rows} cells of {cell_w}x{cell_h})")


def map_face_to_cell(box: dict, n_frames: int, cols: int, rows: int,
                     cell_w: int, cell_h: int) -> tuple[int, dict] | None:
    """
    Map one Rekognition BoundingBox (grid-relative ratios) to
    (frame_index_in_grid, frame-relative box {l,t,w,h}).

    Returns None when the face is a bleed artifact: its extent crosses the
    owning cell's boundary by more than CELL_BLEED_MAX_OVERHANG of its own
    size, or its centre lands on a separator/empty cell.
    """
    grid_w = cols * cell_w + (cols - 1) * _SEP
    grid_h = rows * cell_h + (rows - 1) * _SEP

    # Absolute pixel box on the grid
    ax = box["Left"] * grid_w
    ay = box["Top"] * grid_h
    aw = box["Width"] * grid_w
    ah = box["Height"] * grid_h
    cx, cy = ax + aw / 2, ay + ah / 2

    col = int(cx // (cell_w + _SEP))
    row = int(cy // (cell_h + _SEP))
    if col >= cols or row >= rows:
        return None
    idx = row * cols + col
    if idx >= n_frames:
        return None  # centre in an empty trailing cell

    ox = col * (cell_w + _SEP)  # cell origin
    oy = row * (cell_h + _SEP)

    # Bleed rejection: overhang beyond the cell relative to face size
    over_l = max(0.0, ox - ax)
    over_t = max(0.0, oy - ay)
    over_r = max(0.0, (ax + aw) - (ox + cell_w))
    over_b = max(0.0, (ay + ah) - (oy + cell_h))
    if aw > 0 and (over_l + over_r) / aw > CELL_BLEED_MAX_OVERHANG:
        return None
    if ah > 0 and (over_t + over_b) / ah > CELL_BLEED_MAX_OVERHANG:
        return None

    # Clamp to the cell and convert to frame-relative ratios
    fx = max(0.0, ax - ox)
    fy = max(0.0, ay - oy)
    fw = min(cell_w - fx, aw)
    fh = min(cell_h - fy, ah)
    return idx, {
        "l": round(fx / cell_w, 6),
        "t": round(fy / cell_h, 6),
        "w": round(fw / cell_w, 6),
        "h": round(fh / cell_h, 6),
    }


def load_rgb(image_bytes: bytes):
    """
    Load ANY supported image as 3-band sRGB, honoring EXIF orientation.
    Required for Phase 2 user photos (the recognition path's MP frames are
    always EXIF-free 3-band JPEG, where this is a no-op):
      - autorotate: phone photos carry EXIF orientation; Rekognition sees
        the grid AFTER our compositing, so an unrotated portrait photo would
        be indexed sideways and likely miss detection.
      - CMYK (4-band without alpha) must be COLOURSPACE-converted — the old
        `img[:3]` band-slice interpreted C/M/Y planes as RGB, producing
        color-inverted tiles.
      - grayscale (1-band) promoted to srgb so downstream embed/insert with
        3-element backgrounds cannot fail.
    """
    import pyvips

    img = pyvips.Image.new_from_buffer(image_bytes, "")
    try:
        img = img.autorot()
    except pyvips.Error:
        pass  # loaders without orientation support (e.g. some TIFFs)
    if img.interpretation != "srgb" or img.bands != 3:
        img = img.colourspace("srgb")
    if img.bands > 3:
        img = img[:3]  # drop alpha (colourspace keeps it)
    return img


def crop_headshot(frame_jpeg: bytes, box: dict, margin: float,
                  out_w: int, out_h: int) -> bytes:
    """
    Crop face box + margin from a frame JPEG, resize to out_w × out_h
    WITHOUT distorting the face: the crop rectangle is expanded (centered)
    to the target aspect ratio before resizing, so faces are never squashed
    or stretched. Falls back to letterboxing on a black canvas only when the
    aspect-matched crop would run off the source frame (face very close to
    an edge).
    """
    import pyvips

    img = load_rgb(frame_jpeg)
    W, H = img.width, img.height
    bx, by = box["l"] * W, box["t"] * H
    bw, bh = box["w"] * W, box["h"] * H
    mx, my = bw * margin, bh * margin
    x0 = bx - mx
    y0 = by - my
    x1 = bx + bw + mx
    y1 = by + bh + my
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"degenerate crop box {box}")

    # Expand the shorter side (centered) to match the target aspect ratio.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = x1 - x0, y1 - y0
    target_ratio = out_w / out_h
    if cw / ch > target_ratio:
        ch = cw / target_ratio
    else:
        cw = ch * target_ratio
    x0, x1 = cx - cw / 2, cx + cw / 2
    y0, y1 = cy - ch / 2, cy + ch / 2

    # Clamp to the source frame. If clamping would re-introduce distortion
    # (crop ran off an edge), pad with black instead of resizing off-ratio.
    cx0, cy0 = max(0, int(x0)), max(0, int(y0))
    cx1, cy1 = min(W, int(x1)), min(H, int(y1))
    if cx0 == int(x0) and cy0 == int(y0) and cx1 == int(x1) and cy1 == int(y1):
        crop = img.crop(cx0, cy0, cx1 - cx0, cy1 - cy0)
    else:
        crop = img.crop(cx0, cy0, cx1 - cx0, cy1 - cy0)
        pad_l, pad_t = cx0 - int(x0), cy0 - int(y0)
        canvas_w, canvas_h = int(x1) - int(x0), int(y1) - int(y0)
        crop = crop.embed(pad_l, pad_t, canvas_w, canvas_h,
                          background=[0, 0, 0])

    crop = crop.thumbnail_image(out_w, height=out_h,
                                size=pyvips.enums.Size.FORCE)
    return crop.jpegsave_buffer(Q=90)


def downscale_fit(image_bytes: bytes, max_edge_px: int) -> bytes:
    """
    Fit-inside downscale of the WHOLE image to max_edge_px on the long edge —
    never upscale. Accepts any format the shipped libvips loads (JPEG, PNG,
    WebP, TIFF, HEIF, GIF — README.md §19); always emits JPEG. Used for the
    full_image derivative (whole source frame/photo, face located by
    normalized coord) and for the batch-index solo fallback.
    """
    import pyvips

    img = load_rgb(image_bytes)
    long_edge = max(img.width, img.height)
    if long_edge > max_edge_px:
        img = img.thumbnail_image(
            max(1, round(img.width * max_edge_px / long_edge)),
            height=max(1, round(img.height * max_edge_px / long_edge)),
            size=pyvips.enums.Size.DOWN)
    return img.jpegsave_buffer(Q=90)


def crop_face_native(frame_jpeg: bytes, box: dict, margin: float,
                     max_edge_px: int) -> bytes:
    """
    Crop face box + margin at NATIVE resolution — downscale only if the
    crop's long edge exceeds max_edge_px, never upscale. For the persistent
    IndexFaces image: a fixed-size resize threw away ~4x linear resolution
    on close-up faces before Rekognition ever saw them (the API takes 15 MB
    via S3Object — shrinking bought nothing). No aspect forcing, no padding:
    the natural crop rectangle is what the recognizer should see.
    """
    import pyvips

    img = load_rgb(frame_jpeg)
    W, H = img.width, img.height
    bx, by = box["l"] * W, box["t"] * H
    bw, bh = box["w"] * W, box["h"] * H
    mx, my = bw * margin, bh * margin
    x0 = max(0, int(bx - mx))
    y0 = max(0, int(by - my))
    x1 = min(W, int(bx + bw + mx))
    y1 = min(H, int(by + bh + my))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"degenerate crop box {box}")

    crop = img.crop(x0, y0, x1 - x0, y1 - y0)
    long_edge = max(crop.width, crop.height)
    if long_edge > max_edge_px:
        crop = crop.thumbnail_image(
            max(1, round(crop.width * max_edge_px / long_edge)),
            height=max(1, round(crop.height * max_edge_px / long_edge)),
            size=pyvips.enums.Size.DOWN)
    return crop.jpegsave_buffer(Q=90)


def crop_headshot_native(frame_jpeg: bytes, box: dict, margin: float,
                         ratio_w: int, ratio_h: int,
                         max_edge_px: int) -> bytes:
    """
    crop_headshot's aspect-matched rectangle at crop_face_native's resolution
    policy: expand the margined face box (centered) to ratio_w:ratio_h without
    distortion, keep NATIVE pixels, downscale only when the long edge exceeds
    max_edge_px — never upscale. For consumer-facing reference crops (the
    shot-metadata-enrichment face gallery, its PLAN §5): crop_headshot's
    fixed output size would upscale small faces into blur, and
    crop_face_native's free aspect makes uneven gallery cells. Pads with
    black only when the aspect-matched rect runs off the frame, same as
    crop_headshot.
    """
    import pyvips

    img = load_rgb(frame_jpeg)
    W, H = img.width, img.height
    bx, by = box["l"] * W, box["t"] * H
    bw, bh = box["w"] * W, box["h"] * H
    mx, my = bw * margin, bh * margin
    x0, y0 = bx - mx, by - my
    x1, y1 = bx + bw + mx, by + bh + my
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"degenerate crop box {box}")

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = x1 - x0, y1 - y0
    target_ratio = ratio_w / ratio_h
    if cw / ch > target_ratio:
        ch = cw / target_ratio
    else:
        cw = ch * target_ratio
    x0, x1 = cx - cw / 2, cx + cw / 2
    y0, y1 = cy - ch / 2, cy + ch / 2

    cx0, cy0 = max(0, int(x0)), max(0, int(y0))
    cx1, cy1 = min(W, int(x1)), min(H, int(y1))
    crop = img.crop(cx0, cy0, cx1 - cx0, cy1 - cy0)
    if not (cx0 == int(x0) and cy0 == int(y0)
            and cx1 == int(x1) and cy1 == int(y1)):
        crop = crop.embed(cx0 - int(x0), cy0 - int(y0),
                          int(x1) - int(x0), int(y1) - int(y0),
                          background=[0, 0, 0])

    long_edge = max(crop.width, crop.height)
    if long_edge > max_edge_px:
        crop = crop.thumbnail_image(
            max(1, round(crop.width * max_edge_px / long_edge)),
            height=max(1, round(crop.height * max_edge_px / long_edge)),
            size=pyvips.enums.Size.DOWN)
    return crop.jpegsave_buffer(Q=90)


def compose_tile_grid(jpeg_tiles: list[bytes], tile_w: int, tile_h: int,
                      cols: int) -> tuple[bytes, list[dict]]:
    """
    Tile fixed-size JPEGs (headshots / montage cells) row-major into one JPEG.
    Returns (jpeg_bytes, [grid-relative {l,t,w,h} per tile]) — the coords are
    what maps RecognizeCelebrities responses back to tiles.
    """
    import pyvips

    n = len(jpeg_tiles)
    rows = (n + cols - 1) // cols
    grid_w = cols * tile_w
    grid_h = rows * tile_h
    canvas = pyvips.Image.black(grid_w, grid_h, bands=3)
    coords = []
    for i, jpg in enumerate(jpeg_tiles):
        r, c = divmod(i, cols)
        img = pyvips.Image.jpegload_buffer(jpg)
        if img.bands == 4:
            img = img[:3]
        if img.width != tile_w or img.height != tile_h:
            img = img.thumbnail_image(tile_w, height=tile_h,
                                      size=pyvips.enums.Size.FORCE)
        canvas = canvas.insert(img, c * tile_w, r * tile_h)
        coords.append({
            "l": c * tile_w / grid_w,
            "t": r * tile_h / grid_h,
            "w": tile_w / grid_w,
            "h": tile_h / grid_h,
        })
    return canvas.jpegsave_buffer(Q=90), coords


def point_in_box(cx: float, cy: float, coord: dict) -> bool:
    return (coord["l"] <= cx <= coord["l"] + coord["w"]
            and coord["t"] <= cy <= coord["t"] + coord["h"])


def match_tile(face: dict, coords: list[dict]) -> int | None:
    """
    Internal centre-in-cell primitive: map ONE Rekognition face
    (BoundingBox in grid-relative coords) to the tile whose cell holds
    the bbox centre. Response-mapping consumers must go through
    dominant_per_tile — calling this per record re-ships the
    two-faces-per-tile contamination it exists to prevent.
    """
    box = face.get("BoundingBox") or {}
    if not box:
        return None
    cx = box["Left"] + box["Width"] / 2
    cy = box["Top"] + box["Height"] / 2
    for i, coord in enumerate(coords):
        if point_in_box(cx, cy, coord):
            return i
    return None


def dominant_per_tile(faces_with_payloads, coords: list[dict],
                      min_share: float = 0.0) -> dict:
    """
    tile_idx -> payload of the face that DOMINATES that tile (largest
    BoundingBox area). A tile is a headshot crop MADE around one tracked
    face, so the subject is by construction the largest face in the
    cell; any smaller face is crop BACKGROUND and must not wear the
    tile's identity. Without this rule a background face indexed into
    the clustering sandbox under the foreground track's id, and any
    match against it dragged that track into a foreign cluster
    (measured on a 2h news run: 65/2027 tracks carried a second sandbox
    face, every one splitting its track across two clusters); in
    RecognizeCelebrities the same shape let a background celebrity —
    a face on a studio monitor is the typical case — overwrite the
    subject's tag. Input: iterable of (face_dict, payload) pairs.

    `min_share` (fraction of the TILE's area, see
    MIN_SUBJECT_TILE_SHARE's derivation) floors the winner: dominance is
    judged only among faces Rekognition returned, so when the SUBJECT
    itself was dropped (quality filter, detection miss) the lone
    background face would win by default — a face too small to be the
    crop subject maps nothing instead.
    """
    best: dict[int, tuple[float, object]] = {}
    for face, payload in faces_with_payloads:
        idx = match_tile(face or {}, coords)
        if idx is None:
            continue
        box = (face or {}).get("BoundingBox") or {}
        area = float(box.get("Width") or 0) * float(box.get("Height") or 0)
        tile = coords[idx]
        if min_share and area < min_share * tile["w"] * tile["h"]:
            continue
        if idx not in best or area > best[idx][0]:
            best[idx] = (area, payload)
    return {idx: payload for idx, (_, payload) in best.items()}
