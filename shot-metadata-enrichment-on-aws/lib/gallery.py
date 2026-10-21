"""
The reference-face gallery image (README.md §5, format settled by the
2026-08-19 probes): ONE image per window — 3:4 cells at 192x256, 3px white
vertical dividers between cells only, left-to-right order carried by the
text label. pyvips, not PIL (user decision; face-recognition's imaging
stack).

Cell source is face-recognition's `rep_face_key` crop — already 3:4 at
native resolution (<=512 long edge, its PLAN §22), so composition here is
a resize to the cell, never a crop. 96x128-scale sources measurably fail
re-identification (GOTCHAS), which is why the cell size is pinned and the
montage (`thumbnail_key`) is never accepted.
"""
from __future__ import annotations


def compose_gallery(face_jpegs: list, *, cell_w: int, cell_h: int,
                    divider_px: int) -> bytes:
    """One row of cells from JPEG bytes, in the given order. JPEG out."""
    import pyvips

    if not face_jpegs:
        raise ValueError("compose_gallery needs at least one face")
    cells = []
    for raw in face_jpegs:
        img = pyvips.Image.jpegload_buffer(raw, access="sequential")
        if img.bands > 3:
            img = img[:3]
        # rep_face crops are 3:4 by contract; FORCE to the exact cell so a
        # rounding-off source (383x512) cannot ripple the row height.
        cells.append(img.thumbnail_image(
            cell_w, height=cell_h, size=pyvips.enums.Size.FORCE))
    n = len(cells)
    row = pyvips.Image.black(n * cell_w + (n - 1) * divider_px, cell_h,
                             bands=3).new_from_image([255, 255, 255])
    for i, cell in enumerate(cells):
        row = row.insert(cell, i * (cell_w + divider_px), 0)
    return row.jpegsave_buffer(Q=90)


def gallery_label(names: list) -> str:
    """The text block that precedes the gallery image — the order contract
    the prompt's Reference-faces section reads."""
    return (f"Reference face gallery — ONE image, faces left to right: "
            f"{', '.join(names)}. White vertical dividers separate the "
            "faces.")
