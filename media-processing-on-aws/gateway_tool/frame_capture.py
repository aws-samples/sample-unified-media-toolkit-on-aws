"""
Frame-capture (marker stills) validation and batching — pure, no AWS.

Contract (README.md §14.4, the companion mode): markers in ms over one
or more sources; one still per marker, snapped to the SOURCE frame grid
(there is no target fps for stills); optional normalized crop per
marker (same reframe window as quick_edit); optional resize canvas
(fit + pad, never stretch). Deterministic basenames
frame_{index:04d}.{ext}; the manifest is the join.
"""
from __future__ import annotations

try:
    from lib.cut_list import CutListError, validate_crop
except ImportError:          # the flat gateway zip has no lib/ package
    from cut_list import CutListError, validate_crop

MAX_MARKERS = 500
MAX_DISTINCT_SOURCES = 25
MARKERS_PER_ITERATION = 100   # bounds one Map iteration's output count
# One iteration = ONE ffmpeg pass reading from the first marker to the
# last (capture_render), so the batch SPAN is capped at the chunk
# pipeline's workload (~10 min of content) — deep seeks into large
# long-GOP MXFs over HTTPS cost minutes, and sequential read is what
# amortizes them (measured live 2026-10-09).
MAX_SPAN_MS = 600_000
FORMATS = frozenset(("jpeg", "png"))


def validate_frame_capture(fc: dict) -> dict:
    """Validate and normalize; every refusal names its condition."""
    if not isinstance(fc, dict):
        raise CutListError("frame_capture must be an object")
    markers = fc.get("markers")
    if not isinstance(markers, list) or not markers:
        raise CutListError("frame_capture.markers must be a non-empty list")
    if len(markers) > MAX_MARKERS:
        raise CutListError(
            f"{len(markers)} markers exceed the {MAX_MARKERS}-marker cap")
    fmt = fc.get("format", "jpeg")
    if fmt not in FORMATS:
        raise CutListError(
            f"format must be one of {', '.join(sorted(FORMATS))}, "
            f"got {fmt!r}")
    resize = fc.get("resize")
    if resize is not None:
        try:
            w, h = (int(v) for v in str(resize).lower().split("x", 1))
        except ValueError:
            raise CutListError(f"resize is not WIDTHxHEIGHT: {resize!r}")
        if w < 2 or h < 2 or w % 2 or h % 2:
            raise CutListError(
                f"resize must be even in both dimensions: {resize!r}")
        resize = (w, h)

    norm, sources = [], set()
    for i, m in enumerate(markers):
        where = f"markers[{i}]"
        if not isinstance(m, dict):
            raise CutListError(f"{where}: must be an object")
        src = m.get("source_s3uri")
        if not isinstance(src, str) or not src.startswith("s3://"):
            raise CutListError(
                f"{where}: source_s3uri must be an s3:// uri, got {src!r}")
        at = m.get("at_ms")
        if not isinstance(at, int) or isinstance(at, bool) or at < 0:
            raise CutListError(
                f"{where}: at_ms must be an integer ≥ 0, got {at!r}")
        crop = m.get("crop")
        if crop is not None:
            crop = validate_crop(crop, where)
        sources.add(src)
        norm.append({"index": i, "source_s3uri": src, "at_ms": at,
                     "crop": crop})
    if len(sources) > MAX_DISTINCT_SOURCES:
        raise CutListError(
            f"{len(sources)} distinct sources exceed the "
            f"{MAX_DISTINCT_SOURCES}-source cap")
    return {"markers": norm, "format": fmt, "resize": resize,
            "distinct_sources": sorted(sources)}


def batch_markers(markers: list[dict]) -> list[list[dict]]:
    """Group by source, SORT by position (the renderer reads forward
    once), and split a group when it exceeds MARKERS_PER_ITERATION
    stills OR its span exceeds MAX_SPAN_MS of content."""
    by_source: dict[str, list[dict]] = {}
    for m in markers:
        by_source.setdefault(m["source_s3uri"], []).append(m)
    batches = []
    for src in sorted(by_source):
        group = sorted(by_source[src], key=lambda m: m["at_ms"])
        batch: list[dict] = []
        for m in group:
            if batch and (len(batch) >= MARKERS_PER_ITERATION or
                          m["at_ms"] - batch[0]["at_ms"] > MAX_SPAN_MS):
                batches.append(batch)
                batch = []
            batch.append(m)
        if batch:
            batches.append(batch)
    return batches
