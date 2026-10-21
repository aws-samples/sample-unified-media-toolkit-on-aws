"""
Frame selection: WHICH archived frames to show the model for a span —
from shot-segmentation's own artifacts only (window manifests, per-window
embedding .npy, per-window _frames.tar.gz). No ffmpeg, no source, no
pixel processing (README.md §5, measured).

Keep a frame on CHANGE or on TIME:
- change: cosine distance to the last KEPT frame's DINO embedding above
  the category's change_distance — a static card run collapses to one
  frame;
- time: >= heartbeat_sec since the last kept frame — a slow scroll
  changes too little frame-to-frame and a change-only gate collapsed a
  real cast roll to nothing (mitj404, measured); the heartbeat guarantees
  sampling through crawls.

PURE STDLIB on purpose: the work is one .npy parse and ~one cosine per
archived frame (against the last kept vector only), which plain Python
does in milliseconds — vendoring numpy into the family's flat zip for
that would be the complexity, not the saving.
"""
from __future__ import annotations

import ast
import math
import struct
from array import array
from fractions import Fraction

# NO module defaults for the selection knobs on purpose: change_distance
# and heartbeat_sec are CATEGORY POLICY (fi_common.CATEGORIES) and every
# caller must choose. The first-guess default 0.25 was the measured
# mitj404 defect (three principal cards skipped) — a known-bad number
# must not be one omitted argument away.


def load_npy(blob: bytes) -> tuple:
    """(rows, dim) from a little-endian float32 2-D .npy — the exact
    format shot-segmentation writes. Anything else refuses loudly."""
    if blob[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = blob[6]
    if major == 1:
        (hlen,) = struct.unpack_from("<H", blob, 8)
        offset = 10 + hlen
    else:
        (hlen,) = struct.unpack_from("<I", blob, 8)
        offset = 12 + hlen
    header = ast.literal_eval(blob[offset - hlen:offset].decode("latin-1"))
    if header.get("descr") not in ("<f4", "|f4") or header.get("fortran_order"):
        raise ValueError(f"unsupported .npy layout: {header}")
    n, dim = header["shape"]
    data = array("f")
    data.frombytes(blob[offset:offset + 4 * n * dim])
    rows = [data[i * dim:(i + 1) * dim] for i in range(n)]
    return rows, dim


def frame_times(window_manifest: dict) -> list:
    """Absolute programme seconds for every frame in one window manifest.
    frame_index holds SOURCE frame numbers; frame_rate is a fraction
    string like '24000/1001'."""
    rate = float(Fraction(window_manifest["frame_rate"]))
    return [idx / rate for idx in window_manifest["frame_index"]]


def select(window_manifest: dict, embeddings_blob: bytes, span: dict,
           change_distance: float, heartbeat_sec: float) -> list:
    """Indices (into this window's frame list) worth showing the model
    for the span, in time order."""
    times = frame_times(window_manifest)
    rows, _dim = load_npy(embeddings_blob)
    kept = []
    last_vec = None
    last_time = -1e9
    for i, t in enumerate(times):
        if t < span["start"] or t > span["end"]:
            continue
        changed = last_vec is None or \
            _cosine_distance(rows[i], last_vec) > change_distance
        if changed or t - last_time >= heartbeat_sec:
            kept.append(i)
            last_vec = rows[i]
            last_time = t
    return kept


def frame_names(window_manifest: dict, indices: list) -> list:
    """(programme_second, archive_member_name) per selected index —
    the tarball members are named frame_{source_frame_number:010d}.jpg."""
    times = frame_times(window_manifest)
    idx = window_manifest["frame_index"]
    return [(round(times[i], 2), f"frame_{idx[i]:010d}.jpg")
            for i in indices]


def evenly_spaced(seq: list, n: int) -> list:
    """At most n items, evenly spaced, both endpoints kept — the runaway
    guard's subsample (coverage over density)."""
    if len(seq) <= n:
        return list(seq)
    step = (len(seq) - 1) / (n - 1)
    return [seq[round(i * step)] for i in range(n)]


def _cosine_distance(a, b) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return 1.0 - dot / (math.sqrt(na * nb) + 1e-9)
