"""
Frame access (README.md §5, GOTCHAS "Frames live inside tarballs"): a shot's
frames exist ONLY inside its `shot_frames/shot_NNNN_frames.tar.gz` archive
— there are no loose per-shot JPEGs anywhere in a shot-segmentation run.
Fetch the archive, pick `n` spread-out frames, return ORIGINAL bytes:
frames pass through untouched (no re-encode, no resize — the proxy
resolution IS the designed evidence, and re-encoding changes it for no
gain; GOTCHAS, the tiling probes).
"""
from __future__ import annotations

import io
import tarfile


def pick_frames(tar_bytes: bytes, n: int) -> list:
    """`n` evenly-spread frame JPEGs (bytes) from the archive, sorted by
    member name (the frames are numbered in capture order). n=1 takes the
    middle frame — one endpoint samples a transition, the middle samples
    the take."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        members = sorted(
            (m for m in tar.getmembers()
             if m.isfile() and m.name.lower().endswith(".jpg")),
            key=lambda m: m.name)
        if not members:
            raise ValueError("shot archive holds no .jpg frames")
        n = max(1, min(n, len(members)))
        if n == 1:
            picks = [members[len(members) // 2]]
        else:
            step = (len(members) - 1) / (n - 1)
            picks = [members[round(i * step)] for i in range(n)]
        return [tar.extractfile(m).read() for m in picks]


def frame_label(image_no: int, shot_id: int, k: int, total: int,
                role: str) -> str:
    """The per-frame text label — the JOIN KEY the prompt trusts over its
    own sense of ordering (README §3 labeling discipline)."""
    return f"Image {image_no} = shot_id {shot_id} ({k} of {total}) [{role}]:"
