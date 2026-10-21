"""
imaging.py — the witness's one image operation: lower-third cropping.

A chyron/lower-third read sends the model the WHOLE frame while the
text it wants occupies the bottom band — the crop cuts the image tokens
~60% for those spans (the cost posture: stacking amortizes the prompt,
cropping shrinks what remains). Requested PER SPAN by the caller
(`"crop": "lower_third"`); the witness never guesses which spans are
chyrons.

The fraction is STRUCTURAL, not fitted: broadcast lower-third graphics
occupy roughly the bottom quarter to third of the frame; 0.4 keeps a
safety band above them. A crop that loses a high super loses evidence
silently, so the default posture stays "no crop" and the caller opts
spans in.
"""
from __future__ import annotations

import pyvips

from lib.fi_common import CROP_MODES  # noqa: F401  (one home: fi_common)

LOWER_THIRD_FRACTION = 0.4


def crop(jpeg_bytes: bytes, mode: str) -> bytes:
    """Apply a validated crop mode to one JPEG frame."""
    if mode != "lower_third":
        raise ValueError(f"unknown crop mode {mode!r} — one of "
                         f"{CROP_MODES}")
    img = pyvips.Image.new_from_buffer(jpeg_bytes, "", access="sequential")
    band = int(img.height * LOWER_THIRD_FRACTION)
    cropped = img.crop(0, img.height - band, img.width, band)
    return cropped.jpegsave_buffer(Q=80)
