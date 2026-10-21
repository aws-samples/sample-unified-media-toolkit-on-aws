"""
Optional shot-segmentation supplement (README.md §21).

Loads a shots.json produced by shot-segmentation-on-aws and validates
it belongs to THIS asset, returning the sorted shot-start frame numbers.
Both pipelines speak SOURCE-ABSOLUTE frame numbers (MP names frames
nominal_start_frame + i*interval even under mark_in_out; shot-seg uses exact
PTS arithmetic), so no offset math exists anywhere.

Every check fails hard: a wrong-asset shots file silently fragmenting or
gluing tracks is worse than a crash. shot_type is deliberately ignored —
every shot start is a boundary (black slivers = two boundaries = correct
containment), and black-frame detection stays with per-frame triage which
judges the actual 1080p pixels (shot-seg's "black" is a ~512px-proxy
verdict; see README.md §21 resolved questions).

Interval semantics — the two consumers differ by one endpoint:
  tracking veto     (a, b]  a face AT a boundary frame is in the NEW shot
                            (mirrors the dHash veto, which tests the
                            transition INTO frame b)
  dense-skip gap    (a, b)  a boundary exactly AT probe b leaves every gap
                            frame in the old, empty-probed shot
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction

from lib.input_schema import (
    SHOT_DURATION_TOLERANCE_PCT,
    SHOT_DURATION_TOLERANCE_SEC,
    SHOT_FPS_MIN_START_TIME_SEC,
    SHOT_FPS_TOLERANCE,
)
from lib.s3_utils import get_json


def load_and_validate(bucket: str, key: str, fps: Fraction,
                      duration_sec: float) -> list[int]:
    """
    Returns sorted shot-start frame numbers. fps is the probe's EXACT frame
    rate; duration_sec is MP's resolved FULL-source duration (mark_in_out
    never clips it — verified against media-processing's probe state).
    """
    doc = get_json(bucket, key)
    shots = doc.get("shots") if isinstance(doc, dict) else None
    if not isinstance(shots, list) or not shots:
        raise ValueError(
            f"shot_segmentation_s3uri: s3://{bucket}/{key} has no non-empty "
            f"'shots' list — not a shot-segmentation shots.json?")

    starts: list[int] = []
    prev = None
    for i, shot in enumerate(shots):
        try:
            start_f = int(shot["start_frame"])
            int(shot["end_frame"])
            start_t = float(shot["start_time"])
            end_t = float(shot["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shot_segmentation_s3uri: shots[{i}] missing/invalid "
                f"start_frame/end_frame/start_time/end_time ({exc})")
        if end_t < start_t:
            raise ValueError(
                f"shot_segmentation_s3uri: shots[{i}] end_time {end_t} < "
                f"start_time {start_t}")
        if prev is not None and start_f <= prev:
            raise ValueError(
                f"shot_segmentation_s3uri: shots[{i}].start_frame "
                f"{start_f} <= previous {prev} — not strictly increasing")
        prev = start_f
        starts.append(start_f)

        # Same-asset fps cross-check on late-enough shots (early shots
        # divide by near-zero times).
        if start_t >= SHOT_FPS_MIN_START_TIME_SEC:
            implied = start_f / start_t
            if abs(implied - float(fps)) / float(fps) > SHOT_FPS_TOLERANCE:
                raise ValueError(
                    f"shot_segmentation_s3uri: shots[{i}] implies "
                    f"{implied:.3f} fps but this asset is {float(fps):.3f} "
                    f"fps — shots.json is from a different asset")

    # Prefer the file's own actual_duration_sec (top-level, exact); the last
    # shot's end_time is the last SAMPLED frame (4 fps coarse — up to ~0.25 s
    # short of EOF), still well inside tolerance as the fallback.
    file_duration = doc.get("actual_duration_sec")
    last_end = (float(file_duration) if file_duration
                else float(shots[-1]["end_time"]))
    tol = max(SHOT_DURATION_TOLERANCE_SEC,
              duration_sec * SHOT_DURATION_TOLERANCE_PCT)
    if abs(last_end - duration_sec) > tol:
        raise ValueError(
            f"shot_segmentation_s3uri: shots file covers {last_end:.1f}s but "
            f"the source is {duration_sec:.1f}s (tolerance {tol:.1f}s) — "
            f"shots.json is from a different asset")
    return starts


def boundary_between(starts: list[int], frame_a: int, frame_b: int) -> bool:
    """Tracking veto: True iff a shot starts in (frame_a, frame_b]."""
    i = bisect_right(starts, frame_a)
    return i < len(starts) and starts[i] <= frame_b


def boundary_inside(starts: list[int], frame_a: int, frame_b: int) -> bool:
    """Dense-skip gap test: True iff a shot starts in (frame_a, frame_b)."""
    i = bisect_right(starts, frame_a)
    return i < len(starts) and starts[i] < frame_b
