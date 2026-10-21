"""
Face quality assessment, fusion ranking, and representative-centroid
selection.

Selection is RELATIVE (rank within the candidate pool), not absolute-window
gating. The old pipeline's ported brightness/sharpness windows were measured
against a real 2-hour broadcast (7,327 detected faces) and found miscalibrated
AND skin-tone-correlated: the sharpness floor (>=30) rejected 86.8% of all
faces (population median 9.9), and the brightness ceiling (<=70) rejected
98-100% of the lightest-lit people's faces while passing 89% of a
dark-skinned anchor's — when the window bound at all, it picked light-skinned
people's darkest frames. Rekognition's Quality.Brightness measures
illumination but correlates with skin reflectance (same set/lighting: light
skin mean 81, dark skin mean 65), so any absolute brightness gate treats
equally-well-lit faces differently by skin tone. Relative ranking within a
person's own candidate pool is skin-tone-neutral by construction; the only
absolute floors kept are extreme ones (near-black crop, unusably blurry).
"""

from __future__ import annotations

import math

from lib.input_schema import (
    CENTROID_MIN_SHARPNESS,
    POSE_MAX_ABS_PITCH,
    POSE_MAX_ABS_YAW,
    QUALITY_MIN_BRIGHTNESS,
)


def to_face_data(detail: dict) -> dict:
    """Extract the per-face record from a Rekognition FaceDetail (grid coords
    are recalibrated by the caller via grid_image.map_face_to_cell)."""
    pose = detail.get("Pose") or {}
    pitch = float(pose.get("Pitch") or 0)
    roll = float(pose.get("Roll") or 0)
    yaw = float(pose.get("Yaw") or 0)
    pose_dist = ((pitch / 180) ** 2 + (roll / 180) ** 2
                 + (yaw / 180) ** 2) ** (1 / 3)

    quality = detail.get("Quality") or {}
    emotions = detail.get("Emotions") or []
    top_emotion = max(emotions, key=lambda e: e.get("Confidence") or 0,
                      default=None)

    def _flag(name):
        v = detail.get(name) or {}
        return bool(v.get("Value")) if "Value" in v else None

    age = detail.get("AgeRange") or {}
    return {
        "confidence": round(float(detail.get("Confidence") or 0), 2),
        "age_range": [int(age.get("Low") or 0), int(age.get("High") or 0)],
        "gender": ((detail.get("Gender") or {}).get("Value")),
        "emotion": (top_emotion or {}).get("Type"),
        "pose": {
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "yaw": round(yaw, 2),
            "pose_dist": round(pose_dist, 4),
        },
        "brightness": round(float(quality.get("Brightness") or 0), 2),
        "sharpness": round(float(quality.get("Sharpness") or 0), 2),
        "eyes_open": _flag("EyesOpen"),
        "eyeglasses": _flag("Eyeglasses"),
        "sunglasses": _flag("Sunglasses"),
        "mouth_open": _flag("MouthOpen"),
        "face_occluded": _flag("FaceOccluded"),
    }


def fusion_rank(face: dict) -> float:
    """w*h*(1-pose_dist) — larger, more frontal faces rank higher."""
    box = face.get("box") or {}
    return (box.get("w") or 0) * (box.get("h") or 0) * (
        1.0 - face["pose"]["pose_dist"])


def quality_ok(face: dict) -> bool:
    """Extreme floor only — near-black crops. NOT a quality window (see
    module docstring for why the absolute windows were removed)."""
    return face["brightness"] >= QUALITY_MIN_BRIGHTNESS


def choose_representative(faces: list[dict]) -> int | None:
    """
    Index of the representative (centroid) face for a track, by relative
    ranking within the track's own faces:
      1. Prefer non-occluded faces above the extreme-brightness floor.
      2. Among candidates: top-3 by box area, then sharpest.
      3. Reject any pick with sharpness < CENTROID_MIN_SHARPNESS.
    Returns None when no face qualifies.
    """
    if not faces:
        return None

    def _area(i):
        b = faces[i].get("box") or {}
        return (b.get("w") or 0) * (b.get("h") or 0)

    pool = [i for i, f in enumerate(faces) if not f.get("face_occluded")]
    if not pool:
        pool = list(range(len(faces)))
    candidates = [i for i in pool if quality_ok(faces[i])]
    if not candidates:
        candidates = pool

    top3 = sorted(candidates, key=_area, reverse=True)[:3]
    best = max(top3, key=lambda i: faces[i]["sharpness"])
    if faces[best]["sharpness"] < CENTROID_MIN_SHARPNESS:
        return None
    return best


def pose_ok(face: dict) -> bool:
    """Hard pose gate: |yaw| and |pitch| within POSE_MAX_ABS_* degrees.
    fusion_rank's (1 - pose_dist) term is only a soft multiplicative penalty
    that a large-enough face overrides — verified on real content, where the
    top-ranked index candidate was a yaw=-25 3/4 profile because its box area
    dominated. Roll is deliberately NOT gated: a tilted-but-frontal head is
    fine for matching (Rekognition normalizes in-plane rotation)."""
    pose = face.get("pose") or {}
    return (abs(pose.get("yaw") or 0) <= POSE_MAX_ABS_YAW
            and abs(pose.get("pitch") or 0) <= POSE_MAX_ABS_PITCH)


def best_faces(faces: list[dict], limit: int) -> list[int]:
    """Indices of the best faces by relative fusion rank, preferring
    frontal-pose + non-occluded + eyes-open candidates; each gate falls back
    to the wider pool rather than emptying it (a cluster with only profile
    views still gets its best available face)."""
    ok = [i for i, f in enumerate(faces) if quality_ok(f)]
    if not ok:
        ok = list(range(len(faces)))
    frontal = [i for i in ok
               if pose_ok(faces[i])
               and not faces[i].get("face_occluded")
               and faces[i].get("eyes_open") is not False]
    if frontal:
        ok = frontal
    ranked = sorted(ok, key=lambda i: fusion_rank(faces[i]), reverse=True)
    if limit == 1 and len(ranked) > 1:
        # Single-face pick (persistent indexing): fusion_rank is
        # area-dominated and can land on a soft frame — same top-3-then-
        # sharpest refinement choose_representative uses.
        return [max(ranked[:3], key=lambda i: faces[i]["sharpness"])]
    return ranked[:limit]
