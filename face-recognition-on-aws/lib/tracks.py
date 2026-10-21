"""
Temporal face tracking without shot boundaries (README.md §4).

A face joins an open track when ALL of:
  1. same gender as the track
  2. frame gap to the track's last face <= max_track_gap_sec
  3. NO hash break between the two frames (any stable patch, from frame_index)
     — the dHash data doubles as a free cut detector, restoring the old
     shot-boundary containment
  4. IoU against the track's last box, DILATED proportionally to the gap
     (dilate = 1 + IOU_GAP_DILATION * gap_sec), >= MIN_TRACK_IOU

Box interpolation: within a track, dedup-skipped frames get linearly
interpolated boxes tagged source="interpolated" (copy-forward at track end
is handled by the rollup, not synthesized here).
"""

from __future__ import annotations

from lib.input_schema import IOU_GAP_DILATION, MIN_TRACK_IOU


def _dilate(box: dict, factor: float) -> dict:
    cx = box["l"] + box["w"] / 2
    cy = box["t"] + box["h"] / 2
    w = box["w"] * factor
    h = box["h"] * factor
    return {"l": cx - w / 2, "t": cy - h / 2, "w": w, "h": h}


def iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["l"], a["t"]
    ax2, ay2 = a["l"] + a["w"], a["t"] + a["h"]
    bx1, by1 = b["l"], b["t"]
    bx2, by2 = b["l"] + b["w"], b["t"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def build_tracks(
    faces: list[dict],
    max_gap_sec: float,
    hash_break_between,
) -> list[dict]:
    """
    faces: [{frame_num, timestamp, box, gender, ...detail}], any order.
    hash_break_between(frame_a, frame_b) -> bool — True when a cut/hash break
        exists between the two frame numbers (exclusive range).
    Returns tracks: [{track_id, gender, faces: [...time-ordered...]}].
    """
    ordered = sorted(faces, key=lambda f: f["frame_num"])
    open_tracks: list[dict] = []
    done: list[dict] = []

    for face in ordered:
        ts = face["timestamp"]
        best = None
        best_iou = 0.0
        for tr in open_tracks:
            last = tr["faces"][-1]
            gap = ts - last["timestamp"]
            if gap > max_gap_sec:
                continue
            if face.get("gender") != tr["gender"]:
                continue
            if hash_break_between(last["frame_num"], face["frame_num"]):
                continue
            dilated = _dilate(last["box"], 1.0 + IOU_GAP_DILATION * gap)
            score = iou(dilated, face["box"])
            if score >= MIN_TRACK_IOU and score > best_iou:
                best, best_iou = tr, score
        if best is not None:
            best["faces"].append(face)
        else:
            open_tracks.append({"gender": face.get("gender"), "faces": [face]})

        # Retire tracks that can no longer accept anything
        still_open = []
        for tr in open_tracks:
            if ts - tr["faces"][-1]["timestamp"] > max_gap_sec:
                done.append(tr)
            else:
                still_open.append(tr)
        open_tracks = still_open

    done.extend(open_tracks)
    done.sort(key=lambda t: t["faces"][0]["frame_num"])
    for i, tr in enumerate(done):
        tr["track_id"] = i
        tr["first_ts"] = tr["faces"][0]["timestamp"]
        tr["last_ts"] = tr["faces"][-1]["timestamp"]
        tr["appearance_count"] = len(tr["faces"])
    return done


def interpolate_skipped(track: dict, skipped_frames: list[dict]) -> None:
    """
    Insert interpolated boxes for dedup-skipped frames that fall inside the
    track's span. skipped_frames: [{frame_num, timestamp}] time-ordered.
    Detected faces carry source="detected"; synthesized ones "interpolated".
    """
    faces = track["faces"]
    for f in faces:
        f.setdefault("source", "detected")
    if len(faces) < 2:
        return

    inserts = []
    for sk in skipped_frames:
        ts = sk["timestamp"]
        if not (track["first_ts"] < ts < track["last_ts"]):
            continue
        # Find bracketing detected faces
        prev = max((f for f in faces if f["timestamp"] <= ts),
                   key=lambda f: f["timestamp"])
        nxt = min((f for f in faces if f["timestamp"] > ts),
                  key=lambda f: f["timestamp"])
        span = nxt["timestamp"] - prev["timestamp"]
        alpha = (ts - prev["timestamp"]) / span if span > 0 else 0.0
        box = {k: round(prev["box"][k] + alpha * (nxt["box"][k] - prev["box"][k]), 6)
               for k in ("l", "t", "w", "h")}
        inserts.append({
            "frame_num": sk["frame_num"],
            "timestamp": ts,
            "box": box,
            "gender": track["gender"],
            "source": "interpolated",
        })
    if inserts:
        faces.extend(inserts)
        faces.sort(key=lambda f: f["frame_num"])
        track["appearance_count"] = len(faces)
