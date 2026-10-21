"""
GrabCandidateFrames — pure code, stdlib only: for every candidate,
fetch its shot's frame archive (shot-segmentation's per-shot tar.gz of
proxy-resolution JPEGs) and persist frames under
frames/{archetype}_shot{id:04d}[_fN].jpg. A missing/broken archive
drops the frame, never the run: the candidate stays in the slate
flagged frame=null, and the rerank simply never sees it.

PERSON archetypes get up to FRAMES_PER_PERSON_CANDIDATE members, and
face-recognition's per-detection attributes pick them (user design
2026-10-06): facegroup.json faces carry timestamped eyes_open /
sunglasses / face_occluded flags, so the members grabbed are the ones
nearest a GOOD face moment — the archives sample ~1 fps and a talking
face's single middle frame is frequently mid-blink (measured: a news
hero slate where the vision judge's own reason was "the most open
eyes AMONG ALL FOUR"). The judge still arbitrates the pixels; FR is
the prior that decides which pixels it gets to see. No usable FR
records degrades to a spread of members (middle, quarter points) —
choice over a single blind midpoint either way. Non-person archetypes
(vistas) keep the single middle member: landscapes don't blink.
"""
from __future__ import annotations

import io
import logging
import os
import re
import tarfile

from lib import s3io
from lib.profiles import ARCHETYPES, FRAMES_PER_PERSON_CANDIDATE

logger = logging.getLogger(__name__)

PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")

_FRAME_NUM = re.compile(r"(\d+)")


def good_face_times(facegroup: dict) -> list[tuple[float, float]]:
    """[(timestamp, sharpness)] of every face record whose attributes
    say the moment is grabbable: eyes open, no sunglasses, not
    occluded. Flags are tri-state (None = Rekognition abstained) —
    only an asserted True eyes_open qualifies, only an asserted True
    sunglasses/occlusion disqualifies."""
    out = []
    for p in facegroup.get("people") or []:
        for tr in p.get("tracks") or []:
            for f in tr.get("faces") or []:
                if f.get("eyes_open") is True \
                        and f.get("sunglasses") is not True \
                        and f.get("face_occluded") is not True \
                        and f.get("timestamp") is not None:
                    out.append((float(f["timestamp"]),
                                float(f.get("sharpness") or 0.0)))
    return out


def derive_fps(facegroup: dict) -> float | None:
    """Archive member names carry SOURCE frame numbers; facegroup
    faces carry both frame_num and timestamp, so the fps that maps a
    member name to seconds is derivable from the data itself."""
    ratios = []
    for p in facegroup.get("people") or []:
        for tr in p.get("tracks") or []:
            for f in tr.get("faces") or []:
                ts = f.get("timestamp")
                fn = f.get("frame_num")
                if ts and fn:
                    ratios.append(fn / ts)
    if not ratios:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]


def choose_members(names: list[str], fps: float | None,
                   good_times: list[tuple[float, float]],
                   k: int) -> list[str]:
    """Up to k archive members, best-prior first. With usable FR
    evidence: the member nearest each good face moment, sharpest
    moments first, distinct members only. Without: the middle member
    plus the quarter points — spread beats a single blind midpoint."""
    n = len(names)
    spread = []
    for i in (n // 2, n // 4, (3 * n) // 4):
        if names[i] not in spread:
            spread.append(names[i])
    spread = spread[:k]
    if not fps or not good_times:
        return spread
    times = []
    for name in names:
        m = _FRAME_NUM.search(name)
        if not m:
            return spread          # unparseable convention: degrade
        times.append(int(m.group(1)) / fps)
    # the member times ARE the shot window: only face moments inside
    # it (half a sampling step of slack) can inform a member choice
    step = (times[-1] - times[0]) / max(1, n - 1)
    inside = [(ts, sh) for ts, sh in good_times
              if times[0] - step / 2 <= ts <= times[-1] + step / 2]
    if not inside:
        return spread
    # score MEMBERS by their distance to the nearest good moment —
    # a member more than one sampling step from every good moment
    # carries no evidence and is never chosen over the spread
    scored = []
    for i, name in enumerate(names):
        d, sh = min((abs(times[i] - ts), sh) for ts, sh in inside)
        if d <= step:
            scored.append((d, -sh, i, name))
    chosen = [name for _, _, _, name in sorted(scored)[:k]]
    return chosen or spread


def _members(tar_bytes: bytes, picks) -> list[tuple[str, bytes]]:
    """Extract `picks(names)`-selected members from the archive."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        names = sorted(m.name for m in tf.getmembers()
                       if m.isfile() and m.name.endswith(".jpg"))
        if not names:
            return []
        return [(name, tf.extractfile(name).read())
                for name in picks(names)]


def lambda_handler(event: dict, context) -> dict:
    plan_s3uri = event["plan_s3uri"]
    candidates_s3uri = event["candidates_s3uri"]
    plan = s3io.get_json(plan_s3uri)
    doc = s3io.get_json(candidates_s3uri)
    prefix = candidates_s3uri.rsplit("/", 1)[0]

    # FR attribute prior, best-effort: absent input, fetch failure or
    # an older FR build without per-face attributes all degrade to the
    # spread — never to a failed run
    fg, fps, good_all = None, None, []
    fr_uri = (plan.get("inputs") or {}).get("face_recognition_s3uri")
    if fr_uri:
        try:
            fg = s3io.get_json(f"{fr_uri.rstrip('/')}/facegroup.json")
            fps = derive_fps(fg)
            good_all = sorted(good_face_times(fg))
        except Exception as exc:            # noqa: BLE001 — degrade
            logger.warning("facegroup unavailable (%s): %s",
                           fr_uri, exc)

    grabbed = failed = 0
    for archetype, cands in doc["slates"].items():
        person = bool((ARCHETYPES.get(archetype) or {}).get("people"))
        for cand in cands:
            key = cand.get("frames_key")
            if not key:
                cand["frame"] = None
                failed += 1
                continue
            good = good_all if person else []
            k = FRAMES_PER_PERSON_CANDIDATE if person else 1

            def picks(names, good=good, k=k):
                if k == 1:
                    return [names[len(names) // 2]]
                return choose_members(names, fps, good, k)

            try:
                members = _members(
                    s3io.get_bytes(f"s3://{PROXY_BUCKET}/{key}"), picks)
            except Exception as exc:        # noqa: BLE001 — degrade
                logger.warning("frame archive unreadable (%s): %s",
                               key, exc)
                members = []
            if not members:
                cand["frame"] = None
                failed += 1
                continue
            uris = []
            for j, (member, body) in enumerate(members):
                suffix = f"_f{j}" if len(members) > 1 else ""
                frame_uri = (f"{prefix}/frames/{archetype}_shot"
                             f"{int(cand['shot_id']):04d}{suffix}.jpg")
                s3io.put_bytes(frame_uri, body, "image/jpeg")
                uris.append(frame_uri)
            cand["frame"] = uris[0]
            cand["frames"] = uris
            cand["frame_member"] = members[0][0]
            grabbed += len(uris)

    s3io.put_json(candidates_s3uri, doc)
    return {"uuid": doc["uuid"], "plan_s3uri": plan_s3uri,
            "candidates_s3uri": candidates_s3uri,
            "frames_grabbed": grabbed, "frames_failed": failed}
