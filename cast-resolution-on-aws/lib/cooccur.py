"""
cooccur.py — speaker↔face co-occurrence, aggregated over the whole asset.

Instantaneous co-occurrence is the weakest evidence and confidently wrong
for whole classes (photos, b-roll, shot/reverse-shot — README.md §18 appendix).
So this module only ever aggregates:

- exclusivity weighting: a face alone on screen during speech earns full
  overlap credit; each extra concurrent face divides it.
- turn counting: a true pair co-occurs across MANY separate turns; one
  long lucky scene does not confirm.
- variance as a classifier: a voice with diverse co-occurring faces and
  no stable partner IS voice_only (narrator/VO/announcer), detected
  rather than mis-bound.

Output bindings are PROPOSED or CONFIRMED. Co-occurrence alone confirms a
pairing (who is on screen when this voice talks) but never NAMES anyone —
naming needs an independent mechanism (persons.py).
"""
from __future__ import annotations

from lib.pb_common import (
    CLASS2_MATCHED_FACE_DISCOUNT,
    MIN_TURNS_TO_JUDGE,
    VOICE_ONLY_MAX_SHARE,
    VOICE_ONLY_MIN_FACES,
    overlap_sec,
)


def bind_speakers_to_faces(turns: list[dict], faces: list[dict],
                           class1=None) -> dict:
    """-> {bindings: [...], speaker_class: {speaker_id: class}}.

    binding: {speaker_id, face_person_id, status: proposed|confirmed,
              share, turns_together, lead, mechanism: "cooccurrence"}
    speaker_class: paired | voice_only | insufficient

    `class1` (lib/class1.match output) fixes the matched heads: a
    matched voice is settled (class paired, its binding is class1's,
    not re-judged here); an UNMATCHED class-1 voice is never classified
    voice_only (the classifier structurally ate desk anchors — crowd
    division on every two-shot — and a lead's exit must be class1's
    recorded abstention, not a silent narrator verdict); and a matched
    face's credit toward OTHER voices is discounted (its presence is
    already explained by its own voice — shot/reverse-shot, not them).
    """
    matched_v = {m["speaker_id"] for m in (class1 or {}).get("matches") or []}
    matched_f = {m["face_person_id"]: m["speaker_id"]
                 for m in (class1 or {}).get("matches") or []}
    c1_voices = set((class1 or {}).get("speakers") or [])

    per: dict[str, dict[int, dict]] = {}      # speaker -> face -> stats
    speech: dict[str, float] = {}
    turn_counts: dict[str, int] = {}

    for turn in turns:
        sid = turn["speaker_id"]
        t0, t1 = turn["start"], turn["end"]
        dur = max(0.0, t1 - t0)
        speech[sid] = speech.get(sid, 0.0) + dur
        turn_counts[sid] = turn_counts.get(sid, 0) + 1

        # faces on screen during this turn, with per-instant concurrency
        # approximated per-span (cheap and adequate at P0 granularity).
        # Credit accrues over LIVE spans only (sources.mark_depicted_faces:
        # spans minus media-depiction shots). The signal is SHOT-level, so
        # a depicted shot's face-speech pairing is untrustworthy for EVERY
        # face in it — subtracting per span (never excluding a face
        # whole-video) is what stops the monitor-guest trap where removing
        # the depicted face handed the co-present live face undivided
        # credit and a false CONFIRMED binding (review, 2026-09-06).
        hits = []
        for f in faces:
            ov = sum(overlap_sec(t0, t1, a, b)
                     for a, b in f.get("live_spans", f["spans"]))
            if ov > 0.0:
                hits.append((f["face_person_id"], ov))
        if not hits:
            continue
        crowd = len(hits)
        for pid, ov in hits:
            st = per.setdefault(sid, {}).setdefault(
                pid, {"exclusive_sec": 0.0, "turns": 0})
            credit = ov / crowd
            if pid in matched_f and matched_f[pid] != sid:
                credit *= CLASS2_MATCHED_FACE_DISCOUNT
            st["exclusive_sec"] += credit
            st["turns"] += 1

    bindings, speaker_class = [], {}
    for sid, total in speech.items():
        if sid in matched_v:
            speaker_class[sid] = "paired"     # settled by class1
            continue
        cands = sorted(per.get(sid, {}).items(),
                       key=lambda kv: kv[1]["exclusive_sec"], reverse=True)
        if turn_counts.get(sid, 0) < MIN_TURNS_TO_JUDGE:
            speaker_class[sid] = "insufficient"
            continue
        if not cands:
            speaker_class[sid] = ("insufficient" if sid in c1_voices
                                  else "voice_only")
            continue

        top_pid, top = cands[0]
        share = top["exclusive_sec"] / total if total else 0.0
        runner = cands[1][1]["exclusive_sec"] if len(cands) > 1 else 0.0
        lead = (top["exclusive_sec"] / runner) if runner else float("inf")

        if share < VOICE_ONLY_MAX_SHARE and len(cands) >= VOICE_ONLY_MIN_FACES:
            speaker_class[sid] = ("insufficient" if sid in c1_voices
                                  else "voice_only")
            if sid not in c1_voices:
                continue

        # PROPOSALS ONLY (2026-09-24): the share-based CONFIRM path was
        # retired — the class-1/2 matcher is the one binding engine
        # (its 12 residual corpus confirms were all unnamed faces, now
        # inside the matcher's tier-2 pool). What remains here is the
        # narrator classifier above and the top-candidate proposal, the
        # dossier/adjudication evidence a confirm never was.
        speaker_class[sid] = "insufficient"
        bindings.append({
            "speaker_id": sid, "face_person_id": top_pid,
            "status": "proposed", "mechanism": "cooccurrence",
            "share": round(share, 4), "turns_together": top["turns"],
            "lead": round(lead, 2) if lead != float("inf") else None,
        })
    return {"bindings": bindings, "speaker_class": speaker_class}
