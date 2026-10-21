"""
class1.py — hierarchical voice↔face matching, tier one.

Top speakers and top faces are CLASS-1 citizens (user direction,
2026-09-23): the handful of voices and faces that carry the programme.
They are matched FIRST, as a small K×K assignment problem, on evidence
that survives the desk two-shot where raw exclusivity share fails
(measured: a two-anchor desk divides every co-shot's credit, so the
co-occurrence classifier ate both anchors' voices). Everything else is
class 2 and runs the existing mechanisms afterward, with the matched
heads FIXED (cooccur.py discounts matched faces and never classifies a
class-1 voice as voice_only).

Selection is STRUCTURAL — fractions of the programme's total speech and
screen time, never fixed counts, so a two-hander and a 2.5h broadcast
both pick their actual leads.

Two angles, per the user's framing (2026-09-24): WHAT IT IS and WHAT IT
IS NOT. Assertion evidence is noisy and aggregates toward confidence;
exclusion evidence is near-certain the moment it fires and needs no
weights. So exclusions strike pairs from the matrix FIRST, and scores
only rank the survivors.

WHAT IT IS NOT (eliminations — structural, weightless):
- F0×GENDER contradiction: the face registry's gender against the voice
  cluster's median F0. Bands from the 2026-09-23 corpus calibration (66
  bound pairs, 41 male / 24 female faces): male p95 = 178 Hz, female
  p05 = 143 Hz. The 140-180 overlap band abstains; the check is
  confidence-gated (>= F0_MIN_SENTENCES voiced sentences).
- ZERO CO-PRESENCE: a face never on screen during any of the voice's
  turns and never cut to at a turn start cannot be its on-camera
  binding (voice-over is what voice_only expresses, not a binding).
- TAKEN: one-to-one assignment — a matched face is nobody else's.
When the eliminations leave a voice exactly ONE survivor, it matches BY
ELIMINATION at a relaxed evidence floor — the struck alternatives are
the corroboration (measured: a 106 Hz male voice whose top onset
candidate was a female face; the veto struck her and the male runner-up
was the right answer).

WHAT IT IS (assertions — aggregate, scored):
- ONSET FOLLOWING: the edit CUTS TO the speaker as they start talking —
  a face whose appearance begins within ONSET_WINDOW_SEC of the voice's
  turn starts (measured on 11 known pairs: 68-95% follow rate on
  scripted content where shot/reverse-shot grammar rules, 34-63% on
  news where desk two-shots don't cut; 10/11 top-1 by this cue alone).
- presence LIFT: P(face on screen | voice speaks) against the face's
  base rate — immune to crowd division (a face that is *always* on
  screen explains nothing; one that shows up disproportionately during
  this voice's turns does).
- SOLO cutaways: the face alone on screen while the voice speaks.
- NAME join: an independently cued voice name (self_id/sign_off/
  resolved vocative) matching the face source's name, exact or token
  subset. A MISMATCH is deliberately NOT a penalty: on scripted content
  the voice carries the CHARACTER name and the face the actor's — a
  mismatch is meaningless there, only agreement is evidence.
- F0 agreement: a small boost, never a decision.

Assignment is one-to-one over the survivors, confident-first — the
highest-scoring pair whose margin over the voice's next-best FREE face
clears the bar. (A forced-by-elimination endgame existed briefly and
was deleted 2026-09-24: zero firings once onset and floor-holding
joined the assertion set.)
Complementarity falls out of the structure — once the desk partner's
own pair locks, the shared two-shot face stops being an alternative and
the margin opens. When no assignable pair remains, every unmatched
class-1 voice exits with a RECORDED abstention carrying its top
candidates AND its struck pairs with reasons (the no-silent-exit
guarantee, both ledgers — exactly what the adjudication round needs);
unmatched class-1 faces are listed too.
"""
from __future__ import annotations

import bisect

from lib import pb_common
from lib.pb_common import (
    FLOOR_BLOCK_JOIN_GAP_SEC,
    FLOOR_BLOCK_MIN_SEC,
    FLOOR_REF,
    CONFIRM_MIN_TURNS,
    CLASS1_LIFT_CAP,
    CLASS1_MIN_MARGIN,
    CLASS1_MIN_SCORE,
    CLASS1_MIN_SHARE,
    CLASS1_NAME_WEIGHT,
    CLASS1_SOLO_REF,
    DEPICTED_SHARE_THRESHOLD,
    F0_FEMALE_MIN_HZ,
    F0_MALE_MAX_HZ,
    F0_CHILD_AGE,
    F0_ELDERLY_AGE,
    F0_MATCH_BONUS,
    F0_MIN_SENTENCES,
    ONSET_REF,
    ONSET_WINDOW_SEC,
    spans_overlap_sec,
    spans_total_sec,
)

_CUE_KINDS = ("self_id", "sign_off", "vocative_resolved")


def _top_by_share(items: list[tuple], total: float) -> list:
    """Structural class-1 selection: every member whose individual
    share of the total clears CLASS1_MIN_SHARE, descending (the floor
    is its own K bound — see pb_common). items: [(key, seconds)]."""
    if total <= 0.0:
        return []
    return [key for key, sec in sorted(items, key=lambda kv: -kv[1])
            if sec >= CLASS1_MIN_SHARE * total]


def _f0_check(f0_entry: dict | None, gender: str | None,
              age_range=None) -> dict:
    """F0×gender compatibility for one pair. Confidence-gated: no F0
    stats, too few voiced sentences, or no recognizer gender ->
    unassessed; inside the calibrated overlap band -> abstain. Only a
    clear cross-register contradiction vetoes — and AGE gates the
    strike (measured 2026-09-24: an 80-year-old actor's real 200 Hz
    voice was struck and the elimination rung bound the wrong
    survivor). Physiology, not tuning: aging shifts F0 toward the
    shared band in both directions, so an elderly face abstains
    instead of striking; a child's face is unassessable (prepubescent
    F0 is high regardless of sex)."""
    if not f0_entry or not gender:
        return {"verdict": "unassessed"}
    f0 = f0_entry.get("f0_median_hz")
    n = f0_entry.get("f0_n_sentences") or 0
    if not f0 or n < F0_MIN_SENTENCES:
        return {"verdict": "unassessed"}
    age = None
    if age_range and len(age_range) == 2:
        age = (float(age_range[0]) + float(age_range[1])) / 2.0
    if age is not None and age <= F0_CHILD_AGE:
        return {"verdict": "unassessed", "age_gate": "child"}
    g = gender.casefold()
    out = {"f0_hz": f0, "gender": gender, "n_sentences": n}
    elderly = age is not None and age >= F0_ELDERLY_AGE
    if g == "male" and f0 > F0_MALE_MAX_HZ:
        return {**out, "verdict": "abstain", "age_gate": "elderly"} \
            if elderly else {**out, "verdict": "veto"}
    if g == "female" and f0 < F0_FEMALE_MIN_HZ:
        return {**out, "verdict": "abstain", "age_gate": "elderly"} \
            if elderly else {**out, "verdict": "veto"}
    if F0_FEMALE_MIN_HZ <= f0 <= F0_MALE_MAX_HZ:
        return {**out, "verdict": "abstain"}   # calibrated overlap band
    return {**out, "verdict": "compatible"}


def _names_join(voice_names: set, face_name: str | None) -> bool:
    """Exact (casefolded) or token-subset agreement, either direction
    (vocative 'Craig' joins collection 'Craig Merton')."""
    if not face_name or not voice_names:
        return False
    ftoks = frozenset(face_name.casefold().split())
    for vn in voice_names:
        vtoks = frozenset(vn.casefold().split())
        if vtoks and (vtoks <= ftoks or ftoks <= vtoks):
            return True
    return False


def match(transcript, faces, cues=None, speaker_f0=None) -> dict | None:
    """-> {speakers, faces, matches, abstentions, unmatched_faces} or
    None when either modality is absent (three-state: not assessed).
    matches are confirmed-binding shaped (mechanism class1_assignment)
    so persons.build_persons merges them like any confirmed binding."""
    if not transcript or not faces:
        return None
    turns = transcript.get("turns") or []
    speakers = transcript.get("speakers") or {}
    cues = cues or []
    speaker_f0 = speaker_f0 or {}

    duration = transcript.get("duration_sec") or max(
        [t["end"] for t in turns]
        + [b for f in faces for _, b in f.get("spans") or []],
        default=0.0)
    if not duration:
        return None

    c1_voices = _top_by_share(list(speakers.items()), sum(speakers.values()))
    live = [f for f in faces
            if (f.get("depicted_share") or 0.0) < DEPICTED_SHARE_THRESHOLD]
    c1_faces = _top_by_share(
        [(f["face_person_id"], f["screen_sec"]) for f in live],
        sum(f["screen_sec"] for f in live))
    if not c1_voices or not c1_faces:
        return None

    # tier 2 (measured 2026-09-24: a supporting player at 3.2% of
    # speech — under the class-1 floor — carried onset 14/14 and lift
    # 11 against her own face, and the legacy share rule crowd-divided
    # her to a proposal): the SAME pair evidence runs over the residue
    # after the heads lock. The floor means ORDERING, not eligibility.
    # Voices need enough separate turns for onset statistics to mean
    # anything (the co-occurrence corroboration constant).
    turn_counts: dict[str, int] = {}
    for t in turns:
        turn_counts[t["speaker_id"]] = turn_counts.get(t["speaker_id"], 0) + 1
    t2_voices = [sid for sid in speakers
                 if sid not in c1_voices
                 and turn_counts.get(sid, 0) >= CONFIRM_MIN_TURNS]

    by_fid = {f["face_person_id"]: f for f in faces}
    voice_spans = {sid: [] for sid in [*c1_voices, *t2_voices]}
    for t in turns:
        if t["speaker_id"] in voice_spans:
            voice_spans[t["speaker_id"]].append((t["start"], t["end"]))
    voice_names = {sid: {c["name"] for c in cues
                         if c.get("speaker_id") == sid
                         and c.get("kind") in _CUE_KINDS
                         and c.get("confidence") == "high"}
                   for sid in voice_spans}
    # a cluster that SELF-identifies as two different people has no
    # identity to join with (measured 2026-09-24: a co-anchor desk
    # cross-merged at the cluster level — the absorbed anchor's "I'm
    # anchor_C" inside the other's cluster won the name join and
    # tipped four statistically tied pairs the WRONG way). Distinct
    # self-claims void that voice's joins and flag its pairs; the tie
    # then abstains honestly into the dossier. Vocative-resolved names
    # stay: they are OTHERS addressing the voice, not the cluster
    # claiming to be someone.
    mixed_voice = set()
    for sid in voice_spans:
        claims = {c["name"].casefold() for c in cues
                  if c.get("speaker_id") == sid
                  and c.get("kind") in ("self_id", "sign_off")
                  and c.get("confidence") == "high"}
        if len(claims) >= 2:
            mixed_voice.add(sid)
            voice_names[sid] = {c["name"] for c in cues
                                if c.get("speaker_id") == sid
                                and c.get("kind") == "vocative_resolved"
                                and c.get("confidence") == "high"}

    # solo cutaways need per-turn concurrency over ALL faces (a face
    # "alone" among the leads but sharing frame with a b-roll extra is
    # not a cutaway) — same O(turns x faces) sweep cooccur runs
    solo: dict[tuple, float] = {}
    for t in turns:
        sid = t["speaker_id"]
        if sid not in voice_spans:
            continue
        hits = []
        for f in faces:
            ov = sum(pb_common.overlap_sec(t["start"], t["end"], a, b)
                     for a, b in f.get("live_spans", f["spans"]))
            if ov > 0.0:
                hits.append((f["face_person_id"], ov))
        if len(hits) == 1:
            key = (sid, hits[0][0])
            solo[key] = solo.get(key, 0.0) + hits[0][1]

    # onset following: a face whose appearance BEGINS within the window
    # after a turn start earned a cut-to vote (bisect over span starts)
    span_starts = {f["face_person_id"]:
                   sorted(a for a, _ in f.get("live_spans", f["spans"]))
                   for f in faces}
    turn_starts = {sid: sorted(a for a, _ in voice_spans[sid])
                   for sid in voice_spans}

    def _onset_share(sid, fid):
        starts, cuts = turn_starts[sid], span_starts.get(fid) or []
        if not starts or not cuts:
            return 0.0
        n = 0
        for t0 in starts:
            i = bisect.bisect_right(cuts, t0)
            if i < len(cuts) and cuts[i] <= t0 + ONSET_WINDOW_SEC:
                n += 1
        return n / len(starts)

    # floor-holding blocks per voice (user direction, 2026-09-24):
    # maximal runs of one speaker's turns, no other speaker
    # intervening, joined across short pauses — where the camera
    # favors the speaker (see pb_common for the measured shares)
    floor_blocks: dict[str, list] = {sid: [] for sid in voice_spans}
    cur_sid, cur = None, None
    for t in sorted(turns, key=lambda t: t["start"]):
        sid = t["speaker_id"]
        if sid == cur_sid and cur \
                and t["start"] - cur[1] <= FLOOR_BLOCK_JOIN_GAP_SEC:
            cur[1] = max(cur[1], t["end"])
            continue
        if cur_sid in floor_blocks and cur \
                and cur[1] - cur[0] >= FLOOR_BLOCK_MIN_SEC:
            floor_blocks[cur_sid].append(tuple(cur))
        cur_sid, cur = sid, [t["start"], t["end"]]
    if cur_sid in floor_blocks and cur \
            and cur[1] - cur[0] >= FLOOR_BLOCK_MIN_SEC:
        floor_blocks[cur_sid].append(tuple(cur))

    def _floor_share(sid, fid):
        blocks = floor_blocks.get(sid) or []
        tot = sum(b - a for a, b in blocks)
        if not tot:
            return 0.0
        f = by_fid[fid]
        return spans_overlap_sec(blocks,
                                 f.get("live_spans", f["spans"])) / tot

    # ── pair matrix: eliminate first, score the survivors ────────────
    pairs: dict[tuple, dict] = {}

    def _pair(sid, fid):
        if (sid, fid) in pairs:
            return pairs[(sid, fid)]
        speech = speakers.get(sid) or 0.0
        f = by_fid[fid]
        lspans = f.get("live_spans", f["spans"])
        on_share = (spans_overlap_sec(voice_spans[sid], lspans) / speech
                    if speech else 0.0)
        base = max(spans_total_sec(lspans) / duration, 0.01)
        lift = on_share / base
        f0 = _f0_check(speaker_f0.get(sid), f.get("gender"),
                       f.get("age_range"))
        join = _names_join(voice_names[sid], f.get("name"))
        onset = _onset_share(sid, fid)
        floor = _floor_share(sid, fid)
        detail = {"speaker_id": sid, "face_person_id": fid,
                  "lift": round(lift, 2),
                  "on_share": round(on_share, 3),
                  "onset_share": round(onset, 3),
                  "floor_share": round(floor, 3),
                  "solo_sec": round(solo.get((sid, fid), 0.0), 1),
                  "name_join": join, "f0": f0}
        if sid in mixed_voice:
            detail["mixed_voice"] = True
        if f0["verdict"] == "veto":
            detail["eliminated"] = "f0_gender"
        elif on_share == 0.0 and onset == 0.0:
            # never on screen while it talks, never cut to at a
            # turn start: not this voice's ON-CAMERA binding
            detail["eliminated"] = "never_present"
        if detail.get("eliminated"):
            detail["score"] = None           # struck, and it says why
        else:
            score = (min(max(lift - 1.0, 0.0), CLASS1_LIFT_CAP)
                     / CLASS1_LIFT_CAP
                     + min(detail["solo_sec"] / max(speech, 1.0)
                           / CLASS1_SOLO_REF, 1.0)
                     + min(onset / ONSET_REF, 1.0)
                     + min(floor / FLOOR_REF, 1.0)
                     + (CLASS1_NAME_WEIGHT if join else 0.0)
                     + (F0_MATCH_BONUS
                        if f0["verdict"] == "compatible" else 0.0))
            detail["score"] = round(score, 3)
        pairs[(sid, fid)] = detail
        return detail

    for sid in c1_voices:
        for fid in c1_faces:
            _pair(sid, fid)

    # ── one-to-one assignment, confident-first ───────────────────────
    def _ranked(sid, free_f):
        cands = sorted((pairs[(sid, fid)] for fid in free_f
                        if pairs[(sid, fid)]["score"] is not None),
                       key=lambda d: -d["score"])
        return cands

    def _take(d, margin, tier=1):
        matches.append({"speaker_id": d["speaker_id"],
                        "face_person_id": d["face_person_id"],
                        "status": "confirmed",
                        "mechanism": "class1_assignment",
                        "tier": tier,
                        "score": d["score"],
                        "margin": round(margin, 3),
                        "lift": d["lift"], "solo_sec": d["solo_sec"],
                        "onset_share": d["onset_share"],
                        "floor_share": d["floor_share"],
                        "name_join": d["name_join"], "f0": d["f0"],
                        **({"mixed_voice": True}
                           if d.get("mixed_voice") else {})})
        free_v.remove(d["speaker_id"])
        free_f.discard(d["face_person_id"])

    matches, abstentions = [], []
    free_v, free_f = list(c1_voices), set(c1_faces)
    while free_v and free_f:
        # confident evidence FIRST: elimination certainty is about
        # striking pairs, never about priority — a weaker voice whose
        # alternatives were all struck must not grab a face another
        # voice owns by evidence (measured on the replay: a tail female
        # cluster force-took the lead anchor's face before her real
        # cluster's confident claim).
        best = None                  # (score, margin, detail)
        for sid in free_v:
            cands = _ranked(sid, free_f)
            if not cands or cands[0]["score"] < CLASS1_MIN_SCORE:
                continue
            margin = cands[0]["score"] - (cands[1]["score"]
                                          if len(cands) > 1 else 0.0)
            if margin < CLASS1_MIN_MARGIN:
                continue             # not confident YET — a later lock
            if best is None or cands[0]["score"] > best[0]:
                best = (cands[0]["score"], margin, cands[0])
        if best is None:
            break                    # no confident pair remains
        _take(best[2], best[1])
        # (the forced-by-elimination endgame that lived here was
        # deleted 2026-09-24: zero firings corpus-wide once onset and
        # floor-holding joined the assertion set — eliminations act
        # through strikes and margins, never a discounted bar)

    for sid in free_v:               # the no-silent-exit guarantee
        cands = _ranked(sid, free_f)[:2]
        struck = [pairs[(sid, fid)] for fid in c1_faces
                  if pairs[(sid, fid)].get("eliminated")]
        reason = ("no_candidate" if not cands
                  else "below_score" if cands[0]["score"] < CLASS1_MIN_SCORE
                  else "thin_margin")
        # both ledgers ride the dossier: what it might be (survivors,
        # ranked) and what it is NOT (struck pairs, each with why)
        abstentions.append({"speaker_id": sid, "reason": reason,
                            "candidates": cands,
                            **({"eliminated": struck} if struck else {})})

    # ── tier 2: the same machinery over the residue ──────────────────
    # Voices under the class-1 floor (plus tier-1 abstainers, who now
    # see a wider face pool) against unmatched class-1 faces and
    # unmatched NAMED live faces. Same confident bar, one-to-one; no
    # dossiers (the no-silent-exit guarantee is about the heads).
    # NOT widened to unnamed faces (measured 2026-09-24: people often
    # exist as a named track PLUS an uncoalesced unnamed twin, and the
    # twin can outrank the named track — one-to-one then orphans the
    # named face; a 799s correspondent went silent_face. Binding the
    # unnamed tail needs a twin-aware rule first.)
    taken_f = {m["face_person_id"] for m in matches}
    free_v2 = t2_voices + [sid for sid in free_v]
    free_f2 = set(free_f) | {f["face_person_id"] for f in live
                             if f.get("name")
                             and f["face_person_id"] not in taken_f
                             and f["face_person_id"] not in free_f}
    for sid in free_v2:
        for fid in free_f2:
            _pair(sid, fid)
    while free_v2 and free_f2:
        best = None
        for sid in free_v2:
            cands = _ranked(sid, free_f2)
            if not cands or cands[0]["score"] < CLASS1_MIN_SCORE:
                continue
            margin = cands[0]["score"] - (cands[1]["score"]
                                          if len(cands) > 1 else 0.0)
            if margin < CLASS1_MIN_MARGIN:
                continue
            if best is None or cands[0]["score"] > best[0]:
                best = (cands[0]["score"], margin, cands[0])
        if best is None:
            break
        d = best[2]
        matches.append({"speaker_id": d["speaker_id"],
                        "face_person_id": d["face_person_id"],
                        "status": "confirmed",
                        "mechanism": "class1_assignment",
                        "tier": 2,
                        "score": d["score"],
                        "margin": round(best[1], 3),
                        "lift": d["lift"], "solo_sec": d["solo_sec"],
                        "onset_share": d["onset_share"],
                        "floor_share": d["floor_share"],
                        "name_join": d["name_join"], "f0": d["f0"],
                        **({"mixed_voice": True}
                           if d.get("mixed_voice") else {})})
        free_v2.remove(d["speaker_id"])
        free_f2.discard(d["face_person_id"])

    # a tier-1 abstainer that tier-2 matched DID exit with a binding —
    # its abstention record is stale; same for a class-1 face taken in
    # tier 2 (the guarantees stay truthful)
    matched_v = {m["speaker_id"] for m in matches}
    matched_f = {m["face_person_id"] for m in matches}
    abstentions = [a for a in abstentions
                   if a["speaker_id"] not in matched_v]
    return {"speakers": c1_voices, "faces": c1_faces,
            "matches": matches, "abstentions": abstentions,
            "unmatched_faces": sorted(f for f in free_f
                                      if f not in matched_f)}
