"""
persons.py — assemble person entities from local identities + bindings.

Rules (README.md §2, locked):
- only CONFIRMED bindings merge identities; proposals are reported, never
  acted on (wrong merges misattribute quotes — as bad as wrong names);
- a person is NAMED only by an independent naming mechanism (face source
  name, self_id/sign_off cue, overlay name at the person's timespan);
- name CONFLICTS abstain: name=None, both candidates in aliases, type
  uncertain — a wrong name is worse than none;
- unresolved residue is first-class output;
- three-state sources block: absent input => dependent fields absent.
"""
from __future__ import annotations

from lib import critic
from lib import pb_common
from lib.pb_common import DEPICTED_SHARE_THRESHOLD, overlap_sec


def _name_from_cues(speaker_id: str, cues: list[dict]) -> dict | None:
    """Strongest direct name for a speaker from its own cues."""
    own = [c for c in cues
           if c["speaker_id"] == speaker_id
           and c["kind"] in ("self_id", "sign_off", "vocative_resolved")
           and c["confidence"] == "high"]
    if not own:
        return None
    # sign_off and self_id agreeing is ideal; disagreeing is a conflict
    names = {c["name"] for c in own}
    if len(names) > 1:
        return {"conflict": sorted(names), "evidence": own}
    # kinds ride along so finalize can tell a pure spoken ADDRESS
    # (vocative only — character-name material) from a self-claimed
    # identity (self_id/sign_off — a real conflict when it disagrees)
    return {"name": own[0]["name"], "evidence": own,
            "kinds": sorted({c["kind"] for c in own})}


def trusted_name_proposals(cues) -> dict:
    """speaker_id -> the ONE name _name_from_cues would act on
    (self_id / sign_off / vocative_resolved, high confidence, no
    conflict). The ONLY proposal set the texts_on_screen join may
    look up: the first corpus A/B (2026-10-05) fed it every cue name
    and every face name instead, and low-confidence vocative
    CANDIDATES ('Like', 'Again', 'Friday') plus celebrity face names
    matched news ticker text — junk persons appeared and nine real
    names abstained away under the flood. A proposal the bind itself
    would not act on must never be corroborated."""
    per: dict = {}
    for c in cues or []:
        if c.get("kind") in ("self_id", "sign_off", "vocative_resolved") \
                and c.get("confidence") == "high" and c.get("name"):
            per.setdefault(c["speaker_id"], set()).add(c["name"])
    return {sid: next(iter(ns))
            for sid, ns in per.items() if len(ns) == 1}


def _canonical_spelling(forms: list[str]) -> str:
    """One display form for spellings that casefold-join: natural
    casing beats ALL-CAPS (chyrons shout, vocatives and credits
    don't), then the most frequent, then first-seen. Exists because
    the finalize vote counted 'JOANNE' and 'Joanne' as rival names
    and ABSTAINED a fully-evidenced contestant (measured 2026-10-05,
    quiz asset: 7 chyron reads + a resolved vocative -> name None)."""
    counts: dict[str, int] = {}
    for f in forms:
        counts[f] = counts.get(f, 0) + 1
    cased = [f for f in counts if f != f.upper()]
    pool = cased or list(counts)
    return max(pool, key=lambda f: counts[f])


def canonicalize_name_spellings(names: list[dict]) -> None:
    """Rewrite each name record's display form so spellings that
    casefold-join become ONE name before any vote or conflict check.
    Mutates in place; evidence keeps the verbatim strings."""
    groups: dict[str, list[str]] = {}
    for n in names:
        if n.get("name"):
            groups.setdefault(str(n["name"]).casefold(),
                              []).append(str(n["name"]))
    display = {k: _canonical_spelling(v) for k, v in groups.items()}
    for n in names:
        if n.get("name"):
            n["name"] = display[str(n["name"]).casefold()]


def name_solo_chyron_faces(faces, overlay_cues) -> int:
    """Name each still-unnamed face that is the SOLE live face on
    screen under a chyron (mutating the face records, BEFORE class-1
    runs — its tier 2 binds voices to named live faces, which is how
    a correspondent speaking over their own lower third gets their
    voice back after the subject-first attach removed the old
    speaker-attach shortcut). Two different chyron names over the same
    face abstain; an already-named face (registry/celebrity) never
    changes. Returns the number of faces named."""
    if not faces or not overlay_cues:
        return 0
    live = [f for f in faces
            if (f.get("depicted_share") or 0.0) < DEPICTED_SHARE_THRESHOLD]
    named = 0
    for f in live:
        if f.get("name"):
            continue
        hits: dict = {}       # casefold key -> verbatim forms seen
        for oc in overlay_cues:
            on = [g for g in live
                  if any(overlap_sec(a, b, oc["t0"], oc["t1"]) > 0
                         for a, b in g.get("spans") or [])]
            if len(on) == 1 and on[0] is f:
                nm = str(oc["name"]).strip()
                hits.setdefault(nm.casefold(), []).append(nm)
        # repetition is the corroboration (the vocative-voting shape):
        # a lower third REPEATS over its subject; a press-clip banner
        # or B-roll super shows once during someone's cutaway —
        # single-hit naming measurably misnamed a co-anchor's face
        # from one clip banner (the dominant_name fixture class).
        # Casefold-keyed: 'JOANNE' and 'Joanne' are one chyron.
        if len(hits) == 1:
            forms = next(iter(hits.values()))
            if len(forms) >= pb_common.VOCATIVE_MIN_VOTES:
                f["name"] = _canonical_spelling(forms)
                f["name_source"] = "overlay_chyron"
                named += 1
    return named


def build_persons(transcript, faces, cooccur_out, cues, overlay_cues,
                  sources: dict) -> dict:
    """-> persons.json shape. Every argument except sources may be None
    (absent input)."""
    turns = (transcript or {}).get("turns") or []
    speakers = (transcript or {}).get("speakers") or {}
    overlaps = (transcript or {}).get("overlap_segments") or []
    faces = faces or []
    cues = cues or []
    overlay_cues = overlay_cues or []
    bindings = (cooccur_out or {}).get("bindings") or []
    speaker_class = (cooccur_out or {}).get("speaker_class") or {}

    # ── seed one person per local identity, merge on confirmed bindings ──
    persons: list[dict] = []
    by_speaker: dict[str, dict] = {}
    by_face: dict[int, dict] = {}

    for f in faces:
        # coalesced tracker duplicates keep every original id in
        # face_ids so remap tables still cover them, and the coalesce is
        # AUDITABLE evidence (the recognizer's own identity assertion)
        fids = [f["face_person_id"]] + list(f.get("coalesced_ids") or [])
        p = {"speaker_ids": [], "face_ids": fids,
             "names": [], "evidence": [], "screen_sec": f["screen_sec"],
             "speech_sec": 0.0}
        if f.get("coalesced_ids"):
            p["evidence"].append({
                "mechanism": "recognizer_coalesce", "face_ids": fids,
                # typed: which key space(s) asserted the identity
                "recognizer_ids": f.get("recognizer_ids") or {}})
        if f.get("name"):
            p["names"].append({"name": f["name"],
                               "mechanism": f.get("name_source") or "face_source"})
        for fid in fids:
            by_face[fid] = p
        persons.append(p)

    for sid, sec in speakers.items():
        p = {"speaker_ids": [sid], "face_ids": [], "names": [],
             "evidence": [], "screen_sec": 0.0, "speech_sec": sec}
        named = _name_from_cues(sid, cues)
        if named and "name" in named:
            p["names"].append({"name": named["name"],
                               "mechanism": "transcript_cue",
                               "kinds": named.get("kinds")})
            p["evidence"].extend(named["evidence"])
        elif named:
            p["names"].extend({"name": n, "mechanism": "transcript_cue",
                               "conflict": True} for n in named["conflict"])
            p["evidence"].extend(named["evidence"])
        by_speaker[sid] = p
        persons.append(p)

    for b in bindings:
        if b["status"] != "confirmed":
            continue
        ps, pf = by_speaker.get(b["speaker_id"]), by_face.get(b["face_person_id"])
        if not ps or not pf or ps is pf:
            continue
        # merge face person into speaker person
        ps["face_ids"].extend(pf["face_ids"])
        ps["names"].extend(pf["names"])
        ps["evidence"].append(b)
        ps["screen_sec"] += pf["screen_sec"]
        for fid in pf["face_ids"]:
            by_face[fid] = ps
        persons.remove(pf)

    # ── overlay (chyron) names: attach to the SUBJECT shown, not the
    # voice heard (2026-09-30 spend audit made the old rule's failure
    # concrete: a chyron names the person it introduces, who during
    # introductions is exactly NOT the speaker — "joining us now, ..."
    # runs over the GUEST's lower third while the ANCHOR talks, and on
    # over-merged clusters even the guest's own speech credits the
    # anchor. Speaker-attach handed guests' chyrons to the anchor/host
    # as junk aliases on every 2h news and quiz asset; one anchor
    # carried 15 guest names, including a head of state.)
    #
    # Precedence: the single live (non-majority-depicted) face on
    # screen during the chyron IS the subject; a multi-face moment
    # (desk two-shot) resolves only by token-matching a face's own
    # source name against the chyron, else abstains; a FACELESS chyron
    # (voice-over b-roll) falls back to the old single-active-speaker
    # rule, with cross-talk widening the ambiguity set to abstention
    # as before (the RNC failure class).
    face_spans = [(a, b, f["face_person_id"]) for f in faces
                  if (f.get("depicted_share") or 0.0)
                  < DEPICTED_SHARE_THRESHOLD
                  for a, b in f.get("spans") or []]
    for oc in overlay_cues:
        on_screen = {fid for a, b, fid in face_spans
                     if overlap_sec(a, b, oc["t0"], oc["t1"]) > 0}
        target = None
        basis = None
        if len(on_screen) == 1:
            target = by_face.get(next(iter(on_screen)))
            basis = "face_on_screen"
        elif len(on_screen) > 1:
            toks = frozenset(str(oc["name"]).casefold().split())
            named = [fid for fid in on_screen
                     if (lambda p: p and any(
                         frozenset(str(n["name"]).casefold().split())
                         & toks for n in p["names"]))(by_face.get(fid))]
            if len(named) == 1:
                target = by_face.get(named[0])
                basis = "face_name_match"
            # several faces, none/many matching: abstain — a two-shot
            # chyron with no name join cannot pick its subject
        else:
            active = [t["speaker_id"] for t in turns
                      if overlap_sec(t["start"], t["end"],
                                     oc["t0"], oc["t1"]) > 0]
            active += [spk for ov in overlaps
                       if overlap_sec(float(ov["start"]),
                                      float(ov["end"]),
                                      oc["t0"], oc["t1"]) > 0
                       for spk in ov.get("speakers") or []]
            if len(set(active)) == 1:
                target = by_speaker.get(active[0])
                basis = "speaker_active"
        if target is not None:
            target["names"].append({"name": oc["name"],
                                    "mechanism": "overlay_name"})
            target["evidence"].append(dict(oc, attach_basis=basis))

    # ── same-name cross-mechanism merge ──────────────────────────────────
    # The Mia Chen case (P0, morning news): one real person split across an
    # overlay-named voice and a collection-named face. Persons whose SOLE
    # resolved name normalizes equal are one person — a split speaker
    # cluster, a fragmented face track, or a cross-modal pair the
    # co-occurrence bar didn't confirm. Same-name collision inside ONE
    # asset is rare; the merge records its mechanism so the corpus gate
    # can measure exactly how rare.
    def _sole_name(p):
        names = {n["name"].casefold(): n["name"] for n in p["names"]
                 if not n.get("conflict")}
        if len(names) == 1 and not any(n.get("conflict") for n in p["names"]):
            return next(iter(names))
        return None

    speech_spans: dict[str, list] = {}
    for t in turns:
        speech_spans.setdefault(t["speaker_id"], []).append(
            (t["start"], t["end"]))
    face_spans = {f["face_person_id"]: f["spans"] for f in faces}

    def _weld(tgt, p, record, rename_to=None):
        """ONE home for merge bookkeeping (same-name and subset merges
        drifted apart as copies — review 2026-09-23). Applies the critic
        veto first: prevention beats repair, both sides carry the
        reason, and critique() demotes downstream. Returns True when the
        weld happened."""
        veto = critic.merge_veto(tgt, p, speech_spans, face_spans)
        if veto:
            veto = {**veto, "name": record["name"],
                    "action": "merge_blocked"}
            vetoes.append(veto)
            tgt["evidence"].append({"mechanism": "critic_veto", **veto})
            p["evidence"].append({"mechanism": "critic_veto", **veto})
            return False
        if rename_to is not None:
            for n in p["names"]:
                n["name"] = rename_to
        tgt["speaker_ids"].extend(p["speaker_ids"])
        tgt["face_ids"].extend(p["face_ids"])
        tgt["names"].extend(p["names"])
        tgt["evidence"].extend(p["evidence"])
        tgt["evidence"].append(record)
        tgt["screen_sec"] += p["screen_sec"]
        tgt["speech_sec"] += p["speech_sec"]
        for sid in p["speaker_ids"]:
            by_speaker[sid] = tgt
        for fid in p["face_ids"]:
            by_face[fid] = tgt
        persons.remove(p)
        return True

    vetoes: list[dict] = []
    by_name: dict[str, dict] = {}
    for p in list(persons):
        key = _sole_name(p)
        if key is None:
            continue
        tgt = by_name.get(key)
        if tgt is None:
            by_name[key] = p
            continue
        _weld(tgt, p, {"mechanism": "name_agreement", "name": key})

    # ── subset-name merge (a first name onto the one full name) ─────────
    # The measured co-anchor split (2026-09-23): the face person is
    # collection-named "Craig Merton" while the desk VOICE — whose
    # co-occurrence never even proposes on split-box/desk shots — gets
    # vocative-named "Craig", and the exact-name merge above cannot join
    # them. A sole name whose tokens are a STRICT subset of exactly ONE
    # other person's sole name is that person; UNIQUENESS is the
    # corroboration (two supersets = an ambiguous first name = abstain),
    # and the same critic veto applies. COMPLEMENTARY HALVES ONLY: the
    # merge may complete a face-only person with its missing voice (or
    # the reverse) — never add a second voice or face to a person that
    # already has one. Without this, a scripted programme where a
    # CHARACTER shares an actor's first name would weld the character's
    # voice onto that actor ("Jack" onto Jack Doe) — the merge must
    # hold on every drama, not just the measured broadcast. The subset
    # person's name entries are rewritten to the full name so finalize
    # sees agreement, not a conflict; the short form stays in evidence.
    for p in list(persons):
        key = _sole_name(p)
        if key is None:
            continue
        toks = frozenset(key.split())
        supers = [q for q in persons
                  if q is not p and _sole_name(q)
                  and toks < frozenset(_sole_name(q).split())]
        if len(supers) != 1:
            continue
        tgt = supers[0]
        if (p["speaker_ids"] and tgt["speaker_ids"]) \
                or (p["face_ids"] and tgt["face_ids"]):
            continue                     # not complementary halves
        # display-cased: _sole_name is the casefolded key, and finalize's
        # name agreement is case-sensitive — rewriting with the folded
        # form would read as a conflict and abstain the merged person
        full = next(n["name"] for n in tgt["names"]
                    if n["name"].casefold() == _sole_name(tgt))
        _weld(tgt, p, {"mechanism": "subset_name",
                       "name": key, "into": full}, rename_to=full)

    # ── finalize: resolve names, assign types ────────────────────────────
    # majority-depicted faces (sources.mark_depicted_faces) — used only
    # for the TYPE label below, where majority-of-total is what the
    # label means (co-occurrence already credits live spans only)
    depicted_face_ids = {
        fid for f in faces
        if (f.get("depicted_share") or 0.0) >= DEPICTED_SHARE_THRESHOLD
        for fid in [f["face_person_id"], *(f.get("coalesced_ids") or [])]}
    # image-registry mechanisms name the PERSON (the registry identity —
    # an actor, a presenter); a vocative names what people on screen CALL
    # them. When one person carries exactly one of each and they are not
    # the same name, that is not a conflict — it is the actor/character
    # split (measured 2026-09-23: dialogue carries the character names
    # the credits' billing card lacks). The registry name stays `name`,
    # the spoken address becomes `character_name` (mechanism
    # spoken_address); a later credits read overwrites it — deterministic
    # credits beat inferred address.
    REGISTRY_MECHS = {"collection", "celebrity",
                      "collection_recovery", "celebrity_recovery"}
    out = []
    for i, p in enumerate(persons):
        # ONE spelling per casefold-joined name BEFORE any vote or
        # conflict check — a chyron's ALL-CAPS read and a vocative's
        # natural casing are the same name, not rivals
        canonicalize_name_spellings(p["names"])
        uniq = {}
        for n in p["names"]:
            uniq.setdefault(n["name"], []).append(n["mechanism"])
        spoken_character = None
        if len(uniq) == 2 and not any(n.get("conflict") for n in p["names"]):
            (n1, m1), (n2, m2) = uniq.items()
            pure_voc = {n["name"] for n in p["names"]
                        if n.get("kinds") == ["vocative_resolved"]}
            reg = n1 if set(m1) <= REGISTRY_MECHS else (
                n2 if set(m2) <= REGISTRY_MECHS else None)
            voc = n1 if n1 in pure_voc else (n2 if n2 in pure_voc else None)
            if reg and voc and reg != voc:
                uniq = {reg: uniq[reg]}
                if frozenset(voc.casefold().split()) \
                        <= frozenset(reg.casefold().split()):
                    # the spoken address is PART of the registry name
                    # ("Craig" on Craig Merton — measured 2026-09-23
                    # once class1 put the voice and face on one person
                    # at bind): AGREEMENT, the same token-subset
                    # corroboration the subset-name merge uses — not
                    # an actor/character split
                    pass
                else:
                    spoken_character = voc
                    p["evidence"].append({"mechanism": "spoken_address",
                                          "name": reg, "character": voc})
        # class-1 registry precedence (measured 2026-09-23): an
        # over-merged voice cluster collects spoken cues and overlay
        # names for everyone it absorbed, and pooling them with the
        # matched face's registry name read as a conflict — abstaining
        # away the very lead the class-1 weld had just recovered (a
        # broadcast's anchor, a drama's title character). The
        # assignment IS the corroboration that this face is this
        # voice, and the recognizer registry is the recorded trust
        # tier above spoken/chyron attach on merged clusters — so the
        # sole registry name wins and every other name demotes to an
        # alias (kept: real evidence about the CLUSTER, just not this
        # person's identity). Conflicts among the DEMOTED names don't
        # block it — they are exactly the absorbed-content noise this
        # rule exists for; two registry names disagreeing is a real
        # conflict and abstains as ever (len(regs) != 1).
        forced = None
        if len(uniq) > 1:
            regs = [n for n, mechs in uniq.items()
                    if set(mechs) & REGISTRY_MECHS]
            welded = any(isinstance(e, dict)
                         and e.get("mechanism") == "class1_assignment"
                         for e in p["evidence"])
            if len(regs) == 1 and welded:
                forced = regs[0]
                p["evidence"].append(
                    {"mechanism": "class1_name_precedence",
                     "name": forced,
                     "demoted": sorted(set(uniq) - {forced})})
            elif not regs and welded:
                # dominant repeated name (measured 2026-09-24: a host
                # whose face the collection lacks carried his own
                # chyron FOUR times among the 12 one-off names his
                # over-merged cluster collected — and abstained). A
                # host's chyron repeats across the programme; a
                # guest's shows once. Same votes+lead shape as
                # vocative resolution: repetition is the
                # corroboration a registry name would have been.
                votes: dict[str, int] = {}
                for n in p["names"]:
                    votes[n["name"]] = votes.get(n["name"], 0) + 1
                ranked = sorted(votes.items(), key=lambda kv: -kv[1])
                top_name, top_n = ranked[0]
                runner = ranked[1][1] if len(ranked) > 1 else 0
                if top_n >= pb_common.VOCATIVE_MIN_VOTES and \
                        (not runner
                         or top_n / runner >= pb_common.VOCATIVE_MIN_LEAD):
                    forced = top_name
                    p["evidence"].append(
                        {"mechanism": "dominant_name", "name": forced,
                         "votes": top_n, "runner_up_votes": runner,
                         "demoted": sorted(set(uniq) - {forced})})
        if forced:
            name = forced
            aliases = sorted(set(uniq) - {forced})
        elif len(uniq) == 1 and not any(n.get("conflict")
                                        for n in p["names"]):
            name = next(iter(uniq))
            aliases = []
        else:
            name = None                      # conflict or nothing: abstain
            aliases = sorted(uniq)
        all_depicted = bool(p["face_ids"]) and all(
            fid in depicted_face_ids for fid in p["face_ids"])
        if p["speaker_ids"] and p["face_ids"] and not all_depicted:
            ptype = "on_camera_speaker"
        elif p["speaker_ids"] and all_depicted:
            # a speaker whose only faces are inside displayed media
            # (phone interview over their photo, joined by the same-name
            # merge): the VOICE is real, the on-camera presence is not —
            # PLAN's taxonomy says that is voice_only (review,
            # 2026-09-06: the first branch used to claim on_camera)
            ptype = "voice_only"
        elif p["speaker_ids"]:
            sid = p["speaker_ids"][0]
            ptype = ("voice_only"
                     if speaker_class.get(sid) == "voice_only"
                     else "uncertain")
        elif all_depicted:
            # every track majority-inside displayed media (the guard) —
            # a photo/monitor/file-footage person: the NAME may be true,
            # the presence is not physical (never binds to speech; the
            # type is what tells a consumer)
            ptype = "depicted_only"
        else:
            ptype = "silent_face"
        # person-level presence timeline (2026-09-08): member faces'
        # spans merged, in facemetadata's begin_ms/end_ms shape — the
        # consolidated answer to "when is this person on screen", so a
        # consumer needing per-window presence (audio-description's
        # people_on_screen) can read it HERE instead of re-joining the
        # raw face-recognition clusters. On a FACELESS run (no face
        # source at all — GDPR posture) the basis degrades HONESTLY to
        # the person's merged speaker turn spans, and appearances_basis
        # says so: the PRODUCER owns presence semantics, consumers
        # never re-derive (review, 2026-09-08).
        if faces:
            spans = [sp for fid in p["face_ids"]
                     for sp in face_spans.get(fid) or []]
            basis = "face"
        else:
            spans = [sp for sid in p["speaker_ids"]
                     for sp in speech_spans.get(sid) or []]
            basis = "speech"
        merged: list = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        row = {"person_id": i, "type": ptype, "name": name,
               "aliases": aliases,
               "speaker_ids": p["speaker_ids"],
               "face_ids": p["face_ids"],
               "speech_sec": round(p["speech_sec"], 2),
               "screen_sec": round(p["screen_sec"], 2),
               "appearances": [{"begin_ms": round(a * 1000),
                                "end_ms": round(b * 1000)}
                               for a, b in merged],
               "appearances_basis": basis,
               "name_mechanisms": sorted(
                   {m for n in p["names"] for m in [n["mechanism"]]}),
               "evidence": p["evidence"]}
        if name and spoken_character:
            row["character_name"] = spoken_character
        out.append(row)

    unresolved = {
        "speakers": sorted(s for s, p in by_speaker.items()
                           if not p["face_ids"] and not p["names"]),
        "faces": sorted(f["face_person_id"] for f in faces
                        if not by_face[f["face_person_id"]]["speaker_ids"]
                        and not by_face[f["face_person_id"]]["names"]),
    }
    doc = {"sources": sources, "persons": out, "bindings": bindings,
           "cues": cues, "unresolved": unresolved, "vetoes": vetoes}
    # post-assembly coherence sweep — demote-only (lib/critic.py)
    return critic.critique(doc)


def _ms_spans(p: dict) -> list[tuple[float, float]]:
    return [(a["begin_ms"] / 1000.0, a["end_ms"] / 1000.0)
            for a in p.get("appearances") or []]


def _merge_ms(a: list[dict], b: list[dict]) -> list[dict]:
    spans = sorted([x["begin_ms"], x["end_ms"]] for x in (a or []) + (b or []))
    out: list = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [{"begin_ms": s, "end_ms": e} for s, e in out]


def _absorb(keeper: dict, voice: dict, mechanism: str, doc: dict) -> None:
    """Weld a voice person into a collection-named face person, in the
    EXPORTED shape (this runs post-adjudication, on persons.json)."""
    keeper["speaker_ids"] = sorted(set(keeper["speaker_ids"]
                                       + voice["speaker_ids"]))
    keeper["face_ids"] = sorted(set(keeper["face_ids"]
                                    + voice["face_ids"]))
    keeper["speech_sec"] = round(keeper.get("speech_sec", 0.0)
                                 + voice.get("speech_sec", 0.0), 2)
    keeper["screen_sec"] = round(keeper.get("screen_sec", 0.0)
                                 + voice.get("screen_sec", 0.0), 2)
    keeper["appearances"] = _merge_ms(keeper.get("appearances"),
                                      voice.get("appearances"))
    keeper["aliases"] = sorted(set((keeper.get("aliases") or [])
                                   + (voice.get("aliases") or [])
                                   + ([voice["name"]] if voice.get("name")
                                      else []))
                               - {keeper.get("name")})
    keeper["name_mechanisms"] = sorted(
        set(keeper.get("name_mechanisms") or [])
        | set(voice.get("name_mechanisms") or []) | {mechanism})
    keeper["evidence"] = ((keeper.get("evidence") or [])
                          + (voice.get("evidence") or [])
                          + [{"mechanism": mechanism,
                              "absorbed_person_id": voice["person_id"],
                              "name": keeper.get("name")}])
    keeper["type"] = "on_camera_speaker"
    doc["persons"].remove(voice)


def accommodate_overclustering(doc: dict) -> dict:
    """Post-adjudication accommodation of diarization OVER-CLUSTERING
    (user direction, 2026-09-09: over-merge is an input condition —
    accommodate it, prefer the curated collection). Runs on the
    EXPORTED persons.json between apply_verdicts and critique. ONE
    pass, merging INTO recognizer-named persons only (trust tiers:
    collection, then celebrity — see below):

    SAME-NAME ABSORPTION — the build-time Mia Chen merge re-run now
    that adjudication has resolved voice persons' names: a voice
    person adjudicated/overlay-named to a collection person's exact
    name is that person (the collision critic would otherwise fire;
    with the collection preference it would keep the face and orphan
    the VOICE — this pass welds them instead).

    TOP-TALKER RECOVERY (pass 2) was DELETED 2026-09-24: measured on
    the 10-asset corpus, 34 of its 35 welds were already made by the
    class-1/2 matcher at bind time — and it produced the one proven
    misbinding (a guest's voice welded onto a host's face by raw span
    overlap, no eliminations, no one-to-one). The matcher is the one
    binding engine now; this pass only serves names that ARRIVE in
    round 2.

    The contradiction check stays: two faces on screen together are two
    people whatever the names say (> MERGE_FACE_OVERLAP_TOLERANCE_SEC
    of appearance overlap blocks, veto recorded)."""
    doc.setdefault("vetoes", [])
    # trust-tiered recognizer-named faces (user, 2026-09-09): collection
    # first, then celebrity — a GDPR posture forbids the face collection
    # (collection_id "") but may allow celebrity detection, so celebrity
    # is the ONLY face-naming tier those runs have. Chyron names attach
    # to speakers, never faces, so no third tier exists by construction.
    def _tier(mech):
        return [p for p in doc["persons"]
                if p.get("name") and mech in (p.get("name_mechanisms")
                                              or [])]
    tiers = [("collection", _tier("collection"))]
    tiers.append(("celebrity", [p for p in _tier("celebrity")
                                if p not in tiers[0][1]]))
    coll = [p for _, tier in tiers for p in tier]
    if not coll:
        return doc

    def face_conflict(a, b):
        ov = pb_common.spans_overlap_sec(_ms_spans(a), _ms_spans(b))
        return ov if ov > pb_common.MERGE_FACE_OVERLAP_TOLERANCE_SEC \
            else None

    # ── pass 1: same-name absorption ─────────────────────────────────
    for keeper in coll:
        if keeper not in doc["persons"]:
            continue                    # itself absorbed by an earlier pass
        key = keeper["name"].casefold()
        for voice in [p for p in list(doc["persons"])
                      if p is not keeper and p.get("speaker_ids")
                      and not set(p.get("speaker_ids"))
                      & set(keeper["speaker_ids"])
                      and (p.get("name") or "").casefold() == key]:
            ov = face_conflict(keeper, voice)
            if ov is not None:
                doc["vetoes"].append(
                    {"check": "face_self_overlap", "name": keeper["name"],
                     "action": "absorb_blocked", "overlap_sec": round(ov, 2),
                     "person_ids": [keeper["person_id"],
                                    voice["person_id"]]})
                continue
            _absorb(keeper, voice, "name_agreement", doc)

    return doc


def enforce_speech_coherence(doc: dict, speakers: dict) -> dict:
    """The invariant (2026-09-23 forensics): person.speech_sec MUST equal
    the sum of its speaker_ids' transcript seconds. A shipped run
    (alpha.20260922a tdy) carried 53/76 single-speaker persons with
    ANOTHER cluster's seconds — the corrupted intermediate was
    overwritten in place, so the mechanism was unrecoverable; current
    code reproduces coherent docs, but the truth is DERIVABLE, so every
    write point recomputes it instead of trusting arithmetic that once
    lied. A repair beyond tolerance is recorded on the person and in
    doc.vetoes — silent self-healing would hide the recurrence this
    exists to catch."""
    if not speakers:
        return doc
    doc.setdefault("vetoes", [])
    for p in doc.get("persons") or []:
        sids = p.get("speaker_ids") or []
        if not sids:
            continue
        truth = round(sum(speakers.get(s, 0.0) for s in sids), 2)
        got = p.get("speech_sec")
        if got is None or abs(truth - got) <= max(0.5, 0.02 * truth):
            p["speech_sec"] = truth
            continue
        repair = {"check": "speech_seconds_incoherent",
                  "person_ids": [p.get("person_id")],
                  "was": got, "recomputed": truth}
        doc["vetoes"].append(repair)
        p.setdefault("evidence", []).append(
            {"mechanism": "coherence_repair", **repair})
        p["speech_sec"] = truth
    return doc


def remap_tables(persons_doc: dict) -> tuple[dict, dict]:
    """persons.json -> (speaker_map, face_map). Only meaningful joined to
    the exact sibling runs named in sources (README.md §1 provenance pin)."""
    smap, fmap = {}, {}
    for p in persons_doc["persons"]:
        entry = {"person_id": p["person_id"], "name": p["name"],
                 "type": p["type"]}
        for sid in p["speaker_ids"]:
            smap[sid] = entry
        for fid in p["face_ids"]:
            fmap[str(fid)] = entry
    return smap, fmap
