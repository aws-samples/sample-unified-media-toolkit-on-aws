"""Layer A (eligibility — hard filter, no taste) + FilterRank
(archetype queries over the shot lattice -> ranked candidate slates).

Ported from the validated poc/recommend.py, behavior-identical, plus
the two settled additions: program-structure span exclusion and
cross-archetype dedup. Pure stdlib on plain dicts — shots.json rows,
shot_metadata.jsonl rows, cast persons.
"""

from lib.profiles import (ARCHETYPES, ARTWORK_SHOT_KINDS, HERO_CAST_N,
                      MIN_SHOT_SEC, RANK_DURATION_CAP_SEC, TOP_CAST_N)


def top_cast(persons, n=TOP_CAST_N):
    """Names by screen time, descending — the derived cast signal."""
    named = [p for p in persons or [] if p.get("name")]
    named.sort(key=lambda p: -(p.get("screen_sec") or 0))
    return [p["name"] for p in named[:n]]


def top_faces(fr_doc, n=TOP_CAST_N):
    """face-recognition FALLBACK identity (used only when
    cast-resolution is absent): top-N persons by tracked screen time
    from facemetadata.json. Artwork needs 'the same prominent person',
    not a name — an unnamed face cluster rides under an anonymous
    label (its source person_id), which is honest and still ranks a
    hero slate; only the brief's must-include needs real names.
    Returns (labels, spans_by_label); spans are (t0, t1) seconds."""
    people = []
    for p in fr_doc.get("people") or []:
        spans = [(a["begin_ms"] / 1000.0, a["end_ms"] / 1000.0)
                 for a in p.get("appearances") or []]
        if not spans:
            continue
        # placeholder names are NOT names — only carry a name the
        # source actually asserted (cast's parse_faces rule)
        label = p.get("name") if p.get("named") \
            else f"person_{p['person_id']}"
        screen = float(p.get("total_screen_time_sec") or 0.0) \
            or sum(b - a for a, b in spans)
        people.append((screen, label, spans))
    people.sort(key=lambda x: -x[0])
    labels = [label for _, label, _ in people[:n]]
    return labels, {label: spans for _, label, spans in people[:n]}


def face_presence(shots, spans_by_label):
    """{shot_id: set(labels whose face tracks overlap the shot)} — the
    temporal join that replaces the metadata name join when identity
    comes from face-recognition (face clusters never string-match
    shot-metadata's people field)."""
    presence = {}
    for sid, s in shots.items():
        t0, t1 = s["start_time"], s["end_time"]
        hit = {label for label, spans in spans_by_label.items()
               if any(a < t1 and t0 < b for a, b in spans)}
        if hit:
            presence[sid] = hit
    return presence


def reject_reason(shot, doc, spoiler_cut_sec, excluded_spans=()):
    """Layer A: can this shot EVER be artwork? Genre-independent.
    Returns the FIRST failing clause's name (the rejection accounting
    the output contract surfaces), or None for an eligible shot.

    `shot_kind` admits normal AND graphics_overlay (news support, user
    direction 2026-10-05): SME types 80%+ of broadcast-news footage
    graphics_overlay — real footage wearing channel furniture (ticker,
    lower third, bug), measured 417/513 and 1028/1389 shots on two
    news assets vs 15/538 on a drama — so excluding the kind excluded
    the genre. On-screen text is likewise no longer a disqualifier:
    it is the TIER key (see tier()) — clean plates strictly outrank
    furniture frames, so content that has clean plates behaves exactly
    as before."""
    if not shot or not doc:
        return "missing"
    if doc.get("shot_kind") not in ARTWORK_SHOT_KINDS:
        return "shot_kind"
    if doc.get("unreadable_text_present"):
        return "unreadable_text"
    if doc.get("brands"):
        return "brands"
    if doc.get("depicted"):
        return "depicted"
    start, end = shot["start_time"], shot["end_time"]
    if end - start < MIN_SHOT_SEC:
        return "short"
    if start > spoiler_cut_sec:
        return "spoiler"
    # program-structure spans (credits/recap/cold-open) when supplied
    if any(start < b and a < end for a, b in excluded_spans):
        return "structure_span"
    return None


def eligible(shot, doc, spoiler_cut_sec, excluded_spans=()):
    return reject_reason(shot, doc, spoiler_cut_sec,
                         excluded_spans) is None


def eligibility_audit(shots, docs, spoiler_cut_sec, excluded_spans=()):
    """{clause: shots it rejected} + eligible counts per tier — why a
    slate is thin is part of the deliverable, never a silent zero."""
    out = {"eligible_clean": 0, "eligible_overlaid": 0}
    for sid, doc in docs.items():
        r = reject_reason(shots.get(sid), doc, spoiler_cut_sec,
                          excluded_spans)
        if r is None:
            out["eligible_clean" if tier(doc) == 0
                else "eligible_overlaid"] += 1
        else:
            out[r] = out.get(r, 0) + 1
    return out


def tier(doc):
    """0 = clean plate (normal, text-free) — today's standard; 1 =
    burned-in graphics (graphics_overlay, or normal with on-screen
    text). Ranking is strictly tier 0 first, so a furniture frame is
    ever selected only where a clean plate does not exist."""
    return 0 if (doc.get("shot_kind") == "normal"
                 and not doc.get("texts_on_screen")) else 1


def rank_key(shot, doc=None):
    """Clean tier first; within the furniture tier, fewer burned-in
    strings; then longer (capped), stiller, busier frames."""
    t = tier(doc) if doc is not None else 0
    return (t,
            len(doc.get("texts_on_screen") or []) if t else 0,
            -min(shot["end_time"] - shot["start_time"],
                 RANK_DURATION_CAP_SEC),
            shot.get("frame_motion_mean") or 0.0,
            -(shot.get("entropy_mean") or 0.0))


def _matches(doc, spec, cast_names, present):
    """`present` is the set of identity labels IN this shot: the
    metadata people field on the cast path, the temporal face-track
    join on the face-recognition fallback path."""
    if doc.get("camera_shot_type") not in spec["camera"]:
        return False
    if doc.get("people_count") not in spec["people_count"]:
        return False
    if spec["people"] == "lead":
        return any(n in present for n in cast_names[:HERO_CAST_N])
    if spec["people"] == "top_cast":
        return any(n in present for n in cast_names)
    return True


def select_candidates(shots, docs, cast_names, archetype_counts,
                      spoiler_cut_sec, excluded_spans=(),
                      presence=None, taken=None):
    """FilterRank: {archetype: ranked candidate slates}. A shot elected
    by one archetype is not offered to the next (cross-archetype
    dedup); archetype order follows archetype_counts. `presence`
    (from face_presence) switches shot membership from the metadata
    name join to the face-track temporal join. `taken` seeds the dedup
    set — a second identity pass (the empty-person-slate fallback)
    must not re-elect shots the first pass already placed."""
    ok = [sid for sid in docs
          if eligible(shots.get(sid), docs[sid], spoiler_cut_sec,
                      excluded_spans)]
    taken = set(taken or ())

    def _present(sid):
        return (presence.get(sid, set()) if presence is not None
                else (docs[sid].get("people") or []))

    slates = {}
    for name, count in archetype_counts.items():
        spec = ARCHETYPES[name]
        pool = sorted(
            (sid for sid in ok if sid not in taken
             and _matches(docs[sid], spec, cast_names, _present(sid))),
            key=lambda sid: rank_key(shots[sid], docs[sid]))
        if spec["people"] == "lead" and cast_names:
            # person-diverse round-robin over the top HERO_CAST_N
            # (user direction 2026-10-06): slot 1 the lead's best
            # close-up, slot 2 the co-lead's, wrapping — the judge
            # sees each top cast member, not four frames of one
            top = cast_names[:HERO_CAST_N]
            by_person = {n: [] for n in top}
            for sid in pool:
                pres = _present(sid)
                owner = next(n for n in top if n in pres)
                by_person[owner].append(sid)
            cands = []
            while len(cands) < count and any(by_person.values()):
                for n in top:
                    if by_person[n] and len(cands) < count:
                        cands.append(by_person[n].pop(0))
        else:
            cands = pool[:count]
        taken.update(cands)
        slates[name] = [_candidate(shots[sid], docs[sid])
                        for sid in cands]
    return slates


def _candidate(shot, doc):
    """One slate entry: the frame pointer plus the evidence fields
    that qualified it (the reviewable part of the contract)."""
    out = {
        "shot_id": shot["shot_id"],
        "t": round((shot["start_time"] + shot["end_time"]) / 2, 2),
        "frames_key": shot.get("frames_key"),
        "camera_shot_type": doc.get("camera_shot_type"),
        "people": doc.get("people") or [],
        "people_count": doc.get("people_count"),
        "description": doc.get("description", ""),
    }
    if tier(doc):
        # the furniture tier carries its burden verbatim — editorial
        # (and the vision judge's intrusion criterion) see exactly
        # which graphics are burned in
        out["shot_kind"] = doc.get("shot_kind")
        out["texts_on_screen"] = doc.get("texts_on_screen") or []
    return out
