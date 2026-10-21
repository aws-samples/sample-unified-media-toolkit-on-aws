"""
Narration windows: WHERE description may speak, and which windows carry a
DUTY (scene entry, reveal). Ported from the validated runner
(/tmp/ad_runner/ad_lib.py, 31 zero-hand-edit runs, 3 series, scored
against 3,800 professional cues) — every measured lesson keeps its
comment. PURE: parsed sibling artifacts in, windows + orientation out;
the states do all I/O. [The PSA-migration candidate — keep it here in
lib/, never spread through states.]
"""
from __future__ import annotations

import re

from lib import ad_common as C


def word_spans(transcript: dict) -> list:
    """Merged word-level speech spans — the WORD-level gap map (a
    sentence-level map hides mid-sentence silences)."""
    spans = []
    for w in transcript["words"]:
        a, b = float(w["start"]), float(w["end"])
        if spans and a - spans[-1][1] <= C.WORD_MERGE:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return spans


def sung_vocal_spans(stem_activity: dict | None, transcript: dict) -> list:
    """Speech-stem activity carrying NO transcript words — untranscribed
    singing (or walla). The 'avoid' policy treats these as speech."""
    act = (stem_activity or {}).get("activity", {})
    words = [(float(w["start"]), float(w["end"]))
             for w in transcript.get("words", [])]
    out = []
    for m0, m1 in act.get("speech", []):
        a, b = m0 / 1000.0, m1 / 1000.0
        if not any(w0 < b and w1 > a for w0, w1 in words):
            out.append([a, b])
    return out


def _speakers_in(sents, sc):
    return {s.get("speaker_id") for s in sents
            if sc["start_time"] <= float(s["start"]) < sc["end_time"]}


def _texture_change(activity, at, tol=1.5):
    """True if a stem's activity starts or stops within tol of the cut —
    the ear gets an audible hint that something changed."""
    return any(abs(m0 / 1000 - at) <= tol or abs(m1 / 1000 - at) <= tol
               for m0, m1 in activity)


def orientation(scenes: list, transcript: dict, spans: list,
                music: list, sfx: list) -> dict:
    """Orientation necessity per scene boundary (would a blind viewer be
    lost here?): necessity = break x (1 - audio_sufficiency). Dialogue
    BRIDGING a cut is misdirection, not continuity — it ZEROES
    sufficiency and RAISES break (measured: Meridian 198.03s, the
    office->car cut). Weights are a documented default chosen for the
    ORDERING they induce, not fitted; every boundary ships its score and
    factors so a consumer can re-threshold."""
    sents = transcript["sentences"]
    words = transcript["words"]
    out = {}
    for i, sc in enumerate(scenes):
        st = float(sc["start_time"])
        prev = scenes[i - 1] if i else None
        f, br = {}, 0.0
        if prev is None:
            br += 0.35                      # program open: everything is new
        else:
            if (sc.get("setting") or None) != (prev.get("setting") or None):
                br += 0.35
                f["setting"] = f"{prev.get('setting')} -> {sc.get('setting')}"
            if (sc.get("time_of_day") or None) != (prev.get("time_of_day") or None):
                br += 0.15
                f["time_of_day"] = (f"{prev.get('time_of_day')} -> "
                                    f"{sc.get('time_of_day')}")
            a, b = set(prev.get("people") or []), set(sc.get("people") or [])
            if a or b:
                j = len(a & b) / max(1, len(a | b))
                br += 0.25 * (1 - j)
                if j < 1:
                    f["cast"] = f"{sorted(a)} -> {sorted(b)}"
            sa, sb = _speakers_in(sents, prev), _speakers_in(sents, sc)
            if sa or sb:
                br += 0.15 * (1 - len(sa & sb) / max(1, len(sa | sb)))
        reason = sc.get("boundary_reason") or ""
        br += {"location_change": 0.25, "after_black": 0.20,
               "graphics_boundary": 0.20}.get(reason, 0.08)
        f["boundary_reason"] = reason
        br = min(1.0, br)

        # audio sufficiency: what the ear already gets
        bridged = any(s0 < st - 0.02 and s1 > st + 0.02 for s0, s1 in spans)
        suff = 0.0
        near = " ".join(w["word"] for w in words
                        if st - 2.0 <= float(w["start"]) <= st + 8.0).lower()
        places = [str(e.get("name", "")).lower()
                  for e in (sc.get("named_entities") or [])
                  if e.get("type") in ("place", "location", "organization")]
        toks = [x for x in re.split(r"[_\s]+", str(sc.get("setting") or ""))
                if len(x) > 3]
        if any(p and p in near for p in places):
            suff += 0.50
            f["place_named_in_dialogue"] = True
        elif any(x in near for x in toks):
            suff += 0.35
            f["setting_word_in_dialogue"] = True
        if _texture_change(music, st) or _texture_change(sfx, st):
            suff += 0.40
            f["audio_texture_change"] = True
        if not any(s0 - 2.0 <= st <= s1 + 2.0 for s0, s1 in spans):
            suff += 0.15
            f["silence_at_cut"] = True   # a beat of silence reads as transition
        if bridged:
            suff = 0.0
            br = min(1.0, br + 0.20)
            f["dialogue_bridged"] = True     # misdirection, not continuity
        suff = min(1.0, suff)
        out[sc["scene_id"]] = {"scene_id": sc["scene_id"], "at": round(st, 2),
                               "break": round(br, 3),
                               "audio_sufficiency": round(suff, 3),
                               "necessity": round(br * (1 - suff), 3),
                               "dialogue_bridged": bridged, "factors": f}

    # MONTAGE CLUSTERING: a maximal run of scenes each shorter than
    # MONTAGE_SCENE_MAX is one orientation event; the head carries the run's
    # duty and max necessity (measured: drama_b 54/78 breaks without this).
    head = None
    for sc in scenes:
        o = out[sc["scene_id"]]
        short = (float(sc["end_time"]) - float(sc["start_time"])
                 < C.MONTAGE_SCENE_MAX)
        if short and head is not None:
            out[head]["montage_scenes"] += 1
            out[head]["montage_end"] = round(float(sc["end_time"]), 2)
            out[head]["necessity"] = max(out[head]["necessity"],
                                         o["necessity"])
            out[head]["break"] = max(out[head]["break"], o["break"])
            o["montage_member_of"] = head
        elif short:
            head = sc["scene_id"]
            o["montage_scenes"] = 1
            o["montage_end"] = round(float(sc["end_time"]), 2)
        else:
            head = None
    for o in out.values():
        if o.get("montage_scenes", 0) == 1:   # a lone short scene is no montage
            o.pop("montage_scenes", None)
            o.pop("montage_end", None)
    return out


# viewing APPARATUS vocabulary — a reveal seen THROUGH something loses its
# sense without the apparatus. Families, so a synonym satisfies the check.
MECHANISMS = (("mirror", "reflection", "reflected"),
              ("windshield", "window", "glass", "pane"),
              ("screen", "monitor", "television", "tv", "projection"),
              ("doorway", "door", "threshold"),
              ("photograph", "photo", "picture", "portrait", "snapshot"),
              ("curtain", "blinds"))


def mark_reveals(wins: list, notes: dict, docs: dict) -> None:
    """A reveal from the watch-through claims its window: as-it-happens,
    NEVER before (a window starting before the reveal narrates it early —
    the cardinal sin). The PRIMARY never migrates; when the as-it-happens
    window is too small for apparatus + subject it emits a PRELUDE and
    the next funded window carries the substance."""
    for r in (notes.get("reveals") or []):
        try:
            R = float(r.get("sec") or 0)
        except (TypeError, ValueError):
            continue
        cands = sorted((w for w in wins
                        if R - 1.0 <= w["start"] <= R + 12.0),
                       key=lambda x: x["start"])
        if not cands:
            continue
        w = cands[0]
        w["reveal"] = str(r.get("what", ""))[:180]
        if w["honest_words"] < C.REVEAL_MIN_WORDS:
            w["reveal_prelude"] = True
            later = [x for x in wins
                     if w["end"] < x["start"] <= R + C.REVEAL_FOLLOW_MAX
                     and x["honest_words"] >= C.REVEAL_MIN_WORDS]
            if later:
                f = min(later, key=lambda x: x["start"])
                f["reveal"] = w["reveal"]
                f["reveal_followthrough"] = round(f["start"] - R, 1)
                w["reveal_late"] = f["reveal_followthrough"]
        # apparatus read from the PRIMARY window's evidence, applied to
        # every part — a prelude must know what it points through
        ev = " ".join(str(docs.get(s, {}).get("description", ""))
                      for s in w["shot_ids"]
                      + w.get("recent_shot_ids", [])).lower()
        for fam in MECHANISMS:
            if any(re.search(r"\b" + m, ev) for m in fam):
                for x in wins:
                    if x.get("reveal") == w["reveal"]:
                        x["reveal_mechanism"] = fam[0]
                        x["reveal_mechanism_family"] = list(fam)
                break


def _budget(seconds: float) -> int:
    """THE seconds->words clamp — word_budget and honest_words share one
    affordability formula (two copies of the 2/26 clamp drifted risk,
    review 2026-08-31)."""
    return max(2, min(26, int(seconds * C.WPS)))


def _honest_words(w, spans, starts):
    """ONE affordability number: the padded budget systematically deducts
    the pad from the writer's allowance; on a duty window that is the
    difference between a sentence and a stub. Measured against the true
    next-word onset, never past the next window."""
    BIG = 1e9
    onset = min((s0 for s0, s1 in spans if s0 > w["anchor"]), default=BIG)
    nxt = min((s for s in starts if s > w["start"]), default=BIG)
    end = min(onset - C.ONSET_PAD, nxt - C.SPACING, w["end"] + C.PAD)
    return _budget(end - w["anchor"])


def build_windows(evidence: dict, watch_notes: dict,
                  sung_vocal_policy: str = C.SUNG_VOCAL_DEFAULT) -> dict:
    """Windows + orientation from the sibling evidence. Returns
    {"windows": [...], "orientation": [...], "summary": {...}}."""
    t = evidence["transcript"]
    scenes = evidence["scenes"]
    shots = sorted(evidence["shots"], key=lambda s: s["start_time"])
    people = evidence["people"]
    docs = evidence["shot_docs"]
    act = (evidence.get("stem_activity") or {}).get("activity", {})
    music, sfx = act.get("music", []), act.get("sfx", [])
    duration = float(t["audio_duration_sec"])
    spans = word_spans(t)

    if sung_vocal_policy == "avoid":
        # untranscribed sung vocals close their windows, like speech
        spans = sorted(spans + sung_vocal_spans(evidence.get("stem_activity"),
                                                t))
        merged = []
        for a, b in spans:
            if merged and a - merged[-1][1] <= C.WORD_MERGE:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        spans = merged

    gaps = []
    edges = [[0.0, 0.0]] + spans + [[duration, duration]]
    for (a1, b1), (a2, b2) in zip(edges, edges[1:]):
        lo, hi = b1 + C.PAD, a2 - C.PAD
        if hi - lo >= C.MIN_USABLE_SCENE:
            gaps.append([round(lo, 2), round(hi, 2)])

    def music_frac(a, b):
        cov = sum(max(0.0, min(b, m1 / 1000) - max(a, m0 / 1000))
                  for m0, m1 in music)
        return cov / max(b - a, 0.01)

    wins = []

    def add(a, b, shot_ids, cutaways=()):
        sc = next((x for x in scenes
                   if x["start_time"] <= a < x["end_time"]), None)
        mus = music_frac(a, b)
        budget = _budget((b - a) * (C.MUSIC_BUDGET_SCALE
                                    if mus > 0.85 else 1.0))
        wins.append({
            "window_id": len(wins), "start": round(a, 2), "end": round(b, 2),
            "seconds": round(b - a, 2), "word_budget": budget,
            "scene_id": sc["scene_id"] if sc else None,
            "scene_entry": False, "entry_for_scene": None,
            "shot_ids": list(shot_ids), "cutaways": list(cutaways),
            "recent_shot_ids": [], "anchor": round(a, 2),
            "music_dominant": mus > 0.85,
            "people_on_screen": sorted({
                p["name"] for p in people
                if (p.get("named")
                    or p.get("total_screen_time_sec", 0) >= 15)
                for ap in p["appearances"]
                if ap["begin_ms"] / 1000 < b and ap["end_ms"] / 1000 > a})})

    for g0, g1 in gaps:
        # split every gap at scene cuts so entry windows BEGIN at the cut
        # (the anchor rule; a window spanning a cut hides the entry)
        cuts = [sc["start_time"] for sc in scenes
                if g0 + C.MIN_USABLE_SCENE < sc["start_time"]
                < g1 - C.MIN_USABLE_SCENE]
        segs, prev = [], g0
        for cpt in sorted(cuts) + [g1]:
            segs.append([prev, cpt])
            prev = cpt
        for a, b in segs:
            if b - a < C.MIN_USABLE_SCENE:
                continue
            if b - a <= C.LONG_GAP:
                sh = [s["shot_id"] for s in shots
                      if s["start_time"] < b and s["end_time"] > a]
                add(a, b, sh)
                continue
            # long dialogue-free stretch: per-shot windows (v3.2 rule)
            inside = [s for s in shots
                      if s["end_time"] > a and s["start_time"] < b]
            for s in inside:
                sa = max(s["start_time"] + 0.05, a)
                sb = min(s["end_time"] - 0.2, b)
                if sb - sa < C.MIN_SHOT_WIN:
                    if wins and s["shot_id"] not in wins[-1]["shot_ids"]:
                        wins[-1]["cutaways"].append(s["shot_id"])
                    continue
                add(sa, sb, [s["shot_id"]])

    _starts = sorted(w["start"] for w in wins)
    for w in wins:
        w["honest_words"] = _honest_words(w, spans, _starts)

    # ── entry duty: the CHEAPEST SUFFICIENT window, never merely the first —
    # measured on Meridian: the office->moving-car entry went to a
    # 0.72s/2-word window ("Car.") while a 2.4s/6-word window 6s later
    # carried no duty at all.
    orient = orientation(scenes, t, spans, music, sfx)
    by_start = sorted(wins, key=lambda w: w["start"])
    for sc in scenes:
        o = orient[sc["scene_id"]]
        if o.get("montage_member_of") is not None:
            continue                 # the cluster head's cue serves this cut
        st = sc["start_time"]
        nec = o["necessity"]
        horizon = (C.SCENE_SEARCH_AFTER_HIGH if nec >= C.NECESSITY_HIGH
                   else C.SCENE_SEARCH_AFTER)
        if o.get("montage_scenes"):    # search across the whole run of cuts
            horizon = max(horizon, o["montage_end"] - st + 2.0)
        cands = [w for w in by_start
                 if st - 1.0 <= w["start"] <= st + horizon
                 and w["end"] > st and not w["scene_entry"]
                 and not w.get("entry_prelude_for_scene")]
        enough = [w for w in cands
                  if w["honest_words"] >= C.MIN_GROUND_WORDS]

        def award(w, **extra):
            w["scene_entry"] = True
            w["entry_for_scene"] = sc["scene_id"]
            w["entry_necessity"] = nec
            if o.get("montage_scenes"):
                w["entry_montage_scenes"] = o["montage_scenes"]
            w.update(extra)

        def take(w, **extra):
            """Award, anchor after the cut, and hand any sub-minimum gap
            between the cut and this window the PRELUDE duty."""
            award(w, **extra)
            w["anchor"] = round(max(w["anchor"], st + 0.1), 2)
            if w["start"] - st > 1.0:
                w["late_entry"] = round(w["start"] - st, 1)
            small = [x for x in cands
                     if x["start"] < w["start"] and x["honest_words"] >= 2]
            if small:
                small[0]["entry_prelude_for_scene"] = sc["scene_id"]
                small[0]["entry_necessity"] = nec
                small[0]["anchor"] = round(max(small[0]["anchor"],
                                               st + 0.1), 2)

        # THE ESCALATION LADDER: as-it-happens outranks well-funded, and
        # both outrank speaking before the picture.
        if enough:                              # 1. funded, after the cut
            take(enough[0])
            continue
        partial = [w for w in cands
                   if w["honest_words"] >= C.MIN_PLACE_WORDS]
        if partial:                             # 2. place-only, after the cut
            take(partial[0], entry_partial=True)
            continue
        late = [w for w in by_start
                if st < w["start"] < sc["end_time"]
                and not w["scene_entry"]
                and not w.get("entry_prelude_for_scene")]
        late_ok = [w for w in late
                   if w["honest_words"] >= C.MIN_GROUND_WORDS
                   and w["start"] - st <= C.LATE_ENTRY_MAX]
        if late_ok:                             # 3. funded, late but useful
            take(late_ok[0])
            continue
        if nec >= C.NECESSITY_HIGH:             # 4. ANTICIPATORY
            # the one deliberate exception to "picture leads voice", and
            # it is LOCATION-ONLY.
            ahead = [w for w in by_start
                     if st - C.ANTICIPATE_MAX <= w["end"] <= st + 0.2
                     and w["honest_words"] >= C.MIN_PLACE_WORDS
                     and not w["scene_entry"]
                     and w.get("anticipatory_for_scene") is None]
            if ahead:
                w = ahead[-1]
                w["anticipatory_for_scene"] = sc["scene_id"]
                w["entry_necessity"] = nec
                w["anticipate_secs"] = round(st - w["end"], 1)
                continue
        if late:                                # 5. anything inside the scene
            take(late[0], entry_underfunded=True)
            continue
        if cands:                               # 6. the stub, honestly flagged
            take(cands[0], entry_underfunded=True)

    # keep: duty-bearing windows at any size, others need MIN_USABLE_STD
    wins = [w for w in wins
            if w["scene_entry"]
            or w.get("entry_prelude_for_scene") is not None
            or w.get("anticipatory_for_scene") is not None
            or w["seconds"] >= C.MIN_USABLE_STD]

    # recent shots (reveals under dialogue)
    for w in wins:
        w["recent_shot_ids"] = [
            s["shot_id"] for s in shots
            if w["start"] - 3.0 <= s["end_time"] <= w["start"] + 0.2
            and s["shot_id"] not in w["shot_ids"]]
    for i, w in enumerate(sorted(wins, key=lambda x: x["start"])):
        w["window_id"] = i
    wins.sort(key=lambda w: w["window_id"])

    mark_reveals(wins, watch_notes or {}, docs)

    # DUTY WINDOWS get the honest budget. Recomputed, not reused:
    # take()/award() move anchors, and the anchor is what the ceiling is
    # measured from.
    starts = sorted(w["start"] for w in wins)
    for w in wins:
        if not (w["scene_entry"]
                or w.get("entry_prelude_for_scene") is not None
                or w.get("anticipatory_for_scene") is not None
                or w.get("reveal") or w.get("reveal_prelude")):
            continue
        honest = _honest_words(w, spans, starts)
        if honest > w["word_budget"]:
            w["word_budget"] = honest

    covered = ({w["entry_for_scene"] for w in wins if w["scene_entry"]}
               | {w.get("anticipatory_for_scene") for w in wins
                  if w.get("anticipatory_for_scene") is not None})
    # a montage member is GROUNDED BY ITS HEAD's cue
    covered |= {o["scene_id"] for o in orient.values()
                if o.get("montage_member_of") in covered}
    summary = {
        "windows": len(wins),
        "scenes_covered": len(covered), "scenes": len(scenes),
        "uncovered_scene_ids": sorted(sc["scene_id"] for sc in scenes
                                      if sc["scene_id"] not in covered),
        "music_dominant": sum(1 for w in wins if w["music_dominant"]),
        "preludes": sum(1 for w in wins
                        if w.get("entry_prelude_for_scene") is not None),
        "anticipatory": sum(1 for w in wins
                            if w.get("anticipatory_for_scene") is not None),
        "underfunded": sum(1 for w in wins if w.get("entry_underfunded")),
        "reveal_windows": sum(1 for w in wins if w.get("reveal")),
        "sung_vocal_policy": sung_vocal_policy,
    }
    return {"windows": wins, "orientation": list(orient.values()),
            "summary": summary}


def duty_flags(w: dict) -> list:
    """The duties a window carries, as prompt text. ONE source of truth:
    the writer and the CRITIC must be told the same thing, or the critic
    edits away a duty it was never shown (measured on Meridian)."""
    flags = []
    if w["scene_entry"]:
        flags.append("SCENE ENTRY" if not w.get("late_entry")
                     else f"LATE ENTRY (scene began {w['late_entry']}s "
                          "earlier)")
        if w.get("entry_underfunded"):
            flags.append("UNDERFUNDED ENTRY — spend EVERY word on the place")
    if w.get("entry_prelude_for_scene") is not None:
        flags.append("ENTRY PRELUDE — name ONLY the new place, nothing "
                     "else; the next window completes the thought")
    if w.get("anticipatory_for_scene") is not None:
        flags.append(f"ANTICIPATORY ENTRY — the scene changes "
                     f"{w.get('anticipate_secs')}s AFTER this window and no "
                     f"silence follows the cut. Name ONLY the new PLACE so "
                     f"the audience is oriented before it happens. Never an "
                     f"event, never a reveal")
    if w.get("entry_montage_scenes"):
        flags.append(f"MONTAGE ENTRY — this cut begins a rapid run of "
                     f"{w['entry_montage_scenes']} scene changes; describe "
                     f"the SEQUENCE as one unit, never cut by cut")
    if (w.get("entry_necessity") or 0) >= C.NECESSITY_HIGH:
        flags.append("HIGH ORIENTATION NECESSITY — nothing in the "
                     "soundtrack tells the audience the scene changed; "
                     "without this cue they are lost")
    if w.get("reveal_prelude"):
        flags.append("REVEAL PRELUDE — the reveal is on screen NOW but "
                     "this gap is too short for it. Emit ONLY "
                     + (f"the apparatus it is seen through: \"In the "
                        f"{w['reveal_mechanism']}.\""
                        if w.get("reveal_mechanism")
                        else "the barest pointer to it")
                     + f" — the window {w.get('reveal_late')}s later "
                       "completes it")
    elif w.get("reveal"):
        flags.append(("REVEAL FOLLOW-THROUGH (the reveal was pointed at "
                      f"{w['reveal_followthrough']}s ago and needs its "
                      "substance now): "
                      if w.get("reveal_followthrough") else "REVEAL WINDOW: ")
                     + w["reveal"][:110]
                     + (f" — the reveal happens THROUGH the "
                        f"{w['reveal_mechanism']}; naming it is MANDATORY"
                        if w.get("reveal_mechanism") else "")
                     + " — ONE subject only")
    if w.get("music_dominant"):
        flags.append("MUSIC/PERFORMANCE")
    allsh = (w["shot_ids"] + w.get("recent_shot_ids", [])
             + w.get("cutaways", []))
    if len(allsh) >= 4:
        flags.append("FAST-CUT MONTAGE")
    return flags


def has_duty(w: dict) -> bool:
    """A window whose cue may NOT be emptied: something in the program is
    unfollowable without it."""
    return bool(w["scene_entry"]
                or w.get("entry_prelude_for_scene") is not None
                or w.get("anticipatory_for_scene") is not None
                or w.get("reveal"))
