"""
Mechanical repair TRIGGERS — pure detectors, every one born from a dated
incident (the runner's measured cases; comments verbatim where the
lesson is load-bearing). Each returns evidence for the repair chain; the
model half of a repair lives in states/repair_chain.py.
"""
from __future__ import annotations

import re

from lib import ad_common as C

# ── shared tokenizers ────────────────────────────────────────────────────────

_CUE_STOP = set("""a an the of in on at to and or is are was were with his her
its their this that as by for from into over under then next up down out he
she they it while there here now still just very onto off toward towards
through above below behind beside between across against about after before
during""".split())

_DESC_STOP = {"the", "a", "an", "and", "with", "in", "on", "of", "his", "her",
              "their", "wearing", "dressed", "who", "is", "has", "appears",
              "appearing", "after", "sec", "seconds", "screen", "time"}

# A bare person noun IDENTIFIES NOBODY (measured on drama_b: "A man, in a
# white shirt, raises his arms" became "Billy, in a white shirt..." on the
# single token 'man' — while the man surrendering is plainly not the
# detective in tactical gear).
_GENERIC_PERSON = {"man", "woman", "person", "figure", "guy", "boy", "girl",
                   "male", "female", "someone", "people", "one", "adult"}

_TITLES = {"detective", "captain", "officer", "doctor", "sergeant",
           "lieutenant", "inspector", "agent", "mrs", "miss", "mister",
           "father", "judge", "professor", "senator", "mayor", "chief",
           "sheriff"}

TEXT_CLAIM = re.compile(
    r"\b(?:a\s+|the\s+)?(?:title|card|sign|text|letters|words|caption|banner|"
    r"headline|marquee|screen|credits?)\b[^.]{0,40}?"
    r"\b(?:reads?|says?|shows?|announces?|appears?|lists?|roll[s]?)\b"
    r"|\bon-?screen text\b|\btitle card\b|\bcredits?\s+roll\b", re.I)

NONDIEGETIC = {"title_card", "credit_roll", "logo_bumper"}


def _sing(tok):
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us",
                                                                "is")):
        return tok[:-1]
    return tok


def cue_toks(txt: str) -> set:
    """Content words of a cue. Deliberately NOT the descriptor tokenizer:
    this one measures how much of one cue's CONTENT another already
    carried."""
    return {w for w in re.findall(r"[a-z']+", (txt or "").lower())
            if len(w) > 2 and w not in _CUE_STOP}


def _desc_toks(txt):
    return {_sing(x) for x in re.findall(r"[a-z'-]+", (txt or "").lower())
            if len(x) >= 3 and x not in _DESC_STOP}


def quoted_toks(s: str) -> set:
    """The checkable part of a claim about on-screen text is what it
    QUOTES (or puts after a colon). The lookbehind guards the APOSTROPHE
    only — guarding the colon too blocked "A title reads: Meridian", the
    commonest form the claim takes."""
    out = set()
    for m in re.finditer(r"""(?:(?<![A-Za-z])['"“]|:)\s*([^'"”.]{2,70})""",
                         s or ""):
        out |= cue_toks(m.group(1))
    return out


def shot_texts(d: dict) -> list:
    """Every string of readable text a shot carries: texts_on_screen
    (post-production text) UNION environment_text entities (diegetic
    signage — measured: drama_b w33's true "HARBOR LIGHT" neon-sign claim was
    destroyed under the post-production-only evidence base) UNION
    overlay_text entities (since SME 2026-09-09, texts_on_screen is the
    UNCLAIMED remainder — a line fully accounted for by an extracted
    identity lives only in the entity list, so without this arm a
    title-card name would vanish from the evidence base). ONE reader
    for all text oracles, so they cannot diverge on what counts."""
    out = list(d.get("texts_on_screen") or [])
    out += [e["name"] for e in (d.get("named_entities") or [])
            if isinstance(e, dict) and e.get("name")
            and e.get("evidence") in ("environment_text", "overlay_text")]
    # ordered dedup: PRE-2026-09-09 docs carry a line AND its extracted
    # entity (texts_on_screen was complete, not the remainder), so the
    # union double-lists there — harmless to the oracles, noise in the
    # prompt evidence built from this reader
    return list(dict.fromkeys(out))


def onscreen_evidence(wid: int, p: dict, ws: dict, docs: dict,
                      tm: dict) -> dict:
    """Every string of on-screen text the audience could be looking at
    while this cue plays: the window's own shots UNION the shots the cue
    actually spans (both are needed — measured on drama_b w201 and on
    Meridian's card-shorter-than-window case)."""
    sids = set(ws[wid]["shot_ids"])
    a, b = p["at"], p["at"] + p["dur"]
    sids |= {s for s, (s0, s1) in tm.items() if s0 <= b and a <= s1}
    return {s: shot_texts(docs.get(s, {})) for s in sorted(sids)}


# ── snap: on-screen-text beats re-timed to the shot carrying the text ────────

def snap_text_beats(watch_notes: dict, docs: dict, tm: dict) -> list:
    """Correct the TIMESTAMP of any watch-note beat asserting on-screen
    text to the shot that actually carries it. Root cause of the Meridian
    title-card failure, three stages upstream of where it showed.
    MUTATES watch_notes; returns [(beat, was_sec, now_sec, shot_id)]."""
    fixed = []
    for b in (watch_notes.get("beats", [])
              + watch_notes.get("reveals", [])):
        what = str(b.get("what") or "")
        if not TEXT_CLAIM.search(what):
            continue
        quoted = quoted_toks(what)
        if not quoted:
            continue
        sec = float(b["sec"])
        hits = [(abs(tm[s][0] - sec), tm[s][0], s)
                for s, d in docs.items()
                if s in tm and cue_toks(" ".join(shot_texts(d))) & quoted]
        if not hits:
            continue
        if any(tm[s][0] - 1.0 <= sec <= tm[s][1] for _, _, s in hits):
            continue                # already inside the right shot
        _, at, sid = min(hits)
        fixed.append((dict(b), sec, round(at, 2), sid))
        b["sec"] = round(at, 2)
    return fixed


# ── detector: false on-screen-text claims ────────────────────────────────────

def false_text_claims(ws: dict, cues: dict, placed: list, docs: dict,
                      tm: dict) -> dict:
    """A cue claiming on-screen text FAILS when it quotes words that
    appear in no reachable shot's text, or claims text where no reachable
    shot has any. A false claim is the worst error available: the
    audience has no way to check it."""
    bad = {}
    for p in placed:
        wid = p["window_id"]
        text = cues.get(str(wid), "").strip()
        if not text or not TEXT_CLAIM.search(text):
            continue
        ev = onscreen_evidence(wid, p, ws, docs, tm)
        have = cue_toks(" ".join(t for v in ev.values() for t in v))
        quoted = quoted_toks(text)
        if quoted:
            if quoted & have:
                continue
            miss = sorted(quoted)
        elif have:
            continue
        else:
            miss = []
        # where the claimed text DOES live decides timing-error vs
        # invention. `s in tm` guard: docs and tm come from two sibling
        # artifacts that DO diverge — snap_text_beats carries the same
        # guard (review 2026-08-31: this one lacked it and a divergent
        # shot crashed the whole oracle over one claim)
        elsewhere = sorted(
            (tm[s][0], " / ".join(t))
            for s, d in docs.items()
            if s in tm
            for t in [shot_texts(d)]
            if t and (not miss or cue_toks(" ".join(t)) & set(miss)))
        bad[wid] = {"window_id": wid, "kind": "false_onscreen_text",
                    "at": round(p["at"], 2), "missing": miss,
                    "evidence": {s: v for s, v in ev.items() if v},
                    "text_lives_at": elsewhere[:3],
                    "note": ("claims on-screen text "
                             + (f"{miss!r} that no shot under this cue "
                                "carries" if miss
                                else "where no shot under this cue has any"))}
    return bad


# ── detector: actor-name leaks ───────────────────────────────────────────────

def name_leaks(cmap: list, cues: dict, placed: list, ws: dict,
               docs: dict) -> dict:
    """No token of any face-key (actor name) may appear in a cue unless
    it equals the mapped character name. Verbatim on-screen text may
    contain actor names legally — reading titles aloud is the AD rule.
    Added after a reroll leaked the face-key's given name into cue text
    ('<given-name> drives', from face-key actor_M2):
    dimensions without oracles regress on every regeneration."""
    banned = {}
    for c in cmap:
        ok = set((c.get("character_name") or "").split())
        for tok in c["face_key"].split():
            if (len(tok) >= 3 and tok not in ok
                    and not c["face_key"].startswith("Person_")):
                banned[tok.lower()] = (c["face_key"],
                                       c.get("character_name")
                                       or c.get("descriptor"))
    leaks = {}
    for p in placed:
        wid = p["window_id"]
        text = cues.get(str(wid), "")
        onscreen = " ".join(t for s in ws[wid]["shot_ids"]
                            for t in shot_texts(docs.get(s, {}))).lower()
        for w_ in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
            if w_.lower() in banned and w_.lower() not in onscreen:
                leaks[wid] = (w_, *banned[w_.lower()])
    return leaks


# ── detector: role descriptor for an already-named character ─────────────────

def descriptor_hits(cmap: list, ws: dict, cues: dict, placed: list,
                    docs: dict) -> dict:
    """'The younger detective' where Sullivan was named two cues earlier
    (user, 2026-08-28) — a join failure, checked mechanically. Gates, all
    mechanical: on screen in that window, name already usable, and the
    descriptor fits the named character STRICTLY BETTER than every OTHER
    character on screen, including the UNNAMED ones (who are the reason
    descriptors exist)."""
    def _name_toks(nm):
        return {t for t in re.findall(r"[a-z']+", nm.lower())
                if len(t) >= 3 and t not in _TITLES}

    named, others = [], []
    for c in cmap:
        nm = (c.get("character_name") or "").strip()
        nouns = _desc_toks(c.get("descriptor"))
        if not nouns:
            continue
        if nm and not c.get("concealed"):
            named.append((c["face_key"], nm, nouns,
                          float(c.get("speakable_from_sec",
                                      c.get("first_established_sec") or 0))))
        else:
            others.append((c["face_key"], nouns))
    if not named:
        return {}
    allnouns = set().union(*(nouns for _, _, nouns, _ in named))
    hits = {}
    for p in placed:
        wid = p["window_id"]
        text = cues.get(str(wid), "")
        if not text.strip():
            continue
        # Face clusters MISS people the shot summaries name outright —
        # the stricter reading of presence is the safe one (presence only
        # ever ADDS rivals).
        ev = " ".join(str(docs.get(sid, {}).get("description", ""))
                      for sid in ws[wid]["shot_ids"]).lower()
        onscreen = (set(ws[wid].get("people_on_screen") or [])
                    | {c["face_key"] for c in cmap
                       if str(c.get("face_key", "")).lower() in ev})
        # re.I is load-bearing: a descriptor usually LEADS the sentence.
        for m in re.finditer(
                r"\b(?:the|a|an)\s+((?:[A-Za-z'-]+\s+){0,2}[A-Za-z'-]+)\b",
                text, re.I):
            # tried at every LENGTH, longest first: a greedy 3-word
            # capture swallows the verb ("steps" -> head='step').
            words = m.group(1).split()
            art = m.group(0)[:m.group(0).lower().index(words[0].lower())]
            for k in range(len(words), 0, -1):
                head = _sing(words[k - 1].lower())
                if head in allnouns:
                    break
            else:
                continue
            phrase = art + " ".join(words[:k])
            pt = _desc_toks(" ".join(words[:k]))
            # ANY token of the name already in the text means the phrase
            # is somebody ELSE (measured: "Billy meeting Billy").
            cands = [(len(pt & nouns), fk, nm)
                     for fk, nm, nouns, est in named
                     if head in nouns and fk in onscreen and est <= p["at"]
                     and (pt & nouns) - _GENERIC_PERSON
                     and not (_name_toks(nm) & _desc_toks(text))]
            if not cands:
                continue
            best = max(c[0] for c in cands)
            top = [c for c in cands if c[0] == best]
            rival = max((len(pt & nouns) for fk, nouns in others
                         if fk in onscreen), default=0)
            if len(top) == 1 and best > rival:
                hits[wid] = (phrase, top[0][1], top[0][2])
                break
    return hits


# ── detector: redundancy between CONSECUTIVE cues ────────────────────────────

def redundant_pairs(cues: dict, placed: list, ws: dict, cmap: list) -> list:
    """Pure detector over consecutive PLACED cues. Character names are
    stripped before measuring: a recurring subject is what continuity
    looks like, not repetition. Measured across the corpus: endemic
    (4/43 meridian, 9/69 drama_c, 9/128 drama_b pairs, n=3 assets); the
    dominant shape is a bare place label followed seconds later by the
    full cue that repeats it — the ECHO case, resolved by deletion."""
    names = set()
    for c in cmap:
        for f in ("character_name", "face_key"):
            names |= cue_toks(str(c.get(f) or ""))
    ordered = sorted(placed, key=lambda p: p["at"])
    out = []
    for a, b in zip(ordered, ordered[1:]):
        ta = cue_toks(cues.get(str(a["window_id"]), "")) - names
        tb = cue_toks(cues.get(str(b["window_id"]), "")) - names
        if not ta or not tb:
            continue
        gap = b["at"] - (a["at"] + a["dur"])
        if gap > C.REDUNDANT_GAP_MAX:
            continue
        shared = ta & tb
        cont = len(shared) / min(len(ta), len(tb))
        if cont < C.REDUNDANT_CONTAIN:
            continue
        wa = len(cues.get(str(a["window_id"]), "").split())
        wb = len(cues.get(str(b["window_id"]), "").split())
        # the echo is the SHORTER cue, and only when the longer one
        # already carries essentially all of it
        echo = None
        if (len(ta) <= len(tb) and wa <= C.REDUNDANT_ECHO_WORDS
                and len(shared) / len(ta) >= 0.9):
            echo = a["window_id"]
        elif (len(tb) < len(ta) and wb <= C.REDUNDANT_ECHO_WORDS
                and len(shared) / len(tb) >= 0.9):
            echo = b["window_id"]
        out.append({"first": a["window_id"], "second": b["window_id"],
                    "at": round(a["at"], 2), "gap_sec": round(gap, 2),
                    "containment": round(cont, 2),
                    "shared": sorted(shared), "echo": echo,
                    "same_scene": ws[a["window_id"]].get("scene_id")
                    == ws[b["window_id"]].get("scene_id")})
    return out


# ── detector: the programme TAIL is one announcement ─────────────────────────

def nondiegetic_blocks(docs: dict, tm: dict) -> list:
    """Contiguous non-diegetic runs, bridged over short black. NOTE:
    program-structure-analysis' lib/markers.py is the ONE home for this
    predicate going forward; this copy serves the tail oracle until the
    consumer migrates (program-structure-analysis-on-aws/README.md
    Design record item 8)."""
    blocks, cur = [], []
    for sid in sorted(tm):
        kind = docs.get(sid, {}).get("shot_kind")
        span = tm[sid][1] - tm[sid][0]
        if kind in NONDIEGETIC:
            cur.append(sid)
        elif kind == "black" and cur and span <= C.TAIL_BRIDGE_MAX:
            cur.append(sid)
        else:
            if any(docs.get(s, {}).get("shot_kind") in NONDIEGETIC
                   for s in cur):
                blocks.append(cur)
            cur = []
    if any(docs.get(s, {}).get("shot_kind") in NONDIEGETIC for s in cur):
        blocks.append(cur)
    return [{"shot_ids": b, "start": tm[b[0]][0], "end": tm[b[-1]][1],
             "kinds": sorted({docs[s]["shot_kind"] for s in b
                              if docs.get(s, {}).get("shot_kind")
                              in NONDIEGETIC})}
            for b in blocks]


def tail_overdescribed(docs: dict, tm: dict, duration: float, cues: dict,
                       placed: list) -> dict | None:
    """The LAST non-diegetic block, if it reaches the end of the
    programme, holding more than one cue. Detection only — WHICH cards
    deserve words is a judgment made by the model (measured: drama_b's
    'In Loving Memory' dedication sits between a credit roll and five
    vendor logos; no word list fitted to three assets could keep it)."""
    blocks = nondiegetic_blocks(docs, tm)
    if not blocks:
        return None
    ordered = sorted(placed, key=lambda p: p["at"])
    for b in reversed(blocks):
        if b["end"] < duration - 2.0:
            break
        spoken = [p for p in ordered
                  if b["start"] - 1.0 <= p["at"] <= b["end"]
                  and cues.get(str(p["window_id"]), "").strip()]
        if len(spoken) > 1:
            return {**b, "cues": [p["window_id"] for p in spoken]}
    return None
