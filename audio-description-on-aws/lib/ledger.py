"""
The cue ledger: two INDEPENDENT axes (importance, compliance) plus a
violation list; disposition DERIVED, never labelled — the ship/review
line is customer POLICY (a Netflix deliverable, a WCAG 1.2.7 extended-AD
deliverable and a sparse broadcast mix each draw it differently), so the
ledger stores measurements and a different policy is a re-derive, not a
re-run. Evidence vs verdict, as everywhere in this family.

Rows include obligations that produced NO cue — a queue holding only what
we managed to write is blind in exactly the way the orientation oracle
exists to end. PURE: parsed artifacts in, rows out.
"""
from __future__ import annotations

from lib import ad_common as C
from lib.windows import has_duty

_WORST = {"minor": 1, "moderate": 2, "major": 3}


def severity(kind: str, m: dict) -> str:
    """COMPUTED from the measurement, never labelled. 'Overlaps dialogue'
    is not inherently minor — it depends on what got overlapped."""
    if kind == "over_budget":
        t = m.get("needed_tempo", 9)
        return ("minor" if t < C.TEMPO_MINOR
                else "major" if t >= C.TEMPO_MAJOR else "moderate")
    if kind == "speech_overlap":
        if m.get("starts_inside_speech"):
            return "major"
        o = m.get("overlap_sec", 0)
        return "minor" if o <= 0.25 else "moderate" if o <= 0.6 else "major"
    if kind == "late_orientation":
        s = m.get("late_sec", 0)
        return "minor" if s <= 20 else "moderate" if s <= 35 else "major"
    # 'redundant' and 'nondiegetic_tail' absences are INTENDED (the words
    # survive in a keeper cue) — still a violation, because the
    # not-in-mix invariant admits no exceptions, but scored minor.
    return {"no_window": "major", "duty_empty": "major",
            "unplaced": "major", "false_onscreen_text": "major",
            "underfunded": "moderate", "anticipatory": "minor",
            "redundant": "minor", "nondiegetic_tail": "minor"
            }.get(kind, "moderate")


def build_ledger(ws: dict, cues: dict, placed_list: list,
                 orientation: list, watch_notes: dict, gaps: list,
                 drops: list, takes: dict | None = None) -> list:
    takes = takes or {}
    placed = {p["window_id"]: p for p in placed_list}
    orient = {o["scene_id"]: o for o in orientation}
    drops_by_id = {d["window_id"]: d for d in drops}
    beats = list(watch_notes.get("beats") or [])

    rows = []
    for wid, w in sorted(ws.items()):
        text = cues.get(str(wid), "").strip()
        p = placed.get(wid)
        if not text and not has_duty(w):
            continue                 # a silent non-duty window is not news
        at = p["at"] if p else w["anchor"]

        # ── AXIS 1: importance — WHY this cue has to exist ──
        why, nec = [], float(w.get("entry_necessity") or 0)
        if w["scene_entry"]:
            why.append(f"ENTRY for scene {w.get('entry_for_scene')}"
                       + (f", {w['late_entry']}s late"
                          if w.get("late_entry") else ""))
        if w.get("entry_prelude_for_scene") is not None:
            why.append(f"ENTRY PRELUDE for scene "
                       f"{w['entry_prelude_for_scene']}")
        if w.get("anticipatory_for_scene") is not None:
            why.append(f"ANTICIPATORY entry for scene "
                       f"{w['anticipatory_for_scene']}")
        if w.get("entry_montage_scenes"):
            why.append(f"MONTAGE of {w['entry_montage_scenes']} cuts")
        if w.get("reveal"):
            why.append(("REVEAL PRELUDE: " if w.get("reveal_prelude") else
                        "REVEAL FOLLOW-THROUGH: "
                        if w.get("reveal_followthrough") else "REVEAL: ")
                       + w["reveal"][:90])
            nec = max(nec, 0.9)      # a reveal is the film's turn
        for b in beats:
            if w["start"] - 3.0 <= float(b["sec"]) <= w["end"] + 3.0:
                why.append(f"BEAT @{float(b['sec']):.0f}s: "
                           f"{str(b.get('what') or '')[:80]}"
                           + (" [inferred]" if b.get("inferred") else ""))
                nec = max(nec, 0.8)

        # ── AXIS 2: compliance. INVARIANT: a row not in the mix ALWAYS
        # carries a violation (measured: w7 scored 'ship' on a cue nobody
        # will ever hear).
        vio, oracle = [], []
        d = drops_by_id.get(wid)
        if d and not p:
            vio.append({**{k: v for k, v in d.items()
                           if k != "window_id"},
                        "severity": severity(d["kind"], d)})
        elif d and p:
            oracle.append(f"an earlier take was dropped ({d['kind']}) "
                          "and recovered")
        elif not text:
            vio.append({"kind": "duty_empty", "severity": "major",
                        "note": "a duty window shipped no words"})
        elif not p:
            vio.append({"kind": "unplaced", "severity": "major",
                        "note": "cue written but no take reached the mix "
                                "— cause is not in the drop register"})
        if p and w["scene_entry"] and w.get("entry_for_scene") is not None:
            o = orient.get(w["entry_for_scene"])
            late = at - float(o["at"]) if o else 0
            if o and late > C.ORIENT_WINDOW:
                vio.append({"kind": "late_orientation",
                            "late_sec": round(late, 1),
                            "severity": severity("late_orientation",
                                                 {"late_sec": late}),
                            "note": f"lands {late:.0f}s after the cut it "
                                    "is meant to orient"})
        if w.get("entry_underfunded"):
            vio.append({"kind": "underfunded", "severity": "moderate",
                        "note": f"only {w['word_budget']} words of "
                                "silence available"})
        if w.get("anticipatory_for_scene") is not None:
            vio.append({"kind": "anticipatory", "severity": "minor",
                        "note": f"voiced {w.get('anticipate_secs')}s "
                                "before the picture (accepted practice "
                                "for LOCATION only)"})

        worst = max((_WORST[v["severity"]] for v in vio), default=0)
        important = nec >= C.NECESSITY_HIGH or bool(w.get("reveal"))
        if not vio:
            disp = "ship" if important else "ship_minor"
        elif worst >= 3:
            disp = "fix_required"
        else:
            disp = "review_tradeoff"
        rows.append({"id": f"w{wid}", "window_id": wid, "kind": "cue",
                     "at": round(at, 2),
                     "dur": round(p["dur"], 2) if p else 0.0,
                     "text": text,
                     "state": ("placed" if p
                               else ("empty" if not text else "dropped")),
                     # the RAW take's uri (in the tts run's prefix) +
                     # tempo is the render instruction; no fitted wav
                     # exists inside this workflow
                     "audio": ((takes.get(str(wid)) or {}).get("wav")
                               if text else None),
                     "tempo": p.get("tempo") if p else None,
                     "budget": w["word_budget"],
                     "necessity": round(nec, 2), "important": important,
                     "why_needed": why, "violations": vio,
                     "oracle": oracle, "disposition": disp})

    # Obligations from the oracles. An obligation whose window IS already
    # a row MERGES into it: 'scene 15 never grounded' and 'w102 shipped
    # no words' are one failure seen from two ends.
    for g in gaps:
        sec = float(g.get("sec", 0))
        what = (g.get("beat") or g.get("reason") or "")[:130]
        near = sorted((r for r in rows if abs(r["at"] - sec) < 2.5),
                      key=lambda r: abs(r["at"] - sec))
        if near:
            r = near[0]
            r["oracle"].append(f"{g.get('kind')}: {g.get('reason', '')}")
            if not r["text"] and what not in " ".join(r["why_needed"]):
                r["why_needed"].append(f"{g.get('kind')}: {what}")
            continue
        rows.append({"id": f"gap@{sec:.1f}", "window_id": None,
                     "kind": "obligation", "at": round(sec, 2),
                     "dur": 0.0, "text": "", "state": "unwritten",
                     "audio": None, "budget": 0,
                     "necessity": float(g.get("necessity") or 0.8),
                     "important": True,
                     "why_needed": [f"{g.get('kind')}: {what}"],
                     "violations": [{"kind": "no_window",
                                     "severity": "major",
                                     "note": g.get("reason", "")}],
                     "oracle": [], "disposition": "no_window"})
    rows.sort(key=lambda r: r["at"])
    return rows


def merge_gaps(gaps: list, kind: str, new: list) -> list:
    """ad_gaps is written by several oracles; each replaces only its own
    kind (check_beats used to clobber succession's entries wholesale)."""
    gaps = [g for g in gaps if g.get("kind") != kind]
    gaps += [{**g, "kind": kind} for g in new]
    gaps.sort(key=lambda g: g.get("sec", 0))
    return gaps
