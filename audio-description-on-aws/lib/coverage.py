"""
The coverage oracles' predicates, PURE and in memory — the neutrality
guard every deleting stage must pass (memory: refactor-aggressively —
prove neutrality before the change it enables). Any stage that deletes a
cue owes the audience a proof that nothing was lost, and the only proof
worth having is the oracles' own test, not a second weaker copy of it.
"""
from __future__ import annotations

from lib import ad_common as C
from lib.oracles import cue_toks, quoted_toks, TEXT_CLAIM


def _norm(cues: dict) -> dict:
    """Str-key the cue map at every entry point. cues.json round-trips
    as str keys but in-memory maps are often int-keyed — the mismatch
    silently DEAD-ARMED the whole neutrality guard once (review
    2026-08-31: every lookup missed, so every deletion 'proved'
    neutral). Normalizing here kills the bug class for all callers."""
    return {str(k): v for k, v in cues.items()}


def beat_near(sec: float, placed: list, cues: dict, what: str = None) -> list:
    """Cues that convey a beat. Proximity is the general test; one class
    is decidable and is exactly the class that failed: a beat that QUOTES
    on-screen text is conveyed only by a cue that reads that text
    (measured: Meridian's own title was never announced, and the beat
    passed on its nearness to "The squad room, next.")."""
    cues = _norm(cues)
    near = [p for p in placed
            if sec + C.BEAT_WINDOW[0] <= p["at"] <= sec + C.BEAT_WINDOW[1]
            and cues.get(str(p["window_id"]), "").strip()]
    q = quoted_toks(what) if what and TEXT_CLAIM.search(what) else set()
    if q:
        return [p for p in near
                if cue_toks(cues.get(str(p["window_id"]), "")) & q]
    return near


def orient_near(o: dict, placed: list, cues: dict) -> list:
    """Cues that could orient this break. EARLY edge measured at the
    cue's END, LATE edge at its start — the same quantities the duty
    award used (measured: a start-only check called a legal anticipatory
    cue unserved; an oracle that contradicts a rule the pipeline
    deliberately applies teaches the reviewer to distrust the oracle)."""
    cues = _norm(cues)
    at = o["at"]
    return [p for p in placed
            if at - C.ANTICIPATE_MAX <= p["at"] + p["dur"]
            and p["at"] <= at + C.ORIENT_WINDOW
            and cues.get(str(p["window_id"]), "").strip()]


def coverage(placed: list, cues: dict, watch_notes: dict,
             orientation: list) -> tuple:
    """(beats covered, high-necessity breaks served). Reveals count
    alongside beats: a reveal is the one thing whose TIMING is
    load-bearing, so a deletion near one must be proved too."""
    return (frozenset(float(b["sec"]) for b in
                      (watch_notes.get("beats", [])
                       + watch_notes.get("reveals", []))
                      if beat_near(float(b["sec"]), placed, cues,
                                   b.get("what"))),
            frozenset(o["at"] for o in orientation
                      if o["necessity"] >= C.NECESSITY_HIGH
                      and o.get("montage_member_of") is None
                      and orient_near(o, placed, cues)))


def safe_deletions(placed: list, cues: dict, candidates: list,
                   watch_notes: dict, orientation: list) -> tuple:
    """Filter proposed deletions down to the ones that cost the audience
    nothing. Tested CUMULATIVELY and in order, because deletions
    interact: two cues may each be redundant against a third and still
    not both be spare. This is why redundancy repair is a deletion rather
    than a rewrite — a deletion can be proved neutral before it ships.
    Returns (ok_window_ids, [(window_id, why_refused)])."""
    keep = list(placed)
    base = coverage(keep, cues, watch_notes, orientation)
    ok, refused = [], []
    for wid in candidates:
        trial = [p for p in keep if p["window_id"] != wid]
        if len(trial) == len(keep):
            continue
        cov = coverage(trial, cues, watch_notes, orientation)
        lost = (base[0] - cov[0], base[1] - cov[1])
        if lost[0] or lost[1]:
            refused.append(
                (wid, "; ".join(x for x in (
                    f"would lose beat(s) at {sorted(lost[0])}"
                    if lost[0] else "",
                    f"would leave orientation break(s) at "
                    f"{sorted(lost[1])} unserved" if lost[1] else "")
                    if x)))
            continue
        keep, base = trial, cov
        ok.append(wid)
    return ok, refused
