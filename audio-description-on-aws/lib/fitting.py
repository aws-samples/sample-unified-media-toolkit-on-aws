"""
TTS text normalization, measured fit (tempo-capped), and the shared
collision pass — ALL PURE ARITHMETIC (settled 2026-08-31): every
placement decision consumes only {at, dur, tempo}, and post-atempo
duration is exactly raw_dur / tempo, so no audio is rendered inside the
workflow. Raw take durations come from tts-on-aws's own per-clip
duration_s; the EQ/atempo chain ships as a RENDER INSTRUCTION on each
placement, applied at delivery time (media-processing owns ffmpeg).
Comments carry the runner's measured lessons verbatim where load-bearing.
"""
from __future__ import annotations

import re

from lib import ad_common as C

# narration EQ + loudness — the runner's chain verbatim, now the render
# INSTRUCTION recorded per placed take (media-processing applies it, plus
# ",atempo={tempo}" when tempo > 1)
AF = ("equalizer=f=420:width_type=o:width=1.2:g=-4,"
      "equalizer=f=3000:width_type=o:width=1.0:g=3,"
      "loudnorm=I=-17:TP=-2:LRA=7")


def next_onset(spans: list, after: float):
    """The true next-word onset after a moment — ONE home for the scan
    that was copy-pasted five times with drifting defaults (review
    2026-08-31). None when no speech follows."""
    return min((s0 for s0, s1 in spans if s0 > after), default=None)


def tts_normalize(text: str) -> str:
    """Years to words, ALL-CAPS to capitalized — the two measured TTS
    stumbles."""
    text = re.sub(r"\b(1[6-9]\d\d|20\d\d)\b",
                  lambda m: _year_words(int(m.group(0))), text)
    text = re.sub(r"\b[A-Z]{2,}\b", lambda m: m.group(0).capitalize(), text)
    return text


def _year_words(y: int) -> str:
    ones = ["", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety"]

    def two(n):
        if n < 20:
            return ones[n]
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")

    hi, lo = divmod(y, 100)
    if lo == 0:
        # even century: 1900 -> "nineteen hundred", but an even
        # MILLENNIUM takes the thousand rule (2000 used to render
        # "twenty hundred" — review 2026-08-31)
        if hi % 10 == 0 and hi >= 20:
            return two(hi // 10) + " thousand"
        return two(hi) + " hundred"
    if hi % 10 == 0 and hi >= 20:                        # 2005
        return two(hi // 10) + " thousand " + (two(lo) if lo else "")
    return two(hi) + " " + (("oh " + ones[lo]) if lo < 10 else two(lo))


def available(at: float, w: dict, spans: list, micro: bool) -> float:
    """Seconds a take starting at `at` may legally occupy: the true
    next-word onset (recovers the PAD the padded window end deducted)
    but NEVER past the window's own end + PAD — the review (2026-08-31)
    found the onset REPLACING the window ceiling, so a take in a window
    ending at a scene cut (or a per-shot window in a long speechless
    gap, where the next onset is 10-30s away) sprawled across the cut
    at tempo 1.0. Mirrors _honest_words' min(); ONE home, used by
    fit_one and the repair chain's cap sizing."""
    window_avail = w["end"] + (0.15 if micro else C.PAD) - at
    onset = next_onset(spans, at)
    if onset is None:
        return window_avail
    return min(window_avail, onset - C.ONSET_PAD - at)


def fit_one(wid: int, raw_dur: float, w: dict, spans: list) -> tuple:
    """Fit one take into its window. Pure: raw_dur is tts's own
    duration_s; the fitted duration is raw_dur / tempo by arithmetic
    (exact to a resample frame — collide's margins are three orders
    larger). Returns (placement, None) or (None, reask_record)."""
    micro = w["seconds"] < 2
    at = max(w["anchor"], w["start"] - (0.15 if micro else 0.0))
    avail = available(at, w, spans, micro)
    tempo = raw_dur / avail if avail > 0 else 99
    if tempo > C.TEMPO_CAP:
        return None, over_budget_record(wid, raw_dur, avail)
    tempo = max(tempo, 1.0)
    return {"window_id": wid, "at": round(at, 2),
            "dur": round(raw_dur / tempo, 2),
            "tempo": round(tempo, 3)}, None


def over_budget_record(wid: int, raw_dur: float, avail: float) -> dict:
    """ONE shape for the over-budget measurement (fit, refit and the
    repair chain's drop records all emit it — it feeds the ledger and
    the reask prompts, so two shapes would split the review surface)."""
    tempo = raw_dur / avail if avail > 0 else 99
    return {"window_id": wid, "avail": round(avail, 2),
            "raw_dur": round(raw_dur, 2),
            "needed_tempo": round(tempo, 2),
            "max_words": max(2, int(avail * C.WPS_REASK))}


def refit(wid: int, raw_dur: float, at: float, spans: list) -> dict | None:
    """One overlap repair, pure: refit to the true next-word onset
    within the tempo cap, else None (the cue drops — never a chipmunk,
    never an overlap). Onset-only by design: the take is already placed
    inside its window; the conflict being repaired is with SPEECH."""
    onset = next_onset(spans, at)
    avail = (onset - C.ONSET_PAD - at) if onset is not None else 1e9
    tempo = raw_dur / avail if avail > 0 else 99
    if tempo > C.TEMPO_CAP:
        return None
    tempo = max(tempo, 1.0)
    return {"dur": round(raw_dur / tempo, 2), "tempo": round(tempo, 3),
            "avail": round(avail, 2)}


def collide(placed: list, spans: list) -> tuple:
    """Shared collision pass: push a cue to prev_end + SPACING, but a
    push that lands the cue inside speech (or past its legal ceiling)
    DROPS it — measured: blind pushes shoved two revived entries fully
    into dialogue. PURE: returns (survivors_sorted, drop_records); the
    caller owns the register. Trial mode = ignore the drop records."""
    ordered = sorted(placed, key=lambda p: p["at"])
    out, drops = [], []
    for p in ordered:
        if out:
            floor = out[-1]["at"] + out[-1]["dur"] + C.SPACING
            if p["at"] < floor:
                p = {**p, "at": round(floor, 2)}
        onset = next_onset(spans, p["at"] - 0.01)
        onset = 1e9 if onset is None else onset
        inside = any(s0 - 0.02 <= p["at"] < s1 for s0, s1 in spans)
        if inside or p["at"] + p["dur"] > onset - 0.05:
            # the MEASUREMENT is the point: "landed in speech" is a
            # verdict, "runs 240 ms into the next line" is a trade a
            # human can price
            over = round(max(0.0, p["at"] + p["dur"] - (onset - 0.05)), 2)
            drops.append({"window_id": p["window_id"],
                          "kind": "speech_overlap",
                          "at": p["at"], "dur": p["dur"],
                          "overlap_sec": (over if not inside
                                          else round(p["dur"], 2)),
                          "starts_inside_speech": inside})
            continue
        out.append(p)
    return out, drops


def collateral(k: int, fit: dict, placed: list, spans: list) -> str | None:
    """Who else suffers if window k's cue grows to `fit`? Run the real
    collision pass on a throwaway copy and diff survivors' placements.
    The no_longer rule enforced as "any growth" destroyed two GOOD
    repairs over 0.08s and 0.12s that harmed nobody — the precise test
    was always "does any OTHER cue get dropped or pushed into speech"."""
    trial = [dict(x, **fit) if x["window_id"] == k else dict(x)
             for x in placed]
    before = {x["window_id"]: x["at"] for x in placed
              if x["window_id"] != k}
    survivors, _ = collide(trial, spans)
    after = {x["window_id"]: x["at"] for x in survivors
             if x["window_id"] != k}
    if not any(x["window_id"] == k for x in survivors):
        return "would itself land in speech"
    lost = sorted(set(before) - set(after))
    moved = [w for w, at in after.items() if abs(at - before[w]) > 0.05]
    if lost:
        return f"would drop w{','.join(map(str, lost))}"
    if moved:
        return f"would push w{','.join(map(str, sorted(moved)))}"
    return None
