"""
The gap map + orientation necessity — the content-agnostic half of the AD
runner's window machinery, lifted to its designed home (README.md §8's
migration note; validated by zero-hand-edit AD runs before landing
here). What stays behind in audio-description: DUTY assignment
(scene entries, reveals, anticipation) — those consult watch notes and are
editorial, not structural.

NEW here, not in the runner: SOFT windows — the graded-speech-clearance
predicate: speech-stem energy carrying NO transcript words is untranscribed
walla/background chatter, which professional describers narrate over
(measured: 12-15% of the reference corpus's narration words). A soft window is airtime a
consumer MAY use at its own policy's discretion; hard windows remain the
default airtime. The trigger is mechanical; no judgment call anywhere.

Constants are the runner's exact values — parity with its windows.json is
the port's acceptance test, so they move together or not at all.
"""
from __future__ import annotations

WORD_MERGE = 0.35
PAD = 0.30
MIN_USABLE = 0.7
LONG_GAP = 20.0
MIN_SHOT_WIN = 2.5
WPS = 2.3
MUSIC_DOMINANT_FRAC = 0.85
MONTAGE_SCENE_MAX = 6.0
SOFT_MIN_SEC = 1.5           # a soft window shorter than a short cue is noise
SOFT_WORD_TOLERANCE = 0     # ANY transcript word inside disqualifies: the
                            # predicate is "speech the transcript cannot
                            # hear", never "speech we decided is unimportant"


def word_spans(transcript: dict) -> list:
    spans = []
    for w in transcript["words"]:
        a, b = float(w["start"]), float(w["end"])
        if spans and a - spans[-1][1] <= WORD_MERGE:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return spans


def gap_map(transcript: dict, scenes: list, shots: list,
            stem_activity: dict | None) -> dict:
    """Hard windows (dialogue-free, scene-cut split, per-shot on long
    stretches, budgeted) + soft windows (untranscribed speech)."""
    act = (stem_activity or {}).get("activity", {})
    music = act.get("music", [])
    duration = float(transcript["audio_duration_sec"])
    spans = word_spans(transcript)

    gaps = []
    edges = [[0.0, 0.0]] + spans + [[duration, duration]]
    for (_, b1), (a2, _) in zip(edges, edges[1:]):
        lo, hi = b1 + PAD, a2 - PAD
        if hi - lo >= MIN_USABLE:
            gaps.append([round(lo, 2), round(hi, 2)])

    def music_frac(a, b):
        cov = sum(max(0.0, min(b, m1 / 1000) - max(a, m0 / 1000))
                  for m0, m1 in music)
        return cov / max(b - a, 0.01)

    windows = []

    def add(a, b, shot_ids, kind="hard", evidence=None):
        sc = next((x for x in scenes
                   if x["start_time"] <= a < x["end_time"]), None)
        mus = music_frac(a, b)
        windows.append({
            "window_id": len(windows), "kind": kind,
            "start": round(a, 2), "end": round(b, 2),
            "seconds": round(b - a, 2),
            "word_budget": max(2, min(26, int((b - a) * WPS))),
            "scene_id": sc["scene_id"] if sc else None,
            "shot_ids": list(shot_ids),
            "music_dominant": mus > MUSIC_DOMINANT_FRAC,
            **({"evidence": evidence} if evidence else {}),
        })

    ordered_shots = sorted(shots, key=lambda s: s["start_time"])
    for g0, g1 in gaps:
        cuts = [sc["start_time"] for sc in scenes
                if g0 + MIN_USABLE < sc["start_time"] < g1 - MIN_USABLE]
        segments, prev = [], g0
        for c in sorted(cuts) + [g1]:
            segments.append([prev, c])
            prev = c
        for a, b in segments:
            if b - a < MIN_USABLE:
                continue
            if b - a <= LONG_GAP:
                add(a, b, [s["shot_id"] for s in ordered_shots
                           if s["start_time"] < b and s["end_time"] > a])
                continue
            for s in ordered_shots:                # long stretch: per shot
                sa = max(s["start_time"] + 0.05, a)
                sb = min(s["end_time"] - 0.2, b)
                if sb - sa >= MIN_SHOT_WIN:
                    add(sa, sb, [s["shot_id"]])

    for a, b in _soft_spans(act.get("speech", []), spans):
        add(a, b, [s["shot_id"] for s in ordered_shots
                   if s["start_time"] < b and s["end_time"] > a],
            kind="soft", evidence="untranscribed_speech")

    return {"windows": windows,
            "hard": sum(1 for w in windows if w["kind"] == "hard"),
            "soft": sum(1 for w in windows if w["kind"] == "soft")}


def _soft_spans(speech_activity: list, word_spans_: list) -> list:
    """Speech-stem spans containing NO transcript words: walla, crowd,
    background chatter the ASR cannot hear as language."""
    out = []
    for m0, m1 in speech_activity:
        a, b = m0 / 1000.0, m1 / 1000.0
        if b - a < SOFT_MIN_SEC:
            continue
        if not any(w0 < b and w1 > a for w0, w1 in word_spans_):
            out.append((round(a + 0.1, 2), round(b - 0.1, 2)))
    return out


def orientation(scenes: list, transcript: dict,
                stem_activity: dict | None) -> list:
    """Per scene boundary: how disorienting is this cut to a listener, and
    how much does the audio already convey — necessity = break * (1 -
    sufficiency). The runner's validated model, verbatim, minus nothing."""
    import re
    act = (stem_activity or {}).get("activity", {})
    music, sfx = act.get("music", []), act.get("sfx", [])
    sents = transcript["sentences"]
    words = transcript["words"]
    spans = word_spans(transcript)
    out = {}
    for i, sc in enumerate(scenes):
        st = float(sc["start_time"])
        prev = scenes[i - 1] if i else None
        f, br = {}, 0.0
        if prev is None:
            br += 0.35
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
            f["silence_at_cut"] = True
        if bridged:
            suff = 0.0
            br = min(1.0, br + 0.20)
            f["dialogue_bridged"] = True
        suff = min(1.0, suff)
        out[sc["scene_id"]] = {
            "scene_id": sc["scene_id"], "at": round(st, 2),
            "break": round(br, 3), "audio_sufficiency": round(suff, 3),
            "necessity": round(br * (1 - suff), 3),
            "dialogue_bridged": bridged, "factors": f}

    head = None                                    # montage clustering
    for sc in scenes:
        o = out[sc["scene_id"]]
        short = (float(sc["end_time"]) - float(sc["start_time"])
                 < MONTAGE_SCENE_MAX)
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
        if o.get("montage_scenes", 0) == 1:
            o.pop("montage_scenes", None)
            o.pop("montage_end", None)
    return [out[sc["scene_id"]] for sc in scenes]


def _speakers_in(sents, sc):
    return {s.get("speaker_id") for s in sents
            if sc["start_time"] <= float(s["start"]) < sc["end_time"]}


def _texture_change(activity, at, tol=1.5):
    return any(abs(m0 / 1000 - at) <= tol or abs(m1 / 1000 - at) <= tol
               for m0, m1 in activity)
