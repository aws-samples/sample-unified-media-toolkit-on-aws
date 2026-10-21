"""
vocatives.py — resolve vocative cues to the speaker they address.

The largest untapped transcript signal (P0: 598 vocatives pooled in news,
301 in episodic, vs 19 self-IDs). Deterministic first (house rule): a
vocative in speaker A's turn addresses A's conversation partner, so the
ADJACENT turn by a different speaker is the candidate — and the power is
in AGGREGATION: one "Tom," is noise, but a speaker consistently addressed
as "Tom" across many separate exchanges, by more than one addresser, is a
name. LLM adjudication (P2) can extend this to harder forms; it must not
replace the vote counting.

Adjacency rule per cue (turn i by speaker A):
  - leading vocative ("Tom, look at this")  -> the PREVIOUS different-
    speaker turn within GAP (A responds to Tom);
  - trailing vocative ("…, Tom.")           -> the NEXT different-speaker
    turn within GAP (A hands off to Tom);
  - the other neighbor is the fallback when the preferred one is absent.
A candidate whose speaker is A itself is discarded (self-address is not
evidence). Votes where the named speaker ALSO self-identified differently
are dropped by persons.py's normal conflict handling, not here.
"""
from __future__ import annotations

import re

from lib.pb_common import VOCATIVE_MAX_GAP_SEC, VOCATIVE_MIN_VOTES, \
    VOCATIVE_MIN_LEAD


def _ordinary_words(turns: list[dict]):
    """Tokens the transcript treats as WORDS, not names. The structural
    screen (measured 2026-09-23: "Mierda", "Ah", "Like", "Focus" all
    minted as vocative names): a NAME is capitalized wherever it
    appears, so its lowercase count is ~0 while its MID-SENTENCE
    capitalized count grows with use; an interjection / expletive /
    ordinary word is the reverse. Majority rule, not a single-occurrence
    veto — one ASR-lowercased "grace" must not erase a character named
    Grace who is capitalized mid-sentence throughout (the screen must
    hold on EVERY drama, not just the measured episode). Sentence-
    initial capitals prove nothing and count for neither side.
    Language-agnostic — no curated stoplist; caseless scripts never
    populate either count and the screen no-ops.

    Returns (words, referenced): `words` = tokens whose lowercase uses
    outnumber mid-sentence capitalized ones; `referenced` = tokens seen
    capitalized MID-sentence at least once. The second set exists for
    the boundary-only interjection class (measured live 2026-09-23:
    "Yowie" and "Ah" survived the majority rule — an exclamation is
    ALWAYS sentence-initial, so it leaves no lowercase evidence at all,
    while a real name gets referenced or trailed mid-sentence somewhere:
    "ask Craig", "over to you, Tom")."""
    lower: dict[str, int] = {}
    cap_mid: dict[str, int] = {}
    for t in turns:
        text = t.get("text") or ""
        boundary = True                       # start of turn = boundary
        for m in re.finditer(r"[^\W\d_]+|[.!?]", text):
            tok = m.group(0)
            if tok in ".!?":
                boundary = True
                continue
            key = tok.lower()
            if tok.islower():
                lower[key] = lower.get(key, 0) + 1
            elif not boundary:
                cap_mid[key] = cap_mid.get(key, 0) + 1
            boundary = False
    words = {w for w, n in lower.items() if n > cap_mid.get(w, 0)}
    return words, set(cap_mid)


def name_screen(turns: list[dict]):
    """-> predicate: is this candidate a NAME by the transcript's own
    usage? ONE home for the two structural screens (word-majority +
    referenced-mid-sentence) — the deterministic resolver AND the
    round-1 LLM window builder both consult it, so tokens are never
    spent asking about a candidate the screens already REJECTED
    (measured 2026-09-24: 48% of round-1 windows were junk sites)."""
    ordinary, referenced = _ordinary_words(turns)

    def _is_name(name: str) -> bool:
        toks = re.findall(r"[^\W\d_]+", name)
        if any(t.lower() in ordinary for t in toks):
            return False                  # the transcript uses it as a word
        # a name is REFERRED TO mid-sentence somewhere; a boundary-only
        # token ("Yowie!", "Ah,") never is — abstain (caseless scripts
        # have an empty referenced set, so this gate no-ops there
        # together with the words screen)
        if referenced and toks and toks[0].lower() not in referenced:
            return False
        return True
    return _is_name


def resolve_vocatives(turns: list[dict], cues: list[dict]) -> list[dict]:
    """-> resolved name cues: {kind: 'vocative_resolved', name,
    speaker_id (the ADDRESSEE), votes, addressers, confidence: 'high'}.
    Only emits when the vote threshold and lead are met — below that the
    raw vocative cues remain available but bind nothing. A candidate any
    of whose tokens the transcript also uses as a lowercase word is a
    WORD, not a name, and never resolves (abstention over fabrication —
    votes cannot launder an interjection into a person)."""
    vocatives = [c for c in cues if c["kind"] == "vocative"]
    if not vocatives:
        return []
    _is_name = name_screen(turns)
    vocatives = [c for c in vocatives if _is_name(c["name"])]
    if not vocatives:
        return []
    by_sentence = {t["sentence_id"]: i for i, t in enumerate(turns)}

    def neighbor(i: int, step: int, sid: str):
        j = i + step
        while 0 <= j < len(turns):
            t = turns[j]
            gap = (turns[i]["start"] - t["end"] if step < 0
                   else t["start"] - turns[i]["end"])
            if gap > VOCATIVE_MAX_GAP_SEC:
                return None
            if t["speaker_id"] != sid:
                return t["speaker_id"]
            j += step
        return None

    # (addressee_speaker, name) -> {votes, addressers}
    votes: dict[tuple[str, str], dict] = {}
    for c in vocatives:
        i = by_sentence.get(c["sentence_id"])
        if i is None:
            continue
        a = turns[i]["speaker_id"]
        leading = turns[i]["text"].startswith(c["name"])
        first, second = (-1, 1) if leading else (1, -1)
        target = neighbor(i, first, a) or neighbor(i, second, a)
        if not target:
            continue
        v = votes.setdefault((target, c["name"]),
                             {"votes": 0, "addressers": set()})
        v["votes"] += 1
        v["addressers"].add(a)

    # per addressee: the winning name must clear the floor AND lead the
    # runner-up decisively (two names voted onto one speaker = noise)
    per_speaker: dict[str, list] = {}
    for (sid, name), v in votes.items():
        per_speaker.setdefault(sid, []).append((v["votes"], name, v))
    resolved = []
    for sid, cands in per_speaker.items():
        cands.sort(reverse=True)
        top_votes, name, v = cands[0]
        runner = cands[1][0] if len(cands) > 1 else 0
        if top_votes < VOCATIVE_MIN_VOTES:
            continue
        if runner and top_votes / runner < VOCATIVE_MIN_LEAD:
            continue
        resolved.append({"kind": "vocative_resolved", "name": name,
                         "speaker_id": sid, "votes": top_votes,
                         "addressers": sorted(v["addressers"]),
                         "confidence": "high"})
    return resolved
