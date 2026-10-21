"""
vocatives_llm.py — P2-b: LLM vocative resolution beyond adjacency.

The measured bottleneck (P2-a sweep, 2026-09-05): cast adjudication's
recall is capped by how many vocatives resolve to an ADDRESSEE — a
paired single-word character is spoken in dialogue yet adjacency voting
(vocatives.py) never resolves it, so the credits pair cannot attach.
This module sends the UNRESOLVED vocative sites to the LLM with their
conversation windows and asks only the narrow question adjacency
guesses at: which speaker in this window is being addressed?

Deterministic-first is preserved by construction:
- adjacency voting runs FIRST; names it resolved are never re-asked;
- the LLM replaces only the per-cue NEIGHBOR HEURISTIC — its answers
  are still VOTES, aggregated under the same thresholds
  (VOCATIVE_MIN_VOTES, VOCATIVE_MIN_LEAD) as the deterministic voter;
- the firewall drops any answer that is not a speaker_id present in
  the window, or that is the cue speaker itself (self-address is not
  evidence) — fabrication is unparseable, not just discouraged.

Emitted cues are ordinary {kind: "vocative_resolved"} name evidence
with resolver: "llm" provenance, so persons assembly, the critic, and
cast adjudication consume them with zero new code paths.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from lib.pb_common import VOCATIVE_MIN_LEAD, VOCATIVE_MIN_VOTES
from lib.transcript_cues import _ADDRESS_TERMS

# prompt+parser are a matched pair — the pins live in the test
# system prompt bound at validate (registry INTERNAL
# cast-vocatives; migrated from in-repo vocatives_system.md,
# 2026-09-08)

_MAX_SITES_PER_NAME = 6
_WINDOW_TURNS = 3          # turns kept on each side of the cue site


def build_windows(turns: list[dict], cues: list[dict],
                  resolved: list[dict]) -> list[dict]:
    """Unresolved vocative sites -> conversation windows.
    `resolved` is adjacency's output; its names are settled and never
    re-asked (deterministic first)."""
    settled = {c["name"].casefold() for c in resolved}
    by_sentence = {t["sentence_id"]: i for i, t in enumerate(turns)}
    per_name: dict[str, int] = defaultdict(int)
    windows = []
    for c in cues:
        if c["kind"] != "vocative" or c["name"].casefold() in settled:
            continue
        if per_name[c["name"].casefold()] >= _MAX_SITES_PER_NAME:
            continue
        i = by_sentence.get(c["sentence_id"])
        if i is None:
            continue
        per_name[c["name"].casefold()] += 1
        lo, hi = max(0, i - _WINDOW_TURNS), min(len(turns),
                                                i + _WINDOW_TURNS + 1)
        windows.append({
            "id": f"w{len(windows):03d}",
            "name": c["name"],
            "cue_speaker": turns[i]["speaker_id"],
            "speakers": sorted({t["speaker_id"] for t in turns[lo:hi]}),
            "lines": [
                {"speaker_id": t["speaker_id"],
                 "text": t["text"],
                 "site": j == i}
                for j, t in ((j, turns[j]) for j in range(lo, hi))],
        })
    return windows


def build_records(windows: list[dict], system: str,
                  per_record: int = 8) -> list[dict]:
    records = []
    for i in range(0, len(windows), per_record):
        chunk = windows[i:i + per_record]
        parts = []
        for w in chunk:
            lines = "\n".join(
                f"{'>>> ' if ln['site'] else '    '}"
                f"{ln['speaker_id']}: {ln['text']}"
                for ln in w["lines"])
            parts.append(f"## Window {w['id']} — who is addressed as "
                         f"{w['name']!r}?\n{lines}")
        records.append({
            "record_id": f"vocative-{i // per_record:03d}",
            "system": system,
            "messages": [{"role": "user", "content": [{"text":
                f"Resolve these {len(chunk)} windows.\n\n"
                + "\n\n".join(parts)}]}],
        })
    return records


_WINDOW_ID = re.compile(r"^w\d{3}$")


def parse_answers(output_texts: list[str],
                  windows: list[dict]) -> list[dict]:
    """Model outputs -> firewalled per-window votes. Dropped: unknown
    window ids, addressees not present in the window, and the cue
    speaker itself."""
    by_id = {w["id"]: w for w in windows}
    votes, answered = [], set()
    for text in output_texts:
        for ln in (text or "").splitlines():
            ln = ln.strip().strip("`")
            if not ln.startswith("{"):
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            wid = row.get("window_id") or ""
            w = by_id.get(wid) if _WINDOW_ID.match(str(wid)) else None
            addressee = row.get("addressee")
            if w is None or not addressee or wid in answered:
                continue          # one window = one cue site = ONE vote
            answered.add(wid)
            if addressee not in w["speakers"] \
                    or addressee == w["cue_speaker"]:
                continue                     # not in window / self-address
            votes.append({"window_id": wid, "name": w["name"],
                          "addressee": addressee,
                          "addresser": w["cue_speaker"],
                          "reason": str(row.get("reason") or "")[:200]})
    return votes


def _name_tokens(text: str) -> set:
    """Casefolded tokens MINUS address terms: 'Detective Porter' and
    'Detective Vega' must never cluster on the shared honorific — the
    roles-never-names distinction transcript_cues already curates
    (review finding, 2026-09-05)."""
    return {w for w in text.casefold().replace(".", "").split()
            if w not in _ADDRESS_TERMS}


def _same_name(a: str, b: str) -> bool:
    """ASR spelling variants of one address ('Porter'/'Porder'/
    'Detective Porter') split the tally and starve every variant below
    the vote threshold (measured 2026-09-05: one addressee, four 1-vote
    variants). Two names cluster when their NON-honorific tokens
    overlap, or when those remainders are near-identical strings — the
    OCR wobble-voting collapse applied to ASR, with the address-term
    vocabulary stripped first so a shared honorific never fuses two
    different names."""
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        # a pure honorific ('Mr.') clusters only with its exact self
        return a.casefold() == b.casefold()
    if ta & tb:
        return True
    return SequenceMatcher(None, " ".join(sorted(ta)),
                           " ".join(sorted(tb))).ratio() >= 0.8


def aggregate(votes: list[dict]) -> list[dict]:
    """Votes -> resolved cues, under the SAME rule shape as the
    deterministic voter: >= VOCATIVE_MIN_VOTES for the winner and a
    VOCATIVE_MIN_LEAD margin over the runner-up. Votes are grouped per
    ADDRESSEE with spelling variants clustered (see _same_name); the
    canonical spelling is the modal variant, longest on ties."""
    by_addressee: dict[str, list[dict]] = defaultdict(list)
    for v in votes:
        by_addressee[v["addressee"]].append(v)

    candidates = []
    for sid, vs in by_addressee.items():
        # cluster this addressee's name variants (multiple DISTINCT
        # names on one speaker is fine — nicknames, rank + surname)
        clusters: list[dict] = []
        for v in vs:
            home = next((c for c in clusters
                         if any(_same_name(v["name"], n)
                                for n in c["names"])), None)
            if home is None:
                home = {"sid": sid, "names": Counter(),
                        "addressers": set()}
                clusters.append(home)
            home["names"][v["name"]] += 1
            home["addressers"].add(v["addresser"])
        candidates += clusters

    def n_votes(c):
        return sum(c["names"].values())

    out = []
    for c in candidates:
        if n_votes(c) < VOCATIVE_MIN_VOTES:
            continue
        # the SAME name pointing at a DIFFERENT speaker competes — one
        # address resolving two ways is the ambiguity the lead test
        # exists for
        runner = max((n_votes(o) for o in candidates
                      if o["sid"] != c["sid"]
                      and any(_same_name(a, b) for a in c["names"]
                              for b in o["names"])), default=0)
        if runner and n_votes(c) / runner < VOCATIVE_MIN_LEAD:
            continue
        top = max(c["names"].items(), key=lambda kv: (kv[1], len(kv[0])))
        out.append({"kind": "vocative_resolved", "name": top[0],
                    "speaker_id": c["sid"], "votes": n_votes(c),
                    "variants": sorted(c["names"]),
                    "addressers": sorted(c["addressers"]),
                    "confidence": "high", "resolver": "llm"})
    return out
