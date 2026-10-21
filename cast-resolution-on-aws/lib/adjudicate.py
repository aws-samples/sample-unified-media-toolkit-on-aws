"""
adjudicate.py — P2 cast adjudication: the JOIN BRIDGE.

The measured gap (pairing roll-in, 2026-09-04): 101 stated
actor↔character pairs but only 3 character_name joins, because credits
pairings name CO-STARS while celebrity recognition names PRINCIPALS —
the two populations barely overlap on a casefold name match. What CAN
join them is dialogue: who gets ADDRESSED by which character name, what
the role supers say, what the quotes are about. Weighing that is a
judgment call — this module frames it for the LLM and firewalls the
output.

HARNESS-FIRST, like tooth 2: this runs via scripts/adjudicate_run.py
against the local corpus and is gate-scored BEFORE any state-machine
wiring. `cast_adjudication` is already a sanctioned character-name
mechanism in critic.py — assignments it makes survive the critic by
design, so the gate must prove precision first.

The fabrication firewall (parse-side, deterministic):
- an assigned `name` must be VERBATIM from the credits universe
  (pair actors + credit names) — the model cannot invent an actor;
- an assigned `character_name` must be verbatim from the pair
  characters — the model cannot invent a character;
- every assignment must cite evidence ids; uncited lines are dropped;
- the Billy rule (user, 2026-09-04) rides in the prompt AND the
  firewall: actor↔character NAME SIMILARITY is never evidence, and an
  assignment whose only cited evidence is the cast list itself
  (no dialogue/overlay item) is dropped.
"""
from __future__ import annotations

import json
import re

from lib import pb_common

# prompt+parser are a matched pair — the pins live in the test
# the system prompt is bound at validate from the registry's
# INTERNAL cast-adjudicate category and rides the plan
# (migrated from in-repo adjudicate_system.md, 2026-09-08)


def build_dossiers(doc: dict, cast: dict, turns: list | None = None,
                   max_quotes: int = 6) -> list[dict]:
    """Eligible persons + the shared cast universe. Eligible = the
    asset HAS a cast universe, and the person is missing part of its
    cast identity while carrying evidence that could decide it."""
    if not (cast.get("pairs") or cast.get("names")):
        return []
    quotes_by_speaker: dict[str, list] = {}
    for t in turns or []:
        quotes_by_speaker.setdefault(t["speaker_id"], []).append(t["text"])
    dossiers = []
    for p in doc["persons"]:
        if p.get("character_name"):
            continue                      # cast identity already complete
        addressed_as = sorted({
            e.get("name") for e in p.get("evidence") or []
            if isinstance(e, dict) and e.get("name")
            and (e.get("kind") or e.get("mechanism")) in
            ("vocative_resolved", "vocative", "introduction", "self_id")})
        overlay = sorted({
            e.get("name") for e in p.get("evidence") or []
            if isinstance(e, dict) and e.get("name")
            and (e.get("kind") or e.get("mechanism")) == "overlay_name"})
        quotes = []
        for sid in p.get("speaker_ids") or []:
            quotes += quotes_by_speaker.get(sid, [])[:max_quotes]
        if not (p.get("name") or addressed_as or overlay or quotes):
            continue                      # nothing to adjudicate FROM
        d = {"person_id": p["person_id"], "type": p["type"],
             "name": p.get("name"), "aliases": p.get("aliases") or [],
             "speech_sec": p.get("speech_sec"),
             "screen_sec": p.get("screen_sec")}
        ev, i = [], 0
        for who in addressed_as:
            ev.append({"id": f"v{i}", "kind": "addressed_as",
                       "name": who}); i += 1
        for who in overlay:
            ev.append({"id": f"o{i}", "kind": "overlay_name",
                       "name": who}); i += 1
        d["evidence"] = ev
        if quotes:
            d["sample_quotes"] = [{"id": f"q{j}", "text": q}
                                  for j, q in enumerate(quotes[:max_quotes])]
        dossiers.append(d)
    return dossiers


def build_records(dossiers: list[dict], cast: dict, system: str,
                  per_record: int = 8) -> list[dict]:
    universe = {"pairs": cast.get("pairs") or [],
                "credit_names": cast.get("names") or []}
    ujson = json.dumps(universe, ensure_ascii=False)
    records = []
    for i in range(0, len(dossiers), per_record):
        chunk = dossiers[i:i + per_record]
        body = "\n\n".join(
            f"## Dossier {d['person_id']}\n{json.dumps(d, ensure_ascii=False)}"
            for d in chunk)
        records.append({
            "record_id": f"adjudicate-{i // per_record:03d}",
            "system": system,
            "messages": [{"role": "user", "content": [{"text":
                f"## Cast universe\n{ujson}\n\nAdjudicate these "
                f"{len(chunk)} dossiers.\n\n{body}"}]}],
        })
    return records


_EVIDENCE_ID = re.compile(r"^[voq]\d+$")


def _tokens(*texts) -> set:
    out = set()
    for t in texts:
        out |= {w for w in re.split(r"[^\w']+", (t or "").casefold()) if w}
    return out


def _supported(row: dict, dossier: dict) -> bool:
    """The AGREEMENT check (added after the first sweep, 2026-09-05:
    the model assigned one credited correspondent's name to a person
    whose own evidence names a DIFFERENT one — verbatim-from-universe
    alone cannot see a wrong JOIN). An assignment must share a token
    with the person's OWN evidence names/aliases: the assigned name
    directly, or the assigned character (whose pair carries the name —
    the addressed-as-a-single-word-character route).

    Quotes are deliberately EXCLUDED from the agreement set: a name in
    a person's OWN speech usually means they are ADDRESSING that person
    — anti-evidence for being them (measured: the model assigned a
    character to the speaker who SAYS that character's name; the
    resolved direction belongs to vocative resolution, not to a
    substring)."""
    have = _tokens(dossier.get("name"), *(dossier.get("aliases") or []),
                   *(e.get("name") for e in dossier.get("evidence") or []))
    return bool(_tokens(row.get("name")) & have
                or _tokens(row.get("character_name")) & have)


def parse_verdicts(output_texts: list[str], cast: dict,
                   dossiers: list[dict] | None = None) -> list[dict]:
    """Model outputs -> validated assignments. The firewall: names
    verbatim from the universe, characters verbatim from the pairs,
    >=1 cited dialogue/overlay evidence id, and the assignment must
    AGREE with the person's own evidence names — anything else is
    dropped (abstention over fabrication, enforced not requested)."""
    actors = {p["actor"] for p in cast.get("pairs") or []}
    actors |= set(cast.get("names") or [])
    characters = {p["character"] for p in cast.get("pairs") or []}
    by_id = {d["person_id"]: d for d in dossiers or []}
    out = []
    for text in output_texts:
        for ln in (text or "").splitlines():
            ln = ln.strip().strip("`")
            if not ln.startswith("{"):
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if row.get("verdict") != "assign" \
                    or not isinstance(row.get("person_id"), int):
                continue
            ids = [i for i in row.get("evidence") or []
                   if isinstance(i, str) and _EVIDENCE_ID.match(i)]
            if not ids:
                continue                  # uncited: dropped
            name = row.get("name")
            character = row.get("character_name")
            if name is not None and name not in actors:
                continue                  # invented actor: dropped
            if character is not None and character not in characters:
                continue                  # invented character: dropped
            if name is None and character is None:
                continue
            d = by_id.get(row["person_id"])
            if d is not None and not _supported(row, d):
                continue                  # wrong join: dropped
            out.append({"person_id": row["person_id"], "name": name,
                        "character_name": character, "evidence_ids": ids,
                        "reason": str(row.get("reason") or "")[:300]})
    return out


def apply_verdicts(doc: dict, verdicts: list[dict]) -> dict:
    """Additive, mechanism cast_adjudication (a sanctioned character
    mechanism — critic.critique accepts it). Never overwrites an
    existing name: an adjudicated name lands on UNNAMED persons only;
    character_name lands wherever it was missing."""
    by_id = {p["person_id"]: p for p in doc["persons"]}
    for v in verdicts:
        p = by_id.get(v["person_id"])
        if p is None:
            continue
        entry = {"mechanism": "cast_adjudication",
                 "evidence_ids": v["evidence_ids"],
                 "reason": v["reason"]}
        changed = False
        if v.get("character_name") and not p.get("character_name"):
            p["character_name"] = v["character_name"]
            entry["character"] = v["character_name"]
            changed = True
        if v.get("name") and not p.get("name"):
            p["name"] = v["name"]
            p.setdefault("name_mechanisms", []).append("cast_adjudication")
            entry["actor"] = v["name"]
            changed = True
        if changed:
            p.setdefault("evidence", []).append(entry)
    return doc
