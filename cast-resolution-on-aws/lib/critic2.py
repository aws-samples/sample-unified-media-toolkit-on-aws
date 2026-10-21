"""
critic2.py — tooth 2: LLM dossier critique. REPORT-ONLY by decision
(user, 2026-09-04): verdicts land as FLAGS in critic_review, never as
demotions, until the corpus gate has scored this critic's own precision —
a trigger-happy critic silently costs recall.

First sweep's lesson (36 assets, 2026-09-04): asked as a general
skeptic, the model flagged 94% of named persons — 570 of 638 flags were
absence-based doubt ("only the celebrity mechanism, no other evidence"),
i.e. re-auditing an external oracle. The 21 flags alleging an actual
CONTRADICTION between evidence items were the only good ones. Hence the
four mechanisms below, all structural:

1. the prompt defines suspicion as CONTRADICTION BETWEEN EVIDENCE ITEMS
   and states that absence of more evidence is never suspect (single-
   source sufficiency is the confirmation rules' job, not the critic's);
2. every evidence item carries a class — `observed` (external recognizer
   or text read verbatim off screen/transcript) vs `inferred` (derived
   by our statistics/voting) — and the critic is told observed items are
   independently produced, never to be re-audited for reliability;
3. only persons with something that CAN contradict get a dossier
   (see _reviewable) — one oracle name with nothing to compare against
   has nothing to critique and rides free;
4. a flag must cite >=2 distinct evidence/quote ids in `conflicts`, at
   least one being evidence — the parser drops anything else, so
   vibes-based flags are unparseable by construction.
"""
from __future__ import annotations

import json
import re

from lib import pb_common

# prompt+parser are a matched pair — the pins live in the test
# system prompt bound at validate (registry INTERNAL cast-critic2;
# migrated from in-repo critic2_system.md, 2026-09-08)


# mechanisms whose payload is an external verdict or verbatim text —
# the critic may use them, never re-audit them
_OBSERVED = ("celebrity", "collection", "credits_cast", "overlay_name",
             "overlay_text", "self_id", "sign_off", "introduction")


def _evidence_class(e: dict) -> str:
    k = e.get("kind") or e.get("mechanism") or ""
    return "observed" if k in _OBSERVED else "inferred"


def _reviewable(p: dict) -> bool:
    """SUSPICION-ONLY (user principle, 2026-09-24: tokens are spent
    only on what the deterministic layer has NOT confirmed). The old
    selector reviewed corroboration itself — two agreeing mechanisms,
    aliases, any character — which is reviewing exactly the persons we
    confirmed hardest, and the precedence/dominant-name rules mint
    aliases on confirmed leads by design. Reviewable now means a live
    red flag or a genuine non-answer:
    - an ABSTAINED name (conflict residue: aliases with no name),
    - a mixed-voice cluster (self-identifies as two people),
    - a coherence repair (speech seconds had to be recomputed),
    - a tooth-1 veto on the person,
    - a character inferred from SPEECH (spoken_address) with no
      credits card behind it — deterministic credits are confirmed,
      an inferred address is worth one look."""
    if p.get("aliases") and not p.get("name"):
        return True
    ev = [e for e in p.get("evidence") or [] if isinstance(e, dict)]
    if any(e.get("mixed_voice") for e in ev):
        return True
    if any(e.get("mechanism") in ("coherence_repair", "critic_veto")
           for e in ev):
        return True
    mechs = {e.get("mechanism") for e in ev}
    if p.get("character_name") and "credits_cast" not in mechs:
        return True
    return False


def build_dossiers(doc: dict, max_quotes: int = 4,
                   turns: list | None = None) -> list[dict]:
    """persons doc -> reviewable dossiers. Evidence items are id'd
    (e0, e1, ...) and classed observed/inferred; quotes are id'd
    (q0, ...) — the ids the model must cite in `conflicts`."""
    quotes_by_speaker: dict[str, list] = {}
    for t in turns or []:
        quotes_by_speaker.setdefault(t["speaker_id"], []).append(t["text"])
    dossiers = []
    for p in doc["persons"]:
        if not p.get("name") and not p.get("aliases"):
            continue
        if not _reviewable(p):
            continue
        evidence = [
            {"id": f"e{i}", "class": _evidence_class(e), **e}
            for i, e in enumerate(p.get("evidence") or [])
            if isinstance(e, dict)]
        d = {"person_id": p["person_id"], "type": p["type"],
             "name": p.get("name"), "aliases": p.get("aliases") or [],
             "character_name": p.get("character_name"),
             "speech_sec": p.get("speech_sec"),
             "screen_sec": p.get("screen_sec"),
             "name_mechanisms": p.get("name_mechanisms") or [],
             "evidence": evidence}
        quotes = []
        for sid in p.get("speaker_ids") or []:
            quotes += quotes_by_speaker.get(sid, [])[:max_quotes]
        if quotes:
            d["sample_quotes"] = [{"id": f"q{i}", "text": q}
                                  for i, q in enumerate(quotes[:max_quotes])]
        dossiers.append(d)
    return dossiers


def build_records(dossiers: list[dict], system: str,
                  per_record: int = 8) -> list[dict]:
    """bedrock-inference records: {record_id, system, messages}. Dossiers
    are batched per record; the JSONL output keys lines by person_id, so
    one bad line damages only itself."""
    records = []
    for i in range(0, len(dossiers), per_record):
        chunk = dossiers[i:i + per_record]
        body = "\n\n".join(
            f"## Dossier {d['person_id']}\n{json.dumps(d, ensure_ascii=False)}"
            for d in chunk)
        records.append({
            "record_id": f"critic-{i // per_record:03d}",
            "system": system,
            "messages": [{"role": "user", "content": [
                {"text": f"Review these {len(chunk)} dossiers.\n\n{body}"}]}],
        })
    return records


_CONFLICT_ID = re.compile(r"^[eq]\d+$")


def _cited(row: dict) -> bool:
    """>=2 distinct well-formed ids, >=1 of them evidence — the
    deterministic filter that makes vibes-based flags unparseable."""
    ids = row.get("conflicts")
    if not isinstance(ids, list):
        return False
    ids = {i for i in ids if isinstance(i, str) and _CONFLICT_ID.match(i)}
    return len(ids) >= 2 and any(i.startswith("e") for i in ids)


def parse_reviews(output_texts: list[str]) -> list[dict]:
    """Model outputs -> flags. Strict-line parse; a bad or uncited line
    is dropped (report-only: losing a flag is acceptable, inventing one
    is not)."""
    flags = []
    for text in output_texts:
        for ln in (text or "").splitlines():
            ln = ln.strip().strip("`")
            if not ln.startswith("{"):
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if row.get("verdict") == "suspect" \
                    and isinstance(row.get("person_id"), int) \
                    and _cited(row):
                flags.append({"person_id": row["person_id"],
                              "suspect_field": row.get("suspect_field"),
                              "conflicts": row["conflicts"],
                              "reason": str(row.get("reason") or "")[:300],
                              "mechanism": "critic2_llm",
                              "action": "flag_only"})
    return flags


def attach_flags(doc: dict, flags: list[dict]) -> dict:
    """Report-only: flags land in doc['critic_review'] and on the person
    as evidence — names and types are NEVER changed here."""
    doc["critic_review"] = flags
    by_id = {p["person_id"]: p for p in doc["persons"]}
    for f in flags:
        p = by_id.get(f["person_id"])
        if p is not None:
            p.setdefault("evidence", []).append(f)
    return doc
