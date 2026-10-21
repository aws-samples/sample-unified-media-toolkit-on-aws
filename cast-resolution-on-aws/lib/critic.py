"""
critic.py — deterministic coherence checks. DEMOTE-ONLY, by design:

    The critic never renames and never promotes. It vetoes a merge before
    it happens, or demotes a name back to uncertain after assembly —
    always attaching a machine-readable reason. A critic that "fixes"
    results is a second binder with the same failure modes; a critic that
    only vetoes cannot introduce a wrong name.

Two hooks:
  merge_veto(...)   — called by persons.build_persons BEFORE a same-name
                      merge: prevention beats repair when the check is
                      cheap and local.
  critique(doc)     — post-assembly sweep for contradictions no single
                      merge decision can see (cross-person name
                      collisions).

Every veto lands in doc["vetoes"] and as an evidence entry on the persons
it touched, so the corpus gate can measure the critic's own precision —
a trigger-happy critic silently costs recall.
"""
from __future__ import annotations

from lib.pb_common import (
    MERGE_FACE_OVERLAP_TOLERANCE_SEC,
    MERGE_SPEECH_OVERLAP_TOLERANCE_SEC,
    spans_overlap_sec,
)


def merge_veto(pa: dict, pb: dict, speech_spans: dict,
               face_spans: dict) -> dict | None:
    """Should persons pa and pb NOT be merged? -> veto record or None.

    A person cannot talk over themselves: two speaker clusters with
    overlapping speech are two people, whatever name they share. Faces
    get a looser bar — one real person CAN be on screen twice (live +
    photo, monitor, split screen replay), so only substantial overlap
    vetoes. (Tracker-duplicate tracks sharing a recognizer face UUID
    never reach this check: parse_faces coalesces them at the SOURCE,
    with an audit trail — review, 2026-09-06; a consumer-side exemption
    here was silent and left the duplicate inflating co-occurrence.)"""
    a_speech = [s for sid in pa["speaker_ids"]
                for s in speech_spans.get(sid, [])]
    b_speech = [s for sid in pb["speaker_ids"]
                for s in speech_spans.get(sid, [])]
    ov = spans_overlap_sec(a_speech, b_speech)
    if ov > MERGE_SPEECH_OVERLAP_TOLERANCE_SEC:
        return {"check": "speech_self_overlap",
                "overlap_sec": round(ov, 2),
                "speaker_ids": pa["speaker_ids"] + pb["speaker_ids"]}
    a_face = [s for fid in pa["face_ids"] for s in face_spans.get(fid, [])]
    b_face = [s for fid in pb["face_ids"] for s in face_spans.get(fid, [])]
    ov = spans_overlap_sec(a_face, b_face)
    if ov > MERGE_FACE_OVERLAP_TOLERANCE_SEC:
        return {"check": "face_self_overlap",
                "overlap_sec": round(ov, 2),
                "face_ids": pa["face_ids"] + pb["face_ids"]}
    return None


def critique(doc: dict) -> dict:
    """Post-assembly sweep. Currently one check: two DISTINCT persons
    finalizing with the same name is a contradiction the pipeline could
    not resolve (a vetoed merge, or independent naming of different
    identities) — demote BOTH names to aliases rather than let a consumer
    trust a coin flip. Mutates and returns doc; every demotion is
    recorded in doc['vetoes'] and on the person."""
    doc.setdefault("vetoes", [])

    # character_name requires a STATED mapping or gate-approved
    # adjudication evidence — never inference from the actor's own name
    # (user rule, 2026-09-04: Billy / "Billy Ray Marsh" is the cautionary
    # coincidence). A character_name without a recognized mechanism is
    # removed, whatever produced it. spoken_address joined the list
    # 2026-09-23 (user-approved): a vote-aggregated vocative on a person
    # the registry independently named is the PROGRAMME stating what it
    # calls them — evidence from other speakers' words, never derived
    # from the actor's own name, so the Billy/Billy-Bob class stays
    # excluded.
    _CHARACTER_MECHANISMS = ("credits_cast", "cast_sheet",
                             "cast_adjudication", "spoken_address")
    for p in doc["persons"]:
        if not p.get("character_name"):
            continue
        supported = any(e.get("mechanism") in _CHARACTER_MECHANISMS
                        for e in p.get("evidence") or [])
        if not supported:
            veto = {"check": "unsupported_character_name",
                    "person_ids": [p["person_id"]],
                    "character_name": p["character_name"]}
            doc["vetoes"].append(veto)
            p.pop("character_name", None)
            p.setdefault("evidence", []).append(
                {"mechanism": "critic_veto", **veto})

    by_name: dict[str, list] = {}
    for p in doc["persons"]:
        if p.get("name"):
            by_name.setdefault(p["name"].casefold(), []).append(p)
    for key, group in by_name.items():
        if len(group) < 2:
            continue
        # collection wins at contradictions (user trust hierarchy,
        # affirmed 2026-09-09 for THIS rule): when exactly one collider
        # carries the name from the curated collection, that assertion
        # is the trustworthy one — keep it, demote only the others.
        # Two collection-named colliders remain a real contradiction:
        # strip all, as before. Over-clustered diarization makes the
        # one-collection case COMMON (a merged voice adjudicates to a
        # host's name while the host's face already owns it), and
        # stripping both erased the host from the whole asset.
        keepers = [p for p in group
                   if "collection" in (p.get("name_mechanisms") or [])]
        losers = group if len(keepers) != 1 else \
            [p for p in group if p is not keepers[0]]
        veto = {"check": "name_collision", "name": group[0]["name"],
                "person_ids": [p["person_id"] for p in group],
                **({"kept_person_id": keepers[0]["person_id"],
                    "resolution": "collection_preferred"}
                   if len(keepers) == 1 else {})}
        doc["vetoes"].append(veto)
        for p in losers:
            p["aliases"] = sorted(set((p.get("aliases") or [])
                                      + [p["name"]]))
            p["name"] = None
            p["type"] = "uncertain"
            p.setdefault("evidence", []).append(
                {"mechanism": "critic_veto", **veto})
    return doc
