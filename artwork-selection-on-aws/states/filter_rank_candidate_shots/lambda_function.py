"""
FilterRankCandidateShots — THE selection, pure code, one retry unit:
loads the shot lattice + enrichment docs (+ cast persons, + program
structure), applies the brief amendments AFTER validating them against
the closed vocabulary (the firewall: an amendment can re-weight, it
can never invent an archetype, touch eligibility, or move the spoiler
cut), runs Layer A + FilterRank, and persists candidates.json.
"""
from __future__ import annotations

import json
import logging
import re

from lib import s3io
from lib.filter_rank import (eligibility_audit, face_presence,
                         select_candidates, top_cast, top_faces)
from lib.profiles import ARCHETYPES

logger = logging.getLogger(__name__)

_MAX_PER_ARCHETYPE = 10


def _brief_amendments(brief_result) -> dict:
    """The translated brief, VALIDATED — anything malformed degrades to
    {} with a log line, never a failed run (the brief is judgment, not
    data we require)."""
    if not isinstance(brief_result, dict) \
            or not brief_result.get("output_s3_prefix") \
            or brief_result.get("status") not in (None, "SUCCEEDED"):
        return {}
    try:
        texts = s3io.get_response_texts(brief_result["output_s3_prefix"])
        m = re.search(r"\{.*\}", texts[0], re.DOTALL)
        raw = json.loads(m.group(0))
    except Exception as exc:                # noqa: BLE001 — degrade
        logger.warning("brief amendments unreadable: %s", exc)
        return {}
    out = {}
    counts = raw.get("archetype_counts")
    if isinstance(counts, dict):
        good = {k: int(v) for k, v in counts.items()
                if k in ARCHETYPES
                and isinstance(v, (int, float))
                and 1 <= int(v) <= _MAX_PER_ARCHETYPE}
        if good:
            out["archetype_counts"] = good
    people = raw.get("must_include_people")
    if isinstance(people, list):
        out["must_include_people"] = [str(p) for p in people if p][:10]
    for key in ("avoid", "emphasis"):
        if raw.get(key):
            out[key] = raw[key]
    return out


def _fr_identity(inputs: dict, shots: dict, get_json):
    """face-recognition identity: (labels, presence) via the temporal
    track join, or None when the input is absent or carries no
    tracked people. ONE builder — the primary cascade and the
    empty-slate fallback must agree on what FR identity means."""
    if not inputs.get("face_recognition_s3uri"):
        return None
    fr = get_json(
        f"{inputs['face_recognition_s3uri']}/facemetadata.json")
    fr_names, spans_by_label = top_faces(fr)
    if not fr_names:
        return None
    return fr_names, face_presence(shots, spans_by_label)


def fill_empty_person_slates(slates, counts, shots, docs,
                             spoiler_cut, excluded, inputs, get_json):
    """Person archetypes never ship empty while face identity exists
    (user direction 2026-10-05: priority casts, THEN top faces). The
    cast path's metadata NAME join misses when SME spells a name
    differently than cast-resolution does (measured live: three
    assets with named casts shipped hero=0 while the FR temporal join
    found all four) — so any person archetype the primary identity
    left EMPTY is refilled from face-recognition, dedup'd against the
    shots the primary pass already placed. Partial slates are left
    alone: the primary identity spoke, this is a fallback, not a
    blend. -> {archetype: "face_recognition"} for the slates refilled."""
    empty = {a: counts[a] for a in counts
             if ARCHETYPES[a].get("people") and not slates.get(a)}
    if not empty:
        return {}
    fr = _fr_identity(inputs, shots, get_json)
    if not fr:
        return {}
    fr_names, fr_presence = fr
    taken = {c["shot_id"] for cs in slates.values() for c in cs}
    refills = select_candidates(shots, docs, fr_names, empty,
                                spoiler_cut, excluded, fr_presence,
                                taken=taken)
    used = {}
    for archetype, cands in refills.items():
        if cands:
            slates[archetype] = cands
            used[archetype] = "face_recognition"
    return used


def resolve_identity(inputs: dict, shots: dict, get_json):
    """Identity precedence: cast-resolution (fused evidence, real
    names, metadata name join) > face-recognition FALLBACK (top faces
    by screen time, temporal track join, anonymous labels for unnamed
    clusters) > none (person archetypes go empty). A CASCADE, not an
    elif (gap fixed 2026-10-05): a cast run whose persons are all
    unnamed (celebrity-less content) yields zero names, and stopping
    there starved the person archetypes while face-recognition
    identity sat unused — the recorded zero-images incident class.
    -> (cast_names, presence | None, identity_source | None)"""
    cast_names, presence, identity_source = [], None, None
    if inputs.get("cast_resolution_s3uri"):
        persons = get_json(
            f"{inputs['cast_resolution_s3uri']}/persons.json")
        cast_names = top_cast(persons.get("persons") or [])
        identity_source = "cast_resolution"
    if not cast_names:
        fr = _fr_identity(inputs, shots, get_json)
        if fr:
            cast_names, presence = fr
            identity_source = "face_recognition"
    return cast_names, presence, identity_source


def _excluded_spans(structure: dict) -> list:
    """Every program-structure marker category is non-content
    (bars/slates/titles/credits/blacks/textless) — union them all."""
    spans = []
    for entries in (structure.get("markers") or {}).values():
        for e in entries or []:
            if isinstance(e, dict) and "start" in e and "end" in e:
                spans.append((float(e["start"]), float(e["end"])))
    return spans


def lambda_handler(event: dict, context) -> dict:
    plan_s3uri = event["plan_s3uri"]
    plan = s3io.get_json(plan_s3uri)
    inputs = plan["inputs"]
    prefix = plan_s3uri.rsplit("/_plan/", 1)[0]

    amendments = _brief_amendments(event.get("brief_result"))
    counts = amendments.get("archetype_counts") \
        or plan["archetype_counts"]

    shots = {s["shot_id"]: s for s in s3io.get_json(
        f"{inputs['shot_segmentation_s3uri']}/embeddings/shots.json"
    )["shots"]}
    docs = {d["shot_id"]: d for d in s3io.get_jsonl(
        f"{inputs['shot_metadata_s3uri']}/shot_metadata.jsonl")}

    cast_names, presence, identity_source = resolve_identity(
        inputs, shots, s3io.get_json)
    # must-include reorders WITHIN the resolved identities (a name the
    # source never saw cannot be conjured — and it silently no-ops
    # against anonymous face labels); the first name plays lead
    must = [p for p in amendments.get("must_include_people", [])
            if p in cast_names]
    cast_names = must + [n for n in cast_names if n not in must]

    excluded = []
    if inputs.get("program_structure_s3uri"):
        excluded = _excluded_spans(s3io.get_json(
            f"{inputs['program_structure_s3uri']}/structure.json"))

    duration = max((s["end_time"] for s in shots.values()), default=0.0)
    spoiler_cut = duration * float(plan["spoiler_cutoff"])

    slates = select_candidates(shots, docs, cast_names, counts,
                               spoiler_cut, excluded, presence)
    identity_fallback = fill_empty_person_slates(
        slates, counts, shots, docs, spoiler_cut, excluded, inputs,
        s3io.get_json) if identity_source == "cast_resolution" else {}

    doc = {
        "uuid": plan["uuid"],
        "archetype_counts": counts,
        "cast_names": cast_names,
        "identity_source": identity_source,
        "identity_fallback": identity_fallback,
        "brief_amendments": amendments,
        "spoiler_cut_sec": round(spoiler_cut, 2),
        "n_excluded_spans": len(excluded),
        # why a slate is thin is part of the deliverable: per-clause
        # rejection counts + eligible counts per tier, never a silent
        # zero (user direction, 2026-10-05)
        "eligibility": eligibility_audit(shots, docs, spoiler_cut,
                                         excluded),
        "slates": slates,
    }
    candidates_s3uri = f"{prefix}/candidates.json"
    s3io.put_json(candidates_s3uri, doc)
    return {"uuid": plan["uuid"], "plan_s3uri": plan_s3uri,
            "candidates_s3uri": candidates_s3uri,
            "n_candidates": {k: len(v) for k, v in slates.items()}}
