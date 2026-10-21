"""
bind_pipeline.py — the ONE bind pipeline.

BindPersons (the workflow state), p0_bind (local harness),
gate_score (the corpus gate) and critic2_run (tooth-2 harness) all bind
through `bind()` below. Before this module each carried its own copy
and they drifted — the gate was scoring a binder WITHOUT assign_roles,
i.e. not the binder that ships (review, 2026-09-04). A measurement of a
different pipeline is worse than no measurement.

`load_local_asset()` is the local-corpus mirror of the S3 loaders —
same parsers, one source of truth for the schemas.
"""
from __future__ import annotations

import json
import os

from lib import class1
from lib import cooccur
from lib import persons
from lib import reads
from lib import sources
from lib import transcript_cues
from lib import vocatives


def mine_cues(transcript, src: dict):
    """Transcript name cues + vocative resolution, language-gated.
    Returns cues (None = not assessed, the three-state contract) and
    records language/assessment into src."""
    if not transcript:
        return None
    lang = transcript.get("language") or "en"
    cues = transcript_cues.extract_cues(transcript["turns"], lang) \
        or (None if lang.split("-")[0] != "en" else [])
    if cues is not None:
        cues += vocatives.resolve_vocatives(transcript["turns"], cues)
    src["transcript_language"] = lang
    src["cues_assessed"] = cues is not None
    return cues


def bind(transcript, faces, overlay, src: dict, credits_doc=None,
         extra_cues=None, speaker_f0=None, screen_texts=None) -> dict:
    """Free evidence + optional witness credits -> the persons doc,
    critic applied, cast joined, roles assigned. THE pipeline.
    `extra_cues` lets a harness inject additional resolved name cues
    (P2-b LLM vocative resolution) AFTER the deterministic mining —
    same shapes, same downstream consumption. `speaker_f0` is
    diarization's per-cluster F0 stats (sources.load_speaker_f0),
    feeding only class1's confidence-gated compatibility veto.
    `screen_texts` is sources.parse_screen_texts() output — SME's
    verbatim on-screen text, joined below as corroboration for names
    the cues/faces already proposed (never a name source itself)."""
    cues = mine_cues(transcript, src)
    if extra_cues:
        cues = (cues or []) + list(extra_cues)
    # texts_on_screen corroboration join (user design, 2026-10-05):
    # look up every TRUSTED proposed name in SME's transcribed
    # on-screen text; matches become ordinary overlay cues — the same
    # confirming leg a paid witness title read supplies, from pixels
    # SME already read. Proposals are persons.trusted_name_proposals
    # ONLY (see its docstring for the measured flood this prevents).
    if screen_texts:
        proposals = set(persons.trusted_name_proposals(cues).values())
        joined = sources.screen_text_cues(screen_texts, proposals,
                                          existing_overlay=overlay)
        if joined:
            overlay = (overlay or []) + joined
            src["screen_text_cues"] = len(joined)
    # EARLY solo-chyron face naming (2026-09-30, completes the
    # subject-first overlay attach): a chyron over exactly one live
    # face names THAT face, and doing it BEFORE class-1 lets tier 2 —
    # which binds voices to NAMED live faces — weld the voice back
    # legitimately (the correspondent-speaks-over-own-lower-third
    # pattern the old speaker-attach got right by accident, and the
    # anchor-narrates-over-guest pattern it got wrong). Overlay is
    # NOT a registry mechanism: precedence rules treat it below
    # collection/celebrity, and an already-named face never changes.
    persons.name_solo_chyron_faces(faces, overlay)
    # tier one: the leads, matched as a small assignment problem; its
    # confirmed matches enter build_persons AHEAD of co-occurrence's
    # (heads fixed first), its abstentions ride the doc for the
    # adjudication round — a lead never exits silently unbound
    c1 = class1.match(transcript, faces, cues=cues, speaker_f0=speaker_f0)
    co = cooccur.bind_speakers_to_faces(transcript["turns"], faces,
                                        class1=c1) \
        if transcript and faces else None
    if c1 and co is not None:
        co["bindings"] = list(c1["matches"]) + co["bindings"]
    doc = persons.build_persons(transcript, faces, co, cues, overlay, src)
    if c1:
        doc["class1"] = c1
    if credits_doc:
        doc = reads.apply_cast(doc, reads.cast_from_credits(credits_doc))
    # programme duration for the speech-threading rung: the transcript's
    # own figure, else the furthest evidence timestamp (faces-only runs).
    # speech_spans feeds the rung TURN spans per speaker — never face
    # appearances, which teases scatter across the programme.
    duration = (transcript or {}).get("duration_sec") or max(
        (b for f in (faces or []) for _, b in f.get("spans") or []),
        default=None)
    spans: dict = {}
    for t in (transcript or {}).get("turns") or []:
        spans.setdefault(t["speaker_id"], []).append(
            (t["start"], t["end"]))
    doc = reads.assign_roles(doc, duration_sec=duration,
                             speech_spans=spans)
    # the speech-seconds invariant, enforced at every write point
    return persons.enforce_speech_coherence(
        doc, (transcript or {}).get("speakers") or {})


def load_local_asset(d: str):
    """Local corpus dir: transcript.json / facemetadata.json /
    shots.json / shot_metadata.jsonl / titles.json / credits.json /
    global_speakers.json (any may be absent). ->
    (transcript, faces, shots, overlay, credits, src, speaker_f0,
    screen_texts)"""
    def j(name):
        p = os.path.join(d, name)
        return json.load(open(p)) if os.path.exists(p) else None
    tx, fm, sh = j("transcript.json"), j("facemetadata.json"), j("shots.json")
    transcript = sources.parse_transcript(tx) if tx else None
    faces = sources.parse_faces(fm) if fm else None
    shots = sources.parse_shots(sh) if sh else None
    gs = j("global_speakers.json")
    speaker_f0 = sources.parse_speaker_f0(gs) if gs else None
    overlay = None
    screen_texts = None
    mdp = os.path.join(d, "shot_metadata.jsonl")
    if os.path.exists(mdp) and shots is not None:
        docs = [json.loads(ln) for ln in open(mdp) if ln.strip()]
        overlay = sources.parse_overlay_names(docs, shots)
        screen_texts = sources.parse_screen_texts(docs, shots)
        # the depicted-face guard (identical one-liner in the state)
        sources.mark_depicted_faces(
            faces or [], sources.parse_media_depiction_spans(docs, shots))
    titles, credits = j("titles.json"), j("credits.json")
    if titles:
        overlay = (overlay or []) + reads.cues_from_titles(titles)
    src = {"local_dir": d,
           "speech_diarization": bool(transcript),
           "face_recognition": bool(faces),
           "shot_segmentation": bool(shots),
           "shot_metadata_enrichment": overlay is not None}
    return (transcript, faces, shots, overlay, credits, src, speaker_f0,
            screen_texts)
