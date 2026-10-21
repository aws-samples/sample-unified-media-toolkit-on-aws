"""
sources.py — normalize sibling workflow outputs into the P0 input shapes.

Split on purpose (refactor rule: one source of truth): `parse_*` functions
are PURE — document dict in, P0 shape out — and are the only place a
sibling's schema is known. The `load_*` wrappers only fetch (S3 prefix
layout lives there); local harnesses call the parsers directly on files.

Every loader tolerates an ABSENT source by returning None — the caller
records which sources were present (three-state semantics: absent input =>
dependent outputs absent, never empty). Nothing here mutates or re-writes
a sibling artifact; identity is an overlay (README.md §1, locked).
"""
from __future__ import annotations

import json

import boto3

from lib import pb_common

_s3 = boto3.client("s3")


def _get_json(uri: str):
    bucket, key = uri[5:].split("/", 1)
    return json.loads(_s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def _get_jsonl(uri: str):
    bucket, key = uri[5:].split("/", 1)
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return [json.loads(ln) for ln in body.splitlines() if ln.strip()]


# ── pure parsers ──────────────────────────────────────────────────────────────

def parse_transcript(doc: dict):
    """speech-diarization final_transcript.json ->
    {speakers: {id: speech_sec}, turns: [{speaker_id, start, end, text,
    sentence_id}], duration_sec}."""
    turns = [{"speaker_id": s.get("speaker_id"),
              "start": float(s.get("start", 0.0)),
              "end": float(s.get("end", s.get("start", 0.0))),
              "text": (s.get("text") or "").strip(),
              "sentence_id": s.get("id", i)}
             for i, s in enumerate(doc.get("sentences") or [])
             if s.get("speaker_id") and (s.get("text") or "").strip()]
    speakers: dict[str, float] = {}
    for t in turns:
        # "UNKNOWN" is diarization's rejection label (speech it could
        # not confidently attribute), not a speaker: it must never
        # become a person — its aggregate is a union of DIFFERENT
        # voices, so cues, overlays, face binding, and top-talker
        # recovery would all misattribute onto it. The turns stay:
        # they are real speech, and keeping them preserves overlay
        # ambiguity abstention and speech-share denominators.
        if t["speaker_id"] == "UNKNOWN":
            continue
        speakers[t["speaker_id"]] = (speakers.get(t["speaker_id"], 0.0)
                                     + max(0.0, t["end"] - t["start"]))
    # overlap_segments (2026-09-18): cross-talk seconds credit BOTH
    # voices — the second voice is structurally absent from sentences[]
    # (one speaker per instant), so interrupted speakers were
    # undercounted exactly where the share/recovery gates bite.
    # SUGGESTIVE, not ground truth (producer contract: identity mapping
    # 81% precise, zh 56-69%) — safe HERE because this is aggregate
    # arithmetic where per-segment errors wash out, and it feeds only
    # denominators/ordering, never a naming mechanism. Only speakers
    # sentences[] already established accrue: an id appearing SOLELY
    # in overlap evidence is too weak to introduce a person.
    for seg in doc.get("overlap_segments") or []:
        dur = max(0.0, float(seg.get("end", 0.0))
                  - float(seg.get("start", 0.0)))
        for spk in seg.get("speakers") or []:
            if spk in speakers:
                speakers[spk] += dur
    return {"turns": turns, "speakers": speakers,
            "overlap_segments": doc.get("overlap_segments") or [],
            "duration_sec": doc.get("audio_duration_sec"),
            "language": (doc.get("language") or "").lower() or None,
            "language_confidence": doc.get("language_confidence")}


def parse_faces(doc: dict):
    """face-recognition facemetadata.json -> faces list:
    {face_person_id, name (None unless the SOURCE named it), name_source,
    screen_sec, spans: [(t0, t1), ...]}.

    TRACKER-DUPLICATE COALESCING (review, 2026-09-06; celebrity ids
    joined 2026-09-07): person entries sharing a non-empty recognizer
    identity assertion — the collection face UUID, or the celebrity id
    the recognizer names a person by — are one person BY THE SOURCE'S
    OWN CLAIM (measured: 5 face-UUID groups / 36 assets, all in 2h
    news, one 99%-identical parallel pair driving the gate's one false
    veto; plus one same-confidence celebrity pair the UUID class could
    not see). They coalesce HERE, at the producer boundary — union
    spans, screen_sec recomputed from the merged spans — so no consumer
    ever sees the duplicate: no false merge veto, no co-occurrence crowd
    inflation, no screen_sec double-count, and unnamed duplicates are
    covered too (a consumer-side veto exemption caught only named
    pairs). `coalesced_ids` is the audit trail; persons seeding records
    it as evidence."""
    faces, by_key = [], {}
    for p in doc.get("people") or []:
        spans = [(a["begin_ms"] / 1000.0, a["end_ms"] / 1000.0)
                 for a in p.get("appearances") or []]
        # BOTH key spaces, namespaced (review, 2026-09-07): an entry can
        # carry a face UUID AND a celebrity id (auto-indexed celebrity —
        # the recognizer's default-policy shape), and an `or`-precedence
        # single key silently hid the shared celebrity assertion behind
        # the distinct face UUIDs — the exact join this exists for.
        keys = [(kind, val) for kind, val in
                (("face_id", p.get("face_id")),
                 ("celebrity_id", p.get("celebrity_id"))) if val]
        face = {
            "face_person_id": p["person_id"],
            # placeholder names ("Person_0") are NOT names — only carry a
            # name the source actually asserted (named=True)
            "name": p.get("name") if p.get("named") else None,
            "name_source": p.get("name_source") if p.get("named") else None,
            "screen_sec": float(p.get("total_screen_time_sec") or 0.0),
            # gender for the class-1 F0 veto: the celebrity registry's
            # assertion (known_gender) over the visual estimate — a
            # STATISTIC consumed only as a confidence-gated
            # compatibility check, never a verdict about anyone
            "gender": p.get("known_gender") or p.get("gender"),
            "age_range": p.get("age_range"),
            "spans": spans,
            # TYPED audit record of the recognizer's assertions — the
            # evidence row must say WHICH key space asserted identity
            "recognizer_ids": {kind: val for kind, val in keys},
        }
        targets = []
        for k in keys:
            hit = by_key.get(k)
            if hit is None or hit in targets:
                continue
            # relative-FP rule (user, 2026-09-07): celebrity detection
            # has a higher false-positive rate than SearchFaces against
            # the curated collection, so at a CONTRADICTION the
            # collection's assertion wins — a celebrity-id match never
            # welds two entries the collection distinguishes. An
            # uncontradicted celebrity id still welds normally.
            if k[0] == "celebrity_id" and _collection_conflict(hit, face):
                continue
            targets.append(hit)
        if not targets:
            faces.append(face)
            for k in keys:
                by_key.setdefault(k, face)
            continue
        primary = targets[0]
        # an entry bridging TWO previously-separate faces welds them too
        # (its keys are the recognizer asserting all three are one) —
        # unless the collection distinguishes the pair (relative-FP rule)
        for extra in targets[1:]:
            if _collection_conflict(primary, extra):
                continue
            _coalesce_faces(primary, extra)
            faces.remove(extra)
            for k, v in by_key.items():
                if v is extra:
                    by_key[k] = primary
        _coalesce_faces(primary, face)
        for k in keys:
            by_key.setdefault(k, primary)
    return faces


_NAME_SOURCE_RANK = {"collection": 0, "indexed": 1, "celebrity": 2}


def _collection_conflict(a: dict, b: dict) -> bool:
    """True when the CURATED COLLECTION asserts these are two different
    people: distinct face UUIDs under different collection-sourced
    names. Celebrity detection carries a higher false-positive rate
    than SearchFaces against the collection (user, 2026-09-07), so a
    celebrity-id agreement does not override that contradiction. Same
    name under two UUIDs is NOT a conflict — that is the auto-index
    duplicate-template shape the celebrity join exists to weld."""
    ra = (a.get("recognizer_ids") or {}).get("face_id")
    rb = (b.get("recognizer_ids") or {}).get("face_id")
    if not ra or not rb or ra == rb:
        return False
    na = a.get("name") if a.get("name_source") == "collection" else None
    nb = b.get("name") if b.get("name_source") == "collection" else None
    return bool(na and nb and na.casefold() != nb.casefold())


def _coalesce_faces(dst: dict, src: dict) -> None:
    """Merge one recognizer-asserted-duplicate face entry into another:
    union spans (single-coverage), recompute screen_sec from the merged
    spans, extend the coalesced_ids audit list, union the typed
    recognizer_ids, adopt a name the keeper lacked."""
    merged = sorted(set(dst["spans"]) | set(src["spans"]))
    flat: list = []
    for a, b in merged:
        if flat and a <= flat[-1][1]:
            flat[-1][1] = max(flat[-1][1], b)
        else:
            flat.append([a, b])
    dst["spans"] = [(a, b) for a, b in flat]
    dst["screen_sec"] = round(pb_common.spans_total_sec(dst["spans"]), 2)
    dst.setdefault("coalesced_ids", []).extend(
        [src["face_person_id"]] + list(src.get("coalesced_ids") or []))
    for kind, val in (src.get("recognizer_ids") or {}).items():
        dst["recognizer_ids"].setdefault(kind, val)
    if src.get("gender") and not dst.get("gender"):
        dst["gender"] = src["gender"]
    # name adoption ranks sources by false-positive rate (user,
    # 2026-09-07): collection > indexed > celebrity — on merge a
    # curated name replaces a celebrity-detected one, never the reverse
    if src["name"] and (
            not dst["name"]
            or _NAME_SOURCE_RANK.get(src.get("name_source"), 3)
            < _NAME_SOURCE_RANK.get(dst.get("name_source"), 3)):
        dst["name"] = src["name"]
        dst["name_source"] = src["name_source"]


def parse_media_depiction_spans(docs: list[dict], shots_by_id: dict):
    """shot_metadata rows -> [(t0, t1)] for shots whose media_depictions
    ATTESTS at least one displayed medium containing a person (photo,
    screen, poster, mirror, projection). Absent key = the prompt never
    asked (three-state) — contributes nothing; [] = attested none."""
    spans = []
    for d in docs:
        span = shots_by_id.get(d.get("shot_id"))
        md = d.get("media_depictions")
        if span and isinstance(md, list) and md:
            spans.append(span)
    return spans


def mark_depicted_faces(faces, depicted_spans):
    """The depicted-face guard's marking pass (review-hardened
    2026-09-06). Media-depiction is a SHOT-level attestation — the
    enrichment says displayed media containing a person is in the
    frame, not WHICH face is the depicted one — so for every face:

    - `live_spans` = spans minus depicted shots: what co-occurrence may
      credit. Face-speech pairing inside a depicted shot is
      untrustworthy for EVERY face in it (the monitor-guest trap: with
      the depicted face merely EXCLUDED whole-video, the co-present
      live face inherited undivided credit and a false CONFIRMED
      binding — verified failure, review 2026-09-06).
    - `depicted_share` = depicted overlap / the SAME spans' total (one
      basis for numerator and denominator) — read by persons typing
      against DEPICTED_SHARE_THRESHOLD for the depicted_only LABEL,
      where a majority-of-total statement is what the label means.

    No spans or no faces: nothing marked, nothing changed (absent
    media_depictions = not assessed, three-state)."""
    if not faces or not depicted_spans:
        return faces
    for f in faces:
        inside = pb_common.spans_overlap_sec(f["spans"], depicted_spans)
        total = pb_common.spans_total_sec(f["spans"])
        f["depicted_share"] = round(inside / total, 3) if total else 0.0
        f["live_spans"] = pb_common.subtract_spans(f["spans"],
                                                   depicted_spans)
    return faces


def parse_text_signals(docs: list[dict], shots_by_id: dict):
    """shot_metadata rows -> [(shot_id, t0, t1)] where on-screen text is
    EVIDENT: `unreadable_text_present` (text exists, never read — the
    credits-read diagnosis) or any overlay_text entity. These are the P1
    read TRIGGERS."""
    out = []
    for d in docs:
        span = shots_by_id.get(d.get("shot_id"))
        if not span:
            continue
        has_text = bool(d.get("unreadable_text_present"))
        if not has_text:
            has_text = any(isinstance(e, dict)
                           and e.get("evidence") == "overlay_text"
                           for e in d.get("named_entities") or [])
        if has_text:
            out.append((d["shot_id"], span[0], span[1]))
    return out


def parse_speaker_f0(doc: dict):
    """speech-diarization global_speakers.json -> {speaker_id:
    {f0_median_hz, f0_n_sentences}} for clusters carrying F0 stats.
    Additive diarization fields (2026-09-23) — an older run simply
    yields {} and the F0 veto reports unassessed."""
    out = {}
    for sid, m in (doc.get("speaker_models") or {}).items():
        if isinstance(m, dict) and m.get("f0_median_hz"):
            out[sid] = {"f0_median_hz": m["f0_median_hz"],
                        "f0_n_sentences": m.get("f0_n_sentences") or 0}
    return out


def parse_shots(doc: dict):
    """shot-segmentation shots.json -> {shot_id: (start, end)}."""
    return {s["shot_id"]: (float(s["start_time"]), float(s["end_time"]))
            for s in doc.get("shots") or []}


def parse_overlay_names(docs: list[dict], shots_by_id: dict):
    """shot-metadata-enrichment shot_metadata.jsonl rows -> chyron-name
    cues: person-typed named_entities whose evidence is overlay_text,
    stamped with their shot's timespan. Free evidence — no OCR was
    scheduled. Rows without named_entities (older prompt versions) simply
    contribute nothing."""
    cues = []
    for d in docs:
        span = shots_by_id.get(d.get("shot_id"))
        if not span:
            continue
        for e in d.get("named_entities") or []:
            if not isinstance(e, dict):
                continue
            if e.get("type") == "person" and \
                    e.get("evidence") == "overlay_text" and e.get("name"):
                cues.append({"kind": "overlay_name", "name": e["name"],
                             "t0": span[0], "t1": span[1],
                             "shot_id": d.get("shot_id")})
    return cues


def parse_screen_texts(docs: list[dict], shots_by_id: dict):
    """shot_metadata rows -> [(shot_id, t0, t1, [verbatim strings])]
    for shots whose `texts_on_screen` carries SME's transcription of
    readable on-screen text. Raw material for the corroboration join
    (screen_text_cues) — never consumed as names by itself."""
    out = []
    for d in docs:
        span = shots_by_id.get(d.get("shot_id"))
        texts = [t for t in (d.get("texts_on_screen") or [])
                 if isinstance(t, str) and t.strip()]
        if span and texts:
            out.append((d["shot_id"], span[0], span[1], texts))
    return out


def screen_text_cues(screen_texts, proposal_names, existing_overlay=None):
    """CORROBORATION-ONLY join (user design, 2026-10-05): a name some
    other mechanism already proposed (persons.trusted_name_proposals —
    nothing looser) is looked up in SME's verbatim `texts_on_screen`; a
    token match (pb_common.name_match — whole tokens, contiguous,
    symmetric) emits an overlay cue at the shot's timespan. The text
    never creates a claim — no proposal, no match, no cue — which is
    what keeps nameplate junk (prices, timers, tickers) inert and the
    PRODUCT_V doctrine intact: SME's model read the pixels; we only
    align its transcription with standing proposals. Shots already
    covered by an overlay cue of the same name (SME typed it, or a
    witness read it) are skipped so one sighting never votes twice."""
    if not screen_texts or not proposal_names:
        return []
    covered = [(str(oc["name"]), oc["t0"], oc["t1"])
               for oc in existing_overlay or []]
    cues = []
    for shot_id, t0, t1, texts in screen_texts:
        for name in sorted(proposal_names):
            if not any(pb_common.name_match(name, t) for t in texts):
                continue
            if any(pb_common.name_match(name, cn)
                   and pb_common.overlap_sec(t0, t1, a, b) > 0
                   for cn, a, b in covered):
                continue
            cues.append({"kind": "overlay_name", "name": name,
                         "t0": t0, "t1": t1, "shot_id": shot_id,
                         "source": "texts_on_screen"})
    return cues


# ── S3 loaders (prefix layout lives here, nothing else) ──────────────────────

def load_transcript(prefix: str | None):
    if not prefix:
        return None
    return parse_transcript(_get_json(
        f"{prefix.rstrip('/')}/transcripts/final_transcript.json"))


def load_faces(prefix: str | None):
    if not prefix:
        return None
    return parse_faces(_get_json(f"{prefix.rstrip('/')}/facemetadata.json"))


def load_speaker_f0(prefix: str | None):
    """Best-effort: F0 stats are an enhancement — a diarization run
    predating them (or a fetch failure) degrades to None, and the
    class-1 F0 check reports unassessed."""
    if not prefix:
        return None
    try:
        return parse_speaker_f0(_get_json(
            f"{prefix.rstrip('/')}/embeddings/global_speakers.json"))
    except Exception:                            # noqa: BLE001 — degrade
        return None


def load_shots(prefix: str | None):
    if not prefix:
        return None
    return parse_shots(_get_json(
        f"{prefix.rstrip('/')}/embeddings/shots.json"))


def load_shot_metadata_docs(prefix: str | None):
    """ONE fetch of shot_metadata.jsonl — the state hands the same docs
    list to every parser (overlay names, media-depiction spans, text
    signals) instead of paying a GET+parse per signal (review,
    2026-09-06: two loaders had grown around one object)."""
    if not prefix:
        return None
    return _get_jsonl(f"{prefix.rstrip('/')}/shot_metadata.jsonl")


def load_overlay_names(prefix: str | None, shots_by_id: dict | None):
    if not prefix or shots_by_id is None:
        return None
    return parse_overlay_names(load_shot_metadata_docs(prefix), shots_by_id)
