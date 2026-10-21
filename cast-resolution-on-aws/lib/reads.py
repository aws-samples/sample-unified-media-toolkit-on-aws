"""
reads.py — P1 focus reads: schedule frame-inspection requests, consume
their products back into name evidence.

Frame-inspection is the witness (frame-inspection-on-aws/README.md §5-§10): `credits-read` and
`title-read` are its EXISTING prompt categories, both returning the card
shape {t, lines[], ...} with the read's OWN typed extraction (names /
pairing kinds — PRODUCT_V). Cast-resolution is the caller — it owns
triggers, need-checks, the ledger, and the cross-frame statistics
(voting, dedup); identity extraction happens at the read, where the
pixels are (settled 2026-09-09, with SME's supers). No new witness
category is needed: chyron reads are title-read WITH EXPLICIT SPANS
from our triggers.

The uniform activation rule (README.md §8): every read class needs (a) a
TRIGGER — free evidence the target exists — and (b) an unmet NEED — the
names aren't already in hand. Never blind-read.

Cost posture (user, 2026-09-04): the witness already stacks
IMAGES_PER_RECORD=18 frames per record (the dominant saving — the prompt
is repeated per record, so stacking amortizes it) and defaults
use_batch=TRUE with pooled assets[] to clear the Bedrock batch minimum
across assets. The scheduler's job is to hand it ONE pooled request per
run and keep the span count inside the budget; chyron-class spans also
request the witness's lower-third crop (SHIPPED 2026-09-05 — fi runs a
container with pyvips for exactly this), ~60% fewer image tokens on
those spans.
"""
from __future__ import annotations

from lib.pb_common import (
    CHYRON_CLASS_REASONS,
    CHYRON_WINDOW_SEC,
    CLASS1_MIN_SHARE,
    CREDITS_PROBE_SEC,
    MAX_READ_SPANS,
    MIN_DEBUT_SCREEN_SEC,
    READS_PER_HEAD,
    TAIL_FRACTION,
    VOCATIVE_MIN_VOTES,
    name_match,
    overlap_sec,
)


def _merge_windows(wins, gap=0.5):
    out = []
    for a, b in sorted(wins):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(round(a, 2), round(b, 2)) for a, b in out]


def need_windows_from_sources(faces, transcript, pad_sec=2.0,
                              bound_speakers=None):
    """Presence windows of UNNAMED MAJOR heads — the last-resort gate's
    input (user direction, 2026-09-30: the witness reads only what free
    evidence could not name). Heads are class-1-share citizens
    (>= CLASS1_MIN_SHARE of total screen/speech — the matcher's own
    constant, one source of truth). A face with a source name
    (registry/celebrity) is already resolved and creates no need;
    `bound_speakers` are voices the PLAN-TIME class-1 forecast matched
    to a NAMED face (2026-09-30, the conditional-witness rule: a major
    voice the free bind will resolve creates no need, so an asset
    whose majors are all resolvable skips the witness entirely via
    the existing empty-plan self-complete). The forecast must run
    WITH the F0 veto: plan-time evidence is a subset of bind-time, so
    a missing veto could predict a bind the real bind strikes —
    under-reading, the unsafe direction; everything else diverges only
    toward a LARGER need set. Remaining voices are conservatively
    unnamed — chyron-while-speaking is exactly how a voice earns a
    name. Returns None when neither source exists (need not
    assessable — the planner then keeps legacy trigger behavior)."""
    if faces is None and transcript is None:
        return None
    bound = bound_speakers or set()
    heads_out = []
    if faces:
        total = sum(f["screen_sec"] for f in faces)
        for f in faces:
            if f.get("name") or total <= 0:
                continue
            if f["screen_sec"] >= CLASS1_MIN_SHARE * total:
                heads_out.append(
                    {"head": f"face:{f['face_person_id']}",
                     "spans": _merge_windows(
                         [(a - pad_sec, b + pad_sec)
                          for a, b in f["spans"]])})
    if transcript:
        speakers = transcript.get("speakers") or {}
        total = sum(speakers.values())
        heads = {s for s, sec in speakers.items()
                 if total > 0 and sec >= CLASS1_MIN_SHARE * total} - bound
        by_spk: dict = {}
        for t in transcript.get("turns") or []:
            if t["speaker_id"] in heads:
                by_spk.setdefault(t["speaker_id"], []).append(
                    (t["start"] - pad_sec, t["end"] + pad_sec))
        for sid, wins in sorted(by_spk.items()):
            heads_out.append({"head": f"voice:{sid}",
                              "spans": _merge_windows(wins)})
    return heads_out

def screen_text_confirmed_speakers(transcript, cues, screen_texts):
    """Voices the bind will name WITHOUT the witness (plan-time twin
    of the texts_on_screen corroboration join, user design 2026-10-05):
    a speaker whose cue-proposed name — the same cue set
    persons._name_from_cues trusts (self_id / sign_off /
    vocative_resolved, high confidence, exactly one name) — is
    corroborated by SME's on-screen text on >= VOCATIVE_MIN_VOTES
    distinct shots overlapping that speaker's turns. Conflicted
    speakers are NEVER subtracted (the bind may abstain, and
    under-reading is the unsafe direction); a miss merely leaves the
    head to the witness, today's behavior."""
    if not transcript or not cues or not screen_texts:
        return set()
    from lib import persons
    proposed = persons.trusted_name_proposals(cues)
    turns: dict[str, list] = {}
    for t in transcript.get("turns") or []:
        turns.setdefault(t["speaker_id"], []).append(
            (t["start"], t["end"]))
    confirmed = set()
    for sid, nm in proposed.items():
        hits = {shot_id for shot_id, t0, t1, texts in screen_texts
                if any(overlap_sec(t0, t1, a, b) > 0
                       for a, b in turns.get(sid) or [])
                and any(name_match(nm, s) for s in texts)}
        if len(hits) >= VOCATIVE_MIN_VOTES:
            confirmed.add(sid)
    return confirmed


import re

# Witness product schema version (2026-09-09): v2 products carry the
# read's OWN typed extraction — title cards a `names` list ({name, type,
# descriptor}), credits pairings a `kind` — replacing the consumer-side
# English lexicons and name-shape/typography heuristics this module used
# to run over the verbatim lines (the anti-pattern ruled out with SME's
# supers: extraction belongs to the model that reads the pixels, in any
# language). A ledger entry from an older product does not satisfy a v2
# need: the span is re-read ONCE and the ledger converges.
PRODUCT_V = 2


# ── scheduling ────────────────────────────────────────────────────────────────

def plan_reads(cues, faces, text_shots, duration_sec,
               satisfied_spans=None, ledger=None,
               credits_hint=False, has_shot_metadata=True,
               need_windows=None, credit_roll_spans=None,
               last_speech_end=None) -> dict:
    """Build the frame-inspection request for one binding run.

    cues        transcript cues (introduction cues trigger chyron reads)
    faces       sources.load_faces() output (debuts trigger chyron reads)
    text_shots  [(shot_id, t0, t1)] with unreadable/overlay text evidence
                (from enrichment) — tail ones gate credits, others are
                read spans themselves
    satisfied_spans  [(t0, t1)] where a name is already evidenced — the
                need-check; reads inside them are skipped
    ledger      previously read spans [(t0, t1, category)] — never re-pay
    credit_roll_spans  [(t0, t1)] of SME shot_kind == "credit_roll"
                shots — the credits EVIDENCE anchor (user design,
                2026-09-30): read exactly the typed roll, nothing
                else. Measured on the 13-asset corpus: every scripted
                asset with a real roll carries 1-9 credit_roll shots;
                all four news assets carry ZERO — so news skips
                credits structurally, with no genre branch, and
                appended textless/black tails are never read because
                only the roll spans are.
    last_speech_end  end of the last dialogue turn (sec) — the
                audio-dip proxy anchoring a HINTED credits read when
                SME typed no roll (rolls start at/after the last
                line; textless elements follow with no dialogue)
    need_windows  the LAST-RESORT gate (user direction + spend audit,
                2026-09-30): [{"head": id, "spans": [(t0, t1), ...]}]
                — presence windows PER unnamed major head
                (class-1-share face without a source name, or
                class-1-share voice). A witness read exists to name
                someone the free evidence could not — a span
                overlapping no unnamed head cannot produce a bindable
                name, whatever text it shows (measured: 58 title reads
                on four dramas, ~$3.20, zero witness-named persons) —
                and each head carries a READS_PER_HEAD budget (see
                pb_common for the measured rationale).
                None = need not assessed (legacy behavior, read on
                triggers alone); [] = assessed and empty (only
                introduction cues read — the cue itself is need
                evidence: someone unnamed is being introduced).

    -> {request: inspect_frames payload | None, spans: [...],
        credits: bool, skipped: {satisfied: n, ledgered: n, budget: n,
        no_need: n}}
    """
    satisfied = satisfied_spans or []
    # ledger entries: (t0, t1, category[, crop[, product_v]]) —
    # a FULL-FRAME entry (crop None) satisfies any need; a CROPPED entry
    # satisfies only a cropped need. Without this, a lower-third read
    # would permanently satisfy a full-frame text-shot need over the
    # same window and the never-re-pay ledger would become never-read
    # (review, 2026-09-05). An entry from an OLDER product schema never
    # satisfies (see PRODUCT_V): its product lacks the typed extraction,
    # so the evidence was NOT obtained — one re-read converges the
    # ledger.
    ledger = [(tuple(e) + (None, None))[:5] for e in (ledger or [])]
    skipped = {"satisfied": 0, "ledgered": 0, "budget": 0, "no_need": 0,
               "head_quota": 0}

    def blocked(t0, t1, category, need_crop=None):
        if any(overlap_sec(t0, t1, a, b) > 0 for a, b in satisfied):
            skipped["satisfied"] += 1
            return True
        if any(c == category and overlap_sec(t0, t1, a, b) > 0
               and (cr is None or cr == need_crop)
               and pv == PRODUCT_V
               for a, b, c, cr, pv in ledger):
            skipped["ledgered"] += 1
            return True
        return False

    tail_start = duration_sec * (1.0 - TAIL_FRACTION)

    # credits activation (user design, 2026-09-30): EVIDENCE or HINT,
    # never a blind tail. SME-typed credit_roll shots are the evidence
    # and the exact read spans; a caller hint covers content whose roll
    # SME didn't type. Text-in-the-tail alone no longer activates —
    # that gate is what bought 24-30 min of channel logos and
    # "WILL RETURN IN:" countdowns on every news asset (~$10-13 of a
    # $20.69 corpus bill; 1,387 frames of one logo on one asset).
    credits = duration_sec > 0 and (
        bool(credit_roll_spans) or bool(credits_hint))

    # chyron/title spans, priority order — the budget keeps the head
    candidates = []
    for c in cues:
        if c["kind"] == "introduction":
            candidates.append(("introduction", c["t"],
                               c["t"] + CHYRON_WINDOW_SEC))
    for f in faces or []:
        if f.get("name") or f["screen_sec"] < MIN_DEBUT_SCREEN_SEC:
            continue                     # named already, or one-off b-roll
        if f["spans"]:
            t0 = f["spans"][0][0]
            candidates.append(("face_debut", t0, t0 + CHYRON_WINDOW_SEC))
    for shot_id, t0, t1 in text_shots:
        if t0 >= tail_start:
            continue                     # tail text is the credits gate
        candidates.append((f"text_shot:{shot_id}", t0, t1))

    spans = []
    quota = ({h["head"]: READS_PER_HEAD for h in need_windows}
             if need_windows is not None else None)
    for reason, t0, t1 in candidates:
        # last-resort gate: a read must serve an unnamed major head,
        # and each head carries a READ BUDGET (READS_PER_HEAD) — the
        # class-priority candidate order means the budget buys the
        # head's debut span first; without it one long-present head
        # legitimizes every overlapping span up to MAX_READ_SPANS.
        # Introduction cues bypass — the cue itself is need evidence.
        serving = None
        if need_windows is not None and reason != "introduction":
            serving = [h["head"] for h in need_windows
                       if any(overlap_sec(t0, t1, a, b) > 0
                              for a, b in h["spans"])]
            if not serving:
                skipped["no_need"] += 1
                continue
            if all(quota.get(h, 0) <= 0 for h in serving):
                skipped["head_quota"] += 1
                continue
        # CHYRON_CLASS_REASONS read lower thirds (the witness crops
        # those frames, ~60% fewer image tokens); text-shot spans keep
        # the full frame — their text can sit anywhere on screen.
        crop = ("lower_third" if reason in CHYRON_CLASS_REASONS
                else None)
        if blocked(t0, t1, "title-read", crop):
            continue
        merged = False
        for span in spans:
            if overlap_sec(t0, t1, span["start"], span["end"]) <= 0:
                continue
            # merge INTO the earlier span — but a merge must never
            # NARROW what gets read (review, 2026-09-05: a full-frame
            # text_shot swallowed by a cropped chyron span lost its
            # top-of-frame text): the union keeps the WIDER frame need
            # and the union of the time bounds.
            span["start"] = round(min(span["start"], t0), 2)
            span["end"] = round(max(span["end"], t1), 2)
            if crop is None:
                span.pop("crop", None)
            merged = True
            break
        if merged:
            continue
        if len(spans) >= MAX_READ_SPANS:
            skipped["budget"] += 1
            continue
        span = {"start": round(t0, 2), "end": round(t1, 2),
                "id": f"pb-{len(spans):03d}", "reason": reason}
        if crop:
            span["crop"] = crop
        spans.append(span)
        if serving:
            # one read moment serves every head it overlaps; merged
            # candidates widen an existing moment and charge nothing
            for h in serving:
                if h in quota:
                    quota[h] -= 1

    # ledger_candidates: what THIS plan will have read if the run
    # succeeds — BindPersons appends them to the cross-run ledger
    # only after the products are actually fetched (evidence obtained,
    # not frames submitted).
    requests, candidates = [], []
    # Credits spans are always EXPLICIT so fi's broad auto-trigger
    # (which counts logo bumpers as credit-ish) never decides what we
    # pay for. Priority: (1) the SME-typed roll spans, padded and
    # merged — reads exactly the roll, skips interleaved textless
    # black; (2) hinted with no typed roll — CREDITS_PROBE_SEC
    # anchored at the last dialogue's end (the audio-dip proxy: rolls
    # start at/after the last line, textless tails carry no dialogue),
    # else the final window of the tail.
    if credit_roll_spans:
        c_spans = _merge_windows([(max(0.0, a - 5.0),
                                   min(duration_sec, b + 5.0))
                                  for a, b in credit_roll_spans])
    elif credits_hint:
        if last_speech_end:
            a = max(0.0, min(last_speech_end - 30.0,
                             duration_sec - CREDITS_PROBE_SEC))
            c_spans = [(a, min(duration_sec, a + CREDITS_PROBE_SEC))]
        else:
            c_spans = [(max(tail_start,
                            duration_sec - CREDITS_PROBE_SEC),
                        duration_sec)]
    else:
        c_spans = []
    c_spans = [(a, b) for a, b in c_spans
               if not blocked(a, b, "credits-read")]
    if credits and c_spans:
        requests.append({"category": "credits-read",
                         "spans": [{"start": round(a, 2),
                                    "end": round(b, 2),
                                    "id": f"pb-credits-{i:02d}"}
                                   for i, (a, b) in enumerate(c_spans)]})
        candidates += [{"t0": round(a, 2), "t1": round(b, 2),
                        "category": "credits-read",
                        "product_v": PRODUCT_V}
                       for a, b in c_spans]
    if spans:
        requests.append({"category": "title-read",
                         "spans": [{k: s[k] for k in
                                    ("start", "end", "id", "crop")
                                    if k in s}
                                   for s in spans]})
        candidates += [
            {"t0": s["start"], "t1": s["end"], "category": "title-read",
             "product_v": PRODUCT_V,
             **({"crop": s["crop"]} if s.get("crop") else {})}
            for s in spans]
    return {"requests": requests or None, "spans": spans,
            "credits": credits, "skipped": skipped,
            "ledger_candidates": candidates}


# ── consumption ─────────────────────────────────────────────────────────────────────

# a typed name must repeat across this many frames of a block to count —
# one frame's read is noise ('MAR INA' vs 'MARINA' at 360p), a repeated
# reading is a real overlay. Same move as vocative voting.
_MIN_READING_FRAMES = 2


def cues_from_titles(titles_doc: dict) -> list[dict]:
    """title-read blocks -> overlay-style name cues, from the witness's
    OWN typed extraction (product v2, 2026-09-09): each card carries
    `names` [{name, type, descriptor}], typed by the model that read the
    frame — any language, no consumer lexicons. This replaced the
    English word lists, possessive rules and name-shape regexes that
    used to interpret the verbatim lines here (Latin-script-only by its
    own admission); chrome never enters because a show brand or bug is
    typed program/organization at the read, and only `person` entries
    become cues. Frame-repetition voting stays — it is cross-frame
    statistics the per-frame read cannot do. Cards WITHOUT `names`
    (products read before v2) contribute nothing: the ledger versioning
    re-reads their spans once instead of keeping heuristics alive.
    """
    blocks = list(titles_doc.get("blocks") or [])
    if titles_doc.get("cards"):
        blocks.append({"cards": titles_doc["cards"],
                       "start": None, "end": None})

    cues = []
    for block in blocks:
        tally: dict = {}                 # casefold -> vote state
        for card in block.get("cards") or []:
            for n in card.get("names") or []:
                if not (isinstance(n, dict) and n.get("name")
                        and n.get("type") == "person"):
                    continue
                k = str(n["name"]).casefold()
                row = tally.setdefault(
                    k, {"frames": 0, "spellings": {}, "descriptors": {},
                        "first_t": float(card.get("t", 0.0))})
                row["frames"] += 1
                s = str(n["name"])
                row["spellings"][s] = row["spellings"].get(s, 0) + 1
                d = n.get("descriptor")
                if d:
                    row["descriptors"][d] = row["descriptors"].get(d, 0) + 1
        for row in tally.values():
            if row["frames"] < _MIN_READING_FRAMES:
                continue
            name = max(row["spellings"], key=row["spellings"].get)
            role_text = (max(row["descriptors"],
                             key=row["descriptors"].get)
                         if row["descriptors"] else None)
            cues.append({"kind": "overlay_name", "name": name,
                         "t0": row["first_t"] - 1.0,
                         "t1": row["first_t"] + CHYRON_WINDOW_SEC,
                         "source": "title-read",
                         "role_text": role_text})
    return cues


def cast_from_credits(credits_doc: dict) -> dict:
    """credits-read blocks -> {"pairs": [{actor, character}],
    "names": [...]}. Both come from the witness's typed pairings
    (product v2, 2026-09-09): `kind: character` states the mapping — the
    card's own claim, read by the vision model in any language — and
    every other kind (job / billing / listing / other) is presence
    evidence only, no mapping guessed. This replaced the consumer-side
    crew/connector lexicons and the ALLCAPS-vs-TitleCase typography
    discrimination (calibrated on one 5-asset trial, English credit
    conventions only). Entries WITHOUT `kind` (products read before v2)
    degrade to presence evidence; the ledger versioning re-reads their
    spans once. Text-side line parses are gone with the heuristics —
    the v2 prompt emits one entry per person name on the card, so the
    lines are no longer a name source."""
    cards = [c for b in credits_doc.get("blocks") or []
             for c in b.get("cards") or []]

    pairs, names, seen = [], [], set()
    for card in cards:
        for p in card.get("pairings") or []:
            if not (isinstance(p, dict) and p.get("name")):
                continue
            name = str(p["name"]).strip()
            pw = (str(p["paired_with"]).strip()
                  if p.get("paired_with") else None)
            if p.get("kind") == "character" and pw:
                key = (name.casefold(), pw.casefold())
                if key not in seen:
                    seen.add(key)
                    pairs.append({"actor": name, "character": pw})
            else:
                names.append(name)
    return {"pairs": pairs, "names": sorted(set(names))}


def apply_cast(doc: dict, cast: dict) -> dict:
    """Join credits pairs onto persons: an actor-named person (celebrity/
    collection face) whose name casefold-matches a credits pair gains
    character_name — the actor->character step AD needs. Never renames;
    `name` stays the actor, `character_name` is additive."""
    by_actor = {p["actor"].casefold(): p["character"]
                for p in cast.get("pairs") or []}
    for p in doc["persons"]:
        if not p.get("name"):
            continue
        character = by_actor.get(p["name"].casefold())
        if character:
            p["character_name"] = character
            p.setdefault("evidence", []).append(
                {"mechanism": "credits_cast", "actor": p["name"],
                 "character": character})
    return doc


# news role vocabulary -> canonical role. English seeds (the multilingual
# path is the witness/LLM, as everywhere).
_ROLE_MAP = (("anchor", "anchor"), ("correspondent", "correspondent"),
             ("reporter", "reporter"), ("meteorologist", "weather"),
             ("weather", "weather"), ("analyst", "analyst"),
             ("host", "host"), ("guest", "guest"),
             ("contributor", "contributor"))


def _role_from(role_text) -> str | None:
    """Whole-word match only: substring matching read 'anchor' out of
    'Anchorage Resident' and 'host' out of 'Ghostwriter' (review F6)."""
    rt = (role_text or "").lower()
    for word, role in _ROLE_MAP:
        if re.search(rf"\b{word}\b", rt):
            return role
    return None


def assign_roles(doc: dict, duration_sec: float | None = None,
                 speech_spans: dict | None = None) -> dict:
    """News roles, migrated from the POC fuse ladder (user, 2026-09-23 —
    roles live where the evidence lives, and the viz builder's copy of
    this ladder is deleted). Rungs, first match wins:

    1. role_text — a chyron stated the role (the v1 rule, unchanged).
    2. anchor by speech threading, ONLY with a bound on-camera face
       (measured failure: over-merged voice-only clusters threading a
       2 h broadcast read as anchors): the person's SPEECH TURN spans
       spread over at least half the programme AND touch at least half
       of its eighths — fractions of the programme, not fitted scalars,
       so the rule scales from a clip to a master. SPEECH spans, never
       face appearances (measured 2026-09-23: teases replay a guest's
       FACE across the whole broadcast, so appearance spread crowned
       three guests — a guest's VOICE stays inside their segment). A
       keynote block threads nothing: one contiguous block touches few
       eighths. `speech_spans` ({speaker_id: [(start, end)]}) comes
       from the caller's transcript; absent, this rung abstains.
       When the doc carries a class-1 assessment, the rung additionally
       requires a CLASS-1 voice: an anchor is by definition a
       programme-carrying voice, while an over-merged tail cluster can
       scatter a guest's speech across every eighth without ever
       carrying real share (measured 2026-09-23: two commercial-pod
       guests crowned anchor on 55-77% spreads riding merged clusters).
       Docs without the assessment keep the spread rule alone.
    3. guest_or_subject — a registry-named on-camera speaker that is
       neither of the above.

    Rungs 2–3 run ONLY when the programme carries broadcast chrome —
    some person named via an overlay (chyron) or role_text. The roles
    vocabulary describes broadcast function, and without the gate a
    MOVIE's lead — on screen and speaking across the whole runtime —
    reads as an "anchor". Chyrons are an evidence-presence signal, not
    a genre branch: news and chyroned reality carry them, scripted
    drama and film do not.

    collection-membership⇒talent is deliberately NOT ported: it assumed
    a curated staff-list collection, and on a broad collection it made
    file-footage subjects "talent" (measured 2026-09-23). It returns
    only behind an explicit collection-is-curated declaration.
    Silent faces and voice-only persons carry no role — a role is a
    claim about on-camera function, and abstention beats a guess."""
    registry = {"collection", "celebrity",
                "collection_recovery", "celebrity_recovery"}
    c1_voices = set((doc.get("class1") or {}).get("speakers") or [])
    # broadcast chrome means LOWER-THIRD people-chyrons (SME overlay
    # evidence). Witness TITLE reads are excluded: a drama's opening
    # production credits carry names + role_text ("PRODUCER Jane Doe")
    # and were opening the gate on scripted content — every lead then
    # threaded to "anchor" (measured live 2026-09-24 on two dramas).
    # No name_mechanisms fallback: the overlay attach writes the cue
    # into evidence alongside the name by construction, and the
    # mechanism alone cannot say whether its source was a chyron or a
    # title card.
    chrome = any(
        e.get("kind") == "overlay_name" or e.get("role_text")
        for p in doc["persons"] for e in p.get("evidence") or []
        if isinstance(e, dict) and e.get("source") != "title-read")
    for p in doc["persons"]:
        if p.get("role"):
            continue
        for e in p.get("evidence") or []:
            if not isinstance(e, dict):
                continue
            role = _role_from(e.get("role_text"))
            if role:
                p["role"] = role
                p.setdefault("evidence", []).append(
                    {"mechanism": "role_text", "role": role,
                     "text": e.get("role_text")})
                break
        if not chrome or p.get("role") \
                or p.get("type") != "on_camera_speaker" \
                or not p.get("name"):
            continue
        spans = [s for sid in p.get("speaker_ids") or []
                 for s in (speech_spans or {}).get(sid) or []]
        if c1_voices and not c1_voices & set(p.get("speaker_ids") or []):
            spans = []                    # not a programme-carrying voice
        if duration_sec and spans:
            spread = max(b for _, b in spans) - min(a for a, _ in spans)
            eighth = duration_sec / 8.0
            touched = {int(a // eighth) for a, b in spans} \
                | {int(min(b, duration_sec - 0.001) // eighth)
                   for a, b in spans}
            if spread >= 0.5 * duration_sec and len(touched) >= 4:
                p["role"] = "anchor"
                p.setdefault("evidence", []).append(
                    {"mechanism": "speech_threading", "role": "anchor",
                     "spread_sec": round(spread, 1),
                     "eighths_touched": sorted(touched)})
                continue
        if set(p.get("name_mechanisms") or []) & registry:
            p["role"] = "guest_or_subject"
    return doc
