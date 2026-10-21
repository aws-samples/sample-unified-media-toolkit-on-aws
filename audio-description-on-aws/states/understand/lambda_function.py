"""
StateUnderstand (loop): the describer's preparation, two bedrock
submissions in sequence — the NAME MAP (character_map.json) then the
WATCH-THROUGH (watch_notes.json, the describer's first full viewing).
Ported from the runner's build_name_map/build_watch_notes; the system
prompts are the ad-name-map / ad-watch-notes registry categories.
"""
from __future__ import annotations

import json
import logging

from lib import ad_common as C
from lib import evidence
from lib import oracles
from lib import loop_state
from lib import parse_partial_output
from lib import prompt_render
from lib import submit

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# STRUCTURAL stoplist only — function words, generic honorifics, days.
# The runner's list also carried "Maestro/Maestros" and "New York"/"Los
# Angeles": exactly the honorific and the two settings of the 3-series
# validation corpus (n=3), removed 2026-08-31 per the
# generalize-never-content-specific rule — a corpus-fitted stoplist
# suppresses corpus cities while surfacing every other city as a name
# candidate. The cost is a few extra evidence lines in the name-map
# prompt; the model, not the list, decides what is a name.
_COMMON = {"I", "The", "A", "It", "He", "She", "They", "We", "You", "And",
           "But", "So", "Well", "Yeah", "No", "Yes", "What", "When",
           "Where", "Who", "How", "Why", "That", "This", "There", "Then",
           "Now", "Okay", "OK", "Oh", "Hey", "If", "Not", "Just", "God",
           "Jesus", "Christ", "Mr", "Mrs", "Ms", "Miss", "Sir",
           "Doctor", "Captain", "Detective", "Officer", "Mom",
           "Dad", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
           "Saturday", "Sunday"}


def lambda_handler(event: dict, context) -> dict:
    plan = C.read_json(*C.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    ck = loop_state.checkpoint_key(prefix, "understand")
    state = C.read_json_if_exists(C.PROXY_BUCKET, ck) or {"phase": "start"}
    token = event["task_token"]
    ev = evidence.fetch(plan["inputs"])

    if state["phase"] == "start":
        rec = _name_map_record(plan, ev)
        state = {"phase": "namemap",
                 "submit": submit.bedrock_submit([rec], token, plan,
                                                 "namemap",
                                                 use_batch=True)}
        C.write_json(C.PROXY_BUCKET, ck, state)
        return {"waiting": "namemap"}

    if state["phase"] == "namemap":
        out = submit.bedrock_outputs(state["submit"], ["namemap"])[0]
        cmap = _parse_name_map(out)
        C.write_json(C.PROXY_BUCKET, f"{prefix}/character_map.json", cmap)
        rec = _watch_record(plan, ev, cmap)
        state = {"phase": "watch",
                 "submit": submit.bedrock_submit([rec], token, plan,
                                                 "watch",
                                                 use_batch=True)}
        C.write_json(C.PROXY_BUCKET, ck, state)
        return {"waiting": "watch"}

    if state["phase"] == "watch":
        out = submit.bedrock_outputs(state["submit"], ["watch"])[0]
        notes = _parse_watch_notes(out)
        C.write_json(C.PROXY_BUCKET, f"{prefix}/watch_notes.json", notes)
        C.write_json(C.PROXY_BUCKET, ck, {"phase": "done"})
        loop_state.self_complete(token)
        return {"waiting": None}

    # phase == done: a redrive re-entered a finished loop — reconcile
    loop_state.self_complete(token)
    return {"waiting": None}


# ── name map ─────────────────────────────────────────────────────────────────

def _cast_block(bindings, people, basis: str = "face") -> str:
    """Render cast-resolution's adjudicated bindings (filtered ONCE in
    validate_plan, riding plan.json) as the record section ABOVE the
    raw hints — ONE compact renderer, the ad_details pattern, so a
    second consumer can never disagree with this one. Empty string when
    the run has no bindings (three-state: absent input = old record,
    byte-identical)."""
    if not bindings:
        return ""
    by_pid = {p["person_id"]: p for p in people or []}
    names = {p["name"] for p in people or []}
    lines = []
    for b in bindings:
        if basis != "face":
            # people ARE the cast persons: face_ids reference the
            # (absent) face-recognition run and must not be joined —
            # the binding's own name IS its identity key, and it is a
            # VALID key exactly when that person made it into the
            # FACE KEYS section (has a presence timeline): membership
            # in `names` is the one rule for face- and speech-basis
            # cast runs alike, so the two sections can never contradict
            faces = [b["name"]] if b["name"] in names else []
        else:
            # deduped, bounded face keys: coalesced duplicates carry
            # every original facemetadata id ON PURPOSE (remap
            # coverage), which here would repeat one name per cluster
            faces = list(dict.fromkeys(
                by_pid[f]["name"] for f in b.get("face_ids") or []
                if f in by_pid))[:4]
        parts = [b.get("type") or "?"]
        if b.get("role"):
            parts.append(f"role={b['role']}")
        if b.get("character_name"):
            # cast-resolution's deterministic credits join (its own
            # persons x kind:"character" pairings) — established
            # actor->character, so the model confirms instead of
            # re-deriving it from raw on-screen text
            parts.append(f"plays {b['character_name']}")
        tail = []
        if b.get("aliases"):
            tail.append("aliases: " + ", ".join(b["aliases"][:3]))
        if b.get("speaker_ids"):
            tail.append("speaks as "
                        + ", ".join(str(s) for s in b["speaker_ids"][:4]))
        if faces:
            tail.append("face keys: " + ", ".join(faces))
        tail.append("via " + ",".join(b.get("name_mechanisms") or []))
        lines.append(f"  {b['name']} ({', '.join(parts)}); "
                     + "; ".join(tail))
    return (
        "ADJUDICATED PERSON BINDINGS (cast-resolution: face + speech "
        "+ on-screen-text evidence, corroboration-gated — treat these "
        "identities as ESTABLISHED and prefer them over the raw hints "
        "below; contradict them only on explicit on-screen proof. A "
        "binding with no face keys is VOICE-ONLY: use its speaker "
        "binding, and give it a character entry only if a matching "
        "face key exists in FACE KEYS below):\n"
        + "\n".join(lines) + "\n\n")


def _name_map_record(plan: dict, ev: dict) -> dict:
    t, ca = ev["transcript"], ev["ca"]
    docs_t = evidence.docs_by_id(ev["shot_docs"])
    shots_t = sorted(ev["shots"], key=lambda x: x["start_time"])

    def shot_at(tm):
        sh = next((x for x in shots_t
                   if x["start_time"] <= tm < x["end_time"]), None)
        return (docs_t.get(sh["shot_id"], {}).get("description", "")
                if sh else "")

    name_lines = []
    for s in t["sentences"]:
        words = s["text"].split()
        hit = any(i > 0 and (cw := w.strip(".,!?\";:'"))
                  and cw[0].isupper() and cw not in _COMMON
                  and not words[i - 1].endswith((".", "!", "?"))
                  for i, w in enumerate(words))
        if hit:
            tm = float(s["start"])
            name_lines.append(f"  {tm:7.1f}s {s.get('speaker_id')}: "
                              f"{s['text'][:130]}")
            ctx = shot_at(tm)
            if ctx:
                name_lines.append(f"           on screen then: {ctx[:110]}")

    speech = ev.get("people_basis") == "cast_speech"
    named = [p for p in ev["people"] if p.get("named")]
    unnamed = [p for p in ev["people"]
               if not p.get("named")
               and p.get("total_screen_time_sec", 0) >= 15]
    face_lines = []
    for p in named:
        aps = p["appearances"]
        total = sum(a["end_ms"] - a["begin_ms"] for a in aps) / 1000
        first3 = ", ".join(f"{a['begin_ms']/1000:.0f}-"
                           f"{a['end_ms']/1000:.0f}s" for a in aps[:3])
        face_lines.append(f"  {p['name']}: {total:.0f}s "
                          f"{'speaking' if speech else 'on screen'} across "
                          f"{len(aps)} spans; first: {first3}")
    for p in unnamed:
        aps = p["appearances"]
        first3 = ", ".join(f"{a['begin_ms']/1000:.0f}-"
                           f"{a['end_ms']/1000:.0f}s" for a in aps[:3])
        face_lines.append(
            f"  {p['name']} (UNNAMED cluster, "
            f"{p.get('gender') or 'gender unknown'}, age "
            f"~{p.get('age_range') or '?'}): "
            f"{p.get('total_screen_time_sec', 0):.0f}s across "
            f"{len(aps)} spans; first: {first3}")

    # the ONE text reader (oracles.shot_texts): on post-2026-09-09 docs
    # texts_on_screen is the unclaimed remainder, and the chyron names
    # the name map most needs live in the overlay entities
    texts = []
    for d in ev["shot_docs"]:
        for x in oracles.shot_texts(d):
            texts.append(f"  shot {d['shot_id']}: {x[:90]}")

    body = (f"Program type: {ca.get('overall_content_type')}; genres "
            f"{ca.get('top_3_genres')}; synopsis: "
            f"{str(ca.get('overall_summary'))[:400]}\n\n"
            + _cast_block(plan.get("cast_bindings"), ev["people"],
                          ev.get("people_basis", "face")) +
            ("FACE KEYS (FACELESS run — every key is a named VOICE and "
             "times are SPEECH spans; describe visuals from the shot "
             "descriptions, never invent on-screen presence):\n"
             if speech else
             "FACE KEYS (actor names, with screen time):\n")
            + "\n".join(face_lines)
            + "\n\nDIALOGUE LINES CONTAINING LIKELY NAMES "
              "(time, speaker, text):\n" + "\n".join(name_lines[:80])
            + "\n\nON-SCREEN TEXT (titles/credits/supers):\n"
            + "\n".join(texts[:60]) + "\n\nReturn the JSON now.")
    r = prompt_render.render(plan["prompts"]["ad-name-map"],
                             {"{{body}}": body})
    return {"record_id": "namemap", "system": r["system"],
            "messages": r["messages"]}


def _parse_name_map(out: str) -> dict:
    """First complete JSON object; trailing commentary ignored. SALVAGE
    before failure (review 2026-08-31: this was the one model-output
    parser with none — a max_tokens truncation or a brace-free refusal
    crashed StateUnderstand after the bedrock spend). A response with no
    recoverable characters still RAISES, loudly and named: a name map is
    load-bearing for every later stage."""
    start = out.find("{")
    if start < 0:
        raise RuntimeError(
            "name map unusable: no JSON object in the model response "
            f"(starts: {out[:120]!r})")
    try:
        cmap, _ = json.JSONDecoder().raw_decode(out[start:])
    except json.JSONDecodeError:
        cmap = parse_partial_output.parse_partial_json(out[start:])
    if not isinstance(cmap, dict) or not cmap.get("characters"):
        raise RuntimeError(
            "name map unusable: no 'characters' recovered from the "
            f"model response (starts: {out[:120]!r})")
    # every downstream consumer hard-indexes c["face_key"] — an entry
    # without one (the model binding a voice-only person anyway) is the
    # hallucinated-id KeyError class; prune it here, before it can
    # crash a later state after the spend (review, 2026-09-07)
    keyed = [c for c in cmap["characters"] if c.get("face_key")]
    if len(keyed) != len(cmap["characters"]):
        logger.warning("name map: pruned %d character(s) without a "
                       "face_key", len(cmap["characters"]) - len(keyed))
    if not keyed:
        raise RuntimeError(
            "name map unusable: no character carries a face_key")
    cmap["characters"] = keyed
    # EARLY NAMING (policy, user decision 2026-08-28):
    # first_established_sec stays the MEASURED fact; speakable_from_sec
    # is the POLICY derived from it, so the evidence survives a policy
    # change. Bounded by the one exception that keeps it safe — a
    # CONCEALED identity is never named early (the name IS the reveal).
    for c in cmap["characters"]:
        est = float(c.get("first_established_sec") or 0)
        c["speakable_from_sec"] = (
            0.0 if C.EARLY_NAMING and c.get("character_name")
            and not c.get("concealed") else est)
        c["naming_policy"] = (
            "early" if c["speakable_from_sec"] < est else
            "concealed" if c.get("concealed") else "as-established")
    return cmap


# ── watch-through ────────────────────────────────────────────────────────────

def _watch_record(plan: dict, ev: dict, cmap: dict) -> dict:
    t, ca = ev["transcript"], ev["ca"]
    docs = evidence.docs_by_id(ev["shot_docs"])
    shots = sorted(ev["shots"], key=lambda s: s["start_time"])

    mm = sorted(s.get("frame_motion_max") or 0 for s in shots)
    p85 = mm[int(len(mm) * 0.85)] if mm else 1.0

    events = []
    for sc in ev["scenes"]:
        events.append((sc["start_time"], 0, f"== SCENE {sc['scene_id']} =="))
    for s in shots:
        d = docs.get(s["shot_id"], {})
        spike = (" [MOTION SPIKE — the description may be missing an "
                 "event]" if (s.get("frame_motion_max") or 0) > p85 else "")
        det = evidence.ad_details(d)
        events.append((s["start_time"], 1,
                       f"shot {s['shot_id']}: "
                       f"{d.get('description', '')[:150]}"
                       f"{(' [' + '; '.join(det) + ']') if det else ''}"
                       f"{spike}"))
    for sent in t["sentences"]:
        events.append((float(sent["start"]), 2,
                       f"{sent.get('speaker_id')}: "
                       f"\"{sent['text'].strip()[:120]}\""))
    events.sort()
    record_lines = [f"{tm:7.1f}s  {txt}" for tm, _, txt in events]
    body = (f"Program: {ca.get('overall_content_type')}; genres "
            f"{ca.get('top_3_genres')}\n"
            "Character map (already derived): "
            + "; ".join(f"{c['face_key']}="
                        f"{c.get('character_name') or c.get('descriptor')}"
                        for c in cmap["characters"]) + "\n\n"
            "THE PROGRAM, IN ORDER:\n" + "\n".join(record_lines)
            + "\n\nWrite the describer's notes JSONL now.")
    r = prompt_render.render(plan["prompts"]["ad-watch-notes"],
                             {"{{body}}": body})
    return {"record_id": "watch", "system": r["system"],
            "messages": r["messages"]}


def _parse_watch_notes(out: str) -> dict:
    notes = {"synopsis": "", "tone": "", "reveals": [], "beats": [],
             "continuity": []}
    bad = 0
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(("```", "//")):
            continue
        obj, _tier = parse_partial_output.parse_jsonl_line(line, ("type",))
        if obj is None:
            bad += 1
            continue
        # NORMALIZE at the producer, once: a model line with "what":
        # null or a non-numeric sec used to crash the ledger/repair
        # consumers that slice b["what"] (review 2026-08-31); every
        # downstream reader may now trust str what + float sec.
        def _s(v):
            return v if isinstance(v, str) else ""

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        k = obj.get("type")
        if k == "synopsis":
            notes["synopsis"] = _s(obj.get("text"))
        elif k == "tone":
            notes["tone"] = _s(obj.get("text"))
        elif k == "reveal":
            notes["reveals"].append(
                {"sec": _f(obj.get("sec")), "what": _s(obj.get("what")),
                 "known_to_audience_after_sec":
                     _f(obj.get("known_after"))})
        elif k == "beat":
            notes["beats"].append(
                {"sec": _f(obj.get("sec")), "what": _s(obj.get("what")),
                 "inferred": bool(obj.get("inferred"))})
        elif k == "continuity":
            notes["continuity"].append(_s(obj.get("text")))
    if bad:
        logger.info("watch: %d unsalvageable line(s) dropped", bad)
    return notes
