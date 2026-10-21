"""
StateWriteScript (loop): the WRITER (one pooled bedrock run, records
chunked WINDOWS_PER_CHUNK windows each — the §10 chunking rule applied
to the model leg; rendering itself is text assembly, seconds, so no Map)
then the CRITIC, whose empty-a-duty-window edits are refused by a
MECHANICAL veto (instruction alone was measured insufficient).
"""
from __future__ import annotations

import logging

from lib import ad_common as C
from lib import evidence
from lib import oracles
from lib import loop_state
from lib import prompt_render
from lib import submit
from lib import windows as windows_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = C.read_json(*C.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    ck = loop_state.checkpoint_key(prefix, "script")
    state = C.read_json_if_exists(C.PROXY_BUCKET, ck) or {"phase": "start"}
    token = event["task_token"]

    if state["phase"] == "start":
        records = _writer_records(plan)
        state = {"phase": "writer",
                 "record_ids": [r["record_id"] for r in records],
                 "submit": submit.bedrock_submit(records, token, plan,
                                                 "writer",
                                                 use_batch=True)}
        C.write_json(C.PROXY_BUCKET, ck, state)
        return {"waiting": "writer"}

    if state["phase"] == "writer":
        outs = submit.bedrock_outputs(state["submit"], state["record_ids"])
        ws = C.read_json(C.PROXY_BUCKET, f"{prefix}/windows.json")
        cues, allbad = {}, []
        for o in outs:
            c, bad = submit.parse_jsonl_cues(o)
            cues.update(c)
            allbad.extend(bad)
        # hallucinated window ids are pruned HERE, before anything can
        # index them (review 2026-08-31: one extra id crashed the run
        # downstream — after the tts spend)
        cues, extras = submit.prune_cues(
            cues, {w["window_id"] for w in ws})
        if extras:
            logger.info("writer: pruned hallucinated window ids %s",
                        extras)
        missing = sorted({w["window_id"] for w in ws} - set(cues))
        logger.info("writer: %d/%d answered, %d non-empty, %d bad, "
                    "missing=%s", len(cues), len(ws),
                    sum(1 for v in cues.values() if v.strip()),
                    len(allbad), missing)
        C.write_json(C.PROXY_BUCKET, f"{prefix}/cues.json",
                     {str(k): v for k, v in cues.items()})
        rec = _critic_record(plan, ws, cues)
        state = {"phase": "critic",
                 "submit": submit.bedrock_submit([rec], token, plan,
                                                 "critic",
                                                 use_batch=True)}
        C.write_json(C.PROXY_BUCKET, ck, state)
        return {"waiting": "critic"}

    if state["phase"] == "critic":
        out = submit.bedrock_outputs(state["submit"], ["critic"])[0]
        fixes, _bad = submit.parse_jsonl_cues(out)
        ws = C.read_json(C.PROXY_BUCKET, f"{prefix}/windows.json")
        wmap = {w["window_id"]: w for w in ws}
        cues = {int(k): v for k, v in C.read_json(
            C.PROXY_BUCKET, f"{prefix}/cues.json").items()}
        for k, v in fixes.items():
            if k not in cues:
                continue
            # MECHANICAL VETO, not trust: the critic's job is to delete
            # filler, and a duty cue reads exactly like filler from the
            # text alone (measured: it emptied an anticipatory entry).
            if not v.strip() and windows_mod.has_duty(wmap[k]):
                logger.info("critic w%d: REFUSED empty on a duty window",
                            k)
                continue
            cues[k] = v
        C.write_json(C.PROXY_BUCKET, f"{prefix}/cues.json",
                     {str(k): v for k, v in cues.items()})
        C.write_json(C.PROXY_BUCKET, ck, {"phase": "done"})
        loop_state.self_complete(token)
        return {"waiting": None}

    loop_state.self_complete(token)          # redrive reconcile
    return {"waiting": None}


# ── writer records (ported from the runner's build_script) ───────────────────

def _shot_line(docs, sid, tag=""):
    d = docs.get(sid)
    if not d:
        return None
    extras = [x for x in (d.get("setting"), d.get("time_of_day")) if x]
    if d.get("sentiment") and d["sentiment"] != "neutral":
        extras.append(f"mood: {d['sentiment']}")
    if d.get("lighting"):
        extras.append("lighting: " + "/".join(d["lighting"]))
    sts = oracles.shot_texts(d)     # both artifact shapes (2026-09-09)
    if sts:
        extras.append("TEXT ON SCREEN: " + " / ".join(sts))
    extras += evidence.ad_details(d)
    return (f"  {tag}shot {sid}: {d.get('description', '')}"
            + (f" [{'; '.join(extras)}]" if extras else ""))


def _name_map_block(cmap: dict) -> str:
    lines = ["NAME MAP (face-key -> speakable name). USE THE NAME from "
             "'usable from' onward; use the descriptor before that, or "
             "whenever the name is NO NAME/CONCEALED:"]
    for c in cmap["characters"]:
        frm = float(c.get("speakable_from_sec",
                          c.get("first_established_sec") or 0))
        lines.append(
            f"- {c['face_key']} -> "
            f"{c.get('character_name') or 'NO NAME'} | descriptor: "
            f"{c.get('descriptor')} | usable from: {frm:.0f}s"
            + (f" (program establishes it at "
               f"{c.get('first_established_sec')}s; EARLY NAMING is in "
               f"force, so name this character from their FIRST "
               f"appearance — a descriptor where a name is available "
               f"reads as a different person)"
               if frm < float(c.get("first_established_sec") or 0) else "")
            + (" | CONCEALED — never name, always the descriptor"
               if c.get("concealed") else ""))
    return "\n".join(lines)


def _writer_records(plan: dict) -> list:
    prefix = plan["output_prefix"]
    ev = evidence.fetch(plan["inputs"])
    ws = C.read_json(C.PROXY_BUCKET, f"{prefix}/windows.json")
    notes = C.read_json(C.PROXY_BUCKET, f"{prefix}/watch_notes.json")
    cmap = C.read_json(C.PROXY_BUCKET, f"{prefix}/character_map.json")
    docs = evidence.docs_by_id(ev["shot_docs"])
    sents = ev["transcript"]["sentences"]
    ca = ev["ca"]
    nm = _name_map_block(cmap)

    def dialogue_around(a, b):
        before = [s for s in sents if float(s["end"]) <= a + 0.05]
        after = [s for s in sents if float(s["start"]) >= b - 0.05]
        out = []
        if before:
            s = before[-1]
            out.append(f'  dialogue before: {s.get("speaker_id")}: '
                       f'"{s["text"].strip()[:110]}"')
        if after:
            s = after[0]
            out.append(f'  dialogue after: {s.get("speaker_id")}: '
                       f'"{s["text"].strip()[:110]}"')
        return out

    records = []
    chunk_size = C.WINDOWS_PER_CHUNK
    for ci in range(0, len(ws), chunk_size):
        chunk = ws[ci:ci + chunk_size]
        c0, c1 = chunk[0]["start"] - 20, chunk[-1]["end"] + 20
        lines = [f"Program: {ca.get('overall_content_type')} — genres "
                 f"{ca.get('top_3_genres')}",
                 f"Synopsis (from your full watch-through): "
                 f"{notes.get('synopsis', str(ca.get('overall_summary')))[:600]}",
                 f"Narration tone: {notes.get('tone', 'measured, neutral')}",
                 "", nm, ""]
        cont = notes.get("continuity") or []
        if cont:
            lines += ["CONTINUITY SHEET (use these exact phrasings for "
                      "recurring things):"] \
                     + [f"- {c}" for c in cont[:10]] + [""]
        beats = [b for b in (notes.get("beats") or [])
                 if c0 <= float(b["sec"]) <= c1]
        if beats:
            lines += ["PLOT-CRITICAL BEATS in this span — each MUST be "
                      "conveyed by the nearest suitable window; never "
                      "sacrificed to place or brevity:"] \
                     + [f"- @{float(b['sec']):.0f}s: {b['what']}"
                        + (" (reconstructed across shots — still voice "
                           "the event itself)" if b.get("inferred")
                           else "") for b in beats] + [""]
        revs = [r for r in (notes.get("reveals") or [])
                if float(r.get("known_to_audience_after_sec", 0)) > c0]
        if revs:
            lines += ["REVEAL GUARD — the audience does not yet know "
                      "these; description must not spoil them before "
                      "their time:"] \
                     + [f"- {r['what']} (audience learns it ~"
                        f"{float(r.get('known_to_audience_after_sec', 0)):.0f}s)"
                        for r in revs[:6]] + [""]
        lines.append(f"NARRATION WINDOWS ({chunk[0]['window_id']}-"
                     f"{chunk[-1]['window_id']} of 0-{len(ws) - 1}):")
        for w in chunk:
            flags = windows_mod.duty_flags(w)
            lines.append(
                f"\nwindow {w['window_id']} | {w['start']:.1f}s-"
                f"{w['end']:.1f}s | BUDGET {w['word_budget']} WORDS"
                + (" | " + " | ".join(flags) if flags else "")
                + f" | on screen (face-keys): "
                  f"{', '.join(w['people_on_screen'][:6]) or 'none named'}")
            if w["scene_entry"]:
                setting = next((docs[s].get("setting")
                                for s in w["shot_ids"]
                                if s in docs and docs[s].get("setting")),
                               None)
                if setting:
                    lines.append(f"  REQUIRED OPENER: begin with the "
                                 f"place — setting evidence: {setting}")
            for sid in w.get("recent_shot_ids", [])[:2]:
                sl = _shot_line(docs, sid,
                                "JUST BEFORE (under dialogue) — ")
                if sl:
                    lines.append(sl)
            for sid in w["shot_ids"][:4]:
                sl = _shot_line(docs, sid)
                if sl:
                    lines.append(sl)
            for sid in w.get("cutaways", [])[:2]:
                sl = _shot_line(docs, sid, "cutaway just after — ")
                if sl:
                    lines.append(sl)
            lines.extend(dialogue_around(w["start"], w["end"]))
        lines.append(f"\nWrite the JSONL now: one line per window, "
                     f"{chunk[0]['window_id']} through "
                     f"{chunk[-1]['window_id']}.")
        r = prompt_render.render(plan["prompts"]["ad-writer"],
                                 {"{{body}}": "\n".join(lines)})
        records.append({"record_id": f"ad-{ci // chunk_size:02d}",
                        "system": r["system"], "messages": r["messages"]})
    return records


def _critic_record(plan: dict, ws: list, cues: dict) -> dict:
    prefix = plan["output_prefix"]
    ev = evidence.fetch(plan["inputs"])
    notes = C.read_json(C.PROXY_BUCKET, f"{prefix}/watch_notes.json")
    docs = evidence.docs_by_id(ev["shot_docs"])
    wmap = {w["window_id"]: w for w in ws}
    lines = []
    if notes.get("beats"):
        lines += ["PLOT-CRITICAL BEATS:"] \
                 + [f"- @{float(b['sec']):.0f}s: {b['what']}"
                    for b in notes["beats"]] + [""]
    lines.append("CUES:")
    for k in sorted(k for k, v in cues.items() if v.strip()):
        w = wmap[k]
        setting = ""
        if w["scene_entry"]:
            setting = next((docs[s].get("setting") for s in w["shot_ids"]
                            if s in docs and docs[s].get("setting")),
                           "") or ""
        evd = " || ".join(str(docs[s].get("description", ""))[:90]
                          for s in w["shot_ids"][:3] if s in docs)
        duties = windows_mod.duty_flags(w)
        lines.append(
            f"w{k} | {w['start']:.0f}-{w['end']:.0f}s | budget "
            f"{w['word_budget']}"
            + (f" | setting: {setting}" if setting else "")
            + (" | DUTIES: " + " ;; ".join(duties) if duties else "")
            + f" | cue: {cues[k]!r} | shots: {evd}")
    r = prompt_render.render(plan["prompts"]["ad-critic"],
                             {"{{body}}": "\n".join(lines)})
    return {"record_id": "critic", "system": r["system"],
            "messages": r["messages"]}
