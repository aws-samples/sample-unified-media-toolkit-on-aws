"""
StateSynthFit (loop): synthesize every non-empty cue, FIT each take into
its window (tempo-capped, pure arithmetic over tts's own duration_s),
and rewrite the over-budget ones SHORTER through a bounded, counted
reask loop (REASK_ROUNDS). Unplaceable cues land in the drop register
with their measurement — never a chipmunk, never an overlap.

TTS submissions are SHARDED at TTS_SHARD_SIZE (tts-on-aws refuses >100
transcripts per batch; the runner's <=90 sharding was measured, and
dropping it failed a typical ~120-cue episode at synthesis — review
2026-08-31) and the shards FAN OUT IN PARALLEL (2026-09-02): this state
resolves its own token with {tts_shard_tags}, the fan-out Map submits
one shard per iteration against its own token, and the re-entry fits
EVERY shard from the submit register — shard wall-clock is max, not
sum. takes.json records each raw take's uri + duration for the repair
chain's revive/refit; placed.json + drops.json are the write-through
registers.
"""
from __future__ import annotations

import logging

from lib import ad_common as C
from lib import evidence
from lib import fitting
from lib import loop_state
from lib import prompt_render
from lib import submit
from lib import windows as windows_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = C.read_json(*C.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    ck = loop_state.checkpoint_key(prefix, "synth")
    state = C.read_json_if_exists(C.PROXY_BUCKET, ck) or {"phase": "start"}
    token = event["task_token"]

    if state["phase"] == "start":
        cues = _cues(prefix)
        order = sorted(k for k, v in cues.items() if v.strip())
        state = {"phase": "tts", "round": 0, "placed": [], "drops": [],
                 "reask": [],
                 "shard_specs": submit.shard_specs("main", order,
                                                   seed=1234)}
        C.write_json(C.PROXY_BUCKET, ck, state)
        return loop_state.request_fanout(token, list(state["shard_specs"]))

    if state["phase"] in ("tts", "reask_tts"):
        loop_state.assert_fanout_checkpoint(state, "synth")
        got = submit.collect_fanout(prefix, state["shard_specs"])
        refan = loop_state.reconcile_fanout(got, state, ck)
        if refan is not None:
            return loop_state.request_fanout(token, refan)
        _fit_all(plan, state, got["outs"])
        return _next_round_or_finish(plan, state, token)

    if state["phase"] == "reask_ask":
        out = submit.bedrock_outputs(state["submit"],
                                     [state["record_id"]])[0]
        newc, _bad = submit.parse_jsonl_cues(out)
        cues = _cues(prefix)
        rids = [r["window_id"] for r in state["reask"]
                if newc.get(r["window_id"], "").strip()]
        # a model-emptied rewrite is a DROP and must reach the register
        # with its measurement (review 2026-08-31: it used to vanish,
        # and the ledger reported 'cause is not in the drop register')
        for r in state["reask"]:
            if r["window_id"] not in rids:
                state["drops"].append(
                    {**{k: v for k, v in r.items() if k != "avail"},
                     "kind": "over_budget",
                     "avail_sec": r.get("avail", r.get("avail_sec")),
                     "note": "reask returned empty — the model had no "
                             "shorter phrasing worth keeping"})
        for k in rids:
            cues[k] = newc[k].strip()
        _write_cues(prefix, cues)
        if not rids:
            state["reask"] = []
            return _next_round_or_finish(plan, state, token)
        state.update({"phase": "reask_tts", "reask": [],
                      "shard_specs": submit.shard_specs(
                          f"reask{state['round']}", rids, seed=4321)})
        C.write_json(C.PROXY_BUCKET, ck, state)
        return loop_state.request_fanout(token, list(state["shard_specs"]))

    loop_state.self_complete(token)          # redrive reconcile
    return {"waiting": None}


def _cues(prefix):
    return {int(k): v for k, v in C.read_json(
        C.PROXY_BUCKET, f"{prefix}/cues.json").items()}


def _write_cues(prefix, cues):
    C.write_json(C.PROXY_BUCKET, f"{prefix}/cues.json",
                 {str(k): v for k, v in cues.items()})


def _spans(plan):
    """Word spans from the transcript ALONE — the fit leg must not pay
    the full sibling-evidence fetch for one list (review 2026-08-31)."""
    ev = evidence.fetch(plan["inputs"], only={"transcript"})
    return windows_mod.word_spans(ev["transcript"])


def _fit_all(plan, state, outs_by_tag):
    """Register every shard's takes and fit each into its window —
    arithmetic over tts's duration_s; no audio is opened. Runs AFTER
    collect_fanout proved every shard SUCCEEDED and complete. Extends
    the in-checkpoint placed/reask lists; a crash here re-fits from S3
    on redrive — deterministic, nothing double-counts."""
    prefix = plan["output_prefix"]
    ws = {w["window_id"]: w for w in C.read_json(
        C.PROXY_BUCKET, f"{prefix}/windows.json")}
    spans = _spans(plan)
    placed = {p["window_id"]: p for p in state["placed"]}
    reask = list(state.get("reask") or [])
    for tag in sorted(outs_by_tag):
        spec = state["shard_specs"][tag]
        outs = outs_by_tag[tag]
        submit.register_takes(prefix, {
            k: {"wav": f"s3://{o['bucket']}/{o['key']}",
                "duration_s": o["duration_s"]}
            for k, o in zip(spec["order"], outs)})
        for k, o in zip(spec["order"], outs):
            p, r = fitting.fit_one(k, o["duration_s"], ws[k], spans)
            if p:
                placed[k] = p
            else:
                reask.append(r)
    state["placed"] = list(placed.values())
    state["reask"] = reask
    logger.info("fit %d shard(s): placed %d, over-budget %d",
                len(outs_by_tag), len(state["placed"]), len(reask))


def _next_round_or_finish(plan, state, token):
    prefix = plan["output_prefix"]
    ck = loop_state.checkpoint_key(prefix, "synth")
    if state["reask"] and state["round"] < C.REASK_ROUNDS:
        state["round"] += 1
        rec_id = f"reask{state['round']}"
        rec = _reask_record(plan, state["reask"], rec_id)
        state.update({"phase": "reask_ask", "record_id": rec_id,
                      "submit": submit.bedrock_submit([rec], token, plan,
                                                      rec_id)})
        C.write_json(C.PROXY_BUCKET, ck, state)
        return {"waiting": rec_id}

    # rounds exhausted (or nothing to reask): register the unplaceable,
    # collide, persist
    drops = list(state["drops"])
    for r in state["reask"]:
        drops.append({"window_id": r["window_id"], "kind": "over_budget",
                      "needed_tempo": r["needed_tempo"],
                      "avail_sec": r["avail"], "raw_dur": r["raw_dur"],
                      "max_words": r["max_words"]})
    placed, collide_drops = fitting.collide(state["placed"],
                                            _spans(plan))
    drops += collide_drops
    C.write_json(C.PROXY_BUCKET, f"{prefix}/placed.json", placed)
    C.write_json(C.PROXY_BUCKET, f"{prefix}/drops.json", drops)
    C.write_json(C.PROXY_BUCKET, ck, {"phase": "done"})
    logger.info("synth done: %d placed, %d drops", len(placed),
                len(drops))
    loop_state.self_complete(token)
    return {"waiting": None}


def _reask_record(plan, reask, record_id):
    prefix = plan["output_prefix"]
    ev = evidence.fetch(plan["inputs"], only={"shot_docs"})
    docs = evidence.docs_by_id(ev["shot_docs"])
    ws = {w["window_id"]: w for w in C.read_json(
        C.PROXY_BUCKET, f"{prefix}/windows.json")}
    cues = _cues(prefix)
    notes = C.read_json(C.PROXY_BUCKET, f"{prefix}/watch_notes.json")
    lines = ["Your audio-description texts below are too long for their "
             "silences once synthesized. Rewrite each SHORTER — the word "
             "cap is ABSOLUTE; prefer fragments. THE LADDER STILL "
             "APPLIES: a plot-critical EVENT is never dropped to save "
             "words — drop adjectives and place instead; a CALM SCENE "
             "ENTRY keeps its place grounding; names per the map; no "
             "bare-pronoun openers.", "", "WINDOWS TO REWRITE:"]
    for r in reask:
        w = ws[r["window_id"]]
        flags = []
        if w["scene_entry"]:
            flags.append("SCENE ENTRY — keep the place")
        rb = [b for b in (notes.get("beats") or [])
              if w["start"] - 10 <= float(b["sec"]) <= w["end"] + 10]
        for b in rb[:2]:
            flags.append(f"BEAT: {str(b.get('what') or '')[:60]} — must "
                         "survive the cut")
        lines.append(
            f"\nwindow {r['window_id']} | {r['avail']:.1f}s of silence "
            f"| HARD CAP {r['max_words']} WORDS"
            + (" | " + " | ".join(flags) if flags else ""))
        lines.append(f"  your previous text: {cues[r['window_id']]!r}")
        for sid in w["shot_ids"][:2]:
            d = docs.get(sid)
            if d:
                lines.append(f"  shot {sid}: "
                             f"{d.get('description', '')[:130]}")
    lines.append('\nReturn JSONL: {"window_id": N, "text": "..."}')
    r = prompt_render.render(plan["prompts"]["ad-writer"],
                             {"{{body}}": "\n".join(lines)})
    return {"record_id": record_id, "system": r["system"],
            "messages": r["messages"]}
