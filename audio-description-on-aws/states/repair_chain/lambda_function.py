"""
StateRepairChain (loop): ONE re-entrant state walking the FIXED repair
sequence (run_all's order, each step born from a dated incident). Each
step: mechanical trigger (lib/oracles) → optional ONE bedrock ask + ONE
tts synthesis → apply under the unified failure policy (drop_on_fail
governs "unfit" AND "longer-with-collateral"). Checkpoint per step;
sibling callbacks re-enter; deletions only through the coverage guard.

The PASSES stay strictly sequential — each accepted repair changes the
placements the next pass is collided against (a data dependency, not a
throttling concern). WITHIN a pass, the re-tts shards of one answer are
independent and fan out through the repair Map (2026-09-02), the same
mechanism as SynthFit's.

Documented divergence from the runner: entry_succession's per-window
second reask (succ2) is folded into the single-ask policy — an unfit
succession rewrite keeps the original cue. The corpus gate re-measures
this trade.
"""
from __future__ import annotations

import logging

from lib import ad_common as C
from lib import coverage
from lib import evidence
from lib import fitting
from lib import ledger as ledger_mod
from lib import loop_state
from lib import oracles
from lib import prompt_render
from lib import submit
from lib import windows as windows_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# (step, needs_model, failure policy)
SEQUENCE = [
    ("revive_entries", False, {}),
    ("fix_overlaps", False, {}),
    ("entry_succession", True, {}),
    ("fix_overlaps", False, {}),
    ("beat_repair", True, {}),
    ("reveal_check", True, {}),
    ("role_descriptor_check", True, {}),
    ("onscreen_text_check", True, {"drop_on_fail": True,
                                   "no_longer": True}),
    ("name_leak_check", True, {"drop_on_fail": True}),
    ("tail_block_check", True, {}),
    ("redundancy_check", True, {"no_longer": True}),
    ("fix_overlaps", False, {}),
]


class _Run:
    """Everything a step touches, loaded once per invocation and
    persisted once per mutation batch."""

    def __init__(self, plan):
        self.plan = plan
        self.prefix = plan["output_prefix"]
        # only what the repair steps actually read — this loop re-enters
        # ~25x per run and must not re-download conversation_analysis /
        # facemetadata it never touches (review 2026-08-31)
        ev = evidence.fetch(plan["inputs"],
                            only={"shots", "shot_docs", "scenes",
                                  "transcript", "credits"})
        self.ev = ev
        self.docs = evidence.docs_by_id(ev["shot_docs"])
        self.tm = evidence.shot_times(ev["shots"])
        self.spans = windows_mod.word_spans(ev["transcript"])
        self.ws = {w["window_id"]: w for w in self._j("windows.json")}
        self.cues = {int(k): v for k, v in self._j("cues.json").items()}
        self.placed = self._j("placed.json")
        self.drops = self._j("drops.json") or []
        self.gaps = C.read_json_if_exists(
            C.PROXY_BUCKET, f"{self.prefix}/ad_gaps.json") or []
        self.notes = self._j("watch_notes.json")
        self.orientation = self._j("orientation.json")
        self.cmap = self._j("character_map.json")["characters"]

    def _j(self, name):
        return C.read_json(C.PROXY_BUCKET, f"{self.prefix}/{name}")

    def persist(self):
        placed, cd = fitting.collide(self.placed, self.spans)
        self.placed = placed
        self.drops += cd
        C.write_json(C.PROXY_BUCKET, f"{self.prefix}/placed.json",
                     self.placed)
        C.write_json(C.PROXY_BUCKET, f"{self.prefix}/cues.json",
                     {str(k): v for k, v in self.cues.items()})
        C.write_json(C.PROXY_BUCKET, f"{self.prefix}/drops.json",
                     self.drops)
        C.write_json(C.PROXY_BUCKET, f"{self.prefix}/ad_gaps.json",
                     self.gaps)

    def drop(self, rec):
        """Write-through: the register persists so the ledger re-derives
        from a finished prefix."""
        self.drops.append(rec)

    def safe_deletions(self, candidates):
        return coverage.safe_deletions(self.placed, self.cues, candidates,
                                       self.notes, self.orientation)

    def ask_record(self, tag, lines):
        r = prompt_render.render(self.plan["prompts"]["ad-repair"],
                                 {"{{body}}": "\n".join(lines)})
        return {"record_id": tag, "system": r["system"],
                "messages": r["messages"]}

    def avail_for(self, p):
        """The SAME availability rule the fitter applies — sizing a
        rewrite cap from a different formula hands the writer seconds
        the fitter then rejects."""
        w = self.ws[p["window_id"]]
        return fitting.available(p["at"], w, self.spans,
                                 w["seconds"] < 2)


def lambda_handler(event: dict, context) -> dict:
    plan = C.read_json(*C.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    ck = loop_state.checkpoint_key(prefix, "repair")
    state = C.read_json_if_exists(C.PROXY_BUCKET, ck) or \
        {"step": 0, "phase": "run"}
    token = event["task_token"]
    run = _Run(plan)

    while state["step"] < len(SEQUENCE):
        name, needs_model, policy = SEQUENCE[state["step"]]
        step = _STEPS[name]

        if state["phase"] == "run":
            ask = step.detect(run, state)
            if not ask:
                _advance(run, state, ck)
                continue
            tag = f"repair{state['step']:02d}-{name}"
            state.update({"phase": "ask", "targets": ask["targets"],
                          "extra": ask.get("extra") or {},
                          "submit": submit.bedrock_submit(
                              [run.ask_record(tag, ask["lines"])],
                              token, plan, tag),
                          "tag": tag})
            C.write_json(C.PROXY_BUCKET, ck, state)
            return {"waiting": tag}

        if state["phase"] == "ask":
            out = submit.bedrock_outputs(state["submit"],
                                         [state["tag"]])[0]
            newc, _bad = submit.parse_jsonl_cues(out)
            # hallucinated ids are pruned BEFORE any tts spend
            newc, extras = submit.prune_cues(newc, set(run.ws))
            if extras:
                logger.info("%s: pruned hallucinated window ids %s",
                            state["tag"], extras)
            newc = step.on_answer(run, state, newc, out)
            rids = [k for k in state["targets"]
                    if newc.get(k, "").strip()]
            if not rids:
                run.persist()
                _advance(run, state, ck)
                continue
            state.update({"phase": "tts",
                          "texts": {str(k): newc[k].strip()
                                    for k in rids},
                          "shard_specs": submit.shard_specs(state["tag"],
                                                            rids)})
            C.write_json(C.PROXY_BUCKET, ck, state)
            return loop_state.request_fanout(
                token, list(state["shard_specs"]))

        if state["phase"] == "tts":
            loop_state.assert_fanout_checkpoint(state, "repair")
            got = submit.collect_fanout(run.prefix, state["shard_specs"])
            refan = loop_state.reconcile_fanout(got, state, ck)
            if refan is not None:
                return loop_state.request_fanout(token, refan)
            for tag in sorted(got["outs"]):
                _apply(run, state, policy,
                       state["shard_specs"][tag]["order"],
                       got["outs"][tag])
            run.persist()
            _advance(run, state, ck)
            continue

    C.write_json(C.PROXY_BUCKET, ck,
                 {"step": len(SEQUENCE), "phase": "done"})
    loop_state.self_complete(token)
    return {"waiting": None}


def _advance(run, state, ck):
    nxt = state["step"] + 1
    state.clear()
    state.update({"step": nxt, "phase": "run"})
    C.write_json(C.PROXY_BUCKET, ck, state)


def _apply(run, state, policy, shard, outs):
    """Synthesise-result application for ONE shard (the pass's shards
    were synthesized in parallel by the fan-out Map; application is
    serial and order-free — each rewrite fits its own window): fit each
    accepted take (arithmetic over tts's duration_s), keep or revert
    (or DROP) under the unified failure policy. no_longer rejects a
    grown take only on measured COLLATERAL (fitting.collateral).

    takes.json is updated ONLY on acceptance (review 2026-08-31: the
    pre-decision register let a REJECTED rewrite's wav replace the
    shipped cue's audio pointer — the mix then played audio the
    srt/vtt/ledger never carried)."""
    drop_on_fail = policy.get("drop_on_fail", False)
    no_longer = policy.get("no_longer", False)
    was = {p["window_id"]: p["dur"] for p in run.placed}
    accepted_takes = {}
    for k, o in zip(shard, outs):
        p, r = fitting.fit_one(k, o["duration_s"], run.ws[k], run.spans)
        why = f"unfit (x{r['needed_tempo']})" if p is None else None
        if why is None and no_longer and k in was \
                and p["dur"] > was[k] + 0.05:
            harm = fitting.collateral(k, p, run.placed, run.spans)
            if harm:
                why = (f"{p['dur'] - was[k]:.2f}s LONGER than the take "
                       f"it replaces and {harm}")
        if why:
            if drop_on_fail:
                logger.info("w%d: repair %s — cue DROPPED", k, why)
                run.placed[:] = [x for x in run.placed
                                 if x["window_id"] != k]
                pend = (state.get("extra") or {}).get("drop_records", {})
                run.drop(pend.get(str(k))
                         or {"window_id": k, "kind": "repair_unfit",
                             "note": why})
            else:
                logger.info("w%d: repair %s — original kept", k, why)
            continue
        cur = next((x for x in run.placed if x["window_id"] == k), None)
        if cur is None:
            continue
        cur.update(p)
        run.cues[k] = state["texts"][str(k)]
        accepted_takes[k] = {"wav": f"s3://{o['bucket']}/{o['key']}",
                             "duration_s": o["duration_s"]}
    if accepted_takes:
        submit.register_takes(run.prefix, accepted_takes)


# ── the steps ────────────────────────────────────────────────────────────────

class _Step:
    def detect(self, run, state):
        """Return None (nothing to do) or {"lines", "targets", "extra"}.
        Code-only steps do their work here and return None."""
        raise NotImplementedError

    def on_answer(self, run, state, newc, raw):
        """Step-specific answer handling (verification, deletions).
        `raw` is the model's full output text — a step needing an extra
        field re-parses it in place instead of re-fetching the response
        from S3 (review 2026-08-31). Returns the texts to synthesize,
        keyed by window_id."""
        return newc


class _ReviveEntries(_Step):
    """Entry cues dropped as unfit get one more chance under the
    true-onset ceiling (their raw takes still exist) before succession
    reassigns duty."""

    def detect(self, run, state):
        placed_ids = {p["window_id"] for p in run.placed}
        takes = C.read_json_if_exists(
            C.PROXY_BUCKET, f"{run.prefix}/takes.json") or {}
        revived = 0
        for w in run.ws.values():
            wid = w["window_id"]
            take = takes.get(str(wid))
            if not (w["scene_entry"] and wid not in placed_ids
                    and run.cues.get(wid, "").strip() and take):
                continue
            p, _r = fitting.fit_one(wid, take["duration_s"], w,
                                    run.spans)
            if p:
                run.placed.append(p)
                revived += 1
        logger.info("revive_entries: %d", revived)
        run.persist()
        return None


class _FixOverlaps(_Step):
    """Residual speech overlaps get one repair: refit to the true
    next-word onset within the tempo cap, else the cue is dropped —
    never a chipmunk, never an overlap."""

    def detect(self, run, state):
        spans = run.spans

        def overlaps():
            return [(p, round(min(p["at"] + p["dur"], s1)
                              - max(p["at"], s0), 2))
                    for p in run.placed for s0, s1 in spans
                    if min(p["at"] + p["dur"], s1)
                    - max(p["at"], s0) > 0.02]

        takes = C.read_json_if_exists(
            C.PROXY_BUCKET, f"{run.prefix}/takes.json") or {}
        seen = set()
        for p, _ov in overlaps():
            if id(p) in seen or p not in run.placed:
                continue          # a cue spanning 2+ spans appears twice
            seen.add(id(p))
            wid = p["window_id"]
            if any(s0 - 0.02 <= p["at"] < s1 for s0, s1 in spans):
                run.placed.remove(p)
                run.drop({"window_id": wid, "kind": "speech_overlap",
                          "at": p["at"], "dur": p["dur"],
                          "overlap_sec": round(p["dur"], 2),
                          "starts_inside_speech": True})
                continue
            take = takes.get(str(wid))
            if not take:
                run.placed.remove(p)
                run.drop({"window_id": wid, "kind": "speech_overlap",
                          "at": p["at"], "dur": p["dur"],
                          "overlap_sec": 0.0,
                          "starts_inside_speech": False,
                          "note": "raw take not in the register"})
                continue
            fix = fitting.refit(wid, take["duration_s"], p["at"], spans)
            if fix:
                p["dur"] = fix["dur"]
                p["tempo"] = fix["tempo"]
            else:
                onset = fitting.next_onset(spans, p["at"])
                avail = ((onset - C.ONSET_PAD - p["at"])
                         if onset is not None else 1e9)
                run.placed.remove(p)
                rec = fitting.over_budget_record(
                    wid, take["duration_s"], avail)
                rec["kind"] = "over_budget"
                rec["avail_sec"] = rec.pop("avail")
                run.drop(rec)
        run.persist()
        return None


class _EntrySuccession(_Step):
    """Entry windows are marked at BUILD time, but cues drop at FIT or
    SCRIPT time — and the scene's grounding duty died with them
    (measured: 5/17 Meridian scenes silently lost their entry). Pass the
    duty to the scene's next PLACED cue; a scene with no placed
    successor lands in ad_gaps."""

    def detect(self, run, state):
        cues, placed = run.cues, run.placed
        placed_ids = {p["window_id"] for p in placed
                      if cues.get(p["window_id"], "").strip()}
        scenes = {s["scene_id"]: s for s in run.ev["scenes"]}
        unmet = []
        for w in run.ws.values():
            duty = (w["entry_for_scene"] if w["scene_entry"]
                    else w.get("anticipatory_for_scene"))
            if duty is not None and w["window_id"] not in placed_ids:
                sc = scenes.get(duty)
                if sc:
                    unmet.append(sc)
        heirs, ungroundable = {}, []
        by_time = sorted((p for p in placed
                          if p["window_id"] in placed_ids),
                         key=lambda p: p["at"])
        for sc in unmet:
            heir = next((p for p in by_time
                         if sc["start_time"] <= p["at"]
                         < sc["end_time"]), None)
            if heir is None:
                ungroundable.append(
                    {"sec": sc["start_time"],
                     "beat": f"scene {sc['scene_id']} never grounded",
                     "reason": "entry cue unfit and no placed successor "
                               "in scene"})
                continue
            heirs[heir["window_id"]] = sc
        logger.info("entry succession: %d lost, %d heirs, "
                    "%d ungroundable", len(unmet), len(heirs),
                    len(ungroundable))
        if ungroundable:
            run.gaps = ledger_mod.merge_gaps(run.gaps, "scene_ungrounded",
                                             ungroundable)
            run.persist()
        if not heirs:
            return None
        lines = ["Each cue below is the FIRST description the audience "
                 "hears after a scene change, but the scene change "
                 "itself went unannounced (the entry window could not "
                 "hold a cue). Rewrite each cue so it ALSO grounds the "
                 "new scene — fold the PLACE (and who, if room) into "
                 "it, keeping the cue's events. The word cap is "
                 "ABSOLUTE. Names per the map; no bare-pronoun openers.",
                 "", "CUES TO REWRITE:"]
        for wid, sc in heirs.items():
            w = run.ws[wid]
            setting = next((run.docs[s].get("setting")
                            for s in w["shot_ids"]
                            if s in run.docs
                            and run.docs[s].get("setting")), None)
            p = next(x for x in placed if x["window_id"] == wid)
            cap = max(3, int((w["end"] + C.GRACE - p["at"])
                              * C.WPS_REPAIR_CAP))
            lines.append(
                f"\nwindow {wid} | scene changed at "
                f"{sc['start_time']:.0f}s"
                + (f" | setting evidence: {setting}" if setting else "")
                + f" | HARD CAP {cap} WORDS")
            lines.append(f"  current cue: {cues[wid]!r}")
            for sid in w["shot_ids"][:2]:
                d = run.docs.get(sid)
                if d:
                    lines.append(f"  shot {sid}: "
                                 f"{d.get('description', '')[:120]}")
        lines.append('\nReturn JSONL, one object per window: '
                     '{"window_id": N, "text": "...", "place_phrase": '
                     '"the exact words WITHIN your text that name the '
                     'place"}')
        return {"lines": lines, "targets": sorted(heirs)}

    def on_answer(self, run, state, newc, raw):
        # VERIFIED repair: the claimed place phrase must actually be in
        # the text (the unverified version returned 'Sullivan turns,
        # alarmed.' as a 'grounding' — a coin flip is not a repair).
        # parse_jsonl_cues keeps only text; re-parse place_phrase from
        # the output the handler already holds.
        from lib import parse_partial_output
        place_of = {}
        for line in raw.splitlines():
            obj, _ = parse_partial_output.parse_jsonl_line(
                line.strip(), ("window_id",))
            if obj:
                place_of[int(obj["window_id"])] = obj.get("place_phrase",
                                                          "")
        for k in list(newc):
            ph = (place_of.get(k) or "").strip().lower()
            if len(ph) < 3 or ph not in newc[k].lower():
                logger.info("w%d: succession rewrite FAILED verification "
                            "(no place) — rejected", k)
                newc[k] = ""
        return newc


class _BeatRepair(_Step):
    """For every plot-critical beat with a covering cue, verify the cue
    VOICES THE EVENT (not only its aftermath). User decision 2026-08-27:
    a beat reconstructed across shots happened on screen."""

    def detect(self, run, state):
        pairs = []
        for b in run.notes.get("beats", []):
            sec = float(b["sec"])
            near = [p for p in run.placed
                    if sec - 10 <= p["at"] <= sec + 15
                    and run.cues.get(p["window_id"], "").strip()]
            if near:
                pairs.append((b, min(near,
                                     key=lambda p: abs(p["at"] - sec))))
        if not pairs:
            return None
        lines = ["Below are plot-critical beats of a program and the "
                 "audio-description cue nearest each. For every cue that "
                 "fails to VOICE THE BEAT'S EVENT ITSELF (it describes "
                 "only the aftermath, or omits the event), rewrite it "
                 "within the word cap so the event is stated — 'A van "
                 "strikes a woman crossing; she lies motionless', never "
                 "just the aftermath. Keep place grounding and names; no "
                 "bare-pronoun openers. Cues that already convey their "
                 "event: do NOT return them. A beat CARRIED BY DIALOGUE "
                 "needs NO cue — narration must never parrot what the "
                 "audience hears; skip those beats entirely.",
                 'Output JSONL: {"window_id": N, "text": "..."} — only '
                 "rewritten cues.", "", "BEAT / CUE PAIRS:"]
        targets = []
        for b, p in pairs:
            wid = p["window_id"]
            cap = max(4, int(run.avail_for(p) * C.WPS_REPAIR_CAP))
            lines.append(
                f"\nbeat @{float(b['sec']):.0f}s: "
                f"{str(b.get('what') or '')[:140]}"
                + (" (reconstructed across shots — voice it)"
                   if b.get("inferred") else ""))
            lines.append(f"  cue w{wid} (cap {cap} words): "
                         f"{run.cues[wid]!r}")
            targets.append(wid)
        return {"lines": lines, "targets": sorted(set(targets))}


class _RevealCheck(_Step):
    """A window carrying a REVEAL whose evidence shows an apparatus must
    NAME it — a reveal stripped of its mirror is unreadable."""

    def detect(self, run, state):
        bad = []
        for p in run.placed:
            w = run.ws[p["window_id"]]
            fam = w.get("reveal_mechanism_family")
            text = run.cues.get(p["window_id"], "")
            if not fam or not text.strip():
                continue
            import re
            if not any(re.search(r"\b" + m, text, re.I) for m in fam):
                bad.append(p["window_id"])
        if not bad:
            return None
        lines = ["Each cue below describes a REVEAL that the shot "
                 "evidence shows happening THROUGH an apparatus — a "
                 "mirror, a screen, a doorway, a photograph — and the "
                 "cue fails to name it. Without the apparatus the "
                 "audience cannot tell where the person is or how they "
                 "are seen. Rewrite each cue within its cap so the "
                 "apparatus is named. ONE subject only: drop the second "
                 "person, the adjectives and the place before you drop "
                 "the apparatus.",
                 'Output JSONL: {"window_id": N, "text": "..."}', "",
                 "CUES:"]
        for wid in bad:
            w = run.ws[wid]
            p = next(x for x in run.placed if x["window_id"] == wid)
            cap = max(4, int(run.avail_for(p) * C.WPS_REPAIR_CAP))
            evd = " || ".join(
                str(run.docs.get(s, {}).get("description", ""))[:110]
                for s in w["shot_ids"][:2])
            lines.append(f"\nw{wid} | apparatus: {w['reveal_mechanism']} "
                         f"| HARD CAP {cap} WORDS")
            lines.append(f"  reveal: {w['reveal'][:140]}")
            lines.append(f"  current cue: {run.cues[wid]!r}")
            lines.append(f"  shot evidence: {evd}")
        return {"lines": lines, "targets": bad}


class _RoleDescriptorCheck(_Step):
    def detect(self, run, state):
        hits = oracles.descriptor_hits(
            run.cmap, run.ws,
            {str(k): v for k, v in run.cues.items()}, run.placed,
            run.docs)
        if not hits:
            return None
        lines = ["Each cue below refers to a character by a ROLE "
                 "DESCRIPTOR when that character's NAME is already "
                 "established for the audience. Audio-description "
                 "practice is to name the person (DCMP: when in doubt "
                 "repeat the name rather than use a pronoun) — a "
                 "descriptor for a named character makes the audience "
                 "wonder whether it is somebody new. Replace the "
                 "descriptor with the name, same length or shorter, "
                 "changing nothing else.",
                 'Output JSONL: {"window_id": N, "text": "..."}', "",
                 "CUES:"]
        for wid, (phrase, fk, nm) in hits.items():
            lines.append(f"w{wid} | {phrase!r} refers to {nm} (face key "
                         f"{fk}, on screen in this window) | cue: "
                         f"{run.cues[wid]!r}")
        return {"lines": lines, "targets": sorted(hits)}


class _OnscreenTextCheck(_Step):
    """A cue we cannot make true is a cue that must not reach the mix
    (drop_on_fail + no_longer)."""

    def detect(self, run, state):
        bad = oracles.false_text_claims(
            run.ws, {str(k): v for k, v in run.cues.items()},
            run.placed, run.docs, run.tm)
        if not bad:
            return None
        lines = ["Each cue below tells the audience that words appear ON "
                 "SCREEN, and the shot evidence under that cue shows no "
                 "such words. A false claim about on-screen text is "
                 "undetectable by a blind listener, so it may not ship. "
                 "Rewrite each cue to describe only what the evidence "
                 "supports — DELETE the on-screen-text clause and spend "
                 "the words on the picture. Do not invent different "
                 "text. Same length or shorter.",
                 'Output JSONL: {"window_id": N, "text": "..."}', "",
                 "CUES:"]
        for wid, m in bad.items():
            w = run.ws[wid]
            evd = " || ".join(
                str(run.docs.get(s, {}).get("description", ""))[:110]
                for s in w["shot_ids"][:2])
            # HARD CAP, not "shorter" in prose (measured: the uncapped
            # repair grew past its slot and an innocent neighbour was
            # dropped for running into dialogue)
            cap = max(4, len(run.cues[wid].split()))
            lines.append(f"\nw{wid} @{m['at']}s | HARD CAP {cap} WORDS "
                         f"| {m['note']}")
            lines.append(f"  current cue: {run.cues[wid]!r}")
            lines.append(
                f"  on-screen text actually under this cue: "
                f"{[t for v in m['evidence'].values() for t in v] or 'NONE'}")
            if m["text_lives_at"]:
                lines.append(
                    "  (that text appears elsewhere in the programme, "
                    "at " + ", ".join(f"{s:.1f}s: {t!r}"
                                      for s, t in m["text_lives_at"])
                    + " — not here)")
            lines.append(f"  shot evidence: {evd}")
        return {"lines": lines, "targets": sorted(bad),
                "extra": {"drop_records": {str(k): v
                                           for k, v in bad.items()}}}


class _NameLeakCheck(_Step):
    def detect(self, run, state):
        leaks = oracles.name_leaks(
            run.cmap, {str(k): v for k, v in run.cues.items()},
            run.placed, run.ws, run.docs)
        if not leaks:
            return None
        lines = ["Each cue below LEAKS an actor's name — forbidden in "
                 "audio description. Rewrite the cue using the correct "
                 "character name or descriptor given, same length or "
                 "shorter, keeping everything else.",
                 'Output JSONL: {"window_id": N, "text": "..."}', "",
                 "CUES:"]
        for wid, (tok, fk, correct) in leaks.items():
            lines.append(f"w{wid} | leaked {tok!r} (actor {fk}); correct "
                         f"reference: {correct!r} | cue: "
                         f"{run.cues[wid]!r}")
        return {"lines": lines, "targets": sorted(leaks),
                "extra": {"drop_records": {
                    str(k): {"window_id": k, "kind": "name_leak",
                             "note": f"leaked {v[0]!r}; fix unfit — a "
                                     "leak may not ship"}
                    for k, v in leaks.items()}}}


class _TailBlockCheck(_Step):
    """The closing block is ONE announcement; principal cards and
    MESSAGES are read, logos get one line, the crew scroll is never read
    role by role. credits.json (frame-inspection) upgrades this from
    guessing at unreadable cards to quoting read ones."""

    def detect(self, run, state):
        sh = run.ev["shots"]
        dur = float(max(s["end_time"] for s in sh))
        m = oracles.tail_overdescribed(
            run.docs, run.tm, dur,
            {str(k): v for k, v in run.cues.items()}, run.placed)
        if not m:
            return None
        # the witness product is {"blocks": [...], "gaps": [...]} —
        # iterating the DOC yielded its string keys (latent since the
        # fi-pointer era, confirmed against the producer 2026-09-08;
        # reachable on every cast-anchored run now that credits ride
        # cast-resolution)
        credits = (run.ev.get("credits") or {}).get("blocks") or []
        lines = ["The programme is OVER. Everything below is closing "
                 "matter. Broadcast audio-description practice, measured "
                 "from professional tracks: tell the listener ONCE that "
                 "the closing credits are running; READ the principal "
                 "cards a sighted viewer receives — guest cast, lead "
                 "producer/creator cards, a based-on-the-source card — "
                 "and any card carrying a MESSAGE (a dedication, a "
                 "memorial, a legal or location notice). Production and "
                 "distributor logos are named in ONE line at most "
                 "('Logos: X, Y.'), never one by one. The crew scroll is "
                 "NEVER read role by role.", "",
                 "For EVERY window listed, return a line. Give the text "
                 "you want spoken, or an empty string to make that "
                 "window silent. Keep each surviving cue within its "
                 "word cap.",
                 "TIMING: a card is read WHILE IT IS ON SCREEN. Cards "
                 "are labeled with their time (t=...); put each reading "
                 "in the window NEAREST that time — never bundle every "
                 "card into the first window and silence the rest.",
                 'Output JSONL: {"window_id": N, "text": "..."}', "",
                 "WINDOWS:"]
        for wid in m["cues"]:
            w = run.ws[wid]
            p = next(x for x in run.placed if x["window_id"] == wid)
            avail = run.avail_for(p)
            cap = max(4, int(avail * C.WPS_REPAIR_CAP))
            sids = w["shot_ids"] or [s for s, (a, b) in run.tm.items()
                                     if a <= p["at"] <= b]
            # the ONE text reader (both artifact shapes: texts_on_screen
            # is the unclaimed remainder since SME 2026-09-09)
            txts = [t for s in sids
                    for t in oracles.shot_texts(run.docs.get(s, {}))]
            lines.append(
                f"\nw{wid} @{p['at']:.0f}s | shot kind(s): "
                f"{sorted({run.docs.get(s, {}).get('shot_kind') for s in sids})}"
                f" | HARD CAP {cap} WORDS")
            lines.append(f"  current cue: {run.cues[wid]!r}")
            lines.append(f"  on-screen text on this card: "
                         f"{txts[:12] or 'NONE'}")
            if credits:
                near = [c for blk in credits for c in blk["cards"]
                        if p["at"] - 2 <= (c.get("t") or -1)
                        <= p["at"] + avail + 2 and c.get("lines")]
                for c in near[:8]:
                    lines.append(f"  CARD READ at t={c['t']}: "
                                 + " / ".join(c["lines"])[:160])
        lines.append(f"\nWhole block for context: {m['start']:.0f}-"
                     f"{m['end']:.0f}s, {len(m['shot_ids'])} shots, "
                     f"kinds {'+'.join(m['kinds'])}.")
        if credits:
            blk = [b for b in credits if b["start"] >= m["start"] - 5]
            allc = [c for b in blk for c in b["cards"] if c.get("lines")]
            lines.append("EVERY CARD READ IN THE BLOCK (verbatim, for "
                         "your selection):")
            for c in allc[:40]:
                lines.append(f"  t={c['t']}: "
                             + " / ".join(c["lines"])[:160])
        return {"lines": lines, "targets": m["cues"],
                "extra": {"tail": m}}

    def on_answer(self, run, state, newc, raw):
        m = state["extra"]["tail"]
        silent = [wid for wid in m["cues"]
                  if not newc.get(wid, "").strip()]
        speak = [wid for wid in m["cues"] if newc.get(wid, "").strip()]
        if not speak:
            # a block that says nothing is worse than one that says too
            # much: keep the earliest cue as the announcement
            speak, silent = m["cues"][:1], m["cues"][1:]
            for wid in speak:
                newc[wid] = run.cues[wid]
            logger.info("tail: model silenced the whole block — keeping "
                        "the first cue as the announcement")
        ok, refused = run.safe_deletions(silent)
        for wid, why in refused:
            logger.info("w%d KEPT despite the plan — %s", wid, why)
            speak.append(wid)
            newc.pop(wid, None)
        for wid in ok:
            run.placed[:] = [x for x in run.placed
                             if x["window_id"] != wid]
            run.drop({"window_id": wid, "kind": "nondiegetic_tail",
                      "keeper": speak[0],
                      "note": f"closing block {m['start']:.0f}-"
                              f"{m['end']:.0f}s: credit and logo cards "
                              f"are not read one by one; announced in "
                              f"w{speak[0]}"})
        run.persist()
        # only synthesize actual CHANGES
        return {wid: newc[wid] for wid in speak
                if wid in newc
                and newc[wid].strip() != run.cues.get(wid, "").strip()}


class _RedundancyCheck(_Step):
    """ECHOES are deleted through the coverage guard (no model, no TTS);
    two-sided overlaps are DIFFERENTIATED — the later cue rewritten to
    say only what is new (the user's 'w10, w11, w12 should combine',
    without pushing the reveal 14s past its cut)."""

    def detect(self, run, state):
        cues_s = {str(k): v for k, v in run.cues.items()}
        pairs = oracles.redundant_pairs(cues_s, run.placed, run.ws,
                                        run.cmap)
        if not pairs:
            return None
        C.write_json(C.PROXY_BUCKET, f"{run.prefix}/redundancy.json",
                     pairs)
        kept, cand = [], []
        for m in pairs:
            e = m["echo"]
            if e is None:
                kept.append(m)
                continue
            # No duty-FLAG pre-filter (measured: it refused deletions
            # that cost nothing); the coverage guard is the one guard.
            if not any(e == prev for prev, _, _ in cand):
                cand.append((e, m["second"] if e == m["first"]
                             else m["first"], m))
        ok, refused = run.safe_deletions([e for e, _, _ in cand])
        for wid, why in refused:
            m = next(m for e, _, m in cand if e == wid)
            m["note_duty"] = f"kept: deleting w{wid} {why}"
            kept.append(m)
        for e, keep, m in ((e, k, m) for e, k, m in cand if e in ok):
            run.placed[:] = [x for x in run.placed
                             if x["window_id"] != e]
            run.drop({"window_id": e, "kind": "redundant",
                      "keeper": keep, "gap_sec": m["gap_sec"],
                      "containment": m["containment"],
                      "note": f"deleted as an echo: every content word "
                              f"was already in w{keep}, "
                              f"{m['gap_sec']}s away"})
        run.persist()
        if not kept:
            return None
        # differentiate CHAINS, not pairs (w10-w11 + w11-w12 are one
        # problem)
        chains = []
        for m in sorted(kept, key=lambda m: m["at"]):
            for c in chains:
                if m["first"] in c or m["second"] in c:
                    c.update([m["first"], m["second"]])
                    break
            else:
                chains.append({m["first"], m["second"]})
        targets, lines = [], [
            "Each GROUP below is a run of consecutive cues that say the "
            "same thing more than once. A listener hears them seconds "
            "apart, so the repetition reads as a stutter. Keep the "
            "EARLIEST cue in each group as it is. Rewrite the later "
            "ones so each says only what is NEW since the one before it "
            "— new subject, new action, new consequence. If a later cue "
            "has nothing new to add, return an empty string.",
            'Output JSONL: {"window_id": N, "text": "..."}', "",
            "GROUPS:"]
        for c in chains:
            ids = sorted(i for i in c
                         if any(p["window_id"] == i for p in run.placed))
            if len(ids) < 2:
                continue
            lines.append(f"\ngroup {ids}:")
            for i, wid in enumerate(ids):
                p = next(x for x in run.placed if x["window_id"] == wid)
                # never MORE than the cue being deduplicated (a
                # music-dominant cap reaches 250 words; a dedupe stage
                # must not hand back a longer cue)
                cap = max(4, min(int(run.avail_for(p) * C.WPS_REPAIR_CAP),
                                 len(run.cues.get(wid, "").split())))
                role = ("KEEP AS IS (earliest)" if i == 0
                        else f"REWRITE, HARD CAP {cap} WORDS")
                lines.append(f"  w{wid} @{p['at']:.1f}s [{role}]: "
                             f"{run.cues.get(wid, '')!r}")
                if i:
                    targets.append(wid)
        if not targets:
            return None
        return {"lines": lines, "targets": targets}

    def on_answer(self, run, state, newc, raw):
        # an empty answer is a DELETION and owes the same proof
        blank = [k for k in state["targets"]
                 if k in newc and not newc[k].strip()]
        ok, refused = run.safe_deletions(blank)
        for wid, why in refused:
            logger.info("w%d kept though the model had nothing to add "
                        "— %s", wid, why)
        for wid in ok:
            run.placed[:] = [x for x in run.placed
                             if x["window_id"] != wid]
            run.drop({"window_id": wid, "kind": "redundant",
                      "note": "nothing new to say after the cue "
                              "before it"})
        run.persist()
        return newc


_STEPS = {
    "revive_entries": _ReviveEntries(),
    "fix_overlaps": _FixOverlaps(),
    "entry_succession": _EntrySuccession(),
    "beat_repair": _BeatRepair(),
    "reveal_check": _RevealCheck(),
    "role_descriptor_check": _RoleDescriptorCheck(),
    "onscreen_text_check": _OnscreenTextCheck(),
    "name_leak_check": _NameLeakCheck(),
    "tail_block_check": _TailBlockCheck(),
    "redundancy_check": _RedundancyCheck(),
}
