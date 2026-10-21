"""
StateFinalize: the two coverage oracles (beat, orientation), the ledger,
and delivery (srt/vtt/DAPT/review.html + render_manifest.json — the mix
is media-processing's ad_mix group, chained by the caller; this workflow
renders no audio). Reviewer-accepted absences (decisions.json) stop
re-reporting here.
"""
from __future__ import annotations

import logging

from lib import ad_common as C
from lib import coverage
from lib import deliver
from lib import evidence
from lib import fitting
from lib import ledger as ledger_mod
from lib import windows as windows_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = C.read_json(*C.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    ev = evidence.fetch(plan["inputs"])
    ws_list = C.read_json(C.PROXY_BUCKET, f"{prefix}/windows.json")
    ws = {w["window_id"]: w for w in ws_list}
    cues = C.read_json(C.PROXY_BUCKET, f"{prefix}/cues.json")
    placed = C.read_json(C.PROXY_BUCKET, f"{prefix}/placed.json")
    notes = C.read_json(C.PROXY_BUCKET, f"{prefix}/watch_notes.json")
    orientation = C.read_json(C.PROXY_BUCKET,
                              f"{prefix}/orientation.json")
    drops = C.read_json_if_exists(C.PROXY_BUCKET,
                                  f"{prefix}/drops.json") or []
    gaps = C.read_json_if_exists(C.PROXY_BUCKET,
                                 f"{prefix}/ad_gaps.json") or []
    takes = C.read_json_if_exists(C.PROXY_BUCKET,
                                  f"{prefix}/takes.json") or {}

    gaps = _check_beats(gaps, notes, placed, cues)
    gaps = _check_orientation(gaps, orientation, placed, cues, ws, drops)

    # reviewer-accepted absences: a decisions.json entry names a gap the
    # reviewer has already priced — it stops re-reporting
    decisions = ev.get("decisions") or {}
    accepted = {round(float(a), 1)
                for a in (decisions.get("accepted_gap_secs") or [])}
    if accepted:
        gaps = [g for g in gaps
                if round(float(g.get("sec", 0)), 1) not in accepted]
    C.write_json(C.PROXY_BUCKET, f"{prefix}/ad_gaps.json", gaps)

    rows = ledger_mod.build_ledger(ws, cues, placed, orientation, notes,
                                   gaps, drops, takes)
    C.write_json(C.PROXY_BUCKET, f"{prefix}/cue_ledger.json", rows)

    title = plan["title"]
    srt, vtt = deliver.subs(placed, cues)
    dapt_doc = deliver.dapt(placed, cues, title,
                            lang=plan.get("language") or "en")
    html, shipped = deliver.review_html(title, f"{title}.mp4", rows, ws)
    dp = f"{prefix}/delivery"
    C.write_bytes(C.PROXY_BUCKET, f"{dp}/{title}.srt",
                  srt.encode("utf-8"), "text/plain")
    C.write_bytes(C.PROXY_BUCKET, f"{dp}/{title}.vtt",
                  vtt.encode("utf-8"), "text/vtt")
    C.write_bytes(C.PROXY_BUCKET, f"{dp}/{title}.dapt.xml",
                  dapt_doc.encode("utf-8"), "application/ttml+xml")
    C.write_bytes(C.PROXY_BUCKET, f"{dp}/review.html",
                  html.encode("utf-8"), "text/html")

    # the render manifest: ONE pointer media-processing's ad_mix output
    # group consumes (bed ducked under these takes) — the mix is a
    # process_media call the CALLER chains, never audio work here
    render_manifest = {
        "workflow_id": plan["workflow_id"],
        "sample_rate": C.SR,
        "af_chain": fitting.AF,
        "takes": [{"window_id": p["window_id"],
                   "wav": (takes.get(str(p["window_id"])) or {}).get("wav"),
                   "at": p["at"], "tempo": p.get("tempo", 1.0)}
                  for p in sorted(placed, key=lambda p: p["at"])
                  if (takes.get(str(p["window_id"])) or {}).get("wav")],
    }
    C.write_json(C.PROXY_BUCKET, f"{prefix}/render_manifest.json",
                 render_manifest)

    outputs = {
        # Labels are hyphen-separated (family convention; the bundle pins
        # it) — the BASENAMES keep their underscores.
        "render-manifest":
            f"s3://{C.PROXY_BUCKET}/{prefix}/render_manifest.json",
        "cue-ledger": f"s3://{C.PROXY_BUCKET}/{prefix}/cue_ledger.json",
        "windows": f"s3://{C.PROXY_BUCKET}/{prefix}/windows.json",
        "gaps": f"s3://{C.PROXY_BUCKET}/{prefix}/ad_gaps.json",
        "srt": f"s3://{C.PROXY_BUCKET}/{dp}/{title}.srt",
        "vtt": f"s3://{C.PROXY_BUCKET}/{dp}/{title}.vtt",
        "dapt": f"s3://{C.PROXY_BUCKET}/{dp}/{title}.dapt.xml",
        "review": f"s3://{C.PROXY_BUCKET}/{dp}/review.html",
    }

    manifest = {"workflow_id": plan["workflow_id"],
                "tool_version": plan["tool_version"],
                "inputs": plan["inputs"],
                "sung_vocal_policy": plan["sung_vocal_policy"],
                "shipped_cues": shipped, "gaps": len(gaps),
                # no audio is rendered in this workflow: each placed
                # take's wav uri (takes.json) + tempo + this chain is
                # everything a renderer (media-processing, once it grows
                # the operation) needs to produce the fitted takes/mix
                "render_instructions": {
                    "af_chain": fitting.AF,
                    "per_take": "apply af_chain, then atempo={tempo} "
                                "when tempo > 1; place at {at}s; "
                                f"{C.SR} Hz — media-processing's ad_mix "
                                "output group is the canonical renderer "
                                "(it renders takes STEREO against the "
                                "stereo bed)"},
                "outputs": outputs}
    C.write_json(C.PROXY_BUCKET, f"{prefix}/ad_manifest.json", manifest)
    logger.info("finalized: %d cues shipped, %d gaps", shipped,
                len(gaps))
    return {"uuid": plan["workflow_id"], "outputs": outputs}


def _check_beats(gaps, notes, placed, cues):
    """Beat-coverage oracle: every plot-critical beat must be conveyed
    by a placed cue within BEAT_WINDOW, else it lands in ad_gaps with a
    reason. Structural guarantee, not taste."""
    new = []
    for b in notes.get("beats", []):
        sec = float(b["sec"])
        if not coverage.beat_near(sec, placed, cues, b.get("what")):
            new.append({"sec": sec, "beat": b["what"],
                        "reason": "no non-empty cue within [-10,+15]s"})
    return ledger_mod.merge_gaps(gaps, "beat", new)


def _check_orientation(gaps, orientation, placed, cues, ws, drops):
    """Every scene boundary the audience cannot follow by ear alone must
    be served by a cue, or reported — WHY it is unserved decides what
    could fix it, so the cause is named (a wrong cause is worse than
    none; measured on MITJ w48)."""
    from lib import ad_common as CC
    hi = [o for o in orientation
          if o["necessity"] >= CC.NECESSITY_HIGH
          and o.get("montage_member_of") is None]
    gone = {d["window_id"]: d["kind"] for d in drops}
    pids = {p["window_id"] for p in placed}
    unserved = []
    for o in hi:
        near = coverage.orient_near(o, placed, cues)
        if near:
            continue
        dw = [w for w in ws.values()
              if w.get("entry_for_scene") == o["scene_id"]
              or w.get("anticipatory_for_scene") == o["scene_id"]
              or w.get("entry_prelude_for_scene") == o["scene_id"]]
        cut = [w["window_id"] for w in dw if w["window_id"] in gone]
        if not dw:
            why = ("no usable silence within the search horizon — this "
                   "cut has no gap to describe into")
        elif any(not cues.get(str(w["window_id"]), "").strip()
                 for w in dw):
            why = ("duty window w%s exists but its cue is EMPTY (writer "
                   "or critic left it blank)"
                   % ",".join(str(w["window_id"]) for w in dw
                              if not cues.get(str(w["window_id"]),
                                              "").strip()))
        elif cut:
            why = ("duty window w%s was REMOVED by this run (%s)"
                   % (",".join(str(w) for w in cut),
                      ", ".join(sorted({gone[w] for w in cut}))))
        elif any(w["window_id"] not in pids for w in dw):
            why = ("duty window w%s has a cue that would not FIT its gap"
                   % ",".join(str(w["window_id"]) for w in dw
                              if w["window_id"] not in pids))
        else:
            why = "duty cue placed outside the orientation window"
        unserved.append(
            {"sec": o["at"],
             "beat": f"scene {o['scene_id']} orientation break "
                     f"({', '.join(f'{k}: {v}' for k, v in o['factors'].items())})",
             "necessity": o["necessity"], "reason": why})
    return ledger_mod.merge_gaps(gaps, "orientation", unserved)
