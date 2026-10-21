"""
ApplyEnrichment — the enrichment consumer, a plain Task that AMENDS.

BindPersons wrote the authoritative persons.json (provenance pin,
witness cues, round-1 addressees, round-1 cost all included — the
durability line). This state only folds round 2 in: split the pooled
response by verdict family, firewall the adjudication verdicts
(verbatim universe + citations + agreement), apply them, re-run the
tooth-1 critic (demote-only — a bad assignment is withdrawn, not
shipped), attach the report-only tooth-2 flags, rewrite persons.json +
remap tables, and set usage_tracking's total from the RECORDED base +
round-2 cost — recomputed, never +=, so a retried apply can never
double-count.

A skipped/failed round 2 is a cheap no-op: everything round 1 delivered
is already in BindPersons' products.
"""
from __future__ import annotations

import time

from lib import adjudicate
from lib import critic
from lib import critic2
from lib import persons
from lib import reads
from lib import s3io
from lib import sources


def lambda_handler(event: dict, context) -> dict:
    plan_s3uri = event["plan_s3uri"]
    plan = s3io.get_json(plan_s3uri)
    uuid = plan["uuid"]
    prefix = plan_s3uri.rsplit("/_plan/", 1)[0]
    round2 = event.get("adjudication_result") or {}

    if round2.get("adjudication") == "skipped" \
            or round2.get("status") != "SUCCEEDED":
        return {"uuid": uuid, "enrichment": "skipped",
                "reason": round2.get("reason")
                or round2.get("status") or "no_round2"}

    doc = s3io.get_json(f"{prefix}/persons.json")
    ctx = s3io.get_json(f"{prefix}/_enrich/context.json")
    texts = s3io.get_response_texts(round2["output_s3_prefix"])

    verdicts = adjudicate.parse_verdicts(texts, ctx["cast"],
                                         ctx["dossiers"])
    doc = adjudicate.apply_verdicts(doc, verdicts)
    # over-clustering accommodation BEFORE the critic (user direction,
    # 2026-09-09): weld adjudicated voices into their collection-named
    # faces — otherwise the collision sweep orphans exactly the people
    # who talk the most. The transcript feeds the roles re-run below;
    # a fetch failure degrades both to no-ops.
    tx, turns = None, None
    try:
        tx = sources.load_transcript(
            plan["inputs"].get("speech_diarization_s3uri"))
        turns = tx["turns"] if tx else None
    except Exception:                            # noqa: BLE001
        pass
    doc = persons.accommodate_overclustering(doc)
    # roles ran at BIND — before adjudication welded faces onto voices,
    # so a co-anchor whose face arrived via a verdict never met the
    # threading rung (measured 2026-09-23: the lead anchor finished
    # role=None while a tease-scattered guest didn't). Re-run on the
    # WELDED doc: assign_roles never rewrites an existing role, only
    # completes the persons the welds just made eligible.
    if tx:
        spans: dict = {}
        for t in turns or []:
            spans.setdefault(t["speaker_id"], []).append(
                (t["start"], t["end"]))
        doc = reads.assign_roles(doc, duration_sec=tx.get("duration_sec"),
                                 speech_spans=spans)
        # the speech-seconds invariant, re-enforced after every weld
        # (this rewrite path is where the 2026-09-22 scramble shipped)
        doc = persons.enforce_speech_coherence(
            doc, tx.get("speakers") or {})
    doc = critic.critique(doc)               # demote-only, AFTER apply
    flags = critic2.parse_reviews(texts)
    doc = critic2.attach_flags(doc, flags)
    # provenance pin carries through untouched — this doc IS Bind's doc
    doc["provenance"]["enriched_at"] = \
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    s3io.put_json(f"{prefix}/persons.json", doc)
    smap, fmap = persons.remap_tables(doc)
    pin = {"sibling_workflow_ids": plan["sibling_workflow_ids"]}
    if "speech_diarization_s3uri" in plan["inputs"]:
        s3io.put_json(f"{prefix}/speaker_map.json",
                      {**pin, "speakers": smap})
    if "face_recognition_s3uri" in plan["inputs"]:
        s3io.put_json(f"{prefix}/face_map.json", {**pin, "faces": fmap})

    # usage: recompute from the recorded pre-enrichment base — running
    # this state twice writes the same numbers (idempotent)
    usage = s3io.get_json(f"{prefix}/usage_tracking.json")
    base = usage.get("pre_enrichment_total",
                     usage.get("total_estimated_cost", 0.0))
    r2p = round2.get("output_s3_prefix")
    cost, complete = None, usage.get("cost_is_complete", True)
    try:
        u2 = s3io.get_json(f"{r2p.rstrip('/')}/usage_tracking.json")
        cost = u2.get("total_estimated_cost")
        if u2.get("cost_is_complete") is False:
            complete = False
    except Exception:                            # noqa: BLE001
        pass
    usage["adjudication_run"] = {"output_s3_prefix": r2p,
                                 "estimated_cost": cost}
    if cost is None:
        complete = False
        usage["total_estimated_cost"] = base
    else:
        usage["total_estimated_cost"] = round(base + cost, 4)
    usage["cost_is_complete"] = complete
    s3io.put_json(f"{prefix}/usage_tracking.json", usage)

    ps = doc["persons"]
    return {"uuid": uuid, "enrichment": "applied",
            "adjudicated": len(verdicts),
            "critic_flags": len(flags),
            "persons": len(ps),
            "named": sum(1 for p in ps if p.get("name"))}
