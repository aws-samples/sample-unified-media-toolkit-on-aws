"""
BindPersons — the whole free-evidence pipeline in one state, because nothing
branches between its halves: load the sibling outputs, bind through
lib/bind_pipeline (the ONE pipeline the local harnesses and the gate
scorer also run), fold witness products in, write the labeled products.
Absent sources make their outputs ABSENT, never empty (three-state
semantics), and the maps are pinned to the sibling workflow_ids recorded
in the plan.

Witness products are located by fi's OWN manifest
({output_s3_prefix}/inspection_manifest.json, products[] with category +
s3uri) from the prefix the WAKE PAYLOAD carries — never by guessing
layout, and never via reads.json, whose write races an absorbed-closed
wake (review F4). Products successfully consumed are appended to the
cross-run ledger: evidence OBTAINED, so a failed read never suppresses
its own retry.
"""
from __future__ import annotations

import time

from lib import bind_pipeline
from lib import ledger
from lib import persons
from lib import reads
from lib import s3io
from lib import sources
from lib import vocatives_llm


def _addressee_cues(prefix: str, round1: dict, src: dict) -> list[dict]:
    """Round-1 wake payload -> resolved addressee cues (votes ->
    thresholds -> variant clustering). Anything short of a clean
    SUCCEEDED read degrades to [] — free evidence still binds."""
    if round1.get("status") != "SUCCEEDED":
        src["addressees"] = round1.get("reason") or \
            round1.get("status") or "skipped"
        return []
    try:
        windows = s3io.get_json(f"{prefix}/_enrich/windows.json")
        texts = s3io.get_response_texts(round1["output_s3_prefix"])
        cues = vocatives_llm.aggregate(
            vocatives_llm.parse_answers(texts, windows))
        src["addressees"] = f"resolved:{len(cues)}"
        return cues
    except Exception as exc:                     # noqa: BLE001 — degrade
        src["addressees"] = f"consume_failed: {str(exc)[:100]}"
        return []


def _witness_products(prefixes: list[str], src: dict):
    """Fetch credits/titles products via each run's manifest.
    -> (credits_doc | None, title_cues, all_fetches_clean)"""
    credits_doc, title_cues, clean = None, [], True
    for base in prefixes:
        base = base.rstrip("/")
        try:
            manifest = s3io.get_json(f"{base}/inspection_manifest.json")
            for prod in manifest.get("products") or []:
                doc = s3io.get_json(prod["s3uri"])
                if prod["category"] == "title-read":
                    title_cues += reads.cues_from_titles(doc)
                elif prod["category"] == "credits-read":
                    credits_doc = doc
        except Exception as exc:                 # noqa: BLE001 — degrade
            src["reads_consumed"] = f"failed: {str(exc)[:120]}"
            clean = False
    return credits_doc, title_cues, clean


def _reads_usage(reads_state: str, wait: dict, prefixes: list[str],
                 clean: bool) -> dict:
    """Honest read-cost accounting (review F8): a run that scheduled
    witness reads reports the witness run's OWN recorded usage — a
    hardcoded $0.00 on a run that spent Bedrock money is a lie."""
    if reads_state == "ready":
        total, complete, runs = 0.0, clean, []
        for base in prefixes:
            u = None
            try:
                u = s3io.get_json(
                    f"{base.rstrip('/')}/usage_tracking.json")
            except Exception:                    # noqa: BLE001
                pass
            if not u or u.get("total_estimated_cost") is None:
                complete = False
                continue
            total += float(u["total_estimated_cost"])
            if u.get("cost_is_complete") is False:
                complete = False
            runs.append({"output_s3_prefix": base,
                         "estimated_cost": u["total_estimated_cost"]})
        return {"total_estimated_cost": round(total, 4),
                "cost_is_complete": complete,
                "witness_runs": runs,
                "note": "witness read cost from the frame-inspection "
                        "run's own usage_tracking"}
    if reads_state == "ledgered":
        return {"total_estimated_cost": 0.0, "cost_is_complete": True,
                "witness_runs": [{"output_s3_prefix": p}
                                 for p in prefixes],
                "note": "witness products reused from the cross-run "
                        "ledger; the read cost was borne by the run "
                        "that obtained them"}
    if reads_state == "failed":
        return {"total_estimated_cost": 0.0, "cost_is_complete": False,
                "note": "witness read run did not succeed; any spend it "
                        "made is recorded in its own usage_tracking",
                "reads_job_id": wait.get("job_id")}
    return {"total_estimated_cost": 0.0, "cost_is_complete": True,
            "note": "free-evidence binding — no witness reads "
                    f"({wait.get('reads_reason') or 'skipped'})"}


def lambda_handler(event: dict, context) -> dict:
    t0 = time.time()
    plan_s3uri = event["plan_s3uri"]
    plan = s3io.get_json(plan_s3uri)
    uuid = plan["uuid"]
    inputs = plan["inputs"]
    prefix = plan_s3uri.rsplit("/_plan/", 1)[0]

    transcript = sources.load_transcript(
        inputs.get("speech_diarization_s3uri"))
    # per-cluster F0 stats (additive diarization fields, 2026-09-23) —
    # best-effort: feeds only class1's compatibility veto
    speaker_f0 = sources.load_speaker_f0(
        inputs.get("speech_diarization_s3uri"))
    faces = sources.load_faces(inputs.get("face_recognition_s3uri"))
    shots = sources.load_shots(inputs.get("shot_segmentation_s3uri"))
    # ONE fetch of shot_metadata.jsonl feeds every signal derived from it
    md_docs = sources.load_shot_metadata_docs(
        inputs.get("shot_metadata_s3uri"))
    overlay = (sources.parse_overlay_names(md_docs, shots)
               if md_docs and shots is not None else None)
    screen_texts = (sources.parse_screen_texts(md_docs, shots)
                    if md_docs and shots is not None else None)
    # the depicted-face guard: mark BEFORE bind() (live_spans for
    # co-occurrence, depicted_share for typing; absent media_depictions
    # upstream = not assessed = nothing marked)
    sources.mark_depicted_faces(
        faces or [],
        sources.parse_media_depiction_spans(md_docs or [], shots or {}))

    src = {k.removesuffix("_s3uri"): v for k, v in inputs.items()}

    # round 1 ran BEFORE this state: its resolved addressees join the
    # bind as ordinary cues — ONE bind, no downstream re-bind, so the
    # written products can never lose evidence to an enrichment step
    round1 = event.get("addressees_result") or {}
    extra_cues = _addressee_cues(prefix, round1, src)

    # reads_result is the witness held state's token payload: fi's
    # handle + terminal status AS DATA, our own skip, or the ledgered
    # prefixes. A failed/skipped read degrades to free evidence.
    wait = event.get("reads_result") or {}
    reads_state = ("skipped" if wait.get("reads") == "skipped"
                   else "ledgered" if wait.get("reads") == "ledgered"
                   else "ready" if wait.get("status") == "SUCCEEDED"
                   else "failed")
    read_prefixes = ([wait["output_s3_prefix"]] if reads_state == "ready"
                     else wait.get("read_prefixes") or []
                     if reads_state == "ledgered" else [])
    # never-re-pay must not become never-RE-USE (measured 2026-09-09:
    # a run whose earlier spans were ledger-blocked freed budget for a
    # FRESH read tranche, took the "ready" path, and consumed ONLY the
    # fresh run's products — the ledgered credits-read vanished from
    # credits.json on every round-2 asset). Union the ledger's recorded
    # product prefixes into consumption on ready AND ledgered paths,
    # LEDGER ORDER first (append-ordered, oldest->newest) with the
    # fresh prefix last, so the newest credits product wins the
    # last-writer overwrite in _witness_products.
    if reads_state in ("ready", "ledgered"):
        try:
            rd = s3io.get_json(plan_s3uri.replace("plan.json",
                                                  "reads.json"))
            ordered, seen = [], set()
            for e in ledger.load(rd["ledger_uri"]):
                pfx = e.get("output_s3_prefix")
                if pfx and pfx not in seen:
                    seen.add(pfx)
                    ordered.append(pfx)
            for pfx in read_prefixes:
                if pfx not in seen:
                    seen.add(pfx)
                    ordered.append(pfx)
            read_prefixes = ordered
        except Exception:                        # noqa: BLE001 — degrade
            pass                    # fresh products alone, as before
    credits_doc, title_cues, clean = _witness_products(read_prefixes, src)
    if title_cues:
        overlay = (overlay or []) + title_cues
    src["reads"] = reads_state

    doc = bind_pipeline.bind(transcript, faces, overlay, src, credits_doc,
                             extra_cues=extra_cues, speaker_f0=speaker_f0,
                             screen_texts=screen_texts)
    if credits_doc:
        # persist the witness product ONCE, at the run root as a
        # CONTRACT output (promoted from _enrich/, 2026-09-08): the
        # adjudication states need the cast universe, and
        # audio-description consumes the card inventory for its
        # closing-credits narration — this replaces AD's direct
        # frame-inspection input (fi is internal; only this workflow
        # can order its reads)
        s3io.put_json(f"{prefix}/credits.json", credits_doc)
    doc["provenance"] = {
        "tool_version": plan["tool_version"],
        "inputs": inputs,
        "sibling_workflow_ids": plan["sibling_workflow_ids"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    smap, fmap = persons.remap_tables(doc)
    pin = {"sibling_workflow_ids": plan["sibling_workflow_ids"]}

    written = ["persons.json"]
    s3io.put_json(f"{prefix}/persons.json", doc)
    if transcript is not None:
        s3io.put_json(f"{prefix}/speaker_map.json",
                      {**pin, "speakers": smap})
        written.append("speaker_map.json")
    if faces is not None:
        s3io.put_json(f"{prefix}/face_map.json", {**pin, "faces": fmap})
        written.append("face_map.json")

    usage = {"pipeline_wall_sec": round(time.time() - t0, 2),
             **_reads_usage(reads_state, wait, read_prefixes, clean)}
    # round 1's spend is accounted HERE, unconditionally — even if
    # round 2 later skips, the money round 1 spent is on the bill
    # (review finding: a round-2 skip must never hide round-1 cost)
    r1p = round1.get("output_s3_prefix")
    if r1p:
        try:
            u1 = s3io.get_json(f"{r1p.rstrip('/')}/usage_tracking.json")
            cost = u1.get("total_estimated_cost")
            usage["addressee_run"] = {"output_s3_prefix": r1p,
                                      "estimated_cost": cost}
            if cost is None or u1.get("cost_is_complete") is False:
                usage["cost_is_complete"] = False
            else:
                usage["total_estimated_cost"] = round(
                    usage["total_estimated_cost"] + cost, 4)
        except Exception:                        # noqa: BLE001
            usage["cost_is_complete"] = False
    # ApplyEnrichment recomputes from this base instead of += so a
    # retried apply can never double-count (idempotent by construction)
    usage["pre_enrichment_total"] = usage["total_estimated_cost"]
    s3io.put_json(f"{prefix}/usage_tracking.json", usage)
    written.append("usage_tracking.json")

    # ledger append LAST, and only for evidence actually consumed clean
    # this run — best-effort (reads.json may lose the race with an
    # absorbed-closed wake; a lost append costs one duplicate read)
    if reads_state == "ready" and clean:
        try:
            rd = s3io.get_json(plan_s3uri.replace("plan.json",
                                                  "reads.json"))
            ledger.append(rd["ledger_uri"], [
                {**c, "output_s3_prefix": wait["output_s3_prefix"]}
                for c in rd.get("ledger_candidates") or []])
        except Exception:                        # noqa: BLE001
            pass

    ps = doc["persons"]
    return {"uuid": uuid,
            "persons": len(ps),
            "named": sum(1 for p in ps if p["name"]),
            "cross_modal": sum(1 for p in ps
                               if p["type"] == "on_camera_speaker"),
            "vetoes": len(doc.get("vetoes") or []),
            "reads": reads_state,
            "outputs_written": written}
