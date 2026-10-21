"""
StateCollectAndFinalize (README.md §3-§4): one state, nothing branches
between its halves. Reads the bedrock run's response.jsonl, parses the
per-window JSONL (strict first; failures climb output_parse's counted
salvage ladder), validates coverage per
shot_id, applies the §4 pairing repair (`false` wins, both directions,
COUNTED — the repair rate is an interleaved-genre detector), assembles
scenes from the repaired flags, and writes the three products:

    shot_metadata.jsonl    label: shots            THE product
    scenes.json            label: scenes           flags → scenes + rollups
    window_manifest.json   label: window-manifest  traceability

Partial coverage is reported HONESTLY and the run SUCCEEDS: unanswered
shot_ids land in `shots_gap` (partial success IS success); the semantic
retry is the CALLER's fresh idempotency_token. A terminally FAILED
bedrock run raises ShotMetadataInferenceFailed — no Catch, no Fail state.
"""
from __future__ import annotations

import json
import logging

from lib import output_parse
from lib import scene_assembly
from lib import sme_common
from lib.custom_exceptions import ShotMetadataInferenceFailed

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = sme_common.read_json(
        *sme_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]

    status = (event.get("wait_result") or {}).get("status")
    if status != "SUCCEEDED":
        raise ShotMetadataInferenceFailed(
            f"bedrock-inference ended {status} — see s3://"
            f"{sme_common.PROXY_BUCKET}/{prefix}/_plan/runs.json and the "
            "run's summary.json for per-record reasons")

    window_results, docs, repairs = _collect(plan, prefix)
    sme_common.write_json(sme_common.PROXY_BUCKET,
                          f"{prefix}/_plan/window_results.json",
                          window_results)
    return _finalize(event, plan, prefix, window_results, docs, repairs)


def _collect(plan: dict, prefix: str):
    runs = sme_common.read_json(
        sme_common.PROXY_BUCKET, f"{prefix}/_plan/runs.json")
    run = runs[-1]
    _, run_key = sme_common.parse_s3_uri(run["output_s3_prefix"])
    response_lines = sme_common.read_text(
        sme_common.PROXY_BUCKET, f"{run_key}/response.jsonl").splitlines()

    by_index = {w["index"]: w for w in plan["windows"]}
    # [] asserts "attested: none of these observables is visible", so it
    # is written ONLY when the bound prompt carries the depicted section
    # (default_news does not ask — its docs must stay honest "not
    # assessed", i.e. no key at all). The prune vocabulary is the BOUND
    # prompt's own declared list, so a custom prompt_version brings its
    # own vocabulary instead of being judged against the builtin's.
    prompt_system = plan["prompt"].get("system") or ""
    depicted_asked = "### depicted" in prompt_system
    depicted_vocab = output_parse.depicted_vocabulary(prompt_system)
    media_asked = "### media_depictions" in prompt_system
    media_vocab = output_parse.media_vocabulary(prompt_system)
    window_results: dict = {}
    docs: dict = {}
    repairs: list = []
    for raw in response_lines:
        line = json.loads(raw)
        record_id = line.get("record_id") or ""
        if not record_id.startswith("window-"):
            continue
        index = int(record_id.split("-")[1])
        window = by_index[index]
        required = list(window["shot_ids"])
        if line.get("status") != "SUCCESS":
            # A FAILED line on a SUCCEEDED run is the documented shape —
            # this window's shots become honest gaps.
            logger.warning("window %d FAILED in bedrock: %s",
                           index, line.get("error"))
            window_results[str(index)] = {
                "ok": False, "missing": required, "extra": [],
                "duplicates": [], "bad_lines": 0, "line_repairs": {},
                "repairs": 0,
                "error": str(line.get("error") or "")[:300]}
            continue
        parsed = output_parse.parse_response(line.get("output_text") or "")
        verdict = output_parse.validate_coverage(parsed["docs"], required)
        window_repairs = 0
        for sid in required:
            doc = parsed["docs"].get(sid)
            if doc is None:
                continue
            r = output_parse.repair_pairing(doc)
            if r["repaired"]:
                window_repairs += 1
                repairs.append({"shot_id": sid, "window": index,
                                "kind": r["kind"],
                                "reason": doc.get("boundary_reason")})
            docs[sid], pruned_acts = output_parse.normalize_doc(
                doc, depicted_vocab, media_vocab)
            if pruned_acts:
                logger.warning(
                    "shot %s: pruned out-of-vocab attested entr(ies) %s — "
                    "the prompt's closed list held everywhere else; a "
                    "rising count here is the prompt-regression signal",
                    sid, pruned_acts)
            if depicted_asked and "depicted" not in docs[sid]:
                # the prompt asked and the sweep is never-optional, so a
                # model omission means "nothing visible" — make that []
                # explicit (the lost-list degradation rule)
                docs[sid]["depicted"] = []
            if media_asked and "media_depictions" not in docs[sid]:
                # same attestation rule for the depicted-face guard
                # channel (cast-resolution branches on []-vs-absent)
                docs[sid]["media_depictions"] = []
        if parsed["line_repairs"]:
            # A recovery must never be silent: a rising count here is the
            # prompt regression signal the strict parser used to give by
            # dropping the line outright.
            logger.warning("window %d: %d line(s) recovered by the salvage "
                           "ladder %s", index,
                           sum(parsed["line_repairs"].values()),
                           parsed["line_repairs"])
        window_results[str(index)] = {
            "ok": verdict["ok"], "missing": verdict["missing"],
            "extra": verdict["extra"],
            "duplicates": parsed["duplicates"],
            "bad_lines": len(parsed["bad_lines"]),
            "line_repairs": parsed["line_repairs"],
            "repairs": window_repairs,
        }
    return window_results, docs, repairs


def _finalize(event: dict, plan: dict, prefix: str,
              window_results: dict, docs: dict, repairs: list) -> dict:
    shots_doc = sme_common.read_json(
        *sme_common.parse_s3_uri(plan["shots"]["s3uri"]))
    shots_by_id = {s["shot_id"]: s for s in shots_doc["shots"]}
    ordered_ids = [sid for w in plan["windows"] for sid in w["shot_ids"]]
    gaps = [sid for sid in ordered_ids if sid not in docs]

    # THE product: one line per answered shot, in shot order, each line
    # self-contained and joined to shots.json by shot_id (times live
    # there — the model never emits one).
    lines = [json.dumps(docs[sid], ensure_ascii=False,
                        separators=(",", ":"))
             for sid in ordered_ids if sid in docs]
    sme_common.write_text(sme_common.PROXY_BUCKET,
                          f"{prefix}/shot_metadata.jsonl",
                          "\n".join(lines) + ("\n" if lines else ""))

    scenes = scene_assembly.assemble_scenes(docs, shots_by_id, ordered_ids) \
        if ordered_ids else []
    scenes_doc = {
        "scenes": scenes,
        "shots_total": len(ordered_ids),
        "shots_gap": gaps,
        "pairing_repairs": repairs,
        # 0% on linear content, 2-8% on interleaved (§4) — the measured
        # genre signal, reported not swallowed.
        "pairing_repair_rate": (round(len(repairs) / len(ordered_ids), 4)
                                if ordered_ids else 0.0),
        # name supers the model transcribed but extracted no entity from
        # — the prompt's pairing rule made this a MUST (2026-09-09); the
        # gauge tells us whether the rule holds, reported not swallowed.
        "name_super_entity_gap": output_parse.name_super_entity_gap(docs),
        "provenance": {
            "prompt": {"category": plan["prompt"]["category"],
                       "version": plan["prompt"]["version"],
                       "source_md5": plan["prompt"]["source_md5"]},
            "model_chain": plan["model_chain"],
            "use_batch": plan["use_batch"],
            "shots_s3uri": plan["shots"]["s3uri"],
            "faces_s3uri": plan["faces"]["s3uri"],
            "transcript_s3uri": plan["transcript"]["s3uri"],
            "window_count": len(plan["windows"]),
            "history_scope": plan["scope"],
            "tool_version": plan["tool_version"],
        },
    }
    sme_common.write_json(sme_common.PROXY_BUCKET,
                          f"{prefix}/scenes.json", scenes_doc)

    runs = sme_common.read_json(
        sme_common.PROXY_BUCKET, f"{prefix}/_plan/runs.json")
    manifest = {
        "workflow_id": plan["workflow_id"],
        "shot_target": plan["shot_target"],
        "image_budget": plan["image_budget"],
        "windows": [
            {**{k: w[k] for k in ("index", "shot_ids", "context_shot_id",
                                  "image_blocks")},
             "resolved": bool((window_results.get(str(w["index"])) or {})
                              .get("ok"))}
            for w in plan["windows"]],
        # POINTERS, the history discipline: raw responses stay at the
        # bedrock run's own prefix — this is the join, never a copy.
        "inference_runs": runs,
    }
    sme_common.write_json(sme_common.PROXY_BUCKET,
                          f"{prefix}/window_manifest.json", manifest)

    logger.info("finalized: %d/%d shot(s), %d scene(s), %d repair(s), "
                "%d gap(s)", len(docs), len(ordered_ids), len(scenes),
                len(repairs), len(gaps))
    return {
        "uuid": event["uuid"],
        "plan_s3uri": event["plan_s3uri"],
        "shots_enriched": len(docs),
        "shots_gap": len(gaps),
        "scenes": len(scenes),
        "pairing_repairs": len(repairs),
    }
