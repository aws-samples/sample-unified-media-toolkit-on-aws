"""
StateAssemble: the whole Layer-2 rollup in one pass — pure code over the
fetched sibling artifacts. NOT chunked, deliberately: the §10 chunking
rule targets long-running legs, and this one is seconds over a few MB of
JSON (the inputs are bounded by episode size, not catalogue size); a Map
here would be ceremony.

Products:
  structure.json    CPL markers + break candidates + (narrative) sequences
                    (+ acts ONLY from as-aired evidence or delivery
                     metadata — README.md §8's honest
                     abstention)
  rundown.json      news mode: shot roles + stories + interstitials
  psa_manifest.json pointers + mode evidence
"""
from __future__ import annotations

import logging

from lib import acts as acts_mod
from lib import gaps as gaps_mod
from lib import markers as markers_mod
from lib import psa_common
from lib import roles as roles_mod
from lib import sequences as sequences_mod
from lib import stories as stories_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = psa_common.read_json(*psa_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    ev = _fetch(plan["inputs"])

    tm = {s["shot_id"]: (s["start_time"], s["end_time"])
          for s in ev["shots"]["shots"]}
    if not tm:
        # refuse by NAME, not with a bare max()-on-empty ValueError
        raise ValueError(
            "shots.json has no shots — nothing to assemble; check that "
            "shot_segmentation_s3uri points at a completed run")
    duration = (ev["shots"].get("actual_duration_sec")
                or max(b for _, b in tm.values()))

    marks = markers_mod.markers(ev["shot_docs"], tm, ev.get("stem"), duration)
    breaks = acts_mod.candidate_seams(marks, tm, ev.get("stem") or {},
                                      ev.get("recurrence"), duration)
    structure = {
        "workflow_id": plan["workflow_id"],
        "mode": plan["mode"],
        "markers": marks,
        "break_candidates": breaks,
        # acts[] deliberately ABSENT on masters: not mechanically
        # recoverable (measured, README.md §8); a
        # delivery-metadata input is the future source.
    }
    outputs = {}
    if ev.get("transcript") and ev.get("scenes"):
        structure["gap_map"] = gaps_mod.gap_map(
            ev["transcript"], ev["scenes"]["scenes"],
            ev["shots"]["shots"], ev.get("stem"))
        structure["orientation"] = gaps_mod.orientation(
            ev["scenes"]["scenes"], ev["transcript"], ev.get("stem"))
    if plan["mode"] == "narrative" and ev.get("scenes"):
        structure["sequences"] = sequences_mod.sequences(
            ev["scenes"]["scenes"], ev["shot_docs"], tm,
            ev.get("recurrence"), ev.get("stem"))
    if plan["mode"] == "news":
        role_map = roles_mod.shot_roles(ev["shot_docs"],
                                        ev.get("recurrence"))
        rundown = {
            "workflow_id": plan["workflow_id"],
            "stories": stories_mod.stories(ev["ca"], ev["shot_docs"], tm,
                                           ev.get("recurrence"),
                                           role_of=role_map,
                                           fences=[b["at"] for b in breaks]),
            "shot_roles": [{"shot_id": sid, **entry}
                           for sid, entry in sorted(role_map.items())],
        }
        key = f"{prefix}/rundown.json"
        psa_common.write_json(psa_common.PROXY_BUCKET, key, rundown)
        outputs["rundown"] = f"s3://{psa_common.PROXY_BUCKET}/{key}"

    key = f"{prefix}/structure.json"
    psa_common.write_json(psa_common.PROXY_BUCKET, key, structure)
    outputs["structure"] = f"s3://{psa_common.PROXY_BUCKET}/{key}"

    manifest = {"workflow_id": plan["workflow_id"],
                "tool_version": plan["tool_version"],
                "mode": plan["mode"],
                "mode_evidence": plan["mode_evidence"],
                "inputs": plan["inputs"],
                "outputs": outputs}
    psa_common.write_json(psa_common.PROXY_BUCKET,
                          f"{prefix}/psa_manifest.json", manifest)
    logger.info("assembled mode=%s outputs=%s", plan["mode"],
                sorted(outputs))
    return {"uuid": plan["workflow_id"], "outputs": outputs}


def _fetch(inputs: dict) -> dict:
    """Every sibling artifact the rollup consumes, by pointer."""
    ss_b, ss_k = psa_common.parse_s3_uri(inputs["shot_segmentation_s3uri"])
    en_b, en_k = psa_common.parse_s3_uri(inputs["shot_metadata_s3uri"])
    ev = {
        "shots": psa_common.read_json(ss_b, f"{ss_k}/embeddings/shots.json"),
        "recurrence": psa_common.read_json_if_exists(
            ss_b, f"{ss_k}/embeddings/recurrence.json"),
        "shot_docs": psa_common.read_jsonl(en_b, f"{en_k}/shot_metadata.jsonl"),
        "scenes": psa_common.read_json_if_exists(en_b, f"{en_k}/scenes.json"),
    }
    if inputs.get("conversation_analysis_s3uri"):
        ca_b, ca_k = psa_common.parse_s3_uri(
            inputs["conversation_analysis_s3uri"])
        ev["ca"] = psa_common.read_json(
            ca_b, _artifact_key(ca_k, "conversation_analysis.json"))
    if inputs.get("transcript_s3uri"):
        tr_b, tr_k = psa_common.parse_s3_uri(inputs["transcript_s3uri"])
        ev["transcript"] = psa_common.read_json(
            tr_b, _artifact_key(tr_k, "transcripts",
                                "final_transcript.json"))
    if inputs.get("audiostem_s3uri"):
        st_b, st_k = psa_common.parse_s3_uri(inputs["audiostem_s3uri"])
        ev["stem"] = psa_common.read_json_if_exists(
            st_b, _artifact_key(st_k, "stem_activity.json"))
    return ev


def _artifact_key(key: str, *parts: str) -> str:
    """One normalization for every optional pointer: tolerate the run
    root, a parent folder, or the full artifact uri — append only the
    path parts the pointer lacks. (The per-branch ad-hoc checks silently
    double-appended on folder pointers and, for the stem, silently LOST
    the artifact on a full-file pointer.) PATH-SEGMENT comparison, never
    str.endswith — a suffix test skips a segment whenever the pointer's
    last segment merely ENDS with it (found 2026-08-31 in the AD twin of
    this helper: hex run ids ending in '0' lost their '/0' folder)."""
    segs = [s for s in key.split("/") if s]
    if segs and segs[-1] == parts[-1]:
        return "/".join(segs)
    for p in parts:
        if not segs or segs[-1] != p:
            segs.append(p)
    return "/".join(segs)
