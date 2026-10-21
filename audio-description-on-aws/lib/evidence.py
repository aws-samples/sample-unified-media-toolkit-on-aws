"""
Sibling-artifact fetch, by pointer. ONE tolerant normalization for every
pointer (run root, parent folder, or full artifact uri — the
per-branch ad-hoc checks in an earlier sibling silently double-appended
on folder pointers and silently LOST optional artifacts on full-file
pointers; found in the 2026-08-31 code review).
"""
from __future__ import annotations

from lib import ad_common as C


def artifact_key(key: str, *parts: str) -> str:
    """PATH-SEGMENT comparison, never str.endswith (review 2026-08-31:
    the suffix test skipped the '0' folder whenever a hex workflow id
    happened to END in '0' — ~1/16 of runs silently lost their credits,
    the exact silent-loss class this helper was written to end)."""
    segs = [s for s in key.split("/") if s]
    if segs and segs[-1] == parts[-1]:
        return "/".join(segs)
    for p in parts:
        if not segs or segs[-1] != p:
            segs.append(p)
    return "/".join(segs)


def fetch(inputs: dict, only: set | None = None) -> dict:
    """Sibling artifacts, parsed; `only` names the keys a caller needs
    (the repair loop re-enters ~25x per run and must not re-download
    conversation_analysis/facemetadata it never reads — review
    2026-08-31). Optional artifacts are None when absent (stem: degrade
    to no music gating; credits: the tail oracle guesses from
    texts_on_screen; decisions: nothing accepted yet). Cast-resolution bindings are NOT fetched here — validate_plan reads persons.json once and forwards the filtered payload in plan.json (cast_bindings)."""
    def want(name):
        return only is None or name in only

    ev = {"shots": None, "shot_docs": None, "scenes": None,
          "transcript": None, "ca": None, "people": None,
          "people_basis": "face",
          "stem_activity": None, "credits": None, "decisions": None}
    if want("shots"):
        ss_b, ss_k = C.parse_s3_uri(inputs["shot_segmentation_s3uri"])
        ev["shots"] = C.read_json(
            ss_b, artifact_key(ss_k, "embeddings", "shots.json"))["shots"]
    if want("shot_docs"):
        en_b, en_k = C.parse_s3_uri(inputs["shot_metadata_s3uri"])
        ev["shot_docs"] = C.read_jsonl(
            en_b, artifact_key(en_k, "shot_metadata.jsonl"))
    if want("scenes"):
        en_b, en_k = C.parse_s3_uri(inputs["shot_metadata_s3uri"])
        ev["scenes"] = C.read_json(
            en_b, artifact_key(en_k, "scenes.json"))["scenes"]
    if want("transcript"):
        di_b, di_k = C.parse_s3_uri(inputs["speech_diarization_s3uri"])
        ev["transcript"] = C.read_json(
            di_b, artifact_key(di_k, "transcripts",
                               "final_transcript.json"))
    if want("ca"):
        ca_b, ca_k = C.parse_s3_uri(inputs["conversation_analysis_s3uri"])
        ev["ca"] = C.read_json(
            ca_b, artifact_key(ca_k, "conversation_analysis.json"))
    if want("people"):
        if inputs.get("face_recognition_s3uri"):
            fr_b, fr_k = C.parse_s3_uri(inputs["face_recognition_s3uri"])
            ev["people"] = C.read_json(
                fr_b, artifact_key(fr_k, "facemetadata.json"))["people"]
        else:
            # cast-resolution as the SOLE people source (2026-09-08):
            # its persons carry consolidated appearance timelines with
            # a PRODUCER-declared basis — `face` (screen spans) or
            # `speech` (merged speaker turn spans, the honest degrade
            # on a faceless run; windows' people_on_screen then goes
            # near-empty by design, since narration windows are speech
            # gaps). Synthesize the facemetadata-people shape every
            # consumer already reads; the basis flag tells the one
            # consumer that renders identity keys (_cast_block) how.
            cr_b, cr_k = C.parse_s3_uri(inputs["cast_resolution_s3uri"])
            doc = C.read_json(cr_b, artifact_key(cr_k, "persons.json"))
            persons = doc.get("persons") or []
            ev["people_basis"] = (
                "cast_speech"
                if any(pr.get("appearances_basis") == "speech"
                       for pr in persons) else "cast")
            ev["people"] = [
                {"person_id": pr["person_id"],
                 "name": pr.get("name") or f"Person_{pr['person_id']}",
                 "named": bool(pr.get("name")),
                 "appearances": pr["appearances"],
                 "total_screen_time_sec": pr.get("screen_sec") or 0.0}
                for pr in persons
                if pr.get("appearances")]
    if want("stem_activity") and inputs.get("audiostem_s3uri"):
        st_b, st_k = C.parse_s3_uri(inputs["audiostem_s3uri"])
        ev["stem_activity"] = C.read_json_if_exists(
            st_b, artifact_key(st_k, "stem_activity.json"))
    if (only is not None and "credits" in only
            and inputs.get("cast_resolution_s3uri")):
        # OPT-IN, never part of the default set: only the repair chain
        # reads ev["credits"], and it already passes an `only` set —
        # loading it on every full fetch re-created the re-download
        # class `only` exists to prevent. The product is the cast run's
        # conditional `credits` output (frame-inspection is internal;
        # replaced the direct frame_inspection_s3uri input, user
        # decision 2026-09-08). Absent = no credits read happened; the
        # tail oracle degrades to texts_on_screen as ever.
        cr_b, cr_k = C.parse_s3_uri(inputs["cast_resolution_s3uri"])
        ev["credits"] = C.read_json_if_exists(
            cr_b, artifact_key(cr_k, "credits.json"))
    elif (only is not None and "credits" in only
            and inputs.get("frame_inspection_s3uri")):
        # LEGACY plans only: the input was retired 2026-09-08 (the
        # gateway refuses it on new submissions), but redrive re-enters
        # mid-run with the ORIGINAL plan.json — a pre-deploy plan
        # carrying this pointer must not silently lose evidence it
        # already paid for. Remove with the 90-day execution retention.
        fi_b, fi_k = C.parse_s3_uri(inputs["frame_inspection_s3uri"])
        ev["credits"] = C.read_json_if_exists(
            fi_b, artifact_key(fi_k, "0", "credits.json"))
    if want("decisions") and inputs.get("decisions_s3uri"):
        de_b, de_k = C.parse_s3_uri(inputs["decisions_s3uri"])
        ev["decisions"] = C.read_json_if_exists(
            de_b, artifact_key(de_k, "decisions.json"))
    return ev


def docs_by_id(shot_docs: list) -> dict:
    return {d["shot_id"]: d for d in shot_docs}


def shot_times(shots: list) -> dict:
    return {s["shot_id"]: (s["start_time"], s["end_time"]) for s in shots}


def ad_details(d: dict) -> list:
    """The enrichment fields added for AD (2026-08-28, reference-corpus diff):
    physical business, readable faces, camera moves — 70% of the measured
    salience gap. ONE compact renderer shared by the writer's shot lines
    and the watch-through record, so the two prompts can never disagree
    about the evidence."""
    out = []
    acts = [a for a in (d.get("actions") or []) if isinstance(a, str)]
    if acts:
        out.append("does: " + "; ".join(acts))
    ex = [e for e in (d.get("expressions") or [])
          if isinstance(e, dict) and e.get("who") and e.get("shows")]
    if ex:
        out.append("faces: " + "; ".join(f"{e['who']} — {e['shows']}"
                                         for e in ex))
    cm = d.get("camera_motion")
    if isinstance(cm, dict) and cm.get("kind"):
        out.append("camera: " + cm["kind"]
                   + (f" revealing {cm['reveals']}"
                      if cm.get("reveals") else ""))
    sc = d.get("state_change")
    if isinstance(sc, dict) and sc.get("what"):
        out.append(f"changes ({sc.get('when', 'mid')}): {sc['what']}")
    return out


def cast_provenance_error(inputs: dict, persons_doc: dict) -> str | None:
    """Bindings are only valid against the EXACT sibling runs they were
    computed from (per-run labels). Pins on the producer's documented
    consumer key, provenance.sibling_workflow_ids; no pin = no claim =
    pass. Rationale and rejected alternatives: README.md Design
    record #6e. Pure:
    validate_plan supplies the downloaded persons.json."""
    wids = ((persons_doc or {}).get("provenance") or {}
            ).get("sibling_workflow_ids") or {}
    for src_key, input_name in (
            ("face_recognition", "face_recognition_s3uri"),
            ("speech_diarization", "speech_diarization_s3uri")):
        wid = wids.get(src_key)
        ours = str(inputs.get(input_name) or "")
        if wid and ours \
                and str(wid) not in ours.rstrip("/").split("/"):
            return (f"the cast_resolution run was computed from "
                    f"{src_key} workflow {wid}, which is not the run "
                    f"this request points at ({ours}) — speaker and "
                    "face ids are per-run labels, so bindings from a "
                    "different run would mislabel every id; point both "
                    "at the same sibling runs")
    return None


def filter_cast_bindings(persons_doc: dict, cap: int) -> list[dict]:
    """The name-map payload: NAMED persons only, top `cap` by
    screen+speech time, stripped to the fields the record renders —
    evidence rows and unnamed silent faces stay behind. Pure; runs ONCE
    in validate_plan, and the result rides plan.json (cast_bindings)."""
    persons = [p for p in (persons_doc or {}).get("persons") or []
               if p.get("name")]
    persons.sort(key=lambda p: -((p.get("screen_sec") or 0)
                                 + (p.get("speech_sec") or 0)))
    return [{k: p.get(k) for k in
             ("name", "type", "role", "character_name", "aliases",
              "speaker_ids", "face_ids", "name_mechanisms")}
            for p in persons[:cap]]
