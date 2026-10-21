"""
StateFinalizeOutputs — durable outputs (README.md §6). Post-release (no
Rekognition).

  facemetadata.json    flat people[] sorted by screen time; appearances
                       first-class; track-span rollup (min 800 ms)
  facegroup.json       full detail: per-face attributes with source flags,
                       track membership, triage drops, detection plan summary
  face_clusters.json   MDS coords from the clustering similarity matrix
"""

from __future__ import annotations

import logging

import numpy as np

from lib.input_schema import names_agree
from lib.cluster_viz import build_similarity_matrix, mds_coords
from lib.input_schema import MIN_APPEARANCE_MS, get_input_config
from lib.rekognition_utils import add_usage
from lib.s3_utils import get_json, parse_s3uri, put_json

logger = logging.getLogger(__name__)


def lambda_handler(event: dict, context) -> dict:
    cfg = get_input_config(event)
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")
    meta = event["data"]["frame_meta"]

    tracks = {tr["track_id"]: tr
              for tr in get_json(out_bucket,
                                 event["data"]["tracks_key"])["tracks"]}
    clusters_doc = get_json(out_bucket, event["data"]["clusters_key"])
    search_doc = get_json(out_bucket, event["data"]["search_index_key"])
    search_by_gid = {c["face_group_id"]: c for c in search_doc["clusters"]}

    # Fail hard on a missing/zero duration — same rule as _resolve_fps: this
    # field comes from the media-processing child's probe, and a schema drift
    # there must break the run, not publish duration_sec=0 as plausible truth.
    raw_duration = event["data"]["media_processing_output"].get("duration_sec")
    duration_sec = float(raw_duration) if raw_duration is not None else 0.0
    if duration_sec <= 0:
        raise ValueError(
            f"media_processing_output.duration_sec missing or non-positive "
            f"({raw_duration!r}) — refusing to publish a fabricated duration")

    # ── people[] ──────────────────────────────────────────────────────────────
    people = []
    for cluster in clusters_doc["clusters"]:
        gid = cluster["face_group_id"]
        sc = search_by_gid.get(gid) or {}
        member_tracks = [tracks[tid] for tid in cluster["track_ids"]
                         if tid in tracks]
        if not member_tracks:
            continue

        appearances = _rollup(member_tracks)
        screen_time = sum((a["end_ms"] - a["begin_ms"]) / 1000.0
                          for a in appearances)
        name, source = _resolve_name(sc)
        # §23: a naming conflict (collection match vs validated celebrity
        # plurality) must be VISIBLE in the product output, whatever the
        # resolution. The Meridian mislabel's correct identification existed
        # only in an ephemeral checkpoint no consumer may read; this puts
        # both candidates where a consumer — or a human — will see them.
        conflict = sc.get("name_conflict")
        # Deterministic tie-break (earliest track wins) — MUST match
        # store_debug_images' rep pick, so `rep_face_key` points at the
        # face these gender/age fields were read from.
        rep_track = max(member_tracks,
                        key=lambda t: (t["appearance_count"], -t["first_ts"]))
        rep_face = rep_track["faces"][rep_track["centroid_idx"]]
        celeb_key = _celebrity_key(sc, name)   # ONE gate, both fields
        people.append({
            "person_id": gid,
            "name": name or f"Person_{gid}",
            "named": bool(name),
            "name_source": source,
            "confidence": _name_confidence(sc),
            "face_id": (sc.get("face_match") or sc.get("indexed") or {}
                        ).get("face_id"),
            # the celebrity analogue of face_id (2026-09-07): the
            # recognizer's own identity key, gated by _celebrity_key
            # (id equality first, name agreement fallback). Two people[]
            # entries sharing it are one person by the recognizer's own
            # claim; cast-resolution coalesces on it.
            "celebrity_id": celeb_key,
            **(_celebrity_info(sc) if celeb_key else {}),
            "gender": rep_face.get("gender"),
            "age_range": rep_face.get("age_range"),
            "total_screen_time_sec": round(screen_time, 2),
            "appearances": appearances,
            "thumbnail_key": f"{out_prefix}/debug/people/person_{gid:04d}.jpg",
            # Single representative face crop (3:4, native res, ≤512 long
            # edge) — the consumer-facing reference face; thumbnail_key is
            # a debug MONTAGE, not a face. Written by store_debug_images,
            # same forward-pointer pattern as thumbnail_key.
            "rep_face_key":
                f"{out_prefix}/debug/people/person_{gid:04d}_face.jpg",
            "num_tracks": len(member_tracks),
            **({"name_conflict": True,
                "name_conflict_detail": {
                    "resolution": conflict.get("resolution"),
                    "celebrity_name": conflict.get("celebrity_name"),
                    "celebrity_confidence":
                        conflict.get("celebrity_confidence"),
                    "first_witness":
                        (conflict.get("first_witness") or {}).get("celeb"),
                    "second_witness":
                        (conflict.get("second_witness") or {}).get("celeb"),
                }} if conflict else {}),
        })
    people.sort(key=lambda p: -p["total_screen_time_sec"])

    frame_index = get_json(out_bucket, meta["frame_index_key"])
    dropped = [{"frame_num": f["frame_num"], "reason": f["reason"]}
               for f in frame_index["frames"] if f.get("dropped")]

    usage = _sum_usage(get_json(out_bucket, event["data"]["tracks_key"]),
                       clusters_doc, search_doc)

    facemetadata = {
        "version": 1,
        "collection_id": cfg["collection_id"],
        "duration_sec": duration_sec,
        "people": people,
        "stats": {
            "frames_analyzed": meta["kept_frames"],
            "frames_skipped": meta["triaged_out"],
            "faces_detected": sum(t["appearance_count"]
                                  for t in tracks.values()),
            "num_tracks": len(tracks),
            "num_people": len(people),
            # Which cut evidence contained these tracks (README.md §21+§23:
            # boundaries are UNIONED with the local dHash veto, never
            # substituted, so shots mode is honestly "shots+dhash")
            "cut_detector": ("shots+dhash" if meta.get("shot_boundaries_key")
                             else "dhash"),
        },
        "usage": usage,
    }
    put_json(out_bucket, f"{out_prefix}/facemetadata.json", facemetadata)

    # ── facegroup.json (full detail) ──────────────────────────────────────────
    plan = get_json(out_bucket, event["data"]["detection_plan_key"])
    facegroup = {
        "version": 1,
        "people": [
            {**p,
             "tracks": [
                 {"track_id": tid,
                  "faces": tracks[tid]["faces"],
                  "centroid_idx": tracks[tid]["centroid_idx"]}
                 for tid in (search_by_gid.get(p["person_id"]) or {}
                             ).get("track_ids", [])
                 if tid in tracks]}
            for p in people
        ],
        "triaged_frames": dropped,
        "detection_plan": {
            "noisy_patches": plan.get("noisy_patches"),
            "skipped_gaps": len(plan.get("skipped_gaps") or []),
            "dedup_skipped": len(plan.get("dedup_skipped") or []),
            "dense_frames": len(plan.get("dense_frames") or []),
        },
    }
    put_json(out_bucket, f"{out_prefix}/facegroup.json", facegroup)

    # ── face_clusters.json (MDS) ──────────────────────────────────────────────
    pairs = clusters_doc.get("similarity_pairs") or []
    if people and pairs:
        gid_order = [p["person_id"] for p in people]
        gid_index = {g: i for i, g in enumerate(gid_order)}
        track_to_gid = {}
        for c in clusters_doc["clusters"]:
            for tid in c["track_ids"]:
                track_to_gid[tid] = c["face_group_id"]
        mapped = []
        for ta, tb, sim in pairs:
            ga, gb = track_to_gid.get(ta), track_to_gid.get(tb)
            if ga in gid_index and gb in gid_index and ga != gb:
                mapped.append((gid_index[ga], gid_index[gb], sim))
        coords = mds_coords(build_similarity_matrix(len(gid_order), mapped))
        put_json(out_bucket, f"{out_prefix}/face_clusters.json", {
            "people": [
                {"person_id": p["person_id"], "name": p["name"],
                 "coord": [round(float(coords[i][0]), 4),
                           round(float(coords[i][1]), 4)],
                 "num_tracks": p["num_tracks"],
                 "thumbnail_key": p["thumbnail_key"]}
                for i, p in enumerate(people)
            ],
        })
        event["data"]["face_clusters_key"] = f"{out_prefix}/face_clusters.json"

    event["data"]["facemetadata_key"] = f"{out_prefix}/facemetadata.json"
    event["data"]["facegroup_key"] = f"{out_prefix}/facegroup.json"
    event["data"]["num_people_recognized"] = sum(1 for p in people if p["named"])
    event["data"]["num_people_unrecognized"] = sum(
        1 for p in people if not p["named"])
    return event


def _rollup(member_tracks) -> list[dict]:
    """Track-span rollup, min MIN_APPEARANCE_MS, merged when overlapping."""
    spans = []
    for tr in member_tracks:
        begin = int(tr["first_ts"] * 1000)
        end = max(int(tr["last_ts"] * 1000), begin + MIN_APPEARANCE_MS)
        rep = tr["faces"][tr["centroid_idx"]]["frame_num"]
        spans.append({"begin_ms": begin, "end_ms": end,
                      "representative_frame": rep})
    spans.sort(key=lambda s: s["begin_ms"])
    merged = []
    for s in spans:
        if merged and s["begin_ms"] <= merged[-1]["end_ms"]:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], s["end_ms"])
        else:
            merged.append(dict(s))
    return merged


def _celebrity_key(sc: dict, name: str | None):
    """The celebrity identity key. Gate, in trust order (2026-09-07):

    1. ID EQUALITY when the matched collection record carries a
       register-time celebrity_id — equal ids keep the key whatever the
       spelling (names are not stable), different ids suppress it (a
       collection HOMONYM is a different person however the strings
       compare).
    2. Otherwise NAME AGREEMENT (names_agree — the same rule the §23
       conflict trigger uses) between the published name and the
       validated vote's name, for records that predate the field.

    Never `source == "celebrity"`: a collection witness that CONFIRMS
    the plurality names the same person and must keep the key, or
    auto-indexing progressively disables the join on every later run."""
    vote = sc.get("celebrity") or {}
    vid = vote.get("celebrity_id")
    if not vid or not name:
        return None
    record_id = ((sc.get("face_match") or {}).get("celebrity_id")
                 or (sc.get("indexed") or {}).get("celebrity_id"))
    if record_id:
        return vid if record_id == vid else None
    return vid if names_agree(vote.get("name"), name) else None


def _celebrity_info(sc: dict) -> dict:
    """celebrity_urls/known_gender from search_index_faces' pass-3
    GetCelebrityInfo lookup. The caller applies the gate (ONE
    _celebrity_key computation covers both fields, so they cannot
    disagree); ABSENT (not null) when never fetched (three-state)."""
    info = (sc.get("celebrity") or {}).get("info")
    if not info:
        return {}
    out = {"celebrity_urls": info.get("urls") or []}
    # three-state: a successful lookup without KnownGender emits NO key,
    # matching the contract's "present or absent, never null"
    if info.get("known_gender"):
        out["known_gender"] = info["known_gender"]
    return out


def _resolve_name(sc: dict) -> tuple[str | None, str | None]:
    fm = sc.get("face_match")
    if fm and fm.get("celeb"):
        return fm["celeb"], "collection"
    celeb = sc.get("celebrity")
    if celeb and celeb.get("name"):
        return celeb["name"], "celebrity"
    idx = sc.get("indexed")
    if idx and idx.get("celeb"):
        return idx["celeb"], "indexed"
    return None, None


def _name_confidence(sc: dict) -> float | None:
    fm = sc.get("face_match")
    if fm:
        return fm.get("similarity")
    celeb = sc.get("celebrity")
    if celeb:
        return celeb.get("confidence")
    return None


def _sum_usage(*docs) -> dict:
    total: dict = {}
    for doc in docs:
        add_usage(total, doc.get("usage") or {})
    return total
