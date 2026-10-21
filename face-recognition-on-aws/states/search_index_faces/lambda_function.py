"""
StateSearchIndexFaces — persistent-collection search + policy-gated
auto-index (README.md §4, §2b).

Policy gates (GDPR):
  collection_id == ""        → whole state is a no-op (search AND index off)
  max_faces_to_index == 0    → search only
  index_named_faces_only     → index candidates require a celebrity name

Per identity cluster with >= min_appearances distinct tracks:
  1. Majority-vote celebrity name (>50% of >10 datapoints & score>90, or any
     score>95).
  2. ONE best face cropped at native resolution, SearchFacesByImage on the
     persistent collection (threshold 80, accept >=85 similarity) + DDB
     batch_get → face_match. (SearchFacesByImage only searches the largest
     face in the query image, so a multi-face tile grid bought nothing.)
  3. Non-matches become index candidates (name-prioritized, policy-capped):
     IndexFaces QualityFilter HIGH, DDB registration, thumbnail/full image
     stored. Phantom faces deleted.

DDB registrations are idempotent by face_id; checkpoint/re-enter per cluster,
plus an immediate checkpoint after every IndexFaces write (permanent, billed
side effect — must never be repeated by a retry). A search match whose
FaceId has no DDB record but whose ExternalImageId embeds THIS run's uuid is
our own orphan from a crash between IndexFaces and register_face — repaired
in place instead of indexing a duplicate.
"""

from __future__ import annotations

import logging

from lib import semaphore
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.face_quality import best_faces, fusion_rank
from lib.face_store import (
    batch_get_faces,
    create_external_image_id,
    register_face,
    store_face_images,
)
from lib.grid_image import crop_face_native
from lib.input_schema import (
    names_agree,
    HEADSHOT_CROP_MARGIN,
    INDEX_CROP_MAX_EDGE_PX,
    NAME_VOTE_INSTANT_SCORE,
    NAME_VOTE_MIN_DATAPOINTS,
    NAME_VOTE_MIN_SCORE,
    PERSISTENT_INDEX_QUALITY_FILTER,
    REMAINING_TIME_BUFFER_MS,
    SEARCH_ACCEPT_SIMILARITY,
    SEARCH_BY_IMAGE_THRESHOLD,
    get_input_config,
)
from lib.rekognition_utils import (
    CELEBRITY_INFO_THREADS,
    CallCounter,
    create_collection_if_absent,
    delete_faces,
    get_celebrity_info,
    index_faces,
    search_faces_by_image,
)
from lib.s3_utils import (
    get_bytes,
    get_json,
    get_json_if_exists,
    parse_s3uri,
    put_bytes,
    put_json,
)

logger = logging.getLogger(__name__)


def lambda_handler(event: dict, context) -> dict:
    cfg = get_input_config(event)
    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")
    result_key = f"{out_prefix}/ephemeral/search_index_faces.json"

    # $$.Execution.Id injected by the ASL (see cluster_identities note).
    execution_arn = event.get("execution_arn") or ""
    if execution_arn:
        semaphore.refresh_holder(execution_arn)

    clusters_doc = get_json(out_bucket, event["data"]["clusters_key"])
    tracks = {tr["track_id"]: tr
              for tr in get_json(out_bucket,
                                 event["data"]["tracks_key"])["tracks"]}
    counter = CallCounter()

    collection_id = cfg["collection_id"]
    max_index = cfg["max_faces_to_index"]
    min_appearances = cfg["min_appearances"]

    # Only a missing checkpoint key is treated as "fresh start" — any other
    # S3 failure propagates instead of silently discarding real search/index
    # progress (re-searching/re-indexing already-billed faces, or worse,
    # re-registering a face already written to the persistent collection).
    result = get_json_if_exists(out_bucket, result_key)
    if result is not None:
        done = {c["face_group_id"] for c in result["clusters"]}
        counter.add(result.get("usage") or {})
    else:
        result = {"clusters": [], "collection_id": collection_id, "usage": {}}
        done = set()

    if collection_id:
        create_collection_if_absent(collection_id)

    frames_bucket = event["data"]["media_processing_output"]["frames_bucket"]
    frames_prefix = event["data"]["frame_meta"]["frames_prefix"]
    facestore_prefix = (f"micro-workflow/face-recognition/_facestore/"
                        f"{collection_id}") if collection_id else ""
    indexed_count = sum(1 for c in result["clusters"] if c.get("indexed"))

    # ── Pass 1: search every validated cluster against the collection ────────
    # Index candidates are only COLLECTED here — the budget is spent in pass 2
    # after ranking, so max_faces_to_index goes to the BEST unmatched groups
    # (screen time / track count / face quality), not the first ones iterated.
    for cluster in clusters_doc["clusters"]:
        gid = cluster["face_group_id"]
        if gid in done:
            continue
        if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
            result["usage"] = counter.as_dict()
            put_json(out_bucket, result_key, result)
            raise LambdaResourceTimeoutException(
                f"search/index checkpointed at cluster {gid}")

        entry: dict = {
            "face_group_id": gid,
            "track_ids": cluster["track_ids"],
            "celebrity": _vote_name(cluster, tracks),
        }
        validated = len(cluster["track_ids"]) >= min_appearances

        if collection_id and validated:
            match = _search_persistent(
                cluster, tracks, entry, event, collection_id,
                out_bucket, out_prefix,
                frames_bucket, frames_prefix, counter)
            if match:
                entry["face_match"] = match

        result["clusters"].append(entry)
        done.add(gid)

    # ── Pass 2: spend the index budget on the highest-ranked candidates ──────
    # Candidates: validated, unmatched, policy-eligible clusters. Rank by
    # screen presence and best-face quality (_index_rank). Idempotent on
    # re-enter: already-indexed entries keep their slot (gid in done skips
    # pass 1; here the "indexed" flag skips re-indexing).
    if collection_id and max_index > 0:
        by_gid = {c["face_group_id"]: c for c in clusters_doc["clusters"]}
        candidates = []
        for entry in result["clusters"]:
            if entry.get("face_match") or entry.get("indexed"):
                continue
            if len(entry["track_ids"]) < min_appearances:
                continue
            if cfg["index_named_faces_only"] and not entry["celebrity"]:
                continue
            cluster = by_gid.get(entry["face_group_id"])
            if cluster is None:
                continue
            candidates.append((
                _index_rank(cluster, tracks), entry["face_group_id"], entry))
        candidates.sort(key=lambda c: -c[0])

        for _, gid, entry in candidates:
            if indexed_count >= max_index:
                break
            if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
                result["usage"] = counter.as_dict()
                put_json(out_bucket, result_key, result)
                raise LambdaResourceTimeoutException(
                    f"index budget checkpointed at cluster {gid}")
            indexed = _index_persistent(
                event, by_gid[gid], tracks, entry, collection_id,
                out_bucket, facestore_prefix,
                frames_bucket, frames_prefix, counter)
            if indexed:
                entry["indexed"] = indexed
                indexed_count += 1
                # Checkpoint IMMEDIATELY after any cluster that wrote to the
                # persistent collection — IndexFaces is a permanent, billed
                # side effect that cannot be rolled back by a retry. Without
                # this, a crash later in the loop re-enters from the previous
                # checkpoint and re-indexes this person as a duplicate face.
                # Bounded cost: at most max_faces_to_index (default 0 — opt-in)
                # extra S3 PUTs per run.
                result["usage"] = counter.as_dict()
                put_json(out_bucket, result_key, result)

    # ── Pass 3: canonical celebrity info (GetCelebrityInfo) ──────────────
    # No per-call charge; the constraint is the API's 50 TPS (user,
    # 2026-09-07) — ONE call per DISTINCT validated celebrity id, fanned
    # CELEBRITY_INFO_THREADS wide. Failures degrade to absent info
    # (three-state; nothing downstream depends on it), and re-enter skips
    # tags that already carry it. Lives HERE, not finalize — finalize is
    # post-release by design and makes no Rekognition calls.
    pending: dict[str, list] = {}
    for entry in result["clusters"]:
        tag = entry.get("celebrity") or {}
        cid = tag.get("celebrity_id")
        if cid and "info" not in tag:
            pending.setdefault(cid, []).append(tag)
    if pending:
        from concurrent.futures import ThreadPoolExecutor

        def _fetch(cid):
            # the throwaway counter satisfies the wrapper's signature;
            # success == exactly one call, counted in the MAIN thread
            # below (CallCounter increments are not thread-safe)
            try:
                return cid, get_celebrity_info(cid, CallCounter())
            except Exception as exc:              # noqa: BLE001 — degrade
                logger.warning("get_celebrity_info(%s) failed: %s",
                               cid, str(exc)[:150])
                return cid, None
        # time-guarded waves, like passes 1-2: a throttled call can burn
        # ~10s in _run's explicit retries, so an unguarded fan-out could
        # ride into a HARD timeout — losing (and re-billing) every
        # pass-1 search since the last checkpoint. The checkpoint below
        # is also what makes the "info" re-enter skip above REAL: it
        # carries the partial info fetched so far.
        ids = list(pending)
        with ThreadPoolExecutor(max_workers=CELEBRITY_INFO_THREADS) as pool:
            for wi in range(0, len(ids), CELEBRITY_INFO_THREADS):
                if context.get_remaining_time_in_millis() \
                        < REMAINING_TIME_BUFFER_MS:
                    result["usage"] = counter.as_dict()
                    put_json(out_bucket, result_key, result)
                    raise LambdaResourceTimeoutException(
                        "celebrity info checkpointed")
                wave = ids[wi:wi + CELEBRITY_INFO_THREADS]
                for cid, info in pool.map(_fetch, wave):
                    if info:
                        counter.get_celebrity_info_api += 1
                        for tag in pending[cid]:
                            tag["info"] = info

    result["usage"] = counter.as_dict()
    put_json(out_bucket, result_key, result)
    event["data"]["search_index_key"] = result_key
    event["data"]["num_faces_indexed"] = indexed_count
    logger.info("search/index done: %d clusters, %d indexed",
                len(result["clusters"]), indexed_count)
    return event


def _vote_name(cluster: dict, tracks: dict) -> dict | None:
    """Majority-vote across per-face celebrity confidences in the cluster.
    The cluster carries its best tag from clustering; validate it against the
    vote thresholds using track appearance counts as datapoints."""
    tag = cluster.get("celebrity")
    if not tag or not tag.get("name"):
        return None
    datapoints = sum(tracks[tid]["appearance_count"]
                     for tid in cluster["track_ids"] if tid in tracks)
    score = float(tag.get("confidence") or 0)
    if score >= NAME_VOTE_INSTANT_SCORE:
        return tag
    if datapoints > NAME_VOTE_MIN_DATAPOINTS and score > NAME_VOTE_MIN_SCORE:
        # No fraction test here: _cluster_celebrity already returns the
        # PLURALITY name with its MEAN confidence, so dominance is enforced
        # upstream where the per-track votes are visible.
        return tag
    return None


def _best_cluster_faces(cluster, tracks, limit):
    faces = []
    for tid in cluster["track_ids"]:
        tr = tracks.get(tid)
        if tr:
            faces.extend(f for f in tr["faces"]
                         if f.get("source") != "interpolated")
    idx = best_faces(faces, limit)
    return [faces[i] for i in idx]


def _index_rank(cluster, tracks) -> float:
    """
    Ranking score for the proactive-index budget (use case: the
    max_faces_to_index slots should go to the people who matter most for
    later tagging, not the first clusters in iteration order).

        screen_time_sec × sqrt(num_tracks) × best-face fusion_rank

    - screen time is the primary signal of editorial importance;
    - sqrt(num_tracks) rewards recurrence across scenes without letting
      hundreds of fragmented micro-tracks dominate a person with fewer,
      longer tracks;
    - the best candidate face's fusion_rank (area × frontal-ness, after
      best_faces' pose/occlusion/eyes-open gates) demotes groups whose
      best available crop would make a poor collection template.
    A group with NO gate-passing face still ranks (best_faces falls back),
    but its low fusion_rank naturally sinks it.
    """
    member = [tracks[tid] for tid in cluster["track_ids"] if tid in tracks]
    if not member:
        return 0.0
    screen_time = sum(t["last_ts"] - t["first_ts"] for t in member)
    chosen = _best_cluster_faces(cluster, tracks, 1)
    quality = fusion_rank(chosen[0]) if chosen else 0.0
    return max(screen_time, 0.1) * (len(member) ** 0.5) * max(quality, 1e-6)


def _best_face_and_track(cluster, tracks, exclude_track=None):
    """Best gate-passing face in the cluster with its owner track id;
    optionally excluding one track. The exclusion is the load-bearing part
    of the §23 corroboration: a contaminated track holds ALL the foreign
    faces (they entered through one bridged cut), so a witness drawn from
    any OTHER track is guaranteed to be a different sample of the cluster's
    real person. Track-level tag filtering cannot do this — the Meridian
    incident's contaminated track carried the majority's own tag."""
    faces, owners = [], []
    for tid in cluster["track_ids"]:
        if tid == exclude_track:
            continue
        tr = tracks.get(tid)
        if tr:
            for f in tr["faces"]:
                if f.get("source") != "interpolated":
                    faces.append(f)
                    owners.append(tid)
    idx = best_faces(faces, 1)
    if not idx:
        return None
    return faces[idx[0]], owners[idx[0]]


def _search_persistent(cluster, tracks, entry, event, collection_id,
                       out_bucket, out_prefix,
                       frames_bucket, frames_prefix, counter) -> dict | None:
    # ONE best face, at native resolution. SearchFacesByImage only ever
    # searches the LARGEST face in the query image — the old 16-tile grid
    # paid 16 serial S3 GETs + 16 crops + a compose per cluster for an API
    # that used exactly one (arbitrary, since equal-size tiles tie on
    # "largest") tile. Single native-res crop: deterministic query face,
    # ~16x less prep I/O, better match quality than a 180x240 shrink.
    chosen = _best_face_and_track(cluster, tracks)
    if not chosen:
        return None
    face, w1_track = chosen
    match = _search_one_face(face, cluster, "", event, collection_id,
                             out_bucket, out_prefix,
                             frames_bucket, frames_prefix, counter, entry)
    if not match:
        return None

    # §23 corroboration: the query face is chosen by PHOTO QUALITY alone, so
    # a cluster contaminated by a single bridged cut can put a foreign face
    # forward as its identity witness (the Meridian mislabel: actor_M2's
    # bright close-up out-ranked every genuine actor_M1 face and renamed
    # the whole 207-second person at similarity 99.95). ONLY when the match
    # name disagrees with a validated celebrity plurality, spend ONE more
    # SearchFacesByImage on the best face from a DIFFERENT track:
    #   - second witness agrees with the first  -> the plurality is the noisy
    #     one (the §15 `news_b` scenario); keep the collection name.
    #   - second witness sides with the plurality -> witness 1 was
    #     contaminated; return witness 2's match (name_source stays
    #     "collection" — both candidates ARE collection answers).
    #   - anything else -> keep witness 1, flag the conflict unresolved.
    # No-conflict clusters (the overwhelming majority) spend nothing extra.
    vote = entry.get("celebrity") or {}
    plurality = vote.get("name")
    if not plurality or not match.get("celeb"):
        return match
    # identity keys outrank spelling: a collection record carrying the
    # SAME celebrity id as the vote is the same person by the
    # recognizer's own assertion — a name variant is not a conflict and
    # spends no witness call (register-time ids, 2026-09-07)
    if match.get("celebrity_id") and vote.get("celebrity_id") \
            and match["celebrity_id"] == vote["celebrity_id"]:
        return match
    if names_agree(match["celeb"], plurality):
        return match

    conflict = {"celebrity_name": plurality,
                "celebrity_confidence":
                    (entry.get("celebrity") or {}).get("confidence"),
                "first_witness": dict(match)}
    second = _best_face_and_track(cluster, tracks, exclude_track=w1_track)
    if second is None:
        conflict["resolution"] = "unresolved_single_track"
        entry["name_conflict"] = conflict
        return match
    face2, _ = second
    match2 = _search_one_face(face2, cluster, "_w2", event, collection_id,
                              out_bucket, out_prefix,
                              frames_bucket, frames_prefix, counter, entry)
    if match2 and names_agree(match2.get("celeb"), match["celeb"]):
        conflict["resolution"] = "witnesses_agree_collection_wins"
        entry["name_conflict"] = conflict
        return match
    if match2 and names_agree(match2.get("celeb"), plurality):
        conflict["resolution"] = "second_witness_sided_with_plurality"
        conflict["second_witness"] = dict(match2)
        entry["name_conflict"] = conflict
        return match2
    conflict["resolution"] = "unresolved"
    if match2:
        conflict["second_witness"] = dict(match2)
    entry["name_conflict"] = conflict
    return match


def _search_one_face(face, cluster, key_suffix, event, collection_id,
                     out_bucket, out_prefix,
                     frames_bucket, frames_prefix, counter,
                     entry) -> dict | None:
    jpeg = get_bytes(frames_bucket, f"{frames_prefix}/{face['frame_key']}")
    query = crop_face_native(jpeg, face["box"], HEADSHOT_CROP_MARGIN,
                             INDEX_CROP_MAX_EDGE_PX)
    grid_key = (f"{out_prefix}/grids/search_"
                f"{cluster['face_group_id']:04d}{key_suffix}.jpg")
    put_bytes(out_bucket, grid_key, query, "image/jpeg")

    matches = search_faces_by_image(collection_id, out_bucket, grid_key,
                                    SEARCH_BY_IMAGE_THRESHOLD, counter)
    for m in matches:
        if float(m.get("Similarity") or 0) < SEARCH_ACCEPT_SIMILARITY:
            continue
        mface = m.get("Face") or {}
        fid = mface.get("FaceId")
        if not fid:
            continue
        record = batch_get_faces([fid]).get(fid)
        if record:
            return {"face_id": fid,
                    "celeb": record.get("celeb"),
                    # register-time identity key (2026-09-07); absent on
                    # records that predate the field — finalize's gate
                    # falls back to name agreement for those
                    "celebrity_id": record.get("celebrity_id"),
                    "collection_id": collection_id,
                    "similarity": round(float(m["Similarity"]), 2)}
        # No DDB record: either an operator-managed face (leave alone) or OUR
        # OWN orphan — IndexFaces succeeded on a prior attempt of THIS run
        # but the crash landed before register_face. The ExternalImageId
        # embeds the workflow uuid, so orphans are self-identifying. Repair
        # (register now with this cluster's vote) instead of falling through
        # to _index_persistent, which would add a duplicate face for the
        # same person and leave the orphan matching future runs forever.
        ext_id = mface.get("ExternalImageId") or ""
        own_hex = event["uuid"].replace("-", "")[:32].ljust(32, "0")
        if ext_id.split(":")[1:2] == [own_hex]:
            vote = entry.get("celebrity") or {}
            celeb = vote.get("name")
            logger.warning("repairing own orphaned face %s (indexed on a "
                           "prior attempt, never registered)", fid)
            register_face(fid, {
                "collection_id": collection_id,
                "external_image_id": ext_id,
                "celeb": celeb,
                "celebrity_id": vote.get("celebrity_id"),
                "origin": "auto",
                "workflow_id": event["uuid"],
            })
            return {"face_id": fid, "celeb": celeb,
                    "celebrity_id": vote.get("celebrity_id"),
                    "collection_id": collection_id,
                    "similarity": round(float(m["Similarity"]), 2)}
    return None


def _index_persistent(event, cluster, tracks, entry, collection_id,
                      out_bucket, facestore_prefix,
                      frames_bucket, frames_prefix, counter) -> dict | None:
    chosen = _best_cluster_faces(cluster, tracks, 1)
    if not chosen:
        return None
    face = chosen[0]
    frame_jpeg = get_bytes(frames_bucket,
                           f"{frames_prefix}/{face['frame_key']}")
    # Native resolution for the index image — never shrink to a tile size.
    headshot = crop_face_native(frame_jpeg, face["box"], HEADSHOT_CROP_MARGIN,
                                INDEX_CROP_MAX_EDGE_PX)
    key = (f"{facestore_prefix}/_staging/"
           f"{event['uuid']}_{cluster['face_group_id']:04d}.jpg")
    put_bytes(out_bucket, key, headshot, "image/jpeg")

    external_id = create_external_image_id(event["uuid"])
    records = index_faces(collection_id, out_bucket, key, external_id,
                          max_faces=1,
                          quality_filter=PERSISTENT_INDEX_QUALITY_FILTER,
                          counter=counter)
    if not records:
        return None
    if len(records) > 1:  # phantoms — keep the largest, delete the rest
        records.sort(key=lambda r: -(r["Face"]["BoundingBox"]["Width"]
                                     * r["Face"]["BoundingBox"]["Height"]))
        delete_faces(collection_id,
                     [r["Face"]["FaceId"] for r in records[1:]])
    face_id = records[0]["Face"]["FaceId"]

    thumb_key, full_key = store_face_images(
        out_bucket, facestore_prefix, face_id, frame_jpeg, face["box"])
    vote = entry.get("celebrity") or {}
    celeb = vote.get("name")
    register_face(face_id, {
        "collection_id": collection_id,
        "external_image_id": external_id,
        "celeb": celeb,
        # the recognizer's identity key rides the record (2026-09-07) so
        # future runs can gate celebrity_id on ID EQUALITY instead of
        # the name string — names are neither unique nor stable
        "celebrity_id": vote.get("celebrity_id"),
        "origin": "auto",
        "gender": face.get("gender"),
        "age_range": face.get("age_range"),
        "confidence": face.get("confidence"),
        "coord": [face["box"]["l"], face["box"]["t"],
                  face["box"]["w"], face["box"]["h"]],
        "thumbnail_key": thumb_key,
        "full_image_key": full_key,
        "workflow_id": event["uuid"],
    })
    return {"face_id": face_id, "celeb": celeb,
            "celebrity_id": vote.get("celebrity_id")}


