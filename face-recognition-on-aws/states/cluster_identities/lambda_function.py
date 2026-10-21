"""
StateClusterIdentities — cross-shot identity clustering via a transient
sandbox collection (README.md §4).

Sandbox "sandbox-{execution_name}" — idempotent per execution (workflow_id IS
the execution name; SFN name-uniqueness guarantees one live execution per id):
create-if-absent, reuse on re-enter. NO in-state deletion — cleanup is
exclusively the EventBridge terminal handler.

Flow: group tracks by celebrity name (or singleton) → one qualified centroid
per group → IndexFaces grids (≤100/call, QualityFilter MEDIUM) → SearchFaces
per indexed faceId (threshold 75, skip-if-strongly-grouped) → TWO-TIER
union-merge → face_group_id. Pairwise similarities captured for the MDS
visualization (zero extra API calls).

Two-tier merge (real contamination found on real content): a single
SearchFaces edge merges two groups only at >= SANDBOX_MERGE_SIMILARITY (90);
weaker edges [75, 90) merge only when >= SANDBOX_WEAK_CORROBORATION (2)
independent weak edges connect the same two components. Union-find is
transitive and irreversible, so under the old single-edge-at-75 rule one
noise-level match could chain two different people into one cluster — traced
on a real broadcast: an unrelated man reached a 289-track cluster through
75.0-77 similarity links stitched between genuine 97-100 links, and on two
episodic runs two different actors' clusters (each with hundreds of
same-name RecognizeCelebrities votes at 98-99 mean confidence) fused the
same way. 89% of genuine same-footage edges score >= 95; every observed
cross-person bridge lived in the 75-85 band. One borderline match is noise;
two independent borderline matches between the same two groups is signal.

Checkpoint/re-enter: indexed-face map + union-find state persist to the
ephemeral JSON; re-entry resumes without re-indexing.
"""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from lib import semaphore
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.face_quality import fusion_rank
from lib.grid_image import (compose_tile_grid, crop_headshot,
                            dominant_per_tile)
from lib.input_schema import (
    CENTROIDS_PER_PERMUTATION_BATCH,
    MIN_SUBJECT_TILE_SHARE,
    CHECKPOINT_EVERY_N_SEARCHES,
    HEADSHOT_CROP_MARGIN,
    REMAINING_TIME_BUFFER_MS,
    SANDBOX_INDEX_QUALITY_FILTER,
    SANDBOX_MERGE_SIMILARITY,
    SANDBOX_SEARCH_MAX_FACES,
    SANDBOX_SEARCH_THRESHOLD,
    SANDBOX_WEAK_CORROBORATION,
)
from lib.rekognition_utils import (
    CallCounter,
    add_usage,
    create_collection_if_absent,
    index_faces,
    search_faces,
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

_TILE_W, _TILE_H = 180, 240
_TILE_COLS = 10


def lambda_handler(event: dict, context) -> dict:
    # $$.Execution.Id injected by the ASL — the real ARN, not a
    # reconstruction (the old STATE_MACHINE_ARN + uuid string-rebuild
    # silently produced a wrong holder key whenever uuid != execution name).
    execution_arn = event.get("execution_arn") or ""
    if execution_arn:
        semaphore.refresh_holder(execution_arn)

    out_bucket, out_prefix = parse_s3uri(event["output"]["s3uri"])
    out_prefix = out_prefix.rstrip("/")
    state_key = f"{out_prefix}/ephemeral/cluster_state.json"
    clusters_key = f"{out_prefix}/ephemeral/identity_clusters.json"

    tracks_doc = get_json(out_bucket, event["data"]["tracks_key"])
    all_tracks = tracks_doc["tracks"]
    celeb_by_track, recognize_usage = _collect_celebrity_tags(event, out_bucket)

    counter = CallCounter()
    sandbox = f"sandbox-{event['uuid']}"

    # ── Resume or fresh ───────────────────────────────────────────────────────
    # Only a missing checkpoint key is treated as "fresh start" — any other
    # S3 failure propagates instead of silently discarding real clustering
    # progress (re-indexing/re-searching already-billed faces).
    state = get_json_if_exists(out_bucket, state_key)
    if state is not None:
        counter.add(state.get("usage") or {})
        logger.info("resuming clustering: %d faces indexed",
                    len(state.get("face_to_track") or {}))
    else:
        state = {"face_to_track": {}, "searched": [], "pairs": [],
                 "parent": {}, "weak_edges": [], "usage": {}}
        # The recognize iterators' recognize_celebrities_api counts are ONLY
        # reachable through event["data"]["iterators"], which this state
        # clears (_finish sets it to None) before usage_tracking runs — sum
        # them into THIS state's counter now, or they are permanently lost
        # (this was the bug: usage_tracking reported
        # recognize_celebrities_api=0 despite real RecognizeCelebrities
        # calls having been made and billed). recognize_usage reflects a
        # FIXED set of already-completed calls (event["data"]["iterators"]
        # does not change across retries of this state), so it is added
        # exactly once here on the FIRST attempt — never again on resume,
        # where it would already be baked into state["usage"] above and
        # double-count.
        counter.add(recognize_usage)

    if not all_tracks:
        _finish(event, out_bucket, clusters_key, [], [], counter)
        return event

    create_collection_if_absent(sandbox)

    # One qualified centroid per name-group (or per track when unnamed).
    # rep_tid is computed ONCE per group and reused for both the sandbox
    # reps AND the cluster expansion below — two independent max() calls
    # previously had to stay in lockstep or a checkpointed face_to_track
    # would desynchronize from group membership.
    groups = _name_groups(all_tracks, celeb_by_track)
    reps = []  # [(track_id, face)]
    group_members: dict[int, list[int]] = {}  # rep_tid -> group track_ids
    for track_ids in groups:
        best_tid = max(track_ids, key=lambda tid: fusion_rank(
            all_tracks[tid]["faces"][all_tracks[tid]["centroid_idx"]]))
        tr = all_tracks[best_tid]
        reps.append((best_tid, tr["faces"][tr["centroid_idx"]]))
        group_members[best_tid] = track_ids

    frames_bucket = event["data"]["media_processing_output"]["frames_bucket"]
    frames_prefix = event["data"]["frame_meta"]["frames_prefix"]
    grids_prefix = f"{out_prefix}/grids"

    # ── Index reps into the sandbox (grid batches ≤100) ──────────────────────
    face_to_track: dict[str, int] = {
        k: int(v) for k, v in (state["face_to_track"] or {}).items()}
    indexed_tids = set(face_to_track.values())
    todo = [(tid, face) for tid, face in reps if tid not in indexed_tids]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for bi in range(0, len(todo), CENTROIDS_PER_PERMUTATION_BATCH):
            if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
                _checkpoint(out_bucket, state_key, face_to_track, state, counter)
                raise LambdaResourceTimeoutException("cluster index checkpointed")
            batch = todo[bi:bi + CENTROIDS_PER_PERMUTATION_BATCH]

            def _tile(item):
                _, face = item
                jpeg = get_bytes(frames_bucket,
                                 f"{frames_prefix}/{face['frame_key']}")
                return crop_headshot(jpeg, face["box"], HEADSHOT_CROP_MARGIN,
                                     _TILE_W, _TILE_H)

            tiles = list(pool.map(_tile, batch))
            grid_jpeg, coords = compose_tile_grid(tiles, _TILE_W, _TILE_H,
                                                  _TILE_COLS)
            grid_key = f"{grids_prefix}/sandbox_{bi // 100:04d}.jpg"
            put_bytes(out_bucket, grid_key, grid_jpeg, "image/jpeg")
            records = index_faces(
                sandbox, out_bucket, grid_key,
                external_image_id=event["uuid"].replace("-", "")[:32] or "x",
                max_faces=CENTROIDS_PER_PERMUTATION_BATCH,
                quality_filter=SANDBOX_INDEX_QUALITY_FILTER,
                counter=counter)
            # ONE face per tile — the dominant one. A background face in
            # a headshot crop otherwise enters the sandbox wearing this
            # track's id, and any match against it drags the track into
            # a foreign cluster (the two-clusters-per-person shape).
            by_tile = dominant_per_tile(
                ((rec.get("Face") or {}, rec) for rec in records), coords,
                min_share=MIN_SUBJECT_TILE_SHARE)
            for idx, rec in by_tile.items():
                if idx < len(batch):
                    face_to_track[rec["Face"]["FaceId"]] = batch[idx][0]
            _checkpoint(out_bucket, state_key, face_to_track, state, counter)

    # ── SearchFaces two-tier union-merge (skip-if-strongly-grouped) ───────────
    parent: dict[str, str] = dict(state["parent"] or {})

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    searched = set(state["searched"] or [])
    pairs: list = list(state["pairs"] or [])
    weak_edges: list = list(state.get("weak_edges") or [])
    face_ids = list(face_to_track)
    since_checkpoint = 0
    for fid in face_ids:
        if fid in searched or find(fid) != fid:
            # Already searched, or already merged via a STRONG edge (only
            # strong edges union during this loop). A weakly-linked face
            # still gets its own search — its weak edges need independent
            # corroboration below, not a free ride on skip-if-grouped.
            continue
        if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
            state.update(searched=sorted(searched), pairs=pairs,
                         parent=parent, weak_edges=weak_edges)
            _checkpoint(out_bucket, state_key, face_to_track, state, counter)
            raise LambdaResourceTimeoutException("cluster search checkpointed")
        matches = search_faces(sandbox, fid, SANDBOX_SEARCH_THRESHOLD,
                               SANDBOX_SEARCH_MAX_FACES, counter)
        searched.add(fid)
        for m in matches:
            mid = (m.get("Face") or {}).get("FaceId")
            sim = float(m.get("Similarity") or 0)
            if mid and mid in face_to_track:
                if sim >= SANDBOX_MERGE_SIMILARITY:
                    union(fid, mid)
                else:
                    weak_edges.append([fid, mid, sim])
                pairs.append([face_to_track[fid], face_to_track[mid], sim])
        since_checkpoint += 1
        if since_checkpoint >= CHECKPOINT_EVERY_N_SEARCHES:
            # Periodic checkpoint: a NON-timeout crash (throttle escaping
            # the retry layers) otherwise discards every SearchFaces call
            # since the last write — the ASL retry re-enters and RE-BILLS
            # them all, and the lost calls never reach state["usage"].
            state.update(searched=sorted(searched), pairs=pairs,
                         parent=parent, weak_edges=weak_edges)
            _checkpoint(out_bucket, state_key, face_to_track, state, counter)
            since_checkpoint = 0

    _merge_corroborated_weak_edges(weak_edges, find, union)

    # ── Build clusters ────────────────────────────────────────────────────────
    roots: dict[str, list[int]] = {}
    for fid, tid in face_to_track.items():
        roots.setdefault(find(fid), []).append(tid)
    # Expand: every track in a rep's name-group joins the rep's cluster
    # (direct dict lookup — the old inverted scan was O(members × tracks)).
    clusters = []
    for members in roots.values():
        track_set = set()
        for rep_tid in members:
            track_set.update(group_members.get(rep_tid, [rep_tid]))
        clusters.append(sorted(track_set))
    # A group whose rep never made it into the sandbox (quality filter
    # dropped the face, or the tile's only detection was background and
    # the subject-share floor rejected it) previously VANISHED from the
    # output entirely. Ship it as its own unmerged cluster instead —
    # abstaining from merges is safe, silently losing a person is not.
    covered = set(face_to_track.values())
    for rep_tid, track_ids in group_members.items():
        if rep_tid not in covered:
            clusters.append(sorted(set(track_ids)))
    clusters.sort(key=len, reverse=True)

    _finish(event, out_bucket, clusters_key, clusters, pairs, counter,
            celeb_by_track)
    return event


def _finish(event, out_bucket, clusters_key, clusters, pairs, counter,
            celeb_by_track=None):
    out = {
        "clusters": [
            {"face_group_id": i, "track_ids": tids,
             "celebrity": _cluster_celebrity(tids, celeb_by_track or {})}
            for i, tids in enumerate(clusters)
        ],
        "similarity_pairs": pairs,
        "usage": counter.as_dict(),
    }
    put_json(out_bucket, clusters_key, out)
    event["data"]["clusters_key"] = clusters_key
    event["data"]["num_clusters"] = len(clusters)
    event["data"]["iterators"] = None
    logger.info("%d identity clusters", len(clusters))


def _merge_corroborated_weak_edges(weak_edges, find, union) -> None:
    """
    Weak-edge corroboration (after all searches): a weak edge
    [SANDBOX_SEARCH_THRESHOLD, SANDBOX_MERGE_SIMILARITY) merges two
    components only when >= SANDBOX_WEAK_CORROBORATION independent weak
    edges connect the SAME two components. Iterated to fixpoint: a
    corroborated merge can make further weak edges corroborating.
    One borderline match is noise; two independent borderline matches
    between the same two groups is signal.
    """
    # Dedup by unordered face pair FIRST: SearchFaces is symmetric, so when
    # both endpoints get searched the same underlying match arrives twice
    # (A→B and B→A). Counting both would let a single bidirectional noise
    # match satisfy SANDBOX_WEAK_CORROBORATION=2 on its own — exactly the
    # single-weak-edge contamination this tier exists to prevent.
    unique_edges = {frozenset((fid, mid)) for fid, mid, _ in weak_edges}
    changed = True
    while changed:
        changed = False
        support: dict[frozenset, int] = {}
        for edge in unique_edges:
            fid, mid = tuple(edge)
            ra, rb = find(fid), find(mid)
            if ra != rb:
                key = frozenset((ra, rb))
                support[key] = support.get(key, 0) + 1
        for key, n in support.items():
            if n >= SANDBOX_WEAK_CORROBORATION:
                ra, rb = tuple(key)
                if find(ra) != find(rb):
                    union(ra, rb)
                    changed = True


def _cluster_celebrity(track_ids, celeb_by_track):
    """
    Plurality-vote celebrity tag among the cluster's tracks — NOT the single
    highest-confidence observation (real bug, found on real content: a
    non-celebrity face got a wrong name from ~1/5 of its RecognizeCelebrities
    calls, never the same wrong name twice, at a mean confidence far below a
    genuine match's; but ONE outlier sample cleared NAME_VOTE_INSTANT_SCORE
    and got auto-indexed under a stranger's name. A real match's confidence
    is high and consistent across nearly every observation — a false
    positive only looks confident if you cherry-pick its single best sample).
    Tally votes per name, return the plurality name's MEAN confidence so
    _vote_name's thresholds see the honest signal instead of an outlier.
    """
    # Votes are tallied by the recognizer's IDENTITY KEY — celebrity Id
    # when the tag carries one, else the name string: the SAME key rule
    # _name_groups clusters by (review, 2026-09-07). Keying by name
    # while grouping by id let one Id's votes fragment across spelling
    # variants and lose to a noise name; keying by id also separates
    # real homonyms so neither steals the other's tally.
    votes: dict[str, list[dict]] = {}
    for tid in track_ids:
        tag = celeb_by_track.get(tid) or {}
        if tag.get("name"):
            votes.setdefault(tag.get("celebrity_id") or tag["name"],
                             []).append(tag)
    if not votes:
        return None

    def _rank(obs):
        return (len(obs),
                sum(float(o.get("confidence") or 0) for o in obs))

    ranked = sorted(votes.items(), key=lambda kv: _rank(kv[1]),
                    reverse=True)
    key, obs = ranked[0]
    finalists = [kv for kv in ranked if _rank(kv[1]) == _rank(obs)]
    name = Counter(o["name"] for o in obs).most_common(1)[0][0]
    cid = obs[0].get("celebrity_id") if obs[0].get("celebrity_id") == key \
        else None
    if len(finalists) > 1:
        # a plurality tied on count AND summed confidence is CONTESTED
        # evidence — never let the first-observed key win a coin flip.
        # Homonym Ids sharing one display name keep the NAME (it is the
        # same string either way) with the id suppressed; different
        # names abstain entirely (a wrong name is worse than none).
        names = {Counter(o["name"] for o in kv[1]).most_common(1)[0][0]
                     .casefold() for kv in finalists}
        if len(names) > 1:
            return None
        cid = None
    return {"name": name,
            "confidence": sum(float(o.get("confidence") or 0)
                              for o in obs) / len(obs),
            "celebrity_id": cid}


def _name_groups(all_tracks, celeb_by_track) -> list[list[int]]:
    # Grouping key: the recognizer's celebrity Id when the tag carries
    # one, else the display-name string (review, 2026-09-07). The Id is
    # the recognizer's actual identity claim — name strings are neither
    # unique (real homonym celebrities stay separate under Id keying)
    # nor stable (spelling variants of one Id now merge HERE, inside
    # face-recognition, where thumbnails, screen totals and the §23
    # conflict machinery live — instead of surfacing as duplicate
    # people[] entries every consumer must coalesce).
    by_key: dict[str, list[int]] = {}
    groups: list[list[int]] = []
    for tr in all_tracks:
        tag = celeb_by_track.get(tr["track_id"]) or {}
        key = tag.get("celebrity_id") or tag.get("name")
        if key:
            by_key.setdefault(key, []).append(tr["track_id"])
        else:
            groups.append([tr["track_id"]])
    groups.extend(by_key.values())
    return groups


def _collect_celebrity_tags(event, out_bucket) -> tuple[dict[int, dict], dict]:
    """
    track_id → {name, confidence} from the recognize Map results, plus the
    summed usage (recognize_celebrities_api) across all recognize iterators —
    this is the ONLY place that usage is ever readable (see call site).
    """
    tags: dict[int, dict] = {}
    usage_total: dict = {}
    for it in event["data"].get("iterators") or []:
        iterator = get_json(out_bucket, it["key"])
        for batch_result in (iterator.get("results") or {}).values():
            for tid_str, tag in batch_result.items():
                tags[int(tid_str)] = tag
        add_usage(usage_total, iterator.get("usage") or {})
    return tags, usage_total


def _checkpoint(out_bucket, state_key, face_to_track, state, counter):
    state["face_to_track"] = {k: v for k, v in face_to_track.items()}
    state["usage"] = counter.as_dict()
    put_json(out_bucket, state_key, state)


