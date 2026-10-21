# lib/recurrence.py
"""
Shot recurrence detection.

Identifies shots that appear multiple times throughout the content
with near-identical visual content, based on cosine distance between
shot centroid embeddings.

Algorithm (gates 9.1-9.4 + 9.6 from README.md Phase 9, back-ported lessons
from the main pipeline's 9-asset live trial)
─────────
  1. CONTENT shots only (9.2): black shots share the identical sentinel
     centroid (distance 0.0), so >= min_count of them formed one giant fake
     group; static shots (bars/slates) recur trivially. Both are excluded
     from the similarity matrix — but NOT from position classification,
     which needs black neighbours to label act_boundary.
  2. L2-normalise shot centroid embeddings
  3. Compute all pairwise cosine distances
  4. Canonical complete-linkage grouping within max_recurrence_threshold,
     so the threshold bounds group DIAMETER: every published group satisfies
     max_distance <= max_recurrence_threshold. (Was Union-Find / single
     linkage until 2026-08-21, which bounded only the nearest-neighbour
     distance and chained groups into mixed bags — README.md 9.11.)
  5. Keep groups with >= min_recurrence_count members
  6. Temporal-separation gate (9.1): members closer than
     min_occurrence_gap_sec collapse into ONE occurrence cluster, and
     min_recurrence_count is enforced on CLUSTERS — dialogue coverage
     (A/B/A/B inside one scene) recurs in raw shot count but is a single
     occurrence; 32 of 33 groups on drama_e were dialogue coverage.
  7. Per group: max_distance, positions, occurrence_count. NO set-recurrence
     ceiling any more (9.3, retired as a DROP 2026-08-21): groups over
     max_recurrence_count — anchor cameras at 69 members, confessional chairs
     at 11-25, which pass every similarity gate because they ARE
     near-identical content — are published, and the consumer derives
     "exceeds the set ceiling" from recurrence_count > the envelope's
     max_recurrence_count. See the note where the gate used to live for why
     the drop was wrong (it destroyed 6 of 12 verified furniture groups).
  8. Hash confirmation (9.6 + 9.4), a SEPARATE entry point —
     confirm_groups_by_hash() — because it needs decoded frames and so runs
     in the merge state after per-shot repack: embeddings confirm the same
     SET, a dHash family confirms the same RENDERED IMAGE. Also report-not-
     drop: it sets hash_confirmed true/false/null per group.

Structure
─────────
Steps 3-7 are one function per decision (`_content_distances`,
`_group_by_complete_linkage`, `_gate_*`, `_serialise_group`), with
`detect_recurrence()` a thin composition over them. Each gate is a pure filter
over a list of `_Candidate` and owns exactly one `stats` counter, so "which
gate ate my card?" is answerable by reading one function.

Output per group (trimmed 2026-08-21 to the facts this pipeline DECIDES;
everything about a member shot — times, frames, timecodes, durations,
entropies, dispersion — lives in shots.json, joinable by shot_id.
recurrence.json and shots.json are a pair.)
────────────────
{
    "group_id":         0,
    "recurrence_count": 4,        # == len(shots); the convenience exception
    "occurrence_count": 4,        # 9.1 gap rule — not derivable from times alone
    "max_distance":     0.032,    # group diameter; needs the centroids
    "hash_confirmed":   true,     # set by confirm_groups_by_hash; see there
    "shots": [
        {"shot_id": 4, "position": "act_boundary"},
        ...
    ]
}
"""
from __future__ import annotations

import logging
from collections import defaultdict, namedtuple
import numpy as np
from lib import frame_hash
from lib.input_schema import (
    DEFAULT_BLACK_PIXEL_THRESHOLD,
    DEFAULT_MAX_RECURRENCE_THRESHOLD,
    DEFAULT_MIN_OCCURRENCE_GAP_SEC,
    DEFAULT_MIN_RECURRENCE_COUNT,
    RECURRENCE_MAX_BLACK_FRACTION,
    RECURRENCE_HASH_MAX_FRAMES,
    RECURRENCE_HASH_MAX_GLOBAL,
    RECURRENCE_HASH_MAX_MEDIAN,
    RECURRENCE_HASH_MIN_ENTROPY,
    RECURRENCE_HASH_NOISE_THRESHOLD,
)

log = logging.getLogger(__name__)

_HEAD_TAIL_FRACTION = 0.10

# Per-gate observability (README.md 9.10). A group can die at four places inside
# detect_recurrence() and a fifth in confirm_groups_by_hash(), and until this
# existed only the last one was reportable: the lib's log.info lines do not
# propagate out of the Lambda, so "why is my card missing?" needed an offline
# replay against the run's own shot_centroids.npy just to learn WHICH gate ate
# it. 9.3 in particular had no counter anywhere despite culling 25 groups and
# 510 member shots across the corpus. Every key is a count of GROUPS unless
# named otherwise.
_GATE_STAT_KEYS = (
    "content_shots",            # 9.2: shots that entered the similarity matrix
    "excluded_non_content",     # 9.2: black/bars/slate shots held out of it
    "components",               # complete-linkage clusters with >= 2 members
    "min_members_dropped",      # components of 2..min_recurrence_count-1
    "occurrence_gate_dropped",  # 9.1: too few temporally-separated occurrences
    "groups",                   # survivors returned to the caller
)
# No set-ceiling counters: since 2026-08-21 the 9.3 ceiling annotates instead
# of dropping, and how many groups exceed it is derivable from the output
# (recurrence_count > the envelope's max_recurrence_count).


def _classify_position(
    shot_idx:     int,
    shots_sorted: list[dict],
    relative_pos: float,
) -> str:
    if relative_pos < _HEAD_TAIL_FRACTION:
        return "head"
    if relative_pos > (1.0 - _HEAD_TAIL_FRACTION):
        return "tail"

    n = len(shots_sorted)
    prev_is_black = (
        shot_idx > 0
        and shots_sorted[shot_idx - 1].get("shot_type") == "black"
    )
    next_is_black = (
        shot_idx < n - 1
        and shots_sorted[shot_idx + 1].get("shot_type") == "black"
    )
    if prev_is_black or next_is_black:
        return "act_boundary"

    return "interior"


def occurrence_labels(spans: list[tuple[float, float]],
                      gap_sec: float) -> list[int]:
    """9.1: which temporally-separated OCCURRENCE each (start, end) span
    belongs to, for spans sorted by start. A member joins the previous
    occurrence unless the GAP between the running END and its START is
    >= gap_sec — end-to-start, matching the main pipeline's shotRecurrence.js
    (timestampRange[0] - prev.timestampRange[1]). Start-to-start would let a
    long dialogue shot promote its own coverage into a second "occurrence"
    just by being long.

    The running end is the MAXIMUM end seen so far, not the previous span's.
    For shot members the two agree (shots are disjoint and sorted, so ends are
    monotone) but the max form is the one that stays correct if a caller ever
    passes overlapping spans.

    Public, and returning labels rather than a count, because the gate has two
    consumers with two views: the pipeline needs only how many, while the
    offline replays need the members of each occurrence to measure the gaps
    between them. They used to carry their own copies of the rule, which had
    drifted apart on exactly the running-end question above.
    """
    labels, cur, running_end = [], 0, None
    for start, end in spans:
        if running_end is not None and start - running_end >= gap_sec:
            cur += 1
        labels.append(cur)
        running_end = end if running_end is None else max(running_end, end)
    return labels


def count_occurrences(spans: list[tuple[float, float]],
                      gap_sec: float) -> int:
    """Number of temporally-separated occurrences (9.1). See
    occurrence_labels()."""
    return (occurrence_labels(spans, gap_sec)[-1] + 1) if spans else 0


# ── the pipeline ─────────────────────────────────────────────────────────
# detect_recurrence() was one 243-line function doing eight things with the
# grouping algorithm and four gates interleaved. It is split so that each
# decision is independently testable and, more to the point, so that replacing
# the grouping algorithm is a substitution of ONE function rather than surgery
# in the middle of a loop — which is how single linkage (the documented root
# defect: unbounded group diameter, README.md 9.11) was swapped for
# `_group_by_complete_linkage` the day after the split.
#
# The carrier between stages is a _Candidate. The gates are pure filters over a
# list of them: each takes the list, returns a shorter list, and touches only
# its own `stats` counter.

_Candidate = namedtuple("_Candidate", "members occurrences")

# Everything the projection stage needs that is NOT per-candidate. Bundled
# because it is threaded through unchanged and eight positional arguments per
# call obscured which ones a stage actually reads.
_Ctx = namedtuple("_Ctx", "shots_sorted content_rows distances duration")


def _resolve_cfg(cfg: dict) -> dict:
    """The three knobs this gate stack reads, with their schema defaults.

    `max_recurrence_count` is deliberately NOT here: since 2026-08-21 the 9.3
    ceiling no longer drops anything, so the pipeline never reads it. The knob
    survives as a published REFERENCE value in the recurrence.json envelope —
    a consumer derives "exceeds the set ceiling" from
    `recurrence_count > max_recurrence_count`, both already in the output.
    """
    return {
        "threshold": float(cfg.get("max_recurrence_threshold",
                                   DEFAULT_MAX_RECURRENCE_THRESHOLD)),
        "min_count": int(cfg.get("min_recurrence_count",
                                 DEFAULT_MIN_RECURRENCE_COUNT)),
        "gap_sec": float(cfg.get("min_occurrence_gap_sec",
                                 DEFAULT_MIN_OCCURRENCE_GAP_SEC)),
    }


def _content_distances(shots_sorted: list[dict],
                       content_rows: list[int]) -> np.ndarray:
    """9.2: pairwise cosine distances over CONTENT shots only.

    Black shots share the identical sentinel centroid (distance 0.0), so
    >= min_count of them formed one giant fake group; static shots (bars,
    slates) recur trivially. Both are held out of the matrix — but not out of
    `shots_sorted`, because position classification needs black NEIGHBOURS to
    label act_boundary.
    """
    centroids = np.array(
        [shots_sorted[i]["centroid_embedding"] for i in content_rows],
        dtype=np.float32,
    )
    normed = (centroids
              / np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-8)
              ).astype(np.float32)
    distances = np.clip(1.0 - (normed @ normed.T).astype(np.float32), 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def complete_linkage(distances, threshold: float) -> list[list[int]]:
    """CANONICAL complete-linkage clustering: each round, merge the pair of
    clusters whose complete-linkage distance (max cross-pair) is SMALLEST, and
    stop when the smallest admissible distance exceeds `threshold`. Every
    resulting cluster has diameter <= threshold, singletons included in the
    return.

    Canonical, not greedy, and that is measurement rather than taste: a
    first-admissible-pair version is order-dependent, and on the corpus sweep
    it reported NO threshold at which all six verified true cards survive,
    where the canonical form has a clean window at 0.065-0.070
    (test/harness/sweep.py). Ties break lexicographically on current cluster
    positions — the same order the harness's original pure-Python
    implementation used, so the two produce identical clusters and the replays'
    recorded numbers stay the method of record.

    Public because the offline replays cluster candidate groups with exactly
    this rule and must not drift from production
    (test/harness/linkage.py re-exports it).

    Cost: O(k^2) expected via the per-row minima cache below (k is a
    single-linkage COMPONENT size, not the shot count). Measured 2026-08-22
    against the shrinking-matrix version it replaced: identical output on
    4930 matrices, and the degenerate every-pair-within-threshold shape went
    12.7 s -> 0.08 s at k=2000 (164x). One measured dead end recorded in
    README.md 9.11: masking alone (no row cache) bought NOTHING — the rescans,
    not the allocations, were the cost.
    """
    S = np.array(distances, dtype=np.float64)
    # Masked in-place with per-row minima (README.md 9.11's deferred
    # optimisation, done 2026-08-22). The naive loop rescanned the full upper
    # triangle every merge — O(k^3) elements touched overall. Here the matrix
    # stays in original index space with merged-away rows at +inf, and each
    # row caches (its minimum over columns to its right, the FIRST column
    # attaining it); a merge recomputes only its own row plus the rows whose
    # cached minimum sat in one of the two merged columns.
    #
    # Tie-break identity is preserved by two structural facts, not luck:
    # (1) the cluster list never REORDERS (a merge lands at the earlier slot
    # and pops the later), so lexicographic order over current list positions
    # is exactly original-index order among live clusters; (2) np.argmin
    # returns FIRST occurrence, so scanning row minima in index order picks
    # precisely the row-major-first global minimum the full-triangle scan
    # chose. A row's cache can only be invalidated by a merge that touched
    # its cached column: fusion (elementwise max) never DECREASES an entry,
    # and killing a column only removes candidates that were not the minimum.
    n = S.shape[0]
    if n == 0:
        return []
    np.fill_diagonal(S, np.inf)

    row_min = np.full(n, np.inf)
    row_arg = np.full(n, -1, dtype=np.int64)

    def _recompute(r: int) -> None:
        if r >= n - 1:
            row_min[r] = np.inf
            row_arg[r] = -1
            return
        seg = S[r, r + 1:]
        k = int(seg.argmin())
        row_min[r] = seg[k]
        row_arg[r] = r + 1 + k

    for r in range(n):
        _recompute(r)

    members: list[list[int] | None] = [[i] for i in range(n)]
    for _ in range(n - 1):
        i = int(row_min.argmin())
        if not (row_min[i] <= threshold):
            break
        j = int(row_arg[i])
        fused = np.maximum(S[i], S[j])   # dead entries stay +inf under max
        S[i, :] = fused
        S[:, i] = fused
        S[i, i] = np.inf
        S[j, :] = np.inf
        S[:, j] = np.inf
        members[i] = members[i] + members[j]
        members[j] = None
        row_min[j] = np.inf
        row_arg[j] = -1
        _recompute(i)
        for r in np.nonzero((row_arg == i) | (row_arg == j))[0]:
            if members[int(r)] is not None and int(r) != i:
                _recompute(int(r))
    return [m for m in members if m is not None]


def _group_by_complete_linkage(distances: np.ndarray,
                               threshold: float) -> list[list[int]]:
    """Grouping = complete linkage, so `threshold` bounds group DIAMETER.

    Until 2026-08-21 this was Union-Find over pairs within threshold — i.e.
    single linkage, which bounds only the nearest-neighbour distance. The
    requirement is "one recurring element per group", which is a diameter
    constraint, and single linkage violated it in shipped output: groups
    chained transitively into mixed bags, with measured diameters up to 2.6x
    the threshold, three published docu-batch groups dissolving entirely under
    the diameter rule, and the same recurrence.json carrying
    `max_recurrence_threshold: 0.05` and `max_distance: 0.1315` (README.md 9.11,
    test/harness/chaining.py). Complete linkage restores the invariant
    max_distance <= max_recurrence_threshold on every published group.

    The Union-Find below is retained as a lossless CONNECTIVITY SCREEN, not as
    the grouping rule: at equal thresholds every complete-linkage cluster is a
    clique of <=threshold pairs, hence connected, hence inside one
    single-linkage component — and any two clusters from different components
    are separated by a >threshold pair, so no cross-component merge is ever
    admissible. Clustering each component independently is therefore EXACTLY
    the global algorithm, and it turns an O(N^3) pass over ~2000 content shots
    into small per-component problems (largest observed component well under
    a hundred members).
    """
    n = distances.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= threshold:
                px, py = find(i), find(j)
                if px != py:
                    parent[px] = py

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    groups: list[list[int]] = []
    for comp in components.values():
        if len(comp) == 1:
            groups.append(comp)
            continue
        sub = distances[np.ix_(comp, comp)]
        groups.extend([comp[i] for i in cluster]
                      for cluster in complete_linkage(sub, threshold))
    return groups


def _gate_min_members(components: list[list[int]], min_count: int,
                      stats: dict | None) -> list[list[int]]:
    """A group needs at least min_recurrence_count raw members."""
    if stats is not None:
        # Singletons are excluded from both counters: on a 1000-shot asset most
        # shots are unique, so counting them would report ~900 "drops" and bury
        # the four numbers that mean something.
        multi = [g for g in components if len(g) >= 2]
        stats["components"] = len(multi)
        stats["min_members_dropped"] = sum(1 for g in multi
                                           if len(g) < min_count)
    return [g for g in components if len(g) >= min_count]


def _annotate_occurrences(components: list[list[int]], ctx: _Ctx,
                          gap_sec: float) -> list[_Candidate]:
    """Attach each component's 9.1 occurrence count. Computing is separate from
    deciding so the count can be published on groups that pass."""
    out = []
    for g in components:
        spans = sorted(
            (float(ctx.shots_sorted[ctx.content_rows[i]].get("start_time",
                                                            0.0)),
             float(ctx.shots_sorted[ctx.content_rows[i]].get("end_time", 0.0)))
            for i in g)
        out.append(_Candidate(g, count_occurrences(spans, gap_sec)))
    return out


def _gate_occurrence_separation(cands: list[_Candidate], min_count: int,
                                gap_sec: float,
                                stats: dict | None) -> list[_Candidate]:
    """9.1: min_recurrence_count applies to OCCURRENCES, not raw members.

    Dialogue coverage (A/B/A/B inside one scene) recurs in raw shot count but
    is a single occurrence — 32 of 33 groups on drama_e were this shape.
    """
    kept = []
    for c in cands:
        if c.occurrences < min_count:
            log.info(
                "Recurrence: dropped group (%d members but only %d "
                "occurrence(s) >= %.0fs apart) — dialogue-coverage shape",
                len(c.members), c.occurrences, gap_sec)
            if stats is not None:
                stats["occurrence_gate_dropped"] += 1
            continue
        kept.append(c)
    return kept


# There is deliberately NO 9.3 gate here (removed 2026-08-21; it dropped
# groups with more than max_recurrence_count raw members, targeting recurring
# SETS — anchor cameras at 69 members, confessional chairs at 11-25, which
# pass every similarity gate because they ARE near-identical content). Two
# measurements retired it as a DROP:
#   * member count is the WEAKEST available correlate of "graphic insert
#     rather than a place" (AUC 0.799, 9th of 20; test/harness/features.py),
#     and the ceiling of 8 destroyed 6 of the corpus's 12 verified furniture
#     groups — cards on 2-hour broadcast blocks legitimately reach 24 members;
#   * every property that separates better (coverage, ent_range, occ_ratio,
#     dur_cv) is derivable by the consumer from data already published, so the
#     verdict can be theirs. `max_recurrence_count` stays in the envelope as
#     the reference value: exceeds-ceiling is
#     `recurrence_count > max_recurrence_count`, no flag needed.
# The set-shaped groups it used to hide now flow to 9.6, which flags rather
# than deletes (hash_confirmed) — nothing on the recurrence path silently
# discards evidence any more.


def _serialise_group(cand: _Candidate, group_id: int, ctx: _Ctx) -> dict:
    """Project a surviving candidate into the published group shape.

    Member entries are {shot_id, position} and nothing else, on the
    single-source-of-truth rule (trimmed 2026-08-21): times, frames,
    timecodes, durations and entropies all live in shots.json, joinable by
    shot_id — recurrence.json and shots.json are a pair, and every copied
    field is a chance for the pair to disagree. `position` is the one
    per-member fact this pipeline decides rather than restates. Group-level
    numbers are kept only where not derivable from the join:
    occurrence_count (needs the 9.1 gap rule), max_distance (needs the
    centroids), and recurrence_count as the single convenience exception
    (== len(shots), used to filter without fetching shots.json).

    Member indices are rows in the CONTENT-ONLY matrix; `content_rows` maps
    them back to positions in the full blacks-included sorted list, which
    classification needs for its black-neighbour checks.
    """
    members = sorted(
        cand.members,
        key=lambda i: ctx.shots_sorted[ctx.content_rows[i]]["start_frame"])
    sub = ctx.distances[np.ix_(members, members)]
    max_dist = round(float(sub.max()), 4)

    shot_entries: list[dict] = []
    for idx in members:
        row = ctx.content_rows[idx]
        shot = ctx.shots_sorted[row]
        start = float(shot.get("start_time", 0.0))
        rel = start / ctx.duration if ctx.duration > 0 else 0.0
        shot_entries.append({
            "shot_id":  shot["shot_id"],
            "position": _classify_position(row, ctx.shots_sorted, rel),
        })

    log.info(
        "Recurrence group %d: count=%d  occurrences=%d  max_dist=%.4f  "
        "positions=%s",
        group_id, len(members), cand.occurrences, max_dist,
        [e["position"] for e in shot_entries])
    return {
        "group_id":         group_id,
        "recurrence_count": len(members),
        "occurrence_count": cand.occurrences,
        "max_distance":     max_dist,
        "shots":            shot_entries,
    }


def detect_recurrence(
    shots:               list[dict],
    cfg:                 dict,
    actual_duration_sec: float | None,
    stats:               dict | None = None,
) -> list[dict]:
    """
    Detect recurring shots across the full content.

    Thin composition over the stages above — group, gate, gate, project.
    Everything it does itself is the part that cannot be a stage: reading cfg,
    the 9.2 content/black split (whose two views the rest of the pipeline
    needs kept aligned), and the two early returns.

    Parameters
    ──────────
    shots               Full shot list with centroid_embedding.
    cfg                 Config dict — reads:
                          max_recurrence_threshold  (default 0.07)
                          min_recurrence_count      (default 3)
                          min_occurrence_gap_sec    (default 60.0, 9.1)
    actual_duration_sec Ground-truth content duration.
    stats               Optional OUT-parameter dict; when given it is
                        populated with the per-gate drop counters listed under
                        _GATE_STAT_KEYS. An out-parameter rather than a second
                        return value because every caller — the merge state,
                        the unit tests, and a dozen offline replay scripts —
                        unpacks a bare list, and this gate needs observability,
                        not a signature migration.

    Returns list of recurrence group dicts sorted by first occurrence.
    """
    if stats is not None:
        stats.update({k: 0 for k in _GATE_STAT_KEYS})
    knobs = _resolve_cfg(cfg)
    min_count = knobs["min_count"]

    # Full sorted list (blacks included) — position classification needs
    # black NEIGHBOURS to label act_boundary; only the similarity matrix
    # excludes them (9.2).
    shots_sorted = sorted(shots, key=lambda s: s["start_frame"])
    content_rows = [i for i, s in enumerate(shots_sorted)
                    if s.get("shot_type") == "content"]

    N = len(content_rows)
    if stats is not None:
        stats["content_shots"] = N
        stats["excluded_non_content"] = len(shots_sorted) - N
    if N < min_count:
        log.info(
            "Recurrence: too few content shots (%d < %d) — skipping",
            N, min_count)
        return []

    log.info(
        "Recurrence detection: %d content shots (of %d)  threshold=%.3f  "
        "min_count=%d  gap=%.0fs",
        N, len(shots_sorted), knobs["threshold"], min_count, knobs["gap_sec"],
    )

    ctx = _Ctx(
        shots_sorted=shots_sorted,
        content_rows=content_rows,
        distances=_content_distances(shots_sorted, content_rows),
        duration=float(actual_duration_sec) if actual_duration_sec else 0.0,
    )

    components = _group_by_complete_linkage(ctx.distances, knobs["threshold"])
    components = _gate_min_members(components, min_count, stats)
    cands = _annotate_occurrences(components, ctx, knobs["gap_sec"])
    cands = _gate_occurrence_separation(cands, min_count, knobs["gap_sec"],
                                        stats)

    if stats is not None:
        stats["groups"] = len(cands)

    if not cands:
        log.info("Recurrence: no groups survive the gates")
        return []

    log.info("Recurrence: %d group(s) found", len(cands))

    # Sorted by first occurrence. `min(members)` is the earliest member because
    # content_rows is increasing and shots_sorted is start_frame-ordered.
    in_time_order = sorted(
        cands,
        key=lambda c: shots_sorted[content_rows[min(c.members)]]["start_frame"])
    return [_serialise_group(c, gid, ctx)
            for gid, c in enumerate(in_time_order)]


def _is_black_dominated(luma) -> bool:
    """9.4(a): >= RECURRENCE_MAX_BLACK_FRACTION of the frame is black. A
    timecode-burn frame passes the entropy floor (the burned glyphs ARE
    texture) and then hashes identically to every other frame in its burn
    window, so entropy cannot be the only unhashable test."""
    cutoff = DEFAULT_BLACK_PIXEL_THRESHOLD * 255.0
    arr = np.asarray(luma)
    if arr.size == 0:
        return True
    return float((arr <= cutoff).mean()) >= RECURRENCE_MAX_BLACK_FRACTION


def confirm_groups_by_hash(groups, load_lumas, frame_entropy=None):
    """9.6 confirmation gate (+ the 9.4 texture floor). Embeddings confirm
    same-SET content; a hash family confirms same-RENDERED-IMAGE — which is
    what separates an act-break card (the same render coming back) from a
    recurring location the embedding also groups.

    Report-not-drop (2026-08-21, README.md 9.9 option 3): this gate DECIDES but
    no longer DELETES. Every group comes back with `hash_confirmed` set:

      true    some frame pair from two DIFFERENT members matches — the same
              rendered image genuinely came back
      false   checked, and no cross-member pair matches — same-set-not-
              same-render, or animated/re-composited furniture (the two
              known false negatives in README.md 9.9 are this class)
      null    UNVERIFIABLE — fewer than two members have hashable (textured,
              non-black-dominated) frames, the 9.4 burn-in/glitch shape.
              Null, not false: "we could not check" must not read as
              "we checked and it failed", the same convention as the
              null dispersion fields in shots.json.

    The old behaviour deleted false/null groups and reindexed the survivors;
    that made the two known animated-furniture false negatives silently
    vanish, recoverable only by offline replay. Group ids are now stable
    through this gate.

    groups         detect_recurrence() output (already gated by 9.1-9.2).
    load_lumas     callable(shot_id) -> time-ordered list of
                   (frame_num, 2-D uint8 luma array) pairs for that shot's
                   archived frames. The caller owns I/O; this module owns
                   the decision.
    frame_entropy  optional {frame_num: entropy} — frames below
                   RECURRENCE_HASH_MIN_ENTROPY are skipped as unhashable
                   (glitch-black and fade frames hash identically and would
                   false-confirm; real cards carry texture).

    A frame is also skipped when it is BLACK-DOMINATED (9.4 option (a),
    >= RECURRENCE_MAX_BLACK_FRACTION of its pixels black). That is a
    separate test from the entropy floor and neither subsumes the other: a
    timecode-burn frame is 99.2% black yet reads HIGH entropy because the
    burned glyphs are real texture, which is exactly how those masters
    manufacture a fake group per burn window.

    Per group: hash up to RECURRENCE_HASH_MAX_FRAMES textured frames per
    member, sampled EVENLY ACROSS the member's archived span (not first-N —
    see the comment at the sampling, README.md 9.9 option 2); chronically-noisy
    patches (a ticker/timecode window churns
    inside one occurrence; card art does not) are classified once per group
    and discounted from patch comparisons.

    That classification runs over ONE flat series of every member's hashes
    in order, NOT per member, and the reason is arithmetic:
    classify_noisy_patches() needs >= 10 adjacent pairs, while a member
    contributes at most RECURRENCE_HASH_MAX_FRAMES (8) frames = 7 pairs — so
    per-member classification could never fire and the discount would be
    dead code. The cost is that member boundaries contribute
    len(members) - 1 cross-member pairs to the series. That cannot flip a
    classification on its own (a patch needs churn in > 70% of pairs, i.e.
    22 of 31 on a 4-member group), and its bias is toward discounting a
    patch, i.e. toward confirming — never toward a false rejection. Do not
    "fix" this by splitting per member without also raising the frame cap
    past the 10-pair floor.

    CONFIRMED iff some frame pair from two DIFFERENT
    members matches on whole-frame global hamming <=
    RECURRENCE_HASH_MAX_GLOBAL or on stable-patch MEDIAN <=
    RECURRENCE_HASH_MAX_MEDIAN — measured on the real drama_e card: the
    max-of-patches break statistic false-rejects genuine re-renders (one
    grainy patch spikes it to 7-13 across members and 19 even within one
    shot), while global reads 0-7 and the median 1-6.

    Returns the same list, each group annotated in place."""
    frame_entropy = frame_entropy or {}
    for group in groups:
        member_hashes = []   # one list of frame-hash dicts per member
        # Churn series for noise classification: every member's hashes in
        # order, ONE flat list — see the docstring for why this is not
        # classified per member.
        churn_series = []
        for entry in group["shots"]:
            lumas = load_lumas(entry["shot_id"]) or []
            hashable = [
                (fn, luma) for fn, luma in lumas
                if not (frame_entropy.get(fn) is not None
                        and frame_entropy[fn] < RECURRENCE_HASH_MIN_ENTROPY)
                and not _is_black_dominated(luma)
            ]
            # Sample the RECURRENCE_HASH_MAX_FRAMES hash frames EVENLY ACROSS
            # the member's archived span, not first-N (README.md 9.9 option 2,
            # 2026-08-21). First-N anchored every member's hashes to its
            # opening frames, so two members of an ANIMATED card sampled at
            # different phases of the animation could present
            # non-overlapping phase windows and never produce the one
            # matching pair the gate needs — the news_b "Tuesday" pan (pre-9.6
            # max_distance 0.0128, tighter than the canonical drama_e true card)
            # was rejected exactly this way. Even sampling keeps the same
            # frame budget but covers the whole animation cycle from both
            # sides. Time order is preserved (the churn series below relies
            # on it).
            k = len(hashable)
            if k > RECURRENCE_HASH_MAX_FRAMES:
                step = (k - 1) / (RECURRENCE_HASH_MAX_FRAMES - 1)
                keep = sorted({round(i * step)
                               for i in range(RECURRENCE_HASH_MAX_FRAMES)})
                hashable = [hashable[i] for i in keep]
            hashes = [frame_hash.frame_hashes(luma) for _, luma in hashable]
            if hashes:
                member_hashes.append(hashes)
                churn_series.extend(hashes)
        if len(member_hashes) < 2:
            log.info(
                "Recurrence: group %d unverifiable — fewer than 2 members "
                "have hashable (textured) frames; burn-in/glitch shape (9.4)",
                group["group_id"])
            group["hash_confirmed"] = None
            continue

        noisy = frame_hash.classify_noisy_patches(
            churn_series, RECURRENCE_HASH_NOISE_THRESHOLD)
        noisy_set = set(noisy)

        def _match(ha, hb):
            if frame_hash.hamming(ha["global"], hb["global"]) \
                    <= RECURRENCE_HASH_MAX_GLOBAL:
                return True
            stable = sorted(
                frame_hash.hamming(pa, pb)
                for p, (pa, pb) in enumerate(zip(ha["patches"],
                                                 hb["patches"]))
                if p not in noisy_set)
            return bool(stable) and \
                stable[len(stable) // 2] <= RECURRENCE_HASH_MAX_MEDIAN

        def _pair_confirms():
            for a in range(len(member_hashes)):
                for b in range(a + 1, len(member_hashes)):
                    for ha in member_hashes[a]:
                        for hb in member_hashes[b]:
                            if _match(ha, hb):
                                return True
            return False

        group["hash_confirmed"] = _pair_confirms()
        if not group["hash_confirmed"]:
            log.info(
                "Recurrence: group %d NOT hash-confirmed — no cross-member "
                "frame pair matches on stable-patch hash (same-set, not "
                "same-render)", group["group_id"])
    return groups
