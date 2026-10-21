"""Shared corpus access for the Phase 9 replays: load assets, build distance
matrices, iterate labelled groups. Every replay imports from here.

Four primitives were previously copy-pasted across the harness, and the copies
had drifted:

* the asset loader — `chaining.py` had `runs_in` (which tolerates a flat
  directory and reports a missing artefact) and `sweep.py` had `load` (which
  attaches centroids), so only one of the two noticed that film_a timed out
  upstream;
* the distance matrix — `chaining.py`, `sweep.py` and `textless.py` each
  L2-normalised and built `1 - V @ V.T` inline, one of them without zeroing the
  diagonal;
* gate 9.1's occurrence clustering — three copies, two of which used the
  previous member's end-time as the running end where production uses the
  maximum so far. This one is no longer here either: it now calls
  `lib.recurrence.occurrence_labels`, which the pipeline refactor exposed for
  exactly this reason;
* the label loop — `verdict.py`, `guardrails.py`, `rule_test.py`, `separate.py`
  and the feature study all ran `detect_recurrence` twice (once with the ceiling
  disabled, once at production defaults), diffed the two, and attached ground
  truth. Five copies of a twenty-line loop that decides what counts as a false
  cull.

`load` is deliberately eager about attaching `centroid_embedding` to each shot
dict, because that is what `detect_recurrence` reads — replaying the shipped
code path on the shipped data is the whole point of the harness.
"""
import glob
import json
import os
import sys
from collections import namedtuple

import numpy as np

# Repo root, so importing this module is enough to reach `lib.*` — every replay
# used to carry its own three-deep dirname chain.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.input_schema import (DEFAULT_MIN_OCCURRENCE_GAP_SEC,   # noqa: E402
                              DEFAULT_MIN_RECURRENCE_COUNT)
from lib.recurrence import (count_occurrences,                  # noqa: E402
                            detect_recurrence, occurrence_labels)
from truth import (cards_for, label_audited,       # noqa: E402 (harness-local)
                   label_culled)

BATCHES = ("phase9-batch", "paytv-batch", "studio-batch", "docu-batch",
           "meridian")

# Re-exported from the schema rather than restated, so a replay can never
# measure against a default the pipeline no longer uses.
MIN_OCCURRENCE_GAP_SEC = DEFAULT_MIN_OCCURRENCE_GAP_SEC
MIN_RECURRENCE_COUNT = DEFAULT_MIN_RECURRENCE_COUNT

Asset = namedtuple("Asset", "name meta by_id s_by_id ordered")


def group_key(g):
    """Threshold-stable identity for a group: its sorted member shot ids."""
    return tuple(sorted(s["shot_id"] for s in g["shots"]))


def distances(vectors):
    """Full symmetric cosine-distance matrix with a zeroed diagonal.

    Accepts any (n, dim) array of centroids; normalises defensively because
    published centroids are means of frame embeddings and are not unit norm.
    """
    V = np.asarray(vectors, dtype=np.float32)
    V = V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-8)
    D = np.clip(1.0 - V @ V.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return D


def shot_distances(asset, shot_ids):
    """Distance matrix over an arbitrary list of shots, in the given order."""
    return distances([asset.by_id[i] for i in shot_ids])


def group_distances(asset, group):
    """Distance matrix over one group's members, in the group's own order."""
    return shot_distances(asset, [s["shot_id"] for s in group["shots"]])


def occurrence_clusters(shots, gap=MIN_OCCURRENCE_GAP_SEC):
    """Gate 9.1's clusters in time order, grouped by PRODUCTION's labelling.

    No replica: the split comes from `lib.recurrence.occurrence_labels`, so a
    change to the gate cannot leave the replays measuring the old rule. This
    is the whole reason `detect_recurrence` was split into a pipeline — three
    copies of the rule used to live in the harness and two of them had drifted
    on whether the running end is the maximum so far or the previous member's.
    """
    ss = sorted(shots, key=lambda s: s["start_time"])
    spans = [(s["start_time"], s["end_time"]) for s in ss]
    cl = []
    for shot, label in zip(ss, occurrence_labels(spans, gap)):
        if label == len(cl):
            cl.append([])
        cl[label].append(shot)
    return cl


def occurrence_count(shots, gap=MIN_OCCURRENCE_GAP_SEC):
    """Number of temporally separated occurrences (gate 9.1)."""
    return count_occurrences(
        sorted((s["start_time"], s["end_time"]) for s in shots), gap)


def runs_in(batch):
    """(name, shots_path, centroids_path) for a batch dir or a flat dir.

    A run named in the manifest but missing artefacts is reported, never
    skipped silently — film_a is in `studio-batch.txt` but timed out upstream,
    and a silent skip reads as 14 assets measured when it was 13.
    """
    manifest = os.path.join(batch, "runs.txt")
    if os.path.exists(manifest):
        names = [l.split()[0] for l in open(manifest) if l.strip()]
    else:
        names = [os.path.basename(p)[:-len(".shots.json")]
                 for p in sorted(glob.glob(os.path.join(batch,
                                                        "*.shots.json")))]
    for n in names:
        shots = os.path.join(batch, f"{n}.shots.json")
        if not os.path.exists(shots):
            print(f"# SKIP {n} - no artefacts (run failed upstream)")
            continue
        yield n, shots, os.path.join(batch, f"{n}.centroids.npy")


def workflow_ids(batch):
    """{run_name: workflow_id} from a batch manifest, for scripts that need to
    fetch further artefacts from S3. Empty for a flat directory."""
    manifest = os.path.join(batch, "runs.txt")
    if not os.path.exists(manifest):
        return {}
    return {l.split()[0]: l.split()[1]
            for l in open(manifest) if len(l.split()) > 1}


def load_asset(name, shots_path, cent_path):
    """One Asset, with centroids attached to the shot dicts in place."""
    meta = json.load(open(shots_path))
    cent = np.load(cent_path)
    for s, v in zip(meta["shots"], cent):
        s["centroid_embedding"] = v.tolist()
    return Asset(
        name=name, meta=meta,
        by_id={s["shot_id"]: v for s, v in zip(meta["shots"], cent)},
        s_by_id={s["shot_id"]: s for s in meta["shots"]},
        ordered=sorted(meta["shots"], key=lambda s: s["start_time"]))


def load(batches=BATCHES):
    """Every usable asset across the named batches."""
    return [load_asset(n, sp, cp)
            for b in batches for n, sp, cp in runs_in(b)]


def candidates(asset, **cfg):
    """`detect_recurrence` on one asset — the shipped code path, shipped data."""
    return detect_recurrence(
        shots=asset.meta["shots"], cfg=cfg,
        actual_duration_sec=asset.meta["actual_duration_sec"])


def labelled_groups(batches=BATCHES, **cfg):
    """Yield (label, asset, group) for every published group.

    Labels, in lookup order:

      FURN    a verified genuine card (TRUE_CARDS), or labelled FURN by the
              2026-08-21 published-pool audit (AUDIT_LABELS, exact
              shot-id-set identity)
      SET     recurring location/camera, per the audit
      TMPL    recurring branded graphic template re-composited per airing,
              per the audit
      PROMO   re-aired package, per the audit
      then    CULLED_LABELS carry-over — the pre-swap audit keyed by
              (run, member count), applied only where a group's identity
              happens to survive the swap
      kept    published, never labelled — unknown rather than correct
    """
    for asset in load(batches):
        want = cards_for(asset.name)
        for g in candidates(asset, **cfg):
            k = group_key(g)
            if any(set(k) == s for _, s in want):
                lab = "FURN"
            else:
                lab = (label_audited(asset.name, k)
                       or label_culled(asset.name, g["recurrence_count"],
                                       g["occurrence_count"])
                       or "kept")
            yield lab, asset, g
