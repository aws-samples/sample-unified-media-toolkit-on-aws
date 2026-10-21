#!/usr/bin/env python3
"""K-trust acceptance replay — offline referee for the binder / floor
/ veto / reconcile mechanisms.

Replays production _cluster_tracks from replay caches (replay_cache.py
+ the live-label augmentation) and checks per asset:
  - K vs the recorded live K (pre-package production output);
  - ear-labeled FRAGMENT pairs merge;
  - ear-labeled DISTINCT speakers stay apart;
  - mechanism diagnostics (binds / floors / vetoes / reconciles).

Cluster ids in the manifests are LIVE run ids (e2e-mixwav-20260929);
descriptions are generic by repo rule. Sentences are matched between
live and replay by sentence identity (dyn_chunk_idx:sentence_idx).

Run:  python3 test/replay_acceptance.py CACHE_DIR [asset ...]
"""
import importlib.util
import os
import pickle  # nosec B403
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lib.powerset_guides import _bind_across_chunks  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "clu_state", os.path.join(HERE, "..", "states",
                              "global_speakers_clustering",
                              "lambda_function.py"))
clu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clu)

# The ear-labeled manifests live NEXT TO THE CACHES, not here: they
# name real corpus assets (repo rule — no titles in repo content) and
# are corpus-specific anyway. acceptance_manifest.json shape:
#   {"must_merge":      {asset: [[live_id, live_id], ...]},   # fragments of ONE voice
#    "must_stay_apart": {asset: [[live_id, live_id], ...]}}   # confirmed different people
MUST_MERGE: dict = {}
MUST_STAY_APART: dict = {}


def load_manifest(cache_dir):
    global MUST_MERGE, MUST_STAY_APART
    path = os.path.join(cache_dir, "acceptance_manifest.json")
    if not os.path.exists(path):
        print(f"WARNING: no {path} — K sanity only, no merge/apart checks")
        return
    import json
    doc = json.load(open(path))
    MUST_MERGE = {a: [tuple(p) for p in ps]
                  for a, ps in (doc.get("must_merge") or {}).items()}
    MUST_STAY_APART = {a: [tuple(p) for p in ps]
                       for a, ps in (doc.get("must_stay_apart") or {}).items()}


def replay(doc):
    """Mirror the NEW load_guides against EXISTING artifacts: the
    embeddings docs' chain ids used the OLD lexical-rank namespace, so
    guides here use the same namespace; the binder gets TRUE chunk
    adjacency and its rename map is applied to the segs."""
    names = sorted(f"powerset_{i}.json" for i in doc["powerset"])
    rank = {int(n.rsplit("_", 1)[1].split(".")[0]): r
            for r, n in enumerate(names)}
    per_chunk = {}
    merged = {"tracks": [], "co_active_pairs": [], "co_active_secs": [],
              "clique_floor": 0, "local_speaker_floor": 0}
    for idx, d in doc["powerset"].items():
        off = (rank[idx] + 1) * 1_000_000
        tracks = [{"id": t["id"] + off, "segs": t["segs"]}
                  for t in d.get("tracks") or []]
        per_chunk[idx] = tracks
        merged["tracks"] += tracks
        pairs = d.get("co_active_pairs") or []
        merged["co_active_pairs"] += [[a + off, b + off]
                                      for a, b in pairs]
        # mirror load_guides: pre-co_active_secs artifacts read each
        # link at the 0.25 s emission-rule lower bound
        merged["co_active_secs"] += (d.get("co_active_secs")
                                     or [0.25] * len(pairs))
        merged["clique_floor"] = max(merged["clique_floor"],
                                     d.get("clique_floor") or 0)
        merged["local_speaker_floor"] = max(
            merged["local_speaker_floor"],
            d.get("local_speaker_floor") or 0)
    _bind_across_chunks(merged, per_chunk)   # candidates only;
    segs = list(doc["segs"])                 # gated application is
    cfg = {"max_k": 50}                      # _cluster_tracks' job
    # mirror the handler: small-N guard + degenerate rerun
    viable, n_chains = clu._tracks_viable(segs, cfg)
    if viable:
        labels, _, method, diag = clu._cluster_tracks(
            segs, merged, doc["high"], cfg, assign_margin=0.05)
        if diag.get("selection_reason") == "fallback_no_valid_candidates":
            viable = False
    if not viable:
        import numpy as np
        M = np.array([s2["embedding"] for s2 in doc["high"]],
                     dtype=np.float32)
        total = sum(s2["end"] - s2["start"] for s2 in doc["high"])
        labels, _, method, diag = clu._select_optimal_clustering(
            M, doc["high"], cfg, total)
        diag = dict(diag)
        diag["smalln_track_fallback"] = n_chains
    return labels, method, diag


def check(asset, doc):
    labels, method, diag = replay(doc)
    high = doc["high"]
    live = doc["live"]["sentence_speakers"]
    # live speaker -> replay label votes, via sentence identity
    votes = {}
    for i, s in enumerate(high):
        if labels[i] < 0:
            continue
        sid = live.get(f"{s['dyn_chunk_idx']}:{s['sentence_idx']}")
        if sid and sid != "UNKNOWN":
            votes.setdefault(sid, Counter())[int(labels[i])] += 1
    rep = {sid: c.most_common(1)[0][0] for sid, c in votes.items()}
    K = len({l for l in labels.tolist() if l >= 0})
    ok, fails = [], []
    for a, b in MUST_MERGE.get(asset, []):
        if a in rep and b in rep:
            (ok if rep[a] == rep[b] else fails).append(f"merge {a}~{b}")
        else:
            fails.append(f"merge {a}~{b} (unmapped)")
    for a, b in MUST_STAY_APART.get(asset, []):
        if a in rep and b in rep:
            (ok if rep[a] != rep[b] else fails).append(f"apart {a}|{b}")
        else:
            fails.append(f"apart {a}|{b} (unmapped)")
    mech = " ".join(f"{k}={diag.get(k)}" for k in
                    ("xchunk_binds", "local_speaker_floor",
                     "clique_floor", "n_post_splits", "n_reconciled",
                     "smalln_track_fallback") if diag.get(k))
    status = "FAIL" if fails else "ok"
    print(f"{asset:20s} K {doc['live']['K']:>3d} -> {K:<3d} "
          f"[{status}] {mech}")
    for f in fails:
        print(f"    FAIL: {f}")
    for o in ok:
        print(f"    pass: {o}")
    return not fails


def main():
    cache_dir = sys.argv[1]
    load_manifest(cache_dir)
    only = set(sys.argv[2:])
    good = True
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith(".pkl"):
            continue
        asset = fn[:-4]
        if only and asset not in only:
            continue
        doc = pickle.load(open(os.path.join(cache_dir, fn), "rb"))  # nosec B301  # nosemgrep: python.lang.security.deserialization.avoid-pickle
        if "live" not in doc:
            print(f"{asset}: cache not augmented with live labels — skip")
            continue
        good &= check(asset, doc)
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
