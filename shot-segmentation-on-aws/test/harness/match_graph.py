#!/usr/bin/env python3
"""Is 9.6's EXISTENTIAL rule the problem, not the group size?

Build the member-match graph per group: nodes = members with hashable frames,
edge a-b iff some frame pair of a,b satisfies the shipped 9.6 predicate. A
same-RENDER group should be a near-complete graph (every instance matches every
other); a same-SET group confirmed by one lucky pair should be nearly empty.

Reports edge density and the largest mutually-matching component (max clique
approximated by the largest connected component AND by greedy clique)."""
import json, sys, glob, os, logging, itertools
import numpy as np
from PIL import Image
from corpus import BATCHES                       # noqa: E402 (repo root on path)
from truth import TRUE_CARDS                     # noqa: E402
from lib import frame_hash
from lib.recurrence import _is_black_dominated
from lib.input_schema import (RECURRENCE_HASH_MAX_FRAMES,
                              RECURRENCE_HASH_MAX_GLOBAL,
                              RECURRENCE_HASH_MAX_MEDIAN,
                              RECURRENCE_HASH_NOISE_THRESHOLD)
logging.disable(logging.CRITICAL)

def lumas(batch, run, sid):
    d = f"{batch}/frames/{run}/shot_{sid:04d}"
    out = []
    for p in sorted(glob.glob(f"{d}/*.jpg")):
        fn = int(os.path.basename(p).split("_")[1].split(".")[0])
        out.append((fn, np.asarray(Image.open(p).convert("L"))))
    return sorted(out)

def analyse(batch, run, sids):
    members, churn = [], []
    for sid in sids:
        hs = []
        for fn, lu in lumas(batch, run, sid):
            if _is_black_dominated(lu):
                continue
            hs.append(frame_hash.frame_hashes(lu))
            if len(hs) >= RECURRENCE_HASH_MAX_FRAMES:
                break
        if hs:
            members.append((sid, hs)); churn.extend(hs)
    noisy = set(frame_hash.classify_noisy_patches(churn, RECURRENCE_HASH_NOISE_THRESHOLD))

    def match(ha, hb):
        if frame_hash.hamming(ha["global"], hb["global"]) <= RECURRENCE_HASH_MAX_GLOBAL:
            return True
        st = sorted(frame_hash.hamming(pa, pb)
                    for p, (pa, pb) in enumerate(zip(ha["patches"], hb["patches"]))
                    if p not in noisy)
        return bool(st) and st[len(st) // 2] <= RECURRENCE_HASH_MAX_MEDIAN

    m = len(members)
    adj = {i: set() for i in range(m)}
    for a, b in itertools.combinations(range(m), 2):
        if any(match(ha, hb) for ha in members[a][1] for hb in members[b][1]):
            adj[a].add(b); adj[b].add(a)
    edges = sum(len(v) for v in adj.values()) // 2
    poss = m * (m - 1) // 2
    # greedy max clique: repeatedly seed from the highest-degree node
    best = []
    for seed in sorted(range(m), key=lambda i: -len(adj[i])):
        clique = [seed]
        for c in sorted(adj[seed], key=lambda i: -len(adj[i])):
            if all(c in adj[x] for x in clique):
                clique.append(c)
        if len(clique) > len(best):
            best = clique
    # largest connected component
    seen, comps = set(), []
    for i in range(m):
        if i in seen: continue
        stack, comp = [i], []
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u); stack.extend(adj[u] - seen)
        comps.append(len(comp))
    return {"hashable": m, "edges": edges,
            "density": (edges / poss) if poss else 0.0,
            "clique": len(best), "lcc": max(comps) if comps else 0}

def batch_of(run):
    """Which batch dir holds this run's frames. Cards are identified by shot-id
    set from truth.py, so nothing here reads a published group_id."""
    for b in BATCHES:
        if os.path.isdir(f"{b}/frames/{run}"):
            return b
    raise SystemExit(f"no fetched frames for {run} - run fetch_frames.sh first")

rows = []
for card in TRUE_CARDS.values():
    run, sids = card["name"], card["shots"]
    rows.append(("TRUE_CARD", run, len(sids),
                 analyse(batch_of(run), run, sids)))
for g in json.load(open("audit_groups.json")):
    rows.append(("CULLED", g["run"], g["n"], analyse(g["batch"], g["run"], g["shots"])))

print(f"{'label':<10} {'run':<16} {'n':>3} {'hash':>4} {'edges':>5} {'density':>7} {'clique':>6} {'lcc':>4}")
print("-" * 66)
for lab, run, n, a in sorted(rows, key=lambda r: (r[0], -r[3]["density"])):
    print(f"{lab:<10} {run:<16} {n:>3} {a['hashable']:>4} {a['edges']:>5} "
          f"{a['density']:>7.3f} {a['clique']:>6} {a['lcc']:>4}")
for lab in ("TRUE_CARD", "CULLED"):
    d = sorted(a["density"] for l, _, _, a in rows if l == lab)
    c = sorted(a["clique"] for l, _, _, a in rows if l == lab)
    print(f"\n{lab}: density min={d[0]:.3f} med={d[len(d)//2]:.3f} max={d[-1]:.3f}"
          f" | clique min={c[0]} med={c[len(c)//2]} max={c[-1]}")
json.dump([[l, r, n, a] for l, r, n, a in rows], open("match_graph.json", "w"), indent=1)
