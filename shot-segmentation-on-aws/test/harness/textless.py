#!/usr/bin/env python3
"""Textless elements: clean (no burned-in text) copies of in-programme shots
appended AFTER end credits for localisation. BBC nomenclature marks them --
'_dualpurpose_' deliveries carry them, '_texted_' should not.

Test: for shots in the tail of the file, does a near-duplicate exist earlier?
A textless BLOCK should show as a contiguous run of tail shots each matching an
earlier shot, separated from the programme by black."""
import json, sys
import numpy as np
from corpus import distances                    # noqa: E402 (harness-local)
TH = 0.05
for line in open("runs.txt"):
    name = line.split()[0]
    try:
        meta = json.load(open(f"{name}.shots.json"))
        cent = np.load(f"{name}.centroids.npy")
    except Exception:
        continue
    shots = meta["shots"]; dur = meta["actual_duration_sec"]
    D = distances(cent)   # tail and body are disjoint, so the diagonal is
                          # never consulted and needs no sentinel
    # candidate textless: a shot in the last 20% whose nearest earlier match
    # (in the first 80%) is within TH
    cut = 0.80 * dur
    tail = [i for i, s in enumerate(shots) if s["start_time"] >= cut]
    body = [i for i, s in enumerate(shots) if s["start_time"] < cut]
    hits = []
    for i in tail:
        if not body: break
        j = min(body, key=lambda b: D[i][b])
        if D[i][j] <= TH:
            hits.append((i, j, float(D[i][j])))
    # longest contiguous run of tail shots that are hits
    hs = {i for i, _, _ in hits}
    best = cur = 0; run_start = None; bs = None
    for i in tail:
        if i in hs:
            if cur == 0: run_start = i
            cur += 1
            if cur > best: best, bs = cur, run_start
        else:
            cur = 0
    frac_black = sum(1 for i in tail if shots[i]["shot_type"] == "black") / max(len(tail), 1)
    print(f"{name:<16} tail_shots={len(tail):>4} near-dup={len(hits):>4} "
          f"longest_run={best:>3} black_frac={frac_black:.2f}"
          + (f"  block@shot {bs}-{bs+best-1}" if best >= 3 else ""))
