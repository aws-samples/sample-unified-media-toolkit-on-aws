#!/usr/bin/env python3
"""For each rejected group, report the BEST cross-member pair statistics
against the shipped thresholds, so a rejection can be read as decisive or
marginal. Uses the same code path the lambda uses."""
import sys, glob, os
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # repo root
from lib import frame_hash
from lib.recurrence import _is_black_dominated
from lib.input_schema import (RECURRENCE_HASH_MAX_FRAMES,
                              RECURRENCE_HASH_MAX_GLOBAL,
                              RECURRENCE_HASH_MAX_MEDIAN,
                              RECURRENCE_HASH_NOISE_THRESHOLD)

GROUPS = {
 ("news_b", "A"): [517, 520, 526, 537, 539, 771],
 ("news_b", "B"): [604, 1145, 1230],
 ("news_b", "C"): [988, 1286, 1289],
 ("news_a", "D"): [45, 67, 1089, 1095],
 ("news_a", "E"): [1085, 1103, 1116],
 ("markets_a", "F"): [4, 156, 284],
 ("markets_a", "G"): [13, 292, 337],
 ("markets_a", "H"): [41, 109, 412],
}

def lumas(run, sid):
    d = f"frames/{run}/shot_{sid:04d}"
    out = []
    for p in sorted(glob.glob(f"{d}/*.jpg")):
        fn = int(os.path.basename(p).split("_")[1].split(".")[0])
        out.append((fn, np.asarray(Image.open(p).convert("L"))))
    return sorted(out)

for (run, tag), sids in GROUPS.items():
    members, churn = [], []
    for sid in sids:
        hs = []
        for fn, lu in lumas(run, sid):
            if _is_black_dominated(lu):
                continue
            hs.append(frame_hash.frame_hashes(lu))
            if len(hs) >= RECURRENCE_HASH_MAX_FRAMES:
                break
        if hs:
            members.append((sid, hs)); churn.extend(hs)
    noisy = set(frame_hash.classify_noisy_patches(
        churn, RECURRENCE_HASH_NOISE_THRESHOLD))
    best_g, best_m, best_pair = 99, 99, None
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            for ha in members[a][1]:
                for hb in members[b][1]:
                    g = frame_hash.hamming(ha["global"], hb["global"])
                    st = sorted(frame_hash.hamming(pa, pb)
                                for p, (pa, pb) in enumerate(
                                    zip(ha["patches"], hb["patches"]))
                                if p not in noisy)
                    m = st[len(st)//2] if st else 99
                    if (min(g/RECURRENCE_HASH_MAX_GLOBAL, m/RECURRENCE_HASH_MAX_MEDIAN)
                            < min(best_g/RECURRENCE_HASH_MAX_GLOBAL,
                                  best_m/RECURRENCE_HASH_MAX_MEDIAN)):
                        best_g, best_m = g, m
                        best_pair = (members[a][0], members[b][0])
    # within-member baseline: adjacent frames of the FIRST member
    within = [frame_hash.hamming(members[0][1][i]["global"],
                                 members[0][1][i+1]["global"])
              for i in range(len(members[0][1]) - 1)] if members else []
    print(f"{run:<15} {tag} members={len(members)}/{len(sids)} "
          f"noisy={sorted(noisy)} best_global={best_g} (<= {RECURRENCE_HASH_MAX_GLOBAL}) "
          f"best_median={best_m} (<= {RECURRENCE_HASH_MAX_MEDIAN}) "
          f"pair={best_pair} within_member_global={within}")
