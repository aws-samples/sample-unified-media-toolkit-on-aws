#!/usr/bin/env python3
"""What linkage threshold does complete-linkage clustering need?

Single-linkage at 0.05 and complete-linkage at 0.05 are not interchangeable:
tightening the criterion from "some pair within t" to "every pair within t"
makes the SAME t stricter, so a threshold calibrated for one is wrong for the
other. This sweeps t and scores each value on whether the six verified true
cards (truth.py) come out INTACT, SPLIT, or LOST.

Since the 2026-08-21 swap, production groups ARE complete-linkage clusters, so
the refine step below is idempotent and this script measures the shipped path
directly. Result on the 27-asset corpus, post-swap:

     t  groups  >8mem  maxn  INTACT  SPLIT  LOST  detail
 0.040     332     18    37       5      0     1  LOST:markets_a:g3
 0.050     366     26    46       5      0     1  LOST:markets_a:g3
 0.060     387     36    52       5      0     1  LOST:markets_a:g3
 0.065     397     39    52       6      0     0
 0.070     415     41    52       6      0     0
 0.080     429     54    52       5      1     0  SPLIT:drama_e:g0

The pre-swap run that set the 0.07 default read 364/384/393/411/425 on rows
0.050-0.080 — up to 4 fewer groups per row, with the card verdicts IDENTICAL at
every threshold. The delta is a defect in the old measurement method, not in
either clustering: pre-swap this script had to simulate complete linkage by
refining an enclosing single-linkage pass, and although that pass ran at
`min_recurrence_count: 2` precisely because a subset can have MORE occurrences
than its parent (intervening members bridge the 60 s gaps), a parent whose
members formed ONE bridged chain (occurrence count 1) was still pruned before
refinement could split it. Production now gates each final cluster directly,
so those rescued clusters exist.

The safe window is (0.0634, ~0.075]: markets_a:g3 has diameter 0.0634 so it
cannot form below that, and drama_e:g0 splits at 0.080. **0.07** is the
shipped default — mid-window, where 0.065 sits only 0.0016 above the failure
point.

The second finding is the one to keep in view: `maxn` stays 52 at EVERY
threshold, and `>8mem` grows from 25 to 40 between 0.05 and 0.07. There is a
genuinely tight 52-member cluster (the markets_a anchor desk) whose diameter is
already inside 0.05, so complete linkage cannot touch it. Chaining and the
recurring-set ceiling are independent problems; fixing the first makes the
second's arithmetic worse, not better.

Usage — from a scratch dir built by setup_workdir.sh:
  python sweep.py                                  # all four batches
  python sweep.py phase9-batch paytv-batch
"""
import logging
import sys

from corpus import (BATCHES, MIN_RECURRENCE_COUNT,   # noqa: E402 harness-local
                    candidates, group_distances, load, occurrence_count)
from linkage import complete_linkage                 # noqa: E402
from truth import cards_for                          # noqa: E402

logging.disable(logging.CRITICAL)
MINC = MIN_RECURRENCE_COUNT
THRESHOLDS = (0.040, 0.050, 0.060, 0.065, 0.070, 0.080)


def refine_at(asset, t):
    """Post-gate group memberships for one asset at threshold t."""
    groups = []
    for g in candidates(asset, max_recurrence_threshold=t,
                        min_recurrence_count=2):
        sids = [s["shot_id"] for s in g["shots"]]
        D = group_distances(asset, g)
        for c in complete_linkage(D.tolist(), t):
            sub = [asset.s_by_id[sids[i]] for i in c]
            if len(sub) < MINC or occurrence_count(sub) < MINC:
                continue
            groups.append(sorted(s["shot_id"] for s in sub))
    return groups


def main(batches):
    assets = load(batches)
    print(f"{'t':>6} {'groups':>7} {'>8mem':>6} {'maxn':>5} {'INTACT':>7} "
          f"{'SPLIT':>6} {'LOST':>5}  detail")
    print("-" * 92)
    for t in THRESHOLDS:
        total = big = maxn = 0
        intact, split, lost = [], [], []
        for asset in assets:
            found = refine_at(asset, t)
            for f in found:
                total += 1
                maxn = max(maxn, len(f))
                if len(f) > 8:
                    big += 1
            for k, want in cards_for(asset.name):
                if any(set(f) == want for f in found):
                    intact.append(k)
                elif any(len(want & set(f)) >= MINC for f in found):
                    split.append(k)
                else:
                    lost.append(k)
        det = ("LOST:" + ",".join(lost)) if lost else ""
        det += ("  SPLIT:" + ",".join(split)) if split else ""
        print(f"{t:>6.3f} {total:>7} {big:>6} {maxn:>5} {len(intact):>7} "
              f"{len(split):>6} {len(lost):>5}  {det}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(BATCHES))
