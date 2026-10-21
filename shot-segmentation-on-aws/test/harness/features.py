#!/usr/bin/env python3
"""Which measurable property separates genuine furniture from a recurring set?

This is the feature study for gate 9.3, and it replaces four scripts that each
answered a slice of the same question over a subset of the corpus and a subset
of the features (`separate.py`, `verdict.py`, `rule_test.py`, `guardrails.py`).
Their findings are preserved below; the reason to merge them is that they
disagreed on which assets they covered (2 batches vs 4) and therefore on how
many genuine cards existed at all.

The question is NOT "what value should `max_recurrence_count` take". It is
whether member count is the right quantity, and the answer measured here is no:

  * `n`, the shipped ceiling's feature, ranks NINTH of fourteen (AUC 0.799),
    and the shipped value of 8 keeps only 6 of the 12 genuine furniture groups.
    Genuine cards run n=3 to n=24.
  * `coverage` — member seconds as a fraction of programme seconds — ranks
    FIRST (0.953) and is duration-normalised by construction, so unlike `n` it
    does not move with programme format.
  * `ent_range` — mean per-shot (entropy_max - entropy_min), already in every
    shots.json and previously read only as a floor in 9.4 — ranks SECOND
    (0.948), and in the informative direction: FURN high (animated idents and
    wipes swing entropy inside one shot), SET low (a locked-off anchor camera
    does not).
  * NO feature achieves clean separation. The `gap?` column is `-` fourteen
    times out of fourteen, which is the finding: no scalar threshold on any of
    these is safe to *drop* on, whatever value it takes.

Duration, tested directly because the previously proposed ceiling was linear in
runtime: Pearson r(hours, n) over the 12 FURN groups is only +0.348. Runtime
barely predicts a card's member count. What runtime is good for is the
denominator, not a multiplier.

Dating: the numbers above were measured on the PRE-SWAP candidate space
(single-linkage grouping at 0.05, labels from the 25-group culled audit). The
2026-08-21 published-pool visual audit (truth.AUDIT_LABELS, 53 groups, labels
corrected by the user's review of the sheets) refreshed the ground truth. The
re-run on 15 FURN / 40 SET:

  * ent_range leads, AUC 0.945. coverage FALLS from 0.953 to 0.833, and the
    reason is a finding, not noise: furniture is TWO classes with opposite
    coverage. Cards/idents/wipes sit at coverage <= 0.0066; the ad-break
    COUNTDOWN SLATES (a teaser plate + "WILL RETURN IN N:NN", ~15 s per
    commercial break) sit at 0.018-0.021 — squarely set-like. Both audits
    initially mislabelled that class as re-aired promos from mid-frame
    thumbnails.
  * member count stays near-noise: AUC 0.612 (was 0.799 pre-swap). The
    retired ceiling's feature is the weakest signal measured.
  * still no clean separation anywhere (gap? is `-` for all 20), and the
    break-slate class defeats every scalar here by construction — one sits
    at ent_range 0.99 against the 1.0 line its siblings clear. Separating it
    needs the plate itself (layout/OCR of the countdown) or a frames look;
    that is what keeps these ranking signals, never drop thresholds.

Findings inherited from the four scripts this replaces:

  verdict.py    ceiling precision 16/25 = 64% — 6 definitive false culls, 3
                promos, 16 correct sets.
  separate.py   duration and density both overlap; no single shots.json feature
                beat member count on the 2-batch subset.
  rule_test.py  best fitted rule was `n<=8 OR (n <= ceil(15*hours) AND
                (occ_ratio>=0.80 OR dur_cv<=0.30))`. Superseded: its thresholds
                were fitted to 25 self-labelled groups, ~8 per parameter.
  guardrails.py REFUTES a minimum-gap floor between occurrences — furniture
                bottoms out at 1.09 min and sets at 1.01, so an 8-min floor
                kills 8/12 furniture and raising `min_occurrence_gap_sec` to
                8 min makes the drama_a card vanish. Retained below as the
                `gap_min` floor sweep so the refutation stays reproducible.

Usage — from a scratch dir built by setup_workdir.sh:
  python features.py                        # all four batches
  python features.py phase9-batch paytv-batch # the labelled subset only
"""
import logging
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # repo root
from corpus import (BATCHES, labelled_groups,       # noqa: E402 harness-local
                    occurrence_clusters, shot_distances)

logging.disable(logging.CRITICAL)

# Features that are duration-normalised or otherwise scale-free, and so are the
# only ones that can generalise across programme formats.
NORMALISED = {"occ_ratio", "dur_cv", "coverage", "span_frac", "gap_cv",
              "ent_range", "black_adj", "diam", "nxt_med", "ab_frac"}

KEYS = ("coverage", "ent_range", "occ_ratio", "dur_cv", "diam", "n_per_hr",
        "black_adj", "ent_mean", "n", "gap_cv", "nxt_med", "ab_frac",
        "mean_dur", "dur_med", "dur_max", "span_frac", "occ", "occ_per_hr",
        "gap_min", "gap_med")


def cv(xs):
    m = st.mean(xs)
    return st.pstdev(xs) / m if m else 0.0


def features(asset, g):
    """Every feature computable from shots.json plus centroids — no frame I/O."""
    sids = [s["shot_id"] for s in g["shots"]]
    mem = [asset.s_by_id[i] for i in sids]
    dur = [m["end_time"] - m["start_time"] for m in mem]
    total = asset.meta["actual_duration_sec"]
    hours = total / 3600.0

    cl = occurrence_clusters(mem)
    gaps = [cl[i + 1][0]["start_time"] - cl[i][-1]["end_time"]
            for i in range(len(cl) - 1)]

    pos = {s["shot_id"]: k for k, s in enumerate(asset.ordered)}
    # Reverse-angle probe: the shot immediately FOLLOWING each member. An
    # anchor desk alternates with one reverse angle, so the followers cluster
    # tightly; a card is followed by whatever scene comes next. Measured at
    # AUC 0.698 — notably weaker than the hypothesis predicted.
    nxt = [asset.ordered[pos[i] + 1]["shot_id"] for i in sids
           if pos[i] + 1 < len(asset.ordered)]
    if len(nxt) >= 2:
        DN = shot_distances(asset, nxt)
        iu = np.triu_indices(len(nxt), 1)
        nxt_med = float(np.median(DN[iu]))
    else:
        nxt_med = float("nan")

    non_content = {s["shot_id"] for s in asset.ordered
                   if s["shot_type"] != "content"}
    adj = sum(1 for i in sids
              if (pos[i] and asset.ordered[pos[i] - 1]["shot_id"] in non_content)
              or (pos[i] + 1 < len(asset.ordered)
                  and asset.ordered[pos[i] + 1]["shot_id"] in non_content))
    positions = [s["position"] for s in g["shots"]]

    return {
        "n": g["recurrence_count"],              # <- the shipped ceiling
        "occ": g["occurrence_count"],
        "occ_ratio": g["occurrence_count"] / g["recurrence_count"],
        "n_per_hr": g["recurrence_count"] / hours,
        "occ_per_hr": g["occurrence_count"] / hours,
        "mean_dur": st.mean(dur),
        "dur_med": st.median(dur),
        "dur_max": max(dur),
        "dur_cv": cv(dur),
        "coverage": sum(dur) / total,
        "span_frac": (max(s["end_time"] for s in mem)
                      - min(s["start_time"] for s in mem)) / total,
        "gap_min": (min(gaps) / 60.0) if gaps else float("nan"),
        "gap_med": (st.median(gaps) / 60.0) if gaps else float("nan"),
        "gap_cv": cv(gaps) if len(gaps) >= 2 else float("nan"),
        "ent_mean": st.mean([s.get("entropy_mean", float("nan"))
                             for s in mem]),
        "ent_range": st.mean([s.get("entropy_max", 0.0)
                              - s.get("entropy_min", 0.0) for s in mem]),
        "black_adj": adj / len(sids),
        "diam": g["max_distance"],
        "ab_frac": sum(1 for p in positions if p == "act_boundary")
        / len(positions),
        "nxt_med": nxt_med,
    }


RULES = {
    "n<=8  (SHIPPED)":            lambda f: f["n"] <= 8,
    "coverage<=0.016":            lambda f: f["coverage"] <= 0.016,
    "coverage<=0.010":            lambda f: f["coverage"] <= 0.010,
    "ent_range>=1.0":             lambda f: f["ent_range"] >= 1.0,
    "occ_ratio>=0.80":            lambda f: f["occ_ratio"] >= 0.80,
    "cov<=0.016 OR entr>=1.0":    lambda f: (f["coverage"] <= 0.016
                                             or f["ent_range"] >= 1.0),
    "cov<=0.016 AND dur_cv<=0.6": lambda f: (f["coverage"] <= 0.016
                                             and f["dur_cv"] <= 0.6),
    "n<=8 OR cov<=0.016":         lambda f: (f["n"] <= 8
                                             or f["coverage"] <= 0.016),
    # the rule rule_test.py recommended, kept so the supersession is visible
    "n<=8 OR (ratio & cv) [old]":  lambda f: (
        f["n"] <= 8 or (f["occ_ratio"] >= 0.80 and f["dur_cv"] <= 0.30)),
}


def auc(fv, sv):
    """Rank separation, symmetric in direction. 0.5 = useless, 1.0 = perfect."""
    wins = sum((a > b) + 0.5 * (a == b) for a in fv for b in sv)
    return max(wins, len(fv) * len(sv) - wins) / (len(fv) * len(sv))


def best_threshold(fv, sv):
    """Best single threshold in either direction, by accuracy."""
    best = (0.0, None, None)
    for t in sorted(set(fv) | set(sv)):
        for d in ("<=", ">="):
            keep = (lambda x: x <= t) if d == "<=" else (lambda x: x >= t)
            tp = sum(1 for x in fv if keep(x))
            tn = sum(1 for x in sv if not keep(x))
            acc = (tp + tn) / (len(fv) + len(sv))
            if acc > best[0]:
                best = (acc, f"{d}{t:.4g}", (tp, len(fv), tn, len(sv)))
    return best


def main(batches):
    rows = [(lab, asset.name, features(asset, g))
            for lab, asset, g in labelled_groups(batches)]
    by = {k: [f for l, _, f in rows if l == k]
          for k in ("FURN", "SET", "PROMO", "TMPL", "kept")}
    print("groups=%d  " % len(rows)
          + "  ".join(f"{k}={len(v)}" for k, v in by.items()))
    F, S = by["FURN"], by["SET"]
    if not F or not S:
        print("\nno labelled FURN/SET in these batches — "
              "the labelled subset is phase9-batch + paytv-batch")
        return

    print(f"\n{'feature':>10} {'':1} | {'FURN range':>21} | {'SET range':>21} "
          f"| {'gap?':>4} | {'AUC':>5} | best single threshold")
    print("-" * 108)
    for k in KEYS:
        fv = sorted(x[k] for x in F if x[k] == x[k])
        sv = sorted(x[k] for x in S if x[k] == x[k])
        if not fv or not sv:
            continue
        clean = "YES" if (min(fv) > max(sv) or max(fv) < min(sv)) else "-"
        acc, rule, hit = best_threshold(fv, sv)
        print(f"{k:>10} {'*' if k in NORMALISED else ' '} | "
              f"{min(fv):9.4f} ..{max(fv):9.4f} | "
              f"{min(sv):9.4f} ..{max(sv):9.4f} | {clean:>4} | "
              f"{auc(fv, sv):.3f} | acc={acc:.2f} {k}{rule:<9} "
              f"FURN {hit[0]}/{hit[1]}, SET killed {hit[2]}/{hit[3]}")
    print("  * = duration-normalised / scale-free")

    print(f"\n{'candidate keep-rule':>28} {'FURN kept':>10} {'SET killed':>11} "
          f"{'PROMO kept':>11} {'unlabelled survivors':>21}")
    print("-" * 88)
    for nm, fn in RULES.items():
        print(f"{nm:>28} {sum(map(fn, F)):>6}/{len(F):<3} "
              f"{sum(1 for f in S if not fn(f)):>7}/{len(S):<3} "
              f"{sum(map(fn, by['PROMO'])):>7}/{len(by['PROMO']):<3} "
              f"{sum(map(fn, by['kept'])):>15}/{len(by['kept']):<4}")

    print("\n--- minimum-gap floor between occurrences (guardrails.py's "
          "refutation, retained)")
    for floor in (2, 3, 4, 5, 6, 8):
        fk = sum(1 for f in F
                 if f["gap_min"] != f["gap_min"] or f["gap_min"] >= floor)
        sk = sum(1 for f in S
                 if f["gap_min"] == f["gap_min"] and f["gap_min"] < floor)
        print(f"    gap >= {floor} min: FURN kept {fk}/{len(F)}, "
              f"SET killed {sk}/{len(S)}")

    print("\n--- does programme duration predict a genuine card's member "
          "count?")
    print(f"{'run':>18} {'hours':>6} {'n':>4} {'occ':>4} {'n/hr':>6} "
          f"{'coverage':>9} {'ent_range':>10} {'gap_cv':>7}")
    for lab, name, f in sorted(rows, key=lambda r: r[2]["n_per_hr"]):
        if lab == "FURN":
            print(f"{name:>18} {f['n'] / f['n_per_hr']:>6.2f} {f['n']:>4} "
                  f"{f['occ']:>4} {f['n_per_hr']:>6.2f} {f['coverage']:>9.5f} "
                  f"{f['ent_range']:>10.3f} {f['gap_cv']:>7.3f}")
    hrs = [f["n"] / f["n_per_hr"] for f in F]
    if len(hrs) > 2:
        print(f"\n    Pearson r(hours, n) over FURN = "
              f"{np.corrcoef(hrs, [f['n'] for f in F])[0, 1]:+.3f}  "
              f"(n={len(hrs)})")


if __name__ == "__main__":
    main(sys.argv[1:] or list(BATCHES))
