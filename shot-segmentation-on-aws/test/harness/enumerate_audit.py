#!/usr/bin/env python3
"""Enumerate published groups for the VISUAL AUDIT, with batch/run/workflow_id
and the discriminating features, so contact sheets can be fetched and built for
exactly the groups a human (or VLM) should eyeball. Writes `audit_groups.json`,
consumed by sheet_all.py; frames come via fetch_frames.sh.

This replaces enumerate_culled.py: since 9.3 annotates instead of dropping
(2026-08-21) there is no culled set to enumerate — the groups that need eyes
are the PUBLISHED ones, which were never audited (the old ground-truth pass
covered only the 25 groups the ceiling used to delete). Default selection is
the stratum most likely to be mislabelled by any rule: groups over the
reference ceiling, plus groups carrying the set signature
(coverage > 0.016 AND ent_range < 1.0 — the inverse of the rule that kept
12/12 verified furniture). Pass --all to enumerate everything for sampling.
"""
import json
import logging
import statistics as st
import sys

from corpus import (BATCHES, candidates, group_key,  # noqa: E402 harness-local
                    load_asset, runs_in, workflow_ids)
from lib.input_schema import DEFAULT_MAX_RECURRENCE_COUNT   # noqa: E402

logging.disable(logging.CRITICAL)


def main(batches, select_all=False):
    out, total = [], 0
    for batch in batches:
        wids = workflow_ids(batch)
        for name, sp, cp in runs_in(batch):
            asset = load_asset(name, sp, cp)
            dur_total = asset.meta["actual_duration_sec"]
            for g in candidates(asset):
                sids = [s["shot_id"] for s in g["shots"]]
                mem = [asset.s_by_id[i] for i in sids]
                dur = [round(m["end_time"] - m["start_time"], 2) for m in mem]
                cov = sum(dur) / dur_total
                entr = st.mean(m.get("entropy_max", 0.0)
                               - m.get("entropy_min", 0.0) for m in mem)
                over = g["recurrence_count"] > DEFAULT_MAX_RECURRENCE_COUNT
                set_sig = cov > 0.016 and entr < 1.0
                if not (select_all or over or set_sig):
                    continue
                out.append({
                    "batch": batch, "run": name, "wid": wids.get(name),
                    "group_id": g["group_id"], "shots": sids,
                    "n": g["recurrence_count"], "occ": g["occurrence_count"],
                    "maxd": g["max_distance"],
                    "over_ceiling": over, "set_sig": set_sig,
                    "coverage": round(cov, 5), "ent_range": round(entr, 3),
                    "pos": [s["position"] for s in g["shots"]],
                    "dur": dur})
                total += len(sids)
    json.dump(out, open("audit_groups.json", "w"), indent=1)
    print(f"{len(out)} groups to audit, {total} member shots to fetch")
    for g in out:
        tag = ("OVER+SIG" if g["over_ceiling"] and g["set_sig"] else
               "OVER" if g["over_ceiling"] else
               "SIG" if g["set_sig"] else "")
        print(f"  {g['run']:<16} g{g['group_id']:<3} n={g['n']:<3} "
              f"occ={g['occ']:<3} cov={g['coverage']:.4f} "
              f"entr={g['ent_range']:<6.2f} {tag}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--all"]
    main(args or list(BATCHES), select_all="--all" in sys.argv[1:])
