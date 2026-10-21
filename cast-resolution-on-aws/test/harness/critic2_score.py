#!/usr/bin/env python3
"""Tooth-2 flag scorer — pooled judgment of the report-only critique.

Reads the critic2 sweep outputs (/tmp/gt/critic2/<ep>/persons.json,
written by critic2_run.py) and scores every critic2_llm flag against
the SAME truth sources the gate uses (positive-only oracles):

  flag_on_corroborated   the flagged NAME is in a truth source — the
                         binder was right, so the flag is a likely
                         FALSE POSITIVE (the tooth-2 cost)
  flag_uncovered         no truth covers the name — needs a human read
                         (list printed for exactly that)

The graduation question (README.md §12, locked): tooth 2 stays report-only
until flags-on-corroborated is ~zero pooled — a critic that vetoes true
names is worse than no critic. Pooled only (±2 single-asset noise).

Usage: python3 scripts/critic2_score.py [--dir /tmp/gt/critic2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root: test/harness/ is two levels below it
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from gate_score import describer_vocab, truth_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/gt/critic2")
    args = ap.parse_args()

    vocab = describer_vocab()
    pooled = {"assets": 0, "persons": 0, "named": 0, "flags": 0,
              "flag_on_corroborated": 0, "flag_uncovered": 0}
    corroborated_rows, uncovered_rows = [], []

    for ep in sorted(os.listdir(args.dir)):
        path = os.path.join(args.dir, ep, "persons.json")
        if not os.path.isfile(path):
            continue
        doc = json.load(open(path))
        credits = None
        cpath = f"/tmp/gt/in_{ep}/credits.json"
        if os.path.exists(cpath):
            credits = json.load(open(cpath))
        curated, cnames, characters = truth_for(ep, credits)

        by_id = {p["person_id"]: p for p in doc["persons"]}
        pooled["assets"] += 1
        pooled["persons"] += len(doc["persons"])
        pooled["named"] += sum(1 for p in doc["persons"] if p.get("name"))
        for f in doc.get("critic_review") or []:
            pooled["flags"] += 1
            p = by_id.get(f["person_id"], {})
            name = (p.get("name") or "").casefold()
            field = f.get("suspect_field")
            row = (ep, p.get("name") or p.get("aliases"),
                   field, f["reason"][:100])
            # the flagged CHARACTER assignment is judged against the
            # character oracles; a flagged NAME against the name ones
            if field == "character_name":
                target = (p.get("character_name") or "").casefold()
                known = target in characters or target in vocab
            else:
                known = bool(name) and (
                    name in (curated or {}) or name in cnames)
            if known:
                pooled["flag_on_corroborated"] += 1
                corroborated_rows.append(row)
            else:
                pooled["flag_uncovered"] += 1
                uncovered_rows.append(row)

    print(f"=== TOOTH-2 (pooled, {pooled['assets']} assets with "
          f"reviewable persons) ===")
    print(f"persons {pooled['persons']}, named {pooled['named']}, "
          f"flags {pooled['flags']}")
    print(f"  flag_on_corroborated : {pooled['flag_on_corroborated']}"
          f"   <- likely false positives, the graduation blocker")
    print(f"  flag_uncovered       : {pooled['flag_uncovered']}"
          f"   <- human-read list below")
    if corroborated_rows:
        print("\nflags on corroborated names (each one argues AGAINST "
              "graduating tooth 2):")
        for ep, who, field, reason in corroborated_rows:
            print(f"  {ep}: {who} ({field}) — {reason}")
    if uncovered_rows:
        print("\nflags needing a human read:")
        for ep, who, field, reason in uncovered_rows:
            print(f"  {ep}: {who} ({field}) — {reason}")


if __name__ == "__main__":
    main()
