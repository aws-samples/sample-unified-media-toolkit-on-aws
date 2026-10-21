#!/usr/bin/env python3
"""P2 adjudication scorer — pooled judgment of cast_adjudication.

Reads the sweep outputs (/tmp/gt/adjudicate/<ep>/persons.json) and
scores every cast_adjudication assignment against the gate's truth
sources (positive-only oracles). NAME assignments: corroborated when
the name is in a curated truth list or the asset's own credits.
CHARACTER assignments: corroborated when the character is in the
asset's credits pairs or the describer vocabulary. Everything is also
listed verbatim for the hand-check — n is small by design (the
firewall drops uncited and non-verbatim verdicts before this).

Usage: python3 scripts/adjudicate_score.py [--dir /tmp/gt/adjudicate]
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
    ap.add_argument("--dir", default="/tmp/gt/adjudicate")
    args = ap.parse_args()

    vocab = describer_vocab()
    pooled = {"assets_swept": 0, "assets_with_assignments": 0,
              "names_assigned": 0, "names_corroborated": 0,
              "characters_assigned": 0, "characters_corroborated": 0}
    rows = []
    for ep in sorted(os.listdir(args.dir)):
        path = os.path.join(args.dir, ep, "persons.json")
        if not os.path.isfile(path):
            continue
        pooled["assets_swept"] += 1
        doc = json.load(open(path))
        credits = None
        cpath = f"/tmp/gt/in_{ep}/credits.json"
        if os.path.exists(cpath):
            credits = json.load(open(cpath))
        curated, cnames, characters = truth_for(ep, credits)

        found = False
        for p in doc["persons"]:
            adj = [e for e in p.get("evidence") or []
                   if isinstance(e, dict)
                   and e.get("mechanism") == "cast_adjudication"]
            if not adj:
                continue
            found = True
            for e in adj:
                if e.get("actor"):
                    pooled["names_assigned"] += 1
                    ok = e["actor"].casefold() in (curated or {}) \
                        or e["actor"].casefold() in cnames
                    pooled["names_corroborated"] += ok
                    rows.append((ep, "name", e["actor"], ok,
                                 e.get("reason", "")[:90]))
                if e.get("character"):
                    pooled["characters_assigned"] += 1
                    ok = e["character"].casefold() in characters \
                        or e["character"].casefold() in vocab
                    pooled["characters_corroborated"] += ok
                    rows.append((ep, "character",
                                 f"{p.get('name')} -> {e['character']}",
                                 ok, e.get("reason", "")[:90]))
        pooled["assets_with_assignments"] += found

    print("=== P2 CAST ADJUDICATION (pooled) ===")
    for k, v in pooled.items():
        print(f"  {k:26s} {v}")
    if rows:
        print("\nassignments (corroborated?):")
        for ep, kind, what, ok, why in rows:
            mark = "Y" if ok else "?"
            print(f"  [{mark}] {ep} {kind}: {what} — {why}")


if __name__ == "__main__":
    main()
