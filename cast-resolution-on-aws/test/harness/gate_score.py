#!/usr/bin/env python3
"""Corpus gate scorer — the pinned, re-runnable measurement (README.md §7).

Re-binds every /tmp/gt/in_* asset with the CURRENT code, then scores the
named persons against whatever truth exists, POOLED (single-asset numbers
are noise — the ±2 memory):

  truth sources, per asset:
    {TRUTH_DIR}/{asset}.json     curated people lists (LOCAL-ONLY,
                                 see gate/truth/README.md)
    in_{asset}/credits.json      the asset's OWN credits names (episodic)
    truth/describer_cues.json   describer vocabulary (character names)

  verdict per named person — truth lists are POSITIVE-ONLY oracles
  (first run's lesson: a public figure on a business-news asset is a
  correctly named SUBJECT missing from an on-air-staff list, not a wrong name):
    corroborated  name (or alias) present in a truth source
    uncovered     no truth source covers this name
  Real wrongness needs per-person verification (claim-verify / human) —
  the headline metric is CORROBORATION, deliberately not precision.

Also audits the critic: vetoes whose withdrawn name IS in truth are the
false-veto cost (the anchor_B2 class — a corpus-measured anchor whose
withdrawn name was in truth; mapping in GOTCHAS.md).

Usage: python3 scripts/gate_score.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root
sys.path.insert(0, ROOT)

from lib import bind_pipeline  # noqa: E402
from lib import reads  # noqa: E402

# gitignored REPO-LOCAL home (gate/truth/.gitignore): untracked —
# the no-identifiers rule needs UNTRACKED, not EPHEMERAL; /tmp is
# purged by age (review finding, 2026-09-05)
TRUTH_DIR = os.environ.get("PB_TRUTH_DIR",
                           os.path.join(ROOT, "gate", "truth"))


def bind_local(d):
    """THE pipeline (lib/bind_pipeline) — the gate must score the
    binder that ships, not a local variant (the assign_roles drift,
    review 2026-09-04)."""
    (transcript, faces, _shots, overlay, credits, src, speaker_f0,
     screen_texts) = bind_pipeline.load_local_asset(d)
    return bind_pipeline.bind(transcript, faces, overlay, src, credits,
                              speaker_f0=speaker_f0,
                              screen_texts=screen_texts), credits


def truth_for(ep, credits):
    """-> (curated_names: {casefold->canonical} | None,
           credits_names: set, characters: set)"""
    curated = None
    path = os.path.join(TRUTH_DIR, f"{ep}.json")
    if os.path.exists(path):
        curated = {}
        for p in json.load(open(path))["people"]:
            curated[p["name"].casefold()] = p["name"]
            for a in p.get("aliases") or []:
                curated[a.casefold()] = p["name"]
    credits_names, characters = set(), set()
    if credits:
        cast = reads.cast_from_credits(credits)
        credits_names = {n.casefold() for n in cast["names"]}
        credits_names |= {p["actor"].casefold() for p in cast["pairs"]}
        characters = {p["character"].casefold() for p in cast["pairs"]}
    return curated, credits_names, characters


def describer_assets():
    """Which local assets the describer vocabulary belongs to — a
    LOCAL config beside the truth lists (asset names identify titles,
    so they never live in code)."""
    path = os.path.join(TRUTH_DIR, "describer_assets.json")
    if not os.path.exists(path):
        return set()
    return set(json.load(open(path)))


def describer_vocab():
    """professional-describer cues -> frequent capitalized tokens =
    character-name vocabulary (data-driven, no curation)."""
    path = os.path.join(TRUTH_DIR, "describer_cues.json")
    if not os.path.exists(path):
        return set()
    words = Counter()
    for c in json.load(open(path)):
        for w in re.findall(r"\b[A-Z][a-z]{2,}\b", c.get("text") or ""):
            words[w] += 1
    return {w.casefold() for w, n in words.items() if n >= 3}


def name_kind(p):
    """finest-grained mechanism for how this person got its name."""
    kinds = set()
    for e in p.get("evidence") or []:
        k = e.get("kind") or e.get("mechanism")
        if k in ("self_id", "sign_off", "vocative_resolved"):
            kinds.add(k)
        elif k == "overlay_name":
            kinds.add("overlay:" + (e.get("source") or "enrichment"))
    for m in p.get("name_mechanisms") or []:
        if m not in ("transcript_cue", "overlay_name"):
            kinds.add(m)
    return sorted(kinds) or ["unknown"]


def _truth_health():
    """An absent oracle must never masquerade as a regression: with no
    truth files every name scores 'uncovered', which is exactly what a
    real mechanism failure looks like (review finding, 2026-09-05)."""
    import glob
    files = glob.glob(os.path.join(TRUTH_DIR, "*.json"))
    if not files:
        print(f"WARNING: no truth files under {TRUTH_DIR} — the corpus "
              f"gate is scoring an EMPTY oracle (every name will read "
              f"'uncovered'). The truth lists are LOCAL-ONLY (see "
              f"gate/truth/README.md); restore them from your backup "
              f"before trusting these numbers.", file=sys.stderr)
    return len(files)


def main():
    _truth_health()
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    gol_vocab = describer_vocab()
    desc_assets = describer_assets()
    per_mech = defaultdict(lambda: {"named": 0, "corroborated": 0,
                                    "uncovered": 0})
    rows, false_vetoes, char_hits = [], [], []
    assets = 0
    for d in sorted(os.listdir("/tmp/gt")):
        if not d.startswith("in_") or not os.listdir(f"/tmp/gt/{d}"):
            continue
        ep = d[3:]
        doc, credits = bind_local(f"/tmp/gt/{d}")
        curated, cnames, characters = truth_for(ep, credits)
        assets += 1

        for p in doc["persons"]:
            if p.get("character_name") and ep in desc_assets:
                char_hits.append((p["name"], p["character_name"],
                                  p["character_name"].split()[0].casefold()
                                  in gol_vocab))
            if not p.get("name"):
                continue
            key = p["name"].casefold()
            if (curated is not None and key in curated) \
                    or key in cnames \
                    or (ep in desc_assets and key in gol_vocab):
                verdict = "corroborated"
            else:
                verdict = "uncovered"
            for mech in name_kind(p):
                per_mech[mech]["named"] += 1
                per_mech[mech][verdict] += 1
            rows.append({"asset": ep, "name": p["name"],
                         "mechanisms": name_kind(p), "verdict": verdict})

        for v in doc.get("vetoes") or []:
            nm = (v.get("name") or "").casefold()
            if curated is not None and nm in curated:
                false_vetoes.append({"asset": ep, **v})

    print(f"\n=== GATE (pooled, {assets} assets) ===")
    print(f"{'mechanism':28s} {'named':>6} {'corrob':>8} {'uncovered':>10}")
    for mech, s in sorted(per_mech.items()):
        print(f"{mech:28s} {s['named']:>6} {s['corroborated']:>8} "
              f"{s['uncovered']:>10}")
    print(f"\nfalse vetoes (withdrawn but true): {len(false_vetoes)}")
    for v in false_vetoes[:10]:
        print(f"  {v['asset']}: {v.get('name')} [{v['check']}]")
    if char_hits:
        ok = sum(1 for _, _, hit in char_hits if hit)
        print(f"\ncharacter_name vs describer vocab: "
              f"{ok}/{len(char_hits)} corroborated")
    if args.json:
        json.dump({"per_mechanism": dict(per_mech), "rows": rows,
                   "false_vetoes": false_vetoes},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
