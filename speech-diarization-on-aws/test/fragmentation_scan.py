#!/usr/bin/env python3
"""Fragmentation scan — the over-split counterpart to the impurity
scoreboard (which only watches over-merge; over-split had no detector,
which is how it crept in unmeasured — user ear-review 2026-09-29).

For each asset: pairwise cosine similarity between per-cluster trimmed
centroids (global_speakers.json speaker_models). Reports
  - SUSPECT pairs (sim >= --suspect, default 0.85): same-voice-split
    candidates, the reconcile pass's prey;
  - NAMED-DIFFERENT pairs (both clusters carry different cast names):
    the must-NOT-merge distribution that calibrates the reconcile
    threshold (hardest measured case: one same-family pair at 0.668).

Run:  python3 test/fragmentation_scan.py STATE_JSON [--suspect 0.85]
STATE_JSON = e2e runner state with diar + cast prefixes per asset.
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

GW = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "media-toolkit-gateway", "scripts")
sys.path.insert(0, GW)
from test_gateway import call_tool, fetch_credentials, get_token  # noqa: E402

creds = fetch_credentials("/media-toolkit-gateway/dev/gateway-credentials",
                          "eu-west-1")
tok = get_token(creds["tokenEndpoint"], creds["clientId"],
                creds["clientSecret"], creds["scope"])


def fetch(uri):
    r = call_tool(creds["endpointUrl"], tok,
                  "media-outputs___get_download_urls",
                  {"uris": [uri], "expiry_seconds": 900})
    return json.loads(urllib.request.urlopen(r["urls"][0]["url"]).read())  # nosec B310


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--suspect", type=float, default=0.85)
    args = ap.parse_args()
    state = json.load(open(args.state))
    tot_suspect = 0
    named_diff_max = (0.0, None)
    for asset, entry in sorted(state.items()):
        if not isinstance(entry, dict) or "diar" not in entry:
            continue
        gs = fetch(entry["diar"]["output_s3_prefix"]
                   + "/embeddings/global_speakers.json")
        models = gs.get("speaker_models") or {}
        names = {}
        if "cast" in entry:
            try:
                sm = fetch(entry["cast"]["output_s3_prefix"]
                           + "/speaker_map.json")["speakers"]
                names = {k: v["name"] for k, v in sm.items()
                         if v.get("name")}
            except Exception:
                pass
        ids = [k for k, m in models.items() if m.get("embedding")]
        if len(ids) < 2:
            continue
        M = np.stack([np.array(models[i]["embedding"], dtype=np.float64)
                      for i in ids])
        M = M / np.linalg.norm(M, axis=1, keepdims=True).clip(1e-10)
        sims = M @ M.T
        suspects, named_diff = [], []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                s = float(sims[i, j])
                a, b = ids[i], ids[j]
                if (names.get(a) and names.get(b)
                        and names[a] != names[b]):
                    named_diff.append((s, a, b))
                    if s > named_diff_max[0]:
                        named_diff_max = (s, f"{asset}:{a}~{b}")
                elif s >= args.suspect:
                    suspects.append((s, a, b))
        tot_suspect += len(suspects)
        top = sorted(suspects, reverse=True)[:3]
        nd = max(named_diff)[0] if named_diff else float("nan")
        print(f"{asset:20s} suspects={len(suspects):>2d} "
              f"top={[(a, b, round(s, 3)) for s, a, b in top]} "
              f"named-diff max={nd:.3f}")
    print(f"\nTOTAL suspect pairs: {tot_suspect}")
    print(f"hardest named-different pair: {named_diff_max[1]} "
          f"sim={named_diff_max[0]:.3f}  <- reconcile threshold must "
          f"clear this with margin")


if __name__ == "__main__":
    main()
