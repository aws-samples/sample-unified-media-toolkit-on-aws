#!/usr/bin/env python3
"""Build offline replay caches for the clustering harness.

For each asset of a corpus run, pulls the flag-on diarization
intermediates and writes ONE pickle:
    {cache_dir}/{asset}.pkl = {
      "high":   [sentence dicts w/ embeddings]      (HIGH quality only)
      "segs":   [track segment dicts]               (track_embeddings)
      "guides": {"co_active_pairs": [...namespaced], "clique_floor": n}
      "powerset": {chunk_idx: {"tracks": [...], "co_active_pairs": [...],
                                "clique_floor": n}}   (per-chunk, local ids)
      "prefix": s3 run prefix
    }
Run:  python3 test/replay_cache.py STATE_JSON CACHE_DIR [--span-approx] [asset ...]
STATE_JSON = an e2e runner state file whose entries carry
data["<asset>"]["diar"]["output_s3_prefix"] (or output_s3_prefix at
the entry root for diar-only runner states).

--span-approx: for artifacts written BEFORE span-corroborated
co_active_secs (2026-09-30), approximate the corroboration at cache
build time — a link keeps its seconds only when BOTH chains' segments
sit within 0.5 s of an overlap_span; otherwise 0.0. This is a
chain-adjacency stand-in for the true frame intersection (which needs
per-frame data that never ships); post-fix artifacts must NOT use it.

Fetches ride gateway-presigned URLs (media-outputs); caches belong in
a cleaner-proof location (outputs/..., NOT /tmp — two prior harness
generations died to the /tmp janitor).
"""
import json
import os
import pickle  # nosec B403
import sys
import urllib.request

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


def _approx_span_corroboration(doc):
    """Pre-fix artifacts: zero a link's seconds unless both chains'
    segments sit within 0.5 s of an overlap_span (chain-adjacency
    stand-in for the shipped frame-intersection semantics)."""
    spans = doc.get("overlap_spans") or []
    tr = {t["id"]: t["segs"] for t in doc.get("tracks") or []}

    def near(ch):
        return any(s0 - 0.5 <= x1 and x0 <= s1 + 0.5
                   for x0, x1 in tr.get(ch, [])
                   for s0, s1 in spans)
    pairs = doc.get("co_active_pairs") or []
    secs = doc.get("co_active_secs") or [0.0] * len(pairs)
    return [s if (spans and near(a) and near(b)) else 0.0
            for (a, b), s in zip(pairs, secs)]


def build(prefix, span_approx=False):
    high, segs = [], []
    n = 0
    while True:
        try:
            doc = fetch(f"{prefix}/dynamic/embeddings/embeddings_{n}.json")
        except Exception:
            break
        high += [s for s in doc.get("sentence_embeddings", [])
                 if s.get("quality") == "high" and s.get("embedding")]
        segs += doc.get("track_embeddings", [])
        n += 1
    powerset = {}
    guides = {"co_active_pairs": [], "clique_floor": 0}
    c = 0
    while True:
        try:
            doc = fetch(f"{prefix}/powerset/powerset_{c}.json")
        except Exception:
            if c > 0:
                break
            # chunk 0 may legitimately be absent only if flag was off
            break
        powerset[c] = {k: doc.get(k) for k in
                       ("tracks", "co_active_pairs", "co_active_secs",
                        "clique_floor", "local_speaker_floor",
                        "fences", "overlap_spans")}
        if span_approx:
            powerset[c]["co_active_secs"] = _approx_span_corroboration(doc)
        ns = (c + 1) * 1_000_000
        guides["co_active_pairs"] += [[ns + a, ns + b] for a, b in
                                      doc.get("co_active_pairs", [])]
        guides["clique_floor"] = max(guides["clique_floor"],
                                     doc.get("clique_floor", 0))
        c += 1
    # live labels for the acceptance harness (which clusters shipped)
    gs = fetch(f"{prefix}/embeddings/global_speakers.json")
    live = {"sentence_speakers": gs.get("sentence_speakers") or {},
            "K": len(gs.get("speaker_models") or {})}
    return {"high": high, "segs": segs, "guides": guides,
            "powerset": powerset, "live": live, "prefix": prefix}


def main():
    state_path, cache_dir = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    span_approx = "--span-approx" in rest
    only = {a for a in rest if not a.startswith("--")}
    os.makedirs(cache_dir, exist_ok=True)
    state = json.load(open(state_path))
    for asset, entry in sorted(state.items()):
        if not isinstance(entry, dict):
            continue
        diar = entry.get("diar", entry)
        if "output_s3_prefix" not in diar:
            continue
        if only and asset not in only:
            continue
        out = os.path.join(cache_dir, f"{asset}.pkl")
        if os.path.exists(out):
            print(f"{asset}: cached")
            continue
        doc = build(diar["output_s3_prefix"], span_approx=span_approx)
        with open(out, "wb") as f:
            pickle.dump(doc, f)  # nosemgrep: python.lang.security.deserialization.avoid-pickle
        print(f"{asset}: high={len(doc['high'])} segs={len(doc['segs'])} "
              f"chunks={len(doc['powerset'])} live_K={doc['live']['K']}")


if __name__ == "__main__":
    main()
