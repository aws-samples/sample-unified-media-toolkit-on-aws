#!/usr/bin/env python3
"""Corpus-gate pipeline runs: give every remaining ground-truth episode
the sibling runs cast-resolution needs (README.md §7 "corpus gate").

Per episode, TWO phases through the m2cv6 gateway:
  P1  audio-stem + shot-segmentation      (parallel)
  P2  speech-diarization (speech stem) + face-recognition (shots)

CA and enrichment are deliberately SKIPPED for gate v1: naming truth
(credits, describer cues, captions-as-validation) needs transcript +
faces; overlay evidence's contribution is already measurable on the six
standing assets. Everything carries one idempotency token, so re-running
this script absorbs instead of re-paying.

Usage:
    python3 scripts/corpus_runs.py [--only <asset1>,<asset2>] [--dry-run]

Writes per-episode state to /tmp/gt/pbgate_{ep}_state.json, then extracts
inputs to /tmp/gt/in_{ep}/ on success.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(HERE))), "media-toolkit-gateway", "scripts"))
# repo root: test/harness/ is two levels below it
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from test_gateway import fetch_credentials, get_token, call_tool  # noqa: E402

RUN_ID = "pbgate-20260904"
TERMINAL = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}

# episode key -> [S3 listing prefix, filename regex with episode group].
# Prefixes and filename patterns name real productions, so the map is
# LOCAL config, never code — loaded LAZILY so --help/--dry-run never
# crash on a purged /tmp (the temp cleaner deletes by age).
def _series():
    path = "/tmp/gt/corpus_series.json"
    if not os.path.exists(path):
        raise SystemExit(f"local corpus config missing: {path} — the "
                         f"series map lives outside the repo (no title "
                         f"identifiers in code); restore it from your "
                         f"backup before running")
    return {k: tuple(v) for k, v in json.load(open(path)).items()}

creds = fetch_credentials("/media-toolkit-gateway/dev/gateway-credentials",
                          "eu-west-1")
_tok, _born, _lock = None, 0.0, threading.Lock()
_log_lock = threading.Lock()


def tok():
    global _tok, _born
    with _lock:
        if _tok is None or time.time() - _born > 2400:
            _tok = get_token(creds["tokenEndpoint"], creds["clientId"],
                             creds["clientSecret"], creds["scope"])
            _born = time.time()
        return _tok


def log(msg):
    with _log_lock:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def call(tool, args, tries=5):
    for attempt in range(1, tries + 1):
        try:
            return call_tool(creds["endpointUrl"], tok(), tool, args)
        except Exception as exc:                     # noqa: BLE001 — transport
            if attempt == tries:
                raise
            log(f"    transient {tool}: {str(exc)[:120]}")
            time.sleep(30 * attempt)


def discover_episodes():
    """{ep_key: source_uri} for every episode file in the three series."""
    eps = {}
    for series, (prefix, pattern) in _series().items():
        bucket, key_prefix = prefix[5:].split("/", 1)
        out = subprocess.run(  # nosec B603
            ["aws", "s3api", "list-objects-v2", "--bucket", bucket,
             "--prefix", key_prefix, "--query", "Contents[].Key",
             "--output", "json", "--region", "eu-west-1"],
            capture_output=True, text=True)
        for key in json.loads(out.stdout or "[]") or []:
            m = re.search(pattern, key)
            if m:
                eps[f"{series}{m.group(1)}"] = f"s3://{bucket}/{key}"
    return eps


def submit(ep, target, tool, args):
    r = call(f"{target}___{tool}", {**args, "idempotency_token": RUN_ID})
    log(f"  [{ep}] {target}: wid={r.get('workflow_id')}"
        f"{' (absorbed)' if 'message' in r else ''}")
    if not r.get("job_id"):
        raise RuntimeError(f"{target}: no job_id — {json.dumps(r)[:200]}")
    return r


def wait(ep, jobs, timeout_s=4 * 3600):
    final, t0 = {}, time.time()
    while len(final) < len(jobs):
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"timeout waiting for "
                               f"{sorted(set(jobs) - set(final))}")
        for key, (target, h) in jobs.items():
            if key in final:
                continue
            r = call(f"{target}___get_job_status", {"job_id": h["job_id"]})
            if r.get("status") in TERMINAL:
                final[key] = r
                log(f"  [{ep}] {key}: {r['status']}")
                if r["status"] != "SUCCEEDED":
                    raise RuntimeError(f"{key} {r['status']}: "
                                       f"{str(r.get('error', ''))[:200]}")
        if len(final) < len(jobs):
            time.sleep(45)
    return final


def out_uri(status, substring):
    return next(o["uri"] for o in status.get("outputs", [])
                if substring in o.get("uri", ""))


def run_episode(ep, uri):
    state_path = f"/tmp/gt/pbgate_{ep}_state.json"
    state = {"episode": ep, "uri": uri, "token": RUN_ID}

    def save(status="running"):
        state["status"] = status
        json.dump(state, open(state_path, "w"), indent=1)
    try:
        save()
        stem = submit(ep, "audio-stem", "separate_audio_stems",
                      {"asset_s3uri": uri})
        seg = submit(ep, "shot-segmentation", "segment_shots",
                     {"asset_s3uri": uri})
        p1 = wait(ep, {"stem": ("audio-stem", stem),
                       "seg": ("shot-segmentation", seg)})
        state["shot_segmentation"] = p1["seg"]["output_s3_prefix"]
        speech = out_uri(p1["stem"], "audio_speech.wav")
        save()

        dia = submit(ep, "speech-diarization", "diarize_speech",
                     {"asset_s3uri": speech})
        face = submit(ep, "face-recognition", "recognize_faces",
                      {"asset_s3uri": uri,
                       "shot_segmentation_s3uri": out_uri(p1["seg"],
                                                          "shots.json"),
                       "collection_id": ""})
        p2 = wait(ep, {"dia": ("speech-diarization", dia),
                       "face": ("face-recognition", face)})
        state["speech_diarization"] = p2["dia"]["output_s3_prefix"]
        state["face_recognition"] = p2["face"]["output_s3_prefix"]
        save("SUCCEEDED")

        cfg = {k: state[k] for k in ("speech_diarization",
                                     "face_recognition",
                                     "shot_segmentation")}
        json.dump(cfg, open(f"/tmp/gt/pbgate_{ep}_cfg.json", "w"))
        subprocess.run([sys.executable,  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                        os.path.join(HERE, "extract_inputs.py"),
                        "--config", f"/tmp/gt/pbgate_{ep}_cfg.json",
                        "-o", f"/tmp/gt/in_{ep}"], check=False)
        log(f"[{ep}] DONE — inputs at /tmp/gt/in_{ep}")
    except Exception as exc:                         # noqa: BLE001 — isolate
        state["error"] = str(exc)[:400]
        save("FAILED")
        log(f"[{ep}] FAILED: {str(exc)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated episode keys")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eps = discover_episodes()
    have = {d[3:] for d in os.listdir("/tmp/gt")
            if d.startswith("in_") and os.listdir(f"/tmp/gt/{d}")}
    todo = {k: v for k, v in sorted(eps.items()) if k not in have}
    if args.only:
        keep = set(args.only.split(","))
        todo = {k: v for k, v in todo.items() if k in keep}
    log(f"episodes discovered={len(eps)} already-extracted={len(have)} "
        f"to-run={len(todo)}: {sorted(todo)}")
    if args.dry_run:
        return
    threads = []
    for ep, uri in todo.items():
        t = threading.Thread(target=run_episode, args=(ep, uri), name=ep)
        threads.append(t)
        t.start()
        time.sleep(20)          # stagger; services queue internally
    for t in threads:
        t.join()
    done = sum(1 for ep in todo
               if json.load(open(f"/tmp/gt/pbgate_{ep}_state.json"))
               ["status"] == "SUCCEEDED")
    log(f"CORPUS RUNS COMPLETE: {done}/{len(todo)} SUCCEEDED")


if __name__ == "__main__":
    main()
