"""
Empirical measurement: for real (professionally translated, FLEURS
parallel corpus) EN source sentences, how does the NATURAL (unstretched)
synthesized duration of the translation compare to the English source
clip's real duration?

This is the data needed before deciding a dubbing fit-tolerance bound —
answers "how far off is a typical line, before any stretching" for
several language directions, not assumed.

Method:
  1. Pull N parallel (sentence_id, en_text, en_audio_duration,
     target_text) tuples from FLEURS test splits (en_us paired against
     each target language by shared id — FLEURS ids are the same
     underlying sentence, professionally translated, confirmed this
     session).
  2. Synthesize target_text with a registered catalog voice for that
     language, NO target_duration_s (natural length).
  3. ratio = synthesized_duration / en_source_duration.

Not measuring: MT quality (translations are real human ones from
FLEURS, not machine-translated) or ASR-derived segment timing (using
the FLEURS clip's own real duration as the "source segment length"
proxy, which is the same role a diarization-derived segment start/end
would play).
"""
from __future__ import annotations

import json
import sys
import time
import uuid as uuid_mod

import os

import boto3
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

_sfn = boto3.client("stepfunctions", region_name="eu-west-1")
_s3 = boto3.client("s3", region_name="eu-west-1")

BUCKET = os.environ["TTS_PROXY_BUCKET"]
SYNTH_ARN = os.environ["TTS_SYNTH_ARN"]  # tts-synthesis state machine

VOICE_BY_LANG = {
    "it": "c300471c-2048-4a95-b6cd-b2bae41c2584",
    "de": "96203e29-e903-46d6-9834-931d1490ecca",
    "es": "329d9db5-866c-441d-87d2-6bbc24a4398e",
    "ja": "f9a0a4db-f433-43af-b83d-b698b1e1b34d",
    "fr": "877e6878-be8d-4227-9486-76a583217e9f",
    "ko": "f8381a7f-1d3e-432d-900c-4ecaa2f2073f",
    "zh": "c453903b-b127-4fa1-8609-dede2ea7bbf7",
    "ru": "e52eab21-b8f7-4111-980d-af344418dd70",
}

FLEURS_CONFIG = {
    "it": "it_it", "de": "de_de", "es": "es_419", "ja": "ja_jp",
    "fr": "fr_fr", "ko": "ko_kr", "zh": "cmn_hans_cn", "ru": "ru_ru",
}


def get_parallel_pairs(target_lang, n_pairs):
    """Return [(id, en_text, en_duration_s, target_text), ...] for
    sentence ids present in BOTH en_us and the target language's test
    split (real parallel translation, not machine-translated)."""
    en_path = hf_hub_download(repo_id="google/fleurs", repo_type="dataset",
                              filename="parquet-data/en_us/test-00000-of-00001.parquet")
    tgt_path = hf_hub_download(repo_id="google/fleurs", repo_type="dataset",
                               filename=f"parquet-data/{FLEURS_CONFIG[target_lang]}/test-00000-of-00001.parquet")

    en_by_id = {}
    en_pf = pq.ParquetFile(en_path)
    for rg in range(en_pf.num_row_groups):
        for rec in en_pf.read_row_group(rg).to_pylist():
            if rec["audio"]["bytes"] is None or len(rec["raw_transcription"]) < 15:
                continue
            en_by_id[rec["id"]] = rec

    tgt_by_id = {}
    tgt_pf = pq.ParquetFile(tgt_path)
    for rg in range(tgt_pf.num_row_groups):
        for rec in tgt_pf.read_row_group(rg).to_pylist():
            if len(rec["raw_transcription"]) < 5:
                continue
            tgt_by_id[rec["id"]] = rec["raw_transcription"]

    common = sorted(set(en_by_id) & set(tgt_by_id))[:n_pairs]
    pairs = []
    for sid in common:
        en_rec = en_by_id[sid]
        # measure real duration from decoded audio, not assumed from any field
        data, sr = sf.read(__import__("io").BytesIO(en_rec["audio"]["bytes"]))
        en_dur = len(data) / sr
        pairs.append((sid, en_rec["raw_transcription"], en_dur, tgt_by_id[sid]))
    return pairs


def start_synth(voice_id, text, tag):
    sfn_input = {
        "uuid": str(uuid_mod.uuid4()),
        "input": {
            "batches": [{"voice_id": voice_id, "transcripts": [{"text": text}]}],
            "config": {"flow_steps": 6},
        },
        "data": {},
    }
    resp = _sfn.start_execution(
        stateMachineArn=SYNTH_ARN, name=f"dubstudy-{tag}-{int(time.time()*1000)}"[:80],
        input=json.dumps(sfn_input))
    return resp["executionArn"]


def poll(arn, timeout_s=240):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = _sfn.describe_execution(executionArn=arn)
        if resp["status"] != "RUNNING":
            if resp["status"] != "SUCCEEDED":
                return {"status": resp["status"], "error": (resp.get("cause") or "")[:200]}
            out = json.loads(resp["output"])
            for it in out["data"].get("iterators", []):
                item = it.get("item_data") or it
                o = item["outputs"][0]
                if "error" in o:
                    return {"status": "SYNTH_ERROR", "error": o["error"][:200]}
                return {"status": "SUCCEEDED", "duration_s": o["duration_s"],
                       "wav_key": o["wav_key"], "out_bucket": item["out_bucket"]}
        time.sleep(2)
    return {"status": "TIMEOUT"}


def main():
    langs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["it", "de", "es", "ja"]
    n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    all_results = {}
    for lang in langs:
        print(f"\n{'='*70}\n{lang.upper()}\n{'='*70}")
        pairs = get_parallel_pairs(lang, n_pairs)
        print(f"{len(pairs)} parallel EN->{lang} pairs")

        arns = []
        for sid, en_text, en_dur, tgt_text in pairs:
            arn = start_synth(VOICE_BY_LANG[lang], tgt_text, f"{lang}-{sid}")
            arns.append((sid, en_text, en_dur, tgt_text, arn))
            time.sleep(0.05)  # gentle ramp, avoid a thundering herd on StartExecution

        lang_results = []
        for sid, en_text, en_dur, tgt_text, arn in arns:
            r = poll(arn)
            if r["status"] == "SUCCEEDED":
                ratio = r["duration_s"] / en_dur
                lang_results.append({
                    "id": sid, "en_text": en_text, "en_duration_s": round(en_dur, 2),
                    "tgt_text": tgt_text, "tgt_duration_s": r["duration_s"],
                    "ratio": round(ratio, 3),
                })
                print(f"  id={sid} en={en_dur:.2f}s tgt={r['duration_s']:.2f}s "
                     f"ratio={ratio:.3f}")
            else:
                print(f"  id={sid} FAILED: {r}")
        all_results[lang] = lang_results

    json.dump(all_results, open("dubbing_duration_study_results.json", "w"),
             indent=2, ensure_ascii=False)

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    import statistics
    for lang, results in all_results.items():
        ratios = [r["ratio"] for r in results]
        if not ratios:
            print(f"{lang}: no successful measurements")
            continue
        within_10 = sum(1 for r in ratios if abs(r - 1.0) <= 0.10)
        within_20 = sum(1 for r in ratios if abs(r - 1.0) <= 0.20)
        within_30 = sum(1 for r in ratios if abs(r - 1.0) <= 0.30)
        print(f"{lang}: n={len(ratios)} mean={statistics.mean(ratios):.3f} "
             f"median={statistics.median(ratios):.3f} "
             f"stdev={statistics.stdev(ratios):.3f} "
             f"min={min(ratios):.3f} max={max(ratios):.3f}")
        print(f"       within ±10%: {within_10}/{len(ratios)} ({within_10/len(ratios)*100:.0f}%)  "
             f"within ±20%: {within_20}/{len(ratios)} ({within_20/len(ratios)*100:.0f}%)  "
             f"within ±30%: {within_30}/{len(ratios)} ({within_30/len(ratios)*100:.0f}%)")


if __name__ == "__main__":
    main()
