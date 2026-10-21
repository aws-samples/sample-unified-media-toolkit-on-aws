"""
Download RAW reference audio + transcripts for the default voice catalog
(README.md §5/§12 — catalog sourcing, per user request).

Two sources, both fetched from their real datasets (not synthesis
outputs, not re-cloned copies — see the session finding that
baked_voice_samples/*.wav and voice_candidates*/*.wav are CosyVoice
OUTPUTS, unusable as cloning references):

  FLEURS (google/fleurs, CC-BY-4.0)      — the 9 languages CosyVoice3
                                            actually supports (verified
                                            against the official model
                                            card's `language:` field:
                                            zh/en/ja/ko/de/es/fr/it/ru
                                            — pt was pulled once, found
                                            unsupported, and dropped)
  LibriSpeech (openslr/librispeech_asr, CC-BY-4.0) — English, gender via
                                            SPEAKERS.TXT (not in the
                                            per-utterance schema)

Both fetched via direct Parquet download (NOT datasets.load_dataset
streaming — that path stalled repeatedly against these two repos this
session; direct hf_hub_download + pyarrow was fast and reliable).

Output layout, one row per selected reference:
  catalog_sources/fleurs/{lang}_{gender}_{id}.wav
  catalog_sources/fleurs/manifest.json         [{id, lang, gender, ref_text, wav}]
  catalog_sources/librispeech/{gender}_{speaker_id}_{id}.wav
  catalog_sources/librispeech/manifest.json    [{id, speaker_id, gender, ref_text, wav}]

ref_text is the ORIGINAL spoken transcript (FLEURS: raw_transcription,
properly cased/punctuated; LibriSpeech: text, upstream is upper-case
only — required for register_voice's own transcript, not invented).
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

HERE = os.path.dirname(os.path.abspath(__file__))

# Exactly CosyVoice3's 9 supported languages (official model card
# `language:` YAML field, FunAudioLLM/Fun-CosyVoice3-0.5B-2512) — no
# more, no less. Do not add a language here without confirming the
# model actually supports it; pt_br was pulled once, found NOT in the
# model card's list, and removed.
FLEURS_LANGS = {
    "en_us": "en", "fr_fr": "fr", "de_de": "de", "es_419": "es",
    "it_it": "it", "ja_jp": "ja", "ko_kr": "ko", "cmn_hans_cn": "zh",
    "ru_ru": "ru",
}
FLEURS_GENDER = {0: "male", 1: "female", 2: "other"}
PER_LANG_PER_GENDER = 2  # 2 male + 2 female per language, where available

LIBRISPEECH_PER_GENDER = 5


def fleurs_manifest_row(rec, lang_code):
    audio = rec["audio"]
    return {
        "lang": lang_code,
        "gender": FLEURS_GENDER.get(rec["gender"], "other"),
        "speaker_hint": f"fleurs-{lang_code}-{rec['id']}",
        "ref_text": rec["raw_transcription"],
        "wav_bytes": audio["bytes"],
        "id": rec["id"],
    }


def download_fleurs():
    out_dir = os.path.join(HERE, "fleurs")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []

    try:
        files = list_repo_files("google/fleurs", repo_type="dataset")
    except Exception as exc:
        print(f"  ERROR listing repo: {exc}", file=sys.stderr)
        return []

    for config, lang_code in FLEURS_LANGS.items():
        picked = {"male": [], "female": []}

        # test first (canonical eval split), then train as a fallback —
        # found live: de_de and pt_br's test/validation splits are 100%
        # male (862/386 utterances, zero female) while their train
        # splits have thousands of female utterances. Not a bug in file
        # discovery — verified directly against the parquet gender
        # column before adding this fallback.
        for split in ("test", "train"):
            if all(len(v) >= PER_LANG_PER_GENDER for v in picked.values()):
                break
            candidates = [f for f in files
                         if f"parquet-data/{config}/{split}" in f and f.endswith(".parquet")]
            if not candidates:
                continue
            for pq_file in candidates:
                if all(len(v) >= PER_LANG_PER_GENDER for v in picked.values()):
                    break
                path = hf_hub_download(repo_id="google/fleurs", repo_type="dataset",
                                       filename=pq_file)
                # read_row_group, not .read()/.iter_batches() — both hit
                # "Nested data conversions not implemented for chunked
                # array outputs" on some FLEURS files (audio struct
                # column spans multiple chunks); per-row-group read
                # avoids the chunked-array path entirely.
                pf = pq.ParquetFile(path)
                for rg_i in range(pf.num_row_groups):
                    for rec in pf.read_row_group(rg_i).to_pylist():
                        gender = FLEURS_GENDER.get(rec["gender"], "other")
                        if gender not in picked or len(picked[gender]) >= PER_LANG_PER_GENDER:
                            continue
                        if rec["audio"]["bytes"] is None:
                            continue
                        # sanity: real transcript, not empty, not absurdly short
                        if not rec["raw_transcription"] or len(rec["raw_transcription"]) < 10:
                            continue
                        picked[gender].append(fleurs_manifest_row(rec, lang_code))

        for gender, rows in picked.items():
            for row in rows:
                fname = f"{lang_code}_{gender}_{row['id']}.wav"
                fpath = os.path.join(out_dir, fname)
                # decode via soundfile to verify + normalize container,
                # not just blit raw bytes (confirms the audio is real)
                data, sr = sf.read(io.BytesIO(row["wav_bytes"]))
                sf.write(fpath, data, sr)
                manifest.append({
                    "wav": fname, "lang": row["lang"], "gender": row["gender"],
                    "ref_text": row["ref_text"], "duration_s": round(len(data) / sr, 2),
                    "sample_rate": sr, "source_id": row["id"],
                })
                print(f"[fleurs:{lang_code}:{gender}] {fname} "
                     f"({round(len(data)/sr,1)}s) — {row['ref_text'][:60]!r}")

    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    return manifest


def load_librispeech_speaker_genders():
    """SPEAKERS.TXT is not in the HF parquet schema — fetched separately
    from the canonical OpenSLR mirror (verified this session: format is
    stable, pipe-delimited, 'ID | SEX | SUBSET | MINUTES | NAME')."""
    import urllib.request
    import tarfile

    url = "https://www.openslr.org/resources/12/raw-metadata.tar.gz"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-2000000"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
        raw = resp.read()
    # Streaming mode ("r|gz"): sequential access only, no full-archive
    # index scan — required since `raw` is a partial (Range-limited)
    # download and SPEAKERS.TXT is near the front of the tar.
    genders = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r|gz") as tf:
        for member in tf:
            if member.name == "LibriSpeech/SPEAKERS.TXT":
                content = tf.extractfile(member).read()
                for line in content.decode("utf-8", errors="replace").splitlines():
                    if line.startswith(";") or not line.strip():
                        continue
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) < 2:
                        continue
                    sid, sex = parts[0], parts[1]
                    if sid.isdigit():
                        genders[int(sid)] = "male" if sex == "M" else "female"
                break
    return genders


def download_librispeech():
    out_dir = os.path.join(HERE, "librispeech")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []

    print("[librispeech] fetching SPEAKERS.TXT for gender mapping...")
    genders = load_librispeech_speaker_genders()
    print(f"[librispeech] {len(genders)} speakers with known gender")

    path = hf_hub_download(repo_id="openslr/librispeech_asr", repo_type="dataset",
                           filename="clean/test/0000.parquet")
    table = pq.ParquetFile(path).read()

    picked = {"male": [], "female": []}
    seen_speakers = {"male": set(), "female": set()}
    for rec in table.to_pylist():
        if all(len(v) >= LIBRISPEECH_PER_GENDER for v in picked.values()):
            break
        sid = rec["speaker_id"]
        gender = genders.get(sid)
        if gender is None or len(picked[gender]) >= LIBRISPEECH_PER_GENDER:
            continue
        if sid in seen_speakers[gender]:
            continue  # one clip per speaker for variety
        if rec["audio"]["bytes"] is None or not rec["text"]:
            continue
        picked[gender].append(rec)
        seen_speakers[gender].add(sid)

    for gender, recs in picked.items():
        for rec in recs:
            data, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]))
            fname = f"{gender}_{rec['speaker_id']}_{rec['id']}.wav"
            fpath = os.path.join(out_dir, fname)
            sf.write(fpath, data, sr)
            # LibriSpeech's upstream 'text' field is upper-case, no
            # punctuation (ASR training convention) — title-case it so
            # it reads as real prose for ref_text, same treatment this
            # session already gave other LibriSpeech transcripts by hand
            ref_text = rec["text"][0] + rec["text"][1:].lower()
            manifest.append({
                "wav": fname, "gender": gender, "speaker_id": rec["speaker_id"],
                "ref_text": ref_text, "duration_s": round(len(data) / sr, 2),
                "sample_rate": sr, "source_id": rec["id"],
            })
            print(f"[librispeech:{gender}] {fname} ({round(len(data)/sr,1)}s) "
                 f"speaker={rec['speaker_id']} — {ref_text[:60]!r}")

    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    return manifest


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "fleurs"):
        rows = download_fleurs()
        print(f"\nFLEURS: {len(rows)} references downloaded")
    if which in ("all", "librispeech"):
        rows = download_librispeech()
        print(f"\nLibriSpeech: {len(rows)} references downloaded")
