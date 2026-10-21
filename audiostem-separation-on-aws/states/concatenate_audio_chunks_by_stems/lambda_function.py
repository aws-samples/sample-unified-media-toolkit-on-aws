# concatenate_handler.py
"""
Step Function State: Concatenate audio chunks by stems
───────────────────────────────────────────────────────
Runs after the Map state completes. Reads each stem's chunk WAV files from
S3 in order, concatenates them, writes the final per-stem WAV files and an
aggregate metrics JSON to the audio directory.

Two concatenation strategies are available:

  _concat_stem_multipart()  [DEFAULT]
  ─────────────────────────────────────
  Streams chunk PCM data directly into an S3 multipart upload. Peak memory
  per stem is bounded to one chunk at a time (~17–34 MB) regardless of total
  audio duration. Correct for content of any length (hours).

  Pre-calculates the exact final WAV data size from iterator metadata so the
  WAV header is written correctly upfront — no fixup pass required.

  S3 multipart constraints respected:
    • Each part except the last must be ≥ 5 MB (5,242,880 bytes).
    • Parts are uploaded in order; the header + first chunk PCM form part 1.
    • If a single chunk's PCM is < 5 MB (very short last chunk), it is
      accumulated with the previous part before uploading.

  _concat_stem_in_memory()  [LEGACY — kept for reference / debugging]
  ──────────────────────────────────────────────────────────────────────
  Downloads all chunk WAVs into RAM, concatenates numpy arrays, uploads the
  result in a single put_object call. Simple but memory scales with duration:
    1 hr music @ 48 kHz ≈ 660 MB peak
    4 hr music @ 48 kHz ≈ 2.6 GB peak  ← will OOM on Lambda
  Use only for short content or local debugging.

Expected input  (from Step Functions after Map state)
──────────────────────────────────────────────────────
{
    "operation": "StateConcatenateAudioChunksByStems",
    "uuid":      "...",
    "input":     { ... },
    "output":    { ... },
    "data": {
        "audio_bucket":   "...",
        "audio_key":      "path/to/audio.wav",
        "output_bucket":  "...",
        "output_prefix":  "path/to/output/prefix",
        "speech_stem":    true,
        "music_stem":     true,
        "sfx_stem":       true,
        "iterators": [
            {
                "chunk_idx":  0,
                "chunk_name": "chunk_000",
                "stems": {
                    "speech": { "s3_key": "...", "resampl_freq": 16000,
                                "rms_db": -28.4, "duration_sec": 180.0 },
                    ...
                },
                "timing_ms":   { ... },
                "metrics_key": "path/to/output/prefix/segment_chunks/chunk_000.json",
                "status":      "COMPLETED",
                "elapsed_sec": 13.1,
                ...
            },
            ...
        ]
    }
}

S3 output
──────────
  path/to/audio/audio_speech.wav       ← final concatenated stems
  path/to/audio/audio_music.wav
  path/to/audio/audio_sfx.wav
  path/to/audio/audio_speech_safe.wav  ← when speech_safe=true (gated mix
                                         substitution; Design record §5)
  path/to/audio/audiostem_sep.json     ← aggregate metrics
  path/to/audio/stem_activity.json     ← activity spans per lane + line-up
                                         tones on the original (semantic)
  path/to/audio/stem_waveforms.json    ← normalized RMS envelopes per lane
                                         (rendering; original + active stems)

Output  (returned to Step Functions)
──────────────────────────────────────
{
    ...original payload...,
    "data": {
        ...original data...,
        "audiostem_sep_output": {
            "speech_key": "path/to/audio/audio_speech.wav",
            "music_key":  "path/to/audio/audio_music.wav",
            "sfx_key":    "path/to/audio/audio_sfx.wav",
            "metrics_key": "path/to/audio/audiostem_sep.json",
            "stem_activity_key":  "path/to/audio/stem_activity.json",
            "stem_waveforms_key": "path/to/audio/stem_waveforms.json"
        }
    }
}
"""

from __future__ import annotations

import json
import logging
import math
import struct
import time
import traceback

import numpy as np

from lib import audio_analysis
from lib.s3_utils import s3 as _s3, put_json
from lib.wav_utils import encode_wav_header_mono, get_wav_info, read_wav_window

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
if not log.handlers:
    log.addHandler(_handler)

_ALL_STEMS = ("speech", "music", "sfx")
_S3_MIN_PART_BYTES = 5 * 1024 * 1024   # 5 MB — S3 multipart minimum part size


# =============================================================================
# HELPERS
# =============================================================================

def _read_wav_pcm_from_s3(bucket: str, key: str) -> tuple[bytes, int]:
    """Download a 16-bit PCM mono WAV from S3. Returns (raw_pcm_bytes, sample_rate)."""
    resp = _s3.get_object(Bucket=bucket, Key=key)
    raw = resp["Body"].read()
    return _parse_wav_pcm(raw, f"s3://{bucket}/{key}")


# =============================================================================
# MULTIPART UPLOAD  (default)
# =============================================================================

def _concat_stem_multipart(
    bucket:     str,
    iterators:  list[dict],
    stem:       str,
    out_key:    str,
) -> dict:
    """
    Stream-concatenate stem chunks into S3 via multipart upload.

    Memory usage: O(one chunk) per stem — independent of total duration.

    Strategy
    ────────
    1. Pre-calculate total PCM bytes from iterator metadata so the WAV
       header can be written correctly upfront (no fixup needed).
    2. Start multipart upload.
    3. Part 1 = WAV header (44 bytes) + chunk_000 PCM.
    4. Parts 2..N = subsequent chunk PCM only (header stripped).
       If a chunk's PCM is < 5 MB (S3 minimum), accumulate it with
       the next chunk before uploading as a single part.
    5. complete_multipart_upload.

    Returns a result dict compatible with the in-memory implementation.
    """
    # ── Step 1: pre-calculate total PCM size ─────────────────────────────────
    # duration_sec and resampl_freq are known from iterator results —
    # no S3 reads needed for sizing.
    sr_seen: int | None = None
    total_pcm_bytes = 0

    for it in iterators:
        stem_info = it.get("stems", {}).get(stem)
        if stem_info is None:
            raise RuntimeError(
                f"Iterator {it.get('chunk_name')} has no '{stem}' stem result"
            )
        sr = int(stem_info["resampl_freq"])
        dur = float(stem_info["duration_sec"])

        if sr_seen is None:
            sr_seen = sr
        elif sr != sr_seen:
            raise RuntimeError(
                f"Sample rate mismatch in {stem}: "
                f"{it.get('chunk_name')} has {sr} Hz, expected {sr_seen} Hz"
            )
        # 16-bit mono: 2 bytes per sample
        total_pcm_bytes += int(round(dur * sr)) * 2

    wav_header = encode_wav_header_mono(total_pcm_bytes // 2, sr_seen)
    total_wav_bytes = 44 + total_pcm_bytes
    envelope = audio_analysis.EnvelopeAccumulator(sr_seen)
    log.info(
        f"  [{stem}] multipart: sr={sr_seen}  "
        f"pcm={total_pcm_bytes:,} bytes  "
        f"wav={total_wav_bytes/1_048_576:.1f} MB  "
        f"chunks={len(iterators)}"
    )

    # ── Step 2: initiate multipart upload ────────────────────────────────────
    mpu = _s3.create_multipart_upload(
        Bucket=bucket,
        Key=out_key,
        ContentType="audio/wav",
    )
    upload_id = mpu["UploadId"]
    parts: list[dict] = []
    part_number = 0

    # RMS tracking (running sum of squares)
    sum_sq = 0.0
    n_samples = 0

    try:
        pending = b""   # accumulator for small chunks < S3 minimum part size

        for ci, it in enumerate(iterators):
            chunk_name = it.get("chunk_name", f"chunk_{it['chunk_idx']:03d}")
            s3_key = it["stems"][stem]["s3_key"]
            log.debug(
                f"    [{stem}] reading {chunk_name}: s3://{bucket}/{s3_key}")

            pcm_bytes, _ = _read_wav_pcm_from_s3(bucket, s3_key)

            # Track RMS incrementally
            pcm_arr = np.frombuffer(
                pcm_bytes, dtype=np.int16).astype(np.float32)
            sum_sq += float(np.sum(pcm_arr.astype(np.float64) ** 2))
            n_samples += len(pcm_arr)

            # Per-bin envelope for the analysis products rides the same
            # stream — the PCM is already in memory, so this is free.
            envelope.feed(pcm_arr / 32_768.0)

            # Prepend WAV header to very first chunk
            if ci == 0:
                pending = wav_header + pcm_bytes
            else:
                pending += pcm_bytes

            is_last = (ci == len(iterators) - 1)

            # Upload part when accumulated bytes exceed minimum OR it's the last chunk
            if len(pending) >= _S3_MIN_PART_BYTES or is_last:
                part_number += 1
                resp = _s3.upload_part(
                    Bucket=bucket,
                    Key=out_key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=pending,
                )
                parts.append({
                    "PartNumber": part_number,
                    "ETag":       resp["ETag"],
                })
                log.debug(
                    f"    [{stem}] part {part_number}: "
                    f"{len(pending):,} bytes  ETag={resp['ETag']}"
                )
                pending = b""

        # ── Step 5: complete ──────────────────────────────────────────────────
        _s3.complete_multipart_upload(
            Bucket=bucket,
            Key=out_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        log.info(
            f"  [{stem}] multipart complete: "
            f"{part_number} parts  s3://{bucket}/{out_key}"
        )

    except Exception:
        # Always abort on failure to avoid orphaned multipart uploads
        log.warning(f"  [{stem}] aborting multipart upload {upload_id}")
        _s3.abort_multipart_upload(
            Bucket=bucket, Key=out_key, UploadId=upload_id)
        raise

    # ── RMS ───────────────────────────────────────────────────────────────────
    rms_linear = math.sqrt(sum_sq / n_samples) / 32_768.0 if n_samples else 0.0
    rms_db = round(20.0 * math.log10(max(rms_linear, 1e-10)), 1)
    dur_sec = round(n_samples / sr_seen, 3) if sr_seen else 0.0

    summary = {
        "s3_key":       out_key,
        "resampl_freq": sr_seen,
        "rms_db":       rms_db,
        "duration_sec": dur_sec,
        "size_bytes":   total_wav_bytes,
    }
    # The envelope travels as its OWN return value — never inside the
    # JSON-shaped summary dict (2026-08-26 review: an ndarray smuggled
    # through a dict that later feeds put_json is a crash held off only
    # by statement order).
    return summary, envelope.finish()



def _parse_wav_pcm(raw: bytes, key: str) -> tuple[bytes, int]:
    """Extract raw PCM bytes and sample rate from an in-memory RIFF/WAVE blob."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {key}")
    pos = 12
    sr = None
    data_start = None
    data_size = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos: pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            sr = struct.unpack_from("<I", raw, body + 4)[0]
        elif chunk_id == b"data":
            data_start = body
            data_size = chunk_size
            break
        pos += 8 + chunk_size + (chunk_size & 1)
    if sr is None or data_start is None:
        raise ValueError(f"Could not parse WAV header: {key}")
    return raw[data_start: data_start + data_size], sr


# =============================================================================
# AUDIO ANALYSIS PRODUCTS  (stem_activity.json + stem_waveforms.json)
# =============================================================================

_ORIGINAL_WINDOW_SEC = 60.0     # byte-range read size for the original pass


def _analyze_original(bucket: str, key: str):
    """Stream the ORIGINAL mix (already 48 kHz mono on every path) through
    the envelope accumulator and the tone detector in bounded windows —
    one byte-range GET per window, O(window) memory.

    -> (rms_bins, tones[]). Tones are measured HERE and only here: which
    stem a pure sine lands in after separation is a model artifact, and
    the phase-coherent mono downmix upstream leaves a tone's frequency
    untouched.
    """
    info = get_wav_info(bucket, key)
    envelope = audio_analysis.EnvelopeAccumulator(info.sample_rate)
    tones = audio_analysis.ToneDetector(info.sample_rate)
    t = 0.0
    while t < info.duration_sec:
        window = read_wav_window(bucket, key, info, t, _ORIGINAL_WINDOW_SEC,
                                 target_sr=None)
        envelope.feed(window)
        tones.feed(window)
        t += _ORIGINAL_WINDOW_SEC
    return envelope.finish(), tones.finish()


def _write_analysis_products(
    output_bucket: str,
    output_prefix: str,
    lane_rms:      dict,      # lane name -> np.ndarray (linear RMS bins)
    tones:         list,
) -> dict:
    """Write the two products and return their keys.

    `activity` lanes: `original` means NON-SILENCE of the mix; a stem
    lane means stem energy above threshold. Lanes are the ACTIVE stems
    plus original — a disabled stem costs nothing anywhere, including
    here. There is deliberately NO bars_and_tone verdict in stem_activity:
    audiostem reports the measured tone; naming what a span MEANS is the
    fusion layer's job (README.md Design record §4).
    """
    params = audio_analysis.params_doc()

    activity_doc = {
        "params": params,
        "activity": {lane: audio_analysis.activity_spans(rms)
                     for lane, rms in lane_rms.items()},
        "tones": tones,
    }
    # speech_stem_gaps (Design record §5): spans where the mix is active
    # but the speech stem is near-silent — audio the stem does not cover
    # (an upper bound on deleted speech; music/sfx-only passages qualify).
    # Needs the RAW speech lane's linear envelope, so it is present
    # exactly when speech_stem was enabled (speech_safe alone does not
    # produce it: the safe lane is patched, so comparing it to the mix
    # would hide the very gaps the lane exists to show). Always written
    # when available (not gated on speech_safe) — the lane is what makes
    # the stem's losses VISIBLE on default runs.
    if "original" in lane_rms and "speech" in lane_rms:
        activity_doc["speech_stem_gaps"] = audio_analysis.stem_gap_spans(
            lane_rms["original"], lane_rms["speech"])
    elif "speech_safe" in lane_rms:
        log.warning(
            "speech_stem_gaps lane SKIPPED: speech_safe ran without "
            "speech_stem, so no raw speech envelope exists to compare "
            "against the mix — enable speech_stem to get the lane")

    activity_key = f"{output_prefix}/stem_activity.json"
    put_json(output_bucket, activity_key, activity_doc)
    log.info(f"  uploaded analysis: s3://{output_bucket}/{activity_key}")

    waveforms_key = f"{output_prefix}/stem_waveforms.json"
    put_json(output_bucket, waveforms_key, {
        "params": {"bin_ms": params["bin_ms"]},
        **{lane: {"rms": audio_analysis.waveform_values(rms)}
           for lane, rms in lane_rms.items()},
    })
    log.info(f"  uploaded analysis: s3://{output_bucket}/{waveforms_key}")

    return {"stem_activity_key": activity_key,
            "stem_waveforms_key": waveforms_key}


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateConcatenateAudioChunksByStems event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        data = event["data"]

        audio_bucket  = data["audio_bucket"]
        audio_key     = data["audio_key"]
        output_bucket = data["output_bucket"]
        output_prefix = data["output_prefix"]
        model_id      = data["model_id"]
        iterators     = data.get("iterators", [])

        if not iterators:
            raise ValueError(
                "data.iterators is empty — nothing to concatenate")

        # ── Validate all iterators completed ──────────────────────────────────
        failed = [
            it.get("chunk_name", f"chunk_{it.get('chunk_idx','?')}")
            for it in iterators
            if it.get("status") != "COMPLETED"
        ]
        if failed:
            raise RuntimeError(
                f"Cannot concatenate — {len(failed)} iterator(s) did not "
                f"complete successfully: {failed}"
            )

        # ── Sort iterators by chunk_idx ───────────────────────────────────────
        iterators = sorted(iterators, key=lambda x: int(x["chunk_idx"]))
        n_chunks = len(iterators)

        # ── Determine active stems ────────────────────────────────────────────
        active_stems = tuple(
            s for s in _ALL_STEMS if data[f"{s}_stem"]
        )
        # speech_safe (gated mix substitution — README.md Design record §5)
        # concatenates exactly like a stem: per-chunk WAVs under
        # segment_chunks/speech_safe/, final audio_speech_safe.wav.
        if data.get("speech_safe", False):
            active_stems = active_stems + ("speech_safe",)
        log.info(
            f"Concatenating {n_chunks} chunk(s)  stems={list(active_stems)}"
        )

        t_concat = time.monotonic()
        stem_outputs: dict[str, dict] = {}
        lane_rms: dict = {}

        for stem in active_stems:
            log.info(f"  Processing stem: {stem}")
            out_key = f"{output_prefix}/audio_{stem}.wav"
            t_stem = time.monotonic()

            result, lane_rms[stem] = _concat_stem_multipart(
                bucket=output_bucket,
                iterators=iterators,
                stem=stem,
                out_key=out_key,
            )

            stem_ms = round((time.monotonic() - t_stem) * 1_000, 1)
            log.info(
                f"  [{stem}] done: {stem_ms:.0f} ms  "
                f"dur={result['duration_sec']:.2f}s  "
                f"rms={result['rms_db']:.1f} dBFS  "
                f"{result['size_bytes']/1_048_576:.1f} MB"
            )
            stem_outputs[stem] = result

        # ── Audio-analysis products (activity + waveforms) ────────────────────
        # Stem envelopes rode the concatenation streams above; the original
        # gets its own windowed pass (it is the lane tone detection runs on,
        # and its activity spans mean non-silence of the mix). GUARDED: the
        # stem WAVs — the paid deliverable — are already uploaded, and an
        # auxiliary product must never turn that into a FAILED run
        # (2026-08-26 review). A degrade is loud (ERROR log, keys absent),
        # never silent.
        analysis_refs: dict = {}
        try:
            t_analysis = time.monotonic()
            original_rms, tones = _analyze_original(audio_bucket, audio_key)
            lane_rms = {"original": original_rms, **lane_rms}
            analysis_refs = _write_analysis_products(
                output_bucket, output_prefix, lane_rms, tones)
            log.info(
                f"  analysis done: "
                f"{round((time.monotonic()-t_analysis)*1000)} ms"
                f"  lanes={list(lane_rms)}  tones={len(tones)}"
            )
        except Exception as exc:                 # noqa: BLE001 — auxiliary
            log.error(
                "AUDIO ANALYSIS FAILED — stem WAVs are complete and the run "
                "continues; stem_activity/stem_waveforms are ABSENT from "
                "this run's outputs. %s: %s", type(exc).__name__, exc)

        # ── Aggregate timing across iterators ─────────────────────────────────
        total_wall_ms = max(
            it.get("timing_ms", {}).get("total", 0) for it in iterators)
        total_compute_ms = sum(
            it.get("timing_ms", {}).get("total", 0) for it in iterators)
        avg_chunk_ms = round(total_compute_ms / n_chunks, 1) if n_chunks else 0
        slowest = max(
            iterators,
            key=lambda x: x.get("timing_ms", {}).get("total", 0)
        )

        # ── Per-chunk summary for metrics JSON ────────────────────────────────
        chunk_summaries = []
        for it in iterators:
            chunk_summaries.append({
                "chunk_idx":       it["chunk_idx"],
                "chunk_name":      it.get("chunk_name"),
                "fetch_start_sec": it.get("fetch_start_sec"),
                "fetch_end_sec":   it.get("fetch_end_sec"),
                "timing_ms":       it.get("timing_ms", {}),
                "stems": {
                    s: {
                        "rms_db":       it.get("stems", {}).get(s, {}).get("rms_db"),
                        "duration_sec": it.get("stems", {}).get(s, {}).get("duration_sec"),
                    }
                    for s in active_stems
                },
                "metrics_key": it.get("metrics_key"),
                "elapsed_sec": it.get("elapsed_sec"),
            })

        total_elapsed = round(time.monotonic() - t0, 2)

        # ── Build and upload aggregate metrics JSON ───────────────────────────
        metrics_key = f"{output_prefix}/audiostem_sep.json"
        metrics_doc = {
            "model_id":       model_id,
            "audio_bucket":   audio_bucket,
            "audio_key":      audio_key,
            "total_chunks":   n_chunks,
            "stems_produced": list(active_stems),
            "output_bucket":  output_bucket,
            "output_keys": {
                stem: stem_outputs[stem]["s3_key"]
                for stem in active_stems
            },
            "stem_summary": stem_outputs,
            "timing_ms": {
                "total_wall":     round(total_wall_ms, 1),
                "total_compute":  round(total_compute_ms, 1),
                "avg_chunk":      avg_chunk_ms,
                "slowest_chunk":  slowest.get("chunk_name"),
            },
            "chunks":  chunk_summaries,
            "elapsed_sec": total_elapsed,
        }
        put_json(output_bucket, metrics_key, metrics_doc)
        log.info(f"  uploaded metrics: s3://{output_bucket}/{metrics_key}")

        # ── Return enriched payload ───────────────────────────────────────────
        output_refs = {
            f"{stem}_key": stem_outputs[stem]["s3_key"]
            for stem in active_stems
        }
        output_refs["metrics_key"] = metrics_key
        output_refs.update(analysis_refs)

        log.info(
            f"DONE  chunks={n_chunks}  stems={list(active_stems)}  "
            f"elapsed={total_elapsed}s"
        )

        return {
            **event,
            "data": {
                **data,
                "audiostem_sep_output": output_refs,
                "iterators": None,
            },
        }

    except Exception as exc:
        log.error(
            "[ERR] StateConcatenateAudioChunksByStems — %s: %s",
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise
