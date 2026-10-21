# states/concatenate_enhanced_audio/lambda_function.py
"""
Step Function State: Concatenate enhanced audio
───────────────────────────────────────────────
Runs after the Map state completes. Streams each enhanced chunk WAV
from S3, in chunk order, into ONE S3 multipart upload, and writes an
aggregate metrics JSON. Seam quality was already settled upstream: the
model state enhanced each chunk with ±1 s context and trimmed it, so
this concatenation is byte-plain.

Multipart streaming (the family pattern, shared with audiostem)
───────────────────────────────────────────────────────────────
  Peak memory is O(one chunk) regardless of total duration. Total PCM
  bytes are pre-calculated from iterator metadata so the WAV header is
  written upfront — no fixup pass. Parts accumulate to the 5 MB S3
  minimum before uploading; the final part is exempt.

Outputs under {output_prefix}/:
  audio_enhanced.wav        the concatenated enhanced audio
  speech_enhancement.json   aggregate metrics (per-chunk summaries,
                            RMS in/out, timings)

data gains:
  enhancement_output: {enhanced_key, resampl_freq, rms_db_in,
                       rms_db_out, duration_sec, size_bytes,
                       metrics_key}
"""
from __future__ import annotations

import json
import logging
import math
import struct
import time

import boto3
import numpy as np

from lib.wav_utils import encode_wav_header_mono

_s3 = boto3.client("s3")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)

_S3_MIN_PART_BYTES = 5 * 1024 * 1024


def _parse_wav_pcm(raw: bytes, key: str) -> tuple[bytes, int]:
    """Extract raw PCM bytes and sample rate from a RIFF/WAVE blob."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {key}")
    pos = 12
    sr = data_start = data_size = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            sr = struct.unpack_from("<I", raw, body + 4)[0]
        elif chunk_id == b"data":
            data_start, data_size = body, chunk_size
            break
        pos += 8 + chunk_size + (chunk_size & 1)
    if sr is None or data_start is None:
        raise ValueError(f"Could not parse WAV header: {key}")
    return raw[data_start:data_start + data_size], sr


def _read_wav_pcm_from_s3(bucket: str, key: str) -> tuple[bytes, int]:
    raw = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return _parse_wav_pcm(raw, f"s3://{bucket}/{key}")


def _concat_multipart(bucket: str, iterators: list[dict],
                      out_key: str) -> dict:
    """Stream-concatenate enhanced chunk WAVs; O(one chunk) memory."""
    sr_seen = None
    total_samples = 0
    for it in iterators:
        enh = it["enhanced"]
        sr = int(enh["resampl_freq"])
        if sr_seen is None:
            sr_seen = sr
        elif sr != sr_seen:
            raise RuntimeError(
                f"Sample rate mismatch: {it.get('chunk_name')} has "
                f"{sr} Hz, expected {sr_seen} Hz")
        # fallback covers a redrive whose Map outputs predate n_samples
        total_samples += int(enh.get("n_samples")
                             or round(float(enh["duration_sec"]) * sr))

    wav_header = encode_wav_header_mono(total_samples, sr_seen)
    total_wav_bytes = len(wav_header) + total_samples * 2
    log.info("Multipart: sr=%d  samples=%s  wav=%.1f MB  chunks=%d",
             sr_seen, f"{total_samples:,}",
             total_wav_bytes / 1_048_576, len(iterators))

    mpu = _s3.create_multipart_upload(Bucket=bucket, Key=out_key,
                                      ContentType="audio/wav")
    upload_id = mpu["UploadId"]
    parts, part_number = [], 0
    sum_sq = 0.0
    n_samples = 0
    sum_sq_in = 0.0
    n_samples_in = 0

    try:
        pending = b""
        for ci, it in enumerate(iterators):
            enh = it["enhanced"]
            pcm_bytes, _ = _read_wav_pcm_from_s3(bucket, enh["s3_key"])

            pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
            sum_sq += float(np.sum(pcm.astype(np.float64) ** 2))
            n_samples += len(pcm)

            rms_db_in = enh.get("rms_db_in")
            if rms_db_in is not None and rms_db_in > -119.0:
                n_in = int(enh.get("n_samples")
                           or round(float(enh["duration_sec"]) *
                                    int(enh["resampl_freq"])))
                sum_sq_in += (10 ** (rms_db_in / 20.0)) ** 2 * n_in
                n_samples_in += n_in

            pending = (wav_header + pcm_bytes) if ci == 0 \
                else pending + pcm_bytes
            if len(pending) >= _S3_MIN_PART_BYTES or ci == len(iterators) - 1:
                part_number += 1
                resp = _s3.upload_part(Bucket=bucket, Key=out_key,
                                       UploadId=upload_id,
                                       PartNumber=part_number, Body=pending)
                parts.append({"PartNumber": part_number,
                              "ETag": resp["ETag"]})
                pending = b""

        _s3.complete_multipart_upload(
            Bucket=bucket, Key=out_key, UploadId=upload_id,
            MultipartUpload={"Parts": parts})
        log.info("Multipart complete: %d parts  s3://%s/%s",
                 part_number, bucket, out_key)
    except Exception:
        log.warning("Aborting multipart upload %s", upload_id)
        _s3.abort_multipart_upload(Bucket=bucket, Key=out_key,
                                   UploadId=upload_id)
        raise

    rms_out_lin = (math.sqrt(sum_sq / n_samples) / 32_768.0) \
        if n_samples else 0.0
    rms_db_out = round(20.0 * math.log10(max(rms_out_lin, 1e-10)), 1)
    rms_db_in_agg = None
    if n_samples_in > 0:
        rms_db_in_agg = round(
            20.0 * math.log10(
                max(math.sqrt(sum_sq_in / n_samples_in), 1e-10)), 1)

    return {"resampl_freq": sr_seen,
            "rms_db_in": rms_db_in_agg,
            "rms_db_out": rms_db_out,
            "duration_sec": round(n_samples / sr_seen, 3) if sr_seen else 0.0,
            "size_bytes": total_wav_bytes}


def lambda_handler(event: dict, context) -> dict:
    log.info("StateConcatenateEnhancedAudio event keys = %s",
             sorted(event.keys()))
    t0 = time.monotonic()

    data = event.get("data", {})
    output_bucket = data["output_bucket"]
    output_prefix = data["output_prefix"].strip("/")
    iterators = data.get("iterators") or []
    if not iterators:
        raise ValueError("data.iterators is empty — nothing to concatenate")

    failed = [it.get("chunk_name", f"chunk_{it.get('chunk_idx', '?')}")
              for it in iterators if it.get("status") != "COMPLETED"]
    if failed:
        raise RuntimeError(
            f"Cannot concatenate — {len(failed)} iterator(s) did not "
            f"complete: {failed}")

    iterators = sorted(iterators, key=lambda x: int(x["chunk_idx"]))
    enhanced_key = f"{output_prefix}/audio_enhanced.wav"

    t_concat = time.monotonic()
    result = _concat_multipart(output_bucket, iterators, enhanced_key)
    concat_ms = round((time.monotonic() - t_concat) * 1_000, 1)
    log.info("Concat done: %.0fms  dur=%.2fs  rms %s → %s dBFS",
             concat_ms, result["duration_sec"], result["rms_db_in"],
             result["rms_db_out"])

    total_compute_ms = sum(it.get("timing_ms", {}).get("total", 0)
                           for it in iterators)
    slowest = max(iterators,
                  key=lambda x: x.get("timing_ms", {}).get("total", 0))
    chunk_summaries = [{
        "chunk_idx": it["chunk_idx"],
        "chunk_name": it.get("chunk_name"),
        "fetch_start_sec": it.get("fetch_start_sec"),
        "fetch_end_sec": it.get("fetch_end_sec"),
        "rms_db_in": it["enhanced"].get("rms_db_in"),
        "rms_db_out": it["enhanced"].get("rms_db_out"),
        "duration_sec": it["enhanced"].get("duration_sec"),
        "timing_ms": it.get("timing_ms", {}),
        "metrics_key": it.get("metrics_key"),
        "elapsed_sec": it.get("elapsed_sec"),
    } for it in iterators]

    metrics_key = f"{output_prefix}/speech_enhancement.json"
    enhancement_output = {"enhanced_key": enhanced_key,
                          **result, "metrics_key": metrics_key}
    metrics_doc = {
        "model_id": data.get("model_id", "deepfilternet3"),
        "audio_key": data.get("audio_key"),
        "enhanced_key": enhanced_key,
        "total_chunks": len(iterators),
        "output": {"s3_key": enhanced_key, **result},
        "timing_ms": {
            "concat": concat_ms,
            "total_compute": round(total_compute_ms, 1),
            "avg_chunk": round(total_compute_ms / len(iterators), 1),
            "slowest_chunk": slowest.get("chunk_name"),
        },
        "chunks": chunk_summaries,
        "status": "COMPLETED",
        "elapsed_sec": round(time.monotonic() - t0, 2),
    }
    _s3.put_object(Bucket=output_bucket, Key=metrics_key,
                   Body=json.dumps(metrics_doc, indent=2).encode(),
                   ContentType="application/json")

    log.info("DONE  chunks=%d  elapsed=%.2fs", len(iterators),
             time.monotonic() - t0)
    return {**event,
            "data": {**data, "enhancement_output": enhancement_output},
            "status": "COMPLETED"}
