"""
Minimal WAV header parser for S3-resident 16-bit PCM mono files.
No external media dependencies — pure Python struct parsing.
"""
from __future__ import annotations

import struct
import logging

from lib.s3_utils import s3 as _s3

log = logging.getLogger(__name__)

_HEADER_READ_BYTES = 256   # enough to find fmt + data chunks in any standard WAV


def _parse_wav_header(raw: bytes, label: str) -> tuple[int, int, int]:
    """
    Parse a RIFF/WAVE header from raw bytes.
    Returns (sample_rate, data_offset, data_size).
    data_offset is the byte position of the first PCM sample within `raw`;
    caller may need to fetch the full object if data_offset >= len(raw).
    """
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {label}")

    pos = 12
    sr: int | None = None
    data_offset: int | None = None
    data_size:   int | None = None

    while pos + 8 <= len(raw):
        chunk_id   = raw[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        body       = pos + 8

        if chunk_id == b"fmt ":
            sr = struct.unpack_from("<I", raw, body + 4)[0]
        elif chunk_id == b"data":
            data_offset = body
            data_size   = chunk_size
            break
        pos += 8 + chunk_size + (chunk_size & 1)

    if sr is None or data_offset is None or data_size is None:
        raise ValueError(f"Could not parse WAV header: {label}")

    return sr, data_offset, data_size


def read_wav_header(bucket: str, key: str) -> tuple[int, int]:
    """
    Byte-range GET the first _HEADER_READ_BYTES of a WAV to read its header.
    Returns (sample_rate, pcm_data_size_bytes).
    Does NOT download the PCM body.
    """
    label = f"s3://{bucket}/{key}"
    resp = _s3.get_object(
        Bucket=bucket, Key=key,
        Range=f"bytes=0-{_HEADER_READ_BYTES - 1}",
    )
    raw = resp["Body"].read()
    sr, _, data_size = _parse_wav_header(raw, label)
    return sr, data_size


def read_wav_pcm(bucket: str, key: str) -> tuple[bytes, int]:
    """
    Full GET of a WAV file. Returns (pcm_bytes, sample_rate) — header stripped.
    """
    label = f"s3://{bucket}/{key}"
    resp  = _s3.get_object(Bucket=bucket, Key=key)
    raw   = resp["Body"].read()
    sr, data_offset, data_size = _parse_wav_header(raw, label)
    return raw[data_offset: data_offset + data_size], sr


def make_wav_header(sr: int, total_pcm_bytes: int) -> bytes:
    """44-byte standard RIFF/WAVE PCM header for a mono 16-bit file."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + total_pcm_bytes, b"WAVE",
        b"fmt ",  16, 1, 1, sr, sr * 2, 2, 16,
        b"data",  total_pcm_bytes,
    )


# ── S3 multipart WAV concatenation ────────────────────────────────────────────

_S3_MIN_PART_BYTES = 5 * 1024 * 1024   # 5 MB — S3 multipart minimum part size


def concat_wav(bucket: str, keys: list[str], out_key: str) -> None:
    """
    Stream-concatenate PCM WAV chunks into a single S3 object via multipart upload.
    No local file write — peak memory is O(one chunk), independent of total duration.

    Pass 1: byte-range GET each chunk header → sum total PCM bytes → build WAV
            header upfront (no fixup pass needed).
    Pass 2: multipart upload — Part 1 = header + chunk_0 PCM; subsequent parts
            accumulate until ≥ 5 MB before upload (S3 minimum part constraint).
    """
    # ── Pass 1: sum total PCM bytes ────────────────────────────────────────────
    sr_seen:         int | None = None
    total_pcm_bytes: int = 0
    for key in keys:
        sr, pcm_size = read_wav_header(bucket, key)
        if sr_seen is None:
            sr_seen = sr
        elif sr != sr_seen:
            raise RuntimeError(
                f"Sample rate mismatch: {key} has {sr} Hz, expected {sr_seen} Hz"
            )
        total_pcm_bytes += pcm_size

    wav_header = make_wav_header(sr_seen, total_pcm_bytes)
    log.info(
        "concat_wav: %d chunks  sr=%d  pcm=%d bytes  wav=%.1f MB",
        len(keys), sr_seen, total_pcm_bytes,
        (44 + total_pcm_bytes) / 1_048_576,
    )

    # ── Pass 2: multipart upload ───────────────────────────────────────────────
    mpu       = _s3.create_multipart_upload(Bucket=bucket, Key=out_key, ContentType="audio/wav")
    upload_id = mpu["UploadId"]
    parts: list[dict] = []
    part_number = 0

    try:
        pending = b""
        for ci, key in enumerate(keys):
            pcm_bytes, _ = read_wav_pcm(bucket, key)
            pending = (wav_header + pcm_bytes) if ci == 0 else (pending + pcm_bytes)

            is_last = ci == len(keys) - 1
            if len(pending) >= _S3_MIN_PART_BYTES or is_last:
                part_number += 1
                resp = _s3.upload_part(
                    Bucket=bucket, Key=out_key,
                    UploadId=upload_id, PartNumber=part_number, Body=pending,
                )
                parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
                log.debug("part %d: %d bytes", part_number, len(pending))
                pending = b""

        _s3.complete_multipart_upload(
            Bucket=bucket, Key=out_key,
            UploadId=upload_id, MultipartUpload={"Parts": parts},
        )
        log.info("concat_wav complete: %d parts  s3://%s/%s", part_number, bucket, out_key)

    except Exception:
        log.warning("aborting multipart upload %s", upload_id)
        _s3.abort_multipart_upload(Bucket=bucket, Key=out_key, UploadId=upload_id)
        raise
