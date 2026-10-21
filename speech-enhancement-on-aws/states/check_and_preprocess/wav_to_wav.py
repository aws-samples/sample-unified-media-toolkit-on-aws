"""
In-Lambda WAV normalisation: any WAV → 48 kHz mono int16 WAV.

Handles source files that are WAV but incompatible with the speech-enhancement pipeline
(wrong sample rate, multi-channel, or both). Avoids downloading the full file —
reads and writes in chunks via S3 byte-range GET and multipart upload.

Public API
──────────
  convert(src_bucket, src_key, dst_bucket, dst_key, src_info,
          target_sr=TARGET_SR, num_threads=8) -> audio_cfg dict

Routing
───────
  total output bytes ≤ SINGLE_PUT_THRESHOLD  →  single put_object (no multipart)
  total output bytes  > SINGLE_PUT_THRESHOLD  →  multipart upload with num_threads

SINGLE_PUT_THRESHOLD is 10 MB ≈ 5.5 min at 48 kHz 16-bit mono.  Files above the
threshold are split into num_threads equal-duration parts, each guaranteed to
produce ≥ 5 MB of output (the S3 minimum per-part size).

Multipart strategy
──────────────────
  total_output_samples is computed analytically before any S3 read:
    total_output_samples = (src_info.total_frames * target_sr
                            // src_info.sample_rate)   # exact int math

  This allows the 44-byte WAV header to be built upfront.  All threads start
  concurrently.  Thread 0 prepends the WAV header to its PCM bytes as part 1;
  threads 1..N-1 upload raw PCM only.

Peak memory per thread
──────────────────────
  ~3 × raw_input_chunk_bytes  (raw bytes + multi-channel float32 decode + mono result)
  For 8 threads on a 4-hour 96 kHz stereo file ≈ 8 × ~363 MB.
  3584 MB Lambda allocation handles this comfortably.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

from lib.input_schema import TARGET_SR
from lib.s3_utils import s3
from lib.wav_utils import WavInfo, read_wav_window, encode_wav_header_mono

log = logging.getLogger(__name__)

# Files whose output is ≤ this threshold use a single put_object.
# 10 MB ≈ 5.5 min at 48 kHz 16-bit mono — comfortably above the 5 MB S3
# minimum so multipart is never triggered for short files.
SINGLE_PUT_THRESHOLD: int = 10 * 1024 * 1024   # 10 MB

# Memory cap (ported from speech-diarization 2026-09-28): long sources
# used to split into at most num_threads parts — a handful of
# 15-minute high-rate windows resampling concurrently exhausts the
# Lambda. Parts are time-capped and the pool bounds concurrency.
MAX_PART_SEC: float = 200.0

# The polyphase resampler is an FIR filter: without context beyond the
# window edges every part boundary carries a filter-edge transient, so
# each part reads a clamped, decimation-grid-aligned pad on both sides
# and trims it in the output domain.
_EDGE_PAD_SEC: float = 0.1

# S3 multipart minimum part size (all parts except the last).
_S3_MIN_PART_BYTES: int = 5 * 1024 * 1024       # 5 MB

# WAV header is always 44 bytes for a standard RIFF PCM file.
_WAV_HEADER_BYTES: int = 44


# =============================================================================
# AUDIO PROCESSING — single chunk
# =============================================================================

def _read_and_encode(
    src_bucket:  str,
    src_key:     str,
    src_info:    WavInfo,
    start_sec:   float,
    duration_sec: float,
    target_sr:   int,
) -> bytes:
    """
    Read one time-window from S3, downmix to mono, resample, encode int16.
    Returns raw PCM bytes (no WAV header).
    """
    src_sr = src_info.sample_rate
    # frame math mirrors read_wav_window so the trimmed output length
    # equals what the unpadded window produced before (the multipart
    # header size is precomputed from the same convention)
    core_start_f = int(start_sec * src_sr)
    core_end_f = int(min(start_sec + duration_sec,
                         src_info.duration_sec) * src_sr)
    if src_sr == target_sr:
        audio = read_wav_window(
            src_bucket, src_key, src_info,
            start_sec=start_sec, duration_sec=duration_sec,
            target_sr=target_sr,
        )
    else:
        from fractions import Fraction
        frac = Fraction(target_sr, src_sr).limit_denominator(10_000)
        total_f = int(src_info.duration_sec * src_sr)
        pre_f = min(int(_EDGE_PAD_SEC * src_sr), core_start_f)
        pre_f -= pre_f % frac.denominator      # keep the decimation
        # grid of the padded slice in phase with the unpadded window
        post_f = min(int(_EDGE_PAD_SEC * src_sr),
                     max(0, total_f - core_end_f))
        # frame-exact seconds: the quarter-frame bias survives the
        # float round trip in read_wav_window's int(sec * sr)
        pad_start_f = core_start_f - pre_f
        pad_end_f = core_end_f + post_f
        audio = read_wav_window(
            src_bucket, src_key, src_info,
            start_sec=(pad_start_f + 0.25) / src_sr,
            duration_sec=(pad_end_f - pad_start_f + 0.5) / src_sr,
            target_sr=target_sr,
        )
        n_pre_out = pre_f * frac.numerator // frac.denominator
        n_out = int((core_end_f - core_start_f) * target_sr / src_sr)
        audio = audio[n_pre_out:n_pre_out + n_out]
        if len(audio) < n_out:
            audio = np.pad(audio, (0, n_out - len(audio)))
    # read_wav_window returns float32 in [-1, 1], already mono, already resampled
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32_767.0).astype(np.int16)
    return pcm.tobytes()


# =============================================================================
# SINGLE PUT PATH
# =============================================================================

def _convert_single(
    src_bucket:   str,
    src_key:      str,
    dst_bucket:   str,
    dst_key:      str,
    src_info:     WavInfo,
    target_sr:    int,
    total_pcm_bytes: int,
) -> None:
    """Read entire file, build WAV in memory, single put_object."""
    log.info("convert: single PUT  (%d bytes)", _WAV_HEADER_BYTES + total_pcm_bytes)
    pcm = _read_and_encode(
        src_bucket, src_key, src_info,
        start_sec=0.0,
        duration_sec=src_info.duration_sec,
        target_sr=target_sr,
    )
    header = encode_wav_header_mono(len(pcm) // 2, target_sr)
    s3.put_object(
        Bucket=dst_bucket, Key=dst_key,
        Body=header + pcm,
        ContentType="audio/wav",
    )


# =============================================================================
# MULTIPART PATH
# =============================================================================

def _convert_multipart(
    src_bucket:   str,
    src_key:      str,
    dst_bucket:   str,
    dst_key:      str,
    src_info:     WavInfo,
    target_sr:    int,
    total_pcm_bytes: int,
    num_threads:  int,
) -> None:
    """Concurrent multipart upload — one thread per S3 part."""
    # ── Compute part layout ───────────────────────────────────────────────────
    # n_parts: guarantee every part (except last) produces ≥ 5 MB output PCM.
    # Cap at num_threads.
    import math
    n_parts = max(1, math.ceil(src_info.duration_sec / MAX_PART_SEC))
    # every non-last part must still emit >= 5 MB of output PCM
    n_parts = min(n_parts, max(1, total_pcm_bytes // _S3_MIN_PART_BYTES))

    total_input_frames = src_info.total_frames
    frames_per_part = total_input_frames // n_parts
    # input duration covered per part
    sec_per_part = frames_per_part / src_info.sample_rate

    log.info(
        "convert: multipart  parts=%d  threads=%d  "
        "total_pcm=%d bytes  sec_per_part=%.1fs",
        n_parts, n_parts, total_pcm_bytes, sec_per_part,
    )

    wav_header = encode_wav_header_mono(total_pcm_bytes // 2, target_sr)

    # ── Start multipart upload ────────────────────────────────────────────────
    mpu = s3.create_multipart_upload(
        Bucket=dst_bucket, Key=dst_key, ContentType="audio/wav")
    upload_id = mpu["UploadId"]

    parts: list[dict] = [{}] * n_parts   # pre-sized; filled by threads

    def _upload_part(part_idx: int) -> None:
        # integer-frame boundaries; the quarter-frame bias survives the
        # float round trip in _read_and_encode's int(sec * sr)
        src_sr = src_info.sample_rate
        in_start_f = part_idx * frames_per_part
        in_end_f = (total_input_frames if part_idx == n_parts - 1
                    else in_start_f + frames_per_part)
        start_sec = (in_start_f + 0.25) / src_sr
        duration = (in_end_f - in_start_f + 0.5) / src_sr

        pcm = _read_and_encode(
            src_bucket, src_key, src_info,
            start_sec=start_sec,
            duration_sec=duration,
            target_sr=target_sr,
        )
        # cumulative-exact output length: each part's share is the
        # DIFFERENCE of floor-scaled cumulative positions, so the parts
        # sum to the header's sample count exactly — per-part truncation
        # was short up to n_parts-1 samples on non-integer ratios,
        # leaving the RIFF data_size claiming bytes that never uploaded
        want = 2 * (in_end_f * target_sr // src_sr
                    - in_start_f * target_sr // src_sr)
        if len(pcm) > want:
            pcm = pcm[:want]
        elif len(pcm) < want:
            pcm = pcm + b"\x00" * (want - len(pcm))
        body = (wav_header + pcm) if part_idx == 0 else pcm
        part_number = part_idx + 1   # S3 part numbers are 1-based

        resp = s3.upload_part(
            Bucket=dst_bucket, Key=dst_key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=body,
        )
        parts[part_idx] = {"PartNumber": part_number, "ETag": resp["ETag"]}
        log.debug("  part %d/%d uploaded  bytes=%d", part_number, n_parts, len(body))

    try:
        with ThreadPoolExecutor(
                max_workers=min(num_threads, n_parts)) as pool:
            futures = {pool.submit(_upload_part, i): i for i in range(n_parts)}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    raise exc

        s3.complete_multipart_upload(
            Bucket=dst_bucket, Key=dst_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        log.info("convert: multipart complete  s3://%s/%s", dst_bucket, dst_key)

    except Exception:
        s3.abort_multipart_upload(
            Bucket=dst_bucket, Key=dst_key, UploadId=upload_id)
        raise


# =============================================================================
# PUBLIC API
# =============================================================================

def convert(
    src_bucket:  str,
    src_key:     str,
    dst_bucket:  str,
    dst_key:     str,
    src_info:    WavInfo,
    target_sr:   int = TARGET_SR,
    num_threads: int = 8,
) -> dict:
    """
    Convert any WAV to 48 kHz mono int16 WAV and upload to S3.

    Parameters
    ──────────
    src_info    : WavInfo already fetched by check_source() — no re-read needed.
    target_sr   : output sample rate (default TARGET_SR = 48000 Hz).
    num_threads : max concurrent S3 parts for the multipart path.

    Returns
    ───────
    audio_cfg dict: {sample_rate, num_channels, bits_per_sample, duration_sec}
    """
    # Pre-compute output size analytically — no S3 read required.
    total_output_samples = (src_info.total_frames * target_sr
                            // src_info.sample_rate)   # exact int math
    total_pcm_bytes = total_output_samples * 2      # int16 = 2 bytes/sample
    duration_sec = round(total_output_samples / target_sr, 3)

    log.info(
        "wav_to_wav.convert: s3://%s/%s  "
        "src=[%d Hz, %d ch, %.1fs]  dst=[%d Hz, mono]  "
        "output_bytes=%d  path=%s",
        src_bucket, src_key,
        src_info.sample_rate, src_info.num_channels, src_info.duration_sec,
        target_sr,
        _WAV_HEADER_BYTES + total_pcm_bytes,
        "single" if total_pcm_bytes <= SINGLE_PUT_THRESHOLD else "multipart",
    )

    if total_pcm_bytes <= SINGLE_PUT_THRESHOLD:
        _convert_single(
            src_bucket, src_key, dst_bucket, dst_key,
            src_info, target_sr, total_pcm_bytes,
        )
    else:
        _convert_multipart(
            src_bucket, src_key, dst_bucket, dst_key,
            src_info, target_sr, total_pcm_bytes, num_threads,
        )

    return {
        "sample_rate":     target_sr,
        "num_channels":    1,
        "bits_per_sample": 16,
        "duration_sec":    duration_sec,
    }
