# lib/wav_utils.py
"""
WAV header parsing and S3 byte-range audio reader.

Supported formats
─────────────────
  PCM integer  : 8-bit unsigned, 16-bit, 24-bit, 32-bit signed
  IEEE float   : 32-bit, 64-bit
  WAVE_FORMAT_EXTENSIBLE (0xFFFE) — reads SubFormat for real codec
  Multi-channel: downmixed to mono by averaging
  Any sample rate: optionally resampled to target_sr (default 48 kHz) via linear interp

No ffmpeg, no libav, no soundfile — only struct + numpy + the shared S3 client
from lib/s3_utils.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

from lib.input_schema import TARGET_SR
from lib.s3_utils import s3 as _s3

log = logging.getLogger(__name__)


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass(frozen=True)
class WavInfo:
    sample_rate:     int
    num_channels:    int
    bits_per_sample: int
    audio_format:    int    # 1 = PCM int, 3 = IEEE float
    data_offset:     int    # byte index of first audio sample in the file
    data_size:       int    # total bytes in the data chunk
    total_frames:    int    # sample frames  (samples / num_channels)
    duration_sec:    float
    bytes_per_frame: int    # num_channels × (bits_per_sample // 8)


# =============================================================================
# WAV HEADER PARSER
# =============================================================================

def parse_wav_header(raw: bytes) -> WavInfo:
    """
    Parse a RIFF/WAVE header from the first N bytes of a file.

    Correctly handles
    ─────────────────
    • Non-standard chunk ordering (JUNK, LIST, BEXT, iXML, … before 'data')
    • WAVE_FORMAT_EXTENSIBLE (0xFFFE) — reads the 2-byte SubFormat GUID
    • Word-alignment padding between chunks (chunk_size + 1 if odd)
    """
    if len(raw) < 44:
        raise ValueError(f"Buffer too short: {len(raw)} bytes (need ≥ 44)")
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(
            f"Not a RIFF/WAVE file "
            f"(magic bytes: {raw[:4]!r} / {raw[8:12]!r})"
        )

    pos = 12        # skip RIFF header
    audio_format = None
    num_channels = None
    sample_rate = None
    bits_per_sample = None
    data_offset = None
    data_size = None

    while pos + 8 <= len(raw):
        chunk_id = raw[pos: pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        body = pos + 8

        if chunk_id == b"fmt ":
            audio_format = struct.unpack_from("<H", raw, body)[0]
            num_channels = struct.unpack_from("<H", raw, body + 2)[0]
            sample_rate = struct.unpack_from("<I", raw, body + 4)[0]
            bits_per_sample = struct.unpack_from("<H", raw, body + 14)[0]
            # WAVE_FORMAT_EXTENSIBLE: SubFormat GUID bytes 0-1 = real format
            if audio_format == 0xFFFE and chunk_size >= 40:
                audio_format = struct.unpack_from("<H", raw, body + 20)[0]

        elif chunk_id == b"data":
            data_offset = body
            data_size = chunk_size
            break   # data chunk is always last — stop scanning

        # WAV chunks are word-aligned (pad byte if chunk_size is odd)
        pos += 8 + chunk_size + (chunk_size & 1)

    # ── Validation ────────────────────────────────────────────────────────────
    if audio_format is None:
        raise ValueError("WAV fmt chunk not found in the header bytes read")
    if data_offset is None:
        raise ValueError(
            "WAV data chunk not found within the header bytes read. "
            "The file may have an unusually large metadata section. "
            "Try increasing the header read size."
        )
    if audio_format not in (1, 3):
        raise ValueError(
            f"Unsupported audio format 0x{audio_format:04X}. "
            f"Only PCM (1) and IEEE-float (3) are supported."
        )
    if bits_per_sample not in (8, 16, 24, 32, 64):
        raise ValueError(f"Unsupported bit depth: {bits_per_sample}")
    if num_channels < 1:
        raise ValueError(f"Invalid channel count: {num_channels}")

    bpf = num_channels * (bits_per_sample // 8)
    n = data_size // bpf if bpf > 0 else 0

    return WavInfo(
        sample_rate=sample_rate,
        num_channels=num_channels,
        bits_per_sample=bits_per_sample,
        audio_format=audio_format,
        data_offset=data_offset,
        data_size=data_size,
        total_frames=n,
        duration_sec=n / sample_rate if sample_rate > 0 else 0.0,
        bytes_per_frame=bpf,
    )


# =============================================================================
# S3 HEADER READ
# =============================================================================

_HEADER_READ_SIZES = (4_096, 65_536)   # try small first; fall back to large


def get_wav_info(bucket: str, key: str) -> WavInfo:
    """
    Read the WAV header from S3 using a byte-range GET (≤ 64 KB).
    No audio data is transferred.

    Tries 4 KB first (covers 99 %+ of files); falls back to 64 KB for
    files with unusually large metadata chunks.
    """
    last_exc: Optional[Exception] = None
    for size in _HEADER_READ_SIZES:
        try:
            resp = _s3.get_object(
                Bucket=bucket, Key=key,
                Range=f"bytes=0-{size - 1}"
            )
            raw = resp["Body"].read()
            info = parse_wav_header(raw)
            log.info(
                f"WAV [{key}]: {info.duration_sec:.1f}s | "
                f"{info.sample_rate} Hz | {info.num_channels}ch | "
                f"{info.bits_per_sample}-bit | "
                f"data_offset={info.data_offset}"
            )
            return info
        except ValueError as exc:
            last_exc = exc
            if "data chunk not found" in str(exc).lower():
                log.warning(
                    f"WAV data chunk not in first {size} bytes; "
                    f"retrying with {_HEADER_READ_SIZES[-1]} bytes"
                )
                continue
            raise   # other parse errors are fatal

    raise ValueError(
        f"Could not locate WAV data chunk in the first "
        f"{_HEADER_READ_SIZES[-1]} bytes of s3://{bucket}/{key}"
    ) from last_exc


# =============================================================================
# S3 BYTE-RANGE AUDIO READER
# =============================================================================

def read_wav_window(
    bucket:       str,
    key:          str,
    info:         WavInfo,
    start_sec:    float,
    duration_sec: float,
    target_sr:    Optional[int] = TARGET_SR,
) -> np.ndarray:
    """
    Fetch a time-bounded audio slice from S3 using a single byte-range GET.

    Returns a float32 mono array. By default resampled to TARGET_SR (48 kHz)
    to preserve compatibility with STT/VAD callers. Pass target_sr=None to
    skip resampling and return audio at the file's native sample rate, or pass
    any int to resample to a specific rate.

    Parameters
    ──────────
    start_sec    : start position in the original file (seconds)
    duration_sec : length to read (seconds); clamped to file end
    target_sr    : output sample rate in Hz, or None to skip resampling
                   (default: TARGET_SR = 48000)
    """
    start_sec = max(0.0, min(start_sec, info.duration_sec))
    end_sec = min(start_sec + duration_sec, info.duration_sec)

    if end_sec <= start_sec:
        log.warning(
            f"read_wav_window: empty range "
            f"[{start_sec:.2f}s, {end_sec:.2f}s] — returning zeros"
        )
        return np.zeros(0, dtype=np.float32)

    start_frame = int(start_sec * info.sample_rate)
    end_frame = int(end_sec * info.sample_rate)

    byte_start = info.data_offset + start_frame * info.bytes_per_frame
    byte_end = info.data_offset + end_frame * info.bytes_per_frame - 1

    n_bytes = byte_end - byte_start + 1
    log.info(
        f"  S3 byte-range GET: {n_bytes:,} bytes "
        f"[{start_sec:.2f}s → {end_sec:.2f}s]  "
        f"s3://{bucket}/{key}"
    )

    resp = _s3.get_object(
        Bucket=bucket, Key=key,
        Range=f"bytes={byte_start}-{byte_end}"
    )
    raw = resp["Body"].read()
    audio = _decode_pcm(raw, info)

    # Multi-channel → mono
    if info.num_channels > 1:
        audio = (
            audio
            .reshape(-1, info.num_channels)
            .mean(axis=1)
            .astype(np.float32)
        )

    # Resample if caller requested a specific target sample rate
    if target_sr is not None and info.sample_rate != target_sr:
        log.info(
            f"  Resampling {info.sample_rate} Hz -> {target_sr} Hz"
        )
        audio = resample(audio, info.sample_rate, target_sr)

    return audio


# =============================================================================
# PCM DECODERS  (private)
# =============================================================================

def _decode_pcm(raw: bytes, info: WavInfo) -> np.ndarray:
    """Decode raw WAV bytes to normalised float32 in [-1.0, 1.0]."""
    fmt = info.audio_format
    bps = info.bits_per_sample

    if fmt == 1:   # Integer PCM
        if bps == 8:
            # 8-bit WAV is unsigned
            a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            return (a - 128.0) / 128.0
        if bps == 16:
            return (
                np.frombuffer(raw, dtype=np.int16)
                .astype(np.float32) / 32_768.0
            )
        if bps == 24:
            return _decode_24bit(raw)
        if bps == 32:
            return (
                np.frombuffer(raw, dtype=np.int32)
                .astype(np.float32) / 2_147_483_648.0
            )

    if fmt == 3:   # IEEE-float PCM
        if bps == 32:
            return np.frombuffer(raw, dtype=np.float32).copy()
        if bps == 64:
            return np.frombuffer(raw, dtype=np.float64).astype(np.float32)

    raise ValueError(
        f"_decode_pcm: unhandled format={fmt} bits={bps}"
    )


def _decode_24bit(raw: bytes) -> np.ndarray:
    """
    Vectorised 24-bit little-endian PCM → normalised float32.

    Each sample: [B0_low, B1_mid, B2_high] in two's-complement.
    Sign-extension: values ≥ 0x800000 are negative.
    """
    n = len(raw) // 3
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    arr = np.frombuffer(raw[: n * 3], dtype=np.uint8).reshape(n, 3)
    i32 = (
        arr[:, 0].astype(np.int32)
        | (arr[:, 1].astype(np.int32) << 8)
        | (arr[:, 2].astype(np.int32) << 16)
    )
    # Sign-extend from 24-bit two's complement
    i32[i32 >= 0x80_0000] -= 0x100_0000
    return (i32 / 8_388_608.0).astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    """RMS level in dBFS. Returns -120.0 for silent/near-silent audio."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-10:
        return -120.0
    return round(20.0 * math.log10(rms), 1)


def resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """
    Anti-aliased polyphase resampler (scipy resample_poly, Kaiser LPF).
    No-op when src_sr == tgt_sr or audio is empty.

    Replaced linear interpolation 2026-09-28 (family-wide): np.interp
    has no anti-alias filter, so a 48 kHz -> 16 kHz downsample folds
    everything above the target Nyquist into the output as aliasing
    noise — every downsampled WAV the family shipped before the fix
    carried it. Output length stays pinned to
    int(n_in * tgt_sr / src_sr) (the previous convention).
    """
    if src_sr == tgt_sr or len(audio) == 0:
        return audio
    from fractions import Fraction

    from scipy.signal import resample_poly
    frac = Fraction(tgt_sr, src_sr).limit_denominator(10_000)
    out = resample_poly(
        audio.astype(np.float32, copy=False),
        frac.numerator, frac.denominator,
    ).astype(np.float32, copy=False)
    n_out = max(1, int(len(audio) * tgt_sr / src_sr))
    if len(out) < n_out:
        out = np.pad(out, (0, n_out - len(out)))
    return out[:n_out]


# =============================================================================
# WAV HEADER BUILDER
# =============================================================================

def encode_wav_header_mono(
    n_samples:       int,
    sample_rate:     int,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Build a 44-byte RIFF/WAVE header for mono PCM audio.

    Parameters
    ──────────
    n_samples       : total sample count in the audio
    sample_rate     : output sample rate in Hz
    bits_per_sample : 16 (PCM int16) or 32 (IEEE float32)

    Returns
    ───────
    44 bytes — ready to prepend to raw PCM data.
    """
    if bits_per_sample == 16:
        audio_format = 1      # PCM integer
    elif bits_per_sample == 32:
        audio_format = 3      # IEEE float
    else:
        raise ValueError(
            f"Unsupported bits_per_sample={bits_per_sample}; use 16 or 32")

    bytes_per_sample = bits_per_sample // 8
    data_size   = n_samples * bytes_per_sample
    byte_rate   = sample_rate * bytes_per_sample
    block_align = bytes_per_sample

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",  36 + data_size,
        b"WAVE",
        b"fmt ",  16,
        audio_format, 1,               # mono
        sample_rate,  byte_rate,
        block_align,  bits_per_sample,
        b"data",  data_size,
    )


# =============================================================================
# WAV ENCODER (in-memory, for small files / utility use)
# =============================================================================

def encode_wav_mono(
    audio:           np.ndarray,
    sample_rate:     int,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Encode a float32 mono audio array as a complete PCM WAV file in memory.

    For large files prefer streaming to S3 via multipart upload using
    encode_wav_header_mono() + chunked int16 conversion to avoid
    materialising the full encoded WAV in RAM.

    Parameters
    ──────────
    audio           : 1-D float32 numpy array, values in [-1.0, 1.0]
    sample_rate     : output sample rate in Hz
    bits_per_sample : 16 (default, PCM int16) or 32 (IEEE float32)

    Returns
    ───────
    Complete WAV file contents as bytes (44-byte RIFF header + PCM data).
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {audio.shape}")

    header = encode_wav_header_mono(len(audio), sample_rate, bits_per_sample)

    if bits_per_sample == 16:
        pcm = np.clip(audio * 32_767.0, -32_768.0, 32_767.0).astype(np.int16)
        data_bytes = pcm.tobytes()
        del pcm
    elif bits_per_sample == 32:
        data_bytes = np.ascontiguousarray(
            audio, dtype=np.float32).tobytes()
    else:
        raise ValueError(
            f"Unsupported bits_per_sample={bits_per_sample}; use 16 or 32")

    return header + data_bytes
