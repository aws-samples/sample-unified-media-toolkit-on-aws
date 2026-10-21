# separate_speaker_channels.py
"""
StateSeparateSpeakerChannels — arm64 Docker Lambda.

Produces one WAV file per speaker from the final transcript.

Modes
─────
  aligned  (default) : Full-length files matching source duration.  Non-speech
                        regions are silent.  All speaker files can be played
                        alongside the source in perfect temporal sync.

  compact            : Short files containing only each speaker's speech
                        segments concatenated with configurable silence gaps.
                        A time_map in the manifest maps compact positions back
                        to original timestamps.

Skip conditions
───────────────
  • output_config.speaker_channels is false  → SKIPPED (disabled)
  • data.num_global_speakers == 0            → SKIPPED (no_speakers)
  • transcript contains no sentences         → SKIPPED (no_sentences)

Config  (event.output.output_config — set via input schema / validate_and_normalize)
─────────────────────────────────────────────────────────────────────────────────────
  speaker_channels   true        Master switch.  When false the handler
                                 returns immediately with SKIPPED status.
  channel_mode       "aligned"   "aligned" | "compact"

  Hidden configurations:
  merge_gap_sec      0.5         Adjacent same-speaker sentence regions
                                 within this gap are merged into one
                                 contiguous audio segment.  Preserves
                                 natural speech continuity (breaths,
                                 micro-pauses between sentences).
  fade_ms            10          Linear fade-in / fade-out applied at
                                 each segment boundary to prevent clicks.
                                 Clamped to half the segment length for
                                 very short segments.
  compact_gap_sec    0.1         Silence inserted between non-adjacent
                                 segments in compact mode.

S3 output
─────────
  {output_prefix}/speakers/SPEAKER_00.wav
  {output_prefix}/speakers/SPEAKER_01.wav
  ...
  {output_prefix}/speakers/speaker_manifest.json

S3 upload
─────────
  Small files (< 6 MB)  : single put_object
  Large files (≥ 6 MB)  : multipart upload — float32 → int16 conversion is
                           streamed in ~8 MB chunks.  Only one chunk + its
                           int16 conversion are in memory at any time.
                           The full encoded WAV is never materialised.

  On multipart failure, the upload is aborted (no orphaned parts).

Memory budget  (2 Cores — 3538 MB)
───────────────────────────────────
  Source float32 in memory for the entire run.
  Per speaker: one output float32 array (aligned=full length, compact=speech
  only).  Streamed to S3 in chunks; no full WAV bytes materialised.

  Peak ≈ source_f32 + output_f32 + encode_chunk
       = N×4 + N×4 + C×8      (C = 4M samples ≈ 32 MB, negligible)
       ≈ N×8

  1 h @ 16 kHz → N = 57.6 M  →  461 MB peak   ✓
  1 h @ 48 kHz → N = 172.8 M → 1.38 GB peak   ✓
  2 h @ 48 kHz → N = 345.6 M → 2.76 GB peak   ✓
  3 h @ 48 kHz → N = 518.4 M → 4.15 GB peak   ✗ (use 3 Cores)

Lambda configuration
─────────────────────
  FunctionName : SpeechDiarizationLambda2Cores
  Timeout      : 900 s
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any
import numpy as np

from lib.input_schema import (
    DEFAULT_SPEAKER_MERGE_GAP_SEC,
    DEFAULT_SPEAKER_FADE_MS,
    DEFAULT_COMPACT_GAP_SEC,
)
from lib.wav_utils import encode_wav_header_mono, get_wav_info, read_wav_window
from lib.s3_utils import s3 as _s3, get_json, put_json

log = logging.getLogger(__name__)
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ── Streaming upload ─────────────────────────────────────────────────────────
_UPLOAD_CHUNK_SAMPLES = 4 * 1024 * 1024   # 4M samples → 8 MB @ 16-bit
_MULTIPART_THRESHOLD_BYTES = 6 * 1024 * 1024   # below this → single put_object


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING WAV UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _encode_chunk_int16(chunk: np.ndarray) -> bytes:
    """Convert a float32 audio chunk to 16-bit PCM bytes."""
    pcm = np.clip(chunk * 32_767.0, -32_768.0, 32_767.0).astype(np.int16)
    raw = pcm.tobytes()
    del pcm
    return raw


def _stream_upload_wav(
    bucket:      str,
    key:         str,
    audio:       np.ndarray,
    sample_rate: int,
) -> int:
    """
    Encode float32 mono audio as 16-bit PCM WAV and upload to S3.

    Small files (< 6 MB) use a single put_object.
    Large files use multipart upload — the float32 → int16 conversion is
    performed in ~8 MB chunks so the full encoded WAV is never held in RAM.

    Returns total bytes written (44-byte header + PCM data).
    """
    s3c = _s3
    n_samples = len(audio)
    data_size = n_samples * 2           # int16 = 2 bytes per sample
    total_bytes = 44 + data_size
    header = encode_wav_header_mono(n_samples, sample_rate)

    # ── Small file: single request ────────────────────────────────────────
    if total_bytes < _MULTIPART_THRESHOLD_BYTES:
        body = header + _encode_chunk_int16(audio)
        s3c.put_object(Bucket=bucket, Key=key, Body=body,
                       ContentType="audio/wav")
        return total_bytes

    # ── Large file: multipart upload ──────────────────────────────────────
    mpu = s3c.create_multipart_upload(
        Bucket=bucket, Key=key, ContentType="audio/wav")
    upload_id = mpu["UploadId"]

    try:
        parts:    list[dict] = []
        part_num: int = 1
        offset:   int = 0

        while offset < n_samples:
            end = min(offset + _UPLOAD_CHUNK_SAMPLES, n_samples)
            body = _encode_chunk_int16(audio[offset:end])

            # Prepend WAV header to the first part
            if offset == 0:
                body = header + body

            resp = s3c.upload_part(
                Bucket=bucket, Key=key, UploadId=upload_id,
                PartNumber=part_num, Body=body)
            parts.append({"ETag": resp["ETag"], "PartNumber": part_num})
            del body

            part_num += 1
            offset = end

        s3c.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": parts})

    except Exception:
        s3c.abort_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id)
        raise

    return total_bytes


# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER GROUPING
# ─────────────────────────────────────────────────────────────────────────────

def _group_by_speaker(
    sentences: list[dict],
) -> dict[str, list[dict]]:
    """Group sentences by speaker_id.  None / empty → "UNKNOWN"."""
    groups: dict[str, list[dict]] = {}
    for sent in sentences:
        spk = sent.get("speaker_id") or "UNKNOWN"
        groups.setdefault(spk, []).append(sent)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# REGION MERGING
# ─────────────────────────────────────────────────────────────────────────────

def _merge_regions(
    sentences:     list[dict],
    sample_rate:   int,
    merge_gap_sec: float,
) -> list[tuple[int, int]]:
    """
    Convert sentence timestamps to sample ranges and merge adjacent or
    overlapping ranges that are within *merge_gap_sec* of each other.

    Returns a sorted list of (start_sample, end_sample) tuples.
    """
    if not sentences:
        return []

    sorted_sents = sorted(sentences, key=lambda s: float(s["start"]))
    merge_gap_samples = int(merge_gap_sec * sample_rate)

    ranges: list[list[int]] = []
    for sent in sorted_sents:
        s = int(float(sent["start"]) * sample_rate)
        e = int(float(sent["end"]) * sample_rate)
        if e > s:
            ranges.append([s, e])

    if not ranges:
        return []

    merged: list[list[int]] = [ranges[0]]
    for s, e in ranges[1:]:
        prev = merged[-1]
        if s <= prev[1] + merge_gap_samples:
            prev[1] = max(prev[1], e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


# ─────────────────────────────────────────────────────────────────────────────
# FADE
# ─────────────────────────────────────────────────────────────────────────────

def _apply_fade(seg: np.ndarray, fade_samples: int) -> None:
    """Apply linear fade-in / fade-out in place.  Clamps to half segment."""
    if fade_samples <= 0 or len(seg) == 0:
        return
    fs = min(fade_samples, len(seg) // 2)
    if fs > 0:
        seg[:fs] *= np.linspace(0.0, 1.0, fs, dtype=np.float32)
        seg[-fs:] *= np.linspace(1.0, 0.0, fs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODES
# ─────────────────────────────────────────────────────────────────────────────

def _build_aligned(
    source:        np.ndarray,
    regions:       list[tuple[int, int]],
    total_samples: int,
    fade_samples:  int,
) -> np.ndarray:
    """
    Full-length output: zeros everywhere except speaker's regions.
    """
    output = np.zeros(total_samples, dtype=np.float32)
    n = len(source)
    for start, end in regions:
        s = max(0, min(start, n))
        e = max(s, min(end, n))
        if e <= s:
            continue
        seg = source[s:e].copy()
        _apply_fade(seg, fade_samples)
        output[s:e] = seg
    return output


def _build_compact(
    source:       np.ndarray,
    regions:      list[tuple[int, int]],
    sample_rate:  int,
    fade_samples: int,
    gap_samples:  int,
) -> tuple[np.ndarray, list[dict]]:
    """
    Compact output: only speaker's speech, concatenated with short gaps.
    Returns (audio_array, time_map).
    """
    segments: list[np.ndarray] = []
    time_map: list[dict] = []
    compact_pos = 0
    n = len(source)

    for i, (start, end) in enumerate(regions):
        s = max(0, min(start, n))
        e = max(s, min(end, n))
        if e <= s:
            continue

        seg = source[s:e].copy()
        _apply_fade(seg, fade_samples)

        # Insert silence gap between non-adjacent segments
        if segments and gap_samples > 0:
            segments.append(np.zeros(gap_samples, dtype=np.float32))
            compact_pos += gap_samples

        time_map.append({
            "original_start": round(s / sample_rate, 3),
            "original_end":   round(e / sample_rate, 3),
            "compact_start":  round(compact_pos / sample_rate, 3),
            "compact_end":    round((compact_pos + len(seg)) / sample_rate, 3),
        })

        segments.append(seg)
        compact_pos += len(seg)

    if segments:
        output = np.concatenate(segments)
    else:
        output = np.zeros(0, dtype=np.float32)

    return output, time_map


# ─────────────────────────────────────────────────────────────────────────────
# SKIP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _skip(event: dict, data: dict, reason: str) -> dict:
    return {
        **event,
        "data": {
            **data,
            "speaker_channels_status":      "SKIPPED",
            "speaker_channels_skip_reason": reason,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    log.info("StateSeparateSpeakerChannels event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── unpack event ──
        data          = event["data"]
        audio_bucket  = data["audio_bucket"]
        audio_key     = data["audio_key"]
        output_bucket = data["output_bucket"]
        output_prefix = data["output_prefix"]

        num_global_speakers = int(
            data.get("num_global_speakers", 0))
        speaker_channels = bool(
            data["speaker_channels"])
        channel_mode = str(
            data["channel_mode"])

        # Hidden output config settings
        output = event["output"]
        output_config = output.get("output_config", {})

        merge_gap_sec = float(
            output_config.get("merge_gap_sec", DEFAULT_SPEAKER_MERGE_GAP_SEC))
        fade_ms = float(
            output_config.get("fade_ms", DEFAULT_SPEAKER_FADE_MS))
        compact_gap_sec = float(
            output_config.get("compact_gap_sec", DEFAULT_COMPACT_GAP_SEC))

        log.info(
            "StateSeparateSpeakerChannels START | "
            "speaker_channels=%s  channel_mode=%s  num_global_speakers=%d",
            speaker_channels, channel_mode, num_global_speakers,
        )

        # ── Skip conditions ───────────────────────────────────────────────
        if not speaker_channels:
            log.info("Speaker channel separation disabled — skipping")
            return _skip(event, data, "disabled")

        if num_global_speakers == 0:
            log.info("No speakers detected — skipping")
            return _skip(event, data, "no_speakers")

        if channel_mode not in ("aligned", "compact"):
            raise ValueError(
                f"Invalid speaker_channels.mode: {channel_mode!r} "
                f"— expected 'aligned' or 'compact'")

        # ── Load transcript ───────────────────────────────────────────────
        transcript_key = data.get("final_transcript_key")
        if not transcript_key:
            raise RuntimeError(
                "final_transcript_key not found in data — "
                "transcription may have failed upstream")

        log.info("Loading transcript  s3://%s/%s",
                 output_bucket, transcript_key)
        doc = get_json(output_bucket, transcript_key)
        sentences = doc.get("sentences", [])
        del doc  # free JSON; we only need sentences

        if not sentences:
            log.warning("No sentences in transcript — skipping")
            return _skip(event, data, "no_sentences")

        # ── Group by speaker ──────────────────────────────────────────────
        speaker_groups = _group_by_speaker(sentences)
        del sentences  # grouped; original list not needed

        log.info(
            "Speakers: %s",
            {k: len(v) for k, v in sorted(speaker_groups.items())},
        )

        # ── Read source audio at native sample rate ──────────────────────
        log.info("Reading source audio  s3://%s/%s", audio_bucket, audio_key)
        wav_info = get_wav_info(audio_bucket, audio_key)
        native_sr = wav_info.sample_rate
        full_audio = read_wav_window(
            audio_bucket, audio_key, wav_info,
            start_sec=0.0,
            duration_sec=wav_info.duration_sec,
            target_sr=None,                       # native sample rate
        )
        total_samples = len(full_audio)

        log.info(
            "Audio loaded: %d samples  %d Hz  %.1fs  (%.0f MB float32)",
            total_samples, native_sr, wav_info.duration_sec,
            total_samples * 4 / 1_048_576,
        )

        # ── Output prefix ─────────────────────────────────────────────────
        speaker_dir = f"{output_prefix}/speakers"

        # ── Process each speaker ──────────────────────────────────────────
        fade_samples = max(1, int(native_sr * fade_ms / 1000))
        manifest_speakers: dict[str, dict] = {}
        total_wav_bytes = 0

        for speaker_id in sorted(speaker_groups.keys()):
            sents = speaker_groups[speaker_id]
            regions = _merge_regions(sents, native_sr, merge_gap_sec)

            if not regions:
                log.warning(
                    "  %s: no valid audio regions — skipping", speaker_id)
                continue

            speech_sec = sum(
                (e - s) / native_sr for s, e in regions)
            log.info(
                "  %s: %d sentences → %d regions  speech=%.1fs",
                speaker_id, len(sents), len(regions), speech_sec,
            )

            # ── Build speaker audio ──────────────────────────────────────
            if channel_mode == "aligned":
                speaker_audio = _build_aligned(
                    full_audio, regions, total_samples, fade_samples)
                time_map = None
            else:
                gap_samples = int(native_sr * compact_gap_sec)
                speaker_audio, time_map = _build_compact(
                    full_audio, regions, native_sr,
                    fade_samples, gap_samples)

            # ── Stream encode + upload ────────────────────────────────────
            out_key = f"{speaker_dir}/{speaker_id}.wav"
            n_bytes = _stream_upload_wav(
                output_bucket, out_key, speaker_audio, native_sr)
            del speaker_audio
            total_wav_bytes += n_bytes

            log.info(
                "    Written s3://%s/%s  (%.1f MB)",
                output_bucket, out_key, n_bytes / 1_048_576,
            )

            # ── Manifest entry ────────────────────────────────────────────
            spk_info: dict[str, Any] = {
                "wav_key":          out_key,
                "num_sentences":    len(sents),
                "num_segments":     len(regions),
                "total_speech_sec": round(speech_sec, 3),
            }
            if channel_mode == "compact" and time_map is not None:
                compact_dur = (
                    speech_sec
                    + max(0, len(regions) - 1) * compact_gap_sec
                )
                spk_info["duration_sec"] = round(compact_dur, 3)
                spk_info["time_map"] = time_map

            manifest_speakers[speaker_id] = spk_info

        del full_audio  # free source audio

        # ── Write manifest ────────────────────────────────────────────────
        manifest: dict[str, Any] = {
            "channel_mode":        channel_mode,
            "source_audio_key":    audio_key,
            "sample_rate":         native_sr,
            "source_duration_sec": round(wav_info.duration_sec, 3),
            "num_speakers":        len(manifest_speakers),
            "merge_gap_sec":       merge_gap_sec,
            "fade_ms":             fade_ms,
            "speakers":            manifest_speakers,
        }
        if channel_mode == "compact":
            manifest["compact_gap_sec"] = compact_gap_sec

        manifest_key = f"{speaker_dir}/speaker_manifest.json"
        put_json(output_bucket, manifest_key, manifest)
        log.info("Written manifest  s3://%s/%s", output_bucket, manifest_key)

        # ── Return ────────────────────────────────────────────────────────
        elapsed = round(time.monotonic() - t0, 2)
        log.info(
            "DONE  elapsed=%.2fs  speakers=%d  output_bucket=%s  "
            "total_wav=%.1f MB",
            elapsed, len(manifest_speakers), output_bucket,
            total_wav_bytes / 1_048_576,
        )

        return {
            **event,
            "data": {
                **data,
                "speaker_manifest_key":         manifest_key,
                "speaker_channels_status":      "COMPLETED",
                "speaker_channels_mode":        channel_mode,
                "speaker_channels_num_files":   len(manifest_speakers),
                "speaker_channels_elapsed_sec": elapsed,
                "iterators":                    None,
            },
        }

    except Exception as exc:
        log.error(
            "[ERR] StateSeparateSpeakerChannels — %s: %s",
            type(exc).__name__, exc,
        )
        traceback.print_exc()
        raise
