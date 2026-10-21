"""
ffmpeg-driven merge helpers for StateMergeChunks.
"""
from __future__ import annotations

import logging
import os

from lib.proc_utils import run as _run
from lib.s3_utils import presign, put_json, upload_file

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

FFMPEG  = os.environ.get("FFMPEG_BIN",  "ffmpeg")
THREADS = int(os.environ.get("FFMPEG_THREADS", "0"))


def _write_file_list(urls: list[str], path: str) -> None:
    with open(path, "w") as f:
        for url in urls:
            f.write(f"file '{url}'\n")


def _presign_keys(bucket: str, keys: list[str]) -> list[str]:
    return [presign(bucket, k) for k in keys]


# ── MP4 proxy merge ────────────────────────────────────────────────────────────

def merge_mp4(
    out_bucket: str,
    video_keys: list[str],
    flac_keys: list[str],
    tmp_dir: str,
    out_key: str,
) -> None:
    """
    Single ffmpeg invocation: concat .ts (codec copy) + concat .flac (AAC re-encode)
    → final MP4. No intermediate files — /tmp only holds the output MP4.
    If flac_keys is empty, video-only MP4 is produced.
    """
    vlist = os.path.join(tmp_dir, "videolist.txt")
    _write_file_list(_presign_keys(out_bucket, video_keys), vlist)

    out_local = os.path.join(tmp_dir, os.path.basename(out_key))

    cmd = [FFMPEG, "-threads", str(THREADS), "-v", "error", "-y"]
    # video chunk list
    cmd += ["-protocol_whitelist", "file,http,https,tcp,tls,crypto",
           "-f", "concat", "-safe", "0", "-i", vlist]
    # audio chunk list
    if flac_keys:
        alist = os.path.join(tmp_dir, "audiolist.txt")
        _write_file_list(_presign_keys(out_bucket, flac_keys), alist)
        cmd += ["-protocol_whitelist", "file,http,https,tcp,tls,crypto",
                "-f", "concat", "-safe", "0", "-i", alist]
        cmd += ["-map", "0:v", "-c:v", "copy",
                "-map", "1:a", "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-ar", "48000"]
    else:
        cmd += ["-c:v", "copy"]

    cmd += ["-movflags", "+faststart", out_local]

    _run(cmd)
    upload_file(out_bucket, out_key, out_local)
    os.unlink(out_local)
    log.info("Merged MP4: s3://%s/%s", out_bucket, out_key)


# ── Audio group merge (WAV / M4A) ──────────────────────────────────────────────

def merge_audio_group(
    out_bucket: str,
    flac_keys: list[str],
    tmp_dir: str,
    out_key: str,
    group: dict,
) -> None:
    """
    Concat .flac segments → decode/re-encode → WAV or M4A.
    Codec, channels, sampling_rate, and (for M4A) bitrate come from group dict.
    Upload to out_bucket/out_key. Cleans up local file.
    """
    fmt      = group["format"]          # "wav" or "m4a"
    codec    = group.get("codec") or ("pcm_s16le" if fmt == "wav" else "aac")
    channels = str(group.get("audio_channels", 1 if fmt == "wav" else 2))
    sr       = str(group.get("sampling_rate", 48000))

    alist = os.path.join(tmp_dir, f"flaclist_{group['group_name']}.txt")
    _write_file_list(_presign_keys(out_bucket, flac_keys), alist)

    out_local = os.path.join(tmp_dir, os.path.basename(out_key))
    cmd = [FFMPEG, "-threads", str(THREADS), "-v", "error", "-y",
           "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
           "-f", "concat", "-safe", "0", "-i", alist,
           "-c:a", codec, "-ac", channels, "-ar", sr]
    if fmt == "m4a":
        cmd += ["-b:a", f"{group.get('audio_bitrate', 96)}k", "-movflags", "+faststart"]
    cmd.append(out_local)

    _run(cmd)
    upload_file(out_bucket, out_key, out_local)
    os.unlink(out_local)
    log.info("Merged %s: s3://%s/%s", fmt.upper(), out_bucket, out_key)


# ── JPEG frames manifest ───────────────────────────────────────────────────────

def write_frames_manifest(
    out_bucket: str,
    manifest_key: str,
    frame_descs: list[dict],
    frames_prefix: str,
    frame_format: str = "jpeg",
    dimension: list[int] | None = None,
) -> None:
    """
    Reconstruct frame key list from compact {start, step, total} descriptors
    (sorted by item_id — caller ensures order) and write manifest JSON to S3.

    frames_prefix: S3 prefix where frames live, e.g. "{out_prefix}/frames".
    """
    ext  = "png" if frame_format == "png" else "jpg"
    keys = []
    for desc in frame_descs:
        start = desc["start"]
        step  = desc["step"]
        total = desc["total"]
        for i in range(total):
            keys.append(f"frame_{start + i * step:07d}.{ext}")

    manifest: dict = {
        "total":  len(keys),
        "prefix": frames_prefix,
        "keys":   keys,
    }
    if dimension:
        manifest["dimension"] = dimension

    if not keys:
        raise RuntimeError(
            "write_frames_manifest: no frame keys reconstructed — "
            "all frame descriptors have total=0"
        )
    put_json(out_bucket, manifest_key, manifest)
    log.info("Frames manifest: %d keys → s3://%s/%s",
             len(keys), out_bucket, manifest_key)