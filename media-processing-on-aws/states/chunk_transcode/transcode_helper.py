"""
ffmpeg command building, validation, and transcode entry point for StateChunkTranscode.
"""
from __future__ import annotations

import glob
import json
import logging
import math
import os
import shlex
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

from lib.ffprobe import _safe_float, find_prior_keyframe
from lib.proc_utils import run as _run
from lib.s3_utils import presign, upload_file
from lib.video_command import SNAP_DENOMINATOR, sec_to_frame

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

FFMPEG  = os.environ.get("FFMPEG_BIN",  "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")
THREADS = int(os.environ.get("FFMPEG_THREADS", "0"))

JPEG_UPLOAD_WORKERS = 50

_FRAME_CONTENT_TYPE = {
    "jpeg": "image/jpeg",
    "png":  "image/png",
}

class ValidationException(Exception):
    """Exact class name required — Step Functions Retry/Catch matches on error name."""
    pass


def upload_frames(
    out_bucket: str,
    out_prefix: str,
    s3_prefix: str,
    tmp_dir: str,
    local_pattern: str,
    nominal_start_frame: int,
    frame_interval: int,
    frame_format: str = "jpeg",
) -> list[str]:
    """
    Upload all frame image files produced by ffmpeg.
    S3 key: {out_prefix}/{s3_prefix}/frame_{global_frame:07d}.{ext}
    Global frame = nominal_start_frame + local_idx * frame_interval.
    Returns ordered list of uploaded S3 keys.
    """
    content_type = _FRAME_CONTENT_TYPE.get(frame_format, "image/jpeg")
    ext          = "png" if frame_format == "png" else "jpg"
    glob_pat     = os.path.join(tmp_dir, local_pattern.replace("%07d", "*"))
    local_files  = sorted(glob.glob(glob_pat))

    if not local_files:
        raise ValidationException(
            f"upload_frames: no frame files found matching pattern {local_pattern!r} in {tmp_dir} — "
            "ffmpeg may have produced no output for this chunk"
        )

    def _upload_one(local_idx: int, local_file: str) -> str:
        global_frame = nominal_start_frame + local_idx * frame_interval
        s3_key       = f"{out_prefix}/{s3_prefix}/frame_{global_frame:07d}.{ext}"
        upload_file(out_bucket, s3_key, local_file, extra_args={"ContentType": content_type})
        return s3_key

    keys: list[str] = [""] * len(local_files)
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=JPEG_UPLOAD_WORKERS) as pool:
        futures = {pool.submit(_upload_one, i, f): i for i, f in enumerate(local_files)}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                errors.append(exc)
            else:
                keys[futures[future]] = future.result()

    if errors:
        raise RuntimeError(
            f"upload_frames: {len(errors)} upload(s) failed; first error: {errors[0]}"
        )
    log.info("Uploaded %d %s frames", len(keys), frame_format.upper())
    return keys


def _substitute(filters: list[str], tokens: dict[str, str]) -> list[str]:
    """Replace all {token} placeholders in each filter string."""
    result = []
    for f in filters:
        for tok, val in tokens.items():
            f = f.replace(tok, val)
        result.append(f)
    return result


def _ffprobe_json(path: str, select_streams: str, show_entries: str) -> dict:
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", select_streams,
        "-show_entries", show_entries,
        "-of", "json",
        path,
    ]
    return json.loads(_run(cmd, timeout=120))


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_frames(path: str, expected: int) -> int:
    """±1 frame tolerance. Raises ValidationException on mismatch."""
    data    = json.loads(_run([
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json",
        path,
    ], timeout=300))
    streams = data.get("streams") or []
    nb      = int((streams[0] if streams else {}).get("nb_read_frames", -1))

    if abs(expected - nb) <= 1:
        return nb
    msg = f"Expect {expected} frames. Received {nb} frames."
    log.error("ValidationException: %s", msg)
    raise ValidationException(msg)


def validate_duration(path: str, expected_t: str) -> float:
    """±10 ms tolerance. Raises ValidationException on mismatch."""
    data    = _ffprobe_json(path, "a:0", "stream=index,codec_type,duration")
    streams = data.get("streams") or []
    actual  = _safe_float((streams[0] if streams else {}).get("duration"), -1.0)

    if actual < 0:
        fmt_data = json.loads(_run([
            FFPROBE, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            path,
        ], timeout=120))
        actual = _safe_float((fmt_data.get("format") or {}).get("duration"), -1.0)

    if actual < 0:
        log.warning("validate_duration: could not determine duration for %s — skipping", path)
        return 0.0

    expected_ms = round(float(expected_t) * 1000)
    actual_ms   = round(actual * 1000)

    if abs(expected_ms - actual_ms) <= 10:
        return actual
    msg = f"Expect {expected_ms} ms duration. Received {actual_ms} ms."
    log.error("ValidationException: %s", msg)
    raise ValidationException(msg)


def validate_jpeg_count(tmp_dir: str, pattern: str, expected: int) -> int:
    """±1 frame tolerance. Raises ValidationException on mismatch."""
    glob_pat = os.path.join(tmp_dir, pattern.replace("%07d", "*"))
    found    = len(glob.glob(glob_pat))
    if found == 0:
        raise ValidationException(
            f"validate_jpeg_count: 0 frame files found matching {pattern!r} — "
            "ffmpeg produced no frame output for this chunk"
        )
    if abs(expected - found) <= 1:
        return found
    msg = f"Expect {expected} JPEG frames. Found {found}."
    log.error("ValidationException: %s", msg)
    raise ValidationException(msg)


# ── Command builder ────────────────────────────────────────────────────────────

def build_ff_cmd(
    url: str,
    seek_ss: float | None,
    filter_complex_str: str,
    chunk_outputs: list[dict],
    tmp_dir: str,
) -> list:
    """
    Build ffmpeg command for multi-output transcode.

    Each chunk_output contributes one output file:
      -map <label> <codec/format flags> <local_path>

    seek_ss, when set, is emitted as a single pre-input -ss (input seeking).

    Frame accuracy does NOT rest on the seek. seek_ss is always a keyframe
    timestamp (or nominal_start_sec on an all-intra source, where every frame is
    one), and the exact frame is then picked by COUNTING with
    trim=start_frame=preroll_frames. Because the seek target is itself a
    keyframe, accurate and fast seek land on the same frame — verified
    frame-exact under both `-ss` and `-noaccurate_seek` on a long-GOP
    1/600-timescale source. Never make the frame selection depend on seek
    precision instead of the trim count.
    filter_complex_str must have all template variables already substituted.
    """
    args = [FFMPEG, "-threads", str(THREADS), "-v", "error", "-y"]

    if seek_ss is not None:
        args += ["-ss", f"{seek_ss:.6f}"]

    args += ["-i", url]

    if filter_complex_str:
        args += ["-filter_complex", filter_complex_str]

    for output in chunk_outputs:
        local_path = os.path.join(tmp_dir, output.get("local_pattern") or output["s3_name"])
        # map field: "[label] -flag val ..." — prepend -map, append local_path
        args += ["-map"] + shlex.split(output["map"]) + [local_path]

    return args


# ── Entry point ────────────────────────────────────────────────────────────────

def transcode(iterator: dict, tmp_dir: str) -> list[dict]:
    """
    Transcode one chunk iterator. Returns chunk_outputs with local_path added.
    Caller is responsible for uploading results and cleaning up tmp_dir.
    """
    inp            = iterator["input"]
    is_all_intra   = iterator["is_all_intra"]
    sampling_rate  = iterator["sampling_rate"]
    chunk_outputs  = iterator["chunk_outputs"]
    fc             = iterator["filter_complex"]
    vfilters       = fc.get("vfilters") or []
    afilters       = fc.get("afilters") or []
    segment_frames = iterator.get("nominal_end_frame", 0) - iterator.get("nominal_start_frame", 0)
    segment_dur    = iterator.get("nominal_end_sec", 0.0) - iterator.get("nominal_start_sec", 0.0)

    # ── Seek position and token values ────────────────────────────────────────
    seek_ss = None
    tokens  = {}

    if vfilters:
        nominal_start_sec   = iterator["nominal_start_sec"]
        nominal_start_frame = iterator["nominal_start_frame"]
        fps_frac            = Fraction(iterator["fps"])
        preroll_frames      = 0

        if nominal_start_sec > 0:
            # Assume all-intra: every frame is a keyframe — direct seek is exact, no preroll needed.
            seek_ss = nominal_start_sec
            if not is_all_intra:
                kf = find_prior_keyframe(inp["bucket"], inp["key"], nominal_start_sec)
                kf_ts = _safe_float(kf.get("best_effort_timestamp_time"), None)
                if kf_ts is None:
                    raise ValidationException(
                        "find_prior_keyframe returned non-numeric best_effort_timestamp_time="
                        f"{kf.get('best_effort_timestamp_time')!r} — cannot compute keyframe position"
                    )
                seek_ss        = kf_ts
                # Same arithmetic that produced nominal_start_frame upstream, so the
                # two frame numbers are on one scale and their difference is meaningful.
                # round, not floor: a keyframe timestamp IS a frame boundary, and
                # ffprobe prints it to 6 dp — 28753 arrives as 1199.239708, which
                # floors to 28752 and buys one frame of phantom preroll.
                keyframe_frame = round(sec_to_frame(kf_ts, fps_frac))
                preroll_frames = nominal_start_frame - keyframe_frame
                if preroll_frames < 0:
                    raise ValidationException(
                        f"preroll_frames={preroll_frames} < 0: keyframe_frame={keyframe_frame} "
                        f"exceeds nominal_start_frame={nominal_start_frame} — "
                        "find_prior_keyframe returned a frame past the segment start"
                    )
                log.info(
                    "Keyframe seek: seek_ss=%.6f  keyframe_frame=%d  preroll_frames=%d",
                    seek_ss, keyframe_frame, preroll_frames,
                )

        tokens["{start_frame}"]  = str(preroll_frames)
        tokens["{end_frame}"]    = str(preroll_frames + segment_frames)
        tokens["{start_sample}"] = str(math.floor(Fraction(preroll_frames) / fps_frac * sampling_rate))
        tokens["{end_sample}"]   = str(math.ceil(Fraction(preroll_frames + segment_frames) / fps_frac * sampling_rate))

    if afilters and not vfilters:
        # Audio-only: seek to near the segment start so ffmpeg doesn't decode from file begin.
        # Pre-roll 3s before the nominal start to land before the first sample we want.
        # atrim sample tokens are relative to the seek point.
        nominal_start_sec = iterator.get("nominal_start_sec", 0.0)
        if nominal_start_sec > 0:
            seek_ss      = max(0.0, nominal_start_sec - 3.0)
            seek_samples = math.floor(Fraction(seek_ss).limit_denominator(SNAP_DENOMINATOR) * sampling_rate)
        else:
            seek_ss      = None
            seek_samples = 0
        tokens["{start_sample}"] = str(iterator["start_sample"] - seek_samples)
        tokens["{end_sample}"]   = str(iterator["end_sample"]   - seek_samples)

    filter_complex_str = ";".join(_substitute(vfilters + afilters, tokens))

    # ── Run ffmpeg ─────────────────────────────────────────────────────────────
    url = presign(inp["bucket"], inp["key"])
    cmd = build_ff_cmd(url, seek_ss, filter_complex_str, chunk_outputs, tmp_dir)
    t0  = time.monotonic()
    _run(cmd)
    log.info("ffmpeg elapsed: %.2fs", time.monotonic() - t0)

    # ── Validate and annotate outputs ─────────────────────────────────────────
    results = []
    for output in chunk_outputs:
        local_path = os.path.join(tmp_dir, output.get("local_pattern") or output["s3_name"])
        if output.get("local_pattern"):
            frame_interval = iterator.get("frame_interval") or 1
            validate_jpeg_count(tmp_dir, output["local_pattern"], math.ceil(segment_frames / frame_interval))
        else:
            if output["s3_name"].endswith(".ts"):
                validate_frames(local_path, segment_frames)
            elif output["s3_name"].endswith(".flac"):
                validate_duration(local_path, str(segment_dur))

        results.append({**output, "local_path": local_path})

    return results