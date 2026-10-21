# lib/ffmpeg_utils.py

import logging
import subprocess
import json

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from lib.s3_utils import presign, put_json

# ── Constants ─────────────────────────────────────────────────────────────────
FFPROBE_TIMEOUT = 840          # 14 min — large MXF files may seek to footer
SEEK_TEST_TIMEOUT = 300        # 5 min — mimics IPB transcode path, fast
SEEK_TEST_POSITION = 10.0      # seconds — seek target for capability test

# =============================================================================
# Seek capability test
# =============================================================================

def test_seek_capability(
    bucket: str,
    key:    str,
) -> tuple[bool, float | None]:
    """
    Test whether the file supports seeking by mimicking the IPB transcode path.

    Command:
        ffmpeg -copyts -ss {SEEK_TEST_POSITION} -t 1
               -i {url} -vf showinfo -f null /dev/null

    This is identical to what the transcode handler runs for IPB content,
    so it is the most faithful test of whether the transcode will succeed.

    Two signals from stderr:
      1. "could not seek to position" → can_seek=False
         File has no index (e.g. Sony eVTR MXF Class B). ffmpeg falls back
         to linear scan from start — correct for all-intra, wrong for long-GOP.
      2. showinfo "pts_time:{value}" → actual position ffmpeg landed on.
         Logged for diagnostics. Not used for routing.

    Returns (can_seek, first_pts_time).
    """
    url = presign(bucket, key)
    cmd = [
        "ffmpeg", "-y",
        "-copyts",
        "-ss", str(SEEK_TEST_POSITION),
        "-t",  "1",
        "-i",  url,
        "-vf", "showinfo",
        "-f",  "null", "/dev/null",
    ]
    log.info(
        "Seek test: s3://%s/%s  position=%.1fs",
        bucket, key, SEEK_TEST_POSITION,
    )

    try:
        result = subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            cmd, capture_output=True, text=True, timeout=SEEK_TEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "Seek test timed out after %ds — treating as can seek "
            "(timeout implies ffmpeg was actively seeking, not that seek "
            "is unsupported; a non-seekable file fails immediately with "
            "'could not seek to position')",
            SEEK_TEST_TIMEOUT,
        )
        return True, None

    stderr = result.stderr

    # ── Seek capability ───────────────────────────────────────────────────────
    can_seek = "could not seek to position" not in stderr

    # ── pts_time of first decoded frame (diagnostic) ──────────────────────────
    # Parse from showinfo output in stderr.
    # showinfo line: "n:   0 pts:    252 pts_time:10.08   duration: ..."
    first_pts_time: float | None = None
    for line in stderr.splitlines():
        if "Parsed_showinfo" not in line:
            continue
        for part in line.split():
            if part.startswith("pts_time:"):
                try:
                    first_pts_time = float(part.split(":", 1)[1])
                    break
                except (ValueError, IndexError):
                    pass
        if first_pts_time is not None:
            break

    if can_seek:
        log.info(
            "Seek test ✓: file can seek  "
            "first_pts_time=%s  (requested=%.1fs)",
            f"{first_pts_time:.3f}s" if first_pts_time is not None else "N/A",
            SEEK_TEST_POSITION,
        )
    else:
        log.warning(
            "Seek test ✗: file cannot seek  "
            "first_pts_time=%s  (requested=%.1fs)  "
            "ffmpeg will read linearly from start",
            f"{first_pts_time:.3f}s" if first_pts_time is not None else "N/A",
            SEEK_TEST_POSITION,
        )
        # Log the seek-related stderr lines for diagnosis
        for line in stderr.splitlines():
            if "seek" in line.lower() or "estimating" in line.lower():
                log.warning("  ffmpeg: %s", line.strip())

    return can_seek, first_pts_time


# =============================================================================
# ffprobe — header-only stream + format probe
# =============================================================================

def run_ffprobe_streams(bucket: str, key: str) -> dict:
    """
    Extract container format + all stream metadata via ffprobe.
    Reads only the container header — O(1) for most formats.

    MXF note: some non-compliant MXF files (e.g. Sony eVTR Class B) store
    the index at the end of the file. ffprobe must seek to the footer via
    an HTTP range request, which can take 60-120s for 50GB+ files over S3.
    FFPROBE_TIMEOUT is set to 840s (14 min) to cover this case.
    """
    url = presign(bucket, key)
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        url,
    ]
    log.info("ffprobe: s3://%s/%s", bucket, key)
    result = subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (rc={result.returncode}): {result.stderr[:500]}"
        )
    return json.loads(result.stdout)

# =============================================================================
# Timecode extraction
# =============================================================================

def _parse_timecode(tc_str: str | None) -> tuple[str | None, bool]:
    if not tc_str:
        return None, False
    drop_frame = ";" in tc_str
    return tc_str.replace(";", ":"), drop_frame


def extract_timecode(
    fmt:         dict,
    streams:     list[dict],
    first_video: dict,
) -> tuple[str | None, bool, str]:
    tc_keys = ("timecode", "Timecode", "TIMECODE")

    for key in tc_keys:
        val = fmt.get("tags", {}).get(key)
        if val:
            tc, df = _parse_timecode(val)
            if tc:
                log.info(
                    "Timecode from container tags: %s (drop_frame=%s)", tc, df)
                return tc, df, "container"

    for key in tc_keys:
        val = first_video.get("tags", {}).get(key)
        if val:
            tc, df = _parse_timecode(val)
            if tc:
                log.info(
                    "Timecode from video stream tags: %s (drop_frame=%s)", tc, df)
                return tc, df, "stream"

    for stream in streams:
        if stream.get("codec_type") == "data":
            for key in tc_keys:
                val = stream.get("tags", {}).get(key)
                if val:
                    tc, df = _parse_timecode(val)
                    if tc:
                        log.info(
                            "Timecode from data stream [%d]: %s (drop_frame=%s)",
                            stream.get("index", -1), tc, df,
                        )
                        return tc, df, "stream"

    log.warning("No timecode found — downstream uses 00:00:00:00")
    return None, False, "calculated"


# =============================================================================
# S3 output
# =============================================================================

def write_probe_to_s3(bucket: str, prefix: str, probe_data: dict) -> str:
    key = f"{prefix}/probe.json"
    put_json(bucket, key, probe_data)
    log.info("probe.json written: s3://%s/%s", bucket, key)
    return key
