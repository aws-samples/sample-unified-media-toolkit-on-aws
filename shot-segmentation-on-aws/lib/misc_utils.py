# lib/misc_utils.py
import math
from fractions import Fraction

def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def fps_from_str(rate_str: str) -> float:
    if not rate_str or rate_str in ("0/0", "0:0"):
        return 0.0
    try:
        return float(Fraction(rate_str)) if "/" in rate_str else float(rate_str)
    except (ValueError, ZeroDivisionError):
        return 0.0

def fmt_time(seconds: float) -> str:
    """Format float seconds as HH:MM:SS.mmm for ffmpeg -ss / -t."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

# =============================================================================
# SMPTE helpers
# =============================================================================

def frame_to_smpte(frame, fps_str, drop_frame, tc_origin):
    fps_frac = Fraction(fps_str.replace(":", "/"))
    fps_round = round_half_up(float(fps_frac))
    origin = smpte_to_frames(
        tc_origin, fps_str, drop_frame) if tc_origin else 0
    af = frame + origin
    if drop_frame and fps_round in (30, 60):
        return frames_to_df_smpte(af, fps_round)
    h = af // (fps_round * 3600)
    m = (af % (fps_round * 3600)) // (fps_round * 60)
    s = (af % (fps_round * 60)) // fps_round
    f = af % fps_round
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def smpte_to_frames(tc, fps_str, drop_frame):
    parts = tc.replace(";", ":").split(":")
    h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    fps_frac = Fraction(fps_str.replace(":", "/"))
    fps_round = round_half_up(float(fps_frac))
    if drop_frame and fps_round in (30, 60):
        d = 2 if fps_round == 30 else 4
        total_mins = 60 * h + m
        return (fps_round * 3600 * h + fps_round * 60 * m
                + fps_round * s + f - d * (total_mins - total_mins // 10))
    return fps_round * (3600 * h + 60 * m + s) + f


def frames_to_df_smpte(af, fps_round):
    d = 2 if fps_round == 30 else 4
    per_10 = fps_round * 600 - d * 9
    per_1 = fps_round * 60 - d
    d10 = af // per_10
    rem = af % per_10
    if rem < fps_round * 60:
        # first minute of each 10-minute block drops nothing
        d1 = 0
    else:
        rem -= fps_round * 60
        d1 = rem // per_1 + 1
        # inside a dropping minute the labels ;00..;(d-1) do not exist at
        # second 0 — physical frame index within the minute maps to label
        # index + d.  Omitting this shifted every timecode in dropping
        # minutes (54 of every 60) two frames early and emitted invalid
        # labels like 00:01:00;00 at minute starts.
        rem = rem % per_1 + d
    total_mins = d10 * 10 + d1
    h, m = total_mins // 60, total_mins % 60
    s, f = rem // fps_round, rem % fps_round
    return f"{h:02d}:{m:02d}:{s:02d};{f:02d}"
