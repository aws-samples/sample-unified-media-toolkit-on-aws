"""
The ad_mix output group's render (2026-08-31, for audio-description-on-aws):
the programme bed ducked under a bus of pre-placed narration takes.

Runs at MERGE over the concatenated .flac bed — the same full-programme
audio pass the m4a group already performs there, so no new duration
ceiling is introduced (the §406 measurement ruling does not apply: this
renders a stereo DELIVERABLE, and the stereo-downmixed bed is exactly
what an AD mix wants).

The AD workflow renders NO audio itself (boto3-only); everything arrives
in its render_manifest.json:

    {"sample_rate": 48000,
     "af_chain": "equalizer=...,loudnorm=I=-17:TP=-2:LRA=7",
     "takes": [{"window_id": 12, "wav": "s3://...", "at": 198.4,
                "tempo": 1.13}, ...]}

Command builders are PURE (args lists out) so they test offline; the
orchestration at the bottom shells out through lib.proc_utils.run.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

FFMPEG_TIMEOUT = 840

# same operator knobs as merge_helper — an FFMPEG_BIN/FFMPEG_THREADS
# override must govern EVERY merge-stage command, ad_mix included
FFMPEG  = os.environ.get("FFMPEG_BIN", "ffmpeg")
THREADS = int(os.environ.get("FFMPEG_THREADS", "0"))

# the concat demuxer refuses the presigned https URLs in the bed list
# unless nested opens are whitelisted — merge_helper passes exactly this
# for the same pattern (found in the 2026-08-31 review: omitting it made
# every ad_mix render die at the final pass)
CONCAT_WHITELIST = ["-protocol_whitelist", "file,http,https,tcp,tls,crypto",
                    "-f", "concat", "-safe", "0"]

# amix takes many inputs, but command length and fd count do not scale
# forever — above this the takes pre-mix into sub-buses first. Takes
# never overlap (the AD workflow's collide pass guarantees spacing), so
# summation order cannot change the result.
MAX_INPUTS_PER_MIX = 150

# duck + final loudness — the AD runner's measured chain verbatim
SIDECHAIN = ("sidechaincompress=threshold=0.015:ratio=8:attack=60:"
             "release=500:makeup=1")
FINAL_LOUDNESS = "loudnorm=I=-24:TP=-2:LRA=11,alimiter=limit=0.9"


def take_render_cmd(src: str, out_path: str, af_chain: str, tempo: float,
                    sample_rate: int) -> list:
    """Render ONE fitted take: the manifest's af_chain, then atempo when
    the fit asked for one. Stereo, so the bus needs no layout juggling
    against the stereo bed."""
    af = af_chain + (f",atempo={tempo:.4f}" if tempo > 1.001 else "")
    return [FFMPEG, "-threads", str(THREADS), "-y", "-v", "error",
            "-i", src, "-af", af,
            "-ar", str(sample_rate), "-ac", "2", out_path]


def bus_mix_cmd(take_paths: list, offsets_sec: list, out_path: str,
                sample_rate: int) -> list:
    """Sum N takes at their placements into one bus: adelay per input,
    amix normalize=0 (a SUM — takes never overlap, so summing equals
    each take alone; amix's default averaging would duck every take by
    1/N)."""
    cmd = [FFMPEG, "-threads", str(THREADS), "-y", "-v", "error"]
    for p in take_paths:
        cmd += ["-i", p]
    parts = []
    for i, off in enumerate(offsets_sec):
        ms = max(0, round(off * 1000))
        parts.append(f"[{i}:a]adelay={ms}:all=1[d{i}]")
    labels = "".join(f"[d{i}]" for i in range(len(take_paths)))
    parts.append(f"{labels}amix=inputs={len(take_paths)}:normalize=0"
                 f":duration=longest[bus]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[bus]",
            "-ar", str(sample_rate), "-ac", "2", out_path]
    return cmd


def duck_mix_cmd(bed_list_path: str, bus_path: str, out_path: str,
                 sample_rate: int, audio_channels: int) -> list:
    """The final pass: concat-demux the .flac bed (input 0), duck it
    under the bus (sidechain), sum, one whole-programme loudness pass.

    BOTH bus legs are apad'ed (review 2026-08-31): sidechaincompress
    and amix end at their SHORTEST input, and the bus ends at the last
    take — unpadded, a programme whose final narration lands at 50:00
    silently lost everything after it. With infinite pads on the bus,
    [duck] runs exactly to the bed's end and amix duration=first pins
    the mix there — the bed's length, in both directions."""
    return ([FFMPEG, "-threads", str(THREADS), "-y", "-v", "error"]
            + CONCAT_WHITELIST + ["-i", bed_list_path,
            "-i", bus_path,
            "-filter_complex",
            "[1:a]apad[busp];[busp]asplit=2[sc][mixin];"
            f"[0:a][sc]{SIDECHAIN}[duck];"
            "[duck][mixin]amix=inputs=2:duration=first:normalize=0[m];"
            f"[m]{FINAL_LOUDNESS}[out]",
            "-map", "[out]", "-ar", str(sample_rate),
            "-ac", str(audio_channels), "-c:a", "pcm_s16le", out_path])


def plan_bus_batches(n_takes: int, cap: int = MAX_INPUTS_PER_MIX) -> list:
    """[(start, end)] index ranges — one range when the take count fits
    one amix, else sub-bus batches merged by a second-stage mix."""
    return [(i, min(i + cap, n_takes)) for i in range(0, n_takes, cap)]


def merge_ad_mix(out_bucket: str, flac_keys: list, tmp_dir: str,
                 out_key: str, group: dict) -> None:
    """Orchestration (called by StateMergeChunks for an ad_mix group):
    fetch manifest -> render takes -> bus (batched) -> duck+loudness ->
    upload. Imports stay inside so the pure builders above import
    without boto3/S3 for offline tests."""
    from lib.input_schema import parse_s3uri
    from lib.proc_utils import run
    from lib.s3_utils import presign, s3, upload_file

    m_bucket, m_key = parse_s3uri(group["render_manifest_s3uri"])
    manifest = json.loads(
        s3.get_object(Bucket=m_bucket, Key=m_key)["Body"].read())
    takes = sorted(manifest.get("takes") or [],
                   key=lambda t: float(t["at"]))
    sample_rate = int(group.get("sampling_rate")
                      or manifest.get("sample_rate") or 48000)
    af_chain = manifest["af_chain"]

    bed_list = os.path.join(tmp_dir, "ad_mix_bedlist.txt")
    with open(bed_list, "w") as f:
        for key in flac_keys:
            f.write(f"file '{presign(out_bucket, key)}'\n")

    out_path = os.path.join(tmp_dir, os.path.basename(out_key))

    if not takes:
        # an empty script is a legal (if sad) delivery: the "mix" is the
        # loudness-normalized bed — same output contract, honest content
        log.warning("ad_mix %r: manifest has no takes — delivering the "
                    "bed alone", group["group_name"])
        run([FFMPEG, "-threads", str(THREADS), "-y", "-v", "error"]
            + CONCAT_WHITELIST + ["-i", bed_list,
            "-af", FINAL_LOUDNESS, "-ar", str(sample_rate),
            "-ac", str(group.get("audio_channels") or 2),
            "-c:a", "pcm_s16le", out_path], timeout=FFMPEG_TIMEOUT)
        upload_file(out_bucket, out_key, out_path)
        return

    take_paths, offsets = [], []
    for t in takes:
        t_bucket, t_key = parse_s3uri(t["wav"])
        # takes are DOWNLOADED, never streamed (2026-09-10): ffmpeg over
        # presigned https has no socket timeout, and one stalled TLS
        # read hangs the render until the subprocess deadline — measured
        # twice on one asset, same take, ~180 serial fetches in. boto3
        # retries with bounds; a ~150 KB wav costs nothing to land. The
        # presigned-https path REMAINS for the bed FLAC concat below,
        # where streaming multi-GB chunks is the point.
        src = os.path.join(tmp_dir,
                           f"take_src_{int(t['window_id']):04d}.wav")
        s3.download_file(t_bucket, t_key, src)
        path = os.path.join(tmp_dir, f"take_{int(t['window_id']):04d}.wav")
        run(take_render_cmd(src, path, af_chain,
                            float(t.get("tempo") or 1.0), sample_rate),
            timeout=FFMPEG_TIMEOUT)
        os.remove(src)
        take_paths.append(path)
        offsets.append(float(t["at"]))

    sub_buses = []
    for bi, (a, b) in enumerate(plan_bus_batches(len(take_paths))):
        sub = os.path.join(tmp_dir, f"ad_bus_{bi:02d}.wav")
        run(bus_mix_cmd(take_paths[a:b], offsets[a:b], sub, sample_rate),
            timeout=FFMPEG_TIMEOUT)
        sub_buses.append(sub)
    if len(sub_buses) == 1:
        bus = sub_buses[0]
    else:
        # sub-buses already carry their absolute placements (adelay was
        # absolute), so the second stage sums at offset zero
        bus = os.path.join(tmp_dir, "ad_bus.wav")
        run(bus_mix_cmd(sub_buses, [0.0] * len(sub_buses), bus,
                        sample_rate), timeout=FFMPEG_TIMEOUT)

    run(duck_mix_cmd(bed_list, bus, out_path, sample_rate,
                     int(group.get("audio_channels") or 2)),
        timeout=FFMPEG_TIMEOUT)
    upload_file(out_bucket, out_key, out_path)
    log.info("ad_mix %r: %d takes mixed -> %s", group["group_name"],
             len(takes), out_key)
