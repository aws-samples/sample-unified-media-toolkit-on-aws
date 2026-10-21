"""
Step Function State: StateProbeMediaFile
-----------------------------------------
1. Parse input.s3uri and output.s3uri
2. head_object → file size (Lambda ARN routing) + archive-tier gate (fail here, named, before ffprobe)
3. ffprobe show_streams + show_format
4. Determine is_all_intra:
     - codec heuristic (fast, no decode)
     - frame scan of first 20s if heuristic returns None
5. Validate output_groups against source streams
6. Build chunk iterators
7. Write probe.json to S3

Metadata output groups are fully served here — ffprobe has already run, and mxf2raw
runs when the source is MXF. The group copy and data.output_keys are written at
probe time, so a metadata-only request needs neither the Map nor the merge state
(data.metadata_only routes the Choice state straight to Usage tracking).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from fractions import Fraction

from lib.audio_command import select_audio_streams
from lib.ffprobe import is_mxf_format, resolve_duration, run_ffprobe, scan_is_all_intra
from lib.input_schema import (
    DEFAULT_VIDEO_GROUP, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    VIDEO_FORMATS, AUDIO_FORMATS, FRAME_FORMATS, METADATA_FORMATS,
    FLAC_BACKED_FORMATS,
    compute_frame_interval,
    parse_output_groups,
    parse_s3uri,
)
from lib.mxf2raw import build_mxf_block
from lib.s3_utils import archive_tier, head_object, put_json
from lib.video_command import SNAP_DENOMINATOR, build_segment_boundaries, determine_is_all_intra
from states.probe_media_file.probe_helper import (
    build_afilters,
    build_chunk_outputs,
    build_metadata_payload,
    build_vfilters,
    resolve_video_group_settings,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)

LARGE_FILE_BYTES    = 20 * 1024 ** 3   # 20 GB
LARGE_DURATION_SECS = 7200             # 2 hrs

# Audio segment constants (audio-only path — time-based, no frame reference)
AUDIO_SEGMENT_SECS = 1800   # 30 min
AUDIO_MIN_SECS     = 180    # 3 min — merge final chunk if shorter


# ── Audio segment boundaries (time-based, audio-only files) ───────────────────

def _audio_segment_boundaries(start_sec: float, end_sec: float) -> list[tuple[float, float]]:
    total = end_sec - start_sec
    boundaries: list[tuple[float, float]] = []
    t = 0.0
    while t < total:
        dur = min(float(AUDIO_SEGMENT_SECS), total - t)
        boundaries.append((start_sec + t, start_sec + t + dur))
        t += dur
    if len(boundaries) > 1 and (boundaries[-1][1] - boundaries[-1][0]) < AUDIO_MIN_SECS:
        last = boundaries.pop()
        boundaries[-1] = (boundaries[-1][0], last[1])
    return boundaries


# ── Iterator builders ──────────────────────────────────────────────────────────

def _build_video_iterators(
    uuid: str,
    src_bucket: str, src_key: str,
    out_bucket: str, out_prefix: str,
    is_all_intra: bool,
    boundaries: list[tuple[int, int]],
    fps: Fraction,
    interlaced: bool,
    audio_streams: list[dict],
    sampling_rate: int,
    video_group_settings: dict | None,  # None → no .ts segment output
    scale_settings: dict,               # always set; drives vfilter chain crop/scale
    frame_group: dict | None,
    output_groups: list[dict],
    fn_arn: str,
    video_mapid: str,
) -> list[dict]:
    has_audio = bool(audio_streams) and (
        video_group_settings is not None  # mp4 needs audio track
        or any(g["format"] in FLAC_BACKED_FORMATS
               for g in output_groups)
    )

    frame_interval = None
    if frame_group:
        frame_interval = compute_frame_interval(fps, frame_group["frame_interval_sec"])

    vfilters = build_vfilters(scale_settings, interlaced, frame_interval,
                              has_ts_output=video_group_settings is not None,
                              video_mapid=video_mapid)
    afilters = build_afilters(audio_streams) if has_audio else []

    iterators = []
    for item_id, (start_f, end_f) in enumerate(boundaries):
        start_sec = float(Fraction(start_f) / fps)
        end_sec   = float(Fraction(end_f)   / fps)

        chunk_outputs = build_chunk_outputs(
            item_id, video_group_settings, has_audio, frame_interval,
            frame_format=frame_group["frame_format"] if frame_group else "jpeg",
            frame_group_name=frame_group["group_name"] if frame_group else "frames",
        )

        iterators.append({
            "function_arn":        fn_arn,
            "uuid":                uuid,
            "item_id":             item_id,
            "input":               {"bucket": src_bucket, "key": src_key},
            "output":              {"bucket": out_bucket, "prefix": out_prefix},
            "type":                "chunk",
            "is_all_intra":        is_all_intra,
            "nominal_start_sec":   round(start_sec, 6),
            "nominal_end_sec":     round(end_sec,   6),
            "nominal_start_frame": start_f,
            "nominal_end_frame":   end_f,
            "fps":                 str(fps),
            "interlaced":          interlaced,
            "sampling_rate":       sampling_rate,
            "frame_interval":      frame_interval,
            "filter_complex":      {"vfilters": vfilters, "afilters": afilters},
            "chunk_outputs":       chunk_outputs,
        })
    return iterators


def _build_audio_only_iterators(
    uuid: str,
    src_bucket: str, src_key: str,
    out_bucket: str, out_prefix: str,
    boundaries: list[tuple[float, float]],
    audio_streams: list[dict],
    sampling_rate: int,
    fn_arn: str,
) -> list[dict]:
    afilters = build_afilters(audio_streams)
    iterators = []
    for item_id, (start_sec, end_sec) in enumerate(boundaries):
        start_sample = math.floor(Fraction(start_sec).limit_denominator(SNAP_DENOMINATOR) * sampling_rate)
        end_sample   = math.ceil( Fraction(end_sec).limit_denominator(SNAP_DENOMINATOR)   * sampling_rate)
        chunk_outputs = build_chunk_outputs(item_id, None, True, None)
        iterators.append({
            "function_arn":      fn_arn,
            "uuid":              uuid,
            "item_id":           item_id,
            "input":             {"bucket": src_bucket, "key": src_key},
            "output":            {"bucket": out_bucket, "prefix": out_prefix},
            "type":              "chunk",
            "is_all_intra":      True,
            "nominal_start_sec": round(start_sec, 6),
            "nominal_end_sec":   round(end_sec,   6),
            "sampling_rate":     sampling_rate,
            "filter_complex":    {"vfilters": [], "afilters": afilters},
            "chunk_outputs":     chunk_outputs,
            "start_sample":      start_sample,
            "end_sample":        end_sample,
        })
    return iterators


# ── Lambda handler ─────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    log.info("StateProbeMediaFile event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    inp  = event.get("input",  {})
    out  = event.get("output", {})
    data = event.get("data",   {})
    uuid = event.get("uuid",   "unknown")

    # ── Quick edit (cut-list mode) — its own branch end to end ───────────────
    # No single input.s3uri exists for an edit over N sources, so the branch
    # sits before the s3uri parse. Everything lives in quick_edit_probe.
    if inp.get("quick_edit") is not None:
        from states.probe_media_file.quick_edit_probe import handle_quick_edit
        qe_out_bucket, qe_out_prefix = parse_s3uri(out["s3uri"])
        return handle_quick_edit(
            event, inp["quick_edit"], uuid=uuid,
            out_bucket=qe_out_bucket, out_prefix=qe_out_prefix or uuid,
            output_groups_req=out.get("output_groups"),
            fn_arn=(os.environ.get("CHUNK_TRANSCODE_FUNCTION_ARN")
                    or getattr(context, "invoked_function_arn", "")))

    if inp.get("frame_capture") is not None:
        from states.probe_media_file.frame_capture_probe import handle_frame_capture
        fc_out_bucket, fc_out_prefix = parse_s3uri(out["s3uri"])
        return handle_frame_capture(
            event, inp["frame_capture"], uuid=uuid,
            out_bucket=fc_out_bucket, out_prefix=fc_out_prefix or uuid,
            output_groups_req=out.get("output_groups"),
            fn_arn=(os.environ.get("CHUNK_TRANSCODE_FUNCTION_ARN")
                    or getattr(context, "invoked_function_arn", "")))

    # ── Parse s3uris ──────────────────────────────────────────────────────────
    src_bucket, src_key = parse_s3uri(inp["s3uri"])
    out_bucket, out_prefix = parse_s3uri(out["s3uri"])
    if not out_prefix:
        out_prefix = uuid

    # ── input_config ──────────────────────────────────────────────────────────
    ic             = inp.get("input_config") or {}
    use_ch_map     = bool(ic.get("use_channel_mapping",  False))
    ch_map         = list(ic.get("audio_channel_mapping") or [])
    use_partial    = bool(ic.get("use_partial",    False))
    mark_in_out    = list(ic.get("mark_in_out") or [])
    ic_width       = int(ic.get("width")  or DEFAULT_WIDTH)
    ic_height      = int(ic.get("height") or DEFAULT_HEIGHT)
    ic_crop_x      = int(ic.get("crop_x") or 0)
    ic_crop_y      = int(ic.get("crop_y") or 0)

    # ── Function ARNs ─────────────────────────────────────────────────────────
    default_arn = (
        os.environ.get("CHUNK_TRANSCODE_FUNCTION_ARN")
        or getattr(context, "invoked_function_arn", "")
    )
    large_arn = os.environ.get("CHUNK_TRANSCODE_FUNCTION_ARN_LARGE") or default_arn

    # ── File size and access tier ─────────────────────────────────────────────
    t1        = time.monotonic()
    head      = head_object(src_bucket, src_key)
    file_size = head.get("ContentLength", 0)
    # restore= carries expiry-date when set: a restored copy expiring mid-run
    # fails at chunk time as a raw InvalidObjectState, and this line is then
    # the only record of when the restored copy was due to lapse.
    log.info("head_object: %d bytes (%.2f GB)  storage_class=%s  archive_status=%s  restore=%s  elapsed=%.2fs",
             file_size, file_size / 1024 ** 3,
             head.get("StorageClass") or "STANDARD", head.get("ArchiveStatus") or "-",
             head.get("Restore") or "-",
             time.monotonic() - t1)

    # Fail here with the tier named, rather than letting ffprobe hit
    # InvalidObjectState on the presigned URL and report a bare "exited 1".
    tier = archive_tier(head)
    if tier:
        fix = ("A restore is already in progress — retry once it completes."
               if head.get("Restore") else
               f"Restore it from {tier} first, then retry.")
        raise ValueError(
            f"s3://{src_bucket}/{src_key} is in the {tier} access tier and cannot be read. {fix}"
        )

    # ── ffprobe ───────────────────────────────────────────────────────────────
    t1    = time.monotonic()
    probe = run_ffprobe(src_bucket, src_key)
    log.info("ffprobe: elapsed=%.2fs", time.monotonic() - t1)

    streams  = probe.get("streams", [])
    fmt      = probe.get("format",  {})
    fmt_name = (fmt.get("format_name") or "").lower()

    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    for i, s in enumerate(vstreams):
        s["mapid"] = f"0:v:{i}"
    for i, s in enumerate(astreams):
        s["mapid"] = f"0:a:{i}"

    has_video = bool(vstreams)

    # ── Audio stream selection (before parse_output_groups so has_audio reflects decodable streams)
    audio_streams = select_audio_streams(astreams, use_ch_map, ch_map)
    has_audio     = bool(audio_streams)
    sampling_rate = 48000   # FLAC intermediate rate; used for atrim sample-accurate boundary math

    # ── Validate + parse output_groups ───────────────────────────────────────
    output_groups = parse_output_groups(out.get("output_groups"), has_video, has_audio)

    video_groups    = [g for g in output_groups if g["format"] in VIDEO_FORMATS]
    # ad_mix rides the audio machinery end to end: it needs the same
    # .flac chunks, and it reconciles/segments exactly like wav/m4a —
    # only the MERGE branches on its format
    audio_groups    = [g for g in output_groups
                       if g["format"] in FLAC_BACKED_FORMATS]
    frame_groups    = [g for g in output_groups if g["format"] in FRAME_FORMATS]
    metadata_groups = [g for g in output_groups if g["format"] in METADATA_FORMATS]

    # Metadata-only — every requested group is a metadata group. Branch on this
    # flag, never on "iterators is empty": an empty iterator list is also what a
    # bug looks like, and that must still fail loudly.
    metadata_only = bool(metadata_groups) and len(metadata_groups) == len(output_groups)

    # Only first video and first frame group used — one video chain, one frame branch
    video_group = video_groups[0] if video_groups else None
    frame_group = frame_groups[0] if frame_groups else None
    if len(video_groups) > 1:
        log.warning("Multiple mp4 output groups specified; only %r will be used",
                    video_group["group_name"])
    if len(frame_groups) > 1:
        log.warning("Multiple frame output groups specified; only %r will be used",
                    frame_group["group_name"])

    # ── fps + duration ────────────────────────────────────────────────────────
    v       = vstreams[0] if has_video else None
    ref     = v if has_video else (astreams[0] if astreams else {})
    fps_str = ref.get("r_frame_rate") or ref.get("avg_frame_rate") or "0/0"

    try:
        fps_frac = Fraction(fps_str.replace(":", "/"))
    except (ValueError, ZeroDivisionError):
        fps_frac = Fraction(0)
    duration_sec, dur_src = resolve_duration(ref, fmt)
    log.info("Duration: %.3fs (%s)  fps=%s", duration_sec, dur_src, fps_str)
    if duration_sec <= 0:
        raise ValueError(
            f"Cannot determine duration from source (got {duration_sec}s) — "
            "file may be corrupt or incomplete")

    # ── is_all_intra ──────────────────────────────────────────────────────────
    # Metadata groups need it too — it is one of the fields no caller can derive
    # from raw ffprobe output, so it is worth the frame-scan fallback (§7.4:
    # ambiguous codecs only).
    is_all_intra: bool = True   # default safe value for audio-only
    if has_video and (video_group or frame_group or metadata_groups):
        heuristic = determine_is_all_intra(v)
        if heuristic is not None:
            is_all_intra = heuristic
        else:
            t1 = time.monotonic()
            is_all_intra = scan_is_all_intra(src_bucket, src_key)
            log.info("scan_is_all_intra: %s  elapsed=%.2fs",
                     is_all_intra, time.monotonic() - t1)

    # ── interlaced ────────────────────────────────────────────────────────────
    interlaced = False
    if has_video:
        field_order = (v.get("field_order") or "progressive").lower()
        interlaced  = field_order not in ("progressive", "unknown", "")

    # ── mark_in_out ───────────────────────────────────────────────────────────
    start_sec = 0.0
    end_sec   = duration_sec
    if use_partial and len(mark_in_out) >= 2:
        start_sec = max(0.0,          float(mark_in_out[0]))
        end_sec   = min(duration_sec, float(mark_in_out[1]))
        if start_sec >= end_sec:
            raise ValueError(
                f"mark_in_out produces empty range: {start_sec}s >= {end_sec}s "
                f"(source duration={duration_sec}s, requested={mark_in_out[:2]})")
        log.info("Partial: %.1fs → %.1fs", start_sec, end_sec)

    # ── Function ARN routing ──────────────────────────────────────────────────
    fn_arn = (
        large_arn
        if (file_size >= LARGE_FILE_BYTES or duration_sec > LARGE_DURATION_SECS)
        else default_arn
    )

    # ── Build iterators ───────────────────────────────────────────────────────
    iterators: list[dict] = []

    if has_video and (video_group or frame_group):
        if fps_frac <= 0:
            raise ValueError(
                f"Cannot build video iterators — invalid fps {fps_str!r}")

        ic_kwargs = dict(
            crop_x=ic_crop_x, crop_y=ic_crop_y,
            target_width=ic_width, target_height=ic_height,
        )
        # video_group_settings — only set when there is an mp4 output; drives .ts segment
        video_group_settings = (
            resolve_video_group_settings(video_group, v, fps_frac, **ic_kwargs)
            if video_group else None
        )
        # scale_settings — always needed to build vfilter chain (crop, scale, PAR normalise).
        # When mp4 group exists, reuse its already-computed settings.
        # When frame-only, fall back to DEFAULT_VIDEO_GROUP for gop/codec fields;
        # crop and scale still come from input_config.
        if video_group_settings:
            scale_settings = video_group_settings
        else:
            scale_settings = resolve_video_group_settings(
                DEFAULT_VIDEO_GROUP, v, fps_frac, **ic_kwargs)

        if frame_group:
            frame_group["width"]  = scale_settings["width"]
            frame_group["height"] = scale_settings["height"]

        boundaries = build_segment_boundaries(start_sec, end_sec, fps_frac)
        iterators = _build_video_iterators(
            uuid, src_bucket, src_key, out_bucket, out_prefix,
            is_all_intra, boundaries,
            fps_frac, interlaced,
            audio_streams, sampling_rate,
            video_group_settings, scale_settings, frame_group,
            output_groups,
            fn_arn,
            video_mapid=v["mapid"],
        )
    elif audio_groups and not (video_group or frame_group):
        # Audio-only source — time-based segmentation
        if not audio_streams:
            raise ValueError("No usable audio stream found")
        boundaries_t = _audio_segment_boundaries(start_sec, end_sec)
        iterators = _build_audio_only_iterators(
            uuid, src_bucket, src_key, out_bucket, out_prefix,
            boundaries_t, audio_streams, sampling_rate, fn_arn,
        )
    elif metadata_only:
        # Nothing to chunk — the metadata group is fully served by this state.
        log.info("Metadata-only request — no iterators; Map and merge are skipped")
    else:
        raise ValueError(
            "Cannot build iterators: no compatible output_groups for source streams")

    log.info("Built %d iterators  is_all_intra=%s  interlaced=%s",
             len(iterators), is_all_intra, interlaced)

    # ── Metadata output groups ────────────────────────────────────────────────
    # Written here, not at merge: ffprobe has already run, and a metadata-only
    # request never reaches the merge state. Mixed requests take the same path so
    # merge never learns about metadata groups.
    output_keys: dict[str, str] = {}
    if metadata_groups:
        mxf_block = None
        if is_mxf_format(fmt_name):
            t1 = time.monotonic()
            # Raises if mxf2raw cannot produce real measurements on a real MXF —
            # a caller handed frame_count=-1 has been handed a wrong number.
            mxf_block = build_mxf_block(src_bucket, src_key)
            log.info("mxf2raw: elapsed=%.2fs", time.monotonic() - t1)

        for group in metadata_groups:
            payload = build_metadata_payload(
                detail=group["detail"],
                fmt_name=fmt_name,
                duration_sec=duration_sec,
                duration_source=dur_src,
                file_size=file_size,
                has_video=has_video,
                has_audio=has_audio,
                is_all_intra=is_all_intra,
                vstream=v,
                audio_streams=audio_streams,
                interlaced=interlaced,
                probe=probe,
                mxf_block=mxf_block,
            )
            meta_key = f"{out_prefix}/{group['group_name']}/{group['name']}.json"
            put_json(out_bucket, meta_key, payload)
            output_keys[group["group_name"]] = meta_key
            log.info("metadata group %r (detail=%s) written: s3://%s/%s",
                     group["group_name"], group["detail"], out_bucket, meta_key)

    # ── Write probe.json ──────────────────────────────────────────────────────
    t1        = time.monotonic()
    probe_key = f"{out_prefix}/probe.json"
    put_json(out_bucket, probe_key, {
        "uuid":             uuid,
        "source_bucket":    src_bucket,
        "source_key":       src_key,
        "format_name":      fmt_name,
        "duration_sec":     duration_sec,
        "duration_source":  dur_src,
        "file_size":        file_size,
        "probe_info":       probe,
        "iterators":        iterators,
    })
    log.info("probe.json uploaded: elapsed=%.2fs", time.monotonic() - t1)

    elapsed = round(time.monotonic() - t0, 2)
    log.info(
        "Probe complete: %.1fs (%s)  has_video=%s  audio_streams=%d  "
        "iterators=%d  elapsed=%.2fs",
        duration_sec, dur_src, has_video, len(audio_streams), len(iterators), elapsed,
    )

    # Reconciled groups — reflects Dolby E filtering (audio_groups may have been cleared)
    reconciled_groups = (
        ([video_group] if video_group else [])
        + ([frame_group] if frame_group else [])
        + audio_groups
        + metadata_groups
    )

    return {
        **event,
        "status": "COMPLETED",
        "data": {
            **data,
            "destination":    {"bucket": out_bucket, "prefix": out_prefix},
            "output_groups":  reconciled_groups,
            "probe_s3_key":   probe_key,
            "duration_sec":   duration_sec,
            "has_video":      has_video,
            "has_audio":      has_audio,
            "is_all_intra":   is_all_intra,
            "metadata_only":  metadata_only,
            "function_arn":   fn_arn, # for StateMergeChunks
            "output_keys":    output_keys,
            "iterators":      iterators,
        },
    }