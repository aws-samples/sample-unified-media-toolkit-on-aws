# dynamic_chunking_handler.py
"""
Step Function State: StateDynamicChunkingBySpeechTurns  (single Lambda invocation)
───────────────────────────────────────────────────────────────────────────────────
Reads all VAD segments_N.json files produced by StateRunVADModel, flattens them
into one timeline, groups temporally close turns into speech groups, and
partitions groups into dynamic transcription chunks.

Speech groups
─────────────
Adjacent turns with inter-turn gaps below speech_group_max_gap_sec (default 1.0s)
are merged into a single speech group.  This ensures:
  • Whisper sees natural audio continuity within groups (real silences preserved)
  • Sentence splitting (wtpsplit) can detect sentences spanning VAD turn boundaries
  • No synthetic silence padding between turns within a group

A group is the atomic transcription unit.  Sentence splitting runs per group,
not per turn, so a sentence like "I think we should go to the [pause] store"
split across two VAD turns is correctly detected as one sentence.

Chunking rules
──────────────
• Accumulate speech seconds greedily from the speech group list.
• Emit a dynamic chunk when accumulated speech >= dynamic_speech_chunk_secs.
• Groups are never split across chunks — the group is the atomic unit.
• Remainder groups at the end are emitted as a final (potentially short) chunk.

No overlap
──────────
Chunk boundaries always fall between groups separated by ≥ 1.0s — a natural
sentence boundary.  No sentence can span this gap, so overlap is unnecessary.

Coverage invariant
──────────────────
Every VAD turn appears in exactly one dynamic chunk.  No turn is duplicated.

Output schema — dynamic_segments_N.json
───────────────────────────────────────
{
    "dyn_chunk_idx":      0,
    "fetch_start_sec":    4.9,
    "fetch_end_sec":      403.0,
    "fetch_duration_sec": 398.1,
    "num_turns":          20,
    "num_groups":         8,
    "speech_groups": [
        {
            "group_idx":    0,
            "start":        4.9,
            "end":          12.3,
            "duration":     7.4,
            "speech_sec":   6.8,
            "num_turns":    3,
            "turn_indices": [0, 1, 2],
            "turns": [
                { "start": 4.9, "end": 6.2, "duration": 1.3,
                  "has_overlap": false, "turn_idx": 0 },
                ...
            ]
        },
        ...
    ],
    "turns": [
        { "start": 4.9, "end": 6.2, "duration": 1.3,
          "has_overlap": false, "turn_idx": 0 },
        ...
    ]
}

Two views of the same data:
  • speech_groups — consumed by the transcription handler (group-level audio,
                    group-level sentence splitting)
  • turns         — consumed by the embedding handler (per-turn embeddings,
                    unchanged from the previous design)
"""

import json
import logging
import time
import traceback
from typing import Any

from lib.s3_utils import s3, get_json, put_json

# ── logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)
log.propagate = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _progress(step: str, detail: str = "", elapsed: float = 0.0) -> None:
    log.info("  ▶ %-32s %s  (+%.2fs)", step, detail, elapsed)


def _read_segments(bucket: str, key: str) -> dict:
    """Fetch and parse a segments_N.json from S3."""
    return get_json(bucket, key)


def _write_segments(bucket: str, key: str, payload: dict) -> None:
    """Serialise and upload a dynamic_segments_N.json to S3."""
    put_json(bucket, key, payload)
    log.debug("  wrote s3://%s/%s", bucket, key)


# ── flatten turns (unchanged from previous design) ───────────────────────────

def _flatten_turns(
    iterators: list[dict],
    bucket: str,
    min_silence_duration_sec: float = 0.3,
) -> list[dict]:
    """
    Read all segments_N.json files in chunk_idx order, flatten into one turn
    list, and merge any turns that were split at a fixed-chunk boundary.

    Split detection: the last turn of chunk N ended within CLIP_TOLERANCE of
    the fixed boundary, AND the gap to the first turn of chunk N+1 is less
    than min_silence_duration_sec.  If the gap is shorter than the minimum
    silence the VAD can recognise, the two halves must be one continuous turn.

    Each turn in the returned list carries only the canonical schema fields:
        start, end, duration, has_overlap, turn_idx  (globally unique)
    """
    sorted_items = sorted(iterators, key=lambda x: x["chunk_idx"])
    all_turns: list[dict] = []
    segs_by_chunk: dict[int, dict] = {}

    for item in sorted_items:
        key = item["segments_key"]
        log.debug("  reading s3://%s/%s", bucket, key)
        seg = _read_segments(bucket, key)
        segs_by_chunk[item["chunk_idx"]] = seg

        boundary_sec = (
            segs_by_chunk[item["chunk_idx"] - 1]["fetch_end_sec"]
            if item["chunk_idx"] > 0
            else None
        )

        for turn_pos, t in enumerate(seg["turns"]):
            _CLIP_TOLERANCE = 0.05  # 50 ms

            if (
                turn_pos == 0
                and all_turns
                and boundary_sec is not None
                and (boundary_sec - all_turns[-1]["end"]) < _CLIP_TOLERANCE
                and (t["start"] - all_turns[-1]["end"]) < min_silence_duration_sec
            ):
                prev = all_turns[-1]
                boundary = prev["end"]
                merged_end = t["end"]
                all_turns[-1] = {
                    "start":       prev["start"],
                    "end":         merged_end,
                    "duration":    round(merged_end - prev["start"], 3),
                    "has_overlap": prev["has_overlap"] or t.get("has_overlap", False),
                    "turn_idx":    prev["turn_idx"],
                }
                log.info(
                    "  boundary merge at %.1fs: [%.3fs → %.3fs] dur=%.3fs "
                    "(%.3fs + %.3fs)",
                    boundary,
                    prev["start"], merged_end,
                    all_turns[-1]["duration"],
                    prev["duration"], t["duration"],
                )
            else:
                all_turns.append({
                    "start":       t["start"],
                    "end":         t["end"],
                    "duration":    t["duration"],
                    "has_overlap": t.get("has_overlap", False),
                    "turn_idx":    len(all_turns),
                })

    return all_turns


# ── speech group formation ────────────────────────────────────────────────────

def _form_speech_groups(
    turns:                    list[dict],
    speech_group_max_gap_sec: float,
) -> list[dict]:
    """
    Merge adjacent turns into speech groups based on temporal proximity.

    Turns with inter-turn gap < max_gap_sec are placed in the same group.
    Guarantees: inter-group gaps are always ≥ max_gap_sec.

    Each group dict contains:
        group_idx    : int         — 0-based position in the group list
        start        : float       — absolute start (first turn's start)
        end          : float       — absolute end (last turn's end)
        duration     : float       — wall-clock duration (end − start),
                                     includes inter-turn silences
        speech_sec   : float       — sum of individual turn durations
        num_turns    : int         — number of turns in the group
        turn_indices : list[int]   — global turn_idx values
        turns        : list[dict]  — the turn dicts themselves
    """
    if not turns:
        return []

    groups: list[dict] = []
    current_turns: list[dict] = [turns[0]]

    for t in turns[1:]:
        gap = t["start"] - current_turns[-1]["end"]
        if gap < speech_group_max_gap_sec:
            current_turns.append(t)
        else:
            groups.append(_build_group(len(groups), current_turns))
            current_turns = [t]

    if current_turns:
        groups.append(_build_group(len(groups), current_turns))

    return groups


def _build_group(group_idx: int, turns: list[dict]) -> dict:
    """Construct a speech group dict from a list of consecutive turns."""
    return {
        "group_idx":    group_idx,
        "start":        turns[0]["start"],
        "end":          turns[-1]["end"],
        "duration":     round(turns[-1]["end"] - turns[0]["start"], 3),
        "speech_sec":   round(sum(t["duration"] for t in turns), 3),
        "num_turns":    len(turns),
        "turn_indices": [t["turn_idx"] for t in turns],
        "turns":        turns,
    }


# ── dynamic chunking ─────────────────────────────────────────────────────────

def _make_dynamic_chunks(
    groups:                    list[dict],
    dynamic_speech_chunk_secs: float,
) -> list[dict]:
    """
    Partition speech groups into dynamic chunks.

    Rules:
      • Greedy accumulation of group speech seconds until >= target_secs.
      • A group is never split across chunks.
      • A single group exceeding target_secs is emitted as its own chunk.
      • Remainder groups at the end form the final chunk.
      • No overlap — each group appears in exactly one chunk.
    """
    chunks: list[dict] = []
    i = 0
    n = len(groups)

    while i < n:
        speech_acc = 0.0
        j = i

        while j < n:
            speech = groups[j]["speech_sec"]
            speech_acc += speech
            # Single group that alone exceeds target — emit immediately
            if j == i and speech >= dynamic_speech_chunk_secs:
                break
            if speech_acc >= dynamic_speech_chunk_secs:
                break
            j += 1
        else:
            # Exhausted groups — emit remainder
            chunk_groups = groups[i:]
            if chunk_groups:
                chunks.append(_build_chunk(len(chunks), chunk_groups))
            break

        chunk_groups = groups[i:j + 1]
        chunks.append(_build_chunk(len(chunks), chunk_groups))
        i = j + 1

    return chunks


def _build_chunk(chunk_idx: int, chunk_groups: list[dict]) -> dict:
    """Construct a dynamic chunk dict from a list of speech groups."""
    all_turn_indices: list[int] = []
    all_turns: list[dict] = []
    for g in chunk_groups:
        all_turn_indices.extend(g["turn_indices"])
        all_turns.extend(g["turns"])

    return {
        "dyn_chunk_idx":  chunk_idx,
        "fetch_start_sec": chunk_groups[0]["start"],
        "fetch_end_sec":   chunk_groups[-1]["end"],
        "speech_sec":      round(
            sum(g["speech_sec"] for g in chunk_groups), 3),
        "num_turns":       sum(g["num_turns"] for g in chunk_groups),
        "num_groups":      len(chunk_groups),
        "turn_range":      [min(all_turn_indices), max(all_turn_indices)],
        "speech_groups":   chunk_groups,
        "turns":           all_turns,
    }


# ── S3 writer ────────────────────────────────────────────────────────────────

def _write_dynamic_segments(
    dyn_chunks:    list[dict],
    bucket:        str,
    output_prefix: str,
) -> None:
    """
    Write dynamic_segments_{n}.json to S3 for each chunk.

    Each file contains:
      • speech_groups — for the transcription handler (group-level audio)
      • turns         — for the embedding handler (per-turn embeddings)
    """
    prefix = f"{output_prefix}/dynamic"
    for c in dyn_chunks:
        n = c["dyn_chunk_idx"]
        key = f"{prefix}/dynamic_segments_{n}.json"
        payload = {
            "dyn_chunk_idx":    n,
            "fetch_start_sec":  c["fetch_start_sec"],
            "fetch_end_sec":    c["fetch_end_sec"],
            "fetch_duration_sec": round(
                c["fetch_end_sec"] - c["fetch_start_sec"], 3),
            "num_turns":        c["num_turns"],
            "num_groups":       c["num_groups"],
            "speech_groups":    c["speech_groups"],
            "turns":            c["turns"],
        }
        _write_segments(bucket, key, payload)
        c["dyn_segment_key"] = key
        log.info(
            "  wrote dynamic_segments_%d.json  groups=%d  turns=%d  "
            "[%.1fs → %.1fs]",
            n, c["num_groups"], c["num_turns"],
            c["fetch_start_sec"], c["fetch_end_sec"],
        )


# ── output shaping ───────────────────────────────────────────────────────────

def _shape_output(
    dyn_chunks:    list[dict],
    audio_bucket:  str,
    audio_key:     str,
    output_bucket: str,
    output_prefix: str,
    model_cfg:     dict,
    audio_cfg:     dict,
    vad_cfg:       dict,
    embed_cfg:     dict,
) -> list[dict]:
    """
    Shape each chunk descriptor into a full iterator item for downstream
    Map states (transcription, embedding).
    """
    result = []
    for c in dyn_chunks:
        item = {
            "dyn_chunk_idx":   c["dyn_chunk_idx"],
            "fetch_start_sec": c["fetch_start_sec"],
            "fetch_end_sec":   c["fetch_end_sec"],
            "speech_sec":      c["speech_sec"],
            "num_turns":       c["num_turns"],
            "num_groups":      c["num_groups"],
            "turn_range":      c["turn_range"],
            "dyn_segment_key": c["dyn_segment_key"],
            "transcript_key":  None,
            "status":          "PENDING",

            "audio_bucket":    audio_bucket,
            "audio_key":       audio_key,
            "output_bucket":   output_bucket,
            "output_prefix":   f"{output_prefix}/dynamic",

            "model_cfg":       model_cfg,
            "audio_cfg":       audio_cfg,
            "vad_cfg":         vad_cfg,
            "embed_cfg":       embed_cfg,
        }
        result.append(item)
    return result


# ── coverage check ───────────────────────────────────────────────────────────

def _verify_coverage(all_turns: list[dict], dyn_chunks: list[dict]) -> None:
    """
    Verify every VAD turn appears in exactly one dynamic chunk.
    Raises RuntimeError on missing turns.  Warns on unexpected duplicates.
    """
    covered: set[int] = set()
    duplicates: set[int] = set()
    for c in dyn_chunks:
        for t in c["turns"]:
            tid = t["turn_idx"]
            if tid in covered:
                duplicates.add(tid)
            covered.add(tid)

    all_indices = {t["turn_idx"] for t in all_turns}
    missing = all_indices - covered

    if duplicates:
        log.warning(
            "  Coverage: %d turn(s) in multiple chunks "
            "(unexpected without overlap): %s",
            len(duplicates), sorted(duplicates)[:10],
        )
    if missing:
        raise RuntimeError(
            f"Coverage violation: {len(missing)} turns not in any dynamic "
            f"chunk: {sorted(missing)[:10]}"
        )
    log.info("  Coverage check: %d/%d turns covered ✓",
             len(covered), len(all_turns))


# ── handler ───────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    log.setLevel(logging.INFO)
    log.info("StateDynamicChunkingBySpeechTurns event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # ── unpack event ──
        data = event["data"]
        audio_bucket = data["audio_bucket"]
        audio_key = data["audio_key"]
        output_bucket = data["output_bucket"]
        output_prefix = data["output_prefix"]

        if not (audio_bucket and audio_key and output_bucket and output_prefix):
            raise ValueError(
                f"data.audio_bucket, data.audio_key, data.output_bucket, data.output_prefix are required "
                f"(got audio_bucket={audio_bucket!r}, audio_key={audio_key!r})"
                f"(got output_bucket={output_bucket!r}, output_prefix={output_prefix!r})"
            )

        model_cfg = data["model_cfg"]
        embed_cfg = data["embed_cfg"]
        audio_cfg = data["audio_cfg"]
        vad_cfg = data["vad_cfg"]

        # ── Get vad config ────────────────────────────────────────────
        dynamic_speech_chunk_secs: float = float(
            vad_cfg["dynamic_speech_chunk_secs"])
        speech_group_max_gap_sec: float = float(
            vad_cfg["speech_group_max_gap_sec"])
        min_silence_duration_sec: float = float(
            vad_cfg["min_silence_duration_sec"])

        iterators: list[dict] = data["iterators"]

        # ── set max elapsed_sec from vad iterators ──
        elapsed_sec_vad = max(
            item.get("elapsed_sec", 0) for item in iterators)

        log.info(
            "  %d VAD chunks  target=%.0fs  max_gap=%.1fs  min_silence=%.3fs",
            len(iterators), dynamic_speech_chunk_secs, speech_group_max_gap_sec, min_silence_duration_sec,
        )

        # ── flatten all VAD turns ──
        _progress("vad: reading all segments",
                  f"{len(iterators)} files", time.monotonic() - t0)
        all_turns = _flatten_turns(
            iterators, output_bucket, min_silence_duration_sec)
        if not all_turns:
            raise RuntimeError(
                "No VAD turns found after flattening — check VAD output")
        log.info("  Flat turn list: %d turns  [%.1fs → %.1fs]",
                 len(all_turns), all_turns[0]["start"], all_turns[-1]["end"])

        # ── form speech groups ──
        _progress("grouping: forming speech groups",
                  f"max_gap={speech_group_max_gap_sec}s", time.monotonic() - t0)
        groups = _form_speech_groups(all_turns, speech_group_max_gap_sec)
        log.info("  Speech groups: %d groups from %d turns",
                 len(groups), len(all_turns))

        if log.isEnabledFor(logging.DEBUG):
            for g in groups[:20]:
                log.debug(
                    "    group=%3d  [%7.1fs → %7.1fs]  dur=%6.1fs  "
                    "speech=%6.1fs  turns=%d",
                    g["group_idx"], g["start"], g["end"],
                    g["duration"], g["speech_sec"], g["num_turns"],
                )
            if len(groups) > 20:
                log.debug("    ... (%d more groups)", len(groups) - 20)

        # ── compute dynamic chunks ──
        _progress("chunking: partitioning groups",
                  f"target={dynamic_speech_chunk_secs}s", time.monotonic() - t0)
        dyn_chunks = _make_dynamic_chunks(groups, dynamic_speech_chunk_secs)
        log.info("  Dynamic chunks: %d", len(dyn_chunks))

        if log.isEnabledFor(logging.DEBUG):
            log.debug("  Chunk table:")
            for c in dyn_chunks:
                log.debug(
                    "    dyn=%2d  [%7.1fs → %7.1fs]  speech=%6.1fs  "
                    "groups=%2d  turns=%3d",
                    c["dyn_chunk_idx"],
                    c["fetch_start_sec"], c["fetch_end_sec"],
                    c["speech_sec"], c["num_groups"], c["num_turns"],
                )

        # ── verify every VAD turn is covered ──
        _verify_coverage(all_turns, dyn_chunks)

        # ── write dynamic_segments_{n}.json files to S3 ──
        _progress("writing dynamic_segments files",
                  f"{len(dyn_chunks)} files", time.monotonic() - t0)
        _write_dynamic_segments(dyn_chunks, output_bucket, output_prefix)

        # ── shape output iterators ──
        shaped = _shape_output(
            dyn_chunks=dyn_chunks,
            audio_bucket=audio_bucket,
            audio_key=audio_key,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            model_cfg=model_cfg,
            audio_cfg=audio_cfg,
            vad_cfg=vad_cfg,
            embed_cfg=embed_cfg,
        )
        log.info("  Output iterators: %d", len(shaped))

        # ── build output event ──
        output = {
            **event,
            "data": {
                **data,
                "vad_cfg": {
                    **vad_cfg,
                    "elapsed_sec_vad": elapsed_sec_vad,
                },
                "iterators": shaped,
            },
        }

        elapsed = round(time.monotonic() - t0, 3)
        log.info("StateDynamicChunkingBySpeechTurns output keys = %s",
                 sorted(output.keys()))
        _progress(
            "DONE",
            f"{len(shaped)} dynamic chunks  elapsed={elapsed:.2f}s",
            elapsed,
        )
        return output

    except Exception as exc:
        log.error(
            "StateDynamicChunkingBySpeechTurns failed — %s: %s",
            type(exc).__name__, exc
        )
        traceback.print_exc()
        raise
