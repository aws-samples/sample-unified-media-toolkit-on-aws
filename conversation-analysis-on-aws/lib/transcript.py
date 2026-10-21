"""
Transcript loading and normalization (README.md §7a).

One entry point, two accepted formats, one internal shape:

    load_transcript(text) -> {
        "sentences": [{"id": int, "start": float, "end": float,
                       "speaker_id": str, "text": str}, ...],
        "duration_sec": float,          # program duration when known, else
                                        # the last cue's end
        "source_format": "diarization-json" | "webvtt",
    }

- speech-diarization `final_transcript.json` is PREFERRED — lossless for
  every chunking signal (native global ids, true sentence boundaries,
  per-sentence speakers).
- Generic WebVTT is the compatibility on-ramp: cue identifiers are used
  when present (and integers), else ids are sequential positions; speakers
  come from `<v NAME>` voice tags, else the single "UNKNOWN" speaker
  (which disables speaker-change cuts — a documented, graceful
  degradation, README.md §7a).

Detection is by CONTENT, not extension: an S3 key tells you what someone
named a file, not what is in it. `transcript_format` ("diarization-json" |
"webvtt") overrides the sniff when the caller knows better.
"""
from __future__ import annotations

import json
import re

# Cue timestamp: HH:MM:SS.mmm or MM:SS.mmm (WebVTT allows both).
_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3})"
_CUE_TIMING_RE = re.compile(rf"^\s*{_TS}\s+-->\s+{_TS}")
_VOICE_RE = re.compile(r"^<v(?:\.[^ >]*)?\s+([^>]+)>\s*")

UNKNOWN_SPEAKER = "UNKNOWN"


class TranscriptError(ValueError):
    """Unparseable or empty transcript; the message says what to supply."""


def load_transcript(text: str, transcript_format: str | None = None) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise TranscriptError(
            "transcript is empty — supply a speech-diarization "
            "final_transcript.json (preferred) or a WebVTT file")
    fmt = transcript_format or _sniff(text)
    if fmt == "diarization-json":
        return _from_diarization_json(text)
    if fmt == "webvtt":
        return _from_webvtt(text)
    raise TranscriptError(
        f"transcript_format must be 'diarization-json' or 'webvtt', "
        f"got {fmt!r}")


def _sniff(text: str) -> str:
    head = text.lstrip()[:16]
    if head.startswith("{"):
        return "diarization-json"
    if head.upper().startswith("WEBVTT"):
        return "webvtt"
    raise TranscriptError(
        "unrecognized transcript content: expected a JSON object "
        "(speech-diarization final_transcript.json) or a file starting "
        "with WEBVTT")


def _from_diarization_json(text: str) -> dict:
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise TranscriptError(f"transcript is not valid JSON: {exc}") from exc
    raw = doc.get("sentences")
    if not isinstance(raw, list) or not raw:
        raise TranscriptError(
            "JSON transcript has no 'sentences' list — expected the "
            "speech-diarization final_transcript.json shape")
    sentences = []
    for i, s in enumerate(raw):
        try:
            sentences.append({
                "id": int(s["id"]),
                "start": float(s["start"]),
                "end": float(s["end"]),
                "speaker_id": str(s.get("speaker_id") or UNKNOWN_SPEAKER),
                "text": str(s["text"]),
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptError(
                f"sentences[{i}] is missing or mistypes a required field "
                f"(id/start/end/text): {exc}") from exc
    duration = float(doc.get("audio_duration_sec") or 0.0) \
        or sentences[-1]["end"]
    return {"sentences": sentences, "duration_sec": duration,
            "source_format": "diarization-json",
            # cross-talk spans with both voices named (2026-09-18,
            # producer-stated identity precision 81%, zh 56-69% —
            # SUGGESTIVE: used only to annotate attribution ambiguity,
            # never to move a fact to another speaker)
            "overlap_segments": doc.get("overlap_segments") or []}


def _from_webvtt(text: str) -> dict:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sentences: list[dict] = []
    i = 0
    n = len(lines)
    # Header line (WEBVTT ...) then blocks separated by blank lines.
    while i < n and lines[i].strip():
        i += 1
    while i < n:
        # skip blank separators
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        block = []
        while i < n and lines[i].strip():
            block.append(lines[i])
            i += 1
        first = block[0].strip()
        if first.startswith(("NOTE", "STYLE", "REGION")):
            continue
        cue_id, timing_idx = None, None
        if _CUE_TIMING_RE.match(first):
            timing_idx = 0
        elif len(block) > 1 and _CUE_TIMING_RE.match(block[1].strip()):
            cue_id, timing_idx = first, 1
        else:
            continue                       # not a cue block
        m = _CUE_TIMING_RE.match(block[timing_idx].strip())
        start = _ts_to_seconds(*m.groups()[0:4])
        end = _ts_to_seconds(*m.groups()[4:8])
        body = " ".join(x.strip() for x in block[timing_idx + 1:]).strip()
        speaker = UNKNOWN_SPEAKER
        vm = _VOICE_RE.match(body)
        if vm:
            speaker = vm.group(1).strip()
            body = body[vm.end():]
        body = body.replace("</v>", "").strip()
        sentences.append({"id": cue_id, "start": start, "end": end,
                          "speaker_id": speaker, "text": body})
    if not sentences:
        raise TranscriptError("WebVTT transcript contains no cues")
    # Cue identifiers become ids only when EVERY cue has a unique integer
    # one; a partial or non-integer scheme falls back to positions for all —
    # a mixed scheme would make id ranges ambiguous.
    ids = [s["id"] for s in sentences]
    if all(isinstance(x, str) and x.isdigit() for x in ids) \
            and len({int(x) for x in ids}) == len(ids):
        for s in sentences:
            s["id"] = int(s["id"])
    else:
        for pos, s in enumerate(sentences):
            s["id"] = pos
    return {"sentences": sentences,
            "duration_sec": sentences[-1]["end"],
            "source_format": "webvtt",
            "overlap_segments": []}


def _ts_to_seconds(hh, mm, ss, mmm) -> float:
    return (int(hh or 0) * 3600 + int(mm) * 60 + int(ss)
            + int(mmm) / 1000.0)


def seconds_to_ts(seconds: float) -> str:
    """HH:MM:SS.mmm — the envelope's and the output's one timestamp format."""
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, mmm = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{mmm:03d}"
