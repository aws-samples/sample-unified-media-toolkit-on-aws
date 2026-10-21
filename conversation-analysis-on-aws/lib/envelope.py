"""
The transcript envelope (README.md §2): the value substituted for
`{{transcript}}` — self-describing WebVTT generated from the normalized
sentence list, one cue per sentence, cue id = sentence id, speaker as a
`<v>` voice tag.

The chunk metadata that v5 passed as four separate template variables
(chunk_start_time / chunk_end_time / total_duration / transcript_duration)
rides IN-BAND in a NOTE block, so a prompt needs only `{{transcript}}` and
the prompt author sees everything the model sees (template + values =
rendered bytes keeps the audit story).

Single-chunk runs — the common case — get a one-line NOTE (program
duration only): zero chunking complexity leaks into short content.

The NOTE deliberately does NOT carry the previous chunk's closing topic:
chunks ship as N records in ONE run_inference submit and run in parallel.
"""
from __future__ import annotations

from lib.transcript import seconds_to_ts

# VTT overhead per cue beside the text: the id line, the timestamp line
# (29 chars), the voice tag and newlines — ~36 chars. 3.5 chars/token is
# deliberately conservative for English prose. The CHUNKER prices cues with
# this same function, so the plan and the rendered envelope cannot drift.
_CUE_OVERHEAD_CHARS = 36
_CHARS_PER_TOKEN = 3.5


def estimate_cue_tokens(sentence: dict) -> float:
    return (len(sentence["text"]) + _CUE_OVERHEAD_CHARS) / _CHARS_PER_TOKEN


def build_envelope(sentences: list[dict], chunk: dict, *,
                   duration_sec: float, total_chunks: int) -> str:
    """The WebVTT string for one chunk (or the whole program when
    total_chunks == 1). `chunk` is a chunker.plan_chunks() entry."""
    out = ["WEBVTT", ""]
    out.extend(_note_lines(sentences, chunk, duration_sec, total_chunks))
    out.append("")

    ctx = chunk.get("context_start_pos")
    first = ctx if ctx is not None else chunk["start_pos"]
    for pos in range(first, chunk["end_pos"] + 1):
        s = sentences[pos]
        out.append(str(s["id"]))
        out.append(f"{seconds_to_ts(s['start'])} --> "
                   f"{seconds_to_ts(s['end'])}")
        out.append(f"<v {s['speaker_id']}>{s['text']}")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def _note_lines(sentences, chunk, duration_sec, total_chunks) -> list[str]:
    lines = ["NOTE",
             f"program duration {seconds_to_ts(duration_sec)}"]
    if total_chunks <= 1:
        return lines
    start_s = sentences[chunk["start_pos"]]
    end_s = sentences[chunk["end_pos"]]
    lines.append(
        f"chunk {chunk['index'] + 1} of {total_chunks} | coverage cues "
        f"{start_s['id']}-{end_s['id']} "
        f"({seconds_to_ts(start_s['start'])}-{seconds_to_ts(end_s['end'])})")
    ctx = chunk.get("context_start_pos")
    if ctx is not None:
        ctx_first = sentences[ctx]["id"]
        ctx_last = sentences[chunk["start_pos"] - 1]["id"]
        lines.append(
            f"cues {ctx_first}-{ctx_last} are CONTEXT from the previous "
            f"chunk - do not segment them; your coverage begins at cue "
            f"{start_s['id']}")
    return lines
