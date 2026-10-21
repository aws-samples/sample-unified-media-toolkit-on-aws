"""
Model-output parsing and coverage validation (README.md §3-§4).

The model returns JSONL: a header line ({overall_content_type,
top_3_genres}) then one line per segment ({start_id, end_id, ...}).
JSONL is the recovery format: truncation loses at most the final line, a
malformed line damages only itself, and the §3 coverage check names
exactly which cue ids remain uncovered — the retryable unit.

parse_response() reads every line STRICTLY first, and only a line that
already failed climbs the ladder in the COPIED parse_partial_output
module (source of truth: bedrock-inference-on-aws; adopted here
2026-08-26, the module's second copy): the escape pass for a quotation
mark the model wrote inside a value without escaping it (the one
malformation measured in the wild — shot-metadata-enrichment lost 10
shots to it), then a gated prefix salvage for truncation or mid-line
damage. The gate is the MINIMUM a line is unplaceable without (user
decision 2026-08-26): a segment needs its cue ids, a header its
content type — a recovered partial that merely lost optional fields is
kept, and normalize_segments defaults them exactly as it always has. The
salvage cannot fabricate: every value it admits parsed strictly from the
model's own bytes, and each recovery is COUNTED per rung in
`line_repairs`, so a prompt regression shows up as a rising number
instead of hiding behind a forgiving parser.
"""
from __future__ import annotations

import json

from lib.parse_partial_output import parse_jsonl_line

_SEGMENT_FIELDS = ("segment_topic", "reasoning", "named_entities",
                   "key_events", "sentiment", "tone", "conversation_type")

# The salvage gate, per line SHAPE (either alternative admits): a header
# without its content type or a segment without its cue ids cannot be
# placed by validate_coverage/merge — everything else degrades by design
# in normalize_segments.
REQUIRED_KEYS = ({"overall_content_type"}, {"start_id", "end_id"})


def parse_response(text: str) -> dict:
    """-> {"header": dict | None, "segments": [dict], "bad_lines":
    [{"line_no", "error", "text"}], "line_repairs": {tier: count}}.
    Never raises on content."""
    header = None
    segments: list[dict] = []
    bad: list[dict] = []
    line_repairs: dict = {}
    for line_no, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        obj, tier = parse_jsonl_line(line, REQUIRED_KEYS)
        if obj is None:
            bad.append({"line_no": line_no, "error": tier,
                        "text": line[:200]})
            continue
        if not isinstance(obj, dict):
            bad.append({"line_no": line_no,
                        "error": "not a JSON object", "text": line[:200]})
            continue
        if tier != "strict":
            # Counted only for a line that yielded a usable object — a
            # salvage whose output is then rejected below still lands in
            # bad_lines, never in both counters (2026-08-26 review).
            line_repairs[tier] = line_repairs.get(tier, 0) + 1
        if "overall_content_type" in obj and header is None:
            header = obj
        elif "start_id" in obj and "end_id" in obj:
            segments.append(obj)
        else:
            bad.append({"line_no": line_no,
                        "error": "neither header nor segment",
                        "text": line[:200]})
    return {"header": header, "segments": segments, "bad_lines": bad,
            "line_repairs": line_repairs}


def validate_coverage(segments: list[dict], coverage_ids: list[int]) -> dict:
    """
    §3: segments are contiguous ascending id ranges that TILE coverage_ids.
    -> {"ok", "gaps": [[a, b]], "problems": [str]} — `gaps` are uncovered
    id runs (the retryable unit); `problems` are structural defects
    (overlap, unknown id, inverted range) that a resubmit of the whole
    chunk, not a tail, must fix.
    """
    problems: list[str] = []
    known = set(coverage_ids)
    covered: set[int] = set()
    for seg in segments:
        try:
            a, b = int(seg["start_id"]), int(seg["end_id"])
        except (TypeError, ValueError, KeyError):
            problems.append(f"segment has non-integer ids: "
                            f"{seg.get('start_id')!r}..{seg.get('end_id')!r}")
            continue
        if a > b:
            problems.append(f"segment range inverted: {a}..{b}")
            continue
        ids = set(range(a, b + 1))
        unknown = ids - known
        if unknown:
            problems.append(
                f"segment {a}..{b} names {len(unknown)} id(s) outside "
                f"coverage (e.g. {sorted(unknown)[:3]})")
        overlap = ids & covered
        if overlap:
            problems.append(
                f"segment {a}..{b} overlaps {len(overlap)} already-covered "
                f"id(s) (e.g. {sorted(overlap)[:3]})")
        covered |= ids & known
    gaps = _runs(sorted(known - covered))
    return {"ok": not problems and not gaps, "gaps": gaps,
            "problems": problems}


def normalize_segments(segments: list[dict]) -> list[dict]:
    """Sorted by start_id, defaulted optional fields, list fields coerced —
    the ONE shape merge.py consumes.

    Fields beyond the core set PASS THROUGH verbatim: a NAMED prompt
    version (the workflows' `prompt_version` knob — e.g. `default_news`,
    which adds news_form/interaction/sig_out/correspondent/slug) may
    extend the segment schema, and extending a prompt must never require
    a workflow code change — dropping declared prompt output here is data
    loss, not parser tolerance. The core fields keep their coercions, so
    the default prompt's output is byte-identical to before pass-through
    existed."""
    out = []
    for seg in segments:
        s = dict(seg)                     # extras pass through
        s["start_id"] = int(seg["start_id"])
        s["end_id"] = int(seg["end_id"])
        s["continues_previous"] = bool(seg.get("continues_previous"))
        for f in _SEGMENT_FIELDS:
            v = seg.get(f)
            if f in ("named_entities", "key_events"):
                s[f] = [str(x) for x in v] if isinstance(v, list) else []
            else:
                s[f] = str(v) if v is not None else ""
        out.append(s)
    return sorted(out, key=lambda s: s["start_id"])


def _runs(ids: list[int]) -> list[list[int]]:
    runs = []
    for x in ids:
        if runs and x == runs[-1][1] + 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    return runs
