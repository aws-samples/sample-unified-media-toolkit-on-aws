"""
Merge per-chunk results into the final analysis (README.md §5-§6, §7e).

Everything here is PURE CODE over per-chunk data — deterministic, free,
auditable. No inference call anywhere (user decision 2026-08-17: header
disagreement is rolled up and deduped in code, never adjudicated by a
model).

- `continues_previous` stitch: chunk N+1's first segment saying true is
  merged into chunk N's last segment — id ranges unioned, the EARLIER
  topic/reasoning kept, entities/events unioned in first-seen order.
- Header reconciliation: coverage-duration-weighted vote; no strict
  majority -> the top-weighted value still leads, and
  `content_type_candidates` reports the deduped disagreement.
- id -> timestamp resolution: authoritative start/end from the source
  sentences, so timestamps never pass through a model.
"""
from __future__ import annotations


def merge_chunks(chunk_results: list[dict], sentences: list[dict],
                 duration_sec: float) -> dict:
    """
    chunk_results: per chunk, in chunk order:
        {"header": dict | None, "segments": [normalized], "gaps": [[a,b]]}
    -> the `analysis` output's analytical core (README.md §7e):
        {"overall_content_type", "content_type_candidates"?,
         "top_3_genres", "audio_segments": [...], "segments_gap": [...]}
    """
    if not chunk_results:
        raise ValueError("merge_chunks: no chunk results")
    by_id = {s["id"]: s for s in sentences}

    segments: list[dict] = []
    for chunk in chunk_results:
        for i, seg in enumerate(chunk["segments"]):
            if i == 0 and seg.get("continues_previous") and segments:
                _stitch(segments[-1], seg)
            else:
                segments.append(dict(seg))

    weights = _chunk_weights(chunk_results, by_id)
    header = _reconcile_header(chunk_results, weights)

    audio_segments = []
    for seg in segments:
        s = dict(seg)
        s.pop("continues_previous", None)
        s["start_time"] = _start_time(by_id, s["start_id"])
        s["end_time"] = _end_time(by_id, s["end_id"])
        audio_segments.append(s)

    out = dict(header)
    out["audio_segments"] = audio_segments
    out["segments_gap"] = sorted(
        (g for c in chunk_results for g in c.get("gaps", [])))
    return out


def annotate_overlapped_attribution(analysis: dict,
                                    sentences: list[dict],
                                    overlap_segments: list[dict]) -> int:
    """Mark discourse_facts whose source cue falls inside a cross-talk
    span involving the attributed speaker: the fact KEEPS its primary
    attribution and gains `attribution_overlapped_with` (the other
    voice(s)) — a recorded ambiguity, never a re-attribution. The
    producer states the spans' identity precision (81%, zh 56-69%), so
    this is SUGGESTIVE evidence; consumers (compliance pages) render
    it as uncertainty. Returns the number of facts annotated."""
    if not overlap_segments:
        return 0
    by_id = {s["id"]: s for s in sentences}
    n = 0
    for seg in analysis.get("audio_segments") or []:
        for fact in seg.get("discourse_facts") or []:
            s = by_id.get(fact.get("cue_id"))
            if s is None:
                continue
            others = sorted({
                spk
                for ov in overlap_segments
                if (float(ov["start"]) < s["end"]
                    and s["start"] < float(ov["end"])
                    and s["speaker_id"] in (ov.get("speakers") or []))
                for spk in ov["speakers"] if spk != s["speaker_id"]})
            if others:
                fact["attribution_overlapped_with"] = others
                n += 1
    return n


def _stitch(prev: dict, cont: dict) -> None:
    """§5: the continuation extends the previous segment. Earlier topic and
    reasoning win (the earlier chunk saw the topic's beginning); list
    fields union in first-seen order."""
    prev["end_id"] = max(prev["end_id"], cont["end_id"])
    for f in ("named_entities", "key_events"):
        seen = set(prev[f])
        prev[f] = prev[f] + [x for x in cont[f] if x not in seen]


def _chunk_weights(chunk_results, by_id) -> list[float]:
    """A chunk's vote weight is its covered timeline duration."""
    weights = []
    for chunk in chunk_results:
        segs = chunk["segments"]
        if not segs:
            weights.append(0.0)
            continue
        lo = min(s["start_id"] for s in segs)
        hi = max(s["end_id"] for s in segs)
        start = by_id[lo]["start"] if lo in by_id else 0.0
        end = by_id[hi]["end"] if hi in by_id else 0.0
        weights.append(max(0.0, end - start))
    return weights


def _reconcile_header(chunk_results, weights) -> dict:
    votes: dict[str, float] = {}
    order: list[str] = []                      # first-seen, for stable ties
    genre_votes: dict[str, float] = {}
    genre_order: list[str] = []
    total = 0.0
    for chunk, w in zip(chunk_results, weights):
        h = chunk.get("header") or {}
        ct = h.get("overall_content_type")
        if ct:
            if ct not in votes:
                order.append(ct)
            votes[ct] = votes.get(ct, 0.0) + w
            total += w
        for g in h.get("top_3_genres") or []:
            if g not in genre_votes:
                genre_order.append(g)
            genre_votes[g] = genre_votes.get(g, 0.0) + w

    if not votes:
        return {"overall_content_type": None, "top_3_genres": []}
    ranked = sorted(order, key=lambda c: -votes[c])
    header: dict = {"overall_content_type": ranked[0]}
    if len(ranked) > 1 and votes[ranked[0]] <= total / 2:
        # No strict majority: report the disagreement, weight-ordered.
        header["content_type_candidates"] = ranked
    header["top_3_genres"] = sorted(
        genre_order, key=lambda g: -genre_votes[g])[:3]
    # Header fields beyond the core set PASS THROUGH by weighted plurality:
    # a NAMED prompt version may extend the header (default_news adds
    # program_name), and extending a prompt must never require a workflow
    # change — the same rule normalize_segments applies to segment fields.
    # Scalars only; the core keys keep their dedicated reconciliation, and
    # content_type_candidates is COMPUTED above — a prompt-emitted field of
    # that name must not overwrite the reconciler's disagreement report.
    core = {"overall_content_type", "top_3_genres",
            "content_type_candidates"}
    extra_votes: dict = {}
    for chunk, w in zip(chunk_results, weights):
        for k, val in (chunk.get("header") or {}).items():
            if k in core or val is None or isinstance(val, (dict, list)):
                continue
            extra_votes.setdefault(k, {})
            extra_votes[k][val] = extra_votes[k].get(val, 0.0) + w
    for k, vals in extra_votes.items():
        header[k] = max(vals.items(), key=lambda kv: kv[1])[0]
    return header


def _start_time(by_id, sid):
    from lib.transcript import seconds_to_ts
    return seconds_to_ts(by_id[sid]["start"]) if sid in by_id else None


def _end_time(by_id, sid):
    from lib.transcript import seconds_to_ts
    return seconds_to_ts(by_id[sid]["end"]) if sid in by_id else None
