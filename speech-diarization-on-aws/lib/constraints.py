# lib/constraints.py
"""Cannot-link constraints from cross-talk evidence (README.md
Design record P3).

lib/segmentation.extract_overlap_evidence() proves, per powerset
window, that the voice in solo span A and the voice in solo span B are
DIFFERENT people (they overlapped in between). This module turns those
time spans into pairs of HIGH-sentence indices for the clusterer:
sentences i and j may never share a cluster — regardless of how
similar their embeddings are. This is the only signal we have that can
separate voices the embedding space cannot (measured: same-family
voices at chance AUC; compact interview mega-clusters).

Called by: states/global_speakers_clustering (mapping),
           lib.cluster_pipeline (enforcement).
"""
from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right

log = logging.getLogger(__name__)

# A span must be MOSTLY inside the matched sentence: the sentence with
# the best overlap also has to cover >= this fraction of the span,
# else the mapping is ambiguous and the record is dropped (a wrong
# cannot-link forces a split — recoverable downstream per the
# prefer-split asymmetry, but there is no reason to spend that budget
# on ambiguous evidence).
MIN_SPAN_COVER = 0.5


def _best_sentence(span, sentences, order, starts, max_dur):
    """Index of the sentence covering most of [t0, t1], or None.

    `order`/`starts`/`max_dur` are the caller's precomputed
    start-sorted index: only sentences whose start lies in
    [t0 - max_dur, t1) can overlap the span, so the scan is
    bisect-bounded instead of O(all sentences) per span — evidence is
    collected from EVERY window since fa875eb, so the naive product
    was O(records x sentences) on the 1-core clustering Lambda."""
    t0, t1 = float(span[0]), float(span[1])
    if t1 <= t0:
        return None
    best_i, best_ov = None, 0.0
    lo = bisect_left(starts, t0 - max_dur)
    hi = bisect_right(starts, t1)
    for k in range(lo, hi):
        i = order[k]
        s = sentences[i]
        ov = min(t1, float(s["end"])) - max(t0, float(s["start"]))
        if ov > best_ov:
            best_ov, best_i = ov, i
    if best_i is None or best_ov / (t1 - t0) < MIN_SPAN_COVER:
        return None
    return best_i


def resolve_evidence(
    evidence:  list[dict],
    sentences: list[dict],
) -> list[dict]:
    """
    Resolve cross-talk evidence records against the sentence list.

    Returns [{"i": <sent idx>, "j": <sent idx>, "overlap": [t0, t1]}]
    for every record where BOTH solo spans map to a sentence with
    >= MIN_SPAN_COVER coverage and the two sentences differ. A record
    whose spans resolve to the SAME sentence is dropped: that sentence
    provably contains two voices (contaminated) — nothing to constrain.
    """
    order = sorted(range(len(sentences)),
                   key=lambda k: float(sentences[k]["start"]))
    starts = [float(sentences[k]["start"]) for k in order]
    max_dur = max((float(s["end"]) - float(s["start"])
                   for s in sentences), default=0.0)
    out = []
    n_same = n_unmapped = 0
    for rec in evidence or []:
        ia = _best_sentence(rec.get("a") or (0, 0), sentences,
                            order, starts, max_dur)
        ib = _best_sentence(rec.get("b") or (0, 0), sentences,
                            order, starts, max_dur)
        if ia is None or ib is None:
            n_unmapped += 1
            continue
        if ia == ib:
            n_same += 1
            continue
        out.append({"i": min(ia, ib), "j": max(ia, ib),
                    "overlap": [float(rec["overlap"][0]),
                                float(rec["overlap"][1])]
                    if rec.get("overlap") else None})
    if evidence:
        log.info(
            "cross-talk evidence: %d records -> %d resolved  "
            "(%d unmapped, %d same-sentence)",
            len(evidence), len(out), n_unmapped, n_same)
    return out


def evidence_to_cannot_links(
    records: list[dict],
) -> list[tuple[int, int]]:
    """Deduplicated (i, j) cannot-link sentence pairs (i < j) from
    ALREADY-RESOLVED records (resolve_evidence output) — the caller
    needs the records for overlap_segments too, so resolving happens
    exactly once."""
    return sorted({(r["i"], r["j"]) for r in records})


def overlap_segments(
    records:  list[dict],
    labels:   list,             # speaker name (or None) per sentence idx
) -> list[dict]:
    """
    Identity-resolved two-speaker segments for the overlap spans (P3
    overlap-aware output — the structural-miss half: our transcript
    emits ONE speaker per instant, so the second voice of every
    cross-talk moment is missed BY CONSTRUCTION; measured as half of
    AMI's miss). Emitted only when both participants resolved to real,
    distinct clusters. Adjacent/overlapping segments with the same
    speaker pair are merged.
    """
    segs = []
    for r in records:
        ov = r.get("overlap")
        if not ov or ov[1] <= ov[0]:
            continue
        sa = labels[r["i"]] if r["i"] < len(labels) else None
        sb = labels[r["j"]] if r["j"] < len(labels) else None
        if not sa or not sb or sa == sb or "UNKNOWN" in (sa, sb):
            continue
        segs.append({"start": round(ov[0], 3), "end": round(ov[1], 3),
                     "speakers": sorted((sa, sb))})
    segs.sort(key=lambda s: (s["start"], s["end"]))
    merged: list[dict] = []
    for s in segs:
        if (merged and s["speakers"] == merged[-1]["speakers"]
                and s["start"] <= merged[-1]["end"] + 0.1):
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(dict(s))
    return merged
