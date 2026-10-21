# lib/burst_evidence.py
"""Burst-run evidence: contiguous runs of very short shots (9.15).

Bursty content — jingles, montages, promo stings — correctly segments into
many sub-second shots ("technically this is correct"), and the cost lands
DOWNSTREAM: a consumer inferring metadata per shot burns tokens on dozens of
near-useless per-card results. Measured on jingle.mp4: 58 shots in 35 s,
94.8% of them inside three runs (longest 35), versus <= 8.5% across the
30-asset corpus — an order-of-magnitude gap between "the asset IS a jingle"
and "the asset contains montage moments".

What is published: the runs as INTERVALS in the shots.json envelope —

    "burst_runs": [{"start_shot_id": 3, "end_shot_id": 37}, ...]

and nothing else, on the same trim discipline as recurrence.json: a burst
run is contiguous by definition, so an interval carries its whole identity;
member count, duration, and the run's character (a jingle's cards are
internally FROZEN — median frame_coherence 0.994, frame_motion_mean 0.005 —
where an action montage carries high intra-shot motion) are all derivable by
joining the id range against the shot list. What the field genuinely adds is
the one thing a join cannot: a SHARED DEFINITION of "this is a burst", the
same role `position` plays for recurrence members.

Shot TYPE is deliberately ignored: jingles separate their cards with black
micro-shots, and a content-only rule breaks every run at each separator
(jingle.mp4's longest run reads 8 content-only, 35 type-blind).

No verdict field ("jingle", "montage") on purpose: the thresholds are fitted
on a handful of assets and belong in this docstring and the contract notes,
not in the payload. The consumer decision this exists for: fold each run
into ONE inference (sample a few frames across it) instead of inferring per
member. Producer-side merging of the shots themselves was considered and
rejected — the boundaries are real, and recurrence and face-tracking
legitimately want them.
"""
from __future__ import annotations

import logging

from lib.input_schema import BURST_MAX_SHOT_SEC, BURST_MIN_RUN

log = logging.getLogger(__name__)


def detect_burst_runs(shots: list[dict]) -> list[dict]:
    """Interval list for the shots.json envelope; [] when nothing qualifies.

    shots  the final merged shot list (shot_id, start_time, end_time; any
           shot_type — see the module docstring for why type is ignored).
    """
    ordered = sorted(shots, key=lambda s: s["start_time"])
    runs: list[dict] = []
    cur: list[dict] = []

    def flush():
        if len(cur) >= BURST_MIN_RUN:
            runs.append({"start_shot_id": cur[0]["shot_id"],
                         "end_shot_id": cur[-1]["shot_id"]})

    for s in ordered:
        if (s["end_time"] - s["start_time"]) < BURST_MAX_SHOT_SEC:
            cur.append(s)
        else:
            flush()
            cur = []
    flush()

    if runs:
        log.info("Burst runs: %d  %s", len(runs),
                 [(r["start_shot_id"], r["end_shot_id"]) for r in runs])
    return runs
