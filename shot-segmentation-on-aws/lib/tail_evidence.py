# lib/tail_evidence.py
"""Programme-tail evidence: what sits after the last long black?

Deliveries routinely carry material after the programme proper — textless
elements (clean copies of in-programme shots appended for localisation),
distributor end boards, or further programme parts separated by long blacks.
A consumer reading `actual_duration_sec` as "programme runtime" overstates it
by the whole tail (measured 14.7%-34.6% of file on the four corpus assets
where a tail exists), and a consumer localising a delivery needs to know
whether the tail IS the textless block.

The evidence published here is picture-only and built from data already in
hand (shot types, timestamps, centroid embeddings — no new I/O):

  separator   the last-resort structural marker: a maximal RUN of consecutive
              black shots spanning at least PROGRAMME_TAIL_MIN_BLACK_SEC,
              starting after half the runtime, followed by any content at
              all. A run, not a single shot, because segmentation legitimately
              splits one physical black on luma flicker or a dark end board:
              drama_c's 33 s separator arrived as FOUR consecutive black
              shots (0.9+9.9+2.3+20.0 s — ground-truthed against the
              delivery's own 00:49:50 textless mark, frame-exact at the run's
              end), and the factual_d episode's 68 s black run opens with
              a near-black distributor card. A single long black is a run of
              one, so every previously-firing asset selects the same
              separator. Among candidates, the one with the MOST content
              after it wins (a multi-part master has several inter-part
              blacks; the tail of interest is the outermost).

              There is deliberately NO minimum-content floor (the 60 s floor
              was removed 2026-08-23): it existed to suppress end-board-only
              tails (measured 12-21 s), but it also suppressed factual_d
              episode's real 55.6 s textless block — a floor between 21 and
              55 would be fitted to n=5, and an end-board tail already reads
              honestly in the numbers (seconds-long tail_content_sec,
              match_fraction ~0). Evidence, not verdicts.

  match stats how the tail's content relates to the programme BODY before the
              separator run: `match_fraction` is the share of tail content
              shots whose nearest body shot sits within
              PROGRAMME_TAIL_MATCH_THRESHOLD (cosine, same space as
              recurrence); `match_duration_fraction` is the share of those
              matches whose durations agree within
              PROGRAMME_TAIL_DURATION_TOLERANCE_SEC — a textless element is
              the SAME CUT as its source, so its duration must agree almost
              exactly (measured frame-exact at the median), while a same-set
              coincidence match will not; `match_monotone` is the share of
              consecutive matched pairs whose body sources advance in order;
              and `match_fraction_duration_corroborated` widens the distance
              band to PROGRAMME_TAIL_CORROBORATED_BAND but demands the
              duration agreement — heavy burned graphics (a credit ROLL, a
              full-frame title) push a clean copy's distance past the strict
              threshold where subtitles do not (factual_d: the credit-roll
              background matched its clean tail copy at 0.058 with duration
              agreeing to 0.12 s; the strict threshold was calibrated on
              subtitle-only sources at <=0.05, n=2 assets), and the duration
              test is what keeps the wider band honest.

The corpus, all visually verified or ground-truthed (2026-08-23, n=6 tails):

              strict match / dur-agree / monotone / corroborated
  cooking_a      0.96 / 0.96 / 0.98 /  --    textless block
  sports_doc_a      0.88 / 0.95 / 0.98 /  --    textless block (body carries
                                            burned subtitles; tail is the
                                            SAME shots clean, 22 min)
  drama_c    0.33 / 1.00 / 1.00 / 0.33+  MIXED tail: a textless block
                                            partially sourced from THIS
                                            episode (28 frame-exact ordered
                                            copies) + textless material from
                                            sibling episodes of the serial
  factual_d  0.33 / 1.00 /  --  / 0.67   textless block, 55.6 s — only
                                            the graphics-bearing shots,
                                            appended clean
  title_card_a      0.01 / 0.00 /  --  /  ~0    part boundary; its ONE accidental
                                            embedding match misses duration
                                            by 2.68 s
  paytv_a  0.00 /  --  /  --  /  ~0    part boundary; programme continues

The reading that separates the kinds: a textless block is matched AND
frame-exact AND ordered — and when match_fraction alone is ambiguous (a
partial or graphics-heavy block), the DURATION-anchored stats do the
separating: drama_d's promo tail read fraction 0.50 with dur-agree 0.00
(similar-looking, wrong cuts — not textless), while drama_c read fraction
0.33 with dur-agree 1.00 (fewer matches, every one the same cut — textless).
Why not audio: the textless block carries full M&E and is
audio-indistinguishable from programme (see test/harness/audio_probe.py) —
the moat is already a published black run.

No verdict field, deliberately: with n=6 the thresholds a verdict needs would
be fitted, not structural. The numbers themselves are structural facts.
"""
from __future__ import annotations

import logging

import numpy as np

from lib.input_schema import (
    PROGRAMME_TAIL_CORROBORATED_BAND,
    PROGRAMME_TAIL_DURATION_TOLERANCE_SEC,
    PROGRAMME_TAIL_MATCH_THRESHOLD,
    PROGRAMME_TAIL_MIN_BLACK_SEC,
)

log = logging.getLogger(__name__)


def analyse_programme_tail(shots: list[dict]) -> dict | None:
    """Evidence object for the shots.json envelope, or None when the
    programme runs to the end of the file.

    shots  the final merged shot list, each dict carrying shot_id,
           shot_type, start_time/end_time and centroid_embedding.
    """
    if not shots:
        return None
    ordered = sorted(shots, key=lambda s: s["start_time"])
    duration = float(ordered[-1]["end_time"])

    best = None   # (separator_run, tail_content_shots, tail_content_sec)
    for run in _black_runs(ordered):
        if run[-1]["end_time"] - run[0]["start_time"] \
                < PROGRAMME_TAIL_MIN_BLACK_SEC:
            continue
        if run[0]["start_time"] <= 0.5 * duration:
            continue
        after = [x for x in ordered
                 if x["start_time"] >= run[-1]["end_time"]
                 and x.get("shot_type") == "content"]
        csec = sum(x["end_time"] - x["start_time"] for x in after)
        if after and (best is None or csec > best[2]):
            best = (run, after, csec)
    if best is None:
        return None
    run, tail, tail_sec = best

    body = [x for x in ordered
            if x.get("shot_type") == "content"
            and x["end_time"] <= run[0]["start_time"]]
    evidence = {
        "separator_shot_id": run[0]["shot_id"],
        "separator_start_time": round(float(run[0]["start_time"]), 3),
        "separator_end_time": round(float(run[-1]["end_time"]), 3),
        "separator_black_sec": round(
            float(run[-1]["end_time"] - run[0]["start_time"]), 3),
        "tail_content_sec": round(float(tail_sec), 3),
        "match_fraction": None,
        "match_duration_fraction": None,
        "match_monotone": None,
        "match_fraction_duration_corroborated": None,
    }
    if body and tail:
        evidence.update(_match_stats(body, tail))
    log.info(
        "Programme tail: separator run %d @ %.1f-%.1fs, %.0fs content "
        "after, match_fraction=%s match_duration_fraction=%s "
        "match_monotone=%s corroborated=%s",
        evidence["separator_shot_id"], evidence["separator_start_time"],
        evidence["separator_end_time"], evidence["tail_content_sec"],
        evidence["match_fraction"], evidence["match_duration_fraction"],
        evidence["match_monotone"],
        evidence["match_fraction_duration_corroborated"])
    return evidence


def _black_runs(ordered: list[dict]) -> list[list[dict]]:
    """Maximal runs of list-adjacent black shots. Segmentation may split one
    physical black on luma flicker or a dark end board; the run is the
    structural unit, and a lone black shot is a run of one. Grouping is by
    LIST position, not by timestamps — adjacency in the merged shot list is
    the ground truth."""
    runs: list[list[dict]] = []
    previous_index = None
    for index, shot in enumerate(ordered):
        if shot.get("shot_type") != "black":
            continue
        if previous_index is not None and index == previous_index + 1:
            runs[-1].append(shot)
        else:
            runs.append([shot])
        previous_index = index
    return runs


def _match_stats(body: list[dict], tail: list[dict]) -> dict:
    """The four tail-to-body statistics over one nearest-neighbour pass."""
    B = _unit(np.array([x["centroid_embedding"] for x in body],
                       dtype=np.float32))
    T = _unit(np.array([x["centroid_embedding"] for x in tail],
                       dtype=np.float32))
    D = np.clip(1.0 - T @ B.T, 0.0, 2.0)
    nearest = D.min(axis=1)
    src = D.argmin(axis=1)
    t_dur = np.array([x["end_time"] - x["start_time"] for x in tail])
    b_dur = np.array([x["end_time"] - x["start_time"] for x in body])
    duration_agrees = (np.abs(t_dur - b_dur[src])
                       <= PROGRAMME_TAIL_DURATION_TOLERANCE_SEC)

    matched = nearest <= PROGRAMME_TAIL_MATCH_THRESHOLD
    corroborated = ((nearest <= PROGRAMME_TAIL_CORROBORATED_BAND)
                    & duration_agrees)
    stats = {
        "match_fraction": round(float(matched.mean()), 4),
        "match_duration_fraction": None,
        "match_monotone": None,
        "match_fraction_duration_corroborated": round(
            float(corroborated.mean()), 4),
    }
    if matched.any():
        stats["match_duration_fraction"] = round(
            float(duration_agrees[matched].mean()), 4)
    if int(matched.sum()) >= 2:
        stats["match_monotone"] = round(
            float(np.mean(np.diff(src[matched]) > 0)), 4)
    return stats


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
