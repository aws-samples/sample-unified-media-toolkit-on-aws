#!/usr/bin/env python3
"""Programme-end detection from picture alone: the BLACK MOAT.

Deliveries separate post-programme material (textless elements, end boards,
distributor cards) from the programme with a long black. That black is already
published as a shot, so the boundary costs no new I/O and no audio:

    black >= MIN_BLACK_SEC, starting after HALF the runtime,
    followed by >= MIN_AFTER_SEC of content

Measured across the 13-asset studio batch — fires on exactly one asset, the
one whose filename carries the '_dualpurpose_' marker:

  cooking_a          black#525@30:02/78s -> 50 content shots, 138s after   FIRES
  drama_b           black#618@47:11/58s ->  4 content shots,  12s after
  studio_a2          black#425@26:04/33s ->  3 content shots,  17s after
  studio_a3          black#316@23:59/55s ->  2 content shots,  20s after
  studio_a4          black#370@28:06/54s ->  2 content shots,  21s after
  arts_doc_a   black#440@52:46/43s ->  0 content shots,   0s after
  film_b / studio_b1-103 / studio_a1 / factual_c / quiz_a — no long tail black

The 12-21s tails on drama_b and studio_a are distributor end boards, an order of
magnitude below the 138s textless block, so MIN_AFTER_SEC separates them
cleanly. arts_doc_a is '_texted_' and correctly has nothing after its
runout. See audio_probe.py for why the silence that coincides with the moat is
NOT a better signal.

Run from a scratch dir holding <run>.shots.json (see setup_workdir.sh):
  python moat.py *.shots.json
"""
import json
import os
import sys

MIN_BLACK_SEC = 30.0
MIN_AFTER_SEC = 60.0

mmss = lambda s: f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def moat(shots, duration):
    """Return (black_shot, content_shots_after, content_sec_after) or None."""
    best = None
    for i, s in enumerate(shots):
        if s.get("shot_type") != "black":
            continue
        if s["end_time"] - s["start_time"] < MIN_BLACK_SEC:
            continue
        if s["start_time"] <= 0.5 * duration:
            continue
        after = [x for x in shots[i + 1:] if x.get("shot_type") != "black"]
        csec = sum(x["end_time"] - x["start_time"] for x in after)
        if csec >= MIN_AFTER_SEC and (best is None or csec > best[2]):
            best = (s, after, csec)
    return best


def main(paths):
    for p in paths:
        name = os.path.basename(p).split(".")[0]
        doc = json.load(open(p))
        shots = doc["shots"] if isinstance(doc, dict) else doc
        duration = shots[-1]["end_time"]
        hit = moat(shots, duration)
        if hit is None:
            print(f"{name:18s} dur {mmss(duration)}   no moat — "
                  f"programme runs to end of file")
            continue
        blk, after, csec = hit
        tail = duration - blk["start_time"]
        print(f"{name:18s} dur {mmss(duration)}   "
              f"programme ends {mmss(blk['start_time'])} "
              f"(black#{blk['shot_id']}, {blk['end_time'] - blk['start_time']:.0f}s) "
              f"-> {len(after)} content shots / {csec:.0f}s after; "
              f"tail is {tail:.0f}s = {tail / duration * 100:.1f}% of file "
              f"(the runtime overstatement, head leader not counted)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
