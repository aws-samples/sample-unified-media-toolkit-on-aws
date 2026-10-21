# lib/powerset_guides.py
"""
Powerset guides — everything the segmentation model knows that plain
VAD discards, distilled into four JSON-shaped artifacts (user design
dialogue, 2026-09-25; corpus probe + offline rehearsal in
/tmp/gt/diar_probe/ validated every mechanism before this module):

  fences          in-speech speaker-change instants (a <0.3s pause
                  never splits VAD; wtpsplit is text-only; the 0.5s
                  slicer gap spans it — the fence outranks all three).
                  Probe: 1.5-9.3% of shipped sentences carry a flip.
  overlap_spans   sustained 2-speaker regions (pyannote's own pipeline
                  sets exclude_overlap=true for embeddings; we now can).
  tracks          window-local speaker labels stitched across the 9s
                  shared by adjacent windows into continuous local
                  speaker tracks. BIND-OR-BREAK: the best permutation
                  must beat the runner-up by a real margin on frames of
                  OBSERVED speakers only (permutations differing solely
                  on unseen labels are the same assignment) — a wrong
                  bind welds two people for the rest of the chunk, a
                  refused bind just ends a track.
  co_active_pairs track pairs proven simultaneous (different people
                  whatever the embeddings say) + the max-clique K floor
                  (floors only, NEVER ceilings — sequential speakers
                  are invisible to co-activity).

The clustering rehearsal verdict this module inherits (2026-09-25):
embedding CONTIGUOUS track segments and duration-averaging per track
rescued the reality_a mega-cluster (ARI -0.03 -> +0.40 on the production
clusterer); averaging per-sentence embedding vectors did NOT (collapsed
right back — embedding(concat) != mean(embedding(pieces))). So the
embedding state embeds track SEGMENTS; nothing pools sentence vectors.

Everything here is per FIXED VAD chunk; chunks share no audio, so
chains never cross them (a track ending at a chunk edge is a short
track, not an error).
"""
from __future__ import annotations

import logging
from itertools import permutations

import numpy as np

from lib.input_schema import (
    POWERSET_BIND_MARGIN,
    POWERSET_FENCE_SUPPORT,
    POWERSET_MIN_EVENT_SEC,
    POWERSET_MIN_SEG_SEC,
    POWERSET_MIN_TRACK_SEC,
    POWERSET_OVERLAP_THRESHOLD,
    POWERSET_TRACK_GAP_SEC,
)

log = logging.getLogger(__name__)

_SOLO = {1: 0, 2: 1, 3: 2}               # powerset class -> solo speaker
_PAIRS = {4: (0, 1), 5: (0, 2), 6: (1, 2)}


class PowersetCollector:
    """Accumulates per-window argmax classes + overlap probability from
    run_segmentation's inner loop (one append per window — the probs
    are already in memory there; this adds no inference)."""

    def __init__(self):
        self.starts: list[float] = []     # absolute window starts (sec)
        self.cls: list[np.ndarray] = []   # [F] int8 argmax class rows
        self.ovl: list[np.ndarray] = []   # [F] float32 P(any pair class)

    def append(self, window_start_abs: float, probs: np.ndarray) -> None:
        self.starts.append(float(window_start_abs))
        self.cls.append(probs.argmax(-1).astype(np.int8))
        self.ovl.append(probs[:, 4:7].sum(-1).astype(np.float32))


def _smooth(row: np.ndarray) -> np.ndarray:
    """Suppress single-frame flickers (a 17ms blip is model noise)."""
    out = row.copy()
    for i in range(1, len(row) - 1):
        if row[i] != row[i - 1] and row[i] != row[i + 1]:
            out[i] = row[i - 1]
    return out


def _window_flips(row, wstart, fdur, min_fr):
    """Sustained solo-speaker changes inside one window, absolute sec.
    Permutation-safe: 'the speaker changed here' needs no identity."""
    flips = []
    solo = [(k, _SOLO[c]) for k, c in enumerate(row) if c in _SOLO]
    for (k1, s1), (k2, s2) in zip(solo, solo[1:]):
        if s1 != s2 and (k2 - k1) * fdur <= 1.0:
            left = list(row[max(0, k1 - min_fr):k1 + 1])
            right = list(row[k2:k2 + min_fr])
            if left.count(row[k1]) >= min_fr - 1 \
                    and right.count(row[k2]) >= min_fr - 1:
                flips.append(wstart + (k1 + k2) / 2 * fdur)
    return flips


def build_guides(collector: PowersetCollector, window_sec: float,
                 step_sec: float,
                 chunk_start: float, chunk_end: float) -> dict:
    """Collector -> the powerset_{n}.json document (absolute times).
    Frame duration derives from the model's own frames-per-window."""
    if not collector.starts:
        return {"fences": [], "overlap_spans": [], "tracks": [],
                "co_active_pairs": [], "co_active_secs": [],
                "clique_floor": 0, "local_speaker_floor": 0}
    starts = np.asarray(collector.starts)
    cls = np.stack(collector.cls)
    ovl = np.stack(collector.ovl)
    W, F = cls.shape
    fdur = window_sec / F
    min_fr = max(2, int(round(POWERSET_MIN_EVENT_SEC / fdur)))
    cls = np.apply_along_axis(_smooth, 1, cls)

    # ── durable local-speaker floor ───────────────────────────────────
    # Within ONE window the powerset's 0/1/2 labels are distinct people
    # by model semantics — sequential presence counts, overlap not
    # required (the clique floor's stricter cousin; the co-active
    # shards it needs die at the track filter, which is how a proven
    # 3-speaker asset shipped K=2). Durability guards: a speaker must
    # be active >= POWERSET_LOCAL_FLOOR_MIN_SEC within the window
    # (flicker can't vote), and a count binds only when >=
    # POWERSET_LOCAL_FLOOR_MIN_WINDOWS windows independently show it.
    from lib.input_schema import (POWERSET_LOCAL_FLOOR_MIN_SEC,
                                  POWERSET_LOCAL_FLOOR_MIN_WINDOWS)
    _SPK = {1: (0,), 2: (1,), 3: (2,), 4: (0, 1), 5: (0, 2), 6: (1, 2)}
    need_fr = max(2, int(round(POWERSET_LOCAL_FLOOR_MIN_SEC / fdur)))
    win_counts = []
    for w in range(W):
        act = [0, 0, 0]
        for c, spks in _SPK.items():
            fr = int(np.sum(cls[w] == c))
            for sp in spks:
                act[sp] += fr
        win_counts.append(sum(1 for a in act if a >= need_fr))
    local_floor = 0
    for c in range(3, 0, -1):
        if sum(1 for wc in win_counts if wc >= c) \
                >= POWERSET_LOCAL_FLOOR_MIN_WINDOWS:
            local_floor = c
            break

    # ── consensus fences: flip events voted across covering windows ──
    all_flips = sorted(t for w in range(W)
                       for t in _window_flips(cls[w], starts[w], fdur,
                                              min_fr))
    fences = []
    i = 0
    while i < len(all_flips):
        j = i
        while j + 1 < len(all_flips) \
                and all_flips[j + 1] - all_flips[i] <= 0.35:
            j += 1
        t = float(np.median(all_flips[i:j + 1]))
        covering = int(np.sum((starts <= t) & (starts + window_sec >= t)))
        if covering and (j - i + 1) / covering >= POWERSET_FENCE_SUPPORT:
            fences.append(round(t, 3))
        i = j + 1

    # ── overlap spans: permutation-invariant mean over windows ───────
    nfr = int((chunk_end - chunk_start) / fdur) + 2
    grid = np.zeros(nfr, dtype=np.float32)
    cnt = np.zeros(nfr, dtype=np.float32)
    for w in range(W):
        g0 = int(round((starts[w] - chunk_start) / fdur))
        n = min(F, nfr - g0)
        if n <= 0:
            continue
        grid[g0:g0 + n] += ovl[w, :n]
        cnt[g0:g0 + n] += 1
    mean = grid / np.maximum(cnt, 1)
    overlap_spans = []
    on = None
    for k, v in enumerate(mean):
        if v > POWERSET_OVERLAP_THRESHOLD and on is None:
            on = k
        elif v <= POWERSET_OVERLAP_THRESHOLD and on is not None:
            overlap_spans.append([round(chunk_start + on * fdur, 3),
                                  round(chunk_start + k * fdur, 3)])
            on = None
    if on is not None:
        overlap_spans.append([round(chunk_start + on * fdur, 3),
                              round(chunk_start + nfr * fdur, 3)])
    in_span = mean > POWERSET_OVERLAP_THRESHOLD      # [nfr] bool grid

    # ── stitch windows into chains (bind-or-break) ────────────────────
    share = int(round((window_sec - step_sec) / fdur))
    chain_of: list[dict] = [dict() for _ in range(W)]
    nxt = 0
    for lbl in {_SOLO[c] for c in cls[0] if c in _SOLO}:
        chain_of[0][lbl] = nxt
        nxt += 1
    # NOTE (2026-09-30): fence-aware junction masking was tried and
    # REVERTED after failing its neutrality proof — our content class
    # carries in-speech fences on 1.5-9.3% of sentences, so masking
    # fence-adjacent evidence rebuilt chains corpus-wide (reality_a
    # retention 0.644 -> 0.485), while the rapid-alternation content
    # it targeted has clean-pause turns (VAD boundaries, no in-speech
    # fences) and barely moved (sims DER +16.3 -> +13.7). The sims'
    # confusion source is NOT junction welds; it remains undiagnosed.
    for w in range(W - 1):
        a, b = cls[w, F - share:], cls[w + 1, :share]
        m = np.zeros((3, 3))
        for i in range(min(len(a), len(b))):
            if a[i] in _SOLO and b[i] in _SOLO:
                m[_SOLO[a[i]], _SOLO[b[i]]] += 1
        # PER-ROW bind-or-break (the synthetic edge that forced it: a
        # whole-junction margin let the MINOR speaker's ambiguity veto
        # the MAJOR speaker's certain binding, fragmenting chains at
        # every uneven handover). Each observed label binds on its own
        # evidence: a decisive row lead and an uncontested column;
        # anything less starts a fresh chain for THAT label only.
        bound = {}
        rows = sorted(range(3), key=lambda i: -m[i].sum())
        taken_cols = set()
        for i in rows:
            if m[i].sum() < min_fr or i not in chain_of[w]:
                continue
            order = np.argsort(m[i])[::-1]
            j, j2 = int(order[0]), int(order[1])
            if j in taken_cols:
                continue
            if m[i, j] >= min_fr and m[i, j] >= 2.0 * m[i, j2]:
                bound[j] = chain_of[w][i]
                taken_cols.add(j)
        for lbl in {_SOLO[c] for c in cls[w + 1] if c in _SOLO}:
            if lbl in bound:
                chain_of[w + 1][lbl] = bound[lbl]
            else:
                chain_of[w + 1][lbl] = nxt
                nxt += 1

    # ── per-frame chain votes -> track segments ──────────────────────
    votes: list[dict] = [dict() for _ in range(nfr)]
    for w in range(W):
        g0 = int(round((starts[w] - chunk_start) / fdur))
        for f in range(min(F, nfr - g0)):
            if g0 + f < 0:
                continue
            c = cls[w, f]
            if c in _SOLO and _SOLO[c] in chain_of[w]:
                ch = chain_of[w][_SOLO[c]]
                votes[g0 + f][ch] = votes[g0 + f].get(ch, 0) + 1
    frame_chain = np.full(nfr, -1, dtype=np.int32)
    for k, v in enumerate(votes):
        if v:
            frame_chain[k] = max(v, key=v.get)
    segs_by_chain: dict = {}
    i = 0
    max_gap_fr = int(round(POWERSET_TRACK_GAP_SEC / fdur))
    while i < nfr:
        ch = frame_chain[i]
        if ch < 0:
            i += 1
            continue
        j, gap = i, 0
        while j + 1 < nfr:
            if frame_chain[j + 1] == ch:
                gap = 0
            elif frame_chain[j + 1] < 0 and gap < max_gap_fr:
                gap += 1
            else:
                break
            j += 1
        if (j - i) * fdur >= POWERSET_MIN_SEG_SEC:
            segs_by_chain.setdefault(int(ch), []).append(
                [round(chunk_start + i * fdur, 3),
                 round(chunk_start + (j - gap) * fdur, 3)])
        i = j + 1
    tracks = [{"id": ch, "segs": segs}
              for ch, segs in sorted(segs_by_chain.items())
              if sum(b - a for a, b in segs) >= POWERSET_MIN_TRACK_SEC]
    kept = {t["id"] for t in tracks}

    # ── co-activity cannot-links + max-clique K floor ─────────────────
    # Per-pair DURABLE SECONDS ride along (co_active_secs, aligned with
    # co_active_pairs): pair-class frames deduped on the absolute grid,
    # because overlapping windows re-observe the same instant. Track
    # segments can NEVER measure this — the frame-vote grid assigns
    # each frame to exactly one chain, so chain segments are disjoint
    # by construction (measured 2026-09-29: all 434 corpus links had
    # 0.00 s segment intersection, which silently killed every
    # duration-gated consumer).
    # SPAN-CORROBORATED (2026-09-30 live run): only frames the mean
    # overlap probability also sustains (in_span) count toward the
    # seconds. Raw argmax pair-frames measure the same on artifacts as
    # on dialogue — a lone narrator's single junction link (a music
    # transition between two chains SEVEN SECONDS apart) measured
    # 1.22 s, dead-center in the true-dialogue distribution (per-link
    # medians 0.84-1.61 s across 4 assets), and blocked the narrator's
    # own reunification. Span intersection separates structurally:
    # 0/1 link corroborated on the narrator vs 20/28 on the quiz show
    # (n=263 links, 4 assets). The link itself (cannot-link + clique
    # floor input) still needs one window with >= 0.25 s of pair
    # frames — emission is unchanged, only the SECONDS are gated.
    colinks = set()
    link_frames: dict = {}
    for w in range(W):
        g0 = int(round((starts[w] - chunk_start) / fdur))
        for c, (i, j) in _PAIRS.items():
            if i not in chain_of[w] or j not in chain_of[w]:
                continue
            fr = np.flatnonzero(cls[w] == c)
            if len(fr) * fdur < 0.25:
                continue
            a, b = chain_of[w][i], chain_of[w][j]
            if a != b and a in kept and b in kept:
                key = (min(a, b), max(a, b))
                colinks.add(key)
                link_frames.setdefault(key, set()).update(
                    int(g0 + f) for f in fr
                    if 0 <= g0 + f < nfr and in_span[g0 + f])
    adj: dict = {}
    for a, b in colinks:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best = [1] if tracks else [0]

    def _bk(r, p, x):
        if not p and not x:
            best[0] = max(best[0], len(r))
            return
        for v in list(p):
            _bk(r | {v}, p & adj.get(v, set()), x & adj.get(v, set()))
            p.remove(v)
            x.add(v)
    _bk(set(), set(adj), set())

    co_pairs = sorted(list(p) for p in colinks)
    return {"fences": fences, "overlap_spans": overlap_spans,
            "tracks": tracks,
            "co_active_pairs": co_pairs,
            "co_active_secs": [round(len(link_frames[(a, b)]) * fdur, 3)
                               for a, b in co_pairs],
            "clique_floor": int(best[0]),
            "local_speaker_floor": int(local_floor)}


# ── consumers ─────────────────────────────────────────────────────────

def run_prefix_of(output_prefix: str) -> str:
    """The Map iterators carry `{run}/dynamic` as their output_prefix
    (dynamic-chunk intermediates land there); the powerset artifacts
    live at the RUN root where the VAD state wrote them. Measured
    live 2026-09-25: both Map-state consumers listed the /dynamic
    prefix and silently no-op'd — the flag-on run reproduced baseline
    exactly (an honest degrade, but not the experiment)."""
    p = output_prefix.rstrip("/")
    return p[:-len("/dynamic")] if p.endswith("/dynamic") else p


def load_guides(bucket: str, output_prefix: str) -> dict | None:
    """Merge every powerset_{n}.json under the run prefix. None when
    the flag was off (absent artifacts = not assessed, three-state).
    Chain ids are namespaced by chunk (chunks share no audio)."""
    import boto3
    from lib.s3_utils import get_json
    s3 = boto3.client("s3")
    keys = []
    token = None
    root = run_prefix_of(output_prefix)
    while True:
        kw = {"Bucket": bucket, "Prefix": f"{root}/powerset/"}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    if not keys:
        return None
    merged = {"fences": [], "overlap_spans": [], "tracks": [],
              "co_active_pairs": [], "co_active_secs": [],
              "clique_floor": 0, "local_speaker_floor": 0}
    by_idx = {}
    for key in keys:
        try:
            idx = int(key.rsplit("_", 1)[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        by_idx[idx] = key
    per_chunk_tracks: dict = {}
    for idx in sorted(by_idx):
        doc = get_json(bucket, by_idx[idx])
        # namespace by TRUE chunk index (was lexical sort rank, which
        # diverges from the index at >=10 chunks — self-consistent but
        # wrong for boundary adjacency)
        off = (idx + 1) * 1_000_000
        merged["fences"] += doc.get("fences", [])
        merged["overlap_spans"] += doc.get("overlap_spans", [])
        tracks = [{"id": t["id"] + off, "segs": t["segs"]}
                  for t in doc.get("tracks", [])]
        merged["tracks"] += tracks
        per_chunk_tracks[idx] = tracks
        pairs = doc.get("co_active_pairs", [])
        merged["co_active_pairs"] += [[a + off, b + off]
                                      for a, b in pairs]
        # artifact predates co_active_secs: each link reads at 0.25 s,
        # the emission rule's guaranteed lower bound (a link only
        # exists when one window carried >= 0.25 s of pair frames)
        merged["co_active_secs"] += (doc.get("co_active_secs")
                                     or [0.25] * len(pairs))
        merged["clique_floor"] = max(merged["clique_floor"],
                                     doc.get("clique_floor", 0))
        merged["local_speaker_floor"] = max(
            merged["local_speaker_floor"],
            doc.get("local_speaker_floor", 0))
    merged["fences"].sort()
    _bind_across_chunks(merged, per_chunk_tracks)
    return merged


def _bind_across_chunks(merged: dict, per_chunk_tracks: dict) -> None:
    """Uncontested continuity binding at chunk ownership boundaries.

    Chunk ownership splits hard at fixed marks and tracks are CLIPPED
    there, so a voice speaking across a mark is guaranteed split into
    two chains. When chunk k's tracks end at time B and chunk k+1's
    begin at the same B (a straddle), and EXACTLY ONE chain sits on
    each side, the voice provably flows across — the voice MAY flow across — emit a bind CANDIDATE
    (applied by the clustering state only when the chain embeddings
    also agree; continuity alone proved insufficient on
    rapid-exchange content).
    Anything contested or discontinuous: break, never guess. Measured
    on the 13-asset corpus: every observed straddle was uncontested
    (27/27); binder A/B left K unchanged and reunified boundary voices
    the embedder had split.
    """
    from lib.input_schema import POWERSET_XCHUNK_BIND_TOL as TOL
    rename: dict = {}
    for idx in sorted(per_chunk_tracks):
        left, right = per_chunk_tracks.get(idx), per_chunk_tracks.get(idx + 1)
        if not left or not right:
            continue
        end_k = max(b for t in left for a, b in t["segs"])
        start_k1 = min(a for t in right for a, b in t["segs"])
        if abs(end_k - start_k1) > TOL:
            continue                     # silence at the mark: no straddle
        enders = [t for t in left
                  if any(abs(b - end_k) <= TOL for a, b in t["segs"])]
        starters = [t for t in right
                    if any(abs(a - start_k1) <= TOL for a, b in t["segs"])]
        if len(enders) != 1 or len(starters) != 1:
            continue                     # contested: break, never guess
        src = starters[0]["id"]
        dst = rename.get(enders[0]["id"], enders[0]["id"])
        rename[src] = dst                # transitive across boundaries
    # candidates only: application moved to the clustering state,
    # which holds chain embeddings — continuity at a hard chunk mark is
    # ALSO the signature of a handover (A stops, B starts exactly where
    # the knife fell), so a bind additionally requires the two chains'
    # voices to agree (measured: one continuity-only bind on
    # rapid-exchange family content welded two people and cost 0.10
    # retention ARI).
    merged["xchunk_bind_candidates"] = sorted(rename.items())
    return rename


def apply_fences(sentences: list[dict], fences: list[float]) -> list[dict]:
    """Split sentences at interior fences, snapping to word gaps. Runs
    AFTER merge_short — one hook covers both lumping paths (wtpsplit
    joining a rapid exchange, and merge_short folding across a sub-0.2s
    pause). A fence with no word boundary near it (mid-word) cannot
    split and is left alone — never truncate a word."""
    if not fences:
        return sentences
    out = []
    for s in sentences:
        words = s.get("words") or []
        inner = [t for t in fences
                 if float(s["start"]) + 0.2 < t < float(s["end"]) - 0.2]
        if not inner or len(words) < 2:
            out.append(s)
            continue
        parts, cur, fi = [], [], 0
        for w in words:
            while fi < len(inner) and float(w["start"]) >= inner[fi]:
                if cur:
                    parts.append(cur)
                cur = []
                fi += 1
            cur.append(w)
        if cur:
            parts.append(cur)
        if len(parts) <= 1:
            out.append(s)
            continue
        for p in parts:
            ns = dict(s)
            ns["words"] = p
            ns["start"] = float(p[0]["start"])
            ns["end"] = float(p[-1]["end"])
            ns["text"] = " ".join(str(w.get("word", "")).strip()
                                  for w in p).strip()
            ns["fence_split"] = True
            out.append(ns)
    return out


def subtract_spans(a: float, b: float,
                   cuts: list) -> list[tuple[float, float]]:
    """[a,b] minus overlap spans -> remaining pieces (for slice
    excision: cross-talk frames contaminate an embedding even when one
    voice dominates the slice)."""
    segs = [(a, b)]
    for c0, c1 in cuts:
        nxt = []
        for s0, s1 in segs:
            if c1 <= s0 or c0 >= s1:
                nxt.append((s0, s1))
                continue
            if c0 > s0:
                nxt.append((s0, c0))
            if c1 < s1:
                nxt.append((c1, s1))
        segs = nxt
    return segs
