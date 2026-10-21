# test/test_powerset_guides.py
"""Powerset guides — fences, stitching, tracks, co-activity, and the
consumers (apply_fences, subtract_spans, the fenced slicer). Synthetic
frame patterns; no ONNX, no S3. Run: python3 test/test_powerset_guides.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.powerset_guides import (  # noqa: E402
    PowersetCollector, apply_fences, build_guides, subtract_spans,
)

WIN, STEP, F = 10.0, 1.0, 100          # fdur = 0.1 s


def _probs(cls_row):
    """one-hot [F, 7] probabilities from a class row."""
    p = np.full((len(cls_row), 7), 0.01, dtype=np.float32)
    for i, c in enumerate(cls_row):
        p[i, c] = 0.93
    return p


def _collect(class_of_t, dur):
    """Build a collector from an absolute-time -> class function."""
    col = PowersetCollector()
    t = 0.0
    while t + WIN <= dur + 1e-6:
        row = [class_of_t(t + f * (WIN / F)) for f in range(F)]
        col.append(t, _probs(row))
        t += STEP
    return col


def test_fence_track_and_stitch_on_a_clean_handover():
    # speaker in slot 1 talks 0-15s, slot 2 talks 15-30s, no pause
    col = _collect(lambda t: 1 if t < 15 else 2, 30.0)
    g = build_guides(col, WIN, STEP, 0.0, 30.0)
    assert len(g["fences"]) == 1, g["fences"]
    assert abs(g["fences"][0] - 15.0) < 0.3
    assert len(g["tracks"]) == 2, g["tracks"]
    spans = sorted((t["segs"][0][0], t["segs"][-1][1])
                   for t in g["tracks"])
    assert spans[0][0] < 1.0 and abs(spans[0][1] - 15.0) < 0.5
    assert abs(spans[1][0] - 15.0) < 0.5
    assert g["co_active_pairs"] == []
    assert g["clique_floor"] == 1


def test_co_activity_yields_cannot_link_and_clique():
    # slot 1 solo, then sustained 1+2 cross-talk, then slot 2 solo
    def cls(t):
        if t % 10 < 4:
            return 1
        if t % 10 < 5:
            return 4        # pair (0, 1) — slots 1 and 2 co-active
        return 2
    col = _collect(cls, 30.0)
    g = build_guides(col, WIN, STEP, 0.0, 30.0)
    assert g["co_active_pairs"], "co-activity must yield a cannot-link"
    assert g["clique_floor"] >= 2
    # durable seconds ride along, deduped on the frame grid (three 1s
    # cross-talk bursts ≈ 3s despite ~10 windows re-observing each) —
    # track segments can never measure this, they are disjoint. Every
    # covering window agrees here, so the mean overlap probability
    # sustains and the frames are span-corroborated.
    assert len(g["co_active_secs"]) == len(g["co_active_pairs"])
    assert 2.0 <= g["co_active_secs"][0] <= 4.0, g["co_active_secs"]


def test_co_activity_secs_require_sustained_span():
    """A pair-class flicker seen by ONE window among ~10 covering it
    still emits the cannot-link (0.25 s window bar met) but counts
    ZERO corroborated seconds — the mean overlap probability never
    sustains. Live lesson (2026-09-30): a music-transition artifact
    between two chains of one narrator measured 1.22 s of raw argmax
    pair frames — inside the true-dialogue distribution — and vetoed
    the narrator's own reunification; span corroboration separates
    the two structurally (0/1 vs 20/28 links on real content)."""
    col = PowersetCollector()
    t, w = 0.0, 0
    while t + WIN <= 30.0 + 1e-6:
        row = []
        for f in range(F):
            abs_t = t + f * (WIN / F)
            row.append(1 if (abs_t % 10) < 5 else 2)
        if w == 10:                       # one flickering window
            for f in range(F):
                abs_t = t + f * (WIN / F)
                if 14.0 <= abs_t < 15.0:
                    row[f] = 4            # pair class, 1s
        col.append(t, _probs(row))
        t += STEP
        w += 1
    g = build_guides(col, WIN, STEP, 0.0, 30.0)
    assert g["co_active_pairs"], "the flicker link must still emit"
    assert all(s == 0.0 for s in g["co_active_secs"]), g["co_active_secs"]


def test_ambiguous_junction_breaks_the_chain():
    # the shared region between adjacent windows is SILENT: no
    # evidence to bind — the chain must break, never guess
    def cls(t):
        return 1 if t < 10.5 else (0 if t < 21 else 2)
    col = _collect(cls, 32.0)
    g = build_guides(col, WIN, STEP, 0.0, 32.0)
    ids = {t["id"] for t in g["tracks"]}
    assert len(ids) == 2, g["tracks"]    # two chains, never welded


def test_floor_is_a_minimum_not_a_target():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "states",
                        "global_speakers_clustering",
                        "lambda_function.py")
    spec = importlib.util.spec_from_file_location("clu_state7", path)
    clu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clu)
    log = [{"k": 2, "k_act": 2, "sil": 0.16, "ch": 5.0, "combined": 0.4},
           {"k": 3, "k_act": 3, "sil": 0.10, "ch": 4.0, "combined": 0.2},
           {"k": 6, "k_act": 6, "sil": 0.12, "ch": 6.0, "combined": 0.3},
           {"k": 7, "k_act": 7, "sil": 0.11, "ch": 5.5, "combined": None}]
    # best VALID candidate >= floor wins — not the floor itself
    assert clu._best_k_at_or_above(log, 3) == 6
    # all >= floor vetoed (combined None): fall back to silhouette
    assert clu._best_k_at_or_above(
        [{"k": 7, "k_act": 7, "sil": 0.11, "combined": None},
         {"k": 8, "k_act": 8, "sil": 0.13, "combined": None}], 5) == 8
    # nothing at or above the floor: the floor itself
    assert clu._best_k_at_or_above(log[:1], 5) == 5
    assert clu._best_k_at_or_above([], 4) == 4


def test_apply_fences_splits_at_word_gap_only():
    words = [{"word": "a", "start": 0.0, "end": 1.0},
             {"word": "b", "start": 1.1, "end": 2.0},
             {"word": "c", "start": 2.1, "end": 3.0}]
    s = {"start": 0.0, "end": 3.0, "text": "a b c", "words": words}
    out = apply_fences([dict(s)], [2.05])
    assert len(out) == 2
    assert out[0]["end"] == 2.0 and out[1]["start"] == 2.1
    assert out[1]["fence_split"] is True
    # a fence mid-word cannot split — never truncate a word
    out = apply_fences([dict(s)], [1.5])
    assert len(out) == 2                 # snaps to the 1.0/1.1 gap side
    # no interior fence -> untouched
    out = apply_fences([dict(s)], [9.0])
    assert len(out) == 1 and "fence_split" not in out[0]


def test_run_prefix_strips_the_dynamic_suffix():
    from lib.powerset_guides import run_prefix_of
    assert run_prefix_of("micro-workflow/sd/abc/dynamic") == \
        "micro-workflow/sd/abc"
    assert run_prefix_of("micro-workflow/sd/abc/dynamic/") == \
        "micro-workflow/sd/abc"
    assert run_prefix_of("micro-workflow/sd/abc") == \
        "micro-workflow/sd/abc"


def test_subtract_spans_excises_crosstalk():
    pieces = subtract_spans(0.0, 10.0, [[3.0, 4.0], [8.0, 12.0]])
    assert pieces == [(0.0, 3.0), (4.0, 8.0)]
    assert subtract_spans(5.0, 6.0, [[0.0, 20.0]]) == []


def test_fenced_slicer_breaks_at_the_fence():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "states", "run_speech_embedding_model"))
    from lambda_function import _find_longest_speech_segment
    words = [{"start": float(i), "end": float(i) + 0.9}
             for i in range(6)]           # one 5.9s run, gaps 0.1s
    seg = _find_longest_speech_segment(words)
    assert seg == (0.0, 5.9)
    seg = _find_longest_speech_segment(words, fences=[2.95])
    assert seg == (3.0, 5.9)              # the longer fenced half


def test_track_f0_hygiene_quarantines_and_cannot_links():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "states",
                        "global_speakers_clustering",
                        "lambda_function.py")
    spec = importlib.util.spec_from_file_location("clu_state", path)
    clu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clu)
    rng = np.random.default_rng(7)
    # identical embeddings for ALL tracks: only F0 separates them
    base = rng.normal(size=256).astype(np.float32)
    base /= np.linalg.norm(base)

    def sents(t0, n, f0):
        out = []
        for i in range(n):
            a = t0 + i * 2.0
            out.append({"start": a, "end": a + 1.5, "duration": 1.5,
                        "segment_start": a, "segment_end": a + 1.5,
                        "f0_hz": f0 + rng.normal(0, 3),
                        "f0_voiced_ratio": 0.6,
                        "embedding": (base
                                      + rng.normal(0, 0.01, 256)
                                      ).tolist(),
                        "dyn_chunk_idx": 0, "sentence_idx": i})
        return out

    # track A: male register (110 Hz); track B: female (210); track C:
    # WELDED — half 110, half 220 (register-straddling: quarantined);
    # track D: EXCITED same-register female (200 vs 320 — huge ratio
    # but both above the divide: the round-3 reality_a lesson, must SURVIVE)
    high = (sents(0, 8, 110) + sents(100, 8, 210)
            + sents(200, 4, 110) + sents(208, 4, 220)
            + sents(300, 4, 200) + sents(308, 4, 320))
    def seg(chain, a, b):
        return {"chain": chain, "start": a, "end": b, "duration": b - a,
                "embedding": base.tolist()}
    track_segs = [seg(1, 0, 16), seg(2, 100, 116), seg(3, 200, 216),
                  seg(4, 300, 316)]
    guides = {"co_active_pairs": [], "clique_floor": 0}
    labels, final, method, diag = clu._cluster_tracks(
        track_segs, guides, high, {"max_k": 10}, assign_margin=0.05)
    assert diag["bimodal_tracks"] == 1, diag       # C detected (report-only), never D
    assert diag["n_f0_cannot_links"] >= 1, diag
    # identical embeddings, but the F0 cannot-link keeps the two
    # register tracks apart
    la = {labels[i] for i, s2 in enumerate(high) if s2["start"] < 20}
    lb = {labels[i] for i, s2 in enumerate(high)
          if 100 <= s2["start"] < 120}
    assert la and lb and not (la & lb), (la, lb)


def test_post_split_divides_separable_cluster_only():
    """Two embedding-separable chain groups forced into one cluster by
    num_speakers=1 must be divided by the post-clustering split pass;
    a homogeneous cluster must survive it untouched."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "states",
                        "global_speakers_clustering",
                        "lambda_function.py")
    spec = importlib.util.spec_from_file_location("clu_state2", path)
    clu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clu)
    rng = np.random.default_rng(11)

    def unit(v):
        return v / np.linalg.norm(v)

    ea = unit(rng.normal(size=256)).astype(np.float32)
    eb = unit(rng.normal(size=256)).astype(np.float32)

    def sents(t0, n, base):
        out = []
        for i in range(n):
            a = t0 + i * 2.0
            out.append({"start": a, "end": a + 1.5, "duration": 1.5,
                        "segment_start": a, "segment_end": a + 1.5,
                        "f0_hz": None, "f0_voiced_ratio": 0.0,
                        "embedding": unit(base + rng.normal(0, 0.05, 256)
                                          ).tolist(),
                        "dyn_chunk_idx": 0, "sentence_idx": i})
        return out

    def seg(chain, a, b, base):
        return {"chain": chain, "start": a, "end": b, "duration": b - a,
                "embedding": unit(base + rng.normal(0, 0.05, 256)
                                  ).tolist()}

    # 4 chains near ea + 4 chains near eb, welded by num_speakers=1
    high, track_segs = [], []
    for c in range(4):
        t0 = c * 40
        high += sents(t0, 4, ea)
        track_segs.append(seg(c + 1, t0, t0 + 8, ea))
    for c in range(4):
        t0 = 400 + c * 40
        high += sents(t0, 4, eb)
        track_segs.append(seg(c + 5, t0, t0 + 8, eb))
    guides = {"co_active_pairs": [], "clique_floor": 0}
    labels, _, method, diag = clu._cluster_tracks(
        track_segs, guides, high, {"max_k": 10, "num_speakers": 1},
        assign_margin=0.05)
    assert diag["n_post_splits"] == 1, diag
    assert "post_split" in method, method
    la = {labels[i] for i, s in enumerate(high) if s["start"] < 200}
    lb = {labels[i] for i, s in enumerate(high) if s["start"] >= 400}
    assert la and lb and not (la & lb), (la, lb)

    # homogeneous: 8 chains of ONE voice under the same forced K=1
    high, track_segs = [], []
    for c in range(8):
        t0 = c * 40
        high += sents(t0, 4, ea)
        track_segs.append(seg(c + 1, t0, t0 + 8, ea))
    labels, _, method, diag = clu._cluster_tracks(
        track_segs, guides, high, {"max_k": 10, "num_speakers": 1},
        assign_margin=0.05)
    assert diag["n_post_splits"] == 0, diag
    assert len(set(labels.tolist())) == 1, set(labels.tolist())


def test_small_n_guard_gates_the_tracks_path():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "states",
                        "global_speakers_clustering",
                        "lambda_function.py")
    spec = importlib.util.spec_from_file_location("clu_state3", path)
    clu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clu)
    segs = [{"chain": c, "start": c * 10.0, "end": c * 10.0 + 5,
             "duration": 5.0, "embedding": [0.0] * 256}
            for c in range(8)]
    cfg = {}                                    # defaults: floor = 5*3 = 15
    ok, n = clu._tracks_viable(segs, cfg)
    assert (ok, n) == (False, 8), (ok, n)      # the measured collapse regime
    segs *= 4                                   # same 8 chains, more segs
    assert clu._tracks_viable(segs, cfg) == (False, 8)
    many = [dict(s, chain=i) for i, s in enumerate(segs)]
    ok, n = clu._tracks_viable(many, cfg)
    assert ok and n == 32, (ok, n)
    # the floor SCALES with min_turns_per_speaker, not an absolute count
    assert clu._tracks_viable(many, {"min_turns_per_speaker": 7})[0] is False
    assert clu._tracks_viable(many, {"min_turns_per_speaker": 1})[0] is True
    assert clu._tracks_viable([], cfg) == (False, 0)
    assert clu._tracks_viable(None, cfg) == (False, 0)


def test_single_narrator_takes_the_fast_path_in_tracks_mode():
    """One-voice content (documentary narrator): the single-speaker
    detector runs BEFORE the K sweep in the tracks path too, and an
    empty cannot-link list must not suppress it. Boundary note: the
    detector keys on dendrogram height + mean pairwise distance vs
    single_speaker_threshold (0.2); same-voice track embeddings sit
    well inside (measured here at cosine distance ~0.03-0.1), while a
    voice noisy enough to cross 0.2 over-splits with the ⚠unreliable
    flag — the pre-existing, downstream-recoverable behavior."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "states",
                        "global_speakers_clustering",
                        "lambda_function.py")
    spec = importlib.util.spec_from_file_location("clu_state4", path)
    clu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clu)
    rng = np.random.default_rng(5)
    base = rng.normal(size=256)
    base /= np.linalg.norm(base)

    def unit(v):
        return v / np.linalg.norm(v)

    high, segs = [], []
    for c in range(20):
        t0 = c * 30.0
        segs.append({"chain": c + 1, "start": t0, "end": t0 + 12,
                     "duration": 12.0,
                     "embedding": unit(base + rng.normal(0, 0.01, 256)
                                       ).tolist()})
        for i in range(3):
            a = t0 + i * 4.0
            high.append({"start": a, "end": a + 3, "duration": 3.0,
                         "segment_start": a, "segment_end": a + 3,
                         "f0_hz": 120.0, "f0_voiced_ratio": 0.6,
                         "embedding": unit(base + rng.normal(0, 0.01, 256)
                                           ).tolist(),
                         "dyn_chunk_idx": 0, "sentence_idx": i})
    guides = {"co_active_pairs": [], "clique_floor": 1}
    labels, _, method, diag = clu._cluster_tracks(
        segs, guides, high, {"max_k": 10}, assign_margin=0.05)
    assert diag.get("selection_reason") == "single_speaker_auto", diag
    assert len({l for l in labels.tolist() if l >= 0}) == 1
    assert "single_speaker" in method, method


def test_reconcile_evidence_gates_use_measured_co_activity():
    """Two same-voice cluster fragments (centroid sim ~0.91) must
    reunify in the reconcile pass when their only cannot-link is a
    sub-second flicker, and must stay apart when the co-presence is
    durable (>= POWERSET_RECONCILE_LINK_MIN_SEC of pair-class frames,
    read from the guides' co_active_secs — 2026-09-29: measuring this
    from track-segment intersection is impossible, segments are
    disjoint by construction, and both gates were dead)."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "states",
                        "global_speakers_clustering",
                        "lambda_function.py")
    spec = importlib.util.spec_from_file_location("clu_state5", path)
    clu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clu)
    rng = np.random.default_rng(23)

    def unit(v):
        return v / np.linalg.norm(v)

    ea = unit(rng.normal(size=256)).astype(np.float32)
    ortho = rng.normal(size=256)
    ortho = unit(ortho - (ortho @ ea) * ea)
    eb = unit(ea + 0.45 * ortho)          # cos(ea, eb) ~ 0.91

    def sents(t0, n, base, i0):
        out = []
        for i in range(n):
            a = t0 + i * 2.0
            out.append({"start": a, "end": a + 1.5, "duration": 1.5,
                        "segment_start": a, "segment_end": a + 1.5,
                        "f0_hz": None, "f0_voiced_ratio": 0.0,
                        "embedding": unit(base + rng.normal(0, 0.03, 256)
                                          ).tolist(),
                        "dyn_chunk_idx": 0, "sentence_idx": i0 + i})
        return out

    def seg(chain, a, b, base):
        return {"chain": chain, "start": a, "end": b, "duration": b - a,
                "embedding": unit(base + rng.normal(0, 0.03, 256)
                                  ).tolist()}

    high, track_segs = [], []
    for c in range(6):
        t0 = c * 40
        high += sents(t0, 4, ea, c * 4)
        track_segs.append(seg(c + 1, t0, t0 + 8, ea))
    for c in range(6):
        t0 = 400 + c * 40
        high += sents(t0, 4, eb, 100 + c * 4)
        track_segs.append(seg(c + 7, t0, t0 + 8, eb))
    cfg = {"max_k": 10, "min_turns_per_speaker": 1}

    def run(secs):
        guides = {"co_active_pairs": [[1, 7]], "co_active_secs": [secs],
                  "clique_floor": 0}
        return clu._cluster_tracks(track_segs, guides, high, cfg,
                                   assign_margin=0.05)

    # durable co-presence: two people, the merge is vetoed
    labels, _, method, diag = run(5.0)
    la = {labels[i] for i, s in enumerate(high) if s["start"] < 300}
    lb = {labels[i] for i, s in enumerate(high) if s["start"] >= 400}
    assert not (la & lb), (la, lb)
    assert any(b[3].startswith("co_active")
               for b in diag["reconcile_blocked"]), diag
    # flicker link: absent durable evidence, the fragments reunify
    labels, _, method, diag = run(0.3)
    assert diag["n_reconciled"] >= 1, diag
    la = {labels[i] for i, s in enumerate(high) if s["start"] < 300}
    lb = {labels[i] for i, s in enumerate(high) if s["start"] >= 400}
    assert la == lb, (la, lb)


def test_conversation_veto_needs_corroborated_co_presence():
    """Quick A<->B alternation blocks a reconcile merge ONLY with >=
    POWERSET_RECONCILE_CORROB_SEC of measured co-presence: real
    dialogue (reality_a weld, 3 links, retention 0.961 -> 0.767 when
    merged) must stay apart, while the same alternation pattern with a
    single emission-floor blip (bbc lone narrator, 1 flicker link)
    must reunify — the corroboration bar, not the handover count, is
    the decider."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "states",
                        "global_speakers_clustering",
                        "lambda_function.py")
    spec = importlib.util.spec_from_file_location("clu_state6", path)
    clu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clu)
    rng = np.random.default_rng(31)

    def unit(v):
        return v / np.linalg.norm(v)

    ea = unit(rng.normal(size=256)).astype(np.float32)
    ortho = rng.normal(size=256)
    ortho = unit(ortho - (ortho @ ea) * ea)
    eb = unit(ea + 0.45 * ortho)          # cos(ea, eb) ~ 0.91

    # groups INTERLEAVED: A speaks at t, B at t+2, gaps 0.5s < QUICK —
    # dozens of quick handovers either way
    high, track_segs = [], []
    for c in range(6):
        base_t = c * 16
        ta = [base_t + k * 4 for k in range(4)]
        tb = [base_t + 2 + k * 4 for k in range(4)]
        for grp, (chain, times, emb) in enumerate(
                ((c + 1, ta, ea), (c + 7, tb, eb))):
            for k, t0 in enumerate(times):
                high.append(
                    {"start": t0, "end": t0 + 1.5, "duration": 1.5,
                     "segment_start": t0, "segment_end": t0 + 1.5,
                     "f0_hz": None, "f0_voiced_ratio": 0.0,
                     "embedding": unit(emb + rng.normal(0, 0.03, 256)
                                       ).tolist(),
                     "dyn_chunk_idx": 0,
                     "sentence_idx": c * 8 + grp * 4 + k})
                # per-sentence chain segments: the groups interleave,
                # so a single covering span would swallow the other
                # group's sentences at midpoint mapping
                track_segs.append(
                    {"chain": chain, "start": t0, "end": t0 + 1.5,
                     "duration": 1.5,
                     "embedding": unit(emb + rng.normal(0, 0.03, 256)
                                       ).tolist()})
    cfg = {"max_k": 10, "min_turns_per_speaker": 1}

    def run(secs):
        guides = {"co_active_pairs": [[1, 7]], "co_active_secs": [secs],
                  "clique_floor": 0}
        return clu._cluster_tracks(track_segs, guides, high, cfg,
                                   assign_margin=0.05)

    def groups(labels):
        la = {labels[i] for i, s in enumerate(high)
              if int(s["start"]) % 4 == 0}
        lb = {labels[i] for i, s in enumerate(high)
              if int(s["start"]) % 4 == 2}
        return la, lb

    # corroborated (0.6s >= 0.5 bar, still < gate 1's 2.0s): dialogue
    labels, _, method, diag = run(0.6)
    la, lb = groups(labels)
    assert not (la & lb), (la, lb)
    assert any(b[3].startswith("conversation")
               for b in diag["reconcile_blocked"]), diag
    # one emission-floor blip (0.25s < 0.5): fragments reunify despite
    # the identical alternation pattern
    labels, _, method, diag = run(0.25)
    assert diag["n_reconciled"] >= 1, diag
    la, lb = groups(labels)
    assert la == lb, (la, lb)


def test_cross_chunk_binding_uncontested_only():
    from lib.powerset_guides import _bind_across_chunks

    def doc(tracks):
        return [{"id": t[0], "segs": t[1]} for t in tracks]

    # chunk0 voice clipped at 720.0, chunk1 continues at 720.0 -> bind
    pc = {0: doc([(1_000_001, [[650.0, 720.0]]),
                  (1_000_002, [[100.0, 130.0]])]),
          1: doc([(2_000_001, [[720.0, 800.0]])]),
          2: doc([(3_000_001, [[1440.0, 1500.0]])])}
    merged = {"tracks": [t for ts in pc.values() for t in ts],
              "co_active_pairs": [[2_000_001, 2_000_009]]}
    # chunk1 also clipped at 1440.0 and chunk2 continues: transitive
    pc[1].append({"id": 2_000_003, "segs": [[1400.0, 1440.0]]})
    rename = _bind_across_chunks(merged, pc)
    # candidates ONLY: application (voice-gated) happens in the
    # clustering state; tracks/co pairs are untouched here
    assert len(rename) == 2, rename
    assert rename[2_000_001] == 1_000_001        # continuity bind
    assert dict(merged["xchunk_bind_candidates"]) == rename
    ids = {t["id"] for t in merged["tracks"]}
    assert 2_000_001 in ids and 1_000_001 in ids  # nothing renamed yet
    assert [2_000_001, 2_000_009] in merged["co_active_pairs"]

    # contested boundary (two enders) -> no bind
    pc2 = {0: doc([(1_000_001, [[700.0, 720.0]]),
                   (1_000_002, [[710.0, 720.0]])]),
           1: doc([(2_000_001, [[720.0, 730.0]])])}
    m2 = {"tracks": [t for ts in pc2.values() for t in ts],
          "co_active_pairs": []}
    assert _bind_across_chunks(m2, pc2) == {}

    # silence at the mark -> no straddle, no bind
    pc3 = {0: doc([(1_000_001, [[650.0, 700.0]])]),
           1: doc([(2_000_001, [[725.0, 760.0]])])}
    m3 = {"tracks": [t for ts in pc3.values() for t in ts],
          "co_active_pairs": []}
    assert _bind_across_chunks(m3, pc3) == {}


if __name__ == "__main__":
    test_fence_track_and_stitch_on_a_clean_handover()
    test_co_activity_yields_cannot_link_and_clique()
    test_co_activity_secs_require_sustained_span()
    test_ambiguous_junction_breaks_the_chain()
    test_floor_is_a_minimum_not_a_target()
    test_apply_fences_splits_at_word_gap_only()
    test_run_prefix_strips_the_dynamic_suffix()
    test_subtract_spans_excises_crosstalk()
    test_fenced_slicer_breaks_at_the_fence()
    test_track_f0_hygiene_quarantines_and_cannot_links()
    test_post_split_divides_separable_cluster_only()
    test_small_n_guard_gates_the_tracks_path()
    test_single_narrator_takes_the_fast_path_in_tracks_mode()
    test_reconcile_evidence_gates_use_measured_co_activity()
    test_conversation_veto_needs_corroborated_co_presence()
    test_cross_chunk_binding_uncontested_only()
    print("all powerset-guide tests passed")
