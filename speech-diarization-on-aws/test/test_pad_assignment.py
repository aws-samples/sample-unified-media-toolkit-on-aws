#!/usr/bin/env python3
"""Pad-boundary word resolution (2026-10-12) — the ladder that stops a
sub-second Whisper slop at a group boundary from becoming a 50 s absolute
error. Fixtures reproduce the two LIVE phantoms from the Meridian
mix/enh/stem comparison plus the designed edge cases. Self-contained:

    python3 test/test_pad_assignment.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.transcribe import (  # noqa: E402
    _resolve_word_groups, rehome_straddle_strays,
)

PAD = 0.5


def groups(*audio_spans):
    """Build groups_meta from (off, dur) pairs laid out with 0.5 s pads."""
    out = []
    for i, (off, dur) in enumerate(audio_spans):
        out.append({"off": off, "end": off + dur, "win_hi": None})
    for i, g in enumerate(out):
        g["win_hi"] = (out[i + 1]["off"] if i + 1 < len(out)
                       else g["end"] + PAD)
    return out


def test_dispatch_geometry_stem():
    """The live stem case: G_a 1.58 s audio, pad, G_b. 'him.' clean in G_a,
    'Yeah,' timestamped 47 ms into the pad before G_b, 'dispatch,' clean in
    G_b (same segment). G_a stays feasible under shrink (0.17 s free), so the
    gated continuity vote decides: left fails the 100 ms gate (453 ms from
    G_a's audio), right straddles (distance 0) with a same-segment successor
    → G_b."""
    gm = groups((0.0, 1.58), (2.08, 6.26))
    words = [
        {"t_s": 1.21, "t_e": 1.41, "seg_id": 0},   # him.   (clean, G_a)
        {"t_s": 2.033, "t_e": 2.233, "seg_id": 1},  # Yeah,  (pad)
        {"t_s": 2.291, "t_e": 2.611, "seg_id": 1},  # dispatch, (clean, G_b)
    ]
    r = _resolve_word_groups(words, gm, support_fn=None)
    assert r[0]["gpos"] == 0 and r[0]["flag"] is None
    assert r[2]["gpos"] == 1 and r[2]["flag"] is None
    # the phantom joins G_b — in absolute terms 311.87+rel, not 259.89
    assert r[1]["gpos"] == 1, r[1]
    assert r[1]["flag"] == "pad_moved"
    # clamped inside G_b, before 'dispatch,'
    assert 0.0 <= r[1]["rel_s"] < r[1]["rel_e"] <= 0.211 + 1e-6, r[1]


def test_empty_group_trap_enh():
    """The live enh case: a 0.54 s noise-blip group with NO words, pad, G_b.
    'Yeah,' straddles the pad/G_b edge. Feasibility passes BOTH sides (the
    blip is empty); the gated continuity vote (same-segment successor in
    G_b, span straddling G_b's boundary = distance 0) must pick G_b."""
    gm = groups((0.0, 0.54), (1.04, 6.28))
    words = [
        {"t_s": 0.754, "t_e": 1.194, "seg_id": 3},  # Yeah,  (pad+straddle)
        {"t_s": 1.333, "t_e": 1.593, "seg_id": 3},  # dispatch. (clean, G_b)
    ]
    r = _resolve_word_groups(words, gm, support_fn=None)
    assert r[0]["gpos"] == 1, r[0]
    assert r[0]["flag"] == "pad_moved"
    assert r[0]["rel_e"] <= 0.293 + 1e-6         # fits before 'dispatch.'


def test_mid_pad_hallucination_dropped_with_aligner():
    """A word floating mid-pad, far (>100 ms) from both edges, no same-
    segment anchor, and the aligner finds no support on either side →
    dropped with provenance. Both groups carry words so feasibility can't
    resolve it alone."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.5, "t_e": 0.9, "seg_id": 0},     # clean, G_a
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},     # mid-pad, isolated segment
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},     # clean, G_b
    ]
    r = _resolve_word_groups(words, gm,
                             support_fn=lambda wi, gp, lo, hi: 0.0)
    assert r[1]["gpos"] is None
    assert r[1]["flag"] == "pad_dropped"


def test_degraded_mode_never_drops():
    """Same geometry, no aligner (multilingual degraded mode): the word is
    KEPT — legacy owner, clamped into real audio — and flagged, never
    dropped."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.5, "t_e": 0.9, "seg_id": 0},
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},
    ]
    r = _resolve_word_groups(words, gm, support_fn=None)
    assert r[1]["gpos"] is not None
    assert r[1]["flag"] in ("pad_unresolved", "pad_moved")
    g = gm[r[1]["gpos"]]
    assert 0.0 <= r[1]["rel_s"] < r[1]["rel_e"] <= (g["end"] - g["off"]) + 1e-6


def test_aligner_adjudicates_when_vote_cannot():
    """Both sides feasible, no same-segment anchor on either side, word
    outside the proximity gate — layer 4 picks the side the audio supports."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.3, "t_e": 0.7, "seg_id": 0},
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},     # isolated pad word
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},
    ]
    r = _resolve_word_groups(
        words, gm,
        support_fn=lambda wi, gp, lo, hi: 0.9 if gp == 1 else 0.1)
    assert r[1]["gpos"] == 1
    assert r[1]["flag"] == "pad_moved"


def test_end_slop_is_clipped_not_snapped():
    """A word 90% inside its group whose END protrudes into the pad keeps
    its start — only the protruding part is clipped (minimal displacement,
    not edge-snapping)."""
    gm = groups((0.0, 10.5), (11.0, 5.0))
    words = [
        {"t_s": 9.0, "t_e": 9.4, "seg_id": 0},      # clean
        {"t_s": 10.0, "t_e": 10.6, "seg_id": 0},    # end-slop into pad
    ]
    r = _resolve_word_groups(words, gm, support_fn=None)
    assert r[1]["gpos"] == 0
    assert abs(r[1]["rel_s"] - 10.0) < 1e-6, r[1]   # start untouched
    assert abs(r[1]["rel_e"] - 10.5) < 1e-6, r[1]   # end clipped to audio


def test_clean_words_byte_identical():
    """Words fully inside a group's audio pass through with their exact
    times and no flag — the healthy-asset no-change guarantee."""
    gm = groups((0.0, 5.0), (5.5, 5.0))
    words = [
        {"t_s": 1.234, "t_e": 1.567, "seg_id": 0},
        {"t_s": 6.111, "t_e": 6.222, "seg_id": 1},
    ]
    r = _resolve_word_groups(words, gm, support_fn=None)
    assert r[0] == {"gpos": 0, "rel_s": 1.234, "rel_e": 1.567, "flag": None}
    assert r[1] == {"gpos": 1, "rel_s": 6.111 - 5.5, "rel_e": 6.222 - 5.5,
                    "flag": None}


def test_single_survivor_still_needs_acoustic_support():
    """Feasibility is a rejector, never a confirmer: when one side is packed
    and the lone survivor has NO acoustic support, the word is dropped (with
    group attribution for stats) — pad hallucinations are not laundered into
    the open side."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.05, "t_e": 1.98, "seg_id": 0},   # packs G_a solid
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},     # mid-pad hallucination
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},     # clean, G_b
    ]
    r = _resolve_word_groups(words, gm,
                             support_fn=lambda wi, gp, lo, hi: 0.0)
    assert r[1]["gpos"] is None
    assert r[1]["flag"] == "pad_dropped"
    # deterministic: sorted() is stable over the insertion-ordered scores
    # dict (left first), so a 0.0/0.0 tie always attributes left
    assert r[1]["near"] == 0


def test_single_survivor_vote_beats_probe():
    """A gated same-segment neighbour is stronger evidence than the probe:
    a lone survivor confirmed by continuity is kept even when the aligner
    reports zero support (false-drop guard for short function words)."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.05, "t_e": 1.98, "seg_id": 0},   # packs G_a solid
        {"t_s": 2.45, "t_e": 2.55, "seg_id": 1},   # straddles G_b's edge
        {"t_s": 2.6, "t_e": 2.9, "seg_id": 1},     # same-segment successor
    ]
    r = _resolve_word_groups(words, gm,
                             support_fn=lambda wi, gp, lo, hi: 0.0)
    assert r[1]["gpos"] == 1
    assert r[1]["flag"] == "pad_moved"


def test_single_survivor_degraded_kept():
    """Degraded mode (no aligner): the lone feasible side is taken as-is —
    nothing is ever dropped without acoustic evidence."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.05, "t_e": 1.98, "seg_id": 0},
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},
    ]
    r = _resolve_word_groups(words, gm, support_fn=None)
    assert r[1]["gpos"] == 1
    assert r[1]["flag"] == "pad_moved"


def test_both_sides_infeasible_drops_duplicate():
    """BOTH sides packed = the slot is already owned — the boundary-
    DUPLICATE signature. Dropped with provenance REGARDLESS of probe
    support (the probe saturates on the twin's audio and cannot tell the
    copy from the original; live-measured: probing here readmitted ~700
    insertions over 19 episodes vs ~20 recovered deletions). Degraded mode
    still never drops."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.05, "t_e": 1.98, "seg_id": 0},   # packs G_a
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},     # pad word (echo twin)
        {"t_s": 2.5, "t_e": 2.9, "seg_id": 2},     # packs G_b's edge
    ]
    r = _resolve_word_groups(
        words, gm,
        support_fn=lambda wi, gp, lo, hi: 0.9 if gp == 1 else 0.1)
    assert r[1]["gpos"] is None, r[1]
    assert r[1]["flag"] == "pad_dropped"
    assert r[1]["near"] == 0                       # stats attribution
    # degraded mode on the same geometry: kept + flagged, never dropped
    r2 = _resolve_word_groups(words, gm, support_fn=None)
    assert r2[1]["gpos"] is not None
    assert r2[1]["flag"] == "pad_unresolved"


def test_probe_abstains_on_unjudgeable_word():
    """A probe that CANNOT judge (None — e.g. '911.' has no alphabetic
    charset) is abstention, not evidence: the word is kept + flagged, never
    dropped."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.5, "t_e": 0.9, "seg_id": 0},
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},     # isolated pad word
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},
    ]
    r = _resolve_word_groups(words, gm,
                             support_fn=lambda wi, gp, lo, hi: None)
    assert r[1]["gpos"] is not None
    assert r[1]["flag"] == "pad_unresolved"


def test_probe_tie_with_room_both_sides_keeps_flagged():
    """Char coverage saturating on BOTH sides (short word, normal speech
    both sides) is non-evidence: within PAD_SUPPORT_MARGIN, with room on
    both sides, the word is never RELOCATED — it stays at its legacy
    window owner (left by construction), flagged."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.5, "t_e": 0.9, "seg_id": 0},
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},
    ]
    r = _resolve_word_groups(words, gm,
                             support_fn=lambda wi, gp, lo, hi: 1.0)
    assert r[1]["gpos"] == 0                 # legacy owner, not a relocation
    assert r[1]["flag"] == "pad_unresolved"


def test_probe_tie_one_feasible_side_relocates():
    """A saturated tie between two SUPPORTED sides is broken by feasibility
    (a rejector choosing between two positives): when exactly one side has
    room, the word relocates there instead of being parked at the packed
    legacy owner."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.05, "t_e": 1.98, "seg_id": 0},   # packs G_a
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},     # pad word, tie 1.0/1.0
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},
    ]
    r = _resolve_word_groups(words, gm,
                             support_fn=lambda wi, gp, lo, hi: 1.0)
    assert r[1]["gpos"] == 1, r[1]
    assert r[1]["flag"] == "pad_moved"


def test_target_charset_is_vocab_driven():
    """The probe's charset filter is the model's own vocab — language
    agnostic. Digits/punctuation fall out for every script (no wav2vec
    vocab emits them) → empty set → abstention, for '911.' in English and
    for a digit word under a char-based vocab alike."""
    import lib.transcribe as T
    old_vocab, old_chars = T._inv_vocab, T._vocab_chars
    try:
        T._inv_vocab = {4: "a", 5: "b", 6: "'", 7: "好"}
        T._vocab_chars = None
        assert T._target_charset("911.") == set()
        assert T._target_charset("Abba!") == {"a", "b"}
        assert T._target_charset("好 911") == {"好"}
    finally:
        T._inv_vocab, T._vocab_chars = old_vocab, old_chars


def test_one_sided_abstention_never_drops():
    """A drop needs EVERY existing side judged: when one side's probe
    abstains (None) and the other scores below floor, the word is kept +
    flagged — low support on one side is not evidence about the other."""
    gm = groups((0.0, 2.0), (2.5, 2.0))
    words = [
        {"t_s": 0.5, "t_e": 0.9, "seg_id": 0},
        {"t_s": 2.2, "t_e": 2.3, "seg_id": 1},
        {"t_s": 3.5, "t_e": 3.9, "seg_id": 2},
    ]
    r = _resolve_word_groups(
        words, gm,
        support_fn=lambda wi, gp, lo, hi: 0.2 if gp == 0 else None)
    assert r[1]["gpos"] is not None
    assert r[1]["flag"] == "pad_unresolved"


def test_degraded_never_drops_even_past_all_windows():
    """Degraded mode, word slopped past the LAST group's window with the
    last slot packed: still kept (clamped into the nearest group) — the
    'nothing is ever dropped without an aligner' invariant holds."""
    gm = groups((0.0, 2.0),)
    words = [
        {"t_s": 0.05, "t_e": 1.98, "seg_id": 0},   # packs the only group
        {"t_s": 2.6, "t_e": 2.7, "seg_id": 0},     # past win_hi (2.5)
    ]
    r = _resolve_word_groups(words, gm, support_fn=None)
    assert r[1]["gpos"] == 0
    assert r[1]["flag"] == "pad_unresolved"
    assert 0.0 <= r[1]["rel_s"] < r[1]["rel_e"] <= 2.0


def _word(word, start, end, flag=None):
    w = {"word": word, "start": start, "end": end, "probability": 0.9}
    if flag:
        w["pad_resolution"] = flag
    return w


def _sent(words):
    return {"text": " ".join(w["word"] for w in words),
            "start": words[0]["start"], "end": words[-1]["end"],
            "words": words}


def test_rehome_stranded_lead_word():
    """The live stem case (m405, 26.6 s stretch): 'You' pad_moved into a
    group holding NO clean sentence-mate while 'should have seen...' sit
    clean in a later group — re-homed to the majority group, placed before
    its anchors, flagged pad_rehomed, sentence span tightened."""
    grps = [{"start": 385.0, "end": 389.44, "audio": None},
            {"start": 413.0, "end": 416.5, "audio": None}]
    s = _sent([_word("You", 389.34, 389.44, flag="pad_moved"),
               _word("should", 413.44, 413.60),
               _word("have", 413.52, 413.68),
               _word("seen", 413.70, 413.86)])
    moved = rehome_straddle_strays([s], grps)
    assert moved == 1
    w = s["words"][0]
    assert w["pad_resolution"] == "pad_rehomed"
    assert 413.0 <= w["start"] < w["end"] <= 413.44, w
    assert s["start"] >= 413.0 and s["end"] - s["start"] < 2.0


def test_rehome_leaves_honest_cross_group_sentence():
    """'Pride ... before a fall': CLEAN words on both sides of a group gap
    is wtpsplit's honest call — byte-identical, zero moved."""
    grps = [{"start": 10.0, "end": 12.0, "audio": None},
            {"start": 15.0, "end": 18.0, "audio": None}]
    s = _sent([_word("Pride", 11.5, 11.9),
               _word("before", 15.1, 15.4),
               _word("a", 15.4, 15.5),
               _word("fall.", 15.5, 15.9)])
    before = json.loads(json.dumps(s))
    assert rehome_straddle_strays([s], grps) == 0
    assert s == before


def test_rehome_respects_sentence_mates_in_same_group():
    """A flagged word whose group ALSO holds a clean sentence-mate is not
    stranded — sentence-mates support the placement, so it stays."""
    grps = [{"start": 10.0, "end": 12.0, "audio": None},
            {"start": 15.0, "end": 18.0, "audio": None}]
    s = _sent([_word("Well,", 11.0, 11.2, flag="pad_moved"),
               _word("yes.", 11.3, 11.6),
               _word("I", 15.1, 15.2),
               _word("agree.", 15.2, 15.6),
               _word("Fully.", 15.7, 16.0)])
    assert rehome_straddle_strays([s], grps) == 0
    assert s["words"][0]["start"] == 11.0


def test_rehome_trailing_stray():
    """A flagged word stranded AFTER its sentence's clean anchors lands
    just after the right edge, inside the majority group's audio."""
    grps = [{"start": 10.0, "end": 14.0, "audio": None},
            {"start": 60.0, "end": 61.0, "audio": None}]
    s = _sent([_word("See", 10.5, 10.8),
               _word("you", 10.8, 11.0),
               _word("later.", 60.1, 60.3, flag="pad_unresolved")])
    assert rehome_straddle_strays([s], grps) == 1
    w = s["words"][2]
    assert w["pad_resolution"] == "pad_rehomed"
    assert 11.0 <= w["start"] < w["end"] <= 14.0, w
    assert s["end"] <= 14.0


def test_rehome_bounded_squeeze_when_group_packed():
    """Anchors starting at the group edge (the stray IS the group's true
    first word): the NEAREST stray squeezes to the edge with anchor overlap
    capped at PAD_MIN_WORD_SEC; the farther stray declines untouched —
    bounded, never stacked."""
    grps = [{"start": 5.0, "end": 5.6, "audio": None},
            {"start": 10.0, "end": 11.6, "audio": None}]
    s = _sent([_word("Oh,", 5.1, 5.3, flag="pad_moved"),
               _word("you", 5.3, 5.5, flag="pad_moved"),
               _word("bet", 10.05, 10.6),
               _word("I", 10.6, 10.8),
               _word("will.", 10.8, 11.0)])
    assert rehome_straddle_strays([s], grps) == 1
    near, far = s["words"][1], s["words"][0]
    assert near["pad_resolution"] == "pad_rehomed"
    assert near["start"] == 10.0 and near["end"] - 10.05 <= 0.1 + 1e-6
    assert (far["start"], far["end"]) == (5.1, 5.3)   # untouched
    assert far["pad_resolution"] == "pad_moved"       # original flag kept


def test_rehome_between_anchors_stays_in_majority_group():
    """A stray textually between clean anchors is clamped between its
    MAJORITY-group neighbours — never anchored on a clean sentence-mate
    from a different group (which would place it outside the group)."""
    grps = [{"start": 10.0, "end": 11.0, "audio": None},
            {"start": 54.0, "end": 56.0, "audio": None},
            {"start": 100.0, "end": 110.0, "audio": None}]
    s = _sent([_word("Now", 100.5, 100.9),
               _word("look,", 10.1, 10.4),          # clean, minority group
               _word("um,", 55.0, 55.2, flag="pad_unresolved"),
               _word("stop.", 109.0, 109.4)])
    assert rehome_straddle_strays([s], grps) == 1
    w = s["words"][2]
    assert w["pad_resolution"] == "pad_rehomed"
    assert 100.9 <= w["start"] < w["end"] <= 109.0, w


def test_rehome_respects_other_sentences_occupancy():
    """A rehomed stray must not double-book audio that belongs to ANOTHER
    sentence occupying the majority group just before the anchors — the
    occupant's end is the placement floor."""
    grps = [{"start": 5.0, "end": 5.6, "audio": None},
            {"start": 10.0, "end": 12.0, "audio": None}]
    other = _sent([_word("Done.", 10.05, 10.55)])
    s = _sent([_word("So", 5.1, 5.3, flag="pad_moved"),
               _word("anyway,", 10.6, 11.0),
               _word("go.", 11.0, 11.3)])
    assert rehome_straddle_strays([other, s], grps) == 1
    w = s["words"][0]
    assert w["pad_resolution"] == "pad_rehomed"
    assert w["start"] >= 10.55 - 1e-6, w     # never under the occupant
    assert w["end"] <= 10.6 + 0.1 + 1e-6     # anchor overlap stays bounded
    assert (other["words"][0]["start"], other["words"][0]["end"]) \
        == (10.05, 10.55)                    # occupant untouched


def test_rehome_straddling_occupant_bounds():
    """An occupant from another sentence STRADDLING the anchor edge still
    bounds the placement (capped at the edge): the stray never lands under
    the occupant's body, and any residual overlap stays within the 0.1 s
    squeeze."""
    grps = [{"start": 5.0, "end": 5.6, "audio": None},
            {"start": 10.0, "end": 12.0, "audio": None}]
    other = _sent([_word("...sir.", 10.3, 10.65)])   # straddles the edge
    s = _sent([_word("So", 5.1, 5.3, flag="pad_moved"),
               _word("anyway,", 10.6, 11.0),
               _word("go.", 11.0, 11.3)])
    assert rehome_straddle_strays([other, s], grps) == 1
    w = s["words"][0]
    assert w["pad_resolution"] == "pad_rehomed"
    assert w["start"] >= 10.6 - 1e-6, w       # capped at the anchor edge
    assert w["end"] - w["start"] <= 0.1 + 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
