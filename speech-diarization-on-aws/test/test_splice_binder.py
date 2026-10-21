# test/test_splice_binder.py
"""Splice binder — the probe policy on a stubbed model and patched
I/O. The stub encodes voice identity in the audio marker value: the
fake session reads the marker of each half of the spliced window and
answers one-slot iff both halves carry the same voice id (mirroring
the real model's saturated verdicts — calibration medians 1.000/0.000).
Run: python3 test/test_splice_binder.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lib.splice_binder as sb  # noqa: E402

SR = sb._SR
F = 100



# Silence console log HANDLERS for the whole suite: the failure-path
# tests exercise refusal/retry/fallback branches whose designed
# behavior includes ERROR/WARNING logs, and those lines made build
# logs read as if the build were failing. Handlers only — record flow
# is untouched, so assertLogs() still sees everything.
import logging as _logging


def _silence_console_logging():
    _logging.getLogger().handlers = [_logging.NullHandler()]
    for _name in list(_logging.root.manager.loggerDict):
        _lg = _logging.getLogger(_name)
        if _lg.handlers:
            _lg.handlers = [_logging.NullHandler()]


_silence_console_logging()


class _StubInput:
    name = "x"


class _StubSession:
    """logits [1,F,7]: one class-1 window when both halves share a
    voice marker; class-1 first half / class-2 second half otherwise."""

    def get_inputs(self):
        return [_StubInput()]

    def run(self, outs, feeds):
        x = feeds["x"][0, 0]
        v1, v2 = float(x[10]), float(x[-10])
        logits = np.full((1, F, 7), -10.0, dtype=np.float32)
        if v1 == v2:
            logits[0, :, 1] = 0.0
        else:
            logits[0, : F // 2, 1] = 0.0
            logits[0, F // 2:, 2] = 0.0
        return [logits]


def _voice(vid):
    """A 'span' of constant marker value = one voice's audio."""
    return np.full(int(sb.SPAN_SEC * SR) + 100, float(vid),
                   dtype=np.float32)


class _Info:
    duration_sec = 3600.0


def _setup(monkey_voices, chunk_tracks, embeddings, existing=None):
    """monkey_voices: {(chunk_idx, track_id): voice_id};
    chunk_tracks: {chunk_idx: [(track_id, dur, seg_start)]};
    embeddings: {namespaced_chain: unit vector}."""
    docs = {}
    for idx, tracks in chunk_tracks.items():
        docs[idx] = {"overlap_spans": [],
                     "tracks": [{"id": tid,
                                 "segs": [[t0, t0 + dur]]}
                                for tid, dur, t0 in tracks]}
    sb._fetch_docs = lambda b, p: docs
    span_of = {}
    for idx, tracks in chunk_tracks.items():
        for tid, dur, t0 in tracks:
            span_of[round(max(t0, t0 + dur - sb.SPAN_SEC), 3)] = \
                monkey_voices[(idx, tid)]
            span_of[round(t0, 3)] = monkey_voices[(idx, tid)]
    sb._read_span = (lambda b, k, i, t:
                     _voice(span_of[round(t, 3)]))
    sb._get_session = lambda *a, **kw: _StubSession()
    import lib.wav_utils as wu
    wu.get_wav_info = lambda b, k: _Info()
    track_segs = [{"chain": ch, "duration": 10.0,
                   "embedding": v.tolist()}
                  for ch, v in embeddings.items()]
    return sb.probe_bind_candidates(
        output_bucket="b", output_prefix="p",
        audio_bucket="ab", audio_key="ak",
        model_bucket="mb", model_prefix="_model_artefacts",
        track_segs=track_segs, existing=existing or {})


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=32)
    return v / np.linalg.norm(v)


def test_binds_when_probe_and_embedding_agree():
    ea, eb = _unit(1), _unit(2)
    cands, diag = _setup(
        {(0, 5): 100, (0, 6): 200, (1, 7): 100, (1, 8): 200},
        {0: [(5, 50.0, 0.0), (6, 40.0, 100.0)],
         1: [(7, 50.0, 720.0), (8, 40.0, 800.0)]},
        {1_000_005: ea, 1_000_006: eb,
         2_000_007: ea, 2_000_008: eb})
    # voice 100: track 5 -> 7; voice 200: track 6 -> 8; both unique
    # rows/columns, both embedding-agreeing
    assert cands == {2_000_007: 1_000_005, 2_000_008: 1_000_006}, cands
    assert diag["n_binds"] == 2 and diag["n_vetoed_rowcol"] == 0, diag


def test_row_contradiction_vetoes():
    ea = _unit(3)
    # left track 5 hears one-slot with BOTH right tracks (7 and 8
    # carry the same voice marker as 5): contradiction -> no bind
    cands, diag = _setup(
        {(0, 5): 100, (1, 7): 100, (1, 8): 100},
        {0: [(5, 50.0, 0.0)],
         1: [(7, 50.0, 720.0), (8, 40.0, 800.0)]},
        {1_000_005: ea, 2_000_007: ea, 2_000_008: ea})
    assert cands == {}, cands
    assert diag["n_vetoed_rowcol"] == 2, diag


def test_embedding_gate_rejects_probe_only_agreement():
    # probe says one voice, embeddings say different (the news-anchor
    # blind regime): no bind
    cands, diag = _setup(
        {(0, 5): 100, (1, 7): 100},
        {0: [(5, 50.0, 0.0)], 1: [(7, 50.0, 720.0)]},
        {1_000_005: _unit(4), 2_000_007: _unit(5)})
    assert cands == {}, cands
    assert diag["n_emb_rejected"] == 1, diag


def test_two_slot_verdict_means_no_bind():
    ea = _unit(6)
    # embeddings identical but the model hears TWO speakers (family
    # regime): no bind
    cands, diag = _setup(
        {(0, 5): 100, (1, 7): 200},
        {0: [(5, 50.0, 0.0)], 1: [(7, 50.0, 720.0)]},
        {1_000_005: ea, 2_000_007: ea})
    assert cands == {}, cands
    assert diag["n_one_slot"] == 0 and diag["n_probed"] == 1, diag


def test_continuity_covered_pair_is_not_probed():
    ea = _unit(7)
    cands, diag = _setup(
        {(0, 5): 100, (1, 7): 100},
        {0: [(5, 50.0, 0.0)], 1: [(7, 50.0, 720.0)]},
        {1_000_005: ea, 2_000_007: ea},
        existing={2_000_007: 1_000_005})
    assert cands == {} and diag["n_probed"] == 0, (cands, diag)


def test_missing_audio_or_model_degrades_to_nothing():
    cands, diag = sb.probe_bind_candidates(
        output_bucket="b", output_prefix="p",
        audio_bucket=None, audio_key=None, model_bucket=None,
        model_prefix="_model_artefacts", track_segs=[])
    assert cands == {} and "skipped" in diag, diag


if __name__ == "__main__":
    test_binds_when_probe_and_embedding_agree()
    test_row_contradiction_vetoes()
    test_embedding_gate_rejects_probe_only_agreement()
    test_two_slot_verdict_means_no_bind()
    test_continuity_covered_pair_is_not_probed()
    test_missing_audio_or_model_degrades_to_nothing()
    print("all splice-binder tests passed")
