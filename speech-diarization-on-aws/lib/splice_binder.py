# lib/splice_binder.py
"""
Splice binder — cross-chunk bind candidates judged by the segmentation
model itself (user design, 2026-09-29/30).

Chunk marks clip every voice speaking across them; the continuity
binder ([lib/powerset_guides.py] _bind_across_chunks) only rescues the
mid-speech straddle. For everything else, this module asks pyannote
directly: take a clean 4.5 s solo span from a track on each side of
the boundary, joint them with 1 s of silence into one 10 s window
(exactly the model's receptive field, splice at dead center), and read
the powerset — ONE sustained solo slot across the pause means the
model heard one person; a second slot means two.

Validated offline before this module existed (13-asset corpus, both
the healthy 2026-09-29 corpus and the wav-fed knife-edge corpus of
2026-09-30):
  - calibration on 196 free-ground-truth controls: verdicts are nearly
    binary (SAME median 1.000 / DIFF median 0.000), and the 1 s
    silence layout beats a crossfade splice (wrongly-bound 25%->18.5%);
  - NEITHER detector may act alone: pure-splice binding welded voices
    at embedding sim +0.004 (retention 0.961->0.507) because the
    powerset tracks conversational continuity, not identity — two
    same-register non-interacting voices read as one slot; embeddings
    alone weld same-register family voices at sim 0.909. Their blind
    regimes are complementary, so a bind requires BOTH to agree;
  - the row/column consistency veto (a side track with more than one
    one-slot verdict, or a track claimed twice) removes 53/69 of the
    pure-splice blind welds before the embedding gate even runs;
  - same-chunk binding is REFUTED (within-chunk pairs lack the
    boundary's structural prior and hit the probe's blind regime —
    retention 0.662->0.266): this module probes ADJACENT-CHUNK pairs
    only;
  - decisive measured win: on a flat silhouette curve (max at k=2,
    k=7..17 within 0.002 of each other) no K-selection rule can find
    the right landing, but ONE accepted bind changed the units under
    the curve and moved the sweep k=10->16 (retention 0.236->0.662).

Candidates only — the clustering state's existing voice-gated
application in _cluster_tracks stays the single authority, exactly
like the continuity binder's candidates.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from lib.model_registry import PYANNOTE_MODEL
from lib.powerset_guides import subtract_spans
from lib.s3_utils import get_json, s3 as _s3

log = logging.getLogger(__name__)

# ── probe constants (module-owned: nothing outside this file reads
#    them; the offline calibration above is their rationale) ──────────
SPAN_SEC = 4.5          # solo speech per side (2x4.5 + 1 = one window)
GAP_SEC = 1.0           # silence separator — a natural pause, which the
                        # model treats as a speaker-change opportunity;
                        # holding one slot across it is the evidence
FADE_SEC = 0.010        # declick into the silence
TOP_TRACKS = 3          # stable tracks per side (<=3x3 probes/boundary)
SLOT_MIN_SEC = 0.5      # a slot "exists" above this presence (flicker
                        # cannot manufacture a second speaker)
BIND_SIM = 0.80         # pooled-chain embedding agreement — stricter
                        # than the continuity binder's 0.70 because a
                        # probe bind has no structural straddle evidence
_SR = 16_000
_SOLO = (1, 2, 3)       # powerset solo classes

_session = None


def _get_session(model_bucket: str, model_prefix: str,
                 memory_mb: int | None):
    """Same bootstrap as the VAD state (registry tarball -> /tmp ->
    ORT session cached across warm invocations)."""
    global _session
    if _session is not None:
        return _session
    import onnxruntime as ort

    from lib.model_store import ensure_in_tmp, find_file
    from lib.ort_init import get_num_cpus
    s3_key = PYANNOTE_MODEL["s3_key"]
    if model_prefix and not s3_key.startswith(model_prefix):
        s3_key = model_prefix.strip("/") + "/" + s3_key.split("/", 1)[1]
    ensure_in_tmp(bucket=model_bucket, s3_key=s3_key,
                  tmp_dir=PYANNOTE_MODEL["tmp_dir"],
                  label="pyannote/community-1 (splice binder)")
    onnx_path = find_file(PYANNOTE_MODEL["tmp_dir"], "segmentation*.onnx")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = get_num_cpus(memory_mb)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    _session = ort.InferenceSession(
        onnx_path, opts, providers=["CPUExecutionProvider"])
    return _session


def _fetch_docs(bucket: str, output_prefix: str) -> dict:
    """{true_chunk_idx: powerset doc} under the run prefix."""
    from lib.powerset_guides import run_prefix_of
    root = run_prefix_of(output_prefix)
    docs, token = {}, None
    while True:
        kw = {"Bucket": bucket, "Prefix": f"{root}/powerset/"}
        if token:
            kw["ContinuationToken"] = token
        resp = _s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            try:
                idx = int(o["Key"].rsplit("_", 1)[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            docs[idx] = get_json(bucket, o["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return docs


def _read_span(bucket: str, key: str, info, start_sec: float):
    from lib.wav_utils import read_wav_window
    return read_wav_window(bucket, key, info, start_sec, SPAN_SEC,
                           target_sr=_SR)


def _splice(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    n = int(SPAN_SEC * _SR)
    fade = int(FADE_SEC * _SR)
    a, b = a[:n].copy(), b[:n].copy()
    if len(a) < n or len(b) < n:
        return None
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    a[-fade:] *= ramp[::-1]
    b[:fade] *= ramp
    return np.concatenate(
        [a, np.zeros(int(GAP_SEC * _SR), dtype=np.float32), b]
    ).astype(np.float32)


def _one_slot(session, window: np.ndarray) -> bool:
    """Does the model hear ONE speaker across the spliced window?"""
    inp = window[np.newaxis, np.newaxis, :].astype(np.float32)
    name = session.get_inputs()[0].name
    raw = session.run(["logits"], {name: inp})[0][0]     # [F, 7]
    e = np.exp(raw - raw.max(-1, keepdims=True))
    cls = (e / e.sum(-1, keepdims=True)).argmax(-1)
    fdur = (2 * SPAN_SEC + GAP_SEC) / len(cls)
    need = int(round(SLOT_MIN_SEC / fdur))
    slots = sum(1 for c in _SOLO if int(np.sum(cls == c)) >= need)
    return slots == 1


def _stable_sides(doc, nearest_end: bool):
    """Top TOP_TRACKS tracks of one chunk with a clean solo span >=
    SPAN_SEC (overlap spans excised), span nearest the boundary."""
    ovl = doc.get("overlap_spans") or []
    out = []
    for t in doc.get("tracks") or []:
        spans = []
        for a, b in t["segs"]:
            for s0, s1 in subtract_spans(a, b, ovl):
                if s1 - s0 >= SPAN_SEC:
                    spans.append((s0, s1))
        if not spans:
            continue
        spans.sort(key=lambda p: -p[1] if nearest_end else p[0])
        dur = sum(b - a for a, b in t["segs"])
        out.append((dur, t["id"], spans[0]))
    out.sort(reverse=True)
    return out[:TOP_TRACKS]


def _chain_vecs(track_segs) -> dict:
    by: dict = {}
    for seg in track_segs:
        by.setdefault(seg["chain"], []).append(seg)
    out = {}
    for ch, segs in by.items():
        ws = np.array([s["duration"] for s in segs])
        vs = np.array([s["embedding"] for s in segs])
        v = np.average(vs, axis=0, weights=np.maximum(ws, 1e-6))
        n = np.linalg.norm(v)
        if n > 1e-10:
            out[ch] = v / n
    return out


def probe_bind_candidates(
    output_bucket: str,
    output_prefix: str,
    audio_bucket: str,
    audio_key: str,
    model_bucket: str,
    model_prefix: str,
    track_segs: list,
    existing: dict | None = None,
    memory_mb: int | None = None,
) -> tuple[dict, dict]:
    """-> ({right_chain: left_chain} additional bind candidates, diag).

    Chain ids are namespaced by TRUE chunk index, matching load_guides
    and the embeddings docs. `existing` = the continuity binder's
    candidates; pairs it already covers are not probed (structural
    evidence outranks the probe)."""
    t0 = time.monotonic()
    diag = {"n_boundaries": 0, "n_probed": 0, "n_one_slot": 0,
            "n_vetoed_rowcol": 0, "n_emb_rejected": 0, "n_binds": 0}
    existing = existing or {}
    if not (audio_bucket and audio_key and model_bucket):
        diag["skipped"] = "missing audio or model location"
        return {}, diag
    docs = _fetch_docs(output_bucket, output_prefix)
    if len(docs) < 2:
        diag["skipped"] = "single chunk"
        return {}, diag
    vecs = _chain_vecs(track_segs)
    from lib.wav_utils import get_wav_info
    info = get_wav_info(audio_bucket, audio_key)
    session = _get_session(model_bucket, model_prefix, memory_mb)

    out: dict = {}
    idxs = sorted(docs)
    for ki, idx in enumerate(idxs[:-1]):
        nxt = idxs[ki + 1]
        if nxt != idx + 1:
            continue                     # non-adjacent: no boundary
        left = _stable_sides(docs[idx], nearest_end=True)
        right = _stable_sides(docs[nxt], nearest_end=False)
        if not left or not right:
            continue
        diag["n_boundaries"] += 1
        verdicts = []                    # (cl, cr, one_slot)
        for dl, tl, pl in left:
            cl = (idx + 1) * 1_000_000 + tl
            audio_l = None
            for dr, tr, pr in right:
                cr = (nxt + 1) * 1_000_000 + tr
                if cr in existing or cl not in vecs or cr not in vecs:
                    continue             # continuity-bound or no voice
                if audio_l is None:
                    audio_l = _read_span(audio_bucket, audio_key, info,
                                         max(pl[0], pl[1] - SPAN_SEC))
                window = _splice(audio_l,
                                 _read_span(audio_bucket, audio_key,
                                            info, pr[0]))
                if window is None:
                    continue
                diag["n_probed"] += 1
                verdicts.append((cl, cr, _one_slot(session, window)))
        # embedding gate FIRST, then the row/column consistency veto
        # over the survivors — the exact order the offline referees
        # validated (a one-slot verdict that embeddings already refute
        # is the probe's known blind regime, not a real competing
        # claim). A surviving side with >1 one-slot verdict then
        # contradicts itself (or is its own fragment group — either
        # way, abstain: a wrong bind is the one unrecoverable error).
        diag["n_one_slot"] += sum(1 for _, _, one in verdicts if one)
        pos = []
        for cl, cr, one in verdicts:
            if not one:
                continue
            if float(vecs[cl] @ vecs[cr]) < BIND_SIM:
                diag["n_emb_rejected"] += 1
                continue
            pos.append((cl, cr))
        lc: dict = {}
        rc: dict = {}
        for cl, cr in pos:
            lc[cl] = lc.get(cl, 0) + 1
            rc[cr] = rc.get(cr, 0) + 1
        for cl, cr in pos:
            if lc[cl] != 1 or rc[cr] != 1:
                diag["n_vetoed_rowcol"] += 1
                continue
            out[cr] = out.get(cl, cl)    # transitive across boundaries
            diag["n_binds"] += 1
    diag["wall_sec"] = round(time.monotonic() - t0, 2)
    return out, diag
