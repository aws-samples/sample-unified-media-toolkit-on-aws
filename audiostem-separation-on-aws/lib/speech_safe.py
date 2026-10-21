# lib/speech_safe.py
"""
Gated mix substitution — the speech_safe output (README.md Design record §5).

The separation model deletes low-prominence speech outright (whispers,
phone-futzed voices, background talkers — a training-distribution
property of the model, measured: no inference-time knob recovers it).
speech_safe guarantees the delivered track never hands a consumer
silence where the original audio had sound: wherever the raw speech
stem is near-silent while the mix still carries energy, attenuated mix
is ADDED under the stem ("observation adding" — provably bounds
artifact damage; remixing attenuated background under separated
dialogue is the shape shipped by production dialogue-enhancement
systems). Everywhere else the stem passes through bit-identical.

The gate operates on the 48 kHz pre-resample arrays inside the Map
state, where the mix slice and the raw stem coexist sample-aligned —
zero extra inference or IO. Patch edges are crossfaded (moving-average
envelope) against clicks and pumping; the gate mask is edge-extended
before smoothing so a patch that continues across a chunk boundary
holds full level at the seam.

Every knob is an ENV constant, never a per-request parameter (it would
enter the workflow_id preimage — Design record §4's rule), and all of
them are recorded in the chunk metrics' params block. The defaults are
a priori engineering choices pinned to the measured deletion shape
(deleted spans sit at digital zero, i.e. far below -55 dBFS, while
surviving speech stays well above it); the caption-GT harness is the
calibration instrument if they ever need to move.
"""

from __future__ import annotations

import os

import numpy as np

from lib.audio_analysis import _FLOOR, EnvelopeAccumulator, mask_to_spans

# ── Knobs (ENV-overridable; recorded in params) ──────────────────────────────

SPEECH_SAFE_BIN_MS = int(os.environ.get("SPEECH_SAFE_BIN_MS", "50"))
# A stem bin below this is "deleted" — the measured deletions are digital
# zero, so anything near the floor is model deletion, not quiet speech.
SPEECH_SAFE_STEM_GATE_DBFS = float(
    os.environ.get("SPEECH_SAFE_STEM_GATE_DBFS", "-55.0"))
# ... and the mix must actually carry something worth patching in.
SPEECH_SAFE_MIX_FLOOR_DBFS = float(
    os.environ.get("SPEECH_SAFE_MIX_FLOOR_DBFS", "-60.0"))
# Mix gain inside a patch. The patch is the whole scene (speech + bed), so
# unity would burst full-level music into an otherwise clean stem; -12 dB
# is the constant-floor sketch's beta=0.75 expressed as a gate.
SPEECH_SAFE_PATCH_GAIN_DB = float(
    os.environ.get("SPEECH_SAFE_PATCH_GAIN_DB", "-12.0"))
# Gaps between patches shorter than this are bridged — a gate that
# toggles per-bin inside one deleted sentence pumps audibly.
SPEECH_SAFE_MERGE_GAP_MS = int(
    os.environ.get("SPEECH_SAFE_MERGE_GAP_MS", "250"))
SPEECH_SAFE_FADE_MS = int(os.environ.get("SPEECH_SAFE_FADE_MS", "50"))


def params_doc() -> dict:
    """The effective knob values, recorded verbatim into chunk metrics."""
    return {
        "bin_ms": SPEECH_SAFE_BIN_MS,
        "stem_gate_dbfs": SPEECH_SAFE_STEM_GATE_DBFS,
        "mix_floor_dbfs": SPEECH_SAFE_MIX_FLOOR_DBFS,
        "patch_gain_db": SPEECH_SAFE_PATCH_GAIN_DB,
        "merge_gap_ms": SPEECH_SAFE_MERGE_GAP_MS,
        "fade_ms": SPEECH_SAFE_FADE_MS,
    }


def _bin_rms_db(x: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """(per-bin RMS in dBFS, bin_samples) via the SAME accumulator the
    stem_activity lanes use — one source of truth for the binning,
    including the actual-count tail bin (zero-padding it would read the
    mix artificially quiet at the chunk tail and suppress a legitimate
    patch there). Fed in 1 s slices so the float64 transient is O(slice),
    not O(chunk), on the memory-tight inference Lambda."""
    acc = EnvelopeAccumulator(sample_rate, bin_ms=SPEECH_SAFE_BIN_MS)
    for i in range(0, x.size, sample_rate):
        acc.feed(x[i:i + sample_rate])
    rms = acc.finish()
    return 20.0 * np.log10(np.maximum(rms, _FLOOR)), acc.bin_samples


def _bridge_gaps(mask: np.ndarray, max_gap_bins: int) -> np.ndarray:
    """Set to True any False run of <= max_gap_bins BETWEEN True runs."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return out
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap_bins + 1:
            out[a + 1:b] = True
    return out


def build_speech_safe(
    mix:  np.ndarray,
    stem: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, list]:
    """speech_safe = stem + gain * mix * gate_envelope, clipped to [-1, 1].

    `mix` and `stem` are the sample-aligned 48 kHz float32 chunk arrays.
    Returns (float32 array, patch spans as [[start_ms, end_ms], ...]
    relative to the chunk). Beyond a fade width of every patch the
    output is the stem, bit-identical (the crossfade ramp extends half a
    fade outside the reported span on each side).
    Deliberately NO minimum patch span: a dropped short
    patch could be a word, and recovering deleted speech is the whole
    product — abstention over deletion.
    """
    n = min(mix.size, stem.size)
    mix = mix[:n]
    stem = stem[:n]
    if n == 0:
        return stem.astype(np.float32), []

    stem_db, bin_samples = _bin_rms_db(stem, sample_rate)
    mix_db, _ = _bin_rms_db(mix, sample_rate)

    mask = (stem_db < SPEECH_SAFE_STEM_GATE_DBFS) & \
           (mix_db >= SPEECH_SAFE_MIX_FLOOR_DBFS)
    max_gap_bins = max(0, int(round(SPEECH_SAFE_MERGE_GAP_MS
                                    / SPEECH_SAFE_BIN_MS)))
    mask = _bridge_gaps(mask, max_gap_bins)
    # No merge/min-span at the SPAN level: bridging already happened on
    # the mask (it shapes the audio), and dropping a short span could
    # drop a word — abstention over deletion. Span ends are clamped to
    # the chunk's real duration: the tail bin is partial, and a span
    # reported past the audio would desynchronize patched_sec from
    # duration_sec in the chunk metrics.
    duration_ms = int(round(n * 1000.0 / sample_rate))
    spans = [[s, min(e, duration_ms)]
             for s, e in mask_to_spans(mask, SPEECH_SAFE_BIN_MS,
                                       merge_gap_ms=0, min_span_ms=0)]
    if not spans:
        return stem.astype(np.float32), []

    # Per-sample 0/1 gate, then a moving-average crossfade. The gate is
    # edge-EXTENDED before smoothing so a patch that CONTINUES across a
    # Map-chunk boundary holds full level at the seam (the neighbor
    # chunk sees the same audio and gates the same way). Accepted
    # residual: a patch whose edge falls within half a fade of the seam
    # gets a step there instead of a fade — distinguishing "ends at the
    # seam" from "continues" would need the NEIGHBOR chunk's stem, i.e.
    # extra inference. The step is bounded at the patch gain (quiet bed
    # appearing/vanishing), and the common case is protected.
    gate = np.repeat(mask.astype(np.float32), bin_samples)[:n]
    k = max(1, int(round(sample_rate * SPEECH_SAFE_FADE_MS / 1000.0)))
    if k > 1:
        half = k // 2
        padded = np.pad(gate, (half, k - 1 - half), mode="edge")
        csum = np.cumsum(np.insert(padded.astype(np.float64), 0, 0.0))
        env = ((csum[k:] - csum[:-k]) / k).astype(np.float32)
    else:
        env = gate

    gain = float(10.0 ** (SPEECH_SAFE_PATCH_GAIN_DB / 20.0))
    safe = np.clip(stem + gain * mix * env, -1.0, 1.0).astype(np.float32)
    return safe, spans
