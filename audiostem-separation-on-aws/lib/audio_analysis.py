# lib/audio_analysis.py
"""
Sample-domain analysis for the two audio-analysis products
(README.md Design record §4):

    stem_waveforms.json   rendering — normalized RMS envelope per lane
    stem_activity.json    semantic  — activity spans per lane + line-up
                          tones measured on the ORIGINAL mix

Pure numpy on float32 mono arrays in [-1, 1] — the arrays this pipeline
already decodes. No ffmpeg, no scipy, no new dependency (the house rule
wav_utils.py states). Everything is deterministic DSP; nothing here
infers, and nothing here NAMES what a measurement means: whether a
1 kHz span is bars-and-tone is the fusion layer's verdict, made by
joining `tones[]` against the visual slate/bars shot_kind.

The mono input is intentional (the separation model requires it) and
does not impair tone detection — a coherent tone on L+R averages to the
same tone. Mono costs only channel ATTRIBUTION, so there is no
`channels` field. Known blind spot, accepted: a phase-INVERTED tone
(L = -R) cancels to silence under the downmix.

Every knob is an ENV constant, never a per-request parameter (it would
enter the workflow_id preimage), and EVERY tone/activity knob is
recorded in the product's `params` block so a rerun can tell what
produced it (2026-08-26 review: three tone gates were initially left
out — a params block that omits a knob that shaped the output lies).

Threshold provenance, stated plainly: the defaults below are a priori
engineering choices — n = 0 measured assets at the time of writing.
They are ENV-overridable and recorded in params precisely so the E2E
corpus can calibrate them; treat a silently-missed real line-up tone or
a dropped quiet dialogue span as a calibration bug, not a tuning taste.
"""

from __future__ import annotations

import math
import os

import numpy as np

# ── Knobs (ENV-overridable; recorded in params) ──────────────────────────────

BIN_MS = int(os.environ.get("ANALYSIS_BIN_MS", "50"))
ACTIVITY_THRESHOLD_DBFS = float(
    os.environ.get("ACTIVITY_THRESHOLD_DBFS", "-45.0"))
ACTIVITY_MIN_SPAN_MS = int(os.environ.get("ACTIVITY_MIN_SPAN_MS", "500"))
ACTIVITY_MERGE_GAP_MS = int(os.environ.get("ACTIVITY_MERGE_GAP_MS", "1000"))

# speech_stem_gaps lane (Design record §5): a bin is a GAP when the mix
# is active (ACTIVITY_THRESHOLD_DBFS) while the speech stem sits below
# this — the measured deletions are digital zero, so -55 separates model
# deletion from merely quiet speech. Tighter span knobs than activity:
# deleted words are shorter than dialogue spans.
SPEECH_STEM_GAP_THRESHOLD_DBFS = float(
    os.environ.get("SPEECH_STEM_GAP_THRESHOLD_DBFS", "-55.0"))
SPEECH_STEM_GAP_MIN_SPAN_MS = int(
    os.environ.get("SPEECH_STEM_GAP_MIN_SPAN_MS", "250"))
SPEECH_STEM_GAP_MERGE_MS = int(
    os.environ.get("SPEECH_STEM_GAP_MERGE_MS", "500"))

TONE_FRAME_MS = int(os.environ.get("TONE_FRAME_MS", "250"))
TONE_MIN_PURITY = float(os.environ.get("TONE_MIN_PURITY", "0.85"))
TONE_MIN_LEVEL_DBFS = float(os.environ.get("TONE_MIN_LEVEL_DBFS", "-45.0"))
TONE_NOMINALS_HZ = (400, 1000, 2000)
TONE_TOLERANCE = float(os.environ.get("TONE_TOLERANCE", "0.02"))
# A steady span shorter than this is a "pop" (the head 2-pop sync mark).
TONE_POP_MAX_MS = int(os.environ.get("TONE_POP_MAX_MS", "500"))
# A steady tone must hold its level: frame-to-frame std above this is
# tremolo/music, not line-up tone.
TONE_MAX_LEVEL_STD_DB = float(os.environ.get("TONE_MAX_LEVEL_STD_DB", "3.0"))

_FLOOR = 1e-6          # -120 dBFS — silence floor for log10


def params_doc() -> dict:
    """The effective knob values, recorded verbatim into both products."""
    return {
        "bin_ms": BIN_MS,
        "activity_threshold_dbfs": ACTIVITY_THRESHOLD_DBFS,
        "activity_min_span_ms": ACTIVITY_MIN_SPAN_MS,
        "activity_merge_gap_ms": ACTIVITY_MERGE_GAP_MS,
        "speech_stem_gap_threshold_dbfs": SPEECH_STEM_GAP_THRESHOLD_DBFS,
        "speech_stem_gap_min_span_ms": SPEECH_STEM_GAP_MIN_SPAN_MS,
        "speech_stem_gap_merge_ms": SPEECH_STEM_GAP_MERGE_MS,
        "tone_frame_ms": TONE_FRAME_MS,
        "tone_min_purity": TONE_MIN_PURITY,
        "tone_freqs_hz": list(TONE_NOMINALS_HZ),
        "tone_tolerance": TONE_TOLERANCE,
        "tone_min_level_dbfs": TONE_MIN_LEVEL_DBFS,
        "tone_pop_max_ms": TONE_POP_MAX_MS,
        "tone_max_level_std_db": TONE_MAX_LEVEL_STD_DB,
    }


# ── RMS envelope (streaming) ──────────────────────────────────────────────────

class EnvelopeAccumulator:
    """Per-lane streaming RMS envelope at fixed BIN_MS bins.

    feed() takes float32 mono chunks in [-1, 1] in stream order — chunk
    boundaries never align to bins, so a remainder carries between
    feeds. finish() returns per-bin RMS (linear, un-normalized: the
    activity threshold is absolute dBFS; normalization for rendering
    happens at serialization).
    """

    def __init__(self, sample_rate: int, bin_ms: int = None):
        self.sample_rate = int(sample_rate)
        self.bin_ms = int(bin_ms or BIN_MS)
        self.bin_samples = max(1, int(round(
            self.sample_rate * self.bin_ms / 1000.0)))
        self._carry = np.zeros(0, dtype=np.float32)
        self._mean_sq: list = []

    def feed(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        buf = np.concatenate([self._carry, x.astype(np.float32, copy=False)])
        n_bins = buf.size // self.bin_samples
        if n_bins:
            whole = buf[: n_bins * self.bin_samples]
            self._mean_sq.extend(
                (whole.astype(np.float64) ** 2)
                .reshape(n_bins, self.bin_samples).mean(axis=1).tolist())
        self._carry = buf[n_bins * self.bin_samples:]

    def finish(self) -> np.ndarray:
        if self._carry.size:
            self._mean_sq.append(
                float((self._carry.astype(np.float64) ** 2).mean()))
            self._carry = np.zeros(0, dtype=np.float32)
        return np.sqrt(np.asarray(self._mean_sq, dtype=np.float64))


def waveform_values(rms: np.ndarray, decimals: int = 3) -> list:
    """Peak-normalized rendering values (the POC viewer's convention)."""
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return [0.0] * int(rms.size)
    return [round(float(v) / peak, decimals) for v in rms]


# ── Activity spans ────────────────────────────────────────────────────────────

def mask_to_spans(mask: np.ndarray, bin_ms: int,
                  merge_gap_ms: int, min_span_ms: int) -> list:
    """[[start_ms, end_ms], ...] from a per-bin boolean mask — gaps
    under merge_gap_ms bridged, spans under min_span_ms dropped.
    Half-open ms ranges on bin edges."""
    spans: list = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append([start * bin_ms, i * bin_ms])
            start = None
    if start is not None:
        spans.append([start * bin_ms, int(mask.size) * bin_ms])

    merged: list = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= merge_gap_ms:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [s for s in merged if s[1] - s[0] >= min_span_ms]


def activity_spans(rms: np.ndarray, bin_ms: int = None,
                   threshold_dbfs: float = None) -> list:
    """[[start_ms, end_ms], ...] where the lane is above threshold."""
    bin_ms = int(bin_ms or BIN_MS)
    threshold = (ACTIVITY_THRESHOLD_DBFS
                 if threshold_dbfs is None else threshold_dbfs)
    if rms.size == 0:
        return []
    db = 20.0 * np.log10(np.maximum(rms, _FLOOR))
    return mask_to_spans(db >= threshold, bin_ms,
                         ACTIVITY_MERGE_GAP_MS, ACTIVITY_MIN_SPAN_MS)


def stem_gap_spans(original_rms: np.ndarray, stem_rms: np.ndarray,
                   bin_ms: int = None) -> list:
    """[[start_ms, end_ms], ...] where the ORIGINAL mix is active while
    the stem sits below SPEECH_STEM_GAP_THRESHOLD_DBFS — audio the
    speech stem does not cover (Design record §5). An UPPER BOUND on
    deleted speech: music/sfx-only passages qualify too; naming what a
    gap MEANS is the fusion layer's job. The two envelopes come from
    different sample rates but share the same bin_ms grid; a one-bin
    length difference from stream remainders is truncated away.
    Delivered waveform JSONs are peak-normalized, so this absolute
    comparison can only be made here, where the linear envelopes exist."""
    bin_ms = int(bin_ms or BIN_MS)
    n = min(original_rms.size, stem_rms.size)
    if n == 0:
        return []
    orig_db = 20.0 * np.log10(np.maximum(original_rms[:n], _FLOOR))
    stem_db = 20.0 * np.log10(np.maximum(stem_rms[:n], _FLOOR))
    mask = (orig_db >= ACTIVITY_THRESHOLD_DBFS) & \
           (stem_db < SPEECH_STEM_GAP_THRESHOLD_DBFS)
    return mask_to_spans(mask, bin_ms, SPEECH_STEM_GAP_MERGE_MS,
                         SPEECH_STEM_GAP_MIN_SPAN_MS)


# ── Line-up tone detection (original mix only) ───────────────────────────────

class ToneDetector:
    """Streaming detector: feed float32 mono at a FIXED sample rate, in
    stream order; finish() -> tones[] spans.

    Per TONE_FRAME_MS frame: Hann-windowed rFFT, dominant peak above
    50 Hz, purity = energy in the peak ±1 bin over total, level in
    dBFS. A frame is tonal when purity and level clear their gates;
    consecutive tonal frames within TONE_TOLERANCE of each other's
    frequency merge into one span. Frequency is refined by parabolic
    interpolation on log magnitudes, so a drifted 998.6 Hz tape tone
    reads as ~998.6, not as the nearest bin.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self.frame_samples = max(16, int(round(
            self.sample_rate * TONE_FRAME_MS / 1000.0)))
        self._carry = np.zeros(0, dtype=np.float32)
        self._frames: list = []          # (freq_hz | None, level_db)
        self._window = np.hanning(self.frame_samples)

    def feed(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        buf = np.concatenate([self._carry, x.astype(np.float32, copy=False)])
        n = buf.size // self.frame_samples
        for i in range(n):
            self._analyze(buf[i * self.frame_samples:
                              (i + 1) * self.frame_samples])
        self._carry = buf[n * self.frame_samples:]

    def _analyze(self, frame: np.ndarray) -> None:
        rms = float(np.sqrt((frame.astype(np.float64) ** 2).mean()))
        level_db = 20.0 * math.log10(max(rms, _FLOOR))
        if level_db < TONE_MIN_LEVEL_DBFS:
            self._frames.append((None, level_db))
            return
        spec = np.abs(np.fft.rfft(frame * self._window))
        power = spec.astype(np.float64) ** 2
        bin_hz = self.sample_rate / self.frame_samples
        lo = max(1, int(math.ceil(50.0 / bin_hz)))      # ignore DC/rumble
        if lo >= power.size - 1:
            self._frames.append((None, level_db))
            return
        peak = lo + int(np.argmax(power[lo:]))
        peak_energy = power[max(0, peak - 1): peak + 2].sum()
        total = power[lo:].sum()
        purity = float(peak_energy / total) if total > 0 else 0.0
        if purity < TONE_MIN_PURITY:
            self._frames.append((None, level_db))
            return
        # Parabolic interpolation on log magnitude around the peak bin.
        freq = peak * bin_hz
        if 0 < peak < spec.size - 1:
            a, b, c = np.log(np.maximum(spec[peak - 1: peak + 2], _FLOOR))
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                freq += 0.5 * float((a - c) / denom) * bin_hz
        if freq < 50.0:
            # Interpolation at the low edge can extrapolate below the
            # 50 Hz analysis floor (a -18.7 Hz "tone" was observed on the
            # sweep-B corpus) — below the floor is not a measurement.
            self._frames.append((None, level_db))
            return
        self._frames.append((freq, level_db))

    def finish(self) -> list:
        # The trailing partial frame is dropped: shorter than the
        # analysis window, its frequency estimate is not comparable.
        self._carry = np.zeros(0, dtype=np.float32)
        spans: list = []
        run: list = []                       # [(index, freq, level_db)]

        def flush():
            if not run:
                return
            start_ms = run[0][0] * TONE_FRAME_MS
            end_ms = (run[-1][0] + 1) * TONE_FRAME_MS
            freqs = np.asarray([f for _, f, _ in run])
            levels = np.asarray([l for _, _, l in run])
            duration = end_ms - start_ms
            kind = "pop" if duration <= TONE_POP_MAX_MS else "steady"
            # A steady tone holds its level; tremolo/music does not.
            if kind == "steady" and float(levels.std()) > TONE_MAX_LEVEL_STD_DB:
                return
            freq = float(np.median(freqs))
            nominal = None
            for nom in TONE_NOMINALS_HZ:
                if abs(freq - nom) <= nom * TONE_TOLERANCE:
                    nominal = nom
                    break
            # CALIBRATED on the 2026-08-26 sweep-B corpus (n=6 assets):
            # short pure notes in program music pass the purity gate by the
            # hundreds (267 null-nominal pops on one reality episode), while
            # every true line-up signal carried a nominal match. A pop is
            # reportable ONLY at a nominal frequency — a 2-pop IS 1 kHz by
            # definition. Steady spans keep null-nominal reporting: a
            # sustained pure tone is notable evidence either way.
            if kind == "pop" and nominal is None:
                return
            spans.append({
                "start_ms": int(start_ms), "end_ms": int(end_ms),
                "freq_hz": round(freq, 1), "nominal_hz": nominal,
                "kind": kind,
            })

        for i, (freq, level) in enumerate(self._frames):
            if freq is None:
                flush()
                run = []
                continue
            if run and (i != run[-1][0] + 1
                        or abs(freq - run[-1][1])
                        > run[-1][1] * TONE_TOLERANCE):
                flush()
                run = []
            run.append((i, freq, level))
        flush()
        return spans
