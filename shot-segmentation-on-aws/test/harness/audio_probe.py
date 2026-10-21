#!/usr/bin/env python3
"""ONE-OFF: does audio identify the programme boundaries better than picture?

Asked because textless elements are nominally "no text and no audio", so a
loudness/silence pass looked like a cheap, definitional discriminator. It is
NOT, and this script is the evidence for that decision (see README.md 9.7):

  cooking_a (_dualpurpose_, 35:13)
    TONE  00:00-00:30   -15.0 dBFS  1000.0 Hz  100% energy in +-20Hz   <- line-up tone
    SIL   00:30-01:00  -180.0 dBFS  (bit-exact zero)
    PROG  01:00-30:03
    SIL   30:03-31:21  -180.0 dBFS  (78s moat)
    PROG  31:21-32:08   -25.7 dBFS  350 Hz  18% conc   <- TEXTLESS BLOCK, carries M&E
    SIL   32:08-35:13  -180.0 dBFS

The textless title block is audio-indistinguishable from programme (-25.7 vs
-23.7 dBFS, flatness 1.0e-04 vs 4.1e-04) because a textless title sequence
ships WITH its music bed. So "programme_end = last non-silent audio" lands at
32:08 and keeps the whole block inside the programme — worse than picture.

Meanwhile the 78s silent moat is already published as a black shot (#525,
30:02-31:20), so audio adds nothing there. Conclusion: shot-segmentation stays
picture-only; the tail boundary comes from the black moat (see textless.py).

The one place audio WOULD add signal is the head: cooking_a shots #0 (29.92s
bars+tone) and #1 (27.04s slate) are both classified "content", so 57s of
leader is invisible to shot_type. That needs a 1kHz tone test, which belongs
to whatever consumes audio — not here.

Requires ffmpeg on PATH and a local copy of the asset (proxy mp4 is enough).
  python audio_probe.py cooking_a.mp4
"""
import subprocess
import sys

import numpy as np

SR = 16000
SILENCE_DBFS = -50.0     # below this = silence (true digital silence reads -180)
TONE_CONC = 0.80         # fraction of energy within +-25Hz of the spectral peak
MIN_RUN_SEC = 3


def decode_mono(path):
    """Decode the whole audio track to mono float32 at SR via ffmpeg."""
    raw = subprocess.run(  # nosec B603
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn",
         "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def per_second(a):
    """Per-second RMS (dBFS), tone concentration, and spectral peak (Hz)."""
    n = len(a) // SR
    seg = a[:n * SR].reshape(n, SR)
    db = 20 * np.log10(np.maximum(np.sqrt((seg ** 2).mean(1)), 1e-9))
    win = np.hanning(SR)
    freq = np.fft.rfftfreq(SR, 1 / SR)
    conc = np.zeros(n)
    peak = np.zeros(n)
    for i in range(n):
        p = np.abs(np.fft.rfft(seg[i] * win)) ** 2 + 1e-20
        k = int(np.argmax(p[20:])) + 20          # ignore DC/rumble
        band = (freq > freq[k] - 25) & (freq < freq[k] + 25)
        conc[i] = p[band].sum() / p.sum()
        peak[i] = freq[k]
    return db, conc, peak


def main(path):
    db, conc, peak = per_second(decode_mono(path))
    lab = np.where(db < SILENCE_DBFS, "SIL",
                   np.where(conc > TONE_CONC, "TONE", "PROG"))
    mmss = lambda s: f"{int(s) // 60:02d}:{int(s) % 60:02d}"
    runs, start = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            runs.append((lab[start], start, i))
            start = i
    for kind, a0, b0 in runs:
        if b0 - a0 < MIN_RUN_SEC:
            continue
        print(f"  {kind:5s} {mmss(a0)} -> {mmss(b0)}  ({b0 - a0:4d}s)  "
              f"medianRMS {np.median(db[a0:b0]):7.1f} dBFS  "
              f"medianPeak {np.median(peak[a0:b0]):6.0f} Hz  "
              f"medianConc {np.median(conc[a0:b0]):.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
