# test/test_wav_resample.py
"""Anti-aliased resampling and windowed conversion continuity.

The 2026-09-28 lesson: the old linear-interp resampler folded >8 kHz
content into the speech band on 48 kHz inputs, and the whole pipeline
consumed the aliased audio. Run: python3 test/test_wav_resample.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.wav_utils import _resample  # noqa: E402


def test_downsample_rejects_above_target_nyquist():
    # 10 kHz tone at 48 kHz is ABOVE the 16 kHz target's Nyquist: a
    # correct resampler removes it; linear interp folded it to 6 kHz
    sr, dur = 48_000, 3.0
    t = np.arange(int(sr * dur)) / sr
    tone = np.sin(2 * np.pi * 10_000 * t).astype(np.float32)
    out = _resample(tone, sr, 16_000)
    assert len(out) == int(len(tone) * 16_000 / 48_000)
    ratio = np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(tone ** 2))
    assert ratio < 0.02, f"aliasing leaked: rms ratio {ratio:.3f}"


def test_downsample_preserves_speech_band():
    sr = 48_000
    t = np.arange(sr * 3) / sr
    tone = np.sin(2 * np.pi * 1_000 * t).astype(np.float32)
    out = _resample(tone, sr, 16_000)
    ref = np.sin(2 * np.pi * 1_000 * np.arange(len(out)) / 16_000)
    # ignore filter edges
    a, b = 1_000, len(out) - 1_000
    ratio = (np.sqrt(np.mean(out[a:b] ** 2))
             / np.sqrt(np.mean(ref[a:b] ** 2)))
    assert 0.95 < ratio < 1.05, ratio
    assert len(out) == int(len(tone) / 3)


def test_windowed_conversion_is_continuous():
    """Parts converted independently must agree with a whole-signal
    resample away from the extreme edges (the FIR pad/trim contract)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "states", "check_and_preprocess"))
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "states",
                        "check_and_preprocess", "wav_to_wav.py")
    spec = importlib.util.spec_from_file_location("w2w", path)
    w2w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w2w)

    sr, dur = 48_000, 30.0
    # steady 1 kHz tone: any filter-edge transient or discontinuity at
    # a part join shows up as an amplitude dip/spike in the envelope
    t = np.arange(int(sr * dur)) / sr
    sig = (0.5 * np.sin(2 * np.pi * 1_000 * t)).astype(np.float32)

    info = types.SimpleNamespace(sample_rate=sr, duration_sec=dur,
                                 num_channels=1, bytes_per_frame=2,
                                 data_offset=44)

    def fake_window(bucket, key, inf, start_sec, duration_sec,
                    target_sr=16_000):
        a = int(max(0.0, start_sec) * sr)
        b = int(min(start_sec + duration_sec, dur) * sr)
        return _resample(sig[a:b], sr, target_sr)

    w2w.read_wav_window = fake_window
    parts = []
    for i in range(3):
        pcm = w2w._read_and_encode("b", "k", info,
                                   start_sec=i * 10.0, duration_sec=10.0,
                                   target_sr=16_000)
        parts.append(np.frombuffer(pcm, dtype=np.int16))
    got = np.concatenate(parts).astype(np.float32) / 32_767.0
    assert abs(len(got) - int(len(sig) / 3)) <= 3, (len(got), len(sig) // 3)
    # 4 ms local RMS envelope must stay flat through both joins
    w = 64
    env = np.sqrt(np.convolve(got ** 2, np.ones(w) / w, mode="valid"))
    interior = env[w:-w]
    a0 = 0.5 / np.sqrt(2)                      # RMS of a 0.5 sine
    assert interior.min() > 0.95 * a0, interior.min() / a0
    assert interior.max() < 1.05 * a0, interior.max() / a0


if __name__ == "__main__":
    test_downsample_rejects_above_target_nyquist()
    test_downsample_preserves_speech_band()
    test_windowed_conversion_is_continuous()
    print("all wav-resample tests passed")
