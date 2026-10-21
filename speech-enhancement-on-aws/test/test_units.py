# test/test_units.py
"""Hermetic tests for the speech-enhancement pure logic: input schema
validation, chunk alignment and iterator construction, WAV round-trips,
and the output-contract/label pin.

numpy only; no S3, no boto3, no model. The stress harness
(test/stress_test.py) deliberately does not match this discover pattern.

    python3 -m unittest discover -s test -p 'test_*.py'
"""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib import input_schema                               # noqa: E402
from lib.wav_utils import encode_wav_mono                  # noqa: E402
from states.fixed_chunking_for_speech_enhancement.lambda_function import (  # noqa: E402
    _align_chunk_size, _build_iterators,
)
from states.concatenate_enhanced_audio.lambda_function import (  # noqa: E402
    _parse_wav_pcm,
)


def _event(input_config=None, output_config=None):
    return {
        "uuid": "u-1",
        "input": {"s3uri": "s3://b/in.mp4",
                  "input_config": input_config},
        "output": {"s3uri": "s3://b/out",
                   "output_config": output_config},
    }


class TestInputSchema(unittest.TestCase):

    def test_defaults_are_merged(self):
        ev = input_schema.validate_and_normalize(_event())
        ic = ev["input"]["input_config"]
        oc = ev["output"]["output_config"]
        self.assertEqual(ic["model_id"], "deepfilternet3")
        self.assertEqual(ic["fixed_chunk_size_secs"],
                         input_schema.DEFAULT_CHUNK_SEC)
        self.assertEqual(oc["output_resampl_freq"],
                         input_schema.DEFAULT_OUT_FREQ)
        self.assertIsNone(oc["atten_lim_db"])

    def test_model_id_is_not_configurable(self):
        ev = input_schema.validate_and_normalize(
            _event(input_config={"model_id": "something_else"}))
        self.assertEqual(ev["input"]["input_config"]["model_id"],
                         "deepfilternet3")

    def test_chunk_size_below_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            input_schema.validate_and_normalize(
                _event(input_config={"fixed_chunk_size_secs": 5}))

    def test_resample_freq_is_validated(self):
        with self.assertRaises(ValueError):
            input_schema.validate_and_normalize(
                _event(output_config={"output_resampl_freq": 22050}))

    def test_atten_lim_db_range_and_types(self):
        for bad in (-1, 100.5, "20", True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                input_schema.validate_and_normalize(
                    _event(output_config={"atten_lim_db": bad}))
        for ok in (0, 100, 12.5, None):
            ev = input_schema.validate_and_normalize(
                _event(output_config={"atten_lim_db": ok}))
            self.assertEqual(ev["output"]["output_config"]["atten_lim_db"],
                             ok)

    def test_missing_required_fields_raise(self):
        for broken in ({"uuid": "u"},
                       {"uuid": "u", "input": {"s3uri": "s3://b/k"}},
                       {"uuid": "u", "input": {"s3uri": "nots3"},
                        "output": {"s3uri": "s3://b/k"}}):
            with self.assertRaises(ValueError):
                input_schema.validate_and_normalize(broken)


class TestChunkAlignment(unittest.TestCase):
    """fixed_chunk_size_secs → nearest multiple of 10 s, ties down,
    floor one alignment unit."""

    def test_alignment_table(self):
        for requested, expected in ((360, 360), (14, 10), (15, 10),
                                    (16, 20), (10, 10), (5, 10),
                                    (355, 350), (356, 360)):
            self.assertEqual(_align_chunk_size(requested), expected,
                             msg=f"requested={requested}")

    def test_iterator_boundaries_clamp_the_last_chunk(self):
        cfg = {"audio_bucket": "b", "audio_key": "k",
               "output_bucket": "ob", "output_prefix": "p/enhanced_chunks",
               "enhancement": {"output_resampl_freq": 16000,
                               "atten_lim_db": None}}
        its = _build_iterators(cfg, 360, 790.5)
        self.assertEqual([i["chunk_name"] for i in its],
                         ["chunk_000", "chunk_001", "chunk_002"])
        self.assertEqual(its[0]["fetch_start_sec"], 0.0)
        self.assertEqual(its[1]["fetch_start_sec"], 360.0)
        self.assertEqual(its[2]["fetch_end_sec"], 790.5)   # clamped, not padded
        for it in its:
            self.assertEqual(it["enhancement"]["output_resampl_freq"], 16000)
            self.assertEqual(it["output_prefix"], "p/enhanced_chunks")

    def test_short_file_is_one_chunk(self):
        cfg = {"audio_bucket": "b", "audio_key": "k",
               "output_bucket": "ob", "output_prefix": "p",
               "enhancement": {}}
        its = _build_iterators(cfg, 360, 42.0)
        self.assertEqual(len(its), 1)
        self.assertEqual(its[0]["fetch_end_sec"], 42.0)


class TestWavRoundTrip(unittest.TestCase):

    def test_encode_then_parse_recovers_pcm_and_rate(self):
        sr = 16_000
        audio = (0.25 * np.sin(2 * np.pi * 440.0 *
                               np.arange(sr) / sr)).astype(np.float32)
        blob = encode_wav_mono(audio, sr)
        pcm, parsed_sr = _parse_wav_pcm(blob, "mem")
        self.assertEqual(parsed_sr, sr)
        self.assertEqual(len(pcm), sr * 2)   # 16-bit mono, 1 s

    def test_non_riff_blob_is_refused(self):
        with self.assertRaises(ValueError):
            _parse_wav_pcm(b"\x00" * 64, "mem")


class TestModelDsp(unittest.TestCase):
    """The pure-DSP half of model_inference — the upstream-fidelity
    behaviors restored by the 2026-10-10 audit (delay compensation,
    conv_lookahead shift, DfNet apply order, atten_lim post-mix)."""

    @classmethod
    def setUpClass(cls):
        from states.run_speech_enhancement_model import model_inference as mi
        cls.mi = mi

    def test_norm_alpha_matches_upstream_rounding(self):
        # upstream get_norm_alpha rounds exp(-0.01)=0.99005 to 0.990
        self.assertEqual(self.mi._calc_norm_alpha(48000, 480, 1), 0.990)

    def test_stft_istft_roundtrip_is_delay_compensated_identity(self):
        mi = self.mi
        rng = np.random.default_rng(7)
        x = rng.standard_normal(48_000).astype(np.float32) * 0.3
        padded = np.pad(x, (0, mi._FFT_SIZE))
        y = mi._istft_full(mi._stft_full(padded), len(padded))
        d = mi._FFT_SIZE - mi._HOP_SIZE
        np.testing.assert_allclose(y[d: d + len(x)], x, atol=1e-4)

    def test_shift_features_drops_front_and_zero_pads_tail(self):
        mi = self.mi
        f = np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1)
        s = mi._shift_features(f.copy(), 2, time_axis=2)
        self.assertEqual(list(s[0, 0, :, 0]),
                         [2, 3, 4, 5, 6, 7, 8, 9, 0, 0])
        np.testing.assert_array_equal(
            mi._shift_features(f.copy(), 0, time_axis=2), f)

    def test_apply_df_delta_coef_is_identity_at_current_frame(self):
        # a 1.0 real tap at order index pad_before reproduces spec[t]
        mi = self.mi
        rng = np.random.default_rng(11)
        T = 6
        spec = (rng.standard_normal((T, mi._N_FREQ)) +
                1j * rng.standard_normal((T, mi._N_FREQ))
                ).astype(np.complex64)
        coefs = np.zeros((T, mi._NB_DF, mi._DF_ORDER, 2), dtype=np.float32)
        coefs[:, :, mi._DF_ORDER - mi._DF_LOOKAHEAD - 1, 0] = 1.0
        out = mi._apply_df(spec, coefs, None)
        np.testing.assert_allclose(out[:, :mi._NB_DF],
                                   spec[:, :mi._NB_DF], atol=1e-6)
        # bins >= nb_df pass through untouched
        np.testing.assert_array_equal(out[:, mi._NB_DF:],
                                      spec[:, mi._NB_DF:])

    def test_atten_lim_is_a_full_band_post_mix(self):
        mi = self.mi
        raw = np.full((3, 5), 1.0 + 0.0j, dtype=np.complex64)
        enh = np.zeros((3, 5), dtype=np.complex64)
        # 20 dB cap → lim = 0.1 → out = 0.1·raw
        out = mi._mix_atten_lim(enh, raw, 20.0)
        np.testing.assert_allclose(out, raw * 0.1, atol=1e-7)
        # extremes: >=100 unlimited, <0.01 bypass, None unlimited
        np.testing.assert_array_equal(mi._mix_atten_lim(enh, raw, 100.0), enh)
        np.testing.assert_array_equal(mi._mix_atten_lim(enh, raw, 0.0), raw)
        np.testing.assert_array_equal(mi._mix_atten_lim(enh, raw, None), enh)

    def test_erb_mask_has_no_atten_floor(self):
        # the mask applies verbatim — a zero band really is zero now
        mi = self.mi
        b2b = mi._build_bin_to_band(mi._ERB_WIDTHS)
        spec = np.ones((2, mi._N_FREQ), dtype=np.complex64)
        mask = np.zeros((2, mi._NB_ERB), dtype=np.float32)
        out = mi._apply_erb_mask(spec, mask, b2b)
        np.testing.assert_array_equal(out, np.zeros_like(spec))


class TestOutputContractPin(unittest.TestCase):
    """The contract file's labels are exactly the labels the gateway
    dispatcher emits — enhanced / metrics / usage."""

    def test_contract_labels(self):
        path = os.path.join(ROOT, "gateway_tool", "output_contract.json")
        with open(path) as f:
            contract = json.load(f)
        self.assertEqual(contract["target"], "speech-enhancement")
        self.assertEqual(set(contract["outputs"]),
                         {"enhanced", "metrics", "usage"})
        self.assertEqual(contract["outputs"]["enhanced"]["basename"],
                         "audio_enhanced.wav")
        self.assertEqual(contract["outputs"]["metrics"]["basename"],
                         "speech_enhancement.json")


if __name__ == "__main__":
    unittest.main()
