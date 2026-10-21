#!/usr/bin/env python3
"""
extract_checkpoint_test.py — 9.13: checkpoint/resume for Extract frame
features across Lambda's hard 900 s ceiling.

Run
---
  python -m unittest test.extract_checkpoint_test -v     # from repo root
  python test/extract_checkpoint_test.py

Why this exists
---------------
One window on very high-cut-rate content (film_a's action window)
deterministically needs more than 900 s of DINOv2 work, and 900 s is
Lambda's HARD ceiling — the timeout cannot be raised, so the only fix is to
make a timeout a normal path: checkpoint between mini-batches, raise
LambdaResourceTimeoutException before the hard kill, resume on the ASL
retry. Same pattern as face-recognition's checkpoint/resume states.

Pinned here are the pure parts (the S3 hop is the handler's, the loop splice
is proven live on film_a):

  1. the pack/unpack round trip is LOSSLESS — accumulator state, embeddings
     (bit-exact float32), the persisted model-input JPEGs the final archive
     needs, and the seam-holdover frames all survive;
  2. the checkpoint gate's arithmetic: a fresh Lambda always proceeds, the
     estimate of "one more batch" is sized by the SLOWEST batch of the
     invocation, and the upload buffer is always reserved;
  3. a checkpoint without state.json is a hard error, not a silent fresh
     start — resuming from a corrupt checkpoint by accident would silently
     re-embed and DUPLICATE every frame processed so far.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.input_schema import (                                 # noqa: E402
    EXTRACT_BATCH_TIME_FACTOR,
    EXTRACT_CHECKPOINT_BUFFER_MS,
)
from states.run_dino_v2_model.lambda_function import (         # noqa: E402
    _pack_checkpoint,
    _should_checkpoint,
    _unpack_checkpoint,
)


class TestPackUnpackRoundTrip(unittest.TestCase):

    def _dirs(self, tmp):
        p = os.path.join(tmp, "persist")
        h = os.path.join(tmp, "holdover")
        os.makedirs(p)
        os.makedirs(h)
        return p, h

    def test_round_trip_is_lossless(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, h = self._dirs(tmp)
            open(os.path.join(p, "frame_0000000042.jpg"), "wb").write(b"AA")
            open(os.path.join(p, "frame_0000000048.jpg"), "wb").write(b"BB")
            open(os.path.join(h, "hold_0000000050.jpg"), "wb").write(b"CC")
            emb = np.random.default_rng(1).normal(
                size=(7, 384)).astype(np.float32)
            state = {
                "batch_pos": 240.0, "batch_n": 2,
                "frame_index": [42, 44, 46, 48, 49, 50, 51],
                "frame_types": ["normal"] * 6 + ["black"],
                "is_coarse": [True, True, True, True, False, False, True],
                "entropy": [4.2, 4.1, 4.0, 3.9, 1.2, 0.1, 5.0],
                "black_frames": [{"frame": 51, "time": 2.125}],
                "prev_tail": {
                    "frames": [{"fnum": 50, "path": "/old/tmp/hold_0000000050.jpg",
                                "is_black": False, "entropy": 1.2}],
                    "last_coarse_emb": emb[3].tolist(),
                },
            }
            tar = os.path.join(tmp, "ck.tar")
            _pack_checkpoint(tar, state, emb, p, h)

            with tempfile.TemporaryDirectory() as tmp2:
                p2 = os.path.join(tmp2, "persist")
                h2 = os.path.join(tmp2, "holdover")
                got_state, got_emb = _unpack_checkpoint(tar, p2, h2)
                self.assertEqual(state, got_state)
                self.assertTrue(np.array_equal(emb, got_emb),
                                "embeddings must survive bit-exact")
                self.assertEqual(
                    {"frame_0000000042.jpg", "frame_0000000048.jpg"},
                    set(os.listdir(p2)))
                self.assertEqual(
                    b"CC",
                    open(os.path.join(h2, "hold_0000000050.jpg"),
                         "rb").read())

    def test_empty_window_state_round_trips(self):
        # a checkpoint written before any batch completed (pathological but
        # possible on an extremely short remaining budget)
        with tempfile.TemporaryDirectory() as tmp:
            p, h = self._dirs(tmp)
            state = {"batch_pos": 0.0, "batch_n": 0, "frame_index": [],
                     "frame_types": [], "is_coarse": [], "entropy": [],
                     "black_frames": [], "prev_tail": None}
            tar = os.path.join(tmp, "ck.tar")
            _pack_checkpoint(tar, state,
                             np.zeros((0, 384), dtype=np.float32), p, h)
            got_state, got_emb = _unpack_checkpoint(
                tar, os.path.join(tmp, "p2"), os.path.join(tmp, "h2"))
            self.assertEqual(state, got_state)
            self.assertEqual((0, 384), got_emb.shape)

    def test_checkpoint_without_state_is_a_hard_error(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            tar = os.path.join(tmp, "bad.tar")
            junk = os.path.join(tmp, "junk.bin")
            open(junk, "wb").write(b"x")
            with tarfile.open(tar, "w") as t:
                t.add(junk, arcname="persist/junk.bin")
            with self.assertRaises(ValueError):
                _unpack_checkpoint(tar, os.path.join(tmp, "p"),
                                   os.path.join(tmp, "h"))


class TestCheckpointGate(unittest.TestCase):

    def test_fresh_lambda_always_proceeds(self):
        # first batch of an invocation: no batch estimate yet, only the
        # upload buffer is reserved — a fresh 900 s Lambda sails through
        self.assertFalse(_should_checkpoint(880_000, 0.0))

    def test_only_the_buffer_is_reserved_with_no_estimate(self):
        self.assertTrue(
            _should_checkpoint(EXTRACT_CHECKPOINT_BUFFER_MS - 1, 0.0))
        self.assertFalse(
            _should_checkpoint(EXTRACT_CHECKPOINT_BUFFER_MS + 1, 0.0))

    def test_slowest_batch_sizes_the_estimate(self):
        need = (EXTRACT_BATCH_TIME_FACTOR * 100_000
                + EXTRACT_CHECKPOINT_BUFFER_MS)
        self.assertTrue(_should_checkpoint(need - 1, 100_000))
        self.assertFalse(_should_checkpoint(need + 1, 100_000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
