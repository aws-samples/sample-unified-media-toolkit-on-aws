#!/usr/bin/env python3
"""
extract_resume_splice_test.py — 9.13: a window processed across TWO
checkpoint/resume hops must produce bit-identical output to one processed
uninterrupted.

Run
---
  python -m unittest test.extract_resume_splice_test -v   # from repo root
  python test/extract_resume_splice_test.py

Requires a local `ffmpeg` (the test synthesises its own 24 s colour-block
video); skips cleanly when absent, so CI without ffmpeg loses this file's
coverage but not its green.

Why this exists
---------------
The checkpoint round trip is unit-tested in extract_checkpoint_test.py, but
the bugs in a resume live in the SPLICE: restored accumulators being
appended after instead of replaced, the seam-holdover frames losing their
files or their metadata across the hop (which silently changes boundary
detection at exactly one batch seam), batch_pos drifting so a batch is
embedded twice or skipped. This test runs the REAL _process_window (real
ffmpeg extraction, real batch loop, real seam holdover) with only two fakes,
both narrow: a deterministic stand-in for the DINOv2 session (per-image
colour means tiled to 384 dims — solid-colour shots embed identically,
colour cuts embed far apart), and module-level upload_file/download_file
redirected to a local directory standing in for S3.

The interrupted run is forced to checkpoint TWICE — the remaining-time stub
reports exhaustion on the 2nd and 4th gate reads (the gate reads the clock
exactly ONCE per pass, which this test would catch regressing: an extra read
per raise silently shifts the whole schedule). That yields: invocation 1
completes one batch then checkpoints at 8 s; invocation 2 resumes, completes
one more, checkpoints at 16 s; invocation 3 resumes and finishes. The
combined output is compared field-by-field — embeddings bit-exact — against
the uninterrupted run's.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.custom_exceptions import LambdaResourceTimeoutException  # noqa: E402
from states.run_dino_v2_model import lambda_function as mod       # noqa: E402

FFMPEG = shutil.which("ffmpeg")


class _FakeIO:
    """get_inputs/get_outputs/run — the whole surface _embed_frames uses."""
    name = "x"


class _FakeSession:
    def get_inputs(self):
        return [_FakeIO()]

    def get_outputs(self):
        return [_FakeIO()]

    def run(self, _outs, feed):
        batch = next(iter(feed.values()))          # [B, 3, 224, 224]
        means = batch.mean(axis=(2, 3))            # [B, 3]
        reps = int(np.ceil(384 / 3))
        return [np.tile(means, (1, reps))[:, :384].astype(np.float32)]


def _make_video(path, fps=24):
    """24 s of four solid 6 s colour blocks — three hard cuts."""
    colours = ("red", "green", "blue", "yellow")
    parts = []
    for i, c in enumerate(colours):
        p = f"{path}.part{i}.mp4"
        subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"color=c={c}:size=256x144:rate={fps}:duration=6",
             "-g", "1", p], check=True)
        parts.append(p)
    lst = f"{path}.txt"
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat",  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                    "-safe", "0", "-i", lst, "-c", "copy", path], check=True)


@unittest.skipUnless(FFMPEG, "ffmpeg not available")
class TestResumeSplice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="splice_")
        cls.video = os.path.join(cls.root, "proxy.mp4")
        _make_video(cls.video)
        cls.item = {
            "window_name": "seg0000_win0000",
            "window_start": 0.0, "window_end": 24.0,
            "frame_rate": "24/1", "frame_rate_fps": 24.0,
            "drop_frame": False, "timecode_first_frame": "00:00:00:00",
        }
        cls.cfg = {"mini_batch_secs": 8, "coarse_fps": 4}
        cls.pts = {"source_frame_numbers": list(range(24 * 24 + 8))}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _fake_s3(self, store_dir):
        os.makedirs(store_dir, exist_ok=True)

        def upload(bucket, key, local_path, extra=None):
            shutil.copyfile(local_path,
                            os.path.join(store_dir, key.replace("/", "_")))

        def download(bucket, key, local_path):
            src = os.path.join(store_dir, key.replace("/", "_"))
            if not os.path.exists(src):
                raise FileNotFoundError(key)
            shutil.copyfile(src, local_path)
        return upload, download

    def _run(self, tmp, time_left=None, bucket=None, key=None):
        return mod._process_window(
            _FakeSession(), self.video, dict(self.pts), self.item, self.cfg,
            tmp, time_left_ms=time_left, proxy_bucket=bucket, ckpt_key=key)

    def test_two_interruptions_reproduce_the_uninterrupted_run(self):
        # baseline: one uninterrupted pass
        base_tmp = os.path.join(self.root, "base")
        os.makedirs(base_tmp)
        (b_emb, b_fi, b_ft, b_bf, b_ic, b_ent, b_persist) = self._run(base_tmp)
        self.assertGreater(len(b_fi), 50)
        base_frames = sorted(os.listdir(b_persist))

        # interrupted: the time stub exhausts the budget on the 2nd and 4th
        # gate reads — invocation 1 checkpoints after one batch (8 s),
        # invocation 2 after one more (16 s), invocation 3 finishes
        store = os.path.join(self.root, "s3")
        up, down = self._fake_s3(store)
        orig_up, orig_down = mod.upload_file, mod.download_file
        mod.upload_file, mod.download_file = up, down
        try:
            calls = {"n": 0}

            def stingy():
                calls["n"] += 1
                return 1_000 if calls["n"] in (2, 4) else 900_000

            key = "embeddings/seg0000_win0000_checkpoint.tar"
            results = None
            raises = 0
            ckpt_positions = []
            for _ in range(5):
                tmp = os.path.join(self.root, f"inv{raises}")
                os.makedirs(tmp, exist_ok=True)
                try:
                    results = self._run(tmp, time_left=stingy,
                                        bucket="b", key=key)
                    break
                except LambdaResourceTimeoutException:
                    raises += 1
                    stored = os.path.join(store, key.replace("/", "_"))
                    st, _ = mod._unpack_checkpoint(
                        stored, os.path.join(tmp, "_p"),
                        os.path.join(tmp, "_h"))
                    ckpt_positions.append(st["batch_pos"])
            self.assertEqual(2, raises, "exactly two checkpoint hops")
            self.assertEqual([8.0, 16.0], ckpt_positions,
                             "each hop must advance by exactly one batch")
            self.assertIsNotNone(results)
            (emb, fi, ft, bf, ic, ent, persist) = results
        finally:
            mod.upload_file, mod.download_file = orig_up, orig_down

        self.assertEqual(b_fi, fi, "frame numbers must splice exactly")
        self.assertEqual(b_ft, ft)
        self.assertEqual(b_ic, ic)
        self.assertEqual(b_ent, ent)
        self.assertEqual(b_bf, bf)
        self.assertTrue(np.array_equal(b_emb, emb),
                        "embeddings must be bit-exact across the splice")
        self.assertEqual(base_frames, sorted(os.listdir(persist)),
                         "the frames archive input must be complete")

    def test_no_time_budget_means_no_checkpointing(self):
        # offline replays and unit paths pass no context — the loop must
        # never touch S3 or raise
        tmp = os.path.join(self.root, "plain")
        os.makedirs(tmp, exist_ok=True)
        out = self._run(tmp)
        self.assertEqual(7, len(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
