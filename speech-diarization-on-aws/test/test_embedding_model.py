#!/usr/bin/env python3
"""
Drop this in /var/task and run it to inspect the embedding ONNX directly.
python /tmp/probe_debug.py
"""
import os
import sys
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/tmp/models/pyannote/community-1"
N_MELS = 80

for fname in ["embedding.int8.onnx", "embedding.onnx"]:
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        print(f"SKIP {fname} — not found")
        continue

    print(f"\n{'='*60}")
    print(f"Model: {path}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    print("\n-- Inputs --")
    for i in sess.get_inputs():
        print(f"  name={i.name!r}  shape={i.shape}  type={i.type}")

    print("\n-- Outputs --")
    for o in sess.get_outputs():
        print(f"  name={o.name!r}  shape={o.shape}  type={o.type}")

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    print(f"\n-- Dummy inference (various frame counts) --")
    for n_frames in [100, 200, 298, 500, 1000]:
        dummy = np.zeros((1, n_frames, N_MELS), dtype=np.float32)
        try:
            outs = sess.run(None, {input_name: dummy})
            shapes = [f"{o.name!r}: {np.array(v).shape}"
                      for o, v in zip(sess.get_outputs(), outs)]
            print(f"  frames={n_frames:>4}  outputs: {shapes}")
        except Exception as e:
            print(f"  frames={n_frames:>4}  ERROR: {e}")

    print(f"\n-- Try 4-D input [1, 1, T, 80] --")
    for n_frames in [298, 500]:
        dummy = np.zeros((1, 1, n_frames, N_MELS), dtype=np.float32)
        try:
            outs = sess.run(None, {input_name: dummy})
            shapes = [f"{o.name!r}: {np.array(v).shape}"
                      for o, v in zip(sess.get_outputs(), outs)]
            print(f"  frames={n_frames:>4}  outputs: {shapes}")
        except Exception as e:
            print(f"  frames={n_frames:>4}  ERROR: {e}")
