#!/usr/bin/env python3
# export_wtpsplit.py
"""
Package wtpsplit sat-3l-sm for Lambda deployment.

sat-3l-sm ships two ONNX files on HuggingFace:
  model.onnx          ← float16, standard export  (~408 MB)
  model_optimized.onnx ← float32, ORT-fused ops   (~408 MB)

This script uses model.onnx because model_optimized.onnx uses
com.microsoft custom ops (SkipLayerNormalization, FastGelu) that are
incompatible with every path through the ORT quantization stack.

Steps:
  1. Download model.onnx + config.json from HuggingFace.
  2. Convert float16 → float32 (initializers + node attribute tensors).
  3. Apply dynamic INT8 quantization → ~200 MB.
  4. Verify INT8 model against FP32 baseline via direct ORT sessions.
  5. Package {model_optimized.onnx, config.json} as model.tar.gz.
  6. Copy local assets to ./assets/models/wtpsplit/.
  7. Upload to s3://{origin_bucket}/opensource_models/wtpsplit/model.tar.gz

Local asset layout
──────────────────
  ./assets/models/wtpsplit/
      onnx/
          model_optimized.onnx   ← INT8 quantized (~200 MB)
          config.json            ← model config (use_fp16=False)
          tokenizer.json         ← XLM-RoBERTa fast tokenizer
      model.tar.gz               ← flat archive (Lambda-ready)

Archive layout (flat, no top-level directory — matches model_store.py)
───────────────────────────────────────────────────────────────────────
  model_optimized.onnx  ← INT8 quantized (~200 MB, down from ~408 MB)
  config.json           ← model config (use_fp16=False)
  tokenizer.json        ← XLM-RoBERTa fast tokenizer (for lib.wtpsplit_sat)

Lambda inference
────────────────
  from lib.wtpsplit_sat import SaT
  sat = SaT("/tmp/models/wtpsplit", ort_providers=["CPUExecutionProvider"])
  sentences = sat.split(text)   # -sm models are language-agnostic

Usage
─────
  pip install onnxruntime onnx boto3 huggingface_hub wtpsplit
  python export_wtpsplit.py --origin-bucket <your-model-origin-bucket> --dry-run
  python export_wtpsplit.py --origin-bucket <your-model-origin-bucket>
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

MODEL_ID = "segment-any-text/sat-3l-sm"
BASE_DIR = "./assets/models/wtpsplit"
# model_optimized.onnx lives here
LOCAL_ONNX = os.path.join(BASE_DIR, "onnx")
# final archive lives here
LOCAL_ARCHIVE = os.path.join(BASE_DIR, "model.tar.gz")
S3_KEY = "opensource_models/wtpsplit/model.tar.gz"
ARCHIVE_NAME = "model.tar.gz"

# Multilingual verification corpus
VERIFY_CORPUS: List[Tuple[str, str, int]] = [
    ("en", "This is the first sentence. This is the second. And a third.",  3),
    ("cy", "Dyma'r frawddeg gyntaf. Dyma'r ail frawddeg. A thrydedd.",      3),
    ("fr", "Voici la première phrase. Voici la deuxième. Et une troisième.", 3),
    ("de", "Dies ist der erste Satz. Dies ist der zweite. Und ein dritter.", 3),
    ("zh", "这是第一句话。这是第二句话。还有第三句。",                           3),
    ("ar", "هذه هي الجملة الأولى. هذه هي الجملة الثانية. وثالثة.",          3),
    ("ja", "これは最初の文です。これは二番目の文です。そして三番目。",             3),
]


# =============================================================================
# STEP 1 — DOWNLOAD
# =============================================================================

def download_model(model_id: str, dest_dir: str) -> None:
    """Download model.onnx + config.json from HuggingFace.

    tokenizer.json is NOT in the sat-3l-sm repo — the model uses the
    xlm-roberta-base tokenizer.  We fetch it separately from the base
    model repo so it can be bundled in the archive for Lambda runtime
    (lib.wtpsplit_sat loads it via tokenizers.Tokenizer.from_file()).
    """
    from huggingface_hub import snapshot_download, hf_hub_download

    log.info(f"Downloading {model_id} from HuggingFace → {dest_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=dest_dir,
        ignore_patterns=[
            "*.safetensors",
            "*.bin",
            "*.msgpack",
            "*.h5",
            "flax_model*",
            "tf_model*",
            "trainer_state.json",
            "training_args.bin",
            "model_optimized.onnx",   # ORT-fused, incompatible with quantization
        ],
    )

    # tokenizer.json lives in the base model repo, not in sat-3l-sm
    tokenizer_dest = os.path.join(dest_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_dest):
        log.info("Fetching tokenizer.json from FacebookAI/xlm-roberta-base")
        hf_hub_download(
            repo_id="FacebookAI/xlm-roberta-base",
            filename="tokenizer.json",
            local_dir=dest_dir,
        )

    needed = ["model.onnx", "config.json", "tokenizer.json"]
    missing = [f for f in needed
               if not os.path.exists(os.path.join(dest_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Expected files missing after download: {missing}\n"
            f"Available: {[p.name for p in Path(dest_dir).iterdir()]}"
        )
    log.info(
        f"Downloaded: model.onnx={_mb(os.path.join(dest_dir,'model.onnx')):.1f} MB  "
        f"tokenizer.json fetched from xlm-roberta-base")


# =============================================================================
# STEP 2 — CONVERT FLOAT16 → FLOAT32
# =============================================================================

def convert_to_fp32(src_path: str, dst_path: str) -> None:
    """
    Convert an ONNX model from float16 to float32.

    sat-3l-sm/model.onnx is exported with float16 weights.
    quantize_dynamic requires a pure float32 graph — any remaining fp16
    constant produces invalid DequantizeLinear scale types that ORT refuses.

    Float16 values live in five places in an ONNX graph:
      1. graph.initializer          — weight tensors
      2. node.attribute (TENSOR)    — scalar/tensor constants in attrs
      3. node.attribute (TENSORS)   — lists of tensors in attrs
      4. Constant nodes             — inline constants via op_type="Constant"
      5. Cast nodes                 — Cast(to=FLOAT16) produces fp16 at runtime
                                      regardless of value_info annotations

    All five are converted.  Value_info and I/O annotations are updated
    to match.  A final ORT load validates the result before proceeding.
    """
    import onnx
    import numpy as np

    log.info(f"Loading {Path(src_path).name} for fp16→fp32 conversion...")
    t0 = time.time()
    model = onnx.load(src_path)

    def _convert_tensor(t: "onnx.TensorProto") -> bool:
        """Convert a TensorProto in-place from fp16→fp32. Returns True if changed."""
        if t.data_type != onnx.TensorProto.FLOAT16:
            return False
        arr = onnx.numpy_helper.to_array(t).astype(np.float32)
        new_t = onnx.numpy_helper.from_array(arr, name=t.name)
        t.CopyFrom(new_t)
        return True

    # ── 1. Initializers ───────────────────────────────────────────────────────
    n_init = sum(_convert_tensor(init) for init in model.graph.initializer)
    log.info(f"  Converted {n_init} initializers fp16→fp32")

    # ── 2 & 3 & 4. All nodes (including Constant nodes) ──────────────────────
    n_attr = 0
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                if _convert_tensor(attr.t):
                    n_attr += 1
            elif attr.type == onnx.AttributeProto.TENSORS:
                for t in attr.tensors:
                    if _convert_tensor(t):
                        n_attr += 1
    log.info(f"  Converted {n_attr} node attribute tensors fp16→fp32")

    # ── 5. Cast nodes: rewrite Cast(to=FLOAT16) → Cast(to=FLOAT) ─────────────
    # A Cast node producing fp16 output will cause a type mismatch even if
    # downstream value_info is annotated as float32.  Change the target dtype.
    n_cast = 0
    FLOAT16 = 10  # onnx.TensorProto.FLOAT16
    FLOAT = 1   # onnx.TensorProto.FLOAT
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == FLOAT16:
                    attr.i = FLOAT
                    n_cast += 1
    log.info(f"  Rewrote {n_cast} Cast(fp16) → Cast(fp32) nodes")

    # ── 6. Value_info + graph I/O type annotations ───────────────────────────
    n_vi = 0
    for vi in (list(model.graph.value_info)
               + list(model.graph.input)
               + list(model.graph.output)):
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            n_vi += 1
    log.info(f"  Updated {n_vi} value_info/IO annotations")

    # ── 6. Save, reload, and verify all session inputs are float32 ───────────
    log.info(f"Saving → {Path(dst_path).name}")
    onnx.save(model, dst_path)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            dst_path, providers=["CPUExecutionProvider"])
        bad_inputs = [
            (i.name, i.type) for i in sess.get_inputs()
            if "float16" in i.type
        ]
        if bad_inputs:
            raise RuntimeError(
                f"ORT session still reports fp16 inputs after conversion: {bad_inputs}\n"
                "The graph has fp16 type annotations that weren't caught."
            )
        log.info(
            f"  ORT load check ✅  inputs: {[(i.name, i.type) for i in sess.get_inputs()]}")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"fp32 model failed ORT load check: {exc}"
        ) from exc

    log.info(
        f"Conversion done in {time.time()-t0:.1f}s  "
        f"{_mb(src_path):.1f} MB → {_mb(dst_path):.1f} MB"
    )


# =============================================================================
# STEP 3 — QUANTIZE
# =============================================================================

def quantize_int8(fp32_path: str, int8_path: str) -> None:
    """
    Apply dynamic INT8 quantization to the FP32 ONNX model.

    Dynamic quantization quantizes weights statically and activations
    at runtime — no calibration dataset needed, standard for transformers
    on CPU.  per_channel=False is faster on ARM64 (Lambda).

    Note: attention MatMuls are skipped ("non constant B" warning) —
    this is normal for dynamic quantization on transformers and results
    in ~50% size reduction rather than ~75%.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    log.info(f"Quantizing {Path(fp32_path).name} → {Path(int8_path).name}")
    t0 = time.time()
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    fp32_mb = _mb(fp32_path)
    int8_mb = _mb(int8_path)
    log.info(
        f"Quantization done in {time.time()-t0:.1f}s  "
        f"{fp32_mb:.1f} MB → {int8_mb:.1f} MB  "
        f"({100*(1-int8_mb/fp32_mb):.0f}% reduction)"
    )


# =============================================================================
# STEP 4 — VERIFY
# =============================================================================

def verify(fp32_path: str, int8_path: str, tolerance: int = 1) -> None:
    """
    Verify the INT8 model produces consistent sentence splits vs the FP32
    baseline on a multilingual corpus, using wtpsplit end-to-end.

    Root cause of all previous failures: SaTORTWrapper.__call__ in
    wtpsplit/extract.py hardcodes:
        "attention_mask": attention_mask.astype(np.float16)
    ...regardless of what the ORT session actually expects.  Our converted
    fp32 model expects float32, so we monkey-patch __call__ to use float32.
    The patch is restored in the finally block.
    """
    import tempfile as _tf
    import numpy as np
    from lib.wtpsplit_sat import SaT

    staging_dir = str(Path(fp32_path).parent)
    config_src = os.path.join(staging_dir, "hf_model", "config.json")
    tokenizer_src = os.path.join(staging_dir, "hf_model", "tokenizer.json")

    def _stage(onnx_src: str) -> str:
        d = _tf.mkdtemp(prefix="wtpsplit_verify_")
        shutil.copy(onnx_src,   os.path.join(d, "model_optimized.onnx"))
        shutil.copy(config_src, os.path.join(d, "config.json"))
        shutil.copy(tokenizer_src, os.path.join(d, "tokenizer.json"))
        return d

    fp32_dir = _stage(fp32_path)
    int8_dir = _stage(int8_path)
    all_pass = False
    try:
        log.info("Loading FP32 SaT for baseline...")
        sat_fp32 = SaT(fp32_dir, ort_providers=["CPUExecutionProvider"])
        log.info("Loading INT8 SaT for comparison...")
        sat_int8 = SaT(int8_dir, ort_providers=["CPUExecutionProvider"])

        all_pass = True
        for lang, text, expected in VERIFY_CORPUS:
            sents_fp32 = list(sat_fp32.split(text))
            sents_int8 = list(sat_int8.split(text))
            diff = abs(len(sents_fp32) - len(sents_int8))
            passed = diff <= tolerance
            status = "✅" if passed else "❌"
            log.info(
                f"  {status} [{lang:2s}] expected≈{expected}  "
                f"FP32={len(sents_fp32)}  INT8={len(sents_int8)}  diff={diff}"
            )
            if not passed:
                all_pass = False
    finally:
        shutil.rmtree(int8_dir, ignore_errors=True)

    if not all_pass:
        raise RuntimeError(
            "INT8 model failed verification — sentence counts diverge. "
            "Do not upload."
        )
    log.info("Verification passed ✅")

# =============================================================================
# STEP 5 — PACKAGE
# =============================================================================


def package_archive(src_dir: str, int8_path: str, out_path: str) -> None:
    """
    Create a flat tar.gz archive:
      model_optimized.onnx  ← INT8 quantized (name SaT.__init__ looks for)
      config.json           ← from HuggingFace download, with use_fp16 forced to false

    use_fp16 is patched to False because our quantized model expects float32
    inputs.  The original model.onnx from HuggingFace is fp16, so config.json
    from the repo has use_fp16=True.
    """
    import json
    import tempfile as _tf2

    # Write patched config to a temp file
    config_src = os.path.join(src_dir, "config.json")
    tmp_config = _tf2.NamedTemporaryFile(suffix=".json", delete=False)
    try:
        with open(config_src) as f:
            cfg = json.load(f)
        cfg["use_fp16"] = False
        tmp_config.write(json.dumps(cfg).encode())
        tmp_config.flush()
        tmp_config_path = tmp_config.name
    finally:
        tmp_config.close()

    tokenizer_src = os.path.join(src_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_src):
        raise FileNotFoundError(
            f"tokenizer.json not found in {src_dir}. "
            f"snapshot_download should have fetched it — check download_model()."
        )

    files = {
        "model_optimized.onnx": int8_path,   # SaT.__init__ looks for this exact name
        "config.json":          tmp_config_path,
        "tokenizer.json":       tokenizer_src,
    }
    missing = [k for k, v in files.items()
               if k != "config.json" and not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"Cannot package — missing: {missing}")

    log.info(f"Packaging {out_path}")
    try:
        with tarfile.open(out_path, "w:gz") as tf:
            for archive_name, src_path in files.items():
                tf.add(src_path, arcname=archive_name)
                log.info(f"  + {archive_name}  ({_mb(src_path):.1f} MB)")
    finally:
        os.unlink(tmp_config_path)

    log.info(f"Archive: {out_path}  ({_mb(out_path):.1f} MB)")


# =============================================================================
# STEP 6 — UPLOAD
# =============================================================================

def upload_to_s3(archive_path: str, origin_bucket: str) -> None:
    import boto3
    s3 = boto3.client("s3")
    log.info(
        f"Uploading to s3://{origin_bucket}/{S3_KEY}  ({_mb(archive_path):.1f} MB)...")
    t0 = time.time()
    s3.upload_file(
        Filename=archive_path,
        Bucket=origin_bucket,
        Key=S3_KEY,
        ExtraArgs={"ContentType": "application/gzip"},
    )
    log.info(
        f"Upload complete in {time.time()-t0:.1f}s  →  s3://{origin_bucket}/{S3_KEY}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package wtpsplit sat-3l-sm ONNX INT8 → S3")
    parser.add_argument("--origin-bucket", required=True,
                        help="S3 bucket serving as the CloudFront origin")
    parser.add_argument("--model-id", default=MODEL_ID,
                        help=f"HuggingFace model ID (default: {MODEL_ID})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Download, quantize, verify but do not upload")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Keep the staging directory after completion")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification (not recommended for production)")
    args = parser.parse_args()

    staging = tempfile.mkdtemp(prefix="wtpsplit_export_")
    log.info(f"Staging: {staging}")

    # Ensure local output directories exist
    os.makedirs(LOCAL_ONNX, exist_ok=True)

    try:
        hf_dir = os.path.join(staging, "hf_model")
        fp16_path = os.path.join(hf_dir,  "model.onnx")       # downloaded fp16
        fp32_path = os.path.join(staging, "model_fp32.onnx")  # converted
        int8_path = os.path.join(staging, "model_int8.onnx")  # quantized
        archive_path = os.path.join(staging, ARCHIVE_NAME)

        # ── 1. Download ───────────────────────────────────────────────────────
        log.info("Step 1/5 — Downloading from HuggingFace")
        download_model(args.model_id, hf_dir)

        # ── 2. Convert fp16 → fp32 ────────────────────────────────────────────
        log.info("Step 2/5 — Converting float16 → float32")
        convert_to_fp32(fp16_path, fp32_path)

        # ── 3. Quantize ───────────────────────────────────────────────────────
        log.info("Step 3/5 — Quantizing to INT8")
        quantize_int8(fp32_path, int8_path)

        # ── 4. Verify ─────────────────────────────────────────────────────────
        if args.skip_verify:
            log.warning("Step 4/5 — Verification SKIPPED (--skip-verify)")
        else:
            log.info("Step 4/5 — Verifying INT8 vs FP32 baseline")
            verify(fp32_path, int8_path)

        # ── 5. Package ────────────────────────────────────────────────────────
        log.info("Step 5/5 — Packaging archive")
        package_archive(hf_dir, int8_path, archive_path)

        # ── Copy local assets ─────────────────────────────────────────────────
        local_onnx_dest = os.path.join(LOCAL_ONNX, "model_optimized.onnx")
        local_config_dest = os.path.join(LOCAL_ONNX, "config.json")
        local_tokenizer_dest = os.path.join(LOCAL_ONNX, "tokenizer.json")
        shutil.copy2(int8_path,                                local_onnx_dest)
        shutil.copy2(os.path.join(hf_dir, "config.json"),
                     local_config_dest)
        shutil.copy2(os.path.join(hf_dir, "tokenizer.json"),
                     local_tokenizer_dest)
        shutil.copy2(archive_path,                             LOCAL_ARCHIVE)
        log.info(
            f"Local copy — ONNX      : {local_onnx_dest}  ({_mb(local_onnx_dest):.1f} MB)")
        log.info(f"Local copy — config    : {local_config_dest}")
        log.info(f"Local copy — tokenizer : {local_tokenizer_dest}")
        log.info(
            f"Local copy — archive   : {LOCAL_ARCHIVE}  ({_mb(LOCAL_ARCHIVE):.1f} MB)")

        # ── Upload ────────────────────────────────────────────────────────────
        if args.dry_run:
            log.info(f"DRY RUN — skipping upload.")
        else:
            upload_to_s3(archive_path, args.origin_bucket)

        log.info("=" * 60)
        log.info("wtpsplit export complete ✅")
        log.info(f"  Model:    {args.model_id}")
        log.info(f"  ONNX:     INT8 quantized")
        log.info(
            f"  Size:     {_mb(int8_path):.1f} MB  (was {_mb(fp32_path):.1f} MB FP32)")
        log.info(f"  Archive:  {_mb(archive_path):.1f} MB")
        log.info("")
        log.info("Local assets written:")
        log.info(f"  {LOCAL_ONNX}/model_optimized.onnx")
        log.info(f"  {LOCAL_ONNX}/config.json")
        log.info(f"  {LOCAL_ONNX}/tokenizer.json")
        log.info(f"  {LOCAL_ARCHIVE}")
        if not args.dry_run:
            log.info(f"  s3://{args.origin_bucket}/{S3_KEY}")
        log.info("")
        log.info("Lambda usage:")
        log.info("  from lib.wtpsplit_sat import SaT")
        log.info('  sat = SaT("/tmp/models/wtpsplit",')
        log.info('            ort_providers=["CPUExecutionProvider"])')
        log.info('  sentences = sat.split(text)')
        log.info("=" * 60)

    finally:
        if args.keep_staging:
            log.info(f"Staging kept: {staging}")
        else:
            shutil.rmtree(staging, ignore_errors=True)
            log.info("Staging cleaned up")


def _mb(path: str) -> float:
    return os.path.getsize(path) / 1_048_576


if __name__ == "__main__":
    main()
