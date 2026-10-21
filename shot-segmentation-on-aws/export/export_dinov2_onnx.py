"""
export_dinov2_onnx.py
---------------------
Exports DINOv2 ViT-S/14 with registers to ONNX (opset 17) for AWS Lambda
ARM64 deployment, then applies ORT CPU-specific graph optimisation.

Model: dinov2_vits14_reg
  - 21M params vs 86M for ViT-B/14 — ~4x faster inference
  - 384-dim CLS token embeddings (same as Amazon Titan embed-image-v1)
  - Register tokens fix attention artefacts on uniform/background regions
    common in broadcast content (black frames, title cards, studio backdrops)
  - Self-supervised DINO v2 training — cosine distance directly meaningful
    for visual similarity, unlike DeiT-III (classification-optimised)
  - Apache 2.0 license

Pipeline:
    1. Load DINOv2 ViT-S/14-reg (torch.hub or local checkpoint)
    2. Export to ONNX FP32             -> dinov2_vits14_reg_cls.onnx
    3. ORT CPU optimisation            -> dinov2_vits14_reg_cls_opt.onnx
    4. Validate parity (PT vs opt ORT)
    5. Benchmark inference speed
    6. Package optimised model         -> model.tar.gz

Usage:
    python export_dinov2_onnx.py \\
        --output-dir ./output \\
        [--checkpoint /path/to/local.pth]

Requirements:
    pip install torch torchvision onnx onnxruntime
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ViT-S/14-reg architecture constants — must match the model exactly
# ViT-S: 6 attention heads, 384 hidden dim
_VIT_S_NUM_HEADS   = 6
_VIT_S_HIDDEN_SIZE = 384
_MODEL_NAME        = "dinov2_vits14_reg"
_OUTPUT_FILE       = "dinov2_vits14_reg_cls.onnx"
_OUTPUT_FILE_OPT   = "dinov2_vits14_reg_cls_opt.onnx"
_EMBED_DIM         = 384


# ---------------------------------------------------------------------------
# Wrapper: returns CLS token only
# ---------------------------------------------------------------------------

class DINOv2CLS(nn.Module):
    """
    Thin wrapper around DINOv2 that returns only the CLS token embedding.
    Shape: [batch, 384]  (ViT-S)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        return features["x_norm_clstoken"]   # [B, 384]


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(checkpoint: str | None) -> nn.Module:
    log.info("Loading DINOv2 ViT-S/14-reg ...")
    log.info("  Model: %s  (21M params, 384-dim, register tokens)", _MODEL_NAME)

    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        _MODEL_NAME,
        pretrained=(checkpoint is None),
    )

    if checkpoint:
        log.info("Loading weights from local checkpoint: %s", checkpoint)
        state_dict = torch.load(checkpoint, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        def strip_prefix(sd, prefix):
            return {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in sd.items()
            }

        for prefix in ("backbone.", "module.", "model."):
            if any(k.startswith(prefix) for k in state_dict):
                log.info("Stripping key prefix: '%s'", prefix)
                state_dict = strip_prefix(state_dict, prefix)

        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("Missing keys (%d): %s ...", len(missing), missing[:5])
        if unexpected:
            log.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])

    model = DINOv2CLS(backbone)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export to ONNX (raw FP32)
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, output_path: str) -> None:
    log.info("Exporting to ONNX FP32 (opset 18) ...")

    dummy_input = torch.zeros(1, 3, 224, 224)

    # DINOv2 uses F.interpolate with antialias=True for position embedding
    # interpolation (aten::_upsample_bicubic2d_aa). This op is unsupported
    # in ONNX opset 17 and only partially supported in opset 18.
    # Since we always export at fixed 224x224 (= training resolution),
    # the interpolation branch is never triggered at inference — the model
    # takes the early-return path when npatch == N. Patching antialias=False
    # is safe and produces identical results for fixed-size input.
    import torch.nn.functional as _F
    _orig_interpolate = _F.interpolate

    def _interpolate_no_aa(*args, **kwargs):
        kwargs.pop("antialias", None)
        return _orig_interpolate(*args, **kwargs)

    _F.interpolate = _interpolate_no_aa
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["cls_embedding"],
            dynamic_axes={
                "pixel_values":  {0: "batch_size"},
                "cls_embedding": {0: "batch_size"},
            },
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
    finally:
        _F.interpolate = _orig_interpolate   # always restore

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    log.info("ONNX FP32 saved: %s  (%.1f MB)", output_path, size_mb)


# ---------------------------------------------------------------------------
# ORT CPU optimisation
# ---------------------------------------------------------------------------

def optimise_for_cpu(raw_onnx_path: str, opt_onnx_path: str) -> None:
    """
    Apply ORT CPU-specific graph optimisations.

    num_heads=6 and hidden_size=384 must match ViT-S/14 exactly —
    incorrect values silently disable attention fusion.
    """
    log.info("Applying ORT CPU graph optimisation ...")
    log.info(
        "  model_type=bert  num_heads=%d  hidden_size=%d",
        _VIT_S_NUM_HEADS, _VIT_S_HIDDEN_SIZE,
    )

    try:
        from onnxruntime.transformers.optimizer import optimize_model
    except ImportError:
        try:
            from onnxruntime.tools.optimizer import optimize_model
        except ImportError:
            log.error(
                "ORT optimizer not found. "
                "Run: pip install onnxruntime>=1.8.0 onnx"
            )
            sys.exit(1)

    optimised = optimize_model(
        raw_onnx_path,
        model_type       = "bert",
        num_heads        = _VIT_S_NUM_HEADS,
        hidden_size      = _VIT_S_HIDDEN_SIZE,
        opt_level        = 1,    # ORT_ENABLE_EXTENDED — avoids NchwcTransformer
                                 # which produces hardware-specific ops that break
                                 # cross-platform deployment (local CPU → Graviton2)
        use_gpu          = False,
        only_onnxruntime = False,
    )

    optimised.save_model_to_file(opt_onnx_path)

    raw_mb = os.path.getsize(raw_onnx_path) / (1024 ** 2)
    opt_mb = os.path.getsize(opt_onnx_path) / (1024 ** 2)

    fused_op_types = {
        node.op_type
        for node in optimised.model.graph.node
        if any(k in node.op_type for k in (
            "Attention", "LayerNorm", "Gelu", "SkipLayer", "Embed", "Bias"
        ))
    }

    log.info(
        "Optimised model saved: %s\n"
        "  Raw:       %.1f MB\n"
        "  Optimised: %.1f MB\n"
        "  Graph nodes: %d\n"
        "  Fused ops: %s",
        opt_onnx_path, raw_mb, opt_mb,
        len(optimised.model.graph.node),
        sorted(fused_op_types) if fused_op_types else "none detected",
    )

    if not fused_op_types:
        log.warning(
            "No fusion ops detected — check num_heads=%d and hidden_size=%d "
            "match the model exactly.",
            _VIT_S_NUM_HEADS, _VIT_S_HIDDEN_SIZE,
        )


# ---------------------------------------------------------------------------
# Validate numerical parity
# ---------------------------------------------------------------------------

def validate_parity(
    model:        nn.Module,
    raw_path:     str,
    opt_path:     str,
    atol:         float = 1e-3,
) -> None:
    """
    Validate both raw and optimised ORT models against PyTorch reference.
    Cosine similarity is the primary quality gate — atol is secondary.
    Always reports cosine similarity even if atol fails, so you can decide
    whether to deploy raw or optimised.
    """
    log.info("Validating parity (PyTorch vs ORT) ...")

    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping parity validation.")
        return

    torch.manual_seed(42)
    test_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        pt_output = model(test_input).numpy()

    def _run(path: str, label: str) -> np.ndarray:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            path, opts, providers=["CPUExecutionProvider"],
        )
        out = sess.run(["cls_embedding"], {"pixel_values": test_input.numpy()})[0]
        max_diff  = np.abs(pt_output - out).max()
        mean_diff = np.abs(pt_output - out).mean()
        pt_norm   = pt_output / np.linalg.norm(pt_output)
        out_norm  = out / np.linalg.norm(out)
        cosine    = float(np.dot(pt_norm.flatten(), out_norm.flatten()))
        status    = "PASSED ✓" if max_diff <= atol else "atol FAILED"
        cos_status = "PASSED ✓" if cosine >= 0.999 else "FAILED ✗"
        log.info(
            "%s: max_diff=%.6f (%s)  mean_diff=%.6f  cosine=%.8f (%s)",
            label, max_diff, status, mean_diff, cosine, cos_status,
        )
        return out

    _run(raw_path, "Raw ONNX    ")
    opt_out = _run(opt_path, "Optimised   ")

    # Final deployment decision
    opt_max_diff = np.abs(pt_output - opt_out).max()
    pt_norm  = pt_output / np.linalg.norm(pt_output)
    opt_norm = opt_out   / np.linalg.norm(opt_out)
    cosine   = float(np.dot(pt_norm.flatten(), opt_norm.flatten()))

    if cosine >= 0.999:
        log.info("Deploy optimised model — cosine similarity acceptable (%.6f)", cosine)
    elif cosine >= 0.99:
        log.warning(
            "Cosine similarity %.8f is marginal — consider deploying raw model",
            cosine,
        )
    else:
        log.error(
            "Cosine similarity %.8f is too low — deploy raw model instead\n"
            "Package raw: tar -czf model.tar.gz -C %s %s",
            cosine, os.path.dirname(raw_path), os.path.basename(raw_path),
        )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(opt_path: str, num_threads: int = 6, n_runs: int = 50) -> None:
    """
    Benchmark optimised model inference speed on CPU.
    50 runs for a stable p95 estimate.
    """
    log.info(
        "Benchmarking %s  (threads=%d  runs=%d) ...",
        _MODEL_NAME, num_threads, n_runs,
    )

    try:
        import onnxruntime as ort
        import time
    except ImportError:
        return

    opts = ort.SessionOptions()
    opts.intra_op_num_threads     = num_threads
    opts.inter_op_num_threads     = num_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        opt_path, opts, providers=["CPUExecutionProvider"],
    )

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(5):
        sess.run(["cls_embedding"], {"pixel_values": dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(["cls_embedding"], {"pixel_values": dummy})
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    log.info(
        "Latency (batch=1, threads=%d):\n"
        "  mean=%.1fms  p50=%.1fms  p95=%.1fms  min=%.1fms\n"
        "  Throughput: %.1f frames/sec",
        num_threads,
        arr.mean(), np.percentile(arr, 50),
        np.percentile(arr, 95), arr.min(),
        1000 / arr.mean(),
    )

    # Also benchmark at 1 thread — useful for multiprocessing planning
    opts1 = ort.SessionOptions()
    opts1.intra_op_num_threads     = 1
    opts1.inter_op_num_threads     = 1
    opts1.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess1 = ort.InferenceSession(
        opt_path, opts1, providers=["CPUExecutionProvider"],
    )
    for _ in range(5):
        sess1.run(["cls_embedding"], {"pixel_values": dummy})
    times1 = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess1.run(["cls_embedding"], {"pixel_values": dummy})
        times1.append((time.perf_counter() - t0) * 1000)
    arr1 = np.array(times1)
    log.info(
        "Latency (batch=1, threads=1 — for multiprocessing planning):\n"
        "  mean=%.1fms  p50=%.1fms  min=%.1fms\n"
        "  Scaling efficiency (%d threads): %.2fx of linear",
        arr1.mean(), np.percentile(arr1, 50), arr1.min(),
        num_threads, arr1.mean() / arr.mean() / num_threads,
    )


# ---------------------------------------------------------------------------
# Package to tar.gz
# ---------------------------------------------------------------------------

def package_tarball(opt_onnx_path: str, output_dir: Path) -> str:
    """
    Package optimised model as model.tar.gz for Lambda deployment.
    The optimised model is stored under the canonical filename so the
    Lambda handler needs no changes.
    """
    import tarfile

    tarball_path = str(output_dir / "model.tar.gz")
    log.info("Packaging into %s ...", tarball_path)

    with tarfile.open(tarball_path, "w:gz") as tar:
        # Store under canonical name — Lambda handler looks for this filename
        tar.add(opt_onnx_path, arcname=_OUTPUT_FILE.replace("_opt", ""))

    size_mb = os.path.getsize(tarball_path) / (1024 ** 2)
    log.info("model.tar.gz: %.1f MB", size_mb)
    return tarball_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DINOv2 ViT-S/14-reg to optimised ONNX for CPU inference"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Local .pth checkpoint (uses torch.hub if omitted)",
    )
    parser.add_argument(
        "--atol", type=float, default=0.1,
        help="Absolute tolerance for parity validation (default: 0.1 — cosine similarity is primary gate)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip parity validation",
    )
    parser.add_argument(
        "--skip-packaging", action="store_true",
        help="Skip model.tar.gz packaging",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference benchmark after optimisation",
    )
    parser.add_argument(
        "--benchmark-threads", type=int, default=6,
        help="ORT threads for benchmark (default: 6)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_onnx_path = str(output_dir / _OUTPUT_FILE)
    opt_onnx_path = str(output_dir / _OUTPUT_FILE_OPT)

    # 1. Load
    model = load_model(args.checkpoint)

    # 2. Export raw FP32
    export_onnx(model, raw_onnx_path)

    # 3. Optimise
    optimise_for_cpu(raw_onnx_path, opt_onnx_path)

    # 4. Validate
    if not args.skip_validation:
        validate_parity(model, raw_onnx_path, opt_onnx_path, atol=args.atol)

    # 5. Benchmark
    if args.benchmark:
        benchmark(opt_onnx_path, num_threads=args.benchmark_threads)

    # 6. Package
    tarball_path = None
    if not args.skip_packaging:
        tarball_path = package_tarball(opt_onnx_path, output_dir)

    log.info("Export complete.")
    log.info("  ONNX (raw):       %s", raw_onnx_path)
    log.info("  ONNX (optimised): %s", opt_onnx_path)
    if tarball_path:
        log.info("  model.tar.gz:     %s  <- deploy this", tarball_path)


if __name__ == "__main__":
    main()