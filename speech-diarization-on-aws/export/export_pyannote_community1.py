#!/usr/bin/env python3
# export_pyannote_community1.py
# ---------------------------------------------------------------------------
# torchcodec stub — MUST be the very first code executed, before any import
# that transitively touches pyannote.audio.
#
# torchcodec has no aarch64 Linux wheels (upstream issue #569).  pyannote.audio
# 4.x imports torchcodec at module-load time, but never calls it during ONNX
# export (no audio files are decoded — only model weights are traced).
#
# Python's import system requires __spec__ to be set on any module that acts
# as a package (i.e. has sub-modules).  A bare ModuleType() leaves __spec__
# as None which causes "torchcodec.__spec__ is None" when Python tries to
# resolve "torchcodec.decoders" as a child import.  We fix this by giving
# every stub module a proper ModuleSpec with is_package=True.
# ---------------------------------------------------------------------------

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import sys as _sys
import types as _types
import importlib.machinery as _imm
import importlib.util as _imu

def _stub_pkg(name: str) -> _types.ModuleType:
    """Create and register a stub module that satisfies is_package checks."""
    if name in _sys.modules:
        return _sys.modules[name]
    m = _types.ModuleType(name)
    # __spec__ with is_package=True prevents ".__spec__ is None" errors
    spec          = _imm.ModuleSpec(name, loader=None, is_package=True)
    m.__spec__    = spec
    m.__loader__  = None
    m.__path__    = []           # marks it as a package
    m.__package__ = name
    _sys.modules[name] = m
    return m

try:
    import torchcodec as _tc_real   # noqa: F401  — real wheel if available
except (ImportError, RuntimeError, OSError):
    _tc  = _stub_pkg("torchcodec")
    _tcd = _stub_pkg("torchcodec.decoders")
    _tca = _stub_pkg("torchcodec.audio")

    # Wire sub-modules as attributes on the parent (mirrors real package layout)
    _tc.decoders = _tcd
    _tc.audio    = _tca

    class _StubDecoder:
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "torchcodec is not available on aarch64 Linux. "
                "This stub exists only to satisfy pyannote.audio imports during "
                "ONNX export. Do not use it to decode audio files."
            )
    _tcd.AudioDecoder = _StubDecoder
    _tca.AudioDecoder = _StubDecoder

    del _tc, _tcd, _tca, _StubDecoder

del _sys, _types, _imm, _imu, _stub_pkg
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# torchaudio compatibility shim
# torchaudio >= 2.9 removed several APIs that pyannote.audio still references
# at import time.  Patch them back onto the existing torchaudio module so the
# rest of the pipeline loads cleanly.  None of these are called during export.
# ---------------------------------------------------------------------------
import torchaudio as _ta
from dataclasses import dataclass as _dc

if not hasattr(_ta, "list_audio_backends"):
    _ta.list_audio_backends = lambda: ["sox_io"]

if not hasattr(_ta, "get_audio_backend"):
    _ta.get_audio_backend = lambda: "sox_io"

if not hasattr(_ta, "set_audio_backend"):
    _ta.set_audio_backend = lambda backend: None

if not hasattr(_ta, "AudioMetaData"):
    @_dc
    class _AudioMetaData:
        sample_rate:      int = 0
        num_frames:       int = 0
        num_channels:     int = 0
        bits_per_sample:  int = 0
        encoding:         str = ""
    _ta.AudioMetaData = _AudioMetaData
    del _AudioMetaData

del _ta, _dc
# ---------------------------------------------------------------------------

"""
export_community1_onnx.py
=========================
Exports the pyannote/speaker-diarization-community-1 segmentation and
embedding models to ONNX format, optionally quantizes to INT8, packages
them into a flat model.tar.gz, and uploads to S3.

Requirements (export machine only — NOT needed at inference time):
    pip install pyannote.audio torch onnx onnxruntime onnxsim boto3 psutil

Usage:
    # Export + quantize + package only
    python export_community1_onnx.py --token hf_YOUR_TOKEN

    # Export + upload to S3
    python export_community1_onnx.py --token hf_YOUR_TOKEN --s3_bucket my-bucket

    # Export FP32 only, no quantization, skip upload
    python export_community1_onnx.py --token hf_YOUR_TOKEN --no_quantize

Output layout:
    ./assets/models/pyannote/
        onnx/
            segmentation.onnx          (FP32)
            segmentation.int8.onnx     (INT8)
            embedding.onnx             (FP32)
            embedding.int8.onnx        (INT8)
            pipeline_config.json
        model.tar.gz                   (flat archive — all files from onnx/)

S3 destination (if --s3_bucket provided):
    s3://BUCKET/opensource_models/pyannote/community-1/model.tar.gz

Notes:
    • The archive extracts flat (model.bin-style): no subfolder prefix.
      Lambda / EFS just extracts it and all files land directly in the target dir.
    • Only the INT8 models are included in the archive by default.
      Pass --package_fp32 to include FP32 models as well (roughly 2x larger).
    • The embedding model (WeSpeaker ResNet34-LM) expects 80-dim log-Mel
      filterbank features as input, NOT raw waveforms.
    • The segmentation model expects raw waveforms: [batch, 1, num_samples].
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# CONSTANTS
# =============================================================================

BASE_DIR        = "./assets/models/pyannote"
S3_KEY_PREFIX   = "opensource_models/pyannote/community-1"
MODEL_ID        = "pyannote/speaker-diarization-community-1"

# Files included in the archive (in addition to ONNX models)
ARCHIVE_EXTRAS  = ["pipeline_config.json", "ATTRIBUTION.md"]

# CC-BY-4.0 attribution — a LICENSE CONDITION of redistributing the
# converted weights (the seeded tarball is this project's one hosted
# redistribution channel; license audit 2026-10-06). Written into the
# model dir before packaging so every rebuild carries it.
ATTRIBUTION_MD = """# Attribution

This archive contains the pyannote **speaker-diarization-community-1**
model weights, converted to ONNX (and INT8-quantized) for CPU
inference. The conversion is the only modification; the weights are
otherwise the original authors' work.

- Original model: https://huggingface.co/pyannote/speaker-diarization-community-1
- Creator: the pyannote project (Herve Bredin and contributors),
  https://github.com/pyannote/pyannote-audio
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- Modifications: exported to ONNX; INT8 quantization of the exported
  graphs. No changes to the model architecture or weights beyond
  format conversion and quantization.

The weights are provided "as is", without warranties, per the license.
"""


# Files that go into the archive when --package_fp32 is set
FP32_MODELS     = ["segmentation.onnx", "embedding.onnx"]

# Files always included in the archive (INT8 by default)
INT8_MODELS     = ["segmentation.int8.onnx", "embedding.int8.onnx"]


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class ExportResult:
    success:         bool  = False
    skipped:         bool  = False
    error:           Optional[str] = None
    duration_sec:    float = 0.0
    onnx_size_mb:    float = 0.0       # total size of all .onnx files
    package_size_mb: float = 0.0
    uploaded:        bool  = False
    s3_uri:          Optional[str] = None


# =============================================================================
# LOGGING / SIZE HELPERS  (mirroring Whisper script conventions)
# =============================================================================

def log(msg: str) -> None:
    print(f"[pyannote] {msg}", flush=True)


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


def dir_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    ) / (1024 * 1024)


def _make_progress_callback(total_bytes: int):
    """Log S3 upload progress at 0 / 25 / 50 / 75 / 100 %."""
    uploaded  = [0]
    last_mile = [-1]

    def callback(chunk: int) -> None:
        uploaded[0] += chunk
        pct       = int(100 * uploaded[0] / total_bytes)
        milestone = (pct // 25) * 25
        if milestone > last_mile[0]:
            last_mile[0] = milestone
            mb_done  = uploaded[0] / (1024 * 1024)
            mb_total = total_bytes / (1024 * 1024)
            log(f"   ↑ {mb_done:.1f}/{mb_total:.1f} MB  ({pct}%)")

    return callback


# =============================================================================
# STATE CHECKS  (idempotency — skip stages already completed)
# =============================================================================

def is_exported(onnx_dir: str) -> bool:
    """True if both FP32 ONNX models are present."""
    return all(
        os.path.exists(os.path.join(onnx_dir, f))
        for f in FP32_MODELS
    )


def is_quantized(onnx_dir: str) -> bool:
    """True if both INT8 ONNX models are present."""
    return all(
        os.path.exists(os.path.join(onnx_dir, f))
        for f in INT8_MODELS
    )


def is_packaged(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "model.tar.gz"))


def is_uploaded_s3(bucket: str) -> bool:
    try:
        import boto3
        boto3.client("s3").head_object(
            Bucket=bucket,
            Key=f"{S3_KEY_PREFIX}/model.tar.gz",
        )
        return True
    except Exception:
        return False


# =============================================================================
# IMPORT CHECK
# =============================================================================

def _check_imports():
    missing = []
    for pkg in ["pyannote.audio", "onnx", "onnxruntime", "onnxsim"]:
        try:
            __import__(pkg.split(".")[0])
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages: {missing}. "
            "Install with: pip install pyannote.audio torch onnx onnxruntime onnxsim"
        )


# =============================================================================
# STAGE 1 — EXPORT SEGMENTATION
# =============================================================================

def stage_export_segmentation(pipeline, onnx_dir: str, opset: int) -> str:
    """Export SincNet + LSTM powerset segmentation model to ONNX."""
    log("⚙️  Exporting segmentation model ...")

    seg_model: nn.Module = pipeline._segmentation.model
    seg_model.eval()

    chunk_samples = 10 * 16000   # 10 s at 16 kHz = 160 000 samples
    dummy         = torch.zeros(2, 1, chunk_samples)
    out_path      = os.path.join(onnx_dir, "segmentation.onnx")

    # dynamo=False forces the legacy TorchScript-based exporter.
    # torch.onnx.export in PyTorch >= 2.9 defaults to dynamo=True which uses
    # torch.export internally — that path cannot handle dynamic LSTM hidden
    # state shapes (the "data-dependent Eq(589*u0, 0)" guard error).
    # The TorchScript exporter traces through LSTM correctly.
    torch.onnx.export(
        seg_model,
        dummy,
        out_path,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={
            "waveform": {0: "batch_size", 2: "num_samples"},
            "logits":   {0: "batch_size", 1: "num_frames"},
        },
        dynamo=False,
    )
    log(f"   Saved  segmentation.onnx  ({file_size_mb(out_path):.1f} MB)")

    _simplify(out_path)
    _verify_segmentation(seg_model, out_path, chunk_samples)

    return out_path


# =============================================================================
# STAGE 2 — EXPORT EMBEDDING
# =============================================================================

class WeSpeakerFbankWrapper(nn.Module):
    """
    Wraps the WeSpeaker ResNet34-LM backbone to accept pre-computed
    log-Mel filterbank features [B, T, 80] rather than raw waveforms.
    Feature extraction is done in pure numpy at inference time.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        # ResNet34 expects [B, 1, T, 80]
        x   = fbank.unsqueeze(1)
        emb = self.backbone(x)
        return torch.nn.functional.normalize(emb, p=2, dim=1)


def _extract_wespeaker_backbone(pipeline) -> nn.Module:
    """
    Navigate pipeline._embedding to find the innermost nn.Module backbone.

    The wrapper class name changed across pyannote.audio versions:
      3.x : WeSpeakerPretrainedSpeakerEmbedding   (is an nn.Module itself)
      4.0 : PyannoteAudioPretrainedSpeakerEmbedding (NOT an nn.Module;
            holds .model -> WeSpeakerModel -> .model -> ResNet34)

    We walk the attribute chain and return the deepest nn.Module we find
    that looks like a real backbone (has > 3 child modules).
    """
    import torch.nn as _nn

    def _is_backbone(obj) -> bool:
        """True if obj is an nn.Module with enough children to be a real network."""
        return isinstance(obj, _nn.Module) and len(list(obj.children())) > 3

    # Walk a fixed list of candidate attribute chains, deepest first
    emb = pipeline._embedding

    # Log what we actually got so future errors are easier to diagnose
    log(f"   _embedding type: {type(emb).__name__}")

    # Deep-introspect every attribute so we can always diagnose failures
    log("   _embedding attributes:")
    for attr in [a for a in dir(emb) if not a.startswith("__")]:
        try:
            val = getattr(emb, attr)
            if callable(val) and not isinstance(val, _nn.Module):
                continue
            vtype = type(val).__name__
            extra = ""
            if isinstance(val, _nn.Module):
                nparams = sum(p.numel() for p in val.parameters())
                children = [n for n, _ in val.named_children()][:5]
                extra = f"  [{nparams:,} params, children: {children}]"
            log(f"     .{attr}: {vtype}{extra}")
        except Exception as e:
            log(f"     .{attr}: <error: {e}>")

    candidates = []

    # Collect every nn.Module reachable via .model chains (up to depth 4)
    def _collect(obj, depth=0):
        if depth > 6:
            return
        if isinstance(obj, _nn.Module):
            candidates.append(obj)
        # model_ (trailing underscore) is used by PyannoteAudioPretrainedSpeakerEmbedding
        for attr in ("model", "model_", "backbone", "encoder", "embedding_model",
                     "speaker_model", "wespeaker_model", "embedding"):
            if hasattr(obj, attr):
                try:
                    child = getattr(obj, attr)
                except Exception:
                    continue
                if child is not obj and child is not None:
                    _collect(child, depth + 1)
        # Also walk all nn.Module children directly
        if isinstance(obj, _nn.Module):
            for _, child in obj.named_children():
                if child not in candidates:
                    _collect(child, depth + 1)

    _collect(emb)

    # Prefer the deepest / most-parametric backbone
    backbones = [c for c in candidates if _is_backbone(c)]
    if backbones:
        # Pick the one with the most parameters (innermost ResNet wins)
        best = max(backbones, key=lambda m: sum(p.numel() for p in m.parameters()))
        log(f"   backbone type : {type(best).__name__}  "
            f"({sum(p.numel() for p in best.parameters()):,} params)")
        return best

    # Fallback: return the first nn.Module we found at all
    nn_mods = [c for c in candidates if isinstance(c, _nn.Module)]
    if nn_mods:
        log(f"   backbone (fallback): {type(nn_mods[0]).__name__}")
        return nn_mods[0]

    raise RuntimeError(
        f"Could not find an nn.Module backbone inside "
        f"pipeline._embedding ({type(emb).__name__}). "
        f"Attributes found: {[a for a in dir(emb) if not a.startswith('_')]}"
    )


def stage_export_embedding(pipeline, onnx_dir: str, opset: int) -> str:
    """Export WeSpeaker ResNet34-LM embedding model to ONNX.

    We export the backbone directly with a 4D input [B, 1, T, 80] rather than
    wrapping it in WeSpeakerFbankWrapper.  The wrapper's unsqueeze(1) causes
    a TorchScript "Unsupported value kind: Tensor" error because the legacy
    exporter cannot resolve the view operation inside the wrapper when combined
    with the ResNet's dynamic BatchNorm graph.  Tracing the backbone directly
    with the correct 4D shape avoids the issue entirely.

    At inference time, infer_community1_onnx.py adds the channel dim with
    fbank[np.newaxis] before passing to the ONNX session — matching this shape.
    """
    log("⚙️  Exporting embedding model ...")

    backbone = _extract_wespeaker_backbone(pipeline)
    # backbone is guaranteed to be an nn.Module by _extract_wespeaker_backbone
    backbone.eval()

    # Probe the backbone to find the right input shape.
    # WeSpeaker ResNet34 expects [B, 1, T, 80] (4D).
    # Some older checkpoints accept [B, T, 80] (3D) directly.
    # We try 4D first; fall back to 3D if that fails.
    dummy_4d = torch.zeros(1, 1, 200, 80)
    dummy_3d = torch.zeros(1, 200, 80)
    with torch.no_grad():
        try:
            _ = backbone(dummy_4d)
            dummy = torch.zeros(2, 1, 200, 80)
            input_is_4d = True
            log("   Backbone input shape: [B, 1, T, 80]  (4D)")
        except Exception:
            _ = backbone(dummy_3d)
            dummy = torch.zeros(2, 200, 80)
            input_is_4d = False
            log("   Backbone input shape: [B, T, 80]  (3D)")
    out_path = os.path.join(onnx_dir, "embedding.onnx")

    # dynamo=False: legacy TorchScript exporter handles ResNet conv/BN graphs
    # reliably; torch.export path in 2.10 has issues with dynamic T dimension.
    if input_is_4d:
        # [B, 1, T, 80] — time axis is dim 2
        dyn_axes = {"fbank": {0: "batch_size", 2: "num_frames"},
                    "embedding": {0: "batch_size"}}
    else:
        # [B, T, 80] — time axis is dim 1
        dyn_axes = {"fbank": {0: "batch_size", 1: "num_frames"},
                    "embedding": {0: "batch_size"}}

    # WeSpeakerResNet34.forward() returns (loss_tensor, embedding).
    # The loss is a dummy 0.0 scalar used only during training.
    # We name both outputs; inference uses "embedding" (index 1) only.
    torch.onnx.export(
        backbone,
        dummy,
        out_path,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["fbank"],
        output_names=["loss", "embedding"],
        dynamic_axes=dyn_axes,
        dynamo=False,
    )
    log(f"   Saved  embedding.onnx  ({file_size_mb(out_path):.1f} MB)")

    _simplify(out_path)
    _verify_embedding(backbone, out_path, T=200, input_is_4d=input_is_4d)

    return out_path, input_is_4d


# =============================================================================
# STAGE 3 — PIPELINE CONFIG
# =============================================================================

def stage_save_config(pipeline, onnx_dir: str,
                      embedding_input_4d: bool = True) -> str:
    """Extract clustering / segmentation hyper-parameters from the pipeline."""
    log("⚙️  Saving pipeline_config.json ...")

    params = pipeline.parameters(instantiated=True)

    config = {
        "segmentation": {
            "duration":   float(pipeline._segmentation.duration),
            "step":       float(pipeline._segmentation.step),
            "batch_size": int(getattr(pipeline._segmentation, "batch_size", 32)),
        },
        "segmentation_params": {
            "min_duration_off": float(
                params.get("segmentation", {}).get("min_duration_off", 0.0)
            ),
        },
        "embedding": {
            "exclude_overlap": bool(
                getattr(pipeline, "embedding_exclude_overlap", True)
            ),
            "batch_size": int(getattr(pipeline, "embedding_batch_size", 32)),
        },
        "clustering": {
            # AHC params (community-1 default; overridden by VBx at runtime)
            "method":           params.get("clustering", {}).get("method", "centroid"),
            "threshold":        float(
                params.get("clustering", {}).get("threshold", 0.715)
            ),
            "min_cluster_size": int(
                params.get("clustering", {}).get("min_cluster_size", 15)
            ),
            # VBx params (defaults; tune via CLI --vbx_fa / --vbx_fb at inference)
            "vbx_fa":            0.3,
            "vbx_fb":            17.0,
            "vbx_lda_dim":       0,
            "vbx_max_iter":      40,
            "vbx_init_threshold": 0.5,
        },
        "sample_rate":              16000,
        "num_powerset_classes":     7,
        "max_speakers_per_chunk":   3,
        "max_speakers_per_frame":   2,
        "embedding_dim":            256,
        # Written by export script from live backbone shape probe.
        # True  -> fbank input is [B, 1, T, 80]
        # False -> fbank input is [B, T, 80]  (WeSpeakerResNet34 default)
        "embedding_input_4d":       embedding_input_4d,  # set from probe
    }

    out_path = os.path.join(onnx_dir, "pipeline_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    log(f"   Saved  pipeline_config.json")
    return out_path


# =============================================================================
# STAGE 4 — INT8 QUANTIZATION
# =============================================================================

def stage_quantize(onnx_dir: str) -> List[str]:
    """Apply dynamic INT8 quantization to both FP32 ONNX models."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    log("⚙️  Quantizing to INT8 ...")
    out_paths = []
    for fname in FP32_MODELS:
        src  = os.path.join(onnx_dir, fname)
        dst  = src.replace(".onnx", ".int8.onnx")
        log(f"   {fname}  →  {os.path.basename(dst)} ...")
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        log(f"   Saved  {os.path.basename(dst)}  ({file_size_mb(dst):.1f} MB)")
        out_paths.append(dst)
    return out_paths


# =============================================================================
# STAGE 5 — PACKAGE
# =============================================================================

def stage_package(
    onnx_dir: str,
    model_dir: str,
    include_fp32: bool = False,
) -> str:
    """
    Pack ONNX models + config into a flat model.tar.gz.
    Archive extracts directly as segmentation.onnx, embedding.onnx, etc.
    — no subdirectory prefix, matching the Whisper script convention.

    By default only INT8 models are included. Pass include_fp32=True to
    also include FP32 models (roughly doubles the archive size).
    """
    log("🗜️  Packaging ...")

    # Decide which model files to include
    model_files = list(INT8_MODELS)
    if include_fp32:
        model_files += FP32_MODELS

    # Collect all target files (model files + extras like pipeline_config.json)
    all_files = model_files + ARCHIVE_EXTRAS
    found, missing = [], []
    for fname in all_files:
        fpath = os.path.join(onnx_dir, fname)
        if os.path.exists(fpath):
            found.append(fname)
        else:
            missing.append(fname)

    if missing:
        log(f"   ⚠️  Skipping missing files: {missing}")
    if not found:
        raise RuntimeError("No files found to package — aborting.")

    # the CC-BY-4.0 attribution file ships in every archive (see
    # ATTRIBUTION_MD) — write it fresh so a rebuild can never drop it
    attr_path = os.path.join(onnx_dir, "ATTRIBUTION.md")
    with open(attr_path, "w", encoding="utf-8") as f:
        f.write(ATTRIBUTION_MD)
    if "ATTRIBUTION.md" not in found:
        found.append("ATTRIBUTION.md")

    package_path = os.path.join(model_dir, "model.tar.gz")
    with tarfile.open(package_path, "w:gz") as tar:
        for fname in sorted(found):
            fpath = os.path.join(onnx_dir, fname)
            tar.add(fpath, arcname=fname)   # flat — no subdirectory prefix
            log(f"   + {fname:<45}  ({file_size_mb(fpath):.1f} MB)")

    pkg_mb    = file_size_mb(package_path)
    onnx_mb   = sum(
        file_size_mb(os.path.join(onnx_dir, f)) for f in found
    )
    reduction = 100 * (1 - pkg_mb / onnx_mb) if onnx_mb > 0 else 0
    log(f"✅ Package ready: {package_path}  "
        f"({pkg_mb:.1f} MB, {reduction:.0f}% reduction)")
    return package_path


# =============================================================================
# STAGE 6 — S3 UPLOAD
# =============================================================================

def stage_upload(
    package_path: str,
    bucket: str,
    skip_if_exists: bool = True,
) -> tuple[bool, Optional[str]]:
    """
    Upload model.tar.gz to:
      s3://{bucket}/opensource_models/pyannote/community-1/model.tar.gz

    Uses boto3 default credential chain (env vars / ~/.aws / IAM role).
    Multipart upload is handled automatically by boto3 for large files.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        log("❌ boto3 not installed — skipping upload  "
            "(pip install boto3)")
        return False, None

    if not os.path.exists(package_path):
        log(f"❌ Package not found: {package_path}")
        return False, None

    s3_key = f"{S3_KEY_PREFIX}/model.tar.gz"
    s3_uri = f"s3://{bucket}/{s3_key}"
    pkg_mb = file_size_mb(package_path)

    if skip_if_exists and is_uploaded_s3(bucket):
        log(f"⏭️  Already uploaded → {s3_uri}")
        return True, s3_uri

    log(f"☁️  Uploading → {s3_uri}  ({pkg_mb:.1f} MB) ...")
    try:
        boto3.client("s3").upload_file(
            package_path,
            bucket,
            s3_key,
            ExtraArgs={"ContentType": "application/gzip"},
            Callback=_make_progress_callback(os.path.getsize(package_path)),
        )
        log(f"✅ Uploaded → {s3_uri}")
        return True, s3_uri

    except NoCredentialsError:
        log("❌ No AWS credentials — set AWS_ACCESS_KEY_ID / "
            "AWS_SECRET_ACCESS_KEY or attach an IAM role")
        return False, None
    except ClientError as e:
        log(f"❌ S3 ClientError: {e}")
        return False, None
    except Exception as e:
        log(f"❌ Upload error: {e}")
        return False, None


# =============================================================================
# VERIFICATION HELPERS
# =============================================================================

def _simplify(onnx_path: str):
    try:
        import onnx
        from onnxsim import simplify
        model, ok = simplify(onnx.load(onnx_path))
        if ok:
            onnx.save(model, onnx_path)
            log(f"   Simplified: {os.path.basename(onnx_path)}")
        else:
            log(f"   ⚠️  onnxsim could not simplify {os.path.basename(onnx_path)}")
    except Exception as e:
        log(f"   ⚠️  Simplification skipped ({e})")


def _verify_segmentation(pt_model: nn.Module, onnx_path: str,
                          chunk_samples: int):
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx_path)
    sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 1, chunk_samples).astype(np.float32)
    with torch.no_grad():
        pt_out  = pt_model(torch.from_numpy(dummy)).numpy()
    ort_out = sess.run(["logits"], {"waveform": dummy})[0]
    diff    = np.abs(pt_out - ort_out).max()
    mark    = "✓" if diff < 1e-4 else "⚠ large!"
    log(f"   Segmentation PT↔ONNX max abs diff: {diff:.6f}  {mark}")


def _verify_embedding(backbone: nn.Module, onnx_path: str, T: int,
                      input_is_4d: bool = True):
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx_path)
    sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = (np.random.randn(1, 1, T, 80) if input_is_4d
             else np.random.randn(1, T, 80)).astype(np.float32)
    with torch.no_grad():
        pt_result = backbone(torch.from_numpy(dummy))
        # WeSpeakerResNet34 returns (loss_scalar, embedding) — take index 1
        pt_out = (pt_result[1] if isinstance(pt_result, (tuple, list))
                  else pt_result).numpy()
    # ONNX outputs [loss, embedding] — fetch by name
    ort_out = sess.run(["embedding"], {"fbank": dummy})[0]
    diff    = np.abs(pt_out - ort_out).max()
    mark    = "✓" if diff < 1e-4 else "⚠ large!"
    log(f"   Embedding  PT↔ONNX max abs diff: {diff:.6f}  {mark}")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(result: ExportResult, onnx_dir: str,
                  s3_bucket: Optional[str]) -> None:
    print("\n" + "=" * 70)
    print("📊 PYANNOTE COMMUNITY-1 ONNX EXPORT SUMMARY")
    print("=" * 70)

    status = "✅ OK" if result.success else "❌ FAILED"
    print(f"  Status      : {status}")
    if result.error:
        print(f"  Error       : {result.error}")
    print(f"  Duration    : {result.duration_sec:.1f} s")
    print(f"  ONNX files  : {result.onnx_size_mb:.1f} MB  (total in onnx/)")
    print(f"  Package     : {result.package_size_mb:.1f} MB  (model.tar.gz)")
    if s3_bucket:
        uploaded = "✅ " + result.s3_uri if result.uploaded else "❌ not uploaded"
        print(f"  S3          : {uploaded}")

    if result.success and os.path.isdir(onnx_dir):
        print(f"\n  Files in {onnx_dir}/")
        for fname in sorted(os.listdir(onnx_dir)):
            fpath = os.path.join(onnx_dir, fname)
            if os.path.isfile(fpath):
                print(f"    {fname:<45}  {file_size_mb(fpath):6.1f} MB")

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# HELPER — put sub-models into eval mode without calling pipeline.eval()
# pyannote.audio Pipeline inherits from pyannote.pipeline.Pipeline which is
# NOT an nn.Module, so .eval() does not exist on it.  We walk the known
# sub-model attributes and call .eval() only on actual nn.Module instances.
# =============================================================================

def _set_eval(pipeline) -> None:
    """Set all nn.Module sub-models inside a pyannote Pipeline to eval mode."""
    import torch.nn as nn

    def _try_eval(obj):
        if isinstance(obj, nn.Module):
            obj.eval()

    # Segmentation model
    if hasattr(pipeline, "_segmentation"):
        seg = pipeline._segmentation
        _try_eval(seg)
        if hasattr(seg, "model"):
            _try_eval(seg.model)

    # Embedding model (WeSpeaker wrapper chain)
    if hasattr(pipeline, "_embedding"):
        emb = pipeline._embedding
        _try_eval(emb)
        if hasattr(emb, "model"):
            _try_eval(emb.model)
            if hasattr(emb.model, "model"):
                _try_eval(emb.model.model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pyannote community-1 → ONNX, package, upload to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output layout:\n"
            f"  {BASE_DIR}/onnx/          ← ONNX models + pipeline_config.json\n"
            f"  {BASE_DIR}/model.tar.gz   ← flat archive (Lambda-ready)\n"
            "\n"
            "S3 destination (if --s3_bucket provided):\n"
            f"  s3://BUCKET/{S3_KEY_PREFIX}/model.tar.gz\n"
        ),
    )
    parser.add_argument(
        "--token", required=True,
        help="HuggingFace access token (must have accepted model conditions)",
    )
    parser.add_argument(
        "--output_dir", default=BASE_DIR,
        help=f"Root output directory (default: {BASE_DIR})",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no_quantize", action="store_true",
        help="Skip INT8 quantization",
    )
    parser.add_argument(
        "--no_simplify", action="store_true",
        help="Skip onnxsim graph simplification",
    )
    parser.add_argument(
        "--package_fp32", action="store_true",
        help="Include FP32 models in the archive in addition to INT8",
    )
    parser.add_argument(
        "--s3_bucket", metavar="BUCKET", default=None,
        help=(
            "S3 bucket to upload model.tar.gz. "
            f"Key: {S3_KEY_PREFIX}/model.tar.gz"
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all stages even if outputs already exist",
    )
    args = parser.parse_args()

    _check_imports()

    t_start  = time.time()
    result   = ExportResult()

    model_dir = args.output_dir
    onnx_dir  = os.path.join(model_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    try:
        # ── Skip logic ────────────────────────────────────────────────────
        already_exported   = is_exported(onnx_dir)   and not args.force
        already_quantized  = is_quantized(onnx_dir)  and not args.force
        already_packaged   = is_packaged(model_dir)  and not args.force

        # ── Load pipeline ─────────────────────────────────────────────────
        if not already_exported:
            log(f"📥 Loading {MODEL_ID} ...")
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(MODEL_ID, token=args.token)
            # pyannote Pipeline is NOT an nn.Module; eval() only the sub-models
            _set_eval(pipeline)
            log("   Loaded ✓")
        else:
            log("⏭️  FP32 ONNX models already present — skipping export")
            pipeline = None

        # ── Stage 1 & 2: Export ───────────────────────────────────────────
        if not already_exported:
            log("\n── Stage 1/2: Export ──────────────────────────────────")
            stage_export_segmentation(pipeline, onnx_dir, args.opset)
            _, emb_input_4d = stage_export_embedding(pipeline, onnx_dir, args.opset)
            log(f"   ONNX dir total: {dir_size_mb(onnx_dir):.1f} MB")
        else:
            # export was skipped — emb_input_4d will come from config or probe below
            emb_input_4d = False  # WeSpeakerResNet34 default (3D)
            pipeline = None

        # ── Stage 3: Config ───────────────────────────────────────────────
        cfg_path = os.path.join(onnx_dir, "pipeline_config.json")
        if not os.path.exists(cfg_path) or args.force:
            if pipeline is None:
                # Need to reload to read hyperparams if we skipped export
                log("📥 Reloading pipeline for config extraction ...")
                from pyannote.audio import Pipeline
                pipeline = Pipeline.from_pretrained(MODEL_ID, token=args.token)
                _set_eval(pipeline)
            log("\n── Stage 3: Config ────────────────────────────────────")
            stage_save_config(pipeline, onnx_dir,
                              embedding_input_4d=emb_input_4d)
        else:
            log("⏭️  pipeline_config.json already present — skipping")

        # Free pipeline memory now that both export and config are done
        pipeline = None
        gc.collect()

        # ── Stage 4: Quantize ─────────────────────────────────────────────
        if not args.no_quantize:
            if not already_quantized:
                log("\n── Stage 4: Quantize ──────────────────────────────────")
                stage_quantize(onnx_dir)
            else:
                log("⏭️  INT8 models already present — skipping quantization")
        else:
            log("⏭️  Skipping quantization (--no_quantize)")

        result.onnx_size_mb = round(dir_size_mb(onnx_dir), 1)

        # ── Stage 5: Package ──────────────────────────────────────────────
        log("\n── Stage 5: Package ───────────────────────────────────")
        if already_packaged:
            log(f"⏭️  model.tar.gz already present — skipping packaging  "
                f"({file_size_mb(os.path.join(model_dir, 'model.tar.gz')):.1f} MB)")
            package_path = os.path.join(model_dir, "model.tar.gz")
        else:
            # If no_quantize was set, we can only include FP32
            if args.no_quantize and not args.package_fp32:
                log("   ⚠️  --no_quantize set; forcing --package_fp32 "
                    "since no INT8 models exist")
                args.package_fp32 = True
            package_path = stage_package(
                onnx_dir, model_dir,
                include_fp32=args.package_fp32,
            )

        result.package_size_mb = round(file_size_mb(package_path), 1)

        # ── Stage 6: Upload ───────────────────────────────────────────────
        if args.s3_bucket:
            log("\n── Stage 6: Upload ────────────────────────────────────")
            result.uploaded, result.s3_uri = stage_upload(
                package_path,
                args.s3_bucket,
                skip_if_exists=not args.force,
            )
        else:
            log("\n── Stage 6: Upload ─── skipped (no --s3_bucket) ──────")

        result.success = True

    except Exception as e:
        result.error = str(e)
        log(f"❌ Fatal error: {e}")
        gc.collect()

    result.duration_sec = round(time.time() - t_start, 1)
    print_summary(result, onnx_dir, args.s3_bucket)
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()