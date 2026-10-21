#!/usr/bin/env python3
"""
export_bandit_v2.py
===================
Exports checkpoint-multi.ckpt (Bandit v2 multilingual) to ONNX INT8 using a
split encoder / per-stem decoder architecture.

Architecture (confirmed from bandit-mus64.yaml + v3-48k-smad-multi-v2b-w.yaml):
    fs=48000, n_fft=2048, hop_length=512, n_bands=64
    rnn_type="GRU", n_sqm_modules=8, emb_dim=128, rnn_dim=256
    in_channels=1 (mono), stems=["speech", "music", "sfx"]
    chunk_size=8s=384000 samples (OLA inference)

Split export rationale:
    The GRU encoder is ~90% of inference compute. By exporting encoder and
    decoders as separate ONNX files, a single Lambda function can run the
    encoder once per chunk and fan out q and x to all required decoder heads
    in-process — no external storage, no redundant GRU passes.

    encoder.onnx
        input:   mixture  [1, 1, T]
        outputs: q        [1, 64, N_frames, 128]    GRU output
                 x        [1, 1, 1025, N_frames, 2] STFT spectrogram (real/imag)

    speech_decoder.onnx / sfx_decoder.onnx / music_decoder.onnx
        inputs:  q        [1, 64, N_frames, 128]
                 x        [1, 1, 1025, N_frames, 2]
                 length   scalar int64  (original chunk sample count)
        output:  stem     [1, 1, T]

    Compute saving (speech + sfx vs old approach):
        Old: 2 independent full models = 2 GRU passes
        New: 1 encoder + 2 decoder heads = 1 GRU pass  (~50% saving)

Requirements (export machine only):
    pip install torch torchaudio onnx onnxruntime onnxsim boto3 librosa numpy

Usage:
    python export_bandit_v2.py \\
        --ckpt /path/to/checkpoint-multi.ckpt \\
        --repo /path/to/bandit-v2-main \\
        --s3_bucket my-bucket          # optional

Output layout:
    ./assets/models/bandit/
        onnx/
            encoder.onnx / encoder.int8.onnx
            speech_decoder.onnx / speech_decoder.int8.onnx
            music_decoder.onnx  / music_decoder.int8.onnx
            sfx_decoder.onnx    / sfx_decoder.int8.onnx
            model_config.json
        model.tar.gz     (flat archive — INT8 only, Lambda-ready)

Notes:
    • dynamo=False forces the legacy TorchScript exporter. The dynamo path
      (PyTorch >= 2.9 default) cannot handle dynamic GRU hidden state shapes.
    • Three checkpoint_sequential calls are patched at class level before
      instantiation — no source edits needed.
    • All complex tensor ops are replaced with real-valued [..., 2] equivalents;
      no complex dtype ever enters the ONNX graph.
    • iSTFT OLA uses scatter_add → ScatterElements(reduction='add', opset 17).
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import argparse
import math
import gc
import json
import os
import sys
import tarfile
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# CONSTANTS
# =============================================================================

BASE_DIR      = "./assets/models/bandit"
S3_KEY_PREFIX = "opensource_models/bandit/v2"
STEMS         = ["speech", "music", "sfx"]

ENCODER_FP32  = "encoder.onnx"
ENCODER_INT8  = "encoder.int8.onnx"
DECODER_FP32  = [f"{s}_decoder.onnx"      for s in STEMS]
DECODER_INT8  = [f"{s}_decoder.int8.onnx" for s in STEMS]
ALL_FP32      = [ENCODER_FP32] + DECODER_FP32
ALL_INT8      = [ENCODER_INT8] + DECODER_INT8
ARCHIVE_EXTRAS = ["model_config.json"]

# Confirmed architecture — do not change without re-validating the checkpoint
BANDIT_KWARGS = dict(
    in_channels=1,
    stems=STEMS,
    band_type="musical",
    n_bands=64,
    normalize_channel_independently=False,
    treat_channel_as_feature=True,
    n_sqm_modules=8,
    emb_dim=128,
    rnn_dim=256,
    bidirectional=True,
    rnn_type="GRU",
    mlp_dim=512,
    hidden_activation="Tanh",
    hidden_activation_kwargs=None,
    complex_mask=True,
    use_freq_weights=True,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    window_fn="hann_window",
    wkwargs=None,
    power=None,
    center=True,
    normalized=True,
    pad_mode="reflect",
    onesided=True,
    fs=48000,
)

FS            = 48000
CHUNK_SECONDS = 8.0
CHUNK_SAMPLES = int(CHUNK_SECONDS * FS)   # 384 000

# Encoder is traced at 2s to keep peak RAM under ~4 GB during export.
# Decoder fold output_size is computed analytically from CHUNK_SAMPLES
# (see CHUNK_N_FRAMES below) so the baked constant is correct at inference.
TRACE_SECONDS = 2.0
TRACE_SAMPLES = int(TRACE_SECONDS * FS)   # 96 000  — encoder trace only

# Analytical N_frames for CHUNK_SAMPLES with Bandit's actual STFT padding.
#
# Source analysis (bandit.py BaseBandit.instantitate_spectral):
#   self.stft = ta.transforms.Spectrogram(n_fft=2048, pad=0, center=True, ...)
#
# ta.transforms.Spectrogram with center=True, pad=0 calls torch.stft(center=True)
# which pads n_fft//2 per side. No additional manual padding anywhere in the model.
# Total pad = n_fft//2 = 1024 per side.
#
# Verified: T=384000 → N=751, out_len=386048, trim=1024, T_out=384000 ✓
# (Previously incorrect: assumed 3*n_fft//2 per side — that was our EncoderWrapper
#  adding its own pad on top of torch.stft's center pad, producing N=759.)
_n_fft  = 2048   # shorthand — do not change without re-reading BANDIT_KWARGS
_hop    = 512
_pad    = _n_fft // 2   # = 1024 per side — only torch.stft center=True padding
CHUNK_N_FRAMES: int = (CHUNK_SAMPLES + 2 * _pad - _n_fft) // _hop + 1   # 751
CHUNK_OUT_LEN:  int = (CHUNK_N_FRAMES - 1) * _hop + _n_fft               # 386048
CHUNK_T:        int = CHUNK_OUT_LEN - 2 * _pad                            # 384000
assert CHUNK_T == CHUNK_SAMPLES, \
    f"CHUNK_T={CHUNK_T} != CHUNK_SAMPLES={CHUNK_SAMPLES} — check n_fft/hop/pad"


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class ExportResult:
    success:         bool          = False
    error:           Optional[str] = None
    duration_sec:    float         = 0.0
    onnx_size_mb:    float         = 0.0
    package_size_mb: float         = 0.0
    uploaded:        bool          = False
    s3_uri:          Optional[str] = None


# =============================================================================
# LOGGING / SIZE HELPERS
# =============================================================================

def log(msg: str) -> None:
    print(f"[bandit] {msg}", flush=True)


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
    uploaded  = [0]
    last_mile = [-1]
    def callback(chunk: int) -> None:
        uploaded[0] += chunk
        pct       = int(100 * uploaded[0] / total_bytes)
        milestone = (pct // 25) * 25
        if milestone > last_mile[0]:
            last_mile[0] = milestone
            log(f"   ↑ {uploaded[0]/1024**2:.1f}/{total_bytes/1024**2:.1f} MB  ({pct}%)")
    return callback


# =============================================================================
# STATE CHECKS  (idempotency — skip stages already completed)
# =============================================================================

def is_exported(onnx_dir: str) -> bool:
    return all(os.path.exists(os.path.join(onnx_dir, f)) for f in ALL_FP32)

def is_quantized(onnx_dir: str) -> bool:
    return all(os.path.exists(os.path.join(onnx_dir, f)) for f in ALL_INT8)

def is_packaged(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "model.tar.gz"))

def is_uploaded_s3(bucket: str) -> bool:
    try:
        import boto3
        boto3.client("s3").head_object(Bucket=bucket,
                                       Key=f"{S3_KEY_PREFIX}/model.tar.gz")
        return True
    except Exception:
        return False


# =============================================================================
# IMPORT CHECK
# =============================================================================

def _check_imports() -> None:
    missing = []
    for pkg in ["torch", "torchaudio", "onnx", "onnxruntime", "onnxsim", "librosa"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages: {missing}. "
            "Install with: pip install torch torchaudio onnx onnxruntime onnxsim librosa"
        )


# =============================================================================
# PATCH 1 — remove checkpoint_sequential
# =============================================================================

def _patch_checkpoint_sequential() -> None:
    """
    Replace torch.utils.checkpoint.checkpoint_sequential with plain sequential
    forward calls in the three modules that use it.

    Patched at CLASS level before any model is instantiated — no source edits.

    Modules patched:
      SeqBandModellingModule  (tfmodel.py)   — GRU stack
      NormMLP                 (maskestim.py) — per-band mask MLP
      NormFC                  (bandsplit.py) — per-band band-split linear
    """
    from src.models.bandit.tfmodel import SeqBandModellingModule
    from src.models.bandit.maskestim import NormMLP
    from src.models.bandit.bandsplit import NormFC

    def _sbmm_forward(self, z):
        if self.parallel_mode:
            for sbm_pair in self.seqband:
                sbm_t, sbm_f = sbm_pair[0], sbm_pair[1]
                zt = sbm_t(z)
                zf = sbm_f(z.transpose(1, 2))
                z  = zt + zf.transpose(1, 2)
        else:
            z = self.seqband(z)
        return z

    def _normmpl_forward(self, qb):
        mb = self.combined(qb)
        mb = self.reshape_output(mb)
        return mb

    def _normfc_forward(self, xb):
        return self.combined(xb)

    SeqBandModellingModule.forward = _sbmm_forward
    NormMLP.forward                = _normmpl_forward
    NormFC.forward                 = _normfc_forward
    log("   Patched: SeqBandModellingModule, NormMLP, NormFC "
        "(checkpoint_sequential → plain forward)")


# =============================================================================
# PATCH 2 — fully real-valued arithmetic (no complex dtypes in ONNX graph)
# =============================================================================

def _patch_for_real_valued_export() -> None:
    """
    Replace every complex tensor op with real-valued equivalents.
    Convention throughout: complex [B,C,F,T] becomes float [B,C,F,T,2]
    where [...,0]=real, [...,1]=imag.

    The TorchScript ONNX exporter (dynamo=False) rejects ALL complex ops:
      view_as_complex / view_as_real / zeros(..., complex64) / etc.
    """
    from src.models.bandit.maskestim import NormMLP, OverlappingMaskEstimationModule
    from src.models.bandit.bandsplit import BandSplitModule

    # NormMLP.reshape_output: skip view_as_complex, return float [..., 2]
    def _normmpl_reshape_output(self, mb):
        batch, n_time, _ = mb.shape
        mb = mb.reshape(batch, n_time, self.in_channels, self.bandwidth, self.reim)
        mb = torch.permute(mb, (0, 2, 3, 1, 4))   # [B, C, BW, T, 2]
        return mb

    NormMLP.reshape_output = _normmpl_reshape_output

    # OverlappingMaskEstimationModule.forward: accumulate into float [..., 2]
    def _overlap_forward(self, q):
        batch, n_bands, n_time, emb_dim = q.shape
        masks = torch.zeros(
            (batch, self.in_channels, self.n_freq, n_time, 2),
            device=q.device, dtype=torch.float32,
        )
        for im in range(n_bands):
            fstart, fend = self.band_specs[im]
            mask = self.compute_mask(q, im)   # [B, C, BW, T, 2]
            if self.use_freq_weights:
                fw   = self.get_buffer(f"freq_weights/{im}")
                mask = mask * fw.reshape(1, 1, -1, 1, 1)
            # torch.cat avoids in-place slice assignment (rejected by TorchScript)
            masks = torch.cat([
                masks[:, :, :fstart,    :, :],
                masks[:, :, fstart:fend,:, :] + mask,
                masks[:, :, fend:,      :, :],
            ], dim=2)
        return masks   # [B, C, F, T, 2]

    OverlappingMaskEstimationModule.forward = _overlap_forward

    # BandSplitModule.forward: input is float [B,C,F,T,2]; skip view_as_real
    def _bandsplit_forward(self, x: torch.Tensor):
        batch, in_chan, n_freq, n_time, _ = x.shape
        z = torch.zeros(
            size=(batch, self.n_bands, n_time, self.emb_dim), device=x.device
        )
        x = torch.permute(x, (0, 3, 1, 2, 4)).contiguous()   # [B, T, C, F, 2]
        for i, nfm in enumerate(self.norm_fc_modules):
            fstart, fend = self.band_specs[i]
            xb = x[:, :, :, fstart:fend, :]
            xb = torch.reshape(xb, (batch, n_time, -1))
            z[:, i, :, :] = nfm(xb)
        return z

    BandSplitModule.forward = _bandsplit_forward

    log("   Patched: NormMLP.reshape_output, OverlappingMaskEstimationModule.forward, "
        "BandSplitModule.forward (fully real-valued)")


# =============================================================================
# MODEL BUILD + CHECKPOINT LOAD
# =============================================================================

def _build_model() -> nn.Module:
    from src.models.bandit.bandit import Bandit
    model = Bandit(**BANDIT_KWARGS)
    n = sum(p.numel() for p in model.parameters())
    log(f"   Model built — {n:,} parameters  ({n/1e6:.1f} M)")
    return model


def _load_checkpoint(model: nn.Module, ckpt_path: str) -> nn.Module:
    """Load Lightning checkpoint; strip 'model.' prefix from state dict keys."""
    log(f"   Loading: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in raw:
        raise ValueError(f"Checkpoint has no 'state_dict'. Keys: {list(raw.keys())}")
    state_dict = raw["state_dict"]
    stripped, skipped = {}, []
    for k, v in state_dict.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v
        else:
            skipped.append(k)
    if skipped:
        log(f"   Skipped {len(skipped)} non-model keys "
            f"(e.g. {skipped[:3]}{'...' if len(skipped) > 3 else ''})")
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        log(f"   ⚠️  Missing keys  ({len(missing)}): {missing[:5]}")
    if unexpected:
        log(f"   ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if not missing and not unexpected:
        log("   Checkpoint loaded cleanly ✓")
    return model


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _get_window(model: nn.Module, win_length: int) -> torch.Tensor:
    for _, buf in model.stft.named_buffers():
        if buf.shape == (win_length,):
            return buf.clone()
    log("   ⚠️  Window buffer not found — using torch.hann_window fallback")
    return torch.hann_window(win_length)


def _make_idft_basis(n_fft: int, normalized: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute real iDFT basis matrices [F, n_fft] in float32.

    Standard one-sided iDFT scale:
      k=0, k=N/2: 1/N
      k=1..N/2-1: 2/N   (factor-2 accounts for conjugate half)

    If normalized=True (torch.stft divided output by sqrt(N)), multiply
    the basis by sqrt(N) to undo that: scale /= sqrt(N) instead of N.
    Without this the iSTFT output has a gain error of 1/sqrt(N) ~ -33 dB.
    """
    F    = n_fft // 2 + 1
    k    = torch.arange(F,     dtype=torch.float32).unsqueeze(1)
    n    = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
    pi_f = torch.tensor(torch.pi, dtype=torch.float32)  # float32 — avoid float64 promotion
    phase = (2.0 * pi_f * k * n / float(n_fft)).to(torch.float32)
    scale = torch.ones(F, 1, dtype=torch.float32)
    scale[1:-1] = 2.0
    if normalized:
        scale = scale / math.sqrt(float(n_fft))   # invert sqrt(N) STFT normalization
    else:
        scale = scale / float(n_fft)
    return (torch.cos(phase) * scale).to(torch.float32), \
           (torch.sin(phase) * scale).to(torch.float32)


# =============================================================================
# ENCODER WRAPPER
# mixture [1,1,T] → q [1,64,N_frames,128], x [1,1,1025,N_frames,2]
# =============================================================================

class EncoderWrapper(nn.Module):
    """STFT → BandSplit → GRU.  Outputs q and x for all decoder heads."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model      = model
        self.n_fft      = BANDIT_KWARGS["n_fft"]
        self.hop_length = BANDIT_KWARGS["hop_length"]
        self.win_length = BANDIT_KWARGS["win_length"]
        self.normalized = BANDIT_KWARGS["normalized"]
        self.center     = BANDIT_KWARGS["center"]
        self.pad_mode   = BANDIT_KWARGS["pad_mode"]
        self.register_buffer("window", _get_window(model, self.win_length))

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """[B,C,T] → [B,C,F,N_frames,2]"""
        B, C, T = audio.shape
        if self.center:
            # Note: the real model applies its own padding before torch.stft.
            # This _stft method is used only for ONNX export tracing reference;
            # the actual encoder export runs model.forward() which handles padding.
            audio = torch.nn.functional.pad(
                audio, (self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode
            )
        audio_2d = audio.reshape(B * C, -1)
        out = torch.stft(
            audio_2d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,           # we apply padding manually above; avoid double-pad
            normalized=self.normalized,
            onesided=True,
            return_complex=False,   # real-valued; see dynamo note in module docstring
        )   # [B*C, F, N, 2]
        F, N = out.shape[1], out.shape[2]
        return out.reshape(B, C, F, N, 2)

    def forward(self, mixture: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mixture: [1, 1, T]
        returns: q [1, n_bands, N_frames, emb_dim]
                 x [1, 1, F, N_frames, 2]
        """
        x = self._stft(mixture)          # [1, 1, F, N_frames, 2]
        z = self.model.band_split(x)     # [1, n_bands, N_frames, emb_dim]
        q = self.model.tf_model(z)       # [1, n_bands, N_frames, emb_dim]
        return q, x


# =============================================================================
# DECODER WRAPPER
# q [1,64,N_frames,128] + x [1,1,F,N_frames,2] + length scalar → stem [1,1,T]
# =============================================================================

class DecoderWrapper(nn.Module):
    """
    mask_estim → complex multiply → iSTFT.

    OLA implementation: torch.scatter_add → ONNX ScatterElements(reduction='add').
    -------------------------------------------------------------------------------
    History of failed approaches on this ORT 1.24.x ARM64 CPU build:
      - index_add / ScatterElements          → duplicate indices unsupported (pre-opset 16)
      - conv_transpose1d(groups=n_fft)       → ConvTranspose(11) missing for grouped case
      - F.fold / col2im                      → Col2Im(18) absent from this build
      - N_frames Python loop                 → wrong frame count baked at trace time
      - n_fft Python loop (2048 Add nodes)   → 14-min ORT graph opt; TorchScript drops writes
      - conv_transpose1d(groups=1, k=n_fft)  → ConvTranspose(11) absent for this kernel shape
      - T_in as third ONNX input (tensor)    → .item()/.narrow() forced tracer to bake constant

    ScatterElements with reduction='add' is registered at opsets 16–17 and 18+
    in this build and explicitly supports duplicate indices (required for OLA
    where hop < n_fft means each output position accumulates multiple frames).

    OLA as scatter_add (→ ONNX ScatterElements reduction='add'):
        updates = frames.reshape(BC, N*n_fft)               # [BC, N*n_fft]
        linear  = arange(N*n_fft)                           # [N*n_fft]
        indices = (linear // n_fft)*hop + (linear % n_fft)  # target positions
        indices = indices.expand(BC, -1)                    # [BC, N*n_fft]
        data    = zeros(BC, out_len)
        output  = scatter_add(data, dim=1, index=indices, src=updates)

    Indices computed at runtime from N_frames so they are correct for any chunk.
    The index formula uses only Div / Mod / Mul / Add — all universal ONNX ops.

    Trim (center=True):
        Source model calls torch.istft(s, length=T) where T = original_audio.shape[-1].
        torch.istft with length=T simply right-truncates: output = output[:T].
        Since TRACE_SAMPLES == CHUNK_SAMPLES == 384000 is a fixed production constant,
        T and the corresponding trim bounds are baked as static Python ints at trace time.
        No third ONNX input needed. The ONNX graph contains a Slice with constant bounds.

    Requires opset >= 16 for ScatterElements(reduction='add'). Export with --opset 17.
    """

    def __init__(self, model: nn.Module, stem: str, trace_n_frames: int):
        """
        trace_n_frames : no longer used for OLA buffer sizing (that is now dynamic).
                         Kept as parameter for API stability; may be removed.
        """
        super().__init__()
        self.model      = model
        self.stem       = stem
        self.n_fft      = BANDIT_KWARGS["n_fft"]
        self.hop_length = BANDIT_KWARGS["hop_length"]
        self.win_length = BANDIT_KWARGS["win_length"]
        self.center     = BANDIT_KWARGS["center"]

        # Static Python ints — baked as constants by the TorchScript tracer.
        # out_len = OLA output length for the trace N_frames.
        # trim_start, trim_end = slice bounds matching torch.istft(length=CHUNK_SAMPLES).
        # torch.istft with length=T right-truncates: it removes leading padding then
        # takes the first T samples.  For center=True that means:
        #   leading_pad = n_fft // 2
        #   output = ola_output[leading_pad : leading_pad + T]
        # T == CHUNK_SAMPLES == TRACE_SAMPLES (enforced by TRACE_SAMPLES = CHUNK_SAMPLES).
        n_fft = self.n_fft
        self._trim_start: int = n_fft // 2          # leading_pad from center=True STFT
        self._trim_end:   int = self._trim_start + CHUNK_SAMPLES

        window = _get_window(model, self.win_length)
        self.register_buffer("window", window)

        cos_basis, sin_basis = _make_idft_basis(self.n_fft, normalized=BANDIT_KWARGS["normalized"])
        self.register_buffer("cos_basis", cos_basis)
        self.register_buffer("sin_basis", sin_basis)

        syn_win = torch.zeros(n_fft, dtype=torch.float32)
        syn_win[:self.win_length] = window
        self.register_buffer("syn_window", syn_win)     # [n_fft]

        win_sq = syn_win ** 2
        self.register_buffer("syn_window_sq", win_sq)  # [n_fft]

    def _complex_multiply(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x[..., 0], x[..., 1]
        m_r, m_i = m[..., 0], m[..., 1]
        return torch.stack([x_r * m_r - x_i * m_i,
                            x_r * m_i + x_i * m_r], dim=-1)

    def _istft(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec : [B, C, F, N_frames, 2] → waveform [B, C, CHUNK_SAMPLES]

        OLA via scatter_add with dynamic buffer size, then static trim.
        The OLA buffer size is computed from frames.shape[1] (dynamic ONNX dim).
        Trim slice bounds are static Python ints (CHUNK_SAMPLES is fixed).
        """
        B, C, F, N_frames, _ = spec.shape
        n_fft = self.n_fft
        hop   = self.hop_length

        spec_2d = spec.reshape(B * C, F, N_frames, 2)
        spec_r  = spec_2d[..., 0]
        spec_i  = spec_2d[..., 1]

        cos_basis  = self.cos_basis.to(spec_r.dtype)
        sin_basis  = self.sin_basis.to(spec_r.dtype)
        syn_window = self.syn_window.to(spec_r.dtype)
        win_sq     = self.syn_window_sq.to(spec_r.dtype)

        BC = B * C

        # 1. Real iDFT → windowed frames [BC, N_frames, n_fft]
        frames = (torch.einsum('bfn,fs->bns', spec_r, cos_basis)
                - torch.einsum('bfn,fs->bns', spec_i, sin_basis))
        frames = frames * syn_window   # [BC, N_frames, n_fft]

        # out_len derived from frames.shape[1] (dynamic dim) so the tracer emits
        # Shape → arithmetic ONNX nodes rather than a baked constant.
        # frames.shape[1] == N_frames at runtime (e.g. 751 or 755).
        out_len = (frames.shape[1] - 1) * hop + n_fft   # dynamic in ONNX graph

        # 2. OLA via scatter_add → ScatterElements(reduction='add')
        linear  = torch.arange(N_frames * n_fft, dtype=torch.long, device=spec.device)
        indices = (linear // n_fft) * hop + (linear % n_fft)   # [N*n_fft]
        indices = indices.unsqueeze(0).expand(BC, -1)           # [BC, N*n_fft]

        updates = frames.reshape(BC, N_frames * n_fft)          # [BC, N*n_fft]

        data   = frames.new_zeros(BC, out_len)                  # dynamic ConstantOfShape
        output = data.scatter_add(1, indices, updates)

        # 3. OLA norm
        win_sq_tiled = win_sq.unsqueeze(0).expand(N_frames, -1).reshape(1, N_frames * n_fft)
        win_sq_tiled = win_sq_tiled.expand(BC, -1)
        norm_data    = frames.new_zeros(BC, out_len)            # same dynamic size
        norm         = norm_data.scatter_add(1, indices, win_sq_tiled)

        output = output / (norm + 1e-8)

        # 4. Trim — static Python int slice bounds baked at trace time.
        # Matches torch.istft(length=CHUNK_SAMPLES) with center=True:
        # remove n_fft//2 leading samples, keep exactly CHUNK_SAMPLES.
        if self.center:
            output = output[:, self._trim_start : self._trim_end]

        return output.reshape(B, C, -1)

    def forward(self,
                q: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        """
        q : [1, n_bands, N_frames, emb_dim]
        x : [1, 1, F, N_frames, 2]
        """
        m = self.model.mask_estim[self.stem](q)
        s = self._complex_multiply(x, m)
        return self._istft(s)


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def _onnx_export(wrapper, dummy_inputs, out_path, input_names, output_names,
                 dynamic_axes, opset):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="stft with return_complex=False")
        warnings.filterwarnings("ignore", message="Exporting a model to ONNX with a batch_size")
        warnings.filterwarnings("ignore", message="Constant folding - Only steps=1")
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            out_path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )


def _simplify(onnx_path: str) -> None:
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


def _verify(wrapper, onnx_path, input_names, output_names, dummy_inputs,
            tolerance=1e-4):
    import onnx as _onnx
    import onnxruntime as ort
    _onnx.checker.check_model(onnx_path)
    with torch.no_grad():
        pt_outs = wrapper(*dummy_inputs)
    if not isinstance(pt_outs, tuple):
        pt_outs = (pt_outs,)
    sess     = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed     = {n: t.numpy() for n, t in zip(input_names, dummy_inputs)}
    ort_outs = sess.run(output_names, feed)
    for name, pt_out, ort_out in zip(output_names, pt_outs, ort_outs):
        diff = np.abs(pt_out.numpy() - ort_out).max()
        mark = "✓" if diff < tolerance else "⚠  LARGE — investigate!"
        log(f"   {name:<22} PT↔ONNX max abs diff: {diff:.6f}  {mark}")


# =============================================================================
# STAGE 1 — EXPORT
# =============================================================================

def stage_export(model: nn.Module, onnx_dir: str, opset: int) -> None:
    """
    Export encoder (traced at 2s) + each decoder head (traced at 8s).

    Encoder is traced at TRACE_SAMPLES=96000 to keep peak RAM under ~4 GB.
    Decoders are traced at CHUNK_N_FRAMES=751 using zero dummy tensors of
    the correct shape — no 8s encoder forward pass required.
    The fold output_size constants in each decoder are therefore exactly
    correct for CHUNK_SAMPLES=384000 inference.
    """
    log("⚙️  Exporting encoder + decoder heads to ONNX FP32 ...")
    log(f"   Encoder trace : {TRACE_SAMPLES} samples ({TRACE_SECONDS}s)")
    log(f"   Decoder trace : CHUNK_N_FRAMES={CHUNK_N_FRAMES}  "
        f"out_len={CHUNK_OUT_LEN}  T={CHUNK_T} (= {CHUNK_SAMPLES} ✓)")
    log("   Legacy TorchScript exporter (dynamo=False) — expected.")

    trace_mixture = torch.zeros(1, 1, TRACE_SAMPLES)

    # ── Encoder (traced at 2s) ────────────────────────────────────────────────
    log("\n   [encoder]")
    enc = EncoderWrapper(model)
    enc.eval()

    with torch.no_grad():
        q_enc_trace, x_enc_trace = enc(trace_mixture)
    log(f"   q: {list(q_enc_trace.shape)}   x: {list(x_enc_trace.shape)}")

    enc_path = os.path.join(onnx_dir, ENCODER_FP32)
    _onnx_export(
        wrapper      = enc,
        dummy_inputs = (trace_mixture,),
        out_path     = enc_path,
        input_names  = ["mixture"],
        output_names = ["q", "x"],
        dynamic_axes = {
            "mixture": {2: "samples"},
            "q":       {2: "n_frames"},
            "x":       {3: "n_frames"},
        },
        opset = opset,
    )
    log(f"   Saved  {ENCODER_FP32}  ({file_size_mb(enc_path):.1f} MB)")
    _simplify(enc_path)
    _verify(enc, enc_path,
            input_names=["mixture"], output_names=["q", "x"],
            dummy_inputs=(trace_mixture,))
    del enc, q_enc_trace, x_enc_trace
    gc.collect()

    # ── Decoder dummy inputs at CHUNK size (zero tensors, no encoder pass) ────
    # Shape from confirmed architecture: q=[1,64,N,128]  x=[1,1,1025,N,2]
    n_bands  = BANDIT_KWARGS["n_bands"]
    emb_dim  = BANDIT_KWARGS["emb_dim"]
    n_fft    = BANDIT_KWARGS["n_fft"]
    F_bins   = n_fft // 2 + 1   # 1025
    N        = CHUNK_N_FRAMES   # 751

    q_dummy = torch.zeros(1, n_bands, N, emb_dim)        # [1, 64, 751, 128]
    x_dummy = torch.zeros(1, 1, F_bins, N, 2)            # [1, 1, 1025, 751, 2]
    log(f"\n   Decoder dummy inputs (zeros):  "
        f"q={list(q_dummy.shape)}  x={list(x_dummy.shape)}")

    # ── Decoder heads (each traced at CHUNK_N_FRAMES) ─────────────────────────
    for stem in STEMS:
        log(f"\n   [{stem}_decoder]")
        dec = DecoderWrapper(model, stem, CHUNK_N_FRAMES)
        dec.eval()

        dummy_inputs = (q_dummy, x_dummy)

        with torch.no_grad():
            out_trace = dec(*dummy_inputs)
        log(f"   output: {list(out_trace.shape)}  (expected [1, 1, {CHUNK_T}])")

        dec_path = os.path.join(onnx_dir, f"{stem}_decoder.onnx")
        _onnx_export(
            wrapper      = dec,
            dummy_inputs = dummy_inputs,
            out_path     = dec_path,
            input_names  = ["q", "x"],
            output_names = [stem],
            dynamic_axes = {
                "q":  {2: "n_frames"},
                "x":  {3: "n_frames"},
                stem: {2: "samples"},
            },
            opset = opset,
        )
        log(f"   Saved  {stem}_decoder.onnx  ({file_size_mb(dec_path):.1f} MB)")
        # onnxsim skipped: would downgrade ScatterElements opset, breaking reduction='add'.
        _verify(dec, dec_path,
                input_names=["q", "x"], output_names=[stem],
                dummy_inputs=dummy_inputs)
        del dec
        gc.collect()

    log(f"\n   ONNX dir total: {dir_size_mb(onnx_dir):.1f} MB")


# =============================================================================
# STAGE 2 — CONFIG
# =============================================================================

def stage_save_config(onnx_dir: str) -> str:
    out_path = os.path.join(onnx_dir, "model_config.json")
    config = {
        "model":           "bandit-v2-multilingual",
        "architecture":    "split_encoder_decoder",
        "checkpoint":      "checkpoint-multi.ckpt",
        "stems":           STEMS,
        "sample_rate":     FS,
        "chunk_seconds":   CHUNK_SECONDS,
        "chunk_samples":   CHUNK_SAMPLES,
        "hop_seconds":     1.0,
        "hop_samples":     FS,
        "n_fft":           BANDIT_KWARGS["n_fft"],
        "hop_length":      BANDIT_KWARGS["hop_length"],
        "n_bands":         BANDIT_KWARGS["n_bands"],
        "rnn_type":        BANDIT_KWARGS["rnn_type"],
        "n_sqm_modules":   BANDIT_KWARGS["n_sqm_modules"],
        "onnx_opset":      17,
        "quantization":    "dynamic_int8",
        "files": {
            "encoder":  ENCODER_INT8,
            "decoders": {s: f"{s}_decoder.int8.onnx" for s in STEMS},
        },
        "decoder_inputs": ["q", "x"],
        "inference_pattern": (
            "q, x = encoder.run(['q','x'], {'mixture': chunk})  # once per chunk\n"
            "speech = speech_decoder.run(['speech'], {'q':q,'x':x})\n"
            "sfx    = sfx_decoder.run(['sfx'],    {'q':q,'x':x})\n"
            "# music_decoder optional — load only when needed"
        ),
        "ola": {
            "window_fn":     "hann_window",
            "pad_mode":      "reflect",
            "chunk_samples": CHUNK_SAMPLES,
            "hop_samples":   FS,
        },
    }
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    log("   Saved  model_config.json")
    return out_path


# =============================================================================
# STAGE 3 — INT8 QUANTIZATION
# =============================================================================

def stage_quantize(onnx_dir: str) -> List[str]:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    log("⚙️  Quantizing to INT8 ...")
    out_paths = []
    for fname in ALL_FP32:
        src = os.path.join(onnx_dir, fname)
        dst = src.replace(".onnx", ".int8.onnx")
        log(f"   {fname}  →  {os.path.basename(dst)} ...")
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        fp32_mb = file_size_mb(src)
        int8_mb = file_size_mb(dst)
        pct     = (1 - int8_mb / fp32_mb) * 100 if fp32_mb > 0 else 0
        log(f"   Saved  {os.path.basename(dst)}  ({int8_mb:.1f} MB, {pct:.0f}% smaller)")
        out_paths.append(dst)
    return out_paths


# =============================================================================
# STAGE 4 — PACKAGE
# =============================================================================

def stage_package(onnx_dir: str, model_dir: str,
                  include_fp32: bool = False) -> str:
    """Pack INT8 models + config into a flat model.tar.gz (Lambda-ready)."""
    log("🗜️  Packaging ...")
    model_files = list(ALL_INT8)
    if include_fp32:
        model_files += ALL_FP32
    all_files      = model_files + ARCHIVE_EXTRAS
    found, missing = [], []
    for fname in all_files:
        fpath = os.path.join(onnx_dir, fname)
        (found if os.path.exists(fpath) else missing).append(fname)
    if missing:
        log(f"   ⚠️  Skipping missing files: {missing}")
    if not found:
        raise RuntimeError("No files found to package.")
    package_path = os.path.join(model_dir, "model.tar.gz")
    with tarfile.open(package_path, "w:gz") as tar:
        for fname in sorted(found):
            fpath = os.path.join(onnx_dir, fname)
            tar.add(fpath, arcname=fname)
            log(f"   + {fname:<52}  ({file_size_mb(fpath):.1f} MB)")
    pkg_mb  = file_size_mb(package_path)
    onnx_mb = sum(file_size_mb(os.path.join(onnx_dir, f)) for f in found)
    pct     = 100 * (1 - pkg_mb / onnx_mb) if onnx_mb > 0 else 0
    log(f"✅ Package: {package_path}  ({pkg_mb:.1f} MB, {pct:.0f}% compression)")
    return package_path


# =============================================================================
# STAGE 5 — S3 UPLOAD
# =============================================================================

def stage_upload(package_path: str, bucket: str,
                 skip_if_exists: bool = True) -> Tuple[bool, Optional[str]]:
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        log("❌ boto3 not installed — skipping upload  (pip install boto3)")
        return False, None
    if not os.path.exists(package_path):
        log(f"❌ Package not found: {package_path}")
        return False, None
    s3_key = f"{S3_KEY_PREFIX}/model.tar.gz"
    s3_uri = f"s3://{bucket}/{s3_key}"
    if skip_if_exists and is_uploaded_s3(bucket):
        log(f"⏭️  Already uploaded → {s3_uri}")
        return True, s3_uri
    log(f"☁️  Uploading → {s3_uri}  ({file_size_mb(package_path):.1f} MB) ...")
    try:
        boto3.client("s3").upload_file(
            package_path, bucket, s3_key,
            ExtraArgs={"ContentType": "application/gzip"},
            Callback=_make_progress_callback(os.path.getsize(package_path)),
        )
        log(f"✅ Uploaded → {s3_uri}")
        return True, s3_uri
    except NoCredentialsError:
        log("❌ No AWS credentials — set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
        return False, None
    except Exception as e:
        log(f"❌ Upload error: {e}")
        return False, None


# =============================================================================
# SUMMARY
# =============================================================================

def _print_summary(result: ExportResult, onnx_dir: str,
                   s3_bucket: Optional[str]) -> None:
    print("\n" + "=" * 70)
    print("📊 BANDIT V2 ONNX EXPORT SUMMARY")
    print("=" * 70)
    print(f"  Status      : {'✅ OK' if result.success else '❌ FAILED'}")
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
                print(f"    {fname:<52}  {file_size_mb(fpath):6.1f} MB")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Bandit v2 → split encoder/decoder ONNX INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output layout:\n"
            f"  {BASE_DIR}/onnx/         ← encoder + decoder ONNX models\n"
            f"  {BASE_DIR}/model.tar.gz  ← flat archive (Lambda-ready)\n"
            "\n"
            "Inference pattern (Lambda handler):\n"
            "  q, x  = enc.run(['q','x'], {'mixture': chunk})\n"
            "  speech = speech_dec.run(['speech'], {'q':q,'x':x})\n"
            "  sfx    = sfx_dec.run(['sfx'],       {'q':q,'x':x})\n"
        ),
    )
    parser.add_argument("--ckpt", required=True,
                        help="Path to checkpoint-multi.ckpt")
    parser.add_argument("--repo", required=True,
                        help="Path to bandit-v2-main (contains src/)")
    parser.add_argument("--output_dir", default=BASE_DIR,
                        help=f"Root output directory (default: {BASE_DIR})")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--no_quantize", action="store_true",
                        help="Skip INT8 quantization")
    parser.add_argument("--no_simplify", action="store_true",
                        help="Skip onnxsim graph simplification")
    parser.add_argument("--package_fp32", action="store_true",
                        help="Include FP32 models in the archive")
    parser.add_argument("--s3_bucket", metavar="BUCKET", default=None,
                        help=f"S3 bucket. Key: {S3_KEY_PREFIX}/model.tar.gz")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages even if outputs already exist")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    _check_imports()

    t_start = time.time()
    result  = ExportResult()

    model_dir = args.output_dir
    onnx_dir  = os.path.join(model_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    try:
        already_exported  = is_exported(onnx_dir)  and not args.force
        already_quantized = is_quantized(onnx_dir) and not args.force
        already_packaged  = is_packaged(model_dir) and not args.force

        # ── Stage 0: Patch + build model ──────────────────────────────────
        if not already_exported:
            log("\n── Stage 0: Patch + build model ───────────────────────")
            _patch_checkpoint_sequential()
            _patch_for_real_valued_export()
            model = _build_model()
            model = _load_checkpoint(model, args.ckpt)
            model.eval()
        else:
            log("⏭️  FP32 ONNX models already present — skipping export")
            model = None

        # ── Stage 1: Export ───────────────────────────────────────────────
        if not already_exported:
            log("\n── Stage 1: Export ────────────────────────────────────")
            stage_export(model, onnx_dir, args.opset)
            del model
            gc.collect()

        # ── Stage 2: Config ───────────────────────────────────────────────
        cfg_path = os.path.join(onnx_dir, "model_config.json")
        if not os.path.exists(cfg_path) or args.force:
            log("\n── Stage 2: Config ────────────────────────────────────")
            stage_save_config(onnx_dir)
        else:
            log("⏭️  model_config.json already present — skipping")

        # ── Stage 3: Quantize ─────────────────────────────────────────────
        if not args.no_quantize:
            if not already_quantized:
                log("\n── Stage 3: Quantize ──────────────────────────────────")
                stage_quantize(onnx_dir)
            else:
                log("⏭️  INT8 models already present — skipping quantization")
        else:
            log("⏭️  Skipping quantization (--no_quantize)")

        result.onnx_size_mb = round(dir_size_mb(onnx_dir), 1)

        # ── Stage 4: Package ──────────────────────────────────────────────
        log("\n── Stage 4: Package ───────────────────────────────────")
        if already_packaged:
            log("⏭️  model.tar.gz already present — skipping")
            package_path = os.path.join(model_dir, "model.tar.gz")
        else:
            if args.no_quantize and not args.package_fp32:
                log("   ⚠️  --no_quantize set; forcing --package_fp32")
                args.package_fp32 = True
            package_path = stage_package(onnx_dir, model_dir,
                                         include_fp32=args.package_fp32)

        result.package_size_mb = round(file_size_mb(package_path), 1)

        # ── Stage 5: Upload ───────────────────────────────────────────────
        if args.s3_bucket:
            log("\n── Stage 5: Upload ────────────────────────────────────")
            result.uploaded, result.s3_uri = stage_upload(
                package_path, args.s3_bucket,
                skip_if_exists=not args.force,
            )
        else:
            log("\n── Stage 5: Upload ─── skipped (no --s3_bucket) ──────")

        result.success = True

    except Exception as e:
        result.error = str(e)
        log(f"❌ Fatal error: {e}")
        traceback.print_exc()
        gc.collect()

    result.duration_sec = round(time.time() - t_start, 1)
    _print_summary(result, onnx_dir, args.s3_bucket)
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()