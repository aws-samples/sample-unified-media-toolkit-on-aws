"""
CosyVoice3 Flow stage — speech tokens + conditioning → mel spectrogram.

Ports the verified onnx_verify/test_flow.py harness. Two deliberate
divergences from the community reference script, both verified against
cosyvoice/flow/flow_matching.py + the official cosyvoice3.yaml
(t_scheduler='cosine', inference_cfg_rate=0.7):

  1. Real classifier-free guidance: the estimator batch is
     [conditioned, UNconditioned] — batch index 1 gets zeroed
     mu/spks/cond. The community script duplicates the conditioned input
     into both halves, which collapses the CFG extrapolation
     (1+r)*a - r*b to a no-op when a == b.
  2. Cosine time schedule: t_span = 1 - cos(linspace(0,1) * pi/2), with
     per-step dt from consecutive span points — not uniform steps.

Step count: 6 is the default (listening-verified equal to 10 across
CrispASR Phase 1 and this ONNX port; mel-level correlation 0.9992),
10 available via parameter.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from lib.sessions import get_session

log = logging.getLogger(__name__)

TOKEN_MEL_RATIO = 2
INFERENCE_CFG_RATE = 0.7   # cosyvoice3.yaml decoder.cfm_params
DEFAULT_N_TIMESTEPS = 6


def _cosine_t_span(n_timesteps: int) -> np.ndarray:
    t = np.linspace(0, 1, n_timesteps + 1)
    return 1 - np.cos(t * 0.5 * np.pi)


def synthesize_mel(
    model_dir: str,
    speech_tokens: list[int],
    prompt_speech_tokens: list[int],
    spk_embedding: np.ndarray,
    prompt_mel: np.ndarray,
    n_timesteps: int = DEFAULT_N_TIMESTEPS,
    seed: int | None = None,
) -> np.ndarray:
    """speech tokens → [1, 80, T] mel (prompt portion already removed).

    prompt_mel: [1, T, 80], already length-aligned to prompt_speech_tokens
    (mel frames == 2 * tokens — frontend.extract_conditioning guarantees
    this; asserted here because a violation silently corrupts conditioning).
    """
    if prompt_mel.shape[1] != len(prompt_speech_tokens) * TOKEN_MEL_RATIO:
        raise ValueError(
            f"prompt_mel frames ({prompt_mel.shape[1]}) != "
            f"2 * prompt tokens ({len(prompt_speech_tokens)}) — "
            "conditioning was not aligned by frontend.extract_conditioning")

    rng = np.random.default_rng(seed)

    token_embed = get_session(f"{model_dir}/flow_token_embedding_fp16.onnx")
    pre_lookahead = get_session(f"{model_dir}/flow_pre_lookahead_fp16.onnx")
    speaker_proj = get_session(f"{model_dir}/flow_speaker_projection_fp16.onnx")
    # FP32 estimator, deliberately NOT the FP16 one. The community FP16
    # estimator export is numerically broken in a platform-dependent way:
    # probed with identical fixed inputs, its output correlates only ~0.47
    # (macOS arm64) / ~0.56 (linux arm64) with the official FP32 estimator,
    # and the two platforms disagree with each other (corr 0.925) — FP16
    # accumulation error in the 22-block DiT, not a load/runtime bug. The
    # FP32 estimator matches itself across platforms to 2e-3 and produced
    # the clean ASR-verified audio; the FP16 one produced garbled speech in
    # the Lambda container. Costs 2x file size (1.26 GB vs 634 MB) —
    # revisit only via our own FP16 re-export WITH accuracy verification.
    estimator = get_session(f"{model_dir}/flow.decoder.estimator.fp32.onnx")

    all_tokens = np.array([prompt_speech_tokens + speech_tokens], dtype=np.int64)
    t_emb = token_embed.run(None, {"token": all_tokens})[0]
    # pre_lookahead has token_mel_ratio=2 repeat_interleave baked into the
    # graph (verified empirically: 10 tokens in → 20 frames out)
    h = pre_lookahead.run(None, {"token_embedded": t_emb.astype(np.float32)})[0]
    mu = h.transpose(0, 2, 1)                       # [1, 80, mel_len]
    mel_len = mu.shape[2]
    mel_len1 = len(prompt_speech_tokens) * TOKEN_MEL_RATIO

    emb_norm = spk_embedding / (np.linalg.norm(spk_embedding, axis=1, keepdims=True) + 1e-8)
    spks = speaker_proj.run(None, {"embedding": emb_norm.astype(np.float32)})[0]

    conds = np.zeros((1, 80, mel_len), dtype=np.float32)
    conds[:, :, :mel_len1] = prompt_mel.transpose(0, 2, 1)
    mask = np.ones((1, 1, mel_len), dtype=np.float32)

    # Euler ODE with CFG — batch 0 conditioned, batch 1 unconditioned
    x = rng.standard_normal((1, 80, mel_len)).astype(np.float32)
    x_in = np.zeros((2, 80, mel_len), dtype=np.float32)
    mask_in = np.zeros((2, 1, mel_len), dtype=np.float32)
    mu_in = np.zeros((2, 80, mel_len), dtype=np.float32)
    t_in = np.zeros((2,), dtype=np.float32)
    spks_in = np.zeros((2, 80), dtype=np.float32)
    cond_in = np.zeros((2, 80, mel_len), dtype=np.float32)

    t_span = _cosine_t_span(n_timesteps)
    t, dt = t_span[0], t_span[1] - t_span[0]
    t0 = time.time()
    for step in range(1, len(t_span)):
        x_in[:] = x
        mask_in[:] = mask
        mu_in[0], mu_in[1] = mu[0], 0
        t_in[:] = t
        spks_in[0], spks_in[1] = spks[0], 0
        cond_in[0], cond_in[1] = conds[0], 0

        out = estimator.run(None, {"x": x_in, "mask": mask_in, "mu": mu_in,
                                   "t": t_in, "spks": spks_in, "cond": cond_in})[0]
        dphi_dt = ((1.0 + INFERENCE_CFG_RATE) * out[0:1]
                   - INFERENCE_CFG_RATE * out[1:2])
        x = x + dt * dphi_dt
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    log.info("Flow solve: %d steps, %d mel frames, %.2fs",
             n_timesteps, mel_len, time.time() - t0)
    return x[:, :, mel_len1:]  # drop the prompt portion
