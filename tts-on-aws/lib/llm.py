"""
CosyVoice3 LLM stage — text + reference conditioning → speech tokens.

Ports the verified onnx_verify/test_llm.py harness. Three deliberate
divergences from the community reference script (ayousanz
onnx_inference_pure.py), each verified against real upstream source
during local ONNX verification:

  1. <|endofprompt|> (id 151646) is REQUIRED between the instruct prefix
     and the reference transcript — upstream CosyVoice3LM.inference()
     asserts on it; the community script omits it entirely.
  2. Sampling is ras_sampling (Repetition Aware Sampling, VALL-E 2 style:
     nucleus top-p/top-k with a repetition-window fallback to pure random)
     — exact port of cosyvoice/utils/common.py. The community script's
     plain top-k produces stutter/repetition artifacts.
  3. The stop condition spans the full 200-id special-token range
     (stop_token_ids = [speech_token_size + i for i in range(200)]),
     not a single EOS id — upstream Qwen2LM.inference_wrapper.

Speaker embedding is deliberately NOT an input at this stage:
CosyVoice3LM.inference() accepts it but never uses it (verified line by
line) — voice identity enters via prompt speech tokens here and via
CAMPPlus embedding in the Flow stage.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from lib.sessions import get_session

log = logging.getLogger(__name__)

HIDDEN_DIM = 896
SPEECH_TOKEN_SIZE = 6561
SOS = SPEECH_TOKEN_SIZE + 0        # 6561
TASK_ID = SPEECH_TOKEN_SIZE + 2    # 6563
STOP_TOKEN_IDS = frozenset(range(SPEECH_TOKEN_SIZE, SPEECH_TOKEN_SIZE + 200))
IGNORE_EOS_MASK_IDX = SPEECH_TOKEN_SIZE  # TransformerLM.sampling_ids masks this index

DEFAULT_INSTRUCT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def _nucleus_sampling(scores, top_p, top_k, rng) -> int:
    """cosyvoice/utils/common.py::nucleus_sampling, exact port."""
    probs = _softmax(scores)
    order = np.argsort(-probs, kind="stable")
    cum, kept_idx, kept_p = 0.0, [], []
    for i in order:
        if cum < top_p and len(kept_idx) < top_k:
            cum += probs[i]
            kept_idx.append(i)
            kept_p.append(probs[i])
        else:
            break
    kept_p = np.array(kept_p)
    return int(rng.choice(kept_idx, p=kept_p / kept_p.sum()))


def ras_sampling(scores, decoded_tokens, top_p=0.8, top_k=25,
                 win_size=10, tau_r=0.1, rng=None) -> int:
    """cosyvoice/utils/common.py::ras_sampling, exact port."""
    rng = rng or np.random.default_rng()
    top_ids = _nucleus_sampling(scores, top_p, top_k, rng)
    rep_num = sum(1 for t in decoded_tokens[-win_size:] if t == top_ids)
    if rep_num >= win_size * tau_r:
        scores = scores.copy()
        scores[top_ids] = -np.inf
        top_ids = int(rng.choice(len(scores), p=_softmax(scores)))
    return top_ids


def generate_speech_tokens(
    tokenizer,
    model_dir: str,
    ref_transcript: str,
    tts_text: str,
    prompt_speech_tokens: list[int],
    instruct_prefix: str = DEFAULT_INSTRUCT_PREFIX,
    sampling_k: int = 25,
    min_token_text_ratio: int = 2,
    max_token_text_ratio: int = 20,
    seed: int | None = None,
) -> list[int]:
    """One transcript → speech tokens. Prompt layout (upstream zero-shot):
    [SOS, embed(instruct_prefix + ref_transcript + tts_text), TASK_ID,
     embed(prompt_speech_tokens)] → autoregressive decode."""
    rng = np.random.default_rng(seed)

    text_ids = tokenizer.encode(instruct_prefix + ref_transcript + tts_text,
                                add_special_tokens=False)
    if 151646 not in text_ids:
        raise ValueError("<|endofprompt|> missing from prompt — instruct_prefix "
                         "must contain it (upstream llm.py asserts on id 151646)")
    tts_only_len = len(tokenizer.encode(tts_text, add_special_tokens=False))

    text_embed = get_session(f"{model_dir}/text_embedding_fp32.onnx")
    # INT4 backbone (community RTN block-32), adopted after measurement:
    # FP16 decoded at ~1066 ms/token on Lambda (359 s for one short
    # sentence — the whole pipeline's dominant cost) and needed a 7168 MB
    # tier because ORT upcasts FP16 weights to FP32 in RAM. INT4 passed
    # the ASR-roundtrip gate on linux/arm64 (per §3's rule) with ~7x less
    # weight memory. Voice-quality listening comparison vs FP16 still
    # owed to the user — flagged, not skipped silently.
    speech_embed = get_session(f"{model_dir}/llm_speech_embedding_int4.onnx")
    initial = get_session(f"{model_dir}/llm_backbone_initial_int4.onnx")
    decode = get_session(f"{model_dir}/llm_backbone_decode_int4.onnx")
    decoder = get_session(f"{model_dir}/llm_decoder_int4.onnx")

    def s_embed(ids):
        return speech_embed.run(None, {"token": np.array([ids], dtype=np.int64)})[0]

    t_emb = text_embed.run(None, {"input_ids": np.array([text_ids], dtype=np.int64)})[0]
    lm_input = np.concatenate(
        [s_embed([SOS]), t_emb, s_embed([TASK_ID]), s_embed(prompt_speech_tokens)],
        axis=1).astype(np.float32)
    seq_len = lm_input.shape[1]

    t0 = time.time()
    hidden, past_kv = initial.run(None, {
        "inputs_embeds": lm_input,
        "attention_mask": np.ones((1, seq_len), dtype=np.float32),
    })
    logits = decoder.run(None, {"hidden_state": hidden[:, -1:, :]})[0]
    log.info("LLM initial pass: %.2fs (context %d)", time.time() - t0, seq_len)

    min_len = tts_only_len * min_token_text_ratio
    max_len = tts_only_len * max_token_text_ratio

    out_tokens: list[int] = []
    t0 = time.time()
    for i in range(max_len):
        scores = logits.squeeze().astype(np.float64).copy()
        if i < min_len:
            scores[IGNORE_EOS_MASK_IDX] = -np.inf
        tok_id = ras_sampling(scores, out_tokens, top_k=sampling_k, rng=rng)
        if tok_id in STOP_TOKEN_IDS:
            break
        out_tokens.append(tok_id)

        hidden, past_kv = decode.run(None, {
            "inputs_embeds": s_embed([tok_id]).astype(np.float32),
            "attention_mask": np.ones((1, seq_len + len(out_tokens)), dtype=np.float32),
            "past_key_values": past_kv,
        })
        logits = decoder.run(None, {"hidden_state": hidden})[0]

    dt = time.time() - t0
    log.info("LLM decode: %d tokens in %.2fs (%.0f ms/token)",
             len(out_tokens), dt, dt / max(len(out_tokens), 1) * 1000)
    if not out_tokens:
        raise RuntimeError("LLM generated zero speech tokens")
    return out_tokens
