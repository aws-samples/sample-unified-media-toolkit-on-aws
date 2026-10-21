"""
Reference-audio conditioning extraction (voice registration / Phase A).

One reference WAV → the three conditioning artifacts every synthesis
against that voice reuses:

  speech_tokens  list[int]        speech_tokenizer_v3 (25 Hz discrete tokens)
  spk_embedding  [1, 192] f32     CAMPPlus x-vector
  prompt_mel     [1, T, 80] f32   matcha mel for Flow conditioning

plus the alignment step upstream frontend_zero_shot applies at 24 kHz:
hard-truncate mel/tokens to a common length (token_len = min(mel_frames//2,
num_tokens)) — verified against cosyvoice/cli/frontend.py; NOT a resize.

Text tokenization for the LLM also lives here (Qwen2 tokenizer + the
CosyVoice3 special tokens, including <|endofprompt|> at id 151646, which
the model's own inference asserts on — verified against upstream llm.py).
"""

from __future__ import annotations

import json
import logging

import numpy as np

from lib import dsp
from lib.audio_io import load_wav
from lib.sessions import get_session

log = logging.getLogger(__name__)

ENDOFPROMPT_ID = 151646  # upstream llm.py: assert 151646 in text

# Upstream cosyvoice/tokenizer/tokenizer.py::CosyVoice3Tokenizer special
# tokens. The community-shipped tokenizer_config.json omits everything past
# <|im_end|> (found during local verification) — these must be added so
# <|endofprompt|> lands at 151646. Only the head of upstream's full list is
# needed for id-151646 correctness; the rest (paralinguistic + phoneme tags)
# are included so instruct-style markup tokenizes as single tokens instead
# of being spelled out.
SPECIAL_TOKENS = {
    "eos_token": "<|endoftext|>",  # nosec B105
    "pad_token": "<|endoftext|>",  # nosec B105
    "additional_special_tokens": [
        "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
        "[breath]", "<strong>", "</strong>", "[noise]",
        "[laughter]", "[cough]", "[clucking]", "[accent]",
        "[quick_breath]", "<laughter>", "</laughter>",
        "[hissing]", "[sigh]", "[vocalized-noise]",
        "[lipsmack]", "[mn]", "<|endofsystem|>",
    ],
}

_tokenizer = None


def get_tokenizer(tokenizer_dir: str):
    """Qwen2 tokenizer with CosyVoice3 special tokens. Cached."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        tok.add_special_tokens(SPECIAL_TOKENS)
        eop = tok.convert_tokens_to_ids("<|endofprompt|>")
        if eop != ENDOFPROMPT_ID:
            raise RuntimeError(
                f"<|endofprompt|> resolved to {eop}, expected {ENDOFPROMPT_ID} — "
                "tokenizer files don't match the CosyVoice3 vocabulary")
        _tokenizer = tok
    return _tokenizer


def extract_conditioning(ref_wav_path: str, model_dir: str) -> dict:
    """Full Phase-A extraction from one reference WAV. Returns the three
    artifacts, already length-aligned (mel_frames == 2 * num_tokens)."""
    audio16 = load_wav(ref_wav_path, 16000)
    audio24 = load_wav(ref_wav_path, 24000)
    if len(audio16) / 16000 > 30:
        raise ValueError("reference audio longer than 30s not supported")

    speech_tokens = extract_speech_tokens(audio16, model_dir)
    spk_embedding = extract_spk_embedding(audio16, model_dir)
    prompt_mel = dsp.matcha_mel(audio24).T[np.newaxis, :, :]  # [1, T, 80]

    # upstream frontend_zero_shot 24 kHz branch: force mel % tokens = 2
    token_len = min(prompt_mel.shape[1] // 2, len(speech_tokens))
    prompt_mel = prompt_mel[:, :2 * token_len, :]
    speech_tokens = speech_tokens[:token_len]

    return {
        "speech_tokens": speech_tokens,
        "spk_embedding": spk_embedding,
        "prompt_mel": prompt_mel,
    }


def extract_speech_tokens(audio16: np.ndarray, model_dir: str) -> list[int]:
    """16 kHz audio → discrete speech tokens (speech_tokenizer_v3)."""
    feat = dsp.whisper_log_mel(audio16)[np.newaxis, :, :]
    sess = get_session(f"{model_dir}/speech_tokenizer_v3.onnx")
    tokens = sess.run(None, {
        sess.get_inputs()[0].name: feat,
        sess.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
    })[0].flatten().tolist()
    return tokens


def extract_spk_embedding(audio16: np.ndarray, model_dir: str) -> np.ndarray:
    """16 kHz audio → [1, 192] speaker embedding (CAMPPlus)."""
    feat = dsp.kaldi_fbank(audio16)
    feat = feat - feat.mean(axis=0, keepdims=True)  # upstream frontend.py
    sess = get_session(f"{model_dir}/campplus.onnx")
    return sess.run(None, {sess.get_inputs()[0].name: feat[np.newaxis, :, :]})[0]


# ── Conditioning blob (de)serialization — the S3 object format ─────────────
# One .npz per voice: compact, self-describing, loads in one call.

def save_conditioning(cond: dict, path: str) -> None:
    np.savez(path,
             speech_tokens=np.array(cond["speech_tokens"], dtype=np.int32),
             spk_embedding=cond["spk_embedding"].astype(np.float32),
             prompt_mel=cond["prompt_mel"].astype(np.float32))


def load_conditioning(path: str) -> dict:
    data = np.load(path)
    return {
        "speech_tokens": data["speech_tokens"].tolist(),
        "spk_embedding": data["spk_embedding"],
        "prompt_mel": data["prompt_mel"],
    }
