# lib/wtpsplit_extract.py
"""
Vendored, stripped-down version of wtpsplit/extract.py.

Changes from the original
──────────────────────────
  1. Removed `from transformers import AutoTokenizer` — the tokenizer is
     passed in as a pre-built `tokenizers.Tokenizer` instance by SaTWrapper
     (see lib/wtpsplit_sat.py).  AutoTokenizer is never called.

  2. Removed `from wtpsplit.utils import Constants, hash_encode` — both are
     only used in the `not use_subwords` branch (character-hash models like
     BertChar).  Our model is xlm-token (use_subwords=True) so that branch
     is dead code.  The import is kept as a no-op stub so the function
     signature is unchanged.

  3. Removed the PyTorchWrapper class and its torch import — the ONNX path
     (SaTORTWrapper) is the only one we use.

  4. Tokenizer interface: the `tokenizer` argument is now expected to be a
     `tokenizers.Tokenizer` instance (from the `tokenizers` package).
     Its __call__ interface is compatible with the transformers tokenizer for
     the fields we use: input_ids, offset_mapping.
     Special token IDs are read from the tokenizer directly.

  No other logic is changed — the sliding window, averaging, and output
  format are identical to wtpsplit 2.1.3.

Runtime dependencies
─────────────────────
  numpy   — array ops
  tqdm    — progress bar (tqdm.auto suppresses output when not interactive)

  NOT needed: torch, transformers, huggingface_hub, wtpsplit
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


# =============================================================================
# ONNX WRAPPERS
# (Replaces the classes in the original extract.py)
# =============================================================================

class SaTORTWrapper:
    """
    Thin wrapper around an onnxruntime.InferenceSession for SaT (XLM) models.
    Inputs are float32 — our INT8 quantized model expects float32 attention masks.
    """

    def __init__(self, config, ort_session):
        self.config = config
        self.ort_session = ort_session

    def __getattr__(self, name):
        assert hasattr(self, "ort_session")
        return getattr(self.ort_session, name)

    def __call__(self, input_ids, attention_mask):
        logits = self.ort_session.run(
            ["logits"],
            {
                # fp32 not fp16
                "attention_mask": attention_mask.astype(np.float32),
                "input_ids":      input_ids.astype(np.int64),
            },
        )[0]
        return {"logits": logits}


# =============================================================================
# EXTRACT  (xlm-token / use_subwords=True path only)
# =============================================================================

def extract(
    batch_of_texts,
    model,
    stride,
    max_block_size,
    batch_size,
    lang_code=None,
    pad_last_batch=False,
    weighting: Literal["uniform", "hat"] = "uniform",
    verbose=False,
    tokenizer=None,
):
    """
    Compute per-character sentence-boundary logits for a batch of texts.

    Identical to wtpsplit 2.1.3 extract() for the xlm-token (use_subwords=True)
    path.  The character-hash path (BertChar / use_subwords=False) is retained
    as unreachable dead code so the function signature stays compatible.

    Parameters
    ──────────
    tokenizer : tokenizers.Tokenizer
        Pre-built XLM-RoBERTa tokenizer loaded from tokenizer.json.
        Must be passed in — it is never fetched from HuggingFace.

    Returns
    ───────
    (all_logits, offset_mapping, tokenizer, tokens)
    """
    if "xlm" in model.config.model_type:
        use_subwords = True

        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided for xlm-token models. "
                "Pass a tokenizers.Tokenizer loaded from tokenizer.json."
            )

        # tokenizers.Tokenizer batch encode
        # enable_padding=False, add_special_tokens=False to match original behaviour
        encodings = tokenizer.encode_batch(
            batch_of_texts, add_special_tokens=False
        )
        batch_of_texts = [enc.ids for enc in encodings]
        offset_mapping = [enc.offsets for enc in encodings]

        # XLM-RoBERTa special token IDs are fixed constants:
        #   CLS (bos) = 0,  PAD = 1,  SEP (eos) = 2
        cls_token_id = tokenizer.token_to_id("<s>") or 0
        sep_token_id = tokenizer.token_to_id("</s>") or 2
        pad_token_id = tokenizer.token_to_id("<pad>") or 1

    else:
        # Character-hash path (BertChar models) — not used with our model.
        # Kept so the function signature stays compatible.
        pad_token_id = 0
        use_subwords = False

    text_lengths = [len(text) for text in batch_of_texts]
    block_size = min(max_block_size, max(text_lengths))

    if use_subwords and block_size > 510:
        block_size -= block_size - 510   # reserve space for CLS and SEP

    downsampling_rate = getattr(model.config, "downsampling_rate", 1)
    block_size = math.ceil(block_size / downsampling_rate) * downsampling_rate

    num_chunks = sum(
        math.ceil(max(length - block_size, 0) / stride) + 1
        for length in text_lengths
    )

    if not use_subwords:
        input_hashes = np.zeros((num_chunks, block_size,
                                 getattr(model.config, "num_hash_functions", 8)),
                                dtype=np.int64)
        attention_mask = np.zeros((num_chunks, block_size), dtype=np.float32)
    else:
        input_ids = np.zeros((num_chunks, block_size + 2), dtype=np.int64)
        attention_mask = np.zeros(
            (num_chunks, block_size + 2), dtype=np.float32)

    locs = np.zeros((num_chunks, 3), dtype=np.int32)

    if not use_subwords:
        codec = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"
        ordinals = np.frombuffer(
            bytearray("".join(batch_of_texts), encoding=codec), dtype=np.int32
        )
        # hash_encode is only needed here — dead code on our model path
        raise RuntimeError(
            "Character-hash path reached unexpectedly — "
            "this model should always use use_subwords=True"
        )

    current_chunk = 0
    offset = 0

    for i in range(len(batch_of_texts)):
        for j in range(0, text_lengths[i], stride):
            start, end = j, j + block_size
            done = False

            if end >= text_lengths[i]:
                end = text_lengths[i]
                start = max(end - block_size, 0)
                done = True

            if use_subwords:
                chunk = [cls_token_id] + \
                    batch_of_texts[i][start:end] + [sep_token_id]
                input_ids[current_chunk, : len(chunk)] = chunk
                attention_mask[current_chunk, : len(chunk)] = 1

            locs[current_chunk, :] = [i, start, end]
            current_chunk += 1

            if done:
                break

        offset += text_lengths[i]

    assert current_chunk == num_chunks

    n_batches = math.ceil(len(attention_mask) / batch_size)

    all_logits = [
        np.zeros((length, model.config.num_labels), dtype=np.float16)
        for length in text_lengths
    ]
    all_counts = [np.zeros(length, dtype=np.float16)
                  for length in text_lengths]

    uses_lang_adapters = getattr(
        model.config, "language_adapter", "off") == "on"
    if uses_lang_adapters:
        raise ValueError(
            "Language adapters are not supported in ONNX models."
        )
    language_ids = None

    if weighting == "uniform":
        weights = np.ones(block_size, dtype=np.float16)
    elif weighting == "hat":
        x = np.linspace(
            -(1 - 1 / block_size), 1 - 1 / block_size, block_size, dtype=np.float16
        )
        weights = 1 - np.abs(x)

    for batch_idx in tqdm(range(n_batches), disable=not verbose):
        start = batch_idx * batch_size
        end = min(len(attention_mask), (batch_idx + 1) * batch_size)

        if use_subwords:
            batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_mask[start:end]

        if len(batch_attention_mask) < batch_size and pad_last_batch:
            n_missing = batch_size - len(batch_attention_mask)
            if use_subwords:
                batch_input_ids = np.pad(
                    batch_input_ids,
                    ((0, n_missing), (0, 0)),
                    constant_values=pad_token_id,
                )
            batch_attention_mask = np.pad(
                batch_attention_mask, ((0, n_missing), (0, 0))
            )

        kwargs = {}
        if use_subwords:
            kwargs["input_ids"] = batch_input_ids

        logits = model(attention_mask=batch_attention_mask, **kwargs)["logits"]

        if use_subwords:
            logits = logits[:, 1:-1, :]   # strip CLS and SEP

        for i in range(start, end):
            original_idx, start_char_idx, end_char_idx = locs[i]
            n = end_char_idx - start_char_idx
            all_logits[original_idx][start_char_idx:end_char_idx] += (
                weights[:n, np.newaxis] * logits[i - start, :n]
            )
            all_counts[original_idx][start_char_idx:end_char_idx] += weights[:n]

    all_logits = [
        (logits / counts[:, None]).astype(np.float16)
        for logits, counts in zip(all_logits, all_counts)
    ]

    return (
        all_logits,
        offset_mapping if use_subwords else None,
        tokenizer if use_subwords else None,
        encodings if use_subwords else None,
    )
