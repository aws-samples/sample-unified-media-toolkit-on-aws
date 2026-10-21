# lib/wtpsplit_sat.py
"""
Minimal replacement for wtpsplit.SaT — ONNX inference path only.

Eliminates all dependencies on torch, transformers, and huggingface_hub.

Runtime dependencies
─────────────────────
  numpy              — array ops
  onnxruntime        — ONNX inference (already required by segmentation + embedding)
  tokenizers         — Rust fast tokenizer, loads tokenizer.json directly
  tqdm               — required by wtpsplit_extract at module level

The model directory (model_dir) must contain:
  model_optimized.onnx   — INT8 quantized SaT ONNX model
  config.json            — model config  (use_fp16 must be False)
  tokenizer.json         — XLM-RoBERTa fast tokenizer serialisation

All three files are produced by export_wtpsplit.py.

SaT.split() is compatible with the original wtpsplit.SaT.split() for
single-string input, which is the only call site in sentence_split.py.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Optional

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from lib.wtpsplit_extract import SaTORTWrapper, extract

log = logging.getLogger(__name__)

# Inference defaults — match wtpsplit 2.1.3 SaT defaults for sat-3l-sm
_DEFAULT_STRIDE = 64
_DEFAULT_MAX_BLOCK_SIZE = 512
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_THRESHOLD = 0.025   # sat-3l-sm default split threshold


class SaT:
    """
    Drop-in replacement for wtpsplit.SaT for ONNX inference.

    Only split() is implemented — that is the only method called from
    sentence_split.py.
    """

    def __init__(
        self,
        model_dir:             str,
        ort_providers:         Optional[list[str]] = None,
        threshold:             float = _DEFAULT_THRESHOLD,
        stride:                int = _DEFAULT_STRIDE,
        max_block_size:        int = _DEFAULT_MAX_BLOCK_SIZE,
        batch_size:            int = _DEFAULT_BATCH_SIZE,
        intra_op_num_threads:  int = 1,
    ):
        model_dir = os.path.abspath(model_dir)
        self.threshold = threshold
        self.stride = stride
        self.max_block_size = max_block_size
        self.batch_size = batch_size

        # ── Config ────────────────────────────────────────────────────────────
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.json not found in {model_dir}"
            )
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        # Patch use_fp16 defensively — export_wtpsplit.py already sets it False,
        # but guard here in case an older archive is used.
        cfg["use_fp16"] = False

        # num_labels defaults to 1 for all sat-3l-sm variants
        cfg.setdefault("num_labels", 1)

        self.config = SimpleNamespace(**cfg)

        # ── Tokenizer (pure Rust — no transformers) ───────────────────────────
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"tokenizer.json not found in {model_dir}. "
                f"Re-export the model with export_wtpsplit.py to include it."
            )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # Disable padding/truncation — extract() manages sequence lengths itself
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        log.info(f"Tokenizer loaded from {tokenizer_path}")

        # ── ONNX session ──────────────────────────────────────────────────────
        onnx_path = os.path.join(model_dir, "model_optimized.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"model_optimized.onnx not found in {model_dir}"
            )
        providers = ort_providers or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = intra_op_num_threads
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(onnx_path, opts, providers=providers)

        self.model = SaTORTWrapper(self.config, session)
        log.info(
            f"SaT ONNX loaded: {onnx_path}  "
            f"(threshold={threshold}, stride={stride})"
        )

    # =========================================================================
    # SPLIT
    # =========================================================================

    def split(
        self,
        text:           str,
        threshold:      Optional[float] = None,
        stride:         Optional[int] = None,
        max_block_size: Optional[int] = None,
        batch_size:     Optional[int] = None,
        weighting:      str = "uniform",
        verbose:        bool = False,
    ) -> list[str]:
        """
        Split `text` into sentences.

        Returns a list of sentence strings.  Equivalent to wtpsplit.SaT.split()
        for single-string input.

        Parameters
        ──────────
        threshold       : PROBABILITY threshold for sentence boundary
                          detection (sigmoid is applied to the model's
                          logits before comparison — upstream wtpsplit
                          semantics).
                          Lower → more splits.  Default 0.025 matches
                          the sat-3l-sm recommended value.
        stride          : sliding window step in subword tokens.
        max_block_size  : maximum tokens per window (capped at 512 for XLM-R).
        batch_size      : ONNX inference batch size.
        """
        if not text or not text.strip():
            return []

        threshold = threshold if threshold is not None else self.threshold
        stride = stride if stride is not None else self.stride
        max_block_size = max_block_size if max_block_size is not None else self.max_block_size
        batch_size = batch_size if batch_size is not None else self.batch_size

        # extract() expects a list of texts; we always pass a single string.
        all_logits, offset_mapping, _, _ = extract(
            batch_of_texts=[text],
            model=self.model,
            stride=stride,
            max_block_size=max_block_size,
            batch_size=batch_size,
            weighting=weighting,
            verbose=verbose,
            tokenizer=self.tokenizer,
        )

        # all_logits[0] is float16[n_tokens, 1] — squeeze to 1-D
        logits = all_logits[0][:, 0].astype(np.float32)   # [n_tokens]
        # list of (char_start, char_end)
        offsets = offset_mapping[0]

        return self._logits_to_sentences(text, logits, offsets, threshold)

    # =========================================================================
    # PRIVATE — logit → sentence boundary conversion
    # =========================================================================

    def _logits_to_sentences(
        self,
        text:      str,
        logits:    np.ndarray,   # float32[n_tokens]
        # [(char_start, char_end), ...]  len == n_tokens
        offsets:   list,
        threshold: float,
    ) -> list[str]:
        """
        Convert per-token boundary logits back to sentence strings.

        The sat model outputs a score at each subword token position.
        A score > threshold marks that position as a sentence boundary —
        the sentence ends AFTER the last character of that token.

        Algorithm
        ─────────
        1. Find token positions where logit > threshold.
        2. Map each boundary token to its character end position via offsets.
        3. Split the original text at those character positions.
        4. Return non-empty stripped strings.
        """
        if len(logits) == 0 or len(offsets) == 0:
            return [text] if text.strip() else []

        # The threshold is a PROBABILITY (upstream wtpsplit semantics:
        # 0.025 for sat models) — the model emits raw logits, so apply
        # the sigmoid before comparing. Comparing raw logits against
        # 0.025 (the bug this replaces) is equivalent to demanding
        # P>50.6%: invisible on punctuated English, where boundary
        # logits are large, but catastrophic on comma-only Mandarin
        # ASR text, where boundary probabilities live in the 5-40%
        # band — measured 2026-09-16 on AISHELL-4: 16 "sentences" for
        # 36 minutes of a 5-person meeting, 42% DER confusion.
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

        # Collect character-level split positions
        split_positions = []
        for i, (prob, (char_start, char_end)) in enumerate(
            zip(probs, offsets)
        ):
            if float(prob) > threshold:
                # Boundary is AFTER this token — split at char_end
                split_positions.append(char_end)

        if not split_positions:
            return [text.strip()] if text.strip() else []

        # Build sentence strings by slicing at split positions
        sentences = []
        prev = 0
        for pos in split_positions:
            chunk = text[prev:pos].strip()
            if chunk:
                sentences.append(chunk)
            prev = pos

        # Remainder after last split
        tail = text[prev:].strip()
        if tail:
            sentences.append(tail)

        return sentences if sentences else [text.strip()]
