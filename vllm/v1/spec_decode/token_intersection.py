# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Token Level Intersection (TLI) for cross-vocabulary speculative decoding.

Implements the vocabulary intersection approach from:
  Timor et al., "Accelerating LLM Inference with Lossless Speculative
  Decoding Algorithms for Heterogeneous Vocabularies," ICML'25.

Given a draft model and target model with different tokenizers, TLI
computes the set of tokens that exist in both vocabularies (by decoded
string identity) and restricts both draft and target logits to this
intersection during speculative decoding.  This ensures the acceptance /
rejection comparison is performed over a common probability space.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from vllm.logger import init_logger
from vllm.v1.spec_decode import tli_timer

logger = init_logger(__name__)


class TokenLevelIntersection:
    """Pre-computes the vocabulary intersection between two tokenizers
    and provides efficient logit masking and token-ID mapping tensors."""

    def __init__(
        self,
        draft_tokenizer_name: str,
        target_tokenizer_name: str,
        draft_vocab_size: int,
        target_vocab_size: int,
        device: torch.device,
    ):
        logger.info(
            "TLI: computing vocabulary intersection between "
            "draft=%s (vocab=%d) and target=%s (vocab=%d)",
            draft_tokenizer_name,
            draft_vocab_size,
            target_tokenizer_name,
            target_vocab_size,
        )

        draft_tokenizer = AutoTokenizer.from_pretrained(
            draft_tokenizer_name, trust_remote_code=True
        )
        target_tokenizer = AutoTokenizer.from_pretrained(
            target_tokenizer_name, trust_remote_code=True
        )

        # Build string → id mappings for both vocabularies.
        draft_str_to_id: dict[str, int] = {}
        for token_id in range(draft_vocab_size):
            try:
                text = draft_tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )
                # Only keep first occurrence (lowest ID) for duplicates.
                if text not in draft_str_to_id:
                    draft_str_to_id[text] = token_id
            except Exception:
                continue

        target_str_to_id: dict[str, int] = {}
        for token_id in range(target_vocab_size):
            try:
                text = target_tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )
                if text not in target_str_to_id:
                    target_str_to_id[text] = token_id
            except Exception:
                continue

        # Find intersection: tokens whose decoded strings match exactly.
        shared_tokens = set(draft_str_to_id.keys()) & set(
            target_str_to_id.keys()
        )

        if not shared_tokens:
            raise ValueError(
                "TLI: vocabulary intersection is empty — draft and target "
                "tokenizers share no tokens.  Cross-vocab speculative "
                "decoding is not possible with these models."
            )

        intersection_size = len(shared_tokens)
        logger.info(
            "TLI: intersection size = %d tokens (%.1f%% of draft, "
            "%.1f%% of target)",
            intersection_size,
            100.0 * intersection_size / draft_vocab_size,
            100.0 * intersection_size / target_vocab_size,
        )

        # ------------------------------------------------------------------
        # Build mapping tensors
        # ------------------------------------------------------------------

        # draft_to_target[draft_id] = corresponding target_id
        #   (only valid for intersection tokens; others map to 0)
        # int32: vocab sizes (≤200k) fit, and token IDs in vLLM are int32 —
        # avoids a dtype cast on every remap call.
        draft_to_target = torch.zeros(draft_vocab_size, dtype=torch.int32)
        # target_to_draft[target_id] = corresponding draft_id
        target_to_draft = torch.zeros(target_vocab_size, dtype=torch.int32)

        draft_intersection_ids: list[int] = []
        target_intersection_ids: list[int] = []

        for text in shared_tokens:
            d_id = draft_str_to_id[text]
            t_id = target_str_to_id[text]
            draft_to_target[d_id] = t_id
            target_to_draft[t_id] = d_id
            draft_intersection_ids.append(d_id)
            target_intersection_ids.append(t_id)

        # Additive masks: 0 for intersection tokens, -inf for others.
        # Adding these to logits before softmax restricts the distribution.
        draft_mask = torch.full(
            (draft_vocab_size,), float("-inf"), dtype=torch.float32
        )
        draft_mask[torch.tensor(draft_intersection_ids, dtype=torch.long)] = (
            0.0
        )

        target_mask = torch.full(
            (target_vocab_size,), float("-inf"), dtype=torch.float32
        )
        target_mask[
            torch.tensor(target_intersection_ids, dtype=torch.long)
        ] = 0.0

        # Move all tensors to the target device.
        self.draft_to_target = draft_to_target.to(device)
        self.target_to_draft = target_to_draft.to(device)
        self.draft_mask = draft_mask.to(device)
        self.target_mask = target_mask.to(device)
        self.intersection_size = intersection_size

    # ------------------------------------------------------------------
    # Runtime helpers (all operate on GPU tensors)
    # ------------------------------------------------------------------

    def mask_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Restrict draft logits to the intersection vocabulary.

        Args:
            logits: [batch, draft_vocab_size]

        Returns:
            logits with non-intersection positions set to -inf.
        """
        if tli_timer.is_enabled():
            _ev = tli_timer.record_start("draft_mask")
            result = logits + self.draft_mask
            tli_timer.record_end("draft_mask", _ev)
            return result
        return logits + self.draft_mask

    def mask_target_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Restrict target logits to the intersection vocabulary.

        Args:
            logits: [num_tokens, target_vocab_size]

        Returns:
            logits with non-intersection positions set to -inf.
        """
        return logits + self.target_mask

    def draft_ids_to_target(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocabulary token IDs to target-vocabulary IDs.

        Args:
            token_ids: tensor of draft token IDs (any shape).

        Returns:
            tensor of target token IDs (same shape).
        """
        if tli_timer.is_enabled():
            _ev = tli_timer.record_start("draft_remap")
            result = self.draft_to_target[token_ids]
            tli_timer.record_end("draft_remap", _ev)
            return result
        return self.draft_to_target[token_ids]

    def target_ids_to_draft(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map target-vocabulary token IDs to draft-vocabulary IDs.

        Args:
            token_ids: tensor of target token IDs (any shape).

        Returns:
            tensor of draft token IDs (same shape).
        """
        if tli_timer.is_enabled():
            _ev = tli_timer.record_start("target_remap")
            result = self.target_to_draft[token_ids]
            tli_timer.record_end("target_remap", _ev)
            return result
        return self.target_to_draft[token_ids]
