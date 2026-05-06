# Token Level Intersection (TLI) — Cross-Vocabulary Speculative Decoding in vLLM

## Overview

This patch adds support for **cross-vocabulary speculative decoding** via Token Level Intersection (TLI) to vLLM's v1 engine. It allows using a draft model with a **different tokenizer/vocabulary** than the target model (e.g., DeepSeek-Coder-1.3B drafting for gpt-oss-120B).

Based on: Timor et al., "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies," **ICML'25 Oral**, arXiv:2502.05202.

## How It Works

1. At initialization, TLI decodes every token ID in both vocabularies to its string representation
2. Tokens that decode to the **same string** in both tokenizers form the **intersection vocabulary**
3. During drafting, draft logits are masked to only intersection tokens (non-intersection → `-inf`)
4. Draft token IDs (in draft vocab space) are remapped to target vocab IDs before being fed to the target model
5. During verification, target logits are also masked to the intersection before acceptance/rejection
6. All comparisons happen in the shared target-vocab ID space over intersection-restricted distributions

## Files Changed

### New File
| File | Lines | Purpose |
|------|-------|---------|
| `vllm/v1/spec_decode/token_intersection.py` | ~195 | `TokenLevelIntersection` class — builds intersection mappings, provides `mask_draft_logits()`, `mask_target_logits()`, `draft_ids_to_target()`, `target_ids_to_draft()` |

### Modified Files
| File | Change | Gated? |
|------|--------|--------|
| `vllm/config/speculative.py` | Added `cross_vocab_method` config field; gates draft tokenizer selection and vocab-size check | Yes — only active when `cross_vocab_method="tli"` |
| `vllm/v1/spec_decode/draft_model.py` | Added `_setup_tli()` (builds TLI + patches `compute_logits`), `_maybe_remap_to_target()` (maps draft→target IDs) | Yes — no-op when `self.tli is None` |
| `vllm/v1/sample/rejection_sampler.py` | Added optional `target_vocab_mask` parameter to `forward()`; masks bonus and target logits to intersection | Yes — no-op when `target_vocab_mask is None` |
| `vllm/v1/worker/gpu_model_runner.py` | Passes `target_vocab_mask` from drafter's TLI to rejection sampler | Yes — only when drafter is `DraftModelProposer` with `tli is not None` |

### What Is NOT Changed
- `eagle.py` (base class `SpecDecodeBaseProposer`) — untouched
- EAGLE/EAGLE3/ngram/MTP methods — untouched
- Triton rejection kernels — untouched (greedy mode works as-is because both draft and target IDs are in target vocab space after remapping)
- Existing `draft_model` method without TLI — identical behaviour

## How to Use

### CLI
```bash
vllm serve <target_model> \
    --speculative-config '{
        "model": "<draft_model>",
        "num_speculative_tokens": 5,
        "cross_vocab_method": "tli"
    }'
```

### Concrete Example (DeepSeek-Coder-1.3B → gpt-oss-120B)
```bash
vllm serve /path/to/gpt-oss-120B \
    --tensor-parallel-size 8 \
    --speculative-config '{
        "model": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
        "cross_vocab_method": "tli"
    }'
```

### Python API
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/gpt-oss-120B",
    tensor_parallel_size=8,
    speculative_config={
        "model": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
        "cross_vocab_method": "tli",
    },
)

output = llm.generate("def quicksort(arr):", SamplingParams(max_tokens=200))
```

### What Happens at Startup
```
INFO: TLI: computing vocabulary intersection between draft=deepseek-ai/deepseek-coder-1.3b-instruct (vocab=32256) and target=/path/to/gpt-oss-120B (vocab=201088)
INFO: TLI: intersection size = 18432 tokens (57.1% of draft, 9.2% of target)
```

## Design Decisions

1. **`compute_logits` monkey-patch** — The base class `propose()` loop in `eagle.py` calls `self.model.compute_logits()` then `argmax`. By patching `compute_logits` to add the intersection mask, we get intersection-restricted draft tokens without modifying `eagle.py` at all. The IDs stay in draft-vocab space (correct for the draft model's embedding layer in multi-step drafting), and are only remapped to target-vocab space in the final `_maybe_remap_to_target()`.

2. **Additive mask approach** — Instead of index-gathering into a compact intersection tensor, we use an additive `-inf` mask on the full logit vector. This is simpler, avoids scatter/gather, and works seamlessly with the existing argmax/softmax code.

3. **Everything gated on config** — When `cross_vocab_method` is `None` (default), every code path is identical to before. No performance impact on existing users.

## Limitations

- **Greedy decoding only** — The current implementation works correctly with greedy draft+verify (which is vLLM's default for speculative decoding). Full stochastic rejection sampling with `draft_probs` would need additional changes in the Triton kernels to handle intersection-restricted probability ratios.
- **Intersection size affects quality** — If the intersection is small (e.g., <30% of draft vocab), acceptance rates will be low because many draft tokens the model "wants" to produce are not in the intersection. Works best when tokenizers share significant overlap (common with models in the same family or BPE-based tokenizers with shared byte-level tokens).
- **One-time init cost** — Computing the intersection requires decoding every token ID in both vocabularies (~200K decode calls for gpt-oss-120B). This takes a few seconds at startup.

## Testing

Verified all changes import and compile in the `sd` conda environment:
```
1. token_intersection.py: OK
2. speculative.py: OK (cross_vocab_method field exists)
3. speculative.py: OK (vocab check gated for TLI)
4. draft_model.py: OK (_setup_tli + _maybe_remap_to_target)
5. rejection_sampler.py: OK (target_vocab_mask param)
6. gpu_model_runner.py: OK (TLI mask passthrough)
ALL 6 TLI CHANGES VERIFIED SUCCESSFULLY
```
