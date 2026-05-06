# Speculative Decoding Optimizations

This document describes the cross-device speculative decoding optimizations and profiling instrumentation added to vLLM's v1 engine.

## Table of Contents
- [Overview](#overview)
- [Communication Profiling](#communication-profiling)
- [Single-GPU Draft Mode](#single-gpu-draft-mode)
- [Replicated Draft Mode](#replicated-draft-mode)
- [Usage Examples](#usage-examples)
- [Metrics Reference](#metrics-reference)

---

## Overview

When running speculative decoding with the draft model and target model on **separate devices** (e.g., draft on GPU 0, target on GPUs 1-7), hidden states must be communicated between devices. For EAGLE3, this means transferring **3×hidden_size** per token per speculation step.

These optimizations enable:
1. **Profiling communication costs** to quantify hidden state transfer overhead
2. **Reducing communication** by running draft on a single GPU or replicating it per-rank

---

## Communication Profiling

### What It Does

Instruments TP `all_reduce` operations with CUDA event timing to measure:
- **Total communication time** across all phases
- **Per-phase breakdown**: `target_forward`, `draft`, `scoring`
- **Allreduce call counts** per phase

This allows you to:
- Identify if communication is a bottleneck (e.g., 20% of total time)
- Quantify hidden state transfer costs for cross-device spec decode
- Compare communication patterns between different spec decode methods

### Implementation

**New file:** `vllm/v1/spec_decode/comm_timer.py`
- Wraps `CudaCommunicator.all_reduce` with CUDA event pairs
- Minimal overhead: 2 event records per allreduce (~microseconds)
- Tracks accumulated time and counts per phase

**Modified files:**
- `vllm/distributed/device_communicators/cuda_communicator.py`: Hooks allreduce calls
- `vllm/v1/worker/gpu_model_runner.py`: Sets phase labels before each stage
- `vllm/v1/spec_decode/metrics.py`: Adds comm metrics to SpecDecodingStats

### How to Enable

Set environment variable before running vLLM:
```bash
export PHASE_TIMING=1  # Enable communication profiling
```

Or in your vLLM config:
```python
from vllm import VLLMConfig
from vllm.config.speculative import SpeculativeConfig

spec_config = SpeculativeConfig(
    method="eagle3",
    model="path/to/eagle3-head",
    num_speculative_tokens=7,
    enable_comm_timing=True,  # Enable comm profiling
)
```

### Prometheus Metrics

When enabled, the following metrics are exposed:

```prometheus
# Total communication time and counts
vllm:spec_decode_comm_time_seconds{...} 12.34
vllm:spec_decode_comm_count{...} 1500

# Per-phase breakdown
vllm:spec_decode_comm_target_forward_time_seconds{...} 8.5
vllm:spec_decode_comm_target_forward_count{...} 500
vllm:spec_decode_comm_draft_time_seconds{...} 2.1
vllm:spec_decode_comm_draft_count{...} 700
vllm:spec_decode_comm_scoring_time_seconds{...} 1.74
vllm:spec_decode_comm_scoring_count{...} 300
```

### Example Analysis

For EAGLE3 with `hidden_size=4096`, `BS=16`, `accepted=5.5`:
- Per-token hidden state transfer: `16 × 5.5 × 4096 × 3 × 2 bytes = 2.06 MB`
- Expected latency (PCIe Gen4): ~90 µs
- Observation: If `comm_draft_time` >> 90 µs, investigate communication overhead

---

## Single-GPU Draft Mode

### What It Does

Runs the draft model on **only one GPU** (rank 0) while the target model runs across multiple GPUs (TP>1). Draft tokens are broadcast to all ranks.

**Use case:** Small draft models (<2GB) where replicating across all GPUs wastes memory.

### Configuration

```python
spec_config = SpeculativeConfig(
    method="draft_model",
    model="path/to/draft-model",
    draft_tensor_parallel_size=1,      # Draft uses 1 GPU
    single_gpu_draft=True,              # Only rank 0 loads draft
)

vllm_config = VLLMConfig(
    model_config=...,
    parallel_config=ParallelConfig(tensor_parallel_size=8),  # Target uses 8 GPUs
    speculative_config=spec_config,
)
```

### How It Works

1. **Rank 0**: Loads and runs the full draft model
2. **Ranks 1-7**: Skip draft model loading, wait for broadcast
3. **After drafting**: Rank 0 broadcasts draft tokens to all ranks
4. **Verification**: All ranks run target model verification in parallel

**Memory saved:** `(TP_size - 1) × draft_model_size`

Example: Draft model = 1.7GB, TP=8 → saves 11.9 GB across 7 GPUs

### Caveats

- **Broadcast overhead**: Adds latency for small token counts (usually <100 µs)
- **Rank 0 load**: Rank 0 does more work (drafting + broadcasting)
- **Best for:** Draft models <3GB where memory savings outweigh broadcast cost

---

## Replicated Draft Mode

### What It Does

Each rank **independently loads and runs** the full draft model (TP=1) while the target model runs with TP>1. No inter-GPU communication for drafting.

**Use case:** Draft models that are compute-bound rather than memory-bound, or when broadcast overhead is significant.

### Configuration

```python
spec_config = SpeculativeConfig(
    method="draft_model",
    model="path/to/draft-model",
    draft_tensor_parallel_size=1,      # Draft uses 1 GPU per rank
    single_gpu_draft=False,             # Each rank loads draft (default)
)

vllm_config = VLLMConfig(
    model_config=...,
    parallel_config=ParallelConfig(tensor_parallel_size=8),  # Target uses 8 GPUs
    speculative_config=spec_config,
)
```

### How It Works

1. **All ranks**: Load the full draft model independently
2. **Drafting**: Each rank runs drafting locally (no communication)
3. **Verification**: All ranks participate in target model verification

**Trade-off:**
- ✅ **No broadcast overhead** - zero draft communication latency
- ❌ **Higher memory usage** - draft model loaded on every GPU
- ✅ **Better for:** Larger draft models (>3GB) where memory is available

### Implementation Details

Creates a **per-rank TP=1 process group** for the draft model:
- Draft model sees `world_size=1`, `rank=0`
- All-reduce operations in draft model are no-ops
- Target model still uses the full TP group

---

## Usage Examples

### Example 1: Profile Communication for EAGLE3

```bash
export PHASE_TIMING=1

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --speculative-model lmsys/SGLang-EAGLE3-Qwen3-235B-A22B-Instruct \
  --speculative-method eagle3 \
  --num-speculative-tokens 7
```

**Check metrics:**
```bash
curl localhost:8000/metrics | grep spec_decode_comm
```

### Example 2: Single-GPU Draft for Memory Efficiency

```python
from vllm import LLM
from vllm.config import SpeculativeConfig

llm = LLM(
    model="Qwen/Qwen3-235B-A22B-GPTQ-Int4",
    tensor_parallel_size=8,
    speculative_config=SpeculativeConfig(
        method="draft_model",
        model="Qwen/Qwen3-1.7B-GPTQ-Int8",
        draft_tensor_parallel_size=1,
        single_gpu_draft=True,  # Only rank 0 loads draft
        num_speculative_tokens=5,
    ),
)

outputs = llm.generate("Write a Python function to compute Fibonacci numbers.")
```

### Example 3: Replicated Draft for Zero Comm Overhead

```python
llm = LLM(
    model="Qwen/Qwen3-235B-A22B-AWQ",
    tensor_parallel_size=8,
    speculative_config=SpeculativeConfig(
        method="draft_model",
        model="Qwen/Qwen3-3B-Instruct",
        draft_tensor_parallel_size=1,
        single_gpu_draft=False,  # All ranks load draft
        num_speculative_tokens=5,
    ),
)
```

### Example 4: Compare Communication Patterns

```bash
# Run with communication profiling enabled
export PHASE_TIMING=1

# Test 1: Draft model method
python benchmark.py --method draft_model --draft-model Qwen3-1.7B

# Test 2: EAGLE3 method
python benchmark.py --method eagle3 --draft-model EAGLE3-head

# Compare comm_time_seconds and comm_count from both runs
```

---

## Metrics Reference

### Compute Metrics (existing)

```prometheus
vllm:spec_decode_target_forward_time_seconds
vllm:spec_decode_draft_time_seconds
vllm:spec_decode_scoring_time_seconds
```

### Communication Metrics (new)

#### Total Metrics
```prometheus
vllm:spec_decode_comm_time_seconds
  - Total TP allreduce communication time (seconds)
  
vllm:spec_decode_comm_count
  - Total number of TP allreduce calls
```

#### Per-Phase Metrics
```prometheus
vllm:spec_decode_comm_target_forward_time_seconds
vllm:spec_decode_comm_target_forward_count
  - Communication during target model forward pass

vllm:spec_decode_comm_draft_time_seconds
vllm:spec_decode_comm_draft_count
  - Communication during draft model execution

vllm:spec_decode_comm_scoring_time_seconds
vllm:spec_decode_comm_scoring_count
  - Communication during verification scoring
```

### Analysis Tips

1. **Communication overhead ratio:**
   ```
   comm_ratio = comm_time_seconds / (target_forward + draft + scoring)
   ```
   - <5%: Communication is not a bottleneck
   - 5-15%: Moderate overhead, consider optimizations
   - >15%: Significant bottleneck, prioritize reduction

2. **Per-token comm cost (EAGLE3):**
   ```
   bytes_per_token = hidden_size × 3 × 2  # (3 layers, bf16)
   expected_latency = bytes_per_token × num_accepted / bandwidth
   ```

3. **Allreduce efficiency:**
   ```
   avg_time_per_call = comm_time / comm_count
   ```
   - Compare to expected PCIe/NVLink latency
   - High values may indicate contention or inefficiency

---

## Technical Details

### CUDA Event Timing

Communication timing uses CUDA events instead of CPU timers to:
- Avoid host-device synchronization overhead
- Accurately measure GPU-side communication time
- Overlap timing with computation (no blocking)

**Overhead:** ~2 event records per allreduce (~1-2 µs), negligible vs allreduce time (100-1000+ µs).

### Process Group Management

For replicated draft (draft_tp=1, target_tp>1):
- Creates `_draft_tp_group` via `init_model_parallel_group([rank])`
- Patches the global TP group temporarily when loading draft model
- Restores original TP group for target model operations

This allows the draft model to operate as if `world_size=1` while target model uses the full TP group.

### Single-GPU Draft Broadcast

Uses `torch.distributed.broadcast` to send draft tokens from rank 0:
```python
if self._single_gpu_draft:
    if self._tp_rank == 0:
        output = self.draft_model(input_ids)
    else:
        output = empty_tensor_on_device(...)
    torch.distributed.broadcast(output, src=0, group=get_world_group())
```

Broadcast latency: ~10-50 µs for typical token counts (10-100 tokens).

---

## Benchmarking

### Benchmark Script

The `sd_bench_vllm_offline.py` script in your benchmark suite uses **CUDA event instrumentation** to measure phase-level GPU time breakdowns:

```python
# Instruments three key phases with CUDA events + NVTX
- Draft proposer (propose() method)
- Target forward (execute_model() method)  
- Scoring/verification (rejection_sample() method)
```

**What it measures:**
- Per-phase GPU time (draft, target_forward, scoring)
- SpecDec counters: drafts, draft_tokens, accepted_tokens
- Acceptance rates overall and per-position
- Request-level latency histograms (TTFT, TPOT, E2E)
- Throughput: output tokens/s, total tokens/s

**Usage:**
```bash
# Compare draft_model vs eagle3 with communication profiling
export PHASE_TIMING=1

python sd_bench_vllm_offline.py \
  --model Qwen/Qwen3-235B-A22B-Instruct \
  --method draft_model \
  --draft-model Qwen/Qwen3-1.7B \
  --Ls 5 \
  --max-num-seqs 16 \
  --dataset-name humaneval

python sd_bench_vllm_offline.py \
  --model Qwen/Qwen3-235B-A22B-Instruct \
  --method eagle3 \
  --eagle-dir lmsys/SGLang-EAGLE3-Qwen3-235B \
  --Ls 7 \
  --max-num-seqs 16 \
  --dataset-name humaneval
```

**Outputs:**
- CSV results with throughput, acceptance rate, latency breakdowns
- Per-phase timing including communication overhead (when PHASE_TIMING=1)
- Prometheus metrics snapshot

This benchmark data can be used to:
1. **Quantify communication costs** - compare `comm_time` across methods
2. **Identify bottlenecks** - see which phase dominates (draft/target/scoring/comm)
3. **Evaluate optimizations** - measure before/after for fusion, single-GPU draft, etc.

---

## Future Work

The profiling infrastructure implemented here enables the following optimizations:

1. **Adaptive draft placement:**
   - Dynamically choose single-GPU vs replicated based on memory/latency
   - Profile-guided optimization

2. **Comm-overlapping scheduling:**
   - Overlap hidden state transfer with target computation
   - Requires async communication primitives

3. **Cross-device optimization:**
   - Further reduce communication overhead based on profiling data
   - Explore architectural changes to minimize hidden state transfers



