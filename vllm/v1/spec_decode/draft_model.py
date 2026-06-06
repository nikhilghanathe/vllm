# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig

from vllm.distributed.parallel_state import (
    _is_tp_patched,
    get_tp_group,
    get_world_group,
    init_model_parallel_group,
    patch_tensor_parallel_group,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.token_intersection import TokenLevelIntersection
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)


logger = init_logger(__name__)

try:
    import nvtx
except ImportError:
    nvtx = None


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        # Determine if single-GPU draft mode is requested BEFORE calling
        # super().__init__() so we can skip buffer allocation on non-draft ranks.
        self._single_gpu_draft = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.single_gpu_draft
        )
        self._tp_rank = get_tp_group().rank_in_group

        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_vocab_size_mismatch()
        self._validate_and_setup_draft_tp()
        self._setup_tli(device)

    def _raise_if_vocab_size_mismatch(self):
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _setup_tli(self, device: torch.device):
        """Initialize Token Level Intersection when cross_vocab_method='tli'.

        This is a no-op when TLI is not configured.  When enabled, it:
        1. Builds the intersection mapping between draft & target vocabs.
        2. Monkey-patches ``self.model.compute_logits`` so that draft
           logits are masked to the intersection vocabulary *before*
           the base-class ``propose()`` calls ``argmax``.
        """
        self.tli: TokenLevelIntersection | None = None

        if self.speculative_config.cross_vocab_method != "tli":
            return

        draft_vocab_size = (
            self.speculative_config.draft_model_config.get_vocab_size()
        )
        target_vocab_size = (
            self.speculative_config.target_model_config.get_vocab_size()
        )
        draft_tokenizer_name = self.speculative_config.model
        target_tokenizer_name = (
            self.speculative_config.target_model_config.tokenizer
        )

        self.tli = TokenLevelIntersection(
            draft_tokenizer_name=draft_tokenizer_name,
            target_tokenizer_name=target_tokenizer_name,
            draft_vocab_size=draft_vocab_size,
            target_vocab_size=target_vocab_size,
            device=device,
        )

        # NOTE: the compute_logits monkey-patch is deferred to
        # load_model() because self.model does not exist yet during
        # __init__.

    def _patch_compute_logits_for_tli(self):
        """Monkey-patch compute_logits to mask draft logits to the
        intersection vocabulary.  Must be called after load_model()
        so that self.model exists."""
        if self.tli is None:
            return

        # Cast the draft mask to match the model's compute dtype so the
        # in-place addition below stays in bfloat16/float16 rather than
        # promoting to float32 (which would allocate a new float32 tensor).
        try:
            model_dtype = next(iter(self.model.parameters())).dtype
            self.tli.draft_mask = self.tli.draft_mask.to(model_dtype)
        except StopIteration:
            pass  # no parameters (unusual); keep float32 mask

        original_compute_logits = self.model.compute_logits
        tli_ref = self.tli

        def _tli_compute_logits(hidden_states: torch.Tensor) -> torch.Tensor:
            logits = original_compute_logits(hidden_states)
            # In-place add: logits is a fresh tensor from compute_logits;
            # mask dtype already matches, so no type promotion or allocation.
            logits += tli_ref.draft_mask
            return logits

        self.model.compute_logits = _tli_compute_logits  # type: ignore[method-assign]

    def _patch_compute_logits_for_probe(self):
        """Wrap compute_logits to capture per-step draft distribution stats
        for the SpecProbe analysis instrument. No-op unless VLLM_SPEC_PROBE
        is set. Wraps whatever compute_logits exists (after the TLI patch),
        so probe + TLI compose."""
        from vllm.v1.spec_decode import spec_probe
        probe = spec_probe.get_probe()
        if not probe.enabled:
            return
        probe.set_rank(self._tp_rank)
        probe.arm()

        original_compute_logits = self.model.compute_logits

        def _probe_compute_logits(hidden_states: torch.Tensor) -> torch.Tensor:
            logits = original_compute_logits(hidden_states)
            probe.record_draft_step(logits)
            return logits

        self.model.compute_logits = _probe_compute_logits  # type: ignore[method-assign]

    def _validate_and_setup_draft_tp(self):
        """Validate draft/target TP sizes and create a self-only TP group
        for replicated draft mode (draft_tp=1, target_tp>1).

        Two modes are supported:
        - Sharded draft (draft_tp == target_tp): draft model is sharded
          across TP ranks just like the target. No special handling.
        - Replicated draft (draft_tp == 1, target_tp > 1): each TP rank
          loads the FULL draft model independently. A per-rank TP=1
          process group is created so the draft model sees world_size=1.
          This avoids all-reduce communication for the small draft model.
        """
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size

        if draft_tp == tgt_tp:
            # Sharded draft — same TP as target, no special handling.
            self._draft_tp_group = None
            return

        if draft_tp == 1 and tgt_tp > 1:
            # Replicated draft — each rank loads the full draft model.
            # With single_gpu_draft, only rank 0 loads and runs the model;
            # other ranks receive draft tokens via broadcast.
            self._draft_tp_group = self._create_self_only_tp_group()
            if self._single_gpu_draft:
                logger.info(
                    "Single-GPU draft mode: draft_tp=%d, target_tp=%d. "
                    "Only rank 0 loads the draft model. Other ranks "
                    "receive draft tokens via broadcast.",
                    draft_tp, tgt_tp)
            else:
                logger.info(
                    "Replicated draft mode: draft_tp=%d, target_tp=%d. "
                    "Each rank loads the full draft model independently "
                    "(no inter-GPU communication for drafting).",
                    draft_tp, tgt_tp)
            return

        raise ValueError(
            f"draft_tensor_parallel_size must be 1 or equal to "
            f"tensor_parallel_size ({tgt_tp}). Got {draft_tp}. "
            "Pass 'draft_tensor_parallel_size' in speculative_config.")

    def _create_self_only_tp_group(self):
        """Create a TP=1 process group where each rank is in its own group.

        For 8 TP ranks, this creates groups: [[0], [1], [2], ..., [7]].
        Each rank ends up with world_size=1, rank_in_group=0.
        All ranks must call this collectively (torch distributed requirement).
        """
        tp_group = get_tp_group()
        # Each rank gets its own singleton group
        self_only_ranks = [[r] for r in tp_group.ranks]
        return init_model_parallel_group(
            group_ranks=self_only_ranks,
            local_rank=get_world_group().local_rank,
            backend="nccl",
            group_name="draft_tp",
        )

    @contextmanager
    def _maybe_patch_tp(self):
        """Patch the global TP group for replicated draft mode.

        When draft_tp=1, this swaps the global TP group to the self-only
        group so the draft model sees world_size=1, rank=0. When draft_tp
        equals target_tp (sharded mode), this is a no-op.

        Re-entrant: if already patched (nested call), just yields.
        """
        if self._draft_tp_group is None:
            # Sharded mode — no patching needed.
            yield
        elif _is_tp_patched():
            # Already patched by an outer call — don't nest.
            yield
        else:
            with patch_tensor_parallel_group(self._draft_tp_group):
                yield

    @override
    def _get_model(self) -> nn.Module:
        # Draft models may be quantized or on different parallelism,
        # so we load them with a modified vllm config.
        # In replicated mode (draft_tp=1), we patch the TP group so the
        # model loader sees world_size=1 and loads all weights (no sharding).
        #
        # IMPORTANT: the model tag is used as the torch.compile cache
        # subdirectory name. Different draft TP sizes shard lm_head/
        # embed_tokens differently (e.g. [151936,2048] at tp=1 vs
        # [75968,2048] at tp=2), so they MUST use separate cache dirs.
        # Without the tp suffix, a cached kernel compiled for tp=1
        # bakes in assert_size_stride(..., (151936, 2048), ...) and
        # crashes when reloaded for tp=2 where the weight is (75968, 2048).
        from vllm.compilation.backends import set_model_tag

        draft_tp = self.speculative_config.draft_parallel_config.tensor_parallel_size
        model_tag = f"draft_model_tp{draft_tp}"

        temp_vllm_config = create_vllm_config_for_draft_model(self.vllm_config)
        with self._maybe_patch_tp(), set_model_tag(model_tag):
            model = get_model(
                vllm_config=temp_vllm_config,
                prefix="draft_model",
            )
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        # Draft models don't share embeddings with the target model
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        # Draft models don't share lm_head with the target model
        pass

    @property
    def _is_draft_rank(self) -> bool:
        """True if this rank should run the draft model.
        In single-GPU mode, only rank 0 runs the draft. Otherwise all ranks do."""
        return not self._single_gpu_draft or self._tp_rank == 0

    @override
    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Skip validation on non-draft ranks where the draft model
        was not loaded and attn_layer_names is empty.
        Also skip when TLI is active — the draft model has a completely
        different architecture and its own independent KV cache."""
        if not self._is_draft_rank:
            return
        if self.tli is not None:
            return
        super().validate_same_kv_cache_group(kv_cache_config)

    @override
    def propose(self, *args, **kwargs) -> torch.Tensor:
        """Override propose to patch TP group for replicated draft.
        In single-GPU mode, only rank 0 runs the draft model and
        broadcasts the result to other ranks.

        When TLI is enabled, the returned token IDs are remapped from
        draft-vocabulary space to target-vocabulary space so that the
        target model can consume them directly."""
        # TLI: remap incoming target-vocab token IDs → draft-vocab IDs
        # before they enter the draft model's embedding layer.
        # Mapping tables are int32, matching the token ID dtype in vLLM,
        # so no .to() cast is needed.
        if self.tli is not None:
            if 'next_token_ids' in kwargs:
                kwargs['next_token_ids'] = self.tli.target_ids_to_draft(
                    kwargs['next_token_ids'])
            elif len(args) > 3:
                args = list(args)
                args[3] = self.tli.target_ids_to_draft(args[3])
                args = tuple(args)
            if 'target_token_ids' in kwargs:
                kwargs['target_token_ids'] = self.tli.target_ids_to_draft(
                    kwargs['target_token_ids'])
            elif len(args) > 0:
                args = list(args)
                args[0] = self.tli.target_ids_to_draft(args[0])
                args = tuple(args)

        from vllm.v1.spec_decode import spec_probe
        probe = spec_probe.get_probe()

        if not self._single_gpu_draft:
            probe.begin_draft()
            with self._maybe_patch_tp():
                draft_token_ids = super().propose(*args, **kwargs)
            remapped = self._maybe_remap_to_target(draft_token_ids)
            probe.end_draft(remapped)
            return remapped

        # Single-GPU draft: only rank 0 runs propose.
        tp_group = get_tp_group()

        # Extract batch_size from common_attn_metadata (always passed as kwarg).
        common_attn_metadata = kwargs['common_attn_metadata']
        batch_size = common_attn_metadata.batch_size()

        if self._is_draft_rank:
            probe.begin_draft()
            with self._maybe_patch_tp():
                draft_token_ids = super().propose(*args, **kwargs)
        else:
            # Allocate buffer to receive broadcast.
            draft_token_ids = torch.empty(
                (batch_size, self.num_speculative_tokens),
                dtype=torch.int32, device=self.device)

        # Broadcast draft tokens from rank 0 to all TP ranks.
        # This is tiny: batch_size * num_spec_tokens * 4 bytes.
        from vllm.v1.spec_decode import comm_timer
        timing = comm_timer.is_enabled()
        if timing:
            start_ev = comm_timer.record_pre()

        tp_group.broadcast(draft_token_ids, src=0)

        if timing:
            comm_timer.record_post(start_ev)

        remapped = self._maybe_remap_to_target(draft_token_ids)
        # Only the draft rank captured per-step stats; finalize there.
        if self._is_draft_rank:
            probe.end_draft(remapped)
        return remapped

    def _maybe_remap_to_target(
        self, draft_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Remap draft token IDs to target vocab space when TLI is active.

        This is a no-op when TLI is not configured."""
        if self.tli is not None:
            # Mapping table is int32, same dtype as draft_token_ids — no cast.
            draft_token_ids = self.tli.draft_ids_to_target(draft_token_ids)
        return draft_token_ids

    @override
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for draft model.

        The shared compilation_config.cudagraph_capture_sizes has been
        adjusted by adjust_cudagraph_sizes_for_spec_decode to multiples
        of (num_speculative_tokens + 1) for the TARGET model. But the
        drafter dispatches with batch_size (1 to max_num_seqs), so it
        needs the original power-of-2 capture sizes.

        We generate drafter-appropriate sizes and point the drafter's
        cudagraph_dispatcher at a modified compilation_config copy.
        """
        if not self._is_draft_rank:
            return
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            # NOTE: We use PIECEWISE, not FULL. FULL CUDA graphs bake in
            # the entire forward including attention. But the drafter's
            # attention metadata is rebuilt each iteration via
            # build_for_drafting() (new tensors each time). A FULL graph
            # would replay the capture-time attention (with None metadata)
            # and produce garbage. PIECEWISE compiles sub-graphs between
            # attention splitting points, running attention eagerly so it
            # always reads the correct per-iteration metadata.
            # To enable FULL, the drafter would need persistent attention
            # metadata buffers updated in-place (like the target model).
            draft_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            draft_cudagraph_mode = CUDAGraphMode.NONE

        # Generate capture sizes appropriate for the drafter:
        # powers of 2 from 1 up to max_num_seqs.
        max_seqs = self.vllm_config.scheduler_config.max_num_seqs
        drafter_capture_sizes = sorted(
            {2**i for i in range(max_seqs.bit_length() + 1)
             if 2**i <= max_seqs} | {max_seqs}
        )
        self._drafter_capture_sizes = drafter_capture_sizes

        # Give the drafter's dispatcher its own compilation_config
        # with the drafter-appropriate capture sizes, so that
        # _compute_bs_to_padded_graph_size and dispatch() use them.
        import copy
        drafter_cc = copy.copy(self.compilation_config)
        drafter_cc.cudagraph_capture_sizes = drafter_capture_sizes
        drafter_cc.max_cudagraph_capture_size = max(drafter_capture_sizes)
        self.cudagraph_dispatcher.compilation_config = drafter_cc

        self.cudagraph_dispatcher.initialize_cudagraph_keys(draft_cudagraph_mode)
        logger.info(
            "DraftModelProposer cudagraph keys initialized: mode=%s, "
            "capture_sizes=%s", draft_cudagraph_mode, drafter_capture_sizes
        )

    @override
    def load_model(self, target_model: nn.Module) -> None:
        if not self._is_draft_rank:
            logger.info(
                "Single-GPU draft: rank %d skipping draft model load.",
                self._tp_rank)
            return
        with self._maybe_patch_tp():
            super().load_model(target_model)
        self._patch_compute_logits_for_tli()
        # Probe wrapper goes LAST so it observes the final (TLI-masked)
        # logits the proposer actually argmaxes over. No-op when disabled.
        self._patch_compute_logits_for_probe()
        # NOTE: We do NOT wrap with CUDAGraphWrapper here.
        # CUDAGraphWrapper is for FULL CUDA graphs, but the drafter
        # uses PIECEWISE (see initialize_cudagraph_keys comment).
        # PIECEWISE sub-graphs are handled by torch.compile's
        # piecewise backend automatically.

    @override
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Override dummy_run to capture PIECEWISE CUDA graphs for batch
        sizes the draft model actually uses at runtime.

        The parent's dummy_run is called from the target model's capture
        loop with target-sized num_tokens (e.g. 11, 22 — multiples of
        num_speculative_tokens+1). But at runtime, the draft iteration
        loop dispatches with batch_size (number of sequences, e.g. 1-256).
        We need PIECEWISE sub-graphs captured for those sizes too.

        During graph capturing (is_graph_capturing=True), we additionally
        run dummy forwards for all drafter capture sizes to warm up the
        PIECEWISE sub-graph captures.
        """
        if not self._is_draft_rank:
            return
        # Always run the parent's dummy_run for the given num_tokens
        # (handles the first-pass forward which uses num_tokens).
        # In replicated mode, patch TP so the draft model sees world_size=1.
        with self._maybe_patch_tp():
            super().dummy_run(
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )
            # During graph capture, also capture for batch-size-scale inputs
            # that the iteration loop in propose() will use.
            if is_graph_capturing and use_cudagraphs:
                drafter_sizes = getattr(self, '_drafter_capture_sizes', [])
                for bs in drafter_sizes:
                    if bs == num_tokens:
                        continue
                    # Run a dummy forward at this batch size to warm up
                    # PIECEWISE sub-graph captures.
                    super().dummy_run(
                        bs,
                        use_cudagraphs=True,
                        is_graph_capturing=is_graph_capturing,
                        slot_mappings=slot_mappings,
                    )
                logger.info(
                    "DraftModelProposer: captured CUDA graph for "
                    "batch_size=%d", bs
                )
                