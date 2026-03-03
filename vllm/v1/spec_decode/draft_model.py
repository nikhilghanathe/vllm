# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model
# from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
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
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _raise_if_vocab_size_mismatch(self):
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    @override
    def _get_model(self) -> nn.Module:
        # Draft models may be quantized or on different parallelism,
        # so we load them with a modified vllm config
        from vllm.compilation.backends import set_model_tag

        temp_vllm_config = create_vllm_config_for_draft_model(self.vllm_config)
        with set_model_tag("draft_model"):
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
        super().load_model(target_model)
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
        # Always run the parent's dummy_run for the given num_tokens
        # (handles the first-pass forward which uses num_tokens).
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
                