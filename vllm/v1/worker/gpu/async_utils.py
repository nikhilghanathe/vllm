# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import contextlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    RoutedExpertsTensors,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.utils import raise_if_nan_logits

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(frozen=True)
class StepTimingSample:
    forward_ms: float
    drafter_ms: float
    num_target_tokens: int
    num_reqs: int
    full_cudagraph: bool
    # Appended last, with a default, so the profiling path's positional
    # construction below is unaffected. Only the streaming path records it.
    scoring_ms: float = 0.0


def _timing_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


@dataclass
class StepTimingEvents:
    forward_start: torch.cuda.Event = field(default_factory=_timing_event)
    forward_end: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_start: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_end: torch.cuda.Event = field(default_factory=_timing_event)
    scoring_start: torch.cuda.Event = field(default_factory=_timing_event)
    scoring_end: torch.cuda.Event = field(default_factory=_timing_event)
    # elapsed_time() raises on an unrecorded pair, and the profiling path never
    # scores, so the pair must announce itself rather than be assumed.
    scoring_recorded: bool = False
    # The batch shape is bound to the step when it is armed, not read off the
    # collector when it closes. execute_model and sample_tokens are separate
    # worker RPCs, so another batch can be recorded in between; reading a mutable
    # field at close time labels this step's cost with that batch's shape.
    batch: tuple[bool, int, int] = (False, 0, 0)


class StepTimingCollector:
    """Times the steps run inside ``collect``; record calls no-op outside it.

    Every step gets its own events, so the steps queue back-to-back and the
    block resolves them all behind one sync on the way out.

    STREAMING MODE (VLLM_SPEC_STEP_TIMING=1) additionally times every *serving*
    step, not just the profiling dummy runs, and folds the results into buckets
    keyed by (full_cudagraph, num_target_tokens, num_reqs).

    Why buckets and not a lifetime total: verify cost is linear in total tokens
    B*(L+1), so a single cumulative mean answers nothing -- the cost *curve* is
    the result. Keying on num_target_tokens also keeps prefill steps (thousands
    of tokens) in their own buckets instead of blending into the verify mean,
    which is the prefill-exclusion problem solved structurally rather than with
    a flag.

    Streaming never feeds the profiling path: ``collect`` owns the events while
    it is open, so adaptive verification's cost curves are unperturbed.
    """

    def __init__(self):
        self._collecting = False
        self._step: StepTimingEvents | None = None
        self._batch = (False, 0, 0)
        self._dummy = False
        self._timed: list[tuple[StepTimingEvents, tuple[bool, int, int]]] = []

        self._streaming = bool(
            os.environ.get("VLLM_SPEC_STEP_TIMING")
        ) and torch.cuda.is_available()
        # key -> [steps, forward_ms, scoring_ms, drafter_ms, scored_steps]
        self._buckets: dict[tuple[bool, int, int], list[float]] = {}
        self._pending: list[tuple[StepTimingEvents, tuple[bool, int, int]]] = []
        # elapsed_time() needs a device sync, so fold in batches: per step it
        # would serialise the GPU and measure its own overhead.
        self._flush_every = int(os.environ.get("VLLM_SPEC_STEP_FLUSH_EVERY", "64"))
        if self._streaming:
            atexit.register(self._flush)

    # ---------------- streaming sink ----------------

    def _rank(self) -> int:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return int(os.environ.get("RANK", "0"))

    def _flush(self) -> None:
        """Resolve pending events into buckets and write the sink."""
        if not self._pending:
            return
        # Same stream, issue order: the last step's end implies all the rest.
        self._pending[-1][0].drafter_end.synchronize()
        for events, (full_cudagraph, num_target_tokens, num_reqs) in self._pending:
            try:
                fwd = events.forward_start.elapsed_time(events.forward_end)
                draft = events.drafter_start.elapsed_time(events.drafter_end)
                score = (
                    events.scoring_start.elapsed_time(events.scoring_end)
                    if events.scoring_recorded
                    else 0.0
                )
            except Exception:
                # A step can unwind between two records (an aborted capture does
                # exactly this). Drop it rather than lose every other bucket.
                continue
            key = (full_cudagraph, num_target_tokens, num_reqs)
            acc = self._buckets.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            acc[0] += 1
            acc[1] += fwd
            acc[2] += score
            acc[3] += draft
            # Not every step scores (a step with no draft tokens takes the plain
            # sampler), so scoring needs its own denominator. Dividing scoring_ms
            # by `steps` would silently under-report the mean.
            acc[4] += 1.0 if events.scoring_recorded else 0.0
        self._pending.clear()
        self._write()

    def _write(self) -> None:
        out = os.environ.get("VLLM_SPEC_STEP_TIMING_OUT")
        if not out:
            return
        base, ext = os.path.splitext(out)
        # Per rank: rank spread is itself a result. Collapsing to rank 0 would
        # report one rank's spin-wait as the cost.
        path = f"{base}.rank{self._rank()}{ext or '.json'}"
        rows = [
            {
                "full_cudagraph": k[0],
                "num_target_tokens": k[1],
                "num_reqs": k[2],
                "steps": int(v[0]),
                "scored_steps": int(v[4]),
                "forward_ms": v[1],
                "scoring_ms": v[2],
                "drafter_ms": v[3],
            }
            for k, v in sorted(self._buckets.items())
        ]
        try:
            # Written incrementally: workers are usually SIGKILLed, so atexit
            # alone is not a reliable path to a sink on disk.
            with open(path, "w") as f:
                json.dump({"_rank": self._rank(), "buckets": rows}, f)
        except Exception:
            pass

    @contextlib.contextmanager
    def collect(self) -> Iterator[list[StepTimingSample]]:
        """Time every step run in this block.

        The yielded list holds one sample per timed step once the block exits;
        it stays empty inside the block, where the timings are still on device.
        """
        samples: list[StepTimingSample] = []
        self._collecting = True
        try:
            yield samples
        finally:
            self._collecting = False
            timed, self._timed, self._step = self._timed, [], None
        if not timed:
            return
        # Same stream, issue order: once the last step is done, so are the rest.
        timed[-1][0].drafter_end.synchronize()
        samples.extend(
            StepTimingSample(
                events.forward_start.elapsed_time(events.forward_end),
                events.drafter_start.elapsed_time(events.drafter_end),
                num_target_tokens,
                num_reqs,
                full_cudagraph,
            )
            for events, (full_cudagraph, num_target_tokens, num_reqs) in timed
        )

    def record_batch(
        self,
        input_batch: "InputBatch",
        full_cudagraph: bool,
        dummy_run: bool = False,
    ) -> None:
        """Costs from different execution modes must not share a cost curve."""
        self._batch = (full_cudagraph, input_batch.num_tokens, input_batch.num_reqs)
        self._dummy = dummy_run

    def forward_start(self) -> None:
        if self._streaming and not self._collecting and self._dummy:
            # A dummy run is warmup, not workload, and it also closes its forward
            # pair itself (in _dummy_run, after execute_model returns) -- so a
            # streamed dummy step would be double-ended as well as meaningless.
            # Profiling still times dummy runs; that is all it times.
            self._step = None
            return
        if self._collecting or self._streaming:
            # A step that never reached drafter_end (no speculator, or an error
            # between the two) is abandoned here rather than left to pair its
            # start with the next step's end.
            self._step = StepTimingEvents(batch=self._batch)
            self._step.forward_start.record()

    def forward_end(self) -> None:
        if self._step is not None:
            self._step.forward_end.record()

    def forward_end_serving(self) -> None:
        """Close the forward pair on the serving path.

        Separate from ``forward_end`` because during profiling the dummy run
        closes the pair itself, *after* execute_model returns; recording twice
        would move the end event and silently reprice adaptive verification's
        cost curves.
        """
        if self._streaming and not self._collecting:
            self.forward_end()

    def scoring_start(self) -> None:
        if self._step is not None:
            self._step.scoring_start.record()

    def scoring_end(self) -> None:
        if self._step is not None:
            self._step.scoring_end.record()
            self._step.scoring_recorded = True

    def drafter_start(self) -> None:
        if self._step is not None:
            self._step.drafter_start.record()

    def drafter_end(self) -> None:
        """Ends the step: only steps that reach here have a draft cost."""
        if self._step is not None:
            self._step.drafter_end.record()
            if self._collecting:
                self._timed.append((self._step, self._step.batch))
            elif self._streaming:
                self._pending.append((self._step, self._step.batch))
                if len(self._pending) >= self._flush_every:
                    self._flush()
            self._step = None


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool = False,
        routed_experts: RoutedExpertsTensors | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.routed_experts = routed_experts
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)
        self._has_fault: torch.Tensor | None = None

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.sampling_mask_tensors: SamplingMaskTensors | None = None
            if sampler_output.sampling_mask_tensors is not None:
                self.sampling_mask_tensors = (
                    sampler_output.sampling_mask_tensors.to_cpu_nonblocking()
                )
            self.routed_experts_cpu: RoutedExpertsTensors | None = None
            if routed_experts is not None:
                self.routed_experts_cpu = routed_experts.to_cpu_nonblocking()
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.sampling_mask_tensors is not None:
            self.model_runner_output.sampling_masks = (
                self.sampling_mask_tensors.tolists(self.num_sampled_tokens_np)
            )

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(self.model_runner_output.num_nans_in_logits)

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        if self.routed_experts_cpu is not None:
            self.model_runner_output.routed_experts = self.routed_experts_cpu.tolists()

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: PoolerOutput,
        finished_mask: list[bool],
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            if isinstance(self.pooler_output, torch.Tensor) and all(finished_mask):
                self.pooler_output_cpu: PoolerOutput = self.pooler_output.to(
                    "cpu", non_blocking=True
                )
            else:
                outputs = (
                    self.pooler_output.unbind()
                    if isinstance(self.pooler_output, torch.Tensor)
                    else self.pooler_output
                )
                self.pooler_output_cpu = [
                    None
                    if output is None or not is_finished
                    else output.to("cpu", non_blocking=True)
                    for output, is_finished in zip(outputs, finished_mask, strict=True)
                ]
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        if isinstance(self.pooler_output_cpu, torch.Tensor):
            pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        else:
            pooler_output = self.pooler_output_cpu
        self.copy_event.synchronize()
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
