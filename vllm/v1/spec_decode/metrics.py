# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    # Number of verify (decode-with-spec) steps. Per-STEP timing denominator:
    # phase times accumulate once per step, while num_drafts counts per request
    # per step (= batch_size x steps), so per-step means must divide by this.
    num_verify_steps: int = 0

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
        )

    # Phase timing (seconds), accumulated per step.
    target_forward_time_s: float = 0.0
    draft_time_s: float = 0.0
    scoring_time_s: float = 0.0
    comm_time_s: float = 0.0
    comm_count: int = 0
    # Per-phase communication breakdown.
    comm_target_forward_time_s: float = 0.0
    comm_target_forward_count: int = 0
    comm_scoring_time_s: float = 0.0
    comm_scoring_count: int = 0
    comm_draft_time_s: float = 0.0
    comm_draft_count: int = 0
    # TLI (Token Level Intersection) per-op overhead (ms stored in _s fields).
    tli_draft_mask_time_s: float = 0.0
    tli_draft_remap_time_s: float = 0.0
    tli_target_remap_time_s: float = 0.0
    tli_target_mask_time_s: float = 0.0

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1

    def add_phase_times(self, phase_times: dict[str, float]):
        """Merge CUDA-event phase times into this stats object."""
        # target_forward is emitted only on verify steps (see gpu_model_runner),
        # so its presence marks one verify step — the per-step timing denominator.
        if "target_forward" in phase_times:
            self.num_verify_steps += 1
        self.target_forward_time_s += phase_times.get("target_forward", 0.0)
        self.draft_time_s += phase_times.get("draft", 0.0)
        self.scoring_time_s += phase_times.get("scoring", 0.0)
        self.comm_time_s += phase_times.get("comm", 0.0)
        self.comm_count += int(phase_times.get("comm_count", 0))
        # Per-phase comm breakdown.
        self.comm_target_forward_time_s += phase_times.get(
            "comm_target_forward", 0.0)
        self.comm_target_forward_count += int(phase_times.get(
            "comm_target_forward_count", 0))
        self.comm_scoring_time_s += phase_times.get(
            "comm_scoring", 0.0)
        self.comm_scoring_count += int(phase_times.get(
            "comm_scoring_count", 0))
        self.comm_draft_time_s += phase_times.get(
            "comm_draft", 0.0)
        self.comm_draft_count += int(phase_times.get(
            "comm_draft_count", 0))
        # TLI per-op times.
        self.tli_draft_mask_time_s += phase_times.get("tli_draft_mask", 0.0)
        self.tli_draft_remap_time_s += phase_times.get("tli_draft_remap", 0.0)
        self.tli_target_remap_time_s += phase_times.get(
            "tli_target_remap", 0.0)
        self.tli_target_mask_time_s += phase_times.get("tli_target_mask", 0.0)
        # logger.info(
        #     "Added phase times to stats: target=%.3fs draft=%.3fs scoring=%.3fs",
        #     self.target_forward_time_s,
        #     self.draft_time_s,
        #     self.scoring_time_s,
        # )

class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
        )
        self.reset()


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.spec_decoding_enabled = speculative_config is not None
        self.tli_enabled = (
            speculative_config is not None and
            getattr(speculative_config, 'cross_vocab_method', None) == "tli"
        )

        # Always register the target-forward timer so it is
        # available in baseline (no spec-decode) mode too.
        counter_target_fwd = self._counter_cls(
            name="vllm:spec_decode_target_forward_time_seconds",
            documentation="Cumulative target-model forward time (s).",
            labelnames=labelnames,
        )
        self.counter_target_forward_time = make_per_engine(
            counter_target_fwd, per_engine_labelvalues
        )

        # Number of verify steps — the correct per-step denominator for all
        # phase-timing means (phase times accumulate once per step).
        counter_verify_steps = self._counter_cls(
            name="vllm:spec_decode_num_verify_steps",
            documentation="Number of verify (decode-with-spec) steps.",
            labelnames=labelnames,
        )
        self.counter_num_verify_steps = make_per_engine(
            counter_verify_steps, per_engine_labelvalues
        )

        # Register comm counters unconditionally so they work in
        # baseline (no spec-decode) mode too.
        counter_comm = self._counter_cls(
            name="vllm:spec_decode_comm_time_seconds",
            documentation="Cumulative TP allreduce communication time (s).",
            labelnames=labelnames,
        )
        self.counter_comm_time = make_per_engine(
            counter_comm, per_engine_labelvalues
        )
        counter_comm_count = self._counter_cls(
            name="vllm:spec_decode_comm_count",
            documentation="Total number of TP allreduce calls.",
            labelnames=labelnames,
        )
        self.counter_comm_count = make_per_engine(
            counter_comm_count, per_engine_labelvalues
        )
        # Per-phase comm counters.
        for phase in ("target_forward", "scoring", "draft"):
            ctr_time = self._counter_cls(
                name=f"vllm:spec_decode_comm_{phase}_time_seconds",
                documentation=f"Cumulative TP allreduce time during {phase} (s).",
                labelnames=labelnames,
            )
            setattr(self, f"counter_comm_{phase}_time",
                    make_per_engine(ctr_time, per_engine_labelvalues))
            ctr_cnt = self._counter_cls(
                name=f"vllm:spec_decode_comm_{phase}_count",
                documentation=f"Number of TP allreduce calls during {phase}.",
                labelnames=labelnames,
            )
            setattr(self, f"counter_comm_{phase}_count",
                    make_per_engine(ctr_cnt, per_engine_labelvalues))

        if not self.spec_decoding_enabled:
            return

        counter_drafts = self._counter_cls(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = make_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        counter_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = make_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        counter_accepted_tokens = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = make_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        # Phase-timing counters (seconds, from CUDA events).
        # NOTE: counter_target_forward_time and comm counters are
        # registered unconditionally above.
        counter_draft = self._counter_cls(
            name="vllm:spec_decode_draft_time_seconds",
            documentation="Cumulative drafter time (s).",
            labelnames=labelnames,
        )
        self.counter_draft_time = make_per_engine(
            counter_draft, per_engine_labelvalues
        )
        counter_scoring = self._counter_cls(
            name="vllm:spec_decode_scoring_time_seconds",
            documentation="Cumulative scoring/verification time (s).",
            labelnames=labelnames,
        )
        self.counter_scoring_time = make_per_engine(
            counter_scoring, per_engine_labelvalues
        )

        # TLI per-op timing counters — only registered when TLI is active.
        if self.tli_enabled:
            for op in ("draft_mask", "draft_remap", "target_remap",
                       "target_mask"):
                ctr = self._counter_cls(
                    name=f"vllm:spec_decode_tli_{op}_time_seconds",
                    documentation=(
                        f"Cumulative TLI {op} operation time (ms "
                        f"accumulated as counter value)."),
                    labelnames=labelnames,
                )
                setattr(self, f"counter_tli_{op}_time",
                        make_per_engine(ctr, per_engine_labelvalues))

        assert speculative_config is not None
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        # Always record target-forward time and comm timing
        # (works in baseline too).
        self.counter_target_forward_time[engine_idx].inc(
            spec_decoding_stats.target_forward_time_s
        )
        self.counter_num_verify_steps[engine_idx].inc(
            spec_decoding_stats.num_verify_steps
        )
        self.counter_comm_time[engine_idx].inc(
            spec_decoding_stats.comm_time_s
        )
        self.counter_comm_count[engine_idx].inc(
            spec_decoding_stats.comm_count
        )
        for phase in ("target_forward", "scoring", "draft"):
            getattr(self, f"counter_comm_{phase}_time")[engine_idx].inc(
                getattr(spec_decoding_stats, f"comm_{phase}_time_s")
            )
            getattr(self, f"counter_comm_{phase}_count")[engine_idx].inc(
                getattr(spec_decoding_stats, f"comm_{phase}_count")
            )
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])

        # Phase timing counters (draft & scoring are spec-decode only;
        # target_forward is already incremented above).
        self.counter_draft_time[engine_idx].inc(
            spec_decoding_stats.draft_time_s
        )
        self.counter_scoring_time[engine_idx].inc(
            spec_decoding_stats.scoring_time_s
        )
        # NOTE: comm counters are incremented above (before the
        # spec_decoding_enabled guard) so they work in baseline too.

        # TLI per-op timing (only when TLI is active).
        if self.tli_enabled:
            for op in ("draft_mask", "draft_remap", "target_remap",
                       "target_mask"):
                getattr(self, f"counter_tli_{op}_time")[engine_idx].inc(
                    getattr(spec_decoding_stats, f"tli_{op}_time_s")
                )


def make_per_engine(
    counter: prometheus_client.Counter,
    per_engine_labelvalues: dict[int, list[object]],
):
    """Create a counter for each label value."""
    return {
        idx: counter.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
