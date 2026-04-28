# SPDX-License-Identifier: Apache-2.0
"""Lightweight TP allreduce communication timer with per-phase tracking.

When enabled (PHASE_TIMING=True), wraps CudaCommunicator.all_reduce with
CUDA event pairs and accumulates elapsed time per phase (target_forward,
draft, scoring).  The accumulated times are read out by the phase-timing
code in gpu_model_runner.py alongside the existing target/draft/scoring
times so that comm-vs-compute breakdown is available.

The overhead is two CUDA event records per allreduce call — negligible
compared to the allreduce itself.
"""

from collections import defaultdict

import torch

_enabled = False

# Per-phase accumulated CUDA event pairs.
# Keys: "target_forward", "scoring", "draft", "" (unknown).
_phase_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
    defaultdict(list)
)
_current_phase: str = ""


def is_enabled() -> bool:
    return _enabled


def enable():
    """Enable comm timing — call once at startup if PHASE_TIMING is True."""
    global _enabled
    _enabled = True


def set_phase(phase: str):
    """Set the current phase label (e.g. "target_forward", "draft", "scoring").
    Called from gpu_model_runner before recording the phase-start CUDA event."""
    global _current_phase
    _current_phase = phase


def reset():
    """Clear all accumulated events.  Called at the start of each step."""
    global _phase_events
    _phase_events = defaultdict(list)


def record_pre():
    """Record a start event just before an allreduce.  Returns the event."""
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def record_post(start_ev: torch.cuda.Event):
    """Record an end event right after an allreduce and stash the pair."""
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    _phase_events[_current_phase].append((start_ev, end_ev))


def _sum_events(
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> float:
    """Sum elapsed times for a list of event pairs (ms)."""
    total = 0.0
    for s, e in events:
        total += s.elapsed_time(e)
    return total


def elapsed_ms_per_phase() -> dict[str, float]:
    """Return {phase: total_comm_ms} for each phase.
    Must be called AFTER a cuda synchronize / event.synchronize()."""
    return {phase: _sum_events(evts) for phase, evts in _phase_events.items()}


def count_per_phase() -> dict[str, int]:
    """Return {phase: num_allreduce_calls} for each phase."""
    return {phase: len(evts) for phase, evts in _phase_events.items()}


def elapsed_ms() -> float:
    """Sum all accumulated allreduce times across all phases (ms).
    Must be called AFTER a cuda synchronize / event.synchronize()."""
    return sum(elapsed_ms_per_phase().values())


def total_count() -> int:
    """Total number of allreduce calls across all phases."""
    return sum(count_per_phase().values())
