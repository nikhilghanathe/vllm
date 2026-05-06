# SPDX-License-Identifier: Apache-2.0
"""Lightweight per-op CUDA timer for Token Level Intersection (TLI) overheads.

When enabled (only when TLI cross-vocab is active), wraps the four TLI
runtime operations with CUDA event pairs and accumulates elapsed time per
operation.  Times are read out by gpu_model_runner.py alongside the existing
phase times and exposed via Prometheus counters.

Operations tracked:
  "draft_mask"   – mask_draft_logits() (restricts draft logits to intersection)
  "draft_remap"  – draft_ids_to_target() (remaps draft IDs to target vocab)
  "target_remap" – target_ids_to_draft() (remaps target IDs to draft vocab)
  "target_mask"  – target vocab masking in rejection sampler (bonus + target)

The overhead is two CUDA event records per operation — negligible compared
to the speculative decode forward passes.
"""

from collections import defaultdict

import torch

_enabled = False

# Per-op accumulated CUDA event pairs.
_op_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
    defaultdict(list)
)


def is_enabled() -> bool:
    return _enabled


def enable():
    """Enable TLI timing — call once at startup when TLI is configured."""
    global _enabled
    _enabled = True


def reset():
    """Clear all accumulated events.  Called at the start of each step."""
    global _op_events
    _op_events = defaultdict(list)


def record_start(op: str) -> torch.cuda.Event:
    """Record a start event just before a TLI operation.  Returns the event."""
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def record_end(op: str, start_ev: torch.cuda.Event):
    """Record an end event right after a TLI operation and stash the pair."""
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    _op_events[op].append((start_ev, end_ev))


def _sum_events(
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> float:
    total = 0.0
    for s, e in events:
        total += s.elapsed_time(e)
    return total


def elapsed_ms_per_op() -> dict[str, float]:
    """Return {op: total_ms} for each tracked operation.
    Must be called AFTER a cuda synchronize / event.synchronize()."""
    return {op: _sum_events(evts) for op, evts in _op_events.items()}


def elapsed_ms_total() -> float:
    """Sum all TLI operation times across all ops (ms).
    Must be called AFTER a cuda synchronize / event.synchronize()."""
    return sum(elapsed_ms_per_op().values())
