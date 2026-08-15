# SPDX-License-Identifier: Apache-2.0
"""Opt-in CUDA-event timing for speculative-decoding phases.

Gives `draft_forward_ms` and `target_forward_ms` as GPU time, with no HTTP, scheduler or
detokenizer in them -- the quantities you cannot obtain from wall-clock benchmarking,
because a single request's per-request overhead can exceed the forward itself.

Enable with VLLM_SPEC_PHASE_TIMING=1 and point VLLM_SPEC_PHASE_TIMING_OUT at a json path.
Off by default: record() is a no-op that allocates nothing.

DESIGN NOTES -- each of these was a real failure mode when the equivalent timer was built
for SGLang, so they are constraints, not preferences.

1. RECORD AT THE CALL SITE, OUTSIDE ANY CAPTURED REGION.
   These wrap `propose_draft_token_ids()` and `_model_forward()` from the outside. That
   keeps them valid with CUDA graphs ON, i.e. in production configuration: the Python at
   the call site still runs and the events bracket the whole replay. Timers placed INSIDE
   the model are a different story -- dynamo refuses to trace Event.record(), and forcing
   eager execution makes the events measure kernel-launch idle rather than compute (at
   batch 1 a graph-off forward measured ~9x its graph-on cost, nearly all of it waiting).

2. SKIP WHILE COMPILING OR CAPTURING.
   vLLM compiles aggressively (CompilationMode, cudagraph_mode=FULL_AND_PIECEWISE).
   Without the guard, dynamo aborts capture with
   `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped`, and the
   half-recorded pair then poisons the next fold with "Both events must be recorded
   before calculating elapsed time". torch.compiler.is_compiling() is traced as a
   constant, so the branch folds away and the compiled graph contains no timing at all.

3. FOLD RARELY, AND MAKE IT TUNABLE.
   elapsed_time() requires a device sync. Syncing per record serialises the GPU; syncing
   too rarely means the sink is not written at your measurement boundaries. Neither
   default is right for every use, hence VLLM_SPEC_PHASE_FLUSH_EVERY.

4. WRITE INCREMENTALLY, PER RANK.
   Workers are separate processes and are usually SIGKILLed, so atexit cannot be relied
   on -- the incremental write is the reliable path. Each rank writes its own file: rank
   spread is itself a result (a 4-rank all-reduce measured 13.8/11.4/13.5/1.0 ms at batch
   1, i.e. three ranks spin-waiting), and collapsing to rank 0 would report one rank's
   idle as the cost.

5. COUNTS ALONGSIDE TOTALS.
   Totals are cumulative over a worker's lifetime. Without the matching step count you
   cannot form a mean, and dividing by an unrelated denominator (e.g. a request count)
   silently produces a wrong per-step number.

COMM TIMING
-----------
`comm_scope()` wraps CudaCommunicator.all_reduce and attributes each call to whichever
phase is currently open, producing `comm.target_forward` / `comm.draft` alongside the
phase totals. That is the split wall-clock benchmarking cannot give you: how much of a
verify step is tensor-parallel communication rather than compute.

READ COMM NUMBERS WITH ONE CAVEAT: an all-reduce kernel's duration includes waiting for
the slowest peer, not just data movement. On a 4-rank NVLink box at batch 1, per-rank comm
measured 13.8 / 11.4 / 13.5 / 1.0 ms -- three ranks spin-waiting on the fourth. At high
batch the spread collapsed to ~2 ms and the number became meaningful. Always compare
across ranks (that is why each rank writes its own sink) before quoting a comm share; a
large spread means you are measuring skew, not interconnect.

Comm fires ~2x per layer, so ~120 calls per forward on a 60-layer model -- far more often
than the phase timers. It therefore gets its own, larger flush interval, or the sync from
folding would dominate what it is trying to measure.

It also has its OWN env gate, VLLM_SPEC_COMM_TIMING, and is off even when phase timing is
on: at 9774 calls in one short run it cost 4x throughput (28.6 -> 7.3 tok/s), which is
fine for a comm study and fatal for a throughput sweep.

SCOPE: this module serves the V1 model runner. The V2 runner -- which DSpark *forces*, see
VllmConfig.use_v2_model_runner -- carries its own per-step timing in
vllm/v1/worker/gpu/async_utils.py (StepTimingCollector, VLLM_SPEC_STEP_TIMING). comm_scope
is shared by both, since it hooks the communicator rather than a runner.
"""

from __future__ import annotations

import atexit
import json
import os
from collections import defaultdict
from contextlib import contextmanager

import torch

_PENDING: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
_TOTALS: dict[str, list[float]] = {}          # phase -> [total_ms, count]
_ENABLED: bool | None = None
_COMM_ENABLED: bool | None = None
_REGISTERED = False

_FLUSH_EVERY = int(os.environ.get("VLLM_SPEC_PHASE_FLUSH_EVERY", "32"))
# Comm fires ~2x per layer (~120x per forward on a 60-layer model) against once per
# forward for a phase, so it needs a much larger interval or folding dominates.
_COMM_FLUSH_EVERY = int(os.environ.get("VLLM_SPEC_COMM_FLUSH_EVERY", "2048"))

_CURRENT_PHASE: str = ""


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def enabled() -> bool:
    global _ENABLED, _REGISTERED
    if _ENABLED is None:
        _ENABLED = bool(os.environ.get("VLLM_SPEC_PHASE_TIMING")) and torch.cuda.is_available()
        if _ENABLED and not _REGISTERED:
            atexit.register(dump)
            _REGISTERED = True
    return _ENABLED


def comm_enabled() -> bool:
    """Comm timing is gated SEPARATELY from phase timing, and defaults off.

    all_reduce fires ~2x per layer -- a measured 9774 calls in a single short
    run against ~1 per forward for a phase -- and turning it on cost 4x
    throughput (28.6 -> 7.3 tok/s). At that point the instrumentation is no
    longer observing the workload, it is defining it, so a throughput sweep must
    be able to keep phase timing on while leaving comm timing off.
    """
    global _COMM_ENABLED
    if _COMM_ENABLED is None:
        _COMM_ENABLED = (
            bool(os.environ.get("VLLM_SPEC_COMM_TIMING")) and torch.cuda.is_available()
        )
        if _COMM_ENABLED:
            enabled()  # ensure the atexit dump is registered
    return _COMM_ENABLED


def _tracing() -> bool:
    """True while this region is being traced/captured rather than plainly executed."""
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        pass
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


@contextmanager
def record(phase: str):
    """Time a phase with a CUDA-event pair. No-op (and no allocation) when disabled.

    Also marks `phase` as current, so any all-reduce inside is attributed to it via
    comm_scope(). Restores the previous label on exit so nesting is safe.
    """
    global _CURRENT_PHASE
    if not enabled() or _tracing():
        yield
        return
    prev = _CURRENT_PHASE
    _CURRENT_PHASE = phase
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _CURRENT_PHASE = prev
        pending = _PENDING[phase]
        pending.append((start, end))
        if len(pending) >= _FLUSH_EVERY:
            _fold(phase)
            _write()


@contextmanager
def record_optional(phase: str | None):
    """record(phase), or a no-op when phase is None.

    Lets a call site decide per-invocation whether this forward should be counted --
    used to keep PREFILL out of target_forward, since a prefill and a verify step
    process completely different token counts and averaging them is meaningless.
    """
    if phase is None:
        yield
        return
    with record(phase):
        yield


@contextmanager
def comm_scope():
    """Time one collective and attribute it to the currently open phase.

    Hooked into CudaCommunicator.all_reduce, which is the single funnel for TP
    all-reduce. Calls outside any phase land in `comm.other` rather than being dropped,
    so the totals stay honest about what was not attributed.
    """
    if not comm_enabled() or _tracing():
        yield
        return
    key = f"comm.{_CURRENT_PHASE or 'other'}"
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        pending = _PENDING[key]
        pending.append((start, end))
        if len(pending) >= _COMM_FLUSH_EVERY:
            _fold(key)
            _write()


def _fold(phase: str) -> None:
    pending = _PENDING.get(phase)
    if not pending:
        return
    torch.cuda.synchronize()  # required before Event.elapsed_time
    total, count = _TOTALS.get(phase, [0.0, 0])
    for s, e in pending:
        # A pair can be half-recorded if the region unwound between start and end (an
        # aborted capture does exactly this). Drop it rather than take down every other
        # phase's totals with a ValueError.
        try:
            total += s.elapsed_time(e)
        except Exception:
            continue
        count += 1
    _TOTALS[phase] = [total, count]
    pending.clear()


def _sink_path() -> str | None:
    out = os.environ.get("VLLM_SPEC_PHASE_TIMING_OUT")
    if not out:
        return None
    base, ext = os.path.splitext(out)
    return f"{base}.rank{_rank()}{ext or '.json'}"


def _write() -> None:
    path = _sink_path()
    if not path:
        return
    data = {ph: {"total_ms": t, "count": c} for ph, (t, c) in _TOTALS.items()}
    data["_rank"] = _rank()
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def dump() -> None:
    """Fold anything pending and write. Best-effort: the incremental _write() in record()
    is the reliable path, since workers are usually killed by signal."""
    if not (_ENABLED or _COMM_ENABLED):
        return
    for phase in list(_PENDING):
        _fold(phase)
    _write()


def reset() -> None:
    """Drop all accumulated timing. For measuring a bounded window (e.g. excluding
    warmup) without restarting the worker."""
    _PENDING.clear()
    _TOTALS.clear()


def snapshot() -> dict[str, tuple[float, int]]:
    """Cumulative folded totals, WITHOUT forcing a fold.

    Deliberately does not sync: this is called every step to attach timings to
    ModelRunnerOutput, and syncing per step would cost more than the thing being
    measured. It therefore lags by up to one flush interval, which is fine for
    monotonically-increasing counters -- the missing tail lands on the next read.
    """
    if not enabled():
        return {}
    return {ph: (t, c) for ph, (t, c) in _TOTALS.items()}


def totals_ms() -> dict[str, dict[str, float]]:
    for phase in list(_PENDING):
        _fold(phase)
    return {ph: {"total_ms": t, "count": c} for ph, (t, c) in _TOTALS.items()}
