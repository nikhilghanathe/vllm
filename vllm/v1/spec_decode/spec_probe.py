# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative-decoding probe: per-position draft/target distribution logging.

Purpose
-------
Offline instrumentation to answer the go/no-go question for entropy-gated
tree speculation: *does a draft-time signal (entropy / top-1 prob) predict
acceptance, and is the target argmax in the draft's top-K?*

It logs, for every speculated token position, the draft-side distribution
stats (available at decision time) and the target-side stats + acceptance
(available only as an offline label, from the verification forward).

Design
------
- OFF by default. Enabled via ``VLLM_SPEC_PROBE=1``. When off, every entry
  point is a cheap boolean check — zero hot-path cost.
- Runs inside the GPU worker process. Both the draft proposer and the
  rejection sampler execute there, so a process-local singleton can join
  draft-side and target-side records without IPC.
- Join is by EXACT draft-token-id match per request (a self-validating
  checksum). Misaligned/unmatched verifications are dropped and counted,
  never silently mis-joined.
- Records buffer in memory and flush to ``VLLM_SPEC_PROBE_DIR`` on process
  exit (atexit), one ``.npz`` per TP rank. Only the draft rank populates
  draft stats (single_gpu_draft), so only it captures.

Output columns (one row per speculated position):
    engine_step, req, spec_pos, accepted (bool),
    draft_top1_prob, draft_entropy, draft_argmax,
    target_top1_prob, target_entropy, target_argmax,
    draft_topk_ids[K], draft_topk_probs[K]
The analyzer derives P(target_argmax in draft_topk) for any K' <= K offline.

WARNING: this is an ANALYSIS instrument. It adds softmax/topk + D2H copies
on the sampling path and perturbs timing. Do NOT use probe runs for
throughput/latency measurement.
"""

from __future__ import annotations

import atexit
import os
from collections import deque

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID = -1  # mirrors rejection_sampler.PLACEHOLDER_TOKEN_ID


def _enabled() -> bool:
    return bool(os.environ.get("VLLM_SPEC_PROBE"))


class SpecProbe:
    def __init__(self) -> None:
        self.enabled = _enabled()
        self.topk = int(os.environ.get("VLLM_SPEC_PROBE_TOPK", "32"))
        self.out_dir = os.environ.get("VLLM_SPEC_PROBE_DIR", "spec_probe_out")
        # Only the rank whose proposer installs the draft hook captures.
        self.draft_active = False
        self.rank = 0

        # Per-propose() scratch: list of per-step GPU stat dicts.
        self._cur_steps: list[dict] = []
        # Recent finished draft records (most-recent last), for the join.
        self._draft_records: deque = deque(maxlen=8)

        self._engine_step = 0
        # Column buffers (lists of numpy scalars / rows).
        self._cols: dict[str, list] = {
            k: [] for k in (
                "engine_step", "req", "spec_pos", "accepted",
                "draft_top1_prob", "draft_entropy", "draft_argmax",
                "target_top1_prob", "target_entropy", "target_argmax",
            )
        }
        self._topk_ids: list[np.ndarray] = []
        self._topk_probs: list[np.ndarray] = []

        self._n_match = 0
        self._n_miss = 0
        self._dumped = False

        if self.enabled:
            atexit.register(self.dump)
            logger.info("SpecProbe ENABLED (topk=%d, out_dir=%s). "
                        "This is an analysis run; timing is perturbed.",
                        self.topk, self.out_dir)

    # ---------- draft side (called from DraftModelProposer) ----------

    def set_rank(self, rank: int) -> None:
        self.rank = int(rank)

    def arm(self) -> None:
        """Mark that the draft model runs in this process/rank."""
        self.draft_active = True

    def begin_draft(self) -> None:
        if not self.enabled:
            return
        self._cur_steps = []

    def record_draft_step(self, logits: torch.Tensor) -> None:
        """Called once per draft step with logits [batch, vocab]."""
        if not self.enabled:
            return
        try:
            lg = logits.detach().float()
            probs = torch.softmax(lg, dim=-1)
            top1, argmax = probs.max(dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            k = min(self.topk, probs.shape[-1])
            tk_probs, tk_ids = torch.topk(probs, k, dim=-1)
            self._cur_steps.append({
                "top1": top1, "argmax": argmax, "ent": ent,
                "tk_ids": tk_ids, "tk_probs": tk_probs,
            })
        except Exception as e:  # never break the run
            logger.warning("SpecProbe.record_draft_step failed: %s", e)

    def end_draft(self, draft_token_ids: torch.Tensor) -> None:
        """Finalize the current propose(): stack per-step stats, move to CPU
        once, and index by per-request draft-id tuple for the join.

        draft_token_ids: [batch, L] (final, target-vocab ids).
        """
        if not self.enabled or not self._cur_steps:
            return
        try:
            steps = self._cur_steps
            self._cur_steps = []
            # [batch, L]
            top1 = torch.stack([s["top1"] for s in steps], dim=1).cpu().numpy()
            ent = torch.stack([s["ent"] for s in steps], dim=1).cpu().numpy()
            argmax = torch.stack([s["argmax"] for s in steps], dim=1).cpu().numpy()
            # [batch, L, K]
            tk_ids = torch.stack([s["tk_ids"] for s in steps], dim=1).cpu().numpy()
            tk_probs = torch.stack([s["tk_probs"] for s in steps], dim=1).cpu().numpy()
            ids = draft_token_ids.detach().cpu().numpy()
            if ids.ndim == 1:
                ids = ids.reshape(-1, top1.shape[1])

            # Map each request row's draft-id tuple -> row index (join key).
            index = {}
            for r in range(ids.shape[0]):
                index[tuple(int(x) for x in ids[r])] = r

            self._draft_records.append({
                "ids": ids, "index": index,
                "top1": top1, "ent": ent, "argmax": argmax,
                "tk_ids": tk_ids, "tk_probs": tk_probs,
            })
        except Exception as e:
            logger.warning("SpecProbe.end_draft failed: %s", e)

    # ---------- target side (called from RejectionSampler.forward) ----------

    def capture_verification(self, metadata, target_logits: torch.Tensor,
                             output_token_ids: torch.Tensor) -> None:
        """Join target-side stats + acceptance against the matching draft
        record and emit per-position rows.

        metadata: SpecDecodeMetadata (draft_token_ids flat, num_draft_tokens,
                  cu_num_draft_tokens).
        target_logits: [total_draft_tokens, vocab] in (req, spec_pos) order.
        output_token_ids: [batch, max_spec_len + 1]; rejected = PLACEHOLDER.
        """
        if not (self.enabled and self.draft_active) or not self._draft_records:
            return
        try:
            num_draft = list(metadata.num_draft_tokens)
            draft_ids_flat = metadata.draft_token_ids.detach().cpu().numpy()

            # Target distribution stats over all speculated positions (1 D2H).
            tlg = target_logits.detach().float()
            tprobs = torch.softmax(tlg, dim=-1)
            t_top1, t_argmax = tprobs.max(dim=-1)
            t_ent = -(tprobs * torch.log(tprobs.clamp_min(1e-12))).sum(dim=-1)
            t_top1 = t_top1.cpu().numpy()
            t_argmax = t_argmax.cpu().numpy()
            t_ent = t_ent.cpu().numpy()

            out = output_token_ids.detach().cpu().numpy()  # [batch, max_spec+1]

            rec = self._draft_records[-1]  # most recent propose()
            step = self._engine_step
            self._engine_step += 1

            start = 0
            for i, n_i in enumerate(num_draft):
                if n_i <= 0:
                    continue
                req_draft_ids = draft_ids_flat[start:start + n_i]
                key = tuple(int(x) for x in req_draft_ids)
                row = rec["index"].get(key)
                if row is None:
                    # fall back to one record back, else miss
                    row = None
                    for older in reversed(self._draft_records):
                        row = older["index"].get(key)
                        if row is not None:
                            rec_use = older
                            break
                    if row is None:
                        self._n_miss += 1
                        start += n_i
                        continue
                else:
                    rec_use = rec

                # Accepted-DRAFT prefix length for request i.
                # NOTE: a rejection writes a *recovered* token (the target
                # argmax) into the output at the reject position — it is NOT a
                # placeholder, so counting non-placeholders overcounts accepted
                # drafts by 1 at every rejection. A draft token is truly
                # accepted only where output == the draft id. Acceptance is a
                # greedy prefix, so take the leading run of matches.
                max_spec = out.shape[1] - 1
                row_out = out[i, :max_spec]
                accepted_len = 0
                for j in range(n_i):
                    if int(row_out[j]) == int(req_draft_ids[j]):
                        accepted_len += 1
                    else:
                        break

                for j in range(n_i):
                    self._cols["engine_step"].append(step)
                    self._cols["req"].append(i)
                    self._cols["spec_pos"].append(j)
                    self._cols["accepted"].append(bool(j < accepted_len))
                    self._cols["draft_top1_prob"].append(float(rec_use["top1"][row, j]))
                    self._cols["draft_entropy"].append(float(rec_use["ent"][row, j]))
                    self._cols["draft_argmax"].append(int(rec_use["argmax"][row, j]))
                    self._cols["target_top1_prob"].append(float(t_top1[start + j]))
                    self._cols["target_entropy"].append(float(t_ent[start + j]))
                    self._cols["target_argmax"].append(int(t_argmax[start + j]))
                    self._topk_ids.append(rec_use["tk_ids"][row, j])
                    self._topk_probs.append(rec_use["tk_probs"][row, j])
                self._n_match += 1
                start += n_i
        except Exception as e:
            logger.warning("SpecProbe.capture_verification failed: %s", e)

    # ---------- output ----------

    def dump(self) -> None:
        if not self.enabled or self._dumped:
            return
        self._dumped = True
        try:
            if not self._cols["req"]:
                logger.warning("SpecProbe: no rows captured "
                               "(match=%d miss=%d).", self._n_match, self._n_miss)
                return
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"spec_probe_rank{self.rank}.npz")
            arrays = {k: np.asarray(v) for k, v in self._cols.items()}
            arrays["draft_topk_ids"] = np.stack(self._topk_ids, axis=0)
            arrays["draft_topk_probs"] = np.stack(self._topk_probs, axis=0)
            np.savez_compressed(path, **arrays)
            logger.info("SpecProbe: wrote %d rows to %s "
                        "(joined requests=%d, misses=%d).",
                        len(arrays["req"]), path, self._n_match, self._n_miss)
        except Exception as e:
            logger.warning("SpecProbe.dump failed: %s", e)


_PROBE: SpecProbe | None = None


def get_probe() -> SpecProbe:
    global _PROBE
    if _PROBE is None:
        _PROBE = SpecProbe()
    return _PROBE
