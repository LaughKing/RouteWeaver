# Copyright 2026 RouteWeaver.
"""The progressive-exploration schedule, as PURE functions (no verl, no torch).

Shared by the agent loop (which decides forced or free at rollout time) and the
support hook (which weights its loss by the same thing). One definition, so the
two cannot disagree.

    rho_b = 1 - (b - FORCED_START_BATCH) / FORCED_ANNEAL_STEPS

A row's mode is forced with probability rho_b, drawn from the sample id alone.
The draw is per QUERY, not per rollout: a GRPO group holding both forced and
free rollouts of one query would compare a trajectory whose mode was given
against one that had to spend a decision on it, and the group baseline would
absorb the difference as if it were policy quality. Being a hash of the id, it
is also identical on every worker, reproducible on replay, and unchanged
across a resume -- none of which a process-local counter would be.
"""
import hashlib
import os

DEFAULT_START_BATCH = int(os.environ.get("FORCED_START_BATCH", "25"))
DEFAULT_STEPS = int(os.environ.get("FORCED_ANNEAL_STEPS", "50"))

MODES = ("single", "multi", "agentic")


def alpha_for(batch_index, start_batch=None, steps=None):
    """rho_b: the fraction of queries still forced at this step. Clamped to [0,1]."""
    start = DEFAULT_START_BATCH if start_batch is None else start_batch
    total = DEFAULT_STEPS if steps is None else steps
    t = int(batch_index) - int(start)
    return min(1.0, max(0.0, 1.0 - t / float(total)))


def is_forced(sample_id, alpha):
    """Deterministic per-QUERY draw: u = sha256(sample_id), forced iff u < alpha."""
    if alpha >= 1.0:
        return True
    if alpha <= 0.0:
        return False
    digest = hashlib.sha256(str(sample_id).encode()).digest()
    u = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return u < alpha


def forced_for_row(extra, start_batch=None, steps=None):
    """Whether this row's mode is forced. The schedule's only entry point.

    A row carrying an explicit arm label is REFUSED rather than honoured: that
    layout decides per query how many rows it occupies and needs a matching
    loss-side mask, and this repository has neither the builder that writes it
    nor the mask that would neutralize its padding rows. Half-honouring it
    would roll padding out as ordinary free rows and train on them.
    """
    if str(extra.get("stage2_arm") or "").strip():
        raise ValueError(
            "extra_info['stage2_arm'] is not supported: the schedule here is "
            "the per-query hash below. Remove the column, or supply the "
            "loss-side mask that layout requires.")
    alpha = alpha_for(extra.get("batch_index") or 0, start_batch, steps)
    return is_forced(extra.get("sample_id") or "", alpha)
