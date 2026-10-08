# Copyright 2026 RouteWeaver.
"""Segment-wise loss reduction for the hierarchical (mode / inner) advantage.

THE PROBLEM. `comet_grpo` (comet_grpo.py) writes A_mode on the mode-selection
tokens of a free rollout and A_inner on the tokens after it, in ONE advantage
tensor. verl then reduces the token-level PPO loss with one global token mean,
so the mode decision -- `<mode>multi</mode>` = 7 tokens against ~194 tokens of
trajectory -- receives ~3.5% of the gradient mass purely because it serialises
into fewer tokens. The relative strength of the two decision levels is set by
tokenizer arithmetic, not by design.

THE CHANGE (ROUTEWEAVER_LOSS_REDUCTION=segment). Reduce each segment by its own
token count and add the two scalars:

    L = sum_{t in mode} L_PG(t) / N_mode  +  sum_{t in inner} L_PG(t) / N_inner

No lambda, no extra weight. Nothing else moves: the same `compute_policy_loss_*`
produces the token-level matrix (same ratio, same clipping, same rollout IS
weights), the advantages are untouched, and entropy and KL keep their original
single reduction over the full response mask.

------------------------------------------------------------------------------
HOW verl NORMALISES, AND THE BUG THIS FILE ONCE HAD
------------------------------------------------------------------------------
`verl/workers/engine/fsdp/transformer_impl.py:617 forward_backward_batch` does,
for one MINI-batch:

    batch_num_tokens = data["loss_mask"].sum()          # loss_mask IS response_mask
    all_reduce(batch_num_tokens, SUM, dp_group)         # -> N, a DP-global constant
    micro_batches = prepare_micro_batches(data)
    for mb in micro_batches:
        loss = loss_function(mb); loss.backward()       # gradients accumulate

and `agg_loss(..., "token-mean")` (core_algos.py:1173) computes, per micro batch

    loss_b = masked_sum(loss_mat_b, mask_b) / N * dp_size

with THE SAME constant N for every micro batch. Summing over micro batches
gives (local masked_sum)/N*dp_size; FSDP then averages grads over DP ranks, so
the optimiser sees (global masked_sum)/N -- a true global token mean, and it is
invariant to the micro-batch split because N does not depend on the split.

The first version of this file replaced N with the CURRENT MICRO BATCH's own
segment token count. That makes each micro batch normalise by its own n_b, so
the accumulated loss is sum_b masked_sum_b / n_b instead of sum_b masked_sum_b / N
-- roughly N_micro times too large, and by construction sensitive to the
micro-batch split. With one row per micro batch it inflates pg_loss by about
the number of micro batches per rank, and the gradient norm with it -- which is
how the bug was found.

THE FIX. A segment's denominator must be a DP-global, mini-batch-wide constant,
computed exactly the way verl computes `batch_num_tokens`:

    N_mode  = sum over the whole mini-batch of mode_token_mask, all-reduced
    N_inner = batch_num_tokens - N_mode        (exact: mode and inner partition
                                                response_mask, and verl's
                                                loss_mask IS response_mask)

`install_engine` wraps `FSDPEngine.forward_backward_batch` to compute N_mode on
the mini-batch before it is split, and publishes it as non-tensor data, so every
micro batch reads the same constant -- the same mechanism, the same guarantees.

DEGENERACY. When no row is free, mode_token_mask is all zero: N_mode = 0 (the
mode branch's masked_sum is 0, so its clamped denominator is irrelevant) and
N_inner = batch_num_tokens exactly, with inner_mask == response_mask. The
segment path is then bit-for-bit the original global reduction.

WHERE IT ATTACHES. `verl.workers.engine_workers` binds `ppo_loss` into its own
namespace at import and freezes it into a partial when the worker is built, so
the swap must happen right after that module executes (sitecustomize.py).
No verl source is edited.
"""
import functools
import os

import torch
import torch.distributed as dist

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding

MASK_KEY = "mode_token_mask"    # per-token 0/1, written by trainers/comet_grpo.publish_mode_token_mask
COUNT_KEY = "mode_num_tokens"   # DP-global mini-batch count of MASK_KEY, written by install_engine
# THIRD segment: the worker-name tokens. Its reduction is NOT a token mean over
# the segment. Requirement: a long worker id must not receive more gradient
# than a short one purely for being longer, so
# each ROW contributes 1/n_i * sum over its own worker tokens and the rows are
# averaged. Implemented by scaling the advantage on worker tokens by
# n_worker_global / (rows * n_i) and reducing with the global denominator, which
# is algebraically identical and reuses the same clipped policy-loss kernel
# (the scale is positive, so it commutes with PPO's min/clip).
WORKER_MASK_KEY = "worker_token_mask"
WORKER_COUNT_KEY = "worker_num_tokens"
WORKER_ROWS_KEY = "worker_num_rows"
# Provider-failure rows. Zeroing the ADVANTAGE alone is not enough: entropy is
# off (coeff 0) but kl_loss_coef=0.001 reduces over response_mask, so a failed
# row would still pull the policy toward the reference. Every mask this file
# reduces over is multiplied by this flag, so such a row contributes exactly
# nothing to the actor gradient -- PG, entropy and KL alike.
CLEAN_ROW_KEY = "routeweaver_clean_row"
_ORIGINAL = {"fn": None}
LAST_STATS = {}


def enabled() -> bool:
    return os.environ.get("ROUTEWEAVER_LOSS_REDUCTION", "global") == "segment"


def segment_masks(response_mask, mode_token_mask, worker_token_mask=None):
    """-> (mode, worker, inner), disjoint, union == response_mask.

    worker_token_mask=None reproduces the two-segment behaviour exactly, with an
    all-false worker mask, so a run without the exploration curriculum is
    bit-for-bit unchanged."""
    resp = response_mask.to(torch.bool)
    mode = (mode_token_mask > 0) & resp
    if worker_token_mask is None:
        worker = torch.zeros_like(resp)
    else:
        worker = (worker_token_mask > 0) & resp & ~mode
    return mode, worker, resp & ~mode & ~worker


def segment_denominators(batch_num_tokens, mode_num_tokens):
    """The two DP-global, mini-batch-wide denominators -> (N_mode, N_inner).

    `batch_num_tokens` is verl's own global response-token count for this mini
    batch and `mode_num_tokens` the global count of the mode segment inside it,
    so the inner count is their difference -- no second all-reduce, and no way
    for the two denominators to disagree with verl's own normalisation.

    clamp to 1: an empty segment must not divide by zero. Its masked_sum is 0
    over every micro batch, so the term contributes exactly 0 either way.
    """
    if batch_num_tokens is None:
        raise ValueError("segment reduction needs verl's global batch_num_tokens")
    n_mode = float(mode_num_tokens)
    n_inner = float(batch_num_tokens) - n_mode
    if n_mode < 0 or n_inner < 0:
        raise ValueError(f"impossible segment counts: mode={n_mode} of {batch_num_tokens}")
    return max(n_mode, 1.0), max(n_inner, 1.0)


def ppo_loss_segment(config, model_output, data, dp_group=None):
    """verl's ppo_loss with the policy-gradient term reduced per segment."""
    if not enabled() or MASK_KEY not in data.keys():
        return _ORIGINAL["fn"](config, model_output, data, dp_group=dp_group)
    if COUNT_KEY not in data.keys():
        # Falling back would silently normalise by a micro-batch-local count --
        # the exact bug this file exists to fix. Fail where it is visible.
        raise RuntimeError(
            f"segment reduction needs '{COUNT_KEY}' on the micro batch; "
            "segment_loss.install_engine() did not run on FSDPEngine")

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info, exactly as the original (used by entropy / KL below)
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    n_worker = float(data[WORKER_COUNT_KEY]) if WORKER_COUNT_KEY in data.keys() else 0.0
    n_rows = float(data[WORKER_ROWS_KEY]) if WORKER_ROWS_KEY in data.keys() else 0.0
    n_mode, n_inner = segment_denominators(
        data["batch_num_tokens"] - n_worker, data[COUNT_KEY])

    fields = ["response_mask", "old_log_probs", "advantages", MASK_KEY]
    if WORKER_MASK_KEY in data.keys():
        fields.append(WORKER_MASK_KEY)
    if CLEAN_ROW_KEY in data.keys():
        fields.append(CLEAN_ROW_KEY)
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(torch.bool)
    if CLEAN_ROW_KEY in data.keys():
        response_mask = response_mask & (data[CLEAN_ROW_KEY] > 0)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
    mode_mask, worker_mask, inner_mask = segment_masks(
        response_mask, data[MASK_KEY],
        data[WORKER_MASK_KEY] if WORKER_MASK_KEY in data.keys() else None)
    # per-row 1/n_i, folded into the advantage so the shared kernel can be reused
    worker_adv = advantages
    if n_worker > 0 and n_rows > 0:
        per_row = worker_mask.sum(dim=-1).clamp_min(1).to(advantages.dtype)
        scale = (n_worker / (n_rows * per_row)).unsqueeze(-1)
        worker_adv = advantages * torch.where(worker_mask, scale,
                                              torch.ones_like(scale))

    policy_loss_fn = get_policy_loss_fn(config.policy_loss.get("loss_mode", "vanilla"))
    saved = config.global_batch_info["batch_num_tokens"]
    metrics = {}
    pg_parts = {}
    segs = [("mode", mode_mask, n_mode, advantages), ("inner", inner_mask, n_inner, advantages)]
    if n_worker > 0 and n_rows > 0:
        segs.append(("worker", worker_mask, n_worker, worker_adv))
    for name, mask, denom, adv_for_seg in segs:
        # each segment is normalised by ITS OWN mini-batch-global token count ->
        # a mean over that segment, not a share of the response's token budget
        config.global_batch_info["batch_num_tokens"] = denom
        loss, seg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=adv_for_seg,
            response_mask=mask,
            loss_agg_mode=config.loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_parts[name] = loss
        if name == "inner":                      # keep the familiar metric names
            metrics.update(Metric.from_dict(seg_metrics, aggregation=AggregationType.MEAN))
    config.global_batch_info["batch_num_tokens"] = saved

    pg_loss = pg_parts["mode"] + pg_parts["inner"] + pg_parts.get("worker", 0.0)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    metrics["actor/pg_loss_mode"] = Metric(value=pg_parts["mode"].detach(), aggregation=metric_aggregation)
    metrics["actor/pg_loss_inner"] = Metric(value=pg_parts["inner"].detach(), aggregation=metric_aggregation)
    if "worker" in pg_parts:
        metrics["actor/pg_loss_worker"] = Metric(value=pg_parts["worker"].detach(), aggregation=metric_aggregation)
    with torch.no_grad():
        # mini-batch-global, so these do NOT change with the micro-batch split
        stats = {
            "actor/mode_tokens_global": n_mode if mode_mask.any() or n_mode > 1 else 0.0,
            "actor/inner_tokens_global": n_inner,
            "actor/mode_token_frac": n_mode / max(n_mode + n_inner, 1.0),
            "actor/A_mode_abs": float((advantages.abs() * mode_mask).sum() / mode_mask.sum().clamp_min(1)),
            "actor/A_inner_abs": float((advantages.abs() * inner_mask).sum() / inner_mask.sum().clamp_min(1)),
            "actor/worker_tokens_global": n_worker,
            "actor/worker_rows_global": n_rows,
            "actor/A_worker_abs": float((advantages.abs() * worker_mask).sum() / worker_mask.sum().clamp_min(1)),
        }
    metrics.update(stats)
    LAST_STATS.clear(); LAST_STATS.update(stats)
    policy_loss = pg_loss

    # ---- entropy and KL: untouched, still one reduction over the full mask --
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info)
        policy_loss -= config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)
    if config.use_kl_loss:
        kld = kl_penalty(logprob=log_prob, ref_logprob=data["ref_log_prob"],
                         kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                           loss_agg_mode=config.loss_agg_mode, **config.global_batch_info)
        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    metrics["actor/loss_total"] = Metric(value=policy_loss.detach(), aggregation=metric_aggregation)
    return policy_loss, metrics


def count_mask_tokens(mask):
    """Token count of a mask that may be nested (NO_PADDING) or padded."""
    return mask.values().sum() if mask.is_nested else mask.sum()


def install(engine_workers_module) -> bool:
    """Swap `ppo_loss` in engine_workers' namespace. Inert unless
    ROUTEWEAVER_LOSS_REDUCTION=segment; the wrapper also falls back to the original
    per call, so a batch without the mask column behaves exactly as before."""
    if not enabled():
        return False
    original = getattr(engine_workers_module, "ppo_loss", None)
    if original is None or getattr(original, "_routeweaver_segment", False):
        return False
    _ORIGINAL["fn"] = original
    ppo_loss_segment._routeweaver_segment = True
    engine_workers_module.ppo_loss = ppo_loss_segment
    print("[segment_loss] segment-wise loss reduction installed", flush=True)
    return True


def install_engine(transformer_impl_module) -> bool:
    """Publish the mini-batch-global mode-token count, mirroring the way
    `forward_backward_batch` publishes `batch_num_tokens`: computed on the whole
    mini batch BEFORE it is split, all-reduced over the data-parallel group, and
    carried into every micro batch as the same constant."""
    if not enabled():
        return False
    cls = getattr(transformer_impl_module, "FSDPEngine", None)
    if cls is None:
        return False
    original = cls.forward_backward_batch
    if getattr(original, "_routeweaver_segment", False):
        return False

    from verl.utils import tensordict_utils as tu
    from verl.utils.device import get_device_id

    @functools.wraps(original)
    def forward_backward_batch(self, data, loss_function, forward_only=False):
        if MASK_KEY in data.keys():
            n = count_mask_tokens(data[MASK_KEY]).to(get_device_id()).to(torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(n, op=dist.ReduceOp.SUM, group=self.get_data_parallel_group())
            tu.assign_non_tensor(data, **{COUNT_KEY: float(n.item())})
        if WORKER_MASK_KEY in data.keys():
            wm = data[WORKER_MASK_KEY]
            nw = count_mask_tokens(wm).to(get_device_id()).to(torch.float64)
            nr = (wm.sum(dim=-1) > 0).sum().to(get_device_id()).to(torch.float64)
            if dist.is_available() and dist.is_initialized():
                g = self.get_data_parallel_group()
                dist.all_reduce(nw, op=dist.ReduceOp.SUM, group=g)
                dist.all_reduce(nr, op=dist.ReduceOp.SUM, group=g)
            tu.assign_non_tensor(data, **{WORKER_COUNT_KEY: float(nw.item()),
                                          WORKER_ROWS_KEY: float(nr.item())})
        return original(self, data, loss_function, forward_only=forward_only)

    forward_backward_batch._routeweaver_segment = True
    cls.forward_backward_batch = forward_backward_batch
    print("[segment_loss] mini-batch-global mode-token count published on FSDPEngine", flush=True)
    return True
