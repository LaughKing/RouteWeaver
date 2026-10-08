# Copyright 2026 RouteWeaver.
"""Reward-decoupled hierarchical advantage: ``comet_gdpo``.

The two-level decomposition is exactly `comet_grpo.comet_grpo`'s -- A_mode on the
mode-decision tokens, A_inner after them, ONE group sigma for both levels. The
only change is that it is done PER REWARD COMPONENT and the two results are
mixed afterwards, GDPO-style:

    for d in (task, eff):
        mu_d, sigma_d  = mean / Bessel std of the group's clean r_d
        mu_{m,d}       = mean of r_d over the rollouts that sampled mode m
        A_mode,d  = (mu_{m,d} - mu_d) / (sigma_d + eps)
        A_inner,d = (r_d - mu_{m,d}) / (sigma_d + eps)

    A_mode  = (1-alpha) A_mode,task  + alpha A_mode,eff
    A_inner = (1-alpha) A_inner,task + alpha A_inner,eff

and NOTHING is whitened afterwards. Mixing after each component is normalised
is the whole point: a task reward on {-1,0,1} and an efficiency reward on (0,1]
have different spreads, and normalising the sum would let the wider one set the
scale.

alpha = COST_ALPHA, a fixed constant. At alpha == 0 this module does not
approximate the baseline, it IS the baseline: the call is delegated to
`comet_grpo.compute_comet_grpo_advantage` unchanged, so the advantages, the gates,
the masks and the loss denominators are the old ones bit for bit.

ZERO-VARIANCE IS PER COMPONENT. A group where every rollout is correct has no
task signal, but if the rollouts cost different amounts the efficiency
component still does -- that group keeps training on cost alone. A group is
skipped only when BOTH components are flat.

COST-INCOMPLETE ROWS. A trajectory with an unmeasurable call (no usage
reported, or a worker the price table does not know) would otherwise look
free. With cost enabled such a row is treated exactly like a dispatch-gated
one: it leaves the gradient AND both group baselines, and it is counted.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch

import grpo_gated as AE
import cost_model
import comet_grpo
from verl.trainer.ppo import core_algos

MODES = comet_grpo.MODES
LAST_STATS = {}
_PENDING = {"info": None}


def enabled() -> bool:
    return os.environ.get("ADV_ESTIMATOR", "") == "comet_gdpo"


def stash_info(info):
    _PENDING["info"] = info


def _group_stats(vals, dev, dt, epsilon):
    """(mean, denom, gated) with verl's conventions: a group of one has mean 0
    and std 1, and a std below the gate threshold carries no signal."""
    if len(vals) == 1:
        return (torch.tensor(0.0, device=dev, dtype=dt),
                torch.tensor(1.0, device=dev, dtype=dt) + epsilon, False)
    stacked = torch.stack(vals)
    mean, std = torch.mean(stacked), torch.std(stacked)
    return mean, std + epsilon, bool(std < AE.GROUP_STD_THRESHOLD)


@core_algos.register_adv_est("comet_gdpo")
def compute_comet_gdpo_advantage(token_level_rewards, response_mask, index,
                                epsilon: float = 1e-6, config=None, **kwargs):
    alpha = cost_model.cost_alpha()
    info = _PENDING["info"]
    _PENDING["info"] = None
    if info is None:
        raise RuntimeError("comet_gdpo called without row info (wrapper not installed?)")

    if alpha == 0.0:
        # THE BASELINE PATH, not a reimplementation of it.
        comet_grpo.stash_info(info)
        return comet_grpo.compute_comet_grpo_advantage(
            token_level_rewards=token_level_rewards, response_mask=response_mask,
            index=index, epsilon=epsilon, config=config, **kwargs)

    if os.environ.get("WORKER_EXPLORE", "0") == "1":
        raise RuntimeError("comet_gdpo does not implement the worker-exploration "
                           "split; run it with WORKER_EXPLORE=0")
    norm_adv = True if config is None else getattr(config, "norm_adv_by_std_in_grpo", True)
    if not norm_adv:
        raise ValueError("comet_gdpo is defined with the group sigma; "
                         "norm_adv_by_std_in_grpo must be True")

    bsz, L = token_level_rewards.shape
    clean = AE._take_clean_mask(bsz)
    dev, dt = token_level_rewards.device, token_level_rewards.dtype
    free, modes, span_end = info["free"], info["mode"], info["span_end"]
    complete = info["cost_complete"]
    r = {"task": token_level_rewards.sum(-1),
         "eff": torch.tensor([float(v) for v in info["r_eff"]], device=dev, dtype=dt)}
    adv = torch.zeros_like(token_level_rewards)
    pos = torch.arange(L, device=dev).unsqueeze(0)

    groups = defaultdict(list)
    for i in range(bsz):
        groups[index[i]].append(i)

    n_free = n_forced = gated_both = dropped_rows = dropped_cost = 0
    gated_d = {"task": 0, "eff": 0}
    rows_d = {"task": [], "eff": []}         # |A_mode| per component
    inner_d = {"task": [], "eff": []}
    mode_count = {m: 0 for m in MODES}
    a_mode_rows, a_inner_rows, uniq, single_rows = [], [], [], 0

    for members in groups.values():
        arms = {int(free[i]) for i in members}
        if len(arms) > 1:
            raise RuntimeError("a GRPO group mixes forced and free rollouts")
        is_free = arms.pop() == 1
        surv = []
        for i in members:
            if clean is not None and not clean[i]:
                dropped_rows += 1
                continue
            if not int(complete[i]):
                dropped_cost += 1
                continue
            surv.append(i)
        if not surv:
            gated_both += 1
            continue

        stats, gated = {}, {}
        for d in ("task", "eff"):
            mean, denom, g = _group_stats([r[d][i] for i in surv], dev, dt, epsilon)
            stats[d] = (mean, denom)
            gated[d] = g
            gated_d[d] += int(g)
        if all(gated.values()):
            gated_both += 1
            continue

        by_mode = defaultdict(list)
        for i in surv:
            by_mode[modes[i]].append(i)
        if is_free:
            n_free += 1
            uniq.append(len(by_mode))
        else:
            n_forced += 1

        mu_m = {d: {m: torch.stack([r[d][i] for i in idx]).mean()
                    for m, idx in by_mode.items()} for d in ("task", "eff")}
        w = {"task": 1.0 - alpha, "eff": alpha}
        for i in surv:
            a_mode = torch.zeros((), device=dev, dtype=dt)
            a_inner = torch.zeros((), device=dev, dtype=dt)
            for d in ("task", "eff"):
                if gated[d]:
                    continue
                mean, denom = stats[d]
                if is_free:
                    m_d = (mu_m[d][modes[i]] - mean) / denom
                    i_d = (r[d][i] - mu_m[d][modes[i]]) / denom
                else:
                    # forced: the declaration is environment text (response_mask
                    # 0) and the whole response is "inner", i.e. plain GRPO on
                    # this component.
                    m_d = torch.zeros((), device=dev, dtype=dt)
                    i_d = (r[d][i] - mean) / denom
                a_mode = a_mode + w[d] * m_d
                a_inner = a_inner + w[d] * i_d
                rows_d[d].append(float(m_d))
                inner_d[d].append(float(i_d))
            if is_free:
                if len(by_mode[modes[i]]) == 1:
                    single_rows += 1
                e = int(span_end[i])
                if e < 0:
                    raise RuntimeError(f"free row {i} has no mode_span_end")
                tok = torch.where(pos[0] < e, a_mode, a_inner)
                adv[i] = tok * response_mask[i]
                a_mode_rows.append(float(a_mode))
                a_inner_rows.append(float(a_inner))
                if modes[i] in mode_count:
                    mode_count[modes[i]] += 1
            else:
                adv[i] = a_inner * response_mask[i]

    n_rows = len(a_mode_rows)
    stats_out = {
        "cost/alpha": alpha,
        "free/groups": n_free, "forced/groups": n_forced,
        "hier/groups_gated": gated_both,
        "hier/rows_dropped_dispatch": dropped_rows,
        "cost/rows_dropped_incomplete": dropped_cost,
        "cost/groups_zero_var_task": gated_d["task"],
        "cost/groups_zero_var_eff": gated_d["eff"],
        "cost/A_mode_abs_task": float(np.mean(np.abs(rows_d["task"]))) if rows_d["task"] else 0.0,
        "cost/A_mode_abs_eff": float(np.mean(np.abs(rows_d["eff"]))) if rows_d["eff"] else 0.0,
        "cost/A_inner_abs_task": float(np.mean(np.abs(inner_d["task"]))) if inner_d["task"] else 0.0,
        "cost/A_inner_abs_eff": float(np.mean(np.abs(inner_d["eff"]))) if inner_d["eff"] else 0.0,
        "free/rows": n_rows,
        "free/unique_modes_per_prompt": (sum(uniq) / len(uniq)) if uniq else 0.0,
        "free/singleton_mode_rate": (single_rows / n_rows) if n_rows else 0.0,
        "free/A_mode_abs_mean": float(np.mean(np.abs(a_mode_rows))) if a_mode_rows else 0.0,
        "free/A_inner_abs_mean": float(np.mean(np.abs(a_inner_rows))) if a_inner_rows else 0.0,
    }
    for m in MODES:
        stats_out[f"free/mode_count_{m}"] = mode_count[m]
    LAST_STATS.clear(); LAST_STATS.update(stats_out)
    print(f"[comet_gdpo] {json.dumps(stats_out)}", flush=True)
    return adv, adv


def _info_from_batch(data):
    """comet_grpo's row info plus the efficiency reward and its completeness."""
    info = comet_grpo._info_from_batch(data)
    cols = data.non_tensor_batch
    n = len(info["free"])
    eff = cols.get("r_eff")
    comp = cols.get("cost_complete")
    if eff is None or comp is None:
        raise RuntimeError(
            "comet_gdpo needs the r_eff and cost_complete columns; run with "
            "CostRewardManager as the reward manager")
    info["r_eff"] = [float(v) for v in eff]
    info["cost_complete"] = [int(v) for v in comp]
    assert len(info["r_eff"]) == n and len(info["cost_complete"]) == n
    return info


def install(ray_trainer_module) -> bool:
    """Wrap compute_advantage so comet_gdpo sees the per-row columns.

    Wraps OUTSIDE comet_grpo's own wrapper, which keeps publishing the segment
    masks and the clean-row column; this one only adds the second reward
    component to what the estimator can see.
    """
    import functools
    if not enabled():
        return False
    original = getattr(ray_trainer_module, "compute_advantage", None)
    if original is None or getattr(original, "_routeweaver_cost", False):
        return False

    @functools.wraps(original)
    def compute_advantage(data, *args, **kwargs):
        stash_info(_info_from_batch(data))
        try:
            return original(data, *args, **kwargs)
        finally:
            stash_info(None)

    compute_advantage._routeweaver_cost = True
    compute_advantage._routeweaver_hier = True
    compute_advantage._router_r1_dispatch_gate = True
    ray_trainer_module.compute_advantage = compute_advantage
    print("[comet_gdpo] compute_advantage wrapped (r_eff + cost_complete for comet_gdpo)",
          flush=True)
    return True
