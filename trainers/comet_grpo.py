# Copyright 2026 RouteWeaver.
"""Hierarchical advantage for FREE prompts inside the existing Stage-2 curriculum.

Registered verl estimator ``comet_grpo``. Forced groups are bit-for-bit the
``grpo_gated`` path (stock GRPO + zero-variance gate + dispatch gate). A
FREE group (K rollouts of one prompt, modes sampled by the policy) gets

    mu, sigma        = mean / Bessel std of the group's (clean) rewards
    A_grpo_i         = (R_i - mu) / (sigma + eps)
    mu_m             = mean of the rewards of the rollouts that sampled mode m
    A_mode_i         = (mu_{m_i} - mu) / (sigma + eps)
    A_inner_i        = (R_i - mu_{m_i}) / (sigma + eps)      => A_mode + A_inner == A_grpo

Token level: response positions [0, mode_span_end_i) (the mode decision,
retries included) carry A_mode; positions >= mode_span_end_i carry A_inner;
environment tokens are already response_mask 0. One sigma per group, never a
per-mode std. A group whose rollouts all sampled one mode has A_mode = 0 and
A_inner = A_grpo (plain GRPO); a singleton mode has A_inner = 0.

Row metadata (free flag, sampled mode, mode_span_end) is handed over by the
compute_advantage wrapper below, exactly like the dispatch gate's clean mask.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch

import grpo_gated as AE
from verl.trainer.ppo import core_algos

MODES = ("single", "multi", "agentic")
# batch column read by segment_loss.py; kept as a literal here so this module
# (imported the moment core_algos loads) never pulls in the loss module's
# verl.workers dependencies.
MODE_TOKEN_MASK_KEY = "mode_token_mask"
# batch column read by segment_loss.py for the THIRD segment. Same literal-here
# reasoning as MODE_TOKEN_MASK_KEY.
WORKER_TOKEN_MASK_KEY = "worker_token_mask"
# per-row 1/0: 0 == this rollout contained a failed worker call. Published so
# segment_loss.py can zero EVERY mask for the row, not merely its advantage --
# with kl_loss_coef=0.001 an advantage of zero still leaves a KL-to-reference
# gradient on the row, and the requirement is that a provider failure produce
# no policy gradient at all.
CLEAN_ROW_KEY = "routeweaver_clean_row"


def parse_worker_spans(cell):
    """'a,b,id;a,b,id' -> [(a, b, id)]. Written by the agent loop, one entry
    per <route model="ID"> the POLICY generated, token indices into the
    response. Empty for a row that wrote no route."""
    out = []
    for part in str(cell or "").split(";"):
        if not part:
            continue
        a, b, wid = part.split(",", 2)
        out.append((int(a), int(b), wid))
    return out
_PENDING = {"info": None}
LAST_STATS = {}


def stash_info(info):
    _PENDING["info"] = info


def enabled():
    return os.environ.get("ROUTEWEAVER_COMET", "0") == "1"


@core_algos.register_adv_est("comet_grpo")
def compute_comet_grpo_advantage(token_level_rewards, response_mask, index,
                                epsilon: float = 1e-6, config=None, **kwargs):
    norm_adv = True if config is None else getattr(config, "norm_adv_by_std_in_grpo", True)
    if not norm_adv:
        raise ValueError("comet_grpo is defined with the global sigma; norm_adv_by_std_in_grpo must be True")
    bsz, L = token_level_rewards.shape
    clean = AE._take_clean_mask(bsz)
    info = _PENDING["info"]; _PENDING["info"] = None
    if info is None:
        raise RuntimeError("comet_grpo called without row info (wrapper not installed?)")
    free, modes, span_end = info["free"], info["mode"], info["span_end"]
    dev, dt = token_level_rewards.device, token_level_rewards.dtype
    scores = token_level_rewards.sum(-1)
    adv = torch.zeros_like(token_level_rewards)
    pos = torch.arange(L, device=dev).unsqueeze(0)          # (1, L)

    groups = defaultdict(list)
    for i in range(bsz):
        groups[index[i]].append(i)

    st = defaultdict(float); n_free_groups = n_forced_groups = 0
    n_free_total = sum(1 for members in groups.values() if int(free[members[0]]) == 1)
    a_mode_rows, a_inner_rows, uniq, single_rows, ident_err = [], [], [], 0, 0.0
    mode_count = {m: 0 for m in MODES}; gated = dropped_rows = 0
    a_worker_rows, a_winner_rows, w_group_sizes, worker_uniq = [], [], [], []
    n_worker_groups = 0
    a_grpo_all = torch.zeros(bsz, device=dev, dtype=dt)
    for members in groups.values():
        arms = {int(free[i]) for i in members}
        if len(arms) > 1:
            raise RuntimeError("a GRPO group mixes forced and free rollouts")
        is_free = arms.pop() == 1
        surv = [i for i in members if (clean is None or clean[i])]
        dropped_rows += len(members) - len(surv)
        if not surv:
            gated += 1; continue
        if len(surv) == 1:
            mean = torch.tensor(0.0, device=dev, dtype=dt); std = torch.tensor(1.0, device=dev, dtype=dt)
        else:
            vals = torch.stack([scores[i] for i in surv])
            mean, std = torch.mean(vals), torch.std(vals)
            if std < AE.GROUP_STD_THRESHOLD:
                gated += 1; continue                      # zero-variance gate
        denom = std + epsilon
        if not is_free:
            n_forced_groups += 1
            wk = [info["worker"][i] for i in surv]
            # WORKER SPLIT (4x2 exploration). Exactly the mode decomposition one
            # level down: mu_w is the mean over the rollouts that were told to
            # use w, A_worker = (mu_w - mu)/sigma rides the worker-name tokens
            # and A_inner = (R - mu_w)/sigma rides everything else, so
            # A_worker + A_inner == A_grpo identically and the total objective is
            # unchanged. Needs >= 2 distinct workers in the group; with one
            # worker mu_w == mu, A_worker == 0 and this reduces to the old path.
            if all(wk) and len(set(wk)) > 1:
                by_w = defaultdict(list)
                for i in surv:
                    by_w[info["worker"][i]].append(i)
                mu_w = {w: torch.stack([scores[i] for i in idx]).mean()
                        for w, idx in by_w.items()}
                for i in surv:
                    w = info["worker"][i]
                    a_g = (scores[i] - mean) / denom
                    a_w = (mu_w[w] - mean) / denom
                    a_in = (scores[i] - mu_w[w]) / denom
                    a_grpo_all[i] = a_g
                    ident_err = max(ident_err, float((a_w + a_in - a_g).abs()))
                    row = torch.full((L,), float(a_in), device=dev, dtype=dt)
                    for a0, b0, _ in info["wspans"][i]:
                        if 0 <= a0 < b0 <= L:
                            row[a0:b0] = a_w
                    adv[i] = row * response_mask[i]
                    a_worker_rows.append(float(a_w))
                    a_winner_rows.append(float(a_in))
                    w_group_sizes.append(len(by_w[w]))
                n_worker_groups += 1
                worker_uniq.append(len(by_w))
                continue
            for i in surv:
                a = (scores[i] - mean) / denom
                a_grpo_all[i] = a
                adv[i] = a * response_mask[i]
            continue
        n_free_groups += 1
        by_mode = defaultdict(list)
        for i in surv:
            by_mode[modes[i]].append(i)
        uniq.append(len(by_mode))
        mu_m = {m: torch.stack([scores[i] for i in idx]).mean() for m, idx in by_mode.items()}
        for i in surv:
            m = modes[i]
            if len(by_mode[m]) == 1:
                single_rows += 1
            a_g = (scores[i] - mean) / denom
            a_m = (mu_m[m] - mean) / denom
            a_in = (scores[i] - mu_m[m]) / denom
            a_grpo_all[i] = a_g
            ident_err = max(ident_err, float((a_m + a_in - a_g).abs()))
            e = int(span_end[i])
            if e < 0:
                raise RuntimeError(f"free row {i} has no mode_span_end")
            tok = torch.where(pos[0] < e, a_m, a_in)         # (L,)
            adv[i] = tok * response_mask[i]
            a_mode_rows.append(float(a_m)); a_inner_rows.append(float(a_in))
            if m in mode_count:
                mode_count[m] += 1
    n_free_rows = len(a_mode_rows)
    stats = {
        "free/groups": n_free_groups, "forced/groups": n_forced_groups,
        "worker/groups_split": n_worker_groups,
        "worker/rows": len(a_worker_rows),
        "worker/unique_workers_per_group": (sum(worker_uniq) / len(worker_uniq)) if worker_uniq else 0.0,
        "worker/rollouts_per_worker": (sum(w_group_sizes) / len(w_group_sizes)) if w_group_sizes else 0.0,
        "worker/A_worker_abs_mean": float(np.mean(np.abs(a_worker_rows))) if a_worker_rows else 0.0,
        "worker/A_worker_nonzero_rate": (float(np.mean([abs(x) > 1e-8 for x in a_worker_rows])) if a_worker_rows else 0.0),
        "worker/A_inner_abs_mean": float(np.mean(np.abs(a_winner_rows))) if a_winner_rows else 0.0,
        "free/groups_total": n_free_total, "free/groups_zero_variance": n_free_total - n_free_groups,
        "hier/groups_gated": gated, "hier/rows_dropped_dispatch": dropped_rows,
        "free/rows": n_free_rows,
        "free/unique_modes_per_prompt": (sum(uniq) / len(uniq)) if uniq else 0.0,
        "free/singleton_mode_rate": (single_rows / n_free_rows) if n_free_rows else 0.0,
        "free/A_mode_mean": float(np.mean(a_mode_rows)) if a_mode_rows else 0.0,
        "free/A_mode_abs_mean": float(np.mean(np.abs(a_mode_rows))) if a_mode_rows else 0.0,
        "free/A_inner_mean": float(np.mean(a_inner_rows)) if a_inner_rows else 0.0,
        "free/A_inner_abs_mean": float(np.mean(np.abs(a_inner_rows))) if a_inner_rows else 0.0,
        "free/A_inner_nonzero_rate": (float(np.mean([abs(x) > 1e-8 for x in a_inner_rows])) if a_inner_rows else 0.0),
        "free/A_mode_nonzero_rate": (float(np.mean([abs(x) > 1e-8 for x in a_mode_rows])) if a_mode_rows else 0.0),
        "free/identity_max_err": ident_err,
    }
    for m in MODES:
        stats[f"free/mode_count_{m}"] = mode_count[m]
    # forced groups must equal the stock estimator when nothing was dropped
    if n_forced_groups and (clean is None or all(clean)):
        stock, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, response_mask=response_mask,
            index=index, epsilon=epsilon, norm_adv_by_std_in_grpo=True, config=config)
        forced_rows = [i for i in range(bsz) if int(free[i]) == 0]
        err = 0.0
        for i in forced_rows:
            # gated groups are 0 here and ±x in stock only if std < 1e-4 (empty band on {-1,0,1})
            err = max(err, float((adv[i] - stock[i]).abs().max()))
        stats["hier/forced_vs_stock_max_err"] = err
    LAST_STATS.clear(); LAST_STATS.update(stats)
    print(f"[comet_grpo] {json.dumps(stats)}", flush=True)
    return adv, adv


def publish_mode_token_mask(data, info):
    """Per-token 1/0 mask marking the MODE-SELECTION tokens of each row, written
    as a batch tensor so the actor's loss can reduce the two segments
    separately (segment_loss.py). Free rows: [0, mode_span_end) intersected with
    response_mask. Forced rows: all zero -- their declaration is environment
    text (response_mask 0) and their whole response is "inner", i.e. plain GRPO.
    Inert for the loss unless ROUTEWEAVER_LOSS_REDUCTION=segment; costs one bool
    tensor either way, and makes the segmentation auditable in the dumps.
    """
    rm = data.batch["response_mask"]
    pos = torch.arange(rm.shape[1], device=rm.device).unsqueeze(0)
    ends = torch.tensor([int(e) if int(f) == 1 else 0
                         for e, f in zip(info["span_end"], info["free"])],
                        device=rm.device).unsqueeze(1)
    data.batch[MODE_TOKEN_MASK_KEY] = ((pos < ends) & (rm > 0)).to(rm.dtype)
    # THIRD segment: the worker-name tokens of every route the policy wrote.
    # Disjoint from the mode mask by construction (a route is always after the
    # declaration) and intersected with response_mask so environment text can
    # never enter it.
    #
    # FIDELITY GATE. `worker_spans` is written by the agent loop
    # UNCONDITIONALLY (unified_loop:389), so on this tree the column
    # exists even with WORKER_EXPLORE=0 -- and with
    # ROUTEWEAVER_LOSS_REDUCTION=segment that silently gives the worker-name tokens
    # a third segment with its own denominator and a per-row 1/n_i scale,
    # a small share of policy tokens. The reported configuration has no such
    # segment, so this mask stays empty unless worker exploration is actually
    # on -- which is what builds the spans in the first place.
    wm = torch.zeros_like(rm)
    if os.environ.get("WORKER_EXPLORE", "0") == "1":
        for i, spans in enumerate(info.get("wspans") or []):
            for a, b, _ in spans:
                if 0 <= a < b <= wm.shape[1]:
                    wm[i, a:b] = 1
    data.batch[WORKER_TOKEN_MASK_KEY] = ((wm > 0) & (rm > 0)).to(rm.dtype)
    clean = info.get("clean")
    cr = (torch.ones(rm.shape[0], device=rm.device) if clean is None
          else torch.tensor([1.0 if c else 0.0 for c in clean], device=rm.device))
    data.batch[CLEAN_ROW_KEY] = cr.to(rm.dtype).unsqueeze(1)


def _info_from_batch(data):
    cols = data.non_tensor_batch
    n = len(data.batch)
    forced = cols.get("stage2_forced")
    if forced is None:
        raise RuntimeError("comet_grpo needs the stage2_forced column")
    mode = cols.get("mode")
    span_end = cols.get("mode_span_end")
    if mode is None or span_end is None:
        raise RuntimeError("comet_grpo needs the mode and mode_span_end columns (reward_extra_info)")
    fc = cols.get("failed_worker_calls")
    inf = cols.get("infra_failed")
    if fc is not None:
        clean = [int(f) == 0 and int(i or 0) == 0
                 for f, i in zip(fc, inf if inf is not None else [0] * len(fc))]
    else:
        clean = None
    fw = cols.get("forced_worker")
    ws = cols.get("worker_spans")
    return {"free": [1 - int(v) for v in forced], "mode": [str(v) for v in mode],
            "span_end": [int(v) for v in span_end],
            # both absent on a run without the worker-exploration curriculum,
            # in which case every forced group takes the unchanged GRPO path
            "clean": clean,
            "worker": ["" for _ in forced] if fw is None else [str(v) for v in fw],
            "wspans": [[] for _ in forced] if ws is None else [parse_worker_spans(v) for v in ws]}


def install(ray_trainer_module) -> bool:
    import functools
    if not enabled():
        return False
    original = getattr(ray_trainer_module, "compute_advantage", None)
    if original is None or getattr(original, "_routeweaver_hier", False):
        return False

    @functools.wraps(original)
    def compute_advantage(data, *args, **kwargs):
        info = _info_from_batch(data)
        stash_info(info)
        try:
            out = original(data, *args, **kwargs)
        finally:
            stash_info(None)
        publish_mode_token_mask(out, info)
        return out

    compute_advantage._routeweaver_hier = True
    compute_advantage._router_r1_dispatch_gate = True
    ray_trainer_module.compute_advantage = compute_advantage
    print("[comet_grpo] compute_advantage wrapped (row info for comet_grpo)", flush=True)
    return True
