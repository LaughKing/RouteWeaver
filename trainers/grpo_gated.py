# Copyright 2026 RouteWeaver.
"""GRPO with a zero-variance group gate, as a registered verl estimator."""
from collections import defaultdict

import torch

from verl.trainer.ppo import core_algos

GROUP_STD_THRESHOLD = 1e-4

_PENDING_CLEAN = {"mask": None}


def stash_clean_mask(mask):
    """Hand the next estimator call its per-row clean flags (or None)."""
    _PENDING_CLEAN["mask"] = mask


def _take_clean_mask(bsz: int):
    """Consume the stashed mask. None means 'assume every row is clean'."""
    mask = _PENDING_CLEAN["mask"]
    _PENDING_CLEAN["mask"] = None
    if mask is None:
        return None
    if len(mask) != bsz:
        print(f"[grpo_gated] clean mask has {len(mask)} rows for a batch "
              f"of {bsz}; skipping the dispatch gate for this step")
        return None
    return mask


def _contaminated_advantage(scores, response_mask, index, epsilon, norm_adv, clean):
    """GRPO with the group baseline taken over the CLEAN members only."""
    adv = torch.zeros_like(scores)
    groups = defaultdict(list)
    for i in range(scores.shape[0]):
        groups[index[i]].append(i)

    dropped_rows = dropped_groups = 0
    for members in groups.values():
        survivors = [i for i in members if clean[i]]
        dropped_rows += len(members) - len(survivors)
        if not survivors:
            dropped_groups += 1
            continue
        if len(survivors) == 1:
            mean = torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
            std = torch.tensor(1.0, dtype=scores.dtype, device=scores.device)
        else:
            vals = torch.stack([scores[i] for i in survivors])
            mean, std = torch.mean(vals), torch.std(vals)
            if std < GROUP_STD_THRESHOLD:
                dropped_groups += 1
                continue  # zero-variance gate, on the clean members
        for i in survivors:
            adv[i] = ((scores[i] - mean) / (std + epsilon)) if norm_adv else (scores[i] - mean)

    print(f"[grpo_gated] dispatch gate: {dropped_rows}/{scores.shape[0]} rows "
          f"excluded (failed worker call), {dropped_groups} group(s) left "
          f"without a usable baseline")
    return adv.unsqueeze(-1) * response_mask


@core_algos.register_adv_est("grpo_gated")
def compute_grpo_gated_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """Stock GRPO, minus zero-variance groups, minus dispatch-contaminated rows."""
    norm_adv = True if config is None else getattr(config, "norm_adv_by_std_in_grpo", True)
    clean = _take_clean_mask(token_level_rewards.shape[0])

    if clean is not None and not all(clean):
        with torch.no_grad():
            scores = token_level_rewards.sum(dim=-1)
            advantages = _contaminated_advantage(
                scores, response_mask, index, epsilon, norm_adv, clean)
        return advantages, advantages

    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv,
        config=config,
    )

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        groups = defaultdict(list)
        for i in range(scores.shape[0]):
            groups[index[i]].append(scores[i])
        gated = {
            idx: (len(members) > 1
                  and torch.std(torch.stack(members)) < GROUP_STD_THRESHOLD)
            for idx, members in groups.items()
        }
        keep = torch.tensor([not gated[index[i]] for i in range(scores.shape[0])],
                            dtype=advantages.dtype, device=advantages.device)
        advantages = advantages * keep.unsqueeze(-1)
        returns = returns * keep.unsqueeze(-1)

    return advantages, returns


def install_compute_advantage_gate(ray_trainer_module) -> bool:
    """Wrap ray_trainer.compute_advantage so the estimator sees the flags."""
    import functools

    original = getattr(ray_trainer_module, "compute_advantage", None)
    if original is None or getattr(original, "_router_r1_dispatch_gate", False):
        return False

    @functools.wraps(original)
    def compute_advantage(data, *args, **kwargs):
        stash_clean_mask(_clean_mask_from_batch(data))
        try:
            return original(data, *args, **kwargs)
        finally:
            stash_clean_mask(None)

    compute_advantage._router_r1_dispatch_gate = True
    ray_trainer_module.compute_advantage = compute_advantage
    return True


def _clean_mask_from_batch(data):
    """Per-row 'no failed worker call' flags, or None if the column is absent."""
    columns = getattr(data, "non_tensor_batch", {})
    failed_calls = columns.get("failed_worker_calls")
    infra = columns.get("infra_failed")
    if failed_calls is not None:
        try:
            fc = [int(v) for v in failed_calls]
            inf = [int(v) for v in infra] if infra is not None else [0] * len(fc)
            return [(f == 0 and i == 0) for f, i in zip(fc, inf)]
        except (TypeError, ValueError):
            pass
    for name in ("infra_failed", "dispatch_failed_calls"):
        failed = columns.get(name)
        if failed is None:
            continue
        try:
            return [int(v) == 0 for v in failed]
        except (TypeError, ValueError):
            print(f"[grpo_gated] {name} is not numeric; "
                  "skipping the dispatch gate for this step")
            return None
    return None
