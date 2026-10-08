# Copyright 2026 RouteWeaver.
"""GRPO with a zero-variance group gate, as a registered verl estimator.

THE GATE (GROUP_STD_EPSILON): a prompt group whose
reward std is below 1e-4 carries no learning signal, and every rollout in it
gets advantage exactly 0. verl 0.8's stock ``grpo`` instead normalizes by
``(std + 1e-6)``:

  * std == 0 exactly -> (r - mean) is also exactly 0, so the stock estimator
    already yields 0. The gate changes nothing there.
  * 0 < std < 1e-4  -> the stock estimator AMPLIFIES the tiny residual to an
    O(1) advantage (diff/std ~= 1); the gate zeroes it. This is the whole
    behavioral difference, and with the current discrete reward set
    ({-1, 0, 1}: em-quality + strict-format, cost_lambda=0) that band is
    normally empty -- the gate is then numerically a no-op. It stops being a
    no-op the moment rewards go continuous (f1 metric, cost_lambda > 0), which
    is exactly when amplified near-tie noise would hurt most.

Groups with std >= 1e-4 keep verl's normalization bit for bit: the gate calls
the stock ``compute_grpo_outcome_advantage`` and only masks gated rows after.

REGISTRATION: importing this module registers ``grpo_gated`` in verl's
advantage-estimator registry. The import happens automatically in every
process that imports ``verl.trainer.ppo.core_algos`` (see sitecustomize.py's
post-import hook) -- in particular the trainer driver, which is where
``compute_advantage`` resolves ``algorithm.adv_estimator`` by name. The
launcher selects it via ``ADV_ESTIMATOR`` (default grpo_gated).

Std convention: same tensor, same estimator as the stock GRPO denominator --
``torch.std`` (Bessel, n-1) over the group's sequence-level reward sums; a
size-1 group keeps verl's std=1.0 convention and is never gated.

THE DISPATCH GATE. A rollout whose trajectory contains a failed
worker call read ``<observation></observation>`` and then answered anyway. It
still scores: 27.7% of stage 2's samples carried at least one such call, and
every one of them fed the gradient at full weight. That teaches the policy to
answer from an empty observation, which is the one thing the environment must
never reward.

Penalising them instead would be worse, not better: the router picked a legal
worker and the provider refused. A negative reward there teaches "do not route
to this worker", a lesson about a provider outage that has nothing to do with
routing quality. The only honest treatment is EXCLUSION.

Exclusion has two halves, and doing only the first is a trap:

  1. the contaminated row's own advantage goes to 0, so it contributes no
     policy gradient;
  2. its reward also leaves the GROUP BASELINE. GRPO scores a rollout against
     its group's mean, so a contaminated sibling -- which usually answered
     wrong, having been handed nothing -- drags the mean down and inflates
     every clean sibling's advantage. Zeroing row 1 alone leaves that bias
     fully intact and silently rewards clean rollouts for being lucky.

So the baseline is recomputed over the clean members only. A group with fewer
than two clean members has no within-group signal left and is zeroed whole.

Bit-for-bit safety: when a batch carries no contamination at all the gate is
not merely equivalent to the old path, it IS the old path -- the branch calls
verl's ``compute_grpo_outcome_advantage`` exactly as before. Only a batch that
actually contains a failed dispatch takes the new arithmetic.

SIGNAL PATH: ``dispatch_failed_calls`` is a per-row reward_extra_info column
(reward_manager.py). verl unions reward_extra_info into non_tensor_batch
(ray_trainer.py:1594) BEFORE calling compute_advantage (ray_trainer.py:1625),
so the flag is already on the batch when advantages are computed. verl's
registry branch does not forward non_tensor_batch to a registered estimator
(only GDPO gets it, ray_trainer.py:258-260), so sitecustomize wraps
``ray_trainer.compute_advantage`` to hand the column over. The wrapper stashes
it for exactly one call and clears it in a finally -- a stale mask applied to
the next batch would silently zero the wrong rows.
"""
from collections import defaultdict

import torch

from verl.trainer.ppo import core_algos

# Strict `<`: a group at exactly the threshold keeps its gradient.
GROUP_STD_THRESHOLD = 1e-4

# Set by the compute_advantage wrapper, consumed by the estimator, cleared by
# the wrapper's finally. Never read twice: a mask outliving its batch would be
# applied to rows it does not describe.
_PENDING_CLEAN = {"mask": None}


def stash_clean_mask(mask):
    """Hand the next estimator call its per-row clean flags (or None)."""
    _PENDING_CLEAN["mask"] = mask


def _take_clean_mask(bsz: int):
    """Consume the stashed mask. None means 'assume every row is clean'.

    A length mismatch is treated as no mask rather than as an error: the gate
    exists to protect the gradient, and a gate that can abort training is a
    worse failure than a batch that goes ungated. It is loud about it.
    """
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
    """GRPO with the group baseline taken over the CLEAN members only.

    Reproduces verl's arithmetic on the surviving members -- same Bessel std,
    same ``+ epsilon`` denominator, same size-1 (mean 0, std 1) convention --
    and returns 0 for every row that is contaminated or whose group has too
    few clean members to form a baseline.
    """
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
            # verl's convention for a group of one: baseline 0, std 1. Kept so
            # a group that contamination happens to reduce to one row behaves
            # exactly like one that was born that way.
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

    # No contamination in this batch: the original path, unchanged, so a clean
    # step is bit-for-bit what it was before the gate existed.
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
    """Wrap ray_trainer.compute_advantage so the estimator sees the flags.

    ray_trainer.fit() resolves ``compute_advantage`` as a module global at call
    time, so replacing the attribute is enough -- no verl source is touched.
    Idempotent, and a no-op if the attribute is missing.
    """
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
            # Whatever happened, the mask must not survive into another batch.
            stash_clean_mask(None)

    compute_advantage._router_r1_dispatch_gate = True
    ray_trainer_module.compute_advantage = compute_advantage
    return True


def _clean_mask_from_batch(data):
    """Per-row 'no failed worker call' flags, or None if the column is absent.

    Absent is normal, not broken: validation batches and any run whose reward
    manager predates the column simply go ungated.

    ``infra_failed`` is preferred when present. It is the reward manager's own
    verdict on the same question and carries the neutralised reward with it, so
    the row the gate drops is exactly the row that was not scored. The older
    ``dispatch_failed_calls`` count is the fallback, which keeps checkpoints and
    dumps from before gating identically.
    """
    columns = getattr(data, "non_tensor_batch", {})
    #, WIDENED. `infra_failed` is the reward manager's verdict and it
    # is NARROWER than "a worker call failed": measured on the 4x2 dry-run, 26
    # rows had failed_worker_calls > 0 and only 7 carried infra_failed=1. The
    # other 19 went out as `policy_invalid` / `correct` / `genuine_incorrect`,
    # because the empty observation made the POLICY produce bad grammar and the
    # row was then judged on that. Those rows entered the gradient at full
    # weight with a strongly negative mean score, concentrated on whichever
    # worker the provider happened to be rate-limiting -- i.e. a provider 429
    # was being written into that worker's name tokens as evidence it is bad.
    #
    # A failed worker call is an EXOGENOUS environment failure. Any row that
    # contains one is excluded for every worker equally: it leaves mu_G, sigma,
    # mu_w, A_worker and A_inner, and its advantage is zero on every token, so
    # its contribution to the policy gradient is exactly 0.
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
