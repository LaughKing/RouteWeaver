# Copyright 2026 RouteWeaver.
"""Minimum-support auxiliary loss for the mode selector.

THE PROBLEM. After Stage 1, pi(multi) at the mode decision is ~3e-6. GRPO only
learns from what it samples, so multi is never sampled, never gets an advantage,
and never recovers. Raising the sampling temperature does not fix it (measured:
T=4 buys 1.4%, and temperature never moves the argmax), and correcting an
epsilon-mixture behaviour policy properly attenuates the very gradient that was
supposed to help.

THE TERM. One-sided hinge in LOG space on the BATCH MARGINAL:

    pi_M(m|x) = softmax_over_the_three( s_m )      s_m = log P(<mode>m</mode>|x)
    pibar(m)  = (1/B) sum_x pi_M(m|x)              B = UNIQUE queries, not K*B
    L_support = (1/3) sum_m max(0, log(delta) - log(pibar(m) + eps))

A mode at or above `delta` contributes exactly zero -- this is a floor, not a
balance constraint, and it never pushes toward 1/3.

WHY LOG SPACE AND NOT (delta - p)^2. At p ~ 3e-6 the probability-space hinge has
gradient ~delta, and the softmax Jacobian factor pi*(1-pi) ~ 3e-6 on top of it,
so the update is ~1e-7 -- indistinguishable from zero. In log space
dL/d log pibar = -1 regardless of how small pibar is, so the pull is constant and
the collapse is actually recoverable.

WHY THE FULL DECLARATION. `agentic` tokenises as "ag"+"entic" while single and
multi are one token each. Scoring only the first token would give agentic a
different effective action length from the other two and bias the marginal.

The whole term reads only mode candidate probabilities. It does not touch the
rollout distribution, the advantage, the PPO ratio, or the forced-mode mask.
"""
from typing import Sequence

import torch

MODES = ("single", "multi", "agentic")
SUPPORT_FLOOR = 0.05
SUPPORT_LAMBDA0 = 0.05
EPS = 1e-12


def lambda_for(alpha, lambda0=SUPPORT_LAMBDA0):
    """lambda_t = lambda0 * alpha_t -- the term retires exactly when forcing does,
    so Stage 2 ends at 100% free rollouts AND zero auxiliary loss and Stage 3
    inherits no distribution jump."""
    return float(lambda0) * max(0.0, float(alpha))


def candidate_logprobs(logits, cand_ids, cand_lens):
    """Summed logprob of each candidate declaration.

    logits:    (N, L, V) for sequences laid out as [prompt | declaration]
    cand_ids:  (N, L) the input ids
    cand_lens: (N,) how many trailing tokens belong to the declaration

    Returns (N,) -- sum over the WHOLE declaration, which is what makes the
    three candidates comparable despite different token counts.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    # logits at position j predict token j+1
    tok_lp = logp[:, :-1].gather(-1, cand_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    out = []
    for i, n in enumerate(cand_lens):
        n = int(n)
        out.append(tok_lp[i, tok_lp.shape[1] - n:].sum())
    return torch.stack(out)


def mode_marginal(scores):
    """scores: (B, 3) summed candidate logprobs -> (3,) batch marginal.

    Softmax is per QUERY and the mean is over queries: mean_x softmax(s_x), not
    softmax(mean_x s_x). The two differ a lot here because the per-query
    distribution is close to a point mass after the cold start, and only the
    first is the quantity the floor is about.
    """
    per_query = torch.softmax(scores, dim=-1)
    return per_query.mean(dim=0), per_query


def support_loss(scores, floor=SUPPORT_FLOOR, eps=EPS):
    """-> (loss, pibar, per_mode_penalty). scores: (B, 3), B = unique queries."""
    pibar, _ = mode_marginal(scores)
    log_floor = torch.log(torch.as_tensor(float(floor), dtype=pibar.dtype,
                                          device=pibar.device))
    penalty = torch.clamp(log_floor - torch.log(pibar + eps), min=0.0)
    return penalty.mean(), pibar, penalty
