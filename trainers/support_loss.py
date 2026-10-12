# Copyright 2026 RouteWeaver.
"""Minimum-support auxiliary loss for the mode selector."""
from typing import Sequence

import torch

MODES = ("single", "multi", "agentic")
SUPPORT_FLOOR = 0.05
SUPPORT_LAMBDA0 = 0.05
EPS = 1e-12


def lambda_for(alpha, lambda0=SUPPORT_LAMBDA0):
    """lambda_t = lambda0 * alpha_t."""
    return float(lambda0) * max(0.0, float(alpha))


def candidate_logprobs(logits, cand_ids, cand_lens):
    """Summed logprob of each candidate declaration."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp[:, :-1].gather(-1, cand_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    out = []
    for i, n in enumerate(cand_lens):
        n = int(n)
        out.append(tok_lp[i, tok_lp.shape[1] - n:].sum())
    return torch.stack(out)


def mode_marginal(scores):
    """scores: (B, 3) summed candidate logprobs -> (3,) batch marginal."""
    per_query = torch.softmax(scores, dim=-1)
    return per_query.mean(dim=0), per_query


def support_loss(scores, floor=SUPPORT_FLOOR, eps=EPS):
    """-> (loss, pibar, per_mode_penalty). scores: (B, 3), B = unique queries."""
    pibar, _ = mode_marginal(scores)
    log_floor = torch.log(torch.as_tensor(float(floor), dtype=pibar.dtype,
                                          device=pibar.device))
    penalty = torch.clamp(log_floor - torch.log(pibar + eps), min=0.0)
    return penalty.mean(), pibar, penalty
