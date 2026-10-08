# Copyright 2026 RouteWeaver.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared turn machinery for the RouteWeaver agent loop.

`UnifiedAgentLoop` (unified_loop.py) drives the episode. What both sides of a
turn have in common lives here:

- ONE generate() call per turn, stopped at the first action boundary and cut in
  TOKEN space -- ids and logprobs are cut at the same index and never
  re-tokenized, because any drift there misaligns the whole trajectory;
- the observation budget: how much of a worker reply a turn may inject,
  computed from what is left of the response window rather than from a fixed
  per-turn allowance;
- the injection itself: observation tokens enter the next turn's context with
  response_mask 0 and logprob 0, so the environment's text is never trained on.

THE ENVIRONMENT IS THE ONLY SOURCE OF AN OBSERVATION. Nothing stops the router
from writing one itself, and when a layer failed validation it was not
dispatched, so the policy saw its own <layer> with nothing after it and simply
wrote the reply it expected. The fabricated text never reached a worker --
refs resolve from dispatch results, not from the transcript -- but it did
corrupt the trajectory the reward reads. So the opening tag is a hard
generation boundary: the router is cut off the moment it writes one.
"""

import logging
import os
import re
from typing import Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopBase

from route_grammar import block_pattern, close_pattern
from obs_utils import truncate_observation_ids

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ANSWER_BLOCK = block_pattern("answer")
_CLOSE_ROUTE = close_pattern("route")
_CLOSE_LAYER = close_pattern("layer")
_CLOSE_ANSWER = close_pattern("answer")

_OBSERVATION_OPEN = "<observation"
_OPEN_OBSERVATION = re.compile(r"<observation\b", re.IGNORECASE)

# Sampling stop strings: one action per turn. single/multi stop at a route;
# agentic stops at a whole LAYER, because a layer legitimately contains several
# routes and stopping at the first </route> would amputate it.
_TURN_STOPS = ["</route>", "</answer>", _OBSERVATION_OPEN]
_MODE_STOPS = ["</mode>", "</answer>", _OBSERVATION_OPEN]

_CLOSE_PATTERNS = {
    "</route>": _CLOSE_ROUTE,
    "</layer>": _CLOSE_LAYER,
    "</answer>": _CLOSE_ANSWER,
    "</mode>": close_pattern("mode"),
    # not a closing tag, but the same thing operationally: the point at which
    # this turn's generation must end.
    _OBSERVATION_OPEN: _OPEN_OBSERVATION,
}


def _has_turn_boundary(text: str, stops) -> bool:
    """True once the text contains a closing tag for any active stop string."""
    return any(_CLOSE_PATTERNS[stop].search(text) is not None for stop in stops)


class RoutingAgentLoopBase(AgentLoopBase):
    """Turn generation, observation budget and token bookkeeping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        router_cfg = self.config.get("router", {}) or {}
        self.max_obs_length = router_cfg.get("max_obs_length", 512)
        # DYNAMIC OBSERVATION BUDGET. max_obs_length alone is a fixed per-turn
        # allowance, and a fixed allowance throws away worker text while the
        # response window is nowhere near full -- the trajectory had room the
        # observation was not allowed to use. The budget is therefore what is
        # actually left of the response window, capped so one observation
        # cannot eat the whole trajectory:
        #
        #     turn_budget = clamp(response_length - used - reserve, 0, cap)
        #
        # `reserve` keeps room for the router's own remaining turns; without it
        # a long early observation would leave nothing to answer with.
        self.obs_budget_cap = int(router_cfg.get(
            "obs_budget_cap", os.environ.get("ROUTEWEAVER_OBS_BUDGET_CAP", 4096)))
        self.obs_budget_reserve = int(router_cfg.get(
            "obs_budget_reserve", os.environ.get("ROUTEWEAVER_OBS_RESERVE", 1024)))
        # floor for ONE observation once a wide layer splits the turn budget;
        # below this an observation is pure wrapper and carries no information
        self.obs_min_per_obs = int(router_cfg.get(
            "obs_min_per_obs", os.environ.get("ROUTEWEAVER_OBS_MIN_PER_OBS", 256)))
        # share of a truncated body given to the TAIL (see obs_utils._head_tail)
        self.obs_tail_fraction = float(router_cfg.get(
            "obs_tail_fraction", os.environ.get("ROUTEWEAVER_OBS_TAIL_FRACTION", 0.5)))
        # set False to restore the fixed max_obs_length allowance exactly
        self.obs_budget_dynamic = str(router_cfg.get(
            "obs_budget_dynamic",
            os.environ.get("ROUTEWEAVER_OBS_BUDGET_DYNAMIC", "1"))).lower() not in (
                "0", "false", "no")
        self.response_length = self.rollout_config.response_length
        self.channel_config_path = router_cfg.get("channel_config_path", None)
        self.dispatch_concurrency = router_cfg.get("dispatch_concurrency", 10)

    def _observation_ids(self, obs_text: str, budget: Optional[int] = None,
                         stats: Optional[dict] = None) -> list[int]:
        """Tokenize one observation to fit ``budget``.

        ``stats``, when given, is a counter dict the caller owns; this method
        adds to it rather than reading it, so a caller that injects several
        observations in one turn gets the turn's totals for free. Without them
        truncation is visible only as a log line, and measuring it after the
        fact means re-tokenizing a finished dump.
        """
        budget = self.max_obs_length if budget is None else budget
        obs_ids, was_truncated, dropped = truncate_observation_ids(
            self.tokenizer, obs_text, budget, self.obs_tail_fraction
        )
        if stats is not None:
            stats["blocks"] = stats.get("blocks", 0) + 1
            stats["truncated"] = stats.get("truncated", 0) + int(was_truncated)
            stats["dropped_tokens"] = stats.get("dropped_tokens", 0) + dropped
            stats["budget"] = budget
        if was_truncated:
            logger.warning(
                "observation exceeded its budget=%s; %d body tokens elided, "
                "head and tail kept, tags preserved", budget, dropped)
        return obs_ids

    def _turn_observation_budget(self, used_response_tokens: int) -> int:
        """Tokens this TURN's observations may occupy, given what is left.

        Returns 0 when the reserve is already gone -- the caller then injects
        nothing and the run loop's own length check ends the episode, which is
        the honest outcome: there is no room for another worker reply.
        """
        if not self.obs_budget_dynamic:
            return self.max_obs_length
        room = self.response_length - used_response_tokens - self.obs_budget_reserve
        return max(0, min(self.obs_budget_cap, room))

    def _layer_observation_budget(self, route_count: int,
                                  turn_budget: Optional[int] = None) -> int:
        """Split the TURN's observation budget across the layer's routes.

        max_obs_length is a per-TURN allowance, not a per-observation one. A
        layer injects every route's reply in a single turn, so a per-observation
        cap would let a 3-route layer put 3 x max_obs_length environment tokens
        into one turn -- with the default 512 that is 1536 tokens against a
        2048-token response budget, and the router's own text gets squeezed out
        of the trajectory it is supposed to be learning.

        single/multi inject exactly one observation per turn, so they are
        unaffected: the share equals max_obs_length.
        """
        turn_budget = self.max_obs_length if turn_budget is None else turn_budget
        if route_count <= 1:
            return turn_budget
        # The floor can make route_count x share exceed the turn budget, by at
        # most (route_count x obs_min_per_obs). That is deliberate and bounded:
        # obs_budget_reserve is larger than the overrun a legal layer can
        # produce, and the run loop's response_length check still hard-stops
        # the episode. An observation cut below the floor is all wrapper.
        share = max(self.obs_min_per_obs, turn_budget // route_count)
        logger.info("layer of %d routes: %d observation tokens each (turn budget %d)",
                    route_count, share, turn_budget)
        return share




    # ------------------------------------------------------------------ turns --


    async def _generate_turn(
        self, request_id, context, sampling_params, stops,
        response_ids, response_mask, response_logprobs, logprobs_seen,
    ):
        """One GENERATE round: server call, token-space boundary cut, state append."""
        per_turn = dict(sampling_params)
        per_turn["stop"] = list(stops)
        per_turn["include_stop_str_in_output"] = True

        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=context,
            sampling_params=per_turn,
        )
        token_ids = list(output.token_ids)
        log_probs = list(output.log_probs) if output.log_probs else None

        token_ids, log_probs, text = self._cut_at_turn_boundary(token_ids, log_probs, stops)

        context += token_ids
        response_ids += token_ids
        response_mask += [1] * len(token_ids)
        if log_probs is not None:
            if not logprobs_seen:
                # First segment that reports logprobs. If earlier segments did
                # not, response_logprobs is short by exactly the tokens they
                # contributed; backfill them or the three lists index-shift and
                # the alignment assert in run() takes the whole rollout down.
                response_logprobs += [0.0] * (
                    len(response_ids) - len(token_ids) - len(response_logprobs))
            logprobs_seen = True
            response_logprobs += log_probs
        elif logprobs_seen:
            response_logprobs += [0.0] * len(token_ids)
        return text, logprobs_seen


    def _cut_at_turn_boundary(self, token_ids, log_probs, stops):
        """Defensive cut after the first closing tag among `stops` — TOKEN-SPACE.

        Cutting in text space and re-tokenizing desyncs tokens from logprobs,
        which nothing downstream can recover from. Instead: binary
        search the smallest prefix of TOKENS whose decode contains a closing
        tag, then cut ids and logprobs at the same index. With stop strings the
        engine already stops at the boundary, so this only fires on servers
        (or mocks) that return trailing content.

        Known difference: if one token fuses the closing tag with trailing
        characters, those characters survive the cut, where a character-space
        cut would have removed them. No re-tokenization is the deliberate
        trade.
        """
        text = self.tokenizer.decode(token_ids)
        if not _has_turn_boundary(text, stops):
            return token_ids, log_probs, text
        lo, hi = 1, len(token_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if _has_turn_boundary(self.tokenizer.decode(token_ids[:mid]), stops):
                hi = mid
            else:
                lo = mid + 1
        cut_ids = token_ids[:lo]
        cut_probs = log_probs[:lo] if log_probs is not None else None
        return cut_ids, cut_probs, self.tokenizer.decode(cut_ids)

    def _inject_observation_ids(
        self, obs_ids, context, response_ids, response_mask,
        response_logprobs, logprobs_seen,
    ) -> None:
        """Append environment tokens with mask 0 and logprob 0. The same tokens
        enter the next turn's context."""
        context += obs_ids
        response_ids += obs_ids
        response_mask += [0] * len(obs_ids)
        if logprobs_seen:
            response_logprobs += [0.0] * len(obs_ids)

    # ---------------------------------------------------------------- grammar --



