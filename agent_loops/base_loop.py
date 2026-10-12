# Copyright 2026 RouteWeaver.
"""Shared turn machinery for the RouteWeaver agent loop."""

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

_TURN_STOPS = ["</route>", "</answer>", _OBSERVATION_OPEN]
_MODE_STOPS = ["</mode>", "</answer>", _OBSERVATION_OPEN]

_CLOSE_PATTERNS = {
    "</route>": _CLOSE_ROUTE,
    "</layer>": _CLOSE_LAYER,
    "</answer>": _CLOSE_ANSWER,
    "</mode>": close_pattern("mode"),
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
        self.obs_budget_cap = int(router_cfg.get(
            "obs_budget_cap", os.environ.get("ROUTEWEAVER_OBS_BUDGET_CAP", 4096)))
        self.obs_budget_reserve = int(router_cfg.get(
            "obs_budget_reserve", os.environ.get("ROUTEWEAVER_OBS_RESERVE", 1024)))
        self.obs_min_per_obs = int(router_cfg.get(
            "obs_min_per_obs", os.environ.get("ROUTEWEAVER_OBS_MIN_PER_OBS", 256)))
        self.obs_tail_fraction = float(router_cfg.get(
            "obs_tail_fraction", os.environ.get("ROUTEWEAVER_OBS_TAIL_FRACTION", 0.5)))
        self.obs_budget_dynamic = str(router_cfg.get(
            "obs_budget_dynamic",
            os.environ.get("ROUTEWEAVER_OBS_BUDGET_DYNAMIC", "1"))).lower() not in (
                "0", "false", "no")
        self.response_length = self.rollout_config.response_length
        self.channel_config_path = router_cfg.get("channel_config_path", None)
        self.dispatch_concurrency = router_cfg.get("dispatch_concurrency", 10)

    def _observation_ids(self, obs_text: str, budget: Optional[int] = None,
                         stats: Optional[dict] = None) -> list[int]:
        """Tokenize one observation to fit ``budget``."""
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
        """Tokens this TURN's observations may occupy, given what is left."""
        if not self.obs_budget_dynamic:
            return self.max_obs_length
        room = self.response_length - used_response_tokens - self.obs_budget_reserve
        return max(0, min(self.obs_budget_cap, room))

    def _layer_observation_budget(self, route_count: int,
                                  turn_budget: Optional[int] = None) -> int:
        """Split the TURN's observation budget across the layer's routes."""
        turn_budget = self.max_obs_length if turn_budget is None else turn_budget
        if route_count <= 1:
            return turn_budget
        share = max(self.obs_min_per_obs, turn_budget // route_count)
        logger.info("layer of %d routes: %d observation tokens each (turn budget %d)",
                    route_count, share, turn_budget)
        return share


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
                response_logprobs += [0.0] * (
                    len(response_ids) - len(token_ids) - len(response_logprobs))
            logprobs_seen = True
            response_logprobs += log_probs
        elif logprobs_seen:
            response_logprobs += [0.0] * len(token_ids)
        return text, logprobs_seen


    def _cut_at_turn_boundary(self, token_ids, log_probs, stops):
        """Defensive cut after the first closing tag among `stops` — TOKEN-SPACE."""
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
        """Append environment tokens with mask 0 and logprob 0."""
        context += obs_ids
        response_ids += obs_ids
        response_mask += [0] * len(obs_ids)
        if logprobs_seen:
            response_logprobs += [0.0] * len(obs_ids)


