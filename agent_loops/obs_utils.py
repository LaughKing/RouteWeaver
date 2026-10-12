# Copyright 2026 RouteWeaver.
"""Observation helpers — copies pinned to the original semantics."""

import re
from typing import List, Tuple

_OBS_WRAPPER = re.compile(
    r'^(\s*<(observation|information)\b[^>]*>)(.*)(</\2\s*>\s*)$', re.DOTALL)


_ELISION = "\n\n... [observation truncated] ...\n\n"


def truncate_observation_ids(tokenizer, text: str, budget: int,
                             tail_fraction: float = 0.5
                             ) -> Tuple[List[int], bool, int]:
    """Tokenize ONE observation, cutting the BODY to fit ``budget`` while
    keeping the wrapper tags intact.
    """
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if len(ids) <= budget:
        return ids, False, 0

    match = _OBS_WRAPPER.match(text)
    if match is None:
        return _head_tail(tokenizer, ids, budget) + (True,)

    open_ids = tokenizer(match.group(1), add_special_tokens=False)['input_ids']
    close_ids = tokenizer(match.group(4), add_special_tokens=False)['input_ids']
    body_ids = tokenizer(match.group(3), add_special_tokens=False)['input_ids']
    body_budget = budget - len(open_ids) - len(close_ids)
    if body_budget <= 0:
        return open_ids + close_ids, True, len(body_ids)
    kept, dropped = _head_tail(tokenizer, body_ids, body_budget, tail_fraction)
    return open_ids + kept + close_ids, True, dropped


def _head_tail(tokenizer, ids: List[int], budget: int,
               tail_fraction: float = 0.5) -> Tuple[List[int], int]:
    """Keep the head and the tail of ``ids`` within ``budget``, marking the gap."""
    if len(ids) <= budget:
        return list(ids), 0
    marker = tokenizer(_ELISION, add_special_tokens=False)['input_ids']
    avail = budget - len(marker)
    if avail <= 0:
        return list(ids[:budget]), len(ids) - budget
    tail_n = int(avail * tail_fraction)
    head_n = avail - tail_n
    if tail_n <= 0:
        return list(ids[:budget]), len(ids) - budget
    kept = list(ids[:head_n]) + marker + list(ids[len(ids) - tail_n:])
    return kept, len(ids) - head_n - tail_n
