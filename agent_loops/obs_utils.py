# Copyright 2026 RouteWeaver.
"""Observation helpers — copies pinned to the original semantics.

``_OBS_WRAPPER`` and the body-preserving truncation are ports of Router-R1's
``routing/generation.py``. They live here rather than being imported, because
that module pulls in a different verl layout -- and because the agentic surface
needs the two deltas noted below.

Token-space contract: observations are ENVIRONMENT text (response_mask 0,
logprob 0), so tokenizing open tag / body / close tag separately is safe — the
result always decodes to balanced tags.
"""

import re
from typing import List, Tuple

# Split a wrapped observation into
# (open tag, tag name, body, close tag); greedy body pairs with the LAST close.
#
# Delta vs the original: the opening tag may carry attributes, because an agentic
# observation is written as <observation id="1-1">. Without this the wrapper
# match fails, truncation falls through to the plain token cut, and a long
# agentic observation loses its </observation> — which the analyzer reads as
# unbalanced_observation and scores -1 for a tokenizer artifact.
_OBS_WRAPPER = re.compile(
    r'^(\s*<(observation|information)\b[^>]*>)(.*)(</\2\s*>\s*)$', re.DOTALL)


# The elision marker written between the kept head and the kept tail. It is a
# FIXED string on purpose: putting the dropped-token count in the text would
# make the marker's own length depend on the number it reports, and that number
# depends on the marker's length. The count is reported in the truncation
# metrics instead, where nothing reads it back into the prompt.
_ELISION = "\n\n... [observation truncated] ...\n\n"


def truncate_observation_ids(tokenizer, text: str, budget: int,
                             tail_fraction: float = 0.5
                             ) -> Tuple[List[int], bool, int]:
    """Tokenize ONE observation, cutting the BODY to fit ``budget`` while
    keeping the wrapper tags intact.

    Returns ``(token_ids, was_truncated, dropped_tokens)``.

    HEAD+TAIL, not head-only. Keeping ``body_ids[:budget]`` keeps the head,
    which is the wrong half here: an executor writes prose and then its
    program, a summarizer writes reasoning and then its answer, so the thing
    the router has to read is at the END. Measured over the evaluation sets, a
    head-only cut cost observation text on half of the CODE rows and most of
    the REASON ones. Both ends are kept now, with ``_ELISION`` marking the gap
    so the policy sees that something was removed rather than reading a
    non-sequitur.

    ``tail_fraction`` is the share of the surviving body given to the tail.
    0.5 splits evenly; higher values favour the end.
    """
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if len(ids) <= budget:
        return ids, False, 0

    match = _OBS_WRAPPER.match(text)
    if match is None:
        # not a wrapped observation: nothing to protect, plain head+tail cut
        return _head_tail(tokenizer, ids, budget) + (True,)

    open_ids = tokenizer(match.group(1), add_special_tokens=False)['input_ids']
    close_ids = tokenizer(match.group(4), add_special_tokens=False)['input_ids']
    body_ids = tokenizer(match.group(3), add_special_tokens=False)['input_ids']
    body_budget = budget - len(open_ids) - len(close_ids)
    if body_budget <= 0:
        # budget smaller than the wrapper itself: emit balanced empty tags
        # rather than a clipped one
        return open_ids + close_ids, True, len(body_ids)
    kept, dropped = _head_tail(tokenizer, body_ids, body_budget, tail_fraction)
    return open_ids + kept + close_ids, True, dropped


def _head_tail(tokenizer, ids: List[int], budget: int,
               tail_fraction: float = 0.5) -> Tuple[List[int], int]:
    """Keep the head and the tail of ``ids`` within ``budget``, marking the gap.

    Returns ``(kept_ids, dropped_count)``. Falls back to a head-only cut when
    the budget cannot even hold the marker, which is the only case where the
    old behaviour survives.
    """
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
