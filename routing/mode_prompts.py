# Copyright 2026 RouteWeaver.
"""The single- and multi-round rule text injected after a mode declaration.

`mode_selector.mode_rules` renders the rules for the declared mode. For
`agentic` it owns the text itself; for `single` and `multi` the rules ARE the
corresponding one-mode action space from `prompt_pool.render_prompt`, which is
what this builder produces. Extracted verbatim from the evaluation harness the
reported runs imported it from, so the rendered text is byte-identical.

`STAGED` keys are historical ("forced_*"): the mode is given, which is also the
situation during the cold start and the forced arm of the transition.
"""
from prompt_pool import render_prompt

ROUTER_RULES = (
    "When you call a worker, give it the task to solve. Do not put your own "
    "candidate answer in the call, and do not ask a worker merely to confirm, "
    "verify or agree with an answer you already have. Each worker must be asked "
    "to do the work itself.")
STAGED = {"forced_single": ("single",), "forced_multi": ("multi",),
          "forced_multi_refs": ("multi",)}
SUFFIX = {"forced_single": "",
          "forced_multi": "Use between 2 and 4 calls as the query requires.",
          "forced_multi_refs": (
              "Use between 2 and 4 calls as the query requires. "
              "Each worker after the first automatically sees the previous "
              "worker's full output, so a task may build directly on it.")}


def build_sm_prompt(question, mode, worker_ids=None):
    """The one-mode action space for `mode`, plus the router rules.

    The explicit_v2 variant restates the multi rule; on an action space that has
    no multi rule to restate (single) it refuses, and the compact variant is the
    intended fallback -- the control flow the reported runs used.
    """
    try:
        text = render_prompt(question, model="qwen", allowed_modes=STAGED[mode],
                             variant="explicit_v2", worker_ids=worker_ids)
    except ValueError as exc:
        if "does not apply to this action space" not in str(exc):
            raise
        text = render_prompt(question, model="qwen", allowed_modes=STAGED[mode],
                             variant="compact", worker_ids=worker_ids)
    extra = [ROUTER_RULES] + ([SUFFIX[mode]] if SUFFIX[mode] else [])
    return text.replace(f"Question: {question}",
                        f"{' '.join(extra)}\n\nQuestion: {question}")
