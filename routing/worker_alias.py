"""Fixed anonymous worker directory (worker_1 .. worker_6). ONE data source.

Every place that renders a worker list -- prompt_pool (single/multi) and
route_syntax (agentic) -- calls `catalog_block()` here, so the three modes
cannot drift into three separately maintained copies of the same text.

THE MAPPING IS FIXED. Not permuted per episode, per resume or per eval. A
permuted mapping would leave the worker-name tokens carrying no stable meaning
and would make every share/entropy number unreadable across steps.

WHAT THE ROUTER SEES: the alias, the capability description, and the output
budget. Never the backend, the provider, a tier word (small/medium/large), a
"strongest in this pool" hint, or a role pre-assignment. The descriptions are
operator-supplied initial capability information taken from official model
documentation; they are NOT a measured ranking inside this pool, and they do not
bind any worker to planner/executor/verifier.

FIELD NAME IS UNCHANGED. Routes still read model="worker_1". Node identity is
separate and still n1/n2/...; refs reference node ids, never worker numbers.


OUTPUT BUDGET. `BUDGET_TOKENS` is the cap the dispatcher actually applies, on
every mode and every role -- confirmed by building the real payload with the
network mocked rather than by reading the config chain. The channel entries'
own max_tokens never bind: `llm_client` resolves `max_tokens or
self.default_max_tokens or entry.get("max_tokens", ...)`, and the loop always
supplies the pool's value.
"""
import os
from typing import Dict, Tuple

# (alias, backend model id, description)
_CATALOG: Tuple[Tuple[str, str, str], ...] = (
    ("worker_1", "qwen3_8b",
     "Dense language model with 8.2B parameters. Supports multilingual "
     "instruction following, dialogue, mathematical and logical reasoning, "
     "and code generation."),
    ("worker_2", "llama31_8b",
     "Dense language model with 8B parameters. Instruction-tuned for "
     "multilingual dialogue, with capabilities in general reasoning, text "
     "generation, and coding."),
    ("worker_3", "qwen3_30b_a3b",
     "Mixture-of-experts language model with 30.5B total and 3.3B active "
     "parameters. Supports instruction following, text comprehension, logical "
     "reasoning, mathematics, science, and coding."),
    ("worker_4", "gemma3_27b_it",
     "Dense language model in the 27B parameter class. Supports multilingual "
     "question answering, summarization, reasoning, mathematical problem "
     "solving, and code generation."),
    ("worker_5", "llama33_70b",
     "Dense language model with 70B parameters. Instruction-tuned for "
     "multilingual dialogue, with capabilities in reasoning, instruction "
     "following, and text and code generation."),
    ("worker_6", "gemini25_flash_lite",
     "Sparse mixture-of-experts model with undisclosed parameter counts. "
     "Supports translation, classification, information extraction, coding, "
     "mathematics, and reasoning."),
)

ALIAS_POOL: Tuple[str, ...] = tuple(a for a, _, _ in _CATALOG)
ALIAS_TO_REAL: Dict[str, str] = {a: r for a, r, _ in _CATALOG}
REAL_TO_ALIAS: Dict[str, str] = {r: a for a, r, _ in _CATALOG}
DESCRIPTIONS: Dict[str, str] = {a: d for a, _, d in _CATALOG}

# The cap the dispatcher applies, on every mode and every role. Verified against
# the real payload construction path, not read off the config.
BUDGET_TOKENS = 4096
BUDGET_SENTENCE = (f"Each worker call has a maximum output budget of "
                   f"{BUDGET_TOKENS} tokens.")

# Worked-example anchors. FIXED, and deliberately not taken from the per-query
# display order: a moving anchor makes demo anchoring indistinguishable from
# position (route_syntax.build).
EXAMPLE_PRIMARY = "worker_1"
EXAMPLE_SECONDARY = "worker_2"
EXAMPLE_TERTIARY = "worker_3"

# Bumped whenever the ids, the descriptions or the budget sentence change, so a
# cached template or a dataset fingerprint cannot silently reuse an old catalog.
CATALOG_VERSION = "alias-v1-2026-09-12"


def enabled() -> bool:
    return os.environ.get("ROUTEWEAVER_WORKER_ALIAS", "0") == "1"


def catalog_block(worker_ids=None) -> str:
    """The block all three modes render, from this one source.

    `worker_ids` is a DISPLAY ORDER only (worker_explore_schedule.display_order);
    it never changes which ids are legal. None keeps the canonical order.
    """
    ids = list(worker_ids or ALIAS_POOL)
    lines = "\n".join(f"{wid}: {DESCRIPTIONS[wid]}" for wid in ids)
    return f"{BUDGET_SENTENCE}\n\n{lines}"


def to_real(alias: str) -> str:
    """alias -> backend model id. An id that is not an alias passes through
    unchanged, so a route naming a real id still resolves."""
    return ALIAS_TO_REAL.get(str(alias).strip().lower(), alias)


def to_alias(real: str) -> str:
    return REAL_TO_ALIAS.get(str(real).strip().lower(), real)


def mapping_banner() -> str:
    """For the launch log, so the run records which backend each alias stood for
    without anyone having to reconstruct it from the code at that revision.
    Never rendered into a prompt."""
    out = [f"[worker_alias] FIXED anonymous directory  version={CATALOG_VERSION}",
           f"[worker_alias] enforced output budget: {BUDGET_TOKENS} tokens "
           f"(every mode, every role)"]
    for alias, real, _ in _CATALOG:
        out.append(f"  {alias:9} -> {real}")
    return "\n".join(out)
