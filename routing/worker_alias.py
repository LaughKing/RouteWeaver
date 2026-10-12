"""Fixed anonymous worker directory (worker_1 .. worker_6). ONE data source."""
import os
from typing import Dict, Tuple

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

BUDGET_TOKENS = 4096
BUDGET_SENTENCE = (f"Each worker call has a maximum output budget of "
                   f"{BUDGET_TOKENS} tokens.")

EXAMPLE_PRIMARY = "worker_1"
EXAMPLE_SECONDARY = "worker_2"
EXAMPLE_TERTIARY = "worker_3"

CATALOG_VERSION = "alias-v1-2026-09-12"


def enabled() -> bool:
    return os.environ.get("ROUTEWEAVER_WORKER_ALIAS", "0") == "1"


def catalog_block(worker_ids=None) -> str:
    """The block all three modes render, from this one source."""
    ids = list(worker_ids or ALIAS_POOL)
    lines = "\n".join(f"{wid}: {DESCRIPTIONS[wid]}" for wid in ids)
    return f"{BUDGET_SENTENCE}\n\n{lines}"


def to_real(alias: str) -> str:
    """alias -> backend model id. An id that is not an alias passes through
    unchanged, so a route naming a real id still resolves.
    """
    return ALIAS_TO_REAL.get(str(alias).strip().lower(), alias)


def to_alias(real: str) -> str:
    return REAL_TO_ALIAS.get(str(real).strip().lower(), real)


def mapping_banner() -> str:
    """Alias -> backend mapping for the launch log."""
    out = [f"[worker_alias] FIXED anonymous directory  version={CATALOG_VERSION}",
           f"[worker_alias] enforced output budget: {BUDGET_TOKENS} tokens "
           f"(every mode, every role)"]
    for alias, real, _ in _CATALOG:
        out.append(f"  {alias:9} -> {real}")
    return "\n".join(out)
