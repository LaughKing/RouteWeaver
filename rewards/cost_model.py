# Copyright 2026 RouteWeaver.
"""Fixed reference pricing and the trajectory cost the efficiency reward reads.

THE PRICE TABLE IS FROZEN CONFIGURATION, NOT A LIVE QUOTE. It is a versioned
constant: nothing here is ever refreshed from a provider's endpoints API, from
a response's usage.cost field, or from a real bill. Two runs are therefore
comparable by construction, even if a provider changes its list price
mid-run. The table says "reference USD per 1M
tokens" and claims nothing about what was actually paid.

WHAT IS COUNTED: worker inference only.

    C_call = (input_tokens * in_price + output_tokens * out_price) / 1e6
    C      = sum over every worker call of the trajectory (single/multi/agentic)

  * the router's instruction to a worker is part of that worker's input and is
    billed as input, which it already is -- input_tokens is what the provider
    charged for the whole prompt;
  * context re-sent on a later call is billed again, again because it is in
    that call's input_tokens;
  * CACHED INPUT is billed at the ordinary input price. cached_input_tokens is
    a SUBSET of input_tokens (the provider reports the same prompt twice, once
    as a total and once as its cached part), so it is deliberately NOT added:
    adding it would bill those tokens twice;
  * REASONING tokens are already inside completion_tokens (:
    completion=88 with reasoning_tokens=64), so they are not added either;
  * a RETRY or a FAILED attempt that reported usage IS billed -- the provider
    charged for it. llm_client accumulates that as failed_input_tokens /
    failed_output_tokens on the same record;
  * usage that is MISSING is never silently zero. The call is counted in
    `usage_unknown` and the trajectory is marked cost-incomplete, which keeps it
    out of cost-enabled updates instead of handing it a fake low cost.

The ROUTER's own generation is measured (policy_token_count / env_token_count
are already on every row) and deliberately does NOT enter C.

S_cost = 1 / (1 + C / C_ref), with C_ref a frozen positive constant supplied by
the launcher. It is 1 at zero cost, 0.5 at C_ref, monotonically decreasing, and
never rescaled by anything in the current batch.
"""
import json
import os
from typing import Dict, Iterable, Optional, Tuple

PRICE_TABLE_VERSION = "fixed-reference-2026-09-20"

# worker id -> (expected backend model id, input USD/1M, output USD/1M)
# The backend id is carried so a silently re-pointed alias fails loudly rather
# than billing Llama-3.3-70B's tokens at Qwen3-8B's price.
FIXED_PRICES: Dict[str, Tuple[str, float, float]] = {
    "worker_1": ("qwen3_8b",            0.02, 0.05),
    "worker_2": ("llama31_8b",          0.02, 0.05),
    "worker_3": ("qwen3_30b_a3b",       0.10, 0.30),
    "worker_4": ("gemma3_27b_it",       0.119, 0.20),
    "worker_5": ("llama33_70b",         0.10, 0.32),
    "worker_6": ("gemini25_flash_lite", 0.10, 0.40),
}


class CostConfigError(RuntimeError):
    """Raised when cost is enabled on a pool the price table cannot describe."""


def enabled() -> bool:
    return cost_alpha() > 0.0


def cost_alpha() -> float:
    return float(os.environ.get("COST_ALPHA", "0") or 0.0)


def c_ref() -> Optional[float]:
    raw = os.environ.get("C_REF", "").strip()
    if not raw:
        return None
    value = float(raw)
    if value <= 0:
        raise CostConfigError(f"C_REF must be a positive constant, got {value}")
    return value


def verify_pool(worker_ids: Iterable[str]) -> Dict[str, str]:
    """Check that every offered worker is priced AND still points at the model
    the price was set for. Returns {worker_id: backend model id}.

    The alias layer maps worker_N to a backend spec at import time, so a change
    of backend is invisible in the run header; this is the check that makes it
    visible. Raises rather than warning: a wrong mapping does not break the run
    in any observable way, it just trains the policy on fictional prices.
    """
    import worker_alias
    out = {}
    for wid in worker_ids:
        wid = str(wid).strip().lower()
        if wid not in FIXED_PRICES:
            raise CostConfigError(
                f"worker {wid!r} is offered to the policy but has no fixed "
                f"price; refusing to run cost-enabled training")
        expected = FIXED_PRICES[wid][0]
        actual = worker_alias.ALIAS_TO_REAL.get(wid, wid)
        if actual != expected:
            raise CostConfigError(
                f"price table pins {wid} to {expected!r} but the alias map "
                f"resolves it to {actual!r}; fix one of them before training")
        out[wid] = actual
    return out


def call_cost(worker_id: str, input_tokens: int, output_tokens: int) -> float:
    spec = FIXED_PRICES.get(str(worker_id).strip().lower())
    if spec is None:
        raise CostConfigError(f"no fixed price for worker {worker_id!r}")
    _, price_in, price_out = spec
    return (int(input_tokens) * price_in + int(output_tokens) * price_out) / 1e6


def _billable(record) -> Tuple[Optional[int], Optional[int]]:
    """(input, output) tokens to bill for one worker record, or (None, None)
    when the provider reported no usage at all for a call that happened.

    A successful call always carries usage; a failed one may carry the usage of
    the attempts that failed (llm_client accumulates failed_* on the record) or
    nothing at all, which is the unknown case.
    """
    known = any(record.get(k) is not None for k in
                ("input_tokens", "output_tokens",
                 "failed_input_tokens", "failed_output_tokens"))
    if not known:
        return None, None
    tin = int(record.get("input_tokens") or 0) + int(record.get("failed_input_tokens") or 0)
    tout = int(record.get("output_tokens") or 0) + int(record.get("failed_output_tokens") or 0)
    if record.get("success") and tin == 0 and tout == 0:
        # a reply arrived but no usage came with it
        return None, None
    return tin, tout


def trajectory_cost(worker_records) -> dict:
    """Per-call detail and the trajectory total.

    Never raises on an unknown worker id: it marks the trajectory incomplete
    instead, so one stray id costs a row rather than the run. An unpriced id in
    the OFFERED pool is caught earlier, by verify_pool, before any training.
    """
    calls, total, unknown, unpriced = [], 0.0, 0, 0
    for record in (worker_records or []):
        wid = str(record.get("worker_id") or "")
        tin, tout = _billable(record)
        entry = {"w": wid, "in": tin, "out": tout,
                 "ok": bool(record.get("success"))}
        if tin is None:
            unknown += 1
            entry["usd"] = None
        elif wid.lower() not in FIXED_PRICES:
            unpriced += 1
            entry["usd"] = None
        else:
            usd = call_cost(wid, tin, tout)
            entry["usd"] = round(usd, 9)
            total += usd
        calls.append(entry)
    return {"cost_usd": total, "calls": calls, "n_calls": len(calls),
            "usage_unknown": unknown, "unpriced_calls": unpriced,
            "cost_complete": int(unknown == 0 and unpriced == 0)}


def s_cost(cost_usd: float, reference: float) -> float:
    """1 / (1 + C / C_ref). Monotone decreasing, in (0, 1]."""
    if reference is None or reference <= 0:
        raise CostConfigError(
            "S_cost needs a frozen positive C_ref; refusing to invent one")
    return 1.0 / (1.0 + float(cost_usd) / float(reference))


def calls_json(detail: dict, limit: int = 32) -> str:
    """Per-call detail for the dump, so cost can be re-derived offline.

    Capped: an agentic trajectory can make many calls and the dump is one line
    per rollout. The total and the counts above are never truncated.
    """
    return json.dumps(detail["calls"][:limit], separators=(",", ":"))
