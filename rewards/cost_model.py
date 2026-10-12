# Copyright 2026 RouteWeaver.
"""Fixed reference pricing and the trajectory cost the efficiency reward reads."""
import json
import os
from typing import Dict, Iterable, Optional, Tuple

PRICE_TABLE_VERSION = "fixed-reference-2026-09-20"

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
    """
    known = any(record.get(k) is not None for k in
                ("input_tokens", "output_tokens",
                 "failed_input_tokens", "failed_output_tokens"))
    if not known:
        return None, None
    tin = int(record.get("input_tokens") or 0) + int(record.get("failed_input_tokens") or 0)
    tout = int(record.get("output_tokens") or 0) + int(record.get("failed_output_tokens") or 0)
    if record.get("success") and tin == 0 and tout == 0:
        return None, None
    return tin, tout


def trajectory_cost(worker_records) -> dict:
    """Per-call detail and the trajectory total."""
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
    """Per-call detail for the dump, so cost can be re-derived offline."""
    return json.dumps(detail["calls"][:limit], separators=(",", ":"))
