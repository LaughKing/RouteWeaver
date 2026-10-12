# Copyright 2026 RouteWeaver.
"""Two reward components: the task verdict, and an efficiency score."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cost_model                                                 # noqa: E402
from task_reward_manager import TaskRewardManager      # noqa: E402


class CostRewardManager(TaskRewardManager):
    """The task reward plus the efficiency component. reward_score is untouched."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = cost_model.cost_alpha()
        self._c_ref = cost_model.c_ref()
        if self._alpha > 0 and not self._c_ref:
            raise cost_model.CostConfigError(
                "COST_ALPHA > 0 requires a frozen C_REF; refusing to train "
                "against a normalizer that does not exist yet")
        pool = [w.strip() for w in
                (os.environ.get("WORKER_IDS") or "").split(",") if w.strip()]
        self._pool = cost_model.verify_pool(pool) if pool else {}
        print(f"[cost] reward manager: alpha={self._alpha} c_ref={self._c_ref} "
              f"prices={cost_model.PRICE_TABLE_VERSION} pool={self._pool}",
              flush=True)

    async def run_single(self, data) -> dict:
        out = await super().run_single(data)
        info = out["reward_extra_info"]
        item = data[-1:][0]
        fields = dict(item.non_tensor_batch.get("tool_extra_fields") or {})

        detail = cost_model.trajectory_cost(fields.get("worker_records") or [])
        complete = int(detail["cost_complete"])
        r_task = float(info.get("reward_value", out["reward_score"]))

        if self._c_ref and complete:
            score = cost_model.s_cost(detail["cost_usd"], self._c_ref)
        else:
            score = 0.0
        if int(info.get("infra_failed") or 0):
            r_eff = 0.0
        elif int(info.get("terminal_invalid") or 0) and r_task < 0:
            r_eff = -1.0
        elif int(info.get("correct") or 0):
            r_eff = float(score) if complete else 0.0
        else:
            r_eff = 0.0

        info["r_task"] = r_task
        info["r_eff"] = float(r_eff)
        info["cost_usd"] = float(detail["cost_usd"])
        info["cost_calls"] = int(detail["n_calls"])
        info["cost_usage_unknown"] = int(detail["usage_unknown"])
        info["cost_unpriced_calls"] = int(detail["unpriced_calls"])
        info["cost_complete"] = complete
        info["s_cost"] = float(score) if self._c_ref and complete else -1.0
        info["cost_calls_json"] = cost_model.calls_json(detail)
        info["cost_price_version"] = cost_model.PRICE_TABLE_VERSION
        info["cost_c_ref"] = float(self._c_ref or 0.0)
        info["cost_alpha"] = float(self._alpha)
        return out
