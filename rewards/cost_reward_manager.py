# Copyright 2026 RouteWeaver.
"""Two reward components: the task verdict, and an efficiency score.

The task verdict is NOT re-derived here. TaskRewardManager stays the only
thing that decides correct / terminal_invalid / infra, and `reward_score` keeps
carrying exactly what it always carried, so a COST_ALPHA=0 run is the baseline
run. This class only ADDS a second component and the cost evidence behind it:

    r_task                     r_eff
    -----------------------------------------------
    +1  final answer correct   S_cost = 1/(1 + C/C_ref)
     0  legal but wrong        0
    -1  terminal_invalid       -1
     0  infra failure          0      (row is gated out of the gradient anyway)

A trajectory whose cost could not be fully measured (a call with no usage, or a
call to a worker the price table does not know) is flagged `cost_complete=0`
and given r_eff 0. The advantage estimator drops those rows from cost-enabled
updates rather than letting an unmeasured trajectory look free.

Cost detail (per call: worker, input, output, USD) goes into the dump so any
number reported later can be recomputed offline from the same fixed table.
"""
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
        # The alias layer can be re-pointed at a different backend without any
        # visible change to a run; this is what makes that fatal instead of
        # silently billing one model's tokens at another's price.
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
