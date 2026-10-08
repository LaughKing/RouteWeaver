#!/usr/bin/env python
"""Freeze C_ref from a calibration rollout's validation dump.

C_ref = P95 of the per-trajectory worker cost over REFERENCE rollouts:

    no infrastructure failure   (infra_failed == 0)
    complete usage              (cost_complete == 1)

Correct and incorrect trajectories both count -- an expensive wrong answer is
still an expensive trajectory -- and single-worker rollouts are not privileged
over multi-worker ones. Nothing is filtered on mode, dataset or reward.

The output is the frozen record -- the value and the price table it was
computed against -- written next to the dump as cost_ref.json and echoed here.
Copy it to configs/cost_ref.json to make it the constant every alpha > 0 run
normalizes by.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rewards"))
import cost_model                                                 # noqa: E402


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q / 100 * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main():
    out = Path(sys.argv[1])
    dumps = sorted((out / "validation_dump").glob("*.jsonl"))
    if not dumps:
        raise SystemExit(f"no validation dump under {out}")
    rows = [json.loads(line) for line in open(dumps[-1])]
    ref, skipped = [], Counter()
    for r in rows:
        if int(r.get("infra_failed") or 0):
            skipped["infra_failed"] += 1
            continue
        if not int(r.get("cost_complete") or 0):
            skipped["cost_incomplete"] += 1
            continue
        ref.append(float(r.get("cost_usd") or 0.0))
    if not ref:
        raise SystemExit("no reference trajectories; refusing to invent a C_ref")

    c_ref = percentile(ref, 95)
    print(f"{len(ref)} reference rollouts, {dict(skipped)} skipped")
    record = {
        "c_ref_usd": c_ref,
        "statistic": "P95 of per-trajectory worker cost",
        "n_reference_rollouts": len(ref),
        "price_table_version": cost_model.PRICE_TABLE_VERSION,
        "prices": {w: {"model": s[0], "in": s[1], "out": s[2]}
                   for w, s in cost_model.FIXED_PRICES.items()},
    }
    (out / "cost_ref.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "prices"}, indent=2))


if __name__ == "__main__":
    main()
