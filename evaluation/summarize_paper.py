#!/usr/bin/env python
"""Paper numbers for one evaluated checkpoint, from its validation dumps.

    python evaluation/summarize_paper.py <eval_root> <tag>

Reads <eval_root>/eval-<tag>-{set600,set700,triviaqa120}/ and any
eval-<tag>-ood-*/ present.

In-distribution (Table 1): nine benchmarks, 1060 questions --
    Math       MATH L2-5 (set600), Omni-MATH (set700)
    Code       APPS (set600), LiveCodeBench, TACO (set700)
    Knowledge  SuperGPQA (set700)
    Reason     SATBench (set700)
    Recall     TriviaQA (triviaqa120), SimpleQA-Verified (set700)
Domain = unweighted mean of its datasets; Avg = unweighted mean of the domains.
set600 also carries SuperGPQA / SATBench / PopQA rows; they are not part of
the reported set. Mode shares (Table 2, Fig. 4) are over the same 1060 rows.
"""
import collections
import glob
import json
import sys

DOMAINS = {
    "Math": ["math_l2-5", "omni_math"],
    "Code": ["apps", "livecodebench", "taco_mm"],
    "Knowledge": ["supergpqa"],
    "Reason": ["satbench"],
    "Recall": ["triviaqa", "simpleqa_verified"],
}
SOURCE = {"math_l2-5": "set600", "apps": "set600", "triviaqa": "triviaqa120"}  # else set700
MODES = ("single", "multi", "agentic")


def load(root, name):
    files = sorted(glob.glob(f"{root}/{name}/validation_dump/*.jsonl"))
    if not files:
        return None
    return [json.loads(line) for line in open(files[-1])]


def ds(row):
    return str(row.get("rfm_dataset")).lower()


def main(root, tag):
    parts = {k: load(root, f"eval-{tag}-{k}") for k in ("set600", "set700", "triviaqa120")}
    rows = []
    for name in (n for d in DOMAINS.values() for n in d):
        src = parts[SOURCE.get(name, "set700")]
        if src is None:
            sys.exit(f"missing dump eval-{tag}-{SOURCE.get(name, 'set700')}")
        rows += [r for r in src if ds(r) == name]
    by = collections.defaultdict(list)
    for r in rows:
        by[ds(r)].append(bool(r.get("correct")))
    acc = {k: sum(v) / len(v) for k, v in by.items()}
    dom = {d: sum(acc[n] for n in names) / len(names) for d, names in DOMAINS.items()}
    avg = sum(dom.values()) / len(dom)
    print(f"== {tag}: {len(rows)} questions")
    print("  per dataset: " + ", ".join(f"{k} {100 * acc[k]:.1f} (n={len(by[k])})" for k in sorted(acc)))
    print("  " + "  ".join(f"{d} {100 * v:.1f}" for d, v in dom.items()) + f"  Avg {100 * avg:.1f}")
    modes = collections.Counter(r.get("mode") for r in rows)
    print("  mode usage S/M/A: " + " / ".join(f"{modes[m] / len(rows):.2f}" for m in MODES))
    for d, names in DOMAINS.items():
        sub = [r for r in rows if ds(r) in names]
        c = collections.Counter(r.get("mode") for r in sub)
        print(f"    {d:<9} " + " / ".join(f"{c[m] / len(sub):.2f}" for m in MODES))

    ood = sorted(glob.glob(f"{root}/eval-{tag}-ood-*"))
    if ood:
        print("  held-out (Table 3):")
        for path in ood:
            name = path.rsplit("-ood-", 1)[1]
            r = load(root, f"eval-{tag}-ood-{name}") or []
            if r:
                c = collections.Counter(x.get("mode") for x in r)
                print(f"    {name:<16} {100 * sum(bool(x.get('correct')) for x in r) / len(r):.1f}"
                      f"  (n={len(r)}, S/M/A " + " / ".join(f"{c[m] / len(r):.2f}" for m in MODES) + ")")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
