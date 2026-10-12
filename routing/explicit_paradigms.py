"""The explicit agentic paradigm section."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from route_grammar import (ACTIVE_WORKER_POOL,  # noqa: E402
                                               PARADIGM_SPECS)

_SCENARIO = {
    "refine": ("a single draft that then gets criticised and rewritten",
               ["Define a solar eclipse in one sentence.",
                "Say what is vague or wrong in the draft.",
                "Rewrite it using the criticism."]),
    "ensemble": ("two independent answers that then get merged",
                 ["Name Saturn's largest moon.",
                  "Give its diameter.",
                  "Combine both into one sentence."]),
    "debate": ("two opposing proposals, two rebuttals, then a verdict",
               ["Argue the metre came from the Earth's meridian.",
                "Argue the metre came from a pendulum.",
                "Rebut the first with evidence.",
                "Rebut the second with evidence.",
                "Say which the evidence supports."]),
    "plan": ("a plan, then its steps, then a summary",
             ["List the steps to convert 5 miles to kilometres.",
              "Carry out the conversion.",
              "State the result in one sentence."]),
}


def _count_phrase(lo, hi):
    if lo == hi:
        return f"exactly {lo} route" + ("s" if lo != 1 else "")
    return f"between {lo} and {hi} routes"


def _refs_phrase(refs_from):
    if not refs_from:
        return "no refs attribute"
    layers = " and ".join(f"layer {n}" for n in refs_from)
    return f"refs = EVERY route id of {layers}, comma separated, not a subset"


def _per_route_refs_warning(paradigm, forced=False):
    """Spell out that refs are required PER ROUTE, where that is confusable."""
    if not forced:
        return []
    spec = PARADIGM_SPECS[paradigm]
    counts = {n: layer.min_routes for n, layer in enumerate(spec, start=1)}
    for n, layer in enumerate(spec, start=1):
        if layer.max_routes <= 1 or not layer.refs_from:
            continue
        if not any(counts[m] > 1 for m in layer.refs_from):
            continue
        prior = [f"{m}-{k}" for m in layer.refs_from for k in range(1, counts[m] + 1)]
        allrefs = ",".join(prior)
        wrong = [f'<route id="{n}-{k}" role="{layer.role}" refs="{prior[min(k - 1, len(prior) - 1)]}">...</route>'
                 for k in range(1, counts[n] + 1)]
        right = [f'<route id="{n}-{k}" role="{layer.role}" refs="{allrefs}">...</route>'
                 for k in range(1, counts[n] + 1)]
        return ["", f"IMPORTANT -- layer {n}: each {layer.role} must reference EVERY "
                    f"route of " + " and ".join(f"layer {m}" for m in layer.refs_from) + ".",
                "References are required PER ROUTE, not collectively across the layer.",
                "", "Correct:"] + right + ["", "Incorrect:"] + wrong + [""]
    return []


def _example(paradigm):
    """A complete, legal trajectory for one paradigm, built from its own spec."""
    spec = PARADIGM_SPECS[paradigm]
    _, tasks = _SCENARIO[paradigm]
    workers = list(ACTIVE_WORKER_POOL)
    lines, task_i, ids_by_layer = [], 0, {}
    for n, layer in enumerate(spec, start=1):
        count = layer.min_routes
        ids = [f"{n}-{k}" for k in range(1, count + 1)]
        ids_by_layer[n] = ids
        refs = ""
        if layer.refs_from:
            prior = [i for m in layer.refs_from for i in ids_by_layer[m]]
            refs = f' refs="{",".join(prior)}"'
        lines.append(f'<layer index="{n}">')
        for k, rid in enumerate(ids):
            task = tasks[task_i] if task_i < len(tasks) else tasks[-1]
            task_i += 1
            lines.append(f'<route id="{rid}" model="{workers[k % len(workers)]}" '
                         f'role="{layer.role}"{refs}>{task}</route>')
        lines.append("</layer>")
        for rid in ids:
            lines.append(f'<observation id="{rid}">...</observation>')
    body = "\n".join(lines)
    return (f"<mode>agentic</mode>\n<paradigm>{paradigm}</paradigm>\n{body}\n"
            f"<answer>FINAL ANSWER</answer>")


def explicit_section(allowed=None) -> str:
    """The replacement for the compressed '## Paradigms' table."""
    names = [p for p in PARADIGM_SPECS if allowed is None or p in allowed]
    assert names, f"no paradigm left after filtering by {allowed!r}"
    if len(names) == 1:
        out = ["## Paradigm",
               "",
               f"You must use the {names[0]} paradigm for this question. Follow its "
               "layer plan exactly: it states how many layers it has, how many "
               "routes each layer takes, the one role string that layer must use, "
               "and which route ids that layer must reference. A layer with the "
               "wrong number of routes, the wrong role string, or incomplete refs "
               "is rejected in full.",
               ""]
    else:
        out = ["## Paradigms",
               "",
               "Pick ONE paradigm and follow its layer plan exactly. Each plan below "
           "states how many layers it has, how many routes each layer takes, the "
           "one role string that layer must use, and which route ids that layer "
           "must reference. A layer with the wrong number of routes, the wrong "
           "role string, or incomplete refs is rejected in full.",
           ""]
    for paradigm in names:
        spec = PARADIGM_SPECS[paradigm]
        out.append(f"### {paradigm} -- {len(spec)} layers: {_SCENARIO[paradigm][0]}")
        for n, layer in enumerate(spec, start=1):
            every = " on every route" if layer.max_routes > 1 else ""
            out.append(f"  layer {n}: {_count_phrase(layer.min_routes, layer.max_routes)}, "
                       f'role="{layer.role}"{every}, '
                       f"{_refs_phrase(layer.refs_from)}")
        out.append("")
        out.extend(_per_route_refs_warning(paradigm, forced=len(names) == 1))
        out.append(f"Complete legal example for {paradigm}:")
        out.append("")
        out.append(_example(paradigm))
        out.append("")
    out.append("The role strings are fixed and case sensitive: "
               + ", ".join(sorted({l.role for s in PARADIGM_SPECS.values() for l in s}))
               + ". A layer uses only the role its plan names.")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(explicit_section())
