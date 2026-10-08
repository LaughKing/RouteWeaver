"""Worked examples for `multi`.

The compact template states multi as a single rule line with no worked example,
while `single` and `agentic` both have one. A mode shown only as notation is
reliably harder to produce than a mode shown in use, so the explicit variant
adds the examples below.

What the rule line does not show, and these examples do:

  * the calls are SERIAL. One <route>, then its <observation>, then the next
    <route>. Not two routes and then two observations.
  * every route after the first USES the previous observation. A multi
    trajectory whose second call could have been written before the first
    observation arrived is a single call with extra steps.
  * 2, 3 or 4 calls are all legal, so both a two-hop and a three-hop shape are
    shown rather than one canonical length the policy might copy exactly.
  * <paradigm> and <layer> never appear. They belong to agentic, and mixing the
    surfaces is `paradigm_unexpected` / `layer_unexpected`.

Nothing here relaxes anything: the route-count bounds, the serial pairing and
the tag vocabulary are all enforced by the trajectory validator, and every
example below is legal under it. The "2 to 4" these examples teach is the
validator's own bound, not a second statement of it.

The scenarios are neutral two- and three-hop lookups. None is a question from
any split, and they teach shape only.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from route_grammar import ACTIVE_WORKER_POOL  # noqa: E402

MIN_ROUTES, MAX_ROUTES = 2, 4          # pinned against the validator by a test


def _hop(worker, task, observation):
    return (f'<route model="{worker}">{task}</route>\n'
            f'<observation>{observation}</observation>')


def two_route_example():
    a, b = ACTIVE_WORKER_POOL[0], ACTIVE_WORKER_POOL[1]
    return ("<mode>multi</mode>\n"
            + _hop(a, "Which film won the Academy Award for Best Picture for 1994?",
                   "Forrest Gump won Best Picture for 1994.")
            + "\n"
            + _hop(b, "Who directed Forrest Gump?",
                   "Forrest Gump was directed by Robert Zemeckis.")
            + "\n<answer>Robert Zemeckis</answer>")


def three_route_example():
    a, b, c = (ACTIVE_WORKER_POOL[0], ACTIVE_WORKER_POOL[1],
               ACTIVE_WORKER_POOL[2 % len(ACTIVE_WORKER_POOL)])
    return ("<mode>multi</mode>\n"
            + _hop(a, "Which country has the Danube as its longest river?",
                   "Hungary's longest river is the Danube.")
            + "\n"
            + _hop(b, "What is the capital of Hungary?",
                   "The capital of Hungary is Budapest.")
            + "\n"
            + _hop(c, "What is the population of Budapest?",
                   "Budapest has a population of about 1.7 million.")
            + "\n<answer>About 1.7 million</answer>")


def explicit_multi_section() -> str:
    """The block inserted after the single example, before any agentic one."""
    return "\n".join([
        "## Example -- multi",
        "",
        f"A multi trajectory makes between {MIN_ROUTES} and {MAX_ROUTES} worker "
        "calls, ONE AT A TIME. Write one <route>, wait for its <observation>, "
        "and only then write the next <route>. Every route after the first must "
        "use what the previous observation returned -- if a later call could "
        "have been written before its predecessor answered, the question did "
        "not need multi. A multi trajectory never contains <paradigm> or "
        "<layer>; those belong to agentic. It ends with one <answer>.",
        "",
        "Two calls:",
        "",
        two_route_example(),
        "",
        "Three calls:",
        "",
        three_route_example(),
        "",
    ])


if __name__ == "__main__":
    print(explicit_multi_section())
