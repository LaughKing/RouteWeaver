"""Worked examples for `multi`."""
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
