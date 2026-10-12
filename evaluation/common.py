"""Shared bootstrap for the benchmark evaluation."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DYN = HERE.parent / "routing"
REPO = HERE.parents[0]

for _p in (str(HERE), str(DYN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scorer as math_scorer                       # noqa: E402  MUST BE FIRST

for _p in (str(REPO / "rewards"), str(REPO / "agent_loops"), str(REPO / "routing")):
    if _p not in sys.path:
        sys.path.append(_p)

SEED = 20260817
DATASETS = ("omni_math", "simpleqa_verified", "supergpqa", "satbench",
            "bbeh_mini", "longbench_v2", "livecodebench", "omni_math_d36x",
            "taco_mm", "musique_100")

POOLED_EXCLUDE = {"omni_math_d36x", "taco_mm", "musique_100"}
MODES = ("forced_single", "forced_multi", "forced_agentic_adaptive_layer")

TERMINATOR = {
    "omni_math": "End your response with the final answer on its own line.",
    "simpleqa_verified": "End your response with the short answer on its own line.",
    "supergpqa": ("End your response with the letter of the correct option on "
                  "its own line, for example: C"),
    "satbench": ("Decide whether all the conditions can hold at the same time. "
                 "End your response with a single word on its own line: SAT if "
                 "they can, UNSAT if they cannot."),
    "bbeh_mini": "End your response with the short answer on its own line.",
    "longbench_v2": ("End your response with the letter of the correct option "
                     "on its own line, for example: C"),
    "livecodebench": ("Write a complete Python solution. End your response "
                      "with the final program as a single fenced code block "
                      "(```python ... ```)."),
    "omni_math_d36x": "End your response with the final answer on its own line.",
    "musique_100": "End your response with the short answer on its own line.",
    "triviaqa": "End your response with the short answer on its own line.",
    "taco_mm": ("Write a complete Python solution that reads from standard "
                "input and writes to standard output. End your response with "
                "the final program as a single fenced code block "
                "(```python ... ```)."),
}
