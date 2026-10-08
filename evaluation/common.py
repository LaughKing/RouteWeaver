"""Shared bootstrap for the benchmark evaluation.

IMPORT ORDER IS LOAD-BEARING: routing's scorer must be imported
before anything puts the repo root on sys.path, or `import verl` resolves to the
frozen the original fork, prime_math is missing, and equivalence scoring silently degrades
to string comparison (which mis-scored a whole day of math results once).

Everything reusable is imported FROM routing rather than copied:
the fixed extractor, the payload-hygiene worker pool and the raw-chat client.
This directory adds only manifests,
per-benchmark scoring, the mode runners and the pipeline.
"""
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

# Extension probes: reported per-dataset but kept OUT of the pooled section
# (omni_math_d36x would double-weight math in the cross-benchmark pool;
# taco_mm and musique_100 are TRAIN-pool calibration probes, not benchmarks).
POOLED_EXCLUDE = {"omni_math_d36x", "taco_mm", "musique_100"}
MODES = ("forced_single", "forced_multi", "forced_agentic_adaptive_layer")

# Answer-format terminators, appended to the question so every mode sees the
# same bytes. These are the clean_v1 style validated in the rebuilt bench_data:
# they say where the answer goes and what vocabulary it uses, never how much the
# model may reason -- the contamination that sank the first six-benchmark run
# was "final answer only / no explanation", which none of these contain (and
# the manifest builder asserts that against worker_client.BANNED_IN_PAYLOAD).
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
    # TRAIN-ONLY scorer_dataset (RouteWeaver RECALL). Same wording as
    # simpleqa_verified so the two RECALL sources ask for the same answer shape;
    # the VERDICT differs (triviaqa is alias-aware binary EM, no containment).
    "triviaqa": "End your response with the short answer on its own line.",
    "taco_mm": ("Write a complete Python solution that reads from standard "
                "input and writes to standard output. End your response with "
                "the final program as a single fenced code block "
                "(```python ... ```)."),
}
