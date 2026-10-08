"""LiveCodeBench correctness: extract the program, run it against the test
cases, all-or-nothing (standard LCB pass@1).

`scorers.py` holds the non-code scorers -- QA sub-EM, gsm8k numeric, math
symbolic and ARC multiple choice -- so the sandboxed code judges live here and
are reached through bench_scoring. `ScoreResult` is imported from
rewards/scorers, so a LiveCodeBench row has the same shape as every other
row.

Generated code is UNTRUSTED. It runs in a separate process with rlimits on
address space, CPU and file size, a per-test alarm, an overall wall-clock
timeout, a scratch cwd, and the usual destructive entry points removed from
the child (the HumanEval `reliability_guard` approach). That is the standard
bar for a code benchmark; it is not a security boundary against deliberately
hostile code, and nothing here should be pointed at anything but benchmark
output.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
if str(REPO / "rewards") not in sys.path:
    sys.path.insert(0, str(REPO / "rewards"))

from scorers import ScoreResult  # noqa: E402

DEFAULT_PER_TEST_TIMEOUT = 6.0
DEFAULT_TOTAL_TIMEOUT = 90.0

_FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


_FENCE_OPEN = re.compile(r"```(?:python|py|python3)?[ \t]*\n", re.IGNORECASE)


def extract_code(text: str) -> str:
    """The program the router put in <answer>.

    Last fenced block wins: a model that explains and then gives the final
    program puts the program last, and taking the first block would grade the
    illustration instead of the answer.

    A reply cut off by max_tokens leaves an OPEN fence and no closing one. The
    code-shaped fallback then returned the entire reply -- prose and all -- so
    the scorer reported `SyntaxError ... line 1` over a program-sized blob of
    commentary. Recovering the tail after
    the last opening fence grades the truncated program, which is what the model
    actually produced, instead of grading its commentary.
    """
    if not text:
        return ""
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()
    opens = list(_FENCE_OPEN.finditer(text))
    if opens:
        return text[opens[-1].end():].strip()
    stripped = text.strip()
    # unfenced but code-shaped (def/class/import at the start of some line)
    if re.search(r"^\s*(def |class |import |from |print\()", stripped, re.MULTILINE):
        return stripped
    return ""


# The child. Kept as a source string so the scorer is one importable file and
# the subprocess has no dependency on this package being importable.
_RUNNER = r'''
import base64, io, json, os, resource, signal, sys

payload = json.loads(base64.b64decode(sys.stdin.buffer.read()).decode())
code = payload["code"]
tests = payload["tests"]
test_type = payload["test_type"]
func_name = payload.get("func_name")
per_test_timeout = payload["per_test_timeout"]

resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (int(payload["total_timeout"]), int(payload["total_timeout"]) + 5))
resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
sys.setrecursionlimit(100000)

# destructive entry points out of reach of generated code
import shutil, subprocess as _sp
os.system = None
os.remove = os.unlink = os.rmdir = os.removedirs = None
os.rename = os.renames = os.replace = None
os.kill = os.killpg = None
shutil.rmtree = shutil.move = shutil.chown = None
_sp.Popen = _sp.run = _sp.call = _sp.check_call = _sp.check_output = None
__builtins__.__dict__["help"] = None


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def norm_lines(text):
    return [line.rstrip() for line in str(text).strip().splitlines()]


def norm_value(value):
    """List/tuple-insensitive, float-tolerant comparison key."""
    if isinstance(value, (list, tuple)):
        return [norm_value(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def make_stdin(text):
    """A stdin that also has .buffer.

    io.StringIO does NOT, and `sys.stdin.buffer.read()` is the standard fast-read
    idiom in competitive Python. Substituting a StringIO turned every such
    program into `AttributeError: '_io.StringIO' object has no attribute
    'buffer'` and scored it 0. On the full slice every such row was judged
    wrong by the harness rather than by the program.
    io.TextIOWrapper carries the binary buffer it
    wraps as .buffer, so both idioms work off the same bytes.
    """
    return io.TextIOWrapper(io.BytesIO(text.encode()), encoding="utf-8", newline="")


def make_stdout():
    """(text_stream, read_value) -- same reason as make_stdin, for
    sys.stdout.buffer.write()."""
    raw = io.BytesIO()
    text = io.TextIOWrapper(raw, encoding="utf-8", newline="", write_through=True)

    def value():
        text.flush()
        return raw.getvalue().decode("utf-8", errors="replace")

    return text, value


def run_stdin(code, test):
    """Fresh namespace per test: a solution that caches globals must not be
    graded on state left by the previous case.

    Returns (ok, actual_stdout). The actual output is what a public-test
    feedback message needs; the verdict itself is unchanged.
    """
    out_stream, out_value = make_stdout()
    real_stdin, real_stdout = sys.stdin, sys.stdout
    sys.stdin = make_stdin(test["input"])
    sys.stdout = out_stream
    try:
        namespace = {"__name__": "__main__"}
        try:
            exec(compile(code, "<solution>", "exec"), namespace)
        except SystemExit:
            pass
    finally:
        sys.stdin, sys.stdout = real_stdin, real_stdout
    actual = out_value()
    return norm_lines(actual) == norm_lines(test["output"]), actual


def build_functional(code):
    namespace = {"__name__": "__solution__"}
    prelude = ("import collections, heapq, itertools, math, functools, bisect, re, string\n"
               "from collections import *\n"
               "from typing import *\n")
    exec(compile(prelude + code, "<solution>", "exec"), namespace)
    return namespace


def run_functional(namespace, test, func_name):
    args = [json.loads(line) for line in test["input"].split("\n") if line.strip()]
    solution_cls = namespace.get("Solution")
    if solution_cls is None:
        return False, "<no Solution class>"
    method = getattr(solution_cls(), func_name)
    got = method(*args)
    expected = json.loads(test["output"])
    return norm_value(got) == norm_value(expected), json.dumps(got, default=str)


passed = 0
failed_at = None
error = None
actual = None
namespace = None
try:
    if test_type == "functional":
        signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
        namespace = build_functional(code)
        signal.setitimer(signal.ITIMER_REAL, 0)
    for index, test in enumerate(tests):
        signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
        try:
            if test_type == "functional":
                ok, got = run_functional(namespace, test, func_name)
            else:
                ok, got = run_stdin(code, test)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if not ok:
            failed_at = index
            actual = got
            break
        passed += 1
except Timeout:
    error = "timeout"
    failed_at = passed
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"[:300]
    failed_at = passed

print("__LCB_VERDICT__" + json.dumps({
    "passed": passed, "total": len(tests), "failed_at": failed_at, "error": error,
    "actual": (actual[:2000] if isinstance(actual, str) else actual),
}))
'''


def run_tests(code, tests, test_type, func_name=None,
              per_test_timeout=DEFAULT_PER_TEST_TIMEOUT,
              total_timeout=DEFAULT_TOTAL_TIMEOUT):
    """Run one program against its test cases in a child process."""
    if not code:
        return {"passed": 0, "total": len(tests), "failed_at": 0, "error": "no_code",
                "actual": None}
    payload = base64_payload({
        "code": code, "tests": tests, "test_type": test_type,
        "func_name": func_name, "per_test_timeout": per_test_timeout,
        "total_timeout": total_timeout,
    })
    with tempfile.TemporaryDirectory(prefix="lcb_") as scratch:
        runner = Path(scratch) / "runner.py"
        runner.write_text(_RUNNER)
        try:
            proc = subprocess.run(
                [sys.executable, str(runner)], input=payload, cwd=scratch,
                capture_output=True, timeout=total_timeout,
                env={"PATH": os.environ.get("PATH", ""), "HOME": scratch,
                     "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1",
                     "OMP_NUM_THREADS": "1"})
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": len(tests), "failed_at": 0,
                    "error": "process_timeout", "actual": None}
    out = proc.stdout.decode(errors="replace")
    marker = out.rfind("__LCB_VERDICT__")
    if marker == -1:
        stderr = proc.stderr.decode(errors="replace")[-300:]
        return {"passed": 0, "total": len(tests), "failed_at": 0,
                "error": f"child_crash: {stderr}", "actual": None}
    return json.loads(out[marker + len("__LCB_VERDICT__"):].splitlines()[0])


def base64_payload(obj) -> bytes:
    import base64 as _b64
    return _b64.b64encode(json.dumps(obj).encode())


def score_livecodebench(response_text, sample, **kwargs) -> ScoreResult:
    """Same return shape as rewards/scorers.score_answer, so a LiveCodeBench row
    slots into the wrapper's row schema unchanged. Binary, like every non-QA
    scorer there: score/em/f1 all carry the same value."""
    code = extract_code(response_text)
    verdict = run_tests(
        code, sample["tests"], sample["test_type"], sample.get("func_name"),
        per_test_timeout=kwargs.get("per_test_timeout", DEFAULT_PER_TEST_TIMEOUT),
        total_timeout=kwargs.get("total_timeout", DEFAULT_TOTAL_TIMEOUT))
    solved = float(verdict["passed"] == verdict["total"] and verdict["total"] > 0)
    return ScoreResult(
        score=solved, scorer_name="livecodebench",
        normalized_prediction=code[:2000], matched_gold=None,
        em=solved, f1=solved,
        detail={"passed": verdict["passed"], "total": verdict["total"],
                "failed_at": verdict["failed_at"], "error": verdict["error"],
                "code_chars": len(code)})
