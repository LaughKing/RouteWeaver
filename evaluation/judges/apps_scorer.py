"""APPS call-based (functional) execution scorer.

APPS's functional format differs from LiveCodeBench's, which is why the LCB
run_functional cannot grade it (it splits `input` on newlines and json.loads
each line; APPS stores `inputs[k]` as an already-parsed arg LIST, with string
args double-JSON-encoded, and `outputs[k]` as a raw value that is sometimes
list-wrapped). This module implements the APPS decoding:

  * args = [decode_arg(x) for x in inputs[k]]   (splatted into the method)
  * decode_arg: json.loads a str arg to strip the "..." encoding, else as-is
  * the callable is Solution().<fn_name> OR a module-level def <fn_name>
  * flexible_match tolerates list-wrapping and list/tuple equivalence on output

Each candidate is run in a child process (base64 JSON payload -> runner), so a
model program that hangs or segfaults cannot take the harness down. The SAME
match function grades the reference solution at build time (build_apps.py) and
the worker at eval time, so "reference passes" guarantees the row is gradable.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_PER_TEST_TIMEOUT = 6.0
DEFAULT_TOTAL_TIMEOUT = 30.0

_RUNNER = r'''
import base64, json, signal, sys, io, contextlib

PRELUDE = ("import collections, heapq, itertools, math, functools, bisect, re, string\n"
           "from collections import *\n"
           "from typing import *\n")

class Timeout(Exception):
    pass

def _handler(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, _handler)

def decode_arg(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return a
    return a

def norm(v):
    try:
        return json.loads(json.dumps(v, default=str))
    except Exception:
        return v

def fmatch(got, exp):
    g, e = norm(got), norm(exp)
    if g == e:
        return True
    if isinstance(e, list) and len(e) == 1 and norm(g) == norm(e[0]):
        return True
    if isinstance(g, list) and len(g) == 1 and norm(g[0]) == e:
        return True
    try:
        if list(g) == list(e):
            return True
    except Exception:
        pass
    return False

def build(code):
    ns = {"__name__": "__solution__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(PRELUDE + code, "<solution>", "exec"), ns)
    return ns

def get_method(ns, fn):
    cls = ns.get("Solution")
    if cls is not None:
        try:
            m = getattr(cls(), fn, None)
            if callable(m):
                return m
        except Exception:
            pass
    f = ns.get(fn)
    if callable(f):
        return f
    return None

payload = json.loads(base64.b64decode(sys.stdin.buffer.read()).decode())
code = payload["code"]
tests = payload["tests"]
fn = payload["func_name"]
per_test = payload["per_test_timeout"]

passed = 0
failed_at = None
error = None
try:
    signal.setitimer(signal.ITIMER_REAL, per_test)
    ns = build(code)
    signal.setitimer(signal.ITIMER_REAL, 0)
    m = get_method(ns, fn)
    if m is None:
        error = "no_callable:" + str(fn)
    else:
        for index, t in enumerate(tests):
            args = [decode_arg(x) for x in t["input"]]
            try:
                signal.setitimer(signal.ITIMER_REAL, per_test)
                with contextlib.redirect_stdout(io.StringIO()):
                    got = m(*args)
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Timeout:
                error = "timeout"; failed_at = index; break
            except BaseException as exc:  # noqa
                error = ("run_err: %s: %s" % (type(exc).__name__, exc))[:200]
                failed_at = index; break
            if not fmatch(got, t["output"]):
                failed_at = index; break
            passed += 1
except Timeout:
    error = "build_timeout"; failed_at = 0
except BaseException as exc:  # noqa
    error = ("build_err: %s: %s" % (type(exc).__name__, exc))[:200]; failed_at = 0

print("__APPS_VERDICT__" + json.dumps({
    "passed": passed, "total": len(tests), "failed_at": failed_at, "error": error}))
'''


def run_functional_apps(code, tests, func_name,
                        per_test_timeout=DEFAULT_PER_TEST_TIMEOUT,
                        total_timeout=DEFAULT_TOTAL_TIMEOUT):
    """tests: list of {"input": [args...], "output": expected}. Returns
    {passed, total, failed_at, error}."""
    if not code:
        return {"passed": 0, "total": len(tests), "failed_at": 0, "error": "no_code"}
    payload = base64.b64encode(json.dumps({
        "code": code, "tests": tests, "func_name": func_name,
        "per_test_timeout": per_test_timeout}).encode())
    with tempfile.TemporaryDirectory(prefix="apps_") as scratch:
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
                    "error": "process_timeout"}
    out = proc.stdout.decode(errors="replace")
    marker = out.rfind("__APPS_VERDICT__")
    if marker == -1:
        stderr = proc.stderr.decode(errors="replace")[-200:]
        return {"passed": 0, "total": len(tests), "failed_at": 0,
                "error": "child_crash: " + stderr}
    return json.loads(out[marker + len("__APPS_VERDICT__"):].splitlines()[0])
