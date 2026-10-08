"""Per-benchmark scoring on the EXTRACTED answer, one dispatch for all modes.

Every mode's final answer goes through the same shared extractor upstream
(extract_answer: <answer> tag -> \\boxed -> last meaningful line), so what
arrives here is a short answer string. This module maps it onto each
benchmark's official-style verdict plus normalized EM and token F1.

SimpleQA's official metric is an LLM grader; here `correct` is boundary-aware
sub-EM (strict for paraphrase), reported as such -- absolute numbers are a lower
bound, the mode COMPARISON is the measurement. `not_attempted` = empty answer.
"""
import os
import re
import sys

from common import HERE, math_scorer
import qa_em_stance as qa_em          # SQuAD-style normalizer + token F1

# The APPS/LCB sandbox judges are vendored HERE, not imported from elsewhere: a
# scorer the training reward depends on cannot live outside the repository that
# trains against it.
_LCB_DIR = HERE.parent / "evaluation/judges"
assert (_LCB_DIR / "lcb_scorer.py").exists() and (_LCB_DIR / "apps_scorer.py").exists(), \
    f"code judges missing from {_LCB_DIR}"


def _lcb(answer, meta):
    """LiveCodeBench pass@1: extract the last fenced program from the answer,
    run it against the row's tests in the evaluation/judges sandbox,
    all-or-nothing. gold is decorative; the tests in meta are the judge."""
    if str(_LCB_DIR) not in sys.path:
        sys.path.insert(0, str(_LCB_DIR))
    from lcb_scorer import score_livecodebench
    s = score_livecodebench(str(answer or ""), meta)
    ok = float(s.score)
    return {"correct": bool(ok), "em": ok, "f1": ok,
            "detail": s.detail, "code_chars": s.detail.get("code_chars", 0),
            "tests_passed": s.detail.get("passed"),
            "tests_total": s.detail.get("total")}

def _stdin_lines_to_text(meta):
    """APPS stores a stdin case either as one string or as a LIST OF LINES.

    215 of the 343 stdin rows in data use the list form
    (e.g. input ["2", "HELLO", "HELL"], output ["BJQEI", "BJQE"]). lcb_scorer's
    stdin runner feeds test["input"] straight into a StringIO, so a list makes
    the child raise AttributeError -- and that surfaces as sandbox_error, i.e.
    infra_failed, which would silently drop ~10% of the CODE domain out of the
    gradient rather than scoring it. Joining on newline is the only reading:
    the elements ARE the lines.

    The manifest is not touched; this normalises on the way into the sandbox.
    Rows already carrying strings pass through unchanged.
    """
    tests = meta.get("tests") or []
    if not any(isinstance(t.get(k), list) for t in tests for k in ("input", "output")):
        return meta
    fixed = [{**t,
              "input": "\n".join(map(str, t["input"])) if isinstance(t.get("input"), list)
                       else t.get("input"),
              "output": "\n".join(map(str, t["output"])) if isinstance(t.get("output"), list)
                        else t.get("output")}
             for t in tests]
    return {**meta, "tests": fixed}


def _apps_functional(answer, meta):
    """APPS call-based pass@1. Same shape as _lcb, different decoder.

    APPS stores inputs[k] as an already-parsed ARG LIST (string args
    double-JSON-encoded) and outputs[k] as a raw value that is sometimes
    list-wrapped; LiveCodeBench's run_functional splits on newlines and
    json.loads each line, so it cannot grade these. apps_scorer implements the
    APPS decoding and is the SAME function build_routeweaver_splits used to decide
    which rows are gradable at all (reference solution must pass), so
    "gradable at build time" and "graded at train time" cannot disagree.
    """
    if str(_LCB_DIR) not in sys.path:
        sys.path.insert(0, str(_LCB_DIR))
    from apps_scorer import run_functional_apps
    from lcb_scorer import extract_code
    code = extract_code(str(answer or ""))
    v = run_functional_apps(code, meta["tests"], meta.get("func_name"))
    ok = float(v["passed"] == v["total"] and v["total"] > 0)
    return {"correct": bool(ok), "em": ok, "f1": ok,
            "detail": {"passed": v["passed"], "total": v["total"],
                       "failed_at": v["failed_at"], "error": v["error"],
                       "code_chars": len(code)},
            "code_chars": len(code),
            "tests_passed": v["passed"], "tests_total": v["total"]}


_LETTER = re.compile(r"\b([A-J])\b")


def _norm(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"^(the\s+)?(final\s+)?answer\s*(is|:)?\s*", "", text)
    text = re.sub(r"[\*\.\s]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _f1(pred, gold):
    p = [t for t in re.split(r"[^0-9a-z]+", _norm(pred)) if t]
    g = [t for t in re.split(r"[^0-9a-z]+", _norm(gold)) if t]
    if not p or not g:
        return float(p == g)
    pool, common = list(g), 0
    for t in p:
        if t in pool:
            pool.remove(t)
            common += 1
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def _letter(answer, gold):
    hits = _LETTER.findall(str(answer or "").upper())
    picked = hits[-1] if hits else None
    ok = float(picked == str(gold[0]).strip().upper())
    return {"correct": bool(ok), "em": ok, "f1": ok,
            "detail": f"picked={picked}", "picked": picked}


def _satbench(answer, gold):
    text = str(answer or "").upper()
    has_unsat = re.search(r"\bUNSAT(ISFIABLE)?\b", text) is not None
    has_sat = re.search(r"(?<!UN)\bSAT(ISFIABLE)?\b", text) is not None
    if has_unsat and not has_sat:
        picked = "UNSAT"
    elif has_sat and not has_unsat:
        picked = "SAT"
    elif has_unsat and has_sat:                       # trust the last mention
        lu = max(m.end() for m in re.finditer(r"\bUNSAT(ISFIABLE)?\b", text))
        ls = max(m.end() for m in re.finditer(r"(?<!UN)\bSAT(ISFIABLE)?\b", text))
        picked = "UNSAT" if lu > ls else "SAT"
    else:
        picked = {"YES": "SAT", "NO": "UNSAT", "TRUE": "SAT",
                  "FALSE": "UNSAT"}.get(text.strip().rstrip("."), None)
    ok = float(picked == gold[0])
    return {"correct": bool(ok), "em": ok, "f1": ok,
            "detail": f"picked={picked}", "picked": picked}


def _em_short(answer, gold, containment=False, both_ways=False):
    """`both_ways` accepts the answer when it is CONTAINED IN a gold string as
    well as when it contains one -- "Catholic" against gold "catholic church",
    "Muslim" against "islam". Enabled for qa_subem (PopQA) only, where gold is
    by construction a list of acceptable surface forms. It must not be turned on
    elsewhere: against a long gold, a one-word answer that happens to sit inside
    it would pass. Word boundaries are required in both directions so "red"
    cannot match "predator".
    """
    np_ = _norm(answer)
    ok_em = float(any(np_ == _norm(g) for g in gold))
    sub = ok_em
    if not sub and containment and np_:
        for g in gold:
            ng = _norm(g)
            if not ng:
                continue
            if re.search(rf"(?<![0-9a-z]){re.escape(ng)}(?![0-9a-z])", np_):
                sub = 1.0
                break
            if both_ways and re.search(
                    rf"(?<![0-9a-z]){re.escape(np_)}(?![0-9a-z])", ng):
                sub = 1.0
                break
    f1 = max(_f1(answer, g) for g in gold)
    return {"correct": bool(sub), "em": ok_em, "f1": f1,
            "sub_em": sub, "detail": "em_short",
            "not_attempted": not str(answer or "").strip()}


def _triviaqa(answer, gold):
    """TriviaQA: alias-aware BINARY exact match. User decisi.

    reward = max over the gold alias list of normalized_exact_match(pred, alias),
    so it is 0 or 1 and nothing else. Deliberately NOT the containment sub-EM the
    short-answer scorer uses: containment credits "Paris, Texas" for a gold of
    "Paris", which is the loophole the earlier EM line was measured hacking.
    No LLM grader either -- this stays deterministic and offline.

    Normalization is `qa_em_stance.normalize_answer`, the standard SQuAD/TriviaQA
    one already in the repo (lowercase, strip punctuation, drop a/an/the, collapse
    whitespace) and applied to the PREDICTION and to every ALIAS identically. It
    is not extended: a prediction that buries the answer in prose scores 0, which
    is what binary EM means.

    EM and token F1 are both reported for logging (`em` is the reward bit, `f1`
    is the standard TriviaQA F1 and is never read by reward.py).
    """
    pred = qa_em.normalize_answer(str(answer or ""))
    aliases = [str(g) for g in (gold or [])]
    norms = [qa_em.normalize_answer(a) for a in aliases]
    em = 1.0 if pred and pred in norms else 0.0
    f1 = max((qa_em.f1_score(str(answer or ""), a) for a in aliases), default=0.0)
    matched = next((a for a, n in zip(aliases, norms) if n == pred and pred), None)
    return {"correct": bool(em), "em": em, "f1": float(f1),
            "detail": "alias_binary_em", "n_aliases": len(aliases),
            "matched_alias": matched,
            "normalized_prediction": pred,
            "not_attempted": not str(answer or "").strip()}


_CODE_FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.S | re.I)
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)


def _code_candidates(text):
    """The program candidates in ONE piece of final system output.

    Rule (decided, replaces the 08-26 whole-trajectory union):

        1. the body of the LAST <answer>...</answer> in `text`, if any
        2. the LAST fenced code block in `text`, if any
        3. ONLY when neither exists: `text` itself (a bare program with no
           fence and no tag, e.g. an executor reply judged in-loop)

    `text` must be a FINAL output -- the router's last turn, or the terminal
    summarizer's reply -- never the whole trajectory. A correct program that
    an executor wrote mid-trajectory and the summarizer failed to carry into
    its own output earns nothing; see score_final_output.
    """
    text = str(text or "")
    out = []
    tags = _ANSWER_TAG.findall(text)
    if tags:
        out.append(tags[-1].strip())
    blocks = _CODE_FENCE.findall(text)
    if blocks:
        out.append(blocks[-1])
    if not out:
        out.append(text)
    seen, uniq = set(), []
    for c in out:
        c = (c or "").strip()
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _best_code_verdict(judge, answer, meta):
    best = None
    for cand in _code_candidates(answer):
        v = judge(cand, meta)
        if v.get("correct"):
            return v
        best = best or v
    return best or {"correct": False, "em": 0.0, "f1": 0.0, "detail": "no_candidate"}


def score(dataset, answer, gold, meta=None):
    if dataset in ("omni_math", "omni_math_d36x"):
        s = math_scorer.score(answer, gold)
        return {"correct": bool(s["equiv_acc"]), "em": s["em"], "f1": s["f1"],
                "equiv_acc": s["equiv_acc"], "detail": s["detail"]}
    if dataset in ("supergpqa", "longbench_v2"):
        return _letter(answer, gold)
    if dataset == "satbench":
        return _satbench(answer, gold)
    if dataset == "triviaqa":
        return _triviaqa(answer, gold)
    if dataset in ("simpleqa_verified", "musique_100"):
        return _em_short(answer, gold, containment=True)
    if dataset == "bbeh_mini":
        return _em_short(answer, gold, containment=False)
    if dataset in ("livecodebench", "taco_mm"):
        return _lcb(answer, meta)

    # ---- data evaluator names -------------------------
    # That manifest keys rows by `terminator_or_evaluator` rather than by
    # benchmark name, because one dataset (APPS) needs two different judges.
    # Every branch below reuses a verdict function already defined above; only
    # apps_functional is new, and it delegates to the prelim sandbox.
    if dataset == "math":                       # math_l2-5
        s = math_scorer.score(answer, gold)
        return {"correct": bool(s["equiv_acc"]), "em": s["em"], "f1": s["f1"],
                "equiv_acc": s["equiv_acc"], "detail": s["detail"]}
    if dataset == "mc_letter":                  # SuperGPQA
        return _letter(answer, gold)
    if dataset == "binary_sat":                 # SATBench
        return _satbench(answer, gold)
    if dataset == "qa_subem":                   # PopQA
        # ONE direction only: a normalized gold alias
        # must be contained in the normalized prediction. "diplomat and
        # statesman" passes gold "diplomat"; "Catholic" FAILS gold "Catholic
        # Church". The reverse direction was tried and rejected: it credits
        # a generic answer against a specific gold.
        return _em_short(answer, gold, containment=True, both_ways=False)
    if dataset == "stdin":                      # APPS, stdin-graded
        m = _stdin_lines_to_text(meta)
        return _best_code_verdict(lambda c, mm: _lcb(c, mm), answer, m)
    if dataset == "apps_functional":            # APPS, call-graded
        return _best_code_verdict(_apps_functional, answer, meta)
    raise KeyError(dataset)


# =========================================================================
# THE ONE ENTRY POINT for a trajectory verdict.
#
# Training reward (reward_manager.py), the zero-update checkpoint
# eval (same manager under VAL_ONLY), the offline dump rescorer
# (data/rescore_dump.py) and the single-worker baselines
# (the baseline harness) all call THIS and nothing else, so
# the four paths cannot disagree.
#
# What counts as "final system output":
#   router_text     the router's LAST assistant turn -- the one that carries
#                   its <answer>. Its answer is the last <answer> body; a
#                   turn with no tag has submitted nothing (the policy's
#                   contract is the tag, and a last-line fallback on a
#                   <route> turn would let a sub-task instruction that quotes
#                   the gold pass RECALL containment).
#   terminal_reply  the terminal worker reply that IS the system's answer in
#                   that mode: the summarizer node's reply for agentic, the
#                   single worker's reply for a worker-only baseline. None
#                   for single/multi router rollouts. Its answer goes through
#                   worker_client.extract_answer (<answer> -> \\boxed -> last
#                   line), the same extractor every baseline table used.
# What is NOT final output: planner / executor / verifier replies, anything
# the router observed mid-trajectory. A program that passed the hidden tests
# inside an executor node and never reached the summarizer's reply or the
# router's <answer> is not credited.
#
# CODE: candidates = union of _code_candidates over the final outputs
# (router turn first, then terminal reply); the first that passes its real
# test payload wins. Every other domain: each final output's extracted answer
# is judged; correct if any passes.
# =========================================================================
CODE_DATASETS = ("stdin", "apps_functional", "livecodebench", "taco_mm")


def _router_answer(router_text):
    tags = _ANSWER_TAG.findall(str(router_text or ""))
    return tags[-1].strip() if tags else ""


def _reply_answer(reply):
    from worker_client import extract_answer
    ans, _how = extract_answer(str(reply or ""))
    return ans


# CODE scoring policy: "union" also credits a
# correct program found in any WORKER reply the router actually called
# (worker_replies), after the router's own final turn and the summarizer.
# "strict" (08-27..08-30) judged only the router's final turn + summarizer.
CODE_SCORER = os.environ.get("ROUTEWEAVER_CODE_SCORER", "union")


def score_final_output(dataset, gold, meta=None, router_text=None,
                       terminal_reply=None, worker_replies=None):
    """Verdict dict for one trajectory. Keys: correct, em, f1, source
    ("router" | "terminal" | "worker" | ""), router_correct, terminal_correct, detail."""
    empty = {"correct": False, "em": 0.0, "f1": 0.0, "source": "",
             "router_correct": False, "terminal_correct": False,
             "detail": "no_final_output"}
    if dataset in CODE_DATASETS:
        cands = []
        if router_text:
            cands += [("router", c) for c in _code_candidates(router_text)]
        if terminal_reply:
            cands += [("terminal", c) for c in _code_candidates(terminal_reply)]
        if CODE_SCORER == "union" and worker_replies:
            seen = {c for _, c in cands}
            for reply in worker_replies:
                for c in _code_candidates(str(reply or "")):
                    if c not in seen and "```" in str(reply or ""):
                        cands.append(("worker", c)); seen.add(c)
        first = None
        for src, cand in cands:
            v = score(dataset, cand, gold, meta)
            if v.get("correct"):
                v = dict(v, source=src, router_correct=(src == "router"),
                         terminal_correct=(src == "terminal"))
                return v
            first = first or dict(v, source="", router_correct=False,
                                  terminal_correct=False)
        return first or empty
    r_ans = _router_answer(router_text) if router_text else ""
    t_ans = _reply_answer(terminal_reply) if terminal_reply else ""
    v_r = score(dataset, r_ans, gold, meta) if r_ans.strip() else None
    v_t = score(dataset, t_ans, gold, meta) if t_ans.strip() else None
    rc = bool(v_r and v_r.get("correct"))
    tc = bool(v_t and v_t.get("correct"))
    pick = v_r if rc else (v_t if tc else (v_r or v_t))
    if pick is None:
        return empty
    return dict(pick, source=("router" if rc else ("terminal" if tc else "")),
                router_correct=rc, terminal_correct=tc)
