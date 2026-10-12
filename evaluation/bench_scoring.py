"""Per-benchmark scoring on the EXTRACTED answer, one dispatch for all modes."""
import os
import re
import sys

from common import HERE, math_scorer
import qa_em_stance as qa_em          # SQuAD-style normalizer + token F1

_LCB_DIR = HERE.parent / "evaluation/judges"
assert (_LCB_DIR / "lcb_scorer.py").exists() and (_LCB_DIR / "apps_scorer.py").exists(), \
    f"code judges missing from {_LCB_DIR}"


def _lcb(answer, meta):
    """LiveCodeBench pass@1: run the last fenced program against the row's tests."""
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
    """APPS stores a stdin case either as one string or as a LIST OF LINES."""
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
    """APPS call-based pass@1. Same shape as _lcb, different decoder."""
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
    """Short-answer EM; `both_ways` also accepts an answer contained in a gold."""
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
    """TriviaQA: binary exact match against any gold alias."""
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
    """The program candidates in ONE piece of final system output."""
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

    if dataset == "math":                       # math_l2-5
        s = math_scorer.score(answer, gold)
        return {"correct": bool(s["equiv_acc"]), "em": s["em"], "f1": s["f1"],
                "equiv_acc": s["equiv_acc"], "detail": s["detail"]}
    if dataset == "mc_letter":                  # SuperGPQA
        return _letter(answer, gold)
    if dataset == "binary_sat":                 # SATBench
        return _satbench(answer, gold)
    if dataset == "qa_subem":                   # PopQA
        return _em_short(answer, gold, containment=True, both_ways=False)
    if dataset == "stdin":                      # APPS, stdin-graded
        m = _stdin_lines_to_text(meta)
        return _best_code_verdict(lambda c, mm: _lcb(c, mm), answer, m)
    if dataset == "apps_functional":            # APPS, call-graded
        return _best_code_verdict(_apps_functional, answer, meta)
    raise KeyError(dataset)


CODE_DATASETS = ("stdin", "apps_functional", "livecodebench", "taco_mm")


def _router_answer(router_text):
    tags = _ANSWER_TAG.findall(str(router_text or ""))
    return tags[-1].strip() if tags else ""


def _reply_answer(reply):
    from worker_client import extract_answer
    ans, _how = extract_answer(str(reply or ""))
    return ans


CODE_SCORER = os.environ.get("ROUTEWEAVER_CODE_SCORER", "union")


def score_final_output(dataset, gold, meta=None, router_text=None,
                       terminal_reply=None, worker_replies=None):
    """Verdict dict for one trajectory. Keys: correct, em, f1, source
    ("router" | "terminal" | "worker" | ""), router_correct, terminal_correct, detail.
    """
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
