"""Three metrics for an open-form math answer, primary first.

equiv_acc  mathematical equivalence, via verl's prime_math. `4.5`, `9/2` and
           `\frac{9}{2}` are one answer, and on Omni-MATH they occur in all
           three shapes, so this is THE metric.
EM         normalised exact match on the answer strings. Secondary and known to
           be pessimistic for exactly the reason above.
F1         token-level overlap. Only informative for the answers that are short
           phrases rather than expressions ("Yes, such a partition is possible").

IMPORT ORDER IS LOAD-BEARING. scorers.py takes its symbolic backend from
verl.utils.reward_score.prime_math. Two verls are importable: the installed 0.8
in site-packages, which has prime_math, and REPO/verl, the frozen the original fork, whose
reward_score package holds only countdown/gsm8k/math/multiply/qa_em. If REPO
lands on sys.path first, `import verl` resolves to the fork, prime_math raises
ImportError, and scorers falls back to _MATH_BACKEND = "string_only" SILENTLY --
which scored `4.5` wrong against a gold of `\frac{9}{2}` for a whole day. The
assertion below is the tripwire; it must never be downgraded to a warning.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]

_shadow = {"", ".", str(REPO)}
sys.path[:] = [p for p in sys.path if p not in _shadow]
for _p in (str(REPO / "rewards"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scorers as _v1_scorers                      # noqa: E402
import qa_em_stance as qa_em                       # noqa: E402

MATH_BACKEND = _v1_scorers._MATH_BACKEND
assert MATH_BACKEND == "prime_math", (
    f"symbolic math backend unavailable ({MATH_BACKEND}); every equivalence "
    f"check would silently degrade to string comparison. Run this with the "
    f"interpreter that has verl 0.8 installed (routeweaver).")

if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

# Cosmetic LaTeX that never changes a value. Sign is NOT in here: qa_em's
# normaliser strips punctuation, minus included, and credited '6' against a gold
# of '-6' until a guard was added downstream.
#
# \( \) and \[ \] earn their place. Workers wrap inline maths in them and the
# Omni-MATH gold strings do not, so `\(\frac{1}{2}\)` and `\frac{1}{2}` compared
# unequal: 10 of the 13 rows where equiv_acc was 1 and EM was 0 differed by
# nothing else, which is 10 points of EM lost to a delimiter. prime_math sees
# through it; a string comparison has to be told.
_TRIVIA = re.compile(r"\\left|\\right|\\!|\\,|\\;|\\ |\\\(|\\\)|\\\[|\\\]|\$|\s+")
# `\%` is NOT cosmetic. Deleting it turns a gold of "80 \%" into "80" while a
# worker's "80%" keeps its sign, so the two still differ -- the notation has to be
# unified, not discarded.
_PERCENT = re.compile(r"\\%")
_WRAPPERS = [(re.compile(r"^\\boxed\{(.*)\}$", re.S), r"\1"),
             (re.compile(r"^\\text\{(.*)\}$", re.S), r"\1"),
             (re.compile(r"^\((.*)\)$", re.S), r"\1"),
             (re.compile(r"^\{(.*)\}$", re.S), r"\1")]

# Four shapes that recur in worker answers, all of them a right answer written
# differently rather than a wrong one.
#
#   "362,880"      vs "362880"      thousands separators
#   "-1 + (-1) = -2" vs "-2"        the worker showed the last step
#   "9/84 = 3/28"  vs "3/28"        same, unreduced then reduced
#   "\pm 3i"       vs "3i,-3i"      one notation for a two-element set
#
# Each is applied to BOTH sides by normalize(), so it can only ever make two
# spellings of one value agree -- none of them can turn distinct values equal.
# Thousands separators only collapse between digits, so "1,2" (a pair) survives.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d\d\d(?!\d))")
# keep only what follows the last top-level "=", so a shown derivation reduces
# to its result; guarded to a single trailing step so "x=1,y=2" is untouched.
_TRAILING_EQ = re.compile(r"^[^=]{1,80}=\s*(.+)$", re.S)
_PM = re.compile(r"^\\?pm\s*(.+)$", re.S)


def normalize(text):
    """Sign-preserving normalisation used for EM and for the equivalence guard."""
    s = str(text or "").strip()
    for _ in range(3):
        before = s
        for pattern, repl in _WRAPPERS:
            s = pattern.sub(repl, s.strip())
        if s == before:
            break
    s = _PERCENT.sub("%", s)
    s = _TRIVIA.sub("", s)
    s = s.rstrip(".").strip()
    s = _THOUSANDS.sub("", s)
    m = _TRAILING_EQ.match(s)
    if m and "=" not in m.group(1):
        s = m.group(1).strip()
    m = _PM.match(s)
    if m:
        body = m.group(1).strip()
        s = f"{body},-{body}"
    return s.lower()


def _tokens(text):
    return [t for t in re.split(r"[^0-9a-z]+", str(text or "").lower()) if t]


def f1(pred, gold):
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = 0
    pool = list(g)
    for t in p:
        if t in pool:
            pool.remove(t)
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def equivalent(pred, gold):
    """prime_math equivalence, with a sign guard on the string-only fallback.

    score_math's last resort compares qa_em.normalize_answer(pred) with
    qa_em.normalize_answer(gold) (scorers.py:319-322). That normaliser drops
    punctuation, so it treats '-6' and '6' as the same string. Any verdict that
    came from that branch is re-checked with a normaliser that keeps the sign.
    """
    result = _v1_scorers.score_answer("math_l35", f"<answer>{pred}</answer>",
                                      list(gold))
    em, detail = float(result.em), result.detail
    if em == 1.0 and detail == "string":
        candidate = result.normalized_prediction or pred
        if not any(normalize(candidate) == normalize(g) for g in gold):
            return 0.0, "string match rejected by sign guard"
    return em, detail


# ---- equivalence gaps prime_math leaves open --------------------
# Measured on the probe set: prime_math accepts "362,880" vs "362880",
# "(A)" vs "A", "-1 + (-1) = -2" vs "-2", LaTeX delimiters, "\\boxed{}",
# "5.0" vs "5", "0.5" vs "\\frac12"; it REJECTS an unreduced fraction
# ("9/84" vs "3/28"), a "\\pm" form against its two-element set, and a
# multi-answer list in a different order. The three helpers below close
# exactly those, on top of prime_math, never instead of it.
from fractions import Fraction as _Fraction

_SIMPLE_FRAC = re.compile(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$")
_LATEX_FRAC = re.compile(r"^\s*([+-]?)\\d?frac\{([+-]?\d+)\}\{([+-]?\d+)\}\s*$")
_SIMPLE_DEC = re.compile(r"^\s*[+-]?(\d+\.?\d*|\.\d+)\s*$")


def _rational(text):
    """Fraction for a plain a/b, \\frac{a}{b}, integer or decimal; else None."""
    s = normalize(text)
    m = _SIMPLE_FRAC.match(s)
    if m and int(m.group(2)) != 0:
        return _Fraction(int(m.group(1)), int(m.group(2)))
    m = _LATEX_FRAC.match(s)
    if m and int(m.group(3)) != 0:
        f = _Fraction(int(m.group(2)), int(m.group(3)))
        return -f if m.group(1) == "-" else f
    if _SIMPLE_DEC.match(s):
        try:
            return _Fraction(s)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _split_top(text):
    """Top-level comma split (depth-aware for () [] {}), for multi-answer
    lists such as "3i, -3i" or "1/2, 3". A list that is itself wrapped in
    one pair of brackets -- an ordered tuple "(1,2)" -- is NOT split."""
    s = str(text or "").strip()
    if not s:
        return []
    if re.match(r"^[\(\[\{].*[\)\]\}]$", s, re.S):
        depth, closes_early = 0, False
        for i, ch in enumerate(s):
            depth += ch in "([{"
            depth -= ch in ")]}"
            if depth == 0 and i < len(s) - 1:
                closes_early = True
                break
        if not closes_early:
            return [s]
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _expand_pm(text):
    """"\\pm 3i" -> ["3i", "-3i"]; "2 \\pm \\sqrt{3}" -> ["2+\\sqrt{3}", "2-\\sqrt{3}"]."""
    s = str(text or "").strip()
    if "\\pm" not in s and "±" not in s:
        return None
    s = s.replace("±", "\\pm")
    if s.count("\\pm") != 1:
        return None
    left, right = s.split("\\pm")
    left, right = left.strip(), right.strip()
    if not right:
        return None
    if not left:
        return [right, f"-{right}"]
    return [f"{left}+{right}", f"{left}-{right}"]


_ARITH = re.compile(r"^[0-9+\-*/^().\s]+$")


def _arith_value(text):
    """Exact value of a bare arithmetic expression such as "-1 + (-1)" or
    "3*(4+1)/2"; None for anything that is not one. sympy, rational only."""
    s = normalize(text).replace("^", "**")
    if not _ARITH.match(s) or not re.search(r"[+\-*/]", s):
        return None
    try:
        import sympy
        v = sympy.nsimplify(sympy.sympify(s, rational=True))
        return v if v.is_Rational else None
    except Exception:                                              # noqa: BLE001
        return None


def _value(text):
    r = _rational(text)
    if r is not None:
        return _Fraction(r)
    v = _arith_value(text)
    return _Fraction(int(v.p), int(v.q)) if v is not None else None


def _single_equiv(pred, gold_item):
    eq, _ = equivalent(pred, [gold_item])
    if eq:
        return True
    a, b = _value(pred), _value(gold_item)
    return a is not None and b is not None and a == b


def _set_equiv(pred, gold_item):
    """Order-insensitive match of two multi-answer lists / \\pm forms."""
    p_items = _expand_pm(pred) or _split_top(pred)
    g_items = _expand_pm(gold_item) or _split_top(gold_item)
    if len(p_items) < 2 or len(p_items) != len(g_items):
        return False
    unused = list(g_items)
    for item in p_items:
        hit = next((g for g in unused if _single_equiv(item, g)), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True


def score(prediction, gold):
    """{equiv_acc, em, f1, detail} for one answer against the gold list."""
    gold = list(gold)
    if prediction is None or not str(prediction).strip():
        return {"equiv_acc": 0.0, "em": 0.0, "f1": 0.0, "detail": "no answer"}
    eq, detail = equivalent(prediction, gold)
    if not eq:
        for g in gold:
            if _value(prediction) is not None and _value(g) is not None \
                    and _value(prediction) == _value(g):
                eq, detail = 1.0, "rational"
                break
            if _set_equiv(prediction, g):
                eq, detail = 1.0, "set"
                break
    np_, ng = normalize(prediction), [normalize(g) for g in gold]
    em = float(any(np_ == g for g in ng))
    best_f1 = max(f1(prediction, g) for g in gold)
    return {"equiv_acc": eq, "em": em, "f1": best_f1, "detail": detail,
            "normalized_prediction": np_}


def answers_agree(a, b):
    """Do two candidate answers denote the same thing? Used for the A/B
    agreement gate, so it must be equivalence rather than string identity --
    otherwise `4.5` and `9/2` count as a disagreement and the verifier is woken
    up for nothing."""
    if not str(a or "").strip() or not str(b or "").strip():
        return False
    if normalize(a) == normalize(b):
        return True
    eq, _ = equivalent(a, [b])
    return bool(eq)
