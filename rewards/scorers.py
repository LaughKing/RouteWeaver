# Copyright 2026 RouteWeaver.
"""Per-dataset answer scoring for the Router-R1 reward.

WHY THIS EXISTS. Every dataset used to be graded by qa_em.em_check, which is
plain text normalization: lowercase, drop punctuation, drop articles. That is
right for short-entity QA and wrong for everything else, and the failure is
silent -- a CORRECT answer scores 0, the correctness gate in
compute_final_reward returns 0.0, and if all n rollouts of a prompt land there
the GRPO group is flat and contributes no gradient at all. Measured against the
real golds in data/router_v1_main:

    MATH   gold '(1,3)'   -> '(1, 3)' scored 0   (a space)
                          -> '\\boxed{28}' scored 0
    GSM8K  gold '135'     -> '135.00' scored 0   (the '.' is punctuation)
    ARC    gold ['B',...] -> 'The answer is B' scored 0

MATH + GSM8K are 3200 of the 9259 main-train rows, so a third of the corpus
was being graded by a rule that does not apply to it.

DISPATCH IS BY data_source, EXACTLY. That column already exists in every
parquet, so nothing has to be rebuilt. Matching is dict lookup on the exact
string -- no aliases, no substring containment. A data_source nobody registered
falls back to the QA scorer (the historical behaviour) and says so once in the
log; it never silently picks the math or ARC scorer, because being graded by
the wrong scorer is precisely the bug this module exists to fix.

QA PARITY IS A HARD CONSTRAINT. The qa scorer calls the same
qa_em.extract_solution / em_check / f1_score in the same order with the same
arguments; qa_em_stance.py is not touched. nq, triviaqa, hotpotqa, 2wiki and
musique score exactly what they scored before.
"""

import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional, Sequence

import qa_em_stance as qa_em

logger = logging.getLogger(__name__)

# ---- the symbolic backend -------------------------------------------------
#
# verl vendors a mature LaTeX-aware comparator (sympy + pylatexenc, both
# present in the environment). Reusing it beats copying a symbolic parser in here:
# it already handles \frac vs a/b, \boxed stripping, sqrt, decimals vs
# fractions and thousands separators, with its own timeout guard. If it ever
# becomes unavailable the math scorer degrades to the string/variant path
# rather than taking the whole reward down with it.
try:
    from verl.utils.reward_score.prime_math import compute_score as _prime_math_score
    _MATH_BACKEND = "prime_math"
except Exception:                                    # pragma: no cover
    _prime_math_score = None
    _MATH_BACKEND = "string_only"


@dataclass(frozen=True)
class ScoreResult:
    """What the reward consumes is `score`. The rest is for tests and dumps --
    none of it becomes a metric key, so the wandb schema cannot drift."""

    score: float
    scorer_name: str
    normalized_prediction: Optional[str] = None
    matched_gold: Optional[str] = None
    # QA reports em and f1 as separate metric columns; other scorers are binary
    # and report their own score in both, so the row schema never changes.
    em: float = 0.0
    f1: float = 0.0
    detail: str = ""


def _golds(gold_targets) -> list:
    if isinstance(gold_targets, str):
        return [gold_targets]
    return [str(g) for g in gold_targets]


# ============================================================== QA (default) ==

# A gold span that is directly negated does not answer the question: "not
# nominated" is not the answer "nominated". Lexical and fixed -- no semantics,
# no parsing, the same family of guard as qa_em._stance_match.
_NEGATORS = frozenset({"not", "no", "never", "neither", "nor", "without"})


def _boundary_subem(answer: str, golds):
    """Which gold appears in the answer as a contiguous TOKEN span, if any.

    WHY THIS EXISTS. Measured on the step-2 canary, 20 of 41 format-valid QA
    trajectories contained the correct answer and scored 0, because the policy
    writes "Melissa Sue Anderson played Mary Ingalls on ..." and strict EM
    wants the whole normalized string to equal the gold. Half the QA training
    signal was being discarded for verbosity.

    WHY NOT RAW subem_check ALONE. Its containment is a plain substring test,
    so "six" matches inside "sixty-four", "Paris" inside "Parisian" and "7/20"
    inside... anything that normalizes to contain "720". It is reused here as
    the CHEAP PRE-FILTER and for its stance handling -- a contiguous token span
    is always a substring of the single-spaced normalized text, so a raw miss
    is a boundary miss too -- and the token check refines what it lets through.

    qa_em_stance.py is not modified: it is golden-fixture-pinned to the original.
    """
    if not qa_em.subem_check(answer, golds):
        return None                      # raw containment already says no

    normalized = qa_em.normalize_answer(answer)
    tokens = normalized.split()
    for gold in golds:
        span = qa_em.normalize_answer(gold).split()
        if not span:
            continue
        if len(span) == 1 and span[0] in ("yes", "no"):
            # stance golds keep the existing protection: the FIRST standalone
            # yes/no decides, so "I do not know" does not answer "no"
            if qa_em._stance_match(normalized, span[0]):
                return gold
            continue
        for start in range(len(tokens) - len(span) + 1):
            if tokens[start:start + len(span)] != span:
                continue
            if start and tokens[start - 1] in _NEGATORS:
                continue
            return gold
    return None


def score_qa(response_text: str, gold_targets, reward_metric: str = "em") -> ScoreResult:
    """Strict EM first; a miss then gets one boundary-aware subEM attempt.

    The strict path is byte-identical to what it always was, so a trajectory
    that scored 1 before still scores 1 for the same reason. Only the 0s are
    revisited, and only for answers that contain the gold as a whole token
    span.

    KNOWN LIMIT, accepted deliberately: gold "Paris" matches "Paris, Texas".
    Ruling that out needs to know which Paris the question meant, which is the
    semantic matching this scorer refuses to do.
    """
    golds = _golds(gold_targets)
    answer = qa_em.extract_solution(solution_str=response_text)
    if answer is None:
        return ScoreResult(0.0, "qa_em", None, None, 0.0, 0.0, "no <answer> block")

    strict = 1.0 if qa_em.em_check(answer, golds) else 0.0
    matched = next((g for g in golds if qa_em.em_check(answer, [g])), None)
    detail = "strict"
    if strict == 0.0:
        matched = _boundary_subem(answer, golds)
        if matched is not None:
            detail = "boundary subem"

    em = 1.0 if matched is not None else 0.0
    f1 = 0.0
    for gold in golds:
        f1 = max(f1, qa_em.f1_score(answer, gold))
    score = f1 if reward_metric == "f1" else em
    return ScoreResult(score, "qa_em", qa_em.normalize_answer(answer), matched,
                       em, f1, detail if matched is not None else "no match")


# ===================================================================== GSM8K ==

_BOXED = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_IS = re.compile(r"(?:answer|result)\s*(?:is|:|=)\s*", re.IGNORECASE)
# a signed number with optional thousands separators, decimals, or a/b
_NUMBER = re.compile(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d[\d,]*(?:\.\d+)?)?\s*%?")
_UNIT_TAIL = re.compile(r"^\s*[a-zA-Z°\u00b0]*\s*$")


_THOUSANDS = re.compile(r"^-?\d{1,3}(?:,\d{3})+$")


def _strip_commas(raw: str) -> Optional[str]:
    """Remove thousands separators, but ONLY from a properly grouped number.

    '1,234' is 1234. '1,35' is not 135 -- in much of the world it is 1.35, and
    the old text path silently turned it into 135 because a comma is just
    punctuation to it. An ambiguous literal must not be credited, so this
    returns None and the answer scores 0 rather than guessing.
    """
    if "," not in raw:
        return raw
    head = raw.split(".")[0]
    return raw.replace(",", "") if _THOUSANDS.match(head) else None


def _to_number(text: str) -> Optional[Fraction]:
    """Exact numeric value of one literal, or None.

    Fraction/Decimal only -- '1/3' must not become 0.333... and then miss an
    exact gold. A trailing % divides by 100, which makes '20%' and '0.2' the
    same quantity and keeps '20%' distinct from a gold of '20'.
    """
    raw = text.strip().replace("$", "").replace(" ", "")
    raw = _strip_commas(raw) if raw else raw
    if not raw:
        return None
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1]
    try:
        if "/" in raw:
            value = Fraction(raw)
        else:
            value = Fraction(Decimal(raw))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return value / 100 if percent else value


def _extract_number(text: str) -> Optional[str]:
    """Priority: \\boxed{...}, then 'answer is ...', then the LAST literal.

    Last, not first: the answer block may restate the arithmetic before giving
    the result, and the result is what comes at the end. Intermediate numbers
    inside <think> never reach here -- only the <answer> body does.
    """
    boxed = _BOXED.findall(text)
    if boxed:
        found = _NUMBER.findall(boxed[-1])
        if found:
            return found[-1]
    tail = _ANSWER_IS.split(text)
    if len(tail) > 1:
        found = _NUMBER.findall(tail[-1])
        if found:
            return found[0]
    found = _NUMBER.findall(text)
    return found[-1] if found else None


def score_gsm8k(response_text: str, gold_targets, reward_metric: str = "em") -> ScoreResult:
    golds = _golds(gold_targets)
    answer = qa_em.extract_solution(solution_str=response_text)
    if answer is None:
        return ScoreResult(0.0, "gsm8k", None, None, 0.0, 0.0, "no <answer> block")

    literal = _extract_number(answer)
    if literal is None:
        return ScoreResult(0.0, "gsm8k", answer.strip(), None, 0.0, 0.0, "no numeric literal")
    value = _to_number(literal)
    if value is None:
        return ScoreResult(0.0, "gsm8k", literal, None, 0.0, 0.0, "unparseable literal")

    for gold in golds:
        gold_literal = _extract_number(gold) or gold
        gold_value = _to_number(gold_literal)
        if gold_value is not None and gold_value == value:
            return ScoreResult(1.0, "gsm8k", str(value), gold, 1.0, 1.0)
    return ScoreResult(0.0, "gsm8k", str(value), None, 0.0, 0.0, "numeric mismatch")


# ====================================================================== MATH ==

_PERCENT_LATEX = re.compile(r"^(-?[\d.,]+)\s*\\?%$")


def _percent_variant(text: str) -> Optional[str]:
    """'50\\%' -> '1/2'. prime_math treats a percent as a bare number, so this
    is done here rather than left to score as 50 vs 0.5."""
    match = _PERCENT_LATEX.match(text.strip())
    if match is None:
        return None
    value = _to_number(match.group(1) + "%")
    return None if value is None else f"{value.numerator}/{value.denominator}"


def _math_equal(prediction: str, gold: str) -> bool:
    if _prime_math_score is None:
        return False
    try:
        result = _prime_math_score(prediction, gold)
    except Exception as exc:                          # sympy raises broadly
        logger.debug("math backend failed on %r vs %r: %s", prediction, gold, exc)
        return False
    # prime_math returns (correct, ...) or a bare bool depending on version
    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


def score_math(response_text: str, gold_targets, reward_metric: str = "em") -> ScoreResult:
    """Symbolic comparison with a string fallback.

    An unsupported form must come back UNMATCHED, never matched-by-accident:
    a false positive teaches the router that a wrong answer was fine, which is
    worse than the missed reward of a false negative.
    """
    golds = _golds(gold_targets)
    answer = qa_em.extract_solution(solution_str=response_text)
    if answer is None:
        return ScoreResult(0.0, "math", None, None, 0.0, 0.0, "no <answer> block")

    candidates = [answer.strip()]
    boxed = _BOXED.findall(answer)
    if boxed:
        candidates.insert(0, boxed[-1].strip())
    for candidate in list(candidates):
        percent = _percent_variant(candidate)
        if percent:
            candidates.append(percent)

    for gold in golds:
        gold_forms = [gold]
        gold_percent = _percent_variant(gold)
        if gold_percent:
            gold_forms.append(gold_percent)
        for candidate in candidates:
            for gold_form in gold_forms:
                if _math_equal(candidate, gold_form):
                    return ScoreResult(1.0, "math", candidate, gold, 1.0, 1.0, _MATH_BACKEND)
                # last resort: the same normalization the QA path uses, which
                # still catches identical spellings when the parser bails
                if qa_em.normalize_answer(candidate) == qa_em.normalize_answer(gold_form) \
                        and qa_em.normalize_answer(candidate):
                    return ScoreResult(1.0, "math", candidate, gold, 1.0, 1.0, "string")
    return ScoreResult(0.0, "math", candidates[0], None, 0.0, 0.0, f"unmatched ({_MATH_BACKEND})")


# ============================================================ ARC-Challenge ==

_LETTERS = "ABCDE"
# a letter only counts in an answer POSITION, never loose in prose -- otherwise
# "option B is wrong, so A" scores as B
_LETTER_PATTERNS = (
    re.compile(r"^\s*\(?([A-E])\)?\s*[.):]?\s*$"),                     # 'B'  '(B)'  'B.'
    re.compile(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([A-E])\)?\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-E])\)?\s*[.):]\s+\S"),                     # 'B. less dense'
)


def _arc_letter(text: str) -> Optional[str]:
    for pattern in _LETTER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def score_arc(response_text: str, gold_targets, reward_metric: str = "em") -> ScoreResult:
    """Gold carries the option LETTER and (usually) the option TEXT.

    Both are checked. When the model supplies both and they disagree, that is
    scored 0: an answer that says 'C. less dense' when B is 'less dense' has
    not identified the option, and crediting it would reward the coincidence.
    """
    golds = _golds(gold_targets)
    gold_letters = {g.strip().upper() for g in golds if len(g.strip()) == 1
                    and g.strip().upper() in _LETTERS}
    gold_texts = [g for g in golds if g.strip().upper() not in _LETTERS or len(g.strip()) > 1]

    answer = qa_em.extract_solution(solution_str=response_text)
    if answer is None:
        return ScoreResult(0.0, "arc", None, None, 0.0, 0.0, "no <answer> block")
    answer = answer.strip()

    letter = _arc_letter(answer)
    normalized = qa_em.normalize_answer(answer)
    text_hit = next((g for g in gold_texts
                     if qa_em.normalize_answer(g) and qa_em.normalize_answer(g) == normalized),
                    None)
    # 'B. less dense' -- strip the leading label before comparing the prose
    if text_hit is None and letter is not None:
        stripped = re.sub(r"^\s*\(?[A-E]\)?\s*[.):]\s*", "", answer)
        stripped_norm = qa_em.normalize_answer(stripped)
        text_hit = next((g for g in gold_texts
                         if qa_em.normalize_answer(g) and qa_em.normalize_answer(g) == stripped_norm),
                        None)

    letter_hit = letter is not None and letter in gold_letters
    letter_wrong = letter is not None and gold_letters and letter not in gold_letters

    if letter_wrong and text_hit is not None:
        return ScoreResult(0.0, "arc", answer, None, 0.0, 0.0,
                           f"conflict: letter {letter} vs text {text_hit!r}")
    if letter_hit:
        return ScoreResult(1.0, "arc", letter, letter, 1.0, 1.0)
    if letter_wrong:
        return ScoreResult(0.0, "arc", letter, None, 0.0, 0.0, "wrong letter")
    if text_hit is not None:
        return ScoreResult(1.0, "arc", normalized, text_hit, 1.0, 1.0, "option text")
    return ScoreResult(0.0, "arc", normalized, None, 0.0, 0.0, "no option identified")


# ================================================================== registry ==

SCORERS = {
    "math_l35": score_math,
    "gsm8k": score_gsm8k,
    "arc_challenge": score_arc,
}
DEFAULT_SCORER = score_qa
QA_DATA_SOURCES = ("nq", "triviaqa", "hotpotqa", "2wiki", "musique")

_warned = set()


def score_answer(data_source, response_text: str, gold_targets: Sequence[str],
                 reward_metric: str = "em") -> ScoreResult:
    """THE entry point. Exact dict lookup on data_source; QA otherwise."""
    key = data_source if isinstance(data_source, str) else None
    scorer = SCORERS.get(key) if key is not None else None
    if scorer is None:
        if key not in QA_DATA_SOURCES and key not in _warned:
            _warned.add(key)
            logger.warning(
                "data_source %r has no registered scorer; falling back to the QA "
                "EM/F1 scorer. Register it in rewards/scorers.SCORERS if it needs "
                "numeric, symbolic or multiple-choice grading.", key)
        scorer = DEFAULT_SCORER
    return scorer(response_text, gold_targets, reward_metric)
