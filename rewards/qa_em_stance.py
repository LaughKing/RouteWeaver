# Copyright 2026 RouteWeaver.
"""Answer scoring — a self-contained port of qa_em."""

import re
import string
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def _stance_match(normalized_prediction, golden_answer):
    """Yes/no gold: grade by the first standalone yes/no token in the prediction."""
    first = re.search(r'\b(yes|no)\b', normalized_prediction)
    return first is not None and first.group(1) == golden_answer


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in ('yes', 'no'):
            if _stance_match(normalized_prediction, golden_answer):
                score = 1
                break
        elif golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer\s*>(.*?)</answer\s*>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    matches = list(match)

    if len(matches) <= 0:
        return None

    return matches[-1].group(1).strip()


def f1_score(prediction: str, ground_truth: str):
    """Token-level F1 between prediction and ground truth (original semantics:
    yes/no/noanswer must match exactly or score 0).
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = 0

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1
