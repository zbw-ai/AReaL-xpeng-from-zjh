"""Fuyao reward function wrappers.

Wraps areal's gsm8k_reward_fn with threaded-env fallback for math_verify.
Adds string-match fallback for non-numeric answers (Yes/No/True/False/etc.)
that math_verify cannot parse.
"""

import re

from math_verify.grader import verify as math_verify_grader
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify.parser import parse as math_verify_parse

from areal.utils import logging

logger = logging.getLogger("FuyaoReward")
_THREADED_ENV_ERRORS = (
    "Math-Verify 'parse' function doesn't support threaded environment",
    "signal only works in main thread",
)

_GOLD_TARGET = (
    ExprExtractionConfig(try_extract_without_anchor=True),
    LatexExtractionConfig(),
)
_PRED_TARGET = (
    ExprExtractionConfig(try_extract_without_anchor=True),
    LatexExtractionConfig(),
)
_PRECISION = 6


def _verify_without_timeout(response: str, ground_truth: str) -> float:
    extracted_predictions = math_verify_parse(
        response, _PRED_TARGET, parsing_timeout=None
    )
    extracted_golds = math_verify_parse(
        ground_truth, _GOLD_TARGET, parsing_timeout=None
    )
    if not extracted_golds or not extracted_predictions:
        return 0.0
    matched = any(
        math_verify_grader(gold, pred, float_rounding=_PRECISION, timeout_seconds=None)
        for gold in extracted_golds
        for pred in extracted_predictions
    )
    return 1.0 if matched else 0.0


def _extract_answer_from_response(response: str) -> str:
    """Extract answer from model response, trying multiple formats.

    Priority:
      1. \\boxed{...}
      2. "Answer: ..." (last occurrence)
      3. "answer is ..." (last occurrence)
    """
    # 1. Try \boxed{} (may be nested braces)
    matches = re.findall(
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response
    )
    if matches:
        return matches[-1].strip()

    # 2. Try "Answer: X" pattern (last line)
    m = re.findall(r"(?:^|\n)\s*Answer:\s*(.+)", response)
    if m:
        return m[-1].strip().rstrip(".")

    # 3. Try "the answer is X"
    m = re.findall(r"(?:the\s+)?answer\s+is\s+(.+?)(?:\.|$)", response, re.I)
    if m:
        return m[-1].strip()

    return ""


def _normalize(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip().lower()
    # Remove \boxed{} wrapper
    m = re.match(r"^\\boxed\{(.*)\}$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    # Remove trailing punctuation
    s = s.rstrip(".,;!?")
    return s


def _string_match_fallback(response: str, ground_truth: str) -> float:
    """Fallback: extract answer from response and compare as normalized strings.

    Handles cases math_verify cannot parse (Yes/No/True/False/text expressions).
    """
    extracted = _extract_answer_from_response(response)
    if not extracted:
        return 0.0

    gt_norm = _normalize(ground_truth)
    pred_norm = _normalize(extracted)

    if gt_norm == pred_norm:
        return 1.0
    return 0.0


def math_verify_with_fallback(response: str, ground_truth: str) -> float:
    """Verify math answer with threaded-env fallback + string match fallback.

    For direct use in workflows (not via AsyncRewardWrapper).
    """
    try:
        score = _verify_without_timeout(response, ground_truth)
        if score > 0:
            return score
    except Exception:
        pass

    # Fallback: string match for non-numeric answers
    return _string_match_fallback(response, ground_truth)


def math_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """Math reward with math_verify + string match fallback.

    Drop-in replacement for areal.reward.gsm8k.gsm8k_reward_fn.
    Works with both numeric answers (math_verify) and non-numeric
    answers like Yes/No/True/False/symbolic expressions (string match).
    """
    response = str(completions)
    ground_truth = str(answer)

    # 1. Try math_verify (handles numeric, latex, equivalence)
    try:
        from areal.reward import get_math_verify_worker

        worker = get_math_verify_worker()
        score = worker.verify(response, ground_truth)
        if score > 0:
            return score
    except Exception as exc:
        if any(msg in str(exc) for msg in _THREADED_ENV_ERRORS):
            try:
                score = _verify_without_timeout(response, ground_truth)
                if score > 0:
                    return score
            except Exception:
                pass

    # 2. Fallback: string match (handles Yes/No/True/False/text)
    return _string_match_fallback(response, ground_truth)
