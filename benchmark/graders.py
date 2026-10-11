"""Exact-match graders for TriviaQA and GSM8K (REQ-BENCH-2).

- **GSM8K grader**: extracts the final numeric answer from the model's
  completion and compares numerically.
- **TriviaQA grader**: normalises case / whitespace / punctuation and
  checks against the gold-answer string.

When a completion contains a ``Final answer: ...`` line (the format the
benchmark prompts ask for, see ``prompts.py``), only that line is graded, so
an alias mentioned in passing in the reasoning can't count as correct.
"""

from __future__ import annotations

import re
import string
from typing import Sequence


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


_FINAL_ANSWER = re.compile(r"final answer\s*[:：]\s*(.*)", re.IGNORECASE)


def final_answer_segment(text: str) -> str:
    """Return the rest of the last ``Final answer:`` line, or *text* if absent."""
    matches = _FINAL_ANSWER.findall(text)
    return matches[-1].strip() if matches else text


# A number must start with a digit, so a lone "," or "-" never matches.
_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def _extract_number(text: str) -> str | None:
    """Extract the last number from a string (handles commas, negatives)."""
    # Look for #### <number> first (GSM8K standard format)
    m = re.search(r"####\s*(" + _NUMBER + ")", text)
    if m:
        return m.group(1).replace(",", "")

    # Then the first number on a "Final answer:" line
    segment = final_answer_segment(text)
    if segment is not text:
        m = re.search(_NUMBER, segment)
        if m:
            return m.group(0).replace(",", "")

    # Fallback: find the last number in the text
    numbers = re.findall(_NUMBER, text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_triviaqa(model_answer: str, gold_answer: str | Sequence[str]) -> bool:
    """Grade a TriviaQA response using normalised whole-word matching.

    *gold_answer* may be a single string or a list of acceptable aliases.
    Returns ``True`` if any normalised gold answer appears in the normalised
    model answer as a whole-word phrase (so gold ``"12"`` does not match
    ``"120"``).
    """
    golds = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
    padded_model = f" {_normalize_text(final_answer_segment(model_answer))} "
    for gold in golds:
        norm_gold = _normalize_text(gold)
        if norm_gold and f" {norm_gold} " in padded_model:
            return True
    return False


def grade_gsm8k(model_answer: str, gold_answer: str) -> bool:
    """Grade a GSM8K response using numeric exact-match.

    Extracts the final number from the model's output and compares it
    to the gold answer numerically.
    """
    predicted = _extract_number(model_answer)
    expected = _extract_number(gold_answer)
    if predicted is None or expected is None:
        return False
    try:
        return float(predicted) == float(expected)
    except ValueError:
        return predicted == expected
