#!/usr/bin/env python3
"""
Scoring functions for benchmark evaluation.

Provides:
  - exact_match: normalized string comparison
  - f1_score: token-level precision/recall
  - exact_match_number: extract final number and compare
  - normalize_answer: lowercase, strip articles/punctuation/whitespace
  - extract_final_number: regex patterns for #### N, "answer is N", last number
  - extract_answer: priority-key search over state.data for final answer
"""

import re
import string


def normalize_answer(s):
    """Lowercase, strip articles/punctuation/extra whitespace."""
    s = str(s).lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s.strip()


def exact_match(predicted, expected):
    """Normalized string comparison. Returns 0.0 or 1.0."""
    return 1.0 if normalize_answer(predicted) == normalize_answer(expected) else 0.0


def f1_score(predicted, expected):
    """Token-level F1 score. Returns 0.0–1.0."""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(expected).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    # Count occurrences
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    num_common = sum(min(pred_counts[t], gold_counts[t]) for t in common)
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_final_number(text):
    """Extract the final numeric answer from text.

    Tries patterns in order:
      1. #### N  (GSM8K standard format)
      2. "the answer is N" / "answer: N"
      3. Last number in the text
    Returns the number as a float, or None if not found.
    """
    text = str(text)

    # Pattern 1: #### N (GSM8K format)
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m:
        return float(m.group(1).replace(',', ''))

    # Pattern 2: "the answer is N" or "answer: N" or "answer = N"
    m = re.search(
        r'(?:the\s+)?answer\s*(?:is|:|=)\s*(-?[\d,]+\.?\d*)',
        text, re.IGNORECASE
    )
    if m:
        return float(m.group(1).replace(',', ''))

    # Pattern 3: "= N" at end of a line (common in math solutions)
    m = re.search(r'=\s*(-?[\d,]+\.?\d*)\s*$', text, re.MULTILINE)
    if m:
        return float(m.group(1).replace(',', ''))

    # Pattern 4: Last standalone number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass

    return None


def exact_match_number(predicted, expected_number):
    """Extract final number from predicted text, compare to expected.
    Returns 0.0 or 1.0."""
    pred_num = extract_final_number(predicted)
    if pred_num is None:
        return 0.0
    try:
        exp = float(expected_number)
    except (ValueError, TypeError):
        return 0.0
    # Compare with tolerance for floating point
    if abs(exp) < 1e-9:
        return 1.0 if abs(pred_num) < 1e-9 else 0.0
    return 1.0 if abs(pred_num - exp) / max(abs(exp), 1e-9) < 1e-6 else 0.0


def extract_answer(state_data):
    """Extract the final answer from agent state data.

    Searches keys in priority order:
      answer, final_answer, final_output, output_text, result,
      current_output, synthesized_answer, response
    Also searches nested dicts (e.g. state.data["xxx_result"]["answer"]).
    """
    if not isinstance(state_data, dict):
        return str(state_data)

    priority_keys = [
        "answer", "final_answer", "final_output", "output_text",
        "result", "current_output", "synthesized_answer", "response",
    ]

    # Direct lookup
    for key in priority_keys:
        val = state_data.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 0:
            return val.strip()

    # Search nested dicts
    for k, v in state_data.items():
        if isinstance(v, dict):
            for key in priority_keys:
                val = v.get(key)
                if val and isinstance(val, str) and len(val.strip()) > 0:
                    return val.strip()

    # Fallback: concatenate all string values
    parts = []
    for v in state_data.values():
        if isinstance(v, str) and len(v.strip()) > 0:
            parts.append(v.strip())
    return ' '.join(parts) if parts else ""


def score_hotpotqa(predicted, expected):
    """Score a HotpotQA prediction. Returns dict with em and f1."""
    return {
        "em": exact_match(predicted, expected),
        "f1": f1_score(predicted, expected),
    }


def score_gsm8k(predicted, expected_number):
    """Score a GSM8K prediction. Returns dict with em."""
    return {
        "em": exact_match_number(predicted, expected_number),
    }
