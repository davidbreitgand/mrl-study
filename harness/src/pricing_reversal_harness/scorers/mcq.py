"""Scorer for MCQ answers: last 'Answer: <LETTER>' in output.

Matches:
  Answer: A
  Answer: $A
  Answer: **A**
  Answer:B    (missing space, tolerated)

Falls back to the last standalone letter on its own line.

Accepts A-J (10 options). This is a superset of the 4-option A-D shape
GPQA uses; MMLU-Pro is the reason for the widening. Ground-truth
validation is checked against the same set.
"""

from __future__ import annotations

import re


_VALID_LETTERS = set("ABCDEFGHIJ")

_ANSWER_LINE = re.compile(
    r"[Aa]nswer\s*:\s*[\$\*\s`]*([A-Ja-j])",
)
_STANDALONE_LETTER = re.compile(r"(?m)^\s*[\$\*\s`]*([A-Ja-j])[\s\.\*`]*$")


def _last_match(pat: re.Pattern[str], text: str) -> str | None:
    last: str | None = None
    for m in pat.finditer(text or ""):
        last = m.group(1).upper()
    return last


def score_mcq_letter(model_output: str, ground_truth: str) -> tuple[float | None, str | None]:
    prediction = _last_match(_ANSWER_LINE, model_output) or _last_match(_STANDALONE_LETTER, model_output)
    if prediction is None:
        return None, None
    gt = (ground_truth or "").strip().upper()
    if gt not in _VALID_LETTERS:
        return None, prediction
    return (1.0 if prediction == gt else 0.0), prediction
