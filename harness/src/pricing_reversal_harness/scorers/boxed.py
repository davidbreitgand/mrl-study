"""Scorer for AIME-style answers: last \\boxed{...} matches ground truth.

Handles the balanced-brace form (e.g. `\\boxed{\\tfrac{25}{8}}`) and falls back
to the last plain numeric answer on a line if no \\boxed{} is present.

Returns (score, prediction) where score is 1.0 on match, 0.0 on mismatch,
None if we could not extract a prediction.
"""

from __future__ import annotations

import re
from fractions import Fraction


_BOX_HEADER = re.compile(r"\\boxed\s*\{")
_NUMERIC_TAIL = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def _extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last balanced-brace \\boxed{...}."""

    if not text:
        return None
    last: str | None = None
    for m in _BOX_HEADER.finditer(text):
        i = m.end()
        depth = 1
        buf: list[str] = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                buf.append(ch)
            else:
                buf.append(ch)
            i += 1
        if depth == 0:
            last = "".join(buf).strip()
    return last


def _normalize_numeric(s: str) -> str | None:
    """Reduce a candidate string to a comparable numeric form.

    - Strips LaTeX formatting (\\tfrac, \\frac, whitespace, $).
    - If it looks like p/q, returns reduced Fraction as "p/q" or "int".
    - Otherwise, keeps the last floating-point / integer literal.
    """

    if not s:
        return None
    s = s.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    s = s.replace("$", "").replace(" ", "").strip()

    m = re.match(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            f = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{f.numerator}" if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        except ZeroDivisionError:
            return None

    m = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", s)
    if m:
        try:
            f = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{f.numerator}" if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        except ZeroDivisionError:
            return None

    m = _NUMERIC_TAIL.search(s)
    if m:
        return m.group(1)
    return None


def score_boxed_integer(model_output: str, ground_truth: str) -> tuple[float | None, str | None]:
    boxed = _extract_last_boxed(model_output)
    prediction = _normalize_numeric(boxed) if boxed is not None else None
    if prediction is None:
        # last-ditch: any trailing numeric on the last non-empty line
        for line in reversed((model_output or "").splitlines()):
            line = line.strip()
            if not line:
                continue
            m = _NUMERIC_TAIL.search(line)
            if m:
                prediction = m.group(1)
                break

    if prediction is None:
        return None, None

    gt_norm = _normalize_numeric(ground_truth)
    if gt_norm is None:
        return None, prediction

    return (1.0 if prediction == gt_norm else 0.0), prediction
