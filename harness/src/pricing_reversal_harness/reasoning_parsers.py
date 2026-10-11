"""Reasoning-token accounting.

vLLM's chat-completions response typically exposes reasoning text under
`choices[0].message.reasoning_content` when the model server is configured
with `--reasoning-parser=<family>`. We count tokens of that string (via the
transformers tokenizer we already have) so the harness output matches the
paper's `thinking_tokens` field.

Fallback path: if `reasoning_content` is empty, sniff for a `<think>...</think>`
block in `content` and count its tokens. If nothing is found, return 0.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@functools.lru_cache(maxsize=8)
def _get_tokenizer(hf_repo: str) -> Any | None:
    """Load a HF tokenizer once per model. Cheap enough to load in-process
    because the harness pod only needs the tokenizer, not the weights."""

    try:
        from transformers import AutoTokenizer  # imported lazily
    except ImportError:  # pragma: no cover
        log.warning("transformers not installed; thinking-token count will be 0")
        return None
    try:
        return AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover
        log.warning("could not load tokenizer for %s: %s", hf_repo, exc)
        return None


def count_prompt_tokens(hf_repo: str, text: str) -> int:
    """Return the token count of a prompt string using the model's own
    tokenizer. Falls back to a chars-per-token estimate if the tokenizer
    can't be loaded (transformers missing / gated repo / offline).

    The fallback deliberately *overestimates* (chars // 3 rather than
    chars // 4) so callers that use this to size a `max_tokens` budget
    stay on the safe side of `max_model_len` when the real count is
    unknown.
    """

    if not text:
        return 0
    tok = _get_tokenizer(hf_repo)
    if tok is None:
        return max(1, len(text) // 3)
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:  # pragma: no cover
        return max(1, len(text) // 3)


def count_thinking_tokens(hf_repo: str, message: dict[str, Any] | None,
                          full_content: str | None) -> int:
    """Return the token count of the reasoning portion of a response.

    `message` is the raw `choices[0].message` object from vLLM's
    /v1/chat/completions response; `full_content` is the raw text of the
    reply used as fallback.
    """

    if message is None:
        message = {}
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    text: str | None = None
    if isinstance(reasoning, str) and reasoning:
        text = reasoning
    elif full_content:
        m = _THINK_BLOCK.search(full_content)
        if m:
            text = m.group(1)

    if not text:
        return 0

    tok = _get_tokenizer(hf_repo)
    if tok is None:
        # Rough fallback so the field is at least non-zero and monotonic.
        return max(1, len(text) // 4)
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:  # pragma: no cover
        return max(1, len(text) // 4)
