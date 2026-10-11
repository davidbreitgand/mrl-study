"""Tests for the NoBackendError short-circuit in VLLMClient.

We don't mock httpx directly — we monkey-patch VLLMClient._client with a
tiny fake that returns pre-programmed httpx.Response objects.
"""

from __future__ import annotations

import pytest
import httpx

from pricing_reversal_harness.vllm_client import (
    NoBackendError, VLLMClient, _looks_like_no_backend,
)


def test_looks_like_no_backend_matches_field_form():
    assert _looks_like_no_backend(
        503,
        "inference error: ServiceUnavailable - failed to find endpoint candidates for serving the request",
    )


def test_looks_like_no_backend_matches_short_form():
    assert _looks_like_no_backend(503, "no endpoint candidates available")


def test_looks_like_no_backend_case_insensitive():
    assert _looks_like_no_backend(503, "FAILED TO FIND ENDPOINT CANDIDATES")


def test_looks_like_no_backend_ignores_non_503():
    assert not _looks_like_no_backend(500, "failed to find endpoint candidates")
    assert not _looks_like_no_backend(429, "failed to find endpoint candidates")


def test_looks_like_no_backend_ignores_unrelated_503():
    assert not _looks_like_no_backend(503, "Service Unavailable: upstream timeout")


class _FakeAsyncClient:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def post(self, url: str, json=None):  # noqa: A002
        self.call_count += 1
        r = self._responses.pop(0) if self._responses else self._responses[-1]
        if isinstance(r, Exception):
            raise r
        return r

    async def aclose(self) -> None:
        pass


def _mk_response(status: int, text: str) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=text.encode(),
        request=httpx.Request("POST", "http://x/v1/chat/completions"),
    )


async def test_chat_raises_no_backend_error_on_no_endpoints_503():
    c = VLLMClient(
        base_url="http://router",
        timeout_seconds=1.0,
        max_concurrent=4,
        retry_max_attempts=3,
        retry_backoff_seconds=0.01,
    )
    c._client = _FakeAsyncClient([
        _mk_response(503, "failed to find endpoint candidates for serving the request"),
    ])

    with pytest.raises(NoBackendError) as ei:
        await c.chat(
            model="m", messages=[{"role": "user", "content": "q"}],
            max_tokens=8, temperature=0.0, top_p=1.0,
        )
    # Should short-circuit — no retries burned.
    assert c._client.call_count == 1
    assert ei.value.status_code == 503


async def test_chat_still_retries_generic_5xx():
    # Non-"no endpoint" 5xx should still consume the retry budget.
    c = VLLMClient(
        base_url="http://router",
        timeout_seconds=1.0,
        max_concurrent=4,
        retry_max_attempts=3,
        retry_backoff_seconds=0.01,
    )
    c._client = _FakeAsyncClient([
        _mk_response(500, "internal server error"),
        _mk_response(502, "bad gateway"),
        _mk_response(504, "gateway timeout"),
    ])
    with pytest.raises(httpx.HTTPStatusError):
        await c.chat(
            model="m", messages=[{"role": "user", "content": "q"}],
            max_tokens=8, temperature=0.0, top_p=1.0,
        )
    # All 3 attempts consumed.
    assert c._client.call_count == 3


async def test_chat_success_returns_body():
    body_ok = _mk_response(200, '{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}')
    c = VLLMClient(
        base_url="http://router",
        timeout_seconds=1.0,
        max_concurrent=4,
        retry_max_attempts=3,
        retry_backoff_seconds=0.01,
    )
    c._client = _FakeAsyncClient([body_ok])
    body, elapsed = await c.chat(
        model="m", messages=[{"role": "user", "content": "q"}],
        max_tokens=8, temperature=0.0, top_p=1.0,
    )
    assert body["choices"][0]["message"]["content"] == "hi"
    assert elapsed >= 0
    assert c._client.call_count == 1
