"""Thin async client for the router's /v1/chat/completions endpoint.

Bounded concurrency + exponential-backoff retries on transient failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx


log = logging.getLogger(__name__)


class NoBackendError(RuntimeError):
    """The EPP returned a body indicating that no eligible endpoint
    candidates exist. Distinct from a transient 5xx: retrying within
    the same query doesn't help — the whole runner needs to pause until
    the backend comes back. See DatasetRunner's `backend_up` gate.
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


# Marker substrings emitted by llm-d-router EPP (v0.9.0+) when its
# scheduler cannot find any endpoint that matches the request. Both
# forms have been observed in the field; we match either.
_NO_BACKEND_MARKERS = (
    "failed to find endpoint candidates",
    "no endpoint candidates",
)


def _looks_like_no_backend(status_code: int, body: str) -> bool:
    if status_code != 503:
        return False
    b = body.lower()
    return any(m in b for m in _NO_BACKEND_MARKERS)


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        max_concurrent: int,
        retry_max_attempts: int,
        retry_backoff_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_max_attempts = max(1, retry_max_attempts)
        self.retry_backoff_seconds = retry_backoff_seconds
        self._sem = asyncio.Semaphore(max_concurrent)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=30.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Send one chat-completion request. Returns (json_body, elapsed_seconds).
        Raises on final failure after all retries.

        `reasoning_effort` is the OpenAI-compatible top-level field
        (`"low"` | `"medium"` | `"high"` | `"none"`). vLLM's server maps
        low/medium/high to `enable_thinking=true` and `"none"` to
        `enable_thinking=false` via chat_template_kwargs auto-injection.
        A None value here omits the field entirely, preserving prior
        behaviour.
        """

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        url = f"{self.base_url}/v1/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            async with self._sem:
                started = time.perf_counter()
                try:
                    resp = await self._client.post(url, json=payload)
                    elapsed = time.perf_counter() - started
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    last_exc = exc
                    log.warning("chat attempt %d transport error: %s", attempt, exc)
                else:
                    if resp.status_code < 500 and resp.status_code != 429:
                        if resp.status_code >= 400:
                            resp.raise_for_status()
                        body_json = resp.json()
                        # CostGuard cross-cluster attribution: the Hub EPP's
                        # destination-endpoint-served-verifier stamps this
                        # header on every response. Fold it into the returned
                        # body so the runner can attribute cost to the arm
                        # that actually served the request.
                        body_json["_served_endpoint"] = resp.headers.get(
                            "x-conformance-test-served-endpoint"
                        )
                        return body_json, elapsed
                    # Structural "no backend" 503 short-circuits the retry
                    # loop — retrying inside this call is pointless while
                    # the entire router has no eligible endpoints. The
                    # caller (DatasetRunner) coordinates a pause + shared
                    # re-probe across all in-flight queries.
                    if _looks_like_no_backend(resp.status_code, resp.text):
                        raise NoBackendError(resp.status_code, resp.text)
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        request=resp.request, response=resp,
                    )
                    log.warning("chat attempt %d bad status: %s", attempt, last_exc)

            if attempt < self.retry_max_attempts:
                delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc

    async def wait_ready(
        self,
        *,
        model: str,
        deadline_seconds: float,
        probe_prompt: str,
        probe_max_tokens: int,
    ) -> bool:
        """Poll /v1/chat/completions with a tiny probe until it succeeds or the
        deadline expires. Returns True on success, False on timeout."""

        deadline = time.time() + deadline_seconds
        while time.time() < deadline:
            try:
                await self.chat(
                    model=model,
                    messages=[{"role": "user", "content": probe_prompt}],
                    max_tokens=probe_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                log.info("ready-probe not yet ready: %s", exc)
                await asyncio.sleep(10.0)
        return False
