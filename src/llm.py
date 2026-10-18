"""LLM provider abstraction (Gemini / Groq / none) over plain HTTPS.

Keys are sent in headers (never URLs), are never logged, and error messages are redacted.
All callers handle ``LLMError`` and fall back gracefully (baseline extraction, extractive answers).
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from functools import lru_cache

import httpx

from src.config import Settings, get_settings
from src.utils import redact_secrets

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for LLM failures (always safe to show to users)."""


class LLMUnavailable(LLMError):
    """No provider/key configured, or the provider is unreachable."""


class LLMRateLimited(LLMError):
    """The free tier quota/rate limit was hit."""


class RateGate:
    """Client-side sliding-window limiter so we stay under the free tier's requests/minute."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, max_wait: float) -> None:
        """Block until a slot is free; raise LLMRateLimited if that would take over ``max_wait`` s."""
        deadline = time.monotonic() + max_wait
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60:
                    self._times.popleft()
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return
                wait = 60 - (now - self._times[0])
            if time.monotonic() + wait > deadline:
                raise LLMRateLimited("The free LLM quota is busy; try again in a few seconds.")
            time.sleep(min(wait, 1.0) + 0.05)


class LLMProvider(ABC):
    """Minimal text-in/text-out interface so providers can be swapped via config."""

    name: str = "base"
    max_wait_seconds: float = 6.0  # how long a request may queue for a rate-limit slot
    gate: RateGate | None = None

    @abstractmethod
    def _request(self, system: str, user: str, json_mode: bool, max_tokens: int) -> str: ...

    def generate(self, system: str, user: str, *, json_mode: bool = False,
                 max_tokens: int = 1024, retries: int = 2) -> str:
        """Call the model, retrying briefly on rate limits / transient 5xx.

        Raises:
            LLMRateLimited: quota still exhausted after retries.
            LLMUnavailable: provider unreachable or misconfigured.
            LLMError: any other provider failure.
        """
        if self.gate is not None:
            self.gate.acquire(self.max_wait_seconds)
        for attempt in range(retries + 1):
            try:
                return self._request(system, user, json_mode, max_tokens)
            except httpx.TimeoutException as exc:
                raise LLMUnavailable("The language model timed out.") from exc
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code in (429, 503) and attempt < retries:
                    wait = min(float(exc.response.headers.get("retry-after", 1.5 * (attempt + 1))), 5.0)
                    logger.warning("%s returned %d; retrying in %.1fs", self.name, code, wait)
                    time.sleep(wait)
                    continue
                if code == 429:
                    raise LLMRateLimited("The free LLM quota is exhausted; try again shortly.") from exc
                if code in (400, 401, 403, 404):
                    detail = redact_secrets(exc.response.text[:200])
                    logger.error("%s rejected the request (%d): %s", self.name, code, detail)
                    raise LLMUnavailable(f"The LLM provider rejected the request ({code}).") from exc
                raise LLMError(f"The LLM provider failed ({code}).") from exc
            except httpx.HTTPError as exc:
                raise LLMUnavailable("Could not reach the LLM provider.") from exc
        raise LLMError("unreachable")  # pragma: no cover


class NullProvider(LLMProvider):
    """Used when no key is configured: always raises LLMUnavailable."""

    name = "none"

    def _request(self, system: str, user: str, json_mode: bool, max_tokens: int) -> str:
        raise LLMUnavailable("No LLM is configured.")


class GeminiProvider(LLMProvider):
    """Google Gemini via the REST API (free tier)."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self._key = settings.gemini_api_key.get_secret_value()
        self._model = settings.gemini_model
        self.gate = RateGate(settings.llm_rpm)
        self._client = httpx.Client(timeout=settings.llm_timeout_seconds)

    def _request(self, system: str, user: str, json_mode: bool, max_tokens: int) -> str:
        gen_cfg: dict = {"temperature": 0.0, "maxOutputTokens": max_tokens}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen_cfg}
        r = self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent",
            headers={"x-goog-api-key": self._key}, json=body)
        r.raise_for_status()
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError) as exc:
            raise LLMError("The model returned no answer (it may have been blocked).") from exc


class GroqProvider(LLMProvider):
    """Groq's OpenAI-compatible chat completions endpoint (free tier)."""

    name = "groq"

    def __init__(self, settings: Settings) -> None:
        self._key = settings.groq_api_key.get_secret_value()
        self._model = settings.groq_model
        self.gate = RateGate(settings.llm_rpm)
        self._client = httpx.Client(timeout=settings.llm_timeout_seconds)

    def _request(self, system: str, user: str, json_mode: bool, max_tokens: int) -> str:
        body: dict = {"model": self._model, "temperature": 0.0, "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self._client.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {self._key}"}, json=body)
        r.raise_for_status()
        try:
            return r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError("The model returned no answer.") from exc


@lru_cache
def get_provider() -> LLMProvider:
    """Return the configured provider (cached); NullProvider when no key is set."""
    s = get_settings()
    if not s.llm_enabled:
        return NullProvider()
    provider: LLMProvider = GeminiProvider(s) if s.llm_provider == "gemini" else GroqProvider(s)
    logger.info("LLM provider: %s", provider.name)
    return provider
