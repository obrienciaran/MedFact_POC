"""Provider-agnostic LLM access for the steps that need language understanding.

Two steps in the filter need a model: claim extraction and stance judgement. Both go
through the single ``LLMClient`` Protocol (see ``base.llm``) so a provider is chosen in one
place. The default ``stub`` client returns deterministic canned text, so the whole filter
runs offline with no key and no cost. Real providers (Anthropic, OpenAI, Gemini) are lazily
imported only when selected, so none of them is a hard dependency.

Select with ``MEDFACT_LLM_PROVIDER`` in {stub, anthropic, openai, gemini} and, for real
providers, the matching API key env var. The model id is overridable with
``MEDFACT_LLM_MODEL``.
"""

from __future__ import annotations

import os
import time

from .base.llm import LLMClient

# The real generative providers. A backend env value matching one of these pins the
# provider, while "stub" and "llm" do not. Single source of truth, so adding a provider
# touches one list plus its client class and default model below.
PROVIDERS = ("anthropic", "openai", "gemini")

# Default models use a current low-cost model for the lowest cost.
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash-lite",
}


class StubLLM:
    """Offline placeholder that calls no LLM. Returns canned JSON so extraction and stance
    can parse a well-formed response with no provider and no network. It is not a real
    signal. Each caller falls back to its own deterministic heuristic when it gets this."""

    name = "stub"

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        # The callers parse JSON out of the response. Return an empty object so each caller
        # falls back to its own deterministic heuristic rather than this client.
        return "{}"


class AnthropicLLM:
    """Claude via the Anthropic SDK (lazy import, needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str) -> None:
        from anthropic import Anthropic  # lazy import

        self.name = f"anthropic:{model}"
        self.model = model
        self._client = Anthropic()

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class OpenAILLM:
    """ChatGPT via the OpenAI SDK (lazy import, needs OPENAI_API_KEY)."""

    def __init__(self, model: str) -> None:
        from openai import OpenAI  # lazy import

        self.name = f"openai:{model}"
        self.model = model
        self._client = OpenAI()

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class GeminiLLM:
    """Gemini via the google-genai SDK (lazy import, needs GEMINI_API_KEY)."""

    def __init__(self, model: str) -> None:
        from google import genai  # lazy import

        self.name = f"gemini:{model}"
        self.model = model
        self._client = genai.Client()

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        # The free tier rate-limits (429 / RESOURCE_EXHAUSTED) and the shared model can be
        # transiently overloaded (503 UNAVAILABLE, 500/502/504). Both are temporary, so back
        # off and retry rather than crash a long batch run; other errors are re-raised.
        transient = ("429", "resource_exhausted", "rate limit", "503", "unavailable",
                    "overloaded", "high demand", "500", "internal", "502", "504")
        delay = 4.0
        for attempt in range(7):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt)
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001 - inspect message to classify transient errors
                msg = str(exc).lower()
                if attempt == 6 or not any(t in msg for t in transient):
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        return ""


def get_llm(provider: str | None = None) -> LLMClient:
    """Build an LLM client. ``provider`` overrides ``MEDFACT_LLM_PROVIDER``. Both default to
    the offline stub."""
    provider = (provider or os.environ.get("MEDFACT_LLM_PROVIDER", "stub")).lower()
    if provider == "stub":
        return StubLLM()
    model = os.environ.get("MEDFACT_LLM_MODEL", _DEFAULT_MODELS.get(provider, ""))
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAILLM(model)
    if provider == "gemini":
        return GeminiLLM(model)
    raise ValueError(f"Unknown MEDFACT_LLM_PROVIDER: {provider}")
