"""The LLMClient interface for provider-agnostic text generation.

Two steps in the filter need a model, claim extraction and stance judgement. Both go
through this single interface so a provider is chosen in one place. This follows the
same Protocol shape as the other plug points in the package.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Return the model's text response to a single user prompt."""
        ...
