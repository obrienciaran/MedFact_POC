"""The ClaimExtractor interface for lifting claims from a paper.

A ClaimExtractor turns a parsed paper into the checkable claims the filter scores.
Backends range from an offline stub to a real LLM. This follows the same Protocol shape
as the other plug points in the package.
"""

from __future__ import annotations

from typing import Protocol

from ..schema import ExtractedClaim, PaperRecord


class ClaimExtractor(Protocol):
    name: str

    def extract(self, paper: PaperRecord) -> list[ExtractedClaim]:
        """Return the checkable claims asserted by ``paper``."""
        ...
