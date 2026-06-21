"""The Source interface shared by every evidence provider.

A Source owns one provider end to end. It builds its own queries from a normalized
claim, calls its API, and returns parsed Candidates. Retrieval depends only on this
interface, so a new provider can be added without touching retrieval logic.

This mirrors the other plug points in the package (Embedder, StanceBackend,
ClaimExtractor, Retriever, LLMClient), so every swappable part follows the same
Protocol shape.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from ..schema import Candidate, NormalizedClaim


class Source(Protocol):
    name: str

    def search_claim(
        self, claim: NormalizedClaim, *, limit: int, client: httpx.Client
    ) -> list[Candidate]:
        """Return candidate evidence for a claim from this provider."""
        ...
