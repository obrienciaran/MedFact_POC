"""The Embedder interface for semantic ranking backends.

An Embedder turns text into vectors. The harness ranks a candidate pool by cosine
similarity to the claim, so any backend that returns vectors can be swapped in. This
follows the same Protocol shape as the other plug points in the package.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
