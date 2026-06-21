"""The Retriever interface for the filter's evidence retrieval.

A Retriever finds the works that refute or debate one claim of a paper. The offline
stub uses the paper's own comment and retraction links. The live backend searches the
evidence sources. This follows the same Protocol shape as the other plug points in the
package.
"""

from __future__ import annotations

from typing import Protocol

from ..schema import Candidate, ExtractedClaim, PaperRecord


class Retriever(Protocol):
    name: str

    def retrieve(self, paper: PaperRecord, claim: ExtractedClaim, *, limit: int) -> list[Candidate]:
        """Return candidate evidence works for one claim of a paper."""
        ...
