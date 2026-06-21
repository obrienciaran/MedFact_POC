"""The StanceBackend interface for stance classification.

A StanceBackend judges whether a candidate supports, refutes, or stays neutral on a
claim as asserted. The harness measures the stance step, so it sits behind this
interface and can be swapped for a stub or a real LLM backend. This follows the same
Protocol shape as the other plug points in the package.
"""

from __future__ import annotations

from typing import Protocol

from ..schema import Candidate, GoldEntry, StanceLabel


class StanceBackend(Protocol):
    name: str

    def classify(self, gold: GoldEntry, candidate: Candidate) -> StanceLabel: ...
