"""Claim extraction: turn a paper into the checkable claims the filter scores.

Scoring at the claim level is more accurate than scoring a whole abstract at once, because
a paper can be right about one thing and wrong about another, and because a claim carries
conditions (population, comparator, dose, setting) that decide whether a piece of evidence
actually contradicts it.

Backends are pluggable behind ``ClaimExtractor`` (see ``base.extractor``):
  * ``stub`` is deterministic and dependency-free. It treats the title as a single claim so
    the pipeline runs offline. It is not a real extraction.
  * ``llm`` is any ``LLMClient`` provider, prompted to return atomic claims as JSON.
"""

from __future__ import annotations

import json
import os

from ..base.extractor import ClaimExtractor
from ..base.llm import LLMClient
from ..llm import PROVIDERS, get_llm
from ..schema import ExtractedClaim, NormalizedClaim, PaperRecord


class StubExtractor:
    """Deterministic placeholder. The paper's title is treated as its single claim."""

    name = "stub"

    def extract(self, paper: PaperRecord) -> list[ExtractedClaim]:
        title = paper.title.strip()
        if not title:
            return []
        return [
            ExtractedClaim(
                paper_pmid=paper.pmid, index=0, claim_text=title,
                normalized=NormalizedClaim(intervention=title, outcome=""),
            )
        ]


_PROMPT = """Extract the main checkable scientific claims this paper asserts. For each, keep \
the conditions attached. Do not invent claims the paper does not make.

TITLE: {title}
ABSTRACT: {abstract}

Respond with ONLY a JSON array, each element:
{{"claim_text": "the claim as asserted", "intervention": "", "outcome": "", \
"population": "", "comparator": "", "direction": "increases|decreases|no_effect|causes|prevents"}}
Use "" for unknown fields and omit "direction" if unclear. Return at most 5 claims."""

_DIRECTIONS = {"increases", "decreases", "no_effect", "causes", "prevents"}


class LLMExtractor:
    """Claim extractor backed by any ``LLMClient`` provider."""

    def __init__(self, client: LLMClient) -> None:
        self.name = f"extract:{client.name}"
        self._client = client

    def extract(self, paper: PaperRecord) -> list[ExtractedClaim]:
        prompt = _PROMPT.format(
            title=paper.title or "(no title)",
            abstract=(paper.abstract or "")[:6000] or "(no abstract)",
        )
        text = self._client.complete(prompt, max_tokens=800)
        return self._parse(paper, text)

    @staticmethod
    def _parse(paper: PaperRecord, text: str) -> list[ExtractedClaim]:
        try:
            start, end = text.index("["), text.rindex("]") + 1
            items = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            items = []
        claims: list[ExtractedClaim] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            claim_text = str(item.get("claim_text", "")).strip()
            intervention = str(item.get("intervention", "")).strip()
            if not (claim_text or intervention):
                continue
            direction = item.get("direction")
            claims.append(
                ExtractedClaim(
                    paper_pmid=paper.pmid, index=i, claim_text=claim_text or intervention,
                    normalized=NormalizedClaim(
                        intervention=intervention or claim_text,
                        outcome=str(item.get("outcome", "")).strip(),
                        population=str(item.get("population", "")).strip() or None,
                        comparator=str(item.get("comparator", "")).strip() or None,
                        direction=direction if direction in _DIRECTIONS else None,
                    ),
                )
            )
        return claims


def get_extractor() -> ClaimExtractor:
    """Build the configured claim extractor. Defaults to the offline stub."""
    backend = os.environ.get("MEDFACT_EXTRACT_BACKEND", "stub").lower()
    if backend == "stub":
        return StubExtractor()
    provider = backend if backend in PROVIDERS else None
    return LLMExtractor(get_llm(provider))
