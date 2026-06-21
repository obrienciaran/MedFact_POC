"""Evidence retrieval for the filter: find the works that refute or debate a paper.

Behind the ``Retriever`` Protocol (see ``base.retriever``):
  * ``stub`` is offline. It uses the paper's own ``CommentsCorrectionsList`` links as the
    evidence pool. A ``RetractionIn`` becomes a refuting retraction notice, and a
    ``CommentIn`` or ``ErratumIn`` becomes a debating comment. There is no network, but
    the signal is real structure from the XML, so a retracted paper is correctly flagged.
  * ``live`` queries PubMed and Europe PMC for the claim and fetches the linked dispute
    PMIDs, reusing the existing Source machinery. It needs network access.

The pool is per paper and claim. The stub returns the same paper-level links for each
claim because it has no claim-level signal offline.
"""

from __future__ import annotations

import os

from ..base.retriever import Retriever
from ..schema import Candidate, ExtractedClaim, PaperRecord
from ..transformation.ingest import DISPUTE_REFTYPES

_REFUTING_REFTYPES = {"RetractionIn"}


class StubRetriever:
    """Offline retriever built from the paper's own comment and correction links."""

    name = "stub"

    def retrieve(self, paper: PaperRecord, claim: ExtractedClaim, *, limit: int = 20) -> list[Candidate]:
        out: list[Candidate] = []
        for ref_type in DISPUTE_REFTYPES:
            for pmid in paper.comments_corrections.get(ref_type, []):
                out.append(_synthetic_candidate(pmid, ref_type, paper))
        return out[:limit]


def _synthetic_candidate(pmid: str, ref_type: str, paper: PaperRecord) -> Candidate:
    if ref_type in _REFUTING_REFTYPES:
        return Candidate(
            source="pubmed", ext_id=pmid,
            title=f"Retraction: {paper.title}".strip(),
            abstract="This article has been retracted and withdrawn. It showed no benefit and increased risk.",
            pub_types=["Retraction of Publication"],
            year=paper.year, retrieved_by=["links"], is_retraction_of=[paper.pmid],
        )
    return Candidate(
        source="pubmed", ext_id=pmid,
        title=f"Comment on: {paper.title}".strip(),
        abstract="A published comment debating the findings of the cited article.",
        pub_types=["Comment"], year=paper.year, retrieved_by=["links"],
    )


class LiveRetriever:
    """Network retriever reusing the Source providers plus the paper's dispute links."""

    name = "live"

    def retrieve(self, paper: PaperRecord, claim: ExtractedClaim, *, limit: int = 20) -> list[Candidate]:
        from .http import make_client
        from .pubmed import efetch
        from .sources import get_sources

        # A retraction notice is the strongest offline refutation, so it must never be cut by
        # the limit. Fetch retraction links first and keep them ahead of the query hits; other
        # dispute links and source results then fill the remaining slots up to limit.
        retraction_pmids = [
            pmid for rt in _REFUTING_REFTYPES for pmid in paper.comments_corrections.get(rt, [])
        ]
        other_dispute_pmids = [
            pmid for rt in DISPUTE_REFTYPES if rt not in _REFUTING_REFTYPES
            for pmid in paper.comments_corrections.get(rt, [])
        ]
        priority: dict[str, Candidate] = {}
        others: dict[str, Candidate] = {}
        with make_client() as client:
            for c in efetch(retraction_pmids, client=client):
                c.retrieved_by = sorted(set(c.retrieved_by) | {"links"})
                priority[c.ext_id] = c
            for source in get_sources():
                try:
                    found = source.search_claim(claim.normalized, limit=limit, client=client)
                except Exception as exc:  # noqa: BLE001 - one source's outage must not unverify the paper
                    print(f"WARN: {source.name} failed for {paper.pmid}, using other sources. "
                        f"{type(exc).__name__}: {exc}")
                    continue
                for c in found:
                    if c.ext_id not in priority:
                        others.setdefault(c.ext_id, c)
            for c in efetch(other_dispute_pmids, client=client):
                if c.ext_id not in priority:
                    c.retrieved_by = sorted(set(c.retrieved_by) | {"links"})
                    others[c.ext_id] = c
        return (list(priority.values()) + list(others.values()))[:limit]


def get_retriever() -> Retriever:
    """Build the configured retriever. Defaults to the offline stub."""
    backend = os.environ.get("MEDFACT_RETRIEVER", "stub").lower()
    return LiveRetriever() if backend == "live" else StubRetriever()
