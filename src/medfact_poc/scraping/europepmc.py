"""Europe PMC source via its REST search API.

Used as a second, independent retrieval channel so that "not_indexed" failures reflect
genuine absence rather than one provider's quirks. Europe PMC returns structured JSON
including pubType and abstractText in a single call.

Europe PMC is an evidence retrieval source, not filter input. The filter ingests PubMed
XML only. This source is queried to find studies that contradict or debate a claim.
"""

from __future__ import annotations

import time

import httpx

from ..schema import Candidate, NormalizedClaim
from ..transformation import query
from .http import generic_throttle, make_client

_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMCSource:
    """Evidence provider backed by the Europe PMC REST search API (single step)."""

    name = "europepmc"

    def search_claim(
        self, claim: NormalizedClaim, *, limit: int, client: httpx.Client
    ) -> list[Candidate]:
        by_id: dict[str, Candidate] = {}
        for q in query.europepmc_queries(claim):
            for c in search(q, page_size=limit, client=client):
                by_id.setdefault(c.ext_id, c)
        return list(by_id.values())


def search(query: str, *, page_size: int = 50, client: httpx.Client | None = None) -> list[Candidate]:
    """Search Europe PMC and return Candidates (abstracts included)."""
    own = client is None
    client = client or make_client()
    try:
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(page_size),
            "resultType": "core",  # includes abstractText and pubTypeList
        }
        # Europe PMC occasionally answers a 502/503 under load. Back off and retry rather
        # than abort the run on a transient gateway error.
        delay = 1.0
        for attempt in range(5):
            generic_throttle()
            r = client.get(_SEARCH, params=params)
            if (r.status_code == 429 or r.status_code >= 500) and attempt < 4:
                time.sleep(delay)
                delay = min(delay * 2, 15.0)
                continue
            break
        r.raise_for_status()
        return parse_search_json(r.json())
    finally:
        if own:
            client.close()


def parse_search_json(payload: dict) -> list[Candidate]:
    """Parse a Europe PMC search response into Candidates. Pure and unit-testable."""
    out: list[Candidate] = []
    for rec in payload.get("resultList", {}).get("result", []):
        # Prefer PMID as ext_id when present so it aligns with the gold answer keys.
        pmid = rec.get("pmid")
        ext_id = pmid or f"{rec.get('source', 'EPMC')}:{rec.get('id', '')}"
        pub_types = rec.get("pubTypeList", {}).get("pubType", [])
        if isinstance(pub_types, str):
            pub_types = [pub_types]
        year = None
        if (yr := rec.get("pubYear")) and str(yr).isdigit():
            year = int(yr)
        out.append(
            Candidate(
                source="europepmc",
                ext_id=ext_id,
                doi=rec.get("doi"),
                title=rec.get("title", "") or "",
                abstract=rec.get("abstractText", "") or "",
                pub_types=[p for p in pub_types if p],
                year=year,
                retrieved_by=[],
            )
        )
    return out
