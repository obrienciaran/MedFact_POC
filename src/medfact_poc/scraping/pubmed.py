"""PubMed source via NCBI E-utilities.

Three operations:
  * ``esearch`` turns a query string into a list of PMIDs.
  * ``efetch`` turns PMIDs into full records (abstract, publication types, year, DOI).
  * retraction and update links are parsed straight from the efetch XML
    (``CommentsCorrectionsList``), which is simpler and more complete than ELink.

Network calls go through ``medfact_poc.scraping.http`` for rate limiting and TLS
handling.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import httpx

from ..schema import Candidate, NormalizedClaim
from ..transformation import medline, query
from .http import make_client, ncbi_params, ncbi_throttle

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_get(client: httpx.Client, url: str, params: dict[str, str]) -> httpx.Response:
    """GET an E-utilities endpoint, retrying on 429/5xx with backoff.

    Without an API key NCBI caps callers at ~3 req/s and answers a burst with 429. The
    shared limiter stays just under that, but leaves no headroom, so back off and retry
    rather than let a transient throttle abort a long run.
    """
    delay = 1.0
    for attempt in range(5):
        ncbi_throttle()
        r = client.get(url, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            if attempt < 4:
                time.sleep(delay)
                delay = min(delay * 2, 15.0)
                continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


class PubMedSource:
    """Evidence provider backed by NCBI E-utilities (esearch then efetch)."""

    name = "pubmed"

    def search_claim(
        self, claim: NormalizedClaim, *, limit: int, client: httpx.Client
    ) -> list[Candidate]:
        pmids: list[str] = []
        seen: set[str] = set()
        for q in query.pubmed_queries(claim):
            for pmid in esearch(q, retmax=limit, client=client):
                if pmid not in seen:
                    seen.add(pmid)
                    pmids.append(pmid)
        return efetch(pmids, client=client)


def esearch(query: str, *, retmax: int = 50, client: httpx.Client | None = None) -> list[str]:
    """Return PMIDs matching ``query`` (most relevant first)."""
    own = client is None
    client = client or make_client()
    try:
        params = ncbi_params(
            {"db": "pubmed", "term": query, "retmax": str(retmax), "retmode": "json", "sort": "relevance"}
        )
        r = _ncbi_get(client, f"{_EUTILS}/esearch.fcgi", params)
        return r.json().get("esearchresult", {}).get("idlist", [])
    finally:
        if own:
            client.close()


def efetch(pmids: list[str], *, client: httpx.Client | None = None) -> list[Candidate]:
    """Fetch full records for ``pmids`` and parse them into Candidates."""
    if not pmids:
        return []
    own = client is None
    client = client or make_client()
    try:
        params = ncbi_params({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        r = _ncbi_get(client, f"{_EUTILS}/efetch.fcgi", params)
        return parse_efetch_xml(r.text)
    finally:
        if own:
            client.close()


def parse_efetch_xml(xml_text: str) -> list[Candidate]:
    """Parse an efetch PubmedArticleSet into Candidates. Pure and unit-testable."""
    root = ET.fromstring(xml_text)
    out: list[Candidate] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = medline.text(art, ".//MedlineCitation/PMID")
        if not pmid:
            continue
        retracted_by, is_retraction_of = _parse_corrections(art)
        out.append(
            Candidate(
                source="pubmed",
                ext_id=pmid,
                doi=medline.doi(art),
                title=medline.text(art, ".//Article/ArticleTitle") or "",
                abstract=medline.abstract(art),
                pub_types=medline.pub_types(art),
                year=medline.year(art),
                retracted_by=retracted_by,
                is_retraction_of=is_retraction_of,
                retrieved_by=[],
            )
        )
    return out


def _parse_corrections(art: ET.Element) -> tuple[list[str], list[str]]:
    """Extract retraction relationships from CommentsCorrectionsList.

    RefType semantics, read from the PMID's perspective:
      * RetractionIn means this article is retracted by the referenced PMID.
      * RetractionOf means this article is a retraction of the referenced PMID.
    """
    retracted_by: list[str] = []
    is_retraction_of: list[str] = []
    for cc in art.findall(".//CommentsCorrectionsList/CommentsCorrections"):
        ref_type = cc.get("RefType", "")
        ref_pmid = medline.text(cc, "PMID")
        if not ref_pmid:
            continue
        if ref_type == "RetractionIn":
            retracted_by.append(ref_pmid)
        elif ref_type == "RetractionOf":
            is_retraction_of.append(ref_pmid)
    return retracted_by, is_retraction_of
