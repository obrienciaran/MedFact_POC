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

import json
import time
import xml.etree.ElementTree as ET

import httpx

from ..schema import Candidate, NormalizedClaim
from ..transformation import medline, query
from .http import make_client, ncbi_params, ncbi_throttle
from .querycache import get_query_cache

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
    """Return PMIDs matching ``query`` (most relevant first).

    Consults the query cache (when enabled) so a recurring query hits the network once.
    """
    cache = get_query_cache()
    if cache is not None and (hit := cache.get("pubmed", query, retmax)) is not None:
        return json.loads(hit)
    own = client is None
    client = client or make_client()
    try:
        params = ncbi_params(
            {"db": "pubmed", "term": query, "retmax": str(retmax), "retmode": "json", "sort": "relevance"}
        )
        r = _ncbi_get(client, f"{_EUTILS}/esearch.fcgi", params)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if cache is not None:
            cache.put("pubmed", query, retmax, json.dumps(ids))
        return ids
    finally:
        if own:
            client.close()


# Cap PMIDs per efetch so the request URL cannot overflow (NCBI returns 414 for long URLs).
EFETCH_BATCH = 150


def efetch(pmids: list[str], *, client: httpx.Client | None = None) -> list[Candidate]:
    """Fetch full records for ``pmids`` and parse them into Candidates.

    Records already in the query cache (when enabled) are served from DuckDB, so a study fetched
    once is never fetched again; only the missing ids hit the network. Results preserve the input
    order.
    """
    if not pmids:
        return []
    cache = get_query_cache()
    cached: dict[str, Candidate] = {}
    to_fetch = list(pmids)
    if cache is not None:
        cached = {
            ext_id: Candidate.model_validate_json(payload)
            for ext_id, payload in cache.get_records(pmids).items()
        }
        to_fetch = [p for p in pmids if p not in cached]
    fetched = _efetch_network(to_fetch, client=client)
    if cache is not None and fetched:
        cache.put_records({c.ext_id: c.model_dump_json() for c in fetched})
    by_id = {**cached, **{c.ext_id: c for c in fetched}}
    return [by_id[p] for p in pmids if p in by_id]


def _efetch_network(pmids: list[str], *, client: httpx.Client | None = None) -> list[Candidate]:
    """Fetch ``pmids`` from NCBI and parse them into Candidates (no cache).

    Large pools are fetched in batches because all ids share one request URL, and a few
    hundred ids overflow it (HTTP 414). Batching keeps each request well under the limit.
    """
    if not pmids:
        return []
    own = client is None
    client = client or make_client()
    try:
        out: list[Candidate] = []
        for start in range(0, len(pmids), EFETCH_BATCH):
            batch = pmids[start:start + EFETCH_BATCH]
            params = ncbi_params({"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
            r = _ncbi_get(client, f"{_EUTILS}/efetch.fcgi", params)
            out.extend(parse_efetch_xml(r.text))
        return out
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
