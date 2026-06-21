"""Fetch a curated mix of 10 PubMed papers as standalone XML for a filter trial run.

Uses NCBI E-utilities (esearch then efetch). The mix is biased toward retracted papers
(they carry RetractionIn links the filter's retriever uses, so they should score refuted)
plus a couple of well-established controls (guideline / systematic review). Each paper is
written as its own valid PubmedArticleSet file under data/trial/<pmid>.xml.

No LLM and no API key needed. Set NCBI_EMAIL for polite identification (optional).
Run: python scripts/fetch_trial_xml.py
"""

from __future__ import annotations

import os
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUT_DIR = Path("data/trial")


def _ssl_context() -> ssl.SSLContext | None:
    """Mirror the project's TLS escape hatches for proxied sandboxes.

    MEDFACT_CA_BUNDLE points at a trusted bundle; MEDFACT_INSECURE_TLS=1 disables
    verification entirely (last resort for a self-signed proxy cert).
    """
    bundle = os.environ.get("MEDFACT_CA_BUNDLE")
    if bundle:
        return ssl.create_default_context(cafile=bundle)
    if os.environ.get("MEDFACT_INSECURE_TLS") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None

# (query, how many top hits to take). Retracted papers first, then controls.
QUERIES: list[tuple[str, int]] = [
    ('hydroxychloroquine AND covid-19 AND "Retracted Publication"[pt]', 2),
    ('ivermectin AND covid-19 AND "Retracted Publication"[pt]', 1),
    ('vaccines AND autism AND "Retracted Publication"[pt]', 1),
    ('"Retracted Publication"[pt] AND cancer AND 2017:2022[dp]', 2),
    ('hypertension AND "Practice Guideline"[pt] AND 2019:2023[dp]', 1),
    ('statins AND cardiovascular AND "Systematic Review"[pt] AND 2020:2023[dp]', 2),
    ('metformin AND type 2 diabetes AND "Randomized Controlled Trial"[pt] AND 2018:2022[dp]', 1),
]
TARGET = 10


def _params(extra: dict[str, str]) -> dict[str, str]:
    params = {**extra, "tool": "medfact_poc_trial"}
    email = os.environ.get("NCBI_EMAIL")
    if email:
        params["email"] = email
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def _get(url: str, extra: dict[str, str]) -> str:
    full = f"{url}?{urllib.parse.urlencode(_params(extra))}"
    with urllib.request.urlopen(full, timeout=30, context=_ssl_context()) as resp:
        return resp.read().decode("utf-8")


def esearch(term: str, retmax: int) -> list[str]:
    raw = _get(
        f"{EUTILS}/esearch.fcgi",
        {"db": "pubmed", "term": term, "retmax": str(retmax), "retmode": "xml", "sort": "relevance"},
    )
    root = ET.fromstring(raw)
    return [el.text for el in root.findall(".//IdList/Id") if el.text]


def efetch(pmids: list[str]) -> str:
    return _get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})


def main() -> None:
    pmids: list[str] = []
    seen: set[str] = set()
    for term, n in QUERIES:
        if len(pmids) >= TARGET:
            break
        try:
            hits = esearch(term, n + 3)
        except Exception as exc:  # noqa: BLE001 - trial script, report and continue
            print(f"WARN esearch failed for [{term}]: {exc}")
            continue
        kept = 0
        for pmid in hits:
            if pmid not in seen and len(pmids) < TARGET and kept < n:
                seen.add(pmid)
                pmids.append(pmid)
                kept += 1
        print(f"query [{term[:50]}...] -> kept {kept}")
        time.sleep(0.5)

    pmids = pmids[:TARGET]
    print(f"Fetching {len(pmids)} papers: {pmids}")
    xml_text = efetch(pmids)
    root = ET.fromstring(xml_text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//MedlineCitation/PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text
        title_el = art.find(".//Article/ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""
        pub_types = [el.text for el in art.findall(".//PublicationType") if el.text]
        ref_types = [cc.get("RefType", "") for cc in art.findall(".//CommentsCorrectionsList/CommentsCorrections")]
        wrapper = ET.Element("PubmedArticleSet")
        wrapper.append(art)
        out_file = OUT_DIR / f"{pmid}.xml"
        out_file.write_bytes(ET.tostring(wrapper, encoding="utf-8", xml_declaration=True))
        written += 1
        print(f"  {pmid}: {title[:70]!r} | types={pub_types} | refs={ref_types}")

    print(f"Wrote {written} XML files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
