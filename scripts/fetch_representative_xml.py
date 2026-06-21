"""Fetch 10 representative recent PubMed papers as standalone XML for a filter trial.

Unlike scripts/fetch_trial_xml.py (a curated, retraction-heavy stress test), this samples
ordinary recent journal articles across a spread of common clinical areas, excluding
retractions. The papers are whatever PubMed returns for each area, not chosen by verdict,
so the run reflects how the filter behaves on typical input (mostly kept).

Uses NCBI E-utilities. No LLM and no API key needed. Set NCBI_EMAIL for politeness, and
MEDFACT_INSECURE_TLS=1 / MEDFACT_CA_BUNDLE behind a TLS-terminating proxy.
Run: python scripts/fetch_representative_xml.py
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
OUT_DIR = Path("data/representative")

# One ordinary recent paper per clinical area, for a representative spread.
_NOT_RETRACTED = 'NOT "Retracted Publication"[pt] NOT "Retraction of Publication"[pt]'
TOPICS = [
    "type 2 diabetes",
    "hypertension",
    "breast cancer",
    "asthma",
    "major depressive disorder",
    "ischemic stroke",
    "rheumatoid arthritis",
    "chronic kidney disease",
    "chronic obstructive pulmonary disease",
    "Alzheimer disease",
]
TARGET = 10


def _ssl_context() -> ssl.SSLContext | None:
    bundle = os.environ.get("MEDFACT_CA_BUNDLE")
    if bundle:
        return ssl.create_default_context(cafile=bundle)
    if os.environ.get("MEDFACT_INSECURE_TLS") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None


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
    for topic in TOPICS:
        if len(pmids) >= TARGET:
            break
        term = f'"{topic}" AND 2024:2025[dp] AND hasabstract AND "Journal Article"[pt] {_NOT_RETRACTED}'
        try:
            hits = esearch(term, 5)
        except Exception as exc:  # noqa: BLE001 - trial script, report and continue
            print(f"WARN esearch failed for [{topic}]: {exc}")
            continue
        for pmid in hits:
            if pmid not in seen:
                seen.add(pmid)
                pmids.append(pmid)
                print(f"topic [{topic}] -> {pmid}")
                break
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
        wrapper = ET.Element("PubmedArticleSet")
        wrapper.append(art)
        (OUT_DIR / f"{pmid}.xml").write_bytes(ET.tostring(wrapper, encoding="utf-8", xml_declaration=True))
        written += 1
        print(f"  {pmid}: {title[:70]!r} | types={pub_types}")

    print(f"Wrote {written} XML files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
